import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from colorama import Fore
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wff3d/src/watermark"))
from gadm_aap import (  # noqa: E402
    GADMAAPConfig,
    compute_gaussian_reliability,
    gadm_loss,
    prune_gaussians,
    sample_prune_ratio,
)
from image_attacks import (  # noqa: E402
    ATTACK_NAMES,
    WatermarkAttackConfig,
    apply_image_attack,
    sample_training_attack,
)
from gaussian_attacks import (  # noqa: E402
    GAUSSIAN_ATTACK_NAMES,
    GaussianAttackConfig,
    apply_gaussian_attack,
)
from hidden_baseline import (  # noqa: E402
    hidden_decode_batch,
    hidden_watermark_batch,
    load_hidden_baseline,
)

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule, get_data_shim
from src.global_cfg import set_cfg
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.loss import get_losses
from src.misc.image_io import save_image
from src.model.decoder import get_decoder
from src.model.model_wrapper import ModelWrapper
from src.model.encoder import get_encoder


class LocalStepTracker:
    def __init__(self) -> None:
        self.step = 0

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def get_step(self) -> int:
        return self.step


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def _move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _latest_manual_ckpt(ckpt_dir: Path) -> Path | None:
    ckpts = glob.glob(str(ckpt_dir / "step_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))
    return Path(ckpts[-1])


def _trainable_prefixes() -> tuple[str, ...]:
    return (
        "encoder.feature_upsampler.",
        "encoder.gaussian_regressor.",
        "encoder.gaussian_head.",
    )


def _save_manual_ckpt(
    ckpt_dir: Path,
    model_wrapper: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefixes = _trainable_prefixes()
    payload = {
        "state_dict": {
            k: v.detach().cpu()
            for k, v in model_wrapper.state_dict().items()
            if any(k.startswith(prefix) for prefix in prefixes)
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
    }
    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def _load_full_checkpoint(model_wrapper: ModelWrapper, ckpt_path: str) -> None:
    print(cyan(f"[DepthSplat] loading checkpoint: {ckpt_path}"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model_wrapper.load_state_dict(state, strict=True)
    print(cyan(f"[DepthSplat] strict checkpoint load ok. missing={len(missing)} unexpected={len(unexpected)}"))


def _load_tuned_checkpoint(model_wrapper: ModelWrapper, ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tuned_state = ckpt.get("state_dict", ckpt)
    current_state = model_wrapper.state_dict()
    expected = {
        key for key in current_state
        if any(key.startswith(prefix) for prefix in _trainable_prefixes())
    }
    provided = set(tuned_state)
    if provided != expected:
        raise RuntimeError(
            f"Tuned checkpoint key mismatch: missing={sorted(expected - provided)[:8]} "
            f"unexpected={sorted(provided - expected)[:8]}"
        )
    for key, value in tuned_state.items():
        if current_state[key].shape != value.shape:
            raise RuntimeError(
                f"Tuned checkpoint shape mismatch for {key}: "
                f"expected {tuple(current_state[key].shape)}, got {tuple(value.shape)}"
            )
        current_state[key] = value
    model_wrapper.load_state_dict(current_state, strict=True)
    step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    print(cyan(f"[DepthSplat] tuned checkpoint loaded strictly: {ckpt_path} step={step}"))
    return step


def _set_trainable_modules(model_wrapper: ModelWrapper) -> list[torch.nn.Parameter]:
    for p in model_wrapper.parameters():
        p.requires_grad = False

    model_wrapper.eval()
    model_wrapper.encoder.eval()
    model_wrapper.decoder.eval()

    trainable_modules = [
        model_wrapper.encoder.feature_upsampler,
        model_wrapper.encoder.gaussian_regressor,
        model_wrapper.encoder.gaussian_head,
    ]
    for module in trainable_modules:
        module.train()
        for p in module.parameters():
            p.requires_grad = True

    params = [p for p in model_wrapper.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    print(cyan(f"[DepthSplat] trainable params: {total / 1e6:.2f}M"))
    return params


def _watermark_loss(
    rendered: torch.Tensor,
    watermark_extractor: torch.nn.Module,
    watermark_key: torch.Tensor,
    aggregate_views: bool = True,
    attack: str = "clean",
    attack_config: WatermarkAttackConfig | None = None,
    training_attack: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, v, c, h, w = rendered.shape
    rendered_flat = rendered.view(b * v, c, h, w)
    if attack != "clean":
        if attack_config is None:
            raise ValueError("An attack config is required for attacked watermark loss")
        rendered_flat = apply_image_attack(
            rendered_flat, attack, attack_config, training=training_attack
        )
    imnet_mean = torch.tensor([0.485, 0.456, 0.406], device=rendered.device).view(1, 1, 3, 1, 1)
    imnet_std = torch.tensor([0.229, 0.224, 0.225], device=rendered.device).view(1, 1, 3, 1, 1)
    rendered_norm = (rendered_flat.view(b, v, c, h, w) - imnet_mean) / imnet_std
    rendered_norm = F.interpolate(
        rendered_norm.view(b * v, c, h, w),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    decoded = watermark_extractor(rendered_norm).view(b, v, -1)
    decoded = decoded.mean(dim=1) if aggregate_views else decoded.flatten(0, 1)
    key_batch = watermark_key.unsqueeze(0).expand(decoded.shape[0], -1)
    loss_wm = F.binary_cross_entropy_with_logits(decoded * 10.0, key_batch)
    bit_acc = ((decoded > 0).float() == key_batch).float().mean()
    return loss_wm, bit_acc


@torch.no_grad()
def evaluate_smoke(
    model_wrapper,
    data_loader,
    watermark_extractor,
    watermark_key,
    cfg,
    device,
    global_step: int,
    num_batches: int,
    attack_config: WatermarkAttackConfig,
    evaluate_attacks: bool,
    evaluate_gaussian_attacks: bool,
    gaussian_attack_config: GaussianAttackConfig,
    hidden_baseline=None,
    evaluate_hidden_attacks: bool = False,
    eval_image_dir: Path | None = None,
) -> None:
    model_wrapper.eval()
    data_shim = get_data_shim(model_wrapper.encoder)
    totals = {"bit_acc": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    attack_totals = {name: 0.0 for name in ATTACK_NAMES}
    gaussian_attack_totals = {
        name: {"bit_acc": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        for name in GAUSSIAN_ATTACK_NAMES
    }
    hidden_totals = {
        "input_bit_acc": 0.0,
        "bit_acc": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }
    hidden_attack_totals = {name: 0.0 for name in ATTACK_NAMES}
    count = 0
    inference_seconds = 0.0
    extraction_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in data_loader:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()
        batch = _move_to_device(batch, device)
        if hidden_baseline is not None:
            watermarked_context, input_view_acc = hidden_watermark_batch(
                hidden_baseline,
                batch["context"]["image"].float().clamp(0, 1),
                watermark_key,
            )
            batch["context"]["image"] = watermarked_context
            hidden_totals["input_bit_acc"] += float(input_view_acc.mean())
        batch = data_shim(batch)
        gaussians = model_wrapper.encoder(batch["context"], global_step, visualization_dump={})
        target = batch["target"]["image"].float().clamp(0, 1)
        _, _, _, h, w = target.shape
        output = model_wrapper.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=cfg.train.depth_mode,
        )
        prediction = output.color.float().clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - batch_start
        extraction_start = time.perf_counter()
        _, bit_acc = _watermark_loss(
            prediction, watermark_extractor, watermark_key, aggregate_views=False
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        extraction_seconds += time.perf_counter() - extraction_start
        if evaluate_attacks:
            for attack_name in ATTACK_NAMES:
                _, attacked_acc = _watermark_loss(
                    prediction,
                    watermark_extractor,
                    watermark_key,
                    aggregate_views=False,
                    attack=attack_name,
                    attack_config=attack_config,
                )
                attack_totals[attack_name] += float(attacked_acc)
        target_flat = target.flatten(0, 1)
        prediction_flat = prediction.flatten(0, 1)
        if hidden_baseline is not None:
            hidden_view_acc = hidden_decode_batch(hidden_baseline, prediction, watermark_key)
            hidden_totals["bit_acc"] += float(hidden_view_acc.mean())
            hidden_totals["psnr"] += float(compute_psnr(target_flat, prediction_flat).mean())
            hidden_totals["ssim"] += float(compute_ssim(target_flat, prediction_flat).mean())
            hidden_totals["lpips"] += float(compute_lpips(target_flat, prediction_flat).mean())
            if evaluate_hidden_attacks:
                for attack_name in ATTACK_NAMES:
                    attacked = apply_image_attack(prediction_flat, attack_name, attack_config)
                    hidden_attack_totals[attack_name] += float(
                        hidden_decode_batch(hidden_baseline, attacked, watermark_key).mean()
                    )
        if evaluate_gaussian_attacks:
            for attack_name in GAUSSIAN_ATTACK_NAMES:
                attacked_gaussians = apply_gaussian_attack(
                    gaussians, attack_name, gaussian_attack_config
                )
                attacked_output = model_wrapper.decoder.forward(
                    attacked_gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=cfg.train.depth_mode,
                )
                attacked_prediction = attacked_output.color.float().clamp(0, 1)
                _, attacked_bit_acc = _watermark_loss(
                    attacked_prediction,
                    watermark_extractor,
                    watermark_key,
                    aggregate_views=False,
                )
                attacked_flat = attacked_prediction.flatten(0, 1)
                values = gaussian_attack_totals[attack_name]
                values["bit_acc"] += float(attacked_bit_acc)
                values["psnr"] += float(compute_psnr(target_flat, attacked_flat).mean())
                values["ssim"] += float(compute_ssim(target_flat, attacked_flat).mean())
                values["lpips"] += float(compute_lpips(target_flat, attacked_flat).mean())
        totals["bit_acc"] += float(bit_acc)
        totals["psnr"] += float(compute_psnr(target_flat, prediction_flat).mean())
        totals["ssim"] += float(compute_ssim(target_flat, prediction_flat).mean())
        totals["lpips"] += float(compute_lpips(target_flat, prediction_flat).mean())
        count += 1
        if eval_image_dir is not None:
            scene_names = batch.get("scene", [f"batch_{count:06d}"])
            for batch_index, scene_name in enumerate(scene_names):
                scene_root = eval_image_dir / str(scene_name).replace("/", "_")
                for view_index in range(prediction.shape[1]):
                    save_image(
                        prediction[batch_index, view_index],
                        scene_root / "watermarked" / f"{view_index:06d}.png",
                    )
                    save_image(
                        target[batch_index, view_index],
                        scene_root / "gt" / f"{view_index:06d}.png",
                    )
                for view_index in range(batch["context"]["image"].shape[1]):
                    save_image(
                        batch["context"]["image"][batch_index].float().clamp(0, 1)[view_index],
                        scene_root / "context" / f"{view_index:06d}.png",
                    )
                sample_prediction = prediction[batch_index : batch_index + 1]
                sample_target = target[batch_index : batch_index + 1]
                _, sample_bit_acc = _watermark_loss(
                    sample_prediction,
                    watermark_extractor,
                    watermark_key,
                    aggregate_views=False,
                )
                sample_prediction = sample_prediction.flatten(0, 1)
                sample_target = sample_target.flatten(0, 1)
                scene_metrics = {
                    "scene": str(scene_name),
                    "bit_acc": float(sample_bit_acc),
                    "psnr": float(compute_psnr(sample_target, sample_prediction).mean()),
                    "ssim": float(compute_ssim(sample_target, sample_prediction).mean()),
                    "lpips": float(compute_lpips(sample_target, sample_prediction).mean()),
                    "views": int(prediction.shape[1]),
                }
                (scene_root / "metrics.json").write_text(
                    json.dumps(scene_metrics, indent=2) + "\n"
                )
        print(f"[WFF3D-EVAL][DepthSplat] batch={count} bit_acc={float(bit_acc):.4f}")
        if count >= num_batches:
            break
    if count == 0:
        raise RuntimeError("DepthSplat smoke evaluation received no batches")
    result = {key: value / count for key, value in totals.items()}
    if evaluate_attacks:
        result.update({f"bit_acc_{key}": value / count for key, value in attack_totals.items()})
    if evaluate_gaussian_attacks:
        for attack_name, values in gaussian_attack_totals.items():
            result.update(
                {f"{metric}_{attack_name}": value / count for metric, value in values.items()}
            )
    result["bit_acc_clean"] = result["bit_acc"]
    if hidden_baseline is not None:
        result.update({f"hidden_{key}": value / count for key, value in hidden_totals.items()})
        if evaluate_hidden_attacks:
            result.update(
                {
                    f"hidden_bit_acc_{key}": value / count
                    for key, value in hidden_attack_totals.items()
                }
            )
    result["seconds_per_scene"] = inference_seconds / count
    result["extraction_ms_per_scene"] = extraction_seconds * 1000 / count
    result["peak_gpu_memory_gb"] = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    result.update({"model": "DepthSplat", "batches": count, "global_step": global_step})
    print("[WFF3D-EVAL-RESULT] " + json.dumps(result, sort_keys=True))


def _base_losses(losses, output, batch, gaussians, global_step: int, cfg) -> tuple[torch.Tensor, dict[str, float]]:
    base_loss = torch.zeros((), device=batch["target"]["image"].device)
    loss_parts: dict[str, float] = {}
    for loss_fn in losses:
        if loss_fn.name == "mse":
            loss = loss_fn.forward(
                output,
                batch,
                gaussians,
                global_step,
                l1_loss=cfg.train.l1_loss,
                clamp_large_error=cfg.train.train_ignore_large_loss,
                valid_depth_mask=None,
            )
        else:
            loss = loss_fn.forward(
                output,
                batch,
                gaussians,
                global_step,
                valid_depth_mask=None,
            )
        base_loss = base_loss + loss
        loss_parts[loss_fn.name] = float(loss.detach().item())
    return base_loss, loss_parts


def train(
    model_wrapper: ModelWrapper,
    train_loader,
    losses,
    optimizer,
    scheduler,
    step_tracker: LocalStepTracker,
    watermark_extractor,
    watermark_key,
    cfg,
    device: torch.device,
    ckpt_dir: Path,
    global_step: int,
    lambda_w: float,
    use_bf16: bool,
    gadm_aap_config: GADMAAPConfig,
    clean_encoder_params: dict[str, torch.Tensor] | None,
    robust_attacks: bool,
    lambda_robust: float,
    attack_config: WatermarkAttackConfig,
) -> None:
    max_steps = int(cfg.trainer.max_steps)
    print_every = int(cfg.train.print_log_every_n_steps)
    save_every = int(cfg.checkpointing.every_n_train_steps)
    data_shim = get_data_shim(model_wrapper.encoder)
    print(cyan(f"Starting training loop at step={global_step}, max_steps={max_steps}, print_every={print_every}."))

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while global_step < max_steps:
        epoch += 1
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= max_steps:
                break

            batch = _move_to_device(batch, device)
            batch = data_shim(batch)
            step_tracker.set_step(global_step)

            clean_cpu_rng = torch.get_rng_state() if gadm_aap_config.gadm else None
            clean_cuda_rng = (
                torch.cuda.get_rng_state(device) if gadm_aap_config.gadm else None
            )
            gaussians = model_wrapper.encoder(
                batch["context"],
                global_step,
                deterministic=False,
                scene_names=batch["scene"],
            )
            if isinstance(gaussians, dict):
                gaussians = gaussians["gaussians"]
            train_cpu_rng = torch.get_rng_state() if gadm_aap_config.gadm else None
            train_cuda_rng = (
                torch.cuda.get_rng_state(device) if gadm_aap_config.gadm else None
            )
            reliability = None
            loss_gadm = gaussians.harmonics.new_zeros(())
            if gadm_aap_config.gadm or gadm_aap_config.aap:
                reliability = compute_gaussian_reliability(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    chunk_size=gadm_aap_config.reliability_chunk_size,
                )
            if gadm_aap_config.gadm:
                if clean_encoder_params is None:
                    raise RuntimeError("GADM requires frozen clean encoder parameters")
                with torch.no_grad():
                    torch.set_rng_state(clean_cpu_rng)
                    torch.cuda.set_rng_state(clean_cuda_rng, device)
                    clean_gaussians = functional_call(
                        model_wrapper.encoder,
                        clean_encoder_params,
                        (batch["context"], global_step),
                        {
                            "deterministic": False,
                            "scene_names": batch["scene"],
                        },
                        strict=False,
                    )
                    if isinstance(clean_gaussians, dict):
                        clean_gaussians = clean_gaussians["gaussians"]
                    torch.set_rng_state(train_cpu_rng)
                    torch.cuda.set_rng_state(train_cuda_rng, device)
                loss_gadm = gadm_loss(
                    gaussians,
                    clean_gaussians,
                    reliability,
                    carrier=gadm_aap_config.gadm_carrier,
                    weighting=gadm_aap_config.gadm_weighting,
                    strict_carrier=gadm_aap_config.gadm_strict_carrier,
                )
                del clean_gaussians

            _, _, _, h, w = batch["target"]["image"].shape
            output = model_wrapper.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=cfg.train.depth_mode,
            )

            with torch.amp.autocast("cuda", enabled=False):
                base_loss, loss_parts = _base_losses(losses, output, batch, gaussians, global_step, cfg)
                watermark_extractor.eval()
                loss_wm, bit_acc = _watermark_loss(
                    output.color.float().clamp(0, 1),
                    watermark_extractor,
                    watermark_key,
                    aggregate_views=False,
                )
                total_loss = base_loss + lambda_w * loss_wm
                attack_name = "clean"
                if robust_attacks:
                    attack_name = sample_training_attack(output.color.device)
                    loss_robust, robust_bit_acc = _watermark_loss(
                        output.color.float().clamp(0, 1),
                        watermark_extractor,
                        watermark_key,
                        aggregate_views=False,
                        attack=attack_name,
                        attack_config=attack_config,
                        training_attack=True,
                    )
                    total_loss = total_loss + lambda_robust * loss_robust
                    loss_parts["robust"] = float(loss_robust.detach())
                    loss_parts["robust_bit_acc"] = float(robust_bit_acc.detach())
                loss_aap = output.color.new_zeros(())
                aap_bit_acc = output.color.new_zeros(())
                if gadm_aap_config.gadm:
                    total_loss = total_loss + gadm_aap_config.lambda_gadm * loss_gadm
                    loss_parts["gadm"] = float(loss_gadm.detach())
                if gadm_aap_config.aap:
                    prune_ratio = sample_prune_ratio(gadm_aap_config, output.color.device)
                    pruned_gaussians, keep_mask = prune_gaussians(
                        gaussians,
                        reliability,
                        prune_ratio,
                        strategy=gadm_aap_config.aap_strategy,
                    )
                    num_views = min(gadm_aap_config.aap_num_views, output.color.shape[1])
                    view_indices = torch.randperm(
                        output.color.shape[1], device=output.color.device
                    )[:num_views]
                    pruned_output = model_wrapper.decoder.forward(
                        pruned_gaussians,
                        batch["target"]["extrinsics"][:, view_indices],
                        batch["target"]["intrinsics"][:, view_indices],
                        batch["target"]["near"][:, view_indices],
                        batch["target"]["far"][:, view_indices],
                        (h, w),
                        depth_mode=cfg.train.depth_mode,
                    )
                    loss_aap, aap_bit_acc = _watermark_loss(
                        pruned_output.color.float().clamp(0, 1),
                        watermark_extractor,
                        watermark_key,
                        aggregate_views=False,
                    )
                    total_loss = total_loss + gadm_aap_config.lambda_aap * loss_aap
                    loss_parts["aap"] = float(loss_aap.detach())
                    loss_parts["aap_bit_acc"] = float(aap_bit_acc.detach())
                    loss_parts["aap_prune_ratio"] = float(prune_ratio.detach())
                    del pruned_gaussians, pruned_output, keep_mask

            total_loss.backward()
            if cfg.trainer.gradient_clip_val is not None and cfg.trainer.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_wrapper.parameters() if p.requires_grad],
                    cfg.trainer.gradient_clip_val,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step == 1 or global_step % print_every == 0:
                parts = "; ".join(f"{k}={v:.4f}" for k, v in loss_parts.items())
                print(
                    f"train step {global_step}; "
                    f"total_loss={total_loss.detach().item():.4f}; "
                    f"depthsplat_loss={base_loss.detach().item():.4f}; "
                    f"watermark_loss={loss_wm.detach().item():.4f}; "
                    f"bit_acc={bit_acc.detach().item():.4f}; "
                    f"attack={attack_name}; "
                    f"lr={optimizer.param_groups[0]['lr']:.6e}; "
                    f"{parts}"
                )

            if global_step % save_every == 0:
                ckpt_path = _save_manual_ckpt(ckpt_dir, model_wrapper, optimizer, scheduler, global_step)
                print(cyan(f"Saved checkpoint to {ckpt_path}"))

    final_ckpt = _save_manual_ckpt(ckpt_dir, model_wrapper, optimizer, scheduler, global_step)
    print(cyan(f"Training finished at step={global_step}. Final checkpoint: {final_ckpt}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="WwF3D manual DepthSplat fine-tuner")
    parser.add_argument("--config-path", type=str, default="config/main.yaml")
    parser.add_argument("--experiment", type=str, default="config/experiment/dl3dv.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Full DepthSplat pretrained checkpoint")
    parser.add_argument("--dataset-root", type=str, required=True, help="Raw DL3DV root")
    parser.add_argument("--wm-ckpt", type=str, required=True)
    parser.add_argument("--hidden-ckpt", type=str, default=None)
    parser.add_argument("--wm-num-bits", type=int, default=48)
    parser.add_argument("--wm-key", type=str, default="111010110101110001010101101011010010011011100010")
    parser.add_argument("--lambda-w", type=float, default=1.0)
    parser.add_argument("--gadm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-gadm", type=float, default=0.05)
    parser.add_argument("--lambda-aap", type=float, default=0.05)
    parser.add_argument("--gadm-carrier", choices=["appearance", "opacity", "geometry", "all"], default="appearance")
    parser.add_argument("--gadm-weighting", choices=["reliability", "uniform"], default="reliability")
    parser.add_argument("--gadm-strict-carrier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aap-strategy", choices=["adaptive", "random", "opacity"], default="adaptive")
    parser.add_argument("--robust-attacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-robust", type=float, default=0.02)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--crop-ratio", type=float, default=0.70)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument("--resize-ratio", type=float, default=0.50)
    parser.add_argument("--color-strength", type=float, default=0.20)
    parser.add_argument("--aap-prune-min", type=float, default=0.05)
    parser.add_argument("--aap-prune-max", type=float, default=0.30)
    parser.add_argument("--aap-num-views", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tuned-ckpt", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--evaluate-attacks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--evaluate-gaussian-attacks",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--gaussian-prune-ratio", type=float, default=0.30)
    parser.add_argument("--gaussian-random-prune-ratio", type=float, default=0.30)
    parser.add_argument("--gaussian-quant-bits", type=int, default=8)
    parser.add_argument("--gaussian-position-noise-std", type=float, default=0.005)
    parser.add_argument("--gaussian-opacity-noise-std", type=float, default=0.10)
    parser.add_argument("--gaussian-sh-noise-std", type=float, default=0.05)
    parser.add_argument("--evaluate-hidden-attacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-index", type=str, default=None)
    parser.add_argument("--eval-context-views", type=int, default=6)
    parser.add_argument("--save-eval-images", action="store_true")
    parser.add_argument("--eval-image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    attack_config = WatermarkAttackConfig(
        jpeg_quality=args.jpeg_quality,
        crop_ratio=args.crop_ratio,
        noise_std=args.noise_std,
        blur_sigma=args.blur_sigma,
        resize_ratio=args.resize_ratio,
        color_strength=args.color_strength,
    )
    gaussian_attack_config = GaussianAttackConfig(
        prune_ratio=args.gaussian_prune_ratio,
        random_prune_ratio=args.gaussian_random_prune_ratio,
        quant_bits=args.gaussian_quant_bits,
        position_noise_std=args.gaussian_position_noise_std,
        opacity_noise_std=args.gaussian_opacity_noise_std,
        sh_noise_std=args.gaussian_sh_noise_std,
    )

    config_path = Path(args.config_path)
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem
    overrides = [
        f"+experiment={Path(args.experiment).stem}",
        "dataset=dl3dv_raw",
        f"dataset.roots=[{args.dataset_root}]",
        "dataset/view_sampler=boundedv2_360",
        "dataset.view_sampler.num_context_views=6",
        "dataset.view_sampler.num_target_views=4",
        "dataset.view_sampler.max_distance_to_context_views=10",
        "data_loader.train.batch_size=1",
        "data_loader.train.num_workers=0",
        "data_loader.train.persistent_workers=false",
        "data_loader.val.num_workers=0",
        "data_loader.test.num_workers=0",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=8",
        "model.encoder.monodepth_vit_type=vitb",
        "model.encoder.supervise_intermediate_depth=false",
        "model.encoder.return_depth=false",
        "checkpointing.every_n_train_steps=500",
        "checkpointing.save_top_k=1",
        "train.print_log_every_n_steps=10",
        "train.eval_model_every_n_val=0",
        "trainer.val_check_interval=null",
        "trainer.num_sanity_val_steps=0",
    ]
    if args.eval_index:
        training_sampler_keys = (
            "dataset/view_sampler=",
            "dataset.view_sampler.num_context_views=",
            "dataset.view_sampler.num_target_views=",
            "dataset.view_sampler.max_distance_to_context_views=",
        )
        overrides = [
            override
            for override in overrides
            if not override.startswith(training_sampler_keys)
        ]
        overrides.extend(
            [
                "dataset/view_sampler=evaluation",
                f"dataset.view_sampler.index_path={args.eval_index}",
                f"dataset.view_sampler.num_context_views={args.eval_context_views}",
                "data_loader.test.batch_size=1",
                "data_loader.test.num_workers=0",
            ]
        )
    if args.max_steps is not None:
        overrides.append(f"trainer.max_steps={args.max_steps}")
    if args.save_every is not None:
        overrides.append(f"checkpointing.every_n_train_steps={args.save_every}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.mode = "test" if args.eval_only and args.eval_index else "train"
    cfg.wandb.mode = "disabled"
    cfg.trainer.num_nodes = 1
    if not args.lpips and "lpips" in cfg.loss:
        del cfg.loss["lpips"]

    cfg_typed = load_typed_root_config(cfg)
    set_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"outputs/exp_wff3d_depthsplat/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    print(cyan(f"Saving outputs to {output_dir}."))

    step_tracker = LocalStepTracker()
    data_module = DataModule(cfg_typed.dataset, cfg_typed.data_loader, step_tracker, global_rank=0)
    data_loader = (
        data_module.test_dataloader()
        if args.eval_only and args.eval_index
        else data_module.train_dataloader()
    )

    encoder, encoder_visualizer = get_encoder(cfg_typed.model.encoder)
    model_wrapper = ModelWrapper(
        cfg_typed.optimizer,
        cfg_typed.test,
        cfg_typed.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg_typed.model.decoder, cfg_typed.dataset),
        get_losses(cfg_typed.loss),
        step_tracker,
    )
    _load_full_checkpoint(model_wrapper, args.ckpt)
    model_wrapper.to(device)
    model_wrapper.log = lambda *a, **kw: None

    params = _set_trainable_modules(model_wrapper)
    clean_encoder_params = None
    if args.gadm:
        clean_encoder_params = {
            name: parameter.detach().clone()
            for name, parameter in model_wrapper.encoder.named_parameters()
            if parameter.requires_grad
        }
    optimizer = torch.optim.AdamW(params, lr=cfg_typed.optimizer.lr, weight_decay=cfg_typed.optimizer.weight_decay)

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(cfg_typed.optimizer.warm_up_steps))
        return min(1.0, float(step + 1) / float(warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    watermark_extractor = torch.jit.load(args.wm_ckpt, map_location="cpu").to(device)
    watermark_extractor.eval()
    for p in watermark_extractor.parameters():
        p.requires_grad = False
    watermark_key = torch.tensor([int(b) for b in args.wm_key], dtype=torch.float32, device=device)
    if watermark_key.numel() != args.wm_num_bits:
        raise ValueError(f"wm-key has {watermark_key.numel()} bits, expected {args.wm_num_bits}")
    print(cyan(f"[WM] HiddenDecoder loaded ({args.wm_num_bits} bits)"))
    hidden_baseline = (
        load_hidden_baseline(args.hidden_ckpt, device) if args.hidden_ckpt else None
    )
    if hidden_baseline is not None:
        print(cyan(f"[HiDDeN] input-image baseline loaded from {args.hidden_ckpt}"))

    global_step = 0
    if args.tuned_ckpt:
        global_step = _load_tuned_checkpoint(model_wrapper, args.tuned_ckpt)
    if args.resume:
        latest = _latest_manual_ckpt(ckpt_dir)
        if latest is not None:
            ckpt = torch.load(latest, map_location="cpu")
            state = model_wrapper.state_dict()
            state.update(ckpt.get("state_dict", {}))
            model_wrapper.load_state_dict(state, strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            global_step = int(ckpt.get("global_step", 0))
            print(cyan(f"Resumed from {latest} at global_step={global_step}"))

    losses = torch.nn.ModuleList(get_losses(cfg_typed.loss)).to(device)
    for loss_fn in losses:
        loss_fn.eval()

    if args.eval_only:
        evaluate_gaussian_attacks = (
            args.evaluate_attacks
            if args.evaluate_gaussian_attacks is None
            else args.evaluate_gaussian_attacks
        )
        evaluate_smoke(
            model_wrapper, data_loader, watermark_extractor, watermark_key,
            cfg_typed, device, global_step, args.eval_batches, attack_config, args.evaluate_attacks,
            evaluate_gaussian_attacks,
            gaussian_attack_config,
            hidden_baseline,
            args.evaluate_hidden_attacks,
            Path(args.eval_image_dir) if args.eval_image_dir else (
                output_dir / "eval_images" if args.save_eval_images else None
            ),
        )
        return

    train(
        model_wrapper=model_wrapper,
        train_loader=data_loader,
        losses=losses,
        optimizer=optimizer,
        scheduler=scheduler,
        step_tracker=step_tracker,
        watermark_extractor=watermark_extractor,
        watermark_key=watermark_key,
        cfg=cfg_typed,
        device=device,
        ckpt_dir=ckpt_dir,
        global_step=global_step,
        lambda_w=args.lambda_w,
        use_bf16=args.bf16,
        gadm_aap_config=GADMAAPConfig(
            gadm=args.gadm,
            aap=args.aap,
            lambda_gadm=args.lambda_gadm,
            lambda_aap=args.lambda_aap,
            prune_ratio_min=args.aap_prune_min,
            prune_ratio_max=args.aap_prune_max,
            aap_num_views=args.aap_num_views,
            gadm_carrier=args.gadm_carrier,
            gadm_weighting=args.gadm_weighting,
            gadm_strict_carrier=args.gadm_strict_carrier,
            aap_strategy=args.aap_strategy,
        ),
        clean_encoder_params=clean_encoder_params,
        robust_attacks=args.robust_attacks,
        lambda_robust=args.lambda_robust,
        attack_config=attack_config,
    )


if __name__ == "__main__":
    main()
