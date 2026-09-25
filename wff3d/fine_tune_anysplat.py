import os
import re
import glob
import json
import sys
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch
from colorama import Fore
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.step_tracker import StepTracker
from src.model.model import get_model
from src.model.model_wrapper import ModelWrapper
from src.watermark import ATTACK_NAMES, GADMAAPConfig, WatermarkAttackConfig


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def _checkpoint_exclude_prefixes() -> tuple[str, ...]:
    return (
        "model.encoder.",
        "watermark_extractor.",
        "watermark_key",
    )


def _trainable_gaussian_prefixes() -> tuple[str, ...]:
    return ("model.encoder.gaussian_param_head.",)


def _latest_manual_ckpt(ckpt_dir: Path) -> Path | None:
    ckpts = glob.glob(str(ckpt_dir / "step_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))
    return Path(ckpts[-1])


def _save_manual_ckpt(
    ckpt_dir: Path,
    model_wrapper: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    epoch: int,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefixes = _trainable_gaussian_prefixes()
    model_state = {
        k: v.detach().cpu()
        for k, v in model_wrapper.state_dict().items()
        if any(k.startswith(p) for p in prefixes)
    }
    payload = {
        "state_dict": model_state,
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
    }
    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def _load_tuned_checkpoint(model_wrapper: ModelWrapper, ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tuned_state = ckpt.get("state_dict", ckpt)
    current_state = model_wrapper.state_dict()
    expected = {
        key for key in current_state
        if any(key.startswith(prefix) for prefix in _trainable_gaussian_prefixes())
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
    print(cyan(f"[AnySplat] tuned checkpoint loaded strictly: {ckpt_path} step={step}"))
    return step


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




def train(model_wrapper, train_loader, optimizer, scheduler, step_tracker, ckpt_dir, cfg, device, global_step, output_dir, use_bf16):
    """Training loop (init is done outside)."""
    max_steps = int(cfg.trainer.max_steps)
    grad_accum = int(cfg.trainer.accumulate_grad_batches)
    print_every = int(cfg.train.print_log_every_n_steps)
    save_every = int(cfg.checkpointing.every_n_train_steps)
    epoch = 0

    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps:
        epoch += 1
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            batch = _move_to_device(batch, device)
            step_tracker.set_step(global_step)
            model_wrapper._manual_global_step = global_step

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model_wrapper.training_step(batch, batch_idx)
            loss = loss / grad_accum
            loss.backward()

            if global_step == 0 and batch_idx == 0:
                total_grad_norm = 0.0
                for n, p in model_wrapper.named_parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"[GRAD-CHECK] step 0 total grad norm = {total_grad_norm:.6f}")

            if (batch_idx + 1) % grad_accum == 0:
                if cfg.trainer.gradient_clip_val is not None and cfg.trainer.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), cfg.trainer.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % print_every == 0:
                    metrics = getattr(model_wrapper, "latest_train_metrics", {})
                    parts = metrics.get("parts", {}) if isinstance(metrics, dict) else {}
                    parts_str = "; ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    if parts_str:
                        parts_str = "; " + parts_str
                    print(
                        f"train step {global_step}; "
                        f"total_loss={metrics.get('total_loss', loss.item() * grad_accum):.4f}; "
                        f"anysplat_loss={metrics.get('anysplat_loss', float('nan')):.4f}; "
                        f"watermark_loss={metrics.get('watermark_loss', float('nan')):.4f}; "
                        f"bit_acc={metrics.get('bit_acc', float('nan')):.4f}; "
                        f"lr={optimizer.param_groups[0]['lr']:.6e}"
                        f"{parts_str}"
                    )

                if global_step % save_every == 0:
                    ckpt_path = _save_manual_ckpt(
                        ckpt_dir=ckpt_dir,
                        model_wrapper=model_wrapper,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                    )
                    print(cyan(f"Saved checkpoint to {ckpt_path}"))

    final_ckpt = _save_manual_ckpt(
        ckpt_dir=ckpt_dir,
        model_wrapper=model_wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=epoch,
    )
    print(cyan(f"Training finished at step={global_step}. Final checkpoint: {final_ckpt}"))


@torch.no_grad()
def evaluate_smoke(model_wrapper, data_loader, step_tracker, device, global_step: int, num_batches: int) -> None:
    model_wrapper.eval()
    model_wrapper.model.encoder.eval()
    model_wrapper.watermark_extractor.eval()
    model_wrapper._manual_global_step = global_step
    model_wrapper._wff3d_eval_metrics = True
    model_wrapper._wff3d_eval_attacks = True
    totals = {"bit_acc": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    attack_totals = {name: 0.0 for name in ATTACK_NAMES}
    count = 0
    for batch_idx, batch in enumerate(data_loader):
        batch = _move_to_device(batch, device)
        step_tracker.set_step(global_step)
        model_wrapper.training_step(batch, batch_idx)
        metrics = model_wrapper.latest_train_metrics
        for key in totals:
            totals[key] += float(metrics[key])
        for key in ATTACK_NAMES:
            attack_totals[key] += model_wrapper.latest_attack_metrics[key]
        count += 1
        print(f"[WFF3D-EVAL][AnySplat] batch={count} bit_acc={metrics['bit_acc']:.4f}")
        if count >= num_batches:
            break
    if count == 0:
        raise RuntimeError("AnySplat smoke evaluation received no batches")
    result = {key: value / count for key, value in totals.items()}
    result.update({f"bit_acc_{key}": value / count for key, value in attack_totals.items()})
    result["bit_acc_clean"] = result["bit_acc"]
    result.update({"model": "AnySplat", "batches": count, "global_step": global_step})
    print("[WFF3D-EVAL-RESULT] " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wff3d manual trainer (non-Lightning)")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/main.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment config path, e.g. config/experiment/dl3dv.yaml",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="Set checkpointing.resume")
    parser.add_argument("--distill", action=argparse.BooleanOptionalAction, default=None, help="Set model.encoder.distill")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None, help="Set model.encoder.freeze_backbone")
    parser.add_argument("--val-check-interval", type=int, default=None, help="Set trainer.val_check_interval")
    parser.add_argument("--num-nodes", type=int, default=None, help="Set trainer.num_nodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Set trainer.max_steps")
    parser.add_argument("--wm-num-bits", type=int, default=48, help="paras.num_bits")
    parser.add_argument("--wm_ckpt", type=str, default=None, help="checkpoints/dec_48b_whit.torchscript.pt")
    parser.add_argument("--wm-key", type=str, default="111010110101110001010101101011010010011011100010", help="paras.key (binary string)")
    parser.add_argument("--tuned-ckpt", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--gadm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-gadm", type=float, default=0.05)
    parser.add_argument("--lambda-aap", type=float, default=1.0)
    parser.add_argument("--gadm-carrier", choices=["appearance", "opacity", "geometry", "all"], default="appearance")
    parser.add_argument("--gadm-weighting", choices=["reliability", "uniform"], default="reliability")
    parser.add_argument("--gadm-strict-carrier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aap-strategy", choices=["adaptive", "random", "opacity"], default="adaptive")
    parser.add_argument("--robust-attacks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reconstruction-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the configured reconstruction and perceptual losses.",
    )
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
    parser.add_argument("--lambda-w", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    args = parser.parse_args()
    attack_config = WatermarkAttackConfig(
        jpeg_quality=args.jpeg_quality,
        crop_ratio=args.crop_ratio,
        noise_std=args.noise_std,
        blur_sigma=args.blur_sigma,
        resize_ratio=args.resize_ratio,
        color_strength=args.color_strength,
    )

    config_path = Path(args.config_path)
    if config_path.is_file():
        config_dir = config_path.parent.resolve()
        config_name = config_path.stem
    else:
        config_dir = config_path.resolve()
        config_name = "main"

    overrides = []
    if args.experiment:
        exp_name = Path(args.experiment).stem
        overrides.append(f"+experiment={exp_name}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    # Simple explicit CLI knobs (take precedence).
    cli_updates = {
        "checkpointing.resume": args.resume,
        "model.encoder.distill": args.distill,
        "model.encoder.freeze_backbone": args.freeze_backbone,
        "trainer.val_check_interval": args.val_check_interval,
        "trainer.num_nodes": args.num_nodes,
        "trainer.max_steps": args.max_steps,
        "optimizer.lr": args.lr,
        "checkpointing.every_n_train_steps": args.save_every,
    }
    for key, value in cli_updates.items():
        if value is not None:
            OmegaConf.update(cfg, key, value, merge=False)

    # ---- Init models, dataset, optimizer ----
    cfg_dict = cfg
    cfg_typed = load_typed_root_config(cfg)
    set_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg_dict.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg_dict.seed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif cfg_typed.checkpointing.resume:
        output_dir = Path(f"output/exp_{cfg_dict.wandb.name}")
    else:
        output_dir = Path(f"output/exp_{cfg_dict.wandb.name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_typed.train.output_path = output_dir
    print(cyan(f"Saving outputs to {output_dir}."))

    step_tracker = StepTracker()
    data_module = DataModule(cfg_typed.dataset, cfg_typed.data_loader, step_tracker, global_rank=0)
    train_loader = data_module.train_dataloader()

    print("[Anysplat] loading anysplat ... ")
    model = get_model(cfg_typed.model.encoder, cfg_typed.model.decoder)
    init_ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    encoder_fixed_cpu_path = init_ckpt_dir / "encoder_fixed_cpu.pt"
    decoder_init_cpu_path = init_ckpt_dir / "decoder_init_cpu.pt"
    if not encoder_fixed_cpu_path.exists() or not decoder_init_cpu_path.exists():
        raise FileNotFoundError(
            f"Missing init checkpoints in {init_ckpt_dir}. "
            "Please run save_init_encoder_decoder.py first."
        )

    encoder_state = torch.load(encoder_fixed_cpu_path, map_location="cpu")
    decoder_state = torch.load(decoder_init_cpu_path, map_location="cpu")
    try:
        model.encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as e:
        if cfg_typed.model.encoder.distill:
            raise RuntimeError(
                "Strict encoder load failed: distill modules missing from encoder_fixed_cpu.pt"
            ) from e
        raise
    model.encoder.gaussian_param_head.load_state_dict(decoder_state, strict=True)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.gaussian_param_head.parameters():
        p.requires_grad = True
    print(f"[Anysplat] fixed encoder loaded from {encoder_fixed_cpu_path}")
    print(f"[Anysplat] init gaussian head loaded from {decoder_init_cpu_path}")
    print("[Anysplat] loaded model done. Encoder fixed, gaussian head trainable.")

    clean_gaussian_head = None
    if args.gadm:
        clean_gaussian_head = deepcopy(model.encoder.gaussian_param_head)
        clean_gaussian_head.eval()
        for parameter in clean_gaussian_head.parameters():
            parameter.requires_grad = False
        print("[GADM] Frozen initialization head created for clean Gaussian supervision.")

    print("[WM] Loading HiddenDecoder ...")
    
    watermark_extractor = torch.jit.load(args.wm_ckpt).to(device)
    watermark_extractor.eval()
    
    for p in watermark_extractor.parameters():
        p.requires_grad = False
    print(f"[WM] HiddenDecoder loaded ({args.wm_num_bits} bits)")
    watermark_key = torch.tensor([int(b) for b in args.wm_key], dtype=torch.float32, device=device)


    model_wrapper = ModelWrapper(
        cfg_typed.optimizer, cfg_typed.test, cfg_typed.train,
        model, get_losses(cfg_typed.loss) if args.reconstruction_loss else [], step_tracker,
        watermark_extractor=watermark_extractor, watermark_key=watermark_key,
        watermark_weight=args.lambda_w,
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
        clean_gaussian_head=clean_gaussian_head,
        robust_attacks=args.robust_attacks,
        robust_weight=args.lambda_robust,
        attack_config=attack_config,
    ).to(device)
    model_wrapper.train()
    model_wrapper.model.encoder.eval()
    if model_wrapper.clean_gaussian_head is not None:
        model_wrapper.clean_gaussian_head.eval()
    for p in model_wrapper.model.encoder.parameters():
        p.requires_grad = False
    for p in model_wrapper.model.encoder.gaussian_param_head.parameters():
        p.requires_grad = True
    if model_wrapper.watermark_extractor is not None:
        model_wrapper.watermark_extractor.eval()
        for p in model_wrapper.watermark_extractor.parameters():
            p.requires_grad = False

    model_wrapper.log = lambda *args, **kwargs: None
    model_wrapper.trainer = SimpleNamespace(global_rank=0, datamodule=data_module)

    tuned_global_step = 0
    if args.tuned_ckpt:
        tuned_global_step = _load_tuned_checkpoint(model_wrapper, args.tuned_ckpt)

    optim_cfg = model_wrapper.configure_optimizers()
    optimizer = optim_cfg["optimizer"]
    scheduler = optim_cfg["lr_scheduler"]["scheduler"]

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = tuned_global_step
    if cfg_typed.checkpointing.resume:
        ckpt_path = _latest_manual_ckpt(ckpt_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            current_state = model_wrapper.state_dict()
            for k, v in ckpt.get("state_dict", {}).items():
                if k in current_state and k.startswith("model.encoder.gaussian_param_head."):
                    current_state[k] = v
            model_wrapper.load_state_dict(current_state, strict=False)
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            global_step = int(ckpt.get("global_step", 0))
            print(cyan(f"Resumed from {ckpt_path} at global_step={global_step}"))
        else:
            print(cyan(f"No manual checkpoint found in {ckpt_dir}, starting fresh."))

    if args.eval_only:
        evaluate_smoke(model_wrapper, train_loader, step_tracker, device, global_step, args.eval_batches)
        raise SystemExit(0)

    train(
        model_wrapper, train_loader, optimizer, scheduler, step_tracker,
        ckpt_dir, cfg_typed, device, global_step, output_dir, args.bf16,
    )
