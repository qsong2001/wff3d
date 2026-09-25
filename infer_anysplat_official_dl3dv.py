#!/usr/bin/env python3
"""Official-style AnySplat DL3DV inference on YoNoSplat index files.

This bypasses the Lightning dataloader and follows AnySplat's upstream
``src/eval_nvs.py`` path:
  - AnySplat.from_pretrained("lhjiang/anysplat")
  - src.utils.image.process_image (448x448 center crop)
  - VGGT predicts target poses from context + target images
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


ROOT = Path(__file__).resolve().parent
ANYSPLAT_ROOT = Path(os.environ.get("ANYSPLAT_ROOT", ROOT / "AnySplat"))
sys.path.insert(0, str(ANYSPLAT_ROOT))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim  # noqa: E402
from src.misc.image_io import save_image  # noqa: E402
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from src.model.model.anysplat import AnySplat  # noqa: E402
from src.model.ply_export import export_ply  # noqa: E402
from src.utils.image import process_image  # noqa: E402

sys.path.insert(0, str(ROOT / "wff3d/src/watermark"))
from image_attacks import (  # noqa: E402
    ATTACK_NAMES,
    WatermarkAttackConfig,
    apply_image_attack,
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


INDEX_SPECS = {
    "6v_tgt8": ("yonosplat_dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json", 6),
    "6v_video": ("yonosplat_dl3dv_start_0_distance_50_ctx_6v_video.json", 6),
    "12v_tgt8": ("yonosplat_dl3dv_start_0_distance_100_ctx_12v_tgt_8v.json", 12),
    "12v_video": ("yonosplat_dl3dv_start_0_distance_100_ctx_12v_video.json", 12),
    "24v_tgt8": ("yonosplat_dl3dv_start_0_distance_150_ctx_24v_tgt_8v.json", 24),
    "24v_video": ("yonosplat_dl3dv_start_0_distance_150_ctx_24v_video.json", 24),
}

BLENDER_TO_OPENCV = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "wff3d/datasets/DL3DV")
    parser.add_argument(
        "--index-root",
        type=Path,
        default=ROOT / "YoNoSplat/assets",
        help="Directory containing the official YoNoSplat DL3DV evaluation JSON files.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/anysplat_official_infer")
    parser.add_argument("--model-id", default="lhjiang/anysplat")
    parser.add_argument("--tuned-ckpt", type=Path, default=None)
    parser.add_argument("--wm-ckpt", type=Path, default=None)
    parser.add_argument("--hidden-ckpt", type=Path, default=None)
    parser.add_argument(
        "--wm-key",
        default="111010110101110001010101101011010010011011100010",
    )
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--crop-ratio", type=float, default=0.70)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument("--resize-ratio", type=float, default=0.50)
    parser.add_argument("--color-strength", type=float, default=0.20)
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
    parser.add_argument("--tags", default="all", help="Comma/colon separated tags or all.")
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--eval-image-dir",
        type=Path,
        default=None,
        help="Optional canonical per-scene image directory used by qualitative figures.",
    )
    parser.add_argument("--save-gaussians", action="store_true")
    parser.add_argument(
        "--baseline-export-root",
        type=Path,
        default=None,
        help="Write self-render-supervised 3DGS watermark baseline packages here.",
    )
    parser.add_argument(
        "--pose-mode",
        choices=["vggt", "gt"],
        default="vggt",
        help="Use VGGT-predicted target poses (official demo) or experimental GT target poses aligned to predicted context poses.",
    )
    parser.add_argument(
        "--image-shape",
        default="448x448",
        help="Preprocess shape as HxW. 448x448 matches upstream eval_nvs.py; 224x448 matches DL3DV training config.",
    )
    parser.add_argument(
        "--image-dirs",
        default="images_4,images_8,images_2,images",
        help="Comma-separated image directory priority when transforms.json points to images/.",
    )
    parser.add_argument(
        "--metric-crop-border",
        type=int,
        default=0,
        help="Optional border crop before computing PSNR/SSIM/LPIPS.",
    )
    return parser.parse_args()


def wanted_tags(tags: str) -> list[str]:
    if tags == "all":
        return list(INDEX_SPECS)
    selected = [t for t in tags.replace(":", ",").split(",") if t]
    unknown = sorted(set(selected) - set(INDEX_SPECS))
    if unknown:
        raise SystemExit(f"Unknown tags: {unknown}")
    return selected


def resolve_index_path(index_root: Path, index_name: str) -> Path:
    """Support both local converted and upstream YoNoSplat index names."""
    candidates = [index_root / index_name]
    if index_name.startswith("yonosplat_"):
        candidates.append(index_root / index_name.removeprefix("yonosplat_"))
    else:
        candidates.append(index_root / f"yonosplat_{index_name}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Evaluation index not found; tried:\n  {tried}")


def resolve_image_path(scene_dir: Path, file_path: str, image_dirs: str = "images_4,images_8,images_2,images") -> Path:
    rel = Path(file_path)
    candidates = [scene_dir / rel]
    parts = list(rel.parts)
    if parts and parts[0] == "images":
        for image_dir in [x for x in image_dirs.split(",") if x]:
            candidates.append(scene_dir / Path(image_dir, *parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image for {scene_dir}: {file_path}")


def blender_to_opencv_c2w(transform_matrix: list[list[float]], device: torch.device | None = None) -> torch.Tensor:
    blender2opencv = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    c2w_blender = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    return c2w_blender @ blender2opencv


def crop_intrinsics(meta: dict, image_path: Path) -> torch.Tensor:
    """Return normalized intrinsics after AnySplat process_image's 448 center crop."""
    with Image.open(image_path) as img:
        width, height = img.convert("RGB").size
    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    sx = new_width / width
    sy = new_height / height
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2

    intr = torch.eye(3, dtype=torch.float32)
    intr[0, 0] = (float(meta["fl_x"]) * sx) / 448.0
    intr[1, 1] = (float(meta["fl_y"]) * sy) / 448.0
    intr[0, 2] = (float(meta["cx"]) * sx - left) / 448.0
    intr[1, 2] = (float(meta["cy"]) * sy - top) / 448.0
    return intr


def parse_shape(shape: str) -> tuple[int, int]:
    h, w = shape.lower().replace(",", "x").split("x")
    return int(h), int(w)


def write_baseline_package(
    root: Path,
    scene_key: str,
    tag: str,
    clean_renders: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    gaussians,
    background_color: torch.Tensor,
    iteration: int = 30000,
    sh_degree: int = 4,
) -> None:
    scene_name = scene_key.replace("/", "_")
    scene_root = root / tag / scene_name
    source_dir = scene_root / "source"
    train_dir = source_dir / "train"
    model_dir = scene_root / "model"
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    native_path = model_dir / "anysplat_native.pt"
    train_dir.mkdir(parents=True, exist_ok=True)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = clean_renders.shape[-2:]
    rel_paths = []
    image_names = []
    for i, image in enumerate(clean_renders):
        name = f"r_{i:05d}.png"
        save_image(image.detach().float().clamp(0, 1).cpu(), train_dir / name)
        rel_paths.append(str(Path("train") / Path(name).stem))
        image_names.append(Path(name).stem)

    d_sh = min(gaussians.harmonics.shape[-1], (sh_degree + 1) ** 2)
    export_ply(
        gaussians.means[0].detach().float().cpu(),
        gaussians.scales[0].detach().float().cpu().clamp_min(1e-8),
        gaussians.rotations[0].detach().float().cpu(),
        gaussians.harmonics[0, :, :, :d_sh].detach().float().cpu(),
        gaussians.opacities[0].detach().float().cpu(),
        ply_path,
        shift_and_scale=False,
        save_sh_dc_only=False,
    )
    torch.save(
        {
            "format": "anysplat-native-v1",
            "means": gaussians.means[0].detach().float().cpu(),
            "covariances": gaussians.covariances[0].detach().float().cpu(),
            "harmonics": gaussians.harmonics[0].detach().float().cpu(),
            "opacities": gaussians.opacities[0].detach().float().cpu(),
            "scales": gaussians.scales[0].detach().float().cpu(),
            "rotations": gaussians.rotations[0].detach().float().cpu(),
            "target_extrinsics": target_extrinsics[0].detach().float().cpu(),
            "target_intrinsics": target_intrinsics[0].detach().float().cpu(),
            "image_shape": [int(h), int(w)],
            "background_color": background_color.detach().float().cpu(),
        },
        native_path,
    )
    points_path = source_dir / "points3d.ply"
    try:
        points_path.unlink(missing_ok=True)
        os.link(ply_path, points_path)
    except OSError:
        shutil.copy2(ply_path, points_path)

    extrinsics = target_extrinsics[0].detach().float().cpu()
    intrinsics = target_intrinsics[0].detach().float().cpu()
    fx = float(intrinsics[0, 0, 0]) * w
    fy = float(intrinsics[0, 1, 1]) * h
    cx = float(intrinsics[0, 0, 2]) * w
    cy = float(intrinsics[0, 1, 2]) * h
    frames = []
    cameras = []
    for i, (rel_path, c2w_cv) in enumerate(zip(rel_paths, extrinsics)):
        c2w_blender = c2w_cv @ BLENDER_TO_OPENCV
        frames.append(
            {
                "file_path": rel_path,
                "transform_matrix": c2w_blender.tolist(),
                "fl_x": float(intrinsics[i, 0, 0]) * w,
                "fl_y": float(intrinsics[i, 1, 1]) * h,
                "cx": float(intrinsics[i, 0, 2]) * w,
                "cy": float(intrinsics[i, 1, 2]) * h,
            }
        )
        cameras.append(
            {
                "id": i,
                "img_name": image_names[i],
                "width": w,
                "height": h,
                "position": c2w_cv[:3, 3].tolist(),
                "rotation": c2w_cv[:3, :3].tolist(),
                "fx": float(intrinsics[i, 0, 0]) * w,
                "fy": float(intrinsics[i, 1, 1]) * h,
            }
        )
    transforms = {
        "camera_angle_x": 2.0 * math.atan(w / (2.0 * fx)),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": frames,
    }
    (source_dir / "transforms_train.json").write_text(json.dumps(transforms, indent=2))
    (source_dir / "transforms_test.json").write_text(json.dumps(transforms, indent=2))
    (model_dir / "cameras.json").write_text(json.dumps(cameras, indent=2))
    (model_dir / "cfg_args").write_text(
        "Namespace("
        f"source_path='{source_dir}', model_path='{model_dir}', images='images', "
        f"resolution=-1, white_background=True, sh_degree={sh_degree}, "
        "data_device='cuda', eval=False)\n"
    )
    print(
        f"[BASELINE-EXPORT] {tag} {scene_key}: clean_views={len(clean_renders)} "
        f"path={scene_root} native={native_path}"
    )


def process_image_to_shape(img_path: Path, shape: tuple[int, int]) -> torch.Tensor:
    if shape == (448, 448):
        return process_image(img_path)
    img = Image.open(img_path).convert("RGB")
    h_out, w_out = shape
    width, height = img.size
    scale = max(h_out / height, w_out / width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - w_out) // 2
    top = (new_height - h_out) // 2
    img = img.crop((left, top, left + w_out, top + h_out))
    return TF.to_tensor(img) * 2.0 - 1.0


def load_scene_images(
    scene_dir: Path,
    frame_indices: list[int],
    image_shape: tuple[int, int],
    image_dirs: str,
) -> torch.Tensor:
    transforms = json.loads((scene_dir / "transforms.json").read_text())
    frames = transforms["frames"]
    images = []
    for idx in frame_indices:
        img_path = resolve_image_path(scene_dir, frames[idx]["file_path"], image_dirs)
        images.append(process_image_to_shape(img_path, image_shape))
    # process_image returns [-1, 1]; official eval_nvs converts to [0, 1].
    return (torch.stack(images, dim=0).unsqueeze(0) + 1.0) * 0.5


def load_scene_cameras(scene_dir: Path, frame_indices: list[int], device: torch.device):
    transforms = json.loads((scene_dir / "transforms.json").read_text())
    frames = transforms["frames"]
    extrinsics, intrinsics = [], []
    for idx in frame_indices:
        frame = frames[idx]
        img_path = resolve_image_path(scene_dir, frame["file_path"])
        extrinsics.append(blender_to_opencv_c2w(frame["transform_matrix"], device))
        intrinsics.append(crop_intrinsics({**transforms, **frame}, img_path).to(device))
    return torch.stack(extrinsics, dim=0).unsqueeze(0), torch.stack(intrinsics, dim=0).unsqueeze(0)


def umeyama_similarity(src: torch.Tensor, dst: torch.Tensor):
    """Find x_dst ~= scale * R @ x_src + t for camera centers."""
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, svals, vh = torch.linalg.svd(cov)
    d = torch.ones(3, device=src.device, dtype=src.dtype)
    if torch.det(u @ vh) < 0:
        d[-1] = -1
    r = u @ torch.diag(d) @ vh
    var_src = (src_c.square().sum(dim=1)).mean().clamp_min(1e-8)
    scale = (svals * d).sum() / var_src
    t = dst_mean[0] - scale * (r @ src_mean[0])
    return scale, r, t


def align_gt_targets_to_pred_context(
    gt_context_extrinsics: torch.Tensor,
    gt_target_extrinsics: torch.Tensor,
    pred_context_extrinsics: torch.Tensor,
) -> torch.Tensor:
    aligned = gt_target_extrinsics.clone()
    for b in range(gt_context_extrinsics.shape[0]):
        scale, rot, trans = umeyama_similarity(
            gt_context_extrinsics[b, :, :3, 3],
            pred_context_extrinsics[b, :, :3, 3],
        )
        aligned[b, :, :3, :3] = rot @ gt_target_extrinsics[b, :, :3, :3]
        aligned[b, :, :3, 3] = scale * (gt_target_extrinsics[b, :, :3, 3] @ rot.T) + trans
    return aligned


def crop_for_metrics(pred: torch.Tensor, gt: torch.Tensor, border: int) -> tuple[torch.Tensor, torch.Tensor]:
    if border <= 0:
        return pred, gt
    h, w = pred.shape[-2:]
    if border * 2 >= h or border * 2 >= w:
        raise ValueError(f"metric crop border {border} too large for {h}x{w}")
    return pred[..., border:-border, border:-border], gt[..., border:-border, border:-border]


def scene_image_manifest(scene_dir: Path, frame_indices: list[int], image_dirs: str) -> list[str]:
    transforms = json.loads((scene_dir / "transforms.json").read_text())
    frames = transforms["frames"]
    return [
        str(resolve_image_path(scene_dir, frames[int(idx)]["file_path"], image_dirs))
        for idx in frame_indices
    ]


def discover_scene_dirs(data_root: Path) -> dict[str, Path]:
    """Map DL3DV scene hashes to local folders across all downloaded subsets."""
    direct_scenes = {
        scene_dir.name: scene_dir
        for scene_dir in data_root.iterdir()
        if scene_dir.is_dir() and (scene_dir / "transforms.json").is_file()
    } if data_root.is_dir() else {}
    if direct_scenes:
        return direct_scenes
    dataset_root = data_root.parent
    subset_roots = [dataset_root / "DL3DV-1K" / "1K"]
    subset_roots.extend(dataset_root / "DL3DV-10K" / f"{subset}K" for subset in range(2, 12))
    scene_dirs: dict[str, Path] = {}
    for subset_root in subset_roots:
        if not subset_root.is_dir():
            continue
        for scene_dir in subset_root.iterdir():
            if scene_dir.is_dir() and (scene_dir / "transforms.json").is_file():
                scene_dirs[scene_dir.name] = scene_dir
    return scene_dirs


def scene_has_images(
    scene_dir: Path,
    frame_indices: list[int],
    image_dirs: str,
) -> bool:
    try:
        transforms = json.loads((scene_dir / "transforms.json").read_text())
        frames = transforms["frames"]
        for frame_index in frame_indices:
            if frame_index >= len(frames):
                return False
            resolve_image_path(scene_dir, frames[frame_index]["file_path"], image_dirs)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False
    return True


@torch.no_grad()
def infer_one_scene(
    model: AnySplat,
    scene_dir: Path,
    ctx_idx: list[int],
    tgt_idx: list[int],
    device: torch.device,
    pose_mode: str,
    image_shape: tuple[int, int],
    image_dirs: str,
    hidden_baseline=None,
    watermark_key: torch.Tensor | None = None,
):
    ctx_images = load_scene_images(scene_dir, ctx_idx, image_shape, image_dirs).to(device)
    tgt_images = load_scene_images(scene_dir, tgt_idx, image_shape, image_dirs).to(device)
    hidden_input_acc = None
    if hidden_baseline is not None:
        ctx_images, hidden_input_view_acc = hidden_watermark_batch(
            hidden_baseline, ctx_images, watermark_key
        )
        hidden_input_acc = float(hidden_input_view_acc.mean())
    b, v, _, h, w = tgt_images.shape

    encoder_output = model.encoder(ctx_images, global_step=0, visualization_dump={})
    gaussians = encoder_output.gaussians
    pred_context_pose = encoder_output.pred_context_pose

    if pose_mode == "gt":
        gt_context_extrinsic, _ = load_scene_cameras(scene_dir, ctx_idx, device)
        gt_target_extrinsic, pred_all_target_intrinsic = load_scene_cameras(scene_dir, tgt_idx, device)
        pred_all_target_extrinsic = align_gt_targets_to_pred_context(
            gt_context_extrinsic,
            gt_target_extrinsic,
            pred_context_pose["extrinsic"],
        )
    else:
        num_context_view = ctx_images.shape[1]
        vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            aggregated_tokens_list, _ = model.encoder.aggregator(
                vggt_input_image,
                intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
            )
        with torch.cuda.amp.autocast(enabled=False):
            fp32_tokens = [token.float() for token in aggregated_tokens_list]
            pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
            pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
                pred_all_pose_enc,
                vggt_input_image.shape[-2:],
            )

        pad = torch.tensor(
            [0, 0, 0, 1],
            device=pred_all_extrinsic.device,
            dtype=pred_all_extrinsic.dtype,
        ).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
        pred_all_extrinsic = torch.cat([pred_all_extrinsic, pad], dim=2).inverse()

        pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
        pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
        pred_all_context_extrinsic = pred_all_extrinsic[:, :num_context_view]
        pred_all_target_extrinsic = pred_all_extrinsic[:, num_context_view:]
        pred_all_target_intrinsic = pred_all_intrinsic[:, num_context_view:]

        scale_factor = (
            pred_context_pose["extrinsic"][:, :, :3, 3].mean()
            / pred_all_context_extrinsic[:, :, :3, 3].mean()
        )
        pred_all_target_extrinsic[..., :3, 3] *= scale_factor

    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v, device=device) * 0.01,
        torch.ones(1, v, device=device) * 100.0,
        (h, w),
    )
    pred = output.color[0].clamp(0, 1)
    gt = tgt_images[0].clamp(0, 1)
    return (
        pred,
        gt,
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic,
        hidden_input_acc,
    )


def render_gaussians(
    model: AnySplat,
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    views = extrinsics.shape[1]
    output = model.decoder.forward(
        gaussians,
        extrinsics,
        intrinsics.float(),
        torch.full((1, views), 0.01, device=extrinsics.device),
        torch.full((1, views), 100.0, device=extrinsics.device),
        image_shape,
    )
    return output.color[0].clamp(0, 1)


@torch.no_grad()
def run_tag(
    model: AnySplat,
    args: argparse.Namespace,
    tag: str,
    device: torch.device,
    scene_dirs: dict[str, Path],
    watermark_extractor=None,
    watermark_key: torch.Tensor | None = None,
    hidden_baseline=None,
) -> dict[str, float]:
    index_name, ctx_views = INDEX_SPECS[tag]
    index_path = resolve_index_path(args.index_root, index_name)
    index = json.loads(index_path.read_text())
    all_items = list(index.items())
    items = []
    missing_scenes = []
    for scene_key, entry in all_items:
        scene_dir = scene_dirs.get(scene_key.split("/")[-1])
        required_indices = list(entry["context"]) + list(entry["target"])
        if scene_dir is None or not scene_has_images(scene_dir, required_indices, args.image_dirs):
            missing_scenes.append(scene_key)
        else:
            items.append((scene_key, entry))
    print(
        f"[AnySplatOfficial] {tag} official_scenes={len(all_items)} "
        f"available_scenes={len(items)} missing_scenes={len(missing_scenes)}",
        flush=True,
    )
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"Invalid shard {args.shard_id}/{args.num_shards}")
    items = [item for index, item in enumerate(items) if index % args.num_shards == args.shard_id]
    print(
        f"[AnySplatOfficial] {tag} shard={args.shard_id}/{args.num_shards} "
        f"selected_scenes={len(items)}",
        flush=True,
    )
    if args.scene_limit > 0:
        items = items[: args.scene_limit]

    out_dir = args.output_root / tag / f"anysplat_dl3dv_{tag}"
    if args.save_images or args.eval_image_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    psnrs, ssims, lpipss, bit_accs = [], [], [], []
    hidden_metrics = {
        "input_bit_acc": [],
        "bit_acc": [],
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }
    hidden_attack_accs = {name: [] for name in ATTACK_NAMES}
    attack_bit_accs = {name: [] for name in ATTACK_NAMES}
    gaussian_attack_metrics = {
        name: {"bit_acc": [], "psnr": [], "ssim": [], "lpips": []}
        for name in GAUSSIAN_ATTACK_NAMES
    }
    evaluate_gaussian_attacks = (
        args.evaluate_attacks
        if args.evaluate_gaussian_attacks is None
        else args.evaluate_gaussian_attacks
    )
    gaussian_attack_config = GaussianAttackConfig(
        prune_ratio=args.gaussian_prune_ratio,
        random_prune_ratio=args.gaussian_random_prune_ratio,
        quant_bits=args.gaussian_quant_bits,
        position_noise_std=args.gaussian_position_noise_std,
        opacity_noise_std=args.gaussian_opacity_noise_std,
        sh_noise_std=args.gaussian_sh_noise_std,
    )
    attack_config = WatermarkAttackConfig(
        jpeg_quality=args.jpeg_quality,
        crop_ratio=args.crop_ratio,
        noise_std=args.noise_std,
        blur_sigma=args.blur_sigma,
        resize_ratio=args.resize_ratio,
        color_strength=args.color_strength,
    )
    per_scene = []
    split_manifest = []
    image_shape = parse_shape(args.image_shape)
    inference_seconds = 0.0
    extraction_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for scene_i, (scene_key, entry) in enumerate(items, start=1):
        ctx_idx = list(entry["context"])
        tgt_idx = list(entry["target"])
        if len(ctx_idx) != ctx_views:
            raise RuntimeError(f"{scene_key}: expected {ctx_views} context views, got {len(ctx_idx)}")
        scene_dir = scene_dirs[scene_key.split("/")[-1]]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scene_start = time.perf_counter()
        (
            pred,
            gt,
            gaussians,
            target_extrinsics,
            target_intrinsics,
            hidden_input_acc,
        ) = infer_one_scene(
            model,
            scene_dir,
            ctx_idx,
            tgt_idx,
            device,
            args.pose_mode,
            image_shape,
            args.image_dirs,
            hidden_baseline,
            watermark_key,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - scene_start

        if args.baseline_export_root is not None:
            write_baseline_package(
                args.baseline_export_root,
                scene_key,
                tag,
                pred,
                target_extrinsics,
                target_intrinsics,
                gaussians,
                model.decoder.background_color,
            )

        if args.save_gaussians:
            gaussian_path = args.output_root / tag / "gaussians" / f"{scene_key.replace('/', '_')}.ply"
            export_ply(
                gaussians.means[0].detach().float(),
                gaussians.scales[0].detach().float().clamp_min(1e-8),
                gaussians.rotations[0].detach().float(),
                gaussians.harmonics[0].detach().float(),
                gaussians.opacities[0].detach().float(),
                gaussian_path,
                shift_and_scale=False,
                save_sh_dc_only=False,
            )

        pred_metric, gt_metric = crop_for_metrics(pred, gt, args.metric_crop_border)
        psnr_views = compute_psnr(gt_metric, pred_metric)
        ssim_views = compute_ssim(gt_metric, pred_metric)
        lpips_views = compute_lpips(gt_metric, pred_metric)
        psnr = psnr_views.mean().item()
        ssim = ssim_views.mean().item()
        lpips = lpips_views.mean().item()
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)
        bit_acc = None
        scene_hidden = {}
        scene_attack_accs = {}
        scene_gaussian_attacks = {}
        if watermark_extractor is not None:
            def decode(images: torch.Tensor) -> float:
                normalized = (
                    images - images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                ) / images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                normalized = F.interpolate(
                    normalized,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                logits = watermark_extractor(normalized)
                bits = watermark_key.view(1, -1).expand(logits.shape[0], -1)
                return ((logits > 0).float() == bits).float().mean().item()

            extraction_start = time.perf_counter()
            bit_acc = decode(pred)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            extraction_seconds += time.perf_counter() - extraction_start
            bit_accs.append(bit_acc)
            if args.evaluate_attacks:
                for attack_name in ATTACK_NAMES:
                    attacked = apply_image_attack(pred, attack_name, attack_config)
                    attacked_acc = decode(attacked)
                    attack_bit_accs[attack_name].append(attacked_acc)
                    scene_attack_accs[attack_name] = attacked_acc
            if evaluate_gaussian_attacks:
                for attack_name in GAUSSIAN_ATTACK_NAMES:
                    attacked_gaussians = apply_gaussian_attack(
                        gaussians, attack_name, gaussian_attack_config
                    )
                    attacked_pred = render_gaussians(
                        model,
                        attacked_gaussians,
                        target_extrinsics,
                        target_intrinsics,
                        pred.shape[-2:],
                    )
                    attacked_metric, attacked_gt = crop_for_metrics(
                        attacked_pred, gt, args.metric_crop_border
                    )
                    values = {
                        "bit_acc": decode(attacked_pred),
                        "psnr": compute_psnr(attacked_gt, attacked_metric).mean().item(),
                        "ssim": compute_ssim(attacked_gt, attacked_metric).mean().item(),
                        "lpips": compute_lpips(attacked_gt, attacked_metric).mean().item(),
                    }
                    scene_gaussian_attacks[attack_name] = values
                    for metric, value in values.items():
                        gaussian_attack_metrics[attack_name][metric].append(value)
        if hidden_baseline is not None:
            hidden_view_acc = hidden_decode_batch(hidden_baseline, pred, watermark_key)
            hidden_metric, hidden_gt = crop_for_metrics(pred, gt, args.metric_crop_border)
            scene_hidden = {
                "hidden_input_bit_acc": hidden_input_acc,
                "hidden_bit_acc": float(hidden_view_acc.mean()),
                "hidden_psnr": compute_psnr(hidden_gt, hidden_metric).mean().item(),
                "hidden_ssim": compute_ssim(hidden_gt, hidden_metric).mean().item(),
                "hidden_lpips": compute_lpips(hidden_gt, hidden_metric).mean().item(),
            }
            for key, value in scene_hidden.items():
                hidden_metrics[key.removeprefix("hidden_")].append(value)
            if args.evaluate_hidden_attacks:
                for attack_name in ATTACK_NAMES:
                    attacked = apply_image_attack(pred, attack_name, attack_config)
                    attacked_acc = float(
                        hidden_decode_batch(hidden_baseline, attacked, watermark_key).mean()
                    )
                    hidden_attack_accs[attack_name].append(attacked_acc)
                    scene_hidden[f"hidden_bit_acc_{attack_name}"] = attacked_acc
        per_scene.append(
            {
                "scene": scene_key,
                "context": ctx_idx,
                "target": tgt_idx,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "bit_acc": bit_acc,
                **scene_hidden,
                **{f"bit_acc_{key}": value for key, value in scene_attack_accs.items()},
                **{
                    f"{metric}_{attack_name}": value
                    for attack_name, values in scene_gaussian_attacks.items()
                    for metric, value in values.items()
                },
                "psnr_per_view": [float(x) for x in psnr_views.detach().cpu()],
                "ssim_per_view": [float(x) for x in ssim_views.detach().cpu()],
                "lpips_per_view": [float(x) for x in lpips_views.detach().cpu()],
            }
        )
        split_manifest.append(
            {
                "scene": scene_key,
                "context": ctx_idx,
                "target": tgt_idx,
                "context_images": scene_image_manifest(scene_dir, ctx_idx, args.image_dirs),
                "target_images": scene_image_manifest(scene_dir, tgt_idx, args.image_dirs),
            }
        )

        if args.save_images or args.eval_image_dir is not None:
            scene_out = (
                args.eval_image_dir / scene_key.replace("/", "_")
                if args.eval_image_dir is not None
                else out_dir / scene_key.replace("/", "_")
            )
            for i, (p, g) in enumerate(zip(pred, gt)):
                prediction_dir = "watermarked" if args.eval_image_dir is not None else "color"
                save_image(p, scene_out / prediction_dir / f"{i:06d}.png")
                save_image(g, scene_out / "gt" / f"{i:06d}.png")
            for i, c in enumerate(load_scene_images(scene_dir, ctx_idx, image_shape, args.image_dirs)[0]):
                save_image(c, scene_out / "context" / f"{i:06d}.png")
            (scene_out / "metrics.json").write_text(
                json.dumps(
                    {
                        "scene": scene_key,
                        "bit_acc": bit_acc,
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lpips,
                        "views": len(tgt_idx),
                    },
                    indent=2,
                )
                + "\n"
            )

        print(
            f"[AnySplatOfficial] {tag} {scene_i}/{len(items)} "
            f"scene={scene_key} running_psnr={sum(psnrs)/len(psnrs):.3f} "
            f"running_ssim={sum(ssims)/len(ssims):.3f} running_lpips={sum(lpipss)/len(lpipss):.3f}"
            + (f" running_bit_acc={sum(bit_accs)/len(bit_accs):.4f}" if bit_accs else ""),
            flush=True,
        )

    result = {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": sum(lpipss) / len(lpipss),
        "scenes": len(items),
        "official_index_scenes": len(all_items),
        "missing_scenes": missing_scenes,
        "index_path": str(index_path),
        "gaussian_root": str(args.output_root / tag / "gaussians") if args.save_gaussians else None,
    }
    if bit_accs:
        result["bit_acc"] = sum(bit_accs) / len(bit_accs)
        result["bit_acc_clean"] = result["bit_acc"]
        if args.evaluate_attacks:
            result.update(
                {
                    f"bit_acc_{key}": sum(values) / len(values)
                    for key, values in attack_bit_accs.items()
                }
            )
        if evaluate_gaussian_attacks:
            for attack_name, values in gaussian_attack_metrics.items():
                result.update(
                    {
                        f"{metric}_{attack_name}": sum(metric_values) / len(metric_values)
                        for metric, metric_values in values.items()
                    }
                )
        result["seconds_per_scene"] = inference_seconds / len(items)
        result["extraction_ms_per_scene"] = extraction_seconds * 1000 / len(items)
        result["peak_gpu_memory_gb"] = (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        )
    if hidden_metrics["bit_acc"]:
        result.update(
            {
                f"hidden_{metric}": sum(values) / len(values)
                for metric, values in hidden_metrics.items()
            }
        )
        if args.evaluate_hidden_attacks:
            result.update(
                {
                    f"hidden_bit_acc_{attack}": sum(values) / len(values)
                    for attack, values in hidden_attack_accs.items()
                }
            )
    (args.output_root / tag).mkdir(parents=True, exist_ok=True)
    (args.output_root / tag / "metrics.json").write_text(json.dumps(result, indent=2))
    (args.output_root / tag / "per_scene_metrics.json").write_text(json.dumps(per_scene, indent=2))
    (args.output_root / tag / "split_image_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    print(
        f"[AnySplatOfficial][FINAL] {tag} scenes={result['scenes']} "
        f"PSNR={result['psnr']:.3f} SSIM={result['ssim']:.3f} LPIPS={result['lpips']:.3f}"
        + (f" BitAcc={result['bit_acc']:.4f}" if "bit_acc" in result else ""),
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading AnySplat official model: {args.model_id}", flush=True)
    model = AnySplat.from_pretrained(args.model_id)
    if args.tuned_ckpt is not None:
        checkpoint = torch.load(args.tuned_ckpt, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        prefix = "model.encoder.gaussian_param_head."
        head_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        if not head_state:
            raise RuntimeError(f"No tuned AnySplat Gaussian head found in {args.tuned_ckpt}")
        model.encoder.gaussian_param_head.load_state_dict(head_state, strict=True)
        print(f"[INFO] Strictly loaded tuned AnySplat head: {args.tuned_ckpt}", flush=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    watermark_extractor = None
    watermark_key = None
    if args.wm_ckpt is not None:
        watermark_extractor = torch.jit.load(str(args.wm_ckpt), map_location="cpu").to(device).eval()
        for parameter in watermark_extractor.parameters():
            parameter.requires_grad = False
        watermark_key = torch.tensor([int(bit) for bit in args.wm_key], device=device, dtype=torch.float32)
        print(f"[INFO] Loaded fixed watermark extractor ({watermark_key.numel()} bits)", flush=True)

    hidden_baseline = None
    if args.hidden_ckpt is not None:
        hidden_baseline = load_hidden_baseline(args.hidden_ckpt, device)
        if watermark_key is None:
            watermark_key = torch.tensor(
                [int(bit) for bit in args.wm_key], device=device, dtype=torch.float32
            )
        if watermark_key.numel() != hidden_baseline.num_bits:
            raise ValueError(
                f"HiDDeN expects {hidden_baseline.num_bits} bits, got {watermark_key.numel()}"
            )
        print(f"[INFO] Loaded HiDDeN input-image baseline: {args.hidden_ckpt}", flush=True)

    scene_dirs = discover_scene_dirs(args.data_root)
    print(f"[INFO] Discovered {len(scene_dirs)} local DL3DV scenes", flush=True)
    for tag in wanted_tags(args.tags):
        run_tag(
            model,
            args,
            tag,
            device,
            scene_dirs,
            watermark_extractor,
            watermark_key,
            hidden_baseline,
        )


if __name__ == "__main__":
    main()
