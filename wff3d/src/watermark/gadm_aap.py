from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class GADMAAPConfig:
    gadm: bool = False
    aap: bool = False
    lambda_gadm: float = 0.05
    lambda_aap: float = 1.0
    prune_ratio_min: float = 0.05
    prune_ratio_max: float = 0.30
    aap_num_views: int = 2
    reliability_chunk_size: int = 65536
    gadm_carrier: str = "appearance"
    gadm_weighting: str = "reliability"
    gadm_strict_carrier: bool = False
    aap_strategy: str = "adaptive"

    def __post_init__(self) -> None:
        if not 0 <= self.prune_ratio_min <= self.prune_ratio_max < 1:
            raise ValueError("AAP pruning ratios must satisfy 0 <= min <= max < 1")
        if self.lambda_gadm < 0 or self.lambda_aap < 0:
            raise ValueError("GADM/AAP loss weights must be non-negative")
        if self.aap_num_views < 1:
            raise ValueError("AAP must supervise at least one rendered view")
        if self.gadm_carrier not in {"appearance", "opacity", "geometry", "all"}:
            raise ValueError(f"Unsupported GADM carrier: {self.gadm_carrier}")
        if self.gadm_weighting not in {"reliability", "uniform"}:
            raise ValueError(f"Unsupported GADM weighting: {self.gadm_weighting}")
        if self.aap_strategy not in {"adaptive", "random", "opacity"}:
            raise ValueError(f"Unsupported AAP strategy: {self.aap_strategy}")


@torch.no_grad()
def compute_gaussian_reliability(
    gaussians,
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Tensor | None = None,
    far: Tensor | None = None,
    chunk_size: int = 65536,
) -> Tensor:
    """Estimate detached per-Gaussian reliability from camera visibility and footprint.

    Extrinsics are camera-to-world matrices and intrinsics are normalized to image size.
    The computation is chunked because feed-forward predictors can emit over 500K
    Gaussians per scene.
    """
    means = gaussians.means.detach().float()
    covariances = gaussians.covariances.detach().float()
    opacities = gaussians.opacities.detach().float().clamp(0, 1)
    extrinsics = extrinsics.detach().float()
    intrinsics = intrinsics.detach().float()
    b, n, _ = means.shape
    _, v, _, _ = extrinsics.shape

    camera_centers = extrinsics[..., :3, 3]
    world_to_camera_rotation = extrinsics[..., :3, :3].transpose(-1, -2)

    sample = means[:, : min(n, 4096)]
    sample_offset = sample[:, None] - camera_centers[:, :, None]
    sample_camera = torch.einsum("bvij,bvnj->bvni", world_to_camera_rotation, sample_offset)
    positive_fraction = (sample_camera[..., 2] > 0).float().mean(dim=(1, 2))
    forward_sign = torch.where(positive_fraction >= 0.5, 1.0, -1.0).view(b, 1, 1)

    if near is None:
        near = torch.zeros((b, v), device=means.device, dtype=means.dtype)
    if far is None:
        far = torch.full((b, v), float("inf"), device=means.device, dtype=means.dtype)
    near = near.detach().float().view(b, v, 1)
    far = far.detach().float().view(b, v, 1)

    fx = intrinsics[..., 0, 0].view(b, v, 1)
    fy = intrinsics[..., 1, 1].view(b, v, 1)
    cx = intrinsics[..., 0, 2].view(b, v, 1)
    cy = intrinsics[..., 1, 2].view(b, v, 1)
    focal = 0.5 * (fx.abs() + fy.abs())

    reliability_parts = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xyz = means[:, start:end]
        offset = xyz[:, None] - camera_centers[:, :, None]
        camera_xyz = torch.einsum("bvij,bvnj->bvni", world_to_camera_rotation, offset)
        depth = camera_xyz[..., 2] * forward_sign
        safe_depth = depth.clamp_min(1e-6)
        u = fx * camera_xyz[..., 0] / safe_depth + cx
        vv = fy * camera_xyz[..., 1] / safe_depth + cy
        visible = (
            (depth > near)
            & (depth < far)
            & (u >= 0)
            & (u <= 1)
            & (vv >= 0)
            & (vv <= 1)
        )
        visibility_frequency = visible.float().mean(dim=1)

        covariance = covariances[:, start:end]
        scale_proxy = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).mean(dim=-1).sqrt()
        projected_radius = focal * scale_proxy[:, None] / safe_depth
        projected_contribution = (
            projected_radius.square().clamp(max=1.0) * visible.float()
        ).mean(dim=1)
        raw = (
            visibility_frequency
            * opacities[:, start:end]
            * projected_contribution
        )
        reliability_parts.append(raw)

    raw_reliability = torch.cat(reliability_parts, dim=1)
    scale = torch.quantile(raw_reliability, 0.95, dim=1, keepdim=True).clamp_min(1e-8)
    reliability = (raw_reliability / scale).clamp(0, 1)

    # Degenerate camera metadata should not silently disable the objective.
    fallback = opacities / opacities.amax(dim=1, keepdim=True).clamp_min(1e-8)
    valid = (raw_reliability.amax(dim=1, keepdim=True) > 0)
    return torch.where(valid, reliability, fallback).detach()


def _per_gaussian_delta(watermarked: Tensor, clean: Tensor) -> Tensor:
    if watermarked.shape != clean.shape:
        raise RuntimeError(
            f"GADM requires aligned Gaussian sets, got {tuple(watermarked.shape)} "
            f"and {tuple(clean.shape)}"
        )
    reduce_dims = tuple(range(2, watermarked.ndim))
    return (watermarked - clean.detach()).square().mean(dim=reduce_dims)


def gadm_loss(
    watermarked_gaussians,
    clean_gaussians,
    reliability: Tensor,
    carrier: str = "appearance",
    weighting: str = "reliability",
    strict_carrier: bool = False,
) -> Tensor:
    """Concentrate parameter changes in reliable Gaussians and selected attributes.

    The default arguments reproduce the original appearance-only GADM objective.
    ``strict_carrier`` additionally penalizes changes in all non-carrier attributes,
    which is used only by the carrier ablation.
    """
    attributes = {
        "appearance": (watermarked_gaussians.harmonics, clean_gaussians.harmonics),
        "opacity": (watermarked_gaussians.opacities, clean_gaussians.opacities),
        "means": (watermarked_gaussians.means, clean_gaussians.means),
        "covariances": (watermarked_gaussians.covariances, clean_gaussians.covariances),
    }
    carrier_attributes = {
        "appearance": {"appearance"},
        "opacity": {"opacity"},
        "geometry": {"means", "covariances"},
        "all": set(attributes),
    }[carrier]
    selected_weight = 1.0 - reliability if weighting == "reliability" else torch.ones_like(reliability)
    selected_losses = []
    noncarrier_losses = []
    for name, (watermarked, clean) in attributes.items():
        delta = _per_gaussian_delta(watermarked, clean)
        if name in carrier_attributes:
            selected_losses.append((selected_weight * delta).mean())
        elif strict_carrier:
            noncarrier_losses.append(delta.mean())

    loss = torch.stack(selected_losses).mean()
    if noncarrier_losses:
        loss = loss + torch.stack(noncarrier_losses).mean()
    return loss


@torch.no_grad()
def sample_prune_ratio(config: GADMAAPConfig, device: torch.device) -> Tensor:
    return torch.empty((), device=device).uniform_(
        config.prune_ratio_min,
        config.prune_ratio_max,
    )


def adaptive_prune_gaussians(gaussians, reliability: Tensor, prune_ratio: Tensor):
    """Zero opacity for the least reliable scene-specific Gaussian fraction."""
    masks = []
    ratio = float(prune_ratio.detach().cpu())
    for scene_reliability in reliability:
        keep_count = max(1, round(scene_reliability.numel() * (1.0 - ratio)))
        keep_indices = torch.topk(scene_reliability, keep_count, sorted=False).indices
        keep_mask = torch.zeros_like(scene_reliability, dtype=torch.bool)
        keep_mask[keep_indices] = True
        masks.append(keep_mask)
    keep_mask = torch.stack(masks, dim=0).detach()
    return replace(gaussians, opacities=gaussians.opacities * keep_mask.to(gaussians.opacities.dtype)), keep_mask


def prune_gaussians(
    gaussians,
    reliability: Tensor,
    prune_ratio: Tensor,
    strategy: str = "adaptive",
):
    """Apply an exact-ratio AAP mask using the requested ablation strategy."""
    if strategy == "adaptive":
        return adaptive_prune_gaussians(gaussians, reliability, prune_ratio)

    ratio = float(prune_ratio.detach().cpu())
    masks = []
    for scene_index, scene_reliability in enumerate(reliability):
        keep_count = max(1, round(scene_reliability.numel() * (1.0 - ratio)))
        if strategy == "random":
            keep_indices = torch.randperm(
                scene_reliability.numel(), device=scene_reliability.device
            )[:keep_count]
        elif strategy == "opacity":
            opacity = gaussians.opacities[scene_index].detach()
            keep_indices = torch.topk(opacity, keep_count, sorted=False).indices
        else:
            raise ValueError(f"Unsupported AAP strategy: {strategy}")
        keep_mask = torch.zeros_like(scene_reliability, dtype=torch.bool)
        keep_mask[keep_indices] = True
        masks.append(keep_mask)
    keep_mask = torch.stack(masks, dim=0).detach()
    pruned = replace(
        gaussians,
        opacities=gaussians.opacities * keep_mask.to(gaussians.opacities.dtype),
    )
    return pruned, keep_mask
