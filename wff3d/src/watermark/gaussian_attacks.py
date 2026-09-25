from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


GAUSSIAN_ATTACK_NAMES = (
    "prune30",
    "random_prune30",
    "quant8",
    "position_noise",
    "opacity_noise",
    "sh_noise",
)


@dataclass(frozen=True)
class GaussianAttackConfig:
    prune_ratio: float = 0.30
    random_prune_ratio: float = 0.30
    quant_bits: int = 8
    position_noise_std: float = 0.005
    opacity_noise_std: float = 0.10
    sh_noise_std: float = 0.05


def _quantize_per_scene(values: Tensor, bits: int) -> Tensor:
    reduce_dims = tuple(range(1, values.ndim))
    minimum = values.amin(dim=reduce_dims, keepdim=True)
    maximum = values.amax(dim=reduce_dims, keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8)
    levels = (1 << bits) - 1
    return ((values - minimum) / scale * levels).round() / levels * scale + minimum


def prune_low_opacity(gaussians, ratio: float = 0.30):
    if not 0 <= ratio < 1:
        raise ValueError("Pruning ratio must be in [0, 1)")
    opacities = gaussians.opacities
    keep_count = max(1, round(opacities.shape[1] * (1.0 - ratio)))
    keep_indices = torch.topk(opacities, keep_count, dim=1, sorted=False).indices
    keep_mask = torch.zeros_like(opacities, dtype=torch.bool)
    keep_mask.scatter_(1, keep_indices, True)
    return replace(gaussians, opacities=opacities * keep_mask.to(opacities.dtype))


def quantize_appearance(gaussians, bits: int = 8):
    """Quantize the stored opacity and SH appearance attributes per scene."""
    return replace(
        gaussians,
        opacities=_quantize_per_scene(gaussians.opacities, bits).clamp(0, 1),
        harmonics=_quantize_per_scene(gaussians.harmonics, bits),
    )


def _deterministic_randn_like(values: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    return torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator)


def random_prune(gaussians, ratio: float = 0.30):
    generator = torch.Generator(device=gaussians.opacities.device).manual_seed(0)
    scores = torch.rand(
        gaussians.opacities.shape,
        device=gaussians.opacities.device,
        dtype=gaussians.opacities.dtype,
        generator=generator,
    )
    keep_count = max(1, round(scores.shape[1] * (1.0 - ratio)))
    keep_indices = torch.topk(scores, keep_count, dim=1, sorted=False).indices
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    keep_mask.scatter_(1, keep_indices, True)
    return replace(
        gaussians,
        opacities=gaussians.opacities * keep_mask.to(gaussians.opacities.dtype),
    )


def perturb_positions(gaussians, relative_std: float = 0.005):
    extent = (gaussians.means.amax(dim=1) - gaussians.means.amin(dim=1)).mean(
        dim=-1, keepdim=True
    )
    noise = _deterministic_randn_like(gaussians.means, seed=1)
    return replace(gaussians, means=gaussians.means + noise * extent.unsqueeze(-1) * relative_std)


def perturb_opacity(gaussians, relative_std: float = 0.10):
    noise = _deterministic_randn_like(gaussians.opacities, seed=2)
    opacities = (gaussians.opacities * (1 + noise * relative_std)).clamp(0, 1)
    return replace(gaussians, opacities=opacities)


def corrupt_harmonics(gaussians, relative_std: float = 0.05):
    reduce_dims = tuple(range(1, gaussians.harmonics.ndim))
    scale = gaussians.harmonics.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    noise = _deterministic_randn_like(gaussians.harmonics, seed=3)
    return replace(gaussians, harmonics=gaussians.harmonics + noise * scale * relative_std)


def apply_gaussian_attack(
    gaussians,
    attack: str,
    config: GaussianAttackConfig | None = None,
):
    config = config or GaussianAttackConfig()
    if attack == "prune30":
        return prune_low_opacity(gaussians, ratio=config.prune_ratio)
    if attack == "random_prune30":
        return random_prune(gaussians, ratio=config.random_prune_ratio)
    if attack == "quant8":
        return quantize_appearance(gaussians, bits=config.quant_bits)
    if attack == "position_noise":
        return perturb_positions(gaussians, relative_std=config.position_noise_std)
    if attack == "opacity_noise":
        return perturb_opacity(gaussians, relative_std=config.opacity_noise_std)
    if attack == "sh_noise":
        return corrupt_harmonics(gaussians, relative_std=config.sh_noise_std)
    raise ValueError(f"Unknown Gaussian attack: {attack}")
