from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


ATTACK_NAMES = ("jpeg", "crop", "noise", "blur", "resize", "color")


@dataclass(frozen=True)
class WatermarkAttackConfig:
    jpeg_quality: int = 50
    crop_ratio: float = 0.70
    noise_std: float = 0.05
    blur_sigma: float = 1.5
    resize_ratio: float = 0.50
    color_strength: float = 0.20

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if not 0 < self.crop_ratio <= 1:
            raise ValueError("crop_ratio must be in (0, 1]")
        if self.noise_std < 0 or self.blur_sigma <= 0:
            raise ValueError("noise_std must be non-negative and blur_sigma positive")
        if not 0 < self.resize_ratio <= 1:
            raise ValueError("resize_ratio must be in (0, 1]")
        if not 0 <= self.color_strength <= 1:
            raise ValueError("color_strength must be in [0, 1]")


@lru_cache(maxsize=1)
def _cpu_dct_matrix() -> Tensor:
    n = torch.arange(8, dtype=torch.float32)
    k = n[:, None]
    matrix = torch.cos(torch.pi * (2 * n + 1) * k / 16)
    matrix[0] *= 1 / torch.sqrt(torch.tensor(2.0))
    return matrix * 0.5


def _ste_round(value: Tensor) -> Tensor:
    return value + (value.round() - value).detach()


def _jpeg_quantization_tables(reference: Tensor, quality: int) -> Tensor:
    luminance = reference.new_tensor(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    chrominance = reference.new_tensor(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]
    )
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    tables = torch.stack((luminance, chrominance, chrominance))
    return ((tables * scale + 50) / 100).floor().clamp(1, 255)


def differentiable_jpeg(images: Tensor, quality: int) -> Tensor:
    """JPEG-like 8x8 DCT quantization with straight-through rounding."""
    n, _, height, width = images.shape
    rgb_to_ycbcr = images.new_tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ]
    )
    offset = images.new_tensor([0.0, 0.5, 0.5]).view(1, 3, 1, 1)
    ycbcr = torch.einsum("ij,njhw->nihw", rgb_to_ycbcr, images) + offset

    pad_h = (-height) % 8
    pad_w = (-width) % 8
    padded = F.pad(ycbcr, (0, pad_w, 0, pad_h), mode="replicate")
    padded_h, padded_w = padded.shape[-2:]
    blocks = F.unfold(padded, kernel_size=8, stride=8)
    blocks = blocks.view(n, 3, 8, 8, -1).permute(0, 1, 4, 2, 3)

    dct = _cpu_dct_matrix().to(device=images.device, dtype=images.dtype)
    coeff = torch.matmul(torch.matmul(dct, blocks * 255 - 128), dct.t())
    quant = _jpeg_quantization_tables(images, quality).view(1, 3, 1, 8, 8)
    coeff = _ste_round(coeff / quant) * quant
    restored = torch.matmul(torch.matmul(dct.t(), coeff), dct) + 128
    restored = restored.permute(0, 1, 3, 4, 2).reshape(n, 3 * 64, -1)
    restored = F.fold(restored, (padded_h, padded_w), kernel_size=8, stride=8) / 255
    restored = restored[..., :height, :width]

    ycbcr_to_rgb = images.new_tensor(
        [
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0],
        ]
    )
    restored = restored - offset
    return torch.einsum("ij,njhw->nihw", ycbcr_to_rgb, restored).clamp(0, 1)


def _crop_and_resize(images: Tensor, ratio: float, training: bool) -> Tensor:
    height, width = images.shape[-2:]
    crop_h = max(1, round(height * ratio))
    crop_w = max(1, round(width * ratio))
    if training:
        top = int(torch.randint(height - crop_h + 1, (), device=images.device))
        left = int(torch.randint(width - crop_w + 1, (), device=images.device))
    else:
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
    cropped = images[..., top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, (height, width), mode="bilinear", align_corners=False)


def _gaussian_blur(images: Tensor, sigma: float) -> Tensor:
    radius = max(1, round(2 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    blurred = F.conv2d(images, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(blurred, vertical, padding=(radius, 0), groups=channels)


def apply_image_attack(
    images: Tensor,
    attack: str,
    config: WatermarkAttackConfig,
    *,
    training: bool = False,
) -> Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [N, 3, H, W] images, got {tuple(images.shape)}")
    images = images.clamp(0, 1)
    if attack == "clean":
        return images
    if attack == "jpeg":
        return differentiable_jpeg(images, config.jpeg_quality)
    if attack == "crop":
        return _crop_and_resize(images, config.crop_ratio, training)
    if attack == "noise":
        return (images + torch.randn_like(images) * config.noise_std).clamp(0, 1)
    if attack == "blur":
        return _gaussian_blur(images, config.blur_sigma).clamp(0, 1)
    if attack == "resize":
        height, width = images.shape[-2:]
        resized_h = max(1, round(height * config.resize_ratio))
        resized_w = max(1, round(width * config.resize_ratio))
        reduced = F.interpolate(images, (resized_h, resized_w), mode="bilinear", align_corners=False)
        return F.interpolate(reduced, (height, width), mode="bilinear", align_corners=False)
    if attack == "color":
        strength = config.color_strength
        if training:
            contrast = 1 + (torch.rand((), device=images.device) * 2 - 1) * strength
            brightness = (torch.rand((), device=images.device) * 2 - 1) * strength * 0.5
        else:
            contrast = images.new_tensor(1 - strength)
            brightness = images.new_tensor(strength * 0.5)
        channel_mean = images.mean(dim=(-2, -1), keepdim=True)
        return ((images - channel_mean) * contrast + channel_mean + brightness).clamp(0, 1)
    raise ValueError(f"Unknown watermark attack: {attack}")


def sample_training_attack(device: torch.device) -> str:
    index = int(torch.randint(len(ATTACK_NAMES), (), device=device))
    return ATTACK_NAMES[index]
