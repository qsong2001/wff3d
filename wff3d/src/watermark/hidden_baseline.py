from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor, nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class HiddenBaseline:
    encoder: nn.Module
    decoder: nn.Module
    scaling_i: float
    scaling_w: float
    scale_channels: bool
    num_bits: int

    def _stats(self, images: Tensor) -> tuple[Tensor, Tensor]:
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return mean, std

    @torch.no_grad()
    def encode(self, images: Tensor, message: Tensor, quantize: bool = True) -> Tensor:
        """Embed one {-1,+1} message per image into RGB images in [0, 1]."""
        mean, std = self._stats(images)
        normalized = (images - mean) / std
        delta = self.encoder(normalized, message)
        if self.scale_channels:
            scale = images.new_tensor(
                [(1 / 4.6) / 0.299, (1 / 4.6) / 0.587, (1 / 4.6) / 0.114]
            ).view(1, 3, 1, 1)
            delta = delta * scale
        watermarked = self.scaling_i * normalized + self.scaling_w * delta
        watermarked = (watermarked * std + mean).clamp(0, 1)
        if quantize:
            watermarked = torch.round(watermarked * 255.0) / 255.0
        return watermarked

    @torch.no_grad()
    def decode(self, images: Tensor) -> Tensor:
        mean, std = self._stats(images)
        return self.decoder((images - mean) / std)


def load_hidden_baseline(checkpoint: str | Path, device: torch.device) -> HiddenBaseline:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from GaussianMarker.hidden.models import HiddenDecoder, HiddenEncoder

    raw = torch.load(checkpoint, map_location="cpu")
    params = raw["params"]
    encoder = HiddenEncoder(
        num_blocks=int(params.encoder_depth),
        num_bits=int(params.num_bits),
        channels=int(params.encoder_channels),
        last_tanh=bool(params.use_tanh),
    )
    decoder = HiddenDecoder(
        num_blocks=int(params.decoder_depth),
        num_bits=int(params.num_bits) * int(params.redundancy),
        channels=int(params.decoder_channels),
    )
    state = {key.removeprefix("module."): value for key, value in raw["encoder_decoder"].items()}
    encoder.load_state_dict(
        {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")},
        strict=True,
    )
    decoder.load_state_dict(
        {key.removeprefix("decoder."): value for key, value in state.items() if key.startswith("decoder.")},
        strict=True,
    )
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return HiddenBaseline(
        encoder=encoder,
        decoder=decoder,
        scaling_i=float(params.scaling_i),
        scaling_w=float(params.scaling_w),
        scale_channels=bool(params.scale_channels),
        num_bits=int(params.num_bits),
    )


@torch.no_grad()
def hidden_watermark_batch(
    baseline: HiddenBaseline,
    images: Tensor,
    key: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return watermarked images and per-image bit accuracy."""
    flat = images.flatten(0, 1) if images.ndim == 5 else images
    key = key.to(device=flat.device, dtype=flat.dtype)
    bits_pm = (2.0 * key - 1.0).unsqueeze(0).expand(flat.shape[0], -1)
    watermarked = baseline.encode(flat, bits_pm)
    logits = baseline.decode(watermarked).view(flat.shape[0], baseline.num_bits, -1).sum(dim=-1)
    target = key.unsqueeze(0).expand_as(logits)
    accuracy = ((logits > 0).to(target.dtype) == target).float().mean(dim=-1)
    if images.ndim == 5:
        watermarked = watermarked.view_as(images)
    return watermarked, accuracy


@torch.no_grad()
def hidden_decode_batch(
    baseline: HiddenBaseline,
    images: Tensor,
    key: Tensor,
) -> Tensor:
    """Return per-image bit accuracy without modifying the rendered images."""
    flat = images.flatten(0, 1) if images.ndim == 5 else images
    key = key.to(device=flat.device, dtype=flat.dtype)
    logits = baseline.decode(flat).view(flat.shape[0], baseline.num_bits, -1).sum(dim=-1)
    target = key.unsqueeze(0).expand_as(logits)
    return ((logits > 0).to(target.dtype) == target).float().mean(dim=-1)
