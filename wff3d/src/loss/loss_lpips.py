from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from src.dataset.types import BatchedExample
from src.misc.nn_module_tools import convert_to_buffer
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    conf: bool = False
    alpha: bool = False
    mask: bool = False


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = (batch["context"]["image"] + 1) / 2
        if global_step < self.cfg.apply_after_step:
            return torch.zeros((), dtype=torch.float32, device=image.device)

        if self.cfg.mask or self.cfg.alpha or self.cfg.conf:
            if self.cfg.mask:
                mask = batch["context"]["valid_mask"]
            elif self.cfg.alpha:
                mask = prediction.alpha
            else:
                mask = depth_dict["conf_valid_mask"]
            expanded_mask = mask.unsqueeze(2).expand_as(prediction.color)
            prediction_image = prediction.color * expanded_mask
            target_image = image * expanded_mask
        else:
            prediction_image = prediction.color
            target_image = image

        prediction_flat = rearrange(prediction_image, "b v c h w -> (b v) c h w")
        target_flat = rearrange(target_image, "b v c h w -> (b v) c h w")
        if self.training and prediction_flat.requires_grad:
            loss = checkpoint(
                lambda predicted, target: self.lpips.forward(
                    predicted, target, normalize=True
                ),
                prediction_flat,
                target_flat,
                use_reentrant=False,
            )
        else:
            loss = self.lpips.forward(prediction_flat, target_flat, normalize=True)
        return self.cfg.weight * torch.nan_to_num(
            loss.mean(), nan=0.0, posinf=0.0, neginf=0.0
        )
