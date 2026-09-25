from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLODCfg:
    mse_weight: float

@dataclass
class LossLODCfgWrapper:
    lod: LossLODCfg

WEIGHT_LEVEL_MAPPING = {0: 0.1, 1: 0.1, 2: 0.2, 3: 0.6}

class LossLOD(Loss[LossLODCfg, LossLODCfgWrapper]):
    def __init__(self, cfg: LossLODCfgWrapper) -> None:
        super().__init__(cfg)
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]
        # breakpoint()
        def mse_loss(x, y):
            delta = x - y
            return torch.nan_to_num((delta**2).mean().mean(), nan=0.0, posinf=0.0, neginf=0.0)
        # Before the specified step, don't apply the loss.
        lod_rendering = prediction.lod_rendering
        loss = 0.0
        for level in lod_rendering.keys():
            # level_weight
            # breakpoint()
            # if level != 3:
            #     continue
            rendered_imgs = lod_rendering[level]['rendered_imgs'].flatten(0, 1)
            _h, _w = rendered_imgs.shape[2:]
            resized_image = F.interpolate(image.clone().flatten(0, 1), size=(_h, _w), mode='bilinear', align_corners=False)
            level_mse_loss = mse_loss(rendered_imgs, resized_image)

            loss += torch.nan_to_num(level_mse_loss, nan=0.0, posinf=0.0, neginf=0.0) * self.cfg.mse_weight
        return loss / len(lod_rendering.keys())
