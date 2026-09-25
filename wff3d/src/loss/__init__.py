from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_depth_gt import LossDepthGT, LossDepthGTCfgWrapper
from .loss_lod import LossLOD, LossLODCfgWrapper
from .loss_depth_consis import LossDepthConsis, LossDepthConsisCfgWrapper
from .loss_normal_consis import LossNormalConsis, LossNormalConsisCfgWrapper
from .loss_chamfer_distance import LossChamferDistance, LossChamferDistanceCfgWrapper
LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossMseCfgWrapper: LossMse,
    LossLpipsCfgWrapper: LossLpips,
    LossOpacityCfgWrapper: LossOpacity,
    LossDepthGTCfgWrapper: LossDepthGT,
    LossLODCfgWrapper: LossLOD,
    LossDepthConsisCfgWrapper: LossDepthConsis,
    LossNormalConsisCfgWrapper: LossNormalConsis,
    LossChamferDistanceCfgWrapper: LossChamferDistance,
}

LossCfgWrapper = LossDepthCfgWrapper | LossMseCfgWrapper | LossLpipsCfgWrapper | LossOpacityCfgWrapper | LossDepthGTCfgWrapper | LossLODCfgWrapper | LossDepthConsisCfgWrapper | LossNormalConsisCfgWrapper | LossChamferDistanceCfgWrapper

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
