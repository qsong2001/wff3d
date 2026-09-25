from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .dataset_dl3dv_raw import DatasetDL3DVRaw, DatasetDL3DVRawCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
    "dl3dv_raw": DatasetDL3DVRaw,
}


DatasetCfg = DatasetRE10kCfg | DatasetDL3DVCfg | DatasetDL3DVRawCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
