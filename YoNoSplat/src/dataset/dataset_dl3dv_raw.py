import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon
from .norm_scale import compute_pose_norm_scale
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class DatasetDL3DVRawCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    pose_norm_method: str = "max_pairwise_d"


@dataclass
class DatasetDL3DVRawCfgWrapper:
    dl3dv: DatasetDL3DVRawCfg


class DatasetDL3DVRaw(IterableDataset):
    cfg: Any
    stage: Stage
    view_sampler: ViewSampler
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: Any,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.root = Path(cfg.roots[0])
        self.scenes = self._load_scene_keys()
        logger.info("DL3DV raw %s: %d scenes", self.stage, len(self.scenes))

    def _load_scene_keys(self) -> list[str]:
        if self.stage == "test" and hasattr(self.view_sampler, "index"):
            return [str(k) for k, v in self.view_sampler.index.items() if v is not None]

        index_path = self.root / f"{self.data_stage}_index.json"
        if index_path.exists():
            return [str(k) for k in json.loads(index_path.read_text())]

        keys: list[str] = []
        for subset_dir in sorted(self.root.glob("*K")):
            if not subset_dir.is_dir():
                continue
            for scene_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
                keys.append(f"{subset_dir.name}/{scene_dir.name}")
        return keys

    def __iter__(self):
        scenes = self.scenes
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            scenes = [
                scene
                for scene_index, scene in enumerate(scenes)
                if scene_index % worker_info.num_workers == worker_info.id
            ]

        for scene in scenes:
            try:
                yield self._load_example(scene)
            except Exception as exc:
                logger.warning("Skipped DL3DV raw scene %s: %s", scene, exc)

    def _load_example(self, scene: str) -> dict:
        scene_path = self.root / scene
        metadata_path = scene_path / "transforms.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = json.loads(metadata_path.read_text())
        intrinsics = self._convert_intrinsics(metadata)
        extrinsics, image_paths = self._convert_frames(scene_path, metadata)

        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise ValueError("field of view too wide")

        context_indices, target_indices, overlap = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
        )

        context_images = self._load_images([image_paths[i.item()] for i in context_indices])
        target_images = self._load_images([image_paths[i.item()] for i in target_indices])

        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise ValueError(
                f"bad image shape context={tuple(context_images.shape)} "
                f"target={tuple(target_images.shape)} expected=(V,3,{self.cfg.original_image_shape})"
            )

        context_extrinsics = extrinsics[context_indices]
        target_extrinsics = extrinsics[target_indices]
        num_context = len(context_indices)
        used_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)

        if self.cfg.make_baseline_1:
            scale = compute_pose_norm_scale(context_extrinsics, self.cfg.pose_norm_method)
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                raise ValueError(f"baseline out of range: {scale:.6f}")
            used_extrinsics[:, :3, 3] /= scale
        else:
            scale = torch.tensor(1.0, dtype=torch.float32)

        if self.cfg.relative_pose:
            used_extrinsics = camera_normalization(used_extrinsics[0:1], used_extrinsics)

        example = {
            "context": {
                "extrinsics": used_extrinsics[:num_context],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "overlap": overlap,
            },
            "target": {
                "extrinsics": used_extrinsics[num_context:],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": scene.replace("/", "_"),
        }
        return apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def _convert_intrinsics(self, metadata: dict) -> torch.Tensor:
        intrinsics = []
        for frame in metadata["frames"]:
            values = {**metadata, **frame}
            h, w = float(values["h"]), float(values["w"])
            intrinsic = torch.eye(3, dtype=torch.float32)
            intrinsic[0, 0] = float(values["fl_x"]) / w
            intrinsic[1, 1] = float(values["fl_y"]) / h
            intrinsic[0, 2] = float(values["cx"]) / w
            intrinsic[1, 2] = float(values["cy"]) / h
            intrinsics.append(intrinsic)
        return torch.stack(intrinsics)

    def _convert_frames(self, scene_path: Path, metadata: dict) -> tuple[torch.Tensor, list[Path]]:
        extrinsics = []
        image_paths: list[Path] = []
        for frame in metadata["frames"]:
            w2c = self._opengl_c2w_to_opencv_w2c(
                np.asarray(frame["transform_matrix"], dtype=np.float32)
            )
            extrinsics.append(torch.from_numpy(np.linalg.inv(w2c)).float())

            rel_path = Path(frame["file_path"])
            if rel_path.is_absolute():
                image_paths.append(rel_path)
            else:
                image_paths.append(scene_path / Path(str(rel_path).replace("images", "images_4")))
        return torch.stack(extrinsics), image_paths

    @staticmethod
    def _opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
        c2w = c2w.copy()
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        return np.linalg.inv(c2w)

    def _load_images(self, image_paths: list[Path]) -> torch.Tensor:
        images = []
        for image_path in image_paths:
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            images.append(self.to_tensor(Image.open(image_path).convert("RGB")))
        return torch.stack(images)

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> torch.Tensor:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    def __len__(self) -> int:
        return len(self.scenes)

    @property
    def test_len(self) -> int:
        return len(self.scenes)
