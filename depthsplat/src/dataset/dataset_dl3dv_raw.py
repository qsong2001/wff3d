import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDL3DVRawCfg(DatasetCfgCommon):
    name: Literal["dl3dv_raw"]
    roots: list[Path]
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    skip_bad_shape: bool = True
    near: float = 0.5
    far: float = 200.0
    shuffle_val: bool = True
    sort_target_index: bool = True
    sort_context_index: bool = True
    image_dir_name: str = "images_4"


class DatasetDL3DVRaw(IterableDataset):
    cfg: DatasetDL3DVRawCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetDL3DVRawCfg,
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
        if self.stage == "test":
            self.scenes = self.scenes[:: cfg.test_chunk_interval]

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
            yield self._load_example(scene)

    def _load_example(self, scene: str) -> dict:
        scene_path = self.root / scene
        metadata_path = scene_path / "transforms.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = json.loads(metadata_path.read_text())
        intrinsics = self._convert_intrinsics(metadata)
        extrinsics, image_paths = self._convert_frames(scene_path, metadata)

        context_indices, target_indices = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
        )
        if self.cfg.sort_context_index:
            context_indices = context_indices.sort()[0]
        if self.cfg.sort_target_index:
            target_indices = target_indices.sort()[0]

        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise ValueError("field of view too wide")
        if self.cfg.make_baseline_1:
            raise ValueError("make_baseline_1 is not supported by the raw DL3DV loader")

        context_images = self._load_images([image_paths[i.item()] for i in context_indices])
        target_images = self._load_images([image_paths[i.item()] for i in target_indices])

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
            },
            "scene": scene.replace("/", "_"),
        }

        if self.cfg.image_shape == list(context_images.shape[2:]):
            return example
        return apply_crop_shim(example, tuple(self.cfg.image_shape))

    def _convert_intrinsics(self, metadata: dict) -> Float[Tensor, "view 3 3"]:
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

    def _convert_frames(self, scene_path: Path, metadata: dict) -> tuple[Tensor, list[Path]]:
        extrinsics = []
        image_paths: list[Path] = []
        for frame in metadata["frames"]:
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            extrinsics.append(torch.from_numpy(self._opengl_c2w_to_opencv_c2w(c2w)).float())

            rel_path = Path(frame["file_path"])
            if rel_path.is_absolute():
                image_paths.append(rel_path)
            else:
                image_paths.append(
                    scene_path / Path(str(rel_path).replace("images", self.cfg.image_dir_name))
                )
        return torch.stack(extrinsics), image_paths

    @staticmethod
    def _opengl_c2w_to_opencv_c2w(c2w: np.ndarray) -> np.ndarray:
        c2w = c2w.copy()
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        return c2w

    def _load_images(self, image_paths: list[Path]) -> Tensor:
        images = []
        for image_path in image_paths:
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            images.append(self.to_tensor(Image.open(image_path).convert("RGB")))
        return torch.stack(images)

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Tensor:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
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
