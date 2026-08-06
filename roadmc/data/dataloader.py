"""RoadMC Data Loading Module  - Lightweight PyTorch Dataset + Lightning DataModule.

Supports:
- Synthetic .npz data loading
- Batched collation with padding for variable N
- Augmentation: random rotation, translation, scaling, occlusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from roadmc.data.curriculum import label_lut, normalize_label_stage
from roadmc.data.features import (
    DEFAULT_GEOMETRY_K_NEIGHBORS,
    compute_observable_features,
    has_observable_feature_schema,
)


class SyntheticPointCloudDataset(Dataset):
    """PyTorch Dataset for synthetic road point cloud .npz files.

    Each .npz file contains one scene with points, labels, feats, normals.

    Legacy files that predate the observable-feature schema are accepted, but
    their geometry channels are recomputed from coordinates and intensity so a
    historical label-derived feature can never enter a new training run.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_points: Optional[int] = 65536,
        augment: bool = False,
        binary: bool = False,
        label_stage: Optional[str] = None,
        recompute_legacy_features: bool = True,
        train_disease_ratio: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_points = max_points
        self.augment = augment and (split == "train")
        if label_stage is None:
            label_stage = "binary" if binary else "full38"
        self.label_stage = normalize_label_stage(label_stage)
        if binary and self.label_stage != "binary":
            raise ValueError("binary=True cannot be combined with a non-binary label_stage")
        self.binary = self.label_stage == "binary"
        self._label_lut = torch.tensor(label_lut(self.label_stage), dtype=torch.long)
        self.recompute_legacy_features = recompute_legacy_features
        self._legacy_feature_warning_emitted = False
        if not 0.0 < train_disease_ratio <= 1.0:
            raise ValueError(f"train_disease_ratio must be in (0, 1], got {train_disease_ratio}")
        self.train_disease_ratio = train_disease_ratio

        self.files = sorted((self.data_dir / split).glob("scene_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {self.data_dir / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single scene."""
        data = np.load(self.files[idx], allow_pickle=True)

        coords = torch.from_numpy(data["points"]).float()
        labels = torch.from_numpy(data["labels"]).long()
        feats = torch.from_numpy(data["feats"]).float()
        normals = torch.from_numpy(data["normals"]).float()
        if feats.ndim != 2 or feats.shape[0] != coords.shape[0] or feats.shape[1] < 1:
            raise ValueError(
                f"{self.files[idx]} has invalid feats shape {tuple(feats.shape)}; expected (N, >=1)"
            )
        intensities = feats[:, 0].clone()
        has_current_schema = (
            "feature_schema" in data.files
            and has_observable_feature_schema(data["feature_schema"])
        )
        recompute_features = self.recompute_legacy_features and not has_current_schema

        if self.max_points is not None and coords.shape[0] > self.max_points:
            n_keep = min(coords.shape[0], self.max_points)
            if self.split == "train":
                # Training can deliberately increase exposure to rare damage,
                # but evaluation must retain the source prevalence below.
                disease_mask = labels > 0
                disease_indices = torch.where(disease_mask)[0]
                bg_indices = torch.where(~disease_mask)[0]
                n_disease = len(disease_indices)
                n_bg_total = len(bg_indices)

                # 病害点封顶比例：大面病害场景（raw_surface 原生密度下
                # slab shatter 等病害点可达 57%）若全部保留，max_points
                # 采样后整张场景挤进病害区域，单窗口占用退化并触发
                # window-attention 偏置显存守卫。封顶保证训练子集里
                # 病害点占比不超过 train_disease_ratio，同时保留背景点
                # 把空间分布撑开。v2（病害 ~2%）不受影响；v3 在 16384
                # 点下自然低于封顶，在低 max_points 下同样被保护。
                max_disease_keep = min(n_disease, int(n_keep * self.train_disease_ratio))
                if n_keep - max_disease_keep > n_bg_total:
                    # 背景不足时回退：不能取超过背景总量的背景点。
                    max_disease_keep = max(0, n_keep - n_bg_total)

                if max_disease_keep > 0:
                    disease_selected = disease_indices[
                        torch.randperm(n_disease)[:max_disease_keep]
                    ]
                    n_bg = n_keep - max_disease_keep
                    if n_bg > 0:
                        bg_selected = bg_indices[torch.randperm(n_bg_total)[:n_bg]]
                        idx_keep = torch.cat([disease_selected, bg_selected])
                    else:
                        idx_keep = disease_selected
                else:
                    idx_keep = torch.randperm(coords.shape[0])[:n_keep]
            else:
                # A fixed uniform sample makes validation/test metrics
                # reproducible and leaves the class prevalence unbiased.
                generator = torch.Generator().manual_seed(42 + idx)
                idx_keep = torch.randperm(coords.shape[0], generator=generator)[:n_keep]

            coords = coords[idx_keep]
            labels = labels[idx_keep]
            feats = feats[idx_keep]
            normals = normals[idx_keep]
            intensities = intensities[idx_keep]
            # A new point subset changes every k-neighborhood.  Recompute from
            # the exact cloud passed to the model, even for current-schema data.
            recompute_features = True

        if recompute_features:
            feats = torch.from_numpy(
                compute_observable_features(
                    coords.numpy(),
                    intensities.numpy(),
                    k_neighbors=DEFAULT_GEOMETRY_K_NEIGHBORS,
                )
            )
            if not has_current_schema and not self._legacy_feature_warning_emitted:
                warnings.warn(
                    "Legacy RoadMC feature schema detected; recomputing observable geometry "
                    "features to exclude historical privileged channels.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._legacy_feature_warning_emitted = True

        if self.label_stage != "full38":
            labels = self._label_lut[labels]

        if self.augment:
            coords, normals = _augment_point_cloud(coords, normals)

        return {
            "coords": coords,
            "feats": feats,
            "labels": labels,
            "normals": normals,
        }


def _augment_point_cloud(
    coords: torch.Tensor,
    normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply feature-preserving augmentation: z-rotation + translation.

    只使用保持可观测特征不变的增广群：绕 z 旋转（两个几何特征均
    旋转不变）与平移。各向同性缩放会改变 signed_local_height_residual
    的物理量纲而 feats 未重算——违反"特征必须对应喂给模型的那个
    点云"的契约，故移除。
    """
    angle = torch.rand(1).item() * 2 * torch.pi
    c, s = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
    rot_z = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    coords = coords @ rot_z.T
    normals = normals @ rot_z.T

    translation = torch.randn(3) * 0.2
    coords = coords + translation

    return coords, normals


def collate_pointcloud_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function that pads variable-length point clouds to same N.

    Uses the maximum N in this batch as target, pads smaller scenes
    with zeros (mask tracks valid).
    """
    max_N = max(item["coords"].shape[0] for item in batch)

    batch_dict = {}
    for key in batch[0]:
        tensors = []
        for item in batch:
            t = item[key]
            if t.shape[0] < max_N:
                pad_size = max_N - t.shape[0]
                # P0-2: pad labels with -1 (ignored in loss), others with 0
                if t.ndim == 2:
                    pad = (0, 0, 0, pad_size)
                else:
                    pad = (0, pad_size)
                pad_value = -1 if key == "labels" else 0
                t = torch.nn.functional.pad(t, pad) if pad_value == 0 else \
                    torch.nn.functional.pad(t, pad, value=pad_value)
            tensors.append(t)
        batch_dict[key] = torch.stack(tensors)
    
    # Add valid_mask: 1 for real points, 0 for padded
    valid_mask = []
    for item in batch:
        N = item["labels"].shape[0]
        vm = torch.ones(max_N, dtype=torch.bool)
        if N < max_N:
            vm[N:] = False
        valid_mask.append(vm)
    batch_dict["valid_mask"] = torch.stack(valid_mask)
    
    return batch_dict


class RoadMCDataModule(pl.LightningDataModule):
    """Lightning DataModule for RoadMC point cloud data."""

    def __init__(
        self,
        data_dir: str = "./data/synthetic_output",
        batch_size: int = 4,
        max_points: int = 65536,
        num_workers: int = 0,
        binary: bool = False,
        label_stage: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.num_workers = num_workers
        if label_stage is None:
            label_stage = "binary" if binary else "full38"
        self.label_stage = normalize_label_stage(label_stage)
        if binary and self.label_stage != "binary":
            raise ValueError("binary=True cannot be combined with a non-binary label_stage")
        self.binary = self.label_stage == "binary"

    def setup(self, stage: Optional[str] = None):
        """Initialize datasets."""
        if stage in (None, "fit"):
            self.train_dataset = SyntheticPointCloudDataset(
                self.data_dir, "train", self.max_points, augment=True,
                binary=self.binary, label_stage=self.label_stage,
            )
            self.val_dataset = SyntheticPointCloudDataset(
                self.data_dir, "val", self.max_points, augment=False,
                binary=self.binary, label_stage=self.label_stage,
            )
        if stage == "test":
            # 独立 test split：目录缺失时直接报错，绝不静默回退到 val
            # ——回退会让"独立测试集"的声明在无人察觉的情况下失效。
            # （stage=None 只建 fit 数据集，旧的双目录数据集仍可训练。）
            self.test_dataset = SyntheticPointCloudDataset(
                self.data_dir, "test", self.max_points, augment=False,
                binary=self.binary, label_stage=self.label_stage,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == '__main__':
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # Create a small synthetic dataset for testing
    from roadmc.data.synthetic.config import GeneratorConfig, RoadSurfaceConfig, DiseaseConfig
    from roadmc.data.synthetic.generator import SyntheticRoadDataset
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate 2 test scenes
        cfg = GeneratorConfig(
            road=RoadSurfaceConfig(width=3.0, length=3.0, grid_res=0.03),
            disease=DiseaseConfig(max_diseases_per_scene=1),
            seed=42, num_points=1024, normalize=True,
        )
        ds = SyntheticRoadDataset(cfg, dataset_size=2)

        os.makedirs(os.path.join(tmpdir, "train"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "val"), exist_ok=True)
        import json

        def _npz_safe(scene: dict) -> dict:
            # resolution_metadata is a nested dict; savez would pickle it into
            # an object array that np.load(allow_pickle=False) refuses to read.
            scene = dict(scene)
            scene["resolution_metadata_json"] = json.dumps(
                scene.pop("resolution_metadata"), ensure_ascii=True, sort_keys=True
            )
            return scene

        for i in range(2):
            scene = _npz_safe(ds.generate_scene(i))
            np.savez_compressed(
                os.path.join(tmpdir, f"train/scene_{i:04d}.npz"),
                **scene,
            )
        # Generate at least one val scene
        for i in range(2):
            scene = _npz_safe(ds.generate_scene(i + 10))
            np.savez_compressed(
                os.path.join(tmpdir, f"val/scene_{i:04d}.npz"),
                **scene,
            )

        # Test loading
        dataset = SyntheticPointCloudDataset(tmpdir, "train", max_points=1024)
        sample = dataset[0]
        assert sample["coords"].shape[-1] == 3
        assert sample["labels"].ndim == 1
        print(f"Dataset loaded: {len(dataset)} files, sample coords={sample['coords'].shape}")

        # Test DataModule
        dm = RoadMCDataModule(data_dir=tmpdir, batch_size=2, max_points=1024)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["coords"].shape[0] == 2, f"Batch size wrong: {batch['coords'].shape}"
        print(f"DataModule: batch coords={batch['coords'].shape}, labels={batch['labels'].shape}")

    print("All DataLoader tests passed.")
