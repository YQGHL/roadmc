"""RoadMC Data Loading Module  - Lightweight PyTorch Dataset + Lightning DataModule.

Supports:
- Synthetic .npz data loading
- Batched collation with padding for variable N
- Augmentation: random rotation, translation, scaling, occlusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SyntheticPointCloudDataset(Dataset):
    """PyTorch Dataset for synthetic road point cloud .npz files.

    Each .npz file contains one scene with points, labels, feats, normals.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_points: Optional[int] = 65536,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_points = max_points
        self.augment = augment and (split == "train")

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

        if self.max_points is not None and coords.shape[0] > self.max_points:
            idx_keep = torch.randperm(coords.shape[0])[:self.max_points]
            coords = coords[idx_keep]
            labels = labels[idx_keep]
            feats = feats[idx_keep]
            normals = normals[idx_keep]

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
    """Apply data augmentation: random rotation, translation, scaling."""
    angle = torch.rand(1).item() * 2 * torch.pi
    c, s = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
    rot_z = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    coords = coords @ rot_z.T
    normals = normals @ rot_z.T

    translation = torch.randn(3) * 0.2
    coords = coords + translation

    scale = 0.8 + torch.rand(1) * 0.4
    coords = coords * scale

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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Initialize datasets."""
        if stage in (None, "fit"):
            self.train_dataset = SyntheticPointCloudDataset(
                self.data_dir, "train", self.max_points, augment=True,
            )
            self.val_dataset = SyntheticPointCloudDataset(
                self.data_dir, "val", self.max_points, augment=False,
            )
        if stage in (None, "test"):
            self.test_dataset = SyntheticPointCloudDataset(
                self.data_dir, "val", self.max_points, augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_pointcloud_batch,
            num_workers=self.num_workers,
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
        for i in range(2):
            scene = ds.generate_scene(i)
            np.savez_compressed(
                os.path.join(tmpdir, f"train/scene_{i:04d}.npz"),
                **scene,
            )
        # Generate at least one val scene
        for i in range(2):
            scene = ds.generate_scene(i + 10)
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
