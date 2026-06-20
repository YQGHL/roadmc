"""Real road point cloud dataset loader with JTG 5210-2018 label mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RealRoadDataset(Dataset):
    """Load real road point clouds from `.npy` or `.ply` files.

    The loader returns the same keys as the synthetic dataset pipeline:
    `coords`, `labels`, `feats`, `normals`, and `intensities`.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_pattern: str = "*.npy",
        max_points: Optional[int] = 65536,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_pattern))
        self.max_points = max_points
        self.normalize = normalize

        if not self.files:
            raise FileNotFoundError(f"No {file_pattern} files in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        points, labels, normals, intensities = self._load_file(self.files[idx])

        if self.max_points is not None and points.shape[0] > self.max_points:
            idx_keep = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idx_keep]
            labels = labels[idx_keep] if labels is not None else None
            normals = normals[idx_keep] if normals is not None else None
            intensities = intensities[idx_keep] if intensities is not None else None

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_radius = np.max(np.linalg.norm(points, axis=1))
            if max_radius > 1e-12:
                points = points / max_radius

        feats = np.stack(
            [
                intensities if intensities is not None else np.zeros(points.shape[0], dtype=np.float32),
                np.zeros(points.shape[0], dtype=np.float32),
                np.zeros(points.shape[0], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        return {
            "coords": torch.from_numpy(points).float(),
            "labels": torch.from_numpy(labels).long() if labels is not None else torch.zeros(points.shape[0], dtype=torch.long),
            "normals": torch.from_numpy(normals).float() if normals is not None else torch.zeros(points.shape[0], 3, dtype=torch.float32),
            "intensities": torch.from_numpy(intensities).float() if intensities is not None else torch.zeros(points.shape[0], dtype=torch.float32),
            "feats": torch.from_numpy(feats).float(),
        }

    @staticmethod
    def _load_file(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        ext = filepath.suffix.lower()

        if ext == ".npy":
            data = np.load(filepath, allow_pickle=True)
            if data.dtype.names:
                points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
                labels = None
                for name in ("label", "labels", "class", "semantic"):
                    if name in data.dtype.names:
                        labels = data[name].astype(np.int64)
                        break
                normals = None
                if {"nx", "ny", "nz"}.issubset(data.dtype.names):
                    normals = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(np.float32)
                intensities = None
                for name in ("intensity", "reflectance"):
                    if name in data.dtype.names:
                        intensities = data[name].astype(np.float32)
                        break
                return points, labels, normals, intensities

            arr = np.asarray(data)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"Unsupported .npy array shape: {arr.shape}")
            points = arr[:, :3].astype(np.float32)
            labels = arr[:, 3].astype(np.int64) if arr.shape[1] > 3 else None
            normals = arr[:, 4:7].astype(np.float32) if arr.shape[1] >= 7 else None
            intensities = arr[:, 7].astype(np.float32) if arr.shape[1] > 7 else None
            return points, labels, normals, intensities

        if ext == ".ply":
            raise ValueError("PLY loading is not implemented in this build; use structured .npy files.")

        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def map_to_jtg(labels_source: np.ndarray, mapping: Optional[Dict[int, int]] = None) -> np.ndarray:
        if mapping is None:
            return labels_source
        mapped = np.zeros_like(labels_source)
        for src, jtg in mapping.items():
            mapped[labels_source == src] = jtg
        return mapped
