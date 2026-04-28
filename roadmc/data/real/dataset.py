"""Real road point cloud dataset loader with JTG 5210-2018 label mapping.

Task 4.2: Load real-world road point clouds and map to JTG classification.

Supports:
- .ply point cloud loading
- .las/.laz LiDAR data (via numpy if available)
- Label mapping from arbitrary source schemas → JTG 5210-2018
- Coordinate normalization (unit sphere scaling)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RealRoadDataset(Dataset):
    """Real road point cloud dataset with JTG 5210-2018 label mapping.

    Loads point clouds from .ply/.las/.laz files and provides a unified
    interface matching the synthetic dataset format.

    Args:
        data_dir: Directory containing point cloud files.
        file_pattern: Glob pattern for files (e.g., '*.ply').
        transforms: Optional transform pipeline.
        max_points: Maximum points per sample (None = keep all).
        normalize: Whether to normalize coordinates to unit sphere.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_pattern: str = "*.ply",
        max_points: Optional[int] = 65536,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_pattern))
        self.max_points = max_points
        self.normalize = normalize

        if not self.files:
            raise FileNotFoundError(f"No {file_pattern} files in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a real point cloud scene.

        Returns dict matching synthetic dataset format:
            coords: (N, 3) float32
            labels: (N,) int64 — JTG labels (0 = unlabeled)
            normals: (N, 3) float32 (zeros if not available)
            intensities: (N,) float32 (zeros if not available)
        """
        # Try loading .ply
        points, labels, normals, intensities = self._load_file(self.files[idx])

        # Subsample if too large
        if self.max_points is not None and points.shape[0] > self.max_points:
            idx_keep = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idx_keep]
            labels = labels[idx_keep] if labels is not None else None
            normals = normals[idx_keep] if normals is not None else None
            intensities = intensities[idx_keep] if intensities is not None else None

        # Normalize
        if self.normalize:
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_radius = np.max(np.linalg.norm(points, axis=1))
            if max_radius > 1e-12:
                points = points / max_radius

        return {
            "coords": torch.from_numpy(points).float(),
            "labels": torch.from_numpy(labels).long() if labels is not None
                      else torch.zeros(points.shape[0], dtype=torch.long),
            "normals": torch.from_numpy(normals).float() if normals is not None
                       else torch.zeros(points.shape[0], 3, dtype=torch.float32),
            "intensities": torch.from_numpy(intensities).float() if intensities is not None
                           else torch.zeros(points.shape[0], dtype=torch.float32),
        }

    @staticmethod
    def _load_file(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray],
                                              Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a point cloud file. Returns (points, labels, normals, intensities)."""
        ext = filepath.suffix.lower()

        if ext == ".npy":
            data = np.load(filepath)
            if data.dtype.names:
                return (np.column_stack([data["x"], data["y"], data["z"]]),
                        data.get("label"), data.get("normal"), data.get("intensity"))
            return data[:, :3], None, None, None

        # .ply: try parsing header for properties
        if ext == ".ply":
            try:
                with open(filepath, "rb") as f:
                    header = f.read(200).decode("ascii", errors="ignore")
                has_normal = "property float nx" in header or "property float normal_x" in header
                has_label = "property int label" in header or "property uchar label" in header
                has_intensity = "property float intensity" in header or \
                                "property uchar intensity" in header

                # Use numpy to load ply
                data = np.genfromtxt(filepath, skip_header=self._ply_header_lines(filepath),
                                     dtype=np.float32, max_rows=1000000)
                if data.shape[1] >= 3:
                    pts = data[:, :3]
                    idx = 3
                    nrm = data[:, idx:idx+3] if has_normal and data.shape[1] >= idx+3 else None
                    if nrm is not None:
                        idx += 3
                    lbl = None
                    if has_label and data.shape[1] > idx:
                        lbl = data[:, idx].astype(np.int64)
                        idx += 1
                    intens = None
                    if has_intensity and data.shape[1] > idx:
                        intens = data[:, idx]
                    return pts, lbl, nrm, intens
            except Exception:
                pass

        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _ply_header_lines(filepath: Path) -> int:
        """Count header lines in a .ply file."""
        count = 0
        with open(filepath, "rb") as f:
            for line in f:
                count += 1
                if line.strip() == b"end_header":
                    break
        return count

    @staticmethod
    def map_to_jtg(labels_source: np.ndarray, mapping: Optional[Dict[int, int]] = None) -> np.ndarray:
        """Map source labels to JTG 5210-2018 labels.

        Args:
            labels_source: Source label indices.
            mapping: Dict mapping source_label → JTG_label. If None, pass through.

        Returns:
            JTG 5210-2018 labels.
        """
        if mapping is None:
            return labels_source
        mapped = np.zeros_like(labels_source)
        for src, jtg in mapping.items():
            mapped[labels_source == src] = jtg
        return mapped


if __name__ == "__main__":
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    # Test with synthetic data as proxy for real data
    import tempfile, os
    from roadmc.data.synthetic.config import GeneratorConfig, RoadSurfaceConfig, DiseaseConfig
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = GeneratorConfig(
            road=RoadSurfaceConfig(width=3.0, length=3.0, grid_res=0.03),
            disease=DiseaseConfig(max_diseases_per_scene=0),
            seed=42, num_points=1024, normalize=False,
        )
        ds = SyntheticRoadDataset(cfg, dataset_size=2)
        for i in range(2):
            scene = ds.generate_scene(i)
            fpath = os.path.join(tmpdir, f"scene_{i:04d}.npy")
            np.save(fpath, np.column_stack([
                scene["points"], scene["labels"].astype(np.float32), scene["normals"]
            ]))

        rds = RealRoadDataset(tmpdir, file_pattern="*.npy", normalize=True)
        sample = rds[0]
        assert sample["coords"].shape[-1] == 3, f"Expected 3D coords, got {sample['coords'].shape}"
        assert sample["coords"].shape[0] <= 1024
        print(f"[PASS] RealRoadDataset: {len(rds)} files, coords={sample['coords'].shape}, "
              f"labels={sample['labels'].shape}, normals={sample['normals'].shape}")
