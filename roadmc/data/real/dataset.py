"""Real road point-cloud loading with unit-aware PLY/LAS/LAZ support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from roadmc.data.features import (
    DEFAULT_GEOMETRY_K_NEIGHBORS,
    compute_observable_features,
)

from .metadata import RoadPointCloudMetadata, load_scene_metadata, metadata_sidecar_path


SUPPORTED_EXTENSIONS = {".npy", ".ply", ".pcd", ".las", ".laz"}


class RealRoadDataset(Dataset):
    """Load real scans into the RoadMC tensor contract.

    The coordinate tensor is always converted to meters before optional unit-ball
    normalization. Intensity is always returned in ``[0, 1]`` according to the
    per-scene sidecar metadata. Labels are optional because most public road
    scans are initially unlabeled.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_pattern: str = "*.npy",
        max_points: Optional[int] = 65536,
        normalize: bool = True,
        label_mapping: Optional[Dict[int, int]] = None,
        require_metadata: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(
            path for path in self.data_dir.glob(file_pattern) if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        self.max_points = max_points
        self.normalize = normalize
        self.label_mapping = label_mapping
        self.require_metadata = require_metadata

        if not self.files:
            raise FileNotFoundError(f"No supported {file_pattern} files in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def metadata_for_index(self, idx: int) -> RoadPointCloudMetadata:
        """Return the validated metadata contract for one source scene."""
        path = self.files[idx]
        fallback = RoadPointCloudMetadata(
            intensity_scale="uint16" if path.suffix.lower() in {".las", ".laz"} else "normalized_0_1"
        )
        return load_scene_metadata(path, fallback=fallback, require_sidecar=self.require_metadata)

    @classmethod
    def load_scene(
        cls,
        filepath: str | Path,
        *,
        label_mapping: Optional[Dict[int, int]] = None,
        require_metadata: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], RoadPointCloudMetadata]:
        """Load one raw scene, convert its units to meters, and normalize intensity."""
        path = Path(filepath)
        points, labels, normals, intensities = cls._load_file(path)
        if metadata_sidecar_path(path).exists() or require_metadata:
            fallback = RoadPointCloudMetadata(
                intensity_scale="uint16" if path.suffix.lower() in {".las", ".laz"} else "normalized_0_1"
            )
            metadata = load_scene_metadata(path, fallback=fallback, require_sidecar=require_metadata)
        else:
            metadata = cls._inferred_metadata(path, intensities)
        points = points.astype(np.float32, copy=False) * metadata.coordinate_scale_to_meters
        valid_points = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
        points = points[valid_points]
        labels = labels[valid_points] if labels is not None else None
        normals = normals[valid_points] if normals is not None else None
        intensities = intensities[valid_points] if intensities is not None else None
        if len(points) == 0:
            raise ValueError(f"No finite, nonzero points remain after loading {path}")
        intensities = metadata.normalize_intensity(intensities)
        if labels is not None:
            labels = cls.map_to_jtg(labels.astype(np.int64, copy=False), label_mapping)
        if normals is not None:
            normals = cls._normalize_normals(normals)
        return points, labels, normals, intensities, metadata

    @staticmethod
    def _inferred_metadata(filepath: Path, intensities: Optional[np.ndarray]) -> RoadPointCloudMetadata:
        """Choose only a conservative fallback when a real scan lacks a sidecar."""
        suffix = filepath.suffix.lower()
        if suffix in {".las", ".laz"}:
            intensity_scale = "uint16"
        elif suffix == ".pcd" and intensities is not None and np.nanmax(intensities) > 1.0:
            intensity_scale = "uint8"
        else:
            intensity_scale = "normalized_0_1"
        return RoadPointCloudMetadata(intensity_scale=intensity_scale)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        points, labels, normals, intensities, _ = self.load_scene(
            self.files[idx],
            label_mapping=self.label_mapping,
            require_metadata=self.require_metadata,
        )

        if self.max_points is not None and points.shape[0] > self.max_points:
            idx_keep = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idx_keep]
            labels = labels[idx_keep] if labels is not None else None
            normals = normals[idx_keep] if normals is not None else None
            intensities = intensities[idx_keep] if intensities is not None else None

        # Match the synthetic feature contract from measurements available on a
        # real scan.  Geometry is computed before coordinate normalization;
        # curvature and the residual are translation/scale invariant.
        features = compute_observable_features(
            points,
            intensities,
            k_neighbors=DEFAULT_GEOMETRY_K_NEIGHBORS,
        )

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_radius = np.max(np.linalg.norm(points, axis=1))
            if max_radius > 1e-12:
                points = points / max_radius

        return {
            "coords": torch.from_numpy(points).float(),
            "labels": torch.from_numpy(labels).long()
            if labels is not None
            else torch.zeros(points.shape[0], dtype=torch.long),
            "normals": torch.from_numpy(normals).float()
            if normals is not None
            else torch.zeros(points.shape[0], 3, dtype=torch.float32),
            "intensities": torch.from_numpy(intensities).float()
            if intensities is not None
            else torch.zeros(points.shape[0], dtype=torch.float32),
            "feats": torch.from_numpy(features).float(),
        }

    @staticmethod
    def _load_file(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        ext = filepath.suffix.lower()
        if ext == ".npy":
            return RealRoadDataset._load_npy(filepath)
        if ext == ".ply":
            return RealRoadDataset._load_ply(filepath)
        if ext == ".pcd":
            return RealRoadDataset._load_pcd(filepath)
        if ext in {".las", ".laz"}:
            return RealRoadDataset._load_las(filepath)
        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_npy(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        data = np.load(filepath, allow_pickle=True)
        if data.dtype.names:
            names = set(data.dtype.names)
            points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
            labels = RealRoadDataset._field(data, names, ("label", "labels", "class", "semantic"), np.int64)
            normals = RealRoadDataset._normals_from_fields(data, names)
            intensities = RealRoadDataset._field(data, names, ("intensity", "reflectance"), np.float32)
            return points, labels, normals, intensities

        array = np.asarray(data)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Unsupported .npy array shape: {array.shape}")
        points = array[:, :3].astype(np.float32)
        labels = array[:, 3].astype(np.int64) if array.shape[1] > 3 else None
        normals = array[:, 4:7].astype(np.float32) if array.shape[1] >= 7 else None
        intensities = array[:, 7].astype(np.float32) if array.shape[1] > 7 else None
        return points, labels, normals, intensities

    @staticmethod
    def _load_ply(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            from plyfile import PlyData
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("PLY support requires `pip install roadmc[real]`") from exc

        ply = PlyData.read(str(filepath))
        if "vertex" not in ply:
            raise ValueError(f"PLY file has no vertex element: {filepath}")
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())
        required = {"x", "y", "z"}
        if not required.issubset(names):
            raise ValueError(f"PLY vertex element must provide x/y/z, got {sorted(names)}")
        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
        labels = RealRoadDataset._field(vertex, names, ("label", "labels", "class", "semantic"), np.int64)
        normals = RealRoadDataset._normals_from_fields(vertex, names)
        intensities = RealRoadDataset._field(
            vertex, names, ("intensity", "reflectance", "scalar_intensity"), np.float32
        )
        return points, labels, normals, intensities

    @staticmethod
    def _load_las(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("LAS/LAZ support requires `pip install roadmc[real]`") from exc

        las = laspy.read(str(filepath))
        names = set(las.point_format.dimension_names)
        points = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
        labels = RealRoadDataset._field(las, names, ("label", "labels", "class", "classification"), np.int64)
        normals = RealRoadDataset._normals_from_fields(las, names)
        intensities = RealRoadDataset._field(las, names, ("intensity", "reflectance"), np.float32)
        return points, labels, normals, intensities

    @staticmethod
    def _load_pcd(
        filepath: Path,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        header, encoding, payload = RealRoadDataset._read_pcd_header(filepath)
        fields = header.get("FIELDS", [])
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if encoding == "ascii" and fields:
            try:
                values = np.loadtxt(payload, dtype=np.float64)
            finally:
                payload.close()
            if values.ndim == 1:
                values = values[None, :]
            offsets: dict[str, int] = {}
            offset = 0
            for field, count in zip(fields, counts):
                offsets[field] = offset
                offset += count
            if not {"x", "y", "z"}.issubset(offsets):
                raise ValueError(f"PCD file must provide x/y/z fields: {filepath}")
            points = np.column_stack([values[:, offsets[name]] for name in ("x", "y", "z")]).astype(np.float32)
            labels = next(
                (values[:, offsets[name]].astype(np.int64) for name in ("label", "labels", "class", "semantic") if name in offsets),
                None,
            )
            normals = None
            for normal_fields in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
                if set(normal_fields).issubset(offsets):
                    normals = np.column_stack([values[:, offsets[name]] for name in normal_fields]).astype(np.float32)
                    break
            intensities = next(
                (values[:, offsets[name]].astype(np.float32) for name in ("intensity", "reflectance") if name in offsets),
                None,
            )
            return points, labels, normals, intensities

        try:
            import open3d as o3d
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("PCD support requires `pip install open3d`") from exc

        cloud = o3d.io.read_point_cloud(str(filepath), remove_nan_points=True)
        points = np.asarray(cloud.points, dtype=np.float32)
        if len(points) == 0:
            raise ValueError(f"PCD file contains no valid points: {filepath}")
        normals = np.asarray(cloud.normals, dtype=np.float32) if cloud.has_normals() else None
        # Open3D exposes PCD intensity-like channels as gray colors for many
        # common exports, including the M2S-RoAD preparation code.
        intensities = np.asarray(cloud.colors, dtype=np.float32)[:, 0] if cloud.has_colors() else None
        return points, None, normals, intensities

    @staticmethod
    def _read_pcd_header(filepath: Path) -> Tuple[dict[str, list[str]], str, object]:
        """Read a PCD header and return an open payload handle for ASCII parsing."""
        handle = filepath.open("rb")
        header: dict[str, list[str]] = {}
        encoding = ""
        try:
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"PCD header ended before DATA declaration: {filepath}")
                text = line.decode("ascii", errors="strict").strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                key = tokens[0].upper()
                values = tokens[1:]
                if key == "DATA":
                    if not values:
                        raise ValueError(f"PCD DATA declaration is empty: {filepath}")
                    encoding = values[0].lower()
                    break
                header[key] = values
            if encoding == "ascii":
                return header, encoding, handle
        except Exception:
            handle.close()
            raise
        handle.close()
        return header, encoding, None

    @staticmethod
    def _field(
        source: object,
        names: set[str],
        candidates: tuple[str, ...],
        dtype: np.dtype,
    ) -> Optional[np.ndarray]:
        for name in candidates:
            if name in names:
                return np.asarray(source[name], dtype=dtype)  # type: ignore[index]
        return None

    @staticmethod
    def _normals_from_fields(source: object, names: set[str]) -> Optional[np.ndarray]:
        for candidates in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
            if set(candidates).issubset(names):
                return np.column_stack([source[name] for name in candidates]).astype(np.float32)  # type: ignore[index]
        return None

    @staticmethod
    def _normalize_normals(normals: np.ndarray) -> np.ndarray:
        result = np.asarray(normals, dtype=np.float32).copy()
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        valid = norm[:, 0] > 1e-12
        result[valid] /= norm[valid]
        result[~valid] = 0.0
        return result

    @staticmethod
    def map_to_jtg(labels_source: np.ndarray, mapping: Optional[Dict[int, int]] = None) -> np.ndarray:
        """Map source semantic IDs to JTG IDs; unmapped IDs become background."""
        if mapping is None:
            return labels_source.astype(np.int64, copy=False)
        mapped = np.zeros_like(labels_source, dtype=np.int64)
        for source_label, jtg_label in mapping.items():
            mapped[labels_source == source_label] = jtg_label
        return mapped
