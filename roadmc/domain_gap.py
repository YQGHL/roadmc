"""Observable synthetic-to-real point-cloud domain-gap diagnostics.

The diagnostics deliberately use quantities measurable from both synthetic and
real scans: local density, normalized intensity, normal tilt, PCA curvature,
and local height residual. They avoid segmentation labels and synthetic-only
privileged features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import energy_distance, wasserstein_distance

from roadmc.data.features import (
    DEFAULT_HEIGHT_RESIDUAL_CLIP,
    OBSERVABLE_FEATURE_NAMES,
    OBSERVABLE_FEATURE_SCHEMA,
    estimate_local_surface_geometry,
)


OBSERVABLE_NAMES = (
    "density_per_m2",
    "intensity",
    "normal_tilt_rad",
    "pca_curvature",
    "height_residual_m",
    "signed_height_residual_over_radius",
)


@dataclass(frozen=True)
class PointCloudRecord:
    """One point cloud represented in metric coordinates and normalized intensity."""

    points: np.ndarray
    intensities: np.ndarray | None = None
    normals: np.ndarray | None = None
    name: str = "scene"


def dominant_ground_surface(
    record: PointCloudRecord,
    *,
    distance_threshold: float = 0.15,
    max_fit_points: int = 10000,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[PointCloudRecord, dict[str, float | int]]:
    """Keep points near the dominant RANSAC plane as a road-surface proxy.

    This is a geometric ROI filter, not a semantic road label. It should be
    used for early sensor-domain diagnostics only, then replaced by a verified
    road ROI when source annotations or calibration become available.
    """
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("ground-plane filtering requires `pip install open3d`") from exc

    points = _validated_points(record.points, None, seed)
    fit_points = _subsample(points, max_fit_points, seed)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fit_points))
    plane, _ = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=iterations,
    )
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal_norm = max(float(np.linalg.norm(normal)), 1e-12)
    normal_z = float(normal[2] / normal_norm)
    if abs(normal_z) < 0.7:
        raise ValueError(
            f"dominant plane for {record.name} is not ground-like (normal_z={normal_z:.3f})"
        )
    signed_distance = (np.asarray(record.points, dtype=np.float64) @ normal + float(plane[3])) / normal_norm
    mask = np.isfinite(signed_distance) & (np.abs(signed_distance) <= distance_threshold)
    if int(mask.sum()) < 4:
        raise ValueError(f"ground-plane filter retained fewer than four points for {record.name}")
    filtered = PointCloudRecord(
        points=np.asarray(record.points)[mask],
        intensities=np.asarray(record.intensities)[mask] if record.intensities is not None else None,
        normals=np.asarray(record.normals)[mask] if record.normals is not None else None,
        name=record.name,
    )
    return filtered, {
        "input_points": int(len(record.points)),
        "retained_points": int(mask.sum()),
        "retained_fraction": float(mask.mean()),
        "plane_normal_z": normal_z,
        "distance_threshold_m": float(distance_threshold),
    }


def _validated_points(points: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    array = array[np.isfinite(array).all(axis=1)]
    if len(array) < 4:
        raise ValueError("at least four finite points are required for geometric diagnostics")
    if max_points is not None and len(array) > max_points:
        rng = np.random.default_rng(seed)
        array = array[rng.choice(len(array), size=max_points, replace=False)]
    return array


def observable_descriptors(
    record: PointCloudRecord,
    *,
    k_neighbors: int = 16,
    max_points: int | None = 4096,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute sensor-observable local geometry descriptors for one scene.

    For a neighborhood covariance matrix with eigenvalues
    ``lambda_0 <= lambda_1 <= lambda_2``, curvature is
    ``lambda_0 / sum(lambda)``. The height residual is the orthogonal distance
    from the query point to its neighbors' PCA tangent plane. Surface density
    uses ``k / (pi r_k^2)``, appropriate for a locally two-dimensional road.
    """
    original_points = np.asarray(record.points)
    valid = np.isfinite(original_points).all(axis=1)
    points = _validated_points(original_points, max_points, seed)
    if len(points) != int(valid.sum()):
        # Sampling happened after finite filtering. Reproduce the selected rows
        # below only for optional per-point fields by matching through indices.
        finite_indices = np.flatnonzero(valid)
        rng = np.random.default_rng(seed)
        selected_indices = finite_indices[rng.choice(len(finite_indices), size=len(points), replace=False)]
    else:
        selected_indices = np.flatnonzero(valid)

    geometry = estimate_local_surface_geometry(points, k_neighbors=k_neighbors)
    curvature = geometry.pca_curvature
    pca_normals = geometry.normals

    normals = pca_normals
    if record.normals is not None:
        supplied = np.asarray(record.normals, dtype=np.float64)
        if supplied.ndim == 2 and supplied.shape == original_points.shape:
            supplied = supplied[selected_indices]
            supplied_norm = np.linalg.norm(supplied, axis=1)
            supplied_valid = supplied_norm > 1e-8
            supplied[supplied_valid] /= supplied_norm[supplied_valid, None]
            supplied[supplied[:, 2] < 0.0] *= -1.0
            normals = pca_normals.copy()
            normals[supplied_valid] = supplied[supplied_valid]

    normal_tilt = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
    height_residual = np.abs(geometry.signed_height_residual)
    density = geometry.density_per_m2
    scaled_signed_residual = np.clip(
        geometry.signed_height_residual / np.maximum(geometry.support_radius, 1e-6),
        -DEFAULT_HEIGHT_RESIDUAL_CLIP,
        DEFAULT_HEIGHT_RESIDUAL_CLIP,
    )

    output = {
        "density_per_m2": density.astype(np.float64),
        "normal_tilt_rad": normal_tilt.astype(np.float64),
        "pca_curvature": curvature.astype(np.float64),
        "height_residual_m": height_residual.astype(np.float64),
        "signed_height_residual_over_radius": scaled_signed_residual.astype(np.float64),
    }
    if record.intensities is not None:
        intensities = np.asarray(record.intensities, dtype=np.float64)
        if intensities.ndim == 1 and len(intensities) == len(original_points):
            intensity = intensities[selected_indices]
            output["intensity"] = np.clip(intensity[np.isfinite(intensity)], 0.0, 1.0)
    return output


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": 0.0, "std": 0.0, "q05": 0.0, "q50": 0.0, "q95": 0.0}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def _subsample(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=limit, replace=False)]


def rbf_mmd(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_samples: int = 512,
    seed: int = 42,
) -> float:
    """Biased RBF-kernel MMD with a median-distance bandwidth heuristic."""
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if source_array.ndim == 1:
        source_array = source_array[:, None]
    if target_array.ndim == 1:
        target_array = target_array[:, None]
    source_array = source_array[np.isfinite(source_array).all(axis=1)]
    target_array = target_array[np.isfinite(target_array).all(axis=1)]
    if not len(source_array) or not len(target_array):
        return float("nan")
    source_array = _subsample(source_array, max_samples, seed)
    target_array = _subsample(target_array, max_samples, seed + 1)
    combined = np.vstack((source_array, target_array))
    mean = combined.mean(axis=0)
    scale = np.maximum(combined.std(axis=0), 1e-8)
    source_array = (source_array - mean) / scale
    target_array = (target_array - mean) / scale
    reference = np.vstack((source_array, target_array))
    pairwise_sq = ((reference[:, None, :] - reference[None, :, :]) ** 2).sum(axis=-1)
    nonzero = pairwise_sq[pairwise_sq > 0.0]
    bandwidth_sq = float(np.median(nonzero)) if len(nonzero) else 1.0
    bandwidth_sq = max(bandwidth_sq, 1e-8)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-squared / (2.0 * bandwidth_sq))

    return float(kernel(source_array, source_array).mean() + kernel(target_array, target_array).mean() - 2.0 * kernel(source_array, target_array).mean())


def compare_domains(
    source_records: Iterable[PointCloudRecord],
    target_records: Iterable[PointCloudRecord],
    *,
    k_neighbors: int = 16,
    max_points_per_scene: int = 4096,
    mmd_max_samples: int = 512,
    seed: int = 42,
) -> dict:
    """Compare two scene collections using W1, energy distance, and MMD."""
    source_records = list(source_records)
    target_records = list(target_records)
    if not source_records or not target_records:
        raise ValueError("both source_records and target_records must be non-empty")

    def collect(records: list[PointCloudRecord], seed_offset: int) -> dict[str, np.ndarray]:
        gathered: dict[str, list[np.ndarray]] = {name: [] for name in OBSERVABLE_NAMES}
        for index, record in enumerate(records):
            descriptors = observable_descriptors(
                record,
                k_neighbors=k_neighbors,
                max_points=max_points_per_scene,
                seed=seed + seed_offset + index,
            )
            for name, values in descriptors.items():
                if len(values):
                    gathered[name].append(values)
        return {
            name: np.concatenate(values) if values else np.empty(0, dtype=np.float64)
            for name, values in gathered.items()
        }

    source = collect(source_records, 0)
    target = collect(target_records, 100000)
    descriptors = {}
    common_names = []
    for name in OBSERVABLE_NAMES:
        source_values = source[name]
        target_values = target[name]
        available = len(source_values) > 0 and len(target_values) > 0
        if available:
            common_names.append(name)
            distances = {
                "wasserstein_1": float(wasserstein_distance(source_values, target_values)),
                "energy": float(energy_distance(source_values, target_values)),
                "rbf_mmd": rbf_mmd(source_values, target_values, max_samples=mmd_max_samples, seed=seed),
            }
        else:
            distances = {"wasserstein_1": None, "energy": None, "rbf_mmd": None}
        descriptors[name] = {
            "available": available,
            "source": _summary(source_values),
            "target": _summary(target_values),
            "distances": distances,
        }

    joint_mmd = None
    if common_names:
        source_matrix = np.column_stack([source[name] for name in common_names])
        target_matrix = np.column_stack([target[name] for name in common_names])
        joint_mmd = rbf_mmd(source_matrix, target_matrix, max_samples=mmd_max_samples, seed=seed)

    return {
        "source_scene_count": len(source_records),
        "target_scene_count": len(target_records),
        "model_input_feature_schema": OBSERVABLE_FEATURE_SCHEMA,
        "model_input_feature_names": list(OBSERVABLE_FEATURE_NAMES),
        "k_neighbors": k_neighbors,
        "max_points_per_scene": max_points_per_scene,
        "descriptors": descriptors,
        "joint_rbf_mmd": joint_mmd,
        "joint_descriptor_names": common_names,
    }
