"""Sensor-observable geometric features shared by synthetic and real point clouds.

The segmentation input contract is deliberately restricted to quantities that
can be computed at inference time:

``[normalized_intensity, pca_curvature, signed_local_height_residual]``.

For a point :math:`x_i` and its local neighborhood, let ``C_i`` be the
covariance matrix of the neighbors.  The smallest-eigenvalue eigenvector is
the local surface normal and ``lambda_min / trace(C_i)`` is the conventional
PCA surface-variation curvature.  The signed local-height residual is the
orthogonal distance from ``x_i`` to that neighborhood's tangent plane,
normalized by the k-nearest-neighbor radius.  Both geometric channels are
therefore dimensionless and invariant to global translation and uniform scale.

No function in this module accepts labels.  Keeping that boundary explicit is
important: synthetic supervision must never become an inference-time feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


OBSERVABLE_FEATURE_SCHEMA = "roadmc.observable_features.v1"
OBSERVABLE_FEATURE_NAMES = (
    "normalized_intensity",
    "pca_curvature",
    "signed_local_height_residual",
)
DEFAULT_GEOMETRY_K_NEIGHBORS = 16
DEFAULT_HEIGHT_RESIDUAL_CLIP = 3.0


@dataclass(frozen=True)
class LocalSurfaceGeometry:
    """Local geometric quantities estimated from a point cloud alone."""

    pca_curvature: np.ndarray
    normals: np.ndarray
    signed_height_residual: np.ndarray
    support_radius: np.ndarray
    density_per_m2: np.ndarray


def _validated_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite before geometric feature extraction")
    return array


def normalized_intensity(intensities: np.ndarray | None, count: int) -> np.ndarray:
    """Return a finite sensor-intensity channel in ``[0, 1]``.

    Source loaders are responsible for unit-aware sensor normalization.  This
    final guard handles absent or partially invalid returns without creating a
    label-dependent fallback.
    """

    if intensities is None:
        return np.zeros(count, dtype=np.float32)
    values = np.asarray(intensities, dtype=np.float64).reshape(-1)
    if len(values) != count:
        raise ValueError(f"intensities must have shape ({count},), got {values.shape}")
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(count, dtype=np.float32)
    replacement = float(np.median(values[finite]))
    values = np.where(finite, values, replacement)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def estimate_local_surface_geometry(
    points: np.ndarray,
    *,
    k_neighbors: int = DEFAULT_GEOMETRY_K_NEIGHBORS,
) -> LocalSurfaceGeometry:
    """Estimate PCA tangent planes and scale-normalized local observables.

    The road coordinate convention is ``z`` up.  PCA normal signs are oriented
    to positive ``z`` so that depressions and protrusions retain opposite signs.
    Tiny scenes have no stable tangent plane and return neutral zero features;
    production scenes should contain far more than four points.
    """

    cloud = _validated_points(points)
    count = len(cloud)
    if count == 0:
        empty = np.empty(0, dtype=np.float64)
        return LocalSurfaceGeometry(empty, np.empty((0, 3), dtype=np.float64), empty, empty, empty)
    if count < 4:
        return LocalSurfaceGeometry(
            pca_curvature=np.zeros(count, dtype=np.float64),
            normals=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
            signed_height_residual=np.zeros(count, dtype=np.float64),
            support_radius=np.zeros(count, dtype=np.float64),
            density_per_m2=np.zeros(count, dtype=np.float64),
        )
    if k_neighbors < 3:
        raise ValueError("k_neighbors must be at least 3")

    k = min(int(k_neighbors), count - 1)
    tree = cKDTree(cloud)
    distances, indices = tree.query(cloud, k=k + 1)
    neighbors = cloud[indices[:, 1:]]
    centers = neighbors.mean(axis=1)
    offsets = neighbors - centers[:, None, :]
    covariance = np.einsum("nki,nkj->nij", offsets, offsets) / max(k - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    total_variance = np.maximum(eigenvalues.sum(axis=1), 1e-12)
    curvature = np.clip(eigenvalues[:, 0] / total_variance, 0.0, 1.0)
    normals = eigenvectors[:, :, 0]
    normals[normals[:, 2] < 0.0] *= -1.0

    residual = np.einsum("ni,ni->n", cloud - centers, normals)
    support_radius = np.maximum(distances[:, -1], 1e-6)
    density = k / (np.pi * support_radius**2)
    return LocalSurfaceGeometry(
        pca_curvature=curvature,
        normals=normals,
        signed_height_residual=residual,
        support_radius=support_radius,
        density_per_m2=density,
    )


def compute_observable_features(
    points: np.ndarray,
    intensities: np.ndarray | None = None,
    *,
    k_neighbors: int = DEFAULT_GEOMETRY_K_NEIGHBORS,
    height_residual_clip: float = DEFAULT_HEIGHT_RESIDUAL_CLIP,
) -> np.ndarray:
    """Build the three-channel RoadMC model input from observable measurements.

    ``height_residual_clip`` limits pathological sparse-neighborhood values but
    does not change the sign of physically meaningful concave or convex damage.
    """

    if height_residual_clip <= 0.0:
        raise ValueError("height_residual_clip must be positive")
    cloud = _validated_points(points)
    geometry = estimate_local_surface_geometry(cloud, k_neighbors=k_neighbors)
    scaled_residual = geometry.signed_height_residual / np.maximum(geometry.support_radius, 1e-6)
    scaled_residual = np.clip(scaled_residual, -height_residual_clip, height_residual_clip)
    return np.column_stack(
        (
            normalized_intensity(intensities, len(cloud)),
            geometry.pca_curvature,
            scaled_residual,
        )
    ).astype(np.float32)


def has_observable_feature_schema(value: object) -> bool:
    """Return whether a saved scalar schema value matches the current contract."""

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return False
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value) == OBSERVABLE_FEATURE_SCHEMA


def checkpoint_feature_schema(payload: object) -> str | None:
    """Read the explicit input-feature schema from a Lightning checkpoint."""

    if not isinstance(payload, Mapping):
        return None
    value = payload.get("roadmc_feature_schema")
    if value is None:
        hyper_parameters = payload.get("hyper_parameters")
        if isinstance(hyper_parameters, Mapping):
            value = hyper_parameters.get("feature_schema")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def require_observable_checkpoint_schema(payload: object, *, context: str) -> str:
    """Reject checkpoints trained before the inference-safe feature contract."""

    schema = checkpoint_feature_schema(payload)
    if schema != OBSERVABLE_FEATURE_SCHEMA:
        raise ValueError(
            f"{context} has feature schema {schema!r}; expected {OBSERVABLE_FEATURE_SCHEMA!r}. "
            "It may have been trained with a label-derived input feature and cannot be used "
            "in a credible observable-feature experiment."
        )
    return schema
