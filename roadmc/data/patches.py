"""Deterministic metric-space patch extraction for RoadMC point clouds.

Synthetic RoadMC scenes store unit-sphere-normalized coordinates together
with the inverse transform used to recover meters::

    points_m = points * coordinate_scale + coordinate_center

Patch membership is always computed from ``points_m``.  The original
``points`` values are retained for model compatibility, while ``points_m``
and metric patch bounds are included for auditing and downstream processing.

This module deliberately depends only on NumPy.  It does not load data,
augment samples, or make assumptions about a PyTorch training pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

Split = Literal["train", "val", "test"]

_REQUIRED_ARRAY_KEYS = ("points", "labels", "feats", "normals")
_PHYSICAL_METADATA_KEYS = ("coordinate_center", "coordinate_scale")
_SPLIT_STREAM = {"train": 0, "val": 1, "test": 2}


def restore_metric_points(scene: Mapping[str, Any]) -> np.ndarray:
    """Restore physical coordinates in meters from a normalized scene.

    Args:
        scene: Mapping containing ``points``, ``coordinate_center``, and
            ``coordinate_scale``.  ``coordinate_center`` must have shape
            ``(3,)`` and ``coordinate_scale`` must be a finite positive scalar.

    Returns:
        A new ``float64`` array of shape ``(N, 3)`` in meters.

    Raises:
        ValueError: If physical metadata is absent or malformed, or if points
            are not finite ``(N, 3)`` coordinates.
    """

    _require_keys(scene, ("points", *_PHYSICAL_METADATA_KEYS))
    points = np.asarray(scene["points"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError("points must be numeric")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")

    center = np.asarray(scene["coordinate_center"], dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError(
            "coordinate_center must be a finite array with shape (3,), "
            f"got {center.shape}"
        )

    scale_array = np.asarray(scene["coordinate_scale"], dtype=np.float64)
    if scale_array.size != 1:
        raise ValueError("coordinate_scale must be a finite positive scalar")
    scale = float(scale_array.reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"coordinate_scale must be a finite positive scalar, got {scale!r}"
        )

    return points.astype(np.float64, copy=False) * scale + center


def extract_overlapping_patches(
    scene: Mapping[str, Any],
    patch_size_m: float | Sequence[float],
    stride_m: float | Sequence[float],
    *,
    points_per_patch: int | None = None,
    split: Split = "train",
    disease_aware: bool = True,
    disease_fraction: float = 0.5,
    background_label: int = 0,
    seed: int = 0,
    include_empty: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Split one scene into deterministic, overlapping XY patches in meters.

    Scalar sizes create square patches.  Two-element values are interpreted as
    ``(x_size, y_size)``.  The final patch on each axis is anchored to the scene
    maximum when a regular stride would leave an uncovered strip.  Intervals
    are half-open except at the global maximum, so touching non-overlapping
    patches do not duplicate boundary points.

    When ``points_per_patch`` is set (typically 4096 or 8192), patches with too
    many points are sampled without replacement.  Training can reserve a
    configurable fraction for non-background labels.  Validation and test
    always use deterministic uniform sampling, regardless of
    ``disease_aware``.  Short patches are zero-padded; labels and source indices
    use ``-1`` and ``valid_mask`` distinguishes observations from padding.

    Args:
        scene: Mapping containing ``points``, ``labels``, ``feats``, ``normals``,
            ``coordinate_center``, and ``coordinate_scale``.  An optional input
            ``valid_mask`` excludes existing padding before spatial slicing.
        patch_size_m: Patch width and length in meters.
        stride_m: Patch stride in meters.  It must not exceed patch size on
            either axis, which guarantees complete spatial coverage.
        points_per_patch: Optional fixed point count, such as 4096 or 8192.
        split: ``"train"``, ``"val"``, or ``"test"``.
        disease_aware: Enable label-aware sampling for training patches.
        disease_fraction: Desired non-background fraction in a downsampled
            training patch.  Available points are never duplicated.
        background_label: Label treated as healthy road surface.
        seed: Non-negative seed.  Sampling is reproducible per grid cell.
        include_empty: Return spatially empty grid cells when true.

    Returns:
        Patch dictionaries retaining ``points``, ``labels``, ``feats``, and
        ``normals`` plus ``points_m``, ``valid_mask``, ``source_indices``,
        ``patch_bounds_m``, ``patch_grid_index``, ``patch_size_m``,
        ``stride_m``, and inverse-coordinate metadata.

    Raises:
        ValueError: If arrays, metadata, sizes, sampling settings, or split are
            invalid.  Missing physical metadata is never guessed.
    """

    _require_keys(scene, (*_REQUIRED_ARRAY_KEYS, *_PHYSICAL_METADATA_KEYS))
    if split not in _SPLIT_STREAM:
        raise ValueError(f"split must be one of {tuple(_SPLIT_STREAM)}, got {split!r}")

    patch_size = _as_xy_pair(patch_size_m, "patch_size_m")
    stride = _as_xy_pair(stride_m, "stride_m")
    if np.any(stride > patch_size):
        raise ValueError("stride_m must not exceed patch_size_m on either axis")
    fixed_count = _validate_points_per_patch(points_per_patch)
    if not 0.0 <= disease_fraction <= 1.0 or not np.isfinite(disease_fraction):
        raise ValueError("disease_fraction must be finite and in [0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    points = np.asarray(scene["points"])
    labels = np.asarray(scene["labels"])
    feats = np.asarray(scene["feats"])
    normals = np.asarray(scene["normals"])
    points_m = restore_metric_points(scene)
    point_count = points.shape[0]
    _validate_point_arrays(point_count, labels, feats, normals)

    if "valid_mask" in scene:
        source_valid = np.asarray(scene["valid_mask"])
        if source_valid.shape != (point_count,):
            raise ValueError(
                f"valid_mask must have shape ({point_count},), got {source_valid.shape}"
            )
        if not np.issubdtype(source_valid.dtype, np.bool_):
            raise ValueError("valid_mask must have boolean dtype")
    else:
        source_valid = np.ones(point_count, dtype=np.bool_)

    valid_indices = np.flatnonzero(source_valid)
    if valid_indices.size == 0:
        raise ValueError("scene contains no valid points to patch")
    valid_points_m = points_m[valid_indices]
    xy_min = np.min(valid_points_m[:, :2], axis=0)
    xy_max = np.max(valid_points_m[:, :2], axis=0)
    x_starts = _axis_starts(xy_min[0], xy_max[0], patch_size[0], stride[0])
    y_starts = _axis_starts(xy_min[1], xy_max[1], patch_size[1], stride[1])
    tolerance = _coordinate_tolerance(valid_points_m[:, :2])

    patches: list[dict[str, np.ndarray]] = []
    grid_linear_index = 0
    for y_index, y_start in enumerate(y_starts):
        y_end = y_start + patch_size[1]
        y_mask = _axis_membership(
            valid_points_m[:, 1],
            y_start,
            y_end,
            is_last=y_index == len(y_starts) - 1,
            tolerance=tolerance,
        )
        for x_index, x_start in enumerate(x_starts):
            x_end = x_start + patch_size[0]
            x_mask = _axis_membership(
                valid_points_m[:, 0],
                x_start,
                x_end,
                is_last=x_index == len(x_starts) - 1,
                tolerance=tolerance,
            )
            source_indices = valid_indices[x_mask & y_mask]
            current_grid_index = grid_linear_index
            grid_linear_index += 1
            if source_indices.size == 0 and not include_empty:
                continue

            sampled_indices = _sample_source_indices(
                source_indices,
                labels,
                fixed_count,
                split=split,
                disease_aware=disease_aware,
                disease_fraction=disease_fraction,
                background_label=background_label,
                seed=int(seed),
                grid_index=current_grid_index,
            )
            patches.append(
                _build_patch(
                    points=points,
                    points_m=points_m,
                    labels=labels,
                    feats=feats,
                    normals=normals,
                    sampled_indices=sampled_indices,
                    source_point_count=source_indices.size,
                    fixed_count=fixed_count,
                    bounds=(x_start, y_start, x_end, y_end),
                    grid_index=(x_index, y_index),
                    patch_size=patch_size,
                    stride=stride,
                    scene=scene,
                )
            )

    return patches


def _require_keys(scene: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in scene]
    if missing:
        raise ValueError(
            "scene is missing required data or physical metadata: " + ", ".join(missing)
        )


def _as_xy_pair(value: float | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 2)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a finite positive scalar or (x, y) pair")
    return array


def _validate_points_per_patch(points_per_patch: int | None) -> int | None:
    if points_per_patch is None:
        return None
    if isinstance(points_per_patch, (bool, np.bool_)) or not isinstance(
        points_per_patch, (int, np.integer)
    ):
        raise ValueError("points_per_patch must be a positive integer or None")
    if points_per_patch <= 0:
        raise ValueError("points_per_patch must be a positive integer or None")
    return int(points_per_patch)


def _validate_point_arrays(
    point_count: int,
    labels: np.ndarray,
    feats: np.ndarray,
    normals: np.ndarray,
) -> None:
    if labels.shape != (point_count,):
        raise ValueError(f"labels must have shape ({point_count},), got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must have an integer dtype")
    if feats.ndim != 2 or feats.shape[0] != point_count:
        raise ValueError(f"feats must have shape ({point_count}, F), got {feats.shape}")
    if normals.shape != (point_count, 3):
        raise ValueError(f"normals must have shape ({point_count}, 3), got {normals.shape}")
    if not np.issubdtype(feats.dtype, np.number) or not np.all(np.isfinite(feats)):
        raise ValueError("feats must be a finite numeric array")
    if not np.issubdtype(normals.dtype, np.number) or not np.all(np.isfinite(normals)):
        raise ValueError("normals must be a finite numeric array")


def _axis_starts(
    axis_min: float,
    axis_max: float,
    patch_size: float,
    stride: float,
) -> np.ndarray:
    span = axis_max - axis_min
    tolerance = max(1e-12, np.finfo(np.float64).eps * max(abs(axis_min), abs(axis_max), 1.0) * 32)
    if span <= patch_size + tolerance:
        return np.array([axis_min], dtype=np.float64)

    final_start = axis_max - patch_size
    regular_count = int(np.floor((final_start - axis_min) / stride + tolerance)) + 1
    starts = axis_min + np.arange(regular_count, dtype=np.float64) * stride
    if starts[-1] < final_start - tolerance:
        starts = np.append(starts, final_start)
    else:
        starts[-1] = final_start
    return starts


def _coordinate_tolerance(xy: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(xy))), 1.0)
    return max(1e-9, np.finfo(np.float64).eps * magnitude * 64.0)


def _axis_membership(
    values: np.ndarray,
    start: float,
    end: float,
    *,
    is_last: bool,
    tolerance: float,
) -> np.ndarray:
    lower = values >= start - tolerance
    if is_last:
        return lower & (values <= end + tolerance)
    return lower & (values < end)


def _sample_source_indices(
    source_indices: np.ndarray,
    labels: np.ndarray,
    points_per_patch: int | None,
    *,
    split: Split,
    disease_aware: bool,
    disease_fraction: float,
    background_label: int,
    seed: int,
    grid_index: int,
) -> np.ndarray:
    if points_per_patch is None or source_indices.size <= points_per_patch:
        return source_indices.astype(np.int64, copy=True)

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, grid_index, _SPLIT_STREAM[split]])
    )
    if split != "train" or not disease_aware:
        return np.sort(
            rng.choice(source_indices, size=points_per_patch, replace=False).astype(np.int64)
        )

    patch_labels = labels[source_indices]
    disease_indices = source_indices[patch_labels != background_label]
    background_indices = source_indices[patch_labels == background_label]
    if disease_indices.size == 0 or background_indices.size == 0:
        return np.sort(
            rng.choice(source_indices, size=points_per_patch, replace=False).astype(np.int64)
        )

    desired_disease = int(round(points_per_patch * disease_fraction))
    if disease_fraction > 0.0:
        desired_disease = max(1, desired_disease)
    disease_count = min(disease_indices.size, desired_disease)
    background_count = min(background_indices.size, points_per_patch - disease_count)
    remaining = points_per_patch - disease_count - background_count
    if remaining:
        extra_disease = min(disease_indices.size - disease_count, remaining)
        disease_count += extra_disease
        remaining -= extra_disease
    if remaining:
        background_count += min(background_indices.size - background_count, remaining)

    chosen_disease = rng.choice(disease_indices, size=disease_count, replace=False)
    chosen_background = rng.choice(background_indices, size=background_count, replace=False)
    return np.sort(np.concatenate((chosen_disease, chosen_background)).astype(np.int64))


def _build_patch(
    *,
    points: np.ndarray,
    points_m: np.ndarray,
    labels: np.ndarray,
    feats: np.ndarray,
    normals: np.ndarray,
    sampled_indices: np.ndarray,
    source_point_count: int,
    fixed_count: int | None,
    bounds: tuple[float, float, float, float],
    grid_index: tuple[int, int],
    patch_size: np.ndarray,
    stride: np.ndarray,
    scene: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    output_count = sampled_indices.size if fixed_count is None else fixed_count
    valid_count = sampled_indices.size
    valid_mask = np.zeros(output_count, dtype=np.bool_)
    valid_mask[:valid_count] = True

    patch = {
        "points": _take_and_pad(points, sampled_indices, output_count, 0),
        "points_m": _take_and_pad(points_m, sampled_indices, output_count, 0.0),
        "labels": _take_and_pad(
            labels.astype(np.int64, copy=False), sampled_indices, output_count, -1
        ),
        "feats": _take_and_pad(feats, sampled_indices, output_count, 0),
        "normals": _take_and_pad(normals, sampled_indices, output_count, 0),
        "valid_mask": valid_mask,
        "source_indices": _pad_source_indices(sampled_indices, output_count),
        "source_point_count": np.asarray(source_point_count, dtype=np.int64),
        "patch_bounds_m": np.asarray(bounds, dtype=np.float64),
        "patch_grid_index": np.asarray(grid_index, dtype=np.int64),
        "patch_size_m": patch_size.copy(),
        "stride_m": stride.copy(),
        "coordinate_center": np.asarray(scene["coordinate_center"], dtype=np.float64).copy(),
        "coordinate_scale": np.asarray(scene["coordinate_scale"], dtype=np.float64).copy(),
    }
    if "coordinates_normalized" in scene:
        patch["coordinates_normalized"] = np.asarray(
            scene["coordinates_normalized"], dtype=np.bool_
        ).copy()
    return patch


def _take_and_pad(
    array: np.ndarray,
    indices: np.ndarray,
    output_count: int,
    fill_value: int | float,
) -> np.ndarray:
    selected = array[indices]
    if selected.shape[0] == output_count:
        return selected.copy()
    output = np.full((output_count, *array.shape[1:]), fill_value, dtype=array.dtype)
    output[: selected.shape[0]] = selected
    return output


def _pad_source_indices(indices: np.ndarray, output_count: int) -> np.ndarray:
    output = np.full(output_count, -1, dtype=np.int64)
    output[: indices.size] = indices
    return output


__all__ = ["extract_overlapping_patches", "restore_metric_points"]
