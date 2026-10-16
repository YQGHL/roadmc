"""Explicit, uncalibrated wall/floor texture mixing for replay-v2 potholes.

The caller supplies the original normalized radius and geometry write mask.
These helpers never infer material identity from labels or change ownership.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np


def validate_partition(spec: Any, material: str, ident: str, validate_texture: Callable) -> None:
    required = {"mode", "floor_rho_start", "floor_rho_end", "floor_texture", "wall_texture"}
    if not isinstance(spec, dict) or set(spec) != required or spec["mode"] != "wall_floor":
        raise ValueError(f"event {ident}: material_partition requires explicit wall_floor fields")
    values = [spec[k] for k in ("floor_rho_start", "floor_rho_end")]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values
    ):
        raise ValueError(f"event {ident}: partition radii must be finite numbers, not bool")
    if not 0 <= values[0] < values[1] <= 1:
        raise ValueError(f"event {ident}: require 0 <= floor_rho_start < floor_rho_end <= 1")
    for region in ("floor", "wall"):
        validate_texture(spec[f"{region}_texture"], material, f"{ident}/{region}")


def wall_weight(rho: np.ndarray, start: float, end: float) -> np.ndarray:
    """Continuous smoothstep; thresholds are explicit hypothesis parameters."""
    t = np.clip((rho - start) / (end - start), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix_fields(
    current: np.ndarray,
    floor: np.ndarray,
    wall: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write a convex mix with an exact same-field limit on the original wins."""
    current[mask] = floor[mask] + weight[mask] * (wall[mask] - floor[mask])


def apply_partition(
    current: np.ndarray,
    rho: np.ndarray,
    wins: np.ndarray,
    spec: dict[str, Any],
    generate_region: Callable,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use event-before current for retain, including earlier events and patches."""
    fields, streams = {}, {}
    for region in ("floor", "wall"):
        field, provenance = generate_region(spec[f"{region}_texture"], region)
        fields[region] = current if field is None else field
        streams[region] = provenance
    weight = wall_weight(rho, spec["floor_rho_start"], spec["floor_rho_end"])
    # RHS is evaluated before current is changed, so two retain regions share the
    # same event-before field even though current is updated in place.
    mix_fields(current, fields["floor"], fields["wall"], weight, wins)
    floor_weight = np.where(wins, 1.0 - weight, 0.0)
    wall_weights = np.where(wins, weight, 0.0)
    return (
        floor_weight,
        wall_weights,
        {
            "config": copy.deepcopy(spec),
            "regions": streams,
            "calibration_status": "not_calibrated",
            "parameter_status": "explicit synthetic hypotheses; no measured wall/floor calibration",
            "retain_semantics": "texture current immediately before this event",
        },
    )


def weighted_statistics(
    values: np.ndarray, weight: np.ndarray, pixel_area: float
) -> dict[str, Any]:
    """Actual final-field statistics, centered RMS and Pearson kurtosis."""
    support = weight > 0
    v, w = values[support], weight[support]
    total = float(w.sum())
    stats = {
        "support_pixels": int(v.size),
        "effective_pixels": total,
        "weighted_area_m2": total * pixel_area,
        "mean_m": None,
        "rms_m": None,
        "skewness": None,
        "pearson_kurtosis": None,
    }
    if total == 0:
        return stats
    mean = float(np.sum(v * w) / total)
    centered = v - mean
    variance = float(np.sum(w * centered**2) / total)
    stats.update(mean_m=mean, rms_m=float(np.sqrt(variance)))
    if variance > 0:
        stats.update(
            skewness=float(np.sum(w * centered**3) / total / variance**1.5),
            pearson_kurtosis=float(np.sum(w * centered**4) / total / variance**2),
        )
    return stats


def final_partition_statistics(
    texture: np.ndarray,
    loss: np.ndarray,
    owner: np.ndarray,
    event_index: int,
    floor_weight: np.ndarray,
    wall_weights: np.ndarray,
    pixel_area: float,
) -> dict[str, Any]:
    """Measure final owner pixels, never the event's historical write mask."""
    owned = owner == event_index
    result: dict[str, Any] = {
        "scope": "final partition owner pixels after every event and patch",
        "owner_pixels": int(owned.sum()),
        "overall_texture": weighted_statistics(texture, owned.astype(float), pixel_area),
    }
    for region, weight in (("floor", floor_weight), ("wall", wall_weights)):
        final_weight = np.where(owned, weight, 0.0)
        result[region] = {
            "texture": weighted_statistics(texture, final_weight, pixel_area),
            "loss": weighted_statistics(loss, final_weight, pixel_area),
        }
    return result
