"""Optional dimensionless bowl profiles; shape parameters are hypotheses.

The footprint remains independently defined. This adds no learned size/depth
prior, aggregate fracture simulation or claim of a validated damage mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def validate_bowl_profile(spec: Mapping) -> dict[str, float | str]:
    if not isinstance(spec, Mapping) or set(spec) != {
        "kind",
        "edge_power",
        "tilt_major",
        "tilt_minor",
    }:
        raise ValueError("bowl profile requires kind/edge_power/tilt_major/tilt_minor only")
    if spec["kind"] != "tilted_power_bowl":
        raise ValueError("unsupported pothole profile kind")
    values = {}
    for key in ("edge_power", "tilt_major", "tilt_minor"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("bowl parameters must be finite JSON numbers")
        values[key] = float(value)
    if values["edge_power"] <= 0:
        raise ValueError("edge_power must be positive")
    _peak(values["edge_power"], values["tilt_major"], values["tilt_minor"])
    return {"kind": "tilted_power_bowl", **values}


def _peak(power: float, major: float, minor: float) -> tuple[float, float, float]:
    norm = float(np.hypot(major, minor))
    if not np.isfinite(norm):
        raise ValueError("tilt norm exceeds the numerically supported range")
    if norm == 0:
        return 0.0, 0.0, 0.0
    scale = max(power, norm)
    p, k = power / scale, norm / scale
    radius = k / (np.hypot(p, k) + p)
    if not 0 <= radius < 1:
        raise ValueError("tilt/power contrast exceeds the numerically supported range")
    log_peak = power * np.log1p(-(radius**2)) + norm * radius
    if not np.isfinite(log_peak):
        raise ValueError("bowl normalization is not finite")
    return radius * major / norm, radius * minor / norm, float(log_peak)


def bowl_profile(u: np.ndarray, v: np.ndarray, spec: Mapping) -> tuple[np.ndarray, dict]:
    """Unit-maximum profile (1-u²-v²)^p exp(ku*u+kv*v), zero outside.

    The maximum is normalized analytically, not by the sampled grid maximum;
    sampling phase therefore does not change the underlying requested depth.
    """
    cfg = validate_bowl_profile(spec)
    p, ku, kv = cfg["edge_power"], cfg["tilt_major"], cfg["tilt_minor"]
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("finite normalized coordinates required")
    with np.errstate(over="ignore"):
        r2 = u**2 + v**2
    inside = r2 < 1
    peak_u, peak_v, log_peak = _peak(p, ku, kv)
    result = np.zeros(u.shape, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        log_value = p * np.log1p(-r2[inside]) + ku * u[inside] + kv * v[inside] - log_peak
        result[inside] = np.exp(np.minimum(log_value, 0))
    if not np.isfinite(result).all():
        raise ValueError("bowl profile contains nonfinite values")
    return result, {
        "kind": cfg["kind"],
        "dimensionless_parameters": cfg,
        "normalized_peak_major_minor": [peak_u, peak_v],
        "continuous_maximum": 1.0,
        "sampled_maximum": float(result.max()) if result.size else None,
        "normalization": "analytic continuous maximum; no grid-dependent rescaling",
        "status": "engineering_hypothesis_not_population_prior",
    }
