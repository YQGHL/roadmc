"""Optional area-normalized radial warp of an ellipse, an engineering hypothesis.

Harmonics 3..5 add low-order irregularity; coefficients are not population priors.
The strict amplitude bound guarantees a positive scale and a one-to-one radial warp.
"""

from __future__ import annotations

import numpy as np


def validate_radial_harmonics(spec) -> dict[str, list[float]]:
    """Validate strict JSON lists a3..a5/b3..b5 and sum hypot(a_n,b_n)<1."""
    if not isinstance(spec, dict) or set(spec) != {"cos", "sin"}:
        raise ValueError("radial_harmonics requires cos and sin only")
    for key in ("cos", "sin"):
        if not isinstance(spec[key], list) or len(spec[key]) != 3:
            raise ValueError(
                "radial_harmonics cos/sin must be JSON lists of length 3 (orders 3..5)"
            )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in spec[key]
        ):
            raise ValueError("radial_harmonics values must be finite JSON numbers")
    cosine, sine = np.asarray(spec["cos"], dtype=float), np.asarray(spec["sin"], dtype=float)
    with np.errstate(over="ignore"):
        amplitude_bound = float(np.hypot(cosine, sine).sum())
    if not np.isfinite(amplitude_bound) or amplitude_bound >= 1.0:
        raise ValueError("radial_harmonics requires sum hypot(a_n,b_n) < 1 for positivity")
    return {"cos": cosine.tolist(), "sin": sine.tolist()}


def radial_scale(theta, spec) -> np.ndarray:
    """Return positive g(theta), with angular mean(g²)=1 and ellipse area πab."""
    cfg = validate_radial_harmonics(spec)
    theta = np.asarray(theta, dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("radial angle must be finite")
    a, b = np.asarray(cfg["cos"]), np.asarray(cfg["sin"])
    result = np.ones(theta.shape, dtype=float)
    if not np.any(a) and not np.any(b):
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        for index, order in enumerate((3, 4, 5)):
            result += a[index] * np.cos(order * theta) + b[index] * np.sin(order * theta)
    result /= np.sqrt(1 + 0.5 * np.sum(a * a + b * b))
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("radial harmonic evaluation is numerically unsupported")
    return result


def warp_normalized_coordinates(u, v, spec) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map ellipse coordinates to the unit disk; use these for footprint and bowl.

    The inverse is (u,v)=(u',v')*g(atan2(v',u')); the angular coordinate is unchanged.
    """
    cfg = validate_radial_harmonics(spec)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("normalized ellipse coordinates must be finite")
    scale = radial_scale(np.arctan2(v, u), cfg)
    a, b = np.asarray(cfg["cos"]), np.asarray(cfg["sin"])
    denominator = float(np.sqrt(1 + 0.5 * np.sum(a * a + b * b)))
    return (
        u / scale,
        v / scale,
        {
            "kind": "area_normalized_radial_harmonics",
            "orders": [3, 4, 5],
            "radial_harmonics": cfg,
            "normalization_denominator": denominator,
            "normalization": "g=(1+sum(a_n cos(n theta)+b_n sin(n theta)))/sqrt(1+0.5 sum(a_n²+b_n²)); angular mean(g²)=1",
            "continuous_area": "pi * semi_major_m * semi_minor_m",
            "coordinate_warp": "u_prime=u/g(theta), v_prime=v/g(theta); shared footprint/bowl coordinates",
            "positivity_amplitude_bound": float(np.hypot(a, b).sum()),
            "status": "engineering_hypothesis_not_population_prior",
        },
    )
