"""Offline shape-only calibration helpers for procedural asphalt textures.

This module is deliberately separate from the production texture generator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np


def canonical_array_sha256(array: np.ndarray) -> str:
    """Hash a parsed finite array independent of source CSV formatting."""
    values = np.asarray(array, dtype="<f8", order="C")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("array must be a finite 2D matrix")
    header = f"float64-le:{values.shape[0]}x{values.shape[1]}\0".encode("ascii")
    return hashlib.sha256(header + values.tobytes(order="C")).hexdigest()


def deduplicate_manifest(records: Iterable[Mapping]) -> tuple[list[dict], list[dict]]:
    """Keep all unique training files, then unique test files without leakage.

    Repeated train records are collapsed by hash; any test record sharing a raw
    or parsed-array hash with train is excluded. Test duplicates are likewise
    collapsed deterministically in input order.
    """
    accepted: list[dict] = []
    excluded: list[dict] = []
    train_raw: set[str] = set()
    train_array: set[str] = set()
    seen_train_raw: set[str] = set()
    seen_train_array: set[str] = set()
    seen_test_raw: set[str] = set()
    seen_test_array: set[str] = set()
    ordered = sorted(
        (dict(r) for r in records),
        key=lambda r: (0 if r["split"] == "train" else 1, r["class"], r["member"]),
    )
    for row in ordered:
        if row["split"] == "train":
            train_raw.add(row["raw_sha256"])
            train_array.add(row["array_sha256"])
            if row["raw_sha256"] in seen_train_raw or row["array_sha256"] in seen_train_array:
                row["exclusion_reason"] = "duplicate_within_train"
                excluded.append(row)
            else:
                seen_train_raw.add(row["raw_sha256"])
                seen_train_array.add(row["array_sha256"])
                accepted.append(row)
        elif row["raw_sha256"] in train_raw or row["array_sha256"] in train_array:
            row["exclusion_reason"] = "duplicate_of_train_prevents_split_leakage"
            excluded.append(row)
        elif row["raw_sha256"] in seen_test_raw or row["array_sha256"] in seen_test_array:
            row["exclusion_reason"] = "duplicate_within_test"
            excluded.append(row)
        else:
            seen_test_raw.add(row["raw_sha256"])
            seen_test_array.add(row["array_sha256"])
            accepted.append(row)
    return accepted, excluded


def remove_hash_matches_to_excluded_tests(
    records: Iterable[Mapping], excluded_tests: Iterable[Mapping]
) -> tuple[list[dict], list[dict]]:
    """Remove test records matching any pre-excluded test by either hash."""
    blocked_raw = {r["raw_sha256"] for r in excluded_tests}
    blocked_array = {r["array_sha256"] for r in excluded_tests}
    kept, excluded = [], []
    for source in records:
        row = dict(source)
        if row.get("split") == "test" and (
            row["raw_sha256"] in blocked_raw or row["array_sha256"] in blocked_array
        ):
            row["exclusion_reason"] = (
                "duplicate_of_previously_explored_test_0_or_1_prevents_holdout_contamination"
            )
            excluded.append(row)
        else:
            kept.append(row)
    return kept, excluded


def radial_cosine_gain(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, dict]:
    """Return the fixed radial analysis gain used for every field."""
    nx, ny = shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    xx, yy = np.meshgrid(fx, fy, indexing="ij")
    radius = np.hypot(xx, yy)
    gain = np.ones(shape, dtype=np.float64)
    gain[radius >= 500.0] = 0.0
    taper = (radius > 400.0) & (radius < 500.0)
    gain[taper] = 0.5 * (1.0 + np.cos(np.pi * (radius[taper] - 400.0) / 100.0))
    return gain, {
        "type": "radial cosine low-pass",
        "gain_1_through_cycles_m": 400.0,
        "cosine_taper_cycles_m": [400.0, 500.0],
        "gain_0_from_cycles_m_inclusive": 500.0,
        "grid_shape_x_major": [nx, ny],
        "dx_m": dx,
        "dy_m": dy,
    }


def preprocess_shape(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Plane detrend, apply fixed analysis low-pass, and RMS standardize."""
    a = np.asarray(field, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 3 or not np.isfinite(a).all():
        raise ValueError("field must be a finite 2D array with each dimension >= 3")
    yy, xx = np.indices(a.shape, dtype=np.float64)
    design = np.column_stack((np.ones(a.size), xx.ravel(), yy.ravel()))
    beta = np.linalg.lstsq(design, a.ravel(), rcond=None)[0]
    detrended = a - (design @ beta).reshape(a.shape)
    gain, _ = radial_cosine_gain(a.shape, dx, dy)
    filtered = np.fft.ifft2(np.fft.fft2(detrended) * gain).real
    rms = float(np.sqrt(np.mean(filtered * filtered)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("preprocessing produced zero or invalid RMS")
    return filtered / rms


def radial_psd_mean(
    field: np.ndarray, dx: float, dy: float, fmax: float = 500.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial annular-mean 2D periodogram density on a fixed grid."""
    a = np.asarray(field, dtype=np.float64)
    nx, ny = a.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    xx, yy = np.meshgrid(fx, fy, indexing="ij")
    radius = np.hypot(xx, yy)
    df = 1.0 / (max(nx * dx, ny * dy))
    edges = np.arange(0.0, fmax + df * 1.0001, df)
    power = np.abs(np.fft.fft2(a)) ** 2 / (nx * ny)
    ids = np.digitize(radius.ravel(), edges) - 1
    vals, centers = [], []
    for k in range(1, len(edges) - 1):
        center = (edges[k] + edges[k + 1]) / 2.0
        if center > fmax:
            continue
        band = power.ravel()[ids == k]
        if band.size:
            vals.append(float(band.mean()))
            centers.append(float(center))
    return np.asarray(centers), np.asarray(vals)


def acf_xy(field: np.ndarray, maxlag: int = 45) -> np.ndarray:
    """Linear, zero-padded normalized positive x then y axis ACF."""
    a = np.asarray(field, dtype=np.float64)
    nx, ny = a.shape
    if maxlag >= min(nx, ny):
        raise ValueError("maxlag must be smaller than both field dimensions")
    corr = np.fft.ifft2(np.abs(np.fft.fft2(a, s=(2 * nx, 2 * ny))) ** 2).real
    corr /= corr[0, 0]
    return np.r_[corr[1 : maxlag + 1, 0], corr[0, 1 : maxlag + 1]]


def directional_marginal_density(
    field: np.ndarray, axis: int, dx: float, dy: float, fmax: float = 500.0
) -> np.ndarray:
    """Mean line-periodogram density along an x-major spatial axis."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (X) or 1 (Y) for an x-major field")
    a = np.asarray(field, dtype=np.float64)
    spacing = dx if axis == 0 else dy
    f = np.fft.rfftfreq(a.shape[axis], d=spacing)
    power = np.abs(np.fft.rfft(a, axis=axis)) ** 2
    power = power.mean(axis=1 - axis)
    keep = (f > 0.0) & (f <= fmax)
    result = power[keep]
    total = float(result.sum())
    return result / total if total > 0 else np.full(result.shape, np.nan)


def shape_descriptors(field: np.ndarray, dx: float, dy: float) -> dict:
    """Descriptors used for fitting and separately reported morphology."""
    a = np.asarray(field, dtype=np.float64)
    f, p = radial_psd_mean(a, dx, dy)
    total = float(p.sum())
    return {
        "radial_frequency_cycles_m": f,
        "radial_psd_density": p / total if total > 0 else np.full(p.shape, np.nan),
        "acf_xy": acf_xy(a),
        "values": a,
    }


def descriptor_scores(reference: Mapping, candidate: Mapping) -> dict:
    p = reference["radial_psd_density"]
    q = candidate["radial_psd_density"]
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid or mismatched radial PSD descriptors")
    a, b = reference["acf_xy"], candidate["acf_xy"]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid or mismatched ACF descriptors")
    return {
        "radial_psd_tv": float(0.5 * np.abs(p - q).sum()),
        "acf_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "fit_score": float(0.5 * np.abs(p - q).sum() + np.sqrt(np.mean((a - b) ** 2))),
    }


def select_setting(rows: Iterable[Mapping]) -> dict:
    """Select by training mean score; exact ties prefer lower band then H."""
    values = [dict(r) for r in rows]
    usable = [
        r
        for r in values
        if r.get("split", "train") == "train"
        and ("expected_rows" not in r or r.get("valid_rows") == r["expected_rows"])
        and r.get("mean_fit_score") is not None
        and np.isfinite(r["mean_fit_score"])
    ]
    if not usable:
        raise ValueError("no valid training candidate scores")
    return min(usable, key=lambda r: (r["mean_fit_score"], r["band_hi_cycles_m"], r["hurst"]))


def moments(field: np.ndarray) -> dict[str, float]:
    a = np.asarray(field, dtype=np.float64).ravel()
    z = (a - a.mean()) / a.std()
    return {"skew": float(np.mean(z**3)), "kurt_pearson": float(np.mean(z**4))}
