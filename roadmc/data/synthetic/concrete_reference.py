"""Explicit-unit KIT ring-profile diagnostics; no production default is changed.

The PSD returned here is a one-dimensional path PSD.  Circular acquisition
mixes directions and does not identify a two-dimensional radial surface PSD.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import RegularGridInterpolator


def load_ela_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the documented semicolon/mm/um format, returning SI arrays.

    Profile columns are paired measurements at one position, not independent
    roads.  No values are discarded or silently interpolated.
    """
    path = Path(path)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    if header != "Pos_mm;Prof1_µm;Prof2_µm":
        raise ValueError(f"unsupported ELA header: {header!r}")
    match = re.fullmatch(r"kit_csv_(A[56])_(L[12])_(M[1-5])_([LR])\.csv", path.name)
    if match is None:
        raise ValueError(f"unsupported ELA filename: {path.name}")
    values = np.loadtxt(path, delimiter=";", skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 8:
        raise ValueError("ELA requires at least eight rows and exactly three columns")
    if not np.isfinite(values).all():
        raise ValueError("ELA contains non-finite values")
    x_m = values[:, 0] * 1e-3
    dx = np.diff(x_m)
    if not np.all(dx > 0) or not np.allclose(dx, dx[0], rtol=0, atol=1e-12):
        raise ValueError("ELA positions must be uniformly increasing")
    road, location, position, half = match.groups()
    finish = "exposed_aggregate" if road == "A5" else (
        "drag_finish" if position in ("M2", "M3") else "broom_finish"
    )
    return x_m, values[:, 1:].T * 1e-6, {
        "road": road, "location": location, "position": position, "half": half,
        "finish": finish, "cluster": f"{road}_{location}_{position}",
        "sampling_m": float(dx[0]), "samples": len(values),
        "circumference_m": float(len(values) * dx[0]),
        "source_units": {"position": "mm", "height": "um"},
        "author_processing": "ISO13473-1:2019 filtered per paper p3 section2.3; A5 report header2004 unresolved",
    }


def sample_rings(
    x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray,
    centers_m: list[tuple[float, float]], *, circumference_m: float = 0.4,
    samples: int = 800, phase_rad: float = 0.0,
) -> np.ndarray:
    """Bilinearly sample closed circles with uniform arc-length increments.

    The first sample is at one increment, matching the 0.5..400mm source
    convention.  Centers and phase are explicit synthetic choices, not a
    registration with the real measurement.  Extrapolation is forbidden.
    """
    x, y, z = np.asarray(x_m), np.asarray(y_m), np.asarray(z_m)
    if (x.ndim != 1 or y.ndim != 1 or z.shape != (len(x), len(y))
            or not np.isfinite(z).all() or np.any(np.diff(x) <= 0)
            or np.any(np.diff(y) <= 0)):
        raise ValueError("ring sampling requires finite x-major increasing rectilinear grid")
    if samples < 8 or circumference_m <= 0 or not np.isfinite(circumference_m):
        raise ValueError("invalid ring dimensions")
    theta = phase_rad + 2 * np.pi * np.arange(1, samples + 1) / samples
    radius = circumference_m / (2 * np.pi)
    interpolate = RegularGridInterpolator((x, y), z, bounds_error=True)
    return np.stack([
        interpolate(np.column_stack((cx + radius * np.cos(theta),
                                     cy + radius * np.sin(theta))))
        for cx, cy in centers_m
    ])


def preprocess_ring(z_m: np.ndarray, dx_m: float, band_cycles_m=None) -> np.ndarray:
    """Remove constant and first harmonic (plane on ring), then optional band.

    A sharp periodic FFT mask is an explicit diagnostic transform, not an
    implementation or certification of an ISO filter.  It is applied equally
    to real and synthetic profiles, after the author's unknown exact filter.
    """
    z = np.asarray(z_m, dtype=np.float64)
    if z.ndim != 1 or len(z) < 8 or not np.isfinite(z).all() or dx_m <= 0:
        raise ValueError("requires a finite 1D profile and positive spacing")
    theta = 2 * np.pi * np.arange(1, len(z) + 1) / len(z)
    design = np.column_stack((np.ones(len(z)), np.cos(theta), np.sin(theta)))
    z = z - design @ np.linalg.lstsq(design, z, rcond=None)[0]
    if band_cycles_m is not None:
        lo, hi = band_cycles_m
        if not 0 <= lo < hi <= 0.5 / dx_m + 1e-9:
            raise ValueError("band must be increasing within the profile Nyquist limit")
        f = np.fft.rfftfreq(len(z), dx_m)
        z = np.fft.irfft(np.fft.rfft(z) * ((f >= lo) & (f <= hi)), n=len(z))
    return z


def describe_ring(z_m: np.ndarray, dx_m: float) -> dict:
    """Physical RMS, standardized quantiles, Welch path PSD and circular ACF."""
    z = np.asarray(z_m, dtype=np.float64)
    z = z - z.mean()
    rms = float(np.sqrt(np.mean(z * z)))
    if z.ndim != 1 or len(z) < 400 or rms <= 0 or not np.isfinite(z).all():
        raise ValueError("description requires a non-degenerate finite 1D profile >=400 samples")
    std = z / rms
    f, p = signal.welch(z, fs=1 / dx_m, window="hann", nperseg=400,
                        noverlap=200, detrend=False, scaling="density")
    acf = np.fft.irfft(np.abs(np.fft.rfft(z)) ** 2, n=len(z))
    acf = acf[:101] / acf[0]
    return {
        "rms_m": rms, "skewness": float(np.mean(std ** 3)),
        "pearson_kurtosis": float(np.mean(std ** 4)),
        "quantile_probabilities": np.linspace(0.01, 0.99, 99).tolist(),
        "standardized_quantiles": np.quantile(std, np.linspace(0.01, 0.99, 99)).tolist(),
        "height_quantiles_m": np.quantile(z, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
        "frequency_cycles_m": f.tolist(), "psd_m3": p.tolist(),
        "acf_lag_m": (np.arange(101) * dx_m).tolist(), "circular_acf": acf.tolist(),
    }


def aggregate_descriptors(descriptors: list[dict]) -> dict:
    """Equal-weight descriptors; caller first groups paired profiles/halves."""
    if not descriptors:
        raise ValueError("cannot aggregate an empty group")
    return {key: np.mean([d[key] for d in descriptors], axis=0).tolist()
            for key in descriptors[0]}


def compare_descriptors(reference: dict, candidate: dict, band=(20.0, 500.0)) -> dict:
    """Separate signed amplitude, marginal, PSD and ACF errors; no total score."""
    f = np.asarray(reference["frequency_cycles_m"])
    keep = (f >= band[0]) & (f <= band[1])
    rp, cp = np.asarray(reference["psd_m3"]), np.asarray(candidate["psd_m3"])
    rn, cn = rp / rp[keep].sum(), cp / cp[keep].sum()
    floor = 1e-12 / keep.sum()
    return {
        "rms_ratio": float(candidate["rms_m"] / reference["rms_m"]),
        "rms_error_mm_signed": float(1000 * (candidate["rms_m"] - reference["rms_m"])),
        "standardized_quantile_mae": float(np.mean(np.abs(
            np.asarray(candidate["standardized_quantiles"])
            - reference["standardized_quantiles"]))),
        "normalized_psd_log10_mae": float(np.mean(np.abs(
            np.log10(np.maximum(rn[keep], floor))
            - np.log10(np.maximum(cn[keep], floor))))),
        "physical_psd_log10_mae": float(np.mean(np.abs(
            np.log10(np.maximum(rp[keep], 1e-30))
            - np.log10(np.maximum(cp[keep], 1e-30))))),
        "acf_mae_0_to_50mm": float(np.mean(np.abs(
            np.asarray(candidate["circular_acf"]) - reference["circular_acf"]))),
    }
