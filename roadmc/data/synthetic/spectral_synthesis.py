"""Explicit spectral statistics and 2-D marginal/spectrum matching.

Radial Hann window: Jacobs, Junge & Pastewka (2017),
10.1088/2051-672X/aa51f8, section 3.1. Alternating amplitude/rank projection:
Schreiber & Schmitz (1996), 10.1103/PhysRevLett.77.635, adapted here to 2-D
procedurally generated fields. Neither method establishes pavement realism.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def radial_hann_window(shape: tuple[int, int], dx: float, dy: float) -> np.ndarray:
    """Return a physically radial Hann with discrete mean squared gain one."""
    if len(shape) != 2 or min(shape) < 3 or not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("window requires a >=3x3 grid and finite positive spacing")
    nx, ny = shape
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    radius = np.hypot(x[:, None], y[None, :])
    support = min(nx * dx, ny * dy) / 2
    window = np.where(radius < support, 0.5 * (1 + np.cos(np.pi * radius / support)), 0)
    return window / np.sqrt(np.mean(window**2))


def windowed_radial_psd(field: np.ndarray, dx: float, dy: float, fmax: float) -> dict:
    """Hann-windowed 2-D density in cycles/m; annular means and powers differ.

    PSD = dx*dy/(nx*ny)*abs(FFT(window*field))**2. Its integral over the
    full 2-D frequency grid equals mean((window*field)**2), by Parseval.
    Radial bins exclude DC; partial last bins contain frequencies <= fmax.
    """
    a = np.asarray(field, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(fmax) or fmax <= 0:
        raise ValueError("finite field and positive fmax required")
    window = radial_hann_window(a.shape, dx, dy)
    nx, ny = a.shape
    fx, fy = np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy)
    q = np.hypot(fx[:, None], fy[None, :])
    power = dx * dy / a.size * np.abs(np.fft.fft2(a * window)) ** 2
    df = 1 / max(nx * dx, ny * dy)
    keep = (q > 0) & (q <= fmax)
    bins = np.floor(q[keep] / df).astype(int)
    counts = np.bincount(bins)
    sums = np.bincount(bins, weights=power[keep], minlength=len(counts))
    frequency_sums = np.bincount(bins, weights=q[keep], minlength=len(counts))
    valid = counts > 0
    return {
        "frequency_cycles_m": frequency_sums[valid] / counts[valid],
        "mean_density": sums[valid] / counts[valid],
        "ring_integrated_power": sums[valid] / (nx * dx * ny * dy),
        "bin_counts": counts[valid],
        "window_mean_square": float(np.mean(window**2)),
        "windowed_height_mean_square": float(np.mean((a * window) ** 2)),
        "full_psd_integral": float(power.sum() / (nx * dx * ny * dy)),
        "frequency_bin_width_cycles_m": df,
        "window": "radial_hann_discrete_energy_normalized",
    }


def _standardize(a: np.ndarray) -> np.ndarray:
    centered = np.asarray(a, dtype=np.float64) - np.mean(a)
    rms = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(centered).all() or not np.isfinite(rms) or rms <= 0:
        raise ValueError("spectral matching requires a nonconstant finite field")
    return centered / rms


def match_marginal_spectrum(
    gaussian: np.ndarray, transformed: np.ndarray, *, iterations: int, tolerance: float
) -> tuple[np.ndarray, dict]:
    """Match a *procedurally generated* marginal to generated Fourier amplitudes.

    Final projection prioritizes the original Fourier amplitudes and strict
    frequency support. Marginal agreement is approximate and explicitly audited.
    The best marginal-error projection is retained, including iteration zero;
    convergence is never inferred just because the iteration limit was reached.
    """
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 500
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("iterations must be 1..500; tolerance must be finite and positive")
    g, current = _standardize(gaussian), _standardize(transformed)
    if g.shape != current.shape or g.ndim != 2:
        raise ValueError("matching fields must have the same 2-D shape")
    target_sorted = np.sort(current.ravel())
    reference = np.fft.fft2(g)
    target_amplitude = np.abs(reference)
    reference_phase = np.divide(
        reference, target_amplitude, out=np.ones_like(reference), where=target_amplitude > 0
    )
    target_amplitude[0, 0] = 0

    def project(values):
        f = np.fft.fft2(values)
        amp = np.abs(f)
        phase = np.divide(f, amp, out=reference_phase.copy(), where=amp > 0)
        return np.fft.ifft2(target_amplitude * phase).real

    def marginal_error(values):
        return float(np.sqrt(np.mean((np.sort(values.ravel()) - target_sorted) ** 2)))

    best = project(current)
    initial_error = best_error = marginal_error(best)
    best_step = 0
    history = [initial_error]
    actual = 0
    for step in range(1, iterations + 1):
        projected = project(current)
        order = np.argsort(projected.ravel(), kind="stable")
        ranked = np.empty(current.size, dtype=np.float64)
        ranked[order] = target_sorted
        candidate = project(ranked.reshape(current.shape))
        error = marginal_error(candidate)
        history.append(error)
        actual = step
        if error < best_error:
            best, best_error, best_step = candidate, error, step
        if best_error <= tolerance or np.array_equal(ranked.reshape(current.shape), current):
            break
        current = ranked.reshape(current.shape)
    amplitude_error = float(
        np.linalg.norm(np.abs(np.fft.fft2(best)) - target_amplitude)
        / np.linalg.norm(target_amplitude)
    )
    return best, {
        "method": "2d_iaaft_final_spectral_projection",
        "algorithm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "10.1103/PhysRevLett.77.635; 2-D extension is a modelling choice",
        "target_marginal_source": "sorted parametric transform of generated field; no measured pixels",
        "iterations_requested": iterations,
        "iterations_run": actual,
        "selected_iteration": best_step,
        "marginal_tolerance": tolerance,
        "marginal_tolerance_reached": best_error <= tolerance,
        "initial_standardized_sorted_rmse": initial_error,
        "final_standardized_sorted_rmse": best_error,
        "relative_fourier_amplitude_error": amplitude_error,
        "marginal_error_history": history,
        "final_projection": "spectrum exact to floating precision; marginal remains approximate",
    }
