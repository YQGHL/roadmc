"""Explicit, metric procedural texture fields for regular road-surface grids.

The module generates an uncalibrated random field. Material identifiers are
provenance labels only; they never select hidden material-specific values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["generate_texture", "load_settings_json"]

_REQUIRED_KEYS = {
    "rms_m",
    "hurst",
    "anisotropy_ratio",
    "orientation_deg",
    "band_lo_cycles_m",
    "band_hi_cycles_m",
    "distribution",
    "material_id",
}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"shape", "spectrum_profile", "spectral_matching"}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_settings_json(path: str | Path) -> dict[str, Any]:
    """Load strict JSON settings, rejecting duplicate keys and NaN/Infinity."""
    raw = Path(path).read_text(encoding="utf-8")
    value = json.loads(
        raw,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("settings JSON root must be an object")
    return value


def _finite_number(settings: Mapping[str, Any], key: str) -> float:
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _canonical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a JSON object-like mapping")
    if any(not isinstance(key, str) for key in settings):
        raise ValueError("settings keys must be strings")
    keys = set(settings)
    profile_mode = "spectrum_profile" in settings
    required = _REQUIRED_KEYS - {"hurst"} if profile_mode else _REQUIRED_KEYS
    missing = required - keys
    extra = keys - _ALLOWED_KEYS
    if missing:
        raise ValueError(f"settings missing required keys: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"unsupported settings keys: {', '.join(sorted(extra))}")
    if profile_mode and "hurst" in settings:
        raise ValueError("hurst and spectrum_profile are mutually exclusive")

    values = {
        key: _finite_number(settings, key)
        for key in (
            "rms_m",
            "hurst",
            "anisotropy_ratio",
            "orientation_deg",
            "band_lo_cycles_m",
            "band_hi_cycles_m",
        )
        if key != "hurst" or not profile_mode
    }
    if values["rms_m"] < 0.0:
        raise ValueError("rms_m must be >= 0")
    if not profile_mode and values["hurst"] <= 0.0:
        raise ValueError("hurst must be > 0")
    if values["anisotropy_ratio"] <= 0.0:
        raise ValueError("anisotropy_ratio must be > 0")
    if values["band_lo_cycles_m"] < 0.0:
        raise ValueError("band_lo_cycles_m must be >= 0")
    if values["band_hi_cycles_m"] <= values["band_lo_cycles_m"]:
        raise ValueError("band_hi_cycles_m must be greater than band_lo_cycles_m")

    distribution = settings["distribution"]
    if distribution not in ("gaussian", "non_gaussian"):
        raise ValueError("distribution must be 'gaussian' or 'non_gaussian'")
    material_id = settings["material_id"]
    if material_id not in ("asphalt", "concrete"):
        raise ValueError("material_id must be 'asphalt' or 'concrete'")
    has_shape = "shape" in settings
    if distribution == "non_gaussian" and not has_shape:
        raise ValueError("non_gaussian distribution requires explicit shape parameters")
    if distribution == "gaussian" and has_shape:
        raise ValueError("shape is only valid when distribution is 'non_gaussian'")

    result: dict[str, Any] = {
        **values,
        "distribution": distribution,
        "material_id": material_id,
    }
    if profile_mode:
        profile = settings["spectrum_profile"]
        if not isinstance(profile, Mapping) or set(profile) != {
            "frequency_cycles_m",
            "relative_psd",
        }:
            raise ValueError("spectrum_profile requires frequency_cycles_m and relative_psd only")
        frequency, psd = profile["frequency_cycles_m"], profile["relative_psd"]
        if (
            not isinstance(frequency, list)
            or not isinstance(psd, list)
            or not 2 <= len(frequency) == len(psd) <= 4096
        ):
            raise ValueError("spectral profile arrays must have equal lengths in 2..4096")
        for value in frequency + psd:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError("spectral profile values must be finite JSON numbers")
        f, p = np.asarray(frequency, dtype=float), np.asarray(psd, dtype=float)
        if f[0] < 0 or np.any(np.diff(f) <= 0) or np.any(p < 0) or not np.any(p > 0):
            raise ValueError(
                "profile frequencies must increase; PSD must be nonnegative and nonzero"
            )
        if values["band_lo_cycles_m"] < f[0] or values["band_hi_cycles_m"] > f[-1]:
            raise ValueError("requested band must stay within explicit spectral-profile support")
        result["spectrum_profile"] = {"frequency_cycles_m": f.tolist(), "relative_psd": p.tolist()}
    if has_shape:
        shape = settings["shape"]
        if not isinstance(shape, Mapping) or set(shape) != {"skew", "kurt"}:
            raise ValueError("shape must contain exactly 'skew' and 'kurt'")
        skew = _finite_number(shape, "skew")
        kurt = _finite_number(shape, "kurt")
        if kurt < 1.0:
            raise ValueError("shape.kurt must be >= 1 (Pearson kurtosis)")
        result["shape"] = {"skew": skew, "kurt": kurt}
    if "spectral_matching" in settings:
        match = settings["spectral_matching"]
        if (
            distribution != "non_gaussian"
            or not isinstance(match, Mapping)
            or set(match) != {"iterations", "tolerance"}
        ):
            raise ValueError(
                "spectral_matching requires non_gaussian and explicit iterations/tolerance"
            )
        iterations = match["iterations"]
        tolerance = _finite_number(match, "tolerance")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= 500
            or tolerance <= 0
        ):
            raise ValueError("matching iterations must be 1..500; tolerance must be positive")
        result["spectral_matching"] = {"iterations": iterations, "tolerance": tolerance}
    return result


def _uniform_axis(axis: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a 1D axis with at least two coordinates")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    steps = np.diff(values)
    if np.any(steps <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    spacing = float(np.mean(steps))
    tolerance = max(abs(spacing) * 1e-9, np.finfo(np.float64).eps * 32.0)
    if not np.allclose(steps, spacing, rtol=1e-9, atol=tolerance):
        raise ValueError(f"{name} must be uniformly spaced")
    return values, spacing


def _moment_statistics(field: np.ndarray) -> dict[str, float | None]:
    mean = float(np.mean(field))
    centered = np.asarray(field, dtype=np.float64) - mean
    rms = float(np.sqrt(np.mean(centered * centered)))
    if rms == 0.0:
        skew = None
        kurt = None
    else:
        normalized = centered / rms
        skew = float(np.mean(normalized**3))
        kurt = float(np.mean(normalized**4))
    return {"mean_m": mean, "rms_m": rms, "skew": skew, "kurt_pearson": kurt}


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def generate_texture(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a zero-mean x-major height field and its audit metadata.

    Coordinates and returned heights use meters; frequencies use cycles/m.
    The field is periodic over the sampled rectangle because it is built by
    a discrete Fourier series. RMS is the demeaned spatial standard deviation.
    """
    cfg = _canonical_settings(settings)
    x, dx = _uniform_axis(x_axis, "x_axis")
    y, dy = _uniform_axis(y_axis, "y_axis")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nx, ny = len(x), len(y)
    nyquist_x = 0.5 / dx
    nyquist_y = 0.5 / dy
    axis_nyquist = min(nyquist_x, nyquist_y)
    high = cfg["band_hi_cycles_m"]
    if high > axis_nyquist * (1.0 + 1e-12):
        raise ValueError(
            "band_hi_cycles_m exceeds the conservative axis Nyquist limit "
            f"({axis_nyquist:.12g} cycles/m for dx={dx:.12g}, dy={dy:.12g})"
        )

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="ij")
    theta = np.deg2rad(cfg["orientation_deg"])
    frequency_u = fx_grid * np.cos(theta) + fy_grid * np.sin(theta)
    frequency_v = -fx_grid * np.sin(theta) + fy_grid * np.cos(theta)
    ratio = cfg["anisotropy_ratio"]
    radial = np.hypot(fx_grid, fy_grid)
    if ratio == 1.0:
        # Avoid angle-dependent roundoff in the isotropic case.
        q = radial
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                q = np.sqrt(
                    (frequency_u / np.sqrt(ratio)) ** 2 + (frequency_v * np.sqrt(ratio)) ** 2
                )
        except FloatingPointError as exc:
            raise ValueError("anisotropy_ratio is outside the numerically supported range") from exc
        if not np.isfinite(q).all():
            raise ValueError("anisotropy_ratio produced non-finite spectral coordinates")
    band = (radial >= cfg["band_lo_cycles_m"]) & (radial <= high)
    band[0, 0] = False  # Preserve zero DC for every distribution.
    if not np.any(band):
        raise ValueError("requested frequency band contains no non-DC grid frequencies")

    if cfg["rms_m"] == 0.0:
        field = np.zeros((nx, ny), dtype=np.float64)
        projection_ratio = 0.0
    else:
        rng = np.random.default_rng(int(seed))
        white = rng.standard_normal((nx, ny))
        spectrum = np.fft.fft2(white)
        amplitude = np.zeros_like(radial)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                if "spectrum_profile" in cfg:
                    profile = cfg["spectrum_profile"]
                    relative = np.asarray(profile["relative_psd"])
                    relative = relative / relative.max()
                    amplitude[band] = np.sqrt(
                        np.interp(q[band], profile["frequency_cycles_m"], relative, left=0, right=0)
                    )
                else:
                    amplitude[band] = q[band] ** (-(cfg["hurst"] + 1.0))
        except FloatingPointError as exc:
            raise ValueError("settings produced numerically unsupported spectral weights") from exc
        if not np.any(amplitude > 0):
            raise ValueError("spectral profile has no positive weights on the requested grid/band")
        shaped = spectrum * amplitude
        shaped[0, 0] = 0.0
        gaussian_field = np.fft.ifft2(shaped).real

        field = gaussian_field
        projection_ratio = 0.0
        if cfg["distribution"] == "non_gaussian":
            from .am_envelope import _sinh_arcsinh_transform

            shape = cfg["shape"]
            try:
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    transformed = _sinh_arcsinh_transform(
                        gaussian_field, shape["skew"], shape["kurt"]
                    )
            except FloatingPointError as exc:
                raise ValueError("shape parameters produced non-finite values") from exc
            if not np.isfinite(transformed).all():
                raise ValueError("shape parameters produced non-finite values")
            before_spectrum = np.fft.fft2(transformed)
            before_power = float(np.sum(np.abs(before_spectrum) ** 2))
            projected_spectrum = np.where(band, before_spectrum, 0.0)
            projected_spectrum[0, 0] = 0.0
            after_power = float(np.sum(np.abs(projected_spectrum) ** 2))
            projection_ratio = (
                max(before_power - after_power, 0.0) / before_power if before_power > 0.0 else 0.0
            )
            field = np.fft.ifft2(projected_spectrum).real
            if "spectral_matching" in cfg:
                from .spectral_synthesis import match_marginal_spectrum

                field, matching_metadata = match_marginal_spectrum(
                    gaussian_field, transformed, **cfg["spectral_matching"]
                )

        field -= float(np.mean(field))
        current_rms = float(np.sqrt(np.mean(field * field)))
        if not np.isfinite(current_rms) or current_rms <= 0.0:
            raise ValueError("requested band and grid produced a degenerate texture field")
        field *= cfg["rms_m"] / current_rms

    spectrum_out = np.fft.fft2(field)
    total_power = float(np.sum(np.abs(spectrum_out) ** 2))
    outside_power = float(np.sum(np.abs(spectrum_out[~band]) ** 2))
    metadata: dict[str, Any] = {
        "algorithm": "radial_profile_fft_v1"
        if "spectrum_profile" in cfg
        else "anisotropic_powerlaw_fft_v1",
        "algorithm_sha256": _module_sha256(),
        "seed": int(seed),
        "actual_settings": cfg,
        "material_id": cfg["material_id"],
        "uncalibrated": True,
        "units": {"coordinates": "m", "height": "m", "frequency": "cycles/m"},
        "grid": {"shape_x_major": [nx, ny], "dx_m": dx, "dy_m": dy},
        "nyquist_cycles_m": {
            "x": nyquist_x,
            "y": nyquist_y,
            "conservative_axis_limit": axis_nyquist,
        },
        "spectrum": {
            "power_law": (
                None
                if "spectrum_profile" in cfg
                else "PSD proportional to q^(-(2H+2)); amplitude proportional to q^(-(H+1))"
            ),
            "rotated_frequency": "u=fx*cos(theta)+fy*sin(theta), v=-fx*sin(theta)+fy*cos(theta)",
            "anisotropic_radius": "q=sqrt((fu/sqrt(ratio))^2+(fv*sqrt(ratio))^2)",
            "radial_band_mask_cycles_m": [
                cfg["band_lo_cycles_m"],
                cfg["band_hi_cycles_m"],
            ],
            "selected_bin_frequency_min_cycles_m": float(np.min(radial[band])),
            "selected_bin_frequency_max_cycles_m": float(np.max(radial[band])),
            "dc_power": float(abs(spectrum_out[0, 0]) ** 2),
            "out_of_band_energy_ratio": outside_power / total_power if total_power else 0.0,
            "non_gaussian_projection_removed_power_ratio": projection_ratio,
        },
        "statistics_of_synthetic_output": _moment_statistics(field),
        "rms_definition": "demeaned spatial standard deviation; field is centered and rescaled to requested rms_m",
    }
    if "spectrum_profile" in cfg:
        metadata["spectrum"]["profile_interpolation"] = (
            "linear frequency / relative PSD; zero outside knot support"
        )
    if "spectral_matching" in cfg:
        metadata["spectral_matching"] = (
            matching_metadata if cfg["rms_m"] > 0 else {"status": "not_run_zero_rms"}
        )
    return field, metadata
