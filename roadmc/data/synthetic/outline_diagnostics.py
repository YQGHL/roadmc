"""Fair, frozen-coefficient diagnostics for observed pothole boundaries.

These functions describe same-view boundary interpolation.  XYZ units and
view independence remain unknown; missing boundary points are never imputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(old: np.ndarray, new: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    old, new = np.asarray(old, float), np.asarray(new, float)
    if old.shape != new.shape or old.ndim != 1:
        raise ValueError("score residuals must be one-dimensional arrays of equal length")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("score residuals must be finite")
    n = len(old)
    if not n:
        return {
            "points": 0,
            "old_mse": None,
            "new_mse": None,
            "mse_delta_new_minus_old": None,
            "old_rmse": None,
            "new_rmse": None,
            "rmse_change_percent": None,
        }
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    w = w / w.sum()
    old_mse, new_mse = float(np.sum(w * old**2)), float(np.sum(w * new**2))
    old_rmse, new_rmse = float(np.sqrt(old_mse)), float(np.sqrt(new_mse))
    return {
        "points": n,
        "old_mse": old_mse,
        "new_mse": new_mse,
        "mse_delta_new_minus_old": new_mse - old_mse,
        "old_rmse": old_rmse,
        "new_rmse": new_rmse,
        "rmse_change_percent": (100 * (new_rmse / old_rmse - 1) if old_rmse else None),
    }


def score_outline_boundary(
    projected_xy: np.ndarray,
    ellipse_parameters: list[float] | np.ndarray,
    coefficients: dict[str, list[float]],
    *,
    complete_edge_count: int | None = None,
    finite_xyz_edge_count: int | None = None,
    pixel_yx: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score one finite XYZ boundary using frozen ellipse and harmonic parameters.

    ``projected_xy`` is the same TLS-plane projection used by the native study.
    The four historical test sectors are retained (sector index modulo 5 == 0).
    The return arrays preserve one record per finite edge sample for NPZ audit.
    """
    xy = np.asarray(projected_xy, dtype=float)
    q = np.asarray(ellipse_parameters, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("projected_xy must be a finite (N,2) array")
    if q.shape != (5,) or not np.isfinite(q).all():
        raise ValueError("ellipse_parameters must contain five finite values")
    if not isinstance(coefficients, dict) or set(coefficients) != {"cos", "sin"}:
        raise ValueError("coefficients must contain cos and sin")
    ca, sa = np.asarray(coefficients["cos"], float), np.asarray(coefficients["sin"], float)
    if (
        ca.shape != (3,)
        or sa.shape != (3,)
        or not np.isfinite(ca).all()
        or not np.isfinite(sa).all()
    ):
        raise ValueError("cos/sin coefficients must each contain three finite values")
    amp = float(np.hypot(ca, sa).sum())
    if amp >= 1 or not np.isfinite(amp):
        raise ValueError("coefficient pair norm sum must be < 1")
    n = len(xy)
    if complete_edge_count is None:
        complete_edge_count = n
    if finite_xyz_edge_count is None:
        finite_xyz_edge_count = n
    if (
        isinstance(complete_edge_count, bool)
        or isinstance(finite_xyz_edge_count, bool)
        or not isinstance(complete_edge_count, (int, np.integer))
        or not isinstance(finite_xyz_edge_count, (int, np.integer))
        or complete_edge_count < 0
        or finite_xyz_edge_count < 0
        or finite_xyz_edge_count != n
        or complete_edge_count < finite_xyz_edge_count
    ):
        raise ValueError(
            "edge counts must satisfy 0 <= finite_xyz_edge_count == N <= complete_edge_count"
        )
    if pixel_yx is not None:
        pix = np.asarray(pixel_yx)
        if pix.shape != (n, 2) or not np.isfinite(pix).all():
            raise ValueError("pixel_yx must be a finite (N,2) array")
    else:
        pix = np.full((n, 2), -1, dtype=int)

    a, b = np.exp(q[2]), np.exp(q[3])
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("ellipse log axes must produce finite positive axes")
    shifted = xy - q[:2]
    ct, st = np.cos(q[4]), np.sin(q[4])
    uv = np.column_stack(
        (
            (ct * shifted[:, 0] + st * shifted[:, 1]) / a,
            (-st * shifted[:, 0] + ct * shifted[:, 1]) / b,
        )
    )
    theta = np.mod(np.arctan2(uv[:, 1], uv[:, 0]), 2 * np.pi)
    radius = np.linalg.norm(uv, axis=1)
    sector = np.floor(theta / (2 * np.pi) * 16).astype(int) if n else np.empty(0, dtype=int)
    test = sector % 5 == 0
    g = np.ones(n, dtype=float)
    for k, order in enumerate((3, 4, 5)):
        g += ca[k] * np.cos(order * theta) + sa[k] * np.sin(order * theta)
    g /= np.sqrt(1 + 0.5 * np.sum(ca * ca + sa * sa))
    if not np.isfinite(g).all() or np.any(g <= 0):
        raise ValueError("frozen coefficients yield an invalid positive radial scale")
    old_common, new_common = radius - 1, radius - g
    new_model = new_common / g
    s2 = a * a * np.cos(theta) ** 2 + b * b * np.sin(theta) ** 2
    scale = np.sqrt(s2)
    old_native, new_native = old_common * scale, new_common * scale

    common = _score(old_common[test], new_common[test])
    model = _score(old_common[test], new_model[test])
    native = _score(old_native[test], new_native[test])
    ntest = int(test.sum())
    delta = new_common**2 - old_common**2
    denominator_add = new_common**2 * (1 / g**2 - 1)
    mean_s2 = float(np.mean(s2[test])) if ntest else None
    anis_add = (s2[test] / mean_s2 - 1) * delta[test] if ntest else np.empty(0)
    denominator_sum = float(np.mean(denominator_add[test])) if ntest else None
    anis_sum = float(np.mean(anis_add)) if ntest else None
    common_delta = common["mse_delta_new_minus_old"]
    model_delta = model["mse_delta_new_minus_old"]
    native_delta = native["mse_delta_new_minus_old"]
    denom_error = (
        abs(common_delta + denominator_sum - model_delta) if common_delta is not None else None
    )
    anis_error = (
        abs(common_delta + anis_sum - native_delta / mean_s2) if common_delta is not None else None
    )

    sectors = []
    for k in range(16):
        m, mt = sector == k, (sector == k) & test
        count = int(m.sum())
        sectors.append(
            {
                "sector": k,
                "start_deg": k * 22.5,
                "end_deg": (k + 1) * 22.5,
                "split": "test" if k % 5 == 0 else "train",
                "points": count,
                "test_points": int(mt.sum()),
                "missing_valid_samples": count == 0,
                "all_boundary_share": (count / n if n else None),
                "test_common_mse_delta_contribution": (
                    float(delta[mt].sum() / ntest) if ntest else None
                ),
                "test_denominator_adjustment_contribution": (
                    float(denominator_add[mt].sum() / ntest) if ntest else None
                ),
                "test_anisotropy_adjustment_contribution": (
                    float(anis_add[sector[test] == k].sum() / ntest) if ntest else None
                ),
                "common": _score(old_common[m], new_common[m]),
            }
        )
    records = {
        "pixel_y": pix[:, 0].astype(int),
        "pixel_x": pix[:, 1].astype(int),
        "u": uv[:, 0],
        "v": uv[:, 1],
        "theta": theta,
        "radius": radius,
        "sector": sector,
        "test": test,
        "g": g,
        "ellipse_error_common": old_common,
        "harmonic_error_common": new_common,
        "harmonic_error_model_normalized": new_model,
        "ellipse_error_native_projected": old_native,
        "harmonic_error_native_projected": new_native,
        "native_scale_squared": s2,
    }
    result = {
        "units": "unknown_native_xyz; no conversion",
        "full_label_edge_points": int(complete_edge_count),
        "finite_xyz_edge_points": int(finite_xyz_edge_count),
        "missing_xyz_edge_points": int(complete_edge_count - finite_xyz_edge_count),
        "finite_edge_coverage": (
            float(finite_xyz_edge_count / complete_edge_count) if complete_edge_count else None
        ),
        "boundary_points_scored": n,
        "test_boundary_points": ntest,
        "sector_rule": "16 equal ellipse-angle sectors; sector index modulo 5 == 0 is test",
        "angles_with_observed_finite_edge": sorted(set(sector.tolist())),
        "empty_sectors": [s["sector"] for s in sectors if s["points"] == 0],
        "ellipse_parameters_center_logaxes_angle": q.tolist(),
        "harmonic_coefficients_frozen": {"cos": ca.tolist(), "sin": sa.tolist()},
        "scores": {
            "historical_model_normalized": model,
            "common_ellipse_coordinates": common,
            "native_projected_radial_error": native,
        },
        "mse_decomposition": {
            "definition": "model_delta=common_delta+mean((r-g)^2*(g^-2-1)); native_delta/mean(s2)=common_delta+mean((s2/mean(s2)-1)*common_point_delta)",
            "common_delta": common_delta,
            "denominator_adjustment": denominator_sum,
            "model_normalized_delta": model_delta,
            "denominator_identity_abs_error": denom_error,
            "mean_native_scale_squared": mean_s2,
            "anisotropy_adjustment_at_common_mean_scale": anis_sum,
            "native_delta_div_mean_native_scale_squared": (
                native_delta / mean_s2 if mean_s2 else None
            ),
            "anisotropy_identity_abs_error": anis_error,
        },
        "sectors": sectors,
        "boundary_quality": {
            "angle_coverage_count": len(set(sector.tolist())),
            "angle_coverage_fraction": len(set(sector.tolist())) / 16,
            "empty_angle_sectors": [s["sector"] for s in sectors if not s["points"]],
            "test_angle_sectors_empty": [
                s["sector"] for s in sectors if s["split"] == "test" and not s["points"]
            ],
            "interpretation": "Missing XYZ boundary samples remain missing; empty angular sectors and unknown units are recorded, not inferred.",
        },
    }
    return result, records


def run_pothole_outline_fairness_study(
    dataset_root: Path, output_dir: Path, frozen_metrics_path: Path
) -> dict[str, Any]:
    """Rebuild three observed boundaries and score verbatim frozen fits, with no optimizer."""
    from PIL import Image
    from scipy.io import loadmat
    from scipy.ndimage import binary_dilation, binary_erosion, label

    from roadmc.scripts import realism_experiments

    _json = realism_experiments._json

    root, out, frozen_path = Path(dataset_root), Path(output_dir), Path(frozen_metrics_path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {out}")
    if not frozen_path.is_file():
        raise FileNotFoundError(f"frozen metrics not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rows = frozen.get("frames")
    if not isinstance(rows, dict) or set(rows) != {"01", "02", "03"}:
        raise ValueError("frozen metrics must contain exactly frames 01, 02, and 03")
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__).resolve(),
        Path(realism_experiments.__file__).resolve(),
        frozen_path.resolve(),
    ]
    hashes: dict[str, str] = {str(p): _sha256(p) for p in source_paths}
    input_paths = [root / "ptcloud" / f"{f}.mat" for f in ("01", "02", "03")] + [
        root / "label" / f"{f}.png" for f in ("01", "02", "03")
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required native pothole input missing: {path}")
        hashes[str(path.resolve())] = _sha256(path)
    hashes_before = dict(hashes)
    result_frames: dict[str, Any] = {}
    csv_rows = []
    for frame in ("01", "02", "03"):
        cloud, lab = root / "ptcloud" / f"{frame}.mat", root / "label" / f"{frame}.png"
        row = rows[frame]
        frozen_source_hashes = row.get("source_sha256", {})
        source_identity = {}
        for path in (cloud, lab):
            expected = frozen_source_hashes.get(str(path))
            actual = hashes[str(path.resolve())]
            if expected is None:
                source_identity[str(path.resolve())] = "unknown_frozen_source_hash_not_recorded"
            elif expected != actual:
                raise ValueError(
                    f"input SHA-256 does not match frozen metrics for frame {frame}: {path}"
                )
            else:
                source_identity[str(path.resolve())] = "sha256_matches_frozen_metrics"
        try:
            q = row["ellipse_parameters_center_logaxes_angle"]
            coeff = row["outline_models"]["area_normalized_radial_harmonic"]["coefficients"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"frozen metrics missing ellipse or coefficients for frame {frame}"
            ) from exc
        xyz = np.asarray(loadmat(cloud, verify_compressed_data_integrity=True)["xyzPoints"], float)
        annotated = np.asarray(Image.open(lab).convert("RGB")).max(axis=-1) > 0
        if xyz.ndim != 3 or xyz.shape[:2] != annotated.shape or xyz.shape[2] != 3:
            raise ValueError(f"XYZ/label shape mismatch for frame {frame}")
        valid = np.isfinite(xyz).all(axis=-1)
        comps, count = label(annotated, structure=np.ones((3, 3), bool))
        if count <= 0:
            raise ValueError(f"no pothole annotation in frame {frame}")
        sizes = np.bincount(comps.ravel())
        sizes[0] = 0
        component = comps == int(sizes.argmax())
        ring = (
            binary_dilation(component, iterations=60)
            & ~binary_dilation(component, iterations=20)
            & valid
        )
        road = xyz[ring]
        if len(road) < 20:
            raise ValueError(f"insufficient TLS plane points for {frame}")
        road = road[:: max(1, len(road) // 100000)]
        origin = road.mean(axis=0)
        _, _, vh = np.linalg.svd(road - origin, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        tangent = vh[0] / np.linalg.norm(vh[0])
        other = np.cross(normal, tangent)
        other /= np.linalg.norm(other)
        tls_identity: dict[str, str] = {}
        for name, actual in (("plane_origin_native", origin), ("plane_normal", normal)):
            expected = row.get(name)
            if expected is None:
                tls_identity[name] = "unknown_frozen_tls_record_not_available"
            elif not np.allclose(actual, np.asarray(expected, float), rtol=0, atol=1e-10):
                raise ValueError(
                    f"reconstructed TLS {name} differs from frozen metrics for frame {frame}"
                )
            else:
                tls_identity[name] = "matches_frozen_metrics_within_1e-10"
        edge_all = component & ~binary_erosion(
            component, structure=np.ones((3, 3), bool), border_value=0
        )
        edge_valid = edge_all & valid
        pix = np.argwhere(edge_valid)
        points = xyz[edge_valid]
        projected = np.column_stack(((points - origin) @ tangent, (points - origin) @ other))
        scores, arrays = score_outline_boundary(
            projected,
            q,
            coeff,
            complete_edge_count=int(edge_all.sum()),
            finite_xyz_edge_count=int(edge_valid.sum()),
            pixel_yx=pix,
        )
        # Preserve the historical native-study fit score contract for direct comparison.
        old = row["outline_models"]["area_normalized_radial_harmonic"]["test"]
        if scores["scores"]["historical_model_normalized"]["points"] != old["points"]:
            raise ValueError(f"frozen boundary test count mismatch for frame {frame}")
        scores["historical_score_reproduction_abs_rmse_error"] = abs(
            scores["scores"]["historical_model_normalized"]["new_rmse"]
            - old["radial_relative_rmse"]
        )
        if scores["historical_score_reproduction_abs_rmse_error"] > 1e-12:
            raise ValueError(f"frozen score reproduction exceeded 1e-12 for frame {frame}")
        scores["frozen_source_identity"] = source_identity
        scores["frozen_tls_identity"] = tls_identity
        inputs_verified = all(
            value == "sha256_matches_frozen_metrics" for value in source_identity.values()
        )
        tls_verified = all(
            value == "matches_frozen_metrics_within_1e-10" for value in tls_identity.values()
        )
        scores["historical_score_reproduction_status"] = (
            "numeric_match_input_and_tls_verified"
            if inputs_verified and tls_verified
            else "numeric_match_only_input_or_tls_unverified"
        )
        result_frames[frame] = scores
        for sector in scores["sectors"]:
            csv_rows.append(
                {
                    "frame": frame,
                    **{key: value for key, value in sector.items() if key != "common"},
                    **{f"common_{key}": value for key, value in sector["common"].items()},
                }
            )
        np.savez_compressed(out / f"boundary_{frame}.npz", **arrays)
    hashes_after = {str(p): _sha256(Path(p)) for p in hashes_before}
    if hashes_after != hashes_before:
        raise RuntimeError("frozen source or one of the six pothole inputs changed during the run")
    all_identity_verified = all(
        row["historical_score_reproduction_status"] == "numeric_match_input_and_tls_verified"
        for row in result_frames.values()
    )
    _json(
        out / "metrics.json",
        {
            "status": "same_view_frozen_coefficient_descriptive_only",
            "historical_score_reproduction_status": (
                "all_three_numeric_scores_reproduced_with_source_and_tls_identity_verified"
                if all_identity_verified
                else "numeric_score_comparison_only; source_or_tls_identity_unverified"
            ),
            "frames": result_frames,
            "limits": [
                "XYZ units unknown",
                "views may be related",
                "no optimizer called",
                "no missing boundary values imputed",
            ],
        },
    )
    fields = sorted(set().union(*(r.keys() for r in csv_rows)))
    with (out / "sector_contributions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    _json(
        out / "provenance.json",
        {
            "frozen_metrics_path": str(frozen_path.resolve()),
            "frozen_metrics_sha256": hashes[str(frozen_path.resolve())],
            "dataset_root": str(root.resolve()),
            "input_and_source_sha256_before": hashes_before,
            "input_and_source_sha256_after": hashes_after,
            "all_inputs_and_source_unchanged": hashes_after == hashes_before,
            "frozen_source_and_tls_identity_verified_for_all_frames": all_identity_verified,
            "optimizer_called": False,
            "texture_rerun": False,
            "coefficient_source": "copied verbatim from frozen_metrics_path; no fit or tuning",
            "units": "unknown native XYZ scale",
        },
    )
    return {
        "output_dir": str(out),
        "metrics_path": str(out / "metrics.json"),
        "provenance_path": str(out / "provenance.json"),
        "frames": list(result_frames),
        "status": "same_view_frozen_coefficient_descriptive_only",
    }
