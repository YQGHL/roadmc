"""Fixed, calibration-only protocols for high-precision road-domain evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from roadmc.data.real.metadata import metadata_sidecar_path
from roadmc.data.real.raw import RawRoadScene, load_raw_scene
from roadmc.domain_gap import PointCloudRecord, observable_descriptors

PROTOCOL_SCHEMA = "roadmc.domain_protocol.v1"
DEFAULT_PROTOCOL = {
    "seed": 20261013,
    "coordinate_units": "m",
    "roi_extent_m": [2.0, 2.0],
    "neighbors": 16,
    "neighborhood_radius_m": 0.05,
    "physical_radius_query_cap": 4096,
    "max_points_per_scene": 4096,
    "texture_band_lo_cycles_m": 10.0,
    "texture_band_hi_cycles_m": 100.0,
    "texture_mpd_baseline_m": 0.1,
    "acf_lags_m": [0.01, 0.025, 0.05, 0.1],
    "mmd_max_roads": 512,
    "road_split": [0.6, 0.2, 0.2],
    "minimum_roads_per_reference_group": 5,
    "bootstrap_repetitions": 500,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("manifest JSON must contain a 'samples' array")
    output = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"manifest sample {index} needs a path")
        item = dict(entry)
        item["path"] = str((manifest_path.parent / item["path"]).resolve())
        item.setdefault("domain", "real")
        item.setdefault("role", "unassigned")
        item.setdefault("sample_id", Path(item["path"]).stem)
        sidecar = metadata_sidecar_path(item["path"])
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                for key in ("road_segment_id", "material_id", "sensor", "coordinate_units", "intensity_scale", "intensity_min", "intensity_max", "coordinate_origin", "sensor_metadata"):
                    if key not in item and key in metadata:
                        item[key] = metadata[key]
        item["sha256"] = sha256_file(item["path"])
        output.append(item)
    return output


def split_real_roads(
    samples: list[dict[str, Any]], *, seed: int = DEFAULT_PROTOCOL["seed"]
) -> dict[str, Any]:
    """Assign entire roads to 60/20/20 calibration/validation/holdout groups."""
    real = [item for item in samples if item.get("domain") == "real"]
    roads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for item in real:
        road = item.get("road_segment_id")
        if not road:
            missing.append(item.get("sample_id", item["path"]))
        else:
            roads[str(road)].append(item)
    road_ids = sorted(roads)
    min_total = int(np.ceil(DEFAULT_PROTOCOL["minimum_roads_per_reference_group"] / 0.2))
    if missing or len(road_ids) < min_total:
        return {
            "status": "insufficient_data",
            "reason": f"60/20/20 road split needs at least {min_total} roads to provide five per reference group; no road_segment_id may be missing",
            "missing_road_id_samples": missing,
            "road_count": len(road_ids),
            "samples": samples,
        }
    rng = np.random.default_rng(seed)
    order = [road_ids[i] for i in rng.permutation(len(road_ids))]
    n_cal = max(1, int(np.floor(0.6 * len(order))))
    n_validation = max(1, int(np.floor(0.2 * len(order))))
    if n_cal + n_validation >= len(order):
        n_cal = len(order) - 2
        n_validation = 1
    assigned = dict.fromkeys(order[:n_cal], "calibration")
    assigned.update(dict.fromkeys(order[n_cal:n_cal + n_validation], "real_validation"))
    assigned.update(dict.fromkeys(order[n_cal + n_validation:], "real_holdout"))
    result_samples = []
    for item in samples:
        clone = dict(item)
        if clone.get("domain") == "real":
            clone["role"] = assigned.get(str(clone.get("road_segment_id")), "unassigned")
        result_samples.append(clone)
    return {
        "status": "ready",
        "seed": seed,
        "road_assignment": assigned,
        "counts_by_road": {role: sum(value == role for value in assigned.values()) for role in set(assigned.values())},
        "samples": result_samples,
    }


def _manifest_metadata(item: dict[str, Any], scene: RawRoadScene) -> dict[str, Any]:
    return {**scene.metadata, **{key: value for key, value in item.items() if key not in {"path"}}}


def _load_sample(item: dict[str, Any], point_key: str | None = None) -> tuple[RawRoadScene, dict[str, Any]]:
    scene = load_raw_scene(item["path"], point_key=point_key, metadata_overrides=item)
    metadata = _manifest_metadata(item, scene)
    road = item.get("road_segment_id") or metadata.get("road_segment_id")
    material = item.get("material_id") or metadata.get("material_id") or "unspecified"
    return scene, {**metadata, "road_segment_id": road, "material_id": material}


def _grid_metrics(scene: RawRoadScene, protocol: dict[str, Any]) -> dict[str, float]:
    grid = scene.grid_z_m
    spacing = scene.grid_spacing_m
    if grid is None or spacing is None or not np.isfinite(grid).all():
        return {}
    dx, dy = spacing
    nyquist = min(0.5 / dx, 0.5 / dy)
    lo = float(protocol["texture_band_lo_cycles_m"])
    hi = float(protocol["texture_band_hi_cycles_m"])
    common_band_supported = hi > lo and nyquist >= hi
    nx, ny = grid.shape
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    design = np.column_stack((np.ones(grid.size), xx.ravel(), yy.ravel()))
    coeff, *_ = np.linalg.lstsq(design, grid.ravel(), rcond=None)
    residual = grid - (coeff[0] + coeff[1] * xx + coeff[2] * yy)
    out: dict[str, float] = {}
    for q in (5, 25, 50, 75, 95):
        out[f"elevation_q{q:02d}_m"] = float(np.percentile(residual, q))
    out["elevation_rms_m"] = float(np.sqrt(np.mean(residual**2)))
    bands = ((lo, min(hi, 25.0)), (min(max(lo, 25.0), hi), hi)) if common_band_supported else ()
    for axis, (axis_spacing, data) in enumerate(((dx, residual.T), (dy, residual))):
        n = data.shape[-1]
        freq = np.fft.rfftfreq(n, d=axis_spacing)
        spectrum = np.fft.rfft(data, axis=-1)
        power = np.mean(np.abs(spectrum) ** 2, axis=0) / max(n**2, 1)
        suffix = "x" if axis == 0 else "y"
        for band_index, (band_lo, band_hi) in enumerate(bands):
            if band_hi <= band_lo:
                continue
            mask = (freq >= band_lo) & (freq < band_hi)
            if mask.any():
                out[f"texture_log_psd_{suffix}_band{band_index}_m2"] = float(np.log(max(power[mask].mean(), 1e-30)))
        for lag_m in protocol["acf_lags_m"]:
            lag = int(round(float(lag_m) / axis_spacing))
            if lag < 1 or lag >= n:
                continue
            left, right = data[..., :-lag], data[..., lag:]
            denom = np.sqrt(np.mean(left**2) * np.mean(right**2))
            if denom > 0.0:
                out[f"texture_acf_{suffix}_{lag_m:g}m"] = float(np.mean(left * right) / denom)
    mpd = _estimate_mpd(residual, dx, dy, lo, hi, float(protocol["texture_mpd_baseline_m"])) if common_band_supported else None
    if mpd is not None:
        out["mpd_m"] = mpd
    return out


def _estimate_mpd(grid: np.ndarray, dx: float, dy: float, lo: float, hi: float, baseline: float) -> float | None:
    def along(profiles: np.ndarray, spacing: float) -> float | None:
        n = profiles.shape[-1]
        frequencies = np.fft.rfftfreq(n, d=spacing)
        keep = (frequencies > lo) & (frequencies <= min(hi, 0.5 / spacing))
        if not keep.any():
            return None
        filtered = np.fft.irfft(np.where(keep, np.fft.rfft(profiles, axis=-1), 0.0), n=n, axis=-1)
        seg_n = int(round(baseline / spacing))
        half = seg_n // 2
        if seg_n < 4 or half < 2 or n // seg_n == 0:
            return None
        x = (np.arange(seg_n) - (seg_n - 1) / 2.0) * spacing
        denom = float(x @ x)
        values = []
        for start in range(0, n - seg_n + 1, seg_n):
            seg = filtered[..., start:start + seg_n]
            slope = np.sum(seg * x, axis=-1) / denom
            detrended = seg - slope[..., None] * x
            detrended -= detrended.mean(axis=-1, keepdims=True)
            values.append(0.5 * (detrended[..., :half].max(axis=-1) + detrended[..., half:2 * half].max(axis=-1)))
        return float(np.concatenate([v.ravel() for v in values]).mean())

    y_value = along(grid, dy)
    x_value = along(grid.T, dx)
    values = [value for value in (x_value, y_value) if value is not None]
    return float(np.mean(values)) if values else None


def _scene_descriptors(scene: RawRoadScene, protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    scene, roi_status = _apply_roi(scene, protocol["roi_extent_m"])
    if roi_status["status"] != "available":
        return {}, {}
    if int(np.isfinite(scene.points_m).all(axis=1).sum()) < 4:
        return {}, {}
    record = PointCloudRecord(scene.points_m, scene.intensity, scene.normals, name=Path(scene.source_path).name)
    observable = observable_descriptors(
        record,
        k_neighbors=int(protocol["neighbors"]),
        max_points=int(protocol["max_points_per_scene"]),
        seed=int(protocol["seed"]),
        normal_source="pca",
    )
    grid_metrics = _grid_metrics(scene, protocol)
    scalar_summaries: dict[str, float] = {}
    for name, values in observable.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            continue
        for suffix, value in (("mean", np.mean(finite)), ("std", np.std(finite)), ("q10", np.quantile(finite, 0.1)), ("q50", np.quantile(finite, 0.5)), ("q90", np.quantile(finite, 0.9))):
            scalar_summaries[f"observable_{name}_{suffix}"] = float(value)
    scalar_summaries.update(grid_metrics)
    physical_counts = _physical_neighbor_counts(
        scene.points_m,
        float(protocol["neighborhood_radius_m"]),
        int(protocol["physical_radius_query_cap"]),
        int(protocol["seed"]),
    )
    if physical_counts.size:
        for suffix, value in (("mean", np.mean(physical_counts)), ("std", np.std(physical_counts)), ("q50", np.quantile(physical_counts, 0.5)), ("q90", np.quantile(physical_counts, 0.9))):
            scalar_summaries[f"physical_neighbors_{protocol['neighborhood_radius_m']:g}m_{suffix}"] = float(value)
    return observable, scalar_summaries


def _physical_neighbor_counts(points: np.ndarray, radius_m: float, query_cap: int, seed: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    finite_points = np.asarray(points, dtype=np.float64)
    finite_points = finite_points[np.isfinite(finite_points).all(axis=1)]
    if len(finite_points) < 2:
        return np.empty(0, dtype=np.float64)
    tree = cKDTree(finite_points)
    if len(finite_points) > query_cap:
        rng = np.random.default_rng(seed)
        query = finite_points[rng.choice(len(finite_points), size=query_cap, replace=False)]
    else:
        query = finite_points
    # return_length avoids materializing dense neighbor lists; every query is
    # an actual cloud point, so subtract exactly one self-neighbor.
    counts = tree.query_ball_point(query, radius_m, return_length=True, workers=1)
    return np.maximum(np.asarray(counts, dtype=np.float64) - 1.0, 0.0)


def _apply_roi(scene: RawRoadScene, extent: list[float]) -> tuple[RawRoadScene, dict[str, Any]]:
    points = scene.points_m
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return scene, {"status": "unavailable", "reason": "no finite points for ROI coverage"}
    center = np.median(points[finite, :2], axis=0)
    half = np.asarray(extent, dtype=np.float64) / 2.0
    source_extent = np.ptp(points[finite, :2], axis=0)
    tolerance = np.asarray(scene.grid_spacing_m or (0.0, 0.0), dtype=np.float64) + 1e-9
    coverage = np.minimum(source_extent / np.asarray(extent, dtype=np.float64), 1.0)
    if np.any(source_extent + tolerance < np.asarray(extent, dtype=np.float64)):
        return scene, {"status": "unavailable", "reason": "scene extent is smaller than the fixed ROI", "actual_extent_m": source_extent.tolist(), "requested_extent_m": list(extent), "coverage_fraction": coverage.tolist()}
    keep = finite & (np.abs(points[:, 0] - center[0]) <= half[0]) & (np.abs(points[:, 1] - center[1]) <= half[1])
    if int(keep.sum()) < 4:
        return replace(scene, points_m=points[keep], intensity=scene.intensity[keep] if scene.intensity is not None else None, labels=scene.labels[keep] if scene.labels is not None else None, normals=scene.normals[keep] if scene.normals is not None else None, grid_z_m=None, grid_spacing_m=None), {"status": "available", "actual_extent_m": source_extent.tolist(), "requested_extent_m": list(extent), "coverage_fraction": coverage.tolist()}
    grid = scene.grid_z_m
    if grid is not None and scene.grid_spacing_m is not None:
        dx, dy = scene.grid_spacing_m
        nx, ny = grid.shape
        gx = (np.arange(nx) - (nx - 1) / 2.0) * dx
        gy = (np.arange(ny) - (ny - 1) / 2.0) * dy
        ix = np.flatnonzero(np.abs(gx) <= half[0])
        iy = np.flatnonzero(np.abs(gy) <= half[1])
        grid = grid[np.ix_(ix, iy)] if len(ix) >= 2 and len(iy) >= 2 else None
    return replace(scene, points_m=points[keep], intensity=scene.intensity[keep] if scene.intensity is not None else None, intensity_raw=scene.intensity_raw[keep] if scene.intensity_raw is not None else None, labels=scene.labels[keep] if scene.labels is not None else None, normals=scene.normals[keep] if scene.normals is not None else None, grid_z_m=grid), {"status": "available", "actual_extent_m": source_extent.tolist(), "requested_extent_m": list(extent), "coverage_fraction": coverage.tolist()}


def fit_protocol(samples: list[dict[str, Any]], *, protocol_options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = {**DEFAULT_PROTOCOL, **(protocol_options or {})}
    unknown_options = set(options) - set(DEFAULT_PROTOCOL)
    if unknown_options:
        raise ValueError(f"unknown protocol settings: {sorted(unknown_options)}")
    if options["coordinate_units"] != "m":
        raise ValueError("protocol coordinate_units is fixed to meters")
    if len(options["roi_extent_m"]) != 2 or any(float(value) <= 0.0 for value in options["roi_extent_m"]):
        raise ValueError("roi_extent_m must contain two positive dimensions")
    if float(options["texture_band_hi_cycles_m"]) <= float(options["texture_band_lo_cycles_m"]):
        raise ValueError("texture_band_hi_cycles_m must exceed texture_band_lo_cycles_m")
    if list(options["road_split"]) != [0.6, 0.2, 0.2]:
        raise ValueError("road_split is fixed at [0.6, 0.2, 0.2]")
    calibration = [item for item in samples if item.get("role") == "calibration" and item.get("domain") == "real"]
    if len(calibration) < 2:
        return {"schema": PROTOCOL_SCHEMA, "status": "not_calibrated", "reason": "at least two real calibration samples are required", "options": options}
    loaded_calibration = [(item, *_load_sample(item)) for item in calibration]
    supported_nyquist = [min(0.5 / scene.grid_spacing_m[0], 0.5 / scene.grid_spacing_m[1]) for _, scene, _ in loaded_calibration if scene.grid_spacing_m is not None and scene.grid_z_m is not None]
    if supported_nyquist:
        options["texture_band_hi_cycles_m"] = min(float(options["texture_band_hi_cycles_m"]), min(supported_nyquist))
    groups: dict[str, list[tuple[dict[str, Any], RawRoadScene, dict[str, np.ndarray], dict[str, float]]]] = defaultdict(list)
    for item, scene, metadata in loaded_calibration:
        observable, summaries = _scene_descriptors(scene, options)
        groups[str(metadata["material_id"])].append((item, scene, observable, summaries))
    materials: dict[str, Any] = {}
    source_hashes = []
    for material, rows in sorted(groups.items()):
        common_features = set(rows[0][3])
        for _, _, _, summary in rows[1:]:
            common_features.intersection_update(summary)
        names = sorted(common_features)
        road_values: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            road_key = str(row[0].get("road_segment_id") or row[0]["sample_id"])
            if names:
                road_values[road_key].append(np.asarray([row[3][name] for name in names], dtype=np.float64))
        road_matrix = np.asarray([np.mean(values, axis=0) for _, values in sorted(road_values.items())], dtype=np.float64) if names else np.empty((0, 0))
        if len(names) and len(road_matrix) >= 2:
            center = road_matrix.mean(axis=0)
            scale = road_matrix.std(axis=0)
            scale[scale < 1e-12] = 1.0
            normalized = (road_matrix - center) / scale
            if len(normalized) > int(options["mmd_max_roads"]):
                rng = np.random.default_rng(int(options["seed"]))
                selected = rng.choice(len(normalized), size=int(options["mmd_max_roads"]), replace=False)
                normalized = normalized[selected]
            distances = _pairwise_distances(normalized)
            positive = distances[distances > 0.0]
            bandwidth = float(np.median(positive)) if positive.size else 1.0
            gamma = 1.0 / max(2.0 * bandwidth**2, 1e-12)
            mmd_features = names
        else:
            center, scale, gamma, mmd_features = np.array([]), np.array([]), None, []
        observable_available = sorted(set.intersection(*(set(row[2]) for row in rows))) if rows else []
        materials[material] = {
            "status": "calibrated" if names else "not_calibrated",
            "reason": None if names else "no descriptors are jointly available across real calibration samples",
            "descriptor_names": names,
            "normalization": {"center": center.tolist(), "scale": scale.tolist()},
            "mmd_gamma": gamma,
            "mmd_feature_names": mmd_features,
            "observable_w1_names": observable_available,
            "calibration_sample_ids": [row[0]["sample_id"] for row in rows],
            "calibration_roads": sorted({str(row[0]["road_segment_id"]) for row in rows if row[0].get("road_segment_id")}),
            "calibration_road_missing": [row[0].get("sample_id", row[0]["path"]) for row in rows if not row[0].get("road_segment_id")],
            "calibration_source_hashes": [row[0]["sha256"] for row in rows],
            "calibration_sources": [{"sample_id": row[0]["sample_id"], "path": row[0]["path"], "sha256": row[0]["sha256"], "road_segment_id": row[0].get("road_segment_id"), "material_id": material} for row in rows],
            "calibration_quality_missing": [row[0].get("sample_id", row[0]["path"]) for row in rows if not _quality_eligible(row[0])],
            "calibration_rows": road_matrix.tolist(),
        }
        source_hashes.extend(row[0]["sha256"] for row in rows)
    protocol = {"schema": PROTOCOL_SCHEMA, "status": "calibrated", "options": options, "materials": materials, "calibration_source_hashes": source_hashes}
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def inspect_synthetic_candidates(samples: list[dict[str, Any]], *, protocol_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report candidate-only support without fitting calibration or an acceptance gate."""
    options = {**DEFAULT_PROTOCOL, **(protocol_options or {})}
    unknown_options = set(options) - set(DEFAULT_PROTOCOL)
    if unknown_options:
        raise ValueError(f"unknown protocol settings: {sorted(unknown_options)}")
    if options["coordinate_units"] != "m" or list(options["road_split"]) != [0.6, 0.2, 0.2]:
        raise ValueError("inspect uses fixed meter units and the fixed 60/20/20 road split contract")
    if len(options["roi_extent_m"]) != 2 or any(float(value) <= 0.0 for value in options["roi_extent_m"]):
        raise ValueError("roi_extent_m must contain two positive dimensions")
    candidates = [item for item in samples if item.get("domain") == "synthetic" and item.get("role") == "candidate"]
    inspected = []
    for item in candidates:
        scene, metadata = _load_sample(item)
        roi = _roi_coverage(scene, options["roi_extent_m"])
        observable, summaries = _scene_descriptors(scene, options)
        inspected.append({
            "sample_id": item.get("sample_id"),
            "path": item["path"],
            "sha256": item.get("sha256"),
            "material_id": metadata.get("material_id"),
            "road_segment_id": metadata.get("road_segment_id"),
            "status": "available" if summaries else "not_evaluated",
            "reason": None if summaries else "fixed ROI or descriptor inputs are insufficient",
            "roi_coverage": roi,
            "available_observable_descriptors": sorted(observable),
            "available_summary_descriptors": sorted(summaries),
            "candidate_summary_values": summaries,
            "grid_texture_unavailable_reason": _grid_unavailable_reason(scene, options),
        })
    return {
        "status": "diagnostic_only",
        "evaluation_scope": "synthetic candidate support inspection only; no calibration, real-domain distance, or acceptance gate",
        "disease_geometry_status": "not_evaluated",
        "real_sample_count": sum(item.get("domain") == "real" for item in samples),
        "synthetic_candidate_count": len(candidates),
        "protocol_options": options,
        "candidates": inspected,
    }


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    delta = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def save_protocol(protocol: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"unsupported protocol schema: {protocol.get('schema')!r}")
    expected = protocol.get("protocol_hash")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if expected and canonical_hash(unhashed) != expected:
        raise ValueError("protocol hash does not match its contents")
    return protocol


def evaluate_manifest(samples: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "calibrated":
        return {
            "status": "not_evaluated",
            "protocol_status": "not_calibrated",
            "protocol_hash": protocol.get("protocol_hash"),
            "reason": protocol.get("reason", "no real calibration samples are available"),
            "real_sample_count": sum(item.get("domain") == "real" for item in samples),
            "synthetic_candidate_count": sum(item.get("domain") == "synthetic" and item.get("role") == "candidate" for item in samples),
            "evaluation_scope": "synthetic diagnostic inputs listed; no calibrated real-domain comparison",
            "disease_geometry_status": "not_evaluated",
            "input_sources": [{"path": item["path"], "sha256": item.get("sha256"), "domain": item.get("domain"), "role": item.get("role"), "material_id": item.get("material_id"), "road_segment_id": item.get("road_segment_id")} for item in samples],
        }
    loaded: list[dict[str, Any]] = []
    for item in samples:
        scene, metadata = _load_sample(item)
        observable, summaries = _scene_descriptors(scene, protocol["options"])
        loaded.append({"manifest": item, "scene": scene, "metadata": metadata, "observable": observable, "summaries": summaries})
    by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_material[str(row["metadata"]["material_id"])].append(row)
    reports = {}
    for material, rows in sorted(by_material.items()):
        fitted = protocol["materials"].get(material)
        candidate = [row for row in rows if row["manifest"].get("role") == "candidate"]
        holdout = [row for row in rows if row["manifest"].get("role") == "real_holdout" and row["manifest"].get("domain") == "real"]
        validation = [row for row in rows if row["manifest"].get("role") == "real_validation" and row["manifest"].get("domain") == "real"]
        if fitted is None:
            reports[material] = {"status": "not_calibrated", "reason": "material has no fitted real calibration protocol"}
            continue
        names = fitted["descriptor_names"]
        result: dict[str, Any] = {
            "status": "not_evaluated",
            "reason": "candidate, calibrated descriptors, or independent real holdout group is absent",
            "candidate_count": len(candidate),
            "real_holdout_count": len(holdout),
            "metrics": {},
        }
        candidate_descriptor_rows = [row for row in candidate if any(name in row["summaries"] for name in names)]
        holdout_descriptor_rows = [row for row in holdout if any(name in row["summaries"] for name in names)]
        result["candidate_descriptor_coverage"] = len(candidate_descriptor_rows) / max(len(candidate), 1)
        result["holdout_descriptor_coverage"] = len(holdout_descriptor_rows) / max(len(holdout), 1)
        if candidate_descriptor_rows and holdout_descriptor_rows and names:
            result["status"] = "evaluated"
            result["reason"] = None
            for name in fitted["observable_w1_names"]:
                source_values = [entry["observable"][name] for entry in candidate if name in entry["observable"]]
                target_values = [entry["observable"][name] for entry in holdout if name in entry["observable"]]
                left = np.concatenate(source_values) if source_values else np.empty(0)
                right = np.concatenate(target_values) if target_values else np.empty(0)
                result["metrics"][f"observable_w1_{name}"] = {
                    "value": float(wasserstein_distance(left, right)) if left.size and right.size else None,
                    "status": "available" if left.size and right.size else "not_evaluated",
                    "candidate_coverage": len(source_values) / len(candidate),
                    "holdout_coverage": len(target_values) / len(holdout),
                }
            result["metrics"].update(_scalar_distance_metrics(candidate, holdout, names, protocol["options"]))
            result["metrics"]["fixed_joint_mmd"] = _fixed_mmd(candidate, holdout, fitted, protocol["options"])
        result["real_real_reference"] = _real_reference(validation, holdout, fitted, protocol["options"])
        result["acceptance_gate"] = _acceptance_gate(result["metrics"], result["real_real_reference"], candidate, holdout, protocol["options"], fitted)
        reports[material] = result
    return {
        "status": "evaluated" if any(item.get("status") == "evaluated" for item in reports.values()) else "not_evaluated",
        "evaluation_scope": "surface texture and observable point-cloud statistics only; no disease-specific geometry evaluation",
        "disease_geometry_status": "not_evaluated",
        "protocol_hash": protocol.get("protocol_hash"),
        "protocol_source_hashes": protocol.get("calibration_source_hashes", []),
        "materials": reports,
        "input_sources": [
            {"path": row["manifest"]["path"], "sha256": row["manifest"]["sha256"], "role": row["manifest"].get("role"), "domain": row["manifest"].get("domain"), "road_segment_id": row["metadata"].get("road_segment_id"), "material_id": row["metadata"].get("material_id"), "sensor": row["metadata"].get("sensor"), "sensor_metadata": row["metadata"].get("sensor_metadata"), "source_units": row["scene"].source_units, "coordinate_units_status": row["scene"].metadata.get("coordinate_units_status"), "coordinate_origin_m": row["scene"].coordinate_origin_m, "intensity_calibration": row["metadata"].get("intensity_scale"), "intensity_calibration_status": row["metadata"].get("intensity_calibration_status"), "labels_available": row["scene"].labels is not None, "available_observable_descriptors": sorted(row["observable"]), "available_summary_descriptors": sorted(row["summaries"]), "grid_texture_available": row["scene"].grid_z_m is not None, "grid_texture_unavailable_reason": _grid_unavailable_reason(row["scene"], protocol["options"]), "roi_coverage": _roi_coverage(row["scene"], protocol["options"]["roi_extent_m"]), "range_field_available": row["scene"].range_values is not None, "validity_policy": row["scene"].metadata.get("point_validity_policy")}
            for row in loaded
        ],
    }


def _grid_unavailable_reason(scene: RawRoadScene, options: dict[str, Any]) -> str | None:
    if scene.grid_z_m is None or scene.grid_spacing_m is None:
        return "no complete regular grid; no interpolation was applied"
    nyquist = min(0.5 / scene.grid_spacing_m[0], 0.5 / scene.grid_spacing_m[1])
    if nyquist < float(options["texture_band_hi_cycles_m"]):
        return "grid Nyquist is below the frozen common-band upper limit; texture band was not shortened"
    return None


def _roi_coverage(scene: RawRoadScene, extent: list[float]) -> dict[str, Any]:
    points = scene.points_m
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return {"status": "unavailable", "reason": "no finite points"}
    actual = np.ptp(points[finite, :2], axis=0)
    tolerance = np.asarray(scene.grid_spacing_m or (0.0, 0.0), dtype=np.float64) + 1e-9
    requested = np.asarray(extent, dtype=np.float64)
    return {"status": "available" if np.all(actual + tolerance >= requested) else "insufficient_extent", "actual_extent_m": actual.tolist(), "requested_extent_m": requested.tolist(), "coverage_fraction": np.minimum(actual / requested, 1.0).tolist()}


def _scalar_distance_metrics(candidate: list[dict[str, Any]], holdout: list[dict[str, Any]], names: list[str], options: dict[str, Any]) -> dict[str, Any]:
    result = {}
    left_roads = _road_means(candidate, names)
    right_roads = _road_means(holdout, names)
    rng = np.random.default_rng(int(options["seed"]))
    repetitions = int(options["bootstrap_repetitions"])
    for name in names:
        left = np.array([row[name] for row in left_roads.values() if name in row], dtype=np.float64)
        right = np.array([row[name] for row in right_roads.values() if name in row], dtype=np.float64)
        n_roads = min(len(left), len(right))
        if n_roads and np.array_equal(np.sort(left), np.sort(right)):
            value = 0.0
        else:
            draws = [float(wasserstein_distance(rng.choice(left, n_roads, replace=True), rng.choice(right, n_roads, replace=True))) for _ in range(repetitions)] if n_roads else []
            value = float(np.mean(draws)) if draws else None
        result[f"road_equalized_w1_{name}"] = {"value": value, "status": "available" if value is not None else "not_evaluated", "candidate_road_coverage": len(left) / max(len(left_roads), 1), "holdout_road_coverage": len(right) / max(len(right_roads), 1), "equal_roads_per_resample": n_roads, "bootstrap_repetitions": repetitions, "unit_of_sampling": "road"}
    return result


def _fixed_mmd(candidate: list[dict[str, Any]], holdout: list[dict[str, Any]], fitted: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    names = fitted["mmd_feature_names"]
    gamma = fitted["mmd_gamma"]
    if not names or gamma is None:
        return {"value": None, "status": "not_calibrated", "coverage": {"candidate": 0.0, "holdout": 0.0}}
    center = np.asarray(fitted["normalization"]["center"], dtype=np.float64)
    scale = np.asarray(fitted["normalization"]["scale"], dtype=np.float64)

    def rows(items: list[dict[str, Any]]) -> np.ndarray:
        road_means = _road_means(items, names)
        values = [[features.get(name, np.nan) for name in names] for features in road_means.values()]
        arr = np.asarray(values, dtype=np.float64).reshape((-1, len(names)))
        arr = arr[np.isfinite(arr).all(axis=1)]
        return (arr - center) / scale if len(arr) else arr

    left, right = rows(candidate), rows(holdout)
    n_roads = min(len(left), len(right), int(options["mmd_max_roads"]))
    repetitions = int(options["bootstrap_repetitions"])
    if n_roads and left.shape == right.shape and _same_rows(left, right):
        value = 0.0
    elif n_roads:
        rng = np.random.default_rng(int(options["seed"]))
        draws = [_rbf_mmd(left[rng.integers(0, len(left), size=n_roads)], right[rng.integers(0, len(right), size=n_roads)], gamma) for _ in range(repetitions)]
        value = float(np.mean(draws))
    else:
        value = None
    return {"value": value, "status": "available" if value is not None else "not_evaluated", "unit_of_sampling": "road", "equal_roads_per_resample": n_roads, "bootstrap_repetitions": repetitions, "coverage": {"candidate_roads": len(left) / max(len({row['metadata'].get('road_segment_id') for row in candidate}), 1), "holdout_roads": len(right) / max(len({row['metadata'].get('road_segment_id') for row in holdout}), 1)}}


def _rbf_mmd(left: np.ndarray, right: np.ndarray, gamma: float) -> float:
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(delta * delta, axis=-1))
    value = float(kernel(left, left).mean() + kernel(right, right).mean() - 2.0 * kernel(left, right).mean())
    return max(value, 0.0)


def _same_rows(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    left_order = np.lexsort(left.T[::-1])
    right_order = np.lexsort(right.T[::-1])
    return bool(np.array_equal(left[left_order], right[right_order]))


def _real_reference(validation: list[dict[str, Any]], holdout: list[dict[str, Any]], fitted: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    min_roads = int(options["minimum_roads_per_reference_group"])
    roads_left = {str(row["metadata"].get("road_segment_id")) for row in validation if row["metadata"].get("road_segment_id")}
    roads_right = {str(row["metadata"].get("road_segment_id")) for row in holdout if row["metadata"].get("road_segment_id")}
    if len(roads_left) < min_roads or len(roads_right) < min_roads:
        return {"status": "not_evaluated", "reason": f"requires at least {min_roads} independent roads in validation and holdout", "validation_road_count": len(roads_left), "holdout_road_count": len(roads_right), "engineering_reference_95": {}}
    if any(not row["metadata"].get("road_segment_id") for row in validation + holdout):
        return {"status": "not_evaluated", "reason": "every real reference sample must identify its road", "engineering_reference_95": {}}
    calibration_roads = {str(road) for road in fitted.get("calibration_roads", []) if road}
    if len(calibration_roads) < min_roads or fitted.get("calibration_road_missing"):
        return {"status": "not_evaluated", "reason": f"calibration requires at least {min_roads} distinct roads with complete road IDs", "calibration_road_count": len(calibration_roads), "engineering_reference_95": {}}
    validation_road_ids = {str(row["metadata"]["road_segment_id"]) for row in validation}
    holdout_road_ids = {str(row["metadata"]["road_segment_id"]) for row in holdout}
    overlaps = {
        "calibration_validation": sorted(calibration_roads & validation_road_ids),
        "calibration_holdout": sorted(calibration_roads & holdout_road_ids),
        "validation_holdout": sorted(validation_road_ids & holdout_road_ids),
    }
    if any(overlaps.values()):
        return {"status": "not_evaluated", "reason": "road IDs overlap across calibration/validation/holdout", "road_overlaps": overlaps, "engineering_reference_95": {}}
    if fitted.get("calibration_quality_missing"):
        return {"status": "not_evaluated", "reason": "real calibration samples lack explicit quality evidence", "quality_evidence_missing": fitted["calibration_quality_missing"], "engineering_reference_95": {}}
    quality_failures = [row["manifest"].get("sample_id", row["manifest"].get("path")) for row in validation + holdout if not _quality_eligible(row["manifest"])]
    if quality_failures:
        return {"status": "not_evaluated", "reason": "real reference samples lack explicit quality evidence", "quality_evidence_missing": quality_failures, "engineering_reference_95": {}}
    names = fitted["descriptor_names"]
    reps = int(options["bootstrap_repetitions"])
    seed = int(options["seed"])
    rng = np.random.default_rng(seed)
    left_by_road = _road_means(validation, names)
    right_by_road = _road_means(holdout, names)
    n_roads = min(len(left_by_road), len(right_by_road))
    limits: dict[str, float] = {}
    roads_per_metric: dict[str, int] = {}
    for name in names:
        draws = []
        left_values = np.array([item[name] for item in left_by_road.values() if name in item])
        right_values = np.array([item[name] for item in right_by_road.values() if name in item])
        metric_roads = min(len(left_values), len(right_values))
        if metric_roads < min_roads:
            continue
        roads_per_metric[f"w1_{name}"] = metric_roads
        for _ in range(reps):
            left_draw = rng.choice(left_values, size=metric_roads, replace=True)
            right_draw = rng.choice(right_values, size=metric_roads, replace=True)
            draws.append(float(wasserstein_distance(left_draw, right_draw)))
        limits[f"w1_{name}"] = float(np.quantile(draws, 0.95))
    left_vectors = _road_vectors(left_by_road, fitted)
    right_vectors = _road_vectors(right_by_road, fitted)
    mmd_roads = min(len(left_vectors), len(right_vectors), int(options["mmd_max_roads"]))
    if mmd_roads >= min_roads and fitted["mmd_gamma"] is not None:
        mmd_draws = []
        for _ in range(reps):
            left_draw = left_vectors[rng.integers(0, len(left_vectors), size=mmd_roads)]
            right_draw = right_vectors[rng.integers(0, len(right_vectors), size=mmd_roads)]
            mmd_draws.append(_rbf_mmd(left_draw, right_draw, float(fitted["mmd_gamma"])))
        limits["fixed_joint_mmd"] = float(np.quantile(mmd_draws, 0.95))
        roads_per_metric["fixed_joint_mmd"] = mmd_roads
    return {"status": "eligible" if limits else "not_evaluated", "reason": None if limits else "no descriptors have sufficient per-road coverage", "validation_road_count": len(roads_left), "holdout_road_count": len(roads_right), "equal_roads_per_bootstrap": n_roads, "roads_per_metric": roads_per_metric, "bootstrap_repetitions": reps, "engineering_reference_95": limits}


def _road_means(rows: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        road = row["metadata"].get("road_segment_id")
        if not road:
            continue
        for name in names:
            if name in row["summaries"]:
                grouped[str(road)][name].append(float(row["summaries"][name]))
    return {road: {name: float(np.mean(values)) for name, values in feature.items()} for road, feature in grouped.items()}


def _road_vectors(road_means: dict[str, dict[str, float]], fitted: dict[str, Any]) -> np.ndarray:
    names = fitted["mmd_feature_names"]
    if not names:
        return np.empty((0, 0), dtype=np.float64)
    center = np.asarray(fitted["normalization"]["center"], dtype=np.float64)
    scale = np.asarray(fitted["normalization"]["scale"], dtype=np.float64)
    rows = [[features[name] for name in names] for features in road_means.values() if all(name in features for name in names)]
    return (np.asarray(rows, dtype=np.float64) - center) / scale if rows else np.empty((0, len(names)))


def _acceptance_gate(metrics: dict[str, Any], reference: dict[str, Any], candidate: list[dict[str, Any]], holdout: list[dict[str, Any]], options: dict[str, Any], fitted: dict[str, Any]) -> dict[str, Any]:
    min_roads = int(options["minimum_roads_per_reference_group"])
    candidate_roads = {row["metadata"].get("road_segment_id") for row in candidate if row["metadata"].get("road_segment_id")}
    holdout_roads = {row["metadata"].get("road_segment_id") for row in holdout if row["metadata"].get("road_segment_id")}
    limits = reference.get("engineering_reference_95", {})
    if reference.get("status") != "eligible" or len(candidate_roads) < min_roads or len(holdout_roads) < min_roads:
        return {"status": "not_evaluated", "reason": "eligible real-real range and at least five independent candidate/holdout roads are required"}
    if any(not row["metadata"].get("road_segment_id") for row in candidate + holdout):
        return {"status": "not_evaluated", "reason": "every candidate and holdout sample must identify its road"}
    if any(not _quality_eligible(row["manifest"]) for row in holdout):
        return {"status": "not_evaluated", "reason": "real holdout lacks explicit quality evidence"}
    if any(row["manifest"].get("domain") != "synthetic" for row in candidate):
        return {"status": "not_evaluated", "reason": "acceptance reference is only defined for programmatic synthetic candidates"}
    names = [name for name in limits if name.startswith("w1_")]
    candidate_roads = _road_means(candidate, [name[3:] for name in names])
    holdout_roads_map = _road_means(holdout, [name[3:] for name in names])
    compared = {}
    for metric in names:
        feature = metric[3:]
        left = np.array([row[feature] for row in candidate_roads.values() if feature in row])
        right = np.array([row[feature] for row in holdout_roads_map.values() if feature in row])
        if left.size and right.size:
            compared[metric] = float(wasserstein_distance(left, right))
    mmd_distance = None
    candidate_vectors = _road_vectors(_road_means(candidate, fitted["descriptor_names"]), fitted)
    holdout_vectors = _road_vectors(_road_means(holdout, fitted["descriptor_names"]), fitted)
    if len(candidate_vectors) and len(holdout_vectors) and fitted.get("mmd_gamma") is not None:
        mmd_distance = _rbf_mmd(candidate_vectors, holdout_vectors, float(fitted["mmd_gamma"]))
    if mmd_distance is not None and "fixed_joint_mmd" in limits:
        compared["fixed_joint_mmd"] = mmd_distance
    if not compared or set(compared) != set(limits):
        return {"status": "not_evaluated", "reason": "candidate/reference descriptors have insufficient road coverage", "available_metric_names": sorted(compared), "required_metric_names": sorted(limits)}
    compatible = all(key in limits and value <= limits[key] for key, value in compared.items())
    return {"status": "within_reference_range" if compatible else "outside_reference_range", "basis": "descriptive 95% road-resampled real-real engineering range; not a disease authenticity claim", "candidate_vs_real_road_equalized": compared, "reference_95": {key: limits[key] for key in compared}}


def _quality_eligible(manifest: dict[str, Any]) -> bool:
    spacing = manifest.get("sampling_spacing_m")
    precision = manifest.get("measurement_precision_m")
    return bool(
        manifest.get("units_verified") is True
        and manifest.get("quality_verified") is True
        and manifest.get("source_id")
        and isinstance(spacing, (int, float))
        and spacing > 0.0
        and isinstance(precision, (int, float))
        and precision > 0.0
    )
