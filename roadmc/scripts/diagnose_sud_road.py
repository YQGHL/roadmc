"""Run the frozen raw-field and native-unit geometry audit for SUD-ROAD."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadmc.data.real.sud_road import (  # noqa: E402
    decoded_fields,
    header_dict,
    iter_blocks,
    parse_header,
    point_view,
    sha256_file,
)

SOURCE_EXPECTED_SHA256 = "360b7d82c8b5c01a3a987c388d268635919a8ffa779e79d453705f7a7bba92de"
BLOCK_SIZE = 1_000_000
TILE_WIDTH = 2.0  # Source coordinate units; no physical unit asserted.
PATCH_POINT_CAP = 300_000
PATCH_LIMIT = 6
PATCH_GRID_STEP = 0.02  # Source coordinate units; reporting remains native-unit only.


def _validate_raw_records(rec: np.ndarray, header) -> dict[str, int]:
    fields = decoded_fields(rec, header)
    xyz = fields["xyz_source_units"]
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite decoded coordinates")
    tol = max(header.scales) * 1.01
    for j, (lo, hi) in enumerate(header.bounds_xyz):
        if xyz[:, j].min() < lo - tol or xyz[:, j].max() > hi + tol:
            raise ValueError(f"decoded coordinate axis {j} lies outside header bounds")
    rf = rec["return_flags"]
    ret, nret = rf & 7, (rf >> 3) & 7
    if np.any(ret == 0) or np.any(nret == 0) or np.any(ret > nret):
        raise ValueError("invalid LAS 1.2 return-number/number-of-returns bits")
    return {"records": len(rec), "finite_xyz": int(len(rec))}


def _tile_key(xyz: np.ndarray, lo: tuple[float, float], width: float) -> np.ndarray:
    ix = np.floor((xyz[:, 0] - lo[0]) / width).astype(np.int64)
    iy = np.floor((xyz[:, 1] - lo[1]) / width).astype(np.int64)
    return np.column_stack((ix, iy))


def _plane_residual_metrics(xyz: np.ndarray) -> dict:
    """Fit a centered least-squares plane and summarize native-unit residuals."""
    xy = xyz[:, :2]
    z = xyz[:, 2]
    center = np.median(xy, axis=0)
    design = np.column_stack((xy[:, 0] - center[0], xy[:, 1] - center[1], np.ones(len(xy))))
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    resid = z - design @ coef
    q = np.quantile(resid, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "plane_slope_source_z_per_source_xy": [float(coef[0]), float(coef[1])],
        "residual_quantiles_source_units": {
            str(p): float(v)
            for p, v in zip((0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), q, strict=True)
        },
        "residual_mad_source_units": float(np.median(np.abs(resid - np.median(resid)))),
        "residual_range_p01_p99_source_units": float(q[-1] - q[0]),
    }


def _raster_metrics(xyz: np.ndarray) -> dict:
    lo = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - lo) / PATCH_GRID_STEP).astype(np.int64)
    span = ij.max(axis=0) + 1
    flat = ij[:, 1] * span[0] + ij[:, 0]
    unique, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    z_order = np.argsort(inv, kind="stable")
    sorted_inv = inv[z_order]
    sorted_z = xyz[z_order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inv)) + 1]
    medians = np.empty(len(unique), dtype=np.float64)
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(sorted_z)
        medians[k] = np.median(sorted_z[s:e])
    return {
        "grid_step_source_units": PATCH_GRID_STEP,
        "grid_shape_yx": [int(span[1]), int(span[0])],
        "occupied_cells": int(len(unique)),
        "occupied_fraction": float(len(unique) / max(int(span[0] * span[1]), 1)),
        "points_per_cell_quantiles": [
            float(v) for v in np.quantile(counts, [0, 0.5, 0.9, 0.99, 1])
        ],
        "cell_median_z_quantiles_source_units": [
            float(v) for v in np.quantile(medians, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
        ],
        "cell_median_z_range_p01_p99_source_units": float(
            np.quantile(medians, 0.99) - np.quantile(medians, 0.01)
        ),
        "note": "Unfiltered cell medians; cell spacing is an analysis choice, not measured instrument resolution.",
    }


def _raw_code_group_metrics(xyz: np.ndarray, intensity: np.ndarray) -> dict:
    """Describe a raw-byte subset without assigning the value a class name."""
    xy = xyz[:, :2]
    center = xy.mean(axis=0)
    centered = xy - center
    eigenvalues = np.linalg.eigvalsh(np.cov(centered.T)) if len(xy) > 2 else np.array([0.0, 0.0])
    eigenvalues = np.maximum(eigenvalues, 0.0)
    slenderness = float(math.sqrt(eigenvalues[0] / eigenvalues[1])) if eigenvalues[1] > 0 else 0.0
    ij = np.floor((xy - xy.min(axis=0)) / PATCH_GRID_STEP).astype(np.int64)
    occupied = {(int(i), int(j)) for i, j in ij}
    edge_cells = sum(
        any((i + dx, j + dy) not in occupied for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        for i, j in occupied
    )
    vals = np.asarray(intensity, dtype=np.float64)
    return {
        "points_saved": int(len(xyz)),
        "bounds_xyz_source_units_saved": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
        "xy_pca_eigenvalues_source_units_squared": [float(v) for v in eigenvalues],
        "xy_pca_slenderness_sqrt_lambda_min_over_max_dimensionless": slenderness,
        "xy_bbox_aspect_ratio_dimensionless": float(
            min(np.ptp(xy, axis=0)) / max(np.ptp(xy, axis=0))
        )
        if np.ptp(xy, axis=0).max() > 0
        else 0.0,
        "xy_occupied_cells_at_fixed_grid_step": int(len(occupied)),
        "xy_grid_edge_cell_fraction": float(edge_cells / len(occupied)) if occupied else 0.0,
        "xy_grid_step_source_units": PATCH_GRID_STEP,
        "intensity_raw_quantiles_0_05_25_50_75_95_100": [
            float(v) for v in np.quantile(vals, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        ],
        "intensity_raw_mean": float(vals.mean()),
        "intensity_raw_std_population": float(vals.std()),
        "plane_residual": _plane_residual_metrics(xyz) if len(xyz) >= 10 else None,
        "interpretation": "unlabeled raw-byte subgroup; shape descriptors are sampling/grid dependent, not a crack or patch decision",
    }


def _implementation_hashes() -> dict[str, str]:
    return {
        "roadmc/data/real/sud_road.py": sha256_file(
            ROOT / "roadmc" / "data" / "real" / "sud_road.py"
        ),
        "roadmc/scripts/diagnose_sud_road.py": sha256_file(Path(__file__).resolve()),
        "tests/test_sud_road.py": sha256_file(ROOT / "tests" / "test_sud_road.py"),
    }


def run(
    source: Path,
    out_dir: Path,
    protocol_path: Path,
    *,
    on_output_created: Callable[[], None] | None = None,
) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    protocol_path = protocol_path.resolve()
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    header = parse_header(source)
    digest_before = sha256_file(source)
    if digest_before != SOURCE_EXPECTED_SHA256:
        raise ValueError(f"source SHA256 mismatch: {digest_before}")
    if protocol.get("source_sha256") != digest_before:
        raise ValueError("frozen protocol source SHA256 does not match input")
    if protocol.get("implementation_file_sha256") != _implementation_hashes():
        raise ValueError("frozen protocol implementation hashes do not match current files")
    if Path(protocol.get("output_directory", "")).resolve() != out_dir.resolve():
        raise ValueError("frozen protocol output directory does not match CLI output")
    expected_parameters = {
        "block_size": BLOCK_SIZE,
        "tile_width_source_units": TILE_WIDTH,
        "global_mixed_entropy_limit": PATCH_LIMIT,
        "global_mixed_min_points": 1000,
        "raw_code_min_points": 100,
        "raw_code_top_tiles_per_byte": 3,
        "sample_cap_per_tile": PATCH_POINT_CAP,
        "raster_step_source_units": PATCH_GRID_STEP,
    }
    if protocol.get("parameters") != expected_parameters:
        raise ValueError("frozen protocol parameters do not match current diagnostic constants")
    if not protocol.get("attempt01_reused_results_sha256"):
        raise ValueError("frozen protocol must bind prior attempt01 result hashes")
    points = point_view(source, header)

    # Pass 1: exact full-file field distributions and global finite/bounds checks.
    raw_cls = np.zeros(256, dtype=np.int64)
    intensity_hist = np.zeros(65536, dtype=np.int64)
    source_hist = np.zeros(65536, dtype=np.int64)
    user_hist = np.zeros(256, dtype=np.int64)
    angle_hist = np.zeros(256, dtype=np.int64)
    return_hist = np.zeros((8, 8), dtype=np.int64)
    class_source_pairs: Counter[tuple[int, int]] = Counter()
    xyz_min = np.full(3, np.inf)
    xyz_max = np.full(3, -np.inf)
    finite_total = 0
    intensity_sum = 0
    intensity_sumsq = 0
    for _start, rec in iter_blocks(points, BLOCK_SIZE):
        _validate_raw_records(rec, header)
        f = decoded_fields(rec, header)
        xyz = f["xyz_source_units"]
        xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
        xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
        finite_total += len(rec)
        raw_cls += np.bincount(f["classification_raw"], minlength=256)
        inten = f["intensity_raw"]
        intensity_hist += np.bincount(inten, minlength=65536)
        intensity_sum += int(inten.astype(np.uint64).sum())
        intensity_sumsq += int((inten.astype(np.uint64) ** 2).sum())
        source_hist += np.bincount(f["point_source_id_raw"], minlength=65536)
        user_hist += np.bincount(f["user_data_raw"], minlength=256)
        angle_hist += np.bincount((f["scan_angle_rank_raw"].astype(np.int16) + 128), minlength=256)
        pair = f["return_number_raw"].astype(np.int64) * 8 + f["number_of_returns_raw"]
        return_hist += np.bincount(pair, minlength=64).reshape(8, 8)
        pairkeys = (f["classification_raw"].astype(np.uint32) << 16) | f[
            "point_source_id_raw"
        ].astype(np.uint32)
        vals, counts = np.unique(pairkeys, return_counts=True)
        class_source_pairs.update(
            {(int(v >> 16), int(v & 0xFFFF)): int(n) for v, n in zip(vals, counts, strict=True)}
        )

    if finite_total != header.legacy_point_count:
        raise ValueError("full point iteration did not cover advertised point count")
    if not np.array_equal(intensity_hist.sum(), header.legacy_point_count):
        raise ValueError("intensity histogram count mismatch")

    # Pass 2: fixed 2.0 x 2.0 native-coordinate tiles, then select six by
    # raw-code entropy among sufficiently populated tiles. Labels/source IDs
    # remain anonymous values; this is a geometry candidate selector only.
    xmin, ymin = xyz_min[:2]
    nx_tiles = int(np.floor((header.bounds_xyz[0][1] - xmin) / TILE_WIDTH)) + 1
    tile_counts: dict[tuple[int, int], Counter[int]] = {}
    tile_intensity: dict[tuple[int, int], list[int]] = {}
    tile_z: dict[tuple[int, int], list[float]] = {}
    for _, rec in iter_blocks(points, BLOCK_SIZE):
        f = decoded_fields(rec, header)
        xyz = f["xyz_source_units"]
        keys = _tile_key(xyz, (xmin, ymin), TILE_WIDTH)
        tile_id = keys[:, 1] * nx_tiles + keys[:, 0]
        combo = tile_id * 256 + f["classification_raw"].astype(np.int64)
        vals, cnts = np.unique(combo, return_counts=True)
        for v, n in zip(vals, cnts, strict=True):
            tile, code = divmod(int(v), 256)
            iy, ix = divmod(tile, nx_tiles)
            tile_counts.setdefault((ix, iy), Counter())[code] += int(n)
        # Vectorized grouped moments and exact extrema avoid per-tile full-array masks.
        unique_tiles, inv = np.unique(tile_id, return_inverse=True)
        n_per = np.bincount(inv, minlength=len(unique_tiles))
        inten = f["intensity_raw"].astype(np.float64)
        sum_per = np.bincount(inv, weights=inten, minlength=len(unique_tiles))
        sumsq_per = np.bincount(inv, weights=inten * inten, minlength=len(unique_tiles))
        zmin_per = np.full(len(unique_tiles), np.inf)
        zmax_per = np.full(len(unique_tiles), -np.inf)
        np.minimum.at(zmin_per, inv, xyz[:, 2])
        np.maximum.at(zmax_per, inv, xyz[:, 2])
        for k, tile in enumerate(unique_tiles):
            iy, ix = divmod(int(tile), nx_tiles)
            key = (ix, iy)
            arr = tile_intensity.setdefault(key, [0, 0.0, 0.0])
            arr[0] += int(n_per[k])
            arr[1] += float(sum_per[k])
            arr[2] += float(sumsq_per[k])
            zr = tile_z.setdefault(key, [float("inf"), float("-inf")])
            zr[0] = min(zr[0], float(zmin_per[k]))
            zr[1] = max(zr[1], float(zmax_per[k]))

    candidates = []
    for key, counts in tile_counts.items():
        n = sum(counts.values())
        if n < 1000 or len(counts) < 2:
            continue
        p = np.asarray(list(counts.values()), dtype=np.float64) / n
        entropy = float(-(p * np.log(p)).sum())
        ii = tile_intensity[key]
        mean = ii[1] / ii[0]
        sd = math.sqrt(max(ii[2] / ii[0] - mean * mean, 0))
        candidates.append((entropy, n, sd, key, counts))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[3][1], x[3][0]))
    selected_by_key = {item[3]: item for item in candidates[:PATCH_LIMIT]}
    code_tile_keys: dict[int, list[tuple[int, int]]] = {}
    selection_reasons: dict[tuple[int, int], set[str]] = {
        item[3]: {"global_mixed_raw_byte_entropy_top6"} for item in candidates[:PATCH_LIMIT]
    }
    for code in np.flatnonzero(raw_cls):
        eligible = [
            (counts[int(code)], key, counts)
            for key, counts in tile_counts.items()
            if counts[int(code)] >= 100
        ]
        eligible.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
        code_keys = [item[1] for item in eligible[:3]]
        code_tile_keys[int(code)] = code_keys
        for _, key, counts in eligible[:3]:
            n = sum(counts.values())
            p = np.asarray(list(counts.values()), dtype=np.float64) / n
            entropy = float(-(p * np.log(p)).sum())
            ii = tile_intensity[key]
            mean = ii[1] / ii[0]
            sd = math.sqrt(max(ii[2] / ii[0] - mean * mean, 0))
            selected_by_key.setdefault(key, (entropy, n, sd, key, counts))
            selection_reasons.setdefault(key, set()).add(f"raw_byte_{int(code)}_top3_count")
    selected = list(selected_by_key.values())
    if not selected:
        raise ValueError("no spatial tiles met the frozen candidate selection rules")

    # Store deterministic source-order capped sample for each selected native tile.
    samples: dict[tuple[int, int], list[np.ndarray]] = {x[3]: [] for x in selected}
    seen: Counter[tuple[int, int]] = Counter()
    selected_linear = np.asarray([iy * nx_tiles + ix for ix, iy in samples], dtype=np.int64)
    linear_to_key = {iy * nx_tiles + ix: (ix, iy) for ix, iy in samples}
    for _, rec in iter_blocks(points, BLOCK_SIZE):
        f = decoded_fields(rec, header)
        xyz = f["xyz_source_units"]
        keys = _tile_key(xyz, (xmin, ymin), TILE_WIDTH)
        tile_id = keys[:, 1] * nx_tiles + keys[:, 0]
        mask = np.isin(tile_id, selected_linear)
        if not mask.any():
            continue
        indices = np.flatnonzero(mask)
        selected_ids = tile_id[indices]
        order = np.argsort(selected_ids, kind="stable")
        ordered_ids = selected_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
        for pos, start in enumerate(starts):
            end = starts[pos + 1] if pos + 1 < len(starts) else len(order)
            tile = linear_to_key[int(ordered_ids[start])]
            remaining = PATCH_POINT_CAP - seen[tile]
            if remaining <= 0:
                continue
            chosen = indices[order[start : min(start + remaining, end)]]
            samples[tile].append(rec[chosen].copy())
            seen[tile] += len(chosen)

    out_dir.mkdir(parents=True, exist_ok=False)
    if on_output_created is not None:
        on_output_created()
    patch_results = []
    for rank, (entropy, n, sd, key, counts) in enumerate(selected, 1):
        rec = np.concatenate(samples[key])
        f = decoded_fields(rec, header)
        xyz = f["xyz_source_units"]
        patch_id = f"candidate_{rank:02d}_tile_{key[0]}_{key[1]}"
        npz_path = out_dir / f"{patch_id}.npz"
        np.savez_compressed(npz_path, **f)
        local_intensity = f["intensity_raw"].astype(np.float64)
        patch_results.append(
            {
                "id": patch_id,
                "selection_reasons": sorted(selection_reasons[key]),
                "tile_index_xy": list(key),
                "tile_bounds_xy_source_units": [
                    xmin + key[0] * TILE_WIDTH,
                    ymin + key[1] * TILE_WIDTH,
                    xmin + (key[0] + 1) * TILE_WIDTH,
                    ymin + (key[1] + 1) * TILE_WIDTH,
                ],
                "point_count_in_tile_full": int(n),
                "point_count_saved": int(len(rec)),
                "capped": bool(n > len(rec)),
                "classification_raw_histogram_full_tile": {
                    str(k): int(v) for k, v in sorted(counts.items())
                },
                "raw_code_group_diagnostics_saved_sample": {
                    str(code): _raw_code_group_metrics(
                        xyz[f["classification_raw"] == code],
                        f["intensity_raw"][f["classification_raw"] == code],
                    )
                    for code in np.unique(f["classification_raw"])
                },
                "raw_code_entropy_nats": entropy,
                "intensity_sd_raw_selector": sd,
                "source_id_histogram_saved": {
                    str(int(k)): int(v)
                    for k, v in zip(
                        *np.unique(f["point_source_id_raw"], return_counts=True), strict=True
                    )
                },
                "bounds_xyz_source_units_saved": [
                    xyz.min(axis=0).tolist(),
                    xyz.max(axis=0).tolist(),
                ],
                "intensity_raw_summary_saved": {
                    "min": int(local_intensity.min()),
                    "max": int(local_intensity.max()),
                    "mean": float(local_intensity.mean()),
                    "std_population": float(local_intensity.std()),
                    "quantiles_0_05_25_50_75_95_100": [
                        float(v)
                        for v in np.quantile(local_intensity, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
                    ],
                },
                "plane_residual": _plane_residual_metrics(xyz),
                "raster": _raster_metrics(xyz),
                "npz": npz_path.name,
                "npz_sha256": sha256_file(npz_path),
            }
        )
        patch_sidecar = {
            "schema": "roadmc.sud-road-patch-sidecar.v1",
            "candidate": patch_results[-1],
            "source_path": str(source.resolve()),
            "source_sha256": digest_before,
            "frozen_protocol_path": str(protocol_path),
            "frozen_protocol_sha256": protocol_hash,
            "implementation_file_sha256": protocol["implementation_file_sha256"],
            "labels": "raw classification byte only; no SUD class crosswalk",
            "coordinates": "LAS source units; physical unit unverified",
        }
        (out_dir / f"{patch_id}.json").write_text(
            json.dumps(patch_sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    patch_by_key = {tuple(p["tile_index_xy"]): p for p in patch_results}
    raw_code_local_comparison = {}
    for code, keys in sorted(code_tile_keys.items()):
        tile_rows = []
        for key in keys:
            patch = patch_by_key[key]
            metrics = patch["raw_code_group_diagnostics_saved_sample"].get(str(code))
            tile_rows.append(
                {
                    "candidate_id": patch["id"],
                    "tile_index_xy": list(key),
                    "full_tile_raw_code_count": int(
                        patch["classification_raw_histogram_full_tile"].get(str(code), 0)
                    ),
                    "saved_group_present": metrics is not None,
                    "saved_group_metrics": metrics,
                }
            )
        slenderness = [
            row["saved_group_metrics"]["xy_pca_slenderness_sqrt_lambda_min_over_max_dimensionless"]
            for row in tile_rows
            if row["saved_group_metrics"] is not None
        ]
        intensity_means = [
            row["saved_group_metrics"]["intensity_raw_mean"]
            for row in tile_rows
            if row["saved_group_metrics"] is not None
        ]
        residual_mads = [
            row["saved_group_metrics"]["plane_residual"]["residual_mad_source_units"]
            for row in tile_rows
            if row["saved_group_metrics"] is not None
        ]
        raw_code_local_comparison[str(code)] = {
            "selection_protocol": "up to three distinct tiles ranked by full-tile count for this exact raw byte, with minimum count 100; ties by row then column",
            "selected_tile_count": len(tile_rows),
            "saved_group_count": sum(row["saved_group_present"] for row in tile_rows),
            "selected_tiles": tile_rows,
            "across_selected_tiles_descriptive_medians": {
                "xy_pca_slenderness_dimensionless": float(np.median(slenderness))
                if slenderness
                else None,
                "intensity_raw_mean": float(np.median(intensity_means))
                if intensity_means
                else None,
                "plane_residual_mad_source_units": float(np.median(residual_mads))
                if residual_mads
                else None,
                "interpretation": "median across at most three selected spatial tiles; descriptive only, not a class-level population estimate",
            },
        }

    digest_after = sha256_file(source)
    if digest_after != digest_before:
        raise ValueError("source changed during SUD audit")
    summary = {
        "schema": "roadmc.sud-road-raw-audit.v1",
        "protocol": {
            "purpose": "preserve original LAS attributes and diagnose candidate local geometry without semantic or metric claims",
            "source_units": "unknown; all coordinates and geometry deltas remain in LAS source-coordinate units",
            "semantic_mapping": "disabled; classification raw/low5/high3 retained separately",
            "point_source_id": "raw grouping only; no scan/road/semantic assumption",
            "full_scan_block_size": BLOCK_SIZE,
            "candidate_tile_width_source_units": TILE_WIDTH,
            "candidate_selection": "union of (a) top six mixed-byte entropy tiles with >=1000 points and >=2 original-byte values, and (b) up to three 2x2 source-unit tiles per observed raw classification byte with >=100 points of that byte, ranked by that byte's full-tile count; ties by tile row/column",
            "sample_cap_per_tile": PATCH_POINT_CAP,
            "raster_step_source_units": PATCH_GRID_STEP,
            "all_units_status": "unverified physical units; no millimeter crack width/depth or GT",
            "all_geometry_results_status": "observed native-coordinate geometry summaries, not calibrated distress measurements",
        },
        "interpretation_evidence": {
            "las12_classification_layout": {
                "source": "ASPRS LAS Specification Version 1.2, Classification Bit Field Encoding and standard class table",
                "url": "https://www.asprs.org/a/society/committees/standards/asprs_las_format_v12.pdf",
                "historical_url_returned_404": "https://www.asprs.org/wp-content/uploads/2010/12/asprs_las_format_v12.pdf",
                "format_level_reading": "bits 0-4 encode the LAS class value; bits 5/6/7 are synthetic/key-point/withheld flags",
                "data_values_observed": "raw byte 0 and 33-38; corresponding low-five values 0-6 and high-three flags 0 or 1",
                "dataset_semantic_mapping": "unresolved: the LAS-standard class names are not a verified crosswalk to the seven SUD-ROAD taxonomy names in the author README; no SUD label assignment is made",
            },
            "sud_road_author_readme": {
                "source": "author repository README",
                "url": "https://github.com/msqiu/SUD-Road",
                "supports": "the seven taxonomy names and manual CloudCompare labeling are described, but integer code assignments are not shown",
            },
            "physical_units": "unverified; direct VLR decoding finds GTModelTypeGeoKey (1024)=0 (unknown/undefined model CRS), and no ProjectedCRSGeoKey (3072) or ProjLinearUnitsGeoKey (3076); coordinate scale alone is not unit evidence",
            "geotiff_geokey_evidence": {
                "source": "OGC GeoTIFF 1.1 Standard",
                "url": "https://docs.ogc.org/is/19-008r4/19-008r4.html",
                "interpretation": "GeoKey 1024 identifies model type; OGC assigns value 0 to unknown/undefined and value 1 to projected CRS. The SUD VLR contains only key 1024 with value 0; it has no key 3072 or 3076.",
            },
        },
        "header": header_dict(header),
        "source_integrity": {
            "sha256_before": digest_before,
            "sha256_after": digest_after,
            "unchanged_during_audit": digest_before == digest_after,
        },
        "frozen_protocol": {
            "path": str(protocol_path),
            "sha256": protocol_hash,
            "implementation_file_sha256": protocol["implementation_file_sha256"],
            "attempt01_reused_results_sha256": protocol["attempt01_reused_results_sha256"],
        },
        "full_file": {
            "points_checked": int(finite_total),
            "all_coordinates_finite": True,
            "decoded_bounds_xyz_source_units": [xyz_min.tolist(), xyz_max.tolist()],
            "classification_raw_histogram": {
                str(i): int(raw_cls[i]) for i in np.flatnonzero(raw_cls)
            },
            "classification_low5_histogram": {
                str(i): int(v)
                for i, v in enumerate(
                    np.bincount(np.arange(256) % 32, weights=raw_cls, minlength=32)
                )
                if v
            },
            "classification_high3_flag_histogram": {
                str(i): int(v)
                for i, v in enumerate(
                    np.bincount(np.arange(256) >> 5, weights=raw_cls, minlength=8)
                )
                if v
            },
            "classification_raw_by_point_source_id_joint": [
                {"classification_raw": a, "point_source_id_raw": b, "count": c}
                for (a, b), c in sorted(class_source_pairs.items())
            ],
            "point_source_id_raw_histogram": {
                str(i): int(source_hist[i]) for i in np.flatnonzero(source_hist)
            },
            "intensity_raw": {
                "min": int(np.flatnonzero(intensity_hist)[0]),
                "max": int(np.flatnonzero(intensity_hist)[-1]),
                "mean": intensity_sum / finite_total,
                "std_population": math.sqrt(
                    max(intensity_sumsq / finite_total - (intensity_sum / finite_total) ** 2, 0)
                ),
                "histogram_nonzero": {
                    str(i): int(intensity_hist[i]) for i in np.flatnonzero(intensity_hist)
                },
            },
            "return_number_by_number_of_returns": return_hist.tolist(),
            "user_data_raw_histogram": {
                str(i): int(user_hist[i]) for i in np.flatnonzero(user_hist)
            },
            "scan_angle_rank_raw_histogram_shifted_index_i_minus_128": {
                str(i - 128): int(angle_hist[i]) for i in np.flatnonzero(angle_hist)
            },
        },
        "patch_candidates": patch_results,
        "raw_code_local_comparison": raw_code_local_comparison,
        "解释_zh": {
            "实测_文件与字段": f"完成 LAS 1.2、format 0、20 字节点记录的逐块读取；校验 {finite_total} 点坐标均有限且落在头部范围内，分类字节、分类高位、强度、点源编号和其它 format-0 原始字段均保留。",
            "实测_分类字节": "全文件分类原始字节为 "
            + "、".join(f"{code}:{int(raw_cls[code])}点" for code in np.flatnonzero(raw_cls))
            + "。低5位/高3位遵循LAS1.2字段布局，但不能据此将这些数值映射到README的七个SUD-ROAD名称；README未给整数码表。",
            "实测_点源编号": "point_source_id 原始值计数已逐点统计并与分类字节做联合统计；文件本身未说明这些编号代表扫描、路段或类别，因此保留为原字段。",
            "定性_单位": "本次直接解码LAS内GeoKey VLR：key 1024值为0（CRS未知/未定义）；没有key 3072（Projected CRS）和key 3076（投影线性单位），故不能确认坐标参考系或单位。以下坐标与几何量只用LAS源坐标单位表达，禁止解释为米或毫米。",
            "定性_几何": f"固定规则选出 {len(patch_results)} 个2×2源坐标单位空间块；每块按分类原始字节分组描述强度、去平面残差、PCA细长率及固定栅格边界支撑。几何特征只用于检查潜在裂缝/修补形态，不是已确认标签或宽度/深度GT。每个原始字节另选最多3个按点数排序的局部空间块，不能当作代表性抽样。",
            "代码映射状态": "未找到作者公开整数码表；ASPRS LAS1.2规范解释的是通用LAS分类，不构成SUD-ROAD七类的交叉映射。",
            "范围限制": "单一LAS文件不能建立独立道路留出；本次没有训练、调参、修改通用导入器或修改默认参数。",
        },
        "excluded_or_unresolved": [
            "No semantic names assigned to any classification byte or bit pattern.",
            "No physical length unit inferred from CRS absence, LAS scale, bounds, dataset span, or sensor metadata.",
            "No point_source_id grouping interpreted as road, pass, scan, or category.",
            "No crack width/depth, patch thickness, or millimeter ground truth reported.",
        ],
    }
    (out_dir / "sud_road_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    artifacts = []
    for artifact in sorted(out_dir.iterdir(), key=lambda item: item.name):
        if artifact.name == "outputs_manifest.json":
            continue
        artifacts.append(
            {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    outputs_manifest = {
        "schema": "roadmc.sud-road-outputs-manifest.v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "source_sha256": digest_before,
        "implementation_file_sha256": protocol["implementation_file_sha256"],
        "artifacts": artifacts,
    }
    (out_dir / "outputs_manifest.json").write_text(
        json.dumps(outputs_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    args = p.parse_args()
    started = time.perf_counter()
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    output_created_by_run = False

    def mark_output_created() -> None:
        nonlocal output_created_by_run
        output_created_by_run = True

    try:
        result = run(
            args.source,
            args.output,
            args.protocol,
            on_output_created=mark_output_created,
        )
    except Exception as exc:
        if output_created_by_run and args.output.is_dir():
            (args.output / "run_record.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "exit_code": 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    manifest_path = args.output / "outputs_manifest.json"
    run_record = {
        "command": command,
        "exit_code": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "output_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(args.protocol),
        "source_sha256": result["source_integrity"]["sha256_after"],
    }
    (args.output / "run_record.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "points": result["full_file"]["points_checked"],
                "patches": len(result["patch_candidates"]),
                "sha256": result["source_integrity"]["sha256_after"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
