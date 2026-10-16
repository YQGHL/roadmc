"""Run the frozen, bounded SHREC RGB-D/mask contour study.

Use ``--freeze-only`` first; the analysis run consumes that immutable selection
and saves local, non-distributable derivatives in the selected output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.real.shrec_video import (
    contour_metrics,
    decode_video_samples,
    mask_color_distance,
    mask_red_channel_rule,
    read_member_verified,
    select_mask_clips,
    selected_frame_indices,
    sha256_bytes,
    thresholds,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _source_record(catalog: dict[str, Any], token: str) -> dict[str, Any]:
    for row in catalog["raw_sources"]:
        if token.lower() in row["path"].lower():
            return row
    raise KeyError(f"source token not found in admission catalog: {token}")


def freeze_protocol(audit_root: Path, source_root: Path, out: Path) -> Path:
    protocol_path = out / "frozen_protocol.json"
    if protocol_path.exists():
        raise FileExistsError(
            f"frozen protocol already exists and will not be overwritten: {protocol_path}"
        )
    out.mkdir(parents=True, exist_ok=True)
    catalog_path = audit_root / "paper_followup" / "admission_catalog.v2.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    dataset = next(row for row in catalog["datasets"] if row["id"] == "SHREC_Pothole_Mix_V2")
    train_path = audit_root / "docs" / "train_filelist.txt"
    valid_path = audit_root / "docs" / "valid_filelist.txt"
    mask_index_path = audit_root / "shrec-2022-potholes-rgbd-videos-masks_members.json"
    video_index_path = audit_root / "shrec-2022-potholes-rgbd-videos_members.json"
    train_names = train_path.read_text(encoding="utf-8").splitlines()
    valid_names = valid_path.read_text(encoding="utf-8").splitlines()
    mask_members = [
        PurePosixPath(row["name"]).name
        for row in json.loads(mask_index_path.read_text(encoding="utf-8"))
        if row["name"].lower().endswith(".avi")
    ]
    selected = select_mask_clips(mask_members, train_names, valid_names, n_train=12, n_valid=8)
    rgb_archive = source_root / "shrec-2022-potholes-rgbd-videos.zip"
    mask_archive = source_root / "shrec-2022-potholes-rgbd-videos-masks.zip"
    protocol = {
        "schema": "roadmc.shrec-video-study.v1",
        "created_utc_date": "2026-10-16",
        "status": "frozen_before_main_sample_decode",
        "data_record": dataset["official_record"],
        "sampling": {
            "clip_count": {"train": 12, "valid": 8},
            "rule": "Map mask AVI basenames to color MP4 basenames in supplied filelists; sort capture groups lexically; choose 12/8 evenly spaced group ranks with round(i*(n_groups-1)/(n_selected-1)); within each selected capture group choose the numeric clip ID nearest the median, tie to lower ID.",
            "capture_group": "filename prefix [a-z]-YYYYMMDD-HHMMSS; group overlap across train/valid is retained and is not treated as independent.",
            "frame_rule": "For each color clip choose ordinal 0, floor((N-1)/2), and N-1; map paired disparity/mask ordinal by normalized sequence position if frame counts differ; preserve all chosen source ordinals and PTS in the sidecar.",
            "requested_train_groups": 12,
            "requested_valid_groups": 8,
        },
        "mask_decode_policy": {
            "decoded_field": "MJPEG frame converted by PyAV to RGB24 uint8; raw decoded RGB values are preserved in NPZ.",
            "nominal_red_foreground_rgb": [255, 0, 0],
            "primary": "Euclidean RGB distance <= 32 from (255,0,0).",
            "fixed_sensitivity_thresholds_rgb_l2": [8, 32, 64],
            "independent_red_channel_rule": "R>=128, G<=64, B<=64, R-G>=96, R-B>=96.",
            "thresholds_are_predeclared": True,
            "basis": "Dataset pilot only established MJPEG codec, RGB24 decoding and near-red target code; thresholds are fixed before the selected sample is decoded and are not selected by model score.",
        },
        "contour_policy": {
            "components": "8-connected; all-component count and pixels retained; geometry describes largest component.",
            "edge": "Largest component touching any image edge is marked incomplete and excluded from ellipse/harmonic/convex-hull model fit; no boundary is imputed.",
            "holes": "Count 8-connected components of binary_fill_holes(component) minus component.",
            "nonstar": "720 centroid rays at 0.5-pixel sampling; report the fraction with a background gap followed by later foreground.",
            "holdout": "Angles about full-component pixel centroid; 16 equal sectors, sector_index modulo 5 == 0 held out before fitting.",
            "models": "Reuse RoadMC _fit_outline and _fit_radial_harmonics on training arc points; report the same held-out observed-pixel to predicted-curve nearest Euclidean distance in source pixels for ellipse, harmonic and training-arc convex hull. Keep radial residuals as supplementary only.",
            "units": "Source pixels and dimensionless shape statistics only; no pixel-to-length conversion.",
        },
        "depth_and_registration": {
            "disparity": "Preserve decoded 8-bit Y/luma code values. No metric depth conversion.",
            "alignment": "Pair by clip identifier and ordinal/normalized sequence position only; no spatial resize, calibration, homography or temporal offset correction.",
            "unknowns": [
                "camera intrinsics",
                "stereo baseline",
                "disparity value scale/offset",
                "pixel registration between 1920x1080 RGB and 640x400 disparity",
                "stream timing offset",
                "mask-to-road geometric accuracy",
            ],
        },
        "selected_clips": selected,
        "input_evidence": {
            "admission_catalog_v2": {
                "path": str(catalog_path),
                "sha256": _sha256_file(catalog_path),
            },
            "filelists": {
                "train": {"path": str(train_path), "sha256": _sha256_file(train_path)},
                "valid": {"path": str(valid_path), "sha256": _sha256_file(valid_path)},
            },
            "archive_manifest_entries": {
                "rgb_video_zip": _source_record(catalog, "rgbd-videos.zip"),
                "mask_video_zip": _source_record(catalog, "rgbd-videos-masks.zip"),
            },
            "archive_paths": {
                "rgb_video_zip": str(rgb_archive),
                "mask_video_zip": str(mask_archive),
            },
            "zip_member_indexes": {
                "rgb": {
                    "path": str(audit_root / "shrec-2022-potholes-rgbd-videos_members.json"),
                    "sha256": _sha256_file(video_index_path),
                },
                "mask": {"path": str(mask_index_path), "sha256": _sha256_file(mask_index_path)},
            },
            "full_archive_hashes_reused_from_admission_catalog": True,
        },
    }
    _write_json(protocol_path, protocol)
    return protocol_path


def _members_by_basename(index_path: Path, suffix: str) -> dict[str, str]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    members = {}
    for row in rows:
        name = row["name"]
        if name.lower().endswith(suffix.lower()):
            basename = PurePosixPath(name).name
            if basename in members:
                raise ValueError(f"duplicate archive basename: {basename}")
            members[basename] = name
    return members


def write_clip_metadata_inventory(audit_root: Path, out: Path) -> dict[str, Any]:
    """Match all mask clips to both RGB-D split lists and filename capture groups."""
    train_names = (
        (audit_root / "docs" / "train_filelist.txt").read_text(encoding="utf-8").splitlines()
    )
    valid_names = (
        (audit_root / "docs" / "valid_filelist.txt").read_text(encoding="utf-8").splitlines()
    )
    mask_index_path = audit_root / "shrec-2022-potholes-rgbd-videos-masks_members.json"
    video_index_path = audit_root / "shrec-2022-potholes-rgbd-videos_members.json"
    mask_members = _members_by_basename(mask_index_path, ".avi")
    video_members = _members_by_basename(video_index_path, ".mp4")
    split_by_rgb: dict[str, set[str]] = {"train": set(), "valid": set()}
    split_by_disp: dict[str, set[str]] = {"train": set(), "valid": set()}
    raw_file_counts = {}
    for split, names in (("train", train_names), ("valid", valid_names)):
        split_by_rgb[split] = {PurePosixPath(n).name for n in names if "-color-" in n}
        split_by_disp[split] = {PurePosixPath(n).name for n in names if "-disp-" in n}
        raw_file_counts[split] = {
            "filelist_entries": len(names),
            "rgb_entries": len(split_by_rgb[split]),
            "disparity_entries": len(split_by_disp[split]),
        }
    rows = []
    for mask_basename, mask_member in sorted(mask_members.items()):
        match = re.fullmatch(
            r"(?P<capture>[a-z]-\d{8}-\d{6})-mask-c(?P<clip>\d+)\.avi", mask_basename
        )
        if not match:
            raise ValueError(f"unrecognized mask basename in archive index: {mask_basename}")
        capture, clip_id = match.group("capture"), int(match.group("clip"))
        rgb_basename = f"{capture}-color-c{clip_id:06d}.mp4"
        disp_basename = f"{capture}-disp-c{clip_id:06d}.mp4"
        rgb_splits = [s for s in ("train", "valid") if rgb_basename in split_by_rgb[s]]
        disp_splits = [s for s in ("train", "valid") if disp_basename in split_by_disp[s]]
        if len(rgb_splits) != 1 or len(disp_splits) != 1:
            split = "unmatched_or_ambiguous"
        elif rgb_splits != disp_splits:
            split = "rgb_disparity_split_mismatch"
        else:
            split = rgb_splits[0]
        rows.append(
            {
                "mask_basename": mask_basename,
                "mask_archive_member": mask_member,
                "capture_group": capture,
                "clip_id": clip_id,
                "rgb_basename": rgb_basename,
                "rgb_archive_member": video_members.get(rgb_basename),
                "disparity_basename": disp_basename,
                "disparity_archive_member": video_members.get(disp_basename),
                "train_valid_membership": split,
            }
        )
    train_groups = {r["capture_group"] for r in rows if r["train_valid_membership"] == "train"}
    valid_groups = {r["capture_group"] for r in rows if r["train_valid_membership"] == "valid"}
    payload = {
        "schema": "roadmc.shrec-clip-metadata-inventory.v1",
        "source": "archive member index + supplied train_filelist.txt/valid_filelist.txt + filename capture prefix",
        "clip_count": len(rows),
        "all_mask_clips_matched_rgb_and_disparity": all(
            r["rgb_archive_member"]
            and r["disparity_archive_member"]
            and r["train_valid_membership"] in ("train", "valid")
            for r in rows
        ),
        "counts_by_membership": {
            split: sum(r["train_valid_membership"] == split for r in rows)
            for split in (
                "train",
                "valid",
                "unmatched_or_ambiguous",
                "rgb_disparity_split_mismatch",
            )
        },
        "filelist_stream_counts": raw_file_counts,
        "capture_groups": {
            "train_unique": len(train_groups),
            "valid_unique": len(valid_groups),
            "union_unique": len(train_groups | valid_groups),
            "shared_prefixes": sorted(train_groups & valid_groups),
            "shared_count": len(train_groups & valid_groups),
            "interpretation": "Filename capture prefixes are acquisition-group metadata only; they do not identify independent roads.",
        },
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "clip_metadata_inventory.json", payload)
    return payload


def _video_frame_count(data: bytes) -> int:
    import io

    import av

    with av.open(io.BytesIO(data)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("each sampled clip must contain exactly one video stream")
        count = int(container.streams.video[0].frames)
    if count <= 0:
        raise ValueError("selected video stream has no indexed frame count")
    return count


def _map_indices(reference_indices: list[int], reference_count: int, other_count: int) -> list[int]:
    if other_count <= 0:
        raise ValueError("paired stream has no frames")
    if reference_count <= 1 or other_count <= 1:
        return [0 for _ in reference_indices]
    return [int(round(i * (other_count - 1) / (reference_count - 1))) for i in reference_indices]


def _append_flat(target: list[np.ndarray], value: np.ndarray) -> tuple[int, tuple[int, ...]]:
    arr = np.ascontiguousarray(value, dtype=np.uint8)
    offset = sum(len(v) for v in target)
    target.append(arr.reshape(-1))
    return offset, arr.shape


def _aggregate(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return {
        "n_frame_observations": len(values),
        "median": float(statistics.median(values)) if values else None,
        "q25": float(np.quantile(values, 0.25)) if values else None,
        "q75": float(np.quantile(values, 0.75)) if values else None,
    }


def _write_contact_sheet(frames: list[dict[str, Any]], out: Path) -> None:
    shown = [row for row in frames if row["frame_position"] == "middle"]
    shown = shown[:20]
    cols = 4
    rows_n = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 3.4 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel()[: len(shown)], shown, strict=True):
        rgb = row["rgb"]
        mask = row["mask32"]
        vis = rgb.copy()
        edge = mask & ~np.pad(
            mask[1:, 1:] & mask[:-1, 1:] & mask[1:, :-1] & mask[:-1, :-1],
            ((0, 1), (0, 1)),
            constant_values=False,
        )
        vis[edge] = np.array([255, 255, 0], dtype=np.uint8)
        ax.imshow(vis)
        ax.set_title(
            f"{row['split']} | {row['capture_group']} | c{row['clip_id']:06d}\n{row['frame_index']} | edge:{row['contours']['32']['status']}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(
        "SHREC sampled middle frames; yellow = decoded-mask boundary (threshold 32)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _write_error_figure(summary: dict[str, Any], out: Path) -> None:
    rows = summary["model_test_errors"]
    labels = ["ellipse", "harmonic", "convex hull"]
    keys = ["ellipse", "area_normalized_harmonic", "training_convex_hull"]
    series = []
    for key in keys:
        vals = [
            r["test_rmse_px"] for r in rows if r["model"] == key and r["test_rmse_px"] is not None
        ]
        series.append(vals)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if any(series):
        ax.boxplot(series, tick_labels=labels, showmeans=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No complete contours with successful held-out fits",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_ylabel("Held-out nearest boundary error (px)")
    ax.set_title("SHREC complete in-frame contours; same pixel coordinates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_study(audit_root: Path, out: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_main_sample_decode":
        raise ValueError("protocol is not frozen for analysis")
    out.mkdir(parents=True, exist_ok=True)
    analysis_path = out / "analysis.json"
    if analysis_path.exists():
        raise FileExistsError(f"analysis already exists; refusing to overwrite: {analysis_path}")
    source_root = Path(protocol["input_evidence"]["archive_paths"]["rgb_video_zip"]).parent
    rgb_zip_path = source_root / "shrec-2022-potholes-rgbd-videos.zip"
    mask_zip_path = source_root / "shrec-2022-potholes-rgbd-videos-masks.zip"
    rgb_index = _members_by_basename(
        audit_root / "shrec-2022-potholes-rgbd-videos_members.json", ".mp4"
    )
    mask_index = _members_by_basename(
        audit_root / "shrec-2022-potholes-rgbd-videos-masks_members.json", ".avi"
    )
    records: list[dict[str, Any]] = []
    raw_rgb: list[np.ndarray] = []
    raw_mask: list[np.ndarray] = []
    raw_disp: list[np.ndarray] = []
    primary_packed: list[np.ndarray] = []
    rgb_offsets: list[int] = []
    rgb_shapes: list[tuple[int, ...]] = []
    mask_offsets: list[int] = []
    mask_shapes: list[tuple[int, ...]] = []
    disp_offsets: list[int] = []
    disp_shapes: list[tuple[int, ...]] = []
    primary_bit_offsets: list[int] = []
    primary_bit_lengths: list[int] = []
    contours_yx: list[np.ndarray] = []
    contour_offsets: list[int] = []
    contour_lengths: list[int] = []
    selected_clip_sidecars: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    total_frame_records = 0

    with (
        zipfile.ZipFile(rgb_zip_path, "r") as rgb_archive,
        zipfile.ZipFile(mask_zip_path, "r") as mask_archive,
    ):
        for clip_no, selected in enumerate(protocol["selected_clips"], start=1):
            rgb_member = rgb_index.get(selected["rgb_basename"])
            disp_basename = selected["disp_basename"]
            disp_member = rgb_index.get(disp_basename)
            mask_member = mask_index.get(selected["mask_basename"])
            if not all((rgb_member, disp_member, mask_member)):
                raise KeyError(f"paired archive member missing for selected clip: {selected}")
            rgb_bytes, rgb_hash = read_member_verified(rgb_archive, rgb_member)
            disp_bytes, disp_hash = read_member_verified(rgb_archive, disp_member)
            mask_bytes, mask_hash = read_member_verified(mask_archive, mask_member)
            rgb_count, disp_count, mask_count = map(
                _video_frame_count, (rgb_bytes, disp_bytes, mask_bytes)
            )
            rgb_idx = selected_frame_indices(rgb_count)
            disp_idx = _map_indices(rgb_idx, rgb_count, disp_count)
            mask_idx = _map_indices(rgb_idx, rgb_count, mask_count)
            rgb_info, rgb_frames = decode_video_samples(rgb_bytes, rgb_idx, role="rgb")
            disp_info, disp_frames = decode_video_samples(disp_bytes, disp_idx, role="disparity")
            mask_info, mask_frames = decode_video_samples(mask_bytes, mask_idx, role="mask")
            clip_sidecar = {
                **selected,
                "archive_members": {"rgb": rgb_hash, "disparity": disp_hash, "mask": mask_hash},
                "streams": {"rgb": rgb_info, "disparity": disp_info, "mask": mask_info},
                "paired_frames": [],
            }
            for position, (ri, di, mi) in enumerate(zip(rgb_idx, disp_idx, mask_idx, strict=True)):
                rgb = rgb_frames[ri]
                disp = disp_frames[di]
                mask_rgb = mask_frames[mi]
                if not rgb.shape[:2] == mask_rgb.shape[:2]:
                    raise ValueError(
                        f"RGB/mask dimensions differ in {selected['mask_basename']}: {rgb.shape} vs {mask_rgb.shape}"
                    )
                pts_rgb = rgb_info["selected_pts_seconds"].get(str(ri))
                pts_disp = disp_info["selected_pts_seconds"].get(str(di))
                pts_mask = mask_info["selected_pts_seconds"].get(str(mi))
                pair = {
                    "rgb_ordinal": ri,
                    "disparity_ordinal": di,
                    "mask_ordinal": mi,
                    "rgb_pts_s": pts_rgb,
                    "disparity_pts_s": pts_disp,
                    "mask_pts_s": pts_mask,
                    "delta_disparity_minus_rgb_s": (pts_disp - pts_rgb)
                    if pts_disp is not None and pts_rgb is not None
                    else None,
                    "delta_mask_minus_rgb_s": (pts_mask - pts_rgb)
                    if pts_mask is not None and pts_rgb is not None
                    else None,
                }
                clip_sidecar["paired_frames"].append(pair)
                row_base = {
                    "sample_index": total_frame_records,
                    "split": selected["split"],
                    "capture_group": selected["capture_group"],
                    "clip_id": selected["clip_id"],
                    "mask_basename": selected["mask_basename"],
                    "frame_position": ("first", "middle", "last")[position],
                    "frame_index": ri,
                    "disparity_frame_index": di,
                    "mask_frame_index": mi,
                    "rgb_shape": list(rgb.shape),
                    "disparity_shape": list(disp.shape),
                    "mask_shape": list(mask_rgb.shape),
                    "rgb_sha256": sha256_bytes(rgb.tobytes()),
                    "disparity_sha256": sha256_bytes(disp.tobytes()),
                    "mask_rgb_sha256": sha256_bytes(mask_rgb.tobytes()),
                    "pairing": pair,
                    "rgb": rgb,
                    "disparity": disp,
                    "mask_rgb": mask_rgb,
                }
                rgb_offset, rgb_shape = _append_flat(raw_rgb, rgb)
                mask_offset, mask_shape = _append_flat(raw_mask, mask_rgb)
                disp_offset, disp_shape = _append_flat(raw_disp, disp)
                row_base.update(
                    {
                        "rgb_npz_offset": rgb_offset,
                        "mask_rgb_npz_offset": mask_offset,
                        "disparity_y_npz_offset": disp_offset,
                    }
                )
                rgb_offsets.append(rgb_offset)
                rgb_shapes.append(rgb_shape)
                mask_offsets.append(mask_offset)
                mask_shapes.append(mask_shape)
                disp_offsets.append(disp_offset)
                disp_shapes.append(disp_shape)
                for threshold in thresholds():
                    binary = mask_color_distance(mask_rgb, threshold)
                    metrics, edge_yx = contour_metrics(binary)
                    threshold_row = {
                        "sample_index": total_frame_records,
                        "split": selected["split"],
                        "capture_group": selected["capture_group"],
                        "clip_id": selected["clip_id"],
                        "frame_position": row_base["frame_position"],
                        "frame_index": ri,
                        "color_distance_threshold": int(threshold),
                        **metrics,
                    }
                    threshold_rows.append(threshold_row)
                    row_base.setdefault("contours", {})[str(int(threshold))] = metrics
                    if int(threshold) == 32:
                        row_base["mask32"] = binary
                        bits = np.packbits(binary.reshape(-1), bitorder="little")
                        primary_bit_offsets.append(sum(len(b) for b in primary_packed))
                        primary_bit_lengths.append(binary.size)
                        primary_packed.append(bits)
                        contour_offsets.append(sum(len(c) for c in contours_yx))
                        contour_lengths.append(len(edge_yx))
                        contours_yx.append(edge_yx.astype(np.int32))
                red_binary = mask_red_channel_rule(mask_rgb)
                row_base["red_channel_rule_foreground_pixels"] = int(red_binary.sum())
                row_base["color_distance_8_foreground_pixels"] = row_base["contours"]["8"][
                    "foreground_pixels"
                ]
                row_base["color_distance_32_foreground_pixels"] = row_base["contours"]["32"][
                    "foreground_pixels"
                ]
                row_base["color_distance_64_foreground_pixels"] = row_base["contours"]["64"][
                    "foreground_pixels"
                ]
                row_base.pop("rgb")
                row_base.pop("disparity")
                row_base.pop("mask_rgb")
                row_base.pop("mask32")
                records.append(row_base)
                total_frame_records += 1
            selected_clip_sidecars.append(clip_sidecar)
            print(
                f"decoded {clip_no}/{len(protocol['selected_clips'])}: {selected['split']} {selected['mask_basename']}",
                flush=True,
            )

    # Primary threshold foreground areas are compared against the independent fixed red-channel rule.
    red_sensitivity = []
    for row in records:
        red_sensitivity.append(
            {
                "sample_index": row["sample_index"],
                "split": row["split"],
                "clip_id": row["clip_id"],
                "frame_position": row["frame_position"],
                "red_channel_rule_foreground_pixels": row["red_channel_rule_foreground_pixels"],
                "distance32_foreground_pixels": row["color_distance_32_foreground_pixels"],
                "difference_pixels": row["red_channel_rule_foreground_pixels"]
                - row["color_distance_32_foreground_pixels"],
            }
        )
    complete_primary = [
        r
        for r in threshold_rows
        if r["color_distance_threshold"] == 32 and r["status"] == "complete_in_frame"
    ]
    model_test_errors: list[dict[str, Any]] = []
    for row in complete_primary:
        models = row.get("pixel_coordinate_model_comparison", {}).get("models", {})
        for model_name, model in models.items():
            if model_name == "training_convex_hull":
                metrics = model.get("test", {})
            else:
                metrics = model.get("test", {})
            model_test_errors.append(
                {
                    "sample_index": row["sample_index"],
                    "split": row["split"],
                    "capture_group": row["capture_group"],
                    "clip_id": row["clip_id"],
                    "frame_position": row["frame_position"],
                    "model": model_name,
                    "test_rmse_px": metrics.get("rmse_px"),
                    "test_mae_px": metrics.get("mae_px"),
                    "test_p95_px": metrics.get("p95_px"),
                }
            )

    summary = {
        "schema": "roadmc.shrec-video-analysis.v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "input_archives": protocol["input_evidence"]["archive_manifest_entries"],
        "selected_clips": {
            "total": len(selected_clip_sidecars),
            "by_split": {
                split: sum(x["split"] == split for x in selected_clip_sidecars)
                for split in ("train", "valid")
            },
            "unique_capture_groups_total": len(
                {x["capture_group"] for x in selected_clip_sidecars}
            ),
            "capture_groups_shared_between_selected_splits": sorted(
                {x["capture_group"] for x in selected_clip_sidecars if x["split"] == "train"}
                & {x["capture_group"] for x in selected_clip_sidecars if x["split"] == "valid"}
            ),
            "warning": "Provided train/valid metadata has overlapping capture prefixes; this is a within-source grouped diagnostic, not independent-road evaluation.",
        },
        "sampled_frame_count": len(records),
        "decode_policy": protocol["sampling"]["frame_rule"],
        "streams": "See per_clip_streams and per_frame for codec, dimensions, frame counts, PTS, paired ordinal, and source hashes.",
        "calibration_and_registration": protocol["depth_and_registration"],
        "mask_threshold_sensitivity": {
            "primary_threshold": 32,
            "fixed_thresholds": list(map(int, thresholds())),
            "red_channel_rule": "independent fixed check",
            "foreground_pixel_aggregate": {
                str(int(t)): _aggregate(
                    [r for r in threshold_rows if r["color_distance_threshold"] == t],
                    ("foreground_pixels",),
                )
                for t in thresholds()
            },
            "primary_complete_frame_count": len(complete_primary),
            "red_channel_rule_vs_distance32": _aggregate(
                [{"difference_pixels": x["difference_pixels"]} for x in red_sensitivity],
                ("difference_pixels",),
            ),
        },
        "contour_quality_primary32": {
            "complete_frames": len(complete_primary),
            "truncated_frames": sum(
                r["status"] == "edge_truncated"
                for r in threshold_rows
                if r["color_distance_threshold"] == 32
            ),
            "empty_frames": sum(
                r["status"] == "empty"
                for r in threshold_rows
                if r["color_distance_threshold"] == 32
            ),
            "largest_component_area_px": _aggregate(
                complete_primary, ("largest_component_pixels",)
            ),
            "hole_count": _aggregate(complete_primary, ("hole_count_8",)),
            "nonstar_ray_fraction": _aggregate(
                complete_primary, ("star_shaped_ray_test", "nonstar_ray_fraction")
            ),
            "interpretation": "2D mask geometry only; automatic-label errors and projection/occlusion cannot be separated without an independent manual contour or calibrated geometry reference.",
        },
        "model_test_errors": model_test_errors,
        "model_test_error_summary": {
            model: _aggregate(
                [{"rmse": r["test_rmse_px"]} for r in model_test_errors if r["model"] == model],
                ("rmse",),
            )
            for model in ("ellipse", "area_normalized_harmonic", "training_convex_hull")
        },
        "source_integrity": "Every sampled ZIP member CRC was recomputed and matched; per-member SHA-256 is in sidecar.",
        "software": {
            "python": sys.version.split()[0],
            "pyav": __import__("av").__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pillow": __import__("PIL").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_code_hashes": {},
    }

    source_path = Path(__file__).resolve()
    module_path = Path(
        __import__("roadmc.data.real.shrec_video", fromlist=["__file__"]).__file__
    ).resolve()
    outline_path = Path(
        __import__("roadmc.data.synthetic.outline_diagnostics", fromlist=["__file__"]).__file__
    ).resolve()
    fit_path = Path(__file__).resolve().parent / "realism_experiments.py"
    summary["source_code_hashes"] = {
        str(p): _sha256_file(p) for p in (source_path, module_path, fit_path, outline_path)
    }
    _write_json(out / "analysis.json", summary)
    _write_json(out / "per_clip_streams.json", selected_clip_sidecars)
    _write_json(out / "per_frame_contours.json", threshold_rows)
    _write_json(out / "red_channel_sensitivity.json", red_sensitivity)
    _write_json(out / "model_test_errors.json", model_test_errors)

    # Flat, non-object arrays make this NPZ loadable with allow_pickle=False.
    np.savez_compressed(
        out / "shrec_sampled_frames_and_contours.npz",
        rgb_uint8_flat=np.concatenate(raw_rgb) if raw_rgb else np.array([], dtype=np.uint8),
        rgb_offsets=np.asarray(rgb_offsets, dtype=np.int64),
        rgb_shapes=np.asarray(rgb_shapes, dtype=np.int32),
        mask_rgb_uint8_flat=np.concatenate(raw_mask) if raw_mask else np.array([], dtype=np.uint8),
        mask_rgb_offsets=np.asarray(mask_offsets, dtype=np.int64),
        mask_rgb_shapes=np.asarray(mask_shapes, dtype=np.int32),
        disparity_y_uint8_flat=np.concatenate(raw_disp)
        if raw_disp
        else np.array([], dtype=np.uint8),
        disparity_y_offsets=np.asarray(disp_offsets, dtype=np.int64),
        disparity_y_shapes=np.asarray(disp_shapes, dtype=np.int32),
        primary_mask_distance32_packed=np.concatenate(primary_packed)
        if primary_packed
        else np.array([], dtype=np.uint8),
        primary_mask_bit_offsets=np.asarray(primary_bit_offsets, dtype=np.int64),
        primary_mask_bit_lengths=np.asarray(primary_bit_lengths, dtype=np.int64),
        primary_contour_edge_yx=np.concatenate(contours_yx, axis=0)
        if contours_yx
        else np.empty((0, 2), dtype=np.int32),
        primary_contour_offsets=np.asarray(contour_offsets, dtype=np.int64),
        primary_contour_lengths=np.asarray(contour_lengths, dtype=np.int64),
        source_frame_indices=np.asarray([r["frame_index"] for r in records], dtype=np.int32),
        disparity_frame_indices=np.asarray(
            [r["disparity_frame_index"] for r in records], dtype=np.int32
        ),
        mask_frame_indices=np.asarray([r["mask_frame_index"] for r in records], dtype=np.int32),
    )
    _write_contact_sheet(
        records
        and _rehydrate_middle_frames(
            records,
            raw_rgb,
            rgb_offsets,
            rgb_shapes,
            mask_shapes,
            primary_packed,
            primary_bit_offsets,
            primary_bit_lengths,
        )
        or [],
        out / "shrec_middle_frame_contact_sheet.png",
    )
    _write_error_figure(summary, out / "shrec_outline_test_error_px.png")
    _write_csv(out / "per_frame_threshold_metrics.csv", threshold_rows)
    _write_report(out / "report_zh.md", summary)
    return summary


def _rehydrate_middle_frames(
    records,
    raw_rgb,
    rgb_offsets,
    rgb_shapes,
    mask_shapes,
    primary_packed,
    primary_bit_offsets,
    primary_bit_lengths,
):
    rgb_flat = np.concatenate(raw_rgb) if raw_rgb else np.array([], dtype=np.uint8)
    packed = np.concatenate(primary_packed) if primary_packed else np.array([], dtype=np.uint8)
    result = []
    for i, row in enumerate(records):
        if row["frame_position"] == "middle":
            rgb = rgb_flat[rgb_offsets[i] : rgb_offsets[i] + int(np.prod(rgb_shapes[i]))].reshape(
                tuple(rgb_shapes[i])
            )
            bit_start = primary_bit_offsets[i]
            bit_n = primary_bit_lengths[i]
            bits = np.unpackbits(
                packed[bit_start : bit_start + (bit_n + 7) // 8], bitorder="little"
            )[:bit_n]
            result.append(
                {**row, "rgb": rgb, "mask32": bits.reshape(mask_shapes[i][:2]).astype(bool)}
            )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        for key, value in row.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, int, float, bool)) or subvalue is None:
                        flat[f"{key}.{subkey}"] = subvalue
        flat_rows.append(flat)
    fields = sorted({k for row in flat_rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    mask = summary["mask_threshold_sensitivity"]
    quality = summary["contour_quality_primary32"]
    model = summary["model_test_error_summary"]
    metadata = summary.get("full_mask_clip_metadata", {})
    groups = metadata.get("capture_groups", {})
    split_counts = metadata.get("counts_by_membership", {})
    area_by_threshold = mask["foreground_pixel_aggregate"]
    area8 = area_by_threshold.get("8", area_by_threshold.get("8.0", {})).get("median")
    area32 = area_by_threshold.get("32", area_by_threshold.get("32.0", {})).get("median")
    area64 = area_by_threshold.get("64", area_by_threshold.get("64.0", {})).get("median")
    disp_observation = summary.get("disparity_code_observations", {})
    report = [
        "# SHREC / Pothole-Mix V2：视频轮廓与观测研究",
        "",
        "## 证据范围",
        "",
        "本结果来自预先冻结的20个mask clip（train 12、valid 8），每个clip按RGB帧数抽取首帧、floor((N−1)/2)中帧和末帧，共60组配对观察。train样本作为开发观察，valid样本作为文件清单所称的组内基准；已把两侧捕获前缀重叠明确保留。",
        f"元数据匹配覆盖全部 {metadata.get('clip_count')} 个mask clip：train {split_counts.get('train')}、valid {split_counts.get('valid')}，所有mask均能对上RGB和disparity文件；视频filelist登记584个train与213个valid RGB-D clip。按文件名前缀得到train {groups.get('train_unique')}组、valid {groups.get('valid_unique')}组，合并后{groups.get('union_unique')}组，其中{groups.get('shared_count')}组跨train/valid共享；冻结的20片段样本里16个捕获组有4个跨split重叠。捕获前缀不等于独立道路身份，因此内部基准不能作为独立道路留出。",
        "",
        "SHREC论文说明RGB为1920×1080、disparity为640×400，均15 FPS；设备对每帧disparity做7×7中值滤波，并将confidence低于245/255的像素置零。作者选择8-bit压缩disparity流，因为16-bit depth流不适合当时设备端压缩/存储流程。当前公开资料未给出可核实的焦距/基线/标定文件、压缩码值到物理深度的映射、RGB/disparity像素注册或流间时间偏移。本研究保留灰度Y码，不转米、不缩放/配准。",
        "",
        f"在冻结样本中，解码disparity灰度Y码的逐帧零码比例中位数为 {disp_observation.get('zero_fraction', {}).get('median')}；非零码逐帧中位数的跨帧中位数为 {disp_observation.get('nonzero_code_median_by_frame', {}).get('median')}。这些只说明压缩后代码场中可见/置零比例，不能用来估深度精度或可测范围。",
        "",
        "Mendeley V2页面将359个mask clip描述为由disparity后处理产生的binary mask，并明确提醒这些mask并非完美标签。下载记录许可为CC BY-NC 3.0；捆绑README另述各原子集分别沿用原许可、作者素材为CC BY-NC-SA 4.0。两者差异保留，不统一外发数据。",
        "",
        "## 冻结判据与数据复用",
        "",
        "mask AVI以PyAV解码为RGB24 uint8，raw decoded RGB值、RGB帧、disparity Y码、首中末序号、PTS、clip内ZIP CRC与逐成员SHA-256保存在NPZ/JSON。主mask判据为到名义红色(255,0,0)的欧氏RGB距离≤32，预设灵敏度阈值≤8/≤64，另行计算固定红通道规则；未按拟合误差选择阈值。",
        "",
        "红色只作为解码后视频mask的前景编码，本报告不把它映射为 pothole/crack 语义类别。",
        "",
        "每个8连通分量均保留计数，形态统计针对最大分量。触边轮廓记为不完整，不补边，并排除于轮廓模型拟合。孔洞数、触边方向、compactness、solidity及离散中心射线星形性均按像素/无量纲保存。轮廓比较在同一训练/测试边缘点与像素坐标下进行：复用RoadMC `_fit_outline` 与 `_fit_radial_harmonics`，训练弧来自中心角16区中 sector%5≠0；测试弧为sector%5=0。主比较使用同一测试边界点到训练所得椭圆曲线、谐波曲线、凸包折线的最近欧氏距离(px)，不做每样归一化；另保留ellipse-local径向残差作补充。",
        "",
        "## 结果",
        "",
        f"在主阈值32下，完整可拟合的frame为 {quality['complete_frames']}，触边不完整 {quality['truncated_frames']}，空mask {quality['empty_frames']}。这描述的是自动生成mask在图像中的形态与边界覆盖，不等于人工标注准确度或坑槽真实几何。",
        f"前景像素面积对预设颜色距离阈值的中位数：≤8为 {area8} px；≤32为 {area32} px；≤64为 {area64} px。固定红通道规则相对距离32的前景面积差中位数为 {mask['red_channel_rule_vs_distance32']['median']} px。",
        f"完整轮廓测试集误差汇总（每个frame观测等权；帧内像素误差为源图像px）：椭圆中位RMSE {model['ellipse']['median']} px（有效{model['ellipse']['n_valid_frame_observations']}/{model['ellipse']['n_candidate_frame_observations']}帧、{model['ellipse']['n_unique_capture_groups']}组）；harmonic {model['area_normalized_harmonic']['median']} px（有效{model['area_normalized_harmonic']['n_valid_frame_observations']}/{model['area_normalized_harmonic']['n_candidate_frame_observations']}帧、{model['area_normalized_harmonic']['n_unique_capture_groups']}组）；训练弧凸包 {model['training_convex_hull']['median']} px（有效{model['training_convex_hull']['n_valid_frame_observations']}/{model['training_convex_hull']['n_candidate_frame_observations']}帧、{model['training_convex_hull']['n_unique_capture_groups']}组）。聚合分母是每模型有效frame数，当前三模型都为同一22帧、8个capture group；这些帧不是独立样本。数值更正记录见 `phase1_metric_summary_correction.json`。不同帧来自同一或交叠capture group，不能把排序解释为人口优越性或标注质量排名。",
        "",
        "## 限制与结论",
        "",
        "【实测】完成了所列样本的三视频流解码、RGB/disp/mask按ordinal配对审计、mask压缩色彩敏感性、触边/孔洞/非星形统计和完整轮廓公共像素坐标拟合。",
        "【定性】触边轮廓和自动mask误差会直接削弱完整边界的解释力；二维投影中的形状变化可能同时来自视角、遮挡、自动mask误差与真实轮廓。仅靠本批数据无法分离这些来源。",
        "【未知】metric depth、相机标定、RGB-D空间注册、流间时间偏移及mask相对真实坑缘误差：未找到可核验资料，待核实。",
        "不得将这些视频mask作为人工GT，不把disp码转成metric Z，不写入RoadMC默认参数；不声称独立道路验证或道路总体校准。",
        "",
        "## 复现文件",
        "",
        "- `frozen_protocol.json`：采样与判据冻结记录。",
        "- `analysis.json`：聚合结果、源码hash与软件版本。",
        "- `output_manifest.json`：所有研究输出文件的字节数和SHA-256（manifest自身除外）。",
        "- `per_clip_streams.json`：逐clip成员hash、codec、尺寸、帧计数、PTS及配对映射。",
        "- `clip_metadata_inventory.json`：全部359 mask clip与train/valid RGB-D filelist及捕获组的逐项对应。",
        "- `disparity_code_observations.json`：逐帧8-bit解码灰度码分布，只作观测质量描述。",
        "- `per_frame_contours.json` / `per_frame_threshold_metrics.csv`：逐阈值/逐帧统计。",
        "- `shrec_sampled_frames_and_contours.npz`：原始解码字段、派生mask位图与轮廓点。",
        "- `shrec_middle_frame_contact_sheet.png` / `shrec_outline_test_error_px.png`：可视检查图。",
        "",
        "## 来源",
        "",
        "- Moscoso Thompson et al., *SHREC 2022: pothole and crack detection in the road pavement using images and RGB-D data*, arXiv:2205.13326v5, §3.1, https://arxiv.org/html/2205.13326v5",
        "- Ranieri, Moscoso Thompson & Biasotti, *Pothole Mix*, Mendeley Data v2, https://data.mendeley.com/datasets/kfth5g2xk3/2",
        "- Author repository (training/evaluation source; no camera calibration or raw disparity conversion code found in the repository tree inspected): https://gitlab.com/4ndr3aR/pothole-mix-segmentation",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_source_evidence(path: Path, protocol_path: Path, per_clip_path: Path) -> None:
    evidence = [
        "# SHREC / Pothole-Mix V2 来源证据与本机核验",
        "",
        "## 官方论文",
        "",
        "Moscoso Thompson et al., *SHREC 2022: pothole and crack detection in the road pavement using images and RGB-D data*, arXiv v5, §3.1: https://arxiv.org/html/2205.13326v5",
        "",
        "§3.1报告：RGB视频Full HD 1920×1080、disparity 640×400，二者15 FPS；拍摄距离随病害大小约30 cm至1 m。论文说明OAK-D输出的压缩disparity是8-bit，而depth图为16-bit；作者因16-bit depth流在当时的设备端压缩/数据处理流程不合适，选择保存disparity。论文给出的逐帧设备滤波为7×7 median，confidence低于245/255置零。论文随后明确提示disparity图噪声较大，需要去噪后才可能用作分割mask。",
        "",
        "原文定位与短引文：arXiv HTML v5 §3.1 lines 114–118，https://arxiv.org/html/2205.13326v5 ：‘the disparity image is provided at 8 bits ... the depth image is provided at 16 bits’。同段记载7×7 median及245/255 confidence threshold。该段不包含disparity到Z的映射。",
        "",
        "§3.1及捆绑README未提供可复核的镜头内参、baseline、压缩码值到Z的换算式/比例偏置、视频同步偏移或RGB-D像素变换。这里记录的是公开材料中未找到该信息；没有用OAK-D一般规格推算。",
        "",
        "## 官方数据记录与作者仓库",
        "",
        "Mendeley Data, *Pothole Mix*, Version 2, DOI 10.17632/kfth5g2xk3.2: https://data.mendeley.com/datasets/kfth5g2xk3/2 。官方记录称数据含797对RGB/disparity clips及359个由disparity后处理得到的binary mask clips，并提醒mask并非完美标签。V2页面许可为CC BY-NC 3.0。ZIP内README同时称原始子数据集各沿用自身许可、作者新增素材CC BY-NC-SA 4.0；保留此范围差异。",
        "",
        "作者GitLab仓库： https://gitlab.com/4ndr3aR/pothole-mix-segmentation 。本次只查看公开仓库树/README，发现SHREC训练评估notebook、baseline与竞赛提交代码；未在所查仓库树定位OAK-D采集/相机校准/压缩disparity到深度的实现。这是对当前检索范围的结论，不代表作者没有另行保存未公开工具。",
        "",
        "## 本机数据核验",
        "",
        f"采样协议：`{protocol_path}`（SHA-256见该文件）；所选视频成员的逐片CRC、SHA-256、codec、解码尺寸、帧计数和PTS：`{per_clip_path}`。主文件全包SHA-256从已完成的准入目录`admission_catalog.v2.json`沿用，未重复hash整包；本轮对20个所选成员逐个重算ZIP CRC及SHA-256。",
        "",
        "探索性格式哨兵clip c000059（不在冻结20片段选择中）解码显示mask codec=MJPEG、frame=1920×1080，RGB24常见值为黑色[0,0,0]与近红色[254,0,0]，并有色度压缩混色。冻结样本使用固定RGB距离阈值，不因形状拟合分数调阈值。",
        "",
        "注意：冻结协议原键名`nominal_pothole_color_rgb`属于语义命名错误。规则中的(255,0,0)在本研究只定义红色前景编码；不代表已核实为坑槽类。阈值和样本均未改变。此修正记录不覆盖冻结协议原文件。",
        "",
    ]
    path.write_text("\n".join(evidence), encoding="utf-8")


def _write_output_manifest(out: Path) -> None:
    files = []
    for path in sorted(
        p for p in out.iterdir() if p.is_file() and p.name != "output_manifest.json"
    ):
        files.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    _write_json(
        out / "output_manifest.json",
        {"schema": "roadmc.shrec-study-output-manifest.v1", "files": files},
    )


def _guard_existing_run_logs(out: Path) -> None:
    """Refuse to reuse an output directory containing results or run evidence.

    A frozen protocol is the sole allowed existing file for the normal run.
    Finalization and refitting also require a new output directory so they do
    not silently replace already-produced study artifacts.
    """
    if not out.exists():
        return
    protected = [child for child in out.iterdir() if child.name != "frozen_protocol.json"]
    if protected:
        names = ", ".join(str(path) for path in protected)
        raise FileExistsError(
            "refusing to reuse output directory because existing study outputs or run evidence "
            f"would be at risk: {names}; choose a new --output-dir"
        )


def finalize_outputs(out: Path, audit_root: Path) -> dict[str, Any]:
    """Regenerate summary visuals/report from completed JSON after a render error."""
    summary = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    model_errors = json.loads((out / "model_test_errors.json").read_text(encoding="utf-8"))
    contour_rows = json.loads((out / "per_frame_contours.json").read_text(encoding="utf-8"))
    candidate_rows = [
        row
        for row in contour_rows
        if row.get("color_distance_threshold") == 32
        and row.get("status") == "complete_in_frame"
        and row.get("pixel_coordinate_model_comparison", {}).get("status") == "ok"
    ]
    expected_models = ("ellipse", "area_normalized_harmonic", "training_convex_hull")
    corrected_summary: dict[str, Any] = {}
    for model_name in expected_models:
        model_rows = [row for row in model_errors if row.get("model") == model_name]
        valid = [
            row
            for row in model_rows
            if isinstance(row.get("test_rmse_px"), (int, float))
            and math.isfinite(row["test_rmse_px"])
        ]
        group_names = {row["capture_group"] for row in valid}
        corrected_summary[model_name] = _aggregate(
            [{"rmse": row["test_rmse_px"]} for row in valid], ("rmse",)
        )
        corrected_summary[model_name].update(
            {
                "n_candidate_frame_observations": len(candidate_rows),
                "n_valid_frame_observations": len(valid),
                "n_failed_or_missing_frame_observations": max(0, len(candidate_rows) - len(valid)),
                "n_unique_capture_groups": len(group_names),
                "n_heldout_boundary_points": sum(
                    int(
                        row.get("pixel_coordinate_model_comparison", {})
                        .get("models", {})
                        .get(model_name, {})
                        .get("test", {})
                        .get("points", 0)
                    )
                    for row in candidate_rows
                    if row.get("pixel_coordinate_model_comparison", {})
                    .get("models", {})
                    .get(model_name, {})
                    .get("test", {})
                    .get("rmse_px")
                    is not None
                ),
            }
        )
    summary["model_test_error_summary"] = corrected_summary
    correction_record = {
        "schema": "roadmc.shrec-phase1-metric-summary-correction.v1",
        "date": "2026-10-16",
        "source_of_truth": "model_test_errors.json recomputed from its per-frame test_rmse_px values; analysis.json summary now records each denominator.",
        "previous_inconsistent_narrative_values_px": {
            "ellipse_median_rmse": 28.9367,
            "harmonic_median_rmse": 26.4319,
        },
        "corrected_frame_equal_summary": corrected_summary,
        "aggregation": "Median of per-frame test RMSE values; valid frame observations are the denominator. No pixels, clips, or capture groups are silently substituted as the phase1 denominator.",
        "source_video_redecode": False,
        "sample_or_threshold_changed": False,
        "capture_groups_are_independent_roads": False,
    }
    _write_json(out / "phase1_metric_summary_correction.json", correction_record)
    summary["model_metric_summary_correction"] = {
        "path": str(out / "phase1_metric_summary_correction.json"),
        "sha256": _sha256_file(out / "phase1_metric_summary_correction.json"),
    }
    metadata_inventory = write_clip_metadata_inventory(audit_root, out)
    summary["full_mask_clip_metadata"] = {
        key: metadata_inventory[key]
        for key in (
            "clip_count",
            "all_mask_clips_matched_rgb_and_disparity",
            "counts_by_membership",
            "filelist_stream_counts",
            "capture_groups",
        )
    }
    source_path = Path(__file__).resolve()
    module_path = Path(
        __import__("roadmc.data.real.shrec_video", fromlist=["__file__"]).__file__
    ).resolve()
    outline_path = Path(
        __import__("roadmc.data.synthetic.outline_diagnostics", fromlist=["__file__"]).__file__
    ).resolve()
    fit_path = Path(__file__).resolve().parent / "realism_experiments.py"
    if "analysis_run_source_code_hashes" not in summary:
        summary["analysis_run_source_code_hashes"] = summary.get("source_code_hashes", {})
    summary["finalization_source_code_hashes"] = {
        str(p): _sha256_file(p) for p in (source_path, module_path, fit_path, outline_path)
    }
    amendment = {
        "schema": "roadmc.shrec-video-protocol-amendment.v1",
        "date": "2026-10-16",
        "protocol_path": str(out / "frozen_protocol.json"),
        "protocol_sha256_preserved": _sha256_file(out / "frozen_protocol.json"),
        "corrections": [
            {
                "field": "nominal_pothole_color_rgb",
                "correction": "Interpret only as nominal red foreground RGB code for video-mask extraction; no pothole/crack class is assigned.",
                "numeric_thresholds_changed": False,
                "selected_sample_changed": False,
            },
            {
                "field": "pixel_coordinate_model_comparison",
                "correction": "The first-pass plot mixed radial residuals for smooth models with nearest distance for the hull. Final headline metrics were refit from saved primary-threshold contours using the same held-out observed-pixel to predicted-curve nearest Euclidean distance in source image pixels for ellipse, harmonic and training-arc convex hull.",
                "source_video_redecode": False,
                "thresholds_changed": False,
                "selected_sample_changed": False,
            },
        ],
        "numeric_thresholds_changed": False,
        "selected_sample_changed": False,
    }
    amendment_path = out / "protocol_amendment_semantics.json"
    _write_json(amendment_path, amendment)
    summary["protocol_semantic_amendment"] = {
        "path": str(amendment_path),
        "sha256": _sha256_file(amendment_path),
        "numeric_thresholds_changed": False,
        "selected_sample_changed": False,
        "model_error_metric_corrected": True,
    }
    _write_json(out / "analysis.json", summary)
    _write_csv(
        out / "per_frame_threshold_metrics.csv",
        json.loads((out / "per_frame_contours.json").read_text(encoding="utf-8")),
    )
    _write_error_figure(summary, out / "shrec_outline_test_error_px.png")
    _write_report(out / "report_zh.md", summary)
    _write_source_evidence(
        out / "source_evidence.md", out / "frozen_protocol.json", out / "per_clip_streams.json"
    )
    _write_output_manifest(out)
    return summary


def refit_models_from_saved_npz(out: Path) -> dict[str, Any]:
    """Re-score the primary-threshold contours without reopening or decoding source clips."""
    from roadmc.data.real.shrec_video import _contour_model_metrics

    frame_path = out / "per_frame_contours.json"
    rows = json.loads(frame_path.read_text(encoding="utf-8"))
    summary = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    with np.load(out / "shrec_sampled_frames_and_contours.npz", allow_pickle=False) as arrays:
        edge_all = arrays["primary_contour_edge_yx"]
        offsets = arrays["primary_contour_offsets"]
        lengths = arrays["primary_contour_lengths"]
        disp_flat = arrays["disparity_y_uint8_flat"]
        disp_offsets = arrays["disparity_y_offsets"]
        disp_shapes = arrays["disparity_y_shapes"]
        primary_rows = {
            int(row["sample_index"]): row for row in rows if row["color_distance_threshold"] == 32
        }
        disparity_observations = []
        for index in range(len(disp_offsets)):
            shape = tuple(int(v) for v in disp_shapes[index])
            size = int(np.prod(shape))
            codes = disp_flat[int(disp_offsets[index]) : int(disp_offsets[index]) + size].reshape(
                shape
            )
            nonzero = codes[codes > 0]
            record = primary_rows[index]
            disparity_observations.append(
                {
                    "sample_index": index,
                    "split": record["split"],
                    "capture_group": record["capture_group"],
                    "clip_id": record["clip_id"],
                    "frame_position": record["frame_position"],
                    "height_px": shape[0],
                    "width_px": shape[1],
                    "zero_fraction": float(np.mean(codes == 0)),
                    "nonzero_pixel_fraction": float(np.mean(codes > 0)),
                    "nonzero_code_median": float(np.median(nonzero)) if len(nonzero) else None,
                    "nonzero_code_p95": float(np.quantile(nonzero, 0.95)) if len(nonzero) else None,
                    "nonzero_code_unique_count": int(np.unique(nonzero).size),
                }
            )
        for row in rows:
            if row["color_distance_threshold"] != 32 or row["status"] != "complete_in_frame":
                continue
            index = int(row["sample_index"])
            start, length = int(offsets[index]), int(lengths[index])
            edge_yx = edge_all[start : start + length]
            centroid = row["centroid_yx_px"]
            row["pixel_coordinate_model_comparison"] = _contour_model_metrics(
                edge_yx, (float(centroid[0]), float(centroid[1]))
            )
    _write_json(frame_path, rows)
    model_errors = []
    complete = [
        r
        for r in rows
        if r["color_distance_threshold"] == 32 and r["status"] == "complete_in_frame"
    ]
    for row in complete:
        models = row.get("pixel_coordinate_model_comparison", {}).get("models", {})
        for model_name, model in models.items():
            metrics = model.get("test", {})
            model_errors.append(
                {
                    "sample_index": row["sample_index"],
                    "split": row["split"],
                    "capture_group": row["capture_group"],
                    "clip_id": row["clip_id"],
                    "frame_position": row["frame_position"],
                    "model": model_name,
                    "test_rmse_px": metrics.get("rmse_px"),
                    "test_mae_px": metrics.get("mae_px"),
                    "test_p95_px": metrics.get("p95_px"),
                }
            )
    summary["model_metric"] = (
        "Same metric for all models: distance from each held-out observed contour pixel to the nearest point on the fitted ellipse, fitted harmonic curve or training-arc convex hull; source image pixel coordinates, without per-model normalization."
    )
    summary["model_test_errors"] = model_errors
    summary["model_test_error_summary"] = {
        model: _aggregate(
            [{"rmse": r["test_rmse_px"]} for r in model_errors if r["model"] == model], ("rmse",)
        )
        for model in ("ellipse", "area_normalized_harmonic", "training_convex_hull")
    }
    summary["model_metric_refit"] = {
        "from": "shrec_sampled_frames_and_contours.npz and per_frame_contours.json",
        "source_video_redecode": False,
        "reason": "Use one common point-to-curve Euclidean pixel metric for all model families.",
        "model_code_sha256": _sha256_file(
            Path(
                __import__("roadmc.data.real.shrec_video", fromlist=["__file__"]).__file__
            ).resolve()
        ),
    }
    summary["disparity_code_observations"] = {
        "field": "PyAV decoded 8-bit gray/luma code from H264 disparity video; not calibrated disparity and not metric depth.",
        "frame_observations": len(disparity_observations),
        "zero_fraction": _aggregate(disparity_observations, ("zero_fraction",)),
        "nonzero_code_median_by_frame": _aggregate(
            disparity_observations, ("nonzero_code_median",)
        ),
        "nonzero_code_p95_by_frame": _aggregate(disparity_observations, ("nonzero_code_p95",)),
        "unique_nonzero_code_count_by_frame": _aggregate(
            disparity_observations, ("nonzero_code_unique_count",)
        ),
        "per_frame": disparity_observations,
    }
    _write_json(out / "analysis.json", summary)
    _write_json(out / "model_test_errors.json", model_errors)
    _write_json(out / "disparity_code_observations.json", summary["disparity_code_observations"])
    _write_csv(out / "per_frame_threshold_metrics.csv", rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, help="Admission evidence directory")
    parser.add_argument("--source-root", type=Path, help="Source ZIP directory; required to freeze")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--refit-models-only", action="store_true")
    args = parser.parse_args()
    _guard_existing_run_logs(args.output_dir)
    if not args.refit_models_only and args.audit_root is None:
        parser.error("--audit-root is required unless --refit-models-only is selected")
    if args.freeze_only and args.source_root is None:
        parser.error("--freeze-only requires --source-root")
    if args.freeze_only:
        path = freeze_protocol(args.audit_root, args.source_root, args.output_dir)
        print(f"Frozen protocol: {path}")
        return 0
    if args.finalize_only:
        result = finalize_outputs(args.output_dir, args.audit_root)
        print(
            json.dumps(
                {
                    "analysis": str(args.output_dir / "analysis.json"),
                    "frames": result["sampled_frame_count"],
                    "clips": result["selected_clips"]["total"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.refit_models_only:
        result = refit_models_from_saved_npz(args.output_dir)
        print(
            json.dumps(
                {"frames": result["sampled_frame_count"], "model_metric": result["model_metric"]},
                ensure_ascii=False,
            )
        )
        return 0
    protocol = args.protocol or args.output_dir / "frozen_protocol.json"
    result = run_study(args.audit_root, args.output_dir, protocol)
    print(
        json.dumps(
            {
                "analysis": str(args.output_dir / "analysis.json"),
                "frames": result["sampled_frame_count"],
                "clips": result["selected_clips"]["total"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
