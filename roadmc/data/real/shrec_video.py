"""Bounded, reproducible readers and pixel-coordinate diagnostics for SHREC clips.

The RGB-D disparity stream is retained as decoded 8-bit luma. It is not
converted to metric depth because this source does not provide a verified
calibration/value mapping. MJPEG mask colors are preserved before thresholding.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
import zlib
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

import av
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, label
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, cKDTree

_MASK_RE = re.compile(r"(?P<capture>[a-z]-\d{8}-\d{6})-mask-c(?P<clip>\d+)\.avi$")
_RGB_RE = re.compile(r"(?P<capture>[a-z]-\d{8}-\d{6})-color-c(?P<clip>\d+)\.mp4$")
_THRESHOLDS = (8.0, 32.0, 64.0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_mask_name(name: str) -> tuple[str, int]:
    match = _MASK_RE.fullmatch(PurePosixPath(name).name)
    if not match:
        raise ValueError(f"unrecognized SHREC mask clip name: {name}")
    return match.group("capture"), int(match.group("clip"))


def select_mask_clips(
    mask_members: list[str],
    train_names: list[str],
    valid_names: list[str],
    *,
    n_train: int = 12,
    n_valid: int = 8,
) -> list[dict[str, Any]]:
    """Select evenly spaced capture groups and the within-group median clip ID.

    Split membership comes only from the supplied train/valid filelists. A
    capture prefix may occur in both; this routine preserves that overlap.
    """
    lists: dict[str, set[str]] = {}
    for split, names in (("train", train_names), ("valid", valid_names)):
        mapped = set()
        for entry in names:
            basename = PurePosixPath(entry).name
            match = _RGB_RE.fullmatch(basename)
            if match:
                mapped.add(f"{match.group('capture')}-mask-c{int(match.group('clip')):06d}.avi")
        lists[split] = mapped

    by_split: dict[str, dict[str, list[tuple[int, str]]]] = {"train": {}, "valid": {}}
    for member in mask_members:
        basename = PurePosixPath(member).name
        capture, clip_id = parse_mask_name(basename)
        for split in ("train", "valid"):
            if basename in lists[split]:
                by_split[split].setdefault(capture, []).append((clip_id, basename))
    selected: list[dict[str, Any]] = []
    for split, count in (("train", n_train), ("valid", n_valid)):
        groups = sorted(by_split[split])
        if len(groups) < count:
            raise ValueError(f"need {count} {split} capture groups, found {len(groups)}")
        # Integer nearest-rank quantiles; group positions are distinct because count <= n.
        positions = [round(i * (len(groups) - 1) / (count - 1)) for i in range(count)]
        for position in positions:
            capture = groups[position]
            candidates = sorted(by_split[split][capture])
            median_id = float(np.median([c[0] for c in candidates]))
            clip_id, basename = min(candidates, key=lambda row: (abs(row[0] - median_id), row[0]))
            selected.append(
                {
                    "split": split,
                    "capture_group": capture,
                    "clip_id": clip_id,
                    "mask_basename": basename,
                    "rgb_basename": f"{capture}-color-c{clip_id:06d}.mp4",
                    "disp_basename": f"{capture}-disp-c{clip_id:06d}.mp4",
                    "capture_group_occurs_in_both_splits": (
                        capture in by_split["train"] and capture in by_split["valid"]
                    ),
                }
            )
    return selected


def selected_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("video frame count must be positive")
    return sorted({0, (frame_count - 1) // 2, frame_count - 1})


def _frame_count(stream: av.video.stream.VideoStream, data: bytes) -> int:
    if stream.frames and stream.frames > 0:
        return int(stream.frames)
    # For containers without an indexed frame count, decode once to count frames.
    with av.open(io.BytesIO(data)) as container:
        return sum(1 for _ in container.decode(video=0))


def decode_video_samples(
    data: bytes, target_indices: list[int], *, role: str
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Decode selected ordinals while auditing the full frame count and timestamps."""
    with av.open(io.BytesIO(data)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"expected one video stream, got {len(container.streams.video)}")
        stream = container.streams.video[0]
        frame_count = _frame_count(stream, data)
        targets = set(target_indices)
        samples: dict[int, np.ndarray] = {}
        selected_pts: dict[int, float | None] = {}
        decoded = 0
        for frame in container.decode(stream):
            index = decoded
            decoded += 1
            if index in targets:
                fmt = "gray" if role == "disparity" else "rgb24"
                samples[index] = frame.to_ndarray(format=fmt)
                selected_pts[index] = float(frame.time) if frame.time is not None else None
        info = {
            "codec": stream.codec_context.name,
            "pixel_format": stream.codec_context.pix_fmt,
            "width": int(stream.width),
            "height": int(stream.height),
            "header_frame_count": int(stream.frames),
            "decoded_frame_count": int(decoded),
            "average_rate": str(stream.average_rate) if stream.average_rate else None,
            "time_base": str(stream.time_base) if stream.time_base else None,
            "selected_pts_seconds": {str(k): selected_pts.get(k) for k in target_indices},
            "decode_policy": "PyAV frame decode; rgb24 for RGB/mask; gray Y/luma conversion for disparity",
        }
    if decoded != frame_count:
        info["header_frame_count_mismatch"] = True
    missing = targets - samples.keys()
    if missing:
        raise ValueError(f"selected {role} frames missing: {sorted(missing)}")
    return info, samples


def mask_color_distance(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Return pixels within fixed Euclidean RGB distance of the nominal red foreground code."""
    value = np.asarray(rgb, dtype=np.int16)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("mask RGB frame must have shape (height,width,3)")
    if threshold not in _THRESHOLDS:
        raise ValueError(f"threshold must be one of frozen values {_THRESHOLDS}")
    delta = value - np.array([255, 0, 0], dtype=np.int16)
    return np.sum(delta.astype(np.int32) ** 2, axis=-1) <= threshold**2


def mask_red_channel_rule(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("mask RGB frame must have shape (height,width,3)")
    r, g, b = value[..., 0], value[..., 1], value[..., 2]
    return (
        (r >= 128)
        & (g <= 64)
        & (b <= 64)
        & ((r.astype(np.int16) - g) >= 96)
        & ((r.astype(np.int16) - b) >= 96)
    )


def _score_pixel_errors(errors: np.ndarray) -> dict[str, float | int | None]:
    err = np.asarray(errors, dtype=float)
    err = err[np.isfinite(err)]
    if not len(err):
        return {"points": 0, "rmse_px": None, "mae_px": None, "p95_px": None}
    return {
        "points": int(len(err)),
        "rmse_px": float(np.sqrt(np.mean(err**2))),
        "mae_px": float(np.mean(np.abs(err))),
        "p95_px": float(np.quantile(np.abs(err), 0.95)),
    }


def _ellipse_local(q: np.ndarray, points: np.ndarray) -> np.ndarray:
    shifted = points - q[:2]
    ct, st = np.cos(q[4]), np.sin(q[4])
    return np.column_stack(
        (
            (ct * shifted[:, 0] + st * shifted[:, 1]) / np.exp(q[2]),
            (-st * shifted[:, 0] + ct * shifted[:, 1]) / np.exp(q[3]),
        )
    )


def _transform_ellipse(q: np.ndarray, uv: np.ndarray) -> np.ndarray:
    ct, st = np.cos(q[4]), np.sin(q[4])
    a, b = np.exp(q[2]), np.exp(q[3])
    return np.column_stack(
        (q[0] + ct * a * uv[:, 0] - st * b * uv[:, 1], q[1] + st * a * uv[:, 0] + ct * b * uv[:, 1])
    )


def _contour_model_metrics(edge_yx: np.ndarray, centroid_yx: tuple[float, float]) -> dict[str, Any]:
    """Fit ellipse/harmonics to fixed training sectors and score test arcs in raw pixels."""
    from roadmc.data.synthetic.pothole_contour import radial_scale
    from roadmc.scripts.realism_experiments import _fit_outline, _fit_radial_harmonics

    points = np.column_stack((edge_yx[:, 1], edge_yx[:, 0])).astype(float)
    if len(points) < 40:
        return {"status": "insufficient_boundary_points", "boundary_points": int(len(points))}
    center = np.array([centroid_yx[1], centroid_yx[0]], dtype=float)
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * np.pi)
    sectors = np.floor(angles / (2 * np.pi) * 16).astype(int)
    test = sectors % 5 == 0
    train = ~test
    if train.sum() < 20 or test.sum() < 5:
        return {
            "status": "insufficient_fixed_arc_coverage",
            "train_points": int(train.sum()),
            "test_points": int(test.sum()),
        }
    q, _, _ = _fit_outline(points[train])
    uv = _ellipse_local(q, points)
    theta = np.mod(np.arctan2(uv[:, 1], uv[:, 0]), 2 * np.pi)
    radius = np.linalg.norm(uv, axis=1)
    harmonic, harmonic_residual = _fit_radial_harmonics(uv, train, test)
    coeff = np.asarray(
        harmonic["coefficients"]["cos"] + harmonic["coefficients"]["sin"], dtype=float
    )
    g = radial_scale(theta, {"cos": coeff[:3].tolist(), "sin": coeff[3:].tolist()})
    direction_scale = np.sqrt(
        (np.exp(q[2]) * np.cos(theta)) ** 2 + (np.exp(q[3]) * np.sin(theta)) ** 2
    )
    ellipse_error_px = (radius - 1.0) * direction_scale
    harmonic_error_px = (radius - g) * direction_scale
    curve_theta = np.linspace(0.0, 2 * np.pi, 8192, endpoint=False)
    unit_curve = np.column_stack((np.cos(curve_theta), np.sin(curve_theta)))
    harmonic_curve_scale = radial_scale(
        curve_theta, {"cos": coeff[:3].tolist(), "sin": coeff[3:].tolist()}
    )
    ellipse_curve = _transform_ellipse(q, unit_curve)
    harmonic_curve = _transform_ellipse(q, unit_curve * harmonic_curve_scale[:, None])
    ellipse_curve_distance = cKDTree(ellipse_curve).query(points[test], k=1)[0]
    harmonic_curve_distance = cKDTree(harmonic_curve).query(points[test], k=1)[0]

    hull_score: dict[str, Any]
    try:
        hull = ConvexHull(points[train])
        hull_points = points[train][hull.vertices]
        dense_hull: list[np.ndarray] = []
        for start, end in zip(hull_points, np.roll(hull_points, -1, axis=0), strict=True):
            count = max(1, int(np.ceil(np.linalg.norm(end - start))))
            dense_hull.append(
                start + np.linspace(0.0, 1.0, count, endpoint=False)[:, None] * (end - start)
            )
        hull_tree = cKDTree(np.concatenate(dense_hull, axis=0))
        hull_distance = hull_tree.query(points, k=1)[0]
        hull_score = {
            "status": "ok",
            "train_hull_vertices": int(len(hull_points)),
            "test": _score_pixel_errors(hull_distance[test]),
            "all_boundary": _score_pixel_errors(hull_distance),
        }
    except Exception as exc:  # Degenerate training arcs are retained as an explicit failed model.
        hull_score = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    model_curves = {
        "ellipse": {
            "train_radial_error_px": _score_pixel_errors(ellipse_error_px[train]),
            "test_radial_error_px": _score_pixel_errors(ellipse_error_px[test]),
            "test": _score_pixel_errors(ellipse_curve_distance),
        },
        "area_normalized_harmonic": {
            "train_radial_error_px": _score_pixel_errors(harmonic_error_px[train]),
            "test_radial_error_px": _score_pixel_errors(harmonic_error_px[test]),
            "test": _score_pixel_errors(harmonic_curve_distance),
            "coefficients": harmonic["coefficients"],
            "coefficient_pair_norm_sum": harmonic["coefficient_pair_norm_sum"],
        },
        "training_convex_hull": hull_score,
    }
    return {
        "status": "ok",
        "coordinate_system": "source image pixels x-right/y-down; no per-model normalization",
        "holdout_rule": "centroid-angle 16 equal sectors; sector index modulo 5 == 0 held out before fit",
        "training_points": int(train.sum()),
        "test_points": int(test.sum()),
        "ellipse_fit_center_logaxes_angle": q.tolist(),
        "models": model_curves,
    }


def contour_metrics(mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Describe the largest 8-connected component; report, never repair, truncation."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")
    labels, count = label(binary, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    if count == 0 or not sizes.max():
        return {"foreground_pixels": 0, "component_count_8": 0, "status": "empty"}, np.empty(
            (0, 2), dtype=np.int32
        )
    component_id = int(sizes.argmax())
    component = labels == component_id
    h, w = component.shape
    edge_flags = {
        "top": bool(component[0, :].any()),
        "bottom": bool(component[-1, :].any()),
        "left": bool(component[:, 0].any()),
        "right": bool(component[:, -1].any()),
    }
    edges = component & ~binary_erosion(
        component, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    area = int(component.sum())
    perimeter = int(edges.sum())
    filled = binary_fill_holes(component)
    holes = filled & ~component
    outer_edges = filled & ~binary_erosion(
        filled, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    edge_yx = np.argwhere(outer_edges).astype(np.int32)
    hole_labels, hole_count = label(holes, structure=np.ones((3, 3), dtype=bool))
    hole_sizes = np.bincount(hole_labels.ravel())[1:] if hole_count else np.array([], dtype=int)
    coords_yx = np.argwhere(component)
    centroid = coords_yx.mean(axis=0)
    star_rays = 720
    max_radius = float(np.hypot(h, w))
    angles = np.linspace(0.0, 2 * np.pi, star_rays, endpoint=False)
    ray_ok = np.ones(star_rays, dtype=bool)
    # Probe each ray at half-pixel intervals and require a continuous foreground prefix.
    for i, angle in enumerate(angles):
        rmax = max_radius
        rr = centroid[0] + np.arange(0.0, rmax, 0.5) * np.sin(angle)
        cc = centroid[1] + np.arange(0.0, rmax, 0.5) * np.cos(angle)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if not valid.any():
            continue
        occupied = component[
            np.clip(np.rint(rr[valid]).astype(int), 0, h - 1),
            np.clip(np.rint(cc[valid]).astype(int), 0, w - 1),
        ]
        false = np.flatnonzero(~occupied)
        if len(false) and occupied[false[0] + 1 :].any():
            ray_ok[i] = False
    convex_area = None
    try:
        hull = ConvexHull(np.column_stack((edge_yx[:, 1], edge_yx[:, 0])))
        convex_area = float(hull.volume)
    except Exception:
        pass
    if any(edge_flags.values()):
        fit = {"status": "excluded_edge_truncated"}
    else:
        try:
            fit = _contour_model_metrics(edge_yx, (float(centroid[0]), float(centroid[1])))
        except Exception as exc:
            fit = {"status": "fit_failed", "error": f"{type(exc).__name__}: {exc}"}
    return (
        {
            "status": "edge_truncated" if any(edge_flags.values()) else "complete_in_frame",
            "foreground_pixels": int(binary.sum()),
            "component_count_8": int(count),
            "largest_component_pixels": area,
            "largest_component_fraction_of_foreground": float(area / binary.sum()),
            "component_bbox_xywh_px": [
                int(coords_yx[:, 1].min()),
                int(coords_yx[:, 0].min()),
                int(coords_yx[:, 1].max() - coords_yx[:, 1].min() + 1),
                int(coords_yx[:, 0].max() - coords_yx[:, 0].min() + 1),
            ],
            "edge_truncation_sides": edge_flags,
            "outline_complete_for_fit": not any(edge_flags.values()),
            "hole_count_8": int(hole_count),
            "hole_pixels": int(holes.sum()),
            "largest_hole_pixels": int(hole_sizes.max()) if len(hole_sizes) else 0,
            "boundary_edge_pixels_8_neighbor": perimeter,
            "compactness_4piA_over_P2": float(4 * np.pi * area / perimeter**2)
            if perimeter
            else None,
            "convex_hull_area_px2": convex_area,
            "solidity_area_over_hull": float(area / convex_area) if convex_area else None,
            "centroid_yx_px": [float(centroid[0]), float(centroid[1])],
            "star_shaped_ray_test": {
                "status": "discrete_pixel_sampling_approximation",
                "rays": star_rays,
                "step_px": 0.5,
                "radially_contiguous_ray_fraction": float(ray_ok.mean()),
                "nonstar_ray_fraction": float(1.0 - ray_ok.mean()),
            },
            "pixel_coordinate_model_comparison": fit,
        },
        edge_yx,
    )


def read_member_verified(archive: zipfile.ZipFile, member: str) -> tuple[bytes, dict[str, Any]]:
    info = archive.getinfo(member)
    data = archive.read(member)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if crc != info.CRC:
        raise ValueError(f"ZIP member CRC mismatch for {member}: {crc:08x} != {info.CRC:08x}")
    return data, {
        "member": member,
        "bytes": len(data),
        "crc32": f"{crc:08x}",
        "sha256": sha256_bytes(data),
    }


def thresholds() -> tuple[float, ...]:
    return _THRESHOLDS


def point_to_closed_polyline_distances(
    points_xy: np.ndarray, vertices_xy: np.ndarray
) -> np.ndarray:
    """Return exact Euclidean distances from points to the closed polygon edges."""
    points = np.asarray(points_xy, dtype=float)
    vertices = np.asarray(vertices_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points_xy must be a finite (n, 2) array")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 2
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("vertices_xy must be a finite (n>=3, 2) closed-polygon vertex array")
    if not len(points):
        return np.empty(0, dtype=float)
    starts = vertices
    vectors = np.roll(vertices, -1, axis=0) - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    out = np.full(len(points), np.inf, dtype=float)
    for start, vector, norm_sq in zip(starts, vectors, lengths_sq, strict=True):
        if norm_sq <= 0:
            projection = np.broadcast_to(start, points.shape)
        else:
            t = np.clip(((points - start) @ vector) / norm_sq, 0.0, 1.0)
            projection = start + t[:, None] * vector
        out = np.minimum(out, np.linalg.norm(points - projection, axis=1))
    return out


def refined_parametric_curve_distances(
    points_xy: np.ndarray,
    curve_xy: Callable[[np.ndarray], np.ndarray],
    *,
    sample_count: int = 65_536,
    coarse_count: int = 8_192,
) -> dict[str, Any]:
    """Refine nearest-point distances on a 2π-periodic curve.

    A KDTree over a dense angular grid supplies a global candidate; bounded
    scalar minimization over the adjacent angular interval removes grid bias.
    Coarser and dense-grid distances are returned for an explicit resolution
    check. The curve callable maps a vector of angles to an (n, 2) array.
    """
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points_xy must be a finite (n, 2) array")
    if sample_count < 32 or coarse_count < 8 or sample_count <= coarse_count:
        raise ValueError(
            "sample_count must exceed coarse_count and both grids must be sufficiently dense"
        )
    if not len(points):
        empty = np.empty(0, dtype=float)
        return {
            "distances_px": empty,
            "coarse_distances_px": empty,
            "dense_grid_distances_px": empty,
        }

    dense_theta = np.arange(sample_count, dtype=float) * (2 * np.pi / sample_count)
    dense_curve = np.asarray(curve_xy(dense_theta), dtype=float)
    if dense_curve.shape != (sample_count, 2) or not np.all(np.isfinite(dense_curve)):
        raise ValueError("curve_xy must return a finite (len(theta), 2) array")
    dense_tree = cKDTree(dense_curve)
    dense_dist, dense_idx = dense_tree.query(points, k=1)

    coarse_theta = np.arange(coarse_count, dtype=float) * (2 * np.pi / coarse_count)
    coarse_curve = np.asarray(curve_xy(coarse_theta), dtype=float)
    if coarse_curve.shape != (coarse_count, 2) or not np.all(np.isfinite(coarse_curve)):
        raise ValueError("curve_xy must return a finite (len(theta), 2) array")
    coarse_dist = cKDTree(coarse_curve).query(points, k=1)[0]

    step = 2 * np.pi / sample_count
    refined = np.empty(len(points), dtype=float)
    for i, (point, index) in enumerate(zip(points, dense_idx, strict=True)):
        center = float(index) * step

        def squared_distance(theta: float, point: np.ndarray = point) -> float:
            location = np.asarray(curve_xy(np.array([theta % (2 * np.pi)])), dtype=float)[0]
            delta = location - point
            return float(delta @ delta)

        result = minimize_scalar(
            squared_distance,
            bounds=(center - step, center + step),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 100},
        )
        refined[i] = np.sqrt(max(0.0, float(result.fun)))
    return {
        "distances_px": refined,
        "coarse_distances_px": np.asarray(coarse_dist, dtype=float),
        "dense_grid_distances_px": np.asarray(dense_dist, dtype=float),
    }
