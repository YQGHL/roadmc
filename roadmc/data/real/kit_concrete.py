"""KIT mesh admission and grids in native units, with no implicit metric conversion."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import ndimage, optimize


class ArchiveRange(io.RawIOBase):
    """A bounded read-only file view used to inspect the ZIP inside an uncompressed TAR."""

    def __init__(self, path: str | Path, start: int, length: int):
        if start < 0 or length < 0 or start + length > Path(path).stat().st_size:
            raise ValueError("archive range is outside the source file")
        self._stack = ExitStack()
        # ExitStack owns the handle for this RawIOBase's lifetime, through close().
        self.stream = self._stack.enter_context(Path(path).open("rb"))  # noqa: SIM115
        self.start, self.length, self.position = start, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.length + offset
        )
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(max(0, self.length - self.position), size if size >= 0 else self.length)
        self.stream.seek(self.start + self.position)
        value = self.stream.read(size)
        self.position += len(value)
        return value

    def close(self):
        self._stack.close()
        super().close()


def parse_kit_mesh(data: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the published XYZ float32/triangle PLY layout, rejecting unsupported layouts."""
    stream = io.BytesIO(data)
    lines = []
    while stream.tell() < min(len(data), 65536):
        line = stream.readline()
        if not line:
            raise ValueError("PLY header is truncated")
        lines.append(line.decode("ascii").strip())
        if lines[-1] == "end_header":
            break
    else:
        raise ValueError("PLY header has no bounded end_header")
    if len(lines) < 9 or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("KIT parser requires binary little-endian PLY")
    # Do not silently skip extra vertex properties and then use an incorrect stride.
    body = [line for line in lines[2:] if not line.startswith(("comment ", "obj_info "))]
    if (
        len(body) != 7
        or body[1:4] != ["property float x", "property float y", "property float z"]
        or body[5:] != ["property list uchar int vertex_indices", "end_header"]
    ):
        raise ValueError("unsupported KIT PLY property layout")
    if not body[0].startswith("element vertex ") or not body[4].startswith("element face "):
        raise ValueError("missing vertex/face element declarations")
    n_vertices, n_faces = int(body[0].split()[-1]), int(body[4].split()[-1])
    if n_vertices <= 0 or n_faces < 0:
        raise ValueError("invalid mesh counts")
    offset = stream.tell()
    if offset + 12 * n_vertices + 13 * n_faces != len(data):
        raise ValueError("PLY payload length does not match XYZ/triangle records")
    points = np.frombuffer(data, dtype="<f4", count=3 * n_vertices, offset=offset).reshape(-1, 3)
    faces = np.frombuffer(
        data,
        dtype=[("count", "u1"), ("indices", "<i4", (3,))],
        count=n_faces,
        offset=offset + 12 * n_vertices,
    )
    if (
        not np.isfinite(points).all()
        or np.any(faces["count"] != 3)
        or np.any(faces["indices"] < 0)
        or np.any(faces["indices"] >= n_vertices)
    ):
        raise ValueError("mesh has nonfinite coordinates or invalid triangle records")
    return (
        points,
        faces["indices"],
        {
            "vertices": n_vertices,
            "faces": n_faces,
            "source_unit_status": "not_declared_in_PLY",
            "member_sha256": hashlib.sha256(data).hexdigest(),
        },
    )


def read_kit_mesh(tar_path: str | Path, member: str, expected_sha256: str | None = None):
    """Read one named mesh, verifying full-member ZIP CRC and an optional frozen hash."""
    name = PurePosixPath(member)
    if name.is_absolute() or ".." in name.parts or "\\" in member:
        raise ValueError("unsafe mesh member name")
    with tarfile.open(tar_path, "r:") as tar:
        candidates = [m for m in tar if m.isfile() and m.name.endswith(".zip")]
    if len(candidates) != 1:
        raise ValueError("KIT TAR must contain exactly one dataset ZIP")
    entry = candidates[0]
    with (
        ArchiveRange(tar_path, entry.offset_data, entry.size) as view,
        zipfile.ZipFile(view) as archive,
    ):
        if sum(m.filename == member for m in archive.infolist()) != 1:
            raise ValueError("mesh member missing or duplicated")
        data = archive.read(member)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("mesh differs from its frozen member hash")
    points, faces, receipt = parse_kit_mesh(data)
    receipt.update(source_path=str(Path(tar_path).resolve()), member=member, crc_verified=True)
    return points, faces, receipt


def native_height_grid(points, *, spacing=0.5, x_bounds=(-90.0, 90.0), y_bounds=(-130.0, 130.0)):
    """Mean residual height per native-coordinate cell, preserving missing support."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("mesh points must be finite Nx3")
    if (
        not np.isfinite(spacing)
        or spacing <= 0
        or x_bounds[1] <= x_bounds[0]
        or y_bounds[1] <= y_bounds[0]
    ):
        raise ValueError("invalid grid bounds or spacing")
    center = (p.min(axis=0) + p.max(axis=0)) / 2
    xy = p[:, :2] - center[:2]
    index = np.linspace(0, len(p) - 1, min(200000, len(p)), dtype=np.int64)
    design = np.column_stack((xy[index], np.ones(len(index))))
    z = p[index, 2]
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    for _ in range(3):
        residual = z - design @ coef
        sigma = max(1e-12, float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
        weight = np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))
        coef = np.linalg.lstsq(design * np.sqrt(weight[:, None]), z * np.sqrt(weight), rcond=None)[
            0
        ]
    residual = p[:, 2] - (xy @ coef[:2] + coef[2])
    nx = int(np.round((x_bounds[1] - x_bounds[0]) / spacing))
    ny = int(np.round((y_bounds[1] - y_bounds[0]) / spacing))
    if nx < 2 or ny < 2 or nx * ny > 10000000:
        raise ValueError("invalid or excessive grid size")
    ix = np.floor((xy[:, 0] - x_bounds[0]) / spacing).astype(np.int64)
    iy = np.floor((xy[:, 1] - y_bounds[0]) / spacing).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ids = ix[valid] * ny + iy[valid]
    count = np.bincount(ids, minlength=nx * ny).reshape(nx, ny)
    total = np.bincount(ids, weights=residual[valid], minlength=nx * ny).reshape(nx, ny)
    grid = np.full((nx, ny), np.nan)
    np.divide(total, count, out=grid, where=count > 0)
    return {
        "grid_height_source_units": grid,
        "grid_count": count,
        "valid_mask": count > 0,
        "x_centers_source_units": x_bounds[0] + (np.arange(nx) + 0.5) * spacing,
        "y_centers_source_units": y_bounds[0] + (np.arange(ny) + 0.5) * spacing,
        "raw_bbox_center_source_units": center,
        "plane_coefficients": coef,
        "spacing_source_units": np.asarray(spacing),
    }


def resample_native_grid(moving, parameters, *, spacing=0.5):
    """Lookup moving heights at R(angle) reference_xy + translation, excluding missing cells."""
    g = np.asarray(moving, float)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError("moving grid must be at least 3x3")
    angle, tx, ty = np.asarray(parameters, float)
    if not np.isfinite([angle, tx, ty, spacing]).all() or spacing <= 0:
        raise ValueError("invalid rigid registration parameters")
    xx, yy = np.indices(g.shape, dtype=float)
    x = (xx - (g.shape[0] - 1) / 2) * spacing
    y = (yy - (g.shape[1] - 1) / 2) * spacing
    theta = np.deg2rad(angle)
    mx = (np.cos(theta) * x - np.sin(theta) * y + tx) / spacing + (g.shape[0] - 1) / 2
    my = (np.sin(theta) * x + np.cos(theta) * y + ty) / spacing + (g.shape[1] - 1) / 2
    valid = np.isfinite(g)
    support = ndimage.map_coordinates(
        valid.astype(float), [mx, my], order=1, mode="constant", cval=0, prefilter=False
    )
    values = ndimage.map_coordinates(
        np.where(valid, g, 0), [mx, my], order=1, mode="constant", cval=0, prefilter=False
    )
    values[support < 1 - 1e-9] = np.nan
    return values


def common_support_repeat_scores(
    reference,
    before,
    after,
    *,
    spacing=0.5,
    fit_x=(-90, -10),
    evaluation_x=(10, 90),
):
    """Score fixed fields on shared support, without estimating an XY transform.

    Before/after are unoffset height arrays on the reference lookup grid.
    Both height offsets are estimated only on the common FIT support. The
    evaluation support is the intersection of all three finite fields.
    Strips separate reference-grid locations after preprocessing; they do not
    establish a two-domain or end-to-end raw-mesh holdout.
    """
    a, b, c = (np.asarray(value, dtype=float) for value in (reference, before, after))
    if a.ndim != 2 or a.shape != b.shape or a.shape != c.shape or min(a.shape) < 3:
        raise ValueError("common-support fields must share a nontrivial 2-D shape")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("common-support spacing must be finite and positive")
    if not np.isfinite([*fit_x, *evaluation_x]).all():
        raise ValueError("common-support strip bounds must be finite")
    x = (np.arange(a.shape[0]) - (a.shape[0] - 1) / 2) * spacing
    fit_rows = (x >= fit_x[0]) & (x <= fit_x[1])
    evaluation_rows = (x >= evaluation_x[0]) & (x <= evaluation_x[1])
    if not fit_rows.any() or not evaluation_rows.any() or np.any(fit_rows & evaluation_rows):
        raise ValueError("fit/evaluation strips must be nonempty and disjoint")
    fit = np.broadcast_to(fit_rows[:, None], a.shape)
    evaluation = np.broadcast_to(evaluation_rows[:, None], a.shape)
    common = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    common_fit = fit & common
    enough_fit = int(common_fit.sum()) >= 100
    before_offset = float(np.mean(a[common_fit] - b[common_fit])) if enough_fit else None
    after_offset = float(np.mean(a[common_fit] - c[common_fit])) if enough_fit else None

    def score(values, region, offset):
        good = common & region
        n = int(good.sum())
        result = {"points": n, "coverage": n / int(region.sum())}
        if n < 100 or offset is None:
            return dict(
                result,
                correlation=None,
                rmse_source_units=None,
                median_absolute_error_source_units=None,
                mae_source_units=None,
            )
        ref, pred = a[good], values[good] + offset
        return dict(
            result,
            correlation=float(np.corrcoef(ref, pred)[0, 1])
            if ref.std() > 0 and pred.std() > 0
            else None,
            rmse_source_units=float(np.sqrt(np.mean((pred - ref) ** 2))),
            median_absolute_error_source_units=float(np.median(np.abs(pred - ref))),
            mae_source_units=float(np.mean(np.abs(pred - ref))),
        )

    return {
        "scope": "fixed transform; reference-grid strips conditional on full-patch preprocessing",
        "mask_definition": "region & finite(reference) & finite(before) & finite(after)",
        "offset_estimation": "common FIT support only; no evaluation height used for offsets",
        "fit_status": "scored" if enough_fit else "insufficient_common_fit_support",
        "before_offset_from_common_fit_source_units": before_offset,
        "after_offset_from_common_fit_source_units": after_offset,
        "fit_before": score(b, fit, before_offset),
        "fit_after": score(c, fit, after_offset),
        "evaluation_before": score(b, evaluation, before_offset),
        "evaluation_after": score(c, evaluation, after_offset),
    }


def align_repeat_grids(
    reference,
    moving,
    *,
    spacing=0.5,
    fit_x=(-90, -10),
    evaluation_x=(10, 90),
    angle_bound=6.0,
    translation_bound=15.0,
):
    """Estimate rigid texture registration from a fit strip; score a separate strip."""
    a, b = np.asarray(reference, float), np.asarray(moving, float)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 12:
        raise ValueError("repeat grids must share a nontrivial shape")
    x = (np.arange(a.shape[0]) - (a.shape[0] - 1) / 2) * spacing
    fit_rows = (x >= fit_x[0]) & (x <= fit_x[1])
    eval_rows = (x >= evaluation_x[0]) & (x <= evaluation_x[1])
    if not fit_rows.any() or not eval_rows.any() or np.any(fit_rows & eval_rows):
        raise ValueError("fit/evaluation strips must be nonempty and disjoint")
    fit_mask = np.broadcast_to(fit_rows[:, None], a.shape)
    evaluation_mask = np.broadcast_to(eval_rows[:, None], a.shape)

    def score(values, region, offset=0.0):
        good = region & np.isfinite(a) & np.isfinite(values)
        n = int(good.sum())
        if n < 100:
            return {
                "points": n,
                "coverage": n / int(region.sum()),
                "correlation": None,
                "rmse_source_units": None,
            }
        ref, pred = a[good], values[good] + offset
        corr = float(np.corrcoef(ref, pred)[0, 1]) if ref.std() > 0 and pred.std() > 0 else None
        return {
            "points": n,
            "coverage": n / int(region.sum()),
            "correlation": corr,
            "rmse_source_units": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "median_absolute_error_source_units": float(np.median(np.abs(pred - ref))),
        }

    def fit_objective(parameters):
        s = score(resample_native_grid(b, parameters, spacing=spacing), fit_mask)
        return (
            1 - s["correlation"] if s["correlation"] is not None and s["coverage"] >= 0.5 else 1e6
        )

    # Coarse translation seeds from the fit strip only. No evaluation height enters optimization.
    starts = []
    ar = a[fit_rows]
    window = np.hanning(ar.shape[0])[:, None] * np.hanning(ar.shape[1])[None, :]
    af = np.fft.fft2(np.where(np.isfinite(ar), ar - np.nanmean(ar), 0) * window)
    limits = np.ceil(translation_bound / spacing).astype(int)
    ix = np.fft.fftfreq(ar.shape[0]) * ar.shape[0]
    iy = np.fft.fftfreq(ar.shape[1]) * ar.shape[1]
    permitted = (np.abs(ix[:, None]) <= limits) & (np.abs(iy[None, :]) <= limits)
    for angle in np.arange(-angle_bound, angle_bound + 0.25, 0.5):
        rotated = resample_native_grid(b, [angle, 0, 0], spacing=spacing)[fit_rows]
        bf = np.fft.fft2(np.where(np.isfinite(rotated), rotated - np.nanmean(rotated), 0) * window)
        cross = af * bf.conj()
        cross /= np.maximum(np.abs(cross), 1e-30) ** 0.5
        corr = np.fft.ifft2(cross).real
        where = np.unravel_index(np.argmax(np.where(permitted, corr, -np.inf)), corr.shape)
        # A negative correlation lag means lookup the moving grid at positive translation.
        shift = np.array([-ix[where[0]], -iy[where[1]]]) * spacing
        theta = np.deg2rad(angle)
        shift = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ shift
        start = np.array([angle, *np.clip(shift, -translation_bound, translation_bound)])
        starts.append((fit_objective(start), start))
    _, initial = min(starts, key=lambda s: s[0])
    fitted = optimize.minimize(
        fit_objective,
        initial,
        method="Powell",
        bounds=[
            (-angle_bound, angle_bound),
            (-translation_bound, translation_bound),
            (-translation_bound, translation_bound),
        ],
        options={"maxiter": 60, "xtol": 0.02, "ftol": 1e-7},
    )
    best = min(
        [(fit_objective(initial), initial), (float(fitted.fun), fitted.x)], key=lambda s: s[0]
    )[1]
    before = resample_native_grid(b, [0, 0, 0], spacing=spacing)
    aligned = resample_native_grid(b, best, spacing=spacing)

    def offset_from_fit(values):
        valid = fit_mask & np.isfinite(a) & np.isfinite(values)
        return float(np.mean(a[valid] - values[valid])) if valid.any() else 0.0

    before_offset, after_offset = offset_from_fit(before), offset_from_fit(aligned)
    after_fit = score(aligned, fit_mask, after_offset)
    return {
        "parameters_angle_deg_translation_source_units": best.tolist(),
        "lookup_transform": "moving(R(angle) @ reference_xy + translation)",
        "height_offset_from_fit_source_units": after_offset,
        "optimizer_success": bool(fitted.success),
        "optimizer_message": str(fitted.message),
        "fit_before": score(before, fit_mask, before_offset),
        "fit_after": after_fit,
        "evaluation_before": score(before, evaluation_mask, before_offset),
        "evaluation_after": score(aligned, evaluation_mask, after_offset),
        "common_support": common_support_repeat_scores(
            a, before, aligned, spacing=spacing, fit_x=fit_x, evaluation_x=evaluation_x
        ),
        "alignment_status": "candidate_alignment"
        if after_fit["correlation"] is not None and after_fit["correlation"] > 0.1
        else "alignment_not_supported",
        "evaluation_scope": "separate strip on the same repeated physical patch; not independent-road validation",
    }, aligned + after_offset
