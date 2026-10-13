"""Torch-free, unit-aware raw point-cloud loading for evaluation tools."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import RoadPointCloudMetadata, metadata_sidecar_path


@dataclass(frozen=True)
class RawRoadScene:
    """A metric point cloud localized in float64, before tensor conversion."""

    points_m: np.ndarray
    coordinate_origin_m: tuple[float, float, float]
    source_units: str
    intensity: np.ndarray | None = None
    intensity_raw: np.ndarray | None = None
    labels: np.ndarray | None = None
    normals: np.ndarray | None = None
    grid_z_m: np.ndarray | None = None
    grid_spacing_m: tuple[float, float] | None = None
    range_values: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


def load_raw_scene(
    path: str | Path,
    *,
    point_key: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> RawRoadScene:
    """Read supported arrays/clouds without importing torch.

    Coordinates are converted to metres in float64 and localized only after
    unit conversion. Missing semantic labels remain ``None``.
    """
    source = Path(path)
    ext = source.suffix.lower()
    arrays: dict[str, np.ndarray] = {}
    if ext == ".npz":
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        raw_metadata = _metadata_from_npz(arrays)
        points = _select_points(arrays, point_key)
        intensity = _first_array(arrays, ("intensity", "intensities", "reflectance"))
        labels = _first_array(arrays, ("labels", "surface_labels", "label"))
        normals = _first_array(arrays, ("normals", "surface_normals"))
        range_values = _first_array(arrays, ("range", "ranges", "range_m"))
        valid_mask = _first_array(arrays, ("valid", "is_valid", "return_valid"))
        z_grid = _first_array(arrays, ("grid_z_m", "height_grid_m", "z_grid", "height_grid"))
        z_grid_in_meters = "grid_z_m" in arrays or "height_grid_m" in arrays
        spacing = _grid_spacing(arrays, raw_metadata)
    elif ext == ".npy":
        data = np.load(source, allow_pickle=False)
        if data.dtype.names:
            names = set(data.dtype.names)
            if not {"x", "y", "z"}.issubset(names):
                raise ValueError(f"structured NPY must include x/y/z fields: {source}")
            points = np.column_stack([data[name] for name in ("x", "y", "z")])
            intensity = _structured_field(data, ("intensity", "reflectance"))
            labels = _structured_field(data, ("label", "labels", "class", "semantic"))
            normals = _structured_normals(data)
        else:
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[1] < 3:
                raise ValueError(f"NPY must be Nx3+ or structured x/y/z, got {data.shape}")
            points = data[:, :3]
            labels = data[:, 3] if data.shape[1] >= 4 else None
            normals = data[:, 4:7] if data.shape[1] >= 7 else None
            intensity = data[:, 7] if data.shape[1] >= 8 else None
        raw_metadata = _metadata_for_path(source)
        range_values, valid_mask = None, None
        z_grid_in_meters = False
        z_grid, spacing = None, None
    elif ext in {".ply", ".pcd", ".las", ".laz"}:
        points, labels, normals, intensity, range_values, valid_mask = _load_cloud(source)
        raw_metadata = _metadata_for_path(source)
        z_grid, spacing = None, None
        z_grid_in_meters = False
    else:
        raise ValueError(f"Unsupported point-cloud format {ext!r}; supported: .npz, .npy, .ply, .pcd, .las, .laz")

    if metadata_overrides:
        raw_metadata.update(metadata_overrides)
    metadata = _validated_metadata(raw_metadata, source)
    points64 = np.asarray(points, dtype=np.float64)
    if points64.ndim != 2 or points64.shape[1] != 3:
        raise ValueError(f"point coordinates must have shape (N, 3), got {points64.shape}")
    if ext == ".npz" and "coordinates_normalized" in arrays and bool(arrays["coordinates_normalized"].item()):
        if "coordinate_center" not in arrays or "coordinate_scale" not in arrays:
            raise ValueError(f"normalized coordinate NPZ requires coordinate_center and coordinate_scale to restore metric coordinates: {source}")
        inverse_scale = float(np.asarray(arrays["coordinate_scale"]).item())
        center = np.asarray(arrays["coordinate_center"], dtype=np.float64)
        if not np.isfinite(inverse_scale) or inverse_scale <= 0.0 or center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"invalid coordinate inverse normalization fields in {source}")
        points64 = points64 * inverse_scale + center
    scale = metadata.coordinate_scale_to_meters
    metric = points64 * scale
    finite = np.isfinite(metric).all(axis=1)
    if not finite.any():
        raise ValueError(f"no finite coordinates in {source}")
    explicit_origin = raw_metadata.get("coordinate_origin", raw_metadata.get("origin"))
    if explicit_origin is not None:
        origin_m = np.asarray(explicit_origin, dtype=np.float64) * scale
        if origin_m.shape != (3,) or not np.isfinite(origin_m).all():
            raise ValueError("coordinate_origin must contain three finite values in source units")
    else:
        origin_m = metric[np.flatnonzero(finite)[0]].copy()
    metric = metric - origin_m
    validity_policy = "finite_coordinates_only_no_range_or_validity_field"
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).astype(bool).reshape(-1)
        if len(valid_mask) != len(metric):
            raise ValueError("validity field length does not match point count")
        finite &= valid_mask
        validity_policy = "explicit_validity_field_false_is_excluded"
    if range_values is not None:
        range_values = np.asarray(range_values, dtype=np.float64).reshape(-1)
        if len(range_values) != len(metric):
            raise ValueError("range field length does not match point count")
        finite &= np.isfinite(range_values) & (range_values > 0.0)
        validity_policy = "explicit_validity_and_range_fields; nonfinite_or_zero_range_is_no_return" if valid_mask is not None else "explicit_range_field_nonfinite_or_zero_is_no_return"
    metric[~finite] = np.nan

    intensity_raw = np.asarray(intensity) if intensity is not None else None
    intensity_declared = "intensity_scale" in raw_metadata or source.suffix.lower() in {".las", ".laz"}
    if intensity is not None and intensity_declared:
        intensity = metadata.normalize_intensity(np.asarray(intensity))
    elif intensity is not None:
        intensity = None
    if labels is not None:
        labels = np.asarray(labels)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
    if z_grid is not None:
        z_grid = np.asarray(z_grid, dtype=np.float64) * (1.0 if z_grid_in_meters else scale)
        z_grid = z_grid - origin_m[2]
        if not bool(np.all(finite)):
            z_grid = None
    if z_grid is None:
        z_grid, inferred_spacing = _exact_grid(metric)
        if spacing is None:
            spacing = inferred_spacing

    enriched = dict(raw_metadata)
    enriched.update(metadata.as_dict())
    enriched["coordinate_origin_m"] = origin_m.tolist()
    enriched["source_path"] = str(source.resolve())
    enriched["point_validity_policy"] = validity_policy
    enriched["intensity_calibration_status"] = "declared" if intensity_declared else "unknown_no_normalized_intensity_metric"
    enriched["coordinate_units_status"] = "declared" if "coordinate_units" in raw_metadata else "unknown_default_m_for_diagnostic_only"
    return RawRoadScene(
        points_m=metric,
        coordinate_origin_m=tuple(float(v) for v in origin_m),
        source_units=metadata.coordinate_units,
        intensity=intensity,
        intensity_raw=intensity_raw,
        labels=labels,
        normals=normals,
        grid_z_m=z_grid,
        grid_spacing_m=spacing,
        range_values=range_values,
        valid_mask=finite,
        metadata=enriched,
        source_path=str(source.resolve()),
    )


def _metadata_for_path(path: Path) -> dict[str, Any]:
    sidecar = metadata_sidecar_path(path)
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata sidecar must be a JSON object: {sidecar}")
    return payload


def _metadata_from_npz(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    value = arrays.get("metadata")
    if value is not None and value.ndim == 0:
        try:
            decoded = json.loads(str(value.item()))
            if isinstance(decoded, dict):
                result.update(decoded)
        except (TypeError, json.JSONDecodeError):
            pass
    for key in ("coordinate_units", "intensity_scale", "road_segment_id", "material_id", "sensor", "sensor_metadata", "coordinate_origin"):
        if key in arrays and arrays[key].ndim == 0:
            item = arrays[key].item()
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            if key in {"sensor_metadata"} and isinstance(item, str):
                with suppress(json.JSONDecodeError):
                    item = json.loads(item)
            result[key] = item
    return result


def _validated_metadata(payload: dict[str, Any], path: Path) -> RoadPointCloudMetadata:
    fallback = RoadPointCloudMetadata(
        intensity_scale="uint16" if path.suffix.lower() in {".las", ".laz"} else "normalized_0_1"
    )
    known = {field.name for field in RoadPointCloudMetadata.__dataclass_fields__.values()}
    values = {key: value for key, value in payload.items() if key in known}
    return RoadPointCloudMetadata.from_dict({**fallback.as_dict(), **values})


def _select_points(arrays: dict[str, np.ndarray], point_key: str | None) -> np.ndarray:
    keys = (point_key,) if point_key else ("points", "surface_points", "base_points")
    for key in keys:
        if key and key in arrays:
            points = np.asarray(arrays[key])
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
    raise ValueError(f"NPZ needs Nx3 points; tried {keys}")


def _first_array(arrays: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    return next((arrays[key] for key in keys if key in arrays), None)


def _grid_spacing(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> tuple[float, float] | None:
    if "grid_spacing_m" in metadata:
        spacing = metadata["grid_spacing_m"]
        return (float(spacing), float(spacing)) if np.isscalar(spacing) else tuple(map(float, spacing))
    for key in ("spacing_m", "grid_res_m", "grid_spacing"):
        if key in arrays and arrays[key].ndim == 0:
            value = float(arrays[key])
            return value, value
    return None


def _exact_grid(points: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    valid = np.isfinite(points).all(axis=1)
    p = points[valid]
    if len(p) < 4:
        return None, None
    x, y = np.unique(p[:, 0]), np.unique(p[:, 1])
    if len(x) * len(y) != len(p) or len(x) < 2 or len(y) < 2:
        return None, None
    dx, dy = np.diff(x), np.diff(y)
    if not np.allclose(dx, dx[0], rtol=1e-7, atol=1e-12) or not np.allclose(dy, dy[0], rtol=1e-7, atol=1e-12):
        return None, None
    ix = np.searchsorted(x, p[:, 0])
    iy = np.searchsorted(y, p[:, 1])
    z = np.full((len(x), len(y)), np.nan, dtype=np.float64)
    z[ix, iy] = p[:, 2]
    if not np.isfinite(z).all():
        return None, None
    return z, (float(dx[0]), float(dy[0]))


def _structured_field(data: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    available = set(data.dtype.names or ())
    return next((np.asarray(data[name]) for name in names if name in available), None)


def _structured_normals(data: np.ndarray) -> np.ndarray | None:
    available = set(data.dtype.names or ())
    for names in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
        if set(names).issubset(available):
            return np.column_stack([data[name] for name in names])
    return None


def _load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ext = path.suffix.lower()
    if ext == ".ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("PLY raw loading requires optional dependency `plyfile`") from exc
        ply = PlyData.read(str(path))
        if "vertex" not in ply:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertex = ply["vertex"].data
        points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
        labels = _structured_field(vertex, ("label", "labels", "class", "semantic"))
        normals = _structured_normals(vertex)
        intensity = _structured_field(vertex, ("intensity", "reflectance", "scalar_intensity"))
        range_values = _structured_field(vertex, ("range", "range_m", "distance"))
        valid = _structured_field(vertex, ("valid", "is_valid", "return_valid"))
        return points, labels, normals, intensity, range_values, valid
    if ext in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError("LAS/LAZ raw loading requires optional dependency `laspy[lazrs]`") from exc
        las = laspy.read(str(path))
        dimensions = set(las.point_format.dimension_names)
        range_values = next((np.asarray(las[name]) for name in ("range", "range_m", "distance") if name in dimensions), None)
        valid = next((np.asarray(las[name]) for name in ("valid", "is_valid", "return_valid") if name in dimensions), None)
        return np.column_stack([las.x, las.y, las.z]), None, None, np.asarray(las.intensity), range_values, valid
    if ext == ".pcd":
        return _load_pcd(path)
    raise ValueError(f"Unsupported point-cloud format: {ext}")


def _load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    with path.open("rb") as handle:
        header: dict[str, list[str]] = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA declaration: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            header[fields[0].upper()] = fields[1:]
            if fields[0].upper() == "DATA":
                encoding = fields[1].lower()
                payload = handle.read()
                break
    names = header.get("FIELDS", [])
    if encoding == "ascii":
        data = np.loadtxt(payload.splitlines(), dtype=np.float64, ndmin=2)
        offsets: dict[str, int] = {}
        offset = 0
        for name, count in zip(names, map(int, header.get("COUNT", ["1"] * len(names))), strict=False):
            offsets[name] = offset
            offset += count
        if not {"x", "y", "z"}.issubset(offsets):
            raise ValueError(f"PCD needs x/y/z fields: {path}")
        points = data[:, [offsets[n] for n in ("x", "y", "z")]]
        labels = next((data[:, offsets[n]] for n in ("label", "labels", "class", "semantic") if n in offsets), None)
        intensity = next((data[:, offsets[n]] for n in ("intensity", "reflectance") if n in offsets), None)
        range_values = next((data[:, offsets[n]] for n in ("range", "range_m", "distance") if n in offsets), None)
        valid = next((data[:, offsets[n]] for n in ("valid", "is_valid", "return_valid") if n in offsets), None)
        normals = None
        for n in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
            if set(n).issubset(offsets):
                normals = data[:, [offsets[k] for k in n]]
                break
        return points, labels, normals, intensity, range_values, valid
    if encoding in {"binary", "binary_compressed"}:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("binary PCD raw loading requires optional dependency `open3d`") from exc
        cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=False)
        points = np.asarray(cloud.points, dtype=np.float64)
        normals = np.asarray(cloud.normals, dtype=np.float64) if cloud.has_normals() else None
        return points, None, normals, None, None, None
    raise ValueError(f"Unsupported PCD DATA encoding {encoding!r}")
