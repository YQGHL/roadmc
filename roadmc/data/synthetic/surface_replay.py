"""Frozen, plan-driven procedural surface replay (hypotheses, not calibration)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .labels import ASPHALT_TARGET_SPECS, CONCRETE_TARGET_SPECS
from .surface_tile import SurfaceTile

_LABELS = {
    "asphalt": {
        "crack": {i for i, s in ASPHALT_TARGET_SPECS.items()
                  if s.disease_type == "crack" and s.variant in {"longitudinal", "transverse"}},
        "pothole": {i for i, s in ASPHALT_TARGET_SPECS.items() if s.disease_type == "pothole"},
        "raveling": {i for i, s in ASPHALT_TARGET_SPECS.items() if s.disease_type == "raveling"},
        "patch": {i for i, s in ASPHALT_TARGET_SPECS.items() if s.disease_type == "patching"},
    },
    "concrete": {
        "crack": {i for i, s in CONCRETE_TARGET_SPECS.items() if s.variant == "slab_crack"},
        "pothole": {i for i, s in CONCRETE_TARGET_SPECS.items() if s.variant == "pitting"},
        "patch": {i for i, s in CONCRETE_TARGET_SPECS.items() if s.variant == "patching"},
    },
}
_PARAMS = {
    "crack": {"center_x_m", "center_y_m", "start_x_m", "start_y_m", "end_x_m",
              "end_y_m", "width_m", "depth_m", "exposed_rms_m"},
    "pothole": {"center_x_m", "center_y_m", "radius_m", "depth_m", "exposed_rms_m"},
    "raveling": {"center_x_m", "center_y_m", "radius_x_m", "radius_y_m",
                 "corrlen_m", "depth_m", "threshold", "exposed_rms_m"},
    "patch": {"center_x_m", "center_y_m", "radius_x_m", "radius_y_m",
              "thickness_m", "exposed_rms_m"},
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def validate_plan(plan: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    required = {"schema_version", "material", "road_id", "events", "provenance", "hypothesis"}
    if set(plan) != required or plan["schema_version"] != 1:
        raise ValueError(f"plan keys must be {sorted(required)} and schema_version=1")
    material = plan["material"]
    if material not in _LABELS or not isinstance(plan["road_id"], str) or not plan["road_id"]:
        raise ValueError("material must be asphalt/concrete and road_id must be non-empty")
    if not isinstance(plan["provenance"], dict) or not isinstance(plan["hypothesis"], dict):
        raise ValueError("provenance and hypothesis must be explicit JSON objects")
    if not isinstance(plan["events"], list):
        raise ValueError("events must be a list")
    events: list[dict[str, Any]] = []
    ids: set[str] = set()
    for event in plan["events"]:
        if not isinstance(event, dict) or set(event) != {
            "instance_id", "type", "seed", "parameters", "label"
        }:
            raise ValueError("each event requires instance_id/type/seed/parameters/label only")
        ident, kind, seed, params, label = (event[k] for k in
                                            ("instance_id", "type", "seed", "parameters", "label"))
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"invalid or duplicate instance_id: {ident!r}")
        ids.add(ident)
        if kind not in _LABELS[material]:
            raise ValueError(f"event type {kind!r} is not defined for {material}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError(f"event {ident}: seed must be unsigned 64-bit integer")
        if label is not None and (isinstance(label, bool) or not isinstance(label, int)
                                  or label not in _LABELS[material][kind]):
            raise ValueError(f"event {ident}: label {label!r} conflicts with {material}/{kind}")
        if not isinstance(params, dict):
            raise ValueError(f"event {ident}: parameters must be an object")
        expected_params = _PARAMS[kind]
        allowed_param_sets = {frozenset(expected_params)}
        if kind == "crack":
            allowed_param_sets.add(frozenset(expected_params | {"severity"}))
        if frozenset(params) not in allowed_param_sets:
            raise ValueError(f"event {ident}: parameters must be exactly {sorted(_PARAMS[kind])}")
        for key, value in params.items():
            if key == "severity":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"event {ident}: parameter {key} must be a finite number")
        if kind == "crack" and label is not None:
            spec = (ASPHALT_TARGET_SPECS if material == "asphalt" else CONCRETE_TARGET_SPECS)[label]
            severity = params.get("severity")
            if severity != spec.severity:
                raise ValueError(f"event {ident}: label {label} requires explicit severity {spec.severity!r}")
            if material == "asphalt":
                width = params["width_m"]
                if (severity == "light" and width >= 0.003) or (severity == "severe" and width < 0.003):
                    raise ValueError(f"event {ident}: asphalt line-crack width conflicts with the 3 mm label rule")
        if kind == "pothole" and material == "asphalt" and label is not None:
            depth = params["depth_m"]
            if (label == 9 and depth > 0.025) or (label == 10 and depth <= 0.025):
                raise ValueError(f"event {ident}: asphalt pothole depth conflicts with the 25 mm label rule")
        if label is None:
            event = {**event, "label": -1}
        events.append(event)
    return hashlib.sha256(_canonical(plan)).hexdigest(), events


def _rng(*parts: Any) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(_canonical(parts)).digest()[:16], "little")
    return np.random.default_rng(seed)


def _need(params: dict[str, Any], names: tuple[str, ...], ident: str) -> list[float]:
    try:
        values = [float(params[n]) for n in names]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"event {ident}: explicit numeric parameters required: {names}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"event {ident}: parameters must be finite")
    return values


@dataclass
class ReplayResult:
    points: np.ndarray
    labels: np.ndarray
    normals: np.ndarray
    geometry: np.ndarray
    material_weight: np.ndarray
    material_texture: np.ndarray
    footprint: np.ndarray
    metadata: dict[str, Any]
    intensities: np.ndarray | None = None


def replay_surface(tile: SurfaceTile, plan: dict[str, Any], *, observation: str = "off",
                   observation_seed: int | None = None,
                   line_profiler_config: Any | None = None) -> ReplayResult:
    """Replay explicit events over a frozen base tile; no production defaults are touched."""
    plan_hash, events = validate_plan(plan)
    if observation not in {"off", "line-profiler"}:
        raise ValueError("observation must be off or line-profiler")
    if observation == "line-profiler" and observation_seed is None:
        raise ValueError("line-profiler requires an explicit observation_seed")
    _validate_tile(tile)
    x, y = np.meshgrid(tile.x, tile.y, indexing="ij")
    shape = tile.design.shape
    labels = np.full(shape, -1, dtype=np.int16)
    footprints = np.zeros((len(events),) + shape, dtype=bool)
    reference_surface = tile.design + tile.roughness
    base_reference = reference_surface.copy()
    texture_current = tile.texture.copy()
    loss_union = np.zeros(shape)
    for index, event in enumerate(events):
        p, kind, ident = event["parameters"], event["type"], event["instance_id"]
        cx, cy = _need(p, ("center_x_m", "center_y_m"), ident)
        if kind == "crack":
            x0, y0, x1, y1, width, depth = _need(
                p, ("start_x_m", "start_y_m", "end_x_m", "end_y_m", "width_m", "depth_m"), ident)
            if width <= 0 or depth < 0 or np.hypot(x1-x0, y1-y0) <= 0:
                raise ValueError(f"event {ident}: width/segment/depth invalid")
            t = np.clip(((x-x0)*(x1-x0)+(y-y0)*(y1-y0))/((x1-x0)**2+(y1-y0)**2), 0, 1)
            dist = np.hypot(x-(x0+t*(x1-x0)), y-(y0+t*(y1-y0)))
            mask = dist <= width / 2
            profile = np.exp(-((2*dist/width)**2))
        elif kind == "pothole":
            radius, depth = _need(p, ("radius_m", "depth_m"), ident)
            if radius <= 0 or depth < 0:
                raise ValueError(f"event {ident}: radius must be >0 and depth >=0")
            r = np.hypot(x-cx, y-cy) / radius
            mask = r <= 1
            profile = np.maximum(0, 1-r**2)
        elif kind == "raveling":
            rx, ry, corr, depth, threshold, rms = _need(
                p, ("radius_x_m", "radius_y_m", "corrlen_m", "depth_m", "threshold", "exposed_rms_m"), ident)
            if min(rx, ry, corr) <= 0 or depth < 0 or not -4 < threshold < 4 or rms < 0:
                raise ValueError(f"event {ident}: invalid raveling parameters")
            envelope = ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1
            dx, dy = abs(float(tile.x[1]-tile.x[0])), abs(float(tile.y[1]-tile.y[0]))
            field = ndimage.gaussian_filter(_rng("disease_shape", ident, event["seed"]).standard_normal(shape),
                                            sigma=(max(corr/dx, .5), max(corr/dy, .5)), mode="reflect")
            field /= max(float(field.std()), 1e-12)
            mask = envelope & (field >= threshold)
            profile = np.clip((field-threshold) / max(4-threshold, 1e-12), 0, 1)
            candidate_loss = depth * profile * mask
            noise = _rng("material", ident, event["seed"]).standard_normal(shape)
        else:  # patch is an ordered resurfacing event
            rx, ry, thickness, rms = _need(p, ("radius_x_m", "radius_y_m", "thickness_m", "exposed_rms_m"), ident)
            if min(rx, ry) <= 0 or thickness < 0 or rms < 0:
                raise ValueError(f"event {ident}: invalid patch parameters")
            r2 = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
            mask = r2 <= 1
            profile = np.clip(1-r2, 0, 1)
            base_reference[mask] = reference_surface[mask] + thickness * profile[mask]
            loss_union[mask] = 0.0
            texture_current[mask] = rms * _rng("material", ident, event["seed"]).standard_normal(shape)[mask]
            labels[mask] = event["label"]
        footprints[index] = mask
        if kind != "patch":
            if kind != "raveling":
                depth = float(p["depth_m"])
                candidate_loss = depth * profile * mask
                noise = _rng("material", ident, event["seed"]).standard_normal(shape)
            wins = mask & (candidate_loss > 0.0) & (candidate_loss >= loss_union)
            loss_union[wins] = candidate_loss[wins]
            labels[wins] = event["label"]
            texture_current[wins] = float(p["exposed_rms_m"]) * noise[wins]
    geometry = base_reference - loss_union + texture_current
    normals = _normals(geometry, float(tile.x[1]-tile.x[0]), float(tile.y[1]-tile.y[0]))
    points = np.column_stack((x.ravel(), y.ravel(), geometry.ravel()))
    flat_labels = labels.ravel()
    intensities = None
    if observation == "line-profiler":
        from dataclasses import asdict

        from .config import LineProfilerConfig
        from .line_profiler import simulate_line_profiler
        cfg = line_profiler_config or LineProfilerConfig()
        if not isinstance(cfg, LineProfilerConfig):
            raise TypeError("line_profiler_config must be a LineProfilerConfig")
        dx, dy = float(tile.x[1]-tile.x[0]), float(tile.y[1]-tile.y[0])
        if not np.isclose(dx, dy, rtol=1e-9, atol=1e-12):
            raise ValueError("line-profiler currently requires an isotropic grid (dx == dy)")
        result = simulate_line_profiler(points, normals.reshape(-1, 3), flat_labels,
            grid_shape=shape, grid_res=dx,
            origin_xy=(float(tile.x[0]), float(tile.y[0])), cfg=cfg,
            rng=_rng("observation", int(observation_seed)))
        points, flat_labels = result.points, result.labels
        normals_out = result.normals
        observation_config = {"values": asdict(cfg), "source": "roadmc LineProfilerConfig; imported/engineering, not calibrated"}
    else:
        normals_out = normals.reshape(-1, 3)
        observation_config = None
    material_weight = np.any(footprints, axis=0).astype(np.float32)
    metadata = {"units": "m", "coordinates_normalized": False, "plan_hash": plan_hash,
                         "base_metadata": copy.deepcopy(tile.metadata), "seeds": {e["instance_id"]: e["seed"] for e in events},
                         "observation": observation, "observation_seed": observation_seed,
                         "observation_config": observation_config,
                         "material_id": plan["material"], "road_segment_id": plan["road_id"],
                         "coordinate_units": "m", "domain": "synthetic",
                         "calibration_status": "not_calibrated", "hypothesis": plan["hypothesis"]}
    return ReplayResult(points, flat_labels, normals_out, geometry, material_weight,
                        texture_current.copy(), footprints, metadata, intensities)


def _normals(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    gx, gy = np.gradient(z, dx, dy, edge_order=1)
    n = np.stack((-gx, -gy, np.ones_like(z)), axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def _validate_tile(tile: SurfaceTile) -> None:
    for name, axis in (("x", tile.x), ("y", tile.y)):
        if axis.ndim != 1 or len(axis) < 2:
            raise ValueError(f"tile {name} axis must be one-dimensional with at least two coordinates")
        delta = np.diff(axis)
        if not np.all(delta > 0):
            raise ValueError(f"tile {name} axis must be strictly increasing")
        if not np.allclose(delta, delta[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f"tile {name} axis must be a regular grid")


def save_replay(path: str | Path, tile: SurfaceTile, plan: dict[str, Any], result: ReplayResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"material_id": plan["material"], "road_segment_id": plan["road_id"],
        "coordinate_units": "m", "coordinates_normalized": False, "domain": "synthetic",
        **copy.deepcopy(result.metadata)}
    fields: dict[str, Any] = {"points": result.points, "labels": result.labels,
        "normals": result.normals, "coordinates_normalized": np.asarray(False),
        "plan_hash": np.asarray(result.metadata["plan_hash"]),
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
        "provenance": np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
        "material_id": np.asarray(plan["material"]), "road_segment_id": np.asarray(plan["road_id"]),
        "coordinate_units": np.asarray("m"), "domain": np.asarray("synthetic"),
        "plan_json": np.asarray(json.dumps(plan, ensure_ascii=False, sort_keys=True)),
        "geometry_m": result.geometry, "material_weight": result.material_weight,
        "material_texture_m": result.material_texture,
        "event_footprints": result.footprint, "x_m": tile.x, "y_m": tile.y,
        "design_m": tile.design, "roughness_m": tile.roughness, "texture_m": tile.texture}
    if result.intensities is not None:
        fields["intensities"] = result.intensities
    np.savez_compressed(path, **fields)
