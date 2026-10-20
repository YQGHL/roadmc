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

from .crack_graph import render_graph, validate_graph
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
    if set(plan) != required or plan["schema_version"] not in {1, 2}:
        raise ValueError(f"plan keys must be {sorted(required)} and schema_version=1 or 2")
    material = plan["material"]
    if material not in _LABELS or not isinstance(plan["road_id"], str) or not plan["road_id"]:
        raise ValueError("material must be asphalt/concrete and road_id must be non-empty")
    if not isinstance(plan["provenance"], dict) or not isinstance(plan["hypothesis"], dict):
        raise ValueError("provenance and hypothesis must be explicit JSON objects")
    if not isinstance(plan["events"], list):
        raise ValueError("events must be a list")
    if plan["schema_version"] == 2:
        if material != "asphalt":
            raise ValueError("schema_version=2 crack graphs currently support asphalt only")
        return _validate_v2_plan(plan, required)
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
                   line_profiler_config: Any | None = None,
                   observation_noise_enabled: bool = True) -> ReplayResult:
    """Replay explicit events over a frozen base tile; no production defaults are touched."""
    plan_hash, events = validate_plan(plan)
    if not isinstance(observation_noise_enabled, bool):
        raise TypeError("observation_noise_enabled must be boolean")
    if plan["schema_version"] == 1 and not observation_noise_enabled:
        raise ValueError("observation_noise_enabled=False is supported only for schema_version=2")
    if observation not in {"off", "line-profiler"}:
        raise ValueError("observation must be off or line-profiler")
    if observation == "line-profiler" and observation_seed is None:
        raise ValueError("line-profiler requires an explicit observation_seed")
    _validate_tile(tile)
    if plan["schema_version"] == 2:
        return _replay_surface_v2(tile, plan, plan_hash, events, observation,
                                  observation_seed, line_profiler_config,
                                  observation_noise_enabled)
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


def _validate_v2_plan(
    plan: dict[str, Any], _required: set[str]
) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    material = plan["material"]
    specs = ASPHALT_TARGET_SPECS
    for event in plan["events"]:
        if not isinstance(event, dict) or set(event) != {
            "instance_id",
            "type",
            "seed",
            "parameters",
            "label",
        }:
            raise ValueError("each event requires instance_id/type/seed/parameters/label only")
        ident, kind, seed, params, label = (
            event[k] for k in ("instance_id", "type", "seed", "parameters", "label")
        )
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f"invalid or duplicate instance_id: {ident!r}")
        seen.add(ident)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError(f"event {ident}: seed must be unsigned 64-bit integer")
        if not isinstance(params, dict):
            raise ValueError(f"event {ident}: parameters must be an object")
        if not isinstance(kind, str):
            raise ValueError(f"event {ident}: type must be a string")
        if kind == "crack":
            if set(params) not in (
                {"variant", "graph", "profile", "texture"},
                {"variant", "graph", "profile", "texture", "severity"},
            ):
                raise ValueError(
                    f"event {ident}: v2 crack requires variant/graph/profile/texture and optional severity"
                )
            variant = params["variant"]
            if not isinstance(variant, str) or variant not in {
                "longitudinal",
                "transverse",
                "alligator",
                "block",
            }:
                raise ValueError(f"event {ident}: unsupported asphalt crack variant {variant!r}")
            validate_graph(params["graph"])
            if not isinstance(params["profile"], str) or params["profile"] not in {
                "legacy_v1_gaussian",
                "compact",
            }:
                raise ValueError(f"event {ident}: invalid profile")
            texture = params["texture"]
            _validate_texture_spec(texture, material, ident)
            expected_variant = {
                i
                for i, spec in specs.items()
                if spec.disease_type == "crack" and spec.variant == variant
            }
            if label is not None:
                if (
                    isinstance(label, bool)
                    or not isinstance(label, int)
                    or label not in expected_variant
                ):
                    raise ValueError(
                        f"event {ident}: label {label!r} conflicts with asphalt/{variant}"
                    )
                severity = params.get("severity")
                if severity != specs[label].severity:
                    raise ValueError(
                        f"event {ident}: known label requires matching explicit severity"
                    )
            elif "severity" in params and (
                not isinstance(params["severity"], str)
                or params["severity"] not in {"light", "severe"}
            ):
                raise ValueError(f"event {ident}: severity must be light or severe")
        else:
            if kind not in _LABELS[material]:
                raise ValueError(f"event {ident}: unsupported event type {kind!r} for {material}")
            if label is not None and (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in _LABELS[material][kind]
            ):
                raise ValueError(f"event {ident}: label conflicts with event type/material")
            expected = _PARAMS[kind]
            allowed = {frozenset(expected)}
            if kind == "pothole":
                allowed.add(frozenset({"center_x_m", "center_y_m", "contour", "depth_m", "exposed_rms_m"}))
            texture_base = expected - {"exposed_rms_m"}
            if kind in {"pothole", "raveling", "patch"}:
                allowed.add(frozenset(texture_base | {"texture"}))
            if kind == "pothole":
                allowed.add(frozenset({"center_x_m", "center_y_m", "contour", "depth_m", "texture"}))
                allowed |= {
                    (fields - {"texture", "exposed_rms_m"}) | {"material_partition"}
                    for fields in list(allowed)
                }
                allowed |= {fields | {"profile"} for fields in list(allowed)}
            if kind == "crack":
                allowed.add(frozenset(expected | {"severity"}))
            if frozenset(params) not in allowed:
                raise ValueError(f"event {ident}: invalid v2 non-crack parameters")
            for key, value in params.items():
                if key in {"severity", "texture", "contour", "profile", "material_partition"}:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    raise ValueError(f"event {ident}: parameter {key} must be finite")
            if kind == "pothole":
                if "profile" in params:
                    from .pothole_profile import validate_bowl_profile

                    validate_bowl_profile(params["profile"])
                if "radius_m" in params and "contour" in params:
                    raise ValueError(f"event {ident}: radius_m and contour are mutually exclusive")
                if "contour" in params:
                    _validate_ellipse_contour(params["contour"], ident)
                elif "radius_m" not in params:
                    raise ValueError(f"event {ident}: pothole requires radius_m or contour")
                if params.get("radius_m", 1) <= 0:
                    raise ValueError(f"event {ident}: radius_m must be >0")
            if kind in {"pothole", "raveling", "patch"}:
                if "material_partition" in params:
                    from .material_partition import validate_partition

                    if kind != "pothole" or "texture" in params or "exposed_rms_m" in params:
                        raise ValueError(f"event {ident}: partition is exclusive pothole texture policy")
                    validate_partition(params["material_partition"], material, ident, _validate_texture_spec)
                elif "texture" in params and "exposed_rms_m" in params:
                    raise ValueError(f"event {ident}: texture and exposed_rms_m are mutually exclusive")
                elif "texture" in params:
                    _validate_texture_spec(params["texture"], material, ident)
                elif "exposed_rms_m" not in params:
                    raise ValueError(f"event {ident}: explicit exposed_rms_m or texture required")
                elif params["exposed_rms_m"] < 0:
                    raise ValueError(f"event {ident}: exposed_rms_m must be >= 0")
            if kind == "pothole" and label is not None and material == "asphalt":
                depth = params["depth_m"]
                if (label == 9 and depth > 0.025) or (label == 10 and depth <= 0.025):
                    raise ValueError(
                        f"event {ident}: asphalt pothole depth conflicts with the 25 mm label rule"
                    )
        events.append({**event, "label": -1 if label is None else label})
    return hashlib.sha256(_canonical(plan)).hexdigest(), events


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
        and value >= 0
    )


def _validate_ellipse_contour(contour: Any, ident: str) -> None:
    required = {
        "shape", "semi_major_m", "semi_minor_m", "orientation_deg"
    }
    if not isinstance(contour, dict) or not required <= set(contour) or set(contour) - required - {"radial_harmonics"}:
        raise ValueError(f"event {ident}: contour requires shape/semi_major_m/semi_minor_m/orientation_deg")
    if contour["shape"] != "ellipse":
        raise ValueError(f"event {ident}: contour shape must be ellipse")
    values = [contour[k] for k in ("semi_major_m", "semi_minor_m", "orientation_deg")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
        raise ValueError(f"event {ident}: ellipse values must be finite numbers")
    a, b, theta = map(float, values)
    if not (0 < b <= a) or not (0 <= theta < 180):
        raise ValueError(f"event {ident}: require 0 < semi_minor_m <= semi_major_m and 0 <= orientation_deg < 180")
    if "radial_harmonics" in contour:
        from .pothole_contour import validate_radial_harmonics

        validate_radial_harmonics(contour["radial_harmonics"])


def _validate_texture_spec(texture: Any, material: str, ident: str) -> None:
    if (
        not isinstance(texture, dict)
        or not isinstance(texture.get("mode"), str)
        or texture["mode"] not in {"retain", "iid", "procedural"}
    ):
        raise ValueError(f"event {ident}: invalid texture mode")
    mode = texture["mode"]
    if mode == "retain" and set(texture) != {"mode"}:
        raise ValueError(f"event {ident}: retain texture accepts only mode")
    if mode == "iid" and (
        set(texture) != {"mode", "rms_m"} or not _finite_nonnegative(texture.get("rms_m"))
    ):
        raise ValueError(f"event {ident}: iid texture requires finite rms_m >= 0")
    if mode == "procedural":
        if set(texture) != {"mode", "settings"} or not isinstance(texture["settings"], dict):
            raise ValueError(f"event {ident}: procedural texture requires explicit settings")
        from .procedural_texture import _canonical_settings

        settings = _canonical_settings(texture["settings"])
        if settings["material_id"] != material:
            raise ValueError(f"event {ident}: procedural texture material_id must match plan material")


def _texture_moments(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean_m": None, "rms_m": None}
    mean = float(np.mean(values))
    return {
        "count": int(values.size),
        "mean_m": mean,
        "rms_m": float(np.sqrt(np.mean((values - mean) ** 2))),
    }


def _generate_event_texture(
    spec: dict[str, Any], ident: str, seed: int, tile: SurfaceTile, shape: tuple[int, ...],
    *, stream: tuple[Any, ...] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Generate the shared IID/procedural field with the established event seed stream."""
    mode = spec["mode"]
    parts = ("material", ident, seed) if stream is None else stream
    if mode == "retain":
        return None, {"mode": mode} if stream is None else {"mode": mode, "stream": list(parts), "random_draws": 0}
    if mode == "iid":
        rms = float(spec["rms_m"])
        field = rms * _rng(*parts).standard_normal(shape)
        return field, {"mode": mode, "rms_m": rms, "stream": list(parts)}
    material_rng = _rng(*parts)
    texture_seed = int(material_rng.integers(0, 2**63, dtype=np.int64))
    from .procedural_texture import generate_texture

    field, provenance = generate_texture(tile.x, tile.y, spec["settings"], texture_seed)
    if stream is not None:
        provenance = {**provenance, "mode": mode, "stream": list(parts), "texture_seed": texture_seed}
    return field, provenance


def _apply_event_texture(
    current: np.ndarray,
    write_mask: np.ndarray,
    spec: dict[str, Any],
    ident: str,
    seed: int,
    tile: SurfaceTile,
    metadata: dict[str, Any],
) -> None:
    """Apply event material texture without coupling it to geometry ownership."""
    mode = spec["mode"]
    generated, provenance = _generate_event_texture(spec, ident, seed, tile, current.shape)
    if mode == "iid":
        current[write_mask] = generated[write_mask]
    elif mode == "procedural":
        current[write_mask] = generated[write_mask]
        provenance = {**provenance, "mode": mode}
    elif mode == "retain":
        provenance["description"] = "retain current texture at event write mask"
    full = current if generated is None else generated
    provenance["whole_tile_statistics"] = _texture_moments(full)
    provenance["winning_mask_statistics"] = _texture_moments(full[write_mask])
    provenance["winning_mask_scope"] = "pixels written by this event; local RMS may differ from settings"
    metadata[ident] = provenance


def _replay_surface_v2(
    tile: SurfaceTile,
    plan: dict[str, Any],
    plan_hash: str,
    events: list[dict[str, Any]],
    observation: str,
    observation_seed: int | None,
    line_profiler_config: Any | None,
    observation_noise_enabled: bool,
) -> ReplayResult:
    """Execute v2 graphs while leaving the v1 code path and RNG sequence intact."""
    from dataclasses import asdict

    x, y = np.meshgrid(tile.x, tile.y, indexing="ij")
    shape = tile.design.shape
    base_reference = tile.design + tile.roughness
    reference = base_reference.copy()
    texture_current = tile.texture.copy()
    loss_union = np.zeros(shape, dtype=float)
    labels = np.full(shape, -1, dtype=np.int16)
    footprints = np.zeros((len(events),) + shape, dtype=bool)
    instance_owner = np.full(shape, -1, dtype=np.int32)
    edge_owner = np.full(shape, -1, dtype=np.int32)
    edge_owner_ids: list[dict[str, str]] = []
    graph_events: list[dict[str, Any]] = []
    event_texture_metadata: dict[str, Any] = {}
    pothole_profiles: dict[str, Any] = {}
    pothole_contours: dict[str, Any] = {}
    partition_enabled = any("material_partition" in e["parameters"] for e in events)
    if partition_enabled:
        partition_owner = np.full(shape, -1, dtype=np.int32)
        partition_floor = np.zeros(shape)
        partition_wall = np.zeros(shape)
        partition_metadata: dict[str, Any] = {}
    for ei, event in enumerate(events):
        p, ident, kind = event["parameters"], event["instance_id"], event["type"]
        if kind == "crack":
            graph = p["graph"]
            event_loss, mask, local_edge_owner = render_graph(graph, x, y, p["profile"])
            footprints[ei] = mask
            candidate = event_loss
            wins = (candidate > 0) & (candidate >= loss_union)
            loss_union[wins] = candidate[wins]
            labels[wins] = event["label"]
            instance_owner[wins] = ei
            if partition_enabled:
                partition_owner[wins] = -1
                partition_floor[wins] = partition_wall[wins] = 0.0
            # Edge ties use stable graph order. Across events, later equal-loss events win.
            chosen_edge = local_edge_owner[wins]
            edges = graph["edges"]
            edge_ids = [edge["id"] for edge in edges]
            global_edge_ids = []
            for edge_id in edge_ids:
                global_edge_ids.append(len(edge_owner_ids))
                edge_owner_ids.append(
                    {"event_index": str(ei), "instance_id": ident, "edge_id": edge_id}
                )
            edge_owner[wins] = np.asarray(global_edge_ids, dtype=np.int32)[chosen_edge]
            mode = p["texture"]["mode"]
            if mode == "iid":
                tex, _ = _generate_event_texture(p["texture"], ident, event["seed"], tile, shape)
                texture_current[wins] = tex[wins]
                rms = float(p["texture"]["rms_m"])
                event_texture_metadata[ident] = {
                    "mode": mode,
                    "rms_m": rms,
                    "stream": ["material", ident, event["seed"]],
                }
            elif mode == "procedural":
                tex, texture_meta = _generate_event_texture(
                    p["texture"], ident, event["seed"], tile, shape
                )
                texture_current[wins] = tex[wins]
                values = tex[wins]
                event_texture_metadata[ident] = {
                    **texture_meta,
                    "local_mask_statistics": {
                        "count": int(values.size),
                        "mean_m": float(values.mean()) if values.size else None,
                        "rms_m": float(np.sqrt(np.mean((values - values.mean()) ** 2)))
                        if values.size
                        else None,
                        "scope": "winning pixels of event mask; differs from full-tile statistics",
                    },
                }
            elif mode == "retain":
                event_texture_metadata[ident] = {
                    "mode": mode,
                    "description": "retain current texture at winning pixels; prior events may have changed it",
                }
            # retain keeps the current texture, which may include an earlier patch/event.
            graph_events.append(
                {
                    "instance_id": ident,
                    "variant": p["variant"],
                    "label": event["label"],
                    "profile": p["profile"],
                    "graph": graph,
                }
            )
        else:
            # V2 opts into ellipse contours and shared texture modes; existing geometric fields retain v1 meanings.
            cx, cy = _need(p, ("center_x_m", "center_y_m"), ident)
            if kind == "pothole":
                depth = _need(p, ("depth_m",), ident)[0]
                if depth < 0:
                    raise ValueError(f"event {ident}: depth must be >=0")
                if "contour" in p:
                    contour = p["contour"]
                    a, b, theta = (float(contour[k]) for k in ("semi_major_m", "semi_minor_m", "orientation_deg"))
                    theta = np.deg2rad(theta)
                    dx, dy = x - cx, y - cy
                    u = dx * np.cos(theta) + dy * np.sin(theta)
                    v = -dx * np.sin(theta) + dy * np.cos(theta)
                    normalized_u, normalized_v = u / a, v / b
                    if "radial_harmonics" in contour:
                        from .pothole_contour import warp_normalized_coordinates

                        normalized_u, normalized_v, contour_audit = warp_normalized_coordinates(
                            normalized_u, normalized_v, contour["radial_harmonics"]
                        )
                        pothole_contours[ident] = {
                            **contour_audit, "continuous_area_m2": float(np.pi * a * b),
                            "semi_major_m": a, "semi_minor_m": b,
                        }
                    rr = np.sqrt(normalized_u**2 + normalized_v**2)
                else:
                    radius = _need(p, ("radius_m",), ident)[0]
                    if radius <= 0:
                        raise ValueError(f"event {ident}: radius must be >0")
                    rr = np.hypot(x - cx, y - cy) / radius
                mask, profile = rr <= 1, np.maximum(0, 1 - rr**2)
                if "profile" in p:
                    from .pothole_profile import bowl_profile

                    if "contour" not in p:
                        normalized_u, normalized_v = (x - cx) / radius, (y - cy) / radius
                    profile, profile_audit = bowl_profile(normalized_u, normalized_v, p["profile"])
                    pothole_profiles[ident] = {
                        **profile_audit, "requested_continuous_depth_m": depth,
                        "sampled_depth_m": depth * float(profile.max()),
                    }
            elif kind == "raveling":
                rx, ry, corr, depth, threshold = _need(
                    p, ("radius_x_m", "radius_y_m", "corrlen_m", "depth_m", "threshold"), ident
                )
                rms = float(p.get("exposed_rms_m", 0.0))
                if min(rx, ry, corr) <= 0 or depth < 0 or not -4 < threshold < 4 or rms < 0:
                    raise ValueError(f"event {ident}: invalid raveling parameters")
                dx, dy = float(tile.x[1] - tile.x[0]), float(tile.y[1] - tile.y[0])
                envelope = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
                field = ndimage.gaussian_filter(
                    _rng("disease_shape", ident, event["seed"]).standard_normal(shape),
                    sigma=(max(corr / dx, 0.5), max(corr / dy, 0.5)),
                    mode="reflect",
                )
                field /= max(float(field.std()), 1e-12)
                mask = envelope & (field >= threshold)
                profile = np.clip((field - threshold) / max(4 - threshold, 1e-12), 0, 1)
            elif kind == "patch":
                rx, ry, thickness = _need(p, ("radius_x_m", "radius_y_m", "thickness_m"), ident)
                rms = float(p.get("exposed_rms_m", 0.0))
                if min(rx, ry) <= 0 or thickness < 0 or rms < 0:
                    raise ValueError(f"event {ident}: invalid patch parameters")
                r2 = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                mask, profile = r2 <= 1, np.clip(1 - r2, 0, 1)
                reference[mask] = base_reference[mask] + thickness * profile[mask]
                loss_union[mask] = 0
                texture_spec = p.get("texture", {"mode": "iid", "rms_m": rms})
                _apply_event_texture(texture_current, mask, texture_spec, ident, event["seed"],
                                     tile, event_texture_metadata)
                labels[mask] = event["label"]
                instance_owner[mask] = ei
                if partition_enabled:
                    partition_owner[mask] = -1
                    partition_floor[mask] = partition_wall[mask] = 0.0
                edge_owner[mask] = -1
                footprints[ei] = mask
                continue
            else:
                raise ValueError(f"event {ident}: unsupported type {kind}")
            footprints[ei] = mask
            depth = float(p["depth_m"])
            candidate = depth * profile * mask
            wins = mask & (candidate > 0) & (candidate >= loss_union)
            loss_union[wins] = candidate[wins]
            labels[wins] = event["label"]
            instance_owner[wins] = ei
            edge_owner[wins] = -1
            if partition_enabled:
                partition_owner[wins] = -1
                partition_floor[wins] = partition_wall[wins] = 0.0
            if "material_partition" in p:
                from .material_partition import apply_partition

                def generate_region(
                    spec: dict[str, Any], region: str, ident: str = ident,
                    seed: int = event["seed"],
                ) -> tuple[np.ndarray | None, dict[str, Any]]:
                    return _generate_event_texture(
                        spec, ident, seed, tile, shape,
                        stream=("material_partition", ident, seed, region),
                    )

                floor, wall, partition_meta = apply_partition(
                    texture_current, rr, wins, p["material_partition"], generate_region
                )
                partition_owner[wins] = ei
                partition_floor[wins] = floor[wins]
                partition_wall[wins] = wall[wins]
                partition_metadata[ident] = {"event_index": ei, **partition_meta}
                event_texture_metadata[ident] = {"mode": "wall_floor", **partition_meta}
            else:
                texture_spec = p["texture"] if "texture" in p else {
                    "mode": "iid", "rms_m": float(p["exposed_rms_m"])
                }
                _apply_event_texture(texture_current, wins, texture_spec, ident, event["seed"],
                                     tile, event_texture_metadata)
    geometry = reference - loss_union + texture_current
    normals = _normals(geometry, float(tile.x[1] - tile.x[0]), float(tile.y[1] - tile.y[0]))
    points = np.column_stack((x.ravel(), y.ravel(), geometry.ravel()))
    flat_labels = labels.ravel()
    observation_config = None
    profile_index = np.zeros(0, dtype=np.int64)
    flyer_mask = np.zeros(0, dtype=bool)
    dropout_count = 0
    noise_enabled = observation == "line-profiler" and bool(observation_noise_enabled)
    if observation == "line-profiler":
        from .config import LineProfilerConfig
        from .line_profiler import simulate_line_profiler

        cfg = line_profiler_config or LineProfilerConfig()
        if not isinstance(cfg, LineProfilerConfig):
            raise TypeError("line_profiler_config must be a LineProfilerConfig")
        dx, dy = float(tile.x[1] - tile.x[0]), float(tile.y[1] - tile.y[0])
        if not np.isclose(dx, dy, rtol=1e-9, atol=1e-12):
            raise ValueError("line-profiler currently requires an isotropic grid (dx == dy)")
        result = simulate_line_profiler(
            points,
            normals.reshape(-1, 3),
            flat_labels,
            grid_shape=shape,
            grid_res=dx,
            origin_xy=(float(tile.x[0]), float(tile.y[0])),
            cfg=cfg,
            rng=_rng("observation", int(observation_seed)),
            noise_enabled=noise_enabled,
        )
        points, flat_labels, normals_out = result.points, result.labels, result.normals
        profile_index, flyer_mask, dropout_count = (
            result.profile_index,
            result.flyer_mask,
            result.dropout_count,
        )
        observation_config = {
            "values": asdict(cfg),
            "source": "roadmc LineProfilerConfig; imported/engineering, not calibrated",
        }
    else:
        normals_out = normals.reshape(-1, 3)
    metadata = {
        "units": "m",
        "coordinates_normalized": False,
        "plan_hash": plan_hash,
        "base_metadata": copy.deepcopy(tile.metadata),
        "seeds": {e["instance_id"]: e["seed"] for e in events},
        "observation": observation,
        "observation_seed": observation_seed,
        "observation_config": observation_config,
        "material_id": plan["material"],
        "road_segment_id": plan["road_id"],
        "coordinate_units": "m",
        "domain": "synthetic",
        "calibration_status": "not_calibrated",
        "hypothesis": plan["hypothesis"],
        "schema_version": 2,
        "label_semantics": "project_taxonomy_unverified",
        "geometry_decomposition": "geometry_m = reference_lowfreq_m - loss_union_m + material_texture_m; loss is max-union and patches reset prior loss",
        "sensor_audit": {
            "called": observation == "line-profiler",
            "noise_enabled": noise_enabled,
            "sample_count": int(len(points)) if observation == "line-profiler" else 0,
            "profile_count": int(len(np.unique(profile_index))) if profile_index.size else 0,
            "flyer_count": int(np.count_nonzero(flyer_mask)),
            "dropout_count": int(dropout_count),
            "sampling_audit": result.sampling_audit if observation == "line-profiler" else None,
        },
    }
    if pothole_profiles:
        metadata["pothole_profiles"] = pothole_profiles
    if pothole_contours:
        metadata["pothole_contours"] = pothole_contours
    from .material_diagnostics import loss_diagnostics

    metadata["material_loss_diagnostics"] = loss_diagnostics(
        tile.x, tile.y, loss_union, instance_owner, footprints,
        [event["instance_id"] for event in events],
    )
    if partition_enabled:
        from .material_partition import final_partition_statistics

        pixel_area = float((tile.x[1] - tile.x[0]) * (tile.y[1] - tile.y[0]))
        for partition_meta in partition_metadata.values():
            partition_meta["final_statistics"] = final_partition_statistics(
                texture_current, loss_union, partition_owner, partition_meta["event_index"],
                partition_floor, partition_wall, pixel_area,
            )
        metadata["material_partition"] = partition_metadata
    result = ReplayResult(
        points,
        flat_labels,
        normals_out,
        geometry,
        np.any(footprints, axis=0).astype(np.float32),
        texture_current.copy(),
        footprints,
        metadata,
        None,
    )
    result._v2_audit = {
        "graph_json": {"events": graph_events},
        "base_height_m": tile.height.copy(),
        "reference_lowfreq_m": reference.copy(),
        "loss_union_m": loss_union.copy(),
        "instance_owner_index": instance_owner,
        "edge_owner_index": edge_owner,
        "edge_owner_ids": edge_owner_ids,
        "profile_index": profile_index,
        "flyer_mask": flyer_mask,
        "dropout_count": dropout_count,
        "observation_noise_enabled": noise_enabled,
        "event_texture_metadata": event_texture_metadata,
    }
    if partition_enabled:
        result._v2_audit.update({
            "material_partition_owner_index": partition_owner,
            "material_partition_floor_weight": partition_floor,
            "material_partition_wall_weight": partition_wall,
            "material_partition_metadata": partition_metadata,
        })
    return result


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
    if plan.get("schema_version") == 2:
        audit = result._v2_audit
        metadata.update(
            {
                "schema_version": 2,
                "observation_noise_enabled": audit["observation_noise_enabled"],
                "graph_json": audit["graph_json"],
                "event_texture_metadata": audit["event_texture_metadata"],
            }
        )
        fields["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False, default=str))
        fields["provenance"] = np.asarray(json.dumps(metadata, ensure_ascii=False, default=str))
        fields.update(
            {
                "graph_json": np.asarray(json.dumps(audit["graph_json"], ensure_ascii=False)),
                "base_height_m": audit["base_height_m"],
                "reference_lowfreq_m": audit["reference_lowfreq_m"],
                "loss_union_m": audit["loss_union_m"],
                "instance_owner_index": audit["instance_owner_index"],
                "edge_owner_index": audit["edge_owner_index"],
                "edge_owner_ids": np.asarray(
                    json.dumps(audit["edge_owner_ids"], ensure_ascii=False)
                ),
                "event_texture_metadata": np.asarray(
                    json.dumps(audit["event_texture_metadata"], ensure_ascii=False)
                ),
                "profile_index": audit["profile_index"],
                "flyer_mask": audit["flyer_mask"],
                "dropout_count": np.asarray(audit["dropout_count"]),
                "observation_noise_enabled": np.asarray(audit["observation_noise_enabled"]),
            }
        )
        if "material_partition_metadata" in audit:
            for name in ("material_partition_owner_index", "material_partition_floor_weight",
                         "material_partition_wall_weight"):
                fields[name] = audit[name]
            fields["material_partition_metadata"] = np.asarray(
                json.dumps(audit["material_partition_metadata"], ensure_ascii=False)
            )
    if result.intensities is not None:
        fields["intensities"] = result.intensities
    np.savez_compressed(path, **fields)
