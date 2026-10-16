"""Frozen synthetic and native-unit experiments for geometry/material hypotheses.

All synthetic results are controlled mechanism diagnostics. They are not road
calibration, sensor calibration, or population estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from scipy.optimize import least_squares


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_output(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _section(protocol: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = protocol.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"protocol[{name!r}] must be an object")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unsupported protocol[{name!r}] keys: {sorted(extra)}")
    return dict(value)


def _alias(cfg: dict[str, Any], preferred: str, alternate: str, default: Any) -> Any:
    if preferred in cfg and alternate in cfg and cfg[preferred] != cfg[alternate]:
        raise ValueError(f"conflicting protocol aliases {preferred!r} and {alternate!r}")
    return cfg.get(preferred, cfg.get(alternate, default))


def _seed_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _protocol(protocol: dict[str, Any], lane: str) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise TypeError("protocol must be a dict")
    declared = protocol.get("status", "synthetic_only_engineering_hypothesis")
    return {"lane": lane, "status": declared, **protocol}


def _texture_stats(values: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(values, dtype=float).ravel()
    if not v.size:
        return {"count": 0, "mean_m": 0.0, "rms_m": 0.0, "skewness": 0.0, "excess_kurtosis": 0.0}
    centered = v - float(v.mean())
    rms = float(np.sqrt(np.mean(centered**2)))
    if rms <= 1e-15:
        skew = kurt = 0.0
    else:
        z = centered / rms
        skew, kurt = float(np.mean(z**3)), float(np.mean(z**4) - 3.0)
    return {
        "count": int(v.size),
        "mean_m": float(v.mean()),
        "rms_m": rms,
        "skewness": skew,
        "excess_kurtosis": kurt,
    }


def _local_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) < 3 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _build_tile(geometry_protocol: dict[str, Any]):
    from roadmc.data.synthetic.surface_tile import SurfaceTile

    dx = float(geometry_protocol.get("grid_spacing_m", 0.001))
    if not math.isclose(dx, 0.001, rel_tol=0, abs_tol=1e-12):
        raise ValueError("frozen geometry experiment requires 1 mm spacing")
    x = np.arange(-0.3, 0.3000001, dx)
    y = np.arange(-0.25, 0.2500001, dx)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    design = 0.0006 * xx - 0.0003 * yy
    rng = np.random.default_rng(int(geometry_protocol.get("seed", 20261015)))
    coarse = rng.standard_normal((31, 26))
    from scipy.ndimage import gaussian_filter, zoom

    roughness = zoom(coarse, (len(x) / 31, len(y) / 26), order=3)[: len(x), : len(y)]
    roughness = gaussian_filter(roughness, sigma=8.0, mode="reflect")
    roughness = 0.00015 * roughness / max(float(roughness.std()), 1e-12)
    # Known deterministic bandlimited texture; it is a synthetic control field.
    texture = 0.00018 * np.sin(2 * np.pi * (13.0 * xx + 7.0 * yy) + 0.3) + 0.00009 * np.cos(
        2 * np.pi * (21.0 * yy - 4.0 * xx) - 0.5
    )
    tile = SurfaceTile(
        x,
        y,
        design,
        roughness,
        texture,
        {
            "provenance": "synthetic frozen study tile; no measured road source",
            "grid_spacing_m": dx,
            "layers": "design plane / seeded low-frequency roughness / known two-wave texture",
            "seed": int(geometry_protocol.get("seed", 20261015)),
        },
    )
    return tile


def _graph_event(seed: int, texture: dict[str, Any]) -> dict[str, Any]:
    graph = {
        "nodes": [
            {"id": "n0", "xy_m": [-0.23, 0.12], "boundary": True},
            {"id": "n1", "xy_m": [-0.16, 0.12], "boundary": False},
            {"id": "n2", "xy_m": [-0.11, 0.17], "boundary": False},
            {"id": "n3", "xy_m": [-0.05, 0.17], "boundary": True},
        ],
        "edges": [
            {
                "id": "e0",
                "start": "n0",
                "end": "n1",
                "points_m": [[-0.23, 0.12], [-0.16, 0.12]],
                "half_width_left_m": [0.0012, 0.0012],
                "half_width_right_m": [0.0012, 0.0012],
                "visible_depth_m": [0.004, 0.005],
            },
            {
                "id": "e1",
                "start": "n1",
                "end": "n2",
                "points_m": [[-0.16, 0.12], [-0.135, 0.145], [-0.11, 0.17]],
                "half_width_left_m": [0.0012, 0.0014, 0.0013],
                "half_width_right_m": [0.0012, 0.0010, 0.0013],
                "visible_depth_m": [0.005, 0.004, 0.006],
            },
            {
                "id": "e2",
                "start": "n2",
                "end": "n3",
                "points_m": [[-0.11, 0.17], [-0.05, 0.17]],
                "half_width_left_m": [0.0013, 0.0013],
                "half_width_right_m": [0.0013, 0.0013],
                "visible_depth_m": [0.006, 0.004],
            },
        ],
        "closed_cells_xy_m": [],
    }
    return {
        "instance_id": "crack-network",
        "type": "crack",
        "seed": seed,
        "label": None,
        "parameters": {
            "variant": "block",
            "graph": graph,
            "profile": "compact",
            "texture": texture,
        },
    }


def _events(
    contour: dict[str, Any],
    texture_mode: str,
    seed: int,
    material: dict[str, Any],
    bowl: dict[str, Any],
) -> list[dict[str, Any]]:
    if texture_mode == "procedural":
        tex: dict[str, Any] = {
            "mode": "procedural",
            "settings": {
                "material_id": "asphalt",
                "rms_m": float(material["rms_m"]),
                "hurst": float(material["hurst"]),
                "anisotropy_ratio": 1.0,
                "orientation_deg": 0.0,
                "band_lo_cycles_m": float(material["band_cycles_m"][0]),
                "band_hi_cycles_m": float(material["band_cycles_m"][1]),
                "distribution": "gaussian",
            },
        }
    elif texture_mode == "iid":
        tex = {"mode": "iid", "rms_m": float(material["rms_m"])}
    elif texture_mode == "retain":
        tex = {"mode": "retain"}
    else:
        raise ValueError(texture_mode)
    center = {"center_x_m": 0.0, "center_y_m": 0.0}
    pothole = {
        **center,
        "contour": contour,
        "depth_m": float(material["depth_m"]),
        "texture": tex,
        "profile": dict(bowl),
    }
    return [
        {
            "instance_id": "pothole-control",
            "type": "pothole",
            "seed": seed + 2,
            "label": None,
            "parameters": pothole,
        },
    ]


def _damage_events(texture_mode: str, seed: int) -> list[dict[str, Any]]:
    if texture_mode == "procedural":
        tex: dict[str, Any] = {
            "mode": "procedural",
            "settings": {
                "material_id": "asphalt",
                "rms_m": 0.0004,
                "hurst": 0.05,
                "anisotropy_ratio": 1.0,
                "orientation_deg": 0.0,
                "band_lo_cycles_m": 0.0,
                "band_hi_cycles_m": 400.0,
                "distribution": "gaussian",
            },
        }
    elif texture_mode == "iid":
        tex = {"mode": "iid", "rms_m": 0.0004}
    else:
        tex = {"mode": "retain"}
    crack = _graph_event(seed + 1, tex)
    crack["parameters"]["graph"] = {
        "nodes": [
            {"id": "a", "xy_m": [-0.25, 0.12], "boundary": True},
            {"id": "b", "xy_m": [-0.13, 0.12], "boundary": True},
        ],
        "edges": [
            {
                "id": "crack",
                "start": "a",
                "end": "b",
                "points_m": [[-0.25, 0.12], [-0.13, 0.12]],
                "half_width_left_m": [0.0015, 0.0015],
                "half_width_right_m": [0.0015, 0.0015],
                "visible_depth_m": [0.006, 0.006],
            }
        ],
        "closed_cells_xy_m": [],
    }
    return [
        crack,
        {
            "instance_id": "raveling-control",
            "type": "raveling",
            "seed": seed + 3,
            "label": None,
            "parameters": {
                "center_x_m": -0.16,
                "center_y_m": -0.12,
                "radius_x_m": 0.055,
                "radius_y_m": 0.038,
                "corrlen_m": 0.006,
                "depth_m": 0.004,
                "threshold": 0.1,
                "texture": tex,
            },
        },
        {
            "instance_id": "patch-control",
            "type": "patch",
            "seed": seed + 4,
            "label": None,
            "parameters": {
                "center_x_m": -0.19,
                "center_y_m": 0.12,
                "radius_x_m": 0.040,
                "radius_y_m": 0.014,
                "thickness_m": 0.003,
                "texture": tex,
            },
        },
    ]


def _replay_plan(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "material": "asphalt",
        "road_id": "frozen-study-tile",
        "events": events,
        "provenance": {"source": "controlled synthetic experiment"},
        "hypothesis": {"scope": "mechanism comparison only"},
    }


def _geometry_metrics(tile, result, event_ids: list[str]) -> dict[str, Any]:
    from roadmc.data.synthetic.material_diagnostics import loss_diagnostics

    md = result._v2_audit
    loss = md["loss_union_m"]
    diag = loss_diagnostics(
        tile.x, tile.y, loss, md["instance_owner_index"], result.footprint, event_ids
    )
    footprints = result.footprint
    overlap = int(np.count_nonzero(np.sum(footprints, axis=0) > 1))
    # Reconstruction identity is checked against the saved tile and replay layers.
    reconstructed = np.asarray(result.geometry)
    expected = np.asarray(md["reference_lowfreq_m"]) - loss + result.material_texture
    exact = float(np.max(np.abs(reconstructed - expected)))
    if "patch-control" in event_ids:
        patch_index = event_ids.index("patch-control")
        patch_mask = footprints[patch_index]
        patched_old_loss_max = float(np.max(loss[patch_mask])) if patch_mask.any() else None
    else:
        patched_old_loss_max = None
    return {
        "loss_diagnostics": diag,
        "footprint_overlap_pixels": overlap,
        "reference_minus_final_loss_plus_texture_max_abs_m": exact,
        "patch_old_loss_reset_max_m": patched_old_loss_max,
        "geometry_only_invariant": "loss and owner fields are replay geometry outputs, independent of texture mode",
    }


def _material_region_masks(tile, result, event_index: int | None = None):
    footprints = np.asarray(result.footprint, dtype=bool)
    if footprints.ndim != 3 or footprints.shape[1:] != tile.design.shape or not len(footprints):
        raise ValueError("material-region analysis requires event footprints aligned with the tile")
    if event_index is None:
        region = np.any(footprints, axis=0)
        other_damage = np.zeros(region.shape, dtype=bool)
        region_source = "union_of_all_event_footprints"
    else:
        if event_index < 0 or event_index >= len(footprints):
            raise IndexError("material region event_index is outside footprint array")
        region = footprints[event_index]
        other_damage = np.any(np.delete(footprints, event_index, axis=0), axis=0)
        region_source = f"event_footprint_index_{event_index}"
    if not np.any(region):
        raise ValueError("material-region event footprint is empty")
    dx, dy = abs(float(tile.x[1] - tile.x[0])), abs(float(tile.y[1] - tile.y[0]))
    inside_distance = distance_transform_edt(region, sampling=(dx, dy))
    outside_distance = distance_transform_edt(~region, sampling=(dx, dy))
    edge_band_m = 0.005
    edge = region & (inside_distance <= edge_band_m)
    interior = region & (inside_distance > edge_band_m)
    normal = (
        ~region & ~other_damage & (outside_distance > edge_band_m) & (outside_distance <= 0.015)
    )
    masks = {"interior": interior, "edge": edge, "normal": normal}
    definitions = {
        "region_source": region_source,
        "region_pixels": int(region.sum()),
        "other_damage_pixels_excluded_from_normal": int(other_damage.sum()),
        "edge_band_m": edge_band_m,
        "normal_band_m": [edge_band_m, 0.015],
        "distance_method": "scipy EDT with physical sampling=(abs(dx),abs(dy)); distances from raster sample centers",
        "interior": "actual event/union footprint samples farther than edge_band_m from footprint exterior",
        "edge": "actual event/union footprint samples at or within edge_band_m of footprint exterior",
        "normal": "outside region, 5..15 mm from region boundary, excluding all other event footprints",
        "grid_spacing_m": [dx, dy],
    }
    return masks, definitions


def _material_metrics(tile, result, event_index: int | None = 0) -> dict[str, Any]:
    # Descriptive spatial statistics; these masks are not used as PSD windows.
    masks, definitions = _material_region_masks(tile, result, event_index)
    texture = np.asarray(result.material_texture)
    summary = {}
    for name, mask in masks.items():
        values = texture[mask]
        pairs_x = mask[:-1, :] & mask[1:, :]
        pairs_y = mask[:, :-1] & mask[:, 1:]
        summary[name] = {
            **_texture_stats(values),
            "mask_pixel_count": int(mask.sum()),
            "corr_x_pair_count": int(pairs_x.sum()),
            "corr_y_pair_count": int(pairs_y.sum()),
            "corr_x_1pixel_descriptive": _local_corr(
                texture[:-1, :][pairs_x], texture[1:, :][pairs_x]
            ),
            "corr_y_1pixel_descriptive": _local_corr(
                texture[:, :-1][pairs_y], texture[:, 1:][pairs_y]
            ),
        }
    definitions["correlation_pair_scope"] = (
        "both adjacent sample endpoints must belong to the same region mask"
    )
    summary["regions"] = definitions
    summary["interpretation"] = (
        "masked RMS, moments and adjacent-pixel correlations are descriptive synthetic controls; no masked PSD or measured distribution claim"
    )
    return summary


def _save_geometry_artifacts(
    out: Path, name: str, tile, result, metrics: dict[str, Any], plan: dict[str, Any]
) -> None:
    from roadmc.data.synthetic.surface_replay import save_replay

    replay_dir = out / "selected_replays"
    replay_dir.mkdir(exist_ok=True)
    save_replay(replay_dir / f"{name}.npz", tile, plan, result)
    # Compact derived arrays support independent reruns without dumping every cloud.
    md = result._v2_audit
    np.savez_compressed(
        out / f"{name}_layers.npz",
        x=tile.x,
        y=tile.y,
        direct_surface=result.geometry,
        loss_m=md["loss_union_m"],
        owner=md["instance_owner_index"],
        texture=result.material_texture,
    )
    metrics[name] = _geometry_metrics(tile, result, [e["instance_id"] for e in plan["events"]])
    material_event_index = next(
        (i for i, event in enumerate(plan["events"]) if event["type"] == "pothole"),
        None,
    )
    metrics[name]["material"] = _material_metrics(tile, result, material_event_index)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    xx, yy = np.meshgrid(tile.x, tile.y, indexing="ij")
    for ax, array, title in (
        (axes[0], result.geometry - tile.design, "Surface - design"),
        (axes[1], md["loss_union_m"], "Latent loss"),
        (axes[2], result.material_texture, "Material texture"),
        (axes[3], md["instance_owner_index"], "Final owner"),
    ):
        im = ax.pcolormesh(
            xx,
            yy,
            array,
            shading="auto",
            cmap="coolwarm" if "texture" in title or "Surface" in title else "viridis",
        )
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out / f"{name}_diagnostics.png", dpi=140)
    plt.close(fig)
    # Section through pit centre and detrended residual around it.
    ix = int(np.argmin(abs(tile.x - 0.0)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].plot(tile.y, result.geometry[ix], label="surface")
    axes[0].plot(tile.y, tile.design[ix] + tile.roughness[ix], label="base design+roughness")
    axes[0].set(title="Pothole centre section", xlabel="y (m)", ylabel="z (m)")
    axes[0].legend()
    residual = (
        result.geometry
        - (tile.design + tile.roughness)
        + md["loss_union_m"]
        - result.material_texture
    )
    im = axes[1].pcolormesh(xx, yy, residual, shading="auto", cmap="coolwarm")
    axes[1].set(title="Geometry residual", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.savefig(out / f"{name}_sections.png", dpi=140)
    plt.close(fig)


def run_geometry_material_study(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Run frozen mechanism and texture controls over one synthetic tile."""
    from roadmc.data.synthetic.config import LineProfilerConfig
    from roadmc.data.synthetic.surface_replay import replay_surface

    out = Path(output_dir)
    _new_output(out)
    cfg = _section(
        protocol,
        "geometry",
        {
            "spacing_m",
            "grid_spacing_m",
            "x_bounds_m",
            "y_bounds_m",
            "geometry_seed",
            "seed",
            "observation_seed",
            "nominal_depth_m",
            "ellipse_axes_m",
            "ellipse_orientation_deg",
            "harmonics",
            "bowl",
            "texture_rms_m",
            "texture_hurst",
            "texture_band_cycles_m",
        },
    )
    spacing_m = float(_alias(cfg, "spacing_m", "grid_spacing_m", 0.001))
    geometry_seed = _seed_value(_alias(cfg, "geometry_seed", "seed", 73), "geometry_seed/seed")
    observation_seed = _seed_value(cfg.get("observation_seed", 11), "observation_seed")
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("geometry spacing_m/grid_spacing_m must be finite and > 0")
    material = {
        "rms_m": float(cfg.get("texture_rms_m", 0.0004)),
        "hurst": float(cfg.get("texture_hurst", 0.05)),
        "band_cycles_m": cfg.get("texture_band_cycles_m", [0.0, 400.0]),
        "depth_m": float(cfg.get("nominal_depth_m", 0.040)),
    }
    bowl = dict(
        cfg.get(
            "bowl",
            {"kind": "tilted_power_bowl", "edge_power": 0.7, "tilt_major": 1.6, "tilt_minor": -0.8},
        )
    )
    semi_major, semi_minor = map(float, cfg.get("ellipse_axes_m", [0.20, 0.13]))
    angle = float(cfg.get("ellipse_orientation_deg", 20.0))
    frozen = {
        "grid_spacing_m": spacing_m,
        "x_m": cfg.get("x_bounds_m", [-0.3, 0.3]),
        "y_m": cfg.get("y_bounds_m", [-0.25, 0.25]),
        "seed": geometry_seed,
        "observation_seed": observation_seed,
        "shape": [601, 501],
        "texture": "known deterministic two-wave bandlimited field",
        "synthetic_only": True,
    }
    frozen["effective_protocol"] = {
        "spacing_m": spacing_m,
        "grid_spacing_m": spacing_m,
        "geometry_seed": geometry_seed,
        "seed": geometry_seed,
        "observation_seed": observation_seed,
        "x_bounds_m": frozen["x_m"],
        "y_bounds_m": frozen["y_m"],
        "nominal_depth_m": material["depth_m"],
        "ellipse_axes_m": [semi_major, semi_minor],
        "ellipse_orientation_deg": angle,
        "harmonics": cfg.get("harmonics", {"cos": [0.08, -0.04, 0.03], "sin": [-0.03, 0.02, 0.01]}),
        "bowl": bowl,
        "texture_rms_m": material["rms_m"],
        "texture_hurst": material["hurst"],
        "texture_band_cycles_m": material["band_cycles_m"],
    }
    if (
        frozen["grid_spacing_m"] != 0.001
        or frozen["x_m"] != [-0.3, 0.3]
        or frozen["y_m"] != [-0.25, 0.25]
    ):
        raise ValueError(
            "full frozen geometry experiment requires 1 mm spacing and fixed x/y bounds"
        )
    tile = _build_tile({**cfg, **frozen})
    harmonics = cfg.get("harmonics", {"cos": [0.08, -0.04, 0.03], "sin": [-0.03, 0.02, 0.01]})
    try:
        from roadmc.data.synthetic.pothole_contour import validate_radial_harmonics
    except ImportError as exc:
        raise RuntimeError(
            "pothole_contour implementation is required before the approved geometry study can run"
        ) from exc
    harmonics = validate_radial_harmonics(harmonics)
    harmonic = {
        "shape": "ellipse",
        "semi_major_m": semi_major,
        "semi_minor_m": semi_minor,
        "orientation_deg": angle,
        "radial_harmonics": harmonics,
    }
    ellipse = {
        "shape": "ellipse",
        "semi_major_m": semi_major,
        "semi_minor_m": semi_minor,
        "orientation_deg": angle,
    }
    cases = [
        ("ellipse_geometry_retain", ellipse, "retain"),
        ("improved_geometry_retain", harmonic, "retain"),
        ("improved_geometry_iid", harmonic, "iid"),
        ("improved_geometry_procedural", harmonic, "procedural"),
    ]
    frozen_protocol = {
        "status": "synthetic_mechanism_experiment_only",
        "geometry": frozen,
        "cases": [{"name": n, "contour": c, "texture_mode": m} for n, c, m in cases],
        "sequential_ablation": "existing add_pothole displacement applied to the same frozen base with its original retained base texture; reported as an ablation reference",
        "damage_sequence": _damage_events("retain", geometry_seed),
        "factor_control": "same frozen tile/event geometry/seed within each comparison; only contour or material mode varies",
        "area_normalized_harmonic": {
            "ellipse_axes_m": [semi_major, semi_minor],
            "ellipse_orientation_deg": angle,
            "harmonics": harmonics,
            "nominal_area_m2": math.pi * semi_major * semi_minor,
            "harmonic_area_m2": math.pi * semi_major * semi_minor,
            "basis": "orthogonal zero-mean radial modes with area normalization from pothole_contour",
        },
        "bowl": bowl,
        "texture_settings": material,
        "texture_metrics_limit": "masked RMS/moments/local correlations only; masked texture PSD is not reported as measured distribution",
        "observation": {
            "line_profiler": "lateral_res_m=0.001; step_m=0.005",
            "noise": "LineProfilerConfig defaults are engineering settings, not sensor calibration",
        },
        "rutting": "existing add_rutting diagnostic only; no shoulder uplift forced",
        "protocol_input": protocol,
    }
    _json(out / "frozen_protocol.json", frozen_protocol)
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "data/synthetic/surface_replay.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/material_diagnostics.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/primitives.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/procedural_texture.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/line_profiler.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/pothole_contour.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/pothole_profile.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/crack_graph.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/surface_tile.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/config.py",
    ]
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    geometry_hashes = {}
    for source in source_paths:
        geometry_hashes[str(source)] = _sha256(source)
        (snapshots / source.name).write_bytes(source.read_bytes())
    _json(out / "source_hashes_before_run.json", geometry_hashes)
    # Original primitive ablation: direct displacement on the frozen base, so the
    # original base texture remains present under the depression.
    from roadmc.data.synthetic.primitives import add_pothole

    xx, yy = np.meshgrid(tile.x, tile.y, indexing="ij")
    original_points = np.column_stack((xx.ravel(), yy.ravel(), tile.height.ravel()))
    original_labels = np.full(len(original_points), -1, dtype=np.int16)
    original_points, original_labels = add_pothole(
        original_points,
        original_labels,
        center=(0.0, 0.0),
        radius=math.sqrt(semi_major * semi_minor),
        depth=float(cfg.get("nominal_depth_m", 0.040)),
        edge_quality=1.0,
        severity="severe",
        seed=geometry_seed,
        beta_range=(2.0, 5.0),
        params={"beta_mode": "overlap"},
    )
    original_surface = original_points[:, 2].reshape(tile.design.shape)
    original_dir = out / "sequential_ablation_original_displacement_retain"
    original_dir.mkdir()
    np.savez_compressed(
        original_dir / "surface.npz",
        x_m=tile.x,
        y_m=tile.y,
        points=original_points,
        labels=original_labels,
        design_m=tile.design,
        roughness_m=tile.roughness,
        base_texture_m=tile.texture,
        realized_surface_m=original_surface,
    )
    original_residual = original_surface - tile.height
    metrics_ablation = {
        "method": "existing primitives.add_pothole applied sequentially to base points",
        "texture_mode": "retain existing frozen base texture",
        "seed": geometry_seed,
        "sample_count": int(len(original_points)),
        "depth_min_m": float(original_residual.min()),
        "depth_max_m": float(original_residual.max()),
        "surface_reconstruction_max_abs_m": float(
            np.max(np.abs(original_points[:, 2] - original_surface.ravel()))
        ),
        "status": "synthetic-only historical primitive ablation",
    }
    _json(original_dir / "metrics.json", metrics_ablation)
    original_fig, original_ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = original_ax.pcolormesh(
        xx, yy, original_surface - tile.design, shading="auto", cmap="coolwarm"
    )
    original_ax.set(
        title="Original displacement / retained base texture",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    original_fig.colorbar(im, ax=original_ax, label="z - design (m)")
    original_fig.savefig(original_dir / "surface.png", dpi=140)
    plt.close(original_fig)
    metrics: dict[str, Any] = {}
    metrics["sequential_ablation_original_displacement_retain"] = metrics_ablation
    plans: dict[str, Any] = {}
    results = {}
    for name, contour, mode in cases:
        plan = _replay_plan(_events(contour, mode, geometry_seed, material, bowl))
        result = replay_surface(tile.frozen_copy(), plan, observation="off")
        plans[name] = plan
        results[name] = result
        _save_geometry_artifacts(out, name, tile, result, metrics, plan)
    # Explicit cross-case invariance assertion: texture treatment must not alter geometry.
    reference_name = "improved_geometry_retain"
    geometry_reference = results[reference_name]._v2_audit["loss_union_m"]
    invariant_max = {}
    for name in ("improved_geometry_iid", "improved_geometry_procedural"):
        invariant_max[name] = float(
            np.max(np.abs(results[name]._v2_audit["loss_union_m"] - geometry_reference))
        )
        if invariant_max[name] != 0:
            raise AssertionError(f"texture mode changed latent geometry loss in {name}")
    metrics["texture_geometry_invariance_max_abs_loss_m"] = invariant_max
    damage_plan = _replay_plan(_damage_events("retain", geometry_seed))
    damage_result = replay_surface(tile.frozen_copy(), damage_plan, observation="off")
    _save_geometry_artifacts(
        out, "secondary_damage_patch_reset", tile, damage_result, metrics, damage_plan
    )
    if metrics["secondary_damage_patch_reset"]["patch_old_loss_reset_max_m"] != 0.0:
        raise AssertionError("patch event failed to reset preceding loss under its footprint")

    # Direct surface, noiseless resampled line profile, and engineering-noise line profile.
    candidate = "improved_geometry_procedural"
    plan = plans[candidate]
    obs_dir = out / "observation_comparison"
    obs_dir.mkdir()
    observation = {}
    for tag, noise in (
        ("line_profiler_no_noise", False),
        ("line_profiler_engineering_noise", True),
    ):
        result = replay_surface(
            tile.frozen_copy(),
            plan,
            observation="line-profiler",
            observation_seed=observation_seed,
            line_profiler_config=LineProfilerConfig(lateral_res_m=0.001, step_m=0.005),
            observation_noise_enabled=noise,
        )
        scan_payload = {"points": result.points, "labels": result.labels, "normals": result.normals}
        if result.intensities is not None:
            scan_payload["intensities"] = result.intensities
        np.savez_compressed(obs_dir / f"{tag}.npz", **scan_payload)
        sensor_audit = result.metadata["sensor_audit"]
        observation[tag] = {
            "points": int(len(result.points)),
            "grid_points_minus_scan_points": int(tile.design.size - len(result.points)),
            "dropout_count": int(sensor_audit["dropout_count"]),
            "flyer_count": int(sensor_audit["flyer_count"]),
            "profile_count": int(sensor_audit["profile_count"]),
            "sampling_audit": sensor_audit["sampling_audit"],
            "metadata": result.metadata,
            "noise_is_engineering_hypothesis": noise,
        }
    direct = results[candidate]
    observation["direct_surface"] = {
        "points": int(len(direct.points)),
        "metadata": direct.metadata,
        "noise_is_engineering_hypothesis": False,
    }
    _json(obs_dir / "comparison.json", observation)

    # Keep rut cross-sections as a diagnostic only; no forced uplift and no model edit.
    from roadmc.data.synthetic.primitives import add_rutting

    xx, yy = np.meshgrid(tile.x, tile.y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), (tile.design + tile.roughness).ravel()))
    labels = np.full(len(points), -1, dtype=np.int16)
    rut_points, _ = add_rutting(
        points,
        labels,
        center_line=0.0,
        wheel_separation=0.18,
        depth=0.012,
        width=0.12,
        severity="light",
        seed=frozen["seed"],
        params={"modulation_wavelength": 1.0, "modulation_epsilon": 0.0},
    )
    rut = rut_points[:, 2].reshape(tile.design.shape)
    mid_y = int(np.argmin(abs(tile.y)))
    rut_diag = {
        "source": "existing add_rutting primitive",
        "cross_section_x_m": tile.x.tolist(),
        "relative_z_m": (rut[:, mid_y] - (tile.design + tile.roughness)[:, mid_y]).tolist(),
        "shoulder_uplift_forced": False,
        "interpretation": "diagnostic only; the existing generator model was not changed",
    }
    _json(out / "rut_cross_section_diagnostic.json", rut_diag)

    _json(out / "source_hashes.json", geometry_hashes)
    _json(out / "metrics.json", metrics)
    report = [
        "# RoadMC 合成几何与材质机制实验",
        "",
        "【合成实验】输出来自冻结的人工合成平面与显式事件，只能比较当前实现机制，不能证明真实道路统计真实性或传感器拟合。",
        "几何量与材质量分开汇报；遮罩内只报告 RMS、矩和局部相邻相关，不把掩膜纹理谱称为实测 PSD。",
        "车辙截面仅诊断现有 add_rutting 输出，没有强加路肩隆起。完整协议、数值指标、回放和源文件哈希见同目录 JSON/NPZ。",
        "",
    ]
    (out / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "output_dir": str(out),
        "metrics_path": str(out / "metrics.json"),
        "cases": list(metrics),
        "source_hashes": geometry_hashes,
        "status": "synthetic_mechanism_experiment_only",
    }


def _fit_outline(points: np.ndarray):
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    direction = eigenvectors[:, -1]
    axes = np.sqrt(np.maximum(2 * eigenvalues[::-1], 1e-12))
    initial = [*center, *np.log(axes), np.arctan2(direction[1], direction[0])]

    def local(q, values):
        shifted = values - q[:2]
        ct, st = np.cos(q[4]), np.sin(q[4])
        return np.column_stack(
            (
                (ct * shifted[:, 0] + st * shifted[:, 1]) / np.exp(q[2]),
                (-st * shifted[:, 0] + ct * shifted[:, 1]) / np.exp(q[3]),
            )
        )

    fit = least_squares(
        lambda q: np.linalg.norm(local(q, points), axis=1) - 1, initial, max_nfev=500
    )
    if not fit.success:
        raise ValueError(f"outline fit failed: {fit.message}")
    return (
        fit.x,
        local,
        float(np.sqrt(np.mean((np.linalg.norm(local(fit.x, points), axis=1) - 1) ** 2))),
    )


def _radial_scale(theta: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    try:
        from roadmc.data.synthetic.pothole_contour import radial_scale
    except ImportError as exc:
        raise RuntimeError(
            "pothole_contour radial harmonics implementation is required before this study can run"
        ) from exc
    spec = {
        "cos": np.asarray(coeff[:3], dtype=float).tolist(),
        "sin": np.asarray(coeff[3:], dtype=float).tolist(),
    }
    return radial_scale(theta, spec)


def _fit_radial_harmonics(xy: np.ndarray, train: np.ndarray, test: np.ndarray):
    # Sector holdout is fixed before fitting. Bounds imply sum hypot(a_n,b_n)<1.
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    radius = np.linalg.norm(xy, axis=1)
    initial = np.zeros(6)
    fit = least_squares(
        lambda c: radius[train] / _radial_scale(theta[train], c) - 1,
        initial,
        bounds=(-0.2 * np.ones(6), 0.2 * np.ones(6)),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=500,
    )
    if not fit.success:
        raise ValueError(f"harmonic boundary fit failed: {fit.message}")
    c = fit.x.copy()
    norm = float(sum(math.hypot(c[i], c[3 + i]) for i in range(3)))
    shrink = 1.0
    # Explicit engineering safeguard, visible in the fit record; no silent clipping.
    if norm >= 0.999:
        shrink = 0.999 / norm
        c *= shrink
    scale = _radial_scale(theta, c)
    if np.any(scale <= 0):
        raise ValueError("positive radial boundary invalid after documented coefficient shrink")
    residual = radius / scale - 1

    def scores(mask):
        return {
            "points": int(mask.sum()),
            "radial_relative_rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "radial_relative_mae": float(np.mean(np.abs(residual[mask]))),
        }

    return {
        "coefficients": {"cos": c[:3].tolist(), "sin": c[3:].tolist()},
        "coefficient_pair_norm_sum": float(sum(math.hypot(c[i], c[3 + i]) for i in range(3))),
        "positivity_bound": "sum_n hypot(a_n,b_n)<1",
        "bound_shrink_factor": shrink,
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "evaluations": int(fit.nfev),
        "active_bounds": fit.active_mask.tolist(),
        "train": scores(train),
        "test": scores(test),
        "test_coefficients_used": False,
        "test_rule": "boundary sectors sector_index%5==0 held out, selected before optimization",
    }, residual


def run_pothole_native_study(
    dataset_root: Path, output_dir: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Fit shared-boundary ellipse and harmonic contours in per-view native units."""
    from roadmc.scripts import diagnose_pothole_profiles as diag

    dataset_root, out = Path(dataset_root), Path(output_dir)
    _new_output(out)
    cfg = _section(protocol, "pothole_native", {"frames"})
    frames = list(cfg.get("frames", ["01", "02", "03"]))
    if frames != ["01", "02", "03"]:
        raise ValueError(
            "pothole_native.frames is frozen to ['01','02','03'] to match the reused depth-profile run"
        )
    frozen = {
        "status": "native_units_within_view_descriptive_only",
        "frames": frames,
        "units": "XYZ scale unknown; do not convert to mm or volume",
        "plane": "finite points outside largest 8-connected annotation dilation by 20..60 px; TLS/SVD",
        "boundary": "largest 8-connected annotated component; full observed boundary is shared by both outline fits",
        "split": "16-pixel blocks; boundary sector index modulo 5 equal to 0 held out",
        "harmonics": "n=3,4,5; least_squares normalized radial residual r/g(theta)-1; pairwise coefficient norm sum <1",
        "ellipse": "same TLS projected observed boundary; fit_outline residual",
        "depth": "existing diagnose_pothole_profiles.py invoked once in nested output with same frozen 16-pixel spatial blocks",
        "limits": [
            "Three views may depict one pit; independence unknown.",
            "Known full outline makes this a transductive interpolation diagnostic.",
            "No mm, physical volume, road-population, or coefficient-prior claim.",
            "No parameters are derived as priors from the three frames.",
        ],
    }
    _json(out / "frozen_protocol.json", {**frozen, "protocol_input": protocol})
    depth_script = Path(diag.__file__).resolve()
    source_paths = [
        Path(__file__).resolve(),
        depth_script,
        Path(__file__).resolve().parents[1] / "data/synthetic/pothole_profile.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/pothole_contour.py",
        Path(__file__).resolve().parents[1] / "data/synthetic/surface_replay.py",
    ]
    native_hashes: dict[str, str] = {}
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for source in source_paths:
        native_hashes[str(source)] = _sha256(source)
        (snapshots / source.name).write_bytes(source.read_bytes())
    for frame in frames:
        for source in (
            dataset_root / "ptcloud" / f"{frame}.mat",
            dataset_root / "label" / f"{frame}.png",
        ):
            native_hashes[str(source)] = _sha256(source)
    _json(out / "source_hashes_before_run.json", native_hashes)
    # Reuse exactly the existing depth diagnostic and keep it isolated as provenance.
    depth_dir = out / "existing_depth_profile_diagnostic"
    diag.run(dataset_root, depth_dir)
    depth_summary = json.loads((depth_dir / "audit.json").read_text(encoding="utf-8"))
    frame_results: dict[str, Any] = {}
    fig, axes = plt.subplots(
        len(frames),
        3,
        figsize=(12, max(4, 3.6 * len(frames))),
        squeeze=False,
        constrained_layout=True,
    )
    source_hashes: dict[str, str] = {}
    for row, frame in enumerate(frames):
        cloud = dataset_root / "ptcloud" / f"{frame}.mat"
        label_path = dataset_root / "label" / f"{frame}.png"
        source_hashes[str(cloud)] = native_hashes[str(cloud)]
        source_hashes[str(label_path)] = native_hashes[str(label_path)]
        xyz = np.asarray(
            loadmat(cloud, verify_compressed_data_integrity=True)["xyzPoints"], dtype=float
        )
        annotated = np.asarray(Image.open(label_path).convert("RGB")).max(axis=-1) > 0
        valid = np.isfinite(xyz).all(axis=-1)
        components, count = label(annotated, structure=np.ones((3, 3), bool))
        if count <= 0:
            raise ValueError(f"no pothole annotation in frame {frame}")
        component = components == 1 + np.argmax(np.bincount(components.ravel())[1:])
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

        def project(p, origin=origin, tangent=tangent, other=other):
            return np.column_stack(((p - origin) @ tangent, (p - origin) @ other))

        boundary_all = component & ~binary_erosion(
            component, structure=np.ones((3, 3), bool), border_value=0
        )
        boundary = boundary_all & valid
        boundary_xy = project(xyz[boundary])
        outline, local, ellipse_rms = _fit_outline(boundary_xy)
        uv = local(outline, boundary_xy)
        theta = np.arctan2(uv[:, 1], uv[:, 0])
        # 16 equal angular bins make spatially interpretable outline holdouts.
        sector = np.floor(((theta + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * 16).astype(int)
        test = sector % 5 == 0
        train = ~test
        harmonic, harmonic_residual = _fit_radial_harmonics(uv, train, test)
        ellipse_residual = np.linalg.norm(uv, axis=1) - 1
        from roadmc.data.synthetic.outline_diagnostics import score_outline_boundary

        outline_fairness, _ = score_outline_boundary(
            boundary_xy,
            outline,
            harmonic["coefficients"],
            complete_edge_count=int(boundary_all.sum()),
            finite_xyz_edge_count=int(boundary.sum()),
            pixel_yx=np.argwhere(boundary),
        )

        def scored(res, mask):
            return {
                "points": int(mask.sum()),
                "radial_relative_rmse": float(np.sqrt(np.mean(res[mask] ** 2))),
                "radial_relative_mae": float(np.mean(np.abs(res[mask]))),
            }

        # Height residual of existing symmetric and tilted candidates is carried over verbatim.
        depth_models = depth_summary["frames"][frame]["models"]
        frame_results[frame] = {
            "source_sha256": {
                str(cloud): source_hashes[str(cloud)],
                str(label_path): source_hashes[str(label_path)],
            },
            "finite_tls_plane_points": int(len(road)),
            "boundary_points": int(len(uv)),
            "boundary_split_counts": {"train": int(train.sum()), "test": int(test.sum())},
            "outline_fairness_and_boundary_quality": outline_fairness,
            "plane_origin_native": origin.tolist(),
            "plane_normal": normal.tolist(),
            "ellipse_parameters_center_logaxes_angle": outline.tolist(),
            "ellipse_fit_radial_rms": ellipse_rms,
            "outline_models": {
                "ellipse_shared_outline": {
                    "train": scored(ellipse_residual, train),
                    "test": scored(ellipse_residual, test),
                    "known_boundary_shared": True,
                },
                "area_normalized_radial_harmonic": harmonic,
            },
            "depth_profiles_existing_16px_block_split": depth_models,
        }
        # Plot the observed projected boundary, ellipse, and harmonic predicted radii.
        t = np.linspace(-np.pi, np.pi, 720)
        ellipse_xy = np.column_stack((np.cos(t), np.sin(t)))
        harmonic_xy = (
            _radial_scale(
                t, np.r_[harmonic["coefficients"]["cos"], harmonic["coefficients"]["sin"]]
            )[:, None]
            * ellipse_xy
        )
        axes[row, 0].scatter(uv[:, 0], uv[:, 1], s=3, label="observed boundary")
        axes[row, 0].plot(ellipse_xy[:, 0], ellipse_xy[:, 1], label="shared ellipse")
        axes[row, 0].plot(harmonic_xy[:, 0], harmonic_xy[:, 1], label="harmonic")
        axes[row, 0].set(title=f"{frame}: normalized boundary", aspect="equal")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(np.rad2deg(theta), ellipse_residual, ".", ms=2, label="ellipse")
        axes[row, 1].plot(np.rad2deg(theta), harmonic_residual, ".", ms=2, label="harmonic")
        axes[row, 1].set(title="Boundary radial residual", xlabel="angle (deg)")
        axes[row, 1].legend(fontsize=7)
        names = ["symmetric_p1", "tilted_power_bowl"]
        vals = [depth_models[n]["test"]["rmse_over_train_q90"] for n in names]
        axes[row, 2].bar(names, vals)
        axes[row, 2].set(title="Existing depth holdout RMSE", ylabel="train q90 units")
    fig.savefig(out / "native_outline_and_depth_diagnostics.png", dpi=150)
    plt.close(fig)
    _json(out / "source_hashes.json", source_hashes)
    _json(
        out / "metrics.json",
        {"frames": frame_results, "depth_diagnostic_protocol": depth_summary["protocol"]},
    )
    _json(
        out / "executing_source_hashes.json", {str(p): native_hashes[str(p)] for p in source_paths}
    )
    (out / "report_zh.md").write_text(
        "# 坑槽原生单位轮廓实验\n\n【原生单位、视图内描述】XYZ 的尺度未知；只报告每视图归一化残差，不换算毫米、体积或道路总体分布。椭圆与径向谐波共享同一已知整帧标签边界，并按角向扇区留出，因此属于传导式轮廓插值诊断。深度对比复用既有 16 像素空间块诊断。三视图是否独立未知，不从这些帧构造系数先验。完整拟合日志、边界和哈希见 metrics.json。\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out),
        "metrics_path": str(out / "metrics.json"),
        "frames": list(frame_results),
        "source_hashes": source_hashes,
        "status": frozen["status"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="lane", required=True)
    geometry = sub.add_parser("geometry-material")
    geometry.add_argument("--output-dir", type=Path, required=True)
    native = sub.add_parser("pothole-native")
    native.add_argument("--dataset-root", type=Path, required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    fairness = sub.add_parser("pothole-outline-fairness")
    fairness.add_argument("--dataset-root", type=Path, required=True)
    fairness.add_argument("--frozen-metrics", type=Path, required=True)
    fairness.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.lane == "geometry-material":
        run_geometry_material_study(args.output_dir, {})
    elif args.lane == "pothole-native":
        run_pothole_native_study(args.dataset_root, args.output_dir, {})
    else:
        from roadmc.data.synthetic.outline_diagnostics import run_pothole_outline_fairness_study

        run_pothole_outline_fairness_study(args.dataset_root, args.output_dir, args.frozen_metrics)


if __name__ == "__main__":
    main()
