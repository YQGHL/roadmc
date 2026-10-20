"""Render geometry diagnostics from one explicit v2 replay NPZ (synthetic only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadmc.crack_diagnostics import analyze_crack_geometry, normalize_v2_graph_json


def _scalar(npz: Any, key: str, default: Any = None) -> Any:
    if key not in npz:
        return default
    value = npz[key]
    return value.item() if value.shape == () else value


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {"status": "unavailable", "reason": "field not present"}
    values = np.asarray(array, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"status": "unavailable", "reason": "no finite samples"}
    return {
        "status": "available",
        "sample_count": int(values.size),
        "mean_m": float(values.mean()),
        "rms_m": float(np.sqrt(np.mean(values**2))),
        "std_m": float(values.std()),
        "q05_m": float(np.quantile(values, 0.05)),
        "median_m": float(np.median(values)),
        "q95_m": float(np.quantile(values, 0.95)),
    }


def _save_figure(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    final: np.ndarray,
    texture: np.ndarray | None,
    graph: dict[str, Any],
) -> None:
    residual = base - final
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = ((base, "Frozen base height (m)"), (final, "Final geometry height (m)"),
              (residual, "Paired base − final residual (m)"))
    for axis, (array, title) in zip((axes[0, 0], axes[0, 1], axes[0, 2]), panels, strict=True):
        image = axis.imshow(array.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        fig.colorbar(image, ax=axis, shrink=0.8)
    ax = axes[1, 0]
    image = ax.imshow(residual.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), aspect="auto", cmap="coolwarm")
    for event in graph.get("events", []):
        raw_graph = event.get("graph", {})
        for edge in raw_graph.get("edges", []):
            points = np.asarray(edge.get("points_m", []), dtype=np.float64)
            if points.ndim == 2 and points.shape[1] == 2 and len(points):
                ax.plot(points[:, 0], points[:, 1], color="black", linewidth=0.8)
        for cell in raw_graph.get("closed_cells_xy_m", []) or []:
            polygon = np.asarray(cell, dtype=np.float64)
            if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon):
                polygon = np.vstack((polygon, polygon[0]))
                ax.plot(polygon[:, 0], polygon[:, 1], color="yellow", linewidth=0.8)
    ax.set_title("Residual + nominal vector graph (overlay only)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    section = axes[1, 1]
    mid_y = len(y) // 2
    section.plot(x, base[:, mid_y], label="frozen base")
    section.plot(x, final[:, mid_y], label="final geometry")
    section.set_title(f"Cross-section at y={y[mid_y]:.6g} m")
    section.set_xlabel("x (m)")
    section.set_ylabel("height (m)")
    section.legend()
    texture_axis = axes[1, 2]
    if texture is not None:
        texture_image = texture_axis.imshow(texture.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), aspect="auto")
        fig.colorbar(texture_image, ax=texture_axis, shrink=0.8)
        texture_axis.set_title("Material texture layer (m)")
        texture_axis.set_xlabel("x (m)")
        texture_axis.set_ylabel("y (m)")
    else:
        texture_axis.text(0.5, 0.5, "material_texture_m unavailable", ha="center", va="center")
        texture_axis.set_axis_off()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def diagnose(input_path: Path, output_dir: Path, thresholds_m: tuple[float, ...]) -> dict[str, Any]:
    with np.load(input_path, allow_pickle=False) as npz:
        needed = ("x_m", "y_m", "base_height_m", "reference_lowfreq_m", "geometry_m", "graph_json")
        missing = [key for key in needed if key not in npz]
        if missing:
            raise ValueError(f"replay NPZ is missing required v2 fields: {missing}")
        x = np.asarray(npz["x_m"], dtype=np.float64)
        y = np.asarray(npz["y_m"], dtype=np.float64)
        base = np.asarray(npz["base_height_m"], dtype=np.float64)
        reference = np.asarray(npz["reference_lowfreq_m"], dtype=np.float64)
        final = np.asarray(npz["geometry_m"], dtype=np.float64)
        graph_doc = _json_object(_scalar(npz, "graph_json"), "graph_json")
        metadata = _json_object(_scalar(npz, "metadata", {}), "metadata")
        loss_union = np.asarray(npz["loss_union_m"], dtype=np.float64) if "loss_union_m" in npz else None
        material_texture = np.asarray(npz["material_texture_m"], dtype=np.float64) if "material_texture_m" in npz else None
        observation_noise = _scalar(npz, "observation_noise_enabled")
        dropout_count = _scalar(npz, "dropout_count")
        flyer_mask = np.asarray(npz["flyer_mask"], dtype=bool) if "flyer_mask" in npz else None
        profile_index = np.asarray(npz["profile_index"]) if "profile_index" in npz else None

    shape = (len(x), len(y))
    if any(array.shape != shape for array in (base, reference, final)):
        raise ValueError(f"base/reference/geometry arrays must have x-major shape {shape}")
    normalized_graph = normalize_v2_graph_json(graph_doc)
    measured = analyze_crack_geometry(
        x, y, base, final, normalized_graph, loss_thresholds_m=thresholds_m
    )
    report: dict[str, Any] = {
        "schema": "roadmc.crack-geometry-diagnostic.v1",
        "status": "diagnostic_only",
        "scope": "synthetic replay geometry inspection; no real data or acceptance gate",
        "real_gap_status": "real_gap_not_evaluated",
        "input_npz": str(input_path.resolve()),
        "metadata": metadata,
        "grid": {"shape_x_major": list(shape), "x_extent_m": [float(x[0]), float(x[-1])],
                 "y_extent_m": [float(y[0]), float(y[-1])]},
        "surface_measurement": measured,
        "field_summaries": {
            "base_height_m": _summary(base),
            "reference_lowfreq_m": _summary(reference),
            "final_geometry_m": _summary(final),
            "paired_before_after_residual_m": _summary(base - final),
            "material_texture_m": _summary(material_texture),
        },
        "paired_residual_definition": "frozen full tile base_height_m minus final geometry_m; both sampled on the same grid",
        "reference_surface_limit": "base_height_m is a synthetic frozen before-state; real before-state reference surfaces, annotations and uncertainty are not supplied or evaluated",
        "observation_audit": {
            "noise_enabled": observation_noise,
            "dropout_count": dropout_count,
            "flyer_sample_count": int(flyer_mask.sum()) if flyer_mask is not None else None,
            "profile_index_values": np.unique(profile_index).tolist() if profile_index is not None else None,
            "note": "observation flags are provenance only; observed points are not used as the geometry residual input",
        },
        "width_semantics": {
            "geometry_support_width": "nominal edge half-widths summed from vector graph",
            "effective_width": "threshold-mask local inscribed-diameter proxy; not normal-crossing opening width",
            "label_width": "label metadata only; not inferred from geometry",
            "real_annotation_status": "unavailable; no independently annotated real crack reference supplied",
        },
        "synthetic_internal_loss_only": {
            "status": "available" if loss_union is not None else "unavailable",
            "field_summary": _summary(loss_union),
            "separation_note": "loss_union_m is simulator-internal max-loss bookkeeping, not an independent surface measurement and is excluded from surface_measurement",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "crack-geometry-diagnostic.json"
    report_path.write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _save_figure(output_dir / "crack-geometry-diagnostic.png", x, y, base, final,
                 material_texture, graph_doc)
    return {"json": str(report_path.resolve()), "figure": str((output_dir / "crack-geometry-diagnostic.png").resolve()), "status": report["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="v2 replay NPZ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds-mm", required=True, help="comma-separated explicit diagnostic loss thresholds in mm")
    args = parser.parse_args()
    try:
        thresholds = tuple(float(item) / 1000.0 for item in args.thresholds_mm.split(","))
        result = diagnose(args.input, args.output_dir, thresholds)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
