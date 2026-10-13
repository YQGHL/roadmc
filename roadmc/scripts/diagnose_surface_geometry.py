"""Render geometry diagnostics from a paired, metric, road-only surface NPZ.

Required arrays are ``base_points (N0, 3)``, ``surface_points (N, 3)``,
``surface_labels (N,)``, and ``surface_normals (N, 3)``. Coordinates are in
meters and must not be normalized. Example::

    python -m roadmc.scripts.diagnose_surface_geometry capture.npz --output-dir out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = ("base_points", "surface_points", "surface_labels", "surface_normals")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"NPZ missing required arrays: {', '.join(missing)}")
        base = np.asarray(data["base_points"], dtype=np.float64)
        surface = np.asarray(data["surface_points"], dtype=np.float64)
        labels = np.asarray(data["surface_labels"])
        normals = np.asarray(data["surface_normals"], dtype=np.float64)
        metadata = _json_scalar(data["metadata"]) if "metadata" in data.files else None
    if base.ndim != 2 or base.shape[1] != 3 or surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError("base_points and surface_points must have shape (N, 3)")
    if labels.shape != (len(surface),) or normals.shape != surface.shape:
        raise ValueError("surface_labels must be (N,) and surface_normals must match surface_points")
    if not len(base) or not len(surface):
        raise ValueError("base_points and surface_points must not be empty")
    if not np.isfinite(base).all() or not np.isfinite(surface).all():
        raise ValueError("Point coordinates contain non-finite values; fix the capture before plotting")
    return base, surface, labels, normals, metadata


def _metric_grid(base: np.ndarray, surface: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Match capture points to the base grid by XY, retaining holes as NaN."""
    tree = cKDTree(base[:, :2])
    distance, base_index = tree.query(surface[:, :2], k=1, workers=-1)
    tol = max(1e-9, np.finfo(np.float64).eps * max(np.abs(base[:, :2]).max(), 1.0) * 16)
    if np.any(distance > tol):
        count = int(np.count_nonzero(distance > tol))
        raise ValueError(
            f"{count} surface XY locations do not exactly match base XY (tolerance {tol:g} m); "
            "the paired capture must share its sampling grid"
        )

    x_values, x_inverse = np.unique(base[:, 0], return_inverse=True)
    y_values, y_inverse = np.unique(base[:, 1], return_inverse=True)
    shape = (len(x_values), len(y_values))
    base_z_sum = np.zeros(shape, dtype=np.float64)
    base_count = np.zeros(shape, dtype=np.int32)
    np.add.at(base_z_sum, (x_inverse, y_inverse), base[:, 2])
    np.add.at(base_count, (x_inverse, y_inverse), 1)
    base_z = np.full(shape, np.nan)
    base_mask = base_count > 0
    base_z[base_mask] = base_z_sum[base_mask] / base_count[base_mask]

    bx = x_inverse[base_index]
    by = y_inverse[base_index]
    residual_sum = np.zeros(shape, dtype=np.float64)
    surface_sum = np.zeros(shape, dtype=np.float64)
    label_grid = np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=np.int32)
    residual = surface[:, 2] - base[base_index, 2]
    np.add.at(residual_sum, (bx, by), residual)
    np.add.at(surface_sum, (bx, by), surface[:, 2])
    np.add.at(counts, (bx, by), 1)
    # Duplicate samples at a cell use majority label; ties choose the smaller label.
    # Group (flat cell, label) pairs once instead of rescanning all samples per cell.
    flat_cell = bx * shape[1] + by
    cell_label_pairs = np.column_stack((flat_cell, labels))
    pairs, pair_counts = np.unique(cell_label_pairs, axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -pair_counts, pairs[:, 0]))
    ordered_cells = pairs[order, 0]
    first_for_cell = np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
    winners = pairs[order[first_for_cell]]
    label_grid.ravel()[winners[:, 0].astype(np.intp)] = winners[:, 1]
    valid = counts > 0
    residual_grid = np.full(shape, np.nan)
    surface_grid = np.full(shape, np.nan)
    residual_grid[valid] = residual_sum[valid] / counts[valid]
    surface_grid[valid] = surface_sum[valid] / counts[valid]
    return {
        "x": x_values, "y": y_values, "base": base_z, "surface": surface_grid,
        "residual": residual_grid, "labels": label_grid, "counts": counts,
        "sample_residual": residual,
    }


def _crop_bounds(grid: dict[str, Any], labels: np.ndarray, label_grid: np.ndarray,
                 target_label: int | None, crop: list[float] | None, padding: float) -> tuple[float, float, float, float]:
    x, y = grid["x"], grid["y"]
    if crop is not None:
        x0, x1, y0, y1 = crop
    elif target_label is not None and np.any(labels == target_label):
        hit = label_grid == target_label
        ix, iy = np.where(hit)
        x0, x1 = x[ix].min() - padding, x[ix].max() + padding
        y0, y1 = y[iy].min() - padding, y[iy].max() + padding
    else:
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Crop bounds must have increasing x and y ranges")
    return max(x0, x.min()), min(x1, x.max()), max(y0, y.min()), min(y1, y.max())


def _range(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"min": None, "max": None}
    return {"min": float(finite.min()), "max": float(finite.max())}


def render(npz_path: Path, output_dir: Path, *, label: int | None = None,
           crop: list[float] | None = None, padding: float = 0.25,
           max_3d_points: int = 100_000, seed: int = 0) -> dict[str, Any]:
    base, surface, labels, normals, metadata = _load(npz_path)
    grid = _metric_grid(base, surface, labels)
    bounds = _crop_bounds(grid, labels, grid["labels"], label, crop, padding)
    x0, x1, y0, y1 = bounds
    xi = np.flatnonzero((grid["x"] >= x0) & (grid["x"] <= x1))
    yi = np.flatnonzero((grid["y"] >= y0) & (grid["y"] <= y1))
    if not len(xi) or not len(yi):
        raise ValueError("Crop contains no grid locations")
    xs, ys = grid["x"][xi], grid["y"][yi]
    extent = [ys.min(), ys.max(), xs.min(), xs.max()]
    height = grid["surface"][np.ix_(xi, yi)]
    residual = grid["residual"][np.ix_(xi, yi)]
    label_map = grid["labels"][np.ix_(xi, yi)]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Geometry has no label overlay: disease boundaries cannot hide surface shape.
    rng = np.random.default_rng(seed)
    take = min(max_3d_points, len(surface))
    selected = rng.choice(len(surface), take, replace=False) if take < len(surface) else np.arange(len(surface))
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(surface[selected, 1], surface[selected, 0], surface[selected, 2],
                    c=surface[selected, 2], cmap="viridis", s=0.35, linewidths=0)
    spans = np.ptp(surface, axis=0)
    spans[spans == 0] = 1e-9
    ax.set_box_aspect(spans[[1, 0, 2]])  # plotted axes are y/x/z, all in meters.
    bounds_min = surface.min(axis=0)
    ax.set_xlim(bounds_min[1], bounds_min[1] + spans[1])
    ax.set_ylim(bounds_min[0], bounds_min[0] + spans[0])
    ax.set_zlim(bounds_min[2], bounds_min[2] + spans[2])
    # The colorbar provides z values without overlapping tick labels on a
    # very flat, physically proportioned vertical axis.
    ax.set_zticks([])
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Road-only surface geometry — {take:,}/{len(surface):,} points; z in meters; true-span aspect")
    fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6).set_label("z (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_3d.png")
    plt.close(fig)

    finite_height = height[np.isfinite(height)]
    hlim = np.percentile(finite_height, [1, 99]) if finite_height.size else (None, None)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    image = ax.imshow(height, origin="lower", extent=extent, aspect="equal", cmap="terrain",
                      vmin=hlim[0], vmax=hlim[1], interpolation="nearest")
    fig.colorbar(image, ax=ax).set_label("surface z (m); original grid, holes shown blank")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Surface height (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "surface_height.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    finite_residual = residual[np.isfinite(residual)]
    rlim = np.percentile(np.abs(finite_residual), 99) if finite_residual.size else 1.0
    rlim = max(float(rlim), np.finfo(float).eps)
    image = ax.imshow(residual, origin="lower", extent=extent, aspect="equal", cmap="coolwarm",
                      vmin=-rlim, vmax=rlim, interpolation="nearest")
    fig.colorbar(image, ax=ax).set_label("residual z = surface z − base z (m); symmetric 99th percentile scale")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Paired height displacement residual (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "height_residual.png")
    plt.close(fig)

    # Labels are an independent categorical plot with their own legend/colorbar.
    unique_labels = np.unique(labels)
    cmap = ListedColormap(plt.get_cmap("tab20", max(len(unique_labels), 1)).colors)
    boundaries = np.arange(len(unique_labels) + 1, dtype=float) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    label_indices = np.full(label_map.shape, np.nan)
    finite_label = np.isfinite(label_map)
    label_indices[finite_label] = np.searchsorted(unique_labels, label_map[finite_label])
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    image = ax.imshow(np.ma.masked_invalid(label_indices), origin="lower", extent=extent,
                      aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, ticks=np.arange(len(unique_labels)))
    cb.ax.set_yticklabels([str(v) for v in unique_labels])
    cb.set_label("surface label")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Surface labels (no geometry overlay)")
    fig.tight_layout()
    fig.savefig(output_dir / "surface_labels.png")
    plt.close(fig)

    # Find maximum |residual| in the chosen region and plot orthogonal profiles.
    local_abs = np.abs(residual)
    if not np.isfinite(local_abs).any():
        raise ValueError("The selected crop has no matched residual samples")
    peak_i, peak_j = np.unravel_index(np.nanargmax(local_abs), local_abs.shape)
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax_x.plot(xs, height[:, peak_j], label="surface z")
    ax_x.plot(xs, grid["base"][np.ix_(xi, yi)][:, peak_j], label="base z")
    ax_x.set_xlabel("x (m)")
    ax_x.set_ylabel("z (m)")
    ax_x.set_title(f"x profile at y={ys[peak_j]:.4f} m")
    ax_x.legend()
    ax_y.plot(ys, height[peak_i, :], label="surface z")
    ax_y.plot(ys, grid["base"][np.ix_(xi, yi)][peak_i, :], label="base z")
    ax_y.set_xlabel("y (m)")
    ax_y.set_ylabel("z (m)")
    ax_y.set_title(f"y profile at x={xs[peak_i]:.4f} m")
    ax_y.legend()
    fig.suptitle(f"Profiles at peak displacement; signed residual = {residual[peak_i, peak_j]:.6g} m")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_profiles.png")
    plt.close(fig)

    normal_lengths = np.linalg.norm(normals, axis=1)
    finite_normals = np.isfinite(normal_lengths)
    norm_deviation = np.abs(normal_lengths[finite_normals] - 1.0)
    finite_res = grid["sample_residual"][np.isfinite(grid["sample_residual"])]
    label_values, label_counts = np.unique(labels, return_counts=True)
    summary = {
        "source": str(npz_path), "units": "meters", "road_only": True,
        "point_counts": {"base": int(len(base)), "surface": int(len(surface))},
        "finite_values": {
            "base_points": int(np.isfinite(base).sum()), "surface_points": int(np.isfinite(surface).sum()),
            "surface_normals": int(np.isfinite(normals).sum()), "labels": int(np.isfinite(labels).sum()),
        },
        "normal_unit_length_deviation": {
            "finite_count": int(finite_normals.sum()), "mean_abs": float(norm_deviation.mean()) if norm_deviation.size else None,
            "max_abs": float(norm_deviation.max()) if norm_deviation.size else None,
        },
        "residual_z_m": {**_range(finite_res), "matched_count": int(finite_res.size),
                          "grid_resolution_xy_m": [float(np.median(np.diff(grid["x"]))) if len(grid["x"]) > 1 else None,
                                                    float(np.median(np.diff(grid["y"]))) if len(grid["y"]) > 1 else None]},
        "label_counts": {
            str(v.item() if isinstance(v, np.generic) else v): int(n)
            for v, n in zip(label_values, label_counts, strict=True)
        },
        "crop_xy_m": {"x_min": float(x0), "x_max": float(x1), "y_min": float(y0), "y_max": float(y1)},
        "max_abs_residual_in_crop": {"x_m": float(xs[peak_i]), "y_m": float(ys[peak_j]),
                                     "residual_z_m": float(residual[peak_i, peak_j])},
        "metadata": metadata,
        "plots": ["geometry_3d.png", "surface_height.png", "height_residual.png",
                  "surface_labels.png", "residual_profiles.png"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="paired metric road-only geometry capture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=int, help="crop to this disease label with --padding around its bbox")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                        help="explicit crop bounds in meters")
    parser.add_argument("--padding", type=float, default=0.25, help="meters around selected label bbox (default: 0.25)")
    parser.add_argument("--max-3d-points", type=int, default=100_000,
                        help="maximum points in the 3D scatter; 2D maps retain the original grid")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_3d_points < 1 or args.padding < 0:
        parser.error("--max-3d-points must be positive and --padding non-negative")
    # Use an installed Chinese font when present; DejaVu remains a safe fallback.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    summary = render(args.npz, args.output_dir, label=args.label, crop=args.crop,
                     padding=args.padding, max_3d_points=args.max_3d_points, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
