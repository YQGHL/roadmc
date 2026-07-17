"""Render RoadMC point cloud scenes into multiple 2D/3D view images.

Outputs:
- 3D point cloud distribution figure
- 2D grayscale height map
- 2D heatmap (height / curvature / labels / intensity)
- Optional contact sheet montage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_scene(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = {"points", "labels", "feats"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {sorted(missing)}")
    return {
        "points": np.asarray(data["points"], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.int32),
        "feats": np.asarray(data["feats"], dtype=np.float32),
        "normals": np.asarray(data["normals"], dtype=np.float32)
        if "normals" in data.files
        else None,
    }


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        return np.zeros_like(grid, dtype=np.float32)
    lo = float(np.nanmin(grid))
    hi = float(np.nanmax(grid))
    if hi - lo < 1e-12:
        return np.zeros_like(grid, dtype=np.float32)
    out = (grid - lo) / (hi - lo)
    out[~valid] = 0.0
    return out.astype(np.float32)


def rasterize_feature(
    points: np.ndarray,
    values: np.ndarray,
    resolution: int = 256,
    reducer: str = "mean",
) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    eps = 1e-9
    xbins = np.linspace(xmin, xmax + eps, resolution + 1)
    ybins = np.linspace(ymin, ymax + eps, resolution + 1)

    xi = np.clip(np.digitize(x, xbins) - 1, 0, resolution - 1)
    yi = np.clip(np.digitize(y, ybins) - 1, 0, resolution - 1)

    sums = np.zeros((resolution, resolution), dtype=np.float64)
    counts = np.zeros((resolution, resolution), dtype=np.int32)
    maxs = np.full((resolution, resolution), -np.inf, dtype=np.float64)

    for idx in range(points.shape[0]):
        row = resolution - 1 - yi[idx]
        col = xi[idx]
        value = float(values[idx])
        sums[row, col] += value
        counts[row, col] += 1
        if value > maxs[row, col]:
            maxs[row, col] = value

    if reducer == "max":
        grid = maxs
        grid[~np.isfinite(grid)] = np.nan
        return grid

    grid = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return grid


def interpolate_feature_grid(
    points: np.ndarray,
    values: np.ndarray,
    resolution: int = 256,
) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(float(x.min()), float(x.max()), resolution)
    yi = np.linspace(float(y.min()), float(y.max()), resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)

    linear_grid = griddata((x, y), values, (grid_x, grid_y), method="linear")
    nearest_grid = griddata((x, y), values, (grid_x, grid_y), method="nearest")
    if linear_grid is None and nearest_grid is None:
        raise RuntimeError("Failed to interpolate point cloud feature grid.")
    if linear_grid is None:
        return np.flipud(nearest_grid.astype(np.float32))
    if nearest_grid is None:
        return np.flipud(linear_grid.astype(np.float32))

    filled = np.where(np.isfinite(linear_grid), linear_grid, nearest_grid)
    return np.flipud(filled.astype(np.float32))


def render_3d_distribution(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=28, azim=-62)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=labels,
        cmap="turbo",
        s=2,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label="Label")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_image(
    grid: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    image = ax.imshow(grid, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    titles: list[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)
    for ax, image_path, title in zip(axes.flat, image_paths, titles):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    for ax in axes.flat[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render RoadMC point cloud scene views")
    parser.add_argument("scene", type=Path, help="Path to scene_XXXX.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to sibling folder named renders/<scene-stem>",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Raster resolution for 2D maps",
    )
    parser.add_argument(
        "--heatmap-source",
        choices=["height", "curvature", "intensity", "labels"],
        default="curvature",
        help="Feature used for the heatmap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()
    scene = load_scene(scene_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else scene_path.parent / "renders" / scene_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    points = scene["points"]
    labels = scene["labels"]
    feats = scene["feats"]

    height_values = points[:, 2]
    curvature_values = feats[:, 1]
    intensity_values = feats[:, 0]

    grayscale_grid = normalize_grid(
        interpolate_feature_grid(points, height_values, resolution=args.resolution)
    )

    feature_map = {
        "height": height_values,
        "curvature": curvature_values,
        "intensity": intensity_values,
        "labels": labels.astype(np.float32),
    }
    if args.heatmap_source == "labels":
        heatmap_grid = rasterize_feature(
            points,
            feature_map[args.heatmap_source],
            resolution=args.resolution,
            reducer="max",
        )
    else:
        heatmap_grid = interpolate_feature_grid(
            points,
            feature_map[args.heatmap_source],
            resolution=args.resolution,
        )

    cloud_path = output_dir / f"{scene_path.stem}_3d_distribution.png"
    gray_path = output_dir / f"{scene_path.stem}_grayscale_height.png"
    heat_path = output_dir / f"{scene_path.stem}_{args.heatmap_source}_heatmap.png"
    montage_path = output_dir / f"{scene_path.stem}_contact_sheet.png"

    render_3d_distribution(points, labels, cloud_path, f"{scene_path.stem} - 3D distribution")
    render_image(grayscale_grid, gray_path, f"{scene_path.stem} - grayscale height", "gray", "Normalized height")
    render_image(
        heatmap_grid,
        heat_path,
        f"{scene_path.stem} - {args.heatmap_source} heatmap",
        "inferno",
        args.heatmap_source,
    )

    render_contact_sheet(
        [cloud_path, gray_path, heat_path],
        montage_path,
        ["3D distribution", "2D grayscale", "Heatmap"],
    )

    print(f"Saved renders to: {output_dir}")


if __name__ == "__main__":
    main()
