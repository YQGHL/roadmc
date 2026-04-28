"""
RoadMC 阶段一点云生成测试与可视化 (v2)。

该脚本：
1. 生成多个包含不同病害类型的点云场景
2. 验证点云输出的 shape 和标签范围
3. 二维叠加图 — 俯视图 + JTG 标签着色
4. 灰度高度图 — z 值映射 + 病害叠加 + 等高线
5. 3D 视图 — 曲面网格 + 散点 + 真实比例
6. 特征通道可视化 — intensity / curvature / crack_boundary_dist
7. 标签分布统计图

输出目录: roadmc/test/output/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import numpy as np

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from roadmc.data.synthetic.config import (
    NUM_CLASSES, LABEL_MAP,
    GeneratorConfig, RoadSurfaceConfig, DiseaseConfig,
    CrackConfig, PotholeConfig, RuttingConfig,
    MicroTextureConfig, LidarNoiseConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLORS = {
    0: "#E8E8E8", 1: "#FF4444", 2: "#CC0000", 3: "#FF8844", 4: "#CC5500",
    5: "#EEDD44", 6: "#CCAA00", 7: "#44DD44", 8: "#00AA00", 9: "#4488FF",
    10: "#0044CC", 11: "#AA44FF", 12: "#7700CC", 13: "#FF44AA", 14: "#CC0077",
    15: "#44EEEE", 16: "#00AAAA", 17: "#FFAA44", 18: "#CC7700", 19: "#FF88FF",
    20: "#BBBBBB",
    21: "#994444", 22: "#662222", 23: "#998844", 24: "#665522",
    25: "#449944", 26: "#226622", 27: "#448899", 28: "#225566",
    29: "#88CCFF", 30: "#996644", 31: "#664422", 32: "#AAAACC",
    33: "#7777AA", 34: "#336699", 35: "#CC8844", 36: "#AA8866", 37: "#999999",
}


def generate_test_scenes() -> list[dict]:
    configs = [
        {
            "name": "asphalt_multi_disease",
            "config": GeneratorConfig(
                seed=42, num_points=65536, normalize=False,
                road=RoadSurfaceConfig(width=3.0, length=5.0, grid_res=0.015,
                                       pavement_type="asphalt", roughness_class="B"),
                micro_texture=MicroTextureConfig(amplitude=0.0008, hurst=0.7, octaves=5),
                disease=DiseaseConfig(
                    disease_probs={"crack": 0.9, "pothole": 0.6, "rutting": 0.5},
                    severity_ratio=0.5, max_diseases_per_scene=3,
                ),
                lidar_noise=LidarNoiseConfig(distance_noise_std=0.003, dropout_rate=0.02),
            ),
        },
        {
            "name": "asphalt_rutting_corrugation",
            "config": GeneratorConfig(
                seed=100, num_points=65536, normalize=False,
                road=RoadSurfaceConfig(width=3.0, length=5.0, grid_res=0.015,
                                       pavement_type="asphalt", roughness_class="C"),
                disease=DiseaseConfig(
                    disease_probs={"rutting": 0.95, "corrugation": 0.95},
                    severity_ratio=0.4, max_diseases_per_scene=2,
                ),
                lidar_noise=LidarNoiseConfig(distance_noise_std=0.002, dropout_rate=0.01),
            ),
        },
        {
            "name": "concrete_damage",
            "config": GeneratorConfig(
                seed=200, num_points=65536, normalize=False,
                road=RoadSurfaceConfig(width=3.0, length=5.0, grid_res=0.015,
                                       pavement_type="concrete", roughness_class="A"),
                disease=DiseaseConfig(
                    disease_probs={"concrete_damage": 0.95},
                    severity_ratio=0.5, max_diseases_per_scene=1,
                ),
                lidar_noise=LidarNoiseConfig(distance_noise_std=0.002, dropout_rate=0.01),
            ),
        },
    ]

    scenes = []
    for cfg_info in configs:
        print(f"\n  生成: {cfg_info['name']}...")
        ds = SyntheticRoadDataset(config=cfg_info["config"], dataset_size=1)
        scene = ds.generate_scene(0)

        pts = scene["points"]
        labels = scene["labels"]
        normals = scene["normals"]
        feats = scene["feats"]
        pt = scene["pavement_type"]

        print(f"    points: {pts.shape}, labels: {labels.shape}, feats: {feats.shape}")
        print(f"    pavement: {pt}, labels range: [{labels.min()}, {labels.max()}]")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            info = LABEL_MAP.get(int(u), {})
            print(f"      label {u}: {info.get('type', '?')} ({info.get('severity', '-')}) — {c} pts ({100*c/len(labels):.1f}%)")

        scenes.append(dict(name=cfg_info["name"], points=pts, labels=labels,
                          normals=normals, feats=feats, pavement_type=pt,
                          config=cfg_info["config"]))
    return scenes


def plot_2d_overlay(scene: dict):
    pts, labels = scene["points"], scene["labels"]
    name = scene["name"]
    unique_labels = np.unique(labels)
    non_bg = unique_labels[unique_labels > 0]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors_arr = np.array([LABEL_COLORS.get(int(l), "#000") for l in labels])
    rgba = np.array([mcolors.to_rgba(c) for c in colors_arr])
    pt_size = max(0.3, min(3.0, 80000 / len(pts)))

    ax.scatter(pts[:, 0], pts[:, 1], c=rgba, s=pt_size, alpha=0.75, edgecolors="none")
    ax.set_title(f"{name}  —  2D Overlay ({len(non_bg)} disease types)  |  {scene['pavement_type']}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("X — Width (m)"); ax.set_ylabel("Y — Length (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    legend_elements = []
    for lbl in sorted(non_bg):
        info = LABEL_MAP.get(int(lbl), {})
        legend_elements.append(plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS.get(int(lbl), "#000"),
            markersize=8, label=f"{int(lbl)}: {info.get('type', '?')} ({info.get('severity', '-')})"))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.85)

    fp = OUTPUT_DIR / f"{name}_2d_overlay.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"    -> {fp.name}")


def plot_grayscale_height(scene: dict):
    pts, labels = scene["points"], scene["labels"]
    name = scene["name"]
    z = pts[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 左: 纯灰度高度图 + 等高线 ---
    ax = axes[0]
    x, y = pts[:, 0], pts[:, 1]
    try:
        grid_res = 0.03
        xi = np.arange(x.min(), x.max(), grid_res)
        yi = np.arange(y.min(), y.max(), grid_res)
        from scipy.interpolate import griddata
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="linear")
        im = ax.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()],
                       origin="lower", cmap="gray_r", aspect="equal")
        levels = np.linspace(np.nanmin(zi), np.nanmax(zi), 12)
        ax.contour(xi, yi, zi, levels=levels, colors="#4444FF", linewidths=0.4, alpha=0.5)
    except Exception:
        ax.scatter(x, y, c=z, cmap="gray_r", s=0.3)
    ax.set_title(f"Height Map + Contours\nz∈[{z.min()*1e3:.1f}, {z.max()*1e3:.1f}] mm", fontsize=10)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label="Height (m)", shrink=0.75)

    # --- 中: 高度 + 病害叠加 ---
    ax2 = axes[1]
    bg = labels == 0; fg = labels > 0
    ax2.scatter(x[bg], y[bg], c=z[bg], cmap="gray_r", s=0.2, alpha=0.5)
    if np.any(fg):
        fg_c = np.array([mcolors.to_rgba(LABEL_COLORS.get(int(l), "#F00"), 0.7) for l in labels[fg]])
        ax2.scatter(x[fg], y[fg], c=fg_c, s=1.5, edgecolors="none")
    ax2.set_title("Height + Disease Overlay", fontsize=10)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_aspect("equal")

    # --- 右: 高度剖面 (沿 y 方向中心线) ---
    ax3 = axes[2]
    x_mid = (x.min() + x.max()) / 2
    band = np.abs(x - x_mid) < 0.10
    if np.any(band):
        idx = np.argsort(y[band])
        ax3.plot(y[band][idx], z[band][idx] * 1e3, "k-", linewidth=0.5, alpha=0.7)
    ax3.set_title("Longitudinal Profile (x≈mid)", fontsize=10)
    ax3.set_xlabel("Y (m)"); ax3.set_ylabel("Z (mm)")

    fp = OUTPUT_DIR / f"{name}_grayscale_height.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"    -> {fp.name}")


def plot_3d_view(scene: dict):
    pts, labels = scene["points"], scene["labels"]
    name = scene["name"]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=(16, 6))

    # --- 左: 曲面网格 (Trisurf) + 标签着色 ---
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    try:
        from scipy.spatial import Delaunay
        # 降采样避免内存爆炸
        n_sample = min(len(pts), 8000)
        idx = np.random.default_rng(42).choice(len(pts), n_sample, replace=False)
        tri = Delaunay(np.column_stack([x[idx], y[idx]]))
        fc = np.array([LABEL_COLORS.get(int(l), "#888888") for l in labels[idx]])
        ax.plot_trisurf(x[idx], y[idx], z[idx] * 1e3, triangles=tri.simplices,
                        facecolor=fc, edgecolor="none", alpha=0.85, shade=True)
    except Exception:
        bg = labels == 0
        ax.scatter(x[bg], y[bg], z[bg] * 1e3, c="#CCCCCC", s=0.2, alpha=0.4, depthshade=True)
        if np.any(~bg):
            fg_c = [LABEL_COLORS.get(int(l), "#F00") for l in labels[~bg]]
            ax.scatter(x[~bg], y[~bg], z[~bg] * 1e3, c=fg_c, s=1.5, alpha=0.7, depthshade=True)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D Surface — {name}", fontsize=11, fontweight="bold")

    # --- 右: 散点 + 法向量 ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    nrm = scene["normals"]
    n_s = min(len(pts), 3000)
    idx2 = np.random.default_rng(42).choice(len(pts), n_s, replace=False)
    fg_c2 = [LABEL_COLORS.get(int(l), "#888888") for l in labels[idx2]]
    ax2.scatter(x[idx2], y[idx2], z[idx2] * 1e3, c=fg_c2, s=1.5, alpha=0.6, depthshade=True)
    # 法向量箭头 (稀疏)
    n_s2 = min(n_s, 200)
    idx3 = idx2[:n_s2]
    scale = 0.15
    ax2.quiver(x[idx3], y[idx3], z[idx3] * 1e3,
               nrm[idx3, 0], nrm[idx3, 1], nrm[idx3, 2],
               length=scale, color="#333333", linewidth=0.3, alpha=0.6)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (mm)")
    ax2.set_title("Point Cloud + Normals", fontsize=11)

    fp = OUTPUT_DIR / f"{name}_3d.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"    -> {fp.name}")


def plot_feature_channels(scene: dict):
    """可视化 feats 的三个通道: intensity, curvature, crack_boundary_dist."""
    pts = scene["points"]
    feats = scene["feats"]
    x, y = pts[:, 0], pts[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Intensity (reflectivity)", "Curvature (PCA eigenvalue ratio)",
              "Crack Boundary Distance (m)"]
    cmaps = ["hot", "coolwarm", "plasma"]

    for i, (ax, title, cmap) in enumerate(zip(axes, titles, cmaps)):
        vals = feats[:, i]
        vmin, vmax = np.percentile(vals, [2, 98]) if np.std(vals) > 1e-10 else (vals.min(), vals.max() + 1e-6)
        sc = ax.scatter(x, y, c=vals, cmap=cmap, s=0.3, alpha=0.7, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.75)

    fig.suptitle(f"Feature Channels — {scene['name']}", fontsize=12, fontweight="bold")
    fp = OUTPUT_DIR / f"{scene['name']}_features.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"    -> {fp.name}")


def plot_label_statistics(scenes: list[dict]):
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 5))
    if len(scenes) == 1: axes = [axes]

    for ax, scene in zip(axes, scenes):
        labels = scene["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        mask = unique > 0
        if np.any(mask):
            u, c = unique[mask], counts[mask]
            names = [f"{int(l)}:{LABEL_MAP.get(int(l),{}).get('type','?')}" for l in u]
            colors = [LABEL_COLORS.get(int(l), "#999") for l in u]
            ax.bar(range(len(u)), c / 1000, color=colors, edgecolor="#333", linewidth=0.3)
            ax.set_xticks(range(len(u)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No diseases", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
        ax.set_title(scene["name"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Point Count (×1000)")

    fp = OUTPUT_DIR / "label_statistics.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"    -> {fp.name}")


def main():
    print("=" * 64)
    print("  RoadMC Stage 1 — Point Cloud Visualization Suite")
    print("=" * 64)

    print("\n[1/5] Generating test scenes...")
    scenes = generate_test_scenes()

    print("\n[2/5] 2D overlay maps...")
    for s in scenes: plot_2d_overlay(s)

    print("\n[3/5] Grayscale height maps + contours + profiles...")
    for s in scenes: plot_grayscale_height(s)

    print("\n[4/5] 3D views (surface + normals)...")
    for s in scenes: plot_3d_view(s)

    print("\n[5/5] Feature channels + statistics...")
    for s in scenes: plot_feature_channels(s)
    plot_label_statistics(scenes)

    print("\n" + "=" * 64)
    print(f"  Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    print("=" * 64)
    print("  Done.")


if __name__ == "__main__":
    main()