"""Compare explicit bowl profiles on Rui Fan dataset3; native units only.

The annotated outline and local plane are shared by both candidates. A spatial
pixel-block split tests within-view interpolation, not independent-road validity.
Numerical search limits are engineering choices, never empirical geometry priors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.optimize import least_squares

from roadmc.data.synthetic.pothole_profile import bowl_profile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fit_outline(points):
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    direction = eigenvectors[:, -1]
    axes = np.sqrt(np.maximum(2 * eigenvalues[::-1], 1e-12))
    initial = [*center, *np.log(axes), np.arctan2(direction[1], direction[0])]

    def coordinates(q, values):
        shifted = values - q[:2]
        ct, st = np.cos(q[4]), np.sin(q[4])
        return (
            (ct * shifted[:, 0] + st * shifted[:, 1]) / np.exp(q[2]),
            (-st * shifted[:, 0] + ct * shifted[:, 1]) / np.exp(q[3]),
        )

    def residual(q):
        u, v = coordinates(q, points)
        return np.hypot(u, v) - 1

    fitted = least_squares(residual, initial, max_nfev=500)
    if not fitted.success:
        raise ValueError(f"outline fit did not converge: {fitted.message}")
    return fitted.x, coordinates, float(np.sqrt(np.mean(residual(fitted.x) ** 2)))


def profile(q, u, v, asymmetric):
    config = {
        "kind": "tilted_power_bowl",
        "edge_power": float(q[1]) if asymmetric else 1.0,
        "tilt_major": float(q[2]) if asymmetric else 0.0,
        "tilt_minor": float(q[3]) if asymmetric else 0.0,
    }
    return q[0] * bowl_profile(u, v, config)[0]


def run(dataset_root, output):
    if output.exists():
        raise FileExistsError(f"output directory must not exist: {output}")
    output.mkdir(parents=True)
    protocol = {
        "status": "native_unit_within_view_interpolation_diagnostic",
        "frames": ["01", "02", "03"],
        "source": "Rui Fan stereo_pothole_datasets/dataset3",
        "units": "unknown native XYZ; no millimetre depth or volume claims",
        "outline": "largest 8-connected label; equal-weight TLS-plane-projected ellipse fit",
        "plane": "finite points in annotation-exterior dilation annulus 20..60 pixels; TLS/SVD",
        "split": "test=(pixel_x//16 + pixel_y//16)%5 == 0; remaining pixels train",
        "fit_points": "at most 30000 evenly spaced training indices; same subset for both candidates",
        "scale": "training signed-depth 90th percentile, sign chosen by training median",
        "loss": "soft_l1, f_scale=0.2 in training-q90-normalized depth units, max_nfev=300",
        "baseline": "fixed ellipse footprint, p=1, zero tilt, fitted depth amplitude",
        "candidate": "same footprint; fitted depth, power and 2 tilt coefficients",
        "search_bounds": {
            "amplitude_over_train_q90": [0, 3],
            "edge_power": [0.2, 4],
            "tilt_major_minor": [-3, 3],
            "status": "engineering choices, not measured priors",
        },
        "limits": [
            "Full-view outline and plane are known to both models; not a blind shape test.",
            "Three views may depict one pothole; independence is unconfirmed.",
            "No independent pit, road, device or absolute-geometry validation.",
            "Stereo reconstruction, labels, reference plane and ellipse mismatch affect residuals.",
            "No holdout scores are used to select parameters or change search settings.",
        ],
    }
    write_json(output / "frozen_protocol.json", protocol)
    from roadmc.data.synthetic import pothole_profile

    source_paths = [Path(__file__).resolve(), Path(pothole_profile.__file__).resolve()]
    (output / "source").mkdir()
    source_hashes = {}
    for path in source_paths:
        source_hashes[path.name] = sha256(path)
        (output / "source" / path.name).write_bytes(path.read_bytes())
    write_json(output / "source_hashes.json", source_hashes)
    results = {}
    fig, plots = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, frame in enumerate(protocol["frames"]):
        cloud_path = dataset_root / "ptcloud" / f"{frame}.mat"
        label_path = dataset_root / "label" / f"{frame}.png"
        xyz = np.asarray(
            loadmat(cloud_path, verify_compressed_data_integrity=True)["xyzPoints"], dtype=float
        )
        annotated = np.asarray(Image.open(label_path).convert("RGB")).max(axis=-1) > 0
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[:2] != annotated.shape:
            raise ValueError("XYZ and annotation must have identical image dimensions")
        valid = np.isfinite(xyz).all(axis=-1)
        structure = np.ones((3, 3), dtype=bool)
        components, count = label(annotated, structure)
        if count == 0:
            raise ValueError(f"no annotated component for {frame}")
        component = components == 1 + np.argmax(np.bincount(components.ravel())[1:])
        ring = (
            binary_dilation(component, iterations=60)
            & ~binary_dilation(component, iterations=20)
            & valid
        )
        road = xyz[ring]
        if len(road) < 20:
            raise ValueError(f"insufficient finite local-plane points for {frame}")
        road = road[:: max(1, len(road) // 100000)]
        origin = road.mean(axis=0)
        _, _, vh = np.linalg.svd(road - origin, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        tangent = vh[0] / np.linalg.norm(vh[0])
        other = np.cross(normal, tangent)
        other /= np.linalg.norm(other)

        def project(points, origin=origin, tangent=tangent, other=other):
            return np.column_stack(((points - origin) @ tangent, (points - origin) @ other))

        boundary = (
            component & ~binary_erosion(component, structure=structure, border_value=0) & valid
        )
        outline, coordinates, radial_error = fit_outline(project(xyz[boundary]))
        pixels_y, pixels_x = np.nonzero(component & valid)
        points = xyz[pixels_y, pixels_x]
        u, v = coordinates(outline, project(points))
        raw_depth = (points - origin) @ normal
        test = (pixels_x // 16 + pixels_y // 16) % 5 == 0
        train = ~test
        if train.sum() < 20 or test.sum() < 20:
            raise ValueError(f"insufficient spatial split points for {frame}")
        sign = 1 if np.median(raw_depth[train]) >= 0 else -1
        depth = raw_depth * sign
        scale = float(np.quantile(depth[train], 0.9))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("positive finite training depth scale required")
        normalized_depth = depth / scale
        train_indices = np.flatnonzero(train)
        fit_indices = train_indices[
            np.linspace(0, len(train_indices) - 1, min(30000, len(train_indices)), dtype=int)
        ]
        fitted = {}
        predictions = {}
        # Each model sees exactly the same training depths and spatial support.
        for name, asymmetric in (("symmetric_p1", False), ("tilted_power_bowl", True)):
            initial = [1.0, 1.0, 0.0, 0.0] if asymmetric else [1.0]
            bounds = (
                ([0.0, 0.2, -3.0, -3.0], [3.0, 4.0, 3.0, 3.0]) if asymmetric else ([0.0], [3.0])
            )

            def residual(
                q,
                fit_u=u[fit_indices],
                fit_v=v[fit_indices],
                target=normalized_depth[fit_indices],
                asymmetric=asymmetric,
            ):
                return profile(q, fit_u, fit_v, asymmetric) - target

            fit = least_squares(
                residual, initial, bounds=bounds, loss="soft_l1", f_scale=0.2, max_nfev=300
            )
            predicted = profile(fit.x, u, v, asymmetric)
            predictions[name] = predicted * scale
            metrics = {}
            for role, mask in (("train", train), ("test", test)):
                residual = predicted[mask] - normalized_depth[mask]
                metrics[role] = {
                    "points": int(mask.sum()),
                    "rmse_over_train_q90": float(np.sqrt(np.mean(residual**2))),
                    "mae_over_train_q90": float(np.mean(abs(residual))),
                    "rmse_native": float(scale * np.sqrt(np.mean(residual**2))),
                }
            fitted[name] = {
                "optimizer_success": bool(fit.success),
                "message": str(fit.message),
                "evaluations": int(fit.nfev),
                "bound_hits": fit.active_mask.tolist(),
                "depth_native": float(fit.x[0] * scale),
                "parameters": fit.x.tolist(),
                **metrics,
            }
        results[frame] = {
            "sources": {str(path): sha256(path) for path in (cloud_path, label_path)},
            "component_count": int(count),
            "plane_fit_points": len(road),
            "finite_component_points": len(points),
            "fit_training_points": len(fit_indices),
            "training_depth_q90_native": scale,
            "depth_sign": sign,
            "reference_plane": {"origin_native": origin.tolist(), "normal": normal.tolist()},
            "outline_center_log_axes_angle": outline.tolist(),
            "outline_radial_rms": radial_error,
            "annotated_points_outside_ellipse_fraction": float(np.mean(u * u + v * v >= 1)),
            "models": fitted,
        }
        stride = max(1, len(points) // 15000)
        projected = project(points)[::stride]
        lo, hi = np.quantile(depth, [0.02, 0.98])
        for column, (title, values) in enumerate((("Observed", depth), *predictions.items())):
            scatter = plots[row, column].scatter(
                projected[:, 0],
                projected[:, 1],
                c=values[::stride],
                s=2,
                vmin=lo,
                vmax=hi,
                cmap="viridis",
            )
            plots[row, column].set(
                title=f"Frame {frame}: {title}",
                xlabel="Local u (native)",
                ylabel="Local v (native)",
            )
            plots[row, column].set_aspect("equal")
        fig.colorbar(
            scatter, ax=plots[row, :].tolist(), label="Plane-relative depth (native)", shrink=0.7
        )
        write_json(output / "partial_results.json", results)
        print(
            f"frame {frame}: "
            + ", ".join(
                f"{name} testRMSE={fit['test']['rmse_over_train_q90']:.6g}"
                for name, fit in fitted.items()
            ),
            flush=True,
        )
    fig.savefig(output / "profiles.png", dpi=160)
    plt.close(fig)
    write_json(output / "audit.json", {"protocol": protocol, "frames": results})
    lines = [
        "# 坑槽剖面：同视图空间块留出诊断",
        "",
        "点云物理单位未确认。深度以原生坐标和训练深度q90归一化值报告，不解释为毫米。",
        "同一标签轮廓、局部TLS平面和椭圆足迹下，比较固定抛物碗与可调偏心幂次碗；只用训练块拟合。",
        "",
        "|帧|对称碗训练RMSE|偏心碗训练RMSE|对称碗留块RMSE|偏心碗留块RMSE|",
        "|---|---:|---:|---:|---:|",
    ]
    for frame, result in results.items():
        models = result["models"]
        values = [
            models[name][role]["rmse_over_train_q90"]
            for role in ("train", "test")
            for name in ("symmetric_p1", "tilted_power_bowl")
        ]
        lines.append("|" + frame + "|" + "|".join(f"{value:.6f}" for value in values) + "|")
    lines += [
        "",
        "这是已知整帧轮廓下的插值诊断，额外参数可能降低拟合残差，不能证明独立道路泛化。",
        "三帧独立性、立体重建误差及Z单位未知；搜索边界属于工程假设，不能作为真实坑槽参数分布。",
        "该原型不模拟不规则坑缘、多层坑壁、崩落骨料或遮挡。默认生成模型未改。",
        "优化器状态、边界命中、样本数、源文件哈希和全部原生参数见audit.json。",
    ]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
