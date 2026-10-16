"""Controlled source-grid/line-profiler experiment, not a real-road accuracy test.

Example: python -m roadmc.scripts.diagnose_sampling_resolution --output-dir D:/trial
The saved 0.5 mm SurfaceTile is the master realization. Coarser tiles are strided
views of this realization, deliberately without a geometry low-pass: aliasing is
a negative control, not a claimed physical sensor model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from roadmc.data.synthetic.config import (
    DEFAULT_MAX_SURFACE_MEMORY_MIB,
    DEFAULT_MAX_SURFACE_POINTS,
    SURFACE_PEAK_BYTES_PER_POINT,
    LineProfilerConfig,
)
from roadmc.data.synthetic.line_profiler import simulate_line_profiler
from roadmc.data.synthetic.surface_replay import replay_surface
from roadmc.data.synthetic.surface_tile import SurfaceTile

MASTER_SPACING_M = 0.0005
SOURCE_STRIDES = (10, 4, 2, 1)


def make_master(extent_m: float, seed: int) -> tuple[SurfaceTile, dict]:
    """Draw phase once; all resolutions share exactly the sampled realization."""
    if not np.isfinite(extent_m) or extent_m < 0.04:
        raise ValueError("extent_m must be finite and >= 0.04")
    max_points = min(
        DEFAULT_MAX_SURFACE_POINTS,
        DEFAULT_MAX_SURFACE_MEMORY_MIB * 1024**2 / SURFACE_PEAK_BYTES_PER_POINT,
    )
    if extent_m > int(np.sqrt(max_points)) * MASTER_SPACING_M:
        raise ValueError("requested master exceeds the existing point or estimated memory budget")
    n = int(np.ceil(extent_m / MASTER_SPACING_M))
    points = n * n
    estimated_mib = points * SURFACE_PEAK_BYTES_PER_POINT / 1024**2
    if points > DEFAULT_MAX_SURFACE_POINTS or estimated_mib > DEFAULT_MAX_SURFACE_MEMORY_MIB:
        raise ValueError("requested master exceeds the existing point or estimated memory budget")
    axis = np.arange(n, dtype=np.float64) * MASTER_SPACING_M
    phase = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 3)
    x, y = axis[:, None], axis[None, :]
    design = np.broadcast_to(0.02 * x + 0.005 * y, (n, n)).copy()
    roughness = np.broadcast_to(0.0005 * np.sin(2 * np.pi * 4 * y), (n, n)).copy()
    texture = (
        0.0004 * np.sin(2 * np.pi * 250 * x + phase[0])
        + 0.0002 * np.sin(2 * np.pi * 150 * y + phase[1])
        + 0.0001 * np.sin(2 * np.pi * 40 * (x + y) + phase[2])
    )
    metadata = {
        "domain": "synthetic",
        "calibration_status": "not_calibrated",
        "source": "analytic Fourier engineering fixture, no real texture copied",
        "seed": seed,
        "phases_rad": phase.tolist(),
        "extent_m": extent_m,
        "master_spacing_m": MASTER_SPACING_M,
        "source_points": points,
        "estimated_surface_peak_mib": estimated_mib,
        "memory_note": "320 B/point model only; full capture and Python overhead are additional",
        "units": "m",
        "texture_frequencies_cycles_m": [250, 150, [40, 40]],
    }
    plan = {
        "schema_version": 1,
        "material": "asphalt",
        "road_id": "sampling-resolution-fixture",
        "provenance": {"source": "analytic engineering fixture; no empirical geometric prior"},
        "hypothesis": {"status": "internal mechanism test, not road calibration"},
        "events": [
            {
                "instance_id": "fixed-pothole",
                "type": "pothole",
                "seed": 17,
                "label": None,
                "parameters": {
                    "center_x_m": extent_m / 2,
                    "center_y_m": extent_m / 2,
                    "radius_m": 0.18 * extent_m,
                    "depth_m": 0.02 * extent_m,
                    "exposed_rms_m": 0.0,
                },
            }
        ],
    }
    return SurfaceTile(axis, axis.copy(), design, roughness, texture, metadata), plan


def analytic_height(x: np.ndarray, y: np.ndarray, metadata: dict, plan: dict) -> np.ndarray:
    """Evaluate the fixture at actual output XY, independently of interpolation."""
    phase = metadata["phases_rad"]
    low = 0.02 * x + 0.005 * y + 0.0005 * np.sin(2 * np.pi * 4 * y)
    texture = (
        0.0004 * np.sin(2 * np.pi * 250 * x + phase[0])
        + 0.0002 * np.sin(2 * np.pi * 150 * y + phase[1])
        + 0.0001 * np.sin(2 * np.pi * 40 * (x + y) + phase[2])
    )
    p = plan["events"][0]["parameters"]
    rho = np.hypot(x - p["center_x_m"], y - p["center_y_m"]) / p["radius_m"]
    inside = rho < 1.0
    return low + np.where(inside, -p["depth_m"] * np.maximum(0, 1 - rho**2), texture)


def run_experiment(output: Path, *, extent_m: float = 1.0, seed: int = 0) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refuse to overwrite a non-empty experiment: {output}")
    hashes_before = _dependency_hashes()
    master, plan = make_master(extent_m, seed)
    output.mkdir(parents=True, exist_ok=True)
    master.save(output / "master_surface.npz")
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    common_end = np.floor(master.x[-1] / 0.01) * 0.01
    profiles = {}
    summaries = []
    # Bound source support identically across grids; do not infer a common scene
    # from repeated RNG calls at different array sizes.
    for stride in SOURCE_STRIDES:
        start = time.perf_counter()
        tile = SurfaceTile(
            master.x[::stride],
            master.y[::stride],
            master.design[::stride, ::stride],
            master.roughness[::stride, ::stride],
            master.texture[::stride, ::stride],
            {**master.metadata, "source_stride": stride},
        )
        surface = replay_surface(tile, plan, observation="off")
        spacing = MASTER_SPACING_M * stride
        for profile_name, step_m in (("existing_1x5mm", 0.005), ("experimental_1x1mm", 0.001)):
            cfg = LineProfilerConfig(lateral_res_m=0.001, step_m=step_m)
            for noise in (False, True):
                observed = simulate_line_profiler(
                    surface.points,
                    surface.normals,
                    surface.labels,
                    grid_shape=tile.design.shape,
                    grid_res=spacing,
                    origin_xy=(0.0, 0.0),
                    cfg=cfg,
                    rng=np.random.default_rng(2026),
                    noise_enabled=noise,
                )
                pts = observed.points
                common = (pts[:, 0] <= common_end + 1e-12) & (pts[:, 1] <= common_end + 1e-12)
                pts = pts[common]
                expected = analytic_height(pts[:, 0], pts[:, 1], master.metadata, plan)
                errors = pts[:, 2] - expected
                norm_error = np.abs(np.linalg.norm(observed.normals, axis=1) - 1)
                row = {
                    "source_spacing_m": spacing,
                    "source_shape": list(tile.design.shape),
                    "source_axis_nyquist_cycles_m": 0.5 / spacing,
                    "observation_profile": profile_name,
                    "observation_config": asdict(cfg),
                    "output_axis_nyquist_cycles_m": {"x": 500.0, "y": 0.5 / step_m},
                    "noise_enabled": noise,
                    "observation_seed": 2026,
                    "common_bounds_m": [0.0, float(common_end)],
                    "retained_points_all": int(len(observed.points)),
                    "retained_points_common": int(len(pts)),
                    "dropout_count": observed.dropout_count,
                    "analytic_rmse_m": float(np.sqrt(np.mean(errors**2))),
                    "analytic_max_abs_error_m": float(np.max(np.abs(errors))),
                    "finite_points": bool(np.isfinite(observed.points).all()),
                    "normal_max_unit_error": float(norm_error.max()),
                    "plan_hash": surface.metadata["plan_hash"],
                    "expected_crack_or_pothole_detection_accuracy": None,
                }
                key = f"source{spacing * 1000:g}mm_{profile_name}_noise{int(noise)}"
                # First scanline is outside the pothole and reveals texture aliasing.
                first = np.isclose(pts[:, 1], 0.0, rtol=0, atol=1e-12)
                line = pts[first]
                profiles[key + "_x_m"] = line[:, 0].copy()
                profiles[key + "_z_m"] = line[:, 2].copy()
                profiles[key + "_expected_z_m"] = expected[first].copy()
                row["first_line_dominant_x_fft_bin_cycles_m"] = None
                if not noise:
                    centered = line[:, 2] - (0.02 * line[:, 0])
                    power = np.abs(np.fft.rfft(centered - centered.mean())) ** 2
                    freq = np.fft.rfftfreq(len(centered), d=cfg.lateral_res_m)
                    row["first_line_dominant_x_fft_bin_cycles_m"] = float(
                        freq[1 + np.argmax(power[1:])]
                    )
                first_column = np.isclose(pts[:, 0], 0.0, rtol=0, atol=1e-12)
                column = pts[first_column]
                column_y = column[:, 1]
                profiles[key + "_column_y_m"] = column_y.copy()
                profiles[key + "_column_z_m"] = column[:, 2].copy()
                longitudinal = column[:, 2] - (
                    0.005 * column_y + 0.0005 * np.sin(2 * np.pi * 4 * column_y)
                )
                # Noise-induced missing samples make an ordinary FFT invalid.
                # Only the intact, noise-off line is used for the alias check.
                row["first_column_dominant_y_fft_bin_cycles_m"] = None
                if not noise:
                    y_power = np.abs(np.fft.rfft(longitudinal - longitudinal.mean())) ** 2
                    y_freq = np.fft.rfftfreq(len(longitudinal), d=step_m)
                    row["first_column_dominant_y_fft_bin_cycles_m"] = float(
                        y_freq[1 + np.argmax(y_power[1:])]
                    )
                row["base_texture_y_frequencies_within_output_nyquist"] = step_m <= 1 / (2 * 150)
                row["frequency_diagnostic_note"] = (
                    "not evaluated: dropout makes retained coordinates nonuniform"
                    if noise
                    else "FFT of intact uniform noise-off scanline"
                )
                summaries.append(row)
                del observed, pts, expected, errors
        print(
            f"source {spacing * 1000:g} mm finished in {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        del surface, tile
        gc.collect()
    np.savez_compressed(output / "profiles.npz", **profiles)
    preview_stride = max(1, len(master.x) // 300)
    x, y = np.meshgrid(master.x[::preview_stride], master.y[::preview_stride], indexing="ij")
    np.savez_compressed(
        output / "height_preview.npz",
        x_m=x,
        y_m=y,
        base_height_m=master.height[::preview_stride, ::preview_stride],
        damaged_height_m=analytic_height(x, y, master.metadata, plan),
    )
    hashes_after = _dependency_hashes()
    audit = {
        "status": "internal_mechanism_test",
        "real_domain_validated": False,
        "source_hashes": hashes_before,
        "source_hashes_after": hashes_after,
        "source_files_stable_during_run": hashes_before == hashes_after,
        "master": master.metadata,
        "rows": summaries,
        "protocol": {
            "source_derivation": "strided samples of one saved master; no RNG redraw",
            "reference": "continuous analytic formula at each actual observed XY",
            "anti_aliasing": "none on geometry; coarse-grid aliasing is an intentional negative control",
            "common_support": "same XY bounds for all error summaries",
            "filter_policy": "texture calibration uses a separate declared analysis filter",
            "noise": "same explicit observation seed; on and off are separate rows",
            "fft_policy": "frequency diagnostics use intact noise-off rows only; never FFT a gapped scan as uniform",
        },
        "limitations": [
            "Engineering sine amplitudes, phases and pothole dimensions, not measured road distributions",
            "Fine-grid success is not sensor accuracy or independent road validation",
            "1x5 mm profile cannot resolve all longitudinal content of this fixture",
            "No physical beam-footprint or occlusion calibration; no hidden geometry smoothing",
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    _render(output, summaries, profiles)
    lines = [
        "# 网格与线激光采样对照",
        "",
        "本试验使用保存的同一0.5毫米母场和同一病害计划。粗网格直接抽取母场；没有对病害几何低通，因此保留混叠反例。",
        "解析公式在实际输出XY处独立求值。表中误差属于内部机制检查，不能解释为真实设备精度或真实域差距。",
        "采样点高度误差为零只表示这些离散坐标求值正确，不代表连续纹理重建无损。纵向150 cycles/m分量在5毫米步距下会混叠，另列纵向FFT核查。",
        "",
        "|源格距mm|输出步距mm(X/Y)|噪声|共同范围点数|高度RMSE mm|最大误差mm|",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"|{row['source_spacing_m'] * 1000:g}|1/{row['observation_config']['step_m'] * 1000:g}"
            f"|{'开' if row['noise_enabled'] else '关'}|{row['retained_points_common']}"
            f"|{row['analytic_rmse_m'] * 1000:.6g}|{row['analytic_max_abs_error_m'] * 1000:.6g}|"
        )
    lines += [
        "",
        "实际频带、坐标支持、源文件hash、种子、法线误差见audit.json。",
        "全场保持生产默认设置；本目录的细网格与1×1毫米观测仅为显式实验配置。",
    ]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def _dependency_hashes() -> dict[str, str]:
    synthetic = Path(__file__).parents[1] / "data" / "synthetic"
    files = [
        synthetic / name
        for name in (
            "line_profiler.py",
            "surface_replay.py",
            "surface_tile.py",
            "config.py",
        )
    ] + [Path(__file__)]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _render(output: Path, rows: list[dict], profiles: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for profile in ("existing_1x5mm", "experimental_1x1mm"):
        subset = [r for r in rows if r["observation_profile"] == profile and not r["noise_enabled"]]
        axes[0].plot(
            [r["source_spacing_m"] * 1000 for r in subset],
            [r["analytic_rmse_m"] * 1000 for r in subset],
            "o-",
            label=profile,
        )
    axes[0].set(
        xlabel="Source spacing (mm)",
        ylabel="Analytic height RMSE (mm)",
        title="Noise off; identical physical scene",
    )
    axes[0].legend(fontsize=8)
    for spacing in (5, 0.5):
        key = f"source{spacing:g}mm_experimental_1x1mm_noise0"
        x = profiles[key + "_x_m"]
        keep = x <= 0.06
        z = profiles[key + "_z_m"] - 0.02 * x
        axes[1].plot(x[keep] * 1000, z[keep] * 1000, label=f"source {spacing:g} mm")
    axes[1].set(
        xlabel="X (mm)",
        ylabel="Texture + fixed offset (mm)",
        title="First scanline: coarse-grid aliasing control",
    )
    axes[1].legend(fontsize=8)
    for profile in ("existing_1x5mm", "experimental_1x1mm"):
        key = f"source0.5mm_{profile}_noise0"
        y = profiles[key + "_column_y_m"]
        z = profiles[key + "_column_z_m"] - (0.005 * y + 0.0005 * np.sin(2 * np.pi * 4 * y))
        d = y[1] - y[0]
        power = np.abs(np.fft.rfft(z - z.mean())) ** 2
        freq = np.fft.rfftfreq(len(y), d=d)
        axes[2].plot(freq, power / power.max(), label=profile)
    axes[2].set(
        xlabel="Y frequency (cycles/m)",
        ylabel="Peak-normalized power",
        title="Dense source: longitudinal sampling still matters",
        xlim=(0, 250),
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "resolution_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extent-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_experiment(args.output_dir, extent_m=args.extent_m, seed=args.seed)
    print(json.dumps({"output": str(args.output_dir), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
