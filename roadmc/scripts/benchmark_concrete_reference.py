"""Freeze, then run a bounded KIT concrete ELA forward-model benchmark.

Run ``--freeze`` first, then ``--run`` against the same output directory.
No source dataset or production configuration is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from roadmc.data.synthetic.concrete_reference import (
    aggregate_descriptors,
    compare_descriptors,
    describe_ring,
    load_ela_csv,
    preprocess_ring,
    sample_rings,
)
from roadmc.data.synthetic.config import MicroTextureConfig
from roadmc.data.synthetic.generator import _draw_mpd_target_m
from roadmc.data.synthetic.primitives import generate_road_surface
from roadmc.data.synthetic.procedural_texture import generate_texture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def freeze(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("protocol.json exists; frozen protocol must not be overwritten")
    inputs = sorted(args.docs.glob("kit_csv_A[56]_L[12]_M[1-5]_[LR].csv"))
    if len(inputs) != 32:
        raise ValueError(f"expected exactly32 documented ELA CSVs, found{len(inputs)}")
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / "data/synthetic/concrete_reference.py",
             root / "data/synthetic/primitives.py", root / "data/synthetic/generator.py",
             root / "data/synthetic/config.py", root / "data/synthetic/procedural_texture.py"]
    evidence = args.docs.parent / "paper_followup"
    protocol = {
        "schema": "roadmc.kit-concrete-ring-protocol.v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_paper": "10.3390/data11100255 p3 section2.3 circular0..400mm and units; p2-5 class/group definitions",
        "inputs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in inputs],
        "evidence": [{"path": str(p), "sha256": sha(p)} for p in [
            evidence / "admission_catalog.v2.json", evidence / "kit_source_groups.json",
            evidence / "paper_text.md"]],
        "code": [{"path": str(p), "sha256": sha(p)} for p in files],
        "grouping": "L1 development only; L2 already exposed internal location benchmark, neither unseen nor independent-road test; equal weight M cluster, then L/R half, then paired Prof1/2",
        "classes": ["exposed_aggregate", "drag_finish", "broom_finish"],
        "sample_spacing_m": 0.0005, "samples_per_ring": 800,
        "circumference_m": 0.4,
        "synthetic_domain_m": [0.2, 0.3],
        "synthetic_centers_m": [[0.1, 0.075], [0.1, 0.225]],
        "synthetic_phase_rad": 0.0,
        "high_resolution_grid_m": 0.00025,
        "default_grid_m": 0.005,
        "training_seeds": [1601, 1602, 1603, 1604],
        "evaluation_seeds": [2601, 2602, 2603, 2604],
        "preprocessing": "constant+first cosine/sine harmonic removal (planar ring); optional sharp periodic FFT diagnostic passband, applied equally. Not ISO filter replication",
        "primary_band_cycles_m": [20.0, 500.0],
        "sensitivity_bands": {"unbanded": None, "common20to80": [20.0, 80.0]},
        "descriptors": "profile RMS; standardized quantiles1..99%, skew/Pearson kurtosis; Hann Welch400 samples50% overlap/no extra detrend; circular ACF0..50mm",
        "candidate_family": "existing procedural_texture.generate_texture isotropic Gaussian2D fields; forward-sample rings; never reuse1D PSD as2D radial PSD",
        "candidate_grid": {"band_lo_cycles_m": [0.0, 20.0, 50.0, 100.0],
                           "hurst": [0.1, 0.5, 0.9], "band_hi_cycles_m": 800.0},
        "selection": "12 candidates per class, L1 only. Min normalized1D PSD log10 MAE + circular ACF MAE (explicit equal coefficient1); stable ties grid order. Then scalar RMS calibration by mean L1 profile RMS / train synthetic RMS. No L2 influence, no marginal/anisotropy fit",
        "baseline": "current production _draw_mpd_target_m concrete defaults + generate_road_surface two-segment default H/break; zero slopes for texture-only study; small-domain finegrid is explicit diagnostic override; also native5mm small domain and default7x5m context sensitivity",
        "exclusions": "No ETD/SPT fitting, no PLY, no distress labels, no standard certification, no independent-road or unfiltered-surface calibration claim",
        "scientific_status": "not_calibrated / not_evaluated",
    }
    write_json(out / "protocol.json", protocol)
    print(json.dumps({"phase": "frozen", "protocol_sha256": sha(out / "protocol.json")}), flush=True)


def process_pool(profiles, band, dx=0.0005):
    return aggregate_descriptors([describe_ring(preprocess_ring(p, dx, band), dx)
                                  for p in profiles])


def measured_groups(protocol):
    records, arrays = [], {}
    for entry in protocol["inputs"]:
        x, z, meta = load_ela_csv(entry["path"])
        if z.shape != (2, 800) or not np.allclose(np.diff(x), 0.0005, atol=1e-12):
            raise ValueError("source profile differs from frozen800 samples/0.5mm spacing")
        key = Path(entry["path"]).stem.removeprefix("kit_csv_")
        arrays[key] = z
        paired = z[0] - z[1]
        records.append({"id": key, **meta, "source_sha256": entry["sha256"],
                        "paired_raw_difference_rms_um": float(1e6 * np.sqrt(np.mean(paired ** 2))),
                        "paired_raw_correlation": float(np.corrcoef(z)[0, 1]),
                        "closure_jump_um_each": (1e6 * np.abs(z[:, -1] - z[:, 0])).tolist()})
    return records, arrays


def real_summary(records, arrays, finish, location, band):
    clusters = sorted({r["cluster"] for r in records
                       if r["finish"] == finish and r["location"] == location})
    by_cluster = []
    for cluster in clusters:
        halves = [process_pool(arrays[r["id"]], band) for r in records
                  if r["cluster"] == cluster]
        by_cluster.append(aggregate_descriptors(halves))
    return aggregate_descriptors(by_cluster)


def production_rings(protocol, seed, grid, full_context=False):
    micro = MicroTextureConfig()
    rng = np.random.default_rng(seed)
    rms, audit = _draw_mpd_target_m(rng, micro, "concrete")
    surface_seed = int(rng.integers(0, 2 ** 31))
    width, length = (7.0, 5.0) if full_context else protocol["synthetic_domain_m"]
    points, _normals = generate_road_surface(
        width, length, grid, roughness_class="A", seed=surface_seed,
        texture_rms=rms, texture_hurst=micro.hurst,
        texture_h_macro=micro.h_macro, texture_h_micro=micro.h_micro,
        texture_break_wavelength_m=micro.break_wavelength_m,
        macro_spectrum_mode="two-segment", crossfall=0, longitudinal_grade=0,
    )
    x, y = np.arange(0, width, grid), np.arange(0, length, grid)
    z = points[:, 2].reshape(len(x), len(y))
    centers = ([(3.5, 2.425), (3.5, 2.575)] if full_context
               else protocol["synthetic_centers_m"])
    rings = sample_rings(x, y, z, centers)
    return rings, {"seed": seed, "surface_seed": surface_seed, "grid_m": grid,
                   "domain_m": [width, length], "mpd_draw": audit,
                   "grid_std_mm": float(z.std() * 1000)}


def candidate_rings(protocol, settings, seeds):
    grid = protocol["high_resolution_grid_m"]
    width, length = protocol["synthetic_domain_m"]
    x, y = np.arange(0, width, grid), np.arange(0, length, grid)
    result = []
    for seed in seeds:
        z, _meta = generate_texture(x, y, settings, seed)
        result.extend(sample_rings(x, y, z, protocol["synthetic_centers_m"]))
    return np.stack(result)


def plot_results(out, summaries, candidates):
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    for col, (finish, summary) in enumerate(summaries.items()):
        for key, label, color in [("L1", "L1 real", "tab:blue"),
                                  ("L2", "L2 real internal", "black"),
                                  ("production_fine", "Production finegrid", "tab:orange"),
                                  ("candidate", "L1-fitted opt-in", "tab:green")]:
            d = summary[key]
            axes[0, col].plot(d["quantile_probabilities"], d["standardized_quantiles"],
                              label=label, color=color)
            axes[1, col].semilogy(d["frequency_cycles_m"], d["psd_m3"], color=color)
            axes[2, col].plot(np.array(d["acf_lag_m"]) * 1000, d["circular_acf"], color=color)
        axes[0, col].set(title=finish.replace("_", " "), xlabel="Quantile probability", ylabel="Height / profile RMS")
        axes[1, col].set(xlim=(20, 500), xlabel="1D path frequency (cycles/m)", ylabel="1D PSD (m^3)")
        axes[2, col].set(xlabel="Arc lag (mm)", ylabel="Circular ACF")
        labels = ["L1", "L2", "Production", "Candidate"]
        axes[3, col].bar(labels, [summary[k]["rms_m"] * 1000 for k in
                                  ["L1", "L2", "production_fine", "candidate"]],
                          color=["tab:blue", "black", "tab:orange", "tab:green"])
        axes[3, col].set(ylabel="Profile RMS (mm)")
        for row in range(4):
            axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("KIT circular profiles: shared 20-500 cycles/m diagnostic band\nL2 is an exposed internal location benchmark; Gaussian isotropic candidates")
    fig.savefig(out / "concrete_ring_comparison.png", dpi=160)
    plt.close(fig)


def run(args):
    out = args.output
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if (out / "results.json").exists():
        raise FileExistsError("results.json exists; do not overwrite prior run")
    for entry in protocol["inputs"] + protocol["code"] + protocol["evidence"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"frozen input/code changed: {entry['path']}")
    start = time.monotonic()
    records, arrays = measured_groups(protocol)
    bands = {"primary": protocol["primary_band_cycles_m"], **protocol["sensitivity_bands"]}
    reference = {f: {loc: {b: real_summary(records, arrays, f, loc, band)
                           for b, band in bands.items()} for loc in ["L1", "L2"]}
                 for f in protocol["classes"]}
    banks, baseline_audit = {}, []
    for kind, grid, full in [("production_fine", .00025, False),
                              ("production_native", .005, False),
                              ("production_default_context", .005, True)]:
        profiles = []
        for seed in protocol["evaluation_seeds"]:
            rings, audit = production_rings(protocol, seed, grid, full)
            profiles.extend(rings)
            baseline_audit.append({"kind": kind, **audit})
        banks[kind] = np.stack(profiles)
    print("Production ring banks complete", flush=True)
    candidate_bank, search = {}, []
    grid = protocol["candidate_grid"]
    for lo in grid["band_lo_cycles_m"]:
        for hurst in grid["hurst"]:
            settings = {"rms_m": .001, "hurst": hurst, "anisotropy_ratio": 1.0,
                        "orientation_deg": 0.0, "band_lo_cycles_m": lo,
                        "band_hi_cycles_m": grid["band_hi_cycles_m"],
                        "distribution": "gaussian", "material_id": "concrete"}
            profiles = candidate_rings(protocol, settings, protocol["training_seeds"])
            desc = process_pool(profiles, bands["primary"])
            key = f"lo{lo:g}_H{hurst:g}"
            candidate_bank[key] = (profiles, settings, desc)
            for finish in protocol["classes"]:
                error = compare_descriptors(reference[finish]["L1"]["primary"], desc)
                search.append({"finish": finish, "candidate": key, "settings_unit_rms": settings,
                               "selection_loss": error["normalized_psd_log10_mae"] + error["acf_mae_0_to_50mm"],
                               "errors_before_amplitude_fit": error})
            print(f"Forward candidate {key} complete", flush=True)
    selected, summaries, comparisons = {}, {}, {}
    for finish in protocol["classes"]:
        best = min([r for r in search if r["finish"] == finish], key=lambda r: r["selection_loss"])
        profiles, settings, desc = candidate_bank[best["candidate"]]
        scale = reference[finish]["L1"]["primary"]["rms_m"] / desc["rms_m"]
        settings = {**settings, "rms_m": settings["rms_m"] * scale}
        evaluation = candidate_rings(protocol, settings, protocol["evaluation_seeds"])
        banks[f"candidate_{finish}"] = evaluation
        selected[finish] = {"selected_grid_id": best["candidate"], "settings": settings,
                            "training_loss": best["selection_loss"],
                            "amplitude_scale": scale, "training_location": "L1",
                            "purpose": "explicit opt-in research texture, isotropy/shape not validated"}
        write_json(out / f"candidate_{finish}.json", settings)
        comparisons[finish] = {}
        for loc in ["L1", "L2"]:
            comparisons[finish][loc] = {}
            for b, band in bands.items():
                comparisons[finish][loc][b] = {}
                for kind in ["production_fine", "production_native", "production_default_context", f"candidate_{finish}"]:
                    desc_kind = process_pool(banks[kind], band)
                    scoreband = band if band is not None else [5.0, 1000.0]
                    comparisons[finish][loc][b][kind] = compare_descriptors(
                        reference[finish][loc][b], desc_kind, scoreband)
        summaries[finish] = {"L1": reference[finish]["L1"]["primary"],
                             "L2": reference[finish]["L2"]["primary"],
                             "production_fine": process_pool(banks["production_fine"], bands["primary"]),
                             "candidate": process_pool(evaluation, bands["primary"])}
    np.savez_compressed(out / "profiles_si.npz", **{f"real_{k}": v for k, v in arrays.items()}, **banks,
                        **{f"train_{k}": v[0] for k, v in candidate_bank.items()})
    results = {"schema": "roadmc.kit-concrete-ring-results.v1",
               "protocol_sha256": sha(out / "protocol.json"), "source_records": records,
               "real_descriptors": reference, "baseline_audit": baseline_audit,
               "finite_search": search, "selected": selected, "comparisons": comparisons,
               "primary_descriptors": summaries,
               "elapsed_s": time.monotonic() - start,
               "runtime": {"python": sys.version, "numpy": np.__version__,
                           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                           "platform": platform.platform()},
               "scientific_status": "not_calibrated / not_evaluated"}
    write_json(out / "results.json", results)
    plot_results(out, summaries, selected)
    report = ["# KIT 正常混凝土环剖面实际对照", "",
              "【实测】本轮读取32份有单位CSV，每份2条800点环剖面，间距0.5mm；合计64条配对剖面。两条公路、四地点、16个M测区；配对剖面和左右半区不作为独立道路。L1仅用于开发，L2是已曝光的内部地点benchmark。",
              "", "协议在运行前写入protocol.json；代码和32个输入逐项SHA256核对。全部数字可追溯results.json。图中主对照统一去平面环谐波并使用20–500 cycles/m诊断带；另存不加频带和20–80 cycles/m敏感性。该变换并非ISO滤波复现。",
              "", "【论文来源】data-11-00255 p3 §2.3说明两次400mm圆环、mm/µm单位、作者ISO2019滤波；p2–5给出表面类型与测点分组。A5报告2004标头仍有差异，已滤波数据不能恢复原始高频。ETD及砂铺直径未参与拟合。",
              "", "【采样】从真实生产_draw_mpd_target_m与generate_road_surface路径取纹理场，采用800点/400mm圆环双线性采样；环半径400/(2π)mm。相位与真实环准确位置未知，合成左右环中心只是固定几何选择。曲线是沿圆环弧长的一维PSD/ACF，不能解释为二维径向PSD，也不能从相位推断行车/横向纹理各向异性。",
              "", "【分辨率】主基线显式将生产网格改为0.25mm、域20×30cm，保留生产默认谱形/幅值抽样；它是研究诊断，并非默认5mm输出。默认5mm仅有100cycles/m轴向Nyquist；20–500主带中其高频为双线性重采样产物，不能当作有物理分辨能力；应看20–80共同带。默认7×5m域对照另存，揭示小域谱支持的影响。生产默认参数均未修改。",
              "", "【有限拟合】每类仅在L1，从12个既有procedural_texture二维各向同性高斯模型前向生成并采圆环，按归一化一维PSD对数MAE+ACF MAE选型；再按L1平均环RMS定幅。没有将一维PSD写进二维径向谱。未拟合边际分布和方向，因此以下边际退步必须保留。候选JSON是显式opt-in研究配置，不是已校准道路模型。",
              "", "|表面|L2实测RMS mm|细网格生产RMS mm|候选RMS mm|生产→候选归一化PSD log10 MAE|生产→候选ACF MAE|生产→候选标准化分位MAE|", "|---|---:|---:|---:|---:|---:|---:|"]
    for finish in protocol["classes"]:
        s = summaries[finish]
        a = comparisons[finish]["L2"]["primary"]["production_fine"]
        b = comparisons[finish]["L2"]["primary"][f"candidate_{finish}"]
        report.append(f"|{finish}|{s['L2']['rms_m']*1000:.6f}|{s['production_fine']['rms_m']*1000:.6f}|{s['candidate']['rms_m']*1000:.6f}|{a['normalized_psd_log10_mae']:.6f} → {b['normalized_psd_log10_mae']:.6f}|{a['acf_mae_0_to_50mm']:.6f} → {b['acf_mae_0_to_50mm']:.6f}|{a['standardized_quantile_mae']:.6f} → {b['standardized_quantile_mae']:.6f}|")
    report += ["", "![共同带描述对照](concrete_ring_comparison.png)", "",
               "以上是有处理链限制的描述性域差距和内部有限开发结果，没有独立道路误差、显著性或置信区间主张。圆环剖面不足以识别二维方向/纹理拓扑；正常表面不等于水泥病害真值。科学状态保持not_calibrated / not_evaluated。",
               "", "复现：先--freeze再--run；同一结果目录禁止覆盖。源数据只读，输入/代码/evidence哈希在protocol.json，产物哈希和退出码在completion_receipt.json。"]
    (out / "KIT正常混凝土实际对照.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    artifacts = [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file() and p.name != "completion_receipt.json"]
    write_json(out / "completion_receipt.json", {"exit_code": 0, "artifacts": artifacts,
               "inputs_verified_after": all(sha(e["path"]) == e["sha256"] for e in protocol["inputs"]),
               "elapsed_s": time.monotonic() - start})
    print(json.dumps({"phase": "complete", "exit_code": 0, "output": str(out),
                      "elapsed_s": time.monotonic() - start}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze(args) if args.freeze else run(args)


if __name__ == "__main__":
    main()
