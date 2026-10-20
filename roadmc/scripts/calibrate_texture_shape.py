"""Run a frozen train/holdout shape calibration on Royal Society texture CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadmc.data.synthetic.procedural_texture import generate_texture
from roadmc.data.synthetic.texture_shape_calibration import (
    canonical_array_sha256,
    deduplicate_manifest,
    descriptor_scores,
    directional_marginal_density,
    moments,
    preprocess_shape,
    radial_cosine_gain,
    remove_hash_matches_to_excluded_tests,
    select_setting,
    shape_descriptors,
)

REPO = Path(__file__).resolve().parents[2]
CLASSES = ("DAC", "MS", "OGFC", "SMA")
SEEDS = tuple(range(5))
H_VALUES = (0.05, 0.2, 0.4, 0.6, 0.8)
BANDS = (30.0, 100.0, 250.0, 400.0)
N = 181
DX = DY = 0.0005
RMS_M = 0.0004  # Engineering assumption; only shape is evaluated.
BASE = {
    "rms_m": RMS_M,
    "anisotropy_ratio": 1.0,
    "orientation_deg": 0.0,
    "band_lo_cycles_m": 0.0,
    "distribution": "gaussian",
    "material_id": "asphalt",
}
PREFIX = "rsta-2022-0166-File001/Original_Pavement_Data/"
MEMBER_RE = re.compile(r"^" + re.escape(PREFIX) + r"(train|test)/(DAC|MS|OGFC|SMA)/(\d+)\.csv$")


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_source(archive_path: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    records, arrays = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            match = MEMBER_RE.match(member)
            if not match:
                continue
            split, material, number = match.groups()
            raw_bytes = archive.read(member)
            values = np.loadtxt(
                io.StringIO(raw_bytes.decode("utf-8-sig")), delimiter=",", dtype=np.float64
            )
            record = {
                "member": member,
                "class": material,
                "split": split,
                "index": int(number),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "array_sha256": canonical_array_sha256(values),
                "bytes": len(raw_bytes),
            }
            if values.shape != (N, N) or not np.isfinite(values).all():
                record["pre_exclusion_reason"] = "invalid_shape_or_nonfinite"
            elif split == "test" and int(number) in (0, 1):
                record["pre_exclusion_reason"] = "previously_explored_test_0_or_1"
            records.append(record)
            arrays[member] = values
    return records, arrays


def standardize_for_metrics(values: np.ndarray) -> np.ndarray:
    # CSV storage is [y,x]; generator is [x,y]. Keep all calculations x-major.
    return preprocess_shape(values.T, DX, DY)


def make_descriptor(field: np.ndarray) -> dict:
    return shape_descriptors(field, DX, DY)


def directional_density(field: np.ndarray, axis: int) -> np.ndarray:
    return directional_marginal_density(field, axis, DX, DY, fmax=500.0)


def evaluate(
    real: np.ndarray,
    syn: np.ndarray,
    real_desc: dict | None = None,
    syn_desc: dict | None = None,
    real_mom: dict | None = None,
    syn_mom: dict | None = None,
    real_sorted: np.ndarray | None = None,
    syn_sorted: np.ndarray | None = None,
) -> dict:
    dr = real_desc if real_desc is not None else make_descriptor(real)
    ds = syn_desc if syn_desc is not None else make_descriptor(syn)
    shape = descriptor_scores(dr, ds)
    rm = real_mom if real_mom is not None else moments(real)
    sm = syn_mom if syn_mom is not None else moments(syn)
    real_order = real_sorted if real_sorted is not None else np.sort(real.ravel())
    syn_order = syn_sorted if syn_sorted is not None else np.sort(syn.ravel())
    return {
        **shape,
        "height_w1_standardized": float(np.mean(np.abs(real_order - syn_order))),
        "skew_real": rm["skew"],
        "skew_synthetic": sm["skew"],
        "kurt_pearson_real": rm["kurt_pearson"],
        "kurt_pearson_synthetic": sm["kurt_pearson"],
        "direction_x_psd_tv": float(
            0.5 * np.abs(directional_density(real, 0) - directional_density(syn, 0)).sum()
        ),
        "direction_y_psd_tv": float(
            0.5 * np.abs(directional_density(real, 1) - directional_density(syn, 1)).sum()
        ),
    }


def mean_rows(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive_path = args.archive.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    print(f"[1/8] Reading archive: {archive_path}", flush=True)
    raw_records, arrays = load_source(archive_path)
    explored = [
        r for r in raw_records if r.get("pre_exclusion_reason") == "previously_explored_test_0_or_1"
    ]
    originally_excluded = [r for r in raw_records if "pre_exclusion_reason" in r]
    eligible = []
    for r in raw_records:
        if "pre_exclusion_reason" in r:
            continue
        eligible.append(r)
    eligible, hash_blocked = remove_hash_matches_to_excluded_tests(eligible, explored)
    accepted, duplicate_excluded = deduplicate_manifest(eligible)
    excluded = originally_excluded + hash_blocked + duplicate_excluded
    # Ensure every class has the approved five original train files.
    counts = {
        c: {
            s: sum(r["class"] == c and r["split"] == s for r in accepted) for s in ("train", "test")
        }
        for c in CLASSES
    }
    if any(counts[c]["train"] != 5 for c in CLASSES):
        raise RuntimeError(f"Expected five usable train records per class; got {counts}")

    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "shape-only calibration; split frozen before descriptor computation/scoring",
        "archive": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_bytes": archive_path.stat().st_size,
        "grid": {
            "csv_shape_yx": [N, N],
            "generator_shape_xy": [N, N],
            "dx_m": DX,
            "dy_m": DY,
            "paper_xy_spacing_m": 0.0005,
            "csv_orientation": "row-major [y,x], transposed to generator x-major",
        },
        "classes_are": "four asphalt surface classes, not four independent roads or calibrated materials",
        "train_test_counts_after_dedup": counts,
        "included": accepted,
        "excluded": excluded,
        "deduplication": "raw member SHA-256 or canonical parsed float64 array SHA-256; train retained before test; test duplicates of train excluded; duplicates within split collapsed",
        "previously_explored_exclusion": "test/0.csv and test/1.csv per class plus any test file sharing raw or canonical array SHA-256 with those files; never scored in this run",
    }
    write_json(out / "split_manifest.json", manifest)
    (out / "split_manifest.sha256").write_text(
        hashlib.sha256((out / "split_manifest.json").read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    print(f"[2/8] Frozen split: {counts}; manifest written before scoring", flush=True)

    # Process the now-frozen training set, then cache every Gaussian candidate.
    real_processed: dict[str, np.ndarray] = {}
    real_desc: dict[str, dict] = {}
    train_moments: dict[str, list[dict]] = defaultdict(list)
    for row in accepted:
        if row["split"] != "train":
            continue
        field = standardize_for_metrics(arrays[row["member"]])
        real_processed[row["member"]] = field
        real_desc[row["member"]] = make_descriptor(field)
        train_moments[row["class"]].append(moments(field))

    gaussian_cache: dict[tuple[float, float, int], tuple[np.ndarray, dict, dict]] = {}
    print(
        "[3/8] Preprocessing training files and caching candidate Gaussian fields/descriptors",
        flush=True,
    )
    gaussian_desc_cache: dict[tuple[float, float, int], dict | None] = {}
    for h in H_VALUES:
        for band in BANDS:
            cfg = dict(BASE, hurst=h, band_hi_cycles_m=band)
            for seed in SEEDS:
                try:
                    raw_field, metadata = generate_texture(
                        np.arange(N) * DX, np.arange(N) * DY, cfg, seed
                    )
                    processed = preprocess_shape(raw_field, DX, DY)
                    gaussian_cache[(h, band, seed)] = (processed, metadata, cfg)
                    gaussian_desc_cache[(h, band, seed)] = make_descriptor(processed)
                except (ValueError, FloatingPointError) as exc:
                    gaussian_cache[(h, band, seed)] = (None, {"error": str(exc)}, cfg)
                    gaussian_desc_cache[(h, band, seed)] = None

    fit_rows, fit_summary, winners = [], {}, {}
    for material in CLASSES:
        rows_for_class = [r for r in accepted if r["split"] == "train" and r["class"] == material]
        candidates = []
        for h in H_VALUES:
            for band in BANDS:
                local = []
                for r in rows_for_class:
                    ref = real_desc[r["member"]]
                    for seed in SEEDS:
                        field, meta, _ = gaussian_cache[(h, band, seed)]
                        row = {
                            "class": material,
                            "member": r["member"],
                            "seed": seed,
                            "hurst": h,
                            "band_hi_cycles_m": band,
                            "distribution": "gaussian",
                        }
                        if field is None:
                            row.update(
                                {
                                    "radial_psd_tv": None,
                                    "acf_rmse": None,
                                    "fit_score": None,
                                    "error": meta["error"],
                                }
                            )
                        else:
                            score = descriptor_scores(ref, gaussian_desc_cache[(h, band, seed)])
                            row.update(score)
                        local.append(row)
                        fit_rows.append(row)
                vals = [r["fit_score"] for r in local if r["fit_score"] is not None]
                expected_rows = len(rows_for_class) * len(SEEDS)
                candidate = {
                    "hurst": h,
                    "band_hi_cycles_m": band,
                    "mean_fit_score": float(np.mean(vals)) if len(vals) == expected_rows else None,
                    "valid_rows": len(vals),
                    "expected_rows": expected_rows,
                }
                candidates.append(candidate)
        winner = select_setting(candidates)
        winners[material] = winner
        fit_summary[material] = {"candidates": candidates, "selected": winner}
        print(
            f"  fitted {material}: H={winner['hurst']}, band_hi={winner['band_hi_cycles_m']}",
            flush=True,
        )

    # Freeze selected Gaussian spectra; only then compute per-class train moments.
    target_moments = {}
    non_gaussian_configs = {}
    for material in CLASSES:
        target = {
            key: float(np.mean([m[key] for m in train_moments[material]]))
            for key in ("skew", "kurt_pearson")
        }
        target["kurt_pearson"] = max(1.0, target["kurt_pearson"])
        target_moments[material] = {
            "equal_file_weight_mean": target,
            "train_file_moments": train_moments[material],
        }
        win = winners[material]
        cfg = dict(
            BASE,
            hurst=win["hurst"],
            band_hi_cycles_m=win["band_hi_cycles_m"],
            distribution="non_gaussian",
            shape={"skew": target["skew"], "kurt": target["kurt_pearson"]},
        )
        non_gaussian_configs[material] = cfg

    config = {
        "protocol": "frozen per-class configuration; no holdout tuning",
        "fixed_assumptions": {
            **BASE,
            "rms_m": RMS_M,
            "rms_label": "engineering assumption; excluded from shape evaluation",
            "anisotropy_ratio": 1.0,
            "orientation_deg": 0.0,
            "band_lo_cycles_m": 0.0,
        },
        "grid_search": {
            "hurst": list(H_VALUES),
            "band_hi_cycles_m": list(BANDS),
            "seeds": list(SEEDS),
            "score": "mean over each training file and seeds of radial PSD TV + xy ACF RMSE",
            "tie_break": "lowest band_hi_cycles_m then lowest H",
        },
        "analysis_filter": radial_cosine_gain((N, N), DX, DY)[1],
        "winners": winners,
        "training_target_moments": target_moments,
        "frozen_non_gaussian_configs": non_gaussian_configs,
        "legacy_config": dict(BASE, hurst=0.8, band_hi_cycles_m=30.0),
        "fit_summary": fit_summary,
        "holdout_scored_once_after_freeze": True,
    }
    write_json(out / "fitted_configurations.json", config)
    print("[5/8] Frozen configurations saved before any holdout scoring", flush=True)
    with (out / "training_fit_rows.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = [
            "class",
            "member",
            "seed",
            "hurst",
            "band_hi_cycles_m",
            "distribution",
            "radial_psd_tv",
            "acf_rmse",
            "fit_score",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fit_rows)

    # Holdout is touched only below, after manifest, winners, and configurations are saved.
    print("[6/8] Generating frozen evaluation variants", flush=True)
    variants: dict[
        tuple[str, str, int],
        tuple[np.ndarray | None, dict, dict | None, dict | None, np.ndarray | None],
    ] = {}
    gen_rows = []
    for material in CLASSES:
        win = winners[material]
        for variant, cfg in (
            ("legacy", config["legacy_config"]),
            (
                "selected_gaussian",
                dict(BASE, hurst=win["hurst"], band_hi_cycles_m=win["band_hi_cycles_m"]),
            ),
            ("selected_non_gaussian", non_gaussian_configs[material]),
        ):
            for seed in SEEDS:
                try:
                    raw_field, metadata = generate_texture(
                        np.arange(N) * DX, np.arange(N) * DY, cfg, seed
                    )
                    processed = preprocess_shape(raw_field, DX, DY)
                    variants[(material, variant, seed)] = (
                        processed,
                        metadata,
                        make_descriptor(processed),
                        moments(processed),
                        np.sort(processed.ravel()),
                    )
                    gen_rows.append(
                        {
                            "class": material,
                            "variant": variant,
                            "seed": seed,
                            "settings": cfg,
                            "status": "ok",
                            "actual_output_moments": variants[(material, variant, seed)][3],
                            "generator_metadata": metadata,
                        }
                    )
                except (ValueError, FloatingPointError) as exc:
                    variants[(material, variant, seed)] = (
                        None,
                        {"error": str(exc)},
                        None,
                        None,
                        None,
                    )
                    gen_rows.append(
                        {
                            "class": material,
                            "variant": variant,
                            "seed": seed,
                            "settings": cfg,
                            "status": "failed",
                            "error": str(exc),
                            "actual_output_moments": None,
                        }
                    )
        print(f"  generated variants for {material}", flush=True)

    print(
        "[7/8] Evaluating holdout once using cached descriptors and distribution summaries",
        flush=True,
    )
    holdout_rows = []
    holdout_real_cache = {}
    for r in accepted:
        if r["split"] != "test":
            continue
        try:
            real = standardize_for_metrics(arrays[r["member"]])
            real_cache = {
                "field": real,
                "descriptor": make_descriptor(real),
                "moments": moments(real),
                "sorted": np.sort(real.ravel()),
            }
            holdout_real_cache[r["member"]] = real_cache
            for variant in ("legacy", "selected_gaussian", "selected_non_gaussian"):
                for seed in SEEDS:
                    syn, metadata, syn_desc, syn_mom, syn_sorted = variants[
                        (r["class"], variant, seed)
                    ]
                    row = {
                        "class": r["class"],
                        "member": r["member"],
                        "seed": seed,
                        "variant": variant,
                        "status": "ok" if syn is not None else "generation_failed",
                    }
                    if syn is None:
                        row.update(
                            {
                                "radial_psd_tv": None,
                                "acf_rmse": None,
                                "fit_score": None,
                                "error": metadata.get("error"),
                            }
                        )
                    else:
                        row.update(
                            evaluate(
                                real,
                                syn,
                                real_cache["descriptor"],
                                syn_desc,
                                real_cache["moments"],
                                syn_mom,
                                real_cache["sorted"],
                                syn_sorted,
                            )
                        )
                    holdout_rows.append(row)
        except (ValueError, FloatingPointError) as exc:
            holdout_rows.append(
                {
                    "class": r["class"],
                    "member": r["member"],
                    "variant": None,
                    "seed": None,
                    "status": "real_processing_failed",
                    "error": str(exc),
                    "radial_psd_tv": None,
                    "acf_rmse": None,
                    "fit_score": None,
                }
            )
        if (
            r["split"] == "test"
            and sum(
                q["split"] == "test" and q["class"] == r["class"]
                for q in accepted
                if q["member"] <= r["member"]
            )
            == counts[r["class"]]["test"]
        ):
            print(f"  holdout evaluated {r['class']}", flush=True)

    keys = [
        "radial_psd_tv",
        "acf_rmse",
        "fit_score",
        "height_w1_standardized",
        "skew_real",
        "skew_synthetic",
        "kurt_pearson_real",
        "kurt_pearson_synthetic",
        "direction_x_psd_tv",
        "direction_y_psd_tv",
    ]
    class_summary = {}
    for material in CLASSES:
        class_summary[material] = {}
        for variant in ("legacy", "selected_gaussian", "selected_non_gaussian"):
            sub = [
                r
                for r in holdout_rows
                if r.get("class") == material
                and r.get("variant") == variant
                and r["status"] == "ok"
            ]
            class_summary[material][variant] = {
                "file_count": len({r["member"] for r in sub}),
                "file_seed_row_count": len(sub),
                **mean_rows(sub, keys),
            }
    equal_class = {}
    for variant in ("legacy", "selected_gaussian", "selected_non_gaussian"):
        equal_class[variant] = {
            key: (
                float(
                    np.mean(
                        [
                            class_summary[c][variant][key]
                            for c in CLASSES
                            if class_summary[c][variant][key] is not None
                        ]
                    )
                )
                if any(class_summary[c][variant][key] is not None for c in CLASSES)
                else None
            )
            for key in keys
        }

    with (out / "holdout_per_file_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = sorted({key for row in holdout_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(holdout_rows)
    with (out / "generation_rows.json").open("w", encoding="utf-8") as f:
        json.dump(gen_rows, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    audit = {
        "provenance": {
            "archive_sha256": manifest["archive_sha256"],
            "calibration_script_path": str(Path(__file__).resolve()),
            "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generator_path": str(REPO / "roadmc/data/synthetic/procedural_texture.py"),
            "generator_sha256": hashlib.sha256(
                (REPO / "roadmc/data/synthetic/procedural_texture.py").read_bytes()
            ).hexdigest(),
            "calibration_module_sha256": hashlib.sha256(
                Path(__file__)
                .parents[1]
                .joinpath("data/synthetic/texture_shape_calibration.py")
                .read_bytes()
            ).hexdigest(),
        },
        "split_manifest_sha256": hashlib.sha256(
            (out / "split_manifest.json").read_bytes()
        ).hexdigest(),
        "reassessment_note": "Attempt 2 is a technical-correction rerun after attempt 1 had already evaluated the same holdout files. Holdout results were not used to change the training-only selection or parameters. The winning H=0.05 and band_hi=400 are search-grid boundary values and identify only the best tested candidates, not converged optima.",
        "train_test_counts": counts,
        "winning_shape_settings": winners,
        "holdout_per_class_equal_file_seed_mean": class_summary,
        "equal_class_mean": equal_class,
        "holdout_rows": holdout_rows,
        "generation_rows": gen_rows,
        "processing": {
            "real": "CSV transpose to x-major; least-squares plane detrend; fixed radial cosine gain; RMS standardize",
            "generated": "same plane detrend, same radial cosine gain, RMS standardize",
            "radial_psd": "2D FFT periodogram power divided by N; annular mean density; centers limited to <=500 cycles/m; common bin width 1/(181*0.5mm), normalized by sum for TV",
            "directional_psd": "marginal one-dimensional line periodogram averaged across orthogonal spatial lines; x-major axis 0 is X, axis 1 is Y; positive nonzero frequencies <=500 cycles/m; normalized by sum for directional TV; diagnostic only, not selected",
            "acf": "2D zero-padded linear autocorrelation; positive x/y axis lags 1..45, same samples",
            "holdout_metric": "each holdout file and seed reported; class means equal-file and equal-seed; final equal-class mean",
            "fit_score": "radial PSD shape TV + ACF RMSE",
        },
        "limits": [
            "four labels are asphalt surface classes, not independent roads or material-site replicates",
            "paper grid spacing is 0.5 mm; published Z units and normalization stage are unresolved",
            "rms_m=0.0004 m is an engineering assumption and not calibrated or compared",
            "shape scores use independently detrended and RMS-standardized fields, not absolute Z",
            "holdout contains only remaining CSVs after known explored and duplicate exclusions",
            "attempt 2 is a technical-correction rerun after the same holdout was inspected in attempt 1; no holdout-based parameter tuning was performed",
            "all winners lie at H lower bound 0.05 and band_hi upper bound 400 cycles/m; these are best candidates in the searched grid, not converged optima",
            "attempt 2 did not capture the executing script SHA-256; the saved source is explicitly a post-run snapshot",
            "no measured surface array is copied into the generator or used as a generated texture",
        ],
    }
    write_json(out / "audit.json", audit)

    # One compact visualization: holdout fit errors by class and generator variant.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    variants_order = ("legacy", "selected_gaussian", "selected_non_gaussian")
    colors = ("#687386", "#2878B5", "#E07832")
    for ax, metric, title in (
        (axes[0], "radial_psd_tv", "径向PSD形状TV"),
        (axes[1], "acf_rmse", "XY方向ACF RMSE"),
    ):
        for i, variant in enumerate(variants_order):
            vals = [class_summary[c][variant][metric] for c in CLASSES]
            ax.plot(
                np.arange(len(CLASSES)) + (i - 1) * 0.18, vals, "o", color=colors[i], label=variant
            )
        ax.set_xticks(np.arange(len(CLASSES)), CLASSES)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("仅剩余测试矩阵：冻结配置的形态误差（越低越好）")
    fig.savefig(out / "holdout_shape_metrics.png", dpi=160)
    plt.close(fig)

    report = [
        "# Royal Society 表面矩阵：纹理形态校准实验",
        "",
        f"冻结划分后保留训练 {sum(counts[c]['train'] for c in CLASSES)} 个矩阵、测试 {sum(counts[c]['test'] for c in CLASSES)} 个矩阵（逐类计数见 audit.json）。测试集不参与选参。已排除每类先前探索过的 test/0、test/1，以及 raw CSV 或解析矩阵重复项。",
        "",
        "在训练集上按每类独立选择 H∈{0.05,0.2,0.4,0.6,0.8} 与 band_hi∈{30,100,250,400} cycles/m，目标为平均径向PSD-TV + ACF-RMSE，精确并列时优先较低截止频率、再较低H。随后固定Gaussian谱参数，再由训练矩阵的等文件权重偏度与Pearson峰度构造非Gaussian参数。非Gaussian输出实际矩由 generation_rows.json 报告；目标值不保证实现值。",
        "",
        "本次attempt 2是在attempt 1已经评估过同一留出文件后，针对哈希排除、方向PSD口径、PSD末频点和候选有效行检查等技术问题重跑。未根据留出结果调整网格、参数或选择规则。所有类别都选到H搜索下界0.05和band_hi搜索上界400 cycles/m；这只表示当前有限候选网格中的最佳候选，不是收敛最优值。",
        "",
        "所有真实和生成矩阵先独立去平面，使用相同的径向分析低通（≤400 cycles/m增益1，400–500余弦渐降，≥500为0），再按RMS标准化。径向PSD是中心频率不超过500 cycles/m的环带均值密度；ACF为零填充线性相关，比较x/y正向1–45格点。方向PSD诊断沿x-major数组axis 0 (X)和axis 1 (Y)分别计算1D线功率谱，并在正交空间轴上取平均。报告的W1、偏峰度、方向PSD只是附加诊断，不参与参数选择。",
        "",
        "|类别|训练/测试文件数|训练选择 H / 截止频率|legacy PSD-TV|Gaussian PSD-TV|非Gaussian PSD-TV|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        win = winners[c]
        values = [class_summary[c][v]["radial_psd_tv"] for v in variants_order]
        cells = [f"{v:.4g}" if v is not None else "null" for v in values]
        report.append(
            f"|{c}|{counts[c]['train']}/{counts[c]['test']}|{win['hurst']:g} / {win['band_hi_cycles_m']:g}|"
            + "|".join(cells)
            + "|"
        )
    report += [
        "",
        "## 解释边界",
        "",
        "这四类是同一数据发布中的沥青表面分类标签，不等于四条独立道路或四种经实测校准的材料。测试矩阵数按类不平衡（尤其MS仅2个），因此同时报告分层数值，并以等类别权重作整体均值。论文给出XY网格间距0.5 mm，但CSV中Z单位、归一化处理阶段及逆映射仍未确认；RMS=0.0004 m是工程假设，本实验不据此声称绝对高度校准。该结果只支持这批剩余矩阵上的形态参数筛选，不能外推到实地道路，也不修改生产默认值。",
        "",
        "执行时未记录runner脚本SHA-256；source/下保存的是明确标注的post-run源快照，不声称与执行字节完全相同。该快照中的helper SHA与audit记录的执行hash相同，生成器SHA也在audit中记录。",
        "",
        "逐文件/逐seed明细、分层均值、失败与生成器实际输出矩见 audit.json、holdout_per_file_seed.csv、generation_rows.json。",
    ]
    (out / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("[8/8] Artifacts and report written", flush=True)
    print(
        json.dumps(
            {"output": str(out), "counts": counts, "winners": winners, "equal_class": equal_class},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
