"""Compare trained spectral envelopes on an explicitly already-seen file benchmark.

Inputs are the archive, frozen split manifest and prior fitted configurations
from calibrate_texture_shape. This command fits no parameter on test files and
never claims that rerunning a previously evaluated corpus is fresh validation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from roadmc.data.synthetic.procedural_texture import generate_texture
from roadmc.data.synthetic.spectral_synthesis import windowed_radial_psd
from roadmc.data.synthetic.texture_shape_calibration import canonical_array_sha256, moments
from roadmc.scripts.calibrate_texture_shape import (
    CLASSES,
    DX,
    DY,
    SEEDS,
    N,
    evaluate,
    make_descriptor,
    mean_rows,
    standardize_for_metrics,
)


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detrended_unit_field(array: np.ndarray) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    x, y = np.indices(a.shape, dtype=float)
    design = np.column_stack((np.ones(a.size), x.ravel(), y.ravel()))
    coefficients = np.linalg.lstsq(design, a.ravel(), rcond=None)[0]
    residual = a - (design @ coefficients).reshape(a.shape)
    if residual.std() <= 0:
        raise ValueError("constant field cannot define a spectral profile")
    return residual / residual.std()


def window_descriptor(field: np.ndarray) -> np.ndarray:
    psd = windowed_radial_psd(field, DX, DY, 500.0)["mean_density"]
    return psd / psd.sum()


def run(archive: Path, manifest_path: Path, prior_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refuse overwrite: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if sha(archive) != manifest["archive_sha256"]:
        raise ValueError("archive hash differs from frozen split manifest")
    included = manifest["included"]
    seen_raw, seen_array = set(), set()
    arrays = {}
    with zipfile.ZipFile(archive) as source:
        for row in included:
            raw = source.read(row["member"])
            a = np.loadtxt(io.StringIO(raw.decode("utf-8-sig")), delimiter=",")
            parsed_hash = canonical_array_sha256(a)
            if (
                hashlib.sha256(raw).hexdigest() != row["raw_sha256"]
                or parsed_hash != row["array_sha256"]
            ):
                raise ValueError(f"source changed since split: {row['member']}")
            if row["raw_sha256"] in seen_raw or parsed_hash in seen_array:
                raise ValueError("frozen manifest contains duplicate content")
            if row["split"] not in {"train", "test"} or a.shape != (N, N):
                raise ValueError("unexpected split or grid")
            seen_raw.add(row["raw_sha256"])
            seen_array.add(parsed_hash)
            arrays[row["member"]] = a
    output.mkdir(parents=True)
    module = Path(__file__).parents[1] / "data" / "synthetic"
    input_audit = {
        "archive_sha256": sha(archive),
        "manifest_sha256": sha(manifest_path),
        "prior_configuration_sha256": sha(prior_path),
        "source_hashes": {
            p.name: sha(p)
            for p in (
                Path(__file__),
                module / "procedural_texture.py",
                module / "spectral_synthesis.py",
                module / "texture_shape_calibration.py",
            )
        },
        "scope": "already-seen file benchmark; no independent-road or absolute-Z validation",
        "training": "frozen train records only; test does not choose spectrum, moments or iterations",
        "profile_estimation": "radial-Hann PSD of raw plane-detrended RMS-standardized training fields, before analysis low-pass",
        "evaluation": "same common analysis low-pass and RMS normalization for every real/generated field",
        "matching": {
            "iterations": 50,
            "tolerance": 1e-5,
            "origin": "fixed numerical settings, not physical priors",
        },
    }
    write_json(output / "inputs_and_source.json", input_audit)
    profiles, settings_by_class, configs = {}, {}, {}
    for category in CLASSES:
        training = [r for r in included if r["class"] == category and r["split"] == "train"]
        if not training:
            raise ValueError(f"no training files for {category}")
        densities, target_moments = [], []
        for row in training:
            native = detrended_unit_field(arrays[row["member"]].T)
            spectral = windowed_radial_psd(native, DX, DY, 500.0)
            density = spectral["mean_density"]
            densities.append(density / density.sum())
            target_moments.append(moments(standardize_for_metrics(arrays[row["member"]])))
        average = np.mean(densities, axis=0)
        minimum = np.nextafter(1 / (N * DX), 0.0)
        frequency = spectral["frequency_cycles_m"]
        # End knots represent declared constant extension inside the first/last
        # resolved bin, not inferred texture outside the acquired bandwidth.
        knots = np.r_[minimum, frequency, 500.0]
        values = np.r_[average[0], average, average[-1]]
        knots, indices = np.unique(knots, return_index=True)
        values = values[indices]
        shape = {
            "skew": float(np.mean([m["skew"] for m in target_moments])),
            "kurt": float(np.mean([m["kurt_pearson"] for m in target_moments])),
        }
        config = {
            "rms_m": 0.0004,
            "anisotropy_ratio": 1.0,
            "orientation_deg": 0.0,
            "band_lo_cycles_m": float(minimum),
            "band_hi_cycles_m": 500.0,
            "distribution": "gaussian",
            "material_id": "asphalt",
            "spectrum_profile": {
                "frequency_cycles_m": knots.tolist(),
                "relative_psd": values.tolist(),
            },
        }
        profiles[category] = {
            "training_members": [r["member"] for r in training],
            "target_shape": shape,
            "estimation_window": spectral["window"],
            "profile_interpolation_assumption": "linear PSD; constant first/last bin extension within explicit band",
            "absolute_height_calibrated": False,
        }
        configs[category] = {
            "diagnostic_legacy": prior["legacy_config"],
            "prior_non_gaussian": prior["frozen_non_gaussian_configs"][category],
            "profile_gaussian": config,
            "profile_non_gaussian": {**config, "distribution": "non_gaussian", "shape": shape},
            "profile_iaaft": {
                **config,
                "distribution": "non_gaussian",
                "shape": shape,
                "spectral_matching": {"iterations": 50, "tolerance": 1e-5},
            },
        }
        settings_by_class[category] = configs[category]
    write_json(output / "frozen_settings.json", {"classes": settings_by_class, "fitting": profiles})
    print("frozen training-only profiles and numerical settings", flush=True)
    # Generate fields before looking at benchmark scores.
    variants, generated_metadata = {}, []
    previews = {}
    x = np.arange(N) * DX
    for category in CLASSES:
        started = time.perf_counter()
        for name, config in configs[category].items():
            for seed in SEEDS:
                field, metadata = generate_texture(x, x, config, seed)
                processed = standardize_for_metrics(field.T)
                variants[(category, name, seed)] = {
                    "field": processed,
                    "descriptor": make_descriptor(processed),
                    "window_density": window_descriptor(processed),
                    "moments": moments(processed),
                    "sorted": np.sort(processed.ravel()),
                }
                generated_metadata.append(
                    {
                        "class": category,
                        "variant": name,
                        "seed": seed,
                        "achieved_analysis_moments": moments(processed),
                        "metadata": metadata,
                    }
                )
                if seed == 0:
                    previews[f"{category}_{name}_generated_m"] = field
        print(f"generated {category}, {time.perf_counter() - started:.1f}s", flush=True)
    write_json(output / "generated_metadata.json", generated_metadata)
    np.savez_compressed(output / "generated_previews.npz", **previews)
    rows = []
    for index, row in enumerate(included):
        if row["split"] != "test":
            continue
        real = standardize_for_metrics(arrays[row["member"]])
        desc, window_density = make_descriptor(real), window_descriptor(real)
        mom, sorted_real = moments(real), np.sort(real.ravel())
        for name in configs[row["class"]]:
            for seed in SEEDS:
                candidate = variants[(row["class"], name, seed)]
                scores = evaluate(
                    real,
                    candidate["field"],
                    desc,
                    candidate["descriptor"],
                    mom,
                    candidate["moments"],
                    sorted_real,
                    candidate["sorted"],
                )
                scores["windowed_radial_psd_tv"] = float(
                    0.5 * np.abs(window_density - candidate["window_density"]).sum()
                )
                scores["skew_abs_error"] = abs(scores["skew_real"] - scores["skew_synthetic"])
                scores["kurt_abs_error"] = abs(
                    scores["kurt_pearson_real"] - scores["kurt_pearson_synthetic"]
                )
                rows.append(
                    {
                        "class": row["class"],
                        "member": row["member"],
                        "variant": name,
                        "seed": seed,
                        **scores,
                    }
                )
        if index % 20 == 0:
            print(f"benchmark progress {index + 1}/{len(included)} manifest entries", flush=True)
    metrics = [
        "radial_psd_tv",
        "windowed_radial_psd_tv",
        "acf_rmse",
        "height_w1_standardized",
        "skew_abs_error",
        "kurt_abs_error",
    ]
    class_summary = defaultdict(dict)
    variant_names = list(configs[CLASSES[0]])
    for category in CLASSES:
        for name in variant_names:
            local = [r for r in rows if r["class"] == category and r["variant"] == name]
            class_summary[category][name] = {
                "files": len({r["member"] for r in local}),
                "file_seed_rows": len(local),
                **mean_rows(local, metrics),
            }
    equal_class = {
        name: {
            metric: float(np.mean([class_summary[c][name][metric] for c in CLASSES]))
            for metric in metrics
        }
        for name in variant_names
    }
    result = {
        "status": "seen_file_morphology_benchmark_not_real_domain_acceptance",
        "independent_roads": None,
        "absolute_z_units": "unresolved",
        "input_audit": input_audit,
        "classes": dict(class_summary),
        "equal_class": equal_class,
        "rows": rows,
        "benchmark_file_count": len({r["member"] for r in rows}),
        "row_count": len(rows),
        "configured_generation_count": len(generated_metadata),
        "limits": [
            "The same test files have been seen in earlier work. No fresh holdout or independent road claim.",
            "Statistical parameters only: generation receives no measured field or measured phase.",
            "500 cycles/m is a declared comparison support, not validated scanner resolving power.",
            "PSD/ACF/marginals are incomplete morphology descriptors; better scores do not prove material mechanics.",
        ],
    }
    write_json(output / "audit.json", result)
    report = [
        "# 训练谱包络与非高斯谱分布匹配",
        "",
        "本轮使用已见过的文件基准。只从冻结训练记录提取统计谱和目标矩，测试文件不选择模型参数。",
        "参数谱来自原始去平面归一化训练场的径向Hann PSD；评估时所有场使用相同低通/RMS处理。这样避免把分析低通重复拟合进生成谱。",
        "径向Hann按离散平方平均值归一；环带平均PSD密度与环积分功率分别计算。",
        "",
        "|候选|原径向TV|加窗径向TV|自相关RMSE|标准化W1|偏度绝对误差|峰度绝对误差|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variant_names:
        report.append(
            "|" + name + "|" + "|".join(f"{equal_class[name][m]:.6g}" for m in metrics) + "|"
        )
    report += [
        "",
        "这些是等类别、等文件、等种子的描述性比较，不能用像素数或比较行数充当独立样本数。",
        "生成器保留了旧方法，并将谱包络与迭代匹配做成显式选项；本轮没有自动替换生产默认参数。",
        "迭代后优先精确保持程序场频谱；边际分布只作近似，未达到数值容差会如实记录。",
        "未知Z单位、未确认道路身份、文件相关性和MS样本少仍限制结论。",
    ]
    (output / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.split_manifest, args.prior_config, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "rows": result["row_count"],
                "equal_class": result["equal_class"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
