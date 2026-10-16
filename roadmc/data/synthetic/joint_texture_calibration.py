"""Training-only joint spectrum/marginal search and an already-seen benchmark.

Leave-one-file-out folds are internal selection, not independent-road validation.
The RMS, isotropy, grid and multiplier search are explicit engineering settings.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from roadmc.data.synthetic.procedural_texture import _canonical_settings, generate_texture
from roadmc.data.synthetic.spectral_synthesis import windowed_radial_psd
from roadmc.data.synthetic.texture_shape_calibration import (
    acf_xy,
    canonical_array_sha256,
    moments,
    preprocess_shape,
)
from roadmc.scripts.benchmark_texture_envelopes import detrended_unit_field
from roadmc.scripts.calibrate_texture_shape import CLASSES, DX, DY, MEMBER_RE, N

METRICS = ("windowed_radial_psd_tv", "acf_rmse", "height_w1_standardized")
BASELINE = "prior_non_gaussian"
SCOPE = "uncalibrated shape-only; internal file CV; no independent-road or absolute-Z validation"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _manifest(archive, split_manifest):
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    if _sha(archive) != manifest["archive_sha256"]:
        raise ValueError("archive hash differs from frozen split manifest")
    rows = manifest["included"]
    seen = {key: set() for key in ("member", "raw_sha256", "array_sha256")}
    groups = {}
    for row in rows:
        match = MEMBER_RE.fullmatch(row["member"])
        if not match or (row["split"], row["class"], str(row["index"])) != match.groups():
            raise ValueError("member split/class/index conflicts with frozen original-file group")
        if row["shape"] != [N, N]:
            raise ValueError("unexpected frozen grid")
        group = (row["class"], row["index"])
        # File indices are scoped to their original archive split, not road IDs.
        group = (row["split"], *group)
        if group in groups:
            raise ValueError("duplicate original-file group")
        groups[group] = row["member"]
        for key in seen:
            if row[key] in seen[key]:
                raise ValueError(f"duplicate frozen {key}")
            seen[key].add(row[key])
    for category in CLASSES:
        indices = [r["index"] for r in rows if r["split"] == "train" and r["class"] == category]
        if len(indices) != 5 or set(indices) != set(range(5)):
            raise ValueError(f"expected five original TRAIN files for {category}")
    blocked = [r for r in manifest.get("excluded", []) if r.get("split") == "test"]
    for row in rows:
        if row["split"] == "test" and (
            row["index"] in (0, 1)
            or any(row[k] == b.get(k) for b in blocked for k in ("raw_sha256", "array_sha256"))
        ):
            raise ValueError("TEST record violates frozen exclusion/hash restrictions")
    return manifest


def _verified_arrays(archive, rows):
    """Read exactly the requested members; the caller controls TRAIN/TEST access."""
    arrays = {}
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            raw = source.read(row["member"])
            a = np.loadtxt(io.StringIO(raw.decode("utf-8-sig")), delimiter=",")
            if (
                hashlib.sha256(raw).hexdigest() != row["raw_sha256"]
                or canonical_array_sha256(a) != row["array_sha256"]
                or a.shape != (N, N)
            ):
                raise ValueError(f"source changed since frozen split: {row['member']}")
            arrays[row["member"]] = a
    return arrays


def _packet(field):
    a = preprocess_shape(field, DX, DY)
    density = windowed_radial_psd(a, DX, DY, 500.0)["mean_density"]
    if not np.isfinite(density).all() or density.sum() <= 0:
        raise ValueError("invalid radial-Hann PSD")
    return {
        "psd": density / density.sum(),
        "acf": acf_xy(a, maxlag=min(45, min(a.shape) - 1)),
        "sorted": np.sort(a.ravel()),
        "moments": moments(a),
    }


def _scores(real, candidate):
    return {
        METRICS[0]: float(0.5 * np.abs(real["psd"] - candidate["psd"]).sum()),
        METRICS[1]: float(np.sqrt(np.mean((real["acf"] - candidate["acf"]) ** 2))),
        METRICS[2]: float(np.mean(np.abs(real["sorted"] - candidate["sorted"]))),
    }


def _training_component(field, analysis_moments=None):
    native = detrended_unit_field(field)
    spectral = windowed_radial_psd(native, DX, DY, 500.0)
    density = spectral["mean_density"]
    return {
        "density": density / density.sum(),
        "frequency": spectral["frequency_cycles_m"],
        "moments": analysis_moments
        if analysis_moments is not None
        else moments(preprocess_shape(field, DX, DY)),
    }


def _training_summary(fields, *, cached=None):
    components = cached if cached is not None else [_training_component(field) for field in fields]
    densities = [component["density"] for component in components]
    shapes = [component["moments"] for component in components]
    average = np.mean(densities, axis=0)
    minimum = float(np.nextafter(1 / (fields[0].shape[0] * DX), 0.0))
    knots = np.r_[minimum, components[0]["frequency"], 500.0]
    values = np.r_[average[0], average, average[-1]]
    knots, indices = np.unique(knots, return_index=True)
    config = {
        "rms_m": 0.0004,
        "anisotropy_ratio": 1.0,
        "orientation_deg": 0.0,
        "band_lo_cycles_m": minimum,
        "band_hi_cycles_m": 500.0,
        "distribution": "gaussian",
        "material_id": "asphalt",
        "spectrum_profile": {
            "frequency_cycles_m": knots.tolist(),
            "relative_psd": values[indices].tolist(),
        },
    }
    return config, {
        "skew": float(np.mean([s["skew"] for s in shapes])),
        "kurt": float(np.mean([s["kurt_pearson"] for s in shapes])),
    }


def _candidates(multipliers):
    result = [
        {"id": BASELINE, "family": BASELINE},
        {"id": "profile_gaussian", "family": "profile_gaussian"},
    ]
    for family in ("profile_non_gaussian", "profile_iaaft"):
        for cs in multipliers:
            for ck in multipliers:
                result.append(
                    {"id": f"{family}_cs{cs:g}_ck{ck:g}", "family": family, "cs": cs, "ck": ck}
                )
    return result


def _setting(candidate, profile, shape, prior, iterations, tolerance):
    if candidate["family"] == BASELINE:
        return _canonical_settings(prior)
    cfg = dict(profile)
    if candidate["family"] != "profile_gaussian":
        cfg.update(
            distribution="non_gaussian",
            shape={
                "skew": candidate["cs"] * shape["skew"],
                "kurt": 3 + candidate["ck"] * (shape["kurt"] - 3),
            },
        )
    if candidate["family"] == "profile_iaaft":
        cfg["spectral_matching"] = {"iterations": iterations, "tolerance": tolerance}
    return _canonical_settings(cfg)


def _select(summaries):
    valid = [s for s in summaries if s["complete"]]
    baseline = next(s for s in valid if s["candidate_id"] == BASELINE)
    frontier = [
        s["candidate_id"]
        for s in valid
        if not any(
            all(other[m] <= s[m] for m in METRICS) and any(other[m] < s[m] for m in METRICS)
            for other in valid
        )
    ]
    eligible = []
    for s in valid:
        if s["candidate_id"] == BASELINE or not all(s[m] <= baseline[m] + 1e-12 for m in METRICS):
            continue
        ratios = [
            s[m] / baseline[m] if baseline[m] else (0.0 if s[m] == 0 else None) for m in METRICS
        ]
        # An improvement ratio against exactly zero is undefined/infinite unless also zero.
        if None not in ratios:
            eligible.append((float(np.mean(ratios)), s["candidate_id"]))
    selected = min(eligible)[1] if eligible else BASELINE
    return {
        "selected_candidate_id": selected,
        "baseline_candidate_id": BASELINE,
        "reason": "all_three_CV_metrics_no_worse" if eligible else "baseline_fallback",
        "eligible_ranked": [
            {"candidate_id": name, "mean_metric_ratio": ratio} for ratio, name in sorted(eligible)
        ],
        "pareto_candidate_ids": sorted(frontier),
        "comparison_tolerance": 1e-12,
        "zero_denominator_rule": "0/0=0; positive/0=ineligible (infinite ratio)",
    }


def _audit(archive, manifest_path, prior_path=None):
    modules = [
        Path(__file__),
        Path(__file__).with_name("procedural_texture.py"),
        Path(__file__).with_name("spectral_synthesis.py"),
        Path(__file__).with_name("texture_shape_calibration.py"),
        Path(__file__).with_name("am_envelope.py"),
        Path(__file__).parents[2] / "scripts" / "benchmark_texture_envelopes.py",
        Path(__file__).parents[2] / "scripts" / "calibrate_texture_shape.py",
    ]
    return {
        "archive_sha256": _sha(archive),
        "manifest_sha256": _sha(manifest_path),
        "prior_configuration_sha256": _sha(prior_path) if prior_path else None,
        "source_hashes": {p.name: _sha(p) for p in modules},
    }


def fit_joint_texture(
    archive: Path,
    split_manifest: Path,
    prior_config: Path,
    output_dir: Path,
    *,
    seeds=(0, 1, 2, 3, 4),
    multipliers=(0.75, 1.0, 1.25),
    iterations=50,
    tolerance=1e-5,
) -> dict:
    """Freeze a recommendation using five leave-one-original-TRAIN-file-out folds."""
    seeds, multipliers = tuple(seeds), tuple(multipliers)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be explicit nonnegative integers")
    if (
        len(set(seeds)) != len(seeds)
        or not multipliers
        or any(
            isinstance(m, bool) or not isinstance(m, (int, float)) or not np.isfinite(m) or m < 0
            for m in multipliers
        )
        or len(set(multipliers)) != len(multipliers)
    ):
        raise ValueError("unique seeds and finite nonnegative unique multipliers required")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 500
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("unsupported numerical settings: iterations=1..500, tolerance>0")
    manifest = _manifest(archive, split_manifest)
    training = sorted(
        (r for r in manifest["included"] if r["split"] == "train"),
        key=lambda r: (r["class"], r["member"]),
    )
    arrays = _verified_arrays(archive, training)  # Never opens TEST contents during fitting.
    prior = json.loads(Path(prior_config).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = _candidates(multipliers)
    packets = {name: _packet(a.T) for name, a in arrays.items()}
    raw = {name: a.T for name, a in arrays.items()}
    components = {
        name: _training_component(field, packets[name]["moments"]) for name, field in raw.items()
    }
    rows, summaries, decisions, selected, comparisons, refit = [], [], {}, {}, {}, {}
    axis = np.arange(N) * DX
    cache = {}
    for category in CLASSES:
        local = [r for r in training if r["class"] == category]
        prior_settings = prior["frozen_non_gaussian_configs"][category]
        for held in local:
            others = [r["member"] for r in local if r["member"] != held["member"]]
            profile, shape = _training_summary(
                [raw[name] for name in others], cached=[components[name] for name in others]
            )
            # Historical H/band remain fixed; fold target moments exclude the held file.
            fold_prior = {**prior_settings, "shape": shape}
            for candidate in candidates:
                for seed in seeds:
                    row = {
                        "class": category,
                        "fold_member": held["member"],
                        "training_members": others,
                        "candidate_id": candidate["id"],
                        "seed": seed,
                    }
                    try:
                        cfg = _setting(candidate, profile, shape, fold_prior, iterations, tolerance)
                        key = (json.dumps(cfg, sort_keys=True), seed)
                        if key not in cache:
                            field, _ = generate_texture(axis, axis, cfg, seed)
                            cache[key] = _packet(field)
                        row.update(_scores(packets[held["member"]], cache[key]), status="ok")
                    except (ValueError, FloatingPointError, OverflowError) as exc:
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
            # Fold-local generation is shared across files/seeds; avoid retaining all fold fields.
            cache.clear()
        class_summaries = []
        for candidate in candidates:
            matched = [
                r for r in rows if r["class"] == category and r["candidate_id"] == candidate["id"]
            ]
            valid = [r for r in matched if r["status"] == "ok"]
            summary = {
                "class": category,
                "candidate_id": candidate["id"],
                "expected_rows": 5 * len(seeds),
                "valid_rows": len(valid),
                "complete": len(valid) == 5 * len(seeds),
                **{m: float(np.mean([r[m] for r in valid])) if valid else None for m in METRICS},
            }
            class_summaries.append(summary)
        decisions[category] = _select(class_summaries)
        summaries.extend(class_summaries)
        profile, shape = _training_summary(
            [raw[r["member"]] for r in local], cached=[components[r["member"]] for r in local]
        )
        chosen = next(
            c for c in candidates if c["id"] == decisions[category]["selected_candidate_id"]
        )
        selected[category] = _setting(chosen, profile, shape, prior_settings, iterations, tolerance)
        center = [
            {"id": BASELINE, "family": BASELINE},
            {"id": "profile_gaussian", "family": "profile_gaussian"},
            *(
                {"id": f, "family": f, "cs": 1.0, "ck": 1.0}
                for f in ("profile_non_gaussian", "profile_iaaft")
            ),
        ]
        comparisons[category] = {
            c["id"]: _setting(c, profile, shape, prior_settings, iterations, tolerance)
            for c in center
        }
        comparisons[category]["selected"] = selected[category]
        refit[category] = {
            "training_members": [r["member"] for r in local],
            "analysis_processed_target_moments": shape,
        }
        print(f"joint texture CV/refit {category}: {chosen['id']}", flush=True)
    result = {
        "status": "uncalibrated",
        "scope": SCOPE,
        "selection_scope": "five original TRAIN files/class; leave-one-file-out; four fitting files/fold",
        "estimation": "PSD: raw plane-detrended unit fields before lowpass; target moments: common analysis-processed TRAIN fields",
        "baseline_definition": "CV: frozen historical H/band with target moments refitted on remaining four TRAIN files; final/benchmark: exact original full-TRAIN prior config; historical H/band themselves were selected on all five TRAIN files, so CV is internal comparison only",
        "analysis": "0.5 mm grid; plane detrend; 400-500 cycles/m cosine taper; RMS standardize",
        "search_settings_status": "cs/ck/iterations/tolerance are numerical settings, not physical priors",
        "seeds": list(seeds),
        "candidates": candidates,
        "iterations": iterations,
        "tolerance": tolerance,
        "selected_settings": selected,
        "comparison_settings": comparisons,
        "decisions": decisions,
        "cv_rows": rows,
        "cv_summaries": summaries,
        "refit": refit,
        "input_provenance": _audit(archive, split_manifest, prior_config),
        "verified_train_members": training,
    }
    _write(output_dir / "frozen_settings.json", result)
    return result


def benchmark_joint_texture(
    archive, split_manifest, fitted_result, output_dir, *, seeds=(0, 1, 2, 3, 4)
) -> dict:
    """Compare frozen settings on already-seen TEST files; never retune."""
    fit = (
        json.loads(Path(fitted_result).read_text(encoding="utf-8"))
        if isinstance(fitted_result, (str, Path))
        else fitted_result
    )
    manifest = _manifest(archive, split_manifest)
    if (
        _sha(archive) != fit["input_provenance"]["archive_sha256"]
        or _sha(split_manifest) != fit["input_provenance"]["manifest_sha256"]
    ):
        raise ValueError("benchmark inputs differ from frozen fit")
    seeds = tuple(seeds)
    if list(seeds) != fit["seeds"]:
        raise ValueError("benchmark seeds must equal frozen common fitting seeds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = {
        c: {**fit["comparison_settings"][c], "selected": fit["selected_settings"][c]}
        for c in CLASSES
    }
    _write(output_dir / "frozen_settings.json", {"classes": configs, "seeds": list(seeds)})
    variants, previews, metadata = {}, {}, []
    axis = np.arange(N) * DX
    # Complete every generation before opening any TEST member.
    for category in CLASSES:
        for name, cfg in configs[category].items():
            for seed in seeds:
                field, audit = generate_texture(axis, axis, cfg, seed)
                variants[(category, name, seed)] = _packet(field)
                metadata.append(
                    {"class": category, "variant": name, "seed": seed, "metadata": audit}
                )
                if seed == seeds[0]:
                    previews[f"{category}_{name}_generated_m"] = field
    np.savez_compressed(output_dir / "generated_previews.npz", **previews)
    _write(output_dir / "generated_metadata.json", metadata)
    tests = [r for r in manifest["included"] if r["split"] == "test"]
    arrays = _verified_arrays(archive, tests)
    rows = []
    for record in tests:
        real = _packet(arrays[record["member"]].T)
        for name in configs[record["class"]]:
            for seed in seeds:
                candidate = variants[(record["class"], name, seed)]
                rows.append(
                    {
                        "class": record["class"],
                        "member": record["member"],
                        "variant": name,
                        "seed": seed,
                        **_scores(real, candidate),
                        "skew_abs_error": abs(
                            real["moments"]["skew"] - candidate["moments"]["skew"]
                        ),
                        "kurt_abs_error": abs(
                            real["moments"]["kurt_pearson"] - candidate["moments"]["kurt_pearson"]
                        ),
                    }
                )
    metrics = (*METRICS, "skew_abs_error", "kurt_abs_error")
    summaries = {}
    for category in CLASSES:
        summaries[category] = {}
        for name in configs[category]:
            local = [r for r in rows if r["class"] == category and r["variant"] == name]
            if not local:
                raise ValueError(f"no frozen TEST files for {category}")
            summaries[category][name] = {m: float(np.mean([r[m] for r in local])) for m in metrics}
    result = {
        "status": "uncalibrated",
        "scope": "already-seen file benchmark; no retuning; " + SCOPE,
        "test_files": len(tests),
        "seeds": list(seeds),
        "rows": rows,
        "class_summary": summaries,
        "equal_class_summary": {
            name: {m: float(np.mean([summaries[c][name][m] for c in CLASSES])) for m in metrics}
            for name in configs[CLASSES[0]]
        },
        "input_provenance": _audit(archive, split_manifest),
        "fit_provenance": fit["input_provenance"],
        "frozen_fit_sha256": hashlib.sha256(json.dumps(fit, sort_keys=True).encode()).hexdigest(),
        "verified_test_members": tests,
    }
    _write(output_dir / "benchmark_summary.json", result)
    return result
