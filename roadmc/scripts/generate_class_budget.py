"""Generate a resume-safe, label-budgeted RoadMC synthetic dataset.

Each generated scene forces exactly one target disease label. Completion is
defined by both an independent-scene quota and an effective target-point quota,
so rare geometric classes cannot be hidden behind an equal file count.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, has_observable_feature_schema
from roadmc.data.synthetic.config import (
    ISO_ROUGHNESS,
    CurbConfig,
    DiseaseConfig,
    GeneratorConfig,
    MicroTextureConfig,
    RadiometricConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.labels import ALL_DISEASE_LABELS

_WORKER_DATASET: Any = None
_WORKER_SPLIT_DIR: Path | None = None


def _parse_labels(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return ALL_DISEASE_LABELS

    labels: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            labels.update(range(start, end + 1))
        else:
            labels.add(int(item))

    invalid = sorted(label for label in labels if label not in ALL_DISEASE_LABELS)
    if invalid:
        raise ValueError(f"Only non-background labels 1..37 are supported, got {invalid}")
    if not labels:
        raise ValueError("At least one target label is required")
    return tuple(sorted(labels))


def _parse_roughness_mix(text: str | None) -> dict[str, float] | None:
    """Parse a ``'D:0.5,E:0.5'`` ISO 8608 mixture spec into class weights.

    ``None`` / an empty string returns ``None`` (mixing disabled).  Each
    comma-separated item must be ``<LEVEL>:<weight>`` with LEVEL one of the
    uppercase ISO 8608 classes, a finite non-negative weight, no repeated
    classes, and a strictly positive total weight; any violation raises
    ``ValueError``.
    """

    if text is None or not text.strip():
        return None

    mix: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError(
                f"--roughness-mix contains an empty item in {text!r}; "
                'expected e.g. "D:0.5,E:0.5"'
            )
        level, sep, weight_text = item.partition(":")
        level = level.strip()
        if not sep or level not in ISO_ROUGHNESS:
            raise ValueError(
                f"--roughness-mix level must be one of {sorted(ISO_ROUGHNESS)}, "
                f"got {item!r}"
            )
        if level in mix:
            raise ValueError(f"--roughness-mix repeats level {level!r}")
        try:
            weight = float(weight_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"--roughness-mix weight must be a float, got {item!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"--roughness-mix weight must be finite and >= 0, got {item!r}"
            )
        mix[level] = weight

    if not mix or sum(mix.values()) <= 0:
        raise ValueError(
            f"--roughness-mix weights must sum to > 0, got {text!r}"
        )
    return mix


def _scene_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"scene_(\d+)\.npz", path.name)
    return int(match.group(1)) if match else None


def _next_scene_id(split_dir: Path) -> int:
    ids = [scene_id for path in split_dir.glob("scene_*.npz") if (scene_id := _scene_id_from_path(path)) is not None]
    return max(ids, default=-1) + 1


def _empty_coverage(labels: tuple[int, ...]) -> dict[int, dict[str, int]]:
    return {
        label: {"scene_count": 0, "instance_count": 0, "point_count": 0}
        for label in labels
    }


def _scan_coverage(
    split_dir: Path,
    labels: tuple[int, ...],
    *,
    ignore_legacy_scenes: bool = False,
) -> dict[int, dict[str, int]]:
    coverage = _empty_coverage(labels)
    if not split_dir.exists():
        return coverage

    target_set = set(labels)
    legacy_scenes: list[str] = []
    for path in sorted(split_dir.glob("scene_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as scene:
                # Scenes without the resolution contract predate the natural
                # prevalence sampler (they were generated under the removed
                # fixed 10% target-label quota).  Counting them toward quotas
                # would silently mix two sampling regimes in one dataset.
                if "resolution_metadata_json" not in scene.files:
                    legacy_scenes.append(path.name)
                    continue
                scene_labels = scene["labels"].astype(np.int64, copy=False)
                target_label = int(scene["target_label"]) if "target_label" in scene.files else -1
        except Exception as exc:  # pragma: no cover - corrupt files are operational failures
            warnings.warn(f"Skipping unreadable scene {path}: {exc}", stacklevel=2)
            continue

        unique_labels, counts = np.unique(scene_labels, return_counts=True)
        count_by_label = {int(label): int(count) for label, count in zip(unique_labels, counts, strict=False)}
        # A quota is evidence only when the scene was deliberately generated
        # for that label.  Counting incidental co-occurrences could otherwise
        # make a rare class appear covered without independent instances.
        if target_label in target_set and count_by_label.get(target_label, 0) > 0:
            coverage[target_label]["scene_count"] += 1
            coverage[target_label]["instance_count"] += 1
            coverage[target_label]["point_count"] += count_by_label[target_label]

    if legacy_scenes:
        if not ignore_legacy_scenes:
            raise SystemExit(
                f"{split_dir} contains {len(legacy_scenes)} legacy scene(s) without the "
                "resolution contract (e.g. generated under the removed fixed 10% "
                "target-label quota); resuming into a mixed-regime directory is not "
                "allowed. Use a fresh --output-dir, or pass --ignore-legacy-scenes to "
                f"exclude them from quota accounting. First offender: {legacy_scenes[0]}"
            )
        warnings.warn(
            f"{split_dir}: excluded {len(legacy_scenes)} legacy scene(s) without the "
            "resolution contract from quota accounting.", stacklevel=2
        )

    return coverage


def _feature_contract_errors(split_dir: Path) -> list[str]:
    """Return data-contract violations that invalidate a class-budget run."""

    errors: list[str] = []
    if not split_dir.exists():
        return errors
    for path in sorted(split_dir.glob("scene_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as scene:
                if "feature_schema" not in scene.files or not has_observable_feature_schema(
                    scene["feature_schema"]
                ):
                    errors.append(f"{path.name}: missing or incompatible observable feature schema")
                    continue
                points = scene["points"]
                feats = scene["feats"]
                if feats.ndim != 2 or feats.shape != (len(points), 3):
                    errors.append(f"{path.name}: feats shape {feats.shape} does not match (N, 3)")
                elif not np.isfinite(feats).all():
                    errors.append(f"{path.name}: feats contain non-finite values")
        except Exception as exc:  # pragma: no cover - corrupt files are operational failures
            errors.append(f"{path.name}: unreadable ({exc})")
    return errors


def _is_satisfied(record: dict[str, int], target_scenes: int, min_points: int) -> bool:
    return record["scene_count"] >= target_scenes and record["point_count"] >= min_points


def _unmet_labels(
    coverage: dict[int, dict[str, int]], target_scenes: int, min_points: int
) -> list[int]:
    return [
        label
        for label, record in coverage.items()
        if not _is_satisfied(record, target_scenes, min_points)
    ]


def _init_worker(config: GeneratorConfig, split_dir: str) -> None:
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir)


def _save_forced_scene(task: tuple[int, int]) -> dict[str, int | bool | str]:
    if _WORKER_DATASET is None or _WORKER_SPLIT_DIR is None:
        raise RuntimeError("Budget worker was not initialized")

    scene_id, target_label = task
    output_path = _WORKER_SPLIT_DIR / f"scene_{scene_id:06d}.npz"
    if output_path.exists():
        return {"ok": True, "skipped": True, "scene_id": scene_id, "target_label": target_label, "target_points": 0}

    try:
        scene = _WORKER_DATASET.generate_scene(scene_id, target_label=target_label)
        points = scene["points"]
        labels = scene["labels"]
        feats = scene["feats"]
        normals = scene["normals"]
        valid = ~np.any(np.isnan(points), axis=1)
        points = points[valid]
        labels = labels[valid]
        feats = feats[valid]
        normals = normals[valid]
        target_points = int(np.count_nonzero(labels == target_label))
        if len(points) == 0 or target_points == 0:
            raise RuntimeError(f"forced label {target_label} was absent after synthesis")

        np.savez_compressed(
            output_path,
            points=points.astype(np.float32),
            labels=labels.astype(np.int32),
            feats=feats.astype(np.float32),
            normals=normals.astype(np.float32),
            pavement_type=scene["pavement_type"],
            roughness_class=scene["roughness_class"],
            scene_id=scene_id,
            target_label=target_label,
            feature_schema=scene["feature_schema"],
            feature_names=scene["feature_names"],
            feature_k_neighbors=scene["feature_k_neighbors"],
            coordinate_center=scene["coordinate_center"],
            coordinate_scale=scene["coordinate_scale"],
            coordinates_normalized=scene["coordinates_normalized"],
            resolution_metadata_json=np.asarray(
                json.dumps(
                    scene["resolution_metadata"],
                    ensure_ascii=True,
                    sort_keys=True,
                )
            ),
            resolution_contract=scene["resolution_contract"],
            surface_grid_spacing_m=scene["surface_grid_spacing_m"],
            surface_grid_shape=scene["surface_grid_shape"],
            surface_grid_point_count=scene["surface_grid_point_count"],
            sensor_output_point_count=scene["sensor_output_point_count"],
            observation_model=scene["observation_model"],
            segment_count=scene["segment_count"],
            segment_length_m=scene["segment_length_m"],
            segment_overlap_m=scene["segment_overlap_m"],
            model_target_points=scene["model_target_points"],
        )
        return {
            "ok": True,
            "skipped": False,
            "scene_id": scene_id,
            "target_label": target_label,
            "target_points": target_points,
        }
    except Exception as exc:  # pragma: no cover - surfaced in the parent summary
        return {
            "ok": False,
            "skipped": False,
            "scene_id": scene_id,
            "target_label": target_label,
            "target_points": 0,
            "error": repr(exc),
        }


def _budget_split(
    output_dir: Path,
    split: str,
    config: GeneratorConfig,
    labels: tuple[int, ...],
    target_scenes: int,
    min_points: int,
    workers: int,
    max_attempts: int,
    wave_size: int,
    dry_run: bool,
    ignore_legacy_scenes: bool = False,
) -> dict[str, Any]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    coverage = _scan_coverage(split_dir, labels, ignore_legacy_scenes=ignore_legacy_scenes)
    initial_coverage = {label: record.copy() for label, record in coverage.items()}
    start_scene_id = _next_scene_id(split_dir)
    attempts = {label: coverage[label]["instance_count"] for label in labels}
    failures: list[dict[str, Any]] = []
    generated = 0
    next_scene_id = start_scene_id

    print(
        f"[{split}] start scenes={len(list(split_dir.glob('scene_*.npz')))}, "
        f"target_scenes/class={target_scenes}, min_points/class={min_points}"
    )

    if dry_run:
        return {
            "split": split,
            "dry_run": True,
            "initial_coverage": initial_coverage,
            "coverage": coverage,
            "unmet_labels": _unmet_labels(coverage, target_scenes, min_points),
        }

    # 各 split 使用不相交的种子段，保证 train/val/test 场景来自
    # 互不重叠的随机序列（独立测试集声明的前提）。
    _SEED_SEGMENT = {"train": 0, "val": 100_000, "test": 200_000}
    pool_config = config
    if config.seed is not None and _SEED_SEGMENT.get(split, 0):
        pool_config = replace(config, seed=config.seed + _SEED_SEGMENT[split])

    with mp.Pool(workers, initializer=_init_worker, initargs=(pool_config, str(split_dir))) as pool:
        while True:
            unmet = _unmet_labels(coverage, target_scenes, min_points)
            eligible = [label for label in unmet if attempts[label] < max_attempts]
            if not eligible:
                break

            tasks: list[tuple[int, int]] = []
            for label in eligible:
                scene_gap = max(0, target_scenes - coverage[label]["scene_count"])
                planned = max(1, min(wave_size, scene_gap if scene_gap > 0 else wave_size))
                planned = min(planned, max_attempts - attempts[label])
                for _ in range(planned):
                    tasks.append((next_scene_id, label))
                    next_scene_id += 1
                    attempts[label] += 1

            for result in pool.imap_unordered(_save_forced_scene, tasks, chunksize=1):
                label = int(result["target_label"])
                if bool(result["ok"]) and not bool(result["skipped"]):
                    generated += 1
                    coverage[label]["scene_count"] += 1
                    coverage[label]["instance_count"] += 1
                    coverage[label]["point_count"] += int(result["target_points"])
                elif not bool(result["ok"]):
                    failures.append(dict(result))

            unmet_after_wave = _unmet_labels(coverage, target_scenes, min_points)
            print(
                f"[{split}] generated={generated}, unmet={len(unmet_after_wave)}, "
                f"failures={len(failures)}"
            )

    final_coverage = _scan_coverage(split_dir, labels)
    unmet = _unmet_labels(final_coverage, target_scenes, min_points)
    feature_contract_errors = _feature_contract_errors(split_dir)
    return {
        "split": split,
        "dry_run": False,
        "generated": generated,
        "failed": len(failures),
        "failure_examples": failures[:10],
        "initial_coverage": initial_coverage,
        "coverage": final_coverage,
        "unmet_labels": unmet,
        "feature_contract_schema": OBSERVABLE_FEATURE_SCHEMA,
        "feature_contract_errors": feature_contract_errors[:20],
        "complete": not unmet and not feature_contract_errors,
        "target_scenes_per_class": target_scenes,
        "min_points_per_class": min_points,
        "max_attempts_per_class": max_attempts,
    }


def _auto_workers() -> int:
    return max(1, min(8, mp.cpu_count() - 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./data/class_budget_output")
    parser.add_argument(
        "--split", choices=("train", "val", "test", "both", "all"), default="both",
        help="'both'=train+val；'all'=train+val+test（独立测试集，种子段 +200000）",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="test split 的配额缩放比（相对 train 配额），与 --val-ratio 同语义",
    )
    parser.add_argument("--labels", default="all", help="all, 1-37, or comma/range syntax such as 1-8,20,37")
    parser.add_argument("--target-scenes-per-class", type=int, default=20)
    parser.add_argument("--min-points-per-class", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-attempts-per-class", type=int, default=0, help="0 chooses a conservative automatic limit")
    parser.add_argument(
        "--target-label-min-output-points",
        type=int,
        default=0,
        help=(
            "minimum surviving target-label points per controlled scene; 0 derives "
            "ceil(min-points-per-class / target-scenes-per-class) so the point quota "
            "stays arithmetically reachable under natural prevalence"
        ),
    )
    parser.add_argument(
        "--max-parallel-memory-mib",
        type=float,
        default=8192.0,
        help="aggregate surface-synthesis memory budget across all workers (MiB)",
    )
    parser.add_argument(
        "--ignore-legacy-scenes",
        action="store_true",
        help="exclude pre-contract scenes from quota accounting instead of aborting",
    )
    parser.add_argument("--wave-size", type=int, default=4, help="maximum new scenes per incomplete class per scheduling wave")
    parser.add_argument("--workers", type=int, default=0, help="0 chooses min(8, cpu_count - 2); use 16 explicitly when appropriate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-res", type=float, default=0.005)
    parser.add_argument("--width", type=float, default=7.0, help="路面宽度 (米)，默认 7.0（双车道口径）")
    parser.add_argument("--length", type=float, default=5.0, help="路面长度（里程向）(米)，默认 5.0")
    parser.add_argument(
        "--observation-model",
        choices=["on", "off"],
        default="off",
        help=(
            "观测模型体制：off（默认，新体制）直出全部表面点并跳过扫描/噪声/"
            "丢点/输出采样（num_points 被忽略并记录在元数据 "
            "ignored_under_observation_off，类预算按全表面点计量）；on 保留"
            "完整传感器管线与 num_points 固定采样，与历史版本逐位一致"
            "（legacy 回归口径）。与 generate_synthetic.py 默认一致。"
        ),
    )
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--pavement", choices=("asphalt", "concrete", "mixed"), default="mixed")
    parser.add_argument("--roughness", choices=("A", "B", "C", "D", "E"), default="B")
    parser.add_argument(
        "--roughness-mix",
        type=str,
        default=None,
        metavar='"D:0.5,E:0.5"',
        help=(
            "per-scene ISO 8608 class mixture (keys A-E, non-negative weights, "
            "positive sum); when set, the class is drawn per scene and takes "
            "precedence over --roughness, which then only acts as the "
            "dataclass fallback"
        ),
    )
    parser.add_argument(
        "--texture-rms",
        type=float,
        default=0.0008,
        help="宏观纹理位移 RMS (m)，映射 micro_texture.amplitude；0 关闭纹理段",
    )
    parser.add_argument(
        "--texture-het-sigma",
        type=float,
        default=0.0,
        help="纹理沿程不均匀的对数域调制强度 σ_ln；0 = 关闭（默认）",
    )
    parser.add_argument(
        "--texture-het-corr",
        type=float,
        default=2.0,
        help="纹理沿程调制的里程向高斯相关长度 (m)，仅 het-sigma > 0 时生效",
    )
    parser.add_argument(
        "--curb-width",
        type=float,
        default=0.0,
        help="缘石条带横向宽度 (m)，0=关闭",
    )
    parser.add_argument(
        "--curb-height",
        type=float,
        default=0.13,
        help="缘石顶面抬高 (m)",
    )
    parser.add_argument(
        "--albedo-sigma-ln",
        type=float,
        default=0.25,
        help="对数正态反照率场 σ_ln（默认 0.25 = 历史硬编码值）",
    )
    parser.add_argument(
        "--speckle-m",
        type=float,
        default=12.0,
        help="乘性 Gamma 斑点形状参数 M（默认 12 = 历史硬编码值）",
    )
    parser.add_argument(
        "--max-diseases",
        type=int,
        default=1,
        help="每场景最多病害数（默认 1 = 历史硬编码值）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_scenes_per_class < 1:
        raise ValueError("--target-scenes-per-class must be >= 1")
    if args.min_points_per_class < 1:
        raise ValueError("--min-points-per-class must be >= 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.wave_size < 1:
        raise ValueError("--wave-size must be >= 1")
    if args.texture_rms < 0:
        raise ValueError("--texture-rms must be >= 0")
    if args.texture_het_sigma < 0:
        raise ValueError("--texture-het-sigma must be >= 0")
    if args.texture_het_sigma > 0 and args.texture_het_corr <= 0:
        raise ValueError("--texture-het-corr must be > 0 when --texture-het-sigma > 0")
    if args.curb_width < 0:
        raise ValueError("--curb-width must be >= 0")
    if args.curb_width > 0 and args.curb_height <= 0:
        raise ValueError("--curb-height must be > 0 when --curb-width > 0")
    if args.albedo_sigma_ln < 0:
        raise ValueError("--albedo-sigma-ln must be >= 0")
    if args.speckle_m < 1:
        raise ValueError("--speckle-m must be >= 1")
    if args.max_diseases < 1:
        raise ValueError("--max-diseases must be >= 1")
    if args.roughness_mix is not None and args.roughness_mix.strip():
        print(
            "[note] --roughness-mix is set: the ISO 8608 class is drawn per scene "
            "from the mixture and takes precedence; --roughness "
            f"({args.roughness}) remains only as the dataclass fallback value.",
            file=sys.stderr,
        )

    labels = _parse_labels(args.labels)
    workers = _auto_workers() if args.workers <= 0 else args.workers
    max_attempts = args.max_attempts_per_class
    if max_attempts <= 0:
        max_attempts = max(args.target_scenes_per_class * 3, 20)

    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            width=args.width,
            length=args.length,
            grid_res=args.grid_res,
            pavement_type=args.pavement,
            roughness_class=args.roughness,
            roughness_mix=_parse_roughness_mix(args.roughness_mix),
        ),
        micro_texture=MicroTextureConfig(
            amplitude=args.texture_rms,
            het_sigma_ln=args.texture_het_sigma,
            het_corr_len_m=args.texture_het_corr,
        ),
        curb=CurbConfig(width_m=args.curb_width, height_m=args.curb_height),
        radiometric=RadiometricConfig(
            albedo_sigma_ln=args.albedo_sigma_ln,
            speckle_m=args.speckle_m,
        ),
        disease=DiseaseConfig(max_diseases_per_scene=args.max_diseases),
        seed=args.seed,
        num_points=args.num_points,
        # 与 generate_synthetic.py 同一冻结体制：CLI 默认 off（直出全表面），
        # on 为 legacy 回归口径（显式传值时与历史版本逐位一致）。
        observation_model=args.observation_model,
    )
    estimate = config.estimate_surface_generation()
    if args.workers <= 0:
        # 自动 worker 数必须尊重并行内存预算：毫米条带单 worker 峰值
        # （分段合成）可达数 GiB，min(8, cpu-2) 的核心数口径会直接把
        # 8 个 worker 压爆 max_parallel_memory_mib。按估算峰值收拢。
        peak = max(estimate.worker_peak_mib, 1.0)
        budget_cap = max(1, int(args.max_parallel_memory_mib // peak))
        workers = min(_auto_workers(), budget_cap)
        if workers < _auto_workers():
            print(
                f"[note] auto workers capped to {workers} by parallel memory "
                f"budget ({args.max_parallel_memory_mib:.0f} MiB / "
                f"~{peak:.0f} MiB per worker)"
            )
    config.validate_parallel_budget(workers, args.max_parallel_memory_mib)
    # 分段条带计划在 on 体制下不可执行（观测管线未分段），必须前置硬
    # 报错而不是逐场景静默失败；off 体制支持分段，放行。条件与
    # generate_synthetic.py 的 _validate_request 保持一致。
    if estimate.segmented and config.observation_model == "on":
        raise SystemExit(
            f"Strip of {estimate.point_count:,} points requires the segmented "
            f"regime ({estimate.segment_count} segments × "
            f"{estimate.segment_length_m:.3f} m), which does not support the "
            "observation pipeline (observation_model='on') in this build. "
            "Re-run with --observation-model off, or raise the surface budget "
            "(GeneratorConfig.max_surface_points / max_surface_memory_mib) to "
            "hold the whole strip in one block, or use a coarser --grid-res."
        )
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in (0, 1)")
    output_dir = Path(args.output_dir)
    if args.split == "both":
        selected_splits: tuple[str, ...] = ("train", "val")
    elif args.split == "all":
        selected_splits = ("train", "val", "test")
    else:
        selected_splits = (args.split,)
    started = time.time()
    reports: dict[str, dict[str, Any]] = {}

    for split in selected_splits:
        ratio = None
        if args.split in ("both", "all"):
            if split == "val":
                ratio = args.val_ratio
            elif split == "test":
                ratio = args.test_ratio
        if ratio is not None:
            target_scenes = max(1, int(np.ceil(args.target_scenes_per_class * ratio)))
            min_points = max(1, int(np.ceil(args.min_points_per_class * ratio)))
        else:
            target_scenes = args.target_scenes_per_class
            min_points = args.min_points_per_class
        # Under natural prevalence, thin classes can survive with only a few
        # points per scene, which makes the point quota arithmetically
        # unreachable within the attempt cap.  The per-scene survival floor is
        # therefore derived from the quota itself (and audited per scene via
        # target_label_protection) instead of silently fixing a percentage.
        min_output_points = args.target_label_min_output_points
        if min_output_points <= 0:
            min_output_points = max(1, int(np.ceil(min_points / target_scenes)))
        if min_output_points > args.num_points:
            raise ValueError(
                f"target-label survival floor {min_output_points} exceeds --num-points "
                f"{args.num_points}; lower --min-points-per-class or raise --num-points"
            )
        split_config = replace(config, target_label_min_output_points=min_output_points)
        print(f"[{split}] target_label_min_output_points={min_output_points}")
        reports[split] = _budget_split(
            output_dir=output_dir,
            split=split,
            config=split_config,
            labels=labels,
            target_scenes=target_scenes,
            min_points=min_points,
            workers=workers,
            max_attempts=max_attempts,
            wave_size=args.wave_size,
            dry_run=args.dry_run,
            ignore_legacy_scenes=args.ignore_legacy_scenes,
        )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_sec": time.time() - started,
        "arguments": vars(args),
        "target_labels": list(labels),
        "workers": workers,
        "reports": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "class_budget_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    complete = all(report.get("complete", False) for report in reports.values())
    print(f"Manifest: {manifest_path}")
    if not complete and not args.dry_run:
        raise SystemExit("Class budget is incomplete; inspect class_budget_manifest.json before training.")


if __name__ == "__main__":
    main()
