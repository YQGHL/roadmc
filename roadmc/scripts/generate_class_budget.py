"""Generate a resume-safe, label-budgeted RoadMC synthetic dataset.

Each generated scene forces exactly one target disease label. Completion is
defined by both an independent-scene quota and an effective target-point quota,
so rare geometric classes cannot be hidden behind an equal file count.
"""

from __future__ import annotations

import argparse
import json
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

from roadmc.data.synthetic.config import DiseaseConfig, GeneratorConfig, RoadSurfaceConfig
from roadmc.data.synthetic.labels import ALL_DISEASE_LABELS
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, has_observable_feature_schema


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


def _scan_coverage(split_dir: Path, labels: tuple[int, ...]) -> dict[int, dict[str, int]]:
    coverage = _empty_coverage(labels)
    if not split_dir.exists():
        return coverage

    target_set = set(labels)
    for path in sorted(split_dir.glob("scene_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as scene:
                scene_labels = scene["labels"].astype(np.int64, copy=False)
                target_label = int(scene["target_label"]) if "target_label" in scene.files else -1
        except Exception as exc:  # pragma: no cover - corrupt files are operational failures
            warnings.warn(f"Skipping unreadable scene {path}: {exc}")
            continue

        unique_labels, counts = np.unique(scene_labels, return_counts=True)
        count_by_label = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        # A quota is evidence only when the scene was deliberately generated
        # for that label.  Counting incidental co-occurrences could otherwise
        # make a rare class appear covered without independent instances.
        if target_label in target_set and count_by_label.get(target_label, 0) > 0:
            coverage[target_label]["scene_count"] += 1
            coverage[target_label]["instance_count"] += 1
            coverage[target_label]["point_count"] += count_by_label[target_label]

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
            scene_id=scene_id,
            target_label=target_label,
            feature_schema=scene["feature_schema"],
            feature_names=scene["feature_names"],
            feature_k_neighbors=scene["feature_k_neighbors"],
            coordinate_center=scene["coordinate_center"],
            coordinate_scale=scene["coordinate_scale"],
            coordinates_normalized=scene["coordinates_normalized"],
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
) -> dict[str, Any]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    coverage = _scan_coverage(split_dir, labels)
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

    pool_config = config
    if split == "val" and config.seed is not None:
        pool_config = replace(config, seed=config.seed + 100000)

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
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--labels", default="all", help="all, 1-37, or comma/range syntax such as 1-8,20,37")
    parser.add_argument("--target-scenes-per-class", type=int, default=20)
    parser.add_argument("--min-points-per-class", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-attempts-per-class", type=int, default=0, help="0 chooses a conservative automatic limit")
    parser.add_argument("--wave-size", type=int, default=4, help="maximum new scenes per incomplete class per scheduling wave")
    parser.add_argument("--workers", type=int, default=0, help="0 chooses min(8, cpu_count - 2); use 16 explicitly when appropriate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-res", type=float, default=0.01)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--pavement", choices=("asphalt", "concrete", "mixed"), default="mixed")
    parser.add_argument("--roughness", choices=("A", "B", "C", "D", "E"), default="B")
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

    labels = _parse_labels(args.labels)
    workers = _auto_workers() if args.workers <= 0 else args.workers
    max_attempts = args.max_attempts_per_class
    if max_attempts <= 0:
        max_attempts = max(args.target_scenes_per_class * 3, 20)

    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            grid_res=args.grid_res,
            pavement_type=args.pavement,
            roughness_class=args.roughness,
        ),
        disease=DiseaseConfig(max_diseases_per_scene=1),
        seed=args.seed,
        num_points=args.num_points,
    )
    output_dir = Path(args.output_dir)
    selected_splits = ("train", "val") if args.split == "both" else (args.split,)
    started = time.time()
    reports: dict[str, dict[str, Any]] = {}

    for split in selected_splits:
        if split == "val" and args.split == "both":
            target_scenes = max(1, int(np.ceil(args.target_scenes_per_class * args.val_ratio)))
            min_points = max(1, int(np.ceil(args.min_points_per_class * args.val_ratio)))
        else:
            target_scenes = args.target_scenes_per_class
            min_points = args.min_points_per_class
        reports[split] = _budget_split(
            output_dir=output_dir,
            split=split,
            config=config,
            labels=labels,
            target_scenes=target_scenes,
            min_points=min_points,
            workers=workers,
            max_attempts=max_attempts,
            wave_size=args.wave_size,
            dry_run=args.dry_run,
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
