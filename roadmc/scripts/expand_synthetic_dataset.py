"""Resume-friendly synthetic dataset expansion.

This script extends an existing RoadMC synthetic dataset to a target size without
overwriting existing scenes. It is intended for long multi-process generation on
Windows, where generation may be interrupted and resumed later.
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
from typing import Iterable

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.synthetic.config import DiseaseConfig, GeneratorConfig, RoadSurfaceConfig

_WORKER_DATASET = None
_WORKER_SPLIT_DIR: Path | None = None


def _scene_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"scene_(\d+)\.npz", path.name)
    return int(match.group(1)) if match else None


def _existing_scene_ids(split_dir: Path) -> set[int]:
    ids: set[int] = set()
    if not split_dir.exists():
        return ids
    for path in split_dir.glob("scene_*.npz"):
        scene_id = _scene_id_from_path(path)
        if scene_id is not None:
            ids.add(scene_id)
    return ids


def _next_scene_ids(split_dir: Path, needed: int) -> list[int]:
    existing = _existing_scene_ids(split_dir)
    scene_ids: list[int] = []
    candidate = (max(existing) + 1) if existing else 0
    while len(scene_ids) < needed:
        if candidate not in existing:
            scene_ids.append(candidate)
        candidate += 1
    return scene_ids


def _init_worker(config: GeneratorConfig, split_dir: str) -> None:
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir)


def _save_scene(scene_id: int) -> dict:
    if _WORKER_DATASET is None or _WORKER_SPLIT_DIR is None:
        raise RuntimeError("worker was not initialized")

    out_path = _WORKER_SPLIT_DIR / f"scene_{scene_id:04d}.npz"
    result = {
        "scene_id": scene_id,
        "ok": False,
        "skipped": False,
        "npoints": 0,
        "labels": [],
    }
    if out_path.exists():
        result["ok"] = True
        result["skipped"] = True
        return result

    try:
        scene = _WORKER_DATASET.generate_scene(scene_id)
        points = scene["points"]
        labels = scene["labels"]
        feats = scene["feats"]
        normals = scene["normals"]
        pavement_type = scene["pavement_type"]

        valid = ~np.any(np.isnan(points), axis=1)
        points = points[valid]
        labels = labels[valid]
        feats = feats[valid]
        normals = normals[valid]
        if len(points) == 0:
            return result

        np.savez_compressed(
            out_path,
            points=points.astype(np.float32),
            labels=labels.astype(np.int32),
            feats=feats.astype(np.float32),
            normals=normals.astype(np.float32),
            pavement_type=pavement_type,
            scene_id=scene_id,
            feature_schema=scene["feature_schema"],
            feature_names=scene["feature_names"],
            feature_k_neighbors=scene["feature_k_neighbors"],
            coordinate_center=scene["coordinate_center"],
            coordinate_scale=scene["coordinate_scale"],
            coordinates_normalized=scene["coordinates_normalized"],
        )
        result["ok"] = True
        result["npoints"] = int(len(points))
        result["labels"] = [int(x) for x in np.unique(labels)]
    except Exception as exc:  # pragma: no cover - best-effort long job logging
        warnings.warn(f"scene {scene_id} failed: {exc}")
    return result


def _iter_results(pool: mp.pool.Pool, scene_ids: Iterable[int], chunksize: int):
    try:
        from tqdm import tqdm

        return tqdm(
            pool.imap_unordered(_save_scene, scene_ids, chunksize=chunksize),
            total=len(scene_ids),  # type: ignore[arg-type]
            desc="generating",
        )
    except ImportError:
        return pool.imap_unordered(_save_scene, scene_ids, chunksize=chunksize)


def _expand_split(
    output_dir: Path,
    split: str,
    target_count: int,
    config: GeneratorConfig,
    workers: int,
    chunksize: int,
    dry_run: bool,
) -> dict:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = len(_existing_scene_ids(split_dir))
    needed = max(0, target_count - existing)
    scene_ids = _next_scene_ids(split_dir, needed)

    print(f"[{split}] existing={existing} target={target_count} needed={needed}")
    if scene_ids:
        print(f"[{split}] next ids: {scene_ids[0]}..{scene_ids[-1]}")
    if dry_run or needed == 0:
        return {
            "split": split,
            "existing_before": existing,
            "target": target_count,
            "requested_new": needed,
            "generated": 0,
            "failed": 0,
            "skipped": 0,
            "point_stats": {},
        }

    split_config = config
    if split == "val" and config.seed is not None:
        split_config = replace(config, seed=config.seed + 100000)

    generated = 0
    failed = 0
    skipped = 0
    point_counts: list[int] = []
    class_counts: dict[int, int] = {}
    start = time.time()

    with mp.Pool(workers, initializer=_init_worker, initargs=(split_config, str(split_dir))) as pool:
        for res in _iter_results(pool, scene_ids, chunksize):
            if res["ok"] and not res["skipped"]:
                generated += 1
                point_counts.append(res["npoints"])
                for label in res["labels"]:
                    class_counts[label] = class_counts.get(label, 0) + 1
            elif res["skipped"]:
                skipped += 1
            else:
                failed += 1

    elapsed = time.time() - start
    total_after = len(_existing_scene_ids(split_dir))
    return {
        "split": split,
        "existing_before": existing,
        "target": target_count,
        "requested_new": needed,
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
        "total_after": total_after,
        "elapsed_sec": elapsed,
        "scenes_per_sec": generated / elapsed if elapsed > 0 else 0.0,
        "point_stats": {
            "mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "min": int(np.min(point_counts)) if point_counts else 0,
            "max": int(np.max(point_counts)) if point_counts else 0,
        },
        "new_scene_class_counts": class_counts,
    }


def _auto_workers() -> int:
    cpu_count = mp.cpu_count()
    return max(1, min(8, cpu_count - 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand RoadMC synthetic dataset to a target size.")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_output")
    parser.add_argument("--target-total", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0, help="0 means auto, currently min(8, cpu_count - 2)")
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-res", type=float, default=0.01)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--target-density", type=float, default=None)
    parser.add_argument("--pavement", choices=["asphalt", "concrete", "mixed"], default="mixed")
    parser.add_argument("--roughness", choices=["A", "B", "C", "D", "E"], default="B")
    parser.add_argument("--max-diseases", type=int, default=3)
    parser.add_argument("--no-stratified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")

    train_count = args.train_count
    val_count = args.val_count
    if train_count is None or val_count is None:
        train_count = int(args.target_total * args.train_ratio)
        val_count = args.target_total - train_count

    workers = _auto_workers() if args.workers <= 0 else args.workers
    output_dir = Path(args.output_dir)
    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            grid_res=args.grid_res,
            roughness_class=args.roughness,
            pavement_type=args.pavement,
        ),
        disease=DiseaseConfig(
            max_diseases_per_scene=args.max_diseases,
            use_stratified=not args.no_stratified,
        ),
        seed=args.seed,
        num_points=args.num_points,
        target_density=args.target_density,
    )

    print("RoadMC dataset expansion")
    print(f"output_dir={output_dir}")
    print(f"target train={train_count}, val={val_count}, total={train_count + val_count}")
    print(f"workers={workers}, chunksize={args.chunksize}, dry_run={args.dry_run}")
    print(
        "config="
        f"grid_res={args.grid_res}, num_points={args.num_points}, "
        f"target_density={args.target_density}, pavement={args.pavement}, "
        f"roughness={args.roughness}, max_diseases={args.max_diseases}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "train": _expand_split(
            output_dir, "train", train_count, config, workers, args.chunksize, args.dry_run
        ),
        "val": _expand_split(
            output_dir, "val", val_count, config, workers, args.chunksize, args.dry_run
        ),
    }

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "script": "roadmc/scripts/expand_synthetic_dataset.py",
        "dry_run": args.dry_run,
        "target": {
            "train": train_count,
            "val": val_count,
            "total": train_count + val_count,
        },
        "config": {
            "seed": args.seed,
            "grid_res": args.grid_res,
            "num_points": args.num_points,
            "target_density": args.target_density,
            "pavement": args.pavement,
            "roughness": args.roughness,
            "max_diseases": args.max_diseases,
            "use_stratified": not args.no_stratified,
            "workers": workers,
            "chunksize": args.chunksize,
        },
        "stats": stats,
    }

    metadata_path = output_dir / "expansion_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nExpansion summary")
    for split, split_stats in stats.items():
        print(
            f"{split}: before={split_stats['existing_before']} "
            f"generated={split_stats['generated']} failed={split_stats['failed']} "
            f"total_after={split_stats.get('total_after', split_stats['existing_before'])}"
        )
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
