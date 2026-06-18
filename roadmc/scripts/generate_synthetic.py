"""批量生成合成路面点云数据集。

用法:
    python -m roadmc.scripts.generate_synthetic --train-count 2000 --val-count 500 --output-dir ./data/synthetic_output
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import multiprocessing as mp
from dataclasses import replace

from roadmc.data.synthetic.config import (
    GeneratorConfig,
    RoadSurfaceConfig,
    NUM_CLASSES,
    LABEL_MAP,
)

# ── Multiprocessing worker globals ──────────────────────────────────────────
_WORKER_DATASET = None       # type: ignore
_WORKER_SPLIT_DIR = None     # type: ignore


def _init_worker(config: GeneratorConfig, split_dir_str: str) -> None:
    """Called once per worker process — initialises a global dataset."""
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset
    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir_str)


def _save_one_scene(dataset, split_dir: Path, scene_id: int) -> dict:
    """Generate one scene and save .npz.  Used by both sequential & parallel paths."""
    result = {"scene_id": scene_id, "ok": False, "npoints": 0, "labels": []}
    try:
        scene = dataset.generate_scene(scene_id)
        pts, lbls, feats, nrm = scene["points"], scene["labels"], scene["feats"], scene["normals"]
        ptype = scene["pavement_type"]

        valid = ~np.any(np.isnan(pts), axis=1)
        pts, nrm, lbls, feats = pts[valid], nrm[valid], lbls[valid], feats[valid]
        if len(pts) == 0:
            return result

        result["ok"] = True
        result["npoints"] = len(pts)
        result["labels"] = list(np.unique(lbls).astype(int))

        np.savez_compressed(
            split_dir / f"scene_{scene_id:04d}.npz",
            points=pts.astype(np.float32),
            labels=lbls.astype(np.int32),
            feats=feats.astype(np.float32),
            normals=nrm.astype(np.float32),
            pavement_type=ptype,
            scene_id=scene_id,
        )
    except Exception as e:
        warnings.warn(f"Scene {scene_id} failed: {e}")
    return result


# ── Multiprocessing worker wrappers (module-level for pickling) ──────────────
_WORKER_DATASET = None
_WORKER_SPLIT_DIR = None


def _init_worker(config: GeneratorConfig, split_dir_str: str) -> None:
    """Called once per worker process — initialises a global dataset."""
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset
    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir_str)


def _worker_save(scene_id: int) -> dict:
    """Multiprocessing worker: generate + save one scene via globals."""
    return _save_one_scene(_WORKER_DATASET, _WORKER_SPLIT_DIR, scene_id)


def generate_dataset(
    count: int,
    config: GeneratorConfig,
    split: str,
    output_dir: Path,
    use_stratified: bool = True,
    num_workers: int = 1,
) -> Dict:
    """Generate and save a dataset split.

    When **num_workers > 1**, scenes are generated in parallel using
    ``multiprocessing.Pool`` — each worker process handles one scene at a time.
    Scene-level RNG is deterministic because seeds are derived from
    ``config.seed + scene_id``.

    Args:
        count: Number of scenes to generate.
        config: Generator configuration.
        split: 'train' or 'val'.
        output_dir: Output directory.
        use_stratified: Use stratified sampling for disease types.
        num_workers: Number of worker processes.  Default 1 (sequential).

    Returns:
        dict with generation statistics.
    """
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Override seed for val split to avoid overlap with train
    scene_config = config
    if split == "val" and config.seed is not None:
        scene_config = replace(config, seed=config.seed + 100000)

    class_counts: dict = {i: 0 for i in range(NUM_CLASSES)}
    point_counts: list[int] = []
    failed = 0

    scene_ids = list(range(count))

    if num_workers <= 1:
        # ── Sequential path ──
        dataset = SyntheticRoadDataset(config=scene_config, dataset_size=count)
        try:
            from tqdm import tqdm
            iterator = tqdm(scene_ids, desc=f"生成 {split} 集")
        except ImportError:
            print(f"生成 {split} 集 ({count} 个场景)...")
            iterator = scene_ids

        for sid in iterator:
            res = _save_one_scene(dataset, split_dir, sid)
            if res["ok"]:
                point_counts.append(res["npoints"])
                for lbl in res["labels"]:
                    if lbl in class_counts:
                        class_counts[lbl] += 1
            else:
                failed += 1
    else:
        # ── Parallel path ──
        print(f"并行生成 {split} 集 ({count} 场景, {num_workers} workers)...")
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(scene_config, str(split_dir)),
        ) as pool:
            try:
                from tqdm import tqdm
                iterator = tqdm(pool.imap_unordered(_worker_save, scene_ids),
                                total=count, desc=f"生成 {split} 集")
            except ImportError:
                iterator = pool.imap_unordered(_worker_save, scene_ids)

            for res in iterator:
                if res["ok"]:
                    point_counts.append(res["npoints"])
                    for lbl in res["labels"]:
                        if lbl in class_counts:
                            class_counts[lbl] += 1
                else:
                    failed += 1

    total_points = sum(point_counts) if point_counts else 0
    stats = {
        "split": split,
        "requested": count,
        "generated": count - failed,
        "failed": failed,
        "class_distribution": class_counts,
        "point_stats": {
            "total": total_points,
            "mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "min": int(np.min(point_counts)) if point_counts else 0,
            "max": int(np.max(point_counts)) if point_counts else 0,
        },
    }
    return stats


def verify_class_distribution(output_dir: Path, split: str) -> Dict[int, int]:
    """Verify class distribution in generated data.

    Args:
        output_dir: Output directory.
        split: 'train' or 'val'.

    Returns:
        Dictionary mapping label -> sample count.
    """
    split_dir = output_dir / split
    if not split_dir.exists():
        print(f"[WARN] {split} 目录不存在: {split_dir}")
        return {}

    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    total_scenes = 0

    print(f"\n=== {split.upper()} 类别分布 ===")
    print(f"{'标签':<6} {'类型':<12} {'严重度':<6} {'场景数':<8} {'占比':<8}")
    print("-" * 50)

    for npz_file in sorted(split_dir.glob("scene_*.npz")):
        try:
            data = np.load(npz_file, allow_pickle=True)
            labels = data["labels"]
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                if lbl in class_counts:
                    class_counts[lbl] += 1
            total_scenes += 1
        except Exception:
            continue

    for label_id in range(NUM_CLASSES):
        info = LABEL_MAP.get(label_id, {})
        dtype = info.get("type", "未知")
        severity = info.get("severity", "-")
        count = class_counts[label_id]
        pct = (count / total_scenes * 100) if total_scenes > 0 else 0
        print(f"{label_id:<6} {dtype:<12} {severity:<6} {count:<8} {pct:.1f}%")

    print(f"\n总计: {total_scenes} 个场景")

    return class_counts


def main():
    parser = argparse.ArgumentParser(description="RoadMC 合成点云数据集批量生成")
    parser.add_argument("--train-count", type=int, default=2000, help="训练集场景数 (默认 2000)")
    parser.add_argument("--val-count", type=int, default=500, help="验证集场景数 (默认 500)")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--grid-res", type=float, default=0.01, help="网格分辨率 (米)，默认 0.01 (10mm)")
    parser.add_argument("--pavement", type=str, default="mixed", choices=["asphalt", "concrete", "mixed"], help="路面类型")
    parser.add_argument("--roughness", type=str, default="B", choices=["A", "B", "C", "D", "E"], help="ISO 8608 粗糙度等级")
    parser.add_argument("--max-diseases", type=int, default=3, help="每场景最多病害数 (默认 3)")
    parser.add_argument("--no-stratified", action="store_true", help="禁用分层采样")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数 (默认 1=串行)")
    args = parser.parse_args()

    pavement_for_config = args.pavement if args.pavement != "mixed" else "asphalt"
    from roadmc.data.synthetic.config import DiseaseConfig

    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            grid_res=args.grid_res,
            roughness_class=args.roughness,
            pavement_type=pavement_for_config,
        ),
        disease=DiseaseConfig(max_diseases_per_scene=args.max_diseases),
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)

    print("\n开始生成数据集...")
    print(f"配置: 网格分辨率={args.grid_res}m, 路面类型={args.pavement}, "
          f"粗糙度={args.roughness}, 每场景最多病害={args.max_diseases}")
    print(f"随机种子: {args.seed}")

    use_stratified = not args.no_stratified

    stats = {}
    stats["train"] = generate_dataset(args.train_count, config, "train", output_dir,
                                       use_stratified=use_stratified, num_workers=args.workers)
    stats["val"] = generate_dataset(args.val_count, config, "val", output_dir,
                                     use_stratified=use_stratified, num_workers=args.workers)

    print("\n" + "=" * 60)
    for split in ["train", "val"]:
        verify_class_distribution(output_dir, split)

    metadata = {
        "config": {
            "grid_res": args.grid_res,
            "pavement": args.pavement,
            "roughness": args.roughness,
            "seed": args.seed,
            "max_diseases": args.max_diseases,
            "num_classes": NUM_CLASSES,
        },
        "generated_at": datetime.now().isoformat(),
        "train_count": args.train_count,
        "val_count": args.val_count,
        "statistics": {
            split: {
                "requested": s["requested"],
                "generated": s["generated"],
                "failed": s["failed"],
                "point_stats": s["point_stats"],
            }
            for split, s in stats.items()
        },
        "class_distribution": {
            split: verify_class_distribution(output_dir, split)
            for split in ["train", "val"]
        },
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== 生成完成 ===")
    print(f"训练集: {stats['train']['generated']} 个场景 ({stats['train']['failed']} 个失败)")
    print(f"验证集: {stats['val']['generated']} 个场景 ({stats['val']['failed']} 个失败)")
    print(f"输出目录: {output_dir}")
    print(f"元数据: {metadata_file}")


if __name__ == "__main__":
    main()