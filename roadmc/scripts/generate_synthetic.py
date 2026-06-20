"""Batch generator for synthetic road point cloud datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.synthetic.config import DiseaseConfig, GeneratorConfig, LABEL_MAP, NUM_CLASSES, RoadSurfaceConfig

_WORKER_DATASET = None
_WORKER_SPLIT_DIR = None


def _init_worker(config: GeneratorConfig, split_dir_str: str) -> None:
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir_str)


def _save_one_scene(dataset, split_dir: Path, scene_id: int) -> dict:
    result = {"scene_id": scene_id, "ok": False, "npoints": 0, "labels": []}
    try:
        scene = dataset.generate_scene(scene_id)
        pts = scene["points"]
        lbls = scene["labels"]
        feats = scene["feats"]
        normals = scene["normals"]
        pavement_type = scene["pavement_type"]

        valid = ~np.any(np.isnan(pts), axis=1)
        pts = pts[valid]
        lbls = lbls[valid]
        feats = feats[valid]
        normals = normals[valid]
        if len(pts) == 0:
            return result

        np.savez_compressed(
            split_dir / f"scene_{scene_id:04d}.npz",
            points=pts.astype(np.float32),
            labels=lbls.astype(np.int32),
            feats=feats.astype(np.float32),
            normals=normals.astype(np.float32),
            pavement_type=pavement_type,
            scene_id=scene_id,
        )
        result["ok"] = True
        result["npoints"] = int(len(pts))
        result["labels"] = list(np.unique(lbls).astype(int))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Scene {scene_id} failed: {exc}")
    return result


def _worker_save(scene_id: int) -> dict:
    return _save_one_scene(_WORKER_DATASET, _WORKER_SPLIT_DIR, scene_id)


def generate_dataset(
    count: int,
    config: GeneratorConfig,
    split: str,
    output_dir: Path,
    use_stratified: bool = True,
    num_workers: int = 1,
) -> Dict:
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    scene_config = config
    if split == "val" and config.seed is not None:
        scene_config = replace(config, seed=config.seed + 100000)
    scene_config = replace(
        scene_config,
        disease=replace(scene_config.disease, use_stratified=use_stratified),
    )

    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    point_counts: list[int] = []
    failed = 0
    scene_ids = list(range(count))

    if num_workers <= 1:
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
        print(f"并行生成 {split} 集 ({count} 个场景, {num_workers} workers)...")
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(scene_config, str(split_dir))) as pool:
            try:
                from tqdm import tqdm

                iterator = tqdm(pool.imap_unordered(_worker_save, scene_ids), total=count, desc=f"生成 {split} 集")
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

    return {
        "split": split,
        "requested": count,
        "generated": count - failed,
        "failed": failed,
        "class_distribution": class_counts,
        "point_stats": {
            "total": int(sum(point_counts)) if point_counts else 0,
            "mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "min": int(np.min(point_counts)) if point_counts else 0,
            "max": int(np.max(point_counts)) if point_counts else 0,
        },
    }


def verify_class_distribution(output_dir: Path, split: str) -> Dict[int, int]:
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
            for lbl in np.unique(labels):
                if int(lbl) in class_counts:
                    class_counts[int(lbl)] += 1
            total_scenes += 1
        except Exception:
            continue

    for label_id in range(NUM_CLASSES):
        info = LABEL_MAP.get(label_id, {})
        dtype = info.get("type", "未知")
        severity = info.get("severity", "-")
        count = class_counts[label_id]
        pct = (count / total_scenes * 100) if total_scenes else 0
        print(f"{label_id:<6} {dtype:<12} {severity:<6} {count:<8} {pct:.1f}%")

    print(f"\n总计: {total_scenes} 个场景")
    return class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="RoadMC 合成点云数据集批量生成")
    parser.add_argument("--train-count", type=int, default=2000, help="训练集场景数 (默认 2000)")
    parser.add_argument("--val-count", type=int, default=500, help="验证集场景数 (默认 500)")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--grid-res", type=float, default=0.01, help="网格分辨率 (米)")
    parser.add_argument("--num-points", type=int, default=65536, help="每个场景的目标点数")
    parser.add_argument("--target-density", type=float, default=None, help="按路面面积指定目标点密度，设置后优先于 num_points")
    parser.add_argument("--pavement", type=str, default="mixed", choices=["asphalt", "concrete", "mixed"], help="路面类型")
    parser.add_argument("--roughness", type=str, default="B", choices=["A", "B", "C", "D", "E"], help="ISO 8608 粗糙度等级")
    parser.add_argument("--max-diseases", type=int, default=3, help="每场景最多病害数")
    parser.add_argument("--no-stratified", action="store_true", help="禁用分层采样")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    args = parser.parse_args()

    pavement_for_config = args.pavement

    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            grid_res=args.grid_res,
            roughness_class=args.roughness,
            pavement_type=pavement_for_config,
        ),
        disease=DiseaseConfig(
            max_diseases_per_scene=args.max_diseases,
            use_stratified=not args.no_stratified,
        ),
        seed=args.seed,
        num_points=args.num_points,
        target_density=args.target_density,
    )

    output_dir = Path(args.output_dir)
    print("\n开始生成数据集...")
    print(f"配置: 网格分辨率={args.grid_res}m, 路面类型={args.pavement}, 粗糙度={args.roughness}, 每场景最多病害={args.max_diseases}")
    print(f"随机种子: {args.seed}")

    stats = {
        "train": generate_dataset(args.train_count, config, "train", output_dir, use_stratified=not args.no_stratified, num_workers=args.workers),
        "val": generate_dataset(args.val_count, config, "val", output_dir, use_stratified=not args.no_stratified, num_workers=args.workers),
    }

    print("\n" + "=" * 60)
    class_distribution = {split: verify_class_distribution(output_dir, split) for split in ("train", "val")}

    metadata = {
        "config": {
            "grid_res": args.grid_res,
            "pavement": args.pavement,
            "num_points": args.num_points,
            "target_density": args.target_density,
            "roughness": args.roughness,
            "seed": args.seed,
            "max_diseases": args.max_diseases,
            "num_classes": NUM_CLASSES,
            "use_stratified": not args.no_stratified,
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
        "class_distribution": class_distribution,
    }

    metadata_file = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== 生成完成 ===")
    print(f"训练集: {stats['train']['generated']} 个场景 ({stats['train']['failed']} 个失败)")
    print(f"验证集: {stats['val']['generated']} 个场景 ({stats['val']['failed']} 个失败)")
    print(f"输出目录: {output_dir}")
    print(f"元数据: {metadata_file}")


if __name__ == "__main__":
    main()
