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

from roadmc.data.synthetic.config import (
    GeneratorConfig,
    RoadSurfaceConfig,
    NUM_CLASSES,
    LABEL_MAP,
)


def _get_label_info(label: int) -> Tuple[str, str]:
    """Get pavement type and severity from label."""
    info = LABEL_MAP.get(label, {})
    return info.get("pavement", "通用"), info.get("severity", "-")


def generate_dataset(
    count: int,
    config: GeneratorConfig,
    split: str,
    output_dir: Path,
    use_stratified: bool = True,
) -> Dict:
    """Generate and save a dataset split.

    Uses SyntheticRoadDataset.generate_scene() to generate each scene,
    then saves to .npz files.

    Args:
        count: Number of scenes to generate.
        config: Generator configuration.
        split: 'train' or 'val'.
        output_dir: Output directory.
        use_stratified: Use stratified sampling for disease types.

    Returns:
        dict with generation statistics.
    """
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    # Create output directory
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracking structures
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    point_counts = []
    failed_count = 0

    # Override seed for val split to avoid overlap with train
    scene_config = config
    if split == "val" and config.seed is not None:
        from dataclasses import replace
        scene_config = replace(config, seed=config.seed + 100000)

    # Create dataset instance
    dataset = SyntheticRoadDataset(config=scene_config, dataset_size=count)

    # Try to use tqdm, fallback to simple print
    try:
        from tqdm import tqdm
        iterator = tqdm(range(count), desc=f"生成 {split} 集")
    except ImportError:
        print(f"生成 {split} 集 ({count} 个场景)...")
        iterator = range(count)

    for i in iterator:
        scene_id = i

        try:
            # Generate scene using the dataset's generate_scene method
            scene = dataset.generate_scene(scene_id)

            # Extract arrays
            points = scene["points"]
            labels = scene["labels"]
            feats = scene["feats"]
            normals = scene["normals"]
            pavement_type = scene["pavement_type"]

            # Filter out any remaining NaN points (shouldn't happen but safety check)
            valid_mask = ~np.any(np.isnan(points), axis=1)
            points = points[valid_mask]
            normals = normals[valid_mask]
            labels = labels[valid_mask]
            feats = feats[valid_mask]

            if len(points) == 0:
                failed_count += 1
                continue

            # Update statistics
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                lbl_int = int(lbl)
                if lbl_int in class_counts:
                    class_counts[lbl_int] += 1
            point_counts.append(len(points))

            # Save scene as .npz
            filename = f"scene_{scene_id:04d}.npz"
            filepath = split_dir / filename

            np.savez_compressed(
                filepath,
                points=points.astype(np.float32),
                labels=labels.astype(np.int32),
                feats=feats.astype(np.float32),
                normals=normals.astype(np.float32),
                pavement_type=pavement_type,
                scene_id=scene_id,
            )

        except Exception as e:
            failed_count += 1
            warnings.warn(f"Scene {scene_id} failed: {e}")
            continue

    # Compute statistics
    total_points = sum(point_counts) if point_counts else 0
    stats = {
        "split": split,
        "requested": count,
        "generated": count - failed_count,
        "failed": failed_count,
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

    Prints statistics about how many samples contain each disease type.

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
    parser.add_argument("--no-stratified", action="store_true", help="禁用分层采样")
    args = parser.parse_args()

    # Create config
    pavement_for_config = args.pavement if args.pavement != "mixed" else "asphalt"
    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            grid_res=args.grid_res,
            roughness_class=args.roughness,
            pavement_type=pavement_for_config,
        ),
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)

    # Generate training set
    print("\n开始生成数据集...")
    print(f"配置: 网格分辨率={args.grid_res}m, 路面类型={args.pavement}, 粗糙度={args.roughness}")
    print(f"随机种子: {args.seed}")

    use_stratified = not args.no_stratified

    stats = {}
    stats["train"] = generate_dataset(args.train_count, config, "train", output_dir, use_stratified=use_stratified)
    stats["val"] = generate_dataset(args.val_count, config, "val", output_dir, use_stratified=use_stratified)

    # Verify class distribution
    print("\n" + "=" * 60)
    for split in ["train", "val"]:
        verify_class_distribution(output_dir, split)

    # Save metadata
    metadata = {
        "config": {
            "grid_res": args.grid_res,
            "pavement": args.pavement,
            "roughness": args.roughness,
            "seed": args.seed,
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