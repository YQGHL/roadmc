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
    ASPHALT_LABELS,
    CONCRETE_LABELS,
    GeneratorConfig,
    RoadSurfaceConfig,
    NUM_CLASSES,
    LABEL_MAP,
)


def _get_label_info(label: int) -> Tuple[str, str]:
    """Get pavement type and severity from label."""
    info = LABEL_MAP.get(label, {})
    return info.get("pavement", "通用"), info.get("severity", "-")


def _sample_disease_label(
    rng: np.random.Generator,
    pavement_type: str,
    disease_probs: Dict[str, float],
) -> Tuple[str, str, int]:
    """Sample a disease label with stratified sampling.

    Args:
        rng: Random generator.
        pavement_type: 'asphalt', 'concrete', or 'mixed'.
        disease_probs: Disease probability dictionary.

    Returns:
        Tuple of (disease_type, severity, label_id).
    """
    # Build label pool based on pavement type
    if pavement_type == "asphalt":
        label_pool = list(ASPHALT_LABELS[1:])  # Skip background
    elif pavement_type == "concrete":
        label_pool = list(CONCRETE_LABELS)
    else:  # mixed
        label_pool = list(ASPHALT_LABELS[1:]) + list(CONCRETE_LABELS)

    # Map label to disease type
    label_to_disease = {}
    for lid, info in LABEL_MAP.items():
        if lid == 0:
            continue
        pav = info.get("pavement", "")
        dtype = info.get("type", "")
        sev = info.get("severity", "-")
        if pav == "沥青":
            key = _cn_to_en.get(dtype, dtype)
        elif pav == "水泥":
            key = _cn_to_en_concrete.get(dtype, dtype)
        else:
            continue
        label_to_disease[lid] = (key, sev)

    # Use disease probabilities to weight selection
    disease_weights = {}
    for lid in label_pool:
        if lid in label_to_disease:
            dtype, _ = label_to_disease[lid]
            disease_weights[lid] = disease_probs.get(dtype, 0.1)

    if not disease_weights or sum(disease_weights.values()) == 0:
        # Fallback to uniform distribution
        weights = np.ones(len(label_pool)) / len(label_pool)
    else:
        weights = np.array([disease_weights.get(lid, 0.01) for lid in label_pool])
        weights = weights / weights.sum()

    selected_label = rng.choice(label_pool, p=weights)
    disease_type, severity = label_to_disease.get(selected_label, ("unknown", "-"))

    return disease_type, severity, selected_label


# Chinese to English disease type mapping (for disease_probs lookup)
_cn_to_en = {
    "龟裂": "alligator",
    "块状裂缝": "block",
    "纵向裂缝": "longitudinal",
    "横向裂缝": "transverse",
    "坑槽": "pothole",
    "松散": "raveling",
    "沉陷": "depression",
    "车辙": "rutting",
    "波浪拥包": "corrugation",
    "泛油": "bleeding",
}

_cn_to_en_concrete = {
    "破碎板": "slab_shatter",
    "裂缝": "slab_crack",
    "板角断裂": "corner_break",
    "错台": "faulting",
    "唧泥": "pumping",
    "边角剥落": "edge_spall",
    "接缝料损坏": "joint_damage",
    "坑洞": "pitting",
    "拱起": "blowup",
    "露骨": "exposed_aggregate",
    "修补": "patching",
}


def generate_single_scene(
    scene_id: int,
    config: GeneratorConfig,
    pavement_type: str,
    rng: np.random.Generator,
) -> Optional[Dict]:
    """Generate a single scene.

    Args:
        scene_id: Scene identifier.
        config: Generator configuration.
        pavement_type: Pavement type for this scene.
        rng: Random generator.

    Returns:
        Dictionary with scene data or None if generation failed.
    """
    try:
        from roadmc.data.synthetic.generator import SyntheticRoadDataset
    except ImportError:
        warnings.warn("SyntheticRoadDataset not available - using fallback generation")
        return None

    try:
        # Create dataset instance for this scene
        dataset = SyntheticRoadDataset(config=config, split="train")

        # Generate base surface
        points, normals, labels, feats = dataset.generate_sample(
            pavement_type=pavement_type,
            seed=rng.integers(0, 2**31) if config.seed is None else config.seed + scene_id,
        )

        return {
            "points": points,
            "normals": normals,
            "labels": labels,
            "feats": feats,
            "pavement_type": pavement_type,
            "scene_id": scene_id,
        }
    except Exception as e:
        warnings.warn(f"Scene {scene_id} generation failed: {e}")
        return None


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


def _apply_disease(
    points: np.ndarray,
    labels: np.ndarray,
    disease_type: str,
    severity: str,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a specific disease to the point cloud.

    Args:
        points: Point cloud (N, 3).
        labels: Labels (N,).
        disease_type: Disease type string.
        severity: 'light' or 'severe'.
        config: Generator configuration.
        rng: Random generator.

    Returns:
        Modified points and labels.
    """
    from roadmc.data.synthetic import primitives as prim

    pts = points.copy()
    lbl = labels.copy()

    # Disease-specific parameters
    if disease_type in ("alligator", "block", "longitudinal", "transverse"):
        crack_params = {
            "d_max": 0.010 if severity == "light" else 0.030,
            "width_mean": 0.005,
            "width_std": 0.3,
        }
        pts, lbl = prim.add_crack(pts, lbl, disease_type, severity, crack_params, seed=rng.integers(0, 2**31))

    elif disease_type == "pothole":
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        center = (
            rng.uniform(x_min + 0.2, x_max - 0.2),
            rng.uniform(y_min + 0.2, y_max - 0.2),
        )
        radius = rng.uniform(0.10, 0.25)
        depth = rng.uniform(0.015, 0.040) if severity == "light" else rng.uniform(0.040, 0.080)
        edge_quality = rng.uniform(0.5, 1.0)
        pts, lbl = prim.add_pothole(pts, lbl, center, radius, depth, edge_quality, severity, seed=rng.integers(0, 2**31))

    elif disease_type == "raveling":
        # Use a random rectangular region
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        region_mask = np.ones(len(pts), dtype=bool)
        # Random sub-region
        mask_x = (pts[:, 0] > rng.uniform(x_min, x_max * 0.7)) & (pts[:, 0] < rng.uniform(x_max * 0.3, x_max))
        mask_y = (pts[:, 1] > rng.uniform(y_min, y_max * 0.7)) & (pts[:, 1] < rng.uniform(y_max * 0.3, y_max))
        region_mask = mask_x & mask_y
        pts, lbl = prim.add_raveling(pts, lbl, region_mask, severity, seed=rng.integers(0, 2**31))

    elif disease_type == "depression":
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        center = (
            rng.uniform(x_min + 0.3, x_max - 0.3),
            rng.uniform(y_min + 0.3, y_max - 0.3),
        )
        radius = rng.uniform(0.5, 1.5)
        depth = rng.uniform(0.015, 0.030) if severity == "light" else rng.uniform(0.030, 0.060)
        pts, lbl = prim.add_depression(pts, lbl, center, radius, depth, severity, seed=rng.integers(0, 2**31))

    elif disease_type == "rutting":
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        center_line = (x_min + x_max) / 2.0
        depth = rng.uniform(0.010, 0.020) if severity == "light" else rng.uniform(0.020, 0.050)
        pts, lbl = prim.add_rutting(
            pts, lbl, center_line, config.rutting.wheel_separation, depth, config.rutting.rut_width, severity, seed=rng.integers(0, 2**31)
        )

    elif disease_type == "corrugation":
        direction = rng.choice(["longitudinal", "transverse"])
        wavelength = rng.uniform(*config.corrugation.wavelength_range)
        amplitude = rng.uniform(*(config.corrugation.amplitude_light if severity == "light" else config.corrugation.amplitude_severe))
        pts, lbl = prim.add_corrugation(pts, lbl, direction, wavelength, amplitude, severity, seed=rng.integers(0, 2**31))

    elif disease_type == "bleeding":
        # Random region for bleeding
        region_mask = rng.choice([True, False], size=len(pts))
        pts, lbl = prim.add_bleeding(pts, lbl, region_mask, seed=rng.integers(0, 2**31))

    elif disease_type in ("slab_shatter", "slab_crack", "corner_break", "faulting", "pumping", "edge_spall", "joint_damage", "pitting", "blowup", "exposed_aggregate"):
        concrete_params = {
            "slab_length": config.concrete_damage.slab_length,
            "slab_width": config.concrete_damage.slab_width,
            "joint_width": config.concrete_damage.joint_width,
        }
        pts, lbl = prim.add_concrete_damage(pts, lbl, disease_type, severity, concrete_params, seed=rng.integers(0, 2**31))

    return pts, lbl


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
            data = np.load(npz_file, allow_dickle=True)
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