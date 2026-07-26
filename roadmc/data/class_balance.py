"""Class-frequency accounting and bounded effective-number loss weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roadmc.data.curriculum import label_lut, normalize_label_stage, num_classes_for_stage


def point_class_counts(data_dir: str | Path, *, split: str, label_stage: str) -> np.ndarray:
    """Count mapped training points without sampling or augmentation.

    Counts are taken from the source files rather than batches so the class
    weights have a stable, inspectable denominator independent of worker order
    or stochastic point subsampling.
    """

    stage = normalize_label_stage(label_stage)
    paths = sorted((Path(data_dir) / split).glob("scene_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No scene_*.npz files in {Path(data_dir) / split}")
    mapping = np.asarray(label_lut(stage), dtype=np.int64)
    counts = np.zeros(num_classes_for_stage(stage), dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as scene:
            if "labels" not in scene.files:
                raise ValueError(f"{path} is missing labels")
            labels = np.asarray(scene["labels"], dtype=np.int64).reshape(-1)
        if np.any((labels < 0) | (labels >= len(mapping))):
            raise ValueError(f"{path} contains labels outside [0, {len(mapping) - 1}]")
        counts += np.bincount(mapping[labels], minlength=len(counts))
    return counts


def effective_number_class_weights(
    counts: np.ndarray,
    *,
    beta: float = 0.999999,
    max_weight: float = 5.0,
) -> np.ndarray:
    """Compute bounded class-balanced weights from the effective sample count.

    For class count ``n_c`` the unnormalized weight is
    ``(1-beta)/(1-beta**n_c)``.  It interpolates smoothly between equal class
    treatment and inverse frequency, while the cap prevents a noisy rare class
    from dominating a batch.  Supported weights are normalized to mean one.
    """

    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("counts must be a finite, non-negative non-empty vector")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")

    supported = values > 0.0
    weights = np.zeros_like(values)
    if not supported.any():
        return weights.astype(np.float32)
    if beta == 0.0:
        weights[supported] = 1.0
    else:
        # -expm1(n log beta) is stable when beta is close to one.
        denominator = -np.expm1(values[supported] * np.log(beta))
        weights[supported] = (1.0 - beta) / np.maximum(denominator, 1e-12)
    weights[supported] /= weights[supported].mean()
    # clip 必须是最终生效语义："归一 → clip → 再归一" 会把被 clip 的
    # 权重放大回 max_weight 之上（38 类极端不平衡下实测可达上界的
    # ~3 倍）。迭代 clip/归一到不动点后以一次硬 clip 收尾，保证
    # max(weights) ≤ max_weight 严格成立（均值可能略低于 1，属文档
    # 声明的取舍）。
    for _ in range(8):
        clipped = np.minimum(weights[supported], max_weight)
        mean = clipped.mean()
        if mean <= 0:
            break
        renormed = clipped / mean
        if np.all(renormed <= max_weight + 1e-9):
            weights[supported] = renormed
            break
        weights[supported] = renormed
    weights[supported] = np.minimum(weights[supported], max_weight)
    return weights.astype(np.float32)


def class_balance_summary(counts: np.ndarray, weights: np.ndarray) -> dict:
    """Build JSON-safe evidence for a class-weight decision."""

    counts_array = np.asarray(counts, dtype=np.int64).reshape(-1)
    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if counts_array.shape != weights_array.shape:
        raise ValueError("counts and weights must have the same shape")
    return {
        "counts": counts_array.astype(int).tolist(),
        "weights": weights_array.astype(float).tolist(),
        "supported_classes": [int(index) for index, count in enumerate(counts_array) if count > 0],
        "total_points": int(counts_array.sum()),
    }
