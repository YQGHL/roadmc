"""Streaming, scene-aware metrics for RoadMC segmentation experiments.

The training loop needs an epoch-global confusion matrix rather than the mean
of per-batch mIoU values.  This module keeps the statistical definitions in a
single place so Lightning validation and offline evaluation report the same
quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch


def confusion_matrix_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    valid_mask: torch.Tensor | None = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Return a ``(target, prediction)`` global confusion matrix.

    Invalid/padded entries and labels outside ``[0, num_classes)`` are ignored.
    Keeping target classes on rows makes support equal to ``matrix.sum(1)``.
    """
    predictions = predictions.reshape(-1).long()
    targets = targets.reshape(-1).long()

    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = valid_mask.reshape(-1).bool()

    valid &= targets.ne(ignore_index)
    valid &= targets.ge(0) & targets.lt(num_classes)
    valid &= predictions.ge(0) & predictions.lt(num_classes)

    if not torch.any(valid):
        return torch.zeros(
            (num_classes, num_classes), device=predictions.device, dtype=torch.long
        )

    encoded = targets[valid] * num_classes + predictions[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def metrics_from_confusion(
    confusion: np.ndarray | torch.Tensor,
    *,
    min_support: int = 1,
    include_background: bool = False,
    tail_labels: Sequence[int] | None = None,
) -> dict:
    """Compute globally defined segmentation metrics from one confusion matrix.

    ``supported_miou`` averages only classes whose target support reaches
    ``min_support``.  This prevents classes absent from a validation split from
    silently becoming artificial zero-IoU terms.  The complete per-class table
    is still returned, so absent labels remain visible in reports.
    """
    matrix = (
        confusion.detach().cpu().numpy() if isinstance(confusion, torch.Tensor) else confusion
    )
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion must be square, got {matrix.shape}")

    num_classes = matrix.shape[0]
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    false_positive = predicted_support - true_positive
    false_negative = true_support - true_positive

    iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    precision = _safe_ratio(true_positive, true_positive + false_positive)

    candidate = np.arange(num_classes)
    if not include_background and num_classes > 1:
        candidate = candidate[candidate != 0]
    supported = candidate[true_support[candidate] >= min_support]

    supported_miou = float(iou[supported].mean()) if supported.size else 0.0
    all_miou = float(iou[candidate].mean()) if candidate.size else 0.0

    if num_classes > 1:
        foreground_tp = int(matrix[1:, 1:].sum())
        foreground_fp = int(matrix[0, 1:].sum())
        foreground_fn = int(matrix[1:, 0].sum())
        foreground_iou = foreground_tp / max(foreground_tp + foreground_fp + foreground_fn, 1)
        foreground_precision = foreground_tp / max(foreground_tp + foreground_fp, 1)
        foreground_recall = foreground_tp / max(foreground_tp + foreground_fn, 1)
    else:
        foreground_iou = 0.0
        foreground_precision = 0.0
        foreground_recall = 0.0

    per_class = []
    for class_id in range(num_classes):
        per_class.append(
            {
                "class_id": class_id,
                "support": int(true_support[class_id]),
                "predicted_support": int(predicted_support[class_id]),
                "tp": int(true_positive[class_id]),
                "fp": int(false_positive[class_id]),
                "fn": int(false_negative[class_id]),
                "iou": float(iou[class_id]),
                "recall": float(recall[class_id]),
                "precision": float(precision[class_id]),
                "supported": bool(true_support[class_id] >= min_support),
            }
        )

    tail = []
    if tail_labels is not None:
        tail = [class_id for class_id in tail_labels if 0 <= class_id < num_classes]
    tail_supported = [class_id for class_id in tail if true_support[class_id] >= min_support]
    tail_miou = float(iou[tail_supported].mean()) if tail_supported else None

    return {
        "num_classes": num_classes,
        "total_points": int(matrix.sum()),
        "min_support": int(min_support),
        "supported_labels": [int(class_id) for class_id in supported],
        "supported_miou": supported_miou,
        "all_non_background_miou": all_miou,
        "foreground_iou": float(foreground_iou),
        "foreground_precision": float(foreground_precision),
        "foreground_recall": float(foreground_recall),
        "tail_labels": [int(class_id) for class_id in tail],
        "tail_supported_labels": [int(class_id) for class_id in tail_supported],
        "tail_miou": tail_miou,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


@dataclass
class CalibrationAccumulator:
    """Streaming top-label calibration and proper scoring rules."""

    n_bins: int = 15
    count: np.ndarray = field(init=False)
    confidence_sum: np.ndarray = field(init=False)
    correct_sum: np.ndarray = field(init=False)
    brier_sum: float = 0.0
    nll_sum: float = 0.0
    total: int = 0

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.count = np.zeros(self.n_bins, dtype=np.int64)
        self.confidence_sum = np.zeros(self.n_bins, dtype=np.float64)
        self.correct_sum = np.zeros(self.n_bins, dtype=np.float64)

    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        ignore_index: int = -1,
    ) -> None:
        """Accumulate ECE, multiclass Brier score and negative log likelihood."""
        if probabilities.shape[-1] < 2:
            raise ValueError("probabilities must contain at least two classes")

        classes = probabilities.shape[-1]
        probs = probabilities.reshape(-1, classes)
        labels = targets.reshape(-1).long()
        if valid_mask is None:
            valid = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid = valid_mask.reshape(-1).bool()
        valid &= labels.ne(ignore_index)
        valid &= labels.ge(0) & labels.lt(classes)
        if not torch.any(valid):
            return

        probs = probs[valid].float()
        labels = labels[valid]
        confidence, prediction = probs.max(dim=-1)
        correct = prediction.eq(labels)
        true_probability = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        brier = probs.square().sum(dim=-1) - 2.0 * true_probability + 1.0

        bin_index = torch.floor(confidence * self.n_bins).long().clamp(0, self.n_bins - 1)
        counts = torch.bincount(bin_index, minlength=self.n_bins).detach().cpu().numpy()
        confidence_sum = torch.zeros(self.n_bins, device=probs.device).scatter_add_(
            0, bin_index, confidence
        )
        correct_sum = torch.zeros(self.n_bins, device=probs.device).scatter_add_(
            0, bin_index, correct.float()
        )

        self.count += counts.astype(np.int64)
        self.confidence_sum += confidence_sum.detach().cpu().numpy()
        self.correct_sum += correct_sum.detach().cpu().numpy()
        self.brier_sum += float(brier.sum().item())
        self.nll_sum += float((-true_probability.log()).sum().item())
        self.total += int(labels.numel())

    def as_dict(self) -> dict:
        if self.total == 0:
            return {"ece": 0.0, "brier": 0.0, "nll": 0.0, "total": 0, "bins": []}

        bins = []
        ece = 0.0
        for index, count in enumerate(self.count):
            if count == 0:
                bins.append({"lower": index / self.n_bins, "upper": (index + 1) / self.n_bins,
                             "count": 0, "confidence": None, "accuracy": None})
                continue
            confidence = self.confidence_sum[index] / count
            accuracy = self.correct_sum[index] / count
            ece += (count / self.total) * abs(accuracy - confidence)
            bins.append(
                {
                    "lower": index / self.n_bins,
                    "upper": (index + 1) / self.n_bins,
                    "count": int(count),
                    "confidence": float(confidence),
                    "accuracy": float(accuracy),
                }
            )

        return {
            "ece": float(ece),
            "brier": float(self.brier_sum / self.total),
            "nll": float(self.nll_sum / self.total),
            "total": int(self.total),
            "bins": bins,
        }


def bootstrap_scene_confidence_intervals(
    scene_confusions: Iterable[np.ndarray | torch.Tensor],
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
    min_support: int = 1,
    tail_labels: Sequence[int] | None = None,
    chunk_size: int = 8,
) -> dict:
    """Block-bootstrap global metrics by resampling complete scenes.

    Point-level IID bootstrap would be mathematically inappropriate because
    adjacent points in one road scene are strongly correlated.  A scene is the
    resampling block, so the resulting interval represents between-scene
    uncertainty rather than artificial point-level certainty.
    """
    matrices = [
        item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
        for item in scene_confusions
    ]
    if not matrices:
        return {"n_scenes": 0, "n_bootstrap": 0, "confidence": confidence, "metrics": {}}
    stacked = np.stack(matrices, axis=0).astype(np.int64, copy=False)
    if stacked.ndim != 3 or stacked.shape[1] != stacked.shape[2]:
        raise ValueError(f"scene_confusions must be (S, C, C), got {stacked.shape}")

    point_estimate = metrics_from_confusion(
        stacked.sum(axis=0), min_support=min_support, tail_labels=tail_labels
    )
    if n_bootstrap <= 0 or stacked.shape[0] < 2:
        return {
            "n_scenes": int(stacked.shape[0]),
            "n_bootstrap": 0,
            "confidence": confidence,
            "metrics": {
                "supported_miou": {"point_estimate": point_estimate["supported_miou"]},
                "foreground_iou": {"point_estimate": point_estimate["foreground_iou"]},
            },
        }

    rng = np.random.default_rng(seed)
    tracked = {"supported_miou": [], "foreground_iou": []}
    if tail_labels:
        tracked["tail_miou"] = []

    scene_count = stacked.shape[0]
    for start in range(0, n_bootstrap, chunk_size):
        size = min(chunk_size, n_bootstrap - start)
        indices = rng.integers(0, scene_count, size=(size, scene_count))
        resampled = stacked[indices].sum(axis=1)
        for matrix in resampled:
            metric = metrics_from_confusion(
                matrix, min_support=min_support, tail_labels=tail_labels
            )
            tracked["supported_miou"].append(metric["supported_miou"])
            tracked["foreground_iou"].append(metric["foreground_iou"])
            if "tail_miou" in tracked and metric["tail_miou"] is not None:
                tracked["tail_miou"].append(metric["tail_miou"])

    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for key, values in tracked.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        intervals[key] = {
            "point_estimate": float(point_estimate[key]),
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1.0 - alpha)),
            "bootstrap_mean": float(arr.mean()),
        }

    return {
        "n_scenes": int(scene_count),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
        "metrics": intervals,
    }


def scan_binary_thresholds(
    disease_probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict]:
    """Evaluate binary foreground IoU for a fixed list of validation thresholds."""
    probs = np.asarray(disease_probabilities, dtype=np.float64).reshape(-1)
    binary_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probs.shape != binary_targets.shape:
        raise ValueError("disease_probabilities and targets must have equal shape")
    valid = (binary_targets == 0) | (binary_targets == 1)
    probs = probs[valid]
    binary_targets = binary_targets[valid]

    output = []
    for threshold in thresholds:
        prediction = (probs >= threshold).astype(np.int64)
        tp = int(np.logical_and(prediction == 1, binary_targets == 1).sum())
        fp = int(np.logical_and(prediction == 1, binary_targets == 0).sum())
        fn = int(np.logical_and(prediction == 0, binary_targets == 1).sum())
        foreground_iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        output.append(
            {
                "threshold": float(threshold),
                "foreground_iou": float(foreground_iou),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
    return output


if __name__ == "__main__":
    # Lightweight self-test that does not require pytest or GPU access.
    pred = torch.tensor([0, 1, 1, 2, 2, 0])
    target = torch.tensor([0, 1, 2, 2, 1, -1])
    matrix = confusion_matrix_from_predictions(pred, target, 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 1, 1]]
    metric = metrics_from_confusion(matrix)
    assert metric["supported_labels"] == [1, 2]

    calibration = CalibrationAccumulator(n_bins=4)
    calibration.update(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]), torch.tensor([0, 1])
    )
    assert calibration.as_dict()["brier"] < 0.03
    print("roadmc.metrics self-test passed")
