"""Regression tests for globally defined RoadMC segmentation metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from roadmc.evaluate import _report_lines
from roadmc.metrics import (
    CalibrationAccumulator,
    bootstrap_scene_confidence_intervals,
    confusion_matrix_from_predictions,
    metrics_from_confusion,
    scan_binary_thresholds,
)


class TestSegmentationMetrics(unittest.TestCase):
    def test_confusion_ignores_padding(self) -> None:
        prediction = torch.tensor([0, 1, 1, 2, 2])
        target = torch.tensor([0, 1, -1, 1, 2])
        valid = torch.tensor([True, True, True, False, True])
        matrix = confusion_matrix_from_predictions(prediction, target, 3, valid)
        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_supported_miou_excludes_absent_label(self) -> None:
        matrix = np.array(
            [
                [10, 0, 0, 0],
                [0, 6, 4, 0],
                [0, 2, 8, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        metrics = metrics_from_confusion(matrix, min_support=1)
        self.assertEqual(metrics["supported_labels"], [1, 2])
        self.assertAlmostEqual(metrics["supported_miou"], (0.5 + 8 / 14) / 2)
        self.assertFalse(metrics["per_class"][3]["supported"])

    def test_calibration_perfect_predictions_are_well_calibrated(self) -> None:
        accumulator = CalibrationAccumulator(n_bins=10)
        accumulator.update(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([0, 1, 0]),
        )
        summary = accumulator.as_dict()
        self.assertLess(summary["ece"], 1e-8)
        self.assertLess(summary["brier"], 1e-8)
        self.assertLess(summary["nll"], 1e-8)

    def test_scene_bootstrap_preserves_point_estimate(self) -> None:
        scenes = [
            np.array([[9, 1], [1, 9]], dtype=np.int64),
            np.array([[8, 2], [2, 8]], dtype=np.int64),
            np.array([[7, 3], [3, 7]], dtype=np.int64),
        ]
        result = bootstrap_scene_confidence_intervals(scenes, n_bootstrap=32, seed=7)
        expected = metrics_from_confusion(sum(scenes))["supported_miou"]
        actual = result["metrics"]["supported_miou"]["point_estimate"]
        self.assertAlmostEqual(actual, expected)
        self.assertLessEqual(
            result["metrics"]["supported_miou"]["lower"],
            result["metrics"]["supported_miou"]["upper"],
        )

    def test_binary_threshold_scan_finds_separating_threshold(self) -> None:
        result = scan_binary_thresholds(
            np.array([0.02, 0.20, 0.75, 0.95]),
            np.array([0, 0, 1, 1]),
            thresholds=[0.1, 0.5, 0.9],
        )
        best = max(result, key=lambda item: item["foreground_iou"])
        self.assertEqual(best["threshold"], 0.5)
        self.assertAlmostEqual(best["foreground_iou"], 1.0)

    def test_report_handles_bootstrap_disabled(self) -> None:
        matrix = np.array([[8, 2], [1, 9]], dtype=np.int64)
        metrics = metrics_from_confusion(matrix)
        bootstrap = bootstrap_scene_confidence_intervals([matrix], n_bootstrap=0)
        lines = _report_lines(
            metrics,
            {"ece": 0.1, "brier": 0.2, "nll": 0.3},
            bootstrap,
            "binary",
        )
        self.assertTrue(any("RoadMC Global Evaluation Report" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
