"""Tests for curriculum-aware class-count and effective-number weighting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.class_balance import effective_number_class_weights, point_class_counts


class ClassBalanceTests(unittest.TestCase):
    def test_effective_number_weights_are_bounded_and_mean_normalized(self) -> None:
        weights = effective_number_class_weights(
            np.array([10_000, 100, 0]),
            beta=0.999,
            max_weight=3.0,
        )

        self.assertEqual(float(weights[2]), 0.0)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertLessEqual(float(weights[1]), 3.0 + 1e-6)
        self.assertAlmostEqual(float(weights[:2].mean()), 1.0, places=6)

    def test_point_counts_follow_binary_curriculum_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split = Path(tmpdir) / "train"
            split.mkdir()
            np.savez_compressed(
                split / "scene_000000.npz",
                labels=np.array([0, 0, 1, 20, 37], dtype=np.int64),
            )
            counts = point_class_counts(tmpdir, split="train", label_stage="binary")

        np.testing.assert_array_equal(counts, np.array([2, 3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
