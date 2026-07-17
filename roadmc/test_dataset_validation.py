"""Tests for synthetic data contract validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, compute_observable_features
from roadmc.scripts.validate_synthetic_dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_validation_accepts_reconstructed_observable_features(self) -> None:
        x, y = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
        points = np.column_stack((x.ravel(), y.ravel(), (0.1 * x**2).ravel())).astype(np.float32)
        feats = compute_observable_features(points)
        with tempfile.TemporaryDirectory() as tmpdir:
            split = Path(tmpdir) / "train"
            split.mkdir()
            np.savez_compressed(
                split / "scene_000000.npz",
                points=points,
                labels=np.zeros(len(points), dtype=np.int64),
                feats=feats,
                normals=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1)),
                feature_schema=OBSERVABLE_FEATURE_SCHEMA,
                target_label=-1,
            )
            report = validate_dataset(tmpdir, split="train", feature_check_scenes=1)

        self.assertTrue(report["valid"])
        self.assertEqual(report["feature_reconstruction_scenes"], 1)

    def test_validation_rejects_missing_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split = Path(tmpdir) / "train"
            split.mkdir()
            np.savez_compressed(
                split / "scene_000000.npz",
                points=np.zeros((4, 3), dtype=np.float32),
                labels=np.zeros(4, dtype=np.int64),
                feats=np.zeros((4, 3), dtype=np.float32),
                normals=np.zeros((4, 3), dtype=np.float32),
            )
            report = validate_dataset(tmpdir, split="train")

        self.assertFalse(report["valid"])
        self.assertEqual(report["error_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
