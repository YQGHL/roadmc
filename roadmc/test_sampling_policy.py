"""Tests that training rebalancing never leaks into validation prevalence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.dataloader import SyntheticPointCloudDataset
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, compute_observable_features


class SamplingPolicyTests(unittest.TestCase):
    def test_validation_sampling_is_deterministic_and_not_stratified(self) -> None:
        x, y = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))
        points = np.column_stack((x.ravel(), y.ravel(), 0.01 * (x + y).ravel()))
        labels = np.zeros(100, dtype=np.int64)
        labels[:20] = 1
        feats = compute_observable_features(points)
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (100, 1))
        with tempfile.TemporaryDirectory() as tmpdir:
            for split in ("train", "val"):
                directory = Path(tmpdir) / split
                directory.mkdir()
                np.savez_compressed(
                    directory / "scene_000000.npz",
                    points=points,
                    labels=labels,
                    feats=feats,
                    normals=normals,
                    feature_schema=OBSERVABLE_FEATURE_SCHEMA,
                )
            train = SyntheticPointCloudDataset(tmpdir, "train", max_points=10)
            val = SyntheticPointCloudDataset(tmpdir, "val", max_points=10)
            train_sample = train[0]
            val_first = val[0]
            val_second = val[0]

        self.assertEqual(int((train_sample["labels"] > 0).sum()), 10)
        self.assertTrue(np.array_equal(val_first["labels"].numpy(), val_second["labels"].numpy()))
        self.assertLess(int((val_first["labels"] > 0).sum()), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
