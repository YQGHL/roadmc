"""Tests for the class-budget dataset accounting helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA
from roadmc.scripts.generate_class_budget import _feature_contract_errors, _parse_labels, _scan_coverage


class ClassBudgetTests(unittest.TestCase):
    def test_parse_labels_supports_ranges_and_rejects_background(self) -> None:
        self.assertEqual(_parse_labels("1-3,20,37"), (1, 2, 3, 20, 37))
        with self.assertRaises(ValueError):
            _parse_labels("0,1")

    def test_coverage_tracks_scenes_instances_and_points_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir)
            np.savez_compressed(
                split_dir / "scene_000000.npz",
                labels=np.array([0, 1, 1, 20, 20, 20], dtype=np.int32),
                target_label=1,
            )
            np.savez_compressed(
                split_dir / "scene_000001.npz",
                labels=np.array([0, 20, 20, 20], dtype=np.int32),
                target_label=20,
            )

            coverage = _scan_coverage(split_dir, (1, 20))

        self.assertEqual(coverage[1], {"scene_count": 1, "instance_count": 1, "point_count": 2})
        self.assertEqual(coverage[20], {"scene_count": 1, "instance_count": 1, "point_count": 3})

    def test_feature_contract_rejects_legacy_or_malformed_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir)
            np.savez_compressed(
                split_dir / "scene_000000.npz",
                points=np.zeros((4, 3), dtype=np.float32),
                feats=np.zeros((4, 3), dtype=np.float32),
                feature_schema=OBSERVABLE_FEATURE_SCHEMA,
            )
            np.savez_compressed(
                split_dir / "scene_000001.npz",
                points=np.zeros((4, 3), dtype=np.float32),
                feats=np.zeros((4, 2), dtype=np.float32),
            )
            errors = _feature_contract_errors(split_dir)

        self.assertEqual(len(errors), 1)
        self.assertIn("missing or incompatible", errors[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
