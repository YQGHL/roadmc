"""Tests for curriculum label partitions and dataset remapping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.curriculum import LABEL_STAGES, label_lut, num_classes_for_stage
from roadmc.data.dataloader import SyntheticPointCloudDataset
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA


class CurriculumTests(unittest.TestCase):
    def test_every_stage_is_a_complete_partition(self) -> None:
        for stage in LABEL_STAGES:
            with self.subTest(stage=stage):
                lut = label_lut(stage)
                self.assertEqual(len(lut), 38)
                self.assertTrue(all(0 <= value < num_classes_for_stage(stage) for value in lut))
                self.assertEqual(set(lut), set(range(num_classes_for_stage(stage))))

    def test_dataset_applies_four_and_eight_class_luts(self) -> None:
        source_labels = np.array([0, 1, 9, 13, 19, 21, 27, 34, 37], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir()
            np.savez_compressed(
                split_dir / "scene_000000.npz",
                points=np.zeros((len(source_labels), 3), dtype=np.float32),
                labels=source_labels,
                feats=np.zeros((len(source_labels), 3), dtype=np.float32),
                normals=np.zeros((len(source_labels), 3), dtype=np.float32),
                feature_schema=OBSERVABLE_FEATURE_SCHEMA,
            )

            four = SyntheticPointCloudDataset(tmpdir, "train", max_points=None, label_stage="four")
            eight = SyntheticPointCloudDataset(tmpdir, "train", max_points=None, label_stage="eight")
            binary = SyntheticPointCloudDataset(tmpdir, "train", max_points=None, binary=True)

            self.assertEqual(four[0]["labels"].tolist(), [0, 1, 2, 3, 2, 2, 3, 2, 3])
            self.assertEqual(eight[0]["labels"].tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 7])
            self.assertEqual(binary[0]["labels"].tolist(), [0, 1, 1, 1, 1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
