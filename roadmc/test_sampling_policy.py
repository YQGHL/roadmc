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

        # train 分层封顶到 train_disease_ratio=0.5：20 病害点中最多保留 5 个
        self.assertEqual(int((train_sample["labels"] > 0).sum()), 5)
        self.assertTrue(np.array_equal(val_first["labels"].numpy(), val_second["labels"].numpy()))
        self.assertLess(int((val_first["labels"] > 0).sum()), 10)

    def test_train_disease_cap_spreads_degenerate_dense_scenes(self) -> None:
        # 模拟 raw_surface 原生密度下的大面病害场景：病害点绝对多数。
        # 修复前采样器会全留病害点，整张场景挤进病害区域导致窗口退化。
        x, y = np.meshgrid(np.arange(40, dtype=np.float32), np.arange(40, dtype=np.float32))
        points = np.column_stack((x.ravel(), y.ravel(), 0.01 * (x + y).ravel()))
        labels = np.zeros(1600, dtype=np.int64)
        labels[:1200] = 1  # 75% 病害
        feats = compute_observable_features(points)
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (1600, 1))
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
            train = SyntheticPointCloudDataset(tmpdir, "train", max_points=100)
            sample = train[0]

        # 封顶 50%：100 点里最多 50 个病害点，其余为背景，空间分布被撑开
        self.assertEqual(int((sample["labels"] > 0).sum()), 50)

    def test_train_cap_background_deficit_falls_back_to_all_disease(self) -> None:
        # 极端情形：场景几乎全为病害、背景不足。封顶逻辑应回退到
        # 可用背景总量，不因取背景点而崩溃。
        points = np.column_stack((
            np.arange(200, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
        ))
        labels = np.ones(200, dtype=np.int64)  # 100% 病害
        feats = compute_observable_features(points)
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (200, 1))
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "train"
            directory.mkdir()
            np.savez_compressed(
                directory / "scene_000000.npz",
                points=points,
                labels=labels,
                feats=feats,
                normals=normals,
                feature_schema=OBSERVABLE_FEATURE_SCHEMA,
            )
            train = SyntheticPointCloudDataset(tmpdir, "train", max_points=100)
            sample = train[0]

        # 背景总量为 0：封顶回退到全留病害（100 点）
        self.assertEqual(int((sample["labels"] > 0).sum()), 100)
        self.assertEqual(sample["labels"].shape[0], 100)

    def test_train_ratio_validation(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticPointCloudDataset("/nonexistent", "train", train_disease_ratio=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
