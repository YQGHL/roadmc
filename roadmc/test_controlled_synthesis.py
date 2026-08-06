"""Regression tests for controlled, label-complete synthetic generation."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA  # noqa: E402
from roadmc.data.synthetic.config import (  # noqa: E402
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset  # noqa: E402
from roadmc.data.synthetic.labels import ALL_DISEASE_LABELS, TARGET_LABEL_SPECS  # noqa: E402
from roadmc.data.synthetic.primitives import add_patching  # noqa: E402


def _deterministic_config() -> GeneratorConfig:
    return GeneratorConfig(
        road=RoadSurfaceConfig(width=3.0, length=3.0, grid_res=0.04, roughness_class="A"),
        micro_texture=MicroTextureConfig(amplitude=0.0),
        lidar_noise=LidarNoiseConfig(
            distance_noise_std=0.0,
            dropout_rate=0.0,
            angular_jitter_deg=0.0,
            enable_edge_mixing=False,
        ),
        seed=123,
        # Keep all raw points and duplicate only when needed. This makes the
        # reachability test deterministic rather than sampling-limited.
        num_points=6144,
    )


class ControlledSynthesisTests(unittest.TestCase):
    def test_target_specs_cover_every_non_background_label(self) -> None:
        self.assertEqual(tuple(TARGET_LABEL_SPECS), tuple(range(1, 38)))
        self.assertEqual(ALL_DISEASE_LABELS, tuple(range(1, 38)))

    def test_repair_patch_has_geometry_and_label_support(self) -> None:
        x, y = np.meshgrid(np.linspace(0.0, 2.0, 81), np.linspace(0.0, 2.0, 81))
        base_z = 0.2 * x - 0.1 * y
        points = np.column_stack((x.ravel(), y.ravel(), base_z.ravel()))
        labels = np.zeros(len(points), dtype=np.int64)

        patched_points, patched_labels = add_patching(
            points,
            labels,
            center=(1.0, 1.0),
            width=1.0,
            length=1.2,
            label=20,
            elevation=0.01,
            edge_width=0.10,
        )

        core = patched_labels == 20
        far = (np.abs(points[:, 0] - 1.0) > 0.7) | (np.abs(points[:, 1] - 1.0) > 0.8)
        height_change = patched_points[:, 2] - points[:, 2]
        self.assertGreater(int(core.sum()), 0)
        self.assertGreater(float(height_change[core].mean()), 0.008)
        self.assertTrue(np.allclose(height_change[far], 0.0))
        self.assertLessEqual(float(np.abs(height_change).max()), 0.01 + 1e-9)

    def test_every_target_label_reaches_final_scene(self) -> None:
        dataset = SyntheticRoadDataset(config=_deterministic_config(), dataset_size=0)
        for scene_id, target_label in enumerate(ALL_DISEASE_LABELS):
            with self.subTest(target_label=target_label):
                scene = dataset.generate_scene(scene_id, target_label=target_label)
                self.assertEqual(scene["target_label"], target_label)
                self.assertEqual(
                    scene["pavement_type"], TARGET_LABEL_SPECS[target_label].pavement_type
                )
                self.assertGreater(int((scene["labels"] == target_label).sum()), 0)
                self.assertEqual(scene["feature_schema"], OBSERVABLE_FEATURE_SCHEMA)
                self.assertEqual(scene["feats"].shape, (len(scene["points"]), 3))
                self.assertTrue(np.isfinite(scene["feats"]).all())

    def test_downsampling_keeps_a_forced_rare_label(self) -> None:
        config = replace(_deterministic_config(), num_points=1024)
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        for scene_id, target_label in enumerate((1, 9, 23, 34)):
            with self.subTest(target_label=target_label):
                scene = dataset.generate_scene(scene_id, target_label=target_label)
                self.assertGreater(int((scene["labels"] == target_label).sum()), 0)

    def test_resample_protection_preserves_natural_prevalence(self) -> None:
        """自然存活数超过最低要求时，不得触发保护，也不得固定 10% 配额。"""
        rng = np.random.default_rng(0)
        n_total, n_protected, target_num = 10_000, 5_000, 1_000
        points = rng.normal(size=(n_total, 3))
        labels = np.zeros(n_total, dtype=np.int64)
        labels[:n_protected] = 7  # 50% 自然占比
        aux = np.zeros(n_total, dtype=np.float32)
        normals = np.zeros((n_total, 3), dtype=np.float32)

        _, out_labels, _, _, _, info = SyntheticRoadDataset._resample_to_target(
            points, labels, aux, aux, normals, target_num, rng,
            protected_label=7, protected_min_points=1,
        )
        ratio = float((out_labels == 7).mean())
        self.assertFalse(info["protection_applied"])
        # 自然占比 50% 附近，绝不能被锁在旧逻辑的 ~10%
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)

    def test_resample_protection_enforces_minimum_survival(self) -> None:
        """目标标签极稀少时，保护采样保证最低存活点数且随参数变化。"""
        rng = np.random.default_rng(1)
        n_total, target_num = 50_000, 1_000
        points = rng.normal(size=(n_total, 3))
        labels = np.zeros(n_total, dtype=np.int64)
        labels[:5] = 3  # 0.01% 自然占比，自然采样大概率存活 < 4 点
        aux = np.zeros(n_total, dtype=np.float32)
        normals = np.zeros((n_total, 3), dtype=np.float32)

        for min_points in (1, 4):
            with self.subTest(min_points=min_points):
                _, out_labels, _, _, _, info = SyntheticRoadDataset._resample_to_target(
                    points, labels, aux, aux, normals, target_num,
                    np.random.default_rng(2),
                    protected_label=3, protected_min_points=min_points,
                )
                survived = int((out_labels == 3).sum())
                self.assertGreaterEqual(survived, min(min_points, 5))
                # 保护只保证下限，不重建固定比例
                self.assertLess(survived / target_num, 0.10)
                self.assertEqual(info["protected_min_points"], min_points)

    def test_resample_without_target_label_never_intervenes(self) -> None:
        """普通（非受控）场景禁用保护采样，保持完全自然下采样。"""
        rng = np.random.default_rng(3)
        n_total, target_num = 10_000, 1_000
        points = rng.normal(size=(n_total, 3))
        labels = np.zeros(n_total, dtype=np.int64)
        labels[:3] = 12
        aux = np.zeros(n_total, dtype=np.float32)
        normals = np.zeros((n_total, 3), dtype=np.float32)

        _, _, _, _, _, info = SyntheticRoadDataset._resample_to_target(
            points, labels, aux, aux, normals, target_num, rng,
            protected_label=None,
        )
        self.assertFalse(info["protection_applied"])
        self.assertEqual(info["protected_label"], -1)

    def test_controlled_scene_records_protection_metadata(self) -> None:
        """受控场景必须在 resolution_metadata 中留下可审计的保护记录。"""
        config = replace(_deterministic_config(), num_points=1024)
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        scene = dataset.generate_scene(0, target_label=9)
        record = scene["resolution_metadata"]["sensor_output"]["target_label_protection"]
        self.assertEqual(record["protected_label"], 9)
        self.assertTrue(record["protection_available"])
        self.assertIn("protection_applied", record)
        self.assertGreaterEqual(record["target_label_output_points"], 1)
        self.assertAlmostEqual(
            record["target_label_output_ratio"],
            record["target_label_output_points"] / len(scene["labels"]),
        )
        contract = scene["resolution_metadata"]["sensor_output"][
            "controlled_target_label_protection"
        ]
        self.assertEqual(contract["strategy"], "minimum_points")

    def test_density_voxel_path_declares_no_protection(self) -> None:
        """density_voxel 路径没有最低存活保证，元数据必须如实声明。"""
        config = replace(
            _deterministic_config(), num_points=1024, target_density=200.0
        )
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        scene = dataset.generate_scene(0, target_label=9)
        sensor_output = scene["resolution_metadata"]["sensor_output"]
        self.assertEqual(sensor_output["mode"], "density_voxel")
        record = sensor_output["target_label_protection"]
        self.assertFalse(record["protection_available"])
        self.assertFalse(record["protection_applied"])
        contract = sensor_output["controlled_target_label_protection"]
        self.assertEqual(contract["strategy"], "none")
        self.assertIsNone(contract["minimum_points"])

    def test_forced_protection_records_shortfall_when_pool_is_small(self) -> None:
        """存活池小于最低要求时，审计记录必须标明实际执行值与缺口。"""
        rng = np.random.default_rng(5)
        n_total, target_num = 20_000, 1_000
        points = rng.normal(size=(n_total, 3))
        labels = np.zeros(n_total, dtype=np.int64)
        labels[:30] = 6  # 存活池 30 < 要求 100
        aux = np.zeros(n_total, dtype=np.float32)
        normals = np.zeros((n_total, 3), dtype=np.float32)

        _, out_labels, _, _, _, info = SyntheticRoadDataset._resample_to_target(
            points, labels, aux, aux, normals, target_num,
            np.random.default_rng(6),
            protected_label=6, protected_min_points=100,
        )
        self.assertTrue(info["protection_applied"])
        self.assertEqual(info["effective_min_keep"], 30)
        self.assertTrue(info["min_unreachable"])
        self.assertEqual(int((out_labels == 6).sum()), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
