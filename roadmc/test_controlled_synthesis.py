"""Regression tests for controlled, label-complete synthetic generation."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.synthetic.config import (  # noqa: E402
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset  # noqa: E402
from roadmc.data.synthetic.labels import ALL_DISEASE_LABELS, TARGET_LABEL_SPECS  # noqa: E402
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA  # noqa: E402
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
