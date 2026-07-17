"""Regression tests for the inference-safe RoadMC feature contract."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from roadmc.data.dataloader import SyntheticPointCloudDataset
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, compute_observable_features
from roadmc.data.synthetic.config import (
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset
from roadmc.models.model_pl import EdgeLoss, RoadMCSegModel


def _surface_points(side: int = 9) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, side), np.linspace(-1.0, 1.0, side))
    z = 0.08 * x**2 - 0.04 * y + 0.015 * np.sin(3.0 * x)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)


class ObservableFeatureTests(unittest.TestCase):
    def test_geometry_features_are_translation_and_uniform_scale_invariant(self) -> None:
        points = _surface_points()
        intensity = np.linspace(0.1, 0.9, len(points), dtype=np.float32)
        baseline = compute_observable_features(points, intensity)
        transformed = compute_observable_features(
            points * 4.5 + np.array([3.0, -2.0, 7.0]),
            intensity,
        )

        np.testing.assert_allclose(baseline, transformed, rtol=2e-5, atol=2e-6)

    def test_synthetic_features_are_recomputed_from_observations_not_labels(self) -> None:
        config = GeneratorConfig(
            road=RoadSurfaceConfig(width=2.0, length=2.0, grid_res=0.04, roughness_class="A"),
            micro_texture=MicroTextureConfig(amplitude=0.0),
            lidar_noise=LidarNoiseConfig(
                distance_noise_std=0.0,
                dropout_rate=0.0,
                angular_jitter_deg=0.0,
                enable_edge_mixing=False,
            ),
            seed=17,
            num_points=512,
        )
        scene = SyntheticRoadDataset(config=config, dataset_size=0).generate_scene(
            0,
            target_label=9,
        )
        reconstructed = compute_observable_features(scene["points"], scene["feats"][:, 0])

        np.testing.assert_allclose(scene["feats"], reconstructed, rtol=2e-5, atol=2e-6)
        relabeled = scene["labels"].copy()
        relabeled[:] = 37
        self.assertEqual(relabeled.shape, scene["labels"].shape)
        np.testing.assert_allclose(scene["feats"], reconstructed, rtol=2e-5, atol=2e-6)

    def test_legacy_dataset_ignores_privileged_third_channel(self) -> None:
        points = _surface_points()
        intensity = np.linspace(0.2, 0.8, len(points), dtype=np.float32)
        legacy_feats = np.column_stack(
            (intensity, np.full(len(points), 0.75), np.full(len(points), 999.0))
        ).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            split = Path(tmpdir) / "train"
            split.mkdir()
            np.savez_compressed(
                split / "scene_000000.npz",
                points=points,
                labels=np.zeros(len(points), dtype=np.int64),
                feats=legacy_feats,
                normals=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1)),
            )
            dataset = SyntheticPointCloudDataset(tmpdir, split="train", max_points=None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sample = dataset[0]

        expected = compute_observable_features(points, intensity)
        np.testing.assert_allclose(sample["feats"].numpy(), expected, rtol=2e-5, atol=2e-6)
        self.assertTrue(any("Legacy RoadMC feature schema" in str(item.message) for item in caught))
        self.assertLess(float(sample["feats"][:, 2].abs().max()), 3.1)

    def test_edge_loss_uses_supervision_but_keeps_prediction_gradients(self) -> None:
        x, y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 8),
            torch.linspace(-1.0, 1.0, 8),
            indexing="ij",
        )
        coords = torch.stack((x.flatten(), y.flatten(), torch.zeros(64)), dim=-1).unsqueeze(0)
        targets = torch.zeros((1, 64), dtype=torch.long)
        targets[:, 20:30] = 1
        logits = torch.randn((1, 64, 2), requires_grad=True)

        loss = EdgeLoss(grid_size=32)(logits, targets, coords)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_checkpoint_contract_rejects_pre_migration_weights(self) -> None:
        model = RoadMCSegModel(
            in_channels=3,
            num_classes=2,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 4, 4),
            window_size=8,
        )
        checkpoint: dict = {}
        model.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["roadmc_feature_schema"], OBSERVABLE_FEATURE_SCHEMA)
        model.on_load_checkpoint(checkpoint)
        with self.assertRaises(ValueError):
            model.on_load_checkpoint({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
