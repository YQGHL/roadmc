"""Tests for observable synthetic-to-real domain-gap diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from roadmc.domain_gap import (
    PointCloudRecord,
    compare_domains,
    dominant_ground_surface,
    observable_descriptors,
)
from roadmc.scripts.diagnose_domain_gap import _apply_ground_filter


class DomainGapTests(unittest.TestCase):
    @staticmethod
    def _plane_record() -> PointCloudRecord:
        x, y = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16))
        z = 0.01 * x + 0.02 * y
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
        intensity = np.linspace(0.2, 0.8, len(points))
        return PointCloudRecord(points, intensity, normals, name="plane")

    def test_observable_descriptors_are_finite(self) -> None:
        descriptors = observable_descriptors(self._plane_record(), max_points=None)
        for values in descriptors.values():
            self.assertTrue(np.isfinite(values).all())
        self.assertIn("signed_height_residual_over_radius", descriptors)

    def test_identical_domains_have_zero_distributional_gap(self) -> None:
        record = self._plane_record()
        report = compare_domains([record], [record], max_points_per_scene=None)
        self.assertAlmostEqual(report["joint_rbf_mmd"], 0.0, places=12)
        for item in report["descriptors"].values():
            if item["available"]:
                self.assertAlmostEqual(item["distances"]["wasserstein_1"], 0.0, places=12)
                self.assertAlmostEqual(item["distances"]["energy"], 0.0, places=12)

    def test_supplied_normals_are_ignored_by_default(self) -> None:
        record = self._plane_record()
        tilted = np.tile(
            np.array([0.0, 0.5, np.sqrt(0.75)]), (len(record.points), 1)
        )
        with_normals = PointCloudRecord(record.points, None, tilted, name="tilted")
        default = observable_descriptors(with_normals, max_points=None)
        supplied = observable_descriptors(with_normals, max_points=None, normal_source="supplied")
        without = observable_descriptors(
            PointCloudRecord(record.points, None, None, name="plain"), max_points=None
        )
        np.testing.assert_allclose(
            default["normal_tilt_rad"], without["normal_tilt_rad"]
        )
        self.assertGreater(
            float(supplied["normal_tilt_rad"].mean()),
            float(default["normal_tilt_rad"].mean()) + 0.1,
        )

    def test_descriptor_sampling_does_not_bias_density(self) -> None:
        rng = np.random.default_rng(0)
        points = np.column_stack(
            (
                rng.uniform(0.0, 2.0, 4000),
                rng.uniform(0.0, 2.0, 4000),
                rng.normal(0.0, 0.002, 4000),
            )
        )
        record = PointCloudRecord(points, None, None, name="random")
        full = observable_descriptors(record, max_points=None)
        capped = observable_descriptors(record, max_points=256)
        self.assertEqual(len(capped["density_per_m2"]), 256)
        full_mean = float(full["density_per_m2"].mean())
        self.assertAlmostEqual(
            float(capped["density_per_m2"].mean()), full_mean, delta=0.15 * full_mean
        )

    @unittest.skipUnless(__import__("importlib").util.find_spec("open3d") is not None, "open3d is not installed")
    def test_ground_plane_filter_retains_planar_scene(self) -> None:
        record = self._plane_record()
        filtered, info = dominant_ground_surface(record, distance_threshold=0.01, iterations=100)
        self.assertEqual(len(filtered.points), len(record.points))
        self.assertGreater(info["retained_fraction"], 0.99)

    @staticmethod
    def _wall_record() -> PointCloudRecord:
        y, z = np.meshgrid(np.linspace(0.0, 2.0, 32), np.linspace(0.0, 2.0, 32))
        points = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
        intensity = np.linspace(0.2, 0.8, len(points))
        return PointCloudRecord(points, intensity, name="wall")

    @unittest.skipUnless(__import__("importlib").util.find_spec("open3d") is not None, "open3d is not installed")
    def test_ground_filter_rejects_wall_dominant_frame_by_default(self) -> None:
        with self.assertRaises(ValueError):
            _apply_ground_filter([self._wall_record()], 0.15, 42)

    @unittest.skipUnless(__import__("importlib").util.find_spec("open3d") is not None, "open3d is not installed")
    def test_ground_filter_skips_wall_dominant_frame_when_asked(self) -> None:
        records = [self._plane_record(), self._wall_record()]
        filtered, details, skipped = _apply_ground_filter(records, 0.15, 42, skip_invalid=True)
        self.assertEqual([record.name for record in filtered], ["plane"])
        self.assertEqual(len(details), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["name"], "wall")
        self.assertIn("not ground-like", skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
