"""Tests for observable synthetic-to-real domain-gap diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from roadmc.domain_gap import PointCloudRecord, compare_domains, dominant_ground_surface, observable_descriptors


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

    @unittest.skipUnless(__import__("importlib").util.find_spec("open3d") is not None, "open3d is not installed")
    def test_ground_plane_filter_retains_planar_scene(self) -> None:
        record = self._plane_record()
        filtered, info = dominant_ground_surface(record, distance_threshold=0.01, iterations=100)
        self.assertEqual(len(filtered.points), len(record.points))
        self.assertGreater(info["retained_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
