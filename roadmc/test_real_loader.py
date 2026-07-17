"""Integration tests for PLY/LAS loading and real-scene metadata conversion."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roadmc.data.features import compute_observable_features
from roadmc.data.real.dataset import RealRoadDataset
from roadmc.data.real.metadata import RoadPointCloudMetadata, write_scene_metadata


HAS_PLY = importlib.util.find_spec("plyfile") is not None
HAS_LAS = importlib.util.find_spec("laspy") is not None
HAS_OPEN3D = importlib.util.find_spec("open3d") is not None


class RealLoaderTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PLY, "plyfile is not installed")
    def test_ply_metadata_converts_millimeters_and_uint8_intensity(self) -> None:
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "road.ply"
            vertices = np.array(
                [
                    (1000.0, 2000.0, 10.0, 7, 0.0, 0.0, 2.0, 128),
                    (2000.0, 3000.0, 20.0, 8, 0.0, 3.0, 0.0, 255),
                ],
                dtype=[
                    ("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "i4"),
                    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("intensity", "u1"),
                ],
            )
            PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
            write_scene_metadata(
                path,
                RoadPointCloudMetadata(
                    sensor="test-lidar",
                    coordinate_units="mm",
                    intensity_scale="uint8",
                    road_segment_id="segment-1",
                    capture_date="2026-07-15",
                ),
            )
            points, labels, normals, intensity, metadata = RealRoadDataset.load_scene(path)

        self.assertTrue(np.allclose(points[0], [1.0, 2.0, 0.01]))
        self.assertEqual(labels.tolist(), [7, 8])
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0))
        self.assertTrue(np.allclose(intensity, [128.0 / 255.0, 1.0]))
        self.assertEqual(metadata.road_segment_id, "segment-1")

    @unittest.skipUnless(HAS_LAS, "laspy is not installed")
    def test_las_loader_reads_classification_and_uint16_intensity(self) -> None:
        import laspy

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "road.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x = np.array([1.0, 2.0])
            las.y = np.array([3.0, 4.0])
            las.z = np.array([0.1, 0.2])
            las.intensity = np.array([32768, 65535], dtype=np.uint16)
            las.classification = np.array([5, 6], dtype=np.uint8)
            las.write(path)
            write_scene_metadata(
                path,
                RoadPointCloudMetadata(
                    sensor="test-lidar",
                    coordinate_units="m",
                    intensity_scale="uint16",
                    capture_date="2026-07-15",
                ),
            )
            points, labels, normals, intensity, _ = RealRoadDataset.load_scene(path)

        self.assertEqual(points.shape, (2, 3))
        self.assertEqual(labels.tolist(), [5, 6])
        self.assertIsNone(normals)
        self.assertTrue(np.allclose(intensity, [32768.0 / 65535.0, 1.0]))

    @unittest.skipUnless(HAS_OPEN3D, "open3d is not installed")
    def test_pcd_loader_reads_points_normals_and_gray_intensity(self) -> None:
        import open3d as o3d

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "road.pcd"
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector([[1.0, 2.0, 0.1], [2.0, 3.0, 0.2]])
            cloud.normals = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
            cloud.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
            o3d.io.write_point_cloud(str(path), cloud)
            write_scene_metadata(
                path,
                RoadPointCloudMetadata(
                    sensor="test-lidar",
                    coordinate_units="m",
                    intensity_scale="normalized_0_1",
                    capture_date="2026-07-15",
                ),
            )
            points, labels, normals, intensity, _ = RealRoadDataset.load_scene(path)

        self.assertEqual(points.shape, (2, 3))
        self.assertIsNone(labels)
        self.assertTrue(np.allclose(normals, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
        self.assertTrue(np.allclose(intensity, [0.2, 0.8], atol=1e-5))

    def test_ascii_pcd_loader_preserves_explicit_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "road_ascii.pcd"
            path.write_text(
                "\n".join(
                    [
                        "# .PCD v0.7",
                        "VERSION 0.7",
                        "FIELDS x y z intensity label",
                        "SIZE 4 4 4 4 4",
                        "TYPE F F F F I",
                        "COUNT 1 1 1 1 1",
                        "WIDTH 2",
                        "HEIGHT 1",
                        "POINTS 2",
                        "DATA ascii",
                        "1 2 3 25 4",
                        "2 3 4 255 5",
                    ]
                ),
                encoding="ascii",
            )
            write_scene_metadata(
                path,
                RoadPointCloudMetadata(
                    sensor="test-lidar",
                    coordinate_units="m",
                    intensity_scale="uint8",
                    capture_date="2026-07-15",
                ),
            )
            points, labels, _, intensity, _ = RealRoadDataset.load_scene(path)

        self.assertTrue(np.allclose(points, [[1, 2, 3], [2, 3, 4]]))
        self.assertEqual(labels.tolist(), [4, 5])
        self.assertTrue(np.allclose(intensity, [25.0 / 255.0, 1.0]))

    def test_loader_removes_zero_return_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "road.npy"
            np.save(path, np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32))
            points, labels, normals, intensity, _ = RealRoadDataset.load_scene(path)

        self.assertTrue(np.allclose(points, [[1.0, 2.0, 3.0]]))
        self.assertIsNone(labels)
        self.assertIsNone(normals)
        self.assertIsNone(intensity)

    def test_dataset_builds_shared_observable_geometry_features(self) -> None:
        x, y = np.meshgrid(np.linspace(1.0, 2.0, 9), np.linspace(3.0, 4.0, 9))
        z = 0.2 + 0.04 * np.sin(5.0 * x) + 0.03 * y**2
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curved.npy"
            np.save(path, points)
            sample = RealRoadDataset(tmpdir, max_points=None, normalize=False)[0]

        expected = compute_observable_features(points)
        np.testing.assert_allclose(sample["feats"].numpy(), expected, rtol=2e-5, atol=2e-6)
        self.assertGreater(float(sample["feats"][:, 1].max()), 1e-5)
        self.assertGreater(float(sample["feats"][:, 2].abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
