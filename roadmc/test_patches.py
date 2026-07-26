"""Tests for deterministic metric-space point-cloud patch extraction."""

from __future__ import annotations

import unittest

import numpy as np

from roadmc.data.patches import extract_overlapping_patches, restore_metric_points


def _scene_from_metric_points(
    points_m: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    center: np.ndarray | None = None,
    scale: float = 4.0,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    points_m = np.asarray(points_m, dtype=np.float64)
    if center is None:
        center = np.array([10.0, -3.0, 0.25], dtype=np.float64)
    if labels is None:
        labels = np.zeros(points_m.shape[0], dtype=np.int64)
    point_ids = np.arange(points_m.shape[0], dtype=np.float32)
    scene = {
        "points": ((points_m - center) / scale).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "feats": np.column_stack((point_ids, point_ids + 0.25, point_ids + 0.5)),
        "normals": np.column_stack(
            (
                np.zeros(points_m.shape[0], dtype=np.float32),
                np.zeros(points_m.shape[0], dtype=np.float32),
                np.ones(points_m.shape[0], dtype=np.float32),
            )
        ),
        "coordinate_center": np.asarray(center, dtype=np.float32),
        "coordinate_scale": np.asarray(scale, dtype=np.float32),
        "coordinates_normalized": np.asarray(True),
    }
    if valid_mask is not None:
        scene["valid_mask"] = np.asarray(valid_mask, dtype=np.bool_)
    return scene


class MetricCoordinateTests(unittest.TestCase):
    def test_restores_inverse_normalization_in_meters(self) -> None:
        points_m = np.array(
            [[0.001, 0.005, -0.002], [7.0, 5.0, 0.035]], dtype=np.float64
        )
        scene = _scene_from_metric_points(points_m, scale=5.5)

        restored = restore_metric_points(scene)

        np.testing.assert_allclose(restored, points_m, rtol=0.0, atol=1e-6)
        self.assertEqual(restored.dtype, np.float64)

    def test_rejects_missing_or_invalid_physical_metadata(self) -> None:
        base = _scene_from_metric_points(np.array([[0.0, 0.0, 0.0]]))
        for missing_key in ("coordinate_center", "coordinate_scale"):
            with self.subTest(missing_key=missing_key):
                scene = dict(base)
                del scene[missing_key]
                with self.assertRaisesRegex(ValueError, missing_key):
                    extract_overlapping_patches(scene, 1.0, 0.5)

        for bad_scale in (0.0, -1.0, np.inf, np.array([1.0, 2.0])):
            with self.subTest(bad_scale=bad_scale):
                scene = dict(base)
                scene["coordinate_scale"] = np.asarray(bad_scale)
                with self.assertRaisesRegex(ValueError, "coordinate_scale"):
                    restore_metric_points(scene)

        scene = dict(base)
        scene["coordinate_center"] = np.zeros(2, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "coordinate_center"):
            restore_metric_points(scene)


class SpatialPatchTests(unittest.TestCase):
    def test_slices_in_metric_space_with_overlap_and_aligned_fields(self) -> None:
        points_m = np.array(
            [[x, y, 0.01 * x] for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0, 3.0)],
            dtype=np.float64,
        )
        scene = _scene_from_metric_points(points_m)

        patches = extract_overlapping_patches(
            scene,
            patch_size_m=(2.0, 2.0),
            stride_m=(1.0, 1.0),
            points_per_patch=None,
            split="val",
        )

        self.assertEqual(len(patches), 2)
        np.testing.assert_allclose(patches[0]["patch_bounds_m"], [0.0, 0.0, 2.0, 2.0])
        np.testing.assert_allclose(patches[1]["patch_bounds_m"], [1.0, 0.0, 3.0, 2.0])
        self.assertEqual(patches[0]["source_indices"].tolist(), [0, 1, 4, 5])
        self.assertEqual(patches[1]["source_indices"].tolist(), [1, 2, 3, 5, 6, 7])

        for patch in patches:
            indices = patch["source_indices"]
            np.testing.assert_array_equal(patch["points"], scene["points"][indices])
            np.testing.assert_allclose(patch["points_m"], points_m[indices], atol=1e-6)
            np.testing.assert_array_equal(patch["labels"], scene["labels"][indices])
            np.testing.assert_array_equal(patch["feats"], scene["feats"][indices])
            np.testing.assert_array_equal(patch["normals"], scene["normals"][indices])
            self.assertTrue(np.all(patch["valid_mask"]))

    def test_terminal_patch_anchors_to_maximum_without_coverage_gaps(self) -> None:
        points_m = np.column_stack(
            (
                np.arange(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )
        )
        scene = _scene_from_metric_points(points_m)

        patches = extract_overlapping_patches(scene, patch_size_m=2.0, stride_m=2.0)

        self.assertEqual([patch["patch_bounds_m"][0] for patch in patches], [0.0, 2.0, 3.0])
        covered = set()
        for patch in patches:
            covered.update(patch["source_indices"].tolist())
        self.assertEqual(covered, set(range(6)))
        self.assertIn(3, patches[1]["source_indices"])
        self.assertIn(3, patches[2]["source_indices"])

    def test_existing_invalid_padding_is_excluded_before_slicing(self) -> None:
        points_m = np.column_stack(
            (
                np.arange(5, dtype=np.float64),
                np.zeros(5, dtype=np.float64),
                np.zeros(5, dtype=np.float64),
            )
        )
        scene = _scene_from_metric_points(
            points_m, valid_mask=np.array([True, True, False, True, True])
        )

        patches = extract_overlapping_patches(scene, patch_size_m=10.0, stride_m=5.0)

        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["source_indices"].tolist(), [0, 1, 3, 4])
        self.assertTrue(np.all(patches[0]["valid_mask"]))

    def test_empty_grid_cells_are_optional_and_use_only_invalid_padding(self) -> None:
        points_m = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]], dtype=np.float64)
        scene = _scene_from_metric_points(points_m)

        nonempty = extract_overlapping_patches(scene, 2.0, 2.0, points_per_patch=4)
        complete_grid = extract_overlapping_patches(
            scene,
            2.0,
            2.0,
            points_per_patch=4,
            include_empty=True,
        )

        self.assertEqual(len(nonempty), 2)
        self.assertEqual(len(complete_grid), 4)
        empty = [patch for patch in complete_grid if not np.any(patch["valid_mask"])]
        self.assertEqual(len(empty), 2)
        for patch in empty:
            self.assertTrue(np.all(patch["labels"] == -1))
            self.assertTrue(np.all(patch["source_indices"] == -1))


class FixedPointSamplingTests(unittest.TestCase):
    def test_short_patches_are_padded_to_supported_training_sizes(self) -> None:
        points_m = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.01], [0.2, 0.2, 0.02]],
            dtype=np.float64,
        )
        labels = np.array([0, 7, 2], dtype=np.int64)
        scene = _scene_from_metric_points(points_m, labels)

        for fixed_count in (4096, 8192):
            with self.subTest(fixed_count=fixed_count):
                patch = extract_overlapping_patches(
                    scene,
                    patch_size_m=1.0,
                    stride_m=0.5,
                    points_per_patch=fixed_count,
                )[0]

                self.assertEqual(patch["points"].shape, (fixed_count, 3))
                self.assertEqual(patch["feats"].shape, (fixed_count, 3))
                self.assertEqual(patch["normals"].shape, (fixed_count, 3))
                self.assertEqual(patch["labels"].shape, (fixed_count,))
                self.assertEqual(int(patch["valid_mask"].sum()), 3)
                np.testing.assert_array_equal(patch["labels"][:3], labels)
                self.assertTrue(np.all(patch["labels"][3:] == -1))
                self.assertTrue(np.all(patch["source_indices"][3:] == -1))
                self.assertTrue(np.all(patch["points"][3:] == 0.0))
                self.assertEqual(int(patch["source_point_count"]), 3)

    def test_training_sampling_is_seeded_and_disease_aware(self) -> None:
        points_m = np.column_stack(
            (
                np.linspace(0.0, 0.99, 100),
                np.zeros(100),
                np.zeros(100),
            )
        )
        labels = np.zeros(100, dtype=np.int64)
        labels[:30] = 5
        scene = _scene_from_metric_points(points_m, labels)
        kwargs = {
            "patch_size_m": 2.0,
            "stride_m": 1.0,
            "points_per_patch": 20,
            "split": "train",
            "disease_aware": True,
            "disease_fraction": 0.5,
        }

        first = extract_overlapping_patches(scene, seed=17, **kwargs)[0]
        repeated = extract_overlapping_patches(scene, seed=17, **kwargs)[0]
        changed_seed = extract_overlapping_patches(scene, seed=18, **kwargs)[0]

        self.assertEqual(int(np.count_nonzero(first["labels"] > 0)), 10)
        np.testing.assert_array_equal(first["source_indices"], repeated["source_indices"])
        self.assertFalse(np.array_equal(first["source_indices"], changed_seed["source_indices"]))
        indices = first["source_indices"]
        np.testing.assert_array_equal(first["labels"], labels[indices])
        np.testing.assert_array_equal(first["feats"], scene["feats"][indices])
        np.testing.assert_array_equal(first["normals"], scene["normals"][indices])

    def test_validation_and_test_sampling_are_uniform_and_deterministic(self) -> None:
        points_m = np.column_stack(
            (
                np.linspace(0.0, 0.99, 100),
                np.zeros(100),
                np.zeros(100),
            )
        )
        labels = np.zeros(100, dtype=np.int64)
        labels[:80] = 3
        scene = _scene_from_metric_points(points_m, labels)

        for split in ("val", "test"):
            with self.subTest(split=split):
                aware = extract_overlapping_patches(
                    scene,
                    2.0,
                    1.0,
                    points_per_patch=20,
                    split=split,
                    disease_aware=True,
                    disease_fraction=0.1,
                    seed=42,
                )[0]
                unaware = extract_overlapping_patches(
                    scene,
                    2.0,
                    1.0,
                    points_per_patch=20,
                    split=split,
                    disease_aware=False,
                    disease_fraction=0.9,
                    seed=42,
                )[0]

                np.testing.assert_array_equal(aware["source_indices"], unaware["source_indices"])
                indices = aware["source_indices"]
                self.assertEqual(len(np.unique(indices)), 20)
                np.testing.assert_array_equal(aware["labels"], labels[indices])
                np.testing.assert_array_equal(aware["feats"], scene["feats"][indices])
                np.testing.assert_array_equal(aware["normals"], scene["normals"][indices])


class PatchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _scene_from_metric_points(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        )

    def test_rejects_invalid_geometry_and_sampling_arguments(self) -> None:
        cases = (
            ({"patch_size_m": 0.0, "stride_m": 0.5}, "patch_size_m"),
            ({"patch_size_m": 1.0, "stride_m": 2.0}, "stride_m"),
            ({"patch_size_m": 1.0, "stride_m": 0.5, "points_per_patch": 0}, "points_per_patch"),
            ({"patch_size_m": 1.0, "stride_m": 0.5, "split": "predict"}, "split"),
            ({"patch_size_m": 1.0, "stride_m": 0.5, "seed": -1}, "seed"),
            (
                {"patch_size_m": 1.0, "stride_m": 0.5, "disease_fraction": 1.1},
                "disease_fraction",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                extract_overlapping_patches(self.scene, **kwargs)

    def test_rejects_misaligned_point_fields_and_empty_valid_scene(self) -> None:
        bad_feats = dict(self.scene)
        bad_feats["feats"] = bad_feats["feats"][:-1]
        with self.assertRaisesRegex(ValueError, "feats"):
            extract_overlapping_patches(bad_feats, 1.0, 0.5)

        no_valid = dict(self.scene)
        no_valid["valid_mask"] = np.zeros(2, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "no valid points"):
            extract_overlapping_patches(no_valid, 1.0, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
