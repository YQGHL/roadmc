"""JTG-criterion regression tests for damage-primitive geometry.

每条断言对应 2026-07 学术审计的一项修复：
- 线状裂缝严重度由缝宽（3 mm 判据）决定，measure-then-label
- 错台为板级位移：整个剖面只有一个台阶（无虚假反向台阶）
- 坑槽半径与严重度解耦（消除平面尺寸捷径）
- 波浪拥包空间包络（不再全场贴标）
- 车辙标签边界几何可观测（绝对阈值 ≥2 mm）
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.synthetic.config import (  # noqa: E402
    DiseaseConfig,
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset  # noqa: E402
from roadmc.data.synthetic.primitives import (  # noqa: E402
    JTG_CRACK_WIDTH_THRESHOLD_M,
    add_concrete_damage,
    add_corrugation,
    add_crack,
    add_rutting,
)


def _flat_grid(extent_x: float, extent_y: float, res: float) -> np.ndarray:
    xs = np.arange(0.0, extent_x, res)
    ys = np.arange(0.0, extent_y, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)


class CrackWidthCriterionTests(unittest.TestCase):
    """裂缝轻/重 = 缝宽 3 mm 判据（JTG 5210-2018），非缝深。"""

    def _labeled_widths(self, severity: str, seed: int) -> tuple[np.ndarray, set]:
        """1 mm 网格上的横向裂缝：按 y 列量测标注带宽。"""
        pts = _flat_grid(0.6, 0.3, 0.001)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_crack(
            pts, labels, crack_type="transverse", severity=severity,
            params={"label_width_floor": 0.0}, seed=seed,
        )
        del out_pts
        widths = []
        xs = np.round(pts[:, 0], 6)
        for x_val in np.unique(xs)[::60]:
            col = xs == x_val
            labeled_y = pts[col & (out_lbl > 0), 1]
            if len(labeled_y) > 0:
                widths.append(labeled_y.max() - labeled_y.min() + 0.001)
        return np.asarray(widths), set(np.unique(out_lbl[out_lbl > 0]).tolist())

    def test_light_crack_width_at_most_3mm(self) -> None:
        widths, label_set = self._labeled_widths("light", seed=11)
        self.assertGreater(len(widths), 3)
        self.assertLessEqual(
            float(np.median(widths)), JTG_CRACK_WIDTH_THRESHOLD_M + 0.0015
        )
        self.assertEqual(label_set, {7})  # 轻横向裂缝

    def test_severe_crack_width_above_3mm(self) -> None:
        widths, label_set = self._labeled_widths("severe", seed=11)
        self.assertGreater(len(widths), 3)
        self.assertGreater(
            float(np.median(widths)), JTG_CRACK_WIDTH_THRESHOLD_M
        )
        self.assertEqual(label_set, {8})  # 重横向裂缝

    def test_severity_no_longer_separable_by_depth_alone(self) -> None:
        """轻/重的定义性差异是宽度侧别；深度只是弱相关参数。"""
        pts = _flat_grid(0.6, 0.3, 0.001)
        labels = np.zeros(len(pts), dtype=np.int64)
        _, lbl_light = add_crack(
            pts, labels, crack_type="transverse", severity="light",
            params={"label_width_floor": 0.0, "d_max": 0.030}, seed=5,
        )
        # d_max 被显式设为"重度深度"，标签仍必须按宽度判轻。
        self.assertEqual(set(np.unique(lbl_light[lbl_light > 0]).tolist()), {7})


class FaultingSlabTests(unittest.TestCase):
    """错台 = 整板一次跳变，无 ±40mm 处的虚假反向台阶。"""

    def test_single_step_profile(self) -> None:
        pts = _flat_grid(4.0, 6.0, 0.01)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_concrete_damage(
            pts, labels, damage_type="faulting", severity="severe",
            params={"slab_length": 3.0, "slab_width": 4.0, "d_max": 0.015},
            seed=3,
        )
        self.assertGreater(int((out_lbl > 0).sum()), 0)
        # 按 y 行取中位高程剖面，统计 |Δz| > 3mm 的跳变次数
        ys = np.round(pts[:, 1], 6)
        y_vals = np.unique(ys)
        profile = np.array([np.median(out_pts[ys == yv, 2]) for yv in y_vals])
        steps = np.abs(np.diff(profile))
        n_steps = int((steps > 0.003).sum())
        self.assertEqual(n_steps, 1)
        # 台阶量 = 完整 fault_offset
        self.assertAlmostEqual(float(steps.max()), 0.015, delta=0.002)


class PotholeDecoupleTests(unittest.TestCase):
    """坑槽半径与严重度解耦：轻/重的半径分布必须重叠。"""

    def test_radius_ranges_overlap(self) -> None:
        config = GeneratorConfig(
            road=RoadSurfaceConfig(
                width=3.0, length=3.0, grid_res=0.03, roughness_class="A",
                crossfall=0.0,
            ),
            micro_texture=MicroTextureConfig(amplitude=0.0),
            disease=DiseaseConfig(max_diseases_per_scene=1),
            lidar_noise=LidarNoiseConfig(
                distance_noise_std=0.0, dropout_rate=0.0,
                angular_jitter_deg=0.0, enable_edge_mixing=False,
            ),
            seed=77,
            num_points=8192,
        )
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        area_per_scene_pt = None

        def est_radius(scene, label):
            n = int((scene["labels"] == label).sum())
            frac = n / len(scene["labels"])
            return float(np.sqrt(frac * 9.0 / np.pi))  # 3×3 m 场景

        light_r = []
        severe_r = []
        for i in range(10):
            light_r.append(est_radius(dataset.generate_scene(i, target_label=9), 9))
            severe_r.append(est_radius(dataset.generate_scene(100 + i, target_label=10), 10))
        del area_per_scene_pt
        # 旧实现：轻 ≤0.15m、重 ≥0.15m 完全不相交；解耦后两组范围重叠。
        self.assertGreater(max(light_r), min(severe_r))


class CorrugationEnvelopeTests(unittest.TestCase):
    """波浪拥包是局部病害：标注比例必须远低于旧实现的 ~94%。"""

    def test_labeled_fraction_is_local(self) -> None:
        pts = _flat_grid(7.0, 5.0, 0.05)
        labels = np.zeros(len(pts), dtype=np.int64)
        fracs = []
        for seed in (1, 2, 3):
            _, lbl = add_corrugation(
                pts, labels, direction="transverse", wavelength=0.5,
                amplitude=0.02, severity="light", seed=seed,
            )
            fracs.append(float((lbl > 0).mean()))
        self.assertGreater(min(fracs), 0.001)
        self.assertLess(max(fracs), 0.5)


class RuttingObservabilityTests(unittest.TestCase):
    """车辙标签边界必须几何可观测（≥2 mm），带宽受物理约束。"""

    def test_label_boundary_deformation_observable(self) -> None:
        pts = _flat_grid(7.0, 5.0, 0.01)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_rutting(
            pts, labels, center_line=3.5, wheel_separation=1.8,
            depth=0.012, width=0.6, severity="light", seed=1,
        )
        labeled = out_lbl > 0
        self.assertGreater(int(labeled.sum()), 0)
        dz = pts[:, 2] - out_pts[:, 2]
        self.assertGreaterEqual(float(dz[labeled].min()), 0.002 - 1e-9)
        # 单轮迹标注带宽：解析上 2σ√(2ln(depth/2mm)) ≈ 0.76 m @ 12mm
        left_track = labeled & (pts[:, 0] < 3.5)
        band = pts[left_track, 0]
        self.assertLess(float(band.max() - band.min()), 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
