"""Physical-correctness regression tests for the road surface synthesis.

每条断言对应 2026-07 学术审计的一项修复：
- ISO 8608 各向同性径向谱（等级缩放律 RMS ∝ √Gd0、毫米量级、带限）
- 自仿射纹理谱段（空间相关性、谱斜率、MPD 落入 ISO 13473-1 实测区间）
- 横坡/路拱的法向倾角贡献
- 表面统计各向同性
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.synthetic.config import (  # noqa: E402
    ISO8608_BAND_MAX_CYCLES_PER_M,
    GeneratorConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.primitives import (  # noqa: E402
    add_micro_texture,
    generate_road_surface,
)


def _grid_z(points: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return points[:, 2].reshape(nx, ny)


def _profile_psd_slope(z: np.ndarray, dx: float, f_lo: float, f_hi: float) -> float:
    """Log-log slope of the row-averaged 1D profile PSD inside a band."""
    f = np.fft.rfftfreq(z.shape[1], d=dx)
    psd = np.mean(np.abs(np.fft.rfft(z - z.mean(), axis=1)) ** 2, axis=0)
    band = (f > f_lo) & (f < f_hi)
    coeffs = np.polyfit(np.log(f[band]), np.log(psd[band]), 1)
    return float(coeffs[0])


def estimate_mpd(profile: np.ndarray, dx: float) -> float:
    """Simplified ISO 13473-1 Mean Profile Depth estimator.

    100 mm 基线分两个 50 mm 半段，去线性趋势后取两半段峰值均值
    减段均值。返回 MPD (与 profile 同单位)。
    """
    n_seg = int(round(0.1 / dx))
    half = n_seg // 2
    if n_seg < 4 or len(profile) < n_seg:
        raise ValueError("profile too short for a 100 mm baseline")
    vals = []
    x_loc = np.arange(n_seg) * dx
    for start in range(0, len(profile) - n_seg + 1, n_seg):
        seg = profile[start : start + n_seg]
        seg = seg - np.polyval(np.polyfit(x_loc, seg, 1), x_loc)
        vals.append((seg[:half].max() + seg[half:].max()) / 2.0 - seg.mean())
    return float(np.mean(vals))


class IsoRoughnessTests(unittest.TestCase):
    """宏观粗糙度：径向谱 + Parseval 定标的可观测后果。"""

    def _surface(self, cls: str, seed: int = 42):
        return generate_road_surface(
            5.0, 5.0, 0.01, roughness_class=cls, seed=seed,
            texture_rms=0.0, crossfall=0.0,
        )

    def test_class_a_rms_is_millimetre_scale(self) -> None:
        points, _ = self._surface("A")
        rms_mm = points[:, 2].std() * 1e3
        # 旧实现为 0.003 mm（偏小 287 倍）；1D 带宽解析值 ~0.86 mm。
        self.assertGreater(rms_mm, 0.4)
        self.assertLess(rms_mm, 3.0)

    def test_class_e_rms_scale(self) -> None:
        points, _ = self._surface("E")
        rms_mm = points[:, 2].std() * 1e3
        self.assertGreater(rms_mm, 8.0)
        self.assertLess(rms_mm, 45.0)

    def test_class_scaling_law_is_sqrt_gd0(self) -> None:
        """E/A 的 RMS 比必须为 √(4096/16)=16，而非旧实现的 256。"""
        rms = {}
        for cls in ("A", "E"):
            points, _ = self._surface(cls, seed=42)
            rms[cls] = points[:, 2].std()
        ratio = rms["E"] / rms["A"]
        self.assertAlmostEqual(ratio, 16.0, delta=1.0)

    def test_profile_psd_slope_matches_iso(self) -> None:
        """各向同性 Ω⁻³ 径向谱的一维剖面边缘谱应为 f⁻² 幂律。"""
        points, _ = self._surface("C", seed=7)
        z = _grid_z(points, 500, 500)
        slope = _profile_psd_slope(z, 0.01, 0.3, 2.0)
        self.assertGreater(slope, -3.2)
        self.assertLess(slope, -1.4)

    def test_band_limited_at_iso_upper_edge(self) -> None:
        """2.83 c/m 以上的能量必须交给纹理段（此处纹理关闭 → 近零）。"""
        points, _ = self._surface("E", seed=3)
        z = _grid_z(points, 500, 500)
        f = np.fft.rfftfreq(500, d=0.01)
        psd = np.mean(np.abs(np.fft.rfft(z - z.mean(), axis=1)) ** 2, axis=0)
        in_band = psd[(f > 0.3) & (f < ISO8608_BAND_MAX_CYCLES_PER_M)].mean()
        above_band = psd[f > 2.0 * ISO8608_BAND_MAX_CYCLES_PER_M].mean()
        self.assertLess(above_band, in_band * 5e-2)

    def test_isotropy_axial_vs_diagonal(self) -> None:
        """等径向距离下轴向与对角向自相关应一致（可分离谱的伪影已消除）。"""
        points, _ = self._surface("C", seed=11)
        z = _grid_z(points, 500, 500)
        z = z - z.mean()
        k_diag = 10  # 对角滞后 √2·k·Δ = 0.1414 m
        k_axial = 14  # 轴向滞后 0.14 m
        axial = np.corrcoef(z[:-k_axial, :].ravel(), z[k_axial:, :].ravel())[0, 1]
        diag = np.corrcoef(
            z[:-k_diag, :-k_diag].ravel(), z[k_diag:, k_diag:].ravel()
        )[0, 1]
        self.assertLess(abs(axial - diag), 0.06)


class TextureSpectrumTests(unittest.TestCase):
    """纹理段：空间相关的自仿射谱，MPD 闭环。"""

    def _textured(self, seed: int = 9):
        return generate_road_surface(
            2.0, 2.0, 0.005, roughness_class="A", seed=seed,
            texture_rms=0.0008, texture_hurst=0.7, crossfall=0.0,
        )

    def test_texture_is_spatially_correlated(self) -> None:
        """旧 Lévy 白噪声实现的 lag-1 自相关 ≈ 0，谱合成必须强正相关。"""
        points, _ = self._textured()
        z = _grid_z(points, 400, 400)
        z = z - z.mean()
        lag1 = np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1]
        self.assertGreater(lag1, 0.5)

    def test_texture_band_spectral_slope(self) -> None:
        """纹理带内剖面谱斜率 ≈ -(2H+1) = -2.4（旧实现为平谱 ≈ 0）。"""
        points, _ = self._textured()
        z = _grid_z(points, 400, 400)
        slope = _profile_psd_slope(z, 0.005, 5.0, 80.0)
        self.assertGreater(slope, -3.6)
        self.assertLess(slope, -1.4)

    def test_texture_rms_semantics(self) -> None:
        """amplitude 语义 = 纹理段输出 RMS（Parseval：两带方差相加）。"""
        points_tex, _ = self._textured()
        points_iso, _ = generate_road_surface(
            2.0, 2.0, 0.005, roughness_class="A", seed=9,
            texture_rms=0.0, crossfall=0.0,
        )
        var_total = float(points_tex[:, 2].std()) ** 2
        var_iso = float(points_iso[:, 2].std()) ** 2
        tex_rms_mm = np.sqrt(max(var_total - var_iso, 0.0)) * 1e3
        self.assertAlmostEqual(tex_rms_mm, 0.8, delta=0.2)

    def test_mpd_within_iso13473_range(self) -> None:
        """默认纹理配置的 MPD 必须落在密级配沥青实测区间。"""
        points, _ = self._textured()
        z = _grid_z(points, 400, 400)
        mpds = [estimate_mpd(z[i], 0.005) for i in range(0, 400, 40)]
        mpd_mm = float(np.mean(mpds)) * 1e3
        self.assertGreater(mpd_mm, 0.15)
        self.assertLess(mpd_mm, 1.6)

    def test_default_config_mpd_contract(self) -> None:
        """GeneratorConfig 默认参数经由完整表面路径满足 MPD 契约。"""
        cfg = GeneratorConfig(
            road=RoadSurfaceConfig(width=2.0, length=2.0, grid_res=0.005, crossfall=0.0),
            micro_texture=MicroTextureConfig(),
            seed=5,
        )
        points, _ = generate_road_surface(
            cfg.road.width, cfg.road.length, cfg.road.grid_res,
            roughness_class=cfg.road.roughness_class, seed=5,
            texture_rms=cfg.micro_texture.amplitude,
            texture_hurst=cfg.micro_texture.hurst,
            crossfall=0.0,
        )
        z = _grid_z(points, 400, 400)
        mpd_mm = float(np.mean([estimate_mpd(z[i], 0.005) for i in range(0, 400, 40)])) * 1e3
        self.assertGreater(mpd_mm, 0.15)
        self.assertLess(mpd_mm, 1.6)


class DesignGeometryTests(unittest.TestCase):
    """横坡/路拱：法向倾角贡献与几何形状。"""

    def test_crowned_crossfall_tilts_normals(self) -> None:
        points, normals = generate_road_surface(
            7.0, 5.0, 0.01, roughness_class="A", seed=1,
            texture_rms=0.0, crossfall=0.02, crossfall_shape="crowned",
        )
        tilt_deg = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))
        # 2% 双向路拱 → 名义倾角 atan(0.02) ≈ 1.15°
        self.assertGreater(float(tilt_deg.mean()), 0.8)
        self.assertLess(float(tilt_deg.mean()), 1.6)

    def test_crown_apex_at_centreline(self) -> None:
        points, _ = generate_road_surface(
            7.0, 5.0, 0.01, roughness_class="A", seed=1,
            texture_rms=0.0, crossfall=0.02, crossfall_shape="crowned",
        )
        z = _grid_z(points, 700, 500)
        x = np.arange(700) * 0.01
        col_mean = z.mean(axis=1)
        apex_x = x[int(np.argmax(col_mean))]
        self.assertAlmostEqual(apex_x, 3.5, delta=0.3)

    def test_zero_crossfall_keeps_normals_vertical(self) -> None:
        points, normals = generate_road_surface(
            5.0, 5.0, 0.01, roughness_class="A", seed=1,
            texture_rms=0.0, crossfall=0.0,
        )
        tilt_deg = np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))
        self.assertLess(float(tilt_deg.mean()), 0.4)


class ScatterTexturePathTests(unittest.TestCase):
    """add_micro_texture 散点兼容路径的契约。"""

    def test_displacement_is_correlated_and_rms_calibrated(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        xs, ys = np.meshgrid(np.linspace(0, 2, n), np.linspace(0, 2, n), indexing="ij")
        points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        out, nrm = add_micro_texture(points, normals, amplitude=0.001, hurst=0.7, seed=3)
        dz = (out[:, 2] - points[:, 2]).reshape(n, n)
        lag1 = np.corrcoef(dz[:, :-1].ravel(), dz[:, 1:].ravel())[0, 1]
        self.assertGreater(lag1, 0.5)
        self.assertAlmostEqual(float(dz.std()) * 1e3, 1.0, delta=0.35)
        # 散点路径显式声明不更新法向
        np.testing.assert_array_equal(nrm, normals)
        del rng

    def test_zero_amplitude_is_identity(self) -> None:
        points = np.random.default_rng(1).normal(size=(64, 3))
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (64, 1))
        out, _ = add_micro_texture(points, normals, amplitude=0.0, hurst=0.7, seed=1)
        np.testing.assert_array_equal(out, points)


if __name__ == "__main__":
    unittest.main(verbosity=2)
