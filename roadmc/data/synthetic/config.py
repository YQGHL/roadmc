"""
RoadMC 合成数据生成器配置模块。

使用 dataclass 定义所有生成参数，替代 YAML 配置文件。
所有参数均有类型注解和合理默认值，支持 seed 控制实现可复现生成。

遵循 JTG 5210-2018《公路技术状况评定标准》病害分类体系，共 38 个标签 (0-37)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

# 全局常量

NUM_CLASSES: int = 38
"""JTG 5210-2018 病害分类总数（含背景标签 0）。"""

RESOLUTION_CONTRACT_VERSION: str = "roadmc-resolution-v2"
"""Version of the surface -> sensor -> model resolution metadata contract.

v2 preserves every v1 field and adds the observation-model regime records
(``sensor_output.observation_model`` plus the skipped-step / occlusion-policy
audit when the sensor pipeline is off) and the strip-segmentation plan
(``surface_grid.segmentation`` with per-worker memory estimates).
"""

SURFACE_PEAK_BYTES_PER_POINT: int = 320
"""Conservative peak-RAM estimate for the current float64 synthesis pipeline."""

ASSEMBLY_BYTES_PER_POINT: int = 48
"""Preallocated float32 output assembly: points12 + normals12 + feats12 +
labels4 + power4 + intensity4 bytes per point (segmented strips write per
segment slices into this buffer; float64 vstack is forbidden)."""

FEATURE_SWEEP_BYTES_PER_POINT: int = 64
"""Per-segment (owned rows + halo) cKDTree and blockwise-query temporary
budget for the chunked final-feature recomputation."""

POTHOLE_INFLUENCE_RADIUS_M: float = 0.45
"""Cross-segment attribution bound: a pothole's full influence (severe
radius 0.30 m × 1.3 spall reach + 0.06 m micro-pit radius) stays inside this
envelope, which the default blend overlap (0.7067 m) must always cover."""

DEFAULT_MAX_SURFACE_POINTS: int = 10_000_000
DEFAULT_MAX_SURFACE_MEMORY_MIB: float = 4096.0

ASPHALT_LABELS: tuple[int, ...] = tuple(range(0, 21))
CONCRETE_LABELS: tuple[int, ...] = tuple(range(21, 38))

# ISO 8608 拟合有效带上限 (cycle/m)：波长 0.353 m（1/2.83 ≈ 0.3534）。更短
# 波长属于 ISO 13473 纹理域，由生成器的纹理谱段负责，两带互补不重叠。
ISO8608_BAND_MAX_CYCLES_PER_M: float = 2.83

# ISO 8608 路面功率谱密度参数 — 各粗糙度等级对应的不平整度系数 (×10⁻⁶ m³/cycle)
ISO_ROUGHNESS: dict[str, float] = {
    "A": 16,      # 极好
    "B": 64,      # 好
    "C": 256,     # 一般
    "D": 1024,    # 差
    "E": 4096,    # 很差
}

# JTG 5210-2018 病害标签定义

LABEL_MAP: dict[int, dict[str, str]] = {
    0:  {"type": "背景",       "severity": "-",  "pavement": "通用"},
    1:  {"type": "龟裂",       "severity": "轻", "pavement": "沥青"},
    2:  {"type": "龟裂",       "severity": "重", "pavement": "沥青"},
    3:  {"type": "块状裂缝",   "severity": "轻", "pavement": "沥青"},
    4:  {"type": "块状裂缝",   "severity": "重", "pavement": "沥青"},
    5:  {"type": "纵向裂缝",   "severity": "轻", "pavement": "沥青"},
    6:  {"type": "纵向裂缝",   "severity": "重", "pavement": "沥青"},
    7:  {"type": "横向裂缝",   "severity": "轻", "pavement": "沥青"},
    8:  {"type": "横向裂缝",   "severity": "重", "pavement": "沥青"},
    9:  {"type": "坑槽",       "severity": "轻", "pavement": "沥青"},
    10: {"type": "坑槽",       "severity": "重", "pavement": "沥青"},
    11: {"type": "松散",       "severity": "轻", "pavement": "沥青"},
    12: {"type": "松散",       "severity": "重", "pavement": "沥青"},
    13: {"type": "沉陷",       "severity": "轻", "pavement": "沥青"},
    14: {"type": "沉陷",       "severity": "重", "pavement": "沥青"},
    15: {"type": "车辙",       "severity": "轻", "pavement": "沥青"},
    16: {"type": "车辙",       "severity": "重", "pavement": "沥青"},
    17: {"type": "波浪拥包",   "severity": "轻", "pavement": "沥青"},
    18: {"type": "波浪拥包",   "severity": "重", "pavement": "沥青"},
    19: {"type": "泛油",       "severity": "-",  "pavement": "沥青"},
    20: {"type": "修补",       "severity": "-",  "pavement": "沥青"},
    21: {"type": "破碎板",     "severity": "轻", "pavement": "水泥"},
    22: {"type": "破碎板",     "severity": "重", "pavement": "水泥"},
    23: {"type": "裂缝",       "severity": "轻", "pavement": "水泥"},
    24: {"type": "裂缝",       "severity": "重", "pavement": "水泥"},
    25: {"type": "板角断裂",   "severity": "轻", "pavement": "水泥"},
    26: {"type": "板角断裂",   "severity": "重", "pavement": "水泥"},
    27: {"type": "错台",       "severity": "轻", "pavement": "水泥"},
    28: {"type": "错台",       "severity": "重", "pavement": "水泥"},
    29: {"type": "唧泥",       "severity": "-",  "pavement": "水泥"},
    30: {"type": "边角剥落",   "severity": "轻", "pavement": "水泥"},
    31: {"type": "边角剥落",   "severity": "重", "pavement": "水泥"},
    32: {"type": "接缝料损坏", "severity": "轻", "pavement": "水泥"},
    33: {"type": "接缝料损坏", "severity": "重", "pavement": "水泥"},
    34: {"type": "坑洞",       "severity": "-",  "pavement": "水泥"},
    35: {"type": "拱起",       "severity": "-",  "pavement": "水泥"},
    36: {"type": "露骨",       "severity": "-",  "pavement": "水泥"},
    37: {"type": "修补",       "severity": "-",  "pavement": "水泥"},
}


# 配置数据类

@dataclass
class RoadSurfaceConfig:
    """路面宏观轮廓生成参数。

    Attributes:
        width: 路面宽度 (米)，默认双车道 7m。
        length: 路面长度 (米)，默认 5m。
        grid_res: 网格分辨率 (米)，默认 5mm 即 0.005m。
        pavement_type: 路面类型，'asphalt' 或 'concrete'。
        roughness_class: ISO 8608 粗糙度等级，'A'~'E'。等级指谱系数
            Gd(n0) 的取值；补丁尺寸限制了可表示的低频带，不构成对
            IRI 的声明。
        crossfall: 排水横坡坡率。JTG D50 直线段沥青路面常取 1.5%-2%，
            默认 0.02。设 0 可关闭。
        crossfall_shape: 'crowned' 双向路拱（默认，7m 双车道口径）或
            'plane' 单向坡（超高段）。
        longitudinal_grade: 纵坡坡率，默认 0（生成时可采样小坡度）。
        roughness_mix: ISO 8608 等级 → 非负权重的混合抽样规格（如
            {'D': 0.5, 'E': 0.5}）。None 表示关闭（默认），此时按
            roughness_class 单等级生成；键取值域复用 ISO_ROUGHNESS 的
            键集合。权重不在此处归一化存储，抽样时归一。
        macro_spectrum_mode: 宏观谱权重的构造口径。
            'iso-marginal'（默认）——ISO 带按 ISO 一维边际谱逐 fy 行
            重构权重（G₁^ISO = Gd0·n0²/f²，近轴单元取行残差），离散
            var_iso 与 ISO 边际谱的离散采样和一致（注意与连续积分不
            同：f_min=1/L 处对 1/f² 的左 Riemann 和存在过计，随 L 增
            大收敛），幅值合成 a = √(Φ·Δf)·Ŵ（含 Hermitian 共轭），
            无实现级 std 重定标；ISO 段与纹理段在 2.83 c/m 的带边以
            半个倍频程余弦 taper 过渡（消除谱指数/幅值跳变）。
            'legacy' —— 恢复历史点采样权重 Φ = Gd0·n0²/4·Ω⁻³、实现
            级 std 重定标与带边硬切换（宏观 σ 约为 ISO 边际口径的
            2 倍，保留仅供回归对照，见 CLI --macro-spectrum）。
            'two-segment'（**库内与 CLI 新默认**）——ISO 段构造与
            'iso-marginal' 逐字不变；纹理段按 VT 2015 两区间幂律拆为
            宏观纹理段 (f_tex_lo, f_b] 与微观纹理段 (f_b, f_nyq]，谱
            指数分别取 H_macro=0.34 / H_micro=0.75，在断裂波长 f_b 处
            解析连续（两尺度律 PSD 式 3-55/3-56，vtthesis2015.txt:
            3230-3261；H 具体数值导入而非自校准，需人工核对 VT 原页）。
            primitives 侧两段谱已随本批合入（generate_road_surface /
            generate_strip_surface / _synth_texture），材质 MPD 幅值
            分档同步生效（micro_texture.mpd_target_m）。
            回退 = --macro-spectrum iso-marginal（库内构造
            RoadSurfaceConfig(macro_spectrum_mode='iso-marginal')）。
    """

    width: float = 7.0
    length: float = 5.0
    grid_res: float = 0.005
    pavement_type: Literal["asphalt", "concrete", "mixed"] = "asphalt"
    roughness_class: Literal["A", "B", "C", "D", "E"] = "A"
    crossfall: float = 0.02
    crossfall_shape: Literal["crowned", "plane"] = "crowned"
    longitudinal_grade: float = 0.0
    roughness_mix: dict[str, float] | None = None
    macro_spectrum_mode: Literal["iso-marginal", "legacy", "two-segment"] = (
        "two-segment"
    )

    def __post_init__(self) -> None:
        if self.macro_spectrum_mode not in (
            "iso-marginal", "legacy", "two-segment",
        ):
            raise ValueError(
                "macro_spectrum_mode must be 'iso-marginal', 'legacy' or "
                f"'two-segment', got {self.macro_spectrum_mode!r}"
            )
        if not math.isfinite(self.width) or self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")
        if not math.isfinite(self.grid_res) or self.grid_res <= 0:
            raise ValueError(f"grid_res must be > 0, got {self.grid_res}")
        if not math.isfinite(self.crossfall) or not 0.0 <= self.crossfall <= 0.08:
            raise ValueError(f"crossfall must be in [0, 0.08], got {self.crossfall}")
        if not math.isfinite(self.longitudinal_grade) or abs(self.longitudinal_grade) > 0.10:
            raise ValueError(
                f"longitudinal_grade must be in [-0.10, 0.10], got {self.longitudinal_grade}"
            )
        if self.roughness_mix is not None:
            invalid_keys = sorted(
                key for key in self.roughness_mix if key not in ISO_ROUGHNESS
            )
            if invalid_keys:
                raise ValueError(
                    f"roughness_mix keys must be ISO 8608 classes "
                    f"{sorted(ISO_ROUGHNESS)}, got {invalid_keys}"
                )
            for key, weight in self.roughness_mix.items():
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(
                        f"roughness_mix['{key}'] must be finite and >= 0, got {weight}"
                    )
            total_weight = sum(self.roughness_mix.values())
            if total_weight <= 0:
                raise ValueError(
                    f"roughness_mix weights must sum to > 0, got {total_weight}"
                )

    @property
    def surface_grid_spacing_m(self) -> float:
        """Physical spacing of the synthesis height field in metres.

        ``grid_res`` remains the stored field for API compatibility.  This
        explicit name prevents it from being mistaken for sensor or model
        point spacing.
        """

        return float(self.grid_res)

    @property
    def surface_area_m2(self) -> float:
        """Nominal horizontal road area in square metres."""

        return float(self.width * self.length)

    @property
    def surface_grid_shape(self) -> tuple[int, int]:
        """Exact shape produced by ``np.arange(0, extent, grid_res)``."""

        nx = max(1, int(math.ceil(self.width / self.grid_res)))
        ny = max(1, int(math.ceil(self.length / self.grid_res)))
        return nx, ny

    @property
    def surface_grid_point_count(self) -> int:
        """Number of points allocated before sensor simulation."""

        nx, ny = self.surface_grid_shape
        return nx * ny


@dataclass
class MicroTextureConfig:
    """宏观纹理谱段参数（自仿射表面，谱斜率 -(2H+2)）。

    纹理段占据 ISO 8608 带上限 (2.83 c/m) 到网格 Nyquist 之间的波数
    带，覆盖 ISO 13473 宏观+兆观纹理带上段；5 mm 网格下不可表达
    0.5-10 mm 波长（含宏观纹理主体）。<0.5mm 的真·微观纹理在毫米级
    网格上不可表达，本模块不声明它。

    Attributes:
        amplitude: 纹理位移 RMS (米)。语义为稠密网格（点密度不低于
            网格密度）上严格等于位移 RMS（经 Parseval 校准）；稀疏
            点云上 RMS 衰减（见 primitives.add_micro_texture 文档）。
            默认 0.0008m = 0.8mm（覆盖宏观+兆观纹理带 10-353mm），
            H=0.7 谱形下 MPD（100mm 基线、去趋势、两半段峰值均值减
            段均值估计器）/纹理带 RMS ≈ 0.52，对应 MPD ≈ 0.42-0.45mm
            （2×2m 与 7×5m 补丁实测，回归测试
            tests/test_texture_metrics.py 固定种子断言 MPD ∈
            [0.35, 0.55] mm）。量级与密级配沥青实测 MPD 文献一致
            （测定方法见 ASTM E1845 / ISO 13473-1；量级综述 DOI
            10.1016/j.jreng.2022.05.003），0.3-1.2mm 为设计目标区间
            而非标准值。
        hurst: Hurst 指数 H，控制谱斜率 β = 2H+2。H 默认 0.7；
            0.6-0.9 为工程经验区间（无直接文献支撑，未实测标定）。
        grid_convention: 纹理谱频率网格口径。
            'consistent'（默认）——fftfreq 间距与实空间 linspace
            (x_min, x_max, gx) 的间距 extent/(gx-1) 一致，消除历史
            ~1/gx 的频率口径偏差；
            'legacy' —— 恢复 fftfreq(d=extent/gx) 历史口径（CLI
            --micro-texture-grid-legacy）。
        het_sigma_ln: 纹理沿程不均匀的对数域调制强度 σ。0.0 表示关闭
            （默认，行为与历史一致）；> 0 时纹理段幅值乘以均值为 1 的
            对数正态调制场。σ 与相关长度均为**无依据设计参数**（无
            文献引用）；敏感性：mean(M) ≈ exp(-σ²/2)，σ=0.3 时
            M ∈ (0.585, 1.367)，σ=0.6 时 M ∈ (0.301, 1.642)。
        het_corr_len_m: 调制场的高斯低通相关长度 (米)，仅在
            het_sigma_ln > 0 时生效。
        het_field: 调制场维度。'2d'（默认）——x/y 二维调制场（x/y
            相关长度可异，默认同 het_corr_len_m）；'1d' —— 历史
            沿里程 y 一维、横跨整幅路宽同缩放的调制场（CLI
            --het-field-1d）。
        h_macro / h_micro: two-segment 谱形的宏观/微观纹理 Hurst
            指数（谱斜率 -(2H+2)）。默认 0.34 / 0.75 为 VT 2015
            两区间幂律口径的导入值（两尺度律 PSD 式 3-55/3-56，
            vtthesis2015.txt:3230-3261），H 数值区间本次未在本地
            OCR 文本定位原页——**导入而非自校准，需人工核对**。
            仅 macro_spectrum_mode='two-segment' 时生效；
            iso-marginal/legacy 下纹理仍用标量 hurst。
        break_wavelength_m: two-segment 谱形的断裂波长 λ_b (米)，
            f_b = 1/λ_b 为宏观/微观纹理段分界。默认 0.010 m；VT
            2015 Table 4-2 实测 roll-off 波长高速公路 0.50–2.30 cm
            （vtthesis2015.txt:3568-3577，导入而非自校准）。
            f_b ≥ 网格 Nyquist 时微观段能量为 0（元数据记
            micro_band_empty=true）。
        mpd_target_m: two-segment 下按 pavement_type 抽样的 MPD
            目标区间 (lo, hi) 米（每场景 MPD~U(lo,hi)，纹理幅值
            texture_rms_eff = MPD/mpd_rms_ratio）。None 用内置
            {"asphalt": (0.0008, 0.0013), "concrete": (0.0002,
            0.0008)}——沥青 MPD 0.8–1.3mm、水泥 0.2–0.8mm，
            **导入而非自校准**。iso-marginal/legacy 下忽略，
            纹理幅值仍取 amplitude 标量。
        mpd_rms_ratio: MPD/纹理带 RMS 比值估计器标定。**默认 0.81**
            为 two-segment 谱形（H_macro=0.34，5mm 默认网格、微观段
            为空）下的工程标定值：7×5m 补丁 4 个种子实测 MPD/RMS =
            0.812–0.814（估计器口径同 tests/test_texture_metrics.py，
            带通 (2.83, Nyquist] c/m）；历史单段 H=0.7 谱形下的 0.52
            见 amplitude docstring。该比值随谱形与网格漂移（微观段
            实体化的 ≤2mm 网格下需另行标定）——grid_res<5mm 且依赖
            MPD 目标带时请显式覆盖本字段。
        het_corr_len_m: 调制场的高斯低通相关长度 (米)，仅在
            het_sigma_ln > 0 时生效。**默认 0.01**（1cm 相关长度；
            历史值 2.0 由 CLI --het-corr-len-m 2.0 恢复）。1cm 在
            5mm 网格仅 2 格，het 开启时建议 grid_res ≤ 2mm 并已
            写入 resolution_contract。
    """

    amplitude: float = 0.0008
    hurst: float = 0.7
    grid_convention: Literal["consistent", "legacy"] = "consistent"
    het_sigma_ln: float = 0.0
    het_corr_len_m: float = 0.01
    het_field: Literal["2d", "1d"] = "2d"
    h_macro: float = 0.34
    h_micro: float = 0.75
    break_wavelength_m: float = 0.010
    mpd_target_m: dict[str, tuple[float, float]] | None = None
    mpd_rms_ratio: float = 0.81

    def __post_init__(self) -> None:
        if self.grid_convention not in ("consistent", "legacy"):
            raise ValueError(
                "grid_convention must be 'consistent' or 'legacy', "
                f"got {self.grid_convention!r}"
            )
        if self.het_field not in ("2d", "1d"):
            raise ValueError(
                f"het_field must be '2d' or '1d', got {self.het_field!r}"
            )
        if not math.isfinite(self.het_sigma_ln) or self.het_sigma_ln < 0.0:
            raise ValueError(
                f"het_sigma_ln must be finite and ≥ 0, got {self.het_sigma_ln}"
            )
        # het 关闭时（sigma_ln == 0）不校验相关长度——与既有行为一致。
        if self.het_sigma_ln > 0.0 and (
            not math.isfinite(self.het_corr_len_m) or self.het_corr_len_m <= 0.0
        ):
            raise ValueError(
                "het_corr_len_m must be finite and > 0 when het_sigma_ln > 0, "
                f"got {self.het_corr_len_m}"
            )
        for name in ("h_macro", "h_micro", "mpd_rms_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not math.isfinite(self.break_wavelength_m) or self.break_wavelength_m <= 0.0:
            raise ValueError(
                f"break_wavelength_m must be finite and > 0, "
                f"got {self.break_wavelength_m}"
            )
        if self.mpd_target_m is not None:
            for key, pair in self.mpd_target_m.items():
                lo, hi = pair
                if key not in ("asphalt", "concrete"):
                    raise ValueError(
                        f"mpd_target_m keys must be 'asphalt'/'concrete', "
                        f"got {key!r}"
                    )
                if not (0.0 < lo <= hi):
                    raise ValueError(
                        f"mpd_target_m[{key!r}] must satisfy 0 < lo <= hi, "
                        f"got {pair}"
                    )


@dataclass
class CrackConfig:
    """裂缝生成参数。

    Attributes:
        crack_types: 允许生成的裂缝类型列表。
        severity_ratio: 轻度裂缝占比的 **per-type 覆盖**。None（默认）
            表示使用 ``disease.severity_ratio`` 全局值；设为 float 时
            仅覆盖裂缝类的轻度占比（历史版本此处为 0.6 但从未被读取，
            属静默失效——现接线为覆盖键）。
        bezier_points: Bézier 曲线控制点数，默认 4。
        fractal_perturbation: value-noise 分形扰动幅度（fBm 型多倍频
            value noise，非 Perlin 梯度噪声——见 primitives.py 实现）。
        perturb_mode: 裂缝走向扰动口径。'nonperiodic'（默认）——在
            覆盖全长的无界域网格上插值，无周期重复；'legacy' ——
            恢复 (x·f) % 1 周期化环绕（每 0.5 m 精确重复，CLI
            --crack-perturb-legacy）。注意切换会改变 rng 流消耗，
            同 seed 场景整体漂移属预期。
        width_truncation: 对数正态缝宽判据侧截断方式。
            'resample'（默认）——越界拒绝重采样（平滑截断分布）；
            'clip' —— 历史钳制到判据边界（分布原子，CLI
            --crack-width-trunc clip）。
        area_lambda_mode: 面状裂缝（龟裂/块裂）槽形 λ 口径。
            'quarter'（默认）——λ=w/4，缝缘深度 e⁻⁴≈1.8%·d_max，与
            线状裂缝的边缘连续原则一致；'half' —— 历史 λ=w/2（缝缘
            0.368·d_max 台阶，CLI --areacrack-lambda-legacy）。
        area_label_mode: 面状裂缝标签口径。'measure'（默认）——按
            Voronoi 单元等效边长度量对照 JTG 块度阈值定标签
            （measure-then-label，与线状裂缝同型；阈值需人工核对
            JTG 5210-2018 原文）；'legacy' —— 直接采信 severity
            字符串（CLI --areacrack-label-legacy）。
        profile_exponent: 缝深剖面指数 p：d(r) = d_max·exp(-(r/λ)^p)。
            默认 2.0（高斯槽，现行行为）；p→1 为近线性 V 形侧壁。
            3D 激光剖面实测裂缝呈深 V 形、缝深 1–5 mm（DOI
            10.1109/ACCESS.2019.2956758），文献未给出 p 数值——
            **导入依据、p 值导入而非自校准**。CLI
            --crack-profile-exponent。
        edge_band_width_m: 缝缘崩碎/松散过渡带宽度 (米)。0（默认）
            = 关闭（现行行为）；> 0 时沿缝缘生成线性衰减的崩碎带，
            **标签掩码同步外扩同宽度**（标签边界=几何边界由构造
            保证）。文献仅对缝缘破损做定性分级（DOI
            10.3390/coatings15070836），无宽度实测——**工程假设/
            导入而非自校准，宽度需自行量测**。CLI
            --crack-edge-band-width。
        edge_band_depth_ratio: 缝缘过渡带深度比例（相对该处缝深），
            仅 edge_band_width_m > 0 时生效；工程假设，无文献值。
    """

    crack_types: list[str] = field(
        default_factory=lambda: ["longitudinal", "transverse", "alligator", "block"]
    )
    severity_ratio: float | None = None
    bezier_points: int = 4
    fractal_perturbation: float = 0.02
    perturb_mode: Literal["nonperiodic", "legacy"] = "nonperiodic"
    width_truncation: Literal["resample", "clip"] = "resample"
    area_lambda_mode: Literal["quarter", "half"] = "quarter"
    area_label_mode: Literal["measure", "legacy"] = "measure"
    profile_exponent: float = 2.0
    edge_band_width_m: float = 0.0
    edge_band_depth_ratio: float = 0.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.profile_exponent) or self.profile_exponent <= 0.0:
            raise ValueError(
                f"profile_exponent must be finite and > 0, "
                f"got {self.profile_exponent}"
            )
        if not math.isfinite(self.edge_band_width_m) or self.edge_band_width_m < 0.0:
            raise ValueError(
                f"edge_band_width_m must be finite and >= 0, "
                f"got {self.edge_band_width_m}"
            )
        if not math.isfinite(self.edge_band_depth_ratio) or not (
            0.0 < self.edge_band_depth_ratio <= 1.0
        ):
            raise ValueError(
                f"edge_band_depth_ratio must be in (0, 1], "
                f"got {self.edge_band_depth_ratio}"
            )
        if self.severity_ratio is not None and not (
            math.isfinite(self.severity_ratio) and 0.0 <= self.severity_ratio <= 1.0
        ):
            raise ValueError(
                "severity_ratio must be None or in [0,1], "
                f"got {self.severity_ratio}"
            )
        if self.perturb_mode not in ("nonperiodic", "legacy"):
            raise ValueError(
                f"perturb_mode must be 'nonperiodic' or 'legacy', got {self.perturb_mode!r}"
            )
        if self.width_truncation not in ("resample", "clip"):
            raise ValueError(
                f"width_truncation must be 'resample' or 'clip', got {self.width_truncation!r}"
            )
        if self.area_lambda_mode not in ("quarter", "half"):
            raise ValueError(
                f"area_lambda_mode must be 'quarter' or 'half', got {self.area_lambda_mode!r}"
            )
        if self.area_label_mode not in ("measure", "legacy"):
            raise ValueError(
                f"area_label_mode must be 'measure' or 'legacy', got {self.area_label_mode!r}"
            )


@dataclass
class PotholeConfig:
    """坑槽生成参数。

    轻/重仅按深度分级（JTG ≤25mm 口径，阈值原文需人工核对）；半径
    不按严重度切分（解耦设计：平面半径不得成为 100% 可分的非规范
    线索），轻/重 (R, d) 支撑集重叠。

    Attributes:
        max_radius_severe: 坑槽最大半径 (米)（轻/重共用同一采样区间
            [0.05, max_radius_severe]）。
        max_depth_light: 轻度坑槽最大深度 (米)，JTG ≤ 25mm。
        max_depth_severe: 重度坑槽最大深度 (米)，JTG > 25mm。
        beta_range: 坑槽超椭圆指数 β 的采样区间 (lo, hi)。默认
            (2.0, 5.0)：轻/重两级共用、重叠支撑集（历史版本轻度恒
            β=2、重度 U(3,5)，仅凭剖面形状即可 100% 判定轻/重，构成
            几何域标签泄漏——已消除）。
        beta_mode: β 口径。'overlap'（默认，轻/重共用 beta_range）；
            'legacy' —— 恢复轻=2.0 字面量、重~U(beta_range) 的历史
            行为（CLI --pothole-beta-legacy，同时建议把 beta_range
            复位为 (3.0, 5.0) 以逐位复现旧行为）。
        aspect_ratio_range: 有界纵横比先验 (lo, hi)，d/R 越界拒绝
            重采（排除沉陷碟形与近竖井类非物理极端几何；设计界，
            非标准规定）。None = 恢复 R、d 独立采样（CLI
            --pothole-aspect-legacy）。
        spall_band: 坑缘剥落环的半径带 (r_lo·R, r_hi·R)。默认
            (0.8, 1.0)：剥落标注不超出主凹陷足迹（历史 0.8–1.2R 使
            标注面积最大 1.44× 凹陷面积，CLI --pothole-spall-band）。
        micropit_rate: 微坑泊松强度基准（λ = micropit_rate ·
            (R/0.30)²，随破损面积增长；工程设定，取审计建议 3-5 的
            中值）。历史为 Poisson(max(1,int(R·5))) ≡ Poisson(1)。
        micropit_mode: 微坑口径。'area'（默认，λ 随面积增长、始终
            生成）；'legacy' —— 恢复钳制到 1 的历史计数且仅在
            edge_quality < 1.0 时生成（CLI --pothole-micropits-legacy）。
        spall_profile: 坑缘剥落深度剖面口径。'uniform'（默认，现行
            行为——(0.8, 1.0)·R 均匀带）；'linear-decay' —— 剥落
            深度从剥落带内缘到外缘线性衰减。坑缘崩碎梯度无可及
            文献实测，两档均标**工程假设**。
    """

    max_radius_severe: float = 0.30
    max_depth_light: float = 0.025
    max_depth_severe: float = 0.10
    beta_range: tuple[float, float] = (2.0, 5.0)
    beta_mode: Literal["overlap", "legacy"] = "overlap"
    aspect_ratio_range: tuple[float, float] | None = (0.05, 0.6)
    spall_band: tuple[float, float] = (0.8, 1.0)
    micropit_rate: float = 4.0
    micropit_mode: Literal["area", "legacy"] = "area"
    spall_profile: Literal["uniform", "linear-decay"] = "uniform"

    def __post_init__(self) -> None:
        if self.spall_profile not in ("uniform", "linear-decay"):
            raise ValueError(
                f"spall_profile must be 'uniform' or 'linear-decay', "
                f"got {self.spall_profile!r}"
            )
        lo, hi = self.beta_range
        if not 0.0 < lo <= hi:
            raise ValueError(f"beta_range must satisfy 0 < lo <= hi, got {self.beta_range}")
        if self.beta_mode not in ("overlap", "legacy"):
            raise ValueError(
                f"beta_mode must be 'overlap' or 'legacy', got {self.beta_mode!r}"
            )
        if self.micropit_mode not in ("area", "legacy"):
            raise ValueError(
                f"micropit_mode must be 'area' or 'legacy', got {self.micropit_mode!r}"
            )
        if not math.isfinite(self.micropit_rate) or self.micropit_rate <= 0:
            raise ValueError(
                f"micropit_rate must be finite and > 0, got {self.micropit_rate}"
            )
        if self.aspect_ratio_range is not None:
            alo, ahi = self.aspect_ratio_range
            if not 0.0 < alo <= ahi:
                raise ValueError(
                    "aspect_ratio_range must satisfy 0 < lo <= hi, got "
                    f"{self.aspect_ratio_range}"
                )
        slo, shi = self.spall_band
        if not 0.0 <= slo < shi:
            raise ValueError(
                f"spall_band must satisfy 0 <= lo < hi, got {self.spall_band}"
            )


@dataclass
class RuttingConfig:
    """车辙生成参数。

    Attributes:
        wheel_separation: 重车轮距 (米)，常规取值 1.8m。这是车辆构造
            参数，**非 BZZ-100 规范参数**（BZZ-100 双轮组内距
            3δ ≈ 0.32m）。
        rut_width: 单条轮迹宽度 (米)，默认 0.6m。
        max_depth_light: 轻度车辙最大深度 (米)，JTG 10-15mm。
        max_depth_severe: 重度车辙最大深度 (米)，JTG > 15mm。
        severe_min_depth: 重度车辙最小深度 (米)，默认 0.0165。
            避开 15mm 分界 ±10% 容差，为分级可分性设计、非标准规定。
        center_mode: 车辙中心横向口径。'wheelpath'（默认）——走
            _sample_wheelpath_x 轮迹带先验（车道中心 ±0.9m 双峰、
            σ=0.25m），与裂缝/坑槽先验一致；7m 双车道口径仅建模
            内侧轮迹对。'fixed' —— 历史恒位于道路中线（CLI
            --rutting-center-legacy）。
        peak_normalize: True（默认）——名义深度按 d' = d/(1+ε) 采样，
            使实现峰值 ≤ 名义深度 d（不再越过 JTG 10/15mm 类分界）；
            False —— 历史 unnormalized 口径（CLI
            --rutting-modulation-legacy）。
        modulation_epsilon: 纵向正弦调制幅度 ε，默认 0.2。设计假设：
            纵向变异的真实来源是重车横向分布与轴载波动（Said 2013,
            ntnu.no），系随机过程而非固定 20% 正弦。
        modulation_wavelength_ratio: 纵向调制波长 Λ 与条带长之比，
            默认 0.5。
    """

    wheel_separation: float = 1.8
    rut_width: float = 0.6
    max_depth_light: float = 0.015
    max_depth_severe: float = 0.040
    severe_min_depth: float = 0.0165
    center_mode: Literal["wheelpath", "fixed"] = "wheelpath"
    peak_normalize: bool = True
    modulation_epsilon: float = 0.2
    modulation_wavelength_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.center_mode not in ("wheelpath", "fixed"):
            raise ValueError(
                f"center_mode must be 'wheelpath' or 'fixed', got {self.center_mode!r}"
            )
        if not 0.0 <= self.modulation_epsilon <= 1.0:
            raise ValueError(
                f"modulation_epsilon must be in [0,1], got {self.modulation_epsilon}"
            )
        if not math.isfinite(self.severe_min_depth) or self.severe_min_depth < 0:
            raise ValueError(
                f"severe_min_depth must be finite and >= 0, got {self.severe_min_depth}"
            )
        if not self.max_depth_light <= self.severe_min_depth <= self.max_depth_severe:
            raise ValueError(
                "severe_min_depth must lie in [max_depth_light, max_depth_severe], "
                f"got {self.severe_min_depth} in [{self.max_depth_light}, "
                f"{self.max_depth_severe}]"
            )
        if not math.isfinite(self.modulation_wavelength_ratio) or (
            self.modulation_wavelength_ratio <= 0
        ):
            raise ValueError(
                "modulation_wavelength_ratio must be finite and > 0, got "
                f"{self.modulation_wavelength_ratio}"
            )


@dataclass
class CorrugationConfig:
    """波浪拥包生成参数。

    Attributes:
        wavelength_range: 波长范围 (米)。
        amplitude_light: 轻度振幅 (米)，JTG 10-25mm（口径为单侧波高；
            若按峰谷差解读则为 2A——JTG 5210-2018 拥包分级口径需人工
            核对标准原文，当前未核实）。
        amplitude_severe: 重度振幅 (米)，JTG > 25mm（口径同上，未核实）。
    """

    wavelength_range: tuple[float, float] = (0.3, 1.0)
    amplitude_light: tuple[float, float] = (0.010, 0.025)
    amplitude_severe: tuple[float, float] = (0.025, 0.050)


@dataclass
class DepressionConfig:
    """沉陷生成参数。

    Attributes:
        radius_range: 沉陷半径采样区间 (米)，默认 (0.5, 2.0)。
        depth_light: 轻度深度范围 (米)，JTG 10-25mm。
        depth_severe: 重度深度范围 (米)，JTG > 25mm。
        center_inset_mode: 中心内缩口径。'radius'（默认）——中心内缩
            量 ≥ 当次采样半径（cx ∈ [R, W−R]、cy ∈ [R, L−R]；若
            R > min(W, L)/2 则拒绝重采 R），完整碗形恒在路面足迹内；
            'legacy' —— 历史固定内缩 0.5m（碗形可被路缘截断，CLI
            --depression-center-legacy）。
        label_radius_mode: 标签半径口径。'R'（默认）——标签盘半径
            = 影响半径 R（边界信号 = depth·e⁻⁴·⁵ ≈ 1.1%·depth，
            0.1-0.8mm，远低于 σ_r=10mm，逐点不可分辨、依靠区域聚合
            可恢复）；'legacy-1pct' —— 历史 |depression| > depth·1%
            隐式外扩（等价 1.0116R，CLI --depression-label-legacy）。
    """

    radius_range: tuple[float, float] = (0.5, 2.0)
    depth_light: tuple[float, float] = (0.010, 0.025)
    depth_severe: tuple[float, float] = (0.025, 0.080)
    center_inset_mode: Literal["radius", "legacy"] = "radius"
    label_radius_mode: Literal["R", "legacy-1pct"] = "R"

    def __post_init__(self) -> None:
        lo, hi = self.radius_range
        if not 0.0 < lo <= hi:
            raise ValueError(
                f"radius_range must satisfy 0 < lo <= hi, got {self.radius_range}"
            )
        if self.center_inset_mode not in ("radius", "legacy"):
            raise ValueError(
                "center_inset_mode must be 'radius' or 'legacy', got "
                f"{self.center_inset_mode!r}"
            )
        if self.label_radius_mode not in ("R", "legacy-1pct"):
            raise ValueError(
                "label_radius_mode must be 'R' or 'legacy-1pct', got "
                f"{self.label_radius_mode!r}"
            )


@dataclass
class RavelingConfig:
    """松散生成参数。

    轻度移除 ~5% 点、重度 ~20%；侵蚀深度 ~N(0.5d, 0.3d) 截断到
    [0, 1.5d]，轻 d=2mm、重 d=5mm，均为未标定工程假设；2.5mm只是
    重度参数0.5d，不是实测结论。历史版
    本这三个字段从未被读取（primitives 内硬编码同值）——现已接线
    为 add_raveling params 覆盖键 'removal_ratio'/'erosion_depth'。

    Attributes:
        point_removal_ratio_light: 轻度松散点移除比例。
        point_removal_ratio_severe: 重度松散点移除比例。
        erosion_depth: 侵蚀深度基准 d 的 (轻, 重) 二元组 (米)。字段名
            从 pitting 改为 erosion：实现是 i.i.d. 高斯侵蚀，非点蚀
            坑洞（名实相符；schema 变更见 changelog）。
    """

    point_removal_ratio_light: float = 0.05
    point_removal_ratio_severe: float = 0.20
    erosion_depth: tuple[float, float] = (0.002, 0.005)

    def __post_init__(self) -> None:
        for name in ("point_removal_ratio_light", "point_removal_ratio_severe"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        lo, hi = self.erosion_depth
        if not 0.0 < lo <= hi:
            raise ValueError(
                f"erosion_depth must satisfy 0 < lo <= hi, got {self.erosion_depth}"
            )


@dataclass
class BleedingConfig:
    """泛油生成参数。

    Attributes:
        intensity_multiplier: 泛油区域反照率乘子，默认 0.55（油膜
            NIR 吸收偏暗，方向有据；数值为**占位值待实测标定**）。
            历史 0.3/0.99 的 intensity_boost 字段对输出零影响，已
            删除（schema 变更见 changelog）。
        geometry_mode: 泛油几何口径。'patch'（默认）——局部斑块：
            中心走轮迹带先验、y 向有限长度、面积占比 U(5%,25%)
            （设计先验）、logistic 平滑边界（过渡带 ~0.1m）；
            'legacy-strip' —— 历史恒定 40% 面积、贯穿全路长、硬二值
            边界的竖直条带（CLI --bleeding-geometry-legacy）。
        region_width: 条带宽度比例 (0-1)。仅在
            geometry_mode='legacy-strip' 时生效（历史硬编码路径）。
    """

    intensity_multiplier: float = 0.55
    geometry_mode: Literal["patch", "legacy-strip"] = "patch"
    region_width: float = 0.4

    def __post_init__(self) -> None:
        if self.geometry_mode not in ("patch", "legacy-strip"):
            raise ValueError(
                f"geometry_mode must be 'patch' or 'legacy-strip', got "
                f"{self.geometry_mode!r}"
            )
        if not math.isfinite(self.intensity_multiplier) or self.intensity_multiplier <= 0:
            raise ValueError(
                "intensity_multiplier must be finite and > 0, got "
                f"{self.intensity_multiplier}"
            )


@dataclass
class PatchingConfig:
    """Geometric limits for asphalt and concrete repair patches.

    Attributes:
        min_width / max_width: 修补宽度范围 (米)。
        min_length / max_length: 修补长度范围 (米)。
        edge_width: 边缘混合带宽度 (米)。
        max_elevation: 修补抬升上限 (米)。
        elevation_range: 修补面相对参考面的竖向采样域 (米)，默认
            (-0.006, 0.015)（= 历史实际采样域 [-6, +15]mm；注意不是
            『±15mm』）。负侧 0.4 系数为压实沉降先验（工程假设，无
            标准依据；北京 DB11 接茬 +0~+5mm、深圳 ≤3mm 为检索摘要，
            需人工核对，未据此改值）。
        edge_profile: 修补边缘过渡剖面。'smoothstep'（默认，现行
            行为——三次 smoothstep 混合）；'step-fillet' —— 尖锐
            阶跃 + 小圆角。两种剖面均属工程假设；接缝错台研究不能
            直接标定沥青修补边缘形状。历史引用DOI 10.3390/s18010044、
            10.1061/9780784413586.048本轮未核验全文，不据此主张实测支持。
        edge_fillet_m: 'step-fillet' 剖面的圆角宽度 (米)，仅
            edge_profile='step-fillet' 时生效；工程假设。
    """

    min_width: float = 0.60
    max_width: float = 1.80
    min_length: float = 0.80
    max_length: float = 2.80
    edge_width: float = 0.08
    max_elevation: float = 0.015
    elevation_range: tuple[float, float] = (-0.006, 0.015)
    edge_profile: Literal["smoothstep", "step-fillet"] = "smoothstep"
    edge_fillet_m: float = 0.01

    def __post_init__(self) -> None:
        if self.edge_profile not in ("smoothstep", "step-fillet"):
            raise ValueError(
                f"edge_profile must be 'smoothstep' or 'step-fillet', "
                f"got {self.edge_profile!r}"
            )
        if not math.isfinite(self.edge_fillet_m) or self.edge_fillet_m < 0.0:
            raise ValueError(
                f"edge_fillet_m must be finite and >= 0, got {self.edge_fillet_m}"
            )
        if not 0.0 < self.min_width <= self.max_width:
            raise ValueError("patch width limits must satisfy 0 < min_width <= max_width")
        if not 0.0 < self.min_length <= self.max_length:
            raise ValueError("patch length limits must satisfy 0 < min_length <= max_length")
        if self.edge_width <= 0.0:
            raise ValueError("edge_width must be > 0")
        if self.max_elevation < 0.0:
            raise ValueError("max_elevation must be >= 0")
        lo, hi = self.elevation_range
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError(
                f"elevation_range must satisfy lo < hi, got {self.elevation_range}"
            )


@dataclass
class ConcreteDamageConfig:
    """水泥路面损坏生成参数。

    Attributes:
        slab_length: 水泥板块长度 (米)，默认 5m。
        slab_width: 水泥板块宽度 (米)，默认 4m。
        joint_width: 接缝宽度 (米)，默认 8mm。
        x_offset: 板块铺设 x 方向偏移 (米)，默认 0。
        y_offset: 板块铺设 y 方向偏移 (米)，默认 0。
        crack_in_slab: 板裂缝（slab_crack）是否限制在单一板块内。
            True（默认）——在板内采样走向与位置并裁剪到板边（历史
            版本为过全云 bbox 中心的无限长直线，可跨 2-3 块板并穿越
            接缝）；False —— 历史口径。
        fault_scope: 错台（faulting）±d_max/2 阶跃的影响域。
            'adjacent-slabs'（默认）——仅接缝两侧相邻两块板位移，远
            处板块保持原高程；'legacy-strip' —— 历史全条带位移
            （100% 点被移动）。
        edge_spall_mode: 边角剥落（edge_spall）深度/空间口径。
            'patch-truncated'（默认）——局部斑块内空间相关的连续剥
            落面，深度截断到 d_max（轻 10mm/重 30mm）；'legacy' ——
            历史接缝带内逐点无界指数深度（单点可达 ~108mm 尖刺）。
        shatter_mode: 破碎板（slab_shatter）Voronoi 刻槽口径。
            'clipped-max'（默认）——无限脊线按板 bbox 裁剪后一并刻
            槽、多脊线命中取最大深度（单点 |Δz| ≤ d_max）；'legacy'
            —— 历史仅有限脊线、逐条累加（三线交汇可达 3×d_max）。
        以上四项的聚合回退为 CLI --concrete-legacy（一次性恢复四项
        旧行为，供回归门使用）。
        fault_fillet_m: 错台（faulting）接缝阶跃的平滑过渡宽度 (米)。
            0（默认）= 现行纯阶跃（primitives faulting 分支）；> 0
            时在该宽度内做平滑过渡。实测口径为局部阶跃（DOI
            10.3390/s18010044），默认保持纯阶跃。
    """

    slab_length: float = 5.0
    slab_width: float = 4.0
    joint_width: float = 0.008
    x_offset: float = 0.0
    y_offset: float = 0.0
    crack_in_slab: bool = True
    fault_scope: Literal["adjacent-slabs", "legacy-strip"] = "adjacent-slabs"
    edge_spall_mode: Literal["patch-truncated", "legacy"] = "patch-truncated"
    shatter_mode: Literal["clipped-max", "legacy"] = "clipped-max"
    fault_fillet_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.fault_fillet_m) or self.fault_fillet_m < 0.0:
            raise ValueError(
                f"fault_fillet_m must be finite and >= 0, got {self.fault_fillet_m}"
            )
        if self.fault_scope not in ("adjacent-slabs", "legacy-strip"):
            raise ValueError(
                f"fault_scope must be 'adjacent-slabs' or 'legacy-strip', got "
                f"{self.fault_scope!r}"
            )
        if self.edge_spall_mode not in ("patch-truncated", "legacy"):
            raise ValueError(
                f"edge_spall_mode must be 'patch-truncated' or 'legacy', got "
                f"{self.edge_spall_mode!r}"
            )
        if self.shatter_mode not in ("clipped-max", "legacy"):
            raise ValueError(
                f"shatter_mode must be 'clipped-max' or 'legacy', got "
                f"{self.shatter_mode!r}"
            )


@dataclass
class AmEnvelopeConfig:
    """振幅调制（AM）慢变包络参数（默认关闭）。

    改进 Gamma 慢变包络 ×载波：对基底高度场按 Wcut 分带，慢变
    Gamma 包络 M 调制短波（纹理）分量，再经保秩 CDF 变换钉住目标
    边际矩（skew/kurt）。默认关闭 → 默认输出逐位不变。

    Attributes:
        enable: 是否启用 AM 包络。False（默认）= 关闭。
        wcut_cycles_per_m: 快/慢波分界频率 (cycle/m)。默认 2.83 =
            ISO 带边界（包络波长 > 0.35 m）。
        gamma_shape: Gamma 包络形状参数 k（scale=1/k 使 E[M]=1）。
        target_skew: CDF 变换的目标偏度。默认 0.0（道路表面高度
            边际近零偏——**导入而非自校准**）。
        target_kurt: CDF 变换的目标峰度。默认 4.5（正超额峰度量
            级——**导入而非自校准**；目标区间 skew∈[-0.35,0.35]、
            kurt∈[3.2,6.0] 按任务给定文献量级设定，本地未检到原文
            实测 skew/kurt，需人工核对）。
    """

    enable: bool = False
    wcut_cycles_per_m: float = 2.83
    gamma_shape: float = 4.0
    target_skew: float = 0.0
    target_kurt: float = 4.5

    def __post_init__(self) -> None:
        for name in ("wcut_cycles_per_m", "gamma_shape"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        for name in ("target_skew", "target_kurt"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if self.target_kurt < 3.0:
            raise ValueError(
                f"target_kurt must be >= 3.0 (Gaussian reference), "
                f"got {self.target_kurt}"
            )


@dataclass
class BaseConditionConfig:
    """基层条件与病害因果链参数（默认关闭，旧行为逐位不变）。

    开启后每场景抽取 (SNP, 龄期, YE4) 并由 HDM-4 路线推导结构性
    裂缝起始年 ICA；age_years < ICA 时结构性裂缝出现概率强制为 0。
    系数为 **导入系数、非自校准**（HDM-4 Volume Five Table B3-4，
    经 Malaysian J. Civil Eng. 27(1):121-143, 2015 转引，
    utm.txt:677-684 核实）。

    Attributes:
        enable: 是否启用因果链。False（默认）= 关闭。
        snp_range: 平均结构数 SNP 抽样区间。
        age_years_range: 场景龄期 (年) 抽样区间。
        ye4_msa_range: 年均当量轴次 YE4 (MSA/车道) 抽样区间。
        cds: 施工缺陷指标 CDS（沥青面层）。
        crt_years: 养护裂缝延迟时间 CRT (年)。
        hdm4_a0 / hdm4_a1 / hdm4_a2: HDM-4 ICA 公式系数
            （ICA = K_cia·{CDS^a0·exp(a1·SNP + a2·YE4/SNP²) + CRT}）。
        tpm_path: Markov 状态转移概率矩阵 JSON 路径。None = 不加载
            （disease_probs 不被 π 占用分布调制）。JSON 需含
            states/matrix/provenance/source 四键，加载时校验行和
            =1±1e-6、对角占优、sound 行上三角（病害不可逆）。
        pothole_cooccur_base: 坑槽-龟裂共现概率基准（工程假设）。
            enable 时共现概率 = base·clip(age/ICA−1, 0, 1)
            + 0.5·base（函数形式为工程假设，显著标注）；disable 时
            保持历史 0.5 字面量。
    """

    enable: bool = False
    snp_range: tuple[float, float] = (3.0, 5.5)
    age_years_range: tuple[float, float] = (0.0, 15.0)
    ye4_msa_range: tuple[float, float] = (0.05, 2.0)
    cds: float = 1.0
    crt_years: float = 0.0
    hdm4_a0: float = 4.21
    hdm4_a1: float = 0.14
    hdm4_a2: float = -17.1
    tpm_path: str | None = None
    pothole_cooccur_base: float = 0.5

    def __post_init__(self) -> None:
        for name in ("snp_range", "age_years_range", "ye4_msa_range"):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi):
                raise ValueError(
                    f"{name} must satisfy 0 <= lo < hi, got {getattr(self, name)}"
                )
        for name in ("cds", "crt_years", "hdm4_a0", "hdm4_a1",
                     "hdm4_a2", "pothole_cooccur_base"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.pothole_cooccur_base <= 1.0:
            raise ValueError(
                f"pothole_cooccur_base must be in [0, 1], "
                f"got {self.pothole_cooccur_base}"
            )


@dataclass
class LineProfilerConfig:
    """线激光剖面仪（line profiler）传感器配置。

    采样步距及噪声默认值是未标定工程设置。已发表仪器的精度指标
    不能直接等同为高程噪声的高斯标准差；sigma_z0_m 的数值仍需设备
    实测残差标定。Csencsics et al. (2022), DOI
    10.1109/I2MTC48687.2022.9806610 仅作为散斑误差背景，当前模型没有
    复现该文的光学传播仿真。入射角作用于噪声缩放，不改变采样XY；
    当前未模拟光斑足迹和几何自遮挡。只有 line-profiler 观测路径消费
    此配置，不能以采样间距代替精度验收。

    Attributes:
        lateral_res_m: 横向重采样分辨率 (米)，默认 0.001。
        step_m: 纵向剖面步距 (米)，默认 0.005。
        sensor_height_m: 传感器高度 (米)，默认 0.35（工程设定）。
        laser_incidence_deg: 光平面入射角 (度，自竖直)，默认 30.0
            （工程设定）。
        sigma_z0_m: 垂直入射基准高程噪声标准差 (米)，默认 0.00025。
        speckle_shape: 散斑亮度 Gamma 形状参数，默认 3.0。
        flyer_prob: 逐点飞点概率，默认 1e-3。
        flyer_sigma_mult: 飞点离群倍数（×σ_z），默认 8.0。
        brightness_dropout_ref: 低亮度丢点归一参考亮度，默认 0.05。
    """

    lateral_res_m: float = 0.001
    step_m: float = 0.005
    sensor_height_m: float = 0.35
    laser_incidence_deg: float = 30.0
    sigma_z0_m: float = 0.00025
    speckle_shape: float = 3.0
    flyer_prob: float = 1e-3
    flyer_sigma_mult: float = 8.0
    brightness_dropout_ref: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "lateral_res_m", "step_m", "sensor_height_m", "sigma_z0_m",
            "speckle_shape", "flyer_sigma_mult", "brightness_dropout_ref",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not math.isfinite(self.laser_incidence_deg) or not (
            0.0 < self.laser_incidence_deg < 90.0
        ):
            raise ValueError(
                f"laser_incidence_deg must be in (0, 90), "
                f"got {self.laser_incidence_deg}"
            )
        if not math.isfinite(self.flyer_prob) or not 0.0 <= self.flyer_prob <= 1.0:
            raise ValueError(
                f"flyer_prob must be in [0, 1], got {self.flyer_prob}"
            )


@dataclass
class LidarNoiseConfig:
    """LiDAR 噪声仿真参数。

    Attributes:
        distance_noise_std: 距离测量高斯噪声标准差 (米)。
        dropout_rate: 点丢失比例。
        angular_jitter_deg: 角度抖动标准差 (度)。
        mixed_pixel_prob: 混合像素效应概率。
        enable_edge_mixing: 是否启用边缘混合效应，默认 True。
    """

    distance_noise_std: float = 0.01
    dropout_rate: float = 0.03
    angular_jitter_deg: float = 0.01
    mixed_pixel_prob: float = 0.02
    enable_edge_mixing: bool = True

    def __post_init__(self) -> None:
        if self.distance_noise_std < 0:
            raise ValueError(f"distance_noise_std must be ≥ 0, got {self.distance_noise_std}")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0,1], got {self.dropout_rate}")


@dataclass
class CurbConfig:
    """缘石条带参数（行车道横向外侧的抬高顶面条带）。

    width_m = 0 表示关闭（默认）：不生成任何缘石点，行为与历史版本
    逐点一致。开启时每场景新增 2 × nx_c × ny 个背景标签点（见
    ``GeneratorConfig.curb_grid_point_count``），会稀释自然 prevalence
    分母——这是该选项的预期效应。

    Attributes:
        width_m: 单侧条带横向宽度 (米)。0 = 关闭。
        height_m: 顶面相对相邻路面设计边缘的抬高量 (米)，仅在
            width_m > 0 时生效。
    """

    width_m: float = 0.0
    height_m: float = 0.13

    def __post_init__(self) -> None:
        if not math.isfinite(self.width_m) or self.width_m < 0:
            raise ValueError(f"width_m must be finite and >= 0, got {self.width_m}")
        if not math.isfinite(self.height_m):
            raise ValueError(f"height_m must be finite, got {self.height_m}")
        if self.width_m > 0 and self.height_m <= 0:
            raise ValueError(
                f"height_m must be > 0 when width_m > 0, got {self.height_m}"
            )


@dataclass
class RadiometricConfig:
    """辐射强度模型的分布参数（对数正态反照率场 + Gamma 斑点）。

    分布参数（albedo_sigma_ln / albedo_corr_len_m / speckle_m）的
    默认值逐字等于 _compute_intensity 的历史硬编码（0.25 / 0.3 m /
    12），默认构造时这些项与历史版本逐点一致。强度归一的 AGC 与
    几何项默认值则**有意变更默认行为**（固定绝对标定、去 cosθ abs、
    反照率上限 0.95→0.98），旧口径均保留为显式回退（见各字段）。

    Attributes:
        albedo_sigma_ln: 对数正态反照率场的对数域标准差 σ_ln。
        albedo_corr_len_m: 反照率场的高斯低通空间相关长度 (米)。
        speckle_m: 乘性 Gamma 斑点的形状参数 M（均值 1、方差 1/M）。
        agc_mode: AGC（自动增益控制）口径。
            'fixed'（默认）——除以固定绝对标定值
            ``agc_reference_power``：沥青/水泥基底比 (~1:2.25) 在强度
            通道保留可分辨签名，远场动态范围不再被最近点压缩，且不
            随场景构成漂移。
            'scene-p99' —— 历史逐场景 p99 归一（材料签名被整除抵消，
            CLI --agc-mode scene-p99）。
        agc_reference_power: 固定 AGC 标定功率（**工程标定值**）：默认
            传感器位姿 (W/2, -1, 2 m)、参考反照率 ρ_ref=0.17（沥青
            名义均值）、平路面（shading=1）7×5 m @ 5 mm 网格上
            ρ·cosθ/R² 的 p99 = 0.025003 ≈ 0.025。仅
            agc_mode='fixed' 时使用。
        cos_abs: True —— 历史 cosθ = clip(|n·(−d̂)|, 0.03, 1)（背向面
            被当作朝向前亮度）；False（默认）——cosθ = n·(−d̂) 直接
            取值，背向面强度趋 0，下限 0.03 仅保留给数值稳定的前向
            掠射点（CLI --lambert-abs-legacy）。
        albedo_clip_hi: 反照率硬截断上限，默认 0.98（历史 0.95 会在
            最亮类别（露骨/唧泥 on 水泥）的对数正态亮尾堆积，
            CLI --albedo-clip-hi 0.95）。
    """

    albedo_sigma_ln: float = 0.25
    albedo_corr_len_m: float = 0.3
    speckle_m: float = 12.0
    agc_mode: Literal["fixed", "scene-p99"] = "fixed"
    agc_reference_power: float = 0.025
    cos_abs: bool = False
    albedo_clip_hi: float = 0.98

    def __post_init__(self) -> None:
        if not math.isfinite(self.albedo_sigma_ln) or self.albedo_sigma_ln < 0:
            raise ValueError(
                f"albedo_sigma_ln must be finite and >= 0, got {self.albedo_sigma_ln}"
            )
        if not math.isfinite(self.albedo_corr_len_m) or self.albedo_corr_len_m <= 0:
            raise ValueError(
                f"albedo_corr_len_m must be finite and > 0, got {self.albedo_corr_len_m}"
            )
        if not math.isfinite(self.speckle_m) or self.speckle_m < 1:
            raise ValueError(
                f"speckle_m must be finite and >= 1 (Gamma shape), got {self.speckle_m}"
            )
        if self.agc_mode not in ("fixed", "scene-p99"):
            raise ValueError(
                f"agc_mode must be 'fixed' or 'scene-p99', got {self.agc_mode!r}"
            )
        if not math.isfinite(self.agc_reference_power) or self.agc_reference_power <= 0:
            raise ValueError(
                "agc_reference_power must be finite and > 0, got "
                f"{self.agc_reference_power}"
            )
        if not math.isfinite(self.albedo_clip_hi) or not 0.0 < self.albedo_clip_hi <= 1.0:
            raise ValueError(
                f"albedo_clip_hi must be in (0, 1], got {self.albedo_clip_hi}"
            )


@dataclass
class LiDARScanConfig:
    """LiDAR 扫描几何配置 — 模拟真实扫描线密度分布 (P1-1).

    Attributes:
        enable: 是否启用扫描线模式。False 时使用规则网格。
        scan_pattern: 扫描模式 — 'rotating' (旋转式) 或 'solid_state' (固态闪光)。
        scan_lines: 扫描线数量 (旋转式) 或激光器通道数 (固态)。
        vertical_fov_deg: 垂直视场角 (度)，默认 40°。扫描线的地面
            位置按 y_i = y_s + h·tan(α_i) 由该视场角均分仰角得到，
            线距随距离二次增长。
        range_decay: 距离衰减系数，远处密度降低倍率。
        incidence_angle_drop: 入射角相关的丢点概率基数。
        sensor_pose: 传感器位姿 (x, y, z)，米制路面坐标。None 时由
            生成器取车载默认 (width/2, -1.0, 2.0)。**整条观测管线
            （扫描重采样、测距噪声投影、强度、丢点）共用这一个
            位姿** —— 这是测量误差模型可审计性的前提。
    """

    enable: bool = False
    scan_pattern: Literal["rotating", "solid_state"] = "rotating"
    scan_lines: int = 64
    vertical_fov_deg: float = 40.0
    range_decay: float = 0.3
    incidence_angle_drop: float = 0.05
    sensor_pose: tuple[float, float, float] | None = None
    line_sigma_m: float = 0.02

    def __post_init__(self) -> None:
        if self.sensor_pose is not None:
            if len(self.sensor_pose) != 3:
                raise ValueError(f"sensor_pose must be (x, y, z), got {self.sensor_pose}")
            if not all(math.isfinite(v) for v in self.sensor_pose):
                raise ValueError(f"sensor_pose must be finite, got {self.sensor_pose}")
            if self.sensor_pose[2] <= 0.0:
                raise ValueError(
                    f"sensor_pose height must be > 0 (above the road), got {self.sensor_pose[2]}"
                )


@dataclass
class DiseaseConfig:
    """病害生成参数聚合配置。

    ``disease_probs`` 是每种病害的**出现概率** (0-1)：默认
    ``sampling_mode='bernoulli'`` 下按病害逐个独立伯努利抽样（恢复
    文档『出现概率』语义），再按 ``max_diseases_per_scene`` 封顶；
    因此期望每场景病害数 ≈ Σp（默认配置 ≈ 0.8 种），**允许出现 0
    病害的背景场景**。0 值表示该病害类型从不生成。
    ``sampling_mode='legacy'`` 恢复历史口径：概率归一化为权重、无
    放回抽恰好 min(max_diseases, n_available) 种（每场景病害数恒
    定、concrete_damage 等被归一化为权重而非概率）。

    Attributes:
        disease_probs: 各病害类型出现概率字典（0 = 不生成）。
        severity_ratio: 全局轻度占比，默认 0.6。裂缝类可被
            ``crack.severity_ratio`` 按类型覆盖。
        max_diseases_per_scene: 每场景最大病害数，默认 3。0 表示强制
            无病害背景场景（两种 sampling_mode 下选择均恒为空）。
        sampling_mode: 病害选择口径，'bernoulli'（默认）| 'legacy'。
        use_stratified: **已弃用**。False 仍强制伯努利（兼容旧
            --no-stratified 语义）；True 时选择语义由 sampling_mode
            决定。保留仅为向后兼容。
    """

    disease_probs: dict[str, float] = field(default_factory=lambda: {
        # 沥青路面病害
        "crack": 0.25,           # 裂缝（含纵向、横向、龟裂、块状）
        "pothole": 0.10,          # 坑槽
        "raveling": 0.08,         # 松散
        "depression": 0.10,       # 沉陷
        "rutting": 0.12,          # 车辙
        "corrugation": 0.05,      # 波浪拥包
        "bleeding": 0.05,         # 泛油
        "patching": 0.05,         # 修补
        # 水泥路面病害
        "concrete_damage": 0.15,  # 水泥路面损坏（含多种子类型）
    })
    severity_ratio: float = 0.6
    max_diseases_per_scene: int = 3
    use_stratified: bool = True
    sampling_mode: Literal["bernoulli", "legacy"] = "bernoulli"

    def __post_init__(self) -> None:
        if self.sampling_mode not in ("bernoulli", "legacy"):
            raise ValueError(
                f"sampling_mode must be 'bernoulli' or 'legacy', got {self.sampling_mode!r}"
            )
        if not 0.0 <= self.severity_ratio <= 1.0:
            raise ValueError(f"severity_ratio must be in [0,1], got {self.severity_ratio}")
        if self.max_diseases_per_scene < 0:
            raise ValueError(f"max_diseases_per_scene must be ≥ 0, got {self.max_diseases_per_scene}")
        for k, v in self.disease_probs.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"disease_probs['{k}'] = {v} not in [0,1]")


@dataclass
class StripSegmentationConfig:
    """分段条带（strip segmentation）参数。

    当 :meth:`GeneratorConfig.estimate_surface_generation` 判定整条带
    超出 per-worker 预算（``max_surface_points`` 或
    ``max_surface_memory_mib`` 换算的点数）时进入分段体制：宏观长波
    场整条带全局一次合成于粗网格、短波纹理逐段 FFT、段间
    root-raised-cosine 混合、het 调制场全局 1-D 共享。``None`` 字段
    使用确定性自动推导，推导结果（段数 / 段长 / 重叠）全部写入场景
    元数据 ``surface_grid.segmentation``。

    Attributes:
        segment_length_m: 单段拥有行长度 (米)。None = 自动推导（默认），
            自动值 = ``(rows_cap - 2 * m_rows) * grid_res``。仅在分段被
            触发（条带超 per-worker 预算）时生效；条带在预算内时保持
            单块路径。
        overlap_m: 段间重叠宽度 (米)。None = 自动推导
            ``max(2 / ISO8608_BAND_MAX_CYCLES_PER_M, 16 * grid_res, 0.2)``
            （默认 ≥ 2 倍纹理带最长波长 0.354 m，并覆盖坑槽 0.45 m 的
            影响半径归属规则）；显式值不得低于 0.45 m。
        macro_grid_factor: 宏观粗网格相对 ISO 带上限频率的过采样倍数
            （粗网格间距 = 1 / (macro_grid_factor * f_iso_max)）。
        feature_halo_m: 最终特征分块重算的邻域 halo 宽度 (米)。
            None = 自动 ``max(0.05, 16 * grid_res)``。
    """

    segment_length_m: float | None = None
    overlap_m: float | None = None
    macro_grid_factor: float = 4.0
    feature_halo_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("segment_length_m", "overlap_m", "feature_halo_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(
                    f"{name} must be finite and > 0 when provided, got {value}"
                )
        if (
            not math.isfinite(self.macro_grid_factor)
            or self.macro_grid_factor < 1.0
        ):
            raise ValueError(
                "macro_grid_factor must be finite and >= 1, "
                f"got {self.macro_grid_factor}"
            )


@dataclass(frozen=True)
class SurfaceGenerationEstimate:
    """Preflight estimate for the full-resolution synthesis surface.

    v2 additions model the segmented-strip regime.  When ``segmented`` is
    False the new fields degrade to whole-strip semantics (a single segment
    spanning the full length with zero overlap), so every consumer can read
    them unconditionally.
    """

    grid_shape: tuple[int, int]
    point_count: int
    estimated_peak_memory_mib: float
    """Per-worker peak-RAM estimate: whole-strip synthesis peak on the
    single-block path, ``worker_peak_mib`` on segmented plans (the naive
    whole-strip figure is no longer any worker's true peak there)."""
    bytes_per_point_assumption: int = SURFACE_PEAK_BYTES_PER_POINT
    # ---- v2: 分段体制估算字段（单块时退化为整条带语义） ----
    segmented: bool = False
    """Whether the strip exceeded the per-worker budget and is planned as
    multiple blended segments."""
    segment_count: int = 1
    """Number of owned segments (1 for the single-block path)."""
    segment_length_m: float = 0.0
    """Owned length of one segment in metres (full strip length when not
    segmented)."""
    overlap_m: float = 0.0
    """Segment blend overlap width in metres (0 for the single-block path)."""
    segment_point_count_ext: int = 0
    """Allocated points of one extended segment: owned rows + both overlap
    halos (plus in-segment curb columns), i.e. the per-segment allocation the
    budget must cover."""
    segment_peak_mib: float = 0.0
    """Synthesis peak estimate for one extended segment at
    ``SURFACE_PEAK_BYTES_PER_POINT``."""
    assembly_mib: float = 0.0
    """Preallocated float32 output assembly for the whole strip at
    ``ASSEMBLY_BYTES_PER_POINT`` (written per-segment, never vstacked)."""
    feature_sweep_mib: float = 0.0
    """Chunked final-feature recomputation budget for one extended segment at
    ``FEATURE_SWEEP_BYTES_PER_POINT``."""
    worker_peak_mib: float = 0.0
    """Per-worker peak = max(segment_peak_mib, assembly_mib +
    feature_sweep_mib); equals ``estimated_peak_memory_mib`` on the
    single-block path."""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "grid_shape": list(self.grid_shape),
            "point_count": int(self.point_count),
            "estimated_peak_memory_mib": float(self.estimated_peak_memory_mib),
            "bytes_per_point_assumption": int(self.bytes_per_point_assumption),
        }


@dataclass
class GeneratorConfig:
    """合成数据集生成器顶层配置。

    组合所有子配置，作为 `SyntheticRoadDataset` 的唯一入口参数。

    Attributes:
        road: 路面轮廓配置。
        micro_texture: 微观纹理配置。
        disease: 病害概率配置。
        crack: 裂缝配置。
        pothole: 坑槽配置。
        rutting: 车辙配置。
        corrugation: 波浪拥包配置。
        depression: 沉陷配置。
        raveling: 松散配置。
        bleeding: 泛油配置。
        concrete_damage: 水泥损坏配置。
        lidar_noise: LiDAR 噪声配置。
        lidar_scan: LiDAR 扫描几何配置 (P1-1)。
        curb: 缘石条带配置（默认关闭）。
        radiometric: 辐射强度模型参数配置（默认值等于历史硬编码）。
        seed: 全局随机种子，默认 None (非确定性)。
        num_points: 传感器输出目标点数 (兼容旧名称)，默认 65536。
        target_density: 传感器输出目标密度 (点/㎡，兼容旧名称)。
        model_target_points: 下游模型每场景目标点数。生成器只记录该契约，
            不执行模型采样；None 表示由训练配置决定。
        target_label_min_output_points: 受控场景下目标标签的最小存活点数。
            默认 1，仅防止标签完全消失，不固定病害点比例。
        point_count_tolerance: 点数波动容忍度 (±比例)，默认 0.2 (P1-2)。
        max_surface_points: 传感器采样前允许分配的最大表面网格点数。
        max_surface_memory_mib: 单个生成 worker 的估算峰值内存上限。
        normalize: 是否坐标归一化到单位球，默认 True。
        observation_model: 观测模型体制开关（见字段文档串；配置默认
            "on"，CLI generate_synthetic.py 默认 "off"）。第三档
            "line-profiler" 为线激光剖面仪观测模型（传感器模块
            roadmc/data/synthetic/line_profiler.py（已交付），
            模块未就绪时报"未安装"类清晰错误，不影响 on/off）。
        strip: 分段条带合成参数（超 per-worker 预算时触发，默认自动）。
        am_envelope: AM 慢变包络配置（默认关闭）。
        base_condition: 基层条件与病害因果链配置（默认关闭）。
        line_profiler: 线激光剖面仪传感器配置（仅在 observation_model
            ="line-profiler" 时使用；传感器模块未就绪前不装配）。
    """

    road: RoadSurfaceConfig = field(default_factory=RoadSurfaceConfig)
    micro_texture: MicroTextureConfig = field(default_factory=MicroTextureConfig)
    disease: DiseaseConfig = field(default_factory=DiseaseConfig)
    crack: CrackConfig = field(default_factory=CrackConfig)
    pothole: PotholeConfig = field(default_factory=PotholeConfig)
    rutting: RuttingConfig = field(default_factory=RuttingConfig)
    corrugation: CorrugationConfig = field(default_factory=CorrugationConfig)
    depression: DepressionConfig = field(default_factory=DepressionConfig)
    raveling: RavelingConfig = field(default_factory=RavelingConfig)
    bleeding: BleedingConfig = field(default_factory=BleedingConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    concrete_damage: ConcreteDamageConfig = field(default_factory=ConcreteDamageConfig)
    lidar_noise: LidarNoiseConfig = field(default_factory=LidarNoiseConfig)
    lidar_scan: LiDARScanConfig = field(default_factory=LiDARScanConfig)
    curb: CurbConfig = field(default_factory=CurbConfig)
    radiometric: RadiometricConfig = field(default_factory=RadiometricConfig)
    seed: int | None = None
    num_points: int = 65536
    target_density: float | None = None  # P1-2: 点/㎡
    model_target_points: int | None = None
    target_label_min_output_points: int = 1
    point_count_tolerance: float = 0.20  # P1-2: ±20%
    max_surface_points: int = DEFAULT_MAX_SURFACE_POINTS
    max_surface_memory_mib: float = DEFAULT_MAX_SURFACE_MEMORY_MIB
    normalize: bool = True
    retain_all_surface_points: bool = False
    """When True, skip sensor output sampling (step 10) and keep every point
    that survives the LiDAR noise / occlusion pipeline.  The resulting file
    contains ~1.4M points at 5 mm grid spacing — useful for density-upper-bound
    references in ablation studies.  ``num_points`` is ignored when this is set."""

    observation_model: Literal["on", "off", "line-profiler"] = "on"
    """观测模型体制开关（交互规则见下）。

    - ``"on"``（**配置默认**）：完整传感器观测管线（扫描重采样 → 曲率 →
      遮挡 → 噪声 → 最近邻传递 → 输出采样），行为与历史版本逐位一致。
    - ``"off"``：观测体制关闭——直接输出全部表面点（retain-all 语义，
      ``noisy_points := points``），跳过扫描线重采样、曲率、几何遮挡、
      LiDAR 噪声、最近邻标签传递与输出采样；``num_points`` /
      ``target_density`` 被忽略但仍以 ``ignored_under_observation_off``
      记录在元数据；路面谱合成、全部病害、松散及 NaN 过滤（表面几何
      属性，非观测效应）、强度计算、归一化、特征全部保留。off 且
      ``lidar_scan.enable=True`` 时发 ``warnings.warn`` 并跳过扫描
      （元数据记录 override），不报硬错。
    - ``"line-profiler"``：线激光剖面仪观测模型——跳过旋转 LiDAR
      扫描重采样（步骤 2.5）与 ``simulate_lidar_noise`` 球坐标噪声
      （步骤 8，扫描式 LiDAR 模型），在病害后的步骤 8 位置运行
      ``simulate_line_profiler``（模块
      ``roadmc.data.synthetic.line_profiler``，固定签名
      提供；模块未就绪时报"未安装"类清晰错误）。暂仅支持单块路径，
      与分段体制组合显式 raise NotImplementedError。

    **配置默认值取 "on"，CLI（generate_synthetic.py）默认才是 "off"**：
    ``GeneratorConfig()`` 的库内默认行为被 dataloader 冒烟配置与生成器
    自检（``points.shape[0] == num_points`` 断言）依赖，配置默认翻成
    off 会让这两条路径无声改变。"""

    strip: StripSegmentationConfig = field(default_factory=StripSegmentationConfig)
    """分段条带合成参数：条带总点数超过 per-worker 预算时按此配置分段，
    字段为 None 时使用确定性自动推导（推导结果写入场景元数据）。"""

    am_envelope: AmEnvelopeConfig = field(default_factory=AmEnvelopeConfig)
    """AM 慢变包络配置：默认关闭，默认输出逐位不变。"""

    surface_layers: bool = False
    """是否启用可选的 design/roughness/texture 三层表面；默认关闭以保留旧行为。"""

    base_condition: BaseConditionConfig = field(default_factory=BaseConditionConfig)
    """基层条件与病害因果链配置：默认关闭（旧行为逐位不变）。"""

    line_profiler: LineProfilerConfig = field(default_factory=LineProfilerConfig)
    """线激光剖面仪传感器配置（仅在 observation_model='line-profiler'
    时消费；传感器模块未就绪前保持占位）。"""

    def __post_init__(self) -> None:
        if self.observation_model not in ("on", "off", "line-profiler"):
            raise ValueError(
                "observation_model must be 'on', 'off' or 'line-profiler', "
                f"got {self.observation_model!r}"
            )
        if not self.retain_all_surface_points and self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")
        if self.target_density is not None and self.target_density < 0:
            raise ValueError(
                f"target_density must be >= 0 when provided, got {self.target_density}"
            )
        if self.model_target_points is not None and self.model_target_points < 1:
            raise ValueError(
                "model_target_points must be >= 1 when provided, "
                f"got {self.model_target_points}"
            )
        if self.target_label_min_output_points < 0:
            raise ValueError(
                "target_label_min_output_points must be >= 0, "
                f"got {self.target_label_min_output_points}"
            )
        if not 0.0 <= self.point_count_tolerance < 1.0:
            raise ValueError(
                "point_count_tolerance must be in [0, 1), "
                f"got {self.point_count_tolerance}"
            )
        if self.max_surface_points < 1:
            raise ValueError(
                f"max_surface_points must be >= 1, got {self.max_surface_points}"
            )
        if (
            not math.isfinite(self.max_surface_memory_mib)
            or self.max_surface_memory_mib <= 0
        ):
            raise ValueError(
                "max_surface_memory_mib must be finite and > 0, "
                f"got {self.max_surface_memory_mib}"
            )
        if self.target_label_min_output_points > self.sensor_output_target_points:
            raise ValueError(
                "target_label_min_output_points cannot exceed the requested sensor "
                f"output point count ({self.sensor_output_target_points}), got "
                f"{self.target_label_min_output_points}"
            )
        if self.curb.width_m > 0 and 2 * self.curb.width_m >= self.road.width:
            raise ValueError(
                f"curb strips of {self.curb.width_m} m per side leave no carriageway: "
                f"2 * curb.width_m must be < road.width ({self.road.width} m)"
            )

        self.validate_surface_budget()

    @property
    def sensor_output_mode(self) -> str:
        """Configured sensor-output sampling mode."""

        if self.target_density is not None and self.target_density > 0:
            return "density_voxel"
        return "fixed_count"

    @property
    def sensor_output_target_points(self) -> int:
        """Requested stored sensor points before any model-side sampling."""

        if self.sensor_output_mode == "density_voxel":
            return max(1, int(self.road.surface_area_m2 * float(self.target_density)))
        return int(self.num_points)

    @property
    def sensor_output_density_points_per_m2(self) -> float | None:
        """Requested sensor density, or ``None`` for fixed-count output."""

        if self.sensor_output_mode == "density_voxel":
            return float(self.target_density)
        return None

    @property
    def curb_grid_point_count(self) -> int:
        """Analytic point count contributed by the optional curb strips.

        Single shared formula for every curb point-count consumer (per the
        curb-strip design: ``2 * nx_c * ny`` with
        ``nx_c = max(1, ceil(curb_width_m / grid_res))``, same rounding rule
        as ``RoadSurfaceConfig.surface_grid_shape`` and the same y-grid as
        the road).  ``estimate_surface_generation`` derives its allocation
        and peak-memory budget from this property; the ``generate_scene``
        point-count guard and the curvature fast-path decision must use the
        same formula once the curb strips are merged into the surface cloud.
        Returns 0 when the curb is disabled.
        """

        if self.curb.width_m <= 0:
            return 0
        nx_c = max(1, int(math.ceil(self.curb.width_m / self.road.grid_res)))
        _, ny = self.road.surface_grid_shape
        return 2 * nx_c * ny

    def estimate_surface_generation(self) -> SurfaceGenerationEstimate:
        """Estimate point allocation and conservative peak RAM per worker.

        The allocation covers the road synthesis grid plus the optional curb
        strips (0 points when disabled), so per-worker and parallel memory
        budgets keep covering the true peak once the curb points are merged
        into the surface cloud.

        When the strip exceeds the per-worker budget (``max_surface_points``
        or the row count implied by ``max_surface_memory_mib``), the estimate
        switches to the segmented-strip plan: deterministic auto derivation
        of overlap width, owned rows per segment and segment count, with the
        per-worker peak taken as ``max(segment peak, assembly + feature
        sweep)``.  Otherwise the single-block plan is returned unchanged and
        the new fields degrade to whole-strip semantics.
        """

        point_count = self.road.surface_grid_point_count + self.curb_grid_point_count
        memory_mib = point_count * SURFACE_PEAK_BYTES_PER_POINT / (1024.0 ** 2)
        grid_shape = self.road.surface_grid_shape
        assembly_mib = point_count * ASSEMBLY_BYTES_PER_POINT / (1024.0 ** 2)
        feature_sweep_mib = (
            point_count * FEATURE_SWEEP_BYTES_PER_POINT / (1024.0 ** 2)
        )

        if not self._strip_needs_segmentation():
            return SurfaceGenerationEstimate(
                grid_shape=grid_shape,
                point_count=point_count,
                estimated_peak_memory_mib=memory_mib,
                segmented=False,
                segment_count=1,
                segment_length_m=float(self.road.length),
                overlap_m=0.0,
                segment_point_count_ext=point_count,
                segment_peak_mib=memory_mib,
                assembly_mib=assembly_mib,
                feature_sweep_mib=feature_sweep_mib,
                worker_peak_mib=memory_mib,
            )
        return self._plan_strip_segmentation()

    def _strip_rows_budget(self) -> int:
        """Per-worker point budget expressed as a number of y-axis rows.

        The budget (whichever of ``max_surface_points`` and the
        ``max_surface_memory_mib`` row budget is tighter) is divided by the
        per-row point footprint: road columns ``nx`` plus, when the curb
        strips are enabled, both curb column groups.  The design formula
        divides by ``nx`` only; with curb enabled that combination can never
        satisfy the design's own ``segment_point_count_ext <=
        max_surface_points`` check (ext grows by the curb column fraction
        while the budget does not), so the division uses the full per-row
        footprint — conservative and feasible for every configuration.
        """

        nx = self.road.surface_grid_shape[0]
        if self.curb.width_m > 0:
            nx_c = max(1, int(math.ceil(self.curb.width_m / self.road.grid_res)))
            nx += 2 * nx_c
        budget_points = min(
            self.max_surface_points,
            math.floor(
                self.max_surface_memory_mib * (1024.0 ** 2)
                / SURFACE_PEAK_BYTES_PER_POINT
            ),
        )
        return budget_points // nx

    def _strip_needs_segmentation(self) -> bool:
        """True when the strip exceeds the per-worker row budget.

        The only trigger is the budget (per the design); an explicit
        ``strip.segment_length_m`` selects the segment length once
        segmentation is triggered and never forces segmentation by itself.
        """

        _, ny = self.road.surface_grid_shape
        return ny > self._strip_rows_budget()

    def strip_overlap_m(self) -> float:
        """Blend overlap width: explicit override or the deterministic auto
        value ``max(2 / ISO8608_BAND_MAX_CYCLES_PER_M, 16 * grid_res, 0.2)``.

        The overlap must never fall below the 0.45 m pothole influence radius
        (the design's cross-segment attribution rule); explicit overrides
        below that bound are rejected.
        """

        if self.strip.overlap_m is not None:
            overlap_m = float(self.strip.overlap_m)
            if overlap_m < POTHOLE_INFLUENCE_RADIUS_M:
                raise ValueError(
                    f"strip.overlap_m must be >= {POTHOLE_INFLUENCE_RADIUS_M} m "
                    f"(pothole influence radius), got {overlap_m}"
                )
            return overlap_m
        return max(
            2.0 / ISO8608_BAND_MAX_CYCLES_PER_M,
            16.0 * self.road.grid_res,
            0.2,
        )

    def strip_halo_rows(self) -> int:
        """Overlap half-width in grid rows (m_rows = ceil(overlap/grid_res))."""

        return max(1, int(math.ceil(self.strip_overlap_m() / self.road.grid_res)))

    def strip_macro_grid_spacing_m(self) -> float:
        """Coarse-grid spacing of the global macro (ISO 8608 band) field:
        ``1 / (macro_grid_factor * f_iso_max)`` with
        ``f_iso_max = min(2.83, Nyquist)``."""

        f_iso_max = min(ISO8608_BAND_MAX_CYCLES_PER_M, 0.5 / self.road.grid_res)
        return 1.0 / (self.strip.macro_grid_factor * f_iso_max)

    def strip_owned_rows_per_segment(self) -> int:
        """Owned grid rows of one segment (auto or explicit), validated for
        feasibility.  The last segment of a strip may be shorter."""

        grid_res = self.road.grid_res
        m_rows = self.strip_halo_rows()
        if self.strip.segment_length_m is not None:
            rows_seg = int(math.ceil(self.strip.segment_length_m / grid_res))
        else:
            rows_seg = self._strip_rows_budget() - 2 * m_rows
        feasibility_floor = max(4 * m_rows, 200)
        if rows_seg < feasibility_floor:
            raise ValueError(
                f"Strip segmentation is infeasible: owned segment rows {rows_seg} "
                f"< max(4 * overlap_rows={4 * m_rows}, 200)={feasibility_floor} at "
                f"{grid_res * 1000.0:.3f} mm spacing with "
                f"{self.strip_overlap_m():.3f} m overlap ({m_rows} halo rows per "
                "side). Raise --max-surface-points / --max-surface-memory-mib, use "
                "a coarser spacing, or shorten the strip; do not force tiny "
                "blend-dominated segments."
            )
        return rows_seg

    def _plan_strip_segmentation(self) -> SurfaceGenerationEstimate:
        """Deterministic segmented-strip plan (all values recorded in scene
        metadata).

        Auto derivation: ``m_rows = ceil(overlap / grid_res)`` halo rows per
        side; ``rows_seg = rows_budget - 2 * m_rows`` owned rows; feasibility
        requires ``rows_seg >= max(4 * m_rows, 200)`` — otherwise the blend
        band overhead would dominate the segment and the texture spectrum
        would be blend-dominated, so this raises instead of silently cutting
        sliver segments.
        """

        grid_res = self.road.grid_res
        nx, ny = self.road.surface_grid_shape
        point_count = self.road.surface_grid_point_count + self.curb_grid_point_count
        overlap_m = self.strip_overlap_m()
        rows_seg = self.strip_owned_rows_per_segment()
        m_rows = self.strip_halo_rows()
        segment_length_m = rows_seg * grid_res

        segment_count = max(1, int(math.ceil(ny / rows_seg)))
        ext_rows = rows_seg + 2 * m_rows
        ext_cols = nx
        if self.curb.width_m > 0:
            nx_c = max(1, int(math.ceil(self.curb.width_m / grid_res)))
            ext_cols += 2 * nx_c
        segment_point_count_ext = ext_rows * ext_cols
        segment_peak_mib = (
            segment_point_count_ext * SURFACE_PEAK_BYTES_PER_POINT / (1024.0 ** 2)
        )
        assembly_mib = point_count * ASSEMBLY_BYTES_PER_POINT / (1024.0 ** 2)
        feature_sweep_mib = (
            segment_point_count_ext * FEATURE_SWEEP_BYTES_PER_POINT / (1024.0 ** 2)
        )
        worker_peak_mib = max(segment_peak_mib, assembly_mib + feature_sweep_mib)
        # estimated_peak_memory_mib 承载 v1 的 per-worker 峰值语义：分段
        # 计划下等于 worker_peak_mib（整条带 naive 值 point_count×320 已
        # 不再是任何 worker 的真实峰值，报告它会让审计误判预算被突破）。
        return SurfaceGenerationEstimate(
            grid_shape=(nx, ny),
            point_count=point_count,
            estimated_peak_memory_mib=worker_peak_mib,
            segmented=True,
            segment_count=segment_count,
            segment_length_m=segment_length_m,
            overlap_m=overlap_m,
            segment_point_count_ext=segment_point_count_ext,
            segment_peak_mib=segment_peak_mib,
            assembly_mib=assembly_mib,
            feature_sweep_mib=feature_sweep_mib,
            worker_peak_mib=worker_peak_mib,
        )

    def validate_surface_budget(self) -> SurfaceGenerationEstimate:
        """Reject a surface that exceeds configured per-worker safety limits.

        Segmented plans are validated on the per-segment extended allocation
        (``segment_point_count_ext``) and the per-worker peak
        (``worker_peak_mib``); single-block plans keep the original
        whole-strip checks verbatim.
        """

        estimate = self.estimate_surface_generation()
        if estimate.segmented:
            if estimate.segment_point_count_ext > self.max_surface_points:
                raise ValueError(
                    "Segmented strip plan would allocate "
                    f"{estimate.segment_point_count_ext:,} points per extended "
                    f"segment at {self.road.surface_grid_spacing_m * 1000.0:.3f} mm, "
                    f"exceeding max_surface_points={self.max_surface_points:,}. "
                    "Increase the spacing, crop the road, or explicitly raise the "
                    "limit after checking RAM."
                )
            if estimate.worker_peak_mib > self.max_surface_memory_mib:
                raise ValueError(
                    "Segmented strip synthesis is estimated to peak at "
                    f"{estimate.worker_peak_mib:.1f} MiB per worker, exceeding "
                    f"max_surface_memory_mib={self.max_surface_memory_mib:.1f}. "
                    "Reduce the surface size/resolution or explicitly raise the "
                    "memory budget."
                )
            return estimate
        if estimate.point_count > self.max_surface_points:
            raise ValueError(
                "Surface grid would allocate "
                f"{estimate.point_count:,} points at "
                f"{self.road.surface_grid_spacing_m * 1000.0:.3f} mm, exceeding "
                f"max_surface_points={self.max_surface_points:,}. This check applies "
                "before sensor/output and model downsampling. Increase the spacing, "
                "crop the road, or explicitly raise the limit after checking RAM."
            )
        if estimate.estimated_peak_memory_mib > self.max_surface_memory_mib:
            raise ValueError(
                "Surface synthesis is estimated to peak at "
                f"{estimate.estimated_peak_memory_mib:.1f} MiB per worker, exceeding "
                f"max_surface_memory_mib={self.max_surface_memory_mib:.1f}. Reduce the "
                "surface size/resolution or explicitly raise the memory budget."
            )
        return estimate

    def validate_parallel_budget(
        self,
        workers: int,
        max_parallel_memory_mib: float,
    ) -> float:
        """Validate aggregate high-resolution memory for parallel generation.

        The per-worker figure multiplied by ``workers`` is
        ``worker_peak_mib`` (identical to the whole-strip estimate on the
        single-block path, the segmented per-segment peak otherwise).
        """

        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if not math.isfinite(max_parallel_memory_mib) or max_parallel_memory_mib <= 0:
            raise ValueError(
                "max_parallel_memory_mib must be finite and > 0, "
                f"got {max_parallel_memory_mib}"
            )
        total_mib = self.validate_surface_budget().worker_peak_mib * workers
        if total_mib > max_parallel_memory_mib:
            raise ValueError(
                f"{workers} workers may require about {total_mib:.1f} MiB for surface "
                f"synthesis, exceeding the parallel budget of {max_parallel_memory_mib:.1f} "
                "MiB. Reduce --workers or explicitly raise --max-parallel-memory-mib."
            )
        return total_mib

    def resolution_metadata(self, actual_output_points: int | None = None) -> dict[str, Any]:
        """Describe distinct surface, sensor-output, and model resolutions.

        The returned structure is JSON-safe and intentionally states that
        model sampling is downstream.  ``actual_output_points`` may be supplied
        after a scene has passed through scan simulation and output sampling.
        """

        estimate = self.estimate_surface_generation()
        requested_points = self.sensor_output_target_points
        output_points = (
            None if actual_output_points is None else int(actual_output_points)
        )
        density = (
            None
            if output_points is None
            else output_points / self.road.surface_area_m2
        )
        nominal_spacing = (
            None
            if output_points is None or output_points == 0
            else math.sqrt(self.road.surface_area_m2 / output_points)
        )
        metadata = {
            "contract_version": RESOLUTION_CONTRACT_VERSION,
            # 分辨率契约的显式布尔记录（设计稿 sensor_contract /
            # spectrum_design 要求落 resolution_contract 的量）。键名
            # 取 resolution_contract_booleans 而非 resolution_contract：
            # 场景 npz 已有同名标量键 resolution_contract（取
            # contract_version 字符串，历史口径），同名会在下游造成
            # 一名两义（前一轮复核实测）。语义同设计稿：
            # crack_resolvable —— 表面网格能否分辨轻裂（中位缝宽
            # 1.8mm 需要点间距 ≪ 缝宽；grid_res>2mm 不可分辨，警告
            # 依据 DOI 10.3390/s26082338；line-profiler 档位相关）；
            # het_grid_adequate —— het 调制相关长度在当前网格上的
            # 采样充分性（corr_len/grid_res ≥ 5 视为充分；1cm 相关
            # 长度在 5mm 网格仅 2 格 → False）。均为配置层判定，
            # 生成器不回填。
            "resolution_contract_booleans": {
                "crack_resolvable": bool(self.road.grid_res <= 0.002),
                "het_grid_adequate": bool(
                    self.micro_texture.het_sigma_ln <= 0.0
                    or self.micro_texture.het_corr_len_m
                    / self.road.grid_res
                    >= 5.0
                ),
            },
            "surface_grid": {
                "spacing_m": self.road.surface_grid_spacing_m,
                "width_m": float(self.road.width),
                "length_m": float(self.road.length),
                "area_m2": self.road.surface_area_m2,
                **estimate.as_dict(),
                "max_surface_points": int(self.max_surface_points),
                "max_surface_memory_mib": float(self.max_surface_memory_mib),
                # 谱带审计：等级标签指谱系数 Gd(n0)。补丁长度 L 截断了
                # λ > L 的低频侧（IRI 加权带基本缺失）。
                "roughness_band_cycles_per_m": [
                    1.0 / float(max(self.road.width, self.road.length)),
                    min(
                        ISO8608_BAND_MAX_CYCLES_PER_M,
                        0.5 / self.road.grid_res,
                    ),
                ],
                "iso8608_full_band_represented": False,
                "texture_band_cycles_per_m": [
                    min(ISO8608_BAND_MAX_CYCLES_PER_M, 0.5 / self.road.grid_res),
                    0.5 / self.road.grid_res,
                ],
                "texture_rms_m": float(self.micro_texture.amplitude),
                "texture_hurst": float(self.micro_texture.hurst),
                "texture_het_sigma_ln": float(self.micro_texture.het_sigma_ln),
                "texture_het_corr_len_m": float(self.micro_texture.het_corr_len_m),
                "macro_spectrum_mode": str(self.road.macro_spectrum_mode),
                # 粗糙度混合开启时逐场景实际抽到的等级在场景字段
                # roughness_class 里；这里记录配置层面的抽样规格。
                "roughness_class": (
                    "per_scene_draw (see per-scene roughness_class)"
                    if self.road.roughness_mix is not None
                    else str(self.road.roughness_class)
                ),
                "roughness_mix": (
                    None
                    if self.road.roughness_mix is None
                    else ",".join(
                        f"{key}:{self.road.roughness_mix[key]:g}"
                        for key in sorted(self.road.roughness_mix)
                    )
                ),
                "curb_width_m": float(self.curb.width_m),
                "curb_height_m": float(self.curb.height_m),
                "crossfall": float(self.road.crossfall),
                "crossfall_shape": str(self.road.crossfall_shape),
                "longitudinal_grade": float(self.road.longitudinal_grade),
                "het_modulation_grid_advisory": (
                    "het_sigma_ln > 0 with corr length "
                    f"{self.micro_texture.het_corr_len_m:g} m spans only "
                    f"{max(1, int(round(self.micro_texture.het_corr_len_m / self.road.grid_res)))} "
                    "grid cells; grid_res <= 0.002 m recommended when the het "
                    "modulation is enabled"
                    if self.micro_texture.het_sigma_ln > 0.0
                    else None
                ),
                # two-segment 谱形审计（配置层规格；逐场景实际抽取的
                # mpd_drawn_m / texture_rms_effective_m 由生成器写入）。
                "texture_spectrum": {
                    "mode": str(self.road.macro_spectrum_mode),
                    "h_macro": float(self.micro_texture.h_macro),
                    "h_micro": float(self.micro_texture.h_micro),
                    "break_wavelength_m": float(self.micro_texture.break_wavelength_m),
                    "break_cycles_per_m": float(
                        1.0 / self.micro_texture.break_wavelength_m
                    ),
                    "nyquist_cycles_per_m": float(0.5 / self.road.grid_res),
                    "micro_band_empty": bool(
                        1.0 / self.micro_texture.break_wavelength_m
                        >= 0.5 / self.road.grid_res
                    ),
                    # 设计稿点名的键：mpd_target_m 报**有效**目标区间
                    # （用户覆盖或内置默认）。override 状态经
                    # mpd_target_m_overridden 布尔披露——两键合读可
                    # 区分覆盖与默认（前一轮曾用 mpd_target_m_effective
                    # 命名，按复核意见恢复设计稿键名）。
                    "mpd_target_m": (
                        {
                            k: [float(v[0]), float(v[1])]
                            for k, v in (
                                self.micro_texture.mpd_target_m
                                if self.micro_texture.mpd_target_m is not None
                                else {
                                    "asphalt": (0.0008, 0.0013),
                                    "concrete": (0.0002, 0.0008),
                                }
                            ).items()
                        }
                    ),
                    "mpd_target_m_overridden": (
                        self.micro_texture.mpd_target_m is not None
                    ),
                    "mpd_rms_ratio": float(self.micro_texture.mpd_rms_ratio),
                    "mpd_drawn_m": None,
                    "texture_rms_effective_m": None,
                    "provenance": (
                        "two-scale power law (VT 2015 Eqs. 3-55/3-56, "
                        "vtthesis2015.txt:3230-3261); roll-off wavelength "
                        "Table 4-2 (vtthesis2015.txt:3568-3577); imported_"
                        "not_self_calibrated; H ranges 0.31-0.37 / 0.68-0.82 "
                        "need manual verification against the VT original pages"
                    ),
                },
                # AM 慢变包络审计（配置层规格；实测 skew/kurt 由生成器
                # 在包络施加后回填）。
                "am_envelope": {
                    "enabled": bool(self.am_envelope.enable),
                    "wcut_cycles_per_m": float(self.am_envelope.wcut_cycles_per_m),
                    "gamma_shape": float(self.am_envelope.gamma_shape),
                    "target_skew": float(self.am_envelope.target_skew),
                    "target_kurt": float(self.am_envelope.target_kurt),
                    "measured_skew": None,
                    "measured_kurt": None,
                    "coefficients_provenance": "imported_not_self_calibrated",
                },
            },
        }
        # 基层条件 / 病害因果链审计（配置层规格；逐场景抽取值由生成器
        # 写入顶层 base_condition 的扩展字段）。
        metadata["base_condition"] = {
            "enabled": bool(self.base_condition.enable),
            "snp_range": [float(v) for v in self.base_condition.snp_range],
            "age_years_range": [float(v) for v in self.base_condition.age_years_range],
            "ye4_msa_range": [float(v) for v in self.base_condition.ye4_msa_range],
            "tpm_path": self.base_condition.tpm_path,
            "coefficients_provenance": "imported_not_self_calibrated",
            "pothole_cooccur_model": (
                "base*clip(age/ICA-1,0,1)+0.5*base (engineering assumption, "
                "significantly annotated)"
                if self.base_condition.enable
                else "legacy literal 0.5"
            ),
        }
        # v2 条带分段计划与 per-worker 内存估算：统一在 config 层写入，
        # 使数据集级 metadata.json 与逐场景 resolution_metadata_json 携带
        # 结构一致（同键同值）的 surface_grid 载荷。
        metadata["surface_grid"]["segmentation"] = {
            "enabled": bool(estimate.segmented),
            "segment_count": int(estimate.segment_count),
            "segment_length_m": float(estimate.segment_length_m),
            "overlap_m": float(estimate.overlap_m),
            "macro_grid_spacing_m": float(self.strip_macro_grid_spacing_m()),
            "macro_band_cycles_per_m": [
                1.0 / float(max(self.road.length, 1e-9)),
                min(ISO8608_BAND_MAX_CYCLES_PER_M, 0.5 / self.road.grid_res),
            ],
            "texture_blend": "root_raised_cosine",
            "het_modulation": "global_1d_shared",
            "design_geometry": "fine_grid_post_blend",
        }
        metadata["surface_grid"]["worker_peak_mib"] = float(estimate.worker_peak_mib)
        metadata["surface_grid"]["assembly_mib"] = float(estimate.assembly_mib)
        metadata["sensor_output"] = {
            "observation_model": str(self.observation_model),
            "agc_mode": str(self.radiometric.agc_mode),
            "line_profiler": {
                "lateral_res_m": float(self.line_profiler.lateral_res_m),
                "step_m": float(self.line_profiler.step_m),
                "sensor_height_m": float(self.line_profiler.sensor_height_m),
                "laser_incidence_deg": float(self.line_profiler.laser_incidence_deg),
                "sigma_z0_m": float(self.line_profiler.sigma_z0_m),
                "speckle_shape": float(self.line_profiler.speckle_shape),
                "flyer_prob": float(self.line_profiler.flyer_prob),
                "brightness_dropout_ref": float(
                    self.line_profiler.brightness_dropout_ref
                ),
                "profile_count": None,
                "points_total": None,
                "points_per_profile_median": None,
                "flyer_ratio": None,
                "dropped_ratio": None,
                "crack_resolvable": None,
                "speckle_provenance": (
                    "Csencsics 2022 (DOI 10.1109/I2MTC48687.2022.9806610) "
                    "imported_not_self_calibrated; 2023 journal extension "
                    "needs manual verification; LSP specs (DOI "
                    "10.1109/ACCESS.2019.2956758) imported_not_self_calibrated"
                ),
            }
            if self.observation_model == "line-profiler"
            else None,
            "agc_reference_power": (
                None
                if self.radiometric.agc_mode == "scene-p99"
                else float(self.radiometric.agc_reference_power)
            ),
            "mode": self.sensor_output_mode,
                "requested_points": requested_points,
                "requested_density_points_per_m2": self.sensor_output_density_points_per_m2,
                "actual_points": output_points,
                "actual_density_points_per_m2": density,
                "nominal_equivalent_spacing_m": nominal_spacing,
                "surface_grid_spacing_is_preserved": False,
                "controlled_target_label_protection": (
                    {
                        "strategy": "minimum_points",
                        "minimum_points": int(self.target_label_min_output_points),
                        "fixed_fraction": None,
                    }
                    if self.sensor_output_mode == "fixed_count"
                    else {
                        # Voxel majority voting has no minimum-survival
                        # guarantee; the contract must say so explicitly.
                        "strategy": "none",
                        "minimum_points": None,
                        "fixed_fraction": None,
                    }
                ),
        }
        metadata["model_input"] = {
            "target_points": (
                None
                if self.model_target_points is None
                else int(self.model_target_points)
            ),
            "sampling_applied_by_generator": False,
        }
        # 病害选择口径审计：bernoulli（默认，disease_probs 即出现概率）
        # 或 legacy（加权恒定计数）。
        metadata["disease_sampling"] = {
            "sampling_mode": str(self.disease.sampling_mode),
            "max_diseases_per_scene": int(self.disease.max_diseases_per_scene),
        }
        return metadata


# 工具函数

def get_severity_label(
    pavement_type: Literal["asphalt", "concrete"],
    damage_type: str,
    severity: Literal["light", "severe"],
) -> int:
    """根据路面类型、病害类型和严重程度返回 JTG 标签 ID。

    Args:
        pavement_type: 路面类型。
        damage_type: 病害类型名称（中文或英文关键字）。
        severity: 严重程度。

    Returns:
        对应的 JTG 5210-2018 标签 ID (0-37)。

    Raises:
        ValueError: 无法识别的病害类型。
    """
    severity_map = {"light": "轻", "severe": "重"}

    asphalt_map = {
        ("龟裂", "轻"): 1, ("龟裂", "重"): 2,
        ("块状裂缝", "轻"): 3, ("块状裂缝", "重"): 4,
        ("纵向裂缝", "轻"): 5, ("纵向裂缝", "重"): 6,
        ("横向裂缝", "轻"): 7, ("横向裂缝", "重"): 8,
        ("坑槽", "轻"): 9, ("坑槽", "重"): 10,
        ("松散", "轻"): 11, ("松散", "重"): 12,
        ("沉陷", "轻"): 13, ("沉陷", "重"): 14,
        ("车辙", "轻"): 15, ("车辙", "重"): 16,
        ("波浪拥包", "轻"): 17, ("波浪拥包", "重"): 18,
        ("泛油", "-"): 19,
        ("修补", "-"): 20,
    }

    en_to_cn = {
        "alligator": "龟裂", "block": "块状裂缝",
        "longitudinal": "纵向裂缝", "transverse": "横向裂缝",
        "pothole": "坑槽", "raveling": "松散",
        "depression": "沉陷", "rutting": "车辙",
        "corrugation": "波浪拥包", "bleeding": "泛油",
        "patching": "修补",
    }

    concrete_map = {
        ("破碎板", "轻"): 21, ("破碎板", "重"): 22,
        ("裂缝", "轻"): 23, ("裂缝", "重"): 24,
        ("板角断裂", "轻"): 25, ("板角断裂", "重"): 26,
        ("错台", "轻"): 27, ("错台", "重"): 28,
        ("唧泥", "-"): 29,
        ("边角剥落", "轻"): 30, ("边角剥落", "重"): 31,
        ("接缝料损坏", "轻"): 32, ("接缝料损坏", "重"): 33,
        ("坑洞", "-"): 34,
        ("拱起", "-"): 35,
        ("露骨", "-"): 36,
        ("修补", "-"): 37,
    }

    concrete_en_to_cn = {
        "slab_shatter": "破碎板", "slab_crack": "裂缝",
        "corner_break": "板角断裂", "faulting": "错台",
        "pumping": "唧泥", "edge_spall": "边角剥落",
        "joint_damage": "接缝料损坏", "pitting": "坑洞",
        "blowup": "拱起", "exposed_aggregate": "露骨",
        "patching": "修补",
    }

    if pavement_type == "asphalt":
        cn_name = en_to_cn.get(damage_type, damage_type)
        sev = severity_map.get(severity, severity)
        key = (cn_name, sev) if sev != "-" else (cn_name, "-")
        if key in asphalt_map:
            return asphalt_map[key]
        # 无严重程度区分的类型
        if cn_name in ("泛油", "修补"):
            return asphalt_map[(cn_name, "-")]
        raise ValueError(f"Unknown asphalt damage: {damage_type} ({cn_name}) with severity {severity}")

    elif pavement_type == "concrete":
        cn_name = concrete_en_to_cn.get(damage_type, damage_type)
        sev = severity_map.get(severity, severity)
        key = (cn_name, sev) if sev != "-" else (cn_name, "-")
        if key in concrete_map:
            return concrete_map[key]
        # 无严重程度区分的类型
        for no_sev_type in ("唧泥", "坑洞", "拱起", "露骨", "修补"):
            if cn_name == no_sev_type:
                return concrete_map[(cn_name, "-")]
        raise ValueError(f"Unknown concrete damage: {damage_type} ({cn_name}) with severity {severity}")

    else:
        raise ValueError(f"Unknown pavement type: {pavement_type}")


if __name__ == "__main__":
    """自检：验证标签映射完整性。"""
    print("=== RoadMC Config Self-Test ===\n")

    assert NUM_CLASSES == 38, f"NUM_CLASSES should be 38, got {NUM_CLASSES}"
    print(f"[PASS] NUM_CLASSES = {NUM_CLASSES}")

    assert len(LABEL_MAP) == NUM_CLASSES, f"LABEL_MAP should have {NUM_CLASSES} entries, got {len(LABEL_MAP)}"
    print(f"[PASS] LABEL_MAP has {len(LABEL_MAP)} entries")

    assert min(ASPHALT_LABELS) == 0 and max(ASPHALT_LABELS) == 20
    print(f"[PASS] ASPHALT_LABELS range: {min(ASPHALT_LABELS)}-{max(ASPHALT_LABELS)}")

    assert min(CONCRETE_LABELS) == 21 and max(CONCRETE_LABELS) == 37
    print(f"[PASS] CONCRETE_LABELS range: {min(CONCRETE_LABELS)}-{max(CONCRETE_LABELS)}")

    assert get_severity_label("asphalt", "alligator", "light") == 1
    assert get_severity_label("asphalt", "alligator", "severe") == 2
    assert get_severity_label("asphalt", "longitudinal", "light") == 5
    assert get_severity_label("asphalt", "pothole", "severe") == 10
    assert get_severity_label("asphalt", "bleeding", "-") == 19
    assert get_severity_label("concrete", "slab_shatter", "light") == 21
    assert get_severity_label("concrete", "faulting", "severe") == 28
    assert get_severity_label("concrete", "pumping", "-") == 29
    assert get_severity_label("concrete", "exposed_aggregate", "-") == 36
    print("[PASS] get_severity_label 返回值正确")

    assert set(ISO_ROUGHNESS.keys()) == {"A", "B", "C", "D", "E"}
    print("[PASS] ISO_ROUGHNESS 包含 A-E 五个等级")

    cfg = GeneratorConfig()
    assert cfg.road.width == 7.0
    assert cfg.road.grid_res == 0.005
    assert cfg.num_points == 65536
    assert cfg.seed is None
    print("[PASS] GeneratorConfig 默认值合理")

    print("\n=== 全部自检通过 ===")
