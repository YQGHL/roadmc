"""
RoadMC 合成数据生成器配置模块。

使用 dataclass 定义所有生成参数，替代 YAML 配置文件。
所有参数均有类型注解和合理默认值，支持 seed 控制实现可复现生成。

遵循 JTG 5210-2018《公路技术状况评定标准》病害分类体系，共 38 个标签 (0-37)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 38
"""JTG 5210-2018 病害分类总数（含背景标签 0）。"""

# 沥青路面病害标签 ID 范围
ASPHALT_LABELS: Tuple[int, ...] = tuple(range(0, 21))
# 水泥路面病害标签 ID 范围
CONCRETE_LABELS: Tuple[int, ...] = tuple(range(21, 38))

# ISO 8608 路面功率谱密度参数 — 各粗糙度等级对应的不平整度系数 (×10⁻⁶ m³/cycle)
ISO_ROUGHNESS: Dict[str, float] = {
    "A": 16,      # 极好
    "B": 64,      # 好
    "C": 256,     # 一般
    "D": 1024,    # 差
    "E": 4096,    # 很差
}

# ---------------------------------------------------------------------------
# JTG 5210-2018 病害标签定义
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[int, Dict[str, str]] = {
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
    36: {"type": "��骨",       "severity": "-",  "pavement": "水泥"},
    37: {"type": "修补",       "severity": "-",  "pavement": "水泥"},
}


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class RoadSurfaceConfig:
    """路面宏观轮廓生成参数。

    Attributes:
        width: 路面宽度 (米)，默认双车道 7m。
        length: 路面长度 (米)，默认 5m。
        grid_res: 网格分辨率 (米)，默认 5mm 即 0.005m。
        pavement_type: 路面类型，'asphalt' 或 'concrete'。
        roughness_class: ISO 8608 粗糙度等级，'A'~'E'。
    """

    width: float = 7.0
    length: float = 5.0
    grid_res: float = 0.005
    pavement_type: Literal["asphalt", "concrete"] = "asphalt"
    roughness_class: Literal["A", "B", "C", "D", "E"] = "A"


@dataclass
class MicroTextureConfig:
    """微观纹理叠加 (fBm) 参数。

    Attributes:
        amplitude: fBm 振幅 (米)，默认 0.001m = 1mm。
        hurst: Hurst 指数，控制表面粗糙度。H≈0.5 为标准布朗运动，
               H→1 为平滑，H→0 为粗糙。默认 0.7。
        octaves: 叠加倍频程数，默认 6。
    """

    amplitude: float = 0.001
    hurst: float = 0.7
    octaves: int = 6


@dataclass
class CrackConfig:
    """裂缝生成参数。

    Attributes:
        crack_types: 允许生成的裂缝类型列表。
        severity_ratio: 轻度裂缝占比，默认 0.6 (重度 0.4)。
        bezier_points: Bézier 曲线控制点数，默认 4。
        fractal_perturbation: Perlin 噪声分形扰动幅度。
    """

    crack_types: List[str] = field(
        default_factory=lambda: ["longitudinal", "transverse", "alligator", "block"]
    )
    severity_ratio: float = 0.6
    bezier_points: int = 4
    fractal_perturbation: float = 0.02


@dataclass
class PotholeConfig:
    """坑槽生成参数。

    Attributes:
        max_radius_light: 轻度坑槽最大半径 (米)。
        max_radius_severe: 重度坑槽最大半径 (米)。
        max_depth_light: 轻度坑槽最大深度 (米)，JTG ≤ 25mm。
        max_depth_severe: 重度坑槽最大深度 (米)，JTG > 25mm。
        beta_range: 超椭圆指数范围，β=2 为椭球，β>2 为平底坑。
        edge_spall_prob: 边缘剥落概率。
    """

    max_radius_light: float = 0.15
    max_radius_severe: float = 0.30
    max_depth_light: float = 0.025
    max_depth_severe: float = 0.10
    beta_range: Tuple[float, float] = (2.0, 4.0)
    edge_spall_prob: float = 0.3


@dataclass
class RuttingConfig:
    """车辙生成参数。

    Attributes:
        wheel_separation: 左右轮迹间距 (米)，默认 1.8m。
        rut_width: 单条轮迹宽度 (米)，默认 0.6m。
        max_depth_light: 轻度车辙最大深度 (米)，JTG 10-15mm。
        max_depth_severe: 重度车辙最大深度 (米)，JTG > 15mm。
    """

    wheel_separation: float = 1.8
    rut_width: float = 0.6
    max_depth_light: float = 0.015
    max_depth_severe: float = 0.040


@dataclass
class CorrugationConfig:
    """波浪拥包生成参数。

    Attributes:
        wavelength_range: 波长范围 (米)。
        amplitude_light: 轻度振幅 (米)，JTG 10-25mm。
        amplitude_severe: 重度振幅 (米)，JTG > 25mm。
    """

    wavelength_range: Tuple[float, float] = (0.3, 1.0)
    amplitude_light: Tuple[float, float] = (0.010, 0.025)
    amplitude_severe: Tuple[float, float] = (0.025, 0.050)


@dataclass
class DepressionConfig:
    """沉陷生成参数。

    Attributes:
        max_radius: 最大沉陷半径 (米)。
        depth_light: 轻度深度范围 (米)，JTG 10-25mm。
        depth_severe: 重度深度范围 (米)，JTG > 25mm。
    """

    max_radius: float = 2.0
    depth_light: Tuple[float, float] = (0.010, 0.025)
    depth_severe: Tuple[float, float] = (0.025, 0.080)


@dataclass
class RavelingConfig:
    """松散生成参数。

    Attributes:
        point_removal_ratio_light: 轻度松散点移除比例。
        point_removal_ratio_severe: 重度松散点移除比例。
        pitting_depth: 点蚀深度 (米)。
    """

    point_removal_ratio_light: float = 0.05
    point_removal_ratio_severe: float = 0.20
    pitting_depth: float = 0.003


@dataclass
class BleedingConfig:
    """泛油生成参数。

    Attributes:
        intensity_boost: 反射率增加量 (0-1 范围)。
        region_width: 泛油区域宽度比例 (0-1)。
    """

    intensity_boost: float = 0.3
    region_width: float = 0.4


@dataclass
class ConcreteDamageConfig:
    """水泥路面损坏生成参数。

    Attributes:
        slab_length: 水泥板块长度 (米)，默认 5m。
        slab_width: 水泥板块宽度 (米)，默认 4m。
        joint_width: 接缝宽度 (米)，默认 8mm。
    """

    slab_length: float = 5.0
    slab_width: float = 4.0
    joint_width: float = 0.008


@dataclass
class LidarNoiseConfig:
    """LiDAR 噪声仿真参数。

    Attributes:
        distance_noise_std: 距离测量高斯噪声标准差 (米)。
        dropout_rate: 点丢失比例。
        angular_jitter_deg: 角度抖动标准差 (度)。
        mixed_pixel_prob: 混合像素效应概率。
    """

    distance_noise_std: float = 0.01
    dropout_rate: float = 0.03
    angular_jitter_deg: float = 0.01
    mixed_pixel_prob: float = 0.02


@dataclass
class DiseaseConfig:
    """病害生成参数聚合配置。

    每种病害出现的概率 (0-1)，0 表示不生成该类型。
    severity_ratio 全局默认轻度占比。

    Attributes:
        disease_probs: 各病害类型生成概率字典。
        severity_ratio: 全局轻度占比，默认 0.6。
        max_diseases_per_scene: 每场景最大病害数，默认 3。
    """

    disease_probs: Dict[str, float] = field(default_factory=lambda: {
        # 沥青路面病害
        "crack": 0.25,           # 裂缝（含纵向、横向、龟裂、块状）
        "pothole": 0.10,          # 坑槽
        "raveling": 0.08,         # 松散
        "depression": 0.10,       # 沉陷
        "rutting": 0.12,          # 车辙
        "corrugation": 0.05,      # 波浪拥包
        "bleeding": 0.05,         # 泛油
        # 水泥路面病害
        "concrete_damage": 0.15,  # 水泥路面损坏（含多种子类型）
    })
    severity_ratio: float = 0.6
    max_diseases_per_scene: int = 3


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
        seed: 全局随机种子，默认 None (非确定性)。
        num_points: 每场景目标点数，默认 65536。
        normalize: 是否坐标归一化到单位球，默认 True。
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
    concrete_damage: ConcreteDamageConfig = field(default_factory=ConcreteDamageConfig)
    lidar_noise: LidarNoiseConfig = field(default_factory=LidarNoiseConfig)
    seed: Optional[int] = None
    num_points: int = 65536
    normalize: bool = True


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

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

    # 沥青路面病害映射
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

    # 英文关键字到中文的映射
    en_to_cn = {
        "alligator": "龟裂", "block": "块状裂缝",
        "longitudinal": "纵向裂缝", "transverse": "横向裂缝",
        "pothole": "坑槽", "raveling": "松散",
        "depression": "沉陷", "rutting": "车辙",
        "corrugation": "波浪拥包", "bleeding": "泛油",
        "patching": "修补",
    }

    # 水泥路面病害映射
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

    # 验证标签数量
    assert NUM_CLASSES == 38, f"NUM_CLASSES should be 38, got {NUM_CLASSES}"
    print(f"[PASS] NUM_CLASSES = {NUM_CLASSES}")

    # 验证标签映射完整性
    assert len(LABEL_MAP) == NUM_CLASSES, f"LABEL_MAP should have {NUM_CLASSES} entries, got {len(LABEL_MAP)}"
    print(f"[PASS] LABEL_MAP has {len(LABEL_MAP)} entries")

    # 验证沥青路面标签范围
    assert min(ASPHALT_LABELS) == 0 and max(ASPHALT_LABELS) == 20
    print(f"[PASS] ASPHALT_LABELS range: {min(ASPHALT_LABELS)}-{max(ASPHALT_LABELS)}")

    # 验证水泥路面标签范围
    assert min(CONCRETE_LABELS) == 21 and max(CONCRETE_LABELS) == 37
    print(f"[PASS] CONCRETE_LABELS range: {min(CONCRETE_LABELS)}-{max(CONCRETE_LABELS)}")

    # 验证 get_severity_label
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

    # 验证 ISO 粗糙度等级
    assert set(ISO_ROUGHNESS.keys()) == {"A", "B", "C", "D", "E"}
    print("[PASS] ISO_ROUGHNESS 包含 A-E 五个等级")

    # 验证配置类默认值
    cfg = GeneratorConfig()
    assert cfg.road.width == 7.0
    assert cfg.road.grid_res == 0.005
    assert cfg.num_points == 65536
    assert cfg.seed is None
    print("[PASS] GeneratorConfig 默认值合理")

    print("\n=== 全部自检通过 ===")