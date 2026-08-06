"""RoadMC 合成数据生成模块。

JTG 5210-2018 病害分类体系，共 38 个标签 (0-37)。
"""

from .config import (  # noqa: F401
    ASPHALT_LABELS,
    CONCRETE_LABELS,
    ISO_ROUGHNESS,
    LABEL_MAP,
    NUM_CLASSES,
    BleedingConfig,
    ConcreteDamageConfig,
    CorrugationConfig,
    CrackConfig,
    DepressionConfig,
    DiseaseConfig,
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    PatchingConfig,
    PotholeConfig,
    RavelingConfig,
    RoadSurfaceConfig,
    RuttingConfig,
    get_severity_label,
)
from .primitives import (  # noqa: F401
    add_bleeding,
    add_concrete_damage,
    add_corrugation,
    add_crack,
    add_depression,
    add_micro_texture,
    add_patching,
    add_pothole,
    add_raveling,
    add_rutting,
    generate_road_surface,
    simulate_lidar_noise,
)
