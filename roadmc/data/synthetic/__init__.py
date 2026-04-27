"""RoadMC 合成数据生成模块。

JTG 5210-2018 病害分类体系，共 38 个标签 (0-37)。
"""

from .config import (  # noqa: F401
    GeneratorConfig,
    RoadSurfaceConfig,
    DiseaseConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    CrackConfig,
    PotholeConfig,
    RuttingConfig,
    CorrugationConfig,
    DepressionConfig,
    RavelingConfig,
    BleedingConfig,
    ConcreteDamageConfig,
    NUM_CLASSES,
    LABEL_MAP,
    ISO_ROUGHNESS,
    ASPHALT_LABELS,
    CONCRETE_LABELS,
    get_severity_label,
)

from .primitives import (  # noqa: F401
    generate_road_surface,
    add_micro_texture,
    add_crack,
    add_pothole,
    add_raveling,
    add_depression,
    add_rutting,
    add_corrugation,
    add_bleeding,
    add_concrete_damage,
    simulate_lidar_noise,
)