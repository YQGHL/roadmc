"""RoadMC 合成数据集生成器 —— SyntheticRoadDataset.

组合 config.py 的配置参数和 primitives.py 的基元函数，
生成符合 JTG 5210-2018 标准的合成道路点云数据集。

数据流 (Data Pipeline)：
  1. 确定路面类型 (asphalt / concrete)
  2. ISO 8608 PSD 路面宏观轮廓生成
  3. fBm 微观纹理叠加
  4. 病害随机选择与组合应用
  5. 特征计算 (强度反射率 + 曲率)
  6. LiDAR 噪声仿真
  7. 标签/特征最近邻传递
  8. 定点数重采样 & 归一化
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import textwrap
import warnings

import numpy as np

if os.environ.get("ROADMC_GENERATOR_NO_TORCH") == "1":
    torch = None  # type: ignore[assignment]
    _DatasetBase = object
else:
    import torch

    _DatasetBase = torch.utils.data.Dataset

try:
    from ..features import (
        DEFAULT_GEOMETRY_K_NEIGHBORS,
        OBSERVABLE_FEATURE_NAMES,
        OBSERVABLE_FEATURE_SCHEMA,
        compute_observable_features,
    )
    from .config import (
        NUM_CLASSES,
        BleedingConfig,
        DepressionConfig,
        GeneratorConfig,
        PatchingConfig,
        PotholeConfig,
    )
    from .labels import TargetLabelSpec, target_spec_for_label
    from .primitives import (
        add_bleeding,
        add_concrete_damage,
        add_corrugation,
        add_crack,
        add_depression,
        add_patching,
        add_pothole,
        add_raveling,
        add_rutting,
        generate_curb_strip,
        generate_road_surface,
        generate_strip_surface,
        geometric_occlusion_keep_mask,
        local_depression_depth,
        resample_to_lidar_pattern,
        simulate_lidar_noise,
        synthesize_strip_macro_band,
    )
except ImportError:
    # Fallback for standalone / -m execution
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parents[2])  # project root
    _parent = str(_Path(__file__).resolve().parent)     # synthetic/ dir
    for _p in (_root, _parent):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    from config import (  # type: ignore[no-redef]
        NUM_CLASSES,
        BleedingConfig,
        DepressionConfig,
        GeneratorConfig,
        PatchingConfig,
        PotholeConfig,
    )
    from labels import TargetLabelSpec, target_spec_for_label  # type: ignore[no-redef]
    from primitives import (  # type: ignore[no-redef]
        add_bleeding,
        add_concrete_damage,
        add_corrugation,
        add_crack,
        add_depression,
        add_patching,
        add_pothole,
        add_raveling,
        add_rutting,
        generate_curb_strip,
        generate_road_surface,
        generate_strip_surface,
        geometric_occlusion_keep_mask,
        local_depression_depth,
        resample_to_lidar_pattern,
        simulate_lidar_noise,
        synthesize_strip_macro_band,
    )

    from roadmc.data.features import (  # type: ignore[no-redef]
        DEFAULT_GEOMETRY_K_NEIGHBORS,
        OBSERVABLE_FEATURE_NAMES,
        OBSERVABLE_FEATURE_SCHEMA,
        compute_observable_features,
    )

CONCRETE_DAMAGE_TYPES: list[str] = [
    "slab_shatter",
    "slab_crack",
    "corner_break",
    "faulting",
    "pumping",
    "edge_spall",
    "joint_damage",
    "pitting",
    "blowup",
    "exposed_aggregate",
    "patching",
]

# 无严重程度区分的混凝土损坏类型（标签直接固定）
CONCRETE_NO_SEVERITY: set = {"pumping", "pitting", "blowup", "exposed_aggregate"}

# 沥青路面可用的病害 key（对应 disease_probs 字典）
ASPHALT_DISEASE_KEYS: list[str] = [
    "crack",
    "pothole",
    "raveling",
    "depression",
    "rutting",
    "corrugation",
    "bleeding",
    "patching",
]

# 病害应用的自然顺序（大尺度 → 小尺度）
DISEASE_APPLY_ORDER: dict[str, int] = {
    "corrugation": 0,
    "rutting": 1,
    "depression": 2,
    "pothole": 3,
    "crack": 4,
    "bleeding": 5,
    "patching": 3,
    "raveling": 6,
    "concrete_damage": 0,
}

# P1-3 修复：病害标签优先级，数值越高优先级越高
# 当多个病害覆盖同一区域时，仅保留优先级最高的标签
# 设计原则（与下表一致）：坑槽 > 龟裂 > 纵向裂缝 > 横向裂缝 ≥ 块状
# 裂缝（重横向 8→4 高于重块状 4→3；轻横向 7 与重块状 4 同为 3），
# 大面积病害（车辙、沉陷、波浪拥包）优先级低于局部病害。
# 打平时的确定性策略：**后应用者保留**——restore 条件是
# old_prio > new_prio 严格大于，优先级相等时后写入的新标签胜出，
# 应用先后由 DISEASE_APPLY_ORDER 决定。
LABEL_PRIORITY: dict[int, int] = {
    0: 0,    # 背景
    # 沥青路面 — 局部破损优先级最高，大面积病害较低
    10: 10,  # 重坑槽 — 最高优先级
    9: 9,    # 轻坑槽
    2: 8,    # 重龟裂
    1: 7,    # 轻龟裂
    6: 6,    # 重纵向裂缝
    5: 5,    # 轻纵向裂缝
    8: 4,    # 重横向裂缝
    4: 3,    # 重块状裂缝
    7: 3,    # 轻横向裂缝
    3: 2,    # 轻块状裂缝
    12: 5,   # 重松散
    11: 4,   # 轻松散
    14: 2,   # 重沉陷
    13: 1,   # 轻沉陷
    16: 2,   # 重车辙
    15: 1,   # 轻车辙
    18: 2,   # 重波浪拥包
    17: 1,   # 轻波浪拥包
    19: 5,   # 泛油（标签特殊，面积区域覆盖）
    20: 6,   # 修补（人工标记，高优先级）
    # 水泥路面
    22: 10,  # 重破碎板
    21: 9,   # 轻破碎板
    24: 8,   # 重水泥裂缝
    23: 7,   # 轻水泥裂缝
    26: 6,   # 重板角断裂
    25: 5,   # 轻板角断裂
    28: 4,   # 重错台
    27: 3,   # 轻错台
    31: 4,   # 重边角剥落
    30: 3,   # 轻边角剥落
    33: 2,   # 重接缝料损坏
    32: 1,   # 轻接缝料损坏
    29: 5,   # 唧泥
    34: 6,   # 坑洞
    35: 7,   # 拱起
    36: 4,   # 露骨
    37: 8,   # 水泥修补
}


# ----------------------------------------------------------------------
# 病害几何采样助手（单块路径与分段计划路径共用；同一 helper 保证两条
# 路径的 rng 消耗顺序一致）
# ----------------------------------------------------------------------

def _sample_wheelpath_x(rng: np.random.Generator, width: float) -> float:
    """轮迹带横向先验：车道中心 ±0.9 m 的双峰高斯混合（σ=0.25 m）。

    疲劳类病害（龟裂、纵缝、坑槽、车辙）机理上集中在轮迹带；
    双轮迹间距取 RuttingConfig 的 1.8 m 口径。
    """
    center = width / 2.0
    offset = 0.9 if rng.random() < 0.5 else -0.9
    return float(np.clip(rng.normal(center + offset, 0.25), 0.3, max(width - 0.3, 0.31)))


def _sample_pothole_geometry(
    rng: np.random.Generator,
    cfg: PotholeConfig,
    severity: str,
) -> tuple[float, float]:
    """坑槽 (radius, depth) 采样。

    轻/重共用半径区间 [0.05, max_radius_severe]（半径不按严重度切
    分——解耦设计，平面半径不得成为 100% 可分的非规范线索）；深度
    按严重度取各自档位。``aspect_ratio_range`` 非 None 时按有界纵横
    比先验 d/R ∈ (lo, hi) 拒绝重采（排除沉陷碟形与近竖井类非物理
    极端几何；设计界，非标准规定）；None = 历史 R、d 独立采样。
    """
    if cfg.aspect_ratio_range is None:
        radius = float(rng.uniform(0.05, cfg.max_radius_severe))
        if severity == "light":
            depth = float(rng.uniform(0.005, cfg.max_depth_light))
        else:
            depth = float(rng.uniform(cfg.max_depth_light * 1.1, cfg.max_depth_severe))
        return radius, depth
    lo, hi = cfg.aspect_ratio_range
    while True:
        radius = float(rng.uniform(0.05, cfg.max_radius_severe))
        if severity == "light":
            depth = float(rng.uniform(0.005, cfg.max_depth_light))
        else:
            depth = float(rng.uniform(cfg.max_depth_light * 1.1, cfg.max_depth_severe))
        ratio = depth / radius
        if lo <= ratio <= hi:
            return radius, depth


def _sample_depression_geometry(
    rng: np.random.Generator,
    cfg: DepressionConfig,
    width: float,
    length: float,
) -> tuple[float, float, float]:
    """沉陷 (cx, cy, radius) 采样。

    ``center_inset_mode='radius'``（默认）：先拒绝重采 R ≤
    min(W, L)/2，再令中心内缩量 ≥ R（cx ∈ [R, W−R]、cy ∈ [R, L−R]），
    完整碗形恒在路面足迹内；``'legacy'``：历史固定内缩 0.5 m（碗形
    可被路缘截断）。
    """
    lo_r, hi_r = cfg.radius_range
    if cfg.center_inset_mode == "legacy":
        cx = float(rng.uniform(0.5, width - 0.5))
        cy = float(rng.uniform(0.5, length - 0.5))
        radius = float(rng.uniform(lo_r, hi_r))
        return cx, cy, radius
    half_min = min(width, length) / 2.0
    if lo_r > half_min:
        raise ValueError(
            f"depression radius_range lower bound {lo_r:g} m exceeds half the "
            f"smaller road extent ({half_min:g} m): no full bowl fits the "
            "footprint. Shrink radius_range or enlarge the road."
        )
    while True:
        radius = float(rng.uniform(lo_r, hi_r))
        if radius <= half_min:
            break
    cx = float(rng.uniform(radius, width - radius))
    cy = float(rng.uniform(radius, length - radius))
    return cx, cy, radius


def _bleeding_hash01(x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
    """确定性逐点散列，返回 [0, 1) 均匀值（泛油边界过渡判定用）。

    同一物理坐标在任何分段窗口得到同一取值——云无关，保证缝两侧
    标签逐位一致。与 primitives 的坐标哈希同族，此处独立实现以免
    耦合基元内部约定。
    """
    h = np.sin(x * 127.1 + y * 311.7 + float(salt) * 74.7) * 43758.5453
    return h - np.floor(h)


def _bleeding_patch_params(
    rng: np.random.Generator,
    cfg: BleedingConfig,
    width: float,
    length: float,
    salt: int,
) -> dict:
    """泛油斑块参数（``geometry_mode='patch'`` 口径）。

    中心走轮迹带先验（机理上泛油与轮迹带交通作用相关）、y 向有限
    长度、面积占比 U(5%, 25%)（设计先验）。面积分解：w = f·W·u、
    l = L/u（u ∈ [1.5, min(4, 0.95/f)]），保证 w ≤ W、l ≤ L 且
    w·l = f·W·L 精确成立。
    """
    frac = float(rng.uniform(0.05, 0.25))
    u = float(rng.uniform(1.5, min(4.0, 0.95 / frac)))
    w_band = min(frac * width * u, width)
    l_band = length / u
    cx = _sample_wheelpath_x(rng, width)
    cy = float(rng.uniform(l_band / 2.0, length - l_band / 2.0))
    return {
        "center": (cx, cy),
        "width": w_band,
        "length": l_band,
        "salt": int(salt),
    }


def _bleeding_region_mask(points: np.ndarray, patch: dict) -> np.ndarray:
    """泛油斑块的布尔掩码：长方形斑块 + logistic 边界过渡。

    p_in = σ(−signed_dist/τ)，τ = 0.025 m ⇒ 10-90% 过渡带 ≈ 0.11 m
    （消除历史硬二值边界）；逐点用确定性散列判定（同一物理点在任
    何段窗口得到同一标签决策，云无关）。
    """
    cx, cy = patch["center"]
    w_band, l_band = float(patch["width"]), float(patch["length"])
    dx = np.abs(points[:, 0] - cx)
    dy = np.abs(points[:, 1] - cy)
    signed_dist = np.maximum(dx - w_band / 2.0, dy - l_band / 2.0)
    tau = 0.025
    with np.errstate(over="ignore"):
        p_in = 1.0 / (1.0 + np.exp(signed_dist / tau))
    u01 = _bleeding_hash01(points[:, 0], points[:, 1], patch["salt"])
    return u01 < p_in


def _lambert_cos_theta(
    normals: np.ndarray,
    d_hat: np.ndarray,
    cos_abs: bool,
) -> np.ndarray:
    """朗伯入射角因子：cosθ = n·(−d̂)（I ∝ ρ·cosθ/R² 的几何项）。

    ``cos_abs=False``（默认）：直接取 n·(−d̂)，背向面 clip 到 0，
    下限 0.03 仅保留给数值稳定的前向掠射点（0 < cosθ < 0.03）。
    ``cos_abs=True``：历史口径 clip(|n·(−d̂)|, 0.03, 1)——背向面被
    当作朝向前亮度（物理方向错误，仅作回退通道保留，
    CLI --lambert-abs-legacy）。车载俯视路面（法向 ≈ +z、传感器在
    上）下两种口径差异集中在侧向路缘/坑壁背光面。
    """
    raw = np.sum(normals * (-d_hat), axis=1)
    if cos_abs:
        return np.clip(np.abs(raw), 0.03, 1.0)
    cos_t = np.clip(raw, 0.0, 1.0)
    return np.where(cos_t > 0.0, np.maximum(cos_t, 0.03), 0.0)


# ----------------------------------------------------------------------
# 装配层 ↔ 基元层的合入顺序桥接
# ----------------------------------------------------------------------

_WARNED_PENDING_KEYS: set[str] = set()
"""已警告过的待合入键（每键每进程一次，避免批量生成刷屏）。"""

# 本轮装配引入、需基元组消费的 params 键。披露范围仅限这些键——不检查
# 基元 params 字典里的历史键（如从未被读取的 "grid_res"），避免把既有
# 死键误报为待合入。
_INTRODUCED_PARAMS_KEYS: frozenset = frozenset({
    "beta_mode", "micropit_rate", "micropit_mode", "spall_band",
    "perturb_mode", "width_truncation", "area_lambda_mode", "area_label_mode",
    "crack_in_slab", "fault_scope", "edge_spall_mode", "shatter_mode",
    "label_radius_mode", "modulation_epsilon", "removal_ratio", "erosion_depth",
})

# 请求值与基元当前行为等价的取值表：某键尚未被基元消费时，若请求值属于
# 该集合，则行为与基元当前实现一致——跳过/忽略不构成降级，无需披露
# （例如 full-legacy 回退运行在当前树上应得到空的待合入集合）。值为集合
# 时任一成员相等即等价；无条目的键请求任何值都视为新行为。
_LEGACY_EQUIVALENT_VALUES: dict[str, frozenset] = {
    "macro_spectrum_mode": frozenset({"legacy"}),
    "grid_convention": frozenset({"legacy"}),
    "het_field": frozenset({"1d"}),
    "label_radius_mode": frozenset({"legacy-1pct"}),
    "beta_mode": frozenset({"legacy"}),
    "micropit_mode": frozenset({"legacy"}),
    "perturb_mode": frozenset({"legacy"}),
    "width_truncation": frozenset({"clip"}),
    "area_lambda_mode": frozenset({"half"}),
    "area_label_mode": frozenset({"legacy"}),
    "fault_scope": frozenset({"legacy-strip"}),
    "edge_spall_mode": frozenset({"legacy"}),
    "shatter_mode": frozenset({"legacy"}),
    "crack_in_slab": frozenset({False}),
    "aspect_ratio_range": frozenset({None}),
    "spall_band": frozenset({(0.8, 1.2)}),
    "modulation_epsilon": frozenset({0.2}),
    "removal_ratio": frozenset({0.05, 0.20}),
    "erosion_depth": frozenset({(0.002, 0.005)}),
}

# 受模式键治理的键：模式键取等价值时被治理键不参与行为（如
# micropit_mode='legacy' 下旧实现按钳制计数，micropit_rate 不生效），
# 一并静默。
_PARAM_MOOT_UNDER: dict[str, tuple[str, frozenset]] = {
    "micropit_rate": ("micropit_mode", frozenset({"legacy"})),
}


def _surface_moot_keys(micro_cfg) -> frozenset:
    """当前配置下行为无关（取任何值输出都相同）的表面装配键集合。

    het_sigma_ln == 0 时 het 调制整体关闭——primitives.py 单块路径
    （``if texture_het_sigma_ln > 0.0``，约 433 行）与条带路径
    （约 683 行）双重门控——het_field 取 '1d'/'2d' 行为完全相同，
    故此时 het_field 不进入待合入披露。这正是 full-legacy 回退配方
    刻意不含 ``--het-field-1d`` 的依据（het 默认关闭，het_field 无
    默认行为变更，设计稿 tuning#10）：缺少本规则时回退运行会把
    het_field 误报为 pending，违背 README『full-legacy 回退运行的
    待合入集合为空』的回归门契约。sigma > 0 时调制实际生效，
    het_field 恢复参与披露（'2d' 未合入 → pending，'1d' → 等价静默）。
    """
    if float(getattr(micro_cfg, "het_sigma_ln", 0.0)) <= 0.0:
        return frozenset({"het_field"})
    return frozenset()

_PARAMS_SOURCE_CACHE: dict[int, str] = {}


def _register_pending(key: str, callee_name: str) -> None:
    if key not in _WARNED_PENDING_KEYS:
        _WARNED_PENDING_KEYS.add(key)
        warnings.warn(
            f"primitive {callee_name!r} does not consume {key!r} yet "
            "(primitives-side change pending); the generator keeps the "
            "primitive's current behaviour for this knob and the scene "
            "metadata records it under 'pending_primitive_upgrades'.",
            stacklevel=3,
        )


def _primitive_accepts_kwarg(callee, name: str) -> bool:
    try:
        sig = inspect.signature(callee)
    except (TypeError, ValueError):
        return False
    return any(
        p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def _primitive_source_text(callee) -> str:
    cached = _PARAMS_SOURCE_CACHE.get(id(callee))
    if cached is None:
        try:
            cached = inspect.getsource(callee)
        except (OSError, TypeError):
            cached = ""
        _PARAMS_SOURCE_CACHE[id(callee)] = cached
    return cached


def _params_key_consumed(callee, key: str, _depth: int = 0) -> bool:
    """键是否被基元从 params 读取（AST 静态分析，非子串匹配）。

    用 ``ast`` 解析基元源码，只识别三种真实读取形态：
    ``params["key"]``、``params.get("key"[, ...])``、``"key" in params``
    （含 ``not in``）。注释与 docstring 不产生对应 AST 节点，不会再
    因源码文本中出现键名而误判为"已消费"（静默漏报的方向被关闭）。
    ``inspect.getsource`` 失败或语法解析失败时按未消费处理
    （over-report，披露语义的安全方向）。另跟随一层"把 params 作为
    实参传给同一定义命名空间内函数"的调用（经 ``__globals__`` 解析，
    覆盖键在辅助函数内读取的常见形态）；更深层的间接读取（别名、
    跨模块）仍按未消费处理并在此声明。
    """
    src = _primitive_source_text(callee)
    if not src:
        return False
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return False

    def _const_str(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _is_params(node) -> bool:
        return isinstance(node, ast.Name) and node.id == "params"

    helper_names: set[str] = set()
    for node in ast.walk(tree):
        # params["key"]
        if (
            isinstance(node, ast.Subscript)
            and _is_params(node.value)
            and _const_str(node.slice) == key
        ):
            return True
        # params.get("key"[, ...])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_params(node.func.value)
            and node.args
            and _const_str(node.args[0]) == key
        ):
            return True
        # "key" in params / "key" not in params
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and _const_str(node.left) == key
            and len(node.comparators) == 1
            and _is_params(node.comparators[0])
        ):
            return True
        # helper(..., params, ...) → 记录候选辅助函数，稍后一层跟随
        if (
            _depth == 0
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and any(_is_params(a) for a in node.args)
        ):
            helper_names.add(node.func.id)

    if _depth == 0 and helper_names:
        # 用函数自身的定义命名空间解析辅助函数（不依赖 sys.modules，
        # 动态加载的模块同样适用）。
        namespace = getattr(callee, "__globals__", None) or {}
        for name in helper_names:
            helper = namespace.get(name)
            if callable(helper) and _params_key_consumed(helper, key, _depth=1):
                return True
    return False


def _filter_unsupported_kwargs(
    callee, requested: dict, moot_keys: frozenset = frozenset()
) -> tuple[dict, list[str]]:
    """顶层 kwargs 桥接：签名支持的键透传；不支持的键剔除。

    被剔除键的处理顺序：(1) 属于 ``moot_keys``（当前配置下行为无关，
    任何取值输出相同，见 ``_surface_moot_keys``）→ 静默剔除，不披露；
    (2) 请求值等于 ``_LEGACY_EQUIVALENT_VALUES`` 中的等价值（基元
    当前行为即所请求行为）→ 静默剔除；(3) 其余登记/警告并返回键名，
    由调用方并入场景级待合入集合。返回 ``(可转发子集, 本次待合入键)``。
    """
    try:
        sig = inspect.signature(callee)
    except (TypeError, ValueError):
        return dict(requested), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(requested), []
    names = frozenset(sig.parameters)
    accepted: dict = {}
    pending: list[str] = []
    callee_name = getattr(callee, "__name__", str(callee))
    for key, value in requested.items():
        if key in names:
            accepted[key] = value
        elif key in moot_keys:
            pass  # 当前配置下行为无关，静默剔除
        elif value in _LEGACY_EQUIVALENT_VALUES.get(key, frozenset()):
            pass  # 请求值 = 基元当前行为，静默剔除
        else:
            pending.append(key)
            _register_pending(key, callee_name)
    return accepted, pending


def _pending_params_keys(callee, params: dict) -> list[str]:
    """params 嵌套键的待合入披露（键仍随 params 下发，不剔除）。

    仅检查本轮装配引入的键（``_INTRODUCED_PARAMS_KEYS``）。判定顺序：
    受模式键治理且模式等价 → 静默；基元源码已从 params 读取该键（AST
    判定，见 ``_params_key_consumed``）→ 已消费；请求值等于 legacy
    等价值 → 静默；其余登记/警告并返回键名。
    基元批次合入后源码引用出现，本函数自动变为空转。
    """
    callee_name = getattr(callee, "__name__", str(callee))
    pending: list[str] = []
    for key, value in params.items():
        if key not in _INTRODUCED_PARAMS_KEYS:
            continue
        governor = _PARAM_MOOT_UNDER.get(key)
        if governor is not None:
            gov_key, gov_legacy = governor
            if params.get(gov_key) in gov_legacy:
                continue
        if _params_key_consumed(callee, key):
            continue
        if value in _LEGACY_EQUIVALENT_VALUES.get(key, frozenset()):
            continue
        pending.append(key)
        _register_pending(key, callee_name)
    return pending


def _bridge_depression_params(label_radius_mode: str) -> tuple[dict, list[str]]:
    """add_depression 的 params 装配桥接（fix#13 基元侧未合入时降级披露）。

    基元签名已支持 params 时透传；否则请求值为 'legacy-1pct'（基元当前
    行为）时静默省略，其他请求值登记/警告为待合入。
    """
    if _primitive_accepts_kwarg(add_depression, "params"):
        return {"params": {"label_radius_mode": label_radius_mode}}, []
    if label_radius_mode in _LEGACY_EQUIVALENT_VALUES["label_radius_mode"]:
        return {}, []
    _register_pending("label_radius_mode", "add_depression")
    return {}, ["label_radius_mode"]


class SyntheticRoadDataset(_DatasetBase):
    """JTG 5210-2018 合成道路点云数据集。

    使用 :class:`GeneratorConfig` 作为唯一配置入口，组合 config.py 的
    参数和 primitives.py 的物理基元，生成带有病害标注的道路点云场景。

    ``__getitem__`` 返回的字典包含:

    - **points**: ``(N, 3)`` torch.float32 — 归一化后的点云坐标
    - **labels**: ``(N,)`` torch.int64 — JTG 5210-2018 标签 (0-37)
    - **feats**: ``(N, 3)`` torch.float32 — [归一化强度, PCA 曲率, 有符号局部高度残差]
    - **normals**: ``(N, 3)`` torch.float32 — 表面单位法向量
    - **pavement_type**: str — 'asphalt' | 'concrete'

    Args:
        config: 顶层配置实例。
        dataset_size: 数据集总样本数，默认 2000。
    """

    def __init__(
        self,
        config: GeneratorConfig,
        dataset_size: int = 2000,
    ) -> None:
        # Re-run the preflight here because dataclass instances are mutable and
        # callers may have changed the road geometry after construction.
        self.surface_estimate = config.validate_surface_budget()
        if self.surface_estimate.segmented and config.observation_model == "on":
            raise NotImplementedError(
                "Strip of "
                f"{self.surface_estimate.point_count:,} points requires the "
                f"segmented regime ({self.surface_estimate.segment_count} segments "
                f"× {self.surface_estimate.segment_length_m:.3f} m), which "
                "currently supports observation_model='off' only: the sensor "
                "pipeline (scan resampling, occlusion, noise, NN transfer, "
                "output sampling) is not segmented in this build. Construct the "
                "dataset with observation_model='off', or raise "
                "max_surface_points / max_surface_memory_mib to hold the whole "
                "strip in one block."
            )
        self.config = config
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """生成第 ``idx`` 个场景并返回 torch 张量字典。"""
        if torch is None:
            raise RuntimeError("torch is disabled for this generator process")
        scene = self.generate_scene(idx)
        return {
            "points": torch.from_numpy(scene["points"]).float(),
            "labels": torch.from_numpy(scene["labels"]).long(),
            "feats": torch.from_numpy(scene["feats"]).float(),
            "normals": torch.from_numpy(scene["normals"]).float(),
            "pavement_type": scene["pavement_type"],
            "roughness_class": scene["roughness_class"],
        }

    def generate_scene(self, idx: int, target_label: int | None = None) -> dict:
        """生成一个完整的合成道路场景。

        内部流程按以下顺序执行：

        1. 选择路面类型 (asphalt / concrete)
        1.5 粗糙度等级抽取（roughness_mix 开启时逐场景按权重抽样）
        2. ISO 8608 PSD 路面轮廓 + fBm 微观纹理
        2.4 缘石条带生成（可选，暂不并入）
        2.5 LiDAR 扫描线重采样 (可选, P1-1；缘石点经临时合并云同模型通过)
        3. 病害组合随机选择 & 应用（按 DISEASE_APPLY_ORDER 排序 + LABEL_PRIORITY）
        3.5 缘石条带并入（可选，curb.width_m > 0；背景标签 0 的非路面点）
        4. KDTree 局部曲率计算 (P2-2)
        5. 应用松散 (raveling) — 产生 NaN 点
        6. 松散标签膨胀 + NaN 过滤 (P0-3)
        7. 强度反射率计算 (P0-2)
        8. LiDAR 噪声仿真 (P2-3: 仅高曲率边缘混合)
        9. 最近邻传递标签/特征/法向量 (P0-4: 带距离阈值)
        10. 传感器输出密度体素化或固定点数重采样
        11. 坐标归一化
        12. 特征拼接与三层分辨率元数据

        观测模型体制（``config.observation_model``）：

        - ``"on"``（配置默认）：以上管线全部执行，与历史版本逐位一致。
        - ``"off"``：观测体制关闭——直接输出全部表面点。跳过 2.5 扫描
          重采样、4 曲率（其唯一消费者是噪声阶段边缘混合）、6.5 几何
          遮挡（传感器可见性现象；``local_depression_depth`` 仍计算并
          供强度遮蔽使用）、8 LiDAR 噪声、9 最近邻传递、10 输出采样
          （取 retain-all 语义，``noisy_points := points``）；保留路面谱
          合成、全部病害、松散及 NaN 过滤、强度、归一化、特征。
          ``num_points``/``target_density`` 被忽略但记录在元数据
          （``ignored_under_observation_off``）；off + 扫描开关开启时
          ``warnings.warn`` 并跳过扫描（元数据记 override）。

        Args:
            idx: 场景索引，用于种子生成 (config.seed + idx)。

        Returns:
            包含 points, labels, feats, normals, pavement_type,
            roughness_class 的字典。
        """
        # 种子管理
        target_spec: TargetLabelSpec | None = None
        if target_label is not None:
            target_spec = target_spec_for_label(target_label)

        scene_seed = self.config.seed + idx if self.config.seed is not None else None
        rng = np.random.default_rng(scene_seed)

        grid_res = self.config.road.grid_res
        width = self.config.road.width
        length = self.config.road.length
        surface_estimate = self.config.validate_surface_budget()
        observation_off = self.config.observation_model == "off"
        if surface_estimate.segmented:
            if not observation_off:
                # on+segmented：观测管线（扫描/遮挡/噪声/最近邻传递/输出
                # 采样）尚未分段化——显式拒绝而不是无声地走单块路径超
                # 预算；off 是分段体制的主 regime，走完整分段运行时。
                raise NotImplementedError(
                    "Strip of "
                    f"{surface_estimate.point_count:,} points requires the "
                    f"segmented regime ({surface_estimate.segment_count} segments "
                    f"× {surface_estimate.segment_length_m:.3f} m, overlap "
                    f"{surface_estimate.overlap_m:.3f} m), which currently "
                    "supports observation_model='off' only: the sensor pipeline "
                    "is not segmented in this build. Use observation_model='off', "
                    "or raise max_surface_points / max_surface_memory_mib to hold "
                    "the whole strip in one block."
                )
            return self._generate_scene_segmented_off(
                idx, target_spec, target_label, surface_estimate, rng
            )

        # 1. 路面类型选择
        pavement_type = (
            target_spec.pavement_type
            if target_spec is not None
            else self._select_pavement_type(rng)
        )

        # 1.5 粗糙度等级抽取：roughness_mix 开启时逐场景按权重抽样一次
        # （开启态唯一新增的 rng 抽取），实际等级写入场景字段并在谱合成
        # 中生效；关闭路径直接回退配置的单一等级，零新增 rng 消耗，既有
        # 随机流逐位不变。
        if self.config.road.roughness_mix is not None:
            mix = self.config.road.roughness_mix
            classes, weights = zip(*sorted(mix.items()), strict=False)
            p = np.asarray(weights, dtype=np.float64)
            p = p / p.sum()
            roughness_actual = str(rng.choice(list(classes), p=p))
        else:
            roughness_actual = self.config.road.roughness_class

        # 2. 路面宏观轮廓 + 纹理谱段 + 设计几何（单次谱合成，法向从
        # 最终高度场统一重算——纹理不再作为独立的逐点位移步骤）。
        # 诚实化口径：两类路面**共享同一几何宏观谱与纹理谱**；
        # pavement_type 仅影响病害集合与强度材料先验（基线反照率），
        # 不建模材料纹理分化。本调用点已停传 pavement_type（fix#17
        # 调用侧一半；基元签名中残留的形参由基元组批次移除，本组不动
        # primitives.py）。
        # macro_spectrum_mode/grid_convention/het_field 为本轮新增装配
        # 键：基元签名尚未支持时由桥接层跳过转发。请求值等于基元当前
        # 行为（legacy 等价值）的键静默；其余键记入本场景的
        # pending_scene，随元数据 pending_primitive_upgrades 披露。
        pending_scene: set[str] = set()
        surface_extra, surface_pending = _filter_unsupported_kwargs(
            generate_road_surface,
            {
                "macro_spectrum_mode": self.config.road.macro_spectrum_mode,
                "grid_convention": self.config.micro_texture.grid_convention,
                "het_field": self.config.micro_texture.het_field,
            },
            moot_keys=_surface_moot_keys(self.config.micro_texture),
        )
        pending_scene.update(surface_pending)
        points, normals = generate_road_surface(
            width=width,
            length=length,
            grid_res=grid_res,
            roughness_class=roughness_actual,
            seed=int(rng.integers(0, 2 ** 31)),
            texture_rms=self.config.micro_texture.amplitude,
            texture_hurst=self.config.micro_texture.hurst,
            crossfall=self.config.road.crossfall,
            crossfall_shape=self.config.road.crossfall_shape,
            longitudinal_grade=self.config.road.longitudinal_grade,
            texture_het_sigma_ln=self.config.micro_texture.het_sigma_ln,
            texture_het_corr_len_m=self.config.micro_texture.het_corr_len_m,
            **surface_extra,
        )
        # 2.4 缘石条带（可选）：行车道横向外侧的抬高顶面条带，几何确定性、
        # 零 rng 消耗。此处只**生成**、暂不并入——第 3 步病害与第 5 步松散
        # 必须在纯路面足迹上进行：否则 add_depression 的影响半径、
        # add_corrugation/add_concrete_damage 的整云 bbox 推导几何、泛油掩码
        # 与 raveling 区域都会越过路缘，把病害标签与 z 调制写到抬高的缘石
        # 结构上（违背设计保证：缘石点恒为背景标签 0，且污染 13/14、17/18
        # 等类的 prevalence 分子）。缘石点在第 3 步之后并入，此后作为普通
        # 非路面点参与曲率/强度/遮挡/噪声/密度统计，并因点数增加稀释自然
        # prevalence 分母——该选项的预期效应。
        curb_points: np.ndarray | None = None
        curb_normals: np.ndarray | None = None
        if self.config.curb.width_m > 0:
            curb_points, curb_normals = generate_curb_strip(
                road_width=width,
                length=length,
                grid_res=grid_res,
                curb_width_m=self.config.curb.width_m,
                curb_height_m=self.config.curb.height_m,
                crossfall=self.config.road.crossfall,
                crossfall_shape=self.config.road.crossfall_shape,
                longitudinal_grade=self.config.road.longitudinal_grade,
            )
        # 路面合成网格点数（不含缘石）；curb 关闭时等于 surface_estimate.point_count。
        road_point_count = surface_estimate.point_count - self.config.curb_grid_point_count
        if len(points) != road_point_count:
            raise RuntimeError(
                "Surface preflight mismatch: estimated "
                f"{road_point_count:,} points but generated {len(points):,}."
            )

        # 2.5 P1-1: LiDAR 扫描线密度重采样（可选）——与噪声/强度共用位姿。
        # 缘石已生成但未并入时，扫描在"路面+缘石"的临时合并云上做单次
        # 通过、再按索引区间拆回：缘石点由此流经与路面完全相同的扫描线/
        # 距离/入射角密度模型（同一次 rng 抽取、同一个 r_max 归一），而
        # 第 3 步病害仍在纯路面足迹上执行。curb 关闭时 scan_input 就是
        # points 本身、road_sel 恒等于 scan_idx，与历史路径逐位一致。
        # 观测体制关闭时整步跳过（无传感器即无扫描线密度模型）；若扫描
        # 开关仍为开启则发 warning 并在元数据记录 override，不报硬错
        # （scan 开关与 observation_model 分属两个配置对象）。
        scan_override: str | None = None
        if observation_off:
            if self.config.lidar_scan.enable:
                warnings.warn(
                    "observation_model='off': skipping LiDAR scan-line resampling; "
                    "lidar_scan.enable=True is overridden (recorded in metadata).",
                    stacklevel=2,
                )
                scan_override = "skipped_under_observation_off"
            scan_pattern_points = int(surface_estimate.point_count)
        elif self.config.lidar_scan.enable:
            n_road_pre_scan = points.shape[0]
            scan_input = (
                points
                if curb_points is None
                else np.vstack([points, curb_points])
            )
            scan_idx = resample_to_lidar_pattern(
                scan_input,
                scan_lines=self.config.lidar_scan.scan_lines,
                vertical_fov_deg=self.config.lidar_scan.vertical_fov_deg,
                scan_pattern=self.config.lidar_scan.scan_pattern,
                range_decay=self.config.lidar_scan.range_decay,
                incidence_angle_drop=self.config.lidar_scan.incidence_angle_drop,
                rng=rng,
                sensor_pose=self._sensor_pose(),
                line_sigma_m=self.config.lidar_scan.line_sigma_m,
            )
            if len(scan_idx) > 10:
                # scan_idx 来自 np.where，严格升序，可按单一阈值拆分。
                road_sel = scan_idx[scan_idx < n_road_pre_scan]
                points = points[road_sel]
                normals = normals[road_sel]
                if curb_points is not None:
                    curb_sel = (
                        scan_idx[scan_idx >= n_road_pre_scan] - n_road_pre_scan
                    )
                    curb_points = curb_points[curb_sel]
                    curb_normals = curb_normals[curb_sel]
            scan_pattern_points = int(
                len(points) + (0 if curb_points is None else curb_points.shape[0])
            )
        else:
            scan_pattern_points = int(
                len(points) + (0 if curb_points is None else curb_points.shape[0])
            )
        # 3. 标签初始化 + 病害选择 & 应用
        labels = np.zeros(points.shape[0], dtype=np.int64)

        if target_spec is None:
            diseases = self._select_diseases(rng, pavement_type)
        else:
            diseases = [(target_spec.disease_type, target_spec.severity)]
        # 按自然顺序排序（大尺度 → 小尺度）
        diseases.sort(key=lambda d: DISEASE_APPLY_ORDER.get(d[0], 99))

        # 分离 raveling（最后处理，不参与曲率计算）
        raveling_entry = None
        non_raveling_diseases = []
        for d in diseases:
            if d[0] == "raveling":
                raveling_entry = d
            else:
                non_raveling_diseases.append(d)

        # 机理相关放置先验：疲劳类病害（龟裂、纵缝、坑槽）集中在轮迹
        # 带；坑槽的主要成因链是 龟裂→水侵→剥落→坑槽，故若本场景有
        # 龟裂/块裂 patch，坑槽以 p=0.5 落在该 patch 内。旧实现所有
        # 病害独立均匀撒点，俯视图一眼可辨合成痕迹。
        planned_crack_type: str | None = None
        if any(d[0] == "crack" for d in non_raveling_diseases):
            planned_crack_type = (
                target_spec.variant
                if target_spec is not None and target_spec.disease_type == "crack"
                else str(rng.choice(self.config.crack.crack_types))
            )
        alligator_region: tuple[float, float] | None = None
        if planned_crack_type in ("alligator", "block"):
            alligator_region = (
                self._sample_wheelpath_x(rng, width),
                float(rng.uniform(0.8, max(length - 0.8, 0.9))),
            )
        pothole_cooccurs = alligator_region is not None and rng.random() < 0.5

        for disease_type, severity in non_raveling_diseases:
            seed = int(rng.integers(0, 2 ** 31))
            old_labels = labels.copy()

            if disease_type == "crack":
                crack_type = planned_crack_type or str(rng.choice(self.config.crack.crack_types))
                params: dict = {
                    "d_max": 0.010 if severity == "light" else 0.030,
                    "label_width_floor": grid_res,
                    # 裂缝形态/截断口径（primitives 侧消费；键名与
                    # config.crack 字段一一对应）
                    "perturb_mode": self.config.crack.perturb_mode,
                    "width_truncation": self.config.crack.width_truncation,
                    "area_lambda_mode": self.config.crack.area_lambda_mode,
                    "area_label_mode": self.config.crack.area_label_mode,
                }
                pending_scene.update(_pending_params_keys(add_crack, params))
                if crack_type == "longitudinal":
                    # 纵缝先验：轮迹带边缘（带中心 ±0.3m 摆动）
                    params["x_center"] = self._sample_wheelpath_x(rng, width) + float(
                        rng.normal(0.0, 0.15)
                    )
                elif alligator_region is not None:
                    params["region_center"] = alligator_region
                points, labels = add_crack(
                    points, labels,
                    crack_type=crack_type, severity=severity,
                    params=params, seed=seed,
                )

            elif disease_type == "pothole":
                if pothole_cooccurs and alligator_region is not None:
                    cx = float(np.clip(
                        rng.normal(alligator_region[0], 0.3), 0.5, width - 0.5
                    ))
                    cy = float(np.clip(
                        rng.normal(alligator_region[1], 0.6), 0.5, length - 0.5
                    ))
                else:
                    cx = self._sample_wheelpath_x(rng, width)
                    cy = float(rng.uniform(0.5, length - 0.5))
                # JTG 5210-2018 坑槽轻/重仅按深度 (25mm) 分级；平面半径
                # 与严重度解耦采样——旧实现按严重度硬切半径区间，使
                # 轻/重可被平面尺寸这一非规范线索 100% 分离。有界纵横
                # 比先验拒绝非物理 d/R 极端（config
                # pothole.aspect_ratio_range；None = 历史独立采样）。
                radius, depth = _sample_pothole_geometry(
                    rng, self.config.pothole, severity
                )
                edge_quality = float(rng.uniform(0.5, 1.0))
                pothole_params = {
                    # β/微坑/剥落环口径（primitives 侧消费；键名与
                    # config.pothole 字段一一对应）
                    "beta_mode": self.config.pothole.beta_mode,
                    "micropit_rate": self.config.pothole.micropit_rate,
                    "micropit_mode": self.config.pothole.micropit_mode,
                    "spall_band": tuple(self.config.pothole.spall_band),
                }
                pending_scene.update(_pending_params_keys(add_pothole, pothole_params))
                points, labels = add_pothole(
                    points, labels,
                    center=(cx, cy), radius=radius, depth=depth,
                    edge_quality=edge_quality, severity=severity,
                    seed=seed,
                    beta_range=self.config.pothole.beta_range,
                    params=pothole_params,
                )

            elif disease_type == "depression":
                cx, cy, radius = _sample_depression_geometry(
                    rng, self.config.depression, width, length
                )
                if severity == "light":
                    depth = float(rng.uniform(*self.config.depression.depth_light))
                else:
                    depth = float(rng.uniform(*self.config.depression.depth_severe))
                # label_radius_mode 需基元组为 add_depression 增加 params
                # 形参（fix#13 基元侧）；请求值等于基元当前行为
                # （legacy-1pct）时静默省略，否则披露待合入。
                dep_kwargs, dep_pending = _bridge_depression_params(
                    self.config.depression.label_radius_mode
                )
                pending_scene.update(dep_pending)
                points, labels = add_depression(
                    points, labels,
                    center=(cx, cy), radius=radius,
                    depth=depth, severity=severity, seed=seed,
                    **dep_kwargs,
                )

            elif disease_type == "rutting":
                # 车辙中心走轮迹带先验（与裂缝/坑槽一致；7m 双车道口径
                # 仅建模内侧轮迹对——外侧两条轮迹不被疲劳病害命中的
                # 现假设显性化）。'fixed' = 历史恒位于道路中线。
                if self.config.rutting.center_mode == "wheelpath":
                    center_line = self._sample_wheelpath_x(rng, width)
                else:
                    center_line = width / 2.0
                wheel_sep = self.config.rutting.wheel_separation
                if severity == "light":
                    # JTG 车辙起判阈值 10mm：5-10mm 不构成车辙病害。
                    depth = float(rng.uniform(0.010, self.config.rutting.max_depth_light))
                else:
                    depth = float(rng.uniform(
                        self.config.rutting.severe_min_depth,
                        self.config.rutting.max_depth_severe,
                    ))
                if self.config.rutting.peak_normalize:
                    # 名义深度按 1/(1+ε) 归一：primitives 的正弦调制使
                    # 实现峰值 = d·(1+ε)·(track≤1)，除以 (1+ε) 后实现
                    # 峰值 ≤ 名义深度（light ≤ 15mm、severe > 15mm，
                    # 不再越过 JTG 档位分界）。
                    depth = depth / (1.0 + self.config.rutting.modulation_epsilon)
                rut_width = self.config.rutting.rut_width
                rutting_params = {
                    "modulation_epsilon": self.config.rutting.modulation_epsilon,
                }
                pending_scene.update(_pending_params_keys(add_rutting, rutting_params))
                points, labels = add_rutting(
                    points, labels,
                    center_line=center_line,
                    wheel_separation=wheel_sep,
                    depth=depth, width=rut_width,
                    severity=severity, seed=seed,
                    params=rutting_params,
                )

            elif disease_type == "corrugation":
                direction = str(rng.choice(["longitudinal", "transverse"]))
                wavelength = float(rng.uniform(*self.config.corrugation.wavelength_range))
                if severity == "light":
                    amplitude = float(rng.uniform(*self.config.corrugation.amplitude_light))
                else:
                    amplitude = float(rng.uniform(*self.config.corrugation.amplitude_severe))
                points, labels = add_corrugation(
                    points, labels,
                    direction=direction, wavelength=wavelength,
                    amplitude=amplitude, severity=severity, seed=seed,
                )

            elif disease_type == "bleeding":
                if self.config.bleeding.geometry_mode == "patch":
                    # 斑块口径：轮迹带中心、有限 y 长度、面积占比随机、
                    # logistic 平滑边界（掩码逐点确定性，云无关）。
                    patch = _bleeding_patch_params(
                        rng, self.config.bleeding, width, length, salt=seed
                    )
                    region_mask = _bleeding_region_mask(points, patch)
                else:
                    # legacy-strip：恒定 40% 面积、贯穿全路长、硬二值
                    # 边界的竖直条带（历史口径）。
                    lane_center = rng.uniform(0.3, 0.7) * width
                    region_mask = (
                        np.abs(points[:, 0] - lane_center)
                        < self.config.bleeding.region_width * width / 2.0
                    )
                labels = add_bleeding(
                    points, labels,
                    region_mask=region_mask, seed=seed,
                )

            elif disease_type == "patching":
                points, labels = self._apply_patching(
                    points,
                    labels,
                    patch_label=20,
                    width=width,
                    length=length,
                    rng=rng,
                )

            elif disease_type == "concrete_damage":
                damage_type = (
                    target_spec.variant
                    if target_spec is not None
                    and target_spec.disease_type == "concrete_damage"
                    else str(rng.choice(CONCRETE_DAMAGE_TYPES))
                )
                if damage_type == "patching":
                    points, labels = self._apply_patching(
                        points,
                        labels,
                        patch_label=37,
                        width=width,
                        length=length,
                        rng=rng,
                    )
                else:
                    sev = "-" if damage_type in CONCRETE_NO_SEVERITY else severity
                    params = {
                        "slab_length": self.config.concrete_damage.slab_length,
                        "slab_width": self.config.concrete_damage.slab_width,
                        "joint_width": self.config.concrete_damage.joint_width,
                        "x_offset": self.config.concrete_damage.x_offset,
                        "y_offset": self.config.concrete_damage.y_offset,
                        "label_width_floor": grid_res,
                        # 水泥病害几何口径（primitives 侧消费；键名与
                        # config.concrete_damage 字段一一对应；聚合回退
                        # 为 CLI --concrete-legacy）
                        "crack_in_slab": self.config.concrete_damage.crack_in_slab,
                        "fault_scope": self.config.concrete_damage.fault_scope,
                        "edge_spall_mode": self.config.concrete_damage.edge_spall_mode,
                        "shatter_mode": self.config.concrete_damage.shatter_mode,
                    }
                    pending_scene.update(
                        _pending_params_keys(add_concrete_damage, params)
                    )
                    points, labels = add_concrete_damage(
                        points, labels,
                        damage_type=damage_type, severity=sev,
                        params=params, seed=seed,
                    )

            # P1-3 + Q3 向量化：仅在优先级更高时保留新标签
            changed_mask = labels != old_labels
            if np.any(changed_mask):
                max_label = max(LABEL_PRIORITY.keys()) + 1
                priority_lut = np.zeros(max_label, dtype=np.int32)
                for k, v in LABEL_PRIORITY.items():
                    priority_lut[k] = v
                old_prio = priority_lut[old_labels[changed_mask]]
                new_prio = priority_lut[labels[changed_mask]]
                restore = old_prio > new_prio
                changed_indices = np.where(changed_mask)[0]
                labels[changed_indices[restore]] = old_labels[changed_indices[restore]]

        # 3.5 缘石并入：在病害之后、曲率之前。缘石点自带背景标签 0，不占用
        # 任何病害标签。road_mask 标记路面足迹（curb 关闭时全 True），供第 5
        # 步松散区域与第 6 步标签膨胀排除非路面点——这两条路径在 curb 关闭时
        # 与历史逐位一致（掩码全 True 不改变任何取值）。
        if curb_points is not None:
            n_curb = curb_points.shape[0]
            # 以并入前的实际路面点数划线，而不是静态网格预算：第 2.5 步
            # 扫描重采样会随机抽稀路面点，静态值既会让守卫必然失配，又会
            # 把（索引超出实际路面点数的）缘石点误标为路面足迹。
            n_road_current = points.shape[0]
            points = np.vstack([points, curb_points])
            normals = np.vstack([normals, curb_normals])
            labels = np.concatenate([labels, np.zeros(n_curb, dtype=labels.dtype)])
            road_mask = np.arange(points.shape[0]) < n_road_current
            # 自洽守卫：扫描关闭时病害步骤不增减点数（第 4 步曲率快速路径
            # 历来依赖同一等式），并入前必须仍等于路面网格预算；扫描开启
            # 时该点数是随机的，无静态期望——网格健全性已由第 2.4 步
            # preflight 守卫覆盖。
            if (
                not self.config.lidar_scan.enable
                and n_road_current != road_point_count
            ):
                raise RuntimeError(
                    "Curb merge mismatch: road cloud holds "
                    f"{n_road_current:,} points before merge, expected the "
                    f"road grid's {road_point_count:,} (scan resampling "
                    "disabled)."
                )
        else:
            road_mask = np.ones(points.shape[0], dtype=bool)

        # 4. Curvature for sensor edge mixing.  The full-resolution surface is
        # still a regular grid unless scan-line resampling was enabled.  A
        # finite-difference fast path avoids millions of individual KDTree
        # queries for 2-5 mm synthesis while retaining the irregular-cloud
        # fallback for scan-pattern data.
        # 观测体制关闭时整步跳过：曲率的唯一消费者是噪声阶段的边缘混合
        # （第 8 步），最终特征由 features.py 独立重算 PCA 曲率，不消费
        # 该量。
        if observation_off:
            curvature = None
            curvature_method = "skipped_observation_off"
        elif (
            not self.config.lidar_scan.enable
            and len(points) == surface_estimate.point_count
            and self.config.curb_grid_point_count == 0
        ):
            curvature = _compute_grid_curvature(
                points[:, 2].reshape(surface_estimate.grid_shape),
                grid_res,
            ).reshape(-1).astype(np.float32)
            curvature_method = "regular_grid_finite_difference"
        else:
            curvature = _compute_kdtree_curvature(points, k_neighbors=20)
            curvature_method = "kdtree_pca"

        # 5. 应用松散 (raveling) — 产生 NaN 点
        if raveling_entry is not None:
            disease_type, severity = raveling_entry
            seed = int(rng.integers(0, 2 ** 31))
            x_frac_min = float(rng.uniform(0.05, 0.35))
            x_frac_max = x_frac_min + float(rng.uniform(0.2, 0.5))
            y_frac_min = float(rng.uniform(0.05, 0.35))
            y_frac_max = y_frac_min + float(rng.uniform(0.2, 0.5))
            # 区域从**路面足迹**推导（curb 关闭时 road_mask 全 True，与历史
            # 逐位一致）：否则缘石会加宽 bbox，使既有病害几何随 curb 开关漂移，
            # 并可能把松散区域推到缘石上。
            road_pts = points[road_mask]
            x_min_r = np.min(road_pts[:, 0]) + x_frac_min * (np.max(road_pts[:, 0]) - np.min(road_pts[:, 0]))
            x_max_r = np.min(road_pts[:, 0]) + x_frac_max * (np.max(road_pts[:, 0]) - np.min(road_pts[:, 0]))
            y_min_r = np.min(road_pts[:, 1]) + y_frac_min * (np.max(road_pts[:, 1]) - np.min(road_pts[:, 1]))
            y_max_r = np.min(road_pts[:, 1]) + y_frac_max * (np.max(road_pts[:, 1]) - np.min(road_pts[:, 1]))
            region_mask = (
                (points[:, 0] >= x_min_r) & (points[:, 0] <= x_max_r)
                & (points[:, 1] >= y_min_r) & (points[:, 1] <= y_max_r)
            )
            raveling_params = {
                # 接线 config.raveling（默认值=历史硬编码，行为逐位不变；
                # 不传 deterministic_seed 时仍走历史 rng 路径）。
                "removal_ratio": (
                    self.config.raveling.point_removal_ratio_light
                    if severity == "light"
                    else self.config.raveling.point_removal_ratio_severe
                ),
                "erosion_depth": tuple(self.config.raveling.erosion_depth),
            }
            pending_scene.update(_pending_params_keys(add_raveling, raveling_params))
            points, labels = add_raveling(
                points, labels,
                region_mask=region_mask, severity=severity, seed=seed,
                params=raveling_params,
            )

        # 6. 松散标签膨胀 + NaN 点过滤
        # P0-3: 在移除 NaN 点之前，将松散标签膨胀到 NaN 点的邻域，
        # 确保噪声仿真后落在原松散区域的点能获得正确的松散标签。
        raveling_labels_mask = (labels == 11) | (labels == 12)
        valid_mask = ~np.any(np.isnan(points), axis=1)

        # 松散标签膨胀：使用网格邻域将松散标签传播到附近的背景点
        # 策略：对每个非 NaN 的松散点，将其周围 grid_res*2 半径内的
        # 背景（label=0）点标记为相同的松散标签
        raveling_valid_mask = raveling_labels_mask & valid_mask
        if np.any(raveling_valid_mask):
            raveling_xy = points[raveling_valid_mask][:, :2]
            raveling_severity = labels[raveling_valid_mask]
            # 膨胀目标限定在路面足迹内：缘石点不得获得松散标签（设计保证
            # 缘石恒为背景标签 0）。curb 关闭时 road_mask 全 True，无影响。
            background_valid_mask = valid_mask & ~raveling_labels_mask & road_mask
            if np.any(background_valid_mask) and raveling_xy.shape[0] > 0:
                from scipy.spatial import KDTree as _KDTree
                bg_xy = points[background_valid_mask][:, :2]
                bg_labels = labels[background_valid_mask]
                bg_global_idx = np.where(background_valid_mask)[0]
                tree_dilate = _KDTree(raveling_xy)
                dilate_radius = grid_res * 3.0
                nn_dist, nn_idx = tree_dilate.query(bg_xy, k=1)
                in_dilate = nn_dist <= dilate_radius
                for i in np.where(in_dilate)[0]:
                    if bg_labels[i] == 0:
                        labels[bg_global_idx[i]] = raveling_severity[nn_idx[i]]

        if np.any(valid_mask):
            points = points[valid_mask]
            labels = labels[valid_mask]
            normals = normals[valid_mask]
            if curvature is not None:
                curvature = curvature[valid_mask]
        else:
            raise RuntimeError("All points removed by raveling — no valid geometry remains.")

        if points.shape[0] == 0:
            raise RuntimeError("Empty point cloud after NaN filtering.")

        # 6.5 几何遮挡：深窄凹陷（裂缝槽）内部点按开口半宽/深度/入射角
        # 概率丢弃——真实窄裂缝是"点缺失的线"，不是被完整采样的深沟。
        # 这是传感器可见性现象，观测体制关闭时跳过（元数据
        # occlusion_policy="off_with_observation_model"）；毫米网格下裂缝
        # 是真实几何凹槽，遮挡丢点会把槽体内部挖空。 depression_depth
        # 仍需计算：它同时供强度阶段的辐射遮蔽（表面属性，与视点无关）
        # 使用。retain-all 的既定用途是密度上限/几何真值参照，真值参照
        # 不应被观测效应删点。
        sensor_origin = self._sensor_pose()
        depression_depth = local_depression_depth(points)
        if not observation_off:
            occl_keep = geometric_occlusion_keep_mask(
                points, sensor_origin, depression_depth, rng
            )
            if not np.all(occl_keep):
                if not np.any(occl_keep):
                    raise RuntimeError("Geometric occlusion removed every point.")
                points = points[occl_keep]
                labels = labels[occl_keep]
                normals = normals[occl_keep]
                curvature = curvature[occl_keep]
                depression_depth = depression_depth[occl_keep]

        # 7. 辐射强度计算（在松散标签应用之后）：I ∝ ρ·cosθ/R²，
        # 同时得到接收功率代理，供 σ_r 与丢点共用（统一辐射预算）。
        intensity, received_power = self._compute_intensity(
            points, normals, labels, pavement_type, sensor_origin, rng,
            depression_depth=depression_depth,
        )

        # 8. LiDAR 噪声仿真 + 9. 最近邻传递标签/特征/法向量 —— 观测体制
        # 关闭时整体跳过：没有传感器就没有测距噪声/丢点，直出点即最终点
        # （noisy_points := points），标签/特征/法向量保持直出值，不做
        # 最近邻回传。
        if observation_off:
            noisy_points = points
            sensor_points_before_output_sampling = int(len(noisy_points))
        else:
            # 保存噪声前状态以传递标签/特征
            pre_points = points.copy()
            pre_labels = labels.copy()
            pre_intensity = intensity.copy()
            pre_curvature = curvature.copy()
            pre_normals = normals.copy()

            # 逐点测距噪声：σ_r ∝ 1/√SNR ∝ 1/√P_recv（限幅 [0.5, 3]×），
            # 暗、掠射、远处的点噪声更大。
            sigma_base = self.config.lidar_noise.distance_noise_std
            if sigma_base > 0.0:
                power_ref = float(np.median(received_power))
                sigma_r = sigma_base * np.clip(
                    np.sqrt(power_ref / np.maximum(received_power, 1e-12)), 0.5, 3.0
                )
            else:
                sigma_r = None

            noisy_points = simulate_lidar_noise(
                points,
                distance_noise_std=self.config.lidar_noise.distance_noise_std,
                dropout_rate=self.config.lidar_noise.dropout_rate,
                angular_jitter_deg=self.config.lidar_noise.angular_jitter_deg,
                seed=int(rng.integers(0, 2 ** 31)),
                enable_edge_mixing=self.config.lidar_noise.enable_edge_mixing,
                mixed_pixel_prob=self.config.lidar_noise.mixed_pixel_prob,
                curvature=pre_curvature,  # P2-3: 仅高曲率边缘点混合
                curvature_threshold=0.5,
                sensor_origin=sensor_origin,
                sigma_r=sigma_r,
                drop_weight=received_power,
            )

            if noisy_points.shape[0] == 0:
                raise RuntimeError("All points dropped during LiDAR noise simulation.")
            sensor_points_before_output_sampling = int(len(noisy_points))

            # 9. 最近邻传递标签/特征/法向量（带距离阈值保护）
            # P0-4: 增加距离阈值，防止标签溢出/侵蚀
            from scipy.spatial import KDTree
            tree = KDTree(pre_points)
            nn_dist, nn_idx = tree.query(noisy_points, k=1)

            labels = pre_labels[nn_idx]
            intensity = pre_intensity[nn_idx]
            curvature = pre_curvature[nn_idx]
            normals = pre_normals[nn_idx]

            # P0-4 距离阈值保护：超过 3σ 距离的点回退到背景标签
            # 防止裂缝边缘噪声点携带错误标签到远处（σ 取逐点噪声均值）
            if sigma_r is not None:
                max_transfer_dist = float(np.mean(sigma_r)) * 3.0
            else:
                max_transfer_dist = self.config.lidar_noise.distance_noise_std * 3.0
            if max_transfer_dist > 0.0:
                uncertain_mask = nn_dist > max_transfer_dist
            else:
                # The spherical-coordinate round trip has machine-scale error even
                # when all requested noise terms are zero. It is not uncertainty.
                uncertain_mask = np.zeros_like(nn_dist, dtype=bool)
            # 对不确定点：仅保留非裂缝标签（标签 > 0 且 <= 8 的裂缝标签降为背景）
            # 但保留松散、车辙、沉陷等大面积病害标签
            crack_mask = (labels >= 1) & (labels <= 8)  # Q1: 布尔掩码替代 set + for
            restore_mask = uncertain_mask & crack_mask
            labels[restore_mask] = 0

        # 10. Sensor/output sampling.  This is separate from the downstream
        # model target point count, which is recorded but never applied here.
        # protection_available stays False on the density_voxel path: voxel
        # majority voting offers no minimum-survival guarantee, and the audit
        # record must not imply one.
        protection_info: dict = {
            "protected_label": -1 if target_label is None else int(target_label),
            "protected_min_points": int(self.config.target_label_min_output_points),
            "protection_available": False,
            "protection_applied": False,
        }
        if observation_off:
            # 观测体制关闭：直出全部表面点（retain-all 语义，
            # noisy_points := points），num_points/target_density 不参与。
            points_final = noisy_points
            labels_final = labels
            intensity_final = intensity
            normals_final = normals
            sensor_output_sampling_method = "none_full_surface"
            protection_info["protection_available"] = True
        elif self.config.retain_all_surface_points:
            # Raw-surface mode: keep every point that survived the sensor
            # pipeline (noise + occlusion), no output sampling.  Used for
            # density-upper-bound ablation references (~1.4M pts at 5 mm).
            points_final = noisy_points
            labels_final = labels
            intensity_final = intensity
            curvature_final = curvature
            normals_final = normals
            sensor_output_sampling_method = "none_full_surface"
            protection_info["protection_available"] = True
        elif self.config.target_density is not None and self.config.target_density > 0:
            road_area = width * length
            target_count = int(road_area * self.config.target_density)
            # 体素大小逼近目标点数：N_voxels ≈ area / voxel_size²
            effective_voxel_size = max(
                np.sqrt(road_area / max(target_count, 1)),
                grid_res * 1.5
            )
            points_final, labels_final, intensity_final, curvature_final, normals_final = (
                self._voxel_downsample(
                    noisy_points, labels, intensity, curvature, normals,
                    voxel_size=effective_voxel_size,
                )
            )
            sensor_output_sampling_method = "density_voxel"
        else:
            # 回退到旧的 num_points 硬重采样
            (
                points_final,
                labels_final,
                intensity_final,
                curvature_final,
                normals_final,
                protection_info,
            ) = self._resample_to_target(
                noisy_points, labels, intensity, curvature, normals,
                self.config.num_points, rng,
                protected_label=target_label,
                protected_min_points=self.config.target_label_min_output_points,
            )
            sensor_output_sampling_method = "fixed_count_random"

        physical_bounds_min_m = points_final.min(axis=0).astype(np.float64)
        physical_bounds_max_m = points_final.max(axis=0).astype(np.float64)

        # 11. 坐标归一化
        coordinate_center = np.zeros(3, dtype=np.float32)
        coordinate_scale = 1.0
        coordinates_normalized = False
        if self.config.normalize:
            points_final, coordinate_center, coordinate_scale = self._normalize_with_metadata(
                points_final
            )
            coordinates_normalized = True

        # 12. Inference-safe features: all channels are re-derived from the
        # exact final coordinate array written to disk.  In particular, no
        # label-derived crack mask or distance is allowed to enter the model
        # input.  The geometric channels are translation/scale invariant.
        points_final = points_final.astype(np.float32, copy=False)
        feats = compute_observable_features(
            points_final,
            intensity_final,
            k_neighbors=DEFAULT_GEOMETRY_K_NEIGHBORS,
        )

        resolution_metadata = self.config.resolution_metadata(
            actual_output_points=len(points_final)
        )
        if pending_scene:
            # 本场景实际遇到的待合入键（场景级准确，不携带其他场景的登
            # 记）：读取方据此知道哪些新口径尚未生效（基元侧改动未合
            # 入）。full-legacy 回退运行下该集合为空、键不出现。
            resolution_metadata["pending_primitive_upgrades"] = sorted(
                pending_scene
            )
        resolution_metadata["surface_grid"]["curvature_method"] = curvature_method
        resolution_metadata["sensor_output"].update(
            {
                "scan_pattern_points": scan_pattern_points,
                "points_before_output_sampling": sensor_points_before_output_sampling,
                "sampling_method": sensor_output_sampling_method,
                "output_to_surface_point_ratio": (
                    len(points_final) / surface_estimate.point_count
                ),
                "fixed_count_upsampling_used": bool(
                    sensor_output_sampling_method == "fixed_count_random"
                    and len(points_final) > sensor_points_before_output_sampling
                ),
            }
        )
        if observation_off:
            resolution_metadata["sensor_output"].update(
                {
                    "mode": "full_surface_observation_off",
                    "pipeline_steps_skipped": [
                        "scan_resample",
                        "curvature",
                        "geometric_occlusion",
                        "lidar_noise",
                        "nn_label_transfer",
                        "output_sampling",
                    ],
                    "occlusion_policy": "off_with_observation_model",
                    # num_points/target_density 的请求值已在
                    # requested_points / requested_density_points_per_m2
                    # 中记录；此处标记它们在本体制下被忽略。
                    "ignored_under_observation_off": True,
                }
            )
            if scan_override is not None:
                resolution_metadata["sensor_output"]["scan_override"] = scan_override
        resolution_metadata["sensor_output"]["sensor_pose_m"] = [
            float(v) for v in sensor_origin
        ]
        resolution_metadata["sensor_output"]["intensity_model"] = (
            "radiometric.v2: rho(x,y)*shading*cos(theta)/R^2, "
            "lognormal albedo field (corr "
            f"{self.config.radiometric.albedo_corr_len_m:g} m, sigma_ln "
            f"{self.config.radiometric.albedo_sigma_ln:g}), "
            f"Gamma speckle M={self.config.radiometric.speckle_m:g}, "
            "8-bit quantization; sigma_r and dropout "
            "share the same received-power proxy"
        )
        # Audit trail for controlled scenes: record how many target-label
        # points actually survived output sampling and whether the minimum
        # survival protection had to intervene.
        if target_label is not None:
            target_output_points = int(np.count_nonzero(labels_final == target_label))
            protection_info["target_label_output_points"] = target_output_points
            protection_info["target_label_output_ratio"] = (
                target_output_points / max(len(labels_final), 1)
            )
        resolution_metadata["sensor_output"]["target_label_protection"] = protection_info
        resolution_metadata["physical_coordinates"] = {
            "unit": "m",
            "bounds_min_m": physical_bounds_min_m.tolist(),
            "bounds_max_m": physical_bounds_max_m.tolist(),
            "stored_coordinates_normalized": coordinates_normalized,
            "normalization_is_reversible": True,
        }

        return {
            "points": points_final,
            "labels": labels_final.astype(np.int64),
            "feats": feats,
            "normals": normals_final.astype(np.float32),
            "pavement_type": pavement_type,
            # 该场景实际生效的 ISO 8608 等级，无论混合开关总是写入（与
            # pavement_type 同型的 0 维 unicode 标量落盘）：混合开启时为
            # 逐场景抽取结果，关闭时为配置的单一回退等级。
            "roughness_class": roughness_actual,
            "target_label": -1 if target_label is None else int(target_label),
            "feature_schema": OBSERVABLE_FEATURE_SCHEMA,
            "feature_names": np.asarray(OBSERVABLE_FEATURE_NAMES),
            "feature_k_neighbors": int(DEFAULT_GEOMETRY_K_NEIGHBORS),
            "coordinate_center": coordinate_center.astype(np.float32),
            "coordinate_scale": float(coordinate_scale),
            "coordinates_normalized": coordinates_normalized,
            "resolution_metadata": resolution_metadata,
            "resolution_contract": resolution_metadata["contract_version"],
            "surface_grid_spacing_m": float(grid_res),
            "surface_grid_shape": np.asarray(surface_estimate.grid_shape, dtype=np.int64),
            "surface_grid_point_count": int(surface_estimate.point_count),
            "sensor_output_point_count": int(len(points_final)),
            "observation_model": str(self.config.observation_model),
            "segment_count": int(surface_estimate.segment_count),
            "segment_length_m": float(surface_estimate.segment_length_m),
            "segment_overlap_m": float(surface_estimate.overlap_m),
            "model_target_points": (
                -1
                if self.config.model_target_points is None
                else int(self.config.model_target_points)
            ),
        }

    # ------------------------------------------------------------------
    # 分段条带运行时（observation_model='off'）
    # ------------------------------------------------------------------

    def _generate_scene_segmented_off(
        self,
        idx: int,
        target_spec: TargetLabelSpec | None,
        target_label: int | None,
        surface_estimate,
        rng: np.random.Generator,
    ) -> dict:
        """observation_model='off' 的分段条带运行时。

        谱分解与病害计划/施加见 segmentation_design：宏观长波整条带
        全局一次合成（接缝精确连续）、短波纹理逐段 FFT + RRC 跨段混合、
        het 调制全局 1-D、设计几何细网格解析叠加；病害在计划阶段一次性
        抽完所有 rng（消耗顺序与单块路径一致），施加阶段按段在扩展云
        （拥有行 + halo）上执行、只写回拥有行——云无关参数经 params
        传入，保证缝两侧逐位一致。观测管线（扫描/遮挡/噪声/最近邻
        传递/输出采样）整体跳过；强度走两遍扫（逐段 received_power →
        整条带 p99 AGC → 逐段斑点/量化），特征用段+halo 的 cKDTree
        分块重算，坐标归一化对预装配数组流式执行。
        """
        config = self.config
        grid_res = config.road.grid_res
        width = config.road.width
        length = config.road.length
        nx, ny_total = surface_estimate.grid_shape

        # 1. 路面类型（与单块路径同一 rng 消耗顺序）
        pavement_type = (
            target_spec.pavement_type
            if target_spec is not None
            else self._select_pavement_type(rng)
        )
        # 1.5 粗糙度等级抽取
        if config.road.roughness_mix is not None:
            mix = config.road.roughness_mix
            classes, weights = zip(*sorted(mix.items()), strict=False)
            p = np.asarray(weights, dtype=np.float64)
            p = p / p.sum()
            roughness_actual = str(rng.choice(list(classes), p=p))
        else:
            roughness_actual = config.road.roughness_class

        # 2. 表面种子（位置与单块路径一致；宏观/纹理随机流从它派生，
        #    不消耗主 rng）
        surface_seed = int(rng.integers(0, 2 ** 31))

        # 3. 病害选择（与单块路径一致）
        if target_spec is None:
            diseases = self._select_diseases(rng, pavement_type)
        else:
            diseases = [(target_spec.disease_type, target_spec.severity)]
        diseases.sort(key=lambda d: DISEASE_APPLY_ORDER.get(d[0], 99))

        raveling_entry = None
        non_raveling_diseases = []
        for d in diseases:
            if d[0] == "raveling":
                raveling_entry = d
            else:
                non_raveling_diseases.append(d)

        planned_crack_type: str | None = None
        if any(d[0] == "crack" for d in non_raveling_diseases):
            planned_crack_type = (
                target_spec.variant
                if target_spec is not None and target_spec.disease_type == "crack"
                else str(rng.choice(config.crack.crack_types))
            )
        alligator_region: tuple[float, float] | None = None
        if planned_crack_type in ("alligator", "block"):
            alligator_region = (
                self._sample_wheelpath_x(rng, width),
                float(rng.uniform(0.8, max(length - 0.8, 0.9))),
            )
        pothole_cooccurs = alligator_region is not None and rng.random() < 0.5

        # 4. 病害计划：主 rng 消耗顺序与单块循环一致；需要跨段一致的
        #    内部抽取用 default_rng(seed) 在计划里预抽并经 params 传入
        #    （设计稿的云无关规则）。坑槽不需要预抽：影响半径 ≤0.45 m
        #    < halo，各相交段的扩展云包含**同一份**按同序排布的环带
        #    网格点，同种子下 rng.choice 选出同一点集。
        # 场景级待合入键集合（与单块路径同语义；元数据只披露本场景遇到
        # 的键，legacy 等价值请求静默）。
        pending_scene: set[str] = set()
        x_max_grid = (nx - 1) * grid_res
        y_max_grid = (ny_total - 1) * grid_res
        x_grid_global = np.arange(0.0, width, grid_res)
        y_grid_global = np.arange(0.0, length, grid_res)
        plan: list[dict] = []
        for disease_type, severity in non_raveling_diseases:
            seed = int(rng.integers(0, 2 ** 31))
            rd = np.random.default_rng(seed)
            entry: dict = {
                "type": disease_type,
                "severity": severity,
                "seed": seed,
                "params": {},
                "args": {},
            }
            params = entry["params"]

            if disease_type == "crack":
                crack_type = planned_crack_type or "longitudinal"
                entry["crack_type"] = crack_type
                params.update(
                    {
                        "d_max": 0.010 if severity == "light" else 0.030,
                        "label_width_floor": grid_res,
                        "num_curve_samples": 400,
                        # 裂缝形态/截断口径（与单块路径同键）
                        "perturb_mode": config.crack.perturb_mode,
                        "width_truncation": config.crack.width_truncation,
                        "area_lambda_mode": config.crack.area_lambda_mode,
                        "area_label_mode": config.crack.area_label_mode,
                    }
                )
                pending_scene.update(_pending_params_keys(add_crack, params))
                if crack_type == "longitudinal":
                    x_center = self._sample_wheelpath_x(rng, width) + float(
                        rng.normal(0.0, 0.15)
                    )
                    params["x_center"] = x_center
                    y_vals = np.linspace(
                        0.1 * y_max_grid, y_max_grid - 0.1 * y_max_grid, 4
                    )
                    offsets = rd.normal(0, 0.05 * x_max_grid, 4)
                    offsets[0] = offsets[0] * 0.5
                    offsets[-1] = offsets[-1] * 0.5
                    params["bezier_control_points"] = np.stack(
                        [x_center + np.full(4, offsets[0]) + offsets, y_vals], axis=1
                    )
                elif crack_type == "transverse":
                    x_vals = np.linspace(
                        0.1 * x_max_grid, x_max_grid - 0.1 * x_max_grid, 4
                    )
                    y_center = 0.5 * y_max_grid
                    offsets = rd.normal(0, 0.05 * y_max_grid, 4)
                    offsets[0] = offsets[0] * 0.5
                    offsets[-1] = offsets[-1] * 0.5
                    params["bezier_control_points"] = np.stack(
                        [x_vals, y_center + np.full(4, offsets[0]) + offsets], axis=1
                    )
                elif alligator_region is not None:
                    params["region_center"] = alligator_region
                if crack_type in ("alligator", "block"):
                    # 区域尺寸与中心的内部抽取在计划里按全局 bbox 复现，
                    # 裁剪成最终 patch 边界经 region_bounds 传入。
                    if crack_type == "alligator":
                        region_w = float(rd.uniform(0.6, 1.2))
                        region_l = float(rd.uniform(1.0, 4.0))
                    else:
                        region_w = float(rd.uniform(1.5, 3.0))
                        region_l = float(rd.uniform(2.0, 4.5))
                    rc = (
                        alligator_region
                        if alligator_region is not None
                        else (
                            float(rd.uniform(0.0, x_max_grid)),
                            float(rd.uniform(0.0, y_max_grid)),
                        )
                    )
                    block_size = (
                        (0.2 if severity == "severe" else 0.5)
                        if crack_type == "alligator"
                        else (1.0 if severity == "light" else 0.6)
                    )
                    x0 = max(0.0, rc[0] - region_w / 2.0)
                    x1 = min(x_max_grid, rc[0] + region_w / 2.0)
                    y0 = max(0.0, rc[1] - region_l / 2.0)
                    y1 = min(y_max_grid, rc[1] + region_l / 2.0)
                    if x1 - x0 < block_size or y1 - y0 < block_size:
                        x0 = max(0.0, x1 - max(region_w, 2 * block_size))
                        x1 = min(x_max_grid, x0 + max(region_w, 2 * block_size))
                        y0 = max(0.0, y1 - max(region_l, 2 * block_size))
                        y1 = min(y_max_grid, y0 + max(region_l, 2 * block_size))
                    params["region_bounds"] = (x0, x1, y0, y1)
                if crack_type in ("longitudinal", "transverse"):
                    # 缝宽剖面：与基元相同的对数正态 + JTG 截断，固定
                    # 400 个采样（num_curve_samples 的饱和值）。
                    if severity == "light":
                        width_mean, width_std = 0.0018, 0.25
                    else:
                        width_mean, width_std = 0.006, 0.3
                    profile = rd.lognormal(
                        mean=np.log(width_mean), sigma=width_std, size=400
                    )
                    if severity == "light":
                        profile = np.minimum(profile, 0.003 - 1e-4)
                    else:
                        profile = np.maximum(profile, 0.003 + 2e-4)
                    params["width_profile"] = profile

            elif disease_type == "pothole":
                if pothole_cooccurs and alligator_region is not None:
                    cx = float(np.clip(
                        rng.normal(alligator_region[0], 0.3), 0.5, width - 0.5
                    ))
                    cy = float(np.clip(
                        rng.normal(alligator_region[1], 0.6), 0.5, length - 0.5
                    ))
                else:
                    cx = self._sample_wheelpath_x(rng, width)
                    cy = float(rng.uniform(0.5, length - 0.5))
                radius, depth = _sample_pothole_geometry(rng, config.pothole, severity)
                edge_quality = float(rng.uniform(0.5, 1.0))
                entry["args"] = {
                    "center": (cx, cy),
                    "radius": radius,
                    "depth": depth,
                    "edge_quality": edge_quality,
                    "beta_range": config.pothole.beta_range,
                }
                # 云无关：坑缘剥落用坐标哈希（盐=病害种子），微坑参数用
                # 专用随机流。否则 rng.choice(edge_idx) 的消耗随扩展域
                # 变化，缝两侧剥落/微坑不一致（半径大、锚点近缝时扩展域
                # 截断剥落环）。
                params["deterministic_seed"] = seed
                params["pit_seed"] = int(rd.integers(0, 2 ** 31))
                # β/微坑/剥落环口径（与单块路径同键）
                params["beta_mode"] = config.pothole.beta_mode
                params["micropit_rate"] = config.pothole.micropit_rate
                params["micropit_mode"] = config.pothole.micropit_mode
                params["spall_band"] = tuple(config.pothole.spall_band)
                pending_scene.update(_pending_params_keys(add_pothole, params))

            elif disease_type == "depression":
                cx, cy, radius = _sample_depression_geometry(
                    rng, config.depression, width, length
                )
                if severity == "light":
                    depth = float(rng.uniform(*config.depression.depth_light))
                else:
                    depth = float(rng.uniform(*config.depression.depth_severe))
                entry["args"] = {
                    "center": (cx, cy), "radius": radius, "depth": depth,
                }
                # 标签半径口径在施加阶段经 _bridge_depression_params 下发
                # （请求值等于基元当前行为时静默，否则记入场景待合入集合）。
                entry["label_radius_mode"] = config.depression.label_radius_mode

            elif disease_type == "rutting":
                # 车辙中心走轮迹带先验（与单块路径同口径、同 rng 消耗
                # 顺序：先中心后深度）；'fixed' = 历史恒位于道路中线。
                if config.rutting.center_mode == "wheelpath":
                    center_line = self._sample_wheelpath_x(rng, width)
                else:
                    center_line = width / 2.0
                if severity == "light":
                    depth = float(rng.uniform(0.010, config.rutting.max_depth_light))
                else:
                    depth = float(rng.uniform(
                        config.rutting.severe_min_depth,
                        config.rutting.max_depth_severe,
                    ))
                if config.rutting.peak_normalize:
                    # 与单块路径同式：实现峰值 = d·(1+ε) ≤ 名义深度。
                    depth = depth / (1.0 + config.rutting.modulation_epsilon)
                entry["args"] = {
                    "center_line": center_line,
                    "wheel_separation": config.rutting.wheel_separation,
                    "width": config.rutting.rut_width,
                }
                # 全局 y 幅 × modulation_wavelength_ratio（设计稿：不得用
                # 段内云 y 范围，分段会漂移）
                params["modulation_wavelength"] = (
                    length * config.rutting.modulation_wavelength_ratio
                )
                params["modulation_epsilon"] = config.rutting.modulation_epsilon
                pending_scene.update(_pending_params_keys(add_rutting, params))
                entry["depth"] = depth

            elif disease_type == "corrugation":
                direction = str(rng.choice(["longitudinal", "transverse"]))
                wavelength = float(rng.uniform(*config.corrugation.wavelength_range))
                if severity == "light":
                    amplitude = float(rng.uniform(*config.corrugation.amplitude_light))
                else:
                    amplitude = float(rng.uniform(*config.corrugation.amplitude_severe))
                entry["args"] = {
                    "direction": direction, "wavelength": wavelength,
                    "amplitude": amplitude,
                }
                # 包络中心/σ 按全局 bbox 抽定（计划阶段复现基元抽取）
                params["envelope_center"] = (
                    float(rd.uniform(0.0, x_max_grid)),
                    float(rd.uniform(0.0, y_max_grid)),
                )
                params["envelope_sigma"] = float(rd.uniform(0.7, 1.3))

            elif disease_type == "bleeding":
                if config.bleeding.geometry_mode == "patch":
                    # 斑块口径：与单块路径同 helper、同 rng 消耗顺序；
                    # 掩码在施加阶段按坐标确定性散列构建（云无关）。
                    entry["patch"] = _bleeding_patch_params(
                        rng, config.bleeding, width, length, salt=seed
                    )
                else:
                    # legacy-strip：恒定 40% 面积、贯穿全路长的竖直条带。
                    entry["lane_center"] = float(rng.uniform(0.3, 0.7) * width)

            elif disease_type == "patching":
                entry["args"] = self._sample_patch_args(
                    rng, config.patching, width, length, label=20
                )
                # 参考面在宏观 + 设计几何的粗网格上拟合（纹理零均值，
                # lstsq 天然平均掉）；宏观场合成后回填。
                entry["needs_reference_plane"] = True

            elif disease_type == "concrete_damage":
                damage_type_c = (
                    target_spec.variant
                    if target_spec is not None
                    and target_spec.disease_type == "concrete_damage"
                    else str(rng.choice(CONCRETE_DAMAGE_TYPES))
                )
                entry["damage_type"] = damage_type_c
                if damage_type_c == "patching":
                    # 复用 patching 计划（标签 37）
                    entry["type"] = "patching"
                    entry["args"] = self._sample_patch_args(
                        rng, config.patching, width, length, label=37
                    )
                    entry["needs_reference_plane"] = True
                else:
                    sev = "-" if damage_type_c in CONCRETE_NO_SEVERITY else severity
                    entry["severity"] = sev
                    params.update(
                        {
                            "slab_length": config.concrete_damage.slab_length,
                            "slab_width": config.concrete_damage.slab_width,
                            "joint_width": config.concrete_damage.joint_width,
                            "x_offset": config.concrete_damage.x_offset,
                            "y_offset": config.concrete_damage.y_offset,
                            "label_width_floor": grid_res,
                            "grid_res": grid_res,
                            "domain_bounds": (0.0, x_max_grid, 0.0, y_max_grid),
                            # 水泥病害几何口径（与单块路径同键）
                            "crack_in_slab": config.concrete_damage.crack_in_slab,
                            "fault_scope": config.concrete_damage.fault_scope,
                            "edge_spall_mode": config.concrete_damage.edge_spall_mode,
                            "shatter_mode": config.concrete_damage.shatter_mode,
                        }
                    )
                    pending_scene.update(
                        _pending_params_keys(add_concrete_damage, params)
                    )
                    slab_len = config.concrete_damage.slab_length
                    slab_wid = config.concrete_damage.slab_width
                    if damage_type_c in ("slab_shatter", "corner_break", "blowup"):
                        # 全局板块列表上固定目标板 + 板块盒（网格精确算术）
                        ix_list = np.arange(
                            0, int(math.floor(x_max_grid / slab_wid)) + 1
                        )
                        iy_list = np.arange(
                            0, int(math.floor(y_max_grid / slab_len)) + 1
                        )
                        slabs = [
                            (int(a), int(b))
                            for a in ix_list
                            for b in iy_list
                        ]
                        target = slabs[int(rd.integers(len(slabs)))]
                        params["target_slab"] = target
                        xs_slab = x_grid_global[
                            (x_grid_global >= target[0] * slab_wid)
                            & (x_grid_global < (target[0] + 1) * slab_wid)
                        ]
                        ys_slab = y_grid_global[
                            (y_grid_global >= target[1] * slab_len)
                            & (y_grid_global < (target[1] + 1) * slab_len)
                        ]
                        if len(xs_slab) == 0 or len(ys_slab) == 0:
                            xs_slab = np.array([target[0] * slab_wid])
                            ys_slab = np.array([target[1] * slab_len])
                        params["slab_bounds"] = (
                            float(xs_slab.min()), float(xs_slab.max()),
                            float(ys_slab.min()), float(ys_slab.max()),
                        )
                    if damage_type_c == "corner_break":
                        sign_x = 1.0 if rd.random() < 0.5 else -1.0
                        sign_y = 1.0 if rd.random() < 0.5 else -1.0
                        params["corner_signs"] = (sign_x, sign_y)
                        params["chord_frac_x"] = float(rd.uniform(0.2, 0.5))
                        params["chord_frac_y"] = float(rd.uniform(0.2, 0.5))
                        # reach 在全局网格窗口上预计算（板块跨缝时各段只
                        # 看到部分三角，倾斜场必须用同一条 reach 定标）
                        bx0, bx1, by0, by1 = params["slab_bounds"]
                        wx = x_grid_global[
                            (x_grid_global >= bx0) & (x_grid_global <= bx1)
                        ]
                        wy = y_grid_global[
                            (y_grid_global >= by0) & (y_grid_global <= by1)
                        ]
                        Wx, Wy = np.meshgrid(wx, wy, indexing="ij")
                        slab_w_local = max(bx1 - bx0, 1e-6)
                        slab_l_local = max(by1 - by0, 1e-6)
                        corner_x = bx0 if sign_x > 0 else bx1
                        corner_y = by0 if sign_y > 0 else by1
                        sx = 1.0 if corner_x == bx0 else -1.0
                        sy = 1.0 if corner_y == by0 else -1.0
                        frac_a = params["chord_frac_x"]
                        frac_b = params["chord_frac_y"]
                        p_a = np.array([
                            corner_x + sx * frac_a * slab_w_local, corner_y
                        ])
                        p_b = np.array([
                            corner_x, corner_y + sy * frac_b * slab_l_local
                        ])
                        chord = p_b - p_a
                        rel = np.stack([Wx.ravel() - p_a[0], Wy.ravel() - p_a[1]], axis=1)
                        cross_pts = chord[0] * rel[:, 1] - chord[1] * rel[:, 0]
                        cross_corner = (
                            chord[0] * (corner_y - p_a[1])
                            - chord[1] * (corner_x - p_a[0])
                        )
                        triangle = cross_pts * np.sign(cross_corner) >= 0.0
                        dist_corner = np.sqrt(
                            (Wx.ravel() - corner_x) ** 2 + (Wy.ravel() - corner_y) ** 2
                        )
                        reach = (
                            float(dist_corner[triangle].max()) if np.any(triangle) else 1.0
                        ) or 1.0
                        params["corner_reach"] = reach
                    if damage_type_c == "faulting":
                        trans_joints = np.arange(
                            config.concrete_damage.y_offset,
                            y_max_grid + slab_len,
                            slab_len,
                        )
                        interior = trans_joints[
                            (trans_joints > 0.0) & (trans_joints < y_max_grid)
                        ]
                        if len(interior):
                            params["joint_y"] = float(interior[int(rd.choice(len(interior)))])
                        else:
                            params["joint_y"] = 0.5 * y_max_grid
                    if damage_type_c == "pitting":
                        params["pit_center"] = (
                            float(rd.uniform(0.05 * x_max_grid, 0.95 * x_max_grid)),
                            float(rd.uniform(0.05 * y_max_grid, 0.95 * y_max_grid)),
                        )
                    entry["args"] = {"damage_type": damage_type_c}

            plan.append(entry)

        # 5. raveling 计划（消耗顺序与单块路径一致；区域换成全局网格
        #    bbox 上的绝对坐标框）
        raveling_plan = None
        if raveling_entry is not None:
            rav_seed = int(rng.integers(0, 2 ** 31))
            x_frac_min = float(rng.uniform(0.05, 0.35))
            x_frac_max = x_frac_min + float(rng.uniform(0.2, 0.5))
            y_frac_min = float(rng.uniform(0.05, 0.35))
            y_frac_max = y_frac_min + float(rng.uniform(0.2, 0.5))
            raveling_plan = {
                "seed": rav_seed,
                "severity": raveling_entry[1],
                "box": (
                    x_frac_min * x_max_grid,
                    x_frac_max * x_max_grid,
                    y_frac_min * y_max_grid,
                    y_frac_max * y_max_grid,
                ),
                # 接线 config.raveling（默认值=历史硬编码，行为逐位不变）；
                # 键的待合入披露在计划阶段做一次，施加阶段直接转发。
                "params": {
                    "deterministic_seed": rav_seed,
                    "removal_ratio": (
                        config.raveling.point_removal_ratio_light
                        if raveling_entry[1] == "light"
                        else config.raveling.point_removal_ratio_severe
                    ),
                    "erosion_depth": tuple(config.raveling.erosion_depth),
                },
            }
            pending_scene.update(
                _pending_params_keys(add_raveling, raveling_plan["params"])
            )

        # 6. 宏观粗网格（全局一次）+ 修补参考面回填
        macro_extra, macro_pending = _filter_unsupported_kwargs(
            synthesize_strip_macro_band,
            {"macro_spectrum_mode": config.road.macro_spectrum_mode},
        )
        pending_scene.update(macro_pending)
        macro_band = synthesize_strip_macro_band(
            width,
            length,
            grid_res,
            roughness_actual,
            np.random.default_rng(surface_seed),
            macro_grid_factor=config.strip.macro_grid_factor,
            **macro_extra,
        )
        for entry in plan:
            if not entry.get("needs_reference_plane"):
                continue
            Xc, Yc = np.meshgrid(macro_band["x"], macro_band["y"], indexing="ij")
            z_macro = macro_band["field"].copy()
            if config.road.crossfall != 0.0:
                if config.road.crossfall_shape == "crowned":
                    z_macro = z_macro - config.road.crossfall * np.abs(
                        Xc - width / 2.0
                    )
                else:
                    z_macro = z_macro - config.road.crossfall * Xc
            if config.road.longitudinal_grade != 0.0:
                z_macro = z_macro + config.road.longitudinal_grade * Yc
            design = np.column_stack(
                (Xc.ravel(), Yc.ravel(), np.ones_like(Xc.ravel()))
            )
            plane, *_ = np.linalg.lstsq(design, z_macro.ravel(), rcond=None)
            entry["params"]["reference_plane"] = plane

        # 7. 辐射计划（消耗顺序与单块 _compute_intensity 一致：base →
        #    96×96 反照率场 → 斑点种子）；反照率场按全局 bbox 建立一次
        curb_half = config.curb.width_m
        x_lo = -curb_half if curb_half > 0 else 0.0
        x_hi = (
            width + (max(1, int(math.ceil(curb_half / grid_res))) * grid_res)
            if curb_half > 0
            else x_max_grid
        )
        if pavement_type == "asphalt":
            base = float(rng.uniform(0.12, 0.22))
        else:
            base = float(rng.uniform(0.32, 0.48))
        gx = gy = 96
        ext_x = max(x_hi - x_lo, 1e-6)
        ext_y = max(y_max_grid, 1e-6)
        fx = np.fft.fftfreq(gx, d=ext_x / gx)
        fy = np.fft.fftfreq(gy, d=ext_y / gy)
        f_r = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
        corr_len = config.radiometric.albedo_corr_len_m
        lowpass = np.exp(-((f_r * corr_len) ** 2))
        field = np.real(np.fft.ifft2(lowpass * np.fft.fft2(rng.normal(size=(gx, gy)))))
        fstd = field.std()
        if fstd > 1e-12:
            field = (field - field.mean()) / fstd
        from scipy.interpolate import RegularGridInterpolator

        albedo_interp = RegularGridInterpolator(
            (np.linspace(x_lo, x_hi, gx), np.linspace(0.0, ext_y, gy)),
            field,
            bounds_error=False,
            fill_value=0.0,
        )
        speckle_seed = int(rng.integers(0, 2 ** 31))

        # 8. 分段装配
        rows_seg = int(round(surface_estimate.segment_length_m / grid_res))
        m_rows = config.strip_halo_rows()
        segment_bounds = []
        k = 0
        while True:
            y0o = k * rows_seg
            if y0o >= ny_total:
                break
            y1o = min(ny_total, y0o + rows_seg)
            segment_bounds.append(
                (y0o, y1o, max(0, y0o - m_rows), min(ny_total, y1o + m_rows))
            )
            k += 1
        if len(segment_bounds) != surface_estimate.segment_count:
            raise RuntimeError(
                "Segment plan mismatch: estimate says "
                f"{surface_estimate.segment_count} segments, runtime derived "
                f"{len(segment_bounds)}."
            )

        n_alloc = surface_estimate.point_count
        asm_points = np.empty((n_alloc, 3), dtype=np.float32)
        asm_normals = np.empty((n_alloc, 3), dtype=np.float32)
        asm_labels = np.empty(n_alloc, dtype=np.int64)
        asm_power = np.empty(n_alloc, dtype=np.float32)
        asm_feats = np.zeros((n_alloc, 3), dtype=np.float32)

        sensor_origin = self._sensor_pose()
        strip_extra, strip_pending = _filter_unsupported_kwargs(
            generate_strip_surface,
            {
                "macro_spectrum_mode": config.road.macro_spectrum_mode,
                "grid_convention": config.micro_texture.grid_convention,
                "het_field": config.micro_texture.het_field,
            },
            moot_keys=_surface_moot_keys(config.micro_texture),
        )
        pending_scene.update(strip_pending)
        strip_gen = generate_strip_surface(
            width,
            length,
            grid_res,
            roughness_actual,
            surface_seed,
            segment_bounds=segment_bounds,
            blend_band_rows=m_rows,
            texture_rms=config.micro_texture.amplitude,
            texture_hurst=config.micro_texture.hurst,
            crossfall=config.road.crossfall,
            crossfall_shape=config.road.crossfall_shape,
            longitudinal_grade=config.road.longitudinal_grade,
            texture_het_sigma_ln=config.micro_texture.het_sigma_ln,
            texture_het_corr_len_m=config.micro_texture.het_corr_len_m,
            macro_grid_factor=config.strip.macro_grid_factor,
            macro_band=macro_band,
            **strip_extra,
        )
        speckle_rng = np.random.default_rng(speckle_seed)
        speckle_m = config.radiometric.speckle_m

        offset = 0
        seg_layout = []
        if config.lidar_scan.enable:
            warnings.warn(
                "observation_model='off': skipping LiDAR scan-line resampling; "
                "lidar_scan.enable=True is overridden (recorded in metadata).",
                stacklevel=2,
            )
        for _k, (y0o, y1o, y0e, y1e) in enumerate(segment_bounds):
            (
                _sk, _y0e, _y1e, _y0o, _y1o,
                xg, ye, z_ext, nrm_ext,
            ) = next(strip_gen)
            # y-major 布局（y 行在外、x 列在内）：row_idx = repeat 语义、
            # 装配块按 y 升序（特征 halo 的 searchsorted 依赖）都以它为准。
            Ym, Xm = np.meshgrid(ye, xg, indexing="ij")
            pts = np.stack(
                [Xm.ravel(), Ym.ravel(), z_ext.T.ravel()], axis=1
            )
            nrm = nrm_ext.transpose(1, 0, 2).reshape(-1, 3)
            labels = np.zeros(pts.shape[0], dtype=np.int64)
            row_idx = np.repeat(np.arange(y0e, y1e, dtype=np.int64), nx)

            # 病害施加（计划顺序）+ 标签优先级合并（与单块路径同逻辑；
            # 计划参数保证云无关 ⇒ 缝两侧逐位一致）
            for entry in plan:
                etype = entry["type"]
                old_labels = labels.copy()
                seed = entry["seed"]
                params = entry["params"]
                if etype == "crack":
                    pts, labels = add_crack(
                        pts, labels,
                        crack_type=entry["crack_type"],
                        severity=entry["severity"],
                        params=params, seed=seed,
                    )
                elif etype == "pothole":
                    pts, labels = add_pothole(
                        pts, labels,
                        severity=entry["severity"], seed=seed,
                        params=params,
                        **entry["args"],
                    )
                elif etype == "depression":
                    # label_radius_mode 经 _bridge_depression_params 下发：
                    # 请求值等于基元当前行为（legacy-1pct）时静默省略，
                    # 否则记入本场景待合入集合。
                    dep_kwargs, dep_pending = _bridge_depression_params(
                        entry["label_radius_mode"]
                    )
                    pending_scene.update(dep_pending)
                    pts, labels = add_depression(
                        pts, labels,
                        severity=entry["severity"], seed=seed,
                        **entry["args"], **dep_kwargs,
                    )
                elif etype == "rutting":
                    pts, labels = add_rutting(
                        pts, labels,
                        depth=entry["depth"],
                        severity=entry["severity"], seed=seed,
                        params=params, **entry["args"],
                    )
                elif etype == "corrugation":
                    pts, labels = add_corrugation(
                        pts, labels,
                        severity=entry["severity"], seed=seed,
                        params=params, **entry["args"],
                    )
                elif etype == "bleeding":
                    if "patch" in entry:
                        # 斑块口径：掩码按坐标确定性散列逐点判定（云无关，
                        # 缝两侧逐位一致）。
                        region_mask = _bleeding_region_mask(pts, entry["patch"])
                    else:
                        region_mask = (
                            np.abs(pts[:, 0] - entry["lane_center"])
                            < config.bleeding.region_width * width / 2.0
                        )
                    labels = add_bleeding(pts, labels, region_mask=region_mask, seed=seed)
                elif etype == "patching":
                    pts, labels = add_patching(
                        pts, labels,
                        params=params, **entry["args"],
                    )
                elif etype == "concrete_damage":
                    pts, labels = add_concrete_damage(
                        pts, labels,
                        severity=entry["severity"],
                        params=params, seed=seed,
                        **entry["args"],
                    )

                changed_mask = labels != old_labels
                if np.any(changed_mask):
                    max_label = max(LABEL_PRIORITY.keys()) + 1
                    priority_lut = np.zeros(max_label, dtype=np.int32)
                    for kk, vv in LABEL_PRIORITY.items():
                        priority_lut[kk] = vv
                    old_prio = priority_lut[old_labels[changed_mask]]
                    new_prio = priority_lut[labels[changed_mask]]
                    restore = old_prio > new_prio
                    changed_indices = np.where(changed_mask)[0]
                    labels[changed_indices[restore]] = old_labels[changed_indices[restore]]

            # 缘石并入（拥有行；几何确定性零 rng）
            if config.curb.width_m > 0:
                y_owned_vals = ye[(y0o - y0e):(y1o - y0e)]
                curb_pts, curb_nrm = generate_curb_strip(
                    road_width=width,
                    length=length,
                    grid_res=grid_res,
                    curb_width_m=config.curb.width_m,
                    curb_height_m=config.curb.height_m,
                    crossfall=config.road.crossfall,
                    crossfall_shape=config.road.crossfall_shape,
                    longitudinal_grade=config.road.longitudinal_grade,
                    y_values=y_owned_vals,
                )
                # 转成 y-major（基元返回 x 块在外的布局）
                n_curb_cols = curb_pts.shape[0] // max(len(y_owned_vals), 1)
                curb_pts = curb_pts.reshape(n_curb_cols, len(y_owned_vals), 3).transpose(1, 0, 2).reshape(-1, 3)
                curb_nrm = curb_nrm.reshape(n_curb_cols, len(y_owned_vals), 3).transpose(1, 0, 2).reshape(-1, 3)
                n_road_current = pts.shape[0]
                pts = np.vstack([pts, curb_pts])
                nrm = np.vstack([nrm, curb_nrm])
                labels = np.concatenate(
                    [labels, np.zeros(curb_pts.shape[0], dtype=labels.dtype)]
                )
                road_mask = np.arange(pts.shape[0]) < n_road_current
                # 缘石点恒为拥有点：row 占位为段首行（owned 判定用
                # ~road_mask，不走行号）
                row_idx = np.concatenate(
                    [row_idx, np.full(curb_pts.shape[0], y0o, dtype=np.int64)]
                )
            else:
                road_mask = np.ones(pts.shape[0], dtype=bool)

            # raveling（计划区域框；框在路面 bbox 内，缘石 x<0 自然排除）
            if raveling_plan is not None:
                bx0, bx1, by0, by1 = raveling_plan["box"]
                region_mask = (
                    (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1)
                    & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
                )
                pts, labels = add_raveling(
                    pts, labels,
                    region_mask=region_mask,
                    severity=raveling_plan["severity"],
                    seed=raveling_plan["seed"],
                    # 云无关：摘除/侵蚀按坐标哈希逐点确定，缝两侧一致
                    # （松散框可占 0.2-0.5·L，必然跨段，无段能完整包含）。
                    params=raveling_plan["params"],
                )

            # 松散标签膨胀 + NaN 过滤（与单块路径同逻辑，halo 供 KDTree）
            raveling_labels_mask = (labels == 11) | (labels == 12)
            valid_mask = ~np.any(np.isnan(pts), axis=1)
            raveling_valid_mask = raveling_labels_mask & valid_mask
            if np.any(raveling_valid_mask):
                raveling_xy = pts[raveling_valid_mask][:, :2]
                raveling_severity = labels[raveling_valid_mask]
                background_valid_mask = valid_mask & ~raveling_labels_mask & road_mask
                if np.any(background_valid_mask) and raveling_xy.shape[0] > 0:
                    from scipy.spatial import KDTree as _KDTree
                    bg_xy = pts[background_valid_mask][:, :2]
                    bg_labels = labels[background_valid_mask]
                    bg_global_idx = np.where(background_valid_mask)[0]
                    tree_dilate = _KDTree(raveling_xy)
                    dilate_radius = grid_res * 3.0
                    nn_dist, nn_idx = tree_dilate.query(bg_xy, k=1)
                    in_dilate = nn_dist <= dilate_radius
                    for i in np.where(in_dilate)[0]:
                        if bg_labels[i] == 0:
                            labels[bg_global_idx[i]] = raveling_severity[nn_idx[i]]

            if not np.any(valid_mask):
                continue
            pts_v = pts[valid_mask]
            labels_v = labels[valid_mask]
            nrm_v = nrm[valid_mask]
            road_mask_v = road_mask[valid_mask]
            row_idx_v = row_idx[valid_mask]
            if not np.any(road_mask_v):
                continue
            depression_depth = local_depression_depth(pts_v)
            power_v = self._strip_received_power(
                pts_v, nrm_v, labels_v, pavement_type, sensor_origin,
                albedo_interp, config.radiometric.albedo_sigma_ln, base,
                depression_depth,
                bleeding_multiplier=config.bleeding.intensity_multiplier,
                cos_abs=config.radiometric.cos_abs,
                albedo_clip_hi=config.radiometric.albedo_clip_hi,
            )

            owned_road = road_mask_v & (row_idx_v >= y0o) & (row_idx_v < y1o)
            owned_curb = ~road_mask_v
            n_own = int(np.count_nonzero(owned_road))
            n_curb = int(np.count_nonzero(owned_curb))
            road_start = offset
            asm_points[road_start:road_start + n_own] = pts_v[owned_road].astype(np.float32)
            asm_normals[road_start:road_start + n_own] = nrm_v[owned_road].astype(np.float32)
            asm_labels[road_start:road_start + n_own] = labels_v[owned_road]
            asm_power[road_start:road_start + n_own] = power_v[owned_road].astype(np.float32)
            offset += n_own
            seg = {
                "road": (road_start, offset),
                "y0o": y0o, "y1o": y1o,
                "curb": None,
            }
            if n_curb:
                curb_start = offset
                asm_points[curb_start:curb_start + n_curb] = pts_v[owned_curb].astype(np.float32)
                asm_normals[curb_start:curb_start + n_curb] = nrm_v[owned_curb].astype(np.float32)
                asm_labels[curb_start:curb_start + n_curb] = labels_v[owned_curb]
                asm_power[curb_start:curb_start + n_curb] = power_v[owned_curb].astype(np.float32)
                offset += n_curb
                seg["curb"] = (curb_start, offset)
            seg_layout.append(seg)

        n_final = offset
        if n_final == 0:
            raise RuntimeError("Segmented strip produced no valid points.")
        asm_points = asm_points[:n_final]
        asm_normals = asm_normals[:n_final]
        asm_labels = asm_labels[:n_final]
        asm_power = asm_power[:n_final]
        asm_feats = asm_feats[:n_final]

        # 9. AGC 标定 + 逐段斑点/量化（两遍扫）。'fixed'（默认）= 固定
        # 绝对标定（config.radiometric.agc_reference_power，工程标定值）：
        # 不随场景构成漂移、材料基底签名保留；'scene-p99' = 历史整条带
        # p99 归一（回退口径）。
        if config.radiometric.agc_mode == "fixed":
            scale = float(config.radiometric.agc_reference_power)
        else:
            scale = float(np.percentile(asm_power, 99.0))
        for seg in seg_layout:
            start = seg["road"][0]
            end = seg["curb"][1] if seg["curb"] else seg["road"][1]
            intensity = asm_power[start:end].astype(np.float64) / max(scale, 1e-12)
            intensity = intensity * speckle_rng.gamma(
                shape=speckle_m, scale=1.0 / speckle_m, size=end - start
            )
            intensity = np.clip(intensity, 0.0, 1.0)
            intensity = np.round(intensity * 255.0) / 255.0
            asm_feats[start:end, 0] = intensity.astype(np.float32)

        # 物理边界（米制）：必须在归一化之前取——单块路径同语义，
        # physical_coordinates.bounds_*_m 才是真正的米制范围。
        physical_bounds_min_m = asm_points.min(axis=0).astype(np.float64)
        physical_bounds_max_m = asm_points.max(axis=0).astype(np.float64)

        # 10. 坐标归一化（流式：质心平移 → 最大半径 → 原地缩放），与
        # 单块路径 _normalize_with_metadata 同语义：减质心 + 除半径，
        # normalize=False 时完全不动点云（center=zeros/scale=1）。
        coordinate_center = np.zeros(3, dtype=np.float32)
        coordinate_scale = 1.0
        coordinates_normalized = False
        max_radius = 1.0
        if config.normalize:
            coord_sum = np.zeros(3, dtype=np.float64)
            for start in range(0, n_final, 1_000_000):
                coord_sum += asm_points[
                    start:start + 1_000_000
                ].sum(axis=0, dtype=np.float64)
            centroid = coord_sum / float(n_final)
            for start in range(0, n_final, 1_000_000):
                asm_points[start:start + 1_000_000] -= centroid
            max_radius = 0.0
            for start in range(0, n_final, 1_000_000):
                d = asm_points[start:start + 1_000_000].astype(np.float64)
                max_radius = max(
                    max_radius, float(np.max(np.linalg.norm(d, axis=1)))
                )
            if max_radius > 1e-12:
                asm_points /= np.float32(max_radius)
            else:
                max_radius = 1.0
            coordinate_center = centroid.astype(np.float32)
            coordinate_scale = float(max_radius)
            coordinates_normalized = True

        # 11. 特征第二遍：段+halo 的 cKDTree，分块 kNN 重算两个几何通道
        # （halo = max(0.05, 16·grid_res)；断言 support_radius < halo）
        halo_m = config.strip.feature_halo_m
        if halo_m is None:
            halo_m = max(0.05, 16.0 * grid_res)
        k_feat = DEFAULT_GEOMETRY_K_NEIGHBORS
        from scipy.spatial import cKDTree as _cKDTree

        for seg in seg_layout:
            own_lo = seg["road"][0]
            own_hi = seg["curb"][1] if seg["curb"] else seg["road"][1]
            own_idx = np.arange(own_lo, own_hi)
            halo_parts = [own_idx]
            # 邻段块中 y 落在本段 halo 带内的点（归一化后 y 带按比例缩放；
            # 段内块按 y 升序，可 searchsorted）
            # halo 带用与点云相同的仿射（减质心 + 除半径）映射到归一化
            # 坐标；normalize=False 时保持米制。
            if coordinates_normalized:
                cy0 = float(coordinate_center[1])
                band_lo = (seg["y0o"] * grid_res - halo_m - cy0) / max_radius
                band_hi = (seg["y1o"] * grid_res + halo_m - cy0) / max_radius
            else:
                band_lo = seg["y0o"] * grid_res - halo_m
                band_hi = seg["y1o"] * grid_res + halo_m
            for other in seg_layout:
                if other is seg:
                    continue
                for block in ("road", "curb"):
                    rng_block = other.get(block)
                    if not rng_block:
                        continue
                    bs, be = rng_block
                    ys = asm_points[bs:be, 1]
                    lo_i = int(np.searchsorted(ys, band_lo))
                    hi_i = int(np.searchsorted(ys, band_hi))
                    if hi_i > lo_i:
                        halo_parts.append(np.arange(bs + lo_i, bs + hi_i))
            local_idx = np.concatenate(halo_parts)
            local_pts = asm_points[local_idx]
            tree = _cKDTree(local_pts)
            k_eff = min(k_feat, len(local_pts) - 1)
            if k_eff < 3:
                continue
            n_own = own_hi - own_lo
            for start in range(0, n_own, 250_000):
                stop = min(start + 250_000, n_own)
                chunk = asm_points[own_lo + start:own_lo + stop].astype(np.float64)
                dist, idx = tree.query(chunk, k=k_eff + 1)
                neighbors = local_pts[idx[:, 1:]]
                centers = neighbors.mean(axis=1)
                offsets = neighbors - centers[:, None, :]
                covariance = np.einsum("nki,nkj->nij", offsets, offsets) / max(k_eff - 1, 1)
                eigvals, eigvecs = np.linalg.eigh(covariance)
                total_variance = np.maximum(eigvals.sum(axis=1), 1e-12)
                curvature = np.clip(eigvals[:, 0] / total_variance, 0.0, 1.0)
                normals_pca = eigvecs[:, :, 0]
                normals_pca[normals_pca[:, 2] < 0.0] *= -1.0
                residual = np.einsum("ni,ni->n", chunk - centers, normals_pca)
                support_radius = np.maximum(dist[:, -1], 1e-6)
                if float(np.max(support_radius)) >= halo_m:
                    raise RuntimeError(
                        "Feature halo too small: support radius "
                        f"{float(np.max(support_radius)):.4f} m >= halo {halo_m:.4f} m "
                        "(raveling holes or grid gaps exceed the kNN support)."
                    )
                scaled = np.clip(residual / support_radius, -3.0, 3.0)
                asm_feats[own_lo + start:own_lo + stop, 1] = curvature.astype(np.float32)
                asm_feats[own_lo + start:own_lo + stop, 2] = scaled.astype(np.float32)
            del tree

        # 12. 元数据与场景字典（与单块 off 路径同结构）
        scan_override = (
            "skipped_under_observation_off" if config.lidar_scan.enable else None
        )
        resolution_metadata = config.resolution_metadata(actual_output_points=n_final)
        if pending_scene:
            # 与单块路径同语义：本场景实际遇到的待合入键（场景级准确）。
            resolution_metadata["pending_primitive_upgrades"] = sorted(
                pending_scene
            )
        resolution_metadata["surface_grid"]["curvature_method"] = "skipped_observation_off"
        resolution_metadata["sensor_output"].update(
            {
                "mode": "full_surface_observation_off",
                "scan_pattern_points": int(surface_estimate.point_count),
                "points_before_output_sampling": int(n_final),
                "sampling_method": "none_full_surface",
                "output_to_surface_point_ratio": n_final / surface_estimate.point_count,
                "fixed_count_upsampling_used": False,
                "pipeline_steps_skipped": [
                    "scan_resample",
                    "curvature",
                    "geometric_occlusion",
                    "lidar_noise",
                    "nn_label_transfer",
                    "output_sampling",
                ],
                "occlusion_policy": "off_with_observation_model",
                "ignored_under_observation_off": True,
            }
        )
        if scan_override is not None:
            resolution_metadata["sensor_output"]["scan_override"] = scan_override
        resolution_metadata["sensor_output"]["sensor_pose_m"] = [
            float(v) for v in sensor_origin
        ]
        resolution_metadata["sensor_output"]["intensity_model"] = (
            "radiometric.v2: rho(x,y)*shading*cos(theta)/R^2, "
            "lognormal albedo field (corr "
            f"{config.radiometric.albedo_corr_len_m:g} m, sigma_ln "
            f"{config.radiometric.albedo_sigma_ln:g}), "
            f"Gamma speckle M={config.radiometric.speckle_m:g}, "
            "8-bit quantization; sigma_r and dropout "
            "share the same received-power proxy"
        )
        protection_info = {
            "protected_label": -1 if target_label is None else int(target_label),
            "protected_min_points": int(config.target_label_min_output_points),
            "protection_available": True,
            "protection_applied": False,
        }
        if target_label is not None:
            target_output_points = int(np.count_nonzero(asm_labels == target_label))
            protection_info["target_label_output_points"] = target_output_points
            protection_info["target_label_output_ratio"] = (
                target_output_points / max(n_final, 1)
            )
        resolution_metadata["sensor_output"]["target_label_protection"] = protection_info
        resolution_metadata["physical_coordinates"] = {
            "unit": "m",
            "bounds_min_m": physical_bounds_min_m.tolist(),
            "bounds_max_m": physical_bounds_max_m.tolist(),
            "stored_coordinates_normalized": coordinates_normalized,
            "normalization_is_reversible": True,
        }

        return {
            "points": asm_points,
            "labels": asm_labels.astype(np.int64),
            "feats": asm_feats,
            "normals": asm_normals.astype(np.float32),
            "pavement_type": pavement_type,
            "roughness_class": roughness_actual,
            "target_label": -1 if target_label is None else int(target_label),
            "feature_schema": OBSERVABLE_FEATURE_SCHEMA,
            "feature_names": np.asarray(OBSERVABLE_FEATURE_NAMES),
            "feature_k_neighbors": int(DEFAULT_GEOMETRY_K_NEIGHBORS),
            "coordinate_center": coordinate_center.astype(np.float32),
            "coordinate_scale": float(coordinate_scale),
            "coordinates_normalized": coordinates_normalized,
            "resolution_metadata": resolution_metadata,
            "resolution_contract": resolution_metadata["contract_version"],
            "surface_grid_spacing_m": float(grid_res),
            "surface_grid_shape": np.asarray(surface_estimate.grid_shape, dtype=np.int64),
            "surface_grid_point_count": int(surface_estimate.point_count),
            "sensor_output_point_count": int(n_final),
            "observation_model": str(config.observation_model),
            "segment_count": int(surface_estimate.segment_count),
            "segment_length_m": float(surface_estimate.segment_length_m),
            "segment_overlap_m": float(surface_estimate.overlap_m),
            "model_target_points": (
                -1
                if config.model_target_points is None
                else int(config.model_target_points)
            ),
        }

    @staticmethod
    def _strip_received_power(
        points: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray,
        pavement_type: str,
        sensor_origin: np.ndarray,
        albedo_interp,
        albedo_sigma_ln: float,
        base: float,
        depression_depth: np.ndarray,
        bleeding_multiplier: float = 0.55,
        cos_abs: bool = False,
        albedo_clip_hi: float = 0.98,
    ) -> np.ndarray:
        """分段体制的强度场第一遍：received_power = ρ·shading·cosθ/R²。

        与 :meth:`_compute_intensity` 的场部分逐式相同（反照率乘子、
        凹陷遮蔽、几何项），AGC/斑点/量化移到整条带标定之后的第二遍。
        """
        N = labels.shape[0]
        mult = np.ones(N, dtype=np.float64)
        mult[labels == 19] = bleeding_multiplier
        mult[(labels == 20) | (labels == 37)] = 0.85
        mult[(labels == 11) | (labels == 12)] = 1.15
        mult[labels == 36] = 1.20
        mult[labels == 29] = 1.10
        rho_field = np.exp(albedo_sigma_ln * albedo_interp(points[:, :2]))
        rho = np.clip(base * mult * rho_field, 0.02, albedo_clip_hi)
        shading = 0.25 + 0.75 * np.exp(-depression_depth / 0.012)
        d = points - sensor_origin[None, :]
        R2 = np.sum(d * d, axis=1) + 1e-12
        d_hat = d / np.sqrt(R2)[:, None]
        cos_theta = _lambert_cos_theta(normals, d_hat, cos_abs)
        return rho * shading * cos_theta / R2

    def _select_pavement_type(self, rng: np.random.Generator) -> str:
        """根据配置的病害概率选择路面类型。

        仅沥青概率 > 0 → 'asphalt'；仅水泥 > 0 → 'concrete'；否则等概率随机。
        """
        disease_probs = self.config.disease.disease_probs
        has_asphalt = any(disease_probs.get(k, 0) > 0 for k in
                         ["crack", "pothole", "raveling", "depression",
                          "rutting", "corrugation", "bleeding", "patching"])
        has_concrete = disease_probs.get("concrete_damage", 0) > 0

        config_type = self.config.road.pavement_type
        if config_type == "asphalt":
            return "asphalt"
        elif config_type == "concrete":
            return "concrete"
        elif config_type == "mixed":
            return str(rng.choice(["asphalt", "concrete"]))
        elif has_asphalt and not has_concrete:
            return "asphalt"
        elif has_concrete and not has_asphalt:
            return "concrete"
        else:
            return str(rng.choice(["asphalt", "concrete"]))

    def _select_diseases(
        self,
        rng: np.random.Generator,
        pavement_type: str,
    ) -> list[tuple[str, str]]:
        """根据路面类型和配置概率随机选择病害组合。

        ``sampling_mode='bernoulli'``（默认）：按 ``disease_probs`` 逐
        病害独立伯努利抽样——概率即文档声明的『出现概率』语义，期望
        每场景病害数 ≈ Σp（默认配置 ≈ 0.8 种），**允许 0 病害背景场
        景**——再按 ``max_diseases_per_scene`` 封顶。
        ``sampling_mode='legacy'``：历史口径——概率归一化为权重、无放
        回抽恰好 min(max_diseases, n_available) 种（每场景病害数恒
        定，各病害出现率远高于名义概率）。
        ``use_stratified=False`` 仍强制伯努利（旧 --no-stratified 语义，
        已弃用）。
        """
        disease_probs = self.config.disease.disease_probs
        max_diseases = self.config.disease.max_diseases_per_scene
        severity_ratio = self.config.disease.severity_ratio
        sampling_mode = self.config.disease.sampling_mode
        if not self.config.disease.use_stratified:
            sampling_mode = "bernoulli"

        available = ASPHALT_DISEASE_KEYS if pavement_type == "asphalt" else ["concrete_damage"]

        if sampling_mode == "legacy":
            weighted = [(d, disease_probs.get(d, 0.0)) for d in available]
            weights = np.array([w for _, w in weighted], dtype=np.float64)
            valid = weights > 0.0
            weighted = [item for item, keep in zip(weighted, valid, strict=False) if keep]
            weights = weights[valid]
            if len(weighted) == 0:
                selected = []
            else:
                weights = weights / weights.sum()
                n_select = min(max_diseases, len(weighted))
                selected = list(
                    rng.choice([d for d, _ in weighted], size=n_select, replace=False, p=weights)
                )
        else:
            # 伯努利：每种病害独立出现；概率 0 的病害从不生成。
            selected = []
            for disease in available:
                prob = disease_probs.get(disease, 0.0)
                if prob > 0.0 and rng.random() < prob:
                    selected.append(disease)

        if len(selected) > max_diseases:
            selected = list(rng.choice(selected, size=max_diseases, replace=False))

        result: list[tuple[str, str]] = []
        for disease in selected:
            if disease == "bleeding":
                result.append((disease, "light"))
            else:
                # crack.severity_ratio 为 per-type 覆盖键（None = 用
                # disease.severity_ratio 全局值）。
                if disease == "crack" and self.config.crack.severity_ratio is not None:
                    ratio = float(self.config.crack.severity_ratio)
                else:
                    ratio = severity_ratio
                severity = "light" if rng.random() < ratio else "severe"
                result.append((disease, severity))

        return result

    def _apply_patching(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        patch_label: int,
        width: float,
        length: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample and apply one physically continuous repair patch."""
        args = self._sample_patch_args(rng, self.config.patching, width, length, patch_label)
        return add_patching(points, labels, **args)

    @staticmethod
    def _sample_patch_args(
        rng: np.random.Generator,
        cfg: PatchingConfig,
        width: float,
        length: float,
        label: int,
    ) -> dict:
        """Sample one repair-patch geometry（单块路径与分段计划路径共用）。

        替换三处 ~30 行复制的抽样块。rng 消耗顺序（三处历史调用点一
        致，勿改动）：宽 → 长 → center_x → center_y → elevation →
        angle。elevation 域取 ``cfg.elevation_range``（默认 (-0.006,
        0.015)，与历史 uniform(-0.4·max_elevation, max_elevation) 逐位
        一致——0.4·0.015 与 0.006 在 IEEE 双精度下精确相等）。

        返回值即 :func:`add_patching` 的关键字参数（含 ``label``）。
        """
        max_patch_width = min(cfg.max_width, width * 0.8)
        max_patch_length = min(cfg.max_length, length * 0.8)
        min_patch_width = min(cfg.min_width, max_patch_width)
        min_patch_length = min(cfg.min_length, max_patch_length)
        patch_width = float(rng.uniform(min_patch_width, max_patch_width))
        patch_length = float(rng.uniform(min_patch_length, max_patch_length))
        half_width = patch_width * 0.5 + cfg.edge_width
        half_length = patch_length * 0.5 + cfg.edge_width

        if width <= 2.0 * half_width:
            center_x = width * 0.5
        else:
            center_x = float(rng.uniform(half_width, width - half_width))
        if length <= 2.0 * half_length:
            center_y = length * 0.5
        else:
            center_y = float(rng.uniform(half_length, length - half_length))

        elevation = float(rng.uniform(*cfg.elevation_range))
        angle_rad = float(rng.uniform(-np.pi / 12.0, np.pi / 12.0))
        return {
            "center": (center_x, center_y),
            "width": patch_width,
            "length": patch_length,
            "label": int(label),
            "angle_rad": angle_rad,
            "elevation": elevation,
            "edge_width": cfg.edge_width,
        }

    # 委托到模块级实现：病害几何采样助手（_bleeding_patch_params 等）
    # 与类内调用点共用同一函数，保证单块/分段两条路径的 rng 消耗一致。
    _sample_wheelpath_x = staticmethod(_sample_wheelpath_x)

    def _sensor_pose(self) -> np.ndarray:
        """整条观测管线共用的传感器位姿 (x, y, z)，米制路面坐标。"""
        pose = self.config.lidar_scan.sensor_pose
        if pose is not None:
            return np.asarray(pose, dtype=np.float64).reshape(3)
        # 车载默认：横向居中、位于场景起点前方 1 m、高 2 m。
        return np.array([self.config.road.width / 2.0, -1.0, 2.0], dtype=np.float64)

    def _compute_intensity(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray,
        pavement_type: str,
        sensor_origin: np.ndarray,
        rng: np.random.Generator,
        depression_depth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """辐射强度模型：I ∝ ρ(x,y)·cosθ / R²，加斑点噪声与 8-bit 量化。

        激光雷达方程 (Jelalian 1992) 的朗伯近似。类别信息通过两条
        受控通道进入强度：**五类材料标签的反照率乘子**（泛油/修补/
        松散/露骨/唧泥 ×0.55/0.85/1.15/1.20/1.10）与**空间相关对数
        正态反照率场 ρ(x,y)**（σ_ln=0.25 的乘性场保证类内方差大、类
        间分布高度重叠——实测泛油/背景单通道 AUC ≈ 0.62，远非可分）。
        旧实现额外的**加性固定偏移**使单通道强度即可近乎完美分类，
        构成隐性标签泄漏，已被移除并被审计确认为强度域差主因；按标
        签查表的乘子保留但与重叠场相乘，泄漏量级可控。

        组成：
        - 材料反照率均值（宽带太阳反照率公开区间的占位取值，未按
          905 nm 实测标定）：沥青 ~0.17 / 水泥 ~0.40；泛油（沥青结合料油膜，NIR 强吸收 + 镜面损耗）
          **变暗** ×``bleeding.intensity_multiplier``（默认 0.55，占位
          值待实测标定）；修补（富油新料）×0.85；松散/露骨（集料
          外露）×1.15-1.2。裂缝/坑槽类不做材料偏移——凹陷内的变暗
          由几何遮蔽因子产生。
        - 空间相关反照率场：对数正态乘性场（FFT 低通，相关长度
          ``radiometric.albedo_corr_len_m``，σ_ln ``radiometric.albedo_sigma_ln``），
          保证类内方差大、类间分布重叠。
        - 几何遮蔽：低于局部参考面的点按 exp(-depth/12 mm) 变暗
          （深槽回波弱），参考面取 0.08 m 粗网格中位数。
        - cosθ/R² 几何项（cosθ 口径见 ``radiometric.cos_abs``）、AGC
          归一（``radiometric.agc_mode``：'fixed' 固定绝对标定 /
          'scene-p99' 历史逐场景 99 分位）、反照率截断
          （``radiometric.albedo_clip_hi``）、乘性 Gamma 斑点
          （M=``radiometric.speckle_m``）、uint8 量化。

        分布参数取自 ``config.radiometric``，默认值逐字等于历史硬编码
        （0.25 / 0.3 / 12.0），因此默认构造时这些项输出逐点一致。注意
        ``received_power`` 同时供 σ_r 异方差与辐射耦合丢点使用，改动
        这些参数会经丢点间接改变输出点数——这是预期的物理耦合，不是
        点数缺陷。

        Returns:
            intensity: 归一化强度 (N,) float32 ∈ [0, 1]，256 级。
            received_power: 未归一化接收功率代理 (N,)，供逐点
                σ_r 与辐射耦合丢点复用（同一 ρcosθ/R² 的两面）。
        """
        N = labels.shape[0]

        # --- 材料反照率均值 ---
        if pavement_type == "asphalt":
            base = float(rng.uniform(0.12, 0.22))
        else:
            base = float(rng.uniform(0.32, 0.48))
        mult = np.ones(N, dtype=np.float64)
        mult[labels == 19] = self.config.bleeding.intensity_multiplier
        mult[(labels == 20) | (labels == 37)] = 0.85    # 修补：富油新料略暗
        mult[(labels == 11) | (labels == 12)] = 1.15    # 松散：集料外露略亮
        mult[labels == 36] = 1.20                       # 露骨
        mult[labels == 29] = 1.10                       # 唧泥：细料泛白

        # --- 空间相关对数正态反照率场 ---
        gx = gy = 96
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        ext_x = max(x_max - x_min, 1e-6)
        ext_y = max(y_max - y_min, 1e-6)
        fx = np.fft.fftfreq(gx, d=ext_x / gx)
        fy = np.fft.fftfreq(gy, d=ext_y / gy)
        f_r = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
        corr_len = self.config.radiometric.albedo_corr_len_m  # m
        lowpass = np.exp(-((f_r * corr_len) ** 2))
        field = np.real(np.fft.ifft2(lowpass * np.fft.fft2(rng.normal(size=(gx, gy)))))
        fstd = field.std()
        if fstd > 1e-12:
            field = (field - field.mean()) / fstd
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (np.linspace(x_min, x_max, gx), np.linspace(y_min, y_max, gy)),
            field, bounds_error=False, fill_value=0.0,
        )
        rho_field = np.exp(self.config.radiometric.albedo_sigma_ln * interp(points[:, :2]))
        rho = np.clip(
            base * mult * rho_field, 0.02, self.config.radiometric.albedo_clip_hi
        )

        # --- 几何遮蔽：相对局部参考面的凹陷深度（与遮挡模型共用） ---
        if depression_depth is None:
            depression_depth = local_depression_depth(points)
        shading = 0.25 + 0.75 * np.exp(-depression_depth / 0.012)

        # --- 几何项与接收功率 ---
        d = points - sensor_origin[None, :]
        R2 = np.sum(d * d, axis=1) + 1e-12
        d_hat = d / np.sqrt(R2)[:, None]
        cos_theta = _lambert_cos_theta(normals, d_hat, self.config.radiometric.cos_abs)
        received_power = rho * shading * cos_theta / R2

        # --- AGC 归一 + 斑点噪声 + 量化 ---
        # 'fixed'（默认）：固定绝对标定（agc_reference_power，工程标定
        # 值）——不随场景构成漂移、材料基底签名保留；'scene-p99'：历史
        # 逐场景 99 分位归一（回退口径）。
        if self.config.radiometric.agc_mode == "fixed":
            scale = float(self.config.radiometric.agc_reference_power)
        else:
            scale = float(np.percentile(received_power, 99.0))
        intensity = received_power / max(scale, 1e-12)
        speckle_m = self.config.radiometric.speckle_m
        intensity = intensity * rng.gamma(shape=speckle_m, scale=1.0 / speckle_m, size=N)
        intensity = np.clip(intensity, 0.0, 1.0)
        intensity = np.round(intensity * 255.0) / 255.0

        return intensity.astype(np.float32), received_power

    @staticmethod
    def _resample_to_target(
        points: np.ndarray,
        labels: np.ndarray,
        intensity: np.ndarray,
        curvature: np.ndarray,
        normals: np.ndarray,
        target_num: int,
        rng: np.random.Generator,
        protected_label: int | None = None,
        protected_min_points: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """重采样到目标点数。多于目标则随机下采样，少于目标则随机重复补齐。

        受控场景 (protected_label 非 None) 先做自然随机下采样，只有当目标
        标签的自然存活点数低于 protected_min_points 时才强制补足到最低数量。
        不再保留任何固定比例配额，因此输出场景的病害点比例反映自然
        prevalence，而不是被生成器人为固定。
        """
        N = points.shape[0]
        protection_info = {
            "protected_label": -1 if protected_label is None else int(protected_label),
            "protected_min_points": int(protected_min_points),
            "protection_available": True,
            "protection_applied": False,
        }
        if target_num == N:
            return points, labels, intensity, curvature, normals, protection_info

        if target_num < N:
            idx = rng.choice(N, size=target_num, replace=False)
            if protected_label is not None and protected_min_points > 0:
                protected_idx = np.flatnonzero(labels == protected_label)
                min_keep = min(len(protected_idx), protected_min_points, target_num)
                natural_kept = int(np.count_nonzero(labels[idx] == protected_label))
                if natural_kept < min_keep:
                    kept_protected = rng.choice(protected_idx, size=min_keep, replace=False)
                    remaining_pool = np.setdiff1d(
                        np.arange(N), kept_protected, assume_unique=False
                    )
                    kept_other = rng.choice(
                        remaining_pool, size=target_num - min_keep, replace=False
                    )
                    idx = np.concatenate((kept_protected, kept_other))
                    protection_info["protection_applied"] = True
                    # min_keep is capped by the surviving pool: record the
                    # value actually enforced so the audit record cannot be
                    # read as "at least protected_min_points points exist".
                    protection_info["effective_min_keep"] = int(min_keep)
                    protection_info["min_unreachable"] = bool(
                        min_keep < protected_min_points
                    )
        else:
            n_extra = target_num - N
            extra_idx = rng.choice(N, size=n_extra, replace=True)
            idx = np.concatenate([np.arange(N), extra_idx])

        return (
            points[idx],
            labels[idx],
            intensity[idx],
            curvature[idx],
            normals[idx],
            protection_info,
        )

    @staticmethod
    def _voxel_downsample(
        points: np.ndarray,
        labels: np.ndarray,
        intensity: np.ndarray,
        curvature: np.ndarray,
        normals: np.ndarray,
        voxel_size: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """P1-2: 体素质心下采样，保留密度结构。"""
        if points.shape[0] < 10:
            return points, labels, intensity, curvature, normals

        # 体素索引仅使用 xy 维度（路面高度变化不应用于体素分割）
        voxel_indices = np.floor(points[:, :2] / voxel_size).astype(np.int64)
        voxel_dict: dict[tuple[int, int], list[int]] = {}
        for i, vi in enumerate(voxel_indices):
            key = (int(vi[0]), int(vi[1]))
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        N_voxels = len(voxel_dict)

        # 聚合
        result_pts = np.zeros((N_voxels, 3), dtype=np.float64)
        result_labels = np.zeros(N_voxels, dtype=np.int64)
        result_intensity = np.zeros(N_voxels, dtype=np.float32)
        result_curvature = np.zeros(N_voxels, dtype=np.float32)
        result_normals = np.zeros((N_voxels, 3), dtype=np.float32)

        for vi, (_key, idx_list) in enumerate(voxel_dict.items()):
            idx_arr = np.array(idx_list)
            result_pts[vi] = np.mean(points[idx_arr], axis=0)
            lbl_counts = np.bincount(labels[idx_arr].astype(np.int64))
            result_labels[vi] = np.argmax(lbl_counts)
            result_intensity[vi] = np.mean(intensity[idx_arr])
            result_curvature[vi] = np.mean(curvature[idx_arr])
            result_normals[vi] = np.mean(normals[idx_arr], axis=0)
            nrm = np.linalg.norm(result_normals[vi])
            if nrm > 1e-12:
                result_normals[vi] /= nrm

        return (
            result_pts.astype(np.float32),
            result_labels,
            result_intensity,
            result_curvature,
            result_normals,
        )

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        """归一化点云到单位球：平移到原点，缩放至最大半径为 1。"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        max_radius = np.max(np.linalg.norm(centered, axis=1))
        if max_radius > 1e-12:
            centered /= max_radius
        return centered

    @staticmethod
    def _normalize_with_metadata(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Normalize points while retaining the metric inverse transform."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        max_radius = float(np.max(np.linalg.norm(centered, axis=1)))
        if max_radius > 1e-12:
            centered /= max_radius
        else:
            max_radius = 1.0
        return centered, centroid.astype(np.float32), max_radius


def _compute_kdtree_curvature(
    points: np.ndarray, k_neighbors: int = 20
) -> np.ndarray:
    """P2-2: Compute local curvature using KDTree neighborhood PCA.

    对每个点，用其 k 近邻做 PCA，最小特征值与特征值总和之比作为局部曲率估计。
    不依赖网格顺序，适用于任意点排列。

    .. math::
        C_i = \\frac{\\lambda_{\\min}}{\\lambda_1 + \\lambda_2 + \\lambda_3}

    其中 :math:`\\lambda` 为邻域协方差矩阵的特征值。

    Args:
        points: 点云 (N, 3)。
        k_neighbors: 邻域点数，默认 20。

    Returns:
        曲率值 (N,) float32，范围约 [0, 1]。
    """
    from scipy.spatial import KDTree as _KDTree
    N = points.shape[0]
    k = min(k_neighbors, N - 1)
    if k < 3:
        return np.zeros(N, dtype=np.float32)

    tree = _KDTree(points)
    curvature = np.zeros(N, dtype=np.float64)

    for i in range(N):
        _, idx = tree.query(points[i], k=k + 1)  # k+1 包含自身
        neighbors = points[idx[1:]]  # 排除自身
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.dot(centered.T, centered) / (k - 1)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            # PCA 曲率 = λ_min / (λ_1 + λ_2 + λ_3)
            total = np.sum(np.abs(eigvals))
            if total > 1e-12:
                curvature[i] = np.abs(eigvals[0]) / total
        except np.linalg.LinAlgError:
            curvature[i] = 0.0

    return curvature.astype(np.float32)


def _compute_grid_curvature(z_grid: np.ndarray, grid_res: float) -> np.ndarray:
    """在规则网格上计算局部曲率（Laplacian of z）。

    使用中心有限差分计算 z 的 Laplacian ∇²z = ∂²z/∂x² + ∂²z/∂y²，
    该量在小坡度近似下正比于平均曲率。

    Args:
        z_grid: 高度场 (nx, ny)。
        grid_res: 网格分辨率 (m)。

    Returns:
        曲率场 (nx, ny)，与 z_grid 同形状。
    """
    nx, ny = z_grid.shape
    curvature = np.zeros_like(z_grid, dtype=np.float64)

    # 内部点：二阶中心差分
    dx2 = grid_res * grid_res
    curvature[1:-1, 1:-1] = (
        (z_grid[2:, 1:-1] - 2.0 * z_grid[1:-1, 1:-1] + z_grid[:-2, 1:-1]) / dx2
        + (z_grid[1:-1, 2:] - 2.0 * z_grid[1:-1, 1:-1] + z_grid[1:-1, :-2]) / dx2
    )

    # 边界：一阶差分近似
    curvature[0, :] = (z_grid[1, :] - z_grid[0, :]) / dx2
    curvature[-1, :] = (z_grid[-1, :] - z_grid[-2, :]) / dx2
    curvature[:, 0] = (z_grid[:, 1] - z_grid[:, 0]) / dx2
    curvature[:, -1] = (z_grid[:, -1] - z_grid[:, -2]) / dx2

    return curvature


if __name__ == "__main__":
    """自检：生成 5 个场景并验证输出。"""
    print("SyntheticRoadDataset Self-Test")
    print("生成 5 个场景并验证输出形状 / 标签范围 / 归一化")

    # 小尺寸测试配置
    try:
        from .config import DiseaseConfig, GeneratorConfig, RoadSurfaceConfig
    except ImportError:
        from config import DiseaseConfig, GeneratorConfig, RoadSurfaceConfig

    cfg = GeneratorConfig(
        road=RoadSurfaceConfig(
            width=2.0,
            length=2.0,
            grid_res=0.02,
        ),
        disease=DiseaseConfig(
            max_diseases_per_scene=2,
            severity_ratio=0.6,
        ),
        seed=42,
        num_points=1024,
        normalize=True,
    )

    dataset = SyntheticRoadDataset(config=cfg, dataset_size=5)
    passed = 0
    total = 5

    print(f"\n数据集大小: {len(dataset)}")
    print(f"num_points: {cfg.num_points}")
    print(f"normalize: {cfg.normalize}")
    print(f"seed: {cfg.seed}")

    pavement_counts: dict[str, int] = {"asphalt": 0, "concrete": 0}
    all_labels: set = set()

    for i in range(len(dataset)):
        print(f"\n--- Scene {i} ---")
        try:
            data = dataset[i]

            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            for key in ("points", "labels", "feats", "normals", "pavement_type"):
                assert key in data, f"Missing key: {key}"

            pts = data["points"]
            lbl = data["labels"]
            feats = data["feats"]
            nrm = data["normals"]
            ptype = data["pavement_type"]

            N = pts.shape[0]
            assert cfg.num_points == N, (
                f"points.shape[0]={N} != {cfg.num_points}"
            )
            assert pts.shape == (N, 3), f"points shape: {pts.shape}"
            assert lbl.shape == (N,), f"labels shape: {lbl.shape}"
            assert feats.shape == (N, 3), f"feats shape: {feats.shape}"
            assert nrm.shape == (N, 3), f"normals shape: {nrm.shape}"
            assert ptype in ("asphalt", "concrete"), f"pavement_type: {ptype}"

            assert pts.dtype == torch.float32, f"points dtype: {pts.dtype}"
            assert lbl.dtype == torch.int64, f"labels dtype: {lbl.dtype}"
            assert feats.dtype == torch.float32, f"feats dtype: {feats.dtype}"
            assert nrm.dtype == torch.float32, f"normals dtype: {nrm.dtype}"

            unique_labels = torch.unique(lbl).tolist()
            for ul in unique_labels:
                assert 0 <= ul < NUM_CLASSES, (
                    f"Label {ul} out of range [0, {NUM_CLASSES - 1}]"
                )
            all_labels.update(unique_labels)

            if cfg.normalize:
                centered = pts - pts.mean(dim=0)
                max_r = torch.max(torch.norm(centered, dim=1))
                assert max_r <= 1.0 + 1e-4, (
                    f"max radius after normalize: {max_r:.6f}"
                )

            assert not torch.isnan(pts).any(), "NaN in points"
            assert not torch.isinf(pts).any(), "Inf in points"
            assert not torch.isnan(nrm).any(), "NaN in normals"

            normal_norms = torch.norm(nrm, dim=1)
            assert normal_norms.min() > 0.9, "Normals too short"
            assert normal_norms.max() < 1.1, "Normals too long"

            assert feats[:, 0].min() >= 0.0, f"Intensity < 0: {feats[:, 0].min()}"
            assert feats[:, 0].max() <= 1.0, f"Intensity > 1: {feats[:, 0].max()}"

            pavement_counts[ptype] = pavement_counts.get(ptype, 0) + 1

            print(f"  Scene {i}: N={N}, labels=[{min(unique_labels)}, {max(unique_labels)}], "
                  f"pavement={ptype}, feat_range=[{feats[:, 0].min():.3f}, {feats[:, 0].max():.3f}]")
            passed += 1

        except Exception as e:
            print(f"  Scene {i} failed: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n场景总数: {total}, 通过: {passed}/{total}")

    if passed == total:
        print("全部通过")

    print(f"路面类型分布: {pavement_counts}")
    print(f"覆盖标签数: {len(all_labels)}/{NUM_CLASSES}")
    print(f"标签范围: [{min(all_labels)}, {max(all_labels)}]")

    # 验证不同场景输出不同
    print("\n检验多样性...")
    scenes = [dataset[i] for i in range(len(dataset))]
    pts_list = [s["points"] for s in scenes]
    different = any(
        not torch.allclose(pts_list[0], pts_list[j])
        for j in range(1, len(pts_list))
    )
    if different:
        print("不同场景产生不同输出")
    else:
        print("所有场景输出相同（可能种子管理有问题）")
