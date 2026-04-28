好的，我仔细研究了 Qwen3.6 的第二版审查报告，结合之前我们讨论的“凑数方法重写文档”，一起整理成一份**完整的后续改进指南**。

---

# RoadMC 后续改进指南

> 整合来源：Qwen3.6 第二版代码审查 + 数学方法审计 + 之前的重写计划
> 生成日期：2026-04-28

---

## 一、改进分类说明

| 标签 | 含义 |
|:---|:---|
| ✅ 立即采纳 | 改动小、收益明确、无副作用 |
| ⏳ 延后执行 | 有道理但不急，等项目跑通后再优化 |
| 🔧 功能增强 | 让“凑数方法”真正发挥作用的重写 |
| ❌ 不采纳 | 有合理理由拒绝 |

---

## 二、Qwen3.6 第二版审查建议的处理

### ✅ 立即采纳

| 编号 | 建议 | 位置 | 改动内容 |
|:---:|:---|:---|:---|
| Q1 | 裂缝标签向量化 | `generator.py` 标签传递部分 | 将 `for i in range(len(labels))` + `set` 查询改为 `np.isin(labels, np.arange(1,9))` + 布尔索引 |
| Q2 | pydantic 配置校验 | `config.py` | 在现有 dataclass 基础上增加 `__post_init__` 校验关键参数范围（如 `dropout_rate ∈ [0,1]`、`grid_res > 0`、`max_diseases_per_scene ≥ 1`） |

**Q1 具体改动**：

```python
# 原代码（generator.py:~L620）
crack_labels = set(range(1, 9))
for i in range(len(labels)):
    if uncertain_mask[i] and labels[i] in crack_labels:
        labels[i] = 0

# 改为：
crack_mask = (labels >= 1) & (labels <= 8)
restore_mask = uncertain_mask & crack_mask
labels[restore_mask] = 0
```

### ⏳ 延后执行

| 编号 | 建议 | 位置 | 理由 |
|:---:|:---|:---|:---|
| Q3 | 标签优先级向量化 | `generator.py` 病害优先级回退 | N=65536 时性能瓶颈不明显（<5% 总耗时），且真正向量化需构建查找表而非 `np.vectorize` |
| Q4 | KDTree 批量查询 | `primitives.py:_compute_kdtree_curvature` | 曲率计算仅在生成时执行一次，非训练热路径，当前逐点查询可接受 |
| Q5 | 体素最小点数保护 | `generator.py:_voxel_downsample` | 当前多数投票在极稀疏体素下确实不稳定，但实际数据中罕见，视后续实验决定 |

**Q3 正确向量化方案（延后时参考）**：

```python
# 预构建查找表（初始化时执行一次）
max_label = max(LABEL_PRIORITY.keys()) + 1
priority_lut = np.zeros(max_label, dtype=np.int32)
for k, v in LABEL_PRIORITY.items():
    priority_lut[k] = v

# 向量化裁决（替代逐点 for 循环）
old_pri = priority_lut[old_labels]
new_pri = priority_lut[labels]
restore_mask = (old_labels != labels) & (old_pri > new_pri)
labels[restore_mask] = old_labels[restore_mask]
```

### ❌ 不采纳

| 编号 | 建议 | 理由 |
|:---:|:---|:---|
| Q6 | Sinkhorn 收敛早停 | W 初始化 `-5.0` 使 `M≈0.00005`，5 次迭代已充分收敛（误差 < 1e-3）。添加早停检查反而增加每次迭代的额外计算，且训练模式下不减少迭代次数。部署时 `deploy()` 已跳过 Sinkhorn |

---

## 三、数学方法“凑数”重写（之前讨论的计划）

| 编号 | 方法 | 当前状态 | 重写方案 | 优先级 |
|:---:|:---|:---|:---|:---|
| M1 | **可分离 2D PSD** | $S(f_x,f_y)=G_d(|f_x|)\cdot G_d(|f_y|)$ | 改为各向异性 PSD：$S(f_x,f_y)=G_d(f_r)\cdot A(\theta)^2$，$A(\theta)=1+(a-1)\cos^2\theta$，$a<1$ 使纵向更平滑 | 🔴 P0 |
| M2 | **泊松点过程** | `rng.poisson()` 仅用于生成随机整数 | 改为 Matern II 带抑制半径的泊松点过程，驱动龟裂成核 | 🟠 P1 |
| M3 | **Lévy α-stable** | 8 octave fBm 中重尾被中心极限定理稀释 | 拆分为独立函数，仅用于裂缝边缘极值剥落（2-3 octave，放宽截断） | 🟡 P2 |
| M4 | **算子半群理论** | 纯概念引用，零代码体现 | 新增 `spectral_analysis.py`：验证 60 层传播的谱范数与能量比 | 🟢 P3 |

### M1 详细方案（各向异性 PSD）

**修改文件**：`primitives.py` — `generate_road_surface`

**核心改动**：将可分离 PSD：
```python
psd_2d = np.outer(psd_x, psd_y)  # G_d(|fx|) * G_d(|fy|)
```
改为径向频率驱动的各向异性 PSD：
```python
# 径向频率
fr = np.sqrt(FX**2 + FY**2)
# 各向异性方向因子
cos_theta = np.where(fr > 0, FY / fr, 0)
anisotropy = 1.0 + (anisotropy_ratio - 1.0) * cos_theta**2
# ISO 8608 径向 PSD × 方向权重
psd_2d = psd_radial * anisotropy**2
```

**新增参数**：`anisotropy_ratio=0.5`（<1 表示纵向比横向更平滑）、`wheel_frequency=0.0`（可选轮距周期叠加）

**验证标准**：生成 5 个样本，纵向 RMS 与横向 RMS 差异 > 20%

### M2 详细方案（Matern II 成核）

**修改文件**：`primitives.py` — `add_crack` 中龟裂/块状裂缝的种子生成

**核心改动**：用带抑制半径的泊松点过程替代 `rng.uniform` 撒点：
```python
def generate_alligator_seeds(
    x_min, x_max, y_min, y_max,
    intensity: float,        # 单位面积裂缝期望数
    inhibition_radius: float, # 最小成核间距
    rng: np.random.Generator
) -> np.ndarray:
    area = (x_max - x_min) * (y_max - y_min)
    expected = int(area * intensity)
    actual = rng.poisson(expected)
    seeds = []
    while len(seeds) < actual:
        candidate = rng.uniform([x_min, y_min], [x_max, y_max])
        if all(np.linalg.norm(candidate - s) > inhibition_radius for s in seeds):
            seeds.append(candidate)
    return np.array(seeds)
```

**新增参数**：`intensity`（替代 `block_size` 间接计算）、`inhibition_radius=0.05`

**验证标准**：100 个样本中种子点最小间距 > inhibition_radius

### M3 详细方案（Lévy 极值剥落）

**修改文件**：`primitives.py` — 新增 `add_edge_spalling_heavy_tail`

**核心改动**：不再将 Lévy 用于全局 fBm 纹理，而是独立函数专门生成裂缝/坑槽边缘的极值剥落：
```python
def add_edge_spalling_heavy_tail(
    points, edge_mask, depth_base, 
    hurst=0.7, trigger_prob=0.05, rng=None
):
    """仅对边缘点中的 5% 施加 Lévy 重尾跳跃"""
    edge_idx = np.where(edge_mask)[0]
    n_trigger = max(1, int(len(edge_idx) * trigger_prob))
    triggered = rng.choice(edge_idx, size=n_trigger)
    
    alpha = max(0.5, 2.0 * hurst)
    jump = stats.levy_stable.rvs(alpha=alpha, beta=0, scale=depth_base, size=n_trigger)
    jump = np.abs(jump)  # 只取向下剥落
    jump = np.clip(jump, 0, depth_base * 3)  # 上限保护
    
    points[triggered, 2] -= jump
    return points
```

**验证标准**：有 Lévy 的裂缝边缘剥落深度峰度（kurtosis）> 3（重尾），无 Lévy 时 ≈ 0（高斯）

### M4 详细方案（谱分析脚本）

**新建文件**：`roadmc/models/mhc/spectral_analysis.py`

**功能**：
1. 对单个 MHC 模块验证谱范数 ≤ 1
2. 模拟 60 层 MHC 级联传播，输出每层的信号能量比曲线
3. 作为 pytest 测试用例，CI 中自动运行

**核心代码**：
```python
def verify_contraction_property(mhc: MHCConnection) -> bool:
    """验证单层 mHC 的谱范数 ≤ 1"""
    H = mhc.stochastic_matrix
    if H is None:
        M = F.softplus(mhc.W1) @ F.softplus(mhc.W2).T
        H = sinkhorn_knopp(M)
    sv_max = torch.linalg.svdvals(H).max()
    return bool(sv_max <= 1.0 + 1e-6)

def verify_depth_stability(dim=768, depth=60) -> dict:
    """模拟深层传播，验证能量不爆炸"""
    mhc_list = [MHCConnection(dim) for _ in range(depth)]
    x0 = torch.randn(1, dim)
    x = x0.clone()
    ratios = []
    for mhc in mhc_list:
        r = torch.randn(1, dim) * 0.1
        x = mhc(x, r)
        ratios.append(float(torch.norm(x) / torch.norm(x0)))
    return {"max_ratio": max(ratios), "min_ratio": min(ratios)}
```

---

## 四、执行建议

### 第一批（本周可完成，约 2 小时）

| 编号 | 内容 | 文件 | 预计行数 |
|:---:|:---|:---|:---:|
| Q1 | 裂缝标签向量化 | `generator.py` | 改 4 行 |
| Q2 | pydantic 配置校验 | `config.py` | 加 20 行 |

### 第二批（功能增强，约 4-6 小时）

| 编号 | 内容 | 文件 | 预计行数 |
|:---:|:---|:---|:---:|
| M1 | 各向异性 PSD | `primitives.py` | 改 ~50 行 |
| M2 | Matern II 龟裂成核 | `primitives.py` | 加 ~40 行 |

### 第三批（锦上添花，有空再做）

| 编号 | 内容 | 文件 | 预计行数 |
|:---:|:---|:---|:---:|
| M3 | Lévy 极值剥落 | `primitives.py` | 加 ~50 行 |
| M4 | 谱分析脚本 | `models/mhc/` | 新文件 ~80 行 |
| Q3 | 标签优先级向量化 | `generator.py` | 改 ~15 行 |

---

## 五、修复后的验证清单

```bash
# 1. 配置校验
python roadmc/data/synthetic/config.py

# 2. 物理基元（验证 M1/M2/M3 未破坏现有功能）
python roadmc/data/synthetic/primitives.py

# 3. 生成器（验证 Q1 改动）
python roadmc/data/synthetic/generator.py

# 4. 谱分析（M4）
python roadmc/models/mhc/spectral_analysis.py

# 5. 端到端生成测试
python -m roadmc.scripts.generate_synthetic --train-count 10 --val-count 5
```

---

以上是完整的后续改进指南，可以直接作为项目文档保存，按批次逐步执行。