<div align="center">

# RoadMC

**物理约束的路面点云生成与病害分割**

合成点云 · 可观测几何特征 · Swin3D / 门控 EMA · DSCM · Muon / AdamW

[English](README.en.md)

</div>

> RoadMC 是一个研究型原型，目标是建立从物理启发的路面点云生成、逐点病害分割，到真实域诊断的可复现技术路线。当前公开的性能证据只覆盖合成数据上的二分类任务。

## 项目定位

RoadMC 包含两个同等重要的模块：

1. **点云生成**：根据路面粗糙度、微纹理、病害几何和 LiDAR 观测过程生成带标签的路面点云。
2. **模型训练**：将点坐标和传感器可观测特征输入点云分割网络，预测每个点的病害类别。

合成数据不是简单的预处理结果，而是方法的一部分。它决定了几何形态、类别覆盖、稀有病害支持度和训练数据的物理合理性。

## 方法总览

### 物理点云生成

<p align="center">
  <img src="readmeimage/synthesis_pipeline.png" alt="RoadMC physics-based point-cloud synthesis pipeline" width="92%" />
</p>

生成管线从左到右为路面先验、表面实现、病害形变、LiDAR 观测和场景导出。病害标签在几何形变阶段生成，并随观测重采样传播；受控场景的目标类别按自然 prevalence 采样，仅保证最低存活点数，且每次干预都写入审计元数据。可观测特征只从最终点云和强度计算。

### 模型训练与评估

<p align="center">
  <img src="readmeimage/training_pipeline.png" alt="RoadMC point-wise segmentation and evaluation pipeline" width="92%" />
</p>

训练管线从左到右为场景加载、按数据划分执行的抽样、输入嵌入、骨干编码、DSCM/解码以及损失和指标聚合。整体路线为：

```text
路面形貌 + 病害形变 + LiDAR 观测
        -> 带标签的场景文件 (.npz)
        -> 可观测输入特征
        -> Swin3D / 门控 EMA + DSCM
        -> 逐点 logits 与全局评估报告
```

## 当前状态

| 项目 | 当前状态 |
| --- | --- |
| 训练任务 | 二分类：背景 `0` / 病害 `1`（38 类课程迁移就绪） |
| 生成器 | 自然 prevalence 采样 + 分辨率/保护审计契约（2026-07） |
| 历史基线 | Disease IoU `0.7235`，产自已移除的固定 10% 配额生成器，**证据效力已失效** |
| 当前基线 v2 | ✅ Disease IoU `0.4781` [0.4435, 0.5105]（独立 test，阈值 0.320，ECE 0.0028） |
| 当前模型 | `Swin3D + DSCM + Muon/AdamW` |
| R2 mixing 消融 | ✅ seed 42 四变体已评估：`hc2` 最优 `0.5139`（单 seed，方向提示） |
| v3 高分辨率路线 | 诊断修复中：8192 点训练已稳定，mixing 增益待 batch 对齐后重测 |
| 自动化测试 | `112 passed / 4 skipped`（CI 全绿） |
| 真实域状态 | 已完成无标签点云域差诊断，尚无真实语义 mIoU |

## 实验结果

### 基线 v2（可信锚点）

在自然 prevalence、无标签泄漏、独立 train/val/test 三分与阈值冻结协议下重建并定稿：

| 指标 | 值 |
| --- | --- |
| **Disease IoU**（支持类 mIoU） | **0.4781** |
| scene-block bootstrap 95% CI | [0.4435, 0.5105] |
| Precision / Recall（前景） | 0.7092 / 0.5947 |
| 冻结二分类阈值 | 0.320 |
| ECE / Brier / NLL | 0.0028 / 0.1061 / 0.1927 |
| 背景 IoU | 0.9251 |
| 评估点数 / 病害支持点 | 1,751,040 / 189,075 |

旧 `0.7235` 因固定患病率 + 强度标签泄漏双重失效，仅作历史记录，不可进入任何对比。

### R2 mixing 消融（seed 42 · 20 epochs）

统一协议：`swin3d`、`muon`、`max-points 2048`、`batch 8`、`lr 1e-3`；阈值在 val 前 170 场景冻结，test 855 场景单次评估 + scene-block bootstrap 95% CI：

| Mixing | Test IoU [95% CI] | Precision | Recall | ECE | Brier | NLL |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 0.4883 [0.4542, 0.5236] | 0.7550 | 0.5803 | 0.0096 | 0.1073 | 0.1970 |
| `dscm` | 0.4817 [0.4476, 0.5168] | 0.7696 | 0.5628 | 0.0169 | 0.1072 | 0.1971 |
| **`hc2`** | **0.5139** [0.4816, 0.5478] | **0.7822** | 0.5997 | **0.0052** | **0.0994** | **0.1835** |
| `hc4` | 0.5040 [0.4702, 0.5388] | 0.7529 | **0.6039** | 0.0051 | 0.1025 | 0.1875 |

> **结论强度红线**：当前仅 seed 42 单 seed，`hc2` 领先只作方向提示；补 seed 43/44 后才能定稿。`dscm` 最弱（IoU 最低、ECE 最高）；`hc2/hc4` 校准显著优于 `none/dscm`。

### v3 高分辨率路线（5 mm 网格 · 8192 点）

修复有效 batch 崩溃后训练稳定，但协议与 v2 不同（网格 5 mm vs 2 cm、点数 8192 vs 2048、batch 4 vs 8），**不可与 `0.4781` 直接对比**：

| 变体 | disease IoU | 95% CI | ECE |
| --- | --- | --- | --- |
| `none` | **0.2738** | [0.2342, 0.3134] | 0.0624 |
| `hc2` | 0.2662 | [0.2202, 0.3121] | 0.0639 |
| `dscm` | 0.2625 | [0.2186, 0.3044] | 0.0628 |

剩余问题：ECE 0.06 仍远高于 R2 的 0.005（有效 batch 4 vs 8），mixing 增益消失（CI 全重叠），class weights 被 effective-number 公式在超大样本下中和成 ≈1.0。

## 安装

项目要求 Python `3.11+`。在项目根目录执行：

```powershell
pip install -e .
```

如果使用 `uv`：

```powershell
uv sync
```

GPU 训练需要与本机 CUDA 驱动匹配的 PyTorch。当前实验使用 RTX 5060 Laptop GPU、8 GB 显存和 PyTorch `2.11.0+cu128`。

## 点云生成

### 生成模型

RoadMC 将路面场景表示为连续表面与离散观测的组合。可将路面高程写成：

$$
z(x,y) = z_{\text{rough}}(x,y) + z_{\text{texture}}(x,y) + \Delta z_{\text{damage}}(x,y).
$$

其中：

| 层次 | 含义 | 实现位置 |
| --- | --- | --- |
| `z_rough` | 由 ISO 8608 功率谱密度控制的道路粗糙度 | `roadmc/data/synthetic/config.py` |
| `z_texture` | fBm 微纹理、局部曲率和法向变化 | `roadmc/data/synthetic/generator.py` |
| `damage` | 裂缝、坑槽、车辙、剥落、修补和接缝等形变 | `roadmc/data/synthetic/primitives.py` |
| 观测层 | 扫描线重采样、距离噪声、角度扰动、自然 prevalence 输出采样 | `roadmc/data/synthetic/generator.py` |
| 标签层 | 逐点 JTG 风格标签，支持二分类和课程迁移 | `roadmc/data/synthetic/labels.py` |

生成过程不是向规则网格简单添加随机噪声，而是先构造表面和病害，再模拟传感器采样。生成器显式区分三层分辨率：物理表面网格间距（`--surface-grid-spacing`，默认 `0.005 m`）、传感器输出点数（`--num-points`）和下游模型输入点数。毫米级表面网格不会自动保留到最终点云；每个场景的三层分辨率与保护采样记录都写入 `resolution_metadata_json` 审计字段。

### 生成数据集

```powershell
python roadmc/scripts/generate_synthetic.py `
  --train-count 2000 `
  --val-count 500 `
  --output-dir ./data/synthetic_output `
  --pavement mixed `
  --roughness B `
  --num-points 2048 `
  --workers 16
```

默认 `5 mm` 表面网格下每 worker 约需数百 MiB 峰值内存；脚本启动前会打印三层分辨率与并行内存估算，超出 `--max-parallel-memory-mib` 预算会在生成前报错。扩展已有数据集时，脚本会复用已有场景、校验既有 `metadata.json` 的 `grid_res` 一致性（避免混合分辨率），并补足缺失数量：

```powershell
python roadmc/scripts/expand_synthetic_dataset.py `
  --output-dir ./data/synthetic_output `
  --target-total 5000 `
  --num-points 2048 `
  --workers 16 `
  --pavement mixed `
  --roughness B
```

### 受控类别预算

正式实验应同时约束每类的强制场景数和有效病害点数，而不能只看场景总数：

```powershell
python roadmc/scripts/generate_class_budget.py `
  --output-dir ./data/credibility_v2 `
  --split both `
  --target-scenes-per-class 112 `
  --min-points-per-class 4000 `
  --num-points 2048 `
  --workers 16 `
  --pavement mixed `
  --roughness B

python roadmc/scripts/validate_synthetic_dataset.py `
  --data-dir ./data/credibility_v2 `
  --split both `
  --feature-check-scenes 64 `
  --output-json ./output/data_validation.json
```

类别预算脚本是可恢复的，并且只有在类别配额、点数配额和特征契约都满足后才报告完成。三个与统计效度相关的行为：

- 受控场景不再固定目标类别点数比例；每场景最低存活点数由 `ceil(min-points / target-scenes)` 自动推导，可用 `--target-label-min-output-points` 覆盖，每次保护干预都写入场景审计字段。
- 拒绝向包含无分辨率契约（旧配额时代）场景的目录续跑，防止两种采样体制静默混入同一数据集；`--ignore-legacy-scenes` 可显式排除旧场景。
- 多 worker 生成前执行聚合内存预算校验。

## 模型训练

### 输入特征契约

模型只接收推理时能够从点云本身计算的量：

```text
roadmc.observable_features.v1
[normalized_intensity, pca_curvature, signed_local_height_residual]
```

- `normalized_intensity`：归一化 LiDAR 强度。
- `pca_curvature`：局部协方差矩阵最小特征值与迹之比，即 `lambda_min / trace(C)`。
- `signed_local_height_residual`：点到局部 PCA 切平面的有符号正交残差，并按邻域支持半径归一化。

这一契约同时用于合成点云、旧 `.npz` 文件和真实点云加载器。标签不能参与特征计算，已移除旧的 `crack_boundary_dist` 标签派生通道。

### 网络与损失

| 模块 | 选择 | 作用 |
| --- | --- | --- |
| Backbone | `swin3d` | 窗口化点云 Transformer，多阶段特征提取 |
| Backbone | `pointmamba` | 门控 EMA 点序列混合（PointMamba-inspired），显存更友好 |
| DSCM | 默认开启 | 双随机通道混合（Sinkhorn，原 mHC）；CLI 仍用 `--use_mhc` / `--mixing dscm` |
| Head | per-point classifier | 输出每个点的类别 logits |
| Loss | Focal + Dice + supervised BEV Edge | 处理类别不平衡并增强边界监督 |
| Optimizer | hybrid Muon + AdamW | 矩阵参数使用 Muon，一维参数使用 AdamW |

验证和测试采用确定性均匀抽样；只有训练集使用病害分层抽样，避免通过改变验证集病害比例人为抬高 mIoU 或校准指标。

### 二分类训练

当前 RTX 5060 Laptop 的参考配置：

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/credibility_v2 `
  --label_stage binary `
  --backbone swin3d `
  --optimizer muon `
  --batch_size 4 `
  --max_points 2048 `
  --embed_dim 48 `
  --depths 1 1 2 1 `
  --num_heads 3 3 6 6 `
  --window_size 32 `
  --max_epochs 5 `
  --num_workers 4 `
  --precision 16-mixed `
  --auto_class_weights `
  --metric_min_support 500
```

Muon 的默认学习率为 `1e-2`，AdamW 的默认学习率为 `1e-3`。可用 `--lr` 覆盖。若环境不提供 Muon，则显式使用 `--optimizer adamw`。

快速诊断使用较少点数和步数：

```powershell
python roadmc/scripts/quick_diagnose.py `
  --binary `
  --backbone pointmamba `
  --steps 200 `
  --batch_size 2 `
  --max_points 1024 `
  --binary_class_weights 1.0,3.0
```

### 课程迁移到 38 类

课程标签空间为：

```text
binary -> four -> eight -> full38
```

每个阶段复用 backbone 和 DSCM 权重，重新初始化任务分类头：

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/credibility_v2 `
  --label_stage four `
  --pretrained_checkpoint ./path/to/binary.ckpt `
  --backbone swin3d `
  --optimizer muon `
  --batch_size 4 `
  --max_points 2048 `
  --max_epochs 20 `
  --num_workers 4 `
  --precision 16-mixed
```

二分类结果不能推断 38 类性能；每个多分类阶段独立报告每类支持度、per-class IoU、macro mIoU 和混淆矩阵。

## 评估与证据

评估脚本支持全局混淆矩阵、阈值扫描、ECE、Brier、NLL 和 scene-block bootstrap：

```powershell
python roadmc/evaluate.py `
  --checkpoint ./path/to/binary.ckpt `
  --data-dir ./data/credibility_v2 `
  --label-stage binary `
  --max-points 2048 `
  --scan-binary-thresholds `
  --threshold-calibration-scenes 170 `
  --bootstrap-samples 1000 `
  --output-json ./output/evaluation.json
```

**历史基线（已被取代）**：2026-07 之前的二分类证据——独立 Disease IoU `0.7235`、Precision/Recall `0.8874 / 0.7966`、ECE `0.0020`、bootstrap 95% CI `[0.7070, 0.7391]`——产自旧生成器。该版本对每个受控场景强制保留约 `10%` 的目标类别点数，验证集病害比例因此是人为固定的，IoU、校准指标和阈值选择都受其影响。此配额已移除，改为自然 prevalence 采样加最低存活审计；上述数字保留仅作历史记录，不再作为当前性能声明。基线 v2 已在自然 prevalence 数据与独立 test split 上重建并定稿：test disease IoU `0.4781`（阈值 0.32，bootstrap 95% CI `[0.4435, 0.5105]`，ECE `0.0028`），低于旧 0.7235 属预期内的纠偏而非退步——详见 `TECHNICAL_REPORT_v2.md` §3。

## 真实点云与域差诊断

真实点云加载器支持常见的 `.npy`、`.ply`、`.pcd`、`.las` 和 `.laz` 输入，并使用相同的可观测特征契约。带传感器、坐标单位、强度尺度、路段和来源信息的 JSON sidecar 可用于记录元数据。

当前使用的 M2S-RoAD 样本只有无标签 PCD，因此只用于域差诊断，不用于真实 mIoU：

```powershell
python roadmc/scripts/diagnose_domain_gap.py `
  --source-dir ./data/credibility_v2 `
  --source-kind synthetic `
  --source-split val `
  --target-dir ./data/real/m2s_road_sample `
  --target-kind real `
  --target-pattern "*.pcd" `
  --target-ground-plane `
  --max-scenes 64 `
  --output-json ./output/domain_gap.json
```

现有诊断表明，局部几何残差的差异已经较小，剩余主要差距在 LiDAR 点密度、强度分布和法向倾角——全部位于传感器观测层。因此下一步优先校准生成器的扫描密度与强度物理模型，而不是直接引入 GAN 或无监督域适配。

## 数据格式

每个场景保存为压缩 `.npz` 文件：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `points` | `(N, 3)` | XYZ 坐标（归一化后，可由中心/尺度逆变换回米制） |
| `labels` | `(N,)` | 38 类标签；课程阶段运行时映射 |
| `feats` | `(N, 3)` | 可观测特征契约中的三个通道 |
| `normals` | `(N, 3)` | 局部表面法向 |
| `pavement_type` | scalar | `asphalt`、`concrete` 或 `mixed` |
| `feature_schema` | scalar | 必须为 `roadmc.observable_features.v1` |
| `coordinate_center` / `coordinate_scale` | `(3,)` / scalar | 可逆坐标归一化参数 |
| `resolution_metadata_json` | scalar | 三层分辨率契约 + 目标类别保护审计（JSON） |
| `surface_grid_spacing_m` 等 | scalar | 表面网格间距/形状/点数、传感器输出点数、模型目标点数 |

## 38 类标签

`0` 为背景；`1-20` 为沥青路面病害；`21-37` 为水泥混凝土路面病害。

| ID | 类别 | ID | 类别 |
| --- | --- | --- | --- |
| 0 | 背景 | 1-8 | 裂缝类及严重程度 |
| 9-10 | 坑槽 | 11-12 | 松散 |
| 13-14 | 沉陷 | 15-16 | 车辙 |
| 17-18 | 波浪拥包 | 19 | 泛油 |
| 20 | 沥青修补 | 21-22 | 水泥板破碎 |
| 23-24 | 水泥裂缝 | 25-26 | 板角断裂 |
| 27-28 | 错台 | 29 | 唧泥 |
| 30-31 | 边角剥落 | 32-33 | 接缝损坏 |
| 34 | 坑洞 | 35 | 拱起 |
| 36 | 露骨 | 37 | 水泥修补 |

## 仓库结构

```text
roadmc/
  data/
    class_balance.py       # 有效类别权重
    curriculum.py          # binary -> four -> eight -> full38
    dataloader.py
    features.py            # observable feature contract
    patches.py             # 米制坐标局部 patch 切分（高分辨率路线）
    real/                  # 真实点云加载与元数据
    synthetic/             # 粗糙度、病害基元、标签和生成器
  models/
    attention/             # window attention
    backbone/              # Swin3D / 门控 EMA mixer
    gan/                   # 实验性生成器和判别器（当前冻结）
    mhc/                   # DSCM 通道混合与谱分析（原 mHC）
    model_pl.py
  scripts/                 # 生成、验证、评估和域差诊断
  domain_gap.py
  metrics.py
  train.py
  evaluate.py
  test_*.py
readmeimage/
  synthesis_pipeline.png
  training_pipeline.png
.github/
  workflows/ci.yml       # GitHub Actions：pytest 门槛 + ruff lint
```

## 已完成与路线图

### 已完成

- 移除标签派生输入，统一合成、旧数据和真实点云的可观测特征契约。
- 修复验证/测试抽样偏置，加入确定性评估、阈值校准和 bootstrap 置信区间。
- 移除受控场景的固定 10% 目标类别配额，改为自然 prevalence 采样 + 最低存活保护，保护行为逐场景写入审计元数据。
- 建立三层分辨率契约（表面网格 / 传感器输出 / 模型输入）与生成前内存预算校验，生成脚本拒绝混合采样体制或混合分辨率续跑。
- 完成米制坐标 patch 切分模块（尚未接入训练）。
- 在 RTX 5060 Laptop 8 GB 上完成带 DSCM 的二分类 GPU 验证及 4/8/38 类迁移 smoke test。
- 完成可信二分类基线 v2：自然 prevalence 数据 + 独立 train/val/test 三分，test disease IoU `0.4781` [0.4435, 0.5105]，ECE `0.0028`（旧 0.7235 双重失效，仅作历史）。
- 实现 n 流 Hyper-Connections 消融臂（`--mixing {none,dscm,hc2,hc4}`），R2 mixing 消融 seed 42 四变体已评估（`hc2` 最优 `0.5139`，方向提示）。
- 修复 v3（5 mm 网格 / 8192 点）有效 batch 崩溃，训练恢复稳定；raw_surface 密度上界数据集触发窗口占用上限（C1）修复。
- `112 passed / 4 skipped` 自动化测试通过（GitHub Actions CI 全绿）。

### 路线图

1. ~~重建可信二分类基线 v2~~ ✅ 已完成（test IoU 0.4781）。
2. R2 mixing 消融补 seed 43/44 定稿（当前 `hc2` 领先仅方向提示），再扩展骨干 × 优化器消融维。
3. 修复 v3 class weights 中和（β 调整或逆频率截断）并在 batch 对齐后重测 mixing 增益；接入 patch 管线，执行 `4096 / 8192 / 16384` 输入密度消融（bootstrap 按源场景聚合）。
4. 校准传感器层（扫描密度、强度物理模型）以收敛域差诊断中的剩余差距。
5. 获取带可靠标签、坐标单位和 JTG 映射的真实道路点云，之后再评估域随机化或域适配。

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

[English](README.en.md)

</div>
