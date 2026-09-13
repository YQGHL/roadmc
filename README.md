<div align="center">

# RoadMC

**物理约束的路面点云生成与病害分割**

合成点云 · 可观测特征 · Swin3D / 门控 EMA · 通道混合 · Muon / AdamW

[English](README.en.md)

</div>

> RoadMC 是一个研究型原型，建立从物理启发的路面点云生成、逐点病害分割到真实域差距诊断的可复现技术路线。当前公开的性能证据只覆盖合成数据上的二分类任务，真实域尚无带标签的分割结果。

## 项目定位

RoadMC 包含两个同等重要的模块：

1. **点云生成**：根据路面粗糙度、微纹理、病害几何和 LiDAR 观测过程生成带标签的路面点云。
2. **模型训练**：将点坐标和传感器可观测特征输入点云分割网络，预测每个点的病害类别。

合成数据不是预处理结果，而是方法的一部分。它决定几何形态、类别覆盖、稀有病害的支持度和训练分布的物理合理性。因此生成器的每一项配置都随场景落盘，可以与性能数字一起被追溯。

## 方法总览

### 物理点云生成

<p align="center">
  <img src="readmeimage/synthesis_pipeline.png" alt="RoadMC physics-based point-cloud synthesis pipeline" width="92%" />
</p>

生成管线从左到右为路面先验、表面实现、病害形变、LiDAR 观测和场景导出。病害标签在几何形变阶段生成，并随观测重采样传播；受控场景的目标类别按自然 prevalence 采样，仅保证最低存活点数，每次干预都写入审计字段。可观测特征只从最终点云和强度计算。

### 模型训练与评估

<p align="center">
  <img src="readmeimage/training_pipeline.png" alt="RoadMC point-wise segmentation and evaluation pipeline" width="92%" />
</p>

训练管线从左到右为场景加载、按数据划分执行的抽样、输入嵌入、骨干编码、通道混合与解码、损失和指标聚合。整体路线为：

```text
路面形貌 + 病害形变 + LiDAR 观测
        -> 带标签的场景文件 (.npz)
        -> 可观测输入特征
        -> Swin3D / 门控 EMA + 通道混合
        -> 逐点 logits 与全局评估报告
```

## 当前状态

| 项目 | 当前状态 |
| --- | --- |
| 训练任务 | 二分类：背景 `0` / 病害 `1`。38 类只有标签体系与课程接口，**没有任何多分类性能数字** |
| 当前结果（30 轮） | ✅ 测试集病害 IoU `0.5483` [0.5126, 0.5820]，固定阈值 0.370，ECE 0.0069 |
| 对照结果（15 轮） | 测试集病害 IoU `0.4781` [0.4435, 0.5105]，固定阈值 0.320，ECE 0.0028 |
| 历史基线 | `0.7235` 产自已移除的固定 10% 配额生成器，同时存在强度标签泄漏，**证据效力已失效** |
| 模型 | `Swin3D + DSCM + Muon/AdamW`，约 7.3M 参数 |
| 架构对比 | ✅ 5 个主流点云分割架构按同一协议跑满 30 轮（见"架构对比"） |
| 通道混合消融 | ✅ 4 个变体 × 3 个种子全部完成，变体间平均差小于组内波动，**不作机制性结论** |
| 生成器真实性选项 | ✅ 粗糙度混合、纹理沿程不均匀、缘石条带、辐射标定四项已入码，全部默认关闭，不改变 v2 行为 |
| 域差距诊断 | ✅ v2 联合 MMD `0.136`（41 帧），441 帧扩展复核 `0.1368`；开启上述选项后重建的数据集在同一规则降到 `0.0398` |
| 自动化测试 | `120 passed / 6 skipped`（GitHub Actions `ruff` + `pytest` 全绿） |
| 真实域性能 | 无真实语义 mIoU。可用的 M2S-RoAD 样本不带标签，只用于域差距诊断 |

## 实验结果

以下结果全部在合成数据、二分类、单种子（除注明外）条件下取得，不外推到真实道路或 38 类任务。

### 基线 v2 与训练长度的影响

数据为 `credibility_v2`（4160 / 855 / 855 场景，每场景 2048 点），自然 prevalence 采样、无标签泄漏、独立三分、阈值只在验证集前缀场景选定后固定：

| 指标 | 15 轮 | 30 轮 |
| --- | --- | --- |
| 验证集 IoU | 0.4608 | 0.5219 [0.4811, 0.5604] |
| **测试集病害 IoU** | **0.4781** [0.4435, 0.5105] | **0.5483** [0.5126, 0.5820] |
| 固定二分类阈值 | 0.320 | 0.370 |
| Precision / Recall（前景） | 0.7092 / 0.5947 | 0.8091 / 0.6298 |
| ECE / Brier / NLL | 0.0028 / 0.1061 / 0.1927 | 0.0069 / 0.0912 / 0.1713 |
| 背景 IoU | 0.9251 | — |
| 评估点数 / 病害支持点 | 1,751,040 / 189,075 | 同左 |

两批除训练轮数（以及随之拉伸的余弦周期 `T_max = max_epochs`）外超参一致，可直接比较。30 轮相对 15 轮提升 `+0.070` 绝对（`+14.7%`），说明 15 轮结果在训练结束时仍未收敛。**报告任何 IoU 都必须同时给出训练轮数**，否则两个数字会被当成同一实验。

旧 `0.7235` 因固定患病率与强度标签泄漏双重失效，仅作历史记录，不进入任何对比。

### 架构对比

5 个主流点云分割架构与本文模型在同一份 v2 数据、同一训练与评估协议下各跑 30 轮。检查点按两种规则选择：规则 A 为验证集 `argmax@0.5` IoU 最优，规则 B 为验证集前缀 170 场景的标定 IoU 最优。

| 模型 | 参数量 | 测试 IoU（规则 B，95% CI） | 测试 IoU（规则 A） | 规则 B ECE | 备注 |
| --- | --- | --- | --- | --- | --- |
| PointNet | 1.67M | 0.0998 [0.0873, 0.1123] | 0.1118 | 0.2029 | 规则 A 选中的检查点来自第 1 轮（规则 A ECE 0.3164） |
| PointNet++ | 0.36M | 0.2772 [0.2490, 0.3067] | 0.2695 | 0.0799 | — |
| DGCNN | 0.32M | 0.3831 [0.3603, 0.4052] | 0.3830 | 0.0218 | — |
| PointMLP | 4.22M | 0.1805 [0.1644, 0.1963] | 0.1746 | 0.1474 | — |
| Point Transformer v1 | 1.46M | 0.5291 [0.4910, 0.5661] | 0.5458 | 0.0097 | — |
| 本文模型（Swin3D + DSCM） | 7.3M | 0.5483 [0.5126, 0.5820] | 0.5483 | 0.0069 | 两种规则选中同一检查点 |

两点结论与一点保留：

- 层级窗口注意力骨干优于纯点式与图卷积网络：PointNet、PointNet++、DGCNN、PointMLP 四者的规则 B 区间上界（最高 0.4052）都低于本文模型区间下界 0.5126。
- Point Transformer v1 是唯一接近的基线：规则 B 0.5291 [0.4910, 0.5661] 对 0.5483 [0.5126, 0.5820]、规则 A 0.5458 对 0.5483，**区间重叠，现有证据不足以支持"本文模型显著领先"**。
- 规则 A 与规则 B 在弱模型上会给出不同答案（PointNet 的规则 A 检查点退化到第 1 轮），因此两种规则都报告；只报一种会掩盖检查点选择规则本身的问题。

### 通道混合消融

`--mixing` 的四个取值各跑 3 个种子（20 轮协议，测试集 IoU）：

| 变体 | seed 42 | seed 43 | seed 44 | 均值 ± 标准差 |
| --- | --- | --- | --- | --- |
| `none`（关闭混合） | 0.4883 | 0.5040 | 0.5077 | 0.5000 ± 0.0103 |
| `hc2`（2 流） | 0.5139 | 0.5203 | 0.4959 | 0.5100 ± 0.0126 |
| `dscm` | 0.4817 | 0.5003 | 0.5214 | 0.5011 ± 0.0199 |
| `hc4`（4 流） | 0.5040 | 0.5184 | 0.5173 | 0.5132 ± 0.0080 |

四个变体的均值落在 0.5000–0.5132，带宽 0.013，而组内标准差最大 0.0199；seed 42 上 `hc2` 领先 `none` 0.026，在 seed 44 上反转为落后 0.012。**结论是通道混合模块的增益未达显著**，性能主要来自层级窗口注意力骨干。单种子消融在这里会直接给出错误的排序，本项目因此把"多种子"作为机制性结论的门槛。

### 与真实路面的差距

真实数据为 M2S-RoAD 的无标签 PCD 帧（OS1-128）。差距用 6 个只依赖点云本身、合成与真实两侧同法计算的描述子衡量：点密度、归一化强度、法向倾角、PCA 曲率、局部高度残差、按邻域半径归一的高度残差。

| 对比 | 场景数 | 联合 RBF-MMD |
| --- | --- | --- |
| v2 合成 vs 真实（修正前） | 855 × 41 | 0.458 |
| v2 合成 vs 真实（修正后） | 855 × 41 | **0.136** |
| v2 合成 vs 真实（扩展复核） | 855 × 441 | 0.1368 |
| 真实性配置数据集 vs 真实 | 858 × 440 | **0.0398** |

上表统一使用 `0.15 m` 的主导平面阈值。把两侧阈值同时收紧到 `0.02 m`（只看更贴近路面层的点），对应读数为 v2 `0.1410`、真实性配置数据集 `0.0337`，改善方向一致。

前三行的差异全部来自诊断方法本身的三类缺陷（法向来源不一致、强度归一化按错误量程、采样窗口不匹配），修正后剩余差距集中在点密度、强度分布和法向倾角，都属于传感器观测层。最后一行是在生成器上开启粗糙度混合、纹理不均匀、缘石与辐射标定四项配置后重建数据集的复测结果，同一套描述子与流程，未改动诊断代码。

数据侧发现（影响所有使用 M2S-RoAD 的工作）：部分会话把无回波写成 `range == 0`，且这些点在多数帧里**不落在原点**，因此不会被"剔除原点附近无效回波"的常规过滤捕获，会同时污染 RANSAC 平面拟合与 k 近邻密度估计。本仓库的报数在计算描述子之前先按 `range` 字段剔除这些点；加载器侧的统一过滤尚未合入，因为它会让已有的 41 帧读数整体变化。

### 已作废的 v3 高分辨率记录

下表产自 2026-08 的"5 mm 网格 / 8192 点"路线（有效批次修复后训练稳定），协议与 v2 不同（网格 5 mm 对 2 cm、点数 8192 对 2048、批次 4 对 8），**不可与 `0.4781` 或 `0.5483` 比较**。对应的数据集目录已改名留存为 `credibility_v3_legacy_16384pt`，当前 `credibility_v3` 指向另一套配置，两者无关。

| 变体 | disease IoU | 95% CI | ECE |
| --- | --- | --- | --- |
| `none` | 0.2738 | [0.2342, 0.3134] | 0.0624 |
| `hc2` | 0.2662 | [0.2202, 0.3121] | 0.0639 |
| `dscm` | 0.2625 | [0.2186, 0.3044] | 0.0628 |

保留的理由是该批次暴露的三个问题仍未解决，且在新配置下重跑前需要一并处理：ECE 0.06 远高于 2048 点批次的 0.005（有效批次差异）、类别权重被 effective-number 公式在超大样本下中和到约 1.0、混合作用在批次对齐后消失。

## 安装

项目要求 Python `3.11+`。在项目根目录执行：

```powershell
pip install -e .
```

读取真实点云需要可选依赖：

```powershell
pip install -e ".[real]"   # laspy / plyfile，用于 .las/.laz/.ply
pip install -e ".[pcd]"     # open3d，用于 .pcd
```

如果使用 `uv`：

```powershell
uv sync
```

GPU 训练需要与本机 CUDA 驱动匹配的 PyTorch。当前实验使用 RTX 5060 Laptop GPU、8 GB 显存。`open3d` 只发布到 Python 3.12 的轮子，需要它的脚本不要建在 3.13+ 环境里。

## 点云生成

### 生成模型

RoadMC 将路面场景表示为连续表面与离散观测的组合：

$$
z(x,y) = z_{\text{rough}}(x,y) + z_{\text{texture}}(x,y) + \Delta z_{\text{damage}}(x,y).
$$

| 层次 | 含义 | 实现位置 |
| --- | --- | --- |
| `z_rough` | 由 ISO 8608 功率谱密度控制的道路粗糙度 | `roadmc/data/synthetic/config.py` |
| `z_texture` | fBm 微纹理、局部曲率和法向变化 | `roadmc/data/synthetic/generator.py` |
| `damage` | 裂缝、坑槽、车辙、松散、修补、接缝等 9 类病害形变 | `roadmc/data/synthetic/primitives.py` |
| 观测层 | 扫描线重采样、距离噪声、角度扰动、辐射强度、按自然 prevalence 的输出采样 | `roadmc/data/synthetic/generator.py` |
| 标签层 | 逐点 JTG 风格标签，支持二分类和课程迁移 | `roadmc/data/synthetic/labels.py` |

生成过程不是向规则网格添加随机噪声，而是先构造表面与病害，再模拟传感器采样。生成器显式区分三层分辨率：物理表面网格间距（`generate_synthetic.py` 用 `--surface-grid-spacing`，类别预算脚本用 `--grid-res`，v2 取 `0.02 m`）、传感器输出点数（`--num-points`）、下游模型输入点数。毫米级表面网格不会自动保留到最终点云，每场景的三层分辨率与保护采样记录都写入 `resolution_metadata_json`。

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

默认表面网格下每 worker 峰值内存约数百 MiB；脚本启动前打印三层分辨率与并行内存估算，超出 `--max-parallel-memory-mib` 预算会在生成前报错。扩展已有数据集时，脚本复用已有场景、校验既有 `grid_res` 一致性以避免混合分辨率，并补足缺失数量：

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

类别预算脚本可断点续跑，只有在类别配额、点数配额和特征校验都满足后才报告完成。三个与统计效度相关的行为：

- 受控场景不固定目标类别点数比例；每场景最低存活点数由 `ceil(min-points / target-scenes)` 推导，可用 `--target-label-min-output-points` 覆盖，每次保护干预写入场景审计字段。
- 拒绝向缺少分辨率记录（旧配额时代）场景的目录续跑，防止两种采样体制混入同一数据集；`--ignore-legacy-scenes` 可显式排除旧场景。**注意续跑只核对配额、不核对生成配置**：同一目录下若已存在另一套配置产出的场景，配额会被判为满足。跨配置重建时应换输出目录，或先确认旧目录已改名。
- 多 worker 生成前执行聚合内存预算校验。

### 贴近真实路面的配置项

下面四组选项用于缩小合成侧与真实侧的观测差距，全部默认关闭；关闭时几何与强度采样与 v2 相同（新增的 `roughness_class` 等记录字段除外，它总是写入）：

| 选项 | CLI | 作用 |
| --- | --- | --- |
| 粗糙度混合 | `--roughness-mix "D:0.5,E:0.5"` | 每场景按权重抽样 ISO 8608 等级，并把实际等级写入场景字段 `roughness_class`，避免整批数据只用一个等级 |
| 纹理沿程不均匀 | `--texture-het-sigma 0.6 --texture-het-corr 2.0` | 对已按 Parseval 定标的短波分量做对数正态幅值调制（先做高低通分解，不改变总方差），使纹理强弱沿里程起伏 |
| 缘石条带 | `--curb-width 0.30 --curb-height 0.13` | 在行车道外加一条抬高带，让非路面点进入密度与法向统计 |
| 辐射标定 | `--albedo-sigma-ln 0.9 --speckle-m 6.0` | 强度改为 `反照率场 × 入射角 × 1/R²` 后做对数正态与 Gamma 标定，强度分布不再整体偏亮 |

一条完整的重建命令（本文"与真实路面的差距"表末行的配置）：

```powershell
python roadmc/scripts/generate_class_budget.py `
  --output-dir ./data/credibility_v3 `
  --split all --labels all `
  --target-scenes-per-class 112 --min-points-per-class 4000 `
  --val-ratio 0.2 --test-ratio 0.2 --wave-size 4 `
  --grid-res 0.02 --num-points 4096 --pavement mixed `
  --roughness D --roughness-mix "D:0.5,E:0.5" `
  --texture-rms 0.0008 --texture-het-sigma 0.6 --texture-het-corr 2.0 `
  --curb-width 0.30 --curb-height 0.13 `
  --albedo-sigma-ln 0.9 --speckle-m 6.0 `
  --max-diseases 2 `
  --workers 16 --seed 42
```

这套配置把联合 MMD 从 0.136 量级压到 0.034，代价是曲率与高度残差在匹配 ROI 下偏大：真实路面表现为"长波起伏多、短波弯曲少"，单一谱形同时命中两种取样窗口仍做不到。剩余可见差距主要是点密度（合成侧约 115 点/m²，真实侧约 156 点/m²）。

## 模型训练

### 可观测输入特征

模型只接收推理时能够从点云本身计算的量：

```text
roadmc.observable_features.v1
[normalized_intensity, pca_curvature, signed_local_height_residual]
```

- `normalized_intensity`：归一化 LiDAR 强度。
- `pca_curvature`：局部协方差矩阵最小特征值与迹之比，即 `lambda_min / trace(C)`。
- `signed_local_height_residual`：点到局部 PCA 切平面的有符号正交残差，按邻域支持半径归一化。

同一条约定同时用于合成点云、旧 `.npz` 文件与真实点云加载器，并由 `require_observable_checkpoint_schema` 在加载检查点时校验。标签不参与特征计算，已移除旧的 `crack_boundary_dist` 标签派生通道。任何对点云重采样或子集化的代码路径，都必须在交给模型的那一份点云上重算特征，因为邻域集合已经改变。

### 网络与损失

| 模块 | 选择 | 作用 |
| --- | --- | --- |
| Backbone | `swin3d` | 窗口化点云 Transformer，多阶段特征提取 |
| Backbone | `pointmamba` | 门控 EMA 点序列混合（PointMamba-inspired），显存更友好 |
| 通道混合 | `--mixing {none,dscm,hc2,hc4}` | DSCM 为双随机通道混合（Sinkhorn，旧名 mHC）；`hc2`/`hc4` 是 2 流与 4 流残差混合消融臂。`--use_mhc` / `--no_mhc` 作为旧别名保留 |
| Head | per-point classifier | 输出每个点的类别 logits |
| Loss | Focal + Dice + supervised BEV Edge | 处理类别不平衡并增强边界监督 |
| Optimizer | hybrid Muon + AdamW | 矩阵参数使用 Muon，一维参数使用 AdamW |

验证和测试采用确定性均匀抽样，只有训练集使用病害分层抽样，避免通过改变验证集病害比例人为抬高 mIoU 或校准指标。

### 二分类训练

8 GB 显存上的参考配置：

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

两处容易踩的默认值：

- Muon 的默认学习率是 `1e-2`，AdamW 是 `1e-3`。**跑本文模型必须显式写 `--lr 1e-3`**，否则训练损失会发散为 NaN。
- 默认 `--embed_dim 96` 对应约 26.9M 参数，上表的 7.3M 模型需要显式 `--embed_dim 48`。
- 每场景 4096 点时批次 8 会在 8 GB 上溢出；与 `8 × 2048` 等 token 量的写法是 `--batch_size 4 --gradient_checkpointing`，并设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

如果环境不提供 Muon，改用 `--optimizer adamw`。快速诊断用较少点数和步数：

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

每个阶段复用骨干和通道混合权重，重新初始化任务分类头：

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

二分类结果不能推断 38 类性能；每个多分类阶段独立报告每类支持度、per-class IoU、macro mIoU 和混淆矩阵。**本仓库尚未发布任何 38 类数字。**

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

报告一个数字时必须同时说明三件事：

1. **检查点规则**。规则 A 用验证集 `argmax@0.5` 的 IoU 选点，规则 B 用验证集前缀 170 场景的标定 IoU 选点。规则 A 隐含"病害概率会跨过 0.5"的前提，弱模型上会退化到选中第一轮检查点。
2. **阈值来源**。阈值只在验证集前缀场景上扫描选定，之后固定，测试集单次评估。
3. **训练轮数**。见上一节，15 轮与 30 轮的差是 0.070。

**历史基线（已被取代）**：2026-07 之前的二分类证据——独立测试集病害 IoU `0.7235`、Precision/Recall `0.8874 / 0.7966`、ECE `0.0020`、bootstrap 95% CI `[0.7070, 0.7391]`——产自旧生成器。该版本对每个受控场景强制保留约 `10%` 的目标类别点数，验证集病害比例人为固定，IoU、校准指标和阈值选择都受影响；同时强度通道存在标签泄漏。此配额已移除，改为自然 prevalence 采样加最低存活审计。上述数字保留仅作历史记录，不再作为当前性能声明。基线 v2 已在自然 prevalence 数据与独立测试划分上重建：`0.4781`（15 轮）与 `0.5483`（30 轮），低于旧 `0.7235` 属预期内的纠偏而非退步。

## 真实点云与域差距诊断

真实点云加载器支持 `.npy`、`.ply`、`.pcd`、`.las`、`.laz`，使用与合成侧相同的可观测特征约定。带传感器、坐标单位、强度量程、路段和来源信息的 JSON sidecar 用于记录元数据；`--require-real-metadata` 会在缺少元数据时直接报错，而不是按猜测的强度量程归一化。

```powershell
python roadmc/scripts/diagnose_domain_gap.py `
  --source-dir ./data/credibility_v2 `
  --source-kind synthetic `
  --source-split val `
  --target-dir ./data/real/m2s_road_sample `
  --target-kind real `
  --target-pattern "*.pcd" `
  --target-ground-plane `
  --source-ground-plane `
  --normal-source pca `
  --skip-non-ground-frames `
  --max-points-per-scene 4096 `
  --k-neighbors 16 `
  --max-scenes 64 `
  --output-json ./output/domain_gap.json
```

参数中有四个决定了结论是否可比，改动其中任何一个都会让历史读数失效：

- `--normal-source`：合成侧与真实侧必须用同一方式求法向。默认 `pca` 表示两侧都由 k 近邻协方差估计；混合两侧来源（例如合成用解析法向、真实用 PCA）会把法向倾角差距放大到不可解释的量级。
- `--source-ground-plane` / `--target-ground-plane` 与 `--ground-distance-threshold`：主导平面剔除的范围决定统计对象是"路面"还是"路面加路侧地物"，两侧必须用同一阈值。
- `--skip-non-ground-frames`：剔除主导平面不是地面的帧，否则整帧几何统计来自路侧物体。
- `--max-points-per-scene` 与 `--k-neighbors`：密度与曲率都是近邻尺度的函数，两侧点数不同则同一描述子含义不同。

诊断脚本输出 6 个描述子的两侧分布、Wasserstein-1 距离、能量距离与 RBF-MMD，以及逐帧合并后的联合 MMD。历史诊断结论是：局部几何残差的差异已经较小，剩余差距在点密度、强度分布和法向倾角，全部属于传感器观测层，因此优先级是校准扫描密度与强度物理模型，而不是引入 GAN 或无监督域适配。

## 数据格式

每个场景保存为压缩 `.npz`：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `points` | `(N, 3)` | XYZ 坐标（归一化后，可由中心/尺度逆变换回米制） |
| `labels` | `(N,)` | 38 类标签；课程阶段运行时映射 |
| `feats` | `(N, 3)` | 可观测特征的三个通道 |
| `normals` | `(N, 3)` | 局部表面法向 |
| `pavement_type` | scalar | `asphalt`、`concrete` 或 `mixed` |
| `roughness_class` | scalar | 该场景实际抽到的 ISO 8608 等级（开启粗糙度混合时逐场景不同） |
| `target_label` | scalar | 受控场景强制注入的类别，非受控为 `-1` |
| `feature_schema` | scalar | 必须为 `roadmc.observable_features.v1` |
| `coordinate_center` / `coordinate_scale` | `(3,)` / scalar | 可逆坐标归一化参数 |
| `resolution_metadata_json` | scalar | JSON，含三层分辨率、保护采样审计和强度模型描述串（`sensor_output.intensity_model`） |
| `surface_grid_spacing_m` 等 | scalar | 表面网格间距/形状/点数、传感器输出点数、模型目标点数 |

## 38 类标签

`0` 为背景；`1-20` 为沥青路面病害；`21-37` 为水泥混凝土路面病害。病害形变基元为 9 类，映射到 38 类标签空间后由课程接口按阶段合并。

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
    features.py            # 可观测特征定义与校验
    patches.py             # 米制坐标局部 patch 切分（尚未接入训练）
    real/                  # 真实点云加载与元数据
    synthetic/             # 粗糙度、病害基元、标签和生成器
  models/
    attention/             # window attention
    backbone/              # Swin3D / 门控 EMA mixer
    gan/                   # 实验性生成器和判别器（当前冻结）
    mhc/                   # DSCM 通道混合与谱分析
    model_pl.py
  scripts/                 # 生成、验证、评估、域差距诊断
  domain_gap.py            # 描述子与距离（合成/真实两侧同法计算）
  metrics.py
  train.py
  evaluate.py
  test_*.py
readmeimage/
  synthesis_pipeline.png
  training_pipeline.png
.github/
  workflows/ci.yml         # GitHub Actions：pytest 门槛 + ruff lint
```

## 已完成与路线图

### 已完成

- 移除标签派生输入，统一合成、旧数据和真实点云的可观测特征定义。
- 修复验证/测试抽样偏置，加入确定性评估、阈值校准和 bootstrap 置信区间。
- 移除受控场景的固定 10% 目标类别配额，改为自然 prevalence 采样加最低存活保护，保护行为逐场景写入审计字段。
- 建立三层分辨率定义（表面网格 / 传感器输出 / 模型输入）与生成前内存预算校验，生成脚本拒绝混合采样体制或混合分辨率续跑。
- 完成米制坐标 patch 切分模块（尚未接入训练）。
- 在 8 GB 显存上完成带通道混合的二分类 GPU 验证及 4/8/38 类迁移 smoke test。
- 完成可信二分类基线 v2：15 轮 `0.4781` [0.4435, 0.5105]，同协议 30 轮 `0.5483` [0.5126, 0.5820]。
- 实现 n 流残差混合消融臂（`--mixing {none,dscm,hc2,hc4}`），并完成 4 变体 × 3 种子，据此把通道混合的增益判定为不显著。
- 完成 5 个主流点云分割架构的同协议 30 轮对比，并明确检查点两种选择规则在弱模型上的分歧。
- 修正域差距诊断中三类导致读数不可比的缺陷（法向来源、强度量程、采样窗口），联合 MMD 由 0.458 降到 0.136，并在 441 帧的扩展子集上复现。
- 在生成器上加入四项真实性配置（粗糙度混合、纹理沿程不均匀、缘石条带、辐射标定），默认关闭；开启后重建数据集的联合 MMD 降到 0.0337。
- 定位并规避 M2S-RoAD 的 `range == 0` 无效回波编码问题。
- 自动化测试 `120 passed / 6 skipped`，GitHub Actions CI 全绿。

### 路线图

1. 在新真实性配置的数据集上跑完架构对比与本文模型的完整测试，再决定这套配置是否进入正式协议。当前该批次尚无任何性能数字。
2. 修复类别权重在大数据量下被中和的问题（β 调整或逆频率截断），并让混合消融在批次对齐后重测。
3. 接入 patch 管线，执行 `4096 / 8192 / 16384` 输入密度对比，bootstrap 按源场景聚合而不是按 patch。
4. 继续收敛传感器层：点密度（115 对 156 点/m²）以及匹配取样窗口下偏大的曲率与高度残差。
5. 把加载器侧的 `range == 0` 过滤正式合入，并重跑全部域差距读数。
6. 获取带可靠标签、坐标单位和 JTG 映射的真实道路点云，之后再评估域随机化或域适配。

## 引用口径

- 二分类当前可引用的结果：30 轮 `0.5483` [0.5126, 0.5820]，或注明轮数的 15 轮 `0.4781` [0.4435, 0.5105]。
- `0.7235` 不得作为当前结果引用。
- 5 个架构的对比中，本文模型与 Point Transformer v1 的置信区间重叠，不得写成显著领先。
- 通道混合的增益未跨种子复现，不得写成模块有效性证据。
- 38 类没有任何分割数字。
- 不同数据网格、点数、批次或训练轮数下的结果不可并列比较。

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

[English](README.en.md)

</div>
