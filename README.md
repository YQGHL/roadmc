<div align="center">

# RoadMC

**物理约束的路面点云生成与病害分割**

合成点云 · 可观测几何特征 · Swin3D / PointMamba · mHC · Muon / AdamW

[English](README.en.md)

</div>

> RoadMC 是一个研究型原型，目标是建立从物理启发的路面点云生成、逐点病害分割，到真实域诊断的可复现技术路线。当前公开的性能证据只覆盖合成数据上的二分类任务。

## 项目定位

RoadMC 包含两个同等重要的模块：

1. **点云生成**：根据路面粗糙度、微纹理、病害几何和 LiDAR 观测过程生成带标签的路面点云。
2. **模型训练**：将点坐标和传感器可观测特征输入点云分割网络，预测每个点的病害类别。

合成数据不是简单的预处理结果，而是方法的一部分。它决定了几何形态、类别覆盖、稀有病害支持度和训练数据的物理合理性。

## 方法总览

### A. 物理点云生成

<p align="center">
  <img src="readmeimage/synthesis_pipeline.png" alt="RoadMC physics-based point-cloud synthesis pipeline" width="96%" />
</p>

图 A 从左到右描述路面先验、表面实现、病害形变、LiDAR 观测和场景导出。病害标签在几何形变阶段生成，并随观测重采样传播；可观测特征只从最终点云和强度计算。

### B. 模型训练与评估

<p align="center">
  <img src="readmeimage/training_pipeline.png" alt="RoadMC point-wise segmentation and evaluation pipeline" width="96%" />
</p>

图 B 从左到右描述场景加载、按数据划分执行的抽样、输入嵌入、骨干编码、mHC/解码以及损失和指标聚合。两张图只表达数据流和模块边界；生成理论、特征定义和实验口径分别在下文说明。整体路线为：

```text
路面形貌 + 病害形变 + LiDAR 观测
        -> 带标签的场景文件 (.npz)
        -> 可观测输入特征
        -> Swin3D / PointMamba + mHC
        -> 逐点 logits 与全局评估报告
```

## 当前状态

| 项目 | 当前状态 |
| --- | --- |
| 训练任务 | 二分类：背景 `0` / 病害 `1` |
| 合成数据 | `4,995` 个场景，训练 `4,144`，验证 `851` |
| 场景点数 | 每场景 `2,048` 点 |
| 当前模型 | `Swin3D + mHC + Muon/AdamW` |
| 独立合成评估 | Disease IoU / 支持类 mIoU `0.7235` |
| Bootstrap 95% CI | `[0.7070, 0.7391]` |
| 自动化测试 | `34/34` 通过 |
| 真实域状态 | 已完成无标签点云域差诊断，尚无真实语义 mIoU |

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

## 一、点云生成

### 1.1 生成模型

RoadMC 将路面场景表示为连续表面与离散观测的组合。可将路面高程写成：

\[
z(x,y) = z_{\text{rough}}(x,y) + z_{\text{texture}}(x,y) + \Delta z_{\text{damage}}(x,y).
\]

其中：

| 层次 | 含义 | 实现位置 |
| --- | --- | --- |
| `z_rough` | 由 ISO 8608 功率谱密度控制的道路粗糙度 | `roadmc/data/synthetic/config.py` |
| `z_texture` | fBm 微纹理、局部曲率和法向变化 | `roadmc/data/synthetic/generator.py` |
| `damage` | 裂缝、坑槽、车辙、剥落、修补和接缝等形变 | `roadmc/data/synthetic/primitives.py` |
| 观测层 | 扫描线重采样、距离噪声、角度扰动和密度控制 | `roadmc/data/synthetic/generator.py` |
| 标签层 | 逐点 JTG 风格标签，支持二分类和课程迁移 | `roadmc/data/synthetic/labels.py` |

生成过程不是向规则网格简单添加随机噪声，而是先构造表面和病害，再模拟传感器采样。这样可以同时控制形状、标签、密度和噪声来源。

### 1.2 快速生成数据

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

扩展已有数据集时，脚本会复用已有场景并补足缺失数量：

```powershell
python roadmc/scripts/expand_synthetic_dataset.py `
  --output-dir ./data/synthetic_output `
  --target-total 5000 `
  --num-points 2048 `
  --workers 16 `
  --pavement mixed `
  --roughness B
```

Windows 环境建议先使用 `16` 个 worker。更高并行度可能增加页面文件和内存压力。

### 1.3 受控类别预算

正式实验应同时约束每类的强制场景数和有效病害点数，而不能只看场景总数：

```powershell
python roadmc/scripts/generate_class_budget.py `
  --output-dir ./data/credibility_v1_5k `
  --split both `
  --target-scenes-per-class 112 `
  --min-points-per-class 4000 `
  --grid-res 0.02 `
  --num-points 2048 `
  --workers 16 `
  --pavement mixed `
  --roughness B

python roadmc/scripts/validate_synthetic_dataset.py `
  --data-dir ./data/credibility_v1_5k `
  --split both `
  --feature-check-scenes 64 `
  --output-json ./output/data_validation.json
```

类别预算脚本是可恢复的，并且只有在类别配额、点数配额和特征契约都满足后才报告完成。

## 二、模型训练

### 2.1 输入特征契约

模型只接收推理时能够从点云本身计算的量：

```text
roadmc.observable_features.v1
[normalized_intensity, pca_curvature, signed_local_height_residual]
```

- `normalized_intensity`：归一化 LiDAR 强度。
- `pca_curvature`：局部协方差矩阵最小特征值与迹之比，即 `lambda_min / trace(C)`。
- `signed_local_height_residual`：点到局部 PCA 切平面的有符号正交残差，并按邻域支持半径归一化。

这一契约同时用于合成点云、旧 `.npz` 文件和真实点云加载器。标签不能参与特征计算，已移除旧的 `crack_boundary_dist` 标签派生通道。

### 2.2 网络与损失

| 模块 | 选择 | 作用 |
| --- | --- | --- |
| Backbone | `swin3d` | 窗口化点云 Transformer，多阶段特征提取 |
| Backbone | `pointmamba` | Morton 顺序点序列混合，显存更友好 |
| mHC | 默认开启 | Sinkhorn 风格通道混合，改善深层特征流动 |
| Head | per-point classifier | 输出每个点的类别 logits |
| Loss | Focal + Dice + supervised BEV Edge | 处理类别不平衡并增强边界监督 |
| Optimizer | hybrid Muon + AdamW | 矩阵参数使用 Muon，一维参数使用 AdamW |

验证和测试采用确定性均匀抽样；只有训练集使用病害分层抽样。这样可以避免通过改变验证集病害比例人为抬高 mIoU 或校准指标。

### 2.3 二分类训练

当前 RTX 5060 Laptop 的正式配置：

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/credibility_v1_5k `
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

### 2.4 课程迁移到 38 类

课程标签空间为：

```text
binary -> four -> eight -> full38
```

每个阶段复用 backbone 和 mHC 权重，重新初始化任务分类头：

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/credibility_v1_5k `
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

二分类结果不能推断 38 类性能。每个多分类阶段都必须独立报告每类支持度、per-class IoU、macro mIoU 和混淆矩阵。

## 三、评估与证据

评估脚本支持全局混淆矩阵、阈值扫描、ECE、Brier、NLL 和 scene-block bootstrap：

```powershell
python roadmc/evaluate.py `
  --checkpoint ./path/to/binary.ckpt `
  --data-dir ./data/credibility_v1_5k `
  --label-stage binary `
  --max-points 2048 `
  --scan-binary-thresholds `
  --threshold-calibration-scenes 170 `
  --bootstrap-samples 1000 `
  --output-json ./output/evaluation.json
```

当前中等规模合成证据：

- 训练集 `4,144` 个场景，验证集 `851` 个场景，每场景 `2,048` 点。
- 前 `170` 个验证场景用于阈值选择，剩余 `681` 个场景用于独立评估。
- 独立 Disease IoU / 支持类 mIoU：`0.7235`。
- Precision / Recall：`0.8874 / 0.7966`。
- ECE：`0.0020`。
- scene-block bootstrap 95% CI：`[0.7070, 0.7391]`。

这些结果只说明模型在当前物理启发合成分布上的二分类能力，不代表真实道路域或 38 类任务的最终性能。

## 四、真实点云与域差诊断

真实点云加载器支持常见的 `.npy`、`.ply`、`.pcd`、`.las` 和 `.laz` 输入，并使用相同的可观测特征契约。带传感器、坐标单位、强度尺度、路段和来源信息的 JSON sidecar 可用于记录元数据。

当前使用的 M2S-RoAD 样本只有无标签 PCD，因此只用于域差诊断，不用于真实 mIoU：

```powershell
python roadmc/scripts/diagnose_domain_gap.py `
  --source-dir ./data/credibility_v1_5k `
  --source-kind synthetic `
  --source-split val `
  --target-dir ./data/real/m2s_road_sample `
  --target-kind real `
  --target-pattern "*.pcd" `
  --target-ground-plane `
  --max-scenes 64 `
  --output-json ./output/domain_gap.json
```

现有诊断表明，局部几何残差的差异已经较小，剩余主要问题是 LiDAR 点密度、强度分布和法向倾角。真实标签和单位确认之前，不将域适配结果写成真实域性能。

## 五、数据格式

每个场景保存为压缩 `.npz` 文件：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `points` | `(N, 3)` | XYZ 坐标 |
| `labels` | `(N,)` | 38 类标签；课程阶段运行时映射 |
| `feats` | `(N, 3)` | 可观测特征契约中的三个通道 |
| `normals` | `(N, 3)` | 局部表面法向 |
| `valid_mask` | `(N,)` | padding 有效点掩码 |
| `pavement_type` | scalar | `asphalt`、`concrete` 或 `mixed` |
| `feature_schema` | scalar | 必须为 `roadmc.observable_features.v1` |

## 六、38 类标签

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

## 七、仓库结构

```text
roadmc/
  data/
    class_balance.py       # 有效类别权重
    curriculum.py          # binary -> four -> eight -> full38
    dataloader.py
    features.py            # observable feature contract
    real/                  # 真实点云加载与元数据
    synthetic/             # 粗糙度、病害基元、标签和生成器
  models/
    attention/             # window attention
    backbone/              # Swin3D / PointMamba
    gan/                   # 实验性生成器和判别器
    mhc/                   # mHC 与谱分析
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
```

## 八、已完成与下一步

### 已完成

- 移除标签派生输入，统一合成、旧数据和真实点云的可观测特征契约。
- 修复验证/测试抽样偏置，加入确定性评估、阈值校准和 bootstrap 置信区间。
- 支持 38 类物理可达性、类别预算、自动类别权重和课程迁移。
- 在 RTX 5060 Laptop 8 GB 上完成带 mHC 的二分类 GPU 验证及 4/8/38 类迁移 smoke test。
- 完成 `34/34` 项自动化测试和编译检查。

### 下一步

1. 固定二分类 checkpoint 和阈值，建立真正独立的合成测试集。
2. 对 Swin3D、PointMamba、mHC 和 Muon/AdamW 做统一协议下的消融。
3. 逐阶段完成 `four`、`eight`、`full38` 训练，并分别报告多分类结果。
4. 获取带可靠标签、坐标单位和 JTG 映射的真实道路点云。
5. 在真实域校准点密度和强度后，再验证域随机化、GAN 或无监督域适配。

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

[English](README.en.md)

</div>
