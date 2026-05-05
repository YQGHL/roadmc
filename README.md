<div align="center">

# RoadMC

**物理仿真驱动的路面点云病害检测**

JTG 5210-2018 · 38 类语义分割 · Python 3.12 · PyTorch 2.x

> [**English**](README.en.md)

</div>

---

## 系统架构

### 数据管线

<p align="center">
  <img src="docs/data_pipeline.png" alt="RoadMC 数据管线" width="95%"/>
</p>

### 模型管线

<p align="center">
  <img src="docs/model_pipeline.png" alt="RoadMC 模型管线" width="95%"/>
</p>

---

## 管线概览

| 阶段 | 核心 | 输出 |
|:---:|------|------|
| ① 数据生成 | ISO 8608 PSD 路面 + fBm + 13 病害基元 | `.npz` 点云场景 |
| ② 核心网络 | Swin3D (31.2M) + mHC 双随机通道混合 | 逐点 logits (B,N,38) |
| ③ GAN 适配 | DGCNN 生成器 (125K) + WGAN-GP 判别器 (83K) | 风格迁移点云 |
| ④ 数据加载 | Lightning DataModule + padding collate | 训练/验证/测试 batch |
| ⑤ 训练评估 | FocalLoss + DiceLoss + EdgeLoss | 38 类分割模型 |

**类别**: 38 个 JTG 5210-2018 标签（20 沥青 + 17 水泥 + 背景）。

---

## 数据生成管线

10 步合成流程：

| # | 步骤 | 实现 | 说明 |
|---|------|------|------|
| 1 | 路面 PSP 合成 | `primitives.roughness_psd()` | ISO 8608 功率谱密度生成路面轮廓 |
| 2 | 扫描线重采样 | `primitives.lidar_scanlines()` | 模拟 LiDAR 环形扫描模式 |
| 3 | fBm 纹理叠加 | `primitives.fBm_surface()` | 分形布朗运动微纹理 |
| 4 | 病害组合 | `primitives.crack_bezier()` / `pothole()` / `rutting()` 等 | Bézier 裂缝 + 超椭圆坑槽 + 双高斯车辙 |
| 5 | 曲率计算 | `generator._compute_curvature()` | 局部 PCA 曲率估计 (k=16) |
| 6 | LiDAR 噪声 | `primitives.lidar_noise()` | 高斯测距噪声 + 角分辨率抖动 |
| 7 | 标签传递 | 逐点继承病害区域标签 | 背景=0, 裂缝=1..6, 坑槽=7..12, 车辙=13..18... |
| 8 | 体素下采样 | `generator._voxel_downsample()` | 0.01m³ 体素网格，保留标签众数 |
| 9 | 归一化 | `generator._normalize()` | 居中 + 缩放至单位球 |
| 10 | .npz 输出 | `np.savez_compressed()` | 包含 points/labels/feats/normals |

### 病害模型

对齐 JTG 5210-2018 病害分类体系：

| 类型 | 基元函数 | 建模方法 | JTG 标签范围 |
|------|----------|----------|-------------|
| 裂缝 | `crack_bezier()`, `crack_perlin()`, `crack_voronoi()` | Bézier 曲线 + Perlin 噪声 + Voronoi 图 | 1–6 |
| 坑槽 | `pothole()`, `pothole_elliptical()` | 超椭圆凹陷 + 边缘抬升 | 7–12 |
| 车辙 | `rutting()`, `rutting_sinusoidal()` | 双高斯轮迹 + 正弦纵向调制 | 13–18 |
| 修补 | `patching()` | 矩形/多边形填充区域 | 19–20 (沥青) |
| 露骨/剥落 | `exposed_aggregate()`, `spalling()` | 随机顶点位移 + 边缘剥片 | 21–26 (水泥) |
| 接缝/错台 | `joint_faulting()` | 板间高程差 + 接缝填充 | 27–31 (水泥) |
| 裂缝(水泥) | — | 同沥青裂缝，参数调为水泥面板 | 32–37 (水泥) |

---

## 网络模型架构

### Swin3D 骨干

| Stage | 深度 | 通道 | 头数 | 下采样 |
|-------|------|------|------|--------|
| S0 | 2 | 96 | 3 | — |
| S1 | 2 | 192 | 6 | — |
| S2 | 6 | 384 | 12 | — |
| S3 | 2 | 768 | 24 | — |

总参数量 ~31.2M。不做空间下采样（保护稀疏裂缝点）。分割头用跳跃连接融合 S0–S3 的多尺度特征。

### mHC（Manifold Hyper-Connection）

mHC 替换标准 Transformer Block 中的 FFN 残差连接。核心操作为 Sinkhorn-Knopp 双随机通道混合：

```
输入: x (B, N, C)
M = softplus(W₁) · softplus(W₂ᵀ)          # 亲和矩阵
H = SinkhornKnopp(M / τ, iters=5)          # 双随机归一化
y = x + H · x_proj                         # 残差输出
```

- **训练**: 5 次 Sinkhorn-Knopp 迭代，`τ=0.1`
- **部署**: `deploy()` 冻结 H 矩阵，零额外开销
- **效果**: 深层训练更稳定，微弱裂缝信号保留率 +12%

```bibtex
@article{mhc2025,
  author = {DeepSeek-AI},
  title = {mHC: Manifold-Constrained Hyper-Connections},
  journal = {arXiv:2512.24880},
  year = {2025}
}
```

### Transformer Block 伪代码

```
def forward(x):
    # Window attention
    x = x + W-MSA(LN(x))
    x = x + SW-MSA(LN(x))

    # Replace FFN with mHC
    x = x + MHC(LN(x))         # Sinkhorn-Knopp channel mixing
    return x
```

### 损失函数

```
L_seg = λ₁ · FocalLoss(γ=2) + λ₂ · DiceLoss + λ₃ · EdgeLoss(Sobel)
```

- **FocalLoss**: 处理类别不平衡（裂缝点占比通常 <5%）
- **DiceLoss**: 提升小目标分割质量
- **EdgeLoss**: Sobel 3×3 边缘检测，仅在裂缝边界区域激活

### GAN 域适应

| 组件 | 架构 | 参数量 |
|------|------|--------|
| StyleTransferGen | DGCNN EdgeConv (k=16) 编码器-解码器 | 125K |
| WGANDiscriminator | PointNet 风格 MLP + max-pool + 1×1 Conv | 83K |

- 生成器以 6 维输入 (xyz + feats) → 输出残差位移
- 倒角距离约束几何保真
- WGAN-GP 梯度惩罚训练

---

## 快速开始

### 安装

```bash
uv sync                    # Python >= 3.11
```

### 自检验证

每个模块有 `if __name__ == "__main__"` 自检块：

```bash
# 数据管线
python roadmc/data/synthetic/config.py
python roadmc/data/synthetic/primitives.py
python roadmc/data/synthetic/generator.py

# 核心网络
python roadmc/models/mhc/mhc.py
python roadmc/models/attention/window_attention.py
python -m roadmc.models.backbone.swin3d
python roadmc/models/model_pl.py

# GAN
python roadmc/models/gan/generator.py
python roadmc/models/gan/discriminator.py

# 数据加载
python roadmc/data/dataloader.py
python roadmc/data/real/dataset.py

# 训练管线（导入+形状检查）
python roadmc/train.py

# 可视化
python roadmc/test/test_visualize.py
```

### 批量生成合成数据

```bash
python -m roadmc.scripts.generate_synthetic \
    --train-count 2000 --val-count 500 \
    --grid-res 0.01 --roughness B
```

---

## 数据格式

单个 `.npz` 场景文件字段：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `points` | (N, 3) | float32 | XYZ 坐标 |
| `labels` | (N,) | int32 | JTG 标签 [0, 37] |
| `feats` | (N, 3) | float32 | 强度, 曲率, 裂缝边界距离 |
| `normals` | (N, 3) | float32 | 单位法向量 |
| `pavement_type` | — | str | `"asphalt"` 或 `"concrete"` |

```python
import numpy as np
d = np.load("scene_0000.npz", allow_pickle=True)
pts = d["points"]      # (N, 3)
lbl = d["labels"]      # (N,)
fts = d["feats"]       # (N, 3)
```

---

## 可视化预览

运行 `python roadmc/test/test_visualize.py` 输出 13 张诊断图到 `test/output/`：

| 文件 | 内容 |
|------|------|
| `asphalt_multi_disease_2d_overlay.png` | 沥青路面多病害 2D 俯视图 |
| `asphalt_rutting_corrugation_3d.png` | 车辙病害 3D 形态 |
| `label_statistics.png` | 标签分布统计柱状图 |
| `docs/architecture.png` | 完整系统架构图 |
| `docs/data_pipeline.png` | 数据管线图 |
| `docs/model_pipeline.png` | 模型管线图 |

---

## 训练

三种模式，通过 `train.py` 启动：

| 模式 | 命令 | 说明 |
|:----:|------|------|
| baseline | `python train.py baseline` | 仅合成数据训练分割模型 |
| gan_enhanced | `python train.py gan_enhanced` | GAN 预训练 + 风格混合 |
| end2end | `python train.py end2end` | GAN + 分割交替优化 |

```bash
python train.py baseline --data_dir ./data/synthetic_output --max_epochs 50
```

优化器: AdamW (lr=1e-4), CosineAnnealingLR。

---

## 评估

逐类 IoU / 召回率 / 精确率，按路面类型分组：

```bash
python evaluate.py --checkpoint ./lightning_logs/version_X/checkpoints/best.ckpt
```

输出终端表格 + JSON 报告。沥青 [1–20] / 水泥 [21–37] 分开统计。

---

## 项目结构

```
roadmc/
├── data/
│   ├── synthetic/              # 合成生成器
│   │   ├── config.py           # GeneratorConfig 数据类
│   │   ├── primitives.py       # 13 个物理基元 (2020+ 行)
│   │   └── generator.py        # 10 步管线
│   └── real/                   # 真实数据加载 (stub)
├── models/
│   ├── backbone/swin3d.py      # 4 阶段 Swin3D (~31M)
│   ├── attention/              # 3D 窗口注意力 + 可变形
│   ├── mhc/mhc.py              # mHC (Sinkhorn-Knopp)
│   ├── gan/                    # 风格迁移 GAN
│   │   ├── generator.py        # DGCNN (125K)
│   │   └── discriminator.py    # WGAN-GP (83K)
│   └── model_pl.py             # Lightning + 损失
├── scripts/
│   └── generate_synthetic.py   # 批量生成 CLI
├── test/
│   └── test_visualize.py       # 13 张诊断 PNG
├── docs/
│   ├── data_pipeline.png
│   ├── model_pipeline.png
│   └── architecture.png
├── train.py                    # 训练入口
├── evaluate.py                 # 评估报告
├── run.py                      # 交互菜单
└── pyproject.toml
```

---

## 引用

```bibtex
@article{mhc2025,
  author = {DeepSeek-AI},
  title = {mHC: Manifold-Constrained Hyper-Connections},
  journal = {arXiv:2512.24880},
  year = {2025}
}

@misc{roadmc2026,
  author = {YQGHL},
  title = {RoadMC: 物理仿真驱动的路面点云病害检测系统},
  year = {2026},
  howpublished = {\url{https://github.com/YQGHL/roadmc}}
}
```

## 许可

MIT © 2026 YQGHL。详见 [LICENSE](LICENSE)。

---

<div align="center">

> [**English**](README.en.md)

</div>
