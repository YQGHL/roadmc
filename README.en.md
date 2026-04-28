<div align="center">

# 🛣️ RoadMC

**Physics-Simulation-Driven Road Surface Point Cloud Disease Detection**

*Strictly Conforming to JTG 5210—2018 Highway Performance Assessment Standard*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![JTG 5210](https://img.shields.io/badge/Standard-JTG%205210--2018-orange?style=flat-square)]()

<br>

> [**中文版 (Chinese)**](README.md)

_Physical Fidelity · Mathematical Rigor · Engineering Precision · Full Pipeline_

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Phase 1: Synthetic Data Generation](#-phase-1-synthetic-data-generation)
- [Phase 2: Network Model](#-phase-2-network-model)
- [Phase 3: GAN Domain Adaptation](#-phase-3-gan-domain-adaptation)
- [Phase 4: Data Loading](#-phase-4-data-loading)
- [Phase 5: Training & Evaluation](#-phase-5-training--evaluation)
- [Getting Started](#-getting-started)
- [Data Format](#-data-format)
- [Visualization](#-visualization)
- [Project Structure](#-project-structure)
- [License & Citation](#-license--citation)

---

## 🌄 Overview

RoadMC is a **physics-simulation-driven**, **mathematically constrained** road surface point cloud disease detection system. It generates high-fidelity synthetic training data through physically accurate road surface modeling and employs a **Swin3D Transformer** architecture enhanced with **Manifold Hyper-Connection (mHC)** for semantic segmentation.

**Key Features:**

- **Physical Fidelity:** ISO 8608 PSD road surface synthesis with 11 mechanically accurate disease models
- **Mathematical Rigor:** Sinkhorn-Knopp doubly-stochastic mixing, Sobel edge loss, PCA curvature
- **Standard Compliance:** Full **JTG 5210—2018** compliance, 38 disease labels across asphalt and concrete
- **Complete Pipeline:** From data generation through model training to evaluation

---

## 🏗 Architecture

<p align="center">
  <img src="docs/architecture.png" alt="RoadMC System Architecture" width="95%"/>
  <br>
  <em>Fig: RoadMC system architecture (Phase 1-5) — Data Generation → Data Loading → Core Network (Swin3D+mHC) → GAN Adaptation → Training/Evaluation</em>
</p>

---

## 🔬 Phase 1: Synthetic Data Generation

### Pipeline

| Step | Component | Model |
|:----:|-----------|-------|
| ① | Pavement selection | Probability-driven (asphalt/concrete) |
| ② | Surface generation | **ISO 8608 PSD** + inverse FFT |
| ③ | LiDAR scan resampling | Gaussian scan line + range decay |
| ④ | Micro-texture | **fBm** fractional Brownian motion |
| ⑤ | Disease application | 11 models with label priority |
| ⑥ | Curvature | PCA eigenvalue ratio |
| ⑦ | LiDAR noise | Spherical noise + Bernoulli dropout |
| ⑧ | Label transfer | KDTree + 3σ threshold |
| ⑨ | Downsampling | 2D voxel averaging |
| ⑩ | Normalization | Unit sphere scaling |

### Disease Models

| Disease | Model | Labels |
|---------|-------|:------:|
| Crack (4 types) | Bézier + Perlin + Voronoi | 1–8 |
| Pothole | Superellipsoid | 9–10 |
| Raveling | Stochastic removal | 11–12 |
| Depression | Gaussian subsidence | 13–14 |
| Rutting | Dual Gaussian tracks | 15–16 |
| Corrugation | Sinusoidal | 17–18 |
| Bleeding | Label only | 19 |
| Concrete (10 types) | Voronoi shattering, faulting, etc. | 21–37 |

### Code Review

| Priority | Items | Status |
|:--------:|:-----:|:------:|
| 🔴 P0 | 4 critical fixes | ✅ |
| 🟠 P1 | 6 important fixes | ✅ |
| 🟡 P2 | 4 suggested fixes | ✅ |

---

## 🧠 Phase 2: Network Model

### Manifold Hyper-Connection

$$M = \text{softplus}(W_1)\,\text{softplus}(W_2)^\top,\quad H = \text{SinkhornKnopp}(M/\tau)$$

$$y = x + H \cdot r \in \mathbb{R}^{B \times C}$$

`deploy()` freezes $H$ for inference.

> 📄 **mHC paper**: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) — Manifold Hyper-Connection

### Transformer Block

```python
x_norm = LN(x)
x_attn = WindowAttention(x_norm) + x          # pre-LN
x_ffn = LN(x_attn)
x_ffn = FFN(x_ffn) + x_attn                  # pre-LN
x_out = MHC(x_ffn, x_attn)                   # mHC mixing
```

### Swin3D Backbone

| Stage | C | Depth | Heads |
|:-----:|:-:|:-----:|:-----:|
| 0 | 96 | 2 | 3 |
| 1 | 192 | 2 | 6 |
| 2 | 384 | 6 | 12 |
| 3 | 768 | 2 | 24 |

**Params**: 3.5M (tiny) / 31.2M (full) | **Output**: $(B,N,38)$

### Loss

$$\mathcal{L} = \lambda_1 \text{FocalLoss} + \lambda_2 \text{DiceLoss} + \lambda_3 \mathcal{L}_{\text{edge}}$$

---

## 🎨 Phase 3: GAN Domain Adaptation

**Generator** (`models/gan/generator.py`): DGCNN EdgeConv encoder-decoder (125K params)  
**Discriminator** (`models/gan/discriminator.py`): PointNet WGAN-GP critic (83K params)  
**Loss**: WGAN-GP + Chamfer distance + normal consistency

---

## 📦 Phase 4: Data Loading

**Synthetic**: `SyntheticPointCloudDataset` + `RoadMCDataModule` (Lightning) with augmentation  
**Real**: `RealRoadDataset` (.ply/.npy, JTG label mapping)  
**Collation**: Auto-padding with `valid_mask` (-1 labels ignored in loss)

---

## 🚀 Phase 5: Training & Evaluation

| Mode | Command | Description |
|:----:|---------|-------------|
| `baseline` | `python roadmc/train.py baseline` | Synthetic only |
| `gan_enhanced` | `python roadmc/train.py gan_enhanced` | GAN → mixed |
| `end2end` | `python roadmc/train.py end2end` | Alternating |

**Evaluation**: `python roadmc/evaluate.py --checkpoint <path>` — per-class IoU/recall/precision, grouped report, JSON output.

---

## 🚀 Getting Started

```bash
# Install
pip install numpy scipy torch matplotlib pytorch-lightning torchmetrics

# Phase 1 tests
python roadmc/data/synthetic/config.py
python roadmc/data/synthetic/primitives.py
python roadmc/data/synthetic/generator.py

# Phase 2 tests
python roadmc/models/mhc/mhc.py
python roadmc/models/attention/window_attention.py
python -m roadmc.models.backbone.swin3d
python roadmc/models/model_pl.py

# Phase 3 tests
python roadmc/models/gan/generator.py
python roadmc/models/gan/discriminator.py

# Phase 4 tests
python roadmc/data/dataloader.py
python roadmc/data/real/dataset.py

# Phase 5 / utilities
python roadmc/models/mhc/spectral_analysis.py

# Visualization
python roadmc/test/test_visualize.py

# Batch generation
python -m roadmc.scripts.generate_synthetic
```

---

## 📦 Data Format

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `points` | $(N,3)$ | `float32` | $(x,y,z)$ coordinates |
| `labels` | $(N,)$ | `int32` | JTG labels $[0,37]$ |
| `feats` | $(N,3)$ | `float32` | intensity, curvature, crack distance |
| `normals` | $(N,3)$ | `float32` | Unit normals |

```python
data = np.load("scene_0000.npz", allow_pickle=True)
```

---

## 🎨 Visualization

13 figures: 2D overlay, 3D mesh, height maps, feature channels, label statistics.

<p align="center">
  <img src="docs/asphalt_multi_disease_2d_overlay.png" alt="2D Disease Overlay" width="45%"/>
  <img src="docs/label_statistics.png" alt="Label Statistics" width="45%"/>
  <br>
  <em>Left: label-colored top-down view  Right: per-scene label distribution</em>
</p>

---

## 📁 Project Structure

```
roadmc/
├── data/
│   ├── synthetic/     # Config + 13 primitives + Generator
│   └── real/          # Real point cloud dataset
├── models/
│   ├── mhc/           # MHC Connection
│   ├── attention/     # WindowAttention3D
│   ├── backbone/      # Swin3D backbone
│   ├── gan/           # Generator + Discriminator
│   └── model_pl.py    # LightningModule
├── scripts/           # CLI batch generation
├── test/              # Visualization suite
├── train.py           # Training (3 modes)
├── evaluate.py        # JTG evaluation report
├── README.md
├── LICENSE
├── .gitignore
└── pyproject.toml
```

---

## 📄 License & Citation

**MIT** © 2026 YQGHL

```bibtex
@misc{roadmc2026,
  author = {YQGHL},
  title = {RoadMC: Physics-Simulation-Driven Road Surface Point Cloud Disease Detection},
  year = {2026}
}
```

---

<div align="center">

**English** · [**中文版 (Chinese)**](README.md)

</div>
