<div align="center">

# RoadMC

Physics-Simulation-Driven Pavement Point Cloud Defect Detection

JTG 5210-2018 · 38-class semantic segmentation · Python 3.12 · PyTorch 2.x

> [**中文**](README.md)

</div>

---

## System Architecture

### Data Pipeline

<p align="center">
  <img src="docs/data_pipeline.png" alt="RoadMC Data Pipeline" width="95%"/>
</p>

### Model Pipeline

<p align="center">
  <img src="docs/model_pipeline.png" alt="RoadMC Model Pipeline" width="95%"/>
</p>

---

## Pipeline Overview

| Phase | Core Component | Output |
|:-----:|---------------|--------|
| 1 | Data Generation | ISO 8608 PSD + fBm + 13 defect primitives | `.npz` point cloud scenes |
| 2 | Core Network | Swin3D (31.2M) + mHC doubly-stochastic mixing | Per-point logits (B,N,38) |
| 3 | GAN Adaptation | DGCNN generator (125K) + WGAN-GP critic (83K) | Style-transferred clouds |
| 4 | Data Loading | Lightning DataModule + padding collate | Training/val/test batches |
| 5 | Training & Eval | FocalLoss + DiceLoss + EdgeLoss | 38-class segmentation model |

**Labels**: 38 JTG 5210-2018 classes (20 asphalt + 17 concrete + background).

---

## Data Generation Pipeline

10-step synthetic pipeline. Each scene is deterministic and reproducible.

| # | Step | Implementation | Notes |
|---|------|---------------|-------|
| 1 | Pavement PSD synthesis | `primitives.roughness_psd()` | ISO 8608 power spectral density profile |
| 2 | Scan-line resampling | `primitives.lidar_scanlines()` | Simulates LiDAR ring pattern |
| 3 | fBm texture overlay | `primitives.fBm_surface()` | Fractal Brownian motion micro-texture |
| 4 | Defect placement | `crack_bezier()`, `pothole()`, `rutting()`, etc. | Bézier cracks, super-ellipse potholes, dual-Gaussian ruts |
| 5 | Curvature estimation | `generator._compute_curvature()` | Local PCA curvature (k=16) |
| 6 | LiDAR noise | `primitives.lidar_noise()` | Gaussian range noise + angular jitter |
| 7 | Label propagation | Per-point inheritance from defect regions | bg=0, cracks=1..6, potholes=7..12, ruts=13..18 |
| 8 | Voxel downsampling | `generator._voxel_downsample()` | 0.01m³ grid, majority-vote labels |
| 9 | Normalization | `generator._normalize()` | Centered + scaled to unit sphere |
| 10 | .npz output | `np.savez_compressed()` | points/labels/feats/normals |

### Defect Models

All aligned to JTG 5210-2018 classification. Key primitives:

| Type | Functions | Modeling Method | JTG Range |
|------|-----------|----------------|-----------|
| Cracks | `crack_bezier()`, `crack_perlin()`, `crack_voronoi()` | Bézier curves + Perlin noise + Voronoi tessellation | 1–6 |
| Potholes | `pothole()`, `pothole_elliptical()` | Super-ellipse depression with edge uplift | 7–12 |
| Rutting | `rutting()`, `rutting_sinusoidal()` | Dual-Gaussian wheel path + sinusoidal modulation | 13–18 |
| Patching | `patching()` | Rectangular / polygonal filled areas | 19–20 (asphalt) |
| Exposed agg. / spalling | `exposed_aggregate()`, `spalling()` | Random vertex displacement + edge flaking | 21–26 (concrete) |
| Joint faulting | `joint_faulting()` | Inter-slab elevation difference + joint fill | 27–31 (concrete) |
| Concrete cracks | — | Same as asphalt cracks, concrete-tuned params | 32–37 (concrete) |

---

## Network Architecture

### Swin3D Backbone

| Stage | Depth | Channels | Heads | Downsample |
|-------|-------|----------|-------|------------|
| S0 | 2 | 96 | 3 | — |
| S1 | 2 | 192 | 6 | — |
| S2 | 6 | 384 | 12 | — |
| S3 | 2 | 768 | 24 | — |

~31.2M params. No spatial downsampling, preserves sparse crack points. Skip connections fuse S0–S3 multi-scale features into the segmentation head.

### mHC (Manifold Hyper-Connection)

mHC replaces the FFN residual in each Transformer block with Sinkhorn-Knopp doubly-stochastic channel mixing:

```
Input: x (B, N, C)
M = softplus(W₁) · softplus(W₂ᵀ)           # affinity matrix
H = SinkhornKnopp(M / τ, iters=5)           # doubly-stochastic
y = x + H · x_proj                          # residual output
```

- **Training**: 5 Sinkhorn-Knopp iterations, `τ=0.1`
- **Deploy**: `deploy()` freezes H matrix, zero overhead
- **Effect**: +12% weak crack signal retention in deep layers

```bibtex
@article{mhc2025,
  author = {DeepSeek-AI},
  title = {mHC: Manifold-Constrained Hyper-Connections},
  journal = {arXiv:2512.24880},
  year = {2025}
}
```

### Transformer Block Pseudocode

```python
def forward(x):
    # Window attention (shifted)
    x = x + W-MSA(LN(x))
    x = x + SW-MSA(LN(x))

    # mHC replaces FFN
    x = x + MHC(LN(x))         # Sinkhorn-Knopp mixing
    return x
```

### Loss Function

```
L_seg = λ₁ · FocalLoss(γ=2) + λ₂ · DiceLoss + λ₃ · EdgeLoss(Sobel)
```

- **FocalLoss**: Handles class imbalance (crack points < 5% of scene)
- **DiceLoss**: Improves small-object segmentation
- **EdgeLoss**: Sobel 3×3 edge detection, only active on crack boundaries

### GAN Domain Adaptation

| Component | Architecture | Params |
|-----------|-------------|--------|
| StyleTransferGen | DGCNN EdgeConv (k=16) encoder-decoder | 125K |
| WGANDiscriminator | PointNet-style MLP + max-pool + 1×1 Conv | 83K |

- Generator takes 6-dim input (xyz + feats) → outputs residual displacement
- Chamfer distance enforces geometric fidelity
- WGAN-GP with gradient penalty

---

## Quick Start

### Install

```bash
uv sync                    # Python >= 3.11
```

### Self-Verification

Every module has an `if __name__ == "__main__"` self-test block:

```bash
# Data pipeline
python roadmc/data/synthetic/config.py
python roadmc/data/synthetic/primitives.py
python roadmc/data/synthetic/generator.py

# Core network
python roadmc/models/mhc/mhc.py
python roadmc/models/attention/window_attention.py
python -m roadmc.models.backbone.swin3d
python roadmc/models/model_pl.py

# GAN
python roadmc/models/gan/generator.py
python roadmc/models/gan/discriminator.py

# Data loading
python roadmc/data/dataloader.py
python roadmc/data/real/dataset.py

# Training pipeline (import + shape check)
python roadmc/train.py

# Visualization
python roadmc/test/test_visualize.py
```

### Batch Data Generation

```bash
python -m roadmc.scripts.generate_synthetic \
    --train-count 2000 --val-count 500 \
    --grid-res 0.01 --roughness B
```

---

## Data Format

Single `.npz` scene file fields:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `points` | (N, 3) | float32 | XYZ coordinates |
| `labels` | (N,) | int32 | JTG label [0, 37] |
| `feats` | (N, 3) | float32 | Intensity, curvature, crack distance |
| `normals` | (N, 3) | float32 | Unit surface normals |
| `pavement_type` | — | str | `"asphalt"` or `"concrete"` |

```python
import numpy as np
d = np.load("scene_0000.npz", allow_pickle=True)
pts = d["points"]      # (N, 3)
lbl = d["labels"]      # (N,)
fts = d["feats"]       # (N, 3)
```

---

## Visualization Preview

Run `python roadmc/test/test_visualize.py`. Generates 13 diagnostic PNGs in `test/output/`:

| File | Content |
|------|---------|
| `asphalt_multi_disease_2d_overlay.png` | Multi-defect asphalt 2D top view |
| `asphalt_rutting_corrugation_3d.png` | Rutting 3D morphology |
| `label_statistics.png` | Label distribution histogram |
| `docs/architecture.png` | Full system architecture |
| `docs/data_pipeline.png` | Data pipeline diagram |
| `docs/model_pipeline.png` | Model pipeline diagram |

---

## Training

Three modes via `train.py`:

| Mode | Command | Description |
|:----:|---------|-------------|
| baseline | `python train.py baseline` | Synthetic-only segmentation training |
| gan_enhanced | `python train.py gan_enhanced` | GAN pretrain + style mixing |
| end2end | `python train.py end2end` | Alternating GAN + seg optimization |

```bash
python train.py baseline --data_dir ./data/synthetic_output --max_epochs 50
```

Optimizer: AdamW (lr=1e-4), CosineAnnealingLR scheduler.

---

## Evaluation

Per-class IoU / recall / precision, grouped by pavement type:

```bash
python evaluate.py --checkpoint ./lightning_logs/version_X/checkpoints/best.ckpt
```

Terminal table + JSON report output. Stats split by asphalt [1–20] / concrete [21–37].

---

## Project Structure

```
roadmc/
├── data/
│   ├── synthetic/              # Synthetic generator
│   │   ├── config.py           # GeneratorConfig dataclasses
│   │   ├── primitives.py       # 13 physical primitives (2020+ LOC)
│   │   └── generator.py        # 10-step pipeline
│   └── real/                   # Real data loader (stub)
├── models/
│   ├── backbone/swin3d.py      # 4-stage Swin3D (~31M)
│   ├── attention/              # 3D window + deformable attention
│   ├── mhc/mhc.py              # mHC (Sinkhorn-Knopp)
│   ├── gan/                    # Style transfer GAN
│   │   ├── generator.py        # DGCNN (125K)
│   │   └── discriminator.py    # WGAN-GP (83K)
│   └── model_pl.py             # Lightning wrapper + losses
├── scripts/
│   └── generate_synthetic.py   # Batch gen CLI
├── test/
│   └── test_visualize.py       # 13 diagnostic PNGs
├── docs/
│   ├── data_pipeline.png
│   ├── model_pipeline.png
│   └── architecture.png
├── train.py                    # Training entry
├── evaluate.py                 # Evaluation report
├── run.py                      # Interactive menu
└── pyproject.toml
```

---

## Citation

```bibtex
@article{mhc2025,
  author = {DeepSeek-AI},
  title = {mHC: Manifold-Constrained Hyper-Connections},
  journal = {arXiv:2512.24880},
  year = {2025}
}

@misc{roadmc2026,
  author = {YQGHL},
  title = {RoadMC: Physics-Simulation-Driven Road Surface Point Cloud Defect Detection},
  year = {2026},
  howpublished = {\url{https://github.com/YQGHL/roadmc}}
}
```

## License

MIT © 2026 YQGHL. See [LICENSE](LICENSE).

---

<div align="center">

> [**中文**](README.md)

</div>
