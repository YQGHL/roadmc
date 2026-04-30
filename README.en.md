<div align="center">

# RoadMC

**Physics-Simulation-Driven Pavement Point Cloud Defect Detection**

*JTG 5210—2018 | 38-class semantic segmentation*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![JTG 5210](https://img.shields.io/badge/Standard-JTG%205210--2018-orange?style=flat-square)]()

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

| Phase | Key Component | Output |
|:-----:|---------------|--------|
| 1 — Data Generation | ISO 8608 PSD, fBm, 11 defect primitives | Synthetic `.npz` point clouds (38 classes) |
| 2 — Network Model | Swin3D Transformer + mHC (Sinkhorn-Knopp) | Per-point segmentation logits |
| 3 — GAN Adaptation | DGCNN generator + PointNet WGAN-GP discriminator | Style-transferred point clouds |
| 4 — Data Loading | PyTorch Lightning DataModule + padding collate | Batched (B, N, 3+C) tensors |
| 5 — Training & Evaluation | FocalLoss + DiceLoss + EdgeLoss, AdamW + CosineLR | Per-class IoU / Recall / Precision |

---

## Quick Start

### Environment

```bash
uv sync                  # Python >= 3.11, installs all dependencies
```

### Self-Verification

Each module contains an `if __name__ == "__main__"` block with assert-based tests.

```bash
python roadmc/models/mhc/mhc.py                  # mHC hyperconnection
python roadmc/data/synthetic/generator.py         # Scene generation pipeline
python roadmc/models/model_pl.py                  # LightningModule wrapper

python roadmc/data/synthetic/config.py            # GeneratorConfig validation
python roadmc/data/synthetic/primitives.py        # 11 physical primitives
python roadmc/models/attention/window_attention.py # 3D window attention
python roadmc/models/backbone/swin3d.py           # Swin3D backbone
python roadmc/models/gan/generator.py             # StyleTransferGen
python roadmc/models/gan/discriminator.py         # WGANDiscriminator
python roadmc/data/dataloader.py                  # DataLoader + DataModule
python roadmc/models/mhc/spectral_analysis.py     # Sinkhorn-Knopp spectral analysis
python roadmc/test/test_visualize.py              # 13 diagnostic PNG images
```

### Generate Synthetic Data

```bash
python -m roadmc.scripts.generate_synthetic \
    --train-count 2000 --val-count 500 \
    --grid-res 0.01 --roughness B
```

---

## Data Format

### Synthetic Point Cloud (`.npz`)

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `points` | (N, 3) | `float32` | 3D coordinates (x, y, z) |
| `labels` | (N,) | `int32` | JTG 5210-2018 class [0, 37] |
| `feats` | (N, 3) | `float32` | Intensity, curvature, crack distance |
| `normals` | (N, 3) | `float32` | Unit surface normals |
| `pavement_type` | — | `str` | `'asphalt'` or `'concrete'` |

```python
import numpy as np
data = np.load("scene_0000.npz", allow_pickle=True)
points = data["points"]   # (N, 3) float32
labels = data["labels"]   # (N,) int32
```

---

## Training

| Mode | Command | Description |
|:----:|---------|-------------|
| `baseline` | `python roadmc/train.py baseline --data_dir ./data/synthetic_output --max_epochs 50` | Synthetic data segmentation only |
| `gan_enhanced` | `python roadmc/train.py gan_enhanced --max_epochs 50` | GAN pre-training + style-transfer mixing |
| `end2end` | `python roadmc/train.py end2end --max_epochs 50` | Alternating GAN + segmentation joint optimization |

Optimizer: AdamW (lr=1e-4, weight_decay=0.05). Scheduler: CosineAnnealingLR.  
Loss: λ₁·FocalLoss(γ=2) + λ₂·DiceLoss + λ₃·EdgeLoss(Sobel).

---

## Evaluation

Per-class IoU, Recall, and Precision across 38 JTG 5210-2018 labels, grouped by asphalt [1–20] and concrete [21–37]. Outputs JSON report and terminal-formatted tables.

```bash
python roadmc/evaluate.py --checkpoint ./lightning_logs/version_X/checkpoints/best.ckpt
```

---

## Project Structure

```
roadmc/
├── data/
│   ├── synthetic/
│   │   ├── config.py              # GeneratorConfig dataclasses
│   │   ├── primitives.py          # 11 physical defect primitives
│   │   └── generator.py           # SyntheticRoadDataset (10-step pipeline)
│   ├── real/                      # Real .ply/.npy loader (stub)
│   └── dataloader.py              # PyTorch Lightning DataModule
├── models/
│   ├── backbone/swin3d.py         # 4-stage Swin3D Transformer (31.2M)
│   ├── attention/                 # WindowAttention3D + DeformableAttention3D
│   ├── mhc/mhc.py                 # Doubly-stochastic channel mixing (arXiv:2512.24880)
│   ├── gan/
│   │   ├── generator.py           # DGCNN StyleTransferGen (125K params)
│   │   └── discriminator.py       # PointNet WGAN-GP critic (83K params)
│   └── model_pl.py                # LightningModule + FocalLoss + DiceLoss + EdgeLoss
├── scripts/
│   └── generate_synthetic.py      # CLI batch .npz generation
├── test/
│   └── test_visualize.py          # 13 diagnostic PNG outputs
├── docs/
│   ├── data_pipeline.png          # Data pipeline diagram
│   ├── model_pipeline.png         # Model pipeline diagram
│   └── architecture.png           # Full system architecture
├── train.py                       # Training entry (baseline / gan_enhanced / end2end)
├── evaluate.py                    # Per-class IoU / Recall / Precision
├── configs/                       # Placeholder (unused, dataclass config instead)
├── LICENSE                        # MIT
└── pyproject.toml
```

---

## Citation

```bibtex
@misc{roadmc2026,
  author       = {YQGHL},
  title        = {RoadMC: Physics-Simulation-Driven Road Surface Point Cloud Defect Detection},
  year         = {2026},
  howpublished = {\url{https://github.com/YQGHL/roadmc}},
}
```

---

## License

MIT © 2026 YQGHL. See [LICENSE](LICENSE).

---

<div align="center">

> [**中文**](README.md)

</div>
