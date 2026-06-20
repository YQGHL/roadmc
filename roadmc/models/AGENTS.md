# models/ — Deep Learning Architecture

## OVERVIEW

Swin3D backbone + mHC channel mixing + GAN domain adaptation for 38-class road disease segmentation.

## STRUCTURE

```
models/
├── backbone/swin3d.py       # Swin3D: 4-stage Transformer + FCN decoder
├── attention/
│   └── window_attention.py  # WindowAttention3D, DeformableWindowAttention3D, ShiftedWindowTransformerBlock
├── mhc/
│   ├── mhc.py               # MHCConnection (Sinkhorn-Knopp doubly-stochastic)
│   └── spectral_analysis.py # SpectralAnalyzer: verify spectral norm ≤ 1
├── gan/
│   ├── generator.py         # StyleTransferGen (DGCNN EdgeConv encoder-decoder)
│   └── discriminator.py     # WGANDiscriminator (PointNet-style)
└── model_pl.py              # LightningModule + FocalLoss + DiceLoss + EdgeLoss
```

## WHERE TO LOOK

| Task | File | Key class |
|------|------|-----------|
| Modify backbone depth/width | `backbone/swin3d.py` | `Swin3D.__init__` params: embed_dim, depths, num_heads |
| Change attention mechanism | `attention/window_attention.py` | `WindowAttention3D` uses per-window loops |
| Tune mHC mixing | `mhc/mhc.py` | `MHCConnection` — sinkhorn_iters, temp params |
| Adjust GAN architecture | `gan/generator.py` | `StyleTransferGen` — EdgeConv k=16, 3 encoder stages |
| Change loss weights | `model_pl.py` | `RoadMCSegModel.__init__` — lambda_focal, lambda_dice, lambda_edge |

## CONVENTIONS

- All modules output `(B, N, C)` tensors (batch, points, channels)
- Pre-LN residual pattern: `LN → Attention → +x → LN → FFN → +x → MHC`
- `deploy()` method on mHC freezes Sinkhorn → static matmul for inference
- Self-test in every file: shape + NaN + gradient flow assertions

## ANTI-PATTERNS

- **Full N×N attention is preferred** for N ≤ 4096 — per-window loop creates ~10k CUDA allocations/step, fragmenting the allocator over 2+ epochs → training hang. The full matrix (B, H, N, N) fits in 8 GB VRAM.
- **NEVER use torch.cdist with N > 8192** — EdgeConv memory warning
- MHCConnection W1/W2 initialized to ~-5.0 (softplus ≈ 0.007) for numerical safety
