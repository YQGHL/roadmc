"""RoadMC system architecture diagram generator (grid layout)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

W, H = 20, 16
DPI = 200

C = {
    "data": "#4A90D9", "data_light": "#AED6F1", "data_bg": "#D6EAF8",
    "model": "#E07B39", "model_light": "#F5B041", "model_bg": "#FDEBD0", "model_dark": "#D35400",
    "opt": "#8E44AD", "opt_light": "#BB8FCE", "opt_bg": "#EBDEF0",
    "inf": "#27AE60", "inf_light": "#ABEBC6", "inf_bg": "#D5F5E3",
    "border": "#2C3E50", "text": "#1A1A2E", "text2": "#555555", "text3": "#888888",
    "bg": "#FAFBFC", "divider": "#BDC3C7", "panel": "#FFFFFF", "neutral": "#F0F0F0",
}

fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("white")

# Helpers
def box(x, y, w, h, fc, ec=None, lw=1.0, alpha=1.0, z=3, rnd=0.06):
    ec = ec or C["border"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rnd}",
                                fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))

def txt(x, y, s, fs=9, fw="normal", c=None, ha="center", va="center", z=4):
    c = c or C["text"]
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c, ha=ha, va=va, zorder=z)

def arrow(x1, y1, x2, y2, c=None, lw=1.2, ls="-", z=2):
    c = c or C["data"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw, linestyle=ls,
                                connectionstyle="arc3,rad=0", shrinkA=4, shrinkB=4), zorder=z)

def label(x, y, s, fs=7, c=None, bg=None):
    c = c or C["text2"]
    kw = dict(fontsize=fs, color=c, ha="center", va="center", zorder=5, style="italic")
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.15", fc=bg, ec="none", alpha=0.85)
    ax.text(x, y, s, **kw)

def mod(x, y, w, h, title, sub="", fc=C["data_light"], ec=None, lw=0.8):
    ec = ec or C["border"]
    box(x, y, w, h, fc=fc, ec=ec, lw=lw, z=3)
    if sub:
        txt(x + w/2, y + h*0.62, title, fs=9, fw="bold", z=4)
        txt(x + w/2, y + h*0.3, sub, fs=7, c=C["text2"], z=4)
    else:
        txt(x + w/2, y + h/2, title, fs=9, fw="bold", z=4)

# Background
box(0.1, 0.1, W-0.2, H-0.2, fc=C["bg"], ec="#DEE2E6", lw=1.0, rnd=0.15, z=0)

# Title
txt(W/2, H-0.6, "RoadMC System Architecture", fs=18, fw="bold", z=5)
txt(W/2, H-1.0, "Physics-Simulation-Driven  ·  Math-Constraint-Enhanced  ·  Pavement Point Cloud Defect Detection",
    fs=9, c=C["text3"], z=5)

# Phase Headers (y=12.0 to 12.5, ABOVE all blocks)
Y_PH = 12.2  # phase header center y
PH_H = 0.6   # phase header height

# Draw phase headers with HIGH z-order (z=10) so they're always visible
for i, (x, w, title, sub, clr) in enumerate([
    (0.3, 3.4, "Phase 1: Data Generation", "Physical Simulation", C["data"]),
    (4.0, 4.8, "Phase 2: Core Network", "Swin3D + mHC", C["model"]),
    (9.0, 4.2, "Phase 3: GAN Adaptation", "Style Transfer", C["inf"]),
    (13.5, 4.0, "Phase 4: Data Loading", "Synthetic + Real", C["opt"]),
]):
    box(x, Y_PH - PH_H/2, w, PH_H, fc=clr, ec=clr, lw=0, alpha=0.4, z=10, rnd=0.08)
    txt(x + w/2, Y_PH + 0.1, title, fs=11, fw="bold", c=clr, z=11)
    txt(x + w/2, Y_PH - 0.15, sub, fs=8, c=C["text3"], z=11)

# Phase 1 — Data Generation (x=0.3, w=3.4)
p1x, p1w = 0.3, 3.4
p1_cx = p1x + p1w/2
BH, BG = 0.7, 0.25

y = 11.2  # first block below phase header
mod(p1x, y, p1w, BH, "GeneratorConfig", "dataclass configuration", C["data_light"])
y -= BH + BG
mod(p1x, y, p1w, BH, "primitives.py", "11 physical primitives", C["data_light"])
y -= BH + BG
mod(p1x, y, p1w, BH, "SyntheticRoadDataset", "10-step pipeline", C["data"], lw=1.2)
y -= BH + BG
mod(p1x, y, p1w, BH, "Output: .npz", "points/labels/feats/normals", C["data_bg"])
y -= BH + BG
# Annotation
box(p1x, y-0.5, p1w, 0.5, fc="#F8F9FA", ec=C["divider"], lw=0.5, alpha=0.7, rnd=0.04, z=3)
txt(p1x + p1w/2, y-0.25, "feats: intensity, curvature, crack-dist", fs=6.5, c=C["text3"], z=4)

# Vertical arrows P1
for i in range(3):
    ay = 11.2 - i*(BH+BG) - 0.05
    arrow(p1_cx, ay, p1_cx, ay - BG + 0.1, c=C["data"], lw=0.8)

# Phase 2 — Core Network (x=4.0, w=4.8)
p2x, p2w = 4.0, 4.8
p2_cx = p2x + p2w/2

mod(p2x, 11.2, p2w, 0.65, "Swin3D Backbone", "31.2M | embed=96 | depths=[2,2,6,2]", C["model_bg"], lw=1.2)

# 4 stages
sw, sg = 1.05, 0.15
sx = p2x
for i, (nm, cfg, clr) in enumerate([("S0","C=96",C["model_light"]),("S1","C=192",C["model"]),
                                     ("S2","C=384",C["model_dark"]),("S3","C=768",C["model_dark"])]):
    mod(sx, 10.3, sw, 0.55, nm, cfg, clr, lw=0.8)
    if i < 3:
        arrow(sx+sw, 10.3+0.275, sx+sw+sg, 10.3+0.275, c=C["model"], lw=1.0)
    sx += sw + sg

arrow(p2_cx, 11.2, p2_cx, 10.9, c=C["model"], lw=0.8)
arrow(p2x+sw/2, 11.2, p2x+sw/2, 10.9, c=C["model"], lw=0.8)

# Transformer
mod(p2x, 9.2, p2w, 1.0, "ShiftedWindowTransformerBlock",
    "pre-LN | Attention3D | FFN | MHCConnection", C["model_bg"], ec=C["model"], lw=1.2)
arrow(p2_cx, 10.3, p2_cx, 10.25, c=C["model"], lw=0.8)

# Segmentation
mod(p2x, 8.2, p2w, 0.65, "SegmentationHead", "FCN decoder -> (B,N,38)", C["model_light"], lw=1.2)
arrow(p2_cx, 9.2, p2_cx, 8.9, c=C["model"], lw=0.8)

# Phase 3 — GAN (x=9.0, w=4.2)
p3x, p3w = 9.0, 4.2
p3_cx = p3x + p3w/2

mod(p3x, 11.2, p3w, 1.0, "StyleTransferGen", "DGCNN EdgeConv | 125K params", C["inf_bg"], ec=C["inf"], lw=1.2)
mod(p3x, 10.0, p3w, 1.0, "WGANDiscriminator", "PointNet WGAN-GP | 83K params", C["inf_light"], lw=0.8)
mod(p3x, 9.0, p3w, 0.7, "GAN Loss", "WGAN-GP + ChamferDist + NormalCos", C["opt_bg"], lw=0.8)

arrow(p3_cx, 11.2, p3_cx, 11.05, c=C["inf"], lw=0.8)
arrow(p3_cx, 10.0, p3_cx, 9.75, c=C["inf_light"], lw=0.8)

# Phase 4 — Data Loading (x=13.5, w=4.0)
p4x, p4w = 13.5, 4.0
p4_cx = p4x + p4w/2

mod(p4x, 11.2, p4w, BH, "SyntheticPointCloudDataset", ".npz | max=65,536", C["data_light"])
y4 = 11.2 - BH - BG
mod(p4x, y4, p4w, BH, "RoadMCDataModule", "Lightning | batch=4", C["data"], lw=1.2)
y4 -= BH + BG
mod(p4x, y4, p4w, BH, "RealRoadDataset", ".ply/.npy/.las", C["inf_light"])
y4 -= BH + BG
mod(p4x, y4, p4w, 0.5, "collate_fn", "padding+valid_mask", C["neutral"], lw=0.6)

for i in range(3):
    ay = 11.2 - i*(BH+BG) - 0.05
    arrow(p4_cx, ay, p4_cx, ay - BG + 0.1, c=C["opt"], lw=0.8)

# Cross-Phase Arrows (labels placed in CLEAR areas)
# P1→P4 (route at y=11.5, above blocks)
ax.annotate("", xy=(p4x, 11.5), xytext=(p1x+p1w, 11.5),
            arrowprops=dict(arrowstyle="->", color=C["data"], lw=1.3,
                            connectionstyle="arc3,rad=-0.05", shrinkA=4, shrinkB=4), zorder=2)
label((p1x+p1w+p4x)/2, 11.7, ".npz", bg=C["data_bg"])

# P4→P2 (route at y=11.0, below phase headers)
ax.annotate("", xy=(p2_cx, 11.0), xytext=(p4_cx, 11.0),
            arrowprops=dict(arrowstyle="->", color=C["data"], lw=1.3,
                            connectionstyle="arc3,rad=0.08", shrinkA=4, shrinkB=4), zorder=2)
label((p2_cx+p4_cx)/2, 11.2, "batch", fs=7, bg=C["data_bg"])

# P3→P2 (styled coords, route above Transformer)
ax.annotate("", xy=(p2_cx+0.8, 9.2), xytext=(p3x, 10.8),
            arrowprops=dict(arrowstyle="->", color=C["inf"], lw=1.0,
                            connectionstyle="arc3,rad=0.3", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=2)
label((p2_cx+0.8+p3x)/2, 9.8, "styled coords", fs=6.5, c=C["inf"], bg=C["inf_bg"])

# P2→P5 (route below SegmentationHead)
arrow(p2_cx, 8.2, p2_cx, 6.8, c=C["model"], lw=1.3)
label(p2_cx+0.8, 7.5, "logits (B,N,38)", fs=7, bg=C["model_bg"])

# Divider
ax.plot([0.3, W-0.3], [6.5, 6.5], color=C["divider"], lw=1.2, ls="--", zorder=1, alpha=0.6)

# Phase 5 — Training (bottom-left)
box(0.3, 0.5, 9.8, 5.5, fc=C["opt_bg"], ec=C["opt"], lw=1.5, alpha=0.25, rnd=0.12, z=1)
txt(5.2, 5.8, "Phase 5: Training & Evaluation", fs=12, fw="bold", c=C["opt"], z=5)

# 3 modes
mw, mg = 3.0, 0.2
mx = 0.5
for i, (nm, desc, flow, clr) in enumerate([
    ("baseline", "synthetic only", "Synthetic -> Swin3D -> L_seg", C["model_light"]),
    ("end2end", "alternating opt", "GAN + seg alternating", C["model_dark"]),
    ("gan_enhanced", "GAN pretrain", "GAN pretrain -> mixed", C["model"])]):
    mod(mx, 4.8, mw, 0.7, nm, desc, clr, lw=0.8)
    txt(mx + mw/2, 4.55, flow, fs=6, c=C["text3"], z=4)
    mx += mw + mg

# Loss function
box(0.5, 3.2, 9.4, 1.0, fc=C["panel"], ec=C["opt"], lw=1.2, rnd=0.08, z=3)
txt(5.2, 3.95, "Loss Function", fs=10, fw="bold", c=C["opt"], z=4)
txt(5.2, 3.55, r"$\mathcal{L}_{seg} = \lambda_{focal} \cdot \mathrm{FocalLoss}(\gamma=2) + \lambda_{dice} \cdot \mathrm{DiceLoss} + \lambda_{edge} \cdot \mathrm{EdgeLoss}(\mathrm{Sobel})$", fs=9, z=4)

# Optimizer
box(0.5, 2.2, 9.4, 0.6, fc="#F8F9FA", ec=C["divider"], lw=0.5, rnd=0.04, z=3)
txt(5.2, 2.5, "Optimizer: AdamW (lr=1e-4) + CosineAnnealingLR  |  Metrics: macro mIoU (38 classes)", fs=7, c=C["text2"], z=4)

txt(5.2, 1.85, "evaluate.py: per-class IoU / recall / precision  |  asphalt [1-20] / concrete [21-37]", fs=6.5, c=C["text3"], z=4)

# Inference Pipeline (bottom-right)
box(10.5, 0.5, 6.8, 5.5, fc=C["inf_bg"], ec=C["inf"], lw=1.5, alpha=0.25, rnd=0.12, z=1)
txt(13.9, 5.8, "Real Data Inference Pipeline", fs=12, fw="bold", c=C["inf"], z=5)

ix, iw = 10.7, 6.4
icx = ix + iw/2

mod(ix, 4.6, iw, 0.7, "RealRoadDataset", "Load .ply / .npy / .las point clouds", C["inf_light"], lw=0.8)
mod(ix, 3.5, iw, 0.7, "Swin3D Inference", "deploy() freezes MHC, skips Sinkhorn", C["model_bg"], lw=1.2)
mod(ix, 2.4, iw, 0.7, "JTG Point-wise Classification", "38 defect classes + visualization", C["inf"], lw=1.2)

arrow(icx, 4.6, icx, 4.25, c=C["inf"], lw=0.8)
arrow(icx, 3.5, icx, 3.15, c=C["model"], lw=0.8)

# Legend
ly = 1.2
txt(icx, ly+0.9, "Legend", fs=8, fw="bold", z=5)
lx = ix
for i, (clr, nm) in enumerate([(C["data"],"Data"),(C["model"],"Model"),(C["opt"],"Loss"),(C["inf"],"Inference")]):
    ax.plot([lx+i*1.6, lx+i*1.6+0.2], [ly+0.5, ly+0.5], color=clr, lw=2.5, zorder=5)
    txt(lx+i*1.6+0.3, ly+0.5, nm, fs=6, c=C["text2"], ha="left", z=5)
txt(icx, ly+0.2, "solid=flow  dashed=adapt", fs=6, c=C["text2"], z=5)

# GAN→P5 Coupling Arrows
ax.annotate("", xy=(3.5, 4.8), xytext=(p3x, 10.0),
            arrowprops=dict(arrowstyle="->", color=C["inf"], lw=1.0,
                            connectionstyle="arc3,rad=-0.3", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=2)
label(5.0, 7.2, "pretrained weights", fs=6.5, c=C["inf"], bg=C["inf_bg"])

ax.annotate("", xy=(6.5, 4.8), xytext=(p3x+2, 10.0),
            arrowprops=dict(arrowstyle="->", color=C["opt"], lw=1.0,
                            connectionstyle="arc3,rad=-0.2", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=2)
label(8.5, 7.5, "alternating opt", fs=6.5, c=C["opt"], bg=C["opt_bg"])

# Footnotes
txt(0.3, 0.2, "mHC = Manifold Hyper-Connection  |  JTG 5210-2018 = Highway Technical Condition Evaluation Standard",
    fs=5.5, c=C["text3"], ha="left", va="bottom", z=5)

# Save
plt.savefig("docs/architecture.png", dpi=DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.1)
print(f"docs/architecture.png generated ({DPI} dpi, {W}x{H})")
