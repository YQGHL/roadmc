"""RoadMC model pipeline diagram generator."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

W, H = 18, 13
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
txt(W/2, 12.3, "RoadMC Model Pipeline", fs=18, fw="bold", z=5)
txt(W/2, 11.9, "Swin3D Transformer + GAN Adaptation + Training + Inference", fs=10, c=C["text3"], z=5)

# Layout Constants
HDR_Y = 11.3      # header center
ULINE_Y = 11.15   # underline (must be > first block top)
BLOCK_H = 0.75
GAP = 0.15
FIRST_Y = 10.2    # first block bottom; top = 10.95 < 11.15

# Phase 2 -- Core Network (center, x=2.5, w=8.0)
p2x, p2w = 2.5, 8.0
p2_cx = p2x + p2w/2

txt(p2_cx, HDR_Y, "Core Network: Swin3D Transformer", fs=13, fw="bold", c=C["model"], z=10)
ax.plot([p2x + 0.2, p2x + p2w - 0.2], [ULINE_Y, ULINE_Y],
        color=C["model"], lw=2.5, solid_capstyle="round", zorder=10)

# Backbone
mod(p2x, FIRST_Y, p2w, BLOCK_H, "Swin3D Backbone",
    "31.2M params | embed_dim=96 | depths=[2,2,6,2]", C["model_bg"], lw=1.2)

# 4 stages
sw, sg = 1.8, 0.2
sx = p2x
stage_y = FIRST_Y - BLOCK_H - GAP
for i, (nm, cfg, clr) in enumerate([("S0","C=96",C["model_light"]),("S1","C=192",C["model"]),
                                     ("S2","C=384",C["model_dark"]),("S3","C=768",C["model_dark"])]):
    mod(sx, stage_y, sw, 0.65, nm, cfg, clr, lw=0.8)
    if i < 3:
        arrow(sx+sw, stage_y+0.325, sx+sw+sg, stage_y+0.325, c=C["model"], lw=1.2)
    sx += sw + sg

arrow(p2_cx, FIRST_Y, p2_cx, stage_y + BLOCK_H + 0.05, c=C["model"], lw=1.0)

# Transformer
trans_y = stage_y - 0.65 - GAP
mod(p2x, trans_y, p2w, 1.1, "ShiftedWindowTransformerBlock",
    "pre-LN -> WindowAttention3D -> FFN -> MHCConnection (Sinkhorn-Knopp)",
    C["model_bg"], ec=C["model"], lw=1.2)
arrow(p2_cx, stage_y, p2_cx, trans_y + 1.1 + 0.05, c=C["model"], lw=1.0)

# Segmentation Head
seg_y = trans_y - 1.1 - GAP
mod(p2x, seg_y, p2w, BLOCK_H, "SegmentationHead",
    "FCN decoder + 4x skip connections -> (B,N,38)", C["model_light"], lw=1.2)
arrow(p2_cx, trans_y, p2_cx, seg_y + BLOCK_H + 0.05, c=C["model"], lw=1.0)

# Phase 3 -- GAN Adaptation (right, x=11.0, w=5.5)
p3x, p3w = 11.0, 5.5
p3_cx = p3x + p3w/2

txt(p3_cx, HDR_Y, "GAN Domain Adaptation", fs=13, fw="bold", c=C["inf"], z=10)
ax.plot([p3x + 0.2, p3x + p3w - 0.2], [ULINE_Y, ULINE_Y],
        color=C["inf"], lw=2.5, solid_capstyle="round", zorder=10)

gan_y1 = FIRST_Y
mod(p3x, gan_y1, p3w, 0.85, "StyleTransferGen",
    "DGCNN EdgeConv encoder-decoder | 125K params", C["inf_bg"], ec=C["inf"], lw=1.2)
gan_y2 = gan_y1 - 0.85 - GAP
mod(p3x, gan_y2, p3w, 0.85, "WGANDiscriminator",
    "PointNet-style WGAN-GP critic | 83K params", C["inf_light"], lw=0.8)
gan_y3 = gan_y2 - 0.85 - GAP
mod(p3x, gan_y3, p3w, 0.65, "GAN Loss",
    "WGAN-GP + ChamferDist + NormalCosine", C["opt_bg"], lw=0.8)

arrow(p3_cx, gan_y1, p3_cx, gan_y1 - 0.42, c=C["inf"], lw=1.0)
arrow(p3_cx, gan_y2, p3_cx, gan_y2 - 0.42, c=C["inf_light"], lw=1.0)

# Styled coords arrow: from GAN to Transformer
ax.annotate("", xy=(p2x + p2w, trans_y + 0.5), xytext=(p3x, gan_y1),
            arrowprops=dict(arrowstyle="->", color=C["inf"], lw=1.0,
                            connectionstyle="arc3,rad=-0.15", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=5)
label(10.0, trans_y + 0.8, "styled coords", fs=8, c=C["inf"], bg=C["inf_bg"])

# Divider
DIVIDER_Y = 5.8
ax.plot([0.3, W-0.3], [DIVIDER_Y, DIVIDER_Y], color=C["divider"], lw=1.2, ls="--", zorder=1, alpha=0.6)

# Phase 5 -- Training (bottom-left, x=0.3, w=9.5)
box(0.3, 0.4, 9.5, 5.0, fc=C["opt_bg"], ec=C["opt"], lw=1.5, alpha=0.25, rnd=0.12, z=1)
txt(5.05, 5.15, "Training & Evaluation", fs=13, fw="bold", c=C["opt"], z=5)

# 3 modes
mw, mg = 2.9, 0.15
mx = 0.5
mode_y = 4.2
for i, (nm, desc, flow, clr) in enumerate([
    ("baseline", "synthetic only", "Synthetic -> Swin3D -> L_seg", C["model_light"]),
    ("end2end", "alternating opt", "GAN + seg alternating", C["model_dark"]),
    ("gan_enhanced", "GAN pretrain", "GAN pretrain -> mixed", C["model"])]):
    mod(mx, mode_y, mw, 0.7, nm, desc, clr, lw=0.8)
    txt(mx + mw/2, mode_y - 0.15, flow, fs=6.5, c=C["text3"], z=4)
    mx += mw + mg

# Loss function
box(0.5, 2.2, 9.1, 1.1, fc=C["panel"], ec=C["opt"], lw=1.2, rnd=0.08, z=3)
txt(5.05, 3.1, "Loss Function", fs=11, fw="bold", c=C["opt"], z=4)
txt(5.05, 2.65, r"$\mathcal{L}_{seg} = \lambda_{focal} \cdot \mathrm{FocalLoss}(\gamma=2) + \lambda_{dice} \cdot \mathrm{DiceLoss} + \lambda_{edge} \cdot \mathrm{EdgeLoss}(\mathrm{Sobel})$", fs=9, z=4)

# Optimizer
box(0.5, 1.3, 9.1, 0.6, fc="#F8F9FA", ec=C["divider"], lw=0.5, rnd=0.04, z=3)
txt(5.05, 1.6, "Optimizer: AdamW (lr=1e-4) + CosineAnnealingLR  |  Metrics: macro mIoU (38 classes)", fs=7.5, c=C["text2"], z=4)

txt(5.05, 0.65, "evaluate.py: per-class IoU / recall / precision  |  asphalt [1-20] / concrete [21-37]", fs=7, c=C["text3"], z=4)

# Inference Pipeline (bottom-right, x=10.2, w=7.3)
box(10.2, 0.4, 7.3, 5.0, fc=C["inf_bg"], ec=C["inf"], lw=1.5, alpha=0.25, rnd=0.12, z=1)
txt(13.85, 5.15, "Real Data Inference Pipeline", fs=13, fw="bold", c=C["inf"], z=5)

ix, iw = 10.4, 6.9
icx = ix + iw/2

mod(ix, 4.2, iw, 0.7, "RealRoadDataset", "Load .ply / .npy / .las point clouds", C["inf_light"], lw=0.8)
mod(ix, 3.1, iw, 0.7, "Swin3D Inference", "deploy() freezes MHC, skips Sinkhorn iterations", C["model_bg"], lw=1.2)
mod(ix, 2.0, iw, 0.7, "JTG Point-wise Classification", "38 defect classes + visualization output", C["inf"], lw=1.2)

arrow(icx, 4.2, icx, 3.85, c=C["inf"], lw=1.0)
arrow(icx, 3.1, icx, 2.75, c=C["model"], lw=1.0)

# Legend
ly = 1.0
txt(icx, ly+0.3, "solid=flow  dashed=adapt", fs=8, c=C["text2"], z=5)

# GAN -> Training Coupling Arrows
ax.annotate("", xy=(3.5, mode_y + 0.35), xytext=(p3x, gan_y2),
            arrowprops=dict(arrowstyle="->", color=C["inf"], lw=1.0,
                            connectionstyle="arc3,rad=-0.3", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=5)
label(5.5, 6.5, "pretrained weights", fs=8, c=C["inf"], bg=C["inf_bg"])

ax.annotate("", xy=(6.0, mode_y + 0.35), xytext=(p3x+2, gan_y2),
            arrowprops=dict(arrowstyle="->", color=C["opt"], lw=1.0,
                            connectionstyle="arc3,rad=-0.2", linestyle="dashed",
                            shrinkA=4, shrinkB=4), zorder=5)
label(9.0, 6.5, "alternating opt", fs=8, c=C["opt"], bg=C["opt_bg"])

# Save
plt.savefig("docs/model_pipeline.png", dpi=DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.1)
print(f"docs/model_pipeline.png generated ({DPI} dpi, {W}x{H})")
