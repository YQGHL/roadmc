"""
RoadMC System Architecture Diagram Generator (v3)
Professional redesign following strict information visualization principles.

Design principles:
- Decouple architecture description from implementation details
- Strict color semantics (one color = one meaning)
- All arrows orthogonal (no diagonals)
- Three-level typography system
- Grid-aligned layout with uniform padding/gaps
- Academic presentation with legend and footnotes

Saves to docs/architecture.png for READMEs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global Config ────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

FIGSIZE = (17, 13)
DPI = 250
PAD = 0.15       # uniform block padding
GAP = 0.25       # uniform gap between blocks

# ── Color Semantics (strictly one color = one meaning) ───────────────────────
# Data flow / pipeline: Blue family
C_DATA_PRIMARY   = "#4A90D9"
C_DATA_SECONDARY = "#7FB3D9"
C_DATA_LIGHT     = "#AED6F1"
C_DATA_BG        = "#D6EAF8"

# Model components / network: Warm orange-red family
C_MODEL_PRIMARY   = "#E07B39"
C_MODEL_SECONDARY = "#D35400"
C_MODEL_LIGHT     = "#F5B041"
C_MODEL_BG        = "#FDEBD0"

# Optimization / loss / math: Purple family
C_OPT_PRIMARY   = "#8E44AD"
C_OPT_SECONDARY = "#BB8FCE"
C_OPT_LIGHT     = "#D2B4DE"
C_OPT_BG        = "#EBDEF0"

# Inference / deployment: Green family
C_INF_PRIMARY   = "#27AE60"
C_INF_SECONDARY = "#58D68D"
C_INF_LIGHT     = "#ABEBC6"
C_INF_BG        = "#D5F5E3"

# Neutral / structural
C_BORDER       = "#2C3E50"
C_TEXT_DARK    = "#1A1A2E"
C_TEXT_MID     = "#555555"
C_TEXT_LIGHT   = "#888888"
C_BG           = "#FAFBFC"
C_PANEL        = "#FFFFFF"
C_DIVIDER      = "#BDC3C7"
C_FOOTNOTE     = "#95A5A6"

# ── Figure Setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, 17)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Helper Functions ─────────────────────────────────────────────────────────

def draw_box(x, y, w, h, facecolor, edgecolor=C_BORDER, lw=1.2, alpha=1.0, zorder=3,
             rounding=0.08):
    """Draw a rounded rectangle patch."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder, alpha=alpha
    )
    ax.add_patch(p)
    return p

def draw_text(x, y, text, fontsize=8.5, fontweight="normal", color=C_TEXT_DARK,
              ha="center", va="center", zorder=4, style="normal"):
    """Draw text at position."""
    ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight, color=color,
            ha=ha, va=va, zorder=zorder, style=style)

def draw_module_block(x, y, w, h, title, subtitle="", color=C_DATA_PRIMARY,
                      lw=1.2, alpha=1.0, zorder=3):
    """
    Draw a module block with L1 title and optional L2 subtitle.
    L1: bold, centered, dark text
    L2: normal, centered, mid-gray
    """
    draw_box(x, y, w, h, facecolor=color, edgecolor=C_BORDER, lw=lw, alpha=alpha, zorder=zorder)
    if subtitle:
        draw_text(x + w/2, y + h*0.62, title, fontsize=11, fontweight="bold",
                  color=C_TEXT_DARK, zorder=zorder+1)
        draw_text(x + w/2, y + h*0.32, subtitle, fontsize=8.5, fontweight="normal",
                  color=C_TEXT_MID, zorder=zorder+1)
    else:
        draw_text(x + w/2, y + h/2, title, fontsize=11, fontweight="bold",
                  color=C_TEXT_DARK, zorder=zorder+1)

def draw_multi_line_block(x, y, w, h, lines, color=C_DATA_LIGHT, edgecolor=C_BORDER,
                          lw=0.8, alpha=1.0, zorder=3):
    """
    Draw a block with multiple lines of text.
    First line: L1 style (bold, 11pt)
    Subsequent lines: L2 style (normal, 8.5pt)
    """
    draw_box(x, y, w, h, facecolor=color, edgecolor=edgecolor, lw=lw, alpha=alpha, zorder=zorder)
    n = len(lines)
    total_h = (n - 1) * 0.22
    start_y = y + h/2 + total_h/2
    for i, (txt, fs, fw, clr) in enumerate(lines):
        draw_text(x + w/2, start_y - i * 0.22, txt, fontsize=fs, fontweight=fw,
                  color=clr, zorder=zorder+1)

def draw_straight_arrow(x1, y1, x2, y2, color=C_DATA_PRIMARY, lw=1.3, style="->", zorder=2):
    """Draw a straight (vertical or horizontal) arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw,
        connectionstyle="arc3,rad=0",
        shrinkA=5, shrinkB=5,
        zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

def draw_ortho_arrow(x1, y1, x2, y2, color=C_DATA_PRIMARY, lw=1.3, style="->", zorder=2):
    """
    Draw an orthogonal arrow using manual polyline routing.
    Routes: horizontal from start to midpoint x, then vertical to end y.
    """
    if abs(x2 - x1) < 0.01 or abs(y2 - y1) < 0.01:
        return draw_straight_arrow(x1, y1, x2, y2, color, lw, style, zorder)

    # Use midpoint x for the turn point
    x_mid = (x1 + x2) / 2
    # Draw the polyline (two segments)
    ax.plot([x1, x_mid, x2], [y1, y1, y2], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    # Add arrowhead at the end
    ax.annotate("", xy=(x2, y2), xytext=(x_mid, y2),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0",
                                shrinkA=0, shrinkB=5),
                zorder=zorder+1)
    return None

def draw_dashed_arrow(x1, y1, x2, y2, color=C_DATA_PRIMARY, lw=1.0, zorder=2):
    """Draw a dashed orthogonal arrow using manual polyline routing."""
    if abs(x2 - x1) < 0.01 or abs(y2 - y1) < 0.01:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->", color=color, lw=lw,
            connectionstyle="arc3,rad=0",
            linestyle="dashed",
            shrinkA=5, shrinkB=5,
            zorder=zorder
        )
        ax.add_patch(arrow)
        return arrow

    x_mid = (x1 + x2) / 2
    ax.plot([x1, x_mid, x2], [y1, y1, y2], color=color, lw=lw,
            linestyle="dashed", solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=(x2, y2), xytext=(x_mid, y2),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=0",
                                linestyle="dashed",
                                shrinkA=0, shrinkB=5),
                zorder=zorder+1)
    return None

def draw_arrow_label(x, y, text, fontsize=7.5, color=C_TEXT_MID, style="italic",
                     ha="center", va="center", zorder=5, bbox_color=None):
    """Draw a label on an arrow with optional background box."""
    if bbox_color:
        ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
                zorder=zorder, style=style,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=bbox_color,
                          edgecolor="none", alpha=0.85))
    else:
        ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
                zorder=zorder, style=style)

def draw_phase_header(x, y, w, h, title, subtitle, color):
    """Draw a phase header with colored background."""
    draw_box(x, y, w, h, facecolor=color, edgecolor=color, lw=0, alpha=0.12, zorder=1)
    draw_text(x + w/2, y + h*0.6, title, fontsize=10, fontweight="bold",
              color=color, zorder=2)
    draw_text(x + w/2, y + h*0.3, subtitle, fontsize=7, fontweight="normal",
              color=C_TEXT_LIGHT, zorder=2)

def draw_annotation_box(x, y, w, h, lines, color=C_TEXT_LIGHT, fontsize=7, zorder=3):
    """Draw a side annotation box with L3 footnote-style text."""
    draw_box(x, y, w, h, facecolor="#F8F9FA", edgecolor=C_DIVIDER, lw=0.6,
             alpha=0.7, zorder=zorder)
    n = len(lines)
    total_h = (n - 1) * 0.18
    start_y = y + h/2 + total_h/2
    for i, txt in enumerate(lines):
        draw_text(x + w/2, start_y - i * 0.18, txt, fontsize=fontsize,
                  fontweight="normal", color=color, zorder=zorder+1, style="italic")

# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND PANEL
# ═════════════════════════════════════════════════════════════════════════════
draw_box(0.15, 0.15, 16.7, 12.7, facecolor=C_BG, edgecolor="#DEE2E6", lw=1.0,
         alpha=1.0, zorder=0, rounding=0.12)

# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
draw_text(8.5, 11.55, "RoadMC System Architecture", fontsize=18, fontweight="bold",
          color=C_TEXT_DARK, zorder=5)
draw_text(8.5, 11.25,
          "Physics-Simulation-Driven  \u00b7  Math-Constraint-Enhanced  \u00b7  Pavement Point Cloud Defect Detection",
          fontsize=9, fontweight="normal", color=C_TEXT_LIGHT, zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE HEADERS (top row)
# ═════════════════════════════════════════════════════════════════════════════
draw_phase_header(0.3, 10.55, 3.2, 0.42, "Phase 1: Data Generation",
                  "Physical Simulation", C_DATA_PRIMARY)
draw_phase_header(3.75, 10.55, 5.0, 0.42, "Phase 2: Core Network",
                  "Swin3D + mHC (Manifold Hyper-Connection)", C_MODEL_PRIMARY)
draw_phase_header(9.0, 10.55, 3.5, 0.42, "Phase 3: GAN Domain Adaptation",
                  "Style Transfer", C_INF_SECONDARY)
draw_phase_header(12.75, 10.55, 4.0, 0.42, "Phase 4 & 5: Data Loading \u00b7 Training",
                  "Pipeline \u00b7 Evaluation", C_OPT_SECONDARY)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Generation (left column)
# ═════════════════════════════════════════════════════════════════════════════

# Module blocks
draw_module_block(0.4, 8.8, 3.0, 0.55, "GeneratorConfig",
                  "dataclass configuration", color=C_DATA_LIGHT, lw=0.8)
draw_module_block(0.4, 7.9, 3.0, 0.55, "primitives.py",
                  "11 physical primitive functions", color=C_DATA_LIGHT, lw=0.8)
draw_module_block(0.4, 7.0, 3.0, 0.55, "SyntheticRoadDataset",
                  "Synthetic point cloud generator", color=C_DATA_PRIMARY, lw=1.2)

# Vertical arrows between P1 modules
draw_straight_arrow(1.9, 8.8, 1.9, 8.45, color=C_DATA_PRIMARY, lw=1.0)
draw_straight_arrow(1.9, 7.9, 1.9, 7.55, color=C_DATA_PRIMARY, lw=1.0)

# Pipeline steps removed - information already in module blocks
# (GeneratorConfig, primitives.py, SyntheticRoadDataset contain the details)

# Output box
draw_module_block(0.4, 5.9, 3.0, 0.55, "Output: .npz files",
                  "points / labels / feats / normals", color=C_DATA_BG, lw=0.8)
draw_straight_arrow(1.9, 7.0, 1.9, 6.45, color=C_DATA_PRIMARY, lw=1.0)

# Feature channels detail (annotation) - below output box
draw_annotation_box(0.4, 4.95, 3.0, 0.6, [
    "feats (N,3): intensity, curvature, crack-dist",
    "normals (N,3): unit surface normals",
])

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Core Network (center column)
# ═════════════════════════════════════════════════════════════════════════════

# Swin3D Backbone header
draw_module_block(3.9, 9.3, 4.8, 0.6, "Swin3D Backbone",
                  "31.2M params  |  embed_dim=96  |  depths=[2,2,6,2]",
                  color=C_MODEL_BG, lw=1.5)

# 4 Stage boxes (uniform grid)
stage_cfg = [
    ("S0", "C=96, d=2, h=3", C_MODEL_LIGHT),
    ("S1", "C=192, d=2, h=6", C_MODEL_PRIMARY),
    ("S2", "C=384, d=6, h=12", C_MODEL_SECONDARY),
    ("S3", "C=768, d=2, h=24", C_MODEL_SECONDARY),
]
stage_w = 1.15
stage_h = 0.55
stage_gap = 0.2
stage_start_x = 3.9
stage_y = 8.35
for i, (name, cfg, col) in enumerate(stage_cfg):
    x = stage_start_x + i * (stage_w + stage_gap)
    draw_module_block(x, stage_y, stage_w, stage_h, name, cfg,
                      color=col, lw=0.8)
    if i < 3:
        draw_straight_arrow(x + stage_w, stage_y + stage_h/2,
                           x + stage_w + stage_gap, stage_y + stage_h/2,
                           color=C_MODEL_PRIMARY, lw=1.2)

# Arrow from backbone to stages
draw_straight_arrow(6.3, 9.3, 6.3, 8.85, color=C_MODEL_PRIMARY, lw=1.0)

# Transformer Block
draw_multi_line_block(3.9, 6.8, 4.8, 1.1, [
    ("ShiftedWindowTransformerBlock", 11, "bold", C_TEXT_DARK),
    ("pre-LN residual structure", 8.5, "normal", C_TEXT_MID),
    ("LN \u2192 WindowAttention3D \u2192 \u2295 residual", 8, "normal", C_TEXT_MID),
    ("LN \u2192 FFN \u2192 \u2295 residual", 8, "normal", C_TEXT_MID),
    ("MHCConnection (Sinkhorn-Knopp algorithm)", 8, "normal", C_TEXT_MID),
], color=C_MODEL_BG, edgecolor=C_MODEL_PRIMARY, lw=1.2)

draw_straight_arrow(6.3, 8.35, 6.3, 7.9, color=C_MODEL_PRIMARY, lw=1.0)

# Segmentation Head
draw_module_block(3.9, 5.6, 4.8, 0.6, "SegmentationHead",
                  "FCN decoder + 4\u00d7 skip connections \u2192 (B,N,38)",
                  color=C_MODEL_LIGHT, lw=1.2)
draw_straight_arrow(6.3, 6.8, 6.3, 6.2, color=C_MODEL_PRIMARY, lw=1.0)

# Skip connections - already described in SegmentationHead subtitle

# MHC Detail box (side annotation) - repositioned to avoid arrow overlap
draw_multi_line_block(9.5, 6.2, 2.9, 1.0, [
    ("mHC (Manifold Hyper-Connection)", 9.5, "bold", C_TEXT_DARK),
    ("Doubly-stochastic channel mixing", 7, "normal", C_TEXT_MID),
    ("M = softplus(W\u2081)\u00b7softplus(W\u2082)\u1D40", 7, "normal", C_TEXT_MID),
    ("H = SinkhornKnopp(M/\u03c4)  5 iter", 7, "normal", C_TEXT_MID),
    ("y = x + H\u00b7r   deploy() freezes H", 7, "normal", C_TEXT_MID),
], color=C_OPT_BG, edgecolor=C_OPT_PRIMARY, lw=1.0)

# Arrow from transformer block to MHC
draw_straight_arrow(8.7, 6.7, 9.5, 6.7, color=C_OPT_PRIMARY, lw=0.8, style="->")
draw_arrow_label(9.1, 6.9, "embedding", fontsize=6, color=C_OPT_PRIMARY)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — GAN Domain Adaptation (upper right)
# ═════════════════════════════════════════════════════════════════════════════

# Generator
draw_multi_line_block(9.0, 8.8, 3.3, 1.1, [
    ("StyleTransferGen", 11, "bold", C_TEXT_DARK),
    ("DGCNN EdgeConv encoder-decoder", 8.5, "normal", C_TEXT_MID),
    ("Encoder: 6\u219264\u2192128\u2192256  (k=16)", 7.5, "normal", C_TEXT_MID),
    ("Decoder: 256\u2192128\u219264\u21926", 7.5, "normal", C_TEXT_MID),
    ("Output: [\u0394x,\u0394y,\u0394z, \u0394nx,\u0394ny,\u0394nz]", 7.5, "normal", C_TEXT_MID),
    ("Params: 125,574", 7.5, "normal", C_TEXT_MID),
], color=C_INF_BG, edgecolor=C_INF_PRIMARY, lw=1.2)

# Discriminator
draw_multi_line_block(9.0, 7.4, 3.3, 1.0, [
    ("WGANDiscriminator", 11, "bold", C_TEXT_DARK),
    ("PointNet-style WGAN-GP critic", 8.5, "normal", C_TEXT_MID),
    ("MLP: 6\u219264\u2192128\u2192256\u2192 maxpool", 7.5, "normal", C_TEXT_MID),
    ("Global: 256\u2192128\u219264\u21921  (no sigmoid)", 7.5, "normal", C_TEXT_MID),
    ("Params: 83,009", 7.5, "normal", C_TEXT_MID),
], color=C_INF_LIGHT, edgecolor=C_INF_SECONDARY, lw=0.8)

# GAN internal arrows
draw_straight_arrow(10.65, 8.8, 10.65, 8.4, color=C_INF_PRIMARY, lw=1.0)
draw_straight_arrow(10.65, 7.4, 10.65, 7.8, color=C_INF_SECONDARY, lw=0.8, style="->")

# GAN Losses - positioned above discriminator, beside generator
draw_module_block(12.6, 8.6, 3.8, 0.55, "GAN Loss",
                  "WGAN-GP + ChamferDist + NormalCosine",
                  color=C_OPT_BG, lw=0.8)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Data Loading (mid right)
# ═════════════════════════════════════════════════════════════════════════════

draw_module_block(12.75, 7.2, 3.65, 0.55, "SyntheticPointCloudDataset",
                  "Reads .npz  |  max_points=65,536",
                  color=C_DATA_LIGHT, lw=0.8)

draw_module_block(12.75, 6.35, 3.65, 0.55, "RoadMCDataModule",
                  "Lightning DataModule  |  batch_size=4",
                  color=C_DATA_PRIMARY, lw=1.2)

draw_module_block(12.75, 5.5, 3.65, 0.55, "RealRoadDataset",
                  ".ply / .npy / .las \u2192 JTG label mapping",
                  color=C_INF_LIGHT, lw=0.8)

# Data augmentation note
draw_text(14.575, 5.15, "Augmentation: Z-rotation + translation + scaling",
          fontsize=7, color=C_TEXT_LIGHT, style="italic")

# Vertical arrows
draw_straight_arrow(14.575, 7.2, 14.575, 6.9, color=C_DATA_PRIMARY, lw=1.0)
draw_straight_arrow(14.575, 6.35, 14.575, 6.05, color=C_DATA_PRIMARY, lw=1.0)

# Collate detail
draw_module_block(12.75, 4.65, 3.65, 0.45, "collate_fn",
                  "padding + valid_mask (pad label=-1)",
                  color="#F0F0F0", lw=0.6)

# ═════════════════════════════════════════════════════════════════════════════
# CROSS-PHASE ARROWS (orthogonal routing)
# ═════════════════════════════════════════════════════════════════════════════

# P1 output \u2192 P4 SyntheticPointCloudDataset
# Route below SegmentationHead to avoid overlap
_p1_xmid = (3.4 + 12.75) / 2
ax.plot([3.4, _p1_xmid, 12.75], [5.6, 5.6, 7.475], color=C_DATA_PRIMARY, lw=1.3,
        solid_capstyle="round", zorder=2)
ax.annotate("", xy=(12.75, 7.475), xytext=(_p1_xmid, 7.475),
            arrowprops=dict(arrowstyle="->", color=C_DATA_PRIMARY, lw=1.3,
                            connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=5), zorder=3)
draw_arrow_label(11.5, 6.0, ".npz files", fontsize=7.5, color=C_DATA_PRIMARY,
                 bbox_color=C_DATA_BG)

# P4 RoadMCDataModule \u2192 P2 Swin3D Backbone
# Route above all blocks
_p4_xmid = (12.75 + 6.3) / 2
ax.plot([12.75, _p4_xmid, 6.3], [6.625, 6.625, 9.95], color=C_DATA_PRIMARY, lw=1.3,
        solid_capstyle="round", zorder=2)
ax.annotate("", xy=(6.3, 9.95), xytext=(_p4_xmid, 9.95),
            arrowprops=dict(arrowstyle="->", color=C_DATA_PRIMARY, lw=1.3,
                            connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=5), zorder=3)
draw_arrow_label(9.5, 10.15, "batch dict {coords, feats, labels}",
                 fontsize=7, color=C_DATA_PRIMARY, bbox_color=C_DATA_BG)

# P3 Generator \u2192 P2 (stylized coords) \u2014 dashed
# Route above backbone
_p3_xmid = (9.0 + 6.3) / 2
ax.plot([9.0, _p3_xmid, 6.3], [9.9, 9.9, 7.35], color=C_INF_SECONDARY, lw=1.0,
        linestyle="dashed", solid_capstyle="round", zorder=2)
ax.annotate("", xy=(6.3, 7.35), xytext=(_p3_xmid, 7.35),
            arrowprops=dict(arrowstyle="->", color=C_INF_SECONDARY, lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle="dashed",
                            shrinkA=0, shrinkB=5), zorder=3)
draw_arrow_label(7.5, 10.1, "styled coords (B,N,6)", fontsize=7,
                 color=C_INF_SECONDARY, bbox_color=C_INF_BG)

# P2 SegmentationHead \u2192 Phase 5
draw_straight_arrow(6.3, 5.6, 6.3, 4.3, color=C_MODEL_PRIMARY, lw=1.3)
draw_arrow_label(6.8, 4.95, "per-point logits (B,N,38)", fontsize=7.5,
                 color=C_MODEL_PRIMARY, bbox_color=C_MODEL_BG)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5 \u2014 Training & Evaluation (bottom-left)
# ═════════════════════════════════════════════════════════════════════════════

# Training panel
draw_box(0.3, 0.5, 10.5, 3.8, facecolor=C_OPT_BG, edgecolor=C_OPT_PRIMARY,
         lw=1.5, alpha=0.25, zorder=1, rounding=0.12)

draw_text(5.55, 4.1, "Phase 5: Training & Evaluation", fontsize=12,
          fontweight="bold", color=C_OPT_PRIMARY, zorder=5)

# 3 Training mode boxes (uniform grid)
modes = [
    ("baseline", "Pure synthetic data training",
     "Synthetic \u2192 Swin3D \u2192 \u2112_seg", C_MODEL_LIGHT),
    ("gan_enhanced", "GAN pretrain \u2192 mixed training",
     "GAN pretrain \u2192 freeze \u2192 stylize \u2192 Swin3D", C_MODEL_PRIMARY),
    ("end2end", "End-to-end alternating optimization",
     "GAN + segmentation alternating updates", C_MODEL_SECONDARY),
]
mode_w = 3.2
mode_h = 0.7
mode_gap = 0.25
mode_start_x = 0.5
mode_y = 3.1
for i, (name, desc, flow, col) in enumerate(modes):
    x = mode_start_x + i * (mode_w + mode_gap)
    draw_module_block(x, mode_y, mode_w, mode_h, name, desc,
                      color=col, lw=0.8)
    draw_text(x + mode_w/2, mode_y - 0.15, flow, fontsize=6.5,
              color=C_TEXT_MID, ha="center", va="top", style="italic")

# Loss function box
draw_box(0.5, 1.6, 10.0, 1.15, facecolor=C_PANEL, edgecolor=C_OPT_PRIMARY,
         lw=1.2, zorder=3, rounding=0.1)

draw_text(5.5, 2.55, "Loss Function", fontsize=10, fontweight="bold",
          color=C_OPT_PRIMARY, zorder=4)

# Loss formula using mathtext
loss_formula = (
    r"$\mathcal{L}_{seg} = \lambda_{focal} \cdot \mathrm{FocalLoss}(\gamma=2) "
    r"+ \lambda_{dice} \cdot \mathrm{DiceLoss} "
    r"+ \lambda_{edge} \cdot \mathrm{EdgeLoss}(\mathrm{Sobel})$"
)
draw_text(5.5, 2.15, loss_formula, fontsize=9, fontweight="normal",
          color=C_TEXT_DARK, zorder=4)

# Optimizer & metrics (separate bottom bar)
draw_box(0.5, 0.65, 10.0, 0.7, facecolor="#F8F9FA", edgecolor=C_DIVIDER,
         lw=0.6, zorder=3, rounding=0.06)

draw_text(5.5, 1.15,
          "Optimizer: AdamW (lr=1e\u207b\u2074) + CosineAnnealingLR    |    "
          "Metrics: macro mIoU (38 JTG 5210-2018 classes)",
          fontsize=7.5, color=C_TEXT_MID, zorder=4)

# Evaluate box
draw_text(5.5, 0.55,
          "evaluate.py: per-class IoU / recall / precision  |  "
          "asphalt group [0-20] / cement group [21-37] / overall mIoU  |  JSON report",
          fontsize=7, color=C_TEXT_MID, ha="center", style="italic")

# ═════════════════════════════════════════════════════════════════════════════
# REAL DATA INFERENCE PIPELINE (bottom-right)
# ═════════════════════════════════════════════════════════════════════════════

# Inference panel (equal height to training panel)
draw_box(11.1, 0.5, 5.6, 3.8, facecolor=C_INF_BG, edgecolor=C_INF_PRIMARY,
         lw=1.5, alpha=0.25, zorder=1, rounding=0.12)

draw_text(13.9, 4.1, "Real Data Inference Pipeline", fontsize=12,
          fontweight="bold", color=C_INF_PRIMARY, zorder=5)

# Inference flow boxes
draw_module_block(11.4, 3.1, 5.0, 0.55, "RealRoadDataset",
                  "Load .ply / .npy / .las point clouds",
                  color=C_INF_LIGHT, lw=0.8)

draw_module_block(11.4, 2.25, 5.0, 0.55, "Swin3D Inference",
                  "deploy() freezes MHC, skips Sinkhorn iterations",
                  color=C_MODEL_BG, lw=1.2)

draw_module_block(11.4, 1.4, 5.0, 0.55, "JTG Point-wise Classification",
                  "38 defect classes + visualization output",
                  color=C_INF_PRIMARY, lw=1.2)

# Vertical arrows
draw_straight_arrow(13.9, 3.1, 13.9, 2.8, color=C_INF_PRIMARY, lw=1.0)
draw_straight_arrow(13.9, 2.25, 13.9, 1.95, color=C_MODEL_PRIMARY, lw=1.0)

# ═════════════════════════════════════════════════════════════════════════════
# GAN \u2192 Phase 5 COUPLING ARROWS (dashed, explicit)
# ═════════════════════════════════════════════════════════════════════════════

# Dashed arrow: GAN pretrained weights \u2192 mixed training
# Route far left, below SegmentationHead
_gan1_xmid = (10.65 + 1.0) / 2
ax.plot([10.65, _gan1_xmid, 1.0], [6.0, 6.0, 3.1], color=C_INF_SECONDARY, lw=1.0,
        linestyle="dashed", solid_capstyle="round", zorder=2)
ax.annotate("", xy=(1.0, 3.1), xytext=(_gan1_xmid, 3.1),
            arrowprops=dict(arrowstyle="->", color=C_INF_SECONDARY, lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle="dashed",
                            shrinkA=0, shrinkB=5), zorder=3)
draw_arrow_label(5.5, 4.55, "pretrained generator weights \u2192 mixed training",
                 fontsize=7, color=C_INF_SECONDARY, bbox_color=C_INF_BG,
                 style="italic")

# Dashed arrow: alternating optimization loop
# Route above the first dashed arrow
_gan2_xmid = (12.3 + 1.0) / 2
ax.plot([12.3, _gan2_xmid, 1.0], [7.5, 7.5, 3.1], color=C_OPT_SECONDARY, lw=1.0,
        linestyle="dashed", solid_capstyle="round", zorder=2)
ax.annotate("", xy=(1.0, 3.1), xytext=(_gan2_xmid, 3.1),
            arrowprops=dict(arrowstyle="->", color=C_OPT_SECONDARY, lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle="dashed",
                            shrinkA=0, shrinkB=5), zorder=3)
draw_arrow_label(6.5, 5.4, "alternating optimization loop",
                 fontsize=7, color=C_OPT_SECONDARY, bbox_color=C_OPT_BG,
                 style="italic")

# (Single dashed arrow for alternating optimization loop already drawn above)

# ═════════════════════════════════════════════════════════════════════════════
# HORIZONTAL DIVIDER between architecture and training/inference
# ═════════════════════════════════════════════════════════════════════════════
ax.plot([0.3, 16.7], [4.45, 4.45], color=C_DIVIDER, lw=1.2, linestyle="--",
        zorder=1, alpha=0.6)

# ═════════════════════════════════════════════════════════════════════════════
# LEGEND (bottom-right corner, inside inference panel area)
# ════════════════════════════════════════════════════════════════════════════
legend_x = 11.3
legend_y = 0.2
legend_w = 5.2
legend_h = 0.3

# Legend title
draw_text(legend_x + legend_w/2, legend_y + legend_h + 0.2, "Legend",
          fontsize=8, fontweight="bold", color=C_TEXT_DARK, zorder=5)

# Legend items - horizontal layout
legend_items = [
    (C_DATA_PRIMARY, "Data flow"),
    (C_MODEL_PRIMARY, "Model components"),
    (C_OPT_PRIMARY, "Optimization/loss"),
    (C_INF_PRIMARY, "Inference"),
]
for i, (color, label) in enumerate(legend_items):
    lx = legend_x + i * 1.3
    ly = legend_y + legend_h
    ax.plot([lx, lx + 0.25], [ly, ly], color=color, lw=2.5, zorder=5)
    draw_text(lx + 0.35, ly, label, fontsize=6, color=C_TEXT_MID,
              ha="left", va="center", zorder=5)

# Arrow style legend
arrow_legend_y = legend_y - 0.15
ax.annotate("", xy=(legend_x + 0.25, arrow_legend_y),
            xytext=(legend_x, arrow_legend_y),
            arrowprops=dict(arrowstyle="->", color=C_TEXT_DARK, lw=1.5,
                            connectionstyle="arc3,rad=0"), zorder=5)
draw_text(legend_x + 0.35, arrow_legend_y, "Data flow", fontsize=6,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

ax.annotate("", xy=(legend_x + 0.25, arrow_legend_y - 0.18),
            xytext=(legend_x, arrow_legend_y - 0.18),
            arrowprops=dict(arrowstyle="->", color=C_TEXT_DARK, lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle="dashed"), zorder=5)
draw_text(legend_x + 0.35, arrow_legend_y - 0.18, "Domain adaptation", fontsize=6,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

# Border style legend
border_legend_y = arrow_legend_y - 0.4
draw_box(legend_x, border_legend_y - 0.06, 0.25, 0.12,
         facecolor="none", edgecolor=C_TEXT_DARK, lw=1.5, zorder=5)
draw_text(legend_x + 0.35, border_legend_y, "Main pipeline", fontsize=6,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

draw_box(legend_x, border_legend_y - 0.22, 0.25, 0.12,
         facecolor="none", edgecolor=C_TEXT_DARK, lw=0.8, zorder=5)
draw_text(legend_x + 0.35, border_legend_y - 0.22, "Auxiliary module", fontsize=6,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

ax.annotate("", xy=(legend_x + 0.3, arrow_legend_y - 0.22),
            xytext=(legend_x, arrow_legend_y - 0.22),
            arrowprops=dict(arrowstyle="->", color=C_TEXT_DARK, lw=1.0,
                            connectionstyle="arc3,rad=0", linestyle="dashed"), zorder=5)
draw_text(legend_x + 0.45, arrow_legend_y - 0.22, "Optional / domain adaptation", fontsize=6.5,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

# Border style legend
border_legend_y = arrow_legend_y - 0.5
draw_box(legend_x, border_legend_y - 0.08, 0.3, 0.16,
         facecolor="none", edgecolor=C_TEXT_DARK, lw=1.5, zorder=5)
draw_text(legend_x + 0.45, border_legend_y, "Model backbone / main pipeline", fontsize=6.5,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

draw_box(legend_x, border_legend_y - 0.28, 0.3, 0.16,
         facecolor="none", edgecolor=C_TEXT_DARK, lw=0.8, zorder=5)
draw_text(legend_x + 0.45, border_legend_y - 0.2, "Auxiliary module / component", fontsize=6.5,
          color=C_TEXT_MID, ha="left", va="center", zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTNOTES (bottom-left)
# ═════════════════════════════════════════════════════════════════════════════
footnote_y = 0.15
footnote_x = 0.3
draw_text(footnote_x, footnote_y,
          "Abbreviations: mHC = Manifold Hyper-Connection  |  "
          "JTG 5210-2018 = Highway Technical Condition Evaluation Standard  |  "
          "Sinkhorn-Knopp = doubly-stochastic matrix normalization algorithm",
          fontsize=6, color=C_FOOTNOTE, ha="left", va="bottom", style="italic", zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
plt.savefig("docs/architecture.png", dpi=DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.1)
print(f"[PASS] docs/architecture.png generated ({DPI} dpi, {FIGSIZE[0]}x{FIGSIZE[1]})")
