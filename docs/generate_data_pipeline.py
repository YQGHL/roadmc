"""RoadMC data pipeline diagram generator."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

W, H = 16, 11
DPI = 200

C = {
    "data": "#4A90D9", "data_light": "#AED6F1", "data_bg": "#D6EAF8",
    "model": "#E07B39", "model_light": "#F5B041", "model_bg": "#FDEBD0",
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
        txt(x + w/2, y + h*0.62, title, fs=10, fw="bold", z=4)
        txt(x + w/2, y + h*0.3, sub, fs=7, c=C["text2"], z=4)
    else:
        txt(x + w/2, y + h/2, title, fs=10, fw="bold", z=4)

# Background
box(0.1, 0.1, W-0.2, H-0.2, fc=C["bg"], ec="#DEE2E6", lw=1.0, rnd=0.15, z=0)

# Title
txt(W/2, 10.3, "RoadMC Data Pipeline", fs=18, fw="bold", z=5)
txt(W/2, 9.9, "Point Cloud Generation → Batch Preparation", fs=10, c=C["text3"], z=5)

# Layout Constants
# Strict grid: headers at top, blocks below with clear gap
HDR_Y = 9.3       # header text center
ULINE_Y = 9.15    # underline y (must be > block top)
BLOCK_H = 0.8
GAP = 0.15
FIRST_Y = 8.2     # first block bottom; top = 9.0 < ULINE_Y(9.15) ✓

# Phase 1 — Data Generation (left, x=0.5, w=4.0)
p1x, p1w = 0.5, 4.0
p1_cx = p1x + p1w/2

# Phase header (text only, no box)
txt(p1_cx, HDR_Y, "Data Generation", fs=13, fw="bold", c=C["data"], z=10)
ax.plot([p1x + 0.2, p1x + p1w - 0.2], [ULINE_Y, ULINE_Y],
        color=C["data"], lw=2.5, solid_capstyle="round", zorder=10)

# 4 blocks
y = FIRST_Y
mod(p1x, y, p1w, BLOCK_H, "GeneratorConfig", "dataclass configuration", C["data_light"])
y -= BLOCK_H + GAP
mod(p1x, y, p1w, BLOCK_H, "primitives.py", "11 physical primitives", C["data_light"])
y -= BLOCK_H + GAP
mod(p1x, y, p1w, BLOCK_H, "SyntheticRoadDataset", "10-step pipeline", C["data"], lw=1.2)
y -= BLOCK_H + GAP
mod(p1x, y, p1w, BLOCK_H, "Output: .npz", "points / labels / feats / normals", C["data_bg"])

# Vertical arrows
for i in range(3):
    ay = FIRST_Y - i*(BLOCK_H+GAP) - 0.05
    arrow(p1_cx, ay, p1_cx, ay - GAP + 0.1, c=C["data"], lw=1.0)

# Annotation (below Output block, no overlap)
ann_y = y - GAP - 0.6
box(p1x, ann_y, p1w, 0.6, fc="#F8F9FA", ec=C["divider"], lw=0.5, alpha=0.7, rnd=0.04, z=3)
txt(p1_cx, ann_y + 0.35, "feats: intensity, curvature, crack-dist", fs=7, c=C["text3"], z=4)
txt(p1_cx, ann_y + 0.15, "normals: unit surface normals", fs=7, c=C["text3"], z=4)

# Cross Arrow: P1 → P4 (route in clear space between blocks)
# Arrow from P1 Output to P4 SyntheticPointCloudDataset
arrow_end_y = FIRST_Y + BLOCK_H*0.5  # middle of first P4 block
ax.annotate("", xy=(8.5, arrow_end_y), xytext=(p1x+p1w, FIRST_Y),
            arrowprops=dict(arrowstyle="->", color=C["data"], lw=1.5,
                            connectionstyle="arc3,rad=0.05", shrinkA=5, shrinkB=5), zorder=5)
# Label in clear space above arrow
label(7.0, FIRST_Y + 0.55, ".npz files", fs=9, bg=C["data_bg"])

# Phase 4 — Data Loading (right, x=8.5, w=4.5)
p4x, p4w = 8.5, 4.5
p4_cx = p4x + p4w/2

# Phase header
txt(p4_cx, HDR_Y, "Data Loading", fs=13, fw="bold", c=C["opt"], z=10)
ax.plot([p4x + 0.2, p4x + p4w - 0.2], [ULINE_Y, ULINE_Y],
        color=C["opt"], lw=2.5, solid_capstyle="round", zorder=10)

# 4 blocks
y = FIRST_Y
mod(p4x, y, p4w, BLOCK_H, "SyntheticPointCloudDataset", ".npz | max=65,536 points", C["data_light"])
y -= BLOCK_H + GAP
mod(p4x, y, p4w, BLOCK_H, "RoadMCDataModule", "Lightning | batch_size=4", C["data"], lw=1.2)
y -= BLOCK_H + GAP
mod(p4x, y, p4w, BLOCK_H, "RealRoadDataset", ".ply / .npy / .las", C["inf_light"])
y -= BLOCK_H + GAP
mod(p4x, y, p4w, 0.55, "collate_fn", "padding + valid_mask (pad label=-1)", C["neutral"], lw=0.6)

# Vertical arrows
for i in range(3):
    ay = FIRST_Y - i*(BLOCK_H+GAP) - 0.05
    arrow(p4_cx, ay, p4_cx, ay - GAP + 0.1, c=C["opt"], lw=1.0)

# Output — Batch Tensor (far right)
out_x, out_w = 13.5, 2.2
out_cx = out_x + out_w/2
out_top = FIRST_Y + 0.3
out_h = 5.2

box(out_x, out_top - out_h, out_w, out_h, fc=C["model_bg"], ec=C["model"],
    lw=1.5, alpha=0.3, rnd=0.12, z=1)
txt(out_cx, out_top - 0.3, "Output Tensor", fs=12, fw="bold", c=C["model"], z=5)

# 5 output blocks
oy = out_top - 0.9
outputs = [
    ("coords", "B×N×3 float32"),
    ("feats", "B×N×3 float32"),
    ("labels", "B×N int32 [0,37]"),
    ("valid_mask", "B×N bool"),
    ("normals", "B×N×3 float32"),
]
for title, sub in outputs:
    mod(out_x + 0.1, oy, out_w - 0.2, 0.65, title, sub, C["model_light"], lw=0.8)
    oy -= 0.65 + 0.12

# Arrow from P4 collate_fn to Output Tensor
ax.annotate("", xy=(out_x, out_top - 1.5), xytext=(p4_cx, FIRST_Y - 3*(BLOCK_H+GAP)),
            arrowprops=dict(arrowstyle="->", color=C["model"], lw=1.2,
                            connectionstyle="arc3,rad=0.15", shrinkA=5, shrinkB=5), zorder=5)
label(12.5, 6.0, "batch tensor", fs=8, c=C["model"], bg=C["model_bg"])

# Data Flow Summary (bottom)
box(0.5, 0.4, 12.5, 1.3, fc=C["panel"], ec=C["divider"], lw=1.0, rnd=0.08, z=2)
txt(6.75, 1.35, "Data Flow Summary", fs=11, fw="bold", c=C["text"], z=5)
txt(6.75, 0.9,
    "GeneratorConfig → primitives (ISO 8608 PSD + fBm + 11 defects) → SyntheticRoadDataset → .npz",
    fs=8, c=C["text2"], z=4)
txt(6.75, 0.6,
    ".npz → SyntheticPointCloudDataset → RoadMCDataModule → collate_fn → batch tensor",
    fs=8, c=C["text2"], z=4)

# Legend (bottom-left, below annotation)
leg_y = 3.5
txt(1.5, leg_y + 0.3, "Legend", fs=9, fw="bold", z=5)
for i, (clr, nm) in enumerate([(C["data"],"Synthetic"),(C["opt"],"Loading"),(C["model"],"Output")]):
    ax.plot([0.7+i*1.6, 0.7+i*1.6+0.3], [leg_y, leg_y], color=clr, lw=3, zorder=5)
    txt(0.7+i*1.6+0.45, leg_y, nm, fs=8, c=C["text2"], ha="left", z=5)

# Save
plt.savefig("docs/data_pipeline.png", dpi=DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.1)
print(f"docs/data_pipeline.png generated ({DPI} dpi, {W}x{H})")
