"""Regenerate the two README pipeline diagrams.

Design constraints:
- No in-image title: the surrounding markdown heading owns the title.
- ~2.2:1 aspect so body text stays legible at README width.
- Low-saturation palette, one hue per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent


def draw_pipeline(
    filename: str,
    stages: list[tuple[str, str, list[str]]],
    footer_label: str,
    footer_text: str,
    footer_note: str,
    *,
    edge: str,
    fill: str,
    tag: str,
    text: str,
) -> None:
    n = len(stages)
    fig, ax = plt.subplots(figsize=(16.0, 7.0), dpi=160)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    gap = 3.6
    box_w = (100 - 2 - gap * (n - 1)) / n
    box_h = 22.0
    y0 = 11.0

    for i, (step, title, lines) in enumerate(stages):
        x0 = 1 + i * (box_w + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                box_w,
                box_h,
                boxstyle="round,pad=0.4,rounding_size=1.4",
                linewidth=1.8,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        ax.text(
            x0 + 1.4,
            y0 + box_h - 2.4,
            step,
            fontsize=11.5,
            fontweight="bold",
            color=tag,
            ha="left",
            va="center",
        )
        ax.text(
            x0 + box_w / 2,
            y0 + box_h - 7.4,
            title,
            fontsize=15,
            fontweight="bold",
            color=text,
            ha="center",
            va="center",
            linespacing=1.15,
        )
        ax.text(
            x0 + box_w / 2,
            y0 + 5.6,
            "\n".join(lines),
            fontsize=11.5,
            color=text,
            ha="center",
            va="center",
            linespacing=1.6,
        )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + box_w + 0.8, y0 + box_h / 2),
                    (x0 + box_w + gap - 0.8, y0 + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=20,
                    linewidth=2.0,
                    color=edge,
                )
            )

    ax.plot([1, 99], [6.4, 6.4], color=edge, linewidth=1.0, alpha=0.35)
    ax.text(1, 4.4, footer_label, fontsize=10.5, fontweight="bold", color=tag, ha="left")
    ax.text(99, 4.4, footer_note, fontsize=10.5, color=tag, ha="right")
    ax.text(1, 1.4, footer_text, fontsize=13.5, fontweight="bold", color=text, ha="left")

    fig.savefig(OUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    draw_pipeline(
        "synthesis_pipeline.png",
        stages=[
            ("01  PRIORS", "Surface\npriors", ["pavement type", "ISO 8608 class", "scene size + seed"]),
            ("02  REALIZE", "Road\nsurface", ["PSD roughness", "fBm micro-texture", "surface normals"]),
            ("03  DEFORM", "Damage\ngeometry", ["37 disease labels", "physical primitives", "point supervision"]),
            ("04  OBSERVE", "LiDAR\nsimulation", ["scan-line resampling", "range/angular noise", "natural prevalence"]),
            ("05  EXPORT", "Scene\ncontract", ["XYZ + intensity", "observable features", "resolution + audit"]),
        ],
        footer_label="OUTPUT",
        footer_text="Train/validation .npz scenes with versioned feature and resolution contracts",
        footer_note="Synthesis remains part of the method",
        edge="#2e5f8a",
        fill="#edf3f9",
        tag="#2e5f8a",
        text="#1c3a55",
    )

    draw_pipeline(
        "training_pipeline.png",
        stages=[
            ("01  LOAD", "Scene\nloader", ["schema validation", "legacy recompute", "curriculum labels"]),
            ("02  SAMPLE", "Split-aware\nsampling", ["train: damage-aware", "val/test: uniform", "padding + mask"]),
            ("03  EMBED", "Input\nembedding", ["XYZ + 3 features", "linear projection", "LayerNorm"]),
            ("04  ENCODE", "Backbone", ["Swin3D windows", "or PointMamba", "multi-stage features"]),
            ("05  DECODE", "mHC +\ndecoder", ["channel mixing", "skip fusion", "point-wise logits"]),
            ("06  REPORT", "Objective +\nmetrics", ["Focal + Dice + Edge", "Muon / AdamW", "mIoU + ECE + CI"]),
        ],
        footer_label="EVALUATION CONTRACT",
        footer_text="Threshold selection and independent reporting use disjoint scenes",
        footer_note="binary -> four -> eight -> full38",
        edge="#b3641f",
        fill="#fbf2e9",
        tag="#b3641f",
        text="#4a2f14",
    )
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
