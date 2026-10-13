"""Draws the model architecture diagram used in the README (SVG and PNG, no data needed).

    python docs/make_architecture_figure.py        # writes docs/figures/architecture.svg and architecture.png

Needs matplotlib. Text is converted to outlines in the SVG, so it looks the same on every machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"

INK = "#111827"
MUTED = "#4b5563"
ARROW = "#6b7280"
FAMILY = {  # fill, edge
    "input": ("#f3f4f6", "#9ca3af"),
    "feature": ("#f1f5f9", "#94a3b8"),
    "anchor": ("#dbeafe", "#2a78d6"),
    "corrector": ("#ede9fe", "#8b5cf6"),
    "neural": ("#d1fae5", "#1baf7a"),
    "terms": ("#ffedd5", "#eb6834"),
    "plain": ("#ffffff", "#9ca3af"),
    "bypass": ("#ffffff", "#9ca3af"),
}
TITLE_H, LINE_H = 2.1, 1.5
TITLE_COLOR = {"feature": "#475569", "bypass": "#6b7280", "neural": "#0f8a5f", "corrector": "#7c3aed"}


def group(ax, x, y, w, h, title, family):
    fill, edge = FAMILY[family]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3", fc=fill, ec=edge, lw=1.3,
                                alpha=0.45, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3", fc="none", ec=edge, lw=1.3,
                                zorder=2))
    ax.text(x + 1.4, y + h - 1.0, title, ha="left", va="top", fontsize=9.2, fontweight="bold",
            color=TITLE_COLOR.get(family, edge), zorder=5)


def box(ax, x, y, w, h, title, lines=(), family="plain", dashed=False, title_color=INK, size=8.8, sub=7.7):
    fill, edge = FAMILY[family]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge, lw=1.4,
                                ls="--" if dashed else "-", zorder=3))
    block = TITLE_H + LINE_H * len(lines)
    top = y + (h + block) / 2
    ax.text(x + w / 2, top - TITLE_H / 2, title, ha="center", va="center", fontsize=size, fontweight="bold",
            color=title_color, zorder=5)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - TITLE_H - LINE_H * (i + 0.5), line, ha="center", va="center", fontsize=sub,
                color=MUTED, zorder=5)
    return (x, y, w, h)


def arrow(ax, points, dashed=False, label=None, label_at=None, label_ha="left"):
    xs, ys = zip(*points)
    ls = (0, (4, 3)) if dashed else "-"
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=ARROW, lw=1.4, ls=ls, solid_capstyle="butt", zorder=4)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=11),
                zorder=4)
    if len(points) == 2 and not dashed:
        pass
    if label:
        lx, ly = label_at
        ax.text(lx, ly, label, ha=label_ha, va="center", fontsize=7.8, color=MUTED, style="italic", zorder=5)


def draw(png: Path | None, svg: Path | None) -> None:
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "pitch-control-architecture",
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9.6, 10.1))
    ax.set_xlim(0, 96)
    ax.set_ylim(-4.5, 98.5)
    ax.axis("off")

    # ---- input ------------------------------------------------------------------------------------------------
    box(ax, 28, 92, 40, 5.2, "One evaluation row", ["game state, players, as-of channels"], "input")
    arrow(ax, [(48, 92), (48, 89.2)])

    # ---- features ---------------------------------------------------------------------------------------------
    group(ax, 2, 72, 92, 17, "Row-local features and frozen train-derived joins (fitted on the training file only)",
          "feature")
    chips = [
        ("v3 features", ["count, base-out,", "platoon, leverage,", "cold start"]),
        ("CE cells", ["pitcher rate and", "pitcher × platoon", "cell rate"]),
        ("Current state", ["season-to-date vs", "preseason snapshot,", "reliability"]),
        ("F03 and R9", ["batter history gap,", "same-hand reverse", "trait"]),
        ("Latent command", ["pitch-type policy", "× pitcher command", "by pitch type"]),
        ("History join", ["frozen pitcher", "history, current", "gap family"]),
    ]
    cw, gap = 13.6, 1.4
    for i, (t, ls_) in enumerate(chips):
        box(ax, 3.4 + i * (cw + gap), 73.5, cw, 10.8, t, ls_, "feature", size=8.4, sub=7.3)

    # ---- anchor -----------------------------------------------------------------------------------------------
    group(ax, 2, 47, 44, 22, "Anchor: CatBoost on the 5-class axis-A target", "anchor")
    box(ax, 4, 55.2, 19.4, 9.8, "R5 head", ["base + history_current_gap", "families, 3 seeds each"], "anchor")
    box(ax, 24.8, 55.2, 19.4, 9.8, "R8 DROP_BALL head", ["frozen, never retrained", "same two families"], "anchor")
    box(ax, 7, 48.6, 38 - 0, 5.2, "p0 = w·R5 + (1 − w)·R8", ["offset −0.0088, w = 0.938"], "plain", size=8.6)
    arrow(ax, [(13.7, 55.2), (13.7, 53.8)])
    arrow(ax, [(34.5, 55.2), (34.5, 53.8)])

    # ---- residual correctors ----------------------------------------------------------------------------------
    group(ax, 50, 47, 44, 22, "Residual correctors", "corrector")
    box(ax, 52, 59.6, 40, 5.6, "Pitcher residual history", ["frozen join, 18 columns"], "corrector", size=8.6)
    box(ax, 52, 48.8, 19.4, 8.6, "c_b", ["8 CatBoost regressors,", "trained on 2024"], "corrector")
    box(ax, 72.6, 48.8, 19.4, 8.6, "c_v", ["8 regressors,", "2024 July onwards"], "corrector")
    arrow(ax, [(61.7, 59.6), (61.7, 57.4)])
    arrow(ax, [(82.3, 59.6), (82.3, 57.4)])

    # features -> anchor / correctors
    arrow(ax, [(13.7, 72), (13.7, 65)])
    arrow(ax, [(34.5, 72), (34.5, 65)])
    arrow(ax, [(72, 72), (72, 65.2)], label="row features", label_at=(73.4, 70.5))

    # ---- combine ----------------------------------------------------------------------------------------------
    box(ax, 26, 38, 44, 6, "p1 = clip(p0 + 1.2·c_b − 0.3·c_v)", [], "plain", size=9)
    arrow(ax, [(33, 48.6), (33, 44)])                  # p0 -> p1
    arrow(ax, [(72, 47), (72, 41), (70, 41)])          # correctors -> p1
    # ---- split ------------------------------------------------------------------------------------------------
    box(ax, 40, 31.5, 16, 3.8, "game_type", [], "plain", size=8.6)
    arrow(ax, [(48, 38), (48, 35.3)])

    # R rows ----------------------------------------------------------------------------------------------------
    group(ax, 2, 14, 71, 14.5, "R rows (regular season)", "neural")
    box(ax, 4, 15.5, 21.6, 9.6, "M1 network", ["0.75·p1 + 0.25·M1", "embedding MLP, 9 seeds"], "neural")
    box(ax, 27.7, 15.5, 21.6, 9.6, "DL member", ["0.8·p + 0.2·DL, 9 seeds,", "embeddings + explicit", "season-to-date state"],
        "neural")
    box(ax, 51.4, 15.5, 19.8, 9.6, "Tail terms", ["logit shift: as-of", "success slope,", "3-0 bump"], "terms")
    arrow(ax, [(25.6, 20.3), (27.7, 20.3)])
    arrow(ax, [(49.3, 20.3), (51.4, 20.3)])
    # F rows ----------------------------------------------------------------------------------------------------
    group(ax, 75, 14, 19, 14.5, "F rows", "bypass")
    box(ax, 76.4, 15.5, 16.2, 9.6, "unchanged", ["second-team games", "skip M1, DL and", "tail terms"], "bypass", dashed=True)

    # split -> branches
    arrow(ax, [(48, 31.5), (48, 30), (37.5, 30), (37.5, 28.5)], label="R rows", label_at=(36.5, 30.9), label_ha="right")
    arrow(ax, [(48, 30), (84.5, 30), (84.5, 28.5)], dashed=True, label="F rows", label_at=(85.7, 30.9))

    # ---- component-aux and output -----------------------------------------------------------------------------
    box(ax, 4, 5.8, 88, 5.8, "Component-aux correction",
        ["logit shift of 0.05 × a residual head (small models fitted from a frozen 2024 artifact)"], "terms", size=8.8)
    arrow(ax, [(61.3, 14), (61.3, 11.6)])
    arrow(ax, [(84.5, 14), (84.5, 11.6)], dashed=True)
    box(ax, 30, 0.4, 36, 3.6, "control_success probability", [], "plain", size=9)
    arrow(ax, [(48, 5.8), (48, 4.0)])

    ax.text(48, -2.8, "Every input is the row itself or a frozen, train-derived artifact; no other evaluation row participates.",
            ha="center", va="center", fontsize=7.8, color=MUTED, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    if svg:
        fig.savefig(svg, format="svg", facecolor="white",
                    metadata={"Date": None, "Title": "Model architecture", "Creator": "matplotlib"})
    if png:
        fig.savefig(png, format="png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw(OUT / "architecture.png", OUT / "architecture.svg")
    print("wrote", OUT / "architecture.svg", "and", OUT / "architecture.png", file=sys.stdout)
