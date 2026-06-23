"""Run trajectory chart with macro F1 and min wrist_smash F1 per panel.

Sibling to ``trajectory_chart.py`` for the 2026-05-11 supervisor presentation.
Two panels sharing x-axis:
  - top: 5-serial mean macro F1 and min wrist_smash F1, with BST (Chang) mean refs.
  - bottom: top-serial macro + min wrist_smash, with BST (Chang) best-serial refs.

Macro is a filled circle, min wrist_smash a white-faced square. Colour stays as
taxonomy on both, so each run sits as a circle / square pair stacked vertically.
Linear y-axis covers ~0.30 to 0.86: macro band clusters near the top, min ws
band sits below, no overlap.

Data is hand-transcribed from `docs/architecture_notes/bst_x_overview.md`
headline-results table (lines 25-46). If a run gets renamed, mirror it in
``trajectory_chart.py`` so the two figures stay in sync.
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "local_scratch/presentation_prep/trajectory_chart_macro_and_min.png"

# Tol muted palette (protanopia-safe, qualitative). One colour per taxonomy.
COLOUR = {
    "merged_25": "#332288",            # indigo
    "une_merge_v1": "#44AA99",         # teal
    "une_merge_v1_nosides": "#DDCC77", # sand
    "bst_ref": "#CC6677",              # rose (external reference)
}

# Each entry: short label, taxonomy key, then four metrics:
# 5-serial mean macro, top-serial macro, 5-serial mean min wrist_smash, top-serial min wrist_smash.
# Phase 2 + sides carries an asterisk pointing to the dropunk-ablation caveat at the
# bottom of the figure.
RUNS = [
    ("LR retune (3 serial)",                                                  "merged_25",            0.826,  0.83, 0.607,  0.63),
    ("Phase 1 baseline (merged_25)",                                          "merged_25",            0.829,  0.83, 0.600,  0.60),
    ("Keypoints fixed (Phase 2) (25c)",                                       "merged_25",            0.831,  0.83, 0.577,  0.58),
    ("Phase 2 + sides (28c)\nuncollapsed smash & drop*",                      "une_merge_v1",         0.739,  0.74, 0.317,  0.32),
    ("Phase 2 + nosides (14c)",                                               "une_merge_v1_nosides", 0.742,  0.74, 0.375,  0.40),
    ("LS=0.15",                                                               "une_merge_v1_nosides", 0.747,  0.75, 0.417,  0.45),
    ("Class weights 2× smash/ws",                                             "une_merge_v1_nosides", 0.748,  0.76, 0.422,  0.52),
    ("CDB-F1 γ=1 τ=1 [focal loss]",                                           "une_merge_v1_nosides", 0.7432, 0.75, 0.4621, 0.49),
    ("MLP head 400→1200",                                                     "une_merge_v1_nosides", 0.7414, 0.74, 0.4138, 0.44),
    ("shuttle_zero_fix [wipe_drop]",                                          "une_merge_v1_nosides", 0.7481, 0.76, 0.4742, 0.49),
    ("shuttle_mask",                                                          "une_merge_v1_nosides", 0.7440, 0.75, 0.4568, 0.49),
    ("jitter-off",                                                            "une_merge_v1_nosides", 0.7401, 0.74, 0.4301, 0.48),
    ("aug v1",                                                                "une_merge_v1_nosides", 0.7388, 0.75, 0.4750, 0.50),
    ("aug v1 + p_jit=0.3",                                                    "une_merge_v1_nosides", 0.7447, 0.75, 0.4779, 0.51),
]

FOOTNOTE = (
    "*No clean drop unknown ablation.\n"
    "Progression suggests though that\n"
    "removing it lowered macro,\n"
    "rather than raise it."
)

# BST-CG-AP (Chang's BST, 25-class merged_25): serial mean on the mean panel,
# best serial on the top-serial panel. Tuple: macro F1, min wrist_smash F1.
BST_REF_MEAN = (0.8097, 0.5762)
BST_REF_BEST = (0.821, 0.611)


def scatter_macro(ax, x, ys, colours):
    """Filled circle markers, one per run, colour by taxonomy."""
    for xi, yi, ci in zip(x, ys, colours):
        ax.scatter(xi, yi, color=ci, s=70, zorder=3,
                   edgecolor="black", linewidth=0.4, marker="o")


def scatter_min_ws(ax, x, ys, colours):
    """White-faced square markers with coloured rim; visually distinct from macro."""
    for xi, yi, ci in zip(x, ys, colours):
        ax.scatter(xi, yi, facecolor="white", edgecolor=ci, s=80, zorder=3,
                   linewidth=1.6, marker="s")


def draw_panel(ax, x, macro_vals, min_vals, colours, ylabel):
    """Plot both metrics on one panel with connecting lines per metric."""
    ax.plot(x, macro_vals, color="lightgrey", linewidth=1, zorder=1)
    ax.plot(x, min_vals,   color="lightgrey", linewidth=1, zorder=1, linestyle=":")
    scatter_macro(ax, x, macro_vals, colours)
    scatter_min_ws(ax, x, min_vals, colours)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.30, 0.86)
    ax.grid(axis="y", alpha=0.3)


def draw_bst_refs(ax, x_text, mean_panel: bool) -> None:
    """Horizontal BST-CG-AP reference lines plus paired labels.

    Macro label sits just above the macro line; min wrist_smash label sits just
    below the min line, so the label pair frames the BST band rather than
    crowding it.

    :param ax: target axis
    :param x_text: x-coordinate (data space) where label text starts; pick a
        value past the last data point so labels live in the right margin.
    :param mean_panel: True on the 5-serial-mean panel (draws BST's serial mean);
        False on the top-serial panel (draws BST's best serial).
    """
    descriptor = "mean" if mean_panel else "best"
    macro, min_ws = BST_REF_MEAN if mean_panel else BST_REF_BEST
    offset = 0.008  # vertical gap between line and label, in F1-axis units
    ax.axhline(macro,  color=COLOUR["bst_ref"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(min_ws, color=COLOUR["bst_ref"], linestyle="--", linewidth=1, alpha=0.55)
    ax.text(x_text, macro + offset,
            f"BST (Chang) {descriptor} macro ({macro:.3f})",
            fontsize=9, va="bottom", color=COLOUR["bst_ref"])
    ax.text(x_text, min_ws - offset,
            f"BST (Chang) {descriptor} min ws ({min_ws:.3f})",
            fontsize=9, va="top", color=COLOUR["bst_ref"])


def main():
    n = len(RUNS)
    x = list(range(1, n + 1))
    labels = [r[0] for r in RUNS]
    taxonomies = [r[1] for r in RUNS]
    mean_macro = [r[2] for r in RUNS]
    best_macro = [r[3] for r in RUNS]
    mean_min   = [r[4] for r in RUNS]
    best_min   = [r[5] for r in RUNS]
    colours = [COLOUR[t] for t in taxonomies]

    fig, (ax_mean, ax_best) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    draw_panel(ax_mean, x, mean_macro, mean_min, colours, "5-serial mean F1")
    ax_mean.set_title("Run trajectory: macro F1 and min wrist_smash F1 across the project's ablation sequence")
    draw_bst_refs(ax_mean, n + 0.3, mean_panel=True)

    draw_panel(ax_best, x, best_macro, best_min, colours, "top-serial F1")
    draw_bst_refs(ax_best, n + 0.3, mean_panel=False)

    ax_best.set_xticks(x)
    ax_best.set_xticklabels(labels, rotation=40, ha="right")

    # Legend: taxonomy colours on the left half, marker-shape key on the right half.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["merged_25"],
                   markeredgecolor="black", markersize=8, label="merged_25 (25c, retains unknown)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["une_merge_v1"],
                   markeredgecolor="black", markersize=8, label="une_merge_v1 (28c, sides; dropunk)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["une_merge_v1_nosides"],
                   markeredgecolor="black", markersize=8, label="une_merge_v1_nosides (14c, nosides; dropunk)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="grey",
                   markeredgecolor="black", markersize=8, label="macro F1 (filled circle)"),
        plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor="white",
                   markeredgecolor="grey", markersize=8, markeredgewidth=1.4,
                   label="min wrist_smash F1 (open square)"),
        plt.Line2D([0], [0], color=COLOUR["bst_ref"], linestyle="--", linewidth=1.2,
                   label="BST (Chang) reference (mean / best per panel)"),
    ]
    ax_mean.legend(handles=legend_handles, loc="lower right", fontsize=8, ncols=2)

    # Headroom on the right so the BST text labels don't get clipped.
    ax_best.set_xlim(0.5, n + 6.5)

    # Footnote box anchored to the bottom-right of the top-serial panel; sits below the
    # BST reference labels and to the right of the data, in otherwise-empty space.
    ax_best.text(0.985, 0.03, FOOTNOTE, transform=ax_best.transAxes,
                 fontsize=8, ha="right", va="bottom",
                 style="italic", color="dimgrey",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="lightgrey", alpha=0.95))

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.22, hspace=0.12)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
