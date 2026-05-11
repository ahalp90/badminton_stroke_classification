"""Run trajectory chart for the 2026-05-11 supervisor presentation.

Two panels sharing x-axis:
  - top: 5-serial mean macro F1 across my project runs.
  - bottom: top-serial macro F1, comparable to BST paper's single-run figures.

Sequential ablation ordering on x; no dates. Colour distinguishes taxonomy.
BST paper reference rows appear as horizontal dashed lines on the best-serial
panel only (BST published single-run, so it's the closer comparison).

Data is hand-transcribed from `scratch/architecture_notes/arch_1_directions.md`
headline-results table (lines 25-46), which is the project's source of truth.
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "scratch/presentation_prep/trajectory_chart.png"

# Tol muted palette (protanopia-safe, qualitative). Family per taxonomy.
COLOUR = {
    "merged_25": "#332288",            # indigo
    "une_merge_v1": "#44AA99",         # teal
    "une_merge_v1_nosides": "#DDCC77", # sand
    "bst_ref": "#CC6677",              # rose (external reference)
}

# Each entry: short label, taxonomy key, 5-serial mean macro F1, best-serial macro F1.
# Order is chronological; LR retune was 3 serials and is flagged in the label.
RUNS = [
    ("LR retune (3s)",           "merged_25",            0.826,  0.83),
    ("sanity C (Phase 2)",       "merged_25",            0.831,  0.83),
    ("sanity B (sides)",         "une_merge_v1",         0.739,  0.74),
    ("sanity A (nosides)",       "une_merge_v1_nosides", 0.742,  0.74),
    ("LS sweep winner",          "une_merge_v1_nosides", 0.747,  0.75),
    ("class-weighting smoke",    "une_merge_v1_nosides", 0.748,  0.76),
    ("CDB-F1 first",             "une_merge_v1_nosides", 0.7432, 0.75),
    ("capacity Run 1",           "une_merge_v1_nosides", 0.7414, 0.74),
    ("wipe_drop best",           "une_merge_v1_nosides", 0.7481, 0.76),
    ("mask-channel 2a",          "une_merge_v1_nosides", 0.7440, 0.75),
    ("jitter-off ablation",      "une_merge_v1_nosides", 0.7401, 0.74),
    ("aug v1 first",             "une_merge_v1_nosides", 0.7388, 0.75),
    ("aug v1 + p_jit=0.3",       "une_merge_v1_nosides", 0.7447, 0.75),
]

# BST paper single-run figures, from arXiv:2502.21085 Table 1 (25-class) and appendix p3 (35-class).
BST_REFS = [
    ("BST paper, 25c variable-length", 0.8097),
    ("BST paper, 25c fixed-width",     0.7983),
    ("BST paper, 35c ShuttleSet",      0.7043),
]


def main():
    n = len(RUNS)
    x = list(range(1, n + 1))
    labels = [r[0] for r in RUNS]
    taxonomies = [r[1] for r in RUNS]
    mean_macro = [r[2] for r in RUNS]
    best_macro = [r[3] for r in RUNS]
    colours = [COLOUR[t] for t in taxonomies]

    fig, (ax_mean, ax_best) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    # Mean panel: scatter + connecting line within same taxonomy runs only.
    ax_mean.plot(x, mean_macro, color="lightgrey", linewidth=1, zorder=1)
    for xi, yi, ci in zip(x, mean_macro, colours):
        ax_mean.scatter(xi, yi, color=ci, s=70, zorder=2, edgecolor="black", linewidth=0.4)
    ax_mean.set_ylabel("5-serial mean macro F1")
    ax_mean.set_ylim(0.65, 0.86)
    ax_mean.grid(axis="y", alpha=0.3)
    ax_mean.set_title("Run trajectory: macro F1 across the project's ablation sequence")

    # Best-serial panel: scatter + line + BST reference horizontal lines.
    ax_best.plot(x, best_macro, color="lightgrey", linewidth=1, zorder=1)
    for xi, yi, ci in zip(x, best_macro, colours):
        ax_best.scatter(xi, yi, color=ci, s=70, zorder=2, edgecolor="black", linewidth=0.4)

    for label, value in BST_REFS:
        ax_best.axhline(value, color=COLOUR["bst_ref"], linestyle="--", linewidth=1, alpha=0.7)
        ax_best.text(
            n + 0.3, value, f"{label} ({value:.3f})",
            fontsize=9, va="center", color=COLOUR["bst_ref"],
        )

    ax_best.set_ylabel("top-serial macro F1")
    ax_best.set_ylim(0.65, 0.86)
    ax_best.grid(axis="y", alpha=0.3)
    ax_best.set_xticks(x)
    ax_best.set_xticklabels(labels, rotation=40, ha="right")

    # Single legend for taxonomy colour, placed on the mean panel.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["merged_25"],
                   markeredgecolor="black", markersize=8, label="merged_25 (25c, retains unknown)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["une_merge_v1"],
                   markeredgecolor="black", markersize=8, label="une_merge_v1 (28c, sides; dropunk)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOUR["une_merge_v1_nosides"],
                   markeredgecolor="black", markersize=8, label="une_merge_v1_nosides (14c, nosides; dropunk)"),
        plt.Line2D([0], [0], color=COLOUR["bst_ref"], linestyle="--", linewidth=1.2,
                   label="BST paper single-run reference"),
    ]
    ax_mean.legend(handles=legend_handles, loc="lower right", fontsize=8)

    # Leave headroom on the right so the BST text labels don't get clipped.
    ax_best.set_xlim(0.5, n + 6.5)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.18, hspace=0.12)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
