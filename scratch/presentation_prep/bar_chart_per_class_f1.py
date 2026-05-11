"""Per-class F1 bar chart for the 2026-05-11 supervisor presentation.

Compares the current best aug-v1 run against the first Phase 2 nosides baseline
(`run_20260430_170325`, LS=0.1 sanity-A; the first time the 14-class nosides
collation was trained against). Both runs share the post-Phase-2 keypoint base,
so the gap reads as the cumulative effect of the loss-side sweep, the
shuttle-unzeroing data fix, and aug v1.

Bars are 5-serial means; error bars span min-max across serials. Sorted ascending
by aug-v1 mean F1 so the smash / wrist_smash floor sits at the left. The
wrist_smash bars carry an annotation callout because that's the project's min F1
class and the lift there is the headline story.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "src/bst_refactor/stroke_classification/main_on_shuttleset/experiments"

# Current best: aug v1 + p_jitter=0.3 (project min-F1 high).
AUG_V1_RUN = "run_20260505_154907"
AUG_V1_LABEL = "aug v1 + p_jit=0.3"
# First Phase 2 nosides run (sanity A, LS=0.1); the cleanest "before any nosides
# ablations" baseline at the same keypoint era as the aug run.
BASELINE_RUN = "run_20260430_170325"
BASELINE_LABEL = "first nosides (Phase 2 LS=0.1)"

OUT_PATH = REPO_ROOT / "scratch/presentation_prep/bar_chart_per_class_f1.png"

# Tol muted (protanopia-safe, qualitative). Sand for baseline, indigo for current best.
COLOUR_BASELINE = "#DDCC77"
COLOUR_AUG      = "#332288"
COLOUR_CALLOUT  = "#CC6677"  # rose for the wrist_smash annotation


def load_per_class(run_id: str) -> tuple[list[str], np.ndarray, float]:
    """Read per_class_f1 across all serials of a run.

    :param run_id: experiment run directory name
    :return: (class_names_in_order, per_serial_f1 of shape (n_serials, n_classes), 5-serial mean macro)
    """
    manifest = yaml.safe_load((EXPERIMENTS / run_id / "manifest.yaml").read_text())
    serials = manifest["serials"]
    class_names = list(serials[0]["metrics"]["per_class_f1"].keys())
    f1_grid = np.array([
        [s["metrics"]["per_class_f1"][cls] for cls in class_names]
        for s in serials
    ])
    macro_mean = float(np.mean([s["metrics"]["macro_f1"] for s in serials]))
    return class_names, f1_grid, macro_mean


def main():
    aug_classes,  aug_f1,  aug_macro  = load_per_class(AUG_V1_RUN)
    base_classes, base_f1, base_macro = load_per_class(BASELINE_RUN)
    assert aug_classes == base_classes, "Class lists differ between runs"
    class_names = aug_classes

    aug_mean  = aug_f1.mean(axis=0)
    base_mean = base_f1.mean(axis=0)
    aug_yerr  = np.stack([aug_mean  - aug_f1.min(axis=0),  aug_f1.max(axis=0)  - aug_mean])
    base_yerr = np.stack([base_mean - base_f1.min(axis=0), base_f1.max(axis=0) - base_mean])

    order = np.argsort(aug_mean)
    class_names = [class_names[i] for i in order]
    aug_mean,  base_mean = aug_mean[order],  base_mean[order]
    aug_yerr,  base_yerr = aug_yerr[:, order], base_yerr[:, order]

    n = len(class_names)
    x = np.arange(n)
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(
        x - bar_width / 2, base_mean, bar_width,
        yerr=base_yerr, capsize=3, color=COLOUR_BASELINE,
        label=f"{BASELINE_LABEL} ({BASELINE_RUN}; macro mean {base_macro:.3f})",
    )
    ax.bar(
        x + bar_width / 2, aug_mean, bar_width,
        yerr=aug_yerr, capsize=3, color=COLOUR_AUG,
        label=f"{AUG_V1_LABEL} ({AUG_V1_RUN}; macro mean {aug_macro:.3f})",
    )

    ax.axhline(base_macro, color=COLOUR_BASELINE, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(aug_macro,  color=COLOUR_AUG,      linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.5)

    # Wrist_smash sits leftmost (sorted ascending by aug mean F1). Pull its delta out as
    # the headline annotation; that's the project's min F1 class and where the gain
    # against baseline matters most.
    ws_idx = class_names.index("wrist_smash")
    ws_base, ws_aug = base_mean[ws_idx], aug_mean[ws_idx]
    delta_pp = (ws_aug - ws_base) * 100
    annotation = (
        f"min F1 (wrist_smash):\n"
        f"{ws_base:.3f} → {ws_aug:.3f}  (+{delta_pp:.1f}pp)"
    )
    ax.annotate(
        annotation,
        xy=(ws_idx + bar_width / 2, ws_aug),
        xytext=(ws_idx + 2.2, ws_aug + 0.18),
        ha="left", va="bottom",
        fontsize=10, fontweight="bold", color=COLOUR_CALLOUT,
        arrowprops=dict(arrowstyle="->", color=COLOUR_CALLOUT, linewidth=1.4),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLOUR_CALLOUT, alpha=0.95),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Per-class F1: {AUG_V1_LABEL} vs {BASELINE_LABEL} "
        f"(5-serial mean; error bars span min-max across serials)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
