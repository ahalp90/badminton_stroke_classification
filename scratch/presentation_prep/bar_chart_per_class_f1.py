"""Per-class F1 bar chart for the 2026-05-11 supervisor presentation.

Compares the current best aug-v1 run against the wipe_drop data-side best, both on
une_merge_v1_nosides (14 classes), 5 serials each. Bars are 5-serial means;
error bars span min-max across serials. Sorted ascending by aug-v1 mean F1 so
the smash / wrist_smash floor sits at the left.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "src/bst_refactor/stroke_classification/main_on_shuttleset/experiments"

# Current best aug-v1 + p_jitter=0.3 (project min-F1 high).
AUG_V1_RUN = "run_20260505_154907"
# Shuttle-unzeroing wipe_drop, no aug; the data-path reference best.
WIPE_DROP_RUN = "run_20260503_172922"

OUT_PATH = REPO_ROOT / "scratch/presentation_prep/bar_chart_per_class_f1.png"

# Tol muted (protanopia-safe pastel qualitative): indigo + sand for the pair.
COLOUR_AUG = "#332288"
COLOUR_WIPE = "#DDCC77"


def load_per_class(run_id: str) -> tuple[list[str], np.ndarray, float]:
    """Read per_class_f1 across all serials of a run.

    :param run_id: experiment run directory name
    :return: (class_names_in_order, per_serial_f1 of shape (n_serials, n_classes), 5-serial mean macro)
    """
    manifest = yaml.safe_load((EXPERIMENTS / run_id / "manifest.yaml").read_text())
    serials = manifest["serials"]
    # Class order is the dict insertion order of the first serial's per_class_f1.
    class_names = list(serials[0]["metrics"]["per_class_f1"].keys())
    f1_grid = np.array([
        [s["metrics"]["per_class_f1"][cls] for cls in class_names]
        for s in serials
    ])
    macro_mean = float(np.mean([s["metrics"]["macro_f1"] for s in serials]))
    return class_names, f1_grid, macro_mean


def main():
    aug_classes, aug_f1, aug_macro = load_per_class(AUG_V1_RUN)
    wipe_classes, wipe_f1, wipe_macro = load_per_class(WIPE_DROP_RUN)
    assert aug_classes == wipe_classes, "Class lists differ between runs"
    class_names = aug_classes

    aug_mean = aug_f1.mean(axis=0)
    wipe_mean = wipe_f1.mean(axis=0)
    # asymmetric yerr: (down, up) = (mean - min, max - mean)
    aug_yerr = np.stack([aug_mean - aug_f1.min(axis=0), aug_f1.max(axis=0) - aug_mean])
    wipe_yerr = np.stack([wipe_mean - wipe_f1.min(axis=0), wipe_f1.max(axis=0) - wipe_mean])

    order = np.argsort(aug_mean)
    class_names = [class_names[i] for i in order]
    aug_mean, wipe_mean = aug_mean[order], wipe_mean[order]
    aug_yerr, wipe_yerr = aug_yerr[:, order], wipe_yerr[:, order]

    n = len(class_names)
    x = np.arange(n)
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(
        x - bar_width / 2, aug_mean, bar_width,
        yerr=aug_yerr, capsize=3, color=COLOUR_AUG,
        label=f"aug v1 (run_20260505_154907; macro mean {aug_macro:.3f})",
    )
    ax.bar(
        x + bar_width / 2, wipe_mean, bar_width,
        yerr=wipe_yerr, capsize=3, color=COLOUR_WIPE,
        label=f"wipe_drop (run_20260503_172922; macro mean {wipe_macro:.3f})",
    )

    ax.axhline(aug_macro, color=COLOUR_AUG, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(wipe_macro, color=COLOUR_WIPE, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Per-class F1: aug v1 vs wipe_drop (5-serial mean; error bars span min-max across serials)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
