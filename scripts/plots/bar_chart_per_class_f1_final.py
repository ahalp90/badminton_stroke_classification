"""Per-class F1 bar chart — final best run vs first Phase 2 nosides baseline.

Asymmetric comparison: the new run's bar is its best serial (S2); the baseline
bar is the 5-serial mean. Error bars span min-max across each run's serial
cell. Final best is run_20260602_143618_156220 (4 serials); baseline is
run_20260430_170325 (5 serials).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments/bst_x/shuttleset"

# Final best BST-X run; serial 2 is best by both test macro and min F1.
BEST_RUN = "run_20260602_143618_156220"
BEST_SERIAL = 2
BEST_LABEL = "final best"
# First Phase 2 nosides run (sanity A, LS=0.1); shown as 5-serial mean to match
# the 11 May presentation chart.
BASELINE_RUN = "run_20260430_170325"
BASELINE_LABEL = "first nosides (Phase 2 LS=0.1)"

OUT_PATH = REPO_ROOT / "docs/images/bar_chart_per_class_f1_final.png"

# Tol muted (protanopia-safe, qualitative). Sand for baseline, indigo for current best.
COLOUR_BASELINE = "#DDCC77"
COLOUR_BEST     = "#332288"
COLOUR_CALLOUT  = "#CC6677"  # rose for the wrist_smash annotation


def load_grid(run_id: str) -> tuple[list[str], np.ndarray, list[dict]]:
    """Read per_class_f1 for every serial of a run.

    :return: (class_names, full grid of shape (n_serials, n_classes), serials list)
    """
    manifest = yaml.safe_load((EXPERIMENTS / run_id / "manifest.yaml").read_text())
    serials = manifest["serials"]
    class_names = list(serials[0]["metrics"]["per_class_f1"].keys())
    f1_grid = np.array([
        [s["metrics"]["per_class_f1"][cls] for cls in class_names]
        for s in serials
    ])
    return class_names, f1_grid, serials


def main():
    best_classes, best_grid, best_serials = load_grid(BEST_RUN)
    base_classes, base_grid, base_serials = load_grid(BASELINE_RUN)
    assert best_classes == base_classes, "Class lists differ between runs"
    class_names = best_classes

    # New run: bar = best serial (S2 here). Baseline: bar = 5-serial mean.
    best_row_full = next(s for s in best_serials if s["serial_no"] == BEST_SERIAL)
    best_row = np.array([best_row_full["metrics"]["per_class_f1"][cls] for cls in class_names])
    best_macro = float(best_row_full["metrics"]["macro_f1"])
    base_row = base_grid.mean(axis=0)
    base_macro = float(np.mean([s["metrics"]["macro_f1"] for s in base_serials]))

    # Error bars span min-max across each run's serial cell, framed against the
    # bar value. Clip negatives (when a non-bar serial beat the bar's row).
    best_yerr = np.stack([best_row - best_grid.min(axis=0), best_grid.max(axis=0) - best_row])
    base_yerr = np.stack([base_row - base_grid.min(axis=0), base_grid.max(axis=0) - base_row])
    best_yerr = np.clip(best_yerr, 0, None)
    base_yerr = np.clip(base_yerr, 0, None)

    order = np.argsort(best_row)
    class_names = [class_names[i] for i in order]
    best_row, base_row = best_row[order], base_row[order]
    best_yerr, base_yerr = best_yerr[:, order], base_yerr[:, order]

    n = len(class_names)
    x = np.arange(n)
    bar_width = 0.4

    n_best_serials = best_grid.shape[0]
    n_base_serials = base_grid.shape[0]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(
        x - bar_width / 2, base_row, bar_width,
        yerr=base_yerr, capsize=3, color=COLOUR_BASELINE,
        label=f"{BASELINE_LABEL} ({BASELINE_RUN}; macro mean {base_macro:.3f})",
    )
    ax.bar(
        x + bar_width / 2, best_row, bar_width,
        yerr=best_yerr, capsize=3, color=COLOUR_BEST,
        label=f"{BEST_LABEL} ({BEST_RUN} S{BEST_SERIAL}; macro {best_macro:.3f})",
    )

    ax.axhline(base_macro, color=COLOUR_BASELINE, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(best_macro, color=COLOUR_BEST,     linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.5)

    # Wrist_smash callout: project's min F1 class, where the gain against
    # baseline matters most.
    ws_idx = class_names.index("wrist_smash")
    ws_base, ws_best = base_row[ws_idx], best_row[ws_idx]
    delta_pp = (ws_best - ws_base) * 100
    annotation = (
        f"min F1 (wrist_smash):\n"
        f"{ws_base:.3f} → {ws_best:.3f}  (+{delta_pp:.1f}pp)"
    )
    ax.annotate(
        annotation,
        xy=(ws_idx + bar_width / 2, ws_best),
        xytext=(ws_idx + 2.2, ws_best + 0.18),
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
        f"Per-class F1: {BEST_LABEL} (best serial) vs {BASELINE_LABEL} (serial mean) "
        f"— error bars span min-max across each cell ({n_best_serials} / {n_base_serials} serials)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
