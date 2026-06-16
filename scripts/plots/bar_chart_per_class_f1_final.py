"""Per-class F1 bar chart — final best BST-X run vs first clean-keypoint baseline.

Like-for-like comparison: each bar is that run's manifest-declared best serial.
Error bars span min-max across the run's full serial cell so seed spread is
still visible. Final best is run_20260602_143618_156220 (4 serials, S2 best);
baseline is run_20260430_170325 (5 serials, S4 best) — the first nosides run
on the cleaned keypoint extraction in this taxonomy.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments/bst_x/shuttleset"

BEST_RUN = "run_20260602_143618_156220"
BEST_LABEL = "final best"
BASELINE_RUN = "run_20260430_170325"
BASELINE_LABEL = "sticky_anchor frame recovery; lr & module scheduling; ls=0.1"

OUT_PATH = REPO_ROOT / "docs/images/bar_chart_class_f1_14c_bst_x.png"

# Tol muted (protanopia-safe, qualitative). Sand for baseline, indigo for current best.
COLOUR_BASELINE = "#DDCC77"
COLOUR_BEST     = "#332288"
COLOUR_CALLOUT  = "#CC6677"  # rose for the wrist_smash annotation


def load_grid(run_id: str) -> tuple[list[str], np.ndarray, list[dict], int]:
    """Read per_class_f1 for every serial of a run plus its declared best serial.

    :return: (class_names, full grid of shape (n_serials, n_classes), serials list, best_serial_no)
    """
    manifest = yaml.safe_load((EXPERIMENTS / run_id / "manifest.yaml").read_text())
    serials = manifest["serials"]
    class_names = list(serials[0]["metrics"]["per_class_f1"].keys())
    f1_grid = np.array([
        [s["metrics"]["per_class_f1"][cls] for cls in class_names]
        for s in serials
    ])
    best_serial_no = manifest["best_serials"][0]
    return class_names, f1_grid, serials, best_serial_no


def main():
    best_classes, best_grid, best_serials, best_sn = load_grid(BEST_RUN)
    base_classes, base_grid, base_serials, base_sn = load_grid(BASELINE_RUN)
    assert best_classes == base_classes, "Class lists differ between runs"
    class_names = best_classes

    # Both bars are the manifest-declared best serial of their run.
    best_row_full = next(s for s in best_serials if s["serial_no"] == best_sn)
    base_row_full = next(s for s in base_serials if s["serial_no"] == base_sn)
    best_row = np.array([best_row_full["metrics"]["per_class_f1"][cls] for cls in class_names])
    base_row = np.array([base_row_full["metrics"]["per_class_f1"][cls] for cls in class_names])
    best_macro = float(best_row_full["metrics"]["macro_f1"])
    base_macro = float(base_row_full["metrics"]["macro_f1"])

    # Error bars span min-max across each run's serial cell, framed against the
    # best-serial bar so seed spread is still visible. Clip negatives
    # (best-by-macro serial may not be the per-class top in every column).
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

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(
        x - bar_width / 2, base_row, bar_width,
        yerr=base_yerr, capsize=3, color=COLOUR_BASELINE,
        label=f"{BASELINE_LABEL} ({BASELINE_RUN} S{base_sn}; macro {base_macro:.3f})",
    )
    ax.bar(
        x + bar_width / 2, best_row, bar_width,
        yerr=best_yerr, capsize=3, color=COLOUR_BEST,
        label=f"{BEST_LABEL} ({BEST_RUN} S{best_sn}; macro {best_macro:.3f})",
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
    ax.set_ylim(0.2, 1)
    fig.suptitle(
        "Per-class F1 on custom 14 class taxonomy testing fine-grained confusion. "
        "Best serial each. Error bars: inter-serial range."
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
