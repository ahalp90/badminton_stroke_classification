"""Confusion matrix render for the supervisor presentation.

Reads a per-split predictions npz (``<split>_serial_<n>.npz`` produced by
``bst_train`` at end-of-serial, or post-hoc by ``bst_infer --fe``) and renders a
dual-panel confusion matrix: precision-normalised (columns sum to 1) and
recall-normalised (rows sum to 1). Class order is performance-ascending by
per-class F1 so the smash / wrist_smash pair sits at the bottom-left.

The 'Blues' colourmap is a single-hue sequential, universally readable
(protanopia-safe by virtue of being one-hue).

Usage::

    python scratch/presentation_prep/confusion_matrix.py \\
        --predictions src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_<id>/predictions/test_serial_5.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = REPO_ROOT / "scratch/presentation_prep/confusion_matrix.png"

# Mapping from raw run_id to the in-doc / in-chat common name. Used to title the figure
# so the reader sees the ablation by its working name first, run_id second. Extend as
# new runs get rendered; missing entries fall back to the run_id alone.
RUN_LABELS: dict[str, str] = {
    "run_20260505_154907": "aug v1 + p_jit=0.3",
    "run_20260503_172922": "shuttle_zero_fix [wipe_drop]",
    "run_20260430_170325": "first nosides (Phase 2 LS=0.1)",
    "run_20260530_225714_593038": "bst_24 baseline (drop unknown)",
    "run_20260530_210600_435552": "bst_25 baseline (keep unknown)",
}


def annotate_cells(ax, matrix: np.ndarray, font_size: int) -> None:
    """Write the value of each cell in white on dark / black on light.

    :param ax: target axis
    :param matrix: 2-D normalised matrix (values in [0, 1])
    :param font_size: text size for cell annotations
    """
    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=font_size,
            )


def render_panel(fig, ax, matrix: np.ndarray, class_names: list[str],
                 title: str, subtitle: str, primary_axis: str,
                 font_size: int = 9) -> None:
    """Heatmap one normalised matrix onto a single axis with class-name ticks.

    :param fig: figure for the colourbar
    :param ax: target axis
    :param matrix: 2-D normalised matrix
    :param class_names: tick labels in the same order as matrix rows/cols
    :param title: panel title
    :param subtitle: italic reading-hint line drawn just under the title
    :param primary_axis: "x" or "y"; the axis whose label is bolded to flag the
        direction along which the matrix's values form a proper distribution.
    :param font_size: text size for ticks and cell annotations
    """
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=font_size,
                       fontweight="bold" if primary_axis == "x" else "normal")
    ax.set_yticklabels(class_names, fontsize=font_size,
                       fontweight="bold" if primary_axis == "y" else "normal")
    ax.set_xlabel("predicted",
                  fontweight="bold" if primary_axis == "x" else "normal")
    ax.set_ylabel("ground truth",
                  fontweight="bold" if primary_axis == "y" else "normal")
    # pad lifts the title to leave room for the italic subtitle sitting at the axes edge.
    ax.set_title(title, pad=24)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontstyle="italic", fontsize=font_size)
    annotate_cells(ax, matrix, font_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to a predictions/<split>_serial_<n>.npz dumped by "
                             "bst_train (end-of-serial) or bst_infer --fe")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--figsize", type=str, default="20,9",
                        help="W,H in inches; bump for taxonomies with many classes (e.g. 28,13 for ~24)")
    parser.add_argument("--font-size", type=int, default=9,
                        help="tick and cell-annotation font size")
    args = parser.parse_args()
    fig_w, fig_h = (float(v) for v in args.figsize.split(","))

    payload = np.load(args.predictions, allow_pickle=True)
    y_true = payload["y_true"]
    y_pred = payload["y_pred_top1"]  # argmax preds (top-1 of the top-k dump)
    class_names = list(payload["class_list"])
    run_id = str(payload["run_id"])
    serial_no = int(payload["serial_no"])
    n_classes = len(class_names)

    # Per-class F1, then sort ascending. Worst-performing classes end up bottom-left.
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes))
    order = np.argsort(per_class_f1)
    sorted_names = [class_names[i] for i in order]

    # Confusion matrix in original index space, then reindex into sorted order.
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cm_sorted = cm[np.ix_(order, order)].astype(np.float32)

    # Two normalisations of the same matrix.
    # sklearn returns cm[i, j] with i=true class, j=predicted class.
    # Precision: each column sums to 1, cm[i,j]/col_sum[j] = P(true=i | predicted=j).
    # Guard against zero-prediction columns (sklearn returns nan otherwise).
    col_sums = cm_sorted.sum(axis=0, keepdims=True)
    precision_m = np.divide(cm_sorted, col_sums, out=np.zeros_like(cm_sorted), where=col_sums > 0)
    # Recall: each row sums to 1, cm[i,j]/row_sum[i] = P(predicted=j | true=i).
    row_sums = cm_sorted.sum(axis=1, keepdims=True)
    recall_m = np.divide(cm_sorted, row_sums, out=np.zeros_like(cm_sorted), where=row_sums > 0)

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    common_name = RUN_LABELS.get(run_id)
    run_id_block = f"{common_name} ({run_id})" if common_name else run_id
    fig.suptitle(
        f"Confusion matrix: {run_id_block} S{serial_no} "
        f"(n={len(y_true)}; classes ordered ascending by per-class F1)",
        fontsize=12,
        y=0.99,
    )
    render_panel(fig, ax_p, precision_m, sorted_names,
                 "precision-normalised (cols sum to 1)",
                 "by prediction[col], truths were these:",
                 primary_axis="x", font_size=args.font_size)
    render_panel(fig, ax_r, recall_m, sorted_names,
                 "recall-normalised (rows sum to 1)",
                 "by truth[row], predictions were these:",
                 primary_axis="y", font_size=args.font_size)

    # rect reserves top strip for suptitle; without it, tight_layout overlaps suptitle
    # with the panel titles + italic subtitles.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
