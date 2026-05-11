"""Confusion matrix render for the 2026-05-11 supervisor presentation.

Reads the prediction dump produced by ``eval_dump_predictions.py`` and renders a
dual-panel confusion matrix: precision-normalised (columns sum to 1) and
recall-normalised (rows sum to 1). Class order is performance-ascending by
per-class F1 so the smash / wrist_smash pair sits at the bottom-left.

The 'Blues' colourmap is a single-hue sequential, universally readable
(protanopia-safe by virtue of being one-hue).

Usage::

    python scratch/presentation_prep/confusion_matrix.py \\
        --predictions src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/predictions/serial_5.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = REPO_ROOT / "scratch/presentation_prep/confusion_matrix.png"


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
                 title: str, font_size: int = 9) -> None:
    """Heatmap one normalised matrix onto a single axis with class-name ticks.

    :param fig: figure for the colourbar
    :param ax: target axis
    :param matrix: 2-D normalised matrix
    :param class_names: tick labels in the same order as matrix rows/cols
    :param title: panel title
    :param font_size: text size for ticks and cell annotations
    """
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=font_size)
    ax.set_yticklabels(class_names, fontsize=font_size)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    annotate_cells(ax, matrix, font_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to predictions/serial_<n>.pt dumped by eval_dump_predictions.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    payload = torch.load(args.predictions, map_location="cpu", weights_only=False)
    y_true = payload["y_true"].numpy()
    y_pred = payload["y_pred"].numpy()
    class_names = payload["active_class_list"]
    run_id = payload["run_id"]
    serial_no = payload["serial_no"]
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

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(
        f"Confusion matrix: {run_id} S{serial_no} "
        f"(n={len(y_true)}; classes ordered ascending by per-class F1)",
        fontsize=12,
    )
    render_panel(fig, ax_p, precision_m, sorted_names, "precision-normalised (cols sum to 1)")
    render_panel(fig, ax_r, recall_m, sorted_names, "recall-normalised (rows sum to 1)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
