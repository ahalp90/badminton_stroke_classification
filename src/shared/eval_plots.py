"""Plotting helpers for classification eval.

Adapted from src/bst_x/result_utils.py and
scratch/presentation_prep/confusion_matrix.py (Ari's presentation-polish
version). Produces a dual-panel precision- and recall-normalised
confusion matrix with classes ordered ascending by per-class F1 so the
worst-performing pairs cluster at the bottom-left.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix, f1_score

DEFAULT_FONT_SIZE = 9
CONFUSION_FIG_SIZE = (20, 9)
SAVE_DPI = 160


def annotate_cells(ax: Axes, matrix: np.ndarray, font_size: int) -> None:
    """Write each cell value in white on dark / black on light."""
    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f'{matrix[i, j]:.2f}',
                ha='center', va='center',
                color='white' if matrix[i, j] > threshold else 'black',
                fontsize=font_size,
            )


def render_panel(
    fig: Figure,
    ax: Axes,
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    subtitle: str,
    primary_axis: str,
    font_size: int = DEFAULT_FONT_SIZE,
) -> None:
    """Heatmap a normalised matrix onto one axis with class-name ticks.

    :param primary_axis: 'x' or 'y'; the axis whose label is bolded to flag the
        direction along which the matrix's values form a proper distribution.
    """
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=font_size,
                       fontweight='bold' if primary_axis == 'x' else 'normal')
    ax.set_yticklabels(class_names, fontsize=font_size,
                       fontweight='bold' if primary_axis == 'y' else 'normal')
    ax.set_xlabel('predicted',
                  fontweight='bold' if primary_axis == 'x' else 'normal')
    ax.set_ylabel('ground truth',
                  fontweight='bold' if primary_axis == 'y' else 'normal')
    ax.set_title(title, pad=24)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontstyle='italic', fontsize=font_size)
    annotate_cells(ax, matrix, font_size)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    need_pre_argmax: bool = False,
    font_size: int = DEFAULT_FONT_SIZE,
    save_name: str | None = None,
    save: bool = True,
) -> None:
    """Plot precision- and recall-normalised confusion matrices side by side.

    Classes are reordered ascending by per-class F1 so the worst-performing
    pairs sit at the bottom-left where confusions tend to cluster.

    :param y_true: shape (N,) class indices, or (N, C) one-hot if need_pre_argmax.
    :param y_pred: shape (N,) class indices, or (N, C) logits/probs if need_pre_argmax.
    :param class_names: length C; label for each class index.
    :param need_pre_argmax: True if y_true / y_pred are (N, C) and need argmax.
    :param save_name: output path stem; final file is ``<save_name>_confusion_matrix.jpg``.
    """
    if need_pre_argmax:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    n_classes = len(class_names)

    # Per-class F1 → ascending sort. Worst classes end up bottom-left.
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes))
    order = np.argsort(per_class_f1)
    sorted_names = [class_names[i] for i in order]

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cm_sorted = cm[np.ix_(order, order)].astype(np.float32)

    # Precision: cols sum to 1 (P(true=i | predicted=j)).
    col_sums = cm_sorted.sum(axis=0, keepdims=True)
    precision_m = np.divide(cm_sorted, col_sums,
                            out=np.zeros_like(cm_sorted), where=col_sums > 0)
    # Recall: rows sum to 1 (P(predicted=j | true=i)).
    row_sums = cm_sorted.sum(axis=1, keepdims=True)
    recall_m = np.divide(cm_sorted, row_sums,
                         out=np.zeros_like(cm_sorted), where=row_sums > 0)

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=CONFUSION_FIG_SIZE)
    fig.suptitle(
        f'Confusion matrix: {model_name} '
        f'(n={len(y_true)}; classes ordered ascending by per-class F1)',
        fontsize=12, y=0.99,
    )
    render_panel(fig, ax_p, precision_m, sorted_names,
                 'precision-normalised (cols sum to 1)',
                 'by prediction[col], truths were these:',
                 primary_axis='x', font_size=font_size)
    render_panel(fig, ax_r, recall_m, sorted_names,
                 'recall-normalised (rows sum to 1)',
                 'by truth[row], predictions were these:',
                 primary_axis='y', font_size=font_size)

    # rect reserves top strip for suptitle; without it, tight_layout overlaps suptitle
    # with the panel titles + italic subtitles.
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if save_name is None:
        save_name = model_name
    if save:
        plt.savefig(f'{save_name}_confusion_matrix.jpg', dpi=SAVE_DPI, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
