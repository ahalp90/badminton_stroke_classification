"""Plotting helpers for classification eval.

Adapted from src/bst_refactor/stroke_classification/result_utils.py
(plot_confusion_matrix + set_one_ax_confusion_matrix). Adaptation: tick
labels accept real class names instead of numeric indices.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix

CELL_TEXT_MAX_CLASSES = 18
DEFAULT_FONT_SIZE = 12
CONFUSION_FIG_SIZE = (15, 7)


def set_one_ax_confusion_matrix(
    fig: Figure,
    ax: Axes,
    matrix: np.ndarray,
    class_names: list[str],
    normalized: bool = True,
    font_size: int = DEFAULT_FONT_SIZE,
) -> None:
    ticks = np.arange(len(matrix))
    ax_img = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(ax_img, ax=ax)
    ax.set_xticks(ticks, class_names, fontsize=font_size, rotation=45, ha='right')
    ax.set_yticks(ticks, class_names, fontsize=font_size)

    if len(matrix) < CELL_TEXT_MAX_CLASSES:
        fmt = '.2f' if normalized else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, format(matrix[i, j], fmt),
                    verticalalignment='center',
                    horizontalalignment='center',
                    color='white' if matrix[i, j] > thresh else 'black',
                    fontsize=font_size,
                )
    else:
        for i in ticks[:-1]:
            mid_point = (ticks[i] + ticks[i + 1]) / 2
            ax.axvline(x=mid_point, color='black', linestyle='-')
            ax.axhline(y=mid_point, color='black', linestyle='-')

    ax.set_xlabel('Prediction', fontsize=font_size)
    ax.set_ylabel('Ground Truth', fontsize=font_size)


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

    :param y_true: shape (N,) class indices, or (N, C) one-hot if need_pre_argmax.
    :param y_pred: shape (N,) class indices, or (N, C) logits/probs if need_pre_argmax.
    :param class_names: length C; label for each class index.
    :param need_pre_argmax: True if y_true / y_pred are (N, C) and need argmax.
    :param save_name: output path stem; final file is ``<save_name>_confusion_matrix.jpg``.
    """
    if need_pre_argmax:
        matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    else:
        matrix = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=CONFUSION_FIG_SIZE)
    fig.suptitle(f'{model_name} Result On Testing Set')
    ax1, ax2 = fig.subplots(1, 2)

    precision_m = matrix.astype(np.float32) / matrix.sum(axis=0)
    ax1.set_title('Confusion Matrix (Precision)')
    set_one_ax_confusion_matrix(fig, ax1, precision_m, class_names,
                                 normalized=True, font_size=font_size)

    recall_m = matrix.astype(np.float32) / matrix.sum(axis=1, keepdims=True)
    ax2.set_title('Confusion Matrix (Recall)')
    set_one_ax_confusion_matrix(fig, ax2, recall_m, class_names,
                                 normalized=True, font_size=font_size)

    if save_name is None:
        save_name = model_name
    if save:
        plt.savefig(f'{save_name}_confusion_matrix.jpg', bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
