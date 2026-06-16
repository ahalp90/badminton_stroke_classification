"""Class train-sample size vs val F1 (14-class taxonomy, split_v2).

Pulls per-class val F1 from the run_20260602_143618_156220 serial-2 prediction
dump (best serial of that run by test macro and min F1) and class train counts
from clips_master.csv with the une_merge_v1 4-into-2 merges applied.
Single-serial val F1 here; the manifest only records test per-class F1 across
serials, and the npz dumps are the only source for the val numbers.

Output: local_scratch/presentation_prep/class_size_vs_val_f1.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path('/home/ariel/Documents/COSC594/badminton_stroke_classification')
RUN_DIR = REPO / 'experiments/bst_x/shuttleset/run_20260602_143618_156220'
SERIAL = 2
CLIPS_CSV = REPO / 'notebooks/clips_master.csv'
OUT_PNG = REPO / 'local_scratch/presentation_prep/class_size_vs_val_f1.png'

# une_merge_v1: 4 raw types fold into existing labels; rest pass through and the
# 14-class taxonomy keeps the result set below.
MERGE_MAP = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}

# Protan-safe: navy circles for strokes, orange diamonds for services (the
# rally-start outliers), neutral grey for grid + the regression hint.
NAVY = '#1e40af'
ORANGE = '#e88806'
GREY = '#6b7280'


def per_class_f1_from_npz(npz_path: Path) -> tuple[dict[str, float], int]:
    """Compute val per-class F1 from the saved logits / argmax dump.

    :param npz_path: path to a `val_serial_*.npz` written by the train loop.
    :return: (class -> F1) dict, and total stroke count.
    """
    data = np.load(npz_path, allow_pickle=True)
    y_true = data['y_true']                              # (n,) int class index
    y_pred = data['y_pred_top1']                         # (n,) int class index
    class_list = [str(c) for c in data['class_list']]    # length-K

    f1_by_class = {}
    for c, name in enumerate(class_list):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_by_class[name] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1_by_class, int(y_true.shape[0])


def train_counts_for_classes(class_list: list[str]) -> dict[str, int]:
    """Train-split clip counts per class, with une_merge_v1 applied."""
    df = pd.read_csv(CLIPS_CSV, usecols=['raw_type_en', 'split_v2'])
    df['class'] = df['raw_type_en'].map(lambda t: MERGE_MAP.get(t, t))
    train = df[(df['split_v2'] == 'train') & (df['class'].isin(class_list))]
    counts = train.groupby('class').size().reindex(class_list, fill_value=0)
    return {cls: int(counts[cls]) for cls in class_list}


def main() -> None:
    npz_path = RUN_DIR / 'predictions' / f'val_serial_{SERIAL}.npz'
    f1_by_class, n_val = per_class_f1_from_npz(npz_path)
    classes = list(f1_by_class.keys())
    counts = train_counts_for_classes(classes)

    x = np.array([counts[c] for c in classes], dtype=float)
    y = np.array([f1_by_class[c] for c in classes], dtype=float)
    is_service = np.array(['service' in c for c in classes])

    # Spearman = Pearson correlation computed on the ranks of X and Y.
    rho_all, p_all = spearmanr(x, y)

    macro = float(np.mean(y))
    min_f1 = float(np.min(y))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xscale('log')

    # Non-service strokes
    ax.scatter(x[~is_service], y[~is_service], s=85, color=NAVY, edgecolor='black',
               linewidth=0.6, zorder=3, label='in-rally strokes (12)')
    # Services as diamonds in orange — visually called out as outliers
    ax.scatter(x[is_service], y[is_service], s=110, color=ORANGE, marker='D',
               edgecolor='black', linewidth=0.6, zorder=3,
               label='services (2, rally start)')

    # Label each point. Offset right except where it would clash with a neighbour
    # or the upper-left Spearman annotation box.
    label_offsets = {
        'net_shot':              (-8, -12),
        'return_net':            (8,  4),
        'smash':                 (8,  4),
        'wrist_smash':           (8,  4),
        'lob':                   (8,  -10),
        'clear':                 (8,  4),
        'drive':                 (8,  4),
        'drop':                  (8,  -10),
        'passive_drop':          (8,  4),
        'push':                  (-8, -12),
        'rush':                  (8,  4),
        'cross_court_net_shot':  (-12, 8),
        'short_service':         (-8, -12),
        'long_service':          (8,  -12),
    }
    for cls, xi, yi in zip(classes, x, y):
        dx, dy = label_offsets.get(cls, (8, 4))
        ha = 'right' if dx < 0 else 'left'
        ax.annotate(cls, (xi, yi), xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, ha=ha, va='center', color='#222')

    ax.set_xlabel('train clips (log scale)')
    ax.set_ylabel('val F1')
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3, which='both')
    ax.set_axisbelow(True)

    min_cls = min(f1_by_class, key=f1_by_class.get)
    ax.set_title(
        'Class train-sample size vs val F1 — 14-class taxonomy, split_v2\n'
        f'run_20260602_143618_156220 serial {SERIAL} (best by test macro+min)  ·  '
        f'n_val={n_val:,}  ·  macro F1 {macro:.3f}  ·  min F1 {min_f1:.3f} ({min_cls})',
        fontsize=10,
    )

    # Rank correlation annotation. Lower-left is the one quadrant with no
    # points (all the small-count classes sit above 0.6).
    txt = (f'X vs Y rank correlation (Pearson): {rho_all:+.2f}  '
           f'({p_all*100:.0f}% noise p)')
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=9, color=GREY,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=GREY, alpha=0.9))

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
