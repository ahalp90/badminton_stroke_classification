"""Per-cell arc figures + cross-cell summary for the alpha over-allocation read.

Run with badminton-cicd (matplotlib):
    /home/ariel/.venvs/badminton-cicd/bin/python plot_arcs.py

Per cell, one PNG with three panels:
  top    val macro-F1 (mean +- serial spread) + min-F1, cosine LR on a twin axis,
         vertical line at the macro-plateau epoch;
  lower-left   per-class VAL F1 arcs, the 6 highest-alpha classes coloured, rest grey;
  lower-right  per-class ALPHA arcs, same colour map, alpha=1 renorm-mean line.
The same colour map across both lower panels lets a class be tracked val<->alpha:
the coloured classes sit flat/low on val while their alpha climbs.

Plus two cross-cell figures: the val-F1-max vs final-alpha scatter grid, and all
six macro arcs on one axis.

Palette: protan light-mode variants (white background); macro near-black, LR navy,
saturated/easy classes light grey.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CELL_ORDER = [
    'shuttleset_18_v2', 'bst_24_v2', 'bst_12_v2',
    'bst_25_baseline', 'bst_24_baseline', 'une_v1_14_v2',
]
EPOCHS = np.arange(1, 81)

# Protan light-mode accents (deep enough to hold on white), 6 qualitative hues.
ACCENTS = ['#7c3aed', '#c2410c', '#0e7490', '#db2777', '#a16207', '#15803d']
MACRO = '#1f2937'       # near-black
LRCOL = '#1e40af'       # navy
GREY = '#b8c0cc'        # background (saturated/easy) classes
PLAT_C = '#c2410c'      # plateaued (warm = attention)
IMPR_C = '#0e7490'      # still improving (cool)
PLAT_LINE = '#9aa3af'


def highlight_map(summ: dict, k: int = 6) -> dict[str, str]:
    """Top-k classes by final alpha -> accent colour; others fall through to grey."""
    pc = summ['per_class']
    top = sorted(summ['classes'], key=lambda c: -pc[c]['a_fin'])[:k]
    return {c: ACCENTS[i] for i, c in enumerate(top)}


def plot_cell(label: str, summ: dict) -> None:
    pc = summ['per_class']
    cmap = highlight_map(summ)
    plateau = summ['plateau_ep']
    lr = np.array([summ['lr_arc'][e] for e in EPOCHS])

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.28, wspace=0.18)
    ax_top = fig.add_subplot(gs[0, :])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_a = fig.add_subplot(gs[1, 1])

    # --- top: macro / min + LR -------------------------------------------
    macro, std, mn = summ['macro_mean'], summ['macro_std'], summ['min_mean']
    ax_top.plot(EPOCHS, macro, color=MACRO, lw=2.4, label='val macro-F1', zorder=5)
    ax_top.fill_between(EPOCHS, macro - std, macro + std, color=MACRO, alpha=0.12, zorder=1)
    ax_top.plot(EPOCHS, mn, color=PLAT_C, lw=1.6, ls='-', label='val min-F1', zorder=4)
    ax_top.axvline(plateau, color=PLAT_LINE, ls='--', lw=1.2, zorder=2)
    ax_top.text(plateau + 0.6, 0.04, f'macro plateau ~e{plateau}',
                color=PLAT_LINE, fontsize=9, transform=ax_top.get_xaxis_transform())
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel('F1 (val)')
    ax_top.set_title(
        f"{label}   ({summ['taxonomy']} / {summ['split']})   "
        f"macro max {summ['run_max']:.3f}   "
        f"best-macro epochs {sorted(summ['best_macro_eps'])}",
        fontsize=11)
    ax_lr = ax_top.twinx()
    ax_lr.plot(EPOCHS, lr, color=LRCOL, lw=1.3, ls=':', label='LR (cosine, reconstructed)')
    ax_lr.set_ylabel('learning rate', color=LRCOL)
    ax_lr.tick_params(axis='y', labelcolor=LRCOL)
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_lr.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=8, framealpha=0.9)

    # --- lower-left: per-class val F1 ------------------------------------
    for c in summ['classes']:
        v = pc[c]['v_mean']
        if c in cmap:
            ax_v.plot(EPOCHS, v, color=cmap[c], lw=2.0, zorder=4,
                      label=f"{c}  (a={pc[c]['a_fin']:.2f}, n={pc[c]['n_train']})")
        else:
            ax_v.plot(EPOCHS, v, color=GREY, lw=0.8, alpha=0.55, zorder=1)
    ax_v.plot(EPOCHS, macro, color=MACRO, lw=2.2, ls=(0, (4, 2)), zorder=5, label='macro')
    ax_v.axvline(plateau, color=PLAT_LINE, ls='--', lw=1.0)
    ax_v.set_ylim(0, 1)
    ax_v.set_xlabel('epoch')
    ax_v.set_ylabel('per-class val F1')
    ax_v.set_title('val F1 arcs (top-6 alpha classes coloured; rest grey)', fontsize=10)
    ax_v.legend(loc='lower right', fontsize=7.5, framealpha=0.9)

    # --- lower-right: per-class alpha ------------------------------------
    for c in summ['classes']:
        a = pc[c]['a_mean']
        if c in cmap:
            ax_a.plot(EPOCHS, a, color=cmap[c], lw=2.0, zorder=4)
        else:
            ax_a.plot(EPOCHS, a, color=GREY, lw=0.8, alpha=0.55, zorder=1)
    ax_a.axhline(1.0, color=MACRO, ls=(0, (4, 2)), lw=1.2, zorder=3)
    ax_a.text(81, 1.0, 'renorm mean', color=MACRO, fontsize=8, va='center')
    ax_a.axvline(plateau, color=PLAT_LINE, ls='--', lw=1.0)
    ax_a.set_xlabel('epoch')
    ax_a.set_ylabel('per-class alpha')
    ax_a.set_title('alpha arcs (same colour map): budget climbs on the coloured classes',
                   fontsize=10)

    out = HERE / f'arc_{label}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')


def plot_scatter_grid(summaries: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, label in zip(axes.ravel(), CELL_ORDER):
        s = summaries[label]
        pc = s['per_class']
        for c in s['classes']:
            d = pc[c]
            if np.isnan(d['v_max']):
                continue
            colour = PLAT_C if d['plateaued'] else IMPR_C
            size = 18 + d['n_train'] / 12
            ax.scatter(d['v_max'], d['a_fin'], s=size, color=colour,
                       alpha=0.8, edgecolor='white', linewidth=0.5, zorder=3)
            if d['a_fin'] > 1.45 or d['v_max'] < 0.30:  # label the extremes
                ax.annotate(c.replace('_', ' '), (d['v_max'], d['a_fin']),
                            fontsize=6.5, xytext=(3, 2), textcoords='offset points')
        ax.axhline(1.0, color=MACRO, ls=(0, (4, 2)), lw=1.0)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel('val F1 max')
        ax.set_ylabel('final alpha')
        ax.set_title(f"{label}\nmacro {s['run_max']:.3f}, "
                     f"{100 * s['frac_abovemean_plateaued']:.0f}% of over-weight on plateaued",
                     fontsize=9.5)
    # Top band, stacked so nothing overlaps: suptitle, then a grey method line
    # explaining the plateau test, then the colour legend below both.
    fig.suptitle('Final alpha vs best val F1, per class (point size ~ n_train). '
                 'Alpha rises as val falls; the high-alpha points are mostly already plateaued.',
                 fontsize=12, y=0.985)
    fig.text(0.5, 0.945,
             'Plateaued (orange) = smoothed (5-epoch) val F1 gained ≤ 0.02 after the cell’s '
             'macro-plateau epoch; macro-plateau epoch = first epoch within 0.01 of the run-max macro-F1.',
             ha='center', fontsize=9, color='#4b5563')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAT_C, markersize=9,
                      label='plateaued by macro-plateau epoch'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor=IMPR_C, markersize=9,
                      label='still improving after')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.915),
               ncol=2, fontsize=10, framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.875])
    out = HERE / 'alpha_vs_valf1_grid.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')


def plot_macro_all(summaries: dict) -> None:
    # Hand-assigned so lines close in height also sit in different colour
    # families (the two bst_24 splits bunch at ~0.83-0.845: keep them
    # cool/warm apart, since orange+ochre converge under protan shift).
    colours = {
        'bst_12_v2': '#0e7490',        # teal  (top, 0.847)
        'bst_24_v2': '#c2410c',        # orange(0.845)
        'bst_24_baseline': '#7c3aed',  # violet(0.829) - cool, away from orange
        'bst_25_baseline': '#db2777',  # pink  (0.820)
        'une_v1_14_v2': '#15803d',     # green (0.768)
        'shuttleset_18_v2': '#a16207', # ochre (0.692, far below; any hue safe)
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label in CELL_ORDER:
        s = summaries[label]
        ax.plot(EPOCHS, s['macro_mean'], color=colours[label], lw=2.0,
                label=f"{label}  (max {s['run_max']:.3f}, plateau ~e{s['plateau_ep']})")
        ax.axvline(s['plateau_ep'], color=colours[label], ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('val macro-F1 (serial-mean)')
    ax.set_ylim(0, 0.9)
    ax.set_title('Val macro-F1 across the six cells: all plateau by ~epoch 26-31, '
                 'then crawl for 50 more epochs', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    out = HERE / 'macro_arcs_all_cells.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')


def main() -> None:
    summaries = pickle.loads((HERE / 'summaries.pkl').read_bytes())
    for label in CELL_ORDER:
        plot_cell(label, summaries[label])
    plot_scatter_grid(summaries)
    plot_macro_all(summaries)


if __name__ == '__main__':
    main()
