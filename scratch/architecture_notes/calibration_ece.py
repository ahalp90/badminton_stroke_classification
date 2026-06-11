"""Top-1 confidence calibration (ECE + reliability diagrams) for dumped prediction npzs.

Reads the per-stroke logit dumps written by ``dump_topk_predictions`` (bst_x_common),
takes the top-1 softmax probability as the served confidence, and reports Expected
and Maximum Calibration Error plus a reliability diagram and confidence histogram for
each split. Also fits a single temperature on val (min NLL) and reports the ECE it
would buy on val and test, so the "serve plain softmax vs temperature-scaled" call
has a number behind it rather than a guess.

Top-1 ECE is the right metric here: the FE shows the argmax class and its confidence,
so what matters is whether that one displayed number matches how often that prediction
is actually right. Temperature scaling never changes the argmax (it divides all logits
by T > 0, which preserves order), so accuracy is identical before and after; only the
confidence number moves.

Run: ~/.venvs/badminton-cicd/bin/python scratch/architecture_notes/calibration_ece.py
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # headless: save PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

EXPERIMENTS = Path(
    '/home/ariel/Documents/COSC594/badminton_stroke_classification/src/bst_x/'
    'stroke_classification/main_on_shuttleset/experiments'
)
OUT_DIR = Path(__file__).resolve().parent / 'calibration'

# Each run kept only its best serial's prediction dump; serial pinned per run.
RUNS = [
    {'label': 'une_v1_14', 'dir': 'run_20260531_005535_005154', 'serial': 3},
    {'label': 'bst_25_baseline', 'dir': 'run_20260530_210600_435552', 'serial': 1},
]

N_BINS = 15  # Guo et al. 2017 convention for ECE reliability bins

# Protan-safe on white: cool navy vs warm orange (well separated under red-channel
# shift), neutral grey for the reference line. No red, no green-beside-orange.
NAVY = '#1e40af'
ORANGE = '#e88806'
GREY = '#6b7280'


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over a ``(n, n_classes)`` logit array (max-shift for stability)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def reliability_stats(
    conf: np.ndarray,
    correct: np.ndarray,
    n_bins: int = N_BINS,
) -> dict:
    """Bin top-1 confidence into ``n_bins`` equal-width bins, return per-bin acc/conf + ECE/MCE.

    :param conf: top-1 softmax probability per clip, shape ``(n,)`` in ``[0, 1]``.
    :param correct: bool correctness of the top-1 prediction per clip, shape ``(n,)``.
    :return: dict with bin centres, per-bin accuracy/confidence/count (NaN for empty
        bins), scalar ``ece`` (count-weighted mean |acc - conf|) and ``mce`` (max gap).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # digitize against the inner edges -> bin index in [0, n_bins-1]; conf == 1.0
    # lands in the last bin, conf == 0.0 in the first.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = idx == b
        bin_count[b] = in_bin.sum()
        if bin_count[b]:
            bin_acc[b] = correct[in_bin].mean()
            bin_conf[b] = conf[in_bin].mean()

    n = conf.shape[0]
    nonempty = bin_count > 0
    gap = np.abs(bin_acc - bin_conf)
    ece = float(np.sum((bin_count[nonempty] / n) * gap[nonempty]))
    mce = float(gap[nonempty].max())
    return {
        'centres': centres,
        'bin_acc': bin_acc,
        'bin_conf': bin_conf,
        'bin_count': bin_count,
        'ece': ece,
        'mce': mce,
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a single scalar temperature on these logits by minimising NLL.

    Optimises over ``log T`` so T stays strictly positive; bounded to a sane range.
    Same idea as Guo et al. 2017, done on whatever split's logits are passed (fit on
    val, then apply the returned T to test).

    :return: fitted temperature T (divide logits by T before softmax).
    """
    rows = np.arange(y_true.shape[0])

    def nll(log_t: float) -> float:
        scaled = logits / np.exp(log_t)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(shifted).sum(axis=1))
        log_p_true = shifted[rows, y_true] - log_partition
        return float(-log_p_true.mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    return float(np.exp(res.x))


def plot_run(label: str, panels: list[dict], out_path: Path) -> None:
    """2x2 figure: reliability (top) + confidence histogram (bottom), val and test columns.

    :param panels: list of per-split dicts (val first, then test) carrying the
        ``reliability_stats`` output plus ``split``, ``n``, ``acc``, ``mean_conf``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, panel in enumerate(panels):
        stats = panel['stats']
        nonempty = stats['bin_count'] > 0
        ax_rel = axes[0, col]
        ax_hist = axes[1, col]

        # --- reliability: accuracy bars vs the per-bin mean confidence ---
        ax_rel.bar(
            stats['centres'], stats['bin_acc'],
            width=0.9 / N_BINS, color=NAVY, label='accuracy', zorder=2,
        )
        ax_rel.plot(
            stats['centres'][nonempty], stats['bin_conf'][nonempty],
            marker='o', ms=4, color=ORANGE, lw=1.5,
            label='mean confidence', zorder=3,
        )
        ax_rel.plot([0, 1], [0, 1], ls='--', color=GREY, lw=1, label='perfect calibration')
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_xlabel('top-1 confidence')
        ax_rel.set_ylabel('accuracy')
        direction = 'over' if panel['mean_conf'] > panel['acc'] else 'under'
        ax_rel.set_title(f"{panel['split']} (n={panel['n']})  {direction}confident")
        ax_rel.text(
            0.03, 0.97,
            f"ECE {stats['ece']:.3f}\nMCE {stats['mce']:.3f}\n"
            f"acc {panel['acc']:.3f}\nconf {panel['mean_conf']:.3f}",
            transform=ax_rel.transAxes, va='top', ha='left', fontsize=9,
            bbox={'boxstyle': 'round', 'fc': 'white', 'ec': GREY, 'alpha': 0.85},
        )
        if col == 0:
            ax_rel.legend(loc='lower right', fontsize=8)

        # --- confidence histogram: where the clips actually sit ---
        frac = stats['bin_count'] / panel['n']
        ax_hist.bar(stats['centres'], frac, width=0.9 / N_BINS, color=NAVY, zorder=2)
        ax_hist.axvline(panel['mean_conf'], color=ORANGE, lw=1.5, label='mean confidence')
        ax_hist.axvline(panel['acc'], color=NAVY, ls='--', lw=1.5, label='accuracy')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel('top-1 confidence')
        ax_hist.set_ylabel('fraction of clips')
        if col == 0:
            ax_hist.legend(loc='upper left', fontsize=8)

    fig.suptitle(f'{label}: top-1 confidence calibration ({N_BINS} bins)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {out_path}')


def load_split(run_dir: str, split: str, serial: int) -> dict:
    """Load one split's npz and compute the calibration payload at T=1."""
    npz_path = EXPERIMENTS / run_dir / 'predictions' / f'{split}_serial_{serial}.npz'
    data = np.load(npz_path, allow_pickle=True)
    logits = data['logits']
    y_true = data['y_true']

    pred = logits.argmax(axis=1)
    # The dump's own top-1 must match a fresh argmax, else the npz is internally
    # inconsistent and every number below is built on sand. Fail loud.
    assert np.array_equal(pred, data['y_pred_top1']), (
        f'{npz_path.name}: argmax(logits) disagrees with stored y_pred_top1'
    )

    probs = softmax(logits)
    conf = probs.max(axis=1)
    correct = pred == y_true
    stats = reliability_stats(conf, correct)
    return {
        'split': split,
        'n': int(y_true.shape[0]),
        'logits': logits,
        'y_true': y_true,
        'y_pred': pred,
        'conf': conf,
        'correct': correct,
        'acc': float(correct.mean()),
        'mean_conf': float(conf.mean()),
        'stats': stats,
        'class_list': [str(c) for c in data['class_list']],
    }


def ece_at_temperature(logits: np.ndarray, correct: np.ndarray, temp: float) -> float:
    """ECE after dividing logits by ``temp`` (argmax/correctness unchanged by T)."""
    conf_t = softmax(logits / temp).max(axis=1)
    return reliability_stats(conf_t, correct)['ece']


MIN_SUPPORT = 30  # below this a per-class gap is too noisy to read into


def per_class_gaps(panel: dict) -> list[dict]:
    """Predicted-class calibration: mean confidence vs precision, per predicted class.

    For every clip the model predicts as class c, the served confidence is its top-1
    softmax P(c) and it is correct iff the true label is c, so the empirical hit rate
    among those clips is precision(c). The per-class gap is ``mean_conf - precision``
    (positive = the model claims more than it delivers for that stroke).

    Gap CI is the proper paired SE of ``conf_i - correct_i`` over the predicted-c clips,
    so the error bar already folds in both the precision noise and the confidence spread;
    a bar whose CI crosses zero is not significantly miscalibrated at that support.

    :return: one dict per class with name, support n, precision, mean conf, gap, ci half-width.
    """
    pred = panel['y_pred']
    conf = panel['conf']
    correct = panel['correct'].astype(float)
    rows = []
    for cls_idx, name in enumerate(panel['class_list']):
        predicted_c = pred == cls_idx
        n = int(predicted_c.sum())
        if n == 0:
            rows.append({'cls': name, 'n': 0, 'prec': np.nan,
                         'conf': np.nan, 'gap': np.nan, 'ci': np.nan})
            continue
        deficit = conf[predicted_c] - correct[predicted_c]  # per-clip conf minus hit
        se = deficit.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            'cls': name,
            'n': n,
            'prec': float(correct[predicted_c].mean()),
            'conf': float(conf[predicted_c].mean()),
            'gap': float(deficit.mean()),
            'ci': float(1.96 * se),
        })
    return rows


def support_weighted_scatter(rows: list[dict]) -> tuple[float, float]:
    """Support-weighted mean and std of per-class gaps over well-supported classes.

    The weighted mean ~ the global gap (what one temperature can remove); the std is the
    per-class spread a single temperature leaves behind. Std comparable to or above the
    mean means scatter dominates, so a global T can't do much.
    """
    solid = [r for r in rows if r['n'] >= MIN_SUPPORT]
    weights = np.array([r['n'] for r in solid], dtype=float)
    gaps = np.array([r['gap'] for r in solid])
    mean = float(np.average(gaps, weights=weights))
    std = float(np.sqrt(np.average((gaps - mean) ** 2, weights=weights)))
    return mean, std


def plot_per_class(label: str, val: dict, test: dict, out_path: Path) -> None:
    """Per-class gap bars (val | test), sorted by test gap, CIs as error bars."""
    val_rows = per_class_gaps(val)
    test_rows = per_class_gaps(test)
    n_cls = len(val_rows)
    # Sort by test gap, most overconfident on top; no-prediction classes sink to bottom.
    order = sorted(
        range(n_cls),
        key=lambda i: test_rows[i]['gap'] if test_rows[i]['n'] else -np.inf,
        reverse=True,
    )
    names = [val_rows[i]['cls'] for i in order]
    ys = np.arange(n_cls)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 0.42 * n_cls + 1.5), sharey=True,
    )
    for ax, rows, split in [(axes[0], val_rows, 'val'), (axes[1], test_rows, 'test')]:
        gaps = np.array([rows[i]['gap'] if rows[i]['n'] else 0.0 for i in order])
        cis = np.array([rows[i]['ci'] if rows[i]['n'] > 1 else 0.0 for i in order])
        ns = [rows[i]['n'] for i in order]
        colours = [ORANGE if g > 0 else NAVY for g in gaps]
        ax.barh(ys, gaps, color=colours, zorder=2)
        ax.errorbar(gaps, ys, xerr=cis, fmt='none', ecolor=GREY,
                    elinewidth=1, capsize=2, zorder=3)
        ax.axvline(0, color=GREY, lw=1)
        for y, gap, n in zip(ys, gaps, ns):
            off = 0.005 if gap >= 0 else -0.005
            ax.text(gap + off, y, f'n={n}', va='center',
                    ha='left' if gap >= 0 else 'right', fontsize=7, color=GREY)
        ax.set_title(split)
        ax.set_xlabel('confidence - precision   (+ overconfident)')
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f'{label}: per predicted-class calibration gap', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {out_path}')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        val = load_split(run['dir'], 'val', run['serial'])
        test = load_split(run['dir'], 'test', run['serial'])

        # Fit T on val, apply the same T to test (honest fit-on-held-out protocol).
        temp = fit_temperature(val['logits'], val['y_true'])
        val_ece_t = ece_at_temperature(val['logits'], val['correct'], temp)
        test_ece_t = ece_at_temperature(test['logits'], test['correct'], temp)

        print(f"\n{run['label']}  (serial {run['serial']})")
        print(f"  {'split':5s} {'n':>5s}  {'acc':>6s} {'conf':>6s}  "
              f"{'ECE':>6s} {'MCE':>6s} | T*={temp:.3f}  {'ECE@T':>6s}")
        for panel, ece_t in [(val, val_ece_t), (test, test_ece_t)]:
            s = panel['stats']
            print(f"  {panel['split']:5s} {panel['n']:5d}  "
                  f"{panel['acc']:.3f}  {panel['mean_conf']:.3f}  "
                  f"{s['ece']:.4f} {s['mce']:.4f} |        {ece_t:.4f}")
        gap_dir = 'over' if val['mean_conf'] > val['acc'] else 'under'
        print(f"  (T* fit on val by NLL, applied to test as-is; model is {gap_dir}confident on val)")

        plot_run(run['label'], [val, test], OUT_DIR / f"{run['label']}_calibration.png")

        # Per-class: gap = mean confidence - precision, sorted by test gap.
        val_rows = per_class_gaps(val)
        test_rows = per_class_gaps(test)
        val_by_cls = {r['cls']: r for r in val_rows}
        order = sorted(
            range(len(test_rows)),
            key=lambda i: test_rows[i]['gap'] if test_rows[i]['n'] else -np.inf,
            reverse=True,
        )
        print(f"  per-class (predicted-class calibration; gap = conf - precision, + = overconfident)")
        print(f"    {'class':22s} {'n_v':>4s} {'gap_v':>7s} | "
              f"{'n_t':>4s} {'prec_t':>6s} {'conf_t':>6s} {'gap_t':>7s} {'ci_t':>6s}")
        for i in order:
            t = test_rows[i]
            v = val_by_cls[t['cls']]
            v_gap = f"{v['gap']:+.3f}" if v['n'] else '   - '
            if t['n']:
                print(f"    {t['cls']:22s} {v['n']:4d} {v_gap:>7s} | "
                      f"{t['n']:4d} {t['prec']:.3f}  {t['conf']:.3f}  "
                      f"{t['gap']:+.3f} {t['ci']:.3f}")
            else:
                print(f"    {t['cls']:22s} {v['n']:4d} {v_gap:>7s} |    0      -       -        -      -")
        mean_t, std_t = support_weighted_scatter(test_rows)
        print(f"    test, classes with n>={MIN_SUPPORT}: support-weighted gap {mean_t:+.3f}, "
              f"per-class scatter (std) {std_t:.3f}")

        plot_per_class(run['label'], val, test, OUT_DIR / f"{run['label']}_per_class.png")


if __name__ == '__main__':
    main()
