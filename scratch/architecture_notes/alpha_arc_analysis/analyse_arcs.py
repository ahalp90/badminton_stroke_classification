"""Per-class arc summaries + over-allocation diagnostics for all six cells.

Run with badminton-cicd (numpy):
    /home/ariel/.venvs/badminton-cicd/bin/python analyse_arcs.py

Consumes arcs.pkl (from parse_arcs.py). For each cell, aggregates the per-class
val-F1 / train-F1 / alpha arcs across the five serials, then asks the handover's
Q4 question per taxonomy: does the renorm-to-mean-1 alpha pour its back-half
budget onto classes that have already stopped improving on val?

Saves summaries.pkl (per-cell + per-class, serial-mean) for the plotter and
writes per-cell markdown tables to tables.md.
"""
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_EPOCHS = 80
CELL_ORDER = [
    'shuttleset_18_v2', 'bst_24_v2', 'bst_12_v2',
    'bst_25_baseline', 'bst_24_baseline', 'une_v1_14_v2',
]


def densify(arc: dict[int, float]) -> np.ndarray:
    """{epoch: value} -> length-80 array indexed by epoch-1, nan where absent."""
    out = np.full(N_EPOCHS, np.nan)
    for ep, v in arc.items():
        if 1 <= ep <= N_EPOCHS:
            out[ep - 1] = v
    return out


def smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Centred moving average ignoring nans; returns same length."""
    out = np.full_like(arr, np.nan)
    half = w // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        window = arr[lo:hi]
        good = window[~np.isnan(window)]
        if good.size:
            out[i] = good.mean()
    return out


def serial_mean(arrs: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean across serials, nan-aware (epoch present in >=1 serial)."""
    stack = np.vstack(arrs)
    with np.errstate(invalid='ignore'):
        return np.nanmean(stack, axis=0)


def last_valid(arr: np.ndarray):
    good = np.where(~np.isnan(arr))[0]
    return arr[good[-1]] if good.size else np.nan


def argmax_valid(arr: np.ndarray):
    """(max_value, epoch) ignoring nans; (nan, nan) if all nan."""
    if np.all(np.isnan(arr)):
        return np.nan, np.nan
    ep = int(np.nanargmax(arr))
    return arr[ep], ep + 1


def analyse_cell(cell: dict) -> dict:
    classes = cell['classes']
    serials = cell['serials']
    sids = sorted(serials)

    # Macro: serial-mean arc + spread.
    macro_arrs = [densify(serials[s]['macro_val']) for s in sids]
    macro_mean = serial_mean(macro_arrs)
    macro_std = np.nanstd(np.vstack(macro_arrs), axis=0)
    min_mean = serial_mean([densify(serials[s]['min_val']) for s in sids])
    run_max = np.nanmax(macro_mean)
    plateau_ep = int(np.where(macro_mean >= run_max - 0.01)[0][0]) + 1
    best_macro_eps = [serials[s]['best_macro_epoch'] for s in sids]

    per_class = {}
    for c in classes:
        # serial-mean arcs
        v_mean = serial_mean([densify(serials[s]['f1_val'][c]) for s in sids])
        t_mean = serial_mean([densify(serials[s]['f1_train'][c]) for s in sids])
        a_mean = serial_mean([densify(serials[s]['alpha'][c]) for s in sids])

        v_max, v_max_ep = argmax_valid(v_mean)
        v_max_s, v_max_ep_s = argmax_valid(smooth(v_mean))  # smoothed peak for plateau logic
        t_max, _ = argmax_valid(t_mean)
        a_peak, a_peak_ep = argmax_valid(a_mean)

        v_fin, t_fin, a_fin = last_valid(v_mean), last_valid(t_mean), last_valid(a_mean)
        gap_max = t_max - v_max if not np.isnan(v_max) else np.nan

        per_class[c] = {
            'n_train': cell['n_train'][c], 'n_val': cell['n_val'][c],
            'v_mean': v_mean, 't_mean': t_mean, 'a_mean': a_mean,
            'v_fin': v_fin, 'v_max': v_max, 'v_max_ep': v_max_ep,
            'v_max_ep_s': v_max_ep_s,  # smoothed peak epoch
            't_fin': t_fin, 't_max': t_max,
            'a_fin': a_fin, 'a_peak': a_peak, 'a_peak_ep': a_peak_ep,
            'gap_max': gap_max,
        }

    # ---- over-allocation diagnostics --------------------------------------
    # Robust "plateaued" test: did the class's SMOOTHED val F1 still gain
    # materially after the macro plateau epoch? (argmax-of-val is unstable on
    # flat noisy arcs, so test the back-half gain, not the peak epoch.)
    # plateaued = gained <= 0.02 between the macro-plateau epoch and the end.
    for c in classes:
        sv = smooth(per_class[c]['v_mean'])
        v_at_plateau = sv[plateau_ep - 1]
        tail = sv[plateau_ep - 1:]
        gain = (np.nanmax(tail) - v_at_plateau) if not np.all(np.isnan(tail)) else np.nan
        per_class[c]['backhalf_gain'] = gain
        per_class[c]['plateaued'] = (not np.isnan(gain)) and gain <= 0.02

    valid = [c for c in classes if per_class[c]['n_val'] > 0]
    a_fin = np.array([per_class[c]['a_fin'] for c in valid])
    v_max = np.array([per_class[c]['v_max'] for c in valid])
    corr = float(np.corrcoef(a_fin, v_max)[0, 1]) if len(valid) > 2 else np.nan

    total_budget = float(sum(per_class[c]['a_fin'] for c in classes))  # ~= n_classes

    def above(c):  # alpha budget above the renorm mean of 1.0
        return max(0.0, per_class[c]['a_fin'] - 1.0)

    abovemean_total = float(sum(above(c) for c in classes))
    plateaued = [c for c in classes if per_class[c]['plateaued']]
    abovemean_plateaued = float(sum(above(c) for c in plateaued))
    floor = [c for c in classes if not np.isnan(per_class[c]['v_max']) and per_class[c]['v_max'] < 0.20]
    abovemean_floor = float(sum(above(c) for c in floor))

    # Classes carrying real over-weight (a_fin > 1.3) whose val had plateaued by
    # the macro plateau: the budget the val-improvability gate would reclaim.
    over = [c for c in classes if per_class[c]['a_fin'] > 1.3 and per_class[c]['plateaued']]

    return {
        'classes': classes, 'run_id': cell['run_id'],
        'taxonomy': cell['taxonomy'], 'split': cell['split'],
        'lr_arc': cell['lr_arc'],
        'macro_mean': macro_mean, 'macro_std': macro_std, 'min_mean': min_mean,
        'macro_at': {e: float(macro_mean[e - 1]) for e in (10, 25, 40, 80)},
        'run_max': float(run_max), 'plateau_ep': plateau_ep,
        'best_macro_eps': best_macro_eps,
        'per_class': per_class,
        'corr_alpha_valmax': corr,
        'total_budget': total_budget,
        'abovemean_total': abovemean_total,
        'abovemean_plateaued': abovemean_plateaued,
        'frac_abovemean_plateaued': abovemean_plateaued / abovemean_total if abovemean_total else 0.0,
        'abovemean_floor': abovemean_floor,
        'over_alloc_classes': over,
    }


def classify(pc: dict) -> str:
    """Coarse advisory label from the train/val numbers (handover's diagnostic)."""
    if pc['n_val'] == 0:
        return 'no-val'
    if np.isnan(pc['v_max']):
        return 'no-val'
    if pc['v_max'] < 0.20:
        return 'FLOOR'        # unlearnable on val regardless of train
    if pc['gap_max'] > 0.20 and pc['v_max'] < 0.62:
        return 'memorise'     # train >> val: data-coverage limited
    if pc['v_max'] < 0.62:
        return 'ceiling'      # both mediocre, small gap: input/confusion ceiling
    if pc['v_max'] < 0.80:
        return 'mid'
    return 'healthy'


def fmt(x, p=3):
    return f'{x:.{p}f}' if not (x is None or (isinstance(x, float) and np.isnan(x))) else '  -  '


def cell_table(label: str, summ: dict) -> str:
    classes = summ['classes']
    pc = summ['per_class']
    order = sorted(classes, key=lambda c: -pc[c]['a_fin'])  # by final alpha desc
    lines = []
    lines.append(f"### {label}  ({summ['taxonomy']} / {summ['split']})  run {summ['run_id']}")
    lines.append('')
    lines.append(f"macro plateau ~epoch {summ['plateau_ep']} (run-max {summ['run_max']:.3f}); "
                 f"macro 10/25/40/80 = {summ['macro_at'][10]:.3f} / {summ['macro_at'][25]:.3f} / "
                 f"{summ['macro_at'][40]:.3f} / {summ['macro_at'][80]:.3f}; "
                 f"best-macro epochs {sorted(summ['best_macro_eps'])}")
    lines.append(f"alpha-vs-valF1max corr = {summ['corr_alpha_valmax']:.2f}; "
                 f"above-mean alpha budget = {summ['abovemean_total']:.2f}, of which "
                 f"{summ['abovemean_plateaued']:.2f} ({100 * summ['frac_abovemean_plateaued']:.0f}%) "
                 f"sits on classes already plateaued by epoch {summ['plateau_ep']} "
                 f"(floor classes alone: {summ['abovemean_floor']:.2f})")
    lines.append('')
    hdr = (f"| {'class':24s} | n_tr | n_val | a_fin | a_pk(ep) | v_max(ep) | v_fin | "
           f"t_max | gap | dV>plat | label |")
    lines.append(hdr)
    lines.append('|' + '---|' * 12)
    for c in order:
        d = pc[c]
        apk = f"{d['a_peak']:.2f}({int(d['a_peak_ep']) if not np.isnan(d['a_peak_ep']) else '-'})"
        vmx = (f"{d['v_max']:.3f}({int(d['v_max_ep']) if not np.isnan(d['v_max_ep']) else '-'})"
               if not np.isnan(d['v_max']) else '   -   ')
        gain = d.get('backhalf_gain', np.nan)
        gflag = ('flat' if d.get('plateaued') else fmt(gain, 2)) if not np.isnan(gain) else '  -  '
        lines.append(
            f"| {c:24s} | {d['n_train']:4d} | {d['n_val']:5d} | {d['a_fin']:5.2f} | "
            f"{apk:>9s} | {vmx:>10s} | {fmt(d['v_fin'])} | {fmt(d['t_max'])} | "
            f"{fmt(d['gap_max'], 2):>5s} | {gflag:>7s} | {classify(d):8s} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    cells = pickle.loads((HERE / 'arcs.pkl').read_bytes())
    summaries = {label: analyse_cell(cells[label]) for label in CELL_ORDER}

    # Console: cell-level headline line per cell.
    print('\n' + '=' * 100)
    print('OVER-ALLOCATION HEADLINE (serial-mean)')
    print('=' * 100)
    print(f"{'cell':18s} {'plateau':>7s} {'macroMax':>8s} {'corr':>6s} "
          f"{'AM_plat/AM_tot':>16s} {'(floor)':>8s}  over-allocated (a>1.3 & plateaued)")
    for label in CELL_ORDER:
        s = summaries[label]
        print(f"{label:18s} {s['plateau_ep']:7d} {s['run_max']:8.3f} "
              f"{s['corr_alpha_valmax']:6.2f} "
              f"{s['abovemean_plateaued']:6.2f}/{s['abovemean_total']:<6.2f}"
              f"({100 * s['frac_abovemean_plateaued']:3.0f}%) {s['abovemean_floor']:7.2f}  "
              f"{', '.join(s['over_alloc_classes'])}")

    md = ['# Per-class arc tables (serial-mean across 5 serials)', '',
          'Source: parse_arcs.py over the taxon_pinned_w_preds 6-cell batch. '
          'a=alpha (renorm mean 1), v=val F1, t=train F1, gap=t_max-v_max. '
          'Labels are advisory (FLOOR/memorise/ceiling/mid/healthy).', '']
    for label in CELL_ORDER:
        md.append(cell_table(label, summaries[label]))
    (HERE / 'tables.md').write_text('\n'.join(md))
    print(f"\nwrote {HERE / 'tables.md'}")

    with open(HERE / 'summaries.pkl', 'wb') as fh:
        pickle.dump(summaries, fh)
    print(f"saved {HERE / 'summaries.pkl'}")


if __name__ == '__main__':
    main()
