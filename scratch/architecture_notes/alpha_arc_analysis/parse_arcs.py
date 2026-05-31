"""Parse the 6-cell x 5-serial taxon_pinned_w_preds batch TB logs into one pickle.

Run with the tb-viewer venv (has tensorboard):
    /home/ariel/.venvs/tb-viewer/bin/python parse_arcs.py

Pulls every per-epoch scalar (per-class F1_val / F1_train / Alpha, plus the
macro/min/loss/aux aggregates) out of each serial's event file, reconstructs
the deterministic cosine LR per cell, and reads ground-truth per-class train /
val counts from the kept predictions npz. Saves arcs.pkl for the downstream
analysis + plotting (which run under badminton-cicd for matplotlib).

Also prints a validation block against the handover's shuttleset_18 anchors;
if those reproduce, the parser is trusted on the other five cells.
"""
import math
import pickle
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXP = Path(__file__).resolve().parents[3] / 'src/bst_refactor/stroke_classification/main_on_shuttleset/experiments'
OUT = Path(__file__).resolve().parent / 'arcs.pkl'

# label -> (run_id, predictions serial kept on disk). Order matches config.yaml.
CELLS = [
    ('shuttleset_18_v2', 'run_20260530_161525_131279', 1),
    ('bst_24_v2',        'run_20260530_174818_410060', 1),
    ('bst_12_v2',        'run_20260530_192738_970644', 4),
    ('bst_25_baseline',  'run_20260530_210600_435552', 1),
    ('bst_24_baseline',  'run_20260530_225714_593038', 5),
    ('une_v1_14_v2',     'run_20260531_005535_005154', 3),
]

BASE_LR = 5e-4
WARMUP = 100
N_EPOCHS = 80
BATCH = 128


def load_scalars(serial_dir: Path) -> dict[str, dict[int, float]]:
    """All scalar tags -> {step: value} for one serial's event dir."""
    ea = EventAccumulator(str(serial_dir), size_guidance={'scalars': 0})
    ea.Reload()
    return {
        tag: {s.step: s.value for s in ea.Scalars(tag)}
        for tag in ea.Tags()['scalars']
    }


def cosine_lr(step: int, total_steps: int) -> float:
    """HF get_cosine_schedule_with_warmup, num_cycles=0.5, base_lr=5e-4."""
    if step < WARMUP:
        return BASE_LR * step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    return BASE_LR * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def class_counts(npz_path: Path) -> tuple[dict[str, int], list[str]]:
    """Per-class instance counts from a predictions npz's y_true + class_list."""
    z = np.load(npz_path, allow_pickle=True)
    classes = [str(c) for c in z['class_list']]
    counts = np.bincount(z['y_true'], minlength=len(classes))
    return {c: int(counts[i]) for i, c in enumerate(classes)}, classes


def parse_cell(run_id: str, pred_serial: int) -> dict:
    run_dir = EXP / run_id
    import yaml
    manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
    cfg = manifest['config']
    classes = cfg['classes']

    # Ground-truth counts (split-determined, so any kept serial is fine).
    pred = run_dir / 'predictions'
    n_train, train_classes = class_counts(pred / f'train_serial_{pred_serial}.npz')
    n_val, _ = class_counts(pred / f'val_serial_{pred_serial}.npz')
    assert train_classes == classes, f'{run_id}: npz class_list != manifest classes'

    N_train = sum(n_train.values())
    steps_per_epoch = math.ceil(N_train / BATCH)
    total_steps = N_EPOCHS * steps_per_epoch
    lr_arc = {e: cosine_lr(e * steps_per_epoch, total_steps) for e in range(1, N_EPOCHS + 1)}

    serials = {}
    for serial_dir in sorted((run_dir / 'tb').glob('serial_*')):
        sn = int(serial_dir.name.split('_')[1])
        sc = load_scalars(serial_dir)
        per = {
            'macro_val': sc.get('F1/Val_macro', {}),
            'min_val': sc.get('F1/Val_min', {}),
            'macro_train': sc.get('F1_train/macro', {}),
            'min_train': sc.get('F1_train/min', {}),
            'loss_train': sc.get('Loss/Train', {}),
            'loss_val': sc.get('Loss/Val', {}),
            'aux': sc.get('Schedule/aux_factor', {}),
            'f1_val': {c: sc.get(f'F1_val/{c}', {}) for c in classes},
            'f1_train': {c: sc.get(f'F1_train/{c}', {}) for c in classes},
            'alpha': {c: sc.get(f'Alpha/{c}', {}) for c in classes},
        }
        mv = per['macro_val']
        if mv:
            best_ep = max(mv, key=mv.get)
            per['best_macro_epoch'] = best_ep
            per['best_macro'] = mv[best_ep]
            per['max_epoch'] = max(mv)
        serials[sn] = per

    return {
        'run_id': run_id,
        'taxonomy': cfg['taxonomy'],
        'split': cfg['split_column'],
        'classes': classes,
        'n_train': n_train,
        'n_val': n_val,
        'N_train': N_train,
        'steps_per_epoch': steps_per_epoch,
        'lr_arc': lr_arc,
        'serials': serials,
    }


def last(arc: dict[int, float]) -> float | None:
    return arc[max(arc)] if arc else None


def peak(arc: dict[int, float]) -> tuple[float, int] | tuple[None, None]:
    if not arc:
        return None, None
    ep = max(arc, key=arc.get)
    return arc[ep], ep


def validate_shuttleset18(cell: dict) -> None:
    """Reproduce the handover's shuttleset_18 anchors (serial_1)."""
    print('\n' + '=' * 70)
    print('VALIDATION: shuttleset_18_v2 serial_1 vs handover anchors')
    print('=' * 70)
    s1 = cell['serials'][1]

    anchors = {  # class: expected_final_alpha (handover Q3 table, serial_1)
        'driven_flight': 1.84, 'wrist_smash': 1.68, 'defensive_return_drive': 1.66,
        'drive': 1.40, 'smash': 0.94, 'short_service': 0.10,
    }
    print(f"{'class':24s} {'alpha_fin':>9s} {'expect':>7s}  {'val_max':>7s} {'val_fin':>7s} {'tr_fin':>7s}")
    for c, exp_a in anchors.items():
        a = last(s1['alpha'][c])
        vmax = max(s1['f1_val'][c].values()) if s1['f1_val'][c] else float('nan')
        vfin = last(s1['f1_val'][c])
        tfin = last(s1['f1_train'][c])
        print(f"{c:24s} {a:9.2f} {exp_a:7.2f}  {vmax:7.3f} "
              f"{(vfin if vfin is not None else float('nan')):7.3f} "
              f"{(tfin if tfin is not None else float('nan')):7.3f}")

    pa, pe = peak(s1['alpha']['driven_flight'])
    print(f"\ndriven_flight alpha peak = {pa:.2f} @ epoch {pe}   (handover: ~2.50 @ 27)")

    mv = s1['macro_val']
    print(f"macro_val: ep10={mv.get(10, float('nan')):.3f} "
          f"ep25={mv.get(25, float('nan')):.3f} ep80={mv.get(80, float('nan')):.3f}   "
          f"(handover: ~0.66 / ~0.68 / ~0.69)")

    best_eps = sorted(cell['serials'][s]['best_macro_epoch'] for s in cell['serials'])
    print(f"best-macro epochs across serials = {best_eps}   (handover: [19, 31, 36, 38, 62])")


def main() -> None:
    cells = {}
    for label, run_id, pred_serial in CELLS:
        print(f'parsing {label} ({run_id}) ...')
        cells[label] = parse_cell(run_id, pred_serial)
        c = cells[label]
        print(f"  taxonomy={c['taxonomy']} split={c['split']} n_classes={len(c['classes'])} "
              f"N_train={c['N_train']} steps/epoch={c['steps_per_epoch']} "
              f"serials={sorted(c['serials'])}")

    validate_shuttleset18(cells['shuttleset_18_v2'])

    with open(OUT, 'wb') as fh:
        pickle.dump(cells, fh)
    print(f'\nsaved {OUT}')


if __name__ == '__main__':
    main()
