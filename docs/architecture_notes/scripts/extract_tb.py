"""Extract per-epoch curves from TB events for all nosides runs."""
import os
import json
from tensorboard.backend.event_processing import event_accumulator

EXP_DIR = 'experiments/bst_x/shuttleset'

NOSIDES_RUNS = [
    ('run_20260425_185421', 'P1 baseline'),
    ('run_20260430_170325', 'LS 0.1*'),
    ('run_20260430_213933', 'LS 0.0'),
    ('run_20260501_073430', 'LS 0.15'),
    ('run_20260501_110525', 'LS 0.15 + cw'),
    ('run_20260501_164658', 'CDB t1g1'),
    ('run_20260501_192113', 'CDB t1g0'),
    ('run_20260501_192519', 'CDB t0.5g1'),
    ('run_20260501_230252', 'CDB t1g1 + cap'),
    ('run_20260502_075808', 'CDB t1g2'),
]

CLASSES = [
    'net_shot', 'return_net', 'smash', 'wrist_smash', 'lob', 'clear',
    'drive', 'drop', 'passive_drop', 'push', 'rush',
    'cross_court_net_shot', 'short_service', 'long_service',
]


def extract_serial(serial_dir: str) -> dict:
    """Pull per-epoch scalars from one serial's TB events."""
    ea = event_accumulator.EventAccumulator(serial_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = set(ea.Tags()['scalars'])

    out = {}
    for tag, key in [
        ('Loss/Train', 'train_loss'),
        ('Loss/Val', 'val_loss'),
        ('F1/Val_macro', 'val_macro'),
        ('F1/Val_min', 'val_min'),
    ]:
        if tag in tags:
            events = ea.Scalars(tag)
            out[key] = [(e.step, e.value) for e in events]

    # Per-class train F1, only logged in CDB runs
    train_per_class = {}
    for cls in CLASSES:
        tag = f'F1_train/{cls}'
        if tag in tags:
            events = ea.Scalars(tag)
            train_per_class[cls] = [(e.step, e.value) for e in events]
    if train_per_class:
        out['train_per_class'] = train_per_class

    # Best-epoch summaries
    for tag, key in [
        ('best/macro_f1', 'best_val_macro'),
        ('best/macro_f1_epoch', 'best_val_macro_epoch'),
        ('best/min_f1', 'best_val_min'),
        ('best/min_f1_epoch', 'best_val_min_epoch'),
        ('stopped_epoch', 'stopped_epoch'),
    ]:
        if tag in tags:
            events = ea.Scalars(tag)
            out[key] = events[-1].value if events else None

    return out


def aggregate_run(run_dir: str) -> dict:
    """Aggregate across 5 serials, keeping per-serial detail too."""
    serials = []
    for n in range(1, 6):
        sd = os.path.join(run_dir, 'tb', f'serial_{n}')
        if os.path.isdir(sd):
            serials.append(extract_serial(sd))
    return serials


def main():
    all_runs = {}
    for run_id, label in NOSIDES_RUNS:
        rd = os.path.join(EXP_DIR, run_id)
        print(f'Loading {run_id}...')
        all_runs[run_id] = {
            'label': label,
            'serials': aggregate_run(rd),
        }

    out_path = '/tmp/nosides_tb_extract.json'
    with open(out_path, 'w') as f:
        json.dump(all_runs, f)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
