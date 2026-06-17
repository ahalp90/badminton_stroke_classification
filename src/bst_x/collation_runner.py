"""Drive bst_x_train.py through a list of (taxonomy, split, knobs) cells.

A stripped-down sibling of hparam_sweep.py: no kill rules, no verdict logic,
no current-best tracking. One fresh run_id per cell, N serials per cell from the
cell config (default 5; headline cells set 10). State persisted to state.json so
a killed runner resumes mid-session without re-running finished serials.

Each serial is a separate `bst_x_train --serial-no` subprocess so a crash on one
serial doesn't take down the session's progress (the manifest + state.json carry
what's done). bst_x_train resolves the collation dir from --taxonomy /
--split-column / --collation-id (+ BST_X_COLLATED_DATA_ROOT), so the runner only
forwards those plus the shared sharing-flags.

Usage (from the repo root, both package roots on PYTHONPATH)::

    PYTHONPATH=src/bst_x \\
        python -m collation_runner \\
        experiments/bst_x/shuttleset/aug_hparam_sweep/your_sweep_name
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / 'experiments' / 'bst_x' / 'shuttleset'
TEST_LOGS_DIR = EXPERIMENTS_DIR / 'test_logs'


def invoke_bst_train(*, serial_no: int, run_id: str, log_path: Path, cell: dict) -> int:
    """Run one bst_x_train serial as a subprocess. Returns the exit code."""
    bst_x_root = SCRIPT_DIR                    # src/bst_x
    src_root = SCRIPT_DIR.parent               # src
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join([str(bst_x_root), str(src_root)])
    cmd = [
        sys.executable, '-m', 'bst_x_train',
        '--serial-no',    str(serial_no),
        '--run-id',       run_id,
        '--log-path',     str(log_path),
        '--taxonomy',     cell['taxonomy'],
        '--split-column', cell['split_column'],
        '--collation-id', cell['collation_id'],
    ]
    # Optional training-time ablation tag (manifest-only); most cells omit it.
    if cell.get('ablation_id'):
        cmd += ['--ablation-id', cell['ablation_id']]
    # Optional val-improvability gate toggle. Present-and-True turns it on,
    # present-and-False forces it off; absent leaves the bst_x_train Hyp default.
    if cell.get('use_val_improvability_gate') is not None:
        cmd += ['--val-improvability-gate' if cell['use_val_improvability_gate']
                else '--no-val-improvability-gate']
    # Optional per-cell weight decay (the WD sweep dimension); cells without it
    # fall back to the bst_x_train Hyp default.
    if cell.get('weight_decay') is not None:
        cmd += ['--weight-decay', str(cell['weight_decay'])]
    return subprocess.run(cmd, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('session_dir', type=Path,
                        help='Dir holding config.yaml; state.json is written alongside.')
    args = parser.parse_args()

    config = yaml.safe_load((args.session_dir / 'config.yaml').read_text())
    state_path = args.session_dir / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'cells': {}}

    for cell in config['cells']:
        name = cell['name']
        n_serials = cell.get('n_serials', 5)
        cstate = state['cells'].setdefault(
            name, {'run_id': None, 'log_path': None, 'serials_done': 0},
        )
        # Mint the cell's run_id + log path once, persisting immediately so a
        # crash before any serial runs doesn't orphan the dir on resume.
        if cstate['run_id'] is None:
            ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            cstate['run_id'] = f'run_{ts}'
            cstate['log_path'] = str(TEST_LOGS_DIR / f'test_{ts}.log')
            state_path.write_text(json.dumps(state, indent=2))

        log_path = Path(cstate['log_path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        while cstate['serials_done'] < n_serials:
            nxt = cstate['serials_done'] + 1
            print(f'[{name}] launching S{nxt}/{n_serials} into {cstate["run_id"]}')
            rc = invoke_bst_train(
                serial_no=nxt, run_id=cstate['run_id'],
                log_path=log_path, cell=cell,
            )
            if rc != 0:
                print(f'[{name}] bst_x_train failed with code {rc} on S{nxt}; aborting cell.')
                sys.exit(rc)
            cstate['serials_done'] = nxt
            state_path.write_text(json.dumps(state, indent=2))

        print(f'[{name}] complete; run_id={cstate["run_id"]}')


if __name__ == '__main__':
    main()
