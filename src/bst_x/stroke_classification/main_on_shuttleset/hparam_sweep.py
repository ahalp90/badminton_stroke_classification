"""Hparam search orchestration wrapper for bst_x_train.py.

Drives a sequence of cells (each a single-knob or multi-knob hparam variant)
through bst_x_train.py one serial at a time, applying within-cell early-kill
rules and between-cell verdict-conditional skipping. State persists to the
session dir so abrupt termination (tmux death, network blip) can be resumed.

Author the search plan as ``<session_dir>/config.yaml``, then::

    python -m main_on_shuttleset.hparam_sweep <session_dir>

To bootstrap a new session with a template config, use::

    python -m main_on_shuttleset.hparam_sweep --new-session <name>

Resumption: re-running with the same session_dir picks up where it left off
by reading per-cell ``manifest.yaml`` files (authoritative for serial counts)
and ``state.json`` (orchestration queue). On conflict, manifest wins.

Full design + decision rationale: ``scratch/architecture_notes/hparam_search_wrapper.md``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


# ==========================================================================
# Paths
# ==========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / 'experiments'
SWEEPS_DIR = EXPERIMENTS_DIR / 'aug_hparam_sweep'
TEST_LOGS_DIR = SCRIPT_DIR / 'test_logs'


# ==========================================================================
# Tunables (overridable from session config)
# ==========================================================================

DEFAULT_TUNABLES = {
    'pruning': {
        # Cumulative-mean macro must stay within `tolerance` of cell-start
        # ref. S1 exempted (too noisy).
        'macro_tolerance': {'s2': 0.025, 's3': 0.015, 's4': 0.007},
        # Single-serial min F1 floor. S1 widened to absorb seed noise on
        # the noisier-than-macro min metric; S2+ uses the empirical floor.
        'min_f1_floor': {'s1': 0.38, 's2_onward': 0.40},
    },
    'verdict': {
        'win_macro_delta': 0.005,
        'tie_macro_delta': 0.005,
        'win_min_delta': -0.005,
        'tie_min_delta': -0.010,
    },
    # Cells with per-seed macro stdev above this trigger an advisory note
    # in the search log; verdict thresholds may not hold for high-variance
    # cells.
    'high_variance_warn_stdev': 0.010,
}

# Empirical wall-clock for bst_x_train on engelbart, used for time-remaining
# estimates only. ETA quality scales with how stable the per-serial wall-clock
# is; revise here if the cluster's per-serial baseline drifts.
SECONDS_PER_SERIAL = 25 * 60


# ==========================================================================
# Config loading + validation
# ==========================================================================

def load_config(session_dir: Path) -> dict:
    """Load and validate ``config.yaml`` in the session dir."""
    config_path = session_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml not found at {config_path}')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Sanity-check a session config. Raises ``ValueError`` on problems."""
    required_top = ['session_name', 'reference', 'base_config', 'cells']
    for key in required_top:
        if key not in config:
            raise ValueError(f'config missing required field {key!r}')

    ref = config['reference']
    for key in ['current_best_run', 'wipe_drop_best_run']:
        if key not in ref:
            raise ValueError(f'config.reference missing {key!r}')

    base_aug = config['base_config'].get('augmentation')
    if not base_aug:
        raise ValueError('config.base_config.augmentation missing')
    aug_keys = ['p_flip', 'p_jitter', 'cap_y', 'cap_x', 'eps']
    for key in aug_keys:
        if key not in base_aug:
            raise ValueError(f'config.base_config.augmentation missing {key!r}')

    cells = config['cells']
    if not cells:
        raise ValueError('config.cells is empty')

    cell_names = set()
    for cell in cells:
        if 'name' not in cell:
            raise ValueError(f'cell entry missing name: {cell!r}')
        if cell['name'] in cell_names:
            raise ValueError(f'duplicate cell name: {cell["name"]!r}')
        cell_names.add(cell['name'])

    # `requires:` clauses can only reference earlier cells.
    seen_so_far = set()
    for cell in cells:
        req = cell.get('requires')
        if req:
            referenced = _extract_requires_referenced(req)
            for r in referenced:
                if r not in seen_so_far:
                    raise ValueError(
                        f'cell {cell["name"]!r} requires references unknown or '
                        f'later cell {r!r}; requires can only reference cells '
                        f'earlier in the queue.'
                    )
        seen_so_far.add(cell['name'])


def _extract_requires_referenced(clause: str) -> list[str]:
    """Return the cell-name identifiers appearing in a `requires:` clause."""
    reserved = {'and', 'or', 'not', 'WIN', 'TIE', 'LOSE', 'True', 'False'}
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', clause)
    return [t for t in tokens if t not in reserved]


def evaluate_requires(clause: str, cell_verdicts: dict) -> bool:
    """Evaluate a `requires:` clause against current cell verdicts.

    :param clause: e.g. ``"p_flip_25 != LOSE"`` or ``"a == WIN and b != LOSE"``.
    :param cell_verdicts: mapping from cell name to verdict string. Skipped
                          parents pass ``'SKIPPED'`` so a downstream
                          ``requires: "parent != LOSE"`` evaluates True for
                          a skipped parent. If you want skipped parents to
                          gate children off, write ``"parent == WIN"`` or
                          ``"parent in (WIN, TIE)"``.
    """
    namespace = {'WIN': 'WIN', 'TIE': 'TIE', 'LOSE': 'LOSE', 'SKIPPED': 'SKIPPED',
                 'FAILED': 'FAILED'}
    namespace.update(cell_verdicts)
    try:
        return bool(eval(clause, {'__builtins__': {}}, namespace))
    except Exception as exc:
        raise ValueError(f'Failed to evaluate requires {clause!r}: {exc}')


def build_tunables(config: dict) -> dict:
    """Merge ``DEFAULT_TUNABLES`` with optional config overrides."""
    tunables = copy.deepcopy(DEFAULT_TUNABLES)
    cfg_pruning = config.get('pruning', {})
    for key, val in cfg_pruning.items():
        if isinstance(val, dict) and isinstance(tunables['pruning'].get(key), dict):
            tunables['pruning'][key].update(val)
        else:
            tunables['pruning'][key] = val
    cfg_verdict = config.get('verdict', {})
    tunables['verdict'].update(cfg_verdict)
    if 'high_variance_warn_stdev' in config:
        tunables['high_variance_warn_stdev'] = config['high_variance_warn_stdev']
    return tunables


# ==========================================================================
# State.json IO (atomic write, halt on failure)
# ==========================================================================

def init_state(config: dict, session_dir: Path) -> dict:
    """Build initial state dict from config + reference run manifests."""
    ref = config['reference']
    for key in ('current_best_run', 'wipe_drop_best_run'):
        manifest_path = _manifest_path(ref[key])
        if not manifest_path.exists():
            raise FileNotFoundError(
                f'config.reference.{key} = {ref[key]!r} but no manifest '
                f'exists at {manifest_path}. Check the run id is correct '
                f'and the run dir lives under {EXPERIMENTS_DIR}.'
            )
    cb_mean = read_run_mean(ref['current_best_run'])
    cb_per_class = read_run_per_class_mean(ref['current_best_run'])
    wd_mean = read_run_mean(ref['wipe_drop_best_run'])

    state = {
        'session_name': config['session_name'],
        'session_dir': str(session_dir.resolve()),
        'started_at': dt.datetime.now().isoformat(timespec='seconds'),
        'current_best_run': ref['current_best_run'],
        'current_best_mean': cb_mean,
        'current_best_per_class': cb_per_class,
        'wipe_drop_best_run': ref['wipe_drop_best_run'],
        'wipe_drop_best_mean': wd_mean,
        'cells': {
            cell['name']: {
                'status': 'pending',          # pending | running | complete | killed | skipped
                'config_index': i,
                'run_id': None,
                'log_path': None,
                'augmentation': None,
                'verdict': None,
                'killed_reason': None,
                'killed_at_serial': None,
                'skipped_reason': None,
                'serials_done': 0,
                'serials': [],                 # per-serial records from manifest.yaml
                'cumulative_mean': None,
                'mean': None,
                'per_class_mean': None,
                'best_serial': None,
                'top_movers': None,
                'kill_ref_macro': None,
                'verdict_ref_macro': None,
                'verdict_ref_min': None,
                'verdict_ref_per_class': None,
                'macro_stdev': None,
            }
            for i, cell in enumerate(config['cells'])
        },
    }
    return state


def save_state(session_dir: Path, state: dict) -> None:
    """Atomic-write ``state.json``. Halt the wrapper on write/rename failure."""
    state_path = session_dir / 'state.json'
    tmp_path = state_path.with_suffix('.json.tmp')
    try:
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, sort_keys=False)
        os.replace(tmp_path, state_path)
    except OSError as exc:
        sys.stderr.write(
            f'\n[hparam_sweep] FATAL: state.json write/rename failed: {exc}\n'
            f'In-memory state ahead of disk; halting to prevent divergence on resume.\n'
        )
        sys.exit(1)


def load_state(session_dir: Path) -> dict | None:
    """Load existing state.json, or None if absent.

    Raises ``RuntimeError`` with a user-friendly message if the file is present
    but malformed. The wrapper has no rebuild-from-manifest fallback: the
    session dir is expected to be in git, so the recovery procedure is
    ``git checkout -- state.json`` (or restore from the previous good state).
    """
    state_path = session_dir / 'state.json'
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'state.json at {state_path} is malformed: {exc}\n'
            f'The wrapper does not auto-rebuild from per-cell manifests. '
            f'Restore from git (`git checkout -- {state_path.name}`) or '
            f'remove the file to start the session over from scratch.'
        ) from exc


# ==========================================================================
# Manifest reading (per-cell run dirs and the reference runs)
# ==========================================================================

def _manifest_path(run_id: str) -> Path:
    return EXPERIMENTS_DIR / run_id / 'manifest.yaml'


def read_manifest(run_id: str) -> dict:
    """Read a run's manifest.yaml. Empty dict if missing."""
    path = _manifest_path(run_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_run_mean(run_id: str) -> dict:
    """Return 5-serial mean of headline metrics for an existing complete run."""
    manifest = read_manifest(run_id)
    serials = manifest.get('serials', [])
    if len(serials) != 5:
        raise ValueError(
            f'{run_id}: expected 5 completed serials in manifest, found {len(serials)}'
        )
    metrics_keys = ['macro_f1', 'min_f1', 'accuracy', 'top2_accuracy']
    return {k: sum(s['metrics'][k] for s in serials) / 5 for k in metrics_keys}


def read_run_per_class_mean(run_id: str) -> dict:
    """Return 5-serial mean per-class F1 for an existing complete run."""
    manifest = read_manifest(run_id)
    serials = manifest.get('serials', [])
    if len(serials) != 5:
        raise ValueError(
            f'{run_id}: expected 5 completed serials in manifest, found {len(serials)}'
        )
    classes = list(serials[0]['metrics']['per_class_f1'].keys())
    return {
        cls: sum(s['metrics']['per_class_f1'][cls] for s in serials) / 5
        for cls in classes
    }


REQUIRED_METRIC_KEYS = ('macro_f1', 'min_f1', 'accuracy', 'top2_accuracy', 'per_class_f1')


def read_cell_serials(run_id: str) -> list[dict]:
    """Return the cell's per-serial entries from its manifest.yaml.

    Verifies each entry has the metrics keys the wrapper depends on. A
    malformed manifest is loud failure rather than a silent KeyError deep
    in the kill check.
    """
    serials = read_manifest(run_id).get('serials', [])
    for s in serials:
        sn = s.get('serial_no', '?')
        metrics = s.get('metrics') or {}
        missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
        if missing:
            raise RuntimeError(
                f'{run_id}/manifest.yaml serial {sn}: missing metrics keys '
                f'{missing}. Manifest may be from an aborted partial write or '
                f'a different bst_x_train version.'
            )
    return serials


# ==========================================================================
# Metric reductions
# ==========================================================================

def cumulative_mean(serials: list[dict]) -> dict | None:
    """Mean of headline metrics across the supplied serials. None if empty."""
    if not serials:
        return None
    keys = ['macro_f1', 'min_f1', 'accuracy', 'top2_accuracy']
    return {k: sum(s['metrics'][k] for s in serials) / len(serials) for k in keys}


def per_class_mean(serials: list[dict]) -> dict:
    """Mean of per-class F1 across serials. Empty dict if no serials."""
    if not serials:
        return {}
    classes = list(serials[0]['metrics']['per_class_f1'].keys())
    return {
        cls: sum(s['metrics']['per_class_f1'][cls] for s in serials) / len(serials)
        for cls in classes
    }


def per_seed_stdev(serials: list[dict], metric_key: str) -> float:
    """Sample stdev across serials for one headline metric. 0.0 if < 2 serials."""
    n = len(serials)
    if n < 2:
        return 0.0
    vals = [s['metrics'][metric_key] for s in serials]
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return variance ** 0.5


def top_movers(cell_per_class: dict, ref_per_class: dict, n: int = 3) -> list[tuple[str, float]]:
    """Top-n classes by absolute delta vs ref. Each item: (class, delta).

    Ties on |delta| broken by alphabetical class order (deterministic).
    """
    deltas = [(cls, val - ref_per_class.get(cls, 0.0)) for cls, val in cell_per_class.items()]
    deltas.sort(key=lambda item: (-abs(item[1]), item[0]))
    return deltas[:n]


def pick_best_serial(serials: list[dict]) -> dict:
    """Highest min F1, ties broken on macro F1."""
    return max(serials, key=lambda s: (s['metrics']['min_f1'], s['metrics']['macro_f1']))


# ==========================================================================
# Kill / verdict logic
# ==========================================================================

def check_kill(serials_done: int, cum_mean: dict, latest_serial: dict,
               kill_ref_macro: float, tunables: dict) -> tuple[bool, str]:
    """Apply kill rules after a serial completes.

    :return: ``(should_kill, reason)``. ``reason`` is ``''`` on no-kill.
    """
    floor = tunables['pruning']['min_f1_floor']
    serial_min = latest_serial['metrics']['min_f1']
    threshold = floor['s1'] if serials_done == 1 else floor['s2_onward']
    if serial_min < threshold:
        return (True,
                f'min F1 floor: serial {serials_done} min F1 {serial_min:.4f} '
                f'below threshold {threshold:.2f}.')

    tol_map = tunables['pruning']['macro_tolerance']
    tol_key = f's{serials_done}'
    if tol_key in tol_map:
        tolerance = tol_map[tol_key]
        deficit = kill_ref_macro - cum_mean['macro_f1']
        if deficit > tolerance:
            return (True,
                    f'macro tolerance: cumulative mean macro after S{serials_done} '
                    f'is {cum_mean["macro_f1"]:.4f}, ref {kill_ref_macro:.4f}, '
                    f'deficit {deficit*100:.2f}% exceeds tolerance {tolerance*100:.1f}%.')

    return (False, '')


def compute_verdict(killed: bool, serials_done: int, mean: dict | None,
                    verdict_ref_macro: float, verdict_ref_min: float,
                    tunables: dict) -> str:
    """Return WIN / TIE / LOSE."""
    if killed or serials_done < 5 or mean is None:
        return 'LOSE'

    v = tunables['verdict']
    macro_delta = mean['macro_f1'] - verdict_ref_macro
    min_delta = mean['min_f1'] - verdict_ref_min

    if macro_delta >= v['win_macro_delta'] and min_delta >= v['win_min_delta']:
        return 'WIN'
    if abs(macro_delta) < v['tie_macro_delta'] and min_delta >= v['tie_min_delta']:
        return 'TIE'
    return 'LOSE'


# ==========================================================================
# bst_x_train invocation
# ==========================================================================

def invoke_bst_train(serial_no: int, run_id: str, log_path: Path,
                     augmentation: dict) -> int:
    """Run bst_x_train.py for one serial via subprocess. Returns exit code."""
    # PYTHONPATH per the bst_x_train.py module header. Same shape regardless of cwd.
    src_root = SCRIPT_DIR.parent.parent
    stroke_root = SCRIPT_DIR.parent
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join([str(src_root), str(stroke_root)])

    cmd = [
        sys.executable, '-m', 'main_on_shuttleset.bst_x_train',
        '--serial-no', str(serial_no),
        '--run-id', run_id,
        '--log-path', str(log_path),
        '--p-flip', str(augmentation['p_flip']),
        '--p-jitter', str(augmentation['p_jitter']),
        '--cap-y', str(augmentation['cap_y']),
        '--cap-x', str(augmentation['cap_x']),
        '--eps', str(augmentation['eps']),
    ]
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ==========================================================================
# Cell flow
# ==========================================================================

def resolve_cell_aug(cell_config: dict, base_config: dict) -> dict:
    """Merge cell-level augmentation overrides on top of base augmentation."""
    base = base_config.get('augmentation', {})
    overrides = cell_config.get('augmentation', {})
    return {**base, **overrides}


def run_cell(state: dict, cell_config: dict, session_dir: Path,
             tunables: dict, base_config: dict) -> None:
    """Run a single cell to completion or kill."""
    cell_name = cell_config['name']
    cell_state = state['cells'][cell_name]

    aug_resolved = resolve_cell_aug(cell_config, base_config)

    # Mint or reuse the cell's run_id and log path. Persist to state.json
    # immediately so a crash before the cell starts work doesn't orphan the
    # minted run dir into a phantom: resume re-uses the same run_id.
    #
    # Run_id uses microsecond resolution (deviating from bst_x_train's
    # second-resolution default) so adjacent fast-killing cells don't share
    # a timestamp and accidentally write into the same run dir. Cells that
    # take a real 2hr won't collide regardless; this is the safety net.
    if cell_state['run_id'] is None:
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_id = f'run_{timestamp}'
        cell_state['run_id'] = run_id
        cell_state['log_path'] = str(TEST_LOGS_DIR / f'test_{timestamp}.log')
        cell_state['augmentation'] = aug_resolved
        save_state(session_dir, state)
    run_id = cell_state['run_id']
    log_path = Path(cell_state['log_path'])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Snapshot kill/verdict refs at cell start. These never move during the
    # cell's lifetime, so the cell is always graded against the same target
    # even if other cells promote current_best mid-run.
    if cell_state['kill_ref_macro'] is None:
        cell_state['kill_ref_macro'] = state['current_best_mean']['macro_f1']
        cell_state['verdict_ref_macro'] = state['current_best_mean']['macro_f1']
        cell_state['verdict_ref_min'] = state['current_best_mean']['min_f1']
        cell_state['verdict_ref_per_class'] = dict(state['current_best_per_class'])

    cell_state['status'] = 'running'
    save_state(session_dir, state)

    print_cell_start_block(state, cell_name, cell_state, aug_resolved)

    # Reconcile against manifest in case a prior wrapper invocation completed
    # serials we don't know about. Manifest is authoritative.
    manifest_serials = read_cell_serials(run_id)
    if len(manifest_serials) > cell_state['serials_done']:
        cell_state['serials'] = manifest_serials
        cell_state['serials_done'] = len(manifest_serials)
        cell_state['cumulative_mean'] = cumulative_mean(manifest_serials)
        save_state(session_dir, state)

    # Drive remaining serials.
    while cell_state['serials_done'] < 5:
        next_serial = cell_state['serials_done'] + 1
        print(f'\n[hparam_sweep] {cell_name}: launching S{next_serial}')
        rc = invoke_bst_train(
            serial_no=next_serial, run_id=run_id, log_path=log_path,
            augmentation=aug_resolved,
        )
        if rc != 0:
            # Don't take the whole session down for one bad serial. CUDA OOM,
            # transient driver hiccups, or filesystem blips can trip a non-zero
            # rc on an otherwise-healthy queue. Mark the cell failed, advance
            # to the next one, surface in the search log so it's visible on
            # return.
            cell_state['status'] = 'failed'
            cell_state['failed_reason'] = (
                f'bst_x_train exited with code {rc} on serial {next_serial}.'
            )
            cell_state['failed_at_serial'] = next_serial
            cell_state['verdict'] = 'LOSE'
            cell_state['mean'] = cell_state.get('cumulative_mean')
            cell_state['per_class_mean'] = per_class_mean(cell_state['serials'])
            save_state(session_dir, state)
            sys.stderr.write(
                f'\n[hparam_sweep] bst_x_train exited with code {rc} on '
                f'cell {cell_name!r} serial {next_serial}. Marking cell '
                f'failed and advancing to next cell.\n'
            )
            sys.stderr.flush()
            return

        # Pull the new serial entry from the cell's manifest.yaml.
        manifest_serials = read_cell_serials(run_id)
        if len(manifest_serials) != next_serial:
            sys.stderr.write(
                f'\n[hparam_sweep] FATAL: expected {next_serial} serials in '
                f'{run_id} manifest after S{next_serial} call, found '
                f'{len(manifest_serials)}. Halting.\n'
            )
            sys.exit(1)

        cell_state['serials'] = manifest_serials
        cell_state['serials_done'] = next_serial
        cell_state['cumulative_mean'] = cumulative_mean(manifest_serials)
        save_state(session_dir, state)

        latest = manifest_serials[-1]
        should_kill, reason = check_kill(
            serials_done=next_serial,
            cum_mean=cell_state['cumulative_mean'],
            latest_serial=latest,
            kill_ref_macro=cell_state['kill_ref_macro'],
            tunables=tunables,
        )
        if should_kill:
            cell_state['status'] = 'killed'
            cell_state['killed_reason'] = reason
            cell_state['killed_at_serial'] = next_serial
            print(f'[hparam_sweep] {cell_name}: KILLED at S{next_serial}. {reason}')
            break

    # Finalise.
    if cell_state['status'] != 'killed':
        cell_state['status'] = 'complete'

    cell_state['mean'] = cell_state['cumulative_mean']
    cell_state['per_class_mean'] = per_class_mean(cell_state['serials'])
    if cell_state['serials']:
        cell_state['top_movers'] = top_movers(
            cell_state['per_class_mean'],
            cell_state['verdict_ref_per_class'],
        )
        cell_state['best_serial'] = pick_best_serial(cell_state['serials'])['serial_no']

    cell_state['verdict'] = compute_verdict(
        killed=(cell_state['status'] == 'killed'),
        serials_done=cell_state['serials_done'],
        mean=cell_state['mean'],
        verdict_ref_macro=cell_state['verdict_ref_macro'],
        verdict_ref_min=cell_state['verdict_ref_min'],
        tunables=tunables,
    )

    cell_state['macro_stdev'] = per_seed_stdev(cell_state['serials'], 'macro_f1')

    # Reference promotion: only after a complete 5-serial cell, only if the
    # mean macro genuinely exceeds current_best (verdict-agnostic).
    if (cell_state['status'] == 'complete'
            and cell_state['serials_done'] == 5
            and cell_state['mean']['macro_f1'] > state['current_best_mean']['macro_f1']):
        state['current_best_run'] = run_id
        state['current_best_mean'] = cell_state['mean']
        state['current_best_per_class'] = cell_state['per_class_mean']
        print(f'[hparam_sweep] {cell_name}: promoted to current_best_run.')

    save_state(session_dir, state)
    print_cell_end_block(state, cell_name, cell_state, tunables)


# ==========================================================================
# Console output
# ==========================================================================

def _format_eta(seconds_remaining: int) -> str:
    eta = dt.datetime.now() + dt.timedelta(seconds=seconds_remaining)
    hours = seconds_remaining // 3600
    mins = (seconds_remaining % 3600) // 60
    return f'~{hours}h{mins:02d}m, expected complete {eta.strftime("%H:%M %d-%b")}'


def print_cell_start_block(state: dict, cell_name: str, cell_state: dict,
                           augmentation: dict) -> None:
    cells = state['cells']
    statuses = [c['status'] for c in cells.values()]
    completed = statuses.count('complete')
    killed = statuses.count('killed')
    skipped = statuses.count('skipped')
    pending = statuses.count('pending')
    cells_remaining_after = pending  # this cell already 'running'
    serials_remaining = 5 - cell_state['serials_done']
    secs_left = (serials_remaining * SECONDS_PER_SERIAL
                 + cells_remaining_after * 5 * SECONDS_PER_SERIAL)

    cb_run = state['current_best_run']
    cb_mean = state['current_best_mean']
    wd_run = state['wipe_drop_best_run']
    wd_mean = state['wipe_drop_best_mean']

    print()
    print('=' * 70)
    print(f'Cell: {cell_name}')
    print(f'Augmentation: {augmentation}')
    print(f'Progress: {completed} done, {killed} killed, {skipped} skipped, '
          f'{cells_remaining_after} pending after this cell')
    print()
    print('Reference (cell-start snapshot):')
    print(f'  Current best: {cb_run}')
    print(f'    Mean {cb_mean["macro_f1"]:.4f} / {cb_mean["min_f1"]:.4f} / '
          f'{cb_mean["accuracy"]:.4f} / {cb_mean["top2_accuracy"]:.4f}')
    print(f'  Wipe_drop best: {wd_run}')
    print(f'    Mean {wd_mean["macro_f1"]:.4f} / {wd_mean["min_f1"]:.4f} / '
          f'{wd_mean["accuracy"]:.4f} / {wd_mean["top2_accuracy"]:.4f}')
    print()
    print(f'Time estimate (no kills): {_format_eta(secs_left)}')
    print('=' * 70, flush=True)


def print_cell_end_block(state: dict, cell_name: str, cell_state: dict,
                         tunables: dict) -> None:
    print()
    print('-' * 70)
    if cell_state['status'] == 'killed':
        print(f'Cell killed: {cell_name}')
        print(f'  Reason: {cell_state["killed_reason"]}')
        print(f'  Verdict: LOSE.')
    else:
        m = cell_state['mean']
        ref_macro = cell_state['verdict_ref_macro']
        ref_min = cell_state['verdict_ref_min']
        wd = state['wipe_drop_best_mean']
        print(f'Cell complete: {cell_name}')
        print(f'  Mean {m["macro_f1"]:.4f} / {m["min_f1"]:.4f} / '
              f'{m["accuracy"]:.4f} / {m["top2_accuracy"]:.4f}')
        print(f'  Best serial: S{cell_state["best_serial"]}')
        print(f'  Vs current best (cell-start): macro {(m["macro_f1"]-ref_macro)*100:+.1f}, '
              f'min {(m["min_f1"]-ref_min)*100:+.1f}')
        print(f'  Vs wipe_drop:                 macro {(m["macro_f1"]-wd["macro_f1"])*100:+.1f}, '
              f'min {(m["min_f1"]-wd["min_f1"])*100:+.1f}, '
              f'acc {(m["accuracy"]-wd["accuracy"])*100:+.1f}, '
              f'top-2 {(m["top2_accuracy"]-wd["top2_accuracy"])*100:+.1f}')
        print(f'  Verdict: {cell_state["verdict"]}')
        if cell_state.get('top_movers'):
            mover_strs = [f'{cls} {delta*100:+.1f}' for cls, delta in cell_state['top_movers']]
            print(f'  Top movers vs cell-start ref: {", ".join(mover_strs)}')
        if (cell_state.get('macro_stdev') or 0) > tunables['high_variance_warn_stdev']:
            print(f'  NOTE: per-seed macro stdev {cell_state["macro_stdev"]:.4f} '
                  f'exceeds calibration threshold; verdict is advisory.')
    print('-' * 70, flush=True)


# ==========================================================================
# Search log (markdown)
# ==========================================================================

def write_search_log(session_dir: Path, state: dict, tunables: dict) -> None:
    """Render and save manifest.md from current state."""
    parts: list[str] = []
    parts.append(f'# Hparam search: {state["session_name"]}\n\n')
    parts.append(f'Started: {state["started_at"]}\n\n')
    parts.append('## Reference\n\n')
    cb = state['current_best_mean']
    parts.append(f'- Current best: {state["current_best_run"]}\n')
    parts.append(f'  - Mean {cb["macro_f1"]:.4f} / {cb["min_f1"]:.4f} / '
                 f'{cb["accuracy"]:.4f} / {cb["top2_accuracy"]:.4f}\n')
    wd = state['wipe_drop_best_mean']
    parts.append(f'- Wipe_drop best: {state["wipe_drop_best_run"]}\n')
    parts.append(f'  - Mean {wd["macro_f1"]:.4f} / {wd["min_f1"]:.4f} / '
                 f'{wd["accuracy"]:.4f} / {wd["top2_accuracy"]:.4f}\n\n')

    # Summary table.
    parts.append('## Summary\n\n')
    parts.append('| Cell | Status | Mean (macro / min) | Best S | Verdict |\n')
    parts.append('|------|--------|--------------------|--------|---------|\n')
    sorted_cells = sorted(
        state['cells'].items(), key=lambda kv: kv[1]['config_index']
    )
    for cell_name, cs in sorted_cells:
        status = cs['status']
        mean_str = '—'
        if cs.get('cumulative_mean'):
            m = cs['cumulative_mean']
            mean_str = f'{m["macro_f1"]:.4f} / {m["min_f1"]:.4f}'
            if cs['serials_done'] < 5:
                mean_str += f' (partial, {cs["serials_done"]}/5)'
        best_s = cs.get('best_serial') or '—'
        verdict = cs.get('verdict') or '—'
        parts.append(f'| {cell_name} | {status} | {mean_str} | {best_s} | {verdict} |\n')
    parts.append('\n')

    # Per-cell sections.
    for cell_name, cs in sorted_cells:
        if cs['status'] == 'pending':
            continue
        parts.append(_render_cell_section(cell_name, cs, state, tunables))

    (session_dir / 'manifest.md').write_text(''.join(parts))


def _render_cell_section(cell_name: str, cs: dict, state: dict, tunables: dict) -> str:
    parts: list[str] = []
    parts.append(f'## Cell: {cell_name}\n\n')
    parts.append(f'- Run id: `{cs.get("run_id") or "—"}`\n')
    parts.append(f'- Augmentation: `{cs.get("augmentation") or "—"}`\n')
    if cs.get('kill_ref_macro') is not None:
        parts.append(
            f'- Cell-start ref: macro {cs["verdict_ref_macro"]:.4f}, '
            f'min {cs["verdict_ref_min"]:.4f}\n'
        )
    parts.append('\n')

    if cs['status'] == 'skipped':
        parts.append(f'**Skipped**: {cs.get("skipped_reason", "")}\n\n')
        return ''.join(parts)

    serials = cs.get('serials') or []
    for s in serials:
        sn = s['serial_no']
        m = s['metrics']
        cum = cumulative_mean(serials[:sn])
        parts.append(
            f'- S{sn}: macro {m["macro_f1"]:.4f}, min {m["min_f1"]:.4f}, '
            f'acc {m["accuracy"]:.4f}, top-2 {m["top2_accuracy"]:.4f} '
            f'— cumulative {cum["macro_f1"]:.4f} / {cum["min_f1"]:.4f}\n'
        )
    if serials:
        parts.append('\n')

    if cs['status'] == 'killed':
        parts.append(f'**Killed at S{cs["killed_at_serial"]}**: '
                     f'{cs["killed_reason"]}\n')
        parts.append('Verdict: LOSE.\n\n')
    elif cs['status'] == 'complete':
        m = cs['mean']
        ref_macro = cs['verdict_ref_macro']
        ref_min = cs['verdict_ref_min']
        wd = state['wipe_drop_best_mean']
        parts.append(f'PICK: S{cs["best_serial"]}\n')
        parts.append(
            f'Mean {m["macro_f1"]:.4f} / {m["min_f1"]:.4f} / '
            f'{m["accuracy"]:.4f} / {m["top2_accuracy"]:.4f}\n'
        )
        parts.append(
            f'Vs cell-start ref: macro {(m["macro_f1"]-ref_macro)*100:+.1f}, '
            f'min {(m["min_f1"]-ref_min)*100:+.1f}\n'
        )
        parts.append(
            f'Vs wipe_drop:      macro {(m["macro_f1"]-wd["macro_f1"])*100:+.1f}, '
            f'min {(m["min_f1"]-wd["min_f1"])*100:+.1f}, '
            f'acc {(m["accuracy"]-wd["accuracy"])*100:+.1f}, '
            f'top-2 {(m["top2_accuracy"]-wd["top2_accuracy"])*100:+.1f}\n'
        )
        parts.append(f'Verdict: {cs["verdict"]}\n')
        if cs.get('top_movers'):
            mover_strs = [f'{cls} {delta*100:+.1f}' for cls, delta in cs['top_movers']]
            parts.append(f'Top movers vs cell-start ref: {", ".join(mover_strs)}\n')
        if (cs.get('macro_stdev') or 0) > tunables['high_variance_warn_stdev']:
            parts.append(
                f'Note: per-seed macro stdev {cs["macro_stdev"]:.4f} exceeds '
                f'calibration threshold; verdict is advisory.\n'
            )
        parts.append('\n')

    return ''.join(parts)


# ==========================================================================
# Session driver
# ==========================================================================

def _acquire_session_lock(session_dir: Path) -> None:
    """Refuse to run if another wrapper process owns this session.

    Stale locks (PID no longer alive) are removed silently. Two wrappers
    pointed at the same session would otherwise trample each other's
    state.json saves; the lockfile catches the common tmux-pane mistake.
    """
    lock_path = session_dir / '.lock'
    if lock_path.exists():
        try:
            other_pid = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            other_pid = None
        if other_pid and _pid_alive(other_pid):
            raise RuntimeError(
                f'Another wrapper (pid {other_pid}) is already running this '
                f'session at {session_dir}. If that process is dead, remove '
                f'{lock_path} and re-run.'
            )
    lock_path.write_text(str(os.getpid()))


def _release_session_lock(session_dir: Path) -> None:
    lock_path = session_dir / '.lock'
    if lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _check_config_drift(config: dict, state: dict) -> None:
    """Refuse to resume if config.yaml's cells diverge from state.json's.

    Detects added, removed, or renamed cells. Mid-session config edits are
    a footgun: the recorded state is keyed on cell names, and silently
    merging would either KeyError on missing cells or skip cells the user
    expected to run. Better to fail loudly with a clear message.
    """
    config_cells = {c['name'] for c in config['cells']}
    state_cells = set(state['cells'].keys())
    if config_cells != state_cells:
        added = config_cells - state_cells
        removed = state_cells - config_cells
        msg_parts = []
        if added:
            msg_parts.append(f'cells added since session start: {sorted(added)}')
        if removed:
            msg_parts.append(f'cells removed since session start: {sorted(removed)}')
        raise RuntimeError(
            'config.yaml has changed since this session started. '
            + '; '.join(msg_parts) + '. '
            'Restore the original config or start a new session.'
        )


def run_session(session_dir: Path) -> None:
    """Drive the full session: walk cells in order, apply requires gating."""
    _acquire_session_lock(session_dir)
    try:
        _run_session_locked(session_dir)
    finally:
        _release_session_lock(session_dir)


def _run_session_locked(session_dir: Path) -> None:
    config = load_config(session_dir)
    state = load_state(session_dir)
    if state is None:
        state = init_state(config, session_dir)
        save_state(session_dir, state)
    else:
        _check_config_drift(config, state)

    tunables = build_tunables(config)
    base_config = config['base_config']

    for cell_config in config['cells']:
        cell_name = cell_config['name']
        cell_state = state['cells'][cell_name]

        if cell_state['status'] in ('complete', 'killed', 'skipped'):
            continue  # already finalised in a prior run

        # Evaluate requires:. Skipped, failed, and verdict-bearing cells
        # all populate the namespace so a clause referencing any earlier
        # cell evaluates without NameError. Status-derived verdict keeps
        # the clause syntax uniform.
        req = cell_config.get('requires')
        if req:
            verdicts = {}
            for n, s in state['cells'].items():
                if s.get('verdict'):
                    verdicts[n] = s['verdict']
                elif s['status'] == 'skipped':
                    verdicts[n] = 'SKIPPED'
                elif s['status'] == 'failed':
                    verdicts[n] = 'FAILED'
            try:
                ok = evaluate_requires(req, verdicts)
            except ValueError as exc:
                cell_state['status'] = 'skipped'
                cell_state['skipped_reason'] = f'requires evaluation error: {exc}'
                save_state(session_dir, state)
                print(f'[hparam_sweep] {cell_name}: skipped (requires error: {exc}).')
                write_search_log(session_dir, state, tunables)
                continue
            if not ok:
                cell_state['status'] = 'skipped'
                cell_state['skipped_reason'] = f'requires not satisfied: {req}'
                save_state(session_dir, state)
                print(f'[hparam_sweep] {cell_name}: skipped (requires not satisfied).')
                write_search_log(session_dir, state, tunables)
                continue

        run_cell(state, cell_config, session_dir, tunables, base_config)
        write_search_log(session_dir, state, tunables)

    print('\n[hparam_sweep] Session complete.')
    print(f'  State: {session_dir / "state.json"}')
    print(f'  Search log: {session_dir / "manifest.md"}')


# ==========================================================================
# CLI
# ==========================================================================

TEMPLATE_CONFIG = """\
# Hparam search session config. Cells run in YAML order; conditional cells
# (requires:) only run when the named verdict condition holds for an earlier
# cell. Augmentation overrides on a cell are merged on top of base_config's
# augmentation; absent keys inherit from base.

session_name: {name}

reference:
  current_best_run: run_20260505_154907   # current aug best
  wipe_drop_best_run: run_20260503_172922 # absolute floor reference

base_config:
  augmentation:
    p_flip:   0.5
    p_jitter: 0.3
    cap_y:    0.05
    cap_x:    0.10
    eps:      0.15

cells:
  - name: example_p_flip_25
    augmentation:
      p_flip: 0.25
"""


def cmd_new_session(name: str) -> Path:
    """Create a new session subdir under aug_hparam_sweep/ with a template config."""
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError(f'session name {name!r} must be alphanumeric / underscore')
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = SWEEPS_DIR / f'sweep_{timestamp}_{name}'
    session_dir.mkdir()
    (session_dir / 'config.yaml').write_text(TEMPLATE_CONFIG.format(name=name))
    print(f'[hparam_sweep] Created session at {session_dir}')
    print('[hparam_sweep] Edit config.yaml then run:')
    print(f'  python -m main_on_shuttleset.hparam_sweep {session_dir}')
    return session_dir


def cmd_dry_run(session_dir: Path) -> None:
    config = load_config(session_dir)
    print(f'Config valid. Session: {config["session_name"]}')
    print(f'  Cells ({len(config["cells"])}):')
    for c in config['cells']:
        req = c.get('requires', '')
        aug = c.get('augmentation', {})
        print(f'    - {c["name"]} aug={aug} requires={req!r}')
    n_serials = len(config['cells']) * 5
    secs = n_serials * SECONDS_PER_SERIAL
    print(f'\nEstimated time (no kills): {secs // 3600}h {(secs % 3600) // 60}m')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='hparam_sweep: drive bst_x_train.py through a hparam search.',
    )
    parser.add_argument(
        'session_dir', nargs='?', type=Path, default=None,
        help='Path to session dir holding config.yaml. Resumes if state.json exists.',
    )
    parser.add_argument(
        '--new-session', type=str, default=None, metavar='NAME',
        help='Create a new session subdir with a template config.yaml and exit.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate config and print queue, then exit.',
    )
    args = parser.parse_args()

    if args.new_session is not None:
        cmd_new_session(args.new_session)
        return

    if args.session_dir is None:
        parser.error('Provide a session_dir or --new-session NAME.')
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        parser.error(f'Session dir does not exist: {session_dir}')
    if not (session_dir / 'config.yaml').exists():
        parser.error(f'config.yaml missing in {session_dir}')

    if args.dry_run:
        cmd_dry_run(session_dir)
        return

    run_session(session_dir)


if __name__ == '__main__':
    main()
