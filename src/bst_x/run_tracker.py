"""Experiment tracker for BST/arch training scripts.

Writes a YAML manifest per run under experiments/<run_id>/ with
hyperparameters, git SHA, and per-serial metrics + output paths. Aim
integration is optional: if the aim package is pip-installed, hparams
and metrics are mirrored into .aim/ for UI browsing (aim up). If aim
is not installed, the YAML manifest alone is the record.

Integration cost per train script: 2 function calls.

    from run_tracker import track_run, track_serial

    run_dir, run_id = track_run(config=hyp, run_id=f'run_{timestamp}')
    # existing loop:
    for serial_no in range(1, 6):
        tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
        weights_path = run_dir / 'weights' / f'serial_{serial_no}.pt'
        # ... existing training + testing ...
        track_serial(run_dir, serial_no,
                     weights_path=weights_path, tb_dir=tb_dir,
                     metrics={'macro_f1': 0.834, 'min_f1': 0.619})

config can be a @dataclass, namedtuple, dict, or any object with vars().
metrics is any flat dict of scalar values.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import aim
    _AIM_AVAILABLE = True
except ImportError:
    _AIM_AVAILABLE = False


DEFAULT_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / 'experiments' / 'bst_x' / 'shuttleset'


def track_run(
    config: Any,
    run_id: str | None = None,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
    project_root: Path | str | None = None,
    extra: dict | None = None,
    log_path: Path | str | None = None,
) -> tuple[Path, str]:
    """Create (or reopen) a run folder and write the initial manifest.yaml.

    :param config: hyperparameters object. Accepts @dataclass, namedtuple,
                   Mapping, or anything accepted by vars(). Stored verbatim
                   in the manifest, so values must be YAML-serializable.
    :param run_id: folder name under experiments_dir. Defaults to
                   'run_YYYYMMDD_HHMMSS'.
    :param experiments_dir: parent folder. Created if missing.
    :param project_root: for the git-SHA lookup. Defaults to cwd.
    :param extra: any additional top-level manifest fields (notes, env info).
    :param log_path: optional path to the run's stdout/test log file. Stored
                     on the manifest so aim_backfill.py can slice per-serial
                     blocks into Aim-run descriptions later.
    :return: (run_dir, run_id). Train script writes weights under
             run_dir/weights/ and TB event files under run_dir/tb/serial_N/.

    Idempotent if called with a run_id whose manifest already exists: the
    existing manifest is kept untouched and the paths are returned. Useful
    for resume flows.
    """
    if run_id is None:
        run_id = f'run_{datetime.now():%Y%m%d_%H%M%S}'

    run_dir = Path(experiments_dir) / run_id
    (run_dir / 'weights').mkdir(parents=True, exist_ok=True)
    (run_dir / 'tb').mkdir(parents=True, exist_ok=True)

    if _manifest_path(run_dir).exists():
        return run_dir, run_id

    proj_root = Path(project_root) if project_root is not None else Path.cwd()
    manifest = {
        'run_id': run_id,
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(proj_root),
        'git_dirty': _git_dirty(proj_root),
        'host': os.uname().nodename if hasattr(os, 'uname') else None,
        'config': _config_to_dict(config),
        'serials': [],
    }
    if log_path is not None:
        manifest['log_path'] = _relpath(log_path)
    if extra:
        manifest['extra'] = extra

    _write_manifest(run_dir, manifest)
    return run_dir, run_id


def track_serial(
    run_dir: Path | str,
    serial_no: int,
    weights_path: Path | str,
    tb_dir: Path | str | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Append (or replace) a serial entry in the run's manifest.yaml.

    Idempotent by serial_no: calling a second time with the same serial_no
    replaces the earlier entry, so re-tests can update metrics in place.
    Also opens a lightweight aim.Run(run_hash='<run_id>_s<n>') with hparams
    + metrics if aim is installed.

    :param run_dir: the Path returned by track_run.
    :param serial_no: 1-indexed serial number within the run.
    :param weights_path: final best-checkpoint path for this serial.
    :param tb_dir: TensorBoard event directory for this serial, if any.
                   Pass log_dir from the SummaryWriter. None if no TB.
    :param metrics: flat dict of scalar values, e.g.
                    {'macro_f1': 0.834, 'min_f1': 0.619, 'accuracy': 0.846}.
    :param extra: any per-serial additional fields (best-epoch, notes, etc).
    """
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)

    entry = {
        'serial_no': serial_no,
        'weights_path': _relpath(weights_path),
        'tb_dir': _relpath(tb_dir) if tb_dir is not None else None,
        'metrics': dict(metrics) if metrics else {},
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
    }
    if extra:
        entry['extra'] = extra

    manifest['serials'] = [
        s for s in manifest.get('serials', []) if s['serial_no'] != serial_no
    ]
    manifest['serials'].append(entry)
    manifest['serials'].sort(key=lambda s: s['serial_no'])

    _write_manifest(run_dir, manifest)

    if _AIM_AVAILABLE and metrics:
        mirror_to_aim(manifest, serial_no, metrics)


def _config_to_dict(config: Any) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, '_asdict'):
        return dict(config._asdict())
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _git_sha(project_root: Path) -> str | None:
    return _run_git(project_root, ['rev-parse', 'HEAD'])


def _git_dirty(project_root: Path) -> bool | None:
    out = _run_git(project_root, ['status', '--porcelain'])
    return None if out is None else bool(out)


def _run_git(project_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ['git', '-C', str(project_root), *args],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _relpath(p: Path | str) -> str:
    """Render a path relative to cwd when possible, else absolute."""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(p)


def _manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / 'manifest.yaml'


def _read_manifest(run_dir: Path) -> dict:
    with open(_manifest_path(run_dir)) as f:
        return yaml.safe_load(f) or {}


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    # Atomic write so concurrent readers (e.g. hparam_sweep wrapper polling
    # the manifest after a per-serial bst_x_train invocation) never see a
    # half-written file.
    target = _manifest_path(run_dir)
    tmp = target.with_suffix('.yaml.tmp')
    with open(tmp, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, target)


def _track_metrics(aim_run: 'aim.Run', metrics: dict) -> None:
    """Track a flat-or-one-level-nested metrics dict into an Aim run.

    Numeric leaves become single-point series. A nested dict expands to one
    'parent/child' series per entry, so per_class_f1 lands as
    'per_class_f1/Top_smash', ... rather than an opaque param blob. Anything
    non-numeric falls back to a run param.

    :param aim_run: an open aim.Run.
    :param metrics: the serial's metrics dict from the manifest.
    """
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                try:
                    aim_run.track(float(sub_value), name=f'{key}/{sub_key}')
                except (TypeError, ValueError):
                    aim_run[f'{key}/{sub_key}'] = sub_value
        else:
            try:
                aim_run.track(float(value), name=key)
            except (TypeError, ValueError):
                aim_run[key] = value


def _parse_run_datetime(manifest: dict) -> datetime | None:
    """The run's start datetime, for dating the Aim run to the actual run.

    Prefers manifest.started_at (ISO); falls back to the YYYYMMDD_HHMMSS
    stamp in run_id. None when neither parses, leaving Aim's import time.
    """
    started = manifest.get('started_at')
    if started:
        try:
            return datetime.fromisoformat(started)
        except ValueError:
            pass
    match = re.search(r'(\d{8})_(\d{6})', manifest.get('run_id', ''))
    if match:
        try:
            return datetime.strptime(f'{match.group(1)}_{match.group(2)}', '%Y%m%d_%H%M%S')
        except ValueError:
            pass
    return None


def _set_run_created_at(repo: Any, run_hash: str, when: datetime) -> None:
    """Backdate a run's created_at in the structured DB to its real run date.

    Aim's public Run API has no creation_time setter (the ORM property is
    declared with_setter=False), but the underlying structured-DB column is
    plain and writable, and the UI's run date reads from it. Best-effort: a
    failure here must not sink the mirror, so errors are logged and swallowed.
    """
    try:
        from aim.storage.structured.sql_engine.models import Run as RunModel
        session = repo.structured_db.get_session()
        run_model = session.query(RunModel).filter_by(hash=run_hash).first()
        if run_model is not None:
            run_model.created_at = when
            session.commit()
    except Exception as e:
        print(f'[run_tracker] set created_at failed (ignored): {e}', file=sys.stderr)


def mirror_to_aim(
    manifest: dict,
    serial_no: int,
    metrics: dict,
    description: str | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    curves: dict[str, list[tuple[int, float]]] | None = None,
    repo: Any = None,
    do_index: bool = True,
) -> bool:
    """Mirror one serial into Aim as a fresh run named '<run_id>_s<N>'.

    Each call creates a new Aim run (auto hash) rather than reopening a
    semantic hash: aim >=3.x's Run(run_hash=...) only *resumes* an existing
    hash and raises on an unknown one, so a chosen hash can't be created.
    Idempotency is therefore the caller's job (aim_backfill deletes the
    same-named run before recreating); within one run the new entry is
    force-indexed so it shows up without waiting for the next `aim up`.

    :param description: freeform text shown in the Aim UI for this run.
                        Typically the per-serial test-log block from
                        test_logs/*.log.
    :param tags: list of tag labels (e.g., 'legacy', 'best', 'anneal_gentle').
    :param name: human-readable alias in the UI. Defaults to '<run_id>_s<N>'.
    :param curves: {tag: [(step, value), ...]} per-epoch series parsed from
                   the serial's TB event dir, tracked at their real epoch step.
    :param repo: aim repo (path str or aim.Repo). None uses the default repo.
    :param do_index: force the new run into the searchable index after close.
    :return: True if the mirror succeeded, False if aim is unavailable or
             the call raised. Errors are logged to stderr and swallowed so
             a broken aim install can't take down a training run.
    """
    if not _AIM_AVAILABLE:
        return False
    try:
        run_id = manifest['run_id']
        aim_run = aim.Run(
            repo=repo,
            system_tracking_interval=None,
            log_system_params=False,
            capture_terminal_logs=False,
        )
        aim_run.name = name if name is not None else f'{run_id}_s{serial_no}'
        aim_run['hparams'] = manifest.get('config', {})
        aim_run['run_id'] = run_id
        aim_run['serial_no'] = serial_no
        if manifest.get('git_sha'):
            aim_run['git_sha'] = manifest['git_sha']
        if manifest.get('notes'):
            aim_run['run_notes'] = manifest['notes']
        if description is not None:
            aim_run.description = description
        for tag in (tags or []):
            aim_run.add_tag(tag)
        _track_metrics(aim_run, metrics or {})
        for tag, points in (curves or {}).items():
            for step, value in points:
                aim_run.track(value, name=tag, step=step)
        # Capture before close: the indexer needs the hash + repo afterwards.
        run_hash, run_repo = aim_run.hash, aim_run.repo
        aim_run.close()
        if do_index:
            # Fresh runs sit in meta/chunks unseen until something reindexes
            # (normally the next `aim up`). Index now so iter_runs / the UI
            # pick the run up straight away.
            from aim.sdk.index_manager import RepoIndexManager
            RepoIndexManager.get_index_manager(run_repo).index(run_hash)
        # Date the run to when it actually trained, not when it was mirrored.
        when = _parse_run_datetime(manifest)
        if when is not None:
            _set_run_created_at(run_repo, run_hash, when)
        return True
    except Exception as e:
        print(f'[run_tracker] aim mirror failed (ignored): {e}', file=sys.stderr)
        return False
