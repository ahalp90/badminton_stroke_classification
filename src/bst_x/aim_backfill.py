"""Backfill the Aim UI from existing run-tracker manifests + TB event files.

Walks every experiments/<run_id>/manifest.yaml and mirrors each serial
into Aim as a run named '<run_id>_s<N>' carrying:

  - hparams     = manifest.config
  - metrics     = manifest.serials[i].metrics (per_class_f1 expands to
                  per_class_f1/<class> single points)
  - curves      = every per-epoch scalar from run_dir/tb/serial_N
                  (Loss/*, F1/*, F1_train/*, F1_val/*, Alpha/*, Aug/*,
                  Schedule/*, best/*), tracked at their real epoch step
  - description = the serial's test-log block (when the log was pulled down)
  - tags        = config-derived (legacy / anneal regime) + 'best' on the
                  serial whose checkpoint was kept (see _best_serials)
  - run date    = the run's started_at, so Aim sorts/shows it by training
                  time rather than backfill-import time

Re-running needs --wipe: it removes the .aim dir and rebuilds from scratch.
An in-place update can't be made clean because aim's delete_run leaves the
tag<->run links behind and recycled run-ids inherit them, bleeding tags
between runs. --repo is required for curve indexing; without it runs land in
the cwd repo.

Run in the tb-viewer venv (aim + tensorboard + tensorflow + pyyaml):
    ~/.venvs/tb-viewer/bin/python aim_backfill.py \
        --repo /path/to/.aim_repos/bst --wipe \
        src/bst_x/stroke_classification/main_on_shuttleset/experiments
    ~/.venvs/tb-viewer/bin/aim up --repo /path/to/.aim_repos/bst
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from run_tracker import _AIM_AVAILABLE, mirror_to_aim


SERIAL_HEADER_RE = re.compile(r'^=== Serial (\d+) \(', re.MULTILINE)


def _read_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _split_log_by_serial(log_text: str) -> dict[int, str]:
    """Carve a test_log file into {serial_no: 'full block text'}.

    Blocks start at '=== Serial N (...' headers and run until the next
    header or EOF. Preserves the leading header line so the Aim
    description is self-explanatory.
    """
    if not log_text.strip():
        return {}
    markers = list(SERIAL_HEADER_RE.finditer(log_text))
    blocks: dict[int, str] = {}
    for i, m in enumerate(markers):
        serial_no = int(m.group(1))
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(log_text)
        blocks[serial_no] = log_text[start:end].rstrip()
    return blocks


def _derive_tags(manifest: dict, serial_no: int) -> list[str]:
    """Auto-tags for navigating the Aim UI.

    Combines the legacy flag and an anneal-regime label from config; any
    manifest-level 'tags' list is appended as-is. The 'best' tag is handled
    separately by _best_serials (it needs the on-disk weights).
    """
    tags: list[str] = []
    if manifest.get('legacy'):
        tags.append('legacy')

    cfg = manifest.get('config') or {}
    if not cfg.get('use_aux_schedule'):
        tags.append('no_aux_anneal')
    else:
        fade = cfg.get('aux_fade_end_epoch') or 0
        n_epochs = cfg.get('n_epochs') or 0
        if fade <= 1:
            tags.append('cg_ap_off_from_start')
        elif n_epochs and fade < n_epochs * 0.3:
            tags.append('anneal_aggressive')
        else:
            tags.append('anneal_gentle')

    for t in (manifest.get('tags') or []):
        if t not in tags:
            tags.append(t)

    return tags


def _best_serials(manifest: dict, run_dir: Path) -> set[int]:
    """Serial(s) to tag 'best', preferring the checkpoint(s) actually kept.

    Priority:
      1. Pruning signal: when fewer weights survive on disk than there are
         serials, the kept checkpoint(s) are the chosen model.
      2. A manual best_serials list in the manifest.
      3. Otherwise the top macro_f1 serial.

    A run with no weights pulled down (kept count 0) skips straight to the
    manifest / macro fallback rather than reading an absent file as a signal.
    """
    serials = manifest.get('serials') or []
    if not serials:
        return set()
    kept = {
        s['serial_no'] for s in serials
        if s.get('weights_path')
        and (run_dir / 'weights' / Path(s['weights_path']).name).exists()
    }
    if 0 < len(kept) < len(serials):
        return kept
    manual = {int(n) for n in (manifest.get('best_serials') or [])}
    if manual:
        return manual
    best = max(serials, key=lambda s: (s.get('metrics') or {}).get('macro_f1', float('-inf')))
    return {best['serial_no']}


def _resolve_log_path(manifest: dict, experiments_dir: Path) -> Path | None:
    """Locate the serial test log in the test_logs/ sibling of experiments/.

    The log pairs with the run by name (test_<run_id timestamp>.log), so we
    rebuild it from run_id rather than trusting manifest.log_path, which was
    stored relative to the train-host cwd and doesn't resolve across
    machines. Returns None when the log wasn't pulled down (descriptions are
    optional).
    """
    test_logs = experiments_dir.parent / 'test_logs'
    run_id = manifest.get('run_id', '')
    candidate = test_logs / f"test_{run_id.removeprefix('run_')}.log"
    if candidate.exists():
        return candidate
    # Fall back to the recorded filename in the same folder, if any.
    rel = manifest.get('log_path')
    if rel:
        alt = test_logs / Path(rel).name
        if alt.exists():
            return alt
    return None


def _read_tb_scalars(tb_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse every scalar series from a serial's TB event dir.

    Returns {tag: [(step, value), ...]} across all event files in the dir;
    EventAccumulator merges the streaming per-epoch run with the end-of-run
    add_hparams summary. size_guidance 0 = keep every point (no downsample).
    Lazy import so run_tracker stays importable without tensorboard.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(tb_dir), size_guidance={'scalars': 0})
    accumulator.Reload()
    series: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get('scalars', []):
        series[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    return series


def backfill_run(
    run_dir: Path,
    experiments_dir: Path,
    repo: Any = None,
) -> int:
    """Mirror every serial of one run into Aim, curves included.

    :param repo: aim.Repo handle. None uses the default repo; curve indexing
                 needs an explicit --repo. The repo is rebuilt clean per run
                 of the backfill (see main), so this only ever creates runs.
    """
    manifest_path = run_dir / 'manifest.yaml'
    if not manifest_path.exists():
        return 0
    manifest = _read_manifest(manifest_path)
    serials = manifest.get('serials') or []
    if not serials:
        return 0

    log_path = _resolve_log_path(manifest, experiments_dir)
    blocks: dict[int, str] = {}
    if log_path is not None:
        blocks = _split_log_by_serial(log_path.read_text())

    run_id = manifest['run_id']
    best_set = _best_serials(manifest, run_dir)
    count = 0
    for s in serials:
        serial_no = s['serial_no']
        metrics = s.get('metrics') or {}
        description = blocks.get(serial_no)
        tags = _derive_tags(manifest, serial_no)
        if serial_no in best_set:
            tags.append('best')
        name = f'{run_id}_s{serial_no}'
        # Reconstruct the TB dir from the local run folder rather than the
        # manifest's stored tb_dir, which is relative to whatever cwd the
        # train host used. Convention: run_dir/tb/serial_N.
        tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
        curves = _read_tb_scalars(tb_dir) if tb_dir.is_dir() else None
        if mirror_to_aim(manifest, serial_no, metrics, description=description,
                         tags=tags, name=name, curves=curves, repo=repo):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments_dir', nargs='?', default='experiments')
    parser.add_argument('--repo', default=None,
                        help='Aim repo path. Needed for curve indexing and '
                             're-run idempotency. Default: aim repo in cwd.')
    parser.add_argument('--wipe', action='store_true',
                        help='Remove the .aim repo dir and rebuild from '
                             'scratch. Needed to re-run: an in-place update '
                             'bleeds tags between runs (deleted run-ids get '
                             'recycled with their old tag links).')
    args = parser.parse_args()

    if not _AIM_AVAILABLE:
        print('aim is not installed. Install with:  pip install aim')
        sys.exit(1)

    experiments_dir = Path(args.experiments_dir).resolve()
    if not experiments_dir.is_dir():
        print(f'Not a directory: {experiments_dir}')
        sys.exit(1)

    repo = None
    if args.repo is not None:
        import shutil
        from aim import Repo
        aim_dir = Path(args.repo) / '.aim'
        if args.wipe and aim_dir.exists():
            # delete_run leaves tag<->run links behind, and a recycled run-id
            # inherits them, so an in-place wipe corrupts every run's tags.
            # Removing .aim and re-initing is the only clean rebuild.
            shutil.rmtree(aim_dir)
            print(f'Removed existing repo at {aim_dir}')
        repo = Repo.from_path(args.repo, init=True)
        existing_n = len(repo.list_all_runs())
        if existing_n and not args.wipe:
            print(f'{args.repo} already holds {existing_n} run(s). Re-run with '
                  f'--wipe to rebuild cleanly (an in-place update bleeds tags).')
            sys.exit(1)

    total_runs = 0
    total_serials = 0
    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        n = backfill_run(run_dir, experiments_dir, repo=repo)
        if n:
            print(f'[{run_dir.name}] mirrored {n} serial(s)')
            total_runs += 1
            total_serials += n
    print(f'Done. {total_runs} run(s), {total_serials} serial(s) mirrored.')


if __name__ == '__main__':
    main()
