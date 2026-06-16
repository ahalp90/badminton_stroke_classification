"""Aggregate all run manifests under experiments/ into one printable table.

Usage:
    python run_overview.py                      # default: ./experiments/
    python run_overview.py path/to/experiments  # custom root
    python run_overview.py -c n_epochs,lr       # pick config columns
    python run_overview.py -m macro_f1,min_f1   # pick metric columns

Prints one row per run. For each selected metric, columns show mean,
stdev, and max across the run's serials. If no -m is given, all metrics
found across manifests are shown.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import yaml


def _read_manifests(experiments_dir: Path):
    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / 'manifest.yaml'
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = yaml.safe_load(f)
        if m:
            yield m


def _agg(values: list[float], stat: str) -> str:
    if not values:
        return '-'
    if stat == 'mean':
        return f'{statistics.fmean(values):.4f}'
    if stat == 'stdev':
        return f'{statistics.stdev(values):.4f}' if len(values) > 1 else '-'
    if stat == 'max':
        return f'{max(values):.4f}'
    return '-'


def _select_metric_keys(manifests: list[dict], wanted: list[str] | None) -> list[str]:
    if wanted is not None:
        return list(wanted)
    # Only auto-discover metrics whose values are scalar on at least one
    # serial. Nested structures (e.g. per_class_f1 dicts) don't aggregate
    # into mean/stdev/max and would surface as junk '-' columns here.
    found: set[str] = set()
    for m in manifests:
        for s in m.get('serials', []):
            for k, v in (s.get('metrics') or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    found.add(k)
    return sorted(found)


def _build_rows(manifests: list[dict],
                config_keys: list[str],
                metric_keys: list[str]) -> list[dict]:
    rows = []
    for m in manifests:
        cfg = m.get('config') or {}
        serials = m.get('serials') or []
        row = {
            'run_id':    m.get('run_id', '-'),
            'n_serials': len(serials),
        }
        for k in config_keys:
            row[k] = cfg.get(k, '-')
        for k in metric_keys:
            vals = [
                s['metrics'].get(k) for s in serials
                if isinstance(s.get('metrics'), dict)
                and isinstance(s['metrics'].get(k), (int, float))
            ]
            row[f'{k}_mean']  = _agg(vals, 'mean')
            row[f'{k}_stdev'] = _agg(vals, 'stdev')
            row[f'{k}_max']   = _agg(vals, 'max')
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print('(no runs)')
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    print(' | '.join(c.ljust(widths[c]) for c in cols))
    print('-+-'.join('-' * widths[c] for c in cols))
    for r in rows:
        print(' | '.join(str(r.get(c, '')).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_experiments = Path(__file__).resolve().parents[2] / 'experiments' / 'bst_x' / 'shuttleset'
    parser.add_argument('experiments_dir', nargs='?', default=str(default_experiments))
    parser.add_argument('-c', '--config', default='n_epochs,use_aux_schedule,aux_fade_end_epoch',
                        help='comma-separated config keys to include (default: a few BST-relevant keys)')
    parser.add_argument('-m', '--metrics', default=None,
                        help='comma-separated metric keys to include (default: all found)')
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.is_dir():
        print(f'Not a directory: {experiments_dir}')
        return

    manifests = list(_read_manifests(experiments_dir))
    config_keys  = [k.strip() for k in args.config.split(',')  if k.strip()]
    wanted_metrics = [k.strip() for k in args.metrics.split(',')] if args.metrics else None
    metric_keys  = _select_metric_keys(manifests, wanted_metrics)

    rows = _build_rows(manifests, config_keys, metric_keys)
    _print_table(rows)


if __name__ == '__main__':
    main()