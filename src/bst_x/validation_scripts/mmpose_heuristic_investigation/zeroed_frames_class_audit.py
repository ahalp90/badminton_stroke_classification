"""Per-class frame-zeroing audit for the current MMPose extracts.

For each class in the active taxonomy, computes:
- N clips
- Mean fraction of frames zeroed (mean across clips of per-clip rate)
- Distribution of clips across 20%-wide drop-rate bins (0-20, 20-40,
  40-60, 60-80, 80-100% of frames zeroed)
- Optional finer-grained 10%-bin histogram for the F1-bottom-N classes

Reads ``_failed.npy`` per clip from a flat dir. Default points at the
sticky_anchor Phase 1 mixed merge (sibling of ``BST_X_MMPOSE_NPY_DIR``
named ``..._h_sticky_anchor_phase1_merged``), so the audit reflects the
current best extracts: original committed for ~30k unbusted stems plus
sticky_anchor outputs for the 1,716 busted stems.

Pass ``--run RUN_NAME`` to use a specific run's per-class F1 for class
ordering and bottom-N selection (e.g. ``--run run_20260425_150548``).
Without ``--run`` the per-class summary is sorted alphabetically and no
bottom-N histogram is produced.

Outputs land under ``analysis_outputs/`` next to this script (or at
``--out-dir``). Two files per invocation: a ``.txt`` of the full stdout
table and a ``.csv`` of the per-class rows for downstream analysis.

Run from anywhere (paths resolve relative to the repo root):

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py --run run_20260425_150548
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
# parents of mmpose_heuristic_investigation/: [0]=validation_scripts,
# [2]=bst_x, [3]=src, [4] doesn't exist; script lives 4 levels deep under repo root.
# After the Stage 4 run-artefact move, experiments/ lives at the repo root, not under src/bst_x/.
BST_X = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CLIPS_CSV = REPO_ROOT / 'notebooks' / 'clips_master.csv'
EXPERIMENTS_DIR = REPO_ROOT / 'experiments' / 'bst_x' / 'shuttleset'
DEFAULT_OUT_DIR = SCRIPT_DIR / 'analysis_outputs'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--flat-dir', type=Path, default=None,
        help='Flat per-clip dir holding `_failed.npy`. Default: '
             '<BST_X_MMPOSE_NPY_DIR>/../<base>_h_sticky_anchor_phase1_merged.',
    )
    parser.add_argument('--clips-csv', type=Path, default=DEFAULT_CLIPS_CSV)
    parser.add_argument('--split-column', default='split_v2')
    parser.add_argument(
        '--run', dest='run', default=None,
        help='Run name (e.g. run_20260425_150548) or full path to a run dir. '
             'Manifest is read for per-class F1 used for class ordering and '
             'bottom-N selection. Without --run, classes are sorted '
             'alphabetically and no bottom-N histogram is produced.',
    )
    parser.add_argument(
        '--serial', type=int, default=None,
        help='Serial in manifest to read F1 from. Default: best_serials[0].',
    )
    parser.add_argument('--bottom-n', type=int, default=10)
    parser.add_argument('--taxonomy', default='une_v1_14')
    parser.add_argument(
        '--bin-pct', type=int, default=20,
        help='Bin width (percent) for the per-class drop-rate breakdown. '
             'Must divide 100 evenly (e.g. 5, 10, 20, 25). Default: 20.',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=DEFAULT_OUT_DIR,
        help=f'Output directory. Default: {DEFAULT_OUT_DIR}',
    )
    args = parser.parse_args()
    if 100 % args.bin_pct != 0:
        raise SystemExit(
            f'--bin-pct must divide 100 evenly; got {args.bin_pct}.'
        )

    sys.path.insert(0, str(BST_X))
    from pipeline.config import derive_class_index, taxonomy_lookup

    flat_dir = _resolve_flat_dir(args.flat_dir)
    if not flat_dir.is_dir():
        raise SystemExit(f'flat_dir not found: {flat_dir}')

    run_dir = _resolve_run_dir(args.run) if args.run else None

    taxonomy = taxonomy_lookup(args.taxonomy)

    df = pd.read_csv(args.clips_csv)
    df = df[df[args.split_column].isin(['train', 'val', 'test'])]

    rates_by_class: dict[str, list[float]] = defaultdict(list)
    missing = 0
    skipped_label = 0
    for raw_type, side, stem in zip(
        df['raw_type_en'], df['player_side'], df['clip_stem'],
    ):
        # Single source of the merge + side + exclusion rule. None means the
        # taxonomy drops this raw type (e.g. unknown), so it's out-of-taxonomy
        # for the audit.
        idx = derive_class_index(taxonomy, raw_type, side)
        if idx is None:
            skipped_label += 1
            continue
        label = taxonomy.classes[idx]
        path = flat_dir / f'{stem}_failed.npy'
        if not path.exists():
            missing += 1
            continue
        rates_by_class[label].append(float(np.load(path).mean()))

    f1_by_class = _read_f1(run_dir, args.serial) if run_dir else None
    bot_n = _bottom_n(f1_by_class, args.bottom_n) if f1_by_class else []
    ordered = _order_classes(list(taxonomy.classes), f1_by_class, rates_by_class)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f'__{run_dir.name}' if run_dir else ''
    # Bin-width tag in filename only when non-default, so the default 20% run
    # keeps its existing artifact path stable across reruns.
    bin_suffix = f'__bins{args.bin_pct}pct' if args.bin_pct != 20 else ''
    suffix = f'{run_suffix}{bin_suffix}'
    txt_path = args.out_dir / f'zeroed_frames_class_audit{suffix}.txt'
    csv_path = args.out_dir / f'zeroed_frames_class_audit{suffix}.csv'

    with txt_path.open('w') as txt_f:
        with redirect_stdout(_Tee(sys.__stdout__, txt_f)):
            rows = _emit_report(
                flat_dir=flat_dir,
                taxonomy=taxonomy,
                split_column=args.split_column,
                run_dir=run_dir,
                missing=missing,
                skipped_label=skipped_label,
                ordered=ordered,
                rates_by_class=rates_by_class,
                f1_by_class=f1_by_class,
                bot_n=bot_n,
                bottom_n_arg=args.bottom_n,
                bin_pct=args.bin_pct,
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'\nwrote {txt_path}')
    print(f'wrote {csv_path}')


def _emit_report(
    *,
    flat_dir: Path,
    taxonomy,
    split_column: str,
    run_dir: Path | None,
    missing: int,
    skipped_label: int,
    ordered: list[str],
    rates_by_class: dict[str, list[float]],
    f1_by_class: dict[str, float] | None,
    bot_n: list[str],
    bottom_n_arg: int,
    bin_pct: int,
) -> list[dict]:
    print('Frame-zeroing audit')
    print(f'  flat_dir:       {flat_dir}')
    print(f'  taxonomy:       {taxonomy.name}')
    print(f'  split_column:   {split_column} (train/val/test)')
    print(f'  bin width:      {bin_pct}%')
    if run_dir is not None:
        print(f'  ordered by F1:  {run_dir.name}')
    if missing:
        print(f'  missing _failed.npy: {missing} clips skipped')
    if skipped_label:
        print(f'  out-of-taxonomy:     {skipped_label} clips skipped')
    print()

    rows: list[dict] = []
    n_bins = 100 // bin_pct
    # 1.001 on the upper edge so a clip with 100% drop falls into the last
    # bin rather than getting clipped out by np.histogram's right-open default.
    edges = [i * bin_pct / 100 for i in range(n_bins + 1)]
    edges[-1] = 1.001
    bin_labels = [f'{i*bin_pct}-{(i+1)*bin_pct}%' for i in range(n_bins)]
    bin_keys = [f'pct_clips_{i*bin_pct}_{(i+1)*bin_pct}' for i in range(n_bins)]

    header = (
        f"{'class':<32} {'N':>5} {'mean%':>7}  "
        + ' '.join(f'{lbl:>7}' for lbl in bin_labels)
    )
    print(header)
    print('-' * len(header))
    for label in ordered:
        rates = np.asarray(rates_by_class.get(label, []))
        if not len(rates):
            continue
        counts, _ = np.histogram(rates, bins=edges)
        bin_pcts = counts / len(rates) * 100
        marker = ' *' if label in bot_n else '  '
        bin_str = ' '.join(f'{p:>6.1f}%' for p in bin_pcts)
        print(
            f'{label + marker:<32} {len(rates):>5d} {rates.mean()*100:>6.1f}%  '
            f'{bin_str}'
        )
        row: dict = {
            'class': label,
            'n_clips': int(len(rates)),
            'mean_drop_pct': round(float(rates.mean() * 100), 2),
        }
        for key, p in zip(bin_keys, bin_pcts):
            row[key] = round(float(p), 2)
        row['f1'] = f1_by_class.get(label) if f1_by_class else None
        row['in_bottom_n'] = label in bot_n
        rows.append(row)
    if bot_n:
        print(f'(* = in F1-bottom-{bottom_n_arg} per --run)')
    print()

    if bot_n:
        print(
            f'Fine-grained drop-rate histogram (10% bins) '
            f'for F1-bottom-{bottom_n_arg}:\n'
        )
        edges_10 = [i / 10 for i in range(11)]
        edges_10[-1] = 1.001
        for label in bot_n:
            rates = np.asarray(rates_by_class.get(label, []))
            if not len(rates):
                continue
            counts, _ = np.histogram(rates, bins=edges_10)
            max_count = counts.max() if counts.any() else 1
            print(
                f'{label}  '
                f'(N={len(rates)}, mean={rates.mean()*100:.1f}%, '
                f'F1={f1_by_class[label]:.3f})'
            )
            for i, c in enumerate(counts):
                lo, hi = i * 10, (i + 1) * 10
                bar = '#' * int(c / max_count * 30)
                pct = c / len(rates) * 100
                print(f'  {lo:>3}-{hi:<3}%  {bar:<30}  {c:>4} clips ({pct:>5.1f}%)')
            print()

    return rows


class _Tee:
    """Write to multiple streams. Lets stdout-redirected prints hit both."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _resolve_flat_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    env = os.environ.get('BST_X_MMPOSE_NPY_DIR', '').strip()
    if not env:
        raise SystemExit(
            'neither --flat-dir nor BST_X_MMPOSE_NPY_DIR is set. '
            'Pass --flat-dir explicitly.'
        )
    committed = Path(env)
    return committed.parent / f'{committed.name}_h_sticky_anchor_phase1_merged'


def _resolve_run_dir(value: str) -> Path:
    # Accept either a bare run name (resolved under EXPERIMENTS_DIR) or a
    # full path. Letting the user pass either keeps the common case short.
    if '/' in value or os.sep in value:
        path = Path(value).expanduser().resolve()
    else:
        path = EXPERIMENTS_DIR / value
    if not path.is_dir():
        raise SystemExit(f'run dir not found: {path}')
    return path


def _read_f1(run_dir: Path, serial: int | None) -> dict[str, float]:
    manifest_path = run_dir / 'manifest.yaml'
    if not manifest_path.exists():
        raise SystemExit(f'manifest not found: {manifest_path}')
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    if serial is None:
        bs = manifest.get('best_serials') or []
        if not bs:
            raise SystemExit(
                f'no best_serials in {manifest_path}; pass --serial explicitly.'
            )
        serial = bs[0]
    for s in manifest.get('serials', []):
        if s['serial_no'] == serial:
            return s['metrics']['per_class_f1']
    raise SystemExit(f'serial {serial} not in {manifest_path}.')


def _bottom_n(f1: dict[str, float], n: int) -> list[str]:
    return [c for c, _ in sorted(f1.items(), key=lambda kv: kv[1])[:n]]


def _order_classes(
    class_list: list[str],
    f1: dict[str, float] | None,
    rates_by_class: dict[str, list[float]],
) -> list[str]:
    present = [c for c in class_list if rates_by_class.get(c)]
    if f1:
        return sorted(present, key=lambda c: f1.get(c, float('inf')))
    return sorted(present)


if __name__ == '__main__':
    main()
