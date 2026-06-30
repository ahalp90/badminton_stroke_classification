#!/usr/bin/env python3
"""Per-class MMPose fail-rate stats joined on clips_master.csv.

Reads the flat per-clip *_failed.npy files, joins them to clips_master.csv,
applies the requested taxonomy and prints per-class
totals. Labels stratify by player_side (Top_smash is separate from Bottom_smash),
which complements validate_zeroed_frames.py's per-stroke-type view that pools
Top and Bottom together.

Usage on engelbart (from repo root) — explicit --dataset-npy-dir:
  python src/bst_x/validation_scripts/fail_rate_per_class.py \\
      --clips-csv notebooks/clips_master.csv \\
      --dataset-npy-dir /scratch/comp320a/ShuttleSet_data_une_v1_14/dataset_npy_between_2_hits_with_max_limits_flat \\
      --split-column split_bst_baseline \\
      --taxonomy une_v1_14

Or with --data-root auto-discovery (picks the *_flat subdir):
  python src/bst_x/validation_scripts/fail_rate_per_class.py \\
      --clips-csv notebooks/clips_master.csv \\
      --data-root /scratch/comp320a/ShuttleSet_data_une_v1_14 \\
      --split-column split_bst_baseline \\
      --taxonomy une_v1_14 \\
      --save-txt
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BST_REFACTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BST_REFACTOR_ROOT))

from pipeline.config import TAXONOMIES, Taxonomy, derive_class_index, taxonomy_lookup  # noqa: E402


def derive_labels(df: pd.DataFrame, taxonomy: Taxonomy) -> pd.Series:
    """Class label per row via the taxonomy's single decision point.

    Routes through derive_class_index, so excluded types (e.g. 'unknown' under a
    drop-unknown taxonomy) come back as None and nosides taxonomies never get a
    Top_/Bottom_ prefix. The caller drops the None rows.
    """
    def _label(row) -> str | None:
        idx = derive_class_index(taxonomy, row['raw_type_en'], row['player_side'])
        return None if idx is None else taxonomy.classes[idx]
    return df.apply(_label, axis=1)


class _Tee:
    """Minimal stdout tee that also captures output for saving to .txt."""

    def __init__(self):
        self._buf = io.StringIO()
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._buf.write(text)

    def flush(self):
        self._stdout.flush()

    def get_text(self) -> str:
        return self._buf.getvalue()


def _resolve_dataset_npy_dir(
    dataset_npy_dir: Path | None, data_root: Path | None,
) -> Path:
    """Resolve the flat per-clip npy dir from explicit arg or --data-root discovery.

    Mirrors validate_zeroed_frames.py's auto-discovery: picks the single
    ``*_flat/`` subdir under ``--data-root``. Fails if zero or multiple
    candidates match.
    """
    if dataset_npy_dir is not None:
        return dataset_npy_dir

    candidates = [
        d for d in sorted(data_root.iterdir())
        if d.is_dir()
        and "npy" in d.name
        and "collated" not in d.name
        and d.name.endswith("_flat")
    ]
    if not candidates:
        print(f"ERROR: no flat per-clip npy dir (*_flat/) found under "
              f"{data_root}.")
        print("Contents:", [d.name for d in data_root.iterdir() if d.is_dir()])
        print("Pass --dataset-npy-dir to override.")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"ERROR: multiple *_flat dirs found under {data_root}:")
        for c in candidates:
            print(f"  {c.name}")
        print("Pass --dataset-npy-dir to pick one explicitly.")
        sys.exit(1)
    return candidates[0]


def _run(args, dataset_npy_dir: Path) -> None:
    """Core per-class fail-rate computation and printing. Wrapped so --save-txt
    can tee stdout around it without duplicating the body."""
    taxonomy = taxonomy_lookup(args.taxonomy)
    df = pd.read_csv(args.clips_csv)
    df['label'] = derive_labels(df, taxonomy)
    # Rows the taxonomy excludes (e.g. 'unknown' under a drop-unknown taxonomy)
    # come back as None from derive_class_index; drop them before counting.
    df = df[df['label'].notna()].copy()

    # Per-clip fail stats.
    totals, faileds, missing = [], [], 0
    for stem in df['clip_stem']:
        path = dataset_npy_dir / f'{stem}_failed.npy'
        if not path.exists():
            totals.append(0)
            faileds.append(0)
            missing += 1
            continue
        arr = np.load(path)
        totals.append(len(arr))
        faileds.append(int(arr.sum()))
    df['total_frames'] = totals
    df['failed_frames'] = faileds

    if missing:
        print(f'WARNING: {missing} clips had no *_failed.npy in {dataset_npy_dir}')

    # Aggregate by (split, label).
    agg = (
        df.groupby([args.split_column, 'label'])
          .agg(clips=('clip_stem', 'size'),
               total_frames=('total_frames', 'sum'),
               failed_frames=('failed_frames', 'sum'))
          .reset_index()
    )
    agg['fail_rate'] = agg['failed_frames'] / agg['total_frames']

    print(f'Taxonomy: {taxonomy.name}   Split: {args.split_column}')
    for split in ('train', 'val', 'test'):
        sub = agg[agg[args.split_column] == split].sort_values(
            'fail_rate', ascending=False,
        )
        if sub.empty:
            continue
        print()
        print(f'[{split}]  {sub["clips"].sum()} clips, '
              f'{sub["failed_frames"].sum():,} / {sub["total_frames"].sum():,} '
              f'frames failed '
              f'({sub["failed_frames"].sum() / sub["total_frames"].sum():.2%} overall)')
        print(f'  {"class":<30} {"clips":>6}   {"failed/total":>22}  {"rate":>7}')
        for _, r in sub.iterrows():
            ratio = f'{r.failed_frames:,} / {r.total_frames:,}'
            print(f'  {r.label:<30} {r.clips:>6}   {ratio:>22}  '
                  f'{r.fail_rate:>6.2%}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips-csv', type=Path, required=True)
    parser.add_argument(
        '--data-root', type=Path, default=None,
        help='ShuttleSet_data_{taxonomy} dir. Enables *_flat auto-discovery '
             '(same pattern as validate_zeroed_frames.py).',
    )
    parser.add_argument(
        '--dataset-npy-dir', type=Path, default=None,
        help='Flat per-clip dir holding {clip_stem}_failed.npy. '
             'Required if --data-root is not given.',
    )
    parser.add_argument('--split-column', default='split_bst_baseline')
    parser.add_argument('--taxonomy', default='une_v1_14',
                        choices=list(TAXONOMIES.keys()))
    parser.add_argument(
        '--save-txt', action='store_true',
        help='Tee stdout to zeroed_frames_analysis_outputs/'
             'fail_rate_per_class_{tax_short}_{split_short}_{ts}.txt.',
    )
    args = parser.parse_args()

    if args.dataset_npy_dir is None and args.data_root is None:
        parser.error("Either --dataset-npy-dir or --data-root is required.")

    dataset_npy_dir = _resolve_dataset_npy_dir(
        args.dataset_npy_dir, args.data_root,
    )

    if args.save_txt:
        tee = _Tee()
        real_stdout = sys.stdout
        sys.stdout = tee
        try:
            _run(args, dataset_npy_dir)
        finally:
            sys.stdout = real_stdout

        syd_now = datetime.now(ZoneInfo("Australia/Sydney"))
        ts = syd_now.strftime("%Y%m%d_%H%M")
        tax_short = args.taxonomy.replace("_", "")
        split_short = args.split_column.replace("split_", "").replace("_", "")
        output_dir = (
            Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = (
            output_dir
            / f"fail_rate_per_class_{tax_short}_{split_short}_{ts}.txt"
        )
        with open(txt_path, "w") as f:
            f.write(tee.get_text())
        print(f"Saved: {txt_path}")
    else:
        _run(args, dataset_npy_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
