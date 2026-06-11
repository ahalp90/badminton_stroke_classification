#!/usr/bin/env python3
"""V1 verification gate: compare CSV-driven label histograms to the existing
baseline collated labels.npy.

Refactor invariant: the new collate_npy() must produce, for the merged_25 +
split_bst_baseline + drop_unknown=False configuration, the same per-class
label histogram as the historical on-disk collated arrays. That confirms
the taxonomy mapping and the master CSV's split/player_side columns line up
with how the original folder-walk path placed clips.

This script does NOT run the full collation (no .npy I/O for the per-clip
shards). It only re-derives label_idx per row of clips_master.csv via the
same taxonomy logic used in collate_npy(), then compares per-class counts
to whatever already exists at <baseline_collated>/{train,val,test}/labels.npy.

Usage on engelbart:
  python scripts/verify_v1_collate.py \\
      --clips-csv notebooks/clips_master.csv \\
      --baseline-collated /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_collated_between_2_hits_with_max_limits_seq_100

Exits non-zero on any per-class count mismatch. Pass before running any
ablation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x'))

from pipeline.config import TAXONOMIES, Taxonomy  # noqa: E402


def derive_label_idx(
    clips_df: pd.DataFrame,
    taxonomy: Taxonomy,
    drop_unknown: bool,
) -> tuple[np.ndarray, list[str]]:
    """Apply the taxonomy to each row, return label indices + ordered class list."""
    df = clips_df
    if drop_unknown:
        df = df[df['raw_type_en'] != 'unknown']

    class_ls = taxonomy.class_list()
    class_to_idx = {s: i for i, s in enumerate(class_ls)}
    standalone_set = taxonomy.standalone_set
    merge_map = taxonomy.merge_map or {}

    out = []
    for raw_type, side in zip(df['raw_type_en'], df['player_side']):
        merged = merge_map.get(raw_type, raw_type)
        label_str = merged if merged in standalone_set else f'{side}_{merged}'
        if label_str not in class_to_idx:
            raise ValueError(
                f'Derived label {label_str!r} not in taxonomy '
                f'{taxonomy.name!r}.class_list()'
            )
        out.append(class_to_idx[label_str])
    return np.asarray(out, dtype=np.int64), class_ls


def histogram(arr: np.ndarray, n_classes: int) -> dict[int, int]:
    counts = Counter(int(x) for x in arr)
    return {i: counts.get(i, 0) for i in range(n_classes)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips-csv', type=Path, required=True)
    parser.add_argument(
        '--baseline-collated', type=Path, required=True,
        help='Existing baseline collated dir containing {train,val,test}/labels.npy.',
    )
    parser.add_argument(
        '--taxonomy', default='merged_25', choices=list(TAXONOMIES.keys()),
        help='Defaults to merged_25 (the historical baseline).',
    )
    parser.add_argument(
        '--split-column', default='split_bst_baseline',
        help='Column in clips_csv giving train/val/test assignment.',
    )
    parser.add_argument(
        '--drop-unknown', action='store_true',
        help='Drop unknown rows before deriving labels (must mirror baseline).',
    )
    args = parser.parse_args()

    taxonomy = TAXONOMIES[args.taxonomy]
    class_ls = taxonomy.class_list()
    n_classes = len(class_ls)

    if not args.clips_csv.exists():
        parser.error(f'clips_csv not found: {args.clips_csv}')
    if not args.baseline_collated.is_dir():
        parser.error(f'baseline_collated dir not found: {args.baseline_collated}')

    clips_df = pd.read_csv(args.clips_csv)
    if args.split_column not in clips_df.columns:
        parser.error(
            f'split_column {args.split_column!r} not in clips_csv columns: '
            f'{list(clips_df.columns)}'
        )

    print(f'Taxonomy: {taxonomy.name}  ({n_classes} classes)')
    print(f'Split column: {args.split_column}')
    print(f'Drop unknown: {args.drop_unknown}')
    print()

    any_mismatch = False
    for set_name in ('train', 'val', 'test'):
        labels_path = args.baseline_collated / set_name / 'labels.npy'
        if not labels_path.exists():
            print(f'  [{set_name}] SKIP: {labels_path} does not exist')
            continue

        # Baseline counts from the historical collated labels.npy.
        baseline_labels = np.load(labels_path)
        baseline_hist = histogram(baseline_labels, n_classes)

        # Derived counts from the CSV (filter on split_column first).
        df_split = clips_df[clips_df[args.split_column] == set_name]
        derived_idx, _ = derive_label_idx(df_split, taxonomy, args.drop_unknown)
        derived_hist = histogram(derived_idx, n_classes)

        baseline_total = sum(baseline_hist.values())
        derived_total = sum(derived_hist.values())

        # Print side-by-side comparison.
        diffs = []
        for i in range(n_classes):
            b = baseline_hist[i]
            d = derived_hist[i]
            if b != d:
                diffs.append((i, class_ls[i], b, d))

        print(f'  [{set_name}] baseline={baseline_total} derived={derived_total} '
              f'(diff={derived_total - baseline_total})')
        if diffs:
            any_mismatch = True
            print(f'    Per-class mismatches ({len(diffs)} classes differ):')
            for idx, name, b, d in diffs:
                print(f'      {idx:3d} {name:30s} baseline={b:6d} derived={d:6d} '
                      f'(diff={d - b:+d})')
        else:
            print(f'    All {n_classes} per-class counts match.')

    print()
    if any_mismatch:
        print('FAIL: at least one per-class count differs. Stop and investigate.')
        return 1
    print('PASS: per-class label histograms match across all splits.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
