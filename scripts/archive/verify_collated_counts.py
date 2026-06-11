"""Quick disk-side check on the three Phase-2 collated trees.

Reads each collated split dir, confirms the expected stack files exist, and
checks that ``labels.npy`` length and ``pos.npy`` first dim match the expected
clip count (taken from the known clips_master.csv split breakdown after
``--drop-unknown``). Pure stdlib + numpy; no project imports, so it runs
under any venv on engelbart or bourbaki.

Run::

    python src/bst_x/validation_scripts/verify_collated_counts.py

Override the scratch root with ``--root /some/other/path`` if needed.
Exits 0 on all-OK, 1 on any mismatch or missing path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


COMBOS = [
    # (taxonomy_name, split_column, expected_per_split_clip_count)
    ('une_merge_v1_nosides', 'split_v2',
        {'train': 22743, 'val': 5250, 'test': 4210}),
    ('une_merge_v1', 'split_v2',
        {'train': 22743, 'val': 5250, 'test': 4210}),
    ('merged_25', 'split_bst_baseline',
        {'train': 24866, 'val': 4000, 'test': 3337}),
]

EXPECTED_FILES = {'pos.npy', 'shuttle.npy', 'videos_len.npy', 'labels.npy', 'JnB_bone.npy'}


def check_split(split_dir: Path, expected_n: int) -> list[str]:
    """Return a list of error strings; empty list = clean."""
    errors: list[str] = []
    if not split_dir.is_dir():
        return [f"split dir missing: {split_dir}"]

    actual_files = {p.name for p in split_dir.glob('*.npy')}
    missing = EXPECTED_FILES - actual_files
    if missing:
        errors.append(f"missing files: {sorted(missing)}")

    labels_path = split_dir / 'labels.npy'
    pos_path = split_dir / 'pos.npy'
    if labels_path.is_file():
        labels = np.load(labels_path)
        if len(labels) != expected_n:
            errors.append(
                f"labels.npy length {len(labels)} != expected {expected_n}"
            )
    if pos_path.is_file():
        pos = np.load(pos_path, mmap_mode='r')
        if pos.shape[0] != expected_n:
            errors.append(
                f"pos.npy shape {pos.shape} first dim != expected {expected_n}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--root', type=Path, default=Path('/scratch/comp320a'),
        help='Scratch root holding ShuttleSet_data_<tax>/ trees '
             '(default: /scratch/comp320a).',
    )
    args = parser.parse_args()

    overall_ok = True

    for tax, split, expected in COMBOS:
        collated = args.root / f'ShuttleSet_data_{tax}' / f'npy_{tax}_{split}_dropunk'
        print(f'\n=== {tax} / {split} ===')
        print(f'Path: {collated}')
        if not collated.is_dir():
            print(f'  MISSING: collated dir not found')
            overall_ok = False
            continue

        for s in ('train', 'val', 'test'):
            sd = collated / s
            errors = check_split(sd, expected[s])
            if errors:
                overall_ok = False
                print(f'  {s}: MISMATCH')
                for e in errors:
                    print(f'    - {e}')
            else:
                labels = np.load(sd / 'labels.npy')
                pos = np.load(sd / 'pos.npy', mmap_mode='r')
                files = sorted(p.name for p in sd.glob('*.npy'))
                print(
                    f"  {s}: OK  files={files}  "
                    f"labels={len(labels)}  pos.shape={pos.shape}"
                )

    print()
    if overall_ok:
        print('ALL OK')
        return 0
    print('FAILURES PRESENT')
    return 1


if __name__ == '__main__':
    sys.exit(main())
