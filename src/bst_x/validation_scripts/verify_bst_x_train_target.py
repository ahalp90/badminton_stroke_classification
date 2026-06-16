"""Pre-flight check for ``bst_x_train.py``: confirm the hyp config resolves
to a collated dir that actually exists on disk.

Reads ``bst_x_train.py``'s active ``hyp`` namedtuple, derives the collated dir
basename via the same helper the script uses, and checks that the resulting
path exists under the expected scratch root.

Run from the repo root::

    PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \\
        python src/bst_x/validation_scripts/verify_bst_train_target.py

Override the scratch root with ``--root /some/other/path`` if needed.
Exits 0 if the hyp-resolved dir exists, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from pipeline.config import derive_npy_collated_dir_basename
from pipeline.data_access import env_path_or_none, load_repo_dotenv


def main() -> int:
    # Mirror bst_x_train's root resolution: BST_X_COLLATED_DATA_ROOT (from .env or
    # the shell) when set, else /scratch/comp320a. Keeps the pre-flight check
    # aimed at the same tree the trainer will read.
    load_repo_dotenv()
    default_root = env_path_or_none('BST_X_COLLATED_DATA_ROOT') or Path('/scratch/comp320a')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--root', type=Path, default=default_root,
        help='Root holding ShuttleSet_data_<tax>/ trees (default: '
             'BST_X_COLLATED_DATA_ROOT, else /scratch/comp320a).',
    )
    args = parser.parse_args()

    # Pull the live hyp namedtuple from bst_x_train.py without running its
    # if __name__ == '__main__': block.
    bst_x_train = importlib.import_module('main_on_shuttleset.bst_x_train')
    hyp = bst_x_train.hyp

    print('hyp config:')
    print(f'  taxonomy:        {hyp.taxonomy}')
    print(f'  split_column:    {hyp.split_column}')
    print(f'  collation_id:    {hyp.collation_id}')
    print(f'  ablation_id:     {hyp.ablation_id}  (training tag; not in the path)')
    print(f'  seq_len:         {hyp.seq_len}')
    print(f'  use_3d_pose:     {hyp.use_3d_pose}')
    print(f'  pose_style:      {hyp.pose_style}')
    print(f'  n_epochs:        {hyp.n_epochs}')
    print(f'  batch_size:      {hyp.batch_size}')

    basename = derive_npy_collated_dir_basename(
        use_3d_pose=hyp.use_3d_pose,
        seq_len=hyp.seq_len,
        split_column=hyp.split_column,
        collation_id=hyp.collation_id,
    )

    expected_dir = args.root / f'ShuttleSet_data_{hyp.taxonomy}' / basename
    print()
    print(f'Resolved collated basename: {basename}')
    print(f'Expected collated path: {expected_dir}')
    print()

    if not expected_dir.is_dir():
        print(f'  MISSING: collated dir does not exist at the expected path.')
        print(f'  Either re-run prepare_train_on_shuttleset.py for this combo,')
        print(f'  or fix hyp to point at an existing combo.')
        return 1

    splits = ('train', 'val', 'test')
    for s in splits:
        sd = expected_dir / s
        if not sd.is_dir():
            print(f'  MISSING split dir: {sd}')
            return 1
        files = sorted(p.name for p in sd.glob('*.npy'))
        print(f'  {s}/: {len(files)} files  {files}')

    print()
    print('OK -- bst_x_train.py is aimed at an existing collated dir.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
