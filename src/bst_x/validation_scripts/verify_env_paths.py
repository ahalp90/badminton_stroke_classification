"""Sanity-check the four BST_* env vars in the repo .env.

Loads the repo .env via ``pipeline.data_access.load_repo_dotenv``, prints
each ``BST_*`` env var, and confirms the resolved paths exist + hold the
expected per-clip files (specifically, that ``BST_X_MMPOSE_NPY_DIR`` contains
32,203 ``_failed.npy`` and ``_pos.npy`` files after the Phase-2 flip).

Run from the repo root::

    PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \\
        python src/bst_x/validation_scripts/verify_env_paths.py

Exits 0 on all-OK, 1 on any failure (missing path, count mismatch).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline.data_access import load_repo_dotenv


EXPECTED_CLIP_COUNT = 32203


def _check_dir(label: str, path_str: str | None) -> bool:
    if not path_str:
        print(f'  {label}: MISSING (env var unset)')
        return False
    p = Path(path_str)
    exists = p.is_dir()
    print(f'  {label}: {path_str}  exists={exists}')
    return exists


def main() -> int:
    load_repo_dotenv()

    clips_dir       = os.environ.get('BST_X_CLIPS_DIR')
    shuttle_dir     = os.environ.get('BST_X_SHUTTLE_NPY_DIR')
    mmpose_dir      = os.environ.get('BST_X_MMPOSE_NPY_DIR')
    clips_csv       = os.environ.get('BST_X_CLIPS_CSV')
    shuttle_csv_dir = os.environ.get('BST_X_SHUTTLE_CSV_DIR')

    print('Env vars (post .env load):')
    ok_clips       = _check_dir('BST_X_CLIPS_DIR         ', clips_dir)
    ok_shuttle     = _check_dir('BST_X_SHUTTLE_NPY_DIR ', shuttle_dir)
    ok_mmpose      = _check_dir('BST_X_MMPOSE_NPY_DIR  ', mmpose_dir)
    ok_shuttle_csv = _check_dir('BST_X_SHUTTLE_CSV_DIR ', shuttle_csv_dir)

    csv_path = Path(clips_csv) if clips_csv else None
    csv_ok = csv_path is not None and csv_path.is_file()
    print(f'  BST_X_CLIPS_CSV         : {clips_csv}  exists={csv_ok}')

    print()
    overall_ok = ok_clips and ok_shuttle and ok_mmpose and csv_ok and ok_shuttle_csv

    # Spot-check the mmpose dir specifically: should have exactly 32,203
    # _failed.npy and _pos.npy files after the Phase-2 flip.
    if ok_mmpose:
        mmpose_path = Path(mmpose_dir)
        n_failed = len(list(mmpose_path.glob('*_failed.npy')))
        n_pos    = len(list(mmpose_path.glob('*_pos.npy')))
        print(f'BST_X_MMPOSE_NPY_DIR clip-file counts:')
        print(f'  *_failed.npy: {n_failed}  expected={EXPECTED_CLIP_COUNT}'
              f'  {"OK" if n_failed == EXPECTED_CLIP_COUNT else "MISMATCH"}')
        print(f'  *_pos.npy:    {n_pos}  expected={EXPECTED_CLIP_COUNT}'
              f'  {"OK" if n_pos == EXPECTED_CLIP_COUNT else "MISMATCH"}')
        if n_failed != EXPECTED_CLIP_COUNT or n_pos != EXPECTED_CLIP_COUNT:
            overall_ok = False

    print()
    if overall_ok:
        print('ALL OK')
        return 0
    print('FAILURES PRESENT')
    return 1


if __name__ == '__main__':
    sys.exit(main())
