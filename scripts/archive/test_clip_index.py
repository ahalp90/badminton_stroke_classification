import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x'))

from pipeline.clip_index import build_clip_path_index  # noqa: E402

clips_dir = Path('/scratch/comp320a/ShuttleSet/clips')
idx = build_clip_path_index(clips_dir)
print(
    f'entries:       {len(idx)}')  # expect 33,481

df = pd.read_csv('/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv')
csv_stems = set(df['clip_stem'])
idx_stems = set(idx.keys())

missing = csv_stems - idx_stems
extra = idx_stems - csv_stems
print(f'in CSV, missing on disk: {len(missing)}')  # expect 0
print(
    f'on disk, missing in CSV: {len(extra)}')  # expect 0

# Spot-check one entry actually resolves
sample_stem, sample_path = next(iter(idx.items()))
print(f'sample: {sample_stem} -> {sample_path}')
assert sample_path.exists(), 'resolved path does not exist'
print('OK')