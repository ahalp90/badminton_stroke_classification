#!/usr/bin/env python3
"""Emit the list of clip stems whose MMPose fail rate exceeds a threshold.

Walks clips_master.csv, applies the taxonomy's split filter, loads each
clip's ``{flat_dir}/{clip_stem}_failed.npy``, computes the fail rate, and
writes stems above ``--threshold`` (one per line, sorted) to ``--output``.

By default the fail rate is computed over the whole clip. Pass ``--hit-zone``
to restrict the rate to a ``+/-hit_window`` frame window around each clip's
hit frame (matches the hit-zone definition used by
``validation_scripts/validate_zeroed_frames.py`` and the hit_zone_heatmap).

Intended for Phase 1 of the MMPose heuristic investigation: produce the
"busted" stems for the decoupled raw extract (see
``docs/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md``).

Usage on engelbart (whole-clip, the original criterion):

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py \\
        --flat-dir /scratch/comp320a/ShuttleSet_data_une_v1_14/dataset_npy_between_2_hits_with_max_limits_flat \\
        --clips-csv /home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv \\
        --taxonomy une_v1_14 \\
        --split-column split_v2 \\
        --threshold 0.50 \\
        --exclude-unknown \\
        --output /home/ahalperi/badminton_stroke_classifier/docs/architecture_notes/busted_whole_clips_phase1.txt

Hit-zone criterion (matches the hit_zone_heatmap filter):

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py \\
        --flat-dir .../dataset_npy_between_2_hits_with_max_limits_flat \\
        --clips-csv notebooks/clips_master.csv \\
        --taxonomy une_v1_14 --split-column split_v2 \\
        --threshold 0.50 --exclude-unknown \\
        --hit-zone \\
        --set-dir /scratch/comp320a/ShuttleSet/set \\
        --video-metadata-csv /scratch/comp320a/ShuttleSet/video_metadata.csv \\
        --hit-window 10 \\
        --output .../busted_hit_zone_clips_phase1.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# File lives at src/bst_x/validation_scripts/mmpose_heuristic_investigation/<this>,
# so the repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x'))
# hit_frame_lookup lives next to validate_zeroed_frames.py as a flat module.
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x' / 'validation_scripts'))

from pipeline.config import TAXONOMIES, label_for_row, resolve_taxonomy  # noqa: E402

SPLITS = ('train', 'val', 'test')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--flat-dir', type=Path, required=True,
                        help='Directory containing {stem}_failed.npy files.')
    parser.add_argument('--clips-csv', type=Path, required=True,
                        help='Master clips CSV (one row per clip).')
    parser.add_argument('--taxonomy', default='une_v1_14',
                        choices=list(TAXONOMIES.keys()))
    parser.add_argument('--split-column', default='split_v2',
                        help='Column in clips_csv giving train/val/test.')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='Emit stems with fail_rate strictly greater than this.')
    parser.add_argument('--exclude-unknown', action='store_true',
                        help='Drop rows the taxonomy excludes (e.g. "unknown" '
                             'under a drop-unknown taxonomy).')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to write one stem per line, sorted.')
    parser.add_argument('--hit-zone', action='store_true',
                        help='Score hit-zone fail rate (+/-hit-window frames '
                             'around the hit frame) instead of whole-clip. '
                             'Requires --set-dir and --video-metadata-csv.')
    parser.add_argument('--set-dir', type=Path, default=None,
                        help='Path to data/shuttleset/set/ (required with --hit-zone).')
    parser.add_argument('--video-metadata-csv', type=Path, default=None,
                        help='Path to data/shuttleset/video_metadata.csv '
                             '(required with --hit-zone).')
    parser.add_argument('--hit-window', type=int, default=10,
                        help='Frames either side of hit to include in the '
                             'window (default 10, matches '
                             'validate_zeroed_frames.py). Ignored unless '
                             '--hit-zone is set.')
    args = parser.parse_args()

    if not args.flat_dir.is_dir():
        parser.error(f'flat-dir not found: {args.flat_dir}')
    if not args.clips_csv.exists():
        parser.error(f'clips-csv not found: {args.clips_csv}')

    hit_lookup: dict[str, int] | None = None
    if args.hit_zone:
        if args.set_dir is None or args.video_metadata_csv is None:
            parser.error('--hit-zone requires --set-dir and --video-metadata-csv')
        if not (args.set_dir / 'match.csv').is_file():
            parser.error(f'no match.csv under --set-dir: {args.set_dir}')
        if not args.video_metadata_csv.is_file():
            parser.error(f'--video-metadata-csv not found: {args.video_metadata_csv}')
        from hit_frame_lookup import build_hit_frame_lookup  # noqa: E402
        print(f'Building hit-frame lookup from {args.set_dir} ...')
        hit_lookup = build_hit_frame_lookup(args.set_dir, args.video_metadata_csv)
        print(f'  {len(hit_lookup):,} clip hit-frame indices computed')

    taxonomy = resolve_taxonomy(args.taxonomy)

    df = pd.read_csv(args.clips_csv)
    if args.split_column not in df.columns:
        parser.error(
            f'split-column {args.split_column!r} not in CSV columns: '
            f'{list(df.columns)}'
        )
    df = df[df[args.split_column].isin(SPLITS)].copy()

    emitted: list[str] = []
    emitted_by_split: Counter[str] = Counter()
    scanned = 0
    missing_npy = 0
    missing_hit = 0
    excluded_unknown = 0

    for row in df.itertuples(index=False):
        clip_stem = row.clip_stem
        split = getattr(row, args.split_column)

        if args.exclude_unknown:
            if label_for_row(taxonomy, row.raw_type_en, row.player_side) is None:
                excluded_unknown += 1
                continue

        fpath = args.flat_dir / f'{clip_stem}_failed.npy'
        if not fpath.exists():
            missing_npy += 1
            continue

        arr = np.load(fpath)
        if len(arr) == 0:
            continue

        if hit_lookup is not None:
            hit_idx = hit_lookup.get(clip_stem)
            if hit_idx is None:
                missing_hit += 1
                continue
            lo = max(0, hit_idx - args.hit_window)
            hi = min(len(arr), hit_idx + args.hit_window + 1)
            window = arr[lo:hi]
            if len(window) == 0:
                missing_hit += 1
                continue
            fail_rate = float(window.sum()) / len(window)
        else:
            fail_rate = float(arr.sum()) / len(arr)

        scanned += 1
        if fail_rate > args.threshold:
            emitted.append(clip_stem)
            emitted_by_split[split] += 1

    emitted.sort()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as fh:
        for stem in emitted:
            fh.write(stem + '\n')

    mode = 'hit-zone' if args.hit_zone else 'whole-clip'
    print(f'Taxonomy:       {taxonomy.name}')
    print(f'Split column:   {args.split_column}')
    print(f'Mode:           {mode}')
    if args.hit_zone:
        print(f'Hit window:     +/-{args.hit_window} frames')
    print(f'Threshold:      fail_rate > {args.threshold}')
    print(f'Exclude unknown: {args.exclude_unknown}')
    print(f'Flat dir:       {args.flat_dir}')
    print(f'Output:         {args.output}')
    print()
    print(f'CSV rows in splits: {len(df)}')
    if args.exclude_unknown:
        print(f'  Excluded (taxonomy drop):       {excluded_unknown}')
    print(f'  Missing _failed.npy on disk:    {missing_npy}')
    if args.hit_zone:
        print(f'  Missing hit index (skipped):    {missing_hit}')
    print(f'  Scanned clips:                  {scanned}')
    print(f'  Emitted (above threshold):      {len(emitted)}')
    for split in SPLITS:
        print(f'    {split:<6} {emitted_by_split[split]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
