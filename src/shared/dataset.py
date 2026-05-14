"""ShuttleSet dataset paths + curation constants — shared across BRIC.

Single source of truth for where ShuttleSet annotations live and which
videos / shots have been curated out. Mirrors the relevant pieces of
`bst_refactor.pipeline.config` so BRIC code never needs to import from
that module.

ShuttleSet upstream annotations live under ``runtime/data/shuttleset/annotations/``:

  - ``ANNOTATIONS_DIR``      — root of the annotations tree (CSVs + set/)
  - ``SET_INFO_DIR``         — annotations/set/ (per-match folders + match.csv)
  - ``FLAW_RECORDS_PATH``    — annotations/flaw_shot_records.csv
  - ``VIDEO_METADATA_PATH``  — annotations/video_metadata.csv

BST-team split configuration lives alongside this module in
``src/shared/`` — it's a curation decision (small, static,
version-controlled) layered on top of the upstream data, not part
of ShuttleSet itself:

  - ``SPLITS_V2_PATH``       — src/shared/shuttleset_splits_v2.csv

Curation constants:

  - ``EXCLUDED_VIDEOS``  — set[int]   of fully dropped match IDs
  - ``REMOVED_SHOTS``    — set[tuple] of individually dropped (vid, set, rally, ball_round)
  - ``CLIP_WINDOW``      — BST's default temporal-window strategy name

Both ``EXCLUDED_VIDEOS`` and ``REMOVED_SHOTS`` are derived from
``flaw_shot_records.csv`` at import time. If the file is missing, both
are empty and a warning is emitted — fine for inspecting the module,
not fine for actual pipeline runs.

Train/val/test splits:

  - ``SPLITS_V2``            — vid -> split mapping derived from the v2 CSV.
                               Active split for BRIC; addresses player leakage
                               between train and val/test that the BST baseline
                               ignored. Use this for any new training run.
  - ``SPLITS_BST_BASELINE``  — vid ranges from BST's original config. Kept for
                               regenerating the ``split_bst_baseline`` column
                               in shots_master.csv (BST parity in
                               comparison reports), not for new training.

Both are vid-level (no vid spans multiple splits in either scheme).
``SPLITS_V2`` is derived at import time from the joined CSVs; the
baseline is hardcoded.
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at <project>/src/shared/dataset.py → parents[2] = <project>.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_DIR = _PROJECT_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations'
SET_INFO_DIR = ANNOTATIONS_DIR / 'set'
FLAW_RECORDS_PATH = ANNOTATIONS_DIR / 'flaw_shot_records.csv'
VIDEO_METADATA_PATH = ANNOTATIONS_DIR / 'video_metadata.csv'
HOMOGRAPHY_CSV_PATH = SET_INFO_DIR / 'homography.csv'

# BST-team derived split — co-located with this module since it's a
# small static curation artefact, not bulk upstream data.
SPLITS_V2_PATH = Path(__file__).resolve().parent / 'shuttleset_splits_v2.csv'


# ---------------------------------------------------------------------------
# BST clip-window default
# ---------------------------------------------------------------------------
# 'between_2_hits_with_max_limits' is BST's default. See
# bst_refactor.pipeline.clip_generator._compute_clip_bounds for the rule:
# [prev_shot, next_shot + 0.25s] clamped to ±1.5s of the target frame.
CLIP_WINDOW = 'between_2_hits_with_max_limits'


# ---------------------------------------------------------------------------
# Flaw-record parsing — derives EXCLUDED_VIDEOS + REMOVED_SHOTS from CSV
# ---------------------------------------------------------------------------
def parse_flaw_records(
    csv_path: Path = FLAW_RECORDS_PATH,
) -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Parse flaw_shot_records.csv into (excluded_videos, removed_shots).

    A row's ``measure`` column is ``'removed'`` for an exclusion. If the
    ``stroke_type`` column is ``'whole'``, the entire match is excluded;
    otherwise the specific (set, rally, ball_round) shot is removed.

    Logic mirrored from ``bst_refactor.pipeline.config.parse_flaw_records``.

    :param csv_path: Path to flaw_shot_records.csv.
    :return: (excluded_video_ids, removed_shot_tuples).
    """
    excluded_videos: set[int] = set()
    removed_shots: set[tuple[int, int, int, int]] = set()

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['measure'] != 'removed':
                continue
            match_id = int(row['match'])
            if row['stroke_type'] == 'whole':
                excluded_videos.add(match_id)
            else:
                removed_shots.add((
                    match_id,
                    int(row['set']),
                    int(row['rally']),
                    int(row['ball_round']),
                ))
    return excluded_videos, removed_shots


def _load_flaw_records() -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Load flaw records lazily; warn + return empty sets if file is missing.

    Lets this module be importable for inspection without the CSV present.
    Pipeline code that actually depends on the curation will fail loudly
    elsewhere if these are empty when they shouldn't be.
    """
    try:
        return parse_flaw_records()
    except FileNotFoundError:
        warnings.warn(
            f'{FLAW_RECORDS_PATH} not found. EXCLUDED_VIDEOS and '
            f'REMOVED_SHOTS are empty. Fine for module inspection, '
            f'wrong for pipeline runs.',
            stacklevel=2,
        )
        return set(), set()


EXCLUDED_VIDEOS, REMOVED_SHOTS = _load_flaw_records()


# ---------------------------------------------------------------------------
# Train/val/test splits
# ---------------------------------------------------------------------------

# BST's original baseline split, from `bst_refactor.pipeline.config._SPLITS_RAW`.
# Raw — does not strip EXCLUDED_VIDEOS. Apply the filter at the call site if
# needed (e.g. enrichment script does ``if vid in EXCLUDED_VIDEOS: continue``).
# Kept here so the enriched master CSV can populate ``split_bst_baseline`` for
# parity with BST's evaluation. NOT the active split for BRIC training —
# this scheme assigns whole videos to splits without considering player
# overlap, which leaks player identity between train and val/test.
SPLITS_BST_BASELINE: dict[str, list[int]] = {
    'train': list(range(1, 35)),
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}


def _load_splits_v2() -> dict[str, list[int]]:
    """Derive vid-level v2 splits from the v2 CSV joined with match.csv.

    The v2 CSV stores per-clip rows keyed by ``match`` (folder name);
    we group by ``split`` and translate match -> vid via match.csv.
    All clips for a given match share a split (verified at load).
    Returns ``{split_name: sorted_vid_list}``; ``{}`` if either CSV
    is missing.
    """
    try:
        match_to_vid: dict[str, int] = {}
        with open(SET_INFO_DIR / 'match.csv', newline='') as f:
            for row in csv.DictReader(f):
                match_to_vid[row['video']] = int(row['id'])

        vid_to_split: dict[int, str] = {}
        with open(SPLITS_V2_PATH, newline='') as f:
            for row in csv.DictReader(f):
                vid = match_to_vid.get(row['match'])
                if vid is None:
                    continue
                prior = vid_to_split.get(vid)
                if prior is not None and prior != row['split']:
                    raise ValueError(
                        f'vid={vid} ({row["match"]}) appears in multiple v2 splits: '
                        f'{prior} and {row["split"]}'
                    )
                vid_to_split[vid] = row['split']
    except FileNotFoundError as e:
        warnings.warn(
            f'{e.filename} not found. SPLITS_V2 is empty. Fine for module '
            f'inspection, wrong for training/eval runs.',
            stacklevel=2,
        )
        return {}

    grouped: dict[str, list[int]] = defaultdict(list)
    for vid, split in vid_to_split.items():
        grouped[split].append(vid)
    return {name: sorted(vids) for name, vids in grouped.items()}


# Active split for BRIC training. Player-leakage corrected.
SPLITS_V2: dict[str, list[int]] = _load_splits_v2()


# ---------------------------------------------------------------------------
# Clip-bounds derivation — mirrored from
# ``bst_refactor.pipeline.clip_generator``.
#
# We mirror rather than import because BST's ``clip_generator`` does
# ``from moviepy import VideoFileClip`` at module load (for clip-writing
# functions BRIC never calls). Importing it just to reach these two pure
# pandas/python helpers would force moviepy into the BRIC env unnecessarily.
# Same pattern as ``parse_flaw_records``: mirror the values + logic, never
# touch BST.
# ---------------------------------------------------------------------------
def compute_temporal_bounds(
    folder_path: Path, shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``start_f``/``end_f`` columns to ``shots_df`` from adjacent shots.

    For each shot, ``start_f`` is the previous shot's ``frame_num`` in the
    same rally, and ``end_f`` is the next shot's. First/last shots in a
    rally get -1 (handled as a fallback by ``compute_clip_bounds``).

    Mirrored from ``bst_refactor.pipeline.clip_generator.compute_temporal_bounds``.
    """
    parts = []
    for set_i, group_idx in shots_df.groupby('set').groups.items():
        df = pd.read_csv(folder_path / f'set{set_i}.csv')
        df = df[['rally', 'ball_round', 'frame_num']]

        df['start_f'] = df['frame_num'].shift(1)
        df['start_f'] = df['start_f'].where(df.duplicated('rally', keep='first'), -1)
        df['end_f'] = df['frame_num'].shift(-1)
        df['end_f'] = df['end_f'].where(df.duplicated('rally', keep='last'), -1)

        merged = pd.merge(
            shots_df.loc[group_idx].reset_index(drop=True),
            df,
            on=['rally', 'ball_round', 'frame_num'],
        )
        merged = merged[[
            'set', 'rally', 'ball_round',
            'start_f', 'frame_num', 'end_f',
            'roundscore_A', 'roundscore_B', 'player', 'type',
        ]]
        parts.append(merged)

    return pd.concat(parts).reset_index(drop=True)


def compute_clip_bounds(row, clip_window: str, fps: float) -> tuple[int, int]:
    """Compute (start_frame, end_frame) for one shot's window.

    ``clip_window`` is one of ``'middle_in_a_sec'``, ``'between_2_hits'``,
    ``'between_2_hits_with_max_limits'``. BRIC uses ``CLIP_WINDOW``
    (= ``'between_2_hits_with_max_limits'``) — the ±1.5s clamped variant.

    Mirrored from ``bst_refactor.pipeline.clip_generator._compute_clip_bounds``.
    """
    t = int(fps) // 2       # frames in 0.5 sec
    frame_num = int(row['frame_num'])

    if clip_window == 'middle_in_a_sec':
        return frame_num - t, frame_num + t

    eps = t // 2  # frames in 0.25 sec (extension past the next hit)
    start_f = int(row['start_f']) if row['start_f'] != -1 else (frame_num - t)
    end_f = int(row['end_f']) + eps if row['end_f'] != -1 else (frame_num + t)

    if clip_window == 'between_2_hits_with_max_limits':
        limit = int(fps) * 3 // 2  # frames in 1.5 sec
        start_f = max(start_f, frame_num - limit)
        end_f = min(end_f, frame_num + limit + eps)

    return start_f, end_f
