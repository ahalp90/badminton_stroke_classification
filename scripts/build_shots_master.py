"""Generate the canonical BRIC shots master CSV (one row per stroke).

Single source of truth for BRIC's per-stroke metadata. Reads from:

  - ShuttleSet upstream annotations under training/data/shuttleset/annotations/
    (match.csv, setN.csv per match, video_metadata.csv, flaw_shot_records.csv).
  - BST-team v2 split CSV under src/shared/shuttleset_splits_v2.csv
    (player-leakage-corrected split, active for BRIC).
  - BST-team curation + logic mirrored into ``shared``:
      * ``shared.dataset`` — paths, EXCLUDED_VIDEOS, REMOVED_SHOTS,
        SPLITS_V2, SPLITS_BST_BASELINE, CLIP_WINDOW,
        compute_temporal_bounds, compute_clip_bounds
      * ``shared.player_mapping`` — collect_shots (A/B → Top/Bottom)
      * ``shared.taxonomy`` — STROKE_TYPES_19_ZH

No imports from ``bst_x`` — BRIC stays self-contained. No
dependency on notebook 03's clips_master.csv either; this script
generates BRIC's superset from the same primary sources notebook 03
reads.

ACTIVE SPLIT FOR BRIC: ``split_v2``. The script also writes
``split_bst_baseline`` for parity reporting against BST, but BRIC
training/eval code MUST select on ``split_v2`` — the baseline scheme
ignores player overlap between train and val/test, leaking player
identity, and any results derived from it are misleading.

Output (`training/data/shuttleset/annotations/shots_master.csv`)
carries every column notebook 03's clips_master has, plus three new
ones in source-video frame coordinates:

  - frame_num        : target stroke frame
  - shuttle_start_f  : shuttle window start, BST 'between_2_hits_with_max_limits':
                       [prev_shot, next_shot + 0.25s] clamped to ±1.5s of frame_num
  - shuttle_end_f    : shuttle window end (exclusive)

Source-video paths are derivable from the `vid` column at lookup time;
they are not stored in the CSV.

Usage:
    python -m scripts.build_shots_master
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from shared.dataset import (  # noqa: E402  — must follow sys.path insertion
    CLIP_WINDOW,
    EXCLUDED_VIDEOS,
    REMOVED_SHOTS,
    SET_INFO_DIR,
    SPLITS_BST_BASELINE,
    SPLITS_V2_PATH,
    VIDEO_METADATA_PATH,
    compute_clip_bounds,
    compute_temporal_bounds,
)
from shared.player_mapping import collect_shots  # noqa: E402
from shared.taxonomy import STROKE_TYPES_19_ZH  # noqa: E402

OUT_PATH = (
    REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations'
    / 'shots_master.csv'
)

# Final column order. ``split_v2`` is listed before ``split_bst_baseline``
# to signal which one BRIC training selects on. Downstream code reads by
# name, so order is for human eyeballing only.
OUT_COLS = [
    'clip_stem',
    'vid', 'match', 'set_id', 'rally', 'ball_round',
    'raw_type_en', 'player_side',
    'split_v2', 'split_bst_baseline',
    'aroundhead', 'backhand',
    'frame_num', 'shuttle_start_f', 'shuttle_end_f',
]


def collect_all_shots() -> tuple[pd.DataFrame, dict[int, int]]:
    """Walk every non-excluded match; return one row per shot with frame indices.

    Returns the raw shots dataframe (post-REMOVED_SHOTS, pre-v2-join) and
    the per-vid fps map. Columns produced here:

      vid, set, rally, ball_round, frame_num, shuttle_start_f, shuttle_end_f,
      player (Top/Bottom), type (English), match (folder name)
    """
    match_df = pd.read_csv(SET_INFO_DIR / 'match.csv')[['id', 'video', 'downcourt']]
    match_df['downcourt'] = match_df['downcourt'].astype(bool)
    match_df = match_df.set_index('id')

    # Excluded vids have empty fps cells; drop them before int cast since
    # they're never indexed into vmeta downstream (skipped by EXCLUDED_VIDEOS).
    vmeta = (
        pd.read_csv(VIDEO_METADATA_PATH)
        .set_index('id')['fps']
        .dropna().astype(int).to_dict()
    )

    parts = []
    for vid, v_info in match_df.iterrows():
        if vid in EXCLUDED_VIDEOS:
            continue
        fps = vmeta[vid]
        folder_path = SET_INFO_DIR / v_info['video']
        shots = collect_shots(SET_INFO_DIR, v_info, STROKE_TYPES_19_ZH)
        if shots.empty:
            print(f'  WARNING: vid={vid} ({v_info["video"]}) returned no shots')
            continue
        shots = compute_temporal_bounds(folder_path, shots).copy()
        # compute_clip_bounds takes a row Series; vectorise via apply.
        bounds = shots.apply(
            lambda r: compute_clip_bounds(r, CLIP_WINDOW, fps), axis=1
        )
        shots['shuttle_start_f'] = [int(b[0]) for b in bounds]
        shots['shuttle_end_f'] = [int(b[1]) for b in bounds]
        shots['frame_num'] = shots['frame_num'].astype(int)
        shots['vid'] = vid
        shots['match'] = v_info['video']
        parts.append(shots)

    raw = pd.concat(parts, ignore_index=True)

    before = len(raw)
    removed_keys = set(REMOVED_SHOTS)
    keys = list(zip(
        raw['vid'].astype(int),
        raw['set'].astype(int),
        raw['rally'].astype(int),
        raw['ball_round'].astype(int),
    ))
    raw = raw[[k not in removed_keys for k in keys]].reset_index(drop=True)
    print(f'Removed {before - len(raw)} shots via REMOVED_SHOTS  ({before:,} -> {len(raw):,})')

    return raw, vmeta


def build_master(raw: pd.DataFrame) -> pd.DataFrame:
    """Add clip_stem, splits, and v2 metadata to the raw shots dataframe."""
    vid_to_split = {v: split for split, vids in SPLITS_BST_BASELINE.items() for v in vids}

    raw = raw.copy()
    raw['ball_round'] = raw['ball_round'].astype(int)
    raw['clip_stem'] = (
        raw['vid'].astype(str) + '_'
        + raw['set'].astype(str) + '_'
        + raw['rally'].astype(str) + '_'
        + raw['ball_round'].astype(str)
    )
    raw['set_id'] = 'set' + raw['set'].astype(str)
    raw['split_bst_baseline'] = raw['vid'].map(vid_to_split)
    raw = raw.rename(columns={'player': 'player_side', 'type': 'raw_type_en'})

    assert raw['split_bst_baseline'].isna().sum() == 0, (
        'rows missing split_bst_baseline — non-excluded vid not in SPLITS?'
    )
    assert raw['clip_stem'].duplicated().sum() == 0, 'duplicate clip_stems'

    v2 = pd.read_csv(SPLITS_V2_PATH)
    v2['ball_round'] = v2['ball_round'].astype(int)
    v2 = v2.rename(columns={'split': 'split_v2'})

    v2_keys = ['match', 'set_id', 'rally', 'ball_round']
    assert v2.duplicated(subset=v2_keys).sum() == 0, 'v2 has duplicate keys'

    master = raw.merge(
        v2[v2_keys + ['split_v2', 'aroundhead', 'backhand']],
        on=v2_keys,
        how='left',
    )

    n_no_v2 = master['split_v2'].isna().sum()
    n_unknown = (master['raw_type_en'] == 'unknown').sum()
    if n_no_v2 != n_unknown:
        diff = master[master['split_v2'].isna() & (master['raw_type_en'] != 'unknown')]
        # Hard fail — split_v2 is the active split for BRIC, can't ship
        # the CSV with non-unknown rows missing it.
        raise AssertionError(
            f'{len(diff)} non-unknown rows lack split_v2 (expected 0). '
            f'First 20 mismatches:\n{diff[v2_keys + ["raw_type_en"]].head(20)}'
        )

    return master[OUT_COLS]


def check_baseline_counts(master: pd.DataFrame) -> None:
    """Pre-filter parity check vs notebook 03's clips_master row counts.

    These baselines come from notebook 01/03 and are sensitive to upstream
    annotator drift (occasional typo fixes in the raw CSVs). Tolerance ≤5
    is enough to absorb that without masking real schema breaks.
    """
    EXPECTED_TOTAL = 33_483
    EXPECTED_NON_UNKNOWN = 32_203

    n_total = len(master)
    n_non_unknown = (master['raw_type_en'] != 'unknown').sum()
    n_v2 = master['split_v2'].notna().sum()
    drift = abs(n_total - EXPECTED_TOTAL)
    print(f'Total rows:              {n_total:,}  (baseline {EXPECTED_TOTAL}, drift {drift})')
    print(f'Non-unknown rows:        {n_non_unknown:,}  (target {EXPECTED_NON_UNKNOWN})')
    print(f'Rows with v2 assignment: {n_v2:,}')
    assert drift <= 5, f'Total {n_total} drifts >5 from {EXPECTED_TOTAL} — investigate raw CSVs'
    assert n_non_unknown == n_v2, (
        f'Non-unknown ({n_non_unknown}) != v2-assigned ({n_v2}); join lost rows'
    )


def _load_upstream_flaw_lookup(
    affected: pd.DataFrame,
) -> dict[tuple[int, int, int, int], str]:
    """Mine the upstream `flaw` column from setN.csv for the given rows only.

    Returns ``{(vid, set, rally, ball_round): flaw_str}`` for keys present
    in the affected DataFrame; missing entries are absent from the dict.
    Reads at most one CSV per (vid, set_id) pair regardless of row count.
    """
    lookup: dict[tuple[int, int, int, int], str] = {}
    for (vid, set_id, match), grp in affected.groupby(['vid', 'set_id', 'match']):
        set_csv = SET_INFO_DIR / match / f'{set_id}.csv'
        if not set_csv.exists():
            continue
        df = pd.read_csv(set_csv, usecols=['rally', 'ball_round', 'flaw'])
        df['rally'] = df['rally'].astype(int)
        df['ball_round'] = df['ball_round'].astype(int)
        df = df.set_index(['rally', 'ball_round'])['flaw']
        set_n = int(set_id.removeprefix('set'))
        for _, row in grp.iterrows():
            key = (int(row['rally']), int(row['ball_round']))
            if key in df.index:
                v = df.loc[key]
                lookup[(int(vid), set_n, key[0], key[1])] = '' if pd.isna(v) else str(v)
    return lookup


def drop_invalid_bounds(master: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where adjacent-shot timing collisions invalidate the bounds.

    Cause is upstream annotator data: multiple consecutive ball_rounds
    sharing the same ``frame_num`` (impossible — each stroke is ~1s of
    real play). When this happens, ``shuttle_start_f`` (= prev shot's
    frame_num) collides with the current ``frame_num``, or the next
    shot's frame_num collides backward. Either way we can't trust the
    timing for this shot — the RGB centre, shuttle window, and court
    lookup would all be wrong.

    BST flagged ~2/3 of these as "time & frame_num not in order" in
    flaw_shot_records.csv but only with ``measure='modified'`` (not
    removed), so the bad data remains. The remaining ~1/3 are silent
    corruption the upstream annotators didn't catch — our bounds check
    surfaces them. Per BRIC's no-edits-in-bst_x rule we apply
    the filter here, downstream of the BST mirror.
    """
    bad_start = master['shuttle_start_f'] >= master['frame_num']
    bad_end = master['frame_num'] >= master['shuttle_end_f']
    bad_mask = bad_start | bad_end
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return master

    affected = master.loc[bad_mask]
    n_bad_unk = int((affected['raw_type_en'] == 'unknown').sum())
    n_bad_named = n_bad - n_bad_unk
    print(
        f'\nDropping {n_bad} rows with inconsistent shuttle bounds — '
        f'{n_bad_named} named + {n_bad_unk} unknown. '
        f'Breakdown: start>=frame_num: {int(bad_start.sum())}, '
        f'frame_num>=end: {int(bad_end.sum())}.'
    )

    # Cross-check against the upstream `flaw` column to separate
    # "upstream-flagged" (BST acknowledged but didn't drop) from
    # "silent corruption" (annotators missed it; BRIC catches it).
    upstream = _load_upstream_flaw_lookup(affected)
    n_flagged = sum(
        1 for r in affected.itertuples(index=False)
        if upstream.get(
            (int(r.vid), int(r.set_id.removeprefix('set')),
             int(r.rally), int(r.ball_round)), ''
        ) not in ('', '0', '0.0')
    )
    print(
        f'  - {n_flagged} were already flagged upstream (flaw=1)\n'
        f'  - {n_bad - n_flagged} were silent corruption (no upstream flag)'
    )

    per_vid = affected.groupby('vid').size().sort_values(ascending=False)
    print(f'Top-10 affected vids:\n{per_vid.head(10).to_string()}')
    return master.loc[~bad_mask].reset_index(drop=True)


def check_invariants(master: pd.DataFrame, vmeta: dict[int, int]) -> None:
    """Post-filter invariants: ordering, non-negativity, window duration spread."""
    bad_order = master[
        (master['shuttle_start_f'] >= master['frame_num']) |
        (master['frame_num'] >= master['shuttle_end_f'])
    ]
    assert len(bad_order) == 0, (
        f'{len(bad_order)} rows still have bad bounds after drop_invalid_bounds'
    )
    assert (master['shuttle_start_f'] >= 0).all(), 'negative shuttle_start_f'
    assert (master['frame_num'] >= 0).all(), 'negative frame_num'

    fps_series = master['vid'].map(vmeta)
    dur_sec = (master['shuttle_end_f'] - master['shuttle_start_f']) / fps_series
    print(
        f'\nWindow duration (sec): min={dur_sec.min():.3f}, '
        f'median={dur_sec.median():.3f}, max={dur_sec.max():.3f}, '
        f'mean={dur_sec.mean():.3f}'
    )
    print(
        'Quantiles: ' + ', '.join(
            f'p{p}={dur_sec.quantile(p/100):.2f}s' for p in (1, 25, 50, 75, 99)
        )
    )


def main() -> None:
    print(f'CLIP_WINDOW: {CLIP_WINDOW!r}')
    print(f'EXCLUDED_VIDEOS: {sorted(EXCLUDED_VIDEOS)}')

    raw, vmeta = collect_all_shots()
    print(f'Collected {len(raw):,} shots across {raw["vid"].nunique()} videos')

    master = build_master(raw)
    check_baseline_counts(master)         # parity vs notebook 03 (pre-filter)
    master = drop_invalid_bounds(master)  # remove timing-corrupted rows
    check_invariants(master, vmeta)       # post-filter invariants

    master.to_csv(OUT_PATH, index=False)
    print(f'\nSaved {len(master):,} rows to {OUT_PATH}')
    print(f'Columns: {list(master.columns)}')


if __name__ == '__main__':
    main()
