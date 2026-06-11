"""ShuttleSet player A/B -> Top/Bottom mapping + per-match shot collection.

Mirrored from ``bst_x.pipeline.player_mapping``. Same pattern as
``shared.dataset.compute_temporal_bounds`` etc. — keep BRIC self-contained
without importing from ``bst_x``.

The ShuttleSet CSVs label players as 'A' and 'B'. Which physical player is
Top (far court) vs Bottom (near court) depends on:

  1. The ``downcourt`` flag in match.csv (initial court assignment)
  2. Which set is being played (sides swap between sets 1 and 2)
  3. In set 3, a mid-game court switch at 11 points
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.taxonomy import ZH_TO_EN

# Columns we need from each set CSV.
_SHOT_COLS = ['rally', 'ball_round', 'frame_num',
              'roundscore_A', 'roundscore_B', 'player', 'type']


def map_players(df: pd.DataFrame, first_A_is_top: bool, set_num: int) -> pd.DataFrame:
    """Replace 'A'/'B' in the 'player' column with 'Top'/'Bottom'.

    The mapping depends on court orientation (``first_A_is_top``) and set
    number. For sets 1 and 2 players swap sides between sets; for set 3
    callers must split at the 11-point switch and call this twice.
    XOR logic:
      - If ``(first_A_is_top XOR set_num == 2)``: A -> Top, B -> Bottom
      - Otherwise: A -> Bottom, B -> Top
    """
    df = df.copy()
    if first_A_is_top ^ (set_num == 2):
        df['player'] = np.where(df['player'] == 'A', 'Top', 'Bottom')
    else:
        df['player'] = np.where(df['player'] == 'B', 'Top', 'Bottom')
    return df


def find_set3_switch_rally(df: pd.DataFrame) -> int:
    """Return the iloc splitting set-3 rallies into pre/post 11-point switch.

    In badminton, players switch sides in set 3 when one reaches 11 points.
    We find the first rally where either player's score reaches 11 and
    return the iloc of the rally AFTER it (the first post-switch rally).
    """
    i_A = df['roundscore_A'].searchsorted(11, side='left')
    i_B = df['roundscore_B'].searchsorted(11, side='left')
    i = min(i_A, i_B)

    # Guard for retirements (nobody hits 11 in the recorded data).
    if i >= len(df):
        return len(df)

    switch_rally = df.iloc[i]['rally']
    return df['rally'].searchsorted(switch_rally, side='right')


def collect_shots(
    set_info_dir: Path,
    v_info: pd.Series,
    stroke_types_zh: list[str],
) -> pd.DataFrame:
    """Collect every shot in a video across all sets, with Top/Bottom mapped.

    :param set_info_dir: Path containing the per-match folders.
    :param v_info: Series from match.csv with 'video' and 'downcourt'.
    :param stroke_types_zh: Chinese stroke-type names to keep (other rows
        are filtered out — usually noise / unrecognised type strings).
    :return: DataFrame with columns
        ``set, rally, ball_round, frame_num, roundscore_A, roundscore_B,
        player ('Top'/'Bottom'), type (English)``.
    """
    folder_path = set_info_dir / v_info['video']
    first_A_is_top = bool(v_info['downcourt'])
    collected = []

    # Sets 1 and 2.
    for set_i in range(1, 3):
        csv_path = folder_path / f'set{set_i}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df = df[df['type'].isin(stroke_types_zh)]
        df.insert(0, 'set', np.full(len(df), set_i, dtype=int))
        df = map_players(df, first_A_is_top, set_i)
        collected.append(df)

    # Set 3 (if it exists): split at the 11-point court switch.
    csv_path = folder_path / 'set3.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df.insert(0, 'set', np.full(len(df), 3, dtype=int))

        i_split = find_set3_switch_rally(df)
        df_before = map_players(df.iloc[:i_split], first_A_is_top, 1)
        df_after = map_players(df.iloc[i_split:], first_A_is_top, 2)

        df_before = df_before[df_before['type'].isin(stroke_types_zh)]
        df_after = df_after[df_after['type'].isin(stroke_types_zh)]
        collected.extend([df_before, df_after])

    if not collected:
        return pd.DataFrame(columns=['set'] + _SHOT_COLS)

    result = pd.concat(collected).reset_index(drop=True)
    result['type'] = result['type'].map(ZH_TO_EN)
    return result
