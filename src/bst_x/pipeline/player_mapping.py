"""Player A/B to Top/Bottom mapping logic, extracted from gen_my_dataset.py.

The ShuttleSet CSVs label players as 'A' and 'B'. Which physical player is
Top (far court) vs Bottom (near court) depends on:
  1. The `downcourt` flag in match.csv (initial court assignment)
  2. Which set is being played (sides swap between sets 1 and 2)
  3. In set 3, a mid-game court switch at 11 points

This module centralises that logic so it isn't duplicated across scripts.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config import ZH_TO_EN

# Columns we need from each set CSV
_SHOT_COLS = ['rally', 'ball_round', 'frame_num',
              'roundscore_A', 'roundscore_B', 'player', 'type']


def map_players(df: pd.DataFrame, first_A_is_top: bool, set_num: int) -> pd.DataFrame:
    """Replace 'A'/'B' in the 'player' column with 'Top'/'Bottom'.

    The mapping depends on court orientation (first_A_is_top) and set number.
    For sets 1 and 2, players swap sides between sets. The XOR logic:
      - If (first_A_is_top XOR set_num==2): A -> Top, B -> Bottom
      - Otherwise: A -> Bottom, B -> Top

    :param df: DataFrame with a 'player' column containing 'A' or 'B'.
    :param first_A_is_top: From match.csv 'downcourt' flag (True = A starts on top).
    :param set_num: 1 or 2 (for set 3, use find_set3_switch_rally and call twice).
    :return: DataFrame with 'player' column replaced by 'Top'/'Bottom'.
    """
    df = df.copy()
    if first_A_is_top ^ (set_num == 2):
        df['player'] = np.where(df['player'] == 'A', 'Top', 'Bottom')
    else:
        df['player'] = np.where(df['player'] == 'B', 'Top', 'Bottom')
    return df


def find_set3_switch_rally(df: pd.DataFrame) -> int:
    """Find the rally index where the set 3 court switch occurs at 11 points.

    In badminton, players switch sides in set 3 when one player reaches 11
    points. This function finds the first rally where either player's score
    reaches 11, then returns the index of the NEXT rally (the first rally
    after the switch).

    :param df: DataFrame with 'roundscore_A', 'roundscore_B', and 'rally' columns.
    :return: iloc index splitting the DataFrame into pre-switch and post-switch.
    """
    # Find the first index where either player reaches 11 points.
    i_A = df['roundscore_A'].searchsorted(11, side='left')
    i_B = df['roundscore_B'].searchsorted(11, side='left')
    i = min(i_A, i_B)

    # Without this guard, df.iloc[len(df)] raises IndexError on retirements
    if i >= len(df):
        return len(df)

    switch_rally = df.iloc[i]['rally']
    return df['rally'].searchsorted(switch_rally, side='right')


def collect_shots(
    set_info_dir: Path,
    v_info: pd.Series,
    stroke_types_zh: list[str],
) -> pd.DataFrame:
    """Collect all shots for a video across all sets, with Top/Bottom mapping.

    Unlike the original collect_shot_types_pos() in gen_my_dataset.py which
    filters to a single player, this returns shots for BOTH players. The
    caller can filter by player if needed.

    :param set_info_dir: Path to ShuttleSet/set/ containing match folders.
    :param v_info: Series from match.csv with 'video' and 'downcourt' fields.
        The Series name (index) should be the video ID.
    :param stroke_types_zh: List of Chinese stroke type strings to include.
    :return: DataFrame with columns: set, rally, ball_round, frame_num,
        roundscore_A, roundscore_B, player ('Top'/'Bottom'), type (English).
    """
    folder_path = set_info_dir / v_info['video']
    first_A_is_top = bool(v_info['downcourt'])
    collected = []

    # Sets 1 and 2
    for set_i in range(1, 3):
        csv_path = folder_path / f'set{set_i}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df = df[df['type'].isin(stroke_types_zh)]
        df.insert(0, 'set', np.full(len(df), set_i, dtype=int))
        df = map_players(df, first_A_is_top, set_i)
        collected.append(df)

    # Set 3 (if exists): handle 11-point court switch
    csv_path = folder_path / 'set3.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)[_SHOT_COLS]
        df.insert(0, 'set', np.full(len(df), 3, dtype=int))

        i_split = find_set3_switch_rally(df)
        # Before switch: same court sides as set 1
        df_before = map_players(df.iloc[:i_split], first_A_is_top, 1)
        # After switch: sides flipped, same as set 2
        df_after = map_players(df.iloc[i_split:], first_A_is_top, 2)

        df_before = df_before[df_before['type'].isin(stroke_types_zh)]
        df_after = df_after[df_after['type'].isin(stroke_types_zh)]
        collected.extend([df_before, df_after])

    if not collected:
        return pd.DataFrame(columns=['set'] + _SHOT_COLS)

    result = pd.concat(collected).reset_index(drop=True)

    # Translate Chinese type names to English
    result['type'] = result['type'].map(ZH_TO_EN)

    return result
