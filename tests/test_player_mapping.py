"""Tests for `shared.player_mapping`.

Three layers:

1. ``map_players`` — XOR logic for the 8 (first_A_is_top, set_num, player)
   combinations. Pure unit tests on synthetic data.
2. ``find_set3_switch_rally`` — split-point detection on synthetic set-3
   score progressions, including the retirement edge case.
3. ``collect_shots`` — integration tests against real ShuttleSet data.
   Verifies specific shots from a curated ground-truth list resolve
   to the expected ``player`` value. The list covers both ``downcourt``
   values and the set-3 11-point switch (pre + post sides).

The expected values in layer 3 were derived once from BST's authoritative
``collect_shots`` output (notebook 03's clips_master.csv) and committed
inline — the test has no runtime dependency on that file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from shared.dataset import SET_INFO_DIR
from shared.player_mapping import (
    collect_shots,
    find_set3_switch_rally,
    map_players,
)
from shared.taxonomy import STROKE_TYPES_19_ZH


# ---------------------------------------------------------------------------
# Layer 1 — map_players XOR logic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('first_A_is_top, set_num, raw_player, expected', [
    # downcourt=True (A starts on top)
    (True,  1, 'A', 'Top'),     # set 1: A on top
    (True,  1, 'B', 'Bottom'),
    (True,  2, 'A', 'Bottom'),  # set 2: sides swap
    (True,  2, 'B', 'Top'),
    # downcourt=False (B starts on top)
    (False, 1, 'A', 'Bottom'),  # set 1: A on bottom
    (False, 1, 'B', 'Top'),
    (False, 2, 'A', 'Top'),     # set 2: sides swap
    (False, 2, 'B', 'Bottom'),
])
def test_map_players_xor(first_A_is_top, set_num, raw_player, expected):
    df = pd.DataFrame({'player': [raw_player]})
    out = map_players(df, first_A_is_top=first_A_is_top, set_num=set_num)
    assert out.iloc[0]['player'] == expected


# ---------------------------------------------------------------------------
# Layer 2 — find_set3_switch_rally
# ---------------------------------------------------------------------------
def test_find_set3_switch_rally_A_reaches_11_first():
    # Players switch sides when the FIRST of either reaches 11.
    # A hits 11 at rally 3 → switch happens after rally 3, so the
    # post-switch chunk begins at rally 4 (iloc 3 since 0-indexed).
    df = pd.DataFrame({
        'rally':        [1, 2, 3, 4, 5],
        'roundscore_A': [9, 10, 11, 11, 11],
        'roundscore_B': [9, 10, 10, 11, 11],
    })
    assert find_set3_switch_rally(df) == 3


def test_find_set3_switch_rally_B_reaches_11_first():
    df = pd.DataFrame({
        'rally':        [1, 2, 3, 4, 5],
        'roundscore_A': [8,  9, 10, 11, 11],
        'roundscore_B': [9, 10, 11, 11, 11],
    })
    assert find_set3_switch_rally(df) == 3


def test_find_set3_switch_rally_retirement_no_one_hits_11():
    # Retirement / abandoned set: nobody reaches 11. Returning len(df)
    # makes the post-switch slice empty rather than raising IndexError.
    df = pd.DataFrame({
        'rally':        [1, 2, 3],
        'roundscore_A': [3, 5, 8],
        'roundscore_B': [4, 6, 9],
    })
    assert find_set3_switch_rally(df) == len(df)


# ---------------------------------------------------------------------------
# Layer 3 — collect_shots integration on real ShuttleSet data
# ---------------------------------------------------------------------------
# Hardcoded ground truth: (vid, set, rally, ball_round, expected_player).
# Derived once from BST's authoritative collect_shots output and
# committed inline so the test is self-contained — no runtime CSV
# dependency beyond the ShuttleSet annotations themselves.
#
# Coverage:
#   vid 1 (downcourt=False, has set 3):
#       sets 1, 2 first shots (XOR sanity), set 3 first/mid/last
#       (exercises pre + post 11-point switch via map_players)
#   vid 2 (downcourt=True, no set 3):
#       sets 1, 2 first shots (other downcourt branch)
#   vid 5 (downcourt=False, has set 3):
#       full set 1/2/3 coverage on a different match
#   vid 6 (downcourt=True, has set 3):
#       full set 1/2/3 coverage on the other downcourt branch
EXPECTED_PLAYERS: list[tuple[int, int, int, int, str]] = [
    # vid=1, downcourt=False
    (1, 1,  1,  1, 'Bottom'),   # set 1 first shot
    (1, 2,  1,  1, 'Top'),      # set 2 first shot (sides swap)
    (1, 3,  1,  1, 'Top'),      # set 3 first shot (pre-switch)
    (1, 3, 21, 10, 'Bottom'),   # set 3 mid-set
    (1, 3, 39, 13, 'Bottom'),   # set 3 final shot (post-switch)
    # vid=2, downcourt=True
    (2, 1,  1,  1, 'Top'),      # set 1 first shot
    (2, 2,  1,  1, 'Bottom'),   # set 2 first shot (sides swap)
    # vid=5, downcourt=False
    (5, 1,  1,  1, 'Bottom'),
    (5, 2,  1,  1, 'Top'),
    (5, 3,  1,  1, 'Top'),      # set 3 pre-switch
    (5, 3, 15,  9, 'Bottom'),
    (5, 3, 37, 21, 'Bottom'),   # post-switch
    # vid=6, downcourt=True
    (6, 1,  1,  1, 'Top'),
    (6, 2,  1,  1, 'Bottom'),
    (6, 3,  1,  1, 'Bottom'),
    (6, 3, 19,  1, 'Bottom'),
    (6, 3, 36,  6, 'Top'),
]


@pytest.fixture(scope='module')
def match_df():
    return pd.read_csv(SET_INFO_DIR / 'match.csv').set_index('id')


@pytest.fixture(scope='module')
def shots_by_vid(match_df) -> dict[int, pd.DataFrame]:
    """Run collect_shots once per vid we test against; cache the result."""
    vids = sorted({t[0] for t in EXPECTED_PLAYERS})
    return {
        vid: collect_shots(SET_INFO_DIR, match_df.loc[vid], STROKE_TYPES_19_ZH)
        for vid in vids
    }


@pytest.mark.parametrize(
    'vid, set_n, rally, ball_round, expected', EXPECTED_PLAYERS
)
def test_collect_shots_player_assignment(
    vid, set_n, rally, ball_round, expected, shots_by_vid,
):
    shots = shots_by_vid[vid]
    matches = shots[
        (shots['set'] == set_n)
        & (shots['rally'] == rally)
        & (shots['ball_round'] == ball_round)
    ]
    assert len(matches) == 1, (
        f'vid={vid} set={set_n} rally={rally} ball_round={ball_round}: '
        f'expected exactly one matching shot, got {len(matches)}'
    )
    assert matches.iloc[0]['player'] == expected
