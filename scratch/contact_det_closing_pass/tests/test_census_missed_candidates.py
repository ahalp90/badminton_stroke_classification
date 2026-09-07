from __future__ import annotations

import numpy as np

from scratch.contact_det_closing_pass.scripts.census_missed_candidates import (
    D_FIXTURES,
    SCORE_DTYPE,
    _population_match_count,
    classify_missed_contact,
)


def _feature_dtype() -> np.dtype:
    fields = [
        ("fixture", "S7"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f4"),
    ]
    fields.extend(
        (name, "u1")
        for name in (
            "region_current_raw",
            "region_relaxed_impulse",
            "region_wrist",
            "region_visibility",
            "region_rally_start",
            "region_scene_start",
            "region_serve_lookback",
        )
    )
    return np.dtype(fields)


def _feature_rows(*frames: int) -> np.ndarray:
    rows = np.zeros(len(frames), dtype=_feature_dtype())
    rows["fixture"] = b"sset_01"
    rows["interval_id"] = 2
    rows["frame"] = frames
    rows["fps"] = 25.0
    rows["region_current_raw"] = 1
    return rows


def _score_rows(*scores: tuple[int, float, bool]) -> np.ndarray:
    rows = np.zeros(len(scores), dtype=SCORE_DTYPE)
    rows["fixture"] = b"sset_01"
    rows["interval_id"] = 2
    rows["frame"] = [item[0] for item in scores]
    rows["fps"] = 25.0
    rows["contact_score"] = [item[1] for item in scores]
    rows["kept"] = [item[2] for item in scores]
    return rows


def test_classifies_each_pipeline_stage() -> None:
    cases = (
        (
            "no_nearby_frozen_row",
            np.empty(0, dtype=_feature_dtype()),
            np.empty(0, dtype=_feature_dtype()),
            np.empty(0, dtype=SCORE_DTYPE),
        ),
        (
            "nearby_but_unselected",
            _feature_rows(100),
            np.empty(0, dtype=_feature_dtype()),
            np.empty(0, dtype=SCORE_DTYPE),
        ),
        (
            "unexpected_unscored_selected_row",
            _feature_rows(100),
            _feature_rows(100),
            np.empty(0, dtype=SCORE_DTYPE),
        ),
        (
            "matching_competition",
            _feature_rows(100),
            _feature_rows(100),
            _score_rows((100, 0.95, True)),
        ),
        (
            "suppression",
            _feature_rows(100),
            _feature_rows(100),
            _score_rows((100, 0.95, False)),
        ),
        (
            "below_cutoff",
            _feature_rows(100),
            _feature_rows(100),
            _score_rows((100, 0.4, False)),
        ),
    )
    for expected, frozen, selected, scores in cases:
        result = classify_missed_contact(100, 5, frozen, selected, scores)
        assert result["category"] == expected


def test_matched_count_uses_only_explicit_d_population() -> None:
    matches = {fixture: {1} for fixture in D_FIXTURES}
    matches["sset_18"] = {2, 3}
    assert _population_match_count(matches) == len(D_FIXTURES)
