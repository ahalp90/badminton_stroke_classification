from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scratch.contact_det_closing_pass.scripts.early_shortlist import (
    expand_early_shortlist,
)


def _video(
    *, fps: float = 25.0, duplicate_distance: int = 6,
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "video": {"fixture": "sset_01", "fps": fps},
        "metadata": {"mutable": [1, 2, 3]},
        "candidate_lists": [{
            "fixture": "sset_01",
            "span_id": 4,
            "interval_id": 1,
            "prefix_start_frame": 50,
            "fixed_contact_frame": 100,
            "duplicate_distance_frames": duplicate_distance,
            "candidates": candidates if candidates is not None else [
                {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
                 "kept": True, "predicted_side": "Top"},
                {"frame": 80, "contact_score": 0.8, "is_fixed_contact": False,
                 "kept": False, "predicted_side": "Bot"},
                {"frame": 70, "contact_score": 0.7, "is_fixed_contact": False,
                 "kept": False, "predicted_side": "Top"},
            ],
        }],
    }


def _score_rows(*values: tuple[int, int, float, bool]) -> np.ndarray:
    rows = np.zeros(
        len(values),
        dtype=[
            ("interval_id", "i4"),
            ("frame", "i4"),
            ("contact_score", "f8"),
            ("kept", "?") ,
        ],
    )
    rows["interval_id"] = [interval for interval, _frame, _score, _kept in values]
    rows["frame"] = [frame for _interval, frame, _score, _kept in values]
    rows["contact_score"] = [score for _interval, _frame, score, _kept in values]
    rows["kept"] = [kept for _interval, _frame, _score, kept in values]
    return rows


def test_existing_candidates_are_unchanged_and_expansion_stops_at_four() -> None:
    video = _video()
    original = deepcopy(video)
    scores = _score_rows(
        (1, 60, 0.6, False), (1, 50, 0.5, True), (1, 40, 0.4, False),
        (1, 30, 0.3, False),
    )
    expanded, counts = expand_early_shortlist(
        video, scores, {50: "Bot", 60: "Top", 40: None, 30: "Bot"},
    )
    candidates = expanded["candidate_lists"][0]["candidates"]
    assert [candidate["frame"] for candidate in candidates] == [100, 80, 70, 60, 50]
    assert candidates[:3] == original["candidate_lists"][0]["candidates"][:3]
    assert counts == {
        "candidate_lists": 1,
        "sections_with_additions": 1,
        "added_earlier_candidates": 2,
    }
    assert video == original


def test_interval_and_prefix_fixed_window_excludes_other_scored_frames() -> None:
    video = _video(candidates=[
        {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
         "kept": True, "predicted_side": "Top"},
    ])
    scores = _score_rows(
        (0, 90, 0.99, False), (1, 49, 0.98, False), (1, 60, 0.8, False),
        (1, 100, 0.97, True), (1, 101, 0.96, False),
    )
    expanded, _counts = expand_early_shortlist(
        video, scores, {60: "Bot"},
    )
    candidates = expanded["candidate_lists"][0]["candidates"]
    assert [candidate["frame"] for candidate in candidates] == [100, 60]


def test_saved_distance_is_used_without_fps_rescaling() -> None:
    video = _video(fps=60.0, duplicate_distance=6, candidates=[
        {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
         "kept": True, "predicted_side": "Top"},
    ])
    scores = _score_rows((1, 94, 0.9, False), (1, 93, 0.8, False))
    expanded, _counts = expand_early_shortlist(
        video, scores, {94: "Bot", 93: None},
    )
    assert [candidate["frame"] for candidate in expanded["candidate_lists"][0]["candidates"]] == [100, 93]


def test_new_candidate_copies_kept_and_automatic_side_fields() -> None:
    video = _video(candidates=[
        {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
         "kept": True, "predicted_side": "Top"},
    ])
    original = deepcopy(video)
    expanded, _counts = expand_early_shortlist(
        video, _score_rows((1, 60, 0.8, False)), {60: None},
    )
    candidate = expanded["candidate_lists"][0]["candidates"][1]
    assert candidate == {
        "frame": 60,
        "contact_score": 0.8,
        "is_fixed_contact": False,
        "kept": False,
        "predicted_side": None,
    }
    assert video == original


def test_missing_side_replay_for_new_frame_fails_loudly() -> None:
    with pytest.raises(KeyError, match="automatic side replay is missing"):
        expand_early_shortlist(
            _video(candidates=[
                {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
                 "kept": True, "predicted_side": "Top"},
            ]),
            _score_rows((1, 60, 0.8, False)),
            {},
        )
