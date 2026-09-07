from __future__ import annotations

from copy import deepcopy

import numpy as np

from scratch.contact_det_closing_pass.scripts.prepare_early_broader_inputs import (
    _expand_video,
)


def test_expand_video_replays_sides_only_for_new_candidates() -> None:
    video = {
        "video": {"fixture": "8", "video_id": 8, "fps": 30.0},
        "spans": [{"span_id": 0, "start_frame": 90, "end_frame": 120}],
        "candidate_lists": [{
            "fixture": "8",
            "span_id": 0,
            "section_start_frame": 90,
            "section_end_frame": 120,
            "interval_id": 0,
            "prefix_start_frame": 50,
            "fixed_contact_frame": 100,
            "duplicate_distance_frames": 6,
            "candidates": [
                {"frame": 100, "contact_score": 0.9, "is_fixed_contact": True,
                 "kept": True, "predicted_side": "Bot"},
                {"frame": 90, "contact_score": 0.8, "is_fixed_contact": False,
                 "kept": False, "predicted_side": "Top"},
                {"frame": 80, "contact_score": 0.7, "is_fixed_contact": False,
                 "kept": False, "predicted_side": None},
            ],
        }],
        "counts": {"candidate_entries": 3, "earlier_candidate_entries": 2,
                   "distinct_replayed_frames": 3},
    }
    original = deepcopy(video)
    rows = np.array([
        (0, 70, 0.6, False),
        (0, 60, 0.5, True),
        (0, 50, 0.4, False),
    ], dtype=[("interval_id", "i4"), ("frame", "i4"),
              ("contact_score", "f8"), ("kept", "?")])
    replayed: list[int] = []

    def load_sides(frames: list[int]) -> dict[int, str]:
        replayed.extend(frames)
        return {frame: "Top" for frame in frames}

    expanded, counts = _expand_video(video, rows, load_sides)

    candidates = expanded["candidate_lists"][0]["candidates"]
    assert [candidate["frame"] for candidate in candidates] == [100, 90, 80, 70, 60]
    assert [candidate["predicted_side"] for candidate in candidates[-2:]] == ["Top", "Top"]
    assert [candidate["kept"] for candidate in candidates[-2:]] == [False, True]
    assert replayed == [60, 70]
    assert counts["added_earlier_candidates"] == 2
    assert counts["candidate_entries_before"] == 3
    assert counts["candidate_entries_after"] == 5
    assert video == original
