from pathlib import Path

import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.run_followup_residual_diagnosis import (
    _annotate_coverage,
    _coverage_counts,
    _early_window_rows,
    _EarlyWindow,
    _error_counts,
    _load_early_windows,
    _VideoEvidence,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "known": True,
        "missing_first": 0,
        "missing_later_count": 0,
        "unmatched_predictions": 0,
        "wrong_voted_side_count": 0,
        "boundary_incomplete": False,
        "multiple_rallies": False,
        "no_labels": False,
    }
    row.update(updates)
    return row


def test_error_counts_normalises_integer_flags_and_preserves_unknowns() -> None:
    counts = _error_counts([
        _row(missing_first=1),
        _row(
            known=False,
            missing_first=None,
            missing_later_count=None,
            unmatched_predictions=None,
            wrong_voted_side_count=None,
            no_labels=True,
        ),
    ])

    errors = counts["errors"]
    assert errors["missing_first"] == {"true": 1, "false": 0, "unknown": 1}
    assert errors["missing_exactly_one_later"] == {"true": 0, "false": 1, "unknown": 1}
    assert errors["missing_more_than_one_later"]["unknown"] == 1
    assert errors["extra_unmatched_prediction"]["unknown"] == 1
    assert errors["wrong_voted_sides"]["unknown"] == 1
    assert errors["incomplete_boundary"]["unknown"] == 1
    assert errors["no_labels"]["true"] == 1


def _score_rows(*values: tuple[int, int]) -> np.ndarray:
    rows = np.zeros(len(values), dtype=[("interval_id", "i4"), ("frame", "i4")])
    rows["interval_id"] = [interval for interval, _frame in values]
    rows["frame"] = [frame for _interval, frame in values]
    return rows


def _physical_rows(*frames: int) -> np.ndarray:
    rows = np.zeros(len(frames), dtype=[("frame", "i4")])
    rows["frame"] = frames
    return rows


def _evidence(score_rows: np.ndarray) -> _VideoEvidence:
    score_frames = score_rows["frame"].astype(np.int64)
    physical = _physical_rows()
    return _VideoEvidence(score_rows, score_frames, physical, np.empty(0, dtype=np.int64))


def _contact(gt_frame: int = 100) -> dict[str, object]:
    return {
        "kind": "first",
        "gt_frame": gt_frame,
        "candidate_frames_within_tolerance": [],
        "existing_predicted_frame_within_tolerance": False,
    }


def _coverage_row(contact: dict[str, object]) -> dict[str, object]:
    return {"fixture": "sset_01", "span_id": 7, "unmatched_gt_contacts": [contact]}


def test_early_window_uses_interval_id_and_half_open_edges() -> None:
    windows = (_EarlyWindow(2, 90, 120, 6),)
    scores = _score_rows((2, 89), (2, 90), (2, 119), (2, 120), (3, 100))
    assert _early_window_rows(scores, windows)["frame"].tolist() == [90, 119]

    row = _coverage_row(_contact())
    _annotate_coverage(
        [row], {}, {"sset_01": _evidence(_score_rows((2, 90), (2, 119), (2, 120)))},
        {"sset_01": 20}, {("sset_01", 7): windows},
    )
    assert row["unmatched_gt_contacts"][0]["early_window_category"] == (
        "inside_window_rank_or_suppression"
    )


def test_early_window_reasons_cover_outside_scored_and_missing_windows() -> None:
    cases = (
        ("outside_window", _score_rows((3, 100)), {("sset_01", 7): (_EarlyWindow(2, 90, 120, 6),)}),
        ("no_scored", _score_rows(), {("sset_01", 7): (_EarlyWindow(2, 90, 120, 6),)}),
        ("no_window", _score_rows((2, 100)), {}),
    )
    for expected, scores, windows in cases:
        row = _coverage_row(_contact())
        _annotate_coverage([row], {}, {"sset_01": _evidence(scores)}, {"sset_01": 20}, windows)
        assert row["unmatched_gt_contacts"][0]["early_window_category"] == expected


def test_loader_keeps_fixed_only_section_window(tmp_path: Path) -> None:
    path = tmp_path / "early_inputs.json.gz"
    write_json(path, {
        "schema": "contact-rally-start-training-inputs/1",
        "status": "complete",
        "labels_read": False,
        "videos": [{
            "candidate_lists": [{
                "fixture": "sset_01", "span_id": 7, "interval_id": 4,
                "prefix_start_frame": 80, "fixed_contact_frame": 120,
                "duplicate_distance_frames": 6, "candidates": [],
            }],
        }],
    })
    windows = _load_early_windows(path, {"sset_01"})
    assert windows[("sset_01", 7)][0].interval_id == 4
    assert windows[("sset_01", 7)][0].prefix_start_frame == 80


def test_shortlisted_contact_is_not_counted_as_a_window_or_ranking_miss() -> None:
    contact = _contact()
    contact["candidate_frames_within_tolerance"] = [100]
    row = _coverage_row(contact)
    _annotate_coverage(
        [row], {}, {"sset_01": _evidence(_score_rows((2, 100)))},
        {"sset_01": 10}, {("sset_01", 7): (_EarlyWindow(2, 90, 120, 6),)},
    )
    assert contact["coverage_category"] == "shortlisted_not_chosen"
    assert contact["early_window_category"] is None


def test_coverage_counts_diagnostic_reasons_without_changing_old_buckets() -> None:
    rows = [_coverage_row(_contact())]
    rows[0]["unmatched_gt_contacts"][0]["early_window_category"] = "outside_window"
    counts = _coverage_counts(rows)
    assert counts["first"] == {"covered": 0, "missed": 1}
    assert counts["later"] == {"covered": 0, "missed": 0}
    assert counts["early_window_category_counts"]["outside_window"] == 1
