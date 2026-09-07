"""Diagnostics for missing contacts in selected follow-up spans."""

from collections.abc import Mapping

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.followup_residuals import residual_rows
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "fixture"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(*frames: int, start: int = 0, end: int = 100, span_id: int = 0) -> FixedSpan:
    return FixedSpan(FIXTURE, span_id, start, end, tuple(_event(frame) for frame in frames))


def _rally(*frames: int, rally_id: str = "set1:1") -> RallyReference:
    return RallyReference(FIXTURE, 0, rally_id, tuple(frames))


def _labels(
    *rallies: RallyReference,
    target_sides: Mapping[tuple[str, int], str] | None = None,
) -> HumanLabels:
    sides = (
        {(rally.fixture, frame): "Top" for rally in rallies for frame in rally.frames}
        if target_sides is None
        else dict(target_sides)
    )
    return HumanLabels({FIXTURE: tuple(rallies)}, sides)


def _option(span: FixedSpan) -> LaterOption:
    return LaterOption(CombinedAction("keep", None, None, span), None, span)


def test_reports_missing_first_and_later_saved_candidate_evidence() -> None:
    span = _span(21, 80)
    labels = _labels(_rally(10, 20, 50, 80))
    selected = {(FIXTURE, 0): _option(span)}
    candidates = {(FIXTURE, 0): (_event(10), _event(49), _event(200))}

    row = residual_rows(selected, candidates, labels, {FIXTURE: 30})[0]

    assert row["known"] is True
    assert row["full_contained"] is True
    assert row["timing_complete"] is False
    assert row["missing_first"] == 1
    assert row["missing_later_count"] == 1
    assert row["unmatched_predictions"] == 0
    assert row["unmatched_gt_contacts"] == [
        {
            "gt_index": 0,
            "gt_frame": 10,
            "kind": "first",
            "any_saved_candidate_within_tolerance": True,
            "candidate_frames_within_tolerance": [10],
            "existing_predicted_frame_within_tolerance": False,
        },
        {
            "gt_index": 2,
            "gt_frame": 50,
            "kind": "later",
            "any_saved_candidate_within_tolerance": True,
            "candidate_frames_within_tolerance": [49],
            "existing_predicted_frame_within_tolerance": False,
        },
    ]


def test_reports_full_timing_with_one_wrong_voted_side() -> None:
    labels = _labels(
        _rally(10, 20, 30),
        target_sides={(FIXTURE, 10): "Top", (FIXTURE, 20): "Bot", (FIXTURE, 30): "Bot"},
    )
    wrong_side_span = FixedSpan(
        FIXTURE, 0, 0, 100, (_event(10), _event(20, "Bot"), _event(30, "Bot")),
    )

    row = residual_rows({(FIXTURE, 0): _option(wrong_side_span)}, {}, labels, {FIXTURE: 30})[0]

    assert row["timing_complete"] is True
    assert row["side_rule_fully_correct"] is False
    assert row["wrong_voted_side_count"] == 1


def test_boundary_cut_is_known_but_not_full_contained() -> None:
    span = _span(20, 30, start=20)
    labels = _labels(_rally(10, 20, 30))

    row = residual_rows({(FIXTURE, 0): _option(span)}, {}, labels, {FIXTURE: 30})[0]

    assert row["known"] is True
    assert row["full_contained"] is False
    assert row["boundary_incomplete"] is True
    assert row["missing_first"] == 1


def test_unknown_sections_keep_rally_dependent_fields_none() -> None:
    span = _span(20, 30)
    selected = {(FIXTURE, 0): _option(span)}

    row = residual_rows(selected, {(FIXTURE, 0): (_event(10),)}, _labels(), {FIXTURE: 30})[0]

    assert row["known"] is False
    assert row["no_labels"] is True
    assert row["multiple_rallies"] is False
    for field in (
        "full_contained", "timing_complete", "side_rule_fully_correct",
        "missing_first", "missing_later_count", "unmatched_predictions",
        "wrong_voted_side_count", "unmatched_gt_contacts",
    ):
        assert row[field] is None
