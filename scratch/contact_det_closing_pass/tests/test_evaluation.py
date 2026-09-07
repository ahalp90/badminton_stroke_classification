"""Boundary tests for the closing-pass timing and section evaluation."""

from __future__ import annotations

from collections.abc import Mapping

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts import evaluation
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "sset_01"


def _rally(
    *frames: int,
    rally_id: str = "set1:1",
    rally_index: int = 0,
) -> RallyReference:
    return RallyReference(FIXTURE, rally_index, rally_id, tuple(frames))


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(
    *events: FixedEvent,
    start: int = 0,
    end: int = 100,
    span_id: int = 0,
) -> FixedSpan:
    return FixedSpan(FIXTURE, span_id, start, end, tuple(events))


def _labels(
    *rallies: RallyReference,
    target_sides: Mapping[tuple[str, int], str] | None = None,
) -> HumanLabels:
    sides = (
        {
            (rally.fixture, frame): "Top"
            for rally in rallies
            for frame in rally.frames
        }
        if target_sides is None
        else dict(target_sides)
    )
    return HumanLabels(
        {FIXTURE: tuple(rallies)},
        sides,
    )


def test_section_result_accepts_one_complete_rally() -> None:
    rally = _rally(10, 20)
    labels = _labels(rally, target_sides={(FIXTURE, 10): "Top", (FIXTURE, 20): "Top"})

    result = evaluation.section_result(
        _span(_event(10), _event(20), start=10, end=30),
        labels,
        tolerance=0,
    )

    assert result["rally_id"] == "set1:1"
    assert result["overlapping_rallies"] == 1
    assert result["whole_rally_contained"] is True
    assert result["matches"] == [(0, 0, 0), (1, 1, 0)]
    assert result["timing_complete"] is True
    assert result["fully_correct"] is True


def test_section_end_is_excluded_from_rally_overlap_and_containment() -> None:
    rally = _rally(10, 20)
    labels = _labels(rally)

    result = evaluation.section_result(
        _span(_event(10), start=10, end=20),
        labels,
        tolerance=10,
    )

    assert result["overlapping_rallies"] == 1
    assert result["whole_rally_contained"] is False
    assert result["timing_complete"] is False
    assert result["fully_correct"] is False

    end_only = evaluation.section_result(
        _span(_event(20), start=10, end=20, span_id=1),
        _labels(_rally(20)),
        tolerance=0,
    )
    assert end_only["overlapping_rallies"] == 0
    assert end_only["rally_id"] is None


def test_contact_just_before_section_start_does_not_complete_rally() -> None:
    rally = _rally(9, 20)
    labels = _labels(rally)

    result = evaluation.section_result(
        _span(_event(10), _event(20), start=10, end=30),
        labels,
        tolerance=2,
    )

    assert result["matches"] == [(0, 0, 1), (1, 1, 0)]
    assert result["whole_rally_contained"] is False
    assert result["timing_complete"] is False
    assert result["fully_correct"] is False


def test_section_with_parts_of_two_rallies_is_rejected() -> None:
    first = _rally(10, 20, rally_id="set1:1")
    second = _rally(30, 40, rally_id="set1:2", rally_index=1)
    labels = _labels(first, second)

    result = evaluation.section_result(
        _span(_event(10), _event(20), _event(30), _event(40), end=50),
        labels,
        tolerance=0,
    )

    assert result["overlapping_rallies"] == 2
    assert result["rally_id"] is None
    assert result["matches"] == []
    assert result["timing_complete"] is False
    assert result["fully_correct"] is False


def test_section_result_compares_raw_sides_with_fixed_side_vote() -> None:
    rally = _rally(10, 20, 30, 40)
    target_sides = {
        (FIXTURE, 10): "Top",
        (FIXTURE, 20): "Bot",
        (FIXTURE, 30): "Top",
        (FIXTURE, 40): "Bot",
    }
    labels = _labels(rally, target_sides=target_sides)

    result = evaluation.section_result(
        _span(_event(10, "Top"), _event(20, "Top"), _event(30, "Top"), _event(40, "Bot")),
        labels,
        tolerance=0,
    )

    assert result["correct_sides"] == 3
    assert result["fully_correct"] is False
    assert result["side_rule_fully_correct"] is True


def test_score_contacts_scales_tolerance_for_a_60_fps_stream() -> None:
    rally = _rally(100, 200)
    labels = _labels(rally, target_sides={(FIXTURE, 100): "Top", (FIXTURE, 200): "Top"})

    result = evaluation.score_contacts(
        {FIXTURE: (_event(120), _event(220))},
        labels,
        {FIXTURE: 60.0},
        tolerance_base30=10,
    )

    assert result["total"]["labelled"] == 2
    assert result["total"]["predicted"] == 2
    assert result["total"]["matched"] == 2
    assert result["total"]["precision"] == 1.0
    assert result["total"]["recall"] == 1.0
    assert [pair[4] for pair in result["by_video"][0]["pairs"]] == [20, 20]


def test_score_contacts_counts_predictions_outside_gt_ranges() -> None:
    rally = _rally(100, 200)
    labels = _labels(rally)

    result = evaluation.score_contacts(
        {FIXTURE: (_event(50), _event(100), _event(200), _event(250))},
        labels,
        {FIXTURE: 30.0},
        tolerance_base30=0,
    )

    counts = result["by_video"][0]
    assert counts["labelled"] == 2
    assert counts["predicted"] == 4
    assert counts["matched"] == 2
    assert result["total"]["precision"] == 0.5


def test_score_contacts_keeps_a_fixture_with_no_labels_in_totals() -> None:
    labels = _labels()

    result = evaluation.score_contacts(
        {FIXTURE: (_event(100),)},
        labels,
        {FIXTURE: 30.0},
        tolerance_base30=10,
    )

    assert result["total"]["labelled"] == 0
    assert result["total"]["predicted"] == 1
    assert result["total"]["matched"] == 0
    assert result["total"]["precision"] == 0.0
    assert result["total"]["recall"] == 0.0
    assert result["total"]["f1"] == 0.0


def test_paired_sections_uses_gt_identity_across_reidentification_and_loss() -> None:
    before = (
        {"fixture": FIXTURE, "span_id": 0, "rally_id": "set1:1", "fully_correct": True},
        {"fixture": FIXTURE, "span_id": 1, "rally_id": "set1:2", "fully_correct": True},
    )
    after = (
        {"fixture": FIXTURE, "span_id": 7, "rally_id": "set1:1", "fully_correct": True},
    )

    result = evaluation.paired_sections(before, after)

    assert result["sections_before"] == 2
    assert result["sections_after"] == 1
    assert result["correct_before"] == 2
    assert result["correct_after"] == 1
    assert result["correct_sections_before"] == 2
    assert result["correct_sections_after"] == 1
    assert result["repaired"] == []
    assert result["lost"] == [(FIXTURE, "set1:2")]
