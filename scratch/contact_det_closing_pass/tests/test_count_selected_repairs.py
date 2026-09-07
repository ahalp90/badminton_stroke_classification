"""Small checks for the selected-repair headroom helpers."""

from __future__ import annotations

from types import SimpleNamespace

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.count_selected_repairs import (
    _fixture,
    _unique_options,
    edit_category,
    evaluate_option_with_padding,
    frame_delta,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "synthetic"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(frames: tuple[int, ...], start: int = 100, end: int = 201) -> FixedSpan:
    events = tuple(
        _event(frame, ("Top", "Bot")[index % 2]) for index, frame in enumerate(frames)
    )
    return FixedSpan(FIXTURE, 0, start, end, events)


def _option(span: FixedSpan) -> LaterOption:
    base = CombinedAction("keep", None, None, span)
    return LaterOption(base, None, span)


def _labels(frames: tuple[int, ...]) -> HumanLabels:
    rally = RallyReference(FIXTURE, 0, "rally", frames)
    sides = {
        (FIXTURE, frame): ("Top", "Bot")[index % 2]
        for index, frame in enumerate(frames)
    }
    return HumanLabels({FIXTURE: (rally,)}, sides)


def test_frame_deltas_and_categories_cover_the_small_edit_pool() -> None:
    assert frame_delta((100, 150, 200), (150, 200)) == ((100,), ())
    assert frame_delta((100, 150, 200), (100, 150, 200, 250)) == ((), (250,))
    assert frame_delta((100, 150, 200), (100, 150, 250)) == ((200,), (250,))

    labels = (100, 150, 200)
    assert (
        edit_category((90,), (), labels, 10, (90, 150, 200))
        == "deletion_before_first_label"
    )
    assert (
        edit_category((220,), (), labels, 10, (100, 150, 220))
        == "deletion_after_last_label"
    )
    assert edit_category((150,), (), labels, 10, (100, 200)) == "deletion_interior"
    assert (
        edit_category((), (90,), labels, 10, (90, 100, 150, 200)) == "insertion_first"
    )
    assert (
        edit_category((), (175,), labels, 10, (100, 150, 175, 200)) == "insertion_later"
    )
    assert (
        edit_category((145,), (155,), labels, 10, (100, 155, 200))
        == "replacement_timing"
    )
    assert (
        edit_category((120,), (180,), labels, 10, (100, 180, 200))
        == "replacement_extra_plus_missing"
    )


def test_equivalent_resulting_spans_are_evaluated_once() -> None:
    current = _option(_span((100, 200)))
    duplicate = _option(_span((100, 150, 200)))
    assert _unique_options((current, duplicate, duplicate), current) == [
        (duplicate, (), (150,), "insertion")
    ]


def test_full_stream_event_preserves_membership_during_padding() -> None:
    current_span = _span((100, 150, 200))
    current = _option(current_span)
    outside = _event(205, None)
    labels = _labels((95, 150, 200))

    raw, padded, result, _secondary = evaluate_option_with_padding(
        current,
        (current_span,),
        (*current_span.events, outside),
        {(FIXTURE, 0): current},
        labels,
        30.0,
    )

    assert raw.events == current_span.events
    assert padded.events == current_span.events
    assert (padded.start_frame, padded.end_frame) == (100, 201)
    assert not result["side_rule_fully_correct"]


def test_padding_can_admit_a_label_outside_the_raw_alternative() -> None:
    current_span = _span((120,), start=90, end=121)
    current = _option(current_span)
    alternative = _option(_span((100,), start=100, end=101))
    identity = (FIXTURE, 0)
    population = SimpleNamespace(
        fps={FIXTURE: 30.0}, groups={FIXTURE: "A"},
        spans=(current_span,), events={FIXTURE: current_span.events},
    )
    rows = _fixture(
        FIXTURE, population, {identity: current}, {identity: [alternative]},
        [{"span_id": 0, "judgements": {"10": {"outcome": "wrong"}}}], _labels((95,)),
    )
    assert rows[0]["repair_types"] == ["replacement"]
    assert rows[0]["examples"][0]["result_10"]["side_rule_fully_correct"]
