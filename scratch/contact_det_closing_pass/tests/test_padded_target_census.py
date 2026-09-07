"""Tests for target scoring through the complete fixed-membership replay."""

from __future__ import annotations

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import section_result
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.run_padded_target_census import (
    _evaluate_fixture,
    evaluate_option_with_padding,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "synthetic"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(*frames: int, start: int, end: int) -> FixedSpan:
    sides = ("Top", "Bot")
    events = tuple(_event(frame, sides[index % 2]) for index, frame in enumerate(frames))
    return FixedSpan(FIXTURE, 0, start, end, events)


def _option(span: FixedSpan) -> LaterOption:
    base = CombinedAction("keep", None, None, span)
    return LaterOption(base, None, span)


def _labels(*rallies: tuple[str, tuple[int, ...]]) -> HumanLabels:
    references = tuple(
        RallyReference(FIXTURE, index, rally_id, frames)
        for index, (rally_id, frames) in enumerate(rallies)
    )
    sides = {
        (FIXTURE, frame): ("Top", "Bot")[index % 2]
        for _rally_id, frames in rallies
        for index, frame in enumerate(frames)
    }
    return HumanLabels({FIXTURE: references}, sides)


def test_tight_boundary_option_becomes_positive_without_changing_contacts() -> None:
    span = _span(105, 150, 195, start=105, end=196)
    option = _option(span)
    labels = _labels(("main", (100, 150, 200)))

    old_result = section_result(span, labels, tolerance=10)
    raw, padded, result, _secondary = evaluate_option_with_padding(
        option, (span,), span.events, {(FIXTURE, 0): option}, labels, 30.0,
    )

    assert not old_result["side_rule_fully_correct"]
    assert (padded.start_frame, padded.end_frame) == (95, 206)
    assert padded.events == raw.events == span.events
    assert result["side_rule_fully_correct"]


def test_outside_event_blocks_padding_that_an_isolated_option_receives() -> None:
    span = _span(100, 150, 200, start=100, end=201)
    option = _option(span)
    labels = _labels(("main", (95, 150, 200)))
    outside = _event(205, None)

    _raw, blocked, blocked_result, _secondary = evaluate_option_with_padding(
        option, (span,), (*span.events, outside), {(FIXTURE, 0): option}, labels, 30.0,
    )
    _raw, isolated, isolated_result, _secondary = evaluate_option_with_padding(
        option, (span,), span.events, {(FIXTURE, 0): option}, labels, 30.0,
    )

    assert (blocked.start_frame, blocked.end_frame) == (100, 201)
    assert not blocked_result["side_rule_fully_correct"]
    assert (isolated.start_frame, isolated.end_frame) == (90, 211)
    assert isolated_result["side_rule_fully_correct"]


def test_unknown_saved_target_remains_unknown_without_reconstruction() -> None:
    span = _span(105, 150, 195, start=105, end=196)
    option = _option(span)
    labels = _labels(("main", (100, 150, 200)))

    result = _evaluate_fixture(
        FIXTURE,
        "A",
        (span,),
        span.events,
        30.0,
        ((17, option, -1),),
        {(FIXTURE, 0): option},
        labels,
        frozenset(),
        frozenset(),
    )

    assert result.updates == ()
    assert result.changes == ()
    assert result.stats["kind=keep;later_insertion=false"]["old_target_minus_one"] == 1
    assert result.stats["kind=keep;later_insertion=false"]["reconstruction_calls"] == 0


def test_padding_can_make_an_old_positive_lose_to_a_neighbouring_labelled_rally() -> None:
    span = _span(100, 150, 200, start=100, end=201)
    option = _option(span)
    labels = _labels(("main", (100, 150, 200)), ("neighbour", (210,)))

    _raw, padded, result, _secondary = evaluate_option_with_padding(
        option, (span,), span.events, {(FIXTURE, 0): option}, labels, 30.0,
    )

    assert (padded.start_frame, padded.end_frame) == (90, 211)
    assert result["overlapping_rallies"] == 2
    assert not result["side_rule_fully_correct"]
