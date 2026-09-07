"""Tests for label-free contact-section boundary padding."""

from __future__ import annotations

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts import boundary_followup

FIXTURE = "fixture"


def _event(frame: int, score: float = 0.8, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, score, side)


def _span(*frames: int, start: int, end: int, span_id: int) -> FixedSpan:
    return FixedSpan(FIXTURE, span_id, start, end, tuple(_event(frame) for frame in frames))


def test_first_contact_at_original_start_gets_a_margin() -> None:
    span = _span(50, 70, start=50, end=80, span_id=0)

    padded = boundary_followup.pad_contact_boundaries(
        (span,), {FIXTURE: span.events}, {FIXTURE: 30.0}
    )

    assert (padded.spans[0].start_frame, padded.spans[0].end_frame) == (40, 81)


def test_padding_scales_once_per_fixture(monkeypatch) -> None:
    calls: list[tuple[int, float]] = []

    def fake_scale(base30: int, fps: float) -> int:
        calls.append((base30, fps))
        return 20

    monkeypatch.setattr(boundary_followup, "scale_base30_frames", fake_scale)
    first = _span(20, start=20, end=30, span_id=0)
    second = _span(50, start=50, end=60, span_id=1)

    padded = boundary_followup.pad_contact_boundaries(
        (first, second), {FIXTURE: first.events + second.events}, {FIXTURE: 60.0}
    )

    assert calls == [(10, 60.0)]
    assert padded.spans[0].start_frame == 0


def test_conflicting_extensions_split_the_original_gap() -> None:
    first = _span(119, start=100, end=120, span_id=0)
    second = _span(130, start=130, end=150, span_id=1)
    events = {FIXTURE: first.events + second.events}

    padded = boundary_followup.pad_contact_boundaries(
        (first, second), events, {FIXTURE: 30.0}
    )

    assert padded.spans[0].end_frame == 125
    assert padded.spans[1].start_frame == 125
    assert padded.spans[0].end_frame <= padded.spans[1].start_frame


def test_newly_included_out_of_section_events_are_not_lost() -> None:
    span = _span(50, 79, start=50, end=80, span_id=0)
    outside_before = _event(45, score=0.7, side="Bot")
    outside_after = _event(85, score=0.6, side=None)
    events = {FIXTURE: (outside_before, *span.events, outside_after)}

    padded = boundary_followup.pad_contact_boundaries(
        (span,), events, {FIXTURE: 30.0}
    )

    assert tuple(event.frame for event in padded.spans[0].events) == (45, 50, 79, 85)
    assert padded.events_by_fixture[FIXTURE] == events[FIXTURE]
    assert padded.spans[0].events[0].predicted_side == "Bot"
    assert padded.spans[0].events[-1].predicted_side is None


def test_empty_sections_keep_edges_and_inputs_remain_unchanged() -> None:
    empty = _span(start=50, end=60, span_id=0)
    original_spans = (empty,)
    original_events = {FIXTURE: ()}

    padded = boundary_followup.pad_contact_boundaries(
        original_spans, original_events, {FIXTURE: 30.0}
    )

    assert (padded.spans[0].start_frame, padded.spans[0].end_frame) == (50, 60)
    assert padded.spans[0].events == ()
    assert original_spans == (empty,)
    assert original_events == {FIXTURE: ()}
    assert np.array_equal(np.asarray([empty.start_frame, empty.end_frame]), np.asarray([50, 60]))


def test_fixed_membership_rejects_new_events_but_still_pads_other_sections() -> None:
    conservative = _span(50, 79, start=50, end=80, span_id=0)
    ordinary = _span(100, 120, start=100, end=130, span_id=1)
    outside = _event(45, score=0.7, side="Bot")
    events = {FIXTURE: (outside, *conservative.events, *ordinary.events)}

    padded = boundary_followup.pad_contact_boundaries(
        (conservative, ordinary), events, {FIXTURE: 30.0}, preserve_membership=True
    )

    assert padded.spans[0] == conservative
    assert padded.spans[1].start_frame == 90
    assert padded.events_by_fixture[FIXTURE] == events[FIXTURE]
