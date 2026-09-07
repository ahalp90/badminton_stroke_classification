"""Boundary tests for local later-contact insertion targets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.local_insertion import (
    insertion_targets,
    local_insertion_quality,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "fixture"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(*frames: int, start: int = 0, end: int = 200, span_id: int = 0) -> FixedSpan:
    return FixedSpan(
        FIXTURE, span_id, start, end, tuple(_event(frame) for frame in frames)
    )


def _rally(
    *frames: int, rally_id: str = "set1:1", rally_index: int = 0
) -> RallyReference:
    return RallyReference(FIXTURE, rally_index, rally_id, tuple(frames))


def _labels(
    *rallies: RallyReference,
    target_sides: Mapping[tuple[str, int], str | None] | None = None,
) -> HumanLabels:
    sides = (
        {(rally.fixture, frame): "Top" for rally in rallies for frame in rally.frames}
        if target_sides is None
        else dict(target_sides)
    )
    return HumanLabels({FIXTURE: tuple(rallies)}, sides)


def _option(
    before: FixedSpan, candidate_frame: int, *, side: str | None = "Top"
) -> LaterOption:
    inserted = _event(candidate_frame, side)
    after = FixedSpan(
        before.fixture,
        before.span_id,
        before.start_frame,
        before.end_frame,
        tuple(sorted((*before.events, inserted), key=lambda event: event.frame)),
    )
    base = CombinedAction("keep", None, None, before)
    return LaterOption(base, inserted, after)


def test_useful_insertion_is_positive_when_a_later_contact_stays_missing() -> None:
    before = _span(130)
    after = _span(100, 130)

    assert local_insertion_quality(before, after, _rally(100, 130, 160), 100, 10)


def test_duplicate_candidate_that_steals_old_match_is_negative() -> None:
    before = _span(104, 130)
    after = _span(100, 104, 130)

    assert not local_insertion_quality(before, after, _rally(100, 130, 160), 100, 10)


def test_candidate_can_trigger_a_harmless_rematching_chain() -> None:
    before = _span(108)
    after = _span(100, 108)

    assert local_insertion_quality(before, after, _rally(100, 116), 100, 10)


def test_two_candidates_for_one_label_are_scored_independently() -> None:
    before = _span(130)
    rally = _rally(100, 130, 160)
    options = (
        _option(before, 94),
        _option(before, 110),
    )

    targets = insertion_targets(options, (before,), _labels(rally), {FIXTURE: 30.0})

    np.testing.assert_array_equal(targets, np.array([1, 1], dtype=np.int8))


def test_wrong_raw_side_does_not_change_timing_target() -> None:
    before = _span(130)
    option = _option(before, 100, side="Bot")

    targets = insertion_targets(
        (option,),
        (before,),
        _labels(_rally(100, 130)),
        {FIXTURE: 30.0},
    )

    np.testing.assert_array_equal(targets, np.array([1], dtype=np.int8))


def test_timing_target_does_not_require_known_player_sides() -> None:
    before = _span(130)
    option = _option(before, 100)
    rally = _rally(100, 130)

    targets = insertion_targets(
        (option,),
        (before,),
        _labels(rally, target_sides={(FIXTURE, 100): "Top", (FIXTURE, 130): None}),
        {FIXTURE: 30.0},
    )

    np.testing.assert_array_equal(targets, np.array([1], dtype=np.int8))


def test_six_to_ten_frame_candidate_offset_is_positive_at_ten_frame_tolerance() -> None:
    before = _span(130)
    option = _option(before, 110)

    targets = insertion_targets(
        (option,),
        (before,),
        _labels(_rally(100, 130)),
        {FIXTURE: 30.0},
    )

    np.testing.assert_array_equal(targets, np.array([1], dtype=np.int8))


def test_frame_tolerance_scales_for_sixty_fps() -> None:
    before = _span(130)
    option = _option(before, 118)

    targets = insertion_targets(
        (option,),
        (before,),
        _labels(_rally(100, 130)),
        {FIXTURE: 60.0},
    )

    np.testing.assert_array_equal(targets, np.array([1], dtype=np.int8))


def test_no_insertion_ambiguous_and_unlabelled_options_are_excluded() -> None:
    before = _span(130, span_id=0)
    no_insertion = LaterOption(CombinedAction("keep", None, None, before), None, before)
    ambiguous = _option(before, 100)
    absent_before = _span(180, start=170, end=200, span_id=1)
    absent = _option(absent_before, 190)
    labels = _labels(
        _rally(100, 130), _rally(130, 160, rally_id="set1:2", rally_index=1)
    )

    targets = insertion_targets(
        (no_insertion, ambiguous, absent),
        (before, absent_before),
        labels,
        {FIXTURE: 30.0},
    )

    np.testing.assert_array_equal(targets, np.array([-1, -1, -1], dtype=np.int8))
