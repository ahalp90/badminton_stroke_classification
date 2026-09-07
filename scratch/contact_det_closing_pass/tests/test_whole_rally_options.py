"""Boundary tests for the bounded whole-rally option helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import (
    build_options,
    choose_options,
    option_identity,
    whole_targets,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "sset_01"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(*frames: int, start: int = 0, end: int = 100, span_id: int = 0) -> FixedSpan:
    return FixedSpan(FIXTURE, span_id, start, end, tuple(_event(frame) for frame in frames))


def _rally(*frames: int, rally_id: str = "set1:1", rally_index: int = 0) -> RallyReference:
    return RallyReference(FIXTURE, rally_index, rally_id, tuple(frames))


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
    return HumanLabels({FIXTURE: tuple(rallies)}, sides)


def _candidate_list(span: FixedSpan, frame: int, side: str = "Top") -> dict[str, object]:
    return {
        "fixture": span.fixture,
        "span_id": span.span_id,
        "section_start_frame": span.start_frame,
        "section_end_frame": span.end_frame,
        "fixed_contact_frame": span.events[0].frame,
        "candidates": [
            {"is_fixed_contact": True, "frame": span.events[0].frame, "contact_score": 0.9},
            {"is_fixed_contact": False, "frame": frame, "contact_score": 0.9, "predicted_side": side},
        ],
    }


def test_build_options_preserves_prefix_and_reuses_predecessor_bound() -> None:
    first = _span(20, 30, start=20, end=40, span_id=0)
    second = _span(50, start=50, end=70, span_id=1)
    raw_videos = [
        {
            "fixture": FIXTURE,
            "candidate_lists": [
                _candidate_list(first, 10),
                _candidate_list(second, 35),
            ],
        }
    ]
    events = {FIXTURE: (_event(10), _event(20), _event(30), _event(50))}

    options = build_options((first, second), raw_videos, events)

    first_add = next(option for option in options[(FIXTURE, 0)] if option.kind == "add")
    assert tuple(event.frame for event in first_add.span.events) == (10, 20, 30)
    assert all(option.kind not in {"add", "replace", "add_delete", "replace_delete"} for option in options[(FIXTURE, 1)])


def test_whole_targets_allow_keep_and_combined_add_delete_to_be_positive() -> None:
    span = _span(20, 30, start=20, end=40)
    options_by_section = build_options(
        (span,),
        [{"candidate_lists": [_candidate_list(span, 10, side="Top")] }],
        {FIXTURE: (_event(20, "Bot"), _event(30, "Top"))},
    )
    options = options_by_section[(FIXTURE, 0)]
    labels = _labels(
        _rally(10, 20),
        target_sides={(FIXTURE, 10): "Top", (FIXTURE, 20): "Bot"},
    )

    targets, report = whole_targets(options, (span,), labels, {FIXTURE: 30.0})

    by_kind = {option.kind: int(target) for option, target in zip(options, targets, strict=True)}
    assert by_kind["add"] == 0
    assert by_kind["delete"] == 0
    assert by_kind["add_delete"] == 1
    assert report["positive_counts"]["add_delete"] == 1
    assert all(
        report["positive_counts"][kind] == 0
        for kind in ("keep", "add", "replace", "delete", "replace_delete")
    )


def test_whole_targets_keep_is_positive_for_an_already_complete_rally() -> None:
    span = FixedSpan(
        FIXTURE,
        0,
        10,
        30,
        (_event(10, "Top"), _event(20, "Bot")),
    )
    options = build_options((span,), [{"candidate_lists": []}], {FIXTURE: span.events})[(FIXTURE, 0)]
    labels = _labels(
        _rally(10, 20),
        target_sides={(FIXTURE, 10): "Top", (FIXTURE, 20): "Bot"},
    )

    targets, _report = whole_targets(options, (span,), labels, {FIXTURE: 30.0})

    keep_index = next(index for index, option in enumerate(options) if option.kind == "keep")
    assert targets[keep_index] == 1


def test_whole_targets_exclude_unknown_labels_but_keep_options() -> None:
    labelled = _span(10, start=10, end=20, span_id=0)
    unknown_side = _span(30, start=30, end=40, span_id=1)
    absent = _span(50, start=50, end=60, span_id=2)
    baseline = (labelled, unknown_side, absent)
    options_by_section = build_options(
        baseline,
        [{"candidate_lists": []}],
        {FIXTURE: labelled.events + unknown_side.events + absent.events},
    )
    options = tuple(option for section_options in options_by_section.values() for option in section_options)
    labels = HumanLabels(
        {FIXTURE: (_rally(10), _rally(30, rally_id="set1:2", rally_index=1))},
        {(FIXTURE, 10): "Top"},
    )

    targets, report = whole_targets(options, baseline, labels, {FIXTURE: 30.0})

    targets_by_section = {}
    offset = 0
    for section_id, section_options in options_by_section.items():
        targets_by_section[section_id] = [
            int(target)
            for target in targets[offset : offset + len(section_options)]
        ]
        offset += len(section_options)
    assert targets_by_section[(FIXTURE, 0)][0] == 1
    assert all(target == -1 for target in targets_by_section[(FIXTURE, 1)])
    assert all(target == -1 for target in targets_by_section[(FIXTURE, 2)])
    assert report["action_counts"] == {
        "keep": 3,
        "add": 0,
        "replace": 0,
        "delete": 3,
        "add_delete": 0,
        "replace_delete": 0,
    }
    assert report["reasons"] == {"eligible": 2, "missing_labelled_sides": 2, "no_labelled_rally": 2}


def test_whole_targets_exclude_an_expansion_that_reaches_another_rally() -> None:
    span = _span(20, start=20, end=30)
    options = build_options(
        (span,),
        [{"candidate_lists": [_candidate_list(span, 10)]}],
        {FIXTURE: span.events},
    )[(FIXTURE, 0)]
    labels = _labels(
        _rally(20),
        _rally(15, rally_id="set1:2", rally_index=1),
        target_sides={(FIXTURE, 15): "Top", (FIXTURE, 20): "Top"},
    )

    targets, report = whole_targets(options, (span,), labels, {FIXTURE: 30.0})

    assert targets[0] == 1
    assert all(
        target == -1
        for option, target in zip(options, targets, strict=True)
        if option.candidate_frame is not None
    )
    assert report["reasons"]["expanded_section_has_other_labels"] == sum(
        option.candidate_frame is not None for option in options
    )


def test_choose_options_uses_priority_for_ties_and_keeps_below_threshold() -> None:
    span = _span(20, 30, start=20, end=40)
    options = build_options(
        (span,),
        [{"candidate_lists": [_candidate_list(span, 10)]}],
        {FIXTURE: span.events},
    )[(FIXTURE, 0)]
    scores = np.asarray(
        [0.60 if option.kind == "keep" else 0.90 if option.kind in {"add", "replace"} else 0.20 for option in options],
        dtype=np.float64,
    )

    selected = choose_options(options, scores, minimum_score=0.90)

    assert selected[(FIXTURE, 0)].kind == "add"
    assert option_identity(selected[(FIXTURE, 0)]) == (FIXTURE, 0, "add", 10, None)
    fallback = choose_options(options, np.full(len(options), 0.60), minimum_score=0.90)
    assert fallback[(FIXTURE, 0)].kind == "keep"
