from __future__ import annotations

from typing import Any, cast

import pytest

from scratch.contact_det.scripts.score_contact_rallies import RallyReference
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    build_acceptance_rows,
    classify_section_result,
    select_acceptance_rules,
    summarise_acceptance,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "sset_01"


def _labels(
    *rallies: RallyReference,
    sides: dict[tuple[str, int], str | None] | None = None,
) -> HumanLabels:
    target_sides = (
        {
            (rally.fixture, frame): "Top"
            for rally in rallies
            for frame in rally.frames
        }
        if sides is None
        else sides
    )
    return HumanLabels(
        {FIXTURE: rallies},
        cast(dict[tuple[str, int], str], target_sides),
    )


def _rally(*frames: int, rally_id: str = "set1:1") -> RallyReference:
    return RallyReference(FIXTURE, 0, rally_id, frames)


def _row(
    *,
    overlap: int = 1,
    contained: bool = True,
    matches: list[tuple[int, int, int]] | None = None,
    labelled: int = 2,
    events: int = 2,
    voted_correct: int = 2,
    fully_correct: bool = True,
    span_id: int = 0,
    group: str = "D",
) -> dict[str, Any]:
    return {
        "fixture": FIXTURE,
        "span_id": span_id,
        "start_frame": 0,
        "end_frame": 100,
        "rally_id": "set1:1" if overlap == 1 else None,
        "overlapping_rallies": overlap,
        "whole_rally_contained": contained,
        "events": events,
        "labelled_contacts": labelled,
        "matches": [(0, 0, 0), (1, 1, 0)] if matches is None else matches,
        "voted_correct_sides": voted_correct,
        "fully_correct": fully_correct,
        "group": group,
    }


def test_known_partial_and_merged_sections_are_wrong() -> None:
    labels = _labels(_rally(10, 20), _rally(40, 50, rally_id="set1:2"))

    partial = classify_section_result(_row(contained=False), labels)
    merged = classify_section_result(_row(overlap=2, fully_correct=False), labels)

    assert partial == {"outcome": "wrong", "reason": "known_partial"}
    assert merged == {"outcome": "wrong", "reason": "known_merged_whole"}


def test_no_retained_label_and_anchor_only_sections_are_unjudgeable() -> None:
    labels = _labels(_rally(10, 20))

    result = classify_section_result(
        _row(overlap=0, labelled=0, events=1, matches=[], fully_correct=False),
        labels,
        (15,),
    )

    assert result == {"outcome": "unjudgeable", "reason": "no_retained_labels"}


def test_known_side_contradiction_precedes_uncertain_anchor() -> None:
    labels = _labels(_rally(10, 20))

    result = classify_section_result(
        _row(voted_correct=1, fully_correct=False),
        labels,
        (15,),
    )

    assert result == {"outcome": "wrong", "reason": "known_side_contradiction"}


def test_unknown_human_side_is_unjudgeable() -> None:
    labels = _labels(
        _rally(10, 20),
        sides={(FIXTURE, 10): "Top", (FIXTURE, 20): None},
    )

    result = classify_section_result(_row(voted_correct=1, fully_correct=False), labels)

    assert result == {"outcome": "unjudgeable", "reason": "unknown_human_side"}


def test_build_joins_keep_score_and_filters_anchors_to_section() -> None:
    labels = _labels(_rally(10, 20))
    sections = {
        tolerance: [_row(span_id=3, group="D")]
        for tolerance in ("10", "5")
    }
    choices = [{"fixture": FIXTURE, "span_id": 3, "kind": "keep", "score": 0.7}]

    rows = build_acceptance_rows(sections, choices, labels, {FIXTURE: [-1, 15, 150]})

    assert rows[0]["kind"] == "keep"
    assert rows[0]["score"] == 0.7
    assert rows[0]["group"] == "D"
    assert rows[0]["judgements"]["10"] == {"outcome": "unjudgeable", "reason": "uncertain_anchor"}
    assert rows[0]["judgements"]["5"] == {"outcome": "unjudgeable", "reason": "uncertain_anchor"}


def _acceptance_row(score: float, outcome: str, *, fixture: str = FIXTURE, group: str = "D") -> dict[str, Any]:
    judgement = {"outcome": outcome, "reason": outcome}
    return {
        "fixture": fixture,
        "span_id": 0,
        "group": group,
        "score": score,
        "judgements": {"10": judgement, "5": judgement},
    }


def test_empty_acceptance_has_none_precision_and_selection_is_unmet() -> None:
    rows = [_acceptance_row(0.4, "correct")]

    summary = summarise_acceptance(rows, 0.5)
    selected = select_acceptance_rules(rows)

    assert summary["accepted_count"] == 0
    assert summary["coverage"] == 0.0
    assert summary["by_tolerance"]["10"]["judged_precision"] is None
    assert selected["target_status"] == "unmet"
    assert selected["selected_rule"] is None
    assert selected["target_rules"]["0.99"]["target_status"] == "unmet"
    assert selected["fallback_rule"] is None
    assert selected["score_is_calibrated"] is False


def test_unknown_accepted_rows_count_for_coverage_not_judged_precision() -> None:
    rows = [_acceptance_row(0.5, "correct"), _acceptance_row(0.5, "unjudgeable", fixture="sset_02")]

    summary = summarise_acceptance(rows, 0.5)

    assert summary["accepted_count"] == 2
    assert summary["coverage"] == 1.0
    assert summary["by_tolerance"]["10"] == {
        "correct": 1,
        "wrong": 0,
        "unjudgeable": 1,
        "judged_count": 1,
        "judged_precision": 1.0,
    }
    assert summary["all_by_tolerance"]["10"] == summary["by_tolerance"]["10"]


def test_selection_uses_nonempty_rules_with_at_least_32_judged_rows() -> None:
    rows = [_acceptance_row(0.5, "correct", fixture=f"sset_{index:02d}") for index in range(32)]

    selected = select_acceptance_rules(rows)

    assert selected["target_status"] == "met"
    assert selected["selected_rule"]["threshold"] == 0.5
    assert selected["selected_rule"]["summary"]["population_count"] == 32
    assert selected["target_rules"]["0.99"]["selected_rule"]["threshold"] == 0.5


def test_selection_includes_all_development_groups_and_rejects_validation() -> None:
    rows = [
        _acceptance_row(0.5, "correct", fixture=f"sset_{index:02d}", group=group)
        for index, group in enumerate("ABCD" * 8)
    ]
    selected = select_acceptance_rules(rows)
    with pytest.raises(ValueError, match="only development"):
        select_acceptance_rules([*rows, _acceptance_row(1.0, "wrong", fixture="validation", group="V")])

    assert selected["included_groups"] == ["A", "B", "C", "D"]
    assert len(selected["curve"]) == 6
    assert selected["curve"][0]["summary"]["population_count"] == 32
    assert selected["selected_rule"]["accepted_count"] == 32


def test_selection_reports_each_precision_target_separately() -> None:
    rows = [
        _acceptance_row(0.5, "correct", fixture=f"sset_{index:02d}")
        for index in range(31)
    ]
    rows.append(_acceptance_row(0.5, "wrong", fixture="sset_31"))

    selected = select_acceptance_rules(rows)

    assert selected["target_rules"]["0.95"]["target_status"] == "met"
    assert selected["target_rules"]["0.99"]["target_status"] == "unmet"
    assert selected["selected_rule"]["threshold"] == 0.5
