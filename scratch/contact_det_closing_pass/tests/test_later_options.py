"""Exercise later-insertion boundaries, duplicate competition and side voting."""

from dataclasses import replace

import numpy as np
import pytest

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import section_result
from scratch.contact_det_closing_pass.scripts.later_evaluation import opportunity
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    apply_options,
    build_later_options,
    select_options,
    select_with_reference,
    shortlist_frames,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    contact_edit_effect,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def event(frame: int, side: str = "Top") -> FixedEvent:
    return FixedEvent("fixture", frame, 0.8, side)


def test_shortlist_uses_real_scores_and_scaled_distance() -> None:
    span = FixedSpan("fixture", 0, 0, 200, (event(20), event(140)))
    scores = np.array([(10, .99), (20, .99), (27, .99), (60, .8), (62, .9), (95, .7), (200, 1.)],
                      dtype=[("frame", "i4"), ("contact_score", "f8")])
    assert shortlist_frames(span, scores, 60) == [62, 95]
    assert shortlist_frames(span, scores, 30, limit=1) == [27]


def test_duplicate_insertion_cannot_claim_a_distinct_contact() -> None:
    span = FixedSpan("fixture", 0, 0, 100, (event(20), event(60)))
    base = CombinedAction("keep", None, None, span)
    candidates = {("fixture", 0): (event(20), event(25), event(28), event(100))}
    options = build_later_options((base,), candidates, {"fixture": 30})
    assert [option.inserted.frame for option in options if option.inserted] == [28]
    effects = contact_edit_effect(span, options[1].span, (25, 60), 10)
    assert effects["newly_matched_contacts"] == 0
    assert effects["unnecessary_added"] == 1


def test_insertion_changes_alternation_and_preserves_raw_guesses() -> None:
    span = FixedSpan("fixture", 0, 0, 120, (event(20), event(80)))
    base = CombinedAction("keep", None, None, span)
    options = build_later_options((base,), {("fixture", 0): (event(50, "Bot"),)}, {"fixture": 30})
    labels = HumanLabels(
        {"fixture": (RallyReference("fixture", 0, "one", (20, 50, 80)),)},
        {("fixture", 20): "Top", ("fixture", 50): "Bot", ("fixture", 80): "Top"},
    )
    assert not section_result(span, labels, 10)["side_rule_fully_correct"]
    assert section_result(options[1].span, labels, 10)["side_rule_fully_correct"]
    assert options[1].span.events[-1] == span.events[-1]


def test_start_and_delete_combinations_keep_insertion_and_fullstream_events() -> None:
    original = FixedSpan("fixture", 0, 20, 100, (event(20), event(70), event(90)))
    edited = FixedSpan("fixture", 0, 10, 100, (event(10, "Bot"), event(20), event(70)))
    base = CombinedAction("add_delete", 10, 90, edited)
    options = build_later_options((base,), {("fixture", 0): (event(45, "Bot"),)}, {"fixture": 30})
    selected = select_options(options, np.array([.3, .8]))
    streams = apply_options((original,), {"fixture": (*original.events, event(150))}, selected)
    assert [contact.frame for contact in streams.events_by_fixture["fixture"]] == [10, 20, 45, 70, 150]
    assert streams.spans[0].start_frame == 10
    assert select_options(options, np.array([.8, .8]))[("fixture", 0)].inserted is None


def test_empty_sections_remain_visible() -> None:
    span = FixedSpan("fixture", 0, 0, 100, ())
    options = build_later_options((CombinedAction("keep", None, None, span),), {}, {"fixture": 30})
    selected = select_options(options, np.array([.1]))
    assert apply_options((span,), {"fixture": ()}, selected).spans == (span,)


def test_score_advantage_restores_reference_start_and_contacts() -> None:
    span = FixedSpan("fixture", 0, 20, 100, (event(20), event(80)))
    keep = CombinedAction("keep", None, None, span)
    reference_span = FixedSpan("fixture", 0, 10, 100, (event(10, "Bot"), *span.events))
    opening = CombinedAction("add", 10, None, reference_span)
    options = build_later_options((keep, opening), {("fixture", 0): (event(50, "Bot"),)}, {"fixture": 30})
    reference = {("fixture", 0): LaterOption(opening, None, reference_span)}
    weak = select_with_reference(options, np.array([.30, .83, .80, .40]), reference)
    assert weak[("fixture", 0)].span == reference_span
    strong = select_with_reference(options, np.array([.30, .90, .80, .40]), reference)
    assert strong[("fixture", 0)] == options[1]


def test_opportunity_includes_missing_opening_and_later_hit_together() -> None:
    span = FixedSpan("fixture", 0, 20, 100, (event(20, "Bot"), event(80, "Bot")))
    keep = CombinedAction("keep", None, None, span)
    opening_span = FixedSpan("fixture", 0, 10, 100, (event(10), *span.events))
    opening = CombinedAction("add", 10, None, opening_span)
    options = build_later_options((keep, opening), {("fixture", 0): (event(50),)}, {"fixture": 30})
    labels = HumanLabels(
        {"fixture": (RallyReference("fixture", 0, "one", (10, 20, 50, 80)),)},
        {("fixture", 10): "Top", ("fixture", 20): "Bot", ("fixture", 50): "Top", ("fixture", 80): "Bot"},
    )
    result = opportunity(options, {("fixture", 0): LaterOption(keep, None, span)},
                         labels, {"fixture": 30}, {"fixture": "A"})
    assert result["10"]["counts"]["repair_with_same_base"] == 0
    assert result["10"]["counts"]["repair_with_start_delete_combinations"] == 1
    assert result["10"]["sections"][0]["distinct_local_candidate_frames"] == [50]


def test_reference_with_insertion_is_scored_and_restored_exactly() -> None:
    span = FixedSpan("fixture", 0, 20, 100, (event(20), event(80)))
    options = build_later_options(
        (CombinedAction("keep", None, None, span),),
        {("fixture", 0): (event(50, "Bot"),)}, {"fixture": 30},
    )
    reference = {("fixture", 0): options[1]}
    selected = select_with_reference(options, np.array([.83, .80]), reference)
    assert selected[("fixture", 0)] is options[1]
    streams = apply_options((span,), {"fixture": (*span.events, event(100))}, selected)
    assert [contact.frame for contact in streams.events_by_fixture["fixture"]] == [20, 50, 80, 100]
    assert streams.spans[0].events == options[1].span.events
    strong = select_with_reference(options, np.array([.90, .80]), reference)
    assert strong[("fixture", 0)] == options[0]


def test_equivalent_representations_use_best_reference_score_and_keep_ties() -> None:
    span = FixedSpan("fixture", 0, 0, 100, (event(20), event(80)))
    original = LaterOption(CombinedAction("keep", None, None, span), None, span)
    equivalent = replace(original, base=CombinedAction("replace", 20, None, span))
    changed_span = replace(span, start_frame=10)
    changed = replace(original, span=changed_span)
    reference = {("fixture", 0): original}
    selected = select_with_reference((original, equivalent, changed), np.array([.7, .8, .83]), reference)
    assert selected[("fixture", 0)] is original
    tied = select_with_reference((changed, equivalent), np.array([.8, .8]), reference, minimum_advantage=0)
    assert tied[("fixture", 0)] is original


def test_reference_edges_and_raw_sides_must_be_present_in_scored_options() -> None:
    span = FixedSpan("fixture", 0, 0, 100, (event(20), event(80)))
    original = LaterOption(CombinedAction("keep", None, None, span), None, span)
    for changed_span in (
        replace(span, end_frame=101),
        replace(span, events=(event(20, "Bot"), event(80))),
    ):
        changed = replace(original, span=changed_span)
        with pytest.raises(ValueError, match="missing from the scored alternatives"):
            select_with_reference((changed,), np.array([.8]), {("fixture", 0): original})


def test_pairs_need_both_contacts_and_respect_scaled_duplicate_distance() -> None:
    span = FixedSpan("fixture", 0, 0, 120, (event(20), event(100, "Bot")))
    base = CombinedAction("keep", None, None, span)
    options = build_later_options(
        (base,), {("fixture", 0): (event(45, "Bot"), event(50), event(75))},
        {"fixture": 60}, max_insertions=2,
    )
    pairs = [option for option in options if option.second_inserted is not None]
    assert [(option.inserted.frame, option.second_inserted.frame) for option in pairs] == [(45, 75), (50, 75)]
    labels = HumanLabels(
        {"fixture": (RallyReference("fixture", 0, "one", (20, 45, 75, 100)),)},
        {("fixture", 20): "Top", ("fixture", 45): "Bot", ("fixture", 75): "Top", ("fixture", 100): "Bot"},
    )
    for tolerance in (10, 20):
        assert not any(section_result(option.span, labels, tolerance)["side_rule_fully_correct"]
                       for option in options if len(option.inserted_events) < 2)
        assert section_result(pairs[0].span, labels, tolerance)["side_rule_fully_correct"]
