"""Deleting a duplicate may remove a matched prediction without losing its GT hit."""

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    contact_edit_effect,
    voted_contact_scores,
)


def span(*frames: int) -> FixedSpan:
    return FixedSpan("fixture", 0, 0, 300, tuple(FixedEvent("fixture", frame, 0.9, "Top") for frame in frames))


def test_delete_duplicate_reassigns_match_without_losing_real_contact() -> None:
    result = contact_edit_effect(span(100, 104, 150), span(104, 150), (100, 150), 10)
    assert result["labelled_contacts_lost"] == 0
    assert result["previously_matched_predictions_removed"] == 1
    assert result["unnecessary_removed"] == 1
    assert result["unnecessary_added"] == 0


def test_combined_edit_can_recover_serve_and_remove_false_event() -> None:
    result = contact_edit_effect(span(140, 170, 200), span(100, 140, 200), (100, 140, 200), 10)
    assert result["newly_matched_contacts"] == 1
    assert result["first_contact_recovered"] == 1
    assert result["unnecessary_removed"] == 1
    assert result["labelled_contacts_lost"] == 0
    assert result["unnecessary_added"] == 0


def test_side_rescore_reuses_pairs_and_preserves_unknown_targets() -> None:
    raw = {"total": {"matched": 2, "side_answered": 1, "side_correct": 0}, "by_video": [
        {"fixture": "fixture", "matched": 2, "side_answered": 1, "side_correct": 0,
         "pairs": [[100, 104, "rally", True, 4, "Top", "Bot"], [150, 150, "rally", False, 0, None, "Bot"]]}]}
    events = {"fixture": (FixedEvent("fixture", 104, 0.9, "Top"), FixedEvent("fixture", 150, 0.9, "Top"))}
    result = voted_contact_scores(raw, events)
    assert result["total"] == {"matched": 2, "side_answered": 1, "side_correct": 1}
    assert result["by_video"][0]["pairs"][0][:6] == raw["by_video"][0]["pairs"][0][:6]
    assert raw["by_video"][0]["pairs"][0][-1] == "Bot"
