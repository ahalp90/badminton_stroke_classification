"""Score combined edits and reuse time pairings for raw and voted player sides."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    paired_sections,
    score_sections,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_closing_pass.scripts.run_start_comparison import _summary
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
)


def section_views(
    spans: Sequence[FixedSpan], labels: HumanLabels, fps: Mapping[str, float],
    groups: Mapping[str, str], tolerance: int,
) -> dict[str, dict[str, Any]]:
    raw = score_sections(spans, labels, fps, tolerance)
    fixed = []
    for row in raw:
        row["group"] = groups[row["fixture"]]
        fixed.append({**row, "fully_correct": row["side_rule_fully_correct"],
                      "correct_sides": row["voted_correct_sides"]})
    return {"raw": {"sections": raw, "summary": _summary(raw)},
            "fixed_side": {"sections": fixed, "summary": _summary(fixed)}}


def paired_evaluations(
    baseline: Mapping[str, dict[str, Any]], edited: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for tolerance in (10, 5):
        before = baseline[str(tolerance)]
        after = edited[str(tolerance)]
        result[str(tolerance)] = {
            "edited_raw": after["raw"], "edited_fixed_side": after["fixed_side"],
            "paired_raw": paired_sections(before["raw"]["sections"], after["raw"]["sections"]),
            "paired_fixed_side": paired_sections(before["fixed_side"]["sections"], after["fixed_side"]["sections"]),
        }
    return result


def voted_contact_scores(
    raw: Mapping[str, Any], events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, Any]:
    """Update side answers on existing full-stream time pairs after the side vote."""
    by_video = []
    for video in raw["by_video"]:
        sides = {event.frame: event.predicted_side for event in events_by_fixture[video["fixture"]]}
        pairs = []
        answered = 0
        correct = 0
        for pair in video["pairs"]:
            gt_frame, pred_frame, rally_id, first, offset, target_side, _raw_side = pair
            side = sides[pred_frame]
            has_answer = target_side is not None and side is not None
            answered += has_answer
            correct += has_answer and side == target_side
            pairs.append([gt_frame, pred_frame, rally_id, first, offset, target_side, side])
        by_video.append({**video, "pairs": pairs, "side_answered": answered, "side_correct": correct})
    totals = dict(raw["total"])
    totals["side_answered"] = sum(video["side_answered"] for video in by_video)
    totals["side_correct"] = sum(video["side_correct"] for video in by_video)
    return {"total": totals, "by_video": by_video}


def contact_edit_effect(before: FixedSpan, after: FixedSpan, rally_frames: Sequence[int], tolerance: int) -> dict[str, int]:
    """Count changed GT coverage and unmatched events after re-pairing an edit."""
    before_frames = [event.frame for event in before.events]
    after_frames = [event.frame for event in after.events]
    before_pairs = match_contacts(rally_frames, before_frames, tolerance)
    after_pairs = match_contacts(rally_frames, after_frames, tolerance)
    before_gt = {pair[0] for pair in before_pairs}
    after_gt = {pair[0] for pair in after_pairs}
    before_matched = {before_frames[pair[1]] for pair in before_pairs}
    removed = set(before_frames) - set(after_frames)
    removed_unmatched = len(removed - before_matched)
    before_unmatched = len(before_frames) - len(before_pairs)
    after_unmatched = len(after_frames) - len(after_pairs)
    unnecessary_added = max(0, after_unmatched - before_unmatched + removed_unmatched)
    false_removed = min(len(removed), max(0, before_unmatched - after_unmatched + unnecessary_added))
    return {
        "newly_matched_contacts": len(after_gt - before_gt),
        "labelled_contacts_lost": len(before_gt - after_gt),
        "previously_matched_predictions_removed": len(removed & before_matched),
        "unnecessary_added": unnecessary_added,
        "unnecessary_removed": false_removed,
        "first_contact_recovered": int(0 in after_gt and 0 not in before_gt),
        "first_contact_lost": int(0 in before_gt and 0 not in after_gt),
    }


def local_harm(
    selected: Mapping[tuple[str, int], CombinedAction], baseline_spans: Sequence[FixedSpan],
    baseline_rows: Sequence[Mapping[str, Any]], labels: HumanLabels, fps: Mapping[str, float], tolerance: int,
) -> dict[str, Any]:
    before_by_section = {(span.fixture, span.span_id): span for span in baseline_spans}
    correct_before = {(row["fixture"], row["span_id"]) for row in baseline_rows if row["fully_correct"]}
    overlaps = {identity: overlapping_rallies(span, labels) for identity, span in before_by_section.items()}
    touched = Counter((fixture, rally.rally_id) for (fixture, _span_id), rallies in overlaps.items() for rally in rallies)
    rows = []
    totals = Counter()
    for identity, option in selected.items():
        if option.kind == "keep":
            continue
        before = before_by_section[identity]
        rallies = overlaps[identity]
        judgeable = (
            len(rallies) == 1 and touched[(before.fixture, rallies[0].rally_id)] == 1
            and overlapping_rallies(option.span, labels) == rallies
        )
        row = {"fixture": before.fixture, "span_id": before.span_id, "kind": option.kind,
               "candidate_frame": option.candidate_frame, "deleted_frame": option.deleted_frame, "judgeable": judgeable}
        totals["selected_edits"] += 1
        if judgeable:
            scaled = ScalingKind.FRAME_COUNT.scale(tolerance, fps[before.fixture])
            effects = contact_edit_effect(before, option.span, rallies[0].frames, scaled)
            useful = effects["newly_matched_contacts"] > 0 or effects["unnecessary_removed"] > 0
            beneficial = useful and effects["labelled_contacts_lost"] == 0 and effects["unnecessary_added"] == 0
            row.update(effects)
            row["beneficial_contact_edit"] = beneficial
            row["rally_id"] = rallies[0].rally_id
            totals.update(effects)
            totals["judgeable_edits"] += 1
            totals["beneficial_contact_edits"] += beneficial
            totals["unsuccessful_edits_in_already_wrong_sections"] += not beneficial and identity not in correct_before
        else:
            totals["unjudgeable_edits"] += 1
        rows.append(row)
    return {"counts": dict(totals), "selected": rows,
            "loss_definition": "labelled contacts matched before but unmatched after re-pairing; removed prediction identities counted separately"}
