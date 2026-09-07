"""Shared timing and whole-section scoring for the closing experiments."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_followup.scripts.score_followup import load_saved_test_labels
from scratch.contact_det_followup.scripts.score_start_model import (
    _with_alternating_sides,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

Matcher = Callable[[Sequence[int], Sequence[int], int], list[tuple[int, int, int]]]


def write_json(path: Path, payload: object) -> None:
    """Save a compressed result without machine-specific execution metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)
        handle.write("\n")


def test_labels() -> HumanLabels:
    saved = load_saved_test_labels().labels
    rallies = {}
    sides = {}
    for fixture, human_rallies in saved.rallies_by_fixture.items():
        references = []
        for index, rally in enumerate(human_rallies):
            frames = tuple(contact.frame for contact in rally.contacts)
            references.append(RallyReference(fixture, index, f"{rally.set_id}:{rally.rally}", frames))
            for contact in rally.contacts:
                sides[(fixture, contact.frame)] = contact.side
        rallies[fixture] = tuple(references)
    return HumanLabels(rallies, sides)


def overlapping_rallies(span: FixedSpan, labels: HumanLabels) -> tuple[RallyReference, ...]:
    return tuple(
        rally for rally in labels.rallies.get(span.fixture, ())
        if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    )


def section_result(
    span: FixedSpan,
    labels: HumanLabels,
    tolerance: int,
    matcher: Matcher = match_contacts,
) -> dict[str, Any]:
    """Require every contact of one labelled rally inside the half-open section."""
    rallies = overlapping_rallies(span, labels)
    rally = rallies[0] if len(rallies) == 1 else None
    contained = rally is not None and all(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    frames = [event.frame for event in span.events]
    matches = [] if rally is None else matcher(rally.frames, frames, tolerance)
    timing = contained and len(matches) == len(frames) == len(rally.frames)
    correct_sides = 0
    voted_correct_sides = 0
    voted = _with_alternating_sides(span)
    if rally is not None:
        for gt_index, pred_index, _offset in matches:
            target = labels.target_sides[(span.fixture, rally.frames[gt_index])]
            correct_sides += target is not None and span.events[pred_index].predicted_side == target
            voted_correct_sides += target is not None and voted.events[pred_index].predicted_side == target
    return {
        "fixture": span.fixture, "span_id": span.span_id,
        "start_frame": span.start_frame, "end_frame": span.end_frame,
        "rally_id": None if rally is None else rally.rally_id,
        "overlapping_rallies": len(rallies), "whole_rally_contained": contained,
        "events": len(frames), "labelled_contacts": 0 if rally is None else len(rally.frames),
        "matches": matches, "correct_sides": int(correct_sides),
        "voted_correct_sides": int(voted_correct_sides),
        "timing_complete": bool(timing),
        "fully_correct": bool(timing and correct_sides == len(frames)),
        "side_rule_fully_correct": bool(timing and voted_correct_sides == len(frames)),
    }


def score_sections(
    spans: Sequence[FixedSpan], labels: HumanLabels, fps: Mapping[str, float],
    tolerance_base30: int, matcher: Matcher = match_contacts,
) -> list[dict[str, Any]]:
    return [
        section_result(span, labels, ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps[span.fixture]), matcher)
        for span in spans
    ]


def score_contacts(
    events: Mapping[str, Sequence[FixedEvent]], labels: HumanLabels, fps: Mapping[str, float],
    tolerance_base30: int, matcher: Matcher = match_contacts,
) -> dict[str, Any]:
    """Match complete video streams; true rally edges never crop predictions."""
    by_video = []
    totals = {name: 0 for name in (
        "labelled", "predicted", "matched", "side_answered", "side_correct", "first_contacts", "first_matched",
    )}
    for fixture, predictions in events.items():
        contacts = []
        for rally in labels.rallies[fixture]:
            for index, frame in enumerate(rally.frames):
                contacts.append((frame, rally.rally_id, index == 0))
        contacts.sort()
        tolerance = ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps[fixture])
        pairs = matcher([row[0] for row in contacts], [event.frame for event in predictions], tolerance)
        matched_rows = []
        counts = {"labelled": len(contacts), "predicted": len(predictions), "matched": len(pairs),
                  "side_answered": 0, "side_correct": 0,
                  "first_contacts": sum(row[2] for row in contacts), "first_matched": 0}
        for gt_index, pred_index, offset in pairs:
            frame, rally_id, first = contacts[gt_index]
            prediction = predictions[pred_index]
            target = labels.target_sides[(fixture, frame)]
            answered = target is not None and prediction.predicted_side is not None
            correct = answered and target == prediction.predicted_side
            counts["side_answered"] += answered
            counts["side_correct"] += correct
            counts["first_matched"] += first
            matched_rows.append([frame, prediction.frame, rally_id, first, offset, target, prediction.predicted_side])
        for key, value in counts.items():
            totals[key] += value
        by_video.append({"fixture": fixture, **counts, "pairs": matched_rows})
    totals["precision"] = totals["matched"] / totals["predicted"] if totals["predicted"] else 0.0
    totals["recall"] = totals["matched"] / totals["labelled"] if totals["labelled"] else 0.0
    denominator = totals["predicted"] + totals["labelled"]
    totals["f1"] = 2 * totals["matched"] / denominator if denominator else 0.0
    return {"total": totals, "by_video": by_video}


def paired_sections(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare by GT rally identity so a vanished correct rally remains a loss."""
    before_correct = {(row["fixture"], row["rally_id"]) for row in before if row["fully_correct"]}
    after_correct = {(row["fixture"], row["rally_id"]) for row in after if row["fully_correct"]}
    return {
        "sections_before": len(before), "sections_after": len(after),
        "correct_before": len(before_correct), "correct_after": len(after_correct),
        "correct_sections_before": sum(row["fully_correct"] for row in before),
        "correct_sections_after": sum(row["fully_correct"] for row in after),
        "repaired": sorted(after_correct - before_correct), "lost": sorted(before_correct - after_correct),
    }
