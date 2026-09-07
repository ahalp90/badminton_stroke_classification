"""Measure whether removing a retained event preserves labelled contact coverage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan, RallyReference
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)


def deletion_effect(before: FixedSpan, after: FixedSpan, rally: RallyReference, tolerance: int) -> dict[str, Any]:
    """Re-match after removal so either member of a duplicate can represent a hit."""
    before_pairs = match_contacts(rally.frames, [event.frame for event in before.events], tolerance)
    after_pairs = match_contacts(rally.frames, [event.frame for event in after.events], tolerance)
    before_gt = {pair[0] for pair in before_pairs}
    after_gt = {pair[0] for pair in after_pairs}
    fewer_extras = len(before.events) - len(before_pairs) > len(after.events) - len(after_pairs)
    return {
        "useful": before_gt <= after_gt and fewer_extras,
        "lost_gt_indices": sorted(before_gt - after_gt),
        "gained_gt_indices": sorted(after_gt - before_gt),
        "unnecessary_before": len(before.events) - len(before_pairs),
        "unnecessary_after": len(after.events) - len(after_pairs),
        "other_missing_hits": len(before_gt) < len(rally.frames),
    }


def deletion_opportunities(
    spans: Sequence[FixedSpan], labels: HumanLabels, fps: Mapping[str, float],
    offered: Mapping[tuple[str, int], set[tuple[int, ...]]], tolerance_base30: int,
) -> dict[str, Any]:
    """Count local deletions on actual current events, including incomplete rallies.

    Unmatched events outside a retained rally's contact envelope are uncertain.
    A timing-compatible duplicate at either end still has direct label support.
    Overlapping or repeated section/rally associations remain unknown.
    """
    overlaps = {(span.fixture, span.span_id): overlapping_rallies(span, labels) for span in spans}
    touching = Counter((span.fixture, rally.rally_id) for span in spans for rally in overlaps[(span.fixture, span.span_id)])
    rows, counts = [], Counter()
    for span in spans:
        identity = (span.fixture, span.span_id)
        rallies = overlaps[identity]
        rally = rallies[0] if len(rallies) == 1 else None
        if rally is not None and touching[(span.fixture, rally.rally_id)] != 1:
            rally = None
        tolerance = scale_base30_frames(tolerance_base30, fps[span.fixture])
        for index, event in enumerate(span.events):
            revised = replace(span, events=span.events[:index] + span.events[index + 1:])
            frames = tuple(contact.frame for contact in revised.events)
            row = {
                "fixture": span.fixture, "span_id": span.span_id, "frame": event.frame,
                "position": "start" if index == 0 else "end" if index == len(span.events) - 1 else "interior",
                "offered": frames in offered.get(identity, set()), "target": -1,
            }
            counts["retained_events"] += 1
            if rally is not None:
                effect = deletion_effect(span, revised, rally, tolerance)
                nearby = [frame for frame in rally.frames if abs(event.frame - frame) <= tolerance]
                duplicate = any(
                    abs(other.frame - frame) <= tolerance
                    for other in revised.events for frame in nearby
                )
                supported = rally.frames[0] <= event.frame <= rally.frames[-1] or duplicate
                row.update(effect)
                row["rally_id"] = rally.rally_id
                row["duplicate"] = duplicate
                row["within_label_support"] = supported
                if effect["lost_gt_indices"] or supported:
                    row["target"] = int(effect["useful"])
                if row["target"] == 1:
                    row["completes_rally"] = section_result(revised, labels, tolerance)["side_rule_fully_correct"]
                    counts["useful"] += 1
                    counts[f"useful_{row['position']}"] += 1
                    counts["useful_duplicate"] += int(duplicate)
                    counts["useful_with_other_miss"] += int(effect["other_missing_hits"])
                    counts["useful_already_offered"] += int(row["offered"])
                    counts["useful_completes_rally"] += int(row["completes_rally"])
                elif effect["useful"]:
                    counts["possible_but_unlabelled"] += 1
            counts[f"target_{row['target']}"] += 1
            rows.append(row)
    useful_sections = {(row["fixture"], row["span_id"]) for row in rows if row["target"] == 1}
    counts["sections_with_useful_deletion"] = len(useful_sections)
    return {"counts": dict(counts), "rows": rows}
