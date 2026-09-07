"""Score label-free whole-rally choices against the available label evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scratch.contact_det.scripts.score_contact_rallies import RallyReference
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

Outcome = Literal["correct", "wrong", "unjudgeable"]
TOLERANCES = ("10", "5")
ACCEPTANCE_THRESHOLDS = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
TARGET_JUDGED_PRECISIONS = (0.95, 0.99)
MIN_JUDGED_ACCEPTED = 32
PRIMARY_TOLERANCE = "10"
DEVELOPMENT_GROUPS = frozenset(("A", "B", "C", "D"))


def _rally_for_row(row: Mapping[str, Any], labels: HumanLabels) -> RallyReference:
    fixture = str(row["fixture"])
    rally_id = row.get("rally_id")
    matches = [rally for rally in labels.rallies.get(fixture, ()) if rally.rally_id == rally_id]
    if len(matches) != 1:
        raise ValueError(f"{fixture}/{rally_id}: expected one retained rally")
    return matches[0]


def _known_sides(rally: RallyReference, labels: HumanLabels) -> int:
    return sum(
        labels.target_sides.get((rally.fixture, frame)) is not None
        for frame in rally.frames
    )


def classify_section_result(
    row: Mapping[str, Any],
    labels: HumanLabels,
    uncertain_anchor_frames: Sequence[int] = (),
) -> dict[str, str]:
    """Classify one fixed-side section row using retained labels only.

    Known contradictions take precedence over uncertain anchors. An anchor is
    therefore an abstention only when the retained labels provide no hard
    contradiction.
    """
    fixture = str(row["fixture"])
    overlap_count = int(row["overlapping_rallies"])
    if overlap_count == 0:
        return {"outcome": "unjudgeable", "reason": "no_retained_labels"}
    if overlap_count > 1:
        return {"outcome": "wrong", "reason": "known_merged_whole"}

    rally = _rally_for_row(row, labels)
    if not bool(row["whole_rally_contained"]):
        return {"outcome": "wrong", "reason": "known_partial"}

    matches = row["matches"]
    labelled_contacts = int(row["labelled_contacts"])
    if len(matches) < labelled_contacts:
        return {"outcome": "wrong", "reason": "missing_known_contact"}

    known_sides = _known_sides(rally, labels)
    if len(matches) == labelled_contacts and int(row["voted_correct_sides"]) < known_sides:
        return {"outcome": "wrong", "reason": "known_side_contradiction"}

    if uncertain_anchor_frames:
        return {"outcome": "unjudgeable", "reason": "uncertain_anchor"}

    event_count = int(row["events"])
    if len(matches) == labelled_contacts and event_count > labelled_contacts:
        return {"outcome": "wrong", "reason": "extra_events"}

    if known_sides < len(rally.frames):
        return {"outcome": "unjudgeable", "reason": "unknown_human_side"}

    if row.get("fully_correct") is not True:
        raise AssertionError(f"{fixture}/{row['span_id']}: fixed row is not fully correct")
    return {"outcome": "correct", "reason": "fully_correct"}


def _sections_by_identity(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        identity = (str(row["fixture"]), int(row["span_id"]))
        if identity in indexed:
            raise ValueError(f"{identity}: section row repeats")
        indexed[identity] = row
    return indexed


def _tolerance_rows(
    sections_by_tolerance: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[tuple[str, int], Mapping[str, Any]]]:
    return {
        tolerance: _sections_by_identity(sections_by_tolerance[tolerance])
        for tolerance in TOLERANCES
    }


def build_acceptance_rows(
    sections_by_tolerance: Mapping[str, Sequence[dict[str, Any]]],
    choices: Sequence[dict[str, Any]],
    labels: HumanLabels,
    uncertain_anchors: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Join selected scores to ±10/±5 judgements without filtering rows."""
    sections = _tolerance_rows(sections_by_tolerance)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for choice in choices:
        fixture = str(choice["fixture"])
        span_id = int(choice["span_id"])
        identity = (fixture, span_id)
        if identity in seen:
            raise ValueError(f"{identity}: selected choice repeats")
        seen.add(identity)
        score = float(choice["score"])
        if not math.isfinite(score):
            raise ValueError(f"{identity}: selected score is not finite")
        rows = {tolerance: sections[tolerance][identity] for tolerance in TOLERANCES}
        section_start = int(rows["10"]["start_frame"])
        section_end = int(rows["10"]["end_frame"])
        anchors = tuple(
            int(frame) for frame in uncertain_anchors.get(fixture, ())
            if section_start <= int(frame) < section_end
        )
        judgements = {
            tolerance: classify_section_result(rows[tolerance], labels, anchors)
            for tolerance in TOLERANCES
        }
        record: dict[str, Any] = {
            "fixture": fixture,
            "span_id": span_id,
            "kind": str(choice["kind"]),
            "score": score,
            "judgements": judgements,
        }
        groups = {row["group"] for row in rows.values() if "group" in row}
        if len(groups) > 1:
            raise ValueError(f"{identity}: section groups differ by tolerance")
        if groups:
            record["group"] = str(next(iter(groups)))
        output.append(record)
    if any(seen != set(sections[tolerance]) for tolerance in TOLERANCES):
        raise ValueError("Acceptance choices must cover every scored section")
    return output


def _judgement_counts(rows: Sequence[Mapping[str, Any]], tolerance: str) -> dict[str, Any]:
    counts = Counter(str(row["judgements"][tolerance]["outcome"]) for row in rows)
    correct = counts["correct"]
    wrong = counts["wrong"]
    unjudgeable = counts["unjudgeable"]
    judged = correct + wrong
    return {
        "correct": correct,
        "wrong": wrong,
        "unjudgeable": unjudgeable,
        "judged_count": judged,
        "judged_precision": correct / judged if judged else None,
    }


def _partition_summary(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = [row for row in rows if float(row["score"]) >= threshold]
    return {
        "population_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "by_tolerance": {
            tolerance: _judgement_counts(accepted, tolerance) for tolerance in TOLERANCES
        },
        "all_by_tolerance": {
            tolerance: _judgement_counts(rows, tolerance) for tolerance in TOLERANCES
        },
    }


def summarise_acceptance(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Summarise score-threshold coverage and judged precision at both tolerances."""
    if not math.isfinite(float(threshold)):
        raise ValueError("acceptance threshold must be finite")
    summary = {"threshold": float(threshold), **_partition_summary(rows, float(threshold))}
    by_video: dict[str, list[Mapping[str, Any]]] = {}
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        fixture = str(row["fixture"])
        by_video.setdefault(fixture, []).append(row)
        if "group" in row:
            group = str(row["group"])
            by_group.setdefault(group, []).append(row)
    summary["by_video"] = {
        fixture: _partition_summary(video_rows, float(threshold))
        for fixture, video_rows in by_video.items()
    }
    if by_group:
        summary["by_group"] = {
            group: _partition_summary(group_rows, float(threshold))
            for group, group_rows in by_group.items()
        }
    return summary


def _threshold_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(summary["threshold"])
    primary = summary["by_tolerance"][PRIMARY_TOLERANCE]
    return {
        "threshold": threshold,
        "target_qualification": {
            str(target): (
                summary["accepted_count"] > 0
                and primary["judged_count"] >= MIN_JUDGED_ACCEPTED
                and primary["judged_precision"] is not None
                and primary["judged_precision"] >= target
            )
            for target in TARGET_JUDGED_PRECISIONS
        },
        "accepted_count": summary["accepted_count"],
        "judged_count": primary["judged_count"],
        "judged_precision": primary["judged_precision"],
        "summary": dict(summary),
    }


def _selection_key(record: Mapping[str, Any]) -> tuple[int, float, float]:
    precision = record["judged_precision"]
    return (
        int(record["accepted_count"]),
        float(precision) if precision is not None else -1.0,
        float(record["threshold"]),
    )


def select_acceptance_rules(
    rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = ACCEPTANCE_THRESHOLDS,
) -> dict[str, Any]:
    """Select development rules and retain a diagnostic unmet fallback."""
    if any(str(row.get("group")) not in DEVELOPMENT_GROUPS for row in rows):
        raise ValueError("Acceptance selection requires only development groups A, B, C and D")
    development_rows = rows
    summaries = [summarise_acceptance(development_rows, threshold) for threshold in thresholds]
    curve = [_threshold_record(summary) for summary in summaries]
    target_rules: dict[str, dict[str, Any]] = {}
    for target in TARGET_JUDGED_PRECISIONS:
        target_key = str(target)
        qualifying = [
            record for record in curve if record["target_qualification"][target_key]
        ]
        selected = max(qualifying, key=_selection_key, default=None)
        target_rules[target_key] = {
            "target_judged_precision": target,
            "target_status": "met" if selected is not None else "unmet",
            "selected_rule": selected,
        }

    judged_fallbacks = [
        record
        for record in curve
        if record["accepted_count"] > 0 and record["judged_count"] >= MIN_JUDGED_ACCEPTED
    ]
    nonempty_fallbacks = [record for record in curve if record["accepted_count"] > 0]
    fallback = max(
        judged_fallbacks,
        key=lambda record: (float(record["judged_precision"]), *_selection_key(record)),
        default=None,
    )
    if fallback is None:
        fallback = max(
            nonempty_fallbacks,
            key=_selection_key,
            default=None,
        )
    primary_rule = target_rules[str(TARGET_JUDGED_PRECISIONS[0])]["selected_rule"]
    return {
        "population": "development",
        "included_groups": sorted(DEVELOPMENT_GROUPS),
        "thresholds": list(thresholds),
        "target_judged_precisions": list(TARGET_JUDGED_PRECISIONS),
        "minimum_judged_accepted": MIN_JUDGED_ACCEPTED,
        "primary_tolerance": int(PRIMARY_TOLERANCE),
        "target_rules": target_rules,
        "target_status": target_rules[str(TARGET_JUDGED_PRECISIONS[0])]["target_status"],
        "score_is_calibrated": False,
        "selected_rule": primary_rule,
        "fallback_rule": fallback,
        "curve": curve,
    }
