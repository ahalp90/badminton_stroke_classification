"""Summarise saved chosen-acceptance rows without reopening labels or fits."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

TOLERANCES = ("10", "5")
SCORE_KEYS = {"base": "base_score", "gap": "gap_score"}
LENGTH_BUCKETS = ("1-5", "6-10", "11-20", "21+", "no_single_labelled_rally")
ERROR_NAMES = ("missed_serve", "later_miss", "extras", "side_errors", "section_problem")
OUTCOMES = ("correct", "wrong", "unjudgeable")
Entry = tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]


def _entries(
    rows: Sequence[Mapping[str, Any]],
    sections_by_tolerance: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[Entry]:
    sections = {}
    for tolerance in TOLERANCES:
        source = sections_by_tolerance[tolerance]
        indexed = {(str(s["fixture"]), int(s["span_id"])): s for s in source}
        if len(indexed) != len(source):
            raise ValueError(f"section identity repeats at tolerance {tolerance}")
        sections[tolerance] = indexed
    output: list[Entry] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        identity = (str(row["fixture"]), int(row["span_id"]))
        if identity in seen:
            raise ValueError(f"{identity}: acceptance row repeats")
        seen.add(identity)
        judgements = row["judgements"]
        joined = {}
        for tolerance in TOLERANCES:
            outcome = str(judgements[tolerance]["outcome"])
            if outcome not in OUTCOMES:
                raise ValueError(f"{identity}/{tolerance}: unknown outcome {outcome}")
            joined[tolerance] = sections[tolerance][identity]
        if not all(math.isfinite(float(row[key])) for key in SCORE_KEYS.values()):
            raise ValueError(f"{identity}: acceptance score is not finite")
        output.append((row, joined))
    for tolerance in TOLERANCES:
        if set(sections[tolerance]) != seen:
            raise ValueError(f"saved rows and tolerance {tolerance} sections differ")
    return output


def _outcomes(entries: Sequence[Entry], tolerance: str) -> dict[str, int]:
    counts = Counter(str(row["judgements"][tolerance]["outcome"]) for row, _ in entries)
    return {"count": len(entries), **{outcome: int(counts[outcome]) for outcome in OUTCOMES}}


def _wrong_errors(entries: Sequence[Entry], tolerance: str) -> dict[str, int]:
    wrong = [
        sections[tolerance]
        for row, sections in entries
        if row["judgements"][tolerance]["outcome"] == "wrong"
    ]
    totals = {name: 0 for name in ERROR_NAMES}
    for section in wrong:
        matches = section["matches"]
        if not isinstance(matches, Sequence) or isinstance(matches, (str, bytes)):
            raise TypeError("section matches must be a list")
        single_rally = int(section["overlapping_rallies"]) == 1
        matched_gt = {int(match[0]) for match in matches}
        labelled_contacts = int(section["labelled_contacts"])
        flags = (
            single_rally and 0 not in matched_gt,
            single_rally and any(index not in matched_gt for index in range(1, labelled_contacts)),
            single_rally and int(section["events"]) > len(matches),
            single_rally and int(section["voted_correct_sides"]) < len(matches),
            not single_rally or not bool(section["whole_rally_contained"]),
        )
        for name, flagged in zip(ERROR_NAMES, flags, strict=True):
            totals[name] += int(flagged)
    return {"wrong_rows": len(wrong), **totals}


def _tolerance_summary(
    entries: Sequence[Entry], tolerance: str, accepted: Sequence[Entry], rejected: Sequence[Entry]
) -> dict[str, Any]:
    def length_bucket(section: Mapping[str, Any]) -> str:
        if int(section["overlapping_rallies"]) != 1 or section.get("rally_id") is None:
            return "no_single_labelled_rally"
        length = int(section["labelled_contacts"])
        return LENGTH_BUCKETS[0 if length <= 5 else 1 if length <= 10 else 2 if length <= 20 else 3]

    def rally_count(values: Sequence[Entry]) -> int:
        return len({
            (str(row["fixture"]), str(section["rally_id"]))
            for row, sections in values
            for section in (sections[tolerance],)
            if row["judgements"][tolerance]["outcome"] == "correct"
            and int(section["overlapping_rallies"]) == 1
            and section.get("rally_id") is not None
        })

    partitions = {"all": entries, "accepted": accepted, "rejected": rejected}
    return {
        "outcomes": {name: _outcomes(values, tolerance) for name, values in partitions.items()},
        "unique_labelled_rallies_fully_correct": {
            name: rally_count(values) for name, values in partitions.items()
        },
        "by_labelled_contact_length": {
            bucket: {
                name: _outcomes(
                    [entry for entry in values if length_bucket(entry[1][tolerance]) == bucket], tolerance
                )
                for name, values in partitions.items()
            }
            for bucket in LENGTH_BUCKETS
        },
        "accepted_wrong_error_categories": _wrong_errors(accepted, tolerance),
    }


def _partition_summary(
    entries: Sequence[Entry], score_key: str, threshold: float, *, include_videos: bool = True
) -> dict[str, Any]:
    accepted = [entry for entry in entries if float(entry[0][score_key]) >= threshold]
    rejected = [entry for entry in entries if float(entry[0][score_key]) < threshold]
    result: dict[str, Any] = {
        "threshold": threshold,
        "population_count": len(entries),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "by_tolerance": {
            tolerance: _tolerance_summary(entries, tolerance, accepted, rejected)
            for tolerance in TOLERANCES
        },
    }
    if include_videos:
        fixtures = sorted({str(row["fixture"]) for row, _ in entries})
        result["by_video"] = {
            fixture: _partition_summary(
                [entry for entry in entries if str(entry[0]["fixture"]) == fixture],
                score_key, threshold, include_videos=False,
            )
            for fixture in fixtures
        }
    return result


def summarise(
    rows: Sequence[Mapping[str, Any]],
    sections_by_tolerance: Mapping[str, Sequence[Mapping[str, Any]]],
    policies: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Summarise saved rows and the saved ``serve_result[tol]["sections"]``.

    Policies are ``{variant: {name: {"threshold": number}}}`` or one
    threshold object per variant. Null policies are skipped; unjudgeable
    outcomes stay separate from wrong.
    """
    entries = _entries(rows, sections_by_tolerance)
    output: dict[str, Any] = {
        "schema": "contact-chosen-acceptance-breakdown/1",
        "variants": {},
    }
    for variant, policy_set in policies.items():
        if variant not in SCORE_KEYS:
            raise ValueError(f"unknown acceptance variant {variant}")
        if policy_set is None:
            continue
        items = [("default", policy_set)] if "threshold" in policy_set else policy_set.items()
        for name, policy in items:
            if policy is None:
                continue
            if not isinstance(policy, Mapping):
                raise TypeError(f"{variant}/{name}: policy must be an object")
            threshold = float(policy["threshold"])
            if not math.isfinite(threshold):
                raise ValueError(f"{variant}/{name}: threshold is not finite")
            variant_result = output["variants"].setdefault(
                variant, {"score_key": SCORE_KEYS[variant], "policies": {}}
            )
            variant_result["policies"][str(name)] = _partition_summary(
                entries, SCORE_KEYS[variant], threshold
            )
    return output
