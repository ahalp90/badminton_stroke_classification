"""Compare actual later-contact alternatives with the preserved combined output."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    paired_sections,
    section_result,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    option_record,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    contact_edit_effect,
    section_views,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)


def opportunity(
    options: Sequence[LaterOption], reference: Mapping[tuple[str, int], LaterOption],
    labels: HumanLabels, fps: Mapping[str, float], groups: Mapping[str, str],
) -> dict[str, Any]:
    """Count a label-chosen ceiling using real proposed timestamps and raw sides."""
    grouped: dict[tuple[str, int], list[LaterOption]] = {}
    for option in options:
        grouped.setdefault(option.base.identity, []).append(option)
    result = {}
    for tolerance in (10, 5):
        rows = []
        for identity, candidates in grouped.items():
            current = reference[identity]
            scaled = scale_base30_frames(tolerance, fps[identity[0]])
            before = section_result(current.span, labels, scaled)
            rallies = overlapping_rallies(current.span, labels)
            missing_later = []
            useful_candidates = []
            if len(rallies) == 1:
                matched = {pair[0] for pair in before["matches"]}
                missing_later = [frame for index, frame in enumerate(rallies[0].frames) if index > 0 and index not in matched]
            possible = []
            same_base = []
            for option in candidates:
                if option.inserted is None:
                    continue
                if option.base == current.base and len(rallies) == 1:
                    after_pairs = match_contacts(rallies[0].frames, [event.frame for event in option.span.events], scaled)
                    after_matched = {pair[0] for pair in after_pairs}
                    if matched < after_matched:
                        useful_candidates.append(option.inserted.frame)
                # Whole correctness requires equal counts; most cross-products
                # can be rejected without running the time matcher.
                if before["overlapping_rallies"] == 1 and len(option.span.events) != before["labelled_contacts"]:
                    continue
                after = section_result(option.span, labels, scaled)
                if after["side_rule_fully_correct"]:
                    possible.append(option_record(option))
                    if option.base == current.base:
                        same_base.append(option_record(option))
            rows.append({
                "fixture": identity[0], "span_id": identity[1], "group": groups[identity[0]],
                "reference_correct": before["side_rule_fully_correct"],
                "labelled_contacts": before["labelled_contacts"], "events": before["events"],
                "missing_matched_contacts": before["labelled_contacts"] - len(before["matches"]),
                "missing_later_frames": missing_later, "distinct_local_candidate_frames": useful_candidates,
                "with_same_base": same_base, "with_any_base": possible,
            })
        totals = Counter()
        by_group: dict[str, Counter] = {}
        for row in rows:
            counts = {
                "sections": 1, "reference_correct": int(row["reference_correct"]),
                "repair_with_same_base": int(not row["reference_correct"] and bool(row["with_same_base"])),
                "repair_with_start_delete_combinations": int(not row["reference_correct"] and bool(row["with_any_base"])),
                "missing_later_contacts_in_single_rally_sections": len(row["missing_later_frames"]),
                "sections_with_distinct_local_candidate": int(bool(row["distinct_local_candidate_frames"])),
            }
            totals.update(counts)
            by_group.setdefault(row["group"], Counter()).update(counts)
        result[str(tolerance)] = {"counts": dict(totals), "by_group": by_group, "sections": rows}
    return result


def compare_outputs(
    reference_spans: Sequence[FixedSpan], selected: Mapping[tuple[str, int], LaterOption],
    labels: HumanLabels, fps: Mapping[str, float], groups: Mapping[str, str],
) -> dict[str, Any]:
    """Keep paired rally identities, all sections, local harm and group counts."""
    after_spans = tuple(option.span for option in selected.values())
    reference = {(span.fixture, span.span_id): span for span in reference_spans}
    overlaps = {identity: overlapping_rallies(span, labels) for identity, span in reference.items()}
    touched = Counter((fixture, rally.rally_id) for (fixture, _span_id), rallies in overlaps.items() for rally in rallies)
    output = {}
    for tolerance in (10, 5):
        before_rows = section_views(reference_spans, labels, fps, groups, tolerance)["fixed_side"]["sections"]
        after_rows = section_views(after_spans, labels, fps, groups, tolerance)["fixed_side"]["sections"]
        before_correct = {(row["fixture"], row["span_id"]): row["fully_correct"] for row in before_rows}
        changes = []
        totals = Counter()
        for identity, option in selected.items():
            before = reference[identity]
            if before == option.span:
                continue
            rallies = overlaps[identity]
            judgeable = (
                len(rallies) == 1 and touched[(before.fixture, rallies[0].rally_id)] == 1
                and overlapping_rallies(option.span, labels) == rallies
            )
            row = {**option_record(option), "judgeable": judgeable, "reference_correct": before_correct[identity]}
            totals["edited_sections"] += 1
            if judgeable:
                effects = contact_edit_effect(
                    before, option.span, rallies[0].frames, scale_base30_frames(tolerance, fps[identity[0]]),
                )
                row.update(effects)
                row["rally_id"] = rallies[0].rally_id
                totals.update(effects)
                totals["judgeable_edits"] += 1
                harmful = effects["labelled_contacts_lost"] > 0 or effects["unnecessary_added"] > 0
                totals["harmful_edits_in_already_wrong_sections"] += harmful and not before_correct[identity]
            else:
                totals["unjudgeable_edits"] += 1
            changes.append(row)
        by_group = {}
        by_video = {}
        for key, container, field in (("group", by_group, groups), ("fixture", by_video, None)):
            names = sorted(set(groups.values()) if field is not None else groups)
            for name in names:
                container[name] = paired_sections(
                    [row for row in before_rows if row[key] == name],
                    [row for row in after_rows if row[key] == name],
                )
        length_groups = {}
        for name, lower, upper in (("1–5", 1, 5), ("6–10", 6, 10), ("11–20", 11, 20), ("21+", 21, np.inf)):
            identities = {
                (row["fixture"], row["span_id"])
                for row in before_rows if lower <= row["labelled_contacts"] <= upper
            }
            length_groups[name] = paired_sections(
                [row for row in before_rows if (row["fixture"], row["span_id"]) in identities],
                [row for row in after_rows if (row["fixture"], row["span_id"]) in identities],
            )
        output[str(tolerance)] = {
            "paired": paired_sections(before_rows, after_rows), "by_group": by_group, "by_video": by_video,
            "by_labelled_length_after_prediction": length_groups,
            "reference_sections": before_rows, "sections": after_rows,
            "local_counts": dict(totals), "local_changes": changes,
        }
    return output
