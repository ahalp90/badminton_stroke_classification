"""Replay simple opening replacements as keep, preserving all other decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    write_json,
)
from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "opening_sides_and_physics"
EFFECT_FIELDS = (
    "newly_matched_contacts", "labelled_contacts_lost", "previously_matched_predictions_removed",
    "unnecessary_added", "unnecessary_removed", "first_contact_recovered", "first_contact_lost",
)


def section_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return row["fixture"], row["span_id"]


def option_key(row: Mapping[str, Any]) -> tuple[str, int, str, int | None, int | None]:
    return (*section_key(row), row["kind"], row["candidate_frame"], row["deleted_frame"])


def replay_choices(
    options: Sequence[Mapping[str, Any]], scores: Sequence[float], selected: Sequence[Mapping[str, Any]],
    cancel_simple_replace: bool,
) -> list[dict[str, Any]]:
    """Use the keep option and its score only for an already-selected simple replace."""
    indexed = {option_key(option): (option, score) for option, score in zip(options, scores, strict=True)}
    output = []
    for choice in selected:
        key = option_key(choice)
        cancelled = cancel_simple_replace and choice["kind"] == "replace"
        if cancelled:
            key = (*section_key(choice), "keep", None, None)
        option, score = indexed[key]
        output.append({**option, "score": float(score), "cancelled_simple_replace": cancelled})
    return output


def replay_rows(
    baseline: Sequence[Mapping[str, Any]], edited: Sequence[Mapping[str, Any]],
    choices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore complete baseline rows for cancelled edits, including original bounds."""
    before = {section_key(row): row for row in baseline}
    cancelled = {section_key(row) for row in choices if row["cancelled_simple_replace"]}
    return [dict(before[section_key(row)] if section_key(row) in cancelled else row) for row in edited]


def replay_harm(
    original: Mapping[str, Any], choices: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cancelled = {section_key(row) for row in choices if row["cancelled_simple_replace"]}
    correct = {section_key(row) for row in baseline if row["fully_correct"]}
    retained = [row for row in original["selected"] if section_key(row) not in cancelled]
    counts: Counter[str] = Counter()
    for row in retained:
        counts["selected_edits"] += 1
        if not row["judgeable"]:
            counts["unjudgeable_edits"] += 1
            continue
        counts["judgeable_edits"] += 1
        counts["beneficial_contact_edits"] += row["beneficial_contact_edit"]
        counts["unsuccessful_edits_in_already_wrong_sections"] += (
            not row["beneficial_contact_edit"] and section_key(row) not in correct
        )
        for field in EFFECT_FIELDS:
            counts[field] += row[field]
    return {"counts": dict(counts), "selected": retained, "loss_definition": original["loss_definition"]}


def compare_population(
    baseline: Mapping[str, Any], reference: Mapping[str, Any], choices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"selected_actions": choices, "evaluation": {}, "harm": {}}
    result["action_counts"] = dict(Counter(row["kind"] for row in choices))
    for tolerance in ("10", "5"):
        original = reference["evaluation"][tolerance]
        result["evaluation"][tolerance] = {}
        for view, edited_name in (("fixed_side", "edited_fixed_side"), ("raw", "edited_raw")):
            before = baseline[tolerance][view]["sections"]
            earlier = original[edited_name]["sections"]
            after = replay_rows(before, earlier, choices)
            result["evaluation"][tolerance][view] = {
                "sections": after, "versus_baseline": paired_sections(before, after),
                "versus_reference": paired_sections(earlier, after),
                "by_video": {},
            }
            for fixture in sorted({row["fixture"] for row in before}):
                result["evaluation"][tolerance][view]["by_video"][fixture] = paired_sections(
                    [row for row in before if row["fixture"] == fixture],
                    [row for row in after if row["fixture"] == fixture],
                )
        result["harm"][tolerance] = replay_harm(
            reference["harm"][tolerance], choices, baseline[tolerance]["fixed_side"]["sections"],
        )
    return result


def run(output_root: Path) -> None:
    original = read_json(ROOT / "results/whole_rally_result.json.gz")
    reference = original["development_descriptive"][VARIANT]
    d_choices = replay_choices(original["development_options"], reference["scores"],
                               reference["selected_actions"], True)
    development = compare_population(original["development_baseline"], reference, d_choices)
    comparison = development["evaluation"]["10"]["fixed_side"]["versus_reference"]
    cancel = comparison["correct_after"] >= comparison["correct_before"]
    policy = {
        "schema": "contact-closing-broad-policy/1", "status": "frozen",
        "reference_commit": "24e4256", "variant": VARIANT, "minimum_score": 0.0,
        "cancel_simple_replace": cancel,
        "selection": "development fully correct rallies at ±10; prefer fewer edits on a tie",
        "development_comparison": comparison,
        "validation_used_for_policy": False, "broader_labels_used": False,
    }
    write_json(output_root / "broader_action_policy.json.gz", policy)
    print("Frozen action policy", "cancel simple replace" if cancel else "retain reference", flush=True)
    for tolerance in ("10", "5"):
        paired = development["evaluation"][tolerance]["fixed_side"]["versus_baseline"]
        print("D", tolerance, paired["correct_after"], len(paired["repaired"]), len(paired["lost"]), flush=True)

    validation_predictions = read_json(ROOT / "results/whole_rally_predictions.json.gz")
    selected = validation_predictions["variants"][VARIANT]
    v_choices = replay_choices(validation_predictions["options"], selected["scores"],
                               selected["selected_actions"], True)
    validation = compare_population(original["validation_baseline"], original["validation"][VARIANT], v_choices)
    for tolerance in ("10", "5"):
        paired = validation["evaluation"][tolerance]["fixed_side"]["versus_baseline"]
        print("V diagnostic", tolerance, paired["correct_after"], len(paired["repaired"]), len(paired["lost"]), flush=True)
    write_json(output_root / "simple_replacement_replay.json.gz", {
        "schema": "contact-closing-simple-replacement-replay/1", "status": "complete",
        "rule": "selected simple replace becomes keep; every other decision unchanged; no reselection",
        "frozen_policy": policy, "development": development, "validation_diagnostic": validation,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    run(parser.parse_args().output_root)


if __name__ == "__main__":
    main()
