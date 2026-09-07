"""Measure a frozen visual serve check on the outputs accepted by the tree."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _partition_metrics,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import ROOT
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)


def visual_gate(parsed: dict | None, anchor: int, source_start: int, fps: float) -> tuple[bool, str, int | None]:
    """Keep the tree decision when the visual answer is missing or uncertain."""
    if parsed is None:
        return True, "missing_or_malformed_keep_tree", None
    state = parsed["serve_state"]
    if state == "unclear":
        return True, "unclear_keep_tree", None
    if state != "visible":
        return False, "service_not_shown", None
    source_contact = source_start + parsed["contact_frame"]
    agrees = abs(source_contact - anchor) <= scale_base30_frames(10, fps)
    return agrees, "timing_agrees" if agrees else "timing_disagrees", source_contact


def run(routing_path: Path, attempts_root: Path) -> None:
    routing = json.loads(routing_path.read_text())
    cases = routing["cases"]
    decisions = []
    model = None
    for case in cases:
        attempt_path = attempts_root / "qwen3-8" / f"{case['case_id']}.json"
        attempt = json.loads(attempt_path.read_text()) if attempt_path.exists() else None
        if attempt is not None:
            model = attempt["model"]
        parsed = None if attempt is None else attempt["parsed_response"]
        keep, reason, source_contact = visual_gate(
            parsed, case["anchor_frame"], case["source_start_frame"], case["fps"],
        )
        decisions.append({
            **case, "keep_tree_acceptance": keep, "gate_reason": reason,
            "attempt_present": attempt is not None,
            "vlm_source_contact_frame": source_contact,
            "raw_response": None if attempt is None else attempt["raw_response"],
            "parsed_response": parsed,
            "parser_error": None if attempt is None else attempt["parser_error"],
            "generation_error": None if attempt is None else attempt["generation_error"],
            "elapsed_seconds": None if attempt is None else attempt["elapsed_seconds"],
        })
    results = ROOT / "results/followups"
    write_json(results / "vlm_acceptance_decisions.json.gz", {
        "schema": "contact-visual-acceptance-decisions/1", "status": "complete",
        "policy": "Keep tree on unclear or malformed replies; otherwise require visible serve timing within ten base-30 frames of its first contact",
        "decision_uses_labels": False, "cases": decisions,
    })
    saved = prediction_io.read_json(results / "gap_acceptance_result.json.gz")
    natural = {(row["fixture"], row["span_id"]): row for row in decisions if row["kind"] == "natural"}
    rows = []
    for row in saved["rows"]:
        decision = natural.get((row["fixture"], row["span_id"]))
        score = row["gap_score"] if decision is None or decision["keep_tree_acceptance"] else -1.0
        rows.append({**row, "visual_score": score, "visual_call": decision is not None})
    threshold = saved["policies"]["gap"]["threshold"]
    routed_rows = [row for row in rows if row["visual_call"]]
    by_case = {row["case_id"]: row for row in decisions}
    controls = []
    for shifted in (row for row in decisions if row["kind"] == "shifted"):
        original = by_case[shifted["paired_natural_id"]]
        first, second = original["vlm_source_contact_frame"], shifted["vlm_source_contact_frame"]
        controls.append({
            "fixture": shifted["fixture"], "span_id": shifted["span_id"],
            "source_window_shift": shifted["source_start_frame"] - original["source_start_frame"],
            "same_gate": original["keep_tree_acceptance"] == shifted["keep_tree_acceptance"],
            "both_timed": first is not None and second is not None,
            "absolute_timing_difference": None if first is None or second is None else abs(first - second),
        })
    output = {
        "schema": "contact-visual-acceptance-comparison/1", "status": "complete",
        "evidence_status": "Naturally routed development calls on the three existing scene fixtures; no new visibility truth",
        "detector": "session_start", "tree_acceptance": "frozen_gap", "threshold": threshold,
        "contact_output_unchanged": True, "natural_calls": len(natural), "shifted_controls": controls,
        "model": model,
        "observation_counts": {
            "requested": len(decisions),
            "attempted": sum(row["attempt_present"] for row in decisions),
            "generation_errors": sum(row["generation_error"] is not None for row in decisions),
            "parser_errors": sum(row["parser_error"] is not None for row in decisions),
        },
        "response_counts": dict(Counter(row["gate_reason"] for row in decisions if row["kind"] == "natural")),
        "routed_population": {name: _partition_metrics(routed_rows, key, threshold)
                              for name, key in (("tree", "gap_score"), ("tree_and_visual", "visual_score"))},
        "full_development_population": {name: _partition_metrics(rows, key, threshold)
                                        for name, key in (("tree", "gap_score"), ("tree_and_visual", "visual_score"))},
        "prediction_seconds": sum(row["elapsed_seconds"] or 0 for row in decisions),
        "prepare_seconds": routing["prepare_seconds"],
        "rows": rows,
    }
    write_json(results / "vlm_acceptance_result.json.gz", output)
    for name, metrics in output["routed_population"].items():
        print(name, "accepted", metrics["accepted_count"], metrics["by_tolerance"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--attempts-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.routing, args.attempts_root)
