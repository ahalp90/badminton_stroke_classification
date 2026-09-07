"""Count small repairs available in the saved selected development clips.

Labels identify possible repairs; this count is not a working correction policy.
Candidate timestamps, selected identities and the current choice map stay fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import joblib

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    section_result,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    apply_options,
    option_record,
)
from scratch.contact_det_closing_pass.scripts.run_padded_target_census import (
    DEFAULT_CURRENT,
    DEFAULT_PREPARED,
    ROOT,
    _fixture_labels,
    _timing_possible,
    evaluate_option_with_padding,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    classify_section_result,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

EDIT_NAMES = {(1, 0): "deletion", (0, 1): "insertion", (1, 1): "replacement"}


def frame_delta(before: Sequence[int], after: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sorted(set(before) - set(after))), tuple(sorted(set(after) - set(before)))


def edit_category(
    removed: Sequence[int], added: Sequence[int], label_frames: Sequence[int],
    tolerance: int, after_frames: Sequence[int],
) -> str:
    """Describe label-relative geometry, without claiming an extra event is physically false."""
    delta = EDIT_NAMES[len(removed), len(added)]
    if delta == "deletion":
        if removed[0] < min(label_frames):
            return "deletion_before_first_label"
        if removed[0] > max(label_frames):
            return "deletion_after_last_label"
        return "deletion_interior"
    if delta == "insertion":
        return "insertion_first" if added[0] == min(after_frames) else "insertion_later"
    shared_label = any(
        abs(frame - removed[0]) <= tolerance and abs(frame - added[0]) <= tolerance
        for frame in label_frames
    )
    return "replacement_timing" if shared_label else "replacement_extra_plus_missing"


def _unique_options(options: Sequence[LaterOption], current: LaterOption) -> list[tuple]:
    before = [event.frame for event in current.span.events]
    seen: set[FixedSpan] = set()
    result = []
    for option in options:
        if option.span in seen:
            continue
        seen.add(option.span)
        removed, added = frame_delta(before, [event.frame for event in option.span.events])
        delta = EDIT_NAMES.get((len(removed), len(added)))
        if delta is not None:
            result.append((option, removed, added, delta))
    return result


def _span_record(span: FixedSpan) -> dict[str, Any]:
    return {"bounds": [span.start_frame, span.end_frame],
            "frames": [event.frame for event in span.events],
            "raw_sides": [event.predicted_side for event in span.events]}


def _fixture(fixture: str, population: Any, current: dict, options: dict, accepted: list, labels: Any) -> list[dict]:
    fps = population.fps[fixture]
    tolerance = scale_base30_frames(10, fps)
    spans = tuple(span for span in population.spans if span.fixture == fixture)
    choices = {identity: option for identity, option in current.items() if identity[0] == fixture}
    full_events = population.events[fixture]
    raw = apply_options(spans, {fixture: full_events}, choices)
    padded = pad_contact_boundaries(raw.spans, raw.events_by_fixture, {fixture: fps}, preserve_membership=True)
    before_by_id = {span.span_id: span for span in padded.spans}
    labels = _fixture_labels(labels, fixture)
    results = []
    calls = 0
    for acceptance in accepted:
        identity = fixture, acceptance["span_id"]
        before = before_by_id[identity[1]]
        baseline = section_result(before, labels, tolerance)
        status = acceptance["judgements"]["10"]["outcome"]
        # Preserve saved uncertainty, including any uncertain-anchor abstentions.
        if status != "unjudgeable":
            fixed = {**baseline, "fully_correct": baseline["side_rule_fully_correct"]}
            assert classify_section_result(fixed, labels)["outcome"] == status, identity
        alternatives = _unique_options(options[identity], choices[identity])
        row = {"fixture": fixture, "span_id": identity[1], "group": population.groups[fixture],
               "status": status, "before": _span_record(before), "before_result": baseline,
               "allowed_edits": len(alternatives), "damaging_edit_available": False,
               "repair_types": [], "repair_categories": [], "examples": []}
        examples = {}
        if status != "unjudgeable":
            for option, removed, added, delta in alternatives:
                # Padding can admit a previously non-overlapping one-contact rally.
                # Timing/count pruning must therefore consider every rally in this video.
                possible = any(_timing_possible(option.span, rally, tolerance) for rally in labels.rallies[fixture])
                if not possible:
                    row["damaging_edit_available"] |= status == "correct"
                    continue
                _, after, scored, secondary = evaluate_option_with_padding(
                    option, spans, full_events, choices, labels, fps,
                )
                calls += 1
                correct = scored["side_rule_fully_correct"]
                row["damaging_edit_available"] |= status == "correct" and not correct
                if status != "wrong" or not correct:
                    continue
                rally = next(rally for rally in labels.rallies[fixture] if rally.rally_id == scored["rally_id"])
                category = edit_category(removed, added, rally.frames, tolerance, [event.frame for event in after.events])
                row["repair_types"].append(delta)
                row["repair_categories"].append(category)
                if category not in examples:
                    examples[category] = {
                        "type": delta, "category": category, "removed": removed, "added": added,
                        "action": option_record(option), "after": _span_record(after),
                        "label_frames": rally.frames, "result_10": scored, "result_5": secondary,
                    }
        row["repair_types"] = sorted(set(row["repair_types"]))
        row["repair_categories"] = sorted(set(row["repair_categories"]))
        row["examples"] = list(examples.values())
        results.append(row)
    print(f"{fixture}: {len(results)} selected, {sum(bool(row['repair_types']) for row in results)} repairable, "
          f"{calls} scored alternatives", flush=True)
    return results


def _summary(rows: list[dict]) -> dict[str, Any]:
    return {
        "selected": len(rows), "baseline": dict(Counter(row["status"] for row in rows)),
        "wrong_repairable": sum(bool(row["repair_types"]) for row in rows),
        "repair_types": dict(Counter(kind for row in rows for kind in row["repair_types"])),
        "repair_categories": dict(Counter(kind for row in rows for kind in row["repair_categories"])),
        "correct_with_allowed_edit": sum(row["status"] == "correct" and row["allowed_edits"] > 0 for row in rows),
        "correct_with_damaging_edit": sum(row["damaging_edit_available"] for row in rows),
    }


def endpoint_gaps(rows: list[dict], fps: dict[str, float]) -> dict[str, Any]:
    """Compare the final predicted gap in tail-repair cases and currently good clips."""
    gaps: dict[str, list[float]] = {"tail_repair": [], "correct": []}
    for row in rows:
        frames = row["before"]["frames"]
        if len(frames) < 2:
            continue
        gap = (frames[-1] - frames[-2]) * 30 / fps[row["fixture"]]
        if row["status"] == "correct":
            gaps["correct"].append(gap)
        if "deletion_after_last_label" in row["repair_categories"]:
            gaps["tail_repair"].append(gap)
    return {
        kind: {"count": len(values), "min": min(values), "median": median(values), "max": max(values)}
        for kind, values in gaps.items() if values
    }


def run(output_root: Path, jobs: int, limit_videos: int | None) -> None:
    started = perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    prepared = joblib.load(DEFAULT_PREPARED)
    population, pool = prepared["base_population"], prepared["options"]
    acceptance = prediction_io.read_json(ROOT / "results/serve_followups/chosen_acceptance_development.json.gz")
    threshold = acceptance["policies"]["gap"]["comparison"]["threshold"]
    accepted = [row for row in acceptance["rows"] if row["gap_score"] >= threshold]
    assert len(accepted) == acceptance["policies"]["gap"]["comparison"]["accepted_count"]
    current = restore_choices(pool, prediction_io.read_json(DEFAULT_CURRENT)["selected_actions"])
    fixtures = sorted({row["fixture"] for row in accepted})
    if limit_videos is not None:
        fixtures = fixtures[:limit_videos]
    identities = {(row["fixture"], row["span_id"]) for row in accepted}
    options = defaultdict(list)
    for option in pool:
        if option.base.identity in identities:
            options[option.base.identity].append(option)
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    tasks = []
    for fixture in fixtures:
        fixture_rows = [row for row in accepted if row["fixture"] == fixture]
        fixture_options = {identity: values for identity, values in options.items() if identity[0] == fixture}
        tasks.append((fixture, population, current, fixture_options, fixture_rows, labels))
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
        results = joblib.Parallel()(joblib.delayed(_fixture)(*args) for args in tasks)
    rows = [row for result in results for row in result]
    summary = _summary(rows)
    if limit_videos is None:
        assert summary["baseline"] == {"correct": 448, "wrong": 119, "unjudgeable": 3}
    write_json(output_root / "census.json.gz", {
        "schema": "contact-selected-repair-headroom/1", "status": "complete" if limit_videos is None else "smoke",
        "evidence": "Label-guided opportunity in existing candidates, not achieved correction performance",
        "tolerance_base30": 10, "secondary_tolerance_base30": 5, "selected_threshold": threshold,
        "boundary_mode": "fixed_membership", "summary": summary,
        "fps": {fixture: population.fps[fixture] for fixture in fixtures},
        "by_group": {group: _summary([row for row in rows if row["group"] == group]) for group in "ABCD"},
        "final_gap_base30": endpoint_gaps(rows, population.fps),
        "proposals": rows, "seconds": perf_counter() - started,
    })
    print(summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()
    run(args.output_root, args.jobs, args.limit_videos)


if __name__ == "__main__":
    main()
