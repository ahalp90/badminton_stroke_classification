"""Count label-guided repair opportunities without fitting or choosing a model."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_sections,
    section_result,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _subset_development,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as old
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"


def main() -> None:
    pack = prediction_io.load_development_predictions()
    census = prediction_io.read_json(ROOT / "results/missed_candidate_census.json.gz")
    output = {
        "schema": "contact-closing-repair-capacity/1",
        "evidence": "label-guided opportunities, not model predictions",
        "start_rule": "eligible existing add/replace actions, corrected matching and fixed side vote",
        "missing_contact_rule": (
            "one otherwise complete contained rally, one ideal event at the missing label "
            "with its labelled side, existing events and section boundaries preserved, fixed side vote"
        ),
        "populations": {},
    }
    for population, groups in (("development", frozenset(old.GROUPS)), ("validation", frozenset({"V"}))):
        videos, spans, events = _subset_development(pack, groups)
        raw = old._candidate_videos() if population == "development" else prediction_io.read_json(prediction_io.VALIDATION_PREDICTIONS)["videos"]
        rows = old.build_action_rows(build_candidate_rows(raw, default_group="V"))
        labels = load_human_labels(old.LABEL_PATH, videos)
        fps = {video.fixture: video.fps for video in videos}
        baseline = old.apply_selected_actions(spans, events, {})
        voted = old.apply_whole_rally_alternation(baseline)
        population_output = {"videos": len(videos), "sections": len(spans), "tolerances": {}}
        for tolerance in (10, 5):
            targets, alternatives = assign_targets(rows, spans, events, labels, fps, tolerance)
            baseline_rows = score_sections(spans, labels, fps, tolerance)
            baseline_correct = {
                (row["fixture"], row["rally_id"])
                for row in baseline_rows if row["side_rule_fully_correct"]
            }
            possible = set()
            actions = []
            for row in rows:
                if not targets[row.identity].whole_rally_correct:
                    continue
                span = alternatives[row.identity]
                scaled = ScalingKind.FRAME_COUNT.scale(tolerance, fps[span.fixture])
                result = section_result(span, labels, scaled)
                if result["side_rule_fully_correct"]:
                    possible.add((span.fixture, result["rally_id"]))
                    actions.append(row.identity)
            record = {
                "original_scorer_baseline": len(old._fully_correct_ids(voted, labels, fps, tolerance_at_30_fps=tolerance)),
                "corrected_baseline": len(baseline_correct),
                "start_possible_repairs": sorted(possible - baseline_correct),
                "start_successful_actions": sorted(actions),
            }
            if population == "development":
                missed = {
                    (row["fixture"], row["frame"]): row
                    for row in census["tolerances"][str(tolerance)]["missed"]
                }
                categories = Counter()
                ideal_repairs = []
                for span, result in zip(spans, baseline_rows, strict=True):
                    if not result["whole_rally_contained"] or result["events"] + 1 != result["labelled_contacts"]:
                        continue
                    if len(result["matches"]) != result["events"]:
                        continue
                    rally = next(rally for rally in labels.rallies[span.fixture] if rally.rally_id == result["rally_id"])
                    matched_gt = {pair[0] for pair in result["matches"]}
                    index = next(index for index in range(len(rally.frames)) if index not in matched_gt)
                    frame = rally.frames[index]
                    event = FixedEvent(span.fixture, frame, 1.0, labels.target_sides[(span.fixture, frame)])
                    revised = replace(span, events=tuple(sorted((*span.events, event), key=lambda event: event.frame)))
                    scaled = ScalingKind.FRAME_COUNT.scale(tolerance, fps[span.fixture])
                    if not section_result(revised, labels, scaled)["side_rule_fully_correct"]:
                        continue
                    census_row = missed.get((span.fixture, frame))
                    category = census_row["category"] if census_row else "matched_elsewhere_in_full_stream"
                    position = "first" if index == 0 else "later"
                    categories[(position, category)] += 1
                    ideal_repairs.append({"fixture": span.fixture, "span_id": span.span_id, "rally_id": rally.rally_id,
                                          "frame": frame, "position": position, "category": category})
                record["ideal_single_contact_repairs"] = ideal_repairs
                record["ideal_single_contact_counts"] = [
                    {"position": position, "category": category, "count": count}
                    for (position, category), count in sorted(categories.items())
                ]
            population_output["tolerances"][str(tolerance)] = record
            print(population, tolerance, "original", record["original_scorer_baseline"],
                  "corrected", len(baseline_correct), "start opportunities", len(possible - baseline_correct), flush=True)
        output["populations"][population] = population_output
    write_json(ROOT / "results/repair_capacity.json.gz", output)


if __name__ == "__main__":
    main()
