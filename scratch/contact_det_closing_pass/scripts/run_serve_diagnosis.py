"""Count development opening failures and supported local deletion opportunities."""

from __future__ import annotations

from collections import Counter, defaultdict
from time import perf_counter

import joblib

from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.local_deletion import (
    deletion_opportunities,
)
from scratch.contact_det_closing_pass.scripts.run_followup_residual_diagnosis import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SCORE_PATH,
    _candidate_pool,
    _early_window_rows,
    _load_early_windows,
    _load_evidence,
    _nearby,
)
from scratch.contact_det_closing_pass.scripts.run_serve_followups import (
    OUTPUT,
    ROOT,
    development_streams,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)


def run() -> None:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population = prepared["base_population"]
    streams, labels, fps, _groups = development_streams()
    current = streams["recommended"]
    candidates, _sources = _candidate_pool(population, prepared["later_candidates"])
    evidence = _load_evidence(population, set(fps), DEFAULT_FEATURE_ROOT, DEFAULT_SCORE_PATH)
    windows = _load_early_windows(
        ROOT.parent / "contact_det_full_ds_fit/raw/training_rally_start_inputs/rally_start_training_inputs.json.gz", set(fps),
    )
    offered = defaultdict(set)
    for option in prepared["options"]:
        offered[option.base.identity].add(tuple(event.frame for event in option.span.events))
    results = {}
    for tolerance in (10, 5):
        matched = score_contacts(current.events_by_fixture, labels, fps, tolerance)
        matched_firsts = {
            (video["fixture"], pair[2])
            for video in matched["by_video"] for pair in video["pairs"] if pair[3]
        }
        rows, counts = [], Counter()
        for fixture in fps:
            scaled = scale_base30_frames(tolerance, fps[fixture])
            video = evidence[fixture]
            for rally in labels.rallies[fixture]:
                if (fixture, rally.rally_id) in matched_firsts:
                    continue
                frame = rally.frames[0]
                nearby_sections = [
                    (span.fixture, span.span_id) for span in population.spans
                    if span.fixture == fixture and span.start_frame - scaled <= frame < span.end_frame + scaled
                ]
                # Saved early windows can begin before their original section.
                for identity, intervals in windows.items():
                    if identity[0] == fixture and any(
                        interval.prefix_start_frame - scaled <= frame < interval.fixed_contact_frame + scaled
                        for interval in intervals
                    ) and identity not in nearby_sections:
                        nearby_sections.append(identity)
                score_rows = _nearby(video.score_rows, video.score_frames, frame, scaled)
                physical = _nearby(video.physical_rows, video.physical_frames, frame, scaled)
                shortlisted = sorted({
                    candidate.frame for identity in nearby_sections for candidate in candidates.get(identity, ())
                    if abs(candidate.frame - frame) <= scaled
                })
                early_windows = tuple(interval for identity in nearby_sections for interval in windows.get(identity, ()))
                in_window = _early_window_rows(score_rows, early_windows)
                if shortlisted:
                    category = "shortlisted_not_chosen"
                elif len(in_window):
                    category = "scored_early_window_not_shortlisted"
                elif len(score_rows):
                    category = "scored_outside_early_windows"
                elif len(physical):
                    category = "prepared_but_not_scored"
                else:
                    category = "no_prepared_evidence"
                counts[category] += 1
                rows.append({
                    "fixture": fixture, "rally_id": rally.rally_id, "gt_frame": frame, "category": category,
                    "nearby_sections": nearby_sections, "shortlisted_frames": shortlisted,
                    "scored_frames": [int(value) for value in score_rows["frame"]],
                    "early_window_frames": [int(value) for value in in_window["frame"]],
                    "physical_frames": [int(value) for value in physical["frame"]],
                })
        deletions = deletion_opportunities(current.spans, labels, fps, offered, tolerance)
        results[str(tolerance)] = {"missed_serves": {"counts": dict(counts), "rows": rows}, "deletions": deletions}
        print(tolerance, "missed serves", dict(counts), "deletions", deletions["counts"], flush=True)
    write_json(OUTPUT / "development_diagnosis.json.gz", {
        "schema": "contact-serve-diagnosis/1", "status": "complete", "population": "development",
        "prediction_changes": False, "by_tolerance": results, "seconds": perf_counter() - started,
        "interpretation": "Opportunity uses labels only for diagnosis; physical presence is not a visibility label",
        "deletion_unknowns": "Unmatched events outside retained label support are unknown, including possible false leads",
    })


if __name__ == "__main__":
    run()
