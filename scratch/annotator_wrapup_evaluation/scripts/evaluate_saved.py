"""Describe the frozen broader detector with the existing one-to-one scorer."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd

from annotator.fps_constants import ScalingKind
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_closing_pass.scripts.summarise_metrics import (
    load_populations,
    load_stream,
    section_rows,
    selection_summary,
)
from scratch.contact_det_followup.scripts.prediction_io import read_json
from scratch.contact_det_followup.scripts.score_start_model import (
    apply_whole_rally_alternation,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "contact_det_closing_pass/results"


def run(annotations: Path, output: Path, limit: int | None) -> None:
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    populations, sides = load_populations(annotations)
    stream, fps = load_stream(BASELINE, "recommended")
    acceptance = read_json(BASELINE / "serve_followups/chosen_acceptance_broader.json.gz")
    reference = read_json(BASELINE / "metric_summary.json.gz")
    threshold = acceptance["frozen_policies"]["gap"]["comparison"]["threshold"]
    policies = {(row["fixture"], row["span_id"]): row for row in acceptance["rows"]}
    identities = {(span.fixture, span.span_id) for span in stream.spans}
    assert identities == set(policies) and len(identities) == 3982
    selected = {key for key, row in policies.items() if row["gap_score"] >= threshold}
    assert len(selected) == 784
    voted = apply_whole_rally_alternation(stream)
    fixtures = list(fps)[:limit]
    spans = tuple(span for span in stream.spans if span.fixture in fixtures)
    summaries, proposals, contacts, predictions, rally_rows = {}, [], [], [], []
    for population, labels in populations.items():
        summaries[population] = {}
        for tolerance in (10, 5):
            rows = section_rows(stream, labels, sides, fps, tolerance)
            kept = [row for row in rows if (row["fixture"], row["span_id"]) in selected]
            labelled_count = sum(map(len, labels.rallies.values()))
            whole = selection_summary(rows, labelled_count)
            chosen = selection_summary(kept, labelled_count)
            assert whole == reference["stages"]["recommended"][population][str(tolerance)]
            assert chosen == reference["selected"][population][str(tolerance)]
            summaries[population][str(tolerance)] = {"all": whole, "selected": chosen}
            by_rally = {(fixture, rally.rally_id): rally for fixture in fixtures for rally in labels.rallies[fixture]}
            for row, span in zip(rows, stream.spans, strict=True):
                if span.fixture not in fixtures:
                    continue
                record = {key: value for key, value in row.items() if key != "matches"}
                one_rally = row["overlapping_rallies"] == 1
                target_sides = sides.get((span.fixture, row["rally_id"]), ())
                matched_known = sum(target_sides[gt] is not None for gt, _, _ in row["matches"])
                record.update(
                    population=population, tolerance_base30=tolerance, fps=fps[span.fixture],
                    selected=(span.fixture, span.span_id) in selected,
                    matched=len(row["matches"]),
                    missing=row["labelled_contacts"] - len(row["matches"]) if one_rally else None,
                    extra=row["events"] - len(row["matches"]) if one_rally else None,
                    wrong_player=matched_known - row["voted_correct_sides"] if one_rally else None,
                    boundary_error=one_rally and not row["whole_rally_contained"],
                    multiple_rallies=row["overlapping_rallies"] > 1,
                    action=policies[span.fixture, span.span_id]["kind"],
                    selection_score=policies[span.fixture, span.span_id]["gap_score"],
                )
                if one_rally:
                    rally = by_rally[span.fixture, row["rally_id"]]
                    record.update(first_label_frame=rally.frames[0], last_label_frame=rally.frames[-1])
                proposals.append(record)
            total = Counter()
            for fixture in fixtures:
                human = []
                for rally in labels.rallies[fixture]:
                    for index, frame in enumerate(rally.frames):
                        human.append((frame, rally.rally_id, index, sides[fixture, rally.rally_id][index], len(rally.frames)))
                human.sort(key=lambda row: row[:3])
                events = voted.events_by_fixture[fixture]
                window = ScalingKind.FRAME_COUNT.scale(tolerance, fps[fixture])
                pairs = match_contacts([row[0] for row in human], [event.frame for event in events], window)
                matched_gt = {gt: (pred, offset) for gt, pred, offset in pairs}
                matched_pred = {pred: gt for gt, pred, _ in pairs}
                total.update(labelled=len(human), predicted=len(events), matched=len(pairs))
                for index, (frame, rally_id, contact_index, side, count) in enumerate(human):
                    pair = matched_gt.get(index)
                    event = events[pair[0]] if pair is not None else None
                    correct = event is not None and side is not None and event.predicted_side == side
                    total["side_correct"] += correct
                    contacts.append({
                        "population": population, "tolerance_base30": tolerance, "fixture": fixture, "fps": fps[fixture],
                        "rally_id": rally_id, "label_index": contact_index, "source_frame": frame, "labelled_contacts": count,
                        "position": "serve" if contact_index == 0 else "last" if contact_index == count - 1 else "middle",
                        "target_side": side, "matched": pair is not None, "player_correct": correct,
                        "prediction_frame": event.frame if event else None,
                        "predicted_side": event.predicted_side if event else None,
                        "offset_base30": pair[1] * 30 / fps[fixture] if pair else None,
                    })
                for index, event in enumerate(events):
                    gt_index = matched_pred.get(index)
                    predictions.append({
                        "population": population, "tolerance_base30": tolerance, "fixture": fixture, "fps": fps[fixture],
                        "source_frame": event.frame, "predicted_side": event.predicted_side, "matched": gt_index is not None,
                        "rally_id": human[gt_index][1] if gt_index is not None else None,
                        "label_index": human[gt_index][2] if gt_index is not None else None,
                    })
                correct_ids = {row["rally_id"] for row in rows if row["fixture"] == fixture and row["fully_correct"]}
                selected_ids = {row["rally_id"] for row in kept if row["fixture"] == fixture and row["fully_correct"]}
                fixture_spans = [span for span in spans if span.fixture == fixture]
                for rally in labels.rallies[fixture]:
                    touching = [span for span in fixture_spans if any(
                        span.start_frame <= frame < span.end_frame for frame in rally.frames
                    )]
                    rally_rows.append({
                        "population": population, "tolerance_base30": tolerance, "fixture": fixture, "fps": fps[fixture],
                        "rally_id": rally.rally_id, "first_frame": rally.frames[0], "last_frame": rally.frames[-1],
                        "labelled_contacts": len(rally.frames), "fully_correct": rally.rally_id in correct_ids,
                        "selected_correct": rally.rally_id in selected_ids, "overlapping_proposals": len(touching),
                        "contained": any(all(span.start_frame <= frame < span.end_frame for frame in rally.frames)
                                      for span in touching),
                    })
                print(population, tolerance, fixture, "matched", len(pairs), flush=True)
            if limit is None:
                expected = reference["contacts"][population][str(tolerance)]
                for key, value in total.items():
                    assert value == expected[key], (population, tolerance, key, value, expected[key])
            summaries[population][str(tolerance)]["contacts"] = dict(total)
    for name, records in (("proposals", proposals), ("contacts", contacts),
                          ("predictions", predictions), ("rallies", rally_rows)):
        pd.DataFrame(records).to_csv(output / f"{name}.csv.gz", index=False)
    write_json(output / "baseline.json.gz", {
        "schema": "annotator-failure-evaluation/1", "stage": "recommended", "selection_threshold": threshold,
        "fixtures": fixtures, "smoke": limit is not None, "summary": summaries, "seconds": perf_counter() - started,
    })
    print("Complete", round(perf_counter() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.annotations, args.output, args.limit)
