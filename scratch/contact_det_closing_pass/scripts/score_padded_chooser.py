"""Compare a refitted chooser with the saved local detector after identical padding."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.summarise_metrics import (
    full_stream_counts,
    load_populations,
    load_stream,
    section_rows,
    selection_summary,
)
from scratch.contact_det_followup.scripts.prediction_io import read_json

ROOT = Path(__file__).resolve().parents[1]


def run(predictions: Path, annotations: Path, results: Path, output: Path) -> None:
    source, fps = load_stream(results, "local")
    payload = read_json(predictions)
    assert payload["status"] == "complete" and payload["labels_read"] is False
    spans, events, new_fps = [], {}, {}
    for video in payload["videos"]:
        stream = restore_stream(video["output"])
        spans.extend(stream.spans)
        events.update(stream.events_by_fixture)
        new_fps[video["fixture"]] = float(video["fps"])
    assert new_fps == fps, "Compare the same complete video population on the same frame clocks"
    before = pad_contact_boundaries(source.spans, source.events_by_fixture, fps, preserve_membership=True)
    after = pad_contact_boundaries(spans, events, fps, preserve_membership=True)
    old_by_id = {(span.fixture, span.span_id): span for span in before.spans}
    new_by_id = {(span.fixture, span.span_id): span for span in after.spans}
    assert old_by_id.keys() == new_by_id.keys()
    acceptance = read_json(results / "serve_followups/chosen_acceptance_broader.json.gz")
    threshold = acceptance["frozen_policies"]["gap"]["comparison"]["threshold"]
    selected = {(row["fixture"], row["span_id"]) for row in acceptance["rows"] if row["gap_score"] >= threshold}
    populations, sides = load_populations(annotations)
    report: dict[str, Any] = {
        "schema": "contact-padded-chooser-comparison/1", "status": "complete",
        "data_status": "Previously examined 47 videos; predictions made without their labels",
        "baseline": "local chooser plus fixed_membership padding",
        "boundary_mode": "fixed_membership", "padding_base30": 10,
        "selected_identities_fixed": True, "selected_count": len(selected), "populations": {},
    }
    for population, labels in populations.items():
        comparisons = {}
        for tolerance in (10, 5):
            old_rows = section_rows(before, labels, sides, fps, tolerance)
            new_rows = section_rows(after, labels, sides, fps, tolerance)
            paired = paired_sections(old_rows, new_rows)
            old_selected = [row for row in old_rows if (row["fixture"], row["span_id"]) in selected]
            new_selected = [row for row in new_rows if (row["fixture"], row["span_id"]) in selected]
            by_video = {}
            for fixture in fps:
                by_video[fixture] = paired_sections([row for row in old_rows if row["fixture"] == fixture],
                                                   [row for row in new_rows if row["fixture"] == fixture])
            labelled = sum(map(len, labels.rallies.values()))
            comparisons[str(tolerance)] = {
                "paired": paired, "by_video": by_video,
                "contacts_before": full_stream_counts(before, labels, sides, fps, tolerance),
                "contacts_after": full_stream_counts(after, labels, sides, fps, tolerance),
                "selected_before": selection_summary(old_selected, labelled),
                "selected_after": selection_summary(new_selected, labelled),
                "selected_paired": paired_sections(old_selected, new_selected),
                "before_rows": old_rows, "after_rows": new_rows,
            }
            if population == "retained" and tolerance == 10:
                assert paired["correct_before"] == 1763
            print(population, tolerance, paired["correct_before"], paired["correct_after"],
                  "repairs", len(paired["repaired"]), "losses", len(paired["lost"]), flush=True)
        report["populations"][population] = comparisons
    report["changed_predictions"] = []
    for identity, old in old_by_id.items():
        new = new_by_id[identity]
        if old != new:
            report["changed_predictions"].append({
                "fixture": identity[0], "span_id": identity[1],
                "selected": identity in selected,
                "before": _span_record(old), "after": _span_record(new),
            })
    write_json(output, report)


def _span_record(span: FixedSpan) -> dict[str, Any]:
    return {"bounds": [span.start_frame, span.end_frame],
            "frames": [event.frame for event in span.events],
            "raw_sides": [event.predicted_side for event in span.events]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    run(args.predictions, args.annotations, args.results, args.output)


if __name__ == "__main__":
    main()
