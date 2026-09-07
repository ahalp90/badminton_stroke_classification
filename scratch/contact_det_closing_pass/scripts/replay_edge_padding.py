"""Recount independent-edge padding using saved predictions and source labels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.summarise_metrics import (
    load_populations,
    load_stream,
    section_rows,
    selection_summary,
)
from scratch.contact_det_followup.scripts.prediction_io import read_json
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import ContactStreams

ROOT = Path(__file__).resolve().parents[1]


def pad_edges(
    spans: Sequence[FixedSpan], events: Mapping[str, Sequence[FixedEvent]], fps: Mapping[str, float],
) -> ContactStreams:
    """Retain each proposed edge up to the nearest outside predicted contact."""
    proposed = pad_contact_boundaries(spans, events, fps)
    revised = []
    for original, candidate in zip(spans, proposed.spans, strict=True):
        start, end = candidate.start_frame, candidate.end_frame
        for event in events[original.fixture]:
            if start <= event.frame < original.start_frame:
                start = event.frame + 1
            elif original.end_frame <= event.frame < end:
                end = event.frame
        revised.append(replace(original, start_frame=start, end_frame=end))
    return ContactStreams(tuple(revised), proposed.events_by_fixture)


def run(annotations: Path, results: Path, output: Path) -> None:
    source, fps = load_stream(results, "local")
    before = pad_contact_boundaries(source.spans, source.events_by_fixture, fps, preserve_membership=True)
    after = pad_edges(source.spans, source.events_by_fixture, fps)
    assert before.events_by_fixture == after.events_by_fixture
    changed = []
    for old, new in zip(before.spans, after.spans, strict=True):
        assert old.events == new.events
        contained = tuple(event for event in after.events_by_fixture[new.fixture]
                          if new.start_frame <= event.frame < new.end_frame)
        assert contained == new.events
        if (old.start_frame, old.end_frame) != (new.start_frame, new.end_frame):
            changed.append({"fixture": old.fixture, "span_id": old.span_id,
                            "before_bounds": [old.start_frame, old.end_frame],
                            "after_bounds": [new.start_frame, new.end_frame]})
    populations, sides = load_populations(annotations)
    acceptance = read_json(results / "serve_followups/chosen_acceptance_broader.json.gz")
    threshold = acceptance["frozen_policies"]["gap"]["comparison"]["threshold"]
    selected = {(row["fixture"], row["span_id"]) for row in acceptance["rows"] if row["gap_score"] >= threshold}
    report: dict[str, Any] = {
        "schema": "contact-edge-padding-replay/1", "status": "complete",
        "input": "followups/local_broader_predictions.json.gz",
        "padding_base30": 10, "events_unchanged": True,
        "videos": len(fps), "sections": len(before.spans), "changed_sections": changed,
        "selected_identities_fixed": True, "selected_count": len(selected), "populations": {},
    }
    for population, labels in populations.items():
        results_by_tolerance = {}
        for tolerance in (10, 5):
            old_rows = section_rows(before, labels, sides, fps, tolerance)
            new_rows = section_rows(after, labels, sides, fps, tolerance)
            paired = paired_sections(old_rows, new_rows)
            old_selected = [row for row in old_rows if (row["fixture"], row["span_id"]) in selected]
            new_selected = [row for row in new_rows if (row["fixture"], row["span_id"]) in selected]
            labelled = sum(map(len, labels.rallies.values()))
            results_by_tolerance[str(tolerance)] = {
                "paired": paired,
                "selected_before": selection_summary(old_selected, labelled),
                "selected_after": selection_summary(new_selected, labelled),
                "changed_rows": [
                    {"before": old, "after": new}
                    for old, new in zip(old_rows, new_rows, strict=True)
                    if (old["start_frame"], old["end_frame"]) != (new["start_frame"], new["end_frame"])
                ],
            }
            if population == "retained" and tolerance == 10:
                assert paired["correct_before"] == 1763, paired
            print(population, tolerance, paired["correct_before"], paired["correct_after"],
                  "repairs", len(paired["repaired"]), "losses", len(paired["lost"]), flush=True)
        report["populations"][population] = results_by_tolerance
    write_json(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    run(args.annotations, args.results, args.output)


if __name__ == "__main__":
    main()
