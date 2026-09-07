"""Recount serve discovery, proposed starts and attribution in saved detector streams."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_sections,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    GROUPS,
    _subset_development,
)
from scratch.contact_det_closing_pass.scripts.serve_metrics import (
    analyse_serves,
    compare_serves,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import ContactStreams
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
OUTPUT = ROOT / "results/serve_followups"
VARIANTS = ("original", "preceding", "guarded_only", "recommended", "wider_early")


def development_streams() -> tuple[dict, Any, dict, dict]:
    """Load archived development choices and replay only the frozen edge rule."""
    pack = prediction_io.load_development_predictions()
    videos, spans, events = _subset_development(pack, frozenset(GROUPS))
    fps = {video.fixture: video.fps for video in videos}
    groups = {fixture: pack.group_by_fixture[fixture] for fixture in fps}
    preceding = restore_stream(prediction_io.read_json(
        ROOT / "results/later/later_margin_predictions.json.gz"
    )["outputs"])
    streams = {"original": ContactStreams(spans, events), "preceding": preceding}
    streams["guarded_only"] = pad_contact_boundaries(
        preceding.spans, preceding.events_by_fixture, fps, preserve_membership=True,
    )
    for name, saved_name in (("recommended", "local"), ("wider_early", "early")):
        archived = ROOT / f"raw/followups/development_predictions/{saved_name}_predictions.json.gz"
        reference = ROOT / f"raw/serve_followups/reference_development_predictions/{saved_name}_predictions.json.gz"
        source = reference if reference.exists() else archived
        stream = restore_stream(prediction_io.read_json(source)["outputs"])
        streams[name] = pad_contact_boundaries(stream.spans, stream.events_by_fixture, fps, preserve_membership=True)
    labels = load_human_labels(start.LABEL_PATH, videos)
    return streams, labels, fps, groups


def broader_streams() -> tuple[dict, Any, dict, dict]:
    """Load the frozen 47-video streams without rebuilding any detector."""
    paths = {
        "preceding": "later/later_broader_predictions.json.gz",
        "guarded_only": "followups/session_start_boundary_broader_predictions_fixed_membership.json.gz",
        "recommended": "followups/local_boundary_broader_predictions_fixed_membership.json.gz",
        "wider_early": "followups/early_boundary_broader_predictions_fixed_membership.json.gz",
    }
    streams, fps = {}, {}
    for name, relative in paths.items():
        records = prediction_io.read_json(ROOT / "results" / relative)["videos"]
        spans, events = [], {}
        for record in records:
            stream = restore_stream(record["output"])
            spans.extend(stream.spans)
            events.update(stream.events_by_fixture)
            fps[record["fixture"]] = float(record["fps"])
        streams[name] = ContactStreams(tuple(spans), events)
    original = prediction_io.load_frozen_test_predictions()
    streams["original"] = ContactStreams(original.spans, original.events_by_fixture)
    return streams, test_labels(), fps, dict.fromkeys(fps, "broader")


def recount(population: str) -> dict[str, Any]:
    started = perf_counter()
    streams, labels, fps, groups = development_streams() if population == "development" else broader_streams()
    results = {}
    for name in VARIANTS:
        stream = streams[name]
        if set(stream.events_by_fixture) != set(fps):
            raise ValueError(f"{population}/{name}: prediction and scoring populations differ")
        tolerances = {}
        for tolerance in (10, 5):
            result = analyse_serves(stream, labels, fps, tolerance)
            sections = score_sections(stream.spans, labels, fps, tolerance)
            result["complete_sections"] = sum(row["side_rule_fully_correct"] for row in sections)
            result["sections"] = sections
            tolerances[str(tolerance)] = result
        results[name] = tolerances
        print(population, name, {key: value["total"] for key, value in tolerances.items()}, flush=True)
    expected = {"development": (1209, 958), "broader": (1763, 1430)}[population]
    observed = tuple(results["recommended"][str(tolerance)]["complete_sections"] for tolerance in (10, 5))
    if observed != expected:
        raise ValueError(f"Saved recommended totals differ: {observed}, expected {expected}")
    comparisons = {}
    for reference in ("preceding", "guarded_only", "recommended"):
        for variant in ("recommended", "wider_early"):
            if reference != variant:
                comparisons[f"{variant}_vs_{reference}"] = {
                    str(tolerance): compare_serves(results[reference][str(tolerance)], results[variant][str(tolerance)])
                    for tolerance in (10, 5)
                }
    return {
        "schema": "contact-serve-followups/1", "population": population, "status": "complete",
        "reference_commit": "02cf446b10966eccf4d8372da92d6377b45d9103", "fps": fps, "groups": groups,
        "reference_serve": "First labelled contact of each retained rally",
        "proposed_start": "First event of each nonempty proposed section",
        "matching": "Full-stream timing-only one-to-one matches, reused for every serve and side measure",
        "cached_score_limitation": "Old detector scores retain cross-group dependence; broader videos were examined before",
        "variants": results, "comparisons": comparisons, "seconds": perf_counter() - started,
    }


def run(population: str, output: Path) -> None:
    for name in ("development", "broader") if population == "all" else (population,):
        result = recount(name)
        write_json(output / f"{name}_serves.json.gz", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("development", "broader", "all"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    run(arguments.population, arguments.output)
