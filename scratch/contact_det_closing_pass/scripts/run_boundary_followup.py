"""Test a fixed contact-timing allowance at automatically proposed section edges."""

import argparse
from time import perf_counter

import joblib

from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.run_later_comparison import ROOT
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

RESULTS = ROOT / "results/followups"
BOUNDARY_MODES = ("padding", "fixed_membership")


def _mode_suffix(boundary_mode: str) -> str:
    if boundary_mode == "padding":
        return ""
    if boundary_mode == "fixed_membership":
        return "_fixed_membership"
    raise ValueError(f"unknown boundary mode: {boundary_mode}")


def run(variant: str, boundary_mode: str = "padding") -> None:
    suffix = _mode_suffix(boundary_mode)
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population = prepared["base_population"]
    current = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    reference = restore_stream(current["outputs"])
    source = current if variant == "session_start" else prediction_io.read_json(RESULTS / f"{variant}_predictions.json.gz")
    before = restore_stream(source["outputs"])
    after = pad_contact_boundaries(
        before.spans,
        before.events_by_fixture,
        population.fps,
        preserve_membership=boundary_mode == "fixed_membership",
    )
    write_json(RESULTS / f"{variant}_boundary_predictions{suffix}.json.gz", {
        "schema": "contact-boundary-predictions/1", "status": "complete", "variant": variant,
        "prediction_selection_uses_labels": False, "boundary_mode": boundary_mode,
        "padding_base30": 10, "outputs": stream_records(after),
    })
    selected = {(span.fixture, span.span_id): LaterOption(CombinedAction("keep", None, None, span), None, span)
                for span in after.spans}
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    comparison = compare_outputs(before.spans, selected, labels, population.fps, population.groups)
    direct = comparison if variant == "session_start" else compare_outputs(
        reference.spans, selected, labels, population.fps, population.groups,
    )
    local_comparisons = {}
    if variant in {"both", "early"}:
        local = restore_stream(prediction_io.read_json(RESULTS / "local_predictions.json.gz")["outputs"])
        local_boundary = restore_stream(prediction_io.read_json(
            RESULTS / f"local_boundary_predictions{suffix}.json.gz"
        )["outputs"])
        before_options = {
            (span.fixture, span.span_id): LaterOption(CombinedAction("keep", None, None, span), None, span)
            for span in before.spans
        }
        local_comparisons = {
            "input_detector_to_local": compare_outputs(
                local.spans, before_options, labels, population.fps, population.groups,
            ),
            "comparison_to_local_boundary": compare_outputs(
                local_boundary.spans, selected, labels, population.fps, population.groups,
            ),
        }
    write_json(RESULTS / f"{variant}_boundary_result{suffix}.json.gz", {
        "schema": "contact-boundary-comparison/1", "status": "complete", "variant": variant,
        "boundary_mode": boundary_mode,
        "comparison_to_input_detector": comparison, "comparison_to_session_start": direct,
        **local_comparisons,
        "padding_base30": 10, "full_stream_contacts": sum(map(len, after.events_by_fixture.values())),
        "raw_contact_stream_unchanged": after.events_by_fixture == before.events_by_fixture,
        "seconds": perf_counter() - started,
    })
    for tolerance, metrics in comparison.items():
        pair = metrics["paired"]
        print(variant, "boundary", tolerance, "correct", pair["correct_before"], pair["correct_after"],
              "repairs", len(pair["repaired"]), "losses", len(pair["lost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("session_start", "local", "pairs", "both", "early"), default="session_start")
    parser.add_argument("--boundary-mode", choices=BOUNDARY_MODES, default="padding")
    arguments = parser.parse_args()
    run(arguments.variant, arguments.boundary_mode)
