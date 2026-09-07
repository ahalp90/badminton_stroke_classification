"""Apply a frozen later-contact chooser to the previously examined 47 videos."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.later_acceptance_features import (
    acceptance_features,
)
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    LaterOption,
    apply_options,
    build_later_options,
    insertion_features,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
    uncertain_anchors,
)
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _partition_metrics,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    _expanded_matrix,
    _merge_measurements,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    build_acceptance_rows,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    voted_contact_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    action_matrix,
    build_whole_features,
    load_measurements,
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    predict_opening_models,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import build_options
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    build_candidate_rows,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"


def restore_stream(record: Mapping[str, Any]) -> ContactStreams:
    """Restore saved raw predictions without recomputing the reference chooser."""
    events = {
        fixture: tuple(FixedEvent(fixture, int(frame), float(score), side) for frame, score, side in contacts)
        for fixture, contacts in record["contacts"].items()
    }
    lookup = {(fixture, event.frame): event for fixture, contacts in events.items() for event in contacts}
    spans = []
    for row in record["spans"]:
        spans.append(FixedSpan(row["fixture"], row["span_id"], row["start_frame"], row["end_frame"],
                               tuple(lookup[(row["fixture"], frame)] for frame in row["frames"])))
    return ContactStreams(tuple(spans), events)


def predict_video(
    video: Mapping[str, Any], later: Mapping[str, Any], physical_names: tuple[str, ...],
    spans: Sequence[FixedSpan], events: Sequence[FixedEvent], models: Mapping[str, Any], feature_root: Path,
    reference_choices: Sequence[Mapping[str, Any]],
) -> tuple[ContactStreams, dict[tuple[str, int], LaterOption], dict[str, Any]]:
    """Score all ordinary sections using frozen models and automatic inputs only."""
    started = perf_counter()
    fixture = str(later["fixture"])
    fps = {fixture: float(later["fps"])}
    candidates = {}
    blocks = {}
    for section in later["sections"]:
        contacts = []
        for row in section["candidates"]:
            contacts.append(FixedEvent(fixture, row["frame"], row["contact_score"], row["predicted_side"]))
            blocks[(fixture, row["frame"])] = np.asarray(row["physical"], dtype=np.float64)
        candidates[(fixture, section["span_id"])] = tuple(contacts)
    if set(candidates) != {(span.fixture, span.span_id) for span in spans}:
        raise ValueError(f"{fixture}: later inputs do not cover every section")
    actions = start.build_action_rows(build_candidate_rows([video], default_group="ShuttleSet22"))
    base_options = tuple(option for section in build_options(spans, [video], {fixture: events}).values() for option in section)
    options = build_later_options(base_options, candidates, fps)
    base_by_key = {
        (option.identity, option.kind, option.candidate_frame, option.deleted_frame): option
        for option in base_options
    }
    reference = {}
    for choice in reference_choices:
        identity = (fixture, choice['span_id'])
        base = base_by_key[(identity, choice['kind'], choice['candidate_frame'], choice['deleted_frame'])]
        reference[identity] = LaterOption(base, None, base.span)
    base_measurements = load_measurements(actions, {fixture: events}, feature_root)
    if base_measurements.audit["missing_identity_count"]:
        raise ValueError(f"{fixture}: saved physical measurements are missing")
    measurements = _merge_measurements(base_measurements, blocks, physical_names)
    action_values = action_matrix(actions, measurements)
    proxies = tuple(option.proxy for option in options)
    static, names, columns = build_whole_features(proxies, spans, actions, fps, measurements)
    insertion, insertion_names = insertion_features(options, fps, measurements)
    if names != models["static_feature_names"] or columns != models["columns"] or insertion_names != models["insertion_feature_names"]:
        raise ValueError("Broader features differ from the frozen later model")
    opening = opening_score_features(proxies, predict_opening_models(models["opening"], actions, action_values))
    input_seconds = perf_counter() - started
    prediction_started = perf_counter()
    scores = _positive_scores(models["whole"], _expanded_matrix(static, insertion, opening))
    selected = select_with_reference(options, scores, reference)
    stream = apply_options(spans, {fixture: events}, selected)
    prediction_seconds = perf_counter() - prediction_started
    acceptance_started = perf_counter()
    evidence, evidence_names, identities = acceptance_features(options, scores, selected, candidates, fps, measurements)
    evidence_records = []
    choices = []
    for identity, values in zip(identities, evidence, strict=True):
        evidence_records.append({"span_id": identity[1], "values": [None if np.isnan(value) else float(value) for value in values]})
        choices.append({**option_record(selected[identity]), "score": float(values[0])})
    records = []
    for option, score in zip(options, scores, strict=True):
        records.append({**option_record(option), "score": float(score),
                        "inserted_side": None if option.inserted is None else option.inserted.predicted_side})
    return stream, selected, {
        "fixture": fixture, "fps": fps[fixture], "options": records, "selected_actions": choices,
        "output": stream_records(stream), "acceptance_feature_names": evidence_names,
        "acceptance_features": evidence_records,
        "timings": {"load_and_features_seconds": input_seconds, "predict_select_apply_seconds": prediction_seconds,
                    "acceptance_evidence_seconds": perf_counter() - acceptance_started,
                    "total_seconds": perf_counter() - started},
    }


def run(
    inputs: Path, later_inputs: Path, feature_root: Path, model_path: Path,
    acceptance_path: Path, output_root: Path,
) -> None:
    started = perf_counter()
    models = joblib.load(model_path)
    acceptance = joblib.load(acceptance_path)
    if acceptance['minimum_edit_advantage'] != MIN_EDIT_ADVANTAGE:
        raise ValueError("Acceptance was fitted for a different detector margin")
    opening_inputs = prediction_io.read_json(inputs)
    later_bundle = prediction_io.read_json(later_inputs)
    if later_bundle["status"] != "complete" or later_bundle["labels_read"] is not False:
        raise ValueError("Later input bundle is incomplete or used labels")
    opening_by_fixture = {str(video["video"]["fixture"]): video for video in opening_inputs["videos"]}
    later_by_fixture = {str(video["fixture"]): video for video in later_bundle["videos"]}
    physical_names = tuple(later_bundle["physical_feature_names"])
    pack = prediction_io.load_frozen_test_predictions()
    if set(opening_by_fixture) != set(pack.events_by_fixture) or set(later_by_fixture) != set(pack.events_by_fixture):
        raise ValueError("Both input bundles must cover the frozen 47 videos")
    reference_predictions = prediction_io.read_json(ROOT / "results/broader_predictions.json.gz")
    reference_by_fixture = {video['fixture']: video for video in reference_predictions['videos']}
    reference_spans = []
    for video in reference_predictions["videos"]:
        reference_spans.extend(restore_stream(video["outputs"]["combined"]).spans)
    all_spans = []
    all_events = {}
    selected = {}
    records = []
    fps = {}
    for raw_video in pack.videos:
        fixture = str(raw_video["fixture"])
        spans = tuple(span for span in pack.spans if span.fixture == fixture)
        stream, chosen, record = predict_video(
            opening_by_fixture[fixture], later_by_fixture[fixture], physical_names, spans,
            pack.events_by_fixture[fixture], models, feature_root,
            reference_by_fixture[fixture]['selected_combined'],
        )
        fps[fixture] = float(raw_video["fps"])
        if record["acceptance_feature_names"] != tuple(acceptance["feature_names"]):
            raise ValueError("Broader acceptance features differ from the frozen development model")
        acceptance_started = perf_counter()
        evidence = np.asarray([row["values"] for row in record["acceptance_features"]], dtype=np.float64)
        record["acceptance_scores"] = {
            "selected_score": _positive_scores(acceptance["models"]["selected_score"], evidence[:, :1]).tolist(),
            "all_evidence": _positive_scores(acceptance["models"]["all_evidence"], evidence).tolist(),
        }
        acceptance_seconds = perf_counter() - acceptance_started
        record["timings"]["acceptance_predict_seconds"] = acceptance_seconds
        record["timings"]["total_seconds"] += acceptance_seconds
        all_spans.extend(stream.spans)
        all_events.update(stream.events_by_fixture)
        selected.update(chosen)
        records.append(record)
        print(f"Predicted later options for video {fixture}", flush=True)
    prediction_seconds = perf_counter() - started
    prediction_path = output_root / "later_broader_predictions.json.gz"
    write_json(prediction_path, {
        "schema": "contact-closing-later-broader-predictions/1", "status": "complete", "labels_read": False,
        "reference_predictions": "broader_predictions.json.gz", "videos": records,
        "minimum_edit_advantage": MIN_EDIT_ADVANTAGE,
        "acceptance_policies": acceptance["frozen_policies"],
        "prediction_seconds": prediction_seconds,
    })
    print("All 47 later predictions saved; now loading labels", flush=True)
    labels = test_labels()
    groups = dict.fromkeys(fps, "ShuttleSet22")
    comparison = compare_outputs(reference_spans, selected, labels, fps, groups)
    choices = [choice for record in records for choice in record["selected_actions"]]
    acceptance_rows = build_acceptance_rows(
        {tolerance: comparison[tolerance]["sections"] for tolerance in ("10", "5")},
        choices, labels, uncertain_anchors(),
    )
    acceptance_by_identity = {(row["fixture"], row["span_id"]): row for row in acceptance_rows}
    for record in records:
        for index, choice in enumerate(record["selected_actions"]):
            row = acceptance_by_identity[(record["fixture"], choice["span_id"])]
            row["raw_selected_score"] = row["score"]
            row["acceptance_selected_score"] = record["acceptance_scores"]["selected_score"][index]
            row["acceptance_all_evidence_score"] = record["acceptance_scores"]["all_evidence"][index]
    acceptance_results = {}
    for name, policy in acceptance["frozen_policies"].items():
        acceptance_results[name] = {
            "frozen_development_policy": policy,
            "curve": [
                {"tail": rule["tail"], "threshold": rule["threshold"],
                 **_partition_metrics(acceptance_rows, policy["score_key"], rule["threshold"])}
                for rule in policy["curve"]
            ],
        }
    stream = ContactStreams(tuple(all_spans), all_events)
    voted = start.apply_whole_rally_alternation(stream)
    contacts = {}
    for tolerance in (10, 5):
        raw_scores = score_contacts(stream.events_by_fixture, labels, fps, tolerance)
        contacts[str(tolerance)] = voted_contact_scores(raw_scores, voted.events_by_fixture)
        paired = comparison[str(tolerance)]["paired"]
        print(tolerance, "correct", paired["correct_before"], "to", paired["correct_after"],
              "repairs", len(paired["repaired"]), "losses", len(paired["lost"]), flush=True)
    write_json(output_root / "later_broader_result.json.gz", {
        "schema": "contact-closing-later-broader-comparison/1", "status": "complete",
        "data_status": "Previously examined videos; predictions made without their labels",
        "sections": len(all_spans), "comparison_to_frozen_combined": comparison, "contacts": contacts,
        "acceptance": acceptance_results, "acceptance_rows": acceptance_rows,
        "timings": {"prediction_seconds": prediction_seconds, "total_seconds": perf_counter() - started,
                    "per_video": [{"fixture": row["fixture"], **row["timings"]} for row in records]},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "raw/broader_inputs/chooser_inputs.json.gz")
    parser.add_argument("--later-inputs", type=Path, default=ROOT / "raw/later_inputs/broader.json.gz")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "raw/broader_inputs/features")
    parser.add_argument("--models", type=Path, default=ROOT / "raw/later_run/models.joblib")
    parser.add_argument("--acceptance-models", type=Path, default=ROOT / "raw/later_acceptance/models.joblib")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/later")
    arguments = parser.parse_args()
    run(arguments.inputs, arguments.later_inputs, arguments.feature_root, arguments.models,
        arguments.acceptance_models, arguments.output_root)


if __name__ == "__main__":
    main()
