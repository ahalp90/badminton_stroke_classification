"""Apply frozen gap acceptance to unchanged, previously examined video outputs."""

from time import perf_counter

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.gap_evidence import gap_evidence
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.run_insertion_broader import (
    _candidate_inputs,
)
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _partition_metrics,
    _tail_curve,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _merge_measurements,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    load_measurements,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)

RESULTS = ROOT / "results/followups"


def predict_video(video: dict, opening: dict, later: dict, physical_names: tuple, frozen: dict) -> dict:
    started = perf_counter()
    fixture = video["fixture"]
    stream = restore_stream(video["output"])
    candidates, physical = _candidate_inputs(later, stream.spans)
    actions = start.build_action_rows(build_candidate_rows([opening], default_group="ShuttleSet22"))
    measurements = load_measurements(actions, stream.events_by_fixture, ROOT / "raw/broader_inputs/features")
    if measurements.audit["missing_identity_count"]:
        raise ValueError(f"{fixture}: missing saved physical rows")
    measurements = _merge_measurements(measurements, physical, physical_names)
    selected = {(span.fixture, span.span_id): LaterOption(CombinedAction("keep", None, None, span), None, span)
                for span in stream.spans}
    features, names, identities = gap_evidence(
        selected, candidates, {fixture: video["fps"]}, measurements, frozen["local"],
    )
    if tuple(names) != tuple(frozen["gap_feature_names"]):
        raise ValueError("Frozen gap feature contract differs")
    if tuple(video["acceptance_feature_names"]) != tuple(frozen["base_feature_names"]):
        raise ValueError("Frozen control feature contract differs")
    saved_base = {row["span_id"]: row["values"] for row in video["acceptance_features"]}
    base = np.asarray([saved_base[identity[1]] for identity in identities], dtype=np.float64)
    control = _positive_scores(frozen["models"]["control"], base)
    gap = _positive_scores(frozen["models"]["gap"], np.column_stack((base, features)))
    return {
        "fixture": fixture, "seconds": perf_counter() - started,
        "rows": [{"fixture": fixture, "span_id": identity[1], "control_score": float(control[index]),
                  "gap_score": float(gap[index])} for index, identity in enumerate(identities)],
    }


def run() -> None:
    started = perf_counter()
    frozen = joblib.load(ROOT / "raw/followups/gap_acceptance_models.joblib")
    reference = prediction_io.read_json(ROOT / "results/later/later_broader_predictions.json.gz")
    opening = prediction_io.read_json(ROOT / "raw/broader_inputs/chooser_inputs.json.gz")
    later = prediction_io.read_json(ROOT / "raw/later_inputs/broader.json.gz")
    opening_by_fixture = {video["video"]["fixture"]: video for video in opening["videos"]}
    later_by_fixture = {video["fixture"]: video for video in later["videos"]}
    with joblib.parallel_config(backend="loky", n_jobs=4, inner_max_num_threads=6):
        videos = joblib.Parallel()(joblib.delayed(predict_video)(
            video, opening_by_fixture[video["fixture"]], later_by_fixture[video["fixture"]],
            tuple(later["physical_feature_names"]), frozen,
        ) for video in reference["videos"])
    prediction_seconds = perf_counter() - started
    write_json(RESULTS / "gap_broader_predictions.json.gz", {
        "schema": "contact-gap-broader-predictions/1", "status": "complete", "detector": "session_start",
        "detector_predictions_file": "later_broader_predictions.json.gz", "identical_detector_outputs": True,
        "labels_read": False, "videos": videos, "policies": frozen["policies"],
        "prediction_seconds": prediction_seconds,
    })
    # The contact output is unchanged, so its saved judgements need no new matching run.
    judged = prediction_io.read_json(ROOT / "results/later/later_broader_result.json.gz")
    by_identity = {(row["fixture"], row["span_id"]): row for row in judged["acceptance_rows"]}
    rows = [{**by_identity[(row["fixture"], row["span_id"])], **row, "group": "ShuttleSet22"}
            for video in videos for row in video["rows"]]
    if len(rows) != len(by_identity):
        raise ValueError("Frozen acceptance must cover every saved detector output")
    policies = {name: {"threshold": policy["threshold"],
                       **_partition_metrics(rows, f"{name}_score", policy["threshold"])}
                for name, policy in frozen["policies"].items()}
    write_json(RESULTS / "gap_broader_result.json.gz", {
        "schema": "contact-gap-broader-comparison/1", "status": "complete", "detector": "session_start",
        "data_status": "Previously examined videos; acceptance predictions made before reading saved judgements",
        "identical_detector_outputs": True, "rows": rows, "frozen_policies": policies,
        "common_coverage_diagnostics": {name: _tail_curve(rows, f"{name}_score") for name in policies},
        "prediction_seconds": prediction_seconds, "total_seconds": perf_counter() - started,
        "per_video": [{"fixture": video["fixture"], "seconds": video["seconds"]} for video in videos],
    })
    for name, policy in policies.items():
        print(name, "accepted", policy["accepted_count"], policy["by_tolerance"], flush=True)


if __name__ == "__main__":
    run()
