"""Apply frozen chosen acceptance to all 47 saved broader detector outputs."""

from __future__ import annotations

import argparse
from time import perf_counter

import joblib

from scratch.contact_det_closing_pass.scripts.broader_acceptance_inputs import (
    prepare_video,
)
from scratch.contact_det_closing_pass.scripts.evaluation import test_labels, write_json
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _partition_metrics,
)
from scratch.contact_det_closing_pass.scripts.run_serve_followups import OUTPUT, ROOT
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    build_acceptance_rows,
)
from scratch.contact_det_closing_pass.scripts.serve_metrics import accepted_serves
from scratch.contact_det_followup.scripts import prediction_io

RAW = ROOT / "raw/serve_followups"
CURRENT = ROOT / "results/followups/local_broader_predictions.json.gz"
GUARDED = ROOT / "results/followups/local_boundary_broader_predictions_fixed_membership.json.gz"


def predict_video(opening: dict, later: dict, spans: tuple, events: tuple, physical_names: tuple,
                  current: dict, guarded: dict, frozen: dict) -> dict:
    started = perf_counter()
    fixture = str(later["fixture"])
    cache_path = RAW / f"chosen_acceptance_broader_{fixture}_features.joblib"
    cached = cache_path.exists()
    if cached:
        prepared = joblib.load(cache_path)
    else:
        prepared = prepare_video(opening, later, spans, events, physical_names, current, guarded, frozen["local"])
        joblib.dump(prepared, cache_path, compress=3)
    scores = {}
    score_started = perf_counter()
    for variant in ("base", "gap"):
        if tuple(prepared[f"{variant}_feature_names"]) != tuple(frozen[f"{variant}_feature_names"]):
            raise ValueError(f"{fixture}: {variant} acceptance features differ from the frozen model")
        scores[variant] = _positive_scores(frozen["models"][variant], prepared[f"{variant}_features"])
    score_seconds = perf_counter() - score_started
    actions = {(row["fixture"], row["span_id"]): row for row in current["selected_actions"]}
    rows = []
    for index, identity in enumerate(prepared["identities"]):
        rows.append({"fixture": identity[0], "span_id": identity[1], "kind": actions[identity]["kind"],
                     "score": prepared["selected_scores"][identity],
                     "base_score": float(scores["base"][index]), "gap_score": float(scores["gap"][index])})
    print("Broader acceptance", fixture, "scored", flush=True)
    return {"fixture": fixture, "rows": rows, "features_reused": cached,
            "original_feature_seconds": prepared["seconds"], "score_seconds": score_seconds,
            "seconds": perf_counter() - started}


def run(jobs: int = 4, limit: int | None = None) -> None:
    started = perf_counter()
    frozen = joblib.load(RAW / "chosen_acceptance_models.joblib")
    opening = prediction_io.read_json(ROOT / "raw/broader_inputs/chooser_inputs.json.gz")
    later = prediction_io.read_json(ROOT / "raw/later_inputs/broader.json.gz")
    current = prediction_io.read_json(CURRENT)
    guarded = prediction_io.read_json(GUARDED)
    for payload in (opening, later, current, guarded):
        if payload["status"] != "complete" or payload["labels_read"]:
            raise ValueError("Broader acceptance requires completed label-free detector inputs")
    pack = prediction_io.load_frozen_test_predictions()
    fixtures = tuple(str(video["fixture"]) for video in pack.videos)
    indexed = [{str(video["fixture"]): video for video in payload["videos"]}
               for payload in (opening, later, current, guarded)]
    if any(set(index) != set(fixtures) for index in indexed):
        raise ValueError("Acceptance inputs must cover the same fixed 47 videos")
    selected_fixtures = fixtures if limit is None else fixtures[:limit]
    opening_by_video, later_by_video, current_by_video, guarded_by_video = indexed
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=8):
        records = joblib.Parallel()(joblib.delayed(predict_video)(
            opening_by_video[fixture], later_by_video[fixture],
            tuple(span for span in pack.spans if span.fixture == fixture), pack.events_by_fixture[fixture],
            tuple(later["physical_feature_names"]), current_by_video[fixture], guarded_by_video[fixture], frozen,
        ) for fixture in selected_fixtures)
    suffix = "_smoke" if limit is not None else ""
    prediction_name = f"chosen_acceptance_broader_predictions{suffix}.json.gz"
    prediction_seconds = perf_counter() - started
    write_json(OUTPUT / prediction_name, {
        "schema": "contact-chosen-acceptance-broader-predictions/1", "status": "complete", "labels_read": False,
        "detector": frozen["detector"], "detector_predictions": GUARDED.name, "video_count": len(records),
        "videos": records, "frozen_policies": frozen["policies"], "prediction_seconds": prediction_seconds,
    })
    print("Acceptance predictions saved before loading evaluation labels", flush=True)

    labels = test_labels()
    serves = prediction_io.read_json(OUTPUT / "broader_serves.json.gz")["variants"]["recommended"]
    views = {}
    for tolerance, result in serves.items():
        views[tolerance] = []
        for row in result["sections"]:
            if row["fixture"] in selected_fixtures:
                views[tolerance].append({**row, "group": "ShuttleSet22",
                                         "fully_correct": row["side_rule_fully_correct"],
                                         "correct_sides": row["voted_correct_sides"]})
    choices = [row for record in records for row in record["rows"]]
    rows = build_acceptance_rows(views, choices, labels, {})
    score_rows = {(row["fixture"], row["span_id"]): row for row in choices}
    for row in rows:
        scores = score_rows[(row["fixture"], row["span_id"])]
        row.update({"base_score": scores["base_score"], "gap_score": scores["gap_score"]})
    metrics, accepted = {}, {}
    for variant, policies in frozen["policies"].items():
        metrics[variant], accepted[variant] = {}, {}
        for name, policy in policies.items():
            if policy is None:
                continue
            threshold = policy["threshold"]
            metrics[variant][name] = _partition_metrics(rows, f"{variant}_score", threshold)
            identities = {(row["fixture"], row["span_id"]) for row in rows if row[f"{variant}_score"] >= threshold}
            accepted[variant][name] = {tolerance: accepted_serves(result, identities)
                                       for tolerance, result in serves.items()}
            print(variant, name, metrics[variant][name]["by_tolerance"], flush=True)
    write_json(OUTPUT / f"chosen_acceptance_broader{suffix}.json.gz", {
        "schema": "contact-chosen-acceptance-broader/1", "status": "complete", "video_count": len(records),
        "detector": frozen["detector"], "prediction_file": prediction_name, "rows": rows,
        "frozen_policies": frozen["policies"], "accepted_metrics": metrics, "accepted_serves": accepted,
        "timings": {"prediction_seconds": prediction_seconds, "total_seconds": perf_counter() - started,
                    "per_video": [
                        {key: value for key, value in record.items() if key != "rows"}
                        for record in records
                    ]},
        "smoke_serve_denominator": "All 47 videos retained; acceptance only from smoke videos" if limit else None,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.limit)
