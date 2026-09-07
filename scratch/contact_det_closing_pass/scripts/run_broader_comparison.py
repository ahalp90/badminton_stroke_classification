"""Predict the 47 saved videos with frozen choosers, then evaluate and report acceptance."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    score_contacts,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.replay_simple_replacements import (
    option_key,
    replay_choices,
    section_key,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
    _select,
    _valid_action_spans,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    ROOT,
    option_record,
    variant_matrix,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    build_acceptance_rows,
    summarise_acceptance,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    local_harm,
    paired_evaluations,
    section_views,
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
from scratch.contact_det_closing_pass.scripts.whole_rally_options import (
    build_options,
    choose_options,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.audit_combined_best_case import (
    CombinedAction,
    _apply_actions,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    build_candidate_rows,
)

VARIANT = "opening_sides_and_physics"


def stream_records(stream: ContactStreams) -> dict[str, Any]:
    span_records = []
    for span in stream.spans:
        frames = []
        for event in span.events:
            frames.append(event.frame)
        span_records.append(
            {
                "fixture": span.fixture,
                "span_id": span.span_id,
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
                "frames": frames,
            }
        )

    contact_records = {}
    for fixture, events in stream.events_by_fixture.items():
        fixture_records = []
        for event in events:
            fixture_records.append(
                [
                    event.frame,
                    event.timing_score,
                    event.predicted_side,
                ]
            )
        contact_records[fixture] = fixture_records

    return {"spans": span_records, "contacts": contact_records}


def predict_video(
    video: Mapping[str, Any],
    spans: Sequence[FixedSpan],
    events: Sequence[FixedEvent],
    models: Mapping[str, Any],
    feature_root: Path,
    policy: Mapping[str, Any],
) -> tuple[dict[str, ContactStreams], dict, list[dict], dict]:
    fixture = str(video["video"]["fixture"])
    fps = {fixture: float(video["video"]["fps"])}
    started = perf_counter()
    actions = start.build_action_rows(
        build_candidate_rows([video], default_group="S22")
    )
    grouped_options = build_options(spans, [video], {fixture: events})
    option_list = []
    for group in grouped_options.values():
        option_list.extend(group)
    options = tuple(option_list)
    proposal_seconds = perf_counter() - started
    started = perf_counter()
    measurements = load_measurements(actions, {fixture: events}, feature_root)
    if measurements.audit["missing_identity_count"]:
        raise ValueError(f"{fixture}: frozen physical measurements are missing from the supplied feature root")
    action_values = action_matrix(actions, measurements)
    loading_seconds = perf_counter() - started
    started = perf_counter()
    static, names, columns = build_whole_features(
        options, spans, actions, fps, measurements
    )
    if names != models["static_feature_names"] or columns != models["columns"]:
        raise ValueError(
            f"{fixture}: broader feature layout differs from the frozen model"
        )
    opening_scores = predict_opening_models(models["opening"], actions, action_values)
    opening_features = opening_score_features(options, opening_scores)
    matrix = variant_matrix(
        static, np.arange(len(options)), columns, opening_features, VARIANT
    )
    feature_seconds = perf_counter() - started
    started = perf_counter()
    scores = _positive_scores(models["whole"], matrix)
    selected = choose_options(options, scores, policy["minimum_score"])
    option_records = []
    for option in options:
        option_records.append(option_record(option))
    selected_records = []
    for option in selected.values():
        selected_records.append(option_record(option))
    choices = replay_choices(
        option_records,
        scores,
        selected_records,
        policy["cancel_simple_replace"],
    )
    by_option = {}
    for option, option_record_value in zip(options, option_records, strict=True):
        by_option[option_key(option_record_value)] = option
    replayed_selected = {}
    for choice in choices:
        replayed_selected[section_key(choice)] = by_option[option_key(choice)]
    selected = replayed_selected
    combined = _apply_actions(spans, {fixture: events}, selected)
    combined_seconds = perf_counter() - started
    started = perf_counter()
    valid = _valid_action_spans(actions, spans, {fixture: events})
    opening_score_values = {}
    for identity, pair in opening_scores.items():
        opening_score_values[identity] = pair[0]
    opening_selected = _select(actions, opening_score_values, valid)
    opening_only = start.apply_selected_actions(
        spans, {fixture: events}, opening_selected
    )
    opening_seconds = perf_counter() - started
    streams = {
        "baseline": ContactStreams(tuple(spans), {fixture: tuple(events)}),
        "opening_only": opening_only,
        "combined": combined,
    }

    selected_opening_records = []
    for row in opening_selected.values():
        selected_opening_records.append(
            {
                "fixture": row.candidate.fixture,
                "span_id": row.candidate.span_id,
                "candidate_frame": row.candidate.frame,
                "kind": row.action,
                "score": opening_scores[row.identity][0],
            }
        )

    output_records = {}
    for name, stream in streams.items():
        output_records[name] = stream_records(stream)

    timings = {
        "proposals_seconds": proposal_seconds,
        "load_measurements_seconds": loading_seconds,
        "build_features_and_opening_scores_seconds": feature_seconds,
        "whole_predict_select_apply_seconds": combined_seconds,
        "opening_only_select_apply_seconds": opening_seconds,
    }
    record = {
        "fixture": fixture,
        "fps": fps[fixture],
        "labels_read": False,
        "options": option_records,
        "whole_scores": scores.tolist(),
        "selected_combined": choices,
        "selected_opening": selected_opening_records,
        "outputs": output_records,
        "feature_join_audit": measurements.audit,
        "timings": timings,
    }
    return streams, selected, choices, record


def uncertain_anchors() -> dict[str, list[int]]:
    coverage = prediction_io.read_json(ROOT / "results/label_coverage.json.gz")
    anchors = {}
    for video in coverage["by_video"]:
        video_anchors = []
        for rally in video["dropped_rallies"]:
            video_anchors.extend(rally["unflagged_in_range_frames"])
        anchors[str(video["video_id"])] = video_anchors
    return anchors


def evaluate_outputs(
    streams: Mapping[str, ContactStreams],
    selected: Mapping[tuple[str, int], CombinedAction],
    choices: Sequence[dict],
    fps: dict[str, float],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    labels = test_labels()
    groups = {}
    for fixture in fps:
        groups[fixture] = "ShuttleSet22"

    baseline = {}
    for tolerance in (10, 5):
        baseline[str(tolerance)] = section_views(
            streams["baseline"].spans,
            labels,
            fps,
            groups,
            tolerance,
        )

    results = {}
    voted_seconds = {}
    for name, stream in streams.items():
        started = perf_counter()
        voted = start.apply_whole_rally_alternation(stream)
        voted_seconds[name] = perf_counter() - started
        if name == "baseline":
            after = baseline
        else:
            after = {}
            for tolerance in (10, 5):
                after[str(tolerance)] = section_views(
                    stream.spans,
                    labels,
                    fps,
                    groups,
                    tolerance,
                )

        contacts = {}
        for tolerance in (10, 5):
            raw = score_contacts(stream.events_by_fixture, labels, fps, tolerance)
            contacts[str(tolerance)] = {
                "raw": raw,
                "fixed_side": voted_contact_scores(raw, voted.events_by_fixture),
            }

        results[name] = {
            "evaluation": paired_evaluations(baseline, after),
            "contacts": contacts,
        }
        if name == "combined":
            harm = {}
            for tolerance in (10, 5):
                harm[str(tolerance)] = local_harm(
                    selected,
                    streams["baseline"].spans,
                    baseline[str(tolerance)]["fixed_side"]["sections"],
                    labels,
                    fps,
                    tolerance,
                )
            results[name]["harm"] = harm

            sections = {}
            for tolerance in ("10", "5"):
                sections[tolerance] = after[tolerance]["fixed_side"]["sections"]
            rows = build_acceptance_rows(sections, choices, labels, uncertain_anchors())
            curve = []
            for threshold in acceptance["thresholds"]:
                curve.append(summarise_acceptance(rows, threshold))
            results[name]["acceptance"] = {
                "rows": rows,
                "frozen_rules": acceptance,
                "curve": curve,
            }

        summary = {}
        for tolerance, result in results[name]["evaluation"].items():
            paired_fixed_side = result["paired_fixed_side"]
            summary[tolerance] = (
                paired_fixed_side["correct_after"],
                len(paired_fixed_side["repaired"]),
                len(paired_fixed_side["lost"]),
            )
        print(name, summary, flush=True)
    combined_versus_opening = {}
    for tolerance in ("10", "5"):
        opening_rows = results["opening_only"]["evaluation"][tolerance]["edited_fixed_side"]["sections"]
        combined_rows = results["combined"]["evaluation"][tolerance]["edited_fixed_side"]["sections"]
        combined_versus_opening[tolerance] = paired_sections(opening_rows, combined_rows)
    return {
        "systems": results,
        "combined_versus_opening": combined_versus_opening,
        "side_vote_seconds": voted_seconds,
    }


def run(
    inputs: Path,
    feature_root: Path,
    model_path: Path,
    output_root: Path,
) -> None:
    started = perf_counter()
    policy = prediction_io.read_json(ROOT / "results/broader_action_policy.json.gz")
    acceptance = prediction_io.read_json(
        ROOT / "results/broader_acceptance_policy.json.gz"
    )
    models = joblib.load(model_path)
    pack = prediction_io.load_frozen_test_predictions()
    candidates = prediction_io.read_json(inputs)
    by_fixture = {}
    for video in candidates["videos"]:
        by_fixture[str(video["video"]["fixture"])] = video
    if set(by_fixture) != set(pack.events_by_fixture):
        raise ValueError("Broader chooser inputs do not cover the frozen 47 videos")
    span_lists = {}
    for name in ("baseline", "opening_only", "combined"):
        span_lists[name] = []
    event_maps = {}
    for name in span_lists:
        event_maps[name] = {}
    selected = {}
    choices = []
    records = []
    fps = {}
    for video in pack.videos:
        fixture = str(video["fixture"])
        spans = tuple(span for span in pack.spans if span.fixture == fixture)
        outputs, video_selected, video_choices, record = predict_video(
            by_fixture[fixture],
            spans,
            pack.events_by_fixture[fixture],
            models,
            feature_root,
            policy,
        )
        fps[fixture] = float(video["fps"])
        for name, stream in outputs.items():
            span_lists[name].extend(stream.spans)
            event_maps[name].update(stream.events_by_fixture)
        selected.update(video_selected)
        choices.extend(video_choices)
        records.append(record)
        print("Predicted", fixture, "sections", len(spans), flush=True)
    streams = {}
    for name, spans_for_name in span_lists.items():
        streams[name] = ContactStreams(tuple(spans_for_name), event_maps[name])
    prediction_seconds = perf_counter() - started
    prediction = {
        "schema": "contact-closing-broader-predictions/1",
        "status": "complete",
        "labels_read": False,
        "action_policy": policy,
        "acceptance_policy": acceptance,
        "videos": records,
        "prediction_seconds": prediction_seconds,
    }
    write_json(output_root / "broader_predictions.json.gz", prediction)
    print("All 47 predictions saved; now loading labels for evaluation", flush=True)
    evaluation = evaluate_outputs(streams, selected, choices, fps, acceptance)
    action_counts = dict(Counter(choice["kind"] for choice in choices))
    per_video_timings = []
    for record in records:
        per_video_timings.append({"fixture": record["fixture"], **record["timings"]})
    result = {
        "schema": "contact-closing-broader-comparison/1",
        "status": "complete",
        "video_count": len(pack.videos),
        "sections": len(pack.spans),
        "action_counts": action_counts,
        "data_status": "Previously examined broader comparison; prediction did not read these labels",
        "action_policy": policy,
        "acceptance_policy": acceptance,
        **evaluation,
        "timings": {
            "prediction_seconds": prediction_seconds,
            "total_seconds": perf_counter() - started,
            "per_video": per_video_timings,
        },
    }
    write_json(output_root / "broader_result.json.gz", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "raw/broader_inputs/chooser_inputs.json.gz",
    )
    parser.add_argument(
        "--feature-root", type=Path, default=ROOT / "raw/broader_inputs/features"
    )
    parser.add_argument(
        "--models", type=Path, default=ROOT / "raw/broader_models.joblib"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.inputs, args.feature_root, args.models, args.output_root)


if __name__ == "__main__":
    main()
