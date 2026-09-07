"""Apply a frozen insertion follow-up model to the previously examined videos."""

from __future__ import annotations

import argparse
import lzma
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.early_shortlist import (
    MAX_EARLY_CANDIDATES,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.followup_options import (
    choice_key,
    contextual_insertions,
    pair_features,
    restore_choices,
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
)
from scratch.contact_det_closing_pass.scripts.run_insertion_followup import (
    local_columns,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _expanded_matrix,
    _merge_measurements,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
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

RAW = ROOT / "raw/followups"
RESULTS = ROOT / "results/followups"
DEFAULT_INPUTS = ROOT / "raw/broader_inputs/chooser_inputs.json.gz"
DEFAULT_EARLY_INPUTS = ROOT / "raw/followups/early_broader_inputs.json.gz"
DEFAULT_LATER_INPUTS = ROOT / "raw/later_inputs/broader.json.gz"
DEFAULT_FEATURE_ROOT = ROOT / "raw/broader_inputs/features"
LOCAL_REFERENCE_PREDICTIONS = ROOT / "results/followups/local_broader_predictions.json.gz"
SESSION_START_REFERENCE_PREDICTIONS = ROOT / "results/later/later_broader_predictions.json.gz"

SectionIdentity = tuple[str, int]


def _pair_options(
    base_options: Sequence[Any],
    candidates: Mapping[SectionIdentity, Sequence[FixedEvent]],
    fps: Mapping[str, float],
) -> tuple[LaterOption, ...]:
    """Return only the two-insertion options from the expanded option pool."""
    expanded = build_later_options(base_options, candidates, fps, max_insertions=2)
    return tuple(option for option in expanded if option.second_inserted is not None)


def _build_variant_options(
    base_options: Sequence[Any],
    candidates: Mapping[SectionIdentity, Sequence[FixedEvent]],
    fps: Mapping[str, float],
    variant: str,
) -> tuple[LaterOption, ...]:
    singles = build_later_options(base_options, candidates, fps, max_insertions=1)
    if variant in {"local", "early"}:
        return singles
    return (*singles, *_pair_options(base_options, candidates, fps))


def _candidate_inputs(
    later: Mapping[str, Any],
    spans: Sequence[FixedSpan],
) -> tuple[
    dict[SectionIdentity, tuple[FixedEvent, ...]], dict[tuple[str, int], np.ndarray]
]:
    """Restore one video of label-free later candidates and physical blocks."""
    fixture = str(later["fixture"])
    candidates: dict[SectionIdentity, tuple[FixedEvent, ...]] = {}
    blocks: dict[tuple[str, int], np.ndarray] = {}
    for section in later["sections"]:
        span_id = int(section["span_id"])
        rows = []
        for row in section["candidates"]:
            frame = int(row["frame"])
            identity = (fixture, frame)
            if identity in blocks:
                raise ValueError(f"{fixture}/{frame}: later candidate frame repeats")
            rows.append(
                FixedEvent(
                    fixture, frame, float(row["contact_score"]), row["predicted_side"]
                )
            )
            blocks[identity] = np.asarray(row["physical"], dtype=np.float64)
        candidates[(fixture, span_id)] = tuple(rows)
    expected = {(span.fixture, span.span_id) for span in spans}
    if set(candidates) != expected:
        raise ValueError(f"{fixture}: later inputs do not cover every section")
    return candidates, blocks


def _insertion_matrix(
    options: Sequence[LaterOption],
    variant: str,
    fps: Mapping[str, float],
    measurements: Any,
    models: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, int]]:
    """Build the frozen original, pair and optional local insertion columns."""
    insertion, names = insertion_features(options, fps, measurements)
    model_names = tuple(models["insertion_feature_names"])
    if names != model_names:
        raise ValueError(
            "Original insertion feature names differ from the frozen model"
        )
    names = tuple(names)
    parts = [insertion]
    if variant in {"pairs", "both"}:
        second, second_names = pair_features(options, fps, measurements)
        parts.append(second)
        names += tuple(second_names)
    local_width = 0
    if variant in {"local", "early", "both"}:
        local_model = models.get("local")
        if local_model is None:
            raise ValueError(f"{variant}: frozen local model is missing")
        if variant in {"local", "early"}:
            included = np.asarray([option.inserted is not None for option in options])
            context_matrix, context_names = insertion[included], model_names
        else:
            contexts = contextual_insertions(options)
            context_matrix, context_names = insertion_features(contexts, fps, measurements)
        expected_context_names = tuple(models.get("local_feature_names") or ())
        if expected_context_names and tuple(context_names) != expected_context_names:
            raise ValueError(
                "Local insertion feature names differ from the frozen model"
            )
        local_scores = _positive_scores(local_model, context_matrix)
        local_width = 2 if variant == "both" else 1
        parts.append(local_columns(tuple(options), local_scores, local_width))
        names += ("local__first_score", "local__second_score")[:local_width]
    matrix = np.column_stack(parts)
    return matrix, names, {"local_width": local_width, "feature_count": matrix.shape[1]}


def predict_video(
    opening_video: Mapping[str, Any],
    later_video: Mapping[str, Any],
    spans: Sequence[FixedSpan],
    events: Sequence[FixedEvent],
    physical_names: tuple[str, ...],
    feature_root: Path,
    models: Mapping[str, Any],
    reference_choices: Sequence[Mapping[str, Any]],
    reference_output: Mapping[str, Any],
    variant: str,
    score_root: Path | None = None,
) -> tuple[
    ContactStreams,
    dict[SectionIdentity, LaterOption],
    dict[SectionIdentity, LaterOption],
    dict[str, Any],
]:
    """Build, score and select one broader video without reading labels."""
    total_started = perf_counter()
    fixture = str(later_video["fixture"])
    fps = {fixture: float(later_video["fps"])}
    load_started = perf_counter()
    candidates, later_physical = _candidate_inputs(later_video, spans)
    actions = start.build_action_rows(
        build_candidate_rows(
            [opening_video],
            default_group="ShuttleSet22",
            max_earlier_candidates=(
                MAX_EARLY_CANDIDATES if variant == "early" else 2
            ),
        )
    )
    base_by_section = build_options(spans, [opening_video], {fixture: events})
    base_options = tuple(
        option for section in base_by_section.values() for option in section
    )
    options = _build_variant_options(base_options, candidates, fps, variant)
    reference = restore_choices(options, reference_choices)
    reference_stream = apply_options(spans, {fixture: events}, reference)
    if stream_records(reference_stream) != reference_output:
        raise ValueError(
            f"{fixture}: restored current output differs from saved prediction"
        )
    base_measurements = load_measurements(actions, {fixture: events}, feature_root)
    if base_measurements.audit["missing_identity_count"]:
        raise ValueError(f"{fixture}: saved physical measurements are missing")
    measurements = _merge_measurements(
        base_measurements, later_physical, physical_names
    )
    action_values = action_matrix(actions, measurements)
    load_seconds = perf_counter() - load_started

    feature_started = perf_counter()
    proxies = tuple(option.proxy for option in options)
    static, static_names, columns = build_whole_features(
        proxies,
        spans,
        actions,
        fps,
        measurements,
    )
    if (
        tuple(static_names) != tuple(models["static_feature_names"])
        or columns != models["columns"]
    ):
        raise ValueError(
            f"{fixture}: broader feature layout differs from the frozen model"
        )
    opening_scores = predict_opening_models(models["opening"], actions, action_values)
    opening = opening_score_features(proxies, opening_scores)
    insertion, insertion_names, insertion_metadata = _insertion_matrix(
        options,
        variant,
        fps,
        measurements,
        models,
    )
    matrix = _expanded_matrix(static, insertion, opening)
    expected_width = getattr(models["whole"], "n_features_in_", matrix.shape[1])
    if int(expected_width) != matrix.shape[1]:
        raise ValueError(
            f"{fixture}: frozen whole model expects {expected_width} features, got {matrix.shape[1]}"
        )
    feature_seconds = perf_counter() - feature_started

    prediction_started = perf_counter()
    scores = _positive_scores(models["whole"], matrix)
    selected = select_with_reference(options, scores, reference, MIN_EDIT_ADVANTAGE)
    stream = apply_options(spans, {fixture: events}, selected)
    prediction_seconds = perf_counter() - prediction_started
    score_by_key = {
        choice_key(option_record(option)): float(score)
        for option, score in zip(options, scores, strict=True)
    }
    score_root = RAW if score_root is None else score_root
    score_root.mkdir(parents=True, exist_ok=True)
    scores_path = score_root / f"{variant}_{fixture}_option_scores.npy.xz"
    with lzma.open(scores_path, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
        np.save(handle, scores)
    selected_records = []
    for option in selected.values():
        record = option_record(option)
        key = choice_key(record)
        selected_records.append({**record, "score": score_by_key[key]})
    record = {
        "fixture": fixture,
        "fps": fps[fixture],
        "labels_read": False,
        "variant": variant,
        "option_scores_file": scores_path.name,
        "option_count": len(options),
        "insertion_option_count": sum(option.inserted is not None for option in options),
        "pair_option_count": sum(option.second_inserted is not None for option in options),
        "selected_actions": selected_records,
        "output": stream_records(stream),
        "feature_join_audit": measurements.audit,
        "feature_count": int(matrix.shape[1]),
        "insertion_feature_count": int(insertion.shape[1]),
        "insertion_feature_names": list(insertion_names),
        "insertion_metadata": insertion_metadata,
        "timings": {
            "load_seconds": load_seconds,
            "feature_seconds": feature_seconds,
            "prediction_seconds": prediction_seconds,
            "total_seconds": perf_counter() - total_started,
        },
    }
    return stream, selected, reference, record


def _prediction_paths(
    variant: str, limit_videos: int | None, output_root: Path
) -> tuple[Path, Path]:
    suffix = "_smoke" if limit_videos is not None else ""
    return (
        output_root / f"{variant}_broader_predictions{suffix}.json.gz",
        output_root / f"{variant}_broader_result{suffix}.json.gz",
    )


def run(
    variant: str,
    limit_videos: int | None = None,
    jobs: int = 4,
    *,
    inputs: Path | None = None,
    later_inputs: Path = DEFAULT_LATER_INPUTS,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    model_path: Path | None = None,
    output_root: Path = RESULTS,
    score_root: Path | None = None,
) -> dict[str, Any]:
    """Replay a frozen follow-up model on all or a bounded smoke subset."""
    if variant not in {"local", "early", "pairs", "both"}:
        raise ValueError(f"unknown follow-up variant: {variant}")
    if limit_videos is not None and limit_videos <= 0:
        raise ValueError("--limit-videos must be positive")
    if jobs <= 0:
        raise ValueError("--jobs must be positive")
    started = perf_counter()
    if inputs is None:
        inputs = DEFAULT_EARLY_INPUTS if variant == "early" else DEFAULT_INPUTS
    model_path = (
        RAW / f"{variant}_models.joblib" if model_path is None else Path(model_path)
    )
    models = joblib.load(model_path)
    opening_inputs = prediction_io.read_json(inputs)
    later_bundle = prediction_io.read_json(later_inputs)
    if (
        later_bundle.get("status") != "complete"
        or later_bundle.get("labels_read") is not False
    ):
        raise ValueError("Later input bundle is incomplete or used labels")
    physical_names = tuple(later_bundle["physical_feature_names"])
    chooser_by_fixture = {
        str(video["video"]["fixture"]): video for video in opening_inputs["videos"]
    }
    later_by_fixture = {
        str(video["fixture"]): video for video in later_bundle["videos"]
    }
    pack = prediction_io.load_frozen_test_predictions()
    fixtures = tuple(str(video["fixture"]) for video in pack.videos)
    if set(chooser_by_fixture) != set(fixtures) or set(later_by_fixture) != set(
        fixtures
    ):
        raise ValueError("Broader input bundles must cover the frozen 47 videos")
    selected_fixtures = fixtures if limit_videos is None else fixtures[:limit_videos]
    reference_path = (
        LOCAL_REFERENCE_PREDICTIONS
        if variant == "early"
        else SESSION_START_REFERENCE_PREDICTIONS
    )
    reference_payload = prediction_io.read_json(reference_path)
    if (
        reference_payload.get("status") != "complete"
        or reference_payload.get("labels_read") is not False
    ):
        raise ValueError("Saved broader reference is incomplete or used labels")
    reference_by_fixture = {
        str(video["fixture"]): video for video in reference_payload["videos"]
    }
    if set(reference_by_fixture) != set(fixtures):
        raise ValueError(
            "Saved broader reference does not cover the frozen 47 videos"
        )
    direct_by_fixture: dict[str, Mapping[str, Any]] = {}
    if variant == "early":
        direct_payload = prediction_io.read_json(SESSION_START_REFERENCE_PREDICTIONS)
        if (
            direct_payload.get("status") != "complete"
            or direct_payload.get("labels_read") is not False
        ):
            raise ValueError("Saved later-broader reference is incomplete or used labels")
        direct_by_fixture = {
            str(video["fixture"]): video for video in direct_payload["videos"]
        }
        if set(direct_by_fixture) != set(fixtures):
            raise ValueError(
                "Saved later-broader predictions do not cover the frozen 47 videos"
            )

    jobs_data = []
    for raw_video in pack.videos:
        fixture = str(raw_video["fixture"])
        if fixture not in selected_fixtures:
            continue
        spans = tuple(span for span in pack.spans if span.fixture == fixture)
        jobs_data.append(
            (
                chooser_by_fixture[fixture],
                later_by_fixture[fixture],
                spans,
                pack.events_by_fixture[fixture],
                physical_names,
                Path(feature_root),
                models,
                reference_by_fixture[fixture]["selected_actions"],
                reference_by_fixture[fixture]["output"],
                variant,
                score_root,
            )
        )
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=6):
        fitted = joblib.Parallel()(
            joblib.delayed(predict_video)(*arguments) for arguments in jobs_data
        )

    records = [result[3] for result in fitted]
    all_spans: list[FixedSpan] = []
    all_events: dict[str, Sequence[FixedEvent]] = {}
    selected: dict[SectionIdentity, LaterOption] = {}
    reference_spans: list[FixedSpan] = []
    direct_reference_spans: list[FixedSpan] = []
    fps: dict[str, float] = {}
    for arguments, result in zip(jobs_data, fitted, strict=True):
        (
            _opening,
            later,
            spans,
            _events,
            _names,
            _root,
            _models,
            _choices,
            _output,
            _variant,
            _score_root,
        ) = arguments
        stream, chosen, reference, _record = result
        fixture = str(later["fixture"])
        all_spans.extend(stream.spans)
        all_events[fixture] = tuple(stream.events_by_fixture[fixture])
        selected.update(chosen)
        reference_spans.extend(
            reference[identity].span
            for identity in ((span.fixture, span.span_id) for span in spans)
        )
        if variant == "early":
            direct_reference_spans.extend(
                restore_stream(direct_by_fixture[fixture]["output"]).spans
            )
        fps[fixture] = float(later["fps"])

    prediction_path, result_path = _prediction_paths(
        variant, limit_videos, Path(output_root)
    )
    prediction_payload = {
        "schema": "contact-followup-broader-predictions/1",
        "status": "complete",
        "labels_read": False,
        "variant": variant,
        "data_status": "Previously examined videos; predictions made without their labels",
        "reference_predictions": (
            "local_broader_predictions.json.gz"
            if variant == "early"
            else "later_broader_predictions.json.gz"
        ),
        "minimum_edit_advantage": MIN_EDIT_ADVANTAGE,
        "videos": records,
        "prediction_seconds": perf_counter() - started,
    }
    write_json(prediction_path, prediction_payload)
    print(f"Saved label-free predictions to {prediction_path}", flush=True)

    labels = test_labels()
    groups = dict.fromkeys(fps, "ShuttleSet22")
    comparison = compare_outputs(tuple(reference_spans), selected, labels, fps, groups)
    direct_comparison = None
    if variant == "early":
        direct_comparison = compare_outputs(
            tuple(direct_reference_spans), selected, labels, fps, groups
        )
    stream = ContactStreams(tuple(all_spans), all_events)
    voted = start.apply_whole_rally_alternation(stream)
    contacts = {}
    for tolerance in (10, 5):
        raw_scores = score_contacts(stream.events_by_fixture, labels, fps, tolerance)
        contacts[str(tolerance)] = voted_contact_scores(
            raw_scores, voted.events_by_fixture
        )
        paired = comparison[str(tolerance)]["paired"]
        print(
            variant,
            tolerance,
            "correct",
            paired["correct_before"],
            "to",
            paired["correct_after"],
            "repairs",
            len(paired["repaired"]),
            "losses",
            len(paired["lost"]),
            flush=True,
        )
    option_count = sum(record["option_count"] for record in records)
    insertion_count = sum(record["insertion_option_count"] for record in records)
    pair_count = sum(record["pair_option_count"] for record in records)
    selected_count = sum(len(record["selected_actions"]) for record in records)
    selected_insertions = sum(
        int(option["inserted_frame"] is not None)
        for record in records
        for option in record["selected_actions"]
    )
    result = {
        "schema": "contact-followup-broader-comparison/1",
        "status": "complete",
        "variant": variant,
        "data_status": "Previously examined videos; predictions made without their labels",
        "counts": {
            "videos": len(records),
            "sections": len(all_spans),
            "options": option_count,
            "insertion_options": insertion_count,
            "pair_options": pair_count,
            "selected_sections": selected_count,
            "selected_insertions": selected_insertions,
        },
        **(
            {
                "comparison_to_local": comparison,
                "comparison_to_session_start": direct_comparison,
            }
            if variant == "early"
            else {"comparison_to_session_start": comparison}
        ),
        "contacts": contacts,
        "lineage": {
            "prediction_selection_uses_labels": False,
            "upstream_detector_scores_retain_cross_group_dependence": True,
            "reference_stream_verified_from_saved_selected_actions": True,
            "selection_rule": "select_with_reference",
            "minimum_edit_advantage": MIN_EDIT_ADVANTAGE,
        },
        "timings": {
            "total_seconds": perf_counter() - started,
            "per_video": [
                record["timings"] | {"fixture": record["fixture"]} for record in records
            ],
        },
    }
    write_json(result_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("local", "early", "pairs", "both"), required=True)
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="run only the first N videos and use smoke output names",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--later-inputs", type=Path, default=DEFAULT_LATER_INPUTS)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    parser.add_argument("--score-root", type=Path, help="save option scores in a separate experiment directory")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = run(
        arguments.variant,
        arguments.limit_videos,
        arguments.jobs,
        inputs=arguments.inputs,
        later_inputs=arguments.later_inputs,
        feature_root=arguments.feature_root,
        model_path=arguments.models,
        output_root=arguments.output_root,
        score_root=arguments.score_root,
    )
    print(f"Finished {result['status']}", flush=True)


if __name__ == "__main__":
    main()
