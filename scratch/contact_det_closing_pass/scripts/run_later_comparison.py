"""Compare saved later-contact insertions against the frozen combined chooser."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.later_evaluation import (
    compare_outputs,
    opportunity,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    MAX_LATER_CANDIDATES,
    LaterOption,
    apply_options,
    build_later_options,
    insertion_features,
    option_record,
    select_options,
)
from scratch.contact_det_closing_pass.scripts.replay_simple_replacements import (
    option_key,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    Population,
    prepare_population,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    option_record as base_option_record,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
    action_matrix,
    build_whole_features,
    load_measurements,
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    WHOLE_MODEL_SETTINGS,
    build_opening_cache,
    fit_whole_model,
    training_opening_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import whole_targets
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
DEFAULT_INPUTS = ROOT / "raw/later_inputs/development.json.gz"
DEFAULT_FEATURE_ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw"
DEFAULT_OUTPUT_ROOT = ROOT / "raw/later_run"
DEFAULT_RESULT_ROOT = ROOT / "results/later"
REFERENCE_RESULT = ROOT / "results/whole_rally_result.json.gz"
BROADER_MODELS = ROOT / "raw/broader_models.joblib"
VARIANT = "opening_sides_and_physics"
OPENING_FEATURE_NAMES = (
    "chosen_summary_opening_score",
    "chosen_physical_opening_score",
    "best_summary_opening_score",
    "best_physical_opening_score",
)
OPPORTUNITY_INPUT_NAME = "later_option_candidates.json.gz"
OPPORTUNITY_RESULT_NAME = "later_opportunity.json.gz"
PREDICTION_NAME = "later_predictions.json.gz"
RESULT_NAME = "later_result.json.gz"
PREPARED_NAME = "prepared.joblib"
MODELS_NAME = "models.joblib"
LATER_INPUT_SCHEMA = "contact-rally-start-later-inputs/1"

SectionIdentity = tuple[str, int]
FrameIdentity = tuple[str, int]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _side(value: object, label: str) -> str | None:
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player side differs")


def _physical_block(value: object, names: Sequence[str], label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != len(names):
        raise ValueError(f"{label}: physical block has the wrong width")
    values = []
    for index, raw_value in enumerate(value):
        if raw_value is None:
            values.append(np.nan)
            continue
        number = _finite_number(raw_value, f"{label}[{index}]")
        values.append(number)
    return np.asarray(values, dtype=np.float64)


def _selected_fixtures(population: Population, smoke_fixtures: int | None) -> frozenset[str]:
    if smoke_fixtures is None:
        return frozenset(video.fixture for video in population.videos)
    if smoke_fixtures <= 0:
        raise ValueError("--smoke-fixtures must be positive")
    return frozenset(video.fixture for video in population.videos[:smoke_fixtures])


def _subset_population(population: Population, fixtures: frozenset[str]) -> Population:
    if fixtures == {video.fixture for video in population.videos}:
        return population
    videos = tuple(video for video in population.videos if video.fixture in fixtures)
    spans = tuple(span for span in population.spans if span.fixture in fixtures)
    events = {fixture: values for fixture, values in population.events.items() if fixture in fixtures}
    actions = tuple(row for row in population.actions if row.candidate.fixture in fixtures)
    options = tuple(option for option in population.options if option.span.fixture in fixtures)
    fps = {fixture: value for fixture, value in population.fps.items() if fixture in fixtures}
    groups = {fixture: value for fixture, value in population.groups.items() if fixture in fixtures}
    return Population(videos, spans, events, actions, options, fps, groups)


def _load_later_inputs(
    path: Path,
    population: Population,
    fixtures: frozenset[str],
) -> tuple[dict[SectionIdentity, tuple[FixedEvent, ...]], dict[FrameIdentity, np.ndarray], tuple[str, ...], dict[str, Any]]:
    payload = prediction_io.read_json(path)
    raw_names = payload.get("physical_feature_names")
    raw_videos = payload.get("videos")
    if (
        payload.get("schema") != LATER_INPUT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
        or not isinstance(raw_names, list)
        or any(not isinstance(name, str) for name in raw_names)
        or len(set(raw_names)) != len(raw_names)
        or not isinstance(raw_videos, list)
    ):
        raise TypeError("later inputs must contain feature names and video records")

    spans_by_fixture = {}
    for span in population.spans:
        spans_by_fixture.setdefault(span.fixture, set()).add(span.span_id)
    candidates_by_section: dict[SectionIdentity, tuple[FixedEvent, ...]] = {}
    physical_by_frame: dict[FrameIdentity, np.ndarray] = {}
    seen_fixtures: set[str] = set()
    input_video_records = []
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("later input video must be an object")
        fixture = raw_video.get("fixture")
        group = raw_video.get("group")
        fps = raw_video.get("fps")
        sections = raw_video.get("sections")
        if not isinstance(fixture, str):
            raise TypeError("later input fixture must be text")
        if fixture not in fixtures:
            continue
        if fixture in seen_fixtures or group != population.groups[fixture]:
            raise ValueError(f"{fixture}: later input identity differs")
        if not np.isclose(_finite_number(fps, f"{fixture}: fps"), population.fps[fixture], rtol=0.0, atol=1e-6):
            raise ValueError(f"{fixture}: later input fps differs")
        if not isinstance(sections, list):
            raise TypeError(f"{fixture}: later sections must be a list")
        seen_fixtures.add(fixture)
        input_video_records.append({"fixture": fixture, "group": group, "fps": float(fps), "sections": len(sections)})
        seen_sections: set[int] = set()
        for raw_section in sections:
            if not isinstance(raw_section, Mapping):
                raise TypeError(f"{fixture}: later section must be an object")
            span_id = raw_section.get("span_id")
            raw_candidates = raw_section.get("candidates")
            if type(span_id) is not int or span_id not in spans_by_fixture[fixture] or span_id in seen_sections:
                raise ValueError(f"{fixture}: later section identity differs")
            if not isinstance(raw_candidates, list) or len(raw_candidates) > MAX_LATER_CANDIDATES:
                raise ValueError(
                    f"{fixture}/{span_id}: later candidates must contain at most {MAX_LATER_CANDIDATES} rows"
                )
            seen_sections.add(span_id)
            section_candidates = []
            for index, raw_candidate in enumerate(raw_candidates):
                if not isinstance(raw_candidate, Mapping):
                    raise TypeError(f"{fixture}/{span_id}: later candidate must be an object")
                frame = raw_candidate.get("frame")
                if type(frame) is not int:
                    raise ValueError(f"{fixture}/{span_id}: later candidate frame differs")
                identity = (fixture, frame)
                if identity in physical_by_frame:
                    raise ValueError(f"{identity}: later candidate frame repeats")
                score = _finite_number(raw_candidate.get("contact_score"), f"{fixture}/{frame}: contact score")
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"{fixture}/{frame}: contact score is outside zero to one")
                event = FixedEvent(fixture, frame, score, _side(raw_candidate.get("predicted_side"), f"{fixture}/{frame}"))
                physical_by_frame[identity] = _physical_block(
                    raw_candidate.get("physical"), raw_names, f"{fixture}/{frame}"
                )
                section_candidates.append(event)
            candidates_by_section[(fixture, span_id)] = tuple(section_candidates)
        if seen_sections != spans_by_fixture[fixture]:
            raise ValueError(f"{fixture}: later input section coverage differs")
    missing_fixtures = fixtures - seen_fixtures
    if missing_fixtures:
        raise ValueError(f"later inputs omit selected fixtures: {sorted(missing_fixtures)}")
    metadata = {
        "path": path.name,
        "physical_feature_names": list(raw_names),
        "video_records": input_video_records,
        "candidate_count": len(physical_by_frame),
    }
    return candidates_by_section, physical_by_frame, tuple(raw_names), metadata


def _merge_measurements(
    base: PhysicalMeasurements,
    later_values: Mapping[FrameIdentity, np.ndarray],
    names: tuple[str, ...],
) -> PhysicalMeasurements:
    if names != base.names:
        raise ValueError("later physical feature names differ from frozen measurements")
    values = dict(base.values)
    for identity, block in later_values.items():
        if block.shape != (len(names),) or np.isinf(block).any():
            raise ValueError(f"{identity}: later physical block differs")
        existing = values.get(identity)
        if existing is not None:
            if not np.allclose(existing, block, equal_nan=True):
                raise ValueError(f"{identity}: supplied physical block differs from frozen measurements")
            continue
        values[identity] = block
    audit = dict(base.audit)
    audit["later_candidate_frame_count"] = len(later_values)
    return PhysicalMeasurements(base.names, values, audit)


def _reference_selection(
    base_options: Sequence[Any], fixtures: frozenset[str],
) -> dict[SectionIdentity, LaterOption]:
    original = prediction_io.read_json(REFERENCE_RESULT)
    reference = original["development_descriptive"][VARIANT]
    by_key = {option_key(base_option_record(option)): option for option in base_options}
    selected: dict[SectionIdentity, LaterOption] = {}
    for raw_choice in reference["selected_actions"]:
        if raw_choice["fixture"] not in fixtures:
            continue
        key = option_key(raw_choice)
        base = by_key.get(key)
        if base is None:
            raise ValueError(f"{key}: frozen reference option is missing")
        selected[base.identity] = LaterOption(base, None, base.span)
    expected = {(option.span.fixture, option.span.span_id) for option in base_options}
    if set(selected) != expected:
        raise ValueError("frozen reference selection does not cover the selected population")
    return selected


def _descriptor(option: LaterOption) -> dict[str, Any]:
    record = option_record(option)
    record["inserted_contact_score"] = None if option.inserted is None else option.inserted.timing_score
    record["inserted_predicted_side"] = None if option.inserted is None else option.inserted.predicted_side
    return record


def _write_option_descriptor(
    path: Path,
    options: Sequence[LaterOption],
    input_metadata: Mapping[str, Any],
) -> None:
    write_json(path, {
        "schema": "contact-closing-later-option-inputs/1",
        "status": "complete",
        "labels_read": False,
        "input": dict(input_metadata),
        "options": [_descriptor(option) for option in options],
    })


def _expanded_matrix(static: np.ndarray, insertion: np.ndarray, opening: np.ndarray) -> np.ndarray:
    if len(static) != len(insertion) or len(static) != len(opening):
        raise ValueError("later feature row coverage differs")
    return np.column_stack((static, opening, insertion))


def _fit_oof_scores(
    population: Population,
    options: Sequence[LaterOption],
    static: np.ndarray,
    insertion: np.ndarray,
    targets: np.ndarray,
    opening_cache: Mapping[frozenset[str], Mapping],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray([population.groups[option.base.identity[0]] for option in options])
    scores = np.full(len(options), np.nan, dtype=np.float64)
    records = []
    for held_out in GROUPS:
        allowed = frozenset(GROUPS) - {held_out}
        train_indices = np.flatnonzero(groups != held_out)
        held_indices = np.flatnonzero(groups == held_out)
        if not len(held_indices):
            raise ValueError(f"{held_out}: held-out group has no later options")
        train_options = [options[index] for index in train_indices]
        held_options = [options[index] for index in held_indices]
        train_opening = opening_score_features(
            [option.proxy for option in train_options],
            training_opening_scores(population.actions, opening_cache, allowed),
        )
        held_opening = opening_score_features(
            [option.proxy for option in held_options], opening_cache[allowed]
        )
        model, fit_seconds = fit_whole_model(
            _expanded_matrix(static[train_indices], insertion[train_indices], train_opening),
            targets[train_indices],
        )
        scores[held_indices] = _positive_scores(
            model,
            _expanded_matrix(static[held_indices], insertion[held_indices], held_opening),
        )
        records.append({
            "held_out_group": held_out,
            "training_groups": sorted(allowed),
            "fit_seconds": fit_seconds,
            "training_options": len(train_indices),
            "held_out_options": len(held_indices),
        })
        print(f"Later outer group {held_out} complete", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("later OOF scores do not cover every option")
    return scores, records


def _load_or_compute_opportunity(
    path: Path,
    options: Sequence[LaterOption],
    reference: Mapping[SectionIdentity, LaterOption],
    labels: Any,
    population: Population,
) -> dict[str, Any]:
    if path.exists():
        return prediction_io.read_json(path)
    result = opportunity(options, reference, labels, population.fps, population.groups)
    write_json(path, {"schema": "contact-closing-later-opportunity/1", "status": "complete", **result})
    return result


def run(
    inputs: Path,
    feature_root: Path,
    output_root: Path,
    result_root: Path,
    opportunity_only: bool = False,
    smoke_fixtures: int | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    pack = prediction_io.load_development_predictions()
    raw_training = start._candidate_videos()
    full_population = prepare_population(pack, frozenset(GROUPS), raw_training)
    fixtures = _selected_fixtures(full_population, smoke_fixtures)
    population = _subset_population(full_population, fixtures)
    candidates, later_physical, physical_names, input_metadata = _load_later_inputs(inputs, population, fixtures)
    base_options = population.options
    options = build_later_options(base_options, candidates, population.fps)
    counts = {
        "development_videos": len(population.videos),
        "development_sections": len(population.spans),
        "base_options": len(base_options),
        "later_options": len(options),
        "later_insertions": sum(option.inserted is not None for option in options),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    _write_option_descriptor(output_root / OPPORTUNITY_INPUT_NAME, options, input_metadata)
    reference = _reference_selection(base_options, fixtures)
    reference_spans = tuple(reference[(span.fixture, span.span_id)].span for span in population.spans)
    if smoke_fixtures is not None:
        result = {"schema": "contact-closing-later-comparison/1", "status": "metadata_smoke", "counts": counts}
        write_json(result_root / "later_smoke.json.gz", result)
        return result

    labels = load_human_labels(start.LABEL_PATH, population.videos)
    opportunity_path = result_root / OPPORTUNITY_RESULT_NAME
    opportunity_result = _load_or_compute_opportunity(
        opportunity_path, options, reference, labels, population,
    )
    if opportunity_only:
        result = {
            "schema": "contact-closing-later-opportunity/1",
            "status": "complete",
            "counts": counts,
            "option_input_file": OPPORTUNITY_INPUT_NAME,
            "opportunity": opportunity_result,
            "labels_read_after_option_input": True,
        }
        write_json(result_root / "later_opportunity_summary.json.gz", result)
        for tolerance in ("10", "5"):
            print(tolerance, opportunity_result[tolerance]["counts"], flush=True)
        return result

    base_measurements = load_measurements(population.actions, population.events, feature_root)
    if base_measurements.audit["missing_identity_count"]:
        raise ValueError("frozen base measurements are missing requested identities")
    measurements = _merge_measurements(base_measurements, later_physical, physical_names)
    proxies = tuple(option.proxy for option in options)
    static, static_names, columns = build_whole_features(
        proxies, population.spans, population.actions, population.fps, measurements,
    )
    insertion, insertion_names = insertion_features(options, population.fps, measurements)
    local_targets, _ = assign_targets(
        population.actions, population.spans, population.events, labels, population.fps,
    )
    targets, target_report = whole_targets(proxies, population.spans, labels, population.fps)
    base_possible = set()
    insertion_possible = set()
    for option, target in zip(options, targets, strict=True):
        if target == 1:
            possible = base_possible if option.inserted is None else insertion_possible
            possible.add(option.base.identity)
    target_report["option_pool_capacity"] = {
        "sections_complete_with_existing_options": len(base_possible),
        "sections_complete_with_insertion_options": len(insertion_possible),
        "sections_newly_completable_only_with_insertion": len(insertion_possible - base_possible),
    }
    parsed_config = load_rally_start_model_config(start.CONFIG_PATH)
    opening_spec = next(model for model in parsed_config.models if model.model_id == "shallow_hgb")
    d_action_matrix = action_matrix(population.actions, measurements)
    opening_cache, opening_fit_records = build_opening_cache(
        population.actions, d_action_matrix, local_targets, opening_spec,
    )
    joblib.dump({
        "base_population": population,
        "options": options,
        "later_candidates": candidates,
        "later_physical": later_physical,
        "static_features": static,
        "static_feature_names": static_names,
        "columns": columns,
        "insertion_features": insertion,
        "insertion_feature_names": insertion_names,
        "measurements": measurements,
        "local_targets": local_targets,
        "targets": targets,
        "opening_cache": opening_cache,
    }, output_root / PREPARED_NAME, compress=3)

    oof_scores, whole_fit_records = _fit_oof_scores(
        population, options, static, insertion, targets, opening_cache,
    )
    selected = select_options(options, oof_scores)
    oof_stream = apply_options(population.spans, population.events, selected)
    oof_records = [_descriptor(option) | {"score": float(score)} for option, score in zip(options, oof_scores, strict=True)]
    write_json(result_root / PREDICTION_NAME, {
        "schema": "contact-closing-later-predictions/1",
        "status": "complete",
        "held_group_labels_used_for_fitting": False,
        "prediction_selection_uses_labels": False,
        "upstream_detector_scores_retain_cross_group_dependence": True,
        "counts": counts,
        "options": oof_records,
        "selected_actions": [_descriptor(option) for option in selected.values()],
        "outputs": stream_records(oof_stream),
    })

    if not BROADER_MODELS.exists():
        raise FileNotFoundError(f"frozen broader models are missing: {BROADER_MODELS.name}")
    frozen = joblib.load(BROADER_MODELS)
    final_opening = frozen.get("opening")
    if final_opening is None:
        raise ValueError("frozen broader models do not contain opening models")
    opening_source = BROADER_MODELS.name
    final_opening_seconds = 0.0
    d_opening = opening_score_features(
        proxies, training_opening_scores(population.actions, opening_cache, frozenset(GROUPS)),
    )
    whole, whole_fit_seconds = fit_whole_model(
        _expanded_matrix(static, insertion, d_opening), targets,
    )
    models = {
        "whole": whole,
        "opening": final_opening,
        "static_feature_names": static_names,
        "columns": columns,
        "opening_feature_names": OPENING_FEATURE_NAMES,
        "insertion_feature_names": insertion_names,
    }
    joblib.dump(models, output_root / MODELS_NAME, compress=3)
    comparison = compare_outputs(reference_spans, selected, labels, population.fps, population.groups)
    for tolerance in ("10", "5"):
        paired = comparison[tolerance]["paired"]
        print(
            "Later", tolerance, "correct", paired["correct_before"], paired["correct_after"],
            "repaired", len(paired["repaired"]), "lost", len(paired["lost"]), flush=True,
        )
    output = {
        "schema": "contact-closing-later-comparison/1",
        "status": "complete",
        "counts": counts,
        "model_settings": WHOLE_MODEL_SETTINGS,
        "target_tolerance_base30": 10,
        "option_input_file": OPPORTUNITY_INPUT_NAME,
        "prediction_file": PREDICTION_NAME,
        "opportunity_file": OPPORTUNITY_RESULT_NAME,
        "opportunity": opportunity_result,
        "static_feature_names": list(static_names),
        "opening_feature_names": list(OPENING_FEATURE_NAMES),
        "insertion_feature_names": list(insertion_names),
        "target_report": target_report,
        "oof_scores": oof_scores.tolist(),
        "selected_actions": [_descriptor(option) for option in selected.values()],
        "outputs": stream_records(oof_stream),
        "comparison_to_frozen_combined": comparison,
        "lineage": {
            "training_groups": list(GROUPS),
            "opening_scores_nested_inside_whole_model_folds": True,
            "upstream_detector_scores_retain_cross_group_dependence": True,
            "reference_variant": VARIANT,
            "validation_used": False,
        },
        "timings": {
            "opening_group_fits": opening_fit_records,
            "whole_group_fits": whole_fit_records,
            "final_opening_fit_seconds": final_opening_seconds,
            "final_opening_source": opening_source,
            "whole_fit_seconds": whole_fit_seconds,
            "total_seconds": perf_counter() - started,
        },
    }
    write_json(result_root / RESULT_NAME, output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--opportunity-only", action="store_true")
    parser.add_argument(
        "--smoke-fixtures",
        type=int,
        help="describe the first N development fixtures and exit before reading labels or fitting models",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = run(
        arguments.inputs,
        arguments.feature_root,
        arguments.output_root,
        arguments.result_root,
        arguments.opportunity_only,
        arguments.smoke_fixtures,
    )
    print("Finished", result["status"], flush=True)


if __name__ == "__main__":
    main()
