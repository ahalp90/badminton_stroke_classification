"""Compare whole-rally choices using summaries, local edit scores and physics."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    score_contacts,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _group_lineage,
    _positive_scores,
    _subset_development,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
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
    GROUPS,
    WHOLE_MODEL_SETTINGS,
    build_opening_cache,
    fit_opening_models,
    fit_whole_model,
    predict_opening_models,
    training_opening_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import (
    build_options,
    choose_options,
    whole_targets,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.audit_combined_best_case import (
    CombinedAction,
    _apply_actions,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import VideoSpec
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
VARIANTS = ("summaries", "opening_and_sides", "opening_sides_and_physics")
MINIMUM_SCORES = (0.0, 0.5, 0.9)
OPENING_FEATURE_NAMES = (
    "chosen_summary_opening_score", "chosen_physical_opening_score",
    "best_summary_opening_score", "best_physical_opening_score",
)
RESULT_NAME = "whole_rally_result.json.gz"
PREDICTION_NAME = "whole_rally_predictions.json.gz"


@dataclass(frozen=True)
class Population:
    videos: tuple[VideoSpec, ...]
    spans: tuple[FixedSpan, ...]
    events: Mapping[str, Sequence[FixedEvent]]
    actions: tuple[start.ActionRow, ...]
    options: tuple[CombinedAction, ...]
    fps: dict[str, float]
    groups: dict[str, str]


def prepare_population(
    pack: prediction_io.DevelopmentPredictionPack, groups: frozenset[str], raw_videos: Sequence[Mapping[str, Any]],
    *, max_earlier_candidates: int = 2,
) -> Population:
    videos, spans, events = _subset_development(pack, groups)
    actions = start.build_action_rows(build_candidate_rows(
        raw_videos, default_group="V", max_earlier_candidates=max_earlier_candidates,
    ))
    grouped_options = build_options(spans, raw_videos, events)
    options = tuple(option for section_options in grouped_options.values() for option in section_options)
    return Population(videos, spans, events, actions, options, {video.fixture: video.fps for video in videos},
                      {video.fixture: pack.group_by_fixture[video.fixture] for video in videos})


def variant_columns(groups: Mapping[str, tuple[int, ...]], variant: str) -> tuple[int, ...]:
    if variant == "summaries":
        return groups["summary"]
    if variant == "opening_and_sides":
        return (*groups["summary"], *groups["side"])
    if variant == "opening_sides_and_physics":
        return (*groups["summary"], *groups["side"], *groups["physical"])
    raise ValueError(f"Unknown feature comparison {variant}")


def variant_matrix(
    static: np.ndarray, indices: np.ndarray, groups: Mapping[str, tuple[int, ...]],
    opening: np.ndarray, variant: str,
) -> np.ndarray:
    values = static[np.ix_(indices, variant_columns(groups, variant))]
    return values if variant == "summaries" else np.column_stack((values, opening))


def option_record(option: CombinedAction) -> dict[str, Any]:
    return {"fixture": option.span.fixture, "span_id": option.span.span_id, "kind": option.kind,
            "candidate_frame": option.candidate_frame, "deleted_frame": option.deleted_frame,
            "start_frame": option.span.start_frame, "end_frame": option.span.end_frame}


def selected_records(selected: Mapping[tuple[str, int], CombinedAction]) -> list[dict[str, Any]]:
    return [option_record(option) for option in selected.values()]


def population_baseline(population: Population, labels: HumanLabels) -> dict[str, Any]:
    return {str(tolerance): section_views(population.spans, labels, population.fps, population.groups, tolerance)
            for tolerance in (10, 5)}


def fit_development_scores(
    population: Population, static: np.ndarray, columns: Mapping[str, tuple[int, ...]],
    opening_cache: Mapping, targets: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    groups = np.asarray([population.groups[option.span.fixture] for option in population.options])
    scores = {variant: np.full(len(population.options), np.nan) for variant in VARIANTS}
    records = []
    for held_out in GROUPS:
        allowed = frozenset(GROUPS) - {held_out}
        train_indices = np.flatnonzero(groups != held_out)
        held_indices = np.flatnonzero(groups == held_out)
        train_options = [population.options[index] for index in train_indices]
        held_options = [population.options[index] for index in held_indices]
        train_opening = opening_score_features(
            train_options, training_opening_scores(population.actions, opening_cache, allowed),
        )
        held_opening = opening_score_features(held_options, opening_cache[allowed])
        for variant in VARIANTS:
            train_matrix = variant_matrix(static, train_indices, columns, train_opening, variant)
            held_matrix = variant_matrix(static, held_indices, columns, held_opening, variant)
            model, fit_seconds = fit_whole_model(train_matrix, targets[train_indices])
            scores[variant][held_indices] = _positive_scores(model, held_matrix)
            records.append({"variant": variant, "held_out_group": held_out,
                            "training_groups": sorted(allowed), "fit_seconds": fit_seconds})
        print(f"Whole-rally outer group {held_out} complete", flush=True)
    for variant, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{variant}: incomplete outer prediction coverage")
    return scores, records


def select_development_policy(
    population: Population, scores: np.ndarray, labels: HumanLabels, baseline: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]], dict, ContactStreams, dict]:
    curve = []
    choices = []
    for minimum in MINIMUM_SCORES:
        selected = choose_options(population.options, scores, minimum)
        edited = _apply_actions(population.spans, population.events, selected)
        after = section_views(edited.spans, labels, population.fps, population.groups, 10)
        paired = paired_sections(baseline["10"]["fixed_side"]["sections"], after["fixed_side"]["sections"])
        edited_count = sum(option.kind != "keep" for option in selected.values())
        record = {"minimum_score": minimum, **paired, "edited_sections": edited_count,
                  "net_correct": paired["correct_after"] - paired["correct_before"]}
        curve.append(record)
        choices.append((minimum, selected, edited, after))
    best_index = max(range(len(curve)), key=lambda index: (
        curve[index]["net_correct"], -len(curve[index]["lost"]),
        -curve[index]["edited_sections"], curve[index]["minimum_score"],
    ))
    minimum, selected, edited, after = choices[best_index]
    return minimum, curve, selected, edited, after


def add_evaluation(
    population: Population, selected: Mapping[tuple[str, int], CombinedAction], edited: ContactStreams,
    labels: HumanLabels, baseline: Mapping[str, Any], after10: Mapping[str, Any],
) -> dict[str, Any]:
    after = {"10": after10, "5": section_views(edited.spans, labels, population.fps, population.groups, 5)}
    harm = {}
    for tolerance in (10, 5):
        harm[str(tolerance)] = local_harm(selected, population.spans,
                                        baseline[str(tolerance)]["fixed_side"]["sections"],
                                        labels, population.fps, tolerance)
    return {"evaluation": paired_evaluations(baseline, after), "harm": harm,
            "action_counts": dict(Counter(option.kind for option in selected.values()))}


def reference_comparisons(validation: Mapping[str, Any]) -> dict[str, Any]:
    historical = prediction_io.read_json(ROOT / "results/historical_start_reference.json.gz")
    standalone = prediction_io.read_json(ROOT / "results/start_comparison_result.json.gz")
    references = {"historical_chooser": historical["evaluation"]}
    for name, result in standalone["validation_frozen_choices"]["variants"].items():
        references[name] = result["evaluation"]
    output = {}
    for variant, result in validation.items():
        comparisons = {}
        for name, reference in references.items():
            comparisons[name] = {}
            for tolerance in ("10", "5"):
                before = reference[tolerance]["edited_fixed_side"]["sections"]
                after = result["evaluation"][tolerance]["edited_fixed_side"]["sections"]
                comparisons[name][tolerance] = paired_sections(before, after)
        output[variant] = comparisons
    return output


def run(feature_root: Path, output_root: Path, smoke: bool = False) -> dict[str, Any]:
    started = perf_counter()
    pack = prediction_io.load_development_predictions()
    raw_training = start._candidate_videos()
    raw_validation = prediction_io.read_json(prediction_io.VALIDATION_PREDICTIONS)["videos"]
    development = prepare_population(pack, frozenset(GROUPS), raw_training)
    validation = prepare_population(pack, frozenset({"V"}), raw_validation)
    lineage = _group_lineage(raw_training, frozenset(validation.groups))
    counts = {"development_videos": len(development.videos), "validation_videos": len(validation.videos),
              "development_sections": len(development.spans), "validation_sections": len(validation.spans),
              "development_options": len(development.options), "validation_options": len(validation.options)}
    print("Option counts", counts, flush=True)
    if smoke:
        return {"status": "metadata_smoke", "counts": counts, "lineage": lineage}

    measurements_started = perf_counter()
    measurements = load_measurements((*development.actions, *validation.actions), pack.events_by_fixture, feature_root)
    loading_seconds = perf_counter() - measurements_started
    d_action_matrix = action_matrix(development.actions, measurements)
    v_action_matrix = action_matrix(validation.actions, measurements)
    features_started = perf_counter()
    d_static, names, columns = build_whole_features(development.options, development.spans, development.actions,
                                                   development.fps, measurements)
    d_feature_seconds = perf_counter() - features_started
    features_started = perf_counter()
    v_static, v_names, v_columns = build_whole_features(validation.options, validation.spans, validation.actions,
                                                       validation.fps, measurements)
    v_feature_seconds = perf_counter() - features_started
    if names != v_names or columns != v_columns:
        raise ValueError("Development and validation feature contracts differ")
    print(f"Features ready: {len(names)} static columns; missing joins {measurements.audit['missing_identity_count']}",
          flush=True)

    d_labels = load_human_labels(start.LABEL_PATH, development.videos)
    start_targets, _ = assign_targets(development.actions, development.spans, development.events, d_labels, development.fps)
    targets, target_report = whole_targets(development.options, development.spans, d_labels, development.fps)
    config = load_rally_start_model_config(start.CONFIG_PATH)
    opening_spec = next(model for model in config.models if model.model_id == "shallow_hgb")
    cache, opening_fit_records = build_opening_cache(development.actions, d_action_matrix, start_targets, opening_spec)
    d_scores, whole_fit_records = fit_development_scores(development, d_static, columns, cache, targets)
    d_baseline = population_baseline(development, d_labels)
    development_results = {}
    policies = {}
    for variant in VARIANTS:
        minimum, curve, selected, edited, after10 = select_development_policy(
            development, d_scores[variant], d_labels, d_baseline,
        )
        policies[variant] = minimum
        development_results[variant] = {
            "minimum_score": minimum, "policy_curve": curve,
            "scores": d_scores[variant].tolist(), "selected_actions": selected_records(selected),
            **add_evaluation(development, selected, edited, d_labels, d_baseline, after10),
        }
        paired = development_results[variant]["evaluation"]["10"]["paired_fixed_side"]
        print("D", variant, minimum, "correct", paired["correct_after"],
              "repaired", len(paired["repaired"]), "lost", len(paired["lost"]), flush=True)

    final_opening, opening_fit_seconds = fit_opening_models(
        development.actions, d_action_matrix, start_targets, frozenset(GROUPS), opening_spec,
    )
    d_opening = opening_score_features(
        development.options, training_opening_scores(development.actions, cache, frozenset(GROUPS)),
    )
    v_opening = opening_score_features(validation.options,
                                      predict_opening_models(final_opening, validation.actions, v_action_matrix))
    prediction_variants = {}
    chosen_validation = {}
    all_d = np.arange(len(development.options))
    all_v = np.arange(len(validation.options))
    for variant in VARIANTS:
        model, fit_seconds = fit_whole_model(variant_matrix(d_static, all_d, columns, d_opening, variant), targets)
        predict_started = perf_counter()
        v_scores = _positive_scores(model, variant_matrix(v_static, all_v, columns, v_opening, variant))
        selected = choose_options(validation.options, v_scores, policies[variant])
        predict_seconds = perf_counter() - predict_started
        apply_started = perf_counter()
        edited = _apply_actions(validation.spans, validation.events, selected)
        apply_seconds = perf_counter() - apply_started
        feature_names = [names[index] for index in variant_columns(columns, variant)]
        if variant != "summaries":
            feature_names.extend(OPENING_FEATURE_NAMES)
        prediction_variants[variant] = {
            "minimum_score": policies[variant], "scores": v_scores.tolist(),
            "selected_actions": selected_records(selected), "feature_names": feature_names,
            "fit_seconds": fit_seconds, "predict_and_select_seconds": predict_seconds, "apply_seconds": apply_seconds,
        }
        chosen_validation[variant] = (selected, edited)
    write_json(output_root / PREDICTION_NAME, {
        "schema": "contact-closing-whole-rally-predictions/1", "status": "complete",
        "validation_labels_read": False, "score_is_calibrated": False,
        "options": [option_record(option) for option in validation.options], "variants": prediction_variants,
    })
    print("All validation predictions saved; now loading validation labels", flush=True)

    v_labels = load_human_labels(start.LABEL_PATH, validation.videos)
    v_baseline = population_baseline(validation, v_labels)
    validation_results = {}
    baseline_stream = start.apply_selected_actions(validation.spans, validation.events, {})
    voted_baseline = start.apply_whole_rally_alternation(baseline_stream)
    baseline_contacts = {}
    for tolerance in (10, 5):
        raw = score_contacts(validation.events, v_labels, validation.fps, tolerance)
        baseline_contacts[str(tolerance)] = {"raw": raw, "fixed_side": voted_contact_scores(raw, voted_baseline.events_by_fixture)}
    for variant, (selected, edited) in chosen_validation.items():
        after10 = section_views(edited.spans, v_labels, validation.fps, validation.groups, 10)
        result = add_evaluation(validation, selected, edited, v_labels, v_baseline, after10)
        voted = start.apply_whole_rally_alternation(edited)
        contact_scores = {}
        for tolerance in (10, 5):
            raw = score_contacts(edited.events_by_fixture, v_labels, validation.fps, tolerance)
            contact_scores[str(tolerance)] = {"raw": raw, "fixed_side": voted_contact_scores(raw, voted.events_by_fixture)}
        result["contact_scores"] = contact_scores
        validation_results[variant] = result
        paired = result["evaluation"]["10"]["paired_fixed_side"]
        print("V", variant, "correct", paired["correct_after"],
              "repaired", len(paired["repaired"]), "lost", len(paired["lost"]), flush=True)

    output = {
        "schema": "contact-closing-whole-rally-comparison/1", "status": "complete",
        "counts": counts, "model_settings": WHOLE_MODEL_SETTINGS, "minimum_score_candidates": MINIMUM_SCORES,
        "policy_selection": "development net correct rallies; ties fewer losses, fewer edits, higher minimum score",
        "opening_model_settings": dict(opening_spec.settings), "target_tolerance_base30": 10,
        "lineage": {**lineage, "opening_scores_nested_inside_whole_model_folds": True,
                    "validation_status": "previously examined, reused validation; excluded from fitting"},
        "feature_join_audit": measurements.audit, "static_feature_names": names, "target_report": target_report,
        "timings": {"load_40_video_measurements_seconds": loading_seconds,
                    "development_feature_matrix_seconds": d_feature_seconds, "validation_feature_matrix_seconds": v_feature_seconds,
                    "opening_group_fits": opening_fit_records, "whole_group_fits": whole_fit_records,
                    "final_opening_fit_seconds": opening_fit_seconds, "total_seconds": perf_counter() - started},
        "development_baseline": d_baseline, "validation_baseline": v_baseline,
        "development_options": [option_record(option) for option in development.options],
        "development_descriptive": development_results, "validation": validation_results,
        "validation_baseline_contacts": baseline_contacts,
        "validation_reference_comparisons": reference_comparisons(validation_results),
        "prediction_file": PREDICTION_NAME,
    }
    write_json(output_root / RESULT_NAME, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=prediction_io.REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.feature_root, args.output_root, args.smoke)
    print("Finished", result["status"], flush=True)


if __name__ == "__main__":
    main()
