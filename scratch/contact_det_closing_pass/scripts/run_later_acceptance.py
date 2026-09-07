"""Fit nested acceptance models for the saved later-contact choices."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.later_acceptance_features import (
    acceptance_features,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    LaterOption,
    option_record,
    select_options,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    _expanded_matrix,
    _reference_selection,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    Population,
)
from scratch.contact_det_closing_pass.scripts.score_acceptance import (
    build_acceptance_rows,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    section_views,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    action_matrix,
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    WHOLE_MODEL_SETTINGS,
    fit_opening_models,
    fit_whole_model,
    predict_opening_models,
    training_opening_scores,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
DEFAULT_PREPARED = ROOT / "raw/later_run/prepared.joblib"
DEFAULT_PREDICTIONS = ROOT / "results/later/later_predictions.json.gz"
DEFAULT_OUTPUT_ROOT = ROOT / "raw/later_acceptance"
DEFAULT_RESULT_ROOT = ROOT / "results/later"
PAIR_CHECKPOINT_NAME = "nested_pair_fits.joblib"
MODEL_NAME = "models.joblib"
RESULT_NAME = "later_acceptance_result.json.gz"
POLICY_NAME = "later_acceptance_policy.json.gz"
PAIR_CACHE_SCHEMA = "contact-closing-later-acceptance-pair-cache/2"
REFERENCE_FEATURE_WIDTH = 304
TOLERANCES = ("10", "5")
MIN_JUDGED_ACCEPTED = 32
TARGET_PRECISION = 0.95
ACCEPTANCE_SEED = 20260905
ACCEPTANCE_SETTINGS = {
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "class_weight": None,
    "early_stopping": False,
    "random_state": ACCEPTANCE_SEED,
}
VARIANTS = ("selected_score", "all_evidence")
SectionIdentity = tuple[str, int]


def _option_identity(option: LaterOption) -> tuple[Any, ...]:
    return (
        option.span.fixture,
        option.span.span_id,
        option.base.kind,
        option.base.candidate_frame,
        option.base.deleted_frame,
        None if option.inserted is None else option.inserted.frame,
    )


def _load_global_scores(path: Path, options: Sequence[LaterOption]) -> np.ndarray:
    payload = prediction_io.read_json(path)
    records = payload.get("options")
    if payload.get("status") != "complete" or not isinstance(records, list):
        raise ValueError("saved later predictions are incomplete")
    if len(records) != len(options):
        raise ValueError("saved later prediction option count differs")
    scores = []
    for option, record in zip(options, records, strict=True):
        if not isinstance(record, Mapping):
            raise TypeError("saved later prediction option must be an object")
        identity = (
            record.get("fixture"), record.get("span_id"), record.get("kind"),
            record.get("candidate_frame"), record.get("deleted_frame"),
            record.get("inserted_frame"),
        )
        if identity != _option_identity(option):
            raise ValueError(f"saved later prediction option differs: {_option_identity(option)}")
        score = float(record["score"])
        if not np.isfinite(score):
            raise ValueError("saved later prediction score is not finite")
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _group_indices(options: Sequence[LaterOption], population: Population) -> dict[str, np.ndarray]:
    groups = np.asarray([population.groups[option.span.fixture] for option in options])
    return {group: np.flatnonzero(groups == group) for group in GROUPS}


def _subset_reference(
    options: Sequence[LaterOption], reference: Mapping[SectionIdentity, LaterOption],
) -> dict[SectionIdentity, LaterOption]:
    identities = {option.base.identity for option in options}
    return {identity: reference[identity] for identity in identities}


def _build_single_group_cache(
    population: Population, measurements: Any, local_targets: Mapping[Any, Any], spec: Any,
) -> tuple[dict[frozenset[str], dict[Any, tuple[float, float]]], list[dict[str, Any]]]:
    matrix = action_matrix(population.actions, measurements)
    cache: dict[frozenset[str], dict[Any, tuple[float, float]]] = {}
    records = []
    for training_group in GROUPS:
        models, fit_seconds = fit_opening_models(
            population.actions, matrix, local_targets, frozenset({training_group}), spec,
        )
        predicted_indices = [
            index for index, row in enumerate(population.actions)
            if row.candidate.group != training_group
        ]
        predicted = predict_opening_models(
            models,
            [population.actions[index] for index in predicted_indices],
            matrix[predicted_indices],
        )
        cache[frozenset({training_group})] = predicted
        records.append({
            "training_groups": [training_group],
            "predicted_groups": sorted(set(GROUPS) - {training_group}),
            "fit_seconds": fit_seconds,
            "predicted_actions": len(predicted_indices),
        })
    return cache, records


def _fit_pair_scores(
    population: Population, options: Sequence[LaterOption], static: np.ndarray,
    insertion: np.ndarray, targets: np.ndarray, opening_cache: Mapping[Any, Any],
    local_targets: Mapping[Any, Any], measurements: Any, spec: Any,
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[SectionIdentity, LaterOption]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    singleton, singleton_records = _build_single_group_cache(
        population, measurements, local_targets, spec,
    )
    groups = np.asarray([population.groups[option.span.fixture] for option in options])
    pair_scores: dict[str, np.ndarray] = {}
    pair_references: dict[str, dict[SectionIdentity, LaterOption]] = {}
    records = []
    for pair_tuple in combinations(GROUPS, 2):
        allowed = frozenset(pair_tuple)
        train_indices = np.flatnonzero(np.isin(groups, tuple(allowed)))
        held_indices = np.flatnonzero(~np.isin(groups, tuple(allowed)))
        train_options = [options[index] for index in train_indices]
        held_options = [options[index] for index in held_indices]
        train_opening = opening_score_features(
            [option.proxy for option in train_options],
            training_opening_scores(population.actions, singleton, allowed),
        )
        held_opening = opening_score_features(
            [option.proxy for option in held_options], opening_cache[allowed],
        )
        model, fit_seconds = fit_whole_model(
            _expanded_matrix(static[train_indices], insertion[train_indices], train_opening),
            targets[train_indices],
        )
        scores = np.full(len(options), np.nan, dtype=np.float64)
        scores[held_indices] = _positive_scores(
            model,
            _expanded_matrix(static[held_indices], insertion[held_indices], held_opening),
        )
        key = "".join(pair_tuple)
        pair_scores[key] = scores

        train_noninsert_positions = np.asarray([
            position for position, index in enumerate(train_indices)
            if options[index].inserted is None
        ])
        held_noninsert_positions = np.asarray([
            position for position, index in enumerate(held_indices)
            if options[index].inserted is None
        ])
        reference_matrix_width = static.shape[1] + train_opening.shape[1]
        if reference_matrix_width != REFERENCE_FEATURE_WIDTH:
            raise ValueError(f"reference chooser expects {REFERENCE_FEATURE_WIDTH} features")
        reference_model, reference_fit_seconds = fit_whole_model(
            np.column_stack((
                static[train_indices[train_noninsert_positions]],
                train_opening[train_noninsert_positions],
            )),
            targets[train_indices[train_noninsert_positions]],
        )
        reference_scores = _positive_scores(
            reference_model,
            np.column_stack((
                static[held_indices[held_noninsert_positions]],
                held_opening[held_noninsert_positions],
            )),
        )
        reference_options = [options[index] for index in held_indices[held_noninsert_positions]]
        pair_references[key] = select_options(reference_options, reference_scores)
        records.append({
            "allowed_training_groups": list(pair_tuple),
            "predicted_groups": sorted(set(GROUPS) - allowed),
            "fit_seconds": fit_seconds,
            "reference_fit_seconds": reference_fit_seconds,
            "reference_feature_width": reference_matrix_width,
            "training_options": len(train_indices),
            "predicted_options": len(held_indices),
        })
        print(f"Acceptance pair {key} complete", flush=True)
    return pair_scores, pair_references, records, singleton_records


def _load_or_fit_pairs(
    checkpoint: Path, population: Population, options: Sequence[LaterOption],
    static: np.ndarray, insertion: np.ndarray, targets: np.ndarray,
    opening_cache: Mapping[Any, Any], local_targets: Mapping[Any, Any], measurements: Any,
    spec: Any,
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[SectionIdentity, LaterOption]],
    list[dict[str, Any]], list[dict[str, Any]], bool,
]:
    if checkpoint.exists():
        saved = joblib.load(checkpoint)
        expected_policy = {
            "minimum_advantage": MIN_EDIT_ADVANTAGE,
            "reference_fit_feature_width": REFERENCE_FEATURE_WIDTH,
            "reference_options_are_noninsert": True,
            "reference_selector": "select_options",
        }
        if saved.get("schema") != PAIR_CACHE_SCHEMA or saved.get("reference_policy") != expected_policy:
            raise ValueError("nested pair checkpoint reference policy differs")
        pair_scores = {key: np.asarray(value, dtype=np.float64) for key, value in saved["pair_scores"].items()}
        if any(values.shape != (len(options),) for values in pair_scores.values()):
            raise ValueError("nested pair checkpoint has the wrong option coverage")
        pair_references = saved["pair_references"]
        if set(pair_references) != set(pair_scores):
            raise ValueError("nested pair checkpoint reference coverage differs")
        for references in pair_references.values():
            if not isinstance(references, Mapping) or any(
                not isinstance(option, LaterOption) or option.inserted is not None
                for option in references.values()
            ):
                raise ValueError("nested pair checkpoint reference format differs")
        return pair_scores, pair_references, saved["pair_fit_records"], saved["singleton_fit_records"], True
    pair_scores, pair_references, pair_records, singleton_records = _fit_pair_scores(
        population, options, static, insertion, targets, opening_cache,
        local_targets, measurements, spec,
    )
    joblib.dump({
        "schema": PAIR_CACHE_SCHEMA,
        "reference_policy": {
            "minimum_advantage": MIN_EDIT_ADVANTAGE,
            "reference_fit_feature_width": REFERENCE_FEATURE_WIDTH,
            "reference_options_are_noninsert": True,
            "reference_selector": "select_options",
        },
        "pair_scores": pair_scores,
        "pair_references": pair_references,
        "pair_fit_records": pair_records,
        "singleton_fit_records": singleton_records,
    }, checkpoint, compress=3)
    return pair_scores, pair_references, pair_records, singleton_records, False


def _section_judgements(
    population: Population, selected: Mapping[SectionIdentity, LaterOption],
    score_by_identity: Mapping[SectionIdentity, float], labels: Any,
) -> list[dict[str, Any]]:
    spans = tuple(option.span for option in selected.values())
    fixtures = {span.fixture for span in spans}
    fps = {fixture: population.fps[fixture] for fixture in fixtures}
    groups = {fixture: population.groups[fixture] for fixture in fixtures}
    views = {
        tolerance: section_views(spans, labels, fps, groups, int(tolerance))["fixed_side"]["sections"]
        for tolerance in TOLERANCES
    }
    choices = [
        option_record(option) | {"score": float(score_by_identity[identity])}
        for identity, option in selected.items()
    ]
    return build_acceptance_rows(views, choices, labels, {})


def _group_features(
    group: str, indices: np.ndarray, options: Sequence[LaterOption], scores: np.ndarray,
    candidates: Mapping[SectionIdentity, Sequence[Any]], population: Population,
    measurements: Any, labels: Any, reference: Mapping[SectionIdentity, LaterOption],
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]], dict[SectionIdentity, LaterOption]]:
    group_options = [options[index] for index in indices]
    group_scores = scores[indices]
    reference_subset = _subset_reference(group_options, reference)
    selected = select_with_reference(
        group_options, group_scores, reference_subset, minimum_advantage=MIN_EDIT_ADVANTAGE,
    )
    selected_scores = {}
    for option, score in zip(group_options, group_scores, strict=True):
        if selected.get(option.base.identity) == option:
            selected_scores[option.base.identity] = float(score)
    rows = _section_judgements(population, selected, selected_scores, labels)
    row_by_identity = {(row["fixture"], row["span_id"]): row for row in rows}
    fps = population.fps
    feature_values, names, identities = acceptance_features(
        group_options, group_scores, selected, candidates, fps, measurements,
    )
    if len(feature_values) != len(rows) or set(identities) != set(row_by_identity):
        raise ValueError(f"{group}: acceptance feature and judgement coverage differs")
    return feature_values, names, [row_by_identity[identity] for identity in identities], selected


def _outcome_value(row: Mapping[str, Any], tolerance: str) -> int:
    outcome = row["judgements"][tolerance]["outcome"]
    if outcome == "correct":
        return 1
    if outcome == "wrong":
        return 0
    return -1


def _fit_acceptance_models(
    features: np.ndarray, labels: np.ndarray,
) -> tuple[dict[str, Any], float]:
    known = labels >= 0
    if set(labels[known].tolist()) != {0, 1}:
        raise ValueError("acceptance fit needs both known classes")
    models = {}
    started = perf_counter()
    from sklearn.ensemble import HistGradientBoostingClassifier

    for variant in VARIANTS:
        columns = slice(None) if variant == "all_evidence" else slice(0, 1)
        model = HistGradientBoostingClassifier(**ACCEPTANCE_SETTINGS)
        model.fit(features[known, columns], labels[known])
        models[variant] = model
    return models, perf_counter() - started


def _nested_oof(
    population: Population, options: Sequence[LaterOption],
    candidates: Mapping[SectionIdentity, Sequence[Any]], measurements: Any,
    pair_scores: Mapping[str, np.ndarray],
    pair_references: Mapping[str, Mapping[SectionIdentity, LaterOption]],
    global_scores: np.ndarray, global_reference: Mapping[SectionIdentity, LaterOption], labels: Any,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], tuple[str, ...]]:
    by_group = _group_indices(options, population)
    output_rows: list[dict[str, Any]] = []
    output_features = []
    fit_records = []
    feature_names: tuple[str, ...] | None = None
    for held_group in GROUPS:
        train_features = []
        train_labels = []
        for training_group in GROUPS:
            if training_group == held_group:
                continue
            pair = "".join(group for group in GROUPS if group not in {held_group, training_group})
            features, names, rows, _selected = _group_features(
                training_group, by_group[training_group], options, pair_scores[pair],
                candidates, population, measurements, labels, pair_references[pair],
            )
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("acceptance feature names differ across folds")
            train_features.append(features)
            train_labels.append(np.asarray([_outcome_value(row, "10") for row in rows], dtype=np.int8))
        all_features = np.vstack(train_features)
        all_labels = np.concatenate(train_labels)
        models, fit_seconds = _fit_acceptance_models(all_features, all_labels)
        held_features, names, held_rows, selected = _group_features(
            held_group, by_group[held_group], options, global_scores,
            candidates, population, measurements, labels, global_reference,
        )
        if names != feature_names:
            raise ValueError("held acceptance feature names differ")
        output_features.append(held_features)
        scores = {
            variant: _positive_scores(
                models[variant], held_features[:, :1] if variant == "selected_score" else held_features,
            )
            for variant in VARIANTS
        }
        for index, row in enumerate(held_rows):
            identity = (str(row["fixture"]), int(row["span_id"]))
            chosen = selected[identity]
            output_rows.append({
                "fixture": identity[0],
                "span_id": identity[1],
                "group": population.groups[identity[0]],
                "outer_held_group": held_group,
                "raw_selected_score": float(held_features[index, 0]),
                "acceptance_selected_score": float(scores["selected_score"][index]),
                "acceptance_all_evidence_score": float(scores["all_evidence"][index]),
                "judgements": row["judgements"],
                "selected_action": option_record(chosen),
            })
        fit_records.append({
            "held_out_group": held_group,
            "training_groups": sorted(set(GROUPS) - {held_group}),
            "training_rows": len(all_labels),
            "known_training_rows": int(np.count_nonzero(all_labels >= 0)),
            "fit_seconds": fit_seconds,
        })
        print(f"Acceptance outer group {held_group} complete", flush=True)
    if feature_names is None:
        raise ValueError("acceptance OOF produced no rows")
    return output_rows, np.vstack(output_features), fit_records, feature_names


def _base_partition_metrics(
    rows: Sequence[Mapping[str, Any]], score_key: str, threshold: float,
) -> dict[str, Any]:
    accepted = [row for row in rows if float(row[score_key]) >= threshold]
    rejected = [row for row in rows if float(row[score_key]) < threshold]
    by_tolerance = {}
    for tolerance in TOLERANCES:
        counts = Counter(row["judgements"][tolerance]["outcome"] for row in accepted)
        correct = int(counts["correct"])
        wrong = int(counts["wrong"])
        unjudgeable = int(counts["unjudgeable"])
        judged = correct + wrong
        rejected_correct = sum(row["judgements"][tolerance]["outcome"] == "correct" for row in rejected)
        by_tolerance[tolerance] = {
            "counts": {"correct": correct, "wrong": wrong, "unjudgeable": unjudgeable},
            "judged_count": judged,
            "judged_precision": correct / judged if judged else None,
            "verified_correct_share_allaccepted": correct / len(accepted) if accepted else None,
            "rejected_correct": int(rejected_correct),
        }

    return {
        "population_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "by_tolerance": by_tolerance,
    }


def _partition_metrics(rows: Sequence[Mapping[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    summary = _base_partition_metrics(rows, score_key, threshold)
    for field, output_name in (("group", "by_group"), ("fixture", "by_video")):
        summary[output_name] = {
            value: _base_partition_metrics(
                [row for row in rows if str(row[field]) == value], score_key, threshold,
            )
            for value in sorted({str(row[field]) for row in rows})
        }
    return summary


def _tail_curve(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("acceptance curve has no rows")
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_key}: acceptance scores are incomplete")
    requested = (min(32, len(rows)), max(1, int(np.ceil(0.05 * len(rows)))),
                 max(1, int(np.ceil(0.10 * len(rows)))), max(1, int(np.ceil(0.20 * len(rows)))),
                 max(1, int(np.ceil(0.40 * len(rows)))))
    labels = ("top32", "top5pct", "top10pct", "top20pct", "top40pct")
    ordered = np.sort(scores)[::-1]
    curve = []
    for label, count in zip(labels, requested, strict=True):
        threshold = float(ordered[count - 1])
        curve.append({
            "tail": label,
            "requested_count": count,
            **_partition_metrics(rows, score_key, threshold),
            "threshold": threshold,
        })
    return curve


def _policy(score_key: str, curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_rules = {}
    for target in (0.95, 0.99):
        qualifying = [
            record for record in curve
            if record["by_tolerance"]["10"]["judged_count"] >= MIN_JUDGED_ACCEPTED
            and record["by_tolerance"]["10"]["judged_precision"] is not None
            and record["by_tolerance"]["10"]["judged_precision"] >= target
        ]
        selected = max(
            qualifying,
            key=lambda record: (
                record["accepted_count"], record["by_tolerance"]["10"]["judged_precision"],
            ),
            default=None,
        )
        fallback_candidates = [
            record for record in curve
            if record["by_tolerance"]["10"]["judged_count"] >= MIN_JUDGED_ACCEPTED
        ]
        fallback = max(
            fallback_candidates,
            key=lambda record: (
                record["by_tolerance"]["10"]["judged_precision"],
                record["accepted_count"],
            ),
            default=None,
        )
        target_rules[str(target)] = {
            "target_judged_precision": target,
            "target_status": "met" if selected is not None else "unmet",
            "selected_rule": dict(selected) if selected is not None else None,
            "nonempty_fallback": dict(fallback) if fallback is not None else None,
        }
    primary = target_rules[str(TARGET_PRECISION)]
    return {
        "score_key": score_key,
        "minimum_judged_accepted": MIN_JUDGED_ACCEPTED,
        "target_rules": target_rules,
        "target_status": primary["target_status"],
        "selected_rule": primary["selected_rule"],
        "nonempty_fallback": primary["nonempty_fallback"],
        "curve": [dict(record) for record in curve],
    }


def run(
    prepared_path: Path = DEFAULT_PREPARED,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    result_root: Path = DEFAULT_RESULT_ROOT,
) -> dict[str, Any]:
    started = perf_counter()
    prepared = joblib.load(prepared_path)
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    static = np.asarray(prepared["static_features"], dtype=np.float64)
    insertion = np.asarray(prepared["insertion_features"], dtype=np.float64)
    targets = np.asarray(prepared["targets"], dtype=np.int8)
    candidates = prepared["later_candidates"]
    measurements = prepared["measurements"]
    opening_cache = prepared["opening_cache"]
    local_targets = prepared["local_targets"]
    if len(static) != len(options) or len(insertion) != len(options) or len(targets) != len(options):
        raise ValueError("prepared later feature coverage differs")
    global_scores = _load_global_scores(predictions_path, options)
    parsed_config = load_rally_start_model_config(start.CONFIG_PATH)
    opening_spec = next(model for model in parsed_config.models if model.model_id == "shallow_hgb")
    output_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / PAIR_CHECKPOINT_NAME
    pair_started = perf_counter()
    pair_scores, pair_references, pair_records, singleton_records, pair_cached = _load_or_fit_pairs(
        checkpoint, population, options, static, insertion, targets, opening_cache,
        local_targets, measurements, opening_spec,
    )
    pair_elapsed = perf_counter() - pair_started
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    global_reference = _reference_selection(
        population.options, frozenset(video.fixture for video in population.videos),
    )
    nested_rows, nested_features, outer_fit_records, feature_names = _nested_oof(
        population, options, candidates, measurements, pair_scores, pair_references,
        global_scores, global_reference, labels,
    )
    curves = {}
    policies = {}
    score_keys = {
        "raw_selected_score": "raw_selected_score",
        "selected_score": "acceptance_selected_score",
        "all_evidence": "acceptance_all_evidence_score",
    }
    for name, score_key in score_keys.items():
        curves[name] = _tail_curve(nested_rows, score_key)
        policies[name] = _policy(score_key, curves[name])
    final_labels = np.asarray([_outcome_value(row, "10") for row in nested_rows], dtype=np.int8)
    final_models, final_seconds = _fit_acceptance_models(nested_features, final_labels)
    final_names = feature_names
    model_payload = {
        "schema": "contact-closing-later-acceptance-models/1",
        "models": final_models,
        "feature_names": list(final_names),
        "selected_score_feature_names": [final_names[0]],
        "all_evidence_feature_names": list(final_names),
        "frozen_policies": policies,
        "minimum_edit_advantage": MIN_EDIT_ADVANTAGE,
        "reference_policy": "select_with_reference",
    }
    joblib.dump(model_payload, output_root / MODEL_NAME, compress=3)
    policy_payload = {
        "schema": "contact-closing-later-acceptance-policy/1",
        "status": "complete",
        "selection_data": "nested development predictions",
        "broader_labels_used": False,
        "scores_are_calibrated_probabilities": False,
        "policies": policies,
        "feature_names": list(feature_names),
    }
    write_json(result_root / POLICY_NAME, policy_payload)
    for name, curve in curves.items():
        print(name, [(row["tail"], row["by_tolerance"]["10"]["judged_precision"]) for row in curve], flush=True)
    result = {
        "schema": "contact-closing-later-acceptance/1",
        "status": "complete",
        "counts": {
            "development_videos": len(population.videos),
            "development_sections": len(population.spans),
            "later_options": len(options),
            "nested_oof_rows": len(nested_rows),
        },
        "prediction_file": predictions_path.name,
        "prepared_file": prepared_path.name,
        "pair_checkpoint_file": checkpoint.name,
        "model_file": MODEL_NAME,
        "policy_file": POLICY_NAME,
        "rows": nested_rows,
        "curves": curves,
        "policies": policies,
        "feature_names": list(feature_names),
        "model_settings": ACCEPTANCE_SETTINGS,
        "upstream_whole_model_settings": WHOLE_MODEL_SETTINGS,
        "minimum_edit_advantage": MIN_EDIT_ADVANTAGE,
        "lineage": {
            "outer_groups": list(GROUPS),
            "nested_pair_whole_fits": True,
            "reference_margin_policy": "select_with_reference",
            "pair_references_are_nested_inside_outer_acceptance_folds": True,
            "global_reference_is_frozen_combined_development_selection": True,
            "pair_training_opening_scores_exclude_row_group": True,
            "pair_prediction_opening_scores_reuse_saved_pair_cache": True,
            "acceptance_training_uses_pair_specific_detector_scores": True,
            "acceptance_prediction_uses_saved_global_oof_scores": True,
            "held_group_labels_used_for_acceptance_fit": False,
            "detector_and_acceptance_prediction_selection_uses_labels": False,
            "acceptance_rule_selection_data": "development labels applied to nested OOF predictions",
            "global_detector_scores_retain_upstream_cross_group_dependence": True,
            "global_whole_predictions_are_reused_for_held_acceptance_rows": True,
            "outputs_remain_unchanged": True,
            "validation_used": False,
        },
        "timings": {
            "pair_fit_seconds": pair_elapsed,
            "pair_fit_cached": pair_cached,
            "singleton_opening_fits": singleton_records,
            "pair_whole_fits": pair_records,
            "outer_acceptance_fits": outer_fit_records,
            "final_acceptance_fit_seconds": final_seconds,
            "total_seconds": perf_counter() - started,
        },
    }
    write_json(result_root / RESULT_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    run(args.prepared, args.predictions, args.output_root, args.result_root)


if __name__ == "__main__":
    main()
