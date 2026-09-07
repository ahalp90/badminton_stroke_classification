"""Reconstruct the unchanged D-trained models and freeze them for broader prediction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.replay_simple_replacements import (
    option_key,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
    _select,
    _valid_action_spans,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    ROOT,
    prepare_population,
    selected_records,
    variant_matrix,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    action_matrix,
    build_whole_features,
    load_measurements,
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    WHOLE_MODEL_SETTINGS,
    fit_opening_models,
    fit_whole_model,
    predict_opening_models,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_options import (
    choose_options,
    whole_targets,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

VARIANT = "opening_sides_and_physics"


def final_training_opening_scores(population: Any, matrix: np.ndarray, targets: Any, spec: Any) -> tuple[dict, list]:
    """Recreate just the four group fits needed for final-model training features."""
    scores = {}
    records = []
    for excluded in GROUPS:
        allowed = frozenset(GROUPS) - {excluded}
        models, seconds = fit_opening_models(population.actions, matrix, targets, allowed, spec)
        indices = [index for index, row in enumerate(population.actions) if row.candidate.group == excluded]
        rows = [population.actions[index] for index in indices]
        scores.update(predict_opening_models(models, rows, matrix[indices]))
        records.append({"excluded_group": excluded, "fit_seconds": seconds})
    return scores, records


def validate_reference(models: dict[str, Any], validation: Any, measurements: Any) -> dict[str, Any]:
    """Check reconstructed models at the saved validation-choice boundary once."""
    action_values = action_matrix(validation.actions, measurements)
    static, names, columns = build_whole_features(
        validation.options, validation.spans, validation.actions, validation.fps, measurements,
    )
    if names != models["static_feature_names"] or columns != models["columns"]:
        raise ValueError("Validation feature layout differs from the frozen training layout")
    opening_scores = predict_opening_models(models["opening"], validation.actions, action_values)
    opening = opening_score_features(validation.options, opening_scores)
    matrix = variant_matrix(static, np.arange(len(validation.options)), columns, opening, VARIANT)
    scores = _positive_scores(models["whole"], matrix)
    selected = choose_options(validation.options, scores, 0.0)
    original = prediction_io.read_json(ROOT / "results/whole_rally_predictions.json.gz")["variants"][VARIANT]
    actual_keys = {option_key(row) for row in selected_records(selected)}
    expected_keys = {option_key(row) for row in original["selected_actions"]}
    if actual_keys != expected_keys:
        raise ValueError("Reconstructed whole-model validation decisions differ from the preserved reference")
    valid = _valid_action_spans(validation.actions, validation.spans, validation.events)
    summary_scores = {identity: pair[0] for identity, pair in opening_scores.items()}
    opening_selected = _select(validation.actions, summary_scores, valid)
    saved = prediction_io.read_json(ROOT / "results/start_comparison_predictions.json.gz")
    expected_opening = {
        (row["fixture"], row["span_id"], row["frame"], row["action"])
        for row in saved["variants"]["summary_opening"]["selected_actions"]
    }
    if {row.identity for row in opening_selected.values()} != expected_opening:
        raise ValueError("Reconstructed opening-only decisions differ from the preserved reference")
    return {"whole_selected_decisions_match": True, "whole_sections": len(selected),
            "maximum_whole_score_difference": float(np.max(np.abs(scores - np.asarray(original["scores"])))),
            "opening_selected_actions": len(opening_selected), "opening_selected_decisions_match": True}


def run(feature_root: Path, output_root: Path) -> None:
    started = perf_counter()
    pack = prediction_io.load_development_predictions()
    development = prepare_population(pack, frozenset(GROUPS), start._candidate_videos())
    raw_validation = prediction_io.read_json(prediction_io.VALIDATION_PREDICTIONS)["videos"]
    validation = prepare_population(pack, frozenset({"V"}), raw_validation)
    measurements = load_measurements((*development.actions, *validation.actions), pack.events_by_fixture, feature_root)
    matrix = action_matrix(development.actions, measurements)
    static, names, columns = build_whole_features(
        development.options, development.spans, development.actions, development.fps, measurements,
    )
    labels = load_human_labels(start.LABEL_PATH, development.videos)
    local_targets, _ = assign_targets(
        development.actions, development.spans, development.events, labels, development.fps,
    )
    targets, whole_report = whole_targets(development.options, development.spans, labels, development.fps)
    config = load_rally_start_model_config(start.CONFIG_PATH)
    spec = next(model for model in config.models if model.model_id == "shallow_hgb")
    training_scores, group_timings = final_training_opening_scores(development, matrix, local_targets, spec)
    opening = opening_score_features(development.options, training_scores)
    whole_matrix = variant_matrix(static, np.arange(len(development.options)), columns, opening, VARIANT)
    whole, whole_seconds = fit_whole_model(whole_matrix, targets)
    opening_models, opening_seconds = fit_opening_models(
        development.actions, matrix, local_targets, frozenset(GROUPS), spec,
    )
    models = {"whole": whole, "opening": opening_models, "static_feature_names": names, "columns": columns}
    reproduction = validate_reference(models, validation, measurements)
    output_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_root / "broader_models.joblib", compress=3)
    write_json(ROOT / "results/broader_model_freeze.json.gz", {
        "status": "frozen", "reference_commit": "24e4256", "training_groups": GROUPS,
        "whole_settings": WHOLE_MODEL_SETTINGS, "opening_settings": dict(spec.settings),
        "whole_target_report": whole_report, "local_target_reasons": dict(Counter(target.reason for target in local_targets.values())),
        "feature_join_audit": measurements.audit, "reference_reproduction": reproduction,
        "broader_labels_read": False, "models_file": "raw/broader_models.joblib",
        "timings": {"group_fits": group_timings, "whole_fit_seconds": whole_seconds,
                    "opening_fit_seconds": opening_seconds, "total_seconds": perf_counter() - started},
    })
    print("Frozen D-trained models", reproduction, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path,
                        default=prediction_io.REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "raw")
    args = parser.parse_args()
    run(args.feature_root, args.output_root)


if __name__ == "__main__":
    main()
