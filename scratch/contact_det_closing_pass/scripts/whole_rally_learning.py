"""Fit local opening evidence without exposing an outer group's edit labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from time import perf_counter
from typing import Any

import numpy as np

from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.targets import EditTarget
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import ModelSpec

ActionIdentity = tuple[str, int, int, str]
OpeningScores = dict[ActionIdentity, tuple[float, float]]
GROUPS = ("A", "B", "C", "D")
WHOLE_MODEL_SETTINGS = {
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": 20260905,
}


def fit_opening_models(
    rows: Sequence[start.ActionRow], matrix: np.ndarray,
    targets: Mapping[ActionIdentity, EditTarget], training_groups: frozenset[str], spec: ModelSpec,
) -> tuple[tuple[Any, Any], float]:
    """Fit summary and physical opening models on the explicitly allowed groups."""
    indices = []
    for index, row in enumerate(rows):
        if row.candidate.group in training_groups and targets[row.identity].included:
            indices.append(index)
    answers = np.asarray([targets[rows[index].identity].opening_correct for index in indices], dtype=np.uint8)
    if set(answers.tolist()) != {0, 1}:
        raise ValueError("Opening fit needs both target classes in the allowed training groups")
    models = []
    started = perf_counter()
    for width in (len(start.ACTION_FEATURE_NAMES), matrix.shape[1]):
        model = start.make_action_model(spec)
        model.fit(matrix[indices, :width], answers)
        models.append(model)
    return (models[0], models[1]), perf_counter() - started


def predict_opening_models(
    models: tuple[Any, Any], rows: Sequence[start.ActionRow], matrix: np.ndarray,
) -> OpeningScores:
    summary_scores = _positive_scores(models[0], matrix[:, :len(start.ACTION_FEATURE_NAMES)])
    physical_scores = _positive_scores(models[1], matrix)
    return {
        row.identity: (float(summary), float(physical))
        for row, summary, physical in zip(rows, summary_scores, physical_scores, strict=True)
    }


def build_opening_cache(
    rows: Sequence[start.ActionRow], matrix: np.ndarray,
    targets: Mapping[ActionIdentity, EditTarget], spec: ModelSpec,
) -> tuple[dict[frozenset[str], OpeningScores], list[dict[str, object]]]:
    """Cache the two- and three-group fits needed for nested edit-model scores.

    Two-group fits supply training features inside each outer three-group fit.
    Three-group fits supply outer predictions and the final four-group training
    features. Cached detector scores still have their earlier upstream dependence.
    """
    cache = {}
    records = []
    for size in (2, 3):
        for names in combinations(GROUPS, size):
            training_groups = frozenset(names)
            models, elapsed = fit_opening_models(rows, matrix, targets, training_groups, spec)
            indices = [index for index, row in enumerate(rows) if row.candidate.group not in training_groups]
            predicted_rows = [rows[index] for index in indices]
            cache[training_groups] = predict_opening_models(models, predicted_rows, matrix[indices])
            records.append({"training_groups": names, "predicted_groups": sorted(set(GROUPS) - training_groups),
                            "fit_seconds": elapsed, "predicted_actions": len(indices)})
    return cache, records


def training_opening_scores(
    rows: Sequence[start.ActionRow], cache: Mapping[frozenset[str], OpeningScores],
    allowed_groups: frozenset[str],
) -> OpeningScores:
    """Use scores fitted without either the row's group or an outer held-out group."""
    scores = {}
    for row in rows:
        group = row.candidate.group
        if group in allowed_groups:
            scores[row.identity] = cache[allowed_groups - {group}][row.identity]
    return scores


def fit_whole_model(matrix: np.ndarray, answers: np.ndarray) -> tuple[Any, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    included = answers >= 0
    if set(answers[included].tolist()) != {0, 1}:
        raise ValueError("Whole-rally fit needs positive and negative labelled alternatives")
    model = HistGradientBoostingClassifier(**WHOLE_MODEL_SETTINGS)
    started = perf_counter()
    model.fit(matrix[included], answers[included])
    return model, perf_counter() - started
