"""Fit group-isolated local insertion scores for the later chooser."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    fit_whole_model,
)


def _validated_inputs(
    matrix: np.ndarray, targets: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalise the row-aligned local-learning inputs."""
    matrix = np.asarray(matrix)
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    if matrix.ndim != 2:
        raise ValueError("local feature matrix must be two-dimensional")
    if targets.ndim != 1 or groups.ndim != 1:
        raise ValueError("local targets and groups must be one-dimensional")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("local targets must be an integer array")
    if len(matrix) != len(targets) or len(matrix) != len(groups):
        raise ValueError("local feature rows, targets and groups must have equal lengths")
    return matrix, targets, groups


def build_local_cache(
    matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    training_sets: Sequence[frozenset[str]],
) -> tuple[dict[frozenset[str], np.ndarray], dict[frozenset[str], Any], list[dict[str, Any]]]:
    """Fit one local model per allowed group set and cache held-group scores.

    ``fit_whole_model`` receives every row from an allowed group, including
    rows whose target is ``-1``.  Its existing target filter excludes those
    rows during fitting.  Prediction remains label-independent and covers all
    rows outside the allowed set.

    :param matrix: Local insertion features with one row per option.
    :param targets: Integer targets aligned to ``matrix``; ``-1`` is unknown.
    :param groups: Fixture group for each option row.
    :param training_sets: Explicit group sets on which to fit models.
    :return: Full-length score vectors, fitted models and fit metadata.
    """
    matrix, targets, groups = _validated_inputs(matrix, targets, groups)
    normalised_sets = tuple(frozenset(training_set) for training_set in training_sets)
    if len(set(normalised_sets)) != len(normalised_sets):
        raise ValueError("local training sets repeat")

    observed_groups = {str(group) for group in groups}
    cache: dict[frozenset[str], np.ndarray] = {}
    models: dict[frozenset[str], Any] = {}
    fit_records: list[dict[str, Any]] = []
    for allowed in normalised_sets:
        allowed_rows = np.isin(groups, tuple(allowed))
        training_indices = np.flatnonzero(allowed_rows)
        predicted_indices = np.flatnonzero(~allowed_rows)
        model, fit_seconds = fit_whole_model(
            matrix[training_indices], targets[training_indices],
        )
        scores = np.full(len(groups), np.nan, dtype=np.float64)
        if len(predicted_indices):
            scores[predicted_indices] = _positive_scores(model, matrix[predicted_indices])
        cache[allowed] = scores
        models[allowed] = model
        fit_records.append({
            "training_groups": sorted(allowed),
            "predicted_groups": sorted(observed_groups - allowed),
            "training_rows": len(training_indices),
            "known_training_rows": int(np.count_nonzero(targets[training_indices] >= 0)),
            "predicted_rows": len(predicted_indices),
            "fit_seconds": fit_seconds,
        })
    return cache, models, fit_records


def local_training_scores(
    groups: np.ndarray,
    cache: dict[frozenset[str], np.ndarray],
    allowed_groups: frozenset[str],
) -> np.ndarray:
    """Assemble scores for allowed groups from caches excluding each row group.

    The returned vector is finite only for rows whose group is in
    ``allowed_groups``.  Rows outside that set remain ``NaN`` because they are
    reserved for the outer prediction side of the fold.

    :param groups: Fixture group for each option row.
    :param cache: Full-length score vectors keyed by fitting group set.
    :param allowed_groups: Groups whose training features should be assembled.
    :return: One full-length, row-aligned local-score vector.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("local groups must be one-dimensional")
    scores = np.full(len(groups), np.nan, dtype=np.float64)
    for group in allowed_groups:
        group_indices = np.flatnonzero(groups == group)
        if not len(group_indices):
            continue
        cache_key = allowed_groups - {group}
        if cache_key not in cache:
            raise KeyError(f"local score cache is missing group set {sorted(cache_key)}")
        cached = np.asarray(cache[cache_key])
        if cached.shape != (len(groups),):
            raise ValueError(f"local score cache {sorted(cache_key)} has the wrong row coverage")
        requested = cached[group_indices]
        if not np.isfinite(requested).all():
            raise ValueError(f"local score cache {sorted(cache_key)} has nonfinite requested scores")
        scores[group_indices] = requested
    return scores
