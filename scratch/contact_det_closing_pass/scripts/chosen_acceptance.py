"""Build the nested local scores and guarded features for chosen acceptance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.gap_evidence import gap_evidence
from scratch.contact_det_closing_pass.scripts.insertion_learning import (
    build_local_cache,
    local_training_scores,
)
from scratch.contact_det_closing_pass.scripts.later_acceptance_features import (
    acceptance_features,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    LaterOption,
    apply_options,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.local_insertion import insertion_targets
from scratch.contact_det_closing_pass.scripts.run_insertion_followup import (
    local_columns,
)
from scratch.contact_det_closing_pass.scripts.run_later_acceptance import (
    _build_single_group_cache,
    _section_judgements,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _expanded_matrix,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    OPENING_FEATURE_NAMES,
    Population,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    fit_whole_model,
    training_opening_scores,
)
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.score_keep_review import (
    FEATURE_NAMES,
    build_feature_vector,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

PAIR_SOURCE = ROOT / "raw/later_acceptance/nested_pair_fits.joblib"
LOCAL_SOURCE = ROOT / "raw/followups/local_local_cache.joblib"
SINGLETON_OPENING_NAME = "chosen_singleton_opening_cache.joblib"
SINGLETON_LOCAL_NAME = "chosen_singleton_local_cache.joblib"
PAIR_OUTPUT_NAME = "chosen_nested_local_pair_fits.joblib"
SCHEMA = "contact-closing-chosen-acceptance-local-pair-cache/1"
SectionIdentity = tuple[str, int]


def _set_key(key: object) -> frozenset[str]:
    if isinstance(key, frozenset):
        return frozenset(str(value) for value in key)
    if isinstance(key, (set, tuple, list)):
        return frozenset(str(value) for value in key)
    if isinstance(key, str) and len(key) == 2:
        return frozenset(key)
    raise TypeError(f"Cannot interpret group-set cache key {key!r}")


def _normalise_cache(cache: Mapping[Any, Any]) -> dict[frozenset[str], Any]:
    normalised = {}
    for key, value in cache.items():
        normalised[_set_key(key)] = value
    return normalised


def _option_order(options: Sequence[LaterOption]) -> dict[str, Any]:
    if not options:
        raise ValueError("chosen acceptance needs at least one option")
    section_count = len({option.base.identity for option in options})
    return {
        "count": len(options),
        "section_count": section_count,
        "first": option_record(options[0]),
        "last": option_record(options[-1]),
    }


def _group_rows(options: Sequence[LaterOption], population: Population) -> np.ndarray:
    return np.asarray([population.groups[option.span.fixture] for option in options])


def _load_old_pairs(options: Sequence[LaterOption]) -> tuple[
    dict[frozenset[str], np.ndarray],
    dict[frozenset[str], dict[SectionIdentity, LaterOption]],
]:
    saved = joblib.load(PAIR_SOURCE)
    old_scores = _normalise_cache(saved["pair_scores"])
    old_references = _normalise_cache(saved["pair_references"])
    expected = {frozenset(pair) for pair in combinations(GROUPS, 2)}
    if set(old_scores) != expected or set(old_references) != expected:
        raise ValueError("saved later pair cache does not cover the six allowed pairs")
    scores = {}
    references = {}
    for allowed in expected:
        values = np.asarray(old_scores[allowed], dtype=np.float64)
        if values.shape != (len(options),):
            raise ValueError(f"saved pair scores {sorted(allowed)} have the wrong option coverage")
        if not isinstance(old_references[allowed], Mapping):
            raise TypeError(f"saved pair references {sorted(allowed)} are not a mapping")
        scores[allowed] = values
        references[allowed] = dict(old_references[allowed])
    return scores, references


def _opening_cache(
    output_root: Path, population: Population, measurements: Any, local_targets: Mapping[Any, Any],
) -> tuple[dict[frozenset[str], Any], list[dict[str, Any]]]:
    path = output_root / SINGLETON_OPENING_NAME
    if path.exists():
        saved = joblib.load(path)
        cache = _normalise_cache(saved["cache"])
        records = list(saved["fit_records"])
        if set(cache) != {frozenset({group}) for group in GROUPS}:
            raise ValueError("chosen singleton opening cache coverage differs")
        return cache, records
    config = load_rally_start_model_config(start.CONFIG_PATH)
    spec = next(model for model in config.models if model.model_id == "shallow_hgb")
    cache, records = _build_single_group_cache(population, measurements, local_targets, spec)
    joblib.dump({"schema": SCHEMA, "cache": cache, "fit_records": records}, path, compress=3)
    return cache, records


def _local_pack(
    output_root: Path, prepared: Mapping[str, Any], population: Population,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = output_root / SINGLETON_LOCAL_NAME
    if path.exists():
        saved = joblib.load(path)
        cache = _normalise_cache(saved["cache"])
        if not {frozenset({group}) for group in GROUPS} <= set(cache):
            raise ValueError("chosen singleton local cache coverage differs")
        return saved, list(saved.get("singleton_fit_records", []))
    source = joblib.load(LOCAL_SOURCE)
    source_cache = _normalise_cache(source["cache"])
    source_models = _normalise_cache(source["models"])
    pair_sets = {frozenset(pair) for pair in combinations(GROUPS, 2)}
    triple_sets = {frozenset(names) for names in combinations(GROUPS, 3)}
    if not pair_sets | triple_sets <= set(source_cache) or not pair_sets | triple_sets <= set(source_models):
        raise ValueError("saved local cache must contain all pair and triple fits")
    options = tuple(prepared["options"])
    insertion = np.asarray(prepared["insertion_features"], dtype=np.float64)
    contexts = tuple(option for option in options if option.inserted is not None)
    included = np.asarray([option.inserted is not None for option in options], dtype=bool)
    matrix = insertion[included]
    if len(matrix) != len(contexts):
        raise ValueError("local insertion rows do not align with contexts")
    if any(len(option.inserted_events) != 1 for option in contexts):
        raise ValueError("chosen local cache requires one inserted event per context")
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    targets = insertion_targets(contexts, population.spans, labels, population.fps)
    groups = np.asarray([population.groups[option.span.fixture] for option in contexts])
    source_groups = np.asarray(source["groups"])
    if source_groups.shape != groups.shape or not np.array_equal(source_groups, groups):
        raise ValueError("saved local cache context groups differ from chosen-detector contexts")
    if tuple(source.get("feature_names", ())) != tuple(prepared["insertion_feature_names"]):
        raise ValueError("saved local cache feature names differ from chosen detector")
    singleton_sets = [frozenset({group}) for group in GROUPS]
    singleton_cache, singleton_models, singleton_records = build_local_cache(
        matrix, targets, groups, singleton_sets,
    )
    cache = {**source_cache, **singleton_cache}
    models = {**source_models, **singleton_models}
    local = {
        **source,
        "schema": SCHEMA,
        "cache": cache,
        "models": models,
        "groups": groups,
        "feature_names": tuple(prepared["insertion_feature_names"]),
        "singleton_fit_records": singleton_records,
        "target_counts": {str(target): int(np.count_nonzero(targets == target)) for target in (-1, 0, 1)},
    }
    joblib.dump(local, path, compress=3)
    return local, singleton_records


def _preceding_reference(
    options: Sequence[LaterOption], indices: np.ndarray, scores: np.ndarray,
    references: Mapping[SectionIdentity, LaterOption],
) -> dict[SectionIdentity, LaterOption]:
    held_options = [options[index] for index in indices]
    held_scores = np.asarray(scores[indices], dtype=np.float64)
    if not np.isfinite(held_scores).all():
        raise ValueError("saved preceding pair scores are incomplete on the held groups")
    return select_with_reference(
        held_options,
        held_scores,
        {identity: references[identity] for identity in {option.base.identity for option in held_options}},
        minimum_advantage=MIN_EDIT_ADVANTAGE,
    )


def _fit_pair(
    allowed: frozenset[str], population: Population, options: Sequence[LaterOption],
    static: np.ndarray, insertion: np.ndarray, targets: np.ndarray,
    opening_cache: Mapping[frozenset[str], Any], local: Mapping[str, Any],
    old_scores: Mapping[frozenset[str], np.ndarray],
    old_references: Mapping[frozenset[str], Mapping[SectionIdentity, LaterOption]],
) -> tuple[frozenset[str], np.ndarray, dict[SectionIdentity, LaterOption], dict[str, Any]]:
    groups = _group_rows(options, population)
    train_indices = np.flatnonzero(np.isin(groups, tuple(allowed)))
    held_indices = np.flatnonzero(~np.isin(groups, tuple(allowed)))
    train_options = [options[index] for index in train_indices]
    held_options = [options[index] for index in held_indices]
    train_opening = opening_score_features(
        [option.proxy for option in train_options],
        training_opening_scores(population.actions, opening_cache, allowed),
    )
    held_opening = opening_score_features(
        [option.proxy for option in held_options], opening_cache[allowed],
    )
    local_groups = np.asarray(local["groups"])
    train_local_values = local_training_scores(local_groups, local["cache"], allowed)
    train_local = local_columns(tuple(options), train_local_values, 1)
    held_local = local_columns(tuple(options), local["cache"][allowed], 1)
    train_matrix = np.column_stack((_expanded_matrix(static[train_indices], insertion[train_indices], train_opening),
                                    train_local[train_indices]))
    held_matrix = np.column_stack((_expanded_matrix(static[held_indices], insertion[held_indices], held_opening),
                                   held_local[held_indices]))
    model, fit_seconds = fit_whole_model(train_matrix, targets[train_indices])
    scores = np.full(len(options), np.nan, dtype=np.float64)
    scores[held_indices] = _positive_scores(model, held_matrix)
    preceding = _preceding_reference(options, held_indices, old_scores[allowed], old_references[allowed])
    record = {
        "allowed_training_groups": sorted(allowed),
        "predicted_groups": sorted(set(GROUPS) - allowed),
        "fit_seconds": fit_seconds,
        "training_options": len(train_indices),
        "predicted_options": len(held_indices),
        "feature_count": int(train_matrix.shape[1]),
        "local_feature_count": 1,
        "preceding_reference": "old_pair_scores_applied_to_pair_references",
    }
    return allowed, scores, preceding, record


def _load_pair_output(path: Path, options: Sequence[LaterOption]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    saved = joblib.load(path)
    if saved.get("schema") != SCHEMA or saved.get("option_order") != _option_order(options):
        raise ValueError("chosen nested pair cache identity differs")
    return saved


def build_nested_local_scores(prepared: dict, output_root: Path, jobs: int = 4) -> dict:
    """Fit local pair choosers with group-isolated upstream features.

    Pair scores are full-length vectors.  Rows in an allowed training pair are
    ``NaN`` and rows in its two held groups are finite.  Pair references are
    selected by applying the old pair score vector to the old non-insert
    references before the new local chooser is used.
    """
    if jobs < 1:
        raise ValueError("jobs must be positive")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    static = np.asarray(prepared["static_features"], dtype=np.float64)
    insertion = np.asarray(prepared["insertion_features"], dtype=np.float64)
    targets = np.asarray(prepared["targets"], dtype=np.int8)
    if static.shape[0] != len(options) or insertion.shape[0] != len(options) or targets.shape != (len(options),):
        raise ValueError("prepared chosen-detector arrays do not share option coverage")
    option_order = _option_order(options)
    path = output_root / PAIR_OUTPUT_NAME
    saved = _load_pair_output(path, options)
    if saved is not None and saved.get("status") == "complete":
        return saved
    opening_cache, opening_records = _opening_cache(
        output_root, population, prepared["measurements"], prepared["local_targets"],
    )
    local, local_records = _local_pack(output_root, prepared, population)
    pair_scores = {} if saved is None else _normalise_cache(saved["pair_scores"])
    pair_references = {} if saved is None else _normalise_cache(saved["pair_references"])
    pair_records = {} if saved is None else {
        _set_key(record["allowed_training_groups"]): record for record in saved["pair_fit_records"]
    }
    pair_sets = [frozenset(pair) for pair in combinations(GROUPS, 2)]
    missing = [allowed for allowed in pair_sets
               if allowed not in pair_scores or allowed not in pair_references or allowed not in pair_records]
    if missing:
        old_scores, old_references = _load_old_pairs(options)
        fitted = joblib.Parallel(n_jobs=min(jobs, len(missing)), prefer="threads")(
            joblib.delayed(_fit_pair)(
                allowed,
                population,
                options,
                static,
                insertion,
                targets,
                {**prepared["opening_cache"], **opening_cache},
                local, old_scores, old_references,
            )
            for allowed in missing
        )
    else:
        fitted = []
    for key, scores, references, record in fitted:
        pair_scores[key] = scores
        pair_references[key] = references
        pair_records[key] = record
        joblib.dump({
            "schema": SCHEMA,
            "status": "partial",
            "option_order": option_order,
            "pair_scores": pair_scores,
            "pair_references": pair_references,
            "pair_fit_records": list(pair_records.values()),
            "singleton_opening_fit_records": opening_records,
            "singleton_local_fit_records": local_records,
        }, path, compress=3)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "option_order": option_order,
        "pair_scores": pair_scores,
        "pair_references": pair_references,
        "pair_fit_records": list(pair_records.values()),
        "singleton_opening": {"cache": opening_cache, "fit_records": opening_records},
        "local": local,
        "feature_names": {
            "static": tuple(prepared["static_feature_names"]),
            "opening": tuple(OPENING_FEATURE_NAMES),
            "insertion": tuple(prepared["insertion_feature_names"]),
            "local": ("local_inserted_score",),
        },
        "cache_paths": {
            "singleton_opening": (output_root / SINGLETON_OPENING_NAME).name,
            "singleton_local": (output_root / SINGLETON_LOCAL_NAME).name,
            "pair_scores": path.name,
        },
    }
    joblib.dump(payload, path, compress=3)
    return payload


def _group_scores(
    group: str, options: Sequence[LaterOption], scores: np.ndarray, population: Population,
) -> tuple[list[LaterOption], np.ndarray]:
    indices = np.asarray([index for index, option in enumerate(options)
                          if population.groups[option.span.fixture] == group], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"{group}: chosen options contain no rows")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape == (len(options),):
        values = values[indices]
    elif values.shape != (len(indices),):
        raise ValueError(f"{group}: score coverage differs from chosen options")
    return [options[index] for index in indices], values


def guarded_group_block(
    group: str, options: Sequence[LaterOption], scores: np.ndarray,
    reference: Mapping[SectionIdentity, LaterOption], population: Population,
    measurements: Any, candidates: Mapping[SectionIdentity, Sequence[Any]], labels: Any,
    local_model: Any,
) -> dict[str, Any]:
    """Build raw and boundary-guarded acceptance features for one group."""
    group_options, group_scores = _group_scores(group, options, scores, population)
    reference_subset = {
        identity: reference[identity]
        for identity in {option.base.identity for option in group_options}
    }
    raw_selected = select_with_reference(
        group_options, group_scores, reference_subset, minimum_advantage=MIN_EDIT_ADVANTAGE,
    )
    score_by_identity = {}
    for option, score in zip(group_options, group_scores, strict=True):
        if raw_selected[option.base.identity] == option:
            score_by_identity[option.base.identity] = float(score)

    group_spans = tuple(span for span in population.spans if population.groups[span.fixture] == group)
    group_events = {fixture: population.events[fixture] for fixture in {span.fixture for span in group_spans}}
    raw_stream = apply_options(group_spans, group_events, raw_selected)
    guarded_stream = pad_contact_boundaries(
        raw_stream.spans, raw_stream.events_by_fixture, population.fps, preserve_membership=True,
    )
    raw_membership = tuple((span.fixture, span.span_id, span.events) for span in raw_stream.spans)
    guarded_membership = tuple((span.fixture, span.span_id, span.events) for span in guarded_stream.spans)
    if raw_membership != guarded_membership:
        raise AssertionError("boundary guard changed contact membership")
    guarded_spans = {(span.fixture, span.span_id): span for span in guarded_stream.spans}
    guarded_selected = {
        identity: replace(option, span=guarded_spans[identity])
        for identity, option in raw_selected.items()
    }

    base, base_names, identities = acceptance_features(
        group_options, group_scores, raw_selected, candidates, population.fps, measurements,
    )
    base = np.asarray(base, dtype=np.float64)
    refresh_boundary_features(base, base_names, identities, guarded_selected, population.fps)
    judged_rows = _section_judgements(population, guarded_selected, score_by_identity, labels)
    rows_by_identity = {(str(row["fixture"]), int(row["span_id"])): row for row in judged_rows}
    if set(rows_by_identity) != set(identities):
        raise ValueError(f"{group}: guarded judgement coverage differs")
    rows = [rows_by_identity[identity] for identity in identities]
    gap, gap_names, gap_identities = gap_evidence(
        guarded_selected, candidates, population.fps, measurements, local_model,
    )
    if tuple(gap_identities) != tuple(identities):
        gap_by_identity = {identity: index for index, identity in enumerate(gap_identities)}
        gap = np.asarray([gap[gap_by_identity[identity]] for identity in identities], dtype=np.float64)
    combined = np.column_stack((base, gap))
    return {
        "base_features": base,
        "base_feature_names": tuple(base_names),
        "gap_features": combined,
        "gap_feature_names": (*base_names, *gap_names),
        "gap_evidence_features": gap,
        "gap_evidence_feature_names": tuple(gap_names),
        "rows": rows,
        "raw_selected": raw_selected,
        "guarded_selected": guarded_selected,
        "stream": guarded_stream,
    }


def refresh_boundary_features(
    base: np.ndarray, names: Sequence[str], identities: Sequence[SectionIdentity],
    guarded: Mapping[SectionIdentity, LaterOption], fps: Mapping[str, float],
) -> None:
    """Refresh output summaries while preserving the saved chooser score columns."""
    columns = [names.index(f"output__{name}") for name in FEATURE_NAMES]
    for row_index, identity in enumerate(identities):
        base[row_index, columns] = build_feature_vector(guarded[identity].span, fps[identity[0]])
