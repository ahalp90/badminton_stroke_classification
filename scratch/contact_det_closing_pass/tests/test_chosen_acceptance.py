"""Focused checks for nested local scores and guarded acceptance features."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts import chosen_acceptance
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    Population,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction


def _population() -> tuple[Population, tuple[LaterOption, ...]]:
    spans = []
    events = {}
    options = []
    groups = {}
    for index, group in enumerate(chosen_acceptance.GROUPS):
        fixture = f"fixture_{group}"
        event = FixedEvent(fixture, 20, 0.8, "Top")
        span = FixedSpan(fixture, 0, 10, 50, (event,))
        inserted = FixedEvent(fixture, 30, 0.7, "Bot")
        spans.append(span)
        events[fixture] = (event,)
        groups[fixture] = group
        base = CombinedAction("keep", None, None, span)
        options.extend((LaterOption(base, None, span), LaterOption(
            base, inserted, replace(span, events=(event, inserted)),
        )))
    population = Population((), tuple(spans), events, (), (), {fixture: 30.0 for fixture in groups}, groups)
    return population, tuple(options)


def test_nested_pair_scores_use_old_scores_to_restore_inserted_preceding_reference(
    tmp_path: Path, monkeypatch,
) -> None:
    population, options = _population()
    option_count = len(options)
    old_scores = {}
    old_references = {}
    for first_index, first in enumerate(chosen_acceptance.GROUPS):
        for second in chosen_acceptance.GROUPS[first_index + 1:]:
            allowed = frozenset((first, second))
            values = np.full(option_count, np.nan)
            references = {}
            for index, option in enumerate(options):
                group = population.groups[option.span.fixture]
                if group not in allowed:
                    values[index] = 1.0 if option.inserted is None else 1.2
                    references.setdefault(option.base.identity, next(
                        candidate for candidate in options[index - index % 2:index + 1]
                        if candidate.inserted is None
                    ))
            old_scores["".join(sorted(allowed))] = values
            old_references["".join(sorted(allowed))] = references
    monkeypatch.setattr(chosen_acceptance, "PAIR_SOURCE", tmp_path / "old_pairs.joblib")
    monkeypatch.setattr(chosen_acceptance, "LOCAL_SOURCE", tmp_path / "old_local.joblib")
    joblib.dump({"pair_scores": old_scores, "pair_references": old_references}, chosen_acceptance.PAIR_SOURCE)

    contexts = len(chosen_acceptance.GROUPS)
    source_cache = {}
    source_models = {}
    for size in (2, 3):
        for selected in combinations(chosen_acceptance.GROUPS, size):
            key = frozenset(selected)
            source_cache[key] = np.full(contexts, 0.4)
            source_models[key] = object()
    joblib.dump({
        "cache": source_cache,
        "models": source_models,
        "groups": np.asarray(chosen_acceptance.GROUPS),
        "feature_names": ("i0",),
    }, chosen_acceptance.LOCAL_SOURCE)

    monkeypatch.setattr(chosen_acceptance, "load_rally_start_model_config", lambda _path: SimpleNamespace(
        models=(SimpleNamespace(model_id="shallow_hgb"),),
    ))
    monkeypatch.setattr(chosen_acceptance, "_build_single_group_cache", lambda *_args: (
        {frozenset({group}): {} for group in chosen_acceptance.GROUPS}, [],
    ))
    monkeypatch.setattr(chosen_acceptance, "load_human_labels", lambda *_args: object())
    monkeypatch.setattr(chosen_acceptance, "insertion_targets", lambda *_args: np.array([0, 1, 0, 1], dtype=np.int8))

    def fake_local_cache(matrix, targets, groups, training_sets):
        del matrix, targets
        cache = {}
        models = {}
        records = []
        for key in training_sets:
            cache[key] = np.where(groups == next(iter(key)), np.nan, 0.6)
            models[key] = object()
            records.append({"training_groups": sorted(key)})
        return cache, models, records

    monkeypatch.setattr(chosen_acceptance, "build_local_cache", fake_local_cache)
    def training_scores(_actions, cache, allowed):
        for group in allowed:
            assert allowed - {group} in cache
        return {}

    monkeypatch.setattr(chosen_acceptance, "training_opening_scores", training_scores)
    monkeypatch.setattr(chosen_acceptance, "opening_score_features", lambda options, _scores: np.ones((len(options), 2)))
    fit_calls = []

    def fake_fit(matrix, answers):
        fit_calls.append((matrix.copy(), answers.copy()))
        return object(), 0.01

    monkeypatch.setattr(chosen_acceptance, "fit_whole_model", fake_fit)
    monkeypatch.setattr(chosen_acceptance, "_positive_scores", lambda _model, matrix: np.full(len(matrix), 0.8))
    prepared = {
        "base_population": population,
        "options": options,
        "static_features": np.ones((option_count, 2)),
        "insertion_features": np.ones((option_count, 1)),
        "targets": np.asarray([index % 2 for index in range(option_count)], dtype=np.int8),
        "opening_cache": {frozenset(pair): {} for pair in combinations(chosen_acceptance.GROUPS, 2)},
        "local_targets": {},
        "measurements": PhysicalMeasurements((), {}, {}),
        "static_feature_names": ("s0", "s1"),
        "insertion_feature_names": ("i0",),
    }
    result = chosen_acceptance.build_nested_local_scores(prepared, tmp_path / "output", jobs=1)
    assert len(fit_calls) == 6
    assert set(result["pair_scores"]) == {
        frozenset(pair) for pair in combinations(chosen_acceptance.GROUPS, 2)
    }
    for key, values in result["pair_scores"].items():
        assert values.shape == (option_count,)
        held = [index for index, option in enumerate(options)
                 if population.groups[option.span.fixture] not in key]
        assert np.isfinite(values[held]).all()
        assert np.isnan(values[[index for index in range(option_count) if index not in held]]).all()
    preceding = result["pair_references"][frozenset(("A", "B"))]
    assert all(option.inserted is not None for option in preceding.values())


def test_select_with_reference_keeps_current_inserted_reference_under_small_gain() -> None:
    span = FixedSpan("fixture_A", 0, 10, 50, (FixedEvent("fixture_A", 20, 0.8, "Top"),))
    inserted = FixedEvent("fixture_A", 30, 0.7, "Bot")
    base = CombinedAction("keep", None, None, span)
    raw = LaterOption(base, None, span)
    current = LaterOption(base, inserted, replace(span, events=(*span.events, inserted)))
    selected = select_with_reference((raw, current), np.asarray((0.14, 0.12)), {base.identity: current})
    assert selected[base.identity] == current


def test_guarded_edges_change_features_and_judgement_without_changing_contacts(monkeypatch) -> None:
    event = FixedEvent("fixture_A", 20, 0.8, "Top")
    span = FixedSpan("fixture_A", 0, 15, 25, (event,))
    base = CombinedAction("keep", None, None, span)
    option = LaterOption(base, None, span)
    population = Population((), (span,), {"fixture_A": (event,)}, (), (), {"fixture_A": 30.0}, {"fixture_A": "A"})
    measurements = PhysicalMeasurements((), {}, {})

    def fake_judgements(_population, selected, _scores, _labels):
        chosen = next(iter(selected.values())).span
        outcome = "correct" if chosen.start_frame == 10 else "wrong"
        return [{"fixture": "fixture_A", "span_id": 0, "judgements": {
            "10": {"outcome": outcome}, "5": {"outcome": outcome},
        }}]

    monkeypatch.setattr(chosen_acceptance, "_section_judgements", fake_judgements)
    block = chosen_acceptance.guarded_group_block(
        "A", (option,), np.asarray((0.8,)), {option.base.identity: option}, population,
        measurements, {option.base.identity: ()}, object(), object(),
    )
    raw_span = block["raw_selected"][option.base.identity].span
    guarded_span = block["guarded_selected"][option.base.identity].span
    assert (raw_span.start_frame, raw_span.end_frame) == (15, 25)
    assert (guarded_span.start_frame, guarded_span.end_frame) == (10, 31)
    assert guarded_span.events == raw_span.events == block["stream"].events_by_fixture["fixture_A"]
    duration_column = block["base_feature_names"].index("output__section_duration_seconds")
    assert block["base_features"][0, duration_column] == 21 / 30
    assert block["base_features"][0, block["base_feature_names"].index("selected_score")] == 0.8
    assert block["rows"][0]["judgements"]["10"]["outcome"] == "correct"
    assert block["gap_features"].shape[1] == len(block["gap_feature_names"])
