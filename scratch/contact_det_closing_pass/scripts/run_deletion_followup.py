"""Test one local deletion score inside the current insertion and opening chooser."""

from __future__ import annotations

import argparse
import lzma
from dataclasses import replace
from itertools import combinations
from time import perf_counter

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.deletion_evidence import (
    deletion_column,
    deletion_inputs,
    deletion_targets,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.insertion_learning import (
    build_local_cache,
    local_training_scores,
)
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import (
    apply_options,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_insertion_followup import (
    local_columns,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    _expanded_matrix,
)
from scratch.contact_det_closing_pass.scripts.run_serve_followups import (
    OUTPUT,
    ROOT,
    development_streams,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.serve_metrics import (
    analyse_serves,
    compare_serves,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    voted_contact_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    fit_whole_model,
    training_opening_scores,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

RAW = ROOT / "raw/serve_followups"


def prepare() -> dict:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    path = RAW / "deletion_inputs.joblib"
    if path.exists():
        deletion = joblib.load(path)
    else:
        deletion = deletion_inputs(options, population.events, population.fps, prepared["measurements"])
        print("Deletion physical features", len(deletion["contexts"]), "contexts ready", flush=True)
        labels = load_human_labels(start.LABEL_PATH, population.videos)
        deletion["targets"] = deletion_targets(deletion["contexts"], population.spans, labels, population.fps)
        deletion["groups"] = np.asarray([population.groups[context.span.fixture] for context in deletion["contexts"]])
        joblib.dump(deletion, path, compress=3)
    counts = {str(value): int(np.sum(deletion["targets"] == value)) for value in (-1, 0, 1)}
    print("Local deletion targets", counts, flush=True)
    cache_path = RAW / "deletion_local_cache.joblib"
    if cache_path.exists():
        cache = joblib.load(cache_path)
    else:
        sets = [frozenset(values) for size in (2, 3) for values in combinations(GROUPS, size)]
        scores, models, records = build_local_cache(deletion["features"], deletion["targets"], deletion["groups"], sets)
        cache = {"cache": scores, "models": models, "records": records}
        joblib.dump(cache, cache_path, compress=3)
    return {
        "prepared": prepared, "population": population, "options": options,
        "insertion_local": joblib.load(ROOT / "raw/followups/local_local_cache.joblib"),
        "deletion": deletion, "deletion_cache": cache, "target_counts": counts,
        "prepare_seconds": perf_counter() - started,
    }


def matrix_for(data: dict, indices: np.ndarray, allowed: frozenset[str], training: bool) -> np.ndarray:
    prepared, options = data["prepared"], data["options"]
    local, deletion = data["insertion_local"], data["deletion"]
    if training:
        opening = training_opening_scores(data["population"].actions, prepared["opening_cache"], allowed)
        inserted_scores = local_training_scores(local["groups"], local["cache"], allowed)
        removed_scores = local_training_scores(deletion["groups"], data["deletion_cache"]["cache"], allowed)
    else:
        opening = prepared["opening_cache"][allowed]
        inserted_scores = local["cache"][allowed]
        removed_scores = data["deletion_cache"]["cache"][allowed]
    opening_features = opening_score_features([options[index].proxy for index in indices], opening)
    base = _expanded_matrix(prepared["static_features"][indices], prepared["insertion_features"][indices], opening_features)
    insertion_column = local_columns(options, inserted_scores, 1)[indices]
    removal_column = deletion_column(deletion["option_context_indices"], removed_scores)[indices]
    return np.column_stack((base, insertion_column, removal_column))


def fit_fold(data: dict, held: str | None) -> tuple:
    groups = np.asarray([data["population"].groups[option.span.fixture] for option in data["options"]])
    allowed = frozenset(GROUPS) if held is None else frozenset(GROUPS) - {held}
    train = np.flatnonzero(np.isin(groups, tuple(allowed)))
    held_indices = np.flatnonzero(groups == held)
    model, seconds = fit_whole_model(matrix_for(data, train, allowed, True), data["prepared"]["targets"][train])
    scores = None if held is None else _positive_scores(model, matrix_for(data, held_indices, allowed, False))
    record = {"held_group": held, "training_groups": sorted(allowed), "fit_seconds": seconds,
              "training_options": len(train), "feature_count": int(model.n_features_in_)}
    print("Deletion chooser", record, flush=True)
    return held_indices, scores, model, record


def run(jobs: int, freeze: bool = False) -> None:
    started = perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    data = prepare()
    if freeze:
        _indices, _scores, whole, record = fit_fold(data, None)
        local = data["deletion"]
        deletion_model, seconds = fit_whole_model(local["features"], local["targets"])
        frozen = joblib.load(ROOT / "raw/followups/local_models.joblib")
        frozen.update({"whole": whole, "deletion": deletion_model,
                       "deletion_feature_names": local["feature_names"], "variant": "local_deletion"})
        joblib.dump(frozen, RAW / "deletion_models.joblib", compress=3)
        write_json(OUTPUT / "deletion_freeze.json.gz", {"status": "complete", "fit": record, "local_fit_seconds": seconds})
        return
    options, population = data["options"], data["population"]
    current = prediction_io.read_json(RAW / "reference_development_predictions/local_predictions.json.gz")
    reference = restore_choices(options, current["selected_actions"])
    if stream_records(apply_options(population.spans, population.events, reference)) != current["outputs"]:
        raise ValueError("Deletion reference failed to restore the exact current stream")
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=8):
        fits = joblib.Parallel()(joblib.delayed(fit_fold)(data, group) for group in GROUPS)
    scores = np.full(len(options), np.nan)
    for indices, values, _model, _record in fits:
        scores[indices] = values
    selected = select_with_reference(options, scores, reference)
    raw_stream = apply_options(population.spans, population.events, selected)
    stream = pad_contact_boundaries(raw_stream.spans, raw_stream.events_by_fixture, population.fps, preserve_membership=True)
    by_identity = {(span.fixture, span.span_id): span for span in stream.spans}
    guarded = {identity: replace(option, span=by_identity[identity]) for identity, option in selected.items()}
    with lzma.open(RAW / "deletion_option_scores.npy.xz", "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
        np.save(handle, scores)
    write_json(OUTPUT / "deletion_predictions.json.gz", {
        "status": "complete", "selection_uses_labels": False, "outputs": stream_records(stream),
        "raw_outputs": stream_records(raw_stream), "selected_actions": [option_record(option) for option in selected.values()],
        "reference": "local insertion plus guarded edges", "minimum_advantage": 0.05,
    })
    streams, labels, fps, groups = development_streams()
    comparisons = {name: compare_outputs(streams[name].spans, guarded, labels, fps, groups)
                   for name in ("recommended", "guarded_only")}
    old_serves = prediction_io.read_json(OUTPUT / "development_serves.json.gz")["variants"]["recommended"]
    serves, serve_changes, contacts = {}, {}, {}
    voted = start.apply_whole_rally_alternation(stream)
    for tolerance in (10, 5):
        key = str(tolerance)
        serves[key] = analyse_serves(stream, labels, fps, tolerance)
        raw = {"total": serves[key]["contact_totals"], "by_video": serves[key]["contact_by_video"]}
        contacts[key] = voted_contact_scores(raw, voted.events_by_fixture)
        serve_changes[key] = compare_serves(old_serves[key], serves[key])
    write_json(OUTPUT / "deletion_development.json.gz", {
        "schema": "contact-local-deletion/1", "status": "complete", "comparisons": comparisons,
        "serves": serves, "serve_changes": serve_changes, "voted_contacts": contacts,
        "local_target_counts": data["target_counts"], "local_fits": data["deletion_cache"]["records"],
        "chooser_fits": [fit[3] for fit in fits], "prepare_seconds": data["prepare_seconds"],
        "seconds": perf_counter() - started,
        "new_local_and_whole_fits_exclude_scored_group": True,
        "old_cached_detector_scores_retain_cross_group_dependence": True,
    })
    for tolerance, comparison in comparisons["recommended"].items():
        print("Deletion development", tolerance, comparison["paired"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.freeze)
