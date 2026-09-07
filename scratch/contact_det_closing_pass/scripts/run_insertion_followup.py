"""Test local insertion evidence and compatible pairs in the complete chooser."""

import argparse
import lzma
from itertools import combinations
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import (
    contextual_insertions,
    pair_features,
    restore_choices,
)
from scratch.contact_det_closing_pass.scripts.insertion_learning import (
    build_local_cache,
    local_training_scores,
)
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import (
    apply_options,
    insertion_features,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.local_insertion import insertion_targets
from scratch.contact_det_closing_pass.scripts.pair_targets import pair_targets
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _expanded_matrix,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    build_whole_features,
    opening_score_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    WHOLE_MODEL_SETTINGS,
    fit_whole_model,
    training_opening_scores,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

RAW = ROOT / "raw/followups"
RESULTS = ROOT / "results/followups"


def local_columns(options: tuple, values: np.ndarray, width: int) -> np.ndarray:
    output = np.full((len(options), width), np.nan)
    offset = 0
    for index, option in enumerate(options):
        count = len(option.inserted_events)
        if count:
            output[index, :count] = values[offset:offset + count]
            offset += count
    if offset != len(values):
        raise ValueError("Local insertion scores do not align with chooser options")
    return output


def prepare(variant: str) -> dict[str, Any]:
    started = perf_counter()
    prepared = joblib.load(ROOT / "raw/later_run/prepared.joblib")
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    static = prepared["static_features"]
    insertion = prepared["insertion_features"]
    targets = prepared["targets"]
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    use_pairs = variant in {"pairs", "both"}
    if use_pairs:
        pair_cache = RAW / "pair_features.joblib"
        if pair_cache.exists():
            pair_pack = joblib.load(pair_cache)
        else:
            pairs = joblib.load(RAW / "pair_options.joblib")
            pair_static, names, columns = build_whole_features(
                tuple(option.proxy for option in pairs), population.spans, population.actions,
                population.fps, prepared["measurements"],
            )
            if names != prepared["static_feature_names"] or columns != prepared["columns"]:
                raise ValueError("Pair static features differ from the reference")
            pair_insertion, _ = insertion_features(pairs, population.fps, prepared["measurements"])
            answers = pair_targets(pairs, options, targets, labels, population.fps)
            report = {str(target): int(np.sum(answers == target)) for target in (-1, 0, 1)}
            pair_pack = {"options": pairs, "static": pair_static, "insertion": pair_insertion,
                         "targets": answers, "target_report": report}
            joblib.dump(pair_pack, pair_cache, compress=3)
        options += tuple(pair_pack["options"])
        static = np.concatenate((static, pair_pack["static"]))
        insertion = np.concatenate((insertion, pair_pack["insertion"]))
        targets = np.concatenate((targets, pair_pack["targets"]))
        second, _ = pair_features(options, population.fps, prepared["measurements"])
        insertion = np.column_stack((insertion, second))
    data = {"population": population, "options": options, "static": static, "insertion": insertion,
            "targets": targets, "opening_cache": prepared["opening_cache"],
            "static_feature_names": prepared["static_feature_names"], "columns": prepared["columns"],
            "insertion_feature_names": prepared["insertion_feature_names"], "variant": variant}
    if variant in {"local", "both"}:
        cache_path = RAW / f"{variant}_local_cache.joblib"
        if cache_path.exists():
            local = joblib.load(cache_path)
        else:
            if variant == "local":
                included = np.asarray([option.inserted is not None for option in options])
                contexts = tuple(option for option in options if option.inserted is not None)
                matrix, names = insertion[included], prepared["insertion_feature_names"]
            else:
                contexts = contextual_insertions(options)
                matrix, names = insertion_features(contexts, population.fps, prepared["measurements"])
            local_targets = insertion_targets(contexts, population.spans, labels, population.fps)
            groups = np.asarray([population.groups[option.span.fixture] for option in contexts])
            training_sets = [frozenset(names) for size in (2, 3) for names in combinations(GROUPS, size)]
            cache, models, records = build_local_cache(matrix, local_targets, groups, training_sets)
            final, seconds = fit_whole_model(matrix, local_targets)
            local = {"cache": cache, "models": models, "groups": groups, "final": final, "feature_names": names,
                     "records": records, "final_fit_seconds": seconds,
                     "target_counts": {str(target): int(np.sum(local_targets == target)) for target in (-1, 0, 1)}}
            joblib.dump(local, cache_path, compress=3)
        data["local"] = local
        print("Local targets", local["target_counts"], flush=True)
    data["prepare_seconds"] = perf_counter() - started
    print(f"Prepared {variant}: {len(options)} alternatives in {data['prepare_seconds']:.1f}s", flush=True)
    return data


def fit_fold(data: dict, held_out: str | None) -> tuple:
    population, options = data["population"], data["options"]
    groups = np.asarray([population.groups[option.span.fixture] for option in options])
    allowed = frozenset(GROUPS) if held_out is None else frozenset(GROUPS) - {held_out}
    train = np.flatnonzero(np.isin(groups, tuple(allowed)))
    held = np.flatnonzero(groups == held_out)
    opening_train = opening_score_features(
        [options[index].proxy for index in train],
        training_opening_scores(population.actions, data["opening_cache"], allowed),
    )
    train_matrix = _expanded_matrix(data["static"][train], data["insertion"][train], opening_train)
    if held_out is not None:
        opening_held = opening_score_features([options[index].proxy for index in held], data["opening_cache"][allowed])
        held_matrix = _expanded_matrix(data["static"][held], data["insertion"][held], opening_held)
    if "local" in data:
        local = data["local"]
        width = 2 if data["variant"] == "both" else 1
        train_local = local_columns(options, local_training_scores(local["groups"], local["cache"], allowed), width)
        train_matrix = np.column_stack((train_matrix, train_local[train]))
        if held_out is not None:
            held_local = local_columns(options, local["cache"][allowed], width)
            held_matrix = np.column_stack((held_matrix, held_local[held]))
    model, seconds = fit_whole_model(train_matrix, data["targets"][train])
    scores = None if held_out is None else _positive_scores(model, held_matrix)
    record = {"held_out": held_out, "training_groups": sorted(allowed), "fit_seconds": seconds,
              "training_options": len(train), "feature_count": train_matrix.shape[1]}
    print("Fit", data["variant"], record, flush=True)
    return held, scores, model, record


def run(variant: str, jobs: int) -> None:
    started = perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    data = prepare(variant)
    population, options = data["population"], data["options"]
    reference_payload = prediction_io.read_json(ROOT / "results/later/later_margin_predictions.json.gz")
    reference = restore_choices(options, reference_payload["selected_actions"])
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=6):
        fitted = joblib.Parallel()(joblib.delayed(fit_fold)(data, group) for group in GROUPS)
    scores = np.full(len(options), np.nan)
    for indices, values, _model, _record in fitted:
        scores[indices] = values
    selected = select_with_reference(options, scores, reference)
    stream = apply_options(population.spans, population.events, selected)
    scores_path = RAW / f"{variant}_option_scores.npy.xz"
    with lzma.open(scores_path, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
        np.save(handle, scores)
    write_json(RESULTS / f"{variant}_predictions.json.gz", {
        "schema": "contact-followup-predictions/1", "status": "complete", "variant": variant,
        "prediction_selection_uses_labels": False, "upstream_detector_scores_retain_cross_group_dependence": True,
        "option_scores_file": scores_path.name,
        "selected_actions": [option_record(option) for option in selected.values()], "outputs": stream_records(stream),
    })
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    comparison = compare_outputs(tuple(option.span for option in reference.values()), selected,
                                 labels, population.fps, population.groups)
    _indices, _values, final, final_record = fit_fold(data, None)
    frozen = joblib.load(ROOT / "raw/later_run/models.joblib")
    models = {**frozen, "whole": final, "variant": variant,
              "local": data.get("local", {}).get("final"),
              "local_feature_names": data.get("local", {}).get("feature_names")}
    joblib.dump(models, RAW / f"{variant}_models.joblib", compress=3)
    output = {"schema": "contact-followup-comparison/1", "status": "complete", "variant": variant,
              "comparison_to_session_start": comparison, "model_settings": WHOLE_MODEL_SETTINGS,
              "minimum_edit_advantage": .05, "target_tolerance_base30": 10,
              "prepare_seconds": data["prepare_seconds"], "total_seconds": perf_counter() - started,
              "fits": [result[3] for result in fitted] + [final_record],
              "local_fits": data.get("local", {}).get("records", []),
              "local_target_counts": data.get("local", {}).get("target_counts", {})}
    write_json(RESULTS / f"{variant}_result.json.gz", output)
    for tolerance, metrics in comparison.items():
        paired = metrics["paired"]
        print(variant, tolerance, "correct", paired["correct_before"], paired["correct_after"],
              "repairs", len(paired["repaired"]), "losses", len(paired["lost"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("local", "pairs", "both"), required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.variant, args.jobs)
