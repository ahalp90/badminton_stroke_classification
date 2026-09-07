"""Evaluate a label-free expansion of the saved early candidate shortlist."""

from __future__ import annotations

import argparse
import lzma
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from scratch.contact_det_closing_pass.scripts import run_insertion_followup
from scratch.contact_det_closing_pass.scripts.early_shortlist import (
    MAX_EARLY_CANDIDATES,
    expand_early_shortlist,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.later_options import (
    MIN_EDIT_ADVANTAGE,
    apply_options,
    build_later_options,
    insertion_features,
    option_record,
    select_with_reference,
)
from scratch.contact_det_closing_pass.scripts.prepare_later_inputs import (
    SCORE_PATH,
    _fixture_score_rows,
    _load_score_rows,
    _side_replay,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    DEFAULT_FEATURE_ROOT as LATER_FEATURE_ROOT,
)
from scratch.contact_det_closing_pass.scripts.run_later_comparison import (
    ROOT,
    _merge_measurements,
)
from scratch.contact_det_closing_pass.scripts.run_whole_rally_comparison import (
    prepare_population,
)
from scratch.contact_det_closing_pass.scripts.targets import assign_targets
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    action_matrix,
    build_whole_features,
    load_measurements,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_learning import (
    GROUPS,
    build_opening_cache,
    fit_opening_models,
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

RAW = ROOT / "raw/followups"
RESULTS = ROOT / "results/followups"
OLD_PREPARED = ROOT / "raw/later_run/prepared.joblib"
OLD_MODELS = ROOT / "raw/later_run/models.joblib"
LOCAL_CACHE = RAW / "local_local_cache.joblib"
LOCAL_PREDICTIONS = RESULTS / "local_predictions.json.gz"
SESSION_START_PREDICTIONS = ROOT / "results/later/later_margin_predictions.json.gz"


def _expand_inputs(
    raw_videos: Sequence[Mapping[str, Any]],
    video_specs: Mapping[str, Any],
    score_rows: np.ndarray,
    side_root: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    expanded: list[dict[str, Any]] = []
    totals = {
        "candidate_lists": 0,
        "candidate_entries_before": 0,
        "candidate_entries_after": 0,
        "added_earlier_candidates": 0,
    }
    for raw_video in raw_videos:
        video_info = raw_video["video"]
        fixture = str(video_info["fixture"])
        group = str(raw_video["group"])
        video_spec = video_specs[fixture]
        rows = _fixture_score_rows(score_rows, fixture, group, float(video_spec.fps))
        original_frames = {
            int(candidate["frame"])
            for candidate_list in raw_video["candidate_lists"]
            for candidate in candidate_list["candidates"]
        }
        placeholder_sides = dict.fromkeys(int(frame) for frame in rows["frame"])
        expanded_video, counts = expand_early_shortlist(
            raw_video, rows, placeholder_sides
        )
        added_frames = {
            int(candidate["frame"])
            for candidate_list in expanded_video["candidate_lists"]
            for candidate in candidate_list["candidates"]
            if int(candidate["frame"]) not in original_frames
        }
        if added_frames:
            side_by_frame, _side_seconds = _side_replay(
                fixture, video_spec, side_root, sorted(added_frames)
            )
        else:
            side_by_frame = {}
        for candidate_list in expanded_video["candidate_lists"]:
            for candidate in candidate_list["candidates"]:
                frame = int(candidate["frame"])
                if frame in added_frames:
                    candidate["predicted_side"] = side_by_frame[frame]
        expanded.append(expanded_video)
        totals["candidate_lists"] += int(counts["candidate_lists"])
        totals["candidate_entries_before"] += len(original_frames)
        totals["candidate_entries_after"] += sum(
            len(candidate_list["candidates"])
            for candidate_list in expanded_video["candidate_lists"]
        )
        totals["added_earlier_candidates"] += int(counts["added_earlier_candidates"])
    totals["videos"] = len(expanded)
    return tuple(expanded), totals


def _reuse_local_cache(
    options: Sequence[Any],
    insertion: np.ndarray,
    groups_by_fixture: Mapping[str, str],
    local_pack: Mapping[str, Any],
) -> dict[str, Any]:
    contexts = [option for option in options if option.inserted_events]
    context_matrix = insertion[[index for index, option in enumerate(options) if option.inserted_events]]
    context_groups = np.asarray(
        [groups_by_fixture[option.span.fixture] for option in contexts], dtype=str
    )
    models = {frozenset(key): model for key, model in local_pack["models"].items()}
    if context_matrix.shape[0] != context_groups.shape[0]:
        raise ValueError("new local feature rows and groups are misaligned")
    cache: dict[frozenset[str], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for allowed, model in models.items():
        outside = np.asarray([group not in allowed for group in context_groups], dtype=bool)
        values = np.full(context_matrix.shape[0], np.nan, dtype=float)
        if np.any(outside):
            values[outside] = run_insertion_followup._positive_scores(model, context_matrix[outside])
        cache[allowed] = values
        records.append({
            "training_groups": sorted(allowed),
            "predicted_rows": int(np.count_nonzero(outside)),
            "reused_model": True,
        })
    feature_names = tuple(local_pack["feature_names"])
    return {
        "cache": cache,
        "models": models,
        "groups": context_groups,
        "final": local_pack["final"],
        "feature_names": feature_names,
        "records": records,
        "final_fit_seconds": float(local_pack.get("final_fit_seconds", 0.0)),
        "target_counts": local_pack.get("target_counts", {}),
    }


def _save_scores(path: Path, scores: np.ndarray) -> None:
    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=6) as handle:
        np.save(handle, scores)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = prediction_io.read_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected an object in {path}")
    return payload


def run(
    jobs: int = 2,
    *,
    side_root: Path,
    feature_root: Path = LATER_FEATURE_ROOT,
    score_path: Path = SCORE_PATH,
) -> Mapping[str, Any]:
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    total_started = perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    old_prepared = joblib.load(OLD_PREPARED)
    raw_videos = start._candidate_videos()
    pack = prediction_io.load_development_predictions()
    video_specs = {str(video.fixture): video for video in pack.videos}
    score_rows = _load_score_rows(score_path)
    expanded_videos, expansion_counts = _expand_inputs(raw_videos, video_specs, score_rows, side_root)
    write_json(RAW / "early_inputs.json.gz", {
        "schema": "contact-closing-early-inputs/1",
        "status": "complete",
        "labels_read": False,
        "max_early_candidates": MAX_EARLY_CANDIDATES,
        "score_file": score_path.name,
        "videos": expanded_videos,
        "counts": expansion_counts,
    })
    print(f"Expanded early shortlist: {expansion_counts}", flush=True)

    population = prepare_population(
        pack, frozenset(GROUPS), expanded_videos, max_earlier_candidates=MAX_EARLY_CANDIDATES,
    )
    options = build_later_options(
        population.options,
        old_prepared["later_candidates"],
        population.fps,
        max_insertions=1,
    )
    measurements = load_measurements(population.actions, population.events, feature_root)
    if measurements.audit["missing_identity_count"]:
        raise ValueError("new early actions are missing frozen physical measurements")
    measurements = _merge_measurements(
        measurements,
        old_prepared["later_physical"],
        tuple(old_prepared["measurements"].names),
    )
    static, static_names, columns = build_whole_features(
        tuple(option.proxy for option in options),
        population.spans,
        population.actions,
        population.fps,
        measurements,
    )
    insertion, insertion_names = insertion_features(options, population.fps, measurements)
    training_labels = load_human_labels(start.LABEL_PATH, population.videos)
    local_targets, _local_revised_spans = assign_targets(
        population.actions,
        population.spans,
        population.events,
        training_labels,
        population.fps,
    )
    targets, target_report = whole_targets(
        tuple(option.proxy for option in options),
        population.spans,
        training_labels,
        population.fps,
    )
    parsed_config = load_rally_start_model_config(start.CONFIG_PATH)
    opening_spec = next(model for model in parsed_config.models if model.model_id == "shallow_hgb")
    action_values = action_matrix(population.actions, measurements)
    opening_cache, opening_records = build_opening_cache(
        population.actions,
        action_values,
        local_targets,
        opening_spec,
    )
    local_pack = joblib.load(LOCAL_CACHE)
    local = _reuse_local_cache(options, insertion, population.groups, local_pack)
    if tuple(insertion_names) != tuple(local["feature_names"]):
        raise ValueError("new insertion feature names do not match saved local models")
    data = {
        "population": population,
        "options": options,
        "static": static,
        "static_feature_names": static_names,
        "columns": columns,
        "insertion": insertion,
        "insertion_feature_names": insertion_names,
        "targets": targets,
        "opening_cache": opening_cache,
        "variant": "early",
        "local": local,
    }
    prepare_seconds = perf_counter() - total_started
    print(
        f"Prepared early: {len(options)} alternatives; "
        f"{len(local['groups'])} local contexts",
        flush=True,
    )

    fit_started = perf_counter()
    with joblib.parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=6):
        fold_results = joblib.Parallel()(
            joblib.delayed(run_insertion_followup.fit_fold)(data, group)
            for group in GROUPS
        )
    _final_indices, _, final_whole, final_record = run_insertion_followup.fit_fold(data, None)
    final_opening, final_opening_seconds = fit_opening_models(
        population.actions,
        action_values,
        local_targets,
        frozenset(GROUPS),
        opening_spec,
    )
    fit_seconds = perf_counter() - fit_started
    scores = np.full(len(options), np.nan, dtype=float)
    for held_indices, held_scores, _, _ in fold_results:
        scores[held_indices] = held_scores
    if not np.all(np.isfinite(scores)):
        raise ValueError("OOF whole-model scores are incomplete")

    local_reference = restore_choices(
        options, _load_json(LOCAL_PREDICTIONS)["selected_actions"]
    )
    session_start_reference = restore_choices(
        options, _load_json(SESSION_START_PREDICTIONS)["selected_actions"]
    )
    selected = select_with_reference(
        options,
        scores,
        local_reference,
        minimum_advantage=MIN_EDIT_ADVANTAGE,
    )
    outputs = apply_options(population.spans, population.events, selected)
    score_path_out = RAW / "early_option_scores.npy.xz"
    _save_scores(score_path_out, scores)
    write_json(RESULTS / "early_predictions.json.gz", {
        "schema": "contact-closing-followup-predictions/1",
        "status": "complete",
        "variant": "early",
        "labels_read": False,
        "prediction_selection_uses_labels": False,
        "upstream_detector_scores_retain_cross_group_dependence": True,
        "option_scores_file": score_path_out.name,
        "selected_actions": [option_record(option) for option in selected.values()],
        "outputs": run_insertion_followup.stream_records(outputs),
    })
    frozen_models = joblib.load(OLD_MODELS)
    early_models = {
        **frozen_models,
        "whole": final_whole,
        "opening": final_opening,
        "local": local_pack["final"],
        "local_feature_names": tuple(local_pack["feature_names"]),
        "variant": "early",
    }
    joblib.dump(early_models, RAW / "early_models.joblib", compress=3)

    evaluation_labels = load_human_labels(start.LABEL_PATH, population.videos)
    local_comparison = run_insertion_followup.compare_outputs(
        tuple(option.span for option in local_reference.values()),
        selected,
        evaluation_labels,
        population.fps,
        population.groups,
    )
    session_start_comparison = run_insertion_followup.compare_outputs(
        tuple(option.span for option in session_start_reference.values()),
        selected,
        evaluation_labels,
        population.fps,
        population.groups,
    )
    result = {
        "schema": "contact-closing-early-followup-result/1",
        "status": "complete",
        "variant": "early",
        "counts": {
            **expansion_counts,
            "base_options": len(population.options),
            "later_options": len(options),
            "later_insertion_options": sum(bool(option.inserted_events) for option in options),
            "existing_later_options": len(old_prepared["options"]),
        },
        "timings": {
            "prepare_seconds": prepare_seconds,
            "fit_seconds": fit_seconds,
            "total_seconds": perf_counter() - total_started,
            "opening_cache_records": len(opening_records),
            "final_opening_seconds": final_opening_seconds,
        },
        "comparison_to_local": local_comparison,
        "comparison_to_session_start": session_start_comparison,
        "target_reports": {
            "local": {
                "included": sum(target.included for target in local_targets.values()),
                "excluded": sum(not target.included for target in local_targets.values()),
                "opening_positive": sum(
                    target.included and target.opening_correct
                    for target in local_targets.values()
                ),
                "whole_positive": sum(
                    target.included and target.whole_rally_correct
                    for target in local_targets.values()
                ),
            },
            "whole": target_report,
        },
        "feature_audit": {
            "measurement_audit": dict(measurements.audit),
            "static_shape": list(static.shape),
            "insertion_shape": list(insertion.shape),
            "insertion_feature_names": list(insertion_names),
            "local_context_rows": len(local["groups"]),
        },
        "cross_group_upstream_caveat": (
            "saved detector score rows and reused local models retain upstream cross-group dependence"
        ),
        "local_models_reused_without_refit": True,
        "prepared_cache_saved": False,
        "fit_records": [record for *_, record in fold_results] + [final_record],
    }
    write_json(RESULTS / "early_result.json.gz", result)
    for comparison_name, comparison in (
        ("local", local_comparison),
        ("session_start", session_start_comparison),
    ):
        for tolerance in ("10", "5"):
            paired = comparison[tolerance]["paired"]
            print(
                f"early {comparison_name} {tolerance}: "
                f"repairs={len(paired['repaired'])} losses={len(paired['lost'])}",
                flush=True,
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--side-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=LATER_FEATURE_ROOT)
    parser.add_argument("--score-path", type=Path, default=SCORE_PATH)
    args = parser.parse_args()
    run(
        jobs=args.jobs,
        side_root=args.side_root,
        feature_root=args.feature_root,
        score_path=args.score_path,
    )


if __name__ == "__main__":
    main()
