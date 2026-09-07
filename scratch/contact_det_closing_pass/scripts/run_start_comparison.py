"""Compare fixed first-contact action models on development and validation videos."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.evaluation import (
    paired_sections,
    score_contacts,
    score_sections,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.features import join_physical_features
from scratch.contact_det_closing_pass.scripts.targets import EditTarget, assign_targets
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as old
from scratch.contact_det_full_ds_fit.scripts.experiment_config import VideoSpec
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model_config import (
    ModelSpec,
    load_rally_start_model_config,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

REPO_ROOT = prediction_io.REPO_ROOT
LABEL_PATH = old.LABEL_PATH
CONFIG_PATH = old.CONFIG_PATH
GROUPS = ("A", "B", "C", "D")
CUTOFF = 0.9
TOLERANCES = (10, 5)
ACTION_WIDTH = len(old.ACTION_FEATURE_NAMES)
RESULT_FILENAME = "start_comparison_result.json.gz"
PREDICTIONS_FILENAME = "start_comparison_predictions.json.gz"
VARIANT_FEATURES: dict[str, tuple[int, ...]] = {
    "summary_whole": tuple(range(ACTION_WIDTH)),
    "summary_opening": tuple(range(ACTION_WIDTH)),
    "physical_whole": tuple(range(180)),
    "physical_opening": tuple(range(180)),
}
VARIANTS = tuple(VARIANT_FEATURES)

ActionIdentity = tuple[str, int, int, str]
SectionIdentity = tuple[str, int]


def _raw_videos(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    videos = payload.get("videos")
    if not isinstance(videos, list) or any(not isinstance(video, Mapping) for video in videos):
        raise TypeError("validation prediction videos must be a list of objects")
    return tuple(videos)


def _video_identity(video: Mapping[str, Any]) -> str:
    identity = video.get("video")
    if not isinstance(identity, Mapping):
        identity = video
    fixture = identity.get("fixture")
    if not isinstance(fixture, str):
        raise TypeError("prediction video fixture must be text")
    return fixture


def _subset_development(
    pack: prediction_io.DevelopmentPredictionPack,
    groups: frozenset[str],
) -> tuple[tuple[VideoSpec, ...], tuple[FixedSpan, ...], dict[str, tuple[Any, ...]]]:
    fixtures = {
        fixture for fixture, group in pack.group_by_fixture.items() if group in groups
    }
    videos = tuple(video for video in pack.videos if video.fixture in fixtures)
    spans = tuple(span for span in pack.spans if span.fixture in fixtures)
    events = {
        fixture: values
        for fixture, values in pack.events_by_fixture.items()
        if fixture in fixtures
    }
    if {video.fixture for video in videos} != fixtures or set(events) != fixtures:
        raise ValueError("development fixture subset is incomplete")
    return videos, spans, events


def _valid_action_spans(
    rows: Sequence[old.ActionRow],
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[Any]],
) -> dict[ActionIdentity, FixedSpan | None]:
    """Build the label-free validity mask using the existing action function."""
    spans_by_identity = {(span.fixture, span.span_id): span for span in spans}
    if len(spans_by_identity) != len(spans):
        raise ValueError("section identities repeat")
    previous_end: dict[SectionIdentity, int] = {}
    last_end: dict[str, int] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        previous_end[identity] = last_end.get(span.fixture, -1)
        last_end[span.fixture] = span.end_frame

    revised: dict[ActionIdentity, FixedSpan | None] = {}
    for row in rows:
        section = row.section_identity
        span = spans_by_identity.get(section)
        if span is None:
            raise ValueError(f"{section}: candidate section is missing")
        revised[row.identity] = old._action_span(
            span,
            row.candidate,
            row.action,
            events_by_fixture[row.candidate.fixture],
            previous_end[section],
        )
    if set(revised) != {row.identity for row in rows}:
        raise ValueError("action validity coverage differs")
    return revised


def _positive_scores(model: Any, values: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("action model positive class differs")
    scores = np.asarray(model.predict_proba(values))[:, int(positive[0])]
    if len(scores) != len(values) or not np.isfinite(scores).all():
        raise ValueError("action model scores differ")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("action model scores must be probabilities")
    return scores


def _target_positive(target: EditTarget, target_kind: str) -> bool:
    if target_kind == "opening":
        return target.opening_correct
    if target_kind == "whole":
        return target.whole_rally_correct
    raise ValueError(f"unknown target kind {target_kind}")


def _training_rows(
    rows: Sequence[old.ActionRow],
    targets: Mapping[ActionIdentity, EditTarget],
    revised: Mapping[ActionIdentity, FixedSpan | None],
    target_kind: str,
) -> tuple[list[int], np.ndarray]:
    indices: list[int] = []
    values: list[int] = []
    for index, row in enumerate(rows):
        target = targets[row.identity]
        if target.included and revised[row.identity] is not None:
            indices.append(index)
            values.append(int(_target_positive(target, target_kind)))
    if not indices or set(values) != {0, 1}:
        raise ValueError("action model training needs positive and negative examples")
    return indices, np.asarray(values, dtype=np.uint8)


def _oof_scores(
    rows: Sequence[old.ActionRow],
    matrix: np.ndarray,
    targets: Mapping[ActionIdentity, EditTarget],
    revised: Mapping[ActionIdentity, FixedSpan | None],
    spec: ModelSpec,
    target_kind: str,
) -> tuple[dict[ActionIdentity, float], float]:
    scores: dict[ActionIdentity, float] = {}
    fit_seconds = 0.0
    for held_out_group in GROUPS:
        train_rows = [
            row for row in rows
            if row.candidate.group != held_out_group
        ]
        train_indices, values = _training_rows(
            train_rows,
            targets,
            revised,
            target_kind,
        )
        global_indices = [
            index for index, row in enumerate(rows)
            if row.candidate.group != held_out_group
        ]
        selected_indices = [global_indices[index] for index in train_indices]
        model = old.make_action_model(spec)
        started = perf_counter()
        model.fit(matrix[selected_indices], values)
        fit_seconds += perf_counter() - started
        held_indices = [
            index for index, row in enumerate(rows)
            if row.candidate.group == held_out_group
        ]
        held_scores = _positive_scores(model, matrix[held_indices])
        for index, score in zip(held_indices, held_scores, strict=True):
            identity = rows[index].identity
            if identity in scores:
                raise ValueError("held-out action score identity repeats")
            scores[identity] = float(score)
    if set(scores) != {row.identity for row in rows}:
        raise ValueError("held-out action score coverage differs")
    return scores, fit_seconds


def _fit_final(
    rows: Sequence[old.ActionRow],
    matrix: np.ndarray,
    targets: Mapping[ActionIdentity, EditTarget],
    revised: Mapping[ActionIdentity, FixedSpan | None],
    spec: ModelSpec,
    target_kind: str,
) -> tuple[Any, float]:
    indices, values = _training_rows(rows, targets, revised, target_kind)
    model = old.make_action_model(spec)
    started = perf_counter()
    model.fit(matrix[indices], values)
    return model, perf_counter() - started


def _select(
    rows: Sequence[old.ActionRow],
    scores: Mapping[ActionIdentity, float],
    revised: Mapping[ActionIdentity, FixedSpan | None],
) -> dict[SectionIdentity, old.ActionRow]:
    valid_rows = [row for row in rows if revised[row.identity] is not None]
    valid_scores = {row.identity: scores[row.identity] for row in valid_rows}
    return old.select_actions(valid_rows, valid_scores, CUTOFF)


def _target_summary(
    rows: Sequence[old.ActionRow],
    targets: Mapping[ActionIdentity, EditTarget],
    target_kind: str,
) -> dict[str, Any]:
    included = [row for row in rows if targets[row.identity].included]
    positives = [
        row for row in included
        if _target_positive(targets[row.identity], target_kind)
    ]
    sections = Counter(row.section_identity for row in positives)
    return {
        "action_rows": len(rows),
        "included_actions": len(included),
        "positive_actions": len(positives),
        "positive_sections": len(sections),
        "multi_positive_sections": sum(count > 1 for count in sections.values()),
        "positive_definition": (
            f"included actions with {target_kind}_correct=true; every acceptable "
            "action remains eligible as a positive"
        ),
    }


def _identity_row(identity: ActionIdentity) -> dict[str, Any]:
    fixture, span_id, frame, action = identity
    return {"fixture": fixture, "span_id": span_id, "frame": frame, "action": action}


def _section_rows(
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
    tolerance: int,
) -> list[dict[str, Any]]:
    return [
        {**row, "group": group_by_fixture[row["fixture"]]}
        for row in score_sections(spans, labels, fps, tolerance)
    ]


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {}
    for row in rows:
        fixture = str(row["fixture"])
        group = str(row["group"])
        counter = counters.setdefault(fixture, Counter())
        counter["sections"] += 1
        counter["timing_complete"] += int(row["timing_complete"])
        counter["fully_correct"] += int(row["fully_correct"])
        counter["side_rule_fully_correct"] += int(row["side_rule_fully_correct"])
        counter["group"] = group
    by_video = [
        {"fixture": fixture, **dict(counter)}
        for fixture, counter in sorted(counters.items())
    ]
    group_counters: dict[str, Counter[str]] = {}
    for row in rows:
        group = str(row["group"])
        counter = group_counters.setdefault(group, Counter())
        counter["sections"] += 1
        counter["timing_complete"] += int(row["timing_complete"])
        counter["fully_correct"] += int(row["fully_correct"])
        counter["side_rule_fully_correct"] += int(row["side_rule_fully_correct"])
    return {
        "by_video": by_video,
        "by_group": {
            group: dict(counts) for group, counts in sorted(group_counters.items())
        },
    }


def _side_fixed_pair(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before_fixed = [
        {**row, "fully_correct": row["side_rule_fully_correct"]}
        for row in before
    ]
    after_fixed = [
        {**row, "fully_correct": row["side_rule_fully_correct"]}
        for row in after
    ]
    return paired_sections(before_fixed, after_fixed)


def _evaluate_streams(
    baseline: ContactStreams,
    revised: ContactStreams,
    labels: HumanLabels,
    fps: Mapping[str, float],
    group_by_fixture: Mapping[str, str],
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        baseline_raw = _section_rows(
            baseline.spans, labels, fps, group_by_fixture, tolerance
        )
        revised_raw = _section_rows(
            revised.spans, labels, fps, group_by_fixture, tolerance
        )
        baseline_fixed_stream = old.apply_whole_rally_alternation(baseline)
        revised_fixed_stream = old.apply_whole_rally_alternation(revised)
        baseline_fixed = _section_rows(
            baseline_fixed_stream.spans, labels, fps, group_by_fixture, tolerance
        )
        revised_fixed = _section_rows(
            revised_fixed_stream.spans, labels, fps, group_by_fixture, tolerance
        )
        evaluations[str(tolerance)] = {
            "baseline_raw": {"sections": baseline_raw, "summary": _summary(baseline_raw)},
            "edited_raw": {"sections": revised_raw, "summary": _summary(revised_raw)},
            "baseline_fixed_side": {
                "sections": baseline_fixed,
                "summary": _summary(baseline_fixed),
            },
            "edited_fixed_side": {
                "sections": revised_fixed,
                "summary": _summary(revised_fixed),
            },
            "paired_raw": paired_sections(baseline_raw, revised_raw),
            "paired_fixed_side": _side_fixed_pair(baseline_fixed, revised_fixed),
        }
    return evaluations


def _harm_metrics(
    selections: Mapping[SectionIdentity, old.ActionRow],
    targets: Mapping[ActionIdentity, EditTarget],
    target_kind: str,
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_section = {
        (str(row["fixture"]), int(row["span_id"])): row for row in baseline_rows
    }
    selected_rows: list[dict[str, Any]] = []
    for _section, row in sorted(selections.items()):
        target = targets[row.identity]
        selected_rows.append({
            **_identity_row(row.identity),
            "target_included": target.included,
            "target_reason": target.reason,
            "correct_local_edit": (
                target.opening_correct if target.included else None
            ),
            "correct_whole_rally_edit": (
                target.whole_rally_correct if target.included else None
            ),
            "unnecessary_added": target.unnecessary_added if target.included else None,
            "real_contacts_removed": target.real_contacts_removed if target.included else None,
        })
    included = [row for row in selected_rows if row["target_included"]]
    unknown = [row for row in selected_rows if not row["target_included"]]
    correct = [row for row in included if row["correct_local_edit"]]
    originally_wrong = [
        row for row in selected_rows
        if not baseline_by_section[
            (row["fixture"], row["span_id"])
        ]["fully_correct"]
    ]
    return {
        "model_target_kind": target_kind,
        "selected_actions": len(selected_rows),
        "selected_targets_included": len(included),
        "unknown_selected": len(unknown),
        "correct_local_edits": len(correct),
        "unnecessary_added": sum(row["unnecessary_added"] or 0 for row in included),
        "real_contacts_removed": sum(row["real_contacts_removed"] or 0 for row in included),
        "errors_in_originally_wrong_sections": sum(
            row["target_included"] and not row["correct_local_edit"]
            for row in originally_wrong
        ),
        "selected": selected_rows,
    }


def _group_lineage(
    raw_training_videos: Sequence[Mapping[str, Any]],
    validation_fixtures: frozenset[str],
) -> dict[str, Any]:
    leaks = []
    for video in raw_training_videos:
        fixture = _video_identity(video)
        training = video.get("model_training_videos", [])
        if not isinstance(training, list):
            raise TypeError(f"{fixture}: model_training_videos must be a list")
        leaks.extend((fixture, name) for name in training if name in validation_fixtures)
    if leaks:
        raise ValueError(f"validation fixtures appear in A-D detector training lists: {leaks}")
    return {
        "candidate_source_commit": "f08621a8",
        "upstream_detector_score_mode": "full_raw",
        "upstream_detector_folds": "cached group scores from prediction_io",
        "validation_fixtures_excluded_from_a_d_detector_training_lists": True,
        "cached_detector_folds_nested_in_action_folds": False,
        "nested_caveat": (
            "The cached upstream detector folds are not nested inside these action "
            "folds. A held-out group's labels may have trained detectors that made "
            "scores used in another group's action training rows."
        ),
    }


def run_experiment(feature_root: Path, output_root: Path) -> dict[str, Any]:
    """Run the fixed four-variant comparison and freeze V choices before labels."""
    config = load_rally_start_model_config(CONFIG_PATH)
    spec = next(model for model in config.models if model.model_id == "shallow_hgb")
    if spec.model_id != "shallow_hgb":
        raise ValueError("shallow_hgb model specification is missing")

    pack = prediction_io.load_development_predictions()
    d_videos, d_spans, d_events = _subset_development(pack, frozenset(GROUPS))
    v_videos, v_spans, v_events = _subset_development(pack, frozenset({"V"}))
    raw_training_videos = old._candidate_videos()
    d_rows = build_candidate_rows(raw_training_videos, default_group="V")
    validation_payload = prediction_io.read_json(prediction_io.VALIDATION_PREDICTIONS)
    raw_validation_videos = _raw_videos(validation_payload)
    v_rows = build_candidate_rows(raw_validation_videos, default_group="V")
    validation_fixtures = frozenset(video.fixture for video in v_videos)
    if frozenset(row.fixture for row in v_rows) != validation_fixtures:
        raise ValueError("validation candidate fixtures differ from the fixed V set")
    if {row.group for row in d_rows} != frozenset(GROUPS):
        raise ValueError("candidate rows do not cover A-D")
    if {row.group for row in v_rows} != {"V"}:
        raise ValueError("validation candidate rows are not in V")
    lineage = _group_lineage(raw_training_videos, validation_fixtures)

    d_action_rows = old.build_action_rows(d_rows)
    v_action_rows = old.build_action_rows(v_rows)
    d_revised = _valid_action_spans(d_action_rows, d_spans, d_events)
    v_revised = _valid_action_spans(v_action_rows, v_spans, v_events)
    d_matrix, feature_names, d_join_audit = join_physical_features(d_action_rows, Path(feature_root))
    v_matrix, v_feature_names, v_join_audit = join_physical_features(v_action_rows, Path(feature_root))
    if feature_names != v_feature_names:
        raise ValueError("development and validation feature names differ")

    d_labels = load_human_labels(LABEL_PATH, d_videos)
    d_fps = {video.fixture: video.fps for video in d_videos}
    v_fps = {video.fixture: video.fps for video in v_videos}
    d_targets_by_tolerance = {
        tolerance: assign_targets(
            d_action_rows, d_spans, d_events, d_labels, d_fps,
            tolerance_base30=tolerance,
        )[0]
        for tolerance in TOLERANCES
    }
    d_targets = d_targets_by_tolerance[10]
    d_baseline = old.apply_selected_actions(d_spans, d_events, {})
    d_group_by_fixture = dict(pack.group_by_fixture)
    v_group_by_fixture = {fixture: "V" for fixture in validation_fixtures}

    development_results: dict[str, Any] = {}
    validation_models: dict[str, Any] = {}
    prediction_variants: dict[str, Any] = {}
    total_oof_fit_seconds = 0.0
    for variant in VARIANTS:
        target_kind = "opening" if variant.endswith("opening") else "whole"
        columns = VARIANT_FEATURES[variant]
        d_variant_matrix = d_matrix[:, columns]
        v_variant_matrix = v_matrix[:, columns]
        d_variant_targets = d_targets
        oof_scores, oof_fit_seconds = _oof_scores(
            d_action_rows,
            d_variant_matrix,
            d_variant_targets,
            d_revised,
            spec,
            target_kind,
        )
        total_oof_fit_seconds += oof_fit_seconds
        d_selections = _select(d_action_rows, oof_scores, d_revised)
        d_edited = old.apply_selected_actions(d_spans, d_events, d_selections)
        d_evaluation = _evaluate_streams(
            d_baseline, d_edited, d_labels, d_fps, d_group_by_fixture
        )
        d_harm = {
            str(tolerance): _harm_metrics(
                d_selections,
                d_targets_by_tolerance[tolerance],
                target_kind,
                d_evaluation[str(tolerance)]["baseline_fixed_side"]["sections"],
            )
            for tolerance in TOLERANCES
        }
        final_model, final_fit_seconds = _fit_final(
            d_action_rows,
            d_variant_matrix,
            d_variant_targets,
            d_revised,
            spec,
            target_kind,
        )
        predict_started = perf_counter()
        v_scores_array = _positive_scores(final_model, v_variant_matrix)
        v_scores = {
            row.identity: float(score)
            for row, score in zip(v_action_rows, v_scores_array, strict=True)
        }
        v_selections = _select(v_action_rows, v_scores, v_revised)
        predict_seconds = perf_counter() - predict_started
        apply_started = perf_counter()
        v_edited = old.apply_selected_actions(v_spans, v_events, v_selections)
        apply_seconds = perf_counter() - apply_started
        validation_models[variant] = {
            "selections": v_selections,
            "scores": v_scores,
            "edited": v_edited,
            "fit_seconds": final_fit_seconds,
            "predict_seconds": predict_seconds,
            "apply_seconds": apply_seconds,
        }
        print(f"{variant}: fitted, predicted V, and applied selected actions")
        prediction_variants[variant] = {
            "target_kind": target_kind,
            "feature_names": [feature_names[index] for index in columns],
            "scores": [
                {
                    **_identity_row(row.identity),
                    "score": v_scores[row.identity],
                    "valid_action": v_revised[row.identity] is not None,
                }
                for row in v_action_rows
            ],
            "selected_actions": [
                _identity_row(row.identity)
                for row in v_selections.values()
            ],
        }
        development_results[variant] = {
            "target_kind": target_kind,
            "feature_names": [feature_names[index] for index in columns],
            "oof_scores": [
                {
                    **_identity_row(row.identity),
                    "score": oof_scores[row.identity],
                    "valid_action": d_revised[row.identity] is not None,
                }
                for row in d_action_rows
            ],
            "target_summary": _target_summary(d_action_rows, d_targets, target_kind),
            "oof_fit_seconds": oof_fit_seconds,
            "selection_cutoff": CUTOFF,
            "selected_actions": [
                _identity_row(row.identity) for row in d_selections.values()
            ],
            "harm": d_harm,
            "evaluation": d_evaluation,
        }

    prediction_path = Path(output_root) / PREDICTIONS_FILENAME
    write_json(
        prediction_path,
        {
            "schema": "contact-detector-start-comparison-predictions/1",
            "status": "complete",
            "labels_read_for_prediction": False,
            "validation_group": "V",
            "validation_fixtures": sorted(validation_fixtures),
            "model_id": spec.model_id,
            "cutoff": CUTOFF,
            "feature_names": list(feature_names),
            "variants": prediction_variants,
        },
    )

    v_labels = load_human_labels(LABEL_PATH, v_videos)
    v_targets_by_tolerance = {
        tolerance: assign_targets(
            v_action_rows, v_spans, v_events, v_labels, v_fps,
            tolerance_base30=tolerance,
        )[0]
        for tolerance in TOLERANCES
    }
    baseline_v = old.apply_selected_actions(v_spans, v_events, {})
    baseline_v_fixed = old.apply_whole_rally_alternation(baseline_v)
    baseline_contact_scores = {
        str(tolerance): {
            "baseline_raw": score_contacts(
                baseline_v.events_by_fixture, v_labels, v_fps, tolerance
            ),
            "baseline_fixed_side": score_contacts(
                baseline_v_fixed.events_by_fixture, v_labels, v_fps, tolerance
            ),
        }
        for tolerance in TOLERANCES
    }
    validation_results: dict[str, Any] = {}
    for variant, model_result in validation_models.items():
        target_kind = "opening" if variant.endswith("opening") else "whole"
        v_selections = model_result["selections"]
        v_edited = model_result["edited"]
        evaluation = _evaluate_streams(
            baseline_v,
            v_edited,
            v_labels,
            v_fps,
            v_group_by_fixture,
        )
        contact_scores: dict[str, Any] = {}
        for tolerance in TOLERANCES:
            contact_scores[str(tolerance)] = {
                **baseline_contact_scores[str(tolerance)],
                "edited_raw": score_contacts(
                    v_edited.events_by_fixture, v_labels, v_fps, tolerance
                ),
                "edited_fixed_side": score_contacts(
                    old.apply_whole_rally_alternation(v_edited).events_by_fixture,
                    v_labels,
                    v_fps,
                    tolerance,
                ),
            }
        validation_results[variant] = {
            "target_kind": target_kind,
            "target_summary": {
                str(tolerance): _target_summary(
                    v_action_rows, v_targets_by_tolerance[tolerance], target_kind
                )
                for tolerance in TOLERANCES
            },
            "harm": {
                str(tolerance): _harm_metrics(
                    v_selections,
                    v_targets_by_tolerance[tolerance],
                    target_kind,
                    evaluation[str(tolerance)]["baseline_fixed_side"]["sections"],
                )
                for tolerance in TOLERANCES
            },
            "evaluation": evaluation,
            "contact_scores": contact_scores,
            "fit_seconds": model_result["fit_seconds"],
            "predict_seconds": model_result["predict_seconds"],
            "apply_seconds": model_result["apply_seconds"],
        }

    result: dict[str, Any] = {
        "schema": "contact-detector-start-comparison/1",
        "status": "complete",
        "run_id": "start-comparison-four-physical-action-variants",
        "labels_read_for_prediction": False,
        "model": {
            "model_id": spec.model_id,
            "kind": str(spec.kind),
            "settings": dict(spec.settings),
            "selection_cutoff": CUTOFF,
            "target_tolerance_at_30_fps": 10,
        },
        "inputs": {
            "candidate_inputs": str(old.TRAINING_INPUT_PATH.relative_to(REPO_ROOT)),
            "validation_inputs": str(prediction_io.VALIDATION_PREDICTIONS.relative_to(REPO_ROOT)),
            "development_split": str(prediction_io.DEVELOPMENT_SPLIT.relative_to(REPO_ROOT)),
            "model_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "contact_labels": str(LABEL_PATH.relative_to(REPO_ROOT)),
            "feature_dataset": Path(feature_root).name,
        },
        "counts": {
            "development_videos": len(d_videos),
            "validation_videos": len(v_videos),
            "development_candidate_rows": len(d_rows),
            "validation_candidate_rows": len(v_rows),
            "development_action_rows": len(d_action_rows),
            "validation_action_rows": len(v_action_rows),
        },
        "feature_names": list(feature_names),
        "feature_join_audit": {
            "development": d_join_audit,
            "validation": v_join_audit,
        },
        "lineage": lineage,
        "timings": {
            "development_oof_fit_seconds": total_oof_fit_seconds,
            "validation_variants": {
                variant: {
                    "fit_seconds": validation_results[variant]["fit_seconds"],
                    "predict_seconds": validation_results[variant]["predict_seconds"],
                    "apply_seconds": validation_results[variant]["apply_seconds"],
                }
                for variant in VARIANTS
            },
            "scoring_excluded": True,
        },
        "development_oof_descriptive": {
            "selection_note": (
                "A-D results are descriptive grouped action OOF results. They do "
                "not claim fully held-out performance because detector folds are cached."
            ),
            "variants": development_results,
        },
        "validation_frozen_choices": {
            "prediction_file": PREDICTIONS_FILENAME,
            "labels_loaded_after_prediction_file": True,
            "variants": validation_results,
        },
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("scratch/contact_det_full_ds_fit/raw/full_raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("scratch/contact_det_closing_pass/results"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="load the fixed config and exit without reading labels or fitting models",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    config = load_rally_start_model_config(CONFIG_PATH)
    if arguments.smoke:
        spec = next(model for model in config.models if model.model_id == "shallow_hgb")
        print({"model_id": spec.model_id, "variants": VARIANTS, "cutoff": CUTOFF})
        return
    feature_root = arguments.feature_root
    output_root = arguments.output_root
    if not feature_root.is_absolute():
        feature_root = REPO_ROOT / feature_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    payload = run_experiment(feature_root, output_root)
    write_json(output_root / RESULT_FILENAME, payload)


if __name__ == "__main__":
    main()
