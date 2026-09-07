"""Diagnose saved later-contact residuals and their candidate coverage."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts import census_missed_candidates as census
from scratch.contact_det_closing_pass.scripts import (
    prepare_later_inputs as later_inputs,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.followup_options import restore_choices
from scratch.contact_det_closing_pass.scripts.followup_residuals import residual_rows
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)
from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
    load_human_labels,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
DEFAULT_PREPARED = ROOT / "raw/later_run/prepared.joblib"
DEFAULT_RESULT_ROOT = ROOT / "results/later"
DEFAULT_OUTPUT = ROOT / "results/followups/residual_diagnosis.json.gz"
DEFAULT_FEATURE_ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw"
DEFAULT_SCORE_PATH = later_inputs.SCORE_PATH
TOLERANCES = (10, 5)
SectionIdentity = tuple[str, int]
EARLY_WINDOW_CATEGORIES = (
    "inside_window_rank_or_suppression",
    "outside_window",
    "no_scored",
    "no_window",
)
DEFAULT_EARLY_INPUTS = start.TRAINING_INPUT_PATH


@dataclass(frozen=True)
class _VideoEvidence:
    score_rows: np.ndarray
    score_frames: np.ndarray
    physical_rows: np.ndarray
    physical_frames: np.ndarray


@dataclass(frozen=True)
class _EarlyWindow:
    """Saved interval window used to build the early candidate shortlist."""

    interval_id: int
    prefix_start_frame: int
    fixed_contact_frame: int
    duplicate_distance_frames: int


def _load_early_windows(
    path: Path, fixtures: set[str],
) -> dict[SectionIdentity, tuple[_EarlyWindow, ...]]:
    """Load interval windows from the label-free early candidate inputs."""
    payload = prediction_io.read_json(path)
    if (
        payload.get("schema") != "contact-rally-start-training-inputs/1"
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("saved early candidate input record differs")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("saved early candidate videos must be a list")

    windows: dict[SectionIdentity, tuple[_EarlyWindow, ...]] = {}
    for raw_video in raw_videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("saved early candidate video must be an object")
        raw_lists = raw_video.get("candidate_lists")
        if raw_lists is None:
            continue
        if not isinstance(raw_lists, list):
            raise TypeError("saved early candidate lists must be a list")
        for raw_list in raw_lists:
            if not isinstance(raw_list, Mapping):
                raise TypeError("saved early candidate list must be an object")
            fixture = raw_list.get("fixture")
            span_id = raw_list.get("span_id")
            if not isinstance(fixture, str) or type(span_id) is not int:
                raise ValueError("saved early candidate list identity is malformed")
            if fixture not in fixtures:
                continue
            identity = (fixture, span_id)
            if identity in windows:
                raise ValueError(f"{identity}: saved early candidate list repeats")
            windows[identity] = (
                _EarlyWindow(
                    interval_id=int(raw_list["interval_id"]),
                    prefix_start_frame=int(raw_list["prefix_start_frame"]),
                    fixed_contact_frame=int(raw_list["fixed_contact_frame"]),
                    duplicate_distance_frames=int(raw_list["duplicate_distance_frames"]),
                ),
            )
    return windows


def _candidate_pool(population: Any, later_candidates: Any) -> tuple[Any, Any]:
    events: dict[SectionIdentity, dict[int, FixedEvent]] = defaultdict(dict)
    sources: dict[SectionIdentity, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for identity, candidates in later_candidates.items():
        for candidate in candidates:
            frame = int(candidate.frame)
            events[identity][frame] = candidate
            sources[identity][frame].add("later")
    for action in population.actions:
        candidate = action.candidate
        identity, frame = candidate.section_identity, int(candidate.frame)
        events[identity].setdefault(
            frame, FixedEvent(candidate.fixture, frame, float(candidate.contact_score), candidate.predicted_side),
        )
        sources[identity][frame].add("early")
    return (
        {identity: tuple(sorted(values.values(), key=lambda event: event.frame)) for identity, values in events.items()},
        {identity: dict(values) for identity, values in sources.items()},
    )


def _sorted_evidence(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows[np.argsort(rows["frame"].astype(np.int64), kind="stable")]
    return ordered, ordered["frame"].astype(np.int64)


def _load_evidence(
    population: Any, fixtures: set[str], feature_root: Path, score_path: Path,
) -> dict[str, _VideoEvidence]:
    all_scores = later_inputs._load_score_rows(score_path)
    feature_rows = census._load_feature_rows(feature_root, sorted(fixtures))
    evidence = {}
    for fixture in sorted(fixtures):
        scores = later_inputs._fixture_score_rows(
            all_scores, fixture, population.groups[fixture], population.fps[fixture],
        )
        scores, score_frames = _sorted_evidence(scores)
        physical, physical_frames = _sorted_evidence(feature_rows[fixture])
        evidence[fixture] = _VideoEvidence(scores, score_frames, physical, physical_frames)
    return evidence


def _nearby(rows: np.ndarray, frames: np.ndarray, target: int, tolerance: int) -> np.ndarray:
    left = int(np.searchsorted(frames, target - tolerance, side="left"))
    right = int(np.searchsorted(frames, target + tolerance, side="right"))
    return rows[left:right]


def _early_window_rows(
    score_rows: np.ndarray, windows: Sequence[_EarlyWindow],
) -> np.ndarray:
    """Return scored rows in the saved interval/prefix/fixed-frame window."""
    if not len(score_rows) or not windows:
        return score_rows[:0]
    mask = np.zeros(len(score_rows), dtype=bool)
    for window in windows:
        mask |= (
            (score_rows["interval_id"] == window.interval_id)
            & (score_rows["frame"] >= window.prefix_start_frame)
            & (score_rows["frame"] < window.fixed_contact_frame)
        )
    return score_rows[mask]


def _early_window_category(
    contact: Mapping[str, Any], score_rows: np.ndarray,
    windows: Sequence[_EarlyWindow],
) -> str | None:
    """Classify a first-contact opportunity against the exact early window."""
    if contact["kind"] != "first":
        return None
    if (contact["candidate_frames_within_tolerance"]
            or contact["existing_predicted_frame_within_tolerance"]):
        return None
    if not windows:
        return "no_window"
    if not len(score_rows):
        return "no_scored"
    if len(_early_window_rows(score_rows, windows)):
        return "inside_window_rank_or_suppression"
    return "outside_window"


def _annotate_coverage(
    rows: list[dict[str, Any]], sources: dict[SectionIdentity, dict[int, set[str]]],
    evidence: dict[str, _VideoEvidence], tolerance_by_fixture: dict[str, int],
    early_windows: Mapping[SectionIdentity, Sequence[_EarlyWindow]] | None = None,
) -> None:
    early_windows = {} if early_windows is None else early_windows
    for row in rows:
        identity = (str(row["fixture"]), int(row["span_id"]))
        for contact in row["unmatched_gt_contacts"] or ():
            target = int(contact["gt_frame"])
            candidate_frames = [int(frame) for frame in contact["candidate_frames_within_tolerance"]]
            candidate_sources = sorted({
                source for frame in candidate_frames for source in sources.get(identity, {}).get(frame, ())
            })
            video = evidence[identity[0]]
            tolerance = tolerance_by_fixture[identity[0]]
            score_rows = _nearby(video.score_rows, video.score_frames, target, tolerance)
            physical_rows = _nearby(video.physical_rows, video.physical_frames, target, tolerance)
            category = (
                "matching_competition" if contact["existing_predicted_frame_within_tolerance"] else
                "shortlisted_not_chosen" if candidate_frames else
                "scored_not_shortlisted" if len(score_rows) else
                "physical_not_scored" if len(physical_rows) else "no_saved_feature"
            )
            contact.update({
                "candidate_sources_within_tolerance": candidate_sources,
                "nearby_score_frames": [int(frame) for frame in score_rows["frame"]],
                "nearby_physical_frames": [int(frame) for frame in physical_rows["frame"]],
                "coverage_category": category,
                "early_window_category": _early_window_category(
                    contact, score_rows, early_windows.get(identity, ()),
                ),
            })


_ERRORS = (
    ("missing_first", lambda row: None if row["missing_first"] is None else bool(row["missing_first"])),
    ("missing_exactly_one_later", lambda row: None if row["missing_later_count"] is None else row["missing_later_count"] == 1),
    (
        "missing_more_than_one_later",
        lambda row: None if row["missing_later_count"] is None else row["missing_later_count"] > 1,
    ),
    (
        "extra_unmatched_prediction",
        lambda row: None if row["unmatched_predictions"] is None else row["unmatched_predictions"] > 0,
    ),
    (
        "wrong_voted_sides",
        lambda row: None if row["wrong_voted_side_count"] is None else row["wrong_voted_side_count"] > 0,
    ),
    ("incomplete_boundary", lambda row: None if row["known"] is not True else row["boundary_incomplete"]),
    ("multiple_labels", lambda row: row["multiple_rallies"]),
    ("no_labels", lambda row: row["no_labels"]),
)


def _error_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows), "known": sum(row["known"] is True for row in rows),
        "unknown": sum(row["known"] is not True for row in rows), "errors": {},
    }
    for name, predicate in _ERRORS:
        values = [predicate(row) for row in rows]
        result["errors"][name] = {
            "true": sum(value is not None and bool(value) for value in values),
            "false": sum(value is not None and not bool(value) for value in values),
            "unknown": sum(value is None for value in values),
        }
    return result


def _coverage_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "first": {"covered": 0, "missed": 0}, "later": {"covered": 0, "missed": 0},
        "unknown_sections": 0,
        "early_window_category_counts": {
            category: 0 for category in EARLY_WINDOW_CATEGORIES
        },
    }
    for row in rows:
        if row["unmatched_gt_contacts"] is None:
            result["unknown_sections"] += 1
            continue
        for contact in row["unmatched_gt_contacts"]:
            bucket = "first" if contact["kind"] == "first" else "later"
            field = "covered" if contact["candidate_frames_within_tolerance"] else "missed"
            result[bucket][field] += 1
            category = contact.get("early_window_category")
            if category in result["early_window_category_counts"]:
                result["early_window_category_counts"][category] += 1
    return result


def _acceptance(path: Path) -> tuple[dict[SectionIdentity, dict[str, Any]], float]:
    payload = prediction_io.read_json(path)
    policy = payload["policies"]["all_evidence"]["target_rules"]["0.95"]["nonempty_fallback"]
    threshold = float(policy["threshold"])
    if payload.get("status") != "complete" or not np.isfinite(threshold):
        raise ValueError("saved acceptance result has no finite complete policy")
    by_identity = {}
    for record in payload["rows"]:
        identity = (str(record["fixture"]), int(record["span_id"]))
        if identity in by_identity:
            raise ValueError(f"duplicate acceptance row: {identity}")
        score = float(record["acceptance_all_evidence_score"])
        if not np.isfinite(score):
            raise ValueError(f"non-finite acceptance score: {identity}")
        by_identity[identity] = {"score": score, "accepted": score >= threshold}
    return by_identity, threshold


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = {
        "baseline": rows, "accepted": [row for row in rows if row["accepted"]],
        "rejected": [row for row in rows if not row["accepted"]],
    }
    return {
        **{name: _error_counts(partition) for name, partition in partitions.items()},
        "candidate_coverage": {name: _coverage_counts(partition) for name, partition in partitions.items()},
    }


def run(
    prepared_path: Path = DEFAULT_PREPARED, result_root: Path = DEFAULT_RESULT_ROOT,
    feature_root: Path = DEFAULT_FEATURE_ROOT, score_path: Path = DEFAULT_SCORE_PATH,
    output_path: Path = DEFAULT_OUTPUT, early_inputs_path: Path = DEFAULT_EARLY_INPUTS,
) -> dict[str, Any]:
    prepared = joblib.load(prepared_path)
    population = prepared["base_population"]
    options = tuple(prepared["options"])
    margin = prediction_io.read_json(result_root / "later_margin_predictions.json.gz")
    selected = restore_choices(options, margin["selected_actions"])
    candidates, sources = _candidate_pool(population, prepared["later_candidates"])
    labels = load_human_labels(start.LABEL_PATH, population.videos)
    acceptance, threshold = _acceptance(result_root / "later_acceptance_result.json.gz")
    if set(selected) != set(acceptance):
        raise ValueError("acceptance rows do not cover current selected sections")
    fixtures = {option.span.fixture for option in selected.values()}
    evidence = _load_evidence(population, fixtures, feature_root, score_path)
    early_windows = _load_early_windows(early_inputs_path, fixtures)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    by_group: dict[str, Any] = {}
    for tolerance_base30 in TOLERANCES:
        rows = residual_rows(selected, candidates, labels, population.fps, tolerance_base30)
        tolerance_by_fixture = {
            fixture: scale_base30_frames(tolerance_base30, population.fps[fixture]) for fixture in fixtures
        }
        for row in rows:
            identity = (str(row["fixture"]), int(row["span_id"]))
            row.update({
                "tolerance_base30": tolerance_base30,
                "scaled_tolerance_frames": tolerance_by_fixture[identity[0]],
                "group": population.groups[identity[0]],
                "acceptance_all_evidence_score": acceptance[identity]["score"],
                "acceptance_threshold": threshold,
                "accepted": acceptance[identity]["accepted"],
            })
        _annotate_coverage(rows, sources, evidence, tolerance_by_fixture, early_windows)
        all_rows.extend(rows)
        counts[str(tolerance_base30)] = _summaries(rows)
        for group in sorted(set(population.groups.values())):
            group_rows = [row for row in rows if row["group"] == group]
            by_group.setdefault(group, {})[str(tolerance_base30)] = _summaries(group_rows)
    payload = {
        "schema": "contact-followup-residual-diagnosis/1", "status": "complete",
        "accepted_threshold": threshold, "counts": counts, "by_group": by_group, "rows": all_rows,
        "inputs": {
            "prepared_file": prepared_path.name, "margin_predictions_file": "later_margin_predictions.json.gz",
            "acceptance_result_file": "later_acceptance_result.json.gz", "score_file": score_path.name,
            "feature_root": feature_root.name, "early_inputs_file": early_inputs_path.name,
        },
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--score-path", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--early-inputs", type=Path, default=DEFAULT_EARLY_INPUTS)
    args = parser.parse_args()
    run(args.prepared, args.result_root, args.feature_root, args.score_path, args.output, args.early_inputs)


if __name__ == "__main__":
    main()
