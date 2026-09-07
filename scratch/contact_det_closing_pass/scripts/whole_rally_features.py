"""Build label-free features for whole-rally action options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts import score_start_model
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_followup.scripts.score_keep_review import (
    FEATURE_NAMES as KEEP_REVIEW_FEATURE_NAMES,
)
from scratch.contact_det_followup.scripts.score_keep_review import build_feature_vector
from scratch.contact_det_followup.scripts.score_start_model import ActionRow

from .features import _frozen_feature_names, _load_fixture

ACTION_KINDS = (
    "keep",
    "add",
    "replace",
    "delete",
    "add_delete",
    "replace_delete",
)
START_ACTION_KINDS = ("add", "replace")
COMBINED_START_KINDS = ("add_delete", "replace_delete")
SIDE_FEATURE_NAMES = (
    "fraction_known",
    "fraction_known_starting_top",
    "fraction_known_starting_bot",
    "fraction_adjacent_known_same_side",
)

SectionIdentity = tuple[str, int]
FrameIdentity = tuple[str, int]
ActionScoreKey = tuple[str, int, int, str]


@dataclass(frozen=True)
class PhysicalMeasurements:
    """Frozen physical feature blocks indexed by fixture and source frame."""

    names: tuple[str, ...]
    values: dict[FrameIdentity, np.ndarray]
    audit: dict[str, Any]


def _all_nan(width: int) -> np.ndarray:
    return np.full(width, np.nan, dtype=np.float64)


def _frame_identity(fixture: str, frame: int) -> FrameIdentity:
    return fixture, int(frame)


def _requested_frames(
    action_rows: Sequence[ActionRow],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, set[int]]:
    requested: dict[str, set[int]] = {
        str(fixture): {int(event.frame) for event in events}
        for fixture, events in events_by_fixture.items()
    }
    for fixture, events in events_by_fixture.items():
        for event in events:
            if event.fixture != fixture:
                raise ValueError(f"{fixture}: fullstream event fixture differs")
    for row in action_rows:
        candidate = row.candidate
        requested.setdefault(candidate.fixture, set()).update(
            (int(candidate.frame), int(candidate.fixed_contact_frame))
        )
    return requested


def load_measurements(
    action_rows: Sequence[ActionRow],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    feature_root: Path,
) -> PhysicalMeasurements:
    """Load each requested fixture once and index its physical measurements.

    :param action_rows: Candidate/action rows whose candidate and fixed frames
        need physical blocks.
    :param events_by_fixture: Full event streams. Every event frame is also
        requested so whole-rally options can use its deleted frame.
    :param feature_root: Root containing ``videos/<fixture>/contact_features``.
    :return: Indexed feature blocks and a path-free load audit.
    """
    names = _frozen_feature_names()
    width = len(names)
    requested_by_fixture = _requested_frames(action_rows, events_by_fixture)
    values: dict[FrameIdentity, np.ndarray] = {}
    missing_identities: list[FrameIdentity] = []
    fixture_audit: dict[str, dict[str, int]] = {}
    root = Path(feature_root)

    for fixture, frames in requested_by_fixture.items():
        index = _load_fixture(root, fixture, names, frames)
        expected_fps = {
            float(row.candidate.fps)
            for row in action_rows
            if row.candidate.fixture == fixture
        }
        if len(expected_fps) > 1:
            raise ValueError(f"{fixture}: action row fps differs")
        if (
            index is not None
            and index.fps is not None
            and expected_fps
            and not np.isclose(
                index.fps,
                next(iter(expected_fps)),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"{fixture}: frozen feature fps differs from action row")
        missing_frames = 0
        measurement_nan_cells = 0
        for frame in sorted(frames):
            block = None if index is None else index.by_frame.get(frame)
            if block is None:
                block = _all_nan(width)
                missing_frames += 1
                missing_identities.append(_frame_identity(fixture, frame))
            else:
                block = np.asarray(block, dtype=np.float64)
                if block.shape != (width,):
                    raise ValueError(f"{fixture}/{frame}: physical block has wrong shape")
                if np.isinf(block).any():
                    raise ValueError(f"{fixture}/{frame}: physical block contains infinity")
            measurement_nan_cells += int(np.isnan(block).sum())
            values[_frame_identity(fixture, frame)] = block
        fixture_audit[fixture] = {
            "requested_frames": len(frames),
            "missing_frames": missing_frames,
            "measurement_nan_cells": measurement_nan_cells,
        }

    measurement_nan_cells_total = sum(
        fixture_values["measurement_nan_cells"]
        for fixture_values in fixture_audit.values()
    )
    audit: dict[str, Any] = {
        "requested_identity_count": len(values),
        "missing_identity_count": len(missing_identities),
        "missing_identities": missing_identities,
        "measurement_nan_cells": measurement_nan_cells_total,
        "fixtures": fixture_audit,
    }
    return PhysicalMeasurements(names, values, audit)


def _measurement_block(
    measurements: PhysicalMeasurements,
    fixture: str,
    frame: int | None,
) -> np.ndarray:
    width = len(measurements.names)
    if frame is None:
        return _all_nan(width)
    block = measurements.values.get(_frame_identity(fixture, frame))
    if block is None:
        return _all_nan(width)
    array = np.asarray(block, dtype=np.float64)
    if array.shape != (width,):
        raise ValueError(f"{fixture}/{frame}: physical block has wrong shape")
    if np.isinf(array).any():
        raise ValueError(f"{fixture}/{frame}: physical block contains infinity")
    return array


def action_matrix(
    action_rows: Sequence[ActionRow],
    measurements: PhysicalMeasurements,
) -> np.ndarray:
    """Join ten action inputs with candidate and fixed physical blocks."""
    rows = tuple(action_rows)
    width = len(measurements.names)
    if len(measurements.names) != len(set(measurements.names)):
        raise ValueError("physical measurement names repeat")
    if not rows:
        return np.empty((0, len(score_start_model.ACTION_FEATURE_NAMES) + 2 * width), dtype=np.float64)

    output: list[np.ndarray] = []
    for row in rows:
        features = np.asarray(row.features, dtype=np.float64)
        if features.shape != (len(score_start_model.ACTION_FEATURE_NAMES),):
            raise ValueError("action rows must contain ten features")
        if np.isinf(features).any():
            raise ValueError("action row features must not contain infinity")
        candidate = row.candidate
        output.append(
            np.concatenate(
                (
                    features,
                    _measurement_block(measurements, candidate.fixture, candidate.frame),
                    _measurement_block(
                        measurements,
                        candidate.fixture,
                        candidate.fixed_contact_frame,
                    ),
                )
            )
        )
    return np.asarray(output, dtype=np.float64)


def _span_lookup(spans: Sequence[FixedSpan]) -> dict[SectionIdentity, FixedSpan]:
    lookup: dict[SectionIdentity, FixedSpan] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in lookup:
            raise ValueError(f"{identity}: section identity repeats")
        lookup[identity] = span
    return lookup


def _action_row_lookup(action_rows: Sequence[ActionRow]) -> dict[tuple[str, int, int, str], ActionRow]:
    lookup: dict[tuple[str, int, int, str], ActionRow] = {}
    for row in action_rows:
        key = row.identity
        if key in lookup and lookup[key] != row:
            raise ValueError(f"{key}: action row identity repeats")
        lookup[key] = row
    return lookup


def _start_kind(kind: str) -> str | None:
    if kind in START_ACTION_KINDS:
        return kind
    if kind in COMBINED_START_KINDS:
        return kind.removesuffix("_delete")
    return None


def _start_features(
    option: CombinedAction,
    action_rows: Mapping[tuple[str, int, int, str], ActionRow],
) -> tuple[float, ...]:
    kind = _start_kind(option.kind)
    if kind is None:
        return (np.nan,) * len(score_start_model.ACTION_FEATURE_NAMES)
    if option.candidate_frame is None:
        raise ValueError(f"{option.identity}: start action candidate is missing")
    key = (*option.identity, option.candidate_frame, kind)
    row = action_rows.get(key)
    if row is None:
        raise KeyError(f"{key}: start action row is missing")
    values = tuple(float(value) for value in row.features)
    if len(values) != len(score_start_model.ACTION_FEATURE_NAMES):
        raise ValueError(f"{key}: start action row has the wrong feature count")
    if np.isinf(values).any():
        raise ValueError(f"{key}: start action row contains infinity")
    return values


def _deleted_score(option: CombinedAction, event_scores: Mapping[FrameIdentity, float]) -> float:
    if option.deleted_frame is None:
        return np.nan
    return event_scores.get(_frame_identity(option.span.fixture, option.deleted_frame), np.nan)


def _event_score_map(
    spans: Sequence[FixedSpan],
    options: Sequence[CombinedAction],
) -> dict[FrameIdentity, float]:
    event_scores: dict[FrameIdentity, float] = {}
    all_spans = (*spans, *(option.span for option in options))
    for span in all_spans:
        for event in span.events:
            identity = _frame_identity(event.fixture, event.frame)
            score = float(event.timing_score)
            previous = event_scores.get(identity)
            if previous is not None and previous != score:
                raise ValueError(f"{identity}: event timing score differs")
            event_scores[identity] = score
    return event_scores


def _side_features(span: FixedSpan) -> tuple[float, ...]:
    guesses = tuple(event.predicted_side for event in span.events)
    event_count = len(guesses)
    known = tuple(side for side in guesses if side is not None)
    known_count = len(known)
    fraction_known = 0.0 if event_count == 0 else known_count / event_count
    if known_count == 0:
        top_agreement = np.nan
        bot_agreement = np.nan
    else:
        top_matches = sum(
            side == ("Top" if index % 2 == 0 else "Bot")
            for index, side in enumerate(guesses)
            if side is not None
        )
        bot_matches = sum(
            side == ("Bot" if index % 2 == 0 else "Top")
            for index, side in enumerate(guesses)
            if side is not None
        )
        top_agreement = top_matches / known_count
        bot_agreement = bot_matches / known_count
    adjacent = tuple(
        (first, second)
        for first, second in pairwise(guesses)
        if first is not None and second is not None
    )
    same_side = (
        np.nan
        if not adjacent
        else sum(first == second for first, second in adjacent) / len(adjacent)
    )
    return (fraction_known, top_agreement, bot_agreement, same_side)


def _fps_for_fixture(fps: Mapping[str, float], fixture: str) -> float:
    if fixture not in fps:
        raise KeyError(f"missing fps for {fixture}")
    value = float(fps[fixture])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{fixture}: fps must be positive and finite")
    return value


def _one_hot(kind: str) -> tuple[float, ...]:
    return tuple(float(kind == candidate) for candidate in ACTION_KINDS)


def build_whole_features(
    options: Sequence[CombinedAction],
    baseline_spans: Sequence[FixedSpan],
    action_rows: Sequence[ActionRow],
    fps: Mapping[str, float],
    measurements: PhysicalMeasurements,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Build summary, raw-side and physical features for action options.

    :param options: Allowed label-free section actions.
    :param baseline_spans: Original predicted sections used as the before view.
    :param action_rows: Existing first-contact action inputs.
    :param fps: Frames per second indexed by fixture.
    :param measurements: Physical blocks loaded by :func:`load_measurements`.
    :return: Feature matrix, column names and contiguous feature-group indices.
    """
    baselines = _span_lookup(baseline_spans)
    starts = _action_row_lookup(action_rows)
    width = len(measurements.names)
    summary_names = tuple(
        [f"before__{name}" for name in KEEP_REVIEW_FEATURE_NAMES]
        + [f"after__{name}" for name in KEEP_REVIEW_FEATURE_NAMES]
        + [f"action__{kind}" for kind in ACTION_KINDS]
        + [f"existing_start__{name}" for name in score_start_model.ACTION_FEATURE_NAMES]
        + ["deleted_contact_score"]
    )
    side_names = tuple(
        [f"before__{name}" for name in SIDE_FEATURE_NAMES]
        + [f"after__{name}" for name in SIDE_FEATURE_NAMES]
    )
    physical_names = tuple(
        [f"original_fixed__{name}" for name in measurements.names]
        + [f"candidate__{name}" for name in measurements.names]
        + [f"deleted__{name}" for name in measurements.names]
    )
    names = (*summary_names, *side_names, *physical_names)
    summary_width = len(summary_names)
    side_width = len(side_names)
    event_scores = _event_score_map(baseline_spans, options)
    baseline_cache: dict[
        SectionIdentity,
        tuple[tuple[float, ...], tuple[float, ...], int | None, float],
    ] = {}
    groups = {
        "summary": tuple(range(summary_width)),
        "side": tuple(range(summary_width, summary_width + side_width)),
        "physical": tuple(
            range(summary_width + side_width, summary_width + side_width + 3 * width)
        ),
    }

    rows: list[np.ndarray] = []
    for option in options:
        if option.kind not in ACTION_KINDS:
            raise ValueError(f"{option.identity}: unknown combined action {option.kind!r}")
        original = baselines.get(option.identity)
        if original is None:
            raise KeyError(f"{option.identity}: baseline span is missing")
        cached = baseline_cache.get(option.identity)
        if cached is None:
            fixture_fps = _fps_for_fixture(fps, option.span.fixture)
            cached = (
                build_feature_vector(original, fixture_fps),
                _side_features(original),
                original.events[0].frame if original.events else None,
                fixture_fps,
            )
            baseline_cache[option.identity] = cached
        before_summary, before_side, original_fixed_frame, fixture_fps = cached
        summary = (
            *before_summary,
            *build_feature_vector(option.span, fixture_fps),
            *_one_hot(option.kind),
            *_start_features(option, starts),
            _deleted_score(option, event_scores),
        )
        side = (*before_side, *_side_features(option.span))
        candidate_frame = option.candidate_frame
        physical = (
            *_measurement_block(measurements, option.span.fixture, original_fixed_frame),
            *_measurement_block(measurements, option.span.fixture, candidate_frame),
            *_measurement_block(measurements, option.span.fixture, option.deleted_frame),
        )
        rows.append(np.asarray((*summary, *side, *physical), dtype=np.float64))
    matrix = (
        np.vstack(rows)
        if rows
        else np.empty((0, len(names)), dtype=np.float64)
    )
    return matrix, names, groups


def _prediction_pair(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("opening scores must contain two model values")
    pair = (float(value[0]), float(value[1]))
    if np.isinf(pair).any():
        raise ValueError("opening score contains infinity")
    return pair


def _start_score_key(option: CombinedAction) -> ActionScoreKey | None:
    kind = _start_kind(option.kind)
    if kind is None:
        return None
    if option.candidate_frame is None:
        raise ValueError(f"{option.identity}: start action candidate is missing")
    return (*option.identity, option.candidate_frame, kind)


def _required_start_score(
    key: ActionScoreKey,
    scores: Mapping[ActionScoreKey, tuple[float, float]],
) -> tuple[float, float]:
    if key not in scores:
        raise KeyError(f"{key}: opening score is missing")
    pair = _prediction_pair(scores[key])
    if not np.isfinite(pair).all():
        raise ValueError(f"{key}: opening score is not finite")
    return pair


def opening_score_features(
    options: Sequence[CombinedAction],
    scores: Mapping[ActionScoreKey, tuple[float, float]],
) -> np.ndarray:
    """Join selected and section-maximum start scores from passed predictions."""
    options_tuple = tuple(options)
    keys_by_section: dict[SectionIdentity, set[ActionScoreKey]] = {}
    for option in options_tuple:
        key = _start_score_key(option)
        if key is None:
            continue
        keys_by_section.setdefault(option.identity, set()).add(key)
    cached_scores = {
        key: _required_start_score(key, scores)
        for keys in keys_by_section.values()
        for key in keys
    }
    maximums = {
        identity: (
            max(cached_scores[key][0] for key in keys),
            max(cached_scores[key][1] for key in keys),
        )
        for identity, keys in keys_by_section.items()
    }

    rows: list[tuple[float, float, float, float]] = []
    for option in options_tuple:
        key = _start_score_key(option)
        if key is None:
            chosen = (np.nan, np.nan)
        else:
            chosen = cached_scores[key]
        maximum = maximums.get(option.identity, (np.nan, np.nan))
        rows.append((*chosen, *maximum))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), 4)
