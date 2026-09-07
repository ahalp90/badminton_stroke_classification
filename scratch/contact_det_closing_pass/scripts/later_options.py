"""Add real saved-score later contacts to the existing whole-rally alternatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
    _side_features,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import (
    CombinedAction,
    _action_priority,
)
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    scale_base30_frames,
)

MAX_LATER_CANDIDATES = 6
MIN_EDIT_ADVANTAGE = 0.05
SectionIdentity = tuple[str, int]


@dataclass(frozen=True)
class LaterOption:
    """An existing start/deletion choice followed by zero, one or two insertions."""

    base: CombinedAction
    inserted: FixedEvent | None
    span: FixedSpan
    second_inserted: FixedEvent | None = None

    @property
    def inserted_events(self) -> tuple[FixedEvent, ...]:
        if self.inserted is None:
            return ()
        if self.second_inserted is None:
            return (self.inserted,)
        return (self.inserted, self.second_inserted)

    @property
    def proxy(self) -> CombinedAction:
        """Preserve the start/deletion inputs while evaluating the revised events."""
        return replace(self.base, span=self.span)


def shortlist_frames(
    span: FixedSpan, scores: np.ndarray, fps: float, limit: int = MAX_LATER_CANDIDATES,
) -> list[int]:
    """Rank saved later frames without labels, suppressing nearby alternatives."""
    if not span.events:
        return []
    distance = scale_base30_frames(DUPLICATE_DISTANCE_AT_30_FPS, fps)
    frames = scores["frame"].astype(np.int64)
    eligible = (frames > span.events[0].frame + distance) & (frames < span.end_frame)
    for event in span.events:
        clear_of_contact = np.abs(frames - event.frame) > distance
        eligible = eligible & clear_of_contact
    indices = np.flatnonzero(eligible & np.isfinite(scores["contact_score"]))
    ranked = sorted(indices, key=lambda index: (-float(scores[index]["contact_score"]), int(frames[index])))
    selected: list[int] = []
    for index in ranked:
        frame = int(frames[index])
        if all(abs(frame - other) > distance for other in selected):
            selected.append(frame)
            if len(selected) == limit:
                break
    return selected


def build_later_options(
    base_options: Sequence[CombinedAction], candidates: Mapping[SectionIdentity, Sequence[FixedEvent]],
    fps: Mapping[str, float], max_insertions: int = 1,
) -> tuple[LaterOption, ...]:
    """Cross existing options with compatible saved candidates, without labels."""
    if max_insertions not in (1, 2):
        raise ValueError("The later comparison supports one or two insertions")
    output = []
    for base in base_options:
        output.append(LaterOption(base, None, base.span))
        distance = scale_base30_frames(DUPLICATE_DISTANCE_AT_30_FPS, fps[base.span.fixture])
        eligible = []
        for candidate in candidates.get(base.identity, ()):
            if candidate.fixture != base.span.fixture:
                raise ValueError("Later candidate belongs to another fixture")
            if not base.span.start_frame <= candidate.frame < base.span.end_frame:
                continue
            if any(abs(candidate.frame - event.frame) <= distance for event in base.span.events):
                continue
            events = tuple(sorted((*base.span.events, candidate), key=lambda event: event.frame))
            output.append(LaterOption(base, candidate, replace(base.span, events=events)))
            eligible.append(candidate)
        if max_insertions == 2:
            for first, second in combinations(eligible, 2):
                if abs(first.frame - second.frame) <= distance:
                    continue
                events = tuple(sorted((*base.span.events, first, second), key=lambda event: event.frame))
                output.append(LaterOption(base, first, replace(base.span, events=events), second))
    return tuple(output)


def option_record(option: LaterOption) -> dict[str, Any]:
    record = {
        "fixture": option.span.fixture, "span_id": option.span.span_id, "kind": option.base.kind,
        "candidate_frame": option.base.candidate_frame, "deleted_frame": option.base.deleted_frame,
        "inserted_frame": None if option.inserted is None else option.inserted.frame,
        "start_frame": option.span.start_frame, "end_frame": option.span.end_frame,
    }
    if option.second_inserted is not None:
        record["second_inserted_frame"] = option.second_inserted.frame
    return record


def select_options(options: Sequence[LaterOption], scores: np.ndarray) -> dict[SectionIdentity, LaterOption]:
    """Choose the highest-scored complete output; prefer fewer edits on ties."""
    if len(options) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Later-option scores are incomplete")
    selected: dict[SectionIdentity, tuple[LaterOption, float]] = {}
    for option, score in zip(options, scores, strict=True):
        previous = selected.get(option.base.identity)
        priority = (len(option.inserted_events), _action_priority(option.base))
        if previous is not None:
            old, old_score = previous
            old_priority = (len(old.inserted_events), _action_priority(old.base))
            if score < old_score or (score == old_score and priority >= old_priority):
                continue
        selected[option.base.identity] = (option, float(score))
    return {identity: pair[0] for identity, pair in selected.items()}


def select_with_reference(
    options: Sequence[LaterOption], scores: np.ndarray,
    reference: Mapping[SectionIdentity, LaterOption], minimum_advantage: float = MIN_EDIT_ADVANTAGE,
) -> dict[SectionIdentity, LaterOption]:
    """Keep the reference output unless an alternative has a clear score advantage."""
    selected = select_options(options, scores)
    if set(reference) != set(selected):
        raise ValueError("Reference choices do not cover the option population")
    reference_scores = {}
    best_scores: dict[SectionIdentity, float] = {}
    for option, score in zip(options, scores, strict=True):
        identity = option.base.identity
        best_scores[identity] = max(float(score), best_scores.get(identity, -np.inf))
        original = reference[identity]
        if option.span == original.span:
            reference_scores[identity] = max(float(score), reference_scores.get(identity, -np.inf))
    if set(reference_scores) != set(reference):
        raise ValueError("Reference output is missing from the scored alternatives")
    for identity, score in best_scores.items():
        advantage = score - reference_scores[identity]
        if advantage <= 0.0 or advantage < minimum_advantage:
            selected[identity] = reference[identity]
    return selected


def apply_options(
    spans: Sequence[FixedSpan], events: Mapping[str, Sequence[FixedEvent]],
    selected: Mapping[SectionIdentity, LaterOption],
) -> ContactStreams:
    """Replace section events while preserving every out-of-section prediction."""
    owned: dict[str, set[int]] = {fixture: set() for fixture in events}
    revised: dict[str, dict[int, FixedEvent]] = {fixture: {} for fixture in events}
    output_spans = []
    for span in spans:
        owned[span.fixture].update(event.frame for event in span.events)
        after = selected[(span.fixture, span.span_id)].span
        output_spans.append(after)
        for event in after.events:
            previous = revised[span.fixture].get(event.frame)
            if previous is not None and previous != event:
                raise ValueError("Conflicting contact records across sections")
            revised[span.fixture][event.frame] = event
    for fixture, fixture_events in events.items():
        for event in fixture_events:
            if event.frame not in owned[fixture]:
                previous = revised[fixture].get(event.frame)
                if previous is not None and previous != event:
                    raise ValueError("Inserted contact conflicts with an existing full-stream event")
                revised[fixture][event.frame] = event
    streams = {fixture: tuple(sorted(rows.values(), key=lambda event: event.frame)) for fixture, rows in revised.items()}
    return ContactStreams(tuple(output_spans), streams)


def insertion_features(
    options: Sequence[LaterOption], fps: Mapping[str, float], measurements: PhysicalMeasurements,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Describe the inserted candidate, its neighbouring gaps and raw side change."""
    names = (
        "has_later_insertion", "later_score", "left_gap_seconds", "right_gap_seconds",
        "left_same_raw_side", "right_same_raw_side", "base_top_vote", "base_bot_vote",
        *(f"later__{name}" for name in measurements.names),
    )
    rows = []
    for option in options:
        candidate = option.inserted
        if candidate is None:
            rows.append([0.0, *([np.nan] * (len(names) - 1))])
            continue
        before = [event for event in option.base.span.events if event.frame < candidate.frame]
        after = [event for event in option.base.span.events if event.frame > candidate.frame]
        left = before[-1] if before else None
        right = after[0] if after else None
        side = candidate.predicted_side
        raw_sides = _side_features(option.base.span)
        block = measurements.values[(candidate.fixture, candidate.frame)]
        rows.append([
            1.0, candidate.timing_score,
            np.nan if left is None else (candidate.frame - left.frame) / fps[candidate.fixture],
            np.nan if right is None else (right.frame - candidate.frame) / fps[candidate.fixture],
            np.nan if left is None or side is None or left.predicted_side is None else float(left.predicted_side == side),
            np.nan if right is None or side is None or right.predicted_side is None else float(right.predicted_side == side),
            raw_sides[1], raw_sides[2], *block,
        ])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)), names
