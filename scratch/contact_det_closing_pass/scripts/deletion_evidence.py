"""Build label-free deletion evidence and local deletion targets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import overlapping_rallies
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    insertion_features,
)
from scratch.contact_det_closing_pass.scripts.local_deletion import deletion_effect
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)

SectionIdentity = tuple[str, int]
RallyIdentity = tuple[str, str]


def _event_lookup(
    original_events: Mapping[str, Sequence[FixedEvent]],
) -> dict[tuple[str, int], FixedEvent]:
    lookup: dict[tuple[str, int], FixedEvent] = {}
    for fixture, events in original_events.items():
        for event in events:
            if event.fixture != fixture:
                raise ValueError(f"{fixture}: original event fixture differs")
            identity = (fixture, event.frame)
            if identity in lookup and lookup[identity] != event:
                raise ValueError(f"{identity}: original event metadata differs")
            lookup[identity] = event
    return lookup


def _context_key(context: LaterOption) -> tuple[Any, ...]:
    after = context.base.span
    return (
        after.fixture,
        after.span_id,
        context.inserted.frame if context.inserted is not None else None,
        after.start_frame,
        after.end_frame,
        after.events,
    )


def deletion_inputs(
    options: Sequence[LaterOption],
    original_events: Mapping[str, Sequence[FixedEvent]],
    fps: Mapping[str, float],
    measurements: PhysicalMeasurements,
) -> dict[str, Any]:
    """Build deduplicated inverse-insertion features for actual deletions.

    A delete option's base.deleted_frame must identify an event in the
    original full stream. The inverse context presents it as the insertion
    candidate, with the option span as its after view.
    """
    event_lookup = _event_lookup(original_events)
    contexts: list[LaterOption] = []
    context_by_key: dict[tuple[Any, ...], int] = {}
    option_context_indices = np.full(len(options), -1, dtype=np.int64)

    for option_index, option in enumerate(options):
        deleted_frame = option.base.deleted_frame
        if deleted_frame is None:
            continue
        identity = (option.span.fixture, int(deleted_frame))
        removed = event_lookup.get(identity)
        if removed is None:
            raise KeyError(
                f"{identity}: deletion metadata does not name an original event"
            )
        if any(event.frame == removed.frame for event in option.span.events):
            raise ValueError(f"{identity}: deleted event remains in option span")
        after = option.span
        before = replace(
            after,
            events=tuple(sorted((*after.events, removed), key=lambda event: event.frame)),
        )
        context = LaterOption(
            CombinedAction("keep", None, None, after),
            removed,
            before,
        )
        key = _context_key(context)
        context_index = context_by_key.get(key)
        if context_index is None:
            context_index = len(contexts)
            context_by_key[key] = context_index
            contexts.append(context)
        option_context_indices[option_index] = context_index

    feature_matrix, feature_names = insertion_features(
        tuple(contexts), fps, measurements
    )
    return {
        "contexts": tuple(contexts),
        "features": feature_matrix,
        "feature_names": feature_names,
        "option_context_indices": option_context_indices,
    }


def _rally_identity(rally: RallyReference) -> RallyIdentity:
    return rally.fixture, rally.rally_id


def _retained_rallies(
    spans: Sequence[FixedSpan],
    labels: HumanLabels,
) -> tuple[
    dict[SectionIdentity, tuple[RallyReference, ...]],
    Counter[RallyIdentity],
]:
    overlaps = {
        (span.fixture, span.span_id): overlapping_rallies(span, labels)
        for span in spans
    }
    touching: Counter[RallyIdentity] = Counter()
    for rallies in overlaps.values():
        for rally in rallies:
            touching[_rally_identity(rally)] += 1
    return overlaps, touching


def _duplicate_supported(
    removed_frame: int,
    after: FixedSpan,
    rally: RallyReference,
    tolerance: int,
) -> bool:
    nearby_gt = tuple(
        frame for frame in rally.frames if abs(removed_frame - frame) <= tolerance
    )
    return any(
        abs(event.frame - gt_frame) <= tolerance
        for event in after.events
        for gt_frame in nearby_gt
    )


def deletion_targets(
    contexts: Sequence[LaterOption],
    original_spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps: Mapping[str, float],
    tolerance_base30: int = 10,
) -> np.ndarray:
    """Label deletion contexts as useful, harmful, or unsupported.

    A positive target preserves all GT contacts represented before removal and
    reduces extras. The removed event needs envelope support or an after-view
    duplicate; unrelated missing GT contacts do not invalidate a positive.
    """
    overlaps, touching = _retained_rallies(original_spans, labels)
    targets: list[int] = []
    overlaps_by_bounds = {}
    tolerances = {fixture: scale_base30_frames(tolerance_base30, source_fps) for fixture, source_fps in fps.items()}

    for context in contexts:
        before = context.span
        after = context.base.span
        identity = (after.fixture, after.span_id)
        rallies = overlaps.get(identity, ())
        if len(rallies) != 1:
            targets.append(-1)
            continue
        rally = rallies[0]
        rally_identity = _rally_identity(rally)
        if touching[rally_identity] != 1:
            targets.append(-1)
            continue
        bounds = (before.fixture, before.start_frame, before.end_frame)
        if bounds not in overlaps_by_bounds:
            overlaps_by_bounds[bounds] = overlapping_rallies(before, labels)
        # Inverse contexts preserve section edges; only their event lists differ.
        before_rallies = overlaps_by_bounds[bounds]
        if len(before_rallies) != 1 or _rally_identity(before_rallies[0]) != rally_identity:
            targets.append(-1)
            continue

        tolerance = tolerances[identity[0]]
        removed = context.inserted
        if removed is None:
            targets.append(-1)
            continue
        effect = deletion_effect(before, after, rally, tolerance)
        if effect["lost_gt_indices"]:
            targets.append(0)
            continue
        supported = (
            rally.frames[0] <= removed.frame <= rally.frames[-1]
            or _duplicate_supported(removed.frame, after, rally, tolerance)
        )
        targets.append(int(effect["useful"]) if supported else -1)

    return np.asarray(targets, dtype=np.int8)


def deletion_column(
    option_context_indices: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Expand context scores to one column in original option order."""
    indices = np.asarray(option_context_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if indices.ndim != 1:
        raise ValueError("option context indices must be one-dimensional")
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("deletion scores must be one-dimensional")
    if np.any(indices < -1):
        raise ValueError("option context indices must be -1 or non-negative")
    if np.any(indices >= len(values)):
        raise ValueError("option context index exceeds score count")
    output = np.full((len(indices), 1), np.nan, dtype=np.float64)
    selected = indices >= 0
    output[selected, 0] = values[indices[selected]]
    return output
