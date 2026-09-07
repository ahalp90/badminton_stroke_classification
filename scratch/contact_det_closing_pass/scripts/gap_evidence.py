"""Summarise label-free local evidence in the gaps of selected spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    insertion_features,
)
from scratch.contact_det_closing_pass.scripts.run_start_comparison import (
    _positive_scores,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)

SectionIdentity = tuple[str, int]

FEATURE_NAMES = (
    "strongest_local_score",
    "second_strongest_local_score",
    "mean_local_score",
    "sum_local_score",
    "strongest_timing_score",
    "second_strongest_timing_score",
    "mean_timing_score",
    "sum_timing_score",
    "usable_candidate_count",
    "distinct_gaps_with_candidates",
    "positive_duration_gap_count",
    "gaps_with_no_candidates",
    "excluded_candidate_count",
    "longest_gap_seconds",
    "last_gap_has_candidate",
)


def _summarise_gap_maxima(values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    groups = tuple(values)
    if not groups:
        return (np.nan, np.nan, np.nan, np.nan)
    maxima = np.asarray([max(gap_values) for gap_values in groups], dtype=np.float64)
    ordered = np.sort(maxima)[::-1]
    return (
        float(ordered[0]),
        float(ordered[1]) if len(ordered) > 1 else np.nan,
        float(np.mean(maxima)),
        float(np.sum(maxima)),
    )


def _context(
    selected: LaterOption,
    candidate: FixedEvent,
) -> LaterOption:
    before = selected.span
    after = replace(
        before,
        events=tuple(
            sorted((*before.events, candidate), key=lambda event: event.frame)
        ),
    )
    base = replace(selected.base, span=before)
    return LaterOption(base, candidate, after)


def gap_evidence(
    selected: Mapping[SectionIdentity, LaterOption],
    candidates: Mapping[SectionIdentity, Sequence[FixedEvent]],
    fps: Mapping[str, float],
    measurements: PhysicalMeasurements,
    local_model: Any,
) -> tuple[np.ndarray, tuple[str, ...], tuple[SectionIdentity, ...]]:
    """Summarise saved candidate evidence in each selected section's gaps.

    A candidate is usable when it is within the selected span's half-open
    edges and strictly beyond the scaled duplicate distance from every selected
    contact.  Local model scores are computed only for usable candidates.
    Missing candidates therefore produce missing score summaries, not evidence
    that a gap is complete.

    :param selected: Actual selected later options, in desired output order.
    :param candidates: Saved later candidates keyed by section identity.
    :param fps: Source frame rates keyed by fixture.
    :param measurements: Saved physical feature blocks for candidate frames.
    :param local_model: Frozen insertion model exposing ``classes_`` and
        ``predict_proba``.
    :return: One feature row per selected identity, feature names, and identities.
    """
    identities = tuple(selected)
    sections: list[dict[str, Any]] = []
    contexts: list[LaterOption] = []
    context_owners: list[tuple[int, int]] = []
    for section_index, identity in enumerate(identities):
        option = selected[identity]
        span = option.span
        fixture = span.fixture
        source_fps = float(fps[fixture])
        duplicate_distance = scale_base30_frames(
            DUPLICATE_DISTANCE_AT_30_FPS,
            source_fps,
        )
        contact_frames = np.asarray(
            [event.frame for event in span.events],
            dtype=np.int64,
        )
        boundaries = np.asarray(
            (span.start_frame, *contact_frames.tolist(), span.end_frame),
            dtype=np.int64,
        )
        gap_durations = np.diff(boundaries)
        positive_gaps = np.flatnonzero(gap_durations > 0)
        usable: list[tuple[FixedEvent, int]] = []
        timing_by_gap: dict[int, list[float]] = {}
        excluded_count = 0
        for candidate in candidates.get(identity, ()):
            inside = span.start_frame <= candidate.frame < span.end_frame
            is_duplicate = bool(
                np.any(np.abs(contact_frames - candidate.frame) <= duplicate_distance)
            )
            if not inside or is_duplicate:
                excluded_count += 1
                continue
            gap_index = int(
                np.searchsorted(contact_frames, candidate.frame, side="right")
            )
            usable.append((candidate, gap_index))
            timing_by_gap.setdefault(gap_index, []).append(
                float(candidate.timing_score)
            )
            contexts.append(_context(option, candidate))
            context_owners.append((section_index, gap_index))
        sections.append(
            {
                "gap_durations": gap_durations,
                "positive_gaps": positive_gaps,
                "usable": usable,
                "timing_by_gap": timing_by_gap,
                "local_by_gap": {},
                "excluded_count": excluded_count,
                "source_fps": source_fps,
            }
        )

    if contexts:
        context_matrix, _ = insertion_features(tuple(contexts), fps, measurements)
        local_scores = _positive_scores(local_model, context_matrix)
        if len(local_scores) != len(context_owners):
            raise ValueError("Local gap scores do not align with saved candidates")
        for (section_index, gap_index), local_score in zip(
            context_owners,
            local_scores,
            strict=True,
        ):
            local_by_gap = sections[section_index]["local_by_gap"]
            local_by_gap.setdefault(gap_index, []).append(float(local_score))

    rows: list[tuple[float, ...]] = []
    for section in sections:
        gap_durations = section["gap_durations"]
        positive_gaps = section["positive_gaps"]
        usable = section["usable"]
        local_summary = _summarise_gap_maxima(section["local_by_gap"].values())
        timing_summary = _summarise_gap_maxima(section["timing_by_gap"].values())
        positive_gap_indices = {int(index) for index in positive_gaps}
        usable_gap_count = len(positive_gap_indices & set(section["timing_by_gap"]))
        positive_gap_count = len(positive_gaps)
        gap_with_no_candidates = positive_gap_count - sum(
            gap_index in positive_gap_indices for gap_index in section["timing_by_gap"]
        )
        longest_gap = (
            float(np.max(gap_durations[positive_gaps]) / section["source_fps"])
            if positive_gaps.size
            else 0.0
        )
        last_gap_index = len(gap_durations) - 1
        rows.append(
            (
                *local_summary,
                *timing_summary,
                float(len(usable)),
                float(usable_gap_count),
                float(positive_gap_count),
                float(gap_with_no_candidates),
                float(section["excluded_count"]),
                longest_gap,
                float(
                    last_gap_index in positive_gap_indices
                    and last_gap_index in section["timing_by_gap"]
                ),
            )
        )
    return (
        np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES)),
        FEATURE_NAMES,
        identities,
    )
