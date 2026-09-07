"""Describe possible missing contacts without changing the proposed rally output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
    _side_features,
)
from scratch.contact_det_followup.scripts.score_keep_review import (
    FEATURE_NAMES,
    build_feature_vector,
)
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    scale_base30_frames,
)


def acceptance_features(
    options: Sequence[LaterOption], scores: np.ndarray,
    selected: Mapping[tuple[str, int], LaterOption],
    candidates: Mapping[tuple[str, int], Sequence[FixedEvent]],
    fps: Mapping[str, float], measurements: PhysicalMeasurements,
) -> tuple[np.ndarray, tuple[str, ...], list[tuple[str, int]]]:
    """Join selected scores with discarded-gap support and original side evidence.

    All physical inputs come from saved automatic measurements. No candidate is
    selected by labels, and an acceptance decision never removes a saved output.
    """
    names = (
        "selected_score", "advantage_over_next_output", "best_insertion_output_score",
        "best_different_start_output_score", "discarded_candidate_count", "strongest_discarded_score",
        "strongest_discarded_left_gap_seconds", "strongest_discarded_right_gap_seconds",
        "best_discarded_side_vote_improvement", "raw_side_fraction_known", "raw_top_vote", "raw_bot_vote",
        "adjacent_same_raw_side", *(f"output__{name}" for name in FEATURE_NAMES),
        *(f"strongest_discarded__{name}" for name in measurements.names),
    )
    grouped: dict[tuple[str, int], list[tuple[LaterOption, float]]] = {}
    for option, score in zip(options, scores, strict=True):
        grouped.setdefault(option.base.identity, []).append((option, float(score)))
    rows = []
    identities = list(selected)
    for identity in identities:
        chosen = selected[identity]
        section_options = grouped[identity]
        selected_score = next(score for option, score in section_options if option == chosen)
        other_scores = [score for option, score in section_options if option.span != chosen.span]
        insertion_scores = [score for option, score in section_options if option.inserted is not None]
        different_start_scores = [
            score for option, score in section_options
            if option.base.candidate_frame is not None and option.base.candidate_frame != chosen.base.candidate_frame
        ]
        span = chosen.span
        source_fps = fps[identity[0]]
        distance = scale_base30_frames(DUPLICATE_DISTANCE_AT_30_FPS, source_fps)
        discarded = []
        for candidate in candidates.get(identity, ()):
            if all(abs(candidate.frame - event.frame) > distance for event in span.events):
                discarded.append(candidate)
        strongest = max(discarded, key=lambda candidate: candidate.timing_score, default=None)
        raw_sides = _side_features(span)
        original_agreement = max(raw_sides[1:3])
        improvements = []
        for candidate in discarded:
            revised = replace(span, events=tuple(sorted((*span.events, candidate), key=lambda event: event.frame)))
            changed_sides = _side_features(revised)
            improvements.append(max(changed_sides[1:3]) - original_agreement)
        left_gap = right_gap = np.nan
        block = np.full(len(measurements.names), np.nan)
        if strongest is not None:
            left = [event.frame for event in span.events if event.frame < strongest.frame]
            right = [event.frame for event in span.events if event.frame > strongest.frame]
            left_gap = (strongest.frame - left[-1]) / source_fps if left else np.nan
            right_gap = (right[0] - strongest.frame) / source_fps if right else np.nan
            block = measurements.values[(identity[0], strongest.frame)]
        rows.append([
            selected_score, selected_score - max(other_scores) if other_scores else np.nan,
            max(insertion_scores, default=np.nan), max(different_start_scores, default=np.nan),
            float(len(discarded)), np.nan if strongest is None else strongest.timing_score,
            left_gap, right_gap, max(improvements, default=np.nan), *raw_sides,
            *build_feature_vector(span, source_fps), *block,
        ])
    return np.asarray(rows, dtype=np.float64), names, identities
