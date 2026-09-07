"""Label-guided quality targets for one local later-contact insertion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import overlapping_rallies
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)

SectionIdentity = tuple[str, int]
RallyIdentity = tuple[str, str]


def local_insertion_quality(
    before: FixedSpan,
    after: FixedSpan,
    rally: RallyReference,
    candidate_frame: int,
    tolerance: int,
) -> bool:
    """Return whether an insertion adds one distinct contact without harm.

    Matching is repeated after the insertion because a nearby candidate can
    change the optimal one-to-one pairing.  Player-side answers are irrelevant
    to this local timing target.

    :param before: The section events before inserting the candidate.
    :param after: The section events after inserting the candidate.
    :param rally: The single rally associated with this section.
    :param candidate_frame: Frame of the newly inserted candidate event.
    :param tolerance: Maximum matching distance in source frames.
    :return: Whether the insertion has a useful, non-harmful local match.
    """
    before_frames = [event.frame for event in before.events]
    after_frames = [event.frame for event in after.events]
    if candidate_frame in before_frames or after_frames.count(candidate_frame) != 1:
        return False

    before_matches = match_contacts(rally.frames, before_frames, tolerance)
    after_matches = match_contacts(rally.frames, after_frames, tolerance)
    before_gt = {gt_index for gt_index, _, _ in before_matches}
    after_gt = {gt_index for gt_index, _, _ in after_matches}
    candidate_matches = {
        gt_index
        for gt_index, prediction_index, _ in after_matches
        if after_frames[prediction_index] == candidate_frame
    }
    if not candidate_matches:
        return False
    if not before_gt < after_gt:
        return False

    unmatched_before = len(before_frames) - len(before_matches)
    unmatched_after = len(after_frames) - len(after_matches)
    if unmatched_after > unmatched_before:
        return False

    before_matched_frames = Counter(
        before_frames[prediction_index] for _, prediction_index, _ in before_matches
    )
    after_event_frames = Counter(after_frames)
    return not before_matched_frames - after_event_frames


def _rally_identity(rally: RallyReference) -> RallyIdentity:
    return rally.fixture, rally.rally_id


def insertion_targets(
    options: Sequence[LaterOption],
    original_spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps: Mapping[str, float],
) -> np.ndarray:
    """Assign local timing targets to single-insertion later options.

    The association checks follow ``whole_targets``: a baseline section must
    have one unambiguous retained rally, and the edited section must still
    refer only to that rally.  No-insertion options are intentionally
    unlabelled because this target is only for the insertion.

    :param options: Later options in the order to label; each has at most one insertion.
    :param original_spans: Unchanged sections used for rally association.
    :param labels: Human rally and player-side labels.
    :param fps: Source frame rates keyed by fixture.
    :return: Int8 targets with ``-1`` for unjudgeable or no-insertion options.
    """
    spans_by_section: dict[SectionIdentity, FixedSpan] = {}
    overlap_by_section: dict[SectionIdentity, tuple[RallyReference, ...]] = {}
    touching_counts: Counter[RallyIdentity] = Counter()
    tolerances: dict[str, int] = {}
    overlaps_by_bounds: dict[tuple[str, int, int], set[RallyIdentity]] = {}
    for span in original_spans:
        identity = (span.fixture, span.span_id)
        if identity in spans_by_section:
            raise ValueError(f"{identity}: original section identity repeats")
        spans_by_section[identity] = span
        overlap = overlapping_rallies(span, labels)
        overlap_by_section[identity] = overlap
        overlaps_by_bounds[(span.fixture, span.start_frame, span.end_frame)] = {
            _rally_identity(rally) for rally in overlap
        }
        for rally in overlap:
            touching_counts[_rally_identity(rally)] += 1
        if span.fixture not in tolerances:
            tolerances[span.fixture] = scale_base30_frames(10, fps[span.fixture])

    targets: list[int] = []
    for option in options:
        if option.inserted is None:
            targets.append(-1)
            continue

        identity = (option.base.span.fixture, option.base.span.span_id)
        if identity not in spans_by_section:
            targets.append(-1)
            continue
        overlaps = overlap_by_section[identity]
        if len(overlaps) != 1:
            targets.append(-1)
            continue
        rally = overlaps[0]
        rally_identity = _rally_identity(rally)
        if touching_counts[rally_identity] != 1:
            targets.append(-1)
            continue

        # Alternatives share section edges even when their contact lists differ.
        bounds = (option.span.fixture, option.span.start_frame, option.span.end_frame)
        if bounds not in overlaps_by_bounds:
            overlaps_by_bounds[bounds] = {
                _rally_identity(expanded) for expanded in overlapping_rallies(option.span, labels)
            }
        expanded_ids = overlaps_by_bounds[bounds]
        if expanded_ids != {rally_identity}:
            targets.append(-1)
            continue

        tolerance = tolerances[option.base.span.fixture]
        targets.append(
            int(
                local_insertion_quality(
                    option.base.span,
                    option.span,
                    rally,
                    option.inserted.frame,
                    tolerance,
                )
            )
        )

    return np.asarray(targets, dtype=np.int8)
