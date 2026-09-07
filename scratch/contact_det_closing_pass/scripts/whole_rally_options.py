"""Build and score bounded whole-rally contact-edit options."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts import evaluation
from scratch.contact_det_followup.scripts.audit_combined_best_case import (
    ACTION_KINDS,
    CombinedAction,
    _action_priority,
    section_actions,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)

SectionIdentity: TypeAlias = tuple[str, int]
OptionIdentity: TypeAlias = tuple[str, int, str, int | None, int | None]


def option_identity(option: CombinedAction) -> OptionIdentity:
    """Return the stable identity of one combined edit option."""
    return (
        option.span.fixture,
        option.span.span_id,
        option.kind,
        option.candidate_frame,
        option.deleted_frame,
    )


def _candidate_lists_by_section(
    raw_videos: Sequence[Mapping[str, object]],
) -> dict[SectionIdentity, Mapping[str, object]]:
    """Flatten saved candidate lists while preserving their stored records."""
    lists_by_section: dict[SectionIdentity, Mapping[str, object]] = {}
    for video in raw_videos:
        raw_lists = video.get("candidate_lists")
        if raw_lists is None:
            continue
        if not isinstance(raw_lists, list):
            raise TypeError("candidate_lists must be a list")
        for raw_list in raw_lists:
            if not isinstance(raw_list, Mapping):
                raise TypeError("candidate list must be an object")
            fixture = raw_list.get("fixture")
            span_id = raw_list.get("span_id")
            if not isinstance(fixture, str) or type(span_id) is not int:
                raise ValueError("candidate list identity is malformed")
            identity = (fixture, span_id)
            if identity in lists_by_section:
                raise ValueError(f"{identity}: candidate list repeats")
            lists_by_section[identity] = raw_list
    return lists_by_section


def build_options(
    spans: Sequence[FixedSpan],
    raw_videos: Sequence[Mapping[str, object]],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[SectionIdentity, tuple[CombinedAction, ...]]:
    """Build every allowed option for every baseline span.

    Candidate lists are label-free saved inputs.  Sections without a saved list
    still receive their keep option.  The previous section end is reset for
    each fixture and is reused by ``section_actions`` to reject overlaps.

    :param spans: Baseline spans in their stored chronological order.
    :param raw_videos: Saved video records containing candidate lists.
    :param events_by_fixture: Fixed event streams keyed by fixture.
    :return: One complete option pool per baseline section.
    """
    lists_by_section = _candidate_lists_by_section(raw_videos)
    options_by_section: dict[SectionIdentity, tuple[CombinedAction, ...]] = {}
    previous_end_by_fixture: dict[str, int] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if identity in options_by_section:
            raise ValueError(f"{identity}: section identity repeats")
        try:
            fixture_events = events_by_fixture[span.fixture]
        except KeyError as error:
            raise KeyError(f"{span.fixture}: event stream is missing") from error
        options_by_section[identity] = section_actions(
            span,
            lists_by_section.get(identity),
            fixture_events,
            previous_end_by_fixture.get(span.fixture, -1),
        )
        previous_end_by_fixture[span.fixture] = span.end_frame
    return options_by_section


def _rally_identity(rally: RallyReference) -> tuple[str, str]:
    """Return a fixture-qualified rally identity for overlap accounting."""
    return rally.fixture, rally.rally_id


def _section_identity(option: CombinedAction) -> SectionIdentity:
    return option.span.fixture, option.span.span_id


def _all_sides_known(rally: RallyReference, labels: HumanLabels) -> bool:
    """Return whether every contact in a rally has a human-side answer."""
    return all(
        (rally.fixture, frame) in labels.target_sides
        and labels.target_sides[(rally.fixture, frame)] is not None
        for frame in rally.frames
    )


def whole_targets(
    options: Sequence[CombinedAction],
    baseline_spans: Sequence[FixedSpan],
    labels: HumanLabels,
    fps: Mapping[str, float],
    tolerance_base30: int = 10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign label-guided whole-rally targets to a flat option sequence.

    An option receives ``-1`` when its baseline section cannot be judged from
    one unambiguous, fully side-labelled rally.  Eligible options receive ``1``
    when the fixed side vote makes the revised section fully correct and ``0``
    otherwise.  A rally that is already complete therefore keeps a positive
    keep target.

    :param options: Flat options in the same order as the returned targets.
    :param baseline_spans: All unchanged spans used to establish eligibility.
    :param labels: Human rally and player-side labels.
    :param fps: Source frame rate keyed by fixture.
    :param tolerance_base30: Matching tolerance expressed on a 30 fps clock.
    :return: Int8 targets and compact counts of actions, exclusions and positives.
    """
    spans_by_section: dict[SectionIdentity, FixedSpan] = {}
    for span in baseline_spans:
        identity = (span.fixture, span.span_id)
        if identity in spans_by_section:
            raise ValueError(f"{identity}: baseline section identity repeats")
        spans_by_section[identity] = span

    overlap_by_section: dict[SectionIdentity, tuple[RallyReference, ...]] = {}
    overlaps_by_bounds: dict[tuple[str, int, int], tuple[RallyReference, ...]] = {}
    touching_counts: Counter[tuple[str, str]] = Counter()
    for span in baseline_spans:
        identity = (span.fixture, span.span_id)
        overlaps = evaluation.overlapping_rallies(span, labels)
        overlap_by_section[identity] = overlaps
        overlaps_by_bounds[(span.fixture, span.start_frame, span.end_frame)] = overlaps
        for rally in overlaps:
            touching_counts[_rally_identity(rally)] += 1

    baseline_reason: dict[SectionIdentity, str] = {}
    baseline_rally: dict[SectionIdentity, tuple[str, str]] = {}
    for identity, overlaps in overlap_by_section.items():
        if len(overlaps) == 0:
            baseline_reason[identity] = "no_labelled_rally"
            continue
        if len(overlaps) > 1:
            baseline_reason[identity] = "multiple_labelled_rallies"
            continue
        rally_identity = _rally_identity(overlaps[0])
        if touching_counts[rally_identity] != 1:
            baseline_reason[identity] = "labelled_rally_touches_multiple_sections"
            continue
        if not _all_sides_known(overlaps[0], labels):
            baseline_reason[identity] = "missing_labelled_sides"
            continue
        baseline_rally[identity] = rally_identity
        baseline_reason[identity] = "eligible"

    tolerances: dict[str, int] = {}
    for option in options:
        fixture = option.span.fixture
        if fixture not in tolerances:
            tolerances[fixture] = scale_base30_frames(tolerance_base30, fps[fixture])

    targets: list[int] = []
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for option in options:
        action_counts[option.kind] += 1
        identity = _section_identity(option)
        reason = baseline_reason.get(identity, "section_not_in_baseline")
        if reason == "eligible":
            expected_rally = baseline_rally[identity]
            bounds = (option.span.fixture, option.span.start_frame, option.span.end_frame)
            if bounds not in overlaps_by_bounds:
                overlaps_by_bounds[bounds] = evaluation.overlapping_rallies(option.span, labels)
            expanded = overlaps_by_bounds[bounds]
            expanded_ids = {_rally_identity(rally) for rally in expanded}
            if expected_rally not in expanded_ids or expanded_ids - {expected_rally}:
                reason = "expanded_section_has_other_labels"
        if reason != "eligible":
            targets.append(-1)
            reason_counts[reason] += 1
            continue

        rally = overlap_by_section[identity][0]
        predicted_frames = sorted(event.frame for event in option.span.events)
        tolerance = tolerances[option.span.fixture]
        target = 0
        # A complete one-to-one timing match must pair equal-length sorted lists.
        # Most edit alternatives fail this cheap condition before side scoring.
        if len(predicted_frames) == len(rally.frames) and all(
            abs(predicted - labelled) <= tolerance
            for predicted, labelled in zip(predicted_frames, sorted(rally.frames), strict=True)
        ):
            result = evaluation.section_result(option.span, labels, tolerance)
            target = int(bool(result["side_rule_fully_correct"]))
        targets.append(target)
        reason_counts[reason] += 1
        if target == 1:
            positive_counts[option.kind] += 1

    report: dict[str, object] = {
        "action_counts": {kind: action_counts[kind] for kind in ACTION_KINDS},
        "reasons": dict(reason_counts),
        "positive_counts": {kind: positive_counts[kind] for kind in ACTION_KINDS},
    }
    return np.asarray(targets, dtype=np.int8), report


def choose_options(
    options: Sequence[CombinedAction],
    scores: np.ndarray,
    minimum_score: float,
) -> dict[SectionIdentity, CombinedAction]:
    """Choose one scored edit per section without reading labels.

    Keep is the fallback for every section.  A non-keep option must meet the
    minimum score and strictly exceed keep.  Equal score candidates use the
    existing combined-action priority order.

    :param options: Flat options grouped by their section identity.
    :param scores: One score for each option, in the same order as ``options``.
    :param minimum_score: Minimum score required before applying an edit.
    :return: One selected option per section.
    """
    score_array = np.asarray(scores)
    if score_array.ndim != 1 or len(score_array) != len(options):
        raise ValueError("option score coverage differs")

    grouped: dict[SectionIdentity, list[tuple[CombinedAction, float]]] = {}
    for option, score in zip(options, score_array, strict=True):
        value = float(score)
        grouped.setdefault(_section_identity(option), []).append((option, value))

    selected: dict[SectionIdentity, CombinedAction] = {}
    for identity, section_options in grouped.items():
        keeps = [option_score for option_score in section_options if option_score[0].kind == "keep"]
        if len(keeps) != 1:
            raise ValueError(f"{identity}: exactly one keep option is required")
        keep, keep_score = keeps[0]
        candidates = [
            (option, score)
            for option, score in section_options
            if option.kind != "keep"
            and np.isfinite(score)
            and score >= minimum_score
            and score > keep_score
        ]
        if not candidates:
            selected[identity] = keep
            continue
        selected[identity] = min(
            candidates,
            key=lambda option_score: (-option_score[1], _action_priority(option_score[0])),
        )[0]
    return selected


__all__ = [
    "CombinedAction",
    "OptionIdentity",
    "SectionIdentity",
    "build_options",
    "choose_options",
    "option_identity",
    "whole_targets",
]
