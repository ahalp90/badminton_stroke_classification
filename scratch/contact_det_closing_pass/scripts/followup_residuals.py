"""Describe labelled contact residuals in selected later-contact spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)

SectionIdentity = tuple[str, int]


def _missing_contact_rows(
    span: LaterOption,
    rally_frames: Sequence[int],
    matches: Sequence[tuple[int, int, int]],
    candidates: Sequence[FixedEvent],
    tolerance: int,
) -> list[dict[str, Any]]:
    """Describe each unmatched ground-truth contact and nearby predictions."""
    matched_gt_indices = {gt_index for gt_index, _prediction_index, _offset in matches}
    predicted_frames = [event.frame for event in span.span.events]
    rows = []
    for gt_index, gt_frame in enumerate(rally_frames):
        if gt_index in matched_gt_indices:
            continue
        candidate_frames = sorted(
            event.frame
            for event in candidates
            if abs(event.frame - gt_frame) <= tolerance
        )
        rows.append({
            "gt_index": gt_index,
            "gt_frame": gt_frame,
            "kind": "first" if gt_index == 0 else "later",
            "any_saved_candidate_within_tolerance": bool(candidate_frames),
            "candidate_frames_within_tolerance": candidate_frames,
            "existing_predicted_frame_within_tolerance": any(
                abs(predicted_frame - gt_frame) <= tolerance
                for predicted_frame in predicted_frames
            ),
        })
    return rows


def residual_rows(
    selected: Mapping[SectionIdentity, LaterOption],
    candidates: Mapping[SectionIdentity, Sequence[FixedEvent]],
    labels: HumanLabels,
    fps: Mapping[str, float],
    tolerance_base30: int = 10,
) -> list[dict[str, Any]]:
    """Return diagnostic contact residuals for each selected section.

    Sections overlapping zero or multiple labelled rallies retain ``None`` for
    rally-dependent fields.  Diagnostics do not affect routing or selection.

    :param selected: Selected later-contact option by section identity.
    :param candidates: Saved later candidates by section identity.
    :param labels: Human rally and side labels used only for diagnostics.
    :param fps: Fixture frame rates.
    :param tolerance_base30: Matching tolerance expressed on a 30 fps clock.
    :return: One diagnostic row per selected section.
    """
    tolerance_by_fixture: dict[str, int] = {}
    for option in selected.values():
        fixture = option.span.fixture
        if fixture not in tolerance_by_fixture:
            tolerance_by_fixture[fixture] = scale_base30_frames(
                tolerance_base30, fps[fixture]
            )

    rows = []
    for identity, option in selected.items():
        fixture = option.span.fixture
        rallies = overlapping_rallies(option.span, labels)
        result = section_result(option.span, labels, tolerance_by_fixture[fixture])
        one_rally = len(rallies) == 1
        no_labels = not rallies
        multiple_rallies = len(rallies) > 1
        if one_rally:
            contained = bool(result["whole_rally_contained"])
            matches = result["matches"]
            rally_frames = rallies[0].frames
            matched_gt_indices = {gt_index for gt_index, _prediction_index, _offset in matches}
            missing_first = int(0 not in matched_gt_indices)
            missing_later_count = sum(
                index > 0 and index not in matched_gt_indices
                for index in range(len(rally_frames))
            )
            unmatched_predictions = len(option.span.events) - len(matches)
            wrong_voted_side_count = len(matches) - int(result["voted_correct_sides"])
            unmatched = _missing_contact_rows(
                option, rally_frames, matches, candidates.get(identity, ()),
                tolerance_by_fixture[fixture],
            )
            timing_complete = bool(result["timing_complete"])
            side_rule_fully_correct = bool(result["side_rule_fully_correct"])
        else:
            contained = None
            missing_first = None
            missing_later_count = None
            unmatched_predictions = None
            wrong_voted_side_count = None
            unmatched = None
            timing_complete = None
            side_rule_fully_correct = None
        rows.append({
            "fixture": identity[0],
            "span_id": identity[1],
            "known": one_rally,
            "full_contained": contained,
            "timing_complete": timing_complete,
            "side_rule_fully_correct": side_rule_fully_correct,
            "missing_first": missing_first,
            "missing_later_count": missing_later_count,
            "unmatched_predictions": unmatched_predictions,
            "wrong_voted_side_count": wrong_voted_side_count,
            "boundary_incomplete": bool(one_rally and not contained),
            "multiple_rallies": multiple_rallies,
            "no_labels": no_labels,
            "unmatched_gt_contacts": unmatched,
        })
    return rows
