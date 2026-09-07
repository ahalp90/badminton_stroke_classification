"""Multi-positive opening and whole-rally training answers from existing labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from annotator.fps_constants import ScalingKind
from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
)
from scratch.contact_det_closing_pass.scripts.matching import match_contacts
from scratch.contact_det_followup.scripts.score_start_model import (
    ActionRow,
    _action_span,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


@dataclass(frozen=True)
class EditTarget:
    included: bool
    reason: str
    opening_correct: bool
    whole_rally_correct: bool
    unnecessary_added: int
    real_contacts_removed: int


def local_quality(
    before: FixedSpan, after: FixedSpan, rally: RallyReference, candidate_frame: int,
    fixed_frame: int, action: str, tolerance: int,
) -> tuple[bool, int, int]:
    """Judge the opening independently of unrelated errors later in the rally."""
    before_frames = [event.frame for event in before.events]
    after_frames = [event.frame for event in after.events]
    before_matches = match_contacts(rally.frames, before_frames, tolerance)
    after_matches = match_contacts(rally.frames, after_frames, tolerance)
    before_gt = {gt_index for gt_index, _, _ in before_matches}
    after_gt = {gt_index for gt_index, _, _ in after_matches}
    before_real = {before_frames[pred_index] for _, pred_index, _ in before_matches}
    # A duplicate may steal a match from an old prediction, so count the change
    # in unmatched events rather than only unmatched newly inserted frames.
    unmatched_before = len(before_frames) - len(before_matches)
    unmatched_after = len(after_frames) - len(after_matches)
    removed_unmatched = len(set(before_frames) - set(after_frames) - before_real)
    unnecessary = max(0, unmatched_after - unmatched_before + removed_unmatched)
    removed = len(before_real - set(after_frames))
    candidate_is_first = any(
        gt_index == 0 and after_frames[pred_index] == candidate_frame
        for gt_index, pred_index, _ in after_matches
    )
    opening_correct = (
        0 not in before_gt and candidate_is_first and before_gt <= after_gt
        and unnecessary == 0 and removed == 0
        and (action == "add" or fixed_frame not in before_real)
    )
    return opening_correct, unnecessary, removed


def assign_targets(
    rows: Sequence[ActionRow], spans: Sequence[FixedSpan],
    events: Mapping[str, Sequence[FixedEvent]], labels: HumanLabels,
    fps: Mapping[str, float], tolerance_base30: int = 10,
) -> tuple[dict[tuple[str, int, int, str], EditTarget], dict[tuple[str, int, int, str], FixedSpan | None]]:
    """Keep every acceptable action positive and exclude insufficient labels."""
    span_lookup = {(span.fixture, span.span_id): span for span in spans}
    previous_end = {}
    last_end = {}
    rally_sections = Counter()
    overlaps = {}
    baseline_timing = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        previous_end[identity] = last_end.get(span.fixture, -1)
        last_end[span.fixture] = span.end_frame
        overlaps[identity] = overlapping_rallies(span, labels)
        tolerance = ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps[span.fixture])
        baseline_timing[identity] = section_result(span, labels, tolerance)["timing_complete"]
        for rally in overlaps[identity]:
            rally_sections[(span.fixture, rally.rally_id)] += 1
    targets = {}
    revised_spans = {}
    for row in rows:
        before = span_lookup[row.section_identity]
        after = _action_span(before, row.candidate, row.action, events[before.fixture], previous_end[row.section_identity])
        revised_spans[row.identity] = after
        original_rallies = overlaps[row.section_identity]
        reason = "eligible"
        if after is None:
            reason = "predecessor_overlap"
        elif len(original_rallies) != 1:
            reason = "no_labelled_rally" if not original_rallies else "multiple_labelled_rallies"
        elif rally_sections[(before.fixture, original_rallies[0].rally_id)] > 1:
            reason = "labelled_rally_touches_multiple_sections"
        elif overlapping_rallies(after, labels) != original_rallies:
            reason = "expanded_section_has_other_labels"
        if reason != "eligible":
            targets[row.identity] = EditTarget(False, reason, False, False, 0, 0)
            continue
        tolerance = ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps[before.fixture])
        local, added, removed = local_quality(
            before, after, original_rallies[0], row.candidate.frame,
            row.candidate.fixed_contact_frame, row.action, tolerance,
        )
        whole = not baseline_timing[row.section_identity] and section_result(after, labels, tolerance)["timing_complete"]
        targets[row.identity] = EditTarget(True, reason, local, whole, added, removed)
    return targets, revised_spans
