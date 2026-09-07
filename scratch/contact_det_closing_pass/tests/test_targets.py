"""Opening repair targets must preserve real contacts and allow later mistakes."""
from __future__ import annotations

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.targets import (
    assign_targets,
    local_quality,
)
from scratch.contact_det_followup.scripts.score_start_model import ActionRow
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    CandidateRow,
    HumanLabels,
)


def span(*frames: int, start: int = 80) -> FixedSpan:
    return FixedSpan('sset_01', 0, start, 200, tuple(FixedEvent('sset_01', frame, 0.9, 'Top') for frame in frames))


def action(frame: int, kind: str = 'add', fixed: int = 130) -> ActionRow:
    candidate = CandidateRow('sset_01', 'A', 30, 0, 80, 200, 70, fixed, frame, 0.8, 0.9,
                             False, 'Top', 'Bot', (0.0,) * 9)
    return ActionRow(candidate, kind, (*candidate.features, float(kind == 'replace')))


def labels() -> HumanLabels:
    rally = RallyReference('sset_01', 0, 'rally-1', (100, 130, 160))
    return HumanLabels({'sset_01': (rally,)}, {('sset_01', frame): 'Top' for frame in rally.frames})


def test_local_add_is_correct_despite_later_miss_and_both_candidates_positive() -> None:
    before = span(130)
    rows = [action(96), action(104)]
    targets, _ = assign_targets(rows, [before], {'sset_01': before.events}, labels(), {'sset_01': 30})
    assert all(target.included and target.opening_correct for target in targets.values())
    assert not any(target.whole_rally_correct for target in targets.values())


def test_replace_cannot_remove_receiver_and_can_remove_false_first_event() -> None:
    rally = labels().rallies['sset_01'][0]
    assert local_quality(span(130, 160), span(100, 160), rally, 100, 130, 'replace', 10) == (False, 0, 1)
    assert local_quality(span(115, 130), span(100, 130), rally, 100, 115, 'replace', 10) == (True, 0, 0)
    assert local_quality(span(115, 130), span(85, 130), rally, 85, 115, 'replace', 10) == (False, 1, 0)


def test_duplicate_serve_and_unmatched_add_are_harmful() -> None:
    rally = labels().rallies['sset_01'][0]
    assert local_quality(span(104, 130), span(100, 104, 130), rally, 100, 104, 'add', 10) == (False, 1, 0)
    assert local_quality(span(130), span(80, 130), rally, 80, 130, 'add', 10) == (False, 1, 0)


def test_absent_labels_are_excluded_and_whole_rally_needs_contained_last_hit() -> None:
    before = span(130, 160)
    rows = [action(100)]
    missing = HumanLabels({'sset_01': ()}, {})
    targets, _ = assign_targets(rows, [before], {'sset_01': before.events}, missing, {'sset_01': 30})
    assert not targets[rows[0].identity].included
    targets, _ = assign_targets(rows, [before], {'sset_01': before.events}, labels(), {'sset_01': 30})
    assert targets[rows[0].identity].whole_rally_correct


def test_whole_target_keeps_the_historical_requirement_to_repair_a_wrong_list() -> None:
    before = span(104, 130, 160)
    row = action(100, 'replace', fixed=104)
    targets, _ = assign_targets([row], [before], {'sset_01': before.events}, labels(), {'sset_01': 30})
    target = targets[row.identity]
    assert target.included
    assert not target.whole_rally_correct
    assert not target.opening_correct
