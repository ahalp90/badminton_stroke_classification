"""Local deletions preserve represented hits after one-to-one re-matching."""

from dataclasses import replace

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.local_deletion import (
    deletion_effect,
    deletion_opportunities,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels


def _span(frames: tuple[int, ...]) -> FixedSpan:
    return FixedSpan("video", 0, 0, 250, tuple(FixedEvent("video", frame, 0.9, "Top") for frame in frames))


def test_distinct_receiver_is_protected() -> None:
    before = _span((100, 150, 200))
    rally = RallyReference("video", 0, "1", (100, 150, 200))
    result = deletion_effect(before, replace(before, events=before.events[:1] + before.events[2:]), rally, 10)
    assert result["lost_gt_indices"] == [1]
    assert not result["useful"]


def test_old_matched_duplicate_can_be_removed_with_another_miss() -> None:
    before = _span((100, 102, 200))
    rally = RallyReference("video", 0, "1", (100, 150, 200))
    result = deletion_effect(before, replace(before, events=before.events[1:]), rally, 10)
    assert result["useful"]
    assert result["other_missing_hits"]
    assert result["lost_gt_indices"] == []


def test_unlabelled_lead_is_unknown_but_supported_extra_is_useful() -> None:
    before = _span((30, 100, 125, 200))
    rally = RallyReference("video", 0, "1", (100, 200))
    labels = HumanLabels({"video": (rally,)}, {("video", 100): "Top", ("video", 200): "Bot"})
    result = deletion_opportunities((before,), labels, {"video": 30.0}, {}, 10)
    targets = {row["frame"]: row["target"] for row in result["rows"]}
    assert targets == {30: -1, 100: 0, 125: 1, 200: 0}
    missing = deletion_opportunities((before,), HumanLabels({}, {}), {"video": 30.0}, {}, 10)
    assert all(row["target"] == -1 for row in missing["rows"])
