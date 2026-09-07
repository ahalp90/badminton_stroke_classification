"""Tests for bounded local deletion evidence."""

import numpy as np
import pytest

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.deletion_evidence import (
    deletion_column,
    deletion_inputs,
    deletion_targets,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import HumanLabels

FIXTURE = "fixture"


def _event(frame: int, side: str | None = "Top", score: float = 0.8) -> FixedEvent:
    return FixedEvent(FIXTURE, frame, score, side)


def _span(*frames: int, span_id: int = 0) -> FixedSpan:
    return FixedSpan(
        FIXTURE,
        span_id,
        0,
        300,
        tuple(_event(frame) for frame in frames),
    )


def _delete(after: FixedSpan, deleted_frame: int, kind: str = "delete") -> LaterOption:
    base = CombinedAction(kind, None, deleted_frame, after)
    return LaterOption(base, None, after)


def _labels(*frames: int, rally_id: str = "set1:1") -> HumanLabels:
    rally = RallyReference(FIXTURE, 0, rally_id, tuple(frames))
    return HumanLabels(
        {FIXTURE: (rally,)},
        {(FIXTURE, frame): "Top" for frame in frames},
    )


def _measurements(frames: list[int]) -> PhysicalMeasurements:
    return PhysicalMeasurements(
        ("physical",),
        {(FIXTURE, frame): np.asarray([float(frame)]) for frame in frames},
        {},
    )


def _inputs(
    after: FixedSpan,
    deleted_frame: int,
    original_frames: tuple[int, ...],
) -> tuple[tuple[LaterOption, ...], dict[str, object]]:
    option = _delete(after, deleted_frame)
    result = deletion_inputs(
        (option,),
        {FIXTURE: tuple(_event(frame) for frame in original_frames)},
        {FIXTURE: 30.0},
        _measurements(list(original_frames)),
    )
    return (option,), result


def test_inverse_context_reuses_removed_event_and_insertion_features() -> None:
    after = _span(100, 200)
    options, result = _inputs(after, 150, (100, 150, 200))

    context = result["contexts"][0]
    assert options[0].base.deleted_frame == 150
    assert context.base.span == after
    assert [event.frame for event in context.span.events] == [100, 150, 200]
    assert context.inserted == _event(150)
    assert result["option_context_indices"].dtype == np.int64
    assert result["option_context_indices"].tolist() == [0]
    assert result["features"].shape == (1, len(result["feature_names"]))
    values = dict(zip(result["feature_names"], result["features"][0], strict=True))
    assert np.isclose(values["left_gap_seconds"], 50 / 30)
    assert np.isclose(values["right_gap_seconds"], 50 / 30)
    assert values["later__physical"] == 150


def test_duplicate_contexts_are_computed_once_and_keep_option_alignment() -> None:
    after = _span(102, 200)
    first = _delete(after, 100)
    second = _delete(after, 100, kind="replace_delete")
    result = deletion_inputs(
        (first, second),
        {FIXTURE: tuple(_event(frame) for frame in (100, 102, 200))},
        {FIXTURE: 30.0},
        _measurements([100, 102, 200]),
    )

    assert len(result["contexts"]) == 1
    assert result["option_context_indices"].tolist() == [0, 0]
    assert result["features"].shape[0] == 1


def test_deletion_targets_cover_receiver_duplicate_extra_and_unsupported_lead() -> None:
    receiver_options, receiver = _inputs(_span(100, 200), 150, (100, 150, 200))
    np.testing.assert_array_equal(
        deletion_targets(receiver["contexts"], (_span(100, 200),), _labels(100, 150, 200), {FIXTURE: 30.0}),
        np.array([0], dtype=np.int8),
    )

    duplicate_options, duplicate = _inputs(_span(102, 200), 100, (100, 102, 200))
    duplicate_target = deletion_targets(
        duplicate["contexts"], (_span(102, 200),), _labels(100, 150, 200), {FIXTURE: 30.0}
    )
    np.testing.assert_array_equal(duplicate_target, np.array([1], dtype=np.int8))
    assert receiver_options[0].base.deleted_frame == 150
    assert duplicate_options[0].base.deleted_frame == 100

    _extra_options, extra = _inputs(_span(30, 100, 200), 125, (30, 100, 125, 200))
    np.testing.assert_array_equal(
        deletion_targets(extra["contexts"], (_span(30, 100, 125, 200),), _labels(100, 200), {FIXTURE: 30.0}),
        np.array([1], dtype=np.int8),
    )

    lead_options, lead = _inputs(_span(100, 200), 30, (30, 100, 200))
    np.testing.assert_array_equal(
        deletion_targets(lead["contexts"], (_span(100, 200),), _labels(100, 200), {FIXTURE: 30.0}),
        np.array([-1], dtype=np.int8),
    )
    assert lead_options[0].base.deleted_frame == 30


def test_missing_deleted_metadata_fails_loudly() -> None:
    option = _delete(_span(100), 150)
    with pytest.raises(KeyError, match="original event"):
        deletion_inputs(
            (option,),
            {FIXTURE: (_event(100),)},
            {FIXTURE: 30.0},
            _measurements([100, 150]),
        )


def test_no_deletion_gets_no_context_and_nan_column() -> None:
    span = _span(100)
    keep = LaterOption(CombinedAction("keep", None, None, span), None, span)
    result = deletion_inputs(
        (keep,),
        {FIXTURE: (span.events[0],)},
        {FIXTURE: 30.0},
        _measurements([100]),
    )

    assert result["contexts"] == ()
    assert result["option_context_indices"].tolist() == [-1]
    assert result["features"].shape == (0, len(result["feature_names"]))
    column = deletion_column(result["option_context_indices"], np.empty(0))
    assert column.shape == (1, 1)
    assert np.isnan(column[0, 0])


def test_deletion_column_expands_deduplicated_scores() -> None:
    indices = np.asarray([1, -1, 0], dtype=np.int64)
    column = deletion_column(indices, np.asarray([0.25, 0.75]))
    np.testing.assert_allclose(column[[0, 2], 0], [0.75, 0.25])
    assert np.isnan(column[1, 0])
