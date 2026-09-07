"""Check label-free local evidence summaries for selected contact gaps."""

import numpy as np
import pytest

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.gap_evidence import gap_evidence
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction

FIXTURE = "fixture"
IDENTITY = (FIXTURE, 0)


def _event(frame: int, score: float = 0.8) -> FixedEvent:
    return FixedEvent(FIXTURE, frame, score, None)


def _option(
    *frames: int, start: int = 0, end: int = 100, span_id: int = 0
) -> LaterOption:
    span = FixedSpan(
        FIXTURE, span_id, start, end, tuple(_event(frame) for frame in frames)
    )
    base = CombinedAction("keep", None, None, span)
    return LaterOption(base, None, span)


def _measurements(frames: list[int]) -> PhysicalMeasurements:
    return PhysicalMeasurements(
        ("measurement",),
        {(FIXTURE, frame): np.asarray([float(frame)]) for frame in frames},
        {},
    )


class _TimingModel:
    classes_: np.ndarray = np.asarray([0, 1])

    def __init__(self) -> None:
        self.calls = 0
        self.values: list[np.ndarray] = []

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.values.append(values.copy())
        scores = values[:, 1]
        return np.column_stack((1.0 - scores, scores))


def _values(matrix: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    return dict(zip(names, matrix[0], strict=True))


def test_pair_selected_context_uses_selected_neighbours() -> None:
    original = FixedSpan(FIXTURE, 0, 0, 100, (_event(20), _event(80)))
    selected_span = FixedSpan(
        FIXTURE,
        0,
        0,
        100,
        (_event(20), _event(40), _event(60), _event(80)),
    )
    first, second = _event(40), _event(60)
    selected = LaterOption(
        CombinedAction("keep", None, None, original),
        first,
        selected_span,
        second,
    )
    model = _TimingModel()
    matrix, names, identities = gap_evidence(
        {IDENTITY: selected},
        {IDENTITY: (first, second, _event(70, 0.7))},
        {FIXTURE: 30},
        _measurements([70]),
        model,
    )
    values = _values(matrix, names)

    assert identities == (IDENTITY,)
    assert model.calls == 1
    assert np.isclose(model.values[0][0, 2], 1 / 3)
    assert np.isclose(model.values[0][0, 3], 1 / 3)
    assert values["usable_candidate_count"] == 1
    assert values["excluded_candidate_count"] == 2


def test_all_selected_sections_share_one_local_model_call() -> None:
    first = _option(20, 80)
    second = _option(30, 70, span_id=1)
    model = _TimingModel()
    matrix, _, identities = gap_evidence(
        {IDENTITY: first, (FIXTURE, 1): second},
        {IDENTITY: (_event(50, 0.7),), (FIXTURE, 1): (_event(50, 0.6),)},
        {FIXTURE: 30},
        _measurements([50]),
        model,
    )

    assert model.calls == 1
    assert model.values[0].shape[0] == 2
    assert identities == (IDENTITY, (FIXTURE, 1))
    assert matrix.shape == (2, 15)


def test_gap_maxima_are_grouped_and_final_gap_is_counted() -> None:
    selected = _option(20, 50, 80)
    candidates = tuple(
        _event(frame, score)
        for frame, score in (
            (30, 0.2),
            (40, 0.8),
            (60, 0.5),
            (90, 0.4),
        )
    )
    matrix, names, _ = gap_evidence(
        {IDENTITY: selected},
        {IDENTITY: candidates},
        {FIXTURE: 30},
        _measurements([30, 40, 60, 90]),
        _TimingModel(),
    )
    values = _values(matrix, names)

    assert values["strongest_local_score"] == 0.8
    assert values["second_strongest_local_score"] == 0.5
    assert np.isclose(values["mean_local_score"], (0.8 + 0.5 + 0.4) / 3)
    assert np.isclose(values["sum_local_score"], 1.7)
    assert values["strongest_timing_score"] == 0.8
    assert values["second_strongest_timing_score"] == 0.5
    assert values["usable_candidate_count"] == 4
    assert values["distinct_gaps_with_candidates"] == 3
    assert values["positive_duration_gap_count"] == 4
    assert values["gaps_with_no_candidates"] == 1
    assert values["excluded_candidate_count"] == 0
    assert np.isclose(values["longest_gap_seconds"], 1.0)
    assert values["last_gap_has_candidate"] == 1


def test_no_proposals_leave_summaries_missing_without_model_call() -> None:
    selected = _option(20, 80)
    model = _TimingModel()
    matrix, names, _ = gap_evidence(
        {IDENTITY: selected},
        {IDENTITY: (_event(20), _event(100), _event(-1))},
        {FIXTURE: 30},
        _measurements([]),
        model,
    )
    values = _values(matrix, names)

    assert model.calls == 0
    for name in names[:8]:
        assert np.isnan(values[name])
    assert values["usable_candidate_count"] == 0
    assert values["distinct_gaps_with_candidates"] == 0
    assert values["positive_duration_gap_count"] == 3
    assert values["gaps_with_no_candidates"] == 3
    assert values["excluded_candidate_count"] == 3
    assert values["last_gap_has_candidate"] == 0


def test_duplicate_distance_scales_and_edges_are_half_open_at_60_fps() -> None:
    selected = _option(50, start=20, end=100)
    candidates = {
        IDENTITY: (_event(37, 0.1), _event(38, 0.2), _event(100, 0.3), _event(19, 0.4))
    }
    matrix, names, _ = gap_evidence(
        {IDENTITY: selected},
        candidates,
        {FIXTURE: 60},
        _measurements([37]),
        _TimingModel(),
    )
    values = _values(matrix, names)

    assert values["usable_candidate_count"] == 1
    assert values["excluded_candidate_count"] == 3
    assert values["last_gap_has_candidate"] == 0


def test_missing_usable_candidate_measurement_raises() -> None:
    with pytest.raises(KeyError):
        gap_evidence(
            {IDENTITY: _option(20, 80)},
            {IDENTITY: (_event(50),)},
            {FIXTURE: 30},
            PhysicalMeasurements(("measurement",), {}, {}),
            _TimingModel(),
        )
