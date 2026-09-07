"""Focused checks for label-free whole-rally feature joins."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
    action_matrix,
    build_whole_features,
    load_measurements,
    opening_score_features,
)
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_followup.scripts.score_start_model import ActionRow
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import CandidateRow


def _candidate(frame: int = 90) -> CandidateRow:
    return CandidateRow(
        "fixture",
        "A",
        30.0,
        0,
        100,
        200,
        80,
        120,
        frame,
        0.8,
        0.9,
        False,
        "Top",
        "Bot",
        (0.1,) * 9,
    )


def _action_rows() -> tuple[ActionRow, ...]:
    candidate = _candidate()
    return (
        ActionRow(candidate, "add", (*candidate.features, 0.0)),
        ActionRow(candidate, "replace", (*candidate.features, 1.0)),
    )


def _baseline() -> FixedSpan:
    return FixedSpan(
        "fixture",
        0,
        100,
        200,
        (
            FixedEvent("fixture", 120, 0.9, "Top"),
            FixedEvent("fixture", 150, 0.8, None),
        ),
    )


def test_load_measurements_requests_fullstream_and_action_frames_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _action_rows()
    calls: list[tuple[str, set[int]]] = []
    names = ("vx", "vy")
    index = SimpleNamespace(
        fps=30.0,
        by_frame={90: np.asarray([1.0, 2.0]), 120: np.asarray([3.0, np.nan])},
    )

    def fake_load(_root: object, fixture: str, _names: tuple[str, ...], frames: set[int]) -> object:
        calls.append((fixture, frames.copy()))
        return index

    monkeypatch.setattr(
        "scratch.contact_det_closing_pass.scripts.whole_rally_features._frozen_feature_names",
        lambda: names,
    )
    monkeypatch.setattr(
        "scratch.contact_det_closing_pass.scripts.whole_rally_features._load_fixture",
        fake_load,
    )

    measurements = load_measurements(
        rows,
        {"fixture": (FixedEvent("fixture", 150, 0.7, "Top"),)},
        tmp_path,
    )

    assert calls == [("fixture", {90, 120, 150})]
    assert np.isnan(measurements.values["fixture", 150]).all()
    assert measurements.audit["missing_identities"] == [("fixture", 150)]
    assert measurements.audit["measurement_nan_cells"] == 3


def test_action_matrix_uses_action_then_candidate_then_fixed_order() -> None:
    rows = _action_rows()
    measurements = PhysicalMeasurements(
        ("vx", "vy"),
        {
            ("fixture", 90): np.asarray([1.0, 2.0]),
            ("fixture", 120): np.asarray([3.0, 4.0]),
        },
        {},
    )

    matrix = action_matrix(rows, measurements)

    assert matrix.shape == (2, 14)
    np.testing.assert_allclose(matrix[0, :10], rows[0].features)
    np.testing.assert_allclose(matrix[0, 10:], (1.0, 2.0, 3.0, 4.0))
    missing_candidate = ActionRow(_candidate(91), "add", (0.1,) * 10)
    assert np.isnan(action_matrix((missing_candidate,), measurements)[0, 10])


def test_whole_features_keep_raw_side_pattern_and_physical_positions() -> None:
    baseline = _baseline()
    option_span = FixedSpan(
        "fixture",
        0,
        90,
        200,
        (
            FixedEvent("fixture", 90, 0.8, "Bot"),
            FixedEvent("fixture", 120, 0.9, "Top"),
            FixedEvent("fixture", 150, 0.8, None),
        ),
    )
    option = CombinedAction("add", 90, None, option_span)
    measurements = PhysicalMeasurements(
        ("vx", "vy"),
        {
            ("fixture", 90): np.asarray([1.0, 2.0]),
            ("fixture", 120): np.asarray([3.0, 4.0]),
        },
        {},
    )

    matrix, names, groups = build_whole_features(
        (option,),
        (baseline,),
        _action_rows(),
        {"fixture": 30.0},
        measurements,
    )

    assert matrix.shape == (1, 37 + 8 + 6)
    assert groups["summary"] == tuple(range(37))
    assert groups["side"] == tuple(range(37, 45))
    assert groups["physical"] == tuple(range(45, 51))
    assert names[37:45] == (
        "before__fraction_known",
        "before__fraction_known_starting_top",
        "before__fraction_known_starting_bot",
        "before__fraction_adjacent_known_same_side",
        "after__fraction_known",
        "after__fraction_known_starting_top",
        "after__fraction_known_starting_bot",
        "after__fraction_adjacent_known_same_side",
    )
    np.testing.assert_allclose(
        matrix[0, 37:45],
        (0.5, 1.0, 0.0, np.nan, 2 / 3, 0.0, 1.0, 0.0),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        matrix[0, 45:],
        (3.0, 4.0, 1.0, 2.0, np.nan, np.nan),
        equal_nan=True,
    )


def test_opening_scores_join_selected_and_section_maximums() -> None:
    baseline = _baseline()
    add = CombinedAction(
        "add_delete",
        90,
        150,
        FixedSpan("fixture", 0, 90, 200, (FixedEvent("fixture", 90, 0.8, "Top"),)),
    )
    keep = CombinedAction("keep", None, None, baseline)
    scores = {
        ("fixture", 0, 90, "add"): (0.2, 0.4),
        ("fixture", 0, 95, "replace"): (0.8, 0.7),
    }
    other_start = CombinedAction(
        "replace_delete",
        95,
        120,
        FixedSpan("fixture", 0, 95, 200, (FixedEvent("fixture", 95, 0.7, "Top"),)),
    )

    result = opening_score_features((add, keep, other_start), scores)

    np.testing.assert_allclose(
        result,
        ((0.2, 0.4, 0.8, 0.7), (np.nan, np.nan, 0.8, 0.7), (0.8, 0.7, 0.8, 0.7)),
        equal_nan=True,
    )


def test_opening_scores_without_start_proposals_are_nan() -> None:
    span = _baseline()
    options = (CombinedAction("delete", None, 120, span),)

    result = opening_score_features(options, {})

    assert result.shape == (1, 4)
    assert np.isnan(result).all()


def test_deleted_prefix_score_comes_from_another_option_span() -> None:
    baseline = _baseline()
    prefix = CombinedAction(
        "add",
        90,
        None,
        FixedSpan(
            "fixture",
            0,
            80,
            200,
            (
                FixedEvent("fixture", 80, 0.6, "Bot"),
                FixedEvent("fixture", 90, 0.8, "Top"),
                *baseline.events,
            ),
        ),
    )
    final = CombinedAction(
        "add_delete",
        90,
        80,
        FixedSpan(
            "fixture",
            0,
            90,
            200,
            (FixedEvent("fixture", 90, 0.8, "Top"), *baseline.events),
        ),
    )

    matrix, _names, _groups = build_whole_features(
        (prefix, final),
        (baseline,),
        _action_rows(),
        {"fixture": 30.0},
        PhysicalMeasurements((), {}, {}),
    )

    assert matrix[1, 36] == 0.6


def test_opening_score_for_valid_start_must_be_present_and_finite() -> None:
    option = CombinedAction(
        "add",
        90,
        None,
        FixedSpan("fixture", 0, 90, 200, (FixedEvent("fixture", 90, 0.8, "Top"),)),
    )

    with pytest.raises(KeyError, match="missing"):
        opening_score_features((option,), {})
    with pytest.raises(ValueError, match="not finite"):
        opening_score_features((option,), {("fixture", 0, 90, "add"): (np.nan, 0.4)})
