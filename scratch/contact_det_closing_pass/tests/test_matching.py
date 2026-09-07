"""Contract tests for deterministic time-only contact matching."""

from __future__ import annotations

from itertools import combinations

import pytest

from annotator.fps_constants import ScalingKind
from scratch.contact_det_closing_pass.scripts.matching import match_contacts


def _brute_force_objective(
    gt_frames: tuple[int, ...],
    predicted_frames: tuple[int, ...],
    tolerance: int,
) -> tuple[int, int]:
    """Find the best cardinality and error over all one-to-one pairings."""
    best = (0, 0)
    prediction_count = len(predicted_frames)
    for match_count in range(1, min(len(gt_frames), prediction_count) + 1):
        for gt_indices in combinations(range(len(gt_frames)), match_count):
            for prediction_indices in combinations(range(prediction_count), match_count):
                for pairing in _permutations(prediction_indices):
                    errors = [
                        abs(predicted_frames[prediction_index] - gt_frames[gt_index])
                        for gt_index, prediction_index in zip(gt_indices, pairing)
                    ]
                    if all(error <= tolerance for error in errors):
                        candidate = (match_count, sum(errors))
                        if candidate[0] > best[0] or (
                            candidate[0] == best[0] and candidate[1] < best[1]
                        ):
                            best = candidate
    return best


def _permutations(values: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return permutations without adding a dependency for this tiny oracle."""
    if not values:
        return [()]
    return [
        (value,) + remainder
        for position, value in enumerate(values)
        for remainder in _permutations(values[:position] + values[position + 1 :])
    ]


def _objective(matches: list[tuple[int, int, int]]) -> tuple[int, int]:
    return len(matches), sum(abs(offset) for _, _, offset in matches)


def test_expected_matches_and_source_frame_tolerance() -> None:
    assert match_contacts([0, 12], [8, 20], 10) == [(0, 0, 8), (1, 1, 8)]
    assert match_contacts([0, 12], [8, 20], 5) == [(1, 0, -4)]


def test_equal_objective_uses_match_then_skip_prediction_then_skip_gt() -> None:
    # The terminal cell can match the later prediction or skip it.  Match has
    # priority on this equal-objective transition.
    assert match_contacts([10], [8, 12], 2) == [(0, 1, 2)]
    # With equal timestamps, stable (frame, original index) order is observable.
    assert match_contacts([10, 10], [10, 10], 0) == [(0, 0, 0), (1, 1, 0)]


def test_duplicate_timestamps_preserve_original_indices() -> None:
    assert match_contacts([5, 5, 9], [4, 6], 1) == [(0, 0, -1), (1, 1, 1)]


def test_empty_inputs_and_negative_tolerance() -> None:
    assert match_contacts([], [], 0) == []
    assert match_contacts([1, 2], [], 3) == []
    assert match_contacts([], [1, 2], 3) == []
    with pytest.raises(ValueError, match="non-negative"):
        match_contacts([1], [1], -1)


@pytest.mark.parametrize("gt_frames", [(), (0,), (0, 1), (1, 1, 2)])
def test_exhaustive_small_sequences_match_brute_force_objective(
    gt_frames: tuple[int, ...],
) -> None:
    for predicted_frames in [(), (0,), (1,), (0, 1), (1, 0), (0, 0, 1)]:
        for tolerance in range(3):
            matches = match_contacts(gt_frames, predicted_frames, tolerance)
            assert _objective(matches) == _brute_force_objective(
                gt_frames,
                predicted_frames,
                tolerance,
            )


def test_callers_scale_base_30_tolerance_to_60_fps() -> None:
    tolerance = int(ScalingKind.FRAME_COUNT.scale(5, 60.0))
    assert tolerance == 10
    assert match_contacts([100], [109], tolerance) == [(0, 0, 9)]
