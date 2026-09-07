"""Deterministic time-only matching of contact frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MATCH = np.uint8(1)
_SKIP_PREDICTION = np.uint8(2)
_SKIP_GT = np.uint8(3)


def _is_better(candidate: tuple[int, int], incumbent: tuple[int, int]) -> bool:
    """Return whether a score has a larger count or a lower error at that count."""
    return candidate[0] > incumbent[0] or (
        candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
    )


def match_contacts(
    gt_frames: Sequence[int],
    predicted_frames: Sequence[int],
    tolerance: int,
) -> list[tuple[int, int, int]]:
    """Match ground-truth and predicted frames in time order.

    A pair is valid when its absolute frame difference is at most ``tolerance``.
    The result maximises the number of one-to-one valid pairs, then minimises the
    sum of their absolute frame differences.  Inputs are stably sorted by
    ``(frame, original index)`` for the chronological dynamic programme and
    deterministic duplicate handling.  Equal-objective transitions prefer a
    match, then skipping a prediction, then skipping a ground-truth frame.

    :param gt_frames: Ground-truth frame numbers in their original input order.
    :param predicted_frames: Predicted frame numbers in their original input order.
    :param tolerance: Maximum allowed absolute frame difference, in source frames.
    :return: Chronological ``(gt_index, prediction_index, prediction_minus_gt)``
        triples using indices from the original input sequences.
    :raises ValueError: If ``tolerance`` is negative.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    sorted_gt = sorted((int(frame), index) for index, frame in enumerate(gt_frames))
    sorted_predictions = sorted(
        (int(frame), index) for index, frame in enumerate(predicted_frames)
    )
    ground_truth_count = len(sorted_gt)
    prediction_count = len(sorted_predictions)

    # Score storage is linear in predictions; traceback uses one byte per prefix cell.
    traceback = np.zeros(
        (ground_truth_count + 1, prediction_count + 1),
        dtype=np.uint8,
    )
    traceback[1:, 0] = _SKIP_GT
    traceback[0, 1:] = _SKIP_PREDICTION

    previous: list[tuple[int, int]] = [(0, 0)] * (prediction_count + 1)
    for gt_position, (gt_frame, _) in enumerate(sorted_gt, start=1):
        current: list[tuple[int, int]] = [(0, 0)] * (prediction_count + 1)
        current[0] = (0, 0)
        for prediction_position, (prediction_frame, _) in enumerate(
            sorted_predictions,
            start=1,
        ):
            # Start with the match so the stated transition priority wins exact
            # ties.  Invalid matches leave the two skip transitions to compare.
            distance = abs(prediction_frame - gt_frame)
            if distance <= tolerance:
                best = (
                    previous[prediction_position - 1][0] + 1,
                    previous[prediction_position - 1][1] + distance,
                )
                action = _MATCH
            else:
                best = current[prediction_position - 1]
                action = _SKIP_PREDICTION

            skip_prediction = current[prediction_position - 1]
            if _is_better(skip_prediction, best):
                best = skip_prediction
                action = _SKIP_PREDICTION

            skip_gt = previous[prediction_position]
            if _is_better(skip_gt, best):
                best = skip_gt
                action = _SKIP_GT

            current[prediction_position] = best
            traceback[gt_position, prediction_position] = action
        previous = current

    matches: list[tuple[int, int, int]] = []
    gt_position = ground_truth_count
    prediction_position = prediction_count
    while gt_position > 0 and prediction_position > 0:
        action = traceback[gt_position, prediction_position]
        if action == _MATCH:
            gt_frame, gt_index = sorted_gt[gt_position - 1]
            prediction_frame, prediction_index = sorted_predictions[prediction_position - 1]
            matches.append((gt_index, prediction_index, prediction_frame - gt_frame))
            gt_position -= 1
            prediction_position -= 1
        elif action == _SKIP_PREDICTION:
            prediction_position -= 1
        elif action == _SKIP_GT:
            gt_position -= 1
        else:
            raise RuntimeError(f"invalid traceback action {int(action)}")

    matches.reverse()
    return matches
