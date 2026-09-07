"""Reuse base-option eligibility when labelling two-insertion alternatives."""

from collections.abc import Mapping, Sequence

import numpy as np

from scratch.contact_det_closing_pass.scripts.evaluation import (
    overlapping_rallies,
    section_result,
)
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    HumanLabels,
    scale_base30_frames,
)


def pair_targets(
    pairs: Sequence[LaterOption], singles: Sequence[LaterOption], single_targets: np.ndarray,
    labels: HumanLabels, fps: Mapping[str, float],
) -> np.ndarray:
    """Keep existing boundary/label eligibility and evaluate possible completions.

    Pair generation retains each base option's section edges. Its cached target
    therefore already establishes whether the same labels can judge the pair.
    With equal sorted contact counts, a complete one-to-one timing match must
    pair corresponding entries. The cheap timing test avoids running the full
    matcher on the many alternatives that cannot possibly complete a rally.
    """
    base_targets = {option.base: int(target) for option, target in zip(singles, single_targets, strict=True)
                    if option.inserted is None}
    tolerances = {fixture: scale_base30_frames(10, value) for fixture, value in fps.items()}
    rallies = {}
    targets = np.zeros(len(pairs), dtype=np.int8)
    for index, option in enumerate(pairs):
        if base_targets[option.base] < 0:
            targets[index] = -1
            continue
        span = option.span
        edges = (span.fixture, span.start_frame, span.end_frame)
        if edges not in rallies:
            rallies[edges] = overlapping_rallies(span, labels)[0]
        frames = rallies[edges].frames
        if len(frames) != len(span.events):
            continue
        tolerance = tolerances[span.fixture]
        if any(abs(frame - event.frame) > tolerance for frame, event in zip(frames, span.events, strict=True)):
            continue
        targets[index] = int(section_result(span, labels, tolerance)["side_rule_fully_correct"])
    return targets
