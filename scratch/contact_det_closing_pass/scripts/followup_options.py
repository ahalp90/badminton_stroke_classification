"""Preserve saved choices and describe the extra contact in pair alternatives."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from scratch.contact_det_closing_pass.scripts.later_options import (
    LaterOption,
    insertion_features,
)
from scratch.contact_det_closing_pass.scripts.whole_rally_features import (
    PhysicalMeasurements,
)


def choice_key(record: Mapping[str, Any]) -> tuple:
    return tuple(record.get(name) for name in (
        "fixture", "span_id", "kind", "candidate_frame", "deleted_frame", "inserted_frame",
        "second_inserted_frame", "start_frame", "end_frame",
    ))


def restore_choices(options: Sequence[LaterOption], records: Sequence[Mapping[str, Any]]) -> dict:
    from scratch.contact_det_closing_pass.scripts.later_options import option_record

    by_key = {choice_key(option_record(option)): option for option in options}
    selected = {}
    for record in records:
        option = by_key[choice_key(record)]
        selected[option.base.identity] = option
    expected = {option.base.identity for option in options}
    if set(selected) != expected:
        raise ValueError("Saved choices do not cover the option population")
    return selected


def contextual_insertions(options: Sequence[LaterOption]) -> tuple[LaterOption, ...]:
    """Describe each insertion against the contact list it actually joins."""
    rows = []
    for option in options:
        for candidate in option.inserted_events:
            before = replace(option.span, events=tuple(
                event for event in option.span.events if event.frame != candidate.frame
            ))
            rows.append(LaterOption(replace(option.base, span=before), candidate, option.span))
    return tuple(rows)


def pair_features(
    options: Sequence[LaterOption], fps: Mapping[str, float], measurements: PhysicalMeasurements,
) -> tuple[np.ndarray, tuple[str, ...]]:
    second_rows = []
    for option in options:
        candidate = option.second_inserted
        before = option.base.span
        if candidate is not None:
            before = replace(option.span, events=tuple(
                event for event in option.span.events if event.frame != candidate.frame
            ))
        second_rows.append(LaterOption(replace(option.base, span=before), candidate, option.span))
    matrix, names = insertion_features(second_rows, fps, measurements)
    return matrix, tuple(f"second__{name}" for name in names)
