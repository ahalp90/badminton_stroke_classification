"""Pad fixed contact sections around their existing label-free predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    scale_base30_frames,
)


def _proposed_bounds(span: FixedSpan, padding: int) -> tuple[int, int]:
    """Return the unclipped half-open bounds proposed by one section."""
    if not span.events:
        return span.start_frame, span.end_frame
    first_frame = min(event.frame for event in span.events)
    last_frame = max(event.frame for event in span.events)
    start = max(0, min(span.start_frame, first_frame - padding))
    end = max(span.end_frame, last_frame + padding + 1)
    return start, end


def _ordered_indices(spans: Sequence[FixedSpan]) -> dict[str, list[int]]:
    """Group section positions in chronological order without changing inputs."""
    by_fixture: dict[str, list[int]] = {}
    for index, span in enumerate(spans):
        by_fixture.setdefault(span.fixture, []).append(index)
    for indices in by_fixture.values():
        indices.sort(key=lambda index: (spans[index].start_frame, spans[index].end_frame, spans[index].span_id))
    return by_fixture


def _clip_to_original_neighbours(
    spans: Sequence[FixedSpan], indices: Sequence[int], bounds: list[list[int]],
) -> None:
    """Keep each proposed extension in the gaps between original sections."""
    for position, index in enumerate(indices):
        if position:
            previous = spans[indices[position - 1]]
            bounds[index][0] = max(bounds[index][0], previous.end_frame)
        if position + 1 < len(indices):
            following = spans[indices[position + 1]]
            bounds[index][1] = min(bounds[index][1], following.start_frame)

    for left_index, right_index in pairwise(indices):
        left = spans[left_index]
        right = spans[right_index]
        if left.end_frame > right.start_frame:
            raise ValueError(f"{left.fixture}: original sections overlap")
        if bounds[left_index][1] <= bounds[right_index][0]:
            continue
        midpoint = (left.end_frame + right.start_frame) // 2
        bounds[left_index][1] = min(bounds[left_index][1], midpoint)
        bounds[right_index][0] = max(bounds[right_index][0], midpoint)


def pad_contact_boundaries(
    spans: Sequence[FixedSpan],
    events: Mapping[str, Sequence[FixedEvent]],
    fps: Mapping[str, float],
    padding_base30: int = 10,
    preserve_membership: bool = False,
) -> ContactStreams:
    """Extend sections around saved predictions while preserving full-stream contacts.

    Each non-empty section gets ``padding_base30`` frames on either side of
    its first and last predicted contact.  Extensions stop at neighbouring
    original section edges.  If both sides reach into one gap, its midpoint
    assigns the shared frames deterministically without changing either
    original boundary.

    :param spans: Original half-open contact sections.
    :param events: Full-stream predicted contacts, keyed by fixture.
    :param fps: Source frame rates, keyed by fixture.
    :param padding_base30: Padding measured at 30 frames per second.
    :param preserve_membership: Keep an original span intact if padding would
        change its contained event tuple.
    :return: New sections and the unchanged full-stream event records.
    """
    spans_tuple = tuple(spans)
    by_fixture = _ordered_indices(spans_tuple)
    fixtures = tuple(by_fixture)
    padding_by_fixture = {
        fixture: scale_base30_frames(padding_base30, fps[fixture])
        for fixture in fixtures
    }
    bounds = [list(_proposed_bounds(span, padding_by_fixture[span.fixture])) for span in spans_tuple]
    for indices in by_fixture.values():
        _clip_to_original_neighbours(spans_tuple, indices, bounds)

    full_stream = {fixture: tuple(fixture_events) for fixture, fixture_events in events.items()}
    output_spans = []
    for index, span in enumerate(spans_tuple):
        if span.fixture not in full_stream:
            raise KeyError(f"{span.fixture}: full-stream events are unavailable")
        start, end = bounds[index]
        section_events = tuple(
            event for event in full_stream[span.fixture] if start <= event.frame < end
        )
        if preserve_membership and section_events != span.events:
            start, end = span.start_frame, span.end_frame
            section_events = span.events
        output_spans.append(FixedSpan(span.fixture, span.span_id, start, end, section_events))
    return ContactStreams(tuple(output_spans), full_stream)
