"""Score fixed contact-boundary padding on the saved broader predictions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.boundary_followup import (
    pad_contact_boundaries,
)
from scratch.contact_det_closing_pass.scripts.evaluation import (
    score_contacts,
    test_labels,
    write_json,
)
from scratch.contact_det_closing_pass.scripts.later_evaluation import compare_outputs
from scratch.contact_det_closing_pass.scripts.later_options import LaterOption
from scratch.contact_det_closing_pass.scripts.run_broader_comparison import (
    stream_records,
)
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.contact_det_closing_pass.scripts.whole_rally_evaluation import (
    voted_contact_scores,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_followup.scripts import score_start_model as start
from scratch.contact_det_followup.scripts.audit_combined_best_case import CombinedAction
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import ContactStreams

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
RESULTS = ROOT / "results/followups"
DEFAULT_LATER_INPUTS = ROOT / "raw/later_inputs/broader.json.gz"
VARIANTS = ("session_start", "local", "pairs", "both", "early")
BOUNDARY_MODES = ("padding", "fixed_membership")
TOLERANCES = (10, 5)


def _mode_suffix(boundary_mode: str) -> str:
    if boundary_mode == "padding":
        return ""
    if boundary_mode == "fixed_membership":
        return "_fixed_membership"
    raise ValueError(f"unknown boundary mode: {boundary_mode}")


def _source_path(variant: str) -> Path:
    if variant == "session_start":
        return ROOT / "results/later/later_broader_predictions.json.gz"
    return RESULTS / f"{variant}_broader_predictions.json.gz"


def _load_source(variant: str) -> tuple[dict[str, Any], str]:
    path = _source_path(variant)
    payload = prediction_io.read_json(path)
    if payload.get("status") != "complete" or payload.get("labels_read") is not False:
        raise ValueError(f"{path.name}: source predictions are incomplete or used labels")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError(f"{path.name}: videos must be a list")
    by_fixture = {}
    for video in raw_videos:
        if not isinstance(video, dict):
            raise TypeError(f"{path.name}: each video must be an object")
        fixture = str(video["fixture"])
        if fixture in by_fixture:
            raise ValueError(f"{fixture}: source video repeats")
        by_fixture[fixture] = video
    return by_fixture, path.name


def _native_fps(path: Path, expected_fixtures: Sequence[str]) -> dict[str, float]:
    payload = prediction_io.read_json(path)
    if payload.get("status") != "complete" or payload.get("labels_read") is not False:
        raise ValueError("Broader inputs are incomplete or used labels")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("Broader input videos must be a list")
    fps = {str(video["fixture"]): float(video["fps"]) for video in raw_videos}
    if set(fps) != set(expected_fixtures):
        raise ValueError("Broader inputs do not cover the frozen prediction fixtures")
    return fps


def _score_stream(
    stream: ContactStreams,
    labels: Any,
    fps: Mapping[str, float],
) -> dict[str, Any]:
    """Score full-stream timing and raw/voted player sides."""
    voted = start.apply_whole_rally_alternation(stream)
    contacts = {}
    for tolerance in TOLERANCES:
        raw = score_contacts(stream.events_by_fixture, labels, fps, tolerance)
        contacts[str(tolerance)] = {
            "raw": raw,
            "fixed_side": voted_contact_scores(raw, voted.events_by_fixture),
        }
    return {"contacts": contacts}


def _fulltime_pair(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Retain before/after full-stream timing and voted-side totals."""
    fulltime = {}
    for tolerance in TOLERANCES:
        key = str(tolerance)
        fulltime[key] = {
            "raw": {
                "before": before["contacts"][key]["raw"]["total"],
                "after": after["contacts"][key]["raw"]["total"],
            },
            "fixed_side": {
                "before": before["contacts"][key]["fixed_side"]["total"],
                "after": after["contacts"][key]["fixed_side"]["total"],
            },
        }
    return fulltime


def _boundary_options(
    before: Sequence[FixedSpan], after: Sequence[FixedSpan],
) -> dict[tuple[str, int], LaterOption]:
    """Represent padded spans as the existing chooser's selected options."""
    if len(before) != len(after):
        raise ValueError("Boundary output changed the number of sections")
    selected = {}
    for old_span, new_span in zip(before, after, strict=True):
        if (old_span.fixture, old_span.span_id) != (new_span.fixture, new_span.span_id):
            raise ValueError("Boundary output changed section identities")
        base = CombinedAction("boundary", None, None, old_span)
        selected[(old_span.fixture, old_span.span_id)] = LaterOption(base, None, new_span)
    return selected


def _edge_records(original: Sequence[FixedSpan], padded: Sequence[FixedSpan]) -> list[dict[str, Any]]:
    if len(original) != len(padded):
        raise ValueError("Boundary output changed the number of sections")
    records = []
    for before, after in zip(original, padded, strict=True):
        if (before.fixture, before.span_id) != (after.fixture, after.span_id):
            raise ValueError("Boundary output changed section identities")
        records.append({
            "span_id": before.span_id,
            "original_start_frame": before.start_frame,
            "original_end_frame": before.end_frame,
            "new_start_frame": after.start_frame,
            "new_end_frame": after.end_frame,
            "start_extended": after.start_frame < before.start_frame,
            "end_extended": after.end_frame > before.end_frame,
        })
    return records


def run(
    variant: str,
    *,
    boundary_mode: str = "padding",
    later_inputs: Path = DEFAULT_LATER_INPUTS,
    output_root: Path = RESULTS,
) -> dict[str, Any]:
    """Pad one saved broader output and score it against both references."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown boundary variant: {variant}")
    suffix = _mode_suffix(boundary_mode)
    started = perf_counter()
    source_by_fixture, source_name = _load_source(variant)
    session_by_fixture, session_name = (
        (source_by_fixture, source_name)
        if variant == "session_start"
        else _load_source("session_start")
    )
    fixtures = tuple(source_by_fixture)
    if len(fixtures) != 47:
        raise ValueError("Boundary source must cover the frozen 47 videos")
    if set(session_by_fixture) != set(fixtures):
        raise ValueError("Direct session-start predictions do not cover the source videos")
    fps = _native_fps(later_inputs, fixtures)
    direct_by_fixture: dict[str, ContactStreams] = {}
    session_by_fixture_stream: dict[str, ContactStreams] = {}
    source_spans: list[FixedSpan] = []
    source_events: dict[str, Sequence[FixedEvent]] = {}
    video_records = []
    for fixture in fixtures:
        source = source_by_fixture[fixture]
        if float(source["fps"]) != fps[fixture]:
            raise ValueError(f"{fixture}: source fps differs from broader input fps")
        direct = restore_stream(source["output"])
        if set(direct.events_by_fixture) != {fixture}:
            raise ValueError(f"{fixture}: source output has unexpected fixtures")
        if tuple(span.fixture for span in direct.spans) != (fixture,) * len(direct.spans):
            raise ValueError(f"{fixture}: source output has unexpected section fixtures")
        session_source = session_by_fixture[fixture]
        if float(session_source["fps"]) != fps[fixture]:
            raise ValueError(f"{fixture}: session-start fps differs from broader input fps")
        session_stream = direct if variant == "session_start" else restore_stream(session_source["output"])
        direct_by_fixture[fixture] = direct
        session_by_fixture_stream[fixture] = session_stream
        source_spans.extend(direct.spans)
        source_events[fixture] = direct.events_by_fixture[fixture]

    source = ContactStreams(tuple(source_spans), source_events)
    boundary = pad_contact_boundaries(
        source.spans,
        source.events_by_fixture,
        fps,
        padding_base30=10,
        preserve_membership=boundary_mode == "fixed_membership",
    )
    source_spans_by_fixture = {
        fixture: tuple(span for span in source.spans if span.fixture == fixture)
        for fixture in fixtures
    }
    boundary_spans_by_fixture = {
        fixture: tuple(span for span in boundary.spans if span.fixture == fixture)
        for fixture in fixtures
    }
    for fixture in fixtures:
        direct = direct_by_fixture[fixture]
        padded = ContactStreams(
            boundary_spans_by_fixture[fixture],
            {fixture: boundary.events_by_fixture[fixture]},
        )
        direct_records = stream_records(direct)
        padded_records = stream_records(padded)
        video_records.append({
            "fixture": fixture,
            "fps": fps[fixture],
            "new_edges": _edge_records(source_spans_by_fixture[fixture], padded.spans),
            "raw_stream_unchanged": direct_records["contacts"] == padded_records["contacts"],
            "output": padded_records,
        })

    prediction_payload = {
        "schema": "contact-boundary-predictions/1",
        "status": "complete",
        "labels_read": False,
        "prediction_selection_uses_labels": False,
        "variant": variant,
        "boundary_mode": boundary_mode,
        "padding_base30": 10,
        "source_predictions": source_name,
        "direct_session_start_predictions": session_name,
        "data_status": "Previously examined videos; boundary padding applied without labels",
        "videos": video_records,
        "prediction_seconds": perf_counter() - started,
    }
    prediction_path = output_root / f"{variant}_boundary_broader_predictions{suffix}.json.gz"
    result_path = output_root / f"{variant}_boundary_broader_result{suffix}.json.gz"
    write_json(prediction_path, prediction_payload)
    print(f"Saved label-free boundary predictions to {prediction_path}", flush=True)

    labels = test_labels()
    groups = dict.fromkeys(fixtures, "ShuttleSet22")
    boundary_options = _boundary_options(source.spans, boundary.spans)
    comparison = compare_outputs(source.spans, boundary_options, labels, fps, groups)
    session_spans = tuple(
        span for stream in session_by_fixture_stream.values() for span in stream.spans
    )
    session_events = {
        fixture: stream.events_by_fixture[fixture]
        for fixture, stream in session_by_fixture_stream.items()
    }
    session = ContactStreams(session_spans, session_events)
    session_comparison = (
        comparison
        if variant == "session_start"
        else compare_outputs(session.spans, boundary_options, labels, fps, groups)
    )
    source_scores = _score_stream(source, labels, fps)
    boundary_scores = _score_stream(boundary, labels, fps)
    session_scores = source_scores if variant == "session_start" else _score_stream(session, labels, fps)
    fulltime = {
        "comparison_to_input": _fulltime_pair(source_scores, boundary_scores),
        "comparison_to_session_start": _fulltime_pair(session_scores, boundary_scores),
    }
    local_comparisons = {}
    if variant in {"both", "early"}:
        local_payload = prediction_io.read_json(
            output_root / f"local_boundary_broader_predictions{suffix}.json.gz"
        )
        local_spans = tuple(
            span for video in local_payload["videos"]
            for span in restore_stream(video["output"]).spans
        )
        local_comparisons["comparison_to_local_boundary"] = compare_outputs(
            local_spans, boundary_options, labels, fps, groups,
        )
    result = {
        "schema": "contact-boundary-comparison/1",
        "status": "complete",
        "labels_read": True,
        "variant": variant,
        "boundary_mode": boundary_mode,
        "padding_base30": 10,
        "data_status": "Previously examined videos; boundary padding scored after label-free predictions were saved",
        "counts": {
            "videos": len(fixtures),
            "sections": len(boundary.spans),
            "raw_streams_equal": sum(record["raw_stream_unchanged"] for record in video_records),
        },
        "comparison_to_input_detector": comparison,
        "comparison_to_session_start": session_comparison,
        **local_comparisons,
        "fulltime": fulltime,
        "full_stream_contacts": sum(len(events) for events in boundary.events_by_fixture.values()),
        "raw_contact_stream_unchanged": all(record["raw_stream_unchanged"] for record in video_records),
        "lineage": {
            "prediction_selection_uses_labels": False,
            "boundary_padding_uses_labels": False,
            "boundary_mode": boundary_mode,
            "raw_stream_preserved": all(record["raw_stream_unchanged"] for record in video_records),
            "padding_base30": 10,
            "source_predictions": source_name,
            "direct_session_start_predictions": session_name,
        },
        "timings": {"total_seconds": perf_counter() - started},
    }
    write_json(result_path, result)
    for tolerance in TOLERANCES:
        pair = comparison[str(tolerance)]["paired"]
        print(
            variant,
            "boundary",
            tolerance,
            "correct",
            pair["correct_before"],
            pair["correct_after"],
            "repairs",
            len(pair["repaired"]),
            "losses",
            len(pair["lost"]),
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--boundary-mode", choices=BOUNDARY_MODES, default="padding")
    parser.add_argument("--later-inputs", type=Path, default=DEFAULT_LATER_INPUTS)
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    arguments = parser.parse_args()
    result = run(
        arguments.variant,
        boundary_mode=arguments.boundary_mode,
        later_inputs=arguments.later_inputs,
        output_root=arguments.output_root,
    )
    print(f"Finished {result['status']}", flush=True)


if __name__ == "__main__":
    main()
