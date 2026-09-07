from __future__ import annotations

from pathlib import Path

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_closing_pass.scripts.prepare_vlm_acceptance import (
    TARGET_FIXTURES,
    _prepared_cases,
    _VideoSource,
    route_cases,
)


def _span(fixture: str, span_id: int, anchor: int) -> FixedSpan:
    event = FixedEvent(fixture, anchor, 0.8, None)
    return FixedSpan(fixture, span_id, 0, 2_000, (event,))


def _score(fixture: str, span_id: int, value: float, **extra: object) -> dict[str, object]:
    return {"fixture": fixture, "span_id": span_id, "gap_score": value, **extra}


def test_route_uses_only_ids_and_scores_not_saved_judgements() -> None:
    spans = [_span("sset_01", 3, 500)]
    first = route_cases([_score("sset_01", 3, 0.8, judgements={"10": {"outcome": "wrong"}})], 0.7, spans)
    second = route_cases([_score("sset_01", 3, 0.8, judgements={"10": {"outcome": "correct"}})], 0.7, spans)
    assert first == second
    assert first[0]["anchor_frame"] == 500


def test_unaccepted_and_empty_routes_are_excluded() -> None:
    spans = [_span("sset_15", 1, 200)]
    assert route_cases([_score("sset_15", 1, 0.69)], 0.7, spans) == ()
    assert route_cases([], 0.7, spans) == ()
    assert route_cases([_score("sset_15", 1, 0.8)], 0.7, [FixedSpan("sset_15", 1, 0, 100, ())]) == ()


def test_routes_only_target_group_of_three_fixtures_in_deterministic_order() -> None:
    spans = [
        _span("sset_21", 2, 220),
        _span("sset_01", 4, 140),
        _span("sset_15", 1, 180),
        _span("sset_02", 0, 100),
    ]
    rows = [
        _score("sset_21", 2, 0.8),
        _score("sset_01", 4, 0.8),
        _score("sset_15", 1, 0.8),
        _score("sset_02", 0, 0.99),
    ]
    routed = route_cases(rows, 0.7, spans)
    assert [record["fixture"] for record in routed] == list(TARGET_FIXTURES)
    assert all(record["kind"] == "natural" for record in routed)


def test_first_four_controls_are_shifted_with_native_120_frame_windows() -> None:
    fixture = "sset_01"
    spans = [_span(fixture, span_id, 500 + span_id * 200) for span_id in range(1, 6)]
    rows = [_score(fixture, span.span_id, 0.8) for span in spans]
    natural = route_cases(rows, 0.7, spans)
    sources = {fixture: _VideoSource(fixture, Path("source.avi"), 60.0, 2_000)}
    prepared = _prepared_cases(natural, sources)

    assert len(prepared) == 9
    assert [case.route["kind"] for case in prepared] == [
        "natural", "shifted", "natural", "shifted", "natural", "shifted",
        "natural", "shifted", "natural",
    ]
    shifted = [case for case in prepared if case.route["kind"] == "shifted"]
    assert [case.route["span_id"] for case in shifted] == [1, 2, 3, 4]
    first_natural = prepared[0]
    first_shifted = prepared[1]
    assert first_natural.manifest["source_start_frame"] == 620
    assert first_shifted.manifest["source_start_frame"] == 600
    assert first_shifted.manifest["source_end_frame"] == 720
    assert first_shifted.route["source_frame_indices"] == list(range(600, 720))
    assert first_shifted.route["fps"] == 60.0
    assert first_shifted.route["source_start_delta_frames"] == -20
    assert first_shifted.route["paired_natural_id"] == first_natural.route["case_id"]
    assert set(first_natural.manifest) == {
        "case_id", "video_id", "clip_path", "source_start_frame", "source_end_frame",
        "sample_fps", "expected_frames", "width", "height",
    }
    assert first_natural.manifest["expected_frames"] == 120
    assert first_natural.manifest["width"] == 512
    assert first_natural.manifest["height"] == 288


def test_shift_delta_records_clamping() -> None:
    fixture = "sset_15"
    natural = route_cases([_score(fixture, 1, 0.8)], 0.7, [_span(fixture, 1, 50)])
    prepared = _prepared_cases(
        natural, {fixture: _VideoSource(fixture, Path("source.avi"), 30.0, 200)},
    )
    shifted = prepared[1]
    assert shifted.route["source_start_frame"] == 0
    assert shifted.route["source_start_delta_frames"] == 0
    assert shifted.route["source_end_frame"] == 120
