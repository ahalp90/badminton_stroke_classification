"""Serve recovery and proposed-start measures."""

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.serve_metrics import (
    accepted_serves,
    analyse_serves,
    compare_serves,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)

FIXTURE = "fixture"


def _event(frame: int, side: str | None = "Top") -> FixedEvent:
    return FixedEvent(FIXTURE, frame, 0.9, side)


def _span(
    *frames: int,
    start: int = 0,
    end: int = 300,
    span_id: int = 0,
    sides: tuple[str | None, ...] | None = None,
) -> FixedSpan:
    event_sides = sides if sides is not None else ("Top",) * len(frames)
    return FixedSpan(
        FIXTURE,
        span_id,
        start,
        end,
        tuple(_event(frame, side) for frame, side in zip(frames, event_sides, strict=True)),
    )


def _labels(
    *frames: int,
    side: str | None = "Top",
    rally_id: str = "set1:1",
) -> HumanLabels:
    rally = RallyReference(FIXTURE, 0, rally_id, tuple(frames))
    return HumanLabels(
        {FIXTURE: (rally,)},
        {(FIXTURE, frame): side for frame in frames},
    )


def _stream(*spans: FixedSpan, events: tuple[FixedEvent, ...] | None = None) -> ContactStreams:
    all_events = events if events is not None else tuple(event for span in spans for event in span.events)
    return ContactStreams(spans, {FIXTURE: all_events})


def test_already_present_serve_is_counted_once_with_final_side() -> None:
    result = analyse_serves(
        _stream(_span(100, 130, sides=("Top", "Bot"))),
        _labels(100, 130, side="Top"),
        {FIXTURE: 30.0},
        10,
    )

    assert result["total"]["firsts"] == 1
    assert result["total"]["matched"] == 1
    assert result["total"]["contact_matched"] == 2
    assert result["total"]["joint_correct"] == 1
    assert result["start_rows"][0]["status"] == "serve"
    assert result["serve_rows"][0]["matched_pred_frame"] == 100


def test_false_lead_before_recovered_serve_is_unknown_but_marked() -> None:
    span = _span(89, 100, start=80, end=140, sides=("Bot", "Top"))
    result = analyse_serves(
        _stream(span),
        _labels(100, 130),
        {FIXTURE: 30.0},
        0,
    )

    start = result["start_rows"][0]
    assert start["status"] == "unknown"
    assert start["unmatched_before_recovered_serve"] is True
    assert result["total"]["recovered_but_preceded_starts"] == 1
    assert result["serve_rows"][0]["matched_pred_frame"] == 100


def test_unmatched_event_inside_rally_envelope_is_extra_leading() -> None:
    result = analyse_serves(
        _stream(_span(110, start=100, end=140)),
        _labels(100, 130),
        {FIXTURE: 30.0},
        0,
    )

    assert result["start_rows"][0]["status"] == "extra_leading"
    assert result["total"]["unknown"] == 0


def test_receiver_start_is_later_hit() -> None:
    result = analyse_serves(
        _stream(_span(130, start=120, end=150)),
        _labels(100, 130),
        {FIXTURE: 30.0},
        0,
    )

    row = result["start_rows"][0]
    assert row["status"] == "later_hit"
    assert row["matched_rally_id"] == "set1:1"
    assert result["total"]["timing_correct_starts"] == 0


def test_duplicate_proposals_do_not_double_count_one_gt_serve() -> None:
    result = analyse_serves(
        _stream(_span(99, 100, sides=("Bot", "Top"))),
        _labels(100),
        {FIXTURE: 30.0},
        0,
    )

    assert result["total"]["firsts"] == 1
    assert result["total"]["matched"] == 1
    assert result["serve_rows"][0]["matched_pred_frame"] == 100


def test_serve_matched_outside_all_spans_is_retained_as_a_diagnosis() -> None:
    event = _event(100)
    result = analyse_serves(
        _stream(_span(start=110, end=120), events=(event,)),
        _labels(100),
        {FIXTURE: 30.0},
        0,
    )

    assert result["start_rows"][0]["status"] == "empty"
    assert result["serve_rows"][0]["matched_span_ids"] == []
    assert result["total"]["serves_outside_all_spans"] == 1


def test_missing_gt_side_is_reported_without_a_joint_success() -> None:
    result = analyse_serves(
        _stream(_span(100, sides=("Top",))),
        _labels(100, side=None),
        {FIXTURE: 30.0},
        0,
    )

    row = result["serve_rows"][0]
    assert row["raw_status"] == "missing_label"
    assert row["final_status"] == "missing_label"
    assert result["total"]["known_side_firsts"] == 0
    assert result["total"]["joint_correct"] == 0


def test_frame_tolerance_scales_from_base_30_at_60_fps() -> None:
    stream = _stream(_span(120, sides=("Top",)))
    labels = _labels(100)

    at_10 = analyse_serves(stream, labels, {FIXTURE: 60.0}, 10)
    at_5 = analyse_serves(stream, labels, {FIXTURE: 60.0}, 5)

    assert at_10["tolerance_by_fixture"][FIXTURE] == 20
    assert at_5["tolerance_by_fixture"][FIXTURE] == 10
    assert at_10["total"]["matched"] == 1
    assert at_5["total"]["matched"] == 0


def test_accepted_sections_scope_successes_but_keep_all_gt_firsts() -> None:
    first = _span(100, start=90, end=120, span_id=0)
    second = _span(200, start=190, end=220, span_id=1)
    result = analyse_serves(
        _stream(first, second),
        HumanLabels(
            {
                FIXTURE: (
                    RallyReference(FIXTURE, 0, "set1:1", (100,)),
                    RallyReference(FIXTURE, 1, "set1:2", (200,)),
                )
            },
            {(FIXTURE, 100): "Top", (FIXTURE, 200): "Top"},
        ),
        {FIXTURE: 30.0},
        0,
        accepted={(FIXTURE, 0)},
    )

    assert result["total"]["firsts"] == 2
    assert result["total"]["matched"] == 2
    assert result["total"]["accepted_matched"] == 1
    assert result["total"]["joint_correct"] == 1
    assert result["serve_rows"][1]["accepted_final_status"] == "missing_prediction"


def test_compare_reports_serve_and_start_transitions() -> None:
    before_stream = _stream(_span(110, start=90, end=140, sides=("Bot",)))
    after_stream = _stream(_span(100, start=90, end=140, sides=("Top",)))
    labels = _labels(100)
    before = analyse_serves(before_stream, labels, {FIXTURE: 30.0}, 0)
    after = analyse_serves(after_stream, labels, {FIXTURE: 30.0}, 0)

    comparison = compare_serves(before, after)
    assert comparison["counts"]["recovered"] == 1
    assert comparison["counts"]["newly_timing_correct"] == 1
    assert comparison["counts"]["newly_joint_correct"] == 1
    assert comparison["recovered"][0]["identity"] == [FIXTURE, "set1:1"]


def test_acceptance_reuses_matches_and_keeps_all_output_counts() -> None:
    stream = _stream(_span(100, span_id=0), _span(200, span_id=1))
    result = analyse_serves(stream, _labels(100, 200), {FIXTURE: 30.0}, 0)
    accepted = accepted_serves(result, {(FIXTURE, 1)})
    assert accepted["total"]["firsts"] == 1
    assert accepted["total"]["matched"] == 1
    assert accepted["total"]["accepted_matched"] == 0
    assert accepted["total"]["serve"] == 1
    assert accepted["total"]["accepted_serve"] == 0
    assert accepted["total"]["accepted_later_hit"] == 1
    assert accepted["total"]["all_starts"] == 2
    assert accepted["total"]["accepted_starts"] == 1
