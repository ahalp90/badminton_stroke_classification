"""Check original-label edge cases that the retained-label reports cannot exercise."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_closing_pass.scripts.regenerate_figures import (
    regenerate_metric_figures,
    selection_metrics,
    stage_recall,
)
from scratch.contact_det_closing_pass.scripts.summarise_metrics import (
    ROOT,
    full_stream_counts,
    section_rows,
    selection_summary,
    write_table,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    ContactStreams,
    HumanLabels,
)


def test_duplicate_source_timestamps_keep_their_separate_player_labels() -> None:
    events = (FixedEvent("video", 100, 1.0, "Top"), FixedEvent("video", 103, 1.0, "Bot"))
    stream = ContactStreams((FixedSpan("video", 0, 90, 110, events),), {"video": events})
    labels = HumanLabels({"video": (RallyReference("video", 0, "rally", (100, 100)),)}, {("video", 100): "Bot"})
    sides = {("video", "rally"): ("Top", "Bot")}
    rows = section_rows(stream, labels, sides, {"video": 30.0}, 5)
    assert rows[0]["outcome"] == "correct"
    assert rows[0]["voted_correct_sides"] == 2
    counts = full_stream_counts(stream, labels, sides, {"video": 30.0}, 5)
    assert counts["side_correct"] == 2
    assert counts["serve_matched"] == counts["serve_side_correct"] == 1


def test_unlabelled_proposal_stays_unknown_without_recovery_credit() -> None:
    events = (FixedEvent("video", 100, 1.0, "Top"),)
    stream = ContactStreams((FixedSpan("video", 0, 90, 110, events),), {"video": events})
    labels = HumanLabels({"video": ()}, {})
    rows = section_rows(stream, labels, {}, {"video": 30.0}, 10)
    summary = selection_summary(rows, 0)
    assert summary["unknown"] == 1
    assert summary["correct"] == summary["unique_complete"] == summary["unique_contained"] == 0


@pytest.mark.parametrize(("fps", "expected"), [(30.0, 0), (60.0, 1)])
def test_full_stream_tolerance_scales_to_source_frames(fps: float, expected: int) -> None:
    events = (FixedEvent("video", 115, 1.0, "Top"),)
    stream = ContactStreams((FixedSpan("video", 0, 90, 120, events),), {"video": events})
    labels = HumanLabels({"video": (RallyReference("video", 0, "rally", (100,)),)}, {("video", 100): "Top"})
    counts = full_stream_counts(stream, labels, {("video", "rally"): ("Top",)}, {"video": fps}, 10)
    assert counts["serve_matched"] == counts["start_matched"] == expected


def test_repeated_proposals_do_not_inflate_unique_rally_recovery() -> None:
    row = {
        "fixture": "video",
        "rally_id": "rally",
        "fully_correct": True,
        "whole_rally_contained": True,
        "overlapping_rallies": 1,
        "outcome": "correct",
    }
    summary = selection_summary([row, row], 1)
    assert summary["correct"] == 2
    assert summary["unique_complete"] == summary["unique_contained"] == 1


def test_reports_and_chart_labels_reproduce_the_saved_counts(tmp_path: Path) -> None:
    with gzip.open(ROOT / "results/metric_summary.json.gz", "rt") as source:
        result = json.load(source)
    table = tmp_path / "serve_tables.md"
    write_table(result, table)
    assert table.read_text() == (ROOT / "serve_tables.md").read_text()

    regenerate_metric_figures(result, tmp_path)
    for generated in tmp_path.glob("*.svg"):
        saved = ROOT / "figures" / generated.name
        # SVG metadata changes on each render; the displayed labels must stay equal.
        assert re.findall(r"<!-- (.*?) -->", generated.read_text()) == re.findall(
            r"<!-- (.*?) -->", saved.read_text()
        ), generated.name


def test_selection_figures_keep_population_denominators_separate() -> None:
    result = {
        "selected": {
            "retained": {"10": {"proposals": 10, "unknown": 2, "unique_complete": 4, "labelled_rallies": 20}},
            "all_gt": {"10": {"proposals": 10, "unknown": 2, "unique_complete": 4, "labelled_rallies": 25}},
        }
    }
    assert selection_metrics(result, "unique_complete") == pytest.approx([50, 20, 200 * 4 / 28])
    assert selection_metrics(result, "unique_complete", "all_gt") == pytest.approx([40, 16, 200 * 4 / 35])


def test_stage_figures_use_requested_order_and_each_population() -> None:
    result = {
        "stages": {
            "final": {
                "retained": {"10": {"unique_complete": 8, "labelled_rallies": 10}},
                "all_gt": {"10": {"unique_complete": 7, "labelled_rallies": 20}},
            },
            "original": {
                "retained": {"10": {"unique_complete": 3, "labelled_rallies": 10}},
                "all_gt": {"10": {"unique_complete": 2, "labelled_rallies": 20}},
            },
        }
    }
    assert stage_recall(result, ["original", "final"]) == [30, 80]
    assert stage_recall(result, ["original", "final"], "all_gt") == [10, 35]
