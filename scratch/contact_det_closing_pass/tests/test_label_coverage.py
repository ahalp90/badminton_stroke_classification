from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from scratch.contact_det_closing_pass.scripts import check_label_coverage as checker


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    annotation_root = tmp_path / "ss22_annotations"
    set_root = annotation_root / "set" / "match"
    set_root.mkdir(parents=True)
    pd.DataFrame({"id": [8], "video": ["match"]}).to_csv(
        annotation_root / "set" / "match.csv", index=False
    )
    pd.DataFrame(
        {
            "rally": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "ball_round": [1, 2, 3, 1, 2, 1, 2, 1, 2],
            "frame_num": [100, 130, 160, 300, 330, 430, "999999999999999999999999", 520, 510],
            "flaw": [None, None, None, None, "uncertain", None, None, None, None],
        }
    ).to_csv(set_root / "set1.csv", index=False)
    prediction_path = tmp_path / "frozen_predictions.json"
    prediction_path.write_text(
        json.dumps(
            {
                "schema": "shuttleset22-contact-predictions-combined/1",
                "source_commit": "example",
                "videos": [
                    {
                        "video_id": 8,
                        "fps": 30,
                        "frame_count": 600,
                        "contacts": [{"frame": frame} for frame in [100, 130, 160, 300, 430, 510]],
                        "spans": [
                            {"span_id": 0, "start_frame": 90, "end_frame": 180},
                            {"span_id": 1, "start_frame": 280, "end_frame": 340},
                            {"span_id": 2, "start_frame": 400, "end_frame": 480},
                            {"span_id": 3, "start_frame": 500, "end_frame": 540},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return annotation_root, prediction_path


def test_cleaning_records_separate_causes_and_uncertain_rows(tmp_path: Path) -> None:
    annotation_root, _ = _write_fixture(tmp_path)

    kept, dropped, counts = checker.clean_and_dropped(
        annotation_root / "set" / "match", 600
    )

    assert kept == [100, 130, 160]
    assert [rally["removal_reasons"] for rally in dropped] == [
        ["flaw"],
        ["invalid_frame"],
        ["non_monotonic_frames"],
    ]
    assert counts["dropped_rallies_with_flaw"] == 1
    assert counts["dropped_rallies_with_invalid_frame"] == 1
    assert counts["dropped_rallies_with_non_monotonic_frames"] == 1

    flawed_rows = dropped[0]["rows"]
    assert flawed_rows[0]["status"] == "uncertain_anchor"
    assert flawed_rows[0]["source_csv"] == "set1.csv"
    assert flawed_rows[0]["source_row_index"] == 3
    assert flawed_rows[1]["status"] == "removed_flaw"
    assert flawed_rows[1]["flaw"] == "uncertain"

    invalid_rows = dropped[1]["rows"]
    assert invalid_rows[0]["status"] == "uncertain_anchor"
    assert invalid_rows[1]["status"] == "removed_invalid_frame"
    assert invalid_rows[1]["raw_timestamp"] == "999999999999999999999999"
    assert invalid_rows[1]["frame"] is None

    assert all(
        row["status"] == "uncertain_anchor"
        for row in dropped[2]["uncertain_anchor_rows"]
    )


def test_diagnosis_records_sections_without_creating_an_ignore_interval(
    tmp_path: Path,
) -> None:
    annotation_root, prediction_path = _write_fixture(tmp_path)

    result = checker.diagnose(annotation_root, prediction_path)
    video = result["by_video"][0]

    assert result["inputs"] == {
        "annotations": {"role": "ShuttleSet22 annotations", "basename": "ss22_annotations"},
        "predictions": {"role": "combined predictions", "basename": "frozen_predictions.json"},
    }
    assert result["counts"]["sections_without_clean_labels"] == 3
    assert result["counts"]["sections_without_clean_labels_but_with_dropped_anchors"] == 3
    assert video["section_ids_without_clean_labels_but_with_dropped_anchors"] == [1, 2, 3]
    assert video["section_records"][0] == {
        "video_id": 8,
        "span_id": 0,
        "start_frame": 90,
        "end_frame": 180,
        "clean_label_count": 3,
        "no_clean_labels": False,
        "uncertain_anchor_count": 0,
        "uncertain_anchors": [],
    }
    invalid_section = video["section_records"][2]
    assert invalid_section["uncertain_anchor_count"] == 1
    assert invalid_section["uncertain_anchors"][0]["frame"] == 430
    assert invalid_section["uncertain_anchors"][0]["source_row_index"] == 5
    assert all(
        anchor["frame"] != 999999999999999999999999
        for record in video["section_records"]
        for anchor in record["uncertain_anchors"]
    )


def test_self_test_and_gzip_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert checker.self_test()["checks_passed"] is True
    output = tmp_path / "coverage.json.gz"
    checker.write_json(output, {"checks_passed": True})
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == {"checks_passed": True}

    monkeypatch.chdir(tmp_path)
    checker.main(["--self-test"])
    assert (tmp_path / "label_coverage.json.gz").is_file()
