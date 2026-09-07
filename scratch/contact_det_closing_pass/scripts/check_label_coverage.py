#!/usr/bin/env python3
"""Check whether dropped ShuttleSet22 labels are being treated as negatives.

This is a diagnostic, not a replacement scorer. It never changes predictions,
never declares excluded labels correct, and never produces a corrected
precision by silently dropping hard cases.

The output keeps valid rows from dropped rallies as explicitly uncertain
anchors. They are useful for inspection, but they are not new ground truth.
"""
from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REVIEWED_COMMIT = "f7b106e20b44fd26ab5d55ed07e2b5753b19e7b1"
TOLERANCES = (5, 10)


def read_json(path: Path) -> dict[str, Any]:
    """Read an object from plain or gzip-compressed JSON."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write JSON, selecting gzip from the output filename suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _json_safe(value: object) -> object:
    """Convert pandas and NumPy scalars into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def greedy_matches(
    expected: Sequence[int], predicted: Sequence[int], tolerance: int
) -> list[tuple[int, int, int]]:
    """Reproduce the reviewed contact scorer's nearest-pair-first matching."""
    predicted_array = np.asarray(predicted, dtype=np.int64)
    possible: list[tuple[int, int, int, int, int, int]] = []
    for expected_index, expected_frame in enumerate(expected):
        for prediction_index in np.flatnonzero(
            np.abs(predicted_array - expected_frame) <= tolerance
        ):
            offset = int(predicted_array[prediction_index]) - int(expected_frame)
            possible.append(
                (
                    abs(offset),
                    int(expected_frame),
                    int(predicted_array[prediction_index]),
                    expected_index,
                    int(prediction_index),
                    offset,
                )
            )
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    result: list[tuple[int, int, int]] = []
    for _, _, _, expected_index, prediction_index, offset in sorted(possible):
        if expected_index in used_expected or prediction_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(prediction_index)
        result.append((expected_index, prediction_index, offset))
    return result


def _row_record(
    row: pd.Series,
    *,
    set_id: str,
    rally: int,
    source_csv: str,
    status: str,
    frame: int | None,
    removal_reasons: Sequence[str],
) -> dict[str, Any]:
    """Return one source-row record with the CSV provenance convention."""
    raw_timestamp = _json_safe(row["frame_num"])
    return {
        "set_id": set_id,
        "rally": rally,
        "source_csv": source_csv,
        # Zero-based data-row index from pandas, excluding the CSV header.
        "source_row_index": int(row["_source_row_index"]),
        "ball_round": _json_safe(row["ball_round"]),
        "raw_timestamp": raw_timestamp,
        "raw_frame_num": raw_timestamp,
        "frame": frame,
        "flaw": _json_safe(row["flaw"]),
        "status": status,
        "row_status": status,
        "removal_reasons": list(removal_reasons),
    }


def _rally_reason_string(reasons: Sequence[str]) -> str:
    """Keep one readable compatibility field while retaining separate causes."""
    return "+".join(reasons)


def _dropped_row_records(
    group: pd.DataFrame,
    *,
    set_id: str,
    rally: int,
    frames: pd.Series,
    invalid: pd.Series,
    flawed: pd.Series,
    reasons: Sequence[str],
) -> list[dict[str, Any]]:
    """Record every row in a dropped rally, including uncertain anchors."""
    records: list[dict[str, Any]] = []
    for row_index, row in group.iterrows():
        row_invalid = bool(invalid.loc[row_index])
        row_flawed = bool(flawed.loc[row_index])
        row_reasons = []
        if row_flawed:
            row_reasons.append("flaw")
        if row_invalid:
            row_reasons.append("invalid_frame")
        if row_reasons:
            status = (
                "removed_flaw_and_invalid_frame"
                if len(row_reasons) == 2
                else f"removed_{row_reasons[0]}"
            )
            row_frame = None
        else:
            status = "uncertain_anchor"
            row_frame = int(frames.loc[row_index])
        record = _row_record(
            row,
            set_id=set_id,
            rally=rally,
            source_csv=str(row["_source_csv"]),
            status=status,
            frame=row_frame,
            removal_reasons=row_reasons,
        )
        record["rally_removal_reasons"] = list(reasons)
        record["rally_removal_reason"] = _rally_reason_string(reasons)
        records.append(record)
    return records


def clean_and_dropped(
    set_directory: Path, frame_count: int
) -> tuple[list[int], list[dict[str, Any]], Counter]:
    """Mirror cleaning while retaining source rows from dropped rallies.

    Source row indices are zero-based indices in each CSV's data rows. A
    valid, unflagged, in-range row in a dropped rally receives the explicit
    ``uncertain_anchor`` status and remains excluded from trusted labels.
    """
    paths = sorted(set_directory.glob("set*.csv"))
    if not paths:
        raise FileNotFoundError(f"No set*.csv files under {set_directory}")
    tables: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_csv(path)
        required = {"rally", "ball_round", "frame_num", "flaw"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        table = table.copy()
        table["_source_csv"] = path.name
        table["_source_row_index"] = np.arange(len(table), dtype=np.int64)
        table["set_id"] = path.stem
        tables.append(table)

    contacts = pd.concat(tables, ignore_index=True)
    frames = pd.to_numeric(contacts["frame_num"], errors="coerce")
    invalid = frames.isna() | (frames < 0) | (frames >= frame_count)
    flawed = contacts["flaw"].notna()
    counts = Counter(source_contact_rows=len(contacts))
    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    for (set_id_value, rally_value), group in contacts.groupby(
        ["set_id", "rally"], sort=True
    ):
        set_id = str(set_id_value)
        rally = int(rally_value)
        indices = group.index
        group_invalid = invalid.loc[indices]
        group_flawed = flawed.loc[indices]
        reasons: list[str] = []
        if bool(group_flawed.any()):
            reasons.append("flaw")
        if bool(group_invalid.any()):
            reasons.append("invalid_frame")

        ordered: pd.DataFrame | None = None
        ordered_frames: list[int] = []
        if not reasons:
            ordered = group.assign(frame_num=frames.loc[indices].astype(int)).sort_values(
                ["ball_round", "frame_num"]
            )
            ordered_frames = ordered["frame_num"].astype(int).tolist()
            if any(
                right <= left for left, right in pairwise(ordered_frames)
            ):
                reasons.append("non_monotonic_frames")

        if not reasons:
            assert ordered is not None
            kept.extend(ordered_frames)
            counts["usable_rallies"] += 1
            counts["usable_contact_rows"] += len(ordered_frames)
            continue

        # If frame/flaw data are bad, order cannot be assessed reliably. This
        # preserves the whole-rally cleaning rule used by the original scorer.
        valid_anchor_mask = ~group_invalid & ~group_flawed
        anchors = frames.loc[indices][valid_anchor_mask].astype(int).tolist()
        row_records = _dropped_row_records(
            group,
            set_id=set_id,
            rally=rally,
            frames=frames,
            invalid=invalid,
            flawed=flawed,
            reasons=reasons,
        )

        counts["dropped_rallies"] += 1
        counts["dropped_contact_rows"] += len(group)
        counts["unflagged_in_range_rows_in_dropped_rallies"] += len(anchors)
        if "flaw" in reasons:
            counts["dropped_rallies_with_flaw"] += 1
        if "invalid_frame" in reasons:
            counts["dropped_rallies_with_invalid_frame"] += 1
        if "non_monotonic_frames" in reasons:
            counts["dropped_rallies_with_non_monotonic_frames"] += 1
        dropped.append(
            {
                "set_id": set_id,
                "rally": rally,
                "reason": _rally_reason_string(reasons),
                "removal_reasons": reasons,
                "source_rows": len(group),
                "rows": row_records,
                "unflagged_in_range_frames": anchors,
                "uncertain_anchor_rows": [
                    row for row in row_records if row["status"] == "uncertain_anchor"
                ],
            }
        )
    return kept, dropped, counts


def _anchor_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Select the provenance needed when an anchor falls inside a section."""
    return {
        "set_id": row["set_id"],
        "rally": row["rally"],
        "source_csv": row["source_csv"],
        "source_row_index": row["source_row_index"],
        "frame": row["frame"],
        "raw_timestamp": row["raw_timestamp"],
        "ball_round": row["ball_round"],
        "status": row["status"],
        "removal_reasons": row["removal_reasons"],
        "rally_removal_reasons": row["rally_removal_reasons"],
        "rally_removal_reason": row["rally_removal_reason"],
    }


def diagnose(annotations: Path, predictions: Path) -> dict[str, Any]:
    """Diagnose unmatched predictions against uncertain dropped-row anchors."""
    root = annotations / "set" if (annotations / "set" / "match.csv").is_file() else annotations
    match_table = pd.read_csv(root / "match.csv", usecols=["id", "video"])
    if match_table["id"].duplicated().any():
        raise ValueError("Duplicate video IDs in match.csv")
    names = dict(zip(match_table["id"].astype(int), match_table["video"].astype(str)))
    payload = read_json(predictions)
    if payload.get("schema") != "shuttleset22-contact-predictions-combined/1":
        raise ValueError("Expected the combined ShuttleSet22 prediction file")
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Prediction file has no videos")

    totals = Counter()
    video_results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for video in videos:
        video_id = int(video["video_id"])
        if video_id in seen:
            raise ValueError(f"Duplicate prediction video: {video_id}")
        seen.add(video_id)
        if abs(float(video["fps"]) - 30.0) > 1e-6:
            raise ValueError("This reproduction expects the original 30-fps ShuttleSet22 files")
        if video_id not in names:
            raise ValueError(f"Missing match.csv identity for video {video_id}")
        kept, dropped, counts = clean_and_dropped(
            root / names[video_id], int(video["frame_count"])
        )
        predicted = [int(contact["frame"]) for contact in video["contacts"]]
        if predicted != sorted(set(predicted)):
            raise ValueError(f"Video {video_id}: repeated or unordered prediction frames")

        anchor_rows = [
            row
            for rally in dropped
            for row in rally["uncertain_anchor_rows"]
        ]
        anchors = [int(row["frame"]) for row in anchor_rows]
        counts["predicted_contacts"] = len(predicted)
        details: dict[str, Any] = {
            "video_id": video_id,
            "video_name": names[video_id],
            "dropped_rallies": dropped,
            "timing": {},
            "section_records": [],
        }
        for tolerance in TOLERANCES:
            matches = greedy_matches(kept, predicted, tolerance)
            used_predictions = {prediction_index for _, prediction_index, _ in matches}
            unmatched = [
                frame
                for prediction_index, frame in enumerate(predicted)
                if prediction_index not in used_predictions
            ]
            near_dropped = greedy_matches(anchors, unmatched, tolerance)
            key = f"at_{tolerance}_frames"
            counts[f"matched_clean_contacts_{key}"] = len(matches)
            counts[f"unmatched_predictions_{key}"] = len(unmatched)
            counts[f"unmatched_near_dropped_unflagged_rows_{key}"] = len(near_dropped)
            details["timing"][str(tolerance)] = {
                "original_clean_label_matches": len(matches),
                "original_precision": len(matches) / len(predicted) if predicted else None,
                "unmatched_near_dropped_unflagged_rows": len(near_dropped),
                "cases_to_inspect": [
                    {
                        "prediction_frame": unmatched[prediction_index],
                        "raw_label_frame": anchors[anchor_index],
                        "set_id": anchor_rows[anchor_index]["set_id"],
                        "rally": anchor_rows[anchor_index]["rally"],
                        "source_csv": anchor_rows[anchor_index]["source_csv"],
                        "source_row_index": anchor_rows[anchor_index]["source_row_index"],
                        "drop_reason": next(
                            rally["reason"]
                            for rally in dropped
                            if any(
                                row["source_row_index"] == anchor_rows[anchor_index]["source_row_index"]
                                and row["source_csv"] == anchor_rows[anchor_index]["source_csv"]
                                for row in rally["uncertain_anchor_rows"]
                            )
                        ),
                        "removal_reasons": anchor_rows[anchor_index]["removal_reasons"],
                        "row_status": anchor_rows[anchor_index]["status"],
                        "offset": offset,
                    }
                    for anchor_index, prediction_index, offset in near_dropped
                ],
            }

        no_label = 0
        sections_with_dropped = 0
        for span in video["spans"]:
            span_id = int(span["span_id"])
            start = int(span["start_frame"])
            end = int(span["end_frame"])
            clean_frames = [frame for frame in kept if start <= frame < end]
            section_anchors = [
                row for row in anchor_rows if start <= int(row["frame"]) < end
            ]
            has_no_clean_labels = not clean_frames
            if has_no_clean_labels:
                no_label += 1
            if has_no_clean_labels and section_anchors:
                sections_with_dropped += 1
            details["section_records"].append(
                {
                    "video_id": video_id,
                    "span_id": span_id,
                    "start_frame": start,
                    "end_frame": end,
                    "clean_label_count": len(clean_frames),
                    "no_clean_labels": has_no_clean_labels,
                    "uncertain_anchor_count": len(section_anchors),
                    "uncertain_anchors": [
                        _anchor_identity(row) for row in section_anchors
                    ],
                }
            )
        counts["sections_without_clean_labels"] = no_label
        counts["sections_without_clean_labels_but_with_dropped_anchors"] = sections_with_dropped
        details["section_ids_without_clean_labels_but_with_dropped_anchors"] = [
            record["span_id"]
            for record in details["section_records"]
            if record["no_clean_labels"] and record["uncertain_anchor_count"]
        ]
        details["counts"] = dict(counts)
        totals.update(counts)
        video_results.append(details)

    expected = {
        "source_contact_rows": 43159,
        "usable_contact_rows": 38218,
        "usable_rallies": 3422,
        "predicted_contacts": 39994,
        "matched_clean_contacts_at_5_frames": 32243,
        "matched_clean_contacts_at_10_frames": 32603,
        "sections_without_clean_labels": 943,
    }
    full_run = len(videos) == 47
    mismatches = {
        key: {"expected": value, "observed": totals[key]}
        for key, value in expected.items()
        if full_run and totals[key] != value
    }
    return {
        "purpose": "Find predictions penalised near explicitly dropped annotation rows; do not change predictions.",
        "reviewed_repository_commit": REVIEWED_COMMIT,
        "inputs": {
            "annotations": {"role": "ShuttleSet22 annotations", "basename": annotations.name},
            "predictions": {"role": "combined predictions", "basename": predictions.name},
        },
        "video_count": len(videos),
        "prediction_source_commit": payload.get("source_commit"),
        "original_full_run_counts_match": not mismatches if full_run else None,
        "original_count_mismatches": mismatches,
        "counts": dict(totals),
        "by_video": video_results,
        "interpretation": [
            "A nearby dropped row is a case to inspect, not a proved correct prediction.",
            "Counts use one-to-one matching to unflagged, in-range rows inside dropped rallies.",
            "Uncertain anchors do not create ignored intervals or corrected precision.",
            "Do not remove all unlabelled video or use this output as an inference input.",
            "Resolve original-count mismatches before drawing conclusions from a full run.",
        ],
    }


def self_test() -> dict[str, Any]:
    """Exercise separate drop causes and the original matching examples."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "match").mkdir()
        pd.DataFrame({"id": [8], "video": ["match"]}).to_csv(
            root / "match.csv", index=False
        )
        pd.DataFrame(
            {
                "rally": [1, 1, 1, 2, 2, 3, 3, 4, 4],
                "ball_round": [1, 2, 3, 1, 2, 1, 2, 1, 2],
                "frame_num": [100, 130, 160, 300, 330, 430, "999999999999999999999999", 520, 510],
                "flaw": [None, None, None, None, "uncertain", None, None, None, None],
            }
        ).to_csv(root / "match" / "set1.csv", index=False)
        prediction_path = root / "predictions.json"
        prediction_path.write_text(
            json.dumps(
                {
                    "schema": "shuttleset22-contact-predictions-combined/1",
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
        result = diagnose(root, prediction_path)
        counts = result["counts"]
        assert counts["usable_contact_rows"] == 3
        assert counts["matched_clean_contacts_at_5_frames"] == 3
        assert counts["unmatched_near_dropped_unflagged_rows_at_5_frames"] == 3
        assert counts["sections_without_clean_labels_but_with_dropped_anchors"] == 3
        dropped = result["by_video"][0]["dropped_rallies"]
        assert dropped[0]["removal_reasons"] == ["flaw"]
        assert dropped[1]["removal_reasons"] == ["invalid_frame"]
        assert dropped[2]["removal_reasons"] == ["non_monotonic_frames"]
        invalid_row = dropped[1]["rows"][1]
        assert invalid_row["status"] == "removed_invalid_frame"
        assert invalid_row["raw_timestamp"] == "999999999999999999999999"
        assert all(
            anchor["status"] == "uncertain_anchor"
            for rally in dropped
            for anchor in rally["uncertain_anchor_rows"]
        )

    expected = [100, 130, 160]

    def timing_complete(frames: Sequence[int]) -> bool:
        return len(frames) == len(expected) == len(greedy_matches(expected, frames, 5))

    assert timing_complete([100, 130, 160])
    assert not timing_complete([100, 130])
    assert timing_complete([96, 130, 160]) and timing_complete([103, 130, 160])
    return {
        "data": "Synthetic examples only; no real model was rerun.",
        "label_coverage": {
            "clean_labels": 3,
            "predictions": 6,
            "matched": 3,
            "original_precision": 0.5,
            "unmatched_near_unflagged_rows_in_dropped_rally": 3,
        },
        "reviewed_whole_rally_target_for_same_serve": {
            "same_real_serve_with_complete_remainder": True,
            "same_real_serve_with_unrelated_later_miss": False,
        },
        "two_valid_start_candidates": {
            "candidate_frames": [96, 103],
            "both_make_timing_complete_at_5": True,
            "reviewed_target_keeps_only_closest_positive": 103,
        },
        "checks_passed": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("label_coverage.json.gz"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    else:
        if args.annotations is None or args.predictions is None:
            parser.error("Provide --annotations and --predictions, or use --self-test")
        result = diagnose(args.annotations, args.predictions)
    write_json(args.output, result)
    print(json.dumps(result.get("counts", result), indent=2, ensure_ascii=False))
    if result.get("original_count_mismatches"):
        print("WARNING: original full-run counts differ; inspect original_count_mismatches before interpreting the result.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
