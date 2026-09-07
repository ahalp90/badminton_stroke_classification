"""Census why unmatched development labels have no nearby detector contact."""

# Direct execution needs the path setup before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct script execution needs the repo path before this import.
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE  # noqa: E402

D_FIXTURES = (
    "sset_01",
    "sset_02",
    "sset_03",
    "sset_04",
    "sset_05",
    "sset_06",
    "sset_07",
    "sset_08",
    "sset_11",
    "sset_13",
    "sset_14",
    "sset_15",
    "sset_16",
    "sset_17",
    "sset_19",
    "sset_20",
    "sset_21",
    "sset_23",
    "sset_26",
    "sset_28",
    "sset_29",
    "sset_32",
    "sset_33",
    "sset_34",
    "sset_35",
    "sset_36",
    "sset_37",
    "sset_38",
    "sset_41",
    "sset_42",
    "sset_43",
    "sset_44",
)
TOLERANCES_BASE30 = (10, 5)
SCORE_CUTOFF = 0.9


def _read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _fixture_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _frame_rows(rows: np.ndarray, target_frame: int, tolerance: int) -> np.ndarray:
    distances = np.abs(rows["frame"].astype(np.int64) - int(target_frame))
    return rows[distances <= tolerance]


def _row_identity(row: np.void) -> tuple[int, int]:
    return int(row["interval_id"]), int(row["frame"])


def _row_detail(row: np.void, target_frame: int, *, score: bool = False) -> dict[str, object]:
    frame = int(row["frame"])
    detail: dict[str, object] = {
        "frame": frame,
        "offset": frame - int(target_frame),
        "distance": abs(frame - int(target_frame)),
        "interval_id": int(row["interval_id"]),
    }
    if "fps" in row.dtype.names:
        detail["fps"] = float(row["fps"])
    if "fixture" in row.dtype.names:
        detail["fixture"] = _fixture_name(row["fixture"])
    if score:
        detail["score"] = float(row["contact_score"])
        detail["kept"] = bool(row["kept"])
    return detail


def _nearest(rows: np.ndarray, target_frame: int, *, score: bool = False) -> dict[str, object] | None:
    if len(rows) == 0:
        return None
    distances = np.abs(rows["frame"].astype(np.int64) - int(target_frame))
    nearest_index = min(
        range(len(rows)),
        key=lambda index: (int(distances[index]), int(rows[index]["frame"]), index),
    )
    return _row_detail(rows[nearest_index], target_frame, score=score)


def _selected_identity_set(rows: np.ndarray) -> set[tuple[int, int]]:
    return {_row_identity(row) for row in rows}


def _population_match_count(
    matched_by_fixture: Mapping[str, set[int]],
    fixtures: Sequence[str] = D_FIXTURES,
) -> int:
    return sum(len(matched_by_fixture.get(fixture, set())) for fixture in fixtures)


def classify_missed_contact(
    target_frame: int,
    tolerance_frames: int,
    frozen_rows: np.ndarray,
    selected_rows: np.ndarray,
    score_rows: np.ndarray,
    *,
    score_cutoff: float = SCORE_CUTOFF,
) -> dict[str, object]:
    """Classify one missed frame using the frozen-row pipeline stages.

    ``frozen_rows`` contains every feature row for one fixture. ``selected_rows``
    is the same fixture after the seven label-blind region masks, and
    ``score_rows`` contains the saved score stream for that fixture.
    """
    nearby_frozen = _frame_rows(frozen_rows, target_frame, tolerance_frames)
    nearby_selected = _frame_rows(selected_rows, target_frame, tolerance_frames)
    nearby_scores = _frame_rows(score_rows, target_frame, tolerance_frames)
    score_identities = _selected_identity_set(nearby_scores)
    missing_score_rows = [
        row for row in nearby_selected if _row_identity(row) not in score_identities
    ]

    if len(nearby_frozen) == 0:
        category = "no_nearby_frozen_row"
    elif len(nearby_selected) == 0:
        category = "nearby_but_unselected"
    elif missing_score_rows:
        category = "unexpected_unscored_selected_row"
    elif np.any(nearby_scores["kept"]):
        category = "matching_competition"
    elif np.any(nearby_scores["contact_score"] >= score_cutoff):
        category = "suppression"
    else:
        category = "below_cutoff"

    return {
        "category": category,
        "nearby_counts": {
            "frozen": len(nearby_frozen),
            "selected": len(nearby_selected),
            "scores": len(nearby_scores),
        },
        "nearest_frozen_row": _nearest(nearby_frozen, target_frame),
        "nearest_selected_row": _nearest(nearby_selected, target_frame),
        "nearest_score_row": _nearest(nearby_scores, target_frame, score=True),
        "missing_score_rows": [
            _row_detail(row, target_frame) for row in missing_score_rows
        ],
    }


def _load_score_rows(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if any(
        name not in (rows.dtype.names or ()) or rows.dtype[name] != SCORE_DTYPE[name]
        for name in SCORE_DTYPE.names
    ):
        raise ValueError(f"Saved score dtype differs: {rows.dtype}")
    return rows


def _fixture_score_rows(scores: np.ndarray, fixture: str) -> np.ndarray:
    return scores[scores["fixture"] == fixture.encode("ascii")]


def _load_feature_rows(feature_root: Path, fixtures: Sequence[str]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for fixture in fixtures:
        path = feature_root / "videos" / fixture / "contact_features.npy.xz"
        with lzma.open(path, "rb") as source:
            rows = np.load(source, allow_pickle=False)
        actual_fields = set(rows.dtype.names or ())
        required_fields = {
            "fixture",
            "interval_id",
            "frame",
            "fps",
        }
        if len(rows) == 0 or not required_fields <= actual_fields:
            raise ValueError(f"{fixture}: frozen feature rows are incomplete")
        output[fixture] = rows
    return output


def _selected_feature_rows(feature_rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
        _seeded_rows,
    )

    return {fixture: rows[_seeded_rows(rows)] for fixture, rows in feature_rows.items()}


def _verify_score_alignment(
    scores: np.ndarray,
    events_by_fixture: Mapping[str, Sequence[Any]],
    canonical_path: Path,
) -> dict[str, object]:
    kept_frames: dict[str, list[int]] = {}
    for fixture in D_FIXTURES:
        fixture_rows = _fixture_score_rows(scores, fixture)
        actual = [int(frame) for frame in fixture_rows["frame"][fixture_rows["kept"]]]
        expected = [int(event.frame) for event in events_by_fixture[fixture]]
        kept_frames[fixture] = actual
        if actual != expected:
            alternate_result = canonical_path.parent.parent / "combined_first" / "training_video_score_result.json"
            alternate_record = _read_json(alternate_result) if alternate_result.exists() else {}
            raise ValueError(
                f"canonical score/event alignment differs for {fixture}; "
                f"inspected combined_first source record {alternate_result.name}: "
                f"schema={alternate_record.get('schema')!r}, "
                f"score_row_count={alternate_record.get('score_row_count')!r}; "
                "no alternate stream was selected"
            )
    return {"canonical_score_file": canonical_path.name, "kept_frames_match": True}


def _corrected_pairs(result: Mapping[str, Any], tolerance: int) -> dict[str, set[int]]:
    tolerance_result = result["tolerances"][str(tolerance)]
    corrected = tolerance_result["corrected"]["contacts"]["by_video"]
    pairs_by_fixture: dict[str, set[int]] = {}
    for row in corrected:
        fixture = str(row["fixture"])
        pairs_by_fixture[fixture] = {int(pair[0]) for pair in row["pairs"]}
    return pairs_by_fixture


def _section_records(
    result: Mapping[str, Any],
    tolerance: int,
    missed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    sections = result["tolerances"][str(tolerance)]["corrected"]["sections"]
    counts: Counter[tuple[str, int]] = Counter()
    for row in missed:
        fixture = str(row["fixture"])
        frame = int(row["frame"])
        for section in sections:
            if (
                section["fixture"] == fixture
                and int(section["start_frame"]) <= frame < int(section["end_frame"])
            ):
                counts[(fixture, int(section["span_id"]))] += 1
    output = []
    for section in sections:
        identity = (str(section["fixture"]), int(section["span_id"]))
        if identity not in counts:
            continue
        output.append(
            {
                "fixture": identity[0],
                "span_id": identity[1],
                "start_frame": int(section["start_frame"]),
                "end_frame": int(section["end_frame"]),
                "missed_contact_count": counts[identity],
                "retained_event_count": int(section["events"]),
                "no_retained_labels": (
                    int(section["overlapping_rallies"]) == 0
                    or int(section["labelled_contacts"]) == 0
                ),
            }
        )
    return output


def build_census(
    matching_path: Path,
    score_path: Path,
    feature_root: Path,
    labels_path: Path,
) -> dict[str, object]:
    """Build the D-only census from frozen scores, features, predictions and labels."""
    from annotator.fps_constants import ScalingKind
    from scratch.contact_det_followup.scripts.prediction_io import (
        load_development_predictions,
    )
    from scratch.contact_det_full_ds_fit.scripts.run_rally_start_model import (
        load_human_labels,
    )

    matching = _read_json(matching_path)
    if matching.get("schema") != "contact-closing-matching/1":
        raise ValueError("matching result has an unexpected schema")
    pack = load_development_predictions()
    d_videos = tuple(video for video in pack.videos if video.fixture in D_FIXTURES)
    if tuple(video.fixture for video in d_videos) != D_FIXTURES:
        raise ValueError("development prediction pack does not cover A-D32")
    labels = load_human_labels(labels_path, d_videos)
    d_events = {fixture: pack.events_by_fixture[fixture] for fixture in D_FIXTURES}
    scores = _load_score_rows(score_path)
    alignment = _verify_score_alignment(scores, d_events, score_path)
    feature_rows = _load_feature_rows(feature_root, D_FIXTURES)
    selected_rows = _selected_feature_rows(feature_rows)
    score_by_fixture = {
        fixture: _fixture_score_rows(scores, fixture)
        for fixture in D_FIXTURES
    }
    fps_by_fixture = {video.fixture: float(video.fps) for video in d_videos}
    output: dict[str, object] = {
        "schema": "contact-closing-missed-candidate-census/1",
        "dataset": "ShuttleSet development",
        "population": {"group": "A-D", "video_count": len(D_FIXTURES), "fixtures": D_FIXTURES},
        "inputs": {
            "matching_result": matching_path.name,
            "scores": score_path.name,
            "labels": labels_path.name,
            "feature_filename": "contact_features.npy.xz",
        },
        "score_alignment": alignment,
        "tolerances": {},
    }
    for tolerance_base30 in TOLERANCES_BASE30:
        tolerance_by_fixture = {
            fixture: int(ScalingKind.FRAME_COUNT.scale(tolerance_base30, fps_by_fixture[fixture]))
            for fixture in D_FIXTURES
        }
        matched = _corrected_pairs(matching, tolerance_base30)
        missed: list[dict[str, object]] = []
        for fixture in D_FIXTURES:
            for rally in labels.rallies[fixture]:
                for index, frame in enumerate(rally.frames):
                    if int(frame) in matched[fixture]:
                        continue
                    analysis = classify_missed_contact(
                        int(frame),
                        tolerance_by_fixture[fixture],
                        feature_rows[fixture],
                        selected_rows[fixture],
                        score_by_fixture[fixture],
                    )
                    missed.append(
                        {
                            "fixture": fixture,
                            "rally_id": rally.rally_id,
                            "frame": int(frame),
                            "is_first": index == 0,
                            "tolerance_frames": tolerance_by_fixture[fixture],
                            **analysis,
                        }
                    )
        counts = Counter(str(row["category"]) for row in missed)
        first_count = sum(bool(row["is_first"]) for row in missed)
        tolerance_output = {
            "base30_frames": tolerance_base30,
            "source_frame_tolerance_by_fixture": tolerance_by_fixture,
            "labelled_contacts": sum(len(rally.frames) for rallies in labels.rallies.values() for rally in rallies),
            "matched_contacts": _population_match_count(matched),
            "missed_contacts": len(missed),
            "first_missed_contacts": first_count,
            "later_missed_contacts": len(missed) - first_count,
            "categories": dict(sorted(counts.items())),
            "unexpected_unscored_selected_row_count": counts[
                "unexpected_unscored_selected_row"
            ],
            "sections_with_misses": _section_records(matching, tolerance_base30, missed),
            "missed": missed,
        }
        output["tolerances"][str(tolerance_base30)] = tolerance_output
        print(
            f"tol={tolerance_base30} labelled={tolerance_output['labelled_contacts']} "
            f"matched={tolerance_output['matched_contacts']} missed={len(missed)} "
            f"first={first_count} later={len(missed) - first_count} "
            f"categories={dict(sorted(counts.items()))}",
            flush=True,
        )
    return output


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(value, target, allow_nan=False)
        target.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matching-result",
        type=Path,
        default=REPO_ROOT / "scratch/contact_det_closing_pass/results/matching_development.json.gz",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/training_video_scores/combined/training_video_scores.npy.xz",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "scratch/contact_det_closing_pass/results/missed_candidate_census.json.gz",
    )
    args = parser.parse_args()
    result = build_census(args.matching_result, args.scores, args.feature_root, args.labels)
    _write_result(args.output, result)
    unexpected = sum(
        int(tolerance_result["unexpected_unscored_selected_row_count"])
        for tolerance_result in result["tolerances"].values()
    )
    if unexpected:
        raise SystemExit(
            f"found {unexpected} unexpected unscored selected rows; "
            f"diagnostic result retained at {args.output}"
        )


if __name__ == "__main__":
    main()
