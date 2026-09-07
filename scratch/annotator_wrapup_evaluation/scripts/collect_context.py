"""Collect upstream context around the frozen recommended proposals."""

# Direct execution needs the path setup before project imports.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scratch.annotator_wrapup_evaluation.scripts.evaluate_saved import BASELINE
from scratch.contact_det_closing_pass.scripts.summarise_metrics import (
    load_populations,
    load_stream,
)


FRAME_WINDOW = 15
SCORE_RADIUS = 10
FEATURE_FIELDS = (
    "sticky_analysed",
    "shuttle_visible",
    "shuttle_visible_t+0",
    "pose_valid_top",
    "pose_valid_top_t+0",
    "pose_valid_bot",
    "pose_valid_bot_t+0",
    "wrist_valid_top_t+0",
    "wrist_valid_bot_t+0",
    "standing_count",
    "wrist_gap_min_t+0",
    "wrist_gap_top_t+0",
    "wrist_gap_bot_t+0",
    "region_current_raw",
    "region_relaxed_impulse",
    "region_wrist",
    "region_visibility",
    "region_rally_start",
    "region_scene_start",
    "region_serve_lookback",
)
SCORE_FIELDS = ("fixture", "interval_id", "frame", "fps", "contact_score", "kept")


def _load_npy_xz(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        return np.load(source, allow_pickle=False)


def _load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return payload


def _decode_fixture(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii")
    if isinstance(value, str):
        return value
    raise TypeError(f"fixture field has unsupported type {type(value).__name__}")


def _as_bool_vector(values: np.ndarray, name: str, frame_count: int) -> np.ndarray:
    if values.shape != (frame_count,) or values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be bool with shape ({frame_count},), got {values.shape} {values.dtype}")
    return values


def _read_court(path: Path, frame_count: int, fixture: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    evidence = _load_json(path)
    if evidence.get("schema") != "court-evidence/0.1":
        raise ValueError(f"{fixture}: unsupported court evidence schema")
    raw_cuts = evidence.get("raw_cuts")
    records = evidence.get("scene_records")
    if not isinstance(raw_cuts, list) or not isinstance(records, list) or not raw_cuts or len(records) != len(raw_cuts):
        raise ValueError(f"{fixture}: court evidence scene records and raw cuts disagree")
    intervals: list[tuple[int, int]] = []
    for item in raw_cuts:
        if not isinstance(item, list) or len(item) != 2 or any(type(value) is not int for value in item):
            raise ValueError(f"{fixture}: malformed raw cut interval")
        intervals.append((item[0], item[1]))
    expected_start = 0
    for start, end in intervals:
        if start != expected_start or end <= start or end > frame_count:
            raise ValueError(f"{fixture}: raw cuts are not contiguous and in bounds")
        expected_start = end
    if expected_start != frame_count:
        raise ValueError(f"{fixture}: raw cuts do not cover the frame count")

    scene_rows: list[dict[str, Any]] = []
    for index, (record, interval) in enumerate(zip(records, intervals, strict=True)):
        if not isinstance(record, dict):
            raise TypeError(f"{fixture}: scene record is not an object")
        if (record.get("scene_index"), record.get("start_frame"), record.get("end_frame")) != (index, *interval):
            raise ValueError(f"{fixture}: scene record {index} differs from raw cuts")
        if str(record.get("video_id")) != fixture:
            raise ValueError(f"{fixture}: scene record identity differs")
        scene_rows.append(record)
    return evidence, np.asarray(intervals, dtype=np.int64), np.asarray(scene_rows, dtype=object), intervals


def _read_receipt(path: Path, fixture: str, frame_count: int) -> float:
    receipt = _load_json(path)
    metadata = receipt.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{fixture}: court receipt has no metadata")
    if receipt.get("completed") is not True or (metadata.get("fps_numerator"), metadata.get("fps_denominator")) != (30, 1):
        raise ValueError(f"{fixture}: court receipt is not complete at exact 30 fps")
    if metadata.get("frame_count") != frame_count:
        raise ValueError(f"{fixture}: court receipt frame count differs from arrays")
    return 30.0


def _homography_rows(evidence: Mapping[str, Any], fixture: str) -> list[dict[str, Any]]:
    inputs = evidence.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError(f"{fixture}: court inputs are absent")
    raw = inputs.get("homography_rows")
    if isinstance(raw, dict) and isinstance(raw.get("columns"), list) and isinstance(raw.get("data"), list):
        columns = raw["columns"]
        return [dict(zip(columns, row, strict=True)) for row in raw["data"]]
    if isinstance(raw, list) and all(isinstance(row, dict) for row in raw):
        return [dict(row) for row in raw]
    raise ValueError(f"{fixture}: court homography rows have an unsupported shape")


def _tracker_covered(evidence: Mapping[str, Any], court_present: np.ndarray, fixture: str) -> np.ndarray:
    from annotator.rally.evidence import tracker_segments

    segments = tracker_segments(_homography_rows(evidence, fixture), court_present, len(court_present))
    covered = np.zeros(len(court_present), dtype=bool)
    for start, end in segments:
        covered[start:end] = True
    return covered


def _feature_index(path: Path, fixture: str, frame_count: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = _load_npy_xz(path)
    if values.ndim != 1 or values.dtype.names is None:
        raise ValueError(f"{fixture}: saved feature rows must be a one-dimensional structured array")
    names = set(values.dtype.names)
    if not {"fixture", "frame", "fps"} <= names:
        raise ValueError(f"{fixture}: saved feature rows lack identity fields")
    source_names = np.asarray([_decode_fixture(value) for value in values["fixture"]])
    if not np.all(source_names == fixture) or not np.allclose(values["fps"], 30.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{fixture}: saved feature identities differ")
    frames = values["frame"].astype(np.int64, copy=False)
    if np.any(frames < 0) or np.any(frames >= frame_count) or len(np.unique(frames)) != len(frames):
        raise ValueError(f"{fixture}: saved feature frames are invalid or duplicated")
    order = np.argsort(frames, kind="stable")
    return frames[order], {name: values[name][order] for name in FEATURE_FIELDS if name in names}


def _score_index(path: Path, fixture: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = _load_npy_xz(path)
    if values.ndim != 1 or values.dtype.names is None or not set(SCORE_FIELDS) <= set(values.dtype.names):
        raise ValueError(f"{fixture}: saved score rows have an unsupported schema")
    source_names = np.asarray([_decode_fixture(value) for value in values["fixture"]])
    if not np.all(source_names == fixture) or not np.allclose(values["fps"], 30.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{fixture}: saved score identities differ")
    frames = values["frame"].astype(np.int64, copy=False)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{fixture}: saved score frames are duplicated")
    return frames, {name: values[name][order] for name in SCORE_FIELDS}


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return _decode_fixture(value)
    return value


def _row_json(index: Mapping[str, np.ndarray], position: int) -> str:
    return json.dumps({name: _json_value(values[position]) for name, values in index.items()}, separators=(",", ":"), sort_keys=True)


def _score_context(frames: np.ndarray, index: Mapping[str, np.ndarray], frame: int) -> dict[str, object]:
    insertion = int(np.searchsorted(frames, frame))
    exact = insertion < len(frames) and int(frames[insertion]) == frame
    start = int(np.searchsorted(frames, frame - SCORE_RADIUS, side="left"))
    stop = int(np.searchsorted(frames, frame + SCORE_RADIUS, side="right"))
    nearby = slice(start, stop)
    distances = np.abs(frames[nearby] - frame)
    return {
        "exact_saved_score_row": _row_json(index, insertion) if exact else "",
        "nearby_saved_rows": int(stop - start),
        "max_nearby_score": float(np.max(index["contact_score"][nearby])) if stop > start else np.nan,
        "nearest_saved_row_distance": int(np.min(distances)) if len(distances) else np.nan,
        "nearby_kept_rows": int(np.count_nonzero(index["kept"][nearby])) if stop > start else 0,
    }


def _feature_context(frames: np.ndarray, index: Mapping[str, np.ndarray], frame: int) -> dict[str, object]:
    insertion = int(np.searchsorted(frames, frame))
    if insertion == len(frames):
        nearest = insertion - 1
    elif insertion == 0:
        nearest = 0
    else:
        nearest = insertion if abs(int(frames[insertion]) - frame) < abs(int(frames[insertion - 1]) - frame) else insertion - 1
    exact = insertion < len(frames) and int(frames[insertion]) == frame
    result: dict[str, object] = {
        "feature_exact_frame": int(frames[insertion]) if exact else np.nan,
        "feature_nearest_frame": int(frames[nearest]) if len(frames) else np.nan,
        "feature_nearest_row_distance": abs(int(frames[nearest]) - frame) if len(frames) else np.nan,
    }
    for name, values in index.items():
        result[name] = _json_value(values[insertion]) if exact else np.nan
    return result


def _window_fractions(values: Mapping[str, np.ndarray], frame: int) -> dict[str, float]:
    start = max(0, frame - FRAME_WINDOW)
    stop = min(len(next(iter(values.values()))), frame + FRAME_WINDOW)
    return {f"{name}_fraction": float(array[start:stop].mean()) for name, array in values.items()}


def _as_csv_value(value: object) -> object:
    value = _json_value(value)
    return "" if value is None else value


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _as_csv_value(row.get(name)) for name in fieldnames})


def _annotation_root(saved_root: Path, supplied: Path | None) -> Path:
    candidates = [] if supplied is None else [supplied]
    candidates += [saved_root / "annotations", saved_root.parent / "annotations", ROOT / "data" / "shuttleset22"]
    for candidate in candidates:
        if (candidate / "set" / "match.csv").is_file():
            return candidate
    raise FileNotFoundError("ShuttleSet22 all-source annotations are required; pass --annotations")


def _fixture_files(prepared: Path, inpainted: Path, saved: Path, fixture: str) -> dict[str, Path]:
    base_matches = list(prepared.glob(f"{int(fixture):02d} *"))
    inpaint_matches = list(inpainted.glob(f"{int(fixture):02d} *"))
    assert len(base_matches) == len(inpaint_matches) == 1, fixture
    base = base_matches[0]
    inpaint = inpaint_matches[0]
    assert base.name == inpaint.name, fixture
    frozen = saved / "videos" / f"ss22_{int(fixture):02d}"
    return {
        "track": base / "shuttle_track.npy.xz",
        "inpaint": inpaint / "shuttle_track_inpainted.npy.xz",
        "court": base / "court_evidence.json.gz",
        "keep_vote": base / "court_keep_vote.npy.xz",
        "court_present": base / "court_present.npy.xz",
        "pose_ndet": base / "pose_ndet.npy.xz",
        "receipt": base / "court_receipt.json.gz",
        "excluded": frozen / "annotation/definitive_exclusion_mask.npy.xz",
        "raw_replay": frozen / "annotation/raw_replay_mask.npy.xz",
        "features": frozen / "contact_features.npy.xz",
        "scores": frozen / "candidate_scores.npy.xz",
    }


def _context_rows(
    fixture: str,
    requested: Sequence[int],
    files: Mapping[str, Path],
    fps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    base_track = _load_npy_xz(files["track"])
    inpaint_track = _load_npy_xz(files["inpaint"])
    if base_track.ndim != 2 or base_track.shape[1] != 3 or inpaint_track.shape != base_track.shape:
        raise ValueError(f"{fixture}: shuttle tracks must both have shape (frames, 3)")
    frame_count = len(base_track)
    fps_from_receipt = _read_receipt(files["receipt"], fixture, frame_count)
    if fps != fps_from_receipt:
        raise ValueError(f"{fixture}: stream fps differs from court receipt")
    base_visible = base_track[:, 2] == 1
    filled_visible = inpaint_track[:, 2] == 1
    evidence, intervals_array, records_array, intervals = _read_court(files["court"], frame_count, fixture)
    court_present = _load_npy_xz(files["court_present"])
    keep_vote = _load_npy_xz(files["keep_vote"])
    court_present = _as_bool_vector(court_present, "court_present", frame_count)
    keep_vote = _as_bool_vector(keep_vote, "court_keep_vote", frame_count)
    expected_present = np.zeros(frame_count, dtype=bool)
    for index, (start, end) in enumerate(intervals):
        expected_present[start:end] = bool(records_array[index]["scene_valid"])
    if not np.array_equal(court_present, expected_present):
        raise ValueError(f"{fixture}: court_present differs from expanded scene_valid")
    tracker_covered = _tracker_covered(evidence, court_present, fixture)
    pose_ndet = _load_npy_xz(files["pose_ndet"])
    if pose_ndet.shape != (frame_count,) or not np.issubdtype(pose_ndet.dtype, np.number):
        raise ValueError(f"{fixture}: pose_ndet shape or dtype differs")
    excluded = _as_bool_vector(_load_npy_xz(files["excluded"]), "excluded", frame_count)
    raw_replay = _as_bool_vector(_load_npy_xz(files["raw_replay"]), "raw_replay", frame_count)
    feature_frames, features = _feature_index(files["features"], fixture, frame_count)
    score_frames, scores = _score_index(files["scores"], fixture)
    bool_values = {
        "court_present": court_present,
        "court_keep_vote": keep_vote,
        "excluded": excluded,
        "raw_replay": raw_replay,
        "base_shuttle_visible": base_visible,
        "filled_shuttle_visible": filled_visible,
        "tracker_covered": tracker_covered,
    }
    scene_ends = intervals_array[:, 1]
    internal_cuts = scene_ends[:-1]
    context_rows: list[dict[str, object]] = []
    for frame in requested:
        if not 0 <= frame < frame_count:
            raise ValueError(f"{fixture}/{frame}: requested frame lies outside the saved arrays")
        scene_index = int(np.searchsorted(scene_ends, frame, side="right"))
        scene_start, scene_end = intervals[scene_index]
        cut_distance = np.min(np.abs(internal_cuts - frame)) / fps if len(internal_cuts) else np.nan
        row: dict[str, object] = {
            "fixture": fixture,
            "source_frame": frame,
            "fps": fps,
            "scene_index": scene_index,
            "scene_start": scene_start,
            "scene_end": scene_end,
            "distance_to_nearest_cut_seconds": float(cut_distance),
            **{name: bool(values[frame]) for name, values in bool_values.items()},
            "pose_ndet": _json_value(pose_ndet[frame]),
        }
        row.update(_window_fractions(bool_values, frame))
        row.update(_score_context(score_frames, scores, frame))
        row.update(_feature_context(feature_frames, features, frame))
        context_rows.append(row)

    scenes = [
        {
            "fixture": fixture,
            "scene_index": int(record["scene_index"]),
            "start": int(record["start_frame"]),
            "end": int(record["end_frame"]),
            "scene_valid": bool(record["scene_valid"]),
            "exactly_two_count": int(record["exactly_two_count"]),
            "exactly_two_fraction": float(record["exactly_two_fraction"]),
            "raw_source": record["raw_source"],
        }
        for record in records_array
    ]
    return context_rows, scenes, {"frame_count": frame_count, "feature_rows": len(feature_frames), "score_rows": len(score_frames)}


def run(
    prepared_root: Path,
    inpainted_root: Path,
    saved_root: Path,
    output: Path,
    annotations: Path | None = None,
    limit: int | None = None,
) -> None:
    started = perf_counter()
    stream, fps_by_fixture = load_stream(BASELINE, "recommended")
    populations, _ = load_populations(_annotation_root(saved_root, annotations))
    all_labels = populations["all_gt"]
    if len(fps_by_fixture) != 47 or any(not np.isclose(fps, 30.0) for fps in fps_by_fixture.values()):
        raise ValueError("recommended stream must contain all 47 fixtures at exact 30 fps")
    requested_by_fixture: dict[str, set[int]] = {fixture: set() for fixture in fps_by_fixture}
    for fixture, rallies in all_labels.rallies.items():
        requested_by_fixture[fixture].update(frame for rally in rallies for frame in rally.frames)
    for span in stream.spans:
        requested_by_fixture[span.fixture].update((span.start_frame, span.end_frame - 1))
    for fixture, events in stream.events_by_fixture.items():
        requested_by_fixture[fixture].update(event.frame for event in events)
    fixtures = list(fps_by_fixture)[:limit] if limit is not None else list(fps_by_fixture)
    context_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    counts = {"fixtures": len(fixtures), "requested_frames": 0, "context_rows": 0, "scene_rows": 0, "feature_exact_rows": 0}
    for fixture in fixtures:
        files = _fixture_files(prepared_root, inpainted_root, saved_root, fixture)
        contexts, scenes, fixture_counts = _context_rows(
            fixture, sorted(requested_by_fixture[fixture]), files, fps_by_fixture[fixture]
        )
        print(fixture, len(contexts), "frames", fixture_counts, flush=True)
        context_rows.extend(contexts)
        scene_rows.extend(scenes)
        counts["requested_frames"] += len(requested_by_fixture[fixture])
        counts["context_rows"] += len(contexts)
        counts["scene_rows"] += len(scenes)
        counts["feature_exact_rows"] += sum(row["feature_exact_frame"] == row["source_frame"] for row in contexts)

    bool_names = ("court_present", "court_keep_vote", "excluded", "raw_replay", "base_shuttle_visible", "filled_shuttle_visible", "tracker_covered")
    base_fields = (
        "fixture", "source_frame", "fps", "scene_index", "scene_start", "scene_end", "distance_to_nearest_cut_seconds",
        *bool_names, "pose_ndet", *(f"{name}_fraction" for name in bool_names),
        "exact_saved_score_row", "nearby_saved_rows", "max_nearby_score", "nearest_saved_row_distance", "nearby_kept_rows",
        "feature_exact_frame", "feature_nearest_frame", "feature_nearest_row_distance",
    )
    feature_columns = [name for name in FEATURE_FIELDS if any(name in row for row in context_rows)]
    context_fields = (*base_fields, *feature_columns)
    _write_csv(output / "contexts.csv.gz", context_rows, context_fields)
    _write_csv(output / "scenes.csv.gz", scene_rows, ("fixture", "scene_index", "start", "end", "scene_valid", "exactly_two_count", "exactly_two_fraction", "raw_source"))
    metadata = {"schema": "annotator-wrapup-context/1", "smoke": limit is not None, **counts, "seconds": perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "metadata.json.gz", "wt", encoding="utf-8") as destination:
        json.dump(metadata, destination, sort_keys=True, separators=(",", ":"))
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--inpainted-root", type=Path, required=True)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.prepared_root, args.inpainted_root, args.saved_root, args.output, args.annotations, args.limit)


if __name__ == "__main__":
    main()
