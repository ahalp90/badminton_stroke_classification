"""Replay the original sequential player picker on the recorded video-17 sample."""

# Direct execution needs the path setup before project imports.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from annotator.court_evidence import _as_ref_corners, detected_court_info
from annotator.rally.evidence import _track_sticky_players, sticky_anchor, tracker_segments
from dataset_builder.vision import load_court_vision, load_npy_xz, load_pose_arrays
from preparing_data.heuristics.base import ClipContext


FIXTURE = "17"
RESOLUTION = (1920.0, 1080.0)
SAMPLE_IDS = ("V01", "V02", "V04", "V06")
FEATURE_FILENAME = "contact_features.npy.xz"


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return payload


def _json_array(value: object) -> str:
    payload = np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
    return json.dumps(payload, separators=(",", ":"))


def _prepared_folder(prepared_root: Path) -> Path:
    folders = sorted(prepared_root.glob("17 *"))
    if len(folders) != 1 or not folders[0].is_dir():
        raise ValueError("prepared root must contain exactly one '17 ...' directory")
    return folders[0]


def _receipt_frame_count(folder: Path) -> int:
    receipt = _load_json_gz(folder / "court_receipt.json.gz")
    metadata = receipt.get("metadata")
    if receipt.get("completed") is not True or not isinstance(metadata, dict):
        raise ValueError("video 17 court receipt is incomplete")
    if (metadata.get("fps_numerator"), metadata.get("fps_denominator")) != (30, 1):
        raise ValueError("video 17 court receipt is not exact 30 fps")
    frame_count = metadata.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("video 17 court receipt has no valid frame count")
    return frame_count


def _sample_targets() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_path = ROOT / "scratch" / "annotator_wrapup_evaluation" / "results" / "visual_sample.csv.gz"
    with gzip.open(sample_path, "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    centres = [row for row in rows if row.get("sample_id") in SAMPLE_IDS and int(row["fixture"]) == 17]
    if {row["sample_id"] for row in centres} != set(SAMPLE_IDS):
        raise ValueError("video-17 visual sample must contain V01, V02, V04 and V06")
    targets = []
    for row in sorted(centres, key=lambda item: SAMPLE_IDS.index(item["sample_id"])):
        centre = int(row["source_frame"])
        for offset in (-30, -15, 0, 15, 30):
            targets.append({"sample_id": row["sample_id"], "centre_frame": centre, "offset": offset, "source_frame": centre + offset})
    return {row["sample_id"]: row for row in centres}, targets


def _scene_for_frame(court: Any, frame: int) -> tuple[int, Any]:
    records = court.evidence.scene_records
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if record.start_frame <= frame < record.end_frame
    ]
    if len(matches) != 1:
        raise ValueError(f"frame {frame} does not belong to exactly one saved scene")
    return matches[0]


def _analysis_record(
    raw: Any,
    frame: int,
    ema: np.ndarray,
    halfcourt: np.ndarray,
    params: Any,
    analysis: Any,
    geometry: str,
) -> dict[str, Any]:
    effective = params.prior_weight * halfcourt + (1 - params.prior_weight) * ema
    if analysis.court_base_pos is None or analysis.filtered_to_raw is None or analysis.bboxes is None:
        return {
            "geometry": geometry, "analysis_picks": None, "raw_picks": None,
            "candidate_scores": None, "raw_indices": None, "bboxes": None,
            "court_base_pos": None, "effective_anchor": effective.tolist(),
            "candidate_distances": None, "sanity_ceiling": params.sanity_ceiling,
            "ema": ema.tolist(), "halfcourt_centre": halfcourt.tolist(),
        }
    raw_indices = np.asarray(analysis.filtered_to_raw, dtype=int)
    distances = np.linalg.norm(
        analysis.court_base_pos[:, None, :] - effective[None, :, :], axis=-1,
    )
    raw_picks = None if analysis.picks is None else [
        -1 if pick < 0 else int(raw_indices[pick]) for pick in analysis.picks
    ]
    scores = raw.scores[frame, raw_indices]
    return {
        "geometry": geometry, "ema": ema.tolist(), "halfcourt_centre": halfcourt.tolist(),
        "analysis_picks": None if analysis.picks is None else list(map(int, analysis.picks)),
        "raw_picks": raw_picks,
        "candidate_scores": scores.tolist(),
        "raw_indices": raw_indices.tolist(),
        "bboxes": analysis.bboxes.tolist(),
        "court_base_pos": analysis.court_base_pos.tolist(),
        "effective_anchor": effective.tolist(),
        "candidate_distances": distances.tolist(),
        "sanity_ceiling": params.sanity_ceiling,
    }


def _feature_check(saved_root: Path, replay: Any, frame_count: int) -> int:
    path = saved_root / "videos" / "ss22_17" / FEATURE_FILENAME
    features = load_npy_xz(path)
    required = {"fixture", "frame", "fps", "pose_valid_top_t+0", "pose_valid_bot_t+0"}
    if features.ndim != 1 or features.dtype.names is None or not required <= set(features.dtype.names):
        raise ValueError("saved video-17 features lack pose-validity fields")
    names = np.char.decode(features["fixture"], "ascii")
    if not np.all(names == FIXTURE) or not np.all(features["fps"] == 30.0):
        raise ValueError("saved video-17 feature identity differs")
    frames = features["frame"].astype(np.int64, copy=False)
    if np.any(frames < 0) or np.any(frames >= frame_count) or len(np.unique(frames)) != len(frames):
        raise ValueError("saved video-17 feature frames are invalid")
    expected = replay.picks[frames] >= 0
    actual = np.column_stack((features["pose_valid_top_t+0"], features["pose_valid_bot_t+0"]))
    if not np.isin(actual, (0.0, 1.0)).all() or not np.array_equal(actual.astype(bool), expected):
        raise ValueError("replayed raw pose picks differ from every saved t+0 pose-validity row")
    return len(features)


def run(prepared_root: Path, saved_root: Path, output: Path) -> None:
    started = perf_counter()
    folder = _prepared_folder(prepared_root)
    frame_count = _receipt_frame_count(folder)
    pose = load_pose_arrays(folder, frame_count)
    court = load_court_vision(folder, video_id=FIXTURE, frame_count=frame_count, resolution=RESOLUTION)
    inputs = court.evidence.inputs
    if inputs is None:
        raise ValueError("video 17 court inputs are absent")
    segments = tracker_segments(inputs.homography_rows.to_dict("records"), court.evidence.court_present, frame_count)
    samples, targets = _sample_targets()
    target_frames = {row["source_frame"] for row in targets}
    observations: dict[tuple[int, str], dict[str, Any]] = {}
    original_analyse = sticky_anchor.analyse_frame

    def observe(raw: Any, frame: int, ema: np.ndarray, halfcourt: np.ndarray, ctx: ClipContext, params: Any) -> Any:
        analysis = original_analyse(raw, frame, ema, halfcourt, ctx, params)
        if frame not in target_frames:
            return analysis
        observations[frame, "saved"] = _analysis_record(raw, frame, ema.copy(), halfcourt.copy(), params, analysis, "saved")
        _, scene = _scene_for_frame(court, frame)
        if scene.raw_corners_px is not None:
            raw_info = detected_court_info(_as_ref_corners(np.asarray(scene.raw_corners_px), RESOLUTION))
            alternate_ctx = ClipContext(ctx.vid, dict(ctx.all_court_info), ctx.res_df.copy())
            alternate_ctx.all_court_info[ctx.vid] = raw_info
            alternate = original_analyse(raw, frame, ema.copy(), halfcourt.copy(), alternate_ctx, params)
            observations[frame, "raw_probe"] = _analysis_record(
                raw, frame, ema.copy(), halfcourt.copy(), params, alternate, "raw_probe"
            )
        return analysis

    sticky_anchor.analyse_frame = observe
    try:
        replay = _track_sticky_players(
            frame_count, segments, pose.bboxes, pose.scores, pose.kps, pose.ndet,
            FIXTURE, inputs.gate_court_info, inputs.gate_resolution_table, RESOLUTION,
        )
    finally:
        sticky_anchor.analyse_frame = original_analyse

    feature_count = _feature_check(saved_root, replay, frame_count)
    csv_rows = []
    for target in targets:
        frame = target["source_frame"]
        scene_index, scene = _scene_for_frame(court, frame)
        saved = observations.get((frame, "saved"), {})
        probe = observations.get((frame, "raw_probe"), {})
        csv_rows.append({
            **target, "fps": 30.0, "scene_index": scene_index, "scene_start": scene.start_frame, "scene_end": scene.end_frame,
            "replay_analysed": bool(replay.analysed[frame]), "replay_picks_raw": _json_array(replay.picks[frame]),
            "saved_analysis": json.dumps(saved, separators=(",", ":"), sort_keys=True) if saved else "",
            "raw_probe_analysis": json.dumps(probe, separators=(",", ":"), sort_keys=True) if probe else "",
            "raw_probe_available": bool(probe),
        })
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(csv_rows[0])
    with gzip.open(output / "replay_player_sample.csv.gz", "wt", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    centre_rows = []
    for sample_id, row in samples.items():
        centre = int(row["source_frame"])
        probe = observations.get((centre, "raw_probe"))
        centre_rows.append({"sample_id": sample_id, "centre_frame": centre, "raw_probe_available": probe is not None,
                            "raw_probe_picks": None if probe is None else probe["raw_picks"],
                            "replay_picks": replay.picks[centre].tolist(), "replay_analysed": bool(replay.analysed[centre])})
    result = {
        "schema": "annotator-wrapup-player-replay/1", "fixture": FIXTURE, "fps": 30.0,
        "frame_count": frame_count, "segment_count": len(segments), "sample_ids": list(SAMPLE_IDS),
        "target_frame_count": len(targets),
        "observed_original_frames": sum(key[1] == "saved" for key in observations),
        "raw_probe_frames": sum(key[1] == "raw_probe" for key in observations),
        "replay_analysed_target_frames": sum(bool(replay.analysed[row["source_frame"]]) for row in targets),
        "saved_feature_rows_checked": feature_count, "centre_observations": centre_rows,
        "reconstructed_sequential_picker": True, "alternate_geometry_is_one_frame_probe": True,
        "seconds": perf_counter() - started,
    }
    with gzip.open(output / "replay_player_sample.json.gz", "wt", encoding="utf-8") as destination:
        json.dump(result, destination, sort_keys=True, separators=(",", ":"))
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "annotator_wrapup_evaluation" / "results")
    args = parser.parse_args()
    run(args.prepared_root, args.saved_root, args.output)


if __name__ == "__main__":
    main()
