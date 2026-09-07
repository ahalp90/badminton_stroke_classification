"""Prepare label-free rally-start inputs from frozen ShuttleSet22 predictions.

The saved contact detector outputs are the source of truth here.  This adapter
rebuilds candidate lists and side answers from those outputs and the checked
vision fixtures, without opening labels or running a model.
"""

from __future__ import annotations

import argparse
import lzma
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from annotator.point_winner import attribute_half
from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.check_rally_start_candidates import (
    DUPLICATE_DISTANCE_AT_30_FPS,
    build_video_candidate_lists,
)
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    VIDEO_IDS,
)
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    COURT_FILENAMES,
    POSE_FILENAMES,
    CheckedVideo,
    _link_stage_inputs,
    _span_id,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import (
    build_candidate_rows,
)
from scratch.contact_det_full_ds_fit.scripts.save_training_rally_start_inputs import (
    _candidate_frames,
    _enriched_candidates,
    _write_json,
)
from scratch.contact_det_full_ds_fit.scripts.save_training_rally_start_inputs import (
    _read_json as _read_saved_json,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_start_inputs import (
    _checked_replay_sides,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    SCORE_DTYPE,
)

PREDICTION_FILENAME = "combined_predictions.json.gz"
OUTPUT_FILENAME = "chooser_inputs.json.gz"
FEATURE_FILENAME = "contact_features.npy.xz"
COMBINED_SCHEMA = "shuttleset22-contact-predictions-combined/1"
VIDEO_SCHEMA = "contact-rally-start-shuttleset22-video/1"
RESULT_SCHEMA = "contact-rally-start-shuttleset22-inputs/1"
EXPECTED_FPS = 30.0
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_TEST_SPAN_COUNT = 3982
EXPECTED_EMPTY_SPAN_COUNT = 257


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _video_directory(root: Path, video_id: int, role: str) -> Path:
    directory_root = Path(root)
    if not directory_root.is_dir():
        raise FileNotFoundError(f"{role} root is not a directory: {directory_root}")
    matches = sorted(
        path
        for path in directory_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{video_id:02d} ")
    )
    if len(matches) != 1:
        raise ValueError(
            f"{role} root must contain one {video_id:02d} video directory; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def _required_file(directory: Path, filename: str, role: str) -> Path:
    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{role}: required file is missing: {path}")
    return path


def _required_directory(directory: Path, role: str) -> Path:
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"{role}: required directory is missing: {path}")
    return path


def _saved_video_directory(saved_root: Path, video_id: int) -> Path:
    path = Path(saved_root) / "videos" / f"ss22_{video_id:02d}"
    if not path.is_dir():
        raise FileNotFoundError(f"saved prediction video directory is missing: {path}")
    return path


def _search_intervals(result: Mapping[str, Any], fixture: str, frame_count: int) -> tuple[tuple[int, int], ...]:
    summary = _mapping(result.get("feature_summary"), f"{fixture}: feature summary")
    raw_intervals = summary.get("search_intervals")
    if not isinstance(raw_intervals, list):
        raise TypeError(f"{fixture}: search intervals must be a list")
    intervals: list[tuple[int, int]] = []
    previous_end = -1
    for raw_interval in raw_intervals:
        if (
            not isinstance(raw_interval, list)
            or len(raw_interval) != 2
            or any(type(value) is not int for value in raw_interval)
        ):
            raise ValueError(f"{fixture}: search interval differs")
        start, end = raw_interval
        if start < 0 or end <= start or end > frame_count or start < previous_end:
            raise ValueError(f"{fixture}: search intervals are not ordered and disjoint")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _load_score_rows(path: Path, fixture: str, result: Mapping[str, Any]) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if not isinstance(rows, np.ndarray) or rows.ndim != 1 or rows.dtype != SCORE_DTYPE:
        raise ValueError(f"{fixture}: saved candidate score dtype differs")
    expected_count = result.get("candidate_row_count")
    if type(expected_count) is not int or expected_count != len(rows):
        raise ValueError(f"{fixture}: saved candidate score count differs")
    expected_fixture = fixture.encode("ascii")
    if not np.all(rows["fixture"] == expected_fixture):
        raise ValueError(f"{fixture}: candidate score fixture differs")
    if not np.all(np.isclose(rows["fps"], EXPECTED_FPS, rtol=0.0, atol=1e-6)):
        raise ValueError(f"{fixture}: candidate score fps differs")
    return rows


def _validate_result(
    saved_directory: Path,
    raw_video: Mapping[str, Any],
    video_id: int,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], tuple[tuple[int, int], ...], np.ndarray]:
    fixture = str(video_id)
    result = _read_saved_json(
        _required_file(saved_directory, "result.json", fixture),
        f"{fixture}: saved prediction result",
    )
    if (
        result.get("schema") != "shuttleset22-contact-prediction-result/1"
        or result.get("status") != "complete"
        or result.get("labels_read") is not False
        or result.get("video_id") != video_id
        or result.get("fixture") != fixture
        or result.get("nearby_distance_at_30_fps") != DUPLICATE_DISTANCE_AT_30_FPS
    ):
        raise ValueError(f"{fixture}: saved prediction result identity differs")
    frame_count = result.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError(f"{fixture}: saved frame count differs")
    if raw_video.get("fixture") != fixture or raw_video.get("video_id") != video_id:
        raise ValueError(f"{fixture}: combined prediction identity differs")
    if float(raw_video.get("fps")) != EXPECTED_FPS or raw_video.get("frame_count") != frame_count:
        raise ValueError(f"{fixture}: combined prediction metadata differs")
    raw_spans = raw_video.get("spans")
    if not isinstance(raw_spans, list):
        raise TypeError(f"{fixture}: spans must be a list")
    intervals = _search_intervals(result, fixture, frame_count)
    scores_path = _required_file(saved_directory, "candidate_scores.npy.xz", fixture)
    rows = _load_score_rows(scores_path, fixture, result)
    if len(intervals) == 0 or (
        len(rows)
        and (
            int(rows["interval_id"].min()) < 0
            or int(rows["interval_id"].max()) >= len(intervals)
        )
    ):
        raise ValueError(f"{fixture}: candidate score interval identity differs")
    for interval_id, (start, end) in enumerate(intervals):
        interval_rows = rows[rows["interval_id"] == interval_id]
        if len(interval_rows) and not np.all(
            (interval_rows["frame"] >= start) & (interval_rows["frame"] < end)
        ):
            raise ValueError(f"{fixture}/{interval_id}: candidate score bounds differ")
    return result, raw_spans, intervals, rows


def _validate_contacts(
    fixture: str,
    raw_contacts: object,
    rows: np.ndarray,
    spans: Sequence[Mapping[str, Any]],
    sides: Mapping[int, str | None],
) -> list[dict[str, object]]:
    contacts = raw_contacts
    if not isinstance(contacts, list):
        raise TypeError(f"{fixture}: contacts must be a list")
    rows_by_frame = {int(row["frame"]): row for row in rows}
    if len(rows_by_frame) != len(rows):
        raise ValueError(f"{fixture}: score frames repeat")
    output: list[dict[str, object]] = []
    previous_frame = -1
    for raw_contact in contacts:
        contact = _mapping(raw_contact, f"{fixture}: saved contact")
        frame = contact.get("frame")
        if type(frame) is not int or frame <= previous_frame:
            raise ValueError(f"{fixture}: saved contact order differs")
        previous_frame = frame
        row = rows_by_frame.get(frame)
        if row is None or not bool(row["kept"]):
            raise ValueError(f"{fixture}/{frame}: saved kept flag differs")
        score = float(contact.get("contact_score"))
        if score != float(row["contact_score"]):
            raise ValueError(f"{fixture}/{frame}: saved contact score differs")
        side = prediction_io.normalise_side(contact.get("predicted_side"), fixture, frame)
        if sides.get(frame) != side:
            raise ValueError(f"{fixture}/{frame}: replayed kept side differs")
        span_id = contact.get("span_id")
        if span_id != _span_id(frame, spans):
            raise ValueError(f"{fixture}/{frame}: saved contact span differs")
        output.append(
            {
                "frame": frame,
                "interval_id": int(row["interval_id"]),
                "contact_score": score,
                "span_id": span_id,
                "predicted_side": side,
            }
        )
    expected_frames = [int(row["frame"]) for row in rows if bool(row["kept"])]
    if [int(contact["frame"]) for contact in output] != expected_frames:
        raise ValueError(f"{fixture}: saved kept-contact coverage differs")
    return output


def _stage_root(
    stage_root: Path,
    fixture: FixtureSpec,
    base_directory: Path,
    inpaint_directory: Path,
    annotation_directory: Path,
) -> None:
    input_directory = Path(stage_root) / "combined-inputs"
    input_directory.mkdir(parents=True)
    inpaint_track = _required_file(
        inpaint_directory, "shuttle_track_inpainted.npy.xz", fixture.name
    )
    (input_directory / "shuttle_track_inpainted.npy.xz").symlink_to(
        inpaint_track.resolve(strict=True)
    )
    for filename in (*POSE_FILENAMES, *COURT_FILENAMES):
        source = _required_file(base_directory, filename, fixture.name)
        (input_directory / filename).symlink_to(source.resolve(strict=True))
    checked = cast(CheckedVideo, SimpleNamespace(directory=input_directory))
    _link_stage_inputs(Path(stage_root), checked, fixture)
    annotation_parent = Path(stage_root) / "stages" / "annotation"
    annotation_parent.mkdir(parents=True)
    annotation_parent.joinpath(fixture.name).symlink_to(
        annotation_directory.resolve(strict=True), target_is_directory=True
    )


def _feature_view(work_root: Path, saved_directory: Path, fixture: str) -> None:
    source = _required_file(saved_directory, FEATURE_FILENAME, fixture)
    destination = Path(work_root) / "features" / "videos" / fixture / FEATURE_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(strict=True))


def _video_record(
    *,
    source_commit: str,
    fixture: str,
    video_id: int,
    fps: float,
    frame_count: int,
    spans: Sequence[Mapping[str, int]],
    kept_contacts: Sequence[Mapping[str, object]],
    candidate_lists: Sequence[Mapping[str, object]],
    skipped: int,
    replayed_frames: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema": VIDEO_SCHEMA,
        "status": "complete",
        "prediction_source_commit": source_commit,
        "labels_read": False,
        "fixture": fixture,
        "video_id": video_id,
        "fps": fps,
        "frame_count": frame_count,
        "video": {
            "fixture": fixture,
            "video_id": video_id,
            "fps": fps,
            "frame_count": frame_count,
        },
        "used_input_roles": [
            "saved_candidate_scores",
            "saved_contacts_and_spans",
            "saved_auto_annotation_outputs",
            "saved_full_contact_features",
            "inpainted_shuttle_track",
            "prepared_pose",
            "prepared_court",
        ],
        "feature_file": f"features/videos/{fixture}/{FEATURE_FILENAME}",
        "spans": [dict(span) for span in spans],
        "kept_contacts": [dict(contact) for contact in kept_contacts],
        "candidate_lists": [dict(candidate_list) for candidate_list in candidate_lists],
        "counts": {
            "detected_sections": len(spans),
            "sections_without_kept_contact": skipped,
            "kept_contacts": len(kept_contacts),
            "candidate_lists": len(candidate_lists),
            "candidate_entries": sum(
                len(candidate_list["candidates"]) for candidate_list in candidate_lists
            ),
            "earlier_candidate_entries": sum(
                len(candidate_list["candidates"]) - 1 for candidate_list in candidate_lists
            ),
            "distinct_replayed_frames": replayed_frames,
        },
        "generation_seconds": elapsed_seconds,
    }


def _process_video(
    *,
    work_root: Path,
    saved_root: Path,
    prepared_root: Path,
    inpaint_root: Path,
    raw_video: Mapping[str, Any],
    video_id: int,
    source_commit: str,
    side_attributor: Any,
) -> dict[str, object]:
    started = time.perf_counter()
    fixture = str(video_id)
    saved_directory = _saved_video_directory(saved_root, video_id)
    result, spans, intervals, rows = _validate_result(saved_directory, raw_video, video_id)
    frame_count = int(result["frame_count"])
    kept_frames = [int(row["frame"]) for row in rows if bool(row["kept"])]
    candidate_lists, skipped = build_video_candidate_lists(
        fixture,
        EXPECTED_FPS,
        rows,
        kept_frames,
        spans,
        intervals,
        DUPLICATE_DISTANCE_AT_30_FPS,
    )
    if any(len(candidate_list["candidates"]) != 3 for candidate_list in candidate_lists):
        raise ValueError(f"{fixture}: candidate list is not compatible with the chooser")
    replay_frames = np.asarray(
        sorted(set(kept_frames) | _candidate_frames(candidate_lists)), dtype=np.int32
    )
    base_directory = _video_directory(prepared_root, video_id, "prepared")
    inpaint_directory = _video_directory(inpaint_root, video_id, "inpaint")
    if base_directory.name != inpaint_directory.name:
        raise ValueError(f"{fixture}: prepared and inpaint directory names differ")
    annotation_directory = _required_directory(saved_directory / "annotation", fixture)
    with tempfile.TemporaryDirectory(prefix=f"stage-{fixture}-", dir=work_root) as temporary:
        stage_root = Path(temporary)
        _stage_root(
            stage_root,
            FixtureSpec(fixture, video_id, EXPECTED_FPS),
            base_directory,
            inpaint_directory,
            annotation_directory,
        )
        track, pose, court, _tracker_intervals, sticky, annotation = _load_inputs(
            stage_root, FixtureSpec(fixture, video_id, EXPECTED_FPS)
        )
        if len(track) != frame_count:
            raise ValueError(f"{fixture}: replay track frame count differs")
        annotation_spans = [
            {"span_id": index, "start_frame": start, "end_frame": end}
            for index, (start, end) in enumerate(annotation.spans)
        ]
        if annotation_spans != spans:
            raise ValueError(f"{fixture}: saved annotation spans differ")
        court_inputs = getattr(getattr(court, "evidence", None), "inputs", None)
        if court_inputs is None:
            raise ValueError(f"{fixture}: court inputs are unavailable")
        net_band = tuple(float(value) for value in court_inputs.net_band)
        if len(net_band) != 2 or not np.isfinite(net_band).all() or net_band[0] > net_band[1]:
            raise ValueError(f"{fixture}: court net band differs")
        sides = _checked_replay_sides(
            fixture,
            replay_frames,
            track,
            sticky,
            pose.bboxes,
            net_band,
            side_attributor,
        )
    kept_contacts = _validate_contacts(
        fixture, raw_video.get("contacts"), rows, spans, sides
    )
    rows_by_frame = {int(row["frame"]): row for row in rows}
    enriched_lists = _enriched_candidates(candidate_lists, rows_by_frame, sides)
    _feature_view(work_root, saved_directory, fixture)
    return _video_record(
        source_commit=source_commit,
        fixture=fixture,
        video_id=video_id,
        fps=EXPECTED_FPS,
        frame_count=frame_count,
        spans=spans,
        kept_contacts=kept_contacts,
        candidate_lists=enriched_lists,
        skipped=skipped,
        replayed_frames=len(sides),
        elapsed_seconds=time.perf_counter() - started,
    )


def prepare_broader_inputs(
    saved_root: Path,
    prepared_root: Path,
    inpaint_root: Path,
    output_root: Path,
    video_id: int | None = None,
) -> Path:
    """Build the label-free candidate input bundle and fixture feature view."""
    saved_root = Path(saved_root)
    prepared_root = Path(prepared_root)
    inpaint_root = Path(inpaint_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    combined_path = _required_file(saved_root, PREDICTION_FILENAME, "saved predictions")
    pack = prediction_io.load_frozen_test_predictions(combined_path)
    payload = _mapping(pack.payload, "combined predictions")
    if (
        payload.get("schema") != COMBINED_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("combined predictions are incomplete or expose labels")
    selected_ids = tuple(VIDEO_IDS if video_id is None else (video_id,))
    if any(selected not in VIDEO_IDS for selected in selected_ids):
        raise ValueError(f"video ID must be one of {VIDEO_IDS}")
    raw_by_id = {int(video["video_id"]): video for video in pack.videos}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True)
    records = []
    for selected in selected_ids:
        record = _process_video(
            work_root=output_root,
            saved_root=saved_root,
            prepared_root=prepared_root,
            inpaint_root=inpaint_root,
            raw_video=raw_by_id[selected],
            video_id=selected,
            source_commit=pack.source_commit,
            side_attributor=attribute_half,
        )
        records.append(record)
        print("Prepared", selected, record["counts"], flush=True)
    build_candidate_rows(records, default_group="T")
    counts = {
        "videos": len(records),
        "spans": sum(len(record["spans"]) for record in records),
        "candidate_lists": sum(len(record["candidate_lists"]) for record in records),
        "candidate_entries": sum(
            sum(len(candidate_list["candidates"]) for candidate_list in record["candidate_lists"])
            for record in records
        ),
        "sections_without_kept_contact": sum(
            int(record["counts"]["sections_without_kept_contact"]) for record in records
        ),
        "all_saved_test_spans": len(pack.spans),
    }
    if video_id is None and (
        counts["spans"] != EXPECTED_TEST_SPAN_COUNT
        or counts["sections_without_kept_contact"] != EXPECTED_EMPTY_SPAN_COUNT
    ):
        raise ValueError("saved test span or empty-section coverage differs")
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "prediction_source_commit": pack.source_commit,
        "labels_read": False,
        "saved_prediction_file": PREDICTION_FILENAME,
        "feature_root": "features",
        "video_ids": list(selected_ids),
        "prediction_provenance": {
            key: payload.get(key)
            for key in ("source_commit", "score_cutoff", "nearby_distance_at_30_fps")
        },
        "counts": counts,
        "videos": records,
    }
    _write_json(output_root / OUTPUT_FILENAME, result)
    return output_root / OUTPUT_FILENAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--inpaint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-id", type=int, choices=VIDEO_IDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = prepare_broader_inputs(
        arguments.saved_root,
        arguments.prepared_root,
        arguments.inpaint_root,
        arguments.output_root,
        arguments.video_id,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
