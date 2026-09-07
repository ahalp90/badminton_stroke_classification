"""Prepare label-free later-contact candidates from frozen ShuttleSet22 scores."""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from annotator.point_winner import attribute_half
from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.features import _frozen_feature_names
from scratch.contact_det_closing_pass.scripts.prepare_broader_inputs import (
    _required_directory,
    _saved_video_directory,
    _stage_root,
    _validate_contacts,
    _validate_result,
    _video_directory,
)
from scratch.contact_det_closing_pass.scripts.prepare_later_inputs import (
    _physical_blocks,
    _sections,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    VIDEO_IDS,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_start_inputs import (
    _checked_replay_sides,
)

RESULT_SCHEMA = "contact-rally-start-later-inputs/1"
PREDICTION_FILENAME = "combined_predictions.json.gz"
OUTPUT_PATH = (
    prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass/raw/later_inputs/broader.json.gz"
)
FEATURE_ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass/raw/broader_inputs/features"
EXPECTED_FPS = 30.0
EXPECTED_WIDTH = 1920.0
EXPECTED_HEIGHT = 1080.0
GROUP = "ShuttleSet22"


def _raw_video_by_id(videos: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    by_id: dict[int, Mapping[str, Any]] = {}
    for video in videos:
        raw_id = video.get("video_id")
        if type(raw_id) is not int or raw_id in by_id:
            raise ValueError("frozen prediction video IDs repeat or have the wrong type")
        by_id[raw_id] = video
    return by_id


def _prepare_video(
    *,
    saved_root: Path,
    prepared_root: Path,
    inpainted_root: Path,
    feature_root: Path,
    raw_video: Mapping[str, Any],
    spans_by_fixture: Mapping[str, Sequence[Any]],
    video_id: int,
    output_parent: Path,
    frozen_names: tuple[str, ...],
) -> dict[str, Any]:
    fixture = str(video_id)
    started = time.perf_counter()
    saved_directory = _saved_video_directory(saved_root, video_id)
    score_started = time.perf_counter()
    _result, raw_spans, _intervals, score_rows = _validate_result(
        saved_directory, raw_video, video_id
    )
    score_load_seconds = time.perf_counter() - score_started
    sections, candidate_frames, shortlist_seconds = _sections(
        fixture, spans_by_fixture[fixture], score_rows, EXPECTED_FPS
    )
    base_directory = _video_directory(prepared_root, video_id, "prepared")
    inpaint_directory = _video_directory(inpainted_root, video_id, "inpainted")
    if base_directory.name != inpaint_directory.name:
        raise ValueError(f"{fixture}: prepared and inpainted directory names differ")
    annotation_directory = _required_directory(saved_directory / "annotation", fixture)
    # The saved full stream also contains contacts outside generated sections.
    event_frames = {int(row["frame"]) for row in score_rows if bool(row["kept"])}
    replay_frames = candidate_frames | event_frames
    side_started = time.perf_counter()
    sides: dict[int, str | None] = {}
    if replay_frames:
        with tempfile.TemporaryDirectory(
            prefix=f"stage-{fixture}-", dir=output_parent
        ) as temporary:
            stage_root = Path(temporary)
            spec = FixtureSpec(
                fixture,
                video_id,
                EXPECTED_FPS,
                EXPECTED_WIDTH,
                EXPECTED_HEIGHT,
            )
            _stage_root(
                stage_root,
                spec,
                base_directory,
                inpaint_directory,
                annotation_directory,
            )
            track, pose, court, _segments, sticky, _annotation = _load_inputs(
                stage_root, spec
            )
            sides = _checked_replay_sides(
                fixture,
                np.asarray(sorted(replay_frames), dtype=np.int32),
                track,
                sticky,
                pose.bboxes,
                court.evidence.inputs.net_band,
                attribute_half,
            )
        _validate_contacts(fixture, raw_video.get("contacts"), score_rows, raw_spans, sides)
    side_seconds = time.perf_counter() - side_started
    if candidate_frames:
        physical, physical_seconds = _physical_blocks(
            fixture, feature_root, candidate_frames, frozen_names
        )
    else:
        physical, physical_seconds = {}, 0.0
    for section in sections:
        for candidate in section["candidates"]:
            frame = int(candidate["frame"])
            candidate["predicted_side"] = sides[frame]
            candidate["physical"] = physical[frame]
    total_seconds = time.perf_counter() - started
    return {
        "fixture": fixture,
        "group": GROUP,
        "fps": EXPECTED_FPS,
        "sections": sections,
        "timings": {
            "score_load_seconds": score_load_seconds,
            "shortlist_seconds": shortlist_seconds,
            "side_seconds": side_seconds,
            "physical_seconds": physical_seconds,
            "total_seconds": total_seconds,
        },
    }


def prepare_later_broader_inputs(
    *,
    saved_root: Path,
    prepared_root: Path,
    inpainted_root: Path,
    feature_root: Path = FEATURE_ROOT,
    output_path: Path = OUTPUT_PATH,
    video_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the frozen 47-video later-input bundle without opening labels."""
    saved_root = Path(saved_root)
    prepared_root = Path(prepared_root)
    inpainted_root = Path(inpainted_root)
    output_path = Path(output_path)
    combined_path = saved_root / PREDICTION_FILENAME
    pack = prediction_io.load_frozen_test_predictions(combined_path)
    selected_ids = tuple(VIDEO_IDS if video_ids is None else video_ids)
    if len(set(selected_ids)) != len(selected_ids) or any(
        video_id not in VIDEO_IDS for video_id in selected_ids
    ):
        raise ValueError("video_ids must be unique members of the frozen 47-video set")
    raw_by_id = _raw_video_by_id(pack.videos)
    spans_by_fixture: dict[str, list[Any]] = {str(video_id): [] for video_id in selected_ids}
    for span in pack.spans:
        if span.fixture in spans_by_fixture:
            spans_by_fixture[span.fixture].append(span)
    frozen_names = _frozen_feature_names()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for video_id in selected_ids:
        record = _prepare_video(
            saved_root=saved_root,
            prepared_root=prepared_root,
            inpainted_root=inpainted_root,
            feature_root=feature_root,
            raw_video=raw_by_id[video_id],
            spans_by_fixture=spans_by_fixture,
            video_id=video_id,
            output_parent=output_path.parent,
            frozen_names=frozen_names,
        )
        records.append(record)
        candidates = sum(len(section["candidates"]) for section in record["sections"])
        print(f"Prepared {video_id}: {candidates} candidates", flush=True)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "labels_read": False,
        "physical_feature_names": list(frozen_names),
        "videos": records,
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--inpainted-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--video-ids", type=int, nargs="+", choices=VIDEO_IDS)
    arguments = parser.parse_args()
    payload = prepare_later_broader_inputs(
        saved_root=arguments.saved_root,
        prepared_root=arguments.prepared_root,
        inpainted_root=arguments.inpainted_root,
        feature_root=arguments.feature_root,
        output_path=arguments.output,
        video_ids=arguments.video_ids,
    )
    print(f"Wrote {len(payload['videos'])} later-input videos to {arguments.output}")


if __name__ == "__main__":
    main()
