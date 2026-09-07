"""Expand the label-free broader chooser inputs to four early candidates."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from annotator.point_winner import attribute_half
from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det_closing_pass.scripts.early_shortlist import (
    MAX_EARLY_CANDIDATES,
    expand_early_shortlist,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.prepare_broader_inputs import (
    _required_directory,
    _saved_video_directory,
    _stage_root,
    _validate_result,
    _video_directory,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    VIDEO_IDS,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_start_inputs import (
    _checked_replay_sides,
)

ROOT = prediction_io.REPO_ROOT / "scratch/contact_det_closing_pass"
DEFAULT_CHOOSER_INPUTS = ROOT / "raw/broader_inputs/chooser_inputs.json.gz"
DEFAULT_OUTPUT = ROOT / "raw/followups/early_broader_inputs.json.gz"
COMBINED_PREDICTIONS = "combined_predictions.json.gz"

SideLoader = Callable[[Sequence[int]], Mapping[int, str | None]]


def _candidate_frames(video: Mapping[str, Any]) -> set[int]:
    return {
        int(candidate["frame"])
        for candidate_list in video["candidate_lists"]
        for candidate in candidate_list["candidates"]
    }


def _expand_video(
    video: Mapping[str, Any], score_rows: np.ndarray, load_sides: SideLoader,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Expand one saved chooser record and replay sides for appended frames only."""
    original_frames = _candidate_frames(video)
    placeholders = dict.fromkeys(int(frame) for frame in score_rows["frame"])
    expanded, counts = expand_early_shortlist(video, score_rows, placeholders)
    added_frames = _candidate_frames(expanded) - original_frames
    sides = dict(load_sides(sorted(added_frames))) if added_frames else {}
    if set(sides) != added_frames:
        raise ValueError("side replay does not cover every appended candidate")
    for candidate_list in expanded["candidate_lists"]:
        for candidate in candidate_list["candidates"]:
            frame = int(candidate["frame"])
            if frame in added_frames:
                candidate["predicted_side"] = sides[frame]

    candidate_entries = sum(
        len(candidate_list["candidates"])
        for candidate_list in expanded["candidate_lists"]
    )
    expanded_counts = dict(expanded.get("counts", {}))
    expanded_counts.update(
        candidate_entries=candidate_entries,
        earlier_candidate_entries=sum(
            not bool(candidate["is_fixed_contact"])
            for candidate_list in expanded["candidate_lists"]
            for candidate in candidate_list["candidates"]
        ),
        distinct_replayed_frames=(
            int(expanded_counts.get("distinct_replayed_frames", 0)) + len(added_frames)
        ),
    )
    expanded["counts"] = expanded_counts
    return expanded, {
        "candidate_lists": int(counts["candidate_lists"]),
        "sections_with_additions": int(counts["sections_with_additions"]),
        "added_earlier_candidates": int(counts["added_earlier_candidates"]),
        "candidate_entries_before": len(original_frames),
        "candidate_entries_after": candidate_entries,
    }


def _replay_sides(
    *,
    fixture: str,
    video_id: int,
    fps: float,
    frame_count: int,
    frames: Sequence[int],
    saved_directory: Path,
    prepared_root: Path,
    inpaint_root: Path,
    temporary_parent: Path,
) -> Mapping[int, str | None]:
    base_directory = _video_directory(prepared_root, video_id, "prepared")
    inpaint_directory = _video_directory(inpaint_root, video_id, "inpaint")
    if base_directory.name != inpaint_directory.name:
        raise ValueError(f"{fixture}: prepared and inpaint directory names differ")
    annotation_directory = _required_directory(saved_directory / "annotation", fixture)
    with tempfile.TemporaryDirectory(
        prefix=f"stage-{fixture}-", dir=temporary_parent
    ) as temporary:
        stage_root = Path(temporary)
        spec = FixtureSpec(fixture, video_id, fps)
        _stage_root(
            stage_root, spec, base_directory, inpaint_directory, annotation_directory
        )
        track, pose, court, _segments, sticky, _annotation = _load_inputs(stage_root, spec)
        if len(track) != frame_count:
            raise ValueError(f"{fixture}: replay track frame count differs")
        net_band = court.evidence.inputs.net_band
        return _checked_replay_sides(
            fixture,
            np.asarray(sorted(frames), dtype=np.int32),
            track,
            sticky,
            pose.bboxes,
            net_band,
            attribute_half,
        )


def _prepare_video(
    video_id: int,
    raw_video: Mapping[str, Any],
    chooser_video: Mapping[str, Any],
    saved_root: Path,
    prepared_root: Path,
    inpaint_root: Path,
    temporary_parent: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Prepare one video using only its saved score and replay artefacts."""
    fixture = str(video_id)
    saved_directory = _saved_video_directory(saved_root, video_id)
    result, _spans, _intervals, score_rows = _validate_result(
        saved_directory, raw_video, video_id
    )
    frame_count = int(result["frame_count"])

    def load_sides(frames: Sequence[int]) -> Mapping[int, str | None]:
        return _replay_sides(
            fixture=fixture,
            video_id=video_id,
            fps=float(raw_video["fps"]),
            frame_count=frame_count,
            frames=frames,
            saved_directory=saved_directory,
            prepared_root=prepared_root,
            inpaint_root=inpaint_root,
            temporary_parent=temporary_parent,
        )

    return _expand_video(chooser_video, score_rows, load_sides)


def prepare_early_broader_inputs(
    *,
    saved_root: Path,
    prepared_root: Path,
    inpaint_root: Path,
    chooser_inputs: Path = DEFAULT_CHOOSER_INPUTS,
    output: Path = DEFAULT_OUTPUT,
    video_ids: Sequence[int] | None = None,
    jobs: int = 4,
) -> dict[str, Any]:
    """Write a label-free broader input bundle with four earlier candidates."""
    started = perf_counter()
    saved_root = Path(saved_root)
    prepared_root = Path(prepared_root)
    inpaint_root = Path(inpaint_root)
    chooser_inputs = Path(chooser_inputs)
    output = Path(output)
    if jobs <= 0:
        raise ValueError("--jobs must be positive")
    chooser_payload = prediction_io.read_json(chooser_inputs)
    if (
        chooser_payload.get("schema") != "contact-rally-start-shuttleset22-inputs/1"
        or chooser_payload.get("status") != "complete"
        or chooser_payload.get("labels_read") is not False
    ):
        raise ValueError("saved chooser inputs are incomplete or used labels")
    chooser_videos = chooser_payload.get("videos")
    if not isinstance(chooser_videos, list):
        raise TypeError("saved chooser videos must be a list")

    pack = prediction_io.load_frozen_test_predictions(
        saved_root / COMBINED_PREDICTIONS
    )
    raw_by_fixture = {str(video["fixture"]): video for video in pack.videos}
    requested = tuple(VIDEO_IDS if video_ids is None else video_ids)
    if len(set(requested)) != len(requested) or any(
        video_id not in VIDEO_IDS for video_id in requested
    ):
        raise ValueError("video_ids must be unique members of the frozen 47-video set")
    chooser_by_fixture_all = {
        str(video["video"]["fixture"]): video for video in chooser_videos
    }
    if len(chooser_by_fixture_all) != len(chooser_videos):
        raise ValueError("chooser inputs contain duplicate fixtures")
    selected_fixtures = {str(video_id) for video_id in requested}
    if not selected_fixtures <= set(chooser_by_fixture_all):
        raise ValueError("chooser inputs do not cover the requested videos")
    chooser_by_fixture = {
        fixture: chooser_by_fixture_all[fixture] for fixture in selected_fixtures
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    totals = {
        "videos": len(requested),
        "candidate_lists": 0,
        "candidate_entries_before": 0,
        "candidate_entries_after": 0,
        "added_earlier_candidates": 0,
        "sections_with_additions": 0,
    }
    with joblib.parallel_config(backend="loky", n_jobs=jobs):
        prepared = joblib.Parallel()(
            joblib.delayed(_prepare_video)(
                video_id,
                raw_by_fixture[str(video_id)],
                chooser_by_fixture[str(video_id)],
                saved_root,
                prepared_root,
                inpaint_root,
                output.parent,
            )
            for video_id in requested
        )

    for video_id, (expanded, counts) in zip(requested, prepared, strict=True):
        fixture = str(video_id)
        records.append(expanded)
        for key in totals:
            if key != "videos":
                totals[key] += int(counts.get(key, 0))
        print(f"Prepared early {fixture}: {counts}", flush=True)

    payload: dict[str, Any] = {
        "schema": "contact-rally-start-early-broader-inputs/1",
        "status": "complete",
        "labels_read": False,
        "source_chooser_inputs": chooser_inputs.name,
        "source_prediction_file": COMBINED_PREDICTIONS,
        "max_earlier_candidates": MAX_EARLY_CANDIDATES,
        "feature_root": "features",
        "video_ids": list(requested),
        "counts": totals,
        "videos": records,
        "prepare_seconds": perf_counter() - started,
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--inpaint-root", type=Path, required=True)
    parser.add_argument("--chooser-inputs", type=Path, default=DEFAULT_CHOOSER_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-ids", type=int, nargs="+", choices=VIDEO_IDS)
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    payload = prepare_early_broader_inputs(
        saved_root=arguments.saved_root,
        prepared_root=arguments.prepared_root,
        inpaint_root=arguments.inpaint_root,
        chooser_inputs=arguments.chooser_inputs,
        output=arguments.output,
        video_ids=arguments.video_ids,
        jobs=arguments.jobs,
    )
    print(f"Wrote {len(payload['videos'])} early broader videos to {arguments.output}")


if __name__ == "__main__":
    main()
