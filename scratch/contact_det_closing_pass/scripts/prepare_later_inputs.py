"""Prepare label-free later-contact candidates from the saved D32 scores."""

from __future__ import annotations

import argparse
import lzma
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from annotator.point_winner import attribute_half
from scratch.contact_det.scripts.freeze_contact_evidence import (
    FixtureSpec,
    _load_inputs,
)
from scratch.contact_det_closing_pass.scripts.evaluation import write_json
from scratch.contact_det_closing_pass.scripts.features import (
    _frozen_feature_names,
    _load_fixture,
)
from scratch.contact_det_closing_pass.scripts.later_options import (
    MAX_LATER_CANDIDATES,
    shortlist_frames,
)
from scratch.contact_det_followup.scripts import prediction_io
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_start_inputs import (
    _checked_replay_sides,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE

RESULT_SCHEMA = "contact-rally-start-later-inputs/1"
SCORE_PATH = (
    prediction_io.REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/training_video_scores/combined/training_video_scores.npy.xz"
)
OUTPUT_PATH = (
    prediction_io.REPO_ROOT
    / "scratch/contact_det_closing_pass/raw/later_inputs/development.json.gz"
)
DEFAULT_FEATURE_ROOT = (
    prediction_io.REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/full_raw"
)
DEVELOPMENT_GROUPS = frozenset(("A", "B", "C", "D"))
SCORE_FIELDS = tuple(SCORE_DTYPE.names or ())


def _load_score_rows(path: Path) -> np.ndarray:
    """Load canonical score fields while allowing the combined ``group`` field."""
    with lzma.open(path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if not isinstance(rows, np.ndarray) or rows.ndim != 1 or rows.dtype.names is None:
        raise ValueError("combined D32 scores must be a one-dimensional structured array")
    if not set(SCORE_FIELDS).issubset(rows.dtype.names):
        raise ValueError("combined D32 scores are missing canonical score fields")
    for field in SCORE_FIELDS:
        if rows.dtype.fields[field][0] != SCORE_DTYPE.fields[field][0]:
            raise ValueError(f"combined D32 score field {field!r} has the wrong dtype")
    if "group" in rows.dtype.names and rows.dtype.fields["group"][0] != np.dtype("S1"):
        raise ValueError("combined D32 score group field has the wrong dtype")
    if len(rows) == 0:
        raise ValueError("combined D32 scores are empty")
    if not np.isfinite(rows["contact_score"]).all():
        raise ValueError("combined D32 scores contain non-finite contact scores")
    if np.any((rows["contact_score"] < 0.0) | (rows["contact_score"] > 1.0)):
        raise ValueError("combined D32 scores are outside zero to one")
    return rows


def _fixture_score_rows(rows: np.ndarray, fixture: str, group: str, fps: float) -> np.ndarray:
    fixture_rows = rows[rows["fixture"] == fixture.encode("ascii")]
    if len(fixture_rows) == 0:
        raise ValueError(f"{fixture}: combined score rows are missing")
    if not np.all(np.isclose(fixture_rows["fps"], fps, rtol=0.0, atol=1e-6)):
        raise ValueError(f"{fixture}: score fps differs from the development split")
    if "group" in fixture_rows.dtype.names and not np.all(
        fixture_rows["group"] == group.encode("ascii")
    ):
        raise ValueError(f"{fixture}: combined score group differs")
    identities = {
        (int(row["interval_id"]), int(row["frame"])) for row in fixture_rows
    }
    if len(identities) != len(fixture_rows):
        raise ValueError(f"{fixture}: score identities repeat")
    return fixture_rows


def _score_by_frame(rows: np.ndarray, fixture: str) -> dict[int, np.void]:
    by_frame: dict[int, np.void] = {}
    for row in rows:
        frame = int(row["frame"])
        if frame in by_frame:
            raise ValueError(f"{fixture}/{frame}: score frame is ambiguous")
        by_frame[frame] = row
    return by_frame


def _side_replay(
    fixture: str,
    video: Any,
    data_root: Path,
    frames: Sequence[int],
) -> tuple[dict[int, str | None], float]:
    started = time.perf_counter()
    spec = FixtureSpec(fixture, int(video.video_id), float(video.fps), video.width, video.height)
    track, pose, court, _segments, sticky, _annotation = _load_inputs(data_root, spec)
    net_band = court.evidence.inputs.net_band
    replay_frames = np.asarray(sorted({int(frame) for frame in frames}), dtype=np.int32)
    if np.any(replay_frames < 0) or np.any(replay_frames >= len(track)):
        raise ValueError(f"{fixture}: replay frame lies outside the automatic track")
    sides = _checked_replay_sides(
        fixture,
        replay_frames,
        track,
        sticky,
        pose.bboxes,
        net_band,
        attribute_half,
    )
    return sides, time.perf_counter() - started


def _physical_blocks(
    fixture: str,
    feature_root: Path,
    requested_frames: set[int],
    frozen_names: tuple[str, ...],
) -> tuple[dict[int, list[float | None]], float]:
    started = time.perf_counter()
    index = _load_fixture(feature_root, fixture, frozen_names, requested_frames)
    if index is None:
        raise FileNotFoundError(f"{fixture}: frozen physical feature file is missing")
    physical: dict[int, list[float | None]] = {}
    for frame in requested_frames:
        block = index.by_frame.get(frame)
        if block is None:
            raise ValueError(f"{fixture}/{frame}: frozen physical feature row is missing")
        physical[frame] = [
            None if not np.isfinite(value) else float(value) for value in block
        ]
    return physical, time.perf_counter() - started


def _sections(
    fixture: str,
    spans: Sequence[Any],
    score_rows: np.ndarray,
    fps: float,
) -> tuple[list[dict[str, Any]], set[int], float]:
    started = time.perf_counter()
    score_by_frame = _score_by_frame(score_rows, fixture)
    sections: list[dict[str, Any]] = []
    candidate_frames: set[int] = set()
    for span in spans:
        selected = shortlist_frames(span, score_rows, fps, limit=MAX_LATER_CANDIDATES)
        candidates: list[dict[str, Any]] = []
        for frame in selected:
            row = score_by_frame.get(frame)
            if row is None:
                raise ValueError(f"{fixture}/{frame}: shortlisted frame is not scored")
            candidate_frames.add(frame)
            candidates.append(
                {
                    "frame": frame,
                    "contact_score": float(row["contact_score"]),
                    "predicted_side": None,
                    "physical": [],
                }
            )
        sections.append({"span_id": int(span.span_id), "candidates": candidates})
    return sections, candidate_frames, time.perf_counter() - started


def prepare_later_inputs(
    *,
    data_root: Path,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    score_path: Path = SCORE_PATH,
    output_path: Path = OUTPUT_PATH,
    fixtures: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the D32 later-candidate records without opening labels."""
    started = time.perf_counter()
    pack = prediction_io.load_development_predictions()
    all_scores = _load_score_rows(score_path)
    requested = None if fixtures is None else tuple(fixtures)
    if requested is not None and len(set(requested)) != len(requested):
        raise ValueError("--fixtures contains a duplicate fixture")
    development_videos = tuple(
        video
        for video in pack.videos
        if pack.group_by_fixture[video.fixture] in DEVELOPMENT_GROUPS
        and (requested is None or video.fixture in requested)
    )
    if requested is not None and {video.fixture for video in development_videos} != set(requested):
        raise ValueError("--fixtures contains an unknown or non-development fixture")
    spans_by_fixture: dict[str, list[Any]] = {video.fixture: [] for video in development_videos}
    for span in pack.spans:
        if span.fixture in spans_by_fixture:
            spans_by_fixture[span.fixture].append(span)
    frozen_names = _frozen_feature_names()
    records: list[dict[str, Any]] = []
    for video in development_videos:
        fixture = video.fixture
        group = pack.group_by_fixture[fixture]
        video_started = time.perf_counter()
        score_started = time.perf_counter()
        score_rows = _fixture_score_rows(all_scores, fixture, group, video.fps)
        score_load_seconds = time.perf_counter() - score_started
        sections, candidate_frames, shortlist_seconds = _sections(
            fixture, spans_by_fixture[fixture], score_rows, video.fps
        )
        event_frames = {
            int(event.frame)
            for span in spans_by_fixture[fixture]
            for event in span.events
        }
        replay_frames = candidate_frames | event_frames
        if replay_frames:
            sides, side_seconds = _side_replay(fixture, video, data_root, sorted(replay_frames))
            for span in spans_by_fixture[fixture]:
                for event in span.events:
                    if sides[event.frame] != event.predicted_side:
                        raise ValueError(f"{fixture}/{event.frame}: saved side replay differs")
            physical, physical_seconds = _physical_blocks(
                fixture, feature_root, candidate_frames, frozen_names
            )
        else:
            sides, side_seconds = {}, 0.0
            physical, physical_seconds = {}, 0.0
        for section in sections:
            for candidate in section["candidates"]:
                frame = int(candidate["frame"])
                candidate["predicted_side"] = sides[frame]
                candidate["physical"] = physical[frame]
        total_seconds = time.perf_counter() - video_started
        records.append(
            {
                "fixture": fixture,
                "group": group,
                "fps": float(video.fps),
                "sections": sections,
                "timings": {
                    "score_load_seconds": score_load_seconds,
                    "shortlist_seconds": shortlist_seconds,
                    "side_seconds": side_seconds,
                    "physical_seconds": physical_seconds,
                    "total_seconds": total_seconds,
                },
            }
        )
        print(f"Prepared {fixture}: {sum(len(section['candidates']) for section in sections)} candidates", flush=True)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "labels_read": False,
        "physical_feature_names": list(frozen_names),
        "videos": records,
        "timings": {"total_seconds": time.perf_counter() - started},
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--scores", type=Path, default=SCORE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--fixtures", nargs="+", default=None)
    arguments = parser.parse_args()
    payload = prepare_later_inputs(
        data_root=arguments.data_root,
        feature_root=arguments.feature_root,
        score_path=arguments.scores,
        output_path=arguments.output,
        fixtures=arguments.fixtures,
    )
    print(f"Wrote {len(payload['videos'])} later-input videos to {arguments.output}")


if __name__ == "__main__":
    main()
