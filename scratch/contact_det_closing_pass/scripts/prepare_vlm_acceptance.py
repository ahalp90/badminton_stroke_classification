"""Prepare label-free VLM clips for frozen gap-acceptance sections."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import FixedSpan
from scratch.contact_det_closing_pass.scripts.run_later_broader import restore_stream
from scratch.vlm_pr80_eval.experiments.rally_start_trials import (
    EXPECTED_FRAMES,
    HEIGHT,
    MANIFEST_SCHEMA,
    WIDTH,
    _render_clip,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_FIXTURES = ("sset_01", "sset_15", "sset_21")
DEFAULT_GAP_RESULT = (
    REPO_ROOT / "scratch/contact_det_closing_pass/results/followups/gap_acceptance_result.json.gz"
)
DEFAULT_MARGIN_OUTPUT = (
    REPO_ROOT / "scratch/contact_det_closing_pass/results/later/later_margin_predictions.json.gz"
)
DEFAULT_EARLY_INPUTS = (
    REPO_ROOT / "scratch/contact_det_full_ds_fit/raw/training_rally_start_inputs/rally_start_training_inputs.json.gz"
)
SHIFT_CONTROL_COUNT = 4
NATURAL_PRE_FRAMES = 80
SHIFTED_PRE_FRAMES = 100


@dataclass(frozen=True)
class _VideoSource:
    fixture: str
    path: Path
    fps: float
    frame_count: int


@dataclass(frozen=True)
class _PreparedCase:
    route: dict[str, Any]
    manifest: dict[str, Any]
    source: _VideoSource


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _saved_video_metadata(path: Path) -> dict[str, tuple[float, int]]:
    payload = _read_json_gz(path)
    if (
        payload.get("schema") != "contact-rally-start-training-inputs/1"
        or payload.get("status") != "complete"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("saved early inputs are not the complete label-free record")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise TypeError("saved early inputs videos must be a list")
    metadata: dict[str, tuple[float, int]] = {}
    for raw_video in videos:
        if not isinstance(raw_video, Mapping):
            raise TypeError("saved early input video must be an object")
        identity = raw_video.get("video")
        if not isinstance(identity, Mapping):
            raise TypeError("saved early input video identity must be an object")
        fixture = identity.get("fixture")
        fps = identity.get("fps")
        frame_count = identity.get("frame_count")
        if (
            not isinstance(fixture, str)
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or fps <= 0.0
            or frame_count < EXPECTED_FRAMES
        ):
            raise ValueError("saved early input video metadata is malformed")
        if fixture in metadata:
            raise ValueError(f"{fixture}: saved video metadata repeats")
        metadata[fixture] = (float(fps), frame_count)
    missing = set(TARGET_FIXTURES) - set(metadata)
    if missing:
        raise ValueError(f"saved early inputs omit target fixtures: {sorted(missing)}")
    return metadata


def _source_video(artifacts_root: Path, fixture: str, metadata: tuple[float, int]) -> _VideoSource:
    directory = artifacts_root / "stages" / "tracknet_input" / fixture
    matches = sorted(directory.glob("*.avi"))
    if len(matches) != 1:
        raise ValueError(f"expected one AVI for {fixture} under {directory}, found {len(matches)}")
    path = matches[0]
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    saved_fps, saved_count = metadata
    if abs(source_fps - saved_fps) > 0.01:
        raise ValueError(f"{fixture}: source FPS {source_fps} differs from saved FPS {saved_fps}")
    if frame_count != saved_count:
        raise ValueError(f"{fixture}: source frame count {frame_count} differs from saved count {saved_count}")
    return _VideoSource(fixture, path, source_fps, frame_count)


def _window(anchor_frame: int, pre_frames: int, total_frames: int) -> tuple[int, int]:
    if total_frames < EXPECTED_FRAMES:
        raise ValueError("source video is shorter than the required 120-frame clip")
    start = min(max(anchor_frame - pre_frames, 0), total_frames - EXPECTED_FRAMES)
    return start, start + EXPECTED_FRAMES


def _score_rows(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[tuple[str, int], float]:
    if not np.isfinite(threshold):
        raise ValueError("gap policy threshold is not finite")
    scores: dict[tuple[str, int], float] = {}
    for row in rows:
        fixture = row.get("fixture")
        span_id = row.get("span_id")
        score = row.get("gap_score")
        if not isinstance(fixture, str) or type(span_id) is not int:
            raise ValueError("gap score row identity is malformed")
        if fixture not in TARGET_FIXTURES:
            continue
        value = float(score)
        if not np.isfinite(value):
            raise ValueError(f"{fixture}/{span_id}: gap score is not finite")
        identity = (fixture, span_id)
        if identity in scores:
            raise ValueError(f"{identity}: gap score row repeats")
        scores[identity] = value
    return scores


def route_cases(
    rows: Sequence[Mapping[str, Any]], threshold: float, spans: Sequence[FixedSpan],
) -> tuple[dict[str, Any], ...]:
    """Return accepted natural cases using only saved IDs, scores, and spans."""
    scores = _score_rows(rows, threshold)
    spans_by_identity: dict[tuple[str, int], FixedSpan] = {}
    for span in spans:
        identity = (span.fixture, span.span_id)
        if span.fixture not in TARGET_FIXTURES:
            continue
        if identity in spans_by_identity:
            raise ValueError(f"{identity}: restored span repeats")
        spans_by_identity[identity] = span
    missing_accepted = sorted(
        identity
        for identity, score in scores.items()
        if score >= threshold and identity not in spans_by_identity
    )
    if missing_accepted:
        raise ValueError(f"accepted score rows have no restored span: {missing_accepted}")
    records = []
    for fixture in TARGET_FIXTURES:
        for identity, span in sorted(spans_by_identity.items()):
            if identity[0] != fixture or identity not in scores:
                continue
            score = scores[identity]
            if score < threshold:
                continue
            if not span.events:
                continue
            records.append({
                "case_id": f"gap-acceptance-{fixture}-s{identity[1]:04d}-natural",
                "fixture": fixture,
                "span_id": identity[1],
                "anchor_frame": int(span.events[0].frame),
                "kind": "natural",
                "tree_score": score,
                "threshold": float(threshold),
            })
    return tuple(records)


def _prepared_cases(
    natural_cases: Sequence[Mapping[str, Any]], sources: Mapping[str, _VideoSource],
) -> tuple[_PreparedCase, ...]:
    by_fixture_index: dict[str, int] = {}
    prepared: list[_PreparedCase] = []
    for natural in natural_cases:
        fixture = str(natural["fixture"])
        source = sources[fixture]
        anchor = int(natural["anchor_frame"])
        natural_start, natural_end = _window(anchor, NATURAL_PRE_FRAMES, source.frame_count)
        natural_id = str(natural["case_id"])
        prepared.append(_case_pair(
            natural, source, natural_start, natural_end, natural_start,
        ))
        index = by_fixture_index.get(fixture, 0)
        if index < SHIFT_CONTROL_COUNT:
            shifted_start, shifted_end = _window(anchor, SHIFTED_PRE_FRAMES, source.frame_count)
            shifted = {
                **natural,
                "case_id": f"{natural_id[:-8]}shifted",
                "kind": "shifted",
                "paired_natural_id": natural_id,
            }
            prepared.append(_case_pair(
                shifted, source, shifted_start, shifted_end, natural_start,
            ))
        by_fixture_index[fixture] = index + 1
    return tuple(prepared)


def _case_pair(
    record: Mapping[str, Any], source: _VideoSource, start: int, end: int,
    natural_start: int,
) -> _PreparedCase:
    case_id = str(record["case_id"])
    route = {
        **record,
        "source_start_frame": start,
        "source_end_frame": end,
        "fps": source.fps,
        "source_frame_indices": list(range(start, end)),
        "source_start_delta_frames": start - natural_start,
    }
    if record["kind"] == "natural":
        route.pop("paired_natural_id", None)
    manifest = {
        "case_id": case_id,
        "video_id": source.fixture,
        "clip_path": str(Path("clips") / f"{case_id}.mp4"),
        "source_start_frame": start,
        "source_end_frame": end,
        "sample_fps": source.fps,
        "expected_frames": EXPECTED_FRAMES,
        "width": WIDTH,
        "height": HEIGHT,
    }
    return _PreparedCase(route, manifest, source)


def _load_gap_inputs(path: Path) -> tuple[list[dict[str, Any]], float]:
    payload = _read_json_gz(path)
    if payload.get("schema") != "contact-gap-acceptance/1" or payload.get("status") != "complete":
        raise ValueError("gap acceptance result is not complete")
    policies = payload.get("policies")
    rows = payload.get("rows")
    if not isinstance(policies, Mapping) or not isinstance(rows, list):
        raise TypeError("gap acceptance result lacks policies or rows")
    gap_policy = policies.get("gap")
    if not isinstance(gap_policy, Mapping):
        raise TypeError("gap acceptance policy is missing")
    threshold = float(gap_policy["threshold"])
    return rows, threshold


def prepare(
    artifacts_root: Path, output_dir: Path, gap_result: Path = DEFAULT_GAP_RESULT,
    margin_output: Path = DEFAULT_MARGIN_OUTPUT, early_inputs: Path = DEFAULT_EARLY_INPUTS,
) -> dict[str, Any]:
    """Render accepted natural clips and deterministic shifted controls."""
    started = perf_counter()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, threshold = _load_gap_inputs(gap_result)
    spans = restore_stream(_read_json_gz(margin_output)["outputs"]).spans
    natural_cases = route_cases(rows, threshold, spans)
    metadata = _saved_video_metadata(early_inputs)
    sources = {
        fixture: _source_video(artifacts_root, fixture, metadata[fixture])
        for fixture in TARGET_FIXTURES
    }
    cases = _prepared_cases(natural_cases, sources)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        _render_clip(
            case.source.path,
            output_dir / case.manifest["clip_path"],
            fps=case.source.fps,
            source_start=case.manifest["source_start_frame"],
            source_end=case.manifest["source_end_frame"],
            selected_cut_frame=None,
        )
    manifest = {"schema": MANIFEST_SCHEMA, "cases": [case.manifest for case in cases]}
    coverage = {}
    target_spans = [span for span in spans if span.fixture in TARGET_FIXTURES]
    natural_by_fixture = {fixture: 0 for fixture in TARGET_FIXTURES}
    for case in natural_cases:
        natural_by_fixture[str(case["fixture"])] += 1
    for fixture in TARGET_FIXTURES:
        section_count = sum(span.fixture == fixture for span in target_spans)
        routed_count = natural_by_fixture[fixture]
        accepted_count = sum(row["fixture"] == fixture and row["gap_score"] >= threshold for row in rows)
        coverage[fixture] = {
            "sections": section_count,
            "accepted": accepted_count,
            "routed": routed_count,
            "unrouted_empty_spans": accepted_count - routed_count,
            "shifted_controls": min(routed_count, SHIFT_CONTROL_COUNT),
        }
    routing = {
        "schema": "contact-gap-vlm-routing/1",
        "status": "complete",
        "reason": "gap_score >= frozen gap policy threshold",
        "threshold": threshold,
        "coverage": coverage,
        "cases": [case.route for case in cases],
        "prepare_seconds": perf_counter() - started,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "routing.json", routing)
    print("Prepared", len(natural_cases), "natural cases and", len(cases) - len(natural_cases), "shifted controls", flush=True)
    return routing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-result", type=Path, default=DEFAULT_GAP_RESULT)
    parser.add_argument("--margin-output", type=Path, default=DEFAULT_MARGIN_OUTPUT)
    parser.add_argument("--early-inputs", type=Path, default=DEFAULT_EARLY_INPUTS)
    args = parser.parse_args()
    prepare(args.artifacts_root, args.output, args.gap_result, args.margin_output, args.early_inputs)


if __name__ == "__main__":
    main()
