"""Player detection + tracking for badminton videos.

Wraps Ultralytics YOLO11 with its built-in tracker (ByteTrack by default).
Returns per-track records with stable bboxes per frame and top/bottom
court-side assignment.

Public API:
  - `detect_and_track(video_path, court_info=...)` — the only function
    other modules should call.
  - `PlayerTrack` — dataclass returned by `detect_and_track`.

Court-side assignment:
  - If ``court_info`` is provided (dict with ``H`` and court borders),
    project bbox foot-centres to normalised court coordinates and assign
    top/bottom by the track's averaged court-y over its on-court frames.
  - If not provided, fall back to a vertical-pixel heuristic
    (top half of frame = top player).

Filtering for >2 detected persons (coaches, ball persons, audience):
  Keep the 2 tracks with the most on-court frames (with court_info) or
  the most detection frames (without). Matches the BST pipeline's
  player-identification approach so both pipelines pick consistent
  player identities.

Known limitation:
  Track IDs are stable within one video but can swap when players cross
  paths (e.g. at the net). Court-side is re-derived from court position
  each frame inside this module, so the *side* assignment stays correct
  even if the underlying tracker swaps IDs. Downstream code should rely
  on court_side, not track_id, for "which player did this".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from shared.video_io import get_video_info
from shared.court import convert_homogeneous, project

# Project-relative default location for the YOLO11n weights.
# runtime/checkpoints/ is gitignored; ultralytics auto-downloads to this path
# on first use. This file lives at <project>/src/bric/perception/players.py, so
# parents[3] = <project>.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YOLO_WEIGHTS = _PROJECT_ROOT / "runtime" / "checkpoints" / "yolo11" / "yolo11n.pt"

CourtSide = Literal["top", "bottom", "unknown"]
Bbox = tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class PlayerTrack:
    """A player's per-frame bboxes + assigned court side, stable within video.

    :param track_id: Per-track integer assigned by the YOLO tracker. Stable
        within a single video; not portable across videos. Can swap when
        two players cross paths — downstream code should rely on
        ``court_side`` for player identity, not ``track_id``.
    :param court_side: 'top' / 'bottom' / 'unknown'. Derived from the
        track's averaged normalised court y-coordinate over on-court
        frames (when court_info is provided), or from the average frame
        y-position when not.
    :param bboxes: Mapping of ``frame_idx → (x1, y1, x2, y2)`` in pixel
        coordinates. Frames where the track was not detected are absent.
    :param confidences: Mapping of ``frame_idx → YOLO detection confidence``,
        same keys as ``bboxes``.
    """

    track_id: int
    court_side: CourtSide
    bboxes: dict[int, Bbox]
    confidences: dict[int, float]

    @property
    def n_frames(self) -> int:
        """Number of frames in which this track was detected."""
        return len(self.bboxes)


def detect_and_track(
    video_path: str | Path,
    court_info: dict | None = None,
    yolo_weights: str | Path = DEFAULT_YOLO_WEIGHTS,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.25,
    max_players: int = 2,
) -> list[PlayerTrack]:
    """Detect + track persons in a video; assign top/bottom court-side.

    :param video_path: Path to video file.
    :param court_info: Optional dict with keys
        ``{H, border_L, border_R, border_U, border_D}`` as produced by
        ``shared.court.get_court_info``. If provided, court-side is
        assigned via homography projection of bbox foot-centres. If
        None, falls back to a vertical-pixel heuristic.
    :param yolo_weights: Path to YOLO11 weights. Ultralytics will
        auto-download on first use if the file is missing.
    :param yolo_imgsz: YOLO inference image size (square).
    :param yolo_conf: Detection confidence threshold in [0, 1].
    :param max_players: Maximum number of tracks to return after
        filtering. Default 2 (singles match). Excess tracks are
        dropped — kept tracks are ranked by on-court frame count
        (or by total detection count when no court_info is provided).
    :return: List of ``PlayerTrack`` records, length <= ``max_players``.
        Empty list if no persons were detected with track IDs.
    :raises FileNotFoundError: if ``video_path`` does not exist.
    """
    # Lazy import: lets this module be importable in environments where
    # ultralytics isn't installed (e.g. a backend-only venv).
    from ultralytics import YOLO

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    yolo_weights = Path(yolo_weights)
    yolo_weights.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(yolo_weights))

    # classes=[0] = COCO 'person'. persist=True keeps tracker state across
    # frames within this stream. stream=True yields frame-by-frame results
    # rather than buffering the whole video into memory.
    results = model.track(
        source=str(video_path),
        persist=True,
        classes=[0],
        imgsz=yolo_imgsz,
        conf=yolo_conf,
        verbose=False,
        stream=True,
    )

    raw_tracks = _collect_raw_tracks(results)
    if not raw_tracks:
        return []

    if court_info is not None:
        scored = _score_by_court(raw_tracks, court_info)
    else:
        scored = _score_by_pixel_height(raw_tracks, video_path)

    # Highest-scoring tracks first; truncate to max_players.
    scored.sort(key=lambda t: t["score"], reverse=True)
    return [_build_player_track(t) for t in scored[:max_players]]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_raw_tracks(yolo_results) -> dict[int, dict[int, tuple[Bbox, float]]]:
    """Gather YOLO tracker output into ``{track_id: {frame_idx: (bbox, conf)}}``.

    Skips frames with no detections or where the tracker didn't assign
    track IDs (which happens briefly on first detection in some
    ultralytics versions).
    """
    raw: dict[int, dict[int, tuple[Bbox, float]]] = {}
    for frame_idx, frame_result in enumerate(yolo_results):
        boxes = frame_result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for tid, xyxy, conf in zip(ids, xyxys, confs):
            bbox: Bbox = (
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
            )
            raw.setdefault(int(tid), {})[frame_idx] = (bbox, float(conf))
    return raw


def _score_by_court(
    raw_tracks: dict[int, dict[int, tuple[Bbox, float]]],
    court_info: dict,
) -> list[dict]:
    """Score each track by on-court frame count; assign side from avg court_y.

    The score is the number of frames in which the track's foot-centre
    projects to inside the court polygon (with a small ``eps`` slack
    for soft borders). Side is averaged from the projected normalised
    court y-coordinate over on-court frames.
    """
    H = court_info["H"]
    border_L = court_info["border_L"]
    border_R = court_info["border_R"]
    border_U = court_info["border_U"]
    border_D = court_info["border_D"]
    eps = 0.01

    out = []
    for tid, frames in raw_tracks.items():
        on_court = 0
        court_y_sum = 0.0
        for bbox, _conf in frames.values():
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]   # bottom edge of bbox
            cam_pt = np.array([[foot_x], [foot_y]])
            court_pt = project(H, convert_homogeneous(cam_pt))
            court_x_n = (court_pt[0, 0] - border_L) / (border_R - border_L)
            court_y_n = (court_pt[1, 0] - border_U) / (border_D - border_U)
            if (-eps <= court_x_n <= 1 + eps) and (-eps <= court_y_n <= 1 + eps):
                on_court += 1
                court_y_sum += court_y_n
        side: CourtSide
        if on_court > 0:
            side = "top" if (court_y_sum / on_court) < 0.5 else "bottom"
        else:
            side = "unknown"
        out.append(
            {
                "track_id": tid,
                "score": on_court,
                "side": side,
                "frames": frames,
            }
        )
    return out


def _score_by_pixel_height(
    raw_tracks: dict[int, dict[int, tuple[Bbox, float]]],
    video_path: Path,
) -> list[dict]:
    """Fallback when no court info: score by detection count, side by avg pixel y.

    Top half of frame (avg bbox-centre-y < height/2) = 'top' player.
    """
    info = get_video_info(video_path)
    midline = info.height / 2

    out = []
    for tid, frames in raw_tracks.items():
        avg_y = float(np.mean([(bb[1] + bb[3]) / 2 for bb, _ in frames.values()]))
        side: CourtSide = "top" if avg_y < midline else "bottom"
        out.append(
            {
                "track_id": tid,
                "score": len(frames),
                "side": side,
                "frames": frames,
            }
        )
    return out


def _build_player_track(scored: dict) -> PlayerTrack:
    """Assemble a PlayerTrack from a scored intermediate dict."""
    bboxes = {f: bb for f, (bb, _c) in scored["frames"].items()}
    confs = {f: c for f, (_bb, c) in scored["frames"].items()}
    return PlayerTrack(
        track_id=scored["track_id"],
        court_side=scored["side"],
        bboxes=bboxes,
        confidences=confs,
    )
