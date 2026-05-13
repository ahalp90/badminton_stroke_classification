"""Video I/O utilities — cv2-based frame iteration + metadata.

Other pipeline modules and bric.* code use this
instead of reaching for cv2 directly, so video reading conventions stay
consistent (RGB ordering, FPS lookup, frame indexing).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for a video file."""
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def get_video_info(path: str | Path) -> VideoInfo:
    """Read video metadata without decoding any frames."""
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {p}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoInfo(path=p, fps=fps, n_frames=n_frames, width=width, height=height)


def iter_frames(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    rgb: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame) for frames in [start_frame, end_frame).

    :param path: Video file path.
    :param start_frame: Inclusive start frame (0-indexed).
    :param end_frame: Exclusive end frame; None means until video ends.
    :param rgb: If True (default), yields RGB; if False, yields cv2's native BGR.
    """
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {p}')
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, frame_idx: int, rgb: bool = True) -> np.ndarray:
    """Read a single frame at the given index. Raises if frame is unreadable."""
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {p}')
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Could not read frame {frame_idx} from {p}')
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    finally:
        cap.release()


def read_frames(
    path: str | Path,
    start_frame: int,
    end_frame: int,
    rgb: bool = True,
) -> np.ndarray:
    """Read [start_frame, end_frame) into a single (T, H, W, 3) uint8 array."""
    frames = [f for _, f in iter_frames(path, start_frame, end_frame, rgb=rgb)]
    if not frames:
        raise ValueError(f'No frames read from {path} for [{start_frame}, {end_frame})')
    return np.stack(frames, axis=0)
