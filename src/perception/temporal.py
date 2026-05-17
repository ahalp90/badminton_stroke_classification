"""Temporal windowing utilities for stroke-centred clip extraction.

Produces frame indices around stroke events for the BRIC pipeline.

Three functions cover the use cases:
  - clip_window_seconds: contiguous (start, end) window defined by ±seconds
    around the user-supplied target_frame (the stroke event). Used for
    shuttle / court feature windows where the encoder handles variable T.
  - clip_window_frames: contiguous (start, end) window of exactly N frames
    around the target_frame. Building block; rarely called directly in v1.
  - subsample_indices: list of N frame indices uniformly subsampled from a
    coverage_sec window centred on target_frame. Stride adapts to source
    fps so real-world coverage stays constant. Used to build the input
    tensor for R(2+1)D-18 (or any single-rate 3D backbone) regardless of
    source frame rate.

The contiguous-window functions return (start, end) where end is EXCLUSIVE —
`range(start, end)` gives the frame indices.
"""

from __future__ import annotations

import numpy as np


def clip_window_seconds(
    target_frame: int,
    fps: float,
    before_sec: float = 0.5,
    after_sec: float = 0.5,
) -> tuple[int, int]:
    """Return (start, end) frame indices for a time-defined window.

    :param target_frame: Frame index of the stroke event (0-indexed).
    :param fps: Source video frame rate.
    :param before_sec: Seconds of context before the stroke (>= 0).
    :param after_sec: Seconds of context after the stroke (>= 0).
    :return: (start_frame, end_frame). end is exclusive; start is clamped
        to >= 0 so windows near the start of a video are short rather
        than negative.
    :raises ValueError: if fps <= 0 or any duration is negative.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if before_sec < 0 or after_sec < 0:
        raise ValueError(
            f"before/after seconds must be >= 0, got before={before_sec}, after={after_sec}"
        )
    n_before = round(before_sec * fps)
    n_after = round(after_sec * fps)
    start = max(0, target_frame - n_before)
    end = target_frame + n_after + 1
    return start, end


def clip_window_frames(
    target_frame: int,
    n_frames: int = 16,
    n_before: int | None = None,
) -> tuple[int, int]:
    """Return (start, end) frame indices spanning exactly ``n_frames`` frames.

    Building block for fixed-length windowing. The R(2+1)D pathway uses
    ``subsample_indices`` instead — this function is kept for callers
    that want a contiguous fixed-length range without subsampling.

    Note on edge cases: if ``target_frame - n_before`` would be negative,
    ``start`` is clamped to 0 and ``end`` shifts to ``n_frames`` so the
    window length stays exact. This means the centre is no longer at
    ``n_before`` from the start — callers needing strict centring near
    a video boundary should pad the source video upstream.

    :param target_frame: Frame index of the stroke event (0-indexed).
    :param n_frames: Total number of frames to include (>= 1).
    :param n_before: Number of frames before the centre. Default
        ``(n_frames - 1) // 2`` (slightly more after if n_frames is even).
    :return: (start_frame, end_frame). end is exclusive.
    :raises ValueError: if n_frames <= 0 or n_before is out of [0, n_frames).
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if n_before is None:
        n_before = (n_frames - 1) // 2
    if n_before < 0 or n_before >= n_frames:
        raise ValueError(f"n_before must be in [0, {n_frames}), got {n_before}")
    start = max(0, target_frame - n_before)
    end = start + n_frames
    return start, end


def subsample_indices(
    target_frame: int,
    fps: float,
    coverage_sec: float = 2.0,
    n: int = 32,
    total_frames: int | None = None,
) -> list[int]:
    """Return ``n`` frame indices uniformly subsampled from a window
    of ``coverage_sec`` real-world seconds centred on ``target_frame``.

    Stride adapts to source fps so the model always sees the same
    real-world coverage regardless of the recording frame rate.
    Indices are uniformly spaced (``np.linspace``-style, integer
    rounded) over the window.

    Defaults (``coverage_sec=2.0``, ``n=32``) give ~16 Hz effective
    sampling over a 2-second window — a sensible badminton starting
    point for R(2+1)D-18. ``coverage_sec`` is a hyperparameter to
    ablate.

    :param target_frame: Frame index of the stroke centre (0-indexed).
    :param fps: Source video frame rate.
    :param coverage_sec: Real-world duration of the sampling window.
    :param n: Number of frames to return.
    :param total_frames: Optional total frame count for clamping. If
        provided, indices are clamped to ``[0, total_frames - 1]``;
        windows near a clip boundary will repeat the boundary frame
        rather than going out of range. If None, callers are
        responsible for handling out-of-range indices.
    :return: List of ``n`` int frame indices.
    :raises ValueError: if fps <= 0, coverage_sec <= 0, or n < 1.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if coverage_sec <= 0:
        raise ValueError(f"coverage_sec must be positive, got {coverage_sec}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got n={n}")

    if n == 1:
        idxs = np.array([target_frame])
    else:
        half_span_frames = (coverage_sec * fps) / 2.0
        idxs = np.linspace(
            target_frame - half_span_frames,
            target_frame + half_span_frames,
            n,
        )

    if total_frames is not None:
        idxs = np.clip(idxs, 0, total_frames - 1)
    return idxs.round().astype(int).tolist()
