"""Shared types for the heuristics package.

Each heuristic variant (``current``, ``sticky_anchor``, etc.) lives in its
own module under this package and exposes an ``apply`` function with the
signature:

    apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput

Kept separate from ``__init__.py`` so variant modules can import the shared
types without triggering the package-level registry build.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

J = 17  # COCO keypoint count, shared across heuristic variants.


class RawClip(NamedTuple):
    """Per-clip raw MMPose output, as written by ``preparing_data.raw_extract``.

    Real detections in frame ``f`` occupy indices ``0 .. ndet[f] - 1`` along
    the detect axis; entries at and beyond ``ndet[f]`` are NaN-padded. Shapes:

    - ``kps``       ``(F, N_max, J, 2)``  float32
    - ``bboxes``    ``(F, N_max, 4)``     float32
    - ``scores``    ``(F, N_max)``        float32
    - ``kp_scores`` ``(F, N_max, J)``     float32
    - ``ndet``      ``(F,)``              int
    """
    kps: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray
    kp_scores: np.ndarray
    ndet: np.ndarray


@dataclass
class ClipContext:
    """Per-clip context needed to project pixel coords into court coords.

    ``all_court_info`` is a ``{vid: court_info}`` map as returned by
    ``pipeline.court_utils.get_court_info``; ``res_df`` is a DataFrame
    indexed by video id with at least ``width`` and ``height`` columns.
    """
    vid: int
    all_court_info: dict
    res_df: pd.DataFrame


class HeuristicOutput(NamedTuple):
    """Per-clip output matching the existing pipeline's filtered schema.

    - ``pos``    ``(F, 2, 2)``   normalised court positions, slot order (Top, Bottom).
    - ``joints`` ``(F, 2, J, 2)`` bbox-diagonal-normalised keypoints.
    - ``failed`` ``(F,)`` bool   True where the frame was zeroed.
    """
    pos: np.ndarray
    joints: np.ndarray
    failed: np.ndarray
