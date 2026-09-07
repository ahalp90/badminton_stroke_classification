"""Expand the saved early-contact shortlist without changing its search windows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from scratch.contact_det_full_ds_fit.scripts.rally_start_model import _normalise_side

MAX_EARLY_CANDIDATES = 4


def _video_fixture(video: Mapping[str, Any]) -> str:
    raw_identity = video.get("video")
    identity = raw_identity if isinstance(raw_identity, Mapping) else video
    fixture = identity.get("fixture")
    if not isinstance(fixture, str) or not fixture:
        raise ValueError("saved video fixture is missing")
    return fixture


def _ordered_score_rows(score_rows: np.ndarray) -> list[np.void]:
    required = {"interval_id", "frame", "contact_score", "kept"}
    if score_rows.ndim != 1 or score_rows.dtype.names is None or not required <= set(score_rows.dtype.names):
        raise ValueError("fixture score rows lack the required shortlist fields")
    if len({int(frame) for frame in score_rows["frame"]}) != len(score_rows):
        raise ValueError("fixture score frames repeat")
    return sorted(
        score_rows,
        key=lambda row: (-float(row["contact_score"]), int(row["frame"])),
    )


def expand_early_shortlist(
    video: Mapping[str, Any],
    score_rows: np.ndarray,
    side_by_frame: Mapping[int, str | None],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Append eligible early candidates to a saved label-free video record.

    ``score_rows`` must already be filtered to the fixture.  ``side_by_frame``
    is the automatic side replay from ``prepare_later_inputs._side_replay``:
    it must contain one normalised ``Top``/``Bot``/``None`` value for every
    newly selected frame.  No labels are consulted here.
    """
    expanded: dict[str, Any] = deepcopy(dict(video))
    fixture = _video_fixture(expanded)
    raw_lists = expanded.get("candidate_lists")
    if not isinstance(raw_lists, list):
        raise TypeError(f"{fixture}: candidate lists must be a list")
    ordered_rows = _ordered_score_rows(score_rows)
    added = 0
    changed_sections = 0

    for list_index, raw_list in enumerate(raw_lists):
        if not isinstance(raw_list, Mapping):
            raise TypeError(f"{fixture}: candidate list must be an object")
        candidate_list = dict(raw_list)
        if candidate_list.get("fixture") != fixture:
            raise ValueError(f"{fixture}: candidate-list video differs")
        raw_candidates = candidate_list.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{fixture}: candidate entries must be a non-empty list")
        interval_id = int(candidate_list["interval_id"])
        prefix_start = int(candidate_list["prefix_start_frame"])
        fixed_frame = int(candidate_list["fixed_contact_frame"])
        duplicate_distance = int(candidate_list["duplicate_distance_frames"])
        retained_frames = [int(candidate["frame"]) for candidate in raw_candidates]
        earlier_count = sum(
            not bool(candidate.get("is_fixed_contact")) for candidate in raw_candidates
        )
        section_added = 0
        for row in ordered_rows:
            if earlier_count >= MAX_EARLY_CANDIDATES:
                break
            frame = int(row["frame"])
            if not (
                int(row["interval_id"]) == interval_id
                and prefix_start <= frame < fixed_frame
            ):
                continue
            if frame in retained_frames or any(
                abs(frame - retained) <= duplicate_distance for retained in retained_frames
            ):
                continue
            if frame not in side_by_frame:
                raise KeyError(f"{fixture}/{frame}: automatic side replay is missing")
            raw_candidates.append({
                "frame": frame,
                "contact_score": float(row["contact_score"]),
                "is_fixed_contact": False,
                "kept": bool(row["kept"]),
                "predicted_side": _normalise_side(
                    side_by_frame[frame], f"{fixture}/{frame}: candidate"
                ),
            })
            retained_frames.append(frame)
            earlier_count += 1
            section_added += 1
        raw_lists[list_index] = candidate_list
        added += section_added
        changed_sections += bool(section_added)

    return expanded, {
        "candidate_lists": len(raw_lists),
        "sections_with_additions": changed_sections,
        "added_earlier_candidates": added,
    }
