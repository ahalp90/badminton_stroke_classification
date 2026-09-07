"""Join frozen per-frame contact features to first-contact action rows."""

from __future__ import annotations

import lzma
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_tree_contact_features import (
    _feature_family_names,
)
from scratch.contact_det_followup.scripts import score_start_model

_ACTION_FEATURE_COUNT = 10
_FEATURE_FILENAME = "contact_features.npy.xz"
_IDENTITY_FIELDS = ("fixture", "frame", "fps")


@dataclass(frozen=True)
class _FeatureIndex:
    """Feature rows for one fixture, indexed by source frame."""

    fps: float | None
    by_frame: Mapping[int, np.ndarray]


def _frozen_feature_names() -> tuple[str, ...]:
    families = _feature_family_names()
    return tuple(families["physics"] + families["missingness"])


def _feature_names() -> tuple[str, ...]:
    frozen = _frozen_feature_names()
    return (
        *score_start_model.ACTION_FEATURE_NAMES,
        *(f"candidate__{name}" for name in frozen),
        *(f"fixed__{name}" for name in frozen),
    )


def _fixture_value(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii")
    if isinstance(value, str):
        return value
    raise TypeError("frozen feature fixture must be text")


def _load_fixture(
    feature_root: Path,
    fixture: str,
    frozen_names: tuple[str, ...],
    requested_frames: set[int],
) -> _FeatureIndex | None:
    path = feature_root / "videos" / fixture / _FEATURE_FILENAME
    if not path.exists():
        return None

    with lzma.open(path, "rb") as handle:
        values = np.load(handle, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype.names is None or values.ndim != 1:
        raise ValueError(f"{fixture}: frozen feature array must be one-dimensional and structured")

    required_fields = set(_IDENTITY_FIELDS) | set(frozen_names)
    missing_fields = required_fields - set(values.dtype.names)
    if missing_fields:
        raise ValueError(f"{fixture}: frozen feature fields are missing {sorted(missing_fields)}")

    frames = values["frame"]
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{fixture}: duplicate frozen feature frame is ambiguous")
    # Only the two competing contacts need feature blocks; avoid constructing
    # Python arrays for every saved frame in the video.
    requested = np.isin(frames, tuple(requested_frames))
    selected_values = values[requested]
    source_fps: float | None = None
    by_frame: dict[int, np.ndarray] = {}
    for source_row in selected_values:
        source_fixture = _fixture_value(source_row["fixture"])
        if source_fixture != fixture:
            raise ValueError(f"{fixture}: frozen feature fixture differs")

        raw_fps = float(source_row["fps"])
        if not np.isfinite(raw_fps):
            raise ValueError(f"{fixture}: frozen feature fps is not finite")
        if source_fps is None:
            source_fps = raw_fps
        elif not np.isclose(raw_fps, source_fps, rtol=0.0, atol=1e-6):
            raise ValueError(f"{fixture}: frozen feature fps differs between rows")

        raw_frame = source_row["frame"]
        if not np.isfinite(raw_frame):
            raise ValueError(f"{fixture}: frozen feature frame is not finite")
        frame = int(raw_frame)
        if raw_frame != frame:
            raise ValueError(f"{fixture}: frozen feature frame is not integral")
        if frame in by_frame:
            raise ValueError(
                f"{fixture}/{frame}: duplicate frozen feature frame is ambiguous"
            )

        block = np.asarray([source_row[name] for name in frozen_names], dtype=np.float64)
        if np.isinf(block).any():
            raise ValueError(f"{fixture}/{frame}: frozen feature contains infinity")
        by_frame[frame] = block

    return _FeatureIndex(source_fps, by_frame)


def _missing_audit() -> dict[str, int]:
    return {
        "candidate_missing_joins": 0,
        "fixed_missing_joins": 0,
        "candidate_missing_cells": 0,
        "fixed_missing_cells": 0,
    }


def join_physical_features(
    action_rows: Sequence[score_start_model.ActionRow],
    feature_root: Path,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Join action rows with frozen physics and missingness features.

    Missing files and frames produce an all-NaN 85-cell block. Existing NaNs are
    retained, while infinite values and duplicate source frames fail loudly.
    """
    frozen_names = _frozen_feature_names()
    names = _feature_names()
    rows = tuple(action_rows)
    existing = np.asarray([row.features for row in rows], dtype=np.float64)
    if not rows:
        existing = np.empty((0, _ACTION_FEATURE_COUNT), dtype=np.float64)
    if existing.ndim != 2 or existing.shape[1] != _ACTION_FEATURE_COUNT:
        raise ValueError("action rows must contain ten features")
    if np.isinf(existing).any():
        raise ValueError("action row features must not contain infinity")

    requested_by_fixture: dict[str, set[int]] = {}
    for row in rows:
        candidate = row.candidate
        requested_by_fixture.setdefault(candidate.fixture, set()).update(
            (candidate.frame, candidate.fixed_contact_frame)
        )
    feature_indices: dict[str, _FeatureIndex | None] = {}
    audit_by_fixture: dict[str, dict[str, int]] = {}
    missing_identities: list[dict[str, Any]] = []
    physical_blocks: list[np.ndarray] = []
    frozen_width = len(frozen_names)
    root = Path(feature_root)

    for row in rows:
        candidate = row.candidate
        fixture = candidate.fixture
        if fixture not in feature_indices:
            feature_indices[fixture] = _load_fixture(root, fixture, frozen_names, requested_by_fixture[fixture])
            audit_by_fixture[fixture] = _missing_audit()
        index = feature_indices[fixture]
        fixture_audit = audit_by_fixture[fixture]
        if index is not None and index.fps is not None and not np.isclose(
            index.fps,
            candidate.fps,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"{fixture}: frozen feature fps differs from action row")

        blocks: list[np.ndarray] = []
        for role, frame in (
            ("candidate", candidate.frame),
            ("fixed", candidate.fixed_contact_frame),
        ):
            block = None if index is None else index.by_frame.get(frame)
            if block is None:
                block = np.full(frozen_width, np.nan, dtype=np.float64)
                fixture_audit[f"{role}_missing_joins"] += 1
                missing_identities.append({"identity": row.identity, "role": role})
            fixture_audit[f"{role}_missing_cells"] += int(np.isnan(block).sum())
            blocks.append(block)
        physical_blocks.append(np.concatenate(blocks))

    physical = (
        np.vstack(physical_blocks)
        if physical_blocks
        else np.empty((0, frozen_width * 2), dtype=np.float64)
    )
    matrix = np.hstack((existing, physical)).astype(np.float64, copy=False)
    audit: dict[str, Any] = {
        "rows": len(rows),
        "frozen_features_per_contact": frozen_width,
        "fixtures": audit_by_fixture,
        "missing_identities": missing_identities,
    }
    return matrix, names, audit
