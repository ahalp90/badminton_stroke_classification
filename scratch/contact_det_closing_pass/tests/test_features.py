"""Tests for joining frozen per-frame features to action rows."""

from __future__ import annotations

import lzma
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det.scripts.freeze_tree_contact_features import (
    _feature_family_names,
)
from scratch.contact_det_closing_pass.scripts.features import join_physical_features
from scratch.contact_det_followup.scripts.score_start_model import (
    ACTION_FEATURE_NAMES,
    ActionRow,
)
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import CandidateRow


def _feature_names() -> tuple[str, ...]:
    families = _feature_family_names()
    return tuple(families["physics"] + families["missingness"])


def _action(
    frame: int,
    *,
    fixed_frame: int = 100,
    fixture: str = "sset_01",
    fps: float = 25.0,
) -> ActionRow:
    candidate = CandidateRow(
        fixture,
        "A",
        fps,
        0,
        0,
        150,
        0,
        fixed_frame,
        frame,
        0.8,
        0.9,
        False,
        "Top",
        "Bot",
        (0.1,) * 9,
    )
    return ActionRow(candidate, "add", (0.1,) * 10)


def _write_features(root: Path, fixture: str, rows: list[tuple[int, float, float]]) -> None:
    names = _feature_names()
    dtype = np.dtype(
        [
            ("fixture", "S16"),
            ("frame", "<i4"),
            ("fps", "<f4"),
            *[(name, "<f4") for name in names],
        ]
    )
    values = np.zeros(len(rows), dtype=dtype)
    for index, (frame, first, second) in enumerate(rows):
        values[index]["fixture"] = fixture.encode("ascii")
        values[index]["frame"] = frame
        values[index]["fps"] = 25.0
        values[index][names[0]] = first
        values[index][names[1]] = second
        for name in names[2:]:
            values[index][name] = float(frame)
    path = root / "videos" / fixture / "contact_features.npy.xz"
    path.parent.mkdir(parents=True)
    with lzma.open(path, "wb") as handle:
        np.save(handle, values, allow_pickle=False)


def test_join_uses_frame_identity_independent_of_file_order(tmp_path: Path) -> None:
    _write_features(tmp_path, "sset_01", [(100, 10.0, 20.0), (20, 1.0, 2.0)])

    matrix, names, audit = join_physical_features((_action(20),), tmp_path)

    assert matrix.shape == (1, 180)
    assert matrix.dtype == np.float64
    assert names[:10] == ACTION_FEATURE_NAMES
    assert names[10] == "candidate__shuttle_vx_t-10"
    assert names[94] == "candidate__wrist_valid_bot_t+10"
    assert names[95] == "fixed__shuttle_vx_t-10"
    assert matrix[0, 10] == 1.0
    assert matrix[0, 11] == 2.0
    assert matrix[0, 95] == 10.0
    assert audit["fixtures"]["sset_01"] == {
        "candidate_missing_joins": 0,
        "fixed_missing_joins": 0,
        "candidate_missing_cells": 0,
        "fixed_missing_cells": 0,
    }


def test_join_preserves_nan_and_zero_and_keeps_missing_rows(tmp_path: Path) -> None:
    _write_features(tmp_path, "sset_01", [(100, 10.0, 20.0), (20, np.nan, 0.0)])
    missing = _action(30)

    matrix, _names, audit = join_physical_features((_action(20), missing), tmp_path)

    assert np.isnan(matrix[0, 10])
    assert matrix[0, 11] == 0.0
    assert matrix.shape[0] == 2
    assert np.isnan(matrix[1, 10:95]).all()
    assert matrix[1, 95] == 10.0
    assert audit["fixtures"]["sset_01"]["candidate_missing_joins"] == 1
    assert audit["fixtures"]["sset_01"]["fixed_missing_joins"] == 0
    assert audit["fixtures"]["sset_01"]["candidate_missing_cells"] == 86
    assert audit["missing_identities"] == [{"identity": missing.identity, "role": "candidate"}]


def test_duplicate_source_frame_is_rejected(tmp_path: Path) -> None:
    _write_features(tmp_path, "sset_01", [(20, 1.0, 2.0), (20, 3.0, 4.0)])

    with pytest.raises(ValueError, match="duplicate"):
        join_physical_features((_action(20),), tmp_path)


def test_infinite_source_cell_is_rejected(tmp_path: Path) -> None:
    _write_features(tmp_path, "sset_01", [(20, np.inf, 2.0), (100, 3.0, 4.0)])

    with pytest.raises(ValueError, match="infinity"):
        join_physical_features((_action(20),), tmp_path)


def test_fps_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_features(tmp_path, "sset_01", [(20, 1.0, 2.0), (100, 3.0, 4.0)])
    action = _action(20, fps=30.0)

    with pytest.raises(ValueError, match="fps"):
        join_physical_features((action,), tmp_path)
