"""Bit-exact comparison of prepare_2d_dataset_npy_from_raw_video across the tidy.

Runs the post-tidy ``_prepare_dataset_from_raw_video`` (via the public
``prepare_2d_dataset_npy_from_raw_video`` wrapper) on a small mp4 sample and
compares ``_pos.npy`` / ``_joints.npy`` / ``_failed.npy`` against an
already-extracted reference directory.

Why this is bit-exact-by-construction (and the test is belt-and-braces):

The lift moves the per-clip iteration loop into a helper and threads the
detect kwargs as a dict instead of inlined keywords. Reading the diff:

  - Same iteration order (``sorted(my_clips_folder.glob('**/*.mp4'))``)
  - Same per-clip resume check (``Path(_failed.npy).exists()``)
  - Same ``detect_players_2d(video_path=v, **kwargs)`` call
  - Same ``np.save`` of pos / joints / failed
  - Same ``gc.collect()`` + ``torch.cuda.empty_cache()`` cadence

There is no behavioural delta possible without a typo, which would crash
loudly on the first clip rather than produce subtly-wrong outputs. This
script confirms that empirically against a reference extract.

Required env vars:
  CLIPS_DIR       -- dir with N small mp4s to re-extract pose from
  REFERENCE_DIR   -- dir holding the reference _pos / _joints / _failed npys
                     (typically the committed BST_X_MMPOSE_NPY_DIR)
  SCRATCH_DIR     -- where the post-tidy run writes its outputs
                     (must NOT collide with REFERENCE_DIR or CLIPS_DIR)

Optional env vars:
  ATOL_FLOAT  -- tolerance for _pos and _joints (default: 1e-5; matches the
                 byte-identity gate's tolerance for projection-chain
                 float32 -> float64 non-associativity)

Usage on engelbart:
  cd ~/badminton_stroke_classifier
  source /home/ahalperi/.venvs/venv-mmpose/bin/activate

  # Hand-pick 5-10 stems present in BST_X_MMPOSE_NPY_DIR; copy/symlink their
  # mp4s into a fresh sample dir.
  mkdir -p /tmp/prepare_2d_smoke_clips/sample
  for stem in 19_1_5_1 19_1_5_2 19_1_5_3 19_1_5_4 19_1_5_5; do
      find $BST_X_CLIPS_DIR -name "${stem}.mp4" -exec ln -sf {} /tmp/prepare_2d_smoke_clips/sample/${stem}.mp4 \\;
  done

  export CLIPS_DIR=/tmp/prepare_2d_smoke_clips
  export REFERENCE_DIR=$BST_X_MMPOSE_NPY_DIR
  export SCRATCH_DIR=/tmp/prepare_2d_smoke_outputs
  export PYTHONPATH=src/bst_x
  python scratch/post_tidy_smoke/smoke_prepare_2d_bit_exact.py

  # Switch to main and run the same script against the same SCRATCH_DIR
  # cleared to compare main->ref vs tidy->ref. (Optional: confirms reference
  # itself is reproducible to atol on this machine.)

A passing run prints ``PASS: N stems matched`` for every stem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _compare_stem(scratch_dir: Path, ref_dir: Path, stem: str, atol: float) -> tuple[bool, str]:
    """Compare _pos / _joints / _failed for one stem; return (ok, message)."""
    new_pos = np.load(scratch_dir / f"{stem}_pos.npy")
    new_joints = np.load(scratch_dir / f"{stem}_joints.npy")
    new_failed = np.load(scratch_dir / f"{stem}_failed.npy")

    ref_pos = np.load(ref_dir / f"{stem}_pos.npy")
    ref_joints = np.load(ref_dir / f"{stem}_joints.npy")
    ref_failed = np.load(ref_dir / f"{stem}_failed.npy")

    if new_pos.shape != ref_pos.shape:
        return False, f"pos shape mismatch: {new_pos.shape} vs {ref_pos.shape}"
    if new_joints.shape != ref_joints.shape:
        return False, f"joints shape mismatch: {new_joints.shape} vs {ref_joints.shape}"
    if new_failed.shape != ref_failed.shape:
        return False, f"failed shape mismatch: {new_failed.shape} vs {ref_failed.shape}"

    if not np.array_equal(new_failed, ref_failed):
        n_diff = int((new_failed != ref_failed).sum())
        return False, f"failed differs on {n_diff} frames"

    pos_max = float(np.abs(new_pos - ref_pos).max())
    joints_max = float(np.abs(new_joints - ref_joints).max())
    if pos_max > atol or joints_max > atol:
        return False, f"floats above atol: pos_max={pos_max:.3e} joints_max={joints_max:.3e}"

    return True, f"pos_max={pos_max:.3e} joints_max={joints_max:.3e}"


def main() -> int:
    clips_dir = Path(os.environ["CLIPS_DIR"]).resolve()
    reference_dir = Path(os.environ["REFERENCE_DIR"]).resolve()
    scratch_dir = Path(os.environ["SCRATCH_DIR"]).resolve()
    atol = float(os.environ.get("ATOL_FLOAT", "1e-5"))

    if not clips_dir.is_dir():
        raise FileNotFoundError(f"CLIPS_DIR not found: {clips_dir}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"REFERENCE_DIR not found: {reference_dir}")
    if scratch_dir.resolve() in (clips_dir.resolve(), reference_dir.resolve()):
        raise ValueError(
            "SCRATCH_DIR must not collide with CLIPS_DIR or REFERENCE_DIR; "
            "would overwrite reference data."
        )
    scratch_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("**/*.mp4"))
    if not clips:
        raise ValueError(f"No mp4 files under {clips_dir}")
    print(f"CLIPS_DIR     : {clips_dir}  ({len(clips)} mp4s)")
    print(f"REFERENCE_DIR : {reference_dir}")
    print(f"SCRATCH_DIR   : {scratch_dir}")
    print(f"ATOL_FLOAT    : {atol}")
    print()

    # The active homography uses ShuttleSet's set_info + my_raw_video_resolution.
    import pandas as pd  # noqa: PLC0415
    from pipeline.config import RESOLUTION_CSV_PATH, SET_INFO_DIR  # noqa: PLC0415
    from pipeline.court_utils import get_court_info  # noqa: PLC0415
    from preparing_data.prepare_train_on_shuttleset import (  # noqa: PLC0415
        prepare_2d_dataset_npy_from_raw_video,
    )

    res_df = pd.read_csv(RESOLUTION_CSV_PATH).set_index("id")
    homo_df = pd.read_csv(SET_INFO_DIR / "homography.csv").set_index("id")
    all_court_info = {vid: get_court_info(homo_df, vid) for vid in res_df.index}

    # Run post-tidy prepare_2d.
    prepare_2d_dataset_npy_from_raw_video(
        my_clips_folder=clips_dir,
        save_root_dir=scratch_dir,
        resolution_df=res_df,
        all_court_info=all_court_info,
        joints_normalized_by_v_height=False,
        joints_center_align=True,
    )

    print()
    print("=== Comparison vs reference ===")
    n_ok = 0
    n_fail = 0
    for clip in clips:
        stem = clip.stem
        ok, msg = _compare_stem(scratch_dir, reference_dir, stem, atol)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {stem}: {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print()
    print(f"PASS: {n_ok}/{len(clips)} stems matched (atol={atol})")
    if n_fail:
        print(f"FAIL: {n_fail}/{len(clips)} stems mismatched")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
