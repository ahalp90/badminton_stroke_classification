"""Inter-frame L/R consistency diagnostic for MMPose keypoint output.

RTMPose runs per-frame with no temporal memory, so anatomical L/R can flip
frame-to-frame on hard-to-disambiguate poses (back-facing players, motion
blur, occlusion). This script measures how often that happens, per slot,
per class, per player, across the whole non-unknown clip set.

Detection idea: a real anatomical-L/R confusion in MMPose flips the entire
keypoint set together. For each frame ``t`` and each anatomical pair
``(L_idx, R_idx)``, compare the same-label trajectory smoothness against
the swap-label smoothness:

    d_no_swap = ||L(t) - L(t-1)|| + ||R(t) - R(t-1)||
    d_swap    = ||L(t) - R(t-1)|| + ||R(t) - L(t-1)||

If ``d_swap < margin * d_no_swap`` (swap at least 1/margin times smoother),
the pair votes for flip. Frame ``t`` is flagged a flip if a majority of
valid pairs (>= 4 of 6) vote for swap.

Pairs used: shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12, knees
13/14, ankles 15/16. Eyes 1/2 and ears 3/4 excluded as face-noise-dominated.

Inputs (read from environment):
  - ``BST_X_MMPOSE_NPY_DIR``: flat dir holding ``{stem}_joints.npy`` per stem.
    The companion ``{stem}_failed.npy`` is deliberately NOT consumed: it
    is shape ``(F,)`` and OR-flags both slots together (see
    sticky_anchor.py:283, :320), so it cannot resolve "Top picked, Bottom
    didn't" frames. Per-slot validity is derived from the joints zero-test.
  - ``BST_X_CLIPS_CSV``: ``clips_master.csv``, source of stem -> class +
    player_side mapping. Class filter excludes ``raw_type_en == 'unknown'``.

Outputs (under ``--out-dir``, default
``scratch/architecture_notes/x3d_integration_macro_plan/stage_2_outputs/``):
  - ``keypoint_lr_interframe_per_clip.csv``: one row per (clip, slot).
  - ``keypoint_lr_interframe_diagnostic.md``: text + numeric summary.
  - ``keypoint_lr_interframe_perclass.csv``: per-class aggregate stats.

Per-(player x slot) aggregation is deliberately NOT produced here; resolving
a clip stem to its hitter requires the A/B-to-Top/Bottom remap from
pipeline.player_mapping, which depends on the downcourt flag, set number,
and the set-3 mid-game switch. Run a follow-up join script over the per-clip
CSV produced here if the per-(player x slot) breakdown is wanted.

Single-process by design; per-clip work is light and concurrent writes
to the same CSV would corrupt rows. Run from a node that can read the
``BST_X_MMPOSE_NPY_DIR`` path (engelbart for the canonical clean dir).

Spec: ``scratch/architecture_notes/x3d_integration_macro_plan/stage_2_wrist_loss_assessment.md``
section "Inter-frame L/R consistency check".
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# COCO-17 paired anatomical keypoints (left, right). Order is (L_idx, R_idx).
# Eyes (1,2) and ears (3,4) excluded: face-noise-dominated, dont track body
# structural L/R confusion well.
PAIRS = (
    (5, 6),    # shoulders
    (7, 8),    # elbows
    (9, 10),   # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
)
N_PAIRS = len(PAIRS)
MAJORITY_THRESHOLD = 4  # >= 4 of 6 valid pairs vote for swap to flag a flip

# Slot indices match apply_heuristic.heuristics.sticky_anchor.
SLOT_TOP = 0
SLOT_BOTTOM = 1
SLOT_NAMES = {SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"}

# Per-slot validity is derived from the joints zero-test, not from
# _failed.npy. _failed.npy is (F,) bool and OR-flags both slots together
# (sticky_anchor.py:283, :320), so it cannot resolve "Top picked, Bottom
# didn't" frames. The producer initialises joints with np.zeros and only
# overwrites the slot when its detection picks; an exactly-(0, 0) coordinate
# in float64 normalised joints is the canonical fail sentinel for that slot.
JOINTS_SUFFIX = "_joints.npy"


@dataclass
class PerClipPerSlotStats:
    """Per-(clip, slot) counters and aggregates.

    The single denominator is n_valid_frame_pairs: pairs (t-1, t) with
    both frames slot-valid. There is no per-pair sub-filter on individual
    keypoints because MMPose always exports complete skeletons even at
    low confidence (with weird interpolated positions), so within a
    slot-valid frame, all 17 keypoints are non-zero. The implication is
    that this diagnostic measures L/R consistency against potentially-
    interpolated keypoints; a clip where MMPose hallucinated low-confidence
    positions can still vote for a flip if the hallucinations happen to
    swap-orient. See the spec's "low-confidence interpolated" note.
    """
    clip_stem: str
    slot: int
    raw_type_en: str
    player_side: str
    n_frames: int                 # raw frame count of the clip
    n_valid_frame_pairs: int      # pairs (t-1, t) with both frames slot-valid
    n_flips: int                  # frames flagged via the majority rule
    flip_rate: float              # n_flips / n_valid_frame_pairs (NaN on /0)
    max_flip_run_length: int      # longest consecutive run of flagged frames
    first_flip_frame: int         # 0-based; -1 if no flip
    first_flip_rel_pos: float     # first_flip_frame / n_frames; NaN if no flip


def _load_clip_joints(
    npy_dir: Path,
    stem: str,
    allow_missing: bool,
) -> np.ndarray | None:
    """Load joints for one clip stem; returns array or None when missing+allowed.

    :param npy_dir: flat dir holding ``{stem}_joints.npy``.
    :param stem: clip stem (without suffix).
    :param allow_missing: if True, return None when the file is missing;
                          if False, raise FileNotFoundError.
    :return: joints array shape (F, 2, 17, 2) or None.
    """
    joints_path = npy_dir / f"{stem}{JOINTS_SUFFIX}"
    if not joints_path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing joints npy for stem {stem!r}: {joints_path}")
    joints = np.load(joints_path)
    if joints.ndim != 4 or joints.shape[1] != 2 or joints.shape[2] != 17 or joints.shape[3] != 2:
        raise ValueError(
            f"Unexpected joints shape {joints.shape} for stem {stem!r}; "
            f"expected (F, 2, 17, 2)"
        )
    return joints


def _per_slot_validity(joints_slot: np.ndarray) -> np.ndarray:
    """Per-frame validity mask for a slot, derived from the joints zero-test.

    A slot is valid at frame f if any of its 17 keypoints has a non-zero
    coordinate. The producer's zero-init pattern (sticky_anchor.py:285)
    means an unfilled slot (no detection picked) is exactly (0, 0) for
    every keypoint. A picked slot is overwritten by normalize_joints output;
    MMPose always exports complete skeletons even at low confidence
    (with weird interpolated positions), so a picked slot has all 17
    keypoints non-zero. The all-or-nothing pattern means validity is a
    clean per-frame yes/no.

    :param joints_slot: shape (F, 17, 2) for one slot.
    :return: shape (F,) bool, True where the slot is valid at that frame.
    """
    return (joints_slot != 0.0).any(axis=(1, 2))


def _max_run_length(flip_flags: np.ndarray) -> int:
    """Longest consecutive run of True values in a 1D bool array.

    :param flip_flags: 1D bool array; True = flagged frame.
    :return: max consecutive-True run; 0 if no True values.
    """
    if not flip_flags.any():
        return 0
    # Boundaries: pad with False on each side, find True->False and False->True
    # transitions, run lengths are the gaps between False->True and True->False.
    padded = np.concatenate(([False], flip_flags, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())


def _process_clip_slot(
    joints: np.ndarray,
    slot: int,
    stem: str,
    raw_type_en: str,
    player_side: str,
    margin: float,
) -> PerClipPerSlotStats:
    """Compute the per-clip-per-slot flip stats.

    :param joints: full clip joints array, shape (F, 2, 17, 2), float. Pre-
                   collation joints (variable F per clip), NOT the seq_len=100
                   collated tensor (which would have legitimate trailing zero
                   pad indistinguishable from per-slot fail).
    :param slot: SLOT_TOP or SLOT_BOTTOM.
    :param stem: clip stem.
    :param raw_type_en: stroke class label.
    :param player_side: 'Top' or 'Bottom' (annotation side, NOT the slot we
                        are processing; both slots get processed for every
                        clip, regardless of which side the annotated stroke
                        was on).
    :param margin: swap-vs-no-swap smoothness ratio threshold; pair votes for
                   swap when d_swap < margin * d_no_swap.
    :return: per-(clip, slot) stats record.
    """
    n_frames = joints.shape[0]
    js = joints[:, slot, :, :]               # (F, 17, 2)
    slot_valid = _per_slot_validity(js)       # (F,) bool

    n_valid_frame_pairs = 0
    flip_flags = np.zeros(n_frames, dtype=bool)

    # t-1 -> t comparisons; start at t=1. The slot-valid guard enforces
    # that comparisons happen only between adjacent valid frames; no
    # across-gap comparisons are made (a fail at t-1 or t skips the pair).
    # Within a valid frame, MMPose always exports complete skeletons, so
    # all six anatomical pairs always contribute a vote (no per-pair
    # zero-skip needed).
    for t in range(1, n_frames):
        if not (slot_valid[t - 1] and slot_valid[t]):
            continue
        n_valid_frame_pairs += 1

        votes_for_swap = 0
        for L_idx, R_idx in PAIRS:
            L_prev = js[t - 1, L_idx]
            R_prev = js[t - 1, R_idx]
            L_curr = js[t, L_idx]
            R_curr = js[t, R_idx]

            # Use real L2 norms (sqrt) rather than squared distances. Sums
            # of norms over multiple vectors do not have the same monotonic
            # comparison properties as sums of squared norms; using the
            # spec'd norm keeps the comparison correctly directional.
            d_no_swap = (
                float(np.linalg.norm(L_curr - L_prev))
                + float(np.linalg.norm(R_curr - R_prev))
            )
            d_swap = (
                float(np.linalg.norm(L_curr - R_prev))
                + float(np.linalg.norm(R_curr - L_prev))
            )

            if d_no_swap == 0.0:
                # Both wrists exactly stationary across the frame: swap can
                # never be 'less than 0.5x' of zero. Pair does not vote.
                continue

            if d_swap < margin * d_no_swap:
                votes_for_swap += 1

        if votes_for_swap >= MAJORITY_THRESHOLD:
            flip_flags[t] = True

    n_flips = int(flip_flags.sum())
    flip_rate = (
        n_flips / n_valid_frame_pairs if n_valid_frame_pairs > 0 else float("nan")
    )
    max_run = _max_run_length(flip_flags)
    first_flip_frame = int(np.argmax(flip_flags)) if n_flips > 0 else -1
    first_flip_rel_pos = (
        float(first_flip_frame) / float(n_frames)
        if first_flip_frame >= 0 and n_frames > 0
        else float("nan")
    )

    return PerClipPerSlotStats(
        clip_stem=stem,
        slot=slot,
        raw_type_en=raw_type_en,
        player_side=player_side,
        n_frames=n_frames,
        n_valid_frame_pairs=n_valid_frame_pairs,
        n_flips=n_flips,
        max_flip_run_length=max_run,
        flip_rate=flip_rate,
        first_flip_frame=first_flip_frame,
        first_flip_rel_pos=first_flip_rel_pos,
    )


# Per-player aggregation deliberately not implemented here. Resolving a
# clip stem to the hitter's player name requires the A-vs-B-to-Top-vs-Bottom
# remap from pipeline.player_mapping, which depends on the downcourt flag,
# set number, and the set-3 mid-game switch rally. Out of scope for this
# diagnostic; if the per-(player x slot) breakdown is wanted, run a
# follow-up join script that consumes the per-clip CSV produced here.


def _summarise_to_md(
    df: pd.DataFrame,
    out_path: Path,
    margin: float,
) -> None:
    """Write the markdown diagnostic summary."""
    n_clip_slot_rows = len(df)
    n_clips = df["clip_stem"].nunique()

    # Per-slot headline.
    by_slot = df.groupby("slot")["flip_rate"].agg(["mean", "median", "std", "count"])
    by_slot = by_slot.rename(index={SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"})

    # Per-class per-slot.
    by_class_slot = (
        df.groupby(["raw_type_en", "slot"])["flip_rate"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_class_slot["slot"] = by_class_slot["slot"].map({SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"})

    # Run-length distribution per slot.
    runs = df.groupby("slot")["max_flip_run_length"].describe()
    runs = runs.rename(index={SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"})

    # First-flip relative-position distribution per slot.
    rel_pos = df.dropna(subset=["first_flip_rel_pos"]).groupby("slot")[
        "first_flip_rel_pos"
    ].describe()
    rel_pos = rel_pos.rename(index={SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"})

    lines: list[str] = []
    lines.append("# Inter-frame L/R consistency diagnostic\n")
    lines.append(
        f"Generated by `keypoint_lr_interframe_diagnostic.py`. "
        f"Margin = {margin} (swap is at least {1/margin:.2f}x smoother to count).\n"
    )
    lines.append(f"- {n_clips} clips processed (non-unknown).")
    lines.append(f"- {n_clip_slot_rows} (clip, slot) rows; both slots are processed per clip.\n")

    lines.append("\n## Per-slot headline flip rate\n")
    lines.append(by_slot.to_markdown(floatfmt=".4f"))
    lines.append("\n")

    lines.append("\n## Per-class per-slot flip rate\n")
    lines.append(by_class_slot.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n")

    lines.append("\n## Max consecutive-flip run-length distribution per slot\n")
    lines.append(runs.to_markdown(floatfmt=".2f"))
    lines.append("\n")

    lines.append("\n## First-flip relative position (within clip) per slot\n")
    lines.append(rel_pos.to_markdown(floatfmt=".4f"))
    lines.append("\n")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npy-dir",
        type=Path,
        default=None,
        help="Path to the flat MMPose npy dir. Defaults to BST_X_MMPOSE_NPY_DIR env.",
    )
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=None,
        help="Path to clips_master.csv. Defaults to BST_X_CLIPS_CSV env.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "scratch/architecture_notes/x3d_integration_macro_plan/stage_2_outputs"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Swap-vs-no-swap smoothness ratio threshold. Pair votes for swap "
             "when d_swap < margin * d_no_swap. Default 0.5 (swap must be 2x smoother). "
             "Must be > 0.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip clips whose joints/failed npy files are missing, instead "
             "of failing hard. Off by default to surface dataset-integrity issues.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process only this many clips (for smoke testing).",
    )
    args = parser.parse_args()

    if args.margin <= 0.0:
        raise SystemExit(f"--margin must be > 0; got {args.margin}")

    # Resolve npy_dir + clips_csv from env if not given.
    npy_dir = args.npy_dir
    if npy_dir is None:
        env_npy = os.environ.get("BST_X_MMPOSE_NPY_DIR")
        if not env_npy:
            raise SystemExit(
                "BST_X_MMPOSE_NPY_DIR env var not set and --npy-dir not provided."
            )
        npy_dir = Path(env_npy)
    if not npy_dir.is_dir():
        raise SystemExit(f"npy-dir does not exist or is not a dir: {npy_dir}")

    clips_csv = args.clips_csv
    if clips_csv is None:
        env_csv = os.environ.get("BST_X_CLIPS_CSV")
        if not env_csv:
            raise SystemExit(
                "BST_X_CLIPS_CSV env var not set and --clips-csv not provided."
            )
        clips_csv = Path(env_csv)
    if not clips_csv.is_file():
        raise SystemExit(f"clips-csv does not exist or is not a file: {clips_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips_df = pd.read_csv(clips_csv)
    # Loose-match guard: strip whitespace, lowercase before comparing.
    is_unknown = (
        clips_df["raw_type_en"].astype(str).str.strip().str.lower() == "unknown"
    )
    clips_df = clips_df[~is_unknown].reset_index(drop=True)

    if args.limit > 0:
        clips_df = clips_df.head(args.limit)

    rows: list[PerClipPerSlotStats] = []
    n_missing = 0

    for _, row in tqdm(clips_df.iterrows(), total=len(clips_df), desc="clips"):
        stem = str(row["clip_stem"])
        raw_type_en = str(row.get("raw_type_en", ""))
        player_side = str(row.get("player_side", ""))

        joints = _load_clip_joints(npy_dir, stem, allow_missing=args.allow_missing)
        if joints is None:
            n_missing += 1
            continue

        for slot in (SLOT_TOP, SLOT_BOTTOM):
            rec = _process_clip_slot(
                joints=joints,
                slot=slot,
                stem=stem,
                raw_type_en=raw_type_en,
                player_side=player_side,
                margin=args.margin,
            )
            rows.append(rec)

    df = pd.DataFrame([r.__dict__ for r in rows])

    per_clip_path = args.out_dir / "keypoint_lr_interframe_per_clip.csv"
    df.to_csv(per_clip_path, index=False)

    # Per-class per-slot aggregate.
    perclass = (
        df.groupby(["raw_type_en", "slot"])
        .agg(
            n_clips=("clip_stem", "nunique"),
            mean_flip_rate=("flip_rate", "mean"),
            median_flip_rate=("flip_rate", "median"),
            mean_max_run=("max_flip_run_length", "mean"),
        )
        .reset_index()
    )
    perclass["slot"] = perclass["slot"].map({SLOT_TOP: "Top", SLOT_BOTTOM: "Bottom"})
    perclass.to_csv(args.out_dir / "keypoint_lr_interframe_perclass.csv", index=False)

    md_path = args.out_dir / "keypoint_lr_interframe_diagnostic.md"
    _summarise_to_md(df, md_path, args.margin)

    print(f"Processed {df['clip_stem'].nunique()} clips; missing-skipped: {n_missing}.")
    print(f"Per-clip CSV: {per_clip_path}")
    print(f"Markdown summary: {md_path}")
    print(
        "Per-(player x slot) aggregation deliberately not produced here; "
        "see header note for the follow-up join recipe."
    )


if __name__ == "__main__":
    main()
