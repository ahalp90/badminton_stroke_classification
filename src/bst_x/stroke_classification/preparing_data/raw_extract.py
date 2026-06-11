"""Raw MMPose extraction for a specified subset of clips.

Sibling to ``prepare_train_on_shuttleset.py``'s 2D pose step, but:

1. Operates only on the clip stems in a supplied list (e.g. the Phase 1
   "busted" set from ``validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py``).
2. Applies no filtering -- no court projection, no "2 players on court"
   requirement, no normalization. Saves everything MMPose returns.
3. Emits five raw numpy arrays per clip:

   - ``{stem}_raw_kps.npy``        ``(F, N_max, J, 2)``  float32, NaN-padded
   - ``{stem}_raw_bboxes.npy``     ``(F, N_max, 4)``     float32, NaN-padded
   - ``{stem}_raw_scores.npy``     ``(F, N_max)``        float32, NaN-padded
   - ``{stem}_raw_kp_scores.npy``  ``(F, N_max, J)``     float32, NaN-padded
   - ``{stem}_raw_ndet.npy``       ``(F,)``              int8 detection count

``_raw_ndet.npy`` is the resume marker, saved last; its presence means all
five outputs landed cleanly for this clip. NaN padding is used (not zero)
so real detected coordinates at origin are not ambiguous with padding.

The raw outputs feed downstream heuristic iteration (``apply_heuristic.py``
and the ``sticky_anchor`` variant, both out of scope for this module).

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \\
        python -m preparing_data.raw_extract --help
"""

from mmpose.apis import MMPoseInferencer

import argparse
import gc
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import CLIPS_OUTPUT_DIR

J = 17  # COCO keypoints returned by MMPoseInferencer("human") / RTMPose-L


def extract_raw_frame(
    result: dict,
    n_max: int,
    clip_stem: str,
    frame_num: int,
    over_det_warned: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return per-frame raw arrays, NaN-padded to ``n_max`` along the detect dim.

    If MMPose returns more than ``n_max`` detections in a frame, keep the
    top-``n_max`` by ``bbox_score``. Log a once-per-clip warning.
    """
    preds = list(result["predictions"][0])
    n = len(preds)
    if n > n_max:
        preds = sorted(preds, key=lambda p: -float(p["bbox_score"]))[:n_max]
        n = n_max
        if clip_stem not in over_det_warned:
            print(
                f"  WARN: {clip_stem} frame {frame_num} had >{n_max} detections; "
                f"truncating to top-{n_max} by bbox_score"
            )
            over_det_warned.add(clip_stem)

    kps = np.full((n_max, J, 2), np.nan, dtype=np.float32)
    bboxes = np.full((n_max, 4), np.nan, dtype=np.float32)
    scores = np.full((n_max,), np.nan, dtype=np.float32)
    kp_scores = np.full((n_max, J), np.nan, dtype=np.float32)

    for i, person in enumerate(preds):
        kps[i] = np.asarray(person["keypoints"], dtype=np.float32)
        bboxes[i] = np.asarray(person["bbox"][0], dtype=np.float32)
        # bbox_score may arrive as a scalar or list-wrapped; squeeze to scalar.
        scores[i] = np.float32(np.asarray(person["bbox_score"]).squeeze())
        kp_scores[i] = np.asarray(person["keypoint_scores"], dtype=np.float32)

    return kps, bboxes, scores, kp_scores, n


def extract_one_clip(
    inferencer: MMPoseInferencer,
    video_path: Path,
    save_branch: str,
    n_max: int,
    over_det_warned: set[str],
) -> None:
    """Run MMPose on one clip and save the five raw arrays."""
    kps_ls: list[np.ndarray] = []
    bboxes_ls: list[np.ndarray] = []
    scores_ls: list[np.ndarray] = []
    kp_scores_ls: list[np.ndarray] = []
    ndet_ls: list[int] = []

    for frame_num, result in enumerate(inferencer(str(video_path), show=False)):
        kps, bboxes, scores, kp_scores, n = extract_raw_frame(
            result, n_max, video_path.stem, frame_num, over_det_warned,
        )
        kps_ls.append(kps)
        bboxes_ls.append(bboxes)
        scores_ls.append(scores)
        kp_scores_ls.append(kp_scores)
        ndet_ls.append(n)

    np.save(save_branch + "_raw_kps.npy", np.stack(kps_ls))
    np.save(save_branch + "_raw_bboxes.npy", np.stack(bboxes_ls))
    np.save(save_branch + "_raw_scores.npy", np.stack(scores_ls))
    np.save(save_branch + "_raw_kp_scores.npy", np.stack(kp_scores_ls))
    # _raw_ndet.npy is saved last so its presence is a reliable resume marker
    # for all five outputs.
    np.save(save_branch + "_raw_ndet.npy", np.asarray(ndet_ls, dtype=np.int8))


def inspect_one_clip(inferencer: MMPoseInferencer, video_path: Path) -> None:
    """Print the structure of the first frame's MMPose result, then return."""
    print(f"Inspect: {video_path}")
    gen = inferencer(str(video_path), show=False)
    result = next(iter(gen))
    preds = result["predictions"][0]
    print(f"Number of detections in frame 0: {len(preds)}")
    if not preds:
        print("No detections in frame 0; try a different clip.")
        return
    p0 = preds[0]
    print(f"Keys on detection[0]: {sorted(p0.keys())}")
    for key, value in p0.items():
        try:
            arr = np.asarray(value)
            shape = arr.shape if arr.dtype != object else f"object(len={len(value)})"
            dtype = arr.dtype
        except (ValueError, TypeError):
            shape = "<unknown>"
            dtype = type(value).__name__
        print(f"  {key!r}: dtype={dtype} shape={shape}")
    print("\nSummary of detection[0]:")
    pprint({
        k: (v if not isinstance(v, list) or len(v) < 4 else f"list(len={len(v)})")
        for k, v in p0.items()
    })


def build_stem_to_path(clips_dir: Path) -> dict[str, Path]:
    """Map every .mp4 stem under ``clips_dir`` to its Path (recursive)."""
    return {mp4.stem: mp4 for mp4 in clips_dir.glob("**/*.mp4")}


def load_stems(path: Path) -> list[str]:
    with path.open() as fh:
        return [line.strip() for line in fh if line.strip()]


RAW_SUFFIXES = (
    "_raw_kps.npy",
    "_raw_bboxes.npy",
    "_raw_scores.npy",
    "_raw_kp_scores.npy",
    "_raw_ndet.npy",
)


def _stored_n_max(save_branch: str) -> int | None:
    """Peek at `_raw_bboxes.npy` to recover the N_max dimension of an existing
    extract, or return None if the file is absent or unreadable.
    """
    path = Path(save_branch + "_raw_bboxes.npy")
    if not path.exists():
        return None
    try:
        return int(np.load(path, mmap_mode="r").shape[1])
    except (OSError, ValueError, IndexError):
        return None


def _clear_raw_files(save_branch: str) -> None:
    """Delete all five raw outputs for a given stem, if present."""
    for suffix in RAW_SUFFIXES:
        Path(save_branch + suffix).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--clips-dir", type=Path, default=CLIPS_OUTPUT_DIR,
        help="Root of raw .mp4 clips (scanned recursively). "
             "Defaults to pipeline.config.CLIPS_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--clip-stems-file", type=Path, required=True,
        help="One clip stem per line (output of find_busted_clips.py).",
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True,
        help="Output dir for raw per-clip .npy files. Must not collide with "
             "the primary filtered flat dir.",
    )
    parser.add_argument(
        "--n-max", type=int, default=8,
        help="Max detections per frame. Excess is truncated by bbox_score.",
    )
    parser.add_argument(
        "--inspect-result", action="store_true",
        help="Print the first frame's MMPose result structure on one clip, "
             "then exit. Run this once before any batch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve stems to mp4 paths and exit without running MMPose.",
    )
    parser.add_argument(
        "--force-reextract", action="store_true",
        help="If an existing clip's stored N_max differs from --n-max, "
             "delete its five raw files and re-extract. Without this flag, "
             "a shape mismatch is a hard error so we can't silently mix "
             "N_max widths in the same save-dir.",
    )
    args = parser.parse_args()

    if not args.clips_dir.is_dir():
        parser.error(f"clips-dir not found: {args.clips_dir}")
    if not args.clip_stems_file.exists():
        parser.error(f"clip-stems-file not found: {args.clip_stems_file}")

    stems = load_stems(args.clip_stems_file)
    print(f"Loaded {len(stems)} stems from {args.clip_stems_file}")

    stem_to_path = build_stem_to_path(args.clips_dir)
    print(f"Indexed {len(stem_to_path)} mp4 files under {args.clips_dir}")

    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for stem in stems:
        path = stem_to_path.get(stem)
        if path is None:
            missing.append(stem)
        else:
            resolved.append((stem, path))
    print(f"Resolved {len(resolved)} / {len(stems)} stems to mp4 paths")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")

    if args.dry_run:
        print("\nDry run: showing first 5 resolved pairs and exiting.")
        for stem, path in resolved[:5]:
            print(f"  {stem}  ->  {path}")
        return 0

    if args.inspect_result:
        if not resolved:
            print("No resolved clips to inspect; aborting.")
            return 1
        inferencer = MMPoseInferencer("human")
        inspect_one_clip(inferencer, resolved[0][1])
        return 0

    args.save_dir.mkdir(parents=True, exist_ok=True)

    inferencer = MMPoseInferencer("human")
    over_det_warned: set[str] = set()
    skipped = 0
    reextracted_mismatch = 0

    for stem, video_path in tqdm(resolved, desc="raw_extract", unit="clip"):
        save_branch = str(args.save_dir / stem)
        ndet_path = Path(save_branch + "_raw_ndet.npy")
        if ndet_path.exists():
            stored = _stored_n_max(save_branch)
            if stored is None:
                # _raw_ndet.npy present but bboxes missing or unreadable.
                # Treat as a corrupted leftover and re-extract from scratch.
                _clear_raw_files(save_branch)
            elif stored == args.n_max:
                skipped += 1
                continue
            elif args.force_reextract:
                _clear_raw_files(save_branch)
                reextracted_mismatch += 1
            else:
                print(
                    f"\nERROR: existing output for {stem} has N_max={stored} "
                    f"but --n-max={args.n_max}. Rerun with --force-reextract "
                    f"to delete and re-extract mismatched clips, or clear the "
                    f"save-dir manually."
                )
                return 1

        extract_one_clip(
            inferencer=inferencer,
            video_path=video_path,
            save_branch=save_branch,
            n_max=args.n_max,
            over_det_warned=over_det_warned,
        )

        # Match the 2D pose step's per-clip GPU cleanup to prevent fragmentation
        # across the batch. Skips above don't allocate on GPU so they fall
        # through without touching the cache.
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\nDone. Processed {len(resolved) - skipped}, skipped {skipped} "
        f"(had _raw_ndet.npy). Missing mp4 for {len(missing)} stems."
    )
    if reextracted_mismatch:
        print(
            f"  Re-extracted {reextracted_mismatch} clip(s) whose stored "
            f"N_max differed from --n-max={args.n_max} (--force-reextract)."
        )
    if over_det_warned:
        print(
            f"Over-detection warnings fired for {len(over_det_warned)} clip(s) "
            f"(frames with >{args.n_max} detections, truncated to top "
            f"{args.n_max} by bbox_score):"
        )
        for stem in sorted(over_det_warned):
            print(f"  {stem}")
    else:
        print("No over-detection warnings fired.")
    return 0


# NOTE: A 3D extraction path (via MMPoseInferencer(pose3d="human3d")) is
# deliberately out of scope for this module's current phase. If needed
# later, mirror the dual-generator structure from
# prepare_train_on_shuttleset.py:detect_players_3d and note the per-clip
# MMPose reload workaround documented there.


if __name__ == "__main__":
    sys.exit(main())
