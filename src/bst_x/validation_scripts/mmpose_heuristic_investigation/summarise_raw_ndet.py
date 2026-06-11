#!/usr/bin/env python3
"""Per-clip and aggregate breakdown of MMPose detection counts.

Walks a ``raw_extract`` save-dir and reports:

1. Per-clip stats (clip sorted by max detections): F, n_max, n_mean,
   hi_max/hi_mean (detections with bbox_score >= threshold), and the
   frame counts at each of several useful thresholds.
2. Aggregate histogram of raw ndet across all frames.
3. Aggregate histogram of score-filtered detection counts per frame.

The score-filtered counts are usually closer to "players plus umpire"
because low-score detections (distant crowd, motion blur ghosts) get
dropped. Players on broadcast footage almost always score >= 0.7.

Usage:

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/summarise_raw_ndet.py \\
        --save-dir /scratch/.../flat_raw_phase1 \\
        --score-threshold 0.5 \\
        --top 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="Flat raw_extract save-dir with *_raw_ndet.npy etc.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="bbox_score cutoff for 'likely real person' count "
                             "(default 0.5).")
    parser.add_argument("--top", type=int, default=25,
                        help="Show N most-crowded clips (default 25).")
    parser.add_argument("--bins", type=str, default="4,8,12,16,20",
                        help="Comma list of thresholds for the 'frames with "
                             "ndet >= t' per-clip columns (default 4,8,12,16,20).")
    args = parser.parse_args()

    if not args.save_dir.is_dir():
        parser.error(f"save-dir not found: {args.save_dir}")

    thresholds = [int(x) for x in args.bins.split(",")]
    ndet_files = sorted(args.save_dir.glob("*_raw_ndet.npy"))
    if not ndet_files:
        print(f"No *_raw_ndet.npy under {args.save_dir}")
        return 1

    rows: list[dict] = []
    all_ndet: list[int] = []
    all_hi: list[int] = []

    for ndet_p in ndet_files:
        stem = ndet_p.name.removesuffix("_raw_ndet.npy")
        ndet = np.load(ndet_p)
        scores = np.load(ndet_p.parent / f"{stem}_raw_scores.npy")  # (F, N_max)
        hi = np.nansum(scores >= args.score_threshold, axis=1).astype(int)

        row = {
            "stem": stem,
            "F": int(len(ndet)),
            "n_max": int(ndet.max()),
            "n_mean": float(ndet.mean()),
            "n_median": float(np.median(ndet)),
            "hi_max": int(hi.max()),
            "hi_mean": float(hi.mean()),
        }
        for t in thresholds:
            row[f"n_ge_{t}"] = int((ndet >= t).sum())
        rows.append(row)

        all_ndet.extend(ndet.tolist())
        all_hi.extend(hi.tolist())

    rows.sort(key=lambda r: (-r["n_max"], -r["n_mean"]))

    print(f"Scanned {len(rows):,} clips, {len(all_ndet):,} total frames.")
    print(f"Score threshold for 'hi' columns: >= {args.score_threshold}")
    print()

    header_thresh = "  ".join(f">={t}" for t in thresholds)
    print(f"--- Top {args.top} most-crowded clips (by max ndet) ---")
    print(
        f"{'stem':<18} {'F':>4} {'n_max':>6} {'n_mean':>7} "
        f"{'hi_max':>6} {'hi_mean':>8}  {header_thresh}"
    )
    for r in rows[: args.top]:
        ge_cols = "  ".join(f"{r[f'n_ge_{t}']:>3}" for t in thresholds)
        print(
            f"{r['stem']:<18} {r['F']:>4} {r['n_max']:>6} "
            f"{r['n_mean']:>7.2f} {r['hi_max']:>6} {r['hi_mean']:>8.2f}  {ge_cols}"
        )

    # Aggregate histograms.
    all_ndet_arr = np.asarray(all_ndet)
    all_hi_arr = np.asarray(all_hi)

    def _print_hist(label: str, arr: np.ndarray) -> None:
        vals, counts = np.unique(arr, return_counts=True)
        total = len(arr)
        print(f"\n--- {label} across {total:,} frames ---")
        print(f"  {'count':>6}  {'frames':>8}  {'pct':>7}  {'cumulative':>10}")
        cum = 0
        for v, c in zip(vals, counts):
            cum += c
            pct = c / total * 100
            cum_pct = cum / total * 100
            print(f"  {int(v):>6d}  {c:>8,}  {pct:>6.2f}%  {cum_pct:>9.2f}%")

    _print_hist("raw ndet", all_ndet_arr)
    _print_hist(
        f"score-filtered detections (score >= {args.score_threshold})",
        all_hi_arr,
    )

    # Per-clip hi_mean distribution — useful for spotting clips where
    # "only 2 people are ever really on camera" vs "chaos".
    hi_means = np.asarray([r["hi_mean"] for r in rows])
    print()
    print("--- Per-clip hi_mean distribution (detections with score >= "
          f"{args.score_threshold}, averaged within clip) ---")
    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        print(f"  p{int(q * 100):>3}  {float(np.quantile(hi_means, q)):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
