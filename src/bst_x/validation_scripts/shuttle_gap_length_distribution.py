"""Shuttle-gap length distribution.

For every contiguous run of ``visibility=0`` frames in a per-clip
shuttle NPY, records the gap length (in frames) and aggregates across
the full 32,203-clip set. Helps disambiguate the cause of the high
shuttle-missing rate (6.34% post-inpaint per the Phase-2 validation
analysis):

- **Cluster around 3-5 frames**: motion blur / brief occlusion. The
  rectification module's typical sweet spot.
- **Cluster around 10-30 frames**: off-screen excursions of typical
  badminton-arc duration. Inpaint can't fill these because there's
  no shuttle in the image to detect.
- **Heavy tail beyond ~30 frames**: inpaint window exceeded; the
  rectification module gives up and leaves these as visibility=0.
  Suggests trajectory extrapolation (rather than just masking) is
  worth pursuing as a future direction.

Usage::

    python -m validation_scripts.shuttle_gap_length_distribution \\
        --shuttle-dir /scratch/comp320a/ShuttleSet/shuttle_npy_flat
"""
from __future__ import annotations

import argparse
import csv
import socket
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
)
DEFAULT_CLIPS_CSV = Path(
    "/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv"
)


def _load_keep_stems(clips_csv: Path) -> set[str]:
    """Return the set of clip_stems with raw_type_en != 'unknown'.

    The 1,278 unknown clips are excluded from training and from all
    per-stroke analyses, so we drop them here too. Without this
    filter, the gap stats are dominated by clips with no real
    stroke-type signal.
    """
    keep: set[str] = set()
    with clips_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("raw_type_en", "").strip().lower() != "unknown":
                keep.add(row["clip_stem"])
    return keep


def _gap_lengths(visibility: np.ndarray) -> np.ndarray:
    """Return array of contiguous-zero run lengths in ``visibility``."""
    if visibility.size == 0:
        return np.empty(0, dtype=np.int64)
    is_missing = visibility == 0
    edges = np.diff(np.concatenate([[0], is_missing.astype(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return (ends - starts).astype(np.int64)


def _percentiles(values: np.ndarray, qs=(50, 75, 90, 95, 99, 99.9)) -> dict:
    if values.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


# Length-class buckets used for the interpretive summary.
LENGTH_CLASSES = [
    ("1-2 frames (single-event blip)", 1, 2),
    ("3-5 frames (motion-blur band)", 3, 5),
    ("6-10 frames (brief occlusion)", 6, 10),
    ("11-30 frames (off-screen-arc band)", 11, 30),
    ("31-60 frames (sustained absence)", 31, 60),
    ("61+ frames (inpaint window exceeded)", 61, 10**9),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--shuttle-dir",
        type=Path,
        default=Path("/scratch/comp320a/ShuttleSet/shuttle_npy_flat"),
        help="Flat per-clip shuttle NPY dir.",
    )
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=DEFAULT_CLIPS_CSV,
        help=(
            "Master clips CSV (one row per clip). Used to drop the 1,278 "
            "unknown clips. Pass --include-unknown to keep them."
        ),
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Don't drop unknown clips (default is to drop them).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.shuttle_dir.is_dir():
        print(f"ERROR: shuttle dir does not exist: {args.shuttle_dir}")
        return 1

    files = sorted(args.shuttle_dir.glob("*.npy"))
    if not files:
        print(f"ERROR: no .npy files found in {args.shuttle_dir}")
        return 1

    n_dropped_unknown = 0
    if not args.include_unknown:
        if not args.clips_csv.is_file():
            print(f"ERROR: clips_csv not found: {args.clips_csv}")
            print("       (pass --include-unknown to skip the unknown-drop)")
            return 1
        keep_stems = _load_keep_stems(args.clips_csv)
        before = len(files)
        files = [p for p in files if p.stem in keep_stems]
        n_dropped_unknown = before - len(files)
        print(
            f"Dropped {n_dropped_unknown} unknown clips "
            f"(of {before} on disk; kept {len(files)})."
        )

    print(f"Scanning {len(files)} shuttle NPY files in {args.shuttle_dir} ...")

    all_lens: list[np.ndarray] = []
    n_clips_with_gaps = 0
    n_clips_all_zero = 0
    n_clips_no_gap = 0

    for i, p in enumerate(files):
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        if (arr[:, 2] == 0).all():
            # All-zero clips have no valid endpoints, so a "gap" here
            # has no boundary semantics: it's "shuttle never detected
            # in this clip", not a flight-phase gap. Count separately
            # but don't pollute the length distribution.
            n_clips_all_zero += 1
            continue
        lens = _gap_lengths(arr[:, 2])
        if lens.size == 0:
            n_clips_no_gap += 1
            continue
        n_clips_with_gaps += 1
        all_lens.append(lens)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} clips processed")

    if not all_lens:
        print("No gaps found anywhere; nothing to report.")
        return 0

    lens = np.concatenate(all_lens)

    print()
    print("=" * 68)
    print(" Shuttle-gap length distribution")
    print("=" * 68)
    print(f"  Clips scanned:                {len(files)}")
    print(f"  Clips with no detections at all: {n_clips_all_zero}")
    print(f"  Clips with at least one gap:  {n_clips_with_gaps}")
    print(f"  Clips with no gaps at all:    {n_clips_no_gap}")
    print(f"  Total gaps:                   {lens.size}")
    print(f"  Total missing frames:         {int(lens.sum())}")
    print()
    print(f"  mean gap length:    {lens.mean():.2f}")
    print(f"  median gap length:  {int(np.median(lens))}")
    pcts = _percentiles(lens)
    print("  percentiles:        "
          + "  ".join(f"{k}={v:.1f}" for k, v in pcts.items()))
    print()
    print("--- Gap-length classification ---")
    for label, lo, hi in LENGTH_CLASSES:
        mask = (lens >= lo) & (lens <= hi)
        count = int(mask.sum())
        frame_share = (
            100 * lens[mask].sum() / lens.sum() if count else 0.0
        )
        gap_share = 100 * count / lens.size
        print(
            f"  {label:42s}  gaps={count:7d} ({gap_share:5.2f}%)  "
            f"frames={int(lens[mask].sum()):7d} ({frame_share:5.2f}%)"
        )
    print()

    if args.no_output:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    host = socket.gethostname().split(".")[0]

    # Histogram PNG: linear up to 30, then log tail. Two panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    short = lens[lens <= 30]
    axes[0].hist(short, bins=np.arange(1, 32) - 0.5, color="#4477aa")
    axes[0].set_xlim(0.5, 30.5)
    axes[0].set_xlabel("gap length (frames)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Short gaps (1-30 frames, n={short.size})")

    long = lens[lens > 30]
    if long.size:
        axes[1].hist(long, bins=50, color="#cc6677")
    axes[1].set_xlabel("gap length (frames)")
    axes[1].set_ylabel("count (log)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Long-tail gaps (>30 frames, n={long.size})")

    fig.suptitle("Shuttle-gap length distribution")
    fig.tight_layout()
    png_path = (
        args.out_dir / f"shuttle_gap_length_distribution_{host}_{timestamp}.png"
    )
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {png_path}")

    md = [
        f"# Shuttle-gap length distribution",
        "",
        f"- shuttle-dir: `{args.shuttle_dir}`",
        f"- host: `{host}`",
        f"- timestamp: {timestamp}",
        f"- dropped unknown clips: {n_dropped_unknown}",
        "",
        f"- clips scanned: {len(files)}",
        f"- clips with no detections at all: {n_clips_all_zero}",
        f"- clips with at least one gap: {n_clips_with_gaps}",
        f"- clips with no gaps at all: {n_clips_no_gap}",
        f"- total gaps: {lens.size}",
        f"- total missing frames: {int(lens.sum())}",
        "",
        f"- mean gap length: {lens.mean():.2f}",
        f"- median gap length: {int(np.median(lens))}",
        "- percentiles: " + ", ".join(f"{k}={v:.1f}" for k, v in pcts.items()),
        "",
        "## Length-class breakdown",
        "",
        "| Class | gaps | gap % | frames | frame % |",
        "|---|---|---|---|---|",
    ]
    for label, lo, hi in LENGTH_CLASSES:
        mask = (lens >= lo) & (lens <= hi)
        count = int(mask.sum())
        frame_share = (
            100 * lens[mask].sum() / lens.sum() if count else 0.0
        )
        gap_share = 100 * count / lens.size
        md.append(
            f"| {label} | {count} | {gap_share:.2f}% | "
            f"{int(lens[mask].sum())} | {frame_share:.2f}% |"
        )

    md_path = (
        args.out_dir / f"shuttle_gap_length_distribution_{host}_{timestamp}.md"
    )
    md_path.write_text("\n".join(md) + "\n")
    print(f"Saved report:    {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
