"""Shuttle-gap boundary y-coordinate distribution.

Tests the off-screen-high hypothesis for the high shuttle-missing rate
(6.34% post-inpaint per the Phase-2 validation analysis). For every
contiguous run of ``visibility=0`` frames in a per-clip shuttle NPY,
records the y-coordinate of the last valid detection before the gap
and the first valid detection after, then aggregates across the full
32,203-clip set.

If gap-boundary y-coords cluster near the top of the frame (small
y in image coords post-``normalize_shuttlecock``), the hypothesis
holds: the shuttle is exiting the camera frame on high arcs and
TrackNetV3-with-inpaint cannot recover frames where the bird is
physically not in any pixel.

If the boundary distribution is roughly uniform, the off-screen-high
mechanism is not the dominant cause and the hypothesis needs revising
toward motion blur / sustained occlusion / inpaint window limits.

Usage::

    python -m validation_scripts.shuttle_gap_y_distribution \\
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


def _find_gaps(visibility: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) gap indices (end exclusive) where vis==0.

    Edge gaps (touching frame 0 or T-1) are included; the boundary-y
    extraction below handles them by returning NaN for the unavailable
    side.
    """
    if visibility.size == 0:
        return []
    is_missing = visibility == 0
    # Edge-padded diff trick: 0->1 = gap start, 1->0 = gap end.
    edges = np.diff(np.concatenate([[0], is_missing.astype(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _boundary_ys(
    shuttle_arr: np.ndarray, gap: tuple[int, int],
) -> tuple[float, float]:
    """Return (last_valid_y_before, first_valid_y_after) for a gap.

    Returns NaN for either side if the gap touches the clip boundary
    (no valid endpoint exists). Coords are in normalised [0, 1] image
    space; y=0 is top of frame, y=1 is bottom.
    """
    start, end = gap
    last_y = float(shuttle_arr[start - 1, 1]) if start > 0 else float("nan")
    n_frames = shuttle_arr.shape[0]
    first_y = (
        float(shuttle_arr[end, 1]) if end < n_frames else float("nan")
    )
    return last_y, first_y


def _percentiles(values: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    if values.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--shuttle-dir",
        type=Path,
        default=Path("/scratch/comp320a/ShuttleSet/shuttle_npy_flat"),
        help=(
            "Flat per-clip shuttle NPY dir (each file shape (t, 3) with "
            "[x_norm, y_norm, visibility])."
        ),
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
        "--min-gap-len",
        type=int,
        default=1,
        help="Skip gaps shorter than this (default 1, includes all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to save the markdown report and histogram PNG.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip writing the markdown / PNG; print stdout summary only.",
    )
    args = parser.parse_args()

    if not args.shuttle_dir.is_dir():
        print(f"ERROR: shuttle dir does not exist: {args.shuttle_dir}")
        return 1

    files = sorted(args.shuttle_dir.glob("*.npy"))
    if not files:
        print(f"ERROR: no .npy files found in {args.shuttle_dir}")
        return 1

    keep_stems: set[str] | None = None
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

    last_ys: list[float] = []
    first_ys: list[float] = []
    n_clips_with_gaps = 0
    n_total_gaps = 0
    n_clips_all_zero = 0
    n_edge_gaps = 0  # gaps touching clip start or end (one-sided boundary)

    for i, p in enumerate(files):
        try:
            arr = np.load(p)  # (t, 3)
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            print(f"  WARNING: unexpected shape {arr.shape} in {p.name}")
            continue
        if (arr[:, 2] == 0).all():
            # All-zero clips have no valid endpoints, so they cannot
            # contribute boundary y-coords to the histogram. Count
            # separately and skip.
            n_clips_all_zero += 1
            continue

        gaps = _find_gaps(arr[:, 2])
        gaps = [g for g in gaps if g[1] - g[0] >= args.min_gap_len]
        if not gaps:
            continue
        n_clips_with_gaps += 1

        for g in gaps:
            n_total_gaps += 1
            last_y, first_y = _boundary_ys(arr, g)
            if np.isnan(last_y) or np.isnan(first_y):
                n_edge_gaps += 1
            if not np.isnan(last_y):
                last_ys.append(last_y)
            if not np.isnan(first_y):
                first_ys.append(first_y)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} clips processed")

    last_arr = np.asarray(last_ys, dtype=np.float64)
    first_arr = np.asarray(first_ys, dtype=np.float64)

    # Combine pre- and post-gap boundaries (the model sees both as
    # "frames adjacent to a gap"); also keep them separated below.
    all_arr = np.concatenate([last_arr, first_arr])

    print()
    print("=" * 68)
    print(" Shuttle-gap boundary y-coordinate distribution")
    print("=" * 68)
    print(f"  Clips scanned:                {len(files)}")
    print(f"  Clips with no detections at all: {n_clips_all_zero}")
    print(f"  Clips with at least one gap:  {n_clips_with_gaps}")
    print(f"  Total gaps (length >= {args.min_gap_len}): {n_total_gaps}")
    print(f"  Edge gaps (touch clip boundary): {n_edge_gaps}")
    print(f"  Pre-gap valid boundaries:     {len(last_arr)}")
    print(f"  Post-gap valid boundaries:    {len(first_arr)}")
    print()
    print("y-coordinate convention:")
    print("  y=0 is TOP of image, y=1 is BOTTOM (normalised image coords).")
    print("  Values < 0.2 cluster near the top, > 0.8 cluster near the bottom.")
    print()

    def _print_block(label: str, values: np.ndarray) -> None:
        print(f"--- {label} (n={values.size}) ---")
        if values.size == 0:
            print("  (no data)")
            return
        pcts = _percentiles(values)
        print(f"  mean={values.mean():.3f}  median={np.median(values):.3f}")
        print(
            "  percentiles:  "
            + "  ".join(f"{k}={v:.3f}" for k, v in pcts.items())
        )
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(values, bins=bins)
        print("  histogram (10 buckets, top to bottom of frame):")
        for lo, hi, count in zip(bins[:-1], bins[1:], hist):
            pct = 100 * count / values.size
            bar = "#" * int(round(pct / 2))
            print(f"    y in [{lo:.1f}, {hi:.1f}): {count:7d}  {pct:5.2f}%  {bar}")
        print()

    _print_block("Pre-gap last-valid y", last_arr)
    _print_block("Post-gap first-valid y", first_arr)
    _print_block("Combined (pre and post)", all_arr)

    if args.no_output:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    host = socket.gethostname().split(".")[0]

    # Histogram PNG: pre / post / combined as side-by-side panels.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, vals, title in (
        (axes[0], last_arr, "Pre-gap last-valid y"),
        (axes[1], first_arr, "Post-gap first-valid y"),
        (axes[2], all_arr, "Combined"),
    ):
        if vals.size == 0:
            ax.set_title(f"{title} (no data)")
            continue
        ax.hist(vals, bins=20, range=(0, 1), color="#4477aa")
        ax.set_title(f"{title} (n={vals.size})")
        ax.set_xlabel("y (0=top, 1=bottom)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("frame count")
    fig.suptitle("Shuttle-gap boundary y distribution")
    fig.tight_layout()

    png_path = args.out_dir / f"shuttle_gap_y_distribution_{host}_{timestamp}.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {png_path}")

    # Markdown summary.
    md = []
    md.append(f"# Shuttle-gap boundary y-coordinate distribution")
    md.append("")
    md.append(f"- shuttle-dir: `{args.shuttle_dir}`")
    md.append(f"- host: `{host}`")
    md.append(f"- timestamp: {timestamp}")
    md.append(f"- min-gap-len: {args.min_gap_len}")
    md.append(f"- dropped unknown clips: {n_dropped_unknown}")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- clips scanned: {len(files)}")
    md.append(f"- clips with no detections at all: {n_clips_all_zero}")
    md.append(f"- clips with at least one gap: {n_clips_with_gaps}")
    md.append(f"- total gaps (length >= {args.min_gap_len}): {n_total_gaps}")
    md.append(f"- edge gaps (touch clip boundary): {n_edge_gaps}")
    md.append("")

    for label, values in (
        ("Pre-gap last-valid y", last_arr),
        ("Post-gap first-valid y", first_arr),
        ("Combined", all_arr),
    ):
        md.append(f"## {label} (n={values.size})")
        md.append("")
        if values.size == 0:
            md.append("(no data)")
            md.append("")
            continue
        pcts = _percentiles(values)
        md.append(
            f"- mean={values.mean():.3f}  median={np.median(values):.3f}"
        )
        md.append(
            "- percentiles: "
            + ", ".join(f"{k}={v:.3f}" for k, v in pcts.items())
        )
        md.append("")
        md.append("| y range | count | % |")
        md.append("|---|---|---|")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(values, bins=bins)
        for lo, hi, count in zip(bins[:-1], bins[1:], hist):
            pct = 100 * count / values.size
            md.append(f"| [{lo:.1f}, {hi:.1f}) | {count} | {pct:.2f}% |")
        md.append("")

    md_path = (
        args.out_dir / f"shuttle_gap_y_distribution_{host}_{timestamp}.md"
    )
    md_path.write_text("\n".join(md) + "\n")
    print(f"Saved report:    {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
