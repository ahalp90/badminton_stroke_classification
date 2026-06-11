"""Per-class whole-clip shuttle miss rate distribution.

Different framing from the existing analysis txt's per-class table,
which is a single aggregate within +-10 frames of contact. Here we
compute per-clip stats first, then summarise per class:

    miss_rate_per_clip = (visibility == 0).sum() / len(visibility)

Per class: median, 1 SD, 2 SD across that class's clips.

We also report what proportion of each clip's missing frames fall in
the central window ``[edge, len - edge)`` (default ``edge = 15``):

    central_share_per_clip = (visibility[edge:-edge] == 0).sum()
                             / (visibility == 0).sum()

Per class: median, 1 SD, 2 SD across that class's clips. Excludes
clips with zero missing frames (ratio undefined) and clips shorter
than ``2 * edge`` frames (no central window).

Usage::

    python -m validation_scripts.perclass_clip_miss_rate \\
        --shuttle-dir /scratch/comp320a/ShuttleSet/shuttle_npy_flat
"""
from __future__ import annotations

import argparse
import csv
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
)
DEFAULT_CLIPS_CSV = Path(
    "/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv"
)

# Frames trimmed from each end to define the "central" window.
DEFAULT_EDGE_TRIM = 15


def _load_clip_classes(clips_csv: Path) -> dict[str, str]:
    """Return ``{clip_stem: raw_type_en}`` excluding unknown clips."""
    out: dict[str, str] = {}
    with clips_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            cls = row.get("raw_type_en", "").strip().lower()
            if cls and cls != "unknown":
                out[row["clip_stem"]] = cls
    return out


def _per_clip_stats(
    visibility: np.ndarray, edge: int
) -> tuple[float, float]:
    """Return ``(miss_rate, central_share)`` for one clip.

    ``central_share`` is NaN when the clip is shorter than ``2 * edge``
    (no central window) or when the clip has zero missing frames
    (ratio undefined).
    """
    n = visibility.size
    if n == 0:
        return 0.0, float("nan")

    missing = visibility == 0
    n_missing = int(missing.sum())
    miss_rate = n_missing / n

    if n <= 2 * edge or n_missing == 0:
        return miss_rate, float("nan")

    n_missing_central = int(missing[edge:-edge].sum())
    return miss_rate, n_missing_central / n_missing


def _summarise(values: np.ndarray) -> dict:
    """Median / mean / 1-SD / 2-SD / min / max for a 1-D array."""
    if values.size == 0:
        return {k: float("nan") for k in
                ("median", "mean", "sd1", "sd2", "min", "max")}
    sd1 = float(values.std())
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "sd1": sd1,
        "sd2": 2.0 * sd1,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _print_summary_table(
    title: str, by_class: list[str], stats: dict[str, dict],
    n_clips: dict[str, int], extra_cols: dict | None = None,
) -> None:
    print()
    print("=" * 100)
    print(f" {title}")
    print("=" * 100)
    header = (
        f"{'Class':<24} {'n':>6}  "
        f"{'median':>8} {'mean':>8} {'sd1':>8} {'sd2':>8} "
        f"{'min':>8} {'max':>8}"
    )
    if extra_cols:
        for col in extra_cols:
            header += f"  {col:>12}"
    print(header)
    for cls in by_class:
        s = stats[cls]
        row = (
            f"{cls:<24} {n_clips[cls]:>6}  "
            f"{s['median']:>8.4f} {s['mean']:>8.4f} "
            f"{s['sd1']:>8.4f} {s['sd2']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f}"
        )
        if extra_cols:
            for col, vals in extra_cols.items():
                row += f"  {vals[cls]:>12d}"
        print(row)


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
        help="Master clips CSV (one row per clip). Used for class labels and to drop unknowns.",
    )
    parser.add_argument(
        "--edge-trim",
        type=int,
        default=DEFAULT_EDGE_TRIM,
        help="Frames trimmed from each end for the central window (default 15).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print to stdout only; skip writing the .md report.",
    )
    args = parser.parse_args()

    if not args.shuttle_dir.is_dir():
        print(f"ERROR: shuttle dir does not exist: {args.shuttle_dir}")
        return 1
    if not args.clips_csv.is_file():
        print(f"ERROR: clips_csv not found: {args.clips_csv}")
        return 1

    clip_class = _load_clip_classes(args.clips_csv)
    print(f"Loaded {len(clip_class)} non-unknown clips from {args.clips_csv}")

    files = sorted(args.shuttle_dir.glob("*.npy"))
    if not files:
        print(f"ERROR: no .npy files found in {args.shuttle_dir}")
        return 1

    miss_rates: dict[str, list[float]] = defaultdict(list)
    central_shares: dict[str, list[float]] = defaultdict(list)
    excl_short_clip: dict[str, int] = defaultdict(int)
    excl_no_missing: dict[str, int] = defaultdict(int)
    n_processed = 0

    for i, p in enumerate(files):
        cls = clip_class.get(p.stem)
        if cls is None:
            continue
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue

        visibility = arr[:, 2]
        miss_rate, central_share = _per_clip_stats(visibility, args.edge_trim)
        miss_rates[cls].append(miss_rate)
        if np.isnan(central_share):
            if visibility.size <= 2 * args.edge_trim:
                excl_short_clip[cls] += 1
            else:
                excl_no_missing[cls] += 1
        else:
            central_shares[cls].append(central_share)
        n_processed += 1

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} clips processed")

    print(
        f"Processed {n_processed} clips across {len(miss_rates)} classes."
    )

    miss_stats = {
        cls: _summarise(np.asarray(vals, dtype=np.float64))
        for cls, vals in miss_rates.items()
    }
    central_stats = {
        cls: _summarise(np.asarray(vals, dtype=np.float64))
        for cls, vals in central_shares.items()
    }
    miss_n = {cls: len(vals) for cls, vals in miss_rates.items()}
    central_n = {cls: len(vals) for cls, vals in central_shares.items()}

    by_class = sorted(
        miss_rates.keys(),
        key=lambda c: -miss_stats[c]["median"],
    )

    _print_summary_table(
        "Per-class whole-clip shuttle miss rate "
        "(per clip = missing / total frames)",
        by_class,
        miss_stats,
        miss_n,
    )
    _print_summary_table(
        f"Per-class share of missing frames in central window "
        f"[{args.edge_trim}, len - {args.edge_trim})",
        by_class,
        central_stats,
        central_n,
        extra_cols={
            "excl_short": excl_short_clip,
            "excl_nomiss": excl_no_missing,
        },
    )

    if args.no_output:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    host = socket.gethostname().split(".")[0]

    md = [
        "# Per-class whole-clip shuttle miss rate",
        "",
        f"- shuttle-dir: `{args.shuttle_dir}`",
        f"- clips-csv: `{args.clips_csv}`",
        f"- host: `{host}`",
        f"- timestamp: {timestamp}",
        f"- edge trim for central window: {args.edge_trim}",
        f"- clips processed (non-unknown): {n_processed}",
        "",
        "## Whole-clip shuttle miss rate",
        "",
        "Per-clip metric: `(visibility == 0).sum() / len(visibility)`. "
        "Per class: median, 1 SD, 2 SD across that class's clips. Sorted "
        "by median descending.",
        "",
        "| Class | n_clips | median | mean | sd1 | sd2 | min | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cls in by_class:
        s = miss_stats[cls]
        md.append(
            f"| {cls} | {miss_n[cls]} | {s['median']:.4f} | "
            f"{s['mean']:.4f} | {s['sd1']:.4f} | {s['sd2']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )

    md += [
        "",
        f"## Share of missing frames in central window "
        f"[{args.edge_trim}, len - {args.edge_trim})",
        "",
        f"Per-clip metric: `(visibility[{args.edge_trim}:-{args.edge_trim}] "
        f"== 0).sum() / (visibility == 0).sum()`. Per class: median, 1 SD, "
        "2 SD across that class's clips. Excludes clips shorter than "
        f"{2 * args.edge_trim} frames (no central window) and clips with "
        "zero missing frames (ratio undefined). Same class ordering as the "
        "miss-rate table.",
        "",
        "| Class | n_clips | median | mean | sd1 | sd2 | min | max | "
        "excl_short | excl_nomiss |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls in by_class:
        s = central_stats[cls]
        md.append(
            f"| {cls} | {central_n[cls]} | {s['median']:.4f} | "
            f"{s['mean']:.4f} | {s['sd1']:.4f} | {s['sd2']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | "
            f"{excl_short_clip[cls]} | {excl_no_missing[cls]} |"
        )

    md_path = (
        args.out_dir / f"perclass_clip_miss_rate_{host}_{timestamp}.md"
    )
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nSaved report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
