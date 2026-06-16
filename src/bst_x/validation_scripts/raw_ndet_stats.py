"""Summarise the per-frame detection counts in a raw mmpose extract.

Walks every ``*_raw_ndet.npy`` under ``--raw-dir`` and reports the
distribution of mmpose detection counts per frame. Two questions
this answers:

1. **How often does the detector miss everyone?** ``ndet==0`` frames
   are the irreducible failure floor for any heuristic that requires
   at least one detection per frame.
2. **How often does the detector return only one person?** ``ndet==1``
   frames will be zeroed by any heuristic that needs two players, so
   that bucket is the realistic per-frame failure floor for the
   downstream pipeline.

The full histogram (frames at each ``ndet`` value, 0..N_max) is also
useful as a baseline against which future re-extracts can be compared.
A change in the bulk of the distribution (e.g. shift in the modal
``ndet``) would indicate a meaningful change in mmpose's behaviour or
the upstream clip set.

Optionally restricts the scan to a subset of stems via
``--stems-file`` (one stem per line). Useful for comparing the 1,716
Phase-1 backfill subset against the freshly extracted bulk.

By default writes a markdown report to a sibling
``raw_ndet_stats_outputs/`` directory next to this script, alongside
the stdout summary. Suppress with ``--no-output``.

Usage:

    python -m validation_scripts.raw_ndet_stats \\
        --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw

    # restricted to the Phase-1 backfill stems
    python -m validation_scripts.raw_ndet_stats \\
        --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw \\
        --stems-file scratch/architecture_notes/busted_hit_zone_clips_phase1.txt
"""
from __future__ import annotations

import argparse
import socket
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "raw_ndet_stats_outputs"


def _load_stems_file(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _format_report(
    *,
    raw_dir: Path,
    stems_file: Path | None,
    n_clips: int,
    total_frames: int,
    zero_frames: int,
    clips_with_any_zero: int,
    per_clip_zero_pct: list[float],
    ndet_hist: Counter[int],
    ran_at: datetime,
    host: str,
) -> str:
    """Render the human-readable report shared by stdout and the markdown file."""
    lines: list[str] = []
    lines.append(f"raw-dir:             {raw_dir}")
    if stems_file is not None:
        lines.append(f"stems-file:          {stems_file}")
    lines.append(f"host:                {host}")
    lines.append(f"ran at:              {ran_at.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"clips:               {n_clips}")
    lines.append(f"frames total:        {total_frames}")
    if total_frames:
        lines.append(
            f"zero-detect frames:  {zero_frames} "
            f"({100 * zero_frames / total_frames:.2f}%)"
        )
        lines.append(
            f"clips with any 0:    {clips_with_any_zero} "
            f"({100 * clips_with_any_zero / n_clips:.2f}%)"
        )
    if per_clip_zero_pct:
        arr = np.asarray(per_clip_zero_pct)
        lines.append(
            f"per-clip 0% rate:    mean={arr.mean():.3f} "
            f"p50={np.median(arr):.3f} "
            f"p95={np.quantile(arr, 0.95):.3f} "
            f"max={arr.max():.3f}"
        )

    if ndet_hist and total_frames:
        lines.append("")
        lines.append("ndet distribution (frames per detection count):")
        for k in sorted(ndet_hist):
            n = ndet_hist[k]
            lines.append(
                f"  ndet={k:>2d}: {n:>10d}  ({100 * n / total_frames:5.2f}%)"
            )

    return "\n".join(lines)


def _write_markdown(
    *,
    out_path: Path,
    report: str,
    raw_dir: Path,
    stems_file: Path | None,
    ran_at: datetime,
    host: str,
) -> None:
    """Write the report as a fenced-code markdown doc, with a small header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = "Raw mmpose ndet distribution"
    if stems_file is not None:
        title += f" (filtered by {stems_file.name})"
    body = (
        f"# {title}\n\n"
        f"- raw-dir: `{raw_dir}`\n"
        + (f"- stems-file: `{stems_file}`\n" if stems_file else "")
        + f"- host: `{host}`\n"
        f"- ran at: `{ran_at.isoformat(timespec='seconds')}`\n\n"
        "## Stats\n\n"
        f"```\n{report}\n```\n"
    )
    out_path.write_text(body)


def _default_output_path(out_dir: Path, raw_dir: Path, stems_file: Path | None) -> Path:
    """Auto-derive a stable, descriptive filename."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = ""
    if stems_file is not None:
        suffix = f"_{stems_file.stem}"
    return out_dir / f"raw_ndet_stats_{raw_dir.name}{suffix}_{stamp}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing per-clip *_raw_ndet.npy files (flat, one level).",
    )
    parser.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="Optional one-stem-per-line filter to restrict the scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Markdown report output path. Defaults to {DEFAULT_OUT_DIR}/<auto>.md.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print to stdout only; don't write a markdown report.",
    )
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        parser.error(f"raw-dir not found: {args.raw_dir}")

    files = sorted(args.raw_dir.glob("*_raw_ndet.npy"))
    if not files:
        parser.error(f"no *_raw_ndet.npy files found under {args.raw_dir}")

    if args.stems_file is not None:
        if not args.stems_file.exists():
            parser.error(f"stems-file not found: {args.stems_file}")
        keep = _load_stems_file(args.stems_file)
        before = len(files)
        files = [f for f in files if f.name.removesuffix("_raw_ndet.npy") in keep]
        print(f"Filtered to {len(files)} of {before} files via {args.stems_file}")

    total_frames = 0
    zero_frames = 0
    clips_with_any_zero = 0
    per_clip_zero_pct: list[float] = []
    ndet_hist: Counter[int] = Counter()

    for f in files:
        nd = np.load(f)
        z = int((nd == 0).sum())
        total_frames += int(nd.size)
        zero_frames += z
        if z:
            clips_with_any_zero += 1
            per_clip_zero_pct.append(z / int(nd.size))
        for v, c in zip(*np.unique(nd, return_counts=True)):
            ndet_hist[int(v)] += int(c)

    ran_at = datetime.now()
    host = socket.gethostname()

    report = _format_report(
        raw_dir=args.raw_dir,
        stems_file=args.stems_file,
        n_clips=len(files),
        total_frames=total_frames,
        zero_frames=zero_frames,
        clips_with_any_zero=clips_with_any_zero,
        per_clip_zero_pct=per_clip_zero_pct,
        ndet_hist=ndet_hist,
        ran_at=ran_at,
        host=host,
    )
    print(report)

    if not args.no_output:
        out_path = args.output or _default_output_path(
            DEFAULT_OUT_DIR, args.raw_dir, args.stems_file
        )
        _write_markdown(
            out_path=out_path,
            report=report,
            raw_dir=args.raw_dir,
            stems_file=args.stems_file,
            ran_at=ran_at,
            host=host,
        )
        print(f"\nReport written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
