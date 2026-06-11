#!/usr/bin/env python3
"""Retry extraction for specific YouTube IDs with a different timestamp strategy.

Usage: python scripts/extract_frames_retry.py <id1> <id2> ...

Tries multiple (set, position) candidates per match and stops at the first
that produces a non-zero JPG. Outputs to the same data/frames/ directory.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SHUTTLESET = REPO_ROOT / "src/bst_x/ShuttleSet"
META_CSV = SHUTTLESET / "video_metadata.csv"
SET_DIR = SHUTTLESET / "set"
OUT_DIR = REPO_ROOT / "frontend/hba-stroke-classifier/data/frames"

# (set_name, fraction-through-rows). Each candidate is tried in order.
CANDIDATES = [
    ("set1.csv", 0.75),
    ("set3.csv", 0.50),
    ("set1.csv", 0.25),
    ("set2.csv", 0.80),
    ("set2.csv", 0.20),
    ("set3.csv", 0.75),
    ("set1.csv", 0.50),
]


def youtube_id(url: str) -> str | None:
    q = urlparse(url)
    if q.hostname == "youtu.be":
        return q.path.lstrip("/")
    if q.hostname and "youtube.com" in q.hostname:
        return parse_qs(q.query).get("v", [None])[0]
    return None


def timestamps_for(folder: Path) -> list[str]:
    out: list[str] = []
    for set_name, frac in CANDIDATES:
        p = folder / set_name
        if not p.exists():
            continue
        with p.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("time")]
        if not rows:
            continue
        idx = min(int(len(rows) * frac), len(rows) - 1)
        out.append(rows[idx]["time"])
    # Dedupe preserving order.
    seen = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def stream_url(url: str) -> str | None:
    try:
        return subprocess.check_output(
            ["yt-dlp", "-g", "-f", "best[height<=720]/best", url],
            text=True, stderr=subprocess.DEVNULL, timeout=60,
        ).strip().splitlines()[0]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return None


def grab(stream: str, ts: str, out: Path) -> bool:
    try:
        subprocess.check_call(
            ["ffmpeg", "-ss", ts, "-i", stream, "-frames:v", "1", "-q:v", "3", "-y", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        return out.exists() and out.stat().st_size > 1024
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main(ids: list[str]) -> int:
    with META_CSV.open() as f:
        meta = {youtube_id(r["url"]): r for r in csv.DictReader(f) if youtube_id(r["url"])}

    for vid in ids:
        row = meta.get(vid)
        if not row:
            print(f"FAIL {vid}: not in metadata"); continue
        folder = SET_DIR / row["video"]
        if not folder.is_dir():
            print(f"FAIL {vid}: no folder"); continue
        candidates = timestamps_for(folder)
        if not candidates:
            print(f"FAIL {vid}: no timestamps"); continue
        stream = stream_url(row["url"])
        if not stream:
            print(f"FAIL {vid}: yt-dlp failed"); continue

        out = OUT_DIR / f"{vid}.jpg"
        for ts in candidates:
            if grab(stream, ts, out):
                print(f"OK   {vid}: {ts} → {out.stat().st_size // 1024} KB")
                break
        else:
            print(f"FAIL {vid}: all {len(candidates)} timestamps failed")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    skip = 0
    if args and args[0].startswith("--skip="):
        skip = int(args[0].split("=", 1)[1])
        args = args[1:]
    if skip:
        CANDIDATES[:] = CANDIDATES[skip:]
    if not args:
        print("usage: extract_frames_retry.py [--skip=N] <youtube_id> ...", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(args))
