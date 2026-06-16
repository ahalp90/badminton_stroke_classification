#!/usr/bin/env python3
"""Extract one representative court-view frame per ShuttleSet match.

Picks a mid-rally stroke timestamp from the match's annotation CSV (guaranteed
to be live play, so we get a clean broadcast court view), then uses yt-dlp to
resolve the YouTube stream URL and ffmpeg to grab a single JPG.

Output: frontend/hba-stroke-classifier/data/frames/<video_id>.jpg
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SHUTTLESET = REPO_ROOT / "data/shuttleset"
META_CSV = SHUTTLESET / "video_metadata.csv"
SET_DIR = SHUTTLESET / "set"
OUT_DIR = REPO_ROOT / "frontend/hba-stroke-classifier/data/frames"


def youtube_id(url: str) -> str | None:
    q = urlparse(url)
    if q.hostname == "youtu.be":
        return q.path.lstrip("/")
    if q.hostname and "youtube.com" in q.hostname:
        return parse_qs(q.query).get("v", [None])[0]
    return None


def pick_timestamp(folder: Path) -> str | None:
    """Pick a mid-rally timestamp. Prefer set2; fall back to set1, set3."""
    for name in ("set2.csv", "set1.csv", "set3.csv"):
        p = folder / name
        if not p.exists():
            continue
        with p.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("time")]
        if not rows:
            continue
        return rows[len(rows) // 2]["time"]
    return None


def extract_one(video_name: str, url: str, out_path: Path) -> tuple[bool, str]:
    folder = SET_DIR / video_name
    if not folder.is_dir():
        return False, f"no folder: {video_name}"
    ts = pick_timestamp(folder)
    if not ts:
        return False, "no timestamp in any set CSV"

    try:
        stream = subprocess.check_output(
            ["yt-dlp", "-g", "-f", "best[height<=720]/best", url],
            text=True, stderr=subprocess.DEVNULL, timeout=60,
        ).strip().splitlines()[0]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, f"yt-dlp failed: {e}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            ["ffmpeg", "-ss", ts, "-i", stream, "-frames:v", "1", "-q:v", "3", "-y", str(out_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, f"ffmpeg failed at {ts}: {e}"

    return True, f"{ts} → {out_path.name} ({out_path.stat().st_size // 1024} KB)"


def main() -> int:
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        print("ERROR: yt-dlp and ffmpeg required", file=sys.stderr)
        return 1

    with META_CSV.open() as f:
        rows = list(csv.DictReader(f))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []

    for i, row in enumerate(rows, 1):
        vid_id = youtube_id(row["url"])
        if not vid_id:
            failures.append((row["video"], "could not parse YouTube ID"))
            print(f"[{i}/{len(rows)}] SKIP {row['video']}: bad URL")
            continue
        out_path = OUT_DIR / f"{vid_id}.jpg"
        if out_path.exists():
            print(f"[{i}/{len(rows)}] CACHED {vid_id}.jpg")
            continue
        ok, msg = extract_one(row["video"], row["url"], out_path)
        prefix = "OK  " if ok else "FAIL"
        print(f"[{i}/{len(rows)}] {prefix} {vid_id}: {msg}")
        if not ok:
            failures.append((row["video"], msg))

    print(f"\nDone. {len(rows) - len(failures)}/{len(rows)} succeeded.")
    if failures:
        print("Failures:")
        for v, m in failures:
            print(f"  {v}: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
