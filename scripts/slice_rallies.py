"""Slice source videos into frame-accurate per-rally mp4 clips.

Mirrors BST's clip extraction approach — ffmpeg + libx264 re-encode with
the half-frame seek offset trick — but operates per-rally rather than
per-shot. Per-rally is the right granularity for the downstream TrackNet
pass: each clip is short enough to keep TrackNet memory bounded, but
large enough that subprocess overhead per rally is amortised across
hundreds of frames of inference.

Why re-encode rather than ``-c copy``: stream copy is keyframe-aligned
(off by up to GOP length, ~50-125 frames at typical 2-5s GOPs). For
frame-accurate slicing we must re-encode. libx264 with ``-crf 18`` is
visually lossless for our purposes.

Why ``-ss`` before ``-i``: fast keyframe seek + decode-and-re-encode
forward to the exact target frame. Modern ffmpeg (≥2.1) handles this
frame-accurately for h264 mp4 — same path moviepy uses internally.
The ``+0.5/fps`` offset on the seek time prevents sub-frame rounding
landing on the previous frame.

For each ``(vid, set, rally)`` group in shots_master.csv:
  - Compute extent ``[min(shuttle_start_f), max(shuttle_end_f))`` across strokes
  - Slice the source video with the above ffmpeg incantation
  - Verify output frame count via ffprobe (allows ±1 tolerance for
    boundary frames; >1 mismatch is a hard fail)
  - Save to ``runtime/data/shuttleset/rally_clips/<vid>_<set>_<rally>.mp4``

Idempotency: skip if output exists. ``--force`` to redo.

Usage:
    uv run python -m scripts.slice_rallies                # all vids
    uv run python -m scripts.slice_rallies --vid 1
    uv run python -m scripts.slice_rallies --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
from functools import partial
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from shared.dataset import VIDEO_METADATA_PATH  # noqa: E402

SHOTS_MASTER_PATH = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
RAW_VIDEO_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'raw_video'
RALLY_CLIPS_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'rally_clips'


def find_source_video(vid: int) -> Path | None:
    matches = sorted(RAW_VIDEO_DIR.glob(f'{vid} *.mp4'))
    return matches[0] if matches else None


def compute_rally_bounds(
    strokes: pd.DataFrame,
) -> list[tuple[str, int, int, int]]:
    """Return per-rally ``(set_id, rally, start_f, end_f)`` from strokes."""
    bounds = []
    for (set_id, rally), grp in strokes.groupby(['set_id', 'rally'], sort=True):
        start_f = max(0, int(grp['shuttle_start_f'].min()))
        end_f = int(grp['shuttle_end_f'].max())
        bounds.append((set_id, int(rally), start_f, end_f))
    return bounds


def probe_frame_count(video_path: Path) -> int:
    """Exact frame count via ffprobe -count_frames (decoded count, slow but accurate)."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'csv=p=0',
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def slice_rally(
    source_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    """Frame-accurate ffmpeg slice with libx264 re-encode."""
    # Half-frame offset → seek lands on the intended frame even with
    # sub-frame timestamp rounding. Mirrors BST's _frame_to_time().
    start_sec = (start_frame + 0.5) / fps
    duration_sec = (end_frame - start_frame) / fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_sec:.6f}',
        '-i', str(source_path),
        '-t', f'{duration_sec:.6f}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',                # near-lossless
        '-an',                        # drop audio
        '-loglevel', 'error',
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def process_one_vid(
    vid: int,
    master: pd.DataFrame,
    fps_by_vid: dict[int, int],
    force: bool,
) -> None:
    """Slice every rally for one source video; verify frame counts."""
    video_path = find_source_video(vid)
    if video_path is None:
        print(f'vid={vid}: WARNING source video not found in {RAW_VIDEO_DIR}, skipping',
              flush=True)
        return
    fps = fps_by_vid.get(vid)
    if fps is None:
        print(f'vid={vid}: WARNING no fps in video_metadata.csv, skipping', flush=True)
        return
    strokes = master[master['vid'] == vid]
    if strokes.empty:
        print(f'vid={vid}: no strokes in shots_master, skipping', flush=True)
        return

    bounds = compute_rally_bounds(strokes)
    print(f'vid={vid}: {len(bounds)} rallies to slice from {video_path.name}', flush=True)

    n_done = n_skipped = n_mismatch = n_failed = 0
    for set_id, rally, start_f, end_f in bounds:
        out_path = RALLY_CLIPS_DIR / f'{vid}_{set_id}_{rally}.mp4'
        if out_path.exists() and not force:
            n_skipped += 1
            continue
        try:
            slice_rally(video_path, out_path, start_f, end_f, fps)
        except subprocess.CalledProcessError as e:
            print(f'vid={vid} {out_path.name}: ffmpeg failed: {e}', flush=True)
            n_failed += 1
            if out_path.exists():
                out_path.unlink()
            continue

        expected = end_f - start_f
        try:
            actual = probe_frame_count(out_path)
        except subprocess.CalledProcessError as e:
            print(f'vid={vid} {out_path.name}: ffprobe failed: {e}', flush=True)
            n_failed += 1
            continue

        if abs(actual - expected) > 1:
            # >1 mismatch is a real frame-accuracy break — drop the bad clip
            # so it's regenerated next run with the diagnostic visible.
            print(f'vid={vid} {out_path.name}: FRAME MISMATCH '
                  f'expected={expected} actual={actual} (>1 off — dropping clip)',
                  flush=True)
            out_path.unlink()
            n_mismatch += 1
        else:
            n_done += 1

    print(
        f'vid={vid}: done — {n_done} sliced, {n_skipped} skipped, '
        f'{n_mismatch} mismatch, {n_failed} failed',
        flush=True,
    )


def _worker(
    vid: int, master: pd.DataFrame, fps_by_vid: dict[int, int], force: bool,
) -> None:
    """Pool worker — wraps process_one_vid with crash isolation."""
    try:
        process_one_vid(vid, master, fps_by_vid, force=force)
    except Exception as e:
        print(f'vid={vid}: ERROR {type(e).__name__}: {e}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, nargs='*',
                        help='Specific vid(s). Default: all vids in shots_master.')
    parser.add_argument('--force', action='store_true',
                        help='Re-slice even if output exists.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent vids (each worker handles one vid serially '
                             'across all its rallies). Default 1.')
    args = parser.parse_args()

    master = pd.read_csv(SHOTS_MASTER_PATH)
    fps_by_vid = (
        pd.read_csv(VIDEO_METADATA_PATH)
        .set_index('id')['fps']
        .dropna().astype(int).to_dict()
    )

    if args.vid:
        vids = sorted(set(args.vid))
    else:
        vids = sorted(master['vid'].unique().tolist())

    n_workers = max(1, min(args.workers, len(vids)))
    print(f'Processing {len(vids)} vid(s) with {n_workers} worker(s): {vids}')

    if n_workers == 1:
        for vid in vids:
            _worker(int(vid), master, fps_by_vid, args.force)
        return

    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(
            partial(_worker, master=master, fps_by_vid=fps_by_vid, force=args.force),
            [int(v) for v in vids],
        )


if __name__ == '__main__':
    main()
