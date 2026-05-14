"""Run TrackNetV3 on per-rally clips; accumulate into per-vid shuttle cache.

Operates on the rally clips produced by ``scripts.slice_rallies``. For
each vid:

  1. Find all rally clips for that vid (``<vid>_<set>_<rally>.mp4``)
  2. For each clip: subprocess ``predict.py`` via the wrapper, parse the
     rally-local CSV (Frame, Visibility, X, Y in clip-frame coords)
  3. Re-key clip-frame -> source-frame using the rally's ``shuttle_start_f``
     (re-derived from shots_master.csv to match the slicer's bounds)
  4. Write a per-vid dense cache at ``runtime/cache/shuttle/<vid>.npz``
     with arrays length = source video frame count. Non-rally frames
     have visibility=0, x=y=0 (placeholder; never read at training time
     since training only slices in-rally windows).

Why per-rally instead of full source video:
  - TrackNet's median-image computation gets a clean rally background
    (no replay/talking-head pollution)
  - TrackNet's memory accumulator stays bounded per-clip — full source
    videos OOM at ~25 GB per worker; rally clips peak at < 1 GB
  - Subprocess overhead amortises over hundreds of frames per rally,
    not 33k subprocess calls per-shot

Idempotency: skip vids whose cache exists. ``--force`` to redo.

Usage:
    uv run python -m scripts.extract_shuttle              # all vids
    uv run python -m scripts.extract_shuttle --vid 1
    uv run python -m scripts.extract_shuttle --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from perception.shuttle import extract_shuttle  # noqa: E402
from perception.video_io import get_video_info  # noqa: E402

SHOTS_MASTER_PATH = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
RAW_VIDEO_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'raw_video'
RALLY_CLIPS_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'rally_clips'
SHUTTLE_CACHE_DIR = REPO_ROOT / 'runtime' / 'cache' / 'shuttle'


def find_source_video(vid: int) -> Path | None:
    matches = sorted(RAW_VIDEO_DIR.glob(f'{vid} *.mp4'))
    return matches[0] if matches else None


def compute_rally_bounds(
    strokes: pd.DataFrame,
) -> dict[tuple[str, int], tuple[int, int]]:
    """Re-derive per-rally ``(start_f, end_f)`` from shots_master rows.

    Must match the slicer's logic exactly so frame offsets line up.
    """
    out = {}
    for (set_id, rally), grp in strokes.groupby(['set_id', 'rally'], sort=True):
        start_f = max(0, int(grp['shuttle_start_f'].min()))
        end_f = int(grp['shuttle_end_f'].max())
        out[(set_id, int(rally))] = (start_f, end_f)
    return out


def parse_rally_clip_name(clip_path: Path) -> tuple[int, str, int]:
    """``<vid>_<set>_<rally>.mp4`` → (vid, 'setN', rally)."""
    stem = clip_path.stem
    parts = stem.split('_')
    if len(parts) != 3:
        raise ValueError(f'unexpected rally clip name: {clip_path.name}')
    return int(parts[0]), parts[1], int(parts[2])


def process_one_vid(vid: int, master: pd.DataFrame, force: bool = False) -> None:
    """Run TrackNet on every rally clip for one vid; write per-vid cache."""
    out_path = SHUTTLE_CACHE_DIR / f'{vid}.npz'
    if out_path.exists() and not force:
        print(f'vid={vid}: shuttle cache exists, skipping (use --force to re-run)',
              flush=True)
        return

    source_video = find_source_video(vid)
    if source_video is None:
        print(f'vid={vid}: WARNING source video not found; cannot determine length, skipping',
              flush=True)
        return

    strokes = master[master['vid'] == vid]
    if strokes.empty:
        print(f'vid={vid}: no strokes in shots_master, skipping', flush=True)
        return

    rally_bounds = compute_rally_bounds(strokes)
    rally_clips = sorted(RALLY_CLIPS_DIR.glob(f'{vid}_*.mp4'))
    if not rally_clips:
        print(f'vid={vid}: no rally clips found in {RALLY_CLIPS_DIR}; '
              f'run scripts.slice_rallies first', flush=True)
        return

    # Pre-allocate dense per-vid arrays sized to source video length.
    info = get_video_info(source_video)
    n_total = info.n_frames
    visibility = np.zeros(n_total, dtype=np.int32)
    x = np.zeros(n_total, dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float32)

    print(f'vid={vid}: {len(rally_clips)} rally clips → {n_total:,}-frame dense cache',
          flush=True)
    n_done = n_skipped = n_failed = 0

    with tempfile.TemporaryDirectory(prefix=f'tracknet_{vid}_') as tmpdir:
        for clip_path in rally_clips:
            try:
                clip_vid, set_id, rally = parse_rally_clip_name(clip_path)
            except ValueError as e:
                print(f'  {clip_path.name}: {e}, skipping', flush=True)
                n_skipped += 1
                continue
            if clip_vid != vid:
                continue
            bounds = rally_bounds.get((set_id, rally))
            if bounds is None:
                print(f'  {clip_path.name}: no bounds in shots_master for '
                      f'set={set_id} rally={rally}, skipping', flush=True)
                n_skipped += 1
                continue
            rally_start_f, _ = bounds

            try:
                csv_path = extract_shuttle(clip_path, save_dir=Path(tmpdir))
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f'  {clip_path.name}: TrackNet failed: {type(e).__name__}: {e}',
                      flush=True)
                n_failed += 1
                continue

            # Re-key clip-frame → source-frame and write into the dense arrays.
            src_frames = df['Frame'].to_numpy(dtype=np.int64) + rally_start_f
            valid = (src_frames >= 0) & (src_frames < n_total)
            src_frames = src_frames[valid]
            visibility[src_frames] = df['Visibility'].to_numpy(dtype=np.int32)[valid]
            x[src_frames] = df['X'].to_numpy(dtype=np.float32)[valid]
            y[src_frames] = df['Y'].to_numpy(dtype=np.float32)[valid]

            # Free the per-rally CSV — temp dir would also clean it up at exit.
            csv_path.unlink(missing_ok=True)
            n_done += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame=np.arange(n_total, dtype=np.int32),
        x=x, y=y, visibility=visibility,
    )
    n_visible = int((visibility > 0).sum())
    print(
        f'vid={vid}: wrote {out_path.name} — {n_total:,} frames total, '
        f'{n_visible:,} visible ({100 * n_visible / n_total:.1f}%); '
        f'{n_done} rallies processed, {n_skipped} skipped, {n_failed} failed',
        flush=True,
    )


def _worker(vid: int, master: pd.DataFrame, force: bool) -> None:
    try:
        process_one_vid(vid, master, force=force)
    except Exception as e:
        print(f'vid={vid}: ERROR {type(e).__name__}: {e}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, nargs='*',
                        help='Specific vid(s). Default: all in shots_master.csv.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if cache exists.')
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Concurrent vids. Each spawns a TrackNet subprocess per rally clip; '
             'memory bounded per-rally so high concurrency is fine. 8 is sensible '
             'on a single Blackwell GPU. Default 1.',
    )
    args = parser.parse_args()

    master = pd.read_csv(SHOTS_MASTER_PATH)
    if args.vid:
        vids = sorted(set(args.vid))
    else:
        vids = sorted(master['vid'].unique().tolist())

    n_workers = max(1, min(args.workers, len(vids)))
    print(f'Processing {len(vids)} vid(s) with {n_workers} worker(s): {vids}')

    if n_workers == 1:
        for vid in vids:
            _worker(int(vid), master, args.force)
        return

    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(partial(_worker, master=master, force=args.force),
                 [int(v) for v in vids])


if __name__ == '__main__':
    main()
