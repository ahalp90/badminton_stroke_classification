"""Run TrackNetV3 over each source video; cache per-frame shuttle predictions.

Subprocess-driven (the vendored ``predict.py`` is CLI-only) per source
video. Caches results as compressed npz at
``runtime/cache/shuttle/<vid>.npz`` with arrays:

  - ``frame``      : (N,) int32   source-video frame indices (0-indexed)
  - ``x``          : (N,) float32 pixel x-coord of detected shuttle
  - ``y``          : (N,) float32 pixel y-coord
  - ``visibility`` : (N,) int32   1 = detected, 0 = no shuttle this frame

Coords stay in source-video pixel space (1920×1080 for ShuttleSet);
normalise at lookup time if needed.

Idempotency: skips vids whose cache file exists. Use ``--force`` to redo.

Parallelism: ``--workers N`` runs N source videos concurrently as
separate processes. Each worker spawns its own TrackNet subprocess
which loads the model fresh — that's ~1-2 GB VRAM per concurrent stream.
On Blackwell with plenty of headroom, 4-8 workers is sensible. Default
1 (serial), predictable for debugging.

Usage:
    uv run python -m scripts.extract_shuttle              # all vids serial
    uv run python -m scripts.extract_shuttle --vid 1
    uv run python -m scripts.extract_shuttle --workers 4
    uv run python -m scripts.extract_shuttle --force
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

SHOTS_MASTER_PATH = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
RAW_VIDEO_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'raw_video'
SHUTTLE_CACHE_DIR = REPO_ROOT / 'runtime' / 'cache' / 'shuttle'


def find_source_video(vid: int) -> Path | None:
    """Locate source mp4 via BST's ``<vid> *.mp4`` naming convention."""
    matches = sorted(RAW_VIDEO_DIR.glob(f'{vid} *.mp4'))
    return matches[0] if matches else None


def process_one_vid(vid: int, force: bool = False) -> None:
    """Run TrackNet on one source video; cache per-frame shuttle as npz."""
    out_path = SHUTTLE_CACHE_DIR / f'{vid}.npz'
    if out_path.exists() and not force:
        print(f'vid={vid}: shuttle cache exists, skipping (use --force to re-run)')
        return

    video_path = find_source_video(vid)
    if video_path is None:
        print(f'vid={vid}: WARNING source video not found in {RAW_VIDEO_DIR}, skipping')
        return

    print(f'vid={vid}: TrackNet on {video_path.name} ...')
    with tempfile.TemporaryDirectory(prefix=f'tracknet_{vid}_') as tmpdir:
        csv_path = extract_shuttle(video_path, save_dir=Path(tmpdir))
        df = pd.read_csv(csv_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame=df['Frame'].to_numpy(dtype=np.int32),
        x=df['X'].to_numpy(dtype=np.float32),
        y=df['Y'].to_numpy(dtype=np.float32),
        visibility=df['Visibility'].to_numpy(dtype=np.int32),
    )
    n_total = len(df)
    n_visible = int((df['Visibility'] > 0).sum())
    print(
        f'vid={vid}: wrote {out_path.name} — {n_total:,} frames, '
        f'{n_visible:,} visible ({100 * n_visible / n_total:.1f}%)'
    )


def _worker(vid: int, force: bool) -> None:
    """Pool worker target — wraps process_one_vid for multiprocessing.Pool."""
    try:
        process_one_vid(vid, force=force)
    except Exception as e:
        print(f'vid={vid}: ERROR {type(e).__name__}: {e}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--vid', type=int, nargs='*',
        help='Specific vid(s) to process. Default: all in shots_master.csv.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run even if cache exists.',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Concurrent vids (each spawns its own TrackNet subprocess, '
             '~1-2 GB VRAM each). 4-8 is sensible on a Blackwell GPU. '
             'Default 1 (serial).',
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
            _worker(int(vid), args.force)
        return

    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(partial(_worker, force=args.force), [int(v) for v in vids])


if __name__ == '__main__':
    main()
