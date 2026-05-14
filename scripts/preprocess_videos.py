"""Single-pass per-source-video preprocessing for BRIC.

Walks each source video sequentially, in-rally frames only. For each
rally: seeks to start, iterates frames, runs YOLO on each frame, buffers
the raw frames in memory. At end of rally, extracts per-stroke RGB
windows from the buffer using the striker bbox at ``target_frame``.

Outputs two caches per source video:

  - ``runtime/cache/players/<vid>.json``   per-frame YOLO detections
                                           (in-rally frames only)
  - ``runtime/cache/rgb/<clip_stem>.npy``  32-frame striker crop tensor
                                           per stroke

A third task (TrackNetV3 → ``runtime/cache/shuttle/<vid>.npz``) lands
in Day 8 in this same loop — code structure leaves a clear slot.

Why one pass, in-memory: the source videos are 1-2hr broadcast mp4s but
the actual play is ~30 min per video, ~4-5× less work than processing
every frame. Doing all preprocessing tasks in the same pass amortises
decode cost, matches the inference pipeline's structure (which runs
YOLO + TrackNet over the whole upload then per-stroke extracts RGB),
and keeps all caches indexed by source-video frame number — direct
lookup, no offset conversion.

Memory: rally buffer peaks at ~4.6 GB at 1080p × 25fps × ~30s — comfortable
on GB10. Buffer freed between rallies to keep total stable.

Idempotency: per-vid skip if the player-tracks cache exists; per-stroke
skip on RGB if the npy file exists. Use ``--force`` to re-run.

Parallelism: ``--workers N`` runs N source videos concurrently as
separate processes (one ``cv2.VideoCapture`` + one YOLO model each).
On GB10's Blackwell GPU, 4-8 workers keep the GPU saturated; CPU decode
in each worker runs in parallel and pushes inference batches to the
shared GPU. Default is 1 for predictable debugging output. Use shell
parallelism (``xargs -P``) instead when you want per-job log files.

Usage:
    python -m scripts.preprocess_videos                    # all vids serial
    python -m scripts.preprocess_videos --vid 1            # one vid
    python -m scripts.preprocess_videos --vid 1 2 3        # several
    python -m scripts.preprocess_videos --workers 8        # 8 vids at once
    python -m scripts.preprocess_videos --force            # ignore caches
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from perception.players import DEFAULT_YOLO_WEIGHTS  # noqa: E402

SHOTS_MASTER_PATH = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
RAW_VIDEO_DIR = REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'raw_video'
PLAYERS_CACHE_DIR = REPO_ROOT / 'runtime' / 'cache' / 'players'
RGB_CACHE_DIR = REPO_ROOT / 'runtime' / 'cache' / 'rgb'

# RGB window: 32 frames, target at index 16 (16 before, 16 after, 16th = target).
RGB_N_FRAMES = 32
RGB_N_BEFORE = 16          # window = [target - 16, target + 16)
RGB_TARGET_SIZE = 224      # square crop side (H = W = 224)
RGB_PAD_PCT = 0.50         # bbox expansion before square padding

# Rally bounds buffer past BST's shuttle bounds, ensures the RGB window
# (±16 frames) for the first/last stroke of a rally fits inside the
# rally extent we read. BST's first/last fallback is ±0.5s (12f @ 25fps),
# so a 4f cushion is sufficient; 8 to be safe.
RALLY_BUFFER_FRAMES = 8


def find_source_video(vid: int) -> Path | None:
    """Locate the source mp4 for a vid via BST's ``<vid> *.mp4`` convention."""
    matches = sorted(RAW_VIDEO_DIR.glob(f'{vid} *.mp4'))
    return matches[0] if matches else None


def compute_rally_extents(strokes: pd.DataFrame) -> list[tuple[int, int, int, pd.DataFrame]]:
    """Group strokes by (set, rally); return per-rally (set, rally, start_f, end_f, strokes).

    Extent = ``[min(shuttle_start_f, min(frame_num) - 16) - buffer,
              max(shuttle_end_f, max(frame_num) + 16) + buffer)``

    Ensures the per-stroke RGB window (±16 frames) fits inside the
    rally read; BST's per-stroke shuttle bounds aren't always wider
    than that (first/last shot fallback is ±0.5s = ±12 frames).
    """
    extents = []
    for (set_id, rally_id), grp in strokes.groupby(['set_id', 'rally'], sort=True):
        rally_start = int(min(
            grp['shuttle_start_f'].min(),
            grp['frame_num'].min() - RGB_N_BEFORE,
        )) - RALLY_BUFFER_FRAMES
        rally_end = int(max(
            grp['shuttle_end_f'].max(),
            grp['frame_num'].max() + (RGB_N_FRAMES - RGB_N_BEFORE),
        )) + RALLY_BUFFER_FRAMES
        rally_start = max(0, rally_start)
        extents.append((set_id, int(rally_id), rally_start, rally_end, grp))
    extents.sort(key=lambda x: x[2])  # ascending start_f for sequential reads
    return extents


def detect_persons(yolo_model, frame: np.ndarray) -> list[dict]:
    """Run YOLO on one frame; return list of person detections.

    Each detection: ``{bbox: (x1, y1, x2, y2), conf: float}``. No
    tracking — frames are independent. Player identity (top/bottom)
    is assigned per frame at lookup time using foot-y position.
    """
    results = yolo_model(frame, classes=[0], verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return []
    xyxys = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    return [
        {'bbox': tuple(float(v) for v in xyxy), 'conf': float(c)}
        for xyxy, c in zip(xyxys, confs)
    ]


def pick_striker_bbox(
    detections: list[dict],
    striker_side: str,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    """Pick the striker's bbox from per-frame YOLO detections.

    Two-step heuristic suited to broadcast singles footage:
      1. Split detections by foot-y position (bottom edge of bbox)
         relative to the frame midline → top half / bottom half.
      2. Within the striker's half, pick the highest-confidence detection.

    Returns ``None`` if the striker's half has no detection at this frame.
    """
    if not detections:
        return None
    midline = frame_height / 2
    side_dets = [
        d for d in detections
        if (d['bbox'][3] < midline) == (striker_side.lower() == 'top')
    ]
    if not side_dets:
        return None
    side_dets.sort(key=lambda d: d['conf'], reverse=True)
    return side_dets[0]['bbox']


def expand_and_squarify(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    pad_pct: float = RGB_PAD_PCT,
) -> tuple[int, int, int, int]:
    """Expand bbox by ``pad_pct`` per side, square it, clamp to frame bounds."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * (1 + 2 * pad_pct)
    h = (y2 - y1) * (1 + 2 * pad_pct)
    side = max(w, h)
    half = side / 2
    x1n, y1n = int(round(cx - half)), int(round(cy - half))
    x2n, y2n = int(round(cx + half)), int(round(cy + half))
    # Clamp; this can degrade aspect ratio at frame edges but that's
    # rare and preferable to OOB indexing.
    x1n = max(0, x1n)
    y1n = max(0, y1n)
    x2n = min(frame_w, x2n)
    y2n = min(frame_h, y2n)
    return x1n, y1n, x2n, y2n


def crop_and_resize(
    frame: np.ndarray,
    crop_box: tuple[int, int, int, int],
    target_size: int = RGB_TARGET_SIZE,
) -> np.ndarray:
    """Crop frame to ``crop_box``, resize to ``target_size × target_size``.

    Returns RGB uint8 array of shape ``(target_size, target_size, 3)``.
    Input frame is BGR (cv2 native); we convert to RGB on the way out.
    """
    x1, y1, x2, y2 = crop_box
    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def extract_stroke_rgb(
    stroke: pd.Series,
    rally_frames: dict[int, np.ndarray],
    detections_at_target: list[dict],
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """Build the (32, 224, 224, 3) RGB tensor for one stroke.

    Returns None (and logs) if the striker bbox is missing at target.
    """
    target_f = int(stroke['frame_num'])
    striker_side = stroke['player_side']
    bbox = pick_striker_bbox(detections_at_target, striker_side, frame_h)
    if bbox is None:
        print(
            f'  stroke {stroke["clip_stem"]}: no {striker_side} detection at frame '
            f'{target_f}, skipping RGB'
        )
        return None
    crop_box = expand_and_squarify(bbox, frame_w, frame_h)

    crops = []
    window_start = target_f - RGB_N_BEFORE
    for offset in range(RGB_N_FRAMES):
        wf = window_start + offset
        # Boundary clamp — should be rare given rally extent guards.
        if wf not in rally_frames:
            available = sorted(rally_frames.keys())
            wf = max(available[0], min(available[-1], wf))
        crops.append(crop_and_resize(rally_frames[wf], crop_box))
    return np.stack(crops, axis=0)  # (32, 224, 224, 3) uint8


def process_rally(
    extent: tuple[int, str, int, int, int, pd.DataFrame],
    cap: cv2.VideoCapture,
    yolo_model,
    frame_w: int,
    frame_h: int,
    force: bool,
) -> dict[int, list[dict]]:
    """Process one rally: read frames, run YOLO, extract per-stroke RGB.

    Returns the per-frame YOLO detections accumulated in this rally
    (frame_idx -> list[detection]) for the caller to merge into the
    per-vid player-tracks cache.
    """
    set_id, rally_id, start_f, end_f, strokes = extent
    print(f'  rally set={set_id} rally={rally_id}: frames [{start_f}, {end_f}) — {len(strokes)} strokes')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    rally_frames: dict[int, np.ndarray] = {}
    rally_dets: dict[int, list[dict]] = {}

    for f_idx in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            print(f'  rally set={set_id} rally={rally_id}: short read at frame {f_idx}, stopping early')
            break
        rally_frames[f_idx] = frame
        rally_dets[f_idx] = detect_persons(yolo_model, frame)

    # Per-stroke RGB extraction from the buffered frames.
    for _, stroke in strokes.iterrows():
        out_path = RGB_CACHE_DIR / f'{stroke["clip_stem"]}.npy'
        if out_path.exists() and not force:
            continue
        target_f = int(stroke['frame_num'])
        if target_f not in rally_dets:
            print(f'  stroke {stroke["clip_stem"]}: target frame {target_f} not buffered, skipping')
            continue
        tensor = extract_stroke_rgb(
            stroke, rally_frames, rally_dets[target_f], frame_w, frame_h,
        )
        if tensor is None:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, tensor)

    return rally_dets


def write_player_tracks_cache(
    vid: int,
    video_path: Path,
    frame_w: int,
    frame_h: int,
    n_frames: int,
    fps: float,
    all_dets: dict[int, list[dict]],
) -> Path:
    """Write per-vid YOLO detections (in-rally only) to JSON cache."""
    payload = {
        'vid':         vid,
        'video_path':  str(video_path),
        'n_frames':    n_frames,
        'fps':         fps,
        'width':       frame_w,
        'height':      frame_h,
        # JSON keys must be strings; dataset code does int(k) on read.
        'detections_per_frame': {
            str(f): [
                {'bbox': list(d['bbox']), 'conf': d['conf']} for d in dets
            ]
            for f, dets in sorted(all_dets.items())
        },
    }
    out_path = PLAYERS_CACHE_DIR / f'{vid}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(',', ':')))
    return out_path


def process_one_vid(vid: int, all_strokes: pd.DataFrame, force: bool = False) -> None:
    """Run the single-pass preprocessing for one source video."""
    players_cache_path = PLAYERS_CACHE_DIR / f'{vid}.json'
    if players_cache_path.exists() and not force:
        print(f'vid={vid}: players cache exists, skipping (use --force to re-run)')
        return

    video_path = find_source_video(vid)
    if video_path is None:
        print(f'vid={vid}: WARNING source video not found in {RAW_VIDEO_DIR}, skipping')
        return

    strokes = all_strokes[all_strokes['vid'] == vid]
    if strokes.empty:
        print(f'vid={vid}: no strokes in shots_master.csv, skipping')
        return

    # Lazy import — keeps the module importable in environments without ultralytics.
    from ultralytics import YOLO
    yolo_model = YOLO(str(DEFAULT_YOLO_WEIGHTS))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'vid={vid}: WARNING could not open {video_path}, skipping')
        return

    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'vid={vid}: {video_path.name} ({frame_w}×{frame_h} @ {fps:.2f}fps, {n_frames} frames)')

        extents = compute_rally_extents(strokes)
        print(f'vid={vid}: {len(extents)} rallies, {len(strokes)} strokes')

        all_dets: dict[int, list[dict]] = {}
        for extent in extents:
            rally_dets = process_rally(extent, cap, yolo_model, frame_w, frame_h, force)
            all_dets.update(rally_dets)
    finally:
        cap.release()

    out_path = write_player_tracks_cache(
        vid, video_path, frame_w, frame_h, n_frames, fps, all_dets,
    )
    print(f'vid={vid}: wrote {out_path.name} — {len(all_dets):,} in-rally frames cached')


def _worker(vid: int, master: pd.DataFrame, force: bool) -> None:
    """Pool worker target — wraps process_one_vid for multiprocessing.Pool."""
    try:
        process_one_vid(vid, master, force=force)
    except Exception as e:
        # Don't let one bad vid kill the whole pool; surface and move on.
        print(f'vid={vid}: ERROR {type(e).__name__}: {e}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--vid', type=int, nargs='*',
        help='Specific vid(s) to process. Default: all vids in shots_master.csv.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run even if caches exist (overwrites players cache + RGB tensors).',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of source videos to process concurrently (separate processes). '
             'Each worker loads its own YOLO model and shares the GPU. '
             '4-8 is a sensible range on a single Blackwell GPU; '
             'tune to GPU memory headroom. Default 1 (serial).',
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

    # spawn (not fork) — ultralytics + CUDA both prefer fresh processes;
    # fork after CUDA init causes hangs on some platforms.
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(partial(_worker, master=master, force=args.force),
                 [int(v) for v in vids])


if __name__ == '__main__':
    main()
