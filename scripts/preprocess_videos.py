"""Single-pass per-source-video preprocessing for BRIC.

Walks each source video sequentially, in-rally frames only. For each
rally: seeks to start, iterates frames, runs YOLO on each frame,
resolves Top + Bottom striker bboxes via court projection, buffers the
raw frames in memory. At end of rally, extracts per-stroke RGB
windows from the buffer using the resolved striker bbox at ``target_frame``.

Outputs two caches per source video:

  - ``runtime/cache/players/<vid>.npz``    wide-format resolved striker
                                           bboxes (Top + Bottom) per frame,
                                           length = source video frames.
                                           Court projection + side
                                           resolution baked in — dataset
                                           does O(1) lookup, no per-frame
                                           filter logic. See
                                           ``write_player_tracks_cache``
                                           docstring for the array shapes.
  - ``runtime/cache/rgb/<clip_stem>.npy``  32-frame striker crop tensor
                                           per stroke

Shuttle (TrackNetV3 → ``runtime/cache/shuttle/<vid>.npz``) is produced
by ``scripts.extract_shuttle`` separately, operating on the per-rally
clips from ``scripts.slice_rallies``.

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
from shared.court import (  # noqa: E402
    convert_homogeneous,
    load_all_court_info,
    project,
    scale_pos_by_resolution,
)
from shared.dataset import HOMOGRAPHY_CSV_PATH  # noqa: E402

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


def _project_foot_to_court(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    court_info: dict,
) -> tuple[float, float]:
    """Project a bbox foot-centre to normalised court coords (x_n, y_n).

    Foot-centre = bbox bottom edge midpoint. Source pixels are scaled to
    the homography's reference resolution (1280×720) before projection.
    Result is in [0, 1] when on-court (with eps slack), outside otherwise.
    """
    foot_x = (bbox[0] + bbox[2]) / 2
    foot_y = bbox[3]
    pt = np.array([[foot_x], [foot_y]])
    pt = scale_pos_by_resolution(pt, width=frame_w, height=frame_h)
    pt = convert_homogeneous(pt)
    court_pt = project(court_info['H'], pt)
    x_n = (court_pt[0, 0] - court_info['border_L']) / (
        court_info['border_R'] - court_info['border_L']
    )
    y_n = (court_pt[1, 0] - court_info['border_U']) / (
        court_info['border_D'] - court_info['border_U']
    )
    return float(x_n), float(y_n)


def pick_striker_detection(
    detections: list[dict],
    striker_side: str,
    frame_w: int,
    frame_h: int,
    court_info: dict | None = None,
) -> dict | None:
    """Pick the striker's detection dict (bbox + conf) from per-frame YOLO output.

    With ``court_info`` (BST homography for this vid):
      1. Project each detection's foot-centre to normalised court coords.
      2. Filter to detections whose foot lands inside the court polygon
         (this drops the umpire, coaches, and audience members whose
         bboxes happen to sit in the wrong frame half).
      3. Bucket survivors by court-y (< 0.5 = top half) and pick the
         highest-confidence detection in the striker's half.

    Without ``court_info`` (fallback only — should not happen for
    ShuttleSet vids):
      Use the pixel-midline heuristic. This is the path that produced
      the umpire-cropping bug; only acceptable when no homography exists.

    Returns ``None`` if no on-court detection in the striker's half.
    """
    if not detections:
        return None

    if court_info is not None:
        eps = 0.02
        on_court_with_y: list[tuple[dict, float]] = []
        for d in detections:
            try:
                x_n, y_n = _project_foot_to_court(d['bbox'], frame_w, frame_h, court_info)
            except Exception:
                continue
            if (-eps <= x_n <= 1 + eps) and (-eps <= y_n <= 1 + eps):
                on_court_with_y.append((d, y_n))
        want_top = striker_side.lower() == 'top'
        side_dets = [d for d, y_n in on_court_with_y if (y_n < 0.5) == want_top]
    else:
        midline = frame_h / 2
        side_dets = [
            d for d in detections
            if (d['bbox'][3] < midline) == (striker_side.lower() == 'top')
        ]

    if not side_dets:
        return None
    side_dets.sort(key=lambda d: d['conf'], reverse=True)
    return side_dets[0]


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
    striker_arrays: dict[str, np.ndarray],
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray | None, int]:
    """Build the (32, 224, 224, 3) RGB tensor for one stroke.

    Looks up the striker bbox from ``striker_arrays`` (the wide per-vid
    arrays populated during the rally loop). If the bbox is missing at
    ``target_frame`` (no on-court detection, transient YOLO miss), walks
    outward ±1, ±2, ... up to ±RGB_N_BEFORE looking for a valid
    neighbour. The found bbox is used as the fixed crop region for all
    32 frames in the RGB window.

    :return: (tensor, offset_used). ``tensor`` is None if no striker
        found within ±RGB_N_BEFORE; ``offset_used`` is 0 for direct hit,
        positive for fallback distance, -1 if not found.
    """
    target_f = int(stroke['frame_num'])
    side = stroke['player_side'].lower()  # 'top' or 'bottom'
    valid_arr = striker_arrays[f'{side}_valid']
    bbox_arr = striker_arrays[f'{side}_bbox']

    bbox = None
    used_offset = -1
    n_frames = len(valid_arr)
    for offset in range(RGB_N_BEFORE + 1):
        candidates = (target_f,) if offset == 0 else (target_f - offset, target_f + offset)
        for f in candidates:
            if 0 <= f < n_frames and valid_arr[f]:
                bbox = tuple(float(v) for v in bbox_arr[f])
                used_offset = offset
                break
        if bbox is not None:
            break

    if bbox is None:
        print(
            f'  stroke {stroke["clip_stem"]}: no {side} detection in '
            f'±{RGB_N_BEFORE} frames around {target_f}, skipping RGB'
        )
        return None, -1

    crop_box = expand_and_squarify(bbox, frame_w, frame_h)
    crops = []
    window_start = target_f - RGB_N_BEFORE
    for i in range(RGB_N_FRAMES):
        wf = window_start + i
        # Boundary clamp — should be rare given rally extent guards.
        if wf not in rally_frames:
            available = sorted(rally_frames.keys())
            wf = max(available[0], min(available[-1], wf))
        crops.append(crop_and_resize(rally_frames[wf], crop_box))
    return np.stack(crops, axis=0), used_offset  # (32, 224, 224, 3) uint8


def process_rally(
    extent: tuple[int, str, int, int, int, pd.DataFrame],
    cap: cv2.VideoCapture,
    yolo_model,
    frame_w: int,
    frame_h: int,
    force: bool,
    court_info: dict | None,
    striker_arrays: dict[str, np.ndarray],
) -> None:
    """Process one rally: read frames, run YOLO, resolve strikers, extract RGB.

    Populates the per-vid wide arrays (``striker_arrays``) in-place for
    every in-rally frame: Top + Bottom striker bbox + confidence +
    valid mask. ``extract_stroke_rgb`` later reads from those wide
    arrays — both for the target_frame lookup and for the
    smoothing-fallback ±k frame walk.
    """
    set_id, rally_id, start_f, end_f, strokes = extent
    print(f'  rally set={set_id} rally={rally_id}: frames [{start_f}, {end_f}) — {len(strokes)} strokes')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    rally_frames: dict[int, np.ndarray] = {}

    # YOLO + resolve strikers per frame; write into the per-vid wide arrays.
    for f_idx in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            print(f'  rally set={set_id} rally={rally_id}: short read at frame {f_idx}, stopping early')
            break
        rally_frames[f_idx] = frame
        dets = detect_persons(yolo_model, frame)
        for side in ('top', 'bottom'):
            d = pick_striker_detection(dets, side, frame_w, frame_h, court_info)
            if d is None:
                continue
            striker_arrays[f'{side}_bbox'][f_idx] = d['bbox']
            striker_arrays[f'{side}_conf'][f_idx] = d['conf']
            striker_arrays[f'{side}_valid'][f_idx] = True

    # Per-stroke RGB extraction from the buffered frames + resolved strikers.
    n_ok = n_fallback = n_failed = 0
    for _, stroke in strokes.iterrows():
        out_path = RGB_CACHE_DIR / f'{stroke["clip_stem"]}.npy'
        if out_path.exists() and not force:
            continue
        target_f = int(stroke['frame_num'])
        if target_f not in rally_frames:
            print(f'  stroke {stroke["clip_stem"]}: target frame {target_f} not buffered, skipping')
            n_failed += 1
            continue
        tensor, used_offset = extract_stroke_rgb(
            stroke, rally_frames, striker_arrays, frame_w, frame_h,
        )
        if tensor is None:
            n_failed += 1
            continue
        if used_offset == 0:
            n_ok += 1
        else:
            n_fallback += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, tensor)

    if n_fallback or n_failed:
        print(f'  rally set={set_id} rally={rally_id}: rgb {n_ok} direct, '
              f'{n_fallback} fallback, {n_failed} failed')


def write_player_tracks_cache(
    vid: int,
    video_path: Path,
    frame_w: int,
    frame_h: int,
    n_frames: int,
    fps: float,
    striker_arrays: dict[str, np.ndarray],
) -> Path:
    """Write per-vid resolved striker bboxes (Top + Bottom) to compressed NPZ.

    Wide-format arrays of length ``n_frames`` (full source video length):

      - ``{top,bottom}_bbox``  (N, 4) float32  pixel (x1, y1, x2, y2); zeros
                                               where no valid striker
      - ``{top,bottom}_conf``  (N,)   float32  YOLO confidence
      - ``{top,bottom}_valid`` (N,)   bool     True for in-rally frames with
                                               an on-court detection of that side

    The dataset class does O(1) lookup by source frame index — no
    filter, no court projection at training time. Court/striker
    resolution is baked into this cache (re-extract if those rules change).
    """
    out_path = PLAYERS_CACHE_DIR / f'{vid}.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        # Metadata — small scalars, kept in npz so consumers don't need a sidecar.
        vid=np.int32(vid),
        video_path=np.array(str(video_path)),
        n_frames=np.int32(n_frames),
        fps=np.float32(fps),
        width=np.int32(frame_w),
        height=np.int32(frame_h),
        **striker_arrays,
    )
    return out_path


def process_one_vid(
    vid: int,
    all_strokes: pd.DataFrame,
    court_info_by_vid: dict[int, dict],
    force: bool = False,
) -> None:
    """Run the single-pass preprocessing for one source video."""
    players_cache_path = PLAYERS_CACHE_DIR / f'{vid}.npz'
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

    court_info = court_info_by_vid.get(vid)
    if court_info is None:
        print(f'vid={vid}: WARNING no homography found, falling back to '
              f'pixel-midline striker selection (umpire may contaminate Top crops)')

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
        print(f'vid={vid}: {video_path.name} ({frame_w}×{frame_h} @ {fps:.2f}fps, {n_frames} frames)'
              f'  court_info={"yes" if court_info else "MISSING"}')

        # Pre-allocate per-vid wide arrays. Zeros for non-rally frames + frames
        # with no on-court detection of the side; valid mask separates the two.
        striker_arrays: dict[str, np.ndarray] = {
            'top_bbox':     np.zeros((n_frames, 4), dtype=np.float32),
            'top_conf':     np.zeros(n_frames, dtype=np.float32),
            'top_valid':    np.zeros(n_frames, dtype=bool),
            'bottom_bbox':  np.zeros((n_frames, 4), dtype=np.float32),
            'bottom_conf':  np.zeros(n_frames, dtype=np.float32),
            'bottom_valid': np.zeros(n_frames, dtype=bool),
        }

        extents = compute_rally_extents(strokes)
        print(f'vid={vid}: {len(extents)} rallies, {len(strokes)} strokes')
        for extent in extents:
            process_rally(
                extent, cap, yolo_model, frame_w, frame_h, force, court_info,
                striker_arrays,
            )
    finally:
        cap.release()

    out_path = write_player_tracks_cache(
        vid, video_path, frame_w, frame_h, n_frames, fps, striker_arrays,
    )
    n_top = int(striker_arrays['top_valid'].sum())
    n_bot = int(striker_arrays['bottom_valid'].sum())
    print(f'vid={vid}: wrote {out_path.name} — '
          f'Top valid {n_top:,} frames, Bottom valid {n_bot:,} frames')


def _worker(
    vid: int,
    master: pd.DataFrame,
    court_info_by_vid: dict[int, dict],
    force: bool,
) -> None:
    """Pool worker target — wraps process_one_vid for multiprocessing.Pool."""
    try:
        process_one_vid(vid, master, court_info_by_vid, force=force)
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
    court_info_by_vid = load_all_court_info(HOMOGRAPHY_CSV_PATH)
    print(f'Loaded court_info for {len(court_info_by_vid)} vids from homography.csv')

    if args.vid:
        vids = sorted(set(args.vid))
    else:
        vids = sorted(master['vid'].unique().tolist())

    n_workers = max(1, min(args.workers, len(vids)))
    print(f'Processing {len(vids)} vid(s) with {n_workers} worker(s): {vids}')

    if n_workers == 1:
        for vid in vids:
            _worker(int(vid), master, court_info_by_vid, args.force)
        return

    # spawn (not fork) — ultralytics + CUDA both prefer fresh processes;
    # fork after CUDA init causes hangs on some platforms.
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        pool.map(
            partial(_worker, master=master, court_info_by_vid=court_info_by_vid,
                    force=args.force),
            [int(v) for v in vids],
        )


if __name__ == '__main__':
    main()
