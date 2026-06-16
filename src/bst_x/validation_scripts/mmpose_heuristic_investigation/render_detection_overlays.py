#!/usr/bin/env python3
"""Render per-frame MMPose-detection overlays for a raw-extract clip.

For each frame of the clip, draws:

- The homography's doubles-court rectangle (cyan).
- Every MMPose detection's bbox, colour-coded:
    - **green**         : inside top-K by bbox_score AND projected on-court.
    - **red**           : inside top-K by bbox_score AND projected off-court.
                          (Non-player winning a top-K slot.)
    - **blue**          : rank > top-K AND projected on-court; drawn thick
                          with a ``DISPLACED`` tag.
                          (Player displaced from the top-K pool. The failure mode we care about.)
    - **grey**          : rank > top-K AND projected off-court. (Fine, ignore.)
- Label per bbox: ``rank score`` (e.g. ``0 0.92``).
- Optional: 17-joint keypoint overlay as pink dots for the top-N detections
  per frame when ``--joints-top-n`` > 0. Useful for eyeballing whether
  MMPose placed keypoints sensibly on the person (vs hallucinating onto
  chair structures etc).

Useful for eyeballing whether audience / officials are stealing top-K slots
from the players. See also ``diagnose_top_k_capture.py`` (sibling) for the
aggregate numerical version.

Usage:

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/render_detection_overlays.py \\
        --clip-path /path/to/3_1_18_3.mp4 \\
        --raw-dir /path/to/flat_raw_phase1 \\
        --clip-stem 3_1_18_3 \\
        --out-dir /tmp/detect_overlays_3_1_18_3 \\
        --top-k 8
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# File lives at src/bst_x/validation_scripts/mmpose_heuristic_investigation/<this>,
# so the repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x'))

from pipeline.court_utils import get_court_info  # noqa: E402
from pipeline.config import HOMOGRAPHY_RESOLUTION  # noqa: E402


# RGBA palette. Chosen to stay legible under common red-green colour deficiency:
# the alarm colour (displaced player) is a saturated blue, unambiguous against
# red/green/cyan on the same image.
COLOURS = {
    'topk_on':  (0, 220, 80, 255),    # green: top-K, on-court -> likely player
    'topk_off': (230, 50, 50, 255),   # red: top-K, off-court -> non-player in pool
    'rest_on':  (40, 90, 255, 255),   # blue: below top-K, on-court -> DISPLACED
    'rest_off': (160, 160, 160, 170), # grey: below top-K, off-court -> don't care
    'court':    (0, 255, 255, 255),   # cyan: doubles court rectangle
    'joint':    (255, 105, 180, 235), # hot pink: MMPose keypoint marker
}


def extract_all_frames(clip_path: Path, out_dir: Path) -> list[Path]:
    """Use ffmpeg to extract every frame of ``clip_path`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(clip_path),
        '-vsync', '0', '-start_number', '0',
        str(out_dir / 'frame_%03d.png'),
    ], check=True)
    return sorted(out_dir.glob('frame_*.png'))


def project_bottom_centre(
    bboxes: np.ndarray, ndet: int, H: np.ndarray,
    src_w: int, src_h: int, borders: dict,
) -> np.ndarray:
    """Project bbox bottom-centres through H to normalised court coords."""
    if ndet == 0:
        return np.zeros((0, 2), dtype=np.float32)
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    bx = (bboxes[:ndet, 0] + bboxes[:ndet, 2]) / 2.0 * (aim_w / src_w)
    by = bboxes[:ndet, 3] * (aim_h / src_h)
    pts = np.stack([bx, by, np.ones_like(bx)], axis=0)
    proj = H @ pts
    proj = proj[:2] / proj[2]
    x_norm = (proj[0] - borders['border_L']) / (borders['border_R'] - borders['border_L'])
    y_norm = (proj[1] - borders['border_U']) / (borders['border_D'] - borders['border_U'])
    return np.stack([x_norm, y_norm], axis=1)


def court_corner_pixels(homo_row: pd.Series, src_w: int, src_h: int) -> list[tuple]:
    """Return the 4 annotated court corners in clip-resolution pixel space.

    homography.csv stores corners at HOMOGRAPHY_RESOLUTION; scale up to the
    clip's native resolution for drawing.
    """
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    sx, sy = src_w / aim_w, src_h / aim_h
    pts = []
    for lbl in ('upleft', 'upright', 'downright', 'downleft'):
        pts.append((float(homo_row[f'{lbl}_x']) * sx, float(homo_row[f'{lbl}_y']) * sy))
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--clip-path', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--clip-stem', type=str, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--homography-csv', type=Path,
                        default=REPO_ROOT / 'data' / 'shuttleset' / 'set' / 'homography.csv')
    parser.add_argument('--resolution-csv', type=Path,
                        default=REPO_ROOT / 'data' / 'shuttleset' / 'video_metadata.csv')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--margin', type=float, default=0.15,
                        help='Normalised tolerance for "on court" (default 0.15).')
    parser.add_argument('--frames', type=str, default='all',
                        help='"all" or a range like "20:40".')
    parser.add_argument('--joints-top-n', type=int, default=0,
                        help='If > 0, draw pink keypoint dots for the top-N '
                             'detections by bbox_score in each frame. '
                             'Default 0 (off).')
    parser.add_argument('--joint-radius', type=int, default=6,
                        help='Pink-dot radius in pixels for joint overlays '
                             '(default 6).')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    vid = int(args.clip_stem.split('_', 1)[0])
    homo_df = pd.read_csv(args.homography_csv).set_index('id')
    homo_row = homo_df.loc[vid]
    court = get_court_info(homo_df, vid)
    H = court['H']

    res_df = pd.read_csv(args.resolution_csv).set_index('id')
    src_w = int(res_df.loc[vid, 'width'])
    src_h = int(res_df.loc[vid, 'height'])

    scores = np.load(args.raw_dir / f'{args.clip_stem}_raw_scores.npy')
    bboxes = np.load(args.raw_dir / f'{args.clip_stem}_raw_bboxes.npy')
    ndet_all = np.load(args.raw_dir / f'{args.clip_stem}_raw_ndet.npy')
    # Only loaded when the joint overlay is requested; avoids the memory hit
    # on older extracts that lack the kps file.
    kps = (np.load(args.raw_dir / f'{args.clip_stem}_raw_kps.npy')
           if args.joints_top_n > 0 else None)
    F, _ = scores.shape

    # Extract frames to a tmp dir.
    work = Path(tempfile.mkdtemp(prefix='render_overlays_'))
    try:
        raw_frames = extract_all_frames(args.clip_path, work / 'raw')
        assert len(raw_frames) == F, f'ffmpeg gave {len(raw_frames)} frames, raw has {F}'

        court_corners = court_corner_pixels(homo_row, src_w, src_h)

        if args.frames == 'all':
            frames_range = range(F)
        else:
            a, b = args.frames.split(':')
            frames_range = range(int(a), int(b))

        for f in frames_range:
            img = Image.open(raw_frames[f]).convert('RGB')
            draw = ImageDraw.Draw(img, 'RGBA')

            # Court rectangle.
            draw.line([*court_corners, court_corners[0]],
                      fill=COLOURS['court'], width=3)

            ndet = int(ndet_all[f])
            if ndet > 0:
                scores_f = scores[f, :ndet]
                bboxes_f = bboxes[f, :ndet]
                court_coords = project_bottom_centre(
                    bboxes_f, ndet, H, src_w, src_h, court,
                )
                on_court = (
                    (court_coords[:, 0] > -args.margin)
                    & (court_coords[:, 0] < 1 + args.margin)
                    & (court_coords[:, 1] > -args.margin)
                    & (court_coords[:, 1] < 1 + args.margin)
                )
                rank_order = np.argsort(-scores_f)  # (ndet,)
                rank_of = np.empty(ndet, dtype=int)
                rank_of[rank_order] = np.arange(ndet)

                for i in range(ndet):
                    r = int(rank_of[i])
                    in_pool = r < args.top_k
                    is_on = bool(on_court[i])
                    key = ('topk_on' if in_pool and is_on else
                           'topk_off' if in_pool and not is_on else
                           'rest_on' if is_on else 'rest_off')
                    colour = COLOURS[key]
                    x1, y1, x2, y2 = bboxes_f[i]
                    # Displaced-player boxes draw extra thick so they can't be
                    # missed even when scrolling PNGs quickly.
                    if key == 'rest_on':
                        width = 5
                    elif in_pool:
                        width = 3
                    else:
                        width = 1
                    draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
                    label = f'{r} {scores_f[i]:.2f}'
                    if key == 'rest_on':
                        label = 'DISPLACED ' + label
                    tx, ty = x1 + 2, y1 + 2
                    lbl_w = 110 if key == 'rest_on' else 45
                    draw.rectangle([tx - 1, ty - 1, tx + lbl_w, ty + 13], fill=(0, 0, 0, 200))
                    draw.text((tx, ty), label, fill=colour)

                # Joint overlay on the top-N by bbox_score. Drawn after the
                # bbox loop so the pink dots sit above the outlines.
                if kps is not None and args.joints_top_n > 0:
                    topn = rank_order[: args.joints_top_n]
                    r = args.joint_radius
                    for i in topn:
                        joints = kps[f, i]  # (17, 2)
                        for jx, jy in joints:
                            if np.isnan(jx) or np.isnan(jy):
                                continue
                            draw.ellipse(
                                [jx - r, jy - r, jx + r, jy + r],
                                fill=COLOURS['joint'],
                                outline=(0, 0, 0, 200),
                                width=1,
                            )

            # Frame header.
            header_h = 96 if args.joints_top_n > 0 else 82
            draw.rectangle([10, 10, 240, header_h], fill=(0, 0, 0, 200))
            draw.text((18, 16), f'frame {f:03d}  ndet={ndet}', fill='white')
            draw.text((18, 30), f'top-{args.top_k} pool shown in bold', fill='white')
            draw.text((18, 44), 'green=player, red=non-player in pool', fill='white')
            draw.text((18, 58), 'blue=DISPLACED player, grey=ignore', fill='white')
            if args.joints_top_n > 0:
                draw.text((18, 72), f'pink=joints of top-{args.joints_top_n}', fill=COLOURS['joint'])

            out_path = args.out_dir / f'detect_{args.clip_stem}_f{f:03d}.png'
            img.save(out_path)

        print(f'Wrote {len(list(frames_range))} overlays to {args.out_dir}')
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
