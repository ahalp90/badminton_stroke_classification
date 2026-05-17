"""Trace the court-coord pipeline for a few sample strokes to locate the bias source.

Earlier validation showed a consistent +0.48 / +0.56 offset between our
YOLO-foot-derived court coords and ShuttleSet's GT player_location coords,
after both go through the same homography pipeline. That's too big to be
torso-vs-feet; some structural mismatch is in play.

This script prints intermediate values at every step of the projection
chain for a small sample (3 strokes), so we can see exactly where the
divergence appears.

USAGE
  uv run python -m bric.diagnostics.debug_court_bias [--n 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from shared.court import (  # noqa: E402
    HOMOGRAPHY_RESOLUTION,
    convert_homogeneous,
    load_all_court_info,
    project,
    scale_pos_by_resolution,
)
from shared.dataset import HOMOGRAPHY_CSV_PATH  # noqa: E402

_SHOTS_MASTER = _REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
_SET_DIR      = _REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'set'
_VIDEO_META   = _REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'video_metadata.csv'
_PLAYERS_DIR  = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'players'


def trace_stroke(row: pd.Series, court_info: dict, video_meta: pd.DataFrame) -> None:
    """Print every intermediate coord for one stroke, both GT and ours."""
    vid = int(row['vid'])
    match = row['match']
    set_id = row['set_id']
    rally = row['rally']
    ball_round = row['ball_round']
    frame_num = int(row['frame_num'])
    side = row['player_side'].lower()

    print()
    print('=' * 78)
    print(f'STROKE: vid={vid}  match={match}  {set_id} rally={rally} ball={ball_round}')
    print(f'        frame={frame_num}  player_side={side}  raw_type={row["raw_type_en"]}')
    print('=' * 78)

    # ---- Resolutions ----
    vmeta = video_meta.loc[vid]
    print(f'\nVideo resolution (video_metadata.csv): {int(vmeta["width"])}x{int(vmeta["height"])}')
    print(f'Homography expects (HOMOGRAPHY_RESOLUTION): {HOMOGRAPHY_RESOLUTION[0]}x{HOMOGRAPHY_RESOLUTION[1]}')

    # ---- Players cache resolution ----
    npz_path = _PLAYERS_DIR / f'{vid}.npz'
    if not npz_path.exists():
        print(f'  ! no players cache for vid={vid}')
        return
    with np.load(npz_path, allow_pickle=True) as f:
        p = {k: f[k] for k in f.files}
    cache_w, cache_h = float(p['width']), float(p['height'])
    print(f'Players cache resolution: {cache_w}x{cache_h}')

    if cache_w != float(vmeta['width']) or cache_h != float(vmeta['height']):
        print('  ! WARNING: players cache resolution != video_metadata resolution')

    # ---- GT lookup ----
    csv_path = _SET_DIR / match / f'{set_id}.csv'
    if not csv_path.exists():
        print(f'  ! no set CSV at {csv_path}')
        return
    set_df = pd.read_csv(csv_path)
    gt_row = set_df[(set_df['rally'] == rally) & (set_df['ball_round'] == ball_round)]
    if len(gt_row) == 0:
        print('  ! no GT row matching (rally, ball_round)')
        return
    gt_row = gt_row.iloc[0]
    gt_px = float(gt_row['player_location_x'])
    gt_py = float(gt_row['player_location_y'])
    gt_area = gt_row.get('player_location_area')
    gt_player = gt_row.get('player')  # 'A' or 'B'

    # ---- Our bbox ----
    valid = p[f'{side}_valid']
    bboxes = p[f'{side}_bbox']
    if frame_num >= len(valid) or not valid[frame_num]:
        print(f'  ! no valid bbox at frame {frame_num} for side={side}')
        return
    bbox = bboxes[frame_num]
    ours_px = (float(bbox[0]) + float(bbox[2])) / 2.0
    ours_py = float(bbox[3])

    # Also compute torso-centre (for comparison — maybe GT uses torso)
    ours_torso_px = (float(bbox[0]) + float(bbox[2])) / 2.0
    ours_torso_py = (float(bbox[1]) + float(bbox[3])) / 2.0

    # And bbox top (head)
    ours_head_px = (float(bbox[0]) + float(bbox[2])) / 2.0
    ours_head_py = float(bbox[1])

    print()
    print('PIXEL COORDS (broadcast frame)')
    print(f'  GT player_location:        ({gt_px:8.2f}, {gt_py:8.2f})  [player={gt_player}, area={gt_area}]')
    print(f'  Our bbox:                  x1,y1=({float(bbox[0]):.1f},{float(bbox[1]):.1f})  '
          f'x2,y2=({float(bbox[2]):.1f},{float(bbox[3]):.1f})')
    print(f'  Our foot-centre  (used):   ({ours_px:8.2f}, {ours_py:8.2f})')
    print(f'  Our torso-centre:          ({ours_torso_px:8.2f}, {ours_torso_py:8.2f})')
    print(f'  Our head-centre:           ({ours_head_px:8.2f}, {ours_head_py:8.2f})')
    print(f'  Pixel Δ (foot - GT):       ({ours_px - gt_px:+.2f}, {ours_py - gt_py:+.2f})')
    print(f'  Pixel Δ (torso - GT):      ({ours_torso_px - gt_px:+.2f}, {ours_torso_py - gt_py:+.2f})')

    # ---- After scale_pos_by_resolution (using cache resolution) ----
    def scale_with(px, py, w, h):
        arr = np.array([[px], [py]])
        return scale_pos_by_resolution(arr, width=w, height=h)

    gt_scaled = scale_with(gt_px, gt_py, cache_w, cache_h)
    ours_scaled = scale_with(ours_px, ours_py, cache_w, cache_h)

    print()
    print(f'AFTER scale_pos_by_resolution (source={cache_w}x{cache_h} -> {HOMOGRAPHY_RESOLUTION})')
    print(f'  GT:     ({gt_scaled[0,0]:.2f}, {gt_scaled[1,0]:.2f})')
    print(f'  Ours:   ({ours_scaled[0,0]:.2f}, {ours_scaled[1,0]:.2f})')

    # Show what GT looks like if it's ALREADY at homography resolution (no scaling needed)
    print()
    print('If GT were already at HOMOGRAPHY_RESOLUTION (no scaling):')
    print(f'  GT raw:  ({gt_px:.2f}, {gt_py:.2f})')

    # ---- After homography projection ----
    H = court_info['H']
    gt_court = project(H, convert_homogeneous(gt_scaled))
    ours_court = project(H, convert_homogeneous(ours_scaled))

    # Also project GT WITHOUT pre-scaling (in case GT is already at H resolution)
    gt_raw_homo = convert_homogeneous(np.array([[gt_px], [gt_py]]))
    gt_court_noscale = project(H, gt_raw_homo)

    print()
    print('AFTER homography projection (court-space coords, pre-normalisation)')
    print(f'  GT  (scaled then projected):   ({gt_court[0,0]:.2f}, {gt_court[1,0]:.2f})')
    print(f'  GT  (no scale, projected):     ({gt_court_noscale[0,0]:.2f}, {gt_court_noscale[1,0]:.2f})')
    print(f'  Ours (scaled then projected):  ({ours_court[0,0]:.2f}, {ours_court[1,0]:.2f})')
    print(f'  Borders (L,R,U,D):             '
          f'({court_info["border_L"]:.2f}, {court_info["border_R"]:.2f}, '
          f'{court_info["border_U"]:.2f}, {court_info["border_D"]:.2f})')

    # ---- Normalised ----
    def norm(c):
        x = (c[0, 0] - court_info['border_L']) / (court_info['border_R'] - court_info['border_L'])
        y = (c[1, 0] - court_info['border_U']) / (court_info['border_D'] - court_info['border_U'])
        return x, y

    gt_n = norm(gt_court)
    gt_n_noscale = norm(gt_court_noscale)
    ours_n = norm(ours_court)

    print()
    print('NORMALISED COURT COORDS  [0, 1]')
    print(f'  GT  (scaled then projected):   ({gt_n[0]:.4f}, {gt_n[1]:.4f})')
    print(f'  GT  (no scale, projected):     ({gt_n_noscale[0]:.4f}, {gt_n_noscale[1]:.4f})')
    print(f'  Ours (scaled then projected):  ({ours_n[0]:.4f}, {ours_n[1]:.4f})')
    print(f'  Δ (ours - gt, with scaling):   ({ours_n[0] - gt_n[0]:+.4f}, {ours_n[1] - gt_n[1]:+.4f})')
    print(f'  Δ (ours - gt, no GT scaling):  ({ours_n[0] - gt_n_noscale[0]:+.4f}, '
          f'{ours_n[1] - gt_n_noscale[1]:+.4f})')


def main(n: int) -> None:
    shots = pd.read_csv(_SHOTS_MASTER)
    video_meta = pd.read_csv(_VIDEO_META).set_index('id')
    court_info_by_vid = load_all_court_info(HOMOGRAPHY_CSV_PATH)

    # Pick first N strokes from different matches for variety
    sample = shots.drop_duplicates('vid').head(n)

    for _, row in sample.iterrows():
        vid = int(row['vid'])
        court_info = court_info_by_vid.get(vid)
        if court_info is None:
            print(f'No homography for vid={vid}, skipping')
            continue
        trace_stroke(row, court_info, video_meta)

    print()
    print('=' * 78)
    print('INTERPRETATION GUIDE')
    print('=' * 78)
    print("""
Compare the two 'Δ' lines at the bottom of each stroke:
  - 'Δ ... with scaling':    treats GT as 1920x1080 (or whatever cache resolution)
  - 'Δ ... no GT scaling':   treats GT as already at 1280x720 (homography resolution)

If the 'no scaling' Δ is small (~0.05), GT is at 1280x720 and we should NOT
be scaling it. Our pipeline scales it as if it were at video resolution.

If both Δs are large, the issue is elsewhere — probably player attribution
(we're picking the opposite player from what GT references). In that case,
look at the pixel coords: is our bbox FAR from the GT in pixel space? If
yes, wrong player; if no, projection issue.

If pixel Δ is small but court Δ is large, the homography itself is producing
divergent outputs for nearby inputs — suspect court border calculation.
""")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=3, help='Number of strokes to trace (default 3)')
    args = p.parse_args()
    main(args.n)
