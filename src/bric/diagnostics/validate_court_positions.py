"""Validate our YOLO-derived player court positions against ShuttleSet GT.

ShuttleSet per-rally annotations include player_location_{x,y} — the
ground-truth striker position at each stroke in broadcast pixel coords.
We compare that (projected through the same homography) against the
court coords our pipeline derives from YOLO+ByteTrack bbox foot-centres.

Decision rule (pre-committed):
  mean error < 0.05 court widths   → pipeline sound, ship as-is
  mean error 0.05 - 0.15           → marginal, investigate failure modes
  mean error > 0.15                → systematic issue, debug before more training

USAGE
  uv run python -m bric.diagnostics.validate_court_positions

  Optionally: --sample N (default 500) to control stroke sample size.
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


def project_pixel_to_court(
    pixel_x: float,
    pixel_y: float,
    court_info: dict,
    img_w: float | None,
    img_h: float | None,
) -> tuple[float, float]:
    """Project a (pixel_x, pixel_y) point through the court homography.

    Returns (x_norm, y_norm) in [0, 1] normalised court coords using
    border_L/R/U/D from the court info, matching dataset._build_court.

    Pass img_w / img_h to scale source-resolution pixel coords to the
    homography's reference resolution. Pass None for both if the input
    coords are ALREADY at the homography resolution (e.g. ShuttleSet's
    ``player_location_x/y`` annotations are at 1280x720).
    """
    pt = np.array([[pixel_x], [pixel_y]])
    if img_w is not None and img_h is not None:
        pt = scale_pos_by_resolution(pt, width=img_w, height=img_h)
    pt = convert_homogeneous(pt)
    court_pt = project(court_info['H'], pt)
    x_n = (court_pt[0, 0] - court_info['border_L']) / (
        court_info['border_R'] - court_info['border_L']
    )
    y_n = (court_pt[1, 0] - court_info['border_U']) / (
        court_info['border_D'] - court_info['border_U']
    )
    return float(x_n), float(y_n)


def main(sample_size: int) -> None:
    print(f'Loading metadata ...')
    shots = pd.read_csv(_SHOTS_MASTER)
    video_meta = pd.read_csv(_VIDEO_META).set_index('id')
    court_info_by_vid = load_all_court_info(HOMOGRAPHY_CSV_PATH)

    # Sample strokes deterministically across vids so we don't miss whole matches.
    rng = np.random.default_rng(seed=42)
    sample = shots.sample(n=min(sample_size, len(shots)), random_state=42).copy()

    # Cache loads: set CSVs per match (lazy), player npz per vid (lazy).
    set_csv_cache: dict[tuple[str, str], pd.DataFrame] = {}
    players_cache: dict[int, dict[str, np.ndarray]] = {}

    # Signed differences (ours - gt) — used for systematic bias.
    diffs_x: list[float] = []
    diffs_y: list[float] = []
    # GT-reported area (ShuttleSet 1-9 grid) for cross-checking.
    gt_areas: list[int] = []
    # Our derived coords, GT derived coords (both normalised) — for area binning.
    ours_coords: list[tuple[float, float]] = []
    gt_coords: list[tuple[float, float]] = []
    skipped_no_gt = 0
    skipped_no_bbox = 0
    skipped_no_homography = 0
    skipped_no_match = 0

    for _, row in sample.iterrows():
        vid = int(row['vid'])
        match = row['match']
        set_id = row['set_id']
        rally = row['rally']
        ball_round = row['ball_round']
        frame_num = int(row['frame_num'])
        side = row['player_side'].lower()

        # ---- GT lookup ----
        key = (match, set_id)
        if key not in set_csv_cache:
            csv_path = _SET_DIR / match / f'{set_id}.csv'
            if not csv_path.exists():
                set_csv_cache[key] = None  # mark missing
            else:
                set_csv_cache[key] = pd.read_csv(csv_path)
        set_df = set_csv_cache[key]
        if set_df is None:
            skipped_no_match += 1
            continue

        gt_row = set_df[(set_df['rally'] == rally) & (set_df['ball_round'] == ball_round)]
        if len(gt_row) == 0:
            skipped_no_gt += 1
            continue
        gt_row = gt_row.iloc[0]
        gt_px = gt_row.get('player_location_x')
        gt_py = gt_row.get('player_location_y')
        if pd.isna(gt_px) or pd.isna(gt_py):
            skipped_no_gt += 1
            continue

        # ---- Homography lookup ----
        court_info = court_info_by_vid.get(vid)
        if court_info is None:
            skipped_no_homography += 1
            continue

        # ---- Our derived bbox from players cache ----
        if vid not in players_cache:
            npz_path = _PLAYERS_DIR / f'{vid}.npz'
            if not npz_path.exists():
                players_cache[vid] = None
            else:
                with np.load(npz_path, allow_pickle=True) as f:
                    players_cache[vid] = {k: f[k] for k in f.files}
        p = players_cache[vid]
        if p is None:
            skipped_no_bbox += 1
            continue

        valid = p[f'{side}_valid']
        bboxes = p[f'{side}_bbox']
        if frame_num >= len(valid) or not valid[frame_num]:
            skipped_no_bbox += 1
            continue

        bbox = bboxes[frame_num]
        img_w = float(p['width'])
        img_h = float(p['height'])

        # ---- Project both into normalized court coords ----
        # Our pipeline uses bbox foot-centre.
        ours_px = (float(bbox[0]) + float(bbox[2])) / 2.0
        ours_py = float(bbox[3])
        # Our bbox is at VIDEO resolution -> scale to homography resolution.
        ours_xn, ours_yn = project_pixel_to_court(ours_px, ours_py, court_info, img_w, img_h)
        # GT player_location is ALREADY at homography resolution (1280x720)
        # per ShuttleSet annotation convention -> do NOT scale.
        gt_xn, gt_yn = project_pixel_to_court(float(gt_px), float(gt_py), court_info, None, None)

        diffs_x.append(ours_xn - gt_xn)
        diffs_y.append(ours_yn - gt_yn)
        ours_coords.append((ours_xn, ours_yn))
        gt_coords.append((gt_xn, gt_yn))

        gt_area = gt_row.get('player_location_area')
        if not pd.isna(gt_area):
            gt_areas.append(int(gt_area))

    n = len(diffs_x)
    print()
    print(f'Sampled: {len(sample)}  Compared: {n}')
    print(f'  skipped no GT location:   {skipped_no_gt}')
    print(f'  skipped no bbox at frame: {skipped_no_bbox}')
    print(f'  skipped no homography:    {skipped_no_homography}')
    print(f'  skipped no set CSV:       {skipped_no_match}')
    if n == 0:
        print('No comparisons possible — check data paths.')
        sys.exit(1)

    dx = np.array(diffs_x)
    dy = np.array(diffs_y)
    ex = np.abs(dx)
    ey = np.abs(dy)
    total = np.sqrt(ex**2 + ey**2)

    # --- 1. Signed bias (systematic offset). ---
    print()
    print('--- Signed bias (ours - gt) ---')
    print(f'  mean Δx (signed): {dx.mean():+.4f}   median: {np.median(dx):+.4f}')
    print(f'  mean Δy (signed): {dy.mean():+.4f}   median: {np.median(dy):+.4f}')
    print(f'  → consistent bias ≈ ({dx.mean():+.3f}, {dy.mean():+.3f}) in court coords')
    print('    (positive Δy means our feet land FURTHER from the net than GT)')

    # --- 2. Absolute error distribution (raw). ---
    print()
    print('--- Raw error magnitudes ---')
    print(f'                {"mean":>8} {"p50":>8} {"p95":>8} {"max":>8}')
    print(f'  |Δx_norm|     {ex.mean():>8.4f} {np.percentile(ex, 50):>8.4f} '
          f'{np.percentile(ex, 95):>8.4f} {ex.max():>8.4f}')
    print(f'  |Δy_norm|     {ey.mean():>8.4f} {np.percentile(ey, 50):>8.4f} '
          f'{np.percentile(ey, 95):>8.4f} {ey.max():>8.4f}')
    print(f'  euclidean     {total.mean():>8.4f} {np.percentile(total, 50):>8.4f} '
          f'{np.percentile(total, 95):>8.4f} {total.max():>8.4f}')

    # --- 3. Bias-corrected error: subtract the mean bias before measuring. ---
    bias_x = dx.mean()
    bias_y = dy.mean()
    corr_ex = np.abs(dx - bias_x)
    corr_ey = np.abs(dy - bias_y)
    corr_total = np.sqrt(corr_ex**2 + corr_ey**2)
    print()
    print('--- Bias-corrected error magnitudes ---')
    print('  (after subtracting the systematic offset above)')
    print(f'                {"mean":>8} {"p50":>8} {"p95":>8} {"max":>8}')
    print(f'  |Δx_corr|     {corr_ex.mean():>8.4f} {np.percentile(corr_ex, 50):>8.4f} '
          f'{np.percentile(corr_ex, 95):>8.4f} {corr_ex.max():>8.4f}')
    print(f'  |Δy_corr|     {corr_ey.mean():>8.4f} {np.percentile(corr_ey, 50):>8.4f} '
          f'{np.percentile(corr_ey, 95):>8.4f} {corr_ey.max():>8.4f}')
    print(f'  euclidean     {corr_total.mean():>8.4f} {np.percentile(corr_total, 50):>8.4f} '
          f'{np.percentile(corr_total, 95):>8.4f} {corr_total.max():>8.4f}')

    # --- 4. 3x3 court-grid agreement (what the user actually cares about). ---
    # Bin both ours and gt coords into a 3x3 grid in [0, 1]^2 and check
    # whether we land in the same cell. Cell index = (col, row) with
    # row 0 = closest to net (smallest y), col 0 = leftmost.
    def bin_3x3(x: float, y: float) -> tuple[int, int]:
        col = max(0, min(2, int(x * 3)))
        row = max(0, min(2, int(y * 3)))
        return col, row

    same_cell = 0
    adjacent = 0   # within Chebyshev distance 1 (king's move)
    far = 0
    for (ox, oy), (gx, gy) in zip(ours_coords, gt_coords):
        oc, orow = bin_3x3(ox, oy)
        gc, grow = bin_3x3(gx, gy)
        if (oc, orow) == (gc, grow):
            same_cell += 1
        elif abs(oc - gc) <= 1 and abs(orow - grow) <= 1:
            adjacent += 1
        else:
            far += 1

    print()
    print('--- 3x3 court-area agreement (raw, no bias correction) ---')
    print(f'  same cell:     {same_cell}/{n}  ({100*same_cell/n:.1f}%)')
    print(f'  adjacent cell: {adjacent}/{n}  ({100*adjacent/n:.1f}%)')
    print(f'  far miss:      {far}/{n}  ({100*far/n:.1f}%)')

    # Same agreement check after applying the systematic bias correction.
    same_cell_corr = 0
    adjacent_corr = 0
    far_corr = 0
    for (ox, oy), (gx, gy) in zip(ours_coords, gt_coords):
        oc, orow = bin_3x3(ox - bias_x, oy - bias_y)
        gc, grow = bin_3x3(gx, gy)
        if (oc, orow) == (gc, grow):
            same_cell_corr += 1
        elif abs(oc - gc) <= 1 and abs(orow - grow) <= 1:
            adjacent_corr += 1
        else:
            far_corr += 1

    print()
    print('--- 3x3 court-area agreement (after bias correction) ---')
    print(f'  same cell:     {same_cell_corr}/{n}  ({100*same_cell_corr/n:.1f}%)')
    print(f'  adjacent cell: {adjacent_corr}/{n}  ({100*adjacent_corr/n:.1f}%)')
    print(f'  far miss:      {far_corr}/{n}  ({100*far_corr/n:.1f}%)')

    # ShuttleSet's own area distribution (just for context).
    if gt_areas:
        unique, counts = np.unique(gt_areas, return_counts=True)
        print()
        print('--- GT player_location_area distribution (ShuttleSet bins) ---')
        for u, c in zip(unique, counts):
            print(f'  area {int(u)}: {c} strokes ({100*c/len(gt_areas):.1f}%)')

    # --- Verdict ---
    print()
    agreement_corr = 100 * same_cell_corr / n
    near_agreement_corr = 100 * (same_cell_corr + adjacent_corr) / n
    if agreement_corr >= 70 and near_agreement_corr >= 95:
        verdict = 'SOUND'
        msg = ('Court features land in the same or adjacent 3x3 cell as GT '
               'nearly always after bias correction. Ship as-is.')
    elif agreement_corr >= 50 and near_agreement_corr >= 90:
        verdict = 'MARGINAL'
        msg = ('Same-cell agreement is moderate; near-agreement is good. '
               'Court features are usable but expect them to contribute weakly.')
    else:
        verdict = 'PIPELINE ISSUE'
        msg = ('Even after bias correction, ours and GT disagree on the 3x3 '
               'court area too often. Investigate before training more variants.')
    print(f'VERDICT: {verdict}. {msg}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=int, default=500,
                   help='Number of strokes to sample (default 500)')
    args = p.parse_args()
    main(args.sample)
