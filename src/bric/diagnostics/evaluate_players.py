"""Measure player-cache data quality: missing frames, anomalous coords, gap shape.

For each stroke in shots_master, evaluate the YOLO+ByteTrack player cache:
  - Was the bbox valid at target_frame?
  - Across the stroke window (shuttle_start_f to shuttle_end_f), what
    fraction of frames had valid bboxes?
  - When invalid runs occur, how long are they?
  - Were any valid bboxes anomalous (player projects outside court, or
    velocity exceeds a plausibility threshold)?

The aggregate report tells us whether smoothing / gap-filling / outlier
filtering is worth implementing.

USAGE
  uv run python -m bric.diagnostics.evaluate_players [--sample N]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
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
_PLAYERS_DIR  = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'players'

# Anomaly thresholds — calibrated for normalised court coords in [0, 1].
# Frame-to-frame movement of >0.20 court widths is implausible for a player
# (that's ~3m in one frame at 25fps = 75m/s; usain bolt does ~12m/s).
_MAX_FRAME_VELOCITY = 0.20
# Coords > _OUT_OF_COURT outside the [0, 1] envelope are physically possible
# (player reaches behind baseline) but past this we consider the bbox suspect.
_OUT_OF_COURT = 0.30


def project_bbox_to_court(
    bbox: np.ndarray, court_info: dict, img_w: float, img_h: float,
) -> tuple[float, float]:
    foot_x = (float(bbox[0]) + float(bbox[2])) / 2
    foot_y = float(bbox[3])
    pt = np.array([[foot_x], [foot_y]])
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


def run_lengths(mask: np.ndarray) -> list[int]:
    """Return lengths of consecutive False runs in a bool array."""
    runs: list[int] = []
    cur = 0
    for v in mask:
        if not v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def main(sample_size: int) -> None:
    print('Loading metadata ...')
    shots = pd.read_csv(_SHOTS_MASTER)
    court_info_by_vid = load_all_court_info(HOMOGRAPHY_CSV_PATH)

    sample = shots.sample(n=min(sample_size, len(shots)), random_state=42).copy()

    players_cache: dict[int, dict[str, np.ndarray] | None] = {}
    total = len(sample)
    target_valid = 0
    target_missing = 0
    target_skipped = 0  # no cache for vid

    window_total_frames = 0
    window_valid_frames = 0
    missing_runs_all: list[int] = []
    n_windows_no_valid = 0

    velocity_outliers = 0
    out_of_court_outliers = 0
    n_pairs_checked = 0
    n_pos_checked = 0

    for _, row in sample.iterrows():
        vid = int(row['vid'])
        target_f = int(row['frame_num'])
        start_f = int(row['shuttle_start_f'])
        end_f = int(row['shuttle_end_f'])
        side = row['player_side'].lower()

        if vid not in players_cache:
            npz_path = _PLAYERS_DIR / f'{vid}.npz'
            if not npz_path.exists():
                players_cache[vid] = None
            else:
                with np.load(npz_path, allow_pickle=True) as f:
                    players_cache[vid] = {k: f[k] for k in f.files}
        p = players_cache[vid]
        if p is None:
            target_skipped += 1
            continue

        valid = p[f'{side}_valid']
        bboxes = p[f'{side}_bbox']
        n = len(valid)
        img_w, img_h = float(p['width']), float(p['height'])
        court_info = court_info_by_vid.get(vid)

        # ---- target frame validity ----
        if 0 <= target_f < n and bool(valid[target_f]):
            target_valid += 1
        else:
            target_missing += 1

        # ---- window stats ----
        a = max(0, start_f)
        b = min(n, end_f)
        if b <= a:
            continue
        win_valid = valid[a:b].astype(bool)
        window_total_frames += len(win_valid)
        window_valid_frames += int(win_valid.sum())
        if not win_valid.any():
            n_windows_no_valid += 1
        missing_runs_all.extend(run_lengths(win_valid))

        # ---- anomaly check on the valid frames in this window ----
        if court_info is None:
            continue
        last_pos: tuple[float, float] | None = None
        for i, f in enumerate(range(a, b)):
            if not valid[f]:
                last_pos = None
                continue
            try:
                pos = project_bbox_to_court(bboxes[f], court_info, img_w, img_h)
            except Exception:
                last_pos = None
                continue
            n_pos_checked += 1
            x, y = pos
            if x < -_OUT_OF_COURT or x > (1 + _OUT_OF_COURT) \
                    or y < -_OUT_OF_COURT or y > (1 + _OUT_OF_COURT):
                out_of_court_outliers += 1
            if last_pos is not None:
                dx = abs(pos[0] - last_pos[0])
                dy = abs(pos[1] - last_pos[1])
                n_pairs_checked += 1
                if max(dx, dy) > _MAX_FRAME_VELOCITY:
                    velocity_outliers += 1
            last_pos = pos

    print()
    print(f'Sampled strokes:         {total}')
    print(f'  skipped (no cache):    {target_skipped}')
    n_eval = total - target_skipped
    if n_eval == 0:
        sys.exit('No strokes evaluable. Check players cache path.')
    print()
    print('--- Target-frame bbox validity ---')
    print(f'  valid at target:       {target_valid:6d} / {n_eval}  ({100*target_valid/n_eval:.1f}%)')
    print(f'  missing at target:     {target_missing:6d} / {n_eval}  ({100*target_missing/n_eval:.1f}%)')
    print()
    print('--- Stroke-window frame validity ---')
    print(f'  total frames in windows:  {window_total_frames}')
    print(f'  valid bboxes:             {window_valid_frames} '
          f'({100*window_valid_frames/max(1,window_total_frames):.1f}%)')
    print(f'  windows with 0 valid:     {n_windows_no_valid}')
    print()
    print('--- Distribution of missing-frame run lengths ---')
    if missing_runs_all:
        arr = np.array(missing_runs_all)
        print(f'  total runs of missing:    {len(arr)}')
        print(f'  mean length:              {arr.mean():.1f} frames')
        print(f'  p50 / p90 / p99:          {int(np.percentile(arr,50))} / '
              f'{int(np.percentile(arr,90))} / {int(np.percentile(arr,99))}')
        print(f'  max length:               {int(arr.max())} frames')
        bins = Counter(min(v, 11) for v in arr)
        print('  length histogram:')
        for k in sorted(bins):
            label = '11+' if k == 11 else str(k)
            bar = '#' * min(40, int(bins[k] * 40 / len(arr)))
            print(f'    {label:>4}: {bins[k]:5d}  {bar}')
    else:
        print('  (no missing runs — every window had complete bboxes)')

    print()
    print('--- Anomaly checks (on valid bboxes only) ---')
    print(f'  positions checked:        {n_pos_checked}')
    if n_pos_checked > 0:
        print(f'  out-of-court (>|0.3|):    {out_of_court_outliers} '
              f'({100*out_of_court_outliers/n_pos_checked:.2f}%)')
    print(f'  frame pairs checked:      {n_pairs_checked}')
    if n_pairs_checked > 0:
        print(f'  velocity > 0.2/frame:     {velocity_outliers} '
              f'({100*velocity_outliers/n_pairs_checked:.2f}%)')

    # --- Verdict on whether smoothing is worth implementing ---
    target_missing_pct = 100 * target_missing / n_eval
    window_valid_pct = 100 * window_valid_frames / max(1, window_total_frames)
    print()
    if target_missing_pct < 1 and window_valid_pct > 95:
        print('VERDICT: data quality is high. Smoothing / interpolation not warranted.')
    elif target_missing_pct < 5 and window_valid_pct > 85:
        print('VERDICT: minor missing data. Smoothing would help marginally; '
              'low priority unless time permits.')
    else:
        print('VERDICT: significant missing data. Linear interpolation across '
              'short gaps (< p90 of run lengths) is likely worth implementing.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=int, default=2000,
                   help='Number of strokes to sample (default 2000)')
    args = p.parse_args()
    main(args.sample)
