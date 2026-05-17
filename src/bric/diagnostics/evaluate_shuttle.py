"""Diagnostics for the TrackNetV3 shuttle cache.

Two modes:

  --quality  (cross-vid aggregate, sample-based)
    Target-frame validity, window-frame validity, missing-run length
    distribution, out-of-frame / velocity outlier rates, and a single
    go/no-go VERDICT line. Use this once after building / rebuilding the
    cache to decide whether tracking quality is good enough to commit to
    encoder ablation runs.

  default    (per-vid breakdown)
    Per-stroke visibility distribution (mean/median/quantiles), fraction
    above thresholds, in-rally vs non-rally comparison, and the top-N
    worst strokes per vid for spot-checking with validate_rgb.py.

Usage:
    uv run python -m bric.diagnostics.evaluate_shuttle --vid 1
    uv run python -m bric.diagnostics.evaluate_shuttle --vid 1 --top 20
    uv run python -m bric.diagnostics.evaluate_shuttle              # all vids per-vid
    uv run python -m bric.diagnostics.evaluate_shuttle --quality    # aggregate verdict
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))

SHOTS_MASTER_PATH = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
SHUTTLE_CACHE_DIR = REPO_ROOT / 'training' / 'bric' / 'cache' / 'shuttle'
PLAYERS_CACHE_DIR = REPO_ROOT / 'training' / 'bric' / 'cache' / 'players'

# Anomaly thresholds in normalised image-frame coords [0, 1].
# A smash covers ~0.17 normalised frame-x per frame at 25fps; 0.5 would
# require unphysical speeds. Out-of-frame tolerance is generous because
# the shuttle genuinely can be tracked slightly outside the visible court.
_MAX_FRAME_VELOCITY = 0.5
_OUT_OF_FRAME = 0.10


# ---------------------------------------------------------------------------
# Per-vid breakdown (default mode)
# ---------------------------------------------------------------------------
def load_visibility_array(vid: int) -> np.ndarray:
    """Load shuttle-visibility booleans (1=detected) indexed by source frame.

    The cache's ``frame`` array is dense 0..N-1 from TrackNet's per-frame
    prediction loop, so ``visibility[i]`` corresponds to source frame i.
    If a future TrackNet variant skips frames, change this to a sparse
    build: ``out = np.zeros(...); out[cache['frame']] = cache['visibility']``.
    """
    path = SHUTTLE_CACHE_DIR / f'{vid}.npz'
    if not path.exists():
        raise FileNotFoundError(f'shuttle cache not found: {path}')
    cache = np.load(path)
    return (cache['visibility'] > 0).astype(bool)


def per_stroke_visibility(
    strokes: pd.DataFrame, visible: np.ndarray,
) -> pd.DataFrame:
    """For each stroke, compute visibility rate in its shuttle window."""
    n_video_frames = len(visible)
    rates = []
    n_window = []
    for _, s in strokes.iterrows():
        a = max(0, int(s['shuttle_start_f']))
        b = min(n_video_frames, int(s['shuttle_end_f']))
        win = visible[a:b]
        n = len(win)
        rates.append(float(win.mean()) if n > 0 else float('nan'))
        n_window.append(n)
    out = strokes[['clip_stem', 'set_id', 'rally', 'ball_round',
                   'raw_type_en', 'player_side', 'frame_num',
                   'shuttle_start_f', 'shuttle_end_f']].copy()
    out['window_n'] = n_window
    out['visibility_rate'] = rates
    return out


def in_rally_mask(visible: np.ndarray, strokes: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(visible), dtype=bool)
    for _, s in strokes.iterrows():
        a = max(0, int(s['shuttle_start_f']))
        b = min(len(visible), int(s['shuttle_end_f']))
        mask[a:b] = True
    return mask


def report_one_vid(
    vid: int, master: pd.DataFrame, top_n: int = 10,
) -> None:
    visible = load_visibility_array(vid)
    strokes = master[master['vid'] == vid].copy()
    if strokes.empty:
        print(f'vid={vid}: no strokes in shots_master, skipping')
        return

    per_stroke = per_stroke_visibility(strokes, visible)
    valid = per_stroke[per_stroke['window_n'] > 0]
    if valid.empty:
        print(f'vid={vid}: no strokes with non-empty windows')
        return

    rates = valid['visibility_rate']
    print(f'\n=== vid={vid} ({len(valid):,} strokes) ===')
    print(
        f'visibility rate per stroke:\n'
        f'  mean   = {rates.mean():.3f}\n'
        f'  median = {rates.median():.3f}\n'
        f'  std    = {rates.std():.3f}\n'
        f'  min    = {rates.min():.3f}\n'
        f'  max    = {rates.max():.3f}'
    )
    print(
        '  quantiles: '
        + ', '.join(f'p{p}={rates.quantile(p / 100):.2f}' for p in (1, 5, 25, 50, 75, 95, 99))
    )
    for thr in (0.30, 0.50, 0.70, 0.90):
        frac = float((rates >= thr).mean())
        print(f'  fraction with visibility >= {thr:.2f}: {frac * 100:5.1f}%')

    rally_mask = in_rally_mask(visible, strokes)
    n_in_rally = int(rally_mask.sum())
    n_out_rally = int((~rally_mask).sum())
    in_rally_vis = visible[rally_mask].mean() if n_in_rally else float('nan')
    out_rally_vis = visible[~rally_mask].mean() if n_out_rally else float('nan')
    print(
        f'\nframe-level visibility (sanity vs non-match content):\n'
        f'  in-rally  : {in_rally_vis:.3f}  ({n_in_rally:,} frames)\n'
        f'  non-rally : {out_rally_vis:.3f}  ({n_out_rally:,} frames)\n'
        f'  ratio     : {(in_rally_vis / max(out_rally_vis, 1e-9)):.2f}× '
        f'(higher = TrackNet finding more shuttle in rallies than non-match content — good)'
    )

    worst = valid.nsmallest(top_n, 'visibility_rate')
    print(f'\nworst {top_n} strokes by visibility rate:')
    for _, s in worst.iterrows():
        print(
            f'  {s["clip_stem"]:>14}  vis={s["visibility_rate"]:.2f}  '
            f'window_n={s["window_n"]:3d}  '
            f'{s["raw_type_en"]:>20}  player={s["player_side"]}'
        )


# ---------------------------------------------------------------------------
# Cross-vid quality verdict (--quality mode)
# ---------------------------------------------------------------------------
def _run_lengths(mask: np.ndarray) -> list[int]:
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


def quality_report(sample_size: int) -> None:
    print('Loading metadata ...')
    shots = pd.read_csv(SHOTS_MASTER_PATH)
    sample = shots.sample(n=min(sample_size, len(shots)), random_state=42).copy()

    shuttle_cache: dict[int, dict[str, np.ndarray] | None] = {}
    players_cache: dict[int, dict[str, np.ndarray] | None] = {}

    total = len(sample)
    target_valid = 0
    target_missing = 0
    target_skipped = 0

    window_total_frames = 0
    window_valid_frames = 0
    missing_runs_all: list[int] = []
    n_windows_no_valid = 0

    out_of_frame_outliers = 0
    velocity_outliers = 0
    n_pairs_checked = 0
    n_pos_checked = 0

    for _, row in sample.iterrows():
        vid = int(row['vid'])
        target_f = int(row['frame_num'])
        start_f = int(row['shuttle_start_f'])
        end_f = int(row['shuttle_end_f'])

        if vid not in shuttle_cache:
            npz = SHUTTLE_CACHE_DIR / f'{vid}.npz'
            if not npz.exists():
                shuttle_cache[vid] = None
            else:
                with np.load(npz) as f:
                    shuttle_cache[vid] = {k: f[k] for k in f.files}
        s = shuttle_cache[vid]
        if s is None:
            target_skipped += 1
            continue

        if vid not in players_cache:
            npz = PLAYERS_CACHE_DIR / f'{vid}.npz'
            if not npz.exists():
                players_cache[vid] = None
            else:
                with np.load(npz, allow_pickle=True) as f:
                    players_cache[vid] = {k: f[k] for k in f.files}
        p = players_cache[vid]
        if p is None:
            target_skipped += 1
            continue
        img_w, img_h = float(p['width']), float(p['height'])

        vis = s['visibility']
        xs = s['x']
        ys = s['y']
        n = len(vis)

        if 0 <= target_f < n and vis[target_f] > 0:
            target_valid += 1
        else:
            target_missing += 1

        a = max(0, start_f)
        b = min(n, end_f)
        if b <= a:
            continue
        win_vis = (vis[a:b] > 0)
        window_total_frames += len(win_vis)
        window_valid_frames += int(win_vis.sum())
        if not win_vis.any():
            n_windows_no_valid += 1
        missing_runs_all.extend(_run_lengths(win_vis))

        last_xy: tuple[float, float] | None = None
        for f in range(a, b):
            if vis[f] <= 0:
                last_xy = None
                continue
            x_n = float(xs[f]) / img_w
            y_n = float(ys[f]) / img_h
            n_pos_checked += 1
            if (x_n < -_OUT_OF_FRAME or x_n > (1 + _OUT_OF_FRAME)
                    or y_n < -_OUT_OF_FRAME or y_n > (1 + _OUT_OF_FRAME)):
                out_of_frame_outliers += 1
            if last_xy is not None:
                dx = abs(x_n - last_xy[0])
                dy = abs(y_n - last_xy[1])
                n_pairs_checked += 1
                if max(dx, dy) > _MAX_FRAME_VELOCITY:
                    velocity_outliers += 1
            last_xy = (x_n, y_n)

    print()
    print(f'Sampled strokes:         {total}')
    print(f'  skipped (no cache):    {target_skipped}')
    n_eval = total - target_skipped
    if n_eval == 0:
        sys.exit('No strokes evaluable. Check shuttle/players cache paths.')

    print()
    print('--- Target-frame shuttle visibility ---')
    print(f'  visible at target:     {target_valid:6d} / {n_eval}  ({100*target_valid/n_eval:.1f}%)')
    print(f'  missing at target:     {target_missing:6d} / {n_eval}  ({100*target_missing/n_eval:.1f}%)')

    print()
    print('--- Stroke-window shuttle visibility ---')
    print(f'  total frames in windows:  {window_total_frames}')
    print(f'  visible frames:           {window_valid_frames} '
          f'({100*window_valid_frames/max(1,window_total_frames):.1f}%)')
    print(f'  windows with 0 visible:   {n_windows_no_valid}')

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
        print('  (no missing runs — every window had complete visibility)')

    print()
    print('--- Anomaly checks (on visible frames only) ---')
    print(f'  positions checked:        {n_pos_checked}')
    if n_pos_checked > 0:
        print(f'  out-of-frame (>|0.1|):    {out_of_frame_outliers} '
              f'({100*out_of_frame_outliers/n_pos_checked:.2f}%)')
    print(f'  frame pairs checked:      {n_pairs_checked}')
    if n_pairs_checked > 0:
        print(f'  velocity > 0.5/frame:     {velocity_outliers} '
              f'({100*velocity_outliers/n_pairs_checked:.2f}%)')

    target_missing_pct = 100 * target_missing / n_eval
    window_valid_pct = 100 * window_valid_frames / max(1, window_total_frames)
    print()
    if target_missing_pct < 10 and window_valid_pct > 80:
        print(f'VERDICT: shuttle data quality is solid '
              f'({100-target_missing_pct:.0f}% target visibility, '
              f'{window_valid_pct:.0f}% window visibility). '
              'Encoder masking handles the residual gaps.')
    elif target_missing_pct < 25 and window_valid_pct > 60:
        print(f'VERDICT: shuttle visibility is moderate '
              f'(target {100-target_missing_pct:.0f}%, '
              f'window {window_valid_pct:.0f}%). '
              'Workable but expect shuttle features to contribute less than '
              'in fully-visible scenarios.')
    else:
        print(f'VERDICT: shuttle tracking has significant gaps '
              f'(target {100-target_missing_pct:.0f}%, '
              f'window {window_valid_pct:.0f}%). '
              'Encoder ablation results may be capped by tracking quality, '
              'not encoder choice.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, nargs='*',
                        help='Specific vid(s). Default: all vids with shuttle cache.')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of worst strokes to print per vid.')
    parser.add_argument('--quality', action='store_true',
                        help='Run the cross-vid aggregate quality verdict '
                             'instead of the per-vid breakdown.')
    parser.add_argument('--sample', type=int, default=2000,
                        help='Sample size for --quality mode (default 2000).')
    args = parser.parse_args()

    if args.quality:
        quality_report(args.sample)
        return

    master = pd.read_csv(SHOTS_MASTER_PATH)

    if args.vid:
        vids = sorted(set(args.vid))
    else:
        vids = sorted(int(p.stem) for p in SHUTTLE_CACHE_DIR.glob('*.npz'))
        if not vids:
            sys.exit(f'no shuttle caches found in {SHUTTLE_CACHE_DIR}')

    for vid in vids:
        try:
            report_one_vid(int(vid), master, top_n=args.top)
        except FileNotFoundError as e:
            print(f'vid={vid}: {e}')


if __name__ == '__main__':
    main()
