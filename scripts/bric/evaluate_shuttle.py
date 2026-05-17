"""Quality diagnostic for the TrackNetV3 shuttle cache.

Per-stroke visibility rate is the metric that matters for training:
how often was the shuttle detected within the [shuttle_start_f,
shuttle_end_f) window of each stroke. Low rate = TrackNet missing the
shuttle in the trajectory window we'd feed the model.

Reports for the requested vid(s):

  - Per-stroke visibility distribution (mean / median / quantiles)
  - Fraction of strokes above visibility thresholds (50%, 70%, 90%)
  - Top-N worst strokes (lowest visibility) — print clip_stems for
    spot-checking with scratch/validate_rgb.py
  - In-rally vs non-rally visibility comparison (sanity check that
    TrackNet is actually finding more shuttle in match frames than
    non-match content)

Usage:
    uv run python -m scripts.bric.evaluate_shuttle --vid 1
    uv run python -m scripts.bric.evaluate_shuttle --vid 1 --top 20
    uv run python -m scripts.bric.evaluate_shuttle              # all vids with cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

SHOTS_MASTER_PATH = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
SHUTTLE_CACHE_DIR = REPO_ROOT / 'training' / 'bric' / 'cache' / 'shuttle'


def load_visibility_array(vid: int) -> np.ndarray:
    """Load shuttle-visibility booleans (1=detected) indexed by source frame."""
    path = SHUTTLE_CACHE_DIR / f'{vid}.npz'
    if not path.exists():
        raise FileNotFoundError(f'shuttle cache not found: {path}')
    cache = np.load(path)
    # The cache's `frame` array is dense 0..N-1 from TrackNet's per-frame
    # prediction loop, so `visibility[i]` corresponds to source frame i.
    # If a future TrackNet variant skips frames, change this to a sparse
    # build: ``out = np.zeros(...); out[cache['frame']] = cache['visibility']``.
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
    """Build a boolean mask: True at frames covered by any stroke's shuttle window."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, nargs='*',
                        help='Specific vid(s). Default: all vids with shuttle cache.')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of worst strokes to print per vid.')
    args = parser.parse_args()

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
