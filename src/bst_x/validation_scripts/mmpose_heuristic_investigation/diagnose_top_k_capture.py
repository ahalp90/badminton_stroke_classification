#!/usr/bin/env python3
"""Diagnose whether the top-K bbox_score detections reliably include the players.

For a given raw-extract clip (from ``raw_extract.py``), projects each detection's
bbox bottom-centre through the homography into normalised court coords, then
for each frame prints the top-K detections ranked by bbox_score with an
on-court flag. Aggregates across frames: how often does the top-K pool include
at least 2 on-court detections (the two players)?

Use this to test the claim that ``--n-max`` truncation does not systematically
lose players to still, high-confidence audience / officials.

Usage:

    python src/bst_x/validation_scripts/mmpose_heuristic_investigation/diagnose_top_k_capture.py \\
        --raw-dir /path/to/flat_raw_phase1 \\
        --clip-stem 3_1_18_3 \\
        --top-k 8 \\
        --margin 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# File lives at src/bst_x/validation_scripts/mmpose_heuristic_investigation/<this>,
# so the repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'bst_x'))

from pipeline.court_utils import get_court_info  # noqa: E402
from pipeline.config import HOMOGRAPHY_RESOLUTION  # noqa: E402


def project_bottom_center(
    bboxes: np.ndarray,
    ndet: int,
    H: np.ndarray,
    src_w: int,
    src_h: int,
    borders: dict,
) -> np.ndarray:
    """Project bbox bottom-centres through H and normalise to [0, 1].

    :param bboxes: (N_max, 4) array; only first ``ndet`` rows are valid.
    :param ndet: Number of valid detections in this frame.
    :param H: 3x3 homography matrix.
    :param src_w: Source video width in pixels.
    :param src_h: Source video height in pixels.
    :param borders: Dict with ``border_L``, ``border_R``, ``border_U``,
                    ``border_D`` from ``get_court_info``.
    :return: (ndet, 2) array of normalised court coords.
    """
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--raw-dir', type=Path, required=True,
                        help='Flat raw-extract dir with per-clip *_raw_*.npy files.')
    parser.add_argument('--clip-stem', type=str, required=True,
                        help='Clip stem, e.g. 3_1_18_3 (video_id is the first field).')
    parser.add_argument('--homography-csv', type=Path,
                        default=REPO_ROOT / 'src' / 'bst_x' / 'ShuttleSet' / 'set' / 'homography.csv')
    parser.add_argument('--resolution-csv', type=Path,
                        default=REPO_ROOT / 'src' / 'bst_x' / 'ShuttleSet' / 'video_metadata.csv')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Pool size to test against (default 8, to match the original N_max).')
    parser.add_argument('--margin', type=float, default=0.15,
                        help='Normalised tolerance beyond [0,1] for "on court" (default 0.15, '
                             'matches the sticky_anchor generous_margin).')
    parser.add_argument('--frames', type=str, default='all',
                        help='"all" or a range like "20:40" to limit per-frame printout.')
    parser.add_argument('--per-frame', action='store_true',
                        help='Print the full per-frame top-K table (default: aggregate only).')
    args = parser.parse_args()

    vid = int(args.clip_stem.split('_', 1)[0])

    homo_df = pd.read_csv(args.homography_csv).set_index('id')
    court = get_court_info(homo_df, vid)
    H = court['H']

    res_df = pd.read_csv(args.resolution_csv).set_index('id')
    src_w = int(res_df.loc[vid, 'width'])
    src_h = int(res_df.loc[vid, 'height'])

    stem = args.clip_stem
    scores = np.load(args.raw_dir / f'{stem}_raw_scores.npy')  # (F, N_max)
    bboxes = np.load(args.raw_dir / f'{stem}_raw_bboxes.npy')  # (F, N_max, 4)
    ndet_all = np.load(args.raw_dir / f'{stem}_raw_ndet.npy')  # (F,)
    F, N_max = scores.shape

    if args.frames == 'all':
        frames_range = range(F)
    else:
        a, b = args.frames.split(':')
        frames_range = range(int(a), int(b))

    print(f'Clip: {stem}  (vid={vid}, {src_w}x{src_h}, {F} frames, N_max={N_max})')
    print(f'Top-K pool size: {args.top_k}')
    print(f'On-court margin: +/-{args.margin} normalised '
          f'({args.margin * 6.10:.2f} m horizontally, {args.margin * 13.40:.2f} m vertically)')
    print()

    n_frames_tested = 0
    n_any_player_in_pool = 0
    n_two_players_in_pool = 0
    score_rank_of_on_court: list[int] = []

    for f in frames_range:
        ndet = int(ndet_all[f])
        if ndet == 0:
            continue
        n_frames_tested += 1

        scores_f = scores[f, :ndet]
        bboxes_f = bboxes[f]

        court_coords = project_bottom_center(
            bboxes_f, ndet, H, src_w, src_h, court,
        )
        on_court = (
            (court_coords[:, 0] > -args.margin)
            & (court_coords[:, 0] < 1 + args.margin)
            & (court_coords[:, 1] > -args.margin)
            & (court_coords[:, 1] < 1 + args.margin)
        )

        rank = np.argsort(-scores_f)[: args.top_k]
        pool_on_court = on_court[rank].sum()

        if pool_on_court >= 1:
            n_any_player_in_pool += 1
        if pool_on_court >= 2:
            n_two_players_in_pool += 1
        for i in np.where(on_court)[0]:
            pos = int(np.where(rank == i)[0][0]) if i in rank else -1
            if pos >= 0:
                score_rank_of_on_court.append(pos)

        if args.per_frame:
            print(f'--- frame {f:3d}  ndet={ndet}  on_court_in_topK={pool_on_court} ---')
            print(f'  {"rnk":>3} {"score":>6} {"cx":>5} {"cy":>5} {"w":>4} {"h":>4} '
                  f'{"court_x":>7} {"court_y":>7} in?')
            for rank_pos, i in enumerate(rank):
                cx = (bboxes_f[i, 0] + bboxes_f[i, 2]) / 2
                cy = (bboxes_f[i, 1] + bboxes_f[i, 3]) / 2
                w = bboxes_f[i, 2] - bboxes_f[i, 0]
                h = bboxes_f[i, 3] - bboxes_f[i, 1]
                print(f'  {rank_pos:>3} {scores_f[i]:>6.3f} {cx:>5.0f} {cy:>5.0f} '
                      f'{w:>4.0f} {h:>4.0f} {court_coords[i, 0]:>7.3f} '
                      f'{court_coords[i, 1]:>7.3f} {"Y" if on_court[i] else "n"}')

    print()
    print(f'--- Aggregate across {n_frames_tested} frames with detections ---')
    pct1 = n_any_player_in_pool / n_frames_tested * 100 if n_frames_tested else 0
    pct2 = n_two_players_in_pool / n_frames_tested * 100 if n_frames_tested else 0
    print(f'At least 1 on-court detection in top-{args.top_k}: '
          f'{n_any_player_in_pool}/{n_frames_tested} ({pct1:.1f}%)')
    print(f'At least 2 on-court detections in top-{args.top_k}: '
          f'{n_two_players_in_pool}/{n_frames_tested} ({pct2:.1f}%)')
    if score_rank_of_on_court:
        ranks = np.asarray(score_rank_of_on_court)
        print()
        print(f'Rank distribution of on-court detections (lower = higher score):')
        for q in (0, 25, 50, 75, 90, 100):
            print(f'  p{q:>3}  rank {float(np.percentile(ranks, q)):.1f}')
        print(f'  mean rank {float(ranks.mean()):.2f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
