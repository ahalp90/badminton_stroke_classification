"""Estimate how often the jitter pushes the shuttle off-screen.

Loads the collated pos and shuttle arrays from a training split,
simulates the same shift the training-time ``ConstrainedJitter``
applies, and reports a few summary numbers: how often the shuttle
is pushed off-screen per clip and per frame. Useful for checking
whether the jitter is replacing too many real shuttle observations
with the off-screen sentinel, especially for shot classes where
the shuttle naturally lives near the camera edge (e.g. net shots).

Usage on remote::

    cd badminton_stroke_classifier
    PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \\
        python scripts/estimate_shuttle_oob_rate.py \\
        --collated-dir src/bst_refactor/stroke_classification/preparing_data/ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop \\
        --n-trials 50

``--n-trials`` controls how many random shifts are drawn per clip.
50 is plenty for stable means at p_jitter = 0.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _layered_bounds_for_clip(
    pos: np.ndarray, eps: float = 0.15,
) -> tuple[float, float, float, float]:
    """Compute the per-clip min and max shift on each axis.

    Mirrors what ``ConstrainedJitter`` does at training time. Sentinel
    zero frames (padding and detection failures) are excluded from the
    min/max so they don't throw off the bound calculation.

    :param pos: shape ``(t, 2, 2)``: time x player x xy.
    :return: ``(dy_min, dy_max, dx_min, dx_max)``.
    """
    is_sentinel = (np.abs(pos) < 1e-9).all(axis=-1)  # (t, m)
    pos_for_max = np.where(is_sentinel[..., None], -np.inf, pos)
    pos_for_min = np.where(is_sentinel[..., None], np.inf, pos)

    y_top_max = pos_for_max[:, 0, 1].max()
    y_top_min = pos_for_min[:, 0, 1].min()
    y_bot_max = pos_for_max[:, 1, 1].max()
    y_bot_min = pos_for_min[:, 1, 1].min()
    x_max = pos_for_max[..., 0].max()
    x_min = pos_for_min[..., 0].min()

    top_max = (0.5 - y_top_max) if y_top_max <= 0.5 else np.inf
    bot_max = (1.0 + eps - y_bot_max) if y_bot_max <= 1.0 + eps else np.inf
    dy_max = min(top_max, bot_max)
    dy_max = 0.0 if not np.isfinite(dy_max) else dy_max

    top_min = (-eps - y_top_min) if y_top_min >= -eps else -np.inf
    bot_min = (0.5 - y_bot_min) if y_bot_min >= 0.5 else -np.inf
    dy_min = max(top_min, bot_min)
    dy_min = 0.0 if not np.isfinite(dy_min) else dy_min

    dx_max = (1.0 + eps - x_max) if x_max <= 1.0 + eps else np.inf
    dx_min = (-eps - x_min) if x_min >= -eps else -np.inf
    dx_max = 0.0 if not np.isfinite(dx_max) else dx_max
    dx_min = 0.0 if not np.isfinite(dx_min) else dx_min

    return dy_min, dy_max, dx_min, dx_max


def estimate(
    pos: np.ndarray,
    shuttle: np.ndarray,
    cap_y: float = 0.05,
    cap_x: float = 0.10,
    eps: float = 0.15,
    p_roll: float = 0.2,
    n_trials: int = 50,
    seed: int = 0,
) -> dict:
    """Simulate the jitter on every clip and return summary statistics.

    :param pos: ``(n_clips, t, 2, 2)`` player positions in court coordinates.
    :param shuttle: ``(n_clips, t, 2)`` shuttle positions in camera coordinates.
    :return: dict of summary rates.
    """
    rng = np.random.default_rng(seed)
    n_clips = pos.shape[0]

    clip_oob_count = 0           # clips where any real shuttle frame went OOB across any trial
    trial_clip_oob_count = 0     # (clip, trial) pairs that fired and produced any OOB
    n_eff_total = 0              # total fired-and-non-degenerate (clip, trial) pairs
    n_total = 0                  # total (clip, trial) pairs simulated
    frame_oob_count = 0          # total real-shuttle frames pushed OOB across all (clip, trial) pairs
    frame_real_total = 0         # total real-shuttle frames * trials, denominator for per-frame rate

    for i in range(n_clips):
        dy_min, dy_max, dx_min, dx_max = _layered_bounds_for_clip(pos[i], eps=eps)

        dy_lo = max(dy_min, -cap_y)
        dy_hi = min(dy_max, cap_y)
        dx_lo = max(dx_min, -cap_x)
        dx_hi = min(dx_max, cap_x)

        dy_degenerate = dy_hi <= dy_lo
        dx_degenerate = dx_hi <= dx_lo
        if dy_degenerate and dx_degenerate:
            n_total += n_trials
            continue

        sh = shuttle[i]                                     # (t, 2)
        real_mask = ~(np.abs(sh) < 1e-9).all(axis=-1)       # (t,)
        n_real = int(real_mask.sum())
        if n_real == 0:
            n_total += n_trials
            continue

        rolls = rng.random(n_trials) < p_roll
        any_oob_for_clip = False
        for trial_no, roll in enumerate(rolls):
            n_total += 1
            if not roll:
                continue
            n_eff_total += 1
            dy = 0.0 if dy_degenerate else rng.uniform(dy_lo, dy_hi)
            dx = 0.0 if dx_degenerate else rng.uniform(dx_lo, dx_hi)
            shifted = sh + np.array([dx, dy])
            oob = (shifted < 0).any(axis=-1) | (shifted > 1).any(axis=-1)
            real_oob = oob & real_mask
            n_real_oob = int(real_oob.sum())
            frame_oob_count += n_real_oob
            frame_real_total += n_real
            if n_real_oob > 0:
                trial_clip_oob_count += 1
                any_oob_for_clip = True
        if any_oob_for_clip:
            clip_oob_count += 1

    return {
        'n_clips':              n_clips,
        'n_trials':             n_trials,
        'p_roll':               p_roll,
        'cap_y':                cap_y,
        'cap_x':                cap_x,
        'eps':                  eps,
        'fraction_clips_ever_oob': clip_oob_count / n_clips,
        'fraction_trials_oob_given_eff': (
            trial_clip_oob_count / n_eff_total if n_eff_total > 0 else 0.0
        ),
        'fraction_trials_oob_overall': trial_clip_oob_count / n_total,
        'fraction_real_shuttle_frames_oob': (
            frame_oob_count / frame_real_total if frame_real_total > 0 else 0.0
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collated-dir', required=True, type=Path)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--n-trials', type=int, default=50)
    p.add_argument('--cap-y', type=float, default=0.05)
    p.add_argument('--cap-x', type=float, default=0.10)
    p.add_argument('--eps', type=float, default=0.15)
    p.add_argument('--p-roll', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    split_dir = args.collated_dir / args.split
    pos = np.load(split_dir / 'pos.npy').astype(np.float32)
    shuttle = np.load(split_dir / 'shuttle.npy').astype(np.float32)

    print(f'Loaded {pos.shape[0]} clips from {split_dir}')
    print(f'  pos shape:     {pos.shape}')
    print(f'  shuttle shape: {shuttle.shape}')
    print()

    stats = estimate(
        pos=pos, shuttle=shuttle,
        cap_y=args.cap_y, cap_x=args.cap_x, eps=args.eps,
        p_roll=args.p_roll, n_trials=args.n_trials, seed=args.seed,
    )

    print('Configuration:')
    for k in ('p_roll', 'cap_y', 'cap_x', 'eps', 'n_trials'):
        print(f'  {k}: {stats[k]}')
    print()
    print('Results:')
    print(
        f"  Per real shuttle frame, how often the shift pushes it "
        f"off-screen: {stats['fraction_real_shuttle_frames_oob']*100:.2f}%"
    )
    print(
        f"  When the shift fires on a clip, how often it pushes any "
        f"shuttle frame off-screen: "
        f"{stats['fraction_trials_oob_given_eff']*100:.2f}%"
    )
    print(
        f"  Across all simulated shifts (rolled or not), how often any "
        f"shuttle frame ends up off-screen: "
        f"{stats['fraction_trials_oob_overall']*100:.2f}%"
    )
    print(
        f"  Across {args.n_trials} simulated shifts per clip, fraction "
        f"of clips that had at least one off-screen event: "
        f"{stats['fraction_clips_ever_oob']*100:.2f}%"
    )


if __name__ == '__main__':
    main()
