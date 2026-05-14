"""Save contact-sheet JPGs for spot-checking the RGB cache.

For each stroke, writes a 5-frame horizontal strip (window-start,
quarter, target, three-quarter, window-end) to
runtime/cache/validation/<clip_stem>.jpg. Tells you the stroke
context (vid/set/rally/ball_round/type/player_side) so you know
what each preview is supposed to show.

Usage:
    # 5 random strokes from any vid that has cache files
    uv run python scratch/validate_rgb.py

    # 5 random strokes from a specific vid
    uv run python scratch/validate_rgb.py --vid 1

    # specific stroke(s) by clip_stem
    uv run python scratch/validate_rgb.py --stem 1_1_3_1 1_1_5_4

    # more strokes
    uv run python scratch/validate_rgb.py --n 20
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / 'runtime' / 'cache' / 'rgb'
SHOTS = REPO / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
OUT = REPO / 'runtime' / 'cache' / 'validation'

# Frames to include in the contact strip.
FRAME_INDICES = [0, 8, 16, 24, 31]


def load_shots() -> pd.DataFrame:
    if not SHOTS.exists():
        sys.exit(f'shots_master.csv not found at {SHOTS} — run build_shots_master first')
    return pd.read_csv(SHOTS)


def pick_stems(args, master: pd.DataFrame) -> list[str]:
    if args.stem:
        return list(args.stem)
    pool = master
    if args.vid is not None:
        pool = pool[pool['vid'] == args.vid]
    cached = [s for s in pool['clip_stem'].tolist() if (CACHE / f'{s}.npy').exists()]
    if not cached:
        sys.exit(f'no cached RGB tensors found for the requested filter')
    return random.sample(cached, min(args.n, len(cached)))


def annotate(strip: np.ndarray, label: str) -> np.ndarray:
    """Burn the stroke label onto the bottom-left of the strip."""
    h, w = strip.shape[:2]
    canvas = np.zeros((h + 30, w, 3), dtype=np.uint8)
    canvas[:h] = strip
    cv2.putText(
        canvas, label, (5, h + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', type=int, help='Restrict random pick to this vid.')
    parser.add_argument('--n', type=int, default=5, help='Number of strokes to sample.')
    parser.add_argument('--stem', nargs='*', help='Specific clip_stem(s) to inspect.')
    args = parser.parse_args()

    master = load_shots()
    stems = pick_stems(args, master)
    OUT.mkdir(parents=True, exist_ok=True)

    info = master.set_index('clip_stem')
    print(f'Writing {len(stems)} previews to {OUT}/')
    for stem in stems:
        cache_path = CACHE / f'{stem}.npy'
        if not cache_path.exists():
            print(f'  {stem}: no cache file, skipping')
            continue
        try:
            row = info.loc[stem]
        except KeyError:
            label = stem
        else:
            label = (
                f'{stem}  set={row["set_id"]} rally={row["rally"]} br={row["ball_round"]}  '
                f'{row["raw_type_en"]}  player={row["player_side"]}'
            )

        t = np.load(cache_path)
        strip = np.concatenate([t[i] for i in FRAME_INDICES], axis=1)
        canvas = annotate(strip, label)
        out_path = OUT / f'{stem}.jpg'
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f'  {out_path.relative_to(REPO)}  {label}')


if __name__ == '__main__':
    main()
