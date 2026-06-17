"""Compute median and std of frame lengths for the BST ShuttleSet dataset
using the between_2_hits_with_max_limits clipping strategy.

Replicates the clipping logic from gen_my_dataset.py using only the CSV annotations
(no video files or .npy data needed).
"""
import sys
from pathlib import Path

# Allow importing pipeline when invoked directly. validation_scripts/ is one
# level below src/bst_x/, so parents[1] resolves to the bst_x package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from pipeline.config import SPLITS, REMOVED_SHOTS


def estimate_fps(set_info_dir: Path, video_name: str, n_sets: int) -> int:
    """Estimate FPS from the time and frame_num columns in the set CSVs.
    Uses the last annotated shot (largest frame_num / time ratio) for accuracy."""
    for set_i in range(1, n_sets + 1):
        csv_path = set_info_dir / video_name / f'set{set_i}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[['time', 'frame_num']].dropna()
        if df.empty:
            continue
        # Use the last row for best precision (larger numbers reduce rounding error)
        last = df.iloc[-1]
        parts = str(last['time']).split(':')
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if seconds > 0:
            raw_fps = last['frame_num'] / seconds
            # Round to nearest standard FPS (24, 25, 30, 60)
            return min([24, 25, 30, 60], key=lambda x: abs(x - raw_fps))
    return 30  # fallback


def compute_clip_lengths_for_video(set_info_dir: Path, vid: int, video_name: str,
                                    n_sets: int, removed_shots: set) -> tuple[np.ndarray, int]:
    fps = estimate_fps(set_info_dir, video_name, n_sets)
    t = fps // 2
    limit = fps * 3 // 2
    eps = t // 2

    clip_lengths = []
    folder = set_info_dir / video_name

    for set_i in range(1, n_sets + 1):
        csv_path = folder / f'set{set_i}.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df = df[['rally', 'ball_round', 'frame_num']].dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['ball_round'] = df['ball_round'].astype(int)

        # Compute start_f / end_f from consecutive shots in each rally
        df['start_f'] = df['frame_num'].shift(1)
        df['end_f'] = df['frame_num'].shift(-1)
        df['start_f'] = df['start_f'].where(df.duplicated('rally', keep='first'), -1)
        df['end_f'] = df['end_f'].where(df.duplicated('rally', keep='last'), -1)

        for row in df.itertuples(index=False):
            # Skip individually removed shots
            if (vid, set_i, row.rally, row.ball_round) in removed_shots:
                continue

            frame_num = int(row.frame_num)
            s = int(row.start_f) if row.start_f != -1 else (frame_num - t)
            e = int(row.end_f) + eps if row.end_f != -1 else (frame_num + t)

            # Clamp to limits
            s = max(s, frame_num - limit)
            e = min(e, frame_num + limit + eps)

            clip_lengths.append(e - s)

    return np.array(clip_lengths, dtype=int), fps


def main():
    set_info_dir = Path('set')
    match_df = pd.read_csv(set_info_dir / 'match.csv')

    # Video splits and removed shots from centralised config
    vids_train = SPLITS['train']
    vids_val = SPLITS['val']
    vids_test = SPLITS['test']
    removed_shots = REMOVED_SHOTS

    vid_to_info = {}
    for _, row in match_df.iterrows():
        vid_to_info[int(row['id'])] = (row['video'], int(row['set']))

    splits = {'train': vids_train, 'val': vids_val, 'test': vids_test}
    all_lengths = []
    all_durations = []  # in seconds
    split_lengths = {}
    fps_map = {}  # vid -> estimated fps

    for split_name, vid_ids in splits.items():
        lengths = []
        durations = []
        for vid in vid_ids:
            if vid not in vid_to_info:
                continue
            video_name, n_sets = vid_to_info[vid]
            cl, fps = compute_clip_lengths_for_video(
                set_info_dir, vid, video_name, n_sets, removed_shots
            )
            fps_map[vid] = fps
            lengths.append(cl)
            durations.append(cl / fps)
        lengths = np.concatenate(lengths)
        durations = np.concatenate(durations)
        split_lengths[split_name] = (lengths, durations)
        all_lengths.append(lengths)
        all_durations.append(durations)

    all_lengths = np.concatenate(all_lengths)
    all_durations = np.concatenate(all_durations)

    # Show estimated FPS per video
    fps_groups = {}
    for vid, fps in sorted(fps_map.items()):
        fps_groups.setdefault(fps, []).append(vid)
    print("Estimated FPS per video:")
    for fps, vids in sorted(fps_groups.items()):
        print(f"  {fps} fps: video IDs {vids[0]}-{vids[-1]} ({len(vids)} videos)")

    print(f"\nTotal clips: {len(all_lengths)}")
    print(f"\nClip length (frames) - overall:")
    print(f"  Median: {np.median(all_lengths):.1f}")
    print(f"  Std:    {np.std(all_lengths):.1f}")
    print(f"  Mean:   {np.mean(all_lengths):.1f}")
    print(f"  Min:    {np.min(all_lengths)}")
    print(f"  Max:    {np.max(all_lengths)}")

    print(f"\nClip duration (seconds) - overall:")
    print(f"  Median: {np.median(all_durations):.2f}s")
    print(f"  Std:    {np.std(all_durations):.2f}s")
    print(f"  Mean:   {np.mean(all_durations):.2f}s")

    print(f"\nBy split:")
    for name, (lens, durs) in split_lengths.items():
        print(f"  {name:5s} ({len(lens):5d} clips): "
              f"median={np.median(lens):.1f} frames / {np.median(durs):.2f}s, "
              f"std={np.std(lens):.1f} frames / {np.std(durs):.2f}s")


if __name__ == '__main__':
    main()
