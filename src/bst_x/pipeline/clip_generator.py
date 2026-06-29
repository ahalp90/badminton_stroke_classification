"""Automated clip generation from raw ShuttleSet match videos.

Replaces the 6 manual runs of gen_my_dataset.py with a single script that:
  1. Generates clips for all splits, both players, all stroke types
  2. Filters out individually removed shots (from flaw_shot_records.csv)
  3. Applies class merging per the active taxonomy with English folder names

Usage:
    python -m pipeline.clip_generator
"""
import argparse
import shutil

from moviepy import VideoFileClip
import pandas as pd
from pathlib import Path

from pipeline.config import (
    SET_INFO_DIR, RAW_VIDEO_DIR, CLIPS_OUTPUT_DIR,
    SPLITS, STROKE_TYPES_19, STROKE_TYPES_19_ZH,
    REMOVED_SHOTS, CLIP_WINDOW, PLAYERS,
    UNPREFIXED_TYPES, Taxonomy, resolve_taxonomy,
)

from pipeline.player_mapping import collect_shots

# Default taxonomy when callers don't pass one. Matches the project's
# working baseline; override via the function arg for one-off runs.
_DEFAULT_TAXONOMY = resolve_taxonomy('une_v1_14')

# Single source for the runtime guard and the CLI --clip-window choices.
VALID_CLIP_WINDOWS = frozenset(
    {'middle_in_a_sec', 'between_2_hits', 'between_2_hits_with_max_limits'}
)


# ---------------------------------------------------------------------------
# Temporal boundary computation (adapted from gen_my_dataset.py:78-106)
# ---------------------------------------------------------------------------
def compute_temporal_bounds(folder_path: Path, shots_df: pd.DataFrame) -> pd.DataFrame:
    """Add start_f and end_f columns to shots_df based on adjacent shots.

    For each shot, the start frame is the previous shot's frame in the same
    rally, and the end frame is the next shot's frame. First/last shots in a
    rally get -1 (handled by the clip window as fallback).

    Adapted from gen_my_dataset.py set_between_2_hits_from_pos().

    :param folder_path: Path to the match folder containing set CSVs.
    :param shots_df: DataFrame with 'set', 'rally', 'ball_round', 'frame_num' columns.
    :return: DataFrame with start_f and end_f columns added.
    """
    parts = []
    for set_i, group_idx in shots_df.groupby('set').groups.items():
        df = pd.read_csv(folder_path / f'set{set_i}.csv')
        df = df[['rally', 'ball_round', 'frame_num']]

        # Use a shift to find adjacent frames.
        # We look at the previous shot, but if this is the first shot of a rally,
        # there is no 'previous', so we fallback to -1.
        df['start_f'] = df['frame_num'].shift(1)
        df['start_f'] = df['start_f'].where(df.duplicated('rally', keep='first'), -1)
        # Similarly, look at the next shot, but fallback to -1 if it's the last shot of the rally.
        df['end_f'] = df['frame_num'].shift(-1)
        df['end_f'] = df['end_f'].where(df.duplicated('rally', keep='last'), -1)

        merged = pd.merge(
            shots_df.loc[group_idx].reset_index(drop=True),
            df,
            on=['rally', 'ball_round', 'frame_num'],
        )
        merged = merged[[
            'set', 'rally', 'ball_round',
            'start_f', 'frame_num', 'end_f',
            'roundscore_A', 'roundscore_B', 'player', 'type',
        ]]
        parts.append(merged)

    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Frame-to-time conversion (from ShuttleSet/utils.py:35-56)
# ---------------------------------------------------------------------------
def _frame_to_time(frame_number: int, fps: float) -> str:
    """Convert a frame number to HH:MM:SS.ssssss time string for MoviePy.

    :param frame_number: Frame index in the video.
    :param fps: Video frames per second.
    :return: Time string formatted as HH:MM:SS.ssssss.
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60 + 0.5 / fps
    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


# ---------------------------------------------------------------------------
# Clip boundary computation per clip window
# ---------------------------------------------------------------------------
def _compute_clip_bounds(row, clip_window: str, fps: float) -> tuple[int, int]:
    """Compute start and end frame for one clip based on the clip window.

    :param row: A Series (from iterrows) with keys frame_num, start_f, end_f.
    :param clip_window: One of 'middle_in_a_sec', 'between_2_hits',
        'between_2_hits_with_max_limits'.
    :param fps: Video frames per second.
    :return: (start_frame, end_frame) as ints.
    """
    t = int(fps) // 2       # frames in 0.5 sec
    frame_num = int(row['frame_num'])

    if clip_window == 'middle_in_a_sec':
        # Fixed 1-second window centred on the shot frame
        return frame_num - t, frame_num + t

    # --- between_2_hits and between_2_hits_with_max_limits ---
    # Use adjacent shot frames if they exist, otherwise fall back to ±0.5 sec
    eps = t // 2  # frames in 0.25 sec (small extension past the next hit)
    start_f = int(row['start_f']) if row['start_f'] != -1 else (frame_num - t)
    end_f = int(row['end_f']) + eps if row['end_f'] != -1 else (frame_num + t)

    if clip_window == 'between_2_hits_with_max_limits':
        # Clamp so clip never exceeds 1.5 sec each side of the shot
        limit = int(fps) * 3 // 2  # frames in 1.5 sec
        start_f = max(start_f, frame_num - limit)
        end_f = min(end_f, frame_num + limit + eps)

    return start_f, end_f


# ---------------------------------------------------------------------------
# Single-video clip writer
# ---------------------------------------------------------------------------
def _write_clips_for_video(
    raw_video_dir: Path,
    out_folder: Path,
    video_id: int,
    shots_df: pd.DataFrame,
    clip_window: str,
    stroke_types: list[str],
    players: tuple[str, ...],
) -> int:
    """Write clip .mp4 files for one source video. Returns count of clips written.

    :param raw_video_dir: Directory containing raw match videos.
    :param out_folder: Split-level output directory (e.g. clips/train/).
    :param video_id: Numeric ID of the video (from match.csv).
    :param shots_df: DataFrame with columns: player, type, set, rally, ball_round,
        frame_num, start_f, end_f. Each row is one shot to clip.
    :param clip_window: Temporal clipping window name.
    :param stroke_types: List of English stroke type names (for folder creation).
    :param players: Tuple of player names ('Top', 'Bottom').
    :return: Number of clips written.
    """
    if clip_window not in VALID_CLIP_WINDOWS:
        raise ValueError(f"Unknown clip window: {clip_window!r}")

    for typ in stroke_types:
        if typ in UNPREFIXED_TYPES:
            (out_folder / typ).mkdir(parents=True, exist_ok=True)
        else:
            for player in players:
                (out_folder / f'{player}_{typ}').mkdir(parents=True, exist_ok=True)

    # Open the source video — bare next() on an empty glob raises StopIteration
    video_matches = list(raw_video_dir.glob(f"{video_id} *"))
    if not video_matches:
        print(f"Warning: Raw video for ID {video_id} not found. Skipping.")
        return 0
    video_path = str(video_matches[0])
    video = VideoFileClip(video_path)
    fps = video.fps
    clips_written = 0

    try:
        for _, row in shots_df.iterrows():
            typ = row['type']
            folder_name = typ if typ in UNPREFIXED_TYPES else f'{row["player"]}_{typ}'
            out_path = (out_folder
                        / folder_name
                        / f'{video_id}_{row["set"]}_{row["rally"]}_{int(row["ball_round"])}.mp4')
            if out_path.exists():
                continue

            start_f, end_f = _compute_clip_bounds(row, clip_window, fps)

            clip = video.subclipped(
                _frame_to_time(start_f, fps),
                _frame_to_time(end_f, fps),
            )
            clip.write_videofile(str(out_path), logger=None)
            clips_written += 1
    finally:
        video.close()

    return clips_written


# ---------------------------------------------------------------------------
# Removed-shot filtering
# ---------------------------------------------------------------------------
def _filter_removed_shots(
    shots_df: pd.DataFrame,
    vid: int,
    removed_shots: set[tuple[int, int, int, int]],
) -> pd.DataFrame:
    """Drop rows matching entries in removed_shots.

    Creates a composite string key per row ("set_rally_ballround") and checks
    membership against the same key format built from removed_shots.

    :param shots_df: DataFrame with 'set', 'rally', 'ball_round' columns.
    :param vid: Video ID (first element of each removed_shots tuple).
    :param removed_shots: Set of (video_id, set, rally, ball_round) tuples.
    :return: Filtered DataFrame.
    """
    # Keep only the removals for this video
    to_remove = {
        f'{s}_{r}_{b}' for v, s, r, b in removed_shots if v == vid
    }
    if not to_remove:
        return shots_df

    # Build one string key per row, then check set membership
    row_keys = (shots_df['set'].astype(int).astype(str)
                + '_' + shots_df['rally'].astype(int).astype(str)
                + '_' + shots_df['ball_round'].astype(int).astype(str))
    return shots_df[~row_keys.isin(to_remove)]


# ---------------------------------------------------------------------------
# Main pipeline functions
# ---------------------------------------------------------------------------
def generate_all_clips(
    raw_video_dir: Path = RAW_VIDEO_DIR,
    set_info_dir: Path = SET_INFO_DIR,
    output_dir: Path = CLIPS_OUTPUT_DIR,
    splits: dict[str, list[int]] | None = None,
    stroke_types_zh: list[str] | None = None,
    removed_shots: set[tuple[int, int, int, int]] | None = None,
    clip_window: str = CLIP_WINDOW,
) -> None:
    """Generate labeled clip .mp4s for all splits, both players, all videos.

    :param raw_video_dir: Directory containing downloaded match videos.
    :param set_info_dir: Directory containing match.csv and per-match set CSVs.
    :param output_dir: Root output directory for clips (split subdirs created inside).
    :param splits: {split_name: [video_ids]}. Defaults to config.SPLITS.
    :param stroke_types_zh: Chinese stroke type names for CSV matching. Defaults to config.
    :param removed_shots: Set of (match, set, rally, ball_round) to exclude. Defaults to config.
    :param clip_window: Temporal clipping window name.
    """
    if splits is None:
        splits = SPLITS
    if stroke_types_zh is None:
        stroke_types_zh = STROKE_TYPES_19_ZH
    if removed_shots is None:
        removed_shots = REMOVED_SHOTS

    # Load match metadata.
    # Each row is a pd.Series with fields: 'video' (str), 'downcourt' (bool).
    # The index (accessed via .name) is the integer video ID.
    match_df = pd.read_csv(set_info_dir / 'match.csv')[['id', 'video', 'downcourt']]
    match_df['downcourt'] = match_df['downcourt'].astype(bool)
    match_df = match_df.set_index('id')

    total_clips = 0
    for split_name, vid_ids in splits.items():
        print(f'\n=== Split: {split_name} ({len(vid_ids)} videos) ===')
        out_folder = output_dir / split_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for vid in vid_ids:
            if vid not in match_df.index:
                continue
            v_info = match_df.loc[vid]

            # Collect all shots (both players, English types)
            shots_df = collect_shots(set_info_dir, v_info, stroke_types_zh)
            if shots_df.empty:
                continue

            # Filter out individually removed shots (vectorized)
            before = len(shots_df)
            shots_df = _filter_removed_shots(shots_df, vid, removed_shots)
            removed_count = before - len(shots_df)

            # Add temporal boundaries (start_f, end_f)
            folder_path = set_info_dir / v_info['video']
            shots_df = compute_temporal_bounds(folder_path, shots_df)

            # Write clips with English folder names
            n = _write_clips_for_video(
                raw_video_dir, out_folder,
                video_id=vid,
                shots_df=shots_df,
                clip_window=clip_window,
                stroke_types=STROKE_TYPES_19,
                players=PLAYERS,
            )
            total_clips += n
            status = f'video {vid:2d}: {n} new clips'
            if removed_count:
                status += f' ({removed_count} removed shots filtered)'
            print(f'  {status}')

    print(f'\nTotal new clips written: {total_clips}')


def apply_class_merge(
    output_dir: Path = CLIPS_OUTPUT_DIR,
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY,
) -> None:
    """Merge rare subtype folders into their parent type folders.

    For example, Top_wrist_smash/*.mp4 -> Top_smash/*.mp4.
    Source folders are removed after merging.

    :param output_dir: Root clips directory containing split subdirs.
    :param taxonomy: Taxonomy whose merge_map defines which subtypes to merge.
    """
    if taxonomy.merge_map is None:
        print('Taxonomy has no merge_map — nothing to merge.')
        return

    split_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
    move_ops = []
    for split_dir in split_dirs:
        for src_type, dst_type in taxonomy.merge_map.items():
            if src_type in UNPREFIXED_TYPES or dst_type in UNPREFIXED_TYPES:
                src = split_dir / src_type
                dst = split_dir / dst_type
                if src.exists():
                    move_ops.append((src, dst))
            else:
                for player in PLAYERS:
                    src = split_dir / f'{player}_{src_type}'
                    dst = split_dir / f'{player}_{dst_type}'
                    if src.exists():
                        move_ops.append((src, dst))

    # Execute each move in a flat loop
    moved = 0
    for src, dst in move_ops:
        dst.mkdir(parents=True, exist_ok=True)
        for clip_file in src.iterdir():
            shutil.move(str(clip_file), str(dst / clip_file.name))
            moved += 1
        try:
            src.rmdir()  # may fail if stray non-clip files remain (e.g. .DS_Store)
        except OSError:
            pass

    print(f'Class merge complete: {moved} clips moved.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate labeled ShuttleSet stroke clips from raw match videos.',
    )
    parser.add_argument('--clip-window', default=CLIP_WINDOW,
                        choices=VALID_CLIP_WINDOWS,
                        help='Temporal clipping window')
    parser.add_argument('--no-merge', action='store_true',
                        help='Skip class merging (keep all 19 types)')
    args = parser.parse_args()

    print('=== Generating clips ===')
    generate_all_clips(clip_window=args.clip_window)

    if not args.no_merge:
        print('\n=== Applying class merge ===')
        apply_class_merge()
