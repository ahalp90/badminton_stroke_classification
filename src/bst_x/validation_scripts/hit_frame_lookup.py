"""Map clip stems to the 0-based frame index of the hit within the clip.

Re-derives clip boundaries from the ShuttleSet set CSVs using the same
``between_2_hits_with_max_limits`` windowing logic as the clip generator,
without needing the actual video files.

Adapted from:
  - pipeline/clip_generator.py:_compute_clip_bounds()  (canonical)
  - ShuttleSet/compute_clip_length_stats.py             (CSV-only version)

Usage as a library::

    from hit_frame_lookup import build_hit_frame_lookup
    lookup = build_hit_frame_lookup(
        Path("ShuttleSet/set"),
        Path("ShuttleSet/video_metadata.csv"),
    )
    # lookup["35_1_10_17"] == 23  (hit is at frame index 23 in the clip)
"""
from pathlib import Path

import pandas as pd


def build_hit_frame_lookup(
    set_dir: Path,
    video_metadata_csv: Path,
) -> dict[str, int]:
    """Map clip stems to the 0-based frame index of the hit within the clip.

    Replicates the ``between_2_hits_with_max_limits`` windowing logic from
    the clip generator using only CSV annotations (no video files).

    FPS is read from ``video_metadata.csv`` (the same source of truth as
    the clip generator), not estimated from annotations.

    :param set_dir: Path to ShuttleSet/set/ containing match.csv and
                    per-match folders with set*.csv files.
    :param video_metadata_csv: Path to ShuttleSet/video_metadata.csv with
                               id and fps columns.
    :return: Dict mapping clip stem (e.g. "35_1_10_17") to the frame index
             of the hit within that clip.
    """
    match_df = pd.read_csv(set_dir / "match.csv")
    # video_id -> (folder_name, number_of_sets)
    id_to_info = {
        int(row["id"]): (row["video"], int(row["set"]))
        for _, row in match_df.iterrows()
    }

    # Actual FPS per video from the metadata CSV.
    meta_df = pd.read_csv(video_metadata_csv)
    meta_df = meta_df.dropna(subset=["fps"])  # excluded videos have no FPS
    id_to_fps = dict(zip(meta_df["id"].astype(int), meta_df["fps"].astype(int)))

    lookup: dict[str, int] = {}

    for vid_id, (folder, n_sets) in id_to_info.items():
        fps = id_to_fps.get(vid_id)
        if fps is None:
            continue  # excluded video, no clips to look up
        t = fps // 2            # 0.5 sec in frames
        limit = fps * 3 // 2    # 1.5 sec in frames

        for set_i in range(1, n_sets + 1):
            csv_path = set_dir / folder / f"set{set_i}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df = df[["rally", "ball_round", "frame_num"]].dropna(
                subset=["frame_num"]
            )
            df["frame_num"] = df["frame_num"].astype(int)
            df["ball_round"] = df["ball_round"].astype(int)

            # Vectorized: previous/next shot frames within each rally.
            # -1 flags "no adjacent shot" (first/last in rally).
            df["start_f"] = df["frame_num"].shift(1)
            df["start_f"] = df["start_f"].where(
                df.duplicated("rally", keep="first"), -1
            )
            # end_f is unused here (only start_f is needed for hit index),
            # but kept for parity with clip_generator.py:_compute_clip_bounds().
            df["end_f"] = df["frame_num"].shift(-1)
            df["end_f"] = df["end_f"].where(
                df.duplicated("rally", keep="last"), -1
            )

            # Per-row: compute clip start and hit index.
            # Conditional branches (first/last shot fallbacks) don't vectorize
            # cleanly, and ~36k rows total runs in under a second.
            for row in df.itertuples(index=False):
                frame_num = int(row.frame_num)

                # Clip start: previous shot frame or fallback to (hit - 0.5s)
                s = int(row.start_f) if row.start_f != -1 else (frame_num - t)
                # Clamp to limit
                s = max(s, frame_num - limit)

                stem = f"{vid_id}_{set_i}_{row.rally}_{row.ball_round}"
                lookup[stem] = frame_num - s

    return lookup
