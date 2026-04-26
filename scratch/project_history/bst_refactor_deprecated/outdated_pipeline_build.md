** outdated_pipeline_build.md — development notes from when the pipeline was being built. All content is now covered by pipeline/README.md**

# Pipeline Build Summary

Summary of changes made to create the `pipeline/` package, which automates the ShuttleSet data pipeline that previously required 6+ manual script runs, hand-editing of config values, and manual file operations.

## Problems Solved

1. **6 manual runs replaced by 1 command.** The original workflow required running `gen_my_dataset.py` six times (once per split-player combination: train/val/test x Top/Bottom), manually adjusting variables each time. `pipeline/clip_generator.py` handles all splits and both players in a single invocation.

2. **Duplicated configuration centralised.** Splits, stroke types, and removed shots were hardcoded in 3+ files (`gen_my_dataset.py`, `compute_clip_length_stats.py`, `shuttleset_dataset.py`). All now import from `pipeline/config.py`.

3. **Duplicated player-mapping logic extracted.** The A/B-to-Top/Bottom XOR logic (with set 3 court-switch handling) was copy-pasted in `gen_my_dataset.py`, `get_each_class_total.py`, and implicitly elsewhere. Now lives once in `pipeline/player_mapping.py`.

4. **English names throughout.** The original code used Chinese stroke type names for folder names, logs, and labels. The pipeline uses English everywhere, with Chinese appearing only at the CSV annotation I/O boundary via `EN_TO_ZH`/`ZH_TO_EN` dicts in `config.py`.

5. **Manual flaw filtering automated.** The 4 excluded videos and 5 individually removed shots (from `flaw_shot_records.csv`) are parsed at import time and applied automatically during clip generation.

6. **Manual class merging automated.** The 19-to-12 stroke type merge (7 rare subtypes folded into parents) was previously done by manually moving folders. `apply_class_merge()` handles it programmatically.

7. **Video download automated.** `download_videos.py` wraps yt-dlp for parallel YouTube download of all 40 match videos and auto-generates the resolution CSV.

## Files Created

### `pipeline/__init__.py`
Empty package init.

### `pipeline/config.py`
Single source of truth. Key contents:
- All paths anchored to `PROJECT_ROOT = Path(__file__).resolve().parent.parent` (works regardless of cwd)
- `EN_TO_ZH` / `ZH_TO_EN` -- bidirectional stroke name translation dicts (19 entries)
- `STROKE_TYPES_19`, `STROKE_TYPES_19_ZH` -- full raw type lists in both languages
- `STROKE_TYPES_12_MERGED` -- the 12 post-merge types
- `STROKE_TYPES_17_RAW` -- the 17 types that get Top_/Bottom_ prefixes in the 35-class system (excludes `unknown` and `driven_flight`)
- `MERGE_MAP` -- maps 6 rare subtypes to their parents (e.g. `'wrist_smash': 'smash'`)
- `SPLITS` -- train (30 videos), val (5), test (5)
- `EXCLUDED_VIDEOS` = {9, 10, 12, 27}
- `REMOVED_SHOTS` -- parsed from `flaw_shot_records.csv` at import time, with assertion that parsed exclusions match `EXCLUDED_VIDEOS`
- `parse_flaw_records()` -- reads the CSV, returns (excluded_video_ids, removed_shot_tuples)
- `get_stroke_types(side, merged)` -- builds label lists with Top_/Bottom_ prefixes; follows BST convention (unknown first in 25-class, last in 35-class)

### `pipeline/player_mapping.py`
Extracted and deduplicated logic:
- `map_players(df, first_A_is_top, set_num)` -- replaces A/B with Top/Bottom using XOR logic
- `find_set3_switch_rally(df)` -- finds the iloc index where set 3 court-switch occurs at 11 points
- `collect_shots(set_info_dir, v_info, stroke_types_zh)` -- collects all shots for BOTH players across all sets, handling set 3 split. Translates types to English before returning. Unlike the original `collect_shot_types_pos()` which filtered to one player.

### `pipeline/clip_generator.py`
Automated clip generation + flaw filtering + class merging:
- `compute_temporal_bounds(folder_path, shots_df)` -- adds start_f/end_f columns based on adjacent shots (adapted from `gen_my_dataset.py set_between_2_hits_from_pos()`)
- `_frame_to_time(frame_number, fps)` -- frame-to-HH:MM:SS conversion (from `ShuttleSet/utils.py`)
- `_compute_clip_bounds(row, clip_window, fps)` -- returns (start_f, end_f) for any of the 3 clip windows, extracted to keep the writer loop flat
- `_filter_removed_shots(shots_df, vid, removed_shots)` -- vectorized removed-shot filter using `pd.MultiIndex.isin()`
- `_write_clips_for_video(...)` -- writes clips for one source video using MoviePy
- `generate_all_clips(...)` -- outer loop over all splits and videos; filters removed shots, computes temporal bounds, writes clips
- `apply_class_merge(output_dir, merge_map)` -- moves clips from rare subtype folders into parent folders, removes empty source dirs

### `pipeline/download_videos.py`
YouTube download automation:
- `download_video(url, video_id, video_name, output_dir)` -- yt-dlp subprocess per video, skips existing
- `download_all_videos(...)` -- parallel download via ThreadPoolExecutor
- `build_resolution_csv(video_dir, output_path)` -- scans downloads with OpenCV, writes `my_raw_video_resolution.csv`

### `pipeline/shuttle_extractor.py`
TrackNetV3 wrapper (shared by both architectures):
- `normalize_shuttlecock(arr, v_width, v_height)` -- normalize xy by resolution to [0,1]
- `extract_shuttle_trajectory(clip_path, tracknet_dir, output_csv_dir)` -- runs TrackNetV3 predict.py subprocess on one clip
- `extract_all_shuttles(...)` -- parallel extraction via ProcessPoolExecutor
- `shuttle_csvs_to_npy(...)` -- converts TrackNetV3 CSVs to normalised (t, 2) numpy arrays, mirroring clip directory structure

### `pipeline/court_utils.py`
Optional homography-based court projection utilities (copied from `prepare_train_on_shuttleset.py`):
- `get_H`, `get_corner_camera`, `convert_homogeneous`, `scale_pos_by_resolution`
- `project` -- applies homography transform
- `get_court_info` -- returns H matrix + court boundaries for a video
- `to_court_coordinate` -- camera pixels to court coords
- `normalize_position` -- court coords to [0,1]
- `check_pos_in_court` -- checks if detected people are on-court
- `load_all_court_info` -- loads court info for all 44 videos from homography.csv

### `pipeline/verify.py`
Post-generation sanity checks:
- `verify_splits_present(clips_dir)` -- checks train/val/test dirs exist with clips
- `verify_no_excluded(clips_dir)` -- confirms no clips from videos 9/10/12/27
- `verify_no_removed_shots(clips_dir)` -- confirms individually removed shots absent
- `verify_class_merge(clips_dir, merge_map)` -- confirms source folders empty after merge
- `print_dataset_summary(clips_dir)` -- prints clip counts per split per class

### `pipeline/build_dataset.py`
One-command orchestrator:
1. Download videos (optional, `--skip-download`)
2. Build resolution CSV
3. Generate labeled clips
4. Apply class merge (optional, `--no-merge`)
5. Verify clips
6. Extract shuttle trajectories (optional, requires `--tracknet-dir`)

All paths come from `config.py`. CLI only exposes runtime flags.

## Files Modified

### `ShuttleSet/compute_clip_length_stats.py`
- Added `sys.path.insert` for cross-directory imports
- Replaced hardcoded `vids_train/val/test` lists and `removed_shots` set with imports from `pipeline.config` (~5 lines changed)
- Verified output unchanged: 33,481 total clips

### `ShuttleSet/get_each_class_total.py`
- Added `sys.path.insert` for cross-directory imports
- Replaced inline XOR player mapping with imports from `pipeline.player_mapping` (`map_players`, `find_set3_switch_rally`)
- ~20 lines changed. Verified: Video 1 still produces 1,644 shots

### `stroke_classification/preparing_data/shuttleset_dataset.py`
- `get_stroke_types()` and `get_merged_stroke_types()` now import base stroke lists from `pipeline.config` via `_en_to_zh_list()` helper
- Still returns Chinese names (BST training code depends on them)
- Source lists come from config instead of being hardcoded
- Verified: merged=25 classes, raw=35 classes, padded versions all correct

## Files NOT Modified

- `ShuttleSet/gen_my_dataset.py` -- left as-is (standalone use still works)
- `ShuttleSet/utils.py` -- unchanged
- `stroke_classification/preparing_data/prepare_train_on_shuttleset.py` -- unchanged
- `stroke_classification/model/` -- untouched
- `stroke_classification/main_on_shuttleset/` -- untouched

## Design Decisions

**Mutable defaults.** All functions that accept mutable containers (dicts, sets) use `None` sentinel with assignment inside the function body, avoiding the Python mutable default argument bug.

**Flat clip writer.** `_compute_clip_bounds()` was extracted so that `_write_clips_for_video()` has a single flat loop instead of deeply nested match/case blocks per clip window.

**`v_info` stays as `pd.Series`.** `match_df.loc[vid]` returns a Series natively; converting to a plain dict would add boilerplate with no benefit since `.loc` already provides named field access.

**`build_dataset.py` uses a plain function, not argparse for paths.** All paths come from `config.py`. The CLI only exposes runtime flags (`--skip-download`, `--skip-shuttle`, `--no-merge`, `--tracknet-dir`, `--workers`).

**Court utils copied, not moved.** BST's `prepare_train_on_shuttleset.py` continues to use its own copy. `pipeline/court_utils.py` exists so the other architecture can import court utilities without pulling in the full BST preparation pipeline.

## Stroke Type Reference

### 12 merged types (25-class system)

| English              | Chinese  | Notes                                      |
|----------------------|----------|--------------------------------------------|
| net_shot             | 放小球   |                                            |
| return_net           | 擋小球   |                                            |
| smash                | 殺球     | Absorbs wrist_smash                        |
| lob                  | 挑球     | Absorbs defensive_return_lob               |
| clear                | 長球     |                                            |
| drive                | 平球     | Absorbs back_court_drive, defensive_return_drive |
| drop                 | 切球     | Absorbs passive_drop                       |
| push                 | 推球     |                                            |
| rush                 | 撲球     |                                            |
| cross_court_net_shot | 勾球     |                                            |
| short_service        | 發短球   |                                            |
| long_service         | 發長球   |                                            |

### 6 rare subtypes merged into parents

| English                | Chinese    | Merges into |
|------------------------|------------|-------------|
| wrist_smash            | 點扣       | smash       |
| defensive_return_lob   | 防守回挑   | lob         |
| driven_flight          | 小平球     | unknown     |
| back_court_drive       | 後場抽平球 | drive       |
| passive_drop           | 過渡切球   | drop        |
| defensive_return_drive | 防守回抽   | drive       |

## Output Structure

```
ShuttleSet/clips/
  train/{Top,Bottom}_{stroke_type}/*.mp4
  val/{Top,Bottom}_{stroke_type}/*.mp4
  test/{Top,Bottom}_{stroke_type}/*.mp4

ShuttleSet/shuttle_npy/
  train/{Top,Bottom}_{stroke_type}/*.npy
  val/{Top,Bottom}_{stroke_type}/*.npy
  test/{Top,Bottom}_{stroke_type}/*.npy
```

Clip filenames: `{video_id}_{set}_{rally}_{ball_round}.mp4`
