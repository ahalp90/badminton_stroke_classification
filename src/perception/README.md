# `perception/`

Architecture-agnostic perception primitives: video I/O, player detection
and tracking, shuttle tracking, and frame-window helpers. Consumed by
preprocessing scripts and inference handlers; no architecture-specific
code lives here.

## Modules

| Module | Exports |
|--------|---------|
| `video_io.py` | `VideoInfo`, `get_video_info`, `iter_frames`, `read_frame_at`, `read_frames`, `write_frame_thumbnail`. Thin cv2 wrapper; defaults to RGB ordering and exposes a `VideoInfo` dataclass for fps / frame-count / resolution. |
| `players.py`  | `PlayerTrack` dataclass, `detect_and_track(video_path, court_info=None, ...)`, `DEFAULT_YOLO_WEIGHTS`. Wraps Ultralytics YOLO11 with its built-in tracker (ByteTrack). When `court_info` is supplied, court-side is assigned by projecting bbox foot-centres through the homography and averaging the track's normalised court-y over its on-court frames; otherwise a vertical-pixel heuristic is used. Filters >2 detected persons (coaches, ball persons, audience) by keeping the 2 tracks with the most on-court frames. |
| `temporal.py` | `clip_window_seconds`, `clip_window_frames`, `subsample_indices`. Frame-window helpers. `subsample_indices` returns N indices uniformly subsampled from a coverage-second window centred on a target frame, with stride that adapts to source fps so real-world coverage stays constant across videos of different frame rates. |
| `shuttle.py`  | `extract_shuttle(video_path, save_dir, weights_dir=...)`. Subprocess wrapper around the vendored TrackNetV3 `predict.py`; returns the path to the per-frame CSV (`Frame, Visibility, X, Y`). |
| `_vendor/tracknetv3/` | Vendored TrackNetV3 source. Used as a subprocess from its own directory because its modules use top-level absolute imports that assume the vendor dir is on `sys.path`. |

`PlayerTrack`'s `track_id` is stable within a single video but can swap
when players cross paths (e.g. at the net). Downstream code that needs
"which player did this" should rely on `court_side`, which is
re-derived from court position each frame, rather than `track_id`.

## Imports

- No imports from `bst_refactor.*`.
- `players.py` imports `shared.court` for homography projection.
- `shuttle.py` invokes the vendored TrackNetV3 via subprocess; it does
  not import from `_vendor/tracknetv3/`.

## Layout assumptions

- YOLO11 weights at `runtime/checkpoints/yolo11/yolo11n.pt`. Auto-downloaded
  by Ultralytics on first call to `detect_and_track`.
- TrackNetV3 weights at `runtime/checkpoints/tracknetv3/{TrackNet_best.pt, InpaintNet_best.pt}`.
  Pulled per the upstream release; not auto-downloaded.

## Consumers

- `scripts/bric/preprocess_videos.py` reads `DEFAULT_YOLO_WEIGHTS` and
  drives YOLO directly with its own per-rally loop.
- `scripts/bric/extract_shuttle.py` calls `extract_shuttle` per rally
  clip and `get_video_info` to size the per-source-video shuttle cache.
- `tests/` exercises `subsample_indices` and `video_io`.

`bric.dataset` does not import this package directly — it reads the
NPZ caches that the preprocessing scripts produce.
