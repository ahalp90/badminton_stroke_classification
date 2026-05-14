# perception/ — model-agnostic perception infrastructure

Video I/O, player detection + tracking, court geometry, shuttle tracking,
and temporal sampling utilities. Shared by training (`bric.dataset`) and
inference (`bric.infer`); also called from the API layer to populate
per-stroke artefacts (frame thumbnails, court positions).

## Modules

| Module | Purpose |
|--------|---------|
| `video_io.py` | cv2 wrappers: `get_video_info`, `iter_frames`, `read_frame_at`, `read_frames`, `write_frame_thumbnail`. RGB-by-default; the only place cv2 should be imported directly. |
| `players.py` | YOLO11 + built-in tracker (ByteTrack) → `PlayerTrack` records with stable per-frame bboxes and `court_side` ('top' / 'bottom' / 'unknown'). Filters >2 detections by on-court frame count. |
| `temporal.py` | `clip_window_seconds`, `clip_window_frames`, `subsample_indices` — frame-window helpers. `subsample_indices` is what feeds R(2+1)D-18 (n=32 over a 2s window centred on `target_frame`, fps-invariant stride). |
| `shuttle.py` | Wrapper for TrackNetV3 → `(T, 3)` `[x_norm, y_norm, visibility]` per clip. |
| `_vendor/tracknetv3/` | TrackNetV3 upstream repository trimmed for inference only.

## Import rules

- `perception.*` **never** imports `bst_refactor.*`. The two architectures live separately.
- Imports `shared.*` for taxonomy and court constants.
- Imports TrackNetV3 from `perception._vendor.tracknetv3.*` (not from BST's vendored copy).

## Public API at a glance

```python
from perception.video_io import get_video_info, read_frame_at, write_frame_thumbnail
from perception.players import detect_and_track, PlayerTrack
from perception.temporal import subsample_indices, clip_window_seconds
```

## Where this fits

- Inference handler (`bric.infer`) calls `detect_and_track`, `subsample_indices`,
  `read_frames`, `write_frame_thumbnail` per stroke.
- Training dataset (`bric.dataset`) calls the same — uses cached `PlayerTrack`
  records from `runtime/cache/players/<clip_stem>.json` to avoid recomputing.
- API layer (`api.main`) is downstream — it never calls perception directly,
  only via the handler dispatcher.

## Related docs

- [`docs/api_contract.md`](../../docs/api_contract.md) — `court_position` and `stroke_frame_url` per-stroke fields are produced here.
- [`docs/storage.md`](../../docs/storage.md) — `strokes.frame_path` records the file `write_frame_thumbnail` produces.
