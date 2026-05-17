# shared/ — values + utilities BRIC consumes

A flat namespace for things BRIC needs that are architecture-agnostic.
Some originated inside BST (taxonomy, court) and were decoupled here
so BRIC doesn't import from `bst_refactor.*` directly; others are
generic helpers (video I/O, frame-window math) that any architecture
working with video would want.

## Modules

| Module | Origin | Purpose |
|--------|--------|---------|
| `taxonomy.py` | mirrors `bst_refactor.pipeline.config` | The 14-class stroke taxonomy + class-merge rules. Single source of truth for `stroke_type` strings emitted by the contract. |
| `court.py` | mirrors `bst_refactor.pipeline.court_utils` | Homography utilities (`project`, `convert_homogeneous`, `get_court_info`) + reference court constants (`REF_COURT_M`, `REF_COURT_CORNERS_M`). |
| `player_mapping.py` | mirrors `bst_refactor.pipeline.player_mapping` | Heuristic player identification: maps top/bottom court-side to which named player it actually is, including set-3 mid-game switch handling. |
| `dataset.py` | new | Canonical paths to ShuttleSet assets (annotations, video metadata, homography CSV). |
| `video_io.py` | new | cv2 wrapper: `get_video_info`, `iter_frames`, `read_frame_at`, `read_frames`, `write_frame_thumbnail`. Defaults to RGB ordering; exposes a `VideoInfo` dataclass. |
| `temporal.py` | new | Frame-window helpers: `clip_window_seconds`, `clip_window_frames`, `subsample_indices`. `subsample_indices` picks N indices uniformly from a coverage-second window centred on a target frame, with stride that adapts to source fps so real-world coverage stays constant. |

## Planned

- `match_metadata.py` — gender/match stratifiers for evaluation
- `eval_metrics.py` — torcheval-backed metric assembly (confusion pairs, min-class F1, stratified summaries)

## Why copy not import (for the BST-mirrored modules)

`src/bst_refactor/` is a dependency for Model A. Importing into BRIC
would couple to BST's internal structure (which can move without
warning) and create a cycle if BST ever needs to share with us.

The trade-off is drift: if BST renames a class or shifts a court
constant, our copy doesn't notice. A `tests/test_shared_drift.py`
asserting parity is a v2 follow-up.

## Public API at a glance

```python
from shared.taxonomy import TAXONOMY_UNE_MERGE_V1_NOSIDES, DEFAULT_TAXONOMY
from shared.court import (
    REF_COURT_M, REF_COURT_CORNERS_M,
    get_court_info, project, convert_homogeneous,
)
from shared.video_io import VideoInfo, get_video_info, iter_frames, read_frame_at, read_frames
from shared.temporal import subsample_indices, clip_window_seconds, clip_window_frames
```

## Related docs

- [`docs/decisions_log.md`](../../docs/decisions_log.md) — DL-014 / DL-015 cover
  the taxonomy choice and the no-touch boundary.
