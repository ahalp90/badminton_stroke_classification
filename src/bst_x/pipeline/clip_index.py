"""Clip-stem to mp4-path index for video-loading Datasets.

The clips dir keeps its ``{split}/{Top,Bottom}_{stroke_type}/*.mp4`` layout
after Phase 2 of the dir-flatten refactor (flattening the .mp4 tree itself
is deferred to Phase 3, which is indefinitely parked). Any CSV-driven
``Dataset`` that needs to load video frames should rglob once at
``__init__`` via the helper below and use the returned dict for O(1)
per-sample lookup rather than walking the tree per ``__getitem__``.

Expected consumers (not in-repo yet):
- Arch 2 3D CNN video Dataset.
- Arch 1 wrist-crop Dataset that pairs pose-derived wrist coords with
  cropped video frames.

Both should pair the returned lookup with a split + label derivation
driven by ``notebooks/clips_master.csv``, the same way
``collate_npy`` does for the pose + shuttle npys.

For a higher-level API that does the CSV read, taxonomy label derivation,
and flat shuttle / mmpose path resolution in one call, see
``pipeline.data_access.get_clip_records``; it calls this helper internally.
"""
from pathlib import Path


def build_clip_path_index(clips_dir: Path) -> dict[str, Path]:
    """Build a {clip_stem -> mp4 Path} lookup over the clips directory.

    Transparent to the nested {split}/{class}/ layout: we emit one entry
    per clip stem regardless of which subfolder the file sits under. The
    caller typically pairs this with a clips_master.csv filter keyed on
    clip_stem to pick train/val/test membership + label per the active
    taxonomy.

    See ``pipeline/README.md`` for a worked ``ClipVideoDataset`` sketch.

    No video-decode backend is assumed; the caller picks their own
    (cv2, decord, torchvision.io, etc.) in the Dataset subclass.

    :param clips_dir: Root clips directory (``CLIPS_OUTPUT_DIR`` or a
        symlinked scratch equivalent).
    :return: Mapping from clip stem (e.g. ``'35_1_10_17'``) to absolute
        ``Path`` of its .mp4 file. One-time O(n) stat cost at index
        construction (~seconds on a 33k-clip dir, cold FS); O(1) lookup
        per sample afterward.
    """
    return {p.stem: p for p in clips_dir.rglob("*.mp4")}
