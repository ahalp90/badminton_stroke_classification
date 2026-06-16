"""Tests for Dataset_npy_collated: load shape, clip_stems sidecar contract,
partial-train slicing.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.bst_x.stroke_classification.preparing_data.shuttleset_dataset import (
    Dataset_npy_collated,
)


def _make_synthetic_split(
    tmp: Path,
    *,
    n: int = 4,
    t: int = 100,
    m: int = 2,
    J: int = 17,
    d: int = 2,
    labels: np.ndarray | None = None,
    videos_len: np.ndarray | None = None,
    with_clip_stems: bool = True,
) -> Path:
    """Write a synthetic collated split dir under ``tmp/train/``.

    :param tmp: temp dir to write into.
    :param n: number of clips.
    :param labels: per-clip labels; defaults to all-zeros.
    :param videos_len: per-clip video length; defaults to all-100. Pass an
        array with some zeros to exercise the zero-length-filter path.
    :param with_clip_stems: if True, write clip_stems.npy alongside; if False,
        omit it to simulate a legacy collation.
    :return: the split dir path (containing the .npy files).
    """
    split_dir = tmp / "train"
    split_dir.mkdir()
    np.save(split_dir / "J_only.npy", np.zeros((n, t, m, J, d), dtype=np.float32))
    np.save(split_dir / "pos.npy", np.zeros((n, t, m, 2), dtype=np.float32))
    np.save(split_dir / "shuttle.npy", np.zeros((n, t, 2), dtype=np.float32))
    videos_len = videos_len if videos_len is not None else np.full(n, 100, dtype=np.int64)
    np.save(split_dir / "videos_len.npy", videos_len)
    labels = labels if labels is not None else np.zeros(n, dtype=np.int64)
    np.save(split_dir / "labels.npy", labels)
    if with_clip_stems:
        stems = np.array([f"stem_{i}" for i in range(n)], dtype=object)
        np.save(split_dir / "clip_stems.npy", stems, allow_pickle=True)
    return split_dir


def test_dataloader_batch_shapes():
    """Loads a batch, verifies shapes + the clip_stems sidecar is row-aligned."""
    n = 4
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_split(Path(tmp), n=n)
        dataset = Dataset_npy_collated(Path(tmp), "train")
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        (human_pose, pos, shuttle), videos_len, labels = next(iter(loader))

        # Print shapes to verify they match expected dimensions
        print("human_pose shape:", human_pose.shape)  # (2, 100, 2, 17, 2)
        print("pos shape:       ", pos.shape)  # (2, 100, 2, 2)
        print("shuttle shape:   ", shuttle.shape)  # (2, 100, 2)
        print("videos_len shape:", videos_len.shape)  # (2,)
        print("labels shape:    ", labels.shape)  # (2,)

        assert human_pose.shape == (2, 100, 2, 17, 2)
        assert pos.shape == (2, 100, 2, 2)
        assert shuttle.shape == (2, 100, 2)
        assert videos_len.shape == (2,)
        assert labels.shape == (2,)

        # clip_stems sidecar (Step C) row-aligns with labels at load time.
        assert dataset.clip_stems is not None
        assert len(dataset.clip_stems) == n


def test_dataset_graceful_none_when_clip_stems_missing():
    """Legacy collations without clip_stems.npy load with graceful None
    fallback + a UserWarning. Pins the back-compat contract for resuming
    runs that predate the taxon_pinned_w_preds refactor.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_split(Path(tmp), with_clip_stems=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset = Dataset_npy_collated(Path(tmp), "train")
    assert dataset.clip_stems is None
    assert any('predates' in str(w.message) for w in caught), (
        f'expected a "predates" warning, got: {[str(w.message) for w in caught]}'
    )


def test_dataset_clip_stems_after_zero_length_filter():
    """Zero-length filter (videos_len > 0) extends to clip_stems when present.

    Pins the parallel slicing of clip_stems alongside labels/pose/etc., so a
    future refactor that drops the clip_stems branch from the filter goes red
    rather than letting the stem array desync from labels.
    """
    n = 4
    # Two clips have videos_len=0 (will be dropped); two survive.
    videos_len = np.array([100, 0, 0, 100], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_split(Path(tmp), n=n, videos_len=videos_len)
        dataset = Dataset_npy_collated(Path(tmp), "train")
    assert dataset.clip_stems is not None
    assert len(dataset.clip_stems) == len(dataset.labels) == 2
    # Rows 0 and 3 survive (videos_len=100); rows 1 and 2 drop (videos_len=0).
    assert dataset.clip_stems.tolist() == ['stem_0', 'stem_3']


def test_dataset_clip_stems_after_partial_train():
    """adjust_to_partial_train_set mirrors clip_stems slicing with labels.

    Per-class slicing keeps the first ``int(n_per_class * train_partial)``
    clips per label; clip_stems must stay row-aligned after the concatenation.
    Pins the slicing alignment so a future refactor that drops the
    new_clip_stems plumbing turns red rather than silently drifting.
    """
    # 4 classes, 2 clips each, train_partial=0.5 -> 1 clip per class -> 4 total.
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_split(Path(tmp), n=8, labels=labels)
        dataset = Dataset_npy_collated(Path(tmp), "train", train_partial=0.5)
    assert dataset.clip_stems is not None
    assert len(dataset.clip_stems) == len(dataset.labels) == 4
    # Per-class slicing keeps choose_i[:typ_n], i.e. the FIRST clip per class.
    # Original first-of-class indices were 0, 2, 4, 6 -> stems 0, 2, 4, 6.
    assert set(dataset.clip_stems.tolist()) == {'stem_0', 'stem_2', 'stem_4', 'stem_6'}
