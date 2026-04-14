"""
Issue 23: Test that loads a batch and prints shapes.
"""

import numpy as np
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader
from src.bst_refactor.stroke_classification.preparing_data.shuttleset_dataset import (
    Dataset_npy_collated,
)


def test_dataloader_batch_shapes():
    # Create fake data that looks like real data (4 samples)
    n, t, m, J, d = 4, 100, 2, 17, 2  # 4 clips, 100 frames, 2 players, 17 joints, x/y

    with tempfile.TemporaryDirectory() as tmp:
        split_dir = Path(tmp) / "train"
        split_dir.mkdir()

        np.save(split_dir / "J_only.npy", np.zeros((n, t, m, J, d), dtype=np.float32))
        np.save(split_dir / "pos.npy", np.zeros((n, t, m, 2), dtype=np.float32))
        np.save(split_dir / "shuttle.npy", np.zeros((n, t, 2), dtype=np.float32))
        np.save(split_dir / "videos_len.npy", np.full(n, 100, dtype=np.int64))
        np.save(split_dir / "labels.npy", np.zeros(n, dtype=np.int64))

        dataset = Dataset_npy_collated(Path(tmp), "train")
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        (human_pose, pos, shuttle), videos_len, labels = next(iter(loader))

        # Print shapes to verify they match expected dimensions - redundant but meets the requirement of "prints shapes"
        print("human_pose shape:", human_pose.shape)  # (2, 100, 2, 17, 2)
        print("pos shape:       ", pos.shape)  # (2, 100, 2, 2)
        print("shuttle shape:   ", shuttle.shape)  # (2, 100, 2)
        print("videos_len shape:", videos_len.shape)  # (2,)
        print("labels shape:    ", labels.shape)  # (2,)

        assert human_pose.shape == (
            2,
            100,
            2,
            17,
            2,
        )  # (batch, frames, players, joints, xy)
        assert pos.shape == (2, 100, 2, 2)  # (batch, frames, players, xy)
        assert shuttle.shape == (2, 100, 2)  # (batch, frames, xy)
        assert videos_len.shape == (2,)  # (batch,)
        assert labels.shape == (2,)  # (batch,)
