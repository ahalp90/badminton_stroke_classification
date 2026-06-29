# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

import warnings

from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------
def pad_class_labels(labels: list[str]) -> list[str]:
    """Pad class label strings to uniform width for aligned display (e.g. F1 table).

    :param labels: Class label list (e.g. ``list(taxonomy.classes)``).
    :return: Labels padded with spaces to the length of the longest label.
    """
    max_len = max(len(s) for s in labels)
    return [s.ljust(max_len) for s in labels]


# How many bone-vector sets each pose style adds on top of joints.
# Used to compute input feature dimension: (n_joints + n_bones * multiplier) * channels
POSE_BONE_MULTIPLIER = {'J_only': 0, 'JnB_bone': 1, 'JnB_interp': 1, 'Jn2B': 2}


def get_bone_pairs(skeleton_format='coco'):
    match skeleton_format:
        case 'coco':
            pairs = [
                (0,1),(0,2),(1,2),(1,3),(2,4),   # head
                (3,5),(4,6),                     # ears to shoulders
                (5,7),(7,9),(6,8),(8,10),        # arms
                (5,6),(5,11),(6,12),(11,12),     # torso
                (11,13),(13,15),(12,14),(14,16)  # legs
            ]
        case _:
            raise NotImplementedError
    return pairs


def make_seq_len_same(
    target_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray
):
    video_len = len(pos)

    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)

        joints = joints[::stride][:target_len]
        pos = pos[::stride][:target_len]
        shuttle = shuttle[::stride][:target_len]

        new_video_len = len(pos)

        if need_padding:
            pad_len = target_len - new_video_len
            joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
            pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
            shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    else:
        # Since they have been normalized, we don't interpolate them.
        new_video_len = video_len

        pad_len = target_len - new_video_len
        joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
        pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
        shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    return joints, pos, shuttle, new_video_len


def create_bones(joints: np.ndarray, pairs) -> np.ndarray:
    '''Same as create_bones_robust in TemPose.'''
    # joints: (t, m, J, 2)
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        # bone: (t, m, 2)
        bones.append(bone)
    return np.stack(bones, axis=-2)


def interpolate_joints(joints: np.ndarray, pairs) -> np.ndarray:
    '''Same as create_limbs_robust when 'num_steps' set to 3 in TemPose.'''
    # joints: (t, m, J, 2)
    mid_joints = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        mid_j = np.where((start_j != 0.0) & (end_j != 0.0), (start_j + end_j) / 2, 0.0)
        # mid_j: (t, m, 2)
        mid_joints.append(mid_j)
    bones_center = np.stack(mid_joints, axis=-2)  # bones_center: (t, m, B, 2)
    return np.concatenate((joints, bones_center), axis=-2)  # (t, m, J+B, 2)


class Dataset_npy_collated(Dataset):
    def __init__(
        self,
        root_dir: Path,
        set_name: str,
        pose_style='J_only',
        train_partial=1.0
    ):
        '''
        Parameters
        - `set_name`: 'train', 'val', 'test'
        - `pose_style`: 'J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'
        
        Notice: There is no random translation here.
        '''
        super().__init__()
        
        assert set_name in ['train', 'val', 'test'], 'Invalid set_name.'
        assert pose_style in ['J_only', 'JnB_interp', 'JnB_bone', 'Jn2B'], 'Invalid pose_style.'

        branch = root_dir/set_name

        self.human_pose = np.load(str(branch/f'{pose_style}.npy'))
        self.pos = np.load(str(branch/'pos.npy'))
        self.shuttle = np.load(str(branch/'shuttle.npy'))
        self.videos_len = np.load(str(branch/'videos_len.npy'))
        self.labels: np.ndarray = np.load(str(branch/'labels.npy'))

        # Row-aligned clip stems sidecar (Step C of the taxon_pinned_w_preds
        # refactor). Legacy collations don't carry the file -- graceful None
        # fallback so old collated dirs still load. New consumers (e.g. the
        # post-hoc FE-JSON converter) should check for None before joining.
        clip_stems_path = branch/'clip_stems.npy'
        if clip_stems_path.exists():
            self.clip_stems: np.ndarray | None = np.load(
                str(clip_stems_path), allow_pickle=True,
            )
        else:
            warnings.warn(
                f'{clip_stems_path} not found; this collation predates the '
                f'taxon_pinned_w_preds refactor. Per-clip alignment unavailable.',
                stacklevel=2,
            )
            self.clip_stems = None

        # ---------------------------------------------------------------
        # DIVERGENCE FROM ORIGINAL BST: Drop zero-length clips.
        #
        # Clips where MMPose failed on EVERY frame end up with
        # videos_len=0 after collation. The transformer's padding mask
        # becomes all-False, causing softmax(all -inf) = NaN, which
        # poisons the entire training run from the first epoch.
        #
        # The original BST author hand-curated his clip set and published
        # pre-extracted .npy files (see BST-original README), so he
        # likely never encountered zero-frame clips. Our automated
        # pipeline processes all clips including degenerate ones.
        #
        # TODO: Investigate whether the original BST dataset_npy files
        # (Google Drive links in BST-original/README.md) contain any
        # zero-length clips. If they do, this is a latent bug in the
        # original; if not, our clip extraction or pose detection is
        # producing clips that his pipeline never generated.
        # ---------------------------------------------------------------
        valid = self.videos_len > 0
        n_dropped = int(np.sum(~valid))
        if n_dropped > 0:
            print(f'  [{set_name}] Dropping {n_dropped} zero-length clips '
                  f'(of {len(valid)} total)')
            self.human_pose = self.human_pose[valid]
            self.pos = self.pos[valid]
            self.shuttle = self.shuttle[valid]
            self.videos_len = self.videos_len[valid]
            self.labels = self.labels[valid]
            if self.clip_stems is not None:
                self.clip_stems = self.clip_stems[valid]

        if set_name == 'train' and train_partial < 1:
            self.adjust_to_partial_train_set(train_partial)

        # J_only: (n, t, m, J, d)
        # JnB: (n, t, m, J+B, d)
        # Jn2B: (n, t, m, J+2B, d)
        # pos: (n, t, m, xy)
        # shuttle: (n, t, xy)
        # videos_len: (n)
        # labels: (n)

    def adjust_to_partial_train_set(self, train_partial):
        new_human_pose = []
        new_pos = []
        new_shuttle = []
        new_videos_len = []
        new_labels = []
        new_clip_stems: list[np.ndarray] | None = (
            [] if self.clip_stems is not None else None
        )

        types = np.unique(self.labels)
        for typ in types:
            choose_i = np.nonzero(self.labels == typ)[0]
            typ_n = int(len(choose_i) * train_partial)
            choose_i = choose_i[:typ_n]

            new_human_pose.append(self.human_pose[choose_i])
            new_pos.append(self.pos[choose_i])
            new_shuttle.append(self.shuttle[choose_i])
            new_videos_len.append(self.videos_len[choose_i])
            new_labels.append(self.labels[choose_i])
            if new_clip_stems is not None:
                new_clip_stems.append(self.clip_stems[choose_i])

        self.human_pose = np.concatenate(new_human_pose)
        self.pos = np.concatenate(new_pos)
        self.shuttle = np.concatenate(new_shuttle)
        self.videos_len = np.concatenate(new_videos_len)
        self.labels = np.concatenate(new_labels)
        if new_clip_stems is not None:
            self.clip_stems = np.concatenate(new_clip_stems)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return (self.human_pose[i], self.pos[i], self.shuttle[i]), \
                self.videos_len[i], self.labels[i]


def prepare_npy_collated_loaders(
    root_dir: Path,
    pose_style='Jn2B',
    batch_size=128,
    use_cuda=True,
    num_workers=(0, 0, 0),
    train_partial=1.0
):
    '''Notice that this one RandomTranslation is not used.'''
    train_set = Dataset_npy_collated(root_dir, 'train', pose_style, train_partial)
    val_set = Dataset_npy_collated(root_dir, 'val', pose_style)
    test_set = Dataset_npy_collated(root_dir, 'test', pose_style)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers[0]
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers[1]
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers[2]
    )
    return train_loader, val_loader, test_loader
