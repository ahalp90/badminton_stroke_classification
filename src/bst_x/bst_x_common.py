"""Shared scaffolding between bst_x_train.py and bst_x_infer.py.

Lifted pre-X3D-S so a third entry point (the X3D-S training script) does
not triplicate the orchestration glue. The BST model graph itself is not
refactored here; this module owns the variant table, the tee'er, the
network builder, and the data-provenance manifest helper only.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preparing_data.shuttleset_dataset import POSE_BONE_MULTIPLIER, get_bone_pairs
from model.bst import BST_0, BST_PPF, BST_CG, BST_AP, BST_CG_AP


# BST variant name -> pre-configured constructor (partials defined in bst.py).
# Both bst_x_train and bst_x_infer dispatch through this single mapping.
#
# 'BST_X' is the project name for the adapted BST_CG_AP network.
# It uses the same modules with different hyperparameters around
# things like scheduling, augmentation, loss, player tracking and
# input frame validation.
MODELS = {
    'BST_0':     BST_0,
    'BST':       BST_PPF,
    'BST_CG':    BST_CG,
    'BST_AP':    BST_AP,
    'BST_CG_AP': BST_CG_AP,
    'BST_X':     BST_CG_AP,
    # 'BST_X_RGB': BST_X_RGB,  # placeholder for the X3D-S fusion variant
}


class Tee:
    """Mirror writes across multiple streams (terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def build_bst_x_network(
    model_name: str,
    *,
    n_joints: int,
    pose_style: str,
    in_channels: int,
    n_class: int,
    seq_len: int = 100,
    depth_tem: int = 2,
    depth_inter: int = 1,
    device: str = 'cuda',
) -> tuple[nn.Module, int]:
    """Construct a BST variant with feature-dim wiring shared between train and infer.

    Returns ``(net, n_bones)``. ``n_bones`` is the count of bone tokens
    appended after the joint tokens along the pose axis of ``human_pose``
    (``len(get_bone_pairs()) * POSE_BONE_MULTIPLIER[pose_style]``). The
    training loop slices ``human_pose[..., -n_bones:, :]`` to keep
    random-translation off the bone rows; inference can ignore it.

    :param in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz).
    """
    n_bones = len(get_bone_pairs()) * POSE_BONE_MULTIPLIER[pose_style]
    in_dim = (n_joints + n_bones) * in_channels
    net = MODELS[model_name](
        in_dim=in_dim,
        n_class=n_class,
        seq_len=seq_len,
        depth_tem=depth_tem,
        depth_inter=depth_inter,
    ).to(device)
    return net, n_bones


@torch.no_grad()
def dump_topk_predictions(
    model: nn.Module,
    loader,
    device,
    k: int = 5,
) -> dict[str, np.ndarray]:
    """Run a loader through the model once, returning logits + a top-k summary.

    The single source of the per-stroke prediction payload that both
    ``bst_x_train`` (end-of-serial dump) and ``bst_x_infer --fe`` (post-hoc dump)
    write to npz. Raw logits are kept so any consumer can derive softmax and
    fit post-hoc temperature scaling without re-running inference.

    Row order follows the loader: a ``shuffle=False`` loader yields rows in the
    dataset's in-memory order, so the returned arrays row-align with that
    dataset's own ``labels`` / ``clip_stems`` (i.e. after the zero-length-clip
    drop and any train_partial reorder), NOT with the raw on-disk
    ``clip_stems.npy``. Callers that want the stems pull them from the same
    dataset and store them alongside (see ``Task.dump_predictions``).

    :param model: a built BST network (any variant); set to eval here.
    :param loader: yields ``((human_pose, pos, shuttle), video_len, labels)``.
    :param device: torch device the model lives on.
    :param k: top-k width; clamped to the head size when the head is smaller.
    :return: dict with ``logits`` (n, n_classes) float32, ``y_true`` (n,)
        int64, ``y_pred_top1`` (n,) int64, ``topk_idx`` (n, k_eff) int64.
    """
    model.eval()
    logits_ls, y_true_ls, top1_ls, topk_idx_ls = [], [], [], []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose = human_pose.to(device)
        shuttle = shuttle.to(device)
        pos = pos.to(device)
        video_len = video_len.to(device)
        # Flatten the (joints/bones, channels) trailing dims into one feature
        # dim, mirroring the train/infer forward massage.
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        k_eff = min(k, logits.shape[-1])
        topk_idx = torch.topk(logits, k=k_eff, dim=-1).indices
        logits_ls.append(logits.cpu().numpy())
        y_true_ls.append(labels.numpy())
        # top-1 via argmax to match every other metric site (equals topk_idx[:, 0] on
        # the tie-free logits a trained model produces; the tie-guard downstream catches any tie).
        top1_ls.append(logits.argmax(dim=-1).cpu().numpy())
        topk_idx_ls.append(topk_idx.cpu().numpy())
    return {
        'logits':      np.concatenate(logits_ls).astype(np.float32),
        'y_true':      np.concatenate(y_true_ls).astype(np.int64),
        'y_pred_top1': np.concatenate(top1_ls).astype(np.int64),
        'topk_idx':    np.concatenate(topk_idx_ls).astype(np.int64),
    }


def _write_prediction_npz(out_path, dump, dataset, taxonomy, run_id, serial):
    """Write a per-split prediction npz with the shared 9-key schema.

    The single payload source for ``bst_x_train.Task.dump_predictions`` and
    ``bst_x_infer.dump_run_predictions``. Both writers always produced the same
    9 keys (proven byte-for-byte in 03 section 6.4); this helper makes that an
    enforced contract instead of an accident. Out_path, the directory, the
    split-loop, and any caller-specific manifest (e.g. the inference run's
    inference_manifest.yaml) stay per-caller.

    ``topk_idx`` ships as ``(N, k_eff) int64`` where ``k_eff = min(k, head_size)``;
    both production writers pass ``k=5`` to ``dump_topk_predictions`` so the schema
    is ``(N, 5)`` in deployment. Caller-side convention, not enforced here so tests
    can exercise smaller heads or smaller k.

    Hard-fails on a ``None`` ``clip_stems`` sidecar (a legacy collation without
    ``clip_stems.npy``). ``np.asarray(None)`` would otherwise write a silent
    0-d array that desyncs every row from its stem.

    :param out_path: full destination path; the caller owns the dir + filename.
    :param dump: one split's output from ``dump_topk_predictions``.
    :param dataset: the in-memory dataset, for ``clip_stems`` row-aligned with
        ``y_true``.
    :param taxonomy: the resolved taxonomy (provides ``classes`` + ``name``).
    :param run_id: the originating run dir's name (string-shaped).
    :param serial: int64-castable serial index.
    """
    assert dataset.clip_stems is not None, (
        f'{out_path.stem}: dataset.clip_stems is None (legacy collation with '
        f'no clip_stems.npy); re-collate before dumping predictions.'
    )
    np.savez(
        out_path,
        logits=dump['logits'],
        y_true=dump['y_true'],
        y_pred_top1=dump['y_pred_top1'],
        topk_idx=dump['topk_idx'],
        clip_stems=np.asarray(dataset.clip_stems, dtype=object),
        class_list=np.array(taxonomy.classes, dtype=object),
        run_id=np.array(run_id, dtype=object),
        serial_no=np.array(serial, dtype=np.int64),
        taxonomy_name=np.array(taxonomy.name, dtype=object),
    )


def compute_data_provenance(
    clips_csv_path: Path,
    collation_id: str,
    npy_collated_dir: str,
) -> dict:
    """Manifest ``extra.data_provenance`` for ``track_run``.

    Hashes the clips CSV so the manifest pins the source-of-truth that
    produced this run's collated arrays. Fail fast if missing.

    ``collation_id`` is the collation generation tag the run trained on; it
    superseded the old auto-derived ``effective_ablation_id`` (auto-derive is
    gone, so the recorded value is just ``hyp.collation_id`` verbatim).
    """
    if not clips_csv_path.exists():
        raise FileNotFoundError(
            f'clips_csv does not exist: {clips_csv_path}\n'
            f'  (Run preparing_data.prepare_train_on_shuttleset to generate '
            f'the collated arrays first.)'
        )
    clips_csv_sha = hashlib.sha256(clips_csv_path.read_bytes()).hexdigest()
    return {
        'data_provenance': {
            'clips_csv_path': str(clips_csv_path),
            'clips_csv_sha256': clips_csv_sha,
            'collation_id': collation_id,
            'npy_collated_dir': npy_collated_dir,
        },
    }
