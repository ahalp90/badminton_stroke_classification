# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

# BST inference for ShuttleSet. Two faces:
#
#   1. Library: infer() + Task — load a checkpoint and predict, for a live
#      single-clip backend (e.g. a Gradio GUI).
#   2. CLI --fe mode: post-hoc batch dump of per-stroke logits + top-k for an
#      already-trained run, writing the same npz schema bst_x_train emits at
#      end-of-serial. Folds in the retired eval_dump_predictions.py; lets the
#      FE-shape converter / calibration run against a run without retraining.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src/bst_x \
#       python -m bst_x_infer --fe \
#           --run-dir .../experiments/bst_x/shuttleset/run_<id> --serial 5
#   The dump lands in <run-dir>/inference_runs/<timestamp>/ (npz +
#   inference_manifest.yaml); pass --fe-output-dir to redirect it elsewhere.
#
# See bst_x_train.py for detailed PyTorch/TF comparison comments.

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from preparing_data.shuttleset_dataset import Dataset_npy_collated
from pipeline.config import (
    Taxonomy,
    collation_id_from_manifest,
    derive_npy_collated_dir_basename,
    resolve_taxonomy,
)
from pipeline.data_access import env_path_or_none, load_repo_dotenv
from bst_x_common import build_bst_x_network, dump_topk_predictions


@torch.no_grad()  # no gradient tracking needed for inference — saves memory
def infer(
    model: nn.Module,
    loader,
    device
):
    model.eval()  # disable dropout, set batchnorm to eval mode
    pred_ls = []

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        # argmax gives predicted class index; .cpu() moves result back from GPU
        pred = torch.argmax(logits, dim=1).cpu()

        pred_ls.append(pred)

    # torch.cat joins list of batch predictions into one tensor
    return torch.cat(pred_ls)


class Task:
    """Live single-clip inference helper (Gradio-style backend).

    Build the head at ``taxonomy.n_classes`` and decode predictions against
    ``taxonomy.classes``. Labels on disk are already in that index space (no
    runtime remap), so the head dim is just the taxonomy size.
    """

    def __init__(self, n_joints=17) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints

    def prepare_loader(
        self,
        npy_collated_dir: Path,
        pose_style='Jn2B',
        batch_size=128,
    ):
        your_set = Dataset_npy_collated(npy_collated_dir, 'test', pose_style)

        self.infer_loader = DataLoader(
            dataset=your_set,
            batch_size=batch_size
        )
        self.pose_style = pose_style

    def get_network_architecture(
        self,
        *,
        taxonomy: Taxonomy,
        model_name: str = 'BST_X',
        seq_len: int = 100,
        in_channels: int = 2,
    ):
        """Build the inference model at the taxonomy head dim.

        The weights being loaded were trained against ``taxonomy.classes``;
        a mismatch between the weight file's head dim and
        ``taxonomy.n_classes`` raises a clear shape error inside
        ``load_state_dict``. For a legacy run, pass the taxonomy the run
        recorded (``resolve_taxonomy(manifest['config']['taxonomy'])``).

        :param taxonomy: the taxonomy the weights were trained under.
        :param in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz).
        """
        self.taxonomy = taxonomy
        self.class_list = list(taxonomy.classes)
        self.net, _n_bones = build_bst_x_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            in_channels=in_channels,
            n_class=taxonomy.n_classes,
            seq_len=seq_len,
            device=self.device,
        )

    def load_weight(self, weight_path: Path):
        self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))

    def infer(self):
        return infer(self.net, self.infer_loader, self.device)


# ==========================================================================
# --fe batch dump: per-stroke logits + top-k npz for an existing run
# ==========================================================================

def _resolve_collated_dir(
    manifest: dict, config: dict, collated_data_root: Path | None, run_dir: Path,
) -> Path:
    """Resolve the collated dir the run trained on, for a post-hoc dump.

    Prefers the recorded ``extra.data_provenance.npy_collated_dir`` (carries the
    historical basename verbatim, including pre-split-fold names like
    ``npy_wipe_drop``); falls back to deriving it from the recorded config.

    Root order: ``--collated-data-root`` override, then
    ``BST_X_COLLATED_DATA_ROOT`` (e.g. /scratch/comp320a on bourbaki), then the
    in-repo ``preparing_data/`` convention. ``run_dir`` is shaped
    ``experiments/bst_x/shuttleset/run_<id>/`` after the Plan 3 restructure, so
    ``run_dir.parents[3]`` walks back up to the repo root.
    """
    recorded_dir = (
        (manifest.get('extra') or {}).get('data_provenance', {}).get('npy_collated_dir')
    )
    basename = recorded_dir or derive_npy_collated_dir_basename(
        use_3d_pose=config['use_3d_pose'],
        seq_len=config['seq_len'],
        split_column=config['split_column'],
        collation_id=collation_id_from_manifest(manifest),
    )
    if collated_data_root is None:
        collated_data_root = (
            env_path_or_none('BST_X_COLLATED_DATA_ROOT')
            or run_dir.parents[3] / 'src' / 'bst_x' / 'preparing_data'
        )
    return collated_data_root / f"ShuttleSet_data_{config['taxonomy']}" / basename


def dump_run_predictions(
    *,
    run_dir: Path,
    serial: int,
    fe_output_dir: Path | None = None,
    splits: tuple[str, ...] = ('val', 'test'),
    collated_data_root: Path | None = None,
    model_name: str = 'BST_X',
    n_joints: int = 17,
    batch_size: int = 128,
) -> Path:
    """Dump per-split prediction npz for an already-trained run.

    Each dump lands in its own timestamped dir so post-hoc inference never
    clobbers the run's training-time ``predictions/`` and re-dumps don't
    collide: ``<base>/inference_runs/<YYYYmmdd_HHMMSS>/``. ``base`` defaults to
    ``run_dir`` (co-located with the run), or ``fe_output_dir/<run_id>`` when an
    override is passed. A small ``inference_manifest.yaml`` records the source
    weights / serial / splits / time alongside the npz.

    Same npz schema as ``bst_x_train``'s end-of-serial dump (logits, y_true,
    y_pred_top1, topk_idx, clip_stems, class_list, run_id, serial_no,
    taxonomy_name). New-schema runs only: labels.npy is in active class space,
    so there's no remap.

    :return: the timestamped output dir holding this dump's npz + manifest.
    """
    manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
    config = manifest['config']
    taxonomy = resolve_taxonomy(config['taxonomy'])

    target = next(
        (s for s in manifest.get('serials', []) if s['serial_no'] == serial), None
    )
    if target is None:
        sys.exit(f'serial {serial} not found in {run_dir}/manifest.yaml')
    weights_path = run_dir / 'weights' / Path(target['weights_path']).name
    if not weights_path.is_file():
        sys.exit(f'weights file missing: {weights_path}')

    collated_dir = _resolve_collated_dir(manifest, config, collated_data_root, run_dir)
    if not collated_dir.is_dir():
        sys.exit(f'collated dir missing: {collated_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, _n_bones = build_bst_x_network(
        model_name,
        n_joints=n_joints,
        pose_style=config['pose_style'],
        in_channels=(3 if config['use_3d_pose'] else 2),
        n_class=taxonomy.n_classes,
        seq_len=config['seq_len'],
        device=device,
    )
    net.load_state_dict(
        torch.load(str(weights_path), map_location=device, weights_only=True)
    )

    print(f'run_dir: {run_dir}')
    print(f'weights: {weights_path}')
    print(f'collated_dir: {collated_dir}')
    print(f'taxonomy: {taxonomy.name} ({taxonomy.n_classes} classes)')

    # Own timestamped dir per dump: co-located in the run by default, or under
    # fe_output_dir/<run_id> when overridden. Never the run's training-time
    # predictions/ dir, so a re-dump can't clobber it.
    now = datetime.now()
    base = (fe_output_dir / run_dir.name) if fe_output_dir is not None else run_dir
    out_dir = base / 'inference_runs' / f'{now:%Y%m%d_%H%M%S}'
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for split in splits:
        dataset = Dataset_npy_collated(collated_dir, split, config['pose_style'])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dump = dump_topk_predictions(net, loader, device, k=5)
        # Hard-fail on a None sidecar (a legacy collation with no clip_stems.npy):
        # np.asarray(None) writes a silent 0-d array that would desync the npz
        # row -> stem join. New collations always carry clip_stems.npy.
        assert dataset.clip_stems is not None, (
            f'{split}: dataset.clip_stems is None (legacy collation with no '
            f'clip_stems.npy); re-collate before dumping predictions.'
        )
        out_path = out_dir / f'{split}_serial_{serial}.npz'
        np.savez(
            out_path,
            logits=dump['logits'],
            y_true=dump['y_true'],
            y_pred_top1=dump['y_pred_top1'],
            topk_idx=dump['topk_idx'],
            # clip_stems off the in-memory dataset: row-aligned through the
            # dataset's filters, so the npz row -> stem join is self-contained.
            clip_stems=np.asarray(dataset.clip_stems, dtype=object),
            class_list=np.array(taxonomy.classes, dtype=object),
            run_id=np.array(run_dir.name, dtype=object),
            serial_no=np.array(serial, dtype=np.int64),
            taxonomy_name=np.array(taxonomy.name, dtype=object),
        )
        written.append(out_path.name)
        print(f'saved: {out_path} ({len(dump["y_true"])} rows)')

    # Small provenance manifest so a dump self-describes when/from-what, beyond
    # what each npz already carries.
    (out_dir / 'inference_manifest.yaml').write_text(yaml.safe_dump({
        'source_run_id': run_dir.name,
        'created_at': now.isoformat(timespec='seconds'),
        'serial_no': serial,
        'splits': list(splits),
        'taxonomy': taxonomy.name,
        'weights_path': str(weights_path),
        'collated_dir': str(collated_dir),
        'npz_files': written,
    }, sort_keys=False))
    print(f'wrote: {out_dir / "inference_manifest.yaml"}')
    return out_dir


if __name__ == '__main__':
    # Load .env so BST_X_COLLATED_DATA_ROOT resolves the same way the collator
    # and bst_x_train do. No-op without .env; shell exports win.
    load_repo_dotenv()

    parser = argparse.ArgumentParser(
        description='BST inference. --fe runs the post-hoc batch dump of '
                    'per-stroke logits + top-k for an existing run.',
    )
    parser.add_argument(
        '--fe', action='store_true',
        help='FE/batch dump mode. Requires --run-dir.',
    )
    parser.add_argument(
        '--fe-output-dir', type=Path, default=None,
        help='Optional override for where the dump lands. Default writes into '
             '<run-dir>/inference_runs/<timestamp>/; with an override, '
             '<fe-output-dir>/<run_id>/inference_runs/<timestamp>/.',
    )
    parser.add_argument(
        '--run-dir', type=Path, default=None,
        help='experiments/bst_x/shuttleset/run_<id>/ whose weights to dump. Required when --fe is set.',
    )
    parser.add_argument('--serial', type=int, default=5,
                        help='Serial number whose weights to evaluate.')
    parser.add_argument('--splits', default='val,test',
                        help='Comma-separated splits to dump (default: val,test).')
    parser.add_argument(
        '--collated-data-root', type=Path, default=None,
        help='Root holding ShuttleSet_data_<tax>/. Defaults to '
             'BST_X_COLLATED_DATA_ROOT, then the in-repo preparing_data/.',
    )
    parser.add_argument('--model-name', default='BST_X',
                        help='BST variant; defaults to BST_X (the project name for BST_CG_AP). '
                             'Pass --model-name BST_CG_AP for a Chang-configuration build; '
                             'saves and resumes lowercase bst_cg_ap_*.pt.')
    args = parser.parse_args()

    # --fe-output-dir is an optional override that only makes sense in --fe mode.
    if args.fe_output_dir is not None and not args.fe:
        parser.error('--fe-output-dir requires --fe (no implicit dump mode)')
    if not args.fe:
        parser.error(
            'bst_x_infer CLI currently only implements --fe (batch dump) mode. '
            'For live single-clip inference, import infer() / Task instead.'
        )
    if args.run_dir is None:
        parser.error('--fe requires --run-dir <experiments/bst_x/shuttleset/run_...>')

    dump_run_predictions(
        run_dir=args.run_dir.resolve(),
        serial=args.serial,
        fe_output_dir=args.fe_output_dir,
        splits=tuple(s.strip() for s in args.splits.split(',') if s.strip()),
        collated_data_root=args.collated_data_root,
        model_name=args.model_name,
    )
