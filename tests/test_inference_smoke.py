"""Inference npz schema smoke for ``bst_infer --fe`` (Step D9 / D10).

Builds a tiny fake run dir (manifest + weights + collation) and runs the
post-hoc batch dump end-to-end, asserting the npz schema matches what
``bst_train`` writes at end-of-serial and that only the requested splits land.

CPU-only; no /scratch. The train-side dump + label-coverage assert live in
tests/test_train_surface.py.

Run from repo root::

    pytest tests/test_inference_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from main_on_shuttleset import bst_infer
from main_on_shuttleset.bst_common import build_bst_network, dump_topk_predictions
from pipeline.config import resolve_taxonomy
from preparing_data.shuttleset_dataset import Dataset_npy_collated
from torch.utils.data import DataLoader


# The npz schema both dump paths (bst_train end-of-serial, bst_infer --fe) emit.
NPZ_FIELDS = {
    'logits', 'y_true', 'y_pred_top1', 'topk_idx',
    'class_list', 'run_id', 'serial_no', 'taxonomy_name',
}

TAX_NAME = 'bst_12'  # registered, 12 classes, no sides — simple head


def _write_split(split_dir: Path, *, n_bones: int, labels: list[int]) -> None:
    n = len(labels)
    j_plus_b = 17 + n_bones
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.save(split_dir / 'JnB_bone.npy', rng.standard_normal((n, 100, 2, j_plus_b, 2)).astype(np.float32))
    np.save(split_dir / 'pos.npy', rng.standard_normal((n, 100, 2, 2)).astype(np.float32))
    np.save(split_dir / 'shuttle.npy', rng.standard_normal((n, 100, 2)).astype(np.float32))
    np.save(split_dir / 'videos_len.npy', np.full(n, 100, dtype=np.int64))
    np.save(split_dir / 'labels.npy', np.array(labels, dtype=np.int64))
    stems = np.array([f'{split_dir.name}_clip_{i}' for i in range(n)], dtype=object)
    np.save(split_dir / 'clip_stems.npy', stems, allow_pickle=True)


def _build_fake_run(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a run dir (manifest + weights) and a sibling collation tree.

    :return: (run_dir, collated_data_root) for dump_run_predictions.
    """
    taxonomy = resolve_taxonomy(TAX_NAME)
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=taxonomy.n_classes, seq_len=100, device='cpu',
    )

    # Collation under collated_data_root/ShuttleSet_data_<tax>/<basename>/.
    basename = 'npy_v2_taxon_pinned_w_preds'
    collated_data_root = tmp_path / 'scratch'
    coll = collated_data_root / f'ShuttleSet_data_{TAX_NAME}' / basename
    # Train covers all 12 classes (not dumped here, but realistic); val/test small.
    _write_split(coll / 'train', n_bones=n_bones, labels=list(range(12)))
    _write_split(coll / 'val', n_bones=n_bones, labels=[0, 5, 11, 3])
    _write_split(coll / 'test', n_bones=n_bones, labels=[7, 2, 9])

    # Run dir with weights + manifest.
    run_dir = tmp_path / 'run_fe_smoke'
    (run_dir / 'weights').mkdir(parents=True)
    weights_path = run_dir / 'weights' / 'model.pt'
    torch.save(net.state_dict(), str(weights_path))

    manifest = {
        'run_id': 'run_fe_smoke',
        'config': {
            'taxonomy': TAX_NAME,
            'split_column': 'split_v2',
            'collation_id': 'taxon_pinned_w_preds',
            'pose_style': 'JnB_bone',
            'seq_len': 100,
            'use_3d_pose': False,
            'classes': list(taxonomy.classes),
        },
        'extra': {'data_provenance': {'npy_collated_dir': basename}},
        'serials': [{'serial_no': 5, 'weights_path': 'weights/model.pt'}],
    }
    (run_dir / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    return run_dir, collated_data_root


def test_dump_run_predictions_writes_requested_splits(tmp_path, monkeypatch):
    # Force CPU: dump_run_predictions auto-selects cuda, which fails on a host
    # whose GPU is too old for the installed torch (the real runs are on GPU hosts).
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    run_dir, collated_data_root = _build_fake_run(tmp_path)
    fe_out = tmp_path / 'fe_dump'

    out_dir = bst_infer.dump_run_predictions(
        run_dir=run_dir, serial=5, fe_output_dir=fe_out,
        splits=('val', 'test'), collated_data_root=collated_data_root,
    )
    assert out_dir == fe_out / 'run_fe_smoke' / 'predictions'

    # Only val + test dumped (FE default); train is not.
    written = sorted(p.name for p in out_dir.glob('*.npz'))
    assert written == ['test_serial_5.npz', 'val_serial_5.npz']

    taxonomy = resolve_taxonomy(TAX_NAME)
    for split, expected_labels in (('val', [0, 5, 11, 3]), ('test', [7, 2, 9])):
        npz = np.load(out_dir / f'{split}_serial_5.npz', allow_pickle=True)
        assert set(npz.files) == NPZ_FIELDS, npz.files
        assert npz['logits'].shape == (len(expected_labels), taxonomy.n_classes)
        assert npz['topk_idx'].shape == (len(expected_labels), 5)  # head=12 >= k=5
        # shuffle=False dump keeps the on-disk label order (row-aligned w/ stems).
        assert npz['y_true'].tolist() == expected_labels
        assert list(npz['class_list']) == list(taxonomy.classes)
        assert str(npz['run_id']) == 'run_fe_smoke'
        assert int(npz['serial_no']) == 5
        assert str(npz['taxonomy_name']) == TAX_NAME


def test_dump_run_predictions_missing_serial_exits(tmp_path):
    run_dir, collated_data_root = _build_fake_run(tmp_path)
    with pytest.raises(SystemExit):
        bst_infer.dump_run_predictions(
            run_dir=run_dir, serial=99, fe_output_dir=tmp_path / 'fe',
            splits=('test',), collated_data_root=collated_data_root,
        )


def test_dump_topk_predictions_k_clamps_to_head(tmp_path):
    """topk width clamps to the head size when k exceeds it."""
    taxonomy = resolve_taxonomy(TAX_NAME)
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=taxonomy.n_classes, seq_len=100, device='cpu',
    )
    coll = tmp_path / 'coll'
    _write_split(coll / 'test', n_bones=n_bones, labels=[0, 1, 2, 3])
    loader = DataLoader(Dataset_npy_collated(coll, 'test', 'JnB_bone'), batch_size=2, shuffle=False)

    dump = dump_topk_predictions(net, loader, 'cpu', k=50)  # k >> head
    assert dump['logits'].shape == (4, taxonomy.n_classes)
    assert dump['topk_idx'].shape == (4, taxonomy.n_classes)  # clamped to 12
    assert dump['y_pred_top1'].tolist() == dump['topk_idx'][:, 0].tolist()
