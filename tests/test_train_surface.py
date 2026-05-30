"""Train-surface tests for the taxon_pinned_w_preds refactor (Step D).

Covers the pieces that replaced the runtime active-class adapter and the
per-stroke predictions dump:

1. ``Task._assert_label_coverage`` — the contract guard that replaced
   ``derive_active_classes_from_labels``. Train must cover the whole taxonomy;
   val/test must not carry classes train never saw.
2. ``Task.dump_predictions`` — the end-of-serial npz dump (logits + top-k +
   ground truth), row-aligned with the split's clip order.
3. ``train_network`` return contract — now ``(model, val_at_best)`` with the
   per-class val F1 snapshot at the best-macro epoch.

CPU-only; no /scratch. The npz schema smoke for the bst_infer --fe post-hoc
path lives in tests/test_inference_smoke.py.

Run from repo root::

    pytest tests/test_train_surface.py -v
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import main_on_shuttleset.bst_train as bt
from pipeline.config import Taxonomy
from main_on_shuttleset.bst_common import build_bst_network
from preparing_data.shuttleset_dataset import Dataset_npy_collated


TAX3 = Taxonomy(
    name='surface3', classes=('a', 'b', 'c'), merge_map=None,
    has_sides=False, excluded_base_stroke_types=frozenset(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_with_labels(taxonomy, train, val, test) -> bt.Task:
    """A Task carrying loaders whose datasets expose only ``.labels``.

    Enough for ``_assert_label_coverage``, which only reads
    ``self.*_loader.dataset.labels``.
    """
    task = bt.Task.__new__(bt.Task)
    task.taxonomy = taxonomy

    def loader(lbls):
        ds = types.SimpleNamespace(labels=np.array(lbls, dtype=np.int64))
        return types.SimpleNamespace(dataset=ds)

    task.train_loader = loader(train)
    task.val_loader = loader(val)
    task.test_loader = loader(test)
    return task


def _make_collation(
    root: Path,
    *,
    n_bones: int,
    labels: dict[str, list[int]],
    videos_len: dict[str, list[int]] | None = None,
) -> Path:
    """Write a tiny JnB_bone collation under ``root/coll`` with clip_stems.

    :param labels: per-split label lists; split set = its keys.
    :param videos_len: optional per-split video lengths; a 0 marks a clip the
        dataset will drop at load. Defaults to all-100 (nothing dropped).
    :return: the collation root (``root/coll``).
    """
    coll = root / 'coll'
    j_plus_b = 17 + n_bones
    for split, lbls in labels.items():
        n = len(lbls)
        sd = coll / split
        sd.mkdir(parents=True)
        rng = np.random.default_rng(0)
        np.save(sd / 'JnB_bone.npy', rng.standard_normal((n, 100, 2, j_plus_b, 2)).astype(np.float32))
        np.save(sd / 'pos.npy', rng.standard_normal((n, 100, 2, 2)).astype(np.float32))
        np.save(sd / 'shuttle.npy', rng.standard_normal((n, 100, 2)).astype(np.float32))
        vlen = (videos_len or {}).get(split)
        vlen = np.array(vlen, dtype=np.int64) if vlen is not None else np.full(n, 100, dtype=np.int64)
        np.save(sd / 'videos_len.npy', vlen)
        np.save(sd / 'labels.npy', np.array(lbls, dtype=np.int64))
        stems = np.array([f'{split}_clip_{i}' for i in range(n)], dtype=object)
        np.save(sd / 'clip_stems.npy', stems, allow_pickle=True)
    return coll


# ---------------------------------------------------------------------------
# 1. _assert_label_coverage
# ---------------------------------------------------------------------------

def test_assert_label_coverage_passes_full_train_subset_val_test():
    """Train covering every class, with val/test subsets, passes silently."""
    task = _task_with_labels(TAX3, train=[0, 1, 2, 0], val=[0, 1], test=[2, 0])
    task._assert_label_coverage()  # no raise


def test_assert_label_coverage_fails_when_train_misses_a_class():
    """A taxonomy class absent from train raises, naming the missing class."""
    task = _task_with_labels(TAX3, train=[0, 1, 0], val=[0, 1], test=[0, 1])
    with pytest.raises(ValueError) as exc:
        task._assert_label_coverage()
    msg = str(exc.value)
    assert 'train covers only' in msg
    assert "'c'" in msg  # the missing class is named
    assert '[2]' in msg  # ...and its index


def test_assert_label_coverage_fails_on_rogue_eval_label():
    """An out-of-range eval label is flagged without IndexError on naming."""
    tax2 = Taxonomy(
        name='surface2', classes=('a', 'b'), merge_map=None,
        has_sides=False, excluded_base_stroke_types=frozenset(),
    )
    # Train covers all in-range classes (0, 1); test carries a corrupt index 5.
    task = _task_with_labels(tax2, train=[0, 1], val=[0], test=[0, 1, 5])
    with pytest.raises(ValueError) as exc:
        task._assert_label_coverage()
    msg = str(exc.value)
    assert 'absent from train' in msg
    assert '<oob:5>' in msg  # OOB-safe naming, not an IndexError


# ---------------------------------------------------------------------------
# 2. Task.dump_predictions npz schema + row alignment
# ---------------------------------------------------------------------------

NPZ_FIELDS = {
    'logits', 'y_true', 'y_pred_top1', 'topk_idx', 'clip_stems',
    'class_list', 'run_id', 'serial_no', 'taxonomy_name',
}


def test_dump_predictions_writes_all_splits_with_schema(tmp_path):
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=TAX3.n_classes, seq_len=100, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)
    coll = _make_collation(
        tmp_path, n_bones=n_bones,
        labels={'train': [0, 1, 2, 0, 1, 2], 'val': [0, 1, 2, 0], 'test': [2, 1, 0]},
    )

    task = bt.Task.__new__(bt.Task)
    task.taxonomy, task.device, task.net = TAX3, 'cpu', net
    task.train_loader = DataLoader(Dataset_npy_collated(coll, 'train', 'JnB_bone'), batch_size=4, shuffle=True)
    task.val_loader = DataLoader(Dataset_npy_collated(coll, 'val', 'JnB_bone'), batch_size=4)
    task.test_loader = DataLoader(Dataset_npy_collated(coll, 'test', 'JnB_bone'), batch_size=4, shuffle=False)

    run_dir = tmp_path / 'run_xyz'
    run_dir.mkdir()
    task.dump_predictions(run_dir=run_dir, serial_no=2, k=5)

    for split, expected_n in (('train', 6), ('val', 4), ('test', 3)):
        npz = np.load(run_dir / 'predictions' / f'{split}_serial_2.npz', allow_pickle=True)
        assert set(npz.files) == NPZ_FIELDS, npz.files
        assert npz['logits'].shape == (expected_n, 3)
        assert npz['y_true'].shape == (expected_n,)
        assert npz['y_pred_top1'].shape == (expected_n,)
        # k=5 requested but head is 3, so topk clamps to 3.
        assert npz['topk_idx'].shape == (expected_n, 3)
        assert list(npz['class_list']) == ['a', 'b', 'c']
        assert str(npz['run_id']) == 'run_xyz'
        assert int(npz['serial_no']) == 2
        assert str(npz['taxonomy_name']) == 'surface3'


def test_dump_predictions_test_rows_align_with_labels(tmp_path):
    """shuffle=False dump keeps y_true in the on-disk labels order (row-aligned
    with clip_stems), so a downstream row->stem join is valid."""
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=TAX3.n_classes, seq_len=100, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)
    test_labels = [2, 0, 1, 2, 0]
    coll = _make_collation(
        tmp_path, n_bones=n_bones,
        labels={'train': [0, 1, 2], 'val': [0], 'test': test_labels},
    )
    task = bt.Task.__new__(bt.Task)
    task.taxonomy, task.device, task.net = TAX3, 'cpu', net
    task.train_loader = DataLoader(Dataset_npy_collated(coll, 'train', 'JnB_bone'), batch_size=4)
    task.val_loader = DataLoader(Dataset_npy_collated(coll, 'val', 'JnB_bone'), batch_size=4)
    task.test_loader = DataLoader(Dataset_npy_collated(coll, 'test', 'JnB_bone'), batch_size=4)

    run_dir = tmp_path / 'run_align'
    run_dir.mkdir()
    task.dump_predictions(run_dir=run_dir, serial_no=1, k=2)
    npz = np.load(run_dir / 'predictions' / 'test_serial_1.npz', allow_pickle=True)
    assert npz['y_true'].tolist() == test_labels


def test_dump_predictions_clip_stems_survive_zero_length_drop(tmp_path):
    """The npz clip_stems follow the post-drop dataset order, not the raw
    on-disk clip_stems.npy. A clip the dataset drops (videos_len==0) vanishes
    from the npz, keeping every later row's stem aligned with its prediction.
    """
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=TAX3.n_classes, seq_len=100, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)
    # test split: clip index 1 is zero-length -> dropped at load.
    coll = _make_collation(
        tmp_path, n_bones=n_bones,
        labels={'train': [0, 1, 2], 'val': [0, 1, 2], 'test': [0, 1, 2, 0]},
        videos_len={'test': [100, 0, 100, 100]},
    )
    task = bt.Task.__new__(bt.Task)
    task.taxonomy, task.device, task.net = TAX3, 'cpu', net
    task.train_loader = DataLoader(Dataset_npy_collated(coll, 'train', 'JnB_bone'), batch_size=4)
    task.val_loader = DataLoader(Dataset_npy_collated(coll, 'val', 'JnB_bone'), batch_size=4)
    task.test_loader = DataLoader(Dataset_npy_collated(coll, 'test', 'JnB_bone'), batch_size=4)

    run_dir = tmp_path / 'run_drop'
    run_dir.mkdir()
    task.dump_predictions(run_dir=run_dir, serial_no=1, k=2)
    npz = np.load(run_dir / 'predictions' / 'test_serial_1.npz', allow_pickle=True)

    # Clip 1 (zero-length) dropped: survivors are clips 0, 2, 3, in order.
    assert npz['clip_stems'].tolist() == ['test_clip_0', 'test_clip_2', 'test_clip_3']
    assert len(npz['clip_stems']) == len(npz['y_true']) == 3
    # The raw on-disk sidecar still has all 4, so a naive on-disk index-join
    # would have misaligned every row from the dropped clip onward.
    on_disk = np.load(coll / 'test' / 'clip_stems.npy', allow_pickle=True)
    assert len(on_disk) == 4
    assert npz['clip_stems'].tolist() != on_disk.tolist()


def test_dump_predictions_clip_stems_track_train_partial_reorder(tmp_path):
    """train_partial<1 regroups the train set by class; the npz clip_stems track
    that reorder (sourced from the in-memory dataset) and stay aligned with
    y_true. Covers the second rearrangement, so the dump contract holds for a
    future data-scaling ablation, not just the default train_partial=1.0 run.
    """
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=TAX3.n_classes, seq_len=100, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)
    coll = _make_collation(
        tmp_path, n_bones=n_bones,
        labels={'train': [0, 1, 2, 0, 1, 2, 0, 1], 'val': [0, 1, 2], 'test': [0]},
    )
    # train_partial=0.5 -> adjust_to_partial_train_set groups by class + halves.
    train_ds = Dataset_npy_collated(coll, 'train', 'JnB_bone', train_partial=0.5)
    task = bt.Task.__new__(bt.Task)
    task.taxonomy, task.device, task.net = TAX3, 'cpu', net
    task.train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    task.val_loader = DataLoader(Dataset_npy_collated(coll, 'val', 'JnB_bone'), batch_size=4)
    task.test_loader = DataLoader(Dataset_npy_collated(coll, 'test', 'JnB_bone'), batch_size=4)

    run_dir = tmp_path / 'run_partial'
    run_dir.mkdir()
    task.dump_predictions(run_dir=run_dir, serial_no=1, k=2)
    npz = np.load(run_dir / 'predictions' / 'train_serial_1.npz', allow_pickle=True)

    # The npz mirrors the in-memory (post-adjust) dataset exactly, both the
    # reordered stems and the matching labels.
    assert npz['clip_stems'].tolist() == train_ds.clip_stems.tolist()
    assert npz['y_true'].tolist() == train_ds.labels.tolist()
    assert len(npz['clip_stems']) == len(npz['y_true'])


# ---------------------------------------------------------------------------
# 3. train_network return contract: (model, val_at_best)
# ---------------------------------------------------------------------------

def test_train_network_returns_model_and_val_at_best(tmp_path, monkeypatch):
    """A short real train returns ``(model, val_at_best)``; the snapshot, when
    present, is a dict with an epoch + a per-class F1 map over present classes.
    """
    # Tiny, deterministic Hyp: plain CE, 2 epochs, no aux schedule. train_network
    # reads the module-global hyp, so patch it for the duration of the call.
    monkeypatch.setattr(bt, 'hyp', bt.hyp._replace(
        n_epochs=2, early_stop_n_epochs=10, warm_up_step=1,
        adaptive_focal=None, class_weights=None, label_smoothing=0.0,
        use_aux_schedule=False, pose_style='JnB_bone',
        augmentation={'p_flip': 0.0, 'p_jitter': 0.0, 'cap_y': 0.05, 'cap_x': 0.10, 'eps': 0.15},
    ))
    torch.manual_seed(0)
    net, n_bones = build_bst_network(
        'BST_CG_AP', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=TAX3.n_classes, seq_len=100, device='cpu',
    )
    coll = _make_collation(
        tmp_path, n_bones=n_bones,
        labels={'train': [0, 1, 2, 0, 1, 2, 0, 1], 'val': [0, 1, 2, 0]},
    )
    train_loader = DataLoader(Dataset_npy_collated(coll, 'train', 'JnB_bone'), batch_size=4, shuffle=True)
    val_loader = DataLoader(Dataset_npy_collated(coll, 'val', 'JnB_bone'), batch_size=4)

    result = bt.train_network(
        model=net, train_loader=train_loader, val_loader=val_loader, device='cpu',
        save_path=tmp_path / 'w.pt', n_bones=n_bones, n_classes=TAX3.n_classes,
        class_ls=list(TAX3.classes), taxonomy=TAX3, tb_dir=tmp_path / 'tb',
    )
    assert isinstance(result, tuple) and len(result) == 2
    model, val_at_best = result
    assert (tmp_path / 'w.pt').exists()  # best checkpoint saved

    # With 2 epochs over a 3-class val, some epoch beats the macro=0.0 init, so
    # val_at_best is populated. Guard the None case anyway (degenerate runs).
    if val_at_best is not None:
        assert set(val_at_best) == {'epoch', 'per_class_f1'}
        assert isinstance(val_at_best['per_class_f1'], dict)
        for cls, f1 in val_at_best['per_class_f1'].items():
            assert cls in TAX3.classes
            assert isinstance(f1, float)
