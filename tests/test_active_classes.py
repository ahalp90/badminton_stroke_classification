"""Tests for the unknown-ghost-channel fix (post-strip implementation).

Covers seven sections matching the per-file changes in §6 of
``scratch/architecture_notes/unknown_channel_fix_review.md``:

1. Taxonomy method correctness (``has_unknown``, ``active_class_list``,
   ``full_to_active_remap``).
2. ``derive_active_classes_from_labels`` happy paths and rejection paths.
3. ``_validate_and_record_arch`` manifest write + resume cross-check +
   ``expected_active_classes`` lever.
4. BST_CG_AP forward+backward smoke at the active head dim.
5. ``class_weights`` renormalisation under active classes (pair-balanced
   sanity + Strip 3 message shape).
6. Real-labels probe (auto-skipped when ``/scratch/comp320a/...`` not
   visible).
7. ``bst_infer`` Strip 1 contract (required arch kwargs).

CPU-only. Run from repo root::

    pytest tests/test_active_classes.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from pipeline.config import TAXONOMIES, Taxonomy
from main_on_shuttleset.bst_common import (
    build_bst_network,
    derive_active_classes_from_labels,
)
from main_on_shuttleset.bst_train import _validate_and_record_arch
from main_on_shuttleset.bst_infer import Task as InferTask


REAL_TAXONOMY_NAMES = ['merged_25', 'une_merge_v1', 'une_merge_v1_nosides', 'raw_35']


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_no_unknown_taxonomy() -> Taxonomy:
    return Taxonomy(
        name='no_unknown_test',
        merge_map=None,
        base_types=('a', 'b', 'c'),
        standalone_types=(),
        unknown_first=False,
    )


def _full_present(tax: Taxonomy) -> set[int]:
    return set(range(tax.n_classes))


def _present_minus_unknown(tax: Taxonomy) -> set[int]:
    full = tax.class_list()
    return _full_present(tax) - {full.index('unknown')}


def _make_synthetic_labels(present: set[int], n_per_class: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(sorted(present), n_per_class)
    rng.shuffle(labels)
    return labels.astype(np.int64)


class _FakeTask:
    def __init__(self, n_active_classes, active_class_list):
        self.n_active_classes = n_active_classes
        self.active_class_list = active_class_list


class _FakeHyp:
    def __init__(self, expected_active_classes=None):
        self.expected_active_classes = expected_active_classes


class _NoopTee:
    def write(self, _data): pass
    def flush(self): pass


def _seed_manifest(run_dir: Path, extra: dict | None = None) -> Path:
    manifest = {'run_id': run_dir.name, 'config': {}, 'serials': []}
    if extra is not None:
        manifest['extra'] = extra
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'manifest.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Section 1: Taxonomy methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_taxonomy_has_unknown(tax_name):
    assert TAXONOMIES[tax_name].has_unknown is True


def test_synthetic_no_unknown_has_unknown_false(synthetic_no_unknown_taxonomy):
    assert synthetic_no_unknown_taxonomy.has_unknown is False


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_full_present_is_identity(tax_name):
    tax = TAXONOMIES[tax_name]
    assert tax.active_class_list(_full_present(tax)) == tax.class_list()


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_minus_unknown(tax_name):
    tax = TAXONOMIES[tax_name]
    active = tax.active_class_list(_present_minus_unknown(tax))
    assert 'unknown' not in active
    assert len(active) == tax.n_classes - 1


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_preserves_relative_order(tax_name):
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    active = tax.active_class_list(_present_minus_unknown(tax))
    assert active == [n for n in full if n != 'unknown']


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_arbitrary_subset(tax_name):
    """Arbitrary subset returns those classes in original order."""
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    subset = {0, 2, len(full) - 1}
    assert tax.active_class_list(subset) == [full[i] for i in sorted(subset)]


def test_active_class_list_out_of_range_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    with pytest.raises(ValueError, match='out of range'):
        tax.active_class_list({99})


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_remap_full_present_is_identity(tax_name):
    tax = TAXONOMIES[tax_name]
    assert tax.full_to_active_remap(_full_present(tax)) == list(range(tax.n_classes))


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_remap_minus_unknown_sentinel_at_unknown_idx(tax_name):
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    unknown_idx = full.index('unknown')
    remap = tax.full_to_active_remap(_present_minus_unknown(tax))
    assert remap[unknown_idx] == -1
    active_values = sorted(v for v in remap if v >= 0)
    assert active_values == list(range(tax.n_classes - 1))


def test_active_class_list_empty_present():
    tax = TAXONOMIES['une_merge_v1_nosides']
    assert tax.active_class_list(set()) == []


def test_remap_empty_present_all_sentinel():
    tax = TAXONOMIES['une_merge_v1_nosides']
    assert tax.full_to_active_remap(set()) == [-1] * tax.n_classes


def test_unknown_position_per_taxonomy():
    """Document the BST-paper convention split via assert."""
    for tax_name in ('merged_25', 'une_merge_v1'):
        assert TAXONOMIES[tax_name].class_list()[0] == 'unknown', tax_name
    for tax_name in ('une_merge_v1_nosides', 'raw_35'):
        assert TAXONOMIES[tax_name].class_list()[-1] == 'unknown', tax_name


# ---------------------------------------------------------------------------
# Section 2: derive_active_classes_from_labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_full_present_train_only(tax_name):
    tax = TAXONOMIES[tax_name]
    train = _make_synthetic_labels(_full_present(tax))
    active, remap, out = derive_active_classes_from_labels(tax, train)
    assert active == tax.class_list()
    assert remap == list(range(tax.n_classes))
    np.testing.assert_array_equal(out['train'], train)


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_minus_unknown_train_only(tax_name):
    tax = TAXONOMIES[tax_name]
    train = _make_synthetic_labels(_present_minus_unknown(tax))
    active, _remap, out = derive_active_classes_from_labels(tax, train)
    assert 'unknown' not in active
    assert len(active) == tax.n_classes - 1
    assert (out['train'] >= 0).all()
    assert (out['train'] < len(active)).all()


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_train_with_val_test_subset_pass(tax_name):
    tax = TAXONOMIES[tax_name]
    present = _present_minus_unknown(tax)
    train = _make_synthetic_labels(present, seed=0)
    val   = _make_synthetic_labels(present, seed=1)
    test  = _make_synthetic_labels(present, seed=2)
    active, _remap, out = derive_active_classes_from_labels(
        tax, train_labels=train,
        validation_label_arrays={'val': val, 'test': test},
    )
    assert set(out.keys()) == {'train', 'val', 'test'}
    for split, arr in out.items():
        assert (arr >= 0).all() and (arr < len(active)).all(), split


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_val_has_class_absent_from_train_raises(tax_name):
    tax = TAXONOMIES[tax_name]
    unknown_idx = tax.class_list().index('unknown')
    train = _make_synthetic_labels(_present_minus_unknown(tax), seed=0)
    val = np.concatenate([
        _make_synthetic_labels(_present_minus_unknown(tax), seed=1),
        np.asarray([unknown_idx], dtype=np.int64),
    ])
    test = _make_synthetic_labels(_present_minus_unknown(tax), seed=2)
    with pytest.raises(ValueError, match=r'\[val\] contains class indices absent from train'):
        derive_active_classes_from_labels(
            tax, train_labels=train,
            validation_label_arrays={'val': val, 'test': test},
        )


def test_derive_test_has_class_absent_from_train_raises():
    tax = TAXONOMIES['merged_25']
    unknown_idx = tax.class_list().index('unknown')
    train = _make_synthetic_labels(_present_minus_unknown(tax), seed=0)
    val   = _make_synthetic_labels(_present_minus_unknown(tax), seed=1)
    test = np.concatenate([
        _make_synthetic_labels(_present_minus_unknown(tax), seed=2),
        np.asarray([unknown_idx], dtype=np.int64),
    ])
    with pytest.raises(ValueError, match=r'\[test\] contains class indices absent from train'):
        derive_active_classes_from_labels(
            tax, train_labels=train,
            validation_label_arrays={'val': val, 'test': test},
        )


def test_derive_empty_train_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    with pytest.raises(ValueError, match='train_labels is empty'):
        derive_active_classes_from_labels(
            tax, train_labels=np.asarray([], dtype=np.int64),
        )


def test_derive_no_validation_arrays_works():
    tax = TAXONOMIES['une_merge_v1_nosides']
    train = _make_synthetic_labels(_full_present(tax))
    _active, _remap, out = derive_active_classes_from_labels(tax, train)
    assert set(out.keys()) == {'train'}


def test_derive_corrupted_label_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    bad_train = np.asarray([0, 1, 99], dtype=np.int64)
    with pytest.raises(ValueError, match='out of range'):
        derive_active_classes_from_labels(tax, bad_train)


def test_derive_no_unknown_taxonomy_round_trip(synthetic_no_unknown_taxonomy):
    tax = synthetic_no_unknown_taxonomy
    train = np.asarray([0, 1, 2, 3, 4, 5] * 3, dtype=np.int64)
    active, remap, out = derive_active_classes_from_labels(tax, train)
    assert active == tax.class_list()
    assert remap == list(range(tax.n_classes))
    np.testing.assert_array_equal(out['train'], train)


# ---------------------------------------------------------------------------
# Section 3: _validate_and_record_arch
# ---------------------------------------------------------------------------

def test_validate_and_record_arch_writes_manifest_block(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=None)
    _seed_manifest(tmp_path)

    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=None, tee=_NoopTee(),
    )

    with open(tmp_path / 'manifest.yaml') as f:
        manifest = yaml.safe_load(f)
    arch = manifest['extra']['arch']
    assert arch['n_classes_full']    == tax.n_classes
    assert arch['n_active_classes']  == len(active)
    assert arch['has_unknown']       is True
    assert arch['unknown_first']     is False
    assert arch['active_class_list'] == active


def test_validate_and_record_arch_resume_match(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior = {
        'arch': {
            'n_classes_full':    tax.n_classes,
            'n_active_classes':  len(active),
            'has_unknown':       True,
            'unknown_first':     False,
            'active_class_list': active,
        },
    }
    _seed_manifest(tmp_path, extra=prior)
    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=prior['arch'], tee=_NoopTee(),
    )


def test_validate_and_record_arch_resume_n_active_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior_arch = {
        'n_active_classes':  len(active) + 1,
        'active_class_list': ['fake'] * (len(active) + 1),
    }
    _seed_manifest(tmp_path, extra={'arch': prior_arch})
    with pytest.raises(ValueError, match='Resume manifest disagrees'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=prior_arch, tee=_NoopTee(),
        )


def test_validate_and_record_arch_resume_list_order_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior_arch = {
        'n_active_classes':  len(active),
        'active_class_list': list(reversed(active)),
    }
    _seed_manifest(tmp_path, extra={'arch': prior_arch})
    with pytest.raises(ValueError, match='Resume manifest disagrees'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=prior_arch, tee=_NoopTee(),
        )


def test_validate_and_record_arch_expected_match_passes(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=active)
    _seed_manifest(tmp_path)
    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=None, tee=_NoopTee(),
    )


def test_validate_and_record_arch_expected_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=['something_else'])
    _seed_manifest(tmp_path)
    with pytest.raises(ValueError, match='expected_active_classes'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=None, tee=_NoopTee(),
        )


# ---------------------------------------------------------------------------
# Section 4: BST_CG_AP forward+backward smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
@pytest.mark.parametrize('present_kind', ['all_classes', 'no_unknown'])
def test_bst_forward_backward_under_active_classes(tax_name, present_kind):
    torch.manual_seed(0)
    tax = TAXONOMIES[tax_name]
    n_active = tax.n_classes if present_kind == 'all_classes' else tax.n_classes - 1

    pose_style = 'JnB_bone'
    seq_len = 100
    batch_size = 4
    n_joints = 17
    in_channels = 2

    net, n_bones = build_bst_network(
        model_name='BST_CG_AP',
        n_joints=n_joints, pose_style=pose_style, in_channels=in_channels,
        n_class=n_active, seq_len=seq_len, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)

    j_plus_b = n_joints + n_bones
    human_pose = torch.randn(batch_size, seq_len, 2, j_plus_b, in_channels)
    human_pose_flat = human_pose.view(*human_pose.shape[:-2], -1)
    pos = torch.randn(batch_size, seq_len, 2, 2)
    shuttle = torch.randn(batch_size, seq_len, 2)
    video_len = torch.full((batch_size,), seq_len, dtype=torch.long)
    labels = torch.randint(0, n_active, (batch_size,))

    logits = net(human_pose_flat, shuttle, pos, video_len)
    assert logits.shape == (batch_size, n_active)

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    grad_count = sum(
        1 for p in net.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0


# ---------------------------------------------------------------------------
# Section 5: class_weights renormalisation under active classes
# ---------------------------------------------------------------------------

def test_class_weights_renorm_pair_balanced():
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    n_active = len(active)
    assert 'unknown' not in active

    class_weights = {'wrist_smash': 2.0, 'smash': 2.0}
    weights = torch.ones(n_active)
    for cls_name, mult in class_weights.items():
        weights[active.index(cls_name)] = mult
    weights = weights * (n_active / weights.sum())

    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    for cls_name in class_weights:
        assert weights[active.index(cls_name)] > 1.0


def test_class_weights_uniform_when_empty():
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    n_active = len(active)
    weights = torch.ones(n_active)
    weights = weights * (n_active / weights.sum())
    assert torch.allclose(weights, torch.ones(n_active))


def test_class_weights_strip3_message_shape():
    """Strip 3: error message names 'not in active class list', the active
    list, and both the active-list size and full-taxonomy size for the
    user to debug."""
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    cls_name = 'unknown'

    msg = (
        f"class_weights key '{cls_name}' not in active class list "
        f"{active}. (Full taxonomy {tax.name!r} has "
        f"{len(tax.class_list())} classes; this run uses {len(active)}.)"
    )
    assert 'not in active class list' in msg
    assert f'this run uses {len(active)}' in msg
    assert f'has {len(tax.class_list())} classes' in msg


# ---------------------------------------------------------------------------
# Section 6: real-labels probe
# ---------------------------------------------------------------------------

CANDIDATE_REAL_DIRS = [
    '/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/'
    'npy_une_merge_v1_nosides_split_v2_dropunk',
    '/scratch/comp320a/ShuttleSet_data_une_merge_v1/'
    'npy_une_merge_v1_split_v2_dropunk',
    '/scratch/comp320a/ShuttleSet_data_merged_25/'
    'npy_merged_25_split_bst_baseline_dropunk',
]

EXPECTED_N_ACTIVE_PER_DIR = {
    'npy_une_merge_v1_nosides_split_v2_dropunk': 14,  # nosides collapses Top_/Bottom_
    'npy_une_merge_v1_split_v2_dropunk':         28,  # 14 base * Top_/Bottom_ sides; unknown empty
    'npy_merged_25_split_bst_baseline_dropunk':  25,  # 12 base * 2 sides + unknown (driven_flight populates it)
}


@pytest.mark.parametrize('dir_path', CANDIDATE_REAL_DIRS)
def test_real_labels_npy_remap_into_active_range(dir_path):
    root = Path(dir_path)
    if not root.exists():
        pytest.skip(f'{dir_path} not visible from this host')

    name = root.name
    expected_n_active = EXPECTED_N_ACTIVE_PER_DIR[name]

    tax_name = None
    for cand in sorted(REAL_TAXONOMY_NAMES, key=len, reverse=True):
        if name.startswith(f'npy_{cand}_'):
            tax_name = cand
            break
    assert tax_name is not None, f'could not infer taxonomy from {name!r}'

    tax = TAXONOMIES[tax_name]
    train = np.load(str(root / 'train' / 'labels.npy'))
    val   = np.load(str(root / 'val'   / 'labels.npy'))
    test  = np.load(str(root / 'test'  / 'labels.npy'))

    active, _remap, out = derive_active_classes_from_labels(
        tax, train_labels=train,
        validation_label_arrays={'val': val, 'test': test},
    )
    assert len(active) == expected_n_active, (
        f'{name}: expected n_active={expected_n_active} got {len(active)}'
    )
    n_active = len(active)
    for split, arr in out.items():
        assert (arr >= 0).all() and (arr < n_active).all(), split


# ---------------------------------------------------------------------------
# Section 7: bst_infer Strip 1 contract
# ---------------------------------------------------------------------------

def test_bst_infer_get_network_architecture_requires_arch_kwargs():
    """Strip 1: bst_infer's get_network_architecture must require both
    n_active_classes and active_class_list. No silent fallback."""
    task = InferTask(n_joints=17)
    task.pose_style = 'JnB_bone'
    with pytest.raises(TypeError):
        task.get_network_architecture(
            model_name='BST_CG_AP', seq_len=100, in_channels=2,
            taxonomy=TAXONOMIES['une_merge_v1_nosides'],
        )
