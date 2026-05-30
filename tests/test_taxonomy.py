"""Tests for the contractual Taxonomy + label_for_row pipeline.

Post-refactor: the 'active vs full' classes distinction is gone. Each Taxonomy
commits its full class list directly; labels.npy lands in active class space;
the collator and ``_derive_class_label`` route through ``label_for_row``.

Coverage:
1. Taxonomy structure (n_classes, has_unknown, __post_init__ contract).
2. resolve_taxonomy + TAXONOMY_ALIASES.
3. label_for_row (merge_map, side-prefixing, excluded_base_stroke_types).
4. _sided_classes helper.
5. BST_CG_AP forward+backward smoke at each taxonomy's n_classes.
6. class_weights renormalisation across the head.
7. Real labels.npy probe (auto-skipped when /scratch/comp320a/... not visible).

Tests for ``Task._assert_label_coverage`` live with the train-surface commit
(Step D). The inference npz schema smoke + FE handler resolver tests live in
``tests/test_inference_smoke.py``, ``tests/test_api_registry.py``, and
``tests/test_api_inference.py`` (added with later commits). CPU-only.

Run from repo root::

    pytest tests/test_taxonomy.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pipeline.config import (
    SIDE_AGNOSTIC_TYPES,
    TAXONOMIES,
    TAXONOMY_ALIASES,
    TAXONOMY_BST_12,
    TAXONOMY_BST_24,
    TAXONOMY_BST_25,
    TAXONOMY_SHUTTLESET_18,
    TAXONOMY_UNE_V1_14,
    TAXONOMY_UNE_V1_15,
    Taxonomy,
    _sided_classes,  # noqa: F401  # private helper, tested below
    collation_id_from_manifest,
    derive_npy_collated_dir_basename,
    label_for_row,
    resolve_taxonomy,
)
from main_on_shuttleset.bst_common import build_bst_network


REAL_TAXONOMY_OBJECTS = [
    TAXONOMY_BST_25, TAXONOMY_BST_24, TAXONOMY_BST_12,
    TAXONOMY_UNE_V1_14, TAXONOMY_UNE_V1_15, TAXONOMY_SHUTTLESET_18,
]


# ---------------------------------------------------------------------------
# Section 1: Taxonomy structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax', REAL_TAXONOMY_OBJECTS, ids=lambda t: t.name)
def test_taxonomy_n_classes_matches_classes_length(tax):
    assert tax.n_classes == len(tax.classes)


@pytest.mark.parametrize('tax', REAL_TAXONOMY_OBJECTS, ids=lambda t: t.name)
def test_taxonomy_unknown_at_minus_one_when_present(tax):
    """When unknown is in the class list, it must sit at index -1."""
    assert ('unknown' not in tax.classes) or (tax.classes[-1] == 'unknown')


@pytest.mark.parametrize('tax', REAL_TAXONOMY_OBJECTS, ids=lambda t: t.name)
def test_taxonomy_has_unknown_property(tax):
    assert tax.has_unknown == ('unknown' in tax.classes)


def test_taxonomy_n_classes_expected_per_taxonomy():
    """Sanity check the six registered taxonomies have the documented sizes."""
    assert TAXONOMY_BST_25.n_classes == 25
    assert TAXONOMY_BST_24.n_classes == 24
    assert TAXONOMY_BST_12.n_classes == 12
    assert TAXONOMY_UNE_V1_14.n_classes == 14
    assert TAXONOMY_UNE_V1_15.n_classes == 15
    assert TAXONOMY_SHUTTLESET_18.n_classes == 18


def test_taxonomy_post_init_rejects_unknown_mid_list():
    """A Taxonomy with unknown anywhere but index -1 must raise on construction."""
    with pytest.raises(ValueError, match='unknown must sit at index -1'):
        Taxonomy(
            name='bad_middle',
            classes=('a', 'unknown', 'b'),
            merge_map=None,
            has_sides=False,
            excluded_base_stroke_types=frozenset(),
        )


def test_taxonomy_post_init_rejects_unknown_at_index_zero():
    """The historical 'unknown at index 0' (BST paper convention) must also raise."""
    with pytest.raises(ValueError, match='unknown must sit at index -1'):
        Taxonomy(
            name='bad_first',
            classes=('unknown', 'a', 'b'),
            merge_map=None,
            has_sides=False,
            excluded_base_stroke_types=frozenset(),
        )


def test_taxonomy_post_init_accepts_unknown_at_minus_one():
    """A Taxonomy with unknown at index -1 must construct without error."""
    tax = Taxonomy(
        name='ok_last',
        classes=('a', 'b', 'unknown'),
        merge_map=None,
        has_sides=False,
        excluded_base_stroke_types=frozenset(),
    )
    assert tax.has_unknown is True


def test_taxonomy_post_init_accepts_no_unknown():
    """A Taxonomy without unknown anywhere is always fine."""
    tax = Taxonomy(
        name='ok_no_unknown',
        classes=('a', 'b', 'c'),
        merge_map=None,
        has_sides=False,
        excluded_base_stroke_types=frozenset({'unknown'}),
    )
    assert tax.has_unknown is False


# ---------------------------------------------------------------------------
# Section 2: resolve_taxonomy + TAXONOMY_ALIASES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('alias,canonical', sorted(TAXONOMY_ALIASES.items()))
def test_resolve_taxonomy_aliases(alias, canonical):
    """Each alias key resolves to its canonical Taxonomy object."""
    resolved = resolve_taxonomy(alias)
    assert resolved is TAXONOMIES[canonical]
    assert resolved.name == canonical


@pytest.mark.parametrize('name', sorted(TAXONOMIES))
def test_resolve_taxonomy_canonical_names(name):
    """Canonical names round-trip through resolve_taxonomy."""
    resolved = resolve_taxonomy(name)
    assert resolved is TAXONOMIES[name]


def test_resolve_taxonomy_unknown_name_raises():
    """A name that's neither canonical nor an alias raises KeyError with context."""
    with pytest.raises(KeyError, match='not registered and not aliased'):
        resolve_taxonomy('not_a_real_taxonomy_xyz')


# ---------------------------------------------------------------------------
# Section 3: label_for_row
# ---------------------------------------------------------------------------

# Parametrize tuples covering: the driven_flight -> drive headline fix,
# side-prefixing rule, excluded_base_stroke_types behaviour, side-agnostic
# types, no-merge taxonomies. Expected_class is the string class name (None
# means the row is filtered out).
LABEL_FOR_ROW_CASES = [
    # bst_25: driven_flight -> drive (the MERGE_MAP_25 paper-faithful fix vs
    # the legacy buggy driven_flight -> unknown convention).
    ('bst_25', 'driven_flight', 'Top',    'Top_drive'),
    ('bst_25', 'driven_flight', 'Bottom', 'Bottom_drive'),
    # bst_25: unknown unprefixed (SIDE_AGNOSTIC_TYPES).
    ('bst_25', 'unknown',       'Top',    'unknown'),
    ('bst_25', 'unknown',       'Bottom', 'unknown'),
    # bst_25: smash gets Top_/Bottom_ prefix.
    ('bst_25', 'smash',         'Top',    'Top_smash'),
    ('bst_25', 'smash',         'Bottom', 'Bottom_smash'),
    # bst_25: wrist_smash merges to smash, then gets sided.
    ('bst_25', 'wrist_smash',   'Top',    'Top_smash'),
    # bst_25: back_court_drive merges to drive, then gets sided.
    ('bst_25', 'back_court_drive', 'Top', 'Top_drive'),
    # bst_24: excludes unknown (None return for unknown rows).
    ('bst_24', 'unknown',       'Top',    None),
    ('bst_24', 'driven_flight', 'Top',    'Top_drive'),
    # bst_12: no sides; driven_flight still merges to drive.
    ('bst_12', 'driven_flight', 'Top',    'drive'),
    ('bst_12', 'smash',         'Top',    'smash'),
    ('bst_12', 'unknown',       'Top',    None),
    # une_v1_14: no sides; UNE_MERGE_V1_MAP folds driven_flight to drive.
    ('une_v1_14', 'driven_flight', 'Top', 'drive'),
    ('une_v1_14', 'wrist_smash',   'Top', 'wrist_smash'),  # kept distinct
    ('une_v1_14', 'unknown',       'Top', None),
    # une_v1_15: same as une_v1_14 but keeps unknown.
    ('une_v1_15', 'unknown',       'Top', 'unknown'),
    ('une_v1_15', 'driven_flight', 'Top', 'drive'),
    # shuttleset_18: no merge, no sides, excludes unknown.
    ('shuttleset_18', 'driven_flight', 'Top', 'driven_flight'),
    ('shuttleset_18', 'smash',         'Top', 'smash'),
    ('shuttleset_18', 'unknown',       'Top', None),
]


@pytest.mark.parametrize(
    'tax_name,raw_type,side,expected_class', LABEL_FOR_ROW_CASES,
)
def test_label_for_row_drives_taxonomy(tax_name, raw_type, side, expected_class):
    """label_for_row composes merge_map + side rule + excluded set correctly.

    The driven_flight -> drive cases lock in the MERGE_MAP_25 paper-faithful
    fix vs the legacy buggy 35-class merge convention.
    """
    tax = TAXONOMIES[tax_name]
    idx = label_for_row(tax, raw_type, side)
    if expected_class is None:
        assert idx is None, (
            f'expected None for {tax_name} {raw_type} {side}, got idx {idx}'
        )
    else:
        assert idx is not None, (
            f'expected idx for {tax_name} {raw_type} {side}, got None'
        )
        assert tax.classes[idx] == expected_class, (
            f'{tax_name} {raw_type} {side}: expected {expected_class!r} '
            f'at idx {idx}, got {tax.classes[idx]!r}'
        )


def test_label_for_row_returns_None_for_excluded_unknown():
    """Rows whose raw_type is in excluded_base_stroke_types return None."""
    assert label_for_row(TAXONOMY_BST_24, 'unknown', 'Top') is None
    assert label_for_row(TAXONOMY_UNE_V1_14, 'unknown', 'Top') is None
    assert label_for_row(TAXONOMY_SHUTTLESET_18, 'unknown', 'Top') is None
    assert label_for_row(TAXONOMY_BST_12, 'unknown', 'Top') is None


def test_label_for_row_keeps_unknown_when_not_in_excluded():
    """Unknown comes through unprefixed when the taxonomy keeps it."""
    idx = label_for_row(TAXONOMY_BST_25, 'unknown', 'Top')
    assert idx is not None
    assert TAXONOMY_BST_25.classes[idx] == 'unknown'

    idx = label_for_row(TAXONOMY_UNE_V1_15, 'unknown', 'Bottom')
    assert idx is not None
    assert TAXONOMY_UNE_V1_15.classes[idx] == 'unknown'


def test_side_agnostic_types_constant_contains_unknown():
    """SIDE_AGNOSTIC_TYPES is the canonical 'never prefixed at label time' set."""
    assert 'unknown' in SIDE_AGNOSTIC_TYPES


def test_label_for_row_raises_descriptive_error_on_missing_class():
    """When the derived label_str isn't in taxonomy.classes (misconfigured
    merge_map or class list), label_for_row raises a ValueError naming the
    taxonomy, raw_type, side, and derived label_str. Bare tuple.index would
    just say 'x not in tuple' which is useless when chasing a config bug.
    """
    # Synthetic taxonomy whose merge_map produces a label_str absent from classes.
    tax = Taxonomy(
        name='test_misconfigured',
        classes=('only_one',),
        merge_map={'smash': 'something_else'},  # 'something_else' not in classes
        has_sides=False,
        excluded_base_stroke_types=frozenset(),
    )
    with pytest.raises(ValueError) as exc_info:
        label_for_row(tax, 'smash', 'Top')
    msg = str(exc_info.value)
    assert 'test_misconfigured' in msg, msg
    assert 'something_else' in msg, msg
    assert 'smash' in msg, msg
    assert 'Top' in msg, msg


def test_label_for_row_filters_before_merge():
    """excluded_base_stroke_types fires BEFORE merge_map.

    Build a synthetic taxonomy where a raw type sits in BOTH the excluded set
    AND the merge_map. Confirm label_for_row returns None (filter won), not
    a merged index (merge won). Pins the operation order; a future refactor
    that reversed it would silently keep these rows under the merged label.
    """
    tax = Taxonomy(
        name='test_filter_first',
        classes=('a', 'b', 'c'),
        merge_map={'driven_flight': 'a'},
        has_sides=False,
        excluded_base_stroke_types=frozenset({'driven_flight'}),
    )
    # Both rules apply to driven_flight: exclude says drop, merge says map to 'a'.
    # Filter-first contract: drop wins -> None.
    assert label_for_row(tax, 'driven_flight', 'Top') is None


# ---------------------------------------------------------------------------
# Section 4: _sided_classes helper
# ---------------------------------------------------------------------------

def test_sided_classes_with_unknown():
    """Tail unknown lands at index -1; Top_ block before Bottom_."""
    result = _sided_classes(['a', 'b'], with_unknown=True)
    assert result == ('Top_a', 'Top_b', 'Bottom_a', 'Bottom_b', 'unknown')


def test_sided_classes_without_unknown():
    """No unknown in the output when with_unknown=False."""
    result = _sided_classes(['a', 'b'], with_unknown=False)
    assert result == ('Top_a', 'Top_b', 'Bottom_a', 'Bottom_b')
    assert 'unknown' not in result


def test_sided_classes_empty_base():
    """Empty base + with_unknown gives just ('unknown',)."""
    result = _sided_classes([], with_unknown=True)
    assert result == ('unknown',)


def test_sided_classes_single_base():
    """Single base type still gets both Top_ and Bottom_ entries."""
    result = _sided_classes(['smash'], with_unknown=False)
    assert result == ('Top_smash', 'Bottom_smash')


# ---------------------------------------------------------------------------
# Section 5: BST_CG_AP forward+backward smoke sized to each taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax', REAL_TAXONOMY_OBJECTS, ids=lambda t: t.name)
def test_bst_forward_backward_sized_by_taxonomy(tax):
    """BST_CG_AP forward + backward at each taxonomy's n_classes produces a
    finite loss and non-zero gradients.
    """
    torch.manual_seed(0)

    pose_style = 'JnB_bone'
    seq_len = 100
    batch_size = 4
    n_joints = 17
    in_channels = 2

    net, n_bones = build_bst_network(
        model_name='BST_CG_AP',
        n_joints=n_joints, pose_style=pose_style, in_channels=in_channels,
        n_class=tax.n_classes, seq_len=seq_len, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)

    j_plus_b = n_joints + n_bones
    human_pose = torch.randn(batch_size, seq_len, 2, j_plus_b, in_channels)
    human_pose_flat = human_pose.view(*human_pose.shape[:-2], -1)
    pos = torch.randn(batch_size, seq_len, 2, 2)
    shuttle = torch.randn(batch_size, seq_len, 2)
    video_len = torch.full((batch_size,), seq_len, dtype=torch.long)
    labels = torch.randint(0, tax.n_classes, (batch_size,))

    logits = net(human_pose_flat, shuttle, pos, video_len)
    assert logits.shape == (batch_size, tax.n_classes)
    assert torch.isfinite(logits).all()

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    grad_count = sum(
        1 for p in net.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0


# ---------------------------------------------------------------------------
# Section 6: class_weights renormalisation across the head
# ---------------------------------------------------------------------------

def test_class_weights_renorm_pair_balanced():
    """class_weights dict with named keys renormalises to mean=1 across the head.

    Under the contractual taxonomy the head is sized to tax.n_classes; the
    renorm is purely numerical (no active/full distinction any more).
    """
    tax = TAXONOMY_UNE_V1_14
    n_classes = tax.n_classes

    class_weights = {'wrist_smash': 2.0, 'smash': 2.0}
    weights = torch.ones(n_classes)
    for cls_name, mult in class_weights.items():
        weights[tax.classes.index(cls_name)] = mult
    weights = weights * (n_classes / weights.sum())

    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    for cls_name in class_weights:
        assert weights[tax.classes.index(cls_name)] > 1.0


def test_class_weights_uniform_when_empty():
    """Empty class_weights yields uniform 1.0 across the head."""
    tax = TAXONOMY_UNE_V1_14
    n_classes = tax.n_classes
    weights = torch.ones(n_classes)
    weights = weights * (n_classes / weights.sum())
    assert torch.allclose(weights, torch.ones(n_classes))


# ---------------------------------------------------------------------------
# Section 6b: collated dir basename contract (writer + reader must agree)
# ---------------------------------------------------------------------------
# The collator writes ShuttleSet_data_<tax>/<basename>/ and bst_train derives
# the same <basename> to read it back. Split folds into the name so two cells
# that share a taxonomy + collation_id but differ by split (the bst_24 case:
# split_v2 vs split_bst_baseline) land in distinct dirs instead of colliding.

@pytest.mark.parametrize('kwargs,expected', [
    (
        dict(use_3d_pose=False, seq_len=100,
             split_column='split_v2', collation_id='taxon_pinned_w_preds'),
        'npy_v2_taxon_pinned_w_preds',
    ),
    (
        dict(use_3d_pose=False, seq_len=100,
             split_column='split_bst_baseline', collation_id='taxon_pinned_w_preds'),
        'npy_bst_baseline_taxon_pinned_w_preds',
    ),
    (
        dict(use_3d_pose=True, seq_len=100,
             split_column='split_v2', collation_id='wipe_drop'),
        'npy_3d_v2_wipe_drop',
    ),
    (
        dict(use_3d_pose=False, seq_len=30,
             split_column='split_v2', collation_id='wipe_drop'),
        'npy_seq30_v2_wipe_drop',
    ),
])
def test_derive_npy_collated_dir_basename_folds_split(kwargs, expected):
    """Split is always in the basename; 3d_ and seq{N}_ tags prepend as before."""
    assert derive_npy_collated_dir_basename(**kwargs) == expected


def test_bst_24_split_variants_get_distinct_basenames():
    """The collision this fold fixes: the same taxonomy + collation_id on the
    two splits must not resolve to the same dir."""
    common = dict(use_3d_pose=False, seq_len=100, collation_id='taxon_pinned_w_preds')
    v2 = derive_npy_collated_dir_basename(split_column='split_v2', **common)
    baseline = derive_npy_collated_dir_basename(split_column='split_bst_baseline', **common)
    assert v2 != baseline


# ---------------------------------------------------------------------------
# Section 6c: collation_id_from_manifest (legacy-manifest fallback for scripts)
# ---------------------------------------------------------------------------

def test_collation_id_from_manifest_new_schema_returns_collation_id():
    # New manifest: collation_id is the collation tag; ablation_id (if set) is
    # the training tag and must NOT be returned as the collation.
    m = {'config': {'collation_id': 'taxon_pinned_w_preds', 'ablation_id': 'aug_v2'}}
    assert collation_id_from_manifest(m) == 'taxon_pinned_w_preds'


def test_collation_id_from_manifest_legacy_ablation_id():
    # Pre-refactor: the old ablation_id WAS the collation tag.
    assert collation_id_from_manifest({'config': {'ablation_id': 'wipe_drop'}}) == 'wipe_drop'


def test_collation_id_from_manifest_legacy_auto_derived_uses_effective():
    # Auto-derived runs left config.ablation_id null; the resolved tag lived in
    # extra.data_provenance.effective_ablation_id.
    m = {
        'config': {'ablation_id': None},
        'extra': {'data_provenance': {
            'effective_ablation_id': 'une_merge_v1_nosides_split_v2_dropunk',
        }},
    }
    assert collation_id_from_manifest(m) == 'une_merge_v1_nosides_split_v2_dropunk'


def test_collation_id_from_manifest_absent_returns_none():
    assert collation_id_from_manifest({'config': {}}) is None
    assert collation_id_from_manifest({}) is None


# ---------------------------------------------------------------------------
# Section 7: Real labels.npy probe (auto-skipped when /scratch... not visible)
# ---------------------------------------------------------------------------
# Under the new contract, labels.npy values are in [0, taxonomy.n_classes)
# directly (no runtime remap). This probe verifies that contract against any
# real collation dir reachable from this host. Each entry pairs a candidate
# /scratch path with the taxonomy name the dir was collated under (resolved
# via TAXONOMY_ALIASES for legacy names).

CANDIDATE_REAL_DIRS: list[tuple[str, str]] = [
    # New collations populate after Step C/E lands. Split is folded into the
    # basename (see derive_npy_collated_dir_basename), so bst_24/une_v1_14 sit
    # under the v2 split and bst_25 under the bst_baseline split:
    ('/scratch/comp320a/ShuttleSet_data_bst_24/npy_v2_taxon_pinned_w_preds', 'bst_24'),
    ('/scratch/comp320a/ShuttleSet_data_bst_25/npy_bst_baseline_taxon_pinned_w_preds', 'bst_25'),
    ('/scratch/comp320a/ShuttleSet_data_une_v1_14/npy_v2_taxon_pinned_w_preds', 'une_v1_14'),
    # Legacy collations (resume via alias):
    (
        '/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/'
        'npy_une_merge_v1_nosides_split_v2_dropunk',
        'une_merge_v1_nosides',
    ),
]


@pytest.mark.parametrize('dir_path,tax_name', CANDIDATE_REAL_DIRS)
def test_real_labels_in_active_class_range(dir_path, tax_name):
    """labels.npy values sit in [0, taxonomy.n_classes); clip_stems.npy
    (if present) row-aligns with labels.npy.
    """
    root = Path(dir_path)
    if not root.exists():
        pytest.skip(f'{dir_path} not visible from this host')

    tax = resolve_taxonomy(tax_name)
    n_classes = tax.n_classes

    for split in ('train', 'val', 'test'):
        labels_path = root / split / 'labels.npy'
        if not labels_path.exists():
            pytest.skip(f'{labels_path} missing')
        labels = np.load(str(labels_path))
        assert labels.min() >= 0, f'{split}: negative label in {dir_path}'
        assert labels.max() < n_classes, (
            f'{split}: max label {labels.max()} >= n_classes {n_classes} '
            f'in {dir_path}; new contract requires active class space'
        )

        clip_stems_path = root / split / 'clip_stems.npy'
        if clip_stems_path.exists():
            clip_stems = np.load(str(clip_stems_path), allow_pickle=True)
            assert len(clip_stems) == len(labels), (
                f'{split}: clip_stems len {len(clip_stems)} != labels len '
                f'{len(labels)} in {dir_path}'
            )
