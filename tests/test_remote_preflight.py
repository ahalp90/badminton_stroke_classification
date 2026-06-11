"""Pre-flight gates for the taxon_pinned_w_preds batch run, to run ON BOURBAKI
(or any host that can see the collations) BEFORE launching the 45-serial batch.

These check the things that can only be confirmed against the real /scratch
collations, the ones that would otherwise crash or quietly misbehave on serial 1
of a cell:

1. Each of the 6 cells resolves to a collation dir that EXISTS on disk with the
   files bst_x_train reads (catches a reader/writer basename mismatch -> the
   FileNotFoundError on serial 1).
2. Each cell's labels pass the same coverage contract bst_x_train asserts at train
   start (train covers every class; val/test carry no class absent from train;
   labels in [0, n_classes)) -- catches a collation/label-space mismatch.
3. The 6 cells resolve to 6 DISTINCT dirs (the two bst_24 cells, split_v2 vs
   split_bst_baseline, must not collide).
4. bst_25 + split_bst_baseline keeps a non-zero unknown class (a dead 25th class
   would show as zero support).
5. Any prediction npz already on disk (e.g. after a 1-serial smoke, or after the
   batch starts) has the full 9-key schema and is internally row-aligned.

No torch, no GPU, no model load -- pure data checks, so it runs fast in any venv
on bourbaki (venv-bst-x is fine). Run with the collation root visible:

    BST_X_COLLATED_DATA_ROOT=/scratch/comp320a \
        PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \
        /home/ahalperi/.venvs/venv-bst-x/bin/python -m pytest tests/test_remote_preflight.py -v

(or rely on .env carrying BST_X_COLLATED_DATA_ROOT). Without the root set, the
/scratch-dependent tests skip, so this is a no-op on the laptop except the
basename-collision check (3), which is pure path arithmetic.

The GPU forward path and the npz-on-real-data are exercised by serial 1 of the
batch itself; check (5) then validates that serial's npz (run this module again
after the first serial lands, or after a 1-serial smoke).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.config import derive_npy_collated_dir_basename, resolve_taxonomy
from pipeline.data_access import env_path_or_none, load_repo_dotenv


COLLATION_ID = 'taxon_pinned_w_preds'
POSE_STYLE = 'JnB_bone'  # what bst_x_train reads for these cells

# The 6 cells, mirroring scratch/runners/taxon_pinned_w_preds/config.yaml.
CELLS: list[tuple[str, str]] = [
    ('shuttleset_18', 'split_v2'),
    ('bst_24', 'split_v2'),
    ('bst_12', 'split_v2'),
    ('bst_25', 'split_bst_baseline'),
    ('bst_24', 'split_bst_baseline'),
    ('une_v1_14', 'split_v2'),
]

SPLITS = ('train', 'val', 'test')
# Files bst_x_train's Dataset_npy_collated loads for a JnB_bone cell.
REQUIRED_NPY = (f'{POSE_STYLE}.npy', 'pos.npy', 'shuttle.npy',
                'videos_len.npy', 'labels.npy', 'clip_stems.npy')

NPZ_FIELDS = {
    'logits', 'y_true', 'y_pred_top1', 'topk_idx', 'clip_stems',
    'class_list', 'run_id', 'serial_no', 'taxonomy_name',
}

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = (
    REPO_ROOT / 'src/bst_x/stroke_classification/main_on_shuttleset/experiments'
)


def _cell_dir(root: Path, taxonomy: str, split_column: str) -> Path:
    """Resolve a cell's collation dir the same way bst_x_train does."""
    basename = derive_npy_collated_dir_basename(
        use_3d_pose=False, seq_len=100,
        split_column=split_column, collation_id=COLLATION_ID,
    )
    return root / f'ShuttleSet_data_{resolve_taxonomy(taxonomy).name}' / basename


def _require_root() -> Path:
    """Collation root, or skip when it's not set (i.e. running off-host)."""
    load_repo_dotenv()
    root = env_path_or_none('BST_X_COLLATED_DATA_ROOT')
    if root is None:
        pytest.skip('BST_X_COLLATED_DATA_ROOT not set; run on bourbaki with the '
                    'collation root visible.')
    return root


_CELL_IDS = [f'{tax}+{split.removeprefix("split_")}' for tax, split in CELLS]


# ---------------------------------------------------------------------------
# 1. Collation dirs exist where bst_x_train will look, with the files it reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('taxonomy,split_column', CELLS, ids=_CELL_IDS)
def test_cell_collation_dir_and_files_exist(taxonomy, split_column):
    root = _require_root()
    cell_dir = _cell_dir(root, taxonomy, split_column)
    assert cell_dir.is_dir(), (
        f'collation dir missing: {cell_dir}\n  bst_x_train would hit this as a '
        f'FileNotFoundError on serial 1. Check the collation basename matches '
        f'derive_npy_collated_dir_basename (split folded in), or re-collate.'
    )
    for split in SPLITS:
        split_dir = cell_dir / split
        assert split_dir.is_dir(), f'missing split dir: {split_dir}'
        missing = [f for f in REQUIRED_NPY if not (split_dir / f).exists()]
        assert not missing, f'{split_dir} missing files: {missing}'


# ---------------------------------------------------------------------------
# 2. Labels pass the bst_x_train coverage contract, on the real arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('taxonomy,split_column', CELLS, ids=_CELL_IDS)
def test_cell_label_coverage(taxonomy, split_column):
    root = _require_root()
    cell_dir = _cell_dir(root, taxonomy, split_column)
    if not cell_dir.is_dir():
        pytest.skip(f'{cell_dir} absent; covered by the dir-exists test.')

    tax = resolve_taxonomy(taxonomy)
    n_classes = tax.n_classes
    present = {}
    for split in SPLITS:
        labels = np.load(str(cell_dir / split / 'labels.npy'))
        assert labels.min() >= 0 and labels.max() < n_classes, (
            f'{taxonomy}/{split}: labels out of [0, {n_classes}); '
            f'min={labels.min()} max={labels.max()} -- collation/head mismatch.'
        )
        stems = np.load(str(cell_dir / split / 'clip_stems.npy'), allow_pickle=True)
        assert len(stems) == len(labels), (
            f'{taxonomy}/{split}: clip_stems ({len(stems)}) != labels ({len(labels)}).'
        )
        present[split] = {int(x) for x in np.unique(labels)}

    missing_in_train = set(range(n_classes)) - present['train']
    assert not missing_in_train, (
        f'{taxonomy}: train misses classes {sorted(missing_in_train)} '
        f'({[tax.classes[i] for i in sorted(missing_in_train)]}). '
        f'_assert_label_coverage would refuse the run.'
    )
    for split in ('val', 'test'):
        rogue = present[split] - present['train']
        assert not rogue, (
            f'{taxonomy}/{split}: classes absent from train {sorted(rogue)}.'
        )


# ---------------------------------------------------------------------------
# 3. The 6 cells land in 6 distinct dirs (pure path arithmetic; runs anywhere)
# ---------------------------------------------------------------------------

def test_cells_resolve_to_distinct_dirs():
    dummy = Path('/__preflight__')
    dirs = [str(_cell_dir(dummy, tax, split)) for tax, split in CELLS]
    assert len(set(dirs)) == len(CELLS), (
        'two cells collide on the same collation dir:\n  '
        + '\n  '.join(sorted(dirs))
    )
    # The specific pair this guards: the two bst_24 cells must differ by split.
    bst24 = sorted(d for d in dirs if 'ShuttleSet_data_bst_24/' in d)
    assert len(bst24) == 2 and bst24[0] != bst24[1]


# ---------------------------------------------------------------------------
# 4. bst_25 keepunk keeps a live unknown class
# ---------------------------------------------------------------------------

def test_bst_25_unknown_has_support():
    root = _require_root()
    cell_dir = _cell_dir(root, 'bst_25', 'split_bst_baseline')
    if not cell_dir.is_dir():
        pytest.skip(f'{cell_dir} absent; covered by the dir-exists test.')

    tax = resolve_taxonomy('bst_25')
    assert tax.classes[-1] == 'unknown'
    unknown_idx = tax.n_classes - 1  # 24

    counts = {}
    for split in SPLITS:
        labels = np.load(str(cell_dir / split / 'labels.npy'))
        counts[split] = int((labels == unknown_idx).sum())
    # Print so the operator can eyeball the 875/241/162 split.
    print(f'bst_25 unknown (idx {unknown_idx}) support: {counts}')
    assert counts['train'] > 0, (
        f'bst_25 train has zero unknown clips -> dead 25th class. counts={counts}'
    )


# ---------------------------------------------------------------------------
# 5. Any prediction npz on disk has the full schema + is row-aligned
# ---------------------------------------------------------------------------

def _find_prediction_npzs() -> list[Path]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    found = list(EXPERIMENTS_DIR.glob('run_*/predictions/*_serial_*.npz'))
    found += list(EXPERIMENTS_DIR.glob('run_*/inference_runs/*/*_serial_*.npz'))
    return found


def test_prediction_npzs_have_full_schema():
    """Validates any npz produced by a real serial (train-time dump or
    bst_x_infer --fe). Skips when none exist yet -- run after serial 1 lands or a
    1-serial smoke, then re-run this module.
    """
    npzs = _find_prediction_npzs()
    if not npzs:
        pytest.skip('no prediction npz on disk yet; run a serial first.')

    for path in npzs:
        z = np.load(str(path), allow_pickle=True)
        assert set(z.files) == NPZ_FIELDS, f'{path}: keys {sorted(z.files)}'
        n = len(z['y_true'])
        assert z['logits'].shape[0] == n
        assert len(z['clip_stems']) == n, (
            f'{path}: clip_stems ({len(z["clip_stems"])}) != rows ({n})'
        )
        assert z['y_pred_top1'].tolist() == z['topk_idx'][:, 0].tolist(), (
            f'{path}: y_pred_top1 != topk_idx[:,0]'
        )
        n_classes = len(z['class_list'])
        assert int(z['y_true'].min()) >= 0 and int(z['y_true'].max()) < n_classes
