"""Namespace migration test suite for the BST to BST-X rebrand.

T1-T12 + H1 per scratch/architecture_notes/namespace_migration_test_design.md.
Each test either holds on today's main or self-gates on a sentinel that signals
its rebrand step has landed (pytest.skip until then). Where the step IS the
contract (T5: Pydantic Literal pin), the test lands in that step's commit
rather than carrying a separate gate.

Lands at Step 0b (this commit): T1, T2, T3 (gated on ``SWITCHOVER_LANDED``),
T6, T7, T8, T9, T10, T12 (all standing on today's main), T11 (each stage
self-gates). T5 lands in the Step 1 commit; T4 lands with the first Step 4
commit.

CPU-only; runs in the laptop ``badminton-cicd`` venv and CI. Asserts only
against git-tracked artefacts so a fresh CI clone is complete; the local
untracked-weight superset is H1's job, not pytest's.
"""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import typing
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

def _first_importable(*candidates: str):
    """Return the first importable module from an ordered candidate list.

    Lets every test that touches a renamed module survive Step 6 / Step 8 by
    asking for the new name first and falling back to the old one. Raises
    rather than skipping if neither imports: that state is a broken tree, not
    a not-yet state.
    """
    last_err: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_err = exc
    raise RuntimeError(
        f'None of {candidates} importable in this tree state; last error: {last_err}'
    )


def _experiments_dir() -> Path:
    candidate = (
        REPO_ROOT / 'src' / 'bst_x' / 'stroke_classification'
        / 'main_on_shuttleset' / 'experiments'
    )
    if candidate.is_dir():
        return candidate
    raise RuntimeError(f'No experiments dir under src/bst_x in {REPO_ROOT}')


def _common_module():
    return _first_importable('main_on_shuttleset.bst_x_common')


def _train_module():
    return _first_importable('main_on_shuttleset.bst_x_train')


def _infer_module():
    return _first_importable('main_on_shuttleset.bst_x_infer')


def _api_inference_module():
    return _first_importable('src.api.bst_x_inference')


def _builder():
    """The shared BST-X network builder."""
    return _common_module().build_bst_x_network


def _switchover_landed() -> bool:
    """True once Step 6b.1 has added 'BST_X' to the MODELS dict."""
    return 'BST_X' in _common_module().MODELS


EXPERIMENTS = _experiments_dir()


# ---------------------------------------------------------------------------
# T1: MODELS alias integrity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed: BST_X absent from MODELS')
def test_t1_models_alias_keys_pin():
    """Exactly the five Chang keys plus BST_X — catches a missed alias AND an
    accidental drop of a Chang key."""
    MODELS = _common_module().MODELS
    assert set(MODELS) == {'BST_0', 'BST', 'BST_CG', 'BST_AP', 'BST_CG_AP', 'BST_X'}


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t1_models_alias_is_identity():
    """BST_X is the alias-by-identity to BST_CG_AP, not a re-declared partial."""
    MODELS = _common_module().MODELS
    assert MODELS['BST_X'] is MODELS['BST_CG_AP']


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
@pytest.mark.parametrize('key', ['BST_0', 'BST', 'BST_CG', 'BST_AP', 'BST_CG_AP', 'BST_X'])
def test_t1_models_dispatch_builds(key):
    """Every MODELS key builds through the shared builder on CPU with a sane head."""
    build = _builder()
    net, _n_bones = build(
        key, n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=14, seq_len=100, device='cpu',
    )
    assert isinstance(net, torch.nn.Module)
    assert sum(p.numel() for p in net.parameters()) > 0


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t1_alias_forward_matches_cg_ap():
    """Behavioural backstop: BST_X and BST_CG_AP produce bit-identical forward
    output under a fixed seed, on the same input batch. Catches a later
    re-declaration that drifts the alias from its target."""
    build = _builder()
    seq_len = 100

    def fresh(name):
        torch.manual_seed(0)
        net, n_bones = build(
            name, n_joints=17, pose_style='JnB_bone', in_channels=2,
            n_class=14, seq_len=seq_len, device='cpu',
        )
        net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)
        net.eval()
        return net, n_bones

    net_x, n_bones_x = fresh('BST_X')
    net_ref, n_bones_ref = fresh('BST_CG_AP')

    assert n_bones_x == n_bones_ref
    assert sum(p.numel() for p in net_x.parameters()) == sum(p.numel() for p in net_ref.parameters())

    j_plus_b = 17 + n_bones_x
    torch.manual_seed(42)
    human_pose = torch.randn(2, seq_len, 2, j_plus_b, 2)
    pos = torch.randn(2, seq_len, 2, 2)
    shuttle = torch.randn(2, seq_len, 2)
    video_len = torch.tensor([seq_len, seq_len])
    pose_flat = human_pose.view(*human_pose.shape[:-2], -1)

    with torch.no_grad():
        out_x = net_x(pose_flat, shuttle, pos, video_len)
        out_ref = net_ref(pose_flat, shuttle, pos, video_len)
    assert torch.equal(out_x, out_ref)


# ---------------------------------------------------------------------------
# T2: default model-name flip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t2_train_default_model_name_is_bst_x():
    sig = inspect.signature(_train_module().Task.get_network_architecture)
    assert sig.parameters['model_name'].default == 'BST_X'


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t2_infer_signature_defaults_are_bst_x():
    infer = _infer_module()
    assert inspect.signature(infer.Task.get_network_architecture).parameters['model_name'].default == 'BST_X'
    assert inspect.signature(infer.dump_run_predictions).parameters.get('model_name') is None or \
        inspect.signature(infer.dump_run_predictions).parameters['model_name'].default == 'BST_X'


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t2_train_main_call_literal_no_bst_cg_ap():
    """The __main__ call site at bst_x_train.py:1394 isn't importable; check the
    source. Comments mentioning BST_CG_AP survive — the regex targets
    assignment/keyword forms only."""
    text = Path(_train_module().__file__).read_text()
    matches = re.findall(r"model_name\s*=\s*'BST_CG_AP'", text)
    assert matches == [], f'Stale model_name="BST_CG_AP" call sites: {matches}'


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t2_infer_argparse_default_is_bst_x():
    """argparse `default='BST_X'` appears in the --model-name add_argument window."""
    text = Path(_infer_module().__file__).read_text()
    matches = re.findall(r"default\s*=\s*'BST_CG_AP'", text)
    assert matches == [], f'Stale argparse default=\'BST_CG_AP\': {matches}'
    window = re.search(r"--model-name.*?\)\s*\n", text, flags=re.DOTALL)
    assert window is not None, 'argparse --model-name block not found'
    assert "default='BST_X'" in window.group(0)


# ---------------------------------------------------------------------------
# T3: weight save-name round trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b.1 not landed')
def test_t3_save_name_round_trip(tmp_path):
    """Builder writes ``bst_x_*.pt``; seek_network_weights reads from that exact
    name. Proves the writer and loader agree end-to-end through the production
    code, with zero training."""
    import types
    from pipeline.config import Taxonomy

    tax3 = Taxonomy(
        name='surface3', classes=('a', 'b', 'c'), merge_map=None,
        has_sides=False, excluded_base_stroke_types=frozenset(),
    )

    build = _builder()
    bt = _train_module()
    torch.manual_seed(0)
    net, _n_bones = build(
        'BST_X', n_joints=17, pose_style='JnB_bone', in_channels=2,
        n_class=tax3.n_classes, seq_len=100, device='cpu',
    )
    weight_dir = tmp_path / 'weights'
    weight_dir.mkdir()

    expected = f'bst_x_JnB_bone_{tax3.name}_2.pt'
    torch.save(net.state_dict(), str(weight_dir / expected))

    ns = types.SimpleNamespace(
        net=net,
        model_name='BST_X',
        pose_style='JnB_bone',
        taxonomy=tax3,
        weight_dir=weight_dir,
        device='cpu',
    )
    weight_existed, val_at_best = bt.Task.seek_network_weights(ns, model_info='', serial_no=2)
    assert weight_existed is True
    assert val_at_best is None
    assert ns.weight_path.name == expected


# ---------------------------------------------------------------------------
# T5: architecture wire format (lands in the Step 1 commit; the step IS the contract)
# ---------------------------------------------------------------------------

def _architecture_literal_values(model_cls) -> set[str]:
    """Pull the Literal values from the architecture Optional[Literal[...]]
    annotation on a Pydantic model. Optional[X] expands to Union[X, None]; the
    Literal sits inside the Union."""
    hints = typing.get_type_hints(model_cls)
    annot = hints['architecture']
    for inner in typing.get_args(annot):
        if typing.get_origin(inner) is typing.Literal:
            return set(typing.get_args(inner))
    raise RuntimeError(f'No Literal in architecture annotation: {annot!r}')


def test_t5_markup_and_library_predict_request_share_architecture_values():
    from src.api.main import Markup, LibraryPredictRequest
    markup_vals = _architecture_literal_values(Markup)
    lp_vals = _architecture_literal_values(LibraryPredictRequest)
    assert markup_vals == lp_vals, (markup_vals, lp_vals)


def test_t5_architecture_values_are_bric_and_bst_x_only():
    from src.api.main import Markup
    values = _architecture_literal_values(Markup)
    assert 'bric' in values
    assert 'bst-x' in values
    assert 'bst' not in values, 'no back-compat window; "bst" dropped in Step 1'


def test_t5_markup_validates_bst_x_and_rejects_bst():
    from pydantic import ValidationError
    from src.api.main import Markup
    Markup(architecture='bst-x')  # no raise
    Markup(architecture='bric')   # no raise
    with pytest.raises(ValidationError):
        Markup(architecture='bst')
    with pytest.raises(ValidationError):
        Markup(architecture='bogus')


def test_t5_library_predict_endpoint_accepts_bst_x_and_422s_legacy():
    """Wire-level smoke: the TestClient pattern from tests/test_api.py.
    A non-422 response means request validation passed (404/503 fine — only
    request schema is under test here)."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    ok_body = {'clip_stem': 'nonexistent', 'architecture': 'bst-x'}
    resp_ok = client.post('/api/library_predict', json=ok_body)
    assert resp_ok.status_code != 422, resp_ok.text
    bad_body = {'clip_stem': 'nonexistent', 'architecture': 'nonsense'}
    resp_bad = client.post('/api/library_predict', json=bad_body)
    assert resp_bad.status_code == 422


# ---------------------------------------------------------------------------
# T4: env-var legacy fallback (lands with the first Step 4 commit; the matrix
# parametrises over each module's ENV_VAR_RENAMES so each new var added to the
# mapping picks up a four-case test row for free)
# ---------------------------------------------------------------------------

def _modules_with_renames():
    out = []
    for modname in ('pipeline.data_access', 'src.api.config'):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if getattr(mod, 'ENV_VAR_RENAMES', None) is not None:
            out.append(mod)
    return out


def _rename_pairs():
    pairs = []
    for mod in _modules_with_renames():
        for new, legacy in mod.ENV_VAR_RENAMES.items():
            pairs.append((mod, new, legacy))
    return pairs


def _pair_id(pair):
    mod, new, _ = pair
    short = mod.__name__.rsplit('.', 1)[-1]
    return f'{short}:{new}'


_T4_PAIRS = _rename_pairs()
_T4_IDS = [_pair_id(p) for p in _T4_PAIRS]


@pytest.mark.skipif(not _T4_PAIRS, reason='Step 4 not started: ENV_VAR_RENAMES empty')
@pytest.mark.parametrize('mod,new,legacy', _T4_PAIRS, ids=_T4_IDS)
def test_t4_legacy_only_resolves_with_deprecation(mod, new, legacy, monkeypatch):
    monkeypatch.delenv(new, raising=False)
    monkeypatch.setenv(legacy, 'value-X')
    with pytest.warns(DeprecationWarning):
        assert mod._resolve_env(new) == 'value-X'


@pytest.mark.skipif(not _T4_PAIRS, reason='Step 4 not started')
@pytest.mark.parametrize('mod,new,legacy', _T4_PAIRS, ids=_T4_IDS)
def test_t4_new_only_resolves_no_deprecation(mod, new, legacy, recwarn, monkeypatch):
    monkeypatch.setenv(new, 'value-Y')
    monkeypatch.delenv(legacy, raising=False)
    assert mod._resolve_env(new) == 'value-Y'
    deprecations = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, f'Unexpected DeprecationWarning: {deprecations}'


@pytest.mark.skipif(not _T4_PAIRS, reason='Step 4 not started')
@pytest.mark.parametrize('mod,new,legacy', _T4_PAIRS, ids=_T4_IDS)
def test_t4_new_wins_when_both_set(mod, new, legacy, monkeypatch):
    monkeypatch.setenv(new, 'new-val')
    monkeypatch.setenv(legacy, 'legacy-val')
    assert mod._resolve_env(new) == 'new-val'


@pytest.mark.skipif(not _T4_PAIRS, reason='Step 4 not started')
@pytest.mark.parametrize('mod,new,legacy', _T4_PAIRS, ids=_T4_IDS)
def test_t4_neither_set_returns_default(mod, new, legacy, monkeypatch):
    monkeypatch.delenv(new, raising=False)
    monkeypatch.delenv(legacy, raising=False)
    assert mod._resolve_env(new) is None


def test_t4_api_config_module_no_longer_carries_rename_machinery():
    """Step 9b deleted ENV_VAR_RENAMES from both api/config.py and
    pipeline/data_access.py. The legacy-only resolution path is gone with it:
    a process that still exports BST_LOCAL_CLIPS_DIR no longer back-doors the
    new var. The mapping's absence is the contract being pinned here; T11
    stage 5 follows up with a tree-wide scan for any legacy BST_* still in
    tracked text."""
    cfg = importlib.import_module('src.api.config')
    assert not hasattr(cfg, 'ENV_VAR_RENAMES')
    assert not hasattr(cfg, '_resolve_env')
    from pipeline import data_access
    assert not hasattr(data_access, 'ENV_VAR_RENAMES')
    assert not hasattr(data_access, '_resolve_env')


# ---------------------------------------------------------------------------
# T6: registry lockstep
# ---------------------------------------------------------------------------

REGISTRY_YAML = REPO_ROOT / 'docs' / 'models_registry.yaml'
MANIFEST_TSV = REPO_ROOT / 'scripts' / 'model_manifest.tsv'


def _load_registry() -> dict:
    return yaml.safe_load(REGISTRY_YAML.read_text())


def _registry_entries() -> list[dict]:
    return _load_registry()['models']


def _tsv_rows() -> list[list[str]]:
    rows = []
    for line in MANIFEST_TSV.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        rows.append(line.split('\t'))
    return rows


def test_t6_registry_minimum_entries_and_architectures():
    entries = _registry_entries()
    assert len(entries) >= 7
    assert {e['architecture'] for e in entries} <= {'bst-x', 'bric'}


@pytest.mark.parametrize('entry', _registry_entries(), ids=lambda e: e['id'])
def test_t6_registry_manifest_resolves(entry):
    assert (REPO_ROOT / entry['manifest_path']).exists(), \
        f'{entry["id"]}: manifest_path missing'


@pytest.mark.parametrize(
    'entry', [e for e in _registry_entries() if e['architecture'] == 'bst-x'],
    ids=lambda e: e['id'],
)
def test_t6_bst_x_weights_resolve(entry):
    """BRIC's weight is intentionally not in-tree (BRIC serves precomputed
    predictions; see scripts/model_manifest.tsv's tail comment). BST-X weights
    must resolve."""
    assert (REPO_ROOT / entry['weights_path']).exists(), \
        f'{entry["id"]}: weights_path missing'


@pytest.mark.parametrize(
    'entry', [e for e in _registry_entries() if e['architecture'] == 'bst-x'],
    ids=lambda e: e['id'],
)
def test_t6_bst_x_manifest_serial_weight_name_agrees(entry):
    """The serial referenced in the registry has a weights_path basename matching
    what the registry advertises. This is the manifest/registry filename
    agreement the rename script must preserve."""
    manifest = yaml.safe_load((REPO_ROOT / entry['manifest_path']).read_text())
    serial_no = entry['serial_no']
    serial = next((s for s in manifest['serials'] if s.get('serial_no') == serial_no), None)
    assert serial is not None, f'serial {serial_no} not in manifest'
    assert Path(serial['weights_path']).name == Path(entry['weights_path']).name


def test_t6_exactly_one_bst_x_is_default():
    bst_x = [e for e in _registry_entries() if e['architecture'] == 'bst-x']
    defaults = [e for e in bst_x if e.get('is_default') is True]
    assert len(defaults) == 1


def test_t6_tsv_dest_paths_resolve_and_match_registry():
    """Every non-comment row's dest_path exists AND equals some registry entry's
    weights_path; sha256 looks like 64 hex chars."""
    registry_weights = {e['weights_path'] for e in _registry_entries()}
    for row in _tsv_rows():
        assert len(row) == 3, f'malformed tsv row: {row!r}'
        dest_path, _asset_name, sha = row
        assert (REPO_ROOT / dest_path).is_file(), f'tsv dest missing: {dest_path}'
        assert dest_path in registry_weights, f'tsv dest not in registry: {dest_path}'
        assert re.fullmatch(r'[0-9a-f]{64}', sha), f'sha256 malformed: {sha!r}'


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b not landed')
def test_t6_post_switchover_weights_prefixed_bst_x():
    """Post-6b.2: every bst-x entry advertises a bst_x_*.pt weight; the string
    'bst_CG_AP' no longer appears anywhere in the registry file; the tsv's
    dest_path basenames are bst_x_*. The asset_name column is exempt
    (release assets keep their pre-rebrand names by design)."""
    raw = REGISTRY_YAML.read_text()
    assert 'bst_CG_AP' not in raw
    for entry in _registry_entries():
        if entry['architecture'] == 'bst-x':
            assert Path(entry['weights_path']).name.startswith('bst_x_'), entry['id']
    for row in _tsv_rows():
        dest_path = row[0]
        assert Path(dest_path).name.startswith('bst_x_'), dest_path


# ---------------------------------------------------------------------------
# T7: live-inference weight literal
# ---------------------------------------------------------------------------

def test_t7_run_dir_and_weights_exist():
    mod = _api_inference_module()
    assert mod.RUN_DIR.is_dir(), f'RUN_DIR missing: {mod.RUN_DIR}'
    assert mod.WEIGHTS_PATH.is_file(), f'WEIGHTS_PATH missing: {mod.WEIGHTS_PATH}'


@pytest.mark.skipif(not _switchover_landed(), reason='Step 6b not landed')
def test_t7_weights_prefixed_bst_x():
    assert _api_inference_module().WEIGHTS_PATH.name.startswith('bst_x_')


# ---------------------------------------------------------------------------
# T8: sidecar schema invariants
# ---------------------------------------------------------------------------

FE_FILES = {'clip_index.json.gz', 'test.json.gz', 'val.json.gz',
            'perclass_stats_test.json.gz', 'perclass_stats_val.json.gz'}

CLIPS_SPLIT_TOP_KEYS = {'run_id', 'serial_no', 'split', 'class_list', 'clips'}
CLIPS_ENTRY_KEYS = {'clip_stem', 'softmax', 'top_k_idx', 'top_k_prob', 'y_pred', 'y_true'}
PERCLASS_TOP_KEYS = {'class_list', 'n_clips', 'per_class', 'split'}
PERCLASS_ENTRY_KEYS = {'f1', 'precision', 'recall', 'support_pred', 'support_true',
                       'top5_when_pred', 'top5_when_true'}
CLIP_INDEX_TOP_KEYS = {'clips'}
CLIP_INDEX_ENTRY_KEYS = {'ball_round', 'match', 'player_side', 'rally',
                         'raw_type_en', 'set_id', 'split', 'video_path'}
NPZ_FIELDS = {'logits', 'y_true', 'y_pred_top1', 'topk_idx', 'clip_stems',
              'class_list', 'run_id', 'serial_no', 'taxonomy_name'}


def _load_gz_json(p: Path) -> dict:
    with gzip.open(p, 'rt') as fh:
        return json.load(fh)


def _check_no_model_name_key(obj, path='root'):
    """Recursive walk asserting 'model_name' never appears as a dict key."""
    if isinstance(obj, dict):
        assert 'model_name' not in obj, f'{path}: unexpected model_name key'
        for k, v in obj.items():
            _check_no_model_name_key(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_model_name_key(v, f'{path}[{i}]')


def _registry_anchored_fe_dirs() -> list[Path]:
    """The bst-x registry's run dirs (six in total today). Slimmed at Step 9b
    from the full-corpus walk: the migration insurance has done its job, this
    keeps the FE schema pin on the shape the API actually serves."""
    out = []
    for entry in _registry_entries():
        if entry['architecture'] != 'bst-x':
            continue
        fe_dir = (REPO_ROOT / entry['manifest_path']).parent / 'fe_jsons'
        if fe_dir.is_dir():
            out.append(fe_dir)
    return sorted(out)


@pytest.mark.parametrize(
    'fe_dir', _registry_anchored_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_fe_jsons_exact_file_set(fe_dir):
    """Every fe_jsons/ dir holds exactly the five gzipped json sidecars."""
    assert {p.name for p in fe_dir.iterdir()} == FE_FILES


@pytest.mark.parametrize(
    'fe_dir', _registry_anchored_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_clips_split_schema(fe_dir):
    run_id = fe_dir.parent.name
    for split in ('test', 'val'):
        data = _load_gz_json(fe_dir / f'{split}.json.gz')
        assert set(data) == CLIPS_SPLIT_TOP_KEYS
        assert data['run_id'] == run_id
        assert data['split'] == split
        assert isinstance(data['clips'], list) and data['clips']
        for clip in data['clips']:
            assert set(clip) == CLIPS_ENTRY_KEYS
        _check_no_model_name_key(data)


@pytest.mark.parametrize(
    'fe_dir', _registry_anchored_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_perclass_stats_schema(fe_dir):
    for split in ('test', 'val'):
        data = _load_gz_json(fe_dir / f'perclass_stats_{split}.json.gz')
        assert set(data) == PERCLASS_TOP_KEYS
        assert data['split'] == split
        assert isinstance(data['per_class'], dict) and data['per_class']
        for cls, stats in data['per_class'].items():
            assert set(stats) == PERCLASS_ENTRY_KEYS
        _check_no_model_name_key(data)


@pytest.mark.parametrize(
    'fe_dir', _registry_anchored_fe_dirs(), ids=lambda p: p.parent.name,
)
def test_t8_clip_index_schema(fe_dir):
    data = _load_gz_json(fe_dir / 'clip_index.json.gz')
    assert set(data) == CLIP_INDEX_TOP_KEYS
    clips = data['clips']
    if isinstance(clips, dict):
        for stem, meta in clips.items():
            assert set(meta) == CLIP_INDEX_ENTRY_KEYS
    else:
        for clip in clips:
            assert set(clip) == CLIP_INDEX_ENTRY_KEYS
    _check_no_model_name_key(data)


def test_t8_registry_anchored_dirs_have_fe_jsons():
    """Vacuous-pass guard: every bst-x registry entry's run dir carries the
    complete five-file fe_jsons/ set."""
    for entry in _registry_entries():
        if entry['architecture'] != 'bst-x':
            continue
        run_dir = (REPO_ROOT / entry['manifest_path']).parent
        fe_dir = run_dir / 'fe_jsons'
        assert fe_dir.is_dir(), f'{entry["id"]}: fe_jsons/ missing'
        assert {p.name for p in fe_dir.iterdir()} == FE_FILES


# ---------------------------------------------------------------------------
# T9: Chang KEEP set, code identifiers
# ---------------------------------------------------------------------------

VARIANT_FLAGS = {
    'BST_0':     {'use_ppf': False, 'use_cg': False, 'use_ap': False},
    'BST_PPF':   {'use_ppf': True,  'use_cg': False, 'use_ap': False},
    'BST_CG':    {'use_ppf': True,  'use_cg': True,  'use_ap': False},
    'BST_AP':    {'use_ppf': True,  'use_cg': False, 'use_ap': True},
    'BST_CG_AP': {'use_ppf': True,  'use_cg': True,  'use_ap': True},
}


def test_t9_bst_variant_partials_keep_their_flag_table():
    bst_mod = importlib.import_module('model.bst')
    BST = bst_mod.BST
    for name, expected in VARIANT_FLAGS.items():
        variant = getattr(bst_mod, name)
        assert isinstance(variant, partial)
        assert variant.func is BST
        assert variant.keywords == expected


def test_t9_models_chang_keys_identity():
    common = _common_module()
    bst_mod = importlib.import_module('model.bst')
    MODELS = common.MODELS
    # 'BST' is the most likely to be mangled by a mechanical rename — it maps to BST_PPF.
    assert MODELS['BST'] is bst_mod.BST_PPF
    assert MODELS['BST_0'] is bst_mod.BST_0
    assert MODELS['BST_CG'] is bst_mod.BST_CG
    assert MODELS['BST_AP'] is bst_mod.BST_AP
    assert MODELS['BST_CG_AP'] is bst_mod.BST_CG_AP


def test_t9_taxonomy_constants_pin():
    from pipeline.config import (
        TAXONOMY_BST_25, TAXONOMY_BST_24, TAXONOMY_BST_12, resolve_taxonomy,
    )
    assert TAXONOMY_BST_25.name == 'bst_25' and TAXONOMY_BST_25.n_classes == 25
    assert TAXONOMY_BST_24.name == 'bst_24' and TAXONOMY_BST_24.n_classes == 24
    assert TAXONOMY_BST_12.name == 'bst_12' and TAXONOMY_BST_12.n_classes == 12
    assert resolve_taxonomy('bst_25') is TAXONOMY_BST_25


def test_t9_splits_bst_baseline_shape():
    from shared.dataset import SPLITS_BST_BASELINE
    assert isinstance(SPLITS_BST_BASELINE, dict)
    assert set(SPLITS_BST_BASELINE) == {'train', 'val', 'test'}
    for split, vids in SPLITS_BST_BASELINE.items():
        assert isinstance(vids, list) and vids


def test_t9_clips_master_carries_split_columns():
    """Header-only read for speed. split_bst_baseline is the schema-level
    column; split_v2 sits next to it. Renaming either invalidates the master."""
    df = pd.read_csv(REPO_ROOT / 'notebooks' / 'clips_master.csv', nrows=0)
    cols = set(df.columns)
    assert 'split_bst_baseline' in cols
    assert 'split_v2' in cols


def test_t9_chang_attribution_intact():
    """Prose with no import surface; greppy by necessity. The attribution
    sits in the BST file header."""
    bst_path = importlib.import_module('model.bst').__file__
    first_line = Path(bst_path).read_text().splitlines()[0]
    assert 'Original BST by Jing-Yuan Chang' in first_line


# ---------------------------------------------------------------------------
# T10: Chang baseline run untouched
# ---------------------------------------------------------------------------

BASELINE_DIR = EXPERIMENTS / 'bst_cg_ap_base_17_04_2026'
BASELINE_PREFIX_MIXED = 'bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25'
BASELINE_PREFIX_LOWER = 'bst_cg_ap_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25'
BASELINE_TRIPLE_MIXED = {
    f'{BASELINE_PREFIX_MIXED}.pt',
    f'{BASELINE_PREFIX_MIXED}_2.pt',
    f'{BASELINE_PREFIX_MIXED}_3.pt',
}
BASELINE_TRIPLE_LOWER = {
    f'{BASELINE_PREFIX_LOWER}.pt',
    f'{BASELINE_PREFIX_LOWER}_2.pt',
    f'{BASELINE_PREFIX_LOWER}_3.pt',
}


def _baseline_lowercased_gate() -> bool:
    """Different signal from the asserted property, per the self-gating rule:
    the run-dir rename lands in the same 6b.2 commit as the baseline
    lowercase. If ANY run_*/weights/bst_x_*.pt exists, 6b.2 has landed."""
    return any(EXPERIMENTS.glob('run_*/weights/bst_x_*.pt'))


def test_t10_baseline_dir_has_three_weights():
    weights = sorted(p.name for p in (BASELINE_DIR / 'weights').glob('*.pt'))
    assert len(weights) == 3, weights


def test_t10_baseline_dir_never_carries_bst_x_prefix():
    """Decision 4: the baseline never becomes ``bst_x_*``; it lowercases in place."""
    bst_x = list((BASELINE_DIR / 'weights').glob('bst_x_*.pt'))
    assert bst_x == []


def test_t10_baseline_weight_triple_matches_state():
    weights = {p.name for p in (BASELINE_DIR / 'weights').glob('*.pt')}
    if _baseline_lowercased_gate():
        assert weights == BASELINE_TRIPLE_LOWER
    else:
        assert weights == BASELINE_TRIPLE_MIXED


def test_t10_baseline_manifest_matches_weight_triple():
    """Catches a files-renamed-manifest-missed half-application. The baseline
    manifest uses ``experiments/``-relative paths; compare basenames so the test
    is convention-agnostic."""
    manifest = yaml.safe_load((BASELINE_DIR / 'manifest.yaml').read_text())
    serial_basenames = {Path(s['weights_path']).name for s in manifest['serials']}
    expected = BASELINE_TRIPLE_LOWER if _baseline_lowercased_gate() else BASELINE_TRIPLE_MIXED
    assert serial_basenames == expected


# ---------------------------------------------------------------------------
# T11: staged orphan-string scan
# ---------------------------------------------------------------------------

TEXT_EXTS = {
    '.py', '.md', '.yaml', '.yml', '.toml', '.sh', '.ipynb', '.txt', '.tsv',
    '.jsx', '.js', '.gitignore',
}
EXPLICIT_TEXT_NAMES = {'.env.example', 'docker-compose.yml', 'docker-compose.dev.yml',
                       'docker-compose.prod.yml'}

GLOBAL_EXCLUDE_PREFIXES = (
    'scratch/project_history/',
    'local_scratch/',
)


def _tracked_text_files() -> list[Path]:
    """git ls-files filtered to text extensions. Tracked-only keeps CI
    deterministic and skips venvs/caches by construction."""
    out = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'ls-files'], text=True,
    ).splitlines()
    paths: list[Path] = []
    for rel in out:
        if any(rel.startswith(p) for p in GLOBAL_EXCLUDE_PREFIXES):
            continue
        p = Path(rel)
        if p.suffix in TEXT_EXTS or p.name in EXPLICIT_TEXT_NAMES:
            paths.append(p)
    return paths


def _scan_pattern(pattern: re.Pattern[str], paths, allow_path):
    """Return list of (relpath, lineno, line) hits not covered by allow_path."""
    hits = []
    for rel in paths:
        if allow_path(rel):
            continue
        try:
            for i, line in enumerate((REPO_ROOT / rel).read_text(errors='ignore').splitlines(), 1):
                if pattern.search(line):
                    hits.append((str(rel), i, line.rstrip()))
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    return hits


def _step8_landed() -> bool:
    """Step 8 lands when the old src/bst_refactor directory is gone and the
    new src/bst_x is in place."""
    return not (REPO_ROOT / 'src' / 'bst_refactor').exists() and \
        (REPO_ROOT / 'src' / 'bst_x').exists()


def _step6_common_landed() -> bool:
    try:
        importlib.import_module('main_on_shuttleset.bst_x_common')
        return True
    except ImportError:
        return False


def _step7_bst_inputs_landed() -> bool:
    compose = REPO_ROOT / 'docker-compose.dev.yml'
    if not compose.exists():
        return False
    return 'bst_x_inputs' in compose.read_text()


def _step5_runtime_landed() -> bool:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text()
    return re.search(r'^\s*bst-x-runtime\s*=', pyproject, flags=re.MULTILINE) is not None


def _step9b_landed() -> bool:
    """Step 9b removed ENV_VAR_RENAMES from pipeline.data_access. On today's
    main the mapping is also absent (Step 4 hasn't landed yet), so detect Step
    4 separately via a new BST_X_* var in .env.example and require both."""
    try:
        from pipeline import data_access
        if hasattr(data_access, 'ENV_VAR_RENAMES'):
            return False
    except ImportError:
        return False
    env_example = REPO_ROOT / '.env.example'
    if not env_example.exists():
        return False
    return 'BST_X_CLIPS_DIR' in env_example.read_text()


def test_t11_stage1_bst_refactor():
    if not _step8_landed():
        pytest.skip('Step 8 not landed: src/bst_refactor still exists')
    # The test file and design doc both quote the pattern verbatim; the two
    # historical-archive narrative docs reference scratch/project_history/
    # bst_refactor_deprecated/ as the actual archived directory name.
    allowed = {
        'scratch/architecture_notes/namespace_migration_test_design.md',
        'scratch/architecture_notes/historical_bst.md',
        'scratch/architecture_notes/pre_phase_2_tidy_plan.md',
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        re.compile(r'bst_refactor'), _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage2_module_paths():
    if not _step6_common_landed():
        pytest.skip('Step 6 not landed: bst_x_common not importable')
    pattern = re.compile(r'main_on_shuttleset\.bst_(train|infer|common)\b|\bbuild_bst_network\b')
    # The test file itself contains the pattern as a regex literal; exempt it
    # to avoid a self-flag. Pickup-doc + assessment historical narratives
    # outside the in-scope code aren't part of this gate either.
    allowed = {
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        pattern, _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage3_bst_inputs_dir():
    if not _step7_bst_inputs_landed():
        pytest.skip('Step 7 not landed: docker-compose.dev.yml does not mention bst_x_inputs')
    # The design doc and this test file quote the rename pattern verbatim;
    # allowlist both to avoid a self-flag.
    allowed = {
        'scratch/architecture_notes/namespace_migration_test_design.md',
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        re.compile(r'\bbst_inputs\b'), _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage4_extras_group():
    if not _step5_runtime_landed():
        pytest.skip('Step 5 not landed: pyproject.toml has no bst-x-runtime group')
    hits = _scan_pattern(
        re.compile(r'\bbst-runtime\b'), _tracked_text_files(),
        allow_path=lambda rel: False,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage5_legacy_env_vars():
    if not _step9b_landed():
        pytest.skip('Step 9b not landed: ENV_VAR_RENAMES still present')
    pattern = re.compile(
        r'\bBST_(CLIPS_DIR|CLIPS_CSV|SHUTTLE_NPY_DIR|MMPOSE_NPY_DIR|INPUTS_DIR'
        r'|DATA_DIR|LOCAL_CLIPS_DIR|REPO_ROOT|REGISTRY_PATH|SHUTTLE_CSV_DIR)\b'
    )
    # Allowlist:
    # - The design doc and this test file quote the legacy names verbatim.
    # - Historical refactor logs / dated tidy-plan narratives reference the
    #   pre-rebrand names as they were at the time.
    allowed = {
        'scratch/architecture_notes/namespace_migration_test_design.md',
        'scratch/architecture_notes/pre_phase_2_tidy_plan.md',
        'scratch/architecture_notes/completed_general_refactors/data_access_integration_plan.md',
        'scratch/architecture_notes/completed_general_refactors/dir_flatten_refactor.md',
        'scratch/collation_taxon_pin_w_preds_refactor_log.md',
        'scratch/collation_taxon_pin_w_preds_refactor.md',
        'tests/test_namespace_migration.py',
    }
    hits = _scan_pattern(
        pattern, _tracked_text_files(),
        allow_path=lambda rel: str(rel) in allowed,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def _stage6_in_scope(rel: str) -> bool:
    """Scoped stage-6 corpus: doc tree, api code, manifest tsv, run manifests,
    and the Chang baseline dir. Scratch history and the ledger legitimately
    mention old filenames; not in scope here."""
    if rel.startswith('docs/'):
        return True
    if rel.startswith('src/api/'):
        return True
    if rel == 'scripts/model_manifest.tsv':
        return True
    if rel.startswith(('src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_',
                       'src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_')):
        return True
    if rel.startswith(('src/bst_x/stroke_classification/main_on_shuttleset/experiments/bst_cg_ap_base_',
                       'src/bst_x/stroke_classification/main_on_shuttleset/experiments/bst_cg_ap_base_')):
        return True
    return False


def test_t11_stage6_bst_cg_ap_filename_prose():
    if not _switchover_landed():
        pytest.skip('Step 6b not landed: SWITCHOVER_LANDED is False')
    pattern = re.compile(r'bst_CG_AP')
    hits = []
    for rel in (str(p) for p in _tracked_text_files()):
        if not _stage6_in_scope(rel):
            continue
        path = REPO_ROOT / rel
        try:
            lines = path.read_text(errors='ignore').splitlines()
        except FileNotFoundError:
            continue
        # tsv's asset_name column (column index 1) is exempt — frozen by design.
        if rel == 'scripts/model_manifest.tsv':
            for i, line in enumerate(lines, 1):
                if line.startswith('#') or not line.strip():
                    continue
                cols = line.split('\t')
                checkable = '\t'.join([cols[0]] + cols[2:]) if len(cols) >= 2 else line
                if pattern.search(checkable):
                    hits.append((rel, i, line.rstrip()))
            continue
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                hits.append((rel, i, line.rstrip()))
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


def test_t11_stage7_legacy_venv_name():
    if os.environ.get('RENAME_SCAN_VENV') != '1':
        pytest.skip('Set RENAME_SCAN_VENV=1 to run the venv-name scan (out-of-repo ops step)')
    hits = _scan_pattern(
        re.compile(r'venv-bst(?!-x)'), _tracked_text_files(),
        allow_path=lambda rel: False,
    )
    assert hits == [], '\n'.join(f'{r}:{n}: {line}' for r, n, line in hits)


# ---------------------------------------------------------------------------
# T12: subprocess and dynamic module strings resolve
# ---------------------------------------------------------------------------

SUBPROCESS_M_PATTERN = re.compile(r"'-m',\s*'([\w\.]+)'")
IMPORT_MODULE_PATTERN = re.compile(r"import_module\('([\w\.]+)'\)")


def _runner_modules() -> list[tuple[str, str]]:
    """Return [(label, source_text)] for the runner modules that ship dynamic
    module-name strings. Uses the import helper so the test survives Step 6."""
    out = []
    for label, modgetter in (
        ('collation_runner', lambda: _first_importable(
            'main_on_shuttleset.collation_runner')),
        ('hparam_sweep', lambda: _first_importable(
            'main_on_shuttleset.hparam_sweep')),
    ):
        mod = modgetter()
        out.append((label, Path(mod.__file__).read_text()))
    # verify_bst_train_target may rename to verify_bst_x_train_target at Step 6.
    for verify_candidate in ('verify_bst_x_train_target', 'verify_bst_train_target'):
        path = REPO_ROOT / 'src' / 'bst_x' / 'validation_scripts' / f'{verify_candidate}.py'
        if not path.exists():
            path = REPO_ROOT / 'src' / 'bst_x' / 'validation_scripts' / f'{verify_candidate}.py'
        if path.exists():
            out.append(('verify_target', path.read_text()))
            break
    return out


def test_t12_dynamic_module_strings_resolve():
    """For each subprocess `-m` and import_module literal, the module spec is
    findable. find_spec imports the parent package only, so it stays cheap."""
    found_any = False
    for label, src in _runner_modules():
        spec_strings = SUBPROCESS_M_PATTERN.findall(src) + IMPORT_MODULE_PATTERN.findall(src)
        if not spec_strings:
            continue
        found_any = True
        for name in spec_strings:
            assert importlib.util.find_spec(name) is not None, \
                f'{label}: cannot import {name!r}'
    assert found_any, 'No -m or import_module captures found across runners (regex rot?)'
