"""FE inference-stub field back-compat tests (Step J3).

`_pick_predictions_pool` reads the per-clip class list from the test-split
predictions JSON. It now prefers the canonical `class_list` field (emitted by
the post-hoc converter / api_contract-aligned) and falls back to the legacy
`active_class_list` for the pre-refactor mock JSONs. This pins that preference.

CPU-only, no PyTorch, no /scratch.

Run from repo root::

    pytest tests/test_api_inference.py -v
"""

from __future__ import annotations

from src.api import inference


def _patch_pool(monkeypatch, preds: dict) -> None:
    """Point _pick_predictions_pool at a single fake model whose test.json is `preds`."""
    monkeypatch.setattr(
        inference, '_load_registry',
        lambda: {'models': [{'manifest_path': 'fake/manifest.yaml'}]},
    )
    monkeypatch.setattr(inference, '_read_json_under_run', lambda *_a, **_k: preds)


def test_pick_pool_prefers_canonical_class_list(monkeypatch):
    _patch_pool(monkeypatch, {
        'class_list': ['smash', 'lob'],
        'active_class_list': ['stale', 'legacy'],
        'clips': [{'clip_stem': 'x'}],
    })
    class_list, clips = inference._pick_predictions_pool()
    assert class_list == ['smash', 'lob']
    assert clips == [{'clip_stem': 'x'}]


def test_pick_pool_falls_back_to_legacy_field(monkeypatch):
    _patch_pool(monkeypatch, {
        'active_class_list': ['net_shot', 'return_net'],
        'clips': [{'clip_stem': 'y'}],
    })
    class_list, clips = inference._pick_predictions_pool()
    assert class_list == ['net_shot', 'return_net']


def test_pick_pool_empty_when_no_models(monkeypatch):
    monkeypatch.setattr(inference, '_load_registry', lambda: {'models': []})
    assert inference._pick_predictions_pool() == ([], [])
