"""FE registry resolver tests for the taxon_pinned_w_preds refactor (Step J).

`_resolve_class_list` is the small helper that sources a model's class list from
whichever field the manifest carries: `config.classes` (canonical, post-refactor
BST + BRIC) with a fallback to `extra.arch.active_class_list` (legacy BST). The
same patch incidentally lights up BRIC entries, whose class list was never in
BST's bandaid block.

CPU-only, no PyTorch, no /scratch.

Run from repo root::

    pytest tests/test_api_registry.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import registry
from src.api.registry import _resolve_class_list
from src.api.main import app


# ---------------------------------------------------------------------------
# _resolve_class_list: schema-variant unit tests
# ---------------------------------------------------------------------------

def test_resolve_class_list_prefers_config_classes():
    """Canonical post-refactor manifests carry config.classes; it wins."""
    manifest = {
        'config': {'classes': ['net_shot', 'smash', 'lob']},
        'extra': {'arch': {'active_class_list': ['stale', 'legacy', 'block']}},
    }
    assert _resolve_class_list(manifest) == ['net_shot', 'smash', 'lob']


def test_resolve_class_list_falls_back_to_legacy_arch_block():
    """Pre-refactor BST manifests have no config.classes; use the bandaid block."""
    manifest = {
        'config': {'taxonomy': 'une_merge_v1_nosides'},  # no 'classes'
        'extra': {'arch': {'active_class_list': ['net_shot', 'return_net']}},
    }
    assert _resolve_class_list(manifest) == ['net_shot', 'return_net']


def test_resolve_class_list_empty_when_neither_present():
    assert _resolve_class_list({}) == []
    assert _resolve_class_list({'config': {}, 'extra': {}}) == []
    # An empty list in either field is treated as absent (falls through).
    assert _resolve_class_list({'config': {'classes': []}}) == []


# ---------------------------------------------------------------------------
# /api/registry integration: J1 fallback resolves against the real mock manifest
# ---------------------------------------------------------------------------

def test_registry_endpoint_resolves_class_list_via_fallback():
    """GET /api/registry resolves a non-empty class_list for the shipped mock
    entry, whose (legacy) manifest carries only extra.arch.active_class_list.
    Proves the J1 fallback path against real data and that the new
    collation_id field is surfaced.
    """
    client = TestClient(app)
    resp = client.get("/api/registry")
    assert resp.status_code == 200
    models = resp.json()["models"]
    if not models:
        pytest.skip("no models registered in docs/models_registry.yaml")

    entry = models[0]
    assert entry["num_classes"] == len(entry["class_list"])
    assert entry["num_classes"] > 0, (
        "class_list resolved empty; the J1 config.classes/legacy fallback "
        "should still find the mock entry's active_class_list."
    )
    # J5: provenance fields surfaced (value may be None for a legacy manifest).
    assert "collation_id" in entry
    assert "ablation_id" in entry
