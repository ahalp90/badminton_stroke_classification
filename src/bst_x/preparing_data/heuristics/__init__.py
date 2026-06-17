"""Heuristic registry for ``apply_heuristic.py``.

Each heuristic variant lives in its own module under this package and
exposes an ``apply`` function with the signature:

    apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput

Dispatch is by name via ``REGISTRY``.
"""
from __future__ import annotations

from . import current, sticky_anchor
from .base import ClipContext, HeuristicOutput, RawClip

REGISTRY = {
    "current": current.apply,
    "sticky_anchor": sticky_anchor.apply,
}

__all__ = ["REGISTRY", "ClipContext", "HeuristicOutput", "RawClip"]
