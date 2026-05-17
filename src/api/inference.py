"""Inference stub for the badminton stroke classifier API.

The real BST/MMPose/TrackNet pipeline lives in src/bst_refactor/ and is the
remit of the ML team (Ari, Scott). This module is a contract-shaped *stub*
that supplies the FE with realistic-looking results so the rest of the
stack (upload → status → results, library_predict, the Results panel) can
be exercised end-to-end before real inference lands.

The stub draws each "prediction" from the mocked test-split predictions
JSON shipped under the registered model's run directory. Different uploads
get different drawn entries; each annotation in the user's markup yields
one stroke. The behaviour is documented in
scratch/inspect_clips/handoff_report.md §9 (Fix 2).
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from .config import REPO_ROOT
from .registry import _load_registry, _read_json_under_run

log = logging.getLogger(__name__)

# How fast the canned animation pretends to be. Keep this aligned with the
# FE's progress poll cadence so the user sees the stage transitions land
# rather than the bar snapping to 100%.
_STUB_LATENCY_SEC = 3.0


def _pick_predictions_pool() -> tuple[list[str], list[dict]]:
    """Return (active_class_list, list of test-split prediction records).

    Uses the first registered model's test predictions. The 56-clip mock
    on this branch is plenty of variety for demo draws; if more models
    register later, we'd want to honour the per-job model_id, but the
    stub doesn't differentiate today."""
    models = _load_registry().get("models", [])
    if not models:
        return [], []
    preds = _read_json_under_run(models[0]["manifest_path"], "predictions", "test.json")
    return preds.get("active_class_list", []), preds.get("clips", [])


def _build_top_k(record: dict, class_list: list[str]) -> list[dict]:
    pairs = list(zip(record.get("top_k_idx", []), record.get("top_k_prob", [])))
    return [
        {"class": class_list[i] if 0 <= i < len(class_list) else f"class_{i}",
         "confidence": round(float(p), 4)}
        for i, p in pairs
    ]


def _synthesise_realistic_topk(
    pred_idx: int,
    class_list: list[str],
    rng: random.Random,
    k: int = 5,
) -> tuple[float, list[dict]]:
    """Return (headline_confidence, top_k list) with a plausible-looking
    softmax distribution centred on `pred_idx`.

    Used when the underlying eval pool is degenerate (e.g. the current
    test.json reports `top_k_prob: [1.0]` for every clip because the
    model has memorised the remapped test stems). A real-looking demo
    needs confidences in the 30-95% range with runner-up classes nearby.
    """
    headline = round(rng.uniform(0.42, 0.92), 4)
    remainder = 1.0 - headline
    # Random complementary weights for the runner-ups, scaled so the
    # whole distribution sums to 1.0 with a graceful decay.
    n_others = min(k - 1, max(0, len(class_list) - 1))
    if n_others == 0:
        return headline, [{
            "class": class_list[pred_idx] if 0 <= pred_idx < len(class_list) else f"class_{pred_idx}",
            "confidence": headline,
        }]
    raw = sorted((rng.random() for _ in range(n_others)), reverse=True)
    raw_sum = sum(raw) or 1.0
    others = [r / raw_sum * remainder for r in raw]
    # Pick other classes from the pool minus the predicted one.
    other_idxs = [i for i in range(len(class_list)) if i != pred_idx]
    rng.shuffle(other_idxs)
    other_idxs = other_idxs[:n_others]
    top_k = [{
        "class": class_list[pred_idx] if 0 <= pred_idx < len(class_list) else f"class_{pred_idx}",
        "confidence": headline,
    }]
    for ci, w in zip(other_idxs, others):
        top_k.append({
            "class": class_list[ci] if 0 <= ci < len(class_list) else f"class_{ci}",
            "confidence": round(float(w), 4),
        })
    return headline, top_k


def _synthesise_strokes(
    annotations: list[dict],
    class_list: list[str],
    pool: list[dict],
    rng: random.Random,
    default_fps: float = 30.0,
) -> list[dict]:
    """One stroke per annotation, drawing predictions from random pool entries.

    If the user supplied no annotations, we return a single canned stroke
    so the Results card has something to show."""
    if not pool or not class_list:
        return [
            {"timestamp_sec": 2.1, "stroke_type": "clear", "confidence": 0.92,
             "stroke_index": 0, "predicted_class": "clear", "confidence_pct": 92,
             "top_k": [{"class": "clear", "confidence": 0.92}]},
        ]

    n = max(1, len(annotations))
    picks = rng.sample(pool, min(n, len(pool)))
    if len(picks) < n:
        # Pool exhausted (unlikely with 28 test entries); cycle through.
        picks = (picks * ((n // len(picks)) + 1))[:n]

    strokes: list[dict] = []
    for i in range(n):
        pick = picks[i]
        anno = annotations[i] if i < len(annotations) else {}
        target_frame = int(anno.get("target_frame", 0))
        # FPS isn't carried by the markup contract; use 30 fps as a
        # conventional demo placeholder. Library matches *do* know their
        # fps but this stub doesn't have it threaded through.
        ts = round(target_frame / default_fps, 2) if target_frame else round(i * 1.5 + 0.5, 2)
        pred_idx = pick["y_pred"] if isinstance(pick.get("y_pred"), int) else 0
        pred_class = class_list[pred_idx] if 0 <= pred_idx < len(class_list) else "unknown"
        true_class = class_list[pick["y_true"]] if 0 <= pick["y_true"] < len(class_list) else None

        # The eval pool we draw from on this branch is degenerate: every
        # entry's `top_k_prob` is `[1.0]` because the test stems were
        # remapped to clips the model trained on. That makes every stroke
        # show 100% confidence in the Results card, which reads as a bug.
        # When we detect that flat shape, synthesise a plausible softmax
        # distribution centred on the predicted class for the demo. Real
        # multi-element top_k entries (the mock data has 5 each) flow
        # through unchanged.
        raw_top = pick.get("top_k_prob", [0.0])
        raw_first = float(raw_top[0]) if raw_top else 0.0
        degenerate = (len(raw_top) <= 1 and raw_first >= 0.99)
        if degenerate:
            top_prob, top_k_entries = _synthesise_realistic_topk(
                pred_idx, class_list, rng,
            )
        else:
            top_prob = raw_first
            top_k_entries = _build_top_k(pick, class_list)

        strokes.append({
            # Legacy fields the current FE renders directly.
            "timestamp_sec": ts,
            "stroke_type":   pred_class,
            "confidence":    round(top_prob, 4),
            # Richer contract-shaped fields (extras; not yet consumed by the
            # FE, but stable so they can be wired without backend changes).
            "stroke_index":     i,
            "target_frame":     target_frame,
            "player_side":      anno.get("player_side"),
            "predicted_class":  pred_class,
            "confidence_pct":   int(round(top_prob * 100)),
            "top_k":            top_k_entries,
            "true_class_hint":  true_class,  # for the curious; not part of contract
            "drawn_from_stem":  pick.get("clip_stem"),
        })
    return strokes


def run_inference(
    video_path: str,
    model_name: str,
    markup: Optional[dict] = None,
) -> dict:
    """Smart stub: returns realistic-looking predictions drawn from the
    mocked test-split JSON on disk.

    Args:
        video_path: ignored by the stub (still logged for parity).
        model_name: ignored by the stub.
        markup:     optional, the validated /api/upload (or /api/library_predict)
                    sidecar. When present, one stroke is generated per
                    annotations[] entry. When absent or empty, a single
                    canned stroke is returned.

    Returns:
        dict with `strokes[]` and `rally_summary`. Each stroke carries both
        the legacy {timestamp_sec, stroke_type, confidence} shape AND the
        contract-shaped {stroke_index, predicted_class, confidence_pct,
        top_k, player_side, target_frame} fields drawn from a random
        mocked-test entry. Different jobs draw different entries.
    """
    log.info(
        "inference stub: video=%s model=%s annotations=%d",
        Path(video_path).name, model_name,
        len(((markup or {}).get("annotations")) or []),
    )
    time.sleep(_STUB_LATENCY_SEC)

    class_list, pool = _pick_predictions_pool()
    annotations = list((markup or {}).get("annotations") or [])

    # Seed from time so consecutive uploads draw distinct entries.
    rng = random.Random(time.time_ns())
    strokes = _synthesise_strokes(annotations, class_list, pool, rng)

    # Rally length spans the FULL marked window:
    # (max(region_end_frame) - min(region_start_frame)) / fps.
    # Previously we used last-stroke-timestamp minus first-stroke-timestamp +
    # 2 fudge seconds, which under-reported because target_frame sits inside
    # the window rather than at its edges. With zero annotations there's
    # no marked rally, so we report 0.0.
    fps = 30.0  # markup contract doesn't carry fps; FE also assumes 30
    starts = [int(a.get("region_start_frame", 0)) for a in annotations
              if a.get("region_start_frame") is not None]
    ends = [int(a.get("region_end_frame", 0)) for a in annotations
            if a.get("region_end_frame") is not None]
    if starts and ends:
        rally_length = round(max(0.0, (max(ends) - min(starts)) / fps), 1)
    else:
        rally_length = 0.0

    return {
        "strokes": strokes,
        "rally_summary": {
            "total_strokes": len(strokes),
            "rally_length_seconds": rally_length,
        },
    }
