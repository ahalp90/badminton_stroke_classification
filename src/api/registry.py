"""Tier 1 read-only registry endpoints.

Loads docs/models_registry.yaml + each entry's manifest.yaml, and serves the
sidecar JSONs (predictions, per-class stats, clip_index) per the contract
in frontend_integration_guide.md sections 1-3.

No PyTorch. No /scratch dependency. The video endpoint is a stub for now
because clip mp4s live on /scratch on UNE HPC; we wire properly once
serving infra is in place.
"""
import gzip
import json
import logging
from functools import lru_cache
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .config import BST_X_CLIPS_DIR, LOCAL_CLIPS_DIR, REGISTRY_PATH, REPO_ROOT

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

VALID_SPLITS = {"val", "test"}


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        log.warning("registry yaml not found at %s", REGISTRY_PATH)
        return {"models": []}
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {"models": []}


@lru_cache(maxsize=8)
def _load_manifest(manifest_rel_path: str) -> dict:
    abs_path = REPO_ROOT / manifest_rel_path
    if not abs_path.exists():
        log.warning("manifest not found at %s", abs_path)
        return {}
    with open(abs_path) as f:
        return yaml.safe_load(f) or {}


def _read_json_under_run(manifest_rel_path: str, *relparts: str) -> dict:
    run_dir = (REPO_ROOT / manifest_rel_path).parent
    json_path = run_dir.joinpath(*relparts)
    # Sidecars come in two flavours: plain .json (e.g. BRIC's predictions) and
    # gzipped .json.gz (the BST-X fe-deliverables are gzipped to keep the repo
    # light). Prefer the plain file when present, else transparently read the
    # gzipped sibling. Returns {} when neither exists.
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    gz_path = json_path.parent / (json_path.name + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {}


# FE sidecar JSONs (clip_index, {split}.json, perclass_stats_{split}.json) live
# in a different subdir per architecture: BST-X parks them under the run's
# `fe_jsons/` dir (the taxon_pinned_w_preds refactor), BRIC writes `predictions/`
# directly under the run dir. Resolve the subdir off the entry's architecture so
# the readers below stay arch-agnostic.
_SIDECAR_SUBDIR = {"bst-x": "fe_jsons", "bric": "predictions"}


def _sidecar_path(entry: dict, *relparts: str) -> tuple:
    """Prefix `relparts` with the sidecar subdir for this entry's architecture."""
    sub = _SIDECAR_SUBDIR.get(entry.get("architecture", "bst-x"), "predictions")
    return (sub, *relparts)


def _live_splits() -> set[str]:
    """Splits where a live BST forward pass could run (the SCP'd tensors are
    present). Retained for the live-inference / Tier 3 workstream; the per-clip
    browser no longer gates on this (see `_pred_splits`), because the registered
    models ship precomputed real predictions that we serve directly."""
    try:
        from . import bst_x_inference
        return bst_x_inference.available_splits()
    except Exception:  # noqa: BLE001 — any import/probe failure => no live
        log.info("registry: live inference unavailable; predictions disabled")
        return set()


def _is_mock_predictions(preds: dict) -> bool:
    """True when a predictions sidecar carries placeholder y_pred (the old
    demo data, flagged `_mock_data`). We never serve those as real."""
    return bool(preds.get("_mock_data"))


def _pred_splits(entry: dict) -> set[str]:
    """Splits whose per-clip browser should light up: real (non-mock, non-empty)
    precomputed predictions AND a clip_index to drive the list/detail endpoints.

    BST-X ships both, so its browser shows the precomputed real predictions with
    no live forward pass and no /data tensor mount. BRIC ships real predictions
    but no clip_index (it's a metrics-only card by design), so its browser stays
    off rather than 404ing the clip_index-dependent endpoints."""
    clip_index = _read_json_under_run(entry["manifest_path"], *_sidecar_path(entry, "clip_index.json"))
    if not clip_index:
        return set()
    out: set[str] = set()
    for s in ("test", "val"):
        preds = _read_json_under_run(entry["manifest_path"], *_sidecar_path(entry, f"{s}.json"))
        if preds and preds.get("clips") and not _is_mock_predictions(preds):
            out.add(s)
    return out


def _get_model_entry(model_id: str) -> dict:
    for m in _load_registry().get("models", []):
        if m.get("id") == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'")


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise HTTPException(
            status_code=400,
            detail=f"Split must be one of {sorted(VALID_SPLITS)}; got '{split}'",
        )


def _format_metrics(raw: dict) -> dict:
    """Round + slim down a manifest serial-metrics block for the FE."""
    if not raw:
        return {}
    return {
        "macro_f1": round(raw["macro_f1"], 4) if "macro_f1" in raw else None,
        "min_f1": round(raw["min_f1"], 4) if "min_f1" in raw else None,
        "accuracy": round(raw["accuracy"], 4) if "accuracy" in raw else None,
        "top2_accuracy": round(raw["top2_accuracy"], 4) if "top2_accuracy" in raw else None,
        "per_class_f1": {k: round(v, 4) for k, v in raw.get("per_class_f1", {}).items()},
    }


def _resolve_class_list(manifest: dict) -> list:
    """Source the class list from whichever field the manifest carries.

    Fallback order:
    1. ``config.classes`` — canonical (post-refactor BST + BRIC manifests).
    2. ``extra.arch.active_class_list`` — legacy BST runs (pre-refactor); also
       lights up BRIC entries, whose class list was never in the bandaid block.
    Empty list when neither is present; the handler then surfaces num_classes=0.
    """
    cfg_classes = manifest.get("config", {}).get("classes")
    if isinstance(cfg_classes, list) and cfg_classes:
        return cfg_classes
    legacy = manifest.get("extra", {}).get("arch", {}).get("active_class_list")
    if isinstance(legacy, list) and legacy:
        return legacy
    return []


def _summarise_model(entry: dict) -> dict:
    manifest = _load_manifest(entry["manifest_path"])
    config = manifest.get("config", {})
    architecture = entry.get("architecture", "bst-x")
    # Metrics + status are arch-specific: BST-X reads per-serial manifest
    # metrics; BRIC reads its eval-summary sidecar. Class list is unified below.
    # Val rides in the BST-X manifest per serial under
    # extra.val_at_best_macro_epoch, aggregates backfilled to match the test
    # field set, so it reads off the manifest like test does — no second forward
    # pass and no val_metrics.json sidecar. BRIC does not support val by design.
    if architecture == "bst-x":
        serial = next(
            (s for s in manifest.get("serials", [])
             if s.get("serial_no") == entry["serial_no"]),
            {},
        )
        test_metrics_raw = serial.get("metrics", {})
        val_metrics_raw = serial.get("extra", {}).get("val_at_best_macro_epoch", {})
        status = "available" if manifest.get("serials") else "pending"
    else:
        test_summary = _read_json_under_run(entry["manifest_path"], "eval/test_summary.json")
        test_metrics_raw = test_summary.get("metrics", {})
        val_metrics_raw = {}
        status = "available" if test_metrics_raw else "pending"
    class_list = _resolve_class_list(manifest)
    # The browser shows when real precomputed per-clip predictions exist for a
    # split, not when live tensors happen to be mounted. Keeps the field name
    # `live_predictions` for FE compatibility; its meaning is now "per-clip
    # predictions available to browse".
    pred = _pred_splits(entry) if status == "available" else set()
    return {
        "id": entry["id"],
        "display_name": entry.get("display_name", entry["id"]),
        "is_default": entry.get("is_default", False),
        "description": entry.get("description", ""),
        # The author-written training notes are owned by the manifest (top-level
        # `notes` block), not the registry YAML — same principle as collation_id /
        # class_list above. Read them off the manifest so the FE shows the real,
        # un-paraphrased text. None when the manifest carries no notes (e.g. BRIC).
        "notes": manifest.get("notes"),
        "taxonomy": entry.get("taxonomy"),
        "split_column": entry.get("split_column"),
        "drop_unknown": entry.get("drop_unknown", True),
        # collation_id / ablation_id are provenance the manifest now owns
        # (config block); read them off it rather than re-copying into the YAML.
        # ablation_id is the training-time tag (None for non-ablation runs).
        "collation_id": config.get("collation_id"),
        "ablation_id": config.get("ablation_id"),
        "architecture": architecture,
        "temperature": entry.get("temperature", 1.0),
        # Prefer the manifest-derived class list (authoritative once trained);
        # fall back to the registry's declared count for not-yet-trained models
        # whose manifest doesn't exist, so the card shows e.g. 14 not 0.
        "num_classes": len(class_list) or entry.get("num_classes", 0),
        "class_list": class_list,
        # Splits this model actually has metrics for: BST-X carries val + test;
        # BRIC has test only (no val by design). Drives the FE split toggle, so
        # a model never offers a split it has no data for.
        "splits_available": [
            s for s, mx in (("val", val_metrics_raw), ("test", test_metrics_raw)) if mx
        ],
        "status": status,
        "live_predictions": {s: (s in pred) for s in ("test", "val")},
        "test_metrics": _format_metrics(test_metrics_raw),
        "val_metrics": _format_metrics(val_metrics_raw),
    }
    

def _read_clip_index(entry: dict) -> dict:
    raw = _read_json_under_run(entry["manifest_path"], *_sidecar_path(entry, "clip_index.json"))
    if not raw:
        raise HTTPException(status_code=404, detail=f"No clip_index for {entry['id']}")
    return _unwrap_clip_index(raw)


def _unwrap_clip_index(raw: dict) -> dict:
    """Mock wraps the lookup in `clips`; real data may be flat. Tolerate both."""
    if "clips" in raw:
        return raw["clips"]
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _build_stem_index() -> dict:
    """Flatten every registered model's clip_index into stem -> video_path.

    Stems are globally unique across the ShuttleSet collation tree, so the
    first model's entry wins; subsequent models just fill any gaps."""
    out: dict[str, str] = {}
    for entry in _load_registry().get("models", []):
        raw = _read_json_under_run(entry["manifest_path"], *_sidecar_path(entry, "clip_index.json"))
        if not raw:
            continue
        for stem, meta in _unwrap_clip_index(raw).items():
            video_path = meta.get("video_path") if isinstance(meta, dict) else None
            if stem not in out and video_path:
                out[stem] = video_path
    return out


@lru_cache(maxsize=1)
def _scan_clips_dir() -> dict:
    """stem -> path relative to BST_X_CLIPS_DIR, by walking the clips tree once.

    The registered models ship clip_index entries with no `video_path`, so
    `_build_stem_index` resolves nothing. Clip stems are globally unique, so a
    single recursive scan of the on-disk clips library (`{split}/{Side}_{class}/
    {stem}.mp4`) gives a robust stem -> file map without depending on the
    taxonomy/side folder naming. Cached: one walk per process; empty (and
    harmless) when BST_X_CLIPS_DIR is unset or absent."""
    out: dict[str, str] = {}
    if BST_X_CLIPS_DIR is None or not BST_X_CLIPS_DIR.exists():
        return out
    for p in BST_X_CLIPS_DIR.rglob("*.mp4"):
        out.setdefault(p.stem, str(p.relative_to(BST_X_CLIPS_DIR)))
    return out


def _resolve_clip_path(stem: str) -> Optional[str]:
    """Path (relative to BST_X_CLIPS_DIR) for a stem: prefer an explicit
    clip_index `video_path`, else fall back to the filesystem scan."""
    return _build_stem_index().get(stem) or _scan_clips_dir().get(stem)


def _read_predictions(entry: dict, split: str) -> dict:
    preds = _read_json_under_run(entry["manifest_path"], *_sidecar_path(entry, f"{split}.json"))
    if not preds:
        raise HTTPException(status_code=404, detail=f"No predictions for {entry['id']} split={split}")
    return preds


def _summary_live(stem: str, live: dict, idx_entry: dict) -> dict:
    """Per-clip row from a real live forward pass. Currently unused: the list
    serves precomputed real predictions (`_summary_from_record`). Retained for
    the live-inference / Tier 3 workstream, which documents this response shape."""
    return {
        "clip_stem": stem,
        "true_class": live["true_class"],
        "predicted_class": live["predicted_class"],
        "is_correct": live["predicted_class"] == live["true_class"],
        "confidence_pct": live["confidence_pct"],
        "match": idx_entry.get("match"),
        "split": idx_entry.get("split"),
    }


def _summary_no_pred(record: dict, class_list: list, idx_entry: dict) -> dict:
    """Per-clip row when live inference is unavailable: ground truth only, no
    predictions. We never serve the placeholder y_pred as if it were real."""
    y_true = record["y_true"]
    return {
        "clip_stem": record["clip_stem"],
        "true_class": class_list[y_true] if 0 <= y_true < len(class_list) else None,
        "predicted_class": None,
        "is_correct": None,
        "confidence_pct": None,
        "match": idx_entry.get("match"),
        "split": idx_entry.get("split"),
    }


def _summary_from_record(record: dict, class_list: list, idx_entry: dict) -> dict:
    """Per-clip row from the precomputed real predictions in the split JSON.

    The registered models ship real `y_pred` / `top_k` in their sidecar (a
    run-level offline forward pass over the whole split), so the list serves
    those directly: correct, consistent with the metrics panel, and O(1) per
    clip. A live forward pass per row would be thousands of passes per page
    load at full-test-set scale, and (today) from a different, stale model."""
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    tkp = record.get("top_k_prob") or []
    return {
        "clip_stem": record["clip_stem"],
        "true_class": class_list[y_true] if 0 <= y_true < len(class_list) else None,
        "predicted_class": class_list[y_pred] if 0 <= y_pred < len(class_list) else None,
        "is_correct": y_true == y_pred,
        "confidence_pct": int(round(tkp[0] * 100)) if tkp else None,
        "match": idx_entry.get("match"),
        "split": idx_entry.get("split"),
    }


def _summaries_for(preds: dict, clip_index: dict) -> list:
    """Build all per-clip summaries for a split from the precomputed
    predictions JSON. Real predictions are served as-is; mock/placeholder
    predictions degrade to ground-truth-only so we never present a fake
    prediction as real."""
    # Canonical `class_list` (post-refactor + BRIC sidecars), legacy
    # `active_class_list` fallback (matches get_clip).
    class_list = preds.get("class_list") or preds.get("active_class_list", [])
    mock = _is_mock_predictions(preds)
    out = []
    for r in preds.get("clips", []):
        idx_entry = clip_index.get(r["clip_stem"], {})
        if mock:
            out.append(_summary_no_pred(r, class_list, idx_entry))
        else:
            out.append(_summary_from_record(r, class_list, idx_entry))
    return out


@router.get("/registry")
def list_models() -> dict:
    return {"models": [_summarise_model(m) for m in _load_registry().get("models", [])]}


@router.get("/registry/{model_id}")
def get_model(model_id: str) -> dict:
    return _summarise_model(_get_model_entry(model_id))


@router.get("/registry/{model_id}/splits/{split}/stats")
def get_perclass_stats(model_id: str, split: str) -> dict:
    _validate_split(split)
    entry = _get_model_entry(model_id)
    stats = _read_json_under_run(
        entry["manifest_path"], *_sidecar_path(entry, f"perclass_stats_{split}.json")
    )
    if not stats:
        raise HTTPException(status_code=404, detail=f"No perclass_stats for {model_id} split={split}")
    return stats


@router.get("/registry/{model_id}/splits/{split}/clips")
def list_clips(
    model_id: str,
    split: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    true_class: Optional[str] = None,
    predicted_class: Optional[str] = None,
    errors_only: bool = False,
) -> dict:
    _validate_split(split)
    entry = _get_model_entry(model_id)
    preds = _read_predictions(entry, split)
    clip_index = _read_clip_index(entry)

    real = not _is_mock_predictions(preds)
    summaries = _summaries_for(preds, clip_index)

    if true_class:
        summaries = [s for s in summaries if s["true_class"] == true_class]
    if predicted_class:
        summaries = [s for s in summaries if s["predicted_class"] == predicted_class]
    if errors_only:
        summaries = [s for s in summaries if s["is_correct"] is False]

    return {
        "model_id": model_id,
        "split": split,
        # Real precomputed predictions are present (vs ground-truth-only mock).
        # Field name kept for FE compatibility.
        "live": real,
        "total": len(summaries),
        "limit": limit,
        "offset": offset,
        "clips": summaries[offset:offset + limit],
    }

@router.get("/registry/{model_id}/splits/{split}/matches")
def list_matches(model_id: str, split: str) -> dict:
    """Return unique match names for a split.

    Avoids the frontend having to fetch all clips just to extract match names,
    and sidesteps the 500 clip limit on /clips. Used by LibraryScreen to filter
    the match library to only videos in the test set.
    """
    _validate_split(split)
    entry = _get_model_entry(model_id)
    preds = _read_predictions(entry, split)
    clip_index = _read_clip_index(entry)
    matches = sorted({
        clip_index.get(r["clip_stem"], {}).get("match")
        for r in preds.get("clips", [])
        if clip_index.get(r["clip_stem"], {}).get("match")
    })
    return {"model_id": model_id, "split": split, "matches": matches}
    
@router.get("/registry/{model_id}/splits/{split}/clips/{stem}")
def get_clip(model_id: str, split: str, stem: str) -> dict:
    _validate_split(split)
    entry = _get_model_entry(model_id)
    preds = _read_predictions(entry, split)
    clip_index = _read_clip_index(entry)
    # Canonical `class_list`, legacy `active_class_list` fallback (see /clips).
    class_list = preds.get("class_list") or preds.get("active_class_list", [])

    record = next((r for r in preds.get("clips", []) if r["clip_stem"] == stem), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Clip '{stem}' not in split={split}")
    idx_entry = clip_index.get(stem, {})

    # Serve the precomputed real predictions from the registered model's split
    # sidecar. This keeps the detail view consistent with the list and the
    # per-class metrics panel (all sourced from the same model's offline run).
    #
    # The on-demand live forward pass (bst_x_inference) is intentionally NOT
    # called here: it is currently pinned to a different, older checkpoint than
    # the registered model and only indexes a 56-clip subset, so firing it would
    # serve a clip a prediction from a model the rest of the page doesn't use.
    # Re-pointing bst_x_inference at the registered run + adding row_index belongs
    # to the live-inference / Tier 3 workstream. See the Tier 2 handoff note.
    top_k = [
        {"class": class_list[i], "confidence": p}
        for i, p in zip(record["top_k_idx"], record["top_k_prob"])
    ]
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    # Honest provenance for the FE: real precomputed predictions vs the old
    # placeholder data (y_pred == y_true at 100%). The registered models ship
    # real preds, so this is "cached_predictions_json"; a flagged mock sidecar
    # reports "placeholder_predictions_json" so the FE shows a pending state
    # instead of a fake "100% correct" result.
    drawn_from = (
        "placeholder_predictions_json"
        if _is_mock_predictions(preds)
        else "cached_predictions_json"
    )
    return {
        "clip_stem": stem,
        "video_url": f"/api/clips/{stem}/video",
        "true_class": class_list[y_true] if 0 <= y_true < len(class_list) else None,
        "predicted_class": class_list[y_pred] if 0 <= y_pred < len(class_list) else None,
        "is_correct": y_true == y_pred,
        "confidence_pct": int(round(record["top_k_prob"][0] * 100)) if record.get("top_k_prob") else 0,
        "top_k": top_k,
        "match": idx_entry.get("match"),
        "set_id": idx_entry.get("set_id"),
        "rally": idx_entry.get("rally"),
        "ball_round": idx_entry.get("ball_round"),
        "split": idx_entry.get("split", split),
        "drawn_from": drawn_from,
    }


@router.get("/clips/{stem}/video")
def get_video(stem: str):
    """Serve the clip mp4 for a stem.

    Resolution order:
      1. Local stem-keyed drop: LOCAL_CLIPS_DIR/<stem>.mp4. Lets you play a
         handful of real clips locally (dev/demo) without the full ShuttleSet
         tree or BST_X_CLIPS_DIR. Keyed by the stable clip_stem, not clip_index's
         (placeholder) video_path.
      2. Dataset tree under BST_X_CLIPS_DIR, resolved by stem: an explicit
         clip_index video_path if present, else a filesystem scan of the clips
         library. The registered models ship null video_path, so the scan is
         what resolves the full test/val set.

    FileResponse handles Range requests automatically, so the <video> element
    on the FE can scrub. Returns 404 with a hint when neither resolves."""
    # 1) Local stem-keyed file. The parent-equality check guards against path
    #    traversal via a crafted stem (e.g. "../secret").
    local = LOCAL_CLIPS_DIR / f"{stem}.mp4"
    if local.resolve().parent == LOCAL_CLIPS_DIR.resolve() and local.is_file():
        return FileResponse(local, media_type="video/mp4")

    # 2) Dataset tree under BST_X_CLIPS_DIR (clip_index video_path, else scan).
    if BST_X_CLIPS_DIR is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No local clip at {local} and BST_X_CLIPS_DIR env var not set; "
                "can't resolve clip mp4 location. Drop <stem>.mp4 into "
                "clips_local/, or on UNE HPC set "
                "BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips."
            ),
        )
    rel_path = _resolve_clip_path(stem)
    if rel_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Clip '{stem}' has no mp4 on this host: no local clip at "
                f"{local}, no clip_index video_path, and nothing matching under "
                "BST_X_CLIPS_DIR."
            ),
        )
    abs_path = BST_X_CLIPS_DIR / rel_path
    if not abs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Clip file not found at {abs_path}, and no local clip at "
                f"{local}. Likely a missing clip on this host."
            ),
        )
    return FileResponse(abs_path, media_type="video/mp4")
