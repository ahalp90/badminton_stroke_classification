"""Tier 1 read-only registry endpoints.

Loads docs/models_registry.yaml + each entry's manifest.yaml, and serves the
sidecar JSONs (predictions, per-class stats, clip_index) per the contract
in frontend_integration_guide.md sections 1-3.

No PyTorch. No /scratch dependency. The video endpoint is a stub for now
because clip mp4s live on /scratch on UNE HPC; we wire properly once
serving infra is in place.
"""
import json
import logging
from functools import lru_cache
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .config import BST_CLIPS_DIR, REGISTRY_PATH, REPO_ROOT

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
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)


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


def _summarise_model(entry: dict) -> dict:
    manifest = _load_manifest(entry["manifest_path"])
    serial_metrics = next(
        (s.get("metrics", {}) for s in manifest.get("serials", [])
         if s.get("serial_no") == entry["serial_no"]),
        {},
    )
    # val_metrics.json is produced by scripts/compute_val_metrics.py, which
    # runs the same checkpoint over /app/bst_inputs/val. Missing file just
    # means the script hasn't been run for this run dir yet — we fall back
    # to {} and the FE shows its "No val metrics available" placeholder.
    val_metrics_raw = _read_json_under_run(entry["manifest_path"], "val_metrics.json")
    class_list = manifest.get("extra", {}).get("arch", {}).get("active_class_list", [])
    return {
        "id": entry["id"],
        "display_name": entry.get("display_name", entry["id"]),
        "description": entry.get("description", ""),
        "taxonomy": entry.get("taxonomy"),
        "split_column": entry.get("split_column"),
        "drop_unknown": entry.get("drop_unknown", True),
        "ablation_id": entry.get("ablation_id"),
        "temperature": entry.get("temperature", 1.0),
        "num_classes": len(class_list),
        "class_list": class_list,
        "splits_available": ["val", "test"],
        "test_metrics": _format_metrics(serial_metrics),
        "val_metrics": _format_metrics(val_metrics_raw),
    }


def _read_clip_index(entry: dict) -> dict:
    raw = _read_json_under_run(entry["manifest_path"], "clip_index.json")
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
        raw = _read_json_under_run(entry["manifest_path"], "clip_index.json")
        if not raw:
            continue
        for stem, meta in _unwrap_clip_index(raw).items():
            video_path = meta.get("video_path") if isinstance(meta, dict) else None
            if stem not in out and video_path:
                out[stem] = video_path
    return out


def _read_predictions(entry: dict, split: str) -> dict:
    preds = _read_json_under_run(entry["manifest_path"], "predictions", f"{split}.json")
    if not preds:
        raise HTTPException(status_code=404, detail=f"No predictions for {entry['id']} split={split}")
    return preds


def _build_summary(record: dict, class_list: list, idx_entry: dict) -> dict:
    y_true = record["y_true"]
    y_pred = record["y_pred"]
    top_prob = record["top_k_prob"][0] if record.get("top_k_prob") else 0.0
    return {
        "clip_stem": record["clip_stem"],
        "true_class": class_list[y_true] if 0 <= y_true < len(class_list) else None,
        "predicted_class": class_list[y_pred] if 0 <= y_pred < len(class_list) else None,
        "is_correct": y_true == y_pred,
        "confidence_pct": int(round(top_prob * 100)),
        "match": idx_entry.get("match"),
        "split": idx_entry.get("split"),
    }


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
        entry["manifest_path"], "predictions", f"perclass_stats_{split}.json"
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
    class_list = preds.get("active_class_list", [])

    summaries = [
        _build_summary(r, class_list, clip_index.get(r["clip_stem"], {}))
        for r in preds.get("clips", [])
    ]
    if true_class:
        summaries = [s for s in summaries if s["true_class"] == true_class]
    if predicted_class:
        summaries = [s for s in summaries if s["predicted_class"] == predicted_class]
    if errors_only:
        summaries = [s for s in summaries if not s["is_correct"]]

    return {
        "model_id": model_id,
        "split": split,
        "total": len(summaries),
        "limit": limit,
        "offset": offset,
        "_mock_data": preds.get("_mock_data", False),
        "clips": summaries[offset:offset + limit],
    }


@router.get("/registry/{model_id}/splits/{split}/clips/{stem}")
def get_clip(model_id: str, split: str, stem: str) -> dict:
    _validate_split(split)
    entry = _get_model_entry(model_id)
    preds = _read_predictions(entry, split)
    clip_index = _read_clip_index(entry)
    class_list = preds.get("active_class_list", [])

    record = next((r for r in preds.get("clips", []) if r["clip_stem"] == stem), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Clip '{stem}' not in split={split}")
    idx_entry = clip_index.get(stem, {})

    # --- Real BST forward pass when the data is available. -------------
    # On this branch, scratch/bst_inputs/{split} carries the SCP'd
    # collated tensors for the 28 test + 28 val real stems, and the
    # repo ships serial_5.pt. bst_inference.predict() slices the
    # right row, builds the (B=1) input, and runs forward. We fall
    # back to whatever predictions JSON the registry has if anything
    # in that chain isn't available — keeps the endpoint working in
    # demo-with-no-bst-inputs deployments.
    live = None
    try:
        from .bst_inference import predict as bst_predict, BstInferenceUnavailable
        live = bst_predict(stem, split)
    except BstInferenceUnavailable as e:
        log.info("clip %s: live BST inference unavailable (%s); using cached predictions", stem, e)
    except Exception as e:
        log.exception("clip %s: live BST inference error; falling back to cached predictions: %s", stem, e)

    if live is not None:
        return {
            "clip_stem": stem,
            "video_url": f"/api/clips/{stem}/video",
            "true_class": live["true_class"],
            "predicted_class": live["predicted_class"],
            "is_correct": live["predicted_class"] == live["true_class"],
            "confidence_pct": live["confidence_pct"],
            "top_k": live["top_k"],
            "match": idx_entry.get("match"),
            "set_id": idx_entry.get("set_id"),
            "rally": idx_entry.get("rally"),
            "ball_round": idx_entry.get("ball_round"),
            "split": idx_entry.get("split", split),
            "drawn_from": live["drawn_from"],
        }

    # --- Fallback: serve cached predictions from the JSON. -------------
    top_k = [
        {"class": class_list[i], "confidence": p}
        for i, p in zip(record["top_k_idx"], record["top_k_prob"])
    ]
    y_true = record["y_true"]
    y_pred = record["y_pred"]
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
        "drawn_from": "cached_predictions_json",
    }


@router.get("/clips/{stem}/video")
def get_video(stem: str):
    """Serve the clip mp4 from BST_CLIPS_DIR.

    FileResponse handles Range requests automatically, so the <video>
    element on the FE can scrub. Returns 404 with a hint when the env
    var is unset (e.g. running natively without /scratch mounted) or
    when the file isn't on this filesystem (e.g. mocked stems)."""
    rel_path = _build_stem_index().get(stem)
    if rel_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Clip '{stem}' not in any registered model's clip_index.",
        )
    if BST_CLIPS_DIR is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "BST_CLIPS_DIR env var not set; can't resolve clip mp4 location. "
                "On UNE HPC this is /scratch/comp320a/ShuttleSet/clips."
            ),
        )
    abs_path = BST_CLIPS_DIR / rel_path
    if not abs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Clip file not found at {abs_path}. "
                "Likely a mocked stem or a missing clip on this host."
            ),
        )
    return FileResponse(abs_path, media_type="video/mp4")
