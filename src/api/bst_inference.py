"""Live BST forward-pass inference for library clips.

Loads the trained BST checkpoint at module import time, plus a per-split
mmap of the SCP'd collated tensors. predict(stem, split) slices one row
out of the collation, builds the (B=1) input batch, runs forward pass,
and returns the model's actual top-k predictions.

This is the *real* inference path for the 56 library clips that this
branch has predictions for. Upload-flow inference (`src/api/inference.py`)
remains a smart stub — that's a separate Ari/Scott workstream.

Bind mounts required (set in docker-compose.yml):
    ./scratch/bst_inputs/{test,val}/{JnB_bone,pos,shuttle,videos_len,labels}.npy
        -> /app/bst_inputs/{test,val}/...

The checkpoint and clip_index.json live in the repo tree at
`src/bst_refactor/.../experiments/run_20260505_154907/`.

The clip_index.json carries `row_index` per stem (added by
scratch/inspect_clips/rebuild_real.py); we use that directly rather than
re-deriving from clips_master.csv at runtime.
"""
from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch

log = logging.getLogger(__name__)


# ─── Path bootstrap ─────────────────────────────────────────────────
# bst_refactor's modules use bare imports (`from pipeline.config import ...`)
# rather than fully-qualified paths, so we have to extend sys.path the same
# way bst_infer.py's docstring tells you to via PYTHONPATH.
REPO_ROOT = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[2]
BST_REFACTOR = REPO_ROOT / "src" / "bst_refactor"
BST_CLASSIFICATION = BST_REFACTOR / "stroke_classification"
for p in (BST_CLASSIFICATION, BST_REFACTOR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ─── Constants ──────────────────────────────────────────────────────
RUN_DIR = REPO_ROOT / "src" / "bst_refactor" / "stroke_classification" / "main_on_shuttleset" / "experiments" / "run_20260505_154907"
WEIGHTS_PATH = RUN_DIR / "weights" / "bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_une_merge_v1_nosides_5.pt"
CLIP_INDEX_PATH = RUN_DIR / "clip_index.json"

BST_INPUTS_DIR = Path("/app/bst_inputs") if Path("/app/bst_inputs").exists() else REPO_ROOT / "scratch" / "bst_inputs"
SPLITS = ("test", "val")
POSE_STYLE = "JnB_bone"

# 14-class une_merge_v1_nosides taxonomy (extra.arch.active_class_list in manifest.yaml).
ACTIVE_CLASS_LIST = [
    "net_shot", "return_net", "smash", "wrist_smash",
    "lob", "clear", "drive", "drop",
    "passive_drop", "push", "rush", "cross_court_net_shot",
    "short_service", "long_service",
]
N_CLASS = len(ACTIVE_CLASS_LIST)
TOP_K = 5

DEVICE = "cpu"  # backend container is CPU-only by design


# ─── Lazy globals ───────────────────────────────────────────────────
_model: torch.nn.Module | None = None
_model_lock = Lock()
_tensors: dict[str, dict[str, np.ndarray]] = {}   # split -> {name: mmap}
_stem_to_meta: dict[str, dict] = {}                # stem -> {row_index, split, ...}


# ─── Initialisation ─────────────────────────────────────────────────
def _build_model() -> torch.nn.Module:
    """Instantiate BST_CG_AP at the right shape and load serial-5 weights."""
    from main_on_shuttleset.bst_common import build_bst_network
    net, _n_bones = build_bst_network(
        "BST_CG_AP",
        n_joints=17,
        pose_style=POSE_STYLE,
        in_channels=2,
        n_class=N_CLASS,
        seq_len=100,
        device=DEVICE,
    )
    state = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    log.info("bst_inference: loaded %s (%.1f MB) on %s",
             WEIGHTS_PATH.name, WEIGHTS_PATH.stat().st_size / 1e6, DEVICE)
    return net


def _load_split_tensors(split: str) -> dict[str, np.ndarray]:
    """Memory-map the four .npy tensors for one split."""
    base = BST_INPUTS_DIR / split
    if not base.exists():
        raise FileNotFoundError(f"BST inputs missing: {base}")
    out = {
        "JnB_bone":   np.load(str(base / "JnB_bone.npy"),   mmap_mode="r"),
        "pos":        np.load(str(base / "pos.npy"),        mmap_mode="r"),
        "shuttle":    np.load(str(base / "shuttle.npy"),    mmap_mode="r"),
        "videos_len": np.load(str(base / "videos_len.npy"), mmap_mode="r"),
    }
    log.info("bst_inference: mmap'd %s split — %d rows, JnB_bone %s",
             split, out["JnB_bone"].shape[0], out["JnB_bone"].shape)
    return out


def _load_stem_index() -> dict[str, dict]:
    """stem -> {row_index, split, raw_type_en, ...} from clip_index.json."""
    if not CLIP_INDEX_PATH.exists():
        raise FileNotFoundError(f"clip_index.json missing: {CLIP_INDEX_PATH}")
    with open(CLIP_INDEX_PATH) as f:
        ci = json.load(f)
    return ci.get("clips", {})


def _ensure_initialised() -> None:
    """Load model + tensors + stem index. Thread-safe, idempotent."""
    global _model, _tensors, _stem_to_meta
    with _model_lock:
        if _model is None:
            _model = _build_model()
        if not _tensors:
            for s in SPLITS:
                try:
                    _tensors[s] = _load_split_tensors(s)
                except FileNotFoundError as e:
                    log.warning("bst_inference: %s", e)
        if not _stem_to_meta:
            _stem_to_meta = _load_stem_index()


# ─── Public API ─────────────────────────────────────────────────────
class BstInferenceUnavailable(Exception):
    """Raised when real inference can't proceed (missing data, bad stem, etc).

    The API layer catches this and falls back to the smart stub."""


def is_available() -> bool:
    """Cheap availability check used by /api/library_predict gating."""
    try:
        _ensure_initialised()
        return _model is not None and any(_tensors.values()) and bool(_stem_to_meta)
    except Exception as e:
        log.warning("bst_inference: not available: %s", e)
        return False


@torch.no_grad()
def predict(stem: str, split: str | None = None) -> dict:
    """Run a real BST forward pass on the requested stem.

    Args:
        stem:  clip stem (e.g. "1_1_10_1") — must be in clip_index.json.
        split: 'test' | 'val'. If None, inferred from clip_index entry.

    Returns:
        {
            "predicted_class": str,
            "confidence_pct":  int,
            "true_class":      str,
            "top_k":           [{"class": str, "confidence": float}, ...],
            "softmax":         [float, ...]  # full distribution over 14 classes
            "drawn_from":      "live_forward_pass",
            "row_index":       int,
            "split":           str,
        }

    Raises:
        BstInferenceUnavailable: if model/tensors/stem index missing.
        KeyError: if the stem isn't in the index.
        ValueError: if the requested split has no SCP'd tensors.
    """
    _ensure_initialised()
    if _model is None:
        raise BstInferenceUnavailable("BST model not loaded")
    meta = _stem_to_meta.get(stem)
    if meta is None:
        raise KeyError(f"stem {stem!r} not in clip_index.json")
    resolved_split = split or meta.get("split")
    if resolved_split not in SPLITS:
        raise ValueError(f"unsupported split {resolved_split!r}")
    if resolved_split not in _tensors:
        raise BstInferenceUnavailable(
            f"no SCP'd tensors for split {resolved_split!r} — "
            f"check {BST_INPUTS_DIR / resolved_split}"
        )
    if "row_index" not in meta:
        raise BstInferenceUnavailable(
            f"stem {stem!r} has no row_index in clip_index.json — "
            "re-run scratch/inspect_clips/rebuild_real.py"
        )

    row = int(meta["row_index"])
    t = _tensors[resolved_split]

    # Slice the single row out of mmap'd arrays. .copy() forces a write
    # into RAM; without it the mmap stays open for the life of the Tensor.
    human_pose = np.asarray(t["JnB_bone"][row]).copy()      # (T=100, M=2, J+B=36, d=2)
    pos        = np.asarray(t["pos"][row]).copy()           # (T, M, 2)
    shuttle    = np.asarray(t["shuttle"][row]).copy()       # (T, 2)
    video_len  = int(t["videos_len"][row])                  # scalar

    # Add batch dim, then flatten the last two pose dims like infer() does.
    human_pose_t = torch.from_numpy(human_pose).unsqueeze(0).to(DEVICE)  # (1, T, M, J+B, d)
    human_pose_t = human_pose_t.view(*human_pose_t.shape[:-2], -1)        # (1, T, M, (J+B)*d)
    pos_t        = torch.from_numpy(pos).unsqueeze(0).to(DEVICE)          # (1, T, M, 2)
    shuttle_t    = torch.from_numpy(shuttle).unsqueeze(0).to(DEVICE)      # (1, T, 2)
    video_len_t  = torch.tensor([video_len], device=DEVICE)               # (1,)

    logits = _model(human_pose_t, shuttle_t, pos_t, video_len_t)  # (1, n_class)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (n_class,)
    top_idx = np.argsort(-probs)[:TOP_K]
    predicted_class = ACTIVE_CLASS_LIST[int(np.argmax(probs))]

    raw_type = meta.get("raw_type_en")
    # Merge per UNE_MERGE_V1_MAP to land in the active class space
    _merge = {
        "defensive_return_lob":   "lob",
        "driven_flight":          "drive",
        "back_court_drive":       "drive",
        "defensive_return_drive": "drive",
    }
    true_class = _merge.get(raw_type, raw_type)

    return {
        "predicted_class": predicted_class,
        "confidence_pct":  int(round(float(probs[np.argmax(probs)]) * 100)),
        "true_class":      true_class,
        "top_k": [
            {"class": ACTIVE_CLASS_LIST[int(i)], "confidence": round(float(probs[int(i)]), 4)}
            for i in top_idx
        ],
        "softmax":   [round(float(p), 4) for p in probs.tolist()],
        "drawn_from": "live_forward_pass",
        "row_index":  row,
        "split":      resolved_split,
    }
