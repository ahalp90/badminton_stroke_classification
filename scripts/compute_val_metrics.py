"""Compute val-set metrics for the registered BST-X v1 serial-5 model.

Reuses src.api.bst_inference to load the model + the mmap'd val tensors,
then runs a batched forward pass over all 5,250 val rows. Aggregates
y_pred vs labels.npy into macro_f1 / accuracy / top2_accuracy / min_f1 /
per_class_f1 and writes the result alongside the run's other artifacts
as `val_metrics.json`.

Intended for local-only one-shot use; the registry endpoint will pick
the JSON up on the next backend reload. Run from inside the backend
container so the /app/bst_inputs/val mount is reachable:

    docker exec badminton-backend python /app/scripts/compute_val_metrics.py

LOCAL-ONLY PATCH (do not commit) — same handling as the other tier-1
integration patches this session.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_val_metrics")


# Make `src.api.bst_inference` importable whether we're in /app or repo root.
REPO_ROOT = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.bst_inference import (  # noqa: E402  (path bootstrap above)
    ACTIVE_CLASS_LIST,
    BST_INPUTS_DIR,
    DEVICE,
    N_CLASS,
    RUN_DIR,
    _ensure_initialised,
)
from src.api import bst_inference as bi  # noqa: E402

SPLIT = "val"
BATCH = 32
OUTPUT_PATH = RUN_DIR / "val_metrics.json"


def _f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 from a (C, C) confusion matrix (rows=true, cols=pred).

    Hand-rolled to avoid a hard sklearn dependency in case the container
    image ever drops it; sklearn agrees to 1e-7 on the same inputs."""
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    _ensure_initialised()
    if bi._model is None:
        raise RuntimeError("BST model failed to load — check bst_inference logs")
    if SPLIT not in bi._tensors:
        raise RuntimeError(f"Val tensors not mmap'd; check {BST_INPUTS_DIR / SPLIT}")

    model = bi._model
    t = bi._tensors[SPLIT]
    labels_path = BST_INPUTS_DIR / SPLIT / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels missing: {labels_path}")
    y_true = np.load(str(labels_path), mmap_mode="r")
    n_rows = int(y_true.shape[0])
    log.info("val rows: %d, batch size: %d", n_rows, BATCH)

    y_pred = np.empty(n_rows, dtype=np.int64)
    top2_correct = np.zeros(n_rows, dtype=bool)

    fwd_start = time.time()
    for start in range(0, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        sl = slice(start, end)

        human_pose = np.asarray(t["JnB_bone"][sl]).copy()      # (B, T, M, J+B, d)
        pos        = np.asarray(t["pos"][sl]).copy()           # (B, T, M, 2)
        shuttle    = np.asarray(t["shuttle"][sl]).copy()       # (B, T, 2)
        video_len  = np.asarray(t["videos_len"][sl]).copy()    # (B,)

        human_pose_t = torch.from_numpy(human_pose).to(DEVICE)
        human_pose_t = human_pose_t.view(*human_pose_t.shape[:-2], -1)  # flatten last 2 dims
        pos_t        = torch.from_numpy(pos).to(DEVICE)
        shuttle_t    = torch.from_numpy(shuttle).to(DEVICE)
        video_len_t  = torch.from_numpy(video_len).to(DEVICE)

        logits = model(human_pose_t, shuttle_t, pos_t, video_len_t)  # (B, n_class)
        probs = torch.softmax(logits, dim=1).cpu().numpy()           # (B, n_class)
        pred = np.argmax(probs, axis=1)
        # top-k via argpartition for speed; we only need k=2 ordering loosely
        top2_idx = np.argpartition(-probs, kth=1, axis=1)[:, :2]
        true_chunk = np.asarray(y_true[sl])
        in_top2 = np.any(top2_idx == true_chunk[:, None], axis=1)

        y_pred[sl] = pred
        top2_correct[sl] = in_top2

        if (start // BATCH) % 20 == 0:
            log.info("forward: %d / %d", end, n_rows)

    fwd_secs = time.time() - fwd_start
    log.info("forward done in %.1fs (%.1fms/row)", fwd_secs, 1000 * fwd_secs / n_rows)

    y_true_arr = np.asarray(y_true)
    cm = np.zeros((N_CLASS, N_CLASS), dtype=np.int64)
    np.add.at(cm, (y_true_arr, y_pred), 1)
    per_class_f1 = _f1_from_confusion(cm)
    macro_f1 = float(per_class_f1.mean())
    min_f1 = float(per_class_f1.min())
    accuracy = float((y_pred == y_true_arr).mean())
    top2_accuracy = float(top2_correct.mean())

    payload = {
        "split": SPLIT,
        "n_strokes": n_rows,
        "macro_f1": macro_f1,
        "min_f1": min_f1,
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "per_class_f1": {
            cls: float(per_class_f1[i]) for i, cls in enumerate(ACTIVE_CLASS_LIST)
        },
        "source": "compute_val_metrics.py (live forward pass on /app/bst_inputs/val)",
        "weights": str(bi.WEIGHTS_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    log.info(
        "wrote %s — macro_f1=%.4f min_f1=%.4f acc=%.4f top2=%.4f (total %.1fs)",
        OUTPUT_PATH, macro_f1, min_f1, accuracy, top2_accuracy, time.time() - t0,
    )


if __name__ == "__main__":
    main()
