"""Build the Tier 1 FE sidecar JSONs for a registered run, from its prediction npz.

Reads the self-contained per-stroke npz that ``bst_x_train`` / ``bst_x_infer --fe`` dump
for the chosen serial (under ``<run_dir>/predictions/``) and writes the five files the
API serves, all under ``<run_dir>/fe_jsons/``:

    fe_jsons/{val,test}.json.gz                  per-clip preds + raw softmax
    fe_jsons/perclass_stats_{val,test}.json.gz   confusion-matrix views
    fe_jsons/clip_index.json.gz                  stem -> clip metadata + mp4 path

Confidence is raw softmax; temperature scaling dropped per ECE.

As elsewhere, CLI call by:
    PYTHONPATH = ``src/bst_x``
    python -m build_fe_stats_jsons \
        --run-dir .../experiments/bst_x/shuttleset/run__<datetime>--serial <n>

``predictions`` and ``perclass_stats`` are pure npz reads; ``clip_index`` needs
the clips tree mounted (engelbart / bourbaki) to fill ``video_path`` (null otherwise).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.clip_index import build_clip_path_index
from pipeline.data_access import DataPaths, load_repo_dotenv


def write_json_gz(path: Path, obj: dict) -> None:
    """Write ``obj`` as gzipped JSON to ``path`` (should end in ``.json.gz``)."""
    # ``mtime=0`` zeroes the gzip header's timestamps; unchanged recompresses are byte-identical.
    path.write_bytes(gzip.compress(json.dumps(obj, indent=2).encode(), mtime=0))


def softmax(logits: np.ndarray) -> np.ndarray:
    # exp(large logit) -> inf; shift so max is 0, bounding exp output to (0, 1].
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def top5(weights: np.ndarray, total: int, class_list: list[str]) -> list[list]:
    """Top-5 ``[class, share]`` of one normalised confusion row (or column)."""
    if total == 0:
        return []
    share = weights / total
    order = np.argsort(-share)[:5]                   # top-5 indices by share, descending
    result = []
    for class_idx in order:
        class_share = share[class_idx]
        if class_share <= 0:
            continue
        result.append([class_list[class_idx], round(float(class_share), 2)])
    return result


def build_predictions_json(dump: np.lib.npyio.NpzFile, split: str) -> dict:
    """Per-clip predictions for one split: raw softmax + top-k."""
    logits = dump["logits"]                          # (n, n_classes) float32
    y_true = dump["y_true"]
    y_pred = dump["y_pred_top1"]
    topk_idx = dump["topk_idx"]                      # (n, k), desc by logit
    stems = dump["clip_stems"]
    class_list = [str(c) for c in dump["class_list"]]

    probs = softmax(logits)
    rows = np.arange(len(y_true))[:, None]
    topk_prob = probs[rows, topk_idx]                # (n, k), aligned to topk_idx

    clips = []
    for clip in range(len(stems)):
        clips.append({
            "clip_stem": str(stems[clip]),
            "y_true": int(y_true[clip]),
            "y_pred": int(y_pred[clip]),
            "softmax": [round(float(p), 4) for p in probs[clip]],
            "top_k_idx": [int(j) for j in topk_idx[clip]],
            "top_k_prob": [round(float(p), 4) for p in topk_prob[clip]],
        })

    return {
        "run_id": str(dump["run_id"].item()),
        "serial_no": int(dump["serial_no"].item()),
        "split": split,
        "class_list": class_list,
        "clips": clips,
    }


def build_perclass_stats(dump: np.lib.npyio.NpzFile, split: str) -> dict:
    """Per-class confusion stats for one split, from ``y_true`` / ``y_pred``."""
    y_true = dump["y_true"]
    y_pred = dump["y_pred_top1"]
    class_list = [str(c) for c in dump["class_list"]]
    n = len(class_list)

    conf = np.zeros((n, n), dtype=np.int64)
    # Each conf[t, p] cell counts clips where y_true=t AND y_pred=p.
    np.add.at(conf, (y_true, y_pred), 1)

    support_true = conf.sum(axis=1)                  # row sums: per true class
    support_pred = conf.sum(axis=0)                  # col sums: per predicted class
    tp = np.diag(conf)                               # diag of confusion matrix is per-class TP
    # Safe vectorised divide: result is 0 where the denom is 0, not NaN/inf.
    precision = np.divide(tp, support_pred, out=np.zeros(n), where=support_pred > 0)
    recall = np.divide(tp, support_true, out=np.zeros(n), where=support_true > 0)
    pr = precision + recall
    f1 = np.divide(2 * precision * recall, pr, out=np.zeros(n), where=pr > 0)

    per_class = {}
    for cls_idx, name in enumerate(class_list):
        per_class[name] = {
            "support_true": int(support_true[cls_idx]),
            "support_pred": int(support_pred[cls_idx]),
            "precision": round(float(precision[cls_idx]), 4),
            "recall": round(float(recall[cls_idx]), 4),
            "f1": round(float(f1[cls_idx]), 4),
            # row c: real-c clips, what got predicted (recall view).
            "top5_when_true": top5(conf[cls_idx, :], int(support_true[cls_idx]), class_list),
            # col c: predicted-c clips, what was actually true (precision view).
            "top5_when_pred": top5(conf[:, cls_idx], int(support_pred[cls_idx]), class_list),
        }

    return {
        "split": split,
        "n_clips": len(y_true),
        "class_list": class_list,
        "per_class": per_class,
    }


def build_clip_index(
    stems_by_split: dict[str, list[str]],
    clips_csv: Path,
    clips_dir: Path,
) -> dict:
    """Stem -> clip metadata + relative mp4 path, across all built splits.

    Metadata (match / set / rally / ball_round / raw type / side) is a direct
    projection of ``clips_master.csv``. ``video_path`` is the clip's path relative to
    the clips dir, found via ``pipeline.clip_index.build_clip_path_index``; the serving
    side joins it back onto its own ``BST_X_CLIPS_DIR``. With no clips tree present,
    ``video_path`` is null and the rest of the entry still populates.
    """
    master = pd.read_csv(clips_csv, dtype={"clip_stem": str}).set_index("clip_stem")
    rows = master.to_dict("index")                   # {stem: {col: value}}, O(1) lookup

    path_by_stem = build_clip_path_index(clips_dir) if clips_dir.is_dir() else {}
    if not path_by_stem:
        print(f"WARNING: no mp4s under {clips_dir}; clip_index video_path will be null "
              f"(run where BST_X_CLIPS_DIR is mounted to populate it).")

    index = {}
    for split, stems in stems_by_split.items():
        for stem in stems:
            row = rows[stem]                         # KeyError if stem not in master: fail loud
            mp4 = path_by_stem.get(stem)
            index[stem] = {
                "video_path": str(mp4.relative_to(clips_dir)) if mp4 else None,
                "match": str(row["match"]),
                "set_id": str(row["set_id"]),
                "rally": int(row["rally"]),
                "ball_round": int(row["ball_round"]),
                "split": split,
                "raw_type_en": str(row["raw_type_en"]),
                "player_side": str(row["player_side"]),
            }
    return {"clips": index}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Tier 1 FE sidecar JSONs (fe_jsons/) from a run's prediction npz."
    )
    ap.add_argument("--run-dir", type=Path, required=True, help="experiments/bst_x/shuttleset/<run_id> dir")
    ap.add_argument("--serial", type=int, required=True, help="serial whose npz to read (1-5)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--clips-csv", type=Path, default=None, help="override DataPaths clips_csv")
    ap.add_argument("--clips-dir", type=Path, default=None, help="override DataPaths clips_dir")
    args = ap.parse_args()

    # .env first so BST_X_CLIPS_DIR / BST_X_CLIPS_CSV resolve the same way bst_x_infer +
    # the collator do; DataPaths then picks them up (or the in-repo defaults).
    load_repo_dotenv()
    path_kwargs = {}
    if args.clips_csv is not None:
        path_kwargs["clips_csv"] = args.clips_csv
    if args.clips_dir is not None:
        path_kwargs["clips_dir"] = args.clips_dir
    paths = DataPaths(**path_kwargs)

    pred_dir = args.run_dir / "predictions"          # raw npz dumps live here
    fe_dir = args.run_dir / "fe_jsons"               # derived FE artefacts go here
    fe_dir.mkdir(parents=True, exist_ok=True)

    stems_by_split: dict[str, list[str]] = {}
    for split in args.splits:
        # Load the npz once; both pure-npz builders read from the same handle.
        dump = np.load(pred_dir / f"{split}_serial_{args.serial}.npz", allow_pickle=True)

        preds = build_predictions_json(dump, split)
        preds_path = fe_dir / f"{split}.json.gz"
        write_json_gz(preds_path, preds)
        print(f"wrote {preds_path} ({len(preds['clips'])} clips)")

        stats = build_perclass_stats(dump, split)
        stats_path = fe_dir / f"perclass_stats_{split}.json.gz"
        write_json_gz(stats_path, stats)
        print(f"wrote {stats_path}")

        stems_by_split[split] = [str(s) for s in dump["clip_stems"]]

    clip_index = build_clip_index(stems_by_split, paths.clips_csv, paths.clips_dir)
    clip_index_path = fe_dir / "clip_index.json.gz"
    write_json_gz(clip_index_path, clip_index)
    print(f"wrote {clip_index_path} ({len(clip_index['clips'])} entries)")


if __name__ == "__main__":
    main()
