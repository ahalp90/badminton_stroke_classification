"""Rebuild the mocked clip_index.json + predictions/{val,test}.json so the
56 stems point at real entries from clips_master.csv (split_v2-test/val,
drop_unknown). Class-balanced: 2 stems per class × 14 classes per split.

Also computes the row index into the SCP'd collated tensors so the
backend's bst_x_inference.predict() can slice directly without re-deriving
the index.

The predictions JSON is populated with PLACEHOLDER y_pred values (a copy
of y_true) at rebuild time. The /api/registry/{model_id}/splits/{split}/
clips/{stem} endpoint is the one the per-clip browser hits for detail;
that endpoint is being patched to call bst_x_inference.predict() live.
The list endpoint (which displays a summary tile per clip) reads from
this file for fast filtering.

After this script runs, the API has to be restarted to clear the
@lru_cache on _build_stem_index in registry.py."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("E:/bsc-tier1")
CSV = REPO / "notebooks/clips_master.csv"
RUN_DIR = REPO / "src/bst_x/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907"
LOCAL_CLIPS = REPO / "scratch/inspect_clips"  # where the 13 train mp4s live

CLASS_LIST = [
    "net_shot", "return_net", "smash", "wrist_smash",
    "lob", "clear", "drive", "drop",
    "passive_drop", "push", "rush", "cross_court_net_shot",
    "short_service", "long_service",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}
UNE_MERGE_V1_MAP = {
    "defensive_return_lob":   "lob",
    "driven_flight":          "drive",
    "back_court_drive":       "drive",
    "defensive_return_drive": "drive",
}
N_PER_CLASS_PER_SPLIT = 2  # 14 classes × 2 = 28 stems per split, 56 total
SEED = 42


def label_for_row(raw_type_en: str) -> str:
    return UNE_MERGE_V1_MAP.get(raw_type_en, raw_type_en)


def pick_stems(csv: pd.DataFrame, split: str, rng) -> list[tuple[int, str, dict]]:
    """Pick 2 stems per class for the given split. Returns list of
    (row_index_into_collation, clip_stem, {csv fields needed for clip_index})."""
    sub = csv[(csv["split_v2"] == split) & (csv["raw_type_en"] != "unknown")].reset_index(drop=True)
    # row index here == row index in collated tensors (verified earlier).
    sub["_class"] = sub["raw_type_en"].map(label_for_row)
    sub["_row"] = sub.index
    picks: list[tuple[int, str, dict]] = []
    for cls in CLASS_LIST:
        candidates = sub[sub["_class"] == cls]
        if len(candidates) == 0:
            print(f"  [{split}] WARNING: no candidate for class {cls!r} — skipping")
            continue
        chosen = candidates.sample(n=min(N_PER_CLASS_PER_SPLIT, len(candidates)), random_state=rng).reset_index(drop=True)
        for _, row in chosen.iterrows():
            picks.append((int(row["_row"]), str(row["clip_stem"]), row.to_dict()))
    return picks


def find_video_path(stem: str, side_prefix: str, true_class: str,
                    train_files_by_class: dict[str, list[str]],
                    rr_index: dict[str, int]) -> str:
    """Resolve a stem to a playable mp4 under scratch/inspect_clips/.

    Priority: (a) the stem's actual file if SCP'd locally; (b) the 13
    train clips, class-aligned where possible; (c) round-robin fallback.

    For (a), we'd need the actual mp4 — which we don't have for the new
    real stems. So always fall through to (b)/(c) using the same logic
    from §1a/§1b.
    """
    # Try class-aligned remap first
    aligned = train_files_by_class.get(true_class, [])
    if aligned:
        idx = rr_index.get(true_class, 0)
        rr_index[true_class] = (idx + 1) % len(aligned)
        return aligned[idx]
    # Spill to global round-robin across ALL train files
    all_files = sorted(p for lst in train_files_by_class.values() for p in lst)
    if not all_files:
        return f"test/{side_prefix}_{true_class}/{stem}.mp4"  # symbolic fallback
    idx = rr_index.get("_global", 0)
    rr_index["_global"] = (idx + 1) % len(all_files)
    return all_files[idx]


def main():
    rng = np.random.RandomState(SEED)
    csv = pd.read_csv(CSV)

    # Build local train-file inventory keyed by class for (a)
    train_by_class: dict[str, list[str]] = {}
    for mp4 in sorted(LOCAL_CLIPS.rglob("*.mp4")):
        rel = mp4.relative_to(LOCAL_CLIPS)
        if len(rel.parts) < 3:
            continue
        # rel.parts: ('train', 'Top_smash', 'X.mp4')
        side_dir = rel.parts[1]
        # Strip side prefix to get the class
        if "_" in side_dir:
            _, cls = side_dir.split("_", 1)
        else:
            cls = side_dir
        train_by_class.setdefault(cls, []).append(str(rel).replace("\\", "/"))

    print(f"Local train mp4s available by class:")
    for cls, files in sorted(train_by_class.items()):
        print(f"  {cls}: {len(files)}")
    print()

    new_clips: dict[str, dict] = {}
    new_predictions: dict[str, dict] = {}
    rr_index: dict[str, int] = {}

    for split in ("test", "val"):
        picks = pick_stems(csv, split, rng)
        print(f"=== {split.upper()}: {len(picks)} stems picked ===")

        # Sort by class then by row for stable display
        picks_sorted = sorted(picks, key=lambda x: (label_for_row(x[2]["raw_type_en"]), x[0]))

        clips_list = []
        for row_idx, stem, meta in picks_sorted:
            true_cls = label_for_row(meta["raw_type_en"])
            true_class_idx = CLASS_TO_IDX[true_cls]
            side_prefix = meta["player_side"]  # 'Top' / 'Bottom'
            video_path = find_video_path(stem, side_prefix, true_cls, train_by_class, rr_index)

            new_clips[stem] = {
                "video_path": video_path,
                "match":      meta["match"],
                "set_id":     meta["set_id"],
                "rally":      int(meta["rally"]),
                "ball_round": int(meta["ball_round"]),
                "split":      split,
                "raw_type_en": meta["raw_type_en"],
                "player_side": side_prefix,
                "row_index":  row_idx,  # NEW: lets bst_x_inference slice without re-deriving
            }

            # Placeholder predictions — real values come from bst_x_inference at request time.
            # y_pred starts as y_true; top_k is a one-hot at y_true with confidence 1.0.
            clips_list.append({
                "clip_stem": stem,
                "y_true":    true_class_idx,
                "y_pred":    true_class_idx,   # placeholder; real inference overrides
                "softmax_calibrated": [1.0 if i == true_class_idx else 0.0 for i in range(len(CLASS_LIST))],
                "top_k_idx":  [true_class_idx],
                "top_k_prob": [1.0],
            })

        new_predictions[split] = {
            "_mock_data": False,
            "_real_stems": True,
            "_notes": "stems sampled from clips_master.csv split_v2 + drop_unknown; "
                      "y_pred is a placeholder that the live BST inference endpoint overrides.",
            "run_id": "run_20260505_154907",
            "serial_no": 5,
            "split": split,
            "active_class_list": CLASS_LIST,
            "temperature": 1.0,
            "clips": clips_list,
        }

    # Wrap clip_index
    new_clip_index = {
        "_mock_data": False,
        "_real_stems": True,
        "_notes": "56 real stems (28 test + 28 val) sampled from clips_master.csv, "
                  "class-balanced. row_index is the row in the collated tensors at "
                  "scratch/bst_x_inputs/{split}/JnB_bone.npy etc.",
        "clips": new_clips,
    }

    # Backup the existing mocked files
    ci_path = RUN_DIR / "clip_index.json"
    pred_dir = RUN_DIR / "predictions"
    backup_root = REPO / "scratch/inspect_clips/mock_backup"
    backup_root.mkdir(parents=True, exist_ok=True)
    if ci_path.exists():
        shutil.copy2(ci_path, backup_root / "clip_index.mock.json")
    for s in ("test", "val"):
        src = pred_dir / f"{s}.json"
        if src.exists():
            shutil.copy2(src, backup_root / f"{s}.mock.json")

    # Write new
    with open(ci_path, "w") as f:
        json.dump(new_clip_index, f, indent=2)
    for s, payload in new_predictions.items():
        with open(pred_dir / f"{s}.json", "w") as f:
            json.dump(payload, f, indent=2)

    # Summary
    print()
    print(f"=== WROTE ===")
    print(f"  {ci_path}  ({len(new_clips)} stems)")
    for s in ("test", "val"):
        n = len(new_predictions[s]["clips"])
        print(f"  {pred_dir / (s + '.json')}  ({n} stems)")
    print(f"\nBackups of original mocks: {backup_root}")

    print()
    print("=== VIDEO PATH SOURCE (which clips are content-aligned vs round-robin) ===")
    aligned_count = 0
    rr_count = 0
    for stem, meta in new_clips.items():
        vp = meta["video_path"]
        # If video_path starts with 'train/<Side>_<class>/' and class matches raw_type_en's
        # merged class, it's content-aligned
        if vp.startswith("train/"):
            parts = vp.split("/")
            if len(parts) >= 3:
                _, side_cls, _ = parts[0], parts[1], parts[2]
                if "_" in side_cls:
                    file_class = side_cls.split("_", 1)[1]
                    if file_class == label_for_row(meta["raw_type_en"]):
                        aligned_count += 1
                    else:
                        rr_count += 1
        else:
            rr_count += 1
    print(f"  content-aligned:  {aligned_count}/56")
    print(f"  round-robin fill: {rr_count}/56")


if __name__ == "__main__":
    main()
