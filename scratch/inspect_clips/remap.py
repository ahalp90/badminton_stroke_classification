"""One-shot remap: point mocked test/val clip_index entries at the local train mp4s.

Same-class matching, no file reuse. Prefers test split before spilling to val."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path("E:/bsc-tier1")
CLIP_INDEX = REPO / "src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/clip_index.json"
PREDS = REPO / "src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/predictions"
LOCAL_ROOT = REPO / "scratch/inspect_clips"

SIDE_RE = re.compile(r"^(Top|Bottom)_")


def class_from_local_dir(name: str) -> str:
    return SIDE_RE.sub("", name)


def inventory_local() -> dict[str, list[str]]:
    """Map class -> list of rel mp4 paths (relative to inspect_clips/)."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for mp4 in sorted(LOCAL_ROOT.rglob("*.mp4")):
        rel = mp4.relative_to(LOCAL_ROOT)
        # parts: e.g. ("train", "Top_smash", "11_1_17_9.mp4")
        if len(rel.parts) < 3:
            continue
        cls = class_from_local_dir(rel.parts[1])
        by_class[cls].append(str(rel).replace("\\", "/"))
    return dict(by_class)


def main():
    with open(CLIP_INDEX) as f:
        data = json.load(f)

    # active class list for sanity check
    with open(PREDS / "test.json") as f:
        test_preds = json.load(f)
    active = set(test_preds.get("active_class_list", []))

    local_by_class = inventory_local()

    print("=== LOCAL FILES BY CLASS ===")
    for cls, files in sorted(local_by_class.items()):
        marker = "" if cls in active else "  (NOT IN ACTIVE LIST)"
        print(f"  {cls}: {len(files)}{marker}")
        for f in files:
            print(f"    {f}")

    # Build entries grouped by split + class. Use raw_type_en as the class key.
    entries_by_split_class: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem, meta in data["clips"].items():
        split = meta.get("split")
        cls = meta.get("raw_type_en")
        entries_by_split_class[(split, cls)].append(stem)

    # Assignment plan: for each class, drain files into test entries first, then val.
    assignments: list[tuple[str, str, str, str]] = []  # (stem, split, class, new_video_path)
    file_pool = {c: list(fs) for c, fs in local_by_class.items()}

    for cls, files in sorted(local_by_class.items()):
        if cls not in active:
            continue
        for split in ("test", "val"):
            stems = entries_by_split_class.get((split, cls), [])
            for stem in stems:
                if not file_pool[cls]:
                    break
                new_path = file_pool[cls].pop(0)
                assignments.append((stem, split, cls, new_path))

    print()
    print(f"=== {len(assignments)} ASSIGNMENTS ===")
    for stem, split, cls, path in assignments:
        old = data["clips"][stem]["video_path"]
        print(f"  [{split:4s}] {stem:14s} class={cls:25s} OLD={old:60s} NEW={path}")
        data["clips"][stem]["video_path"] = path

    # Leftover (unused) files
    leftover = {c: fs for c, fs in file_pool.items() if fs}
    if leftover:
        print()
        print("=== UNUSED LOCAL FILES ===")
        for c, fs in leftover.items():
            for f in fs:
                print(f"  {c}: {f}")

    # Write back
    with open(CLIP_INDEX, "w") as f:
        json.dump(data, f, indent=2)
    print()
    print(f"Wrote {CLIP_INDEX}")

    # Per-split playable summary
    playable_test = []
    playable_val = []
    fallback_test = []
    fallback_val = []
    for stem, meta in data["clips"].items():
        is_playable = meta["video_path"].startswith("train/")
        bucket = (playable_test if is_playable else fallback_test) if meta["split"] == "test" \
                 else (playable_val if is_playable else fallback_val)
        bucket.append((stem, meta["raw_type_en"], meta["video_path"]))

    print()
    print(f"=== TEST: {len(playable_test)} playable, {len(fallback_test)} fallback ===")
    for s, c, p in playable_test:
        print(f"  PLAY     {s:14s} {c:25s} -> {p}")
    for s, c, p in fallback_test:
        print(f"  FALLBACK {s:14s} {c:25s} -> {p}")
    print()
    print(f"=== VAL: {len(playable_val)} playable, {len(fallback_val)} fallback ===")
    for s, c, p in playable_val:
        print(f"  PLAY     {s:14s} {c:25s} -> {p}")
    for s, c, p in fallback_val:
        print(f"  FALLBACK {s:14s} {c:25s} -> {p}")


if __name__ == "__main__":
    main()
