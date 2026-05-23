"""Round-robin fill: assign one of the 13 local train mp4s to every mocked
clip_index entry whose video_path still points at a test/ or val/ file that
isn't on disk locally. Preserves the 10 class-aligned remaps from remap.py.

Content will NOT match the displayed class for the 46 filled entries —
that's the explicit honesty-tax of getting every clip to play."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path("E:/bsc-tier1")
CLIP_INDEX = REPO / "src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907/clip_index.json"
LOCAL_ROOT = REPO / "scratch/inspect_clips"


def main():
    with open(CLIP_INDEX) as f:
        data = json.load(f)

    pool = sorted(
        str(p.relative_to(LOCAL_ROOT)).replace("\\", "/")
        for p in LOCAL_ROOT.rglob("*.mp4")
    )
    assert len(pool) == 13, f"expected 13 local mp4s, got {len(pool)}"
    print(f"=== POOL ({len(pool)} files, cycled in this order) ===")
    for i, f in enumerate(pool):
        print(f"  [{i:2d}] {f}")

    # Identify entries still pointing at a non-local path. We keep anything
    # already pointing under train/ (the 10 class-aligned remaps from
    # remap.py) untouched.
    ordered_stems: list[tuple[str, str]] = []  # (split, stem)
    for split in ("test", "val"):
        for stem, meta in data["clips"].items():
            if meta.get("split") != split:
                continue
            if not meta["video_path"].startswith("train/"):
                ordered_stems.append((split, stem))

    print()
    print(f"=== {len(ordered_stems)} ENTRIES TO FILL (test first, then val) ===")

    assignments: list[tuple[str, str, str, str, str]] = []
    for i, (split, stem) in enumerate(ordered_stems):
        new_path = pool[i % len(pool)]
        old = data["clips"][stem]["video_path"]
        true_cls = data["clips"][stem]["raw_type_en"]
        data["clips"][stem]["video_path"] = new_path
        assignments.append((split, stem, true_cls, old, new_path))
        print(f"  [{i:2d}] {split:4s} {stem:14s} class={true_cls:25s} OLD={old:60s} NEW={new_path}")

    with open(CLIP_INDEX, "w") as f:
        json.dump(data, f, indent=2)
    print()
    print(f"Wrote {CLIP_INDEX}")

    # Final sanity: every video_path should now resolve to a real file.
    missing = []
    for stem, meta in data["clips"].items():
        abs_path = LOCAL_ROOT / meta["video_path"]
        if not abs_path.exists():
            missing.append((stem, meta["video_path"]))
    if missing:
        print()
        print(f"!!! {len(missing)} ENTRIES STILL UNRESOLVED:")
        for s, p in missing:
            print(f"   {s}: {p}")
    else:
        print(f"\n✓ all {len(data['clips'])} clip entries resolve to an on-disk mp4")

    # Per-split summary
    for split in ("test", "val"):
        sp = [m for m in data["clips"].values() if m["split"] == split]
        playable = sum(1 for m in sp if (LOCAL_ROOT / m["video_path"]).exists())
        print(f"  {split}: {playable}/{len(sp)} playable")


if __name__ == "__main__":
    main()
