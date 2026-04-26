"""Phase 1 mixed-retrain symlink merge.

Builds a flat dir where:
- For each stem in the busted hit-zone list (the 1,716 sticky_anchor-processed
  clips), symlinks ``_pos.npy`` / ``_joints.npy`` / ``_failed.npy`` from the
  sticky_anchor output dir.
- For all other stems in clips_master with ``split_v2`` in (train, val, test),
  symlinks the same three files from the committed extract dir.

The result feeds ``prepare_train_on_shuttleset``'s collation step. After
collation, ``bst_train`` reads the merged dataset under the
``ablation_id``-tagged collated dir name.

Run from the repo root.

Examples:

    # Defaults: BST_MMPOSE_NPY_DIR for the committed dir; sibling dirs for the rest.
    python scripts/symlink_merge_phase1.py

    # Explicit paths if env-var resolution is awkward.
    python scripts/symlink_merge_phase1.py \\
        --committed-dir /scratch/comp320a/.../dataset_npy_..._flat \\
        --sticky-anchor-dir /scratch/comp320a/.../dataset_npy_..._flat_h_sticky_anchor \\
        --merged-dir /scratch/comp320a/.../dataset_npy_..._flat_h_sticky_anchor_phase1_merged
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


THREE_FILES = ("_pos.npy", "_joints.npy", "_failed.npy")
DEFAULT_CLIPS_CSV = Path("notebooks/clips_master.csv")
DEFAULT_BUSTED = Path("scratch/architecture_notes/busted_hit_zone_clips_phase1.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-dir", type=Path, default=None)
    parser.add_argument("--sticky-anchor-dir", type=Path, default=None)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--clips-csv", type=Path, default=DEFAULT_CLIPS_CSV)
    parser.add_argument("--busted-stems-file", type=Path, default=DEFAULT_BUSTED)
    parser.add_argument(
        "--split-column",
        default="split_v2",
        help="Column in clips_master to filter the universe of stems.",
    )
    parser.add_argument(
        "--clip-stem-column",
        default="clip_stem",
        help="Column in clips_master holding the per-clip stem.",
    )
    args = parser.parse_args()

    committed_dir = _resolve_committed_dir(args.committed_dir)
    parent = committed_dir.parent
    base = committed_dir.name
    sticky_dir = args.sticky_anchor_dir or parent / f"{base}_h_sticky_anchor"
    merged_dir = args.merged_dir or parent / f"{base}_h_sticky_anchor_phase1_merged"

    _check_exists(committed_dir, "committed dir")
    _check_exists(sticky_dir, "sticky_anchor dir")
    _check_exists(args.clips_csv, "clips_csv")
    _check_exists(args.busted_stems_file, "busted stems file")

    df = pd.read_csv(args.clips_csv)
    if args.split_column not in df.columns:
        raise SystemExit(f"split column '{args.split_column}' not in clips_csv")
    if args.clip_stem_column not in df.columns:
        raise SystemExit(f"stem column '{args.clip_stem_column}' not in clips_csv")
    in_split = df[args.split_column].isin(["train", "val", "test"])
    universe = sorted(
        {str(s) for s in df.loc[in_split, args.clip_stem_column].astype(str).tolist()}
    )

    with args.busted_stems_file.open() as fh:
        busted = {line.strip() for line in fh if line.strip()}

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"merged_dir: {merged_dir}")
    print(f"committed_dir: {committed_dir}")
    print(f"sticky_anchor_dir: {sticky_dir}")
    print()

    n_busted, n_committed = 0, 0
    missing_busted: list[str] = []
    missing_committed: list[str] = []

    for stem in universe:
        is_busted = stem in busted
        src = sticky_dir if is_busted else committed_dir
        if not all((src / f"{stem}{suf}").exists() for suf in THREE_FILES):
            (missing_busted if is_busted else missing_committed).append(stem)
            continue

        for suf in THREE_FILES:
            link = merged_dir / f"{stem}{suf}"
            target = src / f"{stem}{suf}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)

        if is_busted:
            n_busted += 1
        else:
            n_committed += 1

    total = len(universe)
    busted_in_universe = sum(1 for s in universe if s in busted)
    unbusted_in_universe = total - busted_in_universe

    print(f"Universe ({args.split_column} in train/val/test):  {total}")
    print(f"  busted (sticky_anchor source):     {busted_in_universe}")
    print(f"  unbusted (committed source):       {unbusted_in_universe}")
    print()
    print(f"Linked from sticky_anchor:           {n_busted}")
    print(f"Linked from committed:               {n_committed}")
    if missing_busted:
        print(f"Missing from sticky_anchor (skipped): {len(missing_busted)}")
        print(f"  example: {missing_busted[:5]}")
    if missing_committed:
        print(f"Missing from committed (skipped):    {len(missing_committed)}")
        print(f"  example: {missing_committed[:5]}")
    print()
    print(f"Total symlinks created:              {(n_busted + n_committed) * 3}")


def _resolve_committed_dir(arg_committed: Path | None) -> Path:
    if arg_committed is not None:
        return arg_committed
    env = os.environ.get("BST_MMPOSE_NPY_DIR", "").strip()
    if not env:
        raise SystemExit(
            "BST_MMPOSE_NPY_DIR not set; pass --committed-dir explicitly or "
            "load .env via pipeline.data_access first."
        )
    return Path(env)


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


if __name__ == "__main__":
    main()
