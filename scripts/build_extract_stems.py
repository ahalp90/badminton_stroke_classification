"""Build the stems list for the bourbaki raw-mmpose extract.

Filters clips_master.csv down to the set of stems that need a fresh
raw_extract on bourbaki. Two filter modes via flags:

  - Default (no mode flag): drops raw_type_en == 'unknown' rows plus the
    1,716 hit-zone-busted stems whose raw_phase1 extracts already live on
    engelbart. Standard Phase-2 path.
  - --only-unknown: extracts ONLY the 1,278 unknown clips, as a sibling
    pass for taxonomies that retain unknown (e.g. bst_25, une_v1_15).
    Mutually exclusive with --keep-unknown.

Reads the canonical busted list at:
    scratch/architecture_notes/busted_hit_zone_clips_phase1.txt

Writes one stem per line to --output, ready for raw_extract.py's
--clip-stems-file flag.

Usage (from bourbaki, against the shared /home repo clone):

    # Standard extract (drops unknown + busted)
    /home/ahalperi/.venvs/venv-bst/bin/python \\
        scripts/build_extract_stems.py \\
        --output /scratch/comp320a/ShuttleSet_keypoints_raw/stems_to_extract.txt

    # Sibling extract for the 1,278 unknown clips
    /home/ahalperi/.venvs/venv-bst/bin/python \\
        scripts/build_extract_stems.py --only-unknown --keep-busted \\
        --output /scratch/comp320a/ShuttleSet_keypoints_raw_unknown/stems_unknown.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS_CSV = REPO_ROOT / "notebooks" / "clips_master.csv"
DEFAULT_BUSTED = (
    REPO_ROOT
    / "scratch"
    / "architecture_notes"
    / "busted_hit_zone_clips_phase1.txt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=DEFAULT_CLIPS_CSV,
        help=f"clips_master.csv (default: {DEFAULT_CLIPS_CSV})",
    )
    parser.add_argument(
        "--busted-file",
        type=Path,
        default=DEFAULT_BUSTED,
        help=f"busted_hit_zone_clips_phase1.txt (default: {DEFAULT_BUSTED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the resulting stems list, one per line.",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Don't drop raw_type_en == 'unknown' rows.",
    )
    parser.add_argument(
        "--only-unknown",
        action="store_true",
        help="Include ONLY raw_type_en == 'unknown' rows. "
             "For sibling extracts on taxonomies that retain unknown. "
             "Mutually exclusive with --keep-unknown.",
    )
    parser.add_argument(
        "--keep-busted",
        action="store_true",
        help="Don't exclude the 1,716 already-extracted hit-zone-busted stems.",
    )
    args = parser.parse_args()

    if args.only_unknown and args.keep_unknown:
        parser.error("--only-unknown and --keep-unknown are mutually exclusive")

    if not args.clips_csv.exists():
        parser.error(f"clips-csv not found: {args.clips_csv}")
    if not args.keep_busted and not args.busted_file.exists():
        parser.error(f"busted-file not found: {args.busted_file}")

    clips = pd.read_csv(args.clips_csv)
    if "clip_stem" not in clips.columns:
        parser.error(
            f"clips-csv {args.clips_csv} is missing a clip_stem column"
        )
    if not args.keep_unknown and "raw_type_en" not in clips.columns:
        parser.error(
            f"clips-csv {args.clips_csv} is missing a raw_type_en column "
            "(needed unless --keep-unknown is passed)"
        )

    total = len(clips)
    n_unknown_total = int((clips["raw_type_en"] == "unknown").sum())

    busted: set[str] = set()
    if not args.keep_busted:
        busted = {
            line.strip()
            for line in args.busted_file.read_text().splitlines()
            if line.strip()
        }

    filtered = clips
    if args.only_unknown:
        filtered = filtered[filtered["raw_type_en"] == "unknown"]
    elif not args.keep_unknown:
        filtered = filtered[filtered["raw_type_en"] != "unknown"]
    if not args.keep_busted:
        filtered = filtered[~filtered["clip_stem"].isin(busted)]

    stems = filtered["clip_stem"].astype(str).drop_duplicates().sort_values()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stems.to_csv(args.output, index=False, header=False)

    print(f"Total clips_master rows:        {total}")
    if args.only_unknown:
        print(f"Mode:                           --only-unknown ({n_unknown_total} unknown rows in CSV)")
    elif args.keep_unknown:
        print(f"Unknown rows kept:              {n_unknown_total}")
    else:
        print(f"Unknown rows dropped:           {n_unknown_total}")
    print(f"Busted (already extracted):     {len(busted)}")
    print(f"Stems written:                  {len(stems)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
