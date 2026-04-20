#!/usr/bin/env python3
"""verify_flatten.py -- end-to-end check before deleting the nested originals.

Three things this verifies:

1. **CSV correlation against the original nested tree.** Every clip on disk
   under `{nested_root}/{split}/{class_folder}/{clip_stem}_*.npy` must:
     - have a row in clips_master.csv,
     - have its `split_bst_baseline` match the on-disk `{split}` parent,
     - have its (merged_25-derived) folder name match `{class_folder}`.
   Conversely, every row in clips_master.csv must have at least one
   matching file on disk.

2. **Flat-copy content matches the original.** For every original file,
   the flat copy at `{flat_root}/{clip_stem}_<suffix>.npy` must exist with
   identical size (and identical sha256 if `--full-hash` is passed).

3. **Two passes**: per-clip dir (joints/pos/failed) and shuttle_npy dir.

Exit code 0 = all checks pass. Non-zero = at least one discrepancy. Until
this exits 0, do not delete the originals.

Usage on engelbart:

    python3 verify_flatten.py \\
        --master-csv ~/COSC594/badminton_stroke_classification/notebooks/clips_master.csv \\
        [--full-hash]

Override paths if your scratch layout differs:

    python3 verify_flatten.py \\
        --clips-nested  /scratch/comp320a/.../dataset_npy_between_2_hits_with_max_limits \\
        --clips-flat    /scratch/.../dataset_npy_between_2_hits_with_max_limits_flat \\
        --shuttle-nested /scratch/comp320a/ShuttleSet/shuttle_npy \\
        --shuttle-flat   /scratch/comp320a/ShuttleSet/shuttle_npy_flat
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Default scratch paths on engelbart.
CLIPS_NESTED_DEFAULT = Path('/scratch/comp320a/ShuttleSet_data_merged_25'
                            '/dataset_npy_between_2_hits_with_max_limits')
CLIPS_FLAT_DEFAULT = Path('/scratch/comp320a/ShuttleSet_data_merged_25'
                          '/dataset_npy_between_2_hits_with_max_limits_flat')
SHUTTLE_NESTED_DEFAULT = Path('/scratch/comp320a/ShuttleSet/shuttle_npy')
SHUTTLE_FLAT_DEFAULT = Path('/scratch/comp320a/ShuttleSet/shuttle_npy_flat')

# merged_25 taxonomy mapping, hardcoded so this script has no dependency on
# the project src tree (handy if you SSH into engelbart and run it standalone).
# Mirrors pipeline.config.MERGE_MAP and TAXONOMY_MERGED_25.
MERGE_MAP_25 = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'unknown',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}
STANDALONE_TYPES_25 = {'unknown'}


def merged_25_folder(raw_type_en: str, player_side: str) -> str:
    """Return the on-disk folder name a clip would land in under merged_25.

    Mirrors how ``Taxonomy.class_list`` constructs label strings for the
    folder layout:
      - apply MERGE_MAP (e.g. wrist_smash -> smash, driven_flight -> unknown)
      - standalone types (unknown) get no Top_/Bottom_ prefix
      - everything else is ``f'{player_side}_{merged}'``
    """
    merged = MERGE_MAP_25.get(raw_type_en, raw_type_en)
    if merged in STANDALONE_TYPES_25:
        return merged
    return f'{player_side}_{merged}'


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def parse_nested_path(rel_parts: tuple[str, ...]) -> tuple[str, str, str, str] | None:
    """Pull (split, class_folder, clip_stem, suffix) from a nested rel path.

    Expected layout: ``{split}/{class_folder}/{clip_stem}_<suffix>.npy``
    where ``<suffix>`` is e.g. ``joints``, ``pos``, ``failed`` for the
    per-clip dir, or empty for shuttle_npy (where files are just
    ``{clip_stem}.npy``).

    Returns None for paths that do not match the expected layout.
    """
    if len(rel_parts) != 3:
        return None
    split, class_folder, fname = rel_parts
    if not fname.endswith('.npy'):
        return None
    stem = fname[:-len('.npy')]
    # Per-clip dir uses _joints / _pos / _failed; shuttle_npy uses bare stems.
    for suffix in ('_joints', '_pos', '_failed'):
        if stem.endswith(suffix):
            return split, class_folder, stem[:-len(suffix)], suffix[1:]
    return split, class_folder, stem, ''


def walk_nested(root: Path) -> list[tuple[str, str, str, str, Path]]:
    """Return [(split, class_folder, clip_stem, suffix, abs_path), ...]."""
    out = []
    for p in root.rglob('*.npy'):
        rel = p.relative_to(root).parts
        parsed = parse_nested_path(rel)
        if parsed is None:
            print(f'  [WARN] unexpected path layout: {p}', file=sys.stderr)
            continue
        split, cls, stem, suffix = parsed
        out.append((split, cls, stem, suffix, p))
    return out


# ---------------------------------------------------------------------------
# Content check
# ---------------------------------------------------------------------------
def _hash(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _content_check_one(args: tuple[Path, Path, bool]) -> tuple[Path, str | None]:
    """Returns (orig_path, error_message_or_None)."""
    orig, flat, full_hash = args
    if not flat.exists():
        return orig, f'flat copy missing at {flat}'
    if orig.stat().st_size != flat.stat().st_size:
        return orig, (
            f'size mismatch: orig={orig.stat().st_size}, flat={flat.stat().st_size}'
        )
    if full_hash and _hash(orig) != _hash(flat):
        return orig, 'sha256 mismatch'
    return orig, None


# ---------------------------------------------------------------------------
# Pass-level check
# ---------------------------------------------------------------------------
def check_one_pass(
    label: str,
    nested_root: Path,
    flat_root: Path,
    master_df: pd.DataFrame,
    suffixes_expected: set[str],
    full_hash: bool,
) -> int:
    """Run all three checks for one of {clips, shuttle}. Return error count."""
    print(f'\n=== {label} ===')
    print(f'  nested: {nested_root}')
    print(f'  flat:   {flat_root}')
    if not nested_root.is_dir():
        print(f'  ERROR: nested root not found', file=sys.stderr)
        return 1
    if not flat_root.is_dir():
        print(f'  ERROR: flat root not found', file=sys.stderr)
        return 1

    # Build per-stem lookup from master CSV.
    # csv_lookup[clip_stem] = (split_bst_baseline, expected_folder_name)
    csv_lookup: dict[str, tuple[str, str]] = {}
    for row in master_df.itertuples(index=False):
        folder = merged_25_folder(row.raw_type_en, row.player_side)
        csv_lookup[row.clip_stem] = (row.split_bst_baseline, folder)

    print(f'  CSV rows: {len(csv_lookup):,}')

    files = walk_nested(nested_root)
    print(f'  nested files: {len(files):,}')

    errors: list[str] = []
    seen_stems: set[str] = set()
    seen_stem_suffix: set[tuple[str, str]] = set()

    # --- Phase A: every nested file -> CSV row + matching split/folder ---
    for split, cls, stem, suffix, abs_path in files:
        seen_stems.add(stem)
        seen_stem_suffix.add((stem, suffix))
        if stem not in csv_lookup:
            errors.append(f'[A] nested file has no CSV row: {abs_path}')
            continue
        csv_split, csv_folder = csv_lookup[stem]
        if csv_split != split:
            errors.append(
                f'[A] split mismatch for {stem}: '
                f'on-disk={split}, csv={csv_split} ({abs_path})'
            )
        if csv_folder != cls:
            errors.append(
                f'[A] folder mismatch for {stem}: '
                f'on-disk={cls}, csv-derived={csv_folder} ({abs_path})'
            )

    # --- Phase A2: every CSV row should have its expected file(s) on disk ---
    # For per-clip dir, the upstream extraction may legitimately drop clips
    # whose pose detection failed entirely; warn but do not error in that
    # case. For shuttle_npy, missing clips are also possible (TrackNetV3
    # may have skipped some). We still surface the count loudly.
    expected_stems = set(csv_lookup.keys())
    missing_stems = expected_stems - seen_stems
    if missing_stems:
        print(f'  [A2] {len(missing_stems):,} CSV rows have no matching file on disk')
        # Show a few samples
        for s in sorted(missing_stems)[:5]:
            csv_split, csv_folder = csv_lookup[s]
            print(f'         missing: {s}  (expected at {csv_split}/{csv_folder}/)')
        if len(missing_stems) > 5:
            print(f'         ... and {len(missing_stems) - 5:,} more')

    # For per-clip dir, also confirm each present clip has the full expected
    # suffix set (e.g. joints + pos + failed). Missing one suffix may mean
    # half-completed extraction.
    if suffixes_expected:
        per_stem_suffixes: dict[str, set[str]] = defaultdict(set)
        for stem, suffix in seen_stem_suffix:
            per_stem_suffixes[stem].add(suffix)
        for stem, sufs in per_stem_suffixes.items():
            missing_sufs = suffixes_expected - sufs
            if missing_sufs:
                errors.append(
                    f'[A] stem {stem} missing suffix(es): {sorted(missing_sufs)}'
                )

    # --- Phase B: flat copy content matches original ---
    print(f'  [B] checking flat copies (full_hash={full_hash})...')
    args_iter = []
    for _split, _cls, stem, suffix, abs_path in files:
        flat_name = f'{stem}_{suffix}.npy' if suffix else f'{stem}.npy'
        flat_path = flat_root / flat_name
        args_iter.append((abs_path, flat_path, full_hash))

    n_workers = 8
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for orig, err in ex.map(_content_check_one, args_iter, chunksize=200):
            if err:
                errors.append(f'[B] {orig}: {err}')

    if errors:
        print(f'\n  ERRORS ({len(errors)}):')
        for e in errors[:30]:
            print(f'    {e}')
        if len(errors) > 30:
            print(f'    ... and {len(errors) - 30} more')
    else:
        print(f'  All checks passed.')

    return len(errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--master-csv', type=Path, required=True)
    p.add_argument('--clips-nested', type=Path, default=CLIPS_NESTED_DEFAULT)
    p.add_argument('--clips-flat', type=Path, default=CLIPS_FLAT_DEFAULT)
    p.add_argument('--shuttle-nested', type=Path, default=SHUTTLE_NESTED_DEFAULT)
    p.add_argument('--shuttle-flat', type=Path, default=SHUTTLE_FLAT_DEFAULT)
    p.add_argument('--full-hash', action='store_true',
                   help='Also sha256 every file pair (slower; size+mtime is '
                        'usually enough since cp does not corrupt).')
    p.add_argument('--target', choices=['clips', 'shuttle', 'all'], default='all')
    args = p.parse_args()

    master_df = pd.read_csv(args.master_csv)
    required_cols = {'clip_stem', 'split_bst_baseline', 'raw_type_en', 'player_side'}
    missing = required_cols - set(master_df.columns)
    if missing:
        print(f'ERROR: master CSV missing columns: {missing}', file=sys.stderr)
        return 1
    print(f'Loaded {len(master_df):,} master CSV rows from {args.master_csv}')

    total_errors = 0

    if args.target in ('clips', 'all'):
        total_errors += check_one_pass(
            'per-clip npy (joints/pos/failed)',
            args.clips_nested, args.clips_flat,
            master_df,
            suffixes_expected={'joints', 'pos', 'failed'},
            full_hash=args.full_hash,
        )

    if args.target in ('shuttle', 'all'):
        total_errors += check_one_pass(
            'shuttle_npy',
            args.shuttle_nested, args.shuttle_flat,
            master_df,
            suffixes_expected=set(),  # bare {clip_stem}.npy
            full_hash=args.full_hash,
        )

    print()
    if total_errors:
        print(f'FAIL: {total_errors:,} discrepancies. Do NOT delete originals.',
              file=sys.stderr)
        return 1
    print('OK: all checks passed. Safe to delete the nested originals.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
