"""Post-generation sanity checks for the ShuttleSet clip pipeline.

Verifies that generated clips match expectations: correct counts, no
excluded videos, no removed shots, and class merge applied correctly.

Usage:
    python -m pipeline.verify [--clips-dir DIR]
"""
import argparse
import re
import sys
from pathlib import Path

from pipeline.config import (
    CLIPS_OUTPUT_DIR, SHUTTLE_OUTPUT_DIR, EXCLUDED_VIDEOS, REMOVED_SHOTS,
    MERGE_MAP, PLAYERS, SPLITS,  # noqa: F401
    UNPREFIXED_TYPES, TAXONOMY_UNE_MERGE_V1, Taxonomy,
)


def _parse_clip_filename(filename: str) -> tuple[int, int, int, int] | None:
    """Extract (video_id, set, rally, ball_round) from a clip filename.

    Expected format: '{video_id}_{set}_{rally}_{ball_round}.mp4'

    :param filename: Clip filename string.
    :return: Tuple of (video_id, set, rally, ball_round) or None if not parseable.
    """
    match = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)\.mp4$', filename)
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def _scan_clips(clips_dir: Path) -> list[Path]:
    """Scan clips directory once and return all .mp4 paths.

    :param clips_dir: Root clips directory.
    :return: Sorted list of .mp4 file paths.
    """
    return sorted(clips_dir.rglob('*.mp4'))


def verify_no_excluded(
    clip_paths: list[Path],
) -> bool:
    """Check that no clips from excluded videos exist.

    :param clip_paths: Pre-scanned list of .mp4 paths.
    :return: True if no excluded clips found.
    """
    violations = []
    for mp4 in clip_paths:
        parsed = _parse_clip_filename(mp4.name)
        if parsed and parsed[0] in EXCLUDED_VIDEOS:
            violations.append(mp4)

    if violations:
        print(f'FAIL: {len(violations)} clips from excluded videos found:')
        for v in violations[:10]:
            print(f'  {v}')
        return False
    print('PASS: No clips from excluded videos.')
    return True


def verify_no_removed_shots(
    clip_paths: list[Path],
) -> bool:
    """Check that individually removed shots are not present.

    :param clip_paths: Pre-scanned list of .mp4 paths.
    :return: True if no removed shots found.
    """
    violations = []
    for mp4 in clip_paths:
        parsed = _parse_clip_filename(mp4.name)
        if parsed and parsed in REMOVED_SHOTS:
            violations.append(mp4)

    if violations:
        print(f'FAIL: {len(violations)} removed shots found:')
        for v in violations:
            print(f'  {v}')
        return False
    print('PASS: No removed shots found.')
    return True


def verify_class_merge(
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    taxonomy: Taxonomy = TAXONOMY_UNE_MERGE_V1,
) -> bool:
    """Check that source folders for merged classes no longer exist.

    :param clips_dir: Root clips directory to scan.
    :param taxonomy: Taxonomy whose merge_map defines expected merges.
    :return: True if all source folders are empty or absent.
    """
    if taxonomy.merge_map is None:
        print('Taxonomy has no merge_map — nothing to verify.')
        return True
    violations = []
    for split_dir in clips_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for src_type in taxonomy.merge_map:
            if src_type in UNPREFIXED_TYPES:
                src = split_dir / src_type
                if src.exists() and any(src.glob('*.mp4')):
                    violations.append(src)
            else:
                for player in PLAYERS:
                    src = split_dir / f'{player}_{src_type}'
                    if src.exists() and any(src.glob('*.mp4')):
                        violations.append(src)

    if violations:
        print(f'FAIL: {len(violations)} unmerged source folders still contain clips:')
        for v in violations:
            n = len(list(v.iterdir()))
            print(f'  {v} ({n} files)')
        return False
    print('PASS: All source type folders merged correctly.')
    return True


def verify_splits_present(
    clips_dir: Path,
    clip_paths: list[Path],
) -> bool:
    """Check that train/val/test split directories exist and have clips.

    :param clips_dir: Root clips directory.
    :param clip_paths: Pre-scanned list of .mp4 paths.
    :return: True if all splits exist and contain clips.
    """
    # Group clip counts by split
    split_counts: dict[str, int] = {}
    for mp4 in clip_paths:
        try:
            split_name = mp4.relative_to(clips_dir).parts[0]
        except (ValueError, IndexError):
            continue
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    ok = True
    for split_name in SPLITS:
        n = split_counts.get(split_name, 0)
        if n == 0:
            split_dir = clips_dir / split_name
            if not split_dir.is_dir():
                print(f'FAIL: Missing split directory: {split_dir}')
            else:
                print(f'FAIL: Split {split_name} has no clips.')
            ok = False
        else:
            print(f'PASS: {split_name} has {n} clips.')
    return ok


def warn_orphan_files(
    clips_dir: Path,
    clip_paths: list[Path],
) -> None:
    """Warn about files that don't match the expected clip naming pattern.

    Not a FAIL -- legitimate non-clip files may exist. But it helps
    catch accidental file drops or half-deleted folders.

    :param clips_dir: Root clips directory.
    :param clip_paths: Pre-scanned list of .mp4 paths.
    """
    orphans = [mp4 for mp4 in clip_paths if _parse_clip_filename(mp4.name) is None]
    if orphans:
        print(f'WARNING: {len(orphans)} .mp4 files don\'t match expected '
              f'naming pattern ({{vid}}_{{set}}_{{rally}}_{{ball_round}}.mp4):')
        for o in orphans[:10]:
            print(f'  {o.relative_to(clips_dir)}')
        if len(orphans) > 10:
            print(f'  ... and {len(orphans) - 10} more')


def verify_file_integrity(clip_paths: list[Path]) -> bool:
    """Check for 0-byte or corrupted clips.

    :param clip_paths: Pre-scanned list of .mp4 paths.
    :return: True if all files have data.
    """
    empty_files = [f for f in clip_paths if f.stat().st_size == 0]

    if empty_files:
        print(f'FAIL: {len(empty_files)} empty (0-byte) .mp4 files found:')
        for f in empty_files[:10]:
            print(f'  {f}')
        return False
    print('PASS: All clips have data (no 0-byte files).')
    return True

def verify_shuttle_sync(
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    shuttle_dir: Path = SHUTTLE_OUTPUT_DIR,
) -> bool:
    """Check that every clip has a corresponding shuttle .npy file.

    Only runs if shuttle_dir exists (step 6 may not have been run yet).

    :param clips_dir: Root clips directory.
    :param shuttle_dir: Root shuttle .npy directory.
    :return: True if all clips have matching .npy files, or if shuttle_dir doesn't exist.
    """
    if not shuttle_dir.is_dir():
        print('SKIP: Shuttle directory not found (step 6 may not have run).')
        return True

    clip_stems = set()
    for mp4 in clips_dir.rglob('*.mp4'):
        rel = mp4.relative_to(clips_dir).with_suffix('.npy')
        clip_stems.add(rel)

    npy_stems = set()
    for npy in shuttle_dir.rglob('*.npy'):
        rel = npy.relative_to(shuttle_dir)
        npy_stems.add(rel)

    missing_npy = clip_stems - npy_stems
    orphan_npy = npy_stems - clip_stems

    ok = True
    if missing_npy:
        print(f'FAIL: {len(missing_npy)} clips have no matching shuttle .npy:')
        for m in sorted(missing_npy)[:10]:
            print(f'  {m}')
        if len(missing_npy) > 10:
            print(f'  ... and {len(missing_npy) - 10} more')
        ok = False

    if orphan_npy:
        print(f'WARNING: {len(orphan_npy)} shuttle .npy files have no matching clip:')
        for o in sorted(orphan_npy)[:10]:
            print(f'  {o}')

    if ok and not orphan_npy:
        print(f'PASS: All {len(clip_stems)} clips have matching shuttle .npy files.')
    return ok


def print_dataset_summary(clips_dir: Path = CLIPS_OUTPUT_DIR) -> None:
    """Print clip counts per split, per class.

    :param clips_dir: Root clips directory to summarise.
    """
    print('\n--- Dataset Summary ---')
    grand_total = 0

    for split_name in ['train', 'val', 'test']:
        split_dir = clips_dir / split_name
        if not split_dir.is_dir():
            print(f'\n  {split_name}: (not found)')
            continue

        counts = {}
        for subdir in sorted(split_dir.iterdir()):
            if subdir.is_dir():
                n = len(list(subdir.glob('*.mp4')))
                if n > 0:
                    counts[subdir.name] = n

        total = sum(counts.values())
        grand_total += total
        print(f'\n  {split_name}: {total} clips across {len(counts)} classes')
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f'    {cls:40s} {n:5d}')

    print(f'\n  Total: {grand_total} clips')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Verify ShuttleSet clip generation pipeline output.',
    )
    parser.add_argument('--clips-dir', type=Path, default=CLIPS_OUTPUT_DIR,
                        help='Directory containing generated clips')
    parser.add_argument('--shuttle-dir', type=Path, default=SHUTTLE_OUTPUT_DIR,
                        help='Directory containing shuttle .npy files')
    args = parser.parse_args()

    clips_dir = args.clips_dir
    if not clips_dir.exists():
        print(f'ERROR: Clips directory does not exist: {clips_dir}')
        print('Run clip_generator.py first.')
        sys.exit(1)

    # Scan once, reuse everywhere
    clip_paths = _scan_clips(clips_dir)

    checks = [
        verify_file_integrity(clip_paths),
        verify_splits_present(clips_dir, clip_paths),
        verify_no_excluded(clip_paths),
        verify_no_removed_shots(clip_paths),
        verify_class_merge(clips_dir),
        verify_shuttle_sync(clips_dir, args.shuttle_dir),
    ]

    warn_orphan_files(clips_dir, clip_paths)
    print_dataset_summary(clips_dir)

    if all(checks):
        print('\nAll checks PASSED.')
    else:
        print('\nSome checks FAILED. See above for details.')
        sys.exit(1)


if __name__ == '__main__':
    main()
