"""One-command orchestrator for the full ShuttleSet data pipeline.

Runs all pipeline steps in sequence:
  1. Download videos from YouTube (optional, skip with --skip-download)
  2. Build resolution CSV from downloaded videos (skip with --skip-resolution)
  3. Generate labeled clips (skip with --skip-clips)
  4. Apply class merge per active taxonomy (skip with --no-merge or --skip-clips)
  5. Verify clips (skip with --skip-verify)
  6. Extract shuttle trajectories via TrackNetV3 (skip with --skip-shuttle)

All paths default to pipeline.config values.

Usage:
    python -m pipeline.build_dataset --tracknet-dir /path/to/TrackNetV3
    python -m pipeline.build_dataset --skip-download --skip-resolution --skip-shuttle
    python -m pipeline.build_dataset --skip-clips --skip-verify --tracknet-dir TrackNetV3
    python -m pipeline.build_dataset --dry-run
"""
import argparse
import sys
from pathlib import Path

from pipeline.config import (
    RAW_VIDEO_DIR, CLIPS_OUTPUT_DIR, RESOLUTION_CSV_PATH,
    SPLITS, EXCLUDED_VIDEOS, REMOVED_SHOTS, CLIP_WINDOW,  # noqa: F401
    TAXONOMIES, Taxonomy, resolve_taxonomy,
)

from pipeline.download_videos import download_all_videos, build_resolution_csv
from pipeline.clip_generator import generate_all_clips, apply_class_merge
from pipeline.verify import (
    _scan_clips, verify_splits_present, verify_no_excluded,
    verify_no_removed_shots, verify_class_merge as verify_merge,
    verify_shuttle_sync, warn_orphan_files, print_dataset_summary,
)
from pipeline.shuttle_extractor import extract_all_shuttles, shuttle_csvs_to_npy

# Default taxonomy for pipeline operations when not explicitly overridden.
# Matches the project's working baseline; pick a different one via the CLI
# --taxonomy flag for one-off runs.
_DEFAULT_TAXONOMY_NAME = 'une_v1_14'
_DEFAULT_TAXONOMY = resolve_taxonomy(_DEFAULT_TAXONOMY_NAME)


def _step(number: int, title: str) -> None:
    """Print a step header."""
    print(f'\n--- Step {number}: {title} ---')


def _validate_inputs(
    tracknet_dir: Path | None,
    skip_download: bool,
    skip_shuttle: bool,
) -> None:
    """Fail fast if inputs are invalid, before any long-running work starts.

    :param tracknet_dir: Path to TrackNetV3 repo.
    :param skip_download: Whether the download step is skipped.
    :param skip_shuttle: Whether the shuttle step is skipped.
    :raises ValueError: If shuttle extraction is expected but no tracknet_dir given.
    :raises FileNotFoundError: If tracknet_dir doesn't exist or is missing predict.py,
        or if raw videos are missing when download is skipped.
    """
    # Shuttle extraction requires --tracknet-dir
    if not skip_shuttle:
        if tracknet_dir is None:
            raise ValueError(
                '--tracknet-dir is required unless --skip-shuttle is provided.'
            )
        if not tracknet_dir.is_dir():
            raise FileNotFoundError(
                f'TrackNetV3 directory not found: {tracknet_dir}'
            )
        if not (tracknet_dir / 'predict.py').exists():
            raise FileNotFoundError(
                f'predict.py not found in TrackNetV3 directory: {tracknet_dir}'
            )

    # If skipping download, raw videos must already exist
    if skip_download and not RAW_VIDEO_DIR.is_dir():
        raise FileNotFoundError(
            f'--skip-download but raw video directory not found: {RAW_VIDEO_DIR}'
        )


def dry_run(
    skip_download: bool = False,
    skip_resolution: bool = False,
    skip_clips: bool = False,
    skip_verify: bool = False,
    skip_shuttle: bool = False,
    no_merge: bool = False,
    tracknet_dir: Path | None = None,
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY,
) -> None:
    """Preview what the pipeline would do without executing anything.

    :param skip_download: Whether the download step is skipped.
    :param skip_resolution: Whether the resolution CSV rebuild is skipped.
    :param skip_clips: Whether clip generation and class merge are skipped.
    :param skip_verify: Whether verification is skipped.
    :param skip_shuttle: Whether the shuttle extraction step is skipped.
    :param no_merge: Whether class merging is skipped.
    :param tracknet_dir: Path to TrackNetV3 repo.
    :param taxonomy: Taxonomy to use for class merging and labelling.
    """
    print('=== DRY RUN (no files will be created or moved) ===\n')
    print(f'  taxonomy: {taxonomy.name} ({taxonomy.n_classes} classes)')

    skip_merge = skip_clips or no_merge or taxonomy.merge_map is None
    vid_count = sum(len(ids) for ids in SPLITS.values())
    split_summary = ', '.join(f'{k}={len(v)}' for k, v in SPLITS.items())

    merge_detail = 'SKIP'
    if not skip_merge:
        merge_detail = f'{len(taxonomy.merge_map)} subtypes merged into parents'

    clip_detail = 'SKIP' if skip_clips else (
        f'{vid_count} videos, splits: {split_summary}, '
        f'window: {CLIP_WINDOW}, '
        f'{len(REMOVED_SHOTS)} shots excluded, '
        f'output: {CLIPS_OUTPUT_DIR}'
    )

    steps = [
        ('1. Download videos',
         'SKIP' if skip_download else f'{vid_count} videos to {RAW_VIDEO_DIR}'),
        ('2. Build resolution CSV',
         'SKIP' if skip_resolution
         else f'Scan {RAW_VIDEO_DIR} and write resolution CSV'),
        ('3. Generate clips', clip_detail),
        ('4. Class merge', merge_detail),
        ('5. Verify clips',
         'SKIP' if skip_verify
         else 'Check splits, excluded videos, removed shots, merge'),
        ('6. Shuttle extraction',
         'SKIP' if skip_shuttle
         else f'TrackNetV3 at {tracknet_dir}'),
    ]

    for label, detail in steps:
        print(f'  {label}: {detail}')

    print(f'\n  Excluded videos: {sorted(EXCLUDED_VIDEOS)}')
    print(f'  Removed shots: {len(REMOVED_SHOTS)} individual shots')
    print('\n=== End dry run ===')


def run_pipeline(
    tracknet_dir: Path | None = None,
    skip_download: bool = False,
    skip_resolution: bool = False,
    skip_clips: bool = False,
    skip_verify: bool = False,
    skip_shuttle: bool = False,
    no_merge: bool = False,
    force: bool = False,
    workers: int = 2,
    batch_size: int = 32,
    tracknet_python: Path | None = None,
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY,
) -> None:
    """Run the full ShuttleSet data pipeline.

    All file paths come from pipeline.config. This function controls
    which steps to run and how many parallel workers to use.

    :param tracknet_dir: Path to cloned TrackNetV3 repo (required for step 6).
    :param skip_download: Skip YouTube download step.
    :param skip_resolution: Skip resolution CSV rebuild (keep existing CSV).
    :param skip_clips: Skip clip generation (step 3) and class merge (step 4).
    :param skip_verify: Skip verification checks (step 5).
    :param skip_shuttle: Skip TrackNetV3 shuttle extraction.
    :param no_merge: Skip class merging (keep all 19 stroke types).
    :param force: Continue to step 6 even if verification fails.
    :param workers: Parallel workers for downloads and TrackNetV3 (default 2).
    :param batch_size: Batch size for TrackNet DataLoader (default 32).
    :param tracknet_python: Python executable in BST venv (shared with TrackNetV3).
    :param taxonomy: Taxonomy to use for class merging.
    """
    _validate_inputs(tracknet_dir, skip_download, skip_shuttle)

    # Step 1: Download videos
    if not skip_download:
        _step(1, 'Downloading videos from YouTube')
        download_all_videos(max_workers=workers)
    else:
        print('Step 1: Skipped (--skip-download)')

    # Step 2: Build resolution CSV (check videos exist first)
    if not skip_resolution:
        _step(2, 'Building resolution CSV')
        video_files = list(RAW_VIDEO_DIR.glob('*.*'))
        if not video_files:
            print(f'ERROR: No video files found in {RAW_VIDEO_DIR}')
            print('Step 1 may have failed silently. Aborting.')
            sys.exit(1)
        build_resolution_csv()
    else:
        print('Step 2: Skipped (--skip-resolution)')

    # Step 3: Generate clips (check resolution CSV exists first)
    if not skip_clips:
        _step(3, 'Generating labeled clips')
        if not RESOLUTION_CSV_PATH.exists():
            print(f'ERROR: Resolution CSV not found: {RESOLUTION_CSV_PATH}')
            print('Step 2 may have failed. Aborting.')
            sys.exit(1)
        generate_all_clips()
    else:
        print('Step 3: Skipped (--skip-clips)')

    # Step 4: Apply class merge
    skip_merge = skip_clips or no_merge or taxonomy.merge_map is None
    if not skip_merge:
        n_merges = len(taxonomy.merge_map)
        _step(4, f'Applying class merge ({n_merges} subtypes -> parents)')
        apply_class_merge(taxonomy=taxonomy)
    else:
        print('Step 4: Skipped' + (' (--skip-clips)' if skip_clips
              else ' (no merge for this taxonomy)'))

    # Step 5: Verify clips
    if not skip_verify:
        _step(5, 'Verifying clips')
        clip_paths = _scan_clips(CLIPS_OUTPUT_DIR)
        checks = [
            verify_splits_present(CLIPS_OUTPUT_DIR, clip_paths),
            verify_no_excluded(clip_paths),
            verify_no_removed_shots(clip_paths),
        ]
        if not skip_merge:
            checks.append(verify_merge(taxonomy=taxonomy))
        warn_orphan_files(CLIPS_OUTPUT_DIR, clip_paths)
        print_dataset_summary()

        if not all(checks):
            if force:
                print('\nWARNING: Verification failed but --force is set. Continuing.')
            else:
                print('\nERROR: Verification failed. Aborting before shuttle extraction.')
                print('Use --force to continue anyway.')
                sys.exit(1)
    else:
        print('Step 5: Skipped (--skip-verify)')

    # Step 6: Extract shuttle trajectories
    if not skip_shuttle:
        _step(6, 'Extracting shuttle trajectories')
        extract_all_shuttles(
            tracknet_dir=tracknet_dir,
            tracknet_python=tracknet_python,
            max_workers=workers,
            batch_size=batch_size,
        )
        print()
        shuttle_csvs_to_npy()
        # Verify clip/shuttle sync
        verify_shuttle_sync()
    else:
        print('Step 6: Skipped (--skip-shuttle)')

    print('\nPipeline complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tracknet-dir', type=Path, default=None,
                        help='Path to cloned TrackNetV3 repository')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel workers for downloads and TrackNetV3')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for TrackNet DataLoader (default 32, use 64 with --workers 1)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip YouTube video download step')
    parser.add_argument('--skip-resolution', action='store_true',
                        help='Skip resolution CSV rebuild (keep existing CSV)')
    parser.add_argument('--skip-clips', action='store_true',
                        help='Skip clip generation and class merge (steps 3-4)')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip verification checks (step 5)')
    parser.add_argument('--skip-shuttle', action='store_true',
                        help='Skip TrackNetV3 shuttle trajectory extraction')
    parser.add_argument('--no-merge', action='store_true',
                        help='Skip class merging (keep all 19 stroke types)')
    parser.add_argument('--taxonomy', default=_DEFAULT_TAXONOMY_NAME, choices=list(TAXONOMIES.keys()),
                        help=f'Stroke type taxonomy (default: {_DEFAULT_TAXONOMY_NAME})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what the pipeline would do without executing')
    parser.add_argument('--force', action='store_true',
                        help='Continue past verification failures')
    parser.add_argument('--tracknet-python', type=Path, default=None,
                        help='Python executable in BST venv (shared with TrackNetV3)')
    args = parser.parse_args()

    # Validate early -- catches bad paths before both dry_run and run_pipeline
    _validate_inputs(args.tracknet_dir, args.skip_download, args.skip_shuttle)

    taxonomy = TAXONOMIES[args.taxonomy]

    if args.dry_run:
        dry_run(
            skip_download=args.skip_download,
            skip_resolution=args.skip_resolution,
            skip_clips=args.skip_clips,
            skip_verify=args.skip_verify,
            skip_shuttle=args.skip_shuttle,
            no_merge=args.no_merge,
            tracknet_dir=args.tracknet_dir,
            taxonomy=taxonomy,
        )
    else:
        run_pipeline(
            tracknet_dir=args.tracknet_dir,
            skip_download=args.skip_download,
            skip_resolution=args.skip_resolution,
            skip_clips=args.skip_clips,
            skip_verify=args.skip_verify,
            skip_shuttle=args.skip_shuttle,
            no_merge=args.no_merge,
            force=args.force,
            workers=args.workers,
            batch_size=args.batch_size,
            tracknet_python=args.tracknet_python,
            taxonomy=taxonomy,
        )
