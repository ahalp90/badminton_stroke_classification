"""Rename teammate's downloaded videos to pipeline convention and generate metadata.

Renames files from '{id}_{resolution}_{fps}.mp4' to '{id} {match_name}.mp4'
(the format expected by clip_generator.py). Also generates:
  - A video metadata CSV with id, video name, url, width, height, fps, notes
  - my_raw_video_resolution.csv for the pipeline (id, width, height)

Usage:
    python scripts/rename_videos.py \
        --video-dir /scratch/comp320a-data \
        --match-csv data/shuttleset/set/match.csv \
        --flaw-csv data/shuttleset/flaw_shot_records.csv \
        --resolution-csv data/shuttleset/my_raw_video_resolution.csv \
        --dry-run
"""
import argparse
import csv
import os
import re
from pathlib import Path

RESOLUTION_MAP = {
    '2160p': (3840, 2160),
    '1080p': (1920, 1080),
    '720p': (1280, 720),
    '480p': (854, 480),
    '360p': (640, 360),
}


def load_match_csv(path: Path) -> dict[int, dict]:
    """Read match.csv and return {id: {video, url}} mapping."""
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            vid_id = int(row['id'])
            rows[vid_id] = {'video': row['video'], 'url': row['url']}
    return rows


def load_excluded_videos(path: Path) -> dict[int, str]:
    """Read flaw_shot_records.csv and return {id: reason} for whole-video exclusions."""
    excluded = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['stroke_type'] == 'whole':
                excluded[int(row['match'])] = row['reason']
    return excluded


def parse_video_filename(name: str) -> dict | None:
    """Parse '{id}_{resolution}_{fps}.mp4' into components.

    :param name: Filename like '01_1080p_25fps.mp4'.
    :return: Dict with id, resolution, width, height, fps, or None if unparseable.
    """
    stem = Path(name).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None

    # Leading digits as ID
    match = re.match(r'^(\d+)$', parts[0])
    if not match:
        return None
    vid_id = int(match.group(1))

    # Find resolution part (e.g. '1080p')
    width, height, res_label = None, None, None
    for part in parts[1:]:
        if part in RESOLUTION_MAP:
            res_label = part
            width, height = RESOLUTION_MAP[part]
            break

    # Find fps part (e.g. '25fps')
    fps = None
    for part in parts[1:]:
        fps_match = re.match(r'^(\d+)fps$', part)
        if fps_match:
            fps = int(fps_match.group(1))
            break

    if width is None:
        return None

    return {
        'id': vid_id,
        'resolution': res_label,
        'width': width,
        'height': height,
        'fps': fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Rename videos to pipeline convention and generate metadata.',
    )
    parser.add_argument('--video-dir', type=Path, required=True,
                        help="Directory with teammate's video files")
    parser.add_argument('--match-csv', type=Path, required=True,
                        help='Path to ShuttleSet match.csv')
    parser.add_argument('--flaw-csv', type=Path, required=True,
                        help='Path to flaw_shot_records.csv')
    parser.add_argument('--output-csv', type=Path, default=None,
                        help='Path for metadata CSV (default: video_metadata.csv in video-dir)')
    parser.add_argument('--resolution-csv', type=Path, default=None,
                        help='Path for my_raw_video_resolution.csv (pipeline format)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would happen without renaming')
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = args.video_dir / 'video_metadata.csv'
    if args.resolution_csv is None:
        args.resolution_csv = args.video_dir / 'my_raw_video_resolution.csv'

    matches = load_match_csv(args.match_csv)
    excluded = load_excluded_videos(args.flaw_csv)

    # Scan and parse local files
    local_files = {}
    unparseable = []
    for f in sorted(args.video_dir.glob('*.mp4')):
        parsed = parse_video_filename(f.name)
        if parsed is None:
            unparseable.append(f.name)
            continue
        parsed['path'] = f
        local_files[parsed['id']] = parsed

    if unparseable:
        print(f'Skipped {len(unparseable)} unparseable files: {", ".join(unparseable)}')

    # Plan renames
    renames = []
    for vid_id, info in sorted(local_files.items()):
        if vid_id not in matches:
            print(f'  WARNING: ID {vid_id} not found in match.csv, skipping')
            continue
        match_name = matches[vid_id]['video']
        new_name = f'{vid_id} {match_name}.mp4'
        old_path = info['path']
        new_path = old_path.parent / new_name
        renames.append((old_path, new_path))

    # Print rename plan
    print(f'\n{"DRY RUN - " if args.dry_run else ""}Renaming {len(renames)} files:\n')
    for old, new in renames:
        print(f'  {old.name}')
        print(f'    -> {new.name}\n')

    # Note missing/excluded
    all_ids = set(matches.keys())
    local_ids = set(local_files.keys())
    for vid_id in sorted(all_ids - local_ids):
        if vid_id in excluded:
            print(f'  ID {vid_id}: not on disk (excluded: {excluded[vid_id]})')
        else:
            print(f'  ID {vid_id}: MISSING — no local file and not excluded')

    if not args.dry_run:
        # Rename files
        for old, new in renames:
            os.rename(old, new)
        print(f'\nRenamed {len(renames)} files.')

        # Write metadata CSV
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'video', 'url', 'width', 'height', 'fps', 'note'])
            for vid_id in sorted(matches):
                m = matches[vid_id]
                if vid_id in excluded:
                    note = f'excluded: {excluded[vid_id]}'
                    writer.writerow([vid_id, m['video'], m['url'], '', '', '', note])
                elif vid_id in local_files:
                    info = local_files[vid_id]
                    writer.writerow([vid_id, m['video'], m['url'],
                                     info['width'], info['height'], info['fps'], ''])
                else:
                    writer.writerow([vid_id, m['video'], m['url'], '', '', '', 'missing'])
        print(f'Metadata CSV written: {args.output_csv}')

        # Write pipeline resolution CSV (all files on disk, with excluded flag)
        with open(args.resolution_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'width', 'height', 'excluded'])
            for vid_id in sorted(local_files):
                info = local_files[vid_id]
                is_excluded = 1 if vid_id in excluded else 0
                writer.writerow([vid_id, info['width'], info['height'], is_excluded])
        print(f'Resolution CSV written: {args.resolution_csv}')
    else:
        print('\nDry run complete. Remove --dry-run to execute.')


if __name__ == '__main__':
    main()
