"""Validate teammate's downloaded videos against ShuttleSet YouTube sources.

Cross-references video durations from ffprobe against YouTube metadata
(via yt-dlp --get-duration) to confirm each file matches the expected match.

Usage:
    python scripts/validate_videos.py \
        --video-dir /scratch/comp320a-data \
        --match-csv src/bst_refactor/ShuttleSet/set/match.csv \
        --cache-file youtube_durations.csv \
        --tolerance 5
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def load_match_csv(path: Path) -> dict[int, dict]:
    """Read match.csv and return {id: {url, video, duration_min}} mapping."""
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = int(row['id'])
            rows[vid_id] = {
                'url': row['url'],
                'video': row['video'],
                'duration_min': int(row['duration']),
            }
    return rows


def fetch_youtube_durations(
    matches: dict[int, dict],
    cache_path: Path,
    sleep_sec: float = 5.0,
) -> dict[int, float]:
    """Get video durations from YouTube, with CSV caching.

    :param matches: {id: {url, ...}} from match.csv.
    :param cache_path: Path to cache CSV (id,url,yt_duration_sec).
    :param sleep_sec: Seconds to sleep between yt-dlp requests.
    :return: {id: duration_in_seconds}.
    """
    cached = {}
    if cache_path.exists():
        with open(cache_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cached[int(row['id'])] = float(row['yt_duration_sec'])
        print(f"Loaded {len(cached)} cached durations from {cache_path}")

    # Figure out which IDs still need fetching
    to_fetch = {vid_id: m for vid_id, m in matches.items() if vid_id not in cached}

    if to_fetch:
        print(f"Fetching {len(to_fetch)} durations from YouTube "
              f"(~{len(to_fetch) * sleep_sec:.0f}s with {sleep_sec}s delay)...")

    for i, (vid_id, m) in enumerate(sorted(to_fetch.items())):
        url = m['url']
        try:
            result = subprocess.run(
                ['yt-dlp', '--get-duration', url],
                capture_output=True, text=True, timeout=30,
            )
            raw = result.stdout.strip()

            if not raw:
                print(f"  WARNING: yt-dlp failed for ID {vid_id}: "
                      f"{result.stderr.strip()[:200]}")
                continue

            # Parse duration string like "1:23:45" or "45:30"
            parts = raw.split(':')
            secs = 0.0
            for part in parts:
                secs = secs * 60 + float(part)
            cached[vid_id] = secs
            print(f"  [{i+1}/{len(to_fetch)}] ID {vid_id}: {raw} ({secs:.1f}s)")

        except subprocess.TimeoutExpired:
            print(f"  WARNING: timeout for ID {vid_id}")
        except Exception as e:
            print(f"  WARNING: error for ID {vid_id}: {e}")

        if i < len(to_fetch) - 1:
            time.sleep(sleep_sec)

    # Write full cache
    with open(cache_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'url', 'yt_duration_sec'])
        for vid_id in sorted(cached):
            url = matches[vid_id]['url'] if vid_id in matches else ''
            writer.writerow([vid_id, url, f"{cached[vid_id]:.2f}"])
    print(f"Cache saved to {cache_path}")

    return cached


def probe_local_durations(video_dir: Path) -> dict[int, float]:
    """Get durations of local video files via ffprobe.

    Expects filenames like '01_1080p_25fps.mp4' — extracts leading digits as ID.

    :param video_dir: Directory containing video files.
    :return: {id: duration_in_seconds}.
    """
    durations = {}
    files = sorted(video_dir.glob('*.mp4'))
    skipped = []

    for f in files:
        if f.suffix == '.part' or '.part' in f.name:
            skipped.append(f.name)
            continue

        match = re.match(r'^(\d+)', f.name)
        if not match:
            skipped.append(f.name)
            continue

        vid_id = int(match.group(1))

        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries',
                 'format=duration', '-of', 'json', str(f)],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            durations[vid_id] = float(data['format']['duration'])
        except Exception as e:
            print(f"  WARNING: ffprobe failed for {f.name}: {e}")

    if skipped:
        print(f"Skipped {len(skipped)} files: {', '.join(skipped)}")

    return durations


def validate(
    yt_durations: dict[int, float],
    local_durations: dict[int, float],
    matches: dict[int, dict],
    tolerance: float = 5.0,
) -> None:
    """Compare YouTube vs local durations and print a report."""
    all_ids = sorted(set(yt_durations) | set(local_durations) | set(matches))

    print()
    print(f"{'ID':>3}  {'Video':.<55} {'YT(s)':>9}  {'Local(s)':>9}  "
          f"{'Delta':>7}  Status")
    print("-" * 100)

    stats = {'match': 0, 'mismatch': 0, 'missing_local': 0,
             'missing_yt': 0, 'partial': 0}

    for vid_id in all_ids:
        name = matches.get(vid_id, {}).get('video', '???')[:55]
        yt = yt_durations.get(vid_id)
        local = local_durations.get(vid_id)

        if yt is None and local is None:
            continue

        yt_str = f"{yt:.1f}" if yt else "---"
        local_str = f"{local:.1f}" if local else "---"

        if yt is not None and local is not None:
            delta = abs(yt - local)
            delta_str = f"{delta:.1f}"
            if delta <= tolerance:
                status = "MATCH"
                stats['match'] += 1
            else:
                status = "MISMATCH"
                stats['mismatch'] += 1
        elif local is None:
            delta_str = "---"
            status = "NO LOCAL FILE"
            stats['missing_local'] += 1
        else:
            delta_str = "---"
            status = "NO YT DATA"
            stats['missing_yt'] += 1

        print(f"{vid_id:>3}  {name:.<55} {yt_str:>9}  {local_str:>9}  "
              f"{delta_str:>7}  {status}")

    print("-" * 100)
    print(f"Results: {stats['match']} match, {stats['mismatch']} mismatch, "
          f"{stats['missing_local']} missing local, "
          f"{stats['missing_yt']} no YT data")


def main():
    parser = argparse.ArgumentParser(
        description="Validate downloaded videos against YouTube sources.",
    )
    parser.add_argument('--video-dir', type=Path, required=True,
                        help="Directory with teammate's video files")
    parser.add_argument('--match-csv', type=Path, required=True,
                        help="Path to ShuttleSet match.csv")
    parser.add_argument('--cache-file', type=Path, default=Path('scratch/youtube_durations.csv'),
                        help="CSV to cache YouTube duration lookups")
    parser.add_argument('--tolerance', type=float, default=5.0,
                        help="Max duration difference (seconds) to count as MATCH")
    parser.add_argument('--skip-youtube', action='store_true',
                        help="Only use cached YouTube data, don't fetch new")
    args = parser.parse_args()

    matches = load_match_csv(args.match_csv)
    print(f"Loaded {len(matches)} videos from {args.match_csv}")

    if args.skip_youtube:
        if not args.cache_file.exists():
            print(f"ERROR: --skip-youtube but {args.cache_file} doesn't exist")
            sys.exit(1)
        yt_durations = {}
        with open(args.cache_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yt_durations[int(row['id'])] = float(row['yt_duration_sec'])
        print(f"Loaded {len(yt_durations)} cached durations (skipping YouTube)")
    else:
        yt_durations = fetch_youtube_durations(matches, args.cache_file)

    print(f"\nProbing local files in {args.video_dir}...")
    local_durations = probe_local_durations(args.video_dir)
    print(f"Found {len(local_durations)} valid video files")

    validate(yt_durations, local_durations, matches, args.tolerance)


if __name__ == '__main__':
    main()
