"""Batch TrackNetV3 inference: load models once, process many clips.

Designed for pipeline shuttle extraction where ~33k short clips need
inference. Loading models once saves ~8s per clip vs. spawning
predict.py as a subprocess each time.

Usage:
    python batch_predict.py \
        --video_list pending_clips.txt \
        --tracknet_file ckpts/TrackNet_best.pt \
        --inpaintnet_file ckpts/InpaintNet_best.pt \
        --save_dir /path/to/shuttle_csv
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

from predict import load_models, predict_video


def main():
    parser = argparse.ArgumentParser(
        description='Batch TrackNetV3 inference: load models once, process many clips.',
    )
    parser.add_argument('--video_list', type=str, required=True,
                        help='Text file with one video path per line')
    parser.add_argument('--tracknet_file', type=str, required=True,
                        help='Path to TrackNet checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='',
                        help='Path to InpaintNet checkpoint (optional)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to write output CSVs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoader (default 16)')
    parser.add_argument('--eval_mode', type=str, default='weight',
                        choices=['nonoverlap', 'average', 'weight'],
                        help='Temporal ensemble mode (default: weight)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Run inference without writing output files')
    args = parser.parse_args()

    # Load models once (the whole point of batch mode)
    print('Loading models...', flush=True)
    tracknet, inpaintnet, t_seq, i_seq, bg_mode = load_models(
        args.tracknet_file, args.inpaintnet_file or None
    )
    print('Models loaded.', flush=True)

    # Read clip list
    video_paths = Path(args.video_list).read_text().strip().splitlines()
    total = len(video_paths)
    print(f'Batch mode: {total} clips to process', flush=True)

    successes, failures, skipped = 0, 0, 0
    for i, video_file in enumerate(video_paths, 1):
        video_file = video_file.strip()
        if not video_file:
            continue

        # Skip already-done clips (resumable after crash)
        stem = Path(video_file).stem
        if not args.dry_run:
            out_csv = Path(args.save_dir) / f'{stem}_ball.csv'
            if out_csv.exists():
                skipped += 1
                continue

        print(f'PROCESSING ({i}/{total}) {stem}', flush=True)
        try:
            predict_video(
                video_file, tracknet, inpaintnet, t_seq, i_seq, bg_mode,
                args.save_dir, eval_mode=args.eval_mode,
                batch_size=args.batch_size, dry_run=args.dry_run,
            )
            successes += 1
        except Exception as e:
            print(f'ERROR ({i}/{total}) {stem}: {e}', file=sys.stderr,
                  flush=True)
            failures += 1

        # Free frame arrays and CUDA cache between clips
        gc.collect()
        torch.cuda.empty_cache()

        # Progress line (parseable by shuttle_extractor)
        print(f'BATCH_PROGRESS ({i}/{total}) {stem}', flush=True)

    print(f'BATCH_COMPLETE successes={successes} failures={failures} '
          f'skipped={skipped}', flush=True)


if __name__ == '__main__':
    main()
