"""Dump per-clip videos_len from the npy collated dirs on bourbaki.

Produces a small CSV mapping clip_stem -> split -> videos_len. Once that file
is local, filter (c) "would-be discarded by bst_train" (= rows with
``videos_len == 0``) can be applied alongside filters (a) and (b) in the
class-by-player overlap analysis.

Run on bourbaki (or any host that has the npy collated dir present), from the
repo root, with the same PYTHONPATH bst_train uses:

    cd ~/badminton_stroke_classifier
    PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \\
      python scratch/research/dump_videos_len.py \\
      --output ~/discard_flags_split_v2_dropunk_nosides.csv

Defaults match the active nosides + dropunk + split_v2 + seq_len=100 + 2d
config. Override flags only if you want a different ablation.

After it runs, rsync the CSV down:

    boursync -av bourbaki:~/discard_flags_split_v2_dropunk_nosides.csv \\
      scratch/research/

The analysis script picks it up from there. The CSV is small (32k rows, ~1MB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same import path bst_train uses
from pipeline.config import (
    derive_npy_collated_dir_basename,
)


def reproduce_clip_order(
    clips_csv: Path,
    root_dir: Path | None,
    split_column: str,
    set_name: str,
    drop_unknown: bool,
    trust_clip_count: bool = False,
) -> list[str]:
    """Mirror the iteration in prepare_train_on_shuttleset.collate_npy.

    Reads clips_master.csv, filters to one split, optionally drops unknown,
    and iterates rows in DataFrame order. For each row, checks that the
    flat per-clip ``{stem}_pos.npy`` exists under root_dir; skips if not.
    Returns the ordered list of clip_stems.

    If ``trust_clip_count=True``, the existence check is skipped entirely
    and every filtered clip stem is returned. This is safe whenever the
    caller has already verified that the resulting list length matches the
    npy collation output (i.e. no clips were skipped during collate_npy).
    """
    df = pd.read_csv(clips_csv)
    if split_column not in df.columns:
        raise ValueError(
            f"{split_column!r} not in clips_master columns {list(df.columns)}"
        )
    df = df[df[split_column] == set_name].copy()
    if drop_unknown:
        df = df[df['raw_type_en'] != 'unknown']

    if trust_clip_count or root_dir is None:
        return [str(s) for s in df['clip_stem']]

    stems: list[str] = []
    missing = 0
    for stem in df['clip_stem']:
        if not (root_dir / f'{stem}_pos.npy').exists():
            missing += 1
            continue
        stems.append(str(stem))
    if missing:
        print(f'  [{set_name}] WARNING: {missing} clips missing flat pose '
              f'files under {root_dir}; skipped (matches collate_npy).',
              file=sys.stderr)
    return stems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips-csv', type=Path,
                    default=Path('notebooks/clips_master.csv'))
    ap.add_argument('--taxonomy', default='une_merge_v1_nosides')
    ap.add_argument('--split-column', default='split_v2')
    ap.add_argument('--drop-unknown', type=int, default=1,
                    help='1 to drop unknown class (default), 0 to keep.')
    ap.add_argument('--seq-len', type=int, default=100)
    ap.add_argument('--use-3d-pose', type=int, default=0)
    ap.add_argument('--ablation-id', default=None)
    ap.add_argument('--collated-root', type=Path, default=None,
                    help='Override the auto-derived collated dir parent. '
                         'Default: src/bst_refactor/stroke_classification/'
                         'preparing_data/ShuttleSet_data_<taxonomy>/<basename>/.')
    ap.add_argument('--flat-clip-dir', type=Path, default=None,
                    help='Override the flat per-clip dir used for the missing-'
                         'pose-file check. Default: collated_root parent / flat. '
                         'Only used when --trust-clip-count is not set.')
    ap.add_argument('--trust-clip-count', action='store_true',
                    help='Skip the per-clip existence check. Safe when the '
                         'filtered clips_master row count equals videos_len.npy '
                         'length (verified per split below). Use this if the '
                         'flat clip dir was pruned after collation.')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output CSV path: clip_stem,split,videos_len')
    args = ap.parse_args()

    drop_unknown = bool(args.drop_unknown)
    use_3d_pose = bool(args.use_3d_pose)

    # Collated dir basename + path (mirrors bst_train.train_network)
    npy_basename = derive_npy_collated_dir_basename(
        taxonomy_name=args.taxonomy,
        split_column=args.split_column,
        drop_unknown=drop_unknown,
        use_3d_pose=use_3d_pose,
        seq_len=args.seq_len,
        ablation_id=args.ablation_id,
    )
    if args.collated_root is None:
        collated_root = (
            Path('src/bst_refactor/stroke_classification/preparing_data')
            / f'ShuttleSet_data_{args.taxonomy}' / npy_basename
        )
    else:
        collated_root = args.collated_root

    # The flat per-clip dir is whatever Step 2 wrote to. It's normally a
    # sibling of the collated dir; the user can override if not.
    flat_clip_dir = (
        args.flat_clip_dir if args.flat_clip_dir is not None
        else collated_root.parent / 'flat'
    )

    print(f'Collated root: {collated_root}')
    if args.trust_clip_count:
        print('Per-clip existence check: disabled (--trust-clip-count).')
    else:
        print(f'Flat per-clip dir (for missing-file check): {flat_clip_dir}')

    if not collated_root.is_dir():
        print(f'ERROR: collated root not found: {collated_root}', file=sys.stderr)
        return 2
    if not args.trust_clip_count and not flat_clip_dir.is_dir():
        print(f'WARNING: flat clip dir not found, missing-file check will skip '
              f'every clip. Pass --flat-clip-dir to fix or --trust-clip-count '
              f'to skip the check entirely.', file=sys.stderr)

    rows: list[dict] = []
    for set_name in ('train', 'val', 'test'):
        videos_len_path = collated_root / set_name / 'videos_len.npy'
        if not videos_len_path.exists():
            print(f'  [{set_name}] missing {videos_len_path}', file=sys.stderr)
            continue
        videos_len = np.load(videos_len_path)
        print(f'  [{set_name}] videos_len.npy: {len(videos_len)} entries '
              f'(min={int(videos_len.min())}, '
              f'zero_count={int((videos_len == 0).sum())})')

        stems = reproduce_clip_order(
            clips_csv=args.clips_csv,
            root_dir=flat_clip_dir if not args.trust_clip_count else None,
            split_column=args.split_column,
            set_name=set_name,
            drop_unknown=drop_unknown,
            trust_clip_count=args.trust_clip_count,
        )
        if len(stems) != len(videos_len):
            print(f'  [{set_name}] LENGTH MISMATCH: stems={len(stems)} '
                  f'!= videos_len={len(videos_len)}. Cannot align safely; '
                  f'check that clips_master.csv and the flat dir are in '
                  f'sync with the collation that produced videos_len.npy.',
                  file=sys.stderr)
            return 3

        for stem, vl in zip(stems, videos_len):
            rows.append({'clip_stem': stem, 'split': set_name,
                         'videos_len': int(vl)})

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    n_zero = int((out['videos_len'] == 0).sum())
    print(f'\nWrote {len(out)} rows to {args.output}')
    print(f'Zero-videos_len clips (would be dropped by bst_train): {n_zero} '
          f'({n_zero / len(out) * 100:.2f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
