"""Access ShuttleSet clips, shuttle npy, and mmpose npy filtered by split and taxonomy class.

On-disk layout (post-Phase-2)
-----------------------------
clips_dir/                       # Still nested (Phase 3 flattening deferred).
  {split_bst_baseline}/          # train | val | test, historical partition only.
    {folder_name}/               # e.g. Top_smash, Bottom_lob, unknown.
      {clip_stem}.mp4

shuttle_npy_dir/                 # Flat after Phase 2.2.
  {clip_stem}.npy

mmpose_npy_dir/                  # Flat after Phase 2.1; only present once pose
  {clip_stem}_joints.npy         # extraction has run.
  {clip_stem}_pos.npy

Split and taxonomy class both come from ``notebooks/clips_master.csv`` rather
than the folder structure. This lets the same backend serve any split column
(``split_bst_baseline``, ``split_v2``, future ablations) without reorganising
the clips tree.

Taxonomy class names
--------------------
The default taxonomy is 'une_v1_14' (14 classes, no sides, no unknown).
Picking a taxonomy that retains unknown (e.g. 'bst_25', 'une_v1_15') keeps
unknown rows in the output; picking a taxonomy whose
``excluded_base_stroke_types`` includes 'unknown' (e.g. 'bst_24', 'une_v1_14')
drops those rows automatically. No separate drop-unknown flag.

List all classes for the active taxonomy:

    python -m pipeline.data_access --list-classes

Python API
----------
    from pipeline.data_access import get_clip_records, DataPaths

    # Defaults: clips, shuttle_npy, clips_master from pipeline.config paths.
    # mmpose_npy_dir is left None until BST_X_MMPOSE_NPY_DIR is set or passed in.
    paths = DataPaths()

    # Filter by split and/or class. Both are optional.
    records = get_clip_records(
        paths,
        split='val',
        taxonomy_class='Top_smash',
        split_column='split_bst_baseline',
        taxonomy_name='bst_25',
    )

    for r in records:
        print(r.clip)           # Path to .mp4, or None if not on disk.
        print(r.shuttle_npy)    # Path to shuttle .npy, or None if missing.
        print(r.mmpose_joints)  # Path to _joints.npy, or None if not generated.

    # When mmpose data exists, pass its flat root directory.
    paths = DataPaths(
        mmpose_npy_dir=Path(
            'preparing_data/ShuttleSet_data_bst_25/'
            'dataset_npy_between_2_hits_with_max_limits_flat'
        )
    )

CLI usage
---------
Run from the project root (or any directory with pipeline importable):

    # Count table for all splits and classes
    python -m pipeline.data_access --summary

    # Count table filtered to one split
    python -m pipeline.data_access --split val --summary

    # Count table for one class across all splits
    python -m pipeline.data_access --class Top_smash --summary

    # Switch to the split_v2 ablation column
    python -m pipeline.data_access --split-column split_v2 --summary

    # Pick a taxonomy that drops unknown (excluded_base_stroke_types handles it)
    python -m pipeline.data_access --taxonomy bst_24 --summary

    # TSV of all file paths (clip, shuttle, mmpose) -- redirect to file
    python -m pipeline.data_access --split train > train_paths.tsv

    # List all class names in the active taxonomy and exit
    python -m pipeline.data_access --list-classes

    # Override default data paths (e.g. different HPC scratch location)
    python -m pipeline.data_access \\
        --clips-dir /scratch/comp320a/ShuttleSet/clips \\
        --shuttle-npy-dir /scratch/comp320a/ShuttleSet/shuttle_npy_flat \\
        --summary

    # Include mmpose paths once pose estimation has been run
    python -m pipeline.data_access \\
        --mmpose-npy-dir /scratch/comp320a/ShuttleSet_data_bst_25/\\
dataset_npy_between_2_hits_with_max_limits_flat \\
        --summary

Environment / .env file
-----------------------
Instead of passing flags every time, set paths in a .env file at the project
root (copy .env.example and fill in your values):

    BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips
    BST_X_SHUTTLE_NPY_DIR=/scratch/comp320a/ShuttleSet/shuttle_npy_flat
    BST_X_MMPOSE_NPY_DIR=/scratch/comp320a/ShuttleSet_data_bst_25/dataset_npy_between_2_hits_with_max_limits_flat
    BST_X_CLIPS_CSV=/home/username/badminton_stroke_classifier/notebooks/clips_master.csv

Then just run with no flags:

    python -m pipeline.data_access --summary
    python -m pipeline.data_access          # interactive TUI

Shell exports take precedence over .env, so on-the-fly overrides work:

    BST_X_CLIPS_DIR=/other/path python -m pipeline.data_access --summary

Relationship to ``clip_index.py``
---------------------------------
``pipeline.clip_index.build_clip_path_index(clips_dir)`` is the zero-dep pathlib
helper that builds a ``{clip_stem -> Path}`` lookup. ``data_access`` calls it
internally to resolve clip paths against the still-nested clips tree. Use
``clip_index`` directly when you only need the stem-to-path map; use
``data_access`` when you also want split + taxonomy filtering and paired
shuttle / mmpose resolution.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pipeline.clip_index import build_clip_path_index
from pipeline.config import (
    CLIPS_OUTPUT_DIR,
    SHUTTLE_OUTPUT_DIR,
    TAXONOMIES,
    Taxonomy,
    label_for_row,
    resolve_taxonomy,
)


SPLITS = ('train', 'val', 'test')

DEFAULT_SPLIT_COLUMN = 'split_bst_baseline'

# Default taxonomy for data_access CLI / library calls when no explicit one is
# passed. Picks the most permissive registered taxonomy (bst_25 keeps every
# raw type and produces sided folder names) so exploring the data doesn't
# silently drop rows. Training uses a narrower default (une_v1_14) in
# bst_x_train.py's Hyp tuple; that's a separate concern from data exploration.
DEFAULT_TAXONOMY_NAME = 'bst_25'

# .env is searched for in the project root. data_access.py lives at
# src/bst_x/pipeline/, so four .parent hops land at the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_PATH = _PROJECT_ROOT / '.env'

# Default clips_master.csv location (repo-relative).
_DEFAULT_CLIPS_CSV = _PROJECT_ROOT / 'notebooks' / 'clips_master.csv'


def load_repo_dotenv(path: Path = _DOTENV_PATH) -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if missing).

    Only sets variables that are not already set in the environment, so
    shell exports always take precedence over the .env file. Idempotent:
    safe to call multiple times.

    :param path: Path to the .env file.
    """
    if not path.is_file():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            # Strip inline comments (# after the value) then surrounding quotes.
            value = value.split('#', 1)[0].strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def env_path(var: str, default: Path) -> Path:
    """Return Path from env var if set, otherwise the default."""
    val = os.environ.get(var, '').strip()
    return Path(val) if val else default


def env_path_or_none(var: str) -> Path | None:
    """Return Path from env var if set and non-empty, otherwise None."""
    val = os.environ.get(var, '').strip()
    return Path(val) if val else None


@dataclass
class DataPaths:
    """Root directories for each data type plus the master clips CSV.

    Path resolution order (highest to lowest priority):
      1. Value passed directly to the constructor.
      2. Environment variable (or .env file entry).
      3. Default from ``pipeline.config`` / repo root.

    Environment variables:
      BST_X_CLIPS_DIR        -- root clips directory (nested by split/class).
      BST_X_SHUTTLE_NPY_DIR  -- flat shuttle npy directory.
      BST_X_MMPOSE_NPY_DIR   -- flat mmpose per-clip npy directory (omit if not
                              generated yet).
      BST_X_CLIPS_CSV        -- path to clips_master.csv.

    These can be set in a .env file at the project root (see .env.example).

    :param clips_dir: Root of the clips tree. Still nested
        ``{split_bst_baseline}/{folder_name}/{stem}.mp4`` (Phase 3 flattening
        deferred).
    :param shuttle_npy_dir: Flat shuttle npy dir: ``{stem}.npy`` per clip.
    :param mmpose_npy_dir: Flat mmpose per-clip npy dir holding
        ``{stem}_joints.npy`` and ``{stem}_pos.npy``, or None if pose
        extraction has not been run.
    :param clips_csv: Path to ``notebooks/clips_master.csv``, the source of
        truth for split + taxonomy-class assignment per clip.
    """

    clips_dir: Path = field(
        default_factory=lambda: env_path('BST_X_CLIPS_DIR', CLIPS_OUTPUT_DIR)
    )
    shuttle_npy_dir: Path = field(
        default_factory=lambda: env_path('BST_X_SHUTTLE_NPY_DIR', SHUTTLE_OUTPUT_DIR)
    )
    mmpose_npy_dir: Path | None = field(
        default_factory=lambda: env_path_or_none('BST_X_MMPOSE_NPY_DIR')
    )
    clips_csv: Path = field(
        default_factory=lambda: env_path('BST_X_CLIPS_CSV', _DEFAULT_CLIPS_CSV)
    )


@dataclass
class ClipRecord:
    """Paths for a single clip and its associated data files.

    :param split: Dataset split ('train', 'val', or 'test') as read from the
        CSV ``split_column``.
    :param taxonomy_class: Derived class label under the active taxonomy, e.g.
        'Top_smash' or 'unknown'. This matches the folder name in the nested
        clips tree.
    :param clip_stem: Clip identifier, e.g. '1_1_3_2'.
    :param clip: Path to the .mp4 clip file, or None if the stem is not found
        on disk.
    :param shuttle_npy: Path to the flat shuttle .npy, or None if missing.
    :param mmpose_joints: Path to the flat ``_joints.npy``, or None if missing
        or mmpose_npy_dir is not set.
    :param mmpose_pos: Path to the flat ``_pos.npy``, or None if missing or
        mmpose_npy_dir is not set.
    """

    split: str
    taxonomy_class: str
    clip_stem: str
    clip: Path | None
    shuttle_npy: Path | None
    mmpose_joints: Path | None
    mmpose_pos: Path | None


def _derive_class_label(
    raw_type_en: str, player_side: str, taxonomy: Taxonomy,
) -> str | None:
    """Resolve a folder-style class label, or None if the row is filtered out.

    Thin wrapper around ``pipeline.config.label_for_row``: that function is the
    single source of truth for the merge_map + side-prefixing + side-agnostic
    rules. This adapter just looks up the resulting class string from
    ``taxonomy.classes`` so callers that work in label-name space (the CSV-row
    loop, the validation scripts) don't have to.

    Returns None when ``raw_type_en`` is in
    ``taxonomy.excluded_base_stroke_types`` (the post-refactor replacement for
    the historical drop_unknown flag).

    :param raw_type_en: ``raw_type_en`` value from clips_master.csv.
    :param player_side: ``'Top'`` or ``'Bottom'``. Ignored when the taxonomy
        is sideless or the merged type is side-agnostic.
    :param taxonomy: target Taxonomy.
    :return: folder-style label string (e.g. ``'Top_smash'``, ``'unknown'``)
        or None if the row should be dropped.
    """
    idx = label_for_row(taxonomy, raw_type_en, player_side)
    return None if idx is None else taxonomy.classes[idx]


def get_clip_records(
    paths: DataPaths,
    split: str | None = None,
    taxonomy_class: str | None = None,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    taxonomy_name: str = DEFAULT_TAXONOMY_NAME,
) -> list[ClipRecord]:
    """Return ClipRecords read from ``clips_master.csv`` under the active taxonomy.

    Reads the master CSV, filters by ``split`` / ``taxonomy_class``, derives
    the folder-style class label via the active taxonomy, and resolves each
    row's paired files on disk: clip from the still-nested clips tree (via
    ``clip_index.build_clip_path_index``), shuttle and mmpose files from the
    flat post-Phase-2 dirs.

    Rows whose ``raw_type_en`` is in
    ``taxonomy.excluded_base_stroke_types`` are dropped automatically; pick a
    taxonomy that retains unknown (``'bst_25'``, ``'une_v1_15'``) to keep
    those rows in.

    Files missing on disk resolve to None on the record rather than dropping
    the row. That keeps the result aligned with the CSV and lets callers
    distinguish "CSV says this clip exists but its shuttle wasn't extracted"
    from "this clip isn't in the CSV at all".

    :param paths: Root directories for each data type plus the master CSV.
    :param split: One of 'train', 'val', 'test', or None for all splits.
    :param taxonomy_class: Derived class label (e.g. 'Top_smash', 'unknown'),
        or None for all classes. Use ``resolve_taxonomy(taxonomy_name).classes``
        to enumerate valid names.
    :param split_column: Column in clips_csv giving the train/val/test
        assignment, e.g. 'split_bst_baseline' (default) or 'split_v2'.
    :param taxonomy_name: Name (or legacy alias) of the Taxonomy whose
        merge_map + side rule + excluded_base_stroke_types drive label
        derivation. Resolved via ``pipeline.config.resolve_taxonomy``.
    :raises ValueError: If ``split`` or ``taxonomy_class`` are not valid under
        the chosen taxonomy.
    :raises KeyError: If ``split_column`` is not a column in clips_csv or
        ``taxonomy_name`` is not in TAXONOMIES (and not a registered alias).
    :return: List of ClipRecord in CSV row order.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}, got {split!r}')

    taxonomy = resolve_taxonomy(taxonomy_name)

    if taxonomy_class is not None:
        valid_classes = set(taxonomy.classes)
        if taxonomy_class not in valid_classes:
            raise ValueError(
                f'{taxonomy_class!r} is not a class in taxonomy '
                f'{taxonomy.name!r}. Valid classes: {sorted(valid_classes)}'
            )

    df = pd.read_csv(paths.clips_csv)
    if split_column not in df.columns:
        raise KeyError(
            f'split_column {split_column!r} not in clips_csv columns: '
            f'{list(df.columns)}'
        )
    df = df[df[split_column].isin(SPLITS)].copy()
    if split is not None:
        df = df[df[split_column] == split].copy()

    if df.empty:
        return []

    # Build stem-to-path lookup once. Empty / missing clips_dir yields {}.
    path_by_stem: dict[str, Path]
    if paths.clips_dir.is_dir():
        path_by_stem = build_clip_path_index(paths.clips_dir)
    else:
        path_by_stem = {}

    records: list[ClipRecord] = []
    for row in df.itertuples(index=False):
        stem = row.clip_stem
        label = _derive_class_label(
            row.raw_type_en, row.player_side, taxonomy,
        )
        # excluded_base_stroke_types drops rows (e.g. unknown rows under
        # bst_24); the old drop_unknown flag is folded in here.
        if label is None:
            continue
        if taxonomy_class is not None and label != taxonomy_class:
            continue

        clip = path_by_stem.get(stem)

        shuttle = paths.shuttle_npy_dir / f'{stem}.npy'
        shuttle = shuttle if shuttle.exists() else None

        joints: Path | None = None
        pos: Path | None = None
        if paths.mmpose_npy_dir is not None:
            j = paths.mmpose_npy_dir / f'{stem}_joints.npy'
            p = paths.mmpose_npy_dir / f'{stem}_pos.npy'
            joints = j if j.exists() else None
            pos = p if p.exists() else None

        records.append(ClipRecord(
            split=getattr(row, split_column),
            taxonomy_class=label,
            clip_stem=stem,
            clip=clip,
            shuttle_npy=shuttle,
            mmpose_joints=joints,
            mmpose_pos=pos,
        ))

    return records


def summarise(
    paths: DataPaths,
    split: str | None = None,
    taxonomy_class: str | None = None,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    taxonomy_name: str = DEFAULT_TAXONOMY_NAME,
) -> None:
    """Print a per-split, per-class count table for the filtered selection.

    :param paths: Root directories for each data type.
    :param split: Split filter, or None for all.
    :param taxonomy_class: Class filter, or None for all.
    :param split_column: CSV column giving the split assignment.
    :param taxonomy_name: Name (or legacy alias) of the Taxonomy for label
        derivation.
    """
    records = get_clip_records(
        paths,
        split=split,
        taxonomy_class=taxonomy_class,
        split_column=split_column,
        taxonomy_name=taxonomy_name,
    )

    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {'clips': 0, 'clips_on_disk': 0, 'shuttle': 0, 'mmpose': 0}
        )
    )
    for r in records:
        c = counts[r.split][r.taxonomy_class]
        c['clips'] += 1
        if r.clip is not None:
            c['clips_on_disk'] += 1
        if r.shuttle_npy:
            c['shuttle'] += 1
        if r.mmpose_joints:
            c['mmpose'] += 1

    for sp in SPLITS:
        if sp not in counts:
            continue
        print(f'\n{sp}:')
        for cls_name, c in sorted(counts[sp].items()):
            mmpose_str = f"  mmpose={c['mmpose']}" if paths.mmpose_npy_dir else ''
            print(
                f"  {cls_name:<40}  clips={c['clips']}"
                f"  on_disk={c['clips_on_disk']}"
                f"  shuttle={c['shuttle']}{mmpose_str}"
            )

    total = len(records)
    on_disk_total = sum(1 for r in records if r.clip is not None)
    shuttle_total = sum(1 for r in records if r.shuttle_npy)
    print(
        f'\nTotal: {total} clip rows, {on_disk_total} clips on disk, '
        f'{shuttle_total} shuttle npys'
    )
    if paths.mmpose_npy_dir:
        mmpose_total = sum(1 for r in records if r.mmpose_joints)
        print(f'       {mmpose_total} mmpose npy sets')


def _menu(prompt: str, options: list[str]) -> str:
    """Print a numbered menu and return the chosen option."""
    print(f'\n{prompt}')
    for i, opt in enumerate(options, 1):
        print(f'  {i}) {opt}')
    while True:
        raw = input('> ').strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f'  Enter a number between 1 and {len(options)}.')


def interactive(
    paths: DataPaths,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    taxonomy_name: str = DEFAULT_TAXONOMY_NAME,
) -> None:
    """Step-through TUI: pick split, class, and output type interactively.

    :param paths: Root directories for each data type.
    :param split_column: Initial split column; overridable at the split-column prompt.
    :param taxonomy_name: Initial taxonomy; overridable at the taxonomy prompt.
    """
    # Step 0: split column (only the columns present in the CSV).
    df = pd.read_csv(paths.clips_csv)
    available_split_cols = [c for c in df.columns if c.startswith('split_')]
    if not available_split_cols:
        print(f'No split_* columns found in {paths.clips_csv}.')
        return
    if split_column not in available_split_cols:
        split_column = available_split_cols[0]
    split_column = _menu('Select split column:', available_split_cols)

    # Step 1: taxonomy.
    taxonomy_name = _menu('Select taxonomy:', list(TAXONOMIES))

    # Step 2: split.
    split_choice = _menu('Select split:', ['all'] + list(SPLITS))
    split = None if split_choice == 'all' else split_choice

    # Step 3: taxonomy class (drawn from the active taxonomy).
    class_options = ['all'] + list(TAXONOMIES[taxonomy_name].classes)
    class_choice = _menu('Select taxonomy class:', class_options)
    taxonomy_class = None if class_choice == 'all' else class_choice

    # Step 4: output. (No drop-unknown prompt; the taxonomy carries the rule
    # via excluded_base_stroke_types -- pick bst_24 to drop, bst_25 to keep.)
    output_choice = _menu('Show:', ['summary table', 'file paths'])

    print()
    if output_choice == 'summary table':
        summarise(
            paths,
            split=split,
            taxonomy_class=taxonomy_class,
            split_column=split_column,
            taxonomy_name=taxonomy_name,
        )
    else:
        records = get_clip_records(
            paths,
            split=split,
            taxonomy_class=taxonomy_class,
            split_column=split_column,
            taxonomy_name=taxonomy_name,
        )
        for r in records:
            clip_str = str(r.clip) if r.clip else 'MISSING_CLIP'
            shuttle_str = str(r.shuttle_npy) if r.shuttle_npy else 'MISSING'
            mmpose_str = str(r.mmpose_joints) if r.mmpose_joints else 'NO_MMPOSE'
            print(
                f'{r.split}\t{r.taxonomy_class}\t{r.clip_stem}\t'
                f'{clip_str}\t{shuttle_str}\t{mmpose_str}'
            )


def _build_cli() -> argparse.ArgumentParser:
    """Build the argparse parser. Factored out for testability."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--split', choices=list(SPLITS), default=None,
        help='Filter to one split (default: all splits).',
    )
    parser.add_argument(
        '--class', dest='taxonomy_class', default=None,
        help='Filter to one taxonomy class, e.g. Top_smash (default: all).',
    )
    parser.add_argument(
        '--split-column', default=DEFAULT_SPLIT_COLUMN,
        help=f'CSV column giving train/val/test (default: {DEFAULT_SPLIT_COLUMN}).',
    )
    parser.add_argument(
        '--taxonomy', choices=list(TAXONOMIES), default=DEFAULT_TAXONOMY_NAME,
        help=f'Taxonomy for label derivation and class validation. The chosen '
             f"taxonomy's excluded_base_stroke_types drives row filtering "
             f'(no separate drop-unknown flag any more). '
             f'(default: {DEFAULT_TAXONOMY_NAME}).',
    )
    parser.add_argument(
        '--clips-dir', type=Path, default=None,
        help='Root clips directory (overrides BST_X_CLIPS_DIR + config default).',
    )
    parser.add_argument(
        '--shuttle-npy-dir', type=Path, default=None,
        help='Flat shuttle npy directory (overrides BST_X_SHUTTLE_NPY_DIR '
             '+ config default).',
    )
    parser.add_argument(
        '--mmpose-npy-dir', type=Path, default=None,
        help='Flat mmpose per-clip npy directory (overrides BST_X_MMPOSE_NPY_DIR).',
    )
    parser.add_argument(
        '--clips-csv', type=Path, default=None,
        help='Path to clips_master.csv (overrides BST_X_CLIPS_CSV + repo default).',
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print per-split/class count table instead of individual paths.',
    )
    parser.add_argument(
        '--list-classes', action='store_true',
        help='List the class names for the active taxonomy and exit.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint. ``argv=None`` uses ``sys.argv[1:]``."""
    load_repo_dotenv()
    args = _build_cli().parse_args(argv)

    # Build DataPaths -- only pass CLI values that were explicitly set so that
    # DataPaths' default_factory can pick up env vars / .env for anything omitted.
    path_kwargs = {}
    if args.clips_dir is not None:
        path_kwargs['clips_dir'] = args.clips_dir
    if args.shuttle_npy_dir is not None:
        path_kwargs['shuttle_npy_dir'] = args.shuttle_npy_dir
    if args.mmpose_npy_dir is not None:
        path_kwargs['mmpose_npy_dir'] = args.mmpose_npy_dir
    if args.clips_csv is not None:
        path_kwargs['clips_csv'] = args.clips_csv
    paths = DataPaths(**path_kwargs)

    # No filters + no action flags -> launch the TUI.
    no_flags = not any([
        args.split, args.taxonomy_class, args.summary, args.list_classes,
    ])
    if no_flags:
        interactive(
            paths,
            split_column=args.split_column,
            taxonomy_name=args.taxonomy,
        )
    elif args.list_classes:
        for name in TAXONOMIES[args.taxonomy].classes:
            print(name)
    elif args.summary:
        summarise(
            paths,
            split=args.split,
            taxonomy_class=args.taxonomy_class,
            split_column=args.split_column,
            taxonomy_name=args.taxonomy,
        )
    else:
        records = get_clip_records(
            paths,
            split=args.split,
            taxonomy_class=args.taxonomy_class,
            split_column=args.split_column,
            taxonomy_name=args.taxonomy,
        )
        for r in records:
            clip_str = str(r.clip) if r.clip else 'MISSING_CLIP'
            shuttle_str = str(r.shuttle_npy) if r.shuttle_npy else 'MISSING'
            mmpose_str = str(r.mmpose_joints) if r.mmpose_joints else 'NO_MMPOSE'
            print(
                f'{r.split}\t{r.taxonomy_class}\t{r.clip_stem}\t'
                f'{clip_str}\t{shuttle_str}\t{mmpose_str}'
            )


if __name__ == '__main__':
    main()
