"""Apply a named heuristic to a raw MMPose extract, producing filtered outputs.

Reads per-clip raw arrays produced by ``preparing_data.raw_extract`` and
dispatches to a heuristic variant registered under ``preparing_data.heuristics``.
Writes the existing per-clip pipeline schema (``{stem}_pos.npy``,
``{stem}_joints.npy``, ``{stem}_failed.npy``) to ``--output-dir`` so that
downstream collation (``prepare_train_on_shuttleset`` step 3) is unchanged.

Refuses to run if ``--output-dir`` collides with ``--raw-dir`` or with the
``BST_X_MMPOSE_NPY_DIR`` environment variable -- the committed filtered
extract is never overwritten by this tool.

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \\
        python -m preparing_data.apply_heuristic \\
            --raw-dir /scratch/.../dataset_npy_..._flat_raw_phase1 \\
            --output-dir /scratch/.../dataset_npy_..._flat_h_sticky_anchor \\
            --heuristic sticky_anchor \\
            --clips-csv notebooks/clips_master.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.config import RESOLUTION_CSV_PATH, SET_INFO_DIR
from pipeline.court_utils import get_court_info
from pipeline.data_access import load_repo_dotenv

from preparing_data.heuristics import REGISTRY, ClipContext, RawClip
from preparing_data.heuristics.sticky_anchor import StickyAnchorParams

# Pull BST_X_MMPOSE_NPY_DIR from the repo-root .env so the output-dir
# collision guard works without a prior shell export.
load_repo_dotenv()


RAW_SUFFIXES = (
    "_raw_kps.npy",
    "_raw_bboxes.npy",
    "_raw_scores.npy",
    "_raw_kp_scores.npy",
    "_raw_ndet.npy",
)

OUT_SUFFIXES = ("_pos.npy", "_joints.npy", "_failed.npy")

DEFAULT_SPLITS = ("train", "val", "test")


@dataclass
class RunStats:
    attempted: int = 0
    processed: int = 0
    skipped_existing: int = 0
    skipped_missing_raw: int = 0
    skipped_missing_mp4_metadata: int = 0


def _resolve_or_none(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None


def _validate_output_dir(output_dir: Path, raw_dir: Path) -> None:
    """Refuse to write into the raw dir or the committed filtered dir.

    Two-line guard against typos destroying data we cannot cheaply
    recompute (the 1,716-clip raw extract is ~20 min of V100 time and the
    committed extract is the baseline for every comparison).
    """
    out_resolved = output_dir.resolve()
    raw_resolved = raw_dir.resolve()
    if out_resolved == raw_resolved:
        raise ValueError(
            f"--output-dir {output_dir} resolves to the same path as --raw-dir; "
            "refusing to overwrite raw extract."
        )

    mmpose_env = os.environ.get("BST_X_MMPOSE_NPY_DIR", "").strip()
    if mmpose_env:
        mmpose_resolved = Path(mmpose_env).resolve()
        if out_resolved == mmpose_resolved:
            raise ValueError(
                f"--output-dir {output_dir} resolves to BST_X_MMPOSE_NPY_DIR "
                f"({mmpose_env}); refusing to overwrite committed filtered extract."
            )


def _vid_from_stem(stem: str) -> int | None:
    head = stem.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _raw_files_present(raw_dir: Path, stem: str) -> bool:
    return all((raw_dir / f"{stem}{suf}").exists() for suf in RAW_SUFFIXES)


def _output_files_present(output_dir: Path, stem: str) -> bool:
    return all((output_dir / f"{stem}{suf}").exists() for suf in OUT_SUFFIXES)


def _load_raw_clip(raw_dir: Path, stem: str) -> RawClip:
    branch = str(raw_dir / stem)
    return RawClip(
        kps=np.load(branch + "_raw_kps.npy"),
        bboxes=np.load(branch + "_raw_bboxes.npy"),
        scores=np.load(branch + "_raw_scores.npy"),
        kp_scores=np.load(branch + "_raw_kp_scores.npy"),
        ndet=np.load(branch + "_raw_ndet.npy"),
    )


def _save_output(output_dir: Path, stem: str, pos, joints, failed) -> None:
    branch = str(output_dir / stem)
    np.save(branch + "_pos.npy", pos)
    np.save(branch + "_joints.npy", joints)
    np.save(branch + "_failed.npy", failed)


def _load_stems_file(path: Path) -> list[str]:
    with path.open() as fh:
        return [line.strip() for line in fh if line.strip()]


def _build_stem_list(
    *,
    clips_csv: Path,
    raw_dir: Path,
    clip_stems_file: Path | None,
    split_column: str | None,
    splits: tuple[str, ...] | None,
) -> list[str]:
    """Intersect clips_master with the raw-dir contents and optional stems filter.

    - Rows are filtered to ``splits`` on ``split_column`` when both are set.
    - Only stems whose five raw files are present on disk are returned.
    - If ``clip_stems_file`` is set, further restricts to that subset.
    - Output is sorted lexicographically for deterministic processing order.
    """
    df = pd.read_csv(clips_csv)
    if "clip_stem" not in df.columns:
        raise ValueError(f"clips_csv {clips_csv} is missing a clip_stem column")

    if split_column and splits:
        if split_column not in df.columns:
            raise ValueError(
                f"split-column {split_column!r} not in clips_csv {clips_csv}"
            )
        df = df[df[split_column].isin(splits)]

    csv_stems = set(df["clip_stem"].astype(str).tolist())

    if clip_stems_file is not None:
        subset = set(_load_stems_file(clip_stems_file))
        csv_stems &= subset

    eligible = sorted(
        stem for stem in csv_stems if _raw_files_present(raw_dir, stem)
    )
    return eligible


def _build_all_court_info(set_info_dir: Path, res_df: pd.DataFrame) -> dict:
    homo_df = pd.read_csv(set_info_dir / "homography.csv").set_index("id")
    return {vid: get_court_info(homo_df, vid) for vid in res_df.index}


def run(
    *,
    raw_dir: Path,
    output_dir: Path,
    heuristic: str,
    clips_csv: Path,
    clip_stems_file: Path | None = None,
    split_column: str | None = None,
    splits: tuple[str, ...] | None = DEFAULT_SPLITS,
    resume: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
    hyperparams: dict | None = None,
) -> RunStats:
    """Library entry point. See ``main`` for CLI plumbing."""
    if heuristic not in REGISTRY:
        raise ValueError(
            f"Unknown heuristic {heuristic!r}; registered: {sorted(REGISTRY)}"
        )

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw-dir not found: {raw_dir}")
    if not clips_csv.exists():
        raise FileNotFoundError(f"clips-csv not found: {clips_csv}")
    if clip_stems_file is not None and not clip_stems_file.exists():
        raise FileNotFoundError(f"clip-stems-file not found: {clip_stems_file}")

    _validate_output_dir(output_dir, raw_dir)

    stems = _build_stem_list(
        clips_csv=clips_csv,
        raw_dir=raw_dir,
        clip_stems_file=clip_stems_file,
        split_column=split_column,
        splits=splits,
    )
    if limit is not None:
        stems = stems[:limit]

    print(f"Eligible stems (raw files present): {len(stems)}")

    if dry_run:
        print("Dry run: showing first 5 eligible stems and exiting.")
        for stem in stems[:5]:
            print(f"  {stem}")
        return RunStats(attempted=len(stems))

    output_dir.mkdir(parents=True, exist_ok=True)

    res_df = pd.read_csv(str(RESOLUTION_CSV_PATH)).set_index("id")
    all_court_info = _build_all_court_info(SET_INFO_DIR, res_df)

    heuristic_fn = REGISTRY[heuristic]
    hyperparams = hyperparams or {}

    stats = RunStats(attempted=len(stems))
    for stem in tqdm(stems, desc=f"apply_heuristic:{heuristic}", unit="clip"):
        if resume and _output_files_present(output_dir, stem):
            stats.skipped_existing += 1
            continue

        vid = _vid_from_stem(stem)
        if vid is None or vid not in all_court_info:
            stats.skipped_missing_mp4_metadata += 1
            continue

        raw = _load_raw_clip(raw_dir, stem)
        ctx = ClipContext(vid=vid, all_court_info=all_court_info, res_df=res_df)
        out = heuristic_fn(raw, ctx, **hyperparams)
        _save_output(output_dir, stem, out.pos, out.joints, out.failed)
        stats.processed += 1

    print(
        f"\nDone. attempted={stats.attempted} processed={stats.processed} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_missing_metadata={stats.skipped_missing_mp4_metadata}"
    )
    return stats


def _add_hyperparam_args(parser: argparse.ArgumentParser) -> None:
    """Hyperparam CLI block, derived from StickyAnchorParams field names + defaults.

    ``current`` ignores these; ``sticky_anchor`` consumes them. Single source
    of truth lives on the dataclass so adding a field here means editing it
    in one place.
    """
    from dataclasses import fields  # local import keeps the module-level imports tidy

    # All fields on StickyAnchorParams are float; the dataclass annotations
    # are stringified by ``from __future__ import annotations`` so we
    # hard-code the argparse type rather than evaluating field.type strings.
    for field in fields(StickyAnchorParams):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=float, default=field.default)


def _hyperparam_dict_from_args(args: argparse.Namespace) -> dict:
    """Marshal argparse values into the kwargs accepted at the registry boundary."""
    from dataclasses import fields

    return {f.name: getattr(args, f.name) for f in fields(StickyAnchorParams)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--heuristic", type=str, required=True, choices=sorted(REGISTRY),
    )
    parser.add_argument(
        "--clips-csv", type=Path, required=True,
        help="clips_master.csv (or compatible) with clip_stem + split columns.",
    )
    parser.add_argument(
        "--clip-stems-file", type=Path, default=None,
        help="Optional one-stem-per-line file narrowing the candidate set.",
    )
    parser.add_argument(
        "--split-column", type=str, default=None,
        help="Column in clips-csv to filter by (e.g. split_v2, split_bst_baseline). "
             "Without a value, no split filter is applied.",
    )
    parser.add_argument(
        "--splits", type=str, default=",".join(DEFAULT_SPLITS),
        help="Comma-separated splits to keep. Ignored unless --split-column is set.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess stems whose output files already exist (default: skip them).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    _add_hyperparam_args(parser)
    args = parser.parse_args()

    splits_tuple = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    try:
        stats = run(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            heuristic=args.heuristic,
            clips_csv=args.clips_csv,
            clip_stems_file=args.clip_stems_file,
            split_column=args.split_column,
            splits=splits_tuple if args.split_column else None,
            resume=not args.no_resume,
            limit=args.limit,
            dry_run=args.dry_run,
            hyperparams=_hyperparam_dict_from_args(args),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if stats is not None else 1


if __name__ == "__main__":
    sys.exit(main())
