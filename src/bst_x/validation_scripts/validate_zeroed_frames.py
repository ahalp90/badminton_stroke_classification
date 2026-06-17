"""Analyze detection failures across the pose/shuttle dataset.

Two independent failure modes are measured:

1. **MMPose failures** (from ``*_failed.npy``): MMPose failed to detect exactly
   2 players on court. Joints, court positions, AND shuttle coordinates are all
   zeroed on these frames at collation time. The BST transformer does NOT mask
   them in attention — they participate as zero vectors.

2. **Shuttle detection failures** (from shuttle NPYs, optional): TrackNetV3
   reported visibility=0 (shuttle not detected). These are independent of
   MMPose — a frame can have good pose data but no shuttle, or vice versa.
   The visibility column is dropped during collation, so these failures are
   invisible to the model as silent (0, 0) shuttle coordinates.

Reports overall, per-split, per-stroke-type failure rates for both modes,
a 2×2 overlap analysis, and hit-frame proximity breakdowns.

Outputs (text + PNGs) are saved to a sibling folder:
    validation_scripts/zeroed_frames_analysis_outputs/

CSV-driven: splits and labels come from ``notebooks/clips_master.csv``.
Per-clip npy files are resolved flat under ``{dataset_npy_dir}/{clip_stem}_*.npy``
and ``{shuttle_npy_dir}/{clip_stem}.npy``.

Usage:
    python validate_zeroed_frames.py \
        --data-root /path/to/ShuttleSet_data_une_v1_14 \
        --clips-csv /path/to/notebooks/clips_master.csv \
        --taxonomy une_v1_14 \
        --split-column split_bst_baseline \
        --threshold 0.5 \
        --set-dir /path/to/data/shuttleset/set \
        --shuttle-npy-dir /path/to/data/shuttleset/shuttle_npy_flat
"""
import argparse
import io
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC (no X11 display)
import matplotlib.pyplot as plt  # noqa: E402

BST_REFACTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BST_REFACTOR_ROOT))

from pipeline.config import TAXONOMIES, Taxonomy, resolve_taxonomy, SIDE_AGNOSTIC_TYPES  # noqa: E402


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class ClipRecord(NamedTuple):
    clip_name: str        # e.g. "35_1_10_17"
    rel_path: str         # e.g. "train/Top_smash/35_1_10_17"
    split: str            # "train", "val", or "test"
    stroke_type: str      # player prefix stripped, e.g. "smash"
    player: str           # "Top", "Bottom", or "" for unprefixed (e.g. unknown)
    total_frames: int
    failed_frames: int
    fail_rate: float
    temporal_bins: np.ndarray  # shape (n_bins,), per-bin fail rates
    is_flaw: bool | None  # True if flaw=1.0 in set CSV, None if lookup unavailable
    failed_arr: np.ndarray    # raw boolean array, shape (total_frames,)
    shuttle_vis: np.ndarray | None = None  # True = shuttle NOT detected (bad), shape (total_frames,)


# ---------------------------------------------------------------------------
# Tee helper — writes to both stdout and a StringIO buffer
# ---------------------------------------------------------------------------

class _Tee:
    """Minimal stdout tee that also captures output for saving to .txt."""

    def __init__(self):
        self._buf = io.StringIO()
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._buf.write(text)

    def flush(self):
        self._stdout.flush()

    def get_text(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
N_TEMPORAL_BINS = 10


def build_flaw_lookup(set_dir: Path) -> dict[str, bool]:
    """Build a clip_name -> is_flaw mapping from the ShuttleSet set CSVs.

    Reads match.csv to map video IDs to folder names, then reads every
    set*.csv to extract the 'flaw' column. Returns a dict keyed by clip
    stem like "35_1_10_17" with True (flaw=1.0) or False (no flaw).

    Clips that were filtered out by the original BST authors won't appear
    in the dataset — they'll just be unused entries in this dict.

    :param set_dir: Path to data/shuttleset/set/ containing match.csv and
                    match folders with set*.csv files.
    :return: Dict mapping clip stem to flaw status.
    """
    # video_id -> match folder name
    match_df = pd.read_csv(set_dir / "match.csv")
    id_to_folder = dict(zip(match_df["id"], match_df["video"]))

    lookup: dict[str, bool] = {}
    for vid_id, folder in id_to_folder.items():
        folder_path = set_dir / folder
        if not folder_path.is_dir():
            continue

        for csv_path in sorted(folder_path.glob("set*.csv")):
            # Extract set number from filename, e.g. "set2.csv" -> 2
            set_num = int(csv_path.stem.removeprefix("set"))

            df = pd.read_csv(csv_path)
            # ball_round is float in the CSVs (e.g. 1.0); cast to int for the key.
            # Build clip stems vectorised: "{vid_id}_{set}_{rally}_{ball_round}"
            stems = (
                str(vid_id) + "_" + str(set_num) + "_"
                + df["rally"].astype(str) + "_"
                + df["ball_round"].astype(int).astype(str)
            )
            is_flaw = df["flaw"].eq(1.0)

            for stem, flaw in zip(stems, is_flaw):
                lookup[stem] = bool(flaw)

    return lookup


def _load_shuttle_vis(
    shuttle_npy_dir: Path, clip_name: str, n_frames: int,
) -> np.ndarray | None:
    """Load shuttle visibility from shuttle NPY and return as a bad-frame mask.

    Shuttle NPYs have shape (t, 3) with [x_norm, y_norm, visibility].
    Visibility is 1=detected, 0=not detected.  We invert so True=bad
    (parallels failed_arr where True = MMPose failed).

    Truncates to n_frames if lengths differ (different video backends can
    produce ±1-2 frame differences).

    :return: Boolean array of shape (n_frames,) or None if file not found.
    """
    shuttle_path = shuttle_npy_dir / f"{clip_name}.npy"
    if not shuttle_path.exists():
        return None
    shuttle_arr = np.load(shuttle_path)  # (t, 3)
    vis = shuttle_arr[:, 2]             # 1=detected, 0=not
    # Truncate to the shorter of the two (pose vs shuttle frame counts).
    min_len = min(len(vis), n_frames)
    return vis[:min_len] == 0  # invert: True = shuttle NOT detected (bad)


def _derive_stroke_player(
    raw_type_en: str, player_side: str, taxonomy: Taxonomy,
) -> tuple[str, str]:
    """Apply the taxonomy's merge_map + side rule to a clips_master row.

    Returns (player, stroke_type) where player is 'Top' / 'Bottom' for sided
    classes or '' for nosides taxonomies and side-agnostic types (e.g.
    'unknown'). Mirrors label_for_row's side decision: a nosides taxonomy
    (has_sides=False, e.g. une_v1_14) never gets a Top_/Bottom_ prefix.
    """
    merged = (taxonomy.merge_map or {}).get(raw_type_en, raw_type_en)
    if taxonomy.has_sides and merged not in SIDE_AGNOSTIC_TYPES:
        return player_side, merged
    return "", merged


def scan_clips(
    dataset_npy_dir: Path,
    clips_csv: Path,
    split_column: str,
    taxonomy: Taxonomy,
    flaw_lookup: dict[str, bool] | None = None,
    shuttle_npy_dir: Path | None = None,
) -> list[ClipRecord]:
    """Iterate clips_master.csv and load every clip's *_failed.npy.

    Split and label come from the CSV (``split_column``, ``raw_type_en``,
    ``player_side``, ``clip_stem``); per-clip files resolve flat at
    ``{dataset_npy_dir}/{clip_stem}_failed.npy`` and
    ``{shuttle_npy_dir}/{clip_stem}.npy``. Clips whose _failed.npy is absent
    (no pose extraction run yet) are silently skipped and counted.

    :param dataset_npy_dir: Flat per-clip npy directory.
    :param clips_csv: Master clips CSV (one row per clip).
    :param split_column: Column in clips_csv giving train/val/test assignment.
    :param taxonomy: Taxonomy for merge_map + side-rule label derivation.
    :param flaw_lookup: Optional dict from build_flaw_lookup(). If provided,
                        each clip's is_flaw field is populated from it.
    :param shuttle_npy_dir: Optional path to flat shuttle NPY dir.
    :return: List of ClipRecord namedtuples, sorted by (split, stroke, clip).
    """
    df = pd.read_csv(clips_csv)
    if split_column not in df.columns:
        raise KeyError(
            f"split_column {split_column!r} not in clips_csv columns: "
            f"{list(df.columns)}"
        )
    df = df[df[split_column].isin(SPLITS)].copy()

    records: list[ClipRecord] = []
    shuttle_miss = 0      # clips where shuttle NPY wasn't found
    missing_files = 0     # clips in CSV with no _failed.npy on disk

    for row in df.itertuples(index=False):
        split = getattr(row, split_column)
        clip_name = row.clip_stem
        player, stroke_type = _derive_stroke_player(
            row.raw_type_en, row.player_side, taxonomy,
        )
        folder_name = f"{player}_{stroke_type}" if player else stroke_type
        rel_path = f"{split}/{folder_name}/{clip_name}"

        fpath = dataset_npy_dir / f"{clip_name}_failed.npy"
        if not fpath.exists():
            missing_files += 1
            continue

        arr = np.load(fpath)
        is_flaw = flaw_lookup.get(clip_name) if flaw_lookup else None

        total = len(arr)
        if total == 0:
            print(f"  WARNING: 0-frame clip: {fpath}")
            records.append(ClipRecord(
                clip_name=clip_name,
                rel_path=rel_path,
                split=split,
                stroke_type=stroke_type,
                player=player,
                total_frames=0,
                failed_frames=0,
                fail_rate=0.0,
                temporal_bins=np.zeros(N_TEMPORAL_BINS),
                is_flaw=is_flaw,
                failed_arr=arr,
            ))
            continue

        failed = int(np.sum(arr))
        fail_rate = failed / total

        # Temporal bins: split the clip into N equal segments and
        # compute the mean failure rate in each.
        bins = np.array_split(arr, N_TEMPORAL_BINS)
        temporal = np.array(
            [float(np.mean(b)) if len(b) > 0 else np.nan for b in bins]
        )

        # Shuttle visibility (optional).
        shuttle_vis = None
        if shuttle_npy_dir:
            shuttle_vis = _load_shuttle_vis(shuttle_npy_dir, clip_name, total)
            if shuttle_vis is None:
                shuttle_miss += 1

        records.append(ClipRecord(
            clip_name=clip_name,
            rel_path=rel_path,
            split=split,
            stroke_type=stroke_type,
            player=player,
            total_frames=total,
            failed_frames=failed,
            fail_rate=fail_rate,
            temporal_bins=temporal,
            is_flaw=is_flaw,
            failed_arr=arr,
            shuttle_vis=shuttle_vis,
        ))

    if missing_files:
        print(f"  WARNING: {missing_files} clips in {clips_csv.name} had no "
              f"_failed.npy under {dataset_npy_dir}; skipped.")
    if shuttle_npy_dir and shuttle_miss:
        print(f"  WARNING: {shuttle_miss} clips had no matching shuttle NPY")

    records.sort(key=lambda r: (r.split, r.stroke_type, r.clip_name))
    return records


# ---------------------------------------------------------------------------
# Text reporters
# ---------------------------------------------------------------------------

def print_overall_stats(
    records: list[ClipRecord], taxonomy: str, data_root: Path,
) -> None:
    total_frames = sum(r.total_frames for r in records)
    total_failed = sum(r.failed_frames for r in records)
    rate = total_failed / total_frames * 100 if total_frames else 0

    print("=" * 65)
    print("  MMPose Detection Failure Analysis")
    print("=" * 65)
    print(f"  Taxonomy:   {taxonomy}")
    print(f"  Data root:  {data_root}")
    print()
    print(f"  Overall: {total_failed:,} / {total_frames:,} frames failed ({rate:.2f}%)")
    print(f"           across {len(records):,} clips")
    print()
    print("  Note: on MMPose-failed frames, joints, court positions, AND shuttle")
    print("  coords are all zeroed. The transformer does NOT mask them in attention.")
    print()


def print_per_split_stats(records: list[ClipRecord]) -> None:
    by_split: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in records:
        by_split[r.split].append(r)

    print("--- Per-split breakdown ---")
    print(f"  {'Split':<8} {'Clips':>7}   {'Total Frames':>14}   "
          f"{'Failed Frames':>14}   {'Fail Rate':>10}")

    for split in SPLITS:
        recs = by_split.get(split, [])
        if not recs:
            continue
        total = sum(r.total_frames for r in recs)
        failed = sum(r.failed_frames for r in recs)
        rate = failed / total * 100 if total else 0
        print(f"  {split:<8} {len(recs):>7,}   {total:>14,}   "
              f"{failed:>14,}   {rate:>9.2f}%")
    print()


def print_per_stroke_stats(records: list[ClipRecord]) -> None:
    by_stroke: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in records:
        by_stroke[r.stroke_type].append(r)

    # Sort by fail rate descending.
    stroke_stats = []
    for stroke, recs in by_stroke.items():
        total = sum(r.total_frames for r in recs)
        failed = sum(r.failed_frames for r in recs)
        rate = failed / total * 100 if total else 0
        stroke_stats.append((stroke, len(recs), failed, total, rate))
    stroke_stats.sort(key=lambda x: x[4], reverse=True)

    print("--- Per-stroke-type fail rates ---")
    print(f"  {'Stroke Type':<24} {'Clips':>6}   {'Failed / Total':>20}   "
          f"{'Fail Rate':>10}")

    for stroke, n_clips, failed, total, rate in stroke_stats:
        ratio = f"{failed:,} / {total:,}"
        print(f"  {stroke:<24} {n_clips:>6}   {ratio:>20}   {rate:>9.2f}%")
    print()


def _is_unknown(r: ClipRecord) -> bool:
    return r.stroke_type == "unknown"


def print_tiered_clip_counts(
    records: list[ClipRecord], threshold: float,
) -> None:
    tiers = [
        ("100% failed (completely blank)", lambda r: r.fail_rate == 1.0),
        ("> 90% failed",                   lambda r: r.fail_rate > 0.9),
        ("> 75% failed",                   lambda r: r.fail_rate > 0.75),
        ("> 50% failed",                   lambda r: r.fail_rate > 0.5),
    ]

    print("--- Clip failure tiers (excl. unknown/) ---")
    for label, pred in tiers:
        real = sum(1 for r in records if pred(r) and not _is_unknown(r))
        unk = sum(1 for r in records if pred(r) and _is_unknown(r))
        unk_note = f"  (+{unk} from unknown/)" if unk else ""
        print(f"  {label + ':':<38} {real:>6} clips{unk_note}")
    print()

    # 100% failed — always list all (excluding unknown).
    blank = [r for r in records if r.fail_rate == 1.0 and not _is_unknown(r)]
    blank_unk = sum(1 for r in records if r.fail_rate == 1.0 and _is_unknown(r))
    unk_note = f"  (+{blank_unk} from unknown/)" if blank_unk else ""
    if blank:
        print(f"--- Completely blank clips (100% failed): "
              f"{len(blank)}{unk_note} ---")
        for r in sorted(blank, key=lambda r: r.rel_path):
            print(f"  {r.rel_path}   ({r.total_frames} frames)")
    else:
        print(f"--- No completely blank clips (excl. unknown/){unk_note} ---")
    print()

    # User threshold — up to 50, sorted by fail rate desc (excluding unknown).
    flagged = sorted(
        [r for r in records if r.fail_rate > threshold and not _is_unknown(r)],
        key=lambda r: -r.fail_rate,
    )
    flagged_unk = sum(
        1 for r in records if r.fail_rate > threshold and _is_unknown(r)
    )
    unk_note = f"  (+{flagged_unk} from unknown/)" if flagged_unk else ""
    print(f"--- Clips above {threshold:.0%} threshold: "
          f"{len(flagged)}{unk_note} ---")
    if flagged:
        show = flagged[:50]
        for r in show:
            print(f"  {r.rel_path:<55} "
                  f"{r.failed_frames}/{r.total_frames} ({r.fail_rate:.1%})")
        if len(flagged) > 50:
            print(f"  ... and {len(flagged) - 50} more (truncated)")
    print()


def print_flaw_vs_fail_stats(
    records: list[ClipRecord], threshold: float,
) -> None:
    """Compare pose failure rates for flaw-annotated vs non-flaw clips.

    Excludes unknown/ clips (garbage class) from the comparison.
    """
    real = [r for r in records if not _is_unknown(r) and r.is_flaw is not None]
    flaw = [r for r in real if r.is_flaw]
    clean = [r for r in real if not r.is_flaw]
    unmatched = sum(1 for r in records if r.is_flaw is None)

    def _stats(recs: list[ClipRecord]) -> tuple[int, int, int, float]:
        n = len(recs)
        total = sum(r.total_frames for r in recs)
        failed = sum(r.failed_frames for r in recs)
        rate = failed / total * 100 if total else 0
        return n, total, failed, rate

    print("--- Flaw annotation vs. pose failure (excl. unknown/) ---")
    print(f"  {'Group':<14} {'Clips':>7}   {'Failed / Total':>22}   "
          f"{'Fail Rate':>10}")

    for label, recs in [("flaw=1.0", flaw), ("non-flaw", clean)]:
        n, total, failed, rate = _stats(recs)
        ratio = f"{failed:,} / {total:,}"
        print(f"  {label:<14} {n:>7,}   {ratio:>22}   {rate:>9.2f}%")

    if unmatched:
        print(f"\n  ({unmatched:,} clips had no match in set CSVs — flaw status unknown)")

    # How many of the high-failure clips are flaw-marked?
    flagged_real = [r for r in real if r.fail_rate > threshold]
    flagged_flaw = sum(1 for r in flagged_real if r.is_flaw)
    if flagged_real:
        print(f"\n  Of {len(flagged_real)} clips above {threshold:.0%} threshold: "
              f"{flagged_flaw} are flaw-marked "
              f"({flagged_flaw / len(flagged_real):.0%})")
    print()


def print_shuttle_failure_stats(records: list[ClipRecord]) -> None:
    """Report shuttle (TrackNet) detection failures and overlap with MMPose.

    Only considers clips where shuttle_vis is available.  Excludes unknown/.
    """
    with_shuttle = [
        r for r in records if not _is_unknown(r) and r.shuttle_vis is not None
    ]
    if not with_shuttle:
        print("--- Shuttle detection analysis: no shuttle data loaded ---\n")
        return

    # --- Overall ---
    total_frames = sum(r.total_frames for r in with_shuttle)
    shuttle_bad = sum(int(np.sum(r.shuttle_vis)) for r in with_shuttle)
    rate = shuttle_bad / total_frames * 100 if total_frames else 0

    print("=" * 65)
    print("  Shuttle (TrackNet) Detection Failure Analysis")
    print("=" * 65)
    print(f"  Overall: {shuttle_bad:,} / {total_frames:,} frames with shuttle "
          f"not detected ({rate:.2f}%)")
    print(f"           across {len(with_shuttle):,} clips with shuttle data")
    print()
    print("  Note: shuttle visibility=0 means TrackNetV3 could not locate the")
    print("  shuttle. This is independent of MMPose — both can fail on the")
    print("  same frame or on different frames.")
    print()

    # --- Per-split ---
    by_split: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in with_shuttle:
        by_split[r.split].append(r)

    print("--- Per-split shuttle detection ---")
    print(f"  {'Split':<8} {'Clips':>7}   {'Total Frames':>14}   "
          f"{'Shuttle Missed':>14}   {'Miss Rate':>10}")
    for split in SPLITS:
        recs = by_split.get(split, [])
        if not recs:
            continue
        tot = sum(r.total_frames for r in recs)
        bad = sum(int(np.sum(r.shuttle_vis)) for r in recs)
        r_pct = bad / tot * 100 if tot else 0
        print(f"  {split:<8} {len(recs):>7,}   {tot:>14,}   "
              f"{bad:>14,}   {r_pct:>9.2f}%")
    print()

    # --- 2×2 overlap: MMPose failed × shuttle not detected ---
    # For each frame, classify into one of four buckets.
    both_bad = 0
    pose_only = 0
    shuttle_only = 0
    both_ok = 0
    for r in with_shuttle:
        # Truncate to the shorter length (shuttle_vis is already truncated
        # to total_frames in _load_shuttle_vis, but failed_arr might be
        # longer if they differ).
        n = min(len(r.failed_arr), len(r.shuttle_vis))
        pose_fail = r.failed_arr[:n]
        shut_fail = r.shuttle_vis[:n]
        both_bad += int(np.sum(pose_fail & shut_fail))
        pose_only += int(np.sum(pose_fail & ~shut_fail))
        shuttle_only += int(np.sum(~pose_fail & shut_fail))
        both_ok += int(np.sum(~pose_fail & ~shut_fail))

    grand = both_bad + pose_only + shuttle_only + both_ok

    print("--- MMPose × Shuttle failure overlap (excl. unknown/) ---")
    print(f"  {'':28} {'Shuttle OK':>14}  {'Shuttle BAD':>14}  {'Row Total':>14}")
    pose_bad_total = both_bad + pose_only
    pose_ok_total = shuttle_only + both_ok
    print(f"  {'MMPose FAILED':<28} "
          f"{pose_only:>14,}  {both_bad:>14,}  {pose_bad_total:>14,}")
    print(f"  {'MMPose OK':<28} "
          f"{both_ok:>14,}  {shuttle_only:>14,}  {pose_ok_total:>14,}")
    col_shut_ok = pose_only + both_ok
    col_shut_bad = both_bad + shuttle_only
    print(f"  {'Column Total':<28} "
          f"{col_shut_ok:>14,}  {col_shut_bad:>14,}  {grand:>14,}")

    if grand:
        print(f"\n  Both OK:          {both_ok:>10,}  ({both_ok / grand * 100:.2f}%)")
        print(f"  MMPose only fail: {pose_only:>10,}  ({pose_only / grand * 100:.2f}%)")
        print(f"  Shuttle only fail:{shuttle_only:>10,}  ({shuttle_only / grand * 100:.2f}%)")
        print(f"  Both fail:        {both_bad:>10,}  ({both_bad / grand * 100:.2f}%)")
    print()


def _hit_zone_slice(arr: np.ndarray, hit_idx: int, window: int):
    """Return (near_slice, lo, hi) for the hit zone, clamped to clip bounds."""
    lo = max(0, hit_idx - window)
    hi = min(len(arr), hit_idx + window + 1)  # +1 for inclusive range
    return arr[lo:hi], lo, hi


def _print_class_split_table(
    data: dict[tuple[str, str], list[float]],
    threshold: float,
    title: str,
    bad_label: str = "Bad",
) -> None:
    """Print a class × split table of clips exceeding *threshold* in the hit zone.

    :param data: {(stroke_type, split): [per_clip_rate, ...]}
    :param threshold: Clip-level rate cutoff (e.g. 0.5).
    :param title: Section header.
    :param bad_label: Column label prefix for the bad-clip count (e.g. "Fail", "Miss").
    """
    if not data:
        return
    rows = []
    for (stroke, split), rates in data.items():
        n_bad = sum(1 for r in rates if r > threshold)
        pct = n_bad / len(rates) * 100 if rates else 0
        rows.append((stroke, split, len(rates), n_bad, pct))
    rows.sort(key=lambda x: (-x[4], x[0], x[1]))

    bad_col = f">{threshold:.0%} {bad_label}"
    print(f"--- {title} ---")
    print(f"  {'Stroke Type':<24} {'Split':<7} {'Clips':>6}   "
          f"{bad_col:>12}   {'%':>7}")
    for stroke, split, total, n_bad, pct in rows:
        print(f"  {stroke:<24} {split:<7} {total:>6}   "
              f"{n_bad:>12}   {pct:>6.1f}%")
    print()


def print_hit_proximity_stats(
    records: list[ClipRecord],
    hit_lookup: dict[str, int],
    hit_window: int,
    threshold: float = 0.5,
) -> dict[str, dict[tuple[str, str], list[float]]]:
    """Compare fail rates near the hit frame vs. away from it.

    For each clip, the "hit zone" is [hit_index - window, hit_index + window],
    clamped to clip bounds.  Excludes unknown/ clips.

    When shuttle_vis data is available on records, also reports shuttle
    detection failures within the hit zone and a combined data-quality metric.

    :param records: All clip records (with failed_arr populated).
    :param hit_lookup: From build_hit_frame_lookup(); clip_stem -> frame index.
    :param hit_window: Number of frames either side of the hit to include.
    :param threshold: Clip-level fail-rate cutoff for class × split tables.
    :return: Dict with keys "mmpose", "shuttle", "either" mapping to
             {(stroke_type, split): [per_clip_rate, ...]} for use by plots.
    """
    # --- MMPose accumulators ---
    near_failed = 0
    near_total = 0
    away_failed = 0
    away_total = 0
    matched = 0
    skipped = 0
    hit_zone_rates: list[tuple[float, str]] = []  # (rate, rel_path)
    stroke_near: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    # --- Shuttle accumulators (only filled when shuttle_vis is present) ---
    has_shuttle = False
    shut_near_bad = 0
    shut_near_total = 0
    shut_away_bad = 0
    shut_away_total = 0
    # Combined "good data" = MMPose OK AND shuttle detected in the hit zone.
    combined_good = 0
    combined_total = 0
    shut_hz_rates: list[tuple[float, str]] = []  # (rate, rel_path)
    stroke_shut_near: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    # Per-clip hit-zone rates keyed by (stroke_type, split) for class×split tables.
    clip_hz: dict[tuple[str, str], list[float]] = defaultdict(list)
    clip_shut_hz: dict[tuple[str, str], list[float]] = defaultdict(list)
    clip_either_hz: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in records:
        if _is_unknown(r) or r.total_frames == 0:
            continue
        hit_idx = hit_lookup.get(r.clip_name)
        if hit_idx is None:
            skipped += 1
            continue

        near_slice, lo, hi = _hit_zone_slice(r.failed_arr, hit_idx, hit_window)
        if len(near_slice) == 0:
            skipped += 1
            continue
        matched += 1
        away_slice = np.concatenate([r.failed_arr[:lo], r.failed_arr[hi:]])

        n_near_fail = int(np.sum(near_slice))
        near_failed += n_near_fail
        near_total += len(near_slice)
        hz_rate = n_near_fail / len(near_slice) if len(near_slice) else 0
        hit_zone_rates.append((hz_rate, r.rel_path))
        clip_hz[(r.stroke_type, r.split)].append(hz_rate)

        if len(away_slice) > 0:
            away_failed += int(np.sum(away_slice))
            away_total += len(away_slice)

        stroke_near[r.stroke_type][0] += n_near_fail
        stroke_near[r.stroke_type][1] += len(near_slice)

        # Shuttle within hit zone (when available).
        # Truncate to common length first — shuttle NPY may be 1-2 frames
        # shorter than failed_arr (different video backends).
        if r.shuttle_vis is not None:
            has_shuttle = True
            n_common = min(len(r.failed_arr), len(r.shuttle_vis))
            shut_arr = r.shuttle_vis[:n_common]
            pose_arr = r.failed_arr[:n_common]

            s_near, s_lo, s_hi = _hit_zone_slice(
                shut_arr, hit_idx, hit_window,
            )
            s_away = np.concatenate([shut_arr[:s_lo], shut_arr[s_hi:]])

            n_s_bad = int(np.sum(s_near))
            shut_near_bad += n_s_bad
            shut_near_total += len(s_near)
            s_hz_rate = n_s_bad / len(s_near) if len(s_near) else 0
            shut_hz_rates.append((s_hz_rate, r.rel_path))
            clip_shut_hz[(r.stroke_type, r.split)].append(s_hz_rate)

            if len(s_away) > 0:
                shut_away_bad += int(np.sum(s_away))
                shut_away_total += len(s_away)

            stroke_shut_near[r.stroke_type][0] += n_s_bad
            stroke_shut_near[r.stroke_type][1] += len(s_near)

            # Combined: frames in hit zone where both MMPose OK and shuttle OK.
            # Use same lo:hi bounds on the truncated pose array so shapes match.
            p_near = pose_arr[s_lo:s_hi]
            both_good = int(np.sum(~p_near & ~s_near))
            combined_good += both_good
            combined_total += len(s_near)

            either_bad = int(np.sum(p_near | s_near))
            either_rate = either_bad / len(s_near) if len(s_near) else 0
            clip_either_hz[(r.stroke_type, r.split)].append(either_rate)

    near_rate = near_failed / near_total * 100 if near_total else 0
    away_rate = away_failed / away_total * 100 if away_total else 0

    # --- MMPose: aggregate near vs away ---
    print(f"--- MMPose failure near hit frames (\u00b1{hit_window} frames, excl. unknown/) ---")
    print(f"  {'Zone':<18} {'Clips':>7}   {'Failed / Total':>22}   "
          f"{'Fail Rate':>10}")

    near_ratio = f"{near_failed:,} / {near_total:,}"
    away_ratio = f"{away_failed:,} / {away_total:,}"
    print(f"  {'near hit':<18} {matched:>7,}   {near_ratio:>22}   "
          f"{near_rate:>9.2f}%")
    print(f"  {'away from hit':<18} {matched:>7,}   {away_ratio:>22}   "
          f"{away_rate:>9.2f}%")

    # --- Tiered hit-zone clip counts (MMPose) ---
    tiers = [
        ("100% of hit zone zeroed", lambda r: r == 1.0),
        ("> 90% of hit zone zeroed", lambda r: r > 0.9),
        ("> 75% of hit zone zeroed", lambda r: r > 0.75),
        ("> 50% of hit zone zeroed", lambda r: r > 0.5),
        ("any zeroed frame",         lambda r: r > 0),
    ]
    print(f"\n  MMPose hit-zone clip tiers (of {matched:,} matched clips):")
    for label, pred in tiers:
        count = sum(1 for rate, _ in hit_zone_rates if pred(rate))
        pct = count / matched * 100 if matched else 0
        print(f"    {label + ':':<30} {count:>6} ({pct:.1f}%)")

    # List clips with 100% zeroed hit zone.
    fully_zeroed = sorted(
        [path for rate, path in hit_zone_rates if rate == 1.0]
    )
    if fully_zeroed:
        print(f"\n  Clips with 100% MMPose-zeroed hit zone ({len(fully_zeroed)}):")
        for path in fully_zeroed:
            print(f"    {path}")

    if skipped:
        print(f"\n  ({skipped:,} clips had no hit-frame lookup — skipped)")
    print()

    # --- Per-stroke hit-zone fail rates (MMPose) ---
    stroke_stats = []
    for stroke, (failed, total) in stroke_near.items():
        rate = failed / total * 100 if total else 0
        stroke_stats.append((stroke, failed, total, rate))
    stroke_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"--- Per-stroke MMPose hit-zone fail rates (\u00b1{hit_window} frames, excl. unknown/) ---")
    print(f"  {'Stroke Type':<24} {'Failed / Total':>20}   {'Fail Rate':>10}")
    for stroke, failed, total, rate in stroke_stats:
        ratio = f"{failed:,} / {total:,}"
        print(f"  {stroke:<24} {ratio:>20}   {rate:>9.2f}%")
    print()

    # --- MMPose hit-zone quality by class × split ---
    _print_class_split_table(
        clip_hz, threshold,
        f"MMPose hit-zone clip quality by class \u00d7 split "
        f"(>{threshold:.0%} fail rate, excl. unknown/)",
        "Fail",
    )

    # ===================================================================
    # Shuttle detection within hit zone (only if shuttle data was loaded)
    # ===================================================================
    if not has_shuttle:
        return {"mmpose": dict(clip_hz)}

    shut_near_rate = shut_near_bad / shut_near_total * 100 if shut_near_total else 0
    shut_away_rate = shut_away_bad / shut_away_total * 100 if shut_away_total else 0
    n_shut = len(shut_hz_rates)

    print(f"--- Shuttle miss rate within \u00b1{hit_window} frames of hit (excl. unknown/) ---")
    print(f"  {'Zone':<18} {'Clips':>7}   {'Missed / Total':>22}   "
          f"{'Miss Rate':>10}")

    s_near_ratio = f"{shut_near_bad:,} / {shut_near_total:,}"
    s_away_ratio = f"{shut_away_bad:,} / {shut_away_total:,}"
    print(f"  {'near hit':<18} {n_shut:>7,}   {s_near_ratio:>22}   "
          f"{shut_near_rate:>9.2f}%")
    print(f"  {'away from hit':<18} {n_shut:>7,}   {s_away_ratio:>22}   "
          f"{shut_away_rate:>9.2f}%")

    # Combined data quality in hit zone.
    if combined_total:
        quality = combined_good / combined_total * 100
        print(f"\n  Combined data quality in hit zone:")
        print(f"    Frames with BOTH MMPose OK and shuttle detected: "
              f"{combined_good:,} / {combined_total:,} ({quality:.2f}%)")

    # --- Tiered hit-zone counts (shuttle) ---
    shut_tiers = [
        ("100% shuttle missed",  lambda r: r == 1.0),
        ("> 90% shuttle missed", lambda r: r > 0.9),
        ("> 75% shuttle missed", lambda r: r > 0.75),
        ("> 50% shuttle missed", lambda r: r > 0.5),
        ("any missed frame",     lambda r: r > 0),
    ]
    print(f"\n  Shuttle hit-zone clip tiers (of {n_shut:,} clips with shuttle data):")
    for label, pred in shut_tiers:
        count = sum(1 for rate, _ in shut_hz_rates if pred(rate))
        pct = count / n_shut * 100 if n_shut else 0
        print(f"    {label + ':':<30} {count:>6} ({pct:.1f}%)")
    print()

    # --- Per-stroke shuttle hit-zone miss rates ---
    shut_stroke_stats = []
    for stroke, (bad, total) in stroke_shut_near.items():
        rate = bad / total * 100 if total else 0
        shut_stroke_stats.append((stroke, bad, total, rate))
    shut_stroke_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"--- Per-stroke shuttle miss rate within \u00b1{hit_window} frames of hit (excl. unknown/) ---")
    print(f"  {'Stroke Type':<24} {'Missed / Total':>20}   {'Miss Rate':>10}")
    for stroke, bad, total, rate in shut_stroke_stats:
        ratio = f"{bad:,} / {total:,}"
        print(f"  {stroke:<24} {ratio:>20}   {rate:>9.2f}%")
    print()

    # --- Shuttle hit-zone quality by class × split ---
    _print_class_split_table(
        clip_shut_hz, threshold,
        f"Shuttle hit-zone clip quality by class \u00d7 split "
        f"(>{threshold:.0%} miss rate, excl. unknown/)",
        "Miss",
    )

    # --- Combined (either-bad) hit-zone quality by class × split ---
    _print_class_split_table(
        clip_either_hz, threshold,
        f"Combined (MMPose OR shuttle) hit-zone clip quality by class \u00d7 split "
        f"(>{threshold:.0%} either-bad rate, excl. unknown/)",
        "Bad",
    )

    return {
        "mmpose": dict(clip_hz),
        "shuttle": dict(clip_shut_hz),
        "either": dict(clip_either_hz),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fail_rate_histogram(
    records: list[ClipRecord], output_path: Path,
) -> None:
    """Histogram of per-clip fail rates (unknown/ clips excluded)."""
    filtered = [r for r in records if not _is_unknown(r)]
    rates = np.array([r.fail_rate for r in filtered])
    mean_rate = float(np.mean(rates))
    median_rate = float(np.median(rates))

    n_excluded = len(records) - len(filtered)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bin_edges, patches = ax.hist(
        rates, bins=20, range=(0.0, 1.0), edgecolor="black", alpha=0.75,
    )
    ax.set_yscale("log")

    # Annotate the first bar with its raw count — on a log scale the
    # dominant bin's height is visually compressed and misleading.
    if counts[0] > 0:
        ax.text(
            (bin_edges[0] + bin_edges[1]) / 2, counts[0],
            f"n={int(counts[0]):,}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.axvline(mean_rate, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_rate:.3f}")
    ax.set_ylabel("Number of Clips (log scale)")
    ax.set_title("Per-clip Zeroed-Frame Rate Distribution")

    # Stats annotation.
    stats_text = (
        f"Clips: {len(filtered):,}\n"
        f"Mean:   {mean_rate:.4f}\n"
        f"Median: {median_rate:.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.legend()
    ax.set_xlabel(
        f"Clip Fail Rate\n(excludes {n_excluded:,} unknown/ clips)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_temporal_pattern(
    records: list[ClipRecord], output_path: Path,
) -> None:
    """Line chart: mean fail rate by normalized clip position (unknown/ excluded)."""
    filtered = [r for r in records if not _is_unknown(r)]
    n_excluded = len(records) - len(filtered)

    by_split: dict[str, list[np.ndarray]] = defaultdict(list)
    all_bins = []

    for r in filtered:
        all_bins.append(r.temporal_bins)
        by_split[r.split].append(r.temporal_bins)

    # Stack and average, ignoring NaN from single-frame edge cases.
    overall = np.nanmean(np.stack(all_bins), axis=0)
    overall_mean = float(np.nanmean(overall))

    x = np.linspace(0.05, 0.95, N_TEMPORAL_BINS)  # bin centres
    x_labels = [f"{i / N_TEMPORAL_BINS:.0%}-{(i + 1) / N_TEMPORAL_BINS:.0%}"
                for i in range(N_TEMPORAL_BINS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, overall, color="black", linewidth=2.5, label="All", zorder=3)

    # Colourblind-friendly palette (cyan / green / red).
    split_colors = {"train": "#88CCEE", "val": "#228833", "test": "#800A01"}
    for split in SPLITS:
        if split not in by_split:
            continue
        avg = np.nanmean(np.stack(by_split[split]), axis=0)
        ax.plot(x, avg, color=split_colors[split], linewidth=1.2, label=split)

    ax.axhline(overall_mean, color="grey", linestyle="--", linewidth=1,
               label=f"Overall mean: {overall_mean:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(
        f"Normalised Clip Position\n(excludes {n_excluded:,} unknown/ clips)",
        fontsize=10,
    )
    ax.set_ylabel("Mean Fail Rate")
    ax.set_title("Temporal Pattern of Zeroed Frames")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hit_frame_profile(
    records: list[ClipRecord],
    hit_lookup: dict[str, int],
    hit_window: int,
    overall_fail_rate: float,
    output_path: Path,
) -> list[tuple[str, str, int, int]]:
    """Line chart: mean fail rate by frame offset from the hit.

    For each clip, extracts a fixed-width window around the hit frame,
    padding with NaN where the window extends past clip bounds so edge
    clips don't bias the average.  Excludes unknown/ clips.

    If shuttle_vis data is available, overlays a shuttle miss-rate line
    so both failure modes can be compared at the same frame offsets.

    :param overall_fail_rate: Overall fail rate (0-1) from general stats,
                              drawn as a reference line.
    :return: List of (clip_name, rel_path, hit_idx, total_frames) for clips
             skipped because the hit index was past the end of the clip.
    """
    width = 2 * hit_window + 1  # total frames in the window
    by_split: dict[str, list[np.ndarray]] = defaultdict(list)
    all_slices: list[np.ndarray] = []
    shuttle_slices: list[np.ndarray] = []  # shuttle miss-rate overlay
    n_excluded = 0
    oob_clips: list[tuple[str, str, int, int]] = []  # out-of-bounds clips

    for r in records:
        if _is_unknown(r):
            n_excluded += 1
            continue
        if r.total_frames == 0:
            continue
        hit_idx = hit_lookup.get(r.clip_name)
        if hit_idx is None:
            continue

        # Build a fixed-width slice centred on the hit, NaN-padded at edges.
        # Skip if the hit index is past the end of the clip (lookup/clip
        # length mismatch from different video backends).
        src_lo = max(0, hit_idx - hit_window)
        src_hi = min(len(r.failed_arr), hit_idx + hit_window + 1)
        if src_lo >= src_hi:
            oob_clips.append((r.clip_name, r.rel_path, hit_idx, r.total_frames))
            continue

        row = np.full(width, np.nan)
        dst_lo = src_lo - (hit_idx - hit_window)  # offset into row
        dst_hi = dst_lo + (src_hi - src_lo)
        row[dst_lo:dst_hi] = r.failed_arr[src_lo:src_hi].astype(float)

        all_slices.append(row)
        by_split[r.split].append(row)

        # Shuttle overlay (same NaN-padding approach).
        if r.shuttle_vis is not None:
            s_src_lo = max(0, hit_idx - hit_window)
            s_src_hi = min(len(r.shuttle_vis), hit_idx + hit_window + 1)
            if s_src_lo < s_src_hi:
                s_row = np.full(width, np.nan)
                s_dst_lo = s_src_lo - (hit_idx - hit_window)
                s_dst_hi = s_dst_lo + (s_src_hi - s_src_lo)
                s_row[s_dst_lo:s_dst_hi] = r.shuttle_vis[s_src_lo:s_src_hi].astype(float)
                shuttle_slices.append(s_row)

    if not all_slices:
        return oob_clips

    overall = np.nanmean(np.stack(all_slices), axis=0)
    x = np.arange(-hit_window, hit_window + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, overall, color="black", linewidth=2.5, label="MMPose fail (all)", zorder=3)

    split_colors = {"train": "#88CCEE", "val": "#228833", "test": "#800A01"}
    for split in SPLITS:
        if split not in by_split:
            continue
        avg = np.nanmean(np.stack(by_split[split]), axis=0)
        ax.plot(x, avg, color=split_colors[split], linewidth=1.2, label=f"MMPose {split}")

    # Shuttle overlay — dashed orange line.
    if shuttle_slices:
        shuttle_overall = np.nanmean(np.stack(shuttle_slices), axis=0)
        ax.plot(x, shuttle_overall, color="#EE7733", linewidth=2.5,
                linestyle="--", label="Shuttle miss (all)", zorder=3)

    ax.axhline(overall_fail_rate, color="grey", linestyle="--", linewidth=1,
               label=f"Overall MMPose mean: {overall_fail_rate:.4f}")
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xlabel(
        f"Frame Offset from Hit\n(excludes {n_excluded:,} unknown/ clips)",
        fontsize=10,
    )
    ax.set_ylabel("Mean Fail Rate")
    title = f"Fail Rate by Proximity to Hit Frame (\u00b1{hit_window})"
    if shuttle_slices:
        title += "  — MMPose vs Shuttle"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return oob_clips


def plot_hit_zone_heatmap(
    clip_hz_data: dict[tuple[str, str], list[float]],
    threshold: float,
    output_path: Path,
    title: str = "Hit-zone clip failure rate",
) -> None:
    """Heatmap of % clips exceeding *threshold* in the hit zone, by class × split.

    Rows = stroke types (sorted worst-first by mean across splits),
    columns = splits (train / val / test).

    :param clip_hz_data: {(stroke_type, split): [per_clip_rate, ...]}
    :param threshold: Clip-level rate cutoff.
    :param output_path: Where to save the PNG.
    :param title: Figure title.
    """
    if not clip_hz_data:
        return

    # Collect all stroke types and compute % bad per (stroke, split).
    strokes: set[str] = set()
    pct: dict[tuple[str, str], float] = {}
    for (stroke, split), rates in clip_hz_data.items():
        strokes.add(stroke)
        n_bad = sum(1 for r in rates if r > threshold)
        pct[(stroke, split)] = n_bad / len(rates) * 100 if rates else 0

    # Sort strokes by mean % across splits (worst first).
    stroke_list = sorted(
        strokes,
        key=lambda s: np.mean([pct.get((s, sp), 0) for sp in SPLITS]),
        reverse=True,
    )
    splits_present = [sp for sp in SPLITS if any(
        (s, sp) in pct for s in stroke_list
    )]

    # Build matrix.
    matrix = np.zeros((len(stroke_list), len(splits_present)))
    for i, stroke in enumerate(stroke_list):
        for j, split in enumerate(splits_present):
            matrix[i, j] = pct.get((stroke, split), 0)

    fig, ax = plt.subplots(figsize=(4 + 0.3 * len(stroke_list), max(6, 0.4 * len(stroke_list))))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(splits_present)))
    ax.set_xticklabels(splits_present)
    ax.set_yticks(range(len(stroke_list)))
    ax.set_yticklabels(stroke_list, fontsize=8)

    # Annotate cells with percentage.
    for i in range(len(stroke_list)):
        for j in range(len(splits_present)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label=f"% clips >{threshold:.0%} fail", shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Split")
    ax.set_ylabel("Stroke Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_surviving_clips(
    clip_hz_data: dict[tuple[str, str], list[float]],
    threshold: float,
    output_path: Path,
    title: str = "Surviving clips after quality filter",
) -> None:
    """Grouped bar chart of per-class clip counts remaining after filtering.

    For each class, shows total clips and surviving clips (those with
    hit-zone fail rate <= threshold) per split, side by side.

    :param clip_hz_data: {(stroke_type, split): [per_clip_rate, ...]}
    :param threshold: Clip-level rate cutoff.
    :param output_path: Where to save the PNG.
    :param title: Figure title.
    """
    if not clip_hz_data:
        return

    strokes: set[str] = set()
    for (stroke, _) in clip_hz_data:
        strokes.add(stroke)

    splits_present = [sp for sp in SPLITS if any(
        (s, sp) in clip_hz_data for s in strokes
    )]
    # Sort strokes alphabetically for readability.
    stroke_list = sorted(strokes)

    n_splits = len(splits_present)
    n_strokes = len(stroke_list)
    bar_width = 0.35
    x = np.arange(n_strokes)

    split_colors = {"train": "#88CCEE", "val": "#228833", "test": "#800A01"}

    fig, axes = plt.subplots(
        n_splits, 1,
        figsize=(max(8, 0.6 * n_strokes), 3 * n_splits),
        sharex=True,
    )
    if n_splits == 1:
        axes = [axes]

    for ax, split in zip(axes, splits_present):
        totals = []
        survivors = []
        for stroke in stroke_list:
            rates = clip_hz_data.get((stroke, split), [])
            n_total = len(rates)
            n_survive = sum(1 for r in rates if r <= threshold)
            totals.append(n_total)
            survivors.append(n_survive)

        ax.bar(x - bar_width / 2, totals, bar_width,
               label="Total", color=split_colors.get(split, "grey"), alpha=0.4)
        ax.bar(x + bar_width / 2, survivors, bar_width,
               label=f"Survive (\u2264{threshold:.0%})", color=split_colors.get(split, "grey"))

        # Annotate bars with counts.
        for i, (t, s) in enumerate(zip(totals, survivors)):
            if t > 0:
                ax.text(i - bar_width / 2, t + 0.3, str(t),
                        ha="center", va="bottom", fontsize=7, alpha=0.6)
                ax.text(i + bar_width / 2, s + 0.3, str(s),
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_ylabel(f"{split} clips")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(stroke_list, rotation=45, ha="right", fontsize=8)
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze MMPose and shuttle detection failures across "
                    "the dataset.",
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to ShuttleSet_data_{taxonomy} directory (holds the flat "
             "per-clip npy subdir).",
    )
    parser.add_argument(
        "--dataset-npy-dir", type=Path, default=None,
        help="Flat per-clip npy directory. Defaults to auto-discovery under "
             "--data-root (picks the first *_flat/ subdir that isn't collated).",
    )
    parser.add_argument(
        "--clips-csv", type=Path,
        default=Path(__file__).resolve().parents[3] / "notebooks" / "clips_master.csv",
        help="Master clips CSV. Defaults to <repo>/notebooks/clips_master.csv.",
    )
    parser.add_argument(
        "--split-column", type=str, default="split_bst_baseline",
        help="Column in clips_csv giving train/val/test assignment "
             "(default: split_bst_baseline).",
    )
    parser.add_argument(
        "--taxonomy", type=str, default="une_v1_14",
        choices=list(TAXONOMIES.keys()),
        help="Taxonomy for label derivation, filenames, and display headers "
             "(default: une_v1_14).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fail-rate threshold for the flagged-clips list (default: 0.5)",
    )
    parser.add_argument(
        "--set-dir", type=Path, default=None,
        help="Path to ShuttleSet set/ directory (enables flaw + hit-frame "
             "analysis). If omitted, checks <repo>/data/shuttleset/set "
             "for a match.csv and uses it when found.",
    )
    parser.add_argument(
        "--hit-window", type=int, default=10,
        help="Frames either side of the hit to check (default: 10). "
             "Requires --set-dir.",
    )
    parser.add_argument(
        "--shuttle-npy-dir", type=Path, default=None,
        help="Path to the flat shuttle npy directory. Enables shuttle "
             "detection failure analysis using TrackNet visibility column.",
    )
    args = parser.parse_args()

    # Auto-detect --set-dir from the repo-relative location if not passed.
    # Explicit --set-dir always wins; this only fills a None.
    # parents[3] from src/bst_x/validation_scripts/ = repo root.
    if args.set_dir is None:
        repo_set_dir = Path(__file__).resolve().parents[3] / "data" / "shuttleset" / "set"
        if (repo_set_dir / "match.csv").is_file():
            args.set_dir = repo_set_dir
            print(f"Auto-detected --set-dir: {args.set_dir}")

    taxonomy = resolve_taxonomy(args.taxonomy)

    # Resolve the flat per-clip npy directory. Prefer explicit --dataset-npy-dir,
    # else auto-discover under --data-root: first *_flat/ subdir that isn't the
    # collated output.
    if args.dataset_npy_dir is not None:
        dataset_npy_dir = args.dataset_npy_dir
    else:
        candidates = [
            d for d in sorted(args.data_root.iterdir())
            if d.is_dir()
            and "npy" in d.name
            and "collated" not in d.name
            and d.name.endswith("_flat")
        ]
        if not candidates:
            print(f"ERROR: no flat per-clip npy dir (*_flat/) found under "
                  f"{args.data_root}.")
            print("Contents:", [d.name for d in args.data_root.iterdir() if d.is_dir()])
            print("Pass --dataset-npy-dir to override.")
            sys.exit(1)
        if len(candidates) > 1:
            print(f"ERROR: multiple *_flat dirs found under {args.data_root}:")
            for c in candidates:
                print(f"  {c.name}")
            print("Pass --dataset-npy-dir to pick one explicitly.")
            sys.exit(1)
        dataset_npy_dir = candidates[0]

    if not args.clips_csv.exists():
        print(f"ERROR: clips_csv not found: {args.clips_csv}")
        sys.exit(1)

    # Sydney timestamp for output filenames.
    syd_now = datetime.now(ZoneInfo("Australia/Sydney"))
    ts = syd_now.strftime("%Y%m%d_%H%M")
    tax_short = args.taxonomy.replace("_", "")  # "une_v1_14" -> "unev114"
    split_short = args.split_column.replace("split_", "").replace("_", "")

    # Output dir is always a sibling folder to this script.
    output_dir = Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout so we can save a .txt copy.
    txt_path = output_dir / f"analysis_{tax_short}_{split_short}_{ts}.txt"
    oob_clips: list[tuple[str, str, int, int]] = []
    hz_data: dict[str, dict[tuple[str, str], list[float]]] = {}
    tee = _Tee()
    sys.stdout = tee

    try:
        # --- Flaw lookup (optional) ---
        flaw_lookup = None
        if args.set_dir:
            if not (args.set_dir / "match.csv").is_file():
                print(f"WARNING: match.csv not found in {args.set_dir}, "
                      f"skipping flaw analysis")
            else:
                print(f"Building flaw lookup from {args.set_dir} ...")
                flaw_lookup = build_flaw_lookup(args.set_dir)
                n_flaw = sum(flaw_lookup.values())
                print(f"  {len(flaw_lookup):,} shots loaded, "
                      f"{n_flaw:,} marked flaw=1.0\n")

        # --- Shuttle NPY validation (optional) ---
        shuttle_npy_dir = args.shuttle_npy_dir
        if shuttle_npy_dir and not shuttle_npy_dir.is_dir():
            print(f"WARNING: shuttle_npy_dir not found: {shuttle_npy_dir}, "
                  f"skipping shuttle analysis")
            shuttle_npy_dir = None

        # --- Scan ---
        print(f"Scanning {dataset_npy_dir} ...")
        print(f"  clips_csv: {args.clips_csv}")
        print(f"  split_column: {args.split_column}   taxonomy: {taxonomy.name}")
        if shuttle_npy_dir:
            print(f"  + loading shuttle visibility from {shuttle_npy_dir}")
        print()
        records = scan_clips(
            dataset_npy_dir,
            clips_csv=args.clips_csv,
            split_column=args.split_column,
            taxonomy=taxonomy,
            flaw_lookup=flaw_lookup,
            shuttle_npy_dir=shuttle_npy_dir,
        )

        if not records:
            print("ERROR: No *_failed.npy files found.")
            print(f"Checked: {dataset_npy_dir}")
            sys.exit(1)

        # --- Text reports ---
        print_overall_stats(records, args.taxonomy, args.data_root)
        print_per_split_stats(records)
        print_per_stroke_stats(records)
        print_tiered_clip_counts(records, args.threshold)

        if flaw_lookup:
            print_flaw_vs_fail_stats(records, args.threshold)
        else:
            print("--- Flaw analysis: skipped (pass --set-dir to enable) ---\n")

        # --- Shuttle detection failures (optional) ---
        if shuttle_npy_dir:
            print_shuttle_failure_stats(records)
        else:
            print("--- Shuttle detection analysis: skipped "
                  "(pass --shuttle-npy-dir to enable) ---\n")

        # --- Hit-frame proximity (requires --set-dir + video_metadata.csv) ---
        hit_lookup = None
        metadata_csv = args.set_dir.parent / "video_metadata.csv" if args.set_dir else None
        if args.set_dir and (args.set_dir / "match.csv").is_file():
            if metadata_csv and metadata_csv.is_file():
                from hit_frame_lookup import build_hit_frame_lookup

                print(f"Building hit-frame lookup from {args.set_dir} ...")
                print(f"  FPS source: {metadata_csv}")
                hit_lookup = build_hit_frame_lookup(args.set_dir, metadata_csv)
                print(f"  {len(hit_lookup):,} clip hit-frame indices computed\n")
                hz_data = print_hit_proximity_stats(
                    records, hit_lookup, args.hit_window, args.threshold,
                )
            else:
                print(f"WARNING: video_metadata.csv not found at {metadata_csv}, "
                      f"skipping hit-frame analysis\n")
        else:
            print("--- Hit-frame analysis: skipped "
                  "(pass --set-dir to enable) ---\n")

        # --- Figures ---
        hist_path = output_dir / f"fail_rate_histogram_{tax_short}_{split_short}_{ts}.png"
        temp_path = output_dir / f"temporal_pattern_{tax_short}_{split_short}_{ts}.png"

        plot_fail_rate_histogram(records, hist_path)
        print(f"Saved: {hist_path}")

        plot_temporal_pattern(records, temp_path)
        print(f"Saved: {temp_path}")

        if hit_lookup:
            profile_path = (
                output_dir / f"hit_frame_profile_{tax_short}_{split_short}_{ts}.png"
            )
            # Overall fail rate for the reference line (excl. unknown).
            real = [r for r in records if not _is_unknown(r)]
            total_f = sum(r.total_frames for r in real)
            overall_rate = sum(r.failed_frames for r in real) / total_f if total_f else 0
            oob_clips = plot_hit_frame_profile(
                records, hit_lookup, args.hit_window, overall_rate, profile_path,
            ) or []
            print(f"Saved: {profile_path}")

            if oob_clips:
                print(f"\n  {len(oob_clips)} clips skipped: hit index past "
                      f"end of clip (FPS/frame-count mismatch)")

        # --- Hit-zone heatmap and surviving-clips chart ---
        if hz_data:
            # Pick the most informative metric: combined if shuttle data
            # was loaded, otherwise MMPose-only.
            if "either" in hz_data and hz_data["either"]:
                hm_data = hz_data["either"]
                hm_label = "Combined (MMPose OR shuttle)"
            else:
                hm_data = hz_data["mmpose"]
                hm_label = "MMPose"

            hm_path = output_dir / f"hit_zone_heatmap_{tax_short}_{split_short}_{ts}.png"
            plot_hit_zone_heatmap(
                hm_data, args.threshold, hm_path,
                title=f"{hm_label} — % clips >{args.threshold:.0%} "
                      f"fail in hit zone (excl. unknown/)",
            )
            print(f"Saved: {hm_path}")

            surv_path = output_dir / f"surviving_clips_{tax_short}_{split_short}_{ts}.png"
            plot_surviving_clips(
                hm_data, args.threshold, surv_path,
                title=f"Surviving clips after >{args.threshold:.0%} "
                      f"{hm_label} hit-zone filter (excl. unknown/)",
            )
            print(f"Saved: {surv_path}")

    finally:
        # Restore real stdout before writing the .txt (so we don't recurse).
        sys.stdout = tee._stdout

    with open(txt_path, "w") as f:
        f.write(tee.get_text())
    print(f"Saved: {txt_path}")

    # --- Save out-of-bounds clip list (if any) ---
    if oob_clips:
        oob_path = output_dir / f"hit_oob_clips_{tax_short}_{split_short}_{ts}.txt"
        with open(oob_path, "w") as f:
            f.write("# Clips where the hit-frame index exceeded the clip length.\n")
            f.write("# These were skipped in the hit-frame profile plot.\n")
            f.write(f"# clip_name  rel_path  hit_idx  total_frames\n\n")
            for clip_name, rel_path, hit_idx, total_frames in sorted(oob_clips):
                f.write(f"{clip_name}  {rel_path}  "
                        f"hit_idx={hit_idx}  total_frames={total_frames}\n")
        print(f"Saved: {oob_path}")


if __name__ == "__main__":
    main()
