"""Byte-identity gate for the raw-extract + apply_heuristic plumbing.

Runs the ``current`` heuristic on a deterministic sample of the 1,716-clip
hit-zone busted list and compares the produced ``_pos``/``_joints``/``_failed``
outputs against the committed extract passed via ``--committed-dir``. Any
mismatch means the plumbing is wrong and no other heuristic variant (e.g.
``sticky_anchor``) should be trusted until it is fixed.

Comparison tolerances:

- ``_failed.npy``: ``np.array_equal`` (bool must match exactly).
- ``_pos.npy`` and ``_joints.npy``: ``np.allclose(rtol=0, atol=1e-5)``
  (absorbs float32 -> float64 projection-chain non-associativity between
  the two code paths).

Sampling is deterministic with no seeding: the 1,716 stems are sorted
lexicographically and every ``len // sample_size``-th stem is kept. This
spreads the sample across video IDs without introducing run-to-run noise.

``--committed-dir`` is required. ``$BST_X_MMPOSE_NPY_DIR`` points at the
``sticky_anchor`` clean extract, which is a different heuristic from
``current`` and so would always fail this gate; the script refuses to default
to it. For a refactor branch gate where no pre-existing ``current``-extract
dir exists, use the dual-invocation pattern: run on main first with
``--scratch-output-dir`` pointed at a fresh ``$REF`` dir (the comparison line
in the log can be ignored on this pass — its purpose is to populate ``$REF``
with main's current-extract NPYs), then run on the branch with
``--committed-dir $REF`` and ``--scratch-output-dir`` pointed at a second
dir. The branch's current-extract output gets compared stem-by-stem against
main's.

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src/bst_x \\
        python -m validation_scripts.failsafe_bst_mmpose_zeroing_check_equivalence \\
            --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw \\
            --committed-dir /scratch/comp320a/clean_current_main_50 \\
            --busted-stems-file docs/architecture_notes/busted_hit_zone_clips_phase1.txt \\
            --clips-csv notebooks/clips_master.csv \\
            --scratch-output-dir /scratch/comp320a/clean_current_branch_50
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preparing_data.apply_heuristic import run as run_apply_heuristic


FLOAT_ATOL = 1e-5


@dataclass
class StemCompare:
    stem: str
    failed_equal: bool
    pos_close: bool
    joints_close: bool
    pos_max_abs_diff: float
    joints_max_abs_diff: float

    @property
    def passed(self) -> bool:
        return self.failed_equal and self.pos_close and self.joints_close


def _load_triple(directory: Path, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = str(directory / stem)
    pos = np.load(branch + "_pos.npy")
    joints = np.load(branch + "_joints.npy")
    failed = np.load(branch + "_failed.npy")
    return pos, joints, failed


def _compare_stem(
    stem: str, scratch_dir: Path, committed_dir: Path,
) -> StemCompare:
    pos_a, joints_a, failed_a = _load_triple(scratch_dir, stem)
    pos_b, joints_b, failed_b = _load_triple(committed_dir, stem)

    failed_equal = bool(np.array_equal(failed_a, failed_b))

    def _safe_close_and_diff(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
        if a.shape != b.shape:
            return False, float("inf")
        close = bool(np.allclose(a, b, rtol=0, atol=FLOAT_ATOL))
        max_abs_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        return close, max_abs_diff

    pos_close, pos_diff = _safe_close_and_diff(pos_a, pos_b)
    joints_close, joints_diff = _safe_close_and_diff(joints_a, joints_b)

    return StemCompare(
        stem=stem,
        failed_equal=failed_equal,
        pos_close=pos_close,
        joints_close=joints_close,
        pos_max_abs_diff=pos_diff,
        joints_max_abs_diff=joints_diff,
    )


def _load_stems_file(path: Path) -> list[str]:
    with path.open() as fh:
        return [line.strip() for line in fh if line.strip()]


def _sample_every_nth(stems: list[str], sample_size: int) -> list[str]:
    """Lex-sort then take every ``len // sample_size``-th stem.

    Returns fewer than ``sample_size`` stems if the input is shorter.
    """
    if not stems:
        return []
    ordered = sorted(stems)
    step = max(1, len(ordered) // sample_size)
    sampled = ordered[::step][:sample_size]
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--committed-dir", type=Path, required=True,
        help="Current-extract committed dir to compare against. Required: "
             "$BST_X_MMPOSE_NPY_DIR points at the sticky_anchor extract, which "
             "would always fail this gate; the script does not fall back to it.",
    )
    parser.add_argument(
        "--busted-stems-file", type=Path, required=True,
        help="Canonical 1,716-stem hit-zone list (one stem per line).",
    )
    parser.add_argument(
        "--clips-csv", type=Path, required=True,
        help="clips_master.csv -- passed through to apply_heuristic.run.",
    )
    parser.add_argument(
        "--scratch-output-dir", type=Path, default=None,
        help="Where the current-variant outputs are written for comparison. "
             "Defaults to a fresh tempdir; pass an explicit path to keep "
             "outputs around for debugging.",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--split-column", type=str, default=None,
        help="Passed through to apply_heuristic.run. Omit to disable the "
             "split filter (recommended for the gate; we only restrict by "
             "the busted-stems file).",
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    committed_dir = args.committed_dir
    if not committed_dir.is_dir():
        print(f"ERROR: committed-dir not found: {committed_dir}", file=sys.stderr)
        return 2

    all_stems = _load_stems_file(args.busted_stems_file)
    print(f"Loaded {len(all_stems)} stems from {args.busted_stems_file}")
    sampled = _sample_every_nth(all_stems, args.sample_size)
    print(f"Sampled {len(sampled)} stems (every {max(1, len(all_stems) // args.sample_size)}-th)")

    with_raw = [
        stem for stem in sampled
        if (args.raw_dir / f"{stem}_raw_ndet.npy").exists()
    ]
    with_committed = [
        stem for stem in with_raw
        if all((committed_dir / f"{stem}{suf}").exists()
               for suf in ("_pos.npy", "_joints.npy", "_failed.npy"))
    ]
    dropped_no_raw = len(sampled) - len(with_raw)
    dropped_no_committed = len(with_raw) - len(with_committed)
    if dropped_no_raw or dropped_no_committed:
        print(
            f"Dropped {dropped_no_raw} sampled stems missing raw output, "
            f"{dropped_no_committed} missing committed output."
        )
    if not with_committed:
        print("No stems to compare after filtering; aborting.", file=sys.stderr)
        return 2

    print(f"Comparing {len(with_committed)} stems.")

    scratch_dir: Path
    scratch_owned_here = False
    if args.scratch_output_dir is not None:
        scratch_dir = args.scratch_output_dir
        scratch_dir.mkdir(parents=True, exist_ok=True)
    else:
        scratch_dir = Path(tempfile.mkdtemp(prefix="failsafe_gate_"))
        scratch_owned_here = True
        print(f"Created scratch dir: {scratch_dir}")

    splits_tuple = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    # Write the sampled stem list to a temp file so apply_heuristic can
    # narrow candidates via --clip-stems-file.
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", prefix="failsafe_stems_",
    ) as fh:
        for stem in with_committed:
            fh.write(stem + "\n")
        stems_file_path = Path(fh.name)

    try:
        run_apply_heuristic(
            raw_dir=args.raw_dir,
            output_dir=scratch_dir,
            heuristic="current",
            clips_csv=args.clips_csv,
            clip_stems_file=stems_file_path,
            split_column=args.split_column,
            splits=splits_tuple if args.split_column else None,
            resume=False,
            limit=None,
            dry_run=False,
            hyperparams=None,
        )
    finally:
        stems_file_path.unlink(missing_ok=True)

    # Compare per stem.
    results: list[StemCompare] = []
    for stem in with_committed:
        if not all((scratch_dir / f"{stem}{suf}").exists()
                   for suf in ("_pos.npy", "_joints.npy", "_failed.npy")):
            print(f"  MISSING: {stem} was not written by apply_heuristic.")
            continue
        results.append(_compare_stem(stem, scratch_dir, committed_dir))

    passes = [r for r in results if r.passed]
    fails = [r for r in results if not r.passed]

    print(
        f"\n=== Gate summary ===\n"
        f"Compared:  {len(results)}\n"
        f"Passed:    {len(passes)}\n"
        f"Failed:    {len(fails)}\n"
    )

    if fails:
        print("Failures:")
        for r in fails:
            print(
                f"  {r.stem}  "
                f"failed_equal={r.failed_equal}  "
                f"pos_close={r.pos_close}(max_abs={r.pos_max_abs_diff:.3e})  "
                f"joints_close={r.joints_close}(max_abs={r.joints_max_abs_diff:.3e})"
            )

    if results:
        max_pos = max(r.pos_max_abs_diff for r in results)
        max_joints = max(r.joints_max_abs_diff for r in results)
        print(
            f"\nFloat tolerance check (atol={FLOAT_ATOL}):\n"
            f"  pos    max abs diff across stems: {max_pos:.3e}\n"
            f"  joints max abs diff across stems: {max_joints:.3e}"
        )

    if scratch_owned_here:
        print(
            f"\nNote: scratch dir {scratch_dir} retained for inspection; "
            f"delete manually when done."
        )

    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
