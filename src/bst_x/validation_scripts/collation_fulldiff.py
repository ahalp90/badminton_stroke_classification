"""Full-dataset, per-clip collation diff.

Compares two collated output dirs produced by ``collate_npy`` (e.g. main vs a
refactor branch), per-clip, aligned by clip_stem, element-wise. This is the
gate for any ``collate_npy`` edit: aggregate training metrics are
permutation-invariant and lossy (a row shuffle or a small value drift can
leave them identical), so they don't gate a split. Diff the write-out, not
the loaded dataset: the zero-frame drop and the train_partial RNG subset are
read-time (``shuttleset_dataset.py`` Dataset loader), downstream of what
``collate_npy`` produces.

Pure NumPy: no ``prepare_train`` / ``mmpose`` import, runs anywhere (e.g.
``venv-bst-x`` on HPC, or any environment with NumPy).
Memory-safe: arrays are ``mmap``'d and compared in row-chunks, so the multi-GB
pose arrays never load fully.

Usage (collate on side A into ``DIR_A`` and on side B into ``DIR_B``, SAME
taxonomy / split-column / seq-len / pose-styles, then):
    python src/bst_x/validation_scripts/collation_fulldiff.py DIR_A DIR_B

Each dir holds per-set subdirs (``train/`` ``val/`` ``test/``), each with
``{pose_style}.npy`` + ``pos`` / ``shuttle`` / ``videos_len`` / ``labels`` /
``clip_stems.npy``. Exit 0 iff every set's every array is identical (to
``COLLATION_ATOL``, default 0.0 = exact) after aligning by ``clip_stem``.

Set ``COLLATION_ATOL>0`` only if A and B were collated on DIFFERENT machines
(float drift); a same-machine before / after split should be exact.

Originally landed as the HPC analogue of the per-batch collation goldens in
the simplification pass (B4 / D6); kept in-repo as the standing gate for any
future ``collate_npy`` edit.
"""
import os
import sys
from pathlib import Path

import numpy as np

ATOL = float(os.environ.get("COLLATION_ATOL", "0.0"))
CHUNK = 512  # rows per block when comparing mmap'd arrays


def _compare_array(path_a: Path, path_b: Path, rows_a, rows_b) -> tuple[str, float | None, int]:
    """Element-wise compare two .npy at the aligned row indices, in row-chunks (mmap).
    Returns (status, max_abs_diff_or_None, n_mismatch). status: 'ok' | 'shape' | 'diff'.
    NaN-aware: NaN in the same position counts as equal; NaN-vs-number is a mismatch (and
    forces max_d to inf so it never passes ATOL). max_d is None for non-float arrays."""
    a = np.load(path_a, mmap_mode="r")
    b = np.load(path_b, mmap_mode="r")
    if a.shape[1:] != b.shape[1:]:
        return ("shape", None, 0)
    is_float = np.issubdtype(a.dtype, np.floating)
    max_d, n_mis = 0.0, 0
    for start in range(0, len(rows_a), CHUNK):
        ca = np.asarray(a[rows_a[start:start + CHUNK]])
        cb = np.asarray(b[rows_b[start:start + CHUNK]])
        if is_float:
            if np.array_equal(ca, cb, equal_nan=True):
                continue
            both_nan = np.isnan(ca) & np.isnan(cb)
            differ = ~((ca == cb) | both_nan)
            n_mis += int(differ.sum())
            d = np.abs(ca[differ].astype(np.float64) - cb[differ].astype(np.float64))
            finite = d[np.isfinite(d)]
            # NaN-vs-number leaves a non-finite diff -> inf so it can't pass ATOL.
            max_d = max(max_d, float(finite.max()) if finite.size else float("inf"))
        elif not np.array_equal(ca, cb):
            n_mis += int((ca != cb).sum())
    if is_float:
        return ("ok" if (n_mis == 0 or max_d <= ATOL) else "diff", max_d, n_mis)
    return ("ok" if n_mis == 0 else "diff", None, n_mis)


def _diff_set(dir_a: Path, dir_b: Path, set_name: str) -> list[str]:
    a, b = dir_a / set_name, dir_b / set_name
    if a.is_dir() != b.is_dir():
        return [f"{set_name}: present in only one dir"]
    if not a.is_dir():
        return []

    problems = []
    files_a = {p.name for p in a.glob("*.npy")}
    files_b = {p.name for p in b.glob("*.npy")}
    if files_a != files_b:
        problems.append(f"{set_name}: file set differs (only-A={sorted(files_a - files_b)}, "
                        f"only-B={sorted(files_b - files_a)})")

    stems_a = np.load(a / "clip_stems.npy", allow_pickle=True).tolist()
    stems_b = np.load(b / "clip_stems.npy", allow_pickle=True).tolist()
    set_a, set_b = set(stems_a), set(stems_b)
    if set_a != set_b:
        only = sorted(set_a ^ set_b)[:5]
        problems.append(f"{set_name}: clip_stem SET differs (only-A={len(set_a - set_b)}, "
                        f"only-B={len(set_b - set_a)}; e.g. {only})")
    if stems_a != stems_b:
        problems.append(f"{set_name}: clip_stem ORDER differs (the comparison is "
                        f"aligned by stem; flag here so a row reorder is visible)")

    idx_a = {s: i for i, s in enumerate(stems_a)}
    idx_b = {s: i for i, s in enumerate(stems_b)}
    common = sorted(set_a & set_b)
    rows_a = np.fromiter((idx_a[s] for s in common), dtype=np.int64, count=len(common))
    rows_b = np.fromiter((idx_b[s] for s in common), dtype=np.int64, count=len(common))

    for fname in sorted((files_a & files_b) - {"clip_stems.npy"}):
        status, max_d, n_mis = _compare_array(a / fname, b / fname, rows_a, rows_b)
        if status == "shape":
            problems.append(f"{set_name}/{fname}: shape mismatch")
        elif status == "diff":
            detail = f" (max abs {max_d:.3e})" if max_d is not None else ""
            problems.append(f"{set_name}/{fname}: differs on {n_mis} elements over "
                            f"{len(common)} aligned clips{detail}")
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: collation_fulldiff.py DIR_A DIR_B")
        return 2
    dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])
    sets = [s for s in ("train", "val", "test") if (dir_a / s).is_dir() or (dir_b / s).is_dir()]
    problems = [p for s in sets for p in _diff_set(dir_a, dir_b, s)]
    if problems:
        print(f"FAIL: {len(problems)} divergence(s) across {sets} (atol={ATOL}):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"OK: {dir_a.name} vs {dir_b.name} identical per-clip across {sets} (atol={ATOL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
