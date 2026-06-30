"""Diff two bst_x_train run dirs for the real-data bit-exact use case.

Compares the artefacts that record what the training trajectory was:
  - per-serial checkpoint .pt SHA256
  - per-serial prediction npz SHA256 (logits / y_true / y_pred_top1 / topk_idx /
    clip_stems / class_list / run_id / serial_no / taxonomy_name)
  - manifest's per-serial test_metrics + topk_metrics + val_at_best_macro_epoch

Pairs with ``seed_and_run_bst_x_train.py``: that launcher pins the RNG and runs
``bst_x_train`` on each side; this script diffs the resulting run dirs.

Usage:
    ~/.venvs/badminton-cicd/bin/python \\
        src/bst_x/validation_scripts/refactoring/compare_b7_real_runs.py \\
        <reference_run_dir> <candidate_run_dir>

Each ``<run_dir>`` is an ``experiments/bst_x/shuttleset/<run_id>/`` tree
containing ``manifest.yaml``, ``weights/``, and ``predictions/``. Exit 0 iff
every artefact matches.

Originally landed in the simplification pass as B7's consumer.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml


# run_id legitimately differs by side (one --run-id per invocation); skip it.
NPZ_RUN_ID_DIFFERENT_BY_SIDE = {"run_id"}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def diff_npz(a_path: Path, b_path: Path) -> list[str]:
    """Per-key value diff (skips run_id which legitimately differs by --run-id)."""
    a = np.load(a_path, allow_pickle=True)
    b = np.load(b_path, allow_pickle=True)
    problems: list[str] = []
    if set(a.files) != set(b.files):
        problems.append(f"{a_path.name}: key set differs: ref={sorted(a.files)}, cand={sorted(b.files)}")
        return problems
    for k in a.files:
        if k in NPZ_RUN_ID_DIFFERENT_BY_SIDE:
            continue
        av, bv = a[k], b[k]
        if av.dtype != bv.dtype:
            problems.append(f"{a_path.name} key {k}: dtype {av.dtype} != {bv.dtype}")
            continue
        if av.shape != bv.shape:
            problems.append(f"{a_path.name} key {k}: shape {av.shape} != {bv.shape}")
            continue
        if av.dtype == object:
            if list(av.flat) != list(bv.flat):
                problems.append(f"{a_path.name} key {k}: object values differ")
        elif not np.array_equal(av, bv):
            d = float(np.abs(av.astype(np.float64) - bv.astype(np.float64)).max())
            problems.append(f"{a_path.name} key {k}: values differ (max abs {d:.3e})")
    return problems


def diff_manifest(a: dict, b: dict) -> list[str]:
    """Compare per-serial metrics + val_at_best across two manifests.

    Skips fields that legitimately differ by side: ``serial_no`` is used for
    indexing, ``weights_path`` and ``tb_dir`` are cwd-relative paths that bake
    in the run_id, ``recorded_at`` is a timestamp.
    """
    problems: list[str] = []
    sa = {s["serial_no"]: s for s in (a.get("serials") or [])}
    sb = {s["serial_no"]: s for s in (b.get("serials") or [])}
    if set(sa.keys()) != set(sb.keys()):
        problems.append(
            f"manifest serial_no set differs: ref={sorted(sa)}, cand={sorted(sb)}"
        )
        return problems
    metric_keys = ("macro_f1", "min_f1", "accuracy", "top2_accuracy",
                   "num_strokes")
    for k in sorted(sa.keys()):
        ma = sa[k].get("metrics") or {}
        mb = sb[k].get("metrics") or {}
        for mk in metric_keys:
            va, vb = ma.get(mk), mb.get(mk)
            if va != vb:
                problems.append(f"serial {k} metrics.{mk}: ref={va!r} cand={vb!r}")
        pa = ma.get("per_class_f1") or {}
        pb = mb.get("per_class_f1") or {}
        if pa != pb:
            problems.append(f"serial {k} metrics.per_class_f1: ref={pa} cand={pb}")
        ea = (sa[k].get("extra") or {}).get("val_at_best_macro_epoch")
        eb = (sb[k].get("extra") or {}).get("val_at_best_macro_epoch")
        if ea != eb:
            problems.append(f"serial {k} extra.val_at_best_macro_epoch: ref={ea!r} cand={eb!r}")
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref_dir = Path(sys.argv[1])
    cand_dir = Path(sys.argv[2])
    if not ref_dir.is_dir() or not cand_dir.is_dir():
        print(f"FAIL: one or both run dirs missing\n  ref: {ref_dir}\n  cand: {cand_dir}")
        return 2

    problems: list[str] = []

    # manifest
    m_ref = yaml.safe_load((ref_dir / "manifest.yaml").read_text())
    m_cand = yaml.safe_load((cand_dir / "manifest.yaml").read_text())
    problems.extend(diff_manifest(m_ref, m_cand))

    # checkpoints
    weights_ref = sorted((ref_dir / "weights").glob("*.pt"))
    weights_cand = sorted((cand_dir / "weights").glob("*.pt"))
    names_ref = [p.name for p in weights_ref]
    names_cand = [p.name for p in weights_cand]
    if names_ref != names_cand:
        problems.append(f"checkpoint names differ: ref={names_ref}, cand={names_cand}")
    for a, b in zip(weights_ref, weights_cand):
        sa = sha_file(a)
        sb = sha_file(b)
        if sa != sb:
            problems.append(f"checkpoint {a.name}: SHA ref={sa[:16]}... != cand={sb[:16]}...")

    # prediction npzs: per-key value diff, run_id (which legitimately differs by
    # --run-id) skipped. A raw SHA compare would always fail because the run_id
    # field is baked into the saved bytes.
    npzs_ref = sorted((ref_dir / "predictions").glob("*.npz"))
    npzs_cand = sorted((cand_dir / "predictions").glob("*.npz"))
    names_ref = [p.name for p in npzs_ref]
    names_cand = [p.name for p in npzs_cand]
    if names_ref != names_cand:
        problems.append(f"npz names differ: ref={names_ref}, cand={names_cand}")
    for a, b in zip(npzs_ref, npzs_cand):
        problems.extend(diff_npz(a, b))

    if problems:
        print(f"FAIL: {len(problems)} divergence(s) between {ref_dir} and {cand_dir}")
        for p in problems:
            print(f"  {p}")
        return 1
    print(
        f"OK: real-data bit-exact IDENTICAL\n"
        f"  ref:  {ref_dir}\n"
        f"  cand: {cand_dir}\n"
        f"  weights: {len(weights_ref)} .pt files, all SHA-equal\n"
        f"  predictions: {len(npzs_ref)} .npz files, per-key value-equal "
        f"(run_id excluded, legitimately differs by --run-id)\n"
        f"  manifest: per-serial test_metrics + val_at_best identical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
