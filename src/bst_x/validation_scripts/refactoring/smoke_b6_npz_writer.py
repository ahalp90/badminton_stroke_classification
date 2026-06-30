"""Byte-compare smoke for the prediction-npz writer.

Writes the same payload via the verbatim pre-refactor inline ``savez`` block
and via ``bst_x_common._write_prediction_npz``, then diffs the two npz files
on key set, key order, per-key dtype + shape, and ``np.array_equal``. Both
``bst_x_train.Task.dump_predictions`` and ``bst_x_infer.dump_run_predictions``
used to inline the same payload; lifting it to the helper has to leave the
on-disk bytes identical because the schema is consumed by
``build_fe_stats_jsons``, ``calibration_ece``, and the FE downstream.

The smoke fakes the inputs both call sites assemble (a dump dict, a dataset
stub exposing ``clip_stems``, a taxonomy stub exposing ``classes`` + ``name``,
``run_id``, ``serial``) and asserts:

  - same key set
  - same key order (so ``np.load(...).files`` lines up)
  - same dtype + shape per key
  - ``np.array_equal`` per key

A pure value-compare passing keys / dtypes / shapes would still hide a
transposed same-dtype field (e.g. ``y_pred_top1`` and ``y_true`` both int64
length N), so the ``np.array_equal`` pass is the actual gate.

Run locally:
    ~/.venvs/badminton-cicd/bin/python \\
        src/bst_x/validation_scripts/refactoring/smoke_b6_npz_writer.py

Expect one OK line. Failure: per-key printout of the diverging field.

Originally landed as B6 in the simplification pass. The inline block being
compared against was lifted verbatim from ``bst_x_infer.py`` lines 256-269 and
``bst_x_train.py`` lines 1076-1087 at the pre-B6 tip.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# script lives at src/bst_x/validation_scripts/refactoring/<name>.py;
# parents[2] is src/bst_x/.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from bst_x_common import _write_prediction_npz  # noqa: E402

N, N_CLASSES, K = 32, 14, 5


def fake_inputs():
    rng = np.random.default_rng(0)
    dump = {
        "logits":      rng.standard_normal((N, N_CLASSES)).astype(np.float32),
        "y_true":      rng.integers(0, N_CLASSES, size=(N,), dtype=np.int64),
        "y_pred_top1": rng.integers(0, N_CLASSES, size=(N,), dtype=np.int64),
        "topk_idx":    rng.integers(0, N_CLASSES, size=(N, K), dtype=np.int64),
    }
    dataset = SimpleNamespace(
        clip_stems=[f"vid{i}_set{i % 3}_match{i % 5}" for i in range(N)],
    )
    taxonomy = SimpleNamespace(
        classes=tuple(f"class_{i}" for i in range(N_CLASSES)),
        name="une_v1_14",
    )
    return dump, dataset, taxonomy, "run_smoke_20260630", 3


def old_inline_savez(out_path, dump, dataset, taxonomy, run_id, serial):
    """Verbatim from bst_x_train.py pre-B6 (lines 1076-1087); identical
    payload to bst_x_infer.py pre-B6 lines 256-269."""
    assert dataset.clip_stems is not None
    np.savez(
        out_path,
        logits=dump["logits"],
        y_true=dump["y_true"],
        y_pred_top1=dump["y_pred_top1"],
        topk_idx=dump["topk_idx"],
        clip_stems=np.asarray(dataset.clip_stems, dtype=object),
        class_list=np.array(taxonomy.classes, dtype=object),
        run_id=np.array(run_id, dtype=object),
        serial_no=np.array(serial, dtype=np.int64),
        taxonomy_name=np.array(taxonomy.name, dtype=object),
    )


def main() -> int:
    dump, dataset, taxonomy, run_id, serial = fake_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        old_path = Path(tmp) / "old.npz"
        new_path = Path(tmp) / "new.npz"
        old_inline_savez(old_path, dump, dataset, taxonomy, run_id, serial)
        _write_prediction_npz(new_path, dump, dataset, taxonomy, run_id, serial)

        a = np.load(old_path, allow_pickle=True)
        b = np.load(new_path, allow_pickle=True)
        if set(a.files) != set(b.files):
            print(f"FAIL: key set differs\n  old: {sorted(a.files)}\n  new: {sorted(b.files)}")
            return 1
        if list(a.files) != list(b.files):
            print(f"FAIL: key ORDER differs\n  old: {a.files}\n  new: {b.files}")
            return 1
        bad = []
        for k in a.files:
            av, bv = a[k], b[k]
            if av.dtype != bv.dtype:
                bad.append(f"  {k}: dtype {av.dtype} != {bv.dtype}")
            if av.shape != bv.shape:
                bad.append(f"  {k}: shape {av.shape} != {bv.shape}")
            if av.dtype == object:
                if not all(x == y for x, y in zip(av.tolist(), bv.tolist())):
                    bad.append(f"  {k}: object values differ")
            elif not np.array_equal(av, bv):
                bad.append(f"  {k}: values differ (max abs {float(np.abs(av.astype(float) - bv.astype(float)).max()):.3e})")
        if bad:
            print("FAIL: old inline savez vs _write_prediction_npz diverge")
            print("\n".join(bad))
            return 1
        print(f"OK: pre-refactor inline savez and _write_prediction_npz produce "
              f"byte-identical npz ({len(a.files)} keys, "
              f"{[f'{k}={a[k].shape}{a[k].dtype}' for k in a.files]})")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
