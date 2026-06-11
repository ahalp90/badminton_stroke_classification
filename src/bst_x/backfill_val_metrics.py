"""Backfill val aggregate metrics into a run's manifest, no forward pass.

Training records val per-class F1 + the best-macro epoch under each serial's
``extra.val_at_best_macro_epoch``, but not the aggregate scalars. This completes
that block so the FE registry can serve val with the same fields as test:

- ``macro_f1`` / ``min_f1``: the unweighted mean / min of the per-class F1
  already in the manifest, so they stay exactly consistent with the per-class the
  card shows (no float drift from a recompute).
- ``accuracy`` / ``top2_accuracy``: read off the saved val prediction npz
  (``y_pred_top1`` and ``topk_idx``). These are exact integer-ratio counts over
  stored predictions, not a re-evaluation, so no checkpoint reload is needed.

The npz per-class is recomputed only as a guard: it must match the manifest's
recorded val per-class, else the npz is a different pass and we fail loud rather
than write mismatched accuracy/top2.

Idempotent: re-running overwrites the four keys. Once a serial's npz is pruned it
keeps any accuracy/top2 already written, so a second pass over a trimmed run dir
is safe.

usage: python src/bst_x/backfill_val_metrics.py <run_dir>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

VAL_BLOCK_KEYS = {"epoch", "per_class_f1", "macro_f1", "min_f1", "accuracy", "top2_accuracy"}


def accuracy_top2_perclass_from_npz(npz_path: Path) -> tuple[float, float, dict[str, float]]:
    """Compute val accuracy, top-2 accuracy, and per-class F1 from a prediction npz.

    accuracy / top-2 are exact ratios over the stored ``y_pred_top1`` and
    ``topk_idx``; per-class is returned only for the correspondence guard.

    :param npz_path: a ``val_serial_{n}.npz`` prediction dump.
    :return: ``(accuracy, top2_accuracy, per_class_f1_by_name)``.
    """
    npz = np.load(npz_path, allow_pickle=True)
    y_true, y_pred, topk = npz["y_true"], npz["y_pred_top1"], npz["topk_idx"]
    classes = list(npz["class_list"])
    accuracy = float((y_true == y_pred).mean())
    in_top2 = np.fromiter((yt in row[:2] for yt, row in zip(y_true, topk)), dtype=bool)
    top2 = float(in_top2.mean())
    per_class: dict[str, float] = {}
    for idx, name in enumerate(classes):  # idx indexes the class axis of logits
        tp = int(((y_pred == idx) & (y_true == idx)).sum())
        fp = int(((y_pred == idx) & (y_true != idx)).sum())
        fn = int(((y_pred != idx) & (y_true == idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, top2, per_class


def backfill_run(run_dir: Path) -> None:
    """Complete every serial's ``val_at_best_macro_epoch`` block in the manifest.

    Writes back with the same ``safe_dump`` call ``run_tracker`` uses, so only the
    four added keys per serial change; the rest of the file is untouched.

    :param run_dir: experiment run directory holding ``manifest.yaml`` + ``predictions/``.
    """
    manifest_path = run_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())

    for serial in manifest["serials"]:
        serial_no = serial["serial_no"]
        val_block = serial["extra"]["val_at_best_macro_epoch"]
        # Fail loud if the block ever grows a key we'd silently drop on rebuild.
        unexpected = set(val_block) - VAL_BLOCK_KEYS
        assert not unexpected, f"serial {serial_no}: unexpected val keys {unexpected}"

        per_class = val_block["per_class_f1"]
        f1s = list(per_class.values())
        macro_f1 = sum(f1s) / len(f1s)
        min_f1 = min(f1s)

        npz_path = run_dir / "predictions" / f"val_serial_{serial_no}.npz"
        if npz_path.exists():
            accuracy, top2, npz_per_class = accuracy_top2_perclass_from_npz(npz_path)
            max_diff = max(abs(npz_per_class[name] - per_class[name]) for name in per_class)
            assert max_diff < 1e-3, (
                f"serial {serial_no}: val npz per-class diverges from the manifest "
                f"(max abs diff {max_diff:.2e}); wrong npz for this serial?"
            )
        elif {"accuracy", "top2_accuracy"} <= set(val_block):
            accuracy, top2 = val_block["accuracy"], val_block["top2_accuracy"]
        else:
            raise FileNotFoundError(
                f"serial {serial_no}: {npz_path.name} missing and no accuracy/top2 "
                f"already in the manifest to keep."
            )

        # Rebuild with aggregates ahead of the per-class block, mirroring the test
        # metrics field set so the registry reads val exactly like test.
        serial["extra"]["val_at_best_macro_epoch"] = {
            "epoch": val_block["epoch"],
            "macro_f1": macro_f1,
            "min_f1": min_f1,
            "accuracy": accuracy,
            "top2_accuracy": top2,
            "per_class_f1": per_class,
        }
        print(
            f"serial {serial_no}: macro {macro_f1:.6f}  min {min_f1:.6f}  "
            f"acc {accuracy:.6f}  top2 {top2:.6f}"
        )

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False))
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    backfill_run(Path(sys.argv[1]))
