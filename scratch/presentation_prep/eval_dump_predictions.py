"""Eval-only dump of (y_true, y_pred) for the 2026-05-11 supervisor confusion matrix.

Loads a trained serial from a completed run, runs the test split with the same
loader the training pipeline used, and writes a new file under
``<run_dir>/predictions/serial_<n>.pt``. Touches nothing else in the run dir.

Run on engelbart from the repo root with the same PYTHONPATH convention as
``bst_infer.py``::

    PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \\
        /path/to/training/venv/bin/python \\
        scratch/presentation_prep/eval_dump_predictions.py \\
        --run-dir src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260505_154907 \\
        --serial 5

A pre-run / post-run listing of the run dir is printed so any unexpected
mutation is visible in the log. Only ``predictions/`` should be new.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Same import surface as bst_infer.py; needs both package roots on PYTHONPATH.
from preparing_data.shuttleset_dataset import Dataset_npy_collated  # noqa: E402
from pipeline.config import TAXONOMIES, derive_npy_collated_dir_basename  # noqa: E402
from main_on_shuttleset.bst_common import build_bst_network  # noqa: E402


def snapshot_run_dir(run_dir: Path) -> dict[str, tuple[int, int]]:
    """Walk top-level entries and capture (size_bytes, mtime_ns) for each.

    Used as a before/after sanity check to confirm the eval script didn't
    touch manifest.yaml / best_model_id.txt / tb/ / weights/ etc.

    :param run_dir: experiment run directory to snapshot
    :return: {relative_path_str: (size_bytes, mtime_ns)} over every regular file
    """
    snapshot = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(run_dir))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def diff_snapshots(before: dict, after: dict) -> tuple[list[str], list[str], list[str]]:
    """Three-way diff: appeared, disappeared, mutated (size or mtime changed).

    :return: (new_files, removed_files, mutated_files)
    """
    before_keys = set(before)
    after_keys = set(after)
    new_files = sorted(after_keys - before_keys)
    removed_files = sorted(before_keys - after_keys)
    mutated_files = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return new_files, removed_files, mutated_files


def build_active_remap(taxonomy_name: str, active_class_list: list[str]) -> np.ndarray:
    """Vector remap from on-disk full-taxonomy label index to active-head index.

    ``Dataset_npy_collated`` returns labels in the taxonomy's full index space.
    Training rebuilds the head at ``len(active_class_list)`` and remaps labels
    in-memory via ``bst_common.derive_active_classes_from_labels``; this helper
    reproduces that remap from the manifest-recorded ``active_class_list``
    (which is the canonical record of what the head was trained to predict).

    :param taxonomy_name: taxonomy key from manifest.config.taxonomy
    :param active_class_list: per manifest.extra.arch.active_class_list
    :return: int64 vector of length ``len(full_class_list)``; ``remap[k]`` is the
        active-head index of full-taxonomy class ``k``, or -1 if class ``k`` is
        not in the active head. Any -1 hit by a test label is a data-config bug.
    """
    full = TAXONOMIES[taxonomy_name].class_list()
    name_to_active_idx = {name: i for i, name in enumerate(active_class_list)}
    remap = np.full(len(full), -1, dtype=np.int64)
    for full_idx, name in enumerate(full):
        if name in name_to_active_idx:
            remap[full_idx] = name_to_active_idx[name]
    return remap


@torch.no_grad()
def run_test_split(model: torch.nn.Module, loader: DataLoader, device: str
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward over the test loader, returning argmax preds and ground-truth labels.

    Mirrors the forward shape massage in ``bst_infer.infer`` (flatten the
    per-joint trailing dim of human_pose before the call).

    :return: (y_pred, y_true) — 1-D long tensors of length n_strokes
    """
    model.eval()
    pred_chunks, label_chunks = [], []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose = human_pose.to(device)
        shuttle = shuttle.to(device)
        pos = pos.to(device)
        video_len = video_len.to(device)
        # Flatten (joints, channels) into a single trailing dim, as bst_infer does.
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        pred_chunks.append(torch.argmax(logits, dim=1).cpu())
        label_chunks.append(labels.cpu())
    return torch.cat(pred_chunks), torch.cat(label_chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Absolute or repo-relative path to experiments/run_<id>/")
    parser.add_argument("--serial", type=int, default=5,
                        help="Serial number whose weights to evaluate")
    parser.add_argument("--model-name", default="BST_CG_AP",
                        help="BST variant string; matches the partial used at train time")
    parser.add_argument("--n-joints", type=int, default=17)
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--collated-root-override", type=Path, default=None,
                        help="Override the parent of npy_<basename>; "
                             "defaults to <bst_refactor>/preparing_data/ShuttleSet_data_<taxonomy>/")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"run dir does not exist: {run_dir}")

    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    config = manifest["config"]
    arch = manifest.get("extra", {}).get("arch")
    if arch is None:
        sys.exit("manifest has no extra.arch block; this script needs the post-fix arch metadata. "
                 "Use bst_infer.py's legacy-fallback path for pre-fix runs.")

    n_active_classes = arch["n_active_classes"]
    active_class_list = arch["active_class_list"]

    # Snapshot before any work.
    before = snapshot_run_dir(run_dir)

    # Resolve weights for the requested serial. Manifest stores a repo-relative path;
    # the file itself lives at run_dir/weights/<basename>, so just rebuild from there.
    target_serial = next(
        (s for s in manifest["serials"] if s["serial_no"] == args.serial), None
    )
    if target_serial is None:
        sys.exit(f"serial {args.serial} not found in manifest")
    weights_basename = Path(target_serial["weights_path"]).name
    weights_path = run_dir / "weights" / weights_basename
    if not weights_path.is_file():
        sys.exit(f"weights file missing: {weights_path}")

    # Reconstruct the npy_collated_dir the training pipeline used. bst_train builds it
    # from Path(__file__).resolve().parent.parent (= bst_refactor); we mirror that by
    # walking up from the run dir: experiments → main_on_shuttleset → stroke_classification → bst_refactor.
    bst_refactor_root = run_dir.parents[3]
    basename = derive_npy_collated_dir_basename(
        taxonomy_name=config["taxonomy"],
        split_column=config["split_column"],
        drop_unknown=config["drop_unknown"],
        use_3d_pose=config["use_3d_pose"],
        seq_len=config["seq_len"],
        ablation_id=config["ablation_id"],
    )
    if args.collated_root_override is not None:
        collated_dir = args.collated_root_override / basename
    else:
        collated_dir = (
            bst_refactor_root
            / f"preparing_data/ShuttleSet_data_{config['taxonomy']}"
            / basename
        )
    if not collated_dir.is_dir():
        sys.exit(f"collated dir missing: {collated_dir}")

    print(f"run_dir: {run_dir}")
    print(f"weights: {weights_path}")
    print(f"collated_dir: {collated_dir}")
    print(f"active classes ({n_active_classes}): {active_class_list}")

    # Test loader, matching bst_infer.prepare_loader.
    test_set = Dataset_npy_collated(collated_dir, "test", config["pose_style"])

    # Remap labels from full-taxonomy index space to active-head index space.
    # Training does this in-memory after constructing the dataset; the on-disk
    # labels.npy is left untouched.
    remap = build_active_remap(config["taxonomy"], active_class_list)
    labels_full = np.asarray(test_set.labels, dtype=np.int64)
    labels_active = remap[labels_full]
    if (labels_active < 0).any():
        rogue = np.unique(labels_full[labels_active < 0]).tolist()
        sys.exit(f"test labels contain full-taxonomy indices not in the active head: {rogue}")
    test_set.labels = labels_active

    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net, _n_bones = build_bst_network(
        args.model_name,
        n_joints=args.n_joints,
        pose_style=config["pose_style"],
        in_channels=args.in_channels,
        n_class=n_active_classes,
        seq_len=config["seq_len"],
        device=device,
    )
    net.load_state_dict(
        torch.load(str(weights_path), map_location=device, weights_only=True)
    )

    y_pred, y_true = run_test_split(net, test_loader, device)
    print(f"n_strokes: {len(y_true)} ({len(y_pred)} preds)")

    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"serial_{args.serial}.pt"
    torch.save({
        "y_true": y_true,
        "y_pred": y_pred,
        "active_class_list": active_class_list,
        "run_id": manifest["run_id"],
        "serial_no": args.serial,
    }, out_path)
    print(f"saved: {out_path}")

    # Sanity check: nothing outside predictions/ should have changed.
    after = snapshot_run_dir(run_dir)
    new_files, removed_files, mutated_files = diff_snapshots(before, after)

    unexpected_new = [f for f in new_files if not f.startswith("predictions/")]
    if removed_files or mutated_files or unexpected_new:
        print("\nWARNING: run dir was mutated outside predictions/:")
        if removed_files:
            print(f"  removed: {removed_files}")
        if mutated_files:
            print(f"  mutated: {mutated_files}")
        if unexpected_new:
            print(f"  unexpected new: {unexpected_new}")
        sys.exit(2)
    else:
        print(f"\nsanity ok: only new files are under predictions/ ({len(new_files)} added)")


if __name__ == "__main__":
    main()
