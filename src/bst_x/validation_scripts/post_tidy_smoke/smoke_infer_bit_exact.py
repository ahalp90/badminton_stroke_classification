"""Bit-exact comparison of BST inference predictions across two code states (e.g. a refactor branch vs main).

Loads a real checkpoint, runs inference on a collated dir's test split, and
saves the predictions tensor as a ``.npy`` file. Run on both branches, then
diff the two files: byte-identical output proves bst_x_infer's lift to
``build_bst_x_network`` is behaviourally inert.

Required env vars:
  BST_X_DATA_DIR  -- path to a collated dir (npy_<split>_<collation_id>/, with test/labels.npy)
  WEIGHT_PATH   -- path to a real .pt checkpoint matching that dir's config

Optional env vars (defaults match the active Hyp on pre-phase-2-tidy):
  TAXONOMY      -- taxonomy name (default: une_v1_14)
  POSE_STYLE    -- pose style (default: JnB_bone)
  SEQ_LEN       -- sequence length (default: 100)
  IN_CHANNELS   -- 2 for 2D keypoints, 3 for 3D (default: 2)
  MODEL_NAME    -- BST variant (default: BST_CG_AP)
  OUT_PATH      -- where to write predictions npy (default: /tmp/smoke_infer_preds.npy)

Usage on engelbart:
  cd ~/badminton_stroke_classifier
  source /home/ahalperi/.venvs/venv-bst/bin/activate
  export BST_X_DATA_DIR=~/badminton_stroke_classifier/src/bst_x/preparing_data/ShuttleSet_data_une_v1_14/npy_v2_taxon_pinned_w_preds
  export WEIGHT_PATH=<full path to a recent .pt checkpoint matching the active Hyp>

  # CuBLAS deterministic mode -- without this CUDA picks different matmul
  # algorithms across runs and the output isn't byte-exact even on the
  # same code. The script sets torch.use_deterministic_algorithms(True);
  # this env var unlocks the same guarantee at the CuBLAS layer.
  export CUBLAS_WORKSPACE_CONFIG=:4096:8

  # PYTHONPATH gives access to both package roots (matches conftest.py
  # for tests and the documented invocation pattern post-step-P).
  export PYTHONPATH=src/bst_x

  # Run on pre-phase-2-tidy
  git checkout pre-phase-2-tidy
  OUT_PATH=/tmp/preds_post_tidy.npy python src/bst_x/validation_scripts/post_tidy_smoke/smoke_infer_bit_exact.py

  # Run on main
  git checkout main
  OUT_PATH=/tmp/preds_main.npy python src/bst_x/validation_scripts/post_tidy_smoke/smoke_infer_bit_exact.py

  # Diff the two prediction files
  python -c "import numpy as np; \
    a=np.load('/tmp/preds_post_tidy.npy'); b=np.load('/tmp/preds_main.npy'); \
    print('IDENTICAL' if np.array_equal(a, b) else f'DIFFER: {(a!=b).sum()}/{len(a)} mismatched class predictions')"

  git checkout pre-phase-2-tidy

A passing diff (``IDENTICAL``) means: same architecture, same loaded weights,
same forward pass output. Bit-exact across the bst_x_infer.py refactor.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from bst_x_infer import Task
from pipeline.config import resolve_taxonomy


def main() -> int:
    bst_data_dir = Path(os.environ["BST_X_DATA_DIR"]).resolve()
    weight_path = Path(os.environ["WEIGHT_PATH"]).resolve()
    taxonomy_name = os.environ.get("TAXONOMY", "une_v1_14")
    pose_style = os.environ.get("POSE_STYLE", "JnB_bone")
    seq_len = int(os.environ.get("SEQ_LEN", "100"))
    in_channels = int(os.environ.get("IN_CHANNELS", "2"))
    model_name = os.environ.get("MODEL_NAME", "BST_CG_AP")
    out_path = Path(os.environ.get("OUT_PATH", "/tmp/smoke_infer_preds.npy")).resolve()

    if not weight_path.exists():
        raise FileNotFoundError(f"WEIGHT_PATH does not exist: {weight_path}")
    if not (bst_data_dir / "test" / "labels.npy").exists():
        raise FileNotFoundError(
            f"BST_X_DATA_DIR does not contain test/labels.npy: {bst_data_dir}"
        )
    taxonomy = resolve_taxonomy(taxonomy_name)

    # Determinism flags. Inference has no augmentation; with these the same
    # checkpoint + same input must produce byte-identical output.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"BST_X_DATA_DIR : {bst_data_dir}")
    print(f"WEIGHT_PATH  : {weight_path}")
    print(f"TAXONOMY     : {taxonomy_name} ({taxonomy.n_classes} classes)")
    print(f"POSE_STYLE   : {pose_style}")
    print(f"SEQ_LEN      : {seq_len}")
    print(f"IN_CHANNELS  : {in_channels}")
    print(f"MODEL_NAME   : {model_name}")
    print(f"OUT_PATH     : {out_path}")

    task = Task(n_joints=17)
    task.prepare_loader(
        npy_collated_dir=bst_data_dir,
        pose_style=pose_style,
        batch_size=128,
    )
    task.get_network_architecture(
        model_name=model_name,
        seq_len=seq_len,
        in_channels=in_channels,
        taxonomy=taxonomy,
        n_active_classes=taxonomy.n_classes,
        active_class_list=list(taxonomy.classes),
    )
    task.load_weight(weight_path)
    preds = task.infer()

    np.save(out_path, preds.cpu().numpy())
    print(f"wrote {len(preds)} predictions to {out_path}")
    print(f"  unique predicted classes: {sorted(torch.unique(preds).tolist())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
