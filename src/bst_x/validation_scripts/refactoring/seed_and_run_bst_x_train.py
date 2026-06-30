"""Launcher that seeds torch + enables CUDA deterministic mode, then runs
``bst_x_train`` with the args it would normally take.

Used as the front half of the real-data bit-exact use case: the live training
path is unseeded by design, so a cross-branch identical comparison needs the
RNG pinned in the comparison harness. This wrapper does that without modifying
``bst_x_train.py``. Pair with ``compare_b7_real_runs.py`` for the back half
(diff two run dirs).

Pass all the bst_x_train CLI flags through after the launcher path. The
``BST_X_SEED`` env var overrides the default seed of 0.

    PYTHONPATH=src/bst_x \\
    BST_X_SEED=0 python src/bst_x/validation_scripts/refactoring/seed_and_run_bst_x_train.py \\
        --serial-no 1 \\
        --run-id seed_main \\
        --log-path /tmp/seed_main.log \\
        --taxonomy une_v1_14 --split-column split_v2 \\
        --collation-id b4_diff

Originally landed as the launcher for the B2+B4+B7 real-data bit-exact in the
simplification pass.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

# script lives at src/bst_x/validation_scripts/refactoring/<name>.py;
# parents[2] is src/bst_x/, the package root runpy.run_module needs on sys.path.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))


def main() -> int:
    seed = int(os.environ.get("BST_X_SEED", "0"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(
        f"[seed_and_run] torch.manual_seed({seed}); "
        f"cuda_available={torch.cuda.is_available()}; "
        f"deterministic_algorithms=True; "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}",
        flush=True,
    )

    # bst_x_train inspects sys.argv via argparse; drop our launcher's argv[0]
    # so it sees its expected program name.
    sys.argv = ["bst_x_train", *sys.argv[1:]]
    runpy.run_module("bst_x_train", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
