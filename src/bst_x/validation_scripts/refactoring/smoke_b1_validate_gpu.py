"""GPU device-discipline smoke for ``bst_x_train.validate()``.

Catches device-mismatch bugs in the per-class accumulators: silent on CPU,
RuntimeError on GPU. Option B keeps ``cum_tp`` / ``cum_fp`` / ``cum_fn`` on
the GPU during the loop and pulls them back with a single ``.cpu()`` after;
the per-batch ``cum_tp += batch_tp`` line is where a mismatch (cum_* on CPU,
batch_* on GPU) would crash. This script reproduces that line's conditions
with a synthetic two-batch loader and fails loud if it crashes.

No real data needed. Self-contained: a BST_PPF model + two synthetic batches.

Run on bourbaki / engelbart under venv-bst-x:
    cd ~/badminton_stroke_classification
    source ~/.venvs/venv-bst-x/bin/activate
    python src/bst_x/validation_scripts/refactoring/smoke_b1_validate_gpu.py

Expect (one line): OK: validate() ran on cuda with no device mismatch
Failure: RuntimeError on cum_tp += batch_tp, or any non-OK exit.

Originally landed as B1 in the simplification pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# script lives at src/bst_x/validation_scripts/refactoring/<name>.py;
# parents[2] is src/bst_x/, where bst_x_train and friends live.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from bst_x_train import validate  # noqa: E402
from model.bst import BST_PPF  # noqa: E402
from pipeline.config import taxonomy_lookup  # noqa: E402

N_CLASSES = taxonomy_lookup("une_v1_14").n_classes
B, T, N = 4, 100, 2
J_TOTAL = 17 + 19            # joints + bones per player
IN_DIM = J_TOTAL * 2         # validate() flattens (J, xy) -> in_dim = 72


def fake_loader(device):
    """Two batches, shapes per the live loader contract: human_pose is
    (b, t, n, J, xy) so validate()'s `.view(*shape[:-2], -1)` flattens to
    (b, t, n, in_dim); shuttle (b, t, 2); pos (b, t, n, 2); video_len (b,);
    labels (b,). Returned as a list so validate()'s `len(loader)` works."""
    torch.manual_seed(0)
    batches = []
    for _ in range(2):
        human_pose = torch.randn((B, T, N, J_TOTAL, 2), device=device)
        shuttle = torch.randn((B, T, 2), device=device)
        pos = torch.randn((B, T, N, 2), device=device)
        video_len = torch.tensor([T] * B, dtype=torch.long, device=device)
        labels = torch.randint(0, N_CLASSES, (B,), device=device)
        batches.append(((human_pose, pos, shuttle), video_len, labels))
    return batches


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: cuda not available; this smoke is GPU-only.")
        return 2
    device = torch.device("cuda")
    torch.manual_seed(2)
    model = BST_PPF(
        in_dim=IN_DIM, seq_len=T, n_class=N_CLASSES, d_model=100,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    try:
        val_loss, f1_avg, f1_min, f1_per_class, present, acc, top2 = validate(
            model=model,
            loss_fn=loss_fn,
            loader=fake_loader(device),
            device=device,
            n_classes=N_CLASSES,
        )
    except RuntimeError as e:
        print(f"FAIL: validate() crashed on cuda: {e}")
        return 1
    if not (f1_per_class.is_floating_point() and present.dtype == torch.bool):
        print(f"FAIL: unexpected dtypes; f1={f1_per_class.dtype}, present={present.dtype}")
        return 1
    print(f"OK: validate() ran on cuda with no device mismatch "
          f"(val_loss={val_loss:.4f}, acc={acc:.4f}, top2={top2:.4f}, "
          f"macro_f1={float(f1_avg):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
