"""Seeded equivalence smoke for ``train_network``.

Pins the RNG, runs a 2-epoch synthetic train, captures
``(state_dict_sha256, val_at_best)``. Re-run on a reference state to capture
the golden, re-run on the proposed change to verify. The live training path is
unseeded by design, so a cross-branch bit-exact comparison needs the seed
pinned before the build+train sequence. On CPU with ``num_workers=0`` a single
``torch.manual_seed(0)`` before that sequence fully determines the run; the
captured pair is bit-stable across processes too.

Synthetic dataset (no fixtures): a small ``Dataset_npy_collated``-shaped
in-memory dataset that matches the live loader contract:
``((human_pose[t,m,J+B,2], pos[t,m,2], shuttle[t,2]), videos_len, label)``,
``m=2``, ``J+B=36`` for JnB_bone.

Usage:
    # On the reference state, capture the golden:
    PYTHONPATH=src/bst_x ~/.venvs/badminton-cicd/bin/python \\
        src/bst_x/validation_scripts/refactoring/smoke_b7_seeded_train.py \\
        capture --out /tmp/seeded_train_golden.pt

    # On the proposed-change state, verify:
    PYTHONPATH=src/bst_x ~/.venvs/badminton-cicd/bin/python \\
        src/bst_x/validation_scripts/refactoring/smoke_b7_seeded_train.py \\
        check --golden /tmp/seeded_train_golden.pt

Expect ``OK: state_dict_sha256 + val_at_best match the captured main golden``.

Force ``--device cpu`` on a Maxwell-or-older laptop GPU: sm_50 CUDA ops error
out even though ``torch.cuda.is_available()`` returns True. ``--device cuda``
enables ``torch.use_deterministic_algorithms`` + a ``CUBLAS_WORKSPACE_CONFIG``
pin so the cross-process compare stays meaningful on HPC.

Originally landed as B7 in the simplification pass.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# script lives at src/bst_x/validation_scripts/refactoring/<name>.py;
# parents[2] is src/bst_x/.
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import bst_x_train as t  # noqa: E402
from model.bst import BST_PPF  # noqa: E402
from pipeline.config import resolve_taxonomy  # noqa: E402
from preparing_data.shuttleset_dataset import get_bone_pairs  # noqa: E402

SEED = 0
N_EPOCHS = 2
SEQ_LEN = 30
BATCH_SIZE = 8
N_TRAIN, N_VAL = 16, 12
TAX_NAME = "une_v1_14"
N_PLAYERS = 2
N_JOINTS = 17
N_BONES = 19  # JnB_bone = 17 joints + 19 bones
J_PLUS_B = N_JOINTS + N_BONES  # 36


class SyntheticDataset(Dataset):
    """Tuple contract per shuttleset_dataset.py:269-271; ``clip_stems`` attr so
    Task.dump_predictions' hard-fail-on-None assert doesn't fire (not used here,
    smoke only runs train_network, but kept for parity with the live shape)."""

    def __init__(self, n_clips: int, n_classes: int, rng: torch.Generator):
        self.human_pose = torch.randn(
            (n_clips, SEQ_LEN, N_PLAYERS, J_PLUS_B, 2), generator=rng,
        )
        self.pos = torch.randn((n_clips, SEQ_LEN, N_PLAYERS, 2), generator=rng)
        self.shuttle = torch.randn((n_clips, SEQ_LEN, 2), generator=rng)
        self.videos_len = torch.full(
            (n_clips,), SEQ_LEN, dtype=torch.long,
        )
        self.labels = torch.randint(
            0, n_classes, (n_clips,), generator=rng,
        )
        self.clip_stems = [f"smoke_{i}" for i in range(n_clips)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            (self.human_pose[i], self.pos[i], self.shuttle[i]),
            self.videos_len[i],
            self.labels[i],
        )


def _hash_state_dict(sd: dict) -> str:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def run_seeded_train(tmp_dir: Path, device: torch.device) -> tuple[str, dict | None]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        # cuda kernel nondeterminism is the only pass-to-pass wobble we can't
        # eliminate via Python-level seeding; pin it for the smoke so the gate
        # stays meaningful when comparing across processes on HPC.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    taxonomy = resolve_taxonomy(TAX_NAME)
    n_classes = taxonomy.n_classes
    class_ls = list(taxonomy.classes)
    n_bones_pairs = len(get_bone_pairs("coco"))  # 19

    # pin hyp for the smoke (n_epochs=2, seq_len=30, small batch, no early stop)
    t.hyp = t.hyp._replace(
        n_epochs=N_EPOCHS,
        early_stop_n_epochs=100,
        warm_up_step=2,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # build dataset + loaders with a SEPARATE seeded generator so the global
    # torch RNG (which the loader shuffle + augs + dropout consume) starts at a
    # known state when train_network runs
    rng = torch.Generator().manual_seed(SEED + 1)
    train_ds = SyntheticDataset(N_TRAIN, n_classes, rng)
    val_ds = SyntheticDataset(N_VAL, n_classes, rng)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Re-seed the global RNG immediately before model build so dropout init,
    # weight init, and every per-batch draw all flow from the same starting
    # state.
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = BST_PPF(
        in_dim=J_PLUS_B * 2, seq_len=SEQ_LEN, n_class=n_classes, d_model=100,
    ).to(device)

    save_path = tmp_dir / "best.pt"
    tb_dir = tmp_dir / "tb"
    model_out, val_at_best = t.train_network(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=save_path,
        n_bones=n_bones_pairs,
        n_classes=n_classes,
        class_ls=class_ls,
        taxonomy=taxonomy,
        tb_dir=tb_dir,
    )
    sd = model_out.state_dict()
    if device.type == "cuda":
        # torch.save embeds the device in serialised cuda tensors, so hash a
        # CPU-copied OrderedDict (preserving type so the CPU sha matches the
        # original golden when both are run with --device cpu).
        sd = type(sd)((k, v.cpu()) for k, v in sd.items())
    sha = _hash_state_dict(sd)
    return sha, val_at_best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "check"))
    parser.add_argument("--out", type=Path, help="capture: where to write the golden")
    parser.add_argument("--golden", type=Path, help="check: golden path to compare against")
    parser.add_argument(
        "--device", default="cpu",
        help="cpu (laptop default) or cuda (HPC). cuda enables deterministic mode.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda but cuda not available")
        return 2

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sha, val = run_seeded_train(Path(tmp), device)

    if args.mode == "capture":
        if args.out is None:
            print("capture mode requires --out")
            return 2
        torch.save({"state_dict_sha256": sha, "val_at_best": val}, args.out)
        print(f"captured (state_dict_sha256, val_at_best) -> {args.out}")
        print(f"  state_dict_sha256={sha}")
        print(f"  val_at_best={val}")
        return 0

    if args.golden is None:
        print("check mode requires --golden")
        return 2
    if not args.golden.exists():
        print(f"no golden at {args.golden}; run capture on the reference state first")
        return 2
    golden = torch.load(args.golden, weights_only=False)
    problems = []
    if sha != golden["state_dict_sha256"]:
        problems.append(f"state_dict_sha256: {sha} != {golden['state_dict_sha256']}")
    if val != golden["val_at_best"]:
        problems.append(
            f"val_at_best: {val} != {golden['val_at_best']}"
        )
    if problems:
        print("FAIL: seeded equivalence broken")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print(
        f"OK: state_dict_sha256 + val_at_best match the captured golden "
        f"(sha={sha[:16]}..., val_at_best epoch={val['epoch'] if val else None})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
