"""BRIC training entry point.

See ``docs/bric_training_design.md`` for hyperparameters, manifest
schema, and CLI surface.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torcheval.metrics.functional import multiclass_f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from bric.dataset import ShuttleSetDataset, collate_strokes
from bric.network import BRICNetwork
from shared.taxonomy import DEFAULT_TAXONOMY, TAXONOMIES, Taxonomy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_DIR = _REPO_ROOT / 'training' / 'bric' / 'experiments'

_ARCHITECTURE = 'bric'
_CHECKPOINT_FILENAME = 'best.pt'

_VARIANTS: dict[str, tuple[bool, bool]] = {
    'rgb_only':           (False, False),
    'rgb_shuttle':        (True,  False),
    'rgb_court':          (False, True),
    'rgb_shuttle_court':  (True,  True),
}

_BST_CLASS_WEIGHTS: dict[str, float] = {'wrist_smash': 2.0, 'smash': 2.0}
_WEIGHTED_CE_LABEL_SMOOTHING = 0.15


def seed_everything(seed: int) -> None:
    """Seed CPU + CUDA RNGs and put cuDNN into deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**31)
    seed = (base + worker_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


class _ClipColorJitter:
    """Sample brightness / contrast / saturation / hue offsets once per clip and apply them to every frame."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, thwc: np.ndarray) -> np.ndarray:
        import torchvision.transforms.v2.functional as F

        bf = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        cf = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
        sf = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
        hf = random.uniform(-self.hue, self.hue)

        t = torch.from_numpy(thwc).permute(0, 3, 1, 2).contiguous()
        t = F.adjust_brightness(t, bf)
        t = F.adjust_contrast(t, cf)
        t = F.adjust_saturation(t, sf)
        t = F.adjust_hue(t, hf)
        return t.permute(0, 2, 3, 1).contiguous().numpy()


def _make_warmup_cosine_lambda(warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _build_loss(
    classes: list[str], weighted: bool, device: torch.device,
) -> nn.CrossEntropyLoss:
    if not weighted:
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    weights = torch.ones(len(classes), dtype=torch.float32)
    for stroke_name, w in _BST_CLASS_WEIGHTS.items():
        # Match bare names (nosides taxonomies) and 'Top_<name>' / 'Bottom_<name>' (sided taxonomies).
        for i, cls in enumerate(classes):
            if cls == stroke_name or cls.endswith(f'_{stroke_name}'):
                weights[i] = w
    return nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=_WEIGHTED_CE_LABEL_SMOOTHING,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _forward_for_variant(model: BRICNetwork, batch: dict[str, Any]) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if model.use_shuttle:
        kwargs['shuttle'] = batch['shuttle']
        kwargs['shuttle_length'] = batch['shuttle_length']
    if model.use_court:
        kwargs['court'] = batch['court']
    return model(batch['rgb'], **kwargs)


def train_one_epoch(
    model: BRICNetwork,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool,
    accumulate_steps: int = 1,
) -> float:
    """One training epoch with optional gradient accumulation.

    With ``accumulate_steps > 1``, the loss is divided by N and gradients
    accumulate across N forward-backward passes before each ``step()`` —
    matches the gradient magnitude of a single batch of size
    ``batch_size * accumulate_steps`` while keeping per-step VRAM bounded.
    Any leftover accumulation at end-of-epoch is flushed.
    """
    model.train()
    total_loss = 0.0
    total_n = 0
    pending_grads = False
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        batch = _move_batch(batch, device)
        labels = batch['label']
        with autocast_ctx:
            logits = _forward_for_variant(model, batch)
            loss = loss_fn(logits, labels)
        # bf16 matches fp32 dynamic range — no GradScaler needed.
        (loss / accumulate_steps).backward()
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        pending_grads = True
        if (i + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_grads = False
    if pending_grads:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(
    model: BRICNetwork,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool,
    n_classes: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    for batch in tqdm(loader, desc='val', leave=False):
        batch = _move_batch(batch, device)
        labels = batch['label']
        with autocast_ctx:
            logits = _forward_for_variant(model, batch)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_preds.append(logits.argmax(dim=1))
        all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    macro_f1 = multiclass_f1_score(
        preds, labels_t, num_classes=n_classes, average='macro',
    ).item()
    acc = (preds == labels_t).float().mean().item()
    return {
        'val_loss': total_loss / max(1, total_n),
        'val_macro_f1': float(macro_f1),
        'val_acc': float(acc),
    }


def _resolve_taxonomy(name: str) -> Taxonomy:
    if name not in TAXONOMIES:
        raise SystemExit(
            f'Unknown taxonomy {name!r}. Choose from {sorted(TAXONOMIES.keys())}'
        )
    return TAXONOMIES[name]


def _build_run_id(variant: str, seed: int) -> str:
    return f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{variant}_{seed}'


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_manifest(
    *,
    args: argparse.Namespace,
    use_shuttle: bool,
    use_court: bool,
    taxonomy: Taxonomy,
    classes: list[str],
    use_amp: bool,
    jitter_enabled: bool,
    backbone_lr: float,
    run_id: str,
    device: torch.device,
    started_at: str,
) -> dict[str, Any]:
    """Return the canonical manifest payload (schema in docs/bric_training_design.md)."""
    return {
        'architecture': _ARCHITECTURE,
        'checkpoint': _CHECKPOINT_FILENAME,
        'config': {
            'variant': args.variant,
            'use_shuttle': use_shuttle,
            'use_court': use_court,
            'taxonomy': taxonomy.name,
            'classes': classes,
        },
        'training': {
            'run_id': run_id,
            'started_at': started_at,
            'finished_at': None,
            'git_sha': _git_sha(),
            'seed': args.seed,
            'device': str(device),
            'best_epoch': None,
            'best_val_macro_f1': None,
            'overfit_n': args.overfit,
            'hparams': {
                'epochs': args.epochs,
                'warmup_epochs': args.warmup_epochs,
                'batch_size': args.batch_size,
                'accumulate_steps': args.accumulate_steps,
                'effective_batch_size': args.batch_size * args.accumulate_steps,
                'lr': args.lr,
                'backbone_lr': backbone_lr,
                'weight_decay': args.weight_decay,
                'weighted_ce': args.weighted_ce,
                'label_smoothing': (
                    _WEIGHTED_CE_LABEL_SMOOTHING if args.weighted_ce else 0.1
                ),
                'amp_dtype': 'bfloat16' if use_amp else 'fp32',
                'color_jitter': jitter_enabled,
            },
        },
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _append_metrics(path: Path, row: dict[str, Any], header: list[str]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = _select_device()
    use_shuttle, use_court = _VARIANTS[args.variant]
    taxonomy = _resolve_taxonomy(args.taxonomy)
    run_id = _build_run_id(args.variant, args.seed)

    experiment_dir = _EXPERIMENTS_DIR / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f'[bric.train] run_id={run_id}', flush=True)
    print(f'[bric.train] device={device} variant={args.variant} taxonomy={taxonomy.name}',
          flush=True)
    print(f'[bric.train] use_shuttle={use_shuttle} use_court={use_court}', flush=True)

    # Overfit mode forces jitter off so memorisation against the same N
    # samples is well-defined; per-call jitter would re-randomise every read.
    jitter_enabled = (not args.no_jitter) and (args.overfit is None)
    color_jitter = _ClipColorJitter() if jitter_enabled else None
    train_ds: ShuttleSetDataset | Subset = ShuttleSetDataset(
        split='train', taxonomy=taxonomy, rgb_transform=color_jitter,
    )
    val_ds: ShuttleSetDataset | Subset = ShuttleSetDataset(
        split='val', taxonomy=taxonomy, rgb_transform=None,
    )

    if args.overfit is not None:
        n = min(args.overfit, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = train_ds
        print(f'[bric.train] OVERFIT MODE n={n} (jitter forced OFF)', flush=True)

    print(f'[bric.train] train n={len(train_ds)}  val n={len(val_ds)}', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        collate_fn=collate_strokes, worker_init_fn=_worker_init_fn,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        collate_fn=collate_strokes, worker_init_fn=_worker_init_fn,
        persistent_workers=(args.workers > 0),
    )

    model = BRICNetwork(
        taxonomy=taxonomy, pretrained=True,
        use_shuttle=use_shuttle, use_court=use_court,
    ).to(device)

    # Discriminative LRs: pretrained backbone gets a smaller LR than the
    # randomly-initialised auxiliary encoders / norms / classifier. The
    # warmup-cosine schedule below applies its multiplier to each group's
    # base LR independently.
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = AdamW(
        [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': other_params, 'lr': args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_make_warmup_cosine_lambda(args.warmup_epochs, args.epochs),
    )
    classes = taxonomy.trainable_class_list()
    loss_fn = _build_loss(classes, weighted=args.weighted_ce, device=device)
    use_amp = (not args.no_amp) and device.type == 'cuda'

    manifest = _build_manifest(
        args=args,
        use_shuttle=use_shuttle, use_court=use_court,
        taxonomy=taxonomy, classes=classes,
        use_amp=use_amp, jitter_enabled=jitter_enabled,
        backbone_lr=backbone_lr,
        run_id=run_id, device=device,
        started_at=datetime.now().isoformat(timespec='seconds'),
    )
    _write_manifest(experiment_dir / 'manifest.yaml', manifest)

    metrics_path = experiment_dir / 'metrics.csv'
    metrics_header = [
        'epoch', 'lr', 'train_loss', 'val_loss', 'val_macro_f1', 'val_acc',
        'epoch_seconds',
    ]
    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, use_amp,
            accumulate_steps=args.accumulate_steps,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, use_amp, len(classes),
        )
        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        row = {
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_macro_f1': val_metrics['val_macro_f1'],
            'val_acc': val_metrics['val_acc'],
            'epoch_seconds': epoch_seconds,
        }
        _append_metrics(metrics_path, row, metrics_header)
        print(
            f'[ep {epoch:3d}] lr={lr:.2e} train_loss={train_loss:.4f} '
            f'val_loss={val_metrics["val_loss"]:.4f} '
            f'val_macro_f1={val_metrics["val_macro_f1"]:.4f} '
            f'val_acc={val_metrics["val_acc"]:.4f} '
            f'({epoch_seconds:.1f}s)',
            flush=True,
        )

        if val_metrics['val_macro_f1'] > best_f1:
            best_f1 = val_metrics['val_macro_f1']
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_macro_f1': best_f1,
                    'variant': args.variant,
                    'use_shuttle': use_shuttle,
                    'use_court': use_court,
                    'taxonomy': taxonomy.name,
                    'classes': classes,
                    'run_id': run_id,
                },
                experiment_dir / _CHECKPOINT_FILENAME,
            )

    manifest['training']['finished_at'] = datetime.now().isoformat(timespec='seconds')
    manifest['training']['best_val_macro_f1'] = best_f1
    manifest['training']['best_epoch'] = best_epoch
    _write_manifest(experiment_dir / 'manifest.yaml', manifest)

    print(
        f'[bric.train] DONE run_id={run_id} best_val_macro_f1={best_f1:.4f} '
        f'@ epoch {best_epoch}',
        flush=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--variant', required=True, choices=sorted(_VARIANTS.keys()),
        help='Selects which encoders the network instantiates.',
    )
    p.add_argument(
        '--taxonomy', default=DEFAULT_TAXONOMY,
        help=f'Taxonomy name (default: {DEFAULT_TAXONOMY}).',
    )
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument(
        '--accumulate-steps', type=int, default=1,
        help='Gradient accumulation steps. Effective batch = batch_size * accumulate_steps.',
    )
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument(
        '--backbone-lr', type=float, default=None,
        help='Learning rate for the pretrained R(2+1)D backbone. Defaults to lr/10.',
    )
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument(
        '--weighted-ce', action='store_true',
        help='{wrist_smash:2.0, smash:2.0} weighting + LS=0.15.',
    )
    p.add_argument(
        '--no-amp', action='store_true',
        help='Disable bf16 autocast (default: enabled on CUDA).',
    )
    p.add_argument(
        '--no-jitter', action='store_true',
        help='Disable training-split color jitter.',
    )
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--overfit', type=int, default=None, metavar='N',
        help='Train + eval on the same first N samples; loss should drop near zero.',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_training(args)


if __name__ == '__main__':
    main(sys.argv[1:])
