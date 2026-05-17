# BRIC training

R(2+1)D-18 RGB stroke classifier with optional fusion of shuttle
trajectory and striker court position. This document describes the
training surface implemented by `src/bric/{network,dataset,train}.py`.

---

## Overview

A single training run produces one model targeting one taxonomy and
one fusion variant. The training entry point is
`python -m bric.train --variant <name>`. Each run writes a
self-contained directory under `training/bric/experiments/<run_id>/`
that doubles as the deployable model package — see
[Run artefacts](#run-artefacts) and [`runtime/README.md`](../runtime/README.md)
for the deployment workflow.

Source data is the ShuttleSet release under
`training/data/shuttleset/`, preprocessed into per-source-video NPZ
caches and per-stroke RGB tensors by `scripts/bric/preprocess_videos.py`
and `scripts/bric/extract_shuttle.py`. The dataset reads from these
caches; no decoding or perception happens at training time.

---

## Inputs

`bric.dataset.ShuttleSetDataset` returns one dict per stroke:

| Field | Shape | Notes |
|-------|-------|-------|
| `rgb` | `(3, 32, 224, 224)` float32 | Channels-first, Kinetics-400 normalised. 32 frames centred on `target_frame`, fixed striker crop expanded by 50% and square-padded to 224×224. |
| `shuttle` | `(T_var, 5)` float32 | Per-frame `[x_norm, y_norm, visibility, dx, dy]` over the stroke's `[shuttle_start_f, shuttle_end_f)` window. `dx`/`dy` are forward differences zeroed across visibility transitions. Variable length per stroke — use `bric.dataset.collate_strokes` as the DataLoader's `collate_fn`. |
| `shuttle_length` | scalar int | Unpadded length; the network's `forward` requires it whenever `shuttle` is supplied. |
| `court` | `(3,)` float32 | `[x_norm, y_norm, valid]` at `target_frame`. `valid ∈ {0.0, 1.0}`. Invariant: `valid == 0.0` ⇒ `x == y == 0.0`. Note the inverse does not hold — `(0, 0, 1.0)` is a legal court-origin position, so consumers must gate on `valid`, not on the coordinates. |
| `label` | scalar long | Class index into the active taxonomy. |
| `clip_stem` | str | Stroke identifier for diagnostics. |

The shuttle window is bounded by adjacent strokes in the rally at
training time (set by the upstream annotation). The bounded window
preserves stroke-duration signal while preventing trajectory leakage
from the next shot.

---

## Architecture

Three parallel encoder lanes feed a fusion classifier. The shuttle and
court lanes are instantiated only when their variant flag is enabled.

| Lane | Input | Encoder | Output |
|------|-------|---------|--------|
| RGB | `(B, 3, 32, 224, 224)` | R(2+1)D-18, Kinetics-400 pretrained, `fc` replaced with `nn.Identity()` | `(B, 512)` |
| Shuttle *(optional)* | `(B, T_max, 5)` + `(B,)` lengths | per-frame MLP `(5 → 64)`, then length-masked mean-pool | `(B, 64)` |
| Court *(optional)* | `(B, 3)` | MLP `(3 → 64)` | `(B, 64)` |

Each present lane's output passes through a per-lane `nn.LayerNorm`
before concatenation. The norms align feature magnitudes across the
pretrained backbone (post-ReLU Kinetics-trained scale) and the
randomly-initialised auxiliary MLPs, preventing one stream from
dominating the gradient through the linear classifier in early epochs.

```
rgb_feat     = LayerNorm(rgb_dim=512)(backbone(rgb))
shuttle_feat = LayerNorm(shuttle_dim=64)(shuttle_encoder(shuttle, lengths))   # if enabled
court_feat   = LayerNorm(court_dim=64)(court_encoder(court))                  # if enabled

fused  = concat(rgb_feat, [shuttle_feat], [court_feat])   # along dim=1
logits = Linear(fusion_dim, n_classes)(fused)             # (B, n_classes)
```

`fusion_dim` is `512 + 64·use_shuttle + 64·use_court` — see
[Variants](#variants) for the four supported combinations.

The R(2+1)D-18 `fc` head is replaced with `nn.Identity()`; the 512-dim
penultimate feature feeds the per-lane LayerNorm and then the fusion
classifier.

The shuttle encoder builds its mask from the per-sample length tensor
so right-padded positions contribute zero to the temporal mean. Without
masking, `frame_mlp(0)` produces a non-zero bias-dominated value and
would shift the pooled feature toward padding noise.

The court encoder operates on a single snapshot at `target_frame`. The
`valid` channel lets the encoder distinguish "striker is at court
origin" from "no striker bbox available" — without it, both inputs
look identical.

---

## Variants

Each variant is a distinct `BRICNetwork` instance with its own
weights. Training one variant does not produce checkpoints for any
other; the ablation isolates each modality's contribution rather than
the network's robustness to a missing input.

| Variant | `use_shuttle` | `use_court` | Fusion dim |
|---------|---------------|-------------|------------|
| `rgb_only` | False | False | 512 |
| `rgb_shuttle` | True | False | 576 |
| `rgb_court` | False | True | 576 |
| `rgb_shuttle_court` | True | True | 640 |

Disabled-modality encoders are not instantiated. The variant name and
the constructor flags are written into `manifest.yaml` so checkpoints
self-identify.

---

## Loss

Default: `nn.CrossEntropyLoss(label_smoothing=0.1)` with no class
weights.

Optional: `--weighted-ce` enables per-class weighting
`{wrist_smash: 2.0, smash: 2.0, all others: 1.0}` with
`label_smoothing=0.15`. The weighted preset matches a sweep-validated
configuration on the same dataset under a different architecture; it
is exposed as a flag rather than the default so a baseline run
isolates the architectural contribution from the loss configuration.

---

## Augmentation

Per-clip color jitter on the training split only:
`brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1`. All 32
frames of a stroke receive the same jitter parameters; per-frame
jitter would corrupt the temporal coherence the 3D convolutions
learn from.

Validation and test splits receive no augmentation. Spatial
augmentation (crop, flip) is not applied because player crops are
positioned by a fixed bbox at `target_frame`; spatial transforms
would either move the player out of frame or invert the court
orientation that the court encoder relies on.

`--no-jitter` disables augmentation. Overfit-mode (`--overfit N`)
forces it off so memorisation against the same N samples is
well-defined.

---

## Optimization

| Knob | Value |
|------|-------|
| Optimizer | AdamW |
| Backbone learning rate | `lr / 10` (default 5e-5) |
| Other-layer learning rate | 5e-4 (auxiliary encoders, LayerNorms, classifier) |
| Weight decay | 1e-4 (shared across groups) |
| Schedule | Linear warmup (5 epochs) → cosine decay; multiplier applied to each group's base LR |
| Epochs | 50 |
| Batch size | 32 |
| Gradient accumulation | 1 (effective batch = `batch_size × accumulate_steps`) |

The full backbone is fine-tuned end-to-end; no layers are frozen. The
ShuttleSet domain (fixed-camera badminton broadcast, sub-second stroke
windows) differs sufficiently from Kinetics-400 that frozen-backbone
linear probing leaves accuracy on the table.

Discriminative learning rates pair the pretrained backbone (small LR,
gentle adaptation) with the randomly-initialised auxiliary encoders,
LayerNorms, and classifier (larger LR, learning from scratch). Both
groups share the same warmup-cosine schedule, applied multiplicatively
to each group's base LR.

Gradient accumulation (`--accumulate-steps N`) divides the loss by `N`
and steps the optimizer every `N`-th forward-backward pass, matching
the gradient magnitude of a single batch of size `batch_size × N`
while keeping per-step VRAM bounded. Defaults to `1` (no accumulation);
useful for reproducing on hardware below the training host's memory
budget.

---

## Mixed precision

`torch.amp.autocast(dtype=torch.bfloat16)` is enabled by default on
CUDA. bfloat16's 8-bit exponent matches fp32 dynamic range, so no
`GradScaler` is required. Optimizer state stays in fp32; activations,
forward, and backward run in bf16.

`--no-amp` disables autocast for debugging. AMP is also disabled
automatically on non-CUDA devices.

---

## Reproducibility

Each run sets:

- `random.seed(seed)`, `numpy.random.seed(seed)`,
  `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`.
- `torch.backends.cudnn.benchmark = False`.
- `torch.backends.cudnn.deterministic = True`.
- DataLoader `worker_init_fn` re-seeds each worker from
  `torch.initial_seed()`.

`torch.use_deterministic_algorithms(True)` is not enabled — the
slowdown and per-op opt-out cost outweigh the marginal determinism
gain over `cudnn.deterministic` on conv-heavy networks.

The seed is a CLI argument with default 42. Seed and git SHA are
written to `manifest.yaml`. Without `cudnn.deterministic`, conv-backward
atomic adds and autotuner kernel selection can drift same-seed runs by
0.5–2% F1 on R(2+1)D-18 — within the range of small ablation deltas.

Single-seed methodology: report deltas <1% F1 as ties to absorb
residual numeric drift.

---

## Hyperparameters

| Knob | Value |
|------|-------|
| Optimizer | AdamW |
| Learning rate (other layers) | 5e-4 |
| Learning rate (backbone) | `lr / 10` (default 5e-5; `--backbone-lr` overrides) |
| Weight decay | 1e-4 |
| Schedule | Linear warmup (5 ep) → cosine decay |
| Epochs | 50 |
| Batch size | 32 |
| Gradient accumulation | 1 (`--accumulate-steps N` for `effective_batch = N × batch_size`) |
| Loss | CE + LS=0.1 (`--weighted-ce` for the BST-style ablation) |
| Mixed precision | bf16 AMP on CUDA (`--no-amp` to disable) |
| Backbone | R(2+1)D-18, Kinetics-400 pretrained, end-to-end fine-tune |
| Shuttle encoder | per-frame MLP (5→64) + length-masked mean pool |
| Court encoder | MLP (3→64) |
| Per-lane norm | LayerNorm on each lane output before concat |
| Fusion | concat → Linear(fusion_dim, n_classes) |
| Color jitter | brightness/contrast/saturation=0.4, hue=0.1; train split only |
| Seed | 42; cuDNN deterministic, benchmark off |
| Best-checkpoint criterion | Val macro F1 |

---

## Run artefacts

```
training/bric/experiments/<run_id>/
├── manifest.yaml      deployment + training metadata (schema below)
├── metrics.csv        per-epoch: epoch, lr, train_loss, val_loss,
│                                  val_macro_f1, val_acc, epoch_seconds
└── best.pt            best-on-val-macro-F1 state_dict + variant metadata
```

`<run_id>` is `<YYYYmmdd_HHMMSS>_<variant>_<seed>`.

The directory is also the deployment unit. Promotion to the API is
described in [`runtime/README.md`](../runtime/README.md).

`manifest.yaml` schema:

```yaml
architecture: bric
checkpoint: best.pt
config:
  variant: rgb_shuttle
  use_shuttle: true
  use_court: false
  taxonomy: une_merge_v1_nosides
  classes: [net_shot, return_net, smash, wrist_smash, ...]
training:
  run_id: 20260516_193000_rgb_shuttle_42
  started_at: 2026-05-16T19:30:00
  finished_at: 2026-05-17T08:14:00
  git_sha: abc1234
  seed: 42
  device: cuda
  best_epoch: 37
  best_val_macro_f1: 0.652
  hparams:
    epochs: 50
    warmup_epochs: 5
    batch_size: 32
    accumulate_steps: 1
    effective_batch_size: 32
    lr: 5.0e-4
    backbone_lr: 5.0e-5
    weight_decay: 1.0e-4
    weighted_ce: false
    label_smoothing: 0.1
    amp_dtype: bfloat16
    color_jitter: true
```

`config` is the minimum the API needs to load and serve the model.
`training` is descriptive metadata for run tracking; the API ignores
it.

---

## CLI

```
python -m bric.train --variant {rgb_only,rgb_shuttle,rgb_court,rgb_shuttle_court}
                     [--taxonomy NAME]
                     [--epochs 50] [--warmup-epochs 5]
                     [--batch-size 32] [--accumulate-steps 1]
                     [--lr 5e-4] [--backbone-lr 5e-5] [--weight-decay 1e-4]
                     [--weighted-ce] [--no-amp] [--no-jitter]
                     [--workers 4] [--seed 42]
                     [--overfit N]
```

`--overfit N` trains and evaluates on the same first N samples of the
train split. Loss should drop near zero and accuracy reach 100% by
the end of the run; if not, the gradient path is misconfigured.
