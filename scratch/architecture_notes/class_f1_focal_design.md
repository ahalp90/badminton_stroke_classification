# Class-F1-driven adaptive focal loss: design

Design doc for the class-F1-driven focal loss arm queued behind the
class-weighting smoke test + manually-alpha focal cells. Output of the
research prompt at `scratch/architecture_notes/class_f1_focal_exploration_prompt.md`.
Recommends the variant, the implementation surface in `bst_train.py`,
and the gate condition for "this worked".

---

## 1. Chosen variant

**CDB-loss (Sinha, Ohashi, Nakamura, ACCV 2020 / IJCV 2022) with the
difficulty signal swapped from per-class accuracy to per-class running
train F1, optionally composed with focal's per-sample focusing term.**

### What it actually does, in plain English

At the end of every training epoch, look at how well the model is
performing *per class*. Compute one number per class that says
"how much room is left to improve here" — that's just `1 - F1_c`,
where F1_c is the running F1 score on class `c`. A class with F1 =
0.9 has 0.1 room to improve; a class with F1 = 0.4 has 0.6.

Use that "room to improve" number as the loss multiplier for that
class. Whenever a training sample's true label is class `c`, scale
its loss term by class `c`'s current multiplier. Classes the model
struggles on get a bigger multiplier (more gradient pressure);
classes the model has nailed get a smaller one (less gradient
pressure, since they don't need it). Renormalise the multipliers
so the average across classes stays at 1.0 — that keeps the overall
loss scale comparable to plain cross-entropy, and only the *shape*
of which class matters more shifts.

The exponent `tau` controls how aggressively to amplify the gap
between best and worst classes. `tau = 1` is "use the gap as-is".
`tau = 2` is "square the gap" — the worst class gets disproportionately
more weight, the best class gets disproportionately less. `tau = 0.5`
is the gentle version. We default to `tau = 1.0`.

The optional focal composition adds a *per-sample* term on top of
the per-class one. For each individual training sample, look at how
confident the model already is in the correct class (`p_t`, the
softmax probability assigned to the true label). If it's already
close to 1 (the model is confident and correct), scale that
sample's loss down — gradient there is "wasted" because the model
already gets it. If it's low (the model is unconfident or wrong),
keep the loss at full weight. This is just standard focal loss
(Lin et al. 2017): the `(1 - p_t)^gamma` term. We default to
`gamma = 1.0`, the gentle setting.

The two layers compose cleanly: per-class weight chooses *which
classes* get more focus; focal chooses *which samples within each
class* get more focus. CDB authors explicitly note that their
`tau` plays a similar role at the class level to focal's `gamma`
at the sample level — same idea, different axis.

### Citations

- Sinha et al., "Class-Wise Difficulty-Balanced Loss for Solving
  Class-Imbalance", ACCV 2020. [openaccess link](https://openaccess.thecvf.com/content/ACCV2020/papers/Sinha_Class-Wise_Difficulty-Balanced_Loss_for_Solving_Class-Imbalance_ACCV_2020_paper.pdf)
- Sinha et al., "Class-Difficulty Based Methods for Long-Tailed Visual
  Recognition", IJCV 2022. [arXiv:2207.14499](https://arxiv.org/abs/2207.14499)
- Reference implementation (Hitachi R&D): https://github.com/hitachi-rd-cv/CDB-loss
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
  (the `(1 - p_t)^gamma` term, eq. 5).

### Reference equations (for completeness)

Direct from the ACCV 2020 paper:

- per-class difficulty `d_c = 1 - A_c`, where `A_c` is per-class
  accuracy on a held-out signal (eq. 1).
- per-class weight `w_c = (1 - A_c)^tau = d_c^tau` (eq. 2).
- weighted softmax CE: `L_CDB = - w_{c,t} * log(p_{c,t})` for the true
  class `c` of sample `t` (eq. 7).
- dynamic `tau_t = 2 / (1 + exp(-b_t))` with bias
  `b_t = (max_c A_c) / (min_c A_c + eps) - 1` (eqs. 3-4). We default
  to fixed `tau = 1.0`; dynamic `tau` is a follow-up cell.

### What we change vs the published version

One substitution: per-class running **train F1** instead of per-class
**val accuracy**. So the weight becomes `w_c = (1 - F1_c)^tau`, with
`F1_c` accumulated on the train loader during each epoch's forward
pass (no extra forward pass, just a TP/FP/FN counter that gets
updated alongside the existing loss accumulator).

Why F1 instead of accuracy:
- **Val cleanliness.** Val is ~5% of clips and we use it for early
  stopping. Pulling per-class signal off it for the loss update
  would couple the two and contaminate the early-stopping criterion.
  Train signal stays inside the training loop where it belongs.
- **F1 catches over-prediction, accuracy doesn't.** If the model
  starts spamming "wrist_smash" at everything in an attempt to lift
  wrist_smash recall, accuracy on wrist_smash goes up (more true
  positives), but precision crashes (lots of false positives). F1
  is the harmonic mean and crashes too. Accuracy alone would
  encourage exactly the cheap-path failure mode we want to avoid.
- **They coincide on the actual bottleneck.** When a class is just
  flat-out being missed (the wrist_smash regime), F1 ≈ accuracy ≈
  low. Both signals would say the same thing; F1 is the safer
  choice for the cases where they'd disagree.

### Reported lift in published work

Verified against the ACCV 2020 paper directly (PDF text extracted
2026-05-01).

Closest analog to combo A is EGTEA Gaze+ (egocentric video action
recognition, 19-class with a 5/14 majority/minority split,
count-vs-difficulty decoupled). CDB-CE vs unweighted CE on the
minority classes (paper **table 7**, not table 5):

|                | Recall | Precision |
|----------------|--------|-----------|
| Unweighted CE  | 61.14  | 56.75     |
| Focal loss     | 55.21  | 53.40     |
| CDB-CE (dyn τ) | 63.31  | 60.42     |

So vs unweighted CE: minority recall **+2.17 pp**, minority
precision **+3.67 pp**. (The headline-friendly "+8.1 pp recall"
number is the lift vs **focal loss baseline**, not vs unweighted
CE — focal underperforms unweighted CE on EGTEA's minority
classes, so CDB recovers that gap and then some.)

On CIFAR-100-LT IF=100 (paper table 3): focal loss 38.41 → CDB-CE
**42.57** (+4.16 pp vs focal; +4.36 pp vs unweighted CE which
sits at 38.21 in table 4 row τ=0).

**Important caveat on the published τ.** Both the EGTEA result
(table 6 caption: "we use dynamically updated τ") and the
headline CIFAR-100-LT result (table 3 caption: "for CDB-CE loss
(Ours), we report the results with dynamically updated τ") are
achieved with the dynamic-τ formula (eqs. 3-4), not fixed τ. Our
design defaults to **fixed τ=1.0**, which on CIFAR-100-LT IF=100
gives 41.67 (table 4) vs 42.57 dynamic — a 0.9 pp shortfall. Fixed
τ is still a clear lift over uniform CE / focal, but the
"published lift" numbers above are the dynamic-τ numbers, not what
fixed τ=1.0 buys directly. If our fixed-τ smoke run shows a
partial lift, escalating to dynamic τ (an additional ~20 LOC in
`update_alpha`) is the natural follow-up.

## 2. Why this variant fits combo A

The bottleneck on `une_merge_v1_nosides + split_v2 + dropunk` is
wrist_smash: 979 train clips (5th-rarest of 14 classes), F1 ~0.38
(bottom of the 14). The rarest class by count, long_service (252
clips), sits at F1 0.99. **Count and difficulty are decoupled on
this taxonomy.**

- **Count-based class-balanced losses miss.** Inverse-frequency,
  inverse-sqrt-frequency, and Cui et al. effective-number all assign
  long_service the largest weight and wrist_smash a mid-pack weight.
  They upweight saturated classes and barely move the actual
  bottleneck. Already analysed and rejected; CDB-style F1-driven
  weighting is the principled corrective.
- **CDB's design premise matches.** Sinha et al.'s motivating result
  (sec. 3 of the ACCV paper) is that the "long tail" of *difficulty*
  is not the same as the long tail of *count*. EGTEA's "Clean" and
  "Spread" classes are sparse but easy (analogous to long_service);
  EGTEA's frequent-but-confused classes are the bottleneck (analogous
  to wrist_smash, which is mid-count but bottom-F1). CDB-loss is the
  one published method built around this exact mismatch.
- **Wrist_smash↔smash is one pair, not a long tail.** CDB-loss is
  scalar-per-class, not pair-aware. Seesaw Loss (Wang et al. CVPR
  2021) is pair-aware but pure-count-driven, and reframing it to
  use F1 is unpublished. The pragmatic compromise: scalar-per-class
  CDB still upweights wrist_smash specifically (low F1 → high
  weight); the pair-confusion structure is left to the model's own
  capacity to disambiguate, given the extra capacity the upweight
  buys. If we get a partial lift but smash drops, the natural
  follow-up is to compose CDB with a manual pair-cap (see §8).
- **Label noise tolerance.** ShuttleSet annotation has non-trivial
  human label noise. F1-driven `w_c` and CE-style `(1 - F1_c)^tau`
  are bounded in [0, 1] and renormalise to mean 1.0, so a noisy
  class can't blow up into runaway alpha. Vanilla focal at high
  gamma is the noise-amplification risk; we keep gamma=1.0 default.

## 3. Stability mechanism

### The oscillation problem, in plain English

If we naively recompute the per-class weights from each epoch's raw
F1 reading, the loss function will chase its own tail:

1. wrist_smash F1 is 0.38 at end of epoch 10. Weight goes high.
2. Epoch 11 trains harder on wrist_smash because of the high weight.
   F1 lifts to 0.45.
3. End of epoch 11: weight drops because F1 lifted.
4. Epoch 12 trains less on wrist_smash. F1 falls back to 0.39.
5. Repeat.

The weight bounces around, the model never gets a stable signal,
and the run probably ends up worse than uniform CE.

### The fix: smooth the F1 signal before using it

Don't use the raw end-of-epoch F1 reading as the input to the
weight calculation. Use a rolling average of recent F1 readings,
biased toward more recent epochs. That way a single jumpy epoch
can't yank the weights around — only a sustained shift in F1 will
move them. This is the **EMA (exponential moving average)** trick,
exactly the same one PyTorch uses for BatchNorm running stats:

```
F1_running_c = momentum * F1_running_c + (1 - momentum) * F1_this_epoch_c
alpha_c = (1 - F1_running_c) ^ tau
alpha = alpha * (n_classes / sum(alpha))   # renormalise mean to 1.0
```

In words: each epoch, the running F1 estimate keeps 90% of its
previous value and absorbs 10% of the latest reading. Then map
that smoothed F1 to the per-class weight via the `tau` exponent,
and renormalise so the average weight across classes stays at 1.0.

### Why momentum = 0.9

- **The half-life works out to ~6.6 epochs.** That means the running
  F1 estimate "forgets" old readings at a rate where information
  from 7 epochs ago counts roughly half as much as today's reading.
  For an 80-epoch training budget, that's a sensible memory horizon
  — long enough to ignore single-epoch jitter, short enough to
  actually track real F1 trends as training progresses.
- **It matches familiar regimes.** PyTorch `BatchNorm.momentum`
  defaults to a 0.1 mix-in (which is the same as 0.9 momentum here
  under the BN sign convention). Adam's first-moment beta is 0.9.
  The value isn't magic, but it's the canonical "smooth a noisy
  online statistic" choice.
- **CDB original uses the cumulative average, which is similar.**
  The published CDB recomputes weights from the cumulative val
  accuracy across all epochs so far. That's mathematically a
  running average where each epoch counts equally. EMA at 0.9 is
  the cleaner online equivalent that biases recent epochs more.

### Why a 5-epoch warm-up

In epoch 1, the model is barely trained. Per-class F1 is mostly
near 0 across the board because the model is essentially random.
That would push every class's weight to `(1 - 0)^tau = 1`, which
after renormalisation is just uniform — fine in principle, but the
EMA needs a few epochs to build up a meaningful estimate before
the weight shape is informative.

So for the first 5 epochs, run uniform CE in the loss (alpha = all
ones). Meanwhile, the EMA is still updating in the background, so
by epoch 6 the running F1 estimate has absorbed 5 epochs of
training and the weights have shape. That shape then drives the
loss for the remaining 75 epochs.

## 4. Hyperparameter defaults

The loss has six knobs. In plain English:

- **`tau`** is the per-class aggressiveness dial. With `tau = 1`,
  a class with F1 = 0.4 gets a multiplier proportional to 0.6, and
  a class with F1 = 0.9 gets one proportional to 0.1. The ratio is
  6:1. With `tau = 2`, those become 0.36 vs 0.01 — a 36:1 ratio. So
  higher `tau` exaggerates the gap between best and worst classes;
  lower `tau` gentles it.
- **`gamma`** is the per-sample aggressiveness dial (the focal
  term). For each individual sample, `(1 - p_t)^gamma` scales its
  loss. A sample the model already gets right with `p_t = 0.9`:
  at `gamma = 1`, scale = 0.1; at `gamma = 2`, scale = 0.01. So
  higher `gamma` ignores easy samples more aggressively. For
  ShuttleSet's known label noise, we want low gamma — high gamma
  amplifies the gradient on noisy hard-to-classify samples, which
  is exactly the regime where label noise hurts most.
- **`momentum`** is the EMA smoothing on the running F1 signal
  (see §3). Higher momentum = smoother but slower to react;
  lower = jumpier but tracks recent epochs more closely.
- **`warm_up_epochs`** is how many epochs to run uniform CE at the
  start, before turning on the per-class weighting. Lets the EMA
  build a sensible F1 estimate first.
- **`update_freq`** is how often to recompute the per-class weights.
  Once per epoch is the canonical setting.
- **`f1_floor`** is a safety floor on the F1 input — if a class
  somehow has F1 = 0 for many epochs, the floor stops the weight
  from saturating at the absolute maximum. Default 0.0 (off);
  renormalisation already handles the saturation case in practice.

| Knob | Default | Range / fallbacks | Why |
|---|---|---|---|
| `tau` | 1.0 | 0.5 (gentler), 2.0 (more aggressive). CDB ACCV table 5 sweeps {0.5, 1, 1.5, 2}. | Linear difficulty-to-alpha map. Dynamic-tau (CDB eq. 4) is a follow-up cell, not the default. |
| `gamma` | 1.0 | 0 (skip focal, CDB-only), 2.0 (more focusing) | Conservative for ShuttleSet label noise. Vanilla focal canonical default is 2.0; we step down. |
| `momentum` | 0.9 | 0.8 (faster response), 0.95 (steadier) | See §3. |
| `warm_up_epochs` | 5 | 3 (short), 10 (long) | Long enough for the macro to stabilise; short enough to leave 75 epochs of adaptive training in an 80-epoch budget. |
| `update_freq` | every epoch | every 2 epochs (smoother) | Per-epoch is the canonical CDB rate. |
| `f1_floor` | 0.0 | 0.05 (light floor) | F1 is naturally bounded; renormalisation handles the "all-zero" edge case. Floor is unnecessary unless we observe runaway alpha. |
| `signal` | per-class train F1 | per-class train accuracy (CDB original) | F1 is a tighter signal under imbalance (catches over-prediction in addition to miss-rate). On the canonical failure mode (class never predicted), F1 and accuracy coincide. |

Active config the smoke cell would carry into `Hyp.adaptive_focal`:

```python
adaptive_focal={
    'tau': 1.0,
    'gamma': 1.0,
    'momentum': 0.9,
    'warm_up_epochs': 5,
    'update_freq': 1,
    'f1_floor': 0.0,
}
```

`label_smoothing` forced to 0.0 in the focal branch (LS softens
targets so even confident-correct samples have `p_t < 1.0`,
contaminating focal's hardness estimate — same constraint that
applies to all focal variants).

## 5. Implementation surface

### Plain-English overview of what changes

We need three new pieces and one branch addition:

1. **A new loss module** that holds the running F1 estimate and the
   per-class weights as buffers (so they persist across forward
   passes), exposes a `forward(logits, labels)` that looks like
   plain `nn.CrossEntropyLoss`, and exposes an `update_alpha(f1)`
   method the training loop calls at end-of-epoch.
2. **A per-class TP/FP/FN counter inside `train_one_epoch`** that
   accumulates as batches go past, then turns into per-class F1 at
   end-of-epoch. This is cheap — same `argmax` we'd use for
   accuracy, plus three `sum()` calls per class per batch. No
   second forward pass.
3. **A loss-build branch in `bst_train.py`** that picks adaptive
   focal when the new `Hyp.adaptive_focal` config field is set,
   keeps the existing class-weighted CE branch when only
   `Hyp.class_weights` is set, and falls back to uniform CE
   otherwise. The two new modes are mutually exclusive (raise if
   both are configured).
4. **Diagnostics**: per-epoch printout of top-3 / bot-3 alpha by
   class, plus per-class scalars to TensorBoard. Manifest
   serialisation already auto-handles the new Hyp field via
   `_asdict()` in `run_tracker.py:160`.

Two files touched:
1. **NEW** `src/bst_refactor/stroke_classification/main_on_shuttleset/loss/adaptive_focal.py`
   — ~120-line module. Skeleton below.
2. `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`
   — 5 small edits.

### 5a. `loss/adaptive_focal.py` skeleton

```python
"""
Class-F1-driven adaptive focal loss for combo A (une_merge_v1_nosides).

Implements CDB-loss (Sinha et al. ACCV 2020 / IJCV 2022) with the
per-class difficulty signal computed from running train F1 instead of
held-out val accuracy, optionally composed with focal's (1 - p_t)^gamma
per-sample focusing. See scratch/architecture_notes/class_f1_focal_design.md
for the full motivation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive focal loss with per-class alpha driven by running train F1.

    Forward signature matches nn.CrossEntropyLoss(reduction='mean').
    Train-loop is responsible for:
      1. accumulating per-class TP / FP / FN during training each epoch,
      2. calling .update_alpha(per_class_f1) at end-of-epoch,
      3. (optional) reading .alpha for diagnostic logging.
    """

    def __init__(
        self,
        n_classes: int,
        class_names: list[str],
        tau: float = 1.0,
        gamma: float = 1.0,
        momentum: float = 0.9,
        warm_up_epochs: int = 5,
        f1_floor: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.class_names = class_names
        self.tau = tau
        self.gamma = gamma
        self.momentum = momentum
        self.warm_up_epochs = warm_up_epochs
        self.f1_floor = f1_floor

        # Running EMA of per-class F1. Initialised to 1.0 so warm-up
        # evaluates to alpha = (1 - 1)^tau = 0 -> renormalised to uniform.
        # Held as a buffer so it moves with the module across .to(device)
        # and persists in module state.
        self.register_buffer(
            'f1_running',
            torch.ones(n_classes, device=device),
        )
        self.register_buffer(
            'alpha',
            torch.ones(n_classes, device=device),
        )
        self.epoch = 0  # bumped by update_alpha; controls warm-up gating

    @torch.no_grad()
    def update_alpha(self, per_class_f1: torch.Tensor) -> None:
        """
        Called at end-of-epoch with the train per-class F1 vector
        (shape [n_classes]). EMA-smooths into self.f1_running, maps to
        alpha = (1 - f1_running)^tau, renormalises mean to 1.0.

        During warm-up (epoch < warm_up_epochs), still updates the EMA
        but the forward() branch falls back to uniform alpha.
        """
        per_class_f1 = per_class_f1.clamp(min=self.f1_floor, max=1.0)
        self.f1_running = (
            self.momentum * self.f1_running
            + (1.0 - self.momentum) * per_class_f1
        )
        raw_alpha = (1.0 - self.f1_running).clamp(min=1e-8) ** self.tau
        # Renormalise to mean 1.0 so loss scale stays comparable to uniform CE.
        self.alpha = raw_alpha * (self.n_classes / raw_alpha.sum())
        self.epoch += 1

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [B, n_classes] pre-softmax.
        labels: [B] int64.
        """
        log_probs = F.log_softmax(logits, dim=-1)               # [B, C]
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        p_t = log_p_t.exp().clamp(max=1.0 - 1e-7)               # [B]

        # Per-class alpha lookup; warm-up forces uniform.
        if self.epoch < self.warm_up_epochs:
            alpha_t = torch.ones_like(p_t)
        else:
            alpha_t = self.alpha[labels]                        # [B]

        # focal modulator: (1 - p_t)^gamma
        if self.gamma > 0:
            focal_mod = (1.0 - p_t) ** self.gamma
        else:
            focal_mod = 1.0

        loss = -alpha_t * focal_mod * log_p_t                   # [B]
        return loss.mean()


def per_class_f1_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-class F1 from running TP / FP / FN tensors of shape [n_classes].
    Used by the train loop end-of-epoch to feed AdaptiveFocalLoss.update_alpha.
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1
```

### 5b. `bst_train.py` edits

(Line numbers below reference the post-class_weights state of the
file, see the surface map at the top of this design pass for current
landmarks. Conceptually small.)

1. **Hyp namedtuple** (line 54-61). Add field `'adaptive_focal'` (dict
   or None). Field order at end so kwargs in the active hyp block
   stay readable.

2. **Active hyp block** (line 79-86). Add the `adaptive_focal=None`
   default and a comment describing the dict shape (`tau`, `gamma`,
   `momentum`, `warm_up_epochs`, `update_freq`, `f1_floor`). Kept
   `None` when running uniform CE / class-weighted CE / manual focal.
   Kept set when running the adaptive arm.

3. **Loss-build block** (line 318-333). Replace the current
   `if hyp.class_weights: ... else: uniform CE` with three branches:

   ```python
   from .loss.adaptive_focal import AdaptiveFocalLoss

   if hyp.adaptive_focal:
       if hyp.class_weights:
           raise ValueError(
               "adaptive_focal and class_weights are mutually exclusive; "
               "set only one at a time."
           )
       if hyp.label_smoothing != 0.0:
           raise ValueError(
               "adaptive_focal requires label_smoothing=0.0 (LS contaminates "
               "focal's hardness estimate)."
           )
       af_cfg = hyp.adaptive_focal
       loss_fn = AdaptiveFocalLoss(
           n_classes=n_classes,
           class_names=class_ls,
           tau=af_cfg.get('tau', 1.0),
           gamma=af_cfg.get('gamma', 1.0),
           momentum=af_cfg.get('momentum', 0.9),
           warm_up_epochs=af_cfg.get('warm_up_epochs', 5),
           f1_floor=af_cfg.get('f1_floor', 0.0),
           device=device,
       )
       print(f"[loss] adaptive focal: tau={af_cfg.get('tau',1.0)}, "
             f"gamma={af_cfg.get('gamma',1.0)}, "
             f"momentum={af_cfg.get('momentum',0.9)}, "
             f"warm_up_epochs={af_cfg.get('warm_up_epochs',5)}")
   elif hyp.class_weights:
       # ... existing class-weighted CE branch unchanged ...
   else:
       loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)
   ```

4. **`train_one_epoch()`** (line 116-163). Add per-class TP/FP/FN
   accumulators alongside the existing loss accumulator. After the
   batch loop, return them along with `train_loss`. No second forward
   pass; counts come from the same `logits.argmax(-1)` call we'd
   make for accuracy. The accumulators are tensors on `device`,
   initialised to zeros each epoch. Cost: ~n_classes float adds per
   batch + 3*n_classes float adds per epoch end. Negligible.

   ```python
   tp = torch.zeros(n_classes, device=device)
   fp = torch.zeros(n_classes, device=device)
   fn = torch.zeros(n_classes, device=device)
   for batch in train_loader:
       # ... existing forward + backward ...
       with torch.no_grad():
           preds = logits.argmax(dim=-1)
           for c in range(n_classes):
               tp[c] += ((preds == c) & (labels == c)).sum()
               fp[c] += ((preds == c) & (labels != c)).sum()
               fn[c] += ((preds != c) & (labels == c)).sum()
   return train_loss, tp, fp, fn
   ```

   (The triple-loop scan is fine for n_classes=14. For larger
   taxonomies, vectorise via `torch.bincount` or scatter-add.)

5. **Epoch loop** (line 355-, calls into `train_one_epoch`). After
   the train pass returns, compute per-class F1 and call the alpha
   updater if the loss is adaptive:

   ```python
   train_loss, tp, fp, fn = train_one_epoch(...)
   train_per_class_f1 = per_class_f1_from_counts(tp, fp, fn)
   if isinstance(loss_fn, AdaptiveFocalLoss):
       loss_fn.update_alpha(train_per_class_f1)
   ```

6. **TensorBoard logging** (line 387-391). Add per-class scalars for
   `F1_train/<class>` and `Alpha/<class>` each epoch:

   ```python
   for i, c in enumerate(class_ls):
       writer.add_scalar(f'F1_train/{c}', train_per_class_f1[i].item(), epoch)
       if isinstance(loss_fn, AdaptiveFocalLoss):
           writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)
   ```

7. **Console print** (line 383-385). Add a top-3 / bot-3 alpha
   summary line each epoch (matches the existing `val top5 / bot5`
   pattern at lines 415-417):

   ```python
   if isinstance(loss_fn, AdaptiveFocalLoss):
       a = loss_fn.alpha.cpu().numpy()
       order = a.argsort()
       print('  alpha bot3: ' + ' '.join(
           f'{class_ls[i]}={a[i]:.2f}' for i in order[:3]))
       print('  alpha top3: ' + ' '.join(
           f'{class_ls[i]}={a[i]:.2f}' for i in order[-3:][::-1]))
   ```

**Manifest auto-serialisation already handles `Hyp.adaptive_focal`**
via `_config_to_dict(...)._asdict()` in `run_tracker.py:159-160`. No
extra serialisation code. Manifest will record the full config dict
(tau, gamma, momentum, warm_up_epochs, etc.) under `config:
adaptive_focal:`.

Total: ~120 lines new module + ~25 lines across 6 small edits in
`bst_train.py`. Within the prompt's stated budget of "~50-150 lines
new module + 4-6 edits to bst_train.py".

## 6. Diagnostic + manifest plumbing

**Per epoch (console)**:
- `alpha bot3: <c1>=<a1> <c2>=<a2> <c3>=<a3>` (the 3 lowest-alpha
  classes — i.e. the classes the loss currently treats as easiest)
- `alpha top3: <c1>=<a1> <c2>=<a2> <c3>=<a3>` (the 3 highest-alpha
  classes — i.e. the classes the loss currently focuses on hardest)

**Per epoch (TensorBoard)**:
- `F1_train/<class_name>` per class (n_classes scalars).
- `Alpha/<class_name>` per class (n_classes scalars).

**Manifest (`manifest.yaml`)**:
- The `Hyp.adaptive_focal` dict gets serialised verbatim under
  `config: adaptive_focal:` (auto via `_asdict()`).
- The trajectory itself (per-epoch alpha and per-epoch F1) lives in
  TensorBoard, not the manifest. Manifest stays static config + final
  summary, per the existing house pattern.

## 7. Comparison protocol

**Baselines for direct comparison** (combo A nosides, 5 seeds each):

| label                                 | run_id              | mean macro | mean min (ws) | mean acc | mean top-2 |
|---------------------------------------|---------------------|------------|---------------|----------|------------|
| LS=0.1 (paper default, canonical)     | run_20260430_170325 | 0.742      | 0.375         | 0.767    | 0.938      |
| LS=0.0 (cell 1, disproved)            | run_20260430_213933 | 0.743      | 0.359         | 0.768    | 0.939      |
| LS=0.15 (cell 2, in flight)           | tbd                 | tbd        | tbd           | tbd      | tbd        |
| class-weights {ws,smash:2.0} (queued) | tbd                 | tbd        | tbd           | tbd      | tbd        |
| manually-alpha focal (gated)          | tbd                 | tbd        | tbd           | tbd      | tbd        |

CDB-style adaptive focal compares against LS=0.1 baseline as the
canonical ground, plus whichever of the simpler arms above ran
last as the "did the simpler form already get there" check.

**Gate condition for "this worked"**:

- **Pass.** Mean wrist_smash F1 ≥ 0.42 (above LS=0.1 baseline mean
  of 0.375 by a 4.5 pp margin, comfortably outside seed variance:
  LS=0.1 wrist_smash range was 0.159 across 5 seeds, so a
  per-seed shift of ~3 pp would be inside variance. Mean shift
  of 4.5 pp is meaningful), AND mean macro F1 ≥ 0.737 (within
  0.5 pp of baseline; macro shouldn't tank to chase wrist_smash).
- **Partial.** Mean wrist_smash F1 in [0.40, 0.42] AND macro flat.
  Worth re-running with `gamma=2.0` or `tau=2.0` for a more
  aggressive variant.
- **Fail.** Mean wrist_smash F1 ≤ 0.39 (within seed variance of
  baseline). Adaptive-focal arm closes; pivot to augmentation or
  X3D-S.

**Tiebreakers** with simpler arms:

- If class-weighting smoke (`{ws:2.0, smash:2.0}`) already lifted
  wrist_smash to ≥0.42, CDB-adaptive's gate moves to ≥0.45 to
  justify the implementation cost.
- If manually-alpha focal also lifted, CDB-adaptive needs to lift
  wrist_smash by another ≥2 pp on top to justify.
- If both simpler arms failed but CDB-adaptive lifts to ≥0.42, big
  win — adaptive class weighting really was the missing piece.

**Variance budget**. 5-seed runs are the unit. A single-seed result
that lifts wrist_smash isn't enough to call a winner — need the
mean to clear, and seed range to not blow up vs LS=0.1 baseline
(range 0.159).

## 8. Failure modes to watch

**Oscillation**. Low-F1 class boosted → its F1 lifts → boost drops
on next epoch's update → F1 falls back. Mitigated by:
- 5-epoch warm-up (alpha builds slowly via EMA before being applied)
- momentum=0.9 (effective horizon ~10 epochs, much longer than the
  oscillation period)
- mean=1.0 renormalisation (no class can run away absolutely)

Watch for: alpha trajectory oscillating between epochs in
TensorBoard. If the wrist_smash alpha jitters by more than ~10%
epoch-to-epoch after warm-up, raise momentum to 0.95.

**Noise amplification at high gamma**. Wang et al. 2019 (and
Sinha's IJCV 2022 follow-up) document that focal at gamma ≥ 2
amplifies label noise: noisy samples have low p_t → up-weighted →
gradient pulls the model toward fitting the noise. Mitigated by:
- gamma = 1.0 default (stays close to plain CE focal regime)
- ShuttleSet annotation noise is moderate, not pathological

Watch for: divergence in train vs val loss curves after epoch 30.
If train loss keeps dropping but val loss flatlines or rises while
focal is on, it's overfit to noise. Drop gamma to 0.5 or run
without focal (CDB-only, gamma=0).

**Head-class collapse**. If alpha gets too aggressive on the tail,
head classes get too little gradient and drift downward. Mitigated
by:
- mean=1.0 renormalisation (head class alpha can't go below
  ~0.3 in practice, since 14 classes share a budget of 14)
- macro-F1 gate condition (won't accept a wrist_smash lift that
  costs >0.5 pp macro)

Watch for: macro F1 on val dropping below LS=0.1 baseline
while wrist_smash lifts. If it does, tau is too high. Drop to 0.5.

**Saturation at end of training**. By epoch 60-70, train F1 is
typically high across all classes (model memorises train set).
F1_running converges, alpha converges to near-uniform. The
adaptive part is doing nothing. This is expected — the lift comes
from the alpha shape during epochs 20-50, not at the end. Not a
failure mode, just a tracker note: don't expect alpha to stay
differentiated.

**Train F1 vs val F1 drift**. Adaptive focal optimises against the
class the model can't learn *on the train set*. If train F1 saturates
on a class but val F1 stays low (overfit), the adaptive signal turns
off too early. Mitigated by:
- Standard early stopping is on val macro (unchanged) — if train
  saturates and val doesn't, early stop fires and we don't get the
  bad regime in the saved checkpoint.
- Keep an eye on wrist_smash val F1 vs train F1 — if val F1 is
  consistently 10+ pp below train F1, train signal is misleading.
  In that case, consider going back to manual class_weights with a
  higher multiplier on wrist_smash.

**Pair-confusion residual**. CDB scalar-per-class can't explicitly
model wrist_smash↔smash confusion. If the lift comes by stealing
recall from smash (smash F1 drops noticeably while wrist_smash
lifts), then a CDB+pair-cap composition is the next refinement —
cap alpha[smash] / alpha[wrist_smash] ratio at, say, 0.7 to keep
both classes upweighted relative to the rest. This is the natural
"CDB + Seesaw-flavoured pair correction" composition; defer until
we see whether the residual confusion shows up in the smoke run.

---

## Final recommendation

**Implement after Cell 1 (class-weighting smoke test) and Cell 2
(manually-alpha focal) results land.**

Reasoning: the class-weighting smoke test is a much cheaper signal
on whether the loss-reweighting axis can move wrist_smash at all.
If it lifts wrist_smash to ≥0.40 with LS=0.1, the next-cheapest
move (manually-alpha focal at gamma=1.0) is the natural extension
and may close the gap on its own. CDB-style adaptive focal is the
right escalation when (a) manual alpha lifts wrist_smash but
plateaus, (b) the plateau is consistent with "fixed alpha is too
coarse — wrist_smash needs more weight early, less late, and the
model knows which it is via running F1", which is exactly the
regime CDB was designed for. Without first having the cheaper-arm
data, the adaptive arm is solving an unconfirmed problem.

Park status: design is complete, prompt is satisfied, ready to
re-enter when class-weighting + manually-alpha focal cells have
landed. No `bst_train.py` edits made yet (per the user's "docs only"
instruction in the design pass).
