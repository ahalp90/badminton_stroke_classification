# Seesaw Loss reframed with class F1: design

Companion to `class_f1_focal_design.md` (CDB-F1). Designs the
Seesaw-loss-style alternative for the class-F1-driven loss arm,
where the per-class signal swap is from cumulative training counts
to running per-class F1. Ends with a side-by-side recommendation:
prefer this over CDB-F1, or not?

Verified against the actual paper PDF (Wang et al., "Seesaw Loss
for Long-Tailed Instance Segmentation", CVPR 2021,
[arXiv:2008.10032](https://arxiv.org/abs/2008.10032)). Equations
and tables transcribed from the PDF directly.

---

## 1. Chosen variant

**Seesaw Loss (Wang et al. CVPR 2021) with the mitigation-factor
signal swapped from cumulative training counts (`N_i, N_j`) to
running per-class train F1 (`F1_i, F1_j`). Compensation factor
left as published.**

### What it actually does, in plain English

Seesaw Loss is a **pair-aware** modification to the softmax
denominator. The mechanism has two layers:

**Layer 1 — class-level mitigation.** When training on a sample
whose true class is `i`, the loss looks at every other class `j`
and asks: "does class `i` already perform better than class `j`?"
If yes, the loss reduces the penalty that this sample's gradient
imposes on class `j`. The intuition: easier classes' gradients
shouldn't be allowed to overwhelm harder classes' positive signal.
A class-level matrix `M_ij` encodes the "how much should we reduce
the i→j penalty" for every (i, j) pair. In the published version,
`M_ij` is computed from cumulative *counts*: head classes' samples
get downweighted when penalising tail classes. In this reframe,
`M_ij` is computed from running *F1*: easier classes' samples get
downweighted when penalising harder classes. The reframe makes the
loss target the *difficulty* axis instead of the *count* axis,
just like the CDB-F1 reframe — but it does it pair-by-pair instead
of class-by-class.

**Layer 2 — sample-level compensation.** For each individual
sample, the loss looks at the predicted softmax probabilities. If
some negative class `j` has higher predicted probability than the
true class `i` (i.e. the model is confused — about to misclassify
this sample), the loss *increases* the penalty on class `j`
specifically for this sample, by a factor depending on how
confused. This is a focal-loss-style hard-sample focuser, but
applied pair-wise: it focuses on whichever wrong-class is winning,
not just on samples generally. The matrix `C_ij` encodes this for
every (i, j) pair. We keep the published version.

The two layers compose: class-level mitigation (`M_ij`) plus
sample-level compensation (`C_ij`) gives the final pair scaling
`S_ij = M_ij · C_ij`. The modified softmax is

```
σ̂_i = exp(z_i) / ( Σ_{j ≠ i} S_ij * exp(z_j) + exp(z_i) )
```

and the loss is just `-log(σ̂_i)` for the true class `i`. So the
math change is *only* in the softmax denominator: the cross-entropy
shell stays the same.

For the wrist_smash case specifically: when training on a `smash`
sample (which the model already does OK on, F1 ≈ 0.66) and
considering the gradient on the `wrist_smash` logit (F1 ≈ 0.38),
the mitigation factor `M_{smash, wrist_smash}` becomes
`(0.38/0.66)^0.8 ≈ 0.63` — the smash sample's gradient on
wrist_smash is reduced by ~37%, protecting wrist_smash from being
suppressed by smash's easier signal. In the reverse direction
(training on a `wrist_smash` sample, gradient on `smash` logit),
F1_wrist_smash ≤ F1_smash so mitigation doesn't kick in — the
wrist_smash sample's gradient on smash stays at full strength. The
asymmetry is the whole point: protect the harder class without
crippling the easier one.

### Citations

- Wang et al., "Seesaw Loss for Long-Tailed Instance Segmentation",
  CVPR 2021. [arXiv:2008.10032](https://arxiv.org/abs/2008.10032).
  [openaccess link](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.pdf).
- Reference implementation in MMDetection:
  https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss

### Reference equations (verified verbatim from the paper)

- Standard CE (eq. 1): `L_CE = -Σ_i y_i log(σ_i)` with
  `σ_i = exp(z_i) / Σ_j exp(z_j)`.
- Modified CE (eq. 4): `L_seesaw = -Σ_i y_i log(σ̂_i)` with
  `σ̂_i = exp(z_i) / (Σ_{j≠i} S_ij * exp(z_j) + exp(z_i))`.
- Pair scaling (eq. 6): `S_ij = M_ij · C_ij`.
- Mitigation factor (eq. 7): `M_ij = 1` if `N_i ≤ N_j`,
  `M_ij = (N_j / N_i)^p` if `N_i > N_j`. Default `p = 0.8`.
- Compensation factor (eq. 8): `C_ij = 1` if `σ_j ≤ σ_i`,
  `C_ij = (σ_j / σ_i)^q` if `σ_j > σ_i`. Default `q = 2`.
- Gradient on negative class (eq. 5):
  `∂L_seesaw/∂z_j = S_ij * (exp(z_j)/exp(z_i)) * σ̂_i`. Confirms
  that `S_ij` acts as a multiplicative gate on the
  negative-class gradient. (Note: the paper's gradient derivation
  treats `S_ij` as a fixed scalar — `C_ij`'s dependence on
  `σ_j, σ_i` is detached from autograd in the implementation.)

### What we change vs the published version

One substitution in the mitigation factor only:

- `M_ij = 1` if `F1_i ≤ F1_j` (no mitigation when class `i` is
  already harder than `j`)
- `M_ij = (max(F1_j, ε) / F1_i)^p` if `F1_i > F1_j` (mitigation
  when class `i` is easier than `j`; `ε = 0.05` is a floor on
  `F1_j` to prevent runaway when a class has near-zero F1 in
  early training)

The compensation factor `C_ij` is unchanged. F1 comes from the
running EMA on per-class train F1, computed end-of-epoch from a
TP/FP/FN accumulator (same mechanism as the CDB-F1 design).

Why F1 instead of counts: same argument as CDB-F1. On combo A
nosides, count and difficulty are decoupled (long_service is
rarest by count but at F1 0.99; wrist_smash is 5th-rarest by
count but at F1 0.38). Count-based mitigation barely moves
wrist_smash. F1-based mitigation targets the actual bottleneck.

### Reported lift in published work (count-based version)

Verified against the full paper PDF directly (sections 3, 4.1-4.5,
tables 1-11, appendix A). Two published evaluation domains:

**LVIS v1 instance segmentation** (table 1; primary evaluation in
the paper), 1203 categories, Mask R-CNN ResNet-101 + FPN backbone,
2x schedule:

| sampler | loss | overall AP | rare AP_r | common AP_c | frequent AP_f |
|---|---|---|---|---|---|
| Random | CE | 20.6 | 0.8 | 19.3 | 30.7 |
| Random | EQL | 22.7 | 3.7 | 23.3 | 30.4 |
| Random | BAGS | 25.6 | 17.3 | 25.0 | 30.1 |
| Random | Seesaw | **26.6** | **18.1** | 25.8 | **31.2** |
| RFS | CE | 25.5 | 16.6 | 24.5 | 30.6 |
| RFS | Seesaw | **27.6** | **20.6** | 27.3 | 31.1 |

Seesaw vs CE with random sampler: **+6.0 AP overall, +17.3 AP_r
(rare classes lift 0.8 → 18.1)**, while not sacrificing AP_f
(30.7 → 31.2 actually slight gain). EQL and BAGS both *lose*
AP_f vs CE — Seesaw is uniquely the method that lifts rare
without dropping frequent.

**ImageNet-LT image classification** (table 10, §4.4), 1000
categories, ResNeXt-50 backbone, 90-epoch end-to-end softmax CE
pipeline. This is the result I missed on first read — and it
matters because it's the same problem class as ours (multi-class
softmax classification, not detection-specific scaffolding):

| method | overall | Many shot (>100) | Medium (20-100) | Few shot (<20) |
|---|---|---|---|---|
| CE | 44.4 | 65.9 | 37.5 | 7.7 |
| Focal Loss | 43.3 | 64.5 | 36.3 | 7.8 |
| CB-Focal | 45.3 | 60.4 | 40.6 | 19.2 |
| EQL | 46.0 | 61.7 | 42.5 | 13.8 |
| Seesaw (decoupled) | 49.7 | 60.7 | 46.8 | 28.9 |
| **Seesaw (end-to-end)** | **50.4** | **67.1** | 45.2 | 21.4 |

Seesaw end-to-end vs CE: **+6.0 pp overall, +13.7 pp on
Few-shot classes (7.7 → 21.4)**, with Many-shot actually
*lifted* (65.9 → 67.1) — same "lifts rare without dropping
frequent" story as LVIS. Notably, the paper itself observes
that Seesaw end-to-end beats Seesaw with the more complex
decoupled-training pipeline, so we don't need any two-phase
training scheme to use it.

Component ablation (table 2, Mask R-CNN ResNet-50 + RFS):

| MF | CF | NLA | AP | AP_r | AP_c | AP_f |
|---|---|---|---|---|---|---|
| – | – | – | 23.7 | 13.5 | 22.8 | 29.3 |
| ✓ | – | – | 25.1 | 16.7 | 24.5 | 29.4 |
| – | ✓ | – | 24.1 | 13.2 | 23.5 | 29.5 |
| ✓ | ✓ | – | 25.7 | 19.1 | 25.0 | 29.4 |
| ✓ | ✓ | ✓ | 26.4 | 19.6 | 26.1 | 29.8 |

Mitigation alone: +1.4 AP, **+3.2 AP_r**. Mitigation +
compensation: +2.0 AP, **+5.6 AP_r**. NLA is a detection-specific
classifier-head normalisation; not relevant to our skeleton-
transformer setup.

**Online-vs-precomputed counts ablation** (table 4, §4.3): the
paper compares three sources for `N_i` — accumulated online during
training (default), pre-recorded from a previously trained Seesaw
model, or computed once from the training-set distribution. All
three land within 0.3 AP (26.4 / 26.3 / 26.1 respectively). This
matters for our reframe: it confirms that **running statistics
work in this loss family**, which is exactly what the F1 reframe
relies on (running per-class F1 EMA instead of running per-class
count). Architectural soundness inherited.

**Caveat on dataset domain**: the LVIS evaluation is 1203-class
detection (different problem class), but the ImageNet-LT
evaluation is 1000-class softmax classification (same problem
class as ours, just 70x more categories). So Seesaw is **not** a
detection-only method — the loss formulation IS proven on
multi-class softmax classification at scale. The remaining
domain-extrapolation concern is small-taxonomy specifically:
14 classes vs 1000, where most of the C×C matrix entries will
have similar F1 and the pair-targeting machinery has fewer
informative cells to work on.

## 2. Why this variant fits combo A — and where the fit breaks

The wrist_smash↔smash bottleneck on combo A nosides is
*structurally* a pair-confusion problem, not a long-tail problem.
The two classes co-occur in the confusion matrix; one underperforms
because the other dominates the gradient. Seesaw's mitigation
factor is the only published method in the survey that explicitly
encodes pair structure rather than per-class scalar weights. So
the structural fit is genuinely better than CDB-F1's scalar-α
approach.

The fit breaks in two places (revised down from three after the
full paper read):

- **The pair-awareness machinery is C×C, but our pair is one cell
  in a 14×14 matrix.** The other 13 columns of the wrist_smash row
  (and the other 13 rows of the smash column) are pulling weight
  too. Most of those entries' mitigation factors will be near 1
  (because their F1s are similar to wrist_smash's or smash's), so
  they'll mostly behave like uniform CE. The targeted effect on
  the wrist_smash↔smash pair is real, but it's diluted by the
  rest of the matrix doing essentially nothing useful. By contrast,
  CDB-F1's α_wrist_smash directly upweights *every* gradient
  flowing through that class regardless of confusion partner —
  simpler and arguably better-targeted at "this class is hard".

- **Compensation factor + label noise interaction.** The
  per-sample compensation `C_ij = (σ_j / σ_i)^q` at the LVIS
  default `q = 2` is aggressive: when the model is confused (σ_j
  exceeds σ_i for some negative `j`), the gradient on `j`
  amplifies quadratically in the σ-ratio. ShuttleSet has
  human-annotation label noise; on noisy samples, the model is
  "confused" because the *label* is wrong, and `C_ij` dutifully
  amplifies the wrong gradient. The paper's own ImageNet-LT
  ablation (table 11) finds `q = 1` is optimal there (50.4 vs
  49.4 at q=2), and the paper itself uses `q = 1` for image
  classification (line 540-541 of the paper). So our `q = 1`
  default isn't just a noise-tolerance hedge — it's what the
  paper recommends for this problem class.

(Removed concern: "LVIS is 1203 classes; combo A is 14, so the
formulation is unproven outside detection". Wrong on first
read — the paper's §4.4 / table 10 evaluates Seesaw on
ImageNet-LT 1000-class softmax classification with the same
formulation. Still a 70x scale extrapolation, but the loss IS
proven outside detection.)

The summary on fit: structurally Seesaw-F1 is more *aimed* at our
specific problem (the smash pair), but the implementation is
designed for a much bigger problem (1203-class detection) and the
sample-level focusing is brittle under label noise. Whether the
structural advantage pays off in practice on a 14-class softmax
is the open question.

## 3. Stability mechanism

### The oscillation problem

Same as CDB-F1: if the F1 readings used to compute `M_ij` jump
around epoch-to-epoch, the pair scaling matrix `S_ij` jumps with
them. Worse than the CDB-F1 case because Seesaw modifies the
softmax denominator — a noisy `M_ij` directly perturbs the
softmax shape, which feeds back into the next epoch's confusion
patterns.

### The fix: same EMA on F1, applied before mapping to M_ij

Same as CDB-F1, with the same momentum:

```
F1_running_c = momentum * F1_running_c + (1 - momentum) * F1_this_epoch_c
   for each class c

M_ij = 1                                 if F1_running_i ≤ F1_running_j
M_ij = (max(F1_running_j, ε) / F1_running_i)^p   otherwise
```

`M_ij` is fully recomputed at end-of-epoch from the smoothed F1
vector and held fixed for the next epoch. `C_ij` stays per-sample
(no smoothing — it's a focal-style mechanism on the current
forward pass).

### Why `momentum = 0.9`

Same reasoning as the CDB-F1 doc — EMA half-life ~6.6 epochs over
an 80-epoch budget, matches BatchNorm/Adam convention, smoother
than per-epoch raw F1.

### Why a 5-epoch warm-up

Same reasoning. In epoch 1, every class's F1 is near 0, so all
F1 ratios collapse to 0/0 territory or to (very small)/(very
small) which the floor `ε` papers over but doesn't make
informative. Run uniform CE for 5 epochs, let the EMA build a
sensible F1 estimate, then turn on Seesaw at epoch 6.

## 4. Hyperparameter defaults

Six knobs, three of them shared with CDB-F1 (momentum,
warm_up_epochs, f1_floor). The three new knobs (`p`, `q`,
`compensation_active`) control Seesaw-specific behavior:

- **`p`** is the pair-mitigation aggressiveness dial. With
  `p = 0.8` (paper default for both LVIS and ImageNet-LT), if
  F1_i = 0.66 and F1_j = 0.38, `M_ij = (0.38/0.66)^0.8 ≈ 0.63` —
  the i→j gradient is reduced by ~37%. The paper's table 5 sweep
  on LVIS goes from p=0.2 to p=1.2 with **p=0.8 the empirical
  peak** (AP_r 19.1 at p=0.8 vs 17.6 at p=1.0 vs 14.7 at p=0.2).
  We default to p=0.8.
- **`q`** is the per-sample compensation aggressiveness dial. With
  `q = 2` (paper default for LVIS detection), if the model
  predicts σ_j = 0.5 for a wrong class while σ_i = 0.3 for the
  true class, `C_ij = (0.5/0.3)^2 ≈ 2.78` — the j-class gradient
  is amplified by 2.78×. With `q = 1`, the same case gives
  `C_ij ≈ 1.67` — milder. With `q = 0` (or
  `compensation_active = False`), `C_ij = 1` always — drop the
  compensation entirely. **For our setup we default to `q = 1`,
  which is what the paper itself uses for ImageNet-LT image
  classification (line 540-541 of the paper). Table 11 confirms
  q=1 is the optimum on ImageNet-LT (50.4 vs q=2 at 49.4)** —
  the LVIS-detection optimum (q=2) doesn't generalise to softmax
  classification, and our problem is closer to ImageNet-LT than
  to LVIS in problem class. ShuttleSet's label noise makes q=1
  doubly preferable.
- **`compensation_active`**: bool. If `False`, skip the
  compensation factor entirely — `S_ij = M_ij` only. Equivalent
  to `q = 0` but cleaner to flag in the manifest. Default `True`
  (keep compensation but at `q = 1`).
- **`momentum`**: 0.9 — same as CDB-F1.
- **`warm_up_epochs`**: 5 — same as CDB-F1.
- **`f1_floor`**: `ε = 0.05` (light floor on F1_j inside `M_ij`,
  to avoid runaway when F1_j → 0). Higher than CDB-F1's `ε = 0.0`
  because the mitigation formula is more sensitive to near-zero
  F1 (the ratio `F1_j / F1_i` blows up downward, not up).

| Knob | Default | Range / fallbacks | Why |
|---|---|---|---|
| `p` | 0.8 | 0.5 (gentler), 1.0 (already past peak per table 5) | Empirical peak on the paper's table 5 sweep (LVIS, AP_r 19.1 at p=0.8 vs 17.6 at p=1.0). |
| `q` | 1.0 | 0 (no compensation), 2.0 (LVIS default) | Paper itself uses q=1 for ImageNet-LT image classification (table 11 optimum: 50.4 vs q=2 at 49.4). LVIS default q=2 is detection-specific. |
| `compensation_active` | True | False (mitigation only) | Mitigation alone gives +3.2 AP_r on LVIS (table 2); compensation adds +2.4 AP_r more. Worth keeping but at lower `q`. |
| `momentum` | 0.9 | 0.8 (faster), 0.95 (steadier) | Same as CDB-F1. |
| `warm_up_epochs` | 5 | 3 (short), 10 (long) | Same as CDB-F1. |
| `f1_floor` (ε) | 0.05 | 0.01 (tighter), 0.1 (looser) | Higher than CDB-F1 because mitigation is more sensitive to small F1_j. |

`label_smoothing` forced to 0.0 (the compensation factor is
focal-flavoured and depends on `σ_j` / `σ_i`; LS contaminates
those ratios — same constraint as CDB-F1 + focal).

## 5. Implementation surface

### Plain-English overview

Seesaw needs more code than CDB-F1 because the math change is
deeper — not "multiply the loss by α_c" but "modify the softmax
denominator with a class-pair gate". Three new pieces:

1. **A new loss module** that holds the running F1 vector, the
   pair-mitigation matrix `M_ij` (C×C, recomputed once per epoch),
   and implements the modified softmax in the forward pass. The
   forward computes `C_ij` per-sample on the fly (inside
   `torch.no_grad()` for the σ_j, σ_i comparison, since the paper
   detaches that path from autograd).
2. **Same per-class TP/FP/FN counter inside `train_one_epoch`** as
   CDB-F1 — gives end-of-epoch per-class F1 to feed the EMA.
3. **A loss-build branch in `bst_train.py`** that picks adaptive
   Seesaw when `Hyp.adaptive_seesaw` is set. The four branches
   (uniform CE / class-weighted CE / adaptive_focal / adaptive_seesaw)
   are mutually exclusive (raise if more than one is set).

The custom forward needs the **log-sum-exp trick** for numerical
stability, because the modified denominator can underflow at large
logit magnitudes. This is the one piece that requires care.

Two files touched:
1. **NEW** `src/bst_refactor/stroke_classification/main_on_shuttleset/loss/adaptive_seesaw.py`
   — ~180 lines.
2. `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`
   — same ~6 edits as the CDB-F1 design, plus the
   `adaptive_seesaw` branch.

### 5a. `loss/adaptive_seesaw.py` skeleton

```python
"""
Class-F1-driven Seesaw Loss for combo A.

Implements Wang et al. (CVPR 2021) Seesaw Loss with the mitigation
factor signal swapped from cumulative training counts to per-class
running train F1. Compensation factor unchanged from the paper.
See scratch/architecture_notes/seesaw_f1_focal_design.md for the
full motivation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSeesawLoss(nn.Module):
    """
    Seesaw Loss with per-class F1-driven mitigation matrix.

    Forward signature matches nn.CrossEntropyLoss(reduction='mean').
    Train-loop responsibilities:
      1. accumulate per-class TP / FP / FN during training each epoch,
      2. call .update_mitigation(per_class_f1) at end-of-epoch,
      3. (optional) read .M_ij and .f1_running for diagnostics.
    """

    def __init__(
        self,
        n_classes: int,
        class_names: list[str],
        p: float = 0.8,
        q: float = 1.0,
        compensation_active: bool = True,
        momentum: float = 0.9,
        warm_up_epochs: int = 5,
        f1_floor: float = 0.05,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.class_names = class_names
        self.p = p
        self.q = q
        self.compensation_active = compensation_active
        self.momentum = momentum
        self.warm_up_epochs = warm_up_epochs
        self.f1_floor = f1_floor

        # Running EMA of per-class F1, init to 1.0 (warm-up regime).
        self.register_buffer(
            'f1_running',
            torch.ones(n_classes, device=device),
        )
        # Pair mitigation matrix, init to all-ones (uniform).
        self.register_buffer(
            'M_ij',
            torch.ones(n_classes, n_classes, device=device),
        )
        self.epoch = 0

    @torch.no_grad()
    def update_mitigation(self, per_class_f1: torch.Tensor) -> None:
        """
        Called at end-of-epoch with per_class_f1 of shape [n_classes].
        EMA-smooths into self.f1_running, recomputes self.M_ij.
        """
        per_class_f1 = per_class_f1.clamp(min=0.0, max=1.0)
        self.f1_running = (
            self.momentum * self.f1_running
            + (1.0 - self.momentum) * per_class_f1
        )
        # M_ij = 1 if F1_i <= F1_j else (max(F1_j, eps) / F1_i)^p
        f1_i = self.f1_running.unsqueeze(1)  # [C, 1]
        f1_j = self.f1_running.unsqueeze(0)  # [1, C]
        f1_j_floored = f1_j.clamp(min=self.f1_floor)
        ratio = f1_j_floored / f1_i.clamp(min=1e-8)
        M_ij = torch.where(
            f1_i > f1_j,
            ratio ** self.p,
            torch.ones_like(ratio),
        )
        self.M_ij = M_ij
        self.epoch += 1

    def forward(
        self,
        logits: torch.Tensor,        # [B, C]
        labels: torch.Tensor,        # [B]
    ) -> torch.Tensor:
        B, C = logits.shape

        # Warm-up: standard CE.
        if self.epoch < self.warm_up_epochs:
            return F.cross_entropy(logits, labels)

        # Look up M_ij row for each sample's true class.
        # M_row[b, j] = M_{labels[b], j}, shape [B, C].
        M_row = self.M_ij[labels]            # [B, C]

        # Compute compensation factor C_ij from current σ.
        # Per-sample: C_{ij} = (σ_j / σ_i)^q if σ_j > σ_i else 1.
        # Detach σ from autograd per the paper's gradient derivation
        # (eq. 5 treats S_ij as a fixed scalar w.r.t. backprop).
        if self.compensation_active and self.q > 0:
            with torch.no_grad():
                sigma = F.softmax(logits, dim=-1)             # [B, C]
                sigma_i = sigma.gather(1, labels.unsqueeze(1))  # [B, 1]
                ratio = sigma / sigma_i.clamp(min=1e-8)       # [B, C]
                C_row = torch.where(
                    sigma > sigma_i,
                    ratio ** self.q,
                    torch.ones_like(ratio),
                )                                              # [B, C]
            S_row = M_row * C_row
        else:
            S_row = M_row

        # Modified softmax with the pair-scaled denominator:
        #   sigma_hat_i = exp(z_i) / (sum_{j!=i} S_ij * exp(z_j) + exp(z_i))
        # Use log-sum-exp form for numerical stability.
        # Build log(S_row) carefully: S_row > 0 guaranteed by construction.
        log_S = torch.log(S_row.clamp(min=1e-12))             # [B, C]

        # Mask out the true-class column from the negative-class sum
        # (we'll add exp(z_i) separately, with log_S = 0 effectively).
        # Equivalent: zero out log_S at the true-class index.
        true_mask = F.one_hot(labels, C).bool()                # [B, C]
        log_S_masked = log_S.masked_fill(true_mask, 0.0)

        # Modified per-class log-prob. Numerator log(exp(z_i)) = z_i.
        # Denominator: log( sum_j exp(z_j + log_S_masked_j) ),
        # where log_S_masked_j == 0 for j == i (so exp(z_i) is included
        # at full weight).
        modified_z = logits + log_S_masked                    # [B, C]
        log_denom = torch.logsumexp(modified_z, dim=-1)       # [B]
        z_true = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        log_p_t = z_true - log_denom

        return -log_p_t.mean()
```

### 5b. `bst_train.py` edits

Largely the same edits as CDB-F1 (per the design doc):

1. **Hyp namedtuple**: add `'adaptive_seesaw'` field.
2. **Active hyp block**: `adaptive_seesaw=None` default. Comment
   describing the dict shape (p, q, compensation_active, momentum,
   warm_up_epochs, f1_floor).
3. **Loss-build block**: add a fourth branch:
   ```python
   if hyp.adaptive_seesaw:
       # mutex check vs class_weights and adaptive_focal
       if hyp.class_weights or hyp.adaptive_focal:
           raise ValueError(
               "adaptive_seesaw is mutually exclusive with class_weights "
               "and adaptive_focal."
           )
       if hyp.label_smoothing != 0.0:
           raise ValueError("adaptive_seesaw requires label_smoothing=0.0.")
       cfg = hyp.adaptive_seesaw
       loss_fn = AdaptiveSeesawLoss(
           n_classes=n_classes, class_names=class_ls,
           p=cfg.get('p', 0.8),
           q=cfg.get('q', 1.0),
           compensation_active=cfg.get('compensation_active', True),
           momentum=cfg.get('momentum', 0.9),
           warm_up_epochs=cfg.get('warm_up_epochs', 5),
           f1_floor=cfg.get('f1_floor', 0.05),
           device=device,
       )
       print(f"[loss] adaptive seesaw: p={cfg.get('p',0.8)}, "
             f"q={cfg.get('q',1.0)}, comp_active={cfg.get('compensation_active',True)}")
   elif hyp.adaptive_focal:
       # ... CDB-F1 branch ...
   elif hyp.class_weights:
       # ... existing class-weighted CE branch ...
   else:
       loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)
   ```
4. **`train_one_epoch()`**: same TP/FP/FN accumulator addition as
   the CDB-F1 design (returns per-class F1 alongside train_loss).
5. **Epoch loop**: after the train pass, call `loss_fn.update_mitigation(...)`
   if it's the adaptive_seesaw type.
6. **TensorBoard logging**: `F1_train/<class>` (same as CDB-F1) +
   `M_ij/<class_i>_<class_j>` for the wrist_smash row of the
   matrix specifically (logging the full 14×14 = 196 scalars
   each epoch is overkill; pin to the wrist_smash row only).
7. **Console print**: `M_ij wrist_smash row` summary each epoch
   (top-3 / bot-3 mitigation values from that row, so we can see
   which classes are getting their gradient most reduced when
   training on wrist_smash siblings).

Total: ~180 lines new module + ~30 lines across 7 edits in
`bst_train.py`. Larger surface than CDB-F1 (~120 + ~25), as
expected for the deeper math change.

## 6. Diagnostic + manifest plumbing

**Per epoch (console)**:
- F1 top-3 / bot-3 across classes (same as CDB-F1).
- Mitigation row for wrist_smash: e.g.
  `M_{wrist_smash, *} bot3: long_service=0.42 short_service=0.43 clear=0.51`
  meaning these classes' gradients on wrist_smash are most
  attenuated (because they're the easiest classes).

**Per epoch (TensorBoard)**:
- `F1_train/<class>` (same as CDB-F1).
- `Seesaw_M/wrist_smash/<other_class>` for each `<other_class>` —
  the mitigation row for our bottleneck class. 13 scalars per epoch.
- `Seesaw_M_mean` — the mean off-diagonal `M_ij` value, for a
  rough "is mitigation kicking in at all" sanity check.

**Manifest (`manifest.yaml`)**:
- The `Hyp.adaptive_seesaw` config dict serialises automatically
  via `_asdict()` in `run_tracker.py:160`.
- The matrix trajectory itself lives in TensorBoard, not the
  manifest.

## 7. Comparison protocol

Same baselines as the CDB-F1 design. Same gate condition for "this
worked":

- **Pass.** Mean wrist_smash F1 ≥ 0.42 AND mean macro F1 ≥ 0.737
  (within 0.5 pp of LS=0.1 baseline).
- **Partial.** Mean wrist_smash F1 in [0.40, 0.42] AND macro flat.
  Worth re-running with `p = 1.0` (more aggressive mitigation) or
  `q = 2.0` (paper default compensation).
- **Fail.** Mean wrist_smash F1 ≤ 0.39 (within seed variance of
  baseline). Seesaw-F1 arm closes.

Tiebreakers:
- If CDB-F1 already lifted wrist_smash to ≥0.42, Seesaw-F1's gate
  moves to ≥0.45 (or smash F1 must lift simultaneously by ≥1 pp,
  which would be the pair-aware payoff).
- Specific-to-Seesaw success signal: **smash F1 should not drop**.
  CDB-F1 risks stealing recall from smash to lift wrist_smash;
  Seesaw-F1's pair structure should explicitly avoid that. If
  Seesaw lifts wrist_smash but smash also drops by ≥2 pp, the
  pair mechanism failed — same outcome as CDB-F1 for less effort.

## 8. Failure modes to watch

**Numerical instability in modified softmax**. The `logsumexp` over
`logits + log(S)` is numerically stable as written, but if `M_ij`
ever becomes 0 (which shouldn't happen given the `f1_floor`, but
worth guarding), `log(S)` blows up to -inf. Watch for: NaN loss
during training. Mitigation: enforce `f1_floor ≥ 0.01` strictly;
clamp `S_row.clamp(min=1e-12)` before log.

**Compensation factor + label noise**. Discussed in §2. With `q = 1`
and ShuttleSet noise level, expected to be moderate. With `q = 2`
(paper default), high risk on the noisier classes (drive, push).
Watch for: training loss diverging on noisy classes' samples after
epoch 30. Mitigation: `q = 0` (turn compensation off entirely).

**Pair-targeting dilution**. Discussed in §2. The mitigation matrix
is C×C but the bottleneck is one cell. If, after Seesaw-F1 runs,
wrist_smash F1 doesn't lift but most of the matrix entries shifted
mid-training, the C×C apparatus is doing work but not on the right
cell. CDB-F1's scalar α might just be more efficient at hitting
this specific class.

**Gradient attenuation cascade**. If `p` is too high, mitigation
factors compound: head→tail gradient gets reduced, head class
loses some training signal too (because the modified softmax
denominator is smaller, so the normalisation pushes head-class
predicted probabilities lower than they should be). Watch for:
mean macro F1 dropping vs LS=0.1 baseline; head classes (clear,
long_service, short_service) losing precision. Mitigation: drop
`p` from 0.8 to 0.5.

**Saturation at end of training**. Same as CDB-F1: by epoch 60-70,
all classes' F1 saturates, ratios → 1, mitigation factors → 1, the
loss reverts to vanilla CE. Expected. The lift comes from the
mid-training regime where F1s differ, not the end.

**Train F1 vs val F1 drift**. Same caveat as CDB-F1.

---

## Compared to the CDB-F1 design

Side-by-side on the dimensions that actually decide:

| dimension | CDB-F1 | Seesaw-F1 |
|---|---|---|
| Pair-aware | No (scalar α_c per class) | Yes (matrix M_ij + sample-level C_ij) |
| Targets the wrist_smash↔smash pair structure? | Indirectly (via low F1_wrist_smash) | Directly (M_{smash, wrist_smash}) |
| Math change | Multiplier on per-class CE term | Modified softmax denominator + multiplier |
| Implementation | ~120 lines, drop-in for `nn.CrossEntropyLoss` forward | ~180 lines, custom logsumexp forward |
| Numerical-stability risk | Low | Medium (custom denominator) |
| Backprop subtlety | None | Compensation factor must be `torch.no_grad()` |
| Published evidence on multi-class softmax classification | EGTEA 19-class video action; CIFAR-100-LT, ImageNet-LT, MNIST | ImageNet-LT 1000-class softmax (table 10), plus LVIS 1203-class detection |
| Closest-class-count match | Direct (EGTEA: 5 majority + 14 minority ≈ our 14) | Extrapolated (smallest tested is ImageNet-LT 1000-class) |
| Online-running-statistic precedent | Yes (CDB cumulative val accuracy) | Yes (Seesaw table 4: online ≈ pre-recorded ≈ from-dataset, all within 0.3 AP) |
| Sample-level focusing | Optional (focal `(1-p_t)^γ`) | Built-in (compensation factor) |
| Sample-level focusing tuneable for label noise? | Yes (`γ = 0` cleanly disables focal) | Yes (`q = 0` cleanly disables compensation; paper's own ImageNet-LT default is q=1) |
| Composition with LS | LS=0 forced | LS=0 forced |
| Risk of head-class collapse | Low (mean=1 renorm bounds α) | Medium (no global renorm; `p` too high cascades — but paper's table 5 shows p>1.0 already past peak, so the natural safe range is well-bounded) |
| Risk of "doing nothing useful" | Low (α targets the bottleneck class directly) | Medium (most of C×C matrix entries are uninformative on small taxonomy) |

### Recommendation

**Default to CDB-F1 as the first adaptive arm. Hold Seesaw-F1 as
the targeted second arm if CDB-F1 lifts wrist_smash but the lift
came at the cost of smash F1 dropping.**

The recommendation hasn't changed after the deeper paper read,
but the *reasons* have. The pre-deep-read version leaned heavily
on a "domain match" argument (CDB tested on 19-class video,
Seesaw tested on 1203-class detection). That argument doesn't
hold up — Seesaw is also evaluated on ImageNet-LT 1000-class
softmax classification (paper §4.4, table 10), which is the same
problem class as ours. Both methods have evidence at scales
larger than our 14-class setup; CDB has the closer
class-count match (EGTEA 19 vs Seesaw ImageNet-LT 1000), but
neither has direct same-scale evidence.

The remaining reasons CDB-F1 still wins as the first arm:

1. **Implementation simplicity, smaller blast radius.** Drop-in
   for `nn.CrossEntropyLoss`. Custom forward for Seesaw is
   implementable but introduces numerical-stability points and a
   backprop subtlety (`C_ij` no-grad) that have to be tested.
   CDB-F1 is days of work; Seesaw-F1 is days-plus-one-debug-
   session of work.
2. **Targeting efficiency on a small taxonomy.** On 14 classes,
   most of Seesaw's C×C matrix is uninformative because most pairs
   have similar F1. The pair-targeted advantage that the matrix
   structure offers is concentrated in a small number of cells; a
   scalar α_c per class hits the same cells more directly.
3. **Label noise tolerance.** Both can be tuned conservatively
   (CDB γ=0; Seesaw q=1, the paper's own ImageNet-LT default), so
   this is a wash.
4. **Closer-class-count published evidence.** EGTEA's 19-class
   video action is closer to 14 than ImageNet-LT's 1000 is. Not
   decisive, but a tiebreaker.

Why Seesaw-F1 second (if needed):

1. **The specific wrist_smash↔smash failure mode that CDB-F1 most
   plausibly triggers** (lift wrist_smash by stealing recall from
   smash) is exactly the failure mode Seesaw's pair structure was
   designed to prevent. If we observe that pattern in CDB-F1's
   results, Seesaw-F1 is the principled escalation, not a
   parallel option.
2. **Mechanism evidence is solid.** The deeper read shows
   Seesaw's lift mechanism is genuine and not detection-specific:
   ImageNet-LT few-shot recall lifts from 7.7 to 21.4 (+13.7 pp)
   end-to-end, with many-shot accuracy actually rising (65.9 →
   67.1). This is exactly the "lift the bottom without sacrificing
   the top" behaviour we want for wrist_smash without trading away
   smash.
3. The "how much smash drops while wrist_smash lifts" diagnostic
   is cheap to compute from CDB-F1's run output — we'll know
   immediately whether Seesaw-F1 is worth the implementation
   cost.

If the simpler arms (class-weighting smoke + manually-alpha focal)
both partially lift wrist_smash without dropping smash, Seesaw-F1
may not be needed at all. If they fail and CDB-F1 also fails,
Seesaw-F1 is unlikely to rescue (the pair-structure advantage on a
14-class problem is small enough that it won't cross a wider gap
than CDB-F1's class-scalar approach can). The narrow case where
Seesaw-F1 is the right escalation is: CDB-F1 partially works,
but at smash's expense.

Park status: design is complete and verified against the full
paper (sections 3, 4.1-4.5, tables 1-11, appendix A). Ready to
invoke after CDB-F1 has run and we have its wrist_smash-vs-smash
lift diagnostic in hand. No code edits made.
