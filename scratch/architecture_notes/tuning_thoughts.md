# Hyperparameter Tuning Strategy — Arch 1

*AI guided expansion on Isiah's look into our hyperparameter tuning options, current to 13/04/2026*

Builds on Isiah's tuning research and adapts it to our specific architectures now that the designs have firmed up.

---

## What carries forward from Isiah's research

The general principles are solid and apply across both architectures:

- **Learning rate is the most important HP.** Consensus in the literature, holds for us.
- **AdamW with cosine annealing + warmup.** Already in use for BST. Standard for fine-tuning pretrained models.
- **Label smoothing at 0.1.** Low risk, prevents overconfident predictions. Already active.
- **Pretrained backbone for 3D CNNs.** Training from scratch on ~33k samples doesn't work — fine-tuning is the way. This applies to both the X3D-S wrist crop extension (Arch 1) and whatever 2+1D backbone Scott settles on (Arch 2).
- **Differential learning rates** for pretrained backbone vs classification head.
- **Mixed precision (torch.cuda.amp)** — essential on V100 16GB for any video model.
- **CE + inverse-frequency class weighting** as loss baseline. Focal loss (gamma=2) if minority classes still struggle, but always compare against the CE baseline.

For Arch 2, since the design is still being worked out, the general fine-tuning guidance (LR range 1e-4 to 1e-3, pretrained backbone, differential LRs, trial budget of 4-12 hrs) is a reasonable starting framework that can be refined once the architecture is locked.

---

## Where things diverge for Arch 1

The tuning research was written against a generic "keypoint transformer + 3D CNN" framing. Now that Arch 1 has a concrete design (BST-CG-AP + X3D-S wrist crop fusion, 12 classes, amateur augmentation), some specifics need updating.

### BST-CG-AP is a specialised architecture, not a tunable generic transformer

BST has three distinct transformer types (Temporal, Cross, Interactional), each with fixed roles and fixed depths. The architectural parameters (d_model=100, n_head=6, d_head=128, depth_tem=2, depth_inter=1) are tightly coupled across the TCN, cross-transformer, interactional transformer, and position fusion components. Changing d_model, for instance, means rewriting multiple interacting modules — it's a research-level architecture change, not something to hand to Optuna.

Similarly, batch size is 128 (BST is lightweight on skeleton data), and training runs ~350 effective epochs with 300-epoch early stopping patience, taking 5-15 hours on V100. This is longer than the 30-90 minute estimate for a generic lightweight transformer.

### We already know the BST-CG-AP hyperparameters

The published/tested BST config (LR=5e-4, dropout=0.3, batch=128, warmup=400, label_smoothing=0.1, AdamW, cosine schedule) is our starting point. We don't need to search for these — they're established. The tuning work is on the **new components**: X3D-S, fusion, and augmentation. And once those are added, the whole model trains end-to-end, so the BST params may shift — but from a known-good starting point, not from scratch.

---

## Ideal tuning budget (if time were not a constraint)

If we had the GPU time, a proper Optuna (TPE) search with median pruning would look like this. Note: this excludes augmentation tuning, since we lack out-of-sample amateur data to validate against — augmentation params are better set from analysis than searched (see caveat below).

**Per-trial training time (full run):**

| Component | V100 16GB | A100 40GB |
|---|---|---|
| BST alone, ~350 effective epochs | 5-15 hrs | 3-8 hrs |
| X3D-S standalone, ~30 epochs fine-tune | 1-3 hrs | 0.5-1.5 hrs |
| Full model e2e (BST+X3D-S+fusion), warm-started, ~150-250 epochs | 5-18 hrs | 3-10 hrs |

With Optuna median pruning killing bad trials early, average trial time drops to roughly 60-70% of a full run.

**Tuning phases:**

| Phase | What's being tuned | Trials | V100 16GB | A100 40GB |
|---|---|---|---|---|
| BST re-tune (e2e, all params live) | LR, dropout, weight decay, warmup, label smoothing | 10-15 | 2-5 days | 1-3 days |
| X3D-S standalone | Fine-tune LR (backbone + head), head dropout | 5-8 | 0.5-1 day | ~half day |
| Fusion variant (a), e2e | All BST params + X3D-S LR + MLP dims/dropout + global e2e LR | 10-15 | 2-6 days | 1-3 days |
| Fusion variant (b), e2e | All above + fusion module HPs (bottleneck tokens, cross-attn layers, insertion depth) | 12-18 | 3-8 days | 2-4 days |

**Ideal totals (excluding augmentation search):**

| Path | V100 16GB | A100 40GB |
|---|---|---|
| Through variant (a) | ~5-12 days | ~3-7 days |
| Through variant (b) | ~6-14 days | ~4-8 days |

---

## Practical constraint: time budget

With Phase 4 starting ~April 25 and the May 17 milestone, the ideal budget above doesn't fit. The proposal below uses literature-informed defaults instead of search, accepting maybe 1-2% accuracy cost vs an exhaustive sweep.

---

## Optimised tuning plan for Arch 1

### Data prep (CPU, runs in parallel with GPU training)

**Wrist crop extraction** — can start immediately, doesn't depend on the model:
- Read MMPose keypoints from existing .npy files (right wrist = COCO index 10)
- Compute crop radius from torso height (k=1.0 to start), apply One Euro Filter smoothing
- Extract and resize to 112x112, 13 frames per clip
- ~430k frame crops across 33k clips, few hours on CPU
- **Run this on Day 1 while BST baseline trains on GPU**

### Training pipeline

| Step | What | Config | Runs |
|---|---|---|---|
| 1. BST baseline | Known-good params, 12-class taxonomy, full patience | LR=5e-4, dropout=0.3, batch=128, warmup=400, label_smoothing=0.1 | 1 |
| 2. Augmentation | One config from amateur dynamics analysis. If >2% pro degradation, try a weaker variant. Reduced patience (~100 epochs) to save time. | Stretch range + probability informed by amateur play analysis | 1-2 |
| 3. X3D-S standalone | Fine-tune K400-pretrained X3D-S on wrist crops (12-class). Standard transfer recipe. | Freeze backbone 5 epochs, unfreeze, backbone LR=1e-4, head LR=1e-3 | 1 |
| 4. Fusion (variant a) + e2e | Late concat + MLP with reasonable defaults. End-to-end fine-tune with all params live. | MLP hidden=256, dropout=0.3, e2e LR=1e-4 | 1 |
| 5. Final run | Best config, full patience | — | 1 |
| *6. Optional: variant (b)* | *Cross-attention / transformer fusion. Only if (a) plateaus and time permits.* | — | *0-1* |

**X3D-S fine-tuning note:** Fine-tune directly on the 112x112 wrist crops, not full badminton frames. That's what X3D-S will see during inference. K400 pretraining already provides general motion understanding; the fine-tune step specialises it to the racket/wrist view.

**Dominant hand note:** The wrist crop extraction targets the dominant (racket) hand, which is the right wrist (COCO index 10) by default. Left-handed players need the crop taken from the left wrist (COCO index 9) instead, and the crop should be horizontally flipped for consistency so X3D-S always sees the same orientation. This requires knowing which hand each player uses. Two options:
- **Heuristic detection script:** Analyse pose keypoints to infer dominant hand automatically (e.g., which wrist moves more during swings, or which side the racket-like motion is on). More robust and generalises to unseen players.
- **Manual lookup (fallback if time-poor):** The players in ShuttleSet are known pros — their dominant hand can be looked up and stored as a player metadata column. Faster to implement, but doesn't generalise to new players at inference time.

### Parallelised timeline

| Day | GPU | CPU (parallel) |
|---|---|---|
| 1 | BST baseline (8-12 hrs V100 / 4-7 hrs A100) | Extract wrist crops from 33k clips |
| 2 | Augmentation run (4-10 hrs, reduced patience) | Crops done, ready for X3D-S |
| 3 | X3D-S standalone (1-3 hrs) then fusion e2e (8-18 hrs) | — |
| 4 | Final best config, full patience (8-12 hrs) | — |
| *5* | *Optional: variant (b) if time allows* | — |

### Time estimates

| | V100 16GB | A100 40GB |
|---|---|---|
| Core pipeline (steps 1-5) | ~2-3 days | ~1-2 days |
| With variant (b) | ~3-4 days | ~2-3 days |

---

## Augmentation caveat

We don't have out-of-sample amateur footage to validate against. Augmentation parameters are tuned against the pro validation set — the measurable signal is "doesn't degrade pro performance," not "improves amateur performance." This is a known limitation worth flagging in the report. The augmentation is a principled methodological step informed by amateur play dynamics analysis, but empirical validation is one-sided.

---

## What's tunable at each stage — summary

**BST baseline:** Nothing to tune. Run with established params.

**Augmentation:** Temporal stretch range (e.g. [1.0, 1.7] vs [1.0, 2.0]), stretch probability, camera augmentation intensity. Informed by amateur dynamics analysis, not searched.

**X3D-S:** Fine-tuning LR (backbone vs head), head dropout. Standard transfer learning recipe.

**Fusion + e2e:** Fusion MLP dimensions and dropout (variant a), or fusion module architecture HPs (variant b). Global end-to-end LR. BST params may shift from the known baseline at this stage since all layers are jointly optimised.

**Loss function:** Start with CE + inverse-frequency class weighting + label smoothing 0.1. Try focal loss (gamma=2) only if specific classes underperform after the baseline run. If using Mixup/CutMix on the video stream, test focal loss interaction carefully (Mixup produces soft labels; focal loss modulates on confidence — the combination needs explicit validation).
