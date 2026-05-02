# Hyperparameter sweep speculations

*Started 2026-04-30, after Phase-2 sanity train run 1 (combo C) showed
seed variance had tightened ~2.5x on macro/acc vs the Phase-1
baseline. With cleaner data, hparam ablations are now more readable:
small effects that sat under the Phase-1 noise floor should be
detectable.*

This doc speculates on which static defaults are worth exploring
once the focal-loss / class-weighting / label-smoothing / augmentation
runs settle. The framing is: what do these knobs actually do,
which are likely to move the needle, and what would a defensible
sweep budget look like? It also walks through the frame-zeroing
question raised on 2026-04-30, with the real Phase-2 numbers from
`validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`.

## TL;DR

Plain-prose summary of the conclusions the rest of this doc backs
up. Specifics, numbers, and citations are in the sections below.

**Currently inherited, never tuned**: `weight_decay=0.01`,
`drop_p=0.3`, `RandomTranslation` magnitude+prob, `tcn_kernel_size=5`
with dilation 1/3, `depth_inter=1`, `(DL, DA, n_head)` sizing,
augmentation set (BST kept TemPose's shifting, dropped TemPose's
flipping). All BST-paper defaults; neither BST nor TemPose swept
any of these knobs (TemPose did sweep depth and `(DL, DA)`, BST
did neither).

**Likeliest single-knob lever after the loss / class-weighting
work finishes**: weight decay. Cheapest-of-all to test because it's a
one-line config swap with no code change. Sane sweep [0.0, 0.05,
0.1] vs the AdamW default 0.01.

**Augmentation status**: cheap 3-cell sweep planned (current /
tighten / off), plus a flipping add-back cell on the side-collapsed
taxonomies. The current 0.3 magnitude is aggressive on coords that
already encode side identity, so probably-defensible-but-unverified.
Expected lever is small to medium. Plays before dropout in the
runbook because the two share regularisation budget and you want
to fix augmentation first.

**Architecture-side priors from the verified TemPose/BST ablations**:
`(DL=100, DA=128)` is empirically validated as a winner on Bad OL,
so the only sane geometry experiment is *moving up* the size curve
(testing whether ShuttleSet's larger N now unlocks the (200, 256)
cell that overfit on Bad OL). Depth-wise, BST inherits `(LT=2,
LN=1)` but TemPose's depth sweep favours `(LT=2, LN=2)` by 0.7%, so
flipping `depth_inter=1 → 2` is a single high-confidence cell.

**Frame-zeroing redesign**: the system currently zeros shuttle on
any pose-fail frame even when TrackNet had a perfectly good shuttle
detection (~14k frames affected). Decoupling that policy is cheap
(re-collation only, one config flag). The real-data numbers say
shuttle fails 7x more often than pose post-sticky-anchor (6.34%
vs 0.93%), so most "frame is partially good" frames involve good
pose with bad shuttle, which the current policy already handles
correctly (no zeroing of pose). The marginal improvement from
decoupling shows up on pose-fail frames where shuttle was OK,
which is small per-frame but clip-clustered.

**Mask channel design (revised after looking at the real per-class
failure rates)**: two channels, not three. `shuttle_missing` at
6.34% positive rate is comfortably learnable. Combined
`pose_missing_either_slot` at 0.93% is borderline per-frame but
rescued by clip-clustering and stroke-correlation. Splitting into
separate Top vs Bottom pose masks halves each rate to
sub-learnability; collapse them.

**BST paper inferences worth acting on**: (1) variable-length
clipping lifts min-F1 by ~5.7 percentage points over fixed-width;
our `between_2_hits_with_max_limits` is a max-capped variant, so a
quick clip-length distribution check is the cheapest possible
experiment in the whole doc. (2) BST-AP-only beats BST-CG-AP on
min-F1 in the paper's Table 1, while BST-CG-AP wins overall accuracy.
If focal loss specifically targets tail performance, BST-AP-only
under focal is worth a re-check. (3) 2D pose beats 3D pose for
badminton on every metric; rules out the 3D pose direction.

**Run-budget total**: ~105-140 A100-hours including the focal /
LS / aug work. Roughly two weeks of overnight runs at one combo
per night.

**Shuttle-missing diagnosis (verified 2026-04-30 via the three
pre-flight scripts)**:

- Off-screen-high hypothesis is **confirmed**: 61.6% of gap
  boundaries cluster in the top 10% of frame; 72.3% of post-gap
  re-appearances cluster there. Gap-length distribution shows
  inpaint is not exceeded (only 1 gap >60 frames in 32k clips);
  85% of missing frames sit in the 11-60 frame band of "shuttle
  genuinely not in any pixel". This is a sensor-level limitation,
  not a model limitation.
- **Shuttle-miss vs F1 is positively correlated** (Pearson +0.516
  on combo A nosides), opposite of the predicted direction. The
  high-miss classes are pose-distinctive serves / clears / lobs at
  F1 ~0.95-0.99; the bottleneck classes (wrist_smash, drive, push,
  cross_court_net_shot) sit at <1% miss rate because their
  trajectories stay in frame. **The shuttle stream is available
  where it's most needed; the model just isn't using it well.**
- This is a classifier-side problem, not a data-side problem.
  Combined with run 2's gate-failure pattern (cleaner data helped
  the head, hurt the tail), the signature matches **label
  smoothing fighting cleaner data on rare classes**: LS=0.1 taxes
  confidence uniformly, but gradient signal is proportional to
  support count, so 600-sample classes lose more than 2,400-sample
  classes to the same smoothing constant.

**Implications for the runbook (revised 2026-05-02 after the
loss-side experiments closed and capacity-bottleneck research
landed)**:

- **LS sweep wrapped; LS=0.15 won.** Three runs on combo A
  nosides. LS=0.0 disproved the rare-class-tax hypothesis (mean
  wrist_smash -1.6 pp vs LS=0.1 baseline). LS=0.15
  (`run_20260501_073430`) lifted mean wrist_smash 0.375 → 0.417
  (+4.2 pp) with head metrics flat and the wrist_smash range
  tightening 0.159 → 0.066. **LS=0.15 kept active for downstream
  runs.** LS=0.05 skipped (two bracketing data points enough);
  LS=0.2 deferred.
- **Class-weighting smoke test landed without a mean shift but
  with a new ceiling.** `run_20260501_110525` (combo A nosides +
  LS=0.15 + `class_weights={'wrist_smash': 2.0, 'smash': 2.0}`)
  added essentially zero on the central tendency past LS=0.15
  alone (+0.005 mean wrist_smash, inside seed variance). But the
  upper end of the achievable distribution shifted clearly: S2
  hit wrist_smash 0.518 — first nosides serial to clear 0.50, and
  a +6 pp project-wide ceiling break (prior best 0.46 LS=0.1 S5).
  S4 of the same run set new ceilings on macro 0.756, acc 0.777,
  and drive 0.66. **Bimodal seed distribution**: one seed found a
  wrist_smash basin no prior nosides serial accessed, three
  others stayed in the LS=0.1-baseline range around 0.37-0.40.
  Diagnosis: static loss reweighting moves the *ceiling* but not
  the *mean*.
- **Basic focal skipped per project decision.** Vanilla focal
  `(1-p_t)^γ * -log(p_t)` and manually-alpha focal `α[c] *
  (1-p_t)^γ * -log(p_t)` are the same lever as class-weighted CE,
  just gated by per-sample confidence. Adding `(1-p_t)^γ` on top
  of pair-balanced 2.0/2.0 alpha would hit the same central-
  tendency ceiling that the smoke test already hit. Skipped.
- **CDB-F1 fully explored; ceiling firmly mapped.** Per-class
  alpha = `(1 - F1_c)^τ` with EMA-smoothed running per-class
  train F1, composed with focal `(1-p_t)^γ`. Design verified
  against the ACCV 2020 paper at
  `scratch/architecture_notes/class_f1_focal_design.md`. First
  run `run_20260501_164658` (tau=1, gamma=1) lifted mean
  wrist_smash +8.7 pp on the LS=0.1 baseline; push +6.7 picked
  up automatically; bimodal-seed problem solved (range 0.140 →
  0.073). Cost: smash -5.5 (pair-confusion), small symmetric
  drops on rush / drive / drop / services. Four follow-ups —
  gamma=0 (`run_20260501_192113`), tau=0.5
  (`run_20260501_192519`), pair-cap (`run_20260501_230252`,
  ratio=0.7 between alpha[smash] and alpha[wrist_smash]), and
  gamma=2 (`run_20260502_075808`) — all traded ws back for smash
  without macro moving. The first run is the floor-lift sweet
  spot; smash pair-confusion is structural and scalar-per-class
  alpha can't resolve it. No CDB run breaks the val/test plateau
  at 0.74-0.75 macro.
- **Capacity-bottleneck research done (2026-05-02).** Writeup at
  `scratch/architecture_notes/model_capacity_bottleneck_question.md`
  argues the plateau is data-bound and signal-bound rather than
  capacity-bound. BST at 1.85M params on 32K clips sits in the
  converged 1-3M zone for skeleton-AR. Famous from-scratch video
  AR baselines (X3D-M 3.76M, MoViNet-A0 3.1M) are 1.5-2x BST;
  flagship transformers (Video Swin-T) need ImageNet pretraining.
  Pure widening expected 0-2 pp on test macro. Two cheap
  capacity bumps queued for confirmation: `mlp_head` 400 → 768
  and `d_model` 100 → 128 with `d_head` trim 128 → 32 (the
  Voita-style head trim pairs with the residual-stream widen).
- **Pair-aware Seesaw-F1 held.** Design at
  `scratch/architecture_notes/seesaw_f1_focal_design.md` (verified
  against the CVPR 2021 paper). Held as a targeted second
  loss-side arm only if a future signal-side gain reopens the
  smash↔ws pair-confusion question.
- **Mask-channel arm still demoted.** Variant 2 design still
  works but would help services / clears / lobs (already at
  0.95+ F1) rather than rescuing the bottleneck classes.
- **Trajectory extrapolation flagged as a future direction** for
  the 11-60 frame off-screen-arc gaps, but not near-term because
  the bottleneck isn't on the high-arc classes.

**Highest-priority near-term experiments (revised 2026-05-02)**,
in order: capacity-bump confirmation runs (`mlp_head` 400 → 768
then `d_model` 100 → 128 with `d_head` trim, both vs
`run_20260501_164658`); horizontal-flip augmentation with COCO
joint-pair swap (the natural intermediate before X3D-S, useful
regardless of capacity-bump outcome); X3D-S fusion build (long-term
primary direction; the lever capacity research argues actually
addresses the data/signal bottleneck); weight decay sweep;
`depth_inter=1 → 2` run.

## Inherited-vs-tuned audit

Per the BST paper (`arxiv:2502.21085`, Appendix D) the published
training settings are: AdamW, batch 128, lr 5e-4, weight_decay 1e-2,
label_smoothing 0.1, cosine annealing with warmup (warm-up 400, cycles
0.25, 1600 epochs / 300-epoch early stop), random shift augmentation
prob 0.3 with shift uniformly in [-0.3, 0.3). Architectural details
(d_model, d_head, n_head, depth_tem, depth_inter, drop_p,
mlp_d_scale, tcn_kernel_size) are inherited from TemPose
(CVPR 2023W) and are not separately ablated in the BST paper. The
BST ablations vary only the module set (PPF / CG / AP) and the
multimodal input combination, not the underlying transformer or TCN
sizing. (Verified by full-paper read on 2026-04-30; paper PDF at
`https://arxiv.org/abs/2502.21085`.)

So everything in `bst_train.py:62-77` and `model/bst.py:140-141`
that is not `taxonomy`, `seq_len`, `pose_style`, the CG/AP module
flags, the LR-schedule retune (Q4) and the aux schedule is on
its paper-default, never tuned here.

The TemPose paper (the source for the architectural defaults)
ablation history is now verified directly from the paper text
(`Ibh_TemPose_..._CVPRW_2023_paper.pdf`, Tables 1-4 and Section 4.2):

**TemPose training settings (Table 1)** are: AdamW, batch 64, lr 1e-4,
warm-up 25%, cosine decay, label smoothing 0.1, **flipping 30%**,
random shifting 30%, dropout 0.3, weight decay 0.01.

So BST inherited TemPose's dropout=0.3, weight_decay=0.01,
label_smoothing=0.1, random_shifting=0.3 directly. **Neither paper
swept dropout, weight_decay, or TCN kernel size**; both treat them
as fixed defaults. (TCN kernel is documented in Section 3.4 as
"kernel size of 5 and stride 1" with dilation 1 then 3 across the
two layers, which the BST repo reproduces in `tempose.py:139` via
`dilation = i * 2 + 1`.) BST changed the LR (5e-4 vs 1e-4), batch
(128 vs 64), and warm-up scheme (cycles + absolute steps vs %
warm-up) and dropped TemPose's flipping augmentation, keeping only
the shift augmentation.

**TemPose architectural ablations (Tables 2 and 4)** are extensive:

- **(DL, DA) joint sweep** (Table 2): TemPose-V at (75, 100) and
  (200, 200); TemPose-TF at (50, 75), (100, 128), (200, 256);
  TemPose-NF at the same three sizes as TF. The winner across all
  variants is **TemPose-TF at (DL=100, DA=128) with 90.7%** on the
  Bad OL dataset. The (200, 256) variant of TF dropped to 88.0%,
  indicating overfitting at larger sizes. **BST inherits the
  empirical winner, not a guess.**
- **(LT, LN) depth sweep** (Table 4, 10 cells): (1,1)=89.7,
  (1,2)=89.9, (2,1)=90.0, **(2,2)=90.7**, (3,3)=88.3, (4,4)=86.6,
  (6,2)=85.5, (2,6)=86.1, (6,6)=85.4, (8,8)=85.2. Performance
  drops sharply past (3,3); the paper explicitly attributes this
  to overfitting on the small Bad OL dataset.
- **Joint+Bone vs Joint-only** (Table 3): J+B beats J for both AcT
  baseline and TemPose-V, in line with prior skeleton-action
  literature.

**One inheritance discrepancy to note: BST uses `depth_tem=2,
depth_inter=1`** (per `bst_common.build_bst_network` defaults) but
TemPose's empirical winner from Table 4 is `(LT=2, LN=2)` at
90.7%. The `(LT=2, LN=1)` cell scored 90.0%, so BST is
inheriting a configuration ~0.7% below the TemPose optimum,
presumably for parameter-count savings. Plausibly worth a one-cell
"flip depth_inter to 2" arm before doing any wider depth sweep,
since the TemPose data already says (2,2) > (2,1) on a similar
dataset.

## What the BST paper's ablations and tables actually constrain

Direct read of the BST paper (`BST_..._Racket_Sports.pdf`,
`arxiv:2502.21085v3`) including the supplementary, against our
results. Most useful constraints fall into five buckets.

### Where the search space is empirically narrowed (verified from paper text)

1. **2D pose beats 3D pose for badminton** (Supplementary B.2,
   Table E). Across BST-0 / BST-CG-AP / TemPose-SF / TemPose-TF,
   2D joints consistently outperform 3D joints by ~0.5-3% on every
   metric. Cause: the HPE model is trained on a general HPE dataset
   so 3D-projected joints "exhibit a broader range of poses than
   those specific to badminton players", with explicit visual
   evidence (Fig. C) of the 3D pose mis-estimating the player's
   facing direction. **This rules out MotionBERT / 3D pose
   as a useful direction unless we get a badminton-specific 3D
   model.** Already aligned with the project's `use_3d_pose=False`
   default; the speculation has empirical backing.

2. **TemPose's (DL, DA) sizing is the empirical winner.** TemPose
   Table 2 swept three sizes per variant; (DL=100, DA=128) won on
   Bad OL. BST inherits this and does not re-sweep. Combined with
   our earlier observation that ShuttleSet has ~33k samples vs
   Bad OL's ~15k, the only sane geometry experiment is **moving up
   the size curve** (where TemPose hit overfitting on the smaller
   set), not changing the d_head ratio.

3. **wrist_smash is the structural confusion class.** Supplementary
   B.1 confusion matrix on 35-class ShuttleSet shows the 2nd-3rd
   strokes (top smash and top wrist_smash) and 19th-20th strokes
   (bottom smash and bottom wrist_smash) confused with each other,
   plus "1st stroke (top return net) often misclassified as the
   13th stroke (top defensive return drive)". The paper's text:
   "the model has difficulty distinguishing between subtle
   variations in the strokes with similar hitting characteristics."
   **This validates wrist_smash specifically as the gate metric on
   `une_merge_v1` taxonomy and confirms our framing.** It also
   suggests the return_net / defensive_return_drive pair as a
   secondary confusion to keep an eye on.

4. **CG/AP modules act as overfitting regularisers, not just
   feature extractors** (Supplementary A, Figure A loss curves).
   TemPose-TF* val loss visibly increases past ~100 epochs while
   train loss continues down (textbook overfitting). BST-CG-AP val
   loss is more stable and continues decreasing longer, despite
   training for 1600 epochs. **This validates CG/AP as a paired
   inductive-bias regulariser** and matches the project's
   already-completed CG/AP scheduling ablations. Constraint: don't
   try to fully zero out CG/AP regardless of focal-loss success;
   their job overlaps with augmentation/dropout but isn't
   identical.

5. **Training regime is very different from ours.** BST paper
   trains at n_epochs=1600, num_cycles=0.25, warm_up=400,
   early_stop=300. Our retune (Q4 in `arch_1_directions.md`)
   compressed to n_epochs=80, num_cycles=0.5, warm_up=100,
   early_stop=40. **Inferred constraint**: WD and dropout sweeps at
   our compressed schedule will land in a different effective
   regularisation regime than the paper's 1600-epoch regime. Our
   results will be specific to our schedule, not directly
   transferable to the paper's. This is fine for what we need but
   worth noting if anyone reads the sweep results as a critique of
   the paper's defaults.

### Where the BST paper's tables suggest concrete sweep priors

6. **CG and AP behave differently on the F1 tail** (Table 1, fixed-
   width strategy, 25-class ShuttleSet, the closest match to our
   `merged_25 + split_bst_baseline + dropunk` combo C):

   | Variant | Acc | Macro-F1 | Min-F1 | Top-2 |
   |---|---|---|---|---|
   | TemPose-TF | 0.8189 | 0.7943 | 0.4928 | 0.9496 |
   | BST | 0.8206 | 0.7952 | 0.5331 | 0.9499 |
   | BST-CG | 0.8210 | 0.7954 | 0.5296 | 0.9481 |
   | **BST-AP** | 0.8229 | **0.7992** | **0.5532** | 0.9484 |
   | BST-CG-AP | **0.8254** | 0.7983 | 0.5196 | **0.9503** |

   **Inferred speculation**: BST-AP alone has the highest min-F1
   (0.5532), beating BST-CG-AP (0.5196) by 3.4 percentage points
   on the bottleneck classes. CG-AP wins on accuracy and top-2 by
   pulling the centre of the distribution up. Combining CG and AP
   *helps the head and hurts the tail* in this table. This is buried
   and not commented on by the authors. If our focal-loss /
   class-weighting work lifts tail performance specifically, **the
   paired CG-AP module set may need re-evaluation against AP-only
   under the new loss**; it's possible the CG denoising step that
   helps overall accuracy is also denoising signal that the focal
   loss specifically wants to amplify on rare classes.

   **Caveat**: this is a 2.7-percentage-point min-F1 spread on a
   single test split with no seed-variance reported in the paper.
   Single-seed results below ~3% spread are within typical
   transformer noise on small datasets. This is a hypothesis from
   one table cell, not a verified effect.

7. **Variable-length clipping lifts min-F1 substantially**
   (Tables 1 vs 2, BST-CG-AP). Fixed-width acc 0.8254, macro
   0.7983, min 0.5196, top-2 0.9503. Variable-length acc 0.8322,
   macro 0.8097, min **0.5762**, top-2 0.9594. **The min-F1 jumps
   5.7 percentage points just from changing the clipping
   strategy.** Our project uses `between_2_hits_with_max_limits`
   which is a max-capped version of the variable-length scheme. If
   our max-limit cap is biting too tight, we may be losing some of
   that lift. **Inferred speculation**: worth verifying that our
   `seq_len=100` cap isn't truncating clips that the variable-
   length scheme would have run longer. A quick check of clip
   lengths post-clipping would confirm or rule out this concern.
   This is the cheapest "data side" experiment in the doc and may
   be larger lever than any of the hparam sweeps.

8. **CG retains its lift better at small sample sizes** (Table 3 /
   Table D, 25%-of-data condition). At 25% training data on
   25-class fixed-width: BST-CG min-F1 0.6334, BST 0.6196,
   BST-AP 0.6302. CG-only takes the macro and min wins at low
   sample. **Inferred speculation**: CG's denoising signal is more
   load-bearing when class supports thin out. This pairs with point
   6: at full data the AP signal dominates the tail, but at scarce
   data CG dominates. We have full data, so AP's the stronger arm
   for our regime.

### What the BST paper does *not* constrain (still open territory)

The BST paper treats the following as fixed defaults with no
ablation:

- Dropout (`drop_p=0.3`): zero text in Section 3, zero entry in
  Supplementary D training settings. Inherited from TemPose by
  unspoken convention. Open to sweep.
- Weight decay (1e-2): set at the value, not swept. Open.
- TCN kernel size (5) and dilation pattern (1, 3): fixed at
  TemPose's choice; no ablation. Open.
- Transformer depth (`depth_tem=2, depth_inter=1`): inherited from
  TemPose, no BST sweep. The TemPose Table 4 sweep (10 cells) is
  the relevant prior; it favours (2, 2) over (2, 1) by 0.7%. Open
  for the one-cell test described elsewhere.
- Head count (`n_head=6`), MLP scale (4), embedding dimensions:
  inherited, not BST-swept.
- Augmentation magnitude (shift range / prob): used at TemPose's
  defaults; flipping was *dropped* (see RandomTranslation section
  below).
- Loss function (cross-entropy + label smoothing 0.1): set, not
  swept. Open for focal / class weighting.
- Frame-zeroing policy: paper text mentions "If there were less
  than two people in the court, we cleared the information (poses
  and shuttlecock trajectory) of that frame to zero, since that
  frame was definitely not in a standard camera perspective."
  (Supplementary E.) That's the exact OR-zeroing behaviour
  documented in our frame-zeroing redesign section. **Paper-side
  validation that the current zeroing is unexamined inheritance,
  not a deliberately tuned choice.**

So our sweep candidates are real gaps, not "things the authors
already ruled out". The BST contribution is architectural (CG/AP
modules + clipping strategy), not optimisation-side.

## Per-knob walkthrough

### `weight_decay` (AdamW default 0.01)

**What it does.** L2-style penalty on the parameter norm, applied
decoupled from the gradient update in AdamW. Pulls weights towards
zero each step. Larger values shrink the model's effective capacity;
smaller values let the model express more without paying a norm tax.
Andriushchenko et al ("Why Do We Need Weight Decay in Modern Deep
Learning?", arxiv:2310.04415) argue WD is "never useful as an
explicit regulariser but instead changes the training dynamics in a
desirable way", and for small-data fine-tuning it stabilises training
and balances the bias-variance tradeoff rather than acting like a
hard capacity cap. Practical literature consensus on small-data
transformers: 0.01 to 0.1, with 0.05-0.1 favoured when overfitting is
visible.

**Genuinely worth sweeping.** Yes. Three reasons:
- We've never moved off the AdamW default. This is the cheapest
  unmoved knob in the whole stack.
- The Phase-2 acc/macro mean is essentially tied with Phase-1 but
  min F1 is slightly worse. That's a hint the model is finding
  almost-the-same solution but the bottleneck classes have shifted
  slightly. WD changes which solution gets found, not (mostly) how
  much it overfits.
- WD interacts with LR and with dropout (next item) more than with
  anything else. Sweeping WD alone after the LR retune is now sound
  because the LR is locked.

**Sane sweep.** [0.0, 0.01, 0.05, 0.1], single arm, 3 serials each,
holding everything else at current values. 12 serials total, ~12-15
hr at A100 5-serial pace.

### `drop_p` (BST default 0.3, applied uniformly)

**What it does.** Standard dropout: per-step, randomly zero a
fraction of activations during training. Used in this codebase as a
single uniform value across attention output, FF, embedding, MLP,
TCN convs (`model/bst.py:56`, `:176`, `:181`, `:194`, `:199`,
`tempose.py` for TCN). The model has a dropout opportunity at
roughly every functional block.

Standard transformer dropout in NLP transformer baselines is 0.1
(per the original Transformer paper; widely confirmed across
implementations). LayerDrop / structured dropout literature
(Fan et al. 2019, openreview.net/pdf?id=SylO2yStDr) uses 0.2 as the
default rate. 0.3 sits on the high side of the typical band, defensible
when the data is small and noisy; less defensible now that Phase-2
sticky_anchor data is meaningfully cleaner and seed variance has
tightened.

**Genuinely worth sweeping.** Yes, but with a caveat. The caveat is
that dropout and **augmentation strength** are the same kind of
regulariser at different layers of the stack: "make the model see
less of the truth at training time so it generalises better". They
share the same regularisation budget. So sweeping `drop_p` alone
without coupling to the augmentation knobs is partial. The BST paper
chose 0.3 dropout at 0.3 augmentation prob; if either lifts, the
other might want to come down.

The likely direction is **down** (towards 0.2 or 0.15), given the
cleaner data and the tighter seed spread. Worth a sweep against the
focal-loss arm specifically because focal also touches the
confidence-tax surface and might pair non-trivially with dropout
strength.

**Sane sweep.** Marginal first: [0.15, 0.2, 0.3], 3 serials, 9
serials total. Then if the marginal sweep finds a plausible region,
do a small joint cell with weight_decay (4 cells, 3 serials each,
12 serials). Together: ~21 serials.

### `RandomTranslation` (`trans_range=(-0.3, 0.3)`, `prob=0.3`)

**What it does.** With probability 0.3, adds a uniform random xy
shift in [-0.3, 0.3] to all pose / shuttle / position arrays in a
batch (`shuttleset_dataset.py:121-136`). The shift is the same for
every sample in the batch (the `RandomTranslation_batch` variant) so
the relative geometry within a sample is preserved; only the
absolute position on court moves.

**Note: BST dropped TemPose's flipping augmentation.** TemPose
Table 1 lists "Flipping 30%" alongside "Random shifting 30%" as
its augmentation set. The BST paper Appendix D and the BST repo
only carry forward random shifting. So BST is running with **half
the augmentation** TemPose's hparams used. Whether this was
intentional (perhaps the side-coordinate is too informative for
flipping to make sense, since flipping a Top player gives a
Bottom-side pose with Top-coded label) or an oversight is unclear.
Worth flagging as a candidate to add back, with the caveat that
flipping changes player-side semantics and would need to combine
with a label flip for the Top_/Bottom_ taxonomies (or just be
skipped on those, and only used for `merged_25` /
`une_merge_v1_nosides` where side is collapsed).

**Worth sweeping?** Probably not as a primary lever, and your
intuition is correct that uniform-origin court coords make this less
necessary than it would be on a less-canonicalised dataset. The
augmentation is doing one of two jobs:

1. *Position invariance*: forcing the model not to memorise that
   "Top_short_service starts the player at (0.5, 0.95) above the
   service line" and so on. ShuttleSet poses are normalised onto
   homography-rectified court coords, so the player's standing
   position is structurally informative (Top vs Bottom is literally
   y-coord), and a too-aggressive shift will scramble that.
2. *Implicit jitter*: adding stochastic input noise to widen the
   training distribution. Useful regardless of how canonical the
   coords are, since real cameras drift, the homography fit
   wobbles, and pose-projection error is meaningful.

The current 0.3-magnitude shift is a substantial perturbation
relative to the half-court height of 0.5 (so a 0.3 shift can land a
Bottom player on the Top court, depending on starting position). The
dataset class then doesn't clip the shifted coords to [0,1], so
out-of-court coords flow into the TCN. This is *probably* fine
because the shifts only fire 30% of the time and the model can
average them out, but it's worth noting that the augmentation is
already aggressive on a coordinate frame where the absolute position
carries side information.

**Could it be removed?** Plausible. If the augmentation is mostly
doing job (1) and the side-coord signal is informative, removing it
might lift accuracy a small amount and the model would lean harder on
the position prior. The stronger argument is that it's cheap to test
both ways:

- *Tighten*: range to (-0.15, 0.15), prob to 0.5. Same expected
  variance contribution but smaller per-event displacement; a
  defensible setting given the canonicalised coord frame.
- *Remove*: range = (0, 0), or just disable the transform. Tells you
  whether the augmentation is doing anything for this dataset.

**Sane sweep.** Three settings: current (paper default), tightened
(-0.15, 0.15) at prob 0.5, off. 3 × 3 = 9 serials, ~9-12 hr.

### `d_model=100`, `d_head=128`, `n_head=6`

**What's unusual about this.** The standard transformer convention is
`d_head = d_model / n_head` so that `d_cat = d_head * n_head =
d_model` and the Q/K/V projections preserve dimensionality
(d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html;
multiple secondary sources confirm). Standard example: d_model=512,
n_head=8, d_head=64, d_cat=512.

Here we have d_model=100, d_head=128, n_head=6, so
d_cat = 768 ~ 7.7x d_model. The QKV linear maps are projecting
*up* into a much wider space, attention happens there, then the
output projection compresses back to d_model
(`model/bst.py:43-57`). That's an inverted-bottleneck-ish structure:
narrow input/output, wide internal compute. Same shape as the X3D /
MobileNet block pattern, though for very different reasons (X3D's
inverted bottleneck is about depth-wise conv efficiency; here it's
just an attention-projection sizing decision).

**TemPose did sweep this jointly with the model variants** (Table 2):
three (DL, DA) sizes per variant, with (100, 128) winning across
TF and NF on Bad OL, and the (200, 256) cells dropping ~2-3% from
overfitting on the small dataset. **The current 100/128/6 sizing
is empirically validated by their sweep, not inherited blindly.**
The interesting follow-up is whether ShuttleSet's larger size shifts
the optimum: TemPose's Bad OL had ~15k samples; ShuttleSet has
~33k, so the curse-of-overfitting that pushed (200, 256) below
(100, 128) is weaker for us. A larger-(DL, DA) cell at our scale
might recover that ground.

**Genuinely worth sweeping?** This is Q5 in `arch_1_directions.md`
("Attention head geometry sweep") and is correctly flagged as
secondary priority. With TemPose's sweep results in hand, it's
even more clearly secondary: they already validated the current
point as the best on a similar dataset. The interesting question
is whether **moving up** the size curve (where TemPose hit
overfitting on their ~15k-sample dataset) now wins on ShuttleSet's
larger ~33k-sample size. That's a different motivation than
"tune the ratio".

**Sane sweep order.** Save it for after loss / aug / zeroing land.
If you do run it, two cells worth testing:
- (DL=200, DA=256, n_head=6): the TemPose cell that overfit on Bad
  OL, retested on ShuttleSet's larger N to see if it now wins.
- (DL=128, d_head=64, n_head=8, d_cat=512): the "standard ratio"
  cell, back to d_cat = d_model. Tells you whether the inverted
  bottleneck is specifically helping or just incidentally chosen.

### `depth_tem=2`, `depth_inter=1`

**What these are.** `depth_tem` is the number of stacked
TransformerEncoder layers in the **temporal** transformer that
processes each player-stream independently along the time axis
(`bst.BST.__init__`, `bst_common.build_bst_network` line 52-53,
`model/bst.py:177`). Each layer is a self-attention + FF block. With
depth_tem=2, the temporal transformer has two stacked encoder
layers; with depth_tem=3, three; etc. Same idea as encoder depth in
the original Transformer.

`depth_inter` is the same for the **interactional** transformer that
runs after the cross-transformer and processes the joined
two-player-plus-shuttle representation as one sequence
(`model/bst.py:186`). With depth_inter=1, the interactional
transformer is a single encoder layer; the heavy lifting is in the
temporal transformer and the cross-transformer. Both depths are
hardcoded in `bst_common.build_bst_network` defaults rather than
exposed via `Hyp`.

**What changing them does.** Adding depth gives the model more
self-attention rounds to integrate features at different temporal
scales (depth_tem) or interaction patterns (depth_inter). Cost is
linear in parameters and compute. Deeper transformers can also
overfit faster on small data, which is what TemPose's depth study
showed.

**TemPose's depth sweep (Table 4) is unusually thorough**, 10 cells
covering (LT, LN) from (1,1) to (8,8). Top cells in order:
(2,2)=90.7%, (2,1)=90.0%, (1,2)=89.9%, (1,1)=89.7%. (3,3) drops to
88.3%, (4,4) to 86.6%, with the deepest (8,8) at 85.2%. Clear
overfitting cliff past (3,3) on Bad OL. **BST inherits LT=2 but
LN=1, which is one notch below TemPose's optimum** ((2,1)=90.0% vs
(2,2)=90.7%). Likely a parameter-count savings move; might or
might not still apply at our scale.

**Worth sweeping?** A targeted one-cell test of `(2, 2)` vs
`(2, 1)` is high signal-to-cost: TemPose's data already says (2,2)
wins on similar data, and we'd be checking whether ShuttleSet
agrees. Skip the wider grid; (3,3) was already worse on Bad OL and
ShuttleSet is unlikely to flip that direction even at larger N.

**Sane sweep.** One cell: flip `depth_inter=1 → 2`, 3 serials.
Compare against the Phase-2 baseline. ~3-5 hr.

### `tcn_kernel_size=5`

**What it does.** The two TCN modules (`tcn_pose`, `tcn_shuttle`)
sit before the temporal transformer and apply 1D temporal
convolutions with this kernel size. They project the per-frame
input (joints+bones for pose, xy for shuttle) through two stacked
conv1d layers each producing d_model channels (`model/bst.py:161-162`).
The actual TCN is **dilated**, not plain stride-1: per
`tempose.py:139` (`dilation = i * 2 + 1`), layer 0 has dilation 1
and layer 1 has dilation 3. This matches TemPose Section 3.4
verbatim ("two 1D-convolutional layers ... with dilation 1 and 3,
respectively, with a kernel size of 5 and stride 1"). Receptive
field: layer 0 = 5 frames, layer 1 = 5 + (5-1)\*3 = **17 frames per
output position**. At 30 fps that's ~570 ms per token, comfortably
covering a full racket-swing build-up plus contact plus
follow-through.

**Was it swept?** Neither TemPose nor BST swept it; both treat
kernel=5 with the dilation pattern as fixed. General TCN
literature (keras-tcn docs, Bai et al "Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence
Modeling", `arxiv:1803.01271`) has kernel 2-8 as the typical band.
For sequence modelling tasks where a downstream transformer
handles long-range integration, the TCN's job is local-motif
extraction.

**Worth tuning down?** Worth testing, but the rationale changes
once you account for the dilation. The current 17-frame receptive
field is already wider than I (and probably you, before this
correction) had assumed; it's *not* a tight 9-frame local window.
The token each transformer position sees is summarising the entire
swing arc, not just the contact window.

So the trade-off is:

1. **Tune down** (kernel=3, dilations 1 and 3): receptive field
   drops to 3 + (3-1)\*3 = 9 frames (~300 ms). Pushes the swing
   build-up vs follow-through *separation* onto the transformer
   to learn rather than baking it into the TCN window. Plausibly
   helpful if you suspect the TCN is over-pooling within the swing
   arc; plausibly harmful if the transformer can't recover the
   structure from rawer per-frame inputs.
2. **Tune up** (kernel=7, dilations 1 and 3): RF = 7 + 6\*3 = 25
   frames (~830 ms). Probably too wide; starts to span multiple
   strokes if the seq_len=100 window contains adjacent shots.
3. **Drop dilation** (keep kernel=5, both dilations=1): RF =
   2k-1 = 9 frames (~300 ms). This isolates "is the dilation
   doing useful work" from the kernel question. Cheapest A/B.

The information-bottleneck instinct ("smaller kernel preserves more
per-frame signal") is correct but at this kernel size the
dominant control over receptive-field width is the dilation, not
the kernel. The most informative single ablation is probably
**option 3 (drop dilation 3 → 1)** rather than the kernel sweep,
because it directly tests whether the wider 17-frame window earns
its keep over a 9-frame one.

**Would it break anything?** Mechanically, no. The kernel size only
affects the conv weights and the receptive field per token; the
transformer downstream sees the same number of tokens regardless
(seq_len=100). Param count *increases* with kernel size linearly, so
going from k=5 to k=3 is a small param reduction. The thing to
check is that no downstream layer hardcodes a receptive-field
assumption. Looking at `model/bst.py` and the TCN definition in
`tempose.py`, nothing does. The risk is empirical, not structural:
maybe the model genuinely benefits from the wider pre-pooling and
the temporal transformer struggles with narrower per-frame
inputs.

**Sane sweep.** Two cells worth running: kernel=3 with current
dilation pattern (RF 9), kernel=5 with dilation 1/1 (RF 9, isolates
the dilation question). 2 cells × 3 serials = 6 serials. Skip
larger kernels; the current 17-frame RF is already at the upper
edge of useful for stroke-arc summarisation.

## Frame zeroing redesign

This is the section with the most direct questions on it. Code trace
and real numbers pulled from
`validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`
to ground the discussion.

### What the code is actually doing

`sticky_anchor.py:303-323` (per-frame loop in `_run_clip`):

- For each frame, `_pick_one_frame` returns either `None` (entire
  frame failed) or `(picks, ...)` where `picks` is a length-2 list
  with `-1` in any unpicked slot.
- For the unpicked slot(s), the per-slot pose stays at the
  initialised zero (lines 284-285) and the slot's EMA resets.
- The frame's `failed` flag is `frame_has_zero` (line 320), i.e.
  `True if either slot is unpicked, else False`. **Single-player
  frames are flagged as failed, even though half the pose data is
  real.**

Then at collation (`prepare_train_on_shuttleset.py:864-867`):

- `if np.any(failed): shuttle[failed, :] = 0`
- So shuttle gets zeroed on every frame where `failed[t]` is True,
  including single-player frames where the picked player's pose
  is real.

The asymmetry: the **picked player's pose stays non-zero** on a
single-player frame (because `pos[f, s] = cbp` and `joints[f, s] =
normalize_joints(...)` actually write the real values for the
picked slot), but **shuttle is wiped** as if the whole frame had
collapsed. Which is what your instinct flagged.

### TrackNet visibility=0 frames

Looking at `prepare_train_on_shuttleset.py:489`:
`df = df.set_index("Frame").drop(columns="Visibility")`. The
TrackNet CSV contains a `Visibility` column that is 0 on frames
where TrackNet did not detect the shuttle, with X=Y=0 in those
rows. The pipeline drops the visibility column and uses the (X, Y)
directly, normalising via `arr[:, 0] / v_width, arr[:, 1] /
v_height`. So a "shuttle missing" frame becomes literal (0, 0) in
the input array, **indistinguishable from a real shuttle position
in the top-left corner**. There is no shuttle-missing flag
preserved anywhere downstream.

### Real-data numbers (Phase-2, 32,203-clip extract)

From the merged_25 / split_bst_baseline analysis:

| Cohort | Frames | % of 1,719,627 |
| --- | --- | --- |
| Both pose and shuttle OK | 1,596,251 | 92.83% |
| MMPose only fail | 14,411 | 0.84% |
| Shuttle only fail | 107,520 | 6.25% |
| Both fail | 1,445 | 0.08% |

So shuttle "fails" (visibility=0 from TrackNet) on **~7x more frames
than mmpose**. Both-fail is rare (0.08%). The current zeroing policy
nukes the shuttle on the 0.84% MMPose-only-fail frames *and*
implicitly leaves the 6.25% shuttle-only-fail frames at literal
(0,0) without a flag.

Per-stroke shuttle miss rates (from the same analysis, for
context): long_service 10.51%, clear 5.90%, lob 2.61%, smash 1.81%
near the hit. Heaviest on slow service strokes where the shuttle
crosses near-stationary in space and TrackNetV3 loses it; lightest
on fast classes (drive, drop, return_net, push, cross_court) where
the shuttle stays inside the visible court.

### Why is the shuttle missing rate so high?

The 6.34% rate is striking given two facts that should be pushing
it down: (a) the sticky_anchor heuristic is now well-tuned and only
zeros pose on frames where the homography projection genuinely
fails or fewer than two players are in court (0.93% mmpose
post-heuristic), and (b) we're using TrackNetV3 *with* the inpaint
rectification module (the qaz812345 fork, the "accidental"
inpaint that beat the BST authors' attention-only build, per
`run_20260417_191851/best_model_id.txt`). So 6.34% is the rate
*after* per-frame interpolation across short missing-detection
gaps. The raw rate without inpaint would be higher.

Decomposing by hit-zone position (from the same validation analysis):

| Zone | Frames | Shuttle missing | Rate |
|---|---|---|---|
| Within ±10 frames of any hit ("near hit") | 672,637 | 10,584 | 1.57% |
| Other frames ("away from hit") | 1,046,918 | 98,381 | 9.40% |

Most of the shuttle-missing problem is **between-hit frames, not
the hit moment**. The away-from-hit rate is 6x the near-hit rate.
That rules out contact-moment detection failure as the dominant
mechanism; it's a flight-phase tracking failure.

Per-stroke pattern, sorted by miss rate near hit:

| Stroke | Miss rate near hit | Trajectory profile |
|---|---|---|
| long_service | 10.51% | Held → high arc to back court |
| clear | 5.90% | High deep shot, peak above court |
| rush | 2.98% | Fast net attack, possible blur |
| lob | 2.61% | High arc |
| smash | 1.81% | Setup is incoming high lob |
| short_service | 1.14% | Held → low across net |
| drive | 0.85% | Flat, mid-height |
| return_net | 0.54% | Low at net |
| drop | 0.52% | Slow descending |
| net_shot | 0.26% | At-net, low |
| push | 0.23% | Flat near net |
| cross_court_net_shot | 0.11% | At-net, low diagonal |

Strong gradient: stroke types that send the shuttle high or far
back have 5-50x the miss rate of net-bound shots. The cleanest
controlled comparison is **long_service vs short_service** (both
involve a held shuttle into a service hit; only the post-contact
trajectory differs): 10.51% vs 1.14%, an order of magnitude.
Held-shuttle detection isn't the differentiator; trajectory altitude
is.

**Likely causes, ranked.** All of these are inferences from the
per-stroke aggregate stats, not from clip-by-clip inspection.
Verification scripts described in the runbook below.

1. **Shuttle exits the broadcast frame on high arcs** (most likely
   primary cause). ShuttleSet videos come from BWF TV broadcasts,
   which frame the visible court but typically cut off well below
   the lighting rig and ceiling. A long_service or clear arc at
   peak altitude is genuinely above the camera's vertical FOV.
   Once the shuttle exits the frame, no detector can recover it,
   including TrackNetV3 with inpaint. The inpaint can interpolate
   *between* valid endpoints, but it can't synthesise frames where
   the shuttle is physically not in the image. Fits the per-stroke
   pattern: long_service / clear / lob / rush at the top, net-bound
   shots at the bottom.
2. **Inpaint window limits.** The rectification module interpolates
   across short missing-detection gaps but has a maximum gap size.
   A long off-screen excursion (15-30 frames of "shuttle not in
   any pixel") exceeds the inpaint window, so those frames stay at
   (0, 0). Consistent with the 9.40% away-from-hit rate: long
   flight phases are exactly when sustained off-screen stretches
   happen.
3. **Motion blur at peak velocity** (minor at most). If blur were
   dominant, fast contact strokes (smash, drive, push) would have
   the highest miss rates. They're at the bottom of the table.
   Blur probably contributes a small constant background, not the
   per-stroke gradient.
4. **Held-shuttle ambiguity** (minor on long_service). A held
   shuttle against a player's body or busy background is harder
   to detect, but the long_service / short_service gap eats this
   as the primary cause: both have a held-shuttle pre-serve frame
   and only one is at 10.51%.
5. **Net-line occlusion** (negligible). If net occlusion mattered,
   net_shot / cross_court_net_shot / push would be elevated.
   They're at the bottom (0.11-0.26%).

**Why this matters for the mask design.** The missing-shuttle
signal is highly *structured*, more so than the 6.34% raw rate
suggests:

- **Spatially structured**: gaps cluster at high altitudes /
  off-court regions, which correlates with stroke type.
- **Temporally structured**: long sustained absences during high
  arcs, not random per-frame dropouts. Gaps are 5-30-frame
  contiguous blocks, not isolated noise.
- **Class-conditional**: a `shuttle_missing` mask channel is partly
  a stroke-class hint (high miss density → likely service / clear
  / lob).

This was originally framed as reinforcing the variant-2 mask
recommendation. **The verified 0c result reinterprets it**: the
high-miss classes are the high-F1 classes, so a `shuttle_missing`
mask channel mostly conditions on stroke-class identity (which the
model already gets right from pose alone). The mask is real signal,
just not pointed at the bottleneck classes. See "Verified findings
(2026-04-30)" subsection below for the diagnostic update.

### Verified findings (2026-04-30, scripts 0a / 0b / 0c)

The three pre-flight scripts ran on the full 32,203-stem extract
(unknowns dropped via `clips_master.csv`). The off-screen-high
hypothesis is **confirmed**, but the diagnostic implications differ
substantially from the original framing.

**0a (`shuttle_gap_y_distribution.py`): boundary y-coords cluster
hard at the top of the frame.**

Combined pre-gap last-valid + post-gap first-valid y-coords (n=13,902,
y=0 is top, y=1 is bottom):

- **61.6% in [0.0, 0.1)** (top 10% of frame)
- 5.6% in [0.1, 0.2)
- 24.6% combined in [0.3, 0.5) (mid-court secondary mode)
- <2% in [0.6, 1.0]
- median y = 0.017, mean = 0.155

Pre-gap last-valid: 53.0% in top decile (mean 0.191, median 0.044).
Post-gap first-valid: **72.3% in top decile** (mean 0.110, median
0.009). Distribution is bimodal: a dominant off-screen-top mode and a
smaller mid-court mode (probably motion blur / background contrast /
net-line occlusion).

The pre-gap mid-court secondary mode sits at y ~ 0.4-0.5 (player
contact level); post-gap mid-court mode sits at y ~ 0.3-0.4
(slightly higher / more back-court). Reads as the shuttle moving
upward from contact, going off-screen, and re-entering at a
slightly higher / further-back position. Consistent with ballistic
arcs.

**0b (`shuttle_gap_length_distribution.py`): inpaint window is not
exceeded; the problem is mid-length off-screen arcs.**

32,203 clips scanned, 24 clips with no shuttle detections at all
(0.07%, negligible), 7,299 clips with at least one gap, 24,880
clips with no gaps at all (77.3%, very robust).

8,466 total gaps across 108,673 missing frames. Gap-length stats:
median 8 frames, p99 45 frames. Distribution by class:

| Length class | Gaps | % of gaps | Frames | % of frames |
|---|---|---|---|---|
| 1-2 (single-event blip) | 2,477 | 29.26% | 3,320 | 3.06% |
| 3-5 (motion-blur band) | 1,152 | 13.61% | 4,385 | 4.04% |
| 6-10 (brief occlusion) | 998 | 11.79% | 7,738 | 7.12% |
| **11-30 (off-screen-arc band)** | **2,884** | **34.07%** | **57,984** | **53.36%** |
| **31-60 (sustained absence)** | **954** | **11.27%** | **35,181** | **32.37%** |
| 61+ (inpaint window exceeded) | 1 | 0.01% | 65 | 0.06% |

The 11-60 frame band accounts for **85% of all missing-shuttle
frames**. Only 1 gap in the entire 32k-clip set exceeds 60 frames.
Implications:

- Inpaint is succeeding at the short end (1-10 frames = 55% of gaps,
  14% of frames). Doing useful work where it can.
- Inpaint **isn't being exceeded** at the long end. Essentially zero
  gaps past the typical inpaint window size.
- The 11-60 frame range is "shuttle is genuinely not in any pixel of
  these frames", which no detection-based system can recover. Arc
  duration matches typical badminton flight times for high lobs /
  clears.

**0c (`perclass_shuttle_miss_vs_f1.py`): shuttle-miss rate is
*positively* correlated with per-class F1, opposite of the
hypothesis.**

Run 3 nosides manifest joined against the nosides analysis txt,
14 classes joined of 14, F1 metric, no-collapse-sides:

- Pearson r: **+0.516**
- Spearman r: **+0.415**

Sorted (high miss → low miss):

| Class | Miss% | Median F1 |
|---|---|---|
| long_service | 10.51 | 0.987 |
| clear | 5.90 | 0.954 |
| rush | 2.98 | 0.746 |
| lob | 2.61 | 0.780 |
| smash | 2.24 | 0.662 |
| **wrist_smash** | **1.17** | **0.360** |
| short_service | 1.14 | 0.986 |
| drive | 0.84 | 0.604 |
| drop | 0.73 | 0.668 |
| return_net | 0.54 | 0.813 |
| net_shot | 0.26 | 0.906 |
| push | 0.23 | 0.596 |
| passive_drop | 0.17 | 0.644 |
| cross_court_net_shot | 0.11 | 0.683 |

The high-shuttle-miss classes (long_service, clear, lob) are exactly
the **pose-distinctive** strokes the model classifies confidently
from skeleton alone. The bottleneck classes (wrist_smash, push,
drive, cross_court_net_shot) sit at sub-1% miss rates because their
shuttle trajectories are short and stay in frame.

**The shuttle stream IS available where it's most needed**; the
model just isn't using it well enough on the bottleneck classes.

### Diagnostic conclusion from 0a / 0b / 0c

The three results combine to firm up the diagnosis:

1. **Off-screen-high is real and structured** (0a): the gap pattern
   is a sensor-level / camera-framing limitation, not a model
   limitation, not a heuristic limitation, not a TrackNetV3
   limitation.
2. **Inpaint is not the bottleneck** (0b): the rectification module
   covers what it can; the missing data is genuinely missing pixels.
3. **Shuttle data is available on the bottleneck classes** (0c): the
   model has the inputs it needs to disambiguate wrist_smash / drive
   / push / cross_court_net_shot but doesn't use them well.

Combined with run 2's wrist_smash gate failure pattern (cleaner
data hurt the rare-support class while helping the head of the
distribution), this triangulates onto a **classifier-side problem**,
specifically how the loss budget is being spent across classes:

- Mean macro / accuracy / top-2 hold or improve with cleaner data
  (head of the distribution wins).
- Min F1 collapses on the small-support classes (tail loses).
- This is the textbook signature of label smoothing fighting
  cleaner data on rare classes: smoothing taxes confidence
  uniformly, but the gradient signal that overcomes the smoothing
  is proportional to support count, so 600-sample classes lose
  more than 2,400-sample classes to the same smoothing constant.

The mask-channel arm becomes a smaller lever than initially
hypothesised. The variant-2 design still works, but it would help
robustness on services / clears / lobs (already at 0.95-0.99 F1)
rather than rescuing the bottleneck classes. Demote.

The label-smoothing arm becomes the highest-priority loss-side
experiment, not the focal-loss arm. **Promote LS sweep ahead of
focal**, since LS is the cheaper test and the failure-mode profile
matches its predicted effect more directly than focal's.

A future direction worth flagging: **trajectory extrapolation** for
the 11-60 frame off-screen-arc gaps. Since the gaps cluster at
predictable phases of ballistic motion (between known endpoints,
typically going up and coming down), a parabolic-fit extrapolation
between pre-gap and post-gap valid coords would fill these frames
with physically plausible positions rather than (0, 0). Not a
near-term priority because the bottleneck isn't on the high-arc
classes anyway, but a real lever once the loss-side knobs settle.

### Three zeroing options, evaluated

**Option A: zero only when both fail (0.08% of frames).** Effect:
- On MMPose-only fail frames (0.84%), keep shuttle non-zero. The
  shuttle stream provides continuous trajectory information across
  the 14,411 frames per analysis where TrackNet had a perfectly
  good detection but the current policy throws it away.
- On shuttle-only fail frames (6.25%), no change because the
  current policy doesn't intervene there anyway. Those frames
  already have shuttle=(0,0) in the data with no flag.
- On both-fail frames (0.08%, 1,445 frames), zero both. Same as
  current behaviour.

This is the easiest decoupling and the one with the most defensible
read on what the model can use: it's giving back the 14,411 frames
of valid shuttle data that the current policy was wastefully
zeroing. Per-frame this is small (~0.84% of the total), but
clip-clustering matters: a clip with a 50% MMPose fail rate
in the hit zone (and there are 65 such clips, including 17 with
100% hit-zone failure) would benefit specifically because shuttle
trajectory still anchors the swing event when the player pose has
collapsed.

**Cost.** Code-side: the `failed` flag in sticky_anchor would need
to flag only the both-zero (full-frame failure) case. Easier
alternative: leave sticky_anchor as-is (it correctly flags
"either slot zero" for downstream telemetry) and move the
"zero shuttle on failed" line out of `prepare_train_on_shuttleset.py`
so shuttle is just left as TrackNet emitted it. Re-collation
required. ~10 minutes per taxonomy combo.

**Risk.** Asymmetric coupling has a logic to it that we'd be
giving up: pose tells you what the player is doing relative to a
court frame, shuttle tells you where the bird is in that frame; if
pose collapses, the shuttle's coord still has meaning *within the
court frame*, which is what we're keeping. So the risk is small.

**Option B: zero only when no keypoints at all (full-frame
failure).** Effect:
- On single-slot-picked frames, **keep the picked player's pose
  non-zero AND keep shuttle non-zero**. This reverses both
  asymmetries.
- The "failed" flag becomes a stricter "both slots unpicked, or
  pre-rally-presence rejection" condition.
- Both-fail behaviour as Option A.

This goes further than A and is the one Ariel's intuition was
already pointing at: "a single player is validly detected ... we're
still zeroing those single players?". The answer is yes, the
shuttle is being zeroed on those frames; the picked player's pose
is *not* being zeroed (it's preserved per the trace above), so the
asymmetry has two sides:

1. Picked player's pose: preserved (current behaviour is correct).
2. Unpicked player's pose: zero (correct, no detection).
3. Shuttle: zeroed (current behaviour, would change under B).

So under Option B vs current: items 1 and 2 are unchanged; item 3
becomes "leave shuttle as TrackNet emitted it". Same code change
as Option A, same recollation cost.

**Why distinguish A vs B?** Same end state on the data side,
different framing of the heuristic flag. Going with B's framing
matters if you also want to **expose the failure flag downstream as
a mask channel** (next subsection): then "single slot picked" is
information the model could use rather than something that just
trips a flag.

**Option C: zero when shuttle is zero (propagate shuttle-fail to
pose).** Effect:
- On the 6.25% shuttle-only-fail frames, zero the pose too. That's
  ~107,520 extra zeroed pose frames, **a 7x increase in
  pose-zeroed frame count vs current**.
- The justification would be "without shuttle context, pose alone
  is misleading because BST is fundamentally a shuttle-relative
  classifier".
- But: this is a lot of training data to throw away. Pose-only
  signal is informative for many strokes (e.g. swing pattern of a
  smash is recognisable without seeing the bird) and the
  `BST_0` ablation (pose-only, no shuttle) in the BST paper still
  performs above-trivially. So zeroing pose on every shuttle-miss
  frame is throwing away training signal in exchange for a cleaner
  "always-paired" invariant.

**Verdict.** Almost certainly the wrong direction unless we also
strip the shuttle stream from the model. Option C combined with
explicit shuttle-failure masking might make sense as part of a
"shuttle-conditional inference" experiment, but as a default
zeroing policy it's destructive.

### Mask-channel speculation

Your "global some_input_is_masked channel" idea is the right
question. The design space splits along two axes: granularity (how
many separate mask channels) and signal density (how often each
channel fires). Both axes matter for learnability.

Variants worth sketching:

1. **Three-channel per-stream**: `pose_top_missing[t]`,
   `pose_bottom_missing[t]`, `shuttle_missing[t]`. Maximum
   information; the model can condition behaviour on which stream
   is broken. Standard "pad mask" pattern in NLP transformers.
2. **Two-channel pose-merged**: `pose_missing_either_slot[t]`,
   `shuttle_missing[t]`. Loses "which player is broken" info;
   keeps "is pose broken at all" + "is shuttle broken".
3. **Single global mask**: one channel that fires on any stream
   failure. Coarsest. Cheapest to implement.
4. **No mask, fix the encoding**: replace shuttle (0, 0) with
   interpolation between last and next valid detections, or a
   NaN-marker the model learns to skip. Different design
   (interpolation over masking); the TCN's conv1d can't see NaNs
   directly, so this needs care.

**The signal-density question, with the actual numbers.** The
1,719,627-frame validation analysis breaks down as:

| Channel | Positive rate | Frames | Learnability read |
|---|---|---|---|
| `shuttle_missing` | 6.34% | ~109k | Plenty |
| `pose_missing` (combined Top + Bottom) | 0.93% | ~16k | Borderline; clip-clustering rescues it |
| `pose_top_missing` only | ~0.5% | ~8k | Too thin per-frame |
| `pose_bottom_missing` only | ~0.5% | ~8k | Too thin per-frame |
| Global "any masked" (OR) | ~7% | ~123k | Plenty but coarse |

Splitting Top vs Bottom pose halves each channel's positive rate
to a sub-0.5% range. At that density, a one-bit input has a hard
time teaching a model anything specific per-frame because the
gradient signal is dominated by the 99.5% of frames where the
flag is the same value. The transformer's integration across
positions does help (clip-level signal is much stronger), but the
per-frame baseline is genuinely thin for splitting Top from Bottom.

**Recommended design: variant 2 (two channels).** Collapse pose
Top and Bottom into one `pose_missing_either_slot` flag. You give
up the "which side broke" information in exchange for keeping each
channel's positive rate above the per-frame learnability floor:

- `shuttle_missing`: 6.34% positive. Strong, well-clustered in
  slow-service classes (long_service 10.51%, clear 5.90%, lob 2.61%
  near hit). Plenty learnable.
- `pose_missing`: 0.93% positive. Borderline per-frame; rescued by
  clip-clustering (785 clips have any zeroed frames, 65 with >50%
  zeroed, 17 with 100% hit-zone-zeroed) and stroke-correlation
  (long_service 3.91%, short_service 3.46% vs return_net 0.08%
  failure rate). Becomes a class-conditional signal in addition to
  a per-frame one.

Strictly more informative than the current "no mask, just zero
coords" encoding. Minimal plumbing cost: two extra input channels
through the existing TCN + transformer stack.

Your "make it global ... maybe it would just become a signal to
interpolate between global last and next known positions across
all inputs" intuition is well-targeted: a single global flag
combined with the model's temporal context is exactly the
invariance the model would learn (treat masked frames as
interpolation problems). That's the **desired behaviour**. The
risk you pointed at, "becoming an interpolation signal across
*all* inputs", is the failure mode where the model forgets that
mask=1 only means *one* stream is broken and over-discounts the
others. Variant 2 (two channels) prevents this by separating the
two failure modes that fail at very different rates and have very
different downstream meanings (shuttle missing usually means a
visible-shuttle problem; pose missing usually means a player-
detection problem in service-heavy clips). Variant 3 (global) is
sloppier and would risk that failure mode.

**Practical ordering for the masking experiment.** Step 1: change
the zeroing policy to Option A (give back shuttle on single-slot-
picked frames). Train and measure. If it lifts metrics, the model
is using the recovered shuttle signal even without an explicit
mask. Step 2: add the two-channel mask (variant 2). Train and
measure. If it lifts further, the model wanted the explicit
signal. Step 3: only if lifts plateau, investigate whether the
current shuttle (0, 0) frames (the 6.25% TrackNet visibility=0
set) are causing visible per-stroke harm, given the model now has
a shuttle-missing mask. Could lead to the fourth variant (real
interpolation).

### Per-clip vs per-frame signal-density rescue

Even with variant 2, the `pose_missing` channel sits at a thin
per-frame rate that warrants a sanity check on whether the
transformer can learn from it at all:

- The signal is **clip-correlated**. ~785 clips have *any* MMPose
  zeroing in their hit zone, 17 are 100%-zeroed, 65 are >50%-zeroed.
  At the clip level, a pose-missing flag is much more informative
  than its 0.93% per-frame rate suggests, because the model sees
  long stretches of correlated mask-on frames for these difficult
  clips. The transformer integrates across positions, so even a
  0.93% per-frame rate becomes a strong per-clip signal in those
  785 clips.
- The signal is **stroke-correlated**. Service strokes are 5-10x
  more affected than smash / drop / clear by both pose and shuttle
  failures. A mask channel becomes a cheap conditioning input that
  tells the model "this is one of those service-side clips with a
  patchy hit zone, lean harder on the parts you do have".

So the worry is reasonable per-frame but the signal becomes
learnable per-clip and per-class. Worth testing.

## Hyperparameter interaction matrix

Most hparams have a primary one-knob effect plus a smaller
interaction with one or two others. The interaction effects worth
calling out (where joint sweeping is more informative than
sequential):

| Pair | Why they interact | Joint sweep recommended? |
| --- | --- | --- |
| `lr` x `weight_decay` | Both modify update magnitude. Higher LR with higher WD is roughly equivalent to lower LR with lower WD on simple loss surfaces. | Already retuned LR (Q4); WD now correctly studied in isolation. **No joint sweep needed.** |
| `weight_decay` x `drop_p` | Shared regularisation budget. Tuning either alone tends to overshoot. | **Yes.** Cheap 4-cell follow-up after the marginal sweeps locate good regions. |
| `drop_p` x augmentation | Same shared regularisation budget at different stack layers. | **Yes**, but probably enough to do them sequentially: first augmentation tuning to locate the right magnitude, then drop_p around that anchor. |
| `label_smoothing` x focal loss | Both regularise classifier confidence. Stack weirdly per the spec in `arch_1_directions.md`. | The focal-loss spec already disables LS in the focal arm. **No joint sweep.** Run focal vs CE separately. |
| `tcn_kernel_size` x `depth_tem` | Both affect effective temporal receptive field and how much pre-pooling vs attention-pooling the model does. | **Marginal yes, low priority.** Could be a single 2-cell follow-up (k=3 with depth_tem=3 vs k=5 with depth_tem=2) once both have been tuned in isolation. |
| Frame zeroing x mask channel | Mask channel meaningless without zeroing-policy decision; zeroing-policy results harder to read without mask channel. | **Sequential**: first pick a zeroing policy, then add mask channel as a follow-up arm. |
| `ablation_id` x `taxonomy` | Independent. Don't sweep jointly. | **No.** |

The general rule from the regularisation literature (multiple
secondary sources confirm; the most explicit is the structured
dropout paper Fan et al 2019): when you have multiple regularisers
operating on the same effective capacity (dropout, weight decay,
label smoothing, augmentation, stochastic depth), they're not
additive; tuning them sequentially without rechecking earlier
choices can leave the model in a different regularisation regime
than intended. Practical recipe: do marginal sweeps to locate good
regions, then do one small joint cell with the two strongest knobs
to confirm they don't fight each other.

## Run-budget arithmetic

Each 5-serial training run on the A100 takes roughly 5-10 hours
(combo C completed in ~4 hr per the run_20260429_202144 timestamps:
20:21 → 22:27, ~2.1 hr for 5 serials at 80 ep, but that's compressed
because early stop is biting). 3-serial runs on the A100 are
~1.5-2.5 hr. V100 is roughly comparable to slightly slower (CPU-bound
aspects of the dataset loader cap GPU utilisation either way).

Tier-1 sweeps (cheap, high expected lever, do first):

| Sweep | Cells | Serials | A100 hours |
| --- | --- | --- | --- |
| `weight_decay` (4 values) | 4 | 12 | ~12-15 |
| `drop_p` (3 values) | 3 | 9 | ~9-12 |
| `weight_decay` x `drop_p` joint (4-cell) | 4 | 12 | ~12-15 |
| Augmentation tighten + off | 3 | 9 | ~9-12 |
| **Tier-1 subtotal** | | **42** | **~42-54 hr** |

Tier-2 (architecture-side, do after tier-1):

| Sweep | Cells | Serials | A100 hours |
| --- | --- | --- | --- |
| Flip `depth_inter=1 → 2` (TemPose's optimum) | 1 | 3 | ~3-5 |
| TCN: kernel=3 vs dilation off | 2 | 6 | ~6-9 |
| Add flipping augmentation (taxonomy permitting) | 1 | 3 | ~3-5 |
| **Tier-2 subtotal** | | **12** | **~12-19 hr** |

Tier-3 (zeroing redesign, do alongside or after data-augmentation arm):

| Sweep | Cells | Serials | A100 hours |
| --- | --- | --- | --- |
| Zeroing policy A (decoupled shuttle) vs current | 2 | 6 | ~6-9 + ~30 min recollation each |
| Per-stream mask channel | 1 (vs Option A baseline) | 3 | ~3-5 + dataset code change |
| **Tier-3 subtotal** | | **9** | **~12-15 hr** |

Architecture geometry (Q5 attention head sweep, save for last):

| Sweep | Cells | Serials | A100 hours |
| --- | --- | --- | --- |
| (DL=200, DA=256): TemPose-overfit cell on ShuttleSet | 1 | 3 | ~5-7 (larger model) |
| (DL=128, DA=64, n_head=8): standard ratio test | 1 | 3 | ~3-5 |
| **Architecture subtotal** | | **6** | **~8-12 hr** |

**Grand total** if everything ran: ~74-100 A100-hours, down from
the earlier estimate now that TemPose's verified ablations let us
skip the wider depth and head-geometry grids. With the
already-flagged label smoothing + focal + augmentation arm
contributing maybe another 30-40 hours on top, the realistic
budget for this whole programme is **105-140 A100-hours**, give or
take. That's roughly two weeks of overnight runs at one combo per
night, or one week at two combos per night.

In practice, marginal results from tier-1 will redirect the rest of
the programme: a clear winner on `drop_p`-down would shorten the
augmentation arm (since they share regularisation budget); a flat
WD response would suggest the model isn't capacity-bound and would
let us skip the zeroing-policy arm. Plan for the tier-1 budget,
plan the rest as informed-by-results.

## Suggested runbook order

0. **Pre-flight: clip-length distribution check** (no training).
   Inspect the post-clipping seq_len distribution from the active
   collated dirs to confirm the `between_2_hits_with_max_limits`
   cap (currently seq_len=100) isn't truncating clips the
   variable-length scheme would have run longer. The BST paper
   reports a 5.7-percentage-point min-F1 lift just from variable-
   length clipping; if our cap is biting, we may be silently
   leaving that on the table. ~30 min of analysis. Skip the rest
   if this pre-flight finds the cap isn't biting; pursue a
   relaxed-cap experiment if it is.
0a. **Pre-flight: shuttle-gap boundary y-coord distribution**
    (no training). For each shuttle-missing run in the per-clip
    `_shuttle.npy` files, record the y-coordinate of the last
    valid detection before the gap and the first valid detection
    after. Plot the distribution. **Hypothesis**: if the gap
    boundaries cluster near the top of the frame (small y in
    image coords, or high in court coords post-normalisation),
    that's direct evidence the shuttle exits the camera frame on
    high arcs. ~1 hr to write, ~5 min to run on the 32k-clip set.
    Output: histogram + percentiles to
    `validation_scripts/zeroed_frames_analysis_outputs/`.
0b. **Pre-flight: shuttle-gap length distribution** (no training).
    For each contiguous run of `visibility=0` frames in the
    per-clip shuttle data, record the gap length. Plot the
    histogram. **Hypothesis**: gaps clustered around 10-30 frames
    suggest off-screen excursions of typical badminton arc
    duration; gaps clustered at 3-5 frames suggest motion-blur or
    brief occlusion; a heavy tail beyond ~30 frames suggests the
    inpaint window is being exceeded for sustained off-screen
    cases. The shape determines whether trajectory extrapolation
    (rather than just masking) is worth pursuing as a future
    direction. ~1 hr to write.
0c. **Pre-flight: per-class shuttle-miss vs per-class F1
    correlation** (no training, uses existing run artefacts).
    Done 2026-04-30 against combo A nosides manifest. **Result
    (verified)**: Pearson r = +0.516, Spearman r = +0.415.
    Positive correlation, opposite of the hypothesised direction.
    The high-miss classes are pose-distinctive serves / clears /
    lobs (already at 0.95-0.99 F1); the bottleneck classes
    (wrist_smash, drive, push, cross_court_net_shot) sit at <1%
    miss rate. Diagnosis: shuttle data is available where it's
    most needed; bottleneck is on the classifier side. See "Why is
    the shuttle missing rate so high?" subsection above.

1. **Label smoothing sweep on combo A nosides** — **DONE
   (2026-05-01)**. Three runs landed; LS=0.15 won (+4.2 pp on
   mean wrist_smash vs LS=0.1 baseline; head metrics flat; range
   tightens 0.159 → 0.066). LS=0.0 disproved the rare-class-tax
   hypothesis. LS=0.05 skipped (two bracketing data points
   enough); LS=0.2 deferred. Full numbers in `arch_1_directions.md`.
1a. **LS sweep + horizontal-flip augmentation on combo B**
    (originally gated on step 1 showing lift). Held: combo A
    LS sweep wrapped without combo B re-test being needed for
    the immediate next step. Re-evaluate when augmentation work
    starts as a standalone arm.
2. **Focal loss / CDB-F1** — **DONE (2026-05-02)**. Class-F1-
   driven adaptive focal fully explored (5 runs); first run
   tau=1 / gamma=1 was the floor-lift sweet spot at +8.7 pp
   wrist_smash on LS=0.1 baseline. All four follow-ups (gamma=0,
   tau=0.5, pair-cap, gamma=2) traded ws back for smash without
   macro moving. No CDB run breaks the val/test plateau at
   0.74-0.75 macro. Loss-side ceiling firmly mapped. Full
   numbers in `arch_1_directions.md`.
2a. **Capacity-bump confirmation runs** — **NEW PRIORITY
    (2026-05-02)**, ahead of the items below. Capacity-bottleneck
    research at `scratch/architecture_notes/model_capacity_bottleneck_question.md`
    argues the plateau is data-bound and signal-bound. Two cheap
    confirmatory runs while compute is around: `mlp_head` hidden
    400 → 768 (surgical, classifier-side only), then `d_model`
    100 → 128 with `d_head` trim 128 → 32 (residual-stream widen
    paired with Voita-style head trim). Both vs `run_20260501_164658`.
    Expected gain 0-2 pp on test macro.
3. **Weight decay sweep [0.0, 0.05, 0.1]**: single-arm, cheapest
   architectural-side win after the capacity-bump runs settle.
4. **Augmentation magnitude sweep** (tighten / off): clarifies
   how aggressive the existing shift aug actually is. Skip if
   step 1a's flip-aug already settled the augmentation regime.
5. **Flip `depth_inter=1 → 2`**: one-cell test of TemPose's depth
   optimum on ShuttleSet. Cheap and high signal-to-cost given
   TemPose's verified ablation table.
6. **Dropout downtune [0.15, 0.2]**: once augmentation is settled,
   sweep dropout against it.
7. **Joint WD x dropout 4-cell** if either sweep returns a clear
   marginal winner; skip if both sweeps were flat.
8. **BST-AP vs BST-CG-AP under focal loss** (inferred speculation
   from Table 1 BST paper, see "BST paper inferences"). After
   focal lands, retest BST-AP-only against BST-CG-AP. Single cell,
   3 serials.
9. **Frame-zeroing Option A** (decouple shuttle from pose-fail):
   re-collate, run 3 serials, compare. Demoted relative to the
   earlier ordering: the 0c result shows shuttle is reliably
   present on the bottleneck classes, so this arm helps the head
   of the F1 distribution rather than rescuing the tail.
10. **Two-channel mask** (`shuttle_missing` + `pose_missing_either_slot`):
    demoted to "only if Option A lifted by a meaningful amount".
    The 0a / 0b results show the mask signal is interpretable as
    "shuttle is in the high-altitude phase of an arc" rather than
    "data is unreliable", but the bottleneck classes don't need
    this info, so the lever size is bounded.
11. **X-flip-only augmentation arm** (orthogonal isolation): if
    step 1a's joint sweep can't tease apart whether the lift came
    from LS or from flip-aug, run a single cell of "LS=0.1 +
    flip-aug" against the no-aug baseline. Resolves the
    attribution.
12. **TCN: kernel=3 vs dilation-off**: 2-cell sweep once the
    training-side knobs are settled.
13. **(DL, DA) larger cell** (DL=200, DA=256): test whether
    TemPose's overfit-on-Bad-OL cell now wins on ShuttleSet's
    larger N.
14. **Trajectory extrapolation for off-screen-arc gaps** (long-term).
    Replace shuttle (0, 0) on the 11-60 frame off-screen-arc
    gaps (85% of missing-shuttle frames per 0b) with a parabolic
    fit between pre-gap and post-gap valid coords. Physically
    plausible positions instead of literal-corner zeros. Not
    near-term because the bottleneck isn't on the high-arc
    classes; flagged for after the loss-side and architectural
    knobs settle. Probably interacts with the masking arm (model
    needs to know whether each frame is real-detection vs
    extrapolated).

The first 3 lines are the highest expected ratio of lever per
A100-hour given the verified diagnosis. Line 1 has the cleanest
predicted-effect-to-cost ratio. Lines 5-7 are joint-sweep gates
depending on what 3 and 4 return. Lines 9-10 are the zeroing
redesign, demoted because the 0c result showed shuttle data is
already available on the bottleneck classes. Lines 11-13 are
architecture / data-side cleanup, save until the rest has been
worked. Line 14 is a longer-term direction flagged by the 0b
result (heavy population of unrecoverable mid-length off-screen
gaps).

## Sources

- BST paper, full read including supplementary (training
  hyperparameters at Section D, ablation tables 1-5 main + B-E
  supplementary, training speed at Table A, loss curves at Fig A,
  per-class data distribution at Tables F-I): `arxiv:2502.21085`
  v3, local copy at
  `~/Documents/COSC594/BST_ Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports.pdf`,
  publisher copy at `https://arxiv.org/html/2502.21085`
- TemPose paper (verified directly from local PDF, including
  Table 1 training hparams, Table 2 (DL, DA) joint sweep, Table 3
  J+B vs J, Table 4 depth sweep, Section 3.4 TCN spec): Ibh et al
  2023, CVPRW. Local copy at
  `~/Documents/COSC594/Ibh_TemPose_A_New_Skeleton-Based_Transformer_Model_Designed_for_Fine-Grained_Motion_CVPRW_2023_paper.pdf`;
  publisher copy at
  `https://openaccess.thecvf.com/content/CVPR2023W/CVSports/papers/Ibh_TemPose_A_New_Skeleton-Based_Transformer_Model_Designed_for_Fine-Grained_Motion_CVPRW_2023_paper.pdf`
- Andriushchenko et al, "Why Do We Need Weight Decay in Modern Deep
  Learning?", `arxiv:2310.04415`
- Fan et al, "Reducing Transformer Depth on Demand with Structured
  Dropout" (LayerDrop, dropout rate baselines):
  `https://openreview.net/pdf?id=SylO2yStDr`
- d_head / d_model / n_head standard ratio:
  `https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html`
- TCN kernel size ranges: keras-tcn docs
  `https://github.com/philipperemy/keras-tcn`; Bai et al 2018
  `arxiv:1803.01271`
- Phase-2 zeroing analysis (real numbers used in the frame-zeroing
  section): `src/bst_refactor/validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`
- Phase-2 raw mmpose ndet baseline:
  `src/bst_refactor/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md`
