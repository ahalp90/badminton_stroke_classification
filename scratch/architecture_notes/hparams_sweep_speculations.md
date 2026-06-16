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
sweep budget look like? The frame-zeroing redesign that this doc
originally covered now lives in
[`frame_zeroing.md`](frame_zeroing.md); a stub remains here as a
research-direction pointer.

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

**Augmentation status**: locked Task 2 set on 2026-05-04. Full
analysis extracted to
[`augmentation_framework.md`](augmentation_framework.md). Active
set: centreline flip (p=0.5, coupled) plus corrected pos+shuttle
constrained-jitter (p=0.2 nominal, ±0.05y / ±0.10x cap). Replaces
the broken `RandomTranslation_batch` (joints-only, decoupled, body-
deforming). First aug ablation slot is A/B-ing the corrected
formulation against the no-aug baseline; magnitude and frequency
sweeps follow. Plays before dropout in the runbook because the two
share regularisation budget and you want to fix augmentation first.

**Architecture-side priors from the verified TemPose/BST ablations**:
`(DL=100, DA=128)` is empirically validated as a winner on Bad OL,
so the only sane geometry experiment is *moving up* the size curve
(testing whether ShuttleSet's larger N now unlocks the (200, 256)
cell that overfit on Bad OL). Depth-wise, BST inherits `(LT=2,
LN=1)` but TemPose's depth sweep favours `(LT=2, LN=2)` by 0.7%, so
flipping `depth_inter=1 → 2` is a single high-confidence cell.

**Frame-zeroing redesign**: extracted to
[`frame_zeroing.md`](frame_zeroing.md). Two live items: drop the
asymmetric shuttle-on-pose-fail wipe in collation (~14k frames
recovered), and add variant-2 mask channels (combined
`pose_missing_either_slot` + `shuttle_missing`) to replace the
implicit "(0, 0) means missing" overload on the 6.25%
TrackNet-visibility=0 frames. Re-promoted from "demoted" on
2026-05-03: per-clip re-cut showed smash and wrist_smash have
asymmetric mid-clip zeroing (whole-clip mean 13.7% / 8.5%, 1.6x
ratio), so the redesign plausibly bites on the bottleneck pair,
not just the head of the F1 distribution. Run after the
capacity-bump confirmations land, framed as a disambiguation
between coarse-class-prior and shuttle-data-quality mechanisms.

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
- **Frame-zeroing redesign re-promoted (2026-05-03).** Initial
  demotion was on the +-10-of-hit framing (shuttle present on
  bottleneck classes). The per-clip re-cut showed smash and
  wrist_smash have asymmetric mid-clip zeroing (whole-clip mean
  13.7% / 8.5%, 1.6x ratio); the redesign plausibly bites on the
  pair-confused bottleneck pair, not just the head. Run after
  the capacity-bump confirmations. Detail in
  [`frame_zeroing.md`](frame_zeroing.md).

**Highest-priority near-term experiments (revised 2026-05-04)**,
in order: capacity-bump Run 2 confirmation (`d_model` 100 → 192 +
`d_head` trim 128 → 32, vs `run_20260501_164658`); landing the
locked augmentation set (centreline flip + corrected pos+shuttle
jitter, see [`augmentation_framework.md`](augmentation_framework.md));
X3D-S fusion build (long-term primary direction; the lever capacity
research argues actually addresses the data/signal bottleneck);
weight decay sweep; `depth_inter=1 → 2` run.

## Augmentation framework

Detailed analysis, code traces, locked Task 2 spec, and Phase 3
candidates extracted to
[`augmentation_framework.md`](augmentation_framework.md) (this
directory) on 2026-05-04 once the augmentation discussion outgrew
this doc. Punch list:

- **Locked Task 2 set**: centreline flip (p=0.5, coupled, COCO
  bilateral joint-index swap) plus a corrected pos+shuttle
  constrained-jitter (p=0.2 nominal, ±0.05y / ±0.10x cap, layered
  conditional bounds, joints/bones untouched, zero-frame
  preservation, shuttle off-screen mirroring). Replaces the broken
  `RandomTranslation_batch` (joints-only, decoupled, body-deforming).
- **Out for Task 2**: temporal speed jitter (Phase 3 candidate),
  Gaussian joint jitter, random joint masking,
  `WeightedRandomSampler`, net flip.
- **Phase 3 / trimester 2 candidates**: temporal speed jitter
  (uniform [1.0, 2.0] coupled with shuttle-velocity downweight cost
  flagged), rotation / scaling / shearing for amateur cameras,
  per-joint adaptive focal as a loss-side research direction.
- **Coordinate spaces verified**: pos in court frame
  (post-homography), shuttle in camera frame, joints in
  bbox-centre-relative frame; PPF fuses pos into JnB at the input,
  before the TCN. Out-of-court pos values flow through unclamped
  within sticky_anchor's [-0.15, 1.15] band.
- **X3D-S hit-frame metadata** derivable without re-extraction via
  `clips_master.csv` correlation (Method A, faithful to annotation,
  susceptible to annotator drift) or shuttle horizontal-velocity
  sign reversals (Method B, independent verification with ±5-frame
  ceiling on soft shots, well within X3D-S's ±19-frame window).
- **Calibration mechanism**: `effective_aug_rate` TB scalar logs
  per-epoch case-1 (fully-degenerate) dropout against nominal
  `p_jitter_roll`; tune nominal upward if effective sits below
  target.
- **First aug ablation slot**: A/B the corrected jitter formulation
  against the no-aug baseline before adding any further
  augmentation arms.
- **Round 1 sweep seed-noise envelope (2026-05-06)**: two
  identical-config 5-serial runs gave run-mean spread ~0.1% on
  macro/accuracy/top-2 and ~2% on min F1. Macro is the reliable
  signal at this sample size; min F1 stays the success criterion
  but is too noisy to drive decisions. Detail:
  [`augmentation_framework.md`](augmentation_framework.md),
  "Seed-noise envelope on run means" section.

Full code traces, implementation outlines, magnitude / frequency
rationale, ablation gates, and physics-of-non-uniform-temporal-aug
analysis all in the framework doc.

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

So everything in `bst_x_train.py:62-77` and `model/bst.py:140-141`
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
depth_inter=1`** (per `bst_x_common.build_bst_x_network` defaults) but
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
   early_stop=300. Our retune (Q4 in `bst_x_overview.md`)
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
- Frame-zeroing policy: paper text says "If there were less than
  two people in the court, we cleared the information (poses and
  shuttlecock trajectory) of that frame to zero" (Supplementary
  E.). That's an OR-zero of pose AND shuttle; our refactor only
  does the shuttle-on-pose-fail half (asymmetric, picked-slot
  pose flows through, shuttle wiped whenever any slot is
  unpicked). Unexamined inheritance, not a tuned choice. Detail
  and redesign target in [`frame_zeroing.md`](frame_zeroing.md).

So our sweep candidates are real gaps, not "things the authors
already ruled out". The BST contribution is architectural (CG/AP
modules + clipping strategy), not optimisation-side.

## Smash/wrist_smash F1 split has flattened (round 1 sweep finding, 2026-05-06)

Picked serials from the aug hparam sweep round 1 (`sweep_20260505_211814_aug_v1_round_1`) showed a pattern shift on the smash↔wrist_smash pair:

| Run | Cell | PICK serial | wrist_smash F1 | smash F1 |
| --- | --- | --- | --- | --- |
| `run_20260505_213008_504674` | p_flip_25 | S2 | 0.510 | 0.567 |
| `run_20260506_011851_522295` | p_jitter_40 | S3 | 0.510 | 0.605 |
| `run_20260506_032632_652587` | p_flip_25_x_p_jitter_30 | S1 | 0.523 | 0.568 |
| `run_20260505_154907` (prior ref) | aug v1 + jit 0.3 | S5 | 0.519 | 0.515 |

ws used to sit clearly below smash and was the project floor on most serials. This round's picks show roughly equal F1 on both (~0.51-0.52 ws, ~0.51-0.60 smash); which one is the floor varies by serial. The prior ref's S5 was the first picked serial where smash was the floor; this round's three picks all show the pair in the same band.

Two reads, same operational implication:

- **Equal confusion**: the model has stopped biasing one class over the other and is confused on both at roughly the same rate. Flip and jitter don't add information for the smash-vs-ws distinction, so the model settles into splitting between them when uncertain. The 50/50 split is the model giving up rather than improving.
- **Signal ceiling**: smash and wrist_smash overlap in pose-only features and ~0.5/0.5 is what the data actually supports. Augmentation can't move it because the missing signal isn't on any aug axis.

Both point at more signal needed. That's the X3D-S wrist crop bet: visual context at the wrist (racket angle, tip motion, contact frame) that pose-2D throws away. Lines up with the capacity-bottleneck read at `model_capacity_bottleneck_question.md`: plateau is data-bound and signal-bound, not capacity-bound.

Caveat: the pattern is at picked-serial level, which is noisier than run means. Companion seed-noise finding at [`augmentation_framework.md`](augmentation_framework.md), "Seed-noise envelope on run means" section, shows run-mean min F1 swings ~2% from seed at 5 serials. Soft claim: no picked serial this round shows the historic wide-gap pattern. Quantitative size of the shift is bounded by the noise floor.

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

**Status correction (2026-05-04).** The original framing of this
section read the live aug as coupled across streams. That's wrong.
Code trace at `bst_x_train.py:198-205`: `random_shift_fn` is called
on `human_pose` only (joints slice of it when bones are active,
since bones are translation-invariant), and `shuttle` and `pos` are
passed straight through. The shift is per-sample (`(n, d)` shape at
`shuttleset_dataset.py:132-133`), not batch-uniform. So the live
aug is **joints-only ±0.3 with p=0.3, shuttle and court untouched**.
That violates Rule 1 of Isiah's writeup §3 and is actively
mis-training the cross-attention. Fix is the first slot in the
2026-05-04 augmentation runbook: either remove or replace with a
coupled per-clip translation across pose+shuttle+court.

**What it does (corrected).** With probability 0.3, adds a uniform
random xy shift in [-0.3, 0.3] to the joint coordinates of each
sample in the batch (`shuttleset_dataset.py:121-136`). Shuttle and
`pos` (court positions) are not shifted, so the relative geometry
*across modalities* breaks for ~30% of training batches.

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

**Genuinely worth sweeping?** This is Q5 in `bst_x_overview.md`
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
(`bst.BST.__init__`, `bst_x_common.build_bst_x_network` line 52-53,
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
hardcoded in `bst_x_common.build_bst_x_network` defaults rather than
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

Extracted to [`frame_zeroing.md`](frame_zeroing.md). Two live items:

1. **Shuttle gets wiped on any pose-fail frame** (collation
   asymmetry, ~14k frames). One-line collation fix: drop the
   `shuttle[failed, :] = 0` line in
   `prepare_train_on_shuttleset.py:864-867`.
2. **Shuttle (0, 0) on TrackNet-visibility=0 frames** (6.25% of
   frames) is indistinguishable from a real top-left detection
   because the visibility flag is dropped at collation. Recommended
   fix: variant-2 mask channels (combined `pose_missing_either_slot`
   + `shuttle_missing`).

Initially demoted on the +-10-of-hit framing (script 0c showed
shuttle-miss rate correlates *positively* with per-class F1 at
+0.516 Pearson). Re-promoted on 2026-05-03 after the per-clip
re-cut: smash and wrist_smash have asymmetric mid-clip zeroing
(whole-clip mean 13.7% / 8.5%, 1.6x ratio) that the +-10 framing
hid. The redesign plausibly bites on the pair-confused
bottleneck pair, framed as a disambiguation between
coarse-class-prior and shuttle-data-quality mechanisms. Run
after the capacity-bump confirmations land. Trajectory
extrapolation still parked; the transformer's learned
representation given a `shuttle_missing` mask would probably
outperform any hand-crafted physics prior anyway.

Full code trace, cohort table, gap analysis, mask-channel
variants, and the parked interpolation idea are in the linked
doc.

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
| `label_smoothing` x focal loss | Both regularise classifier confidence. Stack weirdly per the spec in `bst_x_overview.md`. | The focal-loss spec already disables LS in the focal arm. **No joint sweep.** Run focal vs CE separately. |
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
| Zeroing redesign (drop shuttle-on-pose-fail wipe) vs current | 2 | 6 | ~6-9 + ~30 min recollation each |
| Mask channels (variant 2: combined pose + shuttle) | 1 (vs zeroing-redesign baseline) | 3 | ~3-5 + dataset code change |
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
   enough); LS=0.2 deferred. Full numbers in `bst_x_overview.md`.
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
   numbers in `bst_x_overview.md`.
2a. **Capacity-bump confirmation runs**.
    Capacity-bottleneck research at
    `scratch/architecture_notes/model_capacity_bottleneck_question.md`
    argues the plateau is data-bound and signal-bound. Two cheap
    confirmatory runs while compute is around.
    - **Run 1 (mlp_head hidden 400 → 1200)** — **DONE
      (2026-05-03)** as `run_20260503_104300`. One-line swap at
      `bst.py:199` from `d_model * mlp_d_scale` to
      `head_dim * mlp_d_scale` (4x of actual head input rather
      than 4x of d_model). 1200 picked over the earlier 768
      candidate for FFN-ratio consistency. Result: head metrics
      flat (mean macro -0.2 pp vs y1t1), wrist_smash mean
      -4.8 pp, pair-confusion trade went smash-up / ws-down.
      The mlp_head swap has been **reverted** at `bst.py:202`
      to keep the baseline clean for Run 2 and any subsequent
      work. Full numbers in `bst_x_overview.md` 2026-05-03
      block.
    - **Run 2 (d_model 100 → 192 + d_head trim 128 → 32)** —
      pending. Encoder-side widening; residual stream goes 1.92x
      wider, d_head trim eats some of the attention budget back
      so the 7.68x d_head:d_model over-provisioning doesn't
      propagate. Run 1's flat result weakens but doesn't void
      the prior: encoder-side capacity is a different mechanism
      from head-side, and the pair-confusion failure mode is
      representation-bound rather than head-bound, so Run 2 has
      a small theoretical advantage Run 1 didn't. Expected gain
      still 0-2 pp on test macro. Implementation surface +
      verification checklist + LR-schedule notes at
      `scratch/architecture_notes/transformer_widening_hparam_changes.md`.
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
9. **Frame-zeroing redesign** (drop the shuttle-on-pose-fail wipe):
   re-collate, run 3 serials, compare. Re-promoted on 2026-05-03
   after the per-clip re-cut: smash and wrist_smash have
   asymmetric mid-clip zeroing (whole-clip mean 13.7% / 8.5%,
   1.6x ratio) that the +-10 framing hid. Frames the experiment
   as a disambiguation between coarse-class-prior and
   shuttle-data-quality mechanisms for the smash↔wrist_smash
   confusion. If both halves of the pair lift, mechanism 2 is
   real; if neither moves, mechanism 1 is doing the work and the
   smash-prior is structural.
10. **Two-channel mask** (`shuttle_missing` + `pose_missing_either_slot`):
    contingent on the zeroing redesign lifting either pair member.
    Per-clip data shows ~50% of the missing-shuttle frames in
    smash and wrist_smash sit in the central window (high-arc
    setup), so an explicit `shuttle_missing` channel could let
    the model condition on "off-screen-arc phase" rather than
    inferring it from the (0, 0)-overload. Lever size depends on
    which mechanism is operative.
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
redesign, re-promoted on 2026-05-03 after the per-clip re-cut
showed asymmetric mid-clip zeroing on smash vs wrist_smash;
positioned to disambiguate the smash↔wrist_smash pair-confusion
mechanism. Lines 11-13 are architecture / data-side cleanup, save
until the rest has been worked. Line 14 is a longer-term
direction flagged by the 0b result (heavy population of
unrecoverable mid-length off-screen gaps).

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
  section): `src/bst_x/validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`
- Phase-2 raw mmpose ndet baseline:
  `src/bst_x/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md`
