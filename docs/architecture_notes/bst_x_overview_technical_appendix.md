# BST-X technical appendix

The dense companion to [`bst_x_overview.md`](bst_x_overview.md). Per-run best/mean numbers live in the ledger ([`bst_x_training_runs.md`](../../experiments/bst_x/bst_x_training_runs.md)); run numbers (`#n`) here index it. This doc holds the workings: per-experiment reasoning and per-class movement, comparability caveats, configuration and architecture detail, and the parked and future work (un-run experiments, unbuilt features, the cleanup backlog). Bug fixes have their own file, [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md). Each experiment-log entry has one bolded takeaway, so skimming the bold lines gives the story.

## Contents

- [Configuration and architecture detail](#configuration-and-architecture-detail)
  - [Schedule](#schedule)
  - [Loss / label smoothing](#loss--label-smoothing)
  - [Data path and collation](#data-path-and-collation)
  - [X3D-S branch: design detail](#x3d-s-branch-design-detail)
- [Experiment log](#experiment-log)
  - [LR schedule retune (Q4), 2026-04-17](#lr-schedule-retune-q4-2026-04-17)
  - [CG/AP annealing ablations (Q3), 2026-04-19](#cgap-annealing-ablations-q3-2026-04-19)
  - [sticky_anchor Phase 1 mixed retrain, 2026-04-25](#sticky_anchor-phase-1-mixed-retrain-2026-04-25)
  - [Collapsed-classes ablation, 2026-04-25](#collapsed-classes-ablation-2026-04-25)
  - [Phase 2 sticky_anchor full extract, 2026-04-29](#phase-2-sticky_anchor-full-extract-2026-04-29)
  - [Phase 2 sanity-train, 2026-04-30](#phase-2-sanity-train-2026-04-30)
  - [Pre-flight diagnostic scripts: shuttle-missing diagnosis, 2026-04-30](#pre-flight-diagnostic-scripts-shuttle-missing-diagnosis-2026-04-30)
  - [Loss-side ablation arm, 2026-05-01 to 2026-05-02](#loss-side-ablation-arm-2026-05-01-to-2026-05-02)
  - [Unknown ghost channel removed, 2026-05-01](#unknown-ghost-channel-removed-2026-05-01)
  - [Capacity-bottleneck research, 2026-05-02](#capacity-bottleneck-research-2026-05-02)
  - [Capacity Run 1: mlp_head 400 to 1200, 2026-05-03](#capacity-run-1-mlp_head-400-to-1200-2026-05-03)
  - [Capacity Run 2: d_model 100 to 192 + d_head trim 128 to 32, pending](#capacity-run-2-d_model-100-to-192--d_head-trim-128-to-32-pending)
  - [Lost-frame recovery (shuttle-unzeroing), 2026-05-03](#lost-frame-recovery-shuttle-unzeroing-2026-05-03)
  - [Mask-channel variant 2a (`shuttle_missing`), 2026-05-03](#mask-channel-variant-2a-shuttle_missing-2026-05-03)
  - [Augmentation set locked, 2026-05-04](#augmentation-set-locked-2026-05-04)
  - [Jitter-off ablation, 2026-05-04](#jitter-off-ablation-2026-05-04)
  - [Aug v1 first run, 2026-05-05](#aug-v1-first-run-2026-05-05)
  - [Aug v1 + p_jitter=0.3 retune, 2026-05-05](#aug-v1--p_jitter03-retune-2026-05-05)
  - [Aug v1 round 1 hparam sweep, 2026-05-06](#aug-v1-round-1-hparam-sweep-2026-05-06)
  - [Series G: multi-taxonomy baseline batch, 2026-05-30](#series-g-multi-taxonomy-baseline-batch-2026-05-30)
  - [Series H: val-gate + focal-alpha-revert across taxa, 2026-05-31](#series-h-val-gate--focal-alpha-revert-across-taxa-2026-05-31)
  - [Series I: weight-decay sweep on une_v1_14, 2026-05-31](#series-i-weight-decay-sweep-on-une_v1_14-2026-05-31)
  - [Series J: wd-endpoints and focal-alpha-revert across taxa, 2026-06-02](#series-j-wd-endpoints-and-focal-alpha-revert-across-taxa-2026-06-02)
- [Parked and future work](#parked-and-future-work)
  - [Experiment ideas not yet run](#experiment-ideas-not-yet-run)
  - [Features not yet built](#features-not-yet-built)
  - [Cleanup backlog](#cleanup-backlog)
- [Comparability caveats)](#comparability-caveats)
- [Cross-references](#cross-references)

## Configuration and architecture detail

### Schedule

`bst_x_train.py:62-79` plus the cosine call at `:308-314`: `n_epochs=80`, `early_stop_n_epochs=40`, `batch_size=128`, `lr=5e-4`, `warm_up_step=100`, `num_cycles=0.5`, `use_aux_schedule=True`, `aux_fade_end_epoch=15`. The shape is a compressed warm-start-then-finetune: ~4 epochs warmup, ~15 epochs of CG/AP warm-start ramping to 0, then ~65 epochs of pure backbone under cooling LR. The BST paper's defaults (`n_epochs=1600`, `warm_up_step=400`, `early_stop_n_epochs=300`, `num_cycles=0.25`, `aux_fade_end_epoch=60`) and the dated retune rationale are at `historical_bst.md` section 3.

### Loss / label smoothing

`adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`, label smoothing 0.0. LS=0.15 won the LS sweep on combo A nosides, but LS=0.0 carries the active CDB-F1 runs because adaptive focal already supplies the rare-class tax that label smoothing was approximating.

### Data path and collation

The shuttle-unzeroing fix landed 2026-05-03 as `ablation_id=wipe_drop` (branch `shuttle/wipe-drop`, merged in commit `4e478fc`): the `shuttle[failed, :] = 0` line at `prepare_train_on_shuttleset.py:866` is gone, replaced by a prose comment. ~14k frames recovered (0.84% of extract); the shuttle now flows through unmodified on pose-fail frames, mirroring the pose-flows-through-on-shuttle-fail behaviour that already existed.

From Series G (2026-05-30) onward the fix is baked into the `taxon_pinned_w_preds` collation rather than applied as the `wipe_drop` flag. That pinned collation (per-taxonomy, canonical registry) is the active tree; it superseded the earlier `npy_wipe_drop` sub-dir, so Series G onward set `collation_id=taxon_pinned_w_preds` and `ablation_id=null`. Both the `--ablation-id wipe_drop` path and the older default-named `npy_<tax>_<split>_dropunk` dirs (still holding the pre-fix shuttle-zeroed collation) are historical now.

**Backward compat with old weights**: the architecture is unchanged, so old weights from `run_20260501_164658` and earlier load cleanly against the new trees (no shape mismatch). But test-time output shifts slightly, because the previously-zeroed frames now carry real shuttle xy the old weights never trained against in that exact distribution. Architecture compat yes; inference reproducibility against the original training distribution no. Flag it when comparing old-weight inference numbers against runs trained on the new trees.

**Active classes**: the head dim comes from the classes present in train `labels.npy` at first serial, via `bst_x_common.derive_active_classes_from_labels` (val/test asserted as subsets); manifest `extra.arch` records `n_classes_full`, `n_active_classes`, `has_unknown`, `unknown_first`, `active_class_list`. Pre-fix (pre-2026-05-01) v1/nosides/raw_35 weights are not mechanically resumable post-fix (head-dim shape mismatch); merged_25 dropunk is comparable across that boundary. Full story in [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md).

### X3D-S branch: design detail

The overview has the summary; this is the reasoning behind the choices.

**Model choice.** X3D-S fits two constraints: it's easily available with weights, and small enough to fine-tune end-to-end in the time available on a v100 16gb. Other strong low-param models (MoViNet) lack prebuilt weights and easy model-zoo integration. X3D would likely do even better with SSv2 pretraining (fine hand motions), but the SSv2 weights only exist as an unofficial TensorFlow port, and the interface bugs would probably eat more time than they save. Within the family I took S over XS/M/L/XL: XS expects 4 frames at stride 12, too coarse for granular racket motion; M/L/XL only drop stride to 5, perform not-that-much better, and cost far more params; X3D-S gives strong accuracy at low param count, default input 13 frames at stride 6.

**Target input shape: 39 frames, stride 1.** I'm fine-tuning toward `frames=39, stride=1` rather than the default `13 x stride=6`. stride 1 gives the model every frame and lets it learn the interactions between them, which granular racket motion needs. 39 is set so that by the final convolutional block the receptive field covers all input frames; that caps the window around ~40 frames, fine for a racket crop centred on a stroke.

**Fusion depth (open).** Three options. Late concat just before the MLP head: easiest, lowest risk, but BST never gets to condition its attention on the racket signal. Tie into the attention earlier, feeding X3D-S output into the cross-attention or interactional transformer so the racket evidence shapes how players and shuttle attend: more expressive, more moving parts. Or a separate tower with a learned gate on how much its prediction counts versus BST's: keeps the branches clean and lets the model decide per-sample how much to trust the racket signal.

**Open training questions.** (1) Schedule: the right sequence for fine-tuning X3D-S on badminton video first, then co-training end-to-end, the length of each phase, the learning rates, what to freeze when. (2) Temporal cut-in: the reported contact times are noisy, so where the X3D-S window sits relative to them matters; options are a fixed offset centred on the reported time, a learned offset, or a wider window that self-aligns. Hit-frame metadata is derivable without re-extraction (Method A: CSV correlation; Method B: shuttle trajectory inversion; detail in `augmentation_framework.md`). (3) MMPose drops: the window has to cope with the residual detection-layer drops (heavy net occlusion); the candidate fix is temporal interpolation, worst case pinning to the shuttle velocity-reversal frame.

**Wiring note.** When I add the X3D-S branch, I'll source the head dim from `task.n_active_classes` and run `_validate_and_record_arch` on serial 1, mirroring the bst_x_train.py pattern. Hardcoding `taxonomy.n_classes` in the fusion module would put the unknown ghost back.

## Experiment log

Chronological. Each entry: setup, what I expected, what happened, the takeaway (bolded), run id(s).

### LR schedule retune (Q4), 2026-04-17

`bst_x_train.py:308-314` calls `get_cosine_schedule_with_warmup`. The original BST recipe passed `num_cycles=0.25` alongside `n_epochs=1600`, `warm_up_step=400`, `early_stop_n_epochs=300`. At `num_cycles=0.25` only a quarter of the cosine runs across the budget, so the LR barely decays; BST-default runs converge around epoch 60 and early-stop fires around 360, so the scheduler never had time to lower the rate. I compressed `n_epochs` to match real convergence and bumped `num_cycles` so the cosine actually hits zero:

| param | was | now |
|---|---|---|
| `n_epochs` | 1600 | 120 |
| `warm_up_step` | 400 | 100 |
| `early_stop_n_epochs` | 300 | 40 |
| `num_cycles` | 0.25 | 0.5 |

Run `run_20260417_191851` (#2, 3 serials on merged_25). The winner hit macro 0.830 / min 0.627 / acc 0.844 / top-2 0.964 against the paper's 0.8097 / 0.5762 / 0.8322, and the val-vs-test direction flipped (old run val 0.831 / test 0.823, retune winner val 0.816 / test 0.830). **All three serials beat the paper on every metric, so it isn't a lucky seed, and the win lands hardest on min-F1: the harder classes benefit most.**

### CG/AP annealing ablations (Q3), 2026-04-19

CG (Clean Gate) and AP (Aim Player) originally ran unweighted for the whole run. I expected their real value to be as a warm-start prior: early on the transformers haven't learnt robust shuttle/player representations, so the hand-crafted interactions are useful bias; later, once the transformers learn their own richer interactions, fixed CG/AP can pin the model to the hand-crafted formulation. Three matched 5-serial runs under the retuned schedule, only the CG/AP schedule varying:

| Arm | aux_factor over epochs | Run | Mean macro |
|---|---|---|---|
| Annealed out | 1.0 at ep1, cosine to 0 by ep15, then 0 | `run_20260418_151139` (#4) | 0.829 |
| Always on | 1.0 for all 80 epochs | `run_20260418_174238` (#5) | 0.826 |
| Always off | 0.0 for all 80 epochs | `run_20260418_234822` (#6) | 0.822 |

Annealed beats always-on beats always-off, small but consistent. The accuracy peak suggests CG/AP cap the top end when sample count is high. **CG/AP earn their keep as warm-start bias, and the tuned LR explains most of the gap from the original BST stats.**

### sticky_anchor Phase 1 mixed retrain, 2026-04-25

I reran the V4 baseline (`run_20260420_171101`) with the 1,716 hit-zone-busted clips swapped for their sticky_anchor-cleaned versions, nothing else changed (`run_20260425_150548` #9). The heuristic doc's decision gate wanted a +0.02 target-class min-F1 lift; this failed it: mean macro +0.007, min -0.056, acc +0.008. Top_wrist_smash dropped 0.057 and Top_smash gained almost exactly what it lost (+0.020), a boundary-allocation trade, cleaner data made the smash family easier and the model spent the gain on the easier head class, not the rare tail. A per-class frame-zeroing audit followed: the F1-bottom classes weren't the heavily-zeroed ones, and the worst-zeroed class hit near-perfect F1. **At the keypoint level, data quality isn't the floor bottleneck**, so Phase 2 was deprioritised on this finding (and only revived later when the shuttle-side asymmetry turned up, which is a different problem).

### Collapsed-classes ablation, 2026-04-25

Same data, only the label space changes: 28 classes to 14 by dropping the Top_/Bottom_ prefix (taxonomy `une_merge_v1_nosides`, `run_20260425_185421` #10). I expected Top_X and Bottom_X to be the same shot mirrored across the net, so splitting them halves per-class N for a redundant distinction. Every metric landed within +/-0.008 of V4, the absolute ceiling didn't move. What did move was rare-class stability: the per-seed test-min range dropped from 0.124 to 0.074 and the worst-seed min lifted from 0.235 to 0.350. The 14-class wrist_smash F1 (~0.42) sits close to the support-weighted mean of the old Top_ (0.33) and Bottom_ (0.45) variants, so the model isn't separating smash from wrist_smash any better, the metric just stopped flipping between two thin slots. **Doubling per-class N cut the seed lottery; absolute performance held.**

### Phase 2 sticky_anchor full extract, 2026-04-29

A full re-extract of all 32,203 stems (Phase 1 only repaired 1,716). Three artefacts:

- **Raw extract**: 32,203 stems at `/scratch/comp320a/ShuttleSet_keypoints_raw/` on both nodes, bit-identical (30,487 freshly re-extracted over ~20h plus the 1,716 Phase-1 backfill). Verified by file counts, an empty cross-node `rsync --checksum`, and a byte-identity gate on the overlap with max abs diff 0.000e+00. Per-frame `ndet` baseline at `raw_ndet_stats_outputs/baseline_2026-04-29.md` (0% `ndet=0`, 0.53% `ndet=1` floor).
- **sticky_anchor + audit**: clean dir at `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`, byte-identical cross-node. Overall fail rate 5.38% to 0.93%, hit-zone near-hit fail 5.98% to 0.58% (the near/away gradient flipped sign: the hit zone is now the cleanest zone, not the noisiest), per-stroke ratios 19x-76x on the worst-hit strokes. 17 residual fully-zeroed clips look like irreducibly broken broadcasts. Writeup at `mmpose_heuristic/phase1_vs_phase2_2026-04-29.md`.
- **Collation + env flip**: three collated trees per active (taxonomy, split) combo, mirrored, byte-identical, counts cross-checked against `clips_master.csv`. `BST_X_MMPOSE_NPY_DIR` flipped to the clean dir, one-step rollback at `.env.bak.2026-04-29`.

### Phase 2 sanity-train, 2026-04-30

Three 5-serial runs across the active combos:

| Combo | Run | macro | min | vs prior baseline |
|---|---|---|---|---|
| C: `merged_25 + baseline` | `run_20260429_202144` (#11) | 0.831 | 0.577 | tied on macro/acc/top-2, -0.022 on min, seed variance ~2.5x tighter |
| B: `une_merge_v1 + v2` | `run_20260430_110101` (#12) | 0.739 | 0.317 | within noise on macro/acc/top-2, min drops 7pp as Top_wrist_smash gets worse with cleaner pose |
| A: `une_merge_v1_nosides + v2` | `run_20260430_170325` (#13) | 0.742 | 0.375 | recovers most of B's wrist_smash floor via the side-collapse (+0.058 min vs B) |

Cleaner pose lifts head metrics and tightens seed variance on common classes, but hurts the small-support tail in the un-pooled 28-class taxonomy. **The diagnosis moved to classifier-side, not data-side.** Caveats on file: keep the legacy `_merged_25` nested tree (the only path to bit-exactly reproduce V4); the unknown class has no pose data (1,278 clips excluded because every active taxonomy drops unknown).

### Pre-flight diagnostic scripts: shuttle-missing diagnosis, 2026-04-30

Three scripts under `validation_scripts/`. `shuttle_gap_y_distribution.py`: 61.6% of gap boundaries cluster in the top 10% of the broadcast frame, 72.3% on the re-appearance side, confirming the off-screen-high hypothesis at the sensor level. `shuttle_gap_length_distribution.py`: the inpaint module isn't being exceeded (1 gap >60 frames in 32k clips); 85% of missing-shuttle frames sit in the 11-60 frame band. `perclass_shuttle_miss_vs_f1.py`: Pearson +0.516, the opposite of the predicted direction, the high-shuttle-miss classes are the pose-distinctive serves/clears/lobs at 0.95-0.99, while the bottleneck classes sit at sub-1% miss rates. **Shuttle data is reliably present where it's most needed; the model just wasn't using it well.** So the mask-channel arm got demoted, and label smoothing became the top loss-side experiment.

### Loss-side ablation arm, 2026-05-01 to 2026-05-02

Sequential 5-serial sweeps on combo A nosides + v2 + dropunk; each step gated the next.

**LS sweep.** LS=0.0 (#14): head metrics flat vs LS=0.1, mean wrist_smash down 1.6pp, so "LS=0.1 was taxing rare-class confidence" is disproved. LS=0.15 (#15): macro +0.005, min +4.2pp, head metrics flat, the wrist_smash range tightens 0.159 to 0.066 and the whole distribution shifts up. LS=0.15 wins.

**Class-weighting smoke** (#16, LS=0.15 + `class_weights{wrist_smash:2.0, smash:2.0}`). The central tendency barely moved (+0.001 macro / +0.005 min), but S2 wrist_smash 0.518 set a new project-wide ceiling (the first nosides serial past 0.50), and S4 set ceilings on macro 0.756, acc 0.777, drive 0.66. The seed distribution went bimodal: one seed found a wrist_smash basin no prior serial reached, three stayed in the 0.37-0.40 range. **Static reweighting moves the ceiling, not the mean, so the loss-side axis isn't exhausted.**

**Decision: skip vanilla focal, jump to CDB-F1.** Vanilla and manually-alpha focal are the same lever as class-weighted CE, just per-sample-gated, so they'd hit the same central-tendency ceiling. CDB-F1 (per-class alpha = `(1 - F1_c)^tau` from an EMA of train F1, optionally times focal `(1 - p_t)^gamma`) is the right escalation: low-F1 classes get persistently escalated weight, which can push bad seeds toward the basin S2 found. Design in `class_f1_focal_design.md` (verified against the ACCV 2020 paper); the Seesaw-style companion in `seesaw_f1_focal_design.md` is held as a second arm.

**CDB-F1 first run** (#17, LS=0.0 + `adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`). vs class-weighted: macro -0.5, min +4.0, acc -0.8. vs LS=0.1 baseline: macro +0.1, min +8.7. The range tightens to 0.413-0.486 (vs 0.378-0.518 class-weighted), so the bimodal-seed problem is solved, though the ceiling isn't broken (S2 0.486 < class-weighted 0.518). Per-class shifts vs class-weighted: ws +4.0, **push +6.7** (adaptive picked it up as a second bottleneck the static config missed), smash -5.5 (pair-confusion with ws); the rest -1 to -3. Shipped as `loss/adaptive_focal.py` (~190 lines) + 6 edits in `bst_x_train.py` + 36 unit tests; with `adaptive_focal=None` the legacy path is bit-identical. **The largest floor lift on wrist_smash of any loss-side run.**

**CDB-F1 follow-ups.** gamma=0 (#18) and tau=0.5 (#19) both lose 2.8-5.0 of the ws lift and smash recovers, so the ws gain came from the aggressive tau=1 alpha and the gamma=1 modulator together; soften either and ws drops back. pair-cap (#20, `alpha[smash]/alpha[wrist_smash] >= 0.7`): smash recovered 1.2 of its 5.5, ws gave back 5.2 of its +4, macro flat as the trade cancelled. gamma=2 (#21, the RetinaNet default): traded 4.1 of ws for 1.3 of smash, macro -0.7. **Every CDB knob is now run, tau=1 + gamma=1 is the floor-lift sweet spot, and the smash drop is structural pair-confusion a scalar per-class alpha can't resolve.** No CDB run breaks the 0.74-0.75 macro plateau; per-class trajectory in `train_val_test_split_analysis.md`.

### Unknown ghost channel removed, 2026-05-01

The head is now sized to the classes present in the train labels, not `taxonomy.n_classes`, so a drop-unknown run no longer carries a dead `unknown` output channel. **This is a bug fix, not an architecture change.** The one comparability fact worth keeping here: pre-fix dropunk v1/nosides/raw_35 weights aren't resumable post-fix (head-dim shape mismatch), while merged_25 dropunk is comparable across the boundary. Full diagnosis in [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md) and `unknown_channel_fix_review.md`.

### Capacity-bottleneck research, 2026-05-02

Full writeup in `model_capacity_bottleneck_question.md`: "are we at the useful parameter ceiling?", with theory plus reference-class numbers. BST at 1.85M params on 32K clips sits in the converged 1-3M zone for skeleton-AR (ST-GCN 1.22M through Hyperformer 2.6M), at the high end of per-sample density (~81 params/sample). From-scratch small-video baselines (X3D-M 3.76M, MoViNet-A0 3.1M) are 1.5-2x BST; flagship transformers need ImageNet pretraining; the from-scratch video-transformer literature points at pretraining, not from-scratch widening, as the small-data lever. The train-test gap concentrates on smash/ws (14-18pp) while pose-distinctive classes generalise within 1-2pp at 0.95+. **The plateau is data-bound and signal-bound, not capacity-bound**, so pure widening should gain only 0-2pp on test macro.

### Capacity Run 1: mlp_head 400 to 1200, 2026-05-03

Classifier-side only: a one-line swap applying the FFN-block 4x ratio to the actual head input (300 on the CG/AP path) rather than to d_model, so hidden went 400 to 1200 with the encoder untouched (#22, a parity test vs #17). Head metrics stayed flat and ws cost 4.8pp on the mean; even the best seed (S1 ws 0.4449) sits below #17's mean. Per-class shifts vs #17: ws -4.8, push -3.6, smash +1.4; the other classes within ~2pp. The bigger head traded ws and push down for smash up, the same pair-confusion direction as pair-cap and gamma=2, larger on ws. **A bigger head doesn't move the head metrics and costs ws, exactly as the capacity research predicted.** Swap reverted at `bst.py:202`.

### Capacity Run 2: d_model 100 to 192 + d_head trim 128 to 32, pending

Encoder-side widening: d_model up 92% on the residual stream, d_head trimmed in the same change so the 7.68x d_head:d_model over-provisioning doesn't propagate (Voita's WMT pruning result, 38/48 heads removable for 0.15 BLEU, says the per-head allocation is over-provisioned). Run 1's flat result weakens but doesn't void the prior: the pair-confusion failure is representation-bound, and a wider head couldn't fix it, but a wider encoder is at least topologically positioned to. The bump propagates through TCN, cross-transformer, interactional transformer and FFN; per-epoch wall-time ~+30-60%. Surface and verifications in `transformer_widening_hparam_changes.md`. **If it lifts macro 1+pp without burning ws, capacity has a small lever; if it flat-lines, the capacity question is answered and X3D-S is the right next thing.**

### Lost-frame recovery (shuttle-unzeroing), 2026-05-03

A per-class shuttle-zeroing audit during the capacity work showed shuttle loss concentrating in the high-arc classes (long_service 24.7%, smash 13.7%, clear 11.9%, lob 9.1%, wrist_smash 8.5%, drop 8.1% whole-clip means, against sub-2% for the rest), driven by the bird leaving the top of the broadcast frame on high setups. Within that family, smash/ws is the only bottleneck pair, so it's the only one the model leans on shuttle to disambiguate; the high-F1 high-arc classes ride pose alone. The collation was wiping `shuttle[failed,:] = 0` whenever any pose slot failed, an asymmetry that never ran the other way. I removed the line, nothing else (#23). vs #17: macro +0.5, min +1.2, acc +0.4, the project-best mean at the time. Per-class shifts (mean, pp): passive_drop +3.2, smash +1.5, wrist_smash +1.2; the rest within ~1. **Both bottleneck-pair members lifted at once, the first single intervention to do that without trading them off.** My earlier "data quality isn't the bottleneck" read was right about keypoints but missed the shuttle-side asymmetry. (See [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md) for the bug angle.)

### Mask-channel variant 2a (`shuttle_missing`), 2026-05-03

TrackNet `Visibility=0` saved as `shuttle_missing.npy` and fused post-TCN via `mask_proj` (`Linear(1,4)`) + `shuttle_fuse` (`Linear(d_model+4, d_model)`), on top of wipe_drop (#24). vs #23: macro -0.4, min -1.7. Likely two compounding causes: the model was already inferring missing-shuttle from xy + temporal context (the mask is mostly redundant), and the new `shuttle_fuse` layer spends budget on a near-identity solution. **The mask channel added nothing on top of just unzeroing the shuttle, so variant 2a is parked** (variant 2b, `pose_missing_either_slot`, was never built). Per-class shifts vs #23 (mean, pp): rush -2.7, wrist_smash -1.7, passive_drop -1.4; the rest flat.

### Augmentation set locked, 2026-05-04

The active set: centreline flip (p=0.5, coupled, COCO bilateral joint-index swap) + corrected pos+shuttle constrained jitter (p=0.2 nominal, +/-0.05y / +/-0.10x cap, layered conditional bounds, joints/bones untouched, zero-frame preservation, shuttle off-screen mirroring). It replaces the broken `RandomTranslation_batch` (joints-only, decoupled, body-deforming). Out for Task 2: temporal speed jitter, Gaussian joint jitter, random joint masking, `WeightedRandomSampler`, net flip. Hit-frame metadata is derivable without re-extraction (Method A CSV correlation; Method B shuttle horizontal-velocity sign reversals). Full spec in `augmentation_framework.md`.

### Jitter-off ablation, 2026-05-04

Was the inherited bbox-centric jitter (joints shifted in their bbox-centre frame, body-deforming, not court-aligned) net-negative or just noise? A single A/B vs #23 with `RandomTranslation_batch(prob=0.0)` (#25). vs #23: macro -0.8, min -4.4, and min lands below the CDB-F1 baseline too, not just below wipe_drop. Wrist_smash takes the brunt (0.4742 to 0.4301 mean; S4/S5 floored at 0.39/0.36). Per-class shifts vs #23 (mean, pp): wrist_smash -4.4, passive_drop -2.1, short_service -1.5; the rest within ~1pp either way. **The bbox-centric jitter is conceptually wrong but empirically regularising, not even close to net-negative.** Defaults restored; the corrected pos+shuttle jitter is the replacement.

### Aug v1 first run, 2026-05-05

Augmentation v1 lands: coupled centreline flip (p=0.5) + corrected pos+shuttle jitter (p=0.2, cap_y=0.05, cap_x=0.10, eps=0.15), JnB_bone only, otherwise identical to #23 (#26). vs #23: macro -0.9, min ~flat, head metrics slipped while min held. S5 is the first project serial where smash takes over as the floor class (smash 0.503, ws 0.518), the CDB-F1 inversion the loss was built for, now in a real serial. Per-class shifts vs #23 (mean, pp): cross_court_net_shot -5.7, smash -3.0, push -2.1; the rest flat to mildly up. **Aug at p_jitter=0.2 is under-regularising on the head metrics**, and cross_court_net_shot down 5.7% is the standout, the candidate concern being that the flip washes out a side-of-court signal. (A jitter bounds bug surfaced here, commit `2291ad8`, fixed before the retune; see [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md).)

### Aug v1 + p_jitter=0.3 retune, 2026-05-05

A single-knob change vs #26, p_jitter 0.2 to 0.3, on top of the bounds fix (#27). vs #26: macro +0.6, min +0.3, acc +0.4. vs #23: macro -0.3, min +0.4, acc -0.2, top-2 +0.4. The under-reg read was right: lifting p_jitter recovered the head-metrics slip without giving back min, so aug now sits at parity-or-better than wipe_drop on head metrics and ahead on min and top-2. S5 is again the picked serial (smash 0.515, ws 0.519), the second run running with smash as the floor, so the inversion is replicating, not a fluke. Cross_court_net_shot is still down ~5% and unmoved by the p_jitter bump, so it's flip-mediated, not on the rate axis. **New project-best aug config, locked as the active baseline.**

### Aug v1 round 1 hparam sweep, 2026-05-06

Four runs via the new `hparam_sweep.py` wrapper against #27, testing whether lower p_flip recovers cross_court_net_shot, whether bigger jitter caps pay, and whether p_jitter past 0.3 pays:

| Cell | Aug | macro | min | Verdict |
|---|---|---|---|---|
| p_flip_25 (#28) | flip 0.25, jit 0.3 | 0.7402 | 0.4783 | TIE |
| cap_bump (#29) | flip 0.5, jit 0.3, cap 0.075/0.15 | 0.7339 | 0.4587 | LOSE (killed S4) |
| p_jitter_40 (#30) | flip 0.5, jit 0.4 | 0.7426 | 0.4822 | TIE |
| p_flip_25_x_p_jitter_30 (#31) | flip 0.25, jit 0.3 (replicate) | 0.7389 | 0.4569 | LOSE |

**Nothing dislodged #27.** Two findings worth keeping. First, a seed-noise envelope: a YAML slip made #31 an identical-config replicate of #28, and across the two, macro spread 0.13%, min-F1 2.14%, acc 0.14%, top-2 0.02%, so macro is reliable at this sample size but min-F1 is too noisy to drive decisions, and the 0.7% macro kill threshold sits at one run-mean spread. Second, the smash/ws split flattened: picked serials this round were ws 0.510/0.510/0.523, smash 0.567/0.605/0.568, the historic ws-below-smash gap gone, which one is the floor now varying by serial. Pose-only signal has run dry on the pair. The best single-serial of the round was #31 S1 (macro 0.7447, min 0.5231 ws), a single-serial min high, useful as deployable weights but not a hparam signal.

### Series G: multi-taxonomy baseline batch, 2026-05-30

One clean 5-serial run per taxonomy on `taxon_pinned_w_preds` and the standard hp baseline, to establish the per-taxonomy reference on the final data; this is the batch I handed the FE team. Per-taxonomy bests are in the overview's Project-bests table. **The cross-cutting reads line up with everything the earlier runs had been pointing at:**

- **split_v2 is 3-4% harder on test than the baseline split** at identical hp. bst_24 v2 (#33) posts val 0.848 but test 0.809 (3.9% gap); the same taxonomy on baseline (#36) sits at val 0.832 / test 0.827 (0.6%). split_v2's player-overlap-minimised test partition is the stricter cross-player signal.
- **The no-sides collapse is a free but minimal boost.** bst_12 v2 (#34) edges bst_24 v2 (#33) by ~0.5-0.7%, so the model was already sharing top/bottom templates and collapsing them just stops splitting thin support.
- **Dropping unknown lifts the real strokes** (see the caveat): bst_24 drop-unknown (#36) edges bst_25 keep-unknown (#35) because the services stop bleeding into the catch-all, Top_long_service recovering ~11pp.
- **The une_v1_14 extra splits are expensive on their parents.** Pulling wrist_smash off smash drops smash 0.87 to 0.61 (wrist_smash itself 0.47, the run min); pulling passive_drop off drop drops drop 0.83 to 0.66. That confusion is most of the step down from bst_12 (#34) to une_v1_14 (#37).
- **shuttleset_18 (#32) is the lowest-macro cut** (0.665 best), and its min-F1 is `driven_flight`, a single test clip (42 train / 9 val / 1 test): it memorises train (F1 0.857) but never holds val, so the 0/1 test F1 is a coin-flip and oversampling can't rescue it, it just repeats the same ~40 clips.

Runs `run_20260530_161525` (shuttleset_18) through `run_20260531_005535` (une_v1_14).

### Series H: val-gate + focal-alpha-revert across taxa, 2026-05-31

The `focal_alpha_revert_overallocated` ablation (val-improvability gate on) across the same six cells as Series G, interleaved with Series I as one batch. The mechanism walks CDB-F1's alpha back on classes it over-allocates to with no F1 return. It barely moves most taxa and pulls single-best macro/min down a touch; the one place it helps is its target, une_v1_14 v2, where it lifts mean min-F1 +0.016. The lift lands on min, not macro: it plausibly gives wrist_smash room to stabilise instead of being hammered by successive large-magnitude alpha-weighted updates, and the tightened mean-vs-best spread says it mostly regularises seed variance. **Worth keeping only as a CDB-F1 regulariser where a class consistently drains gradient weight to no benefit, and even there gradient clipnorm might get the same for far less machinery.** Runs `run_20260531_163906`+.

### Series I: weight-decay sweep on une_v1_14, 2026-05-31

AdamW weight-decay from 1e-2 to 4e-1 on une_v1_14 v2, val-gate off, norms / bias / embeddings excluded from decay across the sweep. The standard baseline applies wd 1e-2 to all layers, so the 1e-2 cell isolates the exclusion. It's a clean regularisation story: as wd rises, macro drifts down a hair and min-F1 drifts up, the trade an F1 sweep should show. Excluding norms/bias/embeddings at 1e-2 (#40) is already a new une_v1_14 high (mean macro/min +0.003 / +0.014 over the all-layers baseline #37) and edges that reference on both. **wd 4e-1 (#47) is the biggest min lift (best +0.04, mean +0.027) and the first config to clear 0.5 on wrist_smash (best serial 0.525, mean 0.498).** The exclusion is folded into the shared baseline. Runs `run_20260531_201350` (1e-2 excl) through `run_20260601_021234` (4e-1).

### Series J: wd-endpoints and focal-alpha-revert across taxa, 2026-06-02

Two arms in one 16-cell batch on the pinned collation, both holding norms/bias/embeddings out of decay. **SET 1 (gate off): the two wd endpoints (1e-2, 4e-1) across the five non-une taxon/splits**, asking whether Series I's une floor gain carries. It does, selectively: rank the readable combos by Series G mean min-F1 and wd 4e-1 lifts the three lowest (une_v1_14 0.471, bst_24/v2 0.534, bst_24/baseline 0.568) and nothing for the two highest (bst_12 0.599, bst_25 0.620). bst_24 gains +5.4% (v2) and +6.3% (baseline) mean min-F1 with a touch of macro; bst_12 already had the lift at the 1e-2 exclusion (4e-1 trims macro without buying floor); bst_25 keep-unknown sits below the all-layers standard at both endpoints. Macro never moves beyond the seed spread. The split cuts across sidedness and class count (une and bst_12 are both no-sides drop-unknown, opposite sides), so it's a starved-floor effect, not a taxonomy-shape one. **wd 4e-1 with the exclusion is adopted as the default optimiser setting: best-or-near-best min-F1 everywhere, at most ~0.9% below the per-combo best (bst_12), so one setting holds without per-taxonomy tuning.** Runs #49-58 (`run_20260601_141054` through `run_20260602_044754`).

**SET 2 (gate on): the same endpoints crossed with `focal_alpha_revert_overallocated`** on bst_25/baseline, une_v1_14/v2 and shuttleset_18/v2, the Series H arm now at the wd endpoints. It never produces the best config for any of the three. The only lift is bst_25 at 1e-2, where it recovers what the bare exclusion gave up, still under the standard; on une, plain wd 4e-1 (gate off, #47) keeps the better floor (mean min 0.498 vs 0.482). **Second batch after Series H to show focal-alpha-revert earns nothing, so it's retired.** Runs #59-64 (`run_20260602_063203` through `run_20260602_143618`). Per-cell figures in the ledger (Series I and J) and the run notes; trimmed summary + plots at `final_sweep_executive_summary.md`.

## Parked and future work

Un-run experiments, then unbuilt features, then the one refactor on deck.

### Experiment ideas not yet run

#### BST attention head geometry (Q5)

`bst.py:145` defaults to `d_model=100`, `d_head=128`, `n_head=6`. The model concatenates across heads to `d_head * n_head = 768`, then projects back down to 100; the temporal and interactional transformers follow the same expand-then-contract pattern. It comes from BST to TemPose to AcT, whose progressive-widening ablations argued a small d_model keeps the bulk of the network cheap while a wide per-head projection gives each head capacity for a distinct specialised view. Nobody has swept it on BST. Worth a pass over `d_head in {32, 64, 96, 128}`, either holding `n_head=6` (which shrinks the model) or holding `d_head * n_head` constant (which tests whether the expansion matters or just the width); if a smaller d_head holds F1, that's a free parameter-efficiency win. Caveat: d_model couples tightly across the TCN, cross-transformer, interactional transformer and PPF, so hold d_model=100 fixed and vary only d_head/n_head. Capacity Run 2 hits this axis from a different angle.

#### TCN dilation pattern

The TCN before the temporal transformer (`tcn_pose`, `tcn_shuttle`) is two stacked dilated 1D convs, kernel 5, dilations 1 then 3 (`tempose.py:139`). The per-token receptive field by the second layer is `5 + (5-1) x 3 = 17` frames, ~570ms at 30fps, covering a full swing build-up plus contact plus follow-through. The hypothesis: 17 frames is wider than the TCN needs (its job is local-motif extraction, the temporal transformer handles long-range), and the wide pre-pool may smooth over the frame-by-frame micro-motion that distinguishes smash from wrist_smash. A 2-cell A/B would test it: kernel=5 dilation off (RF 9 frames, isolates the dilation question) versus kernel=3 dilation retained (RF 9 frames the other way, tests kernel versus dilation). Neither breaks anything mechanically (output channels and token count unchanged), the risk is empirical. Slot in after Capacity Run 2 if it leaves room, ~3-5h each on A100. Detail in `hparams_sweep_speculations.md`.

#### Swap val and test (cross-player generalisation reframe)

From the player-overlap analysis (`class_player_split_overlap_exploration.md`): val and test in split_v2 have very different overlap profiles with train. Train-val sits at 55% clip-weighted overlap (val isn't held out by player), train-test at 15% (test mostly is). The split was designed that way, val for early-stop, test for unbiased eval, so the model gets early-stopped on a partly within-player signal and then reported on a stricter cross-player one. The cheap experiment for a generalisation-focused model: flip the roles, early-stop on the current test set, report on the current val set, no re-splitting, one run. I'd expect early-stop to fire later, the best checkpoint to end up tuned for cross-player generalisation, and the headline to climb purely because reporting moves to a looser distribution. Two reasons not to now: it breaks comparability with everything in the ledger; and val is much smaller per-class (wrist_smash 331, long_service 33, smash 299), so reporting rare-class F1 on val would let one or two clips swing it several pp. Worth doing on a future generalisation-focused architecture or as a reporting addendum once the active arms close; if so, keep the headline on a fixed test set across runs and only change the early-stop driver.

### Features not yet built

#### Cross-cutting MMPose recovery routes

Two recovery routes for the residual ~0.93% extraction failures, both specced in `mmpose_heuristic/mmpose_heuristic_investigation.md`, neither built. A homography-fail X3D-S-only rescue: for clips where the court homography itself doesn't fit so no court coords are possible, a pixel-space fallback picker (largest bbox per screen-half, torso-diagonal crop sizing) could feed the X3D-S stream while BST inputs stay zeroed; it needs a new metadata flag in the extract and is parked until per-class residuals show it's worth building. And gap-fill for partial-success frames: linear interpolation of `pos` and `joints` across short MMPose detection gaps when one slot picked cleanly and the other zeroed, bounded to ~15-frame gaps, gated on endpoint-proximity, explicitly not a fallback to sticky_anchor-rejected raw bboxes (those margins are generous enough that a rejection is diagnostic of upstream failure); a new post-processing module after sticky_anchor that preserves the byte-identity chain.

#### Per-joint adaptive focal (Phase 3 / trimester 2)

Sketched in `augmentation_framework.md`: extend CDB-F1 from per-class scalar alpha to per-joint x per-class weighting, focusing the loss on the joints that carry each class's signal (wrist for wrist_smash, hip rotation for clear/lob). The architectural problem is that BST has no per-joint prediction head, so the decomposition needs either an auxiliary task or a Shapley-style attribution loop. Cost non-trivial, benefit speculative.

### Cleanup backlog

#### Dedup bst_x_train.py and bst_x_infer.py scaffolding

`bst_x_infer.py` and `bst_x_train.py` still each carry their own `Task` class with `get_network_architecture`, the `pose_style` + `in_dim` arithmetic, and the dataloader setup from `preparing_data.shuttleset_dataset`. The `MODELS` dict and `derive_active_classes_from_labels` already moved to `bst_x_common.py` (created for the active-class fix), so the `Task` duplication is what's left. The genuinely different parts are small: `bst_x_infer.py` does argmax-only predictions with no metrics, and its Task has a `load_weight` instead of the cache-or-train `seek_network_weights`. Two entry points is few enough that I'm leaving the rest for now; when a third arrives (a Gradio backend, ONNX export, or the fusion pipeline once X3D-S lands), the move is to lift the shared `Task` base and the dataloader helpers into `bst_x_common.py` alongside `MODELS`. A mirror TODO is pinned at the top of `bst_x_infer.py`.

## Comparability caveats

- **Taxonomy boundaries**: macro and min are computed across the active class set, so cross-taxonomy comparison is not direct. `bst_25` (and legacy `merged_25`) keep the 'unknown' class slot; `bst_24`, `bst_12` and the `une_v1_14` family drop it.
- **Drop-unknown effect**: the Series G controlled test (`bst_25` keep-unknown #35 vs `bst_24` drop-unknown #36, same baseline split, same pinned data, only the unknown handling differs) shows dropping unknown slightly *lifts* macro on the 24 real strokes (+0.35% test, +1.1% val), it doesn't lower it. The mechanism is the service classes: with unknown kept they bleed up to 14% of their truths into the catch-all (Top_long_service alone loses ~11pp), and dropping it lets them recover. The other 20 classes do marginally better with unknown kept (+0.7% as a group), and unknown is itself a high-F1 class (~0.85 test, recall 0.91 / precision 0.79), so keeping it inflates the macro *pool* even as it taxes the services. A higher `bst_25` / `merged_25` macro is therefore partly pool composition, not better recognition of the strokes that matter. Confusion matrices: [keep-unknown](confusion_bst25_baseline_keep_unknown_S1.png) shows the service bleed, [drop-unknown](confusion_bst24_baseline_drop_unknown_S5.png) shows them clean.
- **MMPose data-quality eras**: original BST extraction, then Phase 1 mixed (sticky_anchor on 1,716 hit-zone-busted clips, rest original), then Phase 2 full-clean (sticky_anchor across all 32,203 clips, landed 2026-04-29 with `run_20260429_202144` as the first full-clean training run), then the `taxon_pinned_w_preds` collation (Series G on, 2026-05-30). Pre-Phase-2 runs train on a different per-frame zeroing distribution from post-Phase-2 runs.
- **TrackNetV3 inpaint mismatch with BST published figures**: Chang's published figures used TrackNetV3 *without* inpaint; my reproduction used the inpaint variant from day one (a cleaner shuttle stream than the published baseline). So the LR retune row beats BST paper by more than the LR retune mechanism alone predicts: some of the lift is the cleaner shuttle data. All runs from `run_20260417_191851` onward use TrackNetV3-with-inpaint.

## Cross-references

- [`bst_x_overview.md`](bst_x_overview.md): the readable overview this appendix backs.
- [`bst_x_training_runs.md`](../../experiments/bst_x/bst_x_training_runs.md): the run ledger, every run's per-metric best and mean.
- [`bst_x_issues_and_bugs_squashed.md`](bst_x_issues_and_bugs_squashed.md): the unknown ghost channel and the other squashed bugs.
- `model/bst.py`: model defaults (`d_model=100, d_head=128, n_head=6`), CG/AP branches in `BST.forward`, the `CrossTransformerLayer` docstring.
- `main_on_shuttleset/bst_x_train.py`: the cosine schedule and Hyp namedtuple configuration.
- `tuning_thoughts.md`: broader HP strategy; Q4/Q5 are items it didn't cover.
- `architecture_1_bst_3dcnn_racket_extension_09_April.md`: the initial X3D-S fusion design doc.
- `mmpose_heuristic/mmpose_heuristic_investigation.md`: the full sticky_anchor design and recovery-routes.
- `model_capacity_bottleneck_question.md`: the research-grounded read on whether widening BST is the lever.
- `train_val_test_split_analysis.md`: per-run train/val/test trajectories pinning the plateau as generalisation-bound on the smash/ws pair.
- `class_f1_focal_design.md`: the CDB-F1 design (verified against ACCV 2020).
- `seesaw_f1_focal_design.md`: the pair-aware Seesaw-style alternative (verified against CVPR 2021).
- `transformer_widening_hparam_changes.md`: Capacity Run 2 implementation surface, verifications, LR notes.
- `frame_zeroing.md`: frame-zeroing redesign detail.
- [`augmentation_framework.md`](augmentation_framework.md): the locked augmentation set, code traces, hit-frame metadata derivation, seed-noise envelope.
- [`hparams_sweep_speculations.md`](hparams_sweep_speculations.md): the per-knob sweep walkthrough and the smash/ws split section.
- [`x3d_integration_macro_plan/`](x3d_integration_macro_plan/): the six-stage X3D-S integration plan.
- `historical_bst.md`: BST paper defaults preserved for reproduction.
- `class_player_split_overlap_exploration.md`: train-val 55% / val-test 15% overlap; informs the swap val/test direction.
- `Augmentation.pdf`: the PDF anchoring the augmentation decisions.
