# Arch 1: Open Research Directions

Arch 1 is the BST + X3D-S wrist crop fusion architecture. This doc tracks the live research arms and decision-making for it. Two novel contributions live here: the X3D-S fusion (primary) and the `sticky_anchor` per-slot player-identification heuristic (secondary but semi-significant in its own right — a data-quality fix that materially repaired the upstream MMPose extraction; see the TLDR section below). Everything else tweaks the inherited BST scaffolding (Q3-Q5).

## Status (2026-05-02)

- **CDB-F1 fully explored; loss-side ceiling firmly mapped (2026-05-02).** Two more follow-ups on the first CDB-F1 (`run_20260501_164658`) landed since the 2026-05-01 update. Pair-cap (`run_20260501_230252`, S4 best) capped `alpha[smash] / alpha[wrist_smash] >= 0.7` after the standard renormalisation. Smash recovered 1.2 of the 5.5 it lost in the first run, but ws gave back 5.2 of its +4 lift. Macro and accuracy stayed flat because the trade cancelled out. Gamma=2 (`run_20260502_075808`, S3 best, the focal-literature default from RetinaNet) traded 4.1 of ws for 1.3 of smash with macro -0.7 and acc -0.6 vs the first run. Same trade sign, different mechanism. Together with the earlier gamma=0 and tau=0.5 follow-ups, every CDB knob has now been run. The original tau=1 + gamma=1 combo is the floor-lift sweet spot in this neighbourhood (mean ws 0.4621, +8.7 pp vs LS=0.1 baseline); the smash drop is structural pair-confusion that scalar-per-class alpha can't resolve regardless of how its tau or gamma get tuned. No CDB run breaks the val/test plateau at 0.74-0.75 macro. Full per-class trajectory at `scratch/architecture_notes/train_val_test_split_analysis.md`.
- **Capacity-bottleneck research done (2026-05-02).** Full writeup at `scratch/architecture_notes/model_capacity_bottleneck_question.md` answering "are we at the useful parameter ceiling?" with theory plus reference-class numbers. Short version: BST at 1.85M params on 32K clips sits in the converged 1-3M zone for skeleton-AR (ST-GCN 1.22M through PoseConv3D 2.0M, MS-G3D 2.8M, ST-TR 3.07M, Hyperformer 2.6M), at the high end of per-sample density relative to peers (~81 params/sample). Famous small video AR baselines that train from scratch (X3D-M 3.76M, MoViNet-A0 3.1M) are 1.5-2x BST; flagship transformers (Video Swin-T) need ImageNet pretraining. From-scratch video-transformer literature (VideoMAE) points at pretraining as the small-data lever, not from-scratch widening. Train-test gap concentrates on smash / ws (14-18 pp); pose-distinctive classes (services, clear, net_shot) generalise within 1-2 pp at 0.95+ test F1. Plateau looks data-bound and signal-bound, not capacity-bound. Theory doesn't argue against three directions: better signal (X3D-S fusion adds information pose-2D throws away on the wrist-vs-full-smash distinction), classifier-side decoupling (Kang cRT/LWS), augmentation that perturbs player cues. Pure widening unlikely to break the plateau; expected gain on test macro is 0-2 pp.
- **Capacity-bump ablations next (for fun, not because the prior favours them).** Two cheap runs while the compute is around. (a) `mlp_head` hidden 400 → 768: surgical, classifier-side only; ~127k → ~310k params on the head MLP. The most direct test of whether the classifier head specifically is the local bottleneck — Kang-flavoured surgery without the full two-stage. (b) `d_model` 100 → 128 with `d_head` trim 128 → 32 in the same change: pairs widening the residual stream with a Voita-style head trim so the 7.68x d_head:d_model over-provisioning doesn't propagate further. Both run combo A nosides + LS=0.0 + CDB-F1 (tau=1, gamma=1) for direct comparison vs `run_20260501_164658`.

## Status (2026-05-01)

- **LS sweep wrapped (2026-05-01)**. Three runs on combo A nosides + split_v2 + dropunk against the LS=0.1 baseline `run_20260430_170325`. LS=0.0 (`run_20260430_213933`, S2 best by wrist_smash 0.404) landed mean macro 0.743 / min 0.359 / acc 0.768 / top-2 0.939; vs LS=0.1 baseline mean (0.742 / 0.375 / 0.767 / 0.938) the head metrics are flat (+0.001) and mean wrist_smash drops 1.6 pp. Hypothesis "LS=0.1 was taxing rare-class confidence" disproved. LS=0.15 (`run_20260501_073430`, S3 best by wrist_smash 0.448) landed mean 0.747 / 0.417 / 0.769 / 0.938: macro +0.005, **min +4.2 pp**, head metrics flat. Wrist_smash range tightens 0.159 → 0.066 and the entire distribution shifts above the LS=0.1 mean. **LS=0.15 wins this sweep and is kept active for downstream runs.** LS=0.05 skipped (two bracketing data points enough); LS=0.2 deferred behind the class-weighting + focal experiments.
- **Class-weighting smoke test landed (2026-05-01)**. `run_20260501_110525`: combo A nosides + LS=0.15 + `class_weights={'wrist_smash': 2.0, 'smash': 2.0}`, S2 best by wrist_smash. Mean macro 0.748 / min 0.422 / acc 0.770 / top-2 0.936. Vs LS=0.15 alone mean (0.747 / 0.417 / 0.769 / 0.938): pair-balanced reweighting added **essentially zero on the central tendency** (+0.001 macro / +0.005 min / +0.001 acc / -0.002 top-2; smash mean +0.005 also flat). But the upper end of the achievable wrist_smash distribution shifted clearly: **S2 wrist_smash 0.518 is a new project-wide ceiling** across all 25 nosides serials run to date (prior best 0.46 LS=0.1 S5; first nosides serial to clear 0.50). S4 of the same run hits new ceilings on macro (0.756, +0.3 pp), accuracy (0.777, +0.4 pp), and drive F1 (0.66, +2 pp). The wrist_smash range widened 0.066 → 0.143; bimodal seed distribution where one seed found a wrist_smash basin no prior nosides serial accessed, three other seeds stayed in the LS=0.1-baseline range around 0.37-0.40. Read as: static loss reweighting moves the *ceiling*, not the *mean*; the loss-side axis is not exhausted. `bst_train.py` `class_weights` Hyp field + renormalised CE branch at `bst_train.py:301-` shipped with the run.
- **Unknown ghost channel removed (2026-05-01)**. BST head dim now matches the empirically present classes in train `labels.npy` (val and test asserted as subsets), derived at first serial via `bst_common.derive_active_classes_from_labels`. Architecture is a function of the data, not a flag; no `drop_unknown`-driven guard. Manifest `extra.arch` records `n_classes_full`, `n_active_classes`, `has_unknown`, `unknown_first`, `active_class_list`. Pre-fix runs (LS runs `run_20260430_170325`, `run_20260430_213933`, `run_20260501_073430`, plus class-weighted `run_20260501_110525`) carry a 15-channel head with the unknown slot as a ghost output channel; post-fix, dropunk runs on `une_merge_v1`, `une_merge_v1_nosides`, and `raw_35` collapse to 28- / 14- / 34-class heads (their MERGE_MAP redirects nothing to unknown, so the slot is empty after the writer's row drop). `merged_25` dropunk keeps its 25-class head because `MERGE_MAP['driven_flight']='unknown'` actively populates that slot at writer time. Pre-fix v1/nosides/raw_35 weights are no longer mechanically resumable post-fix (`load_state_dict` shape mismatch); inference on those weights still works if the caller passes `n_active_classes=taxonomy.n_classes, active_class_list=taxonomy.class_list()` explicitly. Comparisons against pre-fix v1/nosides/raw_35 runs carry an architectural-era boundary caveat; merged_25 dropunk runs are directly comparable across the boundary because the head dim is unchanged there. Resume now auto-backs up `manifest.yaml` to `manifest.yaml.<timestamp>.bak` next to the live file before the serial-1 rewrite, so the original is always recoverable. End-to-end smokes `run_20260501_151131` (nosides dropunk, 14-class head, no unknown entry) and `run_20260501_152835` (merged_25 dropunk, 25-class head matching `run_20260429_202144`) both passed: `[arch]` printout in tee'd log, manifest `extra.arch` populated, class-weighted CE printout right size, per_class_f1 block right shape.
- **Loss-side decision: skip basic focal, jump straight to class-F1-driven adaptive focal (CDB-F1)**. Previous plan had vanilla / manually-alpha focal as the gated next step. The class-weighting smoke result revoked that gating: static reweighting hit the central-tendency ceiling that vanilla / manually-alpha focal would also hit (same lever, just per-sample-gated). The CDB-F1 design (per-class alpha = `(1 - F1_c)^τ` driven by running EMA of train F1, optionally composed with focal `(1-p_t)^γ` per-sample focusing) is structurally the right next escalation precisely because it targets the bimodal failure mode this run revealed: low-F1 classes get persistently escalated weight, which can push bad seeds toward the wrist_smash basin S2 found. Design verified against the ACCV 2020 paper directly at `scratch/architecture_notes/class_f1_focal_design.md` (~700 lines, 8-section format, plain-English plus equations). Companion design for the pair-aware Seesaw-loss-style alternative (verified against the CVPR 2021 paper) at `scratch/architecture_notes/seesaw_f1_focal_design.md` — held as the targeted second arm if CDB-F1 lifts wrist_smash but at smash's expense. CDB-F1 implementation slated next.
- **First CDB-F1 run + two follow-ups landed (2026-05-01)**. First run `run_20260501_164658` (combo A nosides + LS=0.0 + `adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`): best S2, mean macro 0.7432, min 0.4621, acc 0.7617, top-2 0.9351. Vs class-weighted `run_20260501_110525` mean: macro -0.5, **min +4.0**, acc -0.8, top-2 -0.1. Vs LS=0.1 baseline: macro +0.1, **min +8.7**, acc -0.6. Largest floor lift on wrist_smash so far in any loss-side run; range tightens to 0.413-0.486 (vs class-weighted 0.378-0.518), so the bimodal-seed problem is solved. Project ceiling not broken: S2 0.486 < class-weighted S2 0.518. Per-class shifts vs class-weighted: ws +4.0, **push +6.7** (adaptive picked it up as a second bottleneck the static config missed), smash -5.5 (pair-confusion with ws), rush -2.7, drive -1.4, passive_drop -1.4, drop -1.1, lob -1.0, services -0.9 to -1.4. Implementation shipped: new module `src/bst_refactor/stroke_classification/main_on_shuttleset/loss/adaptive_focal.py` (~190 lines, `AdaptiveFocalLoss` + `per_class_f1_from_counts` + `accumulate_class_counts`); 6 edits in `bst_train.py` (Hyp.adaptive_focal field, mutex-checked loss branch, TP/FP/FN accumulator in `train_one_epoch`, end-of-epoch alpha update, per-class TB scalars, console alpha summary); 36 unit tests at `tests/test_adaptive_focal.py`. With `adaptive_focal=None` the legacy class_weights / LS path is bit-identical. Best weight at `experiments/run_20260501_164658/weights/...nosides_2.pt`. Two follow-ups ran in parallel on bourbaki + engelbart to triangulate the cost: gamma=0 (`run_20260501_192113`, S1 best) and tau=0.5 (`run_20260501_192519`, S2 best). Both lose 2.8-5.0 of the ws lift, smash recovers in both, macro doesn't move much. So the wrist_smash gain in the first run came from both the aggressive tau=1 alpha and the gamma=1 modulator together; soften either knob and ws drops back. The smash pair-confusion is a separate residual that retuning the alpha mechanism can't fix. See pair-cap and gamma=2 follow-ups in the 2026-05-02 status block above.

## Status (2026-04-29)

- **Phase 2 raw extract done**: full 32,203-stem unified raw dir at `/scratch/comp320a/ShuttleSet_keypoints_raw/` on both bourbaki and engelbart (bit-identical). Composed of 30,487 freshly re-extracted (across two shards over ~20h wall) plus the 1,716 Phase-1 backfill rsynced in. Verification clean: file counts match, cross-node `rsync --checksum` empty, failsafe byte-identity gate 50/50 on the 1,716 overlap with max abs diff 0.000e+00. Per-frame `ndet` baseline at `src/bst_refactor/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md` (0% `ndet=0`, 0.53% `ndet=1` floor). Next: `apply_heuristic.py --heuristic sticky_anchor` over the unified raw → `_keypoints_clean/`, then `validate_zeroed_frames.py` for Phase-1 vs Phase-2 comparison, then collate + flip `BST_MMPOSE_NPY_DIR` + sanity-train. Phase-2 motivation now is whether removing the heuristic-frame-drop bug across the full training set lifts the `Top_wrist_smash` floor that Phase-1 mixed retrain failed to lift.
- **Phase 2 sticky_anchor + zeroed-frame audit done (2026-04-29)**: clean dir at `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`, 32,203 stems × 3 files, byte-identical bourbaki/engelbart. Three `validate_zeroed_frames.py` reports landed (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`); identical underlying numbers, only the bucketing differs. Phase-1 vs Phase-2 comparison written up at `scratch/architecture_notes/mmpose_heuristic/phase1_vs_phase2_2026-04-29.md`. Headlines: overall fail rate 5.38% → 0.93%, hit-zone near-hit fail 5.98% → 0.58% (the near/away gradient sign flipped: hit zone is now the cleanest zone instead of the noisiest), per-stroke ratios 19x-76x on the strokes Phase-1 was failing hardest on (smash, clear, drop, return_net, wrist_smash). The 17 residual 100%-hit-zone-zeroed clips look like irreducibly broken broadcasts (off-frame players, replay overlays). Next gate: collate + sanity-train against the new clean dir.
- **Phase 2 collation + env flip done (2026-04-29)**: three collated trees written under `/scratch/comp320a/ShuttleSet_data_<tax>/npy_<tax>_<split>_dropunk/`, one per active (taxonomy, split) combo (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`). All run via `prepare_train_on_shuttleset.py --skip-trajectory --skip-pose --clip-npy-dir /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor`, mirrored to bourbaki, byte-identical cross-node (`rsync -avn --delete --checksum` empty on all three). Per-split clip counts cross-verified against `clips_master.csv` filtered for `--drop-unknown` via `src/bst_refactor/validation_scripts/verify_collated_counts.py` — all `OK`. `BST_MMPOSE_NPY_DIR` in `~/badminton_stroke_classifier/.env` flipped from the legacy `ShuttleSet_data_merged_25/dataset_npy_..._flat` to `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor`; one-step rollback at `.env.bak.2026-04-29`. Next gate: sanity-train BST baseline on the new collated tree (`une_merge_v1_nosides + split_v2`) and compare against the V4 baseline numbers. The decision gate is whether `Top_wrist_smash` clears the V4 floor that Phase-1 mixed retrain failed to clear.
- **Phase 2 sanity-train wrapped (2026-04-30)**: three full 5-serial runs across the active combos. Combo C (`merged_25 + split_bst_baseline`, `run_20260429_202144`, S2 best) lands mean macro 0.831 / min 0.577 / acc 0.848 / top-2 0.969, essentially tied with the Phase-1 BST baseline `run_20260418_151139` on macro / acc / top-2 (-0.022 on min) but with seed variance ~2.5x tighter on macro and accuracy. Combo B (`une_merge_v1 + split_v2`, `run_20260430_110101`, S4 best) lands 0.739 / 0.317 / 0.766 / 0.938; macro / acc / top-2 hold within noise of V4 baseline (`run_20260420_171101`) but **min drops 7 pp** because Top_wrist_smash specifically gets worse with cleaner pose data. Combo A (`une_merge_v1_nosides + split_v2`, `run_20260430_170325`, S4 best) lands 0.742 / 0.375 / 0.767 / 0.938, recovering most of combo B's wrist_smash floor via the structural side-collapse (+0.058 min vs combo B); essentially tied with the Phase-1 collapsed-classes ablation (`run_20260425_185421`) on macro / acc / top-2. Together these runs show cleaner pose data lifts head metrics and tightens seed variance on common classes but hurts the small-support tail in the un-pooled 28-class taxonomy. **Diagnosis is now classifier-side, not data-side**; three pre-flight scripts pin this down (see next bullet).
- **Pre-flight scripts: shuttle-missing diagnosis verified (2026-04-30)**: three new scripts under `src/bst_refactor/validation_scripts/`. `shuttle_gap_y_distribution.py` confirms the off-screen-high hypothesis at the sensor level: 61.6% of gap boundaries cluster in the top 10% of the broadcast frame, 72.3% on the post-gap re-appearance side. `shuttle_gap_length_distribution.py` shows the inpaint module isn't being exceeded (only 1 gap >60 frames in 32k clips); 85% of missing-shuttle frames sit in the 11-60 frame band of "shuttle genuinely not in any pixel". `perclass_shuttle_miss_vs_f1.py` against the combo A nosides manifest returns Pearson **+0.516** (Spearman +0.415), opposite of the predicted direction. The high-shuttle-miss classes are the pose-distinctive serves / clears / lobs at F1 ~0.95-0.99; the bottleneck classes (wrist_smash, drive, push, cross_court_net_shot) sit at sub-1% miss rates and have shuttle data available. Combined diagnosis: shuttle data is reliably present where it's most needed; the model just isn't using it well. The mask-channel arm gets demoted; trajectory extrapolation flagged as a longer-term direction for the off-screen-arc gaps. **Label smoothing is now the highest-priority loss-side experiment**; full search-space analysis at `scratch/architecture_notes/hparams_sweep_speculations.md`.
  - **Don't delete the legacy `_merged_25` nested tree yet.** It's the only path to bit-exactly reproduce the V4 / Phase-1 baseline. The new extract should be a strict improvement (drop-bug fixed, sticky_anchor over the full set instead of 1,716), so the realistic risk is just losing the historical baseline number, not losing useful results. Keep until the Phase-2 sanity-train numbers are in and the writeup commits to a baseline.
  - **Unknown class still has no pose data.** The 1,278 `raw_type_en == 'unknown'` clips were excluded from the Phase-2 extract because every active taxonomy uses `--drop-unknown`. If we ever want them (noise / distractor class for a robustness ablation), extract to a sibling `/scratch/comp320a/ShuttleSet_keypoints_raw_unknown/` so the garbage bucket can never accidentally enter canonical training via a permissive glob. ~17 h single-process; not blocking.

## Status (2026-04-25)

- **BST LR-schedule retune (Q4)**: done. Compressed schedule beats the paper on every test metric (macro F1, min F1, accuracy, top-2). Active settings in `bst_train.py`. Numbers in "LR schedule retune" below.
- **CG/AP annealing (Q3)**: done. Three matched 5-serial runs. Annealed-out best (mean macro F1 0.829), always-on close behind, always-off trails. Annealed kept as the active config.
- **Attention head geometry sweep (Q5)**: open, not started. Secondary priority.
- **X3D-S racket crop fusion**: model + input shape decided; fusion depth, training schedule, temporal cut-in, and MMPose-drop handling all open. Primary research direction; build slated for late next week. *Wiring note for the active-class fix (2026-05-01)*: when adding the X3D-S branch (or any new training script), source the head dim from `task.n_active_classes` and run `_validate_and_record_arch` on serial 1, mirroring the bst_train.py pattern. Hardcoding `taxonomy.n_classes` in the fusion module would put the unknown ghost back; the existing `Task.get_network_architecture` shows the correct pattern.
- **Label smoothing sweep**: **closed on combo A nosides as of 2026-05-01**. Three runs; LS=0.15 won (mean wrist_smash 0.417, +4.2 pp vs LS=0.1 baseline mean 0.375; head metrics flat; range tightens 0.159 → 0.066). LS=0.15 kept active for downstream runs. LS=0.0 disproved the rare-class-tax hypothesis. LS=0.05 skipped (two bracketing data points enough); LS=0.2 deferred. See the 2026-05-01 status block above for full numbers.
- **Class weighting smoke test**: **landed (2026-05-01)** as `run_20260501_110525`. Pair-balanced `{'wrist_smash': 2.0, 'smash': 2.0}` on top of LS=0.15. Mean shift past LS=0.15 alone was flat (+0.005 wrist_smash, inside seed variance), but S2 set a new project-wide wrist_smash ceiling at 0.518 (prior best 0.46); S4 set new ceilings on macro 0.756 / acc 0.777 / drive 0.66. Bimodal seed distribution: one seed found a wrist_smash basin no prior nosides serial accessed. Read as: static reweighting moves the ceiling, not the mean.
- **CDB-F1 ablation**: **fully explored as of 2026-05-02**. First run `run_20260501_164658` (tau=1, gamma=1) lifted mean wrist_smash +4.0 on class-weighted and +8.7 on the LS=0.1 baseline (the largest floor lift on any single loss-side run); push +6.7 picked up automatically; bimodal-seed problem solved (range tightens 0.140 → 0.073). Cost: smash -5.5 (pair-confusion), small symmetric drops on rush / drive / drop / services. Four follow-ups (gamma=0 `run_20260501_192113`, tau=0.5 `run_20260501_192519`, pair-cap `run_20260501_230252`, gamma=2 `run_20260502_075808`) all traded ws back for smash without macro moving. The first run is the floor-lift sweet spot in this neighbourhood. Smash pair-confusion is structural and scalar-per-class alpha can't resolve it. No CDB run breaks the val/test plateau at 0.74-0.75 macro.
- **Capacity-bottleneck research**: **done (2026-05-02)**. Writeup at `scratch/architecture_notes/model_capacity_bottleneck_question.md` argues the plateau is data-bound and signal-bound rather than capacity-bound. BST at 1.85M params on 32K clips sits in the converged 1-3M zone for skeleton-AR; the from-scratch video AR baselines that train without pretraining (X3D-M, MoViNet-A0) are 1.5-2x BST. Pure widening expected 0-2 pp on test macro. See 2026-05-02 status block.
- **Data augmentation**: probable intermediate step after the loss-side knobs settle. Particulars TBD; horizontal-flip-with-COCO-swap may join the active set if combo B's LS+flip joint sweep validates it (see hparams_sweep_speculations.md runbook step 1a).
- **MMPose extraction quality**: Phase 1 sticky_anchor heuristic shipped (95.05% of 1,716 busted clips perfectly clean). Phase 1 mixed retrain (`run_20260425_150548`) failed the decision gate on `Top_wrist_smash` (-0.057 mean) while macro/acc/top-2 lifted ~0.007 each. The per-class frame-zeroing audit then showed the F1-bottom classes aren't the heavily-zeroed ones; the data-quality-bottleneck hypothesis is empirically dead. Phase 2 deprioritised but not killed: the decoupled `raw_extract` is faster per clip than the original pipeline, so re-running ~31k clips is more affordable than the original ~50 hr V100 estimate. Full state in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.
- **Collapsed-classes ablation** (`run_20260425_185421`): 28 classes to 14, dropping the Top_/Bottom_ side prefix on the sticky_anchor data. Rare-class seed variance halved, but absolute metrics within noise of V4 baseline -- doubled per-class N stabilised wrist_smash without lifting it. Next step before bolting on the 3dcnn is augmentation (per Isiah's writeup) and class weighting / focal loss. Full writeup under "Completed experiments".
- **bst_train / bst_infer dedup**: deferred until a third entry point arrives.

## MMPose extraction context (sticky_anchor TLDR)

The BST original zeroed an entire frame whenever a player's ankle midpoint projected outside the soft court rectangle (`eps = 0.01`) or fewer than 2 people were detected. Airborne smashes were the worst-affected class: jump geometry pushes projected feet ~0.17-0.24 normalised units off court (Padel paper `H_z * tan(θ)`), so the model saw zeros at the most informative moment.

`sticky_anchor` replaces that filter with per-slot tracking. Each slot has an anchor at its court half-centre (75% fixed, 25% running EMA of recent picks). The closest-to-anchor detection wins; off-court picks are still output but don't update the EMA. Bottom picks first; a closer-to-own-anchor Voronoi pre-filter blocks cross-half capture; a bbox-area + sitting-pose tiebreaker handles ambiguous frames. On the 1,716 hit-zone-busted clips: 95.05% perfectly clean post-fix; residual 61 are mostly irrecoverable framings (closeup, side-on, cutaway). Phase 1 mixed retrain done (`run_20260425_150548`); decision gate failed and a per-class frame-zeroing audit then ruled out the data-quality-bottleneck hypothesis for the F1 floor (full status in the heuristic doc). Full design + decision log in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.

Methodologically this is a small novel contribution in its own right: reframes per-frame player identification from eligibility-filter ("zero the frame if either player projects off-court") to tracking-by-anchor ("each slot picks its own closest in-court candidate, with a Voronoi guard against cross-half capture"). The eligibility-filter formulation fails catastrophically on airborne strokes because the most informative frames are also the ones most likely to be filtered out; the tracking formulation keeps those frames usable.

## Core research direction: X3D-S racket crop fusion

### Model choice: X3D-S

X3D-S is the model I'm going with for the racket-crop branch. This fits two constraints:
- Easily available with weights, and
- small enough (params) to fine-tune end-to-end in the short time we have, on a v100 16gb.

There are other strong, low-param models, MoViNet for example, but none with prebuilt appropriate weights and easy model zoo integration. X3D would probably do even better with SSv2 pretraining (fine hand motions), but the SSv2 weights only exist as an unofficial TensorFlow port, and interface bugs will probably eat more time than the engineering.

Within the X3D family I picked S over XS and the larger variants:

- **vs XS.** XS expects 4 frames × stride=12, too coarse for granular badminton racket motion.
- **vs M / L / XL.** They only drop stride to 5, perform not-that-much better, and way more params.
- **X3D-S.** Strong accuracy at a low parameter count. Expected input is 13 frames × stride=6.

### Target input shape: frames=39, stride=1

I'm fine-tuning X3D-S toward `frames=39, stride=1`, not its default `13 × stride=6`.

`stride=1` gives the model access to every frame and lets it learn the interactions between them, which is what granular badminton racket motion needs. I set `39` so that by the final convolutional block the receptive field covers all input frames. That imposes a hard limit around ~40 frames, which is fine for a racket crop centred on a stroke event.

### Fusion depth: where X3D-S output enters BST (open)

Competing ideas on how deep into BST the X3D-S signal cuts in:

- **Late concat, just before the MLP head.** Easiest to implement, lowest risk, but gives BST no chance to condition its attention on the racket signal.
- **Tie into attention earlier, in a meaningful way.** X3D-S output feeds into the cross-attention or the interactional transformer, so the racket evidence shapes how players and shuttle attend to each other. More expressive, more moving parts.
- **Separate tower with learned significance weighting.** X3D-S runs as its own tower and a learned scalar (or vector) gates how much its prediction counts vs BST's. Keeps the two branches clean and lets the model decide per-sample how much to trust the racket signal.

### Open training/integration questions

Three things I still need to pin down:

1. **Fine-tuning and end-to-end schedule.** What's the right sequence for fine-tuning X3D-S on badminton video first, then co-training it end-to-end with the rest of Arch 1? Length of each phase, learning rates, what to freeze when.
2. **Temporal cut-in of X3D-S feedback.** The reported stroke racket contact times are noisy. I need to pick where the X3D-S input window sits relative to the reported contact time so the feature stays responsive to the stroke event even when the reported time is slightly off. Options: a fixed offset centred on the reported time, a learned offset, or a slightly wider window that lets X3D-S self-align.
3. **Juggling MMPose drops.** MMPose periodically drops frames, sometimes with alarming frequency for certain stroke categories. The X3D-S window has to cope with that. The aggressive frame-zeroing concern is now addressed at the extraction layer by sticky_anchor (see the heuristic doc); the residual drops that survive sticky_anchor are detection-layer (heavy occlusion at the net, etc.) and the candidate fix is temporal interpolation. Worst case: pin the camera to the shuttle velocity reversal position.

## Secondary: BST attention head geometry (Q5)

`bst.py:145` defaults to `d_model=100`, `d_head=128`, `n_head=6`. The model concatenates across heads to `d_head * n_head = 768`, then `MultiHeadCrossAttention.tail` (`bst.py:59-62`) projects back down to 100. The temporal and interactional transformers in `tempose.py` follow the same pattern.

I traced the ratio back to see where it came from. BST inherits it from TemPose, which inherits it from AcT (Action Transformer). AcT ran progressive-widening ablations on exactly this expand-then-contract pattern, and I read their results as: a small `d_model` keeps the bulk of the network cheap, while the wide per-head projection gives each head enough capacity to learn a distinct specialised view. Low total parameter count, rich per-head representations.

As far as I can tell, nobody has swept this on BST. Worth a pass over `d_head ∈ {32, 64, 96, 128}`, either holding `n_head=6` (which shrinks the model) or holding `d_head * n_head` constant (which tests whether the expansion matters or just the total width). If a smaller `d_head` holds F1, we get a free parameter-efficiency win.

One caveat I've already hit: `d_model` couples tightly across TCN, cross-transformer, interactional transformer, and PPF, which I wrote up in `tuning_thoughts.md`. So I'd hold `d_model=100` fixed and only vary `d_head` / `n_head`.

## Cross-cutting (parked, see mmpose heuristic doc)

Two recovery routes for residual MMPose-extraction failures, both relevant to Arch 1's data quality but specced out in the heuristic doc rather than here.

- **Homography-fail X3D-S-only rescue (Phase 2 candidate)**. For clips where the court homography itself doesn't fit (so no court coords are possible at all). Pixel-space fallback picker (largest bbox per screen-half, torso-diagonal crop sizing per question 3 above) could feed the X3D-S stream while BST inputs stay zeroed. Needs a new metadata flag in the extract output. Parked until per-class Phase 1 residuals show whether it's worth building. Full writeup under "Homography-fail frames: crop-only recovery" in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.
- **Gap-fill for partial-success frames (could fit this trimester, else Phase 2)**. Linear interpolation of `pos` and `joints` across short MMPose detection gaps when one slot picked cleanly and the other zeroed. Bounded to ~15-frame gaps, gated on endpoint-proximity. Explicitly NOT a fallback to sticky_anchor-rejected raw bboxes; those margins are generous enough that a rejection is diagnostic of upstream failure. New post-processing module that runs after sticky_anchor and preserves the byte-identity chain. Full design under "Gap-fill post-processing (proposed, 2026-04-25)" in the heuristic doc.

## Parked: swap val and test

Followup from the player-overlap analysis (`scratch/research/class_player_split_overlap_exploration.md`): val and test in `split_v2` have very different player-overlap profiles with train. Train-val sits at 55% clip-weighted overlap (val isn't held-out by player), train-test sits at 15% (test mostly is). The split was deliberately designed that way: val for early-stop, test for unbiased evaluation. So the model as it stands gets early-stopped on a signal that's partly within-player fit, then gets reported on a stricter cross-player signal.

For a model genuinely focused on cross-player generalisation, the obvious cheap experiment is to flip the roles: early-stop on the current test set (the harder signal), report final on the current val set. No re-splitting work, single training run. Predictions: early-stop fires later, best-checkpoint weights end up tuned for cross-player generalisation rather than within-player fit, and the headline number probably climbs purely because reporting is now on a looser distribution.

Two reasons not to do it now:
- We're already a long way down the loss-side path with the current val/test convention; swapping mid-stream breaks comparability with everything in `nosides_runs_table.md`.
- Val is much smaller than test on a per-class basis. Reporting on val for the rare classes (wrist_smash 331, long_service 33, smash 299) would push the bottom-cluster headline numbers into a regime where one or two clip idiosyncrasies move the per-class F1 by several pp. The current arrangement uses the larger pool (test) for the headline, which is the right call for unbiased evaluation even if it's the wrong call for early-stop.

Worth doing on a future generalisation-focused architecture or as a reporting addendum once the active loss-side / augmentation arms close. If we do swap, headline reporting still wants to land on the same fixed test set across runs to keep comparability; only the early-stop driver should change.

## Next: capacity-bump ablations (for fun)

Loss-side experiments are settled (see 2026-05-02 status above). Capacity research at `scratch/architecture_notes/model_capacity_bottleneck_question.md` argues the plateau is data-bound and signal-bound rather than capacity-bound. Pure widening should land somewhere in 0-2 pp on test macro. That's still worth two confirmatory runs while the compute is around — and a small chance one of them surprises.

### Run 1: `mlp_head` hidden 400 → 768

Surgical, classifier-side only. The encoder is untouched; only the per-player MLP that maps the 100-dim conclusion to logits widens. Param cost on the head MLP ~127k → ~310k. The most direct test of whether the classifier head specifically is the local bottleneck — Kang-flavoured surgery without the full two-stage cRT/LWS protocol.

Config: combo A nosides + LS=0.0 + CDB-F1 (tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0); only `mlp_head` hidden width changes. Direct comparison vs `run_20260501_164658`.

### Run 2: `d_model` 100 → 128 with `d_head` trim 128 → 32

Tests the "residual width is the cap" hypothesis. `d_model` rises from 100 to 128 (+28% on the residual stream); `d_head` trims from 128 to 32 in the same change so the 7.68x d_head:d_model over-provisioning doesn't propagate further. Voita's WMT pruning result (38/48 heads removable for 0.15 BLEU drop) says the existing per-head allocation is over-provisioned; trimming d_head while widening d_model rebalances toward d_model carrying more.

The d_model bump propagates through TCN, cross-transformer, interactional transformer, and PPF; that coupling is real and the param accounting compounds. d_head trim eats some of it back. Net should still be a modest param bump.

Config: same as Run 1 plus the d_model and d_head changes. Direct comparison vs `run_20260501_164658`.

### After this

If either bump lifts macro 1+ pp without burning ws, capacity has a small lever and we follow with a joint d_model + mlp_head run on combo B for cross-taxonomy validation. If both come in inside seed noise, the capacity question is empirically answered as well as theoretically: data and signal are the bottleneck, and X3D-S fusion is the right next thing.

X3D-S remains the long-term primary direction: racket-pixel information pose-only can't see. Augmentation (horizontal-flip + COCO joint-pair swap, per Isiah's writeup at `scratch/research/Augmentation.pdf`) is the natural intermediate. Seesaw-F1 (paper-verified design at `scratch/architecture_notes/seesaw_f1_focal_design.md`) is held as a targeted second loss-side arm only if a future signal-side gain reopens the smash↔ws pair-confusion question.

## Current LR + aux schedule

Active settings (`bst_train.py:62-79` plus the cosine call at `:308-314`): `n_epochs=80`, `early_stop_n_epochs=40`, `batch_size=128`, `lr=5e-4`, `warm_up_step=100`, `num_cycles=0.5`, `use_aux_schedule=True`, `aux_fade_end_epoch=15`. Compressed warm-start-then-finetune schedule paired with the CG/AP cosine fade: ~4 epochs warmup, ~15 epochs of CG/AP warm-start tapering to 0, then ~65 epochs of pure-backbone training under cooling LR. The BST paper's defaults (`n_epochs=1600`, `warm_up_step=400`, `early_stop_n_epochs=300`, `num_cycles=0.25`, `aux_fade_end_epoch=60`) and the dated retune rationale are captured in `scratch/architecture_notes/historical_bst.md` section 3 for reproduction work.

## Completed experiments

### LR schedule retune (Q4) — 2026-04-17

`bst_train.py:308-314` calls `get_cosine_schedule_with_warmup`. The original BST recipe passed `num_cycles=0.25` alongside `n_epochs=1600`, `warm_up_step=400`, and `early_stop_n_epochs=300`. At `num_cycles=0.25` only a quarter of the cosine curve runs across the full budget, so the LR barely decays. BST-default runs converge around epoch 60 and early-stopping fires around epoch 360, so the scheduler never actually had time to lower the rate.

Compressed `n_epochs` to match the real convergence timeframe and bumped `num_cycles` so the cosine curve actually hits zero. `Apr17_13-04-35` showed best F1 macro 0.8311 at epoch 41 (out of 1600), val loss peaked by epoch 27, early-stop at 341.

Active settings (old values preserved commented in `bst_train.py`):

| param | was | now |
|---|---|---|
| `n_epochs` | 1600 | 120 |
| `warm_up_step` | 400 | 100 |
| `early_stop_n_epochs` | 300 | 40 |
| `num_cycles` | 0.25 | 0.5 |

Run `run_20260417_191851` (commit 2cb78b8), 3 serials on merged_25, test set (num_strokes 3486):

| | F1 macro | F1 min | Accuracy | Top-2 |
|---|---|---|---|---|
| BST paper (published) | 0.8097 | 0.5762 | 0.8322 | — |
| Prior best (commit 8810e95, old schedule) | 0.823 | 0.585 | 0.841 | 0.963 |
| **Retune serial 1 (winner)** | **0.830** | **0.627** | **0.844** | **0.964** |
| Retune serial 2 | 0.822 | 0.610 | 0.841 | 0.963 |
| Retune serial 3 | 0.827 | 0.585 | 0.841 | 0.963 |

All three serials beat the paper on every metric, so it's not just a lucky random seed. Huge jump on F1 min (+4.2 points vs prior best, +5.1 vs paper). Harder classes get a massive benefit.

The val-vs-test direction flipped too: the old run had val macro 0.8311 but test macro 0.823; the retune's winner had val macro 0.816 but test macro 0.830.

We might be hitting a data quality cap soon. 3% are 'unknown', a catch-all garbage class. Another 3% have known bad labels. And 25% of the majority class (smash) have serious problems with over-strict frame zeroing by mmpose, the bulk of which sticky_anchor now repairs.

Winning weight kept at `main_on_shuttleset/experiments/run_20260417_191851/weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25.pt` and tracked via an `!` override in `.gitignore`. Numbers verified from `test_logs/test_20260417_191851.log`.

### CG/AP annealing ablations (Q3) — 2026-04-19

Right now CG (Clean Gate) and AP (Aim Player) run unweighted for the whole training run; see the `use_cg`/`use_ap` branches in `bst.py`. The BST paper shows both modules improve accuracy over the bare transformer, so they're pulling real weight.

My hypothesis: their strongest role is as a **warm-start prior**. Early in training the transformers haven't yet learnt robust shuttle- or player-aware representations, so the hand-crafted CG/AP interactions could be useful inductive bias in that regime. Later, once the transformers have learnt their own (analogous, potentially richer) interactions, a fixed CG/AP contribution could start to constrain the model, pinning it to the hand-crafted formulation instead of letting it find something better. If it outgrows the heuristics without annealing, the following layers probably learned to down-tune their feedback while also imperfectly reconstructing and using the original signal that was downsampled and filtered through the CG and AP modules.

Three matched 5-serial runs under the retuned LR schedule (`n_epochs=80`, `num_cycles=0.5`, `lr=5e-4`, `warm_up_step=100`, `early_stop_n_epochs=40`, `batch_size=128`). Only the CG/AP schedule varies.

| Arm | aux_factor over epochs | Run | Mean macro F1 | Best serial (macro F1, acc, min F1) |
|---|---|---|---|---|
| Annealed out | 1.0 at ep. 1, cosine to 0.0 by ep. 15, then 0 | `run_20260418_151139` | 0.829 | S2: 0.831, 0.850, 0.600 |
| Always on | 1.0 for all 80 epochs | `run_20260418_174238` (Run A) | 0.826 | S3: 0.828, 0.844, 0.603 |
| Always off | 0.0 for all 80 epochs | `run_20260418_234822` (Run B) | 0.822 | S2: 0.830, 0.842, 0.586 |

Annealed > always-on > always-off, with small but consistent gaps. Peak performance, particularly accuracy, suggests CG and AP limit the model's top end when there are lots of samples available (likely the accuracy-macro F1 divergence). Deserves a run with precise per-class result reporting to confirm.

Broadly, CG and AP offer a demonstrably useful warm-start inductive bias. The tuned LR explains most of the difference from the original BST stats. A perfectly tuned and even slower LR might let the model naturally settle into the same minimum, but barring that, CG/AP are an objectively useful nudge in the right direction. Particularly helpful in lifting performance for minimally represented classes.

Pointers for the raw numbers: per-serial metrics in each run's `experiments/run_.../manifest.yaml` and the Serial blocks in `test_logs/test_20260418_151139.log`, `test_20260418_174238.log`, `test_20260418_234822.log`.

### Sticky_anchor mixed retrain — 2026-04-25

Reran the V4 baseline (`run_20260420_171101`) with the 1,716 hit-zone-busted clips swapped in for their sticky_anchor-cleaned versions, everything else unchanged. The decision gate from the heuristic doc wanted a +0.02 target-class min-F1 lift; this run failed it.

Mean across 5 serials, vs V4 baseline:

| | sticky mean | V4 mean | Δ |
|---|---|---|---|
| macro F1 | 0.748 | 0.741 | +0.007 |
| min F1 | 0.333 | 0.389 | -0.056 |
| accuracy | 0.774 | 0.766 | +0.008 |
| top-2 | 0.942 | 0.936 | +0.006 |

Top_wrist_smash mean dropped 0.057. Top_smash gained almost exactly what wrist_smash lost (+0.020), which fits the boundary-allocation tradeoff: cleaner data made the smash family easier and the model spent the gain on the easier head class instead of the rare tail.

Per-class frame-zeroing audit (`zeroed_frames_class_audit.py`) followed. F1-bottom classes weren't the heavily-zeroed ones, and the worst-zeroed class hit near-perfect F1. So data quality isn't the floor bottleneck. Phase 2 deprioritised on that finding (full writeup in the heuristic doc).

Run + manifest at `experiments/run_20260425_150548/`; best S3.

### Collapsed classes ablation — 2026-04-25

Same data as the run above. Only the label space changes: 28 classes to 14 by dropping the Top_/Bottom_ side prefix (new taxonomy `une_merge_v1_nosides`). Hypothesis: Top_X and Bottom_X are essentially the same shot mirrored across the net; forcing them to be separate classes halves per-class N and asks the model to learn a redundant distinction.

Mean across 5 serials:

| | nosides mean | sticky mean | V4 mean |
|---|---|---|---|
| macro F1 | 0.743 | 0.748 | 0.741 |
| min F1 | 0.397 | 0.333 | 0.389 |
| accuracy | 0.766 | 0.774 | 0.766 |
| top-2 | 0.938 | 0.942 | 0.936 |

vs V4 every metric is within ±0.008 (noise band). Absolute ceiling didn't move.

What did move was rare-class stability. Per-seed test-min range dropped from 0.124 (sticky) to 0.074 (nosides), and worst-seed min lifted from 0.235 to 0.350. The 14-class wrist_smash F1 (~0.42 mean) is close to the support-weighted mean of the old 28-class Top_wrist_smash (0.33) and Bottom_wrist_smash (0.45) -- so the model isn't actually distinguishing smash from wrist_smash any better, the metric just stopped flipping between two thin slots.

Doubled per-class N reduced the seed lottery on rare classes; absolute performance didn't change. Next step before bolting on the 3dcnn is augmentation (per Isiah's writeup at `scratch/research/Augmentation.pdf`) and class weighting / focal loss.

Run + manifest at `experiments/run_20260425_185421/`; best S1 (top min, top top-2).

## Cleanup backlog

### Dedup `bst_train.py` and `bst_infer.py` scaffolding

`bst_infer.py` and `bst_train.py` both carry their own copy of the `MODELS` dict, a `Task` class with `get_network_architecture`, the `pose_style` + `in_dim` arithmetic, and the dataloader setup from `preparing_data.shuttleset_dataset`. The genuinely different parts are small: `bst_infer.py` does argmax-only predictions with no metrics, and its Task has a `load_weight` instead of the cache-or-train `seek_network_weights`.

Two entry points is few enough that I'm leaving it for now. When a third arrives (Gradio backend, ONNX export, or the Arch 1 fusion pipeline once X3D-S lands), the right move is a `bst_common.py` holding `MODELS`, a base `Task`, and the shared dataloader helpers, with `bst_train.py` and `bst_infer.py` importing from it. A mirror TODO is pinned at the top of `bst_infer.py`.

## Cross-references

- `src/bst_refactor/stroke_classification/model/bst.py`: model defaults (`d_model=100, d_head=128, n_head=6`), CG/AP branches in `BST.forward`, and the `CrossTransformerLayer` docstring.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`: cosine schedule and Hyp namedtuple configuration.
- `scratch/architecture_notes/tuning_thoughts.md`: broader HP strategy; Q4/Q5 here are new items it didn't cover, and the X3D-S schedule open question refines a stub there.
- `scratch/architecture_notes/architecture_1_bst_3dcnn_racket_extension_09_April.md`: the initial X3D-S fusion design doc; this section refines it.
- `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`: full sticky_anchor heuristic + recovery-routes design (homography-fail, gap-fill).
- `scratch/architecture_notes/model_capacity_bottleneck_question.md`: research-grounded read on whether widening BST is the missing lever (theory + reference-class numbers; the capacity-bump runs in this doc are the empirical confirmation).
- `scratch/architecture_notes/train_val_test_split_analysis.md`: per-run train/val/test trajectories pinning the plateau as generalisation-bound on the smash↔ws confusion pair.
- `scratch/research/class_player_split_overlap_exploration.md`: train-val 55% / val-test 5% clip-weighted player overlap; informs the "swap val and test" parked direction.
