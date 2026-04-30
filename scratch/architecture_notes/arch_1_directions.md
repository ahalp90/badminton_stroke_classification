# Arch 1: Open Research Directions

Arch 1 is the BST + X3D-S wrist crop fusion architecture. This doc tracks the live research arms and decision-making for it. Two novel contributions live here: the X3D-S fusion (primary) and the `sticky_anchor` per-slot player-identification heuristic (secondary but semi-significant in its own right — a data-quality fix that materially repaired the upstream MMPose extraction; see the TLDR section below). Everything else tweaks the inherited BST scaffolding (Q3-Q5).

## Status (2026-05-01)

- **LS sweep arc closed (2026-05-01)**. Two cells run on combo A nosides + split_v2 + dropunk against the LS=0.1 baseline `run_20260430_170325`. Cell 1 LS=0.0 (`run_20260430_213933`, S2 best by wrist_smash 0.404) lands mean macro 0.743 / min 0.359 / acc 0.768 / top-2 0.939; vs LS=0.1 baseline mean (0.742 / 0.375 / 0.767 / 0.938) the head metrics are flat (+0.001) and mean wrist_smash drops 1.6 pp. Best-of-run wrist_smash drops from 0.459 (LS=0.1 S5) to 0.404 (LS=0.0 S2). Hypothesis "LS=0.1 was taxing rare-class confidence" is disproved on this taxonomy. Cell 2 LS=0.15 currently running; reads as the second data point on the LS axis with the prior expectation that more smoothing might help rare-class gradient noise. If LS=0.15 doesn't lift wrist_smash mean above ~0.40, the LS arm closes and any future work uses LS=0.1 as the baseline.
- **Class weighting smoke test prepped (2026-05-01)**. `bst_train.py` Hyp namedtuple now carries a `class_weights` field (dict[class_name, float] or None) with a renormalised CE branch in the loss build at `bst_train.py:301-`. Active config is `{'wrist_smash': 2.0, 'smash': 2.0}` for the wrist_smash-and-smash confusion-pair smoke test: tests whether loss-side reweighting alone can move the wrist_smash F1 floor without stealing recall from smash, by upweighting both classes equally so the gradient has no directional bias. Pair-balanced because asymmetric weighting (wrist_smash up alone) lets the model cheap-path by flipping the boundary toward wrist_smash predictions. Drive deliberately excluded — drive's confusion partner is different (probably clear / push) and a singleton bump there is a separate single-class bet. Run will kick after the LS=0.15 cell lands; LS revert to 0.1 is a one-character manual edit if LS=0.15 doesn't win.
- **Focal loss reframed**. Previous "next focal loss" framing assumed count-rebalancing was the bottleneck mechanism. The class-weighting smoke test is now the cleaner gate ahead of focal: if pair-balanced 2.0/2.0 weighting moves wrist_smash F1, focal becomes the natural refinement (manually-alpha focal with the same pair, gamma=1 first then gamma=2). If it doesn't move, vanilla sample-based focal is unlikely to help either, and the right next pivot is augmentation rather than a more sophisticated loss. Vanilla focal is deferred until class weighting reports. Separately, **class-F1-driven adaptive focal** (CDB-loss / Seesaw / manually-iterative variants) is queued as a research arm with its own exploration prompt at `scratch/architecture_notes/class_f1_focal_exploration_prompt.md`; the count-vs-difficulty decoupling on combo A means class-count-based alpha (the standard focal balanced variant) is structurally a poor fit, but a per-class-F1-driven alpha is a coherent fit and worth a literature-grounded design pass before implementing.

## Status (2026-04-29)

- **Phase 2 raw extract done**: full 32,203-stem unified raw dir at `/scratch/comp320a/ShuttleSet_keypoints_raw/` on both bourbaki and engelbart (bit-identical). Composed of 30,487 freshly re-extracted (across two shards over ~20h wall) plus the 1,716 Phase-1 backfill rsynced in. Verification clean: file counts match, cross-node `rsync --checksum` empty, failsafe byte-identity gate 50/50 on the 1,716 overlap with max abs diff 0.000e+00. Per-frame `ndet` baseline at `src/bst_refactor/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md` (0% `ndet=0`, 0.53% `ndet=1` floor). Next: `apply_heuristic.py --heuristic sticky_anchor` over the unified raw → `_keypoints_clean/`, then `validate_zeroed_frames.py` for Phase-1 vs Phase-2 comparison, then collate + flip `BST_MMPOSE_NPY_DIR` + sanity-train. Phase-2 motivation now is whether removing the heuristic-frame-drop bug across the full training set lifts the `Top_wrist_smash` floor that Phase-1 mixed retrain failed to lift.
- **Phase 2 sticky_anchor + zeroed-frame audit done (2026-04-29)**: clean dir at `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`, 32,203 stems × 3 files, byte-identical bourbaki/engelbart. Three `validate_zeroed_frames.py` reports landed (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`); identical underlying numbers, only the bucketing differs. Phase-1 vs Phase-2 comparison written up at `scratch/architecture_notes/mmpose_heuristic/phase1_vs_phase2_2026-04-29.md`. Headlines: overall fail rate 5.38% → 0.93%, hit-zone near-hit fail 5.98% → 0.58% (the near/away gradient sign flipped: hit zone is now the cleanest zone instead of the noisiest), per-stroke ratios 19x-76x on the strokes Phase-1 was failing hardest on (smash, clear, drop, return_net, wrist_smash). The 17 residual 100%-hit-zone-zeroed clips look like irreducibly broken broadcasts (off-frame players, replay overlays). Next gate: collate + sanity-train against the new clean dir.
- **Phase 2 collation + env flip done (2026-04-29)**: three collated trees written under `/scratch/comp320a/ShuttleSet_data_<tax>/npy_<tax>_<split>_dropunk/`, one per active (taxonomy, split) combo (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`). All run via `prepare_train_on_shuttleset.py --skip-trajectory --skip-pose --clip-npy-dir /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor`, mirrored to bourbaki, byte-identical cross-node (`rsync -avn --delete --checksum` empty on all three). Per-split clip counts cross-verified against `clips_master.csv` filtered for `--drop-unknown` via `src/bst_refactor/validation_scripts/verify_collated_counts.py` — all `OK`. `BST_MMPOSE_NPY_DIR` in `~/badminton_stroke_classifier/.env` flipped from the legacy `ShuttleSet_data_merged_25/dataset_npy_..._flat` to `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor`; one-step rollback at `.env.bak.2026-04-29`. Next gate: sanity-train BST baseline on the new collated tree (`une_merge_v1_nosides + split_v2`) and compare against the V4 baseline numbers. The decision gate is whether `Top_wrist_smash` clears the V4 floor that Phase-1 mixed retrain failed to clear.
- **Phase 2 sanity-train arc done (2026-04-30)**: three full 5-serial runs across the active combos. Combo C (`merged_25 + split_bst_baseline`, `run_20260429_202144`, S2 best) lands mean macro 0.831 / min 0.577 / acc 0.848 / top-2 0.969, essentially tied with the Phase-1 BST baseline `run_20260418_151139` on macro / acc / top-2 (-0.022 on min) but with seed variance ~2.5x tighter on macro and accuracy. Combo B (`une_merge_v1 + split_v2`, `run_20260430_110101`, S4 best) lands 0.739 / 0.317 / 0.766 / 0.938; macro / acc / top-2 hold within noise of V4 baseline (`run_20260420_171101`) but **min drops 7 pp** because Top_wrist_smash specifically gets worse with cleaner pose data. Combo A (`une_merge_v1_nosides + split_v2`, `run_20260430_170325`, S4 best) lands 0.742 / 0.375 / 0.767 / 0.938, recovering most of combo B's wrist_smash floor via the structural side-collapse (+0.058 min vs combo B); essentially tied with the Phase-1 collapsed-classes ablation (`run_20260425_185421`) on macro / acc / top-2. Together the arc shows cleaner pose data lifts head metrics and tightens seed variance on common classes but hurts the small-support tail in the un-pooled 28-class taxonomy. **Diagnosis is now classifier-side, not data-side**; three pre-flight scripts pin this down (see next bullet).
- **Pre-flight scripts: shuttle-missing diagnosis verified (2026-04-30)**: three new scripts under `src/bst_refactor/validation_scripts/`. `shuttle_gap_y_distribution.py` confirms the off-screen-high hypothesis at the sensor level: 61.6% of gap boundaries cluster in the top 10% of the broadcast frame, 72.3% on the post-gap re-appearance side. `shuttle_gap_length_distribution.py` shows the inpaint module isn't being exceeded (only 1 gap >60 frames in 32k clips); 85% of missing-shuttle frames sit in the 11-60 frame band of "shuttle genuinely not in any pixel". `perclass_shuttle_miss_vs_f1.py` against the combo A nosides manifest returns Pearson **+0.516** (Spearman +0.415), opposite of the predicted direction. The high-shuttle-miss classes are the pose-distinctive serves / clears / lobs at F1 ~0.95-0.99; the bottleneck classes (wrist_smash, drive, push, cross_court_net_shot) sit at sub-1% miss rates and have shuttle data available. Combined diagnosis: shuttle data is reliably present where it's most needed; the model just isn't using it well. The mask-channel arm gets demoted; trajectory extrapolation flagged as a longer-term direction for the off-screen-arc gaps. **Label smoothing is now the highest-priority loss-side experiment**; full search-space analysis at `scratch/architecture_notes/hparams_sweep_speculations.md`.
  - **Don't delete the legacy `_merged_25` nested tree yet.** It's the only path to bit-exactly reproduce the V4 / Phase-1 baseline. The new extract should be a strict improvement (drop-bug fixed, sticky_anchor over the full set instead of 1,716), so the realistic risk is just losing the historical baseline number, not losing useful results. Keep until the Phase-2 sanity-train numbers are in and the writeup commits to a baseline.
  - **Unknown class still has no pose data.** The 1,278 `raw_type_en == 'unknown'` clips were excluded from the Phase-2 extract because every active taxonomy uses `--drop-unknown`. If we ever want them (noise / distractor class for a robustness ablation), extract to a sibling `/scratch/comp320a/ShuttleSet_keypoints_raw_unknown/` so the garbage bucket can never accidentally enter canonical training via a permissive glob. ~17 h single-process; not blocking.

## Status (2026-04-25)

- **BST LR-schedule retune (Q4)**: done. Compressed schedule beats the paper on every test metric (macro F1, min F1, accuracy, top-2). Active settings in `bst_train.py`. Numbers in "LR schedule retune" below.
- **CG/AP annealing (Q3)**: done. Three matched 5-serial runs. Annealed-out best (mean macro F1 0.829), always-on close behind, always-off trails. Annealed kept as the active config.
- **Attention head geometry sweep (Q5)**: open, not started. Secondary priority.
- **X3D-S racket crop fusion**: model + input shape decided; fusion depth, training schedule, temporal cut-in, and MMPose-drop handling all open. Primary research direction; build slated for late next week.
- **Label smoothing sweep**: **closed on combo A nosides as of 2026-05-01**. Two cells run; LS=0.0 disproved the rare-class-tax hypothesis (mean wrist_smash dropped 1.6 pp vs LS=0.1 baseline). LS=0.15 cell currently in flight as the second axis point. If LS=0.15 doesn't lift wrist_smash mean above ~0.40, the arm closes and we revert to LS=0.1 for downstream cells. See the 2026-05-01 status block above for full numbers, and `scratch/architecture_notes/hparams_sweep_speculations.md` for the original design.
- **Class weighting smoke test**: prepped (2026-05-01) and queued behind LS=0.15. Manual pair-balanced weights `{'wrist_smash': 2.0, 'smash': 2.0}` to test whether loss-side reweighting alone can move the wrist_smash floor without stealing recall from smash. Code branch landed; one-line activation when LS=0.15 reports. See "Next: class weighting then focal" below.
- **Focal loss ablation**: deferred behind the class-weighting smoke test. Vanilla sample-based focal becomes the natural refinement only if class weighting moves wrist_smash F1; otherwise focal is unlikely to either and the right pivot is augmentation. Class-F1-driven adaptive focal queued as a separate research arm; see exploration prompt at `scratch/architecture_notes/class_f1_focal_exploration_prompt.md`.
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

## Next: class weighting then focal

Three loss-side cells are queued, in order. The LS sweep arc closing without lift on combo A nosides reframed the next steps; the framing here supersedes the earlier "Next: focal loss" plan.

### Cell 1: pair-balanced class weighting (smoke test, gates everything else)

`class_weights={'wrist_smash': 2.0, 'smash': 2.0}` already wired into `bst_train.py:79-` (Hyp field) and `bst_train.py:301-` (CE branch with renormalisation to mean weight 1.0 so LR / grad-clip behaviour stays comparable). Activation is the single line in the active hyp block; no other code change. Run kicks after LS=0.15 lands.

The pair-balanced design is the load-bearing decision. Asymmetric weighting (wrist_smash up alone) creates a directional gradient bias the model can exploit by flipping the smash↔wrist_smash boundary toward wrist_smash predictions, inflating wrist_smash recall at the cost of smash precision; macro might not move. Equal 2.0 on both says "spend more capacity disambiguating this pair" without nominating a winner. Smash sits at F1 ~0.66 (not ceiling-saturated), so realistic upside is both classes lift.

Drive (the next bottom-F1 class after wrist_smash and smash) deliberately excluded — its confusion partner is probably clear or push, and a singleton bump on drive is a separate single-class bet rather than a pair-balance bet. Run drive in a follow-up cell only if the wrist_smash + smash result reads as informative.

Compare against the LS=0.1 baseline `run_20260430_170325` (or LS=0.15 if that wins on the cell currently in flight). Watch head metrics — clear / services / lob — for a dip; the renormalisation expects ~0.005-0.01 loss-budget shift on these. Anything bigger means the bump is too aggressive and 1.5 is the recovery default.

### Cell 2: vanilla sample-based focal (gated on Cell 1)

Run only if Cell 1 moves wrist_smash F1 by more than seed noise (~0.02 mean). Vanilla focal `FL = -(1-p_t)^gamma * log(p_t)` at gamma=1.0 first (conservative on a 979-sample wrist_smash with annotation noise), gamma=2.0 as follow-up if gamma=1.0 lifts. **No class-count-based alpha** — the count-vs-difficulty decoupling on combo A (wrist_smash is 5th-rarest by count but bottom by F1; long_service is rarest by count but at F1 0.99) means inverse-count alpha would upweight saturated classes and barely move wrist_smash. **LS=0 is non-negotiable** in any focal arm — LS softens targets so even confident-correct samples have `p_t < 1.0`, which contaminates focal's hardness estimate.

If Cell 1 lifts and we want to combine focal with the directional bet, the form is **manually-alpha focal**: `alpha={'wrist_smash': 2.0, 'smash': 2.0, default: 1.0} * (1-p_t)^gamma * log(p_t)`. This isn't class-balanced focal in the standard inverse-count sense; it's manual alpha matching the smoke-test pair plus per-sample focusing on top.

Implementation sketch (4 edits in `bst_train.py`):

1. Add a small `FocalLoss(nn.Module)` wrapping `F.cross_entropy(..., reduction='none', weight=optional_alpha_tensor)` with `(1-p_t)^gamma` reweighting.
2. Add `focal_gamma` to the `Hyp` namedtuple; gate behaviour on `focal_gamma > 0`.
3. Branch loss construction to `FocalLoss(gamma=hyp.focal_gamma, weight=resolved_alpha)` when `focal_gamma > 0`, else fall through to the existing CE path (uniform or class-weighted).
4. Force `label_smoothing=0` in the focal branch with a print warning if the user set it non-zero.

Watch train-vs-val loss curves. If train loss keeps dropping while val plateaus, focal is fitting label noise (the well-known noise-amplification failure mode of focal at higher gamma). The existing `early_stop_n_epochs=40` catches the worst of this; tighten to 20 if the focal-arm curves look suspicious.

### Cell 3 (research arm): class-F1-driven adaptive focal

The "obvious" idea — read per-class F1 each epoch on train, set the next epoch's per-class alpha from those F1s, combine with focal's per-sample gamma — does exist in the literature (CDB-loss, Seesaw, Equalisation Loss v2) but isn't a drop-in pytorch primitive. Worth a literature-grounded design pass before implementing, since the count-vs-difficulty decoupling on combo A makes this conceptually the best-targeted loss for our specific failure mode. Self-contained exploration prompt at `scratch/architecture_notes/class_f1_focal_exploration_prompt.md`. Outcome of that prompt feeds into whether Cell 3 runs in this trimester or gets parked for the writeup-side discussion.

### Decision tree

| Cell 1 result | Next cell |
|---------------|-----------|
| wrist_smash + smash both lift, head holds | Cell 2 manually-alpha focal at gamma=1.0 with the same 2.0/2.0 alpha; if it lifts further, gamma=2.0 |
| wrist_smash lifts, smash drops | Cell 2 vanilla sample-based focal (let the per-sample focusing find its own balance instead of prescribing one) |
| wrist_smash flat or worse | Loss-side knobs likely exhausted on this taxonomy; pivot to augmentation (horizontal flip + tightened random shift) or jump straight to X3D-S |

If Cell 2 also lifts, Cell 3 (adaptive focal) is the principled refinement. If Cell 2 doesn't lift after Cell 1 did, Cell 3 is unlikely to help further (you'd be tuning a more sophisticated mechanism for the same effect). Augmentation is the natural intermediate step before X3D-S in the loss-side-exhausted branch. X3D-S remains the long-term solution either way since it adds racket-pixel information pose-only can't see.

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
