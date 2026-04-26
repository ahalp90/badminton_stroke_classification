# Arch 1: Open Research Directions

Arch 1 is the BST + X3D-S wrist crop fusion architecture. This doc tracks the live research arms and decision-making for it. Two novel contributions live here: the X3D-S fusion (primary) and the `sticky_anchor` per-slot player-identification heuristic (secondary but semi-significant in its own right — a data-quality fix that materially repaired the upstream MMPose extraction; see the TLDR section below). Everything else tweaks the inherited BST scaffolding (Q3-Q5).

## Status (2026-04-25)

- **BST LR-schedule retune (Q4)**: done. Compressed schedule beats the paper on every test metric (macro F1, min F1, accuracy, top-2). Active settings in `bst_train.py`. Numbers in "LR schedule retune" below.
- **CG/AP annealing (Q3)**: done. Three matched 5-serial runs. Annealed-out best (mean macro F1 0.829), always-on close behind, always-off trails. Annealed kept as the active config.
- **Attention head geometry sweep (Q5)**: open, not started. Secondary priority.
- **X3D-S racket crop fusion**: model + input shape decided; fusion depth, training schedule, temporal cut-in, and MMPose-drop handling all open. Primary research direction; build slated for late next week.
- **Focal loss ablation**: next experiment. Drop-in loss-fn change on the existing sticky_anchor data, targets the wrist_smash floor's representational instability rather than data quality. Spec in "Next: focal loss" below.
- **Data augmentation**: probable intermediate step after focal loss, before X3D-S. Particulars TBD.
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

## Next: focal loss for the wrist_smash floor

The Phase 1 sticky_anchor retrain (`run_20260425_150548`) failed the decision gate: `Top_wrist_smash` mean -0.057 vs V4 baseline, even though macro / acc / top-2 each lifted by ~0.007. Cleaner data shifted the smash / wrist_smash boundary toward smash, not toward the floor. The per-class frame-zeroing audit (`src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/zeroed_frames_class_audit__run_20260425_150548.txt`) confirmed it: the F1-bottom classes aren't the heavily-zeroed ones, and the worst-zeroed class has near-perfect F1. So the wrist_smash bottleneck is structural (representation / boundary allocation), not data quality.

**Focal loss ablation (~2 hr V100).** Replace `nn.CrossEntropyLoss(label_smoothing=0.1)` at `bst_train.py:301` with multiclass focal loss `(1 - p_y)^gamma * CE`. Focal explicitly upweights misclassified borderline cases, which is where the wrist_smash variance comes from. No data-pipeline change; same flat dir, same collated dir (just retag the `ablation_id`). Compare against `run_20260425_150548` (the sticky_anchor run, not V4 baseline) so the loss change is isolated from the data-quality change.

Implementation sketch (4 edits in `bst_train.py`):

1. Add a small `FocalLoss(nn.Module)` class wrapping `F.cross_entropy(..., reduction='none')` with `(1-p_y)^gamma` reweighting.
2. Add `focal_gamma` to the `Hyp` namedtuple; set `focal_gamma=1.5` and tag `ablation_id` with `_focal15` in the active hyp block.
3. Branch loss construction at `bst_train.py:301` to use `FocalLoss(gamma=hyp.focal_gamma, label_smoothing=0.0)` when `focal_gamma > 0`. Drop label smoothing in the focal path; smoothing and focal are both confidence regularisers and stack weirdly.
4. No collator change.

Practical notes: gamma=1.5 is the conservative starting point (Lin et al. used 2.0 on much harsher imbalance). Aux-schedule fade is still active during epochs 1-15, so the focal run stacks two regularisers shifting at once; tolerable for a first try, set `use_aux_schedule=False` for a perfectly clean ablation.

If focal lifts the wrist_smash floor, data augmentation is the natural intermediate step before X3D-S. If it doesn't, jump to X3D-S directly. X3D-S is the principled long-term solution either way, since it adds racket-pixel information that pose-only can't see.

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
