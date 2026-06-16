# BST-X: Badminton Stroke Transformer, extended

BST-X starts from Chang's **BST-CG-AP** (Badminton Stroke Transformer, 2025, `arXiv:2502.21085`) and reworks it: a heavily retuned schedule, a new class-F1-weighted loss (**CDB-F1**), and a new player-detection heuristic, **`sticky_anchor`**, that recovers ~90% of the frames the old pipeline was throwing away at the moment of racket contact and is robust to spectators walking across the court (amateur generalisation). The groundwork is laid to add a 3D-CNN video stream (**X3D-S**) that watches the racket itself, to pull apart the stroke pairs pose alone can't. That will be built by early July.

The "X" in "BST-X" shows the model heritage: it marks where I've extended the BST baseline, and where the X3D-S wrist crop will extend it again in the coming weeks.

## Summary (trimester-1 close, 2026-06-02)

- **Performance**: ~0.75 macro on the 14-class no-sides taxonomy, and the hardest class, wrist_smash, finally cleared **0.5 min-F1** (0.525, from the weight-decay sweep), a fine-grained discrimination milestone for smash-wrist_smash. The coarser taxonomies reach 0.84 macro. Per-taxonomy bests are below; every run's numbers sit in the [ledger](../bst_x_training_runs.md).
- **The feature signal needs a new channel**: shown by smash vs wrist_smash. Pose-2D just doesn't carry the wrist-versus-full-swing tell, so the encoder can't split them. Everything pose/shuttle motion *can* discriminate sits at 0.95+.
- **What I tried**: schedule, the CDB-F1 loss, more model capacity, lost-frame recovery, a data-loss masking channel, augmentation, weight decay. Each nudged the floor a bit, but nothing moved the macro-F1 plateau by more than a couple of points.
- **Next**: the model needs more signal. The X3D-S wrist crop sees what pose can't. Series J settled the optimiser (wd 4e-1 with the decay exclusion is the new default; focal-alpha-revert retired); capacity Run 2 is the one small loose end left.

### Project bests (taxon_pinned_w_preds, standard baseline)

One clean 5-serial run per taxonomy on the stable pinned collation, same hp throughout: the per-taxonomy reference I handed the FE team. Best serial / 5-serial mean, test set:

| Taxonomy | Split | Unknown | Macro (best / mean) | Min-F1 (best / mean) | Run |
|---|---|---|---|---|---|
| bst_24 | baseline | drop | 0.842 / 0.827 | 0.612 / 0.568 | #36 |
| bst_25 | baseline | keep | 0.830 / 0.824 | 0.656 / 0.620 | #35 |
| bst_12 | v2 | drop | 0.821 / 0.812 | 0.631 / 0.599 | #34 |
| bst_24 | v2 | drop | 0.816 / 0.809 | 0.571 / 0.534 | #33 |
| une_v1_14 | v2 | drop | 0.751 / 0.742 | 0.485 / 0.471 | #37 |
| shuttleset_18 | v2 | drop | 0.665 / 0.654 | driven_flight: 1 test clip (n/a) | #32 |

All six share one config (the standard baseline below). bst_25 paper reference is 0.8097 / 0.5762 (variable-length, `arXiv:2502.21085` Table 1). On `une_v1_14` the weight-decay sweep, with norms, bias and embeddings left out of the decay, edged this reference and pushed wrist_smash past 0.5 (best min 0.525), and Series J carried the same wd 4e-1 lift onto bst_24 both splits (+5 to +6% mean min; see What I learned). wd 4e-1 with the exclusion is now the default optimiser setting; these rows stay the standard-baseline reference the FE team holds. Min-F1 only means something within one test population (split + unknown handling); shuttleset_18's is `driven_flight`, a single test clip, so its score is a coin-flip. Run numbers point into the ledger.

Shared baseline for the pinned-collation runs:

```
adaptive_focal: tau 1.0 · gamma 1.0 · momentum 0.9 · warm_up_epochs 5 · f1_floor 0.0
augmentation:   p_flip 0.5 · p_jitter 0.3 · cap_y 0.05 · cap_x 0.1 · eps 0.15
optimiser:      AdamW · wd 1e-2 · all layers decay
schedule:       n_epochs 80 · batch_size 128 · lr 5e-4 · cosine decay 0.5 · cg_ap ramp to 0 by epoch 15
detection:      sticky-anchor player detection
collation:      taxon_pinned_w_preds (shuttle-unzeroing baked in)
```

## What I learned

- **One pair drags classification: smash vs wrist_smash.** The encoder can't pull them apart (representation-bound) and pose-2D never gave it the cue to (signal-bound). Even train can't completely overfit these (only +14-18pp). Most other classes achieve test performance of 0.95+. My main work throughout this stage of the project was lifting min-F1.
- **Pose has nothing left to give on this pair.** The augmentation sweep flattened the picked-serial split to ~0.51 each: flip and jitter say nothing about wrist-versus-swing. So the next move is new signal, not another loss or pose-aug knob, which is exactly what X3D-S is for.
- **The loss-side is mapped.** Five CDB-F1 variants; tau=1 / gamma=1 lifts the floor best. A scalar per-class weight can't untangle the smash/ws confusion however I set tau or gamma.
- **As usual, data was the most important hp.** The old collation wiped the shuttle track whenever pose failed on a frame, and pose itself failed 5.38% of the time (now <1%). Recovering these frames, and then reactivating the shuttle for the remaining 1%, lifted mean min +1.2 and mean macro +0.5 over the best loss-side run on the same hparams, and it lifted both halves of the pair at once.
- **Regularisation picks up a bit at min-F1, but hardly lifts macro.** Weight decay, with norms, bias and embeddings left out, at 4e-1 pushed wrist_smash past 0.5 for the first time (best 0.525). Focal-alpha-revert was an interesting experiment (dynamically limit class alpha overweighting to those classes that show they're improving). But it only really balances as a CDB-F1 regulariser on its target taxon (une_v1 ,14), and even there it helps min, not macro. Series J then swept those wd endpoints across the other taxa: wd 4e-1 lifts the floor wherever the worst class started low (bst_24 both splits, +5 to +6% mean min) and does nothing once it's comfortable (bst_12, bst_25). So wd 4e-1 with the exclusion is now my default optimiser; focal-alpha-revert, two batches in with no real gain, is retired.
- **The May 30-31 multi-taxonomy sweep consolidated all my observations.** split_v2 is 3-4% harder on test than the baseline split at the same hp; collapsing the sides (bst_24 to bst_12) is a free but tiny win, so the model was already tying the top and bottom variants together; and dropping unknown helps the real strokes, since the services stop leaking into it.

## How I got here

Phase boundaries only; the [ledger](../bst_x_training_runs.md) has the per-run detail and the appendix the workings.

- **Apr 17-18**: reproduced the BST baseline on bst_25, then retuned the LR schedule and the CG/AP auxiliary-loss schedule.
- **Apr 20-25**: moved to the UNE taxonomy + split_v2 on a CSV pipeline, and tested collapsing the sides.
- **Apr 29-30**: the Phase-2 full-clean MMPose extract (sticky_anchor over all 32,203 clips), then a per-taxonomy sanity train.
- **May 1-3**: the loss-side sweep (label smoothing, class weighting, the CDB-F1 family), and capacity Run 1 (head-MLP widening, reverted).
- **May 3-6**: the lost-frame recovery (the big floor lift), then augmentation v1 locked and swept.
- **May 30 - Jun 2**: the taxon-pinned registry landed, then the multi-taxonomy reference batch, the focal-alpha-revert arm, and the weight-decay sweep across taxa (Series J). wd 4e-1 + exclusion adopted as the default; alpha-revert retired.
- **Pending**: capacity Run 2 and the X3D-S build.

## The architecture

Same four pieces as up top: an inherited core I retuned, a new loss, a new detector, and the video stream that's coming.

**BST-CG-AP core, retuned schedule.** This a tuned version of Chang's BST: a TCN front-end feeding temporal and interactional transformers, with the CG (Clean Gate) and AP (Aim Player) heuristic modules for warm-start priors. I schedule CG/AP down to zero over the first 15 epochs, then let the rest of the backbone train at a decaying learning rate (BST's original train length ran ~20x longer, though the best epoch was always in the first 80). Defaults `d_model=100, d_head=128, n_head=6` (`bst.py`). Schedule rationale in the appendix.

**CDB-F1 loss.** A class-balanced focal loss that boosts whichever classes are lagging: a per-class weight of `(1 - F1_c)^tau`, driven by a running average of train F1, so a class that stays weak keeps drawing more weight, times the usual focal `(1 - p_t)^gamma` term. tau=1 / gamma=1 lifts the floor best of the variants tested. Built on the CDB loss (ACCV 2020); design in `class_f1_focal_design.md`.

**sticky_anchor player detection (done).** BST used to zero a whole frame the moment a player's ankle midpoint projected off-court, or fewer than two people were detected. That hit airborne smashes hardest: the jump throws projected feet ~0.17-0.24 units off court, so the model went blind at the exact instant the stroke happened. sticky_anchor swaps that filter for per-slot tracking: each slot grabs its closest in-court candidate against a court-half anchor (75% fixed, 25% running average), with a guard so the two halves can't steal the other's anchor. Because it tracks the right player instead of just counting bodies, it ignores extra people near the court, a spectator crossing or a line judge, which is what amateur footage is full of.

Across all 32,203 clips it cut the overall drop rate 5.38% to 0.93%, and at the moment of contact 5.98% to 0.58%, a 19-76x recovery on the worst-hit strokes. The ~0.93% still failing is mostly unrecoverable framing (close-ups, side-on, cutaways). The design and the two parked follow-on routes (gap-fill interpolation, homography-fail rescue, neither built) are in the appendix and `mmpose_heuristic/`.

**X3D-S wrist-crop fusion (building, ~July).** The major architectural extension planned and partly developed. A small X3D-S video model runs on a crop centred on the wrist and discriminates the features that keypoints can't sufficiently distinguish (i.e., rotation). I hope this will recover the smash/wrist_smash confusion. X3D-S over XS/M/L: fine enough for racket motion without the parameters. I'm fine-tuning toward 39 frames at stride 1 so the receptive field spans the whole stroke. Still open: where the fusion goes (late concat, into the attention, or a separate tower), the fine-tune-then-co-train schedule, where to sit the input window given the reported contact time is noisy, and what to do about MMPose drops. Six-stage plan at `x3d_integration_macro_plan/`.

## Next steps

1. **X3D-S wrist-crop fusion** (primary): the signal booster. Model and input shape are settled; fusion depth, schedule, window placement and drop-handling are open.
2. **Capacity Run 2**: widen the encoder (d_model 100 to 192, trim d_head 128 to 32). Run 1 widened the head MLP instead, went flat and cost wrist_smash. Unlikely to be worthwhile compared to improving the data signal, but worth exploring. I could not find any literature in the model heritage justifying the inverted bottleneck transformer widths.
3. **Series J (wd endpoints + alpha-revert)**: done. wd 4e-1 with the decay exclusion is the new default optimiser setting (it lifts the floor where the worst class is starved, bst_24 both splits and une; flat elsewhere); focal-alpha-revert retired. Summary at [`bst_x_sweep_summary_wd_x_focal_alpha_revert.md`](bst_x_sweep_summary_wd_x_focal_alpha_revert.md).
4. Smaller loose ends (augmentation round 2 flip ablation, the TCN dilation A/B): in the appendix.

## Pointers

- [`bst_x_training_runs.md`](../bst_x_training_runs.md): the run ledger, every run's per-metric best and mean, grouped global / per-taxonomy / per-series.
- [`bst_x_overview_technical_appendix.md`](bst_x_overview_technical_appendix.md): the workings, per-experiment detail, comparability caveats, decisions on hold, secondary investigations, cross-references.
- [`augmentation_framework.md`](augmentation_framework.md) and [`x3d_integration_macro_plan/`](x3d_integration_macro_plan/): the two most active sub-docs.

## Appendix: namespace conventions (post-rebrand)

Three spellings coexist by design after the BST -> BST-X rebrand. Each lives in one register; the others do not encroach.

**`bst` (Chang lineage only).** Identifiers that ground us in Chang 2025's published baseline rather than the project derivative:

- `class BST` and the five variant partials (`BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`, `BST_CG_AP`) in `src/bst_x/stroke_classification/model/bst.py`.
- The Chang-key entries of the `MODELS` dispatch dict in `bst_x_common.py` (the project-side alias `'BST_X'` sits alongside them).
- Taxonomy ids `bst_25` / `bst_24` / `bst_12` and the constants `TAXONOMY_BST_25/24/12` (paper-faithful taxonomies).
- The split column `split_bst_baseline` and constant `SPLITS_BST_BASELINE`.
- The Chang-baseline run dir `experiments/bst_cg_ap_base_17_04_2026/` and its lowercase `bst_cg_ap_*.pt` weights.
- The paper transcript at `local_scratch/bst_paper_md/`.

**`bst_x` (Python identifiers; snake_case file + dir names).** The project rebrand in code:

- Package dir `src/bst_x/`.
- Module files `bst_x_common.py`, `bst_x_train.py`, `bst_x_infer.py`, `src/api/bst_x_inference.py`.
- Builder `build_bst_x_network`, constants `BST_X_REFACTOR` / `BST_X_CLASSIFICATION`, exception `BstXInferenceUnavailable`.
- Env vars `BST_X_CLIPS_DIR`, `BST_X_INPUTS_DIR`, `BST_X_MMPOSE_NPY_DIR`, etc.
- Model-name dispatch key `'BST_X'` (the project alias of `BST_CG_AP`).
- Weight prefix `bst_x_*.pt` in `experiments/run_*/weights/`.
- Registry model ids `bst_x_une_v1_14_v2`, `bst_x_bst_24_bst_baseline`, etc.

**`bst-x` (wire format; kebab-case).** Anything that crosses a JSON / YAML / package-manager boundary:

- The `architecture` enum value `"bst-x"` in `docs/models_registry.yaml`, the Pydantic `Markup` / `LibraryPredictRequest` schemas, and the FE picker.
- The `pyproject.toml` extras group `bst-x-runtime`.
- The training venv name `venv-bst-x` (operator convention).
- The FE filter / display strings the model-picker uses.

The rule of thumb: pick the spelling by where the name is read. Python parser reads it as a Python identifier -> `bst_x`. JSON parser, package-manager, shell, or user reads it -> `bst-x`. Chang-baseline reference -> `bst`.
