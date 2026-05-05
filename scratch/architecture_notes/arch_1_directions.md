# Arch 1: BST + X3D-S Wrist Crop Fusion

Arch 1 extends BST (Badminton Stroke Transformer, Chang 2025, `arXiv:2502.21085`) with an X3D-S video branch on a wrist crop, fused for joint stroke classification on ShuttleSet. Two novel contributions live here:

1. **X3D-S wrist crop fusion** (primary): pixel-level racket-and-wrist motion that pose-2D throws away, fused into the BST skeleton classifier.
2. **`sticky_anchor` per-slot player-identification heuristic** (secondary but methodologically distinct): reframes per-frame player picking from eligibility-filter ("zero the frame if either player projects off-court") to tracking-by-anchor ("each slot picks its own closest in-court candidate, with a Voronoi guard against cross-half capture"). Materially repaired the upstream MMPose extraction.

Everything else tweaks the inherited BST scaffolding (Q3-Q5 in the codebase: CG/AP scheduling, LR-schedule retune, attention head geometry).

## TLDR (2026-05-04)

**Active baseline**: combo A (`une_merge_v1_nosides + split_v2 + dropunk`), 14-class head, CDB-F1 loss with `adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`, compressed LR schedule (`n_epochs=80, lr=5e-4, num_cycles=0.5, warm_up_step=100, early_stop_n_epochs=40`), CG/AP annealed-out (1.0 → 0.0 by epoch 15), and the **shuttle-unzeroing-on-keypoint-fail** collation fix (drops the asymmetric `shuttle[failed,:] = 0` wipe at `prepare_train_on_shuttleset.py:866-867`). Best run `run_20260503_172922`: mean macro 0.7481 / min wrist_smash 0.4742 / acc 0.7653 / top-2 0.9353. Best S2 within that run: 0.7559 / 0.4935 / 0.7684 / 0.9334.

### Headline results across the project arms

Mean across 5 serials except where noted; combo A nosides except as flagged. **Mean is bolded.** "Best serial" column shows that run's top-serial macro / min wrist_smash to 2dp. BST paper rows are Chang's published single-run figures (`arXiv:2502.21085` Table 1), not our reproduction.

**Comparability caveats — read before treating any two rows as apples-to-apples**:

- **Taxonomy boundaries**: macro and min are computed across the active class set, so cross-taxonomy comparison is not direct. merged_25 keeps the 'unknown' class slot via `MERGE_MAP['driven_flight']='unknown'` even with dropunk on. `une_merge_v1` and `une_merge_v1_nosides` drop the unknown slot entirely from the active head once dropunk lands.
- **Drop-unknown effect**: counterintuitively, dropping unknown *lowers* macro and min. The model was reliably good at flagging the unknown class (it's the obvious garbage / weird-clip catch-all), so 'unknown' contributed an above-mean F1 to the macro pool. Removing it leaves only the harder classes, so macro and min drop a few points without the model actually getting worse on the strokes that matter. Affects every cross-taxonomy comparison between merged_25 and the une_merge_v1 family.
- **MMPose data-quality eras**: original BST extraction → Phase 1 mixed (sticky_anchor on 1,716 hit-zone-busted clips, rest original) → Phase 2 full-clean (sticky_anchor across all 32,203 clips, landed 2026-04-29 with `run_20260429_202144` as the first full-clean training run). Pre-Phase-2 runs train on a different per-frame zeroing distribution from post-Phase-2 runs.
- **TrackNetV3 inpaint mismatch with BST published figures**: Chang's published figures used TrackNetV3 *without* inpaint; our reproduction accidentally used the inpaint variant from day one (cleaner shuttle stream than the published baseline). So our LR retune row beats BST paper by more than just the LR retune mechanism would predict; some of the lift is the cleaner shuttle data the inpaint variant gives. All our runs from `run_20260417_191851` onward use TrackNetV3-with-inpaint.

| Run / arm | Run id | macro (mean) | min ws (mean) | acc (mean) | top-2 (mean) | Best serial macro / min ws |
| --- | --- | --- | --- | --- | --- | --- |
| *— merged_25 (25-class, retains 'unknown' slot) —* | | | | | | |
| BST paper, fixed-width strategy *(published, single run; original BST extraction; TrackNet **without** inpaint)* | n/a | 0.7983 | 0.5196 | 0.8254 | 0.9503 | n/a |
| BST paper, variable-length strategy *(published, single run; same caveats)* | n/a | 0.8097 | 0.5762 | 0.8322 | 0.9594 | n/a |
| LR retune (Q4, 3 serials; original BST extraction + our TrackNet-with-inpaint) | `run_20260417_191851` | **0.826** | **0.607** | **0.842** | **0.963** | 0.83 / 0.63 (S1) |
| Phase 2 sanity combo C *(clean MMPose; first full-clean run)* | `run_20260429_202144` | **0.831** | **0.577** | **0.848** | **0.969** | 0.83 / 0.58 (S2) |
| *— une_merge_v1 (28-class with sides; dropunk active, no 'unknown' slot) —* | | | | | | |
| Phase 2 sanity combo B *(clean MMPose, dropunk)* | `run_20260430_110101` | **0.739** | **0.317** | **0.766** | **0.938** | 0.74 / 0.32 (S4) |
| *— une_merge_v1_nosides (14-class; dropunk active, no 'unknown' slot) —* | | | | | | |
| Phase 2 sanity combo A *(clean MMPose, dropunk; LS=0.1 baseline)* | `run_20260430_170325` | **0.742** | **0.375** | **0.767** | **0.938** | 0.74 / 0.40 (S4) |
| LS sweep winner (LS=0.15) | `run_20260501_073430` | **0.747** | **0.417** | **0.769** | **0.938** | 0.75 / 0.45 (S3) |
| Class-weighting smoke (2.0/2.0 on smash, ws) | `run_20260501_110525` | **0.748** | **0.422** | **0.770** | **0.936** | 0.76 / 0.52 (S2; project-wide ws ceiling) |
| CDB-F1 tau=1, gamma=1 (first loss-side best) | `run_20260501_164658` | **0.7432** | **0.4621** | **0.7617** | **0.9351** | 0.75 / 0.49 (S2) |
| Capacity Run 1 (mlp_head 400→1200) | `run_20260503_104300` | **0.7414** | **0.4138** | **0.7604** | **0.9320** | 0.74 / 0.44 (S1) |
| **Shuttle-unzeroing wipe_drop (current best)** | `run_20260503_172922` | **0.7481** | **0.4742** | **0.7653** | **0.9353** | **0.76 / 0.49 (S2)** |
| Shuttle-mask variant 2a (mask_wiring) | `run_20260503_192718` | **0.7440** | **0.4568** | **0.7630** | **0.9365** | 0.75 / 0.49 (S4) |
| Jitter-off ablation (`RandomTranslation_batch(prob=0.0)`) | `run_20260504_152529` | **0.7401** | **0.4301** | **0.7586** | **0.9365** | 0.74 / 0.48 (S2) |

### Key learnings

- **Loss-side knobs explored thoroughly, but the loss-side wasn't the only available lever.** 5 CDB-F1 variants explored (tau=1/gamma=1, gamma=0, tau=0.5, pair-cap, gamma=2). Original tau=1+gamma=1 was the floor-lift sweet spot among the loss arms at +8.7 pp wrist_smash on the LS=0.1 baseline; smash ↔ wrist_smash pair-confusion via scalar-per-class alpha is structural and unfixable by retuning τ or γ. **The shuttle-unzeroing data-side fix (`run_20260503_172922`) then lifted mean macro +0.5 / mean min ws +1.2 / mean acc +0.4 over the loss-side best on identical hparams**, beating what any further loss-knob retune managed. Loss-side ceiling on this loss family looks effectively mapped; what isn't is the broader plateau, since data-side moved the needle.
- **Capacity might still be a lever, just not on the head-MLP axis in isolation.** Capacity-bottleneck research at `scratch/architecture_notes/model_capacity_bottleneck_question.md` argued plateau is data-bound and signal-bound. Run 1 (mlp_head 400→1200) confirmed mlp_head widening alone: head metrics flat, ws -4.8 pp, swap reverted. But the smash-up / ws-down pattern under widening hints at a capacity-bound representation somewhere — wider head let the easier head class run away from the harder pair member, which is consistent with insufficient encoder capacity to separate the pair. Run 2 (d_model 100→192 + d_head trim 128→32) is still the natural test of this; Run 1's flat result weakens but doesn't void the prior because encoder-side capacity is structurally different from head-side.
- **Data quality is a real lever after all.** Phase 1 sticky_anchor mixed retrain (`run_20260425_150548`) and the per-class frame-zeroing audit *had* ruled out the gross data-quality-bottleneck hypothesis at the keypoint-extraction level. Shuttle-zeroing concentrates in the high-arc classes (long_service 24.7%, smash 13.7%, clear 11.9%, lob 9.1%, wrist_smash 8.5%, drop 8.1% whole-clip means against sub-2% for the rest) because the bird leaves the top of the broadcast frame on high setups, per `validation_scripts/shuttle_gap_y_distribution.py` (61.6% of gap boundaries cluster in the top 10% of the frame). Within that high-arc family, smash↔ws is the only bottleneck pair, so it's the only one the model leans on shuttle to disambiguate; dropping the asymmetric shuttle-on-pose-fail collation wipe gave the floor-lift the loss-side knobs hadn't found. Earlier "data quality is not the bottleneck" framing was right about the keypoint side but missed the shuttle-side collation asymmetry.
- **The bottleneck is the smash↔wrist_smash pair specifically**: representation-bound (encoder isn't separating the pair on its own) and signal-bound (pose-2D doesn't carry the wrist-vs-full-swing distinction that X3D-S could see at the racket-pixel level). Train-test gap concentrates on this pair (14-18 pp); pose-distinctive classes generalise within 1-2 pp at 0.95+ test F1. The shuttle-unzeroing fix lifted both members of the pair (smash +1.5, ws +1.2) without trading them off, which is the first time a single intervention has done that.

### Active priorities (in order)

1. **Capacity-bump Run 2**: encoder-side widening (d_model 100→192 + d_head trim 128→32). Implementation surface + verifications + LR notes at `scratch/architecture_notes/transformer_widening_hparam_changes.md`. Now framed as testing whether encoder capacity is the local bottleneck, given Run 1's smash-up/ws-down pattern hints at one.
2. **Augmentation set landing**: centreline flip (p=0.5, coupled, COCO bilateral joint-index swap) + corrected pos+shuttle constrained-jitter (p=0.2, ±0.05y/±0.10x cap). Replaces the broken `RandomTranslation_batch`, which the jitter-off ablation (`run_20260504_152529`) showed was net-positive as regularisation despite being structurally wrong. Don't drop, replace. Full spec in [`augmentation_framework.md`](augmentation_framework.md).
3. **X3D-S fusion build**: long-term primary direction; addresses the signal-bound bottleneck. Model + input shape decided; fusion depth, training schedule, temporal cut-in, and MMPose-drop handling open. See "Primary research direction: X3D-S wrist crop fusion" below.

**TCN dilation flagged for investigation**: the inherited TCN runs two layers with `kernel=5` and dilations 1 then 3 (RF = 17 frames ~570 ms), plausibly pooling over the frame-by-frame micro-motion that discriminates the smash↔wrist_smash pair. 2-cell A/B (kernel=5 with dilation off, vs kernel=3 with dilation retained) writes up at "Secondary: TCN dilation pattern (open investigation)" below. Slot in after Capacity-bump Run 2 if the encoder-widening result leaves room.

### Held / parked

- **Seesaw-F1 loss-side second arm**: design at `scratch/architecture_notes/seesaw_f1_focal_design.md`, verified against the CVPR 2021 paper. Held as a targeted second arm only if a future signal-side gain reopens the smash↔ws pair-confusion question.
- **Mask-channel variant 2b (`pose_missing_either_slot`)**: variant 2a (`shuttle_missing` only) was tried (`run_20260503_192718`) and gave no benefit on top of just unzeroing the shuttle. Variant 2b never built; held in case a future change makes pose-fail signal worth surfacing explicitly.
- **Per-joint adaptive focal**: Phase 3 / trimester 2 research direction. Detail in `augmentation_framework.md`.
- **Swap val/test**: cross-player generalisation reframe. See "Parked decisions" below.
- **Homography-fail X3D-S-only rescue**, **gap-fill for partial-success frames**: cross-cutting MMPose recovery routes. Detail in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.

## Timeline

Two project-wide boundaries worth flagging up-front: **2026-04-29** is the cut-over from Phase 1 mixed MMPose data (1,716 clips repaired, rest original BST extraction) to Phase 2 full-clean MMPose (sticky_anchor over all 32,203 clips). **dropunk** also activates around this point on the une_merge_v1 family taxonomies, removing the easy 'unknown' class from the active head (and slightly lowering macro/min as a side effect; see Headline-results comparability caveats above).

| Date | Milestone | Run / artefact |
| --- | --- | --- |
| 2026-04-17 | LR schedule retune (Q4) — *original BST extraction; TrackNet-with-inpaint (mismatch vs BST published)* | `run_20260417_191851` (3 serials, merged_25) |
| 2026-04-19 | CG/AP annealing ablations (Q3) — *original BST extraction* | `run_20260418_151139` (annealed, winner), `run_20260418_174238` (always-on), `run_20260418_234822` (always-off) |
| 2026-04-25 | Phase 1 sticky_anchor mixed retrain — *Phase 1 mixed MMPose* | `run_20260425_150548` (decision-gate failure) |
| 2026-04-25 | Collapsed-classes ablation — *Phase 1 mixed; dropunk active on `une_merge_v1_nosides`* | `run_20260425_185421` (28→14 via taxonomy `une_merge_v1_nosides`) |
| **2026-04-29** | **Phase 2 full-clean MMPose boundary**: raw extract done | 32,203 stems unified, bit-identical bourbaki/engelbart |
| 2026-04-29 | Phase 2 sticky_anchor + zeroed-frame audit | overall fail rate 5.38% → 0.93%; full writeup `mmpose_heuristic/phase1_vs_phase2_2026-04-29.md` |
| 2026-04-29 | Phase 2 collation + env flip | three taxonomy combos collated, byte-identical cross-node |
| 2026-04-30 | Phase 2 sanity-train | A: `run_20260430_170325`, B: `run_20260430_110101`, C: `run_20260429_202144` |
| 2026-04-30 | Pre-flight diagnostic scripts | shuttle-missing diagnosis verified (Pearson +0.516 against predicted direction) |
| 2026-05-01 | Label smoothing sweep | `run_20260430_213933` (LS=0.0), `run_20260501_073430` (LS=0.15, winner) |
| 2026-05-01 | Class-weighting smoke test | `run_20260501_110525` |
| 2026-05-01 | Unknown ghost channel removed (architectural fix) | smokes `run_20260501_151131`, `run_20260501_152835` |
| 2026-05-01 | CDB-F1 first run (tau=1, gamma=1) | `run_20260501_164658` (loss-side best) |
| 2026-05-01 | CDB-F1 follow-ups (gamma=0, tau=0.5) | `run_20260501_192113`, `run_20260501_192519` |
| 2026-05-02 | CDB-F1 follow-ups (pair-cap, gamma=2) | `run_20260501_230252`, `run_20260502_075808` |
| 2026-05-02 | Capacity-bottleneck research | writeup at `model_capacity_bottleneck_question.md` |
| 2026-05-03 | Capacity-bump Run 1 (mlp_head 400→1200) | `run_20260503_104300`, swap reverted |
| 2026-05-03 | Frame-zeroing redesign: shuttle-unzeroing wipe-drop | `run_20260503_172922` (new project best mean) |
| 2026-05-03 | Frame-zeroing redesign: mask-channel variant 2a | `run_20260503_192718` (no benefit on top; parked) |
| 2026-05-04 | Augmentation set locked | full spec in `augmentation_framework.md` |
| 2026-05-04 | Jitter-off ablation: `RandomTranslation_batch(prob=0.0)` | `run_20260504_152529` (defaults restored; broken jitter is empirically regularising) |
| **pending** | Capacity-bump Run 2 (d_model 100→192 + d_head trim) | next gate |
| **pending** | Augmentation set landing | A/B vs no-aug baseline |
| **pending** | X3D-S fusion build | primary direction; build slated post-Run 2 |

## Architecture

### Active baseline configuration

`bst_train.py:62-79` plus the cosine call at `:308-314`: `n_epochs=80`, `early_stop_n_epochs=40`, `batch_size=128`, `lr=5e-4`, `warm_up_step=100`, `num_cycles=0.5`, `use_aux_schedule=True`, `aux_fade_end_epoch=15`. Compressed warm-start-then-finetune schedule paired with the CG/AP cosine fade: ~4 epochs warmup, ~15 epochs of CG/AP warm-start tapering to 0, then ~65 epochs of pure-backbone training under cooling LR. The BST paper's defaults (`n_epochs=1600`, `warm_up_step=400`, `early_stop_n_epochs=300`, `num_cycles=0.25`, `aux_fade_end_epoch=60`) and the dated retune rationale are at `scratch/architecture_notes/historical_bst.md` section 3.

Loss config: `adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`, label smoothing 0.0 (LS=0.15 was the winner of the LS sweep on combo A nosides; LS=0.0 carries the active CDB-F1 runs because adaptive focal already supplies the rare-class tax LS was approximating).

Data path: shuttle-unzeroing-on-keypoint-fail (`ablation_id=wipe_drop`, landed 2026-05-03 via the `shuttle/wipe-drop` branch, merged in commit `4e478fc`). The `shuttle[failed, :] = 0` line at `prepare_train_on_shuttleset.py:866` is *gone* from the active code; what's there now is just a prose comment explaining the absence. ~14k frames recovered (0.84% of extract); shuttle now flows through unmodified on pose-fail frames, mirroring the existing pose-flows-through-on-shuttle-fail behaviour. Mask-channel variant 2a (`shuttle_missing` channel via `mask_proj` + `shuttle_fuse`) was tested on top (`run_20260503_192718`) and gave no benefit; not adopted.

**Active collated tree location**: `npy_wipe_drop` sub-dir under the per-taxonomy collation root (e.g. `/scratch/comp320a/ShuttleSet_data_<tax>/npy_wipe_drop/{train,val,test}/`). Naming derives from `pipeline.config.derive_npy_collated_dir_basename` with `--ablation-id wipe_drop`; default-named dirs (`npy_<tax>_<split>_dropunk`) still hold the *pre-fix* shuttle-zeroed collation and are stale relative to the active code path until re-collation overwrites them.

**Backward compat with old weights**: model architecture is unchanged so old weights from `run_20260501_164658` and earlier load cleanly against the new collated trees (no shape mismatch). But test-time inference output will shift slightly because the previously-zeroed frames now carry real shuttle xy values that the old weights weren't trained against in that exact distribution. Architecture compat: yes. Inference reproducibility against the original training distribution: no. Flag explicitly when comparing old-weight inference numbers against runs trained on the new collated trees.

Active classes: BST head dim derived empirically from train `labels.npy` at first serial via `bst_common.derive_active_classes_from_labels` (val/test asserted as subsets); manifest `extra.arch` records `n_classes_full`, `n_active_classes`, `has_unknown`, `unknown_first`, `active_class_list`. Pre-fix (pre-2026-05-01) v1/nosides/raw_35 weights are not mechanically resumable post-fix because head dim shape mismatched; merged_25 dropunk is comparable across the boundary.

### Novel contribution 2: sticky_anchor MMPose extraction fix

The BST original zeroed an entire frame whenever a player's ankle midpoint projected outside the soft court rectangle (`eps = 0.01`) or fewer than 2 people were detected. Airborne smashes were the worst-affected class: jump geometry pushes projected feet ~0.17-0.24 normalised units off court (Padel paper `H_z * tan(θ)`), so the model saw zeros at the most informative moment.

`sticky_anchor` replaces that filter with per-slot tracking. Each slot has an anchor at its court half-centre (75% fixed, 25% running EMA of recent picks). The closest-to-anchor detection wins; off-court picks are still output but don't update the EMA. Bottom picks first; a closer-to-own-anchor Voronoi pre-filter blocks cross-half capture; a bbox-area + sitting-pose tiebreaker handles ambiguous frames. On the 1,716 hit-zone-busted clips (Phase 1): 95.05% perfectly clean post-fix; residual 61 are mostly irrecoverable framings (closeup, side-on, cutaway). Phase 2 (full extract over all 32,203 clips) cut overall fail rate 5.38% → 0.93%, hit-zone near-hit fail 5.98% → 0.58%.

Methodologically this is a small novel contribution in its own right: reframes per-frame player identification from eligibility-filter to tracking-by-anchor. The eligibility-filter formulation fails catastrophically on airborne strokes because the most informative frames are also the ones most likely to be filtered out; the tracking formulation keeps those frames usable. Full design + decision log in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.

### Primary research direction: X3D-S wrist crop fusion

#### Model choice: X3D-S

X3D-S for the racket-crop branch fits two constraints: easily available with weights, and small enough (params) to fine-tune end-to-end in the short time available on a v100 16gb. Other strong low-param models (MoViNet) lack prebuilt appropriate weights and easy model-zoo integration. X3D would probably do even better with SSv2 pretraining (fine hand motions), but the SSv2 weights only exist as an unofficial TensorFlow port and interface bugs would probably eat more time than the engineering.

Within the X3D family, S over XS / M / L / XL:
- **vs XS**: XS expects 4 frames × stride=12, too coarse for granular badminton racket motion.
- **vs M / L / XL**: they only drop stride to 5, perform not-that-much better, and use way more params.
- **X3D-S**: strong accuracy at low parameter count. Expected input is 13 frames × stride=6.

#### Target input shape: frames=39, stride=1

Fine-tuning toward `frames=39, stride=1`, not the default `13 × stride=6`. `stride=1` gives the model access to every frame and lets it learn the interactions between them, which is what granular badminton racket motion needs. `39` is set so that by the final convolutional block the receptive field covers all input frames. That imposes a hard limit around ~40 frames, which is fine for a racket crop centred on a stroke event.

#### Fusion depth: where X3D-S output enters BST (open)

Three competing options:
- **Late concat, just before the MLP head.** Easiest to implement, lowest risk, but gives BST no chance to condition its attention on the racket signal.
- **Tie into attention earlier, in a meaningful way.** X3D-S output feeds into the cross-attention or the interactional transformer, so the racket evidence shapes how players and shuttle attend to each other. More expressive, more moving parts.
- **Separate tower with learned significance weighting.** X3D-S runs as its own tower and a learned scalar (or vector) gates how much its prediction counts vs BST's. Keeps the two branches clean and lets the model decide per-sample how much to trust the racket signal.

#### Open training/integration questions

1. **Fine-tuning and end-to-end schedule**: right sequence for fine-tuning X3D-S on badminton video first, then co-training it end-to-end with the rest of Arch 1. Length of each phase, learning rates, what to freeze when.
2. **Temporal cut-in of X3D-S feedback**: the reported stroke racket contact times are noisy. Need to pick where the X3D-S input window sits relative to the reported contact time so the feature stays responsive even when the reported time is slightly off. Options: fixed offset centred on reported time, learned offset, or a slightly wider window that lets X3D-S self-align. Hit-frame metadata derivation now scoped without re-extraction (Method A: CSV correlation; Method B: shuttle trajectory inversion); detail in `augmentation_framework.md`.
3. **Juggling MMPose drops**: MMPose periodically drops frames, sometimes with alarming frequency for certain stroke categories. The X3D-S window has to cope. Aggressive frame-zeroing is now addressed at the extraction layer by sticky_anchor; the residual drops are detection-layer (heavy occlusion at the net, etc.) and the candidate fix is temporal interpolation. Worst case: pin the camera to the shuttle velocity reversal position.

**Wiring note for the active-class fix (2026-05-01)**: when adding the X3D-S branch (or any new training script), source the head dim from `task.n_active_classes` and run `_validate_and_record_arch` on serial 1, mirroring the bst_train.py pattern. Hardcoding `taxonomy.n_classes` in the fusion module would put the unknown ghost back; the existing `Task.get_network_architecture` shows the correct pattern.

### Secondary: BST attention head geometry (Q5)

`bst.py:145` defaults to `d_model=100`, `d_head=128`, `n_head=6`. The model concatenates across heads to `d_head * n_head = 768`, then `MultiHeadCrossAttention.tail` (`bst.py:59-62`) projects back down to 100. The temporal and interactional transformers in `tempose.py` follow the same pattern.

The ratio comes from BST → TemPose → AcT (Action Transformer). AcT ran progressive-widening ablations on exactly this expand-then-contract pattern: a small `d_model` keeps the bulk of the network cheap, while the wide per-head projection gives each head enough capacity to learn a distinct specialised view. Low total parameter count, rich per-head representations.

Nobody has swept this on BST. Worth a pass over `d_head ∈ {32, 64, 96, 128}`, either holding `n_head=6` (which shrinks the model) or holding `d_head * n_head` constant (which tests whether the expansion matters or just the total width). If a smaller `d_head` holds F1, free parameter-efficiency win. Caveat: `d_model` couples tightly across TCN, cross-transformer, interactional transformer, and PPF (see `tuning_thoughts.md`), so hold `d_model=100` fixed and only vary `d_head` / `n_head`. Capacity-bump Run 2 explores this axis from a different angle (encoder-side widening).

### Secondary: TCN dilation pattern (open investigation)

The TCN before the temporal transformer (`tcn_pose`, `tcn_shuttle`) is two stacked dilated 1D convolutions with kernel size 5, dilations 1 and 3 in that order (per `tempose.py:139` `dilation = i * 2 + 1`). Per-token receptive field by the second layer = `5 + (5 - 1) × 3 = 17 frames` ~570 ms at 30 fps. That covers a full racket-swing build-up plus contact plus follow-through.

**Hypothesis to investigate**: 17 frames is wider than necessary for the TCN's job (local-motif extraction; the temporal transformer downstream handles long-range integration), and the wide pre-pool may be smoothing over informative frame-by-frame motion data — the kind of micro-motion that distinguishes the smash↔wrist_smash pair. Worth a 2-cell A/B:

- **kernel=5, dilation off** (both layers at dilation=1): RF = 9 frames (~300 ms). Cheapest A/B; isolates the dilation question from the kernel question.
- **kernel=3, dilation 1/3 retained**: RF = `3 + 2 × 3 = 9 frames` (~300 ms). Same effective RF as above but achieved by trimming kernel rather than dilation; tests whether the kernel size or the dilation is doing the work.

Neither should break anything mechanically — the TCN's output channel count (`d_model=100`) is unchanged regardless, the temporal transformer downstream receives the same number of tokens (`seq_len=100`), and nothing downstream hardcodes a TCN receptive-field assumption. Risk is empirical: maybe the model genuinely benefits from the wider pre-pool. Detail and rationale at `scratch/architecture_notes/hparams_sweep_speculations.md` (Per-knob walkthrough → `tcn_kernel_size=5` block).

Slot in after Capacity-bump Run 2 if the encoder-widening result leaves room. Cheap (~3-5 hr each on A100 per the existing 5-serial pace).

## Experiment log

Chronological. Each entry: setup, hypothesis, result, takeaway, run id(s).

### LR schedule retune (Q4) — 2026-04-17

`bst_train.py:308-314` calls `get_cosine_schedule_with_warmup`. Original BST recipe passed `num_cycles=0.25` alongside `n_epochs=1600`, `warm_up_step=400`, and `early_stop_n_epochs=300`. At `num_cycles=0.25` only a quarter of the cosine curve runs across the full budget, so the LR barely decays. BST-default runs converge around epoch 60 and early-stopping fires around epoch 360, so the scheduler never had time to lower the rate.

Compressed `n_epochs` to match the real convergence timeframe and bumped `num_cycles` so the cosine curve actually hits zero. Active settings (old values preserved commented in `bst_train.py`):

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

All three serials beat the paper on every metric, so it's not a lucky seed. Huge jump on F1 min (+4.2 points vs prior best, +5.1 vs paper). Harder classes get a massive benefit. The val-vs-test direction flipped too: old run had val macro 0.8311 / test 0.823; the retune's winner had val 0.816 / test 0.830.

Winning weight kept at `main_on_shuttleset/experiments/run_20260417_191851/weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_25.pt`, tracked via `!` override in `.gitignore`. Numbers verified from `test_logs/test_20260417_191851.log`.

### CG/AP annealing ablations (Q3) — 2026-04-19

CG (Clean Gate) and AP (Aim Player) originally ran unweighted for the whole training run. Hypothesis: their strongest role is as a **warm-start prior**. Early in training the transformers haven't yet learnt robust shuttle- or player-aware representations, so the hand-crafted CG/AP interactions are useful inductive bias; later, once the transformers have learnt their own (analogous, potentially richer) interactions, fixed CG/AP could constrain the model, pinning it to the hand-crafted formulation.

Three matched 5-serial runs under the retuned LR schedule. Only the CG/AP schedule varies.

| Arm | aux_factor over epochs | Run | Mean macro F1 | Best serial (macro F1, acc, min F1) |
|---|---|---|---|---|
| Annealed out | 1.0 at ep. 1, cosine to 0.0 by ep. 15, then 0 | `run_20260418_151139` | 0.829 | S2: 0.831, 0.850, 0.600 |
| Always on | 1.0 for all 80 epochs | `run_20260418_174238` (Run A) | 0.826 | S3: 0.828, 0.844, 0.603 |
| Always off | 0.0 for all 80 epochs | `run_20260418_234822` (Run B) | 0.822 | S2: 0.830, 0.842, 0.586 |

Annealed > always-on > always-off, with small but consistent gaps. Peak performance (particularly accuracy) suggests CG and AP limit the model's top end when sample count is high (likely the accuracy-macro F1 divergence driver). CG/AP demonstrably useful as warm-start inductive bias; the tuned LR explains most of the difference from the original BST stats.

Pointers for raw numbers: per-serial metrics in each run's `experiments/run_.../manifest.yaml`; Serial blocks in `test_logs/test_20260418_*.log`.

### Sticky_anchor Phase 1 mixed retrain — 2026-04-25

Reran the V4 baseline (`run_20260420_171101`) with the 1,716 hit-zone-busted clips swapped in for their sticky_anchor-cleaned versions; everything else unchanged. Decision gate from the heuristic doc wanted a +0.02 target-class min-F1 lift; this run failed it.

Mean across 5 serials, vs V4 baseline:

| | sticky mean | V4 mean | Δ |
|---|---|---|---|
| macro F1 | 0.748 | 0.741 | +0.007 |
| min F1 | 0.333 | 0.389 | -0.056 |
| accuracy | 0.774 | 0.766 | +0.008 |
| top-2 | 0.942 | 0.936 | +0.006 |

Top_wrist_smash mean dropped 0.057. Top_smash gained almost exactly what wrist_smash lost (+0.020) — boundary-allocation tradeoff: cleaner data made the smash family easier and the model spent the gain on the easier head class instead of the rare tail.

Per-class frame-zeroing audit (`zeroed_frames_class_audit.py`) followed. F1-bottom classes weren't the heavily-zeroed ones, and the worst-zeroed class hit near-perfect F1. So **data quality isn't the floor bottleneck**; the data-quality-bottleneck hypothesis is empirically dead. Phase 2 deprioritised on this finding (full writeup in the heuristic doc).

Run + manifest at `experiments/run_20260425_150548/`; best S3.

### Collapsed-classes ablation — 2026-04-25

Same data as above. Only the label space changes: 28 classes to 14 by dropping the Top_/Bottom_ side prefix (new taxonomy `une_merge_v1_nosides`). Hypothesis: Top_X and Bottom_X are essentially the same shot mirrored across the net; forcing them separate halves per-class N and asks the model to learn a redundant distinction.

Mean across 5 serials:

| | nosides mean | sticky mean | V4 mean |
|---|---|---|---|
| macro F1 | 0.743 | 0.748 | 0.741 |
| min F1 | 0.397 | 0.333 | 0.389 |
| accuracy | 0.766 | 0.774 | 0.766 |
| top-2 | 0.938 | 0.942 | 0.936 |

vs V4 every metric is within ±0.008 (noise band). Absolute ceiling didn't move.

What did move was rare-class stability. Per-seed test-min range dropped from 0.124 (sticky) to 0.074 (nosides), and worst-seed min lifted from 0.235 to 0.350. The 14-class wrist_smash F1 (~0.42 mean) is close to the support-weighted mean of the old 28-class Top_wrist_smash (0.33) and Bottom_wrist_smash (0.45) — model isn't actually distinguishing smash from wrist_smash any better; the metric just stopped flipping between two thin slots. Doubled per-class N reduced the seed lottery on rare classes; absolute performance didn't change.

Run + manifest at `experiments/run_20260425_185421/`; best S1 (top min, top top-2).

### Phase 2 sticky_anchor full extract — 2026-04-29

Full re-extract of all 32,203 stems (Phase 1 only repaired 1,716 hit-zone-busted clips). Three artefacts:

- **Raw extract**: 32,203 stems at `/scratch/comp320a/ShuttleSet_keypoints_raw/` on both bourbaki and engelbart, bit-identical. Composed of 30,487 freshly re-extracted (across two shards over ~20h wall) plus the 1,716 Phase-1 backfill rsynced in. Verification: file counts match, cross-node `rsync --checksum` empty, failsafe byte-identity gate 50/50 on the 1,716 overlap with max abs diff 0.000e+00. Per-frame `ndet` baseline at `src/bst_refactor/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md` (0% `ndet=0`, 0.53% `ndet=1` floor).
- **Sticky_anchor + zeroed-frame audit**: clean dir at `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`, 32,203 stems × 3 files, byte-identical bourbaki/engelbart. Three `validate_zeroed_frames.py` reports landed (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`). Headlines: overall fail rate 5.38% → 0.93%, hit-zone near-hit fail 5.98% → 0.58% (the near/away gradient sign flipped: hit zone is now the cleanest zone instead of the noisiest). Per-stroke ratios 19x-76x on the strokes Phase 1 was failing hardest on (smash, clear, drop, return_net, wrist_smash). 17 residual 100%-hit-zone-zeroed clips look like irreducibly broken broadcasts. Comparison writeup at `scratch/architecture_notes/mmpose_heuristic/phase1_vs_phase2_2026-04-29.md`.
- **Collation + env flip**: three collated trees written under `/scratch/comp320a/ShuttleSet_data_<tax>/npy_<tax>_<split>_dropunk/`, one per active (taxonomy, split) combo. All run via `prepare_train_on_shuttleset.py --skip-trajectory --skip-pose --clip-npy-dir ...`, mirrored to bourbaki, byte-identical cross-node. Per-split clip counts cross-verified against `clips_master.csv` filtered for `--drop-unknown` via `verify_collated_counts.py` — all `OK`. `BST_MMPOSE_NPY_DIR` flipped from the legacy nested path to the new clean dir; one-step rollback at `.env.bak.2026-04-29`.

### Phase 2 sanity-train — 2026-04-30

Three full 5-serial runs across the active combos:

| Combo | Run | macro | min | acc | top-2 | vs prior baseline |
| --- | --- | --- | --- | --- | --- | --- |
| C: `merged_25 + split_bst_baseline` | `run_20260429_202144` (S2 best) | 0.831 | 0.577 | 0.848 | 0.969 | tied with Phase-1 BST baseline `run_20260418_151139` on macro/acc/top-2; -0.022 on min; seed variance ~2.5x tighter |
| B: `une_merge_v1 + split_v2` | `run_20260430_110101` (S4 best) | 0.739 | 0.317 | 0.766 | 0.938 | within noise of V4 baseline on macro/acc/top-2; **min drops 7 pp** because Top_wrist_smash specifically gets worse with cleaner pose data |
| A: `une_merge_v1_nosides + split_v2` | `run_20260430_170325` (S4 best) | 0.742 | 0.375 | 0.767 | 0.938 | recovers most of combo B's wrist_smash floor via structural side-collapse (+0.058 min vs B); essentially tied with the Phase-1 collapsed-classes ablation on macro/acc/top-2 |

Together: cleaner pose data lifts head metrics and tightens seed variance on common classes but hurts the small-support tail in the un-pooled 28-class taxonomy. **Diagnosis is now classifier-side, not data-side.**

Caveats kept on file: don't delete the legacy `_merged_25` nested tree yet (only path to bit-exactly reproduce V4 / Phase-1 baseline); unknown class still has no pose data (1,278 `raw_type_en == 'unknown'` clips excluded from Phase 2 because every active taxonomy uses `--drop-unknown`; if ever wanted as noise/distractor, extract to a sibling dir).

### Pre-flight diagnostic scripts: shuttle-missing diagnosis verified — 2026-04-30

Three new scripts under `src/bst_refactor/validation_scripts/`:

- `shuttle_gap_y_distribution.py` confirms the off-screen-high hypothesis at the sensor level: 61.6% of gap boundaries cluster in the top 10% of the broadcast frame, 72.3% on the post-gap re-appearance side.
- `shuttle_gap_length_distribution.py` shows the inpaint module isn't being exceeded (only 1 gap >60 frames in 32k clips); 85% of missing-shuttle frames sit in the 11-60 frame band of "shuttle genuinely not in any pixel".
- `perclass_shuttle_miss_vs_f1.py` returns Pearson **+0.516** (Spearman +0.415), opposite of the predicted direction. High-shuttle-miss classes are the pose-distinctive serves / clears / lobs at F1 ~0.95-0.99; bottleneck classes (wrist_smash, drive, push, cross_court_net_shot) sit at sub-1% miss rates.

Combined diagnosis: **shuttle data is reliably present where it's most needed; the model just isn't using it well.** Mask-channel arm gets demoted; trajectory extrapolation flagged as a longer-term direction for the off-screen-arc gaps. Label smoothing becomes the highest-priority loss-side experiment.

### Loss-side ablation arm — 2026-05-01 to 2026-05-02

Sequential 5-serial sweeps on combo A nosides + split_v2 + dropunk; each step's outcome gated the next.

**LS sweep**:
- LS=0.0 (`run_20260430_213933`, S2 best by ws 0.404): mean macro 0.743 / min 0.359 / acc 0.768 / top-2 0.939. vs LS=0.1 baseline (`run_20260430_170325`) head metrics flat (+0.001), mean wrist_smash drops 1.6 pp. **Hypothesis "LS=0.1 was taxing rare-class confidence" disproved.**
- LS=0.15 (`run_20260501_073430`, S3 best by ws 0.448): mean 0.747 / 0.417 / 0.769 / 0.938. macro +0.005, **min +4.2 pp**, head metrics flat. Wrist_smash range tightens 0.159 → 0.066 and the entire distribution shifts above the LS=0.1 mean. **LS=0.15 wins.**
- LS=0.05 skipped (two bracketing data points sufficient); LS=0.2 deferred behind class-weighting / focal experiments.

**Class-weighting smoke test**: `run_20260501_110525` (LS=0.15 + `class_weights={'wrist_smash': 2.0, 'smash': 2.0}`, S2 best by ws). Mean macro 0.748 / min 0.422 / acc 0.770 / top-2 0.936. vs LS=0.15 alone the central-tendency shift was essentially zero (+0.001 macro / +0.005 min / +0.001 acc / -0.002 top-2). But: **S2 wrist_smash 0.518 = new project-wide ceiling** across all 25 nosides serials (prior best 0.46 LS=0.1 S5; first nosides serial to clear 0.50). S4 set new ceilings on macro 0.756, accuracy 0.777, drive F1 0.66. Bimodal seed distribution: one seed found a wrist_smash basin no prior nosides serial accessed; three others stayed in the LS=0.1-baseline range around 0.37-0.40. **Read: static reweighting moves the ceiling, not the mean. The loss-side axis is not exhausted.** Implementation: `bst_train.py` `class_weights` Hyp field + renormalised CE branch at `bst_train.py:301-`.

**Loss-side decision: skip basic focal, jump straight to CDB-F1.** Vanilla focal `(1-p_t)^γ * -log(p_t)` and manually-alpha focal are the same lever as class-weighted CE, just per-sample-gated. Adding `(1-p_t)^γ` on top of pair-balanced 2.0/2.0 alpha would hit the same central-tendency ceiling. CDB-F1 (per-class alpha = `(1 - F1_c)^τ` driven by EMA of train F1, optionally composed with focal `(1-p_t)^γ`) is structurally the right next escalation: low-F1 classes get persistently escalated weight, which can push bad seeds toward the wrist_smash basin S2 found. Design at `scratch/architecture_notes/class_f1_focal_design.md` (verified against ACCV 2020 paper). Companion Seesaw-loss-style design at `scratch/architecture_notes/seesaw_f1_focal_design.md` (verified against CVPR 2021 paper) held as targeted second arm.

**CDB-F1 first run**: `run_20260501_164658` (LS=0.0 + `adaptive_focal{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0}`, S2 best). Mean macro 0.7432 / min 0.4621 / acc 0.7617 / top-2 0.9351. vs class-weighted: macro -0.5, **min +4.0**, acc -0.8, top-2 -0.1. vs LS=0.1 baseline: macro +0.1, **min +8.7**, acc -0.6. **Largest floor lift on wrist_smash so far in any loss-side run**; range tightens to 0.413-0.486 (vs class-weighted 0.378-0.518), so the bimodal-seed problem is solved. Project ceiling not broken (S2 0.486 < class-weighted S2 0.518). Per-class shifts vs class-weighted: ws +4.0, **push +6.7** (adaptive picked it up as a second bottleneck the static config missed), smash -5.5 (pair-confusion with ws), rush -2.7, drive -1.4, passive_drop -1.4, drop -1.1, lob -1.0, services -0.9 to -1.4.

Implementation shipped: new module `src/bst_refactor/stroke_classification/main_on_shuttleset/loss/adaptive_focal.py` (~190 lines, `AdaptiveFocalLoss` + `per_class_f1_from_counts` + `accumulate_class_counts`); 6 edits in `bst_train.py`; 36 unit tests at `tests/test_adaptive_focal.py`. With `adaptive_focal=None` the legacy class_weights / LS path is bit-identical. Best weight at `experiments/run_20260501_164658/weights/...nosides_2.pt`.

**CDB-F1 follow-ups**:
- gamma=0 (`run_20260501_192113`, S1 best) and tau=0.5 (`run_20260501_192519`, S2 best): both lose 2.8-5.0 of the ws lift, smash recovers in both, macro doesn't move much. **The wrist_smash gain came from both the aggressive tau=1 alpha and the gamma=1 modulator together; soften either knob and ws drops back.**
- pair-cap (`run_20260501_230252`, S4 best): capped `alpha[smash] / alpha[wrist_smash] >= 0.7` after standard renormalisation. Smash recovered 1.2 of the 5.5 it lost in the first run, but ws gave back 5.2 of its +4 lift. Macro and accuracy stayed flat because the trade cancelled out.
- gamma=2 (`run_20260502_075808`, S3 best, the focal-literature default from RetinaNet): traded 4.1 of ws for 1.3 of smash with macro -0.7 and acc -0.6 vs the first run.

Together with the gamma=0 / tau=0.5 follow-ups, every CDB knob has now been run. The original tau=1 + gamma=1 combo is the floor-lift sweet spot. **Smash drop is structural pair-confusion that scalar-per-class alpha can't resolve regardless of how its tau or gamma get tuned.** No CDB run breaks the val/test plateau at 0.74-0.75 macro. Full per-class trajectory at `scratch/architecture_notes/train_val_test_split_analysis.md`.

### Unknown ghost channel removed (architectural fix) — 2026-05-01

BST head dim now matches the empirically present classes in train `labels.npy` (val/test asserted as subsets), derived at first serial via `bst_common.derive_active_classes_from_labels`. Architecture is a function of the data, not a flag; no `drop_unknown`-driven guard. Manifest `extra.arch` records `n_classes_full`, `n_active_classes`, `has_unknown`, `unknown_first`, `active_class_list`.

**Pre-fix runs** (LS runs `run_20260430_170325`, `run_20260430_213933`, `run_20260501_073430`, plus class-weighted `run_20260501_110525`) carry a 15-channel head with the unknown slot as a ghost output channel. **Post-fix**, dropunk runs on `une_merge_v1`, `une_merge_v1_nosides`, and `raw_35` collapse to 28- / 14- / 34-class heads. `merged_25` dropunk keeps its 25-class head because `MERGE_MAP['driven_flight']='unknown'` actively populates that slot at writer time.

Pre-fix v1/nosides/raw_35 weights are no longer mechanically resumable post-fix (`load_state_dict` shape mismatch); inference still works if the caller passes `n_active_classes=taxonomy.n_classes, active_class_list=taxonomy.class_list()` explicitly. Comparisons against pre-fix v1/nosides/raw_35 carry an architectural-era boundary caveat; merged_25 dropunk is directly comparable across the boundary. Resume now auto-backs up `manifest.yaml` to `manifest.yaml.<timestamp>.bak` before the serial-1 rewrite, so the original is always recoverable.

End-to-end smokes `run_20260501_151131` (nosides dropunk, 14-class head, no unknown entry) and `run_20260501_152835` (merged_25 dropunk, 25-class head matching `run_20260429_202144`) both passed.

### Capacity-bottleneck research — 2026-05-02

Full writeup at `scratch/architecture_notes/model_capacity_bottleneck_question.md` answering "are we at the useful parameter ceiling?" with theory plus reference-class numbers.

Short version: BST at 1.85M params on 32K clips sits in the converged 1-3M zone for skeleton-AR (ST-GCN 1.22M through PoseConv3D 2.0M, MS-G3D 2.8M, ST-TR 3.07M, Hyperformer 2.6M), at the high end of per-sample density relative to peers (~81 params/sample). Famous small video AR baselines that train from scratch (X3D-M 3.76M, MoViNet-A0 3.1M) are 1.5-2x BST; flagship transformers (Video Swin-T) need ImageNet pretraining. From-scratch video-transformer literature (VideoMAE) points at pretraining as the small-data lever, not from-scratch widening. Train-test gap concentrates on smash / ws (14-18 pp); pose-distinctive classes (services, clear, net_shot) generalise within 1-2 pp at 0.95+ test F1.

**Plateau looks data-bound and signal-bound, not capacity-bound.** Theory doesn't argue against three directions: better signal (X3D-S fusion adds information pose-2D throws away on the wrist-vs-full-smash distinction), classifier-side decoupling (Kang cRT/LWS), augmentation that perturbs player cues. Pure widening unlikely to break the plateau; expected gain on test macro is 0-2 pp.

### Capacity-bump Run 1: mlp_head 400→1200 — 2026-05-03

Surgical, classifier-side only. Mechanism: one-line swap at `bst.py:199` from `d_model * mlp_d_scale` to `head_dim * mlp_d_scale`, applying the FFN-block 4x ratio to the actual head input (300 on the CG/AP path) rather than to d_model. Hidden went 400 → 1200; param cost on the head MLP ~127k → ~377k; encoder untouched. Settled on 1200 over the earlier 768 candidate because 1200 = 4x the head's actual input has architectural meaning.

Config: combo A nosides + LS=0.0 + CDB-F1 (tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0). Direct parity test vs `run_20260501_164658`.

Result (`run_20260503_104300`): mean macro 0.7414, min 0.4138, acc 0.7604, top-2 0.9320. **Best S1 by ws** (0.4449 / macro 0.7434 / acc 0.7570 / top-2 0.9365). vs y1t1 mean: macro -0.2, min -4.8, acc -0.1, top-2 -0.3. Head metrics flat; ws cost 4.8 pp on the mean. S1 ws sits below y1t1 mean ws (0.4622), so even the best seed doesn't reach y1t1's average.

Per-class shifts vs y1t1 mean (pp): smash +1.4, passive_drop +1.9, ccn_shot +1.6, long_service +1.2; ws -4.8, push -3.6, drive -1.3. Net_shot, return_net, lob, clear, drop, rush, short_service all within 0.7 pp. Bigger head traded ws and push down for smash up: same pair-confusion direction as the pair-cap and gamma=2 follow-ups, larger on ws than either.

Wrist_smash range: 0.396-0.445 (range 0.050) vs y1t1's 0.413-0.486 (range 0.073). Range tightens but the whole distribution shifted down. Smash range here 0.580-0.642 (range 0.062), slightly wider than y1t1's 0.045.

**Read.** Bigger head doesn't move the head metrics and costs ws. Lines up with the capacity-bottleneck research read: plateau is data-bound and signal-bound, not capacity-bound. The earlier scrapped `run_20260503_063338` (gamma=2 leaked from a stale bourbaki checkout) gave the same flat-head-metrics + ws-cost shape against gamma=2 as this gives against y1t1, so the capacity flat-line repeats across two adjacent loss configs.

**The mlp_head swap has been reverted at `bst.py:202` back to `d_model * mlp_d_scale`.** Keeps the baseline clean for Run 2 and any subsequent capacity work; revisit the override path (e.g. an explicit `head_hidden_dim` kwarg on `BST.__init__`) when there's a reason to set head hidden independent of d_model.

### Capacity-bump Run 2: d_model 100→192 + d_head trim 128→32 — pending

Encoder-side widening. `d_model` rises from 100 to 192 (+92% on the residual stream); `d_head` trims from 128 to 32 in the same change so the 7.68x d_head:d_model over-provisioning doesn't propagate further. Voita's WMT pruning result (38/48 heads removable for 0.15 BLEU drop) says the existing per-head allocation is over-provisioned; trimming d_head while widening d_model rebalances toward d_model carrying more.

Run 1's flat result weakens but doesn't void the prior here: encoder-side capacity is a structurally different intervention from head-side. The pair-confusion failure mode is representation-bound (the encoder isn't separating smash from wrist_smash), and a wider head couldn't fix it (Run 1 confirmed). Run 2 is at least topologically positioned to address pair-confusion more directly because it widens the representation that's failing to separate the pair. Whether 1.92x is enough to actually move separation is the open question.

The d_model bump propagates through TCN, cross-transformer, interactional transformer, and FFN; that coupling is real and the param accounting compounds. d_head trim eats some of it back. Per-epoch wall-time ballpark: +30-60% vs y1t1.

Config: same loss config as Run 1 plus the d_model and d_head changes. Direct comparison vs `run_20260501_164658`. Implementation surface, verifications-before-launch, and LR-schedule notes in `scratch/architecture_notes/transformer_widening_hparam_changes.md`.

**After this**: if Run 2 lifts macro 1+ pp without burning ws, capacity has a small lever and we follow with a joint d_model + mlp_head run on combo B for cross-taxonomy validation. If Run 2 also flat-lines (the prior after Run 1), the capacity question is empirically answered as well as theoretically: data and signal are the bottleneck, and X3D-S fusion is the right next thing.

### Frame-zeroing redesign: shuttle-unzeroing wipe-drop — 2026-05-03

Motivation: a per-class shuttle-zeroing audit run during the capacity-bump exploration showed shuttle loss concentrating in high-arc classes (long_service 24.7%, smash 13.7%, clear 11.9%, lob 9.1%, wrist_smash 8.5%, drop 8.1% whole-clip means against sub-2% for the rest of the classes), driven by the bird leaving the top of the broadcast frame on high setups (per `validation_scripts/shuttle_gap_y_distribution.py`, 61.6% of gap boundaries cluster in the top 10% of the frame). Within the high-arc family, smash↔ws is the only bottleneck pair, so it's the only one the model leans on shuttle to disambiguate; the high-F1 high-arc classes (services / clear / lob) ride pose alone and the asymmetric wipe doesn't move them. The collation step (`prepare_train_on_shuttleset.py:866-867`) was wiping `shuttle[failed,:] = 0` whenever any pose slot failed for a frame — an asymmetry that didn't exist in the other direction (pose flowed through unchanged on shuttle-fail frames). Roughly 14k frames (0.84% of extract) carried this asymmetric wipe.

Branch `shuttle/wipe-drop` removed the line. No other change. Single A/B against the loss-side best (`run_20260501_164658`).

Run `run_20260503_172922` (combo A nosides + LS=0.0 + CDB-F1{tau=1, gamma=1, momentum=0.9, warm_up_epochs=5, f1_floor=0} + `ablation_id=wipe_drop`):

| Serial | macro | min ws | acc | top-2 | smash F1 | ws F1 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 0.7452 | 0.4462 | 0.7637 | 0.9374 | — | — |
| S2 (best) | 0.7559 | 0.4935 | 0.7684 | 0.9334 | 0.610 | 0.494 |
| S3 | 0.7430 | 0.5044 (top ws) | 0.7608 | 0.9348 | — | — |
| S4 | 0.7461 | 0.4561 | 0.7684 | 0.9307 | — | — |
| S5 | 0.7504 | 0.4706 | 0.7653 | 0.9403 | — | — |
| **Mean** | **0.7481** | **0.4742** | **0.7653** | **0.9353** | | |

Vs first CDB-F1 run (`run_20260501_164658`) mean (0.7432 / 0.4621 / 0.7617 / 0.9351): macro **+0.5**, min ws **+1.2**, acc **+0.4**, top-2 flat. **New project-best mean across all arms.** S2's smash 0.610 + ws 0.494 = pair-sum 1.104 is also a project run-high.

Per-class shifts vs first CDB-F1 (5-serial mean, pp absolute): passive_drop +3.2, smash +1.5, wrist_smash +1.2, rush +0.9, cross_court_net_shot +0.5, lob +0.3, short_service +0.1, long_service +0.1, net_shot +0.2, drop +0.2, return_net -0.2, drive ~0, push -0.6, clear -0.5. **Both bottleneck-pair members lifted simultaneously**: first single intervention to do that without trading them off.

**Read.** Earlier "data quality is not the bottleneck" framing (post Phase 1 mixed retrain + per-class frame-zeroing audit) was right about the keypoint-extraction side but missed the shuttle-side collation asymmetry. The wipe-drop is now the active baseline data path; carries forward into Capacity-bump Run 2 and any subsequent runs.

The train loop has no breadcrumb for this change; `ablation_id=wipe_drop` is the only signal that the shuttle-unzeroing collation was loaded. Flag explicitly when comparing across runs.

### Mask-channel variant 2a (`shuttle_missing`) — 2026-05-03

Variant 2a from the frame-zeroing redesign: TrackNet `Visibility=0` saved as `shuttle_missing.npy` and fused into the shuttle stream post-TCN via `mask_proj` (`Linear(1, 4)`) + `shuttle_fuse` (`Linear(d_model + 4, d_model)`). Carries the wipe-drop change plus the new mask channel + fusion module. Hparams otherwise identical to `run_20260503_172922`. Branch `shuttle/mask-wiring`, ablation_id `mask_wiring`.

Run `run_20260503_192718`:

| Serial | macro | min ws | acc | top-2 |
| --- | --- | --- | --- | --- |
| S1 | 0.7491 (top) | 0.4400 | 0.7687 (top) | 0.9369 |
| S2 | 0.7388 | 0.4661 | 0.7573 | 0.9353 |
| S3 | 0.7477 | 0.4154 (low) | 0.7668 | 0.9357 |
| S4 (best) | 0.7456 | 0.4899 (top ws) | 0.7646 | 0.9391 (top) |
| S5 | 0.7390 | 0.4724 | 0.7577 | 0.9355 |
| **Mean** | **0.7440** | **0.4568** | **0.7630** | **0.9365** |

Vs shuttle-unzeroing run (`run_20260503_172922`) mean (0.7481 / 0.4742 / 0.7653 / 0.9353): macro -0.4, min -1.7, acc -0.2, top-2 +0.1. **Mask channel did not add signal on top of just unzeroing the shuttle.** Smash held flat, ws gave back 1.7% on average.

Likely two compounding causes: the model was already inferring missing-shuttle from shuttle xy + temporal context (mask is mostly redundant), and the new `shuttle_fuse` layer absorbs some learning budget for a near-identity solution on the original 100 dims.

**Decision: shuttle-unzeroing alone is the keeper. Mask channel variant 2a parked.** Variant 2b (`pose_missing_either_slot`) never built; held in case a future change makes pose-fail signal worth surfacing explicitly.

Per-class shifts vs shuttle-unzeroing (5-serial mean, pp absolute): clear +0.6, push +0.2, smash +0.1, drive +0.1, long_service +0.1, net_shot ~0, short_service -0.1, drop -0.1, cross_court_net_shot -0.1, return_net -0.3, lob -0.4, passive_drop -1.4, wrist_smash -1.7, rush -2.7.

### Augmentation set locked — 2026-05-04

Active set: centreline flip (p=0.5, coupled, COCO bilateral joint-index swap) + corrected pos+shuttle constrained-jitter (p=0.2 nominal, ±0.05y/±0.10x cap, layered conditional bounds, joints/bones untouched, zero-frame preservation, shuttle off-screen mirroring). Replaces the broken `RandomTranslation_batch` (joints-only, decoupled, body-deforming). Out for Task 2: temporal speed jitter (Phase 3 candidate), Gaussian joint jitter, random joint masking, `WeightedRandomSampler`, net flip.

Hit-frame metadata derivable without re-extraction via Method A (`clips_master.csv` correlation, faithful to annotation) or Method B (shuttle horizontal-velocity sign reversals, independent verification, ±5-frame ceiling on soft shots which X3D-S's ±19-frame window absorbs comfortably).

Full spec, implementation outlines, magnitude/frequency rationale, ablation gates, code traces, and Phase 3 candidates at [`augmentation_framework.md`](augmentation_framework.md).

### Jitter-off ablation: `RandomTranslation_batch(prob=0.0)` — 2026-05-04

The augmentation framework lock left a "first aug ablation slot" question open: was the inherited bbox-centric jitter (joints shifted in their bbox-centre-relative frame, body-deforming, not court-position-aligned) net-negative or noise vs the active baseline? Single A/B against the wipe_drop best (`run_20260503_172922`): `RandomTranslation_batch(prob=0.0)` at `bst_train.py:375`, hparams otherwise identical.

Run `run_20260504_152529`:

| Serial | macro | min ws | acc | top-2 |
| --- | --- | --- | --- | --- |
| S1 | 0.7474 (top) | 0.4408 | 0.7639 (top) | 0.9403 (top) |
| S2 (best) | 0.7398 | 0.4848 (top ws) | 0.7584 | 0.9348 |
| S3 | 0.7447 | 0.4722 | 0.7601 | 0.9398 |
| S4 | 0.7369 | 0.3911 | 0.7570 | 0.9343 |
| S5 | 0.7317 | 0.3617 (low) | 0.7535 | 0.9334 |
| **Mean** | **0.7401** | **0.4301** | **0.7586** | **0.9365** |

Vs wipe_drop best mean (0.7481 / 0.4742 / 0.7653 / 0.9353): macro **-0.8**, min **-4.4**, acc **-0.7**, top-2 +0.1. Vs first CDB-F1 baseline mean (0.7432 / 0.4621 / 0.7617 / 0.9351): macro -0.3, min **-3.2**, acc -0.3, top-2 +0.1. Wrist_smash takes the brunt (mean 0.4742 → 0.4301; S4/S5 floor at 0.39 / 0.36, well below the [0.45, 0.50] band the jitter-on serials sat in). S2 best on pair-sum 1.225 (top ws); S1 wins macro / acc / top-2 but its min sits run-low among the upper three.

Per-class shifts vs wipe_drop best (5-serial mean, pp absolute): smash +0.5, drive +0.5, return_net -0.1, clear -0.1, long_service -0.2, lob -0.4, net_shot -0.5, drop -0.5, push -0.7, rush -0.7, cross_court_net_shot -1.1, short_service -1.5, passive_drop -2.1, wrist_smash -4.4.

**Read.** The bbox-centric jitter is conceptually wrong (deforms the body around its own centre, doesn't simulate court-position movement) but empirically regularising. Min F1 ends up below the CDB-F1 baseline too, not just below wipe_drop, so the broken-but-helpful jitter isn't even close to net-negative. Defaults restored at `bst_train.py:375`; corrected pos+shuttle jitter from [`augmentation_framework.md`](augmentation_framework.md) is the eventual replacement. The "first aug ablation slot" three-option menu (Remove / Couple-current / Couple-tighten) collapses on the result: Remove is the loser arm, so the next aug experiment takes the corrected formulation rather than the disable-and-see route.

## Parked decisions

### Vanilla focal skipped per project decision

Class-weighting smoke test result revoked the gating: static reweighting hit the central-tendency ceiling that vanilla / manually-alpha focal would also hit (same lever, just per-sample-gated). Adding `(1-p_t)^γ` on top of pair-balanced 2.0/2.0 alpha would hit the same central-tendency ceiling. Skipped in favour of going straight to CDB-F1.

### Swap val and test (cross-player generalisation reframe)

Followup from the player-overlap analysis (`scratch/research/class_player_split_overlap_exploration.md`): val and test in `split_v2` have very different player-overlap profiles with train. Train-val sits at 55% clip-weighted overlap (val isn't held-out by player), train-test sits at 15% (test mostly is). Split was deliberately designed that way: val for early-stop, test for unbiased evaluation. So the model gets early-stopped on a signal that's partly within-player fit, then gets reported on a stricter cross-player signal.

Cheap experiment for a model genuinely focused on cross-player generalisation: flip the roles. Early-stop on the current test set (the harder signal), report final on the current val set. No re-splitting work, single training run. Predictions: early-stop fires later, best-checkpoint weights end up tuned for cross-player generalisation rather than within-player fit, and the headline number probably climbs purely because reporting is now on a looser distribution.

Two reasons not to do it now:
- Already a long way down the loss-side path with the current val/test convention; swapping mid-stream breaks comparability with everything in `nosides_runs_table.md`.
- Val is much smaller than test on a per-class basis. Reporting on val for the rare classes (wrist_smash 331, long_service 33, smash 299) would push the bottom-cluster headline numbers into a regime where one or two clip idiosyncrasies move the per-class F1 by several pp. Current arrangement uses the larger pool (test) for the headline.

Worth doing on a future generalisation-focused architecture or as a reporting addendum once the active loss-side / augmentation arms close. If we do swap, headline reporting still wants to land on the same fixed test set across runs to keep comparability; only the early-stop driver should change.

### Cross-cutting MMPose recovery routes

Two recovery routes for residual MMPose-extraction failures, both relevant to Arch 1's data quality but specced out in `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md` rather than here.

- **Homography-fail X3D-S-only rescue (Phase 2 candidate)**. For clips where the court homography itself doesn't fit (so no court coords are possible at all). Pixel-space fallback picker (largest bbox per screen-half, torso-diagonal crop sizing per X3D-S open-question 3) could feed the X3D-S stream while BST inputs stay zeroed. Needs a new metadata flag in the extract output. Parked until per-class Phase 1 residuals show whether it's worth building.
- **Gap-fill for partial-success frames**. Linear interpolation of `pos` and `joints` across short MMPose detection gaps when one slot picked cleanly and the other zeroed. Bounded to ~15-frame gaps, gated on endpoint-proximity. Explicitly NOT a fallback to sticky_anchor-rejected raw bboxes; those margins are generous enough that a rejection is diagnostic of upstream failure. New post-processing module that runs after sticky_anchor and preserves the byte-identity chain.

### Per-joint adaptive focal (Phase 3 / trimester 2)

Sketched at `augmentation_framework.md` as a low-priority Phase 3 research direction: extend CDB-F1 from per-class scalar α to per-joint × per-class weighting, letting the model focus on the joints that actually carry each class's signal (wrist for wrist_smash, hip rotation for clear/lob, etc). Architectural problem: BST has no per-joint prediction head, so per-joint loss decomposition needs either an auxiliary task or a Shapley-style attribution loop. Implementation cost non-trivial; benefit speculative.

## Cleanup backlog

### Dedup `bst_train.py` and `bst_infer.py` scaffolding

`bst_infer.py` and `bst_train.py` both carry their own copy of the `MODELS` dict, a `Task` class with `get_network_architecture`, the `pose_style` + `in_dim` arithmetic, and the dataloader setup from `preparing_data.shuttleset_dataset`. The genuinely different parts are small: `bst_infer.py` does argmax-only predictions with no metrics, and its Task has a `load_weight` instead of the cache-or-train `seek_network_weights`.

Two entry points is few enough that I'm leaving it for now. When a third arrives (Gradio backend, ONNX export, or the Arch 1 fusion pipeline once X3D-S lands), the right move is a `bst_common.py` holding `MODELS`, a base `Task`, and the shared dataloader helpers, with `bst_train.py` and `bst_infer.py` importing from it. A mirror TODO is pinned at the top of `bst_infer.py`.

## Cross-references

- `src/bst_refactor/stroke_classification/model/bst.py`: model defaults (`d_model=100, d_head=128, n_head=6`), CG/AP branches in `BST.forward`, the `CrossTransformerLayer` docstring.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`: cosine schedule and Hyp namedtuple configuration.
- `scratch/architecture_notes/tuning_thoughts.md`: broader HP strategy; Q4/Q5 here are new items it didn't cover.
- `scratch/architecture_notes/architecture_1_bst_3dcnn_racket_extension_09_April.md`: initial X3D-S fusion design doc; this section refines it.
- `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`: full sticky_anchor heuristic + recovery-routes design.
- `scratch/architecture_notes/model_capacity_bottleneck_question.md`: research-grounded read on whether widening BST is the missing lever (theory + reference-class numbers; the capacity-bump runs in this doc are the empirical confirmation).
- `scratch/architecture_notes/train_val_test_split_analysis.md`: per-run train/val/test trajectories pinning the plateau as generalisation-bound on the smash↔ws confusion pair.
- `scratch/architecture_notes/class_f1_focal_design.md`: CDB-F1 design verified against ACCV 2020 paper.
- `scratch/architecture_notes/seesaw_f1_focal_design.md`: pair-aware Seesaw-loss-style alternative design verified against CVPR 2021 paper.
- `scratch/architecture_notes/transformer_widening_hparam_changes.md`: capacity-bump Run 2 implementation surface + verifications + LR notes.
- `scratch/architecture_notes/frame_zeroing.md`: frame-zeroing redesign detail.
- [`augmentation_framework.md`](augmentation_framework.md): locked Task 2 augmentation set, code traces, Phase 3 candidates, hit-frame metadata derivation.
- [`x3d_integration_macro_plan/`](x3d_integration_macro_plan/): macro plan for X3D-S wrist-crop integration (six stages: hit-frame derivation → wrist-loss assessment → crop sizing + dominant-wrist heuristic → extraction pipeline → solo X3D-S fine-tune → fusion build), surfaces the open questions per stage; each stage gets its own sub-task. Source MD, print-tuned MD, printable PDF, and a `print_assets/build_pdf.sh` rebuild script all live in the subfolder.
- `scratch/architecture_notes/historical_bst.md`: BST paper defaults preserved for reproduction.
- `scratch/research/class_player_split_overlap_exploration.md`: train-val 55% / val-test 15% clip-weighted player overlap; informs the swap val/test parked direction.
- `scratch/research/Augmentation.pdf`: Isiah's PDF anchoring the augmentation decisions.
