# Model capacity bottleneck: open question on the macro plateau

## The question

CDB t1γ1 is the winner, though the hacky manual LS 0.15 + cw{ws,sm}=2.0 run is close. We've found real space to lift min class performance. But the dense clustering at 0.74 mean / 0.75 max best macro shows macro is our cap. 

So macro stays near-identical despite changing per-class F1 values and drastic bumps to min-F1.

Does that suggest a structural constraint: that we're limited on model capacity for the richer representation that would allow top class F1 to stay high while min F1 lifts?

## Research questions implied

- Is the macro plateau a model capacity ceiling, dataset-intrinsic label noise, or some mix?
- If capacity, where does the bottleneck live: encoder (d_model, depth) or classifier head (MLP_Head width)?
- If classifier-side, would two-stage training (Kang et al. cRT/LWS) localise it cheaply before any structural surgery?
- If encoder-side, which lever moves the needle: d_model width, n_head/d_head ratio, MLP_Head hidden, or depth?
- How much of the plateau is irreducible label noise on confusion pairs (smash↔ws, drop↔passive_drop)? Architecture 2 (the RGB-3dCNN-core model) is one independent test.
- What's the train F1 vs val/test F1 gap at the early-stop best epoch? A wide gap suggests the model has rep capacity but doesn't generalise; a narrow gap suggests capacity is genuinely capped.

## Where d_model sits in the architecture

`d_model = 100` is the residual stream width through the entire encoder. Specifically:

- TCN_pose outputs d_model channels (in_dim → 100 via two dilated 1D convs).
- TCN_shuttle outputs d_model channels (2 → 50 → 100).
- Each transformer layer (2 temporal + 1 cross + 1 interactional, 4 layers total) preserves d_model on its residual stream. Self-attention and FFN both project back to 100 every time.
- Each cls token is d_model dim.
- The MLP_Head input is `3 * d_model = 300` for the CG/AP variant (concat of p1_conclusion + p2_conclusion + shuttle_cls).

So d_model is the bottleneck on every token throughout the model. The TCN sets it; the rest of the encoder preserves it. If the rep doesn't have enough dimensions to separate 14 classes plus their confusion pairs, no transformer layer can rescue that downstream because they all live inside the same 100-dim subspace.

For reference, ViT-Tiny is d_model=192; ViT-Small is 384; BERT-base is 768. BST sits well below the bottom of the standard transformer range.

## On bumping d_head with d_model

Current shape: d_model=100, n_head=6, d_head=128, total head dim = 6 × 128 = 768. The reverse bottleneck inside attention is 100 → 768 → 100, a 7.68x expansion.

Standard transformer convention is `d_head * n_head ≈ d_model` (no expansion in the head split). Our attention is ~8x over-provisioned by that yardstick. Voita et al. 2019 (ACL, arXiv:1905.09418) found that when heads are over-provisioned, only a small fraction do the specialised work (positional, syntactic, rare-word); the rest can be pruned without serious cost. On English-Russian WMT they pruned 38 of 48 encoder heads for a 0.15 BLEU drop. So preserving BST's current 7.68x ratio when widening d_model probably isn't right: we'd be scaling up dead weight.

If we widen d_model to, say, 192, the cleaner options are:

- Standard ratio: n_head=6, d_head=32. Total head dim 192. No reverse bottleneck inside attention.
- Modest expansion: n_head=6, d_head=64. Total head dim 384, ratio 2x. Same shape ViT-Small uses.
- Preserve current 7.68x: n_head=6, d_head=246. Total head dim 1474. Probably scales up the redundancy.

I'd lean standard or 2x expansion, not preserving 7.68x. The current attention shape is unusual; widening d_model is a chance to fix it, not propagate it.

## The three MLPs

Three named MLPs, plus four FFN MLPs inside the transformer layers (those are wrapped inside `FeedForward`, sitting in each transformer layer's residual block).

| MLP | Shape | ~Params | Role | Worth investigating |
|---|---|---|---|---|
| `mlp_positions` (PPF) | 2 → 256 → 72 | ~19k | Modulate pose features by court xy before TCN | No: small, feature-prep stage |
| `mlp_clean` (CG) | 100 → 100 → 100 | ~20k | Compute "dirt" subtracted from shuttle CLS | No: annealed out by epoch 15 |
| `mlp_head` (classifier) | 300 → 400 → 14 | ~127k | Final classification | **Yes: largest of the three, sits at the decision boundary** |
| FFN (per transformer layer) | 100 → 400 → 100 | ~80k each, 4 layers | Inside each layer's residual block | Tied to d_model; widening d_model widens these too |

`mlp_head` is the clear candidate. It's the only stage where 14 class-specific decision boundaries get carved, and its hidden width (400) is the only free parameter on the classifier side. Widening it from 400 to 768 or 1024 directly buys decision-side capacity without touching the encoder. That's also the surgical version of the Kang et al. classifier-retraining argument below.

`mlp_positions` is too small and too early in the pipeline to be the bottleneck. `mlp_clean` is dormant for most of training so it's a non-target.

## Why MLP_Head expands to 400 before the logits

Same reverse-bottleneck principle as the transformer FFN: project into a wider space, apply a non-linearity (GELU), project to the next layer. The wider intermediate space gives the non-linearity room to carve decision regions; the final projection collapses to logits. The 4x ratio (400 = 4 × d_model) is BERT/ViT convention rather than a tuned choice. Concretely, MLP_Head is `LayerNorm(300) → Linear(300, 400) → GELU → Dropout → Linear(400, 14)`.

What the 300-node input takes in, decomposed:

- `p1_conclusion` (100 dim) = p1_cls (temporal CLS for player 1) + p1_shuttle_cls (interactional CLS for p1 attending shuttle)
- `p2_conclusion` (100 dim) = p2_cls + p2_shuttle_cls (same shape, player 2)
- `shuttle_cls` (100 dim) = shuttle's temporal CLS, with CG dirt subtracted in CG_AP variant

So each per-player slot is a *sum* of two 100-dim CLS tokens, not a concat. The rank of each player slot is bounded by 100, not 200. Aggregate effective rank of the 300-dim input is at most 300 if the three slots are independent, and in practice less because shuttle features feed both player conclusions (via cross-attention) and shuttle_cls directly.

Per BST paper Figure 2 (Chang 2025, arXiv:2502.21085): each per-player conclusion is `⊕` (element-wise add) of the Trans 1 player CLS and the per-player CrossTrans output, both 100-dim. The three conclusions are then concat'd into a 300-dim MLP_Head input.

The paper doesn't explicitly say "we sum because X", but the rationale is implied at two points. Section 3.3.1 states "Trans1 and Trans2 are inspired by TemPose [18]", so the fusion pattern is inherited from TemPose rather than a fresh BST choice. And equations (5)-(6) plus Figure 2 show every transformer block is the standard residual form (`X(l) = X(l-1) + MHSA(LN(X(l-1)))`, then plus FFN); the cross-attention is treated as another residual update on the player CLS, not a parallel track to be concatenated. Switching the per-player combination to concat would break the residual-stream interpretation that BST inherits from TemPose and from standard transformer convention. So the sum is the conventional choice in this lineage; the doc's earlier framing that this was an obscure BST authorship decision was wrong.

Two notes on this:

- Switching the per-player combination from sum to concat would double per-player rank to 200, giving the head a 500-dim input instead of 300. That's a structural departure from the residual-stream pattern rather than an integer bump, so out of scope for the current question.
- The MLP_Head's hidden 400 is only 33% wider than its input 300. Most classifier heads run wider expansions (4-8x). If 0.74 macro is the head saturating, going 300 → 1024 → 14 is a small parameter cost (~310k vs current 127k) for substantially more decision room.

## Kang et al. decoupling: not what we're already doing

Easy to mistake the decoupled approach for either the existing CDB sweep or a within-stage curriculum (e.g. LS=0.1 first then CDB t1γ1 second). It's neither. Kang et al.'s decoupled training is structurally different:

- **Stage 1**: train backbone + classifier together with **instance-balanced sampler** (no rebalancing on the data side, no class-weighted loss). Vanilla CE. The features learn from the raw distribution.
- **Stage 2**: **freeze the backbone**, retrain ONLY the classifier head with **class-balanced sampler** (oversample tail, undersample head). The head re-learns its decision boundary on rebalanced supervision; features stay fixed.

The key empirical finding is that features and classifiers want **opposite** training regimes: features benefit from instance-balanced (more diverse signal per class), classifiers benefit from class-balanced (uniform per-class supervision). Doing both at once with class-weighted CE or focal loss compromises both. Their decoupled approach beats joint approaches on long-tail benchmarks.

Mapped to BST, cRT (their simplest variant) would be:

1. Train BST end-to-end with vanilla CE, no class weighting, no LS, instance-balanced sampler. ~80 epochs as we do now.
2. Freeze everything except `mlp_head`. Retrain `mlp_head` with class-balanced sampling (oversample wrist_smash, smash, undersample net_shot, return_net) for 10-20 extra epochs.

What we've been running (CDB-F1, class weights, LS sweep, pair-cap) is the joint baseline their paper outperforms. None of our runs is structurally a Kang two-stage. That's the genuinely untried experiment; cheap (~10-20% extra training time on top of stage 1) and a clean test of whether the head is the bottleneck.

LWS (learned weight scaling) is an even cheaper variant: skip the retraining, just learn one scalar per class to multiply the existing classifier weights by. Single Linear layer scaling the existing head, ~14 params extra.

## Neural collapse: (a) untried, (b) addressed

Neural collapse describes how late-stage training pushes class features into an equiangular tight frame in the head's input space, with long-tail asymmetry concentrating principal directions on head classes. Two responses:

- (a) wider feature space so the collapse has room: **untried**. The integer-bump path is widening d_model. (Sum-to-concat for the player slots would also widen the head's input space, but it's a structural departure from the residual-stream pattern, not an integer bump; see §Why MLP_Head expands to 400.)
- (b) classifier-side intervention that doesn't rely on better features: **what we've been doing**. CDB-F1, class weights, LS sweep, focal modulation, pair-cap all redistribute decision boundaries on top of the existing rep. The pair-cap's failure mode (forced smash up, ws down by 5pp) is exactly this: the rep can't separate the pair, so the classifier just trades sides.

## Another layer

Stacking another vanilla transformer layer probably doesn't buy much here. Each layer is a non-linear refinement on a 100-dim residual stream; more layers give more iterations of the same compress-expand-compress shape, not a wider rep. At 4 layers (2 + 1 + 1) BST is shallow but not severely under-parameterised on depth for the task (ViT-Tiny is 12 layers, but on 1000-class ImageNet).

**Width vs depth at fixed N: shape barely matters.** Kaplan et al. (arXiv:2001.08361, §3.1):

> Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width [...] Aspect ratio in particular can vary by a factor of 40 while only slightly impacting performance
>
> [Figure 5:] an n_layer=6, d_model=4288 model reaches within 3% of an n_layer=48, d_model=1600 model

**Depth specifically helps with compositional tasks.** Petty et al. (NAACL 2024, arXiv:2310.19956) train LMs at fixed param count across the depth-width tradeoff and find:

> deeper models generalize more compositionally than shallower models do [...] even after correcting for the fact that their language modeling perplexity is lower and their in-distribution performance on the fine-tuning task is higher
>
> [On diminishing returns:] most of the benefit of depth is gained by having only a few layers

Compositional tasks reward depth (parsing, multi-step reasoning, hierarchical structure). 14-class classification rewards separability instead, which is more a width axis. So we're in the Kaplan regime where shape barely matters at fixed N rather than the Petty regime where depth wins by a few points.

The two cases where adding "a layer" actually helps are when the layer adds something the existing stack can't reach: a **new information path** (e.g. letting players attend to each other directly rather than only via shuttle: that's a new edge in the graph, not a deeper stack), or a **new modality** (e.g. the X3D-S branch in Architecture 1 fusion: raw video adds a feature axis pose-2D throws away). Neither is a vanilla extra layer; both are different beasts.

## Label noise floor and the 0.9 macro / 0.7 min target

With no domain knowledge to manually verify confusion-pair clips, the irreducible label-error rate is hard to pin down. Some structural priors:

- ShuttleSet has known label noise (this term's data-cleaning work has been partly about it).
- smash↔wrist_smash is partly a labeling convention question: what counts as "wrist" vs "full"? Annotator inconsistency probably contributes to the pair confusion.
- Original BST paper hit ~0.83 macro on the 8-class merged taxonomy. Going to 14 classes is harder; 0.9 is more ambitious than just "BST + better config".

Plausibility read: 0.85-0.90 macro and 0.7 min are achievable but require either (or both) of: better data signal on the bottleneck classes (X3D-S fusion bringing wrist-vs-full-smash detail pose-2D throws away; class-aware augmentation), and resolving label ambiguity on the worst confusion pairs. Pure parameter widening (wider d_model, wider MLP_Head) is unlikely to get there on its own at this data scale; see §Research-grounded read for the case. Loss-side methods alone definitely won't.

Architecture 2 (the RGB-3dCNN-core model) is a useful independent test: if a substantially different model also caps near 0.75 macro, the data is the dominant factor. If it breaks through, BST's specific shape is the bottleneck.

## Train vs val/test gap as the diagnostic

Reading off train F1 at the early-stop best epoch is the cleanest diagnostic for this question. The signal is:

- **Wide gap** (e.g., train macro 0.92 vs val/test 0.75): model has the rep capacity to separate, doesn't generalise. Pointing at regularisation, augmentation, or label noise rather than raw capacity.
- **Narrow gap** (e.g., train macro 0.78 vs val/test 0.75): capacity is genuinely capped. Pointing at structural widening (d_model, MLP_Head) or two-stage training.
- **Mid gap** with class-specific divergence (e.g., minority classes much wider than majority): mixed; tail classes generalise worse, possibly because their seen-instance count is too low for the rep to memorise *and* generalise.

The full analysis sits at `train_val_test_split_analysis.md` with per-run tables and curves across the ten nosides runs. **Headline read**: it's the third case. Train macro reaches 0.85-0.86 by end of training across CDB runs while val plateaus at 0.77-0.78 and test at 0.74-0.75. So the model has 6-9 pp of train headroom that doesn't transfer. Per-class breakdown shows the gap is concentrated on smash and wrist_smash (18-27 pp train-test gaps at end of training), while pose-distinctive classes (services, clear, net_shot) generalise within 1-2 pp. So the macro plateau is **generalisation-bound on the confusion pairs**, not a uniform capacity ceiling. That argues against widening d_model as a first move and toward (a) the cRT test on the existing encoder and (b) X3D-S fusion to add information that pose-2D extraction throws away on the wrist-vs-full-smash distinction.

Subsequent γ=2 / run_20260502_075808 was the last untouched CDB knob: test macro 0.74, test min 0.42, both below the γ=1 baseline. Every CDB knob has now been run. No CDB run breaks the val/test plateau. The loss-side ceiling for this architecture and data is firmly mapped.

## Research-grounded read: are we at the useful parameter ceiling?

The empirical picture from `train_val_test_split_analysis.md` and `class_player_split_overlap_exploration.md` is now: a 14-18 pp train-test gap on the confusion classes (smash, ws), pose-distinctive classes (services, clear, net_shot) generalising at within 1-2 pp, train-val 55% clip-weighted player overlap, val-test 5%. The model has rep capacity it doesn't transfer to test.

Question for this section: does double descent argue that bumping capacity past this point keeps reducing test error until we hit a useful-parameter ceiling? And can we tell from theory plus our model specs alone whether we're at that ceiling?

Three pieces of theory matter here.

**Double descent and EMC (Nakkiran 2020; Belkin 2019).**

Nakkiran et al. (ICLR 2020, arXiv:1912.02292) define Effective Model Complexity as "the maximum number of samples on which it can achieve close to zero training error" and propose three regimes:

- Under-parameterised (EMC < n): more complexity reduces test error.
- Critically parameterised (EMC ≈ n): test error peaks; perturbations can push either way.
- Over-parameterised (EMC > n): more complexity reduces test error again.

They flag that double descent shows up "most strongly in settings with label noise", studied at 5-20% label-noise levels on CIFAR-10/100 with ResNet18. They also frame label noise as "merely a proxy for making distributions harder", which lines up with our smash/ws pair confusion: pose-2D doesn't separate the pair cleanly, so the labels look noisy from the encoder's perspective even when the human label is right.

Where BST sits: train macro reaches 0.86 across 5 serials, well clear of any random-feature floor. We've crossed EMC into the over-parameterised regime by Nakkiran's definition. By their prediction, more capacity past here keeps reducing test error in principle.

Belkin et al. (PNAS 2019, arXiv:1812.11118) add two caveats that bite us. First, on regularisation:

> "Regularization, of all forms, can both prevent interpolation and change the effective capacity of the function class, thus attenuating or masking the interpolation peak."
>
> "Early stopping has a strong regularizing effect."

We have dropout, label smoothing (configurable, has been on for most runs), and early-stop on val macro. So whatever second-descent benefit exists is partly already harvested or partly washed out.

Second, on data scale:

> "For datasets as large as ImageNet... the classical regime of the U-shaped risk curve is more appropriate to understand generalization."

ImageNet is 1.28M images; we have 32,203 clips. We're well below ImageNet but still small enough that scaling-law gains from over-parameterisation are not the regime where the literature shows them most cleanly.

**Scaling laws (Kaplan 2020; Hoffmann 2022).**

Kaplan et al. (arXiv:2001.08361) for transformer language models:

> "Performance improves predictably as long as we scale up N and D in tandem, but enters a regime of diminishing returns if either N or D is held fixed while the other increases. The performance penalty depends predictably on the ratio N^0.74/D, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty." (§1.1, §4)

Their direct heuristic on data needed to avoid overfitting (eq. 4.4):

> "to avoid overfitting to within that threshold of convergence we require D ≳ (5 × 10^3) N^0.74"

Plugging BST's N ≈ 2 × 10⁶ trainable params:

```
D ≳ (5 × 10³) · N^0.74
   = 5 × 10³ × (2 × 10⁶)^0.74
   ≈ 1.7 × 10⁸ tokens
```

We're at 3.2 × 10⁴ clips. Several orders of magnitude below (their D is tokens, not clips, but the gap is clear).

Hoffmann et al. (Chinchilla, arXiv:2203.15556) sharpen the rule:

> "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled."

Chinchilla 70B is trained on 1.4T tokens (20:1 ratio). For our 32K-clip dataset that heuristic places compute-optimal N at roughly 1,600 params, ~1000x below where BST sits. Their parametric loss form is:

```
L̂(N, D) = E + A / N^α + B / D^β
```

E is the irreducible-loss term capped by data quality. Once D dominates, more N just reduces the A/N^α term to its asymptote without touching E.

These heuristics are calibrated on next-token language modelling and do not transfer cleanly to classification on pose features (a clip carries far more bits than a token, and our task is 14-way categorical, not next-token prediction). The directional read transfers (more N without more D walks into diminishing returns), but the orders-of-magnitude estimates do not. For an analogue closer to our setting we have to look at video and skeleton-based action recognition.

**Video and skeleton-based action recognition: the right reference classes.**

Our data is temporal: per-frame 2D pose for two players plus shuttle trajectory, sequence length 100 frames per clip. The right reference classes are (a) famous small video action recognition models on Kinetics-400, where the discipline of "what counts as small enough to train from scratch on a video benchmark" is well-mapped; and (b) skeleton-based action recognition on NTU RGB+D, which shares our pose-only input modality.

Architectural-family caveat: GCN models use shared graph kernels with O(d_model^2) params per layer regardless of joint count, while transformers add Q/K/V/O projections plus a wider FFN, so transformer layers are heavier per d_model than GCN layers at the same width. In principle transformers should out-param GCNs at fixed d_model. In practice the modern skeleton-AR field has converged: transformers use smaller d_model (Hyperformer 216), GCNs grow via multi-scale graph aggregation (MS-G3D, DC-GCN). Both families cluster in the 1-3M range on NTU. Plizzari et al. specifically show that swapping a TCN block for temporal self-attention reduces params by 1.34M on their ST-TR setup, so the family direction can go either way at the layer level depending on design.

Numbers across both reference classes:

| Family | Model | Params | Pretrain | Dataset | Train clips |
|---|---|---|---|---|---|
| Raw-video transformer | Video Swin-T (Liu 2022) | 28.2M | ImageNet-1K | Kinetics-400 | ~240,000 |
| Raw-video transformer | MViT-B 32×3 (Fan 2021) | 36.6M | none | Kinetics-400 | ~240,000 |
| Raw-video CNN | I3D (Carreira 2017) | 25.0M | ImageNet-1K | Kinetics-400 | ~240,000 |
| Raw-video CNN | X3D-M (Feichtenhofer 2020) | 3.76M | none | Kinetics-400 | ~240,000 |
| Raw-video CNN | X3D-XXL (Feichtenhofer 2020) | 20.3M | none | Kinetics-400 | ~240,000 |
| Mobile 3D CNN | MoViNet-A0 (Kondratyuk 2021) | 3.1M | none | Kinetics-400 | ~240,000 |
| Mobile 3D CNN | MoViNet-A2 (Kondratyuk 2021) | 4.8M | none | Kinetics-400 | ~240,000 |
| Skeleton GCN | ST-GCN (Yan 2018) | 1.22M | none | NTU 60 X-sub | 40,320 |
| Skeleton GCN | MS-G3D (Liu 2020) | 2.8M | none | NTU 60 X-sub | 40,320 |
| Skeleton transformer | ST-TR S-TR (Plizzari 2021) | 3.07M | none | NTU 60 X-sub | 40,320 |
| Skeleton transformer | Hyperformer (Zhou 2022) | 2.6M | none | NTU 60 X-sub | 40,320 |
| Pose-heatmap 3D CNN | PoseConv3D / Pose-SlowOnly (Duan 2022) | 2.0M | none | NTU 60 X-sub | 40,320 |

Numbers from: Video Swin Table 1 (Liu et al. CVPR 2022, arXiv:2106.13230); X3D and X3D-XXL from Feichtenhofer 2020 + Liu 2022 Table 1 (arXiv:2004.04730, arXiv:2106.13230); MoViNet from Kondratyuk et al. 2021 Table 9 (arXiv:2103.11511); skeleton-AR GCN/transformer from CTR-GCN Table 4, ST-TR, and Hyperformer Table 1; PoseConv3D from Duan et al. CVPR 2022 Table 3 (arXiv:2104.13586) reporting Pose-SlowOnly at 2.0M params, 15.9G FLOPs, 93.7% NTU-60 XSub vs MS-G3D 91.9% at 2.8M params.

Two reads off this.

Raw-video AR on Kinetics-400 (240K clips) clusters at ~3-30M params. The smallest models that train respectably from scratch (X3D-M at 3.76M, MoViNet-A0 at 3.1M) sit at the low end. The flagship transformers (Video Swin-T at 28.2M, I3D at 25M) need ImageNet pretraining to reach their reported numbers. So "famous, small enough to train from scratch on a video benchmark" maps to the X3D / MoViNet zone of 3-5M.

Pose-only inputs save substantial params by skipping the spatial conv stack that dominates raw-video models: skeleton-AR on NTU (40K clips) clusters at 1-3M. That's roughly the X3D / MoViNet floor scaled down by ~3-5x for the missing visual encoder.

BST sits at 1.85M trainable params on 22,743 train clips (Chang 2025 Supplementary Table A reports BST-CG-AP 1.85M, TemPose-TF 1.71M, ST-GCN 3.08M, SkateFormer 2.38M, BlockGCN 1.50M, ProtoGCN 4.11M, sequence length 100).

Per-sample density across the reference classes:

| Model | Params/sample |
|---|---|
| X3D-M | ~16 |
| ST-GCN | ~30 |
| PoseConv3D | ~50 |
| Hyperformer | ~64 |
| MS-G3D | ~70 |
| ST-TR S-TR | ~76 |
| **BST** | **~81** |
| Video Swin-T (with pretraining) | ~117 |

PoseConv3D at 2.0M is the closest direct analogue: pose-derived input (3D heatmap volume rather than graph or coordinates) on a 3D-CNN backbone (SlowOnly-R50), beating MS-G3D's 2.8M GCN on NTU-60 (93.7 vs 91.9). The pose-AR niche has converged on ~2M regardless of input encoding (graph, coordinates, heatmap volume) or backbone family (GCN, transformer, 3D CNN). BST sits squarely in that converged zone: high end of per-sample density relative to peers, well below raw-video transformer density. Not under-parameterised, not dramatically oversized.

For raw-RGB video transformers, Tong et al. (VideoMAE, NeurIPS 2022, arXiv:2203.12602) make two findings directly relevant. First: "training video transformers from scratch yields unsatisfied results", with a 37 pp gap on Something-Something v2 between from-scratch (32.6%) and MAE-pretrained (69.6%) on the same architecture. Second: "data quality is more important than data quantity for SSVP", with 42K matched-domain pretraining videos beating 240K mismatched-domain videos.

For BST: there's no Kinetics → ShuttleSet pretraining step in the current pipeline. The encoder fits from scratch on 32K clips. The video-transformer literature has converged on pretraining as the main lever for small-data action classification, not from-scratch widening, so the prior on widening-from-scratch is correspondingly weak. The data-quality finding generalises the label-noise argument: at our scale, the bottleneck on confusion classes is what the pose-2D signal carries about the wrist-vs-full-smash distinction, not how wide the encoder is.

**Attention head over-provisioning (Voita 2019).**

Covered in full at §On bumping d_head with d_model. Short version: BST's 7.68x d_head ratio is over-provisioned per Voita's WMT pruning result (38/48 heads pruned for 0.15 BLEU drop). Yet another yardstick that doesn't say "more capacity"; it says "the existing capacity isn't fully exploited."

**Net read.**

Theory does not unambiguously predict that bumping parameters breaks the plateau here.

By EMC we're past interpolation, so Nakkiran's second descent applies in principle. But Belkin's caveats mean the slope is partly already absorbed by our regularisation, and the studied gains were sharpest at data scales (CIFAR) where the rep is fed by raw pixels rather than 2D pose features.

By LM scaling laws (Kaplan, Chinchilla) we sit deep in the "much more model than data" zone, but those laws are calibrated on tokens not clips, so the magnitude doesn't transfer. By the right reference classes (pose-only skeleton-AR at 1-3M params on NTU's 40K clips; famous small video-AR with X3D-M at 3.76M and MoViNet-A0 at 3.1M as the from-scratch baselines on Kinetics-400's 240K clips), BST at 1.85M is in normal territory after granting the pose-only savings: high end of skeleton-AR density, well below raw-video transformer density. None of these reference points say "more N is the missing lever." The famous video-AR transformers (Video Swin-T, I3D) all use ImageNet pretraining; the from-scratch literature converges on pretraining transfer rather than width as the small-data lever, and BST has no pretraining stage in the current pipeline.

The bottleneck classes (smash, ws) carry a train-test gap structurally similar to label noise. Nakkiran specifically notes this is the regime where the test-error peak is sharpest. The second descent kicks in only if the smooth-interpolator inductive bias holds across the relevant features, but pose-only features apparently don't carry the wrist-vs-full-smash distinction cleanly, so widening the encoder gives the same ambiguous features more room to memorise rather than separate.

The classes that already generalise (services, clear, net_shot) sit at 0.95-0.98 test F1. No headroom for more capacity to lift them.

The player-overlap finding (train-val 55% clip-weighted, val-test 5%) means a chunk of the train-val gap closure isn't representation generalisation; it's player memorisation. More capacity makes that memorisation room larger, not smaller.

**The honest answer: we're at or near the useful parameter ceiling for this data and signal source.** The plateau looks data-bound and signal-bound, not capacity-bound. Theory can't say that with full certainty (the residual is empirical, sensitive to data quality), but every yardstick lines up: pose-only skeleton-AR and famous small video-AR sit around or above BST's parameter range, EMC plus our regularisation says the second descent is shallow, the ceiling classes are saturated, head over-provisioning means we're already carrying slack, and the from-scratch video-AR literature points at pretraining rather than width as the small-data lever. None of these point to "more parameters is the missing lever".

The directions theory does not argue against, listed for completeness rather than as new proposals: better data signal (X3D-S fusion brings information that pose-2D throws away on the wrist-vs-full-smash distinction), classifier-side decoupling (Kang et al. cRT/LWS), augmentation that perturbs player-specific cues so the encoder can't ride them.

**If we want to bump integers anyway: the smallest-risk options.**

Even granting the above, there's a defensible small-risk capacity sweep. None of the options below is predicted to break the macro plateau; expected gain on test macro is 0-2 pp, and on a small dataset some of these may regress through more aggressive over-fit.

| Lever | Current | Bump candidate | Notes |
|---|---|---|---|
| `d_head` × `n_head` shape | 128 × 6 = 768 (7.68x d_model) | 32 × 6 = 192 (1.0x d_model) | Voita-style trim, not bump. Reclaims param budget without performance loss. |
| `d_model` | 100 | 128 or 192 | ViT-Tiny is 192. Widens FFN automatically (FFN params scale ~d_model^2). Cheap test of whether residual stream width is the cap. Pair with d_head=32 if going to 192 so the 7.68x ratio doesn't propagate. |
| `mlp_head` hidden | 400 | 1200 (= `head_dim * mlp_d_scale`; **shipped 2026-05-03**) | Decision-side widening only, encoder untouched. ~377k params (was ~127k). Most surgical test of whether the classifier head specifically is the local bottleneck (Kang-flavoured intervention without the full two-stage). 1200 picked over 768 because it's 4x the head's actual input (300), matching the FFN expansion ratio; 768 had no relationship to the head shape. |
| FFN ratio inside transformer layers | 4x (400 = 4 × d_model) | unchanged | BERT/ViT convention; no theoretical case for changing. |

The lowest-risk standalone test is `mlp_head` hidden 400 → 1200, via the `head_dim * mlp_d_scale` swap at `bst.py:199`: surgical, classifier-side only. If also widening `d_model`, the d_head trim to 32 belongs in the same change so the 7.68x ratio doesn't carry through.

(Run launched 2026-05-03; results due later today. Live status in `bst_x_overview.md` 2026-05-03 block.)
