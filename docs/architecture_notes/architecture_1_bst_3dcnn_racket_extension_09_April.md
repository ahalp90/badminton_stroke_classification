# Extending BST with a video crop sub-model for racket motion

**X3D-S is the optimal 3D CNN backbone for a wrist-centered video crop stream in badminton stroke classification.** At just **3.8M parameters and under 1 GFLOP** on a 112×112 input, it adds negligible overhead to the existing BST skeleton transformer while capturing the racket manipulation and forearm torsion that keypoints miss. Foundation models are a poor fit here — their patch-based tokenization produces only 36 spatial tokens on a 96×96 crop, destroying fine-grained detail. The practical path is: extract wrist crops using skeleton-guided adaptive sizing, process them through a pretrained-then-finetuned X3D-S, and fuse features with the BST via late fusion initially, upgrading to MMTM or bottleneck fusion once the baseline is established.

---

## X3D dominates the lightweight 3D CNN landscape

Among all candidate architectures, **X3D-S** offers the strongest efficiency-accuracy tradeoff by a wide margin. It was explicitly designed as an efficient expansion of the SlowFast Fast pathway concept, using depthwise-separable 3D convolutions, squeeze-and-excitation blocks, and swish activations. At its native 160×160 resolution with 13 frames, it achieves **73.1% top-1 on Kinetics-400** with only **1.96 GFLOPs**. Downscaling to 112×112 drops this to roughly **0.96 GFLOPs**, and 96×96 pushes it below **0.72 GFLOPs** — all while maintaining the same 3.8M parameter count.

MoViNet-A0 (3.1M params, ~0.3 GFLOPs at 112×112) is a strong alternative with a unique streaming inference mode for real-time processing, but its official implementation is TensorFlow-only, with PyTorch ports being unofficial. R(2+1)D-18 is readily available via `torchvision` but carries **31.5M parameters** and ~40 GFLOPs at its native 112×112 — nearly 40× the compute of X3D-S for lower accuracy. The SlowFast Fast pathway alone achieves only 51.7% on K400, confirming it was never intended for standalone use.

Transformer-based models are systematically too heavy for this sub-component role. Video Swin-T (28M params, ~11 GFLOPs at reduced resolution), MViTv2-S (35M params, ~16 GFLOPs), and TimeSformer (121M params, ~49 GFLOPs) all carry 10–50× the computational overhead of X3D-S. With only ~25k training samples, their large parameter counts also create serious overfitting risk. **CNN architectures with strong inductive biases — locality, shift equivariance, hierarchical feature extraction — outperform transformers when data is limited and spatial resolution is small.**

| Model | Params | GFLOPs @112² | K400 Top-1 | PyTorch weights | 25k fine-tune |
|-------|--------|-------------|-----------|----------------|--------------|
| **X3D-XS** | **3.8M** | **~0.29** (4f) | 68.7% | ✅ PyTorch Hub | ✅ Excellent |
| **X3D-S** | **3.8M** | **~0.96** (13f) | 73.1% | ✅ PyTorch Hub | ✅ Excellent |
| MoViNet-A0 | 3.1M | ~0.30 (16f) | 71.5% | ⚠️ Unofficial | ✅ Excellent |
| MoViNet-A1 | 4.6M | ~0.65 (16f) | 76.0% | ⚠️ Unofficial | ✅ Good |
| R(2+1)D-18 | 31.5M | ~40.5 (16f) | 57.5% | ✅ torchvision | ⚠️ Overfit risk |
| Video Swin-T | 28M | ~11 (16f) | 78.8% | ✅ mmaction2 | ⚠️ Moderate |
| MViTv2-S | 35M | ~16 (16f) | 81.0% | ✅ PySlowFast | ⚠️ Heavy |
| TimeSformer | 121M | ~49 (8f) | 78.0% | ✅ HuggingFace | ❌ Too large |

X3D is fully convolutional with adaptive pooling, so it handles non-standard spatial resolutions (96×96, 112×112) natively — no patch embedding interpolation required. Load pretrained weights with a single line: `torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)`.

---

## Foundation models fail on tight spatial crops

Video foundation models like VideoMAE, InternVideo, and Hiera are designed for whole-scene understanding at 224×224 resolution. Applying them to a 96×96 wrist crop creates a fundamental mismatch across three dimensions.

**Token sparsity kills fine-grained discrimination.** VideoMAE uses 16×16 spatial patches. On a 96×96 crop, this yields only **(96/16)² = 36 spatial tokens per frame** — each patch covers roughly 17% of the crop width. For a task requiring discrimination of subtle grip angles and racket face orientation, this resolution is catastrophically coarse. At 224×224, the same model gets 196 spatial tokens per frame — 5.4× more spatial granularity. Scaling the crop up to 224×224 via interpolation doesn't recover lost information; it just blurs the already-small region.

**Positional embedding mismatch degrades transfer.** Pretrained ViT positional embeddings encode a 14×14 spatial grid. Interpolating these to a 6×6 grid distorts the learned spatial relationships, and there is no reliable way to recover them without full retraining. CNNs, by contrast, have no fixed spatial resolution — their convolutional filters work identically at any input size.

**Computational overhead is prohibitive for a sub-component.** Even the smallest practical foundation model, VideoMAE ViT-S (22M params), requires an estimated **8–12 GFLOPs** at 96×96 — 10× more than X3D-S while delivering representations pretrained on coarse scene-level actions (Kinetics classes like "playing tennis" encode zero stroke-level information). InternVideo2's smallest distilled variant adds similar or greater overhead, and its 14×14 patch size doesn't even divide 96 evenly.

**The practical verdict:** a Kinetics-pretrained X3D-S, fine-tuned end-to-end on ~25k wrist crop samples, will almost certainly outperform a frozen or fine-tuned foundation model on this task. If foundation model knowledge is desired, the most viable path is **offline knowledge distillation** — run VideoMAE ViT-B on full-resolution frames to generate soft labels, then train X3D-S on crops using those targets. This captures scene-level context without inflating runtime cost. However, given that the BST skeleton stream already provides whole-body context, the marginal value of this distillation is questionable.

---

## Fusion strategy should start simple, then upgrade to MMTM

Combining the BST skeleton transformer with an X3D-S video crop stream requires a fusion strategy that handles architectural heterogeneity (transformer tokens vs. CNN feature maps) while preserving pretrained weights. Research across sports action recognition strongly suggests a progressive approach.

**Late fusion is the correct starting point.** Extract the BST's penultimate feature vector and X3D-S's global-average-pooled features, L2-normalize both, concatenate, and pass through a 2-layer MLP with dropout. This approach requires zero modification to either encoder, allows independent pretraining of each stream, and provides a strong baseline. The recommended training protocol is: (1) freeze BST and train X3D-S independently with its own classification head, (2) freeze both encoders and train only the fusion MLP, (3) fine-tune everything end-to-end with 10× lower learning rate for the encoders. Weight the skeleton stream's scores higher initially (~0.6–0.7), since skeleton-based models typically achieve higher standalone accuracy for fine-grained sports actions.

**MMTM is the recommended upgrade.** The Multimodal Transfer Module (Joze et al., CVPR 2020) uses squeeze-and-excitation mechanics to enable mid-network cross-modal interaction. Each stream's features are globally pooled, concatenated, passed through a shared FC layer, then split into per-stream channel-wise excitation signals. This module handles different spatial dimensions naturally — ideal for transformer features versus CNN feature maps. It adds minimal parameters, preserves pretrained weights, and has proven effectiveness on skeleton+RGB tasks (NTU RGB+D). Code is available at `github.com/haamoon/mmtm`.

**A critical finding from RacketVision (2025)** — a multi-sport benchmark covering badminton, tennis, and table tennis — showed that **naive concatenation of racket pose features actually degraded performance** versus a unimodal baseline. Only when cross-attention was applied did multi-modal fusion unlock gains. This confirms that going beyond simple late fusion is worthwhile for racket sports, but the upgrade path should be incremental.

**Bottleneck fusion tokens offer the best accuracy-to-compute ratio** for advanced fusion. The Multimodal Bottleneck Transformer (MBT, NeurIPS 2021) introduces 4 learnable bottleneck tokens that mediate information flow between modalities. Each stream's tokens attend freely to their own tokens plus bottleneck tokens; bottleneck tokens attend to both streams. This adds negligible computation while outperforming vanilla cross-attention by 2+ mAP points at less than half the cost. Implementing this requires modifying BST slightly to insert cross-attention layers with bottleneck tokens at mid-network depth.

Avoid full cross-attention between heterogeneous architectures (quadratic cost, awkward alignment of transformer tokens to CNN features) and Mixture-of-Experts (overkill for two modalities, complex training dynamics).

| Fusion method | Implementation effort | Pretrained-friendly | Cross-modal depth | Recommendation |
|---------------|----------------------|--------------------|--------------------|---------------|
| Late score fusion | Trivial | ✅ Excellent | None | Quick sanity check |
| Feature concat + MLP | Low | ✅ Excellent | Minimal | **Phase 1 baseline** |
| MMTM | Low–Medium | ✅ Excellent | Channel-level | **Phase 2 upgrade** |
| Bottleneck tokens (MBT) | Medium | ⚠️ Needs BST mods | Rich | Phase 3 if needed |
| FiLM conditioning | Low | ✅ Good | Channel-level | Alternative to MMTM |
| Full cross-attention | High | ⚠️ Architecture changes | Rich | Not recommended here |

---

## Something-Something V2 pretraining transfers better to racket manipulation

The SSv2 dataset contains **220,847 videos across 174 classes** of hand-object manipulation — pushing, pulling, putting, picking up, bending, folding, and other object interaction primitives. Unlike Kinetics-400, where many classes can be predicted from a single frame based on scene appearance, SSv2 **requires genuine temporal reasoning**: "pushing something left" and "pushing something right" share identical appearance but differ only in motion direction. This property maps directly to badminton stroke discrimination, where forehand drives, clears, and drops share similar preparatory poses but diverge in racket acceleration patterns and wrist supination/pronation.

A key finding from the UniFormerV2 paper confirms this: **extra Kinetics pretraining actually harms representations for SSv2-type temporal tasks** (Table 13). The LAPA paper further validates SSv2's relevance by using its 220K manipulation videos to pretrain robot manipulation policies — the learned latent actions transfer successfully to real robot tasks, demonstrating that SSv2 captures generalizable manipulation primitives.

**The availability gap matters.** X3D — the recommended backbone — has no official SSv2-pretrained checkpoints. Only Kinetics-400 weights are available via PyTorchVideo and PySlowFast. The models with the best SSv2 checkpoint availability are VideoMAE (HuggingFace: `MCG-NJU/videomae-base-finetuned-ssv2`, 70.6% top-1), Video Swin Transformer (via GitHub releases), MViTv2 (via PySlowFast), UniFormer (via Google Drive), and TimeSformer (via GitHub). However, these are all heavy transformer models unsuitable as a lightweight sub-component.

The practical resolution is to **use K400-pretrained X3D-S weights as initialization, then fine-tune on the target badminton wrist crops.** With 25k domain-specific training samples, the model will learn stroke-specific temporal features regardless of pretraining dataset. The K400 pretraining still provides useful low-level motion features (edges, textures, short-range motion patterns) even though its high-level action classes are scene-centric. An alternative worth testing: initialize X3D-S from K400 weights, then intermediate-fine-tune on SSv2 (168.9K training videos) before final fine-tuning on the badminton crop data. This two-stage transfer (K400 → SSv2 → badminton) would inject manipulation-specific temporal reasoning at modest training cost.

The top SSv2 leaderboard for reference: V-JEPA 2.1 ViT-G (**77.7%**), InternVideo2-1B (**77.5%**), VideoMAE V2 ViT-g (**77.0%**), VideoMAE ViT-H (**75.4%**), MViTv2-B (**72.1%**), UniFormer-B (**71.2%**), VideoMAE ViT-B (**70.6%**), Video Swin-B (**69.6%**), VideoMAE ViT-S (**66.8%**).

---

## Adaptive crop sizing uses torso height as a perspective anchor

The crop region around the player's wrist must scale with camera distance. A player near the camera needs a smaller pixel radius than one far away to capture the same real-world region. **Skeleton keypoint scale provides a reliable, frame-by-frame calibration signal** that requires no camera intrinsics or depth estimation.

**Torso height** — the Euclidean distance from midpoint(shoulders) to midpoint(hips) — is the most stable scale reference. It uses four keypoints (COCO indices 5, 6, 11, 12), providing robustness to individual keypoint noise, and remains approximately constant regardless of arm pose. A badminton racket is ~68cm long, a forearm ~25cm, giving a forearm-to-racket span of ~93cm. With adult torso height at ~50–55cm, capturing the full forearm + hand + racket head requires a crop diameter of roughly **1.6–2.4× torso height**. The recommended formula:

```
crop_radius = k × torso_height_pixels    (start with k = 1.0, tune between 0.8–1.2)
```

The crop should be centered on the dominant wrist keypoint (COCO index 10 for right wrist), optionally offset 0.3 × torso_height along the elbow→wrist direction to better capture the racket head beyond the wrist.

**Temporal smoothing is critical.** Raw keypoint coordinates jitter by 3–10 pixels frame-to-frame even in stationary poses. The **One Euro Filter** is the recommended smoothing approach — it adaptively adjusts its cutoff frequency based on movement speed, tracking fast stroke motions faithfully while suppressing slow jitter. Set `min_cutoff ≈ 1.0 Hz` and `beta ≈ 0.5` for the crop center coordinates, and `beta ≈ 0.1` for the crop radius (which should change more slowly). Apply separate filter instances for center_x, center_y, and radius. The filter is available via `pip install OneEuroFilter` and is also built into OpenMMLab's mmpose.

**For differentiable end-to-end training**, use `torchvision.ops.roi_align` to perform crop-and-resize in a single GPU operation with bilinear interpolation. This accepts boxes in `[batch_idx, x1, y1, x2, y2]` format and outputs fixed-size feature maps. For offline data preprocessing, simple tensor slicing with `F.pad` for edge handling and `F.interpolate` for resizing is sufficient. Use square crops — the racket can point in any direction during a stroke, so no consistent aspect ratio exists. Handle frame edges by shifting the crop center inward or zero-padding.

Fallback hierarchy when keypoints are occluded: (1) torso height from all four keypoints, (2) shoulder width × 1.5, (3) upper arm length × known proportionality constant, (4) bounding box height × 0.3 from the detector. If the wrist keypoint itself is missing, interpolate from the elbow plus the previous frame's wrist direction, or carry forward the last valid crop position with EMA decay.

---

## Conclusion

The most effective architecture for this BST extension is **X3D-S processing 112×112 wrist crops at 13 frames**, fused with the skeleton transformer via late feature concatenation initially and MMTM once the baseline is validated. This adds under 4M parameters and under 1 GFLOP per clip — negligible alongside the BST skeleton stream. Three insights stand out beyond what the individual component analyses might suggest.

First, **the crop sub-model and the skeleton model are complementary in a precise, non-redundant way**: the skeleton stream captures full-body kinematics (weight transfer, trunk rotation, arm trajectory) while the video crop captures local appearance details that skeletons fundamentally cannot represent — racket face angle, grip type, wrist supination/pronation, and shuttle contact dynamics. This complementarity is exactly why naive late fusion often fails in racket sports (as RacketVision demonstrated) — the streams encode different physical properties at different spatial scales, and the fusion mechanism must learn to align them.

Second, **the K400-to-SSv2-to-badminton transfer chain** is worth the engineering effort despite X3D lacking official SSv2 weights. Fine-tuning X3D-S on SSv2's 168.9K manipulation videos before domain-specific training would inject precisely the temporal reasoning biases that Kinetics' scene-centric pretraining misses. This intermediate step costs one additional training run but could substantially improve discrimination of strokes that differ primarily in wrist acceleration patterns.

Third, **the adaptive crop pipeline should be treated as a first-class component, not a preprocessing afterthought.** The One Euro Filter's speed-adaptive smoothing is essential because badminton wrist velocities range from near-zero during preparation to over 50 pixels/frame during striking — fixed-bandwidth smoothing either blurs the strike or fails to suppress preparation jitter. Making the crop differentiable via RoIAlign opens the door to end-to-end learning of crop parameters, potentially allowing the model to learn task-optimal crop placement rather than relying on hand-tuned heuristics.