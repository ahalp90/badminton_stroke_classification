# Architecture 2: Video-Based Badminton Stroke Classifier

## Design Document — X3D-M + TrackNetV3 Multi-Stream Architecture

> **Training target:** NVIDIA V100 16GB (primary), A100 40GB (occasional hyperparameter search sessions, ≤12 hours each)
>
> **Dataset:** ShuttleSet (~25k labeled stroke clips from BWF broadcast matches)
>
> **Differentiation from Architecture 1:** This architecture operates directly on RGB player crops, capturing racket angle, grip, wrist supination/pronation, and shuttle contact dynamics — visual details invisible to the skeleton/keypoint representations used by BST. Architecture 1 (BST + optional racket crop extension) processes extracted pose sequences through temporal transformers with shuttle trajectory cross-attention.

---

## 1. Player Detection and Crop Extraction

### Detector: YOLO26m

YOLO26 (released January 2026) is the recommended detector. Its NMS-free end-to-end design produces deterministic latency and consistent frame-by-frame bounding boxes — critical for generating temporally smooth player crops. YOLO26m achieves ~50+ mAP on COCO while running at real-time speeds, and its ProgLoss + STAL loss functions specifically boost small-object detection for far-court players. The COCO-pretrained person class works out-of-the-box for broadcast badminton since players are prominent subjects. YOLO11m is a proven fallback if YOLO26 integration proves unstable.

### Tracker: BoT-SORT

BoT-SORT (the Ultralytics default tracker) is ideal because it includes camera motion compensation via frame-to-frame homography estimation — essential for broadcast cameras that pan and zoom. Its improved Kalman filter estimates width and height directly (8-tuple state), producing more accurate bounding boxes than ByteTrack. Apply **exponential moving average smoothing** (α = 0.8) on bounding box coordinates on top of BoT-SORT to further reduce crop jitter.

### Crop specification

- **Resolution:** Extract crops at **256×256**, then let the training pipeline random-crop to **224×224** (data augmentation) or center-crop at inference. The 224×224 resolution preserves subtle racket/wrist detail that 112×112 would lose.
- **Frame rate:** Extract at **native broadcast rate** (25 or 30 fps). Let the 3D CNN's temporal stride handle subsampling.
- **Which players:** Detect and track **both** players. Crop the **stroke-executing player** for the 3D CNN input. Encode the opponent's court position as a 2D coordinate vector fed to the trajectory stream. Distinguish near/far player by foot y-coordinate relative to court lines.
- **Court filtering:** Use court line detection to define the playing area polygon, then discard any person detections outside it (removes spectators, coaches, line judges).

---

## 2. Shuttle Tracking: TrackNetV3

**TrackNetV3** remains the best available shuttlecock tracker, achieving 97.5% accuracy with 99.3% recall on the Shuttlecock Trajectory Dataset. Its two-module design — trajectory prediction via background estimation with mixup augmentation, plus trajectory rectification via inpainting for occluded frames — handles the extreme speed of professional shuttlecocks (300+ km/h) better than any YOLO-based alternative. **TrackNetV4** (ICASSP 2025) adds a motion-aware fusion mechanism and is a worthwhile plug-and-play upgrade, but TrackNetV3 alone is sufficient.

### Trajectory representation

Use **per-frame normalized 2D coordinates** (x/width, y/height) — this is exactly what BST uses and achieves SOTA results without requiring court homography estimation. Optionally compute **velocity vectors** as finite differences (v_t = pos_t − pos_{t−1}) as auxiliary features, since shuttle speed at the hit frame strongly discriminates stroke types. For frames where the shuttle is occluded, TrackNetV3's built-in InpaintNet rectification module fills gaps automatically. Store a per-frame confidence flag so downstream models can weight uncertain positions lower.

The trajectory window should be **wider than the video window** — include shuttle positions spanning 2–3 strokes before and after the target stroke. BST demonstrated that incoming trajectory (speed, angle from the opponent's previous stroke) is critical for classifying the response stroke type.

---

## 3. Racket Contact Detection

### Recommended method: shuttle velocity flip from TrackNet coordinates

The most pragmatic and effective approach is to detect the racket contact frame as the moment of shuttle velocity minimum or direction reversal, derived directly from TrackNet's per-frame 2D positions. This is physically equivalent to what dedicated hit-detection models (e.g. HitNet from MonoTrack) attempt to detect, but avoids introducing an additional model checkpoint, inference pass, and failure mode.

**Implementation:** Compute velocity as a simple finite difference on court-normalized TrackNet coordinates, apply a small Gaussian smoothing kernel (σ=2 frames) to suppress single-frame jitter, then find zero-crossings in the velocity component. For serves, the shuttle has no prior trajectory — fall back to the first frame where TrackNet detects the shuttle with upward velocity.

The velocity flip method has one minor weakness: on high clears or lobs where the shuttle decelerates gradually, the minimum-speed frame can be ambiguous across 2–3 frames. For stroke classification this is negligible — a ±1 frame error on contact detection shifts the 32-frame window by one frame, which has no meaningful impact on classification accuracy.

HitNet adds a separate model dependency for marginal precision gain on an anchor point that does not require sub-frame accuracy. The cost-benefit does not justify it.

---

## 4. 3D CNN Backbone: X3D-M

After evaluating seven model families against the specific requirements — fine-grained discrimination on ~25k samples, V100 16GB memory, and 224×224 player crops — **X3D-M emerges as the clear winner**, with R(2+1)D-18 as runner-up and VideoMAE ViT-S as a high-accuracy alternative.

**X3D-M** (Meta) has just **3.8M parameters** and **6.2 GFLOPs** per clip. This tiny parameter count gives an exceptionally favorable sample-to-parameter ratio of ~6.5:1 on 25k samples, virtually eliminating overfitting risk. It natively accepts inputs from 160×160 to 224×224 with 16 frames, fits **batch sizes of 16–32 on V100 16GB with FP16**, and trains in approximately **5–10 hours for 30 epochs**. Kinetics-400 pretrained weights are available through PyTorchVideo and PySlowFast.

| Model | Params | GFLOPs | V100 batch (FP16) | Training time | Practical score |
|---|---|---|---|---|---|
| **X3D-M** | **3.8M** | **6.2** | **16–32** | **5–10 h** | **★★★★★** |
| R(2+1)D-18 | 31.5M | 40.7 | 8–16 | 15–30 h | ★★★★ |
| VideoMAE ViT-S | 22M | ~70 | 4–8 (+ checkpointing) | 30–100 h | ★★★ |
| SlowFast R50 4×16 | 34M | 36 | 4–8 | 15–30 h | ★★★ |
| Video Swin-T | 28M | 88 | 2–4 (+ checkpointing) | 30+ h | ★★ |
| UniFormer-S | 21.7M | 42 | 4–8 | 20–40 h | ★★★ |
| TimeSformer | 121M | 590+ | 1–2 | impractical | ★ |

### Why not PoseC3D?

PoseC3D converts keypoints into heatmap volumes and processes them through a 3D CNN. It is deliberately excluded because it discards the entire rationale for Architecture 2. The BST/TemPose skeleton pipeline already captures body kinematics extremely well (that is Architecture 1). The gap BST cannot close is *appearance-level detail*: racket face angle at contact, grip type, wrist supination/pronation, string-bed contact point, and the visual texture of the shuttle leaving the racket. These are precisely the cues that distinguish confusable stroke pairs like top smash vs. top wrist smash (the categories BST's confusion matrices show it struggles with most). PoseC3D would provide a different encoding of the same skeleton information BST already uses, not complementary information.

The VideoBadminton benchmark included PoseC3D — it scored 80.76% top-1 on 18 classes, lower than SlowFast's 82.8% from raw RGB. That 2-point gap represents the appearance information keypoint representations discard.

---

## 5. Temporal Windowing

### Video window: asymmetric, contact-anchored

**Recommended window: −24 to +8 frames around the hit frame** at 30 fps (~0.8s pre-contact + 0.27s post-contact = 32 frames total). The pre-contact phase — backswing, body positioning, footwork — is the primary discriminator between stroke types. A drop shot and a smash originate from similar court positions but have radically different preparation. Post-contact follow-through provides confirmation but carries less discriminative information.

This 32-frame clip maps directly to X3D-M's 16-frame input via stride-2 sampling, always anchoring the contact frame at position 12 of 16.

### Frame sampling

Use **dense sampling with contact-frame anchoring** — ensure the hit frame is always included and not randomly dropped during temporal augmentation. Sparse TSN-style sampling risks missing the critical contact moment. From a 32-frame window, sample every 2nd frame with the contact frame guaranteed at position 12.

### Trajectory window: wider than video

For the trajectory stream, use a wider window spanning **2–3 strokes** before and after the target, providing rally context that the tight video window cannot capture. This asymmetry between modality windows is a key design insight: the video captures fine-grained stroke mechanics while the trajectory captures tactical context.

---

## 6. Multi-Stream Fusion

### Stage 1 — MMTM at intermediate layers

Insert 2–3 MMTM (Multimodal Transfer Module) modules at the last residual stages of X3D-M. MMTM uses squeeze-and-excitation to generate cross-modal recalibration signals, allowing trajectory information to modulate which visual features the CNN emphasizes. MMTM was designed specifically for modalities with different spatial dimensions — the squeeze operation collapses spatial dimensions, making it natural to fuse a 3D feature volume with a 1D trajectory vector. It adds negligible computation.

### Stage 2 — Cross-attention after feature extraction

After the 3D CNN produces temporal feature vectors and a small TCN encodes the trajectory sequence, apply **one layer of cross-attention** where video features query trajectory features and vice versa. This mirrors BST's CrossTrans module, which was shown to be a critical component.

### No gated fusion on the trajectory branch

A gated sigmoid on the trajectory branch to suppress it during TrackNetV3 failures is unnecessary. At 97.5% accuracy and 99.3% recall, TrackNetV3 fails on roughly 1 in 40 frames, and those failures are predominantly on occluded frames that the InpaintNet module already rectifies via trajectory interpolation. A sigmoid gate would converge to a near-constant value of ~1.0 (pass everything through) and add noise. The failure rate is too low for the gate to learn a meaningful modulation signal.

If trajectory noise is later found to cause problems on specific stroke types, gating can be reintroduced, but this is unlikely to be needed on ShuttleSet's broadcast-quality video.

### Temporal pooling

Replace the standard temporal global average pooling (TGAP) with **temporal attention pooling** — a single self-attention layer over temporal positions. This allows the model to identify the most discriminative frames rather than treating all equally. No additional TCN or transformer stack is needed on the video side, since X3D-M's receptive field already covers the full ~1-second clip. The TCN is reserved for encoding the trajectory stream.

### Classification head

Standard MLP head: LayerNorm → Linear → ReLU → Dropout(0.5) → Linear → n_classes.

---

## 7. Training Strategy

### Progressive unfreezing on V100 16GB

Frozen backbones are catastrophic for fine-grained temporal tasks — AIM (ICLR 2023) showed frozen ViT on Something-Something v2 achieves only 15.1% versus 59.5% with full fine-tuning. Kinetics features capture "what sport" but not "what specific stroke." However, X3D-M's 3.8M parameters are small enough that full fine-tuning is safe on 25k samples.

**Stage 1 — Head warmup (5 epochs):** Freeze X3D-M backbone entirely. Train only the classification head, fusion modules (MMTM, cross-attention), and trajectory TCN. Learning rate 1e-3 with Adam. Batch size 32+.

**Stage 2 — Full fine-tuning (25 epochs):** Unfreeze all X3D-M layers. Learning rate 1e-4 with cosine annealing to 1e-6. Apply strong regularisation: random spatial crop (224 from 256), horizontal flip, colour jitter, Mixup (α=0.2), CutMix, weight decay 5e-4, dropout 0.5 on the head. **Keep all BatchNorm layers frozen** — BN statistics from Kinetics pretraining are more stable than what 25k samples can estimate. Batch size 16 with FP16 mixed precision.

**Stage 3 — A100 refinement (optional):** Use A100 40GB sessions for hyperparameter search over learning rate, temporal window parameters, fusion module placement, and Mixup alpha. Batch size 32–64 enables faster iteration.

### Mixed precision

Non-negotiable on V100. The V100's Tensor Cores deliver ~125 TFLOPS FP16 versus ~15 TFLOPS FP32. Use PyTorch's `torch.cuda.amp.autocast()` and `GradScaler`. Gradient accumulation over 2 steps (effective batch 32 from micro-batch 16) is recommended for training stability.

### Data loading

Decode videos to individual frames offline and store as JPEG/PNG or in LMDB format. On-the-fly video decoding wastes 30–50% of training time. Use 4–8 dataloader workers.

---

## 8. Generalisation and Domain Adaptation

### Relationship to Architecture 1 for cross-domain robustness

An RGB model trained exclusively on BWF broadcast footage will encode camera angle, court colour, lighting, jersey style, and broadcast resolution into its features. Amateur footage from different camera positions and lighting conditions will violate these assumptions. Architecture 1 (BST) is inherently more invariant to these factors because skeleton representations abstract away visual domain — **Architecture 1 generalises broadly while Architecture 2 achieves higher peak accuracy on broadcast-quality footage**.

Adding a parallel keypoint stream to Architecture 2 will not solve the generalisation problem. During training on ShuttleSet's professional data, the RGB stream will dominate because it carries strictly more information — the model will learn to weight the keypoint stream near zero through its fusion layers. Amateur training data would be needed to force reliance on keypoints, and ShuttleSet does not provide any.

The most effective approach to improving Architecture 2's robustness without architectural changes is **aggressive visual augmentation**: random colour jitter, random grayscale conversion (p=0.2), random Gaussian blur, and aggressive random resized cropping. This forces the model to rely less on colour/texture priors and more on motion patterns. It will not close the full domain gap to amateur footage, but meaningfully reduces overfitting to broadcast aesthetics at zero architectural cost.

---

## 9. Tuning and Augmentation for Generalisation to Amateur Footage

> **Important:** This section should be implemented only *after* the initial BST-comparable model has been trained, validated, and its hyperparameters locked in. The generalisation augmentations described here are a second-phase refinement, not part of the initial model development.

### 9.1 Temporal stretch augmentation

Amateur players execute strokes more slowly and with more variable timing than professionals. Biomechanical research confirms an approximately 2:1 speed differential: professional shuttle speeds reach 300–400 km/h for smashes versus 150–250 km/h for club-level players; motion capture studies show advanced players achieve racket head speeds of ~45 m/s versus ~20–25 m/s for novices. However, the temporal distortion between amateur and professional play is phase-dependent and non-uniform — amateurs have substantially longer and less decisive preparation phases, but the forward swing phase (the 60–100ms immediately before contact) is only moderately slower (~1.5× rather than 2×).

### 9.2 Implementation: resampling before frame selection

The temporal distortion is applied *before* frame sampling, not after. The 3D CNN always receives exactly 16 frames — the augmentation changes *which* frames are selected.

1. Draw a temporal scale factor s uniformly from **[1.0, 2.0]** (unidirectional — professionals are already the fastest case; there is no reason to simulate faster-than-professional play).
2. Multiply the 32-frame raw window boundaries by s, yielding a window of 32–64 raw frames.
3. Uniformly sample 16 frames from that expanded window, anchoring the contact frame at position 12.
4. For non-integer sample positions, use **nearest-neighbour selection** — frame blending creates ghosting artifacts that don't correspond to any physical camera behaviour.

At s=2.0, the model captures roughly twice the real-world duration in the same 16 frames, matching the observed speed differential. Each sampled frame is spaced further apart in real time, simulating what a camera would capture of a slower player executing the same stroke type.

**Edge-case clamping:** At s=2.0 the window extends to ~2.1 seconds at 30fps, which may encroach on the preceding stroke in fast professional rallies. Clamp the window to not extend past the previous hit frame (as detected by the TrackNet velocity-flip method).

### 9.3 Matched trajectory stretching

If the video window is temporally stretched, the *identical* stretch factor must be applied to the trajectory sequence for that sample. Resample the shuttle coordinates to the same temporal grid as the video frames using linear interpolation between TrackNet's per-frame positions. Linear interpolation is physically appropriate — shuttle trajectories are smooth parabolic arcs.

### 9.4 Per-frame temporal jitter

Beyond global stretch, add per-frame jitter of ±1 frame to each of the 16 sample positions (applied after computing the stretched grid). This simulates the timing irregularity of amateur strokes — professionals have metronomic preparation phases, amateurs do not. Enforce monotonicity (never let sample i land after sample i+1).

### 9.5 What to avoid

- **Frame duplication/dropping:** Creates unnatural repeated-frame or discontinuity artifacts that 3D convolution kernels learn to detect and ignore. The resampling approach achieves the same kinematic effect without information destruction.
- **Feature-level temporal stretch:** Interpolating in learned feature space has no physical meaning and will degrade performance.
- **Sub-1.0 stretch factors:** Training is exclusively on professional data and the target amateur population is uniformly slower. There is no use case for simulating faster-than-professional play.

### 9.6 Hyperparameter search for generalisation augmentation

The initial BST-comparable model's hyperparameters substantially constrain this follow-up search. Parameters are divided into three categories:

**Transfer directly (do not re-search):** Learning rate schedule, optimizer, weight decay, batch size, epoch count, dropout rate, MMTM placement, fusion architecture, backbone selection, frozen-BN decision. These are driven by dataset size and architecture, not the augmentation mix.

**Quick validation sweep (1–2 values each):** Contact frame anchor position (may shift from 12 to 13–14 to accommodate more stretched preparation frames). Base video window boundaries (may widen from −24/+8 to −28/+8 to provide more raw frames for high stretch factors). A few hours of V100 time.

**Focused search (new parameters):** Stretch factor upper bound (1.5 vs 1.7 vs 2.0), probability of applying stretch per sample (p=0.3–0.5), per-frame jitter magnitude (±1 vs ±2), and optionally the non-uniform pre-contact vs post-contact stretch ratio. This is a ~4-dimensional search over a small grid. At ~20 minutes per training run, an exhaustive 81-run grid takes ~27 hours, or a reduced search of 15–20 runs takes 5–6 hours on V100.

**Key validation criterion:** If the amateur-adapted model's accuracy on the original ShuttleSet test set drops by more than 1–2 points compared to the BST-comparable baseline, the stretch augmentation is too aggressive. The temporal stretch should act as regularisation that does not hurt professional-footage accuracy while adding amateur robustness — not a tradeoff between the two.

---

## 10. Relevant Literature and Benchmarks

- **BST-CG-AP** (Chang, 2025): Current SOTA on ShuttleSet at ~77% accuracy / ~70.4% macro-F1 across 35 classes using skeleton + shuttle trajectory. Leaves significant headroom for a complementary RGB approach.
- **VideoBadminton** (Li et al., IEEE BigData 2024): Benchmarked 3D CNNs on badminton RGB video. SlowFast achieved 82.8% top-1 on 18 classes from 7,822 clips, outperforming skeleton-based models (ST-GCN 74.4%, PoseC3D 80.8%). Validates viability of the RGB approach.
- **FineBadminton** (ACM MM 2025): Multi-level semantic annotations with hit-centric keyframe selection, aligning with our asymmetric windowing approach.
- **TrackNetV3** (ACM MMAsia 2023): 97.5% accuracy shuttle tracking with trajectory rectification. Remains the gold standard.
- **TrackNetV4** (ICASSP 2025): Motion-aware fusion for improved occluded shuttle tracking. Optional upgrade.
- **X3D** (Feichtenhofer, CVPR 2020): 3.8M params, 6.2 GFLOPs. Optimal efficiency-accuracy tradeoff for small-dataset video classification.
- **MMTM** (Joze et al., CVPR 2020): Squeeze-and-excitation cross-modal fusion for heterogeneous modalities. Designed for modalities with different spatial dimensions.
- **Biomechanical speed differentials** (Ramasamy et al., Sports Biomechanics 2024; Kwan et al., 2011): Elite smash speeds ~300–400 km/h, racket head speed ~45 m/s. Club-level: 150–250 km/h, racket head speed ~20–25 m/s. Novice body mechanics are fundamentally different (compensatory strategies, excessive joint motion) rather than simply slower versions of professional mechanics.

---

## 11. Reuse of BST/TemPose Code

### Components that transfer directly

- **`pipeline/` directory:** Entirely model-agnostic. Produces labeled video clips and shuttle trajectories.
- **`pipeline/config.py`:** Single source of truth for stroke types, class labels, splits, merge rules, and taxonomies.
- **`result_utils.py`:** Works with any model producing `(predictions, ground_truth)` tensors. `show_f1_results()` and `plot_confusion_matrix()` are architecture-agnostic.
- **TrackNetV3 integration** in `shuttle_extractor.py`: Produces shuttle trajectory `.npy` files consumed by both architectures.

### Building blocks from `tempose.py` worth importing

- `TCN` (dilated 1D temporal convolutions) — reusable for the trajectory encoder stream.
- `MLP`, `MLP_Head` — generic components for the classification head.
- `FeedForward` — usable in cross-attention layers.
- `TransformerEncoder` — if temporal attention pooling is implemented as a transformer layer.
- `MultiHeadAttention` — basis for the cross-attention fusion module (needs adaptation to cross-attention variant, but the `MultiHeadCrossAttention` in `bst.py` provides exactly this).

### Components that must be replaced

- **`prepare_train_on_shuttleset.py`:** Architecture 2 operates on raw video, not pose estimation output. Write a new preparation script that extracts player crops via YOLO26 + BoT-SORT rather than running MMPose. Reuse `pipeline.config` for label definitions.
- **`shuttleset_dataset.py`:** BST's dataset classes return `(human_pose, pos, shuttle), video_len, label`. Architecture 2 needs a dataset class returning `(video_crop_tensor, shuttle_trajectory), video_len, label`. Key decisions: whether to decode frames on-the-fly or from pre-extracted images, and how to handle the dual temporal windows (tight for video, wide for trajectory).
- **`bst.py`:** The BST model itself. Replace with the X3D-M + MMTM + cross-attention + temporal attention pooling architecture. The `CrossTransformerLayer` from `bst.py` is a useful reference for the cross-attention fusion implementation.

### Training/inference scripts

The overall structure of `bst_train.py` and `bst_infer.py` (training loop, checkpoint saving, early stopping, multi-trial evaluation) can be adapted. The `Hyp` namedtuple pattern for hyperparameter management, the AdamW + cosine annealing setup, and the label smoothing configuration all transfer. The progressive unfreezing stages (Section 7) require modifications to the training loop that BST's single-stage training does not have.
