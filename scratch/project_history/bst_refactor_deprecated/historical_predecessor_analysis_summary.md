# Predecessor Project Analysis Summary

## Document Purpose

This document is a comprehensive summary of a detailed code review and analysis of the COSC591 Group D (UNE, Trimester 2, 2025) badminton shot classification project. It is intended to be carried into a new conversation exploring the updated official BST repository, to inform the design of two successor classifiers that must significantly outperform the predecessors on the same ShuttleSet dataset.

---

## Project Overview

**Team:** Richard Pinter, Conor Molloy, Drew Roberts, Tim Robinson (COSC591, UNE)
**Client:** Farshid Hajati, Hunter Badminton Club
**Objective:** Automated badminton player skill assessment through video analysis

The team inherited pose estimation infrastructure from a prior group but discovered in Week 2 that ShuttleSet's own models (ShuttleScorer, ShuttleNet, DyMF) were designed for stroke *forecasting* and movement prediction, not shot *type classification*. They pivoted to building classification from scratch and delivered two models:

- **LSTM classifier:** 36.25% accuracy (TensorFlow/Keras)
- **BST (Badminton Shot Transformer):** 53.6% accuracy (PyTorch, adapted from Chang 2025)

State-of-the-art badminton shot classifiers now achieve 90-98% mAP.

---

## Dataset: ShuttleSet

- **Claimed size:** 103,997 stroke instances — this number is inflated. The core ShuttleSet (2018-2021) has ~36,000 strokes; ShuttleSet22 adds ~33,000. The 103,997 figure likely results from counting Top/Bottom player annotations as separate instances and/or combining both dataset versions without acknowledgement.
- **Actual training set used (LSTM):** 8,634 shots from 39 matches (but the deployed model file is named `15Matches_LSTM`, suggesting only 15 matches were used for the final model).
- **Shot types:** ShuttleSet has 18 labelled stroke types + 2 error types. The team mapped these to 6 "frontier" classes: Clear, Drive, Drop, Lob, Net, Smash (based on the Sheng et al., 2022 instrument / Frontiers in Psychology taxonomy).
- **Official splits ignored:** ShuttleSet provides match-level train/val/test splits. The team ignored these and did random shot-level 90/10 splitting, creating data leakage (see below).

### Frontier Class Mapping (Table 4.2 from their report)

| Frontier Class | Raw Shot Types |
|---|---|
| Clear | clear |
| Drive | drive, driven flight, back-court drive, push, rush, defensive return drive |
| Drop | drop, passive drop |
| Lob | lob, defensive return lob |
| Net | net shot, return net, cross-court net shot, short service, long service |
| Smash | smash, wrist smash |

### Class Distribution (Training Set)

| Class | Count | % |
|---|---|---|
| Net | 3,491 | 40.5% |
| Drive | 1,257 | 14.5% |
| Lob | 1,241 | 14.4% |
| Smash | 1,192 | 13.9% |
| Clear | 816 | 9.5% |
| Drop | 637 | 7.4% |

Heavily imbalanced — Net is 5.5x more frequent than Drop. No class balancing was applied.

---

## Root Cause Analysis: Why Accuracy Was So Poor

### 1. CRITICAL: Label Mapping Bugs

**File:** `models/lstm/convert_bst_to_lstm.py:96-116`

The 35-to-6 frontier mapping in code contradicts the project's own Table 4.2:

```python
stroke_to_frontier = {
    'drop': 'net',              # BUG: should be 'drop'
    'cross-court_net_shot': 'drop',  # BUG: should be 'net'
    'drive_variant': 'drop',    # Questionable
    # ... other mappings
}
```

The most canonical member of the "Drop" class is mapped to "Net", and a net-area shot is mapped to "Drop". This is a bidirectional label swap.

**Evidence of impact:** The LSTM confusion matrix shows 0% accuracy on Clear, Drop, Lob, and Drive, with only Net (72.63%) and Smash (46.38%) showing any signal — the model collapsed to predicting the majority class.

### 2. CRITICAL: Single-Player Filtering

The team filtered the dataset to retain only bottom-player annotations, discarding top-player data. This:
- Halved the effective training set
- Removed opponent context essential for shot type discrimination (a drop vs clear decision depends on opponent position)
- Undermined the BST's cross-attention mechanism, which was specifically designed for two-player interaction modelling

**Code:** `convert_bst_to_lstm.py:317` uses only `sample_joints[:, 0, :, :]` (player index 0).
**Dataset classes:** `Dataset_npy_collated_one_side` (dataset.py:308) explicitly halves the dataset by side.

### 3. HIGH: LSTM Architecture Fundamentally Inadequate

**File:** `models/lstm/shot_classifier.py:117-132`

Architecture:
```python
model = Sequential()
model.add(Masking(mask_value=0, input_shape=input_shape))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
```

Problems:
- **Code/report mismatch:** Report says cross-validation selected "Normal128P6" (128 units, 0.6 dropout), but deployed code uses 256 units, 0.2 dropout — a completely different configuration.
- **Normalisation flaw (line 112):** `shapedData/np.max(shapedData)` normalises by training-set global max, but `_predictModel` (line 184) applies NO normalisation at inference time. Train/test distribution mismatch.
- **Checkpoint monitors training loss (line 128):** `monitor='loss'` selects the most overfit checkpoint, not the best generalising one.
- **400 epochs, batch size 100, no early stopping:** Guarantees overfitting.
- **No LR scheduler:** Default Adam lr=0.001 for all 400 epochs.
- **No pre-trained backbone:** Trained from scratch on small data.
- **CPU-only training:** TensorFlow GPU incompatibilities were never resolved, severely limiting iteration.
- **Input:** Raw (x,y) pixel coordinates only — no velocity, acceleration, or angular features.

### 4. HIGH: BST Temporal Processing Loses Critical Motion Peaks

**File:** `models/bst/models/dataset.py:55-88`

The `make_seq_len_same()` function uses stride-based subsampling:
```python
stride = video_len // target_len + int(need_padding)
joints = joints[::stride][:target_len]
```

A fast smash occurs in ~3-5 frames. With stride=5, the hitting frame can be completely skipped. Zero-padding (for short sequences) is indistinguishable from missing keypoints — both are 0.0.

BST_8 uses d_model=100, only 2 temporal transformer layers and 1 interactional layer — relatively small.

### 5. HIGH: Coordinate Space Mismatch Between Modalities

**File:** `models/preprocessing/prepare_train.py`

The three BST input modalities are each normalised to incompatible coordinate frames:

| Modality | Normalisation | Coordinate Space |
|---|---|---|
| Joints (line 122-152) | `(x - bbox_x1) / bbox_diagonal` | Player bounding-box relative |
| Player positions (line 105-119) | Homography → court metres → [0,1] | Real court coordinates |
| Shuttlecock (line 155-164) | `x / video_width`, `y / video_height` | Raw pixel-space (camera-dependent) |

The shuttlecock should have been projected through the homography matrix into court coordinates, like player positions were. Instead, the same shuttle position on court produces different normalised values depending on camera placement, preventing generalisation across venues.

### 6. HIGH: No Data Augmentation Applied in Practice

- **LSTM:** Zero augmentation.
- **BST:** `RandomTranslation` class defined (dataset.py:118) but `Dataset_npy_collated` explicitly documents: *"Notice: There is no random translation here."* (line 249). The augmentation infrastructure exists but is never applied to the training pipeline actually used.
- No temporal augmentation (speed variation, frame jitter).
- No spatial augmentation (flip, rotate, scale).

### 7. MEDIUM-HIGH: 630 Lines of Feature Engineering Built but Never Used

**File:** `models/lstm/feature_extraction.py` (630 lines)

Contains velocity, acceleration, DTW distance, movement patterns, and body speed calculations. **None of this is imported or used by `shot_classifier.py`.** The LSTM trains on raw (x,y) coordinates only, which are camera-dependent and lack the temporal derivatives that distinguish shot types.

### 8. MEDIUM-HIGH: No Class Imbalance Handling

No weighted loss, no oversampling/undersampling, no focal loss, no balanced batch sampling. The 40.5% Net majority class dominates predictions.

### 9. MEDIUM: Data Leakage in K-Fold Cross-Validation

**LSTM training (p20-21 of report):**
- 9,594 shots pooled from 39 matches
- Random 90/10 shot-level split (seed 42)
- 4-fold CV on the 90% training portion

Shots from the same rally (consecutive strokes, same players, same camera) can end up in different folds. The model validates against near-duplicates of its training data. This inflates CV accuracy, making model selection unreliable.

ShuttleSet provides official match-level splits specifically to prevent this. They were ignored.

### 10. MEDIUM: BST Training Not Reproducible

No BST training script exists in the repository. The BST README references `stroke_classification/bst_main.py` but this file is missing. Training configuration is described only in the PDF report (p30): Adam lr=1e-3, weight decay 1e-4, cosine annealing, batch 16, cross-entropy with label smoothing 0.1, early stopping patience 20.

### 11. MEDIUM: Same Convergence on 6 vs 35 Classes

The report notes (p30): *"Note that we noticed the same behaviour for the raw shot types, raising questions about the model implementation."* The BST converged at ~1,660 steps regardless of whether classifying 6 frontier classes or 35 raw shot types. This indicates the input representation (not the label space) is the bottleneck — the preprocessing pipeline has stripped out so much discriminative information that the model hits the same information ceiling either way.

---

## "FPCA" Player Profiling — Not Actually FPCA

The report (pp36-38) describes a "Functional Principal Component Analysis" for player grading. This is a misnomer.

**What real FPCA is:** Treats observations as continuous functions, represents them in a basis (B-spline/Fourier), computes a covariance operator, performs eigendecomposition to extract principal modes of variation. Output: eigenfunctions and scores.

**What they actually implemented:**
1. Bin each player's shot speeds into 5 discrete categories (Very Slow/Slow/Medium/Fast/Very Fast) per shot type — a histogram, not a function
2. Average all 102 players' histograms to create a "reference" (just the population mean, not an "A Grade" target — all ShuttleSet players are already professionals)
3. Compute a scalar distance between individual histogram and reference
4. Combine three metrics with fixed (not data-driven) weights:
   - Mean speed (z-scored, min-max normalised)
   - Histogram distance (min-max normalised)
   - Speed std dev / consistency (min-max normalised)
   - → Weighted average → "Final Weighted Score" (0-1)

**No eigendecomposition, no SVD, no PCA, no functional basis representation, no `scikit-fda` or any FDA library.** The Hall et al. (2006) citation is aspirational.

**Additional fatal flaw:** The entire grading pipeline is downstream of the classifiers. With 0% accuracy on 4 of 6 shot types (LSTM) or ~47% error rate (BST), the per-shot-type speed distributions being analysed are composed largely of misclassified shots. The reference distributions are built from the same misclassified data. The grading scores are meaningless.

---

## What's in the Repository

### Key Files

| File | Purpose | Issues |
|---|---|---|
| `models/lstm/shot_classifier.py` | LSTM model + training | Architecture mismatch with report, normalisation bug, monitors training loss |
| `models/lstm/convert_bst_to_lstm.py` | 35→6 label mapping + data conversion | Label mapping bugs (drop↔net swap) |
| `models/lstm/feature_extraction.py` | 630 lines of velocity/accel/DTW features | Never imported or used by training pipeline |
| `models/lstm/match_loader.py` | CSV data loading | Functional but tied to their data format |
| `models/lstm/config.py` | Configuration | Hardcoded constants |
| `models/bst/models/bst.py` | BST architecture (3130 lines, 27+ variants) | Modified from original BST paper author's repo |
| `models/bst/models/dataset.py` | Data loading, bone pairs, augmentation | Augmentation defined but not applied; stride-based downsampling |
| `models/bst/models/tempose.py` | Temporal pose encoding | From original BST repo |
| `models/bst/models/shuttlepose.py` | Pose-shuttlecock fusion | From original BST repo |
| `models/preprocessing/prepare_train.py` | Full preprocessing pipeline | Coordinate space mismatches between modalities |
| `models/preprocessing/dataset.py` | Utility functions (bone pairs, seq padding) | Functional |
| `models/tracknet/model.py` | TrackNetV2 with CBAM attention | From TrackNet paper |
| `app.py` | Gradio unified GUI | Tightly coupled to their model APIs |
| `weights/15Matches_LSTM.keras` | Trained LSTM weights (1.6MB) | Trained on buggy pipeline |

### Model Weights Present

- `weights/15Matches_LSTM.keras` (1.6MB) — LSTM
- `weights/15Matches_LSTM.h5` (1.6MB) — alternate format
- BST weights (`bst_8_JnB_bone_bottom_frontier_6class.pt`, ~7.5MB) — referenced but NOT in repo (Google Drive)
- TrackNet weights (`tracknet_model.pt`, ~174MB) — referenced but NOT in repo (Google Drive)

### Dependencies

Mixed TensorFlow + PyTorch environment:
- torch>=2.0.0, tensorflow>=2.12.0
- mmpose>=1.0.0, mmcv>=2.0.0, mmdet>=3.0.0 (for BST pose estimation)
- ultralytics>=8.0.35 (for LSTM's YOLO11x-pose)
- scikit-learn==1.4.2 (pinned)

---

## Decision: Fork or Fresh?

**Recommendation: Start fresh.**

Reasons:
1. **3D CNN architecture needs video frames** — this repo's entire pipeline extracts keypoints and discards frames. The data flow is fundamentally wrong for a 3DCNN approach.
2. **Core ML pipeline is broken** — label mapping bugs, coordinate space mismatches, missing training scripts, data leakage in splits.
3. **BST code is a modified copy** of the original BST author's repo, which has since been cleaned and reorganised. Go to the source.
4. **Mixed TF/PyTorch dependencies** create unnecessary environment complexity.
5. **Gradio GUI** is tightly coupled to their specific model interfaces — a comparison GUI for two new architectures is straightforward to build fresh.
6. **Nothing in this repo would save more time than it costs to debug.**

The updated official BST repo provides clean ShuttleSet data handling and feature normalisation. Use that as a starting point for data loading, and build clean training/evaluation pipelines with correct match-level splits from day one.

---

## Checklist of Mistakes to Avoid in Successor Project

1. **Verify label mappings** — cross-reference every 35→6 mapping against the original ShuttleSet annotations and your chosen taxonomy
2. **Use match-level train/val/test splits** — never split at the shot or rally level (ShuttleSet provides official splits)
3. **Use both players' data** — opponent context is essential for shot type discrimination
4. **Normalise all modalities to the same coordinate space** — use court coordinates via homography for everything spatial
5. **Apply data augmentation** — spatial (flip, rotate, scale) and temporal (speed jitter, frame dropout)
6. **Handle class imbalance** — weighted loss, oversampling, or focal loss
7. **Use pre-trained backbones** where possible (especially for 3D CNN — Kinetics-pretrained I3D/SlowFast)
8. **Monitor validation metrics** for checkpointing, not training loss
9. **Use early stopping** with patience on validation performance
10. **Preserve training scripts** in the repository for reproducibility
11. **Use proper temporal sampling** — attention-weighted or multi-scale, not uniform stride subsampling
12. **Apply normalisation consistently** between training and inference
13. **Verify dataset size claims** — count unique physical strokes, not inflated annotation counts
14. **If building a player grading system, ensure it's downstream of accurate classification** — grading on misclassified data is meaningless

---

## Successor Project Requirements

- **Two architectures:**
  1. 3D CNN → timeseries (video-level features, e.g. I3D, SlowFast, or similar)
  2. Keypoint → timeseries (skeleton-based, building on BST or similar transformer approach)
- **Same dataset:** ShuttleSet (use official train/val/test splits)
- **Benchmark:** Must significantly outperform LSTM (36.25%) and BST (53.6%) — target SOTA range (90-98%)
- **Output:** Client-facing Gradio GUI for model comparison
- **Data source for setup:** Updated official BST repo (clean data loading, feature normalisation)
