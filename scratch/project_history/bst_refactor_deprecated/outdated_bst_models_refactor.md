# BST Model Refactoring — Complete Analysis & Notes

## What Was Done

### Model file: `stroke_classification/model/bst.py`

**Before:** 857 lines, 4 near-identical model classes (`BST_0`, `BST`, `BST_AP`, `BST_CG_AP`) plus two building blocks (`MultiHeadCrossAttention`, `CrossTransformerLayer`).

**After:** 430 lines (including generous comments). One unified `BST` class with three boolean flags (`use_ppf`, `use_cg`, `use_ap`), backward-compatible aliases via `functools.partial`, and the same two building blocks unchanged.

**What was removed:** ~530 lines of pure copy-paste duplication. The four classes differed only at 4 branch points in their `forward()` methods and in which optional modules they created in `__init__`. The `init_weights()` and `init_weights_recursive()` methods were identical across all four.

### Training scripts: `stroke_classification/main_on_shuttleset/`

**Before:** 4 files totalling 1627 lines:
- `bst_main.py` (506 lines) — main training script, 5 serial runs, `BST_CG_AP`, `seq_len=30`
- `bst_main_summary_writer.py` (489 lines) — identical but with TensorBoard logging, `BST_AP`, `seq_len=100`
- `bst_backbone_main.py` (467 lines) — identical but restricted to `BST_0`
- `bst_infer.py` (165 lines) — inference only, loads checkpoint and predicts

**After:** 2 files totalling ~640 lines:
- `bst_train.py` (~510 lines) — consolidated training with TensorBoard always on
- `bst_infer.py` (~130 lines) — simplified inference

**Deleted:** `bst_main.py`, `bst_main_summary_writer.py`, `bst_backbone_main.py`

**What changed in the consolidation:**
- The only real code differences across the 3 training scripts were: (a) which model to instantiate (now a single dict lookup), (b) whether to pass `pos` to the model (now always passed; BST ignores it if `use_ppf=False`), (c) 3 lines of TensorBoard logging (now always included), (d) different hyperparameter values (now one `hyp` at the top).
- `Task.get_network_architecture()` collapsed from 4 copy-pasted match/case blocks (~35 lines each) to a dict lookup + one `BST(...)` call (~20 lines total).
- `Task.seek_network_weights()` unified the weight filename generation to handle both `BST_0` (no underscore) and `BST_CG_AP` (underscore-separated) names.

---

## BST Architecture — How It Works

### Pipeline Overview

```
Input: skeleton keypoints + shuttle xy + player court positions + video_len (frame count)
  │
  ├── [PPF] Pose Position Fusion (optional)
  │     Court xy → MLP → multiply with skeleton features (modulates pose by position)
  │
  ├── TCN (temporal convolution): skeleton → (b, n, t, d_model)
  ├── TCN (temporal convolution): shuttle  → (b, 1, t, d_model)
  │
  ├── Concatenate along player axis → 3 streams: [player1, player2, shuttle]
  │
  ├── Temporal Transformer (self-attention per stream, independently)
  │     Each stream gets a CLS token prepended; attention is masked for padding
  │     Output: CLS tokens summarise each stream's temporal dynamics
  │
  ├── Cross Transformer (player1 attends to shuttle; player2 attends to shuttle)
  │     Lets each player's representation incorporate shuttle trajectory info
  │
  ├── Interactional Transformer (self-attention on each player-shuttle pair)
  │     Models cross-player dynamics with a new CLS token per player
  │
  ├── [AP] Aim Player (optional)
  │     Cosine similarity between each player's interactional CLS and shuttle CLS
  │     Derives alpha ∈ [0,1] to weight player contributions
  │
  ├── [CG] Clean Gate (optional)
  │     Element-wise minimum of both players' interactional CLS → MLP → subtract from shuttle CLS
  │     Removes shared/redundant player signal from shuttle representation
  │
  └── MLP Head → class logits (25 or 35 stroke types)
```

### Input Shapes

| Input | Shape | Description |
|-------|-------|-------------|
| `JnB` | `(b, t, n, input_dim)` | Skeleton features. `n=2` players. `input_dim` depends on pose_style. |
| `shuttle` | `(b, t, 2)` | Shuttle xy per frame |
| `pos` | `(b, t, n, 2)` | Player court xy per frame (only used if `use_ppf=True`) |
| `video_len` | `(b,)` | Real frame count per sample (rest is zero-padding) |

### Key Dimensions (defaults)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `d_model` | 100 | Feature dimension throughout the transformer pipeline |
| `d_head` | 128 | Dimension per attention head |
| `n_head` | 6 | Number of parallel attention heads |
| `depth_tem` | 2 | Number of layers in temporal transformer |
| `depth_inter` | 1 | Number of layers in interactional transformer |
| `seq_len` | 30 or 100 | Max sequence length (frames) |

### Model Variants

| Name | Flags | MLP Head Input | ~Params | Notes |
|------|-------|---------------|---------|-------|
| `BST_0` | ppf=F, cg=F, ap=F | `3 × d_model` | 1.81M | Backbone only |
| `BST` | ppf=T, cg=F, ap=F | `3 × d_model` | 1.83M | + position fusion |
| `BST_CG` | ppf=T, cg=T, ap=F | `3 × d_model` | 1.85M | + clean gate |
| `BST_AP` | ppf=T, cg=F, ap=T | `2 × d_model` | 1.79M | + aim player (drops shuttle from head) |
| `BST_CG_AP` | ppf=T, cg=T, ap=T | `3 × d_model` | 1.85M | Full model |

---

## Module-Level Analysis

### PPF (Pose Position Fusion)

**What it does:** Projects 2D court coordinates through an MLP to match the skeleton feature dimension, then multiplies element-wise with skeleton features and adds a residual: `JnB = JnB * pos_impact + JnB`, equivalent to `JnB * (1 + pos_impact)`.

**Where:** Very first operation in `forward()`, before the TCN.

**Purpose:** Lets the model learn that the same skeleton pose means different things at different court positions.

### TCN (Temporal Convolution Network)

**What it does:** 1D dilated convolutions along the time axis. Two layers, both outputting `d_model` channels.

**Where:** After PPF, before the transformer stack.

**Purpose:** Extracts local temporal patterns from raw features. The dilated convolutions give a wider receptive field than standard convs without increasing parameter count.

**Note:** TCN processes padded frames without masking — garbage features at padded positions are handled later by transformer attention masks.

### Temporal Transformer

**What it does:** Self-attention across time within each stream (player1, player2, shuttle) independently. All three streams are batched together for efficiency (`b*n` batch dimension).

**Where:** After TCN, before cross transformer.

**Key concept — CLS token:** A learnable vector prepended to the sequence at position 0. After attention, this token has "seen" the whole sequence and serves as a fixed-size summary. This is how transformers produce a single output vector from a variable-length sequence (unlike LSTMs which use the final hidden state).

**Key concept — Positional embeddings:** Added to the input so the transformer knows frame order. Transformers have no built-in notion of position (unlike LSTMs which process sequentially).

**Key concept — Padding mask:** Boolean mask that prevents attention from attending to zero-padded frames. Padded positions get -inf attention scores, which softmax converts to zero weight.

### Cross Transformer

**What it does:** Cross-attention where each player's representation queries the shuttle trajectory. Player 1 attends to shuttle; player 2 attends to shuttle (using the same shared cross-transformer weights).

**Where:** After temporal transformer, before interactional transformer.

**Purpose:** Fuses player and shuttle information. After this, each player's representation encodes how their motion relates to the shuttle trajectory.

**Key concept — Cross-attention vs self-attention:** In self-attention, one sequence attends to itself (Q, K, V all from same input). In cross-attention, queries come from one sequence and keys/values from another. Here, the player "asks questions" and the shuttle "provides answers."

### Interactional Transformer

**What it does:** Self-attention on each player-shuttle pair independently, with a new CLS token. Structurally identical to the temporal transformer but with `depth_inter=1` layer.

**Where:** After cross transformer, before the optional modules.

**Purpose:** Further refines the player-shuttle interaction representations. The new CLS token captures a summary of the full player-shuttle interaction.

### AP (Aim Player)

**What it does:** Computes cosine similarity between each player's interactional CLS token and the shuttle CLS token. Derives `alpha = (sim_p1 - sim_p2 + 2) / 4` which maps to [0, 1]. Scales player conclusions by `alpha` and `(1-alpha)`.

**Where:** After interactional transformer, before CG.

**Purpose:** Determines which player is the "aim player" (the one who hit the shuttle) and weights their contribution accordingly.

**Observations from our analysis:**
- The alpha formula is a hardcoded linear mapping with no learned temperature or per-class adaptation.
- Results show AP alone provides negligible improvement over base BST — likely because the cosine similarity signal at this late stage is too crude to add useful information beyond what attention already learned.
- Moving AP earlier (after temporal transformer) and annealing it out could theoretically help warm-start training, but the semantic meaning of cosine similarity is weaker before cross-attention has mixed player and shuttle information.

### CG (Clean Gate)

**What it does:** Takes element-wise minimum of both players' interactional CLS tokens → passes through MLP → subtracts result ("dirt") from shuttle CLS token.

**Where:** After AP (if present), before MLP head.

**Purpose:** The element-wise minimum isolates what's shared/redundant between both players' representations. The MLP learns to transform this into a noise signal that's subtracted from the shuttle representation, making it more discriminative.

**Observations from our analysis:**
- CG alone shows marginal improvement.
- The MLP inside CG (d_model → d_model → d_model, ~20k params) might be unnecessary — raw `shuttle_cls = shuttle_cls - torch.minimum(p1, p2)` could work similarly.
- CG + AP together show a small synergistic improvement, likely because CG cleans the shuttle representation, making AP's cosine similarity comparison more meaningful.

### MLP Head

**What it does:** LayerNorm → Linear → GELU → Dropout → Linear. Maps concatenated representations to class logits.

**Input dimension varies:**
- AP without CG: `2 × d_model` (shuttle info encoded in alpha, not concatenated)
- All others: `3 × d_model` (player1 + player2 + shuttle)

---

## Variable-Length Input Handling

The data pipeline (`make_seq_len_same()` in `shuttleset_dataset.py`) forces all sequences to a fixed `seq_len`:

- **Shorter than seq_len:** Zero-padded at the end. `video_len` records the real length.
- **Longer than seq_len:** Uniformly strided (subsampled) to fit, e.g., a 60-frame rally with `seq_len=30` takes every other frame.

The TCN processes padded frames without masking (produces garbage features at padded positions). The transformer attention mask prevents these from influencing the CLS token.

**Limitation:** Stride-based downsampling is lossy. Fast strokes (smash in 3-5 frames) can be entirely skipped by large strides. Longer `seq_len` reduces this but increases transformer compute quadratically (attention is O(t²)).

**Alternatives discussed:**
- DTW normalization to a median temporal profile (principled but introduces design decisions and removes raw tempo signal)
- Longer window with sequence packing or flash attention (simpler, preserves raw timing, compute cost manageable for this dataset size)
- Non-uniform sampling based on shuttle acceleration (allocates frame budget around contacts)

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Adam with decoupled weight decay |
| Learning rate | 5e-4 | With cosine annealing after 400-step warmup |
| Loss | CrossEntropyLoss | label_smoothing=0.1 |
| Early stopping | 300 epochs patience | On macro F1 |
| Max epochs | 1600 | Rarely reached due to early stopping |
| Augmentation | RandomTranslation | Applied to joints only (bones are translation-invariant) |
| Batch size | 128 | |
| Scheduler | Cosine with warmup | 0.25 cycle (quarter-cosine decay) |

**Weight naming convention:** `{model_name}_{pose_style}_{data_info}_{merged_flag}_{serial_no}.pt`
Example: `bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt`

---

## Building Blocks (in `tempose.py`, unchanged)

| Class | Purpose |
|-------|---------|
| `TCN` | Dilated 1D convolutions for temporal feature extraction |
| `MLP` | Linear → GELU → Dropout → Linear |
| `MLP_Head` | LayerNorm → MLP (classification head) |
| `FeedForward` | MLP + Dropout (used inside transformer layers) |
| `MultiHeadAttention` | Standard self-attention |
| `TransformerLayer` | LayerNorm → Attention → Residual → LayerNorm → FeedForward → Residual |
| `TransformerEncoder` | Stack of TransformerLayers |

These are clean, correctly sized for their purpose, and reusable for new architectures.

---

## Data Loader Coupling Points

These were flagged but not addressed in the refactoring:

1. **Forward signature contract:** The data loader returns `((human_pose, pos, shuttle), video_len, labels)`. All training functions destructure this tuple. Changing the data format requires changing `train_one_epoch`, `validate`, `test`, and `test_topk`.

2. **`make_seq_len_same()`:** Determines how sequences are padded/strided to `seq_len`. The model's `seq_len` init param must match. This is the function to modify for DTW or longer windows.

3. **Skeleton format:** `RandomTranslation_batch` and bone computation are COCO 17-joint specific. Switching to MediaPipe 33 joints would require new augmentation and bone pair definitions.

4. **3D CNN extension:** Adding a racket blob feature means adding a new element to the data loader tuple and a new parameter to `BST.forward()`. The unified class makes this a single-file change.

---

## Weight Compatibility

Existing `.pt` checkpoint files are compatible with the refactored model because:
- Module names in `state_dict()` are determined by attribute names in `__init__` (e.g., `self.tcn_pose`, `self.encoder_tem`), which are unchanged.
- Optional modules (`mlp_positions`, `mlp_clean`, `cos_sim`) use the same attribute names as the original classes.
- The `partial` aliases create instances of the same `BST` class with the same module structure.

To load an existing checkpoint: `BST(use_ppf=True, use_cg=True, use_ap=True, ...).load_state_dict(torch.load('weight.pt'))` — or equivalently via the alias: `BST_CG_AP(...).load_state_dict(...)`.

---

## Files Modified

| File | Action | Lines before → after |
|------|--------|---------------------|
| `stroke_classification/model/bst.py` | Rewritten | 857 → 430 |
| `stroke_classification/main_on_shuttleset/bst_train.py` | Created (new) | 0 → 510 |
| `stroke_classification/main_on_shuttleset/bst_infer.py` | Rewritten | 165 → 130 |
| `stroke_classification/main_on_shuttleset/bst_main.py` | Deleted | 506 → 0 |
| `stroke_classification/main_on_shuttleset/bst_main_summary_writer.py` | Deleted | 489 → 0 |
| `stroke_classification/main_on_shuttleset/bst_backbone_main.py` | Deleted | 467 → 0 |
| `stroke_classification/model/tempose.py` | Untouched | — |
| `stroke_classification/model/blockgcn.py` | Untouched | — |

**Total:** 2484 lines → 1070 lines (57% reduction, zero functionality lost).