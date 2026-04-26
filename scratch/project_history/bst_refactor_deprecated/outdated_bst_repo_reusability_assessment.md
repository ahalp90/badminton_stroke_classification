**outdated_bst_repo_reusability_assessment.md — initial analysis of the original BST repo before refactoring. All actionable items have been implemented; the rest is historical context.**

# BST Repository Reusability Assessment
**This assessment was carried out before the Claude refactor to trim reused clode blocks from the original repo. Details in: bst_models_refactor.md**

Assessment of what can be reused from the published BST (Badminton Stroke-type Transformer) research codebase for a novel comparison of two badminton stroke classifier architectures.

**Target architectures:**
- **Arch 1 (heavyweight):** 3D CNN -> deep temporal convnet + self-attention
- **Arch 2 (fast):** YOLOv26 -> MediaPipe -> TCN
- **Both:** ShuttleNet/TrackNet for shuttle trajectory features

---

## 1. Clip Generation Pipeline

### `ShuttleSet/gen_my_dataset.py` — REUSE DIRECTLY (with automation wrapper)

The single most valuable reusable component. Generates labelled MP4 stroke clips from full-match YouTube videos with correct match-level train/val/test splits.

**What it does:**
- Reads match metadata from `set/match.csv` (video IDs, player names, `downcourt` flag, YouTube URLs)
- Reads per-set shot annotations from `set/<match_name>/set{1,2,3}.csv`
- Maps player A/B to Top/Bottom using the `downcourt` flag, including mid-game court switches at 11 points in set 3 (lines 34-37, 50-66)
- Clips video segments using MoviePy based on one of three temporal strategies:
  - `middle_in_a_sec`: fixed 1-second window centred on the shot frame
  - `between_2_hits`: from previous hit to next hit (variable length)
  - `between_2_hits_with_max_limits`: same but capped at 1.5 seconds each side (**best performing, use this**)
- Outputs clips to `{split}/{player}_{stroke_type}/` folder structure — labels come from the folder names

**Match-level splits (lines 222-224):**
```python
vids_train = list(range(1, 9)) + [11] + list(range(13, 27)) + list(range(28, 35))
vids_val = list(range(35, 39)) + [41]
vids_test = [39, 40, 42, 43, 44]
```

**How to use it:**
- For Arch 1 (3D CNN): the generated MP4 clips are your direct input. Load them as video tensors.
- For Arch 2 (YOLO -> MediaPipe -> TCN): run YOLO + MediaPipe on these clips to extract keypoints.
- For shuttle features: run TrackNetV3 on these clips to get shuttle trajectories.

**Modifications needed:**
- Currently requires 6 manual runs (2 players x 3 splits), editing `player` and `set_name` variables each time. Wrap in a single automated script.
- After generation, must manually delete 7 individual clips marked "removed" in `flaw_shot_records.csv`. Automate this by filtering during generation or with a post-generation cleanup script.

### `ShuttleSet/utils.py` — REUSE DIRECTLY

Utility functions imported by `gen_my_dataset.py`:

| Function | Purpose | Reusability |
|---|---|---|
| `frameNum_2_time(frame_number, fps)` | Converts frame number to `HH:MM:SS.ssssss` time string for MoviePy's `subclip()` | Direct reuse in any clip generation |
| `time_2_frameNum(time_str, fps)` | Inverse conversion | Direct reuse |
| `write_video(output_path, cap, start_t, num_frames)` | OpenCV-based video writer (alternative to MoviePy) | Reference — MoviePy version is what's actually used |
| `is_time_timeStr_convert_correct(test_frame_len, fps)` | Validation utility | Reference |

### `ShuttleSet/get_each_class_total.py` — REUSE AS VERIFICATION TOOL

Reads all `set/*.csv` annotations, applies the same Top/Bottom player mapping logic (including set 3 court changes), counts strokes per type per match, writes to `class_total_gen.xlsx`. Run after clip generation to verify your counts match `class_total.xlsx`.

---

## 2. ShuttleSet Annotation Data

### `ShuttleSet/set/match.csv` — REUSE DIRECTLY

Master index of all 44 matches. Key columns:
- `id`: video ID (1-44, with gaps at 9, 10, 12, 27)
- `video`: folder name / video file name stem
- `downcourt`: boolean — whether player A starts on top (0) or bottom (1) of the court
- `url`: YouTube download link
- `set`: number of sets played (2 or 3)
- `winner`, `loser`: player names

### `ShuttleSet/set/homography.csv` — REUSE FOR COURT PROJECTION

Pre-computed 3x3 homography matrices per match, plus four court corner coordinates in camera pixel space (`upleft_x/y`, `upright_x/y`, `downleft_x/y`, `downright_x/y`). Used by the normalization code to project camera coordinates to court coordinates via `cv2.perspectiveTransform()`.

**How to use:** If either architecture benefits from court-relative player positions or shuttle trajectory, use these matrices to project pixel coordinates to a normalised court coordinate system [0,1].

### `ShuttleSet/set/<match_name>/set{1,2,3}.csv` — REUSE FOR LABELS + POTENTIAL AUXILIARY FEATURES

Per-set shot-level annotations. Currently only 5 columns are used by `gen_my_dataset.py`: `rally`, `ball_round`, `frame_num`, `player`, `type`. The remaining columns are untapped:

| Column | Description | Potential use |
|---|---|---|
| `hit_x`, `hit_y` | Shuttle position at the moment of the stroke (camera pixels) | Auxiliary input feature; multi-task supervision |
| `hit_height` | Height category of the hit (1=low, 2=high) | Could inform stroke classification |
| `hit_area` | Court zone of the hit (numbered areas) | Spatial context feature |
| `landing_x`, `landing_y` | Where the shuttle landed (camera pixels) | Stroke outcome feature |
| `landing_height`, `landing_area` | Landing height/zone | Stroke outcome feature |
| `aroundhead` | Boolean — was it an around-the-head shot? | Technique flag |
| `backhand` | Boolean — was it a backhand shot? | Technique flag |
| `player_location_x/y/area` | Position of the striking player | Already available as a feature in BST; reusable |
| `opponent_location_x/y/area` | Position of the opponent | Contextual feature for stroke prediction |
| `lose_reason`, `win_reason` | Why the rally ended | Not directly useful for classification |
| `getpoint_player` | Who won the rally | Not directly useful for classification |
| `flaw` | Quality flag (referenced in flaw_shot_records.csv) | Data quality filtering |

---

## 3. Supporting Data Files

### `ShuttleSet/my_raw_video_resolution.csv` — MUST RECREATE

Maps video IDs to download resolution. Three resolution tiers in the author's downloads:
- 1280x720 (majority)
- 854x480 (videos 11, 17)
- 640x360 (videos 30, 32, 41, 42)

Videos 9, 10, 12, 27 are absent (excluded matches).

**How to use:** You must create your own version matching the resolutions you actually download. This CSV is consumed by `prepare_train_on_shuttleset.py`'s normalization functions to correctly scale coordinates. If you use TrackNetV3 for shuttle features, resolution information is needed for normalization.

### `ShuttleSet/flaw_shot_records.csv` — REUSE AS DATA QUALITY REFERENCE

Documents errors found in the original ShuttleSet annotations. Two types of entries:

| Measure | Meaning | Action |
|---|---|---|
| `modified` | Timing/ordering errors already fixed in this repo's `set/*.csv` files | None — just use this repo's CSVs |
| `removed` | Whole-match or individual shot exclusions | Delete clips after generation |

**Whole-match exclusions (already absent from split lists):**
- Video 9: "whole labeling incorrect"
- Video 10: "whole labeling incorrect"
- Video 12: "all frame numbers incorrect"
- Video 27: "video removed"

**Individual shot removals (7 shots):** Must be manually deleted after clip generation. Each is identified by `match`, `set`, `rally`, `ball_round`.

### `ShuttleSet/class_total.xlsx` — REUSE FOR CLASS WEIGHT COMPUTATION

Complete per-match stroke count distribution for all 19 original stroke types across all 44 matches, with train/val/test summary totals. The totals row is critical for computing class weights for weighted loss / focal loss:

**Sample of class imbalance (Available Total column):**
| Stroke type | Count | % |
|---|---|---|
| 放小球 (drop) | 2,982 | 17.8% |
| 挑球 (lob) | 2,343 | 14.0% |
| 殺球 (smash) | 1,287 | 7.7% |
| 防守回挑 (defensive lob) | 123 | 0.7% |
| 小平球 (flat drive variant) | 22 | 0.1% |

**How to use:** Extract the train split totals to compute inverse-frequency class weights for `torch.nn.CrossEntropyLoss(weight=...)` or focal loss. The severe imbalance (小平球 at 22 samples vs 放小球 at 2,982) means class weighting is essential for good per-class F1.

### `ShuttleSet/class_total_gen.xlsx` — REFERENCE ONLY

Working copy generated by `get_each_class_total.py`. Has per-player sheets (Top/Bottom) with the same counts. Less useful than `class_total.xlsx` which has the consolidated view.

---

## 4. TrackNetV3 Integration Pattern

### `stroke_classification/preparing_data/prepare_train_on_shuttleset.py` — PARTIAL REUSE

This file runs three extraction steps. Only the shuttle tracking step and normalization functions are relevant to your project.

**Shuttle trajectory extraction (subprocess call pattern):**
The function `detect_shuttlecock_by_TrackNetV3_with_attension()` invokes TrackNetV3 as a subprocess on each video clip. The shuttle trajectory output is `(t, 2)` — x,y coordinates per frame.

**How to use:** Adapt this subprocess invocation pattern to run TrackNetV3 on your generated clips. The shuttle trajectory becomes a shared input feature for both architectures.

**Two TrackNetV3 variants are referenced in the README:**
- [TrackNetV3 (using attention)](https://github.com/alenzenx/TrackNetV3) — used in the BST paper
- [TrackNetV3 (with rectification module)](https://github.com/qaz812345/TrackNetV3)

**Normalization functions worth reusing/referencing:**

| Function | Lines | What it does | Reusability |
|---|---|---|---|
| `normalize_shuttlecock(arr, v_width, v_height)` | 150-159 | Divides x,y by video resolution to get [0,1] range | **Direct reuse** for shuttle features |
| `normalize_position(arr, court_info)` | 102-114 | Normalises court coordinates by court boundary to [0,1] | **Reference** — useful if you want court-relative player positions |
| `to_court_coordinate(arr_camera, vid, all_court_info, res_df)` | 79-99 | Projects camera pixels to court coords via homography | **Reference** — uses `homography.csv` |
| `normalize_joints(arr, bbox, v_height, center_align)` | 117-147 | Bbox-relative joint normalization, normalised by diagonal | **Not applicable** — MediaPipe outputs normalised coords already |
| `check_pos_in_court(keypoints, vid, all_court_info, res_df)` | 162+ | Checks if detected people are on-court, returns court-normalised foot positions | **Reference** — relevant if you need to filter/identify players |

**What you do NOT need from this file:**
- MMPose pose estimation (you're using MediaPipe)
- The COCO skeleton processing
- The full 3-step pipeline orchestration

**Note:** The file has hardcoded Windows paths in its `__main__` block (`C:/MyResearch/TrackNetV3-main`, etc.). Easy to change.

---

## 5. Label Definitions

### `stroke_classification/preparing_data/shuttleset_dataset.py` (lines 10-53) — REUSE DIRECTLY

Two label mapping functions that define class name -> index mappings:

**`get_merged_stroke_types()` — 25 classes (recommended):**
```
12 stroke types x 2 sides (Top/Bottom) + 1 unknown = 25 classes
放小球, 擋小球, 殺球, 挑球, 長球, 平球, 切球, 推球, 撲球, 勾球, 發短球, 發長球
```

**`get_stroke_types()` — 35 classes:**
```
17 stroke types x 2 sides + 1 unknown = 35 classes
(includes 5 additional rare subtypes: 點扣, 防守回挑, 小平球, 後場抽平球, 過渡切球, 防守回抽)
```

**How to use:** Import these functions for consistent label encoding across both architectures. The 35->25 merging folds rare subtypes into their parent categories and moves 小平球 into "unknown" (`未知球種`).

**The merging is a folder-level operation:** After generating 35-class clips, you physically move `Top_小平球` and `Bottom_小平球` clips into the `未知球種` folder. The code then picks up labels from folder names.

---

## 6. Temporal Sequence Handling

### `shuttleset_dataset.py` — `make_seq_len_same()` (lines 71-104) — ADAPT CONCEPT

Normalises variable-length sequences to a fixed target length via stride-based downsampling + zero-padding.

**Algorithm:**
1. If `video_len > target_len`: compute stride, subsample every `stride`-th frame, pad remainder if needed
2. If `video_len <= target_len`: zero-pad to `target_len`
3. Returns actual video length alongside padded data (for attention masking)

**How to use:**
- **Arch 1 (3D CNN):** You'll need a similar frame sampling strategy — uniform temporal sampling to a fixed number of frames. The stride-based approach here is one option; you might also consider random temporal crops during training.
- **Arch 2 (TCN):** Directly applicable — MediaPipe keypoint sequences need padding/downsampling to a fixed length for batching.
- **Shuttle features:** Same temporal alignment needed — shuttle trajectory must match the sequence length of the primary input.

**The current implementation is coupled to `(joints, pos, shuttle)` tuple format.** You'd refactor to handle your own data shapes, but the stride/padding logic transfers.

**Caution from predecessor analysis:** Stride-based subsampling can skip critical motion peaks (e.g., a smash spanning only 3-5 frames with stride=5 would lose it entirely). Consider interpolation-based resampling as an alternative for very short, fast strokes.

---

## 7. TCN Module

### `stroke_classification/model/tempose.py` — TCN class (lines 132-153) — REFERENCE

A simple dilated 1D temporal convolutional network:

```python
class TCN(nn.Module):
    def __init__(self, in_channel, channels: list[int], kernel_size=5, drop_p=0.3):
        # Stacked Conv1d layers with increasing dilation (1, 3, 5, ...)
        # Each layer: Conv1d -> BatchNorm1d -> GELU -> Dropout
        # Padding preserves sequence length
```

**How to use for Arch 2:** Your architecture includes a TCN stage after MediaPipe keypoint extraction. This implementation is a clean starting point but is intentionally lightweight (2 layers for embedding, not deep classification). For your "deep temporal convnet" you'll likely want:
- More layers / residual connections
- Possibly separate TCN branches for skeleton and shuttle features
- Output projection to classification head

Worth reading as a reference for the dilated convolution + padding calculation pattern.

---

## 8. Training Infrastructure

### `stroke_classification/main_on_shuttleset/bst_main.py` — REUSE AS TEMPLATE

The training loop and `Task` class provide a solid, tested pattern.

**`train_one_epoch()` (lines 47-85):**
- Applies `RandomTranslation_batch` augmentation to joints (not bones) before forward pass
- Standard: forward -> loss -> backward -> optimizer step -> scheduler step
- Returns average loss

**`validate()` (lines 88-137):**
- Computes per-class TP, TN, FP, FN
- Calculates macro F1-score and min F1-score
- Uses `torcheval.metrics.functional.multiclass_f1_score`

**`test()` and `test_topk()` (lines 140-200+):**
- Collects predictions and ground truth for confusion matrix plotting
- Top-k accuracy computation

**`train_network()` (lines ~200-280):**
- AdamW optimizer
- Cosine scheduler with warmup via `transformers.get_cosine_schedule_with_warmup` (400 warmup steps)
- `CrossEntropyLoss(label_smoothing=0.1)`
- Early stopping on macro F1-score (300 epoch patience)
- Saves best model state dict

**`Task` class:**
- Orchestrates data loading, model construction, weight save/load, testing
- `get_network_architecture()` acts as a model factory

**How to use:** Copy the training loop structure. Key modifications needed:
1. **Replace loss function** with weighted `CrossEntropyLoss` or focal loss (compute weights from `class_total.xlsx` train totals)
2. **Replace `Hyp` namedtuple** with a proper config system (YAML, argparse, or dataclass)
3. **Adapt data unpacking** — BST's loop unpacks `(human_pose, pos, shuttle), video_len, labels`; your architectures will have different input shapes
4. **Add logging** — see `bst_main_summary_writer.py` for a TensorBoard variant

**Critical gotcha:** The default hyperparameters in `bst_main.py` include `train_partial=0.25` (line 43), meaning only 25% of training data is used. The `bst_main_summary_writer.py` variant has `train_partial=1` and `seq_len=100` — these are the "production" settings.

### `stroke_classification/main_on_shuttleset/bst_main_summary_writer.py` — REFERENCE for TensorBoard logging

Same as `bst_main.py` but with `SummaryWriter` integration for TensorBoard. Also has corrected defaults (`train_partial=1`, `seq_len=100`). Use as reference for adding logging to your training loops.

### `stroke_classification/main_on_shuttleset/bst_infer.py` — REFERENCE for Gradio backend

Clean inference example with a simplified `Task` class:
- Loads pre-trained weights
- Creates a DataLoader
- Runs `@torch.no_grad()` inference
- Maps prediction indices to class names

**How to use:** Template for your Gradio GUI's backend inference functions. Adapt the data loading to accept single video uploads rather than batch .npy files.

---

## 9. Evaluation Utilities

### `stroke_classification/result_utils.py` — REUSE DIRECTLY

Architecture-agnostic evaluation visualisation:

| Function | Purpose | Reusability |
|---|---|---|
| `show_f1_results(model_name, f1_score_each, class_ls, show_details)` | Prints per-class F1 scores as a pandas DataFrame with average and minimum | Direct reuse — just pass your class list |
| `plot_confusion_matrix(y_true, y_pred, need_pre_argmax, model_name, ...)` | Generates side-by-side precision-normalised and recall-normalised confusion matrices | Direct reuse — works with any y_true/y_pred arrays |

**How to use:** Import and call after testing each architecture. Pass in your own class name list from `get_merged_stroke_types()` or `get_stroke_types()`.

---

## 10. Collation Pattern

### `stroke_classification/preparing_data/prepare_train_on_shuttleset_merged.py` — REFERENCE for parallel data processing

Shows a clean pattern for collating individual per-sample files into large arrays for fast training:
1. `ThreadPoolExecutor` for parallel .npy file loading (I/O bound)
2. `ProcessPoolExecutor` for parallel bone computation and padding (CPU bound)
3. Stack into large arrays and save as single .npy files per split

**How to use:** If your feature extraction produces per-clip outputs (e.g., one .npy per clip from MediaPipe or TrackNetV3), adapt this pattern to collate them into large arrays for efficient DataLoader access.

---

## 11. Environment Setup Reference

### `stroke_classification/preparing_data/prepare_train_env.txt` — REFERENCE ONLY

Documents the MMPose venv setup (Python 3.11, PyTorch 2.3.1+cu121, mmpose, mmdet, mmcv). Not directly useful since you're using MediaPipe, but documents:
- The CUDA/PyTorch version combination that works
- A compatibility hack: line 25 notes modifying `mmdet/__init__.py` version string to force compatibility

**Your project will need two separate environments:**
1. Feature extraction (YOLO + MediaPipe + TrackNetV3)
2. Training (PyTorch + your model code + Gradio)

---

## 12. Pre-computed Data on Google Drive

The README links to pre-computed .npy files:

| Dataset | seq_len | Link purpose |
|---|---|---|
| ShuttleSet merged 25-class | 30 | `dataset_npy` — individual .npy files per stroke |
| ShuttleSet merged 25-class | 30 | `dataset_3d_npy` — 3D pose variant |
| ShuttleSet merged 25-class | 100 | `dataset_npy_between_2_hits_with_max_limits` |
| ShuttleSet 35-class | 30, 30 (3D), 100 | Same variants |
| BadmintonDB 18-class | 72 | Individual .npy files |
| TenniSet 6-class | 100 | Individual .npy files |

**Relevance to your project:** These contain COCO-skeleton keypoints from MMPose — **not useful for your MediaPipe-based Arch 2**. However, the shuttle trajectory data (`*_shuttle.npy`) within these archives could potentially be extracted and reused if the temporal alignment matches your clips. Would need investigation.

Pre-trained BST weights are also available — not useful for your custom architectures, but could serve as a baseline comparison point.

---

## 13. Things NOT Reusable

For completeness, these components exist in the repo but are not useful for your project:

| Component | File | Why not useful |
|---|---|---|
| BST model (all variants) | `model/bst.py` | Custom architectures being built |
| TemPose model (all variants) | `model/tempose.py` | Custom architectures (TCN module excepted as reference) |
| ST-GCN, BlockGCN, SkateFormer, ProtoGCN | `model/stgcn.py`, `model/blockgcn.py`, `model/skateformer.py`, `model/protogcn.py` | Comparison baselines, not needed |
| All per-model main files | `main_on_shuttleset/*_main.py` (except bst_main) | Architecture-specific training scripts |
| COCO bone pairs | `shuttleset_dataset.py:56-68` | MediaPipe uses 33 landmarks, not COCO 17 |
| Bone computation functions | `shuttleset_dataset.py:107-131` | Tied to COCO skeleton topology |
| RandomTranslation augmentation | `shuttleset_dataset.py:134-170` | Tied to COCO joint structure |
| Dataset_npy / Dataset_npy_collated | `shuttleset_dataset.py:180-400+` | Shaped for BST's `(pose, pos, shuttle)` tuple format |
| MMPose extraction pipeline | `prepare_train_on_shuttleset.py` (steps 2-3) | Arch 2 uses MediaPipe |
| Joint normalization | `prepare_train_on_shuttleset.py:117-147` | MediaPipe outputs pre-normalised coordinates |
| BadmintonDB / TenniSet pipelines | `*badDB*`, `*tenniSet*` files | Different datasets |

---

## 14. Known Bugs and Gotchas

| Issue | Location | Impact | Action |
|---|---|---|---|
| `train_partial=0.25` default | `bst_main.py:43` | Silently uses only 25% of training data | If referencing this file's hyperparameters, ensure you set `train_partial=1` |
| Augmentation no-op in Dataset_npy | `shuttleset_dataset.py:227` | `self.random_shift(joints)` discards return value — augmentation silently does nothing | Only affects the non-collated `Dataset_npy` class; collated path works correctly |
| flaw_shot_records.csv not consumed by code | `ShuttleSet/flaw_shot_records.csv` | 7 individual bad shots are generated but never automatically removed | Must manually delete or add filtering logic |
| 35->25 class merging is manual | README step 5 | Must physically move `小平球` clips into `未知球種` folder | Automate this step |
| Shuttlecock normalization is camera-space | `prepare_train_on_shuttleset.py:150-159` | Predecessor analysis flagged this, but it's a deliberate design choice validated by SOTA results | Do not "fix" — camera-space trajectory preserves visual motion information |
| Windows paths in __main__ | `prepare_train_on_shuttleset.py` | Hardcoded `C:/` paths | Only in `__main__` block, easy to change |
| seq_len=30 vs 100 | Multiple files | Best results use seq_len=100 with `between_2_hits_with_max_limits` | Use seq_len=100 strategy for your project |
| No class imbalance handling anywhere | All training code | `CrossEntropyLoss(label_smoothing=0.1)` with no class weighting | Must add weighted loss or focal loss |

---

## Summary: What to Fork vs. What to Build

### Fork/reuse directly
1. `ShuttleSet/gen_my_dataset.py` + `ShuttleSet/utils.py` — clip generation with proper splits
2. `ShuttleSet/set/` — all annotation data (match.csv, homography.csv, per-set CSVs)
3. `ShuttleSet/flaw_shot_records.csv` + `ShuttleSet/get_each_class_total.py` — data quality tools
4. `ShuttleSet/class_total.xlsx` — class distribution for computing weights
5. Label definitions from `shuttleset_dataset.py` (`get_stroke_types()`, `get_merged_stroke_types()`)
6. `normalize_shuttlecock()` from `prepare_train_on_shuttleset.py`
7. `result_utils.py` — evaluation visualisation
8. Training loop pattern from `bst_main.py` (adapt, don't copy wholesale)

### Use as reference/inspiration
1. `make_seq_len_same()` — temporal padding/downsampling logic
2. TCN module in `tempose.py` — clean dilated convnet reference
3. `to_court_coordinate()` / `normalize_position()` — court projection pattern
4. `prepare_train_on_shuttleset_merged.py` — parallel collation pattern
5. `bst_main_summary_writer.py` — TensorBoard integration
6. `bst_infer.py` — inference pattern for Gradio backend
7. TrackNetV3 subprocess invocation pattern

### Build new
1. 3D CNN model + video frame Dataset class
2. YOLO -> MediaPipe extraction pipeline + keypoint Dataset class
3. Deep TCN / temporal convnet + self-attention model
4. Class imbalance handling (weighted loss / focal loss)
5. Gradio GUI
6. Unified config system
7. Video and keypoint augmentation pipelines
