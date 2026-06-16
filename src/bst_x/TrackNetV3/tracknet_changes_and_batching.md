# TrackNetV3 changes and batch processing

This directory is an **inference-only fork** of the [original TrackNetV3 repo](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv3). Training, evaluation, preprocessing, and analysis modules have been removed. The inference code has been restructured to support batch processing of ~33k short clips through the pipeline without reloading model weights per clip.

---

## Files removed

| File | Purpose | Why removed |
|------|---------|-------------|
| `train.py` | Model training loop | Inference only |
| `test.py` | Evaluation, metrics, pycocotools integration | Inference only; 3 needed functions extracted to `inference_utils.py` |
| `preprocess.py` | Dataset frame extraction for training | Inference only |
| `generate_mask_data.py` | InpaintNet training data generation | Inference only |
| `error_analysis.py` | Dash-based error analysis UI | Inference only |
| `correct_label.py` | Manual label correction UI | Inference only |
| `corrected_test_label/drop_frame.json` | Label data for test correction | Inference only |
| `utils/metric.py` | Loss functions and training metrics | Inference only |
| `utils/visualize.py` | Training visualisation helpers | Inference only |

---

## Files added

### `inference_utils.py`

Three functions extracted verbatim from the original `test.py`:

- `get_ensemble_weight(seq_len, eval_mode)` — weighting tensor for temporal ensemble
- `predict_location(heatmap)` — largest-contour bounding box from a heatmap
- `generate_inpaint_mask(pred_dict, th_h)` — occlusion mask for InpaintNet

**Logic is unchanged.** The only reason these were moved is that `test.py` has a module-level `from pycocotools.coco import COCO` that crashes if pycocotools is not installed. Since `predict.py` only needs these three functions (none of which use pycocotools), they now live in a standalone module.

### `batch_predict.py`

New batch inference wrapper. Loads models once via `load_models()`, reads a text file of clip paths, and calls `predict_video()` per clip. Designed to be invoked as a subprocess by `pipeline/shuttle_extractor.py`.

Key features:
- **Resume logic**: skips clips whose output CSV already exists (safe to re-run after crashes)
- **Memory management**: `gc.collect()` + `torch.cuda.empty_cache()` between clips
- **Parseable progress**: prints `BATCH_PROGRESS (i/total) stem` and `BATCH_COMPLETE successes=N failures=N skipped=N`
- **Dry-run support**: `--dry_run` runs inference without writing CSVs

---

## Files modified

### `predict.py`

The original `predict.py` had all logic in `__main__`. We restructured it into reusable functions so `batch_predict.py` can call them without subprocess overhead per clip.

**Change 1 — Import path**

```python
# Original
from test import predict_location, get_ensemble_weight, generate_inpaint_mask

# Refactored
from inference_utils import predict_location, get_ensemble_weight, generate_inpaint_mask
```

Avoids the pycocotools import in `test.py`.

**Change 2 — New `load_models()` function**

Extracted model loading from `__main__` into a standalone function:

```python
def load_models(tracknet_file, inpaintnet_file=None):
    """Load TrackNet and optionally InpaintNet onto GPU.
    Returns: (tracknet, inpaintnet, tracknet_seq_len, inpaintnet_seq_len, bg_mode)
    """
```

Call once, then pass the returned values to `predict_video()` for each clip. This avoids reloading weights per clip (~8s savings each).

**Change 3 — New `predict_video()` function**

Extracted the entire inference loop from `__main__` into a callable function. The inference logic itself (TrackNet forward pass, temporal ensemble buffering, InpaintNet inpainting, heatmap thresholding, coordinate scaling) is **unchanged** from the original. Three sub-changes within the function:

1. **`num_workers=0`** (original: up to 16). Frames are already loaded into a single numpy array in RAM by `generate_frames()`. Spawning DataLoader worker subprocesses just to index into that array adds pickle/IPC overhead with no benefit, especially for short clips (~75-105 frames).

2. **Pre-resize in `generate_frames()`**: calls `generate_frames(video_file, resize_to=(WIDTH, HEIGHT))` instead of `generate_frames(video_file)`. See the `utils/general.py` section below for why this is bit-identical.

3. **`dry_run` parameter**: when True, runs the full inference pipeline but skips writing the output CSV. Used for testing.

**Change 4 — `__main__` simplified**

Now just parses arguments, calls `load_models()`, then `predict_video()`. No logic changes.

**Unchanged in `predict.py`:**
- `predict()` function (heatmap/coordinate dict construction)
- TrackNet forward pass and temporal ensemble buffer management
- InpaintNet inpainting and coordinate thresholding
- `img_scaler` coordinate scaling

### `utils/general.py`

One function modified: `generate_frames()`.

**Added `resize_to` parameter:**

```python
# Original
def generate_frames(video_file):

# Refactored
def generate_frames(video_file, resize_to=None):
```

When `resize_to` is provided (e.g. `(512, 288)`), each frame is pre-resized via `Image.fromarray(frame).resize(resize_to)` before appending to the frame list.

**Why this is bit-identical to the original path:**

The original code path loads frames at native resolution, then `Dataset.__getitem__()` resizes each frame with `Image.fromarray(img).resize(size=(self.WIDTH, self.HEIGHT))`. The pre-resize path does the exact same PIL `Image.resize()` call with the same default BICUBIC resampling and the same target dimensions — it just does it once up front instead of repeatedly in the DataLoader. Since BICUBIC interpolation is per-channel, the BGR channel ordering from cv2 (vs the RGB that PIL assumes) does not affect the interpolated pixel values.

**Added `cap.release()`:**

The original `generate_frames()` opened a `cv2.VideoCapture` but never released it. Added `cap.release()` before returning — a resource leak fix.

### `requirements.txt`

Comment-only update. No package versions changed. Added documentation explaining:
- Python 3.11 / CUDA 12.1 compatibility notes
- BST venv sharing (this module uses the same venv as the BST-X training venv (`src/bst_x/requirements.txt`))
- Upgrade path for standalone installation

### `README.md`

Rewritten for inference-only focus:
- Removed training, evaluation, preprocessing, and error analysis sections
- Added pretrained weight download instructions
- Added pipeline usage examples alongside standalone usage
- Added explicit list of removed modules with pointers to the original repo

---

## Files unchanged

| File | Contents |
|------|----------|
| `model.py` | TrackNet and InpaintNet architectures (byte-identical to original) |
| `dataset.py` | `Shuttlecock_Trajectory_Dataset`, `Video_IterableDataset` (byte-identical to original) |
| `utils/__init__.py` | Empty init file |

---

## Batch processing pipeline

### Call chain

```
pipeline/build_dataset.py          Step 6 calls extract_all_shuttles()
    |
pipeline/shuttle_extractor.py      Scans for pending clips, splits across workers
    |
TrackNetV3/batch_predict.py        Loads models once, loops over clip list
    |
TrackNetV3/predict.py              predict_video() runs inference on one clip
    |
TrackNetV3/dataset.py              DataLoader feeds frame batches to the model
```

### How it operates

1. **`shuttle_extractor.py`** scans the clips directory for `.mp4` files that don't yet have a corresponding `_ball.csv` in the output directory. It splits pending clips round-robin across N worker processes. Each worker gets a temporary text file listing its assigned clips.

2. Each worker is launched as a **subprocess** running `batch_predict.py`. The subprocess receives paths to the clip list, model weights, output directory, and batch size as command-line arguments.

3. **`batch_predict.py`** calls `load_models()` once to load TrackNet (and optionally InpaintNet) onto the GPU. It then iterates over its clip list, calling `predict_video()` for each clip. Between clips it runs `gc.collect()` and `torch.cuda.empty_cache()` to prevent memory accumulation.

4. **`predict_video()`** loads all frames from the clip via `generate_frames()` (pre-resized to 512x288), creates a `Shuttlecock_Trajectory_Dataset` wrapping the frame array, and runs the TrackNet forward pass through a DataLoader. If InpaintNet weights are available, it runs a second pass to fill occluded shuttle positions. The result is written as a CSV with columns `Frame, X, Y, Visibility`.

5. After all workers finish, **`shuttle_extractor.py`** converts the raw CSVs to normalised `.npy` arrays (`shuttle_csvs_to_npy()`), producing `(t, 3)` arrays of `[x_norm, y_norm, visibility]` per clip.

### Design justifications

**Subprocess per worker (not Python multiprocessing):**
Each worker loads its own CUDA context and model copy. Subprocesses avoid GIL contention and CUDA fork-safety issues that plague `multiprocessing` with GPU workloads.

**`num_workers=0` in DataLoader:**
Frames are already a single numpy array in RAM (loaded by `generate_frames()`). DataLoader workers would just pickle-copy array slices across processes via IPC. For short clips (~75-105 frames) this adds overhead with no throughput gain.

**Pre-resize in `generate_frames()`:**
In temporal ensemble mode (the default `eval_mode='weight'`), the Dataset uses `sliding_step=1`: for a clip with N frames and `seq_len=L`, it produces `N - L + 1` overlapping windows of L frames each. Every frame appears in up to L different windows. Without pre-resize, each `__getitem__()` call resizes all L frames in its window from native resolution (e.g. 1280x720) down to 512x288 via PIL — so a single frame gets resized up to L times across different windows. For a typical 90-frame clip with `seq_len=3`, that's `(90 - 3 + 1) * 3 = 264` resize operations. With pre-resize, each of the 90 frames is resized once during `generate_frames()`, and the Dataset's resize calls become no-ops (source already matches target size). This also reduces the in-memory size of the frame array (e.g. 1280x720x3 per frame ~2.6 MB vs 512x288x3 ~0.4 MB), which matters when processing thousands of clips in sequence. Output is bit-identical (same PIL BICUBIC interpolation, same dimensions).

**`gc.collect()` + `torch.cuda.empty_cache()` between clips:**
Individual short clips produce small tensors, but over 33k iterations unreferenced GPU memory fragments accumulate. Periodic cleanup prevents gradual OOM drift.

**Two-level resume logic:**
- `batch_predict.py` checks for existing output CSVs before processing each clip (crash recovery within a single worker run).
- `shuttle_extractor.py` filters out clips with existing CSVs when building the pending list (crash recovery across separate pipeline invocations).

Both levels are needed: the first handles mid-run crashes, the second handles re-running the pipeline after partial completion.

**`--workers 1` on V100 16GB:**
Each model copy (TrackNet + InpaintNet) consumes ~4-5 GB. Two copies exceed the 16 GB VRAM budget. On A100 40GB or multi-GPU nodes, `--workers 2` roughly halves wall time.

### Where the efficiency comes from

The original `predict.py` was designed for one-off single-video inference. Running it as a subprocess per clip for ~33k clips introduces three bottlenecks that the batch pipeline eliminates:

1. **Model loading (~8s per clip):** Each subprocess call loads TrackNet + InpaintNet weights from disk, rebuilds the model, and transfers it to the GPU. For 33k clips this alone would take ~73 hours. `batch_predict.py` loads once and reuses across all clips, reducing this to ~8 seconds total.

2. **Redundant frame resizing:** The original path loads frames at native resolution (e.g. 1280x720), then the Dataset's `__getitem__()` resizes each frame to 512x288 every time it appears in a sliding window. In temporal ensemble mode with `sliding_step=1`, each frame is resized up to `seq_len` times across overlapping windows. Pre-resizing once in `generate_frames()` eliminates all redundant resize work and reduces the in-memory frame array by ~6x (for 720p input).

3. **Process startup overhead:** Spawning a Python subprocess, importing PyTorch, and initialising CUDA adds several seconds per invocation on top of model loading. The batch loop avoids this entirely — one process, one CUDA context, many clips.

---

## Accuracy guarantees

No accuracy-affecting changes were made. Specifically:

- **`predict()` function**: unchanged — same heatmap-to-coordinate and inpainted-coordinate-to-coordinate logic
- **TrackNet / InpaintNet forward passes**: unchanged — same model architecture, same weights, same `torch.no_grad()` inference
- **Temporal ensemble**: unchanged — same `get_ensemble_weight()`, same buffer management, same incomplete-buffer and last-batch handling
- **Coordinate scaling** (`img_scaler`): unchanged — same `w/WIDTH`, `h/HEIGHT` computation
- **Heatmap thresholding** (> 0.5) and contour detection (`cv2.findContours`): unchanged
- **InpaintNet masking and coordinate thresholding** (`COOR_TH`): unchanged
- **Frame pre-resize**: bit-identical to the Dataset's own resize (same PIL BICUBIC, same target dimensions, same source data)
