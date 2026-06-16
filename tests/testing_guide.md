# Testing Guide

## Quick start

From the project root:

```bash
pytest
```

This runs all tests except the HPC integration test, which auto-skips when `BST_X_DATA_DIR` is not set.

## Test files

### `test_environment.py`
**Environment sanity check.** Imports core dependencies (torch, torchvision, numpy, pandas, matplotlib, sklearn) and fails if any are missing. Useful after setting up a new venv.

- **Prerequisites:** Project dependencies installed (`pip install -r requirements.txt`)

### `test_data_access.py`
**`pipeline.data_access` filtering tests.** Builds a synthetic `clips_master.csv` plus a fake clips/shuttle/mmpose tree (matching post-Phase-2 layout: nested clips, flat npy) and verifies CSV-driven `get_clip_records`, `_derive_class_label`, `summarise`, and the interactive menu helpers behave correctly across taxonomies and splits.

- **Prerequisites:** Project dependencies

### `test_sticky_anchor.py`
**Sticky_anchor heuristic invariant tests.** Seven pinning tests for the per-slot Voronoi + EMA tracker (`src/bst_x/stroke_classification/preparing_data/heuristics/sticky_anchor.py`). Synthetic-only — uses an identity-homography court at 1280x720 so picking and EMA-reset behaviour can be verified deterministically. The X3D-S wrist-crop layer will consume the same per-slot pose stream, so these invariants are pinned before that work lands.

- **Prerequisites:** Project dependencies

### `test_dataset.py`
**DataLoader batch shape validation.** Creates synthetic npy data matching the real dataset format (4 clips, 100 frames, 2 players, 17 joints) and verifies that `Dataset_npy_collated` and PyTorch `DataLoader` produce tensors with the expected shapes.

- **Prerequisites:** Project dependencies

### `test_api.py`
**FastAPI endpoint smoke tests.** Tests the `/api/upload`, `/api/status`, `/api/results`, and `/api/models` endpoints using FastAPI's `TestClient`.

- **Prerequisites:** Project dependencies

### `test_integration.py`
**End-to-end downstream pipeline test.** Validates the full path from real preprocessed npy files through to a BST_0 forward pass:

1. Load real npy files via `Dataset_npy_collated`
2. Batch via `DataLoader`
3. Flatten pose tensor (mirrors `bst_x_train.py:101`)
4. Run `BST_0` forward pass
5. Verify output shape is `(batch_size, n_classes)`

- **Prerequisites:** Preprocessed npy dataset (output of `prepare_train_on_shuttleset.py`)

To run, point `BST_X_DATA_DIR` at a collated `npy_[3d_][seq{N}_]{split}_{collation_id}` directory (should contain `train/`, `val/`, `test/` subdirectories). Prefix tags (`3d_`, `seq{N}_`) appear only for non-default configs. Split is folded into the name; `collation_id` is the generation tag, so re-collations of the same taxonomy + split coexist:

```bash
BST_X_DATA_DIR=/scratch/.../npy_v2_taxon_pinned_w_preds \
    pytest tests/test_integration.py -v
```

Historical note: pre-2026-04-21 collated dirs used a longer prefix (`dataset_npy_collated_between_2_hits_with_max_limits_seq_100_..._{ablation_id}`). V3 and V4 on engelbart still live under the old name; everything going forward uses the shorter `npy_...` form.

Without `BST_X_DATA_DIR` set, this test auto-skips.

**Note:** This test validates against `BST_0`, the baseline and parent class for BST-origin architectures. It covers the shared data pipeline (pose, shuttle, position npy files) but will need to evolve as Arch 1 and Arch 2 mature — Arch 1 will additionally ingest 3D CNN latent representations, and Arch 2 will have its own 3D CNN latents, TrackNet npy data, and potentially other input streams.

## CI

GitHub Actions runs `pytest` on every push and PR (`.github/workflows/ci.yml`). The integration test auto-skips in CI since `BST_X_DATA_DIR` is not set.

## conftest.py

The root `conftest.py` adds two entries to `sys.path` so that imports used inside `bst_x` work from the test directory:

- `src/bst_x` — allows `from pipeline.config import ...`, `from run_tracker import ...`
- `src/bst_x/stroke_classification` — allows `from preparing_data.shuttleset_dataset import ...`, `from main_on_shuttleset.bst_x_common import ...`, `from model.tempose import ...`

The same pair is the documented PYTHONPATH for non-test invocation post-step-P (`PYTHONPATH=src/bst_x:src/bst_x/stroke_classification python -m main_on_shuttleset.bst_x_train`), so tests and production share one resolution layout.
