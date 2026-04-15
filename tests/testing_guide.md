# Testing Guide

## Quick start

From the project root:

```bash
pytest
```

This runs all tests except the HPC integration test, which auto-skips when `BST_DATA_DIR` is not set.

## Test files

### `test_environment.py`
**Environment sanity check.** Imports core dependencies (torch, torchvision, numpy, pandas, matplotlib, sklearn, mediapipe) and fails if any are missing. Useful after setting up a new venv.

- **Prerequisites:** Project dependencies installed (`pip install -r requirements.txt`)

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
3. Flatten pose tensor (mirrors `bst_train.py:101`)
4. Run `BST_0` forward pass
5. Verify output shape is `(batch_size, n_classes)`

- **Prerequisites:** Preprocessed npy dataset (output of `prepare_train_on_shuttleset.py`)

To run, point `BST_DATA_DIR` at the `dataset_npy_collated` directory (should contain `train/`, `val/`, `test/` subdirectories):

```bash
BST_DATA_DIR=/path/to/dataset_npy_collated pytest tests/test_integration.py -v
```

Without `BST_DATA_DIR` set, this test auto-skips.

**Note:** This test validates against `BST_0`, the baseline and parent class for BST-origin architectures. It covers the shared data pipeline (pose, shuttle, position npy files) but will need to evolve as Arch 1 and Arch 2 mature — Arch 1 will additionally ingest 3D CNN latent representations, and Arch 2 will have its own 3D CNN latents, TrackNet npy data, and potentially other input streams.

## CI

GitHub Actions runs `pytest` on every push and PR (`.github/workflows/ci.yml`). The integration test auto-skips in CI since `BST_DATA_DIR` is not set.

## conftest.py

The root `conftest.py` adds two entries to `sys.path` so that imports used inside `bst_refactor` work from the test directory:

- `src/bst_refactor` — allows `from pipeline.config import ...`
- `src/bst_refactor/stroke_classification` — allows `from model.tempose import ...`
