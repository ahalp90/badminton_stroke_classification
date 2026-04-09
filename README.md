# Badminton Stroke Classifier

Badminton Stroke Classification using AI Computer Vision (Contribution to long-term Badminton Objective Player Grading Project).

## Project Structure

- `src/`: Core application code (data loading, models, training, API)
- `src/bst_refactor/`: Standalone data pipeline and refactored BST stroke classifier — has its own pinned environments
- `tests/`: Pytest test suite
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `configs/`: Hyperparameter and pipeline configurations
- `scripts/`: Utility scripts
- `scratch/`: Team notes and temporary files

## Local Setup Instructions

> **Environment setup (Docker, venv, HPC) is owned by Ethan** — see his documentation once available.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Setup

Run the base test suite to ensure all dependencies are installed and importing correctly:

```bash
pytest tests/
```

If the tests pass, your environment is ready to go!

### 3. Experiment Tracking (MLflow)

MLflow runs fully locally — no account or server setup needed. After installing dependencies:

```bash
# Run the example script to verify MLflow is working
python scripts/example_mlflow_run.py

# Open the experiment dashboard
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

When writing training code, log experiments like this:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"learning_rate": 0.001, "batch_size": 32})
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("val_accuracy", acc, step=epoch)
```

MLflow will auto-create an `mlruns/` directory to store results locally.

### 4. Run API & View in browser

From the root directory, run:

```bash
uvicorn src.api.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in the browser.

### 5. BST Stroke Classifier (`src/bst_refactor/`)

The BST subproject has its own tightly pinned dependencies (three separate venvs) that are **not** covered by the root `requirements.txt`. 
Do not add its packages globally — the MMPose stack requires numpy < 2.0, which conflicts with the main project.

See [`src/bst_refactor/data_pipeline_to_model_train.md`](src/bst_refactor/data_pipeline_to_model_train.md#quick-start-end-to-end-execution)
  for:
  - Three-venv setup (pipeline, MMPose, BST training)
  - Full execution order from video download through model training
  - Requirements files: `pipeline/requirements.txt`, `stroke_classification/preparing_data/requirements.txt`,
  `stroke_classification/requirements.txt`
  
(detailed pipeline-only README.md in the relevant subdir)

### 6. HPC Data Storage (engelbart)

Video data and generated datasets are too large for home directories (40GB quota). On engelbart, these directories should be symlinked to `/scratch` before running the pipeline.

**One-time setup:**

```bash
# Create shared data directories on scratch
mkdir -p /scratch/comp320a/ShuttleSet/raw_video
mkdir -p /scratch/comp320a/ShuttleSet/clips
mkdir -p /scratch/comp320a/ShuttleSet/shuttle_npy

# Symlink from your project into scratch
cd ~/badminton_stroke_classifier/src/bst_refactor/ShuttleSet
ln -s /scratch/comp320a/ShuttleSet/raw_video raw_video
ln -s /scratch/comp320a/ShuttleSet/clips clips
ln -s /scratch/comp320a/ShuttleSet/shuttle_npy shuttle_npy
```

Everyone shares the same `/scratch` data, so videos only need to be downloaded once. Make sure permissions are open after downloading:

```bash
chmod -R 775 /scratch/comp320a/ShuttleSet
```

**Important notes:**
- `/scratch` is **not backed up** and is **local to each HPC host** — data on engelbart's scratch is not visible from bourbaki.
- Do not store videos or clips in your home directory — they will exceed your quota.
- These symlinks are tracked in git. They will be broken on non-HPC machines — this is expected. The symlinks only need to work on engelbart.