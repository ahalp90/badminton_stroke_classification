# Badminton Stroke Classifier

Badminton Stroke Classification using AI Computer Vision (Contribution to long-term Badminton Objective Player Grading Project).

## Project Structure

- `src/`: Core application code (data loading, models, training, API)
- `src/bst_refactor/`: Standalone data pipeline and refactored BST stroke classifier; has its own pinned environments
- `tests/`: Pytest test suite
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `configs/`: Hyperparameter and pipeline configurations
- `scripts/`: Utility scripts
- `scratch/`: Team notes and temporary files
- `data/`: Local dataset storage (`raw/`, `processed/`, `checkpoints/`, `logs/`)

---

## Local Setup Instructions

Dev currently runs in Python venvs; see subproject READMEs for the pinned environments. The Docker setup below is a work-in-progress target for deployment.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build and run

```bash
docker compose up --build
```

### 3. Enter container

```bash
docker exec -it badminton-dev bash
```

### 4. Verify setup

```bash
pytest tests/
```

If the tests pass, your environment is ready to go.

---

## Run API & view in browser

Inside the container:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
http://127.0.0.1:8000/docs

---

## Experiment Tracking

BST training runs log through `src/bst_refactor/run_tracker.py`. Each run writes a manifest, per-serial metrics, and TensorBoard events under `src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/<run_id>/`.

Optional Aim UI (from `main_on_shuttleset/`): `python ../../aim_backfill.py` (one-shot, idempotent), then `aim up`. Details in [`src/bst_refactor/run_tracker.md`](src/bst_refactor/run_tracker.md).

There's also a partial MLflow setup in `scripts/example_mlflow_run.py` if someone wants to plug in, but it's probably more than this project needs; the manifest tracker above integrates with Aim at near-zero effort.

---

## Verify Environment

The project includes a base environment test:

- `tests/test_environment.py`

Optional manual checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import fastapi; print('fastapi ok')"
```

---

## Data Directory

```text
data/
├── raw/
├── processed/
├── checkpoints/
└── logs/
```

Create it with:

```bash
bash scripts/setup_data.sh
```

---

## UNE HPC Setup

- Project guide: `scratch/hpc_quickstart.md`
- GPU notes: `scratch/gpu-access.md`

Notes:

- Use GPU hosts (e.g. `engelbart`) for training
- Build environments on the GPU host (not just `turing`)
- Store data in `/scratch`, not your home directory
- Run long training jobs inside `tmux` so they survive SSH drops

---

## Notes

- HPC is used for GPU training workloads
- Keep large files out of the repository

## BST Stroke Classifier (`src/bst_refactor/`)

The BST subproject has its own tightly pinned dependencies (three separate venvs) that are **not** covered by the root `requirements.txt`. 
Do not add its packages globally — the MMPose stack requires numpy < 2.0, which conflicts with the main project.

See [`src/bst_refactor/data_pipeline_to_model_train.md`](src/bst_refactor/data_pipeline_to_model_train.md#quick-start-end-to-end-execution)
  for:
  - Three-venv setup (pipeline, MMPose, BST training)
  - Full execution order from video download through model training
  - Requirements files: `pipeline/requirements.txt`, `stroke_classification/preparing_data/requirements.txt`,
  `stroke_classification/requirements.txt`
  
(detailed pipeline-only README.md in the relevant subdir)

### Inspecting available clips (`pipeline.data_access`)

One-shot CLI + Python API for "give me all clips for `split=X` and class=`Y`, paired with their shuttle + mmpose files", driven by `notebooks/clips_master.csv` under the active taxonomy:

```bash
# Per-split / per-class counts across clips, shuttle, and mmpose
PYTHONPATH=src/bst_refactor python -m pipeline.data_access --summary

# TSV of clip / shuttle / mmpose paths for a filter
PYTHONPATH=src/bst_refactor python -m pipeline.data_access --split val --class Top_smash

# Interactive TUI (no flags)
PYTHONPATH=src/bst_refactor python -m pipeline.data_access
```

Machine-specific paths go in a local `.env` (copy from [`.env.example`](.env.example); gitignored). Full API + CLI reference: [`src/bst_refactor/pipeline/README.md`](src/bst_refactor/pipeline/README.md#higher-level-access-pipelinedata_accesspy).

## HPC Data Storage (engelbart)

Video data and generated datasets are too large for home directories (40GB quota). On engelbart, these directories should be symlinked to `/scratch` before running the pipeline.

**One-time setup (pipeline data):**

```bash
# Create shared data directories on scratch
mkdir -p /scratch/comp320a/ShuttleSet/raw_video
mkdir -p /scratch/comp320a/ShuttleSet/clips
mkdir -p /scratch/comp320a/ShuttleSet/shuttle_csv
mkdir -p /scratch/comp320a/ShuttleSet/shuttle_npy

# Symlink from your project into scratch
cd ~/badminton_stroke_classifier/src/bst_refactor/ShuttleSet
ln -s /scratch/comp320a/ShuttleSet/raw_video raw_video
ln -s /scratch/comp320a/ShuttleSet/clips clips
ln -s /scratch/comp320a/ShuttleSet/shuttle_csv shuttle_csv
ln -s /scratch/comp320a/ShuttleSet/shuttle_npy shuttle_npy
```

**One-time setup (pose estimation output):**

MMPose saves per-clip `.npy` files under a taxonomy-specific directory (`ShuttleSet_data_{taxonomy}/`). The script auto-creates this directory and all subdirectories, but on engelbart you want the data on scratch, so symlink first:

```bash
# Create the taxonomy output dir on scratch (replace taxonomy name as needed)
mkdir -p /scratch/comp320a/ShuttleSet_data_une_merge_v1

# Symlink into the preparing_data dir where the script expects it
cd ~/badminton_stroke_classifier/src/bst_refactor/stroke_classification/preparing_data
ln -s /scratch/comp320a/ShuttleSet_data_une_merge_v1 ShuttleSet_data_une_merge_v1
```

**Note on taxonomy and pose data:** Pose data is physically taxonomy-independent -- the same clip produces byte-identical keypoints regardless of which taxonomy it's organized under. Clip filenames (`{vid}_{set}_{rally}_{ball}`) are physical identifiers, so pose results from one taxonomy can in principle be reused by another via filename matching. The taxonomy folder only determines which stroke-type subdirectories the `.npy` files land in.

Everyone shares the same `/scratch` data, so videos only need to be downloaded once. Make sure permissions are open after downloading:

```bash
chmod -R 775 /scratch/comp320a/ShuttleSet
```

**Important notes:**
- `/scratch` is **not backed up** and is **local to each HPC host** — data on engelbart's scratch is not visible from bourbaki.
- Do not store videos or clips in your home directory — they will exceed your quota.
- These symlinks are tracked in git. They will be broken on non-HPC machines — this is expected. The symlinks only need to work on engelbart.
