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
- `frontend/`: React app user interface
- `data/`: Local dataset storage (`raw/`, `processed/`, `checkpoints/`, `logs/`)
- `docs/`: Project documentation including decision log
---

## Local Setup Instructions

The project runs in Docker. See `src/bst_refactor/` subproject READMEs for the separate HPC training environments.

### 1. Build and run

```bash
docker compose up --build
```

Backend API: http://localhost:24082/docs
Frontend: http://localhost:5173

### 2. Enter the backend container

```bash
docker exec -it badminton-backend bash
```

### 3. Verify setup

```bash
pytest tests/
```

If the tests pass, your environment is ready to go.

---

## Experiment Tracking

BST training runs log through `src/bst_refactor/run_tracker.py`. Each run writes a manifest, per-serial metrics, and TensorBoard events under `src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/<run_id>/`.

Optional Aim UI (from `main_on_shuttleset/`): `python ../../aim_backfill.py` (one-shot, idempotent), then `aim up`. Details in [`src/bst_refactor/run_tracker.md`](src/bst_refactor/run_tracker.md).

There's also a partial MLflow setup in `scripts/example_mlflow_run.py` if someone wants to plug in, but it's probably more than this project needs; the manifest tracker above integrates with Aim at near-zero effort. The MLflow stub will be deleted before delivery if Scott has not picked it up by then.

---

## Current data state and weight compatibility

Active collated data on engelbart and bourbaki:

```
/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop/
```

`ablation_id=wipe_drop` corresponds to the shuttle-unzeroing-on-keypoint-fail change from the `shuttle/wipe-drop` branch. The script that used to wipe shuttle xy to (0, 0) on frames where MMPose dropped a player no longer fires (~14k frames, 0.84% of the extract). Pose path is unchanged. Rationale in [`scratch/architecture_notes/frame_zeroing.md`](scratch/architecture_notes/frame_zeroing.md). Comparison run lives at `experiments/run_20260503_172922/`.

A variant 2a shuttle_missing channel was tested on top of this and didn't lift; the design + diff is archived in [`scratch/architecture_notes/shuttle_mask_archive.md`](scratch/architecture_notes/shuttle_mask_archive.md), code not in main.

### Weight compatibility with prior runs

The shuttle-unzeroing change is a data change only; model architecture is unchanged. So:

- `load_state_dict` succeeds on any pre-shuttle-unzeroing weight file (e.g. `run_20260501_164658`).
- But re-testing those weights on the new collation is a small distribution shift: ~14k frames that were wiped to (0, 0) at training time now carry their TrackNet shuttle xy. Test metrics will move slightly; not apples-to-apples with the original run.
- Forward-going, train fresh on the new collation. Old weights tested on the old collation remain the canonical numbers for those runs.

The dropped shuttle-mask branch added new state_dict keys (`mask_proj`, `shuttle_fuse`); its weights are not loadable into the active code.

---

## Verify Environment

The project's pytest suite covers environment, data access, dataset, API, sticky_anchor heuristic invariants, and an integration smoke (auto-skipped without `BST_DATA_DIR`):

- `tests/test_environment.py`
- `tests/test_data_access.py`
- `tests/test_dataset.py`
- `tests/test_api.py`
- `tests/test_sticky_anchor.py`
- `tests/test_integration.py`

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

Lists clips for a given `split` + `class` filter, paired with their shuttle and mmpose files. Reads from `notebooks/clips_master.csv` under the active taxonomy (default `une_merge_v1`). Three modes:

```bash
# Run from the repo root. Set PYTHONPATH once for the session, or prepend
# PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification to each
# command instead.
export PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification

# 1. Counts: how many clips per split/class, how many on disk, how many
#    have shuttle/mmpose files. Quick health check on what's available.
python -m pipeline.data_access --summary

# 2. List of paths: one tab-separated row per matching clip
#    (split, class, clip_stem, clip_path, shuttle_path, mmpose_path).
#    Save it (>file.tsv) or pipe it to another script that needs the paths.
python -m pipeline.data_access --split val --class Top_smash

# 3. Walk-through (no flags): six numbered prompts ask for split column,
#    taxonomy, split, class, drop-unknown, and summary-vs-paths in turn.
#    Useful when you don't remember the flag names.
python -m pipeline.data_access
```

Paths differ between machines (local vs engelbart); keep them in a local/remote `.env` file. Copy [`.env.example`](.env.example) to `.env` and fill in your paths; the `.env` is gitignored so each person's paths stay local.

Full CLI flag list and Python API: [`src/bst_refactor/pipeline/README.md`](src/bst_refactor/pipeline/README.md#higher-level-access-pipelinedata_accesspy).

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
cd ~/badminton_stroke_classification/src/bst_refactor/ShuttleSet
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
cd ~/badminton_stroke_classification/src/bst_refactor/stroke_classification/preparing_data
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
