# Badminton Stroke Classifier

Badminton Stroke Classification using AI Computer Vision (Contribution to long-term Badminton Objective Player Grading Project).

## Project Structure

- `src/`: Core application code (data loading, models, training, API)
- `tests/`: Pytest test suite
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `configs/`: Hyperparameter and pipeline configurations
- `scripts/`: Utility scripts
- `scratch/`: Team notes and temporary files
- `data/`: Local dataset storage (`raw/`, `processed/`, `checkpoints/`, `logs/`)

---

## Local Setup Instructions

This project uses Docker to ensure a consistent environment across the team.

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

## Experiment Tracking (MLflow)

Inside the container:

```bash
python scripts/example_mlflow_run.py
mlflow ui
```

Open:
http://127.0.0.1:5000

MLflow will automatically create an `mlruns/` directory.

---

## Verify Environment

The project includes a base environment test:

- `tests/test_environment.py`

Optional manual checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import mediapipe as mp; print('mediapipe ok')"
python -c "import fastapi; print('fastapi ok')"
python -c "import mlflow; print('mlflow ok')"
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

---

## Notes

- Docker is the --only supported local development environment--
- HPC is used for GPU training workloads
- Keep large files out of the repository

