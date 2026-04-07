# HPC Quickstart (Badminton Stroke Classifier)

This explains how to run this project on UNE HPC.

---

## 1. Connect to HPC

```bash
ssh <your-username>@turing
ssh -Y engelbart
```

---

## 2. Create project directories

```bash
mkdir -p /scratch/comp320a/badminton-stroke-classifier/{repo,data/raw,data/processed,data/checkpoints,data/logs}
```

---

## 3. Clone repository

```bash
cd /scratch/comp320a/badminton-stroke-classifier/repo
git clone <your-repo-url> .
```

---

## 4. Create Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Verify environment

```bash
pytest tests/
```

This ensures:

- dependencies correctly installed
- FastAPI imports work
- base environment is valid

---

## 6. Verify GPU access

```bash
nvidia-smi
```

Create a test file:

```python
# test_gpu.py
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Run:

```bash
python test_gpu.py
```

---

## 7. Run API (optional)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 8. Data locations

Use `/scratch` for all large files:

```text
/scratch/comp320a/badminton-stroke-classifier/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── checkpoints/
│   └── logs/
```

---

## 9. Notes

- Build environments on `engelbart`, not only on `turing`
- GPU nodes use a different OS (Rocky Linux)
- `/scratch` is fast but isn't backed up
- Avoid storing large data in your home directory
