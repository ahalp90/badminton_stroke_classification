# BRIC training environment setup

End-to-end checklist for setting up a development / training environment
for BRIC. Platform-agnostic — covers Apple Silicon (MPS), Linux + Nvidia
CUDA (x86_64 or ARM64), and CPU-only fallback.

Estimated time: ~10 minutes once the host is reachable and Python 3.11+
is available.

---

## 1. Prerequisites

- **Python 3.11 or newer** on PATH (`python3 --version`).
  `uv` will manage the project's actual Python version — system Python
  just needs to be recent enough to bootstrap.
- **Git**.
- **Recommended:** an Nvidia GPU with current drivers, OR an Apple
  Silicon Mac. Training on CPU is supported but slow.
- **For Linux + CUDA:** check the driver's CUDA version with `nvidia-smi`
  (top-right of the table). Make sure it matches the wheel index URL
  configured in `pyproject.toml` — see §4 below if it doesn't.

## 2. Install uv

`uv` is the project's Python package + venv manager
([docs](https://docs.astral.sh/uv/)).

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or ~/.zshrc; or restart shell
uv --version
```

**Windows (untested for this project):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 3. Clone the repo

```bash
git clone <REPO_URL> badminton_stroke_classification
cd badminton_stroke_classification
git checkout <branch>     # e.g. main, or a feature branch
```

## 4. Install BRIC dependencies

From the project root:

```bash
uv sync --extra bric
```

What this does:
- Reads `pyproject.toml`.
- Creates a `.venv/` if not present, using a Python interpreter that
  satisfies `requires-python = ">=3.11"`.
- Installs the **shared base** (always): torch, torchvision, numpy,
  pandas, scipy, opencv-python, Pillow, tqdm, parse, pycocotools,
  pyyaml, fastapi, uvicorn, python-multipart.
- Installs the **`bric` extras** on top: ultralytics, transformers,
  torcheval (combines `bric-runtime` + `bric-train`). R(2+1)D-18 ships
  in torchvision (a base dep) so no extra ML lib is needed beyond
  ultralytics.
- For `torch` / `torchvision` on **Linux**, pulls from the CUDA wheel
  index configured in `[tool.uv.sources]` (currently `cu128`). On
  **macOS**, falls back to the default PyPI wheels (MPS-capable on
  Apple Silicon).
- Writes `uv.lock` so the install is reproducible.

Other install patterns:

| Use case | Command |
|----------|---------|
| BRIC training | `uv sync --extra bric --extra dev` |
| BRIC inference | `uv sync --extra bric-runtime` |
| Unified API server (BRIC + BST-X handlers) | `uv sync --extra bric-runtime --extra bst-x-runtime` |
| Dev tooling only (lint / tests) | `uv sync --extra dev` |

### Adjusting the CUDA wheel version

The default `[[tool.uv.index]].url` in `pyproject.toml` is
`https://download.pytorch.org/whl/cu128`. If your driver reports a
different CUDA version, you may need to switch:

| Driver CUDA | Try this index URL |
|-------------|-------------------|
| 12.1        | `https://download.pytorch.org/whl/cu121` |
| 12.4        | `https://download.pytorch.org/whl/cu124` |
| 12.6        | `https://download.pytorch.org/whl/cu126` |
| 12.8        | `https://download.pytorch.org/whl/cu128` |
| 13.x        | `https://download.pytorch.org/whl/cu128` (forward-compat) or PyTorch nightly |

Newer Nvidia drivers are backward-compatible with older CUDA runtimes,
so picking a slightly older index is generally safe — you may just miss
the latest GPU-specific optimisations.

### Common install failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No matching distribution found for torch` | Wrong CUDA wheel for your driver | Edit `[[tool.uv.index]].url` in `pyproject.toml` per the table above |
| `nvidia-smi: command not found` | CUDA driver not installed | Install via your distro's package manager or Nvidia's installer |
| `Killed` during install | Out of memory during wheel build | Add swap, or `uv sync --extra bric --no-build-isolation` |
| pycocotools wheel build fails | C compiler / headers missing | Install `build-essential` (Linux) or Xcode CLI tools (macOS); or remove pycocotools and patch the TrackNetV3 import that uses it |

## 5. Run the BRIC smoke test

```bash
uv run python -m bric.smoke_test
```

This script checks:

1. Platform basics
2. PyTorch installed and accelerator (CUDA / MPS / CPU) visible
3. Tensor matmul on the accelerator
4. YOLO11n forward pass on the accelerator
5. R(2+1)D-18 forward pass on the accelerator (Kinetics-400 pretrained,
   `torchvision.models.video`)
6. OpenCV importable
7. BRIC's own modules (`shared.taxonomy`, `shared.court`,
   `shared.video_io`) import cleanly

Expected output ends with:

```
All checks passed. Environment is ready for BRIC training.
```

If any check fails, the script exits with a Python traceback. Re-run
the smoke test any time you want to verify the env is still healthy
(after a system update, after pulling code, etc.).

---

## 6. Common commands

```bash
# Pull latest
git pull

# Re-verify env after pulling (cheap)
uv run python -m bric.smoke_test

# Run training
uv run python -m bric.train --overfit 32 --epochs 30   # sanity check
uv run python -m bric.train --epochs 50                 # full run

# Inference
uv run python -m bric.infer <video_path>

# Tests
uv run pytest tests/
```

Anything started with `uv run` activates the project venv automatically;
no need for `source .venv/bin/activate` first.

---

## Hardware-specific notes

### Apple Silicon (M1/M2/M3 Pro/Max)

- PyTorch uses the MPS backend; CUDA-only ops fall back to CPU silently.
  Most BRIC ops work, but full training is slower than on a discrete GPU
  — fine for development and small experiments.
- Some video / vision libraries have MPS rough edges; if you hit one,
  move that workload to a CUDA host.

### Linux + Nvidia x86_64

- Most well-trodden path. PyPI's default `torch` wheels on Linux
  include CUDA, but we override that via `[tool.uv.sources]` to use the
  PyTorch CUDA index for explicit version control.

### Linux + Nvidia ARM64 (Grace Hopper / Grace Blackwell)

- Newer ecosystem, fewer prebuilt third-party wheels. Most things in
  the BRIC dep set have ARM64 wheels; if one doesn't, `uv` will try to
  build from source and may need `build-essential` or equivalent.
- The PyTorch CUDA index (cu126/cu128) does ship `linux_aarch64` wheels;
  this is the path the `[tool.uv.sources]` block targets.

### CPU-only

- Training R(2+1)D-18 on CPU is impractical (days per epoch). Use this
  only for module imports, dataset iteration, and unit tests. For real
  training, get GPU access.

---

## Troubleshooting log

Append your own gotchas as you encounter them. Future maintainers will
thank you.
