#!/usr/bin/env python3
"""BRIC environment smoke test.

Verifies the training environment is ready: ML stack works on the available
accelerator AND BRIC's own modules import cleanly.

Run from the project root after `uv sync --extra bric`:
    uv run python scripts/bric_smoke_test.py

Or from an activated venv:
    python scripts/bric_smoke_test.py

What it checks:
  1. Platform basics
  2. PyTorch installed and accelerator (CUDA / MPS / CPU) visible
  3. Tensor matmul on the accelerator
  4. YOLO11n forward pass on the accelerator
  5. R(2+1)D-18 forward pass on the accelerator (Kinetics-400 pretrained)
  6. OpenCV importable
  7. BRIC's own modules import cleanly (shared.taxonomy, shared.court,
     perception.video_io)

Exits 0 on full success, 1 on first failure with a clear diagnostic.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from pathlib import Path

# Project root, derived once. Any cache / artefact paths the smoke test
# touches resolve from here so the script is invariant to cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ANSI colours for terminal output. No-op if NO_COLOR or non-tty.
def _supports_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get('NO_COLOR')

GREEN = '\033[32m' if _supports_color() else ''
RED = '\033[31m' if _supports_color() else ''
YELLOW = '\033[33m' if _supports_color() else ''
DIM = '\033[2m' if _supports_color() else ''
RESET = '\033[0m' if _supports_color() else ''


def banner(msg: str) -> None:
    print(f'\n{YELLOW}=== {msg} ==={RESET}')


def ok(msg: str) -> None:
    print(f'{GREEN}[PASS]{RESET} {msg}')


def fail(msg: str, exc: Exception | None = None) -> None:
    print(f'{RED}[FAIL]{RESET} {msg}')
    if exc is not None:
        print(f'{DIM}{traceback.format_exc()}{RESET}')
    sys.exit(1)


def info(msg: str) -> None:
    print(f'{DIM}  {msg}{RESET}')


# ---------------------------------------------------------------------------
# 1. Platform basics
# ---------------------------------------------------------------------------
banner('1. Platform')
print(f'  python:   {sys.version.split()[0]}')
print(f'  platform: {platform.system()} {platform.machine()}')
print(f'  release:  {platform.release()}')


# ---------------------------------------------------------------------------
# 2. PyTorch + accelerator
# ---------------------------------------------------------------------------
banner('2. PyTorch + accelerator')
try:
    import torch
    info(f'torch: {torch.__version__}')
except ImportError as e:
    fail('torch not installed. Run: uv sync --extra bric', e)

# Pick the best available accelerator.
device: str
if torch.cuda.is_available():
    device = 'cuda'
    info(f'CUDA: {torch.version.cuda}, devices: {torch.cuda.device_count()}')
    info(f'GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    info('MPS (Apple Silicon) available')
else:
    device = 'cpu'
    info(f'{YELLOW}No GPU/accelerator detected; falling back to CPU.{RESET}')
    info('On the GB10 you should see CUDA. If you see this, debug before training.')

ok(f'PyTorch can use device: {device}')


# ---------------------------------------------------------------------------
# 3. Simple tensor op on accelerator
# ---------------------------------------------------------------------------
banner('3. Tensor op on accelerator')
try:
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = a @ b
    if device != 'cpu':
        torch.cuda.synchronize() if device == 'cuda' else None
    ok(f'matmul on {device}: shape={tuple(c.shape)}, dtype={c.dtype}')
except Exception as e:
    fail(f'Matmul on {device} failed', e)


# ---------------------------------------------------------------------------
# 4. YOLO11 forward pass (Ultralytics)
# ---------------------------------------------------------------------------
banner('4. YOLO11 forward pass')
try:
    from ultralytics import YOLO
    info('ultralytics imported')
except ImportError as e:
    fail('ultralytics not installed. Run: uv sync --extra bric', e)

try:
    # Resolve weights to the canonical project cache (gitignored).
    # ultralytics downloads the file on first use if not present at this path.
    yolo_weights = PROJECT_ROOT / 'runtime' / 'checkpoints' / 'yolo11' / 'yolo11n.pt'
    yolo_weights.parent.mkdir(parents=True, exist_ok=True)
    info(f'YOLO weights path: {yolo_weights}')
    model = YOLO(str(yolo_weights))
    # Synthetic image — RGB ndarray, 640x640.
    import numpy as np
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model.predict(img, verbose=False, device=device if device != 'mps' else 'cpu')
    n_det = len(results[0].boxes) if results[0].boxes is not None else 0
    ok(f'YOLO11n forward pass on {device}: {n_det} detections (synthetic image, expect 0)')
except Exception as e:
    fail('YOLO11n forward pass failed', e)


# ---------------------------------------------------------------------------
# 5. R(2+1)D-18 forward pass (torchvision)
# ---------------------------------------------------------------------------
banner('5. R(2+1)D-18 forward pass')
try:
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    info('torchvision.models.video imported')
except ImportError as e:
    fail('torchvision not installed. Run: uv sync --extra bric', e)

try:
    # R(2+1)D-18 takes a single (B, C=3, T, H, W) tensor. Pretrained
    # Kinetics-400 weights are trained on 16 frames at 112x112; finetuning
    # on different (T, H, W) works because the temporal conv adapts to T
    # and the spatial convs adapt to H/W. BRIC's dataset will sample
    # n=32 over a ~2s window at 224x224 — smoke test exercises that shape.
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1).to(device).eval()
    x = torch.randn(1, 3, 32, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)
    ok(f'R(2+1)D-18 forward pass on {device}: output shape={tuple(y.shape)}')
except Exception as e:
    fail('R(2+1)D-18 forward pass failed', e)


# ---------------------------------------------------------------------------
# 6. OpenCV import (used by perception.video_io)
# ---------------------------------------------------------------------------
banner('6. OpenCV')
try:
    import cv2
    info(f'cv2 version: {cv2.__version__}')
    ok('OpenCV importable')
except ImportError as e:
    fail('opencv-python not installed. Run: uv sync --extra bric', e)


# ---------------------------------------------------------------------------
# 7. BRIC modules import cleanly
# ---------------------------------------------------------------------------
banner('7. BRIC modules')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from shared import taxonomy
    n_classes = len(taxonomy.TAXONOMY_UNE_MERGE_V1_NOSIDES.class_list())
    info(f'shared.taxonomy: {n_classes} classes ({taxonomy.DEFAULT_TAXONOMY})')
except Exception as e:
    fail('shared.taxonomy failed to import', e)

try:
    from shared import court
    info(f'shared.court: REF_COURT_M = {court.REF_COURT_M}')
except Exception as e:
    fail('shared.court failed to import', e)

try:
    from perception import video_io
    info(f'perception.video_io: VideoInfo dataclass present = {hasattr(video_io, "VideoInfo")}')
except Exception as e:
    fail('perception.video_io failed to import', e)

ok('BRIC modules importable')


# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
print(f'\n{GREEN}All checks passed.{RESET} Environment is ready for BRIC training.')
