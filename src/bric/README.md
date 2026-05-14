# BRIC — Badminton RGB Inference Classifier

Alternative stroke classifier to BST. R(2+1)D-18 over RGB player crops,
fused with shuttle-trajectory features and player coordinates.

## Layout

```
src/
├── bst_refactor/           # Model A (pose extraction) code
├── api/                    # Inference serving API code
├── shared/                 # Shared configurations and utils
│   ├── taxonomy.py
│   └── court.py
├── perception/             # Model-agnostic perception infra
│   ├── video_io.py
│   ├── players.py          # Player detection and tracking (YOLO11)
│   ├── temporal.py         # Window utils + uniform-temporal subsample_indices
│   ├── shuttle.py          # Wraps vendored TrackNetV3
│   └── _vendor/tracknetv3/ # TrackNetv3 vendor code
└── bric/                   # Model B (R(2+1)D classifier) code
    ├── network.py
    ├── dataset.py
    ├── train.py
    ├── infer.py
    └── eval.py
```

## Import rules

- `perception.*` and `bric.*` **never** `import bst_refactor.*`.
- They import from `shared.*` for taxonomy / court utilities.
- They import TrackNetV3 from `perception._vendor.tracknetv3.*`.
- Cross-checking against BST (e.g. `scripts/run_bst_inference.py`)
  invokes BST as a subprocess, not via Python import.

## What's trained by us

Only **BRIC (R(2+1)D-18 + fusion head)** — fine-tuned on ShuttleSet
single-stroke clips with ground-truth stroke timing. Everything else
(TrackNetV3, YOLO11) is pretrained and frozen. Heuristic stroke
localisation (SRA / swing detector) is deferred to v2.
