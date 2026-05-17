# BRIC — Badminton RGB Inference Classifier

Alternative stroke classifier to BST. R(2+1)D-18 over RGB player crops,
fused with shuttle-trajectory features and player coordinates.

## Layout

```
src/bric/
├── network.py                # R(2+1)D-18 + fusion head
├── dataset.py                # ShuttleSetDataset, reads NPZ caches
├── train.py                  # CLI training entry point
├── infer.py                  # Inference handler (planned, PR 4)
├── eval.py                   # Test-set evaluation (planned, PR 3)
├── smoke_test.py             # `uv run python -m bric.smoke_test`
├── perception/               # BRIC's perception stack
│   ├── players.py            # YOLO11 + ByteTrack
│   ├── shuttle.py            # TrackNetV3 wrapper
│   └── _vendor/tracknetv3/   # Vendored upstream
├── preprocessing/            # Cache producers (run once per source video)
│   ├── slice_rallies.py      # Rally-level mp4 extraction
│   ├── preprocess_videos.py  # YOLO+ByteTrack → players cache + RGB tensors
│   └── extract_shuttle.py    # TrackNetV3 → shuttle cache
└── diagnostics/              # Cache + data quality checks
    ├── evaluate_players.py   # Player cache visibility / outliers
    ├── evaluate_shuttle.py   # Shuttle cache visibility / verdict
    ├── validate_court_positions.py
    ├── validate_rgb.py       # Contact-sheet JPGs from RGB cache
    └── debug_court_bias.py   # Per-stroke court coord diagnostics
```

Generic utilities BRIC consumes live in `src/shared/`: taxonomy,
court geometry, player mapping, video I/O, frame-window helpers. A
per-stroke clip slicer (`shared/slicer.py`) is planned for PR 4 as
the cross-arch primitive for the live-upload inference path.

## Import rules

- `bric.*` modules **never** `import bst_refactor.*`.
- They import generic utilities (taxonomy, court, video_io, temporal)
  from `shared.*`.
- They import their own perception layer from `bric.perception.*`.

## What's trained by us

**BRIC (R(2+1)D-18 + fusion head)** — fine-tuned on ShuttleSet
single-stroke clips with ground-truth stroke timing. Everything else
(TrackNetV3, YOLO11) is pretrained and frozen. Heuristic stroke
localisation (SRA / swing detector) is deferred to v2.
