# `training/`

Source data, derived caches, and per-architecture experiment records.
Required to train a model; not required by a serving host.

```
training/
├── data/                       source training data (shared across architectures)
└── <arch>/
    ├── cache/                  re-derivable preprocessing per architecture
    └── experiments/<run_id>/   training run record + weights
```

## `data/`

Source training data. The ShuttleSet release lives under
`data/shuttleset/`:

```
training/data/shuttleset/
├── annotations/    upstream CSVs (per-match, video metadata, flaw records)
├── raw_video/      source mp4s 
├── rally_clips/    per-rally mp4s sliced from raw_video for shuttle extraction
└── clips/          per-stroke mp4s 
```

Annotations are checked in at the directory level (`.gitkeep`); video
files are gitignored and rsync'd onto the training host. See
`scripts/slice_rallies.py` for how `rally_clips/` is produced.

## `<arch>/cache/`

Per-architecture preprocessing caches. Re-derivable from `data/` and
the architecture's preprocessing scripts; not checked in.

For BRIC:

```
training/bric/cache/
├── players/      per-source-video striker bbox tracks (npz, wide format)
├── shuttle/      per-source-video TrackNetV3 trajectory (npz)
└── rgb/          per-stroke 32-frame striker crop tensor (npy)
```

Caches are content-addressable by the source video they derive from;
running the producing script (`scripts/bric/preprocess_videos.py`,
`scripts/bric/extract_shuttle.py`) on the same input is idempotent.

## `<arch>/experiments/<run_id>/`

One directory per training run, written by the architecture's training
entry point (e.g. `python -m bric.train`). Contains everything needed
to identify the run, reproduce its evaluation, and deploy its model:

```
training/bric/experiments/<run_id>/
├── manifest.yaml    architecture, taxonomy, variant, classes,
                     hparams, seed, git SHA
├── metrics.csv      per-epoch train/val loss, val macro F1, val acc, lr
└── best.pt          best-on-val-macro-F1 model weights
```

A run directory is the deployment unit. To deploy a chosen run, point
the corresponding `runtime/deployed/<arch>/` slot at it — see
`runtime/README.md` for the symlink / rsync workflow.

## Adding a new architecture

A new architecture (`<arch>`) follows the same shape as `bric/`:

1. Create `training/<arch>/cache/` and `training/<arch>/experiments/`
   subdirectories with `.gitkeep` markers.
2. Add the architecture's preprocessing scripts under `scripts/`,
   writing into `training/<arch>/cache/`.
3. Add the architecture's training entry point. It must write its
   experiment dir to `training/<arch>/experiments/<run_id>/` with the
   manifest schema declared above so the deployment workflow applies
   uniformly.
4. Update `.gitignore` to track the new subdirectories' `.gitkeep`
   files following the existing pattern.
5. Register the architecture in the API's inference dispatcher
   (`src/api/inference.py`) keyed on the manifest's `architecture`
   field.
