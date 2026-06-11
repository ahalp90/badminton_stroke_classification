# `training/`

Source training data, per-architecture caches, and experiment records.
Required to train a model; not required by a serving host.

```
training/
├── data/                       source training data (shared)
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
├── rally_clips/    per-rally mp4s
└── clips/          per-stroke mp4s
```

Annotations are checked in at the directory level (`.gitkeep`); video
files are gitignored and rsync'd onto the training host. See
`src/bric/preprocessing/slice_rallies.py` for how `rally_clips/` is produced.

## Per-architecture

Each architecture has its own training pipeline. Current state:

- **BRIC** uses this tree — see [BRIC layout](#bric-layout) below.
- **BST-X** organises its training data and experiments under
  `src/bst_x/`; see that subproject's own documentation.

### BRIC layout

`training/bric/cache/` — per-architecture preprocessing caches.
Re-derivable from `data/` and BRIC's preprocessing scripts; not
checked in.

```
training/bric/cache/
├── players/      per-source-video striker bbox tracks (npz, wide format)
├── shuttle/      per-source-video TrackNetV3 trajectory (npz)
└── rgb/          per-stroke 32-frame striker crop tensor (npy)
```

Caches are content-addressable by the source video they derive from;
re-running the producing script (`src/bric/preprocessing/preprocess_videos.py`,
`src/bric/preprocessing/extract_shuttle.py`) on the same input is idempotent.

`training/bric/experiments/<run_id>/` — one directory per training
run, written by `python -m bric.train`. Contains everything needed to
identify the run, reproduce its evaluation, and deploy its model:

```
training/bric/experiments/<run_id>/
├── manifest.yaml    architecture, taxonomy, variant, classes,
                     hparams, seed, git SHA
├── metrics.csv      per-epoch train/val loss, val macro F1, val acc, lr
└── best.pt          best-on-val-macro-F1 model weights
```

## Deploying a new model variant (the hot path)

This is the framework's genuinely pluggable unit: a new training run
of an existing architecture goes live with zero code changes.

For architectures that use this tree's `experiments/<run_id>/`
convention (currently BRIC):

1. Train a new run → produces `training/<arch>/experiments/<run_id>/`
   with `manifest.yaml`, `best.pt`, `metrics.csv`.
2. Symlink or rsync the run into `runtime/deployed/<arch>/` —
   see [`runtime/README.md`](../runtime/README.md).
3. Add a registry entry to `docs/models_registry.yaml`.
4. Backend dispatcher loads the manifest at boot; frontend renders the
   new variant from registry data.

No code changes. New checkpoint → live in production.

## Adding a new architecture

A new architecture is a project, not a drop-in. It requires:

- A new `src/<arch>/` package (dataset, network, train, infer, eval)
  designed for its own input shape and training loop
- Registration in `src/api/inference.py`'s handler dispatcher
- Its own preprocessing scripts and any perception infra it needs
- Optionally adopting the `training/<arch>/` and
  `runtime/deployed/<arch>/` conventions for the deployment hot-path

The conventions in this tree (experiment manifest schema, cache
layout per arch, deployment symlink/rsync workflow) are **opt-in**:
an architecture that adopts them inherits the hot-deployable variant
workflow. An architecture that doesn't manages its own conventions.

See `src/bric/` for one implementation that uses this tree;
`src/bst_x/` for one that maintains its own.

## Shared conventions

For architectures that opt into this tree:

- **Experiment manifest schema** — each
  `<arch>/experiments/<run_id>/manifest.yaml` should declare
  `architecture`, `taxonomy`, `variant`, `classes`, hyperparameters,
  seed, and git SHA. Lets `runtime/deployed/<arch>/` slots point at
  any run uniformly.
- **Cache idempotency** — caches should be content-addressable on
  their source so re-running producing scripts is safe.
- **`.gitignore` pattern** — `<dir>/*` plus `!<dir>/.gitkeep` keeps
  the structure in git without committing data.
