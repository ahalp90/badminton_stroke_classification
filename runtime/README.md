# `runtime/`

Operational state for the deployed application. Everything the API
service needs to handle inference requests lives here. Nothing under
this tree is required to train a model.

```
runtime/
├── deployed/<arch>/        live model package per architecture
├── checkpoints/<dep>/      third-party perception used at inference
├── uploads/                user-uploaded inference videos
├── jobs/                   per-job inference artefacts
└── state/                  backend SQLite (inference.db)
```

## `deployed/<arch>/`

The API serves whatever model is in the architecture's `deployed/`
slot. Exactly one model per architecture; replacing it is the
deployment action.

```
runtime/deployed/bric/
├── manifest.yaml           architecture, taxonomy, variant, classes
└── best.pt                 model weights
```

Promotion onto a host the operator has access to:

- Same host (training also lives here): symlink the chosen experiment.
  ```
  ln -snf training/bric/experiments/<run_id> runtime/deployed/bric
  ```
- Remote serving host: rsync the experiment dir into the deployed slot.
  ```
  rsync -a training/bric/experiments/<run_id>/ \
        host:.../runtime/deployed/bric/
  ```

The API code reads `runtime/deployed/<arch>/manifest.yaml` and loads
`best.pt`; whether `deployed/<arch>/` is a symlink or a real directory
is invisible to the API.

To register a new architecture, create `runtime/deployed/<arch>/` and
add a handler entry in `src/api/inference.py` keyed on the manifest's
`architecture` field.

## `checkpoints/<dep>/`

Third-party perception weights consumed by inference handlers
(`yolo11`, `tracknetv3`). Each subdirectory carries the upstream
release the project pinned. Update by replacing the subdirectory
contents; checked-in `.gitkeep` files mark the expected layout.

## `uploads/`

Inbound video files keyed by job ID. Files persist for the lifetime of
the corresponding `jobs` row in `state/inference.db`.

## `jobs/`

Per-job inference artefacts: stroke frame thumbnails, per-stroke
softmax distributions, and any other intermediate outputs an
inference handler chooses to persist. Layout per job is handler-defined;
see `docs/storage.md`.

## `state/`

Backend operational state. Single SQLite file at
`runtime/state/inference.db` carrying the `players`, `jobs`, and
`strokes` tables. Schema is created on backend boot via `CREATE TABLE
IF NOT EXISTS`. Wiping this file loses inference history but does not
break training or serving — see `docs/storage.md`.
