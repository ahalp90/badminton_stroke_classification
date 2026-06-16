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

Architectures that adopt this tree (currently BRIC) drop their
deployed model into `runtime/deployed/<arch>/`. The backend dispatcher
routes requests to the registered handler for that architecture.
Adding a new architecture also requires implementing the handler — see
`src/api/inference.py` and `src/bric/infer.py` for one example.

## `checkpoints/<dep>/`

Third-party perception weights consumed by inference handlers
(`yolo11`, `tracknetv3`). Each subdirectory carries the upstream
release the project pinned. Update by replacing the subdirectory
contents; checked-in `.gitkeep` files mark the expected layout.

## Fetching the inference weights

Live BRIC inference needs four weight files. All are gitignored, so a
fresh clone has only `.gitkeep` placeholders and the BRIC card degrades
to stub results until they are in place. The model card stays GPU-gated,
so on a CPU-only box nothing is needed here. On the GPU/demo box, fetch
them before `docker compose -f docker-compose.prod.yml up --build` (prod
bakes the image with `COPY . .`, so the files must sit in the repo dir
first; dev bind-mounts the repo, so they are picked up live).

One command fetches and places all four:

```
./scripts/fetch_runtime_weights.sh
```

It needs `gh` (authenticated), `curl`, `unzip`, and `gdown` (`pip install
gdown`) for the Google Drive step; see its `--help` for overrides. The
sources it pulls from, if you ever need them by hand:

1. **BRIC model** (`deployed/bric/<run>/best.pt`). Published in the
   GitHub `models-v1` release.
   ```
   RUN_DIR=runtime/deployed/bric/20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42
   gh release download models-v1 --pattern 'bric_20260518_013238*' --dir "$RUN_DIR"
   mv "$RUN_DIR"/bric_20260518_013238*.pt "$RUN_DIR/best.pt"
   ```

2. **TrackNetV3 shuttle tracker** (`checkpoints/tracknetv3/TrackNet_best.pt`
   and `InpaintNet_best.pt`). From the upstream TrackNetV3 checkpoints
   zip linked in `src/bric/perception/_vendor/tracknetv3/README.md`;
   unzip so the two `.pt` files land directly in `checkpoints/tracknetv3/`.

3. **YOLO11n player detector** (`checkpoints/yolo11/yolo11n.pt`). The stock
   ultralytics YOLO11n checkpoint from their GitHub releases (ultralytics
   also auto-downloads it on first use).

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
