# Run tracker

Tiny YAML-based experiment tracker. Every train-script invocation writes
one manifest with hparams + per-serial metrics + paths to weights/TB dirs.
Optional Aim UI on top if you want it; works fine without.

## Where the code lives

| File | What it does |
|---|---|
| `src/bst_x/run_tracker.py` | `track_run(config, run_id, log_path=...)` and `track_serial(run_dir, serial_no, weights_path, tb_dir, metrics)`. Writes `manifest.yaml` and (optionally) mirrors into `.aim/`. |
| `src/bst_x/run_overview.py` | Aggregator. `python run_overview.py` prints a table across all runs under `experiments/` (mean/stdev/max per metric). |
| `src/bst_x/aim_backfill.py` | Rebuilds Aim from every manifest + its TB event files: per-epoch curves, hparams, tags (incl `best` on the kept-checkpoint serial), each run dated to its `started_at`. Re-run with `--wipe` for a clean rebuild (see Aim UI below). |
| `src/bst_x/bst_x_train.py` | Integrated: two calls to the tracker, test methods now return metric dicts, TB directory is threaded through to `train_network`. |

## How it's wired into bst_x_train.py

```python
from run_tracker import track_run, track_serial

run_dir, run_id = track_run(config=hyp, run_id=f'run_{timestamp}')
weight_dir = run_dir / 'weights'

for serial_no in range(1, 6):
    tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
    task.seek_network_weights(model_info=..., serial_no=serial_no, tb_dir=tb_dir)
    test_metrics = task.test(...)
    topk_metrics = task.test_topk_acc(k=2)
    track_serial(run_dir, serial_no,
                 weights_path=task.weight_path,
                 tb_dir=tb_dir,
                 metrics={**test_metrics, **topk_metrics})
```

What each call configures:

- **`config=hyp`**: the hparam payload on `track_run`. Accepts any dataclass, namedtuple, dict, or object with `vars()`; lands verbatim under `config:` in `manifest.yaml`.
- **`run_id`**: names the `experiments/<run_id>/` subfolder. `run_{timestamp}` is the convention for regular runs; pass any string for a named/legacy run (e.g. `foundation_chang_baseline`), or pass `None` to let `track_run` auto-generate a `run_YYYYMMDD_HHMMSS` id.
- **`weights_path` / `tb_dir` / `metrics`**: the per-serial payload on `track_serial`; lands in the manifest's `serials:` list. No layout is enforced, but by convention weights live at `run_dir/weights/` and TB events at `run_dir/tb/serial_N/`. `track_serial` is keyed by `serial_no`, so re-running a test updates the entry in place.
- **`log_path=<path>`** (optional, on `track_run`): stored on the manifest so `aim_backfill.py` can slice per-serial blocks out of the test log later. Not needed during the live run; only matters if you want the backfill to enrich Aim descriptions.
- **Aim mirror**: if `aim` is pip-installed (it isn't on the HPC train venv, so usually a no-op), `track_serial` also writes the serial into Aim as a fresh run and force-indexes it. Re-running a serial adds another Aim run rather than overwriting it; the clean rebuild is `aim_backfill.py --wipe` (see Aim UI below). Skips silently when aim is absent; nothing in the training loop breaks either way.

That's the whole integration. Any other train script (Arch 2 3D CNN, or
any future extension) can do the same two calls.

## Directory layout

```
src/bst_x/
  experiments/
    run_20260418_174244/
      manifest.yaml                             (tracked in git)
      weights/bst_x_..._merged_25.pt            (gitignored)
      tb/serial_1/, serial_2/, ...              (gitignored)
  test_logs/
    test_20260418_174244.log                    (unchanged, pairs with run_id)
```

Launch TensorBoard with `tensorboard --logdir experiments/<run_id>/tb` to
see all serials of a run grouped together.

## Manifest format

```yaml
run_id: run_20260418_174244
started_at: 2026-04-18T17:42:44
git_sha: e2c2b74...
git_dirty: true
host: engelbart.une.edu.au
log_path: test_logs/test_20260418_174244.log   # optional; enables aim_backfill
config:
  n_epochs: 80
  lr: 0.0005
  use_aux_schedule: false
  collation_id: taxon_pinned_w_preds   # collation generation (path + manifest tag)
  ablation_id: null                    # nullable training-time tag (augs/loss/wiring)
  classes: [net_shot, return_net, ...] # resolved taxonomy.classes (FE reads this)
  ...
extra:
  data_provenance:
    clips_csv_path: /path/to/clips_master.csv
    clips_csv_sha256: 4b6f...
    collation_id: taxon_pinned_w_preds
    npy_collated_dir: npy_v2_taxon_pinned_w_preds
serials:
  - serial_no: 1
    weights_path: experiments/run_.../weights/bst_x_..._1.pt
    tb_dir: experiments/run_.../tb/serial_1
    metrics:
      macro_f1: 0.834
      min_f1: 0.591
      accuracy: 0.846
      top2_accuracy: 0.963
      num_strokes: 3486
    recorded_at: 2026-04-18T17:48:12
best_serials: [1, 4]                           # optional; 'best'-tag fallback when weights aren't pruned (kept serial wins)
notes: ...                                     # optional; shown as Aim 'run_notes' param
tags: [arch1_baseline]                         # optional; extra Aim tags
```

`track_serial` is idempotent by `serial_no` so re-running a test updates
the entry in place rather than appending a duplicate.

## Aggregator usage

```bash
cd src/bst_x
python ../run_overview.py                              # default experiments/
python ../run_overview.py -c n_epochs,use_aux_schedule -m macro_f1,min_f1
```

Prints one row per run with mean/stdev/max across serials.

## Aim UI (optional)

If `aim` is pip-installed, each `track_serial` call also writes the serial
into Aim (hparams + metrics), force-indexed so it appears without waiting
for the next `aim up`. aim >=3.x can't reopen a chosen run hash, so each
call mints a fresh run (auto hash) named `<run_id>_s<N>` and dates it to
the run's `started_at`. Re-running a serial therefore adds another Aim run
rather than overwriting; the clean rebuild is the backfill below. Browse
(point `--repo` at your Aim repo, or omit for a `.aim` in cwd):

```bash
pip install aim
aim up --repo /path/to/.aim_repos/bst        # local UI at http://localhost:43800
```

If aim is not installed, the tracker silently skips the mirror. Nothing in
the training loop breaks either way.

### Backfill (the canonical way to populate Aim)

`aim_backfill.py` rebuilds Aim from every `experiments/*/manifest.yaml`
plus the run's TB event files. Each serial becomes a run `<run_id>_s<N>`
carrying:

- per-epoch curves from `run_dir/tb/serial_N` (`Loss/*`, `F1/*`,
  `F1_train/*`, `F1_val/*`, `Alpha/*`, `Aug/*`, `Schedule/*`, `best/*`)
  tracked at their real epoch step,
- per-class final F1 (`per_class_f1/<class>`) plus the scalar test metrics,
- hparams, and the test-log block as the description when the log is local,
- tags: `legacy` / anneal-regime, plus `best` on the serial whose
  checkpoint was kept (falls back to manifest `best_serials`, then top
  `macro_f1`),
- the run dated to its `started_at`, not the backfill-import time.

Re-running needs `--wipe`: it removes `.aim` and rebuilds from scratch. An
in-place update can't be made clean because aim's `delete_run` leaves
tag<->run links behind and recycled run-ids inherit them, bleeding tags
between runs. Run it in a venv with aim + tensorboard (locally, tb-viewer):

```bash
~/.venvs/tb-viewer/bin/python aim_backfill.py \
    --repo /path/to/.aim_repos/bst --wipe \
    experiments/bst_x/shuttleset
~/.venvs/tb-viewer/bin/aim up --repo /path/to/.aim_repos/bst
```

Filter to the kept-checkpoint runs in the UI search bar with
`'best' in run.tags`. Re-run after pulling new runs, or after editing
manifest tags / notes.

## Other loggers

The tracker records *paths* for any logger (TB, W&B offline, CSV, plain
text). It does not try to parse arbitrary event formats, so the
cross-run aggregator (`run_overview.py`) only reads metrics from
`manifest.yaml` (which the train script populates from whatever source
it wants). If the team wants cross-run metric scraping to just work,
standardize on passing final metrics into `track_serial(metrics=...)`
regardless of which logger produced them.

## Dependencies

- `pyyaml>=6.0,<7` (required) — add to
  `src/bst_x/requirements.txt` if not
  already there.
- `aim` (optional) — only needed for the Aim UI / `aim_backfill.py`.
- `tensorboard` (optional) — `aim_backfill.py` reads the TB event files
  through it to mirror curves; the tb-viewer venv has both.
