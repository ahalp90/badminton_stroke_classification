# BST-X weight-decay sweep: implementation plan

Plan for adding a swept AdamW `weight_decay` to the BST-X trainer, with a proper no-decay parameter group and per-epoch LR logging. All code anchors are against `feat/taxon-pinned-w-preds` as read on 2026-05-30. Rationale and the timescale maths live in `hp_and_aug_speculations_30_05_2026.md` (Q2); this doc is the build.

Model variant in play: `BST_CG_AP` (hardcoded at `bst_x_train.py:1259`). The verified param split for it is 27 decay / 55 no-decay (instantiated and counted).

## 1. Goal and scope

- Add `weight_decay` as a Hyp field + CLI override, sweepable per cell.
- Replace the single-group `AdamW(model.parameters(), ...)` with a two-group optimiser that excludes norm gains, biases, and the learned tokens/positional embeddings from decay.
- Log the cosine LR per epoch to TB (currently not logged).
- Run a 5-value sweep on `une_v1_14`, schedule + aug + loss held at defaults.

Why: at the current `weight_decay=0.01` (PyTorch's AdamW default, never set explicitly), the AdamW EMA timescale is `tau_epoch = 1/(lambda * eta_peak * M) ~ 1125` epochs against an 80-epoch run, so decay is effectively off. See Q2.

Non-goals: not touching the loss, augmentation, or LR schedule. One lever.

## 2. What the code does today (anchors)

- `Hyp` namedtuple: `bst_x_train.py:70-78`. No `weight_decay` field. Default `hyp = Hyp(...)`: `bst_x_train.py:79-142` (`lr=5e-4` at line 83).
- Optimiser: `bst_x_train.py:517-519`, inside `train_network` (signature at 388-399). `optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)` reads the module-global `hyp`; `weight_decay` unset, so PyTorch's 0.01 applies to every parameter. No param groups.
- Scheduler: `bst_x_train.py:523-528` (cosine + warmup). In scope at the TB logging site.
- TB logging: `bst_x_train.py:597-634`. No LR scalar. `scheduler` is a local here.
- CLI: `bst_x_train.py:1078-1139`. Cell selectors parse into a `cell_overrides` dict, applied via `hyp = hyp._replace(**cell_overrides)` (1129-1139).
- Manifest: `bst_x_train.py:1209-1214`. `config_payload = dict(hyp._asdict())` serialises every Hyp field verbatim, so a new field auto-appears in `manifest.yaml` under `config.` (no manifest-writer edit needed).
- Runner: `collation_runner.py:38-56` (`invoke_bst_train`) builds the per-cell subprocess command (`--taxonomy/--split-column/--collation-id`, optional `--ablation-id`). The 103-line `collation_runner.py` is the simple "one run_id per cell, N serials" driver that ran the current batch; the 1065-line `hparam_sweep.py` is the heavier adaptive sweeper (augmentation, kill rules). The WD sweep uses `collation_runner.py`.

## 3. Code changes

Five edits across two files, plus one config file. The manifest needs no change.

### 3.1 `bst_x_train.py` — add the Hyp field

`bst_x_train.py:70-78`, add `'weight_decay'` to the field list (next to `lr`):

```python
Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'weight_decay', 'warm_up_step',
    'taxonomy', 'seq_len', 'early_stop_n_epochs',
    'pose_style', 'use_3d_pose', 'train_partial',
    'use_aux_schedule', 'aux_fade_end_epoch',
    'clips_csv', 'split_column', 'collation_id', 'ablation_id',
    'label_smoothing', 'class_weights', 'adaptive_focal',
    'augmentation',
])
```

`bst_x_train.py:83`, add the default just after `lr=5e-4,`:

```python
    lr=5e-4,
    # AdamW decoupled weight decay. 0.01 is PyTorch's AdamW default and what
    # every prior run used implicitly; kept as the default so non-sweep runs
    # barely move (norm/bias/embeddings now excluded from decay, but 0.01 on
    # them was near-inert anyway). The sweep overrides this per cell. Optimal
    # lambda for this dataset/LR/run-length is likely 0.1-0.3; see
    # docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md (Q2).
    weight_decay=0.01,
```

namedtuple fields take no per-field defaults here; the module-level `hyp = Hyp(...)` supplies them all by keyword, so adding the field to both the list and the instantiation is the whole change. No positional `Hyp(...)` exists anywhere (tests use `hyp._replace(...)`), so nothing else breaks.

### 3.2 `bst_x_train.py` — no-decay param groups

Replace `bst_x_train.py:517-519` (the comment lines + the single-group AdamW) with:

```python
    # AdamW with decoupled weight decay. Exclude norm gains, biases, and the
    # learned tokens / positional embeddings from decay: decaying an LN/BN gain
    # pulls its scale toward zero, and decaying a sinusoidally-seeded positional
    # embedding erodes the positional signal. Matters at the lambda 0.1-0.4 the
    # sweep covers; standard transformer recipe (Wang & Aitchison don't decay
    # normalization layers). ndim<=1 catches every norm gain/beta and bias;
    # the two name hints catch the five ndim>=2 BST-owned params a shape rule
    # misses. Verified split for BST_CG_AP: 27 decay / 55 no-decay tensors.
    no_decay_name_hints = ('embedding_', 'learned_token_')
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        norm_or_bias = param.ndim <= 1
        token_or_posemb = any(hint in name for hint in no_decay_name_hints)
        (no_decay if norm_or_bias or token_or_posemb else decay).append(param)
    print(f'[optim] AdamW lr={hyp.lr} weight_decay={hyp.weight_decay} '
          f'(decay={len(decay)} tensors, no_decay={len(no_decay)})')
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': hyp.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=hyp.lr,
    )
```

The scheduler block (`bst_x_train.py:523-528`) is unchanged: `get_cosine_schedule_with_warmup` takes the optimiser regardless of how many param groups it has, and applies the same LR multiplier to both.

### 3.3 `bst_x_train.py` — log the LR

After `bst_x_train.py:607` (`writer.add_scalar('Schedule/aux_factor', aux_factor, epoch)`), add:

```python
        # Cosine LR per epoch. Deterministic from the schedule, but logging it
        # saves the reconstruction and overlays cleanly with the per-class F1 /
        # alpha arcs. get_last_lr()[0] = LR after this epoch's final step.
        writer.add_scalar('Schedule/learning_rate', scheduler.get_last_lr()[0], epoch)
```

### 3.4 `bst_x_train.py` — CLI override

Add the arg after `bst_x_train.py:1091` (`--ablation-id`), before `parse_args()`:

```python
    parser.add_argument('--weight-decay', type=float, default=None)
```

Add to the `cell_overrides` block after `bst_x_train.py:1137` (the `--ablation-id` clause), before `if cell_overrides:`:

```python
    if args.weight_decay is not None:
        cell_overrides['weight_decay'] = args.weight_decay
```

This rides the existing `hyp._replace(**cell_overrides)` at line 1139, and `dict(hyp._asdict())` at 1209 then writes `config.weight_decay` into the manifest for free.

### 3.5 `collation_runner.py` — forward the flag

In `invoke_bst_train`, after `collation_runner.py:55` (the `--ablation-id` clause), before the `return`:

```python
    # Optional per-cell weight decay (the WD sweep dimension); cells without it
    # fall back to the bst_x_train Hyp default.
    if cell.get('weight_decay') is not None:
        cmd += ['--weight-decay', str(cell['weight_decay'])]
```

`hparam_sweep.py` needs no change; the WD sweep runs through `collation_runner.py`.

## 4. Sweep session

New file `scratch/runners/wd_sweep_une_v1_14/config.yaml`. `collation_runner` reads `config['cells']`; each cell needs `name`, `taxonomy`, `split_column`, `collation_id`, and (now) `weight_decay`, plus optional `n_serials`. All five cells share the une_v1_14 / split_v2 / taxon_pinned_w_preds collation and differ only in `weight_decay`. Aug, loss, and schedule come from the bst_x_train module defaults (flip 0.5 / jitter 0.3, CDB-F1 tau1 gamma1, cosine + warmup 100), so the sweep isolates weight decay.

```yaml
# AdamW weight-decay sweep on une_v1_14. One cell per lambda; everything else
# at bst_x_train module defaults (aug flip0.5/jitter0.3, CDB-F1 tau1 gamma1,
# cosine warmup100, 80 epochs). lambda 0.01 is the apples-to-apples anchor
# (same no-decay grouping, just the old value).
cells:
  - name: wd_0p01
    taxonomy: une_v1_14
    split_column: split_v2
    collation_id: taxon_pinned_w_preds
    weight_decay: 0.01
    n_serials: 5
  - name: wd_0p05
    taxonomy: une_v1_14
    split_column: split_v2
    collation_id: taxon_pinned_w_preds
    weight_decay: 0.05
    n_serials: 5
  - name: wd_0p1
    taxonomy: une_v1_14
    split_column: split_v2
    collation_id: taxon_pinned_w_preds
    weight_decay: 0.1
    n_serials: 5
  - name: wd_0p2
    taxonomy: une_v1_14
    split_column: split_v2
    collation_id: taxon_pinned_w_preds
    weight_decay: 0.2
    n_serials: 5
  - name: wd_0p4
    taxonomy: une_v1_14
    split_column: split_v2
    collation_id: taxon_pinned_w_preds
    weight_decay: 0.4
    n_serials: 5
```

What each lambda means as a timescale (une_v1_14: N=22743, batch 128 so M=178 iters/epoch, eta_peak=5e-4; `tau_epoch = 11.25/lambda`, run is 80 epochs):

| lambda | tau_epoch | vs 80-epoch run |
|---:|---:|---|
| 0.01 | ~1125 | 14x (near-inert, current) |
| 0.05 | ~225 | 2.8x |
| 0.1 | ~112 | 1.4x |
| 0.2 | ~56 | 0.7x |
| 0.4 | ~28 | 0.35x |

Launch (repo root, both package roots on PYTHONPATH; on bourbaki use venv-bst and set `BST_X_COLLATED_DATA_ROOT` so the collation dir resolves):

```bash
PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \
  python -m main_on_shuttleset.collation_runner scratch/runners/wd_sweep_une_v1_14
```

`state.json` is written alongside `config.yaml`, so a killed runner resumes mid-sweep without re-running finished serials.

## 5. Validation before launch

Run on the laptop CPU first (`/home/ariel/.venvs/badminton-cicd/bin/python`, which has the model stack + pytest; `conftest.py` wires the src paths):

```bash
/home/ariel/.venvs/badminton-cicd/bin/python -m pytest \
    tests/test_train_surface.py tests/test_network.py -q
```

`test_train_surface.py::test_train_network_returns_model_and_val_at_best` calls `bt.train_network` for a real 2-epoch CPU run, so it exercises the new param-group optimiser AND the new `get_last_lr()` logging line directly. It patches `hyp` via `_replace` without setting `weight_decay`, so it inherits the 0.01 default. If the param-group build or the LR-logging line throws, this test fails. Run the full suite too (`pytest -q`) since the manifest gains a `config.weight_decay` key (no test pins the exact config key set, but confirm).

Param-split sanity (one-off; expect `decay=27 no_decay=55`): the `[optim]` print now emits the split at every run start, so the first serial's stdout/TB is the check. To eyeball it before launch, the standalone enumerator is at `/tmp/wd_groups.py` (instantiates BST_CG_AP and applies the exact rule).

Dry run: launch the sweep, watch the first serial of `wd_0p01`. Confirm the `[optim]` line reads `weight_decay=0.01 (decay=27 no_decay=55)`, and that `Schedule/learning_rate` shows the warmup ramp then cosine descent in TB. Kill and let the full sweep proceed if clean.

## 6. Run plan and cost

5 cells x 5 serials = 25 serials. The current batch did 5 serials in ~1.5 h, so ~18 min/serial gives ~7.5 h for the sweep. First-pass option: set `n_serials: 3` (15 serials, ~4.5 h) to find the region, then top up the winner's cell to 5 by editing its `n_serials` and re-running the runner (it resumes from `state.json`).

Hold the schedule fixed across all cells (it is, by construction). Don't co-tune.

## 7. Analysis

Per cell, read `manifest.yaml`: `serials[].metrics` (test per-class + macro/min) and `serials[].extra.val_at_best_macro_epoch` (val per-class at the saved epoch). Compare macro and min-F1 vs lambda across the five cells; pick the lambda that lifts macro without sinking min-F1 (tie-break on min). Expect the useful zone around 0.1-0.2; 0.4 may start to underfit from scratch.

The new `Schedule/learning_rate` plus the existing per-class `F1_val/{c}` and `Alpha/{c}` arcs support the per-epoch overlays from the Q4 analysis, now with LR included.

If a lambda clearly wins on une_v1_14, confirm it transfers with single cells on shuttleset_18 and bst_24 (same lambda values; bst_24 has N=24866 so its tau_epoch is ~8% lower per lambda, negligible). Then update the `weight_decay=0.01` Hyp default to the winner.

## 8. Risks and rollback

- Baseline shift: with the param groups in, even the default 0.01 stops decaying norm/bias/embeddings. The effect is negligible (0.01 decay had tau_epoch ~1125, near-inert), but it means a future default run is not bit-identical to the currently-running batch (`run_20260530_161525`, old all-params-at-0.01 behaviour). State this when comparing.
- From-scratch instability at the 0.4 end: watch for loss spikes / stalled macro in the `wd_0p4` cell; if it diverges, that cell just loses and the rest stand.
- Don't edit `bst_x_train.py` while the current batch can still relaunch a serial (the runner spawns fresh subprocesses that import the edited module). Land these edits on the branch and run the WD sweep as its own session after the current batch finishes.
- Rollback: the change is behaviourally close to the old path at `weight_decay=0.01`. To fully revert, restore the single-line `optim.AdamW(model.parameters(), lr=hyp.lr)` and drop the Hyp field / CLI arg / runner clause; the manifest simply stops carrying `config.weight_decay`.

## 9. Edit checklist

- [ ] `bst_x_train.py:70-78` add `'weight_decay'` to the Hyp field list
- [ ] `bst_x_train.py:83` add `weight_decay=0.01` to the default `hyp`
- [ ] `bst_x_train.py:517-519` replace AdamW with the two-group build + `[optim]` print
- [ ] `bst_x_train.py` after 607 add the `Schedule/learning_rate` scalar
- [ ] `bst_x_train.py:1091` add `--weight-decay` arg; after 1137 add it to `cell_overrides`
- [ ] `collation_runner.py:55` forward `--weight-decay` from `cell.get('weight_decay')`
- [ ] `scratch/runners/wd_sweep_une_v1_14/config.yaml` new (5 cells)
- [ ] `pytest tests/test_train_surface.py tests/test_network.py` green on CPU
- [ ] dry-run first serial: `[optim]` reads 27/55, LR curve logs
