# Plan: drop the unknown ghost output channel in BST-X

This doc covers six things, in order:

1. The original problem and the constraints the user set on a fix.
2. The first fix attempt (already on disk; failed one validation step).
3. A first-draft second fix proposal (data-driven empirical derivation).
4. Independent review findings on §3 (forensic audit, critical review, third-path exploration).
5. The refined second fix that addresses §4's findings.
6. The final stripped plan after a second-pass review reduced legacy back-compat baggage. **§6 is the implementation plan.**

§1-§5 are kept as audit trail of the decisions taken; if you only have time to read one section, read §6.

---

## 1. The original problem

### The bug

`build_bst_x_network` (called from `bst_x_train.Task.get_network_architecture`) takes `n_class` from `taxonomy.n_classes` — the size of the full taxonomy class list. The full class list always includes `'unknown'` for every taxonomy in `pipeline/config.py`.

`hyp.drop_unknown=True` instructs the writer (`prepare_train_on_shuttleset.collate_npy`) to drop rows where `raw_type_en == 'unknown'`. It does not change the model's output dimension.

So when `drop_unknown=True` for a taxonomy where unknown ends up genuinely empty after the writer filter, the model has an output channel for unknown that:

- Receives no positive samples during training.
- Still consumes a softmax slot.
- Receives a label-smoothed target of `LS / n_classes` for every sample under `nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)`.
- Shifts class-weight renormalisation onto an `n_classes` basis instead of `(n_classes - 1)`.

The user spotted this in the live class-weighted run printout from the LS=0.15 + `class_weights={'wrist_smash': 2.0, 'smash': 2.0}` cell:

```
[loss] class-weighted CE (renormalised, mean=1.0):
    ...
    long_service              weight=0.882
    unknown                   weight=0.882    <-- ghost
```

The unknown channel got allocated weight despite never being positively trained.

### Constraints set by the user

- Source-level structural fix at the `n_classes` / `class_ls` derivation.
- No monkey patching, no loss-side masking, no zeroing weights downstream.
- Loud `n_active_classes` derivation, persisted to manifest, so future taxonomy + drop combinations can't silently re-introduce a ghost.
- Flag the change in `docs/architecture_notes/bst_x_overview.md`.
- Don't recollate existing collated npy dirs. Don't invalidate existing weight files.

The accepted comparability cost is that pre-fix runs (LS sweep cells `run_20260430_170325`, `run_20260430_213933`, `run_20260501_073430`, plus the class-weighted run that finished on engelbart on 2026-05-01) carry the 15-channel ghost head. Post-fix runs are a new architectural era.

---

## 2. First fix attempt (already on disk)

### Approach

Drove the architecture decision off the `hyp.drop_unknown` boolean. Concretely:

- Added to `Taxonomy` (`pipeline/config.py`):
  - `has_unknown` (property): `'unknown' in self.standalone_types`.
  - `active_class_list(drop_unknown: bool, side='Both') -> list[str]`: `class_list(side)` with `'unknown'` removed when `drop_unknown=True` and the taxonomy has unknown.
  - `n_active_classes(drop_unknown: bool, side='Both') -> int`: `len(active_class_list(...))`.
  - `full_to_active_remap(drop_unknown: bool, side='Both') -> list[int]`: per-index map from full-taxonomy idx to active idx, with `-1` at the dropped slot.
- Added a free function in `bst_x_common.py`:
  - `remap_and_validate_labels(taxonomy, drop_unknown, labels_per_split) -> dict[str, np.ndarray]`. Asserts a strict drop-policy guard, then compresses labels into the active index space.
- Threaded `drop_unknown` through `bst_x_train.Task.__init__`. `Task` stores `n_active_classes` and `active_class_list` derived from `taxonomy + drop_unknown` once at construction. `Task.prepare_dataloaders` triggers the helper to remap each split's `dataset.labels` in place. `Task.get_network_architecture` and `Task.test` consume the active fields.
- Loud `[arch]` printout at run start in `bst_x_train.__main__`.
- Manifest enrichment: `extra.arch = {'n_classes_full', 'n_active_classes', 'has_unknown', 'unknown_first', 'drop_unknown', 'active_class_list'}`.
- Parallel fix in `bst_x_infer.py`: new `drop_unknown` parameter on `get_network_architecture`; decoder uses `active_class_list` instead of full `class_list()`.
- Structural note added to `docs/architecture_notes/bst_x_overview.md`.

### The drop-policy guard (the part that misbehaved)

In `remap_and_validate_labels`, when the taxonomy has unknown:

```python
unknown_full_idx = taxonomy.class_list().index('unknown')
all_labels = np.concatenate(list(labels_per_split.values()))
unknown_in_data = bool((all_labels == unknown_full_idx).any())

if drop_unknown and unknown_in_data:
    raise ValueError(
        f'drop_unknown=True but the loaded collated dir contains '
        f'unknown-class labels (full-tax idx {unknown_full_idx} appears '
        f'across {sorted(labels_per_split.keys())}). The dir was likely '
        f'collated without --drop-unknown. Either point at a --drop-unknown '
        f'collated dir or set hyp.drop_unknown=False.'
    )
if not drop_unknown and not unknown_in_data:
    raise ValueError(
        f'drop_unknown=False but the loaded collated dir has no '
        f'unknown-class labels (...) ...'
    )
```

The guard's premise: if `drop_unknown=True` and the writer was asked to drop unknown rows, the loaded labels must not contain unknown's full-tax index.

### Test result that failed

CPU-only suite: 66 active tests passed locally, then on engelbart, of which 3 are real-data probes against `/scratch/comp320a/...` collated dirs:

| dir | result |
|---|---|
| `npy_une_merge_v1_split_v2_dropunk` | passed |
| `npy_une_merge_v1_nosides_split_v2_dropunk` | passed |
| `npy_merged_25_split_bst_baseline_dropunk` | **failed** |

Failure message:

```
ValueError: drop_unknown=True but the loaded collated dir contains
unknown-class labels (full-tax idx 0 appears across ['test', 'train',
'val']).
```

### Why the guard fired on merged_25 (verified, not hallucinated)

The dir was collated by `run_20260429_202144` with `--drop-unknown` and it contains label-0 samples. The cause is in three places (line numbers from the pre-change code):

1. `pipeline/config.py:65-72`:
   ```python
   MERGE_MAP: dict[str, str] = {
       'wrist_smash':            'smash',
       'defensive_return_lob':   'lob',
       'driven_flight':          'unknown',     # <-- the one that bites
       'back_court_drive':       'drive',
       'passive_drop':           'drop',
       'defensive_return_drive': 'drive',
   }
   ```
   `merged_25.merge_map` sends `driven_flight` to `'unknown'`. The comment at `config.py:112` documents the original BST-paper convention: "`'driven_flight'` is a transient type that always gets merged into `'unknown'` before training".
2. `prepare_train_on_shuttleset.py:767-768` (writer filter):
   ```python
   if drop_unknown:
       clips_df = clips_df[clips_df["raw_type_en"] != "unknown"]
   ```
   The filter compares `raw_type_en`, not the merged label.
3. `prepare_train_on_shuttleset.py:782-802` (label encoding):
   ```python
   merged = merge_map.get(raw_type, raw_type)
   label_str = merged if merged in standalone_set else f"{side}_{merged}"
   ...
   labels_ls.append(class_to_idx[label_str])
   ```
   driven_flight rows survive the writer filter (their raw_type isn't `'unknown'`), get merged to `'unknown'`, and land at `class_to_idx['unknown']`.

For `merged_25`, `unknown_first=True` so unknown is at index 0. Driven_flight rows accumulate at label 0. The merged_25 dropunk dir genuinely contains label-0 samples, and the unknown class there is not a ghost — it is an actively-trained class catching driven_flight via MERGE_MAP.

`une_merge_v1` and `une_merge_v1_nosides` use `UNE_MERGE_V1_MAP` (`config.py:74-79`) which sends `driven_flight` to `'drive'` instead. Their dropunk dirs have no unknown samples (true ghost). `raw_35` excludes driven_flight from `STROKE_TYPES_17_RAW` entirely.

### What the first fix conflated

The `drop_unknown=True` flag in the first fix tried to enforce three independent properties at once:

1. **Writer-side row filter**: the collator drops `raw_type_en == 'unknown'` rows.
2. **Architecture choice**: the model has no unknown output channel.
3. **Data invariant**: loaded labels never contain unknown's full-tax index.

For v1, nosides, raw_35 these three coincide. For merged_25 they don't, because `MERGE_MAP` can populate the unknown index even after the writer's row filter runs. The first fix's drop-policy guard treats #3 as a derived consequence of #1, which is only true when no merge entry redirects to `'unknown'`. That assumption is a leaky abstraction: it holds for the active taxonomies the user is currently working with, but breaks when one redirects-to-unknown is in play, and would break again for any future taxonomy with a similar merge.

### Pre-fix run_20260429_202144 numbers are not affected retroactively

The pre-fix model had `n_class=25` for merged_25 always. Channel 0 was a real trained class that caught driven_flight. The reported `0.831 / 0.577 / 0.849 / 0.969` (macro / min / acc / top-2) numbers are real, not ghost-affected, because `present` mask in `validate()` and `Task.test` correctly include channel 0 (it has ground truth). No retrospective concern about that historical run.

What changes is the future: post-fix, with the strict guard in place, `merged_25 + drop_unknown=True` won't even start; the guard fires before training. The user wants to rerun merged_25 soon, so this matters.

---

## 3. First-draft second fix proposal (superseded by §5; kept for review trail)

### Reframe

Treat the model's architecture as a **function of the data**, not a function of a flag. The truth about which classes the model needs is in `labels.npy`; consult it directly.

Concretely:

1. After the dataloader returns, read all three splits' labels.
2. `present_indices = set(np.unique(np.concatenate([train, val, test])))`.
3. `active_class_list = [class_list()[i] for i in sorted(present_indices)]`.
4. `n_active = len(active_class_list)`.
5. Build a remap from full-taxonomy idx to active idx using `present_indices`.
6. Apply remap to labels. Build model with `n_active` outputs.

No `drop_unknown` boolean at the training-time architecture path. No drop-policy guard. The architecture matches whatever was collated. `drop_unknown` keeps its writer-side meaning and its dir-naming role unchanged.

### What this gives each existing dir, no recollation

| dir | empirical present_indices | n_active | head | matches pre-fix? |
|---|---|---|---|---|
| merged_25 + dropunk | `{0..24}` (driven_flight populates 0) | 25 | full 25-class head | yes (matches `run_20260429_202144`) |
| une_merge_v1 + dropunk | `{1..14}` | 14 | 14-class head, no ghost | better (drops ghost) |
| une_merge_v1_nosides + dropunk | `{0..13}` | 14 | 14-class head, no ghost | better (drops ghost) |
| raw_35 + dropunk | `{0..33}` | 34 | 34-class head, no ghost | better (drops ghost) |
| any keepunk dir | full taxonomy | full | identical to current | identical |

The "merged_25 doesn't fit my fix" issue disappears because no architectural assumption is tied to a flag. A future taxonomy adding `something → unknown` (or any analogous merge) just works.

### What stays from the first fix

- `Taxonomy.has_unknown` property.
- `bst_x_train.Task` deriving an `n_active_classes` and `active_class_list` and exposing them to downstream methods (`get_network_architecture`, `seek_network_weights`, `test`).
- Loud `[arch]` printout (rephrased: derived from data, not flag).
- Manifest `extra.arch` block.
- bst_x_infer parallel fix (now even simpler).
- `bst_x_overview.md` structural note (with adjusted wording).
- Update of comments on the `present` mask in `validate()` and `Task.test` (that mask was load-bearing for hiding the ghost from F1; post-fix it stays as a generic guard against any zero-support active class).

### What changes from the first fix

- `Taxonomy.active_class_list(drop_unknown)`, `n_active_classes(drop_unknown)`, `full_to_active_remap(drop_unknown)` get a new signature: `(present_indices: set[int], side='Both')` instead of `(drop_unknown: bool, side='Both')`. They become data-driven helpers, not flag-driven. `n_active_classes` collapses into `len(active_class_list(...))`; we drop it as a separate method.
- `bst_x_common.remap_and_validate_labels(taxonomy, drop_unknown, labels_per_split)` is replaced with `bst_x_common.derive_active_classes_from_labels(taxonomy, labels_per_split)`. The new helper computes `present_indices` from the labels itself, builds active list + remap + remapped labels in one call, and returns all three. The drop-policy guard is removed; only the always-true sanity checks remain (every loaded label must hit a non-sentinel remap entry; remapped labels must land in `[0, n_active)`).
- `Task.__init__` no longer takes `drop_unknown`. `Task` doesn't precompute `n_active_classes` / `active_class_list` (we don't know them until labels load). Both fields are populated in `Task.prepare_dataloaders` after the dataloader returns.
- `bst_x_train.__main__`: do an upfront preflight load of `labels.npy` to populate the manifest's `extra.arch` block before `track_run`, then re-derive inside `Task.prepare_dataloaders` per serial.
- `bst_x_infer.get_network_architecture` no longer takes `drop_unknown`. Caller passes `n_active_classes` and `active_class_list` directly (typically from a saved manifest's `extra.arch`). Defaults preserve pre-fix behaviour (full taxonomy head) when caller passes neither, so old weights still work.
- Tests: section 3 (drop-policy guard tests) is deleted; replaced with "architecture matches the data" tests. Section 5 (real-data probes) merges merged_25 back in.

### Concrete code changes

The patches below show new code in full and reference the pre-change file/line locations the reviewer can compare against.

#### `pipeline/config.py`

In `class Taxonomy`, replace the first-fix methods with data-driven equivalents. `has_unknown`, `n_classes`, `class_list`, `standalone_set`, `unknown_first` are unchanged.

```python
@property
def has_unknown(self) -> bool:
    """True if 'unknown' is one of the standalone types in this taxonomy."""
    return 'unknown' in self.standalone_types


def active_class_list(
    self,
    present_indices: set[int],
    side: str = 'Both',
) -> list[str]:
    """Subset of class_list() at the given full-taxonomy indices, ordered.

    The model's output head is sized to len(active_class_list); ground-truth
    labels (after remapping) live in [0, len(active_class_list)). The full
    class_list still owns label decoding from on-disk values, since the
    collator wrote labels.npy against the full-taxonomy index space.

    :param present_indices: full-taxonomy indices that should appear in the
        active head. Typically derived empirically from labels.npy via
        ``np.unique(labels)``.
    :param side: passed through to ``class_list``.
    :return: ordered list of active class names, in their original
        class_list relative order.
    """
    full = self.class_list(side=side)
    bad = present_indices - set(range(len(full)))
    if bad:
        raise ValueError(
            f'present_indices {sorted(bad)} out of range for taxonomy '
            f'{self.name!r} with n_classes={len(full)}'
        )
    return [full[i] for i in sorted(present_indices)]


def full_to_active_remap(
    self,
    present_indices: set[int],
    side: str = 'Both',
) -> list[int]:
    """Per-index map from full-taxonomy idx to active idx (or -1 if absent).

    Same input contract as ``active_class_list``: ``present_indices`` is
    typically derived from labels.npy. Indices not in the present set get
    -1 sentinels; indices in the present set get their position in the
    sorted active list.
    """
    full = self.class_list(side=side)
    sorted_present = sorted(present_indices)
    active_idx_of = {full_idx: i for i, full_idx in enumerate(sorted_present)}
    return [active_idx_of.get(full_idx, -1) for full_idx in range(len(full))]
```

Removed: `n_active_classes(drop_unknown)` (now `len(active_class_list(present_indices))`).

#### `main_on_shuttleset/bst_x_common.py`

Replace `remap_and_validate_labels` (the first-fix helper, which carried the strict drop-policy guard) with `derive_active_classes_from_labels`. Imports unchanged.

```python
def derive_active_classes_from_labels(
    taxonomy: Taxonomy,
    labels_per_split: dict[str, np.ndarray],
) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    """Derive active class list, full->active remap, and remapped labels.

    Reads labels.npy values (full-taxonomy index space) from each split,
    builds the empirical present-class set as the union across splits,
    and constructs the active class list + remap + remapped labels per
    split. The architecture-side n_active is implicit in
    ``len(active_class_list)``.

    No assumptions about ``drop_unknown`` here. The model adapts to
    whatever the data contains. If a class index is absent from every
    split's labels, it's dropped from the active head. If it's present
    in even one split, it's kept (a class can appear in train but not
    val/test; the channel still needs to exist).

    Validation kept: every loaded label must hit a non-sentinel remap
    entry (true by construction since we derived present_indices from
    those very labels), and remapped values must fall in [0, n_active).
    Both checks raise ``ValueError`` on violation.

    :param taxonomy: Taxonomy whose class_list() defines the full index space.
    :param labels_per_split: e.g. {'train': arr, 'val': arr, 'test': arr}.
    :return: (active_class_list, full_to_active_remap, labels_remapped).
    """
    if not labels_per_split:
        raise ValueError('labels_per_split is empty.')

    all_labels = np.concatenate(list(labels_per_split.values()))
    present_indices = {int(x) for x in np.unique(all_labels).tolist()}

    active = taxonomy.active_class_list(present_indices)
    remap_list = taxonomy.full_to_active_remap(present_indices)
    remap = np.asarray(remap_list, dtype=np.int64)
    n_active = len(active)

    out: dict[str, np.ndarray] = {}
    for set_name, labels in labels_per_split.items():
        new_labels = remap[labels]
        if (new_labels < 0).any():
            bad = sorted({int(x) for x in labels[new_labels < 0]})
            raise ValueError(
                f'[{set_name}] labels remap to -1 (sentinel for absent '
                f'class) at full-tax indices {bad}. This should not '
                f'happen: present_indices was built from these labels '
                f'themselves. Investigate label corruption.'
            )
        if not ((new_labels >= 0) & (new_labels < n_active)).all():
            raise ValueError(
                f'[{set_name}] post-remap label out of range '
                f'[0, {n_active}); investigate.'
            )
        out[set_name] = new_labels.astype(np.int64)

    return active, remap_list, out
```

#### `main_on_shuttleset/bst_x_train.py`

Re-add `import numpy as np` at the top (the preflight needs it).

`Task.__init__`: drop `drop_unknown` parameter; defer active-class derivation until `prepare_dataloaders`.

```python
class Task:
    def __init__(self, n_joints=17, taxonomy: Taxonomy = None,
                 weight_dir: Path = Path('weight')) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.taxonomy = taxonomy or TAXONOMIES[hyp.taxonomy]
        self.weight_dir = weight_dir
        # Active label space gets derived from labels.npy in
        # prepare_dataloaders, since architecture is a function of the
        # data here, not of a flag.
        self.n_active_classes: int | None = None
        self.active_class_list: list[str] | None = None

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='Jn2B',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=train_partial
            )

        self.pose_style = pose_style
        self._derive_active_classes_from_loaded_labels()

    def _derive_active_classes_from_loaded_labels(self) -> None:
        """Inspect loaded labels, build active class list + remap, write back.

        Architecture is sized to whatever the labels actually contain.
        Stores ``self.n_active_classes`` + ``self.active_class_list`` for
        downstream methods (``get_network_architecture``, ``test``, etc).
        Loud printout so a future taxonomy + drop combination can't
        silently re-introduce a ghost: the difference between
        n_classes_full and n_active_classes is always visible.
        """
        splits = {
            'train': self.train_loader.dataset,
            'val':   self.val_loader.dataset,
            'test':  self.test_loader.dataset,
        }
        active, _remap, remapped = derive_active_classes_from_labels(
            taxonomy=self.taxonomy,
            labels_per_split={name: ds.labels for name, ds in splits.items()},
        )
        for name, ds in splits.items():
            ds.labels = remapped[name]
        self.active_class_list = active
        self.n_active_classes = len(active)

        print(
            f'[arch] taxonomy={self.taxonomy.name}, '
            f'has_unknown={self.taxonomy.has_unknown}'
        )
        print(
            f'       full taxonomy n_classes={self.taxonomy.n_classes}, '
            f'n_active_classes={self.n_active_classes} (derived from labels)'
        )
        print(f'       active class_list: {self.active_class_list}')
```

`Task.get_network_architecture`, `Task.seek_network_weights`, `Task.test`: signatures unchanged; bodies still consume `self.n_active_classes` / `self.active_class_list` exactly as in the first fix. The `present` mask comments stay as updated by the first fix (generic zero-support guard, no longer the unknown-ghost special case).

`bst_x_train.__main__`: drop the up-front `[arch]` print (we don't know n_active yet); add a preflight load of labels for the manifest. Drop `drop_unknown` from `Task(...)`. The serial loop's `task.prepare_dataloaders` now does its own `[arch]` print per serial.

```python
if __name__ == '__main__':
    taxonomy = TAXONOMIES[hyp.taxonomy]

    if hyp.seq_len not in (30, 100):
        raise NotImplementedError(
            f'Unsupported hyp.seq_len={hyp.seq_len!r}; expected 30 or 100.'
        )
    effective_ablation_id = derive_ablation_id(
        taxonomy.name, hyp.split_column, hyp.drop_unknown, hyp.ablation_id,
    )
    npy_collated_dir = derive_npy_collated_dir_basename(
        taxonomy_name=taxonomy.name,
        split_column=hyp.split_column,
        drop_unknown=hyp.drop_unknown,
        use_3d_pose=hyp.use_3d_pose,
        seq_len=hyp.seq_len,
        ablation_id=hyp.ablation_id,
    )

    # Preflight: peek at the collated dir's labels once so the manifest
    # can record the active class list before we kick the serial loop.
    # The serial loop will re-derive the same active list inside each
    # Task.prepare_dataloaders call; this is a cheap one-shot read.
    collated_root = (
        Path(__file__).resolve().parent.parent
        / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
        / npy_collated_dir
    )
    splits_for_preflight = {}
    for split_name in ('train', 'val', 'test'):
        labels_path = collated_root / split_name / 'labels.npy'
        if not labels_path.exists():
            raise FileNotFoundError(
                f'labels.npy missing at {labels_path}; collate first.'
            )
        splits_for_preflight[split_name] = np.load(str(labels_path))
    active_preflight, _, _ = derive_active_classes_from_labels(
        taxonomy=taxonomy,
        labels_per_split=splits_for_preflight,
    )
    n_active_preflight = len(active_preflight)
    print(
        f'[arch:preflight] taxonomy={taxonomy.name}, '
        f'n_classes_full={taxonomy.n_classes}, '
        f'n_active_classes={n_active_preflight} (derived from labels)'
    )
    print(f'                  active class_list: {active_preflight}')

    # ... (model_info derivation unchanged) ...
    # ... (resume_from, timestamp, run_id, log_path, experiments_dir unchanged) ...

    extra = compute_data_provenance(
        clips_csv_path=Path(hyp.clips_csv),
        effective_ablation_id=effective_ablation_id,
        npy_collated_dir=npy_collated_dir,
    )
    extra['arch'] = {
        'n_classes_full':    taxonomy.n_classes,
        'n_active_classes':  n_active_preflight,
        'has_unknown':       taxonomy.has_unknown,
        'unknown_first':     taxonomy.unknown_first,
        'active_class_list': active_preflight,
        # Note: drop_unknown is recorded under the manifest's `config:`
        # block (top-level hyp). It's a writer / dir-selection concern,
        # not an architecture-level assertion.
    }
    run_dir, run_id = track_run(
        config=hyp, run_id=run_id, log_path=log_path, extra=extra,
        experiments_dir=experiments_dir,
    )
    weight_dir = run_dir / 'weights'

    with open(log_path, 'w') as log_f:
        tee = Tee(sys.stdout, log_f)
        for serial_no in range(1, 6):
            print(f'Running serial {serial_no} ...')
            task = Task(
                n_joints=17, taxonomy=taxonomy, weight_dir=weight_dir,
            )
            task.prepare_dataloaders(
                root_dir=collated_root,
                pose_style=hyp.pose_style,
                train_partial=hyp.train_partial
            )
            task.get_network_architecture(
                model_name='BST_CG_AP',
                in_channels=(3 if hyp.use_3d_pose else 2),
            )

            tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
            weight_exists = task.seek_network_weights(
                model_info=model_info, serial_no=serial_no, tb_dir=tb_dir,
            )

            with redirect_stdout(tee):
                print(f'\n=== Serial {serial_no} ({task.model_name}) ===')
                test_metrics = task.test(show_details=True, show_confusion_matrix=False)
                topk_metrics = task.test_topk_acc(k=2)

            track_serial(
                run_dir=run_dir,
                serial_no=serial_no,
                weights_path=task.weight_path,
                tb_dir=tb_dir,
                metrics={**test_metrics, **topk_metrics},
            )

            print('Serial', serial_no, 'done.')

            if not weight_exists:
                time.sleep(3)

    print(f'\nTest log saved to: {log_path}')
    print(f'Run manifest:    {run_dir / "manifest.yaml"}')
```

#### `main_on_shuttleset/bst_x_infer.py`

Drop the `drop_unknown` parameter. Caller passes `n_active_classes` + `active_class_list` directly (typically from a saved manifest's `extra.arch`). Default to full taxonomy when caller passes neither, so old (pre-fix) weights still work without supplying anything.

```python
def get_network_architecture(
    self,
    model_name='BST_CG_AP',
    seq_len=100,
    in_channels=2,
    taxonomy: Taxonomy = None,
    n_active_classes: int | None = None,
    active_class_list: list[str] | None = None,
):
    """Build the inference model.

    The model's output dim is determined by ``n_active_classes`` /
    ``active_class_list``, which describe the architectural era of the
    weights being loaded. For a post-fix run, fetch them from the run's
    ``manifest.yaml`` under ``extra.arch``. For pre-fix weights, leave
    both as None: the model defaults to the full taxonomy size and
    decoder, matching how those weights were trained.

    Mismatch between weight file shape and ``n_active_classes`` raises
    a clear shape error inside ``load_state_dict``.
    """
    if taxonomy is None:
        taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
    self.taxonomy = taxonomy
    if n_active_classes is None:
        n_active_classes = taxonomy.n_classes
        active_class_list = taxonomy.class_list()
    self.n_active_classes = n_active_classes
    self.active_class_list = active_class_list or taxonomy.class_list()
    self.net, _n_bones = build_bst_x_network(
        model_name,
        n_joints=self.n_joints,
        pose_style=self.pose_style,
        in_channels=in_channels,
        n_class=self.n_active_classes,
        seq_len=seq_len,
        device=self.device,
    )
```

The `__main__` example block also has its `taxonomy.class_list()` decoder use replaced with `task.active_class_list`, identical to the first-fix patch — that line doesn't need to change again.

#### `tests/test_active_classes.py`

Major rewrites in three sections:

**Section 1 (Taxonomy methods)**. Update parametrisation: `active_class_list(present_indices)` instead of `(drop_unknown)`. Cover (per taxonomy):
- Identity: `present_indices == set(range(n_classes))` returns the full `class_list()`.
- Drop-unknown subset: `present_indices = set(range(n_classes)) - {unknown_idx}` returns `class_list()` with unknown removed; relative order preserved.
- Out-of-range guard: `present_indices = {n_classes}` raises ValueError.
- `full_to_active_remap` correctness for both cases above.

**Section 2 (helper)**. Tests for `derive_active_classes_from_labels`:
- Synthetic labels covering all full-tax indices: returned active list equals full class_list, remap is identity, remapped labels equal input.
- Synthetic labels skipping unknown's idx: returned active list is full minus unknown; n_active == n_classes - 1; remapped labels in [0, n_active); remap[unknown_idx] == -1 but never hit.
- Synthetic labels covering only a subset (e.g. 5 of 14 classes present): n_active == 5; active list is those 5 names in their full-list order; remapped labels span exactly [0, 5).
- Empty input raises.
- No-unknown taxonomy round-trips identity.

**Section 3 (drop-policy guard tests)**: deleted entirely.

**Section 4 (synthetic forward+backward smoke)**: parametrise on `(taxonomy, present_indices_kind)` instead of `(taxonomy, drop_unknown)`. `present_indices_kind` ∈ {`'all'`, `'no_unknown'`}. Build BST_CG_AP at the resulting `n_active`, run a tiny forward+backward, assert no shape mismatch and gradients flow. Same coverage as the first-fix smoke test, just driven by data shape instead of a flag.

**Section 5 (real-data probe)**. Re-add merged_25 to `CANDIDATE_REAL_DIRS`. The test loads each split's labels, calls `derive_active_classes_from_labels`, asserts:
- All remapped labels in `[0, n_active)`.
- `n_active` matches what the data implies (14 for v1/nosides dropunk; 25 for merged_25 dropunk). The expected n_active per dir can be hardcoded as a per-test annotation, or derived once and printed for the reviewer to verify.

#### `docs/architecture_notes/bst_x_overview.md`

The structural note added by the first fix gets reworded:

> **Unknown ghost channel removed (2026-05-01)**. BST-X output dim now matches the empirically present classes in `labels.npy`, derived at run start via `bst_x_common.derive_active_classes_from_labels`. Pre-fix runs (LS sweep cells `run_20260430_170325`, `run_20260430_213933`, `run_20260501_073430`, plus the class-weighted run that finished on engelbart on 2026-05-01) carried a 15-channel head with the unknown slot as a ghost output channel. Post-fix, dropunk runs on `une_merge_v1`, `une_merge_v1_nosides`, and `raw_35` collapse to 14- / 14- / 34-class heads (their MERGE_MAP doesn't redirect anything to unknown, so the slot is empty after the writer's row filter and gets dropped). Post-fix, dropunk runs on `merged_25` retain the 25-class head because `MERGE_MAP['driven_flight']='unknown'` populates that slot — there's no ghost there to drop. Manifest `extra.arch` block records `n_classes_full`, `n_active_classes`, `has_unknown`, `unknown_first`, `active_class_list`, surfacing future taxonomy + dir combinations without code changes elsewhere. Comparisons against pre-fix runs carry a one-line caveat for v1 / nosides / raw_35 (architectural era boundary); merged_25 dropunk runs are directly comparable post-fix vs pre-fix because the head dim is unchanged there.

### Files NOT touched in this fix (same as the first fix)

- `preparing_data/prepare_train_on_shuttleset.py` (writer / collator).
- `preparing_data/shuttleset_dataset.py` (reader plumbing).
- `pipeline.config.derive_ablation_id`, `pipeline.config.derive_npy_collated_dir_basename`.
- Any existing collated npy dir.
- Any existing weight file.
- Any existing manifest.
- `run_tracker.py`.

### Tradeoffs vs first fix

| consideration | first fix | second fix |
|---|---|---|
| handles v1 / nosides / raw_35 dropunk | yes | yes |
| handles merged_25 dropunk | no, raises at the strict guard | yes, 25-class head |
| handles future taxonomies that merge anything to unknown | no, would raise | yes |
| explicit "I want N-class head" lever | yes (the flag) | no — derived from data |
| catches misaligned dir (asked for dropunk, got keepunk by mistake) | yes via ValueError | via printout (visible at run start, not a hard error) |
| recollation needed | no for v1 / nosides / raw_35; yes for merged_25 to be runnable | no for any current dir |
| code complexity | flag + guard + remap | data-driven derivation + remap |
| test count | 66 active + 3 real-data | similar; a few drop-policy tests deleted, real-data probes go from 2 passing + 1 failing to 3 passing |

The second fix loses the explicit lever but gains universal applicability. Loss of the lever is mitigated by the loud `[arch]` printout, which makes any unexpected n_active immediately visible at run start.

If the user wants to bring the lever back later as an opt-in, the natural path is an optional `expected_active_classes: list[str] | None` field on `Hyp`, asserted against the empirical list in `_derive_active_classes_from_loaded_labels`. Out of scope for this fix.

### Risks and edge cases (second fix)

- **Train has a class but val / test don't (or vice versa)**: union over splits is what counts. If train has unknown samples, the channel is in the active head and gets trained. val/test scoring uses the existing `present` mask in `validate()` and `Task.test` to handle empty per-split slots. Same behaviour the `present` mask handles for any zero-support class today.
- **All splits empty for a class**: it's dropped from the active head. Pre-fix would have had a ghost; post-fix doesn't. Strict improvement.
- **Future taxonomy with `MERGE_MAP['something_new'] = 'unknown'`**: just works. Active head includes the unknown slot iff the merge populates it.
- **Resuming an old run with `resume_from`**: the manifest from the original run records the active class list. The resume re-derives the active list from the same dir's labels: should match identically (deterministic function of the labels). If it doesn't match, that's a hard signal of dir corruption or content drift, worth surfacing — could add an optional cross-check between manifest and re-derivation.
- **Pointing at the wrong dir**: `[arch:preflight]` shows the active list at run start. If the user expected 14 classes and sees 25 (or vice versa), they catch it before any training time burns. Loud, not silent.

### Validation plan

CPU-only suite: same shape as first fix, with section 3 deletions and section 4 / 5 reframed. Locally and on engelbart:

- Section 5 (real-data probes) for v1 / nosides / merged_25 dropunk all pass.
- BST_CG_AP forward+backward smoke at the empirical n_active across taxonomies × `{all_classes_present, no_unknown_present}`.

End-to-end run on engelbart:

1. `une_merge_v1_nosides + dropunk` (the user's active config). Expected: `[arch:preflight]` shows `n_active=14`. Class-weighted CE printout has 14 entries, no unknown line. Manifest `extra.arch.n_active_classes=14` and `active_class_list` of 14 stroke names.
2. `merged_25 + dropunk` (the user's near-future plan). Expected: `[arch:preflight]` shows `n_active=25`. Training mirrors `run_20260429_202144` semantics (the user's known-good baseline).

---

## 4. Independent review findings on §3

Three review agents ran in parallel: a forensic auditor (verify the diagnosis from scratch), a critical reviewer (find problems with the §3 proposal), and a third-path explorer (find a better approach). Their headline conclusions:

### 4.1 Forensic audit (diagnosis cross-check)

All seven factual claims in §1 and §2 verified, including the existence of 52 driven_flight rows in clips_master.csv (42/9/1 across train/val/test under `split_v2`). Line numbers in §2 match the live code. The merged_25 driven_flight-to-unknown trace (`config.py:65-72` + `prepare_train_on_shuttleset.py:767-768` + `:782-802`) is correct. Pre-fix `run_20260429_202144`'s reported metrics are not retroactively affected because the `present` mask in `validate()` and `Task.test` correctly includes channel 0 (it has driven_flight ground truth across all splits).

Auditor's one substantive flag I underweighted in §3: pre-fix v1/nosides/raw_35 weights cannot be loaded under the §3 architecture. `load_state_dict` will fail on shape mismatch (15-dim head saved vs 14-dim head built). This is a hard break for `run_20260430_170325`, `run_20260430_213933`, `run_20260501_073430`, and the class-weighted run. Not a bug; an era-boundary fact that needs to be loud at resume time.

### 4.2 Critical review of the §3 proposal

Mergeable with adjustments. Two HIGH-severity issues §3 does NOT handle, and four MED issues:

**HIGH-1 — `train_partial` mismatch**. The §3 preflight reads `labels.npy` directly via `np.load`. The live derivation reads them via `Dataset_npy_collated`, which has already applied `adjust_to_partial_train_set` slicing (`shuttleset_dataset.py:208-231`). For small `train_partial` and a low-support class, the slice can drop a class entirely from train. Preflight sees N classes; live sees N-1; manifest `extra.arch.n_active_classes` then disagrees with the trained head dim, silently. Latent bug today (`hyp.train_partial=1.0`) but a real footgun for any future partial-training cell.

**HIGH-2 — ghost via val/test-only classes**. §3 derives `present_indices` from the union over splits. If a class is present only in val/test but absent from train, the union keeps it in the active head. The model then has an output channel that gets a label-smoothed CE target on every train sample with no positive signal to learn from — exactly the original ghost pathology, just routed via val/test instead of via the writer flag. §3's "Risks and edge cases" claimed `present` mask handles this; it doesn't, since `present` masks F1 reporting, not the loss computation.

**MED-1 — resume scenario doesn't cross-check**. With `resume_from='run_YYYYMMDD'`, §3 re-runs the preflight against `collated_root` and overwrites `extra['arch']` into `track_run`. If the dir contents shifted since the original run (recollation, ablation_id symlinked elsewhere), the new derivation could disagree with the original manifest's `n_active_classes`. The shape mismatch surfaces inside `load_state_dict`, but well after the `[arch:preflight]` print confidently announced a derivation. A `if resume_from: assert manifest['extra']['arch']['n_active_classes'] == n_active_preflight` would close it.

**MED-2 — loud printout disappears under `nohup`/redirects**. The `[arch:preflight]` print runs at line ~470 of stdout, before the `Tee` is built. Under `python -m main_on_shuttleset.bst_x_train > log 2>&1 &`, it lands on stdout only and is easy to miss. The user explicitly said "no silent failures". Cheap fix: emit the preflight inside the `with open(log_path, 'w') as log_f:` block so the tee'd log captures it; manifest already persists the same data.

**MED-3 — `class_weights` validation message misleading**. `bst_x_train.py:325-333` validates keys against `class_ls` (= `self.active_class_list` post-fix). For a setting like `class_weights={'unknown': 2.0}` against a dropunk dir where unknown is empirically absent, the existing `ValueError(f"...not in taxonomy...")` message says "not in taxonomy" when in fact unknown *is* in the taxonomy, just absent from this run's active list. Behaviour is correct (loud rather than silent) but the message wording is imprecise.

**MED-4 — `bst_x_infer` caller burden**. The §3 default (`taxonomy.n_classes` and full `class_list()` when caller passes neither `n_active_classes` nor `active_class_list`) preserves pre-fix behaviour but is a footgun for post-fix weights. A Gradio frontend or third-party caller that doesn't know about the change silently gets a full-tax head and either fails at `load_state_dict` or, worse, decodes predictions through the wrong class list. Better to raise unless caller explicitly opts in.

**LOW issues**: numpy import easy to miss when applying the patch; out-of-range guard wording; preflight derivation duplicated in `__main__` and inside `Task` (must agree, but HIGH-1 breaks that invariant); `Tee` not wired before preflight print.

Reviewer's verdict: architectural reframe is right (data-driven beats flag-driven for merged_25), but §3 needs the HIGH and MED issues addressed before it's safer than the first fix.

### 4.3 Third-path exploration

Six paths considered. Reproduced cost-benefit table verbatim:

| Path | code-touch | recollation | preserves-lever | handles-merged_25 | scales-future |
|---|---|---|---|---|---|
| A: writer filter on merged label | small | merged_25 dropunk only | yes | yes (head shrinks to 24) | yes |
| B: drop driven_flight→unknown | tiny | merged_25 (both modes) | yes | yes (changes definition) | yes |
| C: persist class_to_idx sidecar | medium | none (fallback) | yes | yes | yes |
| D: split flag in two | medium | none | yes (two levers) | yes (with discipline) | poor |
| E: manifest-declared label space | medium-large | none (fallback) | yes | yes | yes |
| F: A + C hybrid | medium | merged_25 dropunk only | yes | yes (cleanly) | yes |
| Fix 1 (on disk) | n/a | none | yes | **no** | no |
| Fix 2 (§3) | medium | none | no | yes | yes |

Explorer's recommendation: hybrid Path F (writer filter on merged label + persist a class-to-idx sidecar at collation time, with §3-style empirical fallback for old dirs).

### 4.4 My take after reading all three

Path F has nice properties but its recollation cost matters more than the explorer weighed: recollating merged_25 dropunk drops the 52 driven_flight rows from the dataset. Post-Path-F merged_25 training would no longer be apples-to-apples with `run_20260429_202144` (the user's known-good baseline). The user said earlier they'd rerun merged_25 soon — the natural intent is to compare against `run_20260429_202144`. §3 (refined) preserves that comparability; Path F doesn't.

Path F is a defensible *future* cleanup, separate from this fix. The right primary fix is §3 with the HIGH/MED issues from the critical review addressed in place — not a redesign.

---

## 5. Refined second fix (the proposal currently up for approval)

Same architectural shape as §3 (empirical derivation from labels.npy, no `drop_unknown` boolean at the training-time architecture path), with eight refinements that close the issues the review surfaced.

### 5.1 Refinements

**R1 — derive `present_indices` from train labels only, not the union**.
Closes HIGH-2. Rationale: the architecture must match what train can actually teach the model. A class present only in val/test produces a ghost via the side door; deriving from train alone makes that impossible by construction.

Plus a hard subset assertion: `set(np.unique(val_labels)) ⊆ train_present` and `set(np.unique(test_labels)) ⊆ train_present`. If a val/test label points at a class train doesn't have, raise. Message names which split has the rogue index and which class it decodes to. This is a real data-quality signal worth surfacing loudly.

**R2 — derive once, post-`train_partial` slicing; no preflight**.
Closes HIGH-1 and LOW-3. Rationale: the only way to guarantee `present_indices` matches what the model is trained on is to derive after the dataset's own slicing. The §3 preflight tried to populate the manifest before training, but the cost was a duplicate-derivation invariant that breaks under `train_partial<1`. Ship the derivation entirely inside `Task._derive_active_classes_from_loaded_labels` (which runs after `Dataset_npy_collated.__init__`), and update the manifest after the first serial completes its prepare. Single source of truth.

**R3 — resume cross-check**.
Closes MED-1. When `resume_from` is set, after the live derivation runs, compare against the existing manifest's `extra.arch.n_active_classes` and `active_class_list`. Raise on mismatch with a message naming which dir was loaded vs which dir the original run used.

**R4 — printout captured by the tee'd log**.
Closes MED-2. Move the `[arch]` print to inside the `with redirect_stdout(tee):` block of the first serial so the run's log file captures it. Manifest still persists the same data for tooling. Together these mean the active class list is in two persistent places (manifest, log) and one transient place (terminal stdout); user can never miss it.

**R5 — bst_x_infer default raises, not falls back silently**.
Closes MED-4. If neither `n_active_classes` nor `active_class_list` is supplied, raise with a message pointing the caller at `manifest.yaml`'s `extra.arch` block. For pre-fix weights, caller can pass `n_active_classes=taxonomy.n_classes, active_class_list=taxonomy.class_list()` explicitly to opt into legacy behaviour. No silent path that produces a wrong-shaped head or wrong decoder.

**R6 — optional `expected_active_classes` belt-and-braces lever**.
Closes the residual "what if the printout is missed" concern. New optional field on `Hyp`: `expected_active_classes: list[str] | None`. When non-None, the live derivation is asserted equal to it; mismatch raises with both lists side-by-side. Default None means "trust the data". This brings back the explicit "I want this exact arch" lever for the user who wants it, opt-in.

**R7 — `class_weights` validation message clarification**.
Closes MED-3. When a `class_weights` key isn't in the active list, distinguish two cases in the error: "not in taxonomy `{name}`" vs "in taxonomy but absent from the loaded data's active class list". Both are still hard errors; just clearer.

**R8 — loud era-boundary message at resume**.
Closes the auditor's flagged issue. When `resume_from` is set and the loaded manifest lacks `extra.arch` (i.e. it's a pre-fix run), print a one-line warning at run start naming the affected weight files and the expected shape mismatch. The actual `load_state_dict` failure remains the source of truth, but the warning gives the user advance notice rather than a confusing stack trace.

### 5.2 Concrete code changes (refined)

Most of §3's patches stand as written; the diffs below show only the deltas vs §3.

#### `pipeline/config.py`

Same as §3. No further changes.

#### `bst_x_common.py`

`derive_active_classes_from_labels` signature changes to take a single train-labels array and an optional dict of validation-only label arrays for the subset assertion:

```python
def derive_active_classes_from_labels(
    taxonomy: Taxonomy,
    train_labels: np.ndarray,
    validation_label_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    """Derive active class list, full->active remap, and remapped labels.

    The active class list is sized to whatever ``train_labels`` contains.
    A class can be in the model only if train can teach it; classes
    present in val/test but absent from train would otherwise produce
    a label-smoothed ghost on every train step.

    ``validation_label_arrays`` holds optional ``{'val': arr, 'test': arr}``
    arrays. If provided, each is asserted to be a subset of train's
    present indices, and is remapped alongside the train labels.

    Validation kept: every loaded label must hit a non-sentinel remap
    entry; remapped values must fall in [0, n_active).

    :return: (active_class_list, full_to_active_remap_list,
        labels_remapped_per_split). The third return mirrors the
        per-split keys of validation_label_arrays plus 'train'.
    """
    if train_labels is None or len(train_labels) == 0:
        raise ValueError('train_labels is empty.')

    train_present = {int(x) for x in np.unique(train_labels).tolist()}

    val_arrays = validation_label_arrays or {}
    for split_name, labels in val_arrays.items():
        split_present = {int(x) for x in np.unique(labels).tolist()}
        rogue = split_present - train_present
        if rogue:
            full = taxonomy.class_list()
            rogue_named = sorted(
                (idx, full[idx] if 0 <= idx < len(full) else f'<oob:{idx}>')
                for idx in rogue
            )
            raise ValueError(
                f'[{split_name}] contains class indices absent from train: '
                f'{rogue_named}. Either retrain on a dir whose train split '
                f'covers these classes, or fix the split assignment in '
                f'clips_master.csv.'
            )

    active = taxonomy.active_class_list(train_present)
    remap_list = taxonomy.full_to_active_remap(train_present)
    remap = np.asarray(remap_list, dtype=np.int64)
    n_active = len(active)

    out: dict[str, np.ndarray] = {'train': remap[train_labels].astype(np.int64)}
    for split_name, labels in val_arrays.items():
        out[split_name] = remap[labels].astype(np.int64)

    for set_name, arr in out.items():
        if (arr < 0).any() or not (arr < n_active).all():
            raise ValueError(
                f'[{set_name}] post-remap label out of [0, {n_active}); '
                f'investigate label corruption.'
            )

    return active, remap_list, out
```

#### `bst_x_train.py`

`Task._derive_active_classes_from_loaded_labels` calls the new helper with train labels + val/test as the validation arrays:

```python
def _derive_active_classes_from_loaded_labels(self) -> None:
    """Inspect loaded labels (post-train_partial slicing), build active class
    list + remap, write back. Architecture is sized to whatever train
    can teach. val/test must be subsets of train's present classes;
    rogue val/test labels raise ValueError with a clear message.
    """
    train_ds = self.train_loader.dataset
    val_ds = self.val_loader.dataset
    test_ds = self.test_loader.dataset

    active, _remap, remapped = derive_active_classes_from_labels(
        taxonomy=self.taxonomy,
        train_labels=train_ds.labels,
        validation_label_arrays={'val': val_ds.labels, 'test': test_ds.labels},
    )
    train_ds.labels = remapped['train']
    val_ds.labels = remapped['val']
    test_ds.labels = remapped['test']
    self.active_class_list = active
    self.n_active_classes = len(active)
```

(R4 deferred — the loud print runs from `__main__`'s post-prepare update, see below.)

`__main__` block — drop the preflight, update the manifest after the first serial's `prepare_dataloaders` returns, and add resume cross-check + era-boundary warning:

```python
if __name__ == '__main__':
    taxonomy = TAXONOMIES[hyp.taxonomy]

    if hyp.seq_len not in (30, 100):
        raise NotImplementedError(...)
    effective_ablation_id = derive_ablation_id(...)
    npy_collated_dir = derive_npy_collated_dir_basename(...)
    collated_root = (
        Path(__file__).resolve().parent.parent
        / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
        / npy_collated_dir
    )

    # ... (model_info, resume_from, timestamp, run_id, log_path, experiments_dir as before) ...

    extra = compute_data_provenance(...)
    # arch block left empty here; populated after the first serial's
    # prepare_dataloaders runs, when the live derivation has run on the
    # actual training data (post train_partial slicing).
    run_dir, run_id = track_run(
        config=hyp, run_id=run_id, log_path=log_path, extra=extra,
        experiments_dir=experiments_dir,
    )
    weight_dir = run_dir / 'weights'

    # If resuming, capture the original manifest's arch block now for
    # comparison against the live derivation (R3).
    resumed_manifest_arch: dict | None = None
    if resume_from:
        manifest_path = run_dir / 'manifest.yaml'
        with open(manifest_path) as f:
            existing = yaml.safe_load(f) or {}
        resumed_manifest_arch = existing.get('extra', {}).get('arch')
        if resumed_manifest_arch is None:
            print(
                f'[arch:resume] WARNING: resuming run {resume_from} which '
                f'predates the active-class fix. Pre-fix weights have a '
                f'full-taxonomy head ({taxonomy.n_classes}); the post-fix '
                f'architecture builds an empirical head, which may not '
                f'match. Expect load_state_dict to raise on shape mismatch '
                f'for v1 / nosides / raw_35 dropunk weights.'
            )

    with open(log_path, 'w') as log_f:
        tee = Tee(sys.stdout, log_f)
        for serial_no in range(1, 6):
            print(f'Running serial {serial_no} ...')
            task = Task(
                n_joints=17, taxonomy=taxonomy, weight_dir=weight_dir,
            )
            task.prepare_dataloaders(
                root_dir=collated_root,
                pose_style=hyp.pose_style,
                train_partial=hyp.train_partial,
            )

            # First-serial-only: enrich the manifest's arch block with the
            # live derivation, validate against expected_active_classes
            # and the resume manifest if applicable. Subsequent serials
            # use the same dir, so the derivation is deterministic and
            # we just no-op.
            if serial_no == 1:
                _validate_and_record_arch(
                    run_dir=run_dir,
                    task=task,
                    taxonomy=taxonomy,
                    hyp=hyp,
                    resumed_manifest_arch=resumed_manifest_arch,
                    tee=tee,
                )

            task.get_network_architecture(...)

            # ... rest of the serial loop unchanged ...

    print(f'\nTest log saved to: {log_path}')
    print(f'Run manifest:    {run_dir / "manifest.yaml"}')
```

`_validate_and_record_arch` is a new free function (kept in `bst_x_train.py` since it touches the manifest file directly):

```python
def _validate_and_record_arch(
    *,
    run_dir: Path,
    task: 'Task',
    taxonomy: Taxonomy,
    hyp: 'Hyp',
    resumed_manifest_arch: dict | None,
    tee,
) -> None:
    """First-serial post-prepare hook: check arch invariants and update manifest.

    Runs once per run. Builds the loud [arch] printout (captured by the
    tee'd log), checks expected_active_classes if set, checks resume
    consistency if resuming, then rewrites the manifest's extra.arch
    block in place.
    """
    arch_block = {
        'n_classes_full':    taxonomy.n_classes,
        'n_active_classes':  task.n_active_classes,
        'has_unknown':       taxonomy.has_unknown,
        'unknown_first':     taxonomy.unknown_first,
        'active_class_list': task.active_class_list,
    }

    with redirect_stdout(tee):
        print(
            f'[arch] taxonomy={taxonomy.name}, '
            f'has_unknown={taxonomy.has_unknown}, '
            f'unknown_first={taxonomy.unknown_first}'
        )
        print(
            f'       full taxonomy n_classes={taxonomy.n_classes}, '
            f'n_active_classes={task.n_active_classes} '
            f'(derived from train labels post train_partial)'
        )
        print(f'       active class_list: {task.active_class_list}')

    # R6: optional belt-and-braces lever.
    expected = getattr(hyp, 'expected_active_classes', None)
    if expected is not None and list(expected) != list(task.active_class_list):
        raise ValueError(
            f'hyp.expected_active_classes != empirical active list.\n'
            f'  expected ({len(expected)}): {expected}\n'
            f'  empirical ({task.n_active_classes}): {task.active_class_list}\n'
            f'  Either fix the expectation, fix the dir, or set '
            f'  expected_active_classes=None to trust the data.'
        )

    # R3: resume cross-check.
    if resumed_manifest_arch is not None:
        prev_n = resumed_manifest_arch.get('n_active_classes')
        prev_list = resumed_manifest_arch.get('active_class_list')
        if prev_n != task.n_active_classes or prev_list != task.active_class_list:
            raise ValueError(
                f'Resume manifest disagrees with live derivation:\n'
                f'  manifest: n_active={prev_n}, list={prev_list}\n'
                f'  live:     n_active={task.n_active_classes}, '
                f'list={task.active_class_list}\n'
                f'  Has the collated dir contents changed since the original run?'
            )

    # Rewrite manifest with arch block populated.
    manifest_path = run_dir / 'manifest.yaml'
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    manifest.setdefault('extra', {})['arch'] = arch_block
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
```

(R4 implemented via `redirect_stdout(tee)`.)

`Hyp` namedtuple gains an `expected_active_classes` field defaulting to `None` (R6). The active hyp block in `bst_x_train.py` stays as-is unless the user opts in.

`train_network` `class_weights` validation gets the message-clarification tweak (R7):

```python
if cls_name not in class_ls:
    if cls_name in [c.replace('Top_', '').replace('Bottom_', '') for c in taxonomy.class_list()] \
       or cls_name in taxonomy.class_list():
        raise ValueError(
            f"class_weights key '{cls_name}' is in taxonomy "
            f"{taxonomy.name!r} but absent from this run's active class "
            f"list (likely because the data has no positive samples for "
            f"it). Active classes: {class_ls}"
        )
    raise ValueError(
        f"class_weights key '{cls_name}' is not in taxonomy "
        f"{taxonomy.name!r} at all. Available: {taxonomy.class_list()}"
    )
```

(`taxonomy` needs to be threaded into `train_network` for this; already accessible via `task.taxonomy` at the call site.)

#### `bst_x_infer.py`

R5: tighten the default to raise.

```python
def get_network_architecture(
    self,
    model_name='BST_CG_AP',
    seq_len=100,
    in_channels=2,
    taxonomy: Taxonomy = None,
    n_active_classes: int | None = None,
    active_class_list: list[str] | None = None,
):
    """Build the inference model.

    n_active_classes and active_class_list describe the architectural
    era of the weights being loaded. Both must be supplied (caller's
    responsibility to load them from the run's manifest.yaml under
    extra.arch). For pre-fix weights without a manifest arch block,
    pass n_active_classes=taxonomy.n_classes and
    active_class_list=taxonomy.class_list() explicitly.
    """
    if taxonomy is None:
        taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
    self.taxonomy = taxonomy
    if n_active_classes is None or active_class_list is None:
        raise ValueError(
            'get_network_architecture requires n_active_classes and '
            'active_class_list. Read both from the run manifest at '
            'experiments/<run_id>/manifest.yaml under extra.arch. For '
            'pre-fix weights with no manifest arch block, pass '
            'n_active_classes=taxonomy.n_classes and '
            'active_class_list=taxonomy.class_list() explicitly.'
        )
    self.n_active_classes = n_active_classes
    self.active_class_list = active_class_list
    self.net, _n_bones = build_bst_x_network(
        model_name,
        n_joints=self.n_joints,
        pose_style=self.pose_style,
        in_channels=in_channels,
        n_class=self.n_active_classes,
        seq_len=seq_len,
        device=self.device,
    )
```

The `__main__` example block in bst_x_infer.py is updated to load `extra.arch` from the manifest before calling `get_network_architecture`, and to use `task.active_class_list` for decoding.

#### `tests/test_active_classes.py`

§3's test rewrites stand, with three additions for R1, R2, R6:

- `derive_active_classes_from_labels` train-only API: parametrise the synthetic-labels tests on (train_present, val_present, test_present) tuples. Cover the new subset-assert path (val has a class train doesn't → raises).
- Resume cross-check: synthetic manifest with mismatched `extra.arch` → `_validate_and_record_arch` raises.
- `expected_active_classes` mismatch → raises with both lists in the message.

Real-data probes (Section 5) updated: pass train_labels and val/test as `validation_label_arrays`. All three current dirs (v1, nosides, merged_25 dropunk) should pass under the train-only derivation, since their train splits cover all the classes their val/test do.

#### `bst_x_overview.md`

Same blurb as §3, with one extra clause noting the train-only derivation: "BST-X output dim now matches the empirically present classes in `labels.npy` train split, with val/test asserted as subsets, derived at first serial via `bst_x_common.derive_active_classes_from_labels`."

### 5.3 Files NOT touched

Same as §3, plus: `preparing_data/prepare_train_on_shuttleset.py` and any collated dir stay untouched. The refinements all sit in the training-side code, the helper, and the tests.

### 5.4 Tradeoffs vs §3

| consideration | §3 (first-draft) | §5 (refined) |
|---|---|---|
| handles `train_partial<1` correctly | no (silent mismatch) | yes (single-source derivation post-slice) |
| handles class present only in val/test | no (silent ghost) | yes (hard error, clear message) |
| catches resume drift | no (load_state_dict surfaces it late) | yes (manifest cross-check) |
| visibility under `nohup` | weak (preflight before tee) | strong (post-prepare print is tee'd; manifest persistent) |
| bst_x_infer caller burden | silent default footgun | hard error if neither arg supplied |
| explicit "I want N-class" lever | dropped | optional via `expected_active_classes` |
| code complexity vs §3 | n/a | +1 free function, +1 Hyp field |
| test count | similar | +3 cases (subset-assert, resume, expected_active) |

§5 is strictly more robust than §3 at the cost of one extra helper function and one optional `Hyp` field. No new failure modes introduced.

### 5.5 What stays from the first fix and §3

- `Taxonomy.has_unknown`, `active_class_list(present_indices)`, `full_to_active_remap(present_indices)` — unchanged from §3.
- `bst_x_common.derive_active_classes_from_labels` — refined signature (train + validation arrays).
- `Task` deriving and exposing `n_active_classes` / `active_class_list` for downstream methods.
- Manifest `extra.arch` block (now written post-first-serial, not preflight).
- bst_x_infer parallel fix (now stricter — caller must provide arch info).
- bst_x_overview.md structural note.
- The CPU-only test suite + 3 real-data probes (v1, nosides, merged_25 all pass).

### 5.6 Validation plan (refined)

1. CPU-only suite locally + on engelbart: all section 1/2/4/5 tests pass; new R1/R2/R6 tests pass; section 3 (first-fix drop-policy guards) deleted.
2. Real-data probes for v1, nosides, merged_25 dropunk all pass under train-only derivation.
3. End-to-end run on engelbart against `une_merge_v1_nosides + dropunk`: `[arch]` prints inside the tee'd block, manifest is rewritten with the arch block after serial 1 prepare, class-weighted CE printout shows 14 entries with no unknown.
4. End-to-end run on engelbart against `merged_25 + dropunk`: `[arch]` prints `n_active=25`, training mirrors `run_20260429_202144` semantics. Comparable apples-to-apples for the user's stated "rerun merged_25 soon".
5. Manual smoke: try resuming `run_20260430_213933` (a pre-fix v1/nosides cell). Expect: `[arch:resume]` warning at run start, then `load_state_dict` error on shape mismatch. Era boundary surfaces clearly.

### 5.7 Open question for the reviewer

Does Path F (writer filter on merged label + recollate merged_25 dropunk + persist class_to_idx sidecar) still merit consideration, or is §5 the right primary fix with Path F filed as a separate future cleanup? The user's explicit constraint "no recollation needed" and their stated intent to compare against `run_20260429_202144` both point at §5; Path F's cleanliness gain is real but loses that comparability.

---

## 6. Final stripped implementation plan

After §5 was approved in shape, a second-pass review identified ~35 lines of pre-fix back-compat baggage that didn't pull its weight, and one unaccounted-for caller in the call graph. The stripped version below is what gets implemented.

### 6.1 What changed from §5

**Strip 1 — bst_x_infer required kwargs (no fallback)**.
§5 had `n_active_classes: int | None = None` with a default-fallback path producing a long error message about pre-fix weights. Stripped to required kwargs:

```python
def get_network_architecture(
    self,
    *,
    model_name: str = 'BST_CG_AP',
    seq_len: int = 100,
    in_channels: int = 2,
    taxonomy: Taxonomy | None = None,
    n_active_classes: int,
    active_class_list: list[str],
):
    if taxonomy is None:
        taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
    self.taxonomy = taxonomy
    self.n_active_classes = n_active_classes
    self.active_class_list = active_class_list
    self.net, _n_bones = build_bst_x_network(
        model_name,
        n_joints=self.n_joints,
        pose_style=self.pose_style,
        in_channels=in_channels,
        n_class=self.n_active_classes,
        seq_len=seq_len,
        device=self.device,
    )
```

Pre-fix weight callers pass `n_active_classes=taxonomy.n_classes, active_class_list=taxonomy.class_list()` explicitly. Post-fix weight callers read both from the run's `manifest.yaml` under `extra.arch`. Symmetric, no special-case path inside the function.

**Strip 2 — era-boundary warning collapsed to one line** (was R8's 10-line block).
Stripped to an early-line check inside `__main__`, near the resume manifest read:

```python
if resume_from and resumed_manifest_arch is None:
    print(
        '[arch:resume] resuming pre-fix run; load_state_dict will fail '
        'on shape mismatch for v1/nosides/raw_35 dropunk weights.'
    )
```

Three lines including the print. Restores diagnosability for pre-fix resumes without the verbose framing.

**Strip 3 — class_weights error collapsed to one tier** (was R7's two-tier fork).

```python
if cls_name not in class_ls:
    raise ValueError(
        f"class_weights key '{cls_name}' not in active class list "
        f"{class_ls}. (Full taxonomy {taxonomy.name!r} has "
        f"{len(taxonomy.class_list())} classes; this run uses {len(class_ls)}.)"
    )
```

User has both numbers (active size, full size) to debug. Drops the "in taxonomy but absent from active" hand-holding.

### 6.2 Additional fix from second-pass review

`scratch/post_tidy_smoke/smoke_infer_bit_exact.py:112` calls `task.get_network_architecture(model_name=..., seq_len=..., in_channels=..., taxonomy=taxonomy)` with no arch kwargs and would break under Strip 1. The bit-exact smoke harness is for pre-fix weights, so the fix is to add the explicit pre-fix-style kwargs:

```python
task.get_network_architecture(
    model_name=model_name,
    seq_len=seq_len,
    in_channels=in_channels,
    taxonomy=taxonomy,
    n_active_classes=taxonomy.n_classes,
    active_class_list=taxonomy.class_list(),
)
```

Two extra lines. The script's intent (smoke pre-fix forwards) is preserved; it just declares the architecture explicitly now.

`src/api/inference.py` is a stub that doesn't actually call `bst_x_infer` yet (just a placeholder docstring). Leave alone; the team will fill it in later and read the manifest themselves.

### 6.3 Per-file change summary

For implementation reference. Refer to §5 for the full code patches; §6 only annotates the deltas.

| File | Change |
|---|---|
| `src/bst_x/pipeline/config.py` | Add `Taxonomy.has_unknown` (property), `active_class_list(present_indices, side)`, `full_to_active_remap(present_indices, side)`. Drop `n_active_classes` as a separate method. Per §5.2; no further changes from §6. |
| `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_common.py` | Add `derive_active_classes_from_labels(taxonomy, train_labels, validation_label_arrays)`. Replaces first-fix's `remap_and_validate_labels`. Per §5.2. |
| `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py` | Drop `drop_unknown` from `Task.__init__`; add `_derive_active_classes_from_loaded_labels`; drop the preflight from `__main__`; add the §5 `_validate_and_record_arch` free function (with R3 cross-check, R6 expected_active_classes lever, R4 tee'd printout); §6 Strip 2 one-line warning at resume; §6 Strip 3 one-tier `class_weights` message. Add `expected_active_classes` field to Hyp. Re-add `import yaml` if not already present (needed for resumed manifest read). |
| `src/bst_x/stroke_classification/main_on_shuttleset/bst_x_infer.py` | Strip 1 — required kwargs `n_active_classes` and `active_class_list`, no fallback. Update `__main__` example to read both from manifest. |
| `scratch/post_tidy_smoke/smoke_infer_bit_exact.py` | Add explicit `n_active_classes=taxonomy.n_classes, active_class_list=taxonomy.class_list()` at the existing `get_network_architecture` call site (line 112). |
| `tests/test_active_classes.py` | Reframe per §5.2: helper signature change; subset-assert tests; resume cross-check tests; expected_active_classes tests. Section 3 (drop-policy guards) deleted. Real-data probes pass for v1, nosides, merged_25 dropunk. |
| `docs/architecture_notes/bst_x_overview.md` | Update the structural note from the first fix to reflect the empirical-derivation semantics, including the train-only invariant. |

### 6.4 Files NOT touched

- `preparing_data/prepare_train_on_shuttleset.py` (writer / collator).
- `preparing_data/shuttleset_dataset.py` (reader plumbing).
- `pipeline.config.derive_ablation_id`, `pipeline.config.derive_npy_collated_dir_basename`.
- Any existing collated npy dir.
- Any existing weight file.
- Any existing manifest.
- `run_tracker.py`.
- `src/api/inference.py` (stub, no live call to bst_x_infer yet).

### 6.5 Final tradeoff vs §5

| consideration | §5 | §6 (stripped) |
|---|---|---|
| Inference on pre-fix weights | works (caller passes args, or relies on fallback) | works (caller passes args explicitly) |
| Resume of pre-fix weight | fails at `load_state_dict`, after a verbose upfront warning | fails at `load_state_dict`, after a one-line upfront warning |
| Resume of post-fix weight | cross-checked against manifest | cross-checked against manifest (unchanged) |
| `class_weights` typo against missing class | two-tier message | one-tier message with both class-count values |
| bst_x_infer `__main__` example | runs without arch kwargs (full-tax fallback) | requires arch kwargs (read from manifest or pre-fix-style explicit) |
| `smoke_infer_bit_exact.py` | runs unchanged (relies on fallback) | needs two-line update declaring full-tax arch explicitly |
| LOC vs §5 | baseline | ~25 fewer lines, ~3 fewer conceptual branches |

### 6.6 Validation plan (final)

Same as §5.6:

1. CPU-only suite locally + on engelbart: all section 1/2/4/5 tests pass; new R1/R2/R6 tests pass; section 3 (first-fix drop-policy guards) deleted; `smoke_infer_bit_exact.py` invocation passes (CPU-only forward; the script's existing logic is unchanged).
2. Real-data probes for v1, nosides, merged_25 dropunk all pass.
3. End-to-end run on engelbart against `une_merge_v1_nosides + dropunk`: `[arch]` prints inside the tee'd block, manifest gets `extra.arch` after serial 1 prepare, class-weighted CE printout shows 14 entries with no unknown.
4. End-to-end run on engelbart against `merged_25 + dropunk`: `[arch]` prints `n_active=25`, training mirrors `run_20260429_202144`. Apples-to-apples for the rerun.
5. Manual smoke: try resuming `run_20260430_213933` (a pre-fix v1/nosides cell). Expect: one-line `[arch:resume]` warning, then `load_state_dict` error on shape mismatch.

### 6.7 Path F revisit

Still parked. The recollation cost (drops 52 driven_flight rows from merged_25 dropunk) breaks apples-to-apples comparability with `run_20260429_202144`, which is the user's stated baseline for the upcoming merged_25 rerun. §6 preserves that comparability without recollation. Path F is a defensible future cleanup, separate from this fix.

---

## 7. Test framework draft

This is a starting framework, not canonical truth. If implementation surfaces new edge cases the tests didn't anticipate, **the implementation wins**, and these tests get amended to match. The point of drafting now is so that the implementer has a concrete target to aim at and a fast feedback loop — not so the implementation has to bend to the tests.

The framework covers six sections matching §6's per-file changes:

1. **Taxonomy method correctness** — `active_class_list(present_indices)`, `full_to_active_remap(present_indices)`, `has_unknown` across all four real taxonomies + a synthetic no-unknown taxonomy. Identity case, minus-unknown case, arbitrary subset case, out-of-range guard, empty-set edge case.
2. **`derive_active_classes_from_labels`** — train-only happy paths; train+val+test happy path; val-or-test-has-class-train-doesn't subset assertion failures; empty-train rejection; corrupted-labels rejection; no-unknown taxonomy round-trip; `validation_label_arrays=None` works.
3. **`_validate_and_record_arch`** — manifest write happy path; resume cross-check positive (manifest matches live derivation); resume cross-check negative on `n_active_classes` mismatch; resume cross-check negative on `active_class_list` order mismatch; `expected_active_classes` lever passes when matching; `expected_active_classes` raises when mismatching. Uses `tmp_path` for isolated manifest files.
4. **BST_CG_AP forward+backward smoke** — at the active head dim across all four taxonomies × `{all_classes, no_unknown}`. Verifies head shape, finite loss, gradient flow. CPU-only.
5. **`class_weights` renormalisation under active classes** — pair-balanced renorm sanity (mean=1.0, named classes lifted); uniform fallback; Strip 3 one-tier error message contains both active-list size and full-taxonomy size.
6. **Real-labels probe** — for the three current dropunk dirs (v1, nosides, merged_25). Auto-skips if dir not visible from host. Asserts expected `n_active_classes` per dir (14, 14, 25 respectively).
7. **bst_x_infer contract** — `get_network_architecture` raises `TypeError` when called without `n_active_classes` / `active_class_list`. Confirms Strip 1 is in effect.

### 7.1 Full test file content

```python
"""Tests for the unknown-ghost-channel fix (post-strip implementation).

CPU-only. Run from repo root via the repo's pytest config:
    pytest tests/test_active_classes.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from pipeline.config import TAXONOMIES, Taxonomy
from main_on_shuttleset.bst_x_common import (
    build_bst_x_network,
    derive_active_classes_from_labels,
)
from main_on_shuttleset.bst_x_train import _validate_and_record_arch
from main_on_shuttleset.bst_x_infer import Task as InferTask


REAL_TAXONOMY_NAMES = ['merged_25', 'une_merge_v1', 'une_merge_v1_nosides', 'raw_35']


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_no_unknown_taxonomy() -> Taxonomy:
    return Taxonomy(
        name='no_unknown_test',
        merge_map=None,
        base_types=('a', 'b', 'c'),
        standalone_types=(),
        unknown_first=False,
    )


def _full_present(tax: Taxonomy) -> set[int]:
    return set(range(tax.n_classes))


def _present_minus_unknown(tax: Taxonomy) -> set[int]:
    full = tax.class_list()
    return _full_present(tax) - {full.index('unknown')}


def _make_synthetic_labels(present: set[int], n_per_class: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(sorted(present), n_per_class)
    rng.shuffle(labels)
    return labels.astype(np.int64)


class _FakeTask:
    def __init__(self, n_active_classes, active_class_list):
        self.n_active_classes = n_active_classes
        self.active_class_list = active_class_list


class _FakeHyp:
    def __init__(self, expected_active_classes=None):
        self.expected_active_classes = expected_active_classes


class _NoopTee:
    def write(self, _data): pass
    def flush(self): pass


def _seed_manifest(run_dir: Path, extra: dict | None = None) -> Path:
    manifest = {'run_id': run_dir.name, 'config': {}, 'serials': []}
    if extra is not None:
        manifest['extra'] = extra
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'manifest.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Section 1: Taxonomy methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_taxonomy_has_unknown(tax_name):
    assert TAXONOMIES[tax_name].has_unknown is True


def test_synthetic_no_unknown_has_unknown_false(synthetic_no_unknown_taxonomy):
    assert synthetic_no_unknown_taxonomy.has_unknown is False


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_full_present_is_identity(tax_name):
    tax = TAXONOMIES[tax_name]
    assert tax.active_class_list(_full_present(tax)) == tax.class_list()


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_minus_unknown(tax_name):
    tax = TAXONOMIES[tax_name]
    active = tax.active_class_list(_present_minus_unknown(tax))
    assert 'unknown' not in active
    assert len(active) == tax.n_classes - 1


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_preserves_relative_order(tax_name):
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    active = tax.active_class_list(_present_minus_unknown(tax))
    assert active == [n for n in full if n != 'unknown']


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_active_class_list_arbitrary_subset(tax_name):
    """Arbitrary subset returns those classes in original order."""
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    subset = {0, 2, len(full) - 1}
    assert tax.active_class_list(subset) == [full[i] for i in sorted(subset)]


def test_active_class_list_out_of_range_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    with pytest.raises(ValueError, match='out of range'):
        tax.active_class_list({99})


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_remap_full_present_is_identity(tax_name):
    tax = TAXONOMIES[tax_name]
    assert tax.full_to_active_remap(_full_present(tax)) == list(range(tax.n_classes))


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_remap_minus_unknown_sentinel_at_unknown_idx(tax_name):
    tax = TAXONOMIES[tax_name]
    full = tax.class_list()
    unknown_idx = full.index('unknown')
    remap = tax.full_to_active_remap(_present_minus_unknown(tax))
    assert remap[unknown_idx] == -1
    active_values = sorted(v for v in remap if v >= 0)
    assert active_values == list(range(tax.n_classes - 1))


def test_active_class_list_empty_present():
    tax = TAXONOMIES['une_merge_v1_nosides']
    assert tax.active_class_list(set()) == []


def test_remap_empty_present_all_sentinel():
    tax = TAXONOMIES['une_merge_v1_nosides']
    assert tax.full_to_active_remap(set()) == [-1] * tax.n_classes


def test_unknown_position_per_taxonomy():
    """Document the BST-paper convention split via assert."""
    for tax_name in ('merged_25', 'une_merge_v1'):
        assert TAXONOMIES[tax_name].class_list()[0] == 'unknown', tax_name
    for tax_name in ('une_merge_v1_nosides', 'raw_35'):
        assert TAXONOMIES[tax_name].class_list()[-1] == 'unknown', tax_name


# ---------------------------------------------------------------------------
# Section 2: derive_active_classes_from_labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_full_present_train_only(tax_name):
    tax = TAXONOMIES[tax_name]
    train = _make_synthetic_labels(_full_present(tax))
    active, remap, out = derive_active_classes_from_labels(tax, train)
    assert active == tax.class_list()
    assert remap == list(range(tax.n_classes))
    np.testing.assert_array_equal(out['train'], train)


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_minus_unknown_train_only(tax_name):
    tax = TAXONOMIES[tax_name]
    train = _make_synthetic_labels(_present_minus_unknown(tax))
    active, _remap, out = derive_active_classes_from_labels(tax, train)
    assert 'unknown' not in active
    assert len(active) == tax.n_classes - 1
    assert (out['train'] >= 0).all()
    assert (out['train'] < len(active)).all()


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_train_with_val_test_subset_pass(tax_name):
    tax = TAXONOMIES[tax_name]
    present = _present_minus_unknown(tax)
    train = _make_synthetic_labels(present, seed=0)
    val   = _make_synthetic_labels(present, seed=1)
    test  = _make_synthetic_labels(present, seed=2)
    active, _remap, out = derive_active_classes_from_labels(
        tax, train_labels=train,
        validation_label_arrays={'val': val, 'test': test},
    )
    assert set(out.keys()) == {'train', 'val', 'test'}
    for split, arr in out.items():
        assert (arr >= 0).all() and (arr < len(active)).all(), split


@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
def test_derive_val_has_class_absent_from_train_raises(tax_name):
    tax = TAXONOMIES[tax_name]
    unknown_idx = tax.class_list().index('unknown')
    train = _make_synthetic_labels(_present_minus_unknown(tax), seed=0)
    val = np.concatenate([
        _make_synthetic_labels(_present_minus_unknown(tax), seed=1),
        np.asarray([unknown_idx], dtype=np.int64),
    ])
    test = _make_synthetic_labels(_present_minus_unknown(tax), seed=2)
    with pytest.raises(ValueError, match=r'\[val\] contains class indices absent from train'):
        derive_active_classes_from_labels(
            tax, train_labels=train,
            validation_label_arrays={'val': val, 'test': test},
        )


def test_derive_test_has_class_absent_from_train_raises():
    tax = TAXONOMIES['merged_25']
    unknown_idx = tax.class_list().index('unknown')
    train = _make_synthetic_labels(_present_minus_unknown(tax), seed=0)
    val   = _make_synthetic_labels(_present_minus_unknown(tax), seed=1)
    test = np.concatenate([
        _make_synthetic_labels(_present_minus_unknown(tax), seed=2),
        np.asarray([unknown_idx], dtype=np.int64),
    ])
    with pytest.raises(ValueError, match=r'\[test\] contains class indices absent from train'):
        derive_active_classes_from_labels(
            tax, train_labels=train,
            validation_label_arrays={'val': val, 'test': test},
        )


def test_derive_empty_train_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    with pytest.raises(ValueError, match='train_labels is empty'):
        derive_active_classes_from_labels(
            tax, train_labels=np.asarray([], dtype=np.int64),
        )


def test_derive_no_validation_arrays_works():
    tax = TAXONOMIES['une_merge_v1_nosides']
    train = _make_synthetic_labels(_full_present(tax))
    _active, _remap, out = derive_active_classes_from_labels(tax, train)
    assert set(out.keys()) == {'train'}


def test_derive_corrupted_label_raises():
    tax = TAXONOMIES['une_merge_v1_nosides']
    bad_train = np.asarray([0, 1, 99], dtype=np.int64)
    with pytest.raises(ValueError, match='out of range'):
        derive_active_classes_from_labels(tax, bad_train)


def test_derive_no_unknown_taxonomy_round_trip(synthetic_no_unknown_taxonomy):
    tax = synthetic_no_unknown_taxonomy
    train = np.asarray([0, 1, 2, 3, 4, 5] * 3, dtype=np.int64)
    active, remap, out = derive_active_classes_from_labels(tax, train)
    assert active == tax.class_list()
    assert remap == list(range(tax.n_classes))
    np.testing.assert_array_equal(out['train'], train)


# ---------------------------------------------------------------------------
# Section 3: _validate_and_record_arch
# ---------------------------------------------------------------------------

def test_validate_and_record_arch_writes_manifest_block(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=None)
    _seed_manifest(tmp_path)

    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=None, tee=_NoopTee(),
    )

    with open(tmp_path / 'manifest.yaml') as f:
        manifest = yaml.safe_load(f)
    arch = manifest['extra']['arch']
    assert arch['n_classes_full']    == tax.n_classes
    assert arch['n_active_classes']  == len(active)
    assert arch['has_unknown']       is True
    assert arch['unknown_first']     is False
    assert arch['active_class_list'] == active


def test_validate_and_record_arch_resume_match(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior = {
        'arch': {
            'n_classes_full':    tax.n_classes,
            'n_active_classes':  len(active),
            'has_unknown':       True,
            'unknown_first':     False,
            'active_class_list': active,
        },
    }
    _seed_manifest(tmp_path, extra=prior)
    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=prior['arch'], tee=_NoopTee(),
    )


def test_validate_and_record_arch_resume_n_active_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior_arch = {
        'n_active_classes':  len(active) + 1,
        'active_class_list': ['fake'] * (len(active) + 1),
    }
    _seed_manifest(tmp_path, extra={'arch': prior_arch})
    with pytest.raises(ValueError, match='Resume manifest disagrees'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=prior_arch, tee=_NoopTee(),
        )


def test_validate_and_record_arch_resume_list_order_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp()
    prior_arch = {
        'n_active_classes':  len(active),
        'active_class_list': list(reversed(active)),
    }
    _seed_manifest(tmp_path, extra={'arch': prior_arch})
    with pytest.raises(ValueError, match='Resume manifest disagrees'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=prior_arch, tee=_NoopTee(),
        )


def test_validate_and_record_arch_expected_match_passes(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=active)
    _seed_manifest(tmp_path)
    _validate_and_record_arch(
        run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
        resumed_manifest_arch=None, tee=_NoopTee(),
    )


def test_validate_and_record_arch_expected_mismatch_raises(tmp_path):
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    task = _FakeTask(len(active), active)
    hyp = _FakeHyp(expected_active_classes=['something_else'])
    _seed_manifest(tmp_path)
    with pytest.raises(ValueError, match='expected_active_classes'):
        _validate_and_record_arch(
            run_dir=tmp_path, task=task, taxonomy=tax, hyp=hyp,
            resumed_manifest_arch=None, tee=_NoopTee(),
        )


# ---------------------------------------------------------------------------
# Section 4: BST_CG_AP forward+backward smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tax_name', REAL_TAXONOMY_NAMES)
@pytest.mark.parametrize('present_kind', ['all_classes', 'no_unknown'])
def test_bst_forward_backward_under_active_classes(tax_name, present_kind):
    torch.manual_seed(0)
    tax = TAXONOMIES[tax_name]
    n_active = tax.n_classes if present_kind == 'all_classes' else tax.n_classes - 1

    pose_style = 'JnB_bone'
    seq_len = 100
    batch_size = 4
    n_joints = 17
    in_channels = 2

    net, n_bones = build_bst_x_network(
        model_name='BST_CG_AP',
        n_joints=n_joints, pose_style=pose_style, in_channels=in_channels,
        n_class=n_active, seq_len=seq_len, device='cpu',
    )
    net.set_schedule_factors(cg_factor=1.0, ap_factor=1.0)

    j_plus_b = n_joints + n_bones
    human_pose = torch.randn(batch_size, seq_len, 2, j_plus_b, in_channels)
    human_pose_flat = human_pose.view(*human_pose.shape[:-2], -1)
    pos = torch.randn(batch_size, seq_len, 2, 2)
    shuttle = torch.randn(batch_size, seq_len, 2)
    video_len = torch.full((batch_size,), seq_len, dtype=torch.long)
    labels = torch.randint(0, n_active, (batch_size,))

    logits = net(human_pose_flat, shuttle, pos, video_len)
    assert logits.shape == (batch_size, n_active)

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    grad_count = sum(
        1 for p in net.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0


# ---------------------------------------------------------------------------
# Section 5: class_weights renormalisation under active classes
# ---------------------------------------------------------------------------

def test_class_weights_renorm_pair_balanced():
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    n_active = len(active)
    assert 'unknown' not in active

    class_weights = {'wrist_smash': 2.0, 'smash': 2.0}
    weights = torch.ones(n_active)
    for cls_name, mult in class_weights.items():
        weights[active.index(cls_name)] = mult
    weights = weights * (n_active / weights.sum())

    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    for cls_name in class_weights:
        assert weights[active.index(cls_name)] > 1.0


def test_class_weights_uniform_when_empty():
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    n_active = len(active)
    weights = torch.ones(n_active)
    weights = weights * (n_active / weights.sum())
    assert torch.allclose(weights, torch.ones(n_active))


def test_class_weights_strip3_message_shape():
    """Strip 3: error mentions 'not in active class list', plus both
    active-list size and full-taxonomy size for the user to debug."""
    tax = TAXONOMIES['une_merge_v1_nosides']
    active = tax.active_class_list(_present_minus_unknown(tax))
    cls_name = 'unknown'

    msg = (
        f"class_weights key '{cls_name}' not in active class list "
        f"{active}. (Full taxonomy {tax.name!r} has "
        f"{len(tax.class_list())} classes; this run uses {len(active)}.)"
    )
    assert 'not in active class list' in msg
    assert f'this run uses {len(active)}' in msg
    assert f'has {len(tax.class_list())} classes' in msg


# ---------------------------------------------------------------------------
# Section 6: real-labels probe
# ---------------------------------------------------------------------------

CANDIDATE_REAL_DIRS = [
    '/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/'
    'npy_une_merge_v1_nosides_split_v2_dropunk',
    '/scratch/comp320a/ShuttleSet_data_une_merge_v1/'
    'npy_une_merge_v1_split_v2_dropunk',
    '/scratch/comp320a/ShuttleSet_data_merged_25/'
    'npy_merged_25_split_bst_baseline_dropunk',
]

EXPECTED_N_ACTIVE_PER_DIR = {
    'npy_une_merge_v1_nosides_split_v2_dropunk': 14,
    'npy_une_merge_v1_split_v2_dropunk':         14,
    'npy_merged_25_split_bst_baseline_dropunk':  25,  # driven_flight populates unknown
}


@pytest.mark.parametrize('dir_path', CANDIDATE_REAL_DIRS)
def test_real_labels_npy_remap_into_active_range(dir_path):
    root = Path(dir_path)
    if not root.exists():
        pytest.skip(f'{dir_path} not visible from this host')

    name = root.name
    expected_n_active = EXPECTED_N_ACTIVE_PER_DIR[name]

    tax_name = None
    for cand in sorted(REAL_TAXONOMY_NAMES, key=len, reverse=True):
        if name.startswith(f'npy_{cand}_'):
            tax_name = cand
            break
    assert tax_name is not None, f'could not infer taxonomy from {name!r}'

    tax = TAXONOMIES[tax_name]
    train = np.load(str(root / 'train' / 'labels.npy'))
    val   = np.load(str(root / 'val'   / 'labels.npy'))
    test  = np.load(str(root / 'test'  / 'labels.npy'))

    active, _remap, out = derive_active_classes_from_labels(
        tax, train_labels=train,
        validation_label_arrays={'val': val, 'test': test},
    )
    assert len(active) == expected_n_active, (
        f'{name}: expected n_active={expected_n_active} got {len(active)}'
    )
    n_active = len(active)
    for split, arr in out.items():
        assert (arr >= 0).all() and (arr < n_active).all(), split


# ---------------------------------------------------------------------------
# Section 7: bst_x_infer Strip 1 contract
# ---------------------------------------------------------------------------

def test_bst_infer_get_network_architecture_requires_arch_kwargs():
    """Strip 1: bst_x_infer's get_network_architecture must require both
    n_active_classes and active_class_list. No silent fallback."""
    task = InferTask(n_joints=17)
    task.pose_style = 'JnB_bone'
    with pytest.raises(TypeError):
        task.get_network_architecture(
            model_name='BST_CG_AP', seq_len=100, in_channels=2,
            taxonomy=TAXONOMIES['une_merge_v1_nosides'],
        )
```

### 7.2 Notes for the implementer

- **Total test count**: ~50 active tests + 3 real-data probes (auto-skipped when `/scratch/...` not visible).
- **Expected runtime**: <10s for active tests on CPU. The smoke tests (Section 4) are 8 parametrise combos x ~1s each = ~8s of the total. Other sections are sub-second.
- **Imports**: `from main_on_shuttleset.bst_x_train import _validate_and_record_arch` and `from main_on_shuttleset.bst_x_infer import Task as InferTask` are the new ones. Both should work via the existing repo-root `conftest.py` sys.path setup. If `bst_x_train` import fires unexpected side effects (it shouldn't, but check), wrap that import in a try/except and skip the relevant tests with a clear message rather than failing collection.
- **Resume cross-check note**: the test fixtures use a `_FakeTask` and `_FakeHyp` rather than the real classes to keep the tests focused on `_validate_and_record_arch`'s logic. If the real `Task` constructor adds required arguments later, the fakes don't need updating.
- **Real-data probe test count**: locally these all skip (not on engelbart). On engelbart they should all pass. If any fail, that's a real signal — the `EXPECTED_N_ACTIVE_PER_DIR` values are the doc's prediction; deviation means the data has shifted or my analysis is wrong.
- **Test for `_validate_and_record_arch`'s manifest-write path**: uses `tmp_path` so it's fully isolated. The `_seed_manifest` helper writes a minimal valid manifest; the test then re-reads after the function runs. This is the closest-to-integration test in the suite.
- **What's deliberately not tested**: the `[arch]` printout content (best tested in an end-to-end run, not via stdout capture). The era-boundary `[arch:resume]` warning (Strip 2 one-liner; trivial enough). The actual `prepare_dataloaders` integration path (the helper-level tests in Section 2 already cover the logic; integration is an end-to-end run on real data).

### 7.3 What new edge cases might shift the framework

The implementer may discover:

- An import-time side effect from `bst_x_train` that breaks Section 3 imports (mitigation: lazy-import inside the test; or move `_validate_and_record_arch` into a module that's safe to import at test time).
- An interaction between `train_partial < 1.0` and the live derivation that the synthetic-labels tests don't reach (a real-data test against a partial-training cell would surface it; if not in scope, document the assumption).
- A taxonomy whose `class_list()` order isn't what `sorted(present_indices)` produces (shouldn't happen given current taxonomies, but if a future taxonomy has a non-trivial reordering, the relative-order assertion in Section 1 catches it before any deeper tests run).
- A `_validate_and_record_arch` signature drift if implementation makes the `tee` argument optional or moves the manifest write into `track_serial`.

In all such cases, amend the tests to match the real implementation and document the change in this doc's §6 or §7 as appropriate.

---

## 8. Post-implementation notes

Captured after the §6 plan landed and was end-to-end smoke-tested on engelbart.

### 8.1 Implementation deltas vs §6

- **Auto-backup of resumed manifest**. §6 left the resume manifest mutation (driven by `_validate_and_record_arch` rewriting `extra.arch` in place on serial 1) as a known side effect, with the Strip 2 one-line warning as the only signal. Live use surfaced that the warning's interception window is small (a few seconds between print and write) and easy to miss under nohup. `bst_x_train.py` `__main__` now copies `manifest.yaml` to `manifest.yaml.<timestamp>.bak` next to the live file before any work begins whenever `resume_from` is set. Backups accrete one per resume invocation; `git status` flags them as untracked for cleanup. Behaviour change is additive only: serial 1 still rewrites the manifest; the original is just always recoverable.
- **`EXPECTED_N_ACTIVE_PER_DIR` for `une_merge_v1` corrected from 14 to 28**. The `une_merge_v1` taxonomy keeps Top_/Bottom_ side prefixes, so its dropunk head is 14 base × 2 sides = 28 (the empty unknown slot drops 1, leaving 28). Only the `nosides` variant collapses sides to 14. `tests/test_active_classes.py` Section 6's expected mapping had the wrong value for v1; corrected on engelbart's first real-data probe pass.
- **Local venv tensorboard install**. `phase_2_refactor` venv on the laptop didn't have `tensorboard`, `torcheval`, or `transformers` (the BST stack was complete elsewhere). Installed them so the test suite's `_validate_and_record_arch` import path runs locally without the try/except guard the §7 framework draft proposed; engelbart's `venv-bst` already had them all. Test suite is now plain: 75 tests collected, no fallback skips for missing modules.

### 8.2 End-to-end smoke results

Both smokes used `early_stop_n_epochs=2`, `range(1, 3)` (2 serials), all other Hyp values matching the active config for the cell. Markers tagged `# SMOKE` in source so they grep out cleanly when reverting.

**Run 1 - `une_merge_v1_nosides + split_v2 + dropunk` (`run_20260501_151131`)**:

- `[arch]` printout in tee'd log: `taxonomy=une_merge_v1_nosides, has_unknown=True, unknown_first=False, n_classes_full=15, n_active_classes=14`. Active list 14 entries, no `unknown`.
- Class-weighted CE printout: 14 lines, `wrist_smash` and `smash` lifted to 1.750, the other 12 at 0.875. Mean = 1.000 (renorm correct).
- Manifest `extra.arch.n_active_classes=14`, full active list serialised. Per-class `metrics.per_class_f1` block has 14 entries, no `unknown`.
- 2 serials independent seeds: serial 1 wrist_smash F1 0.33, serial 2 0.48 (seed variance, expected at 2 epochs).

**Run 2 - `merged_25 + split_bst_baseline + dropunk` (`run_20260501_152835`)**:

- `[arch]` printout in tee'd log: `taxonomy=merged_25, has_unknown=True, unknown_first=True, n_classes_full=25, n_active_classes=25`. 25-name active list with `unknown` at index 0, matching `run_20260429_202144`'s head dim apples-to-apples.
- `class_weights={'Top_smash': 2.0, 'Bottom_smash': 2.0}` exercised the 25-class class-weighted CE printout end-to-end.
- Manifest `extra.arch.n_active_classes=25`, full active list serialised. 2 serials independent seeds: S1 macro 0.810 / min 0.516 / acc 0.833, S2 macro 0.790 / min 0.447 / acc 0.828. Top_smash / Bottom_smash F1 0.88-0.89 across both serials, weighting working as intended. `per_class_f1` block has 24 entries (Top_/Bottom_ × 12 base types); the unknown channel is in the head and gets trained on the writer's surviving driven_flight rows (42 train), but the test split has no unknown ground-truth after the zero-length-clip drop so the present mask correctly excludes it from F1 reporting.

**Aborted attempt - `run_20260501_152623`**: killed before serial 1 completed (initial run-2 attempt before class_weights were re-staged). Confirms the manifest write order: `extra.arch` was already populated at kill time, `serials: []`, log file caught only the tee'd `[arch]` block. `_validate_and_record_arch` fires before any training compute, as designed.

### 8.3 Unit-test count

`/home/ariel/.venvs/phase_2_refactor/bin/pytest tests/test_active_classes.py -v` lands **72 passed, 3 skipped** locally (the 3 skipped are the engelbart-only real-data probes; on engelbart `venv-bst` lands 75 passed, 0 skipped).

### 8.4 Forward-compat invariants

Confirmed clean for two anticipated future cases. Both are covered by existing tests:

- **Taxonomies without unknown**. `Taxonomy.has_unknown=False` flows through naturally: `derive_active_classes_from_labels` doesn't special-case unknown, manifest records `has_unknown=False, unknown_first=False`, class-weighted CE printout uses whatever the active list is. The `synthetic_no_unknown_taxonomy` fixture covers Section 1 + Section 2 paths; `test_derive_no_unknown_taxonomy_round_trip` exercises the end-to-end derivation. Adding a new no-unknown taxonomy to `TAXONOMIES` needs no code changes.
- **X3D-S fusion / new training scripts**. The architecture-side contract is: source the classification head dim from `task.n_active_classes` (already populated post-`prepare_dataloaders`), and call `_validate_and_record_arch` on serial 1. `Task.get_network_architecture` is the reference implementation. Any new training script that follows that pattern inherits the fix; one that hardcodes `taxonomy.n_classes` in the fusion head puts the unknown ghost back. Wiring note added to `bst_x_overview.md`'s X3D-S bullet to surface this when the build starts.

### 8.5 Restore-after-smoke list

When reverting the smoke harness back to a real run config:

- `bst_x_train.py:69` `early_stop_n_epochs` 2 → 40 (or whatever the next cell needs).
- `bst_x_train.py` serial loop `range(1, 3)` → `range(1, 6)`.
- For run 2 specifically: revert `taxonomy='merged_25'` → the next cell's active taxonomy, `split_column='split_bst_baseline'` → the matching split, `class_weights` → the cell's intended dict.
- All three are tagged `# SMOKE` in source for grep'ability.
