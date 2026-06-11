# Curtis's `data_access.py` Integration Plan

## Context

Curtis has a working branch that tops out at commit
`c726a1895440068413dbecef021ceab16fbcc0e1` ("Use ClipRecord in tests to
verify return type of get_clip_records"). The branch diverges from the
pre-Phase-2 `main` and builds a `pipeline/data_access.py` module that
provides filtered access to clips + shuttle npy + mmpose npy by split
and taxonomy class, with a Python API, a CLI, an interactive TUI, a
`.env` mechanism for path config, and a full test suite.

Scope of his branch vs main (as of the ultimate commit):

```
 .env.example                             |  30 ++
 .gitignore                               |   2 +
 src/bst_x/pipeline/data_access.py | 452 +++++++
 tests/test_data_access.py                | 311 +++++
 4 files changed, 795 insertions(+)
```

Issue addressed on his branch: #56, per commit messages. Co-authored
with Curtis Martin `<86826694+curtislmartin@users.noreply.github.com>`.

### Where this clashes with our Phase 2 work

Curtis's module assumes the **pre-Phase-2 nested layout for all three
data trees**:

```
clips_dir/{split}/{class}/*.mp4
shuttle_npy_dir/{split}/{class}/*.npy
mmpose_npy_dir/{split}/{class}/*_joints.npy + *_pos.npy
```

Split + label are read off the folder names. This is the exact layout
Phase 2 moved *away* from for shuttle + mmpose:

- Shuttle npy is now **flat**: `shuttle_npy_flat/{stem}.npy`
  (Phase 2.2, `shuttle_csvs_to_npy` rewrite).
- MMPose per-clip npy is now **flat**: `dataset_npy_..._flat/{stem}_*.npy`
  (Phase 2.1, pose writers rewrite).
- Clips dir is **still nested** — Phase 3 flattening is deferred.
- Split + taxonomy assignment now come from
  `notebooks/clips_master.csv`, not the folder structure.

So a straight git merge would produce a module that assumes a data
layout that no longer matches reality for two of the three trees, and
reads labels/splits from folder names instead of the canonical CSV.

### What's useful regardless

Curtis's **API shape and CLI design are clean and well-tested**. They're
worth keeping. What needs swapping is the backend that walks the filesystem
and decides split/label — the public `get_clip_records(...)` + `ClipRecord`
+ `DataPaths` interface is perfectly fine.

We also already have `src/bst_x/pipeline/clip_index.py`
(commit `41f3487`, November 2026) which is a one-function subset
(`build_clip_path_index(clips_dir) -> dict[str, Path]`) of what Curtis
built. Ported properly, `data_access.py` could supersede the low-level
helper or complement it.

## Goals

1. **Keep Curtis's API** as the canonical way to ask
   "give me all clips for `split=...` and `taxonomy_class=...` with
    their matched shuttle + mmpose file paths". API is
   `DataPaths`, `ClipRecord`, `get_clip_records`, `summarise`,
   `interactive`, plus the CLI.
2. **Swap the backend** to read split + taxonomy from `clips_master.csv`
   (our post-Phase-2 source of truth) and resolve flat paths for shuttle
   and mmpose.
3. **Preserve Curtis's `.env` pattern**: useful across environments
   (local, engelbart scratch, future team members), and it's
   independent of the layout question.
4. **Port the tests** with fake filesystems matching the current layout
   (nested clips, flat shuttle, flat mmpose).
5. **Absorb or preserve `clip_index.py`** — decide whether to delete the
   stub in favour of `data_access.get_clip_records()`, or keep it as a
   low-level helper that the new `data_access` uses internally.
6. **Preserve co-authorship attribution** on the final commit — Curtis's
   design work should remain credited even if the tree diff doesn't
   match his branch line-for-line.

## Out of scope

- A git merge of Curtis's branch into main. The conflicts with Phase 2
  would be unfixable without rewriting every line of his module anyway,
  so we port by hand rather than fight git.
- Touching `clip_generator.py`, MMPose extraction, or any other pipeline
  producer. The clips tree staying nested keeps Curtis's clips-walk
  logic mostly valid; shuttle and mmpose need flat resolution instead.
- Phase 3 (flattening the clips directory).
- Any change to the current `validate_zeroed_frames.py` /
  `fail_rate_per_class.py` scan logic. Migrating them to use
  `data_access.get_clip_records()` is a reasonable follow-up but
  not part of this integration.

## File inventory

| File on Curtis's branch | Port? | Notes |
|---|---|---|
| `src/bst_x/pipeline/data_access.py` (452 LOC) | **Port with rewrite** of the backend. API + CLI + TUI surface kept; filesystem-walk logic replaced with CSV-driven resolution. |
| `tests/test_data_access.py` (311 LOC) | **Port with fake-fs updates**. Fake filesystems need the new layout (nested clips + flat shuttle + flat mmpose), and the split/class source changes from folder names to a synthetic CSV fixture. |
| `.env.example` (30 LOC) | **Port as-is** with path examples updated for the current `shuttle_npy_flat` + flat mmpose convention. |
| `.gitignore` additions | **Port** (ensures `.env` stays local). Two lines. |

No file deleted on Curtis's branch, so no reverse-ports needed.

On our side (to review during the port):
- `src/bst_x/pipeline/clip_index.py` (62 LOC, committed at `41f3487`) — decide whether to keep, fold into `data_access.py`, or have `data_access.py` call it internally.
- `src/bst_x/pipeline/config.py` — source of `CLIPS_OUTPUT_DIR`, `SHUTTLE_OUTPUT_DIR`, `TAXONOMIES`, `DEFAULT_TAXONOMY`, `Taxonomy`. Curtis imports these already; no change expected.
- `notebooks/clips_master.csv` — the CSV that provides split + raw_type_en + player_side per clip_stem. Curtis's backend will read this instead of walking folders.

## Approach: port with a CSV-driven backend

### Proposed module shape

Keep `DataPaths` and `ClipRecord` essentially unchanged. Rewrite
`get_clip_records(paths, split=None, taxonomy_class=None, taxonomy=None)`
so that it:

1. Reads `clips_master.csv` once (path from `DataPaths`, defaulting to
   `notebooks/clips_master.csv` via an env var like `BST_X_CLIPS_CSV` +
   config default).
2. Accepts a new `split_column` parameter (defaulting to
   `'split_bst_baseline'` for backward compat with Curtis's intent
   around training splits).
3. Filters CSV rows by the `split_column` value (train/val/test) and
   by a derived `taxonomy_class` computed from `raw_type_en` +
   `player_side` via the active taxonomy's `merge_map` +
   `standalone_set`. Same derivation pattern as
   `fail_rate_per_class._derive_labels()` and
   `collate_npy._label_str()`.
4. For each surviving CSV row, resolves:
   - `clip` from `paths.clips_dir / split / folder_name / f'{stem}.mp4'`
     (nested layout, still valid).
   - `shuttle_npy` from `paths.shuttle_npy_dir / f'{stem}.npy'` (flat).
   - `mmpose_joints` / `mmpose_pos` from
     `paths.mmpose_npy_dir / f'{stem}_joints.npy'` /
     `{stem}_pos.npy` (flat).
5. Builds a `ClipRecord` per row. Files that don't exist on disk
   (e.g. mmpose not generated yet, or shuttle missing for a handful of
   clips) resolve to `None` on the record, same as Curtis's current
   behaviour.

The `split`, `taxonomy_class`, and `taxonomy` filter arguments keep
Curtis's signature. Folder names (`Top_smash`, etc.) remain the
class-name contract for the public CLI — same strings the current clips
directory tree uses.

### Signature sketch

```python
@dataclass
class DataPaths:
    clips_dir: Path = field(default_factory=lambda: _env_path('BST_X_CLIPS_DIR', CLIPS_OUTPUT_DIR))
    shuttle_npy_dir: Path = field(default_factory=lambda: _env_path('BST_SHUTTLE_NPY_DIR', SHUTTLE_OUTPUT_DIR))
    mmpose_npy_dir: Path | None = field(default_factory=lambda: _env_path_or_none('BST_X_MMPOSE_NPY_DIR'))
    clips_csv: Path = field(default_factory=lambda: _env_path('BST_X_CLIPS_CSV', REPO_ROOT / 'notebooks' / 'clips_master.csv'))


def get_clip_records(
    paths: DataPaths,
    split: str | None = None,
    taxonomy_class: str | None = None,
    split_column: str = 'split_bst_baseline',
    taxonomy_name: str = DEFAULT_TAXONOMY,
    drop_unknown: bool = False,
) -> list[ClipRecord]: ...
```

Exposed CLI flags mirror the python args, with `--split-column`,
`--taxonomy`, and `--drop-unknown` added to Curtis's existing set
(`--split`, `--class`, `--summary`, `--list-classes`, the three path
overrides). The TUI gains a split-column prompt before the class
prompt, or auto-defaults to `split_bst_baseline` if the user doesn't
care.

### `clip_index.py`: fold or keep

Two paths:

- **Fold**. Replace `build_clip_path_index` with a one-liner on top of
  `get_clip_records(paths, split, taxonomy_class)`:
  `{r.clip.stem: r.clip for r in records}`. One less module. The
  callers (Arch 2 3D CNN, Arch 1 wrist crop, future Datasets) get a
  richer object back from `data_access` anyway.

- **Keep**. `clip_index.py` stays as a zero-dep `pathlib`-only helper;
  `data_access.py` uses it internally. Users who only want stems
  don't have to go through the CSV machinery.

**Recommendation: keep**. `clip_index.py` is 60 lines of pure pathlib
with no CSV dependency — useful as a bootstrap helper for scripts or
Datasets that don't want to parse the master CSV. `data_access.py`
becomes the richer, CSV-aware module on top. Documented relationship
in both modules.

### `.env` mechanism

Port as-is. It's a self-contained loader that reads
`.env` (project root) into `os.environ` only for unset keys, so shell
overrides always win. Curtis's `.env.example` is a good template;
update the commented HPC paths to reflect post-Phase-2 layout:

```
BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips
BST_SHUTTLE_NPY_DIR=/scratch/comp320a/ShuttleSet/shuttle_npy_flat
BST_X_MMPOSE_NPY_DIR=/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat
BST_X_CLIPS_CSV=/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv
```

`.env` goes to `.gitignore` (per Curtis's branch). No secrets — just
local paths — but still treat it as per-developer config.

### Tests

The existing `test_data_access.py` builds fake filesystems in
`tempfile.TemporaryDirectory` and verifies filtering / summarisation.
Port each test with:
- Nested clips dir stays.
- Shuttle dir becomes flat (`shuttle_npy_dir/{stem}.npy`).
- MMPose dir becomes flat (`mmpose_npy_dir/{stem}_joints.npy`,
  `_pos.npy`).
- A synthetic `clips_master.csv` fixture replaces the folder-inferred
  split/class. Test helper to generate it with configurable
  split_column + raw_type_en + player_side per row.

Most existing test names should keep working (filter-by-split,
filter-by-class, summarise counts, etc.) — only the fixture builder
changes.

## Integration points

| Module | Current use | Post-integration |
|---|---|---|
| `pipeline/clip_index.py` | Low-level `{stem: Path}` index, used by nothing in-repo yet (designed for upcoming Arch 2 + Arch 1 wrist crop Datasets) | Unchanged. `data_access.py` calls it internally in `get_clip_records` when building the clips-on-disk side of the record. |
| `pipeline/config.py` | Source of `CLIPS_OUTPUT_DIR`, `SHUTTLE_OUTPUT_DIR`, `TAXONOMIES`, `DEFAULT_TAXONOMY`, `Taxonomy` | No change. `data_access.py` imports these; new `BST_X_CLIPS_CSV` env var resolves via `data_access` only. |
| `preparing_data/prepare_train_on_shuttleset.py::collate_npy` | CSV-driven since Phase 1. Derives labels via `taxonomy.merge_map` + `standalone_set`, resolves flat per-clip files. | Could optionally call `data_access.get_clip_records()` for the clip enumeration step. Not required — it already has its own loop that works. Follow-up simplification, not blocking. |
| `validation_scripts/validate_zeroed_frames.py::scan_clips` | CSV-driven since Phase 2. Iterates `clips_master.csv` directly. | Candidate for `data_access.get_clip_records()` refactor. Follow-up once `data_access` is ported. |
| `validation_scripts/fail_rate_per_class.py::main` | CSV-driven since Phase 2. | Same candidate as above. |
| `notebooks/` analysis scripts | Manual filtering, often bespoke. | After `data_access` ports, `from pipeline.data_access import get_clip_records` becomes the one-liner for "give me these clips". |

## Files to add / modify

| File | Action |
|---|---|
| `src/bst_x/pipeline/data_access.py` | New file. ~500 LOC, same shape as Curtis's but CSV-driven backend. |
| `tests/test_data_access.py` | New file. Port Curtis's tests; update fake filesystem + add synthetic `clips_master.csv` fixture. |
| `.env.example` | New file at repo root. Based on Curtis's with paths updated for post-Phase-2 layout. |
| `.gitignore` | Add `.env`. |
| `src/bst_x/pipeline/README.md` | Add `data_access.py` row to the Module Reference table. Expand the "For Downstream Consumers" section to cover the CSV-aware API alongside the `clip_index.py` helper. |
| `src/bst_x/data_pipeline_to_model_train.md` | Add a reference to `data_access.py` in the Stage 3 video-Dataset subsection (next to `clip_index.py`). |
| `src/bst_x/pipeline/clip_index.py` | Cross-reference `data_access.py` in the module docstring. No code change. |

## Verification

### V1. Test suite passes locally

```bash
cd ~/Documents/COSC594/badminton_stroke_classification
pytest tests/test_data_access.py -v
```

Expected: all ported tests pass against fake fixtures. No network / HPC
needed.

### V2. API parity with the existing CSV-driven scripts

Run `get_clip_records` with `split='train'`,
`split_column='split_bst_baseline'`, `taxonomy_name='une_merge_v1'`,
`drop_unknown=True` and confirm the count matches V3 train totals
(24,866 clips per
`experiments/run_20260420_141629/manifest.yaml.extra.data_provenance`).
Same for val (4,000) and test (3,337).

Similarly for `split_column='split_v2'` against V4 totals (22,743 /
5,250 / 4,210).

### V3. CLI sanity on engelbart

```bash
cd ~/badminton_stroke_classifier
# Set .env first (or pass --clips-dir etc.)

python -m pipeline.data_access --summary
python -m pipeline.data_access --split train --summary
python -m pipeline.data_access --split val --class Top_smash
python -m pipeline.data_access --list-classes
python -m pipeline.data_access  # TUI
```

Expected: table matches `clips_master.csv` counts. Running without any
flag drops into the TUI. `--list-classes` returns the 29-element
une_merge_v1 class list (plus any historical extras if the clips dir
has unmerged folders).

### V4. Cross-check against the existing Phase 1 clip_index helper

```bash
python scripts/test_clip_index.py
```

Expected: still passes as-is (the helper's contract is unchanged). If
we fold `clip_index.py` into `data_access.py`, update this script too.
If we keep the split, the test stays valid.

### V5. Downstream smoke

Import `data_access` into an Arch 2 / Arch 1 Dataset skeleton:

```python
from pipeline.data_access import get_clip_records, DataPaths

records = get_clip_records(
    DataPaths(),
    split='train',
    split_column='split_v2',
    taxonomy_name='une_merge_v1',
    drop_unknown=True,
)
assert all(r.clip.exists() for r in records)
assert sum(r.shuttle_npy is None for r in records) < 10  # small missing-shuttle tail OK
```

## Decision gate (should we merge at all?)

The work is worth integrating **if** at least one of the following is
true:

1. Multiple teammates + I will benefit from a shared CLI / TUI for
   ad-hoc "give me these clips" queries. Currently each of us hand-rolls
   this logic. One canonical script across the team reduces drift.
2. The upcoming Arch 2 3D CNN + Arch 1 wrist-crop Datasets benefit
   from a higher-level helper than `clip_index.build_clip_path_index`
   alone. `get_clip_records` returning triples of (clip, shuttle,
   mmpose) is exactly what both architectures need.
3. We want to formalize the path-config-per-environment story. `.env`
   is the cheapest mechanism for this that doesn't require every
   script to duplicate argparse for path overrides.

My read on all three: **yes, yes, yes**. The only real cost is the port
itself, which I estimate at 2-4 hours for the main module + tests +
doc updates.

If two of the three don't hold in your view, skip the port. Keep
`clip_index.py` as the minimal helper and leave `data_access.py`
un-ported. Curtis's branch stays as a reference we can come back to.

## Risks and rollback

| Risk | Mitigation |
|---|---|
| Porting loses subtle behaviour Curtis's tests verify | Port the tests first against the new backend; if any fail, understand why before skipping or modifying. |
| `.env` introduces a new config pathway that confuses teammates used to pure CLI flags | `.env.example` is documented; the precedence order (CLI > shell env > .env > defaults) is explicit; none of the existing CLIs break — `.env` is purely additive. |
| `clip_index.py` and `data_access.py` evolve into redundant APIs | Document the "`clip_index` = one-liner pathlib helper, `data_access` = CSV-aware richer API" split explicitly in both modules' docstrings. |
| A follow-up changes `DataPaths` / `ClipRecord` fields and breaks Curtis's original API | Treat the `data_access` public API as stable once ported. Any later field addition goes on the end of the dataclass / NamedTuple with a default so downstream code doesn't break. |
| Tests using tempdirs can't load `pipeline.config` without repo path setup | Port Curtis's test fixtures to use the same `sys.path.insert` pattern as `validate_zeroed_frames.py` and `fail_rate_per_class.py`. |

**Rollback**: `git revert` the port commit. No on-disk data changes;
`.env` files stay local.

## Credit

Preserve Curtis's authorship on the port commit via a `Co-authored-by:`
trailer in the commit message:

```
Co-authored-by: Curtis Martin <86826694+curtislmartin@users.noreply.github.com>
```

Reference his branch's ultimate commit in the commit body:

> Ported from Curtis's branch ending at c726a18 ("Use ClipRecord in
> tests to verify return type of get_clip_records") and adapted to the
> post-Phase-2 flat shuttle + flat mmpose layout. Original folder-walk
> backend replaced with CSV-driven resolution against
> notebooks/clips_master.csv. API shape, CLI, TUI, and .env mechanism
> preserved from the original design.

## Order of operations

```
[ ] 0. Fetch Curtis's branch locally for reference:
       git fetch origin 'refs/heads/*:refs/remotes/origin/*'
       git show c726a18 > /tmp/curtis_port_reference.txt
[ ] 1. Port data_access.py to CSV-driven backend (~2 hr)
[ ] 2. Port + update tests (~1 hr)
[ ] 3. Port .env.example with post-Phase-2 paths (~10 min)
[ ] 4. Update .gitignore (~2 min)
[ ] 5. Update pipeline/README.md + data_pipeline_to_model_train.md (~30 min)
[ ] 6. Cross-reference in clip_index.py docstring (~5 min)
[ ] V1 pytest local
[ ] V2 count parity vs V3/V4 manifests
[ ] V3 CLI sanity on engelbart
[ ] V4 test_clip_index still passes
[ ] V5 downstream smoke
[ ] 7. Commit with Co-authored-by trailer + branch reference
[deferred] Migrate validate_zeroed_frames.py + fail_rate_per_class.py to use data_access.get_clip_records
```

## Next-session kickoff

1. Confirm main is at the commit carrying this plan file (plus the
   MMPose investigation plan and the validation_scripts refactor plan).
2. Fetch Curtis's branch locally:
   `git fetch origin 'refs/heads/*:refs/remotes/origin/*'` (if it's not
   already in reflog) or pull the commit diff as reference.
3. Read Curtis's `data_access.py` end-to-end to confirm the API surface
   matches what the port above describes.
4. Work through steps 1-7 above.
