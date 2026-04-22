# Directory Flatten Refactor — Comprehensive Plan

Status as of 2026-04-21: Phase 1 + Phase 2 complete. V3 committed at 3a98149; V4 ran on engelbart 2026-04-20, best S2 macro 0.743 / min 0.432 / acc 0.766. Pre-commit gate passed (dry-run, verify_shuttle_sync, shuttle_csvs_to_npy diff, V4 re-collate diff all bit-identical; Step 2 single-clip pose writer smoke good; validate_zeroed_frames CSV-driven run sane; test_dataset.py green locally). Next user-driven steps: re-extract busted smash MMPoses against the flat writer, then V5 (`rm -r` the nested originals). Phase 3 (flatten the .mp4 clips dir) still deferred.

Post-Phase-2 cosmetic: collated dir naming shortened to `npy_[3d_][seq{N}_]{ablation_id}`. Prefix tags (`3d_`, `seq{N}_`) appear only for non-default configs (2D + seq_len=100 strips both). Drops the `dataset_npy_collated_between_2_hits_with_max_limits_seq_100_...` prefix that duplicated info already in manifest.yaml. Existing V3/V4 dirs on engelbart keep their old names; the rename applies only to new runs. References updated across `prepare_train_on_shuttleset.py`, `bst_train.py`, `bst_infer.py`, `shuttleset_dataset.py.__main__`, `test_integration.py`, `data_pipeline_to_model_train.md`, `testing_guide.md`.

Also landed: `--pose-styles` CLI arg on Step 3, default `JnB_bone` (the only style any committed run has used). Non-requested pose representations skip both compute and save. Cuts pose-tensor disk from ~928MB to ~232MB per ablation (~75%).

## Goals

1. **Decouple labels and split assignment from physical directory layout.** Per-clip `.npy` files live flat (`{root}/{clip_stem}_*.npy`); split + label come from `clips_master.csv` at collation time. (Phase 1, done.)
2. **Make future re-extractions write flat directly** so the pipeline is internally consistent and we never recreate the legacy nested `{split}/{class}/` layout. (Phase 2.)
3. **Keep the existing `merged_25` baseline and the two une_merge_v1 ablations reproducible** — the flat collation path already passed V1 (histogram match) and V2 (smoke-train).

## Out of scope

- Flattening the upstream `.mp4` clips directory (`CLIPS_OUTPUT_DIR / {split} / {class} / *.mp4`). Pose extraction already reads it recursively via `glob('**/*.mp4')`, so it doesn't block anything. Defer to Phase 3.
- Any model architecture changes.
- Any taxonomy changes beyond those already in `pipeline/config.py` (`merged_25`, `une_merge_v1`, `raw_35`).
- V5 (deleting the nested originals on engelbart). User-initiated after Phase 2 lands and the re-extraction is verified.

## What the V3 result changes about priority

V3 best serial (S3): macro 0.772, min 0.381, acc 0.791. Mean across 5 serials dropped roughly 0.06 macro / 0.25 min / 0.06 acc vs the merged_25 baseline. The min-F1 drop is almost entirely Top_smash, which carries a 24.33% MMPose fail rate in train (980 clips). Under merged_25 that class pooled with wrist_smash at ~21%; uncollapsing exposed both the 24.33% Top_smash and 16.40% Top_wrist_smash tiers, and the rare classes cannot absorb the loss.

Re-extracting the busted smash MMPoses is the next substantive work. Phase 2 Bundle 1 lands before that re-extraction so the new files write flat and never reintroduce the nested layout.

## File inventory

Every file that reads or writes the per-clip npy or shuttle_npy layout, with line citations, grouped by role.

### Producers — write into the nested `{split}/{class}/` tree

| File | Lines | What it writes | Refactor needed |
|---|---|---|---|
| `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py` | 495-504 (`mk_same_dir_structure`), 538-606 (`prepare_2d_dataset_npy_from_raw_video`), 609-665 (`prepare_3d_dataset_npy_from_raw_video`) | `_pos.npy`, `_joints.npy`, `_failed.npy` per clip into `{save_root}/{set_split_dir.name}/{ball_type_dir.name}/{clip_stem}_*.npy` | Yes — Phase 2.1. Drop the `{split}/{class}/` parents and write to `{save_root}/{clip_stem}_*.npy`. Delete `mk_same_dir_structure`. |
| `src/bst_refactor/pipeline/shuttle_extractor.py` | 246-312 (`shuttle_csvs_to_npy`), specifically 272-274 mirrors clip path, 308 per-file mkdir | `{SHUTTLE_OUTPUT_DIR}/{split}/{class}/{clip_stem}.npy` from flat shuttle_csv inputs | Yes — Phase 2.2. Write to `{SHUTTLE_OUTPUT_DIR}/{clip_stem}.npy`; drop the `rel.with_suffix('.npy')` mirror and the per-file mkdir. |
| `src/bst_refactor/pipeline/clip_generator.py` | 127-189 (`_write_clips_for_video`), 151-156 mkdirs class folders | `.mp4` clips into `{CLIPS_OUTPUT_DIR}/{split}/{Top,Bottom}_{class}/{clip_stem}.mp4` | Out of scope — clips dir stays nested for now (Phase 3). |

### Producers — already flat

| File | Lines | What it writes |
|---|---|---|
| `src/bst_refactor/pipeline/shuttle_extractor.py` | 58-156 (`extract_shuttle_trajectory`, `extract_all_shuttles`) | TrackNetV3 CSVs to flat `{shuttle_csv_dir}/{clip_stem}_ball.csv`. No change. |

### Consumers — already CSV-driven (done in Phase 1)

| File | Lines | Role |
|---|---|---|
| `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py` | 704-921 (`collate_npy`) | Reads clips_master.csv, filters by `split_column`, derives label via taxonomy, resolves `{root_dir}/{clip_stem}_*.npy` directly. Handles missing flat files with warn-and-skip. |

### Consumers — still walk the nested tree

| File | Lines | What it reads | Refactor needed |
|---|---|---|---|
| `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py` | 142-224 (`Dataset_npy`) | Walks `{root_dir}/{set_name}/{class}/*_pos.npy` lazily. | Phase 2.6 — deprecate with a `DeprecationWarning`. Only caller is the `compare_pred_gt_on_specific_type` debug helper in `bst_train.py`. |
| `src/bst_refactor/validation_scripts/validate_zeroed_frames.py` | 140-162 (`_load_shuttle_vis`), 165-266 (`scan_clips`), 1226-1246 (main auto-discovery) | `{shuttle_npy_dir}/{split}/{folder_name}/{clip_name}.npy` and `{dataset_npy_dir}/{split}/{class_dir}/*_failed.npy` | Phase 2.3 — switch to CSV-driven iteration keyed on clips_master.csv. Derive stroke_type + player via taxonomy merge_map (same pattern as `fail_rate_per_class.py`). |
| `src/bst_refactor/pipeline/verify.py` | 200-246 (`verify_shuttle_sync`) | Compares `mp4.relative_to(clips_dir).with_suffix('.npy')` paths to `npy.relative_to(shuttle_dir)` paths | Phase 2.4 — compare by `clip_path.stem` instead (clips stay nested, shuttle goes flat). |
| `src/bst_refactor/pipeline/verify.py` | 90-125 (`verify_class_merge`), 128-159 (`verify_splits_present`), 162-181 (`warn_orphan_files`), 248-275 (`print_dataset_summary`) | Walk the clips dir (stays nested). | No change. |

### Consumers — collated stacked arrays (layer is semantic, not organizational)

These read `{root_dir}/{train,val,test}/{J_only.npy, JnB_bone.npy, pos.npy, shuttle.npy, labels.npy, videos_len.npy}`. The `{train,val,test}/` layer is what each split contains, not directory mirroring, and does not need a layout change.

| File | Lines | Role |
|---|---|---|
| `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py` | 227-326 (`Dataset_npy_collated`) + variants | Loads stacked arrays per split. No change. |
| `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py` | Hyp (139-156), `Task.prepare_dataloaders`, collated-dir resolution | Already threaded through clips_csv / split_column / drop_unknown / ablation_id. No change. |
| `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_infer.py` | 26, 75-87, 121-127 | Inference. No change required for flatten; could be threaded for ablation provenance as follow-up. |
| `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/test_fwd.py` | 7-14 | Smoke-test script, hardcoded path at old collated dir. Phase 2.5 — repoint. `n_class=25` on line 7 also needs bumping to 29 if pointing at a une_merge_v1 ablation. |
| `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/test_train_step.py` | 9-15 | Same pattern. Phase 2.5. |
| `tests/test_integration.py` | 8-88 | Uses `BST_DATA_DIR` env var pointing at a collated dir root. No layout change; user points it at any collated dir. |
| `tests/test_dataset.py` | 1-end | Synthetic npy; no external data. |

### Documentation references

Each needs a 1-line update to describe flat shuttle_npy output and a forward pointer to this refactor doc where relevant.

| File | Lines | Notes |
|---|---|---|
| `src/bst_refactor/pipeline/README.md` | 106-149 (shuttle_npy setup + output), 219-222 (output structure tree), 312-314 (loader example) | Primary user-facing description. |
| `src/bst_refactor/data_pipeline_to_model_train.md` | 98 (shuttle_extractor cell), 112-117 (pipeline output tree), 260-262 (Dataset_npy entry), 445-450 (call-chain diagram) | High-level pipeline writeup. |
| `src/bst_refactor/stroke_classification/preparing_data/mmpose_changes.md` | 151-159 (per-clip output table) | Notes the flat output layout. |
| `src/bst_refactor/stroke_classification/preparing_data/keypoints_schema.md` | 77-81 | Per-clip failure retention. Comment on layout stays minimal. |
| `tests/testing_guide.md` | 40-44 (BST_DATA_DIR example) | Collated dir name now ablation-tagged; update example path. |
| `notebooks/03_build_clips_master.ipynb` | (entire notebook) | Already documents the master-CSV approach. |

### Migration tooling (already used)

| File | Role |
|---|---|
| `scripts/flatten_copy.sh` | Copies nested → flat staging dirs. Ran once on engelbart for Phase 1. After Phase 2.1 + 2.2 land, new writes go flat directly and this script becomes obsolete (keep for reference until V5). |
| `scripts/verify_flatten.py` | Confirmed flat copies match original content + every clip correlates to master CSV under `merged_25`. Already passed. |
| `notebooks/03_build_clips_master.ipynb` | Builds `clips_master.csv` (33,481 rows) from `pipeline.player_mapping.collect_shots`, `pipeline.config.SPLITS`, `EXCLUDED_VIDEOS`, `REMOVED_SHOTS`, joined to `shuttleset_splits_v2.csv`. |

### Already-landed support (do not revisit)

| File | Change | Commit |
|---|---|---|
| `src/bst_refactor/validation_scripts/fail_rate_per_class.py` | CSV-driven per-class MMPose fail-rate diagnostic. Provided the Top_smash 24.33% / Top_wrist_smash 16.40% fail-rate data behind the V3 min-F1 finding. | 3a98149 |
| `src/bst_refactor/run_overview.py` | Scalar-only filter on metric auto-discovery so `per_class_f1` dicts don't produce junk aggregation columns. | 3a98149 |
| `bst_train.py` | Empty-class mask in `validate()` and `Task.test()`, per-class F1 top-5/bot-5 on "Picked!" val epochs, `per_class_f1` dict in `track_serial` manifest output, `show_details=True` at test. | 3a98149 |

## Phase 1 — minimum viable for V1-V4 (complete)

Landed in commits up to 3a98149. Every gate passed:

- V1 (baseline collation reproduces histograms): pass.
- V2 (smoke-train baseline matches benchmark): pass.
- V3 (une_merge_v1 + split_bst_baseline + drop_unknown): ran 2026-04-20; best S3 macro 0.772, min 0.381, acc 0.791. Five serials on engelbart, manifest + best_model_id.txt + tactical unignore committed.
- V4 (une_merge_v1 + split_v2 + drop_unknown): collation landed (22,743 / 5,250 / 4,210), training in flight on engelbart.

`collate_npy()` is CSV-driven; `bst_train.py` Hyp pins `clips_csv / split_column / drop_unknown / ablation_id`; manifest writes `data_provenance` with CSV sha256 and `effective_ablation_id`. Per-ablation collated dir naming (`{ablation_id}` suffix) prevents collision.

## Phase 2 — full end-state cleanup (complete 2026-04-21)

All bundle items landed in the combined Phase-2 + V4 commit. The subsections below (2.1-2.7) remain as historical reference showing what changed where; the gate results at the bottom of the doc record what was verified.

**Bonus landed in the same commit**: per-iteration `gc.collect()` + `torch.cuda.empty_cache()` hoisted inside the `if not Path(save_branch + "_failed.npy").exists():` branch in both 2D and 3D pose writers. Skip iterations drop from ~100 ms to ~1 ms — cuts single-clip resume wall time from ~60 min to ~1 min on the 33k-clip glob, which matters for the upcoming smash re-extract (many clips, most skipped).

Two bundles. Bundle 1 blocks the user-initiated smash MMPose re-extraction that follows the V3 min-F1 finding; Bundle 2 is cleanup and docs.

### Bundle 1 (blocks re-extraction)

#### 2.1 `prepare_2d/3d_dataset_npy_from_raw_video` write flat — `prepare_train_on_shuttleset.py`

Current (2D, same shape for 3D):

```python
# line 562
mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)
...
# lines 571-574
ball_type_dir = video_path.parent
set_split_dir = ball_type_dir.parent
save_branch = str(
    save_root_dir / set_split_dir.name / ball_type_dir.name / video_path.stem
)
```

After:

```python
# function entry
save_root_dir.mkdir(parents=True, exist_ok=True)
...
# per clip
save_branch = str(save_root_dir / video_path.stem)
```

The `glob('**/*.mp4')` input walk at line 564 / 628 does not change — clips stay nested until Phase 3.

Same edits for `prepare_3d_dataset_npy_from_raw_video` at lines 626, 636-640. Delete `mk_same_dir_structure` (495-504) once both callers no longer reference it.

In `main()` around lines 1049-1081, `npy_raw_dir` becomes redundant: Step 2's `save_root_dir` should be `flat_clip_npy_dir`, which is what Step 3's collation already reads. Simplify:

```python
# was
npy_raw_dir = preparing_root / f'dataset{str_3d}_npy_between_2_hits_with_max_limits'
flat_clip_npy_dir = args.clip_npy_dir or (
    preparing_root / f'dataset{str_3d}_npy_between_2_hits_with_max_limits_flat'
)

# now — single flat dir, used as both Step 2 output and Step 3 input
flat_clip_npy_dir = args.clip_npy_dir or (
    preparing_root / f'dataset{str_3d}_npy_between_2_hits_with_max_limits_flat'
)
```

The `dry_run` block (1090) and the Step 2 call site (1131-1145) lose the `npy_raw_dir` reference and pass `flat_clip_npy_dir` instead. Update the flat_clip_npy_dir comment (1075-1081) to describe "flat writer + flat reader" rather than "flatten_copy.sh". Update the `--skip-collate` error message (1154-1158) — after 2.1 lands the flat dir is produced by Step 2 directly, not by `scripts/flatten_copy.sh`.

**Note for re-extraction workflow**: Step 2's resume check is `if not Path(save_branch + "_failed.npy").exists()`. To force re-extraction of busted smash clips, the user deletes the three `{clip_stem}_{pos,joints,failed}.npy` files for those clips from the flat dir, then re-runs Step 2. Step 2 then writes fresh flat files alongside the remaining intact ones.

#### 2.2 `shuttle_csvs_to_npy` writes flat — `pipeline/shuttle_extractor.py`

Current (lines 271-309):

```python
for clip_path in sorted(clips_dir.rglob('*.mp4')):
    rel = clip_path.relative_to(clips_dir)
    npy_path = npy_output_dir / rel.with_suffix('.npy')
    ...
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), shuttle_norm)
```

After:

```python
npy_output_dir.mkdir(parents=True, exist_ok=True)  # hoisted to function entry
for clip_path in sorted(clips_dir.rglob('*.mp4')):
    npy_path = npy_output_dir / (clip_path.stem + '.npy')
    ...
    np.save(str(npy_path), shuttle_norm)
```

Update the docstring at 253-262 to describe flat output. The `clips_dir.rglob('*.mp4')` input walk stays — clips remain nested.

#### 2.3 `validate_zeroed_frames.py` CSV-driven — `validation_scripts/validate_zeroed_frames.py`

Two surfaces change.

`_load_shuttle_vis` (140-163). Drop `split` and `folder_name` parameters:

```python
def _load_shuttle_vis(
    shuttle_npy_dir: Path, clip_name: str, n_frames: int,
) -> np.ndarray | None:
    shuttle_path = shuttle_npy_dir / f'{clip_name}.npy'
    ...
```

`scan_clips` (165-266). Switch from the nested dir walk (186-203) to a CSV-driven iteration. Accept a `clips_csv` path + `split_column` + `taxonomy` and derive `player`, `stroke_type`, `split`, `clip_name` from the CSV directly (same `derive_labels` pattern as `fail_rate_per_class.py`):

```python
def scan_clips(
    dataset_npy_dir: Path,
    clips_csv: Path,
    split_column: str,
    taxonomy: Taxonomy,
    flaw_lookup: dict[str, bool] | None = None,
    shuttle_npy_dir: Path | None = None,
) -> list[ClipRecord]:
    df = pd.read_csv(clips_csv)
    standalone_set = taxonomy.standalone_set
    merge_map = taxonomy.merge_map or {}
    for row in df.itertuples():
        split = getattr(row, split_column)
        if split not in SPLITS:
            continue
        clip_name = row.clip_stem
        failed_path = dataset_npy_dir / f'{clip_name}_failed.npy'
        if not failed_path.exists():
            continue
        arr = np.load(failed_path)
        # Derive player + stroke_type via the taxonomy.
        merged = merge_map.get(row.raw_type_en, row.raw_type_en)
        if merged in standalone_set:
            player, stroke_type = '', merged
        else:
            player, stroke_type = row.player_side, merged
        ...
```

Update the call site at 240-244 to match the new `_load_shuttle_vis` signature.

Update `main()` auto-discovery (1226-1246). Today it looks for `dataset*npy*` subdirs that contain split subfolders and excludes `collated`. Post-flatten, the flat dir has no split subfolders. Switch to looking for names matching `*_flat` (primary) with a fallback to a user-provided `--dataset-npy-dir`. Add CLI args: `--clips-csv`, `--split-column`, `--taxonomy`.

The `rel_path` field in `ClipRecord` (line 53) keeps its legacy format `f"{split}/{folder_name}/{clip_name}"` for display purposes; it is built synthetically from the CSV-derived fields.

#### 2.4 `verify.py` shuttle_sync flat — `pipeline/verify.py`

Current (216-224):

```python
clip_stems = set()
for mp4 in clips_dir.rglob('*.mp4'):
    rel = mp4.relative_to(clips_dir).with_suffix('.npy')
    clip_stems.add(rel)

npy_stems = set()
for npy in shuttle_dir.rglob('*.npy'):
    rel = npy.relative_to(shuttle_dir)
    npy_stems.add(rel)
```

After (compare by stem, since shuttle goes flat but clips stay nested):

```python
clip_stems = {mp4.stem for mp4 in clips_dir.rglob('*.mp4')}
npy_stems = {npy.stem for npy in shuttle_dir.rglob('*.npy')}
```

Adjust `missing_npy` / `orphan_npy` output formatting (230-242) to print stems (or flat paths), not relative paths.

`verify_class_merge`, `verify_splits_present`, `warn_orphan_files`, `verify_file_integrity`, `print_dataset_summary`: no change (operate on the clips dir, which stays nested).

### Bundle 2 (cleanup follow-up)

#### 2.5 `tmp/test_*.py` hardcoded paths

`main_on_shuttleset/tmp/test_fwd.py` (7-14): repoint the three `np.load` paths at a current ablation's collated dir. `n_class=25` at line 7 also bumps to 29 if pointing at a une_merge_v1 ablation.

`main_on_shuttleset/tmp/test_train_step.py` (9-15): same; the `root` variable on line 9 collects the three loads.

Add a 1-line comment noting these are smoke-test scripts and the path is not invariant across ablations.

#### 2.6 `Dataset_npy` deprecation — `shuttleset_dataset.py:142-224`

Only caller is `bst_train.py`'s `compare_pred_gt_on_specific_type` debug helper. Add a `DeprecationWarning` at `__init__` and a short docstring noting it expects the legacy nested layout and will not work against the flat dir without a CSV-driven rewrite. Do not delete (still wired into the debug helper).

#### 2.7 Documentation refresh

Each doc gets a 1-line update describing the flat layout. The output-tree diagrams shift from:

```
shuttle_npy/
  train/{Top,Bottom}_{stroke_type}/*.npy
  val/{Top,Bottom}_{stroke_type}/*.npy
  test/{Top,Bottom}_{stroke_type}/*.npy
```

to:

```
shuttle_npy/
  {clip_stem}.npy   # flat, split + label come from clips_master.csv at collation time
```

Specific files:
- `pipeline/README.md` (106-149, 219-222, 312-314).
- `data_pipeline_to_model_train.md` (98, 112-117, 260-262, 445-450).
- `preparing_data/mmpose_changes.md` (151-159 per-clip output table).
- `preparing_data/keypoints_schema.md` (77-81). Minor — retention note is layout-agnostic.
- `tests/testing_guide.md` (40-44). Update BST_DATA_DIR example path to the ablation-tagged collated dir name.

Add a forward pointer to `scratch/architecture_notes/completed_general_refactors/dir_flatten_refactor.md` from `pipeline/README.md` and `data_pipeline_to_model_train.md`.

## Phase 3 (deferred) — flatten the `.mp4` clips dir

Affects `clip_generator.py` writer + `shuttle_extractor.py` input scanner + `verify.py` clip-side scanners + downstream pose-extraction `**/*.mp4` glob. Self-contained but touches more files. Not blocking the ablations or the smash re-extraction.

## Verification

Pre-commit gate run 2026-04-21, all green:

| Gate | What it exercises | Result |
|---|---|---|
| Dry-run (`prepare_train_on_shuttleset --dry-run --taxonomy une_merge_v1 --split-column split_v2 --drop-unknown`) | main() arg plumbing, path composition, `npy_raw_dir` removal | Single flat_clip_npy line, correct ablation_id |
| `verify.py --shuttle-dir shuttle_npy_flat` | 2.4 stem-based shuttle sync | PASS, 33,481 clips matched |
| `shuttle_csvs_to_npy` to `/tmp/shuttle_npy_phase2_test` + `diff -q -r` vs committed flat | 2.2 flat writer | Zero output, bit-identical |
| V4 re-collate with `--ablation-id ..._phase2test` + 24-file diff vs committed V4 arrays | main() wiring + collate_npy end-to-end | Zero `differ` lines across train/val/test × 8 arrays |
| Single-clip delete + Step 2 re-extract | 2.1 pose writer flat | All three `_pos/_joints/_failed.npy` land at `$FLAT/{stem}_*.npy`, resume check works |
| `validate_zeroed_frames.py --clips-csv --split-column --taxonomy` full run | 2.3 CSV-driven iteration | 33,481 clips scanned, sensible per-stroke fail rates, `match.csv` path warning (orthogonal to validator) |
| `pytest tests/test_dataset.py` locally | Dataset_npy_collated shape contracts | 1 test passed |

V4 training result (ran 2026-04-20, pre Phase 2 land): best S2 macro 0.743 / min 0.432 / acc 0.766. Per_class_f1 in every serial, empty-class mask working (min not pinned at 0). Split totals 22,743 / 5,250 / 4,210 matched expectations at collation.

### V5 (user-initiated, follow-up)

After the smash re-extraction + re-training reports healthy smash F1, delete the nested originals on engelbart:

```bash
rm -r /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits
rm -r /scratch/comp320a/ShuttleSet/shuttle_npy
```

After this, the flat dirs (`..._flat/`, `shuttle_npy_flat/`) are the only on-disk source. The `_flat` suffix on the dir name can be dropped at leisure after V5 (rename on disk + update `flat_clip_npy_dir` default in `prepare_train_on_shuttleset.py` main()).

## Risks and rollback

| Risk | Mitigation |
|---|---|
| Post-2.1 Step 2 accidentally writes flat output at a different root than Step 3 reads | Both now derive from the same `flat_clip_npy_dir` variable in main(). Smoke test catches divergence immediately. |
| Re-extraction overwrites an intact clip's files | Resume check (`_failed.npy` existence) only writes when files are absent. User must delete first to force re-extract — deliberate and reversible. |
| validate_zeroed_frames.py output format shifts enough to break downstream tooling | No known downstream tooling parses its output. The .txt + PNG outputs land in the same `zeroed_frames_analysis_outputs/` sibling dir. |
| `Dataset_npy` debug helper breaks after deprecation | Helper is invoked rarely. If needed, refactor then instead of now. |

Rollback: `git revert`. Flat dirs and master CSV are inputs that don't move. Phase 1 is already at HEAD 3a98149; Phase 2 is purely additive until 2.1 flips the writer, at which point a revert plus deleting the incorrectly-placed flat files restores prior state.

## Order of operations checklist

```
[done] Phase 0: master CSV, copy script, verify script, V1/V2 patches
[done] Phase 1.1: collate_npy refactor (CSV-driven)
[done] Phase 1.2: bst_train.py knobs (clips_csv, split_column, drop_unknown, ablation_id)
[done] Phase 1.3: per-ablation collated dir naming
[done] V1: baseline collation reproduces histograms
[done] V2: smoke-train baseline matches benchmark
[done] V3: ablation 1 (une_merge_v1 + bst baseline split, drop unknown)
[done] V4: ablation 2 (une_merge_v1 + v2 split, drop unknown)
[done] V4 artifact package (manifest notes + best_model_id.txt + tactical unignore + commit)
[done] Phase 2.1: pose writers go flat (+ drop npy_raw_dir in main())
[done] Phase 2.2: shuttle_csvs_to_npy writes flat
[done] Phase 2.3: validate_zeroed_frames.py CSV-driven
[done] Phase 2.4: verify_shuttle_sync compares by stem
[done] Phase 2.5: tmp test path updates
[done] Phase 2.6: Dataset_npy deprecation
[done] Phase 2.7: doc updates
[done] Bonus: gc/empty_cache hoisted inside Step 2 inference branch (~60x faster resume path)
[done] Pre-commit gate: all 7 checks green (dry-run, verify, shuttle diff, collate diff, Step 2 smoke, validate_zeroed_frames, pytest test_dataset)
[user] Re-extract busted smash MMPoses on engelbart (flat writer)
[user] V5: rm -r the nested originals on engelbart
[deferred] Phase 3: flatten .mp4 clips dir
```
