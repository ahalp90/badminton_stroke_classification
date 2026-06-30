# Phase 1 Execution Plan: Identify Busted Clips + Decoupled Raw MMPose Extract

> **Status (2026-04-29):** Phase 2 raw extract done. Full 32,203-stem unified raw dir at `/scratch/comp320a/ShuttleSet_keypoints_raw/` on both bourbaki and engelbart (bit-identical), composed of 30,487 freshly re-extracted clips plus the 1,716 Phase-1 backfill merged in. Verification clean: file counts match, cross-node `rsync --checksum` empty, failsafe gate 50/50 byte-identical on the 1,716 overlap, `ndet` baseline captured at `validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md`. Next: `apply_heuristic.py --heuristic sticky_anchor` over the unified dir, then validate / collate / sanity-train. Earlier Phase-1 status preserved below.

> **Status (2026-04-22):** Phase 1 raw extraction + apply_heuristic + sticky_anchor + byte-identity gate all done. 1,716 -> 61 residual busted clips after sticky_anchor (96.4% reduction). Next work is symlink-merge + collate + V4 retrain; see `historical_mmpose_heuristic_investigation.md` section "Revisions 2026-04-22 (Phase 1 execution completion)" for the tactical list and findings. Earlier sections below are the plan as-written pre-execution; read them for context but treat the execution log as the source of truth for current state.

Target session: originally 2026-04-21. Scope: identify the ~222 busted clips and run raw MMPose extraction on them, saving everything we'll need for later heuristic iteration. Heuristic application (`apply_heuristic.py`, `sticky_anchor` variant) is explicitly out of scope and will follow once raw outputs land on disk.

Parent plan: `historical_mmpose_heuristic_investigation.md` (same dir, section "Phase 1").

## Goals for today

1. Produce `docs/architecture_notes/busted_whole_clips_phase1.txt` containing the ~222 stems with `fail_rate > 0.50` in the current extract, excluding unknown.
2. Run raw MMPose extraction across those ~222 clips on engelbart, with no filtering. Save per-clip `_raw_kps`, `_raw_bboxes`, `_raw_scores`, `_raw_kp_scores`, `_raw_ndet` files to a dedicated `_flat_raw_phase1` directory distinct from the committed flat dir.
3. Rsync the raw outputs back to local for downstream heuristic work on a laptop. ~35 MB total; trivial transfer.

Explicitly NOT today: apply_heuristic, heuristics package, mixed re-train, any model training.

## Success criteria for today

- `busted_whole_clips_phase1.txt` materialised, committed (or at least archived locally) at `docs/architecture_notes/busted_whole_clips_phase1.txt`.
- The 222 clips have all five raw output files populated at the engelbart `_flat_raw_phase1` directory.
- Sample verification: for at least one smash clip and one "fully succeeded under current heuristic" control clip, the raw outputs pass a spot-check on shape, NaN padding, and non-zero kp_scores.
- Raw files rsync'd to a local mirror path for laptop-side iteration.

## Task A: `scripts/find_busted_clips.py` scanner

### Purpose

Walk the current engelbart flat dir, compute per-clip MMPose fail rate from each `*_failed.npy`, filter by the taxonomy + split scope, and emit stems that exceed a threshold.

### Design

Reuse the scan structure in `src/bst_x/validation_scripts/validate_zeroed_frames.py`:
- That script already iterates `clips_master.csv`, filters by taxonomy (applies merge_map / standalone_set), and derives split from the `split_column` argument.
- For each clip that survives filtering, it loads `{flat_dir}/{clip_stem}_failed.npy` and computes the fail rate.
- We lean on the same pattern but write a smaller script that focuses on emitting the stem list rather than printing analysis.

### CLI

```
python scripts/find_busted_clips.py \
    --flat-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat \
    --clips-csv /home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv \
    --taxonomy une_merge_v1 \
    --split-column split_v2 \
    --threshold 0.50 \
    --exclude-unknown \
    --output /home/ahalperi/badminton_stroke_classifier/docs/architecture_notes/busted_whole_clips_phase1.txt
```

### Implementation notes

- ~30-40 lines of python, one `argparse` block, one loop over clip rows.
- Import `TAXONOMIES` from `pipeline.config`, resolve the `Taxonomy` object, use `merge_map` + `standalone_set` to compute the display label per clip (matches `_derive_stroke_player` in `validate_zeroed_frames.py:175-187`).
- Filter: `clips_df[clips_df[split_column].isin(['train','val','test'])]`, then optionally exclude unknown rows.
- For each surviving row:
  - Build stem from `clip_stem` column.
  - Load `{flat_dir}/{stem}_failed.npy`. If the file doesn't exist, skip and log.
  - Compute `fail_rate = failed_arr.sum() / len(failed_arr)`.
  - If `fail_rate > threshold`, append the stem to the output list.
- Write the output file: one stem per line, sorted alphabetically for determinism.
- Print a small summary at the end: total scanned, total emitted, split breakdown.

### Expected output shape

Based on the 2026-04-21 analysis of the current extract (`analysis_unemergev1_v2_20260421_1159.txt`):

- ~222 clips emitted at threshold 0.50 excluding unknown.
- Expected split distribution: 110 train / 49 val / 33 test.
- Expected class skew: `Top_smash` heavy (70+ clips across splits), `Top_wrist_smash` / `Top_drop` / `Top_clear` and equivalents following.

Sanity check: if the output size is materially different from 222, investigate before proceeding. A different clip count likely means an out-of-date flat dir or a taxonomy mismatch.

### Validation

After the first run:
1. `wc -l busted_whole_clips_phase1.txt` should return ~222.
2. `cut -d/ -f1 busted_whole_clips_phase1.txt | sort | uniq -c` would show split breakdown, except the stems don't encode split. Instead join back with clips_master.csv to confirm counts (optional; not blocking).
3. Visual spot-check: the file should start with stems that roughly match the top entries of the pre-existing truncated list in `analysis_unemergev1_v2_20260421_1159.txt`.

## Task B: `raw_extract.py` module

### Purpose

Run MMPose 2D pose estimation on a clip-stem subset, without applying any court filtering, and save all raw model outputs per clip for downstream heuristic iteration.

### Design

Sibling module to `prepare_train_on_shuttleset.py`, lives at:
`src/bst_x/stroke_classification/preparing_data/raw_extract.py`

Mirrors the structure of `prepare_2d_dataset_npy_from_raw_video` (line 526 in `prepare_train_on_shuttleset.py`) but:

1. Takes an explicit clip-stem list (from Task A output) and only processes mp4 files whose stems match.
2. Skips the `check_pos_in_court` call entirely; does not construct Top/Bottom slot assignments.
3. Emits the five-array raw schema instead of the three-file `_pos/_joints/_failed` triple.
4. Uses `_raw_ndet.npy` as the resume marker (saved last).

3D path is out of scope for today. Include a commented-out scaffold in the module with a clear toggle note (mirrors the existing `prepare_3d_dataset_npy_from_raw_video` structure) so it's a one-uncomment fix if needed later.

### MMPose result structure

The existing code accesses these keys from the `MMPoseInferencer("human")` result dict:

```python
for frame_num, result in enumerate(inferencer(str(video_path), show=False)):
    for person in result["predictions"][0]:  # batch_size=1
        person["keypoints"]   # list of 17 (x, y) pairs
        person["bbox"][0]     # list-wrapped bbox: [x1, y1, x2, y2]
```

Additional fields expected per MMPose API:
- `person["keypoint_scores"]` — list of 17 per-joint confidence floats.
- `person["bbox_score"]` — per-detection person-level confidence float.

**Action on first run**: print the full structure of `result["predictions"][0][0]` (keys + dtypes) once, with a `--inspect-result` or one-shot debug flag, before the main loop starts. Verifies the field names and shapes match expectations before we commit all 222 clips to the format.

### Output schema

Per clip, five files written to the output flat dir. File naming follows the existing `{stem}_{suffix}.npy` convention for collation-dir-compatibility.

| File | Shape | Dtype | Padding | Resume marker |
|---|---|---|---|---|
| `{stem}_raw_kps.npy` | `(F, N_max, 17, 2)` | float32 | NaN for `i >= ndet[f]` | no |
| `{stem}_raw_bboxes.npy` | `(F, N_max, 4)` | float32 | NaN for `i >= ndet[f]` | no |
| `{stem}_raw_scores.npy` | `(F, N_max)` | float32 | NaN for `i >= ndet[f]` | no |
| `{stem}_raw_kp_scores.npy` | `(F, N_max, 17)` | float32 | NaN for `i >= ndet[f]` | no |
| `{stem}_raw_ndet.npy` | `(F,)` | int8 | (not applicable) | **yes** |

Chosen constants:
- `N_max = 8`. Safe upper bound for badminton broadcast footage (typical is 2-4; hard to imagine 8+ persons in-frame on a well-composed shot).
- **NaN padding** (not zero) because zero is a valid detected coordinate value and would be ambiguous. NaN is unambiguous and standard numpy dtype.
- int8 for `ndet`: range [0, 127] is more than enough for a per-frame count bounded by 8.

Total storage per clip: ~160 KB (dominated by the four float32 arrays at F~60 frames, N_max=8). Across 222 clips: ~35 MB. Trivial for both engelbart scratch and rsync-to-laptop.

### Over-detection handling (ndet > N_max)

If MMPose returns more than 8 detections in a frame:
- Truncate to top 8 by `bbox_score`.
- Log a warning with the clip stem and frame number.
- This should basically never fire on ShuttleSet; if it fires frequently, the padding dimension is too small and N_max needs to grow.

### Resume semantics

- `_raw_ndet.npy` is saved last, after all four float32 arrays are flushed.
- Resume check at the start of each clip: `if Path(save_branch + "_raw_ndet.npy").exists(): continue`.
- This matches the idiom in `prepare_2d_dataset_npy_from_raw_video` where `_failed.npy` is the last-saved file and the resume marker.

### GPU cleanup

Mirror the existing `gc.collect() + torch.cuda.empty_cache()` pattern at the end of each clip's processing block. Engelbart has fragmented memory issues across long runs.

### CLI

```
python -m preparing_data.raw_extract \
    --clips-dir /scratch/comp320a/ShuttleSet_data_merged_25/ShuttleSet/clips \
    --clip-stems-file /home/ahalperi/badminton_stroke_classifier/docs/architecture_notes/busted_whole_clips_phase1.txt \
    --save-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1 \
    --n-max 8
```

Expected arguments:
- `--clips-dir`: root of raw mp4 clips. Default to `CLIPS_OUTPUT_DIR` from `pipeline.config` for local dev; override on engelbart.
- `--clip-stems-file`: output of Task A. One stem per line.
- `--save-dir`: output flat dir for raw files. Must not collide with the primary `_flat/` dir.
- `--n-max`: safety bound for detections per frame. Default 8.
- `--inspect-result`: prints the first frame's result structure then exits. Used once before first batch run.
- `--dry-run`: prints the list of (stem, mp4 path) pairs it would process and exits. Sanity check.

### Stem to mp4 path resolution

The existing Step 2 uses `my_clips_folder.glob("**/*.mp4")` and iterates all. For our subset, we need to filter that glob by stem membership. Options:

- **Filter the glob**: `[p for p in all_mp4_paths if p.stem in stems_set]`. Single disk-scan pass, straightforward. Fast enough (33k files is ~seconds to glob).
- **Dict lookup**: `stem_to_path = {p.stem: p for p in all_mp4_paths}`, then iterate the stems list. Same cost; slightly more explicit iteration order.

Go with the dict lookup: easier to log missing stems (i.e., stems listed in the input file whose mp4 doesn't exist on disk). Log + skip, don't abort.

### Edge cases to log (don't abort)

- Stem in input list but mp4 not found on disk: log + skip. Could be a manual-delete or a symlink issue.
- MMPose returns 0 detections in a frame: save that frame's arrays as all-NaN (for float arrays) and `ndet[f] = 0`.
- MMPose returns > N_max detections: truncate to top 8 by bbox_score, log warning.
- Clip already has `_raw_ndet.npy` in the save dir: resume-skip.

## Task C: execute on engelbart

### Sequence

1. Sync the repo to engelbart. Verify `git log --oneline -1` matches local.
2. Run Task A: generate `busted_whole_clips_phase1.txt`. Wall time seconds.
3. Inspect: `wc -l` and `head -20` of the output file. Confirm ~222 stems and that the first few look like what we expect.
4. Run Task B with `--inspect-result` on a single known-good smash clip (pick any stem from the list). Confirm the MMPose result structure matches the code's expectations (key names, shapes).
5. Run Task B with `--dry-run` on the full stems file. Confirm stem-to-mp4 resolution works for all 222.
6. Run Task B for real against the full stems file. Wall time ~20 min on V100 (222 clips × ~5s/clip).
7. Verify the output dir has 222 × 5 = 1110 `.npy` files. Spot-check one clip: load the arrays, confirm shapes, check that `kp_scores` are non-trivial (not all zero / not all NaN beyond the padding).
8. Rsync the output dir back to local for downstream heuristic work. `rsync -aL engelbart:/scratch/comp320a/.../flat_raw_phase1/ ./local-mirror/`. ~35 MB.

### Wall-time budget

- Scanner (Task A): < 1 minute.
- Raw extract (Task B) dry run + inspect: ~30 seconds.
- Raw extract (Task B) full batch: ~20 minutes (V100, 222 clips × ~5s/clip).
- Rsync: ~1 minute.
- Total: ~25 minutes of engelbart time, plus coding + verification.

### Paths

Canonical paths used in the plan:

- **Repo on engelbart**: `/home/ahalperi/badminton_stroke_classifier/`
- **Current flat dir** (MMPose output, filtered, current heuristic): `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat/`
- **Clips dir** (raw mp4 input): `/scratch/comp320a/ShuttleSet_data_merged_25/ShuttleSet/clips/` (verify on engelbart)
- **New raw output dir**: `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/`
- **clips_master.csv**: `/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv`
- **busted_whole_clips_phase1.txt**: `/home/ahalperi/badminton_stroke_classifier/docs/architecture_notes/busted_whole_clips_phase1.txt`

On local:
- **Raw output mirror**: a folder under local scratch (pick a path; not critical). Suggest `/home/ariel/Documents/COSC594/badminton_stroke_classification/scratch/raw_extract_phase1/`.

### Venv

Per `reference_venv_paths` memory: BST/TrackNet venv on engelbart. Same venv used for the existing pipeline extraction command; `raw_extract.py` has the same imports (mmpose, numpy, torch, tqdm, pandas). No new dependencies.

## File additions for today

New files created:

- `scripts/find_busted_clips.py` (~30-40 lines).
- `src/bst_x/stroke_classification/preparing_data/raw_extract.py` (~120-160 lines; structural clone of `prepare_2d_dataset_npy_from_raw_video` plus the padded-array save logic).
- `docs/architecture_notes/busted_whole_clips_phase1.txt` (materialized by running Task A; one stem per line).

Intentionally NOT created today:

- `src/bst_x/stroke_classification/preparing_data/apply_heuristic.py` — follow-up.
- `src/bst_x/stroke_classification/preparing_data/heuristics/` package — follow-up.
- Any modification to `prepare_train_on_shuttleset.py`, `collate_npy`, `bst_x_train.py`, `bst_x_infer.py`, `shuttleset_dataset.py`, or `pipeline/`.

## Verification checklist before closing the session

- [ ] `busted_whole_clips_phase1.txt` has ~222 entries. Split distribution roughly matches 110/49/33.
- [ ] `raw_extract.py` `--inspect-result` run confirmed MMPose returns `keypoints`, `bbox`, `keypoint_scores`, `bbox_score` fields per detection.
- [ ] Test-run on one clip produced 5 files: `_raw_kps.npy`, `_raw_bboxes.npy`, `_raw_scores.npy`, `_raw_kp_scores.npy`, `_raw_ndet.npy`.
- [ ] Test-run array shapes match the schema: `(F, 8, 17, 2)`, `(F, 8, 4)`, `(F, 8)`, `(F, 8, 17)`, `(F,)`.
- [ ] `kp_scores` values are non-trivial (not all 0, not all NaN beyond padding).
- [ ] NaN padding verified: `np.isnan(raw_kps[f, ndet[f]:, :, :]).all()` holds for a sample frame.
- [ ] Full batch completed for all 222 clips. `ls {save_dir} | wc -l` returns 1110.
- [ ] Output dir is distinct from the primary filtered flat dir. Existing `_failed.npy` files there are untouched.
- [ ] Rsync'd to local, 35 MB arrived intact.

## Sanity references

- MMPose result structure: `src/bst_x/stroke_classification/preparing_data/prepare_train_on_shuttleset.py:261-292`.
- Resume marker pattern: same file, line 566 (`if not Path(save_branch + "_failed.npy").exists():`).
- GPU cleanup: same file, lines 589-590.
- Clip scanning / filtering logic to reuse in the scanner: `src/bst_x/validation_scripts/validate_zeroed_frames.py:175-280`.
- Homography + projection code (NOT called in raw_extract, but referenced in the parent plan's sticky_anchor math): `prepare_train_on_shuttleset.py:100-154` (`get_court_info`, `to_court_coordinate`, `normalize_position`).

## What follows (not today)

1. Write `apply_heuristic.py` CLI + `heuristics/` package with `current` (reference gate) + `sticky_anchor` (primary variant).
2. Run `apply_heuristic.py` twice on the raw output: once with `current` (byte-identity check against committed filtered extract), once with `sticky_anchor`.
3. Build symlink-merged flat dir for Phase-1 mixed re-train (32k - 222 unchanged symlinks + 222 real `sticky_anchor` files).
4. Collate + re-train V4 on the mixed data, compare against committed V4 baseline.
5. Decision gate: if Phase 1 signal clears, proceed to Phase 2 (full 33k re-extract + re-train). If not, trial `body_length_fallback` as Plan B.

See the parent plan `historical_mmpose_heuristic_investigation.md` for the full Phase 1 measurement protocol, decision gate thresholds, and Phase 2 structure.

## Scope anchors and anti-hallucination reminders (appended 2026-04-21)

For a session starting clean on this plan: re-read these before starting work. These are the most likely drift and hallucination points.

### Strict scope today

- **Do**: write `find_busted_clips.py`, write `raw_extract.py`, run both on engelbart, rsync the raw outputs back.
- **Do not**: write `apply_heuristic.py`, write any heuristics package modules, modify `prepare_train_on_shuttleset.py`, modify `collate_npy`, modify training or data-loading code. Phase 1 part 2 (heuristic application) is a separate session.
- **Do not**: run MMPose extraction outside the 222-stem list. Primary `_flat/` dir stays untouched.

### Module size bounds (keep minimal)

- `find_busted_clips.py`: ~30-40 lines. argparse + single clips_master.csv iteration + per-clip `_failed.npy` loads + filter + write. Do not reproduce the per-class analysis, PNG generation, or hit-frame proximity code that lives in `validate_zeroed_frames.py`.
- `raw_extract.py`: ~120-160 lines. Clone the structure of `prepare_2d_dataset_npy_from_raw_video` (at `prepare_train_on_shuttleset.py:526`) with the filter replaced by raw-save logic. No CLI for heuristic application, no court-projection code, no collation.

### MMPose API verification (first step every time)

- `--inspect-result` runs once on a single clip, prints the first frame's result structure, exits. Always run this before the batch. Never trust memory of field names.
- Expected keys per detection in `result["predictions"][0][i]`:
  - `keypoints` (shape ~ list of 17 (x, y)) — confirmed from `prepare_train_on_shuttleset.py:263`.
  - `bbox` (list-wrapped `[x1, y1, x2, y2]`; code uses `person["bbox"][0]`) — confirmed from line 291.
  - `keypoint_scores` (list of 17 floats) — expected per MMPose API but not used in existing code. Verify on first run.
  - `bbox_score` (float) — expected per MMPose API. Verify on first run.

If the field names differ, the only safe action is to adjust the extractor accordingly and re-run inspection before proceeding to the batch.

### Critical save details (easy to miss)

- **Five** output files per clip, not three. The raw pipeline emits `_raw_kps`, `_raw_bboxes`, `_raw_scores`, `_raw_kp_scores`, `_raw_ndet`. Do not forget `_raw_kp_scores.npy`; it is the new one, and the sticky_anchor ankle-confidence fallback depends on it.
- `_raw_ndet.npy` is the resume marker and must be saved last. Resume logic checks for its existence.
- NaN padding (not zero padding) for all float arrays. Zero is ambiguous with valid detected coordinates at the origin.
- int8 dtype for `_raw_ndet` (range 0-127, more than enough for the N_max=8 cap).
- The raw pipeline does **not** emit `_failed.npy`. That file belongs to the current filtered pipeline only. The raw pipeline has no per-frame failure concept because it applies no filter.

### Path canon

Do not improvise paths from memory. Use these exactly:

- Engelbart repo: `/home/ahalperi/badminton_stroke_classifier/`
- Engelbart current flat dir: `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat/`
- Engelbart clips dir: `/scratch/comp320a/ShuttleSet_data_merged_25/ShuttleSet/clips/` **VERIFY on engelbart**. The `pipeline.config.CLIPS_OUTPUT_DIR` default is `PROJECT_ROOT / 'ShuttleSet' / 'clips'`, which resolves to the local-dev path. Engelbart may use a scratch-dir override or symlink.
- New raw output dir: `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/` (distinct from the primary `_flat/`).
- busted_whole_clips_phase1.txt: `/home/ahalperi/badminton_stroke_classifier/docs/architecture_notes/busted_whole_clips_phase1.txt`.

### Order of operations (do not skip steps)

1. Scanner first. Verify count (~222) and rough split breakdown before proceeding.
2. `--inspect-result` on one clip. Verify MMPose field names match expectations.
3. `--dry-run` on full list. Verify all 222 mp4 paths resolve on disk.
4. Full batch run. Verify 1110 files in output dir (222 clips × 5 files each).
5. Rsync to local mirror. Verify ~35 MB arrived.

Steps 1-3 must pass before the batch. Skipping them risks burning 20 min of GPU time on a broken extractor.

### Things a fresh session is likely to hallucinate or get wrong

- **Thinking the raw extract also writes `_failed.npy`**. It does not. Resume marker is `_raw_ndet.npy`.
- **Including 3D extraction**. Explicitly out of scope for this pass. Commented scaffold only; do not uncomment.
- **Over-engineering the scanner**. It just emits stems. Per-class analysis, PNG rendering, split-balanced sampling are all out of scope; they duplicate existing validation tooling.
- **Adding stroke-type priority filtering to the scanner**. `fail_rate > 0.50` + exclude unknown is the entire filter rule.
- **Growing `N_max` beyond 8**. 8 is already generous for broadcast singles. Only grow if the batch run logs frequent truncation warnings.
- **Zero-padding instead of NaN-padding**. Zero is a valid coordinate value and is ambiguous.
- **Filtering the raw extract by in-court detection**. That is the whole thing we are decoupling from. No filter. If you find yourself writing `check_pos_in_court`, stop.
- **Re-implementing court-projection code**. The raw extract does not need it. Projection is a downstream concern for `apply_heuristic.py`.
- **Running the full batch before inspect + dry-run**. Always verify MMPose field names and path resolution first.
- **Silently overwriting the primary `_flat/` dir**. Output must go to `_flat_raw_phase1/`. If the output dir arg defaults to `_flat/`, something is wrong with the defaults.

### What follows (deliberately out of scope for this session)

See the parent plan `historical_mmpose_heuristic_investigation.md` for:
- `apply_heuristic.py` CLI + `heuristics/` package design.
- `current` variant byte-identity gate protocol.
- `sticky_anchor` algorithm detail.
- Symlink-merged flat dir pattern for Phase-1 mixed re-train.
- Decision gate thresholds and Phase 2 structure.

## Execution log (2026-04-22)

Raw extraction for Phase 1 is done. This section records what actually happened, where we diverged from the plan as-written, and what the next steps look like.

### What was executed

1. Wrote `scripts/find_busted_clips.py`. First pass used the plan's whole-clip `fail_rate > 0.50` criterion, emitted **222 stems**, split 143/45/34 train/val/test. (Plan guessed 110/49/33 from an older analysis; total matched, split ratios didn't.)
2. Clarified that the intended criterion was the **hit-zone fail rate** (matches the `hit_zone_heatmap` filter in `validate_zeroed_frames.py`), not whole-clip. Added `--hit-zone`, `--set-dir`, `--video-metadata-csv`, `--hit-window` flags. Kept whole-clip as the default mode so the flag doesn't change existing behaviour.
3. Re-ran the scanner with `--hit-zone --hit-window 10 --exclude-unknown`. Emitted **1,716 stems**. This is the canonical Phase 1 busted set from here on. The 222-stem whole-clip list is retained as a historical artefact at `docs/architecture_notes/busted_whole_clips_phase1.txt`; the canonical hit-zone list lives at `docs/architecture_notes/busted_hit_zone_clips_phase1.txt`.
4. Wrote `src/bst_x/stroke_classification/preparing_data/raw_extract.py`. Five-array schema (`_raw_kps`, `_raw_bboxes`, `_raw_scores`, `_raw_kp_scores`, `_raw_ndet`), NaN padding, int8 `ndet`, `_raw_ndet.npy` as the resume marker. `--inspect-result` and `--dry-run` flags for pre-flight verification. End-of-run summary prints the unique list of clips that triggered over-detection warnings (survives tmux scrollback truncation).
5. Initial extract ran at `N_max = 8` against the 222-clip whole-clip list. Completed successfully, but **193 of 222 clips** triggered the over-detection cap (87%). Plan said "basically never fires"; that was true for the full 33k corpus, not for the busted subset which is specifically over-represented in crowded-frame clips.
6. Preserved the N=8 extract at `/scratch/.../dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1_n8/` for optional later diffing against N=16 on the stems where the two lists overlap.
7. Re-extracted the 1,716-clip hit-zone list at `N_max = 16` to `/scratch/.../dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/`. At N=16 only 0.79% of frames (~780 / ~98k) hit the cap; raising further isn't needed.
8. Rsynced the canonical raw extract back to local for heuristic iteration.
9. Added `src/bst_x/validation_scripts/mmpose_heuristic_investigation/summarise_raw_ndet.py` as a diagnostic helper (per-clip and aggregate detection-count breakdown, with optional `bbox_score` filter for "likely real persons"). Not in the plan as-written; added to quantify the over-detection picture.

### Divergences from the plan as-written

| Plan said | Actual | Reason |
|---|---|---|
| Criterion: whole-clip `fail_rate > 0.50` | Hit-zone fail rate (+/-10 frames around the hit frame) > 0.50 | Matches `hit_zone_heatmap` definition; whole-clip was mis-specified in the original draft. |
| ~222 clips | 1,716 clips (hit-zone) | Hit-zone catches clips where the stroke-moment specifically failed, even if the rest of the clip is clean. |
| Expected split 110/49/33 | 143/45/34 on whole-clip (222 total); hit-zone split not yet tabulated | Old analysis input; the total was the load-bearing figure. |
| `N_max = 8` is safe | N=8 hit on 193/222 (87%) of whole-clip run; bumped to 16 | Busted subset is crowded-frame-heavy. At N=16, <1% of frames truncate. |
| Clips dir at `/scratch/comp320a/ShuttleSet_data_merged_25/ShuttleSet/clips` | Actual: `/scratch/comp320a/ShuttleSet/clips` | Plan's nested path was a wrong guess; `--clips-dir` override was the intended correction point. |
| `set/` + `video_metadata.csv` under `/scratch` | Actual: repo-tracked under `src/bst_x/ShuttleSet/` | These files are committed to git, not symlinked out to `/scratch` like `clips/` / `raw_video/` / `shuttle_csv/` / `shuttle_npy/`. |
| Two new files (scanner + extractor) | Three: added `summarise_raw_ndet.py` | Needed to quantify ndet distribution after the N=8 warnings fired. |
| No end-of-run warning summary | Added a unique-clip summary at end of raw_extract run | tmux scrollback can lose per-clip warnings; end-of-run survives. |

### Ndet findings (from `summarise_raw_ndet.py` on the 1,716-clip N=16 extract)

- 98,370 total frames across 1,716 clips (mean ~57 frames/clip).
- Raw ndet distribution peaks at 9-10 per frame (73.79% of frames <= 10 detections). Cap-hit rate at N=16 is 0.79%.
- Score-filtered (`bbox_score >= 0.5`) distribution peaks at 8 per frame (32.84%). Median clip has `hi_mean ~= 7.9`. The 0.5 threshold catches players + umpire + line judges + visible crowd; score alone does not isolate players.
- User visual audit: in every inspected clip, players were the most salient detections by bbox size and horizontal centrality. This is load-bearing information for sticky_anchor's selector design (discussion in-flight; selector likely leans on area + centrality with score as a filter/tiebreaker rather than as the primary criterion).

### Artefacts produced

New files on local (not yet committed):
- `src/bst_x/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py` (+ `--hit-zone` / `--set-dir` / `--video-metadata-csv` / `--hit-window` flags).
- `src/bst_x/stroke_classification/preparing_data/raw_extract.py`.
- `src/bst_x/validation_scripts/mmpose_heuristic_investigation/summarise_raw_ndet.py`.
- `src/bst_x/validation_scripts/mmpose_heuristic_investigation/diagnose_top_k_capture.py` and `render_detection_overlays.py` (added later in the session as the sticky_anchor selector design developed).

On engelbart (not yet committed):
- `docs/architecture_notes/busted_whole_clips_phase1.txt` (222 stems, whole-clip).
- `docs/architecture_notes/busted_hit_zone_clips_phase1.txt` (1,716 stems, hit-zone; canonical).

Scratch-dir extracts on engelbart:
- `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1_n8/`: historical N=8 run on the 222-clip whole-clip set.
- `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/`: canonical N=16 run on the 1,716-clip hit-zone set.

Raw extract also synced to local for heuristic iteration (path is per-machine; not canonical).

### Next steps

> [Items 1-4 and 7 complete as of the Phase 1 execution completion (see `historical_mmpose_heuristic_investigation.md` > "Revisions 2026-04-22 (Phase 1 execution completion)"). Items 5 and 6 are the active next steps; current canonical list lives in the parent doc's renumbered "Still to do for Phase 1" block.]

1. **Revisit sticky_anchor design** with the ndet findings as input. The score-alone-is-not-discriminative observation and the player-size/centrality dominance change the selector weighting. Design discussion is in-flight with Ariel; do not pre-empt.
2. Write `apply_heuristic.py` + `heuristics/` package (`current` + `sticky_anchor`).
3. Run `current` variant as the byte-identity gate on the overlap between the 1,716-clip hit-zone set and the committed filtered extract. Investigate plumbing if it doesn't reproduce bit-identically.
4. Run `sticky_anchor` on the full 1,716-clip raw extract.
5. Build symlink-merged flat dir for Phase 1 mixed re-train (32k - 1,716 unchanged symlinks + 1,716 sticky_anchor outputs).
6. Collate + train V4 on merged data; compare min-F1 and zeroing rate against committed V4.
7. Commit the three new scripts + the 1,716-stem hit-zone list as a reproducible Phase 1 anchor once the design discussion lands.

### Inputs for the next sticky_anchor design pass

- Ndet data shows ~8 confident detections per typical frame; selector must discriminate inside that crowd.
- Player-dominance-by-size/centrality confirmed by user audit; area + centre-distance are reliable signals.
- 5 clips with 100% zeroed hit-zones have been rsynced for visual inspection; findings pending.
- The 222 vs 1,716 scope change means the heuristic runs over 7.7x more clips than originally scoped; storage and I/O still trivial at N=16 (under 1 GB for the full raw set).
- Homography is calibrated to the outer (doubles) taped court. Current `eps = 0.01` filter gives only ~6 cm sideline / ~13 cm baseline of slack, ~1/8 to 1/20 of real in-play overflow. See `historical_mmpose_heuristic_investigation.md` "Court-space geometry and buffer sizing (2026-04-22)" for the full audit including overlay PNG reference (`src/bst_x/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/homography_overlay_3_1_18_3_f032.png`).

### Design stance: irrecoverable clips

Some fraction of the 1,716 hit-zone-busted clips are genuinely irrecoverable by any heuristic that operates on raw MMPose output: broadcast extreme close-ups (body cropped to a wrist), side-on or front-on close-ups with no court visible, cuts to a different subject (e.g. a doubles game in the background), and similar artistic broadcast decisions. sticky_anchor cannot fix these because the failure is upstream: MMPose either detects nothing usable, detects the wrong subject with no court-projection basis for rejecting it, or the court-projection itself is invalid for the close-up frame.

Position on how to handle them:

- **Full 1,716 remains the denominator** for the Phase 1 decision gate (zeroing-rate drop, min-F1 lift). No subset carve-out for "recoverable" vs "irrecoverable" clips. If sticky_anchor's improvement is diluted by irrecoverable clips, the metric will reflect that honestly.
- **Irrecoverable clips stay zeroed after sticky_anchor runs.** That's correct behaviour, not a heuristic failure.
- **No manual categorisation pass** over the irrecoverable tail. Time is better spent on the design and training-side measurements; the unrecoverable clips being zeroed is already the current pipeline's behaviour and doesn't regress anything.
- **Training-side view**: zeroed irrecoverable clips at worst regularise the model (noise the transformer learns to ignore or attend around), at best give the model novel representations to learn from. No drop-from-training proposal unless later evidence shows they actively harm performance.

