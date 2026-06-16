# Validation Scripts

Analysis tools for the pose/shuttle dataset. Some run on the raw mmpose extract (pre-heuristic), others on the post-heuristic / post-collation per-clip arrays. Use them before training to assess data quality and surface failure floors.

The post-collation CLI scripts (`validate_zeroed_frames.py`, `fail_rate_per_class.py`) are CSV-driven: splits and labels come from `notebooks/clips_master.csv` (per `--split-column` and `--taxonomy`), not from the on-disk folder tree. Per-clip `.npy` files resolve flat at `{dataset_npy_dir}/{clip_stem}_*.npy`. This matches the Phase 2 flat-dir layout and the taxonomy definitions in `pipeline/config.py`.

## Scripts

### `raw_ndet_stats.py`

Summarises the per-frame mmpose detection-count distribution across a raw extract dir (the `*_raw_ndet.npy` files written by `preparing_data/raw_extract.py`). Establishes the irreducible per-frame failure floor for any heuristic that requires two players: frames where mmpose returns `ndet < 2` will be zeroed by the heuristic regardless of variant.

Run this **after** raw extraction but **before** `apply_heuristic.py` to confirm the raw extract is sane before sinking time into the postprocessing pass.

**Usage** (full canonical raw dir):

```bash
python -m validation_scripts.raw_ndet_stats \
    --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw
```

**Usage** (restricted to a stem subset, e.g. for comparing the Phase-1 backfill against the freshly extracted bulk):

```bash
python -m validation_scripts.raw_ndet_stats \
    --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw \
    --stems-file docs/architecture_notes/busted_hit_zone_clips_phase1.txt
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--raw-dir` | Yes | - | Flat dir holding `*_raw_ndet.npy` files. |
| `--stems-file` | No | - | One-stem-per-line filter restricting the scan. |
| `--output` | No | `raw_ndet_stats_outputs/<auto>.md` | Markdown report path. |
| `--no-output` | No | off | Print stdout only; skip the markdown report. |

**Output**: a markdown report at `raw_ndet_stats_outputs/raw_ndet_stats_{raw_dir_name}[_stems_subset]_{timestamp}.md`, plus the same content on stdout. Reports the clip count, total frames, ndet==0 floor, per-clip zero-rate quantiles, and the full ndet histogram. The most useful single number is the `ndet=1` percentage — that's the realistic per-frame failure floor for the downstream pipeline.

`baseline_2026-04-29.md` in the outputs dir is the post Phase-2 consolidation reference snapshot to compare future re-extracts against.

### `validate_zeroed_frames.py`

Analyses two independent detection failure modes across the dataset:

1. **MMPose failures** (from `*_failed.npy`): MMPose failed to detect exactly 2 players on court — joints, court positions, and shuttle coordinates are all zeroed on these frames. The BST-X transformer does **not** mask them in attention, so they act as noise.

2. **Shuttle detection failures** (from shuttle NPYs, optional): TrackNetV3 reported visibility=0 (shuttle not detected). Independent of MMPose — the visibility column is dropped during collation, so these failures are invisible to the model as silent (0, 0) shuttle coordinates.

**Minimal usage** (MMPose failure stats only, from repo root):

```bash
python src/bst_x/validation_scripts/validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_merged_25 \
    --split-column split_bst_baseline \
    --taxonomy une_merge_v1
```

`--clips-csv` defaults to `<repo>/notebooks/clips_master.csv`. `--dataset-npy-dir` is auto-discovered under `--data-root` (the single `*_flat/` subdir). `--set-dir` is auto-detected at `<repo>/data/shuttleset/set` if a `match.csv` is present there, which also enables the flaw and hit-frame sections below.

**Full usage** (explicit paths for flaw cross-reference, hit-frame proximity, and shuttle analysis):

```bash
python src/bst_x/validation_scripts/validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_merged_25 \
    --split-column split_v2 \
    --taxonomy une_merge_v1 \
    --set-dir data/shuttleset/set \
    --hit-window 10 \
    --shuttle-npy-dir /scratch/comp320a/ShuttleSet/shuttle_npy_flat
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-root` | Yes | - | Path to `ShuttleSet_data_{taxonomy}` directory. The per-clip npy directory is auto-discovered inside it. |
| `--dataset-npy-dir` | No | auto-discover | Explicit flat per-clip npy dir. Required when `--data-root` holds more than one `*_flat/` subdir. |
| `--clips-csv` | No | `<repo>/notebooks/clips_master.csv` | Master clips CSV (one row per clip). |
| `--split-column` | No | `split_bst_baseline` | Column in clips_csv giving train/val/test assignment. |
| `--taxonomy` | No | `une_merge_v1` | Taxonomy name (choices from `TAXONOMIES` in `pipeline/config.py`). Used for label derivation, filenames, and display headers. |
| `--threshold` | No | `0.5` | Fail-rate cutoff for the flagged-clips list. |
| `--set-dir` | No | repo-relative fallback | Path to `data/shuttleset/set/`. Enables flaw cross-reference and hit-frame proximity. If omitted, checks `<repo>/data/shuttleset/set` for a `match.csv` and uses it when found. |
| `--hit-window` | No | `10` | Frames either side of the hit frame to check. Requires `--set-dir`. |
| `--shuttle-npy-dir` | No | - | Path to `data/shuttleset/shuttle_npy_flat/`. Enables shuttle detection failure analysis via the TrackNet visibility column. |

**Output** (all saved to `zeroed_frames_analysis_outputs/`):

| File | Contents |
|---|---|
| `analysis_{tax_short}_{split_short}_{date}_{time}.txt` | Full text report (mirrors terminal output) |
| `fail_rate_histogram_{tax_short}_{split_short}_{date}_{time}.png` | Per-clip fail rate distribution (log y-axis) |
| `temporal_pattern_{tax_short}_{split_short}_{date}_{time}.png` | Mean fail rate by normalised clip position |
| `hit_frame_profile_{tax_short}_{split_short}_{date}_{time}.png` | Fail rate by frame offset from hit, with shuttle overlay if available *(requires `--set-dir`)* |
| `hit_zone_heatmap_{tax_short}_{split_short}_{date}_{time}.png` | Heatmap of % clips exceeding threshold in hit zone, by class × split *(requires `--set-dir`)* |
| `surviving_clips_{tax_short}_{split_short}_{date}_{time}.png` | Per-class clip counts remaining after hit-zone quality filter, by split *(requires `--set-dir`)* |
| `hit_oob_clips_{tax_short}_{split_short}_{date}_{time}.txt` | Clips where hit-frame index exceeded clip length, skipped from hit-frame profile *(requires `--set-dir`; only written when OOB clips exist)* |

`tax_short` strips underscores from the taxonomy (`une_merge_v1` -> `unemergev1`); `split_short` strips the `split_` prefix and remaining underscores (`split_bst_baseline` -> `bstbaseline`). Output filenames disambiguate by split so back-to-back runs on different `--split-column` values don't overwrite each other.

Timestamps use Sydney time (AEST/AEDT). The `unknown/` garbage class is excluded from figures, tiered clip counts, flaw cross-reference, shuttle overlap, and hit-frame proximity sections. It is included in overall, per-split, and per-stroke stats (visible as a row).

**Report sections:**

1. **Overall MMPose stats** — total failed frames / total frames across all clips
2. **Per-split breakdown** — train/val/test MMPose fail rates
3. **Per-stroke-type** — MMPose fail rates by stroke class, sorted highest first
4. **Tiered clip counts** — clips at 100%, >90%, >75%, >50% failure, with names for the worst offenders
5. **Flaw cross-reference** *(requires `--set-dir`)* — compares fail rates for shots marked `flaw=1.0` in the original ShuttleSet annotations vs. non-flaw shots
6. **Shuttle detection failures** *(requires `--shuttle-npy-dir`)* — overall and per-split shuttle non-detection rates, plus a 2×2 overlap table showing how MMPose and shuttle failures correlate (both fail, only one, or neither)
7. **Hit-frame proximity** *(requires `--set-dir`)* — compares MMPose fail rates near the hit vs. away, tiered hit-zone clip counts, per-stroke breakdown. When shuttle data is available, also reports shuttle miss rates near the hit, combined data-quality metric (frames where both MMPose and shuttle succeeded), and per-stroke shuttle hit-zone breakdown

### `fail_rate_per_class.py`

Per-class MMPose fail-rate stats joined on `clips_master.csv`. Reads the flat per-clip `*_failed.npy` files, applies the requested taxonomy (and optional `--drop-unknown`), and prints per-class totals so you can see which class is carrying the most zeroed frames. Useful for seeing how the per-class pose-quality picture shifts between taxonomies (e.g. `merged_25` vs `une_merge_v1`) without rerunning the full `validate_zeroed_frames.py` report.

**With explicit `--dataset-npy-dir`:**

```bash
python src/bst_x/validation_scripts/fail_rate_per_class.py \
    --clips-csv notebooks/clips_master.csv \
    --dataset-npy-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat \
    --split-column split_bst_baseline \
    --taxonomy une_merge_v1 \
    --drop-unknown
```

**With `--data-root` auto-discovery** (mirrors `validate_zeroed_frames.py`):

```bash
python src/bst_x/validation_scripts/fail_rate_per_class.py \
    --clips-csv notebooks/clips_master.csv \
    --data-root /scratch/comp320a/ShuttleSet_data_merged_25 \
    --split-column split_v2 \
    --taxonomy une_merge_v1 \
    --drop-unknown \
    --save-txt
```

Exactly one of `--dataset-npy-dir` or `--data-root` must be given. Auto-discovery fails explicitly when >1 `*_flat/` subdirs are found under `--data-root` (pass `--dataset-npy-dir` to disambiguate).

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--clips-csv` | Yes | - | Master clips CSV. |
| `--data-root` | No* | - | `ShuttleSet_data_{taxonomy}` dir; enables `*_flat` auto-discovery. |
| `--dataset-npy-dir` | No* | - | Explicit flat per-clip dir holding `{clip_stem}_failed.npy`. |
| `--split-column` | No | `split_bst_baseline` | Column in clips_csv giving train/val/test assignment. |
| `--taxonomy` | No | `une_merge_v1` | Taxonomy name. |
| `--drop-unknown` | No | off | Drop `raw_type_en == "unknown"` rows before aggregating. |
| `--save-txt` | No | off | Tee stdout to `zeroed_frames_analysis_outputs/fail_rate_per_class_{tax_short}_{split_short}_{ts}.txt`. |

*One of `--data-root` / `--dataset-npy-dir` is required.

### Pre-flight verification scripts

Three small scripts that confirm a sanity-train run is pointed at the right artefacts before launch. Each exits 0 on all-OK, 1 on any mismatch, so they double as `set -e` guards in launch wrappers.

- **`verify_env_paths.py`** — loads `.env` via `pipeline.data_access.load_repo_dotenv`, prints the four `BST_X_*` vars, and confirms each resolves to an existing path. Spot-checks `BST_X_MMPOSE_NPY_DIR` for 32,203 `_failed.npy` and `_pos.npy` files (the post-Phase-2 expected count).
- **`verify_collated_counts.py`** — pure-stdlib check that the three active collated dirs (combo A / B / C) contain the per-split clip counts expected from `clips_master.csv` filtered for `--drop-unknown`. Hardcoded combo expectations; no external CSV read needed at run time.
- **`verify_bst_train_target.py`** — imports the live `hyp` namedtuple from `bst_x_train` without running its `__main__` block, derives the basename via the same `derive_npy_collated_dir_basename` helper the script uses (`npy_[3d_][seq{N}_]{split}_{collation_id}`), and confirms the resolved collated dir exists with its train/val/test sub-dirs and `.npy` files. Resolves the root the same way `bst_x_train` does (`BST_X_COLLATED_DATA_ROOT`, else `/scratch/comp320a`). The standard pre-launch check after a `hyp` edit.

All three run from the repo root:

```bash
PYTHONPATH=src/bst_x \
    python src/bst_x/validation_scripts/<script>.py
```

### Phase-2 shuttle-missing diagnosis scripts

Three scripts authored 2026-04-30 to verify the off-screen-high hypothesis for the 6.34% post-inpaint shuttle-missing rate, and to test whether the bottleneck classes correlate with shuttle availability.

- **`shuttle_gap_y_distribution.py`** — for every contiguous run of `visibility=0` frames in a per-clip shuttle NPY, records the y-coordinate of the last valid detection before the gap and the first valid detection after, then aggregates across the canonical 32k-clip set. Drops unknowns by default via `--clips-csv`. Saves a histogram PNG + markdown report to `zeroed_frames_analysis_outputs/`.
- **`shuttle_gap_length_distribution.py`** — gap-length histogram + classification by length-class (1-2 / 3-5 / 6-10 / 11-30 / 31-60 / 61+ frames) interpreted as motion-blur / inpaint sweet spot / off-screen-arc / sustained / inpaint-window-exceeded. Same drop-unknown default.
- **`perclass_shuttle_miss_vs_f1.py`** — joins per-class median F1 from a run's `manifest.yaml` against the per-stroke shuttle-miss-rate table parsed from a `validate_zeroed_frames.py` analysis txt. Pearson + Spearman correlation, scatter PNG, sorted markdown table. Defaults to F1 (the metric currently in the manifest schema); `--metric precision` and `--metric recall` are placeholders gated on the schema extension. Use `--no-collapse-sides` when the manifest's labels already match the analysis table directly (nosides taxonomy).

Run on engelbart or bourbaki for the gap scripts (data is on `/scratch`, host-local). The correlation script can run anywhere since manifest + analysis txt both live on `/home`.

### `hit_frame_lookup.py`

Reusable library module (not a CLI script). Maps clip stems to the 0-based frame index of the hit within the clip by re-deriving clip boundaries from the ShuttleSet set CSVs.

```python
from hit_frame_lookup import build_hit_frame_lookup
lookup = build_hit_frame_lookup(Path("data/shuttleset/set"), Path("data/shuttleset/video_metadata.csv"))
# lookup["35_1_10_17"] == 23  means the hit is at frame index 23
```

Uses the same `between_2_hits_with_max_limits` windowing logic as the clip generator, without needing video files. FPS is read from `video_metadata.csv` (the same source of truth as the clip generator) rather than estimated from annotations.

See also `src/bst_x/pipeline/clip_index.py`: the analogous helper for Datasets needing O(1) clip-stem -> video-path lookup against the same `clips_master.csv`.

## Dependencies

- Python 3.10+ (uses `X | None` union syntax)
- `numpy`, `matplotlib`, `pandas` — all available in the mmpose venv
- `zoneinfo` — stdlib (Python 3.9+)
- `pipeline.config` (in-repo): both CLI scripts import `TAXONOMIES` and `Taxonomy` from `src/bst_x/pipeline/config.py` to drive label derivation and the `--taxonomy` choices list.
