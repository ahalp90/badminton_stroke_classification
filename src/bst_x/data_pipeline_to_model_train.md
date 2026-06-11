# Data Pipeline to Model Training: Module Reference

End-to-end walkthrough of the modules needed to go from raw ShuttleSet data to a trained BST-X model, with notes on where a custom (non-BST-X) architecture would diverge.

New here? [`data_pipeline_and_model_train_overview.md`](data_pipeline_and_model_train_overview.md) is the 5-minute narrative; this doc is the module-by-module detail. For the `pipeline/` folder on its own, see [`pipeline/README.md`](pipeline/README.md).

## Contents

- [Quick Start: End-to-End Execution](#quick-start-end-to-end-execution)
- [Part 1: BST-X on ShuttleSet](#part-1-bst-x-on-shuttleset)
  - [Stage 1: Build the Dataset](#stage-1----build-the-dataset-pipeline)
  - [Stage 2: Prepare Training Data](#stage-2----prepare-training-data-stroke_classificationpreparing_data)
  - [Between Stages 2 and 3: Data Quality Validation](#between-stages-2-and-3----data-quality-validation-validation_scripts)
  - [Stage 3: Dataset Loading](#stage-3----dataset-loading-stroke_classificationpreparing_datashuttleset_datasetpy)
  - [Stage 4: Model](#stage-4----model-stroke_classificationmodel)
  - [Stage 5: Training](#stage-5----training-stroke_classificationmain_on_shuttleset)
  - [Stage 6: Inference](#stage-6----inference-stroke_classificationmain_on_shuttlesetbst_inferpy)
  - [Stage 7: Results](#stage-7----results-stroke_classificationresult_utilspy)
  - [Full dependency chain](#full-dependency-chain-bst-x-on-shuttleset)
- [Part 2: Adapting for a Custom (Non-BST-X) Model](#part-2-adapting-for-a-custom-non-bst-x-model)

---

## Quick Start: End-to-End Execution

The project uses three separate Python environments because the OpenMMLab stack (MMPose) requires numpy < 2.0 while the training stack uses numpy 2.x. All three target **Python 3.11**.

| Environment | Requirements file | Purpose |
|---|---|---|
| **Pipeline** | `pipeline/requirements.txt` | Download videos, generate clips, verify output |
| **MMPose** | `stroke_classification/preparing_data/requirements.txt` | Pose estimation (steps 1-2 of data preparation) |
| **BST training** | `stroke_classification/requirements.txt` | Collation, training, inference. Also shared by TrackNetV3. |

### Environment setup

```bash
# 1. Pipeline venv
python3.11 -m venv venv-pipeline
source venv-pipeline/bin/activate
pip install -r pipeline/requirements.txt

# 2. MMPose venv (requires C++ compiler + CUDA toolkit for mmcv source build)
python3.11 -m venv venv-mmpose
source venv-mmpose/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
mim install mmcv==2.1.0
pip install -r stroke_classification/preparing_data/requirements.txt

# 3. BST training venv
python3.11 -m venv venv-bst-x
source venv-bst-x/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r stroke_classification/requirements.txt
```

### Execution order

```bash
# ── Stage 1: Build dataset (pipeline venv) ──────────────────────────
source venv-pipeline/bin/activate

python -m pipeline.build_dataset --dry-run                # preview
python -m pipeline.build_dataset --skip-shuttle            # download + clips + verify
# Optional: shuttle extraction (uses BST venv for TrackNetV3)
python -m pipeline.build_dataset --skip-download \
    --tracknet-dir TrackNetV3 \
    --tracknet-python /path/to/venv-bst-x/bin/python
# Resume after crash (skip completed steps 3-5, run only shuttle extraction)
python -m pipeline.build_dataset \
    --skip-download --skip-resolution --skip-clips --skip-verify \
    --tracknet-dir TrackNetV3 \
    --tracknet-python /path/to/venv-bst-x/bin/python

# ── Stage 2: Pose estimation (MMPose venv) ──────────────────────────
source venv-mmpose/bin/activate

# On engelbart, symlink the taxonomy output dir to scratch first (see Stage 2 Setup below).
# Run from the repo root with both package roots on PYTHONPATH (matches conftest.py for tests).
export PYTHONPATH=src/bst_x:src/bst_x/stroke_classification

python -m preparing_data.prepare_train_on_shuttleset \
    --skip-trajectory --skip-collate                       # pose only (no shuttle CSV needed)

# ── Stage 3: Collation + training (BST venv) ────────────────────────
source venv-bst-x/bin/activate
export PYTHONPATH=src/bst_x:src/bst_x/stroke_classification

python -m preparing_data.prepare_train_on_shuttleset \
    --skip-trajectory --skip-pose                          # collate (reads shuttle CSVs)

python -m main_on_shuttleset.bst_x_train                     # train (5 serial trials)
python -m main_on_shuttleset.bst_x_infer                     # inference
```

The same `PYTHONPATH=src/bst_x:src/bst_x/stroke_classification` setting is what `conftest.py` inserts for the test suite, so test and production invocation share one resolution layout. Pre-step-P invocation (`cd main_on_shuttleset && python bst_x_train.py`) no longer works after the proper-packages refactor.

Each stage's output feeds the next. Stages are independently re-runnable — use `--skip-*` flags to avoid repeating completed work. **Important:** after class merge (step 4) has run, always pass `--skip-clips` on re-runs to avoid re-generating clips that were moved into merged folders.

---

## Part 1: BST-X on ShuttleSet

### Stage 1 -- Build the Dataset (`pipeline/`)

The pipeline downloads match videos, cuts them into labeled stroke clips, optionally extracts shuttle trajectories, and verifies the result. All configuration is centralized in `config.py`; the orchestrator `build_dataset.py` runs the steps in sequence.

#### Modules

| Module | Role | Key functions / concepts |
|--------|------|--------------------------|
| `config.py` | Single source of truth for paths, stroke types, splits, flaw records, and merge rules. Every other pipeline module imports from here. | • `Taxonomy`: frozen dataclass pinning `classes` (the ordered class list), `merge_map`, `has_sides`, `excluded_base_stroke_types`; `n_classes`/`has_unknown` are properties.<br>• `TAXONOMIES`: the six pinned taxonomies (`bst_25`, `bst_24`, `bst_12`, `une_v1_14`, `une_v1_15`, `shuttleset_18`).<br>• `TAXONOMY_ALIASES` + `resolve_taxonomy()`: map legacy names (`une_merge_v1`, `merged_25`, ...) to current ones.<br>• `label_for_row()`: the single merge + side + exclusion decision, used by the collator.<br>• `UNE_MERGE_V1_MAP` (19 -> 14) + `MERGE_MAP_25` (19 -> 12, paper-faithful `driven_flight -> drive`).<br>• `SPLITS` (excluded videos auto-stripped), `parse_flaw_records()` -> `EXCLUDED_VIDEOS`/`REMOVED_SHOTS`, `EN_TO_ZH`/`ZH_TO_EN` (CSV I/O only), `UNPREFIXED_TYPES`. |
| `build_dataset.py` | One-command orchestrator. Runs steps 1-6 in order with CLI flags to skip individual steps (`--skip-download`, `--skip-resolution`, `--skip-clips`, `--skip-verify`, `--skip-shuttle`). `--skip-clips` skips both clip generation (step 3) and class merge (step 4) since they are tightly coupled: the merge moves clips out of their original folders, so re-running step 3 after a merge would re-generate them from video. | `run_pipeline()` (main entry point), `dry_run()` (preview without side effects), `_validate_inputs()` (fail-fast checks before long work). |
| `download_videos.py` | Downloads 40 ShuttleSet match videos from YouTube via yt-dlp. Also builds a resolution CSV by scanning each video with OpenCV. | `download_all_videos(max_workers)`, `build_resolution_csv()`. Output: `ShuttleSet/raw_video/{id} {match_name}.mp4` and `ShuttleSet/my_raw_video_resolution.csv`. |
| `clip_generator.py` | Extracts individual stroke clips from full match videos. Reads ShuttleSet CSV annotations (Chinese column names), maps A/B players to Top/Bottom, filters excluded videos and removed shots, and organizes clips into `{split}/{Player}_{stroke_type}/` folders. | `generate_all_clips()`, `apply_class_merge()` (moves clips from rare subtype folders into their parent type folders per the active taxonomy's merge map). Three clip window modes: `middle_in_a_sec`, `between_2_hits`, `between_2_hits_with_max_limits` (default, clamps to 1.5s each side). |
| `player_mapping.py` | Maps the A/B player labels in ShuttleSet annotations to Top/Bottom court positions. Handles set-3 court switches. | `get_top_bottom_mapping(video_id, set_num)`. |
| `verify.py` | Post-generation sanity checks: all splits present, no clips from excluded videos, no removed shots, merged subtype folders empty, no orphan files. | `verify_splits_present()`, `verify_no_excluded()`, `verify_no_removed_shots()`, `verify_class_merge()`, `verify_shuttle_sync()`, `print_dataset_summary()`. |
| `shuttle_extractor.py` | Runs TrackNetV3 on each clip to detect shuttle positions, then converts CSVs to normalized `(t, 3)` numpy arrays `[x_norm, y_norm, visibility]`. Uses **batch mode** (`batch_predict.py`) to load models once per worker and iterate over clips in-process, avoiding the ~8s model-reload per clip. Uses the default `eval_mode='weight'` (full temporal ensemble) for maximum detection accuracy. `--batch_size` (default 32, configurable via CLI) controls GPU utilization. Inference runs in **FP32** to preserve detection accuracy on fast-moving shuttles (FP16 rounding can flip the 0.5 heatmap threshold on faint responses). Frames are pre-resized during loading using PIL BICUBIC (bit-identical to the Dataset's own resize). VideoCapture handles are explicitly released and `gc.collect()` + `torch.cuda.empty_cache()` run between clips to prevent resource exhaustion. `--workers N` launches N parallel batch workers, each with its own model copy (use 1 on V100 16GB, 2+ on larger GPUs). On V100 16GB, batch_size 16 fits most clips; a few may OOM, so re-run with batch_size 8 to pick up stragglers (resume logic skips clips that already have CSVs). `--dry-run` processes clips without writing output files (for testing). TrackNetV3 shares the BST training venv. **Pretrained weights** (`ckpts/TrackNet_best.pt`, `ckpts/InpaintNet_best.pt`) must be downloaded separately (~150 MB, gitignored) — see `TrackNetV3/README.md`. | `extract_all_shuttles(tracknet_dir, tracknet_python, max_workers, batch_size, dry_run)`, `shuttle_csvs_to_npy()`. Intermediate output: `ShuttleSet/shuttle_csv/` (flat dir of per-clip CSVs, taxonomy/split independent). Final output: `ShuttleSet/shuttle_npy/{clip}.npy` (flat; split + label come from `notebooks/clips_master.csv` at collation time). |
| `court_utils.py` | Optional. Homography-based camera-to-court coordinate projection. Not required for the core pipeline. | `project_to_court()`, `normalize_court_position()`. |

#### Pipeline output structure

```
ShuttleSet/
  raw_video/                         # Full match videos
  my_raw_video_resolution.csv        # Width/height per video
  clips/                             # Labeled stroke clips (still nested)
    train/{Top,Bottom}_{type}/*.mp4
    val/{Top,Bottom}_{type}/*.mp4
    test/{Top,Bottom}_{type}/*.mp4
  shuttle_csv/                       # TrackNetV3 intermediate CSVs (flat)
    {vid}_{set}_{rally}_{ball_round}_ball.csv
  shuttle_npy/                       # Shuttle trajectories (flat, optional)
    {vid}_{set}_{rally}_{ball_round}.npy
```

Split and label assignment for `shuttle_npy/` (and downstream pose npys) come from `notebooks/clips_master.csv` at collation time, not from directory structure. The clips directory stays nested for now. See `scratch/architecture_notes/completed_general_refactors/dir_flatten_refactor.md` for the migration.

#### Key concepts

- **Class merging**: Each taxonomy pins its full ordered class list and a merge map that folds rare raw subtypes into parents. The current Architecture 1 active config is `une_v1_14` (4 subtypes folded -> 14 merged types, no sides, unknown excluded); `une_v1_15` is the same plus `unknown` (15). The BST-paper family folds 6 subtypes to 12 base types: `bst_25` (12 x Top/Bottom + `unknown` = 25), `bst_24` (no unknown = 24), `bst_12` (nosides = 12). `shuttleset_18` keeps the 18 raw types (no merge, no sides, no unknown). Legacy names (`une_merge_v1`, `une_merge_v1_nosides`, `merged_25`, `raw_35`) resolve to their current equivalent via `TAXONOMY_ALIASES` / `resolve_taxonomy()`. Whether `unknown` is kept is contractual per taxonomy (its `excluded_base_stroke_types`), not a runtime flag.
- **Flaw records**: `flaw_shot_records.csv` is the single source of truth for data exclusions. Whole-video exclusions and individual shot removals are parsed at import time.
- **Clip windows**: Control how much temporal context surrounds each stroke. `between_2_hits_with_max_limits` (default) uses the interval between adjacent shots, clamped to 1.5s per side.
- **Homography resolution**: The pre-computed homography matrices in `ShuttleSet/set/homography.csv` were calculated at 1280x720. `court_utils.scale_pos_by_resolution()` rescales coordinates from the video's native resolution to 1280x720 before applying the homography. This quantization is negligible for court-position features (~1cm precision on a 13m court), but worth keeping in mind if homography-derived coordinates are ever combined with features extracted at native resolution (e.g., shuttle trajectory positions relative to a video crop). In practice any mismatch would be sub-pixel at typical crop sizes and likely acts as minor augmentation noise.
- **Video resolution**: The pipeline downloads the best available mp4 (video-only, no audio). Downstream models resize frames internally — TrackNetV3 to 512x288 (`TrackNetV3/utils/general.py`), MMPose to ~256x192 depending on model config — so resolutions above 720p provide no practical benefit while increasing file size and processing time.

---

### Stage 2 -- Prepare Training Data (`stroke_classification/preparing_data/`)

The pipeline produces **video clips** and **shuttle .npy files**. BST-X does not operate on raw video -- it needs pre-extracted skeletal pose, court position, and shuttle trajectory arrays. This stage bridges the gap.

#### Module

| Module | Role | Key functions / concepts |
|--------|------|--------------------------|
| `prepare_train_on_shuttleset.py` | Runs MMPose on each clip to extract 2D (or 3D) player keypoints, combines them with shuttle trajectories at collation time, normalizes everything, and collates per-sample arrays into batch-ready `.npy` files. | **Step 1**: `prepare_trajectory()` -- run TrackNetV3 on clips, saving CSVs to `ShuttleSet/shuttle_csv/` (if shuttle extraction wasn't done in the pipeline stage). **Step 2**: `prepare_2d_dataset_npy_from_raw_video()` -- run MMPose pose estimation, extract court positions via homography, normalize joints by bounding box, save per-clip `_joints.npy`, `_pos.npy`, `_failed.npy`. Shuttle data is intentionally not read here -- keeping this step independent of CSV availability prevents a missing CSV from silently blocking the expensive GPU job. **Step 3**: `collate_npy(taxonomy=..., shuttle_csv_dir=..., resolution_df=...)` -- reads shuttle CSVs from the canonical `ShuttleSet/shuttle_csv/` dir, applies temporal alignment and failed-frame masking, pads all samples to uniform `seq_len`, computes bone vectors and interpolated joints, stacks into single arrays per split. The `taxonomy` parameter (a `Taxonomy` instance from `pipeline.config`) determines the class list for label assignment. MMPose resizes input frames internally (typically 256x192 for RTMPose COCO-17), so video resolution does not affect pose estimation quality beyond ~720p. |

#### Setup

On the HPC nodes the collation output lives on scratch. Set `BST_X_COLLATED_DATA_ROOT` (e.g. `/scratch/comp320a/`) in `.env` and both the collator and `bst_x_train` write/read `<root>/ShuttleSet_data_<taxonomy>/<basename>/` there directly. With the env var unset, both fall back to the in-repo `preparing_data/` convention; if you'd rather keep the data on scratch under that fallback, symlink it:

```bash
# Fallback (no BST_X_COLLATED_DATA_ROOT); replace taxonomy name as needed:
mkdir -p /scratch/comp320a/ShuttleSet_data_une_v1_14
cd ~/badminton_stroke_classification/src/bst_x/stroke_classification/preparing_data
ln -s /scratch/comp320a/ShuttleSet_data_une_v1_14 ShuttleSet_data_une_v1_14
```

If running locally or without scratch, no setup is needed -- the script creates `ShuttleSet_data_{taxonomy}/` and all subdirectories automatically.

**Taxonomy independence of pose data:** Pose data is physically taxonomy-independent. Clip filenames (`{vid}_{set}_{rally}_{ball}`) are physical identifiers -- the same clip produces byte-identical keypoints regardless of which taxonomy folder it sits in. In principle, pose results from one taxonomy can be reused by another via filename matching (the folder structure differs but the data is identical). A future refactor could flatten pose output entirely and defer taxonomy-aware organization to collation.

#### CLI usage

Run from the repo root with both package roots on PYTHONPATH:

```bash
export PYTHONPATH=src/bst_x:src/bst_x/stroke_classification

# Preview what would be done:
python -m preparing_data.prepare_train_on_shuttleset --dry-run

# Common case: shuttle CSVs already exist from the pipeline.
# Run pose only (no shuttle CSV dependency -- can run without them present):
python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-collate

# Then collate (reads shuttle CSVs from ShuttleSet/shuttle_csv/):
python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose

# Point to a non-default shuttle CSV location:
python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose \
    --shuttle-csv-dir /scratch/comp320a/ShuttleSet/shuttle_csv

# Full run including TrackNetV3 shuttle extraction:
python -m preparing_data.prepare_train_on_shuttleset --tracknet-dir /path/to/TrackNetV3
```

Key flags: `--seq-len` (30 or 100), `--taxonomy` (`bst_25`, `bst_24`, `bst_12`, `une_v1_14`, `une_v1_15`, or `shuttleset_18`; legacy names resolve via the alias table), `--collation-id` (required generation tag, e.g. `taxon_pinned_w_preds`), `--split-column` (`split_v2` / `split_bst_baseline`), `--use-3d-pose`, `--skip-trajectory`, `--skip-pose`, `--skip-collate`, `--clips-dir`, `--tracknet-dir`, `--shuttle-csv-dir` (default: `ShuttleSet/shuttle_csv/`), `--dry-run`.

#### Data transformations in detail

1. **Pose detection** (`detect_players_2d`): MMPose extracts 17 COCO keypoints per frame. Players are identified by court projection of their feet -- only the two players whose feet project inside the court boundaries are kept, ordered Top-first by y-coordinate. See [`keypoints_schema.md`](stroke_classification/preparing_data/keypoints_schema.md) for the full joint index map, bone pairs, and JnB representation details.

2. **Joint normalization** (`normalize_joints`): Keypoints are normalized relative to the player's bounding box diagonal. Optionally center-aligned.

3. **Shuttle normalization** (`normalize_shuttlecock`): Shuttle xy divided by video resolution to get [0,1] range. Done at collation time (Step 3). Frames where pose detection failed (recorded in `_failed.npy` by Step 2) have their shuttle coordinates zeroed out. This zeroing is baked into the saved collated `shuttle.npy` -- the model receives pre-zeroed data, not a separate mask. The per-clip `_failed.npy` files preserve the raw boolean mask for debugging or future use, but the source `shuttle_csv/` files are never modified.

4. **Padding and augmentation** (`pad_and_augment_one_npy_video`): Each sample is padded (or strided) to a fixed `seq_len` (30 or 100 frames). Four pose representations are supported; only those passed in `--pose-styles` (default `JnB_bone`) are computed and saved:
   - `J_only`: raw joints `(t, 2, 17, 2)`
   - `JnB_interp`: joints + bone midpoints `(t, 2, 36, 2)`
   - `JnB_bone`: joints + bone vectors `(t, 2, 36, 2)` — **default**, what BST-X training loads
   - `Jn2B`: interpolated joints + bone vectors `(t, 2, 55, 2)`

5. **Collation** (`collate_npy`): All samples in a split are stacked into single arrays and saved:
   - `{pose_style}.npy` (one file per requested style), `pos.npy`, `shuttle.npy`, `videos_len.npy`, `labels.npy`

#### Collated output structure

```
preparing_data/ShuttleSet_data_{taxonomy.name}/npy_[3d_][seq{N}_]{split}_{collation_id}/
  train/
    JnB_bone.npy                                    # default single pose file
    pos.npy, shuttle.npy, videos_len.npy, labels.npy
  val/
    ...
  test/
    ...
```

Passing `--pose-styles J_only,JnB_bone,Jn2B` (etc.) saves the listed styles instead.

For example, `ShuttleSet_data_une_v1_14/`, `ShuttleSet_data_bst_25/`, or `ShuttleSet_data_shuttleset_18/`.

---

### Between Stages 2 and 3 -- Data Quality Validation (`validation_scripts/`)

Before training, run the validation scripts to assess detection quality. Two independent failure modes are invisible at training time and worth quantifying:

1. **MMPose failures** (`_failed.npy`): frames where MMPose couldn't detect exactly 2 players. Joints, court positions, and shuttle coordinates are all zeroed on these frames at collation. The BST-X transformer does **not** mask them -- they participate in attention as zero vectors.

2. **Shuttle detection failures** (shuttle NPY visibility column): frames where TrackNetV3 reported visibility=0. The visibility column is dropped during collation, so these failures become silent (0, 0) shuttle coordinates with no way for the model to distinguish them from a shuttle at the origin.

#### Usage

Run from `src/bst_x/` (MMPose or BST venv -- only needs numpy, matplotlib, pandas):

```bash
# Minimal (MMPose failure stats only):
python validation_scripts/validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_une_v1_14 \
    --taxonomy une_v1_14

# Full (adds flaw cross-reference, hit-frame proximity, shuttle analysis):
python validation_scripts/validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_une_v1_14 \
    --taxonomy une_v1_14 \
    --set-dir ShuttleSet/set \
    --shuttle-npy-dir ShuttleSet/shuttle_npy
```

Optional flags: `--threshold` (flagged-clip cutoff, default 0.5), `--hit-window` (frames either side of hit, default 10), `--taxonomy` (for label derivation and output filenames, default `une_v1_14`).

#### Output

All saved to `validation_scripts/zeroed_frames_analysis_outputs/`:

- **Text report** (`analysis_{taxonomy}_{date}_{time}.txt`): overall/per-split/per-stroke failure rates, tiered clip counts, flaw cross-reference, shuttle detection stats with MMPose x shuttle 2x2 overlap, hit-frame proximity breakdown for both MMPose and shuttle.
- **Figures**: fail rate histogram (log y-axis), temporal pattern by clip position, hit-frame profile (MMPose vs shuttle overlay).

See `validation_scripts/README.md` for full argument and report section documentation.

---

### Stage 3 -- Dataset Loading (`stroke_classification/preparing_data/shuttleset_dataset.py`)

Bridges collated `.npy` files to PyTorch `DataLoader`s. Imports `Taxonomy` from `pipeline.config` for class list construction.

#### Key classes and functions

| Name | Role |
|------|------|
| `Dataset_npy_collated` | Primary Dataset class for BST-X. Loads pre-collated arrays from disk. Supports `train_partial` to use a fraction of training data. Returns `(human_pose, pos, shuttle), video_len, label` per sample. **Filters out zero-length clips at load time** (see known divergence below). |
| `prepare_npy_collated_loaders()` | Convenience function: creates train/val/test `DataLoader`s from a collated directory. |
| `make_seq_len_same()` | Pads or strides a sample to match `seq_len`. Used by `collate_npy`. |
| `create_bones()` / `interpolate_joints()` | Bone vector and midpoint computation from joint arrays. |
| `POSE_BONE_MULTIPLIER` | Dict mapping pose style names to bone-set multipliers: `{'J_only': 0, 'JnB_bone': 1, 'JnB_interp': 1, 'Jn2B': 2}`. Used by train/infer scripts to compute `in_dim`. |
| `pad_class_labels()` | Pads class label strings to uniform width for aligned F1 display. |
| `RandomTranslation` / `RandomTranslation_batch` | Data augmentation: small random xy shifts applied to joint coordinates during training. |

#### Known divergence: zero-length clip filtering

`Dataset_npy_collated` drops clips with `videos_len == 0` at load time.

**Background:** Our automated pipeline processes all clips from ShuttleSet, including degenerate ones where MMPose fails to detect 2 players on every single frame. These clips end up with `videos_len=0` after collation — the entire sample is zero-padded with no real frames. When the transformer builds its padding mask, all positions are masked out, causing `softmax(all -inf) = NaN`, which poisons the loss and the entire training run.

#### Tensor shapes at model input

```
human_pose:  (batch, seq_len, 2, n_pose_features, 2)  ->  flattened to (batch, seq_len, 2, in_dim)
pos:         (batch, seq_len, 2, 2)
shuttle:     (batch, seq_len, 2)
video_len:   (batch,)
labels:      (batch,)
```

#### Loading clip video frames (`pipeline/clip_index.py`)

`Dataset_npy_collated` covers pose + shuttle + position streams from the npy collated dir. For any model that also needs the raw `.mp4` clip frames (Arch 2 3D CNN, Arch 1 wrist crop), the clips directory is still nested as `{split}/{Top,Bottom}_{stroke_type}/*.mp4` (Phase 3 flattening is deferred). Rather than walk the tree per `__getitem__`, use `pipeline.clip_index.build_clip_path_index(clips_dir)` to build a `{clip_stem -> Path}` lookup once at Dataset `__init__`; subsequent per-sample lookup is O(1).

Skeleton showing the CSV-driven pattern (split + label come from `clips_master.csv` with taxonomy applied at init, matching how `collate_npy` builds its npy arrays):

```python
import pandas as pd
from torch.utils.data import Dataset

from pipeline.clip_index import build_clip_path_index
from pipeline.config import CLIPS_OUTPUT_DIR, TAXONOMIES


class ClipVideoDataset(Dataset):
    def __init__(self, clips_csv, split_column, taxonomy_name,
                 split='train', clips_dir=CLIPS_OUTPUT_DIR):
        df = pd.read_csv(clips_csv)
        df = df[df[split_column] == split]
        taxonomy = TAXONOMIES[taxonomy_name]
        self._path_by_stem = build_clip_path_index(clips_dir)
        self.items = [
            (row.clip_stem, _derive_label(row, taxonomy))
            for row in df.itertuples()
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, label = self.items[i]
        return load_video(self._path_by_stem[stem]), label
```

`label_for_row(taxonomy, row.raw_type_en, row.player_side)` produces the int label (or `None` to skip the row), applying the taxonomy's merge map + side rule + exclusions in one place; see `collate_npy` in `prepare_train_on_shuttleset.py` for the canonical reference implementation. The video decoder (`load_video`) is caller's choice — cv2, decord, or torchvision.io. With this pattern the nested `clips/` layout stays transparent: any `split_column` in `clips_master.csv` (e.g. `split_bst_baseline`, `split_v2`) works without reorganizing the clips tree.

For ad-hoc queries or when a Dataset wants a higher-level "give me clip + shuttle + mmpose triples for this split and class" API, `pipeline.data_access.get_clip_records` wraps the CSV read, taxonomy label derivation, and flat-path resolution into one call (and exposes the same thing via CLI / TUI at `python -m pipeline.data_access`). `clip_index.build_clip_path_index` remains the zero-dep pathlib helper it calls internally for clip-stem lookup.

---

### Stage 4 -- Model (`stroke_classification/model/`)

#### Modules

| Module | Role |
|--------|------|
| `tempose.py` | Building blocks reused by BST-X: `TCN` (dilated 1D temporal convolutions), `MLP`, `MLP_Head` (LayerNorm + MLP), `FeedForward` (MLP + Dropout), `MultiHeadAttention`, `TransformerLayer`, `TransformerEncoder`. The four standalone TemPose variants (`TemPose_V`/`PF`/`SF`/`TF`) were excised pre-phase-2 and live verbatim in `scratch/architecture_notes/historical_bst.md` section 1. |
| `bst.py` | The BST-X model. Imports `TCN`, `FeedForward`, `MLP`, `MLP_Head`, `TransformerEncoder` from `tempose.py`. Adds `MultiHeadCrossAttention` and `CrossTransformerLayer` for player-shuttle interaction. Also defines pre-configured variant partials (`BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`, `BST_CG_AP`) — these are the single source of truth for variant flag combinations, imported by the train/infer scripts. |

#### BST-X architecture (forward pass)

1. **PPF (Pose Position Fusion)** -- optional: projects court positions to `in_dim` via MLP, multiplies with skeleton features (multiplicative fusion with residual).
2. **TCN feature extraction**: separate TCNs for pose `(b*n, in_dim, t) -> (b*n, d_model, t)` and shuttle `(b, 2, t) -> (b, d_model, t)`.
3. **Temporal Transformer**: each of the 3 streams (player1, player2, shuttle) gets a learnable CLS token prepended, positional embeddings added, then processed by shared self-attention layers independently. Padding mask prevents attention to zero-padded frames.
4. **Cross Transformer**: each player's frame-level representation attends to the shuttle's representation via cross-attention (player queries, shuttle provides keys+values).
5. **Interactional Transformer**: combines player-shuttle interactions across players with another CLS token and self-attention.
6. **CG (Clean Gate)** -- optional: subtracts shared player noise from shuttle CLS via learned MLP. Scaled by `cg_factor` buffer (see CG/AP warm-start schedule below).
7. **AP (Aim Player)** -- optional: weights player contributions by cosine similarity to shuttle CLS. Alpha multipliers blend toward pass-through via `ap_factor` buffer (see CG/AP warm-start schedule below).
8. **MLP Head**: concatenated CLS tokens -> LayerNorm -> MLP -> class logits.

#### CG/AP warm-start schedule (BST_CG_AP only)

A cosine schedule fades CG and AP out across training so the transformer backbone takes over. The model holds two scalar buffers (`cg_factor`, `ap_factor`, both in `[0, 1]`) that modulate the two optional blocks:

- CG: `shuttle_cls = shuttle_cls - cg_factor * dirt`. At `cg_factor=0` the subtraction vanishes.
- AP: `eff_a_p1 = ap_factor * alpha + (1 - ap_factor)`, `eff_a_p2 = ap_factor * (1 - alpha) + (1 - ap_factor)`. At `ap_factor=0` both multipliers become exactly 1.0 (`p1_conclusion` and `p2_conclusion` pass through unchanged). At `ap_factor=1` the original AP gating is recovered.

The training loop calls `model.set_schedule_factors(cg_factor, ap_factor)` once per epoch with a factor from `aux_schedule_factor(epoch, fade_end_epoch)` (cosine from 1.0 at epoch 1 to 0.0 at `fade_end_epoch`, pinned at 0 after). CG and AP currently share one factor. The buffers are part of `state_dict`, so the best-F1 checkpoint captures whichever value was active at that epoch; `task.test()` runs with those restored values, no override. Controls live in the `hyp` namedtuple in `bst_x_train.py` (see Stage 5).

#### BST variants

```python
BST_0     = BST(use_ppf=False, use_cg=False, use_ap=False)  # Bare transformer
BST_PPF   = BST(use_ppf=True,  use_cg=False, use_ap=False)  # + Pose Position Fusion
BST_CG    = BST(use_ppf=True,  use_cg=True,  use_ap=False)  # + Clean Gate
BST_AP    = BST(use_ppf=True,  use_cg=False, use_ap=True)   # + Aim Player
BST_CG_AP = BST(use_ppf=True,  use_cg=True,  use_ap=True)   # Full model
```

#### Key hyperparameters (defaults from `bst_x_train.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 100 | Hidden dimension throughout |
| `d_head` | 128 | Dimension per attention head |
| `n_head` | 6 | Number of attention heads |
| `depth_tem` | 2 | Temporal transformer layers |
| `depth_inter` | 1 | Interactional transformer layers |
| `drop_p` | 0.3 | Dropout rate |
| `tcn_kernel_size` | 5 | TCN convolution kernel |

---

### Stage 5 -- Training (`stroke_classification/main_on_shuttleset/`)

Stage 5 spans two files:

- `bst_x_train.py` — top-level training loop (`Hyp`, `train_one_epoch`, `validate`, `train_network`, `Task`).
- `bst_x_common.py` — shared scaffolding lifted out by step 5c so `bst_x_train.py` and `bst_x_infer.py` agree on a single source of truth (`MODELS`, `build_bst_x_network`, `Tee`, `compute_data_provenance`).

#### Key components

| Name | Lives in | Role |
|------|----------|------|
| `Hyp` (namedtuple) | `bst_x_train.py` | Active training config, in the `Hyp`/`hyp` block near the top of `bst_x_train.py`.<br>• Schedule: `n_epochs=80`, `early_stop_n_epochs=40`, `warm_up_step=100`, `use_aux_schedule=True`, `aux_fade_end_epoch=15` (compressed warm-start-then-finetune, paired with the CG/AP cosine fade).<br>• Data: `taxonomy='une_v1_14'`, `split_column='split_v2'`, `collation_id='taxon_pinned_w_preds'`, `seq_len=100`, `pose_style='JnB_bone'`, `use_3d_pose=False`, `train_partial=1.0`.<br>• Optim: `batch_size=128`, `lr=5e-4`.<br>• `ablation_id` is a nullable training-time tag, separate from the `collation_id` path tag. `drop_unknown`/`expected_active_classes` were removed in the taxon_pinned_w_preds refactor: `excluded_base_stroke_types` carries the unknown-drop rule and labels.npy lands in active class space.<br>• BST-paper originals (`n_epochs=1600`, `warm_up_step=400`, `early_stop_n_epochs=300`, `taxonomy='merged_25'`, `aux_fade_end_epoch=60`) live verbatim in `historical_bst.md`; current LR + schedule rationale in `bst_x_overview.md`. |
| `train_one_epoch()` | `bst_x_train.py` | Standard PyTorch training loop: forward pass, cross-entropy loss (with label smoothing 0.1), backward, optimizer step, scheduler step. Applies `RandomTranslation_batch` to joints (not bones). |
| `validate()` | `bst_x_train.py` | Evaluates on val set. Accumulates per-class TP/FP/FN across batches, computes macro F1 and min-class F1. |
| `test()` | `bst_x_train.py` (`Task.test`) | Runs inference on test set, returns `(predictions, ground_truth)` tensors. |
| `train_network()` | `bst_x_train.py` | Full training loop with AdamW optimizer, cosine LR schedule with warmup, early stopping on macro F1, and best-checkpoint saving. Applies the CG/AP warm-start schedule at the top of each epoch via `model.set_schedule_factors(cg_factor, ap_factor)`. Logs per-epoch scalars (`Loss/Train`, `Loss/Val`, `F1/Val_macro`, `F1/Val_min`, `Schedule/aux_factor`) plus an end-of-run **HParams** entry: best + 2nd-best macro F1 and min F1 (with their epochs), best val loss (with epoch), and `stopped_epoch`. `stopped_epoch - best/macro_f1_epoch == early_stop_n_epochs` confirms a clean early-stop vs a crash. |
| `Task` (class) | `bst_x_train.py` | Orchestrates the full workflow: `prepare_dataloaders()` -> `get_network_architecture()` -> `seek_network_weights()` (loads existing or trains) -> `test()`. |
| `MODELS` (dict) | `bst_x_common.py` | Maps variant names (`'BST_0'`, `'BST'`, etc.) to pre-configured partials imported from `model/bst.py`. Single dispatch point shared by `bst_x_train.py` and `bst_x_infer.py`. |
| `build_bst_x_network()` | `bst_x_common.py` | Builds the network from `MODELS[name]` and returns `(net, n_bones)`. `n_bones` is the trailing-bone-channel count derived from `pose_style` x `get_bone_pairs()` and is the single source of truth used downstream. |
| `Tee` (class) | `bst_x_common.py` | Duplicates writes across multiple streams (terminal + file). Used by `bst_x_train.py`'s `__main__` to auto-tee test output to `test_logs/test_<timestamp>.log` so test metrics survive a dropped terminal. Training output stays terminal-only (TB has it). |
| `compute_data_provenance()` | `bst_x_common.py` | Hashes `clips_master.csv` + collated-dir naming into the `extra:` block of the manifest so each run is rebindable to its exact data input. |

#### Training flow

```
Task()
  .prepare_dataloaders(root_dir, pose_style, train_partial)
  .get_network_architecture(model_name='BST_CG_AP', in_channels=2)
  .seek_network_weights(model_info, serial_no)   # trains if no checkpoint found
  .test(show_details, show_confusion_matrix)
  .test_topk_acc(k=2)
```

The `__main__` block runs 5 serial trials (`range(1, 6)`) to measure seed variance. Each invocation mints one timestamp and uses it to name both (a) the run folder `experiments/run_<timestamp>/` (holding `manifest.yaml`, `weights/`, and `tb/serial_N/`) and (b) the test log `test_logs/test_<timestamp>.log`, so artefacts for a single invocation line up on disk. All five serials' weights, per-serial TB event dirs, and test output land under that run folder. `Task.test()` and `task.test_topk_acc()` are wrapped in `redirect_stdout(Tee(sys.stdout, log_f))` so test metrics land in both the terminal and the log file. The script is wired into `run_tracker.py` with two function calls (`track_run` + `track_serial`) so the manifest captures hparams + per-serial metrics automatically; see the **Run tracker + aggregator** section below. Set `resume_from = '<run_folder_name>'` at the top of `__main__` to re-test an existing run's weights without retraining; leave it `None` for normal fresh-train behaviour.

#### Outputs

Every invocation writes under `main_on_shuttleset/experiments/<run_id>/`, where `<run_id>` is `run_<timestamp>` on a fresh run or the `resume_from` folder name on a re-test. That folder is the single collection point: manifest + per-serial weights + per-serial TB dirs all live side by side.

- **Manifest** (`experiments/<run_id>/manifest.yaml`): source of truth for hparams, git SHA + host, per-serial metrics (`macro_f1`, `min_f1`, `accuracy`, `top2_accuracy`, `num_strokes`), paths to each serial's weight file and TB dir, plus a `log_path:` pointer back to the matching test log. Tracked in git.
- **Best-model notes** (`experiments/<run_id>/best_model_id.txt`): freeform notes flagging the best-performing serial(s) and the config context, written by hand after eyeballing the test log. Tracked in git alongside the manifest.
- **Model weights** (`experiments/<run_id>/weights/bst_x_..._une_v1_14[_N].pt`): one best-validation-F1 checkpoint per serial. Gitignored by default; `src/bst_x/stroke_classification/.gitignore` carries a per-run tactical `!` unignore for the serial(s) flagged in `best_model_id.txt`, so git history stays small while the best checkpoints are still shareable.
- **TensorBoard logs** (`experiments/<run_id>/tb/serial_N/`): per-serial event directories grouped under one run folder. Launch with `tensorboard --logdir experiments/<run_id>/tb` to see all serials of a run in one view. Each subfolder holds **two** event files: a larger one (60-70 KB) with the per-epoch scalar curves (train/val loss, val macro/min F1, `Schedule/aux_factor`) and a tiny one (~1.6 KB) with the end-of-run HParams summary (best/2nd-best macro F1 and min F1, best val loss, their epochs, `stopped_epoch`). Gitignored.
- **Test logs** (`main_on_shuttleset/test_logs/test_<timestamp>.log`): all serials' test-set output (`=== Serial N (...) ===` headers, macro F1 table, accuracy, top-2 accuracy) auto-captured via the `Tee` class so metrics survive a dropped terminal. One file per script invocation; the run's manifest points at it via `log_path:`. Grep with `grep -E 'Accuracy|macro' test_logs/test_*.log` for a quick summary across runs, or use `run_overview.py` for a proper tabulation.

#### Run tracker + aggregator

Cross-run comparison and the optional Aim UI are handled by the YAML-based tracker at `src/bst_x/run_tracker.py`. `bst_x_train.py` wires it in with two function calls (`track_run` + `track_serial`), so any future training script (Arch 2 3D CNN, or any further extension) can plug in the same way. Full details in [`src/bst_x/run_tracker.md`](run_tracker.md).

- **`run_overview.py`** aggregates every `experiments/<run_id>/manifest.yaml` into one table with mean / stdev / max per metric across serials:
  ```bash
  cd main_on_shuttleset
  python ../../run_overview.py                              # default: experiments/
  python ../../run_overview.py -c n_epochs,use_aux_schedule -m macro_f1,min_f1
  ```
- **`aim_backfill.py`** rebuilds the Aim UI from every manifest + its TB event files: per-epoch curves, per-class final F1, hparams, auto-derived tags (`legacy`, the anneal-regime label, and `best` on the serial whose checkpoint was kept), and each run dated to its `started_at` rather than backfill-import time. Re-running needs `--wipe` (it removes `.aim` and rebuilds from scratch): aim 3.29 can't reopen a stable run hash, and an in-place update bleeds tags between runs. Runs in the tb-viewer venv (aim + tensorboard); `--repo` points at the Aim repo. Filter the kept-checkpoint runs in the UI search bar with `'best' in run.tags`.
  ```bash
  ~/.venvs/tb-viewer/bin/python ../../aim_backfill.py \
      --repo /path/to/.aim_repos/bst --wipe experiments
  ~/.venvs/tb-viewer/bin/aim up --repo /path/to/.aim_repos/bst   # UI at http://localhost:43800
  ```
  The live `track_serial` mirror (above) also writes to Aim when aim is importable, creating a fresh run per call; `aim_backfill.py --wipe` is the canonical, idempotent population.

---

### Stage 6 -- Inference (`stroke_classification/main_on_shuttleset/bst_x_infer.py`)

Lightweight script for loading a trained checkpoint and predicting stroke types. Suitable as a Gradio backend.

| Name | Role |
|------|------|
| `infer()` | Runs the model in eval mode on a DataLoader, returns predicted class indices. |
| `Task` (class) | `prepare_loader()` -> `get_network_architecture()` -> `load_weight()` -> `infer()`. |

---

### Stage 7 -- Results (`stroke_classification/result_utils.py`)

| Name | Role |
|------|------|
| `show_f1_results()` | Displays per-class and macro/min F1 scores as a pandas DataFrame. |
| `plot_confusion_matrix()` | Generates side-by-side precision and recall confusion matrices using matplotlib. |

---

### Full dependency chain (BST-X on ShuttleSet)

```
pipeline/config.py                     # Taxonomy, stroke types, splits, paths, merge map
    |
    v
pipeline/build_dataset.py             # Orchestrates Steps 1-6 (--taxonomy flag)
  -> download_videos.py               # Step 1: yt-dlp download
  -> clip_generator.py                # Steps 3-4: clip extraction + class merge
     -> player_mapping.py             # A/B -> Top/Bottom
  -> verify.py                        # Step 5: sanity checks
  -> shuttle_extractor.py             # Step 6: TrackNetV3 shuttle detection
    |
    v  (produces ShuttleSet/clips/ and ShuttleSet/shuttle_npy/)
    |
preparing_data/prepare_train_on_shuttleset.py  (--taxonomy, --split-column, --collation-id, --clip-npy-dir)
  -> MMPose (2D/3D pose estimation)   # Writes {clip_stem}_*.npy flat
  -> collate_npy(clips_csv, split_column, taxonomy, ...)  # CSV-driven; stacks per collation
    |
    v  (produces preparing_data/ShuttleSet_data_{taxonomy.name}/npy_[3d_][seq{N}_]{split}_{collation_id}/)
    |
validation_scripts/validate_zeroed_frames.py  # Data quality check (optional, pre-training)
  -> validation_scripts/hit_frame_lookup.py   # Hit-frame index derivation from set CSVs
    |
    v
preparing_data/shuttleset_dataset.py  # PyTorch Dataset + DataLoader wrappers
  -> pipeline.config                  # Imports Taxonomy, TAXONOMIES
    |
    v
model/tempose.py                      # TCN, MLP, TransformerEncoder, etc.
model/bst.py                          # BST-X model (imports tempose building blocks)
    |
    v
main_on_shuttleset/bst_x_common.py      # MODELS dispatch, build_bst_x_network, Tee, provenance
main_on_shuttleset/bst_x_train.py       # Training loop (taxonomy in Hyp namedtuple)
main_on_shuttleset/bst_x_infer.py       # Inference from checkpoint
    |
    v
result_utils.py                       # F1 scores, confusion matrices
```

---

## Part 2: Adapting for a Custom (Non-BST-X) Model

### What stays the same

- **The entire `pipeline/` directory.** The pipeline produces labeled video clips and shuttle trajectories. It is model-agnostic -- it doesn't know or care what architecture consumes its output.
- **`pipeline/config.py`** remains the single source of truth for stroke types, class labels, splits, and merge rules. Your custom dataset loader should import from here to stay in sync.
- **`result_utils.py`** works with any model that produces `(predictions, ground_truth)` tensors. `show_f1_results()` and `plot_confusion_matrix()` are architecture-agnostic.

### What changes or may be replaced

#### 1. Data preparation (`prepare_train_on_shuttleset.py`)

This is the most likely point of divergence.

- **If your model operates on raw video** (e.g. a video transformer, 3D CNN, or SlowFast): you can skip pose estimation entirely. Load clips directly from `ShuttleSet/clips/` using a standard video DataLoader. The folder structure already encodes labels via directory names (`{Player}_{stroke_type}`).

- **If your model uses different input features**: you may need different preprocessing. For example, optical flow, different skeleton formats (not COCO-17), or different normalization schemes. Write your own preparation script, but reuse `pipeline.config` for label definitions.

- **If your model uses pose but at different granularity**: the existing `collate_npy()` supports 4 pose styles (J_only, JnB_interp, JnB_bone, Jn2B). If these suffice, you can reuse the collated arrays directly. If not (e.g., you need raw unnormalized keypoints, or a different skeleton topology), modify the preparation step.

#### 2. Dataset class (`shuttleset_dataset.py`)

BST-X's dataset classes return a specific tuple format: `(human_pose, pos, shuttle), video_len, label`.

- **If your model expects different inputs**: write a new Dataset class. Key decisions:
  - Does your model need all 3 input streams (pose, position, shuttle)? BST-X uses all three. TemPose variants use subsets.
  - Does your model handle variable-length sequences internally (e.g. via packed sequences or attention masks), or does it need pre-padded fixed-length input? BST-X uses fixed-length padding + a `video_len` mask.
  - Does your model operate on pre-collated batched arrays, or per-clip files? `Dataset_npy_collated` loads pre-collated arrays into RAM at init; if a future model needs lazy per-clip loading, write a new Dataset (the legacy `Dataset_npy` lazy loader was excised pre-phase-2; the verbatim source is in `scratch/architecture_notes/historical_bst.md` section 4.1).

- **Label list construction**: All class labels are now English. Read `taxonomy.classes` (the authoritative ordered tuple) from any `Taxonomy` in `pipeline.config.TAXONOMIES`, or `resolve_taxonomy(name)` to follow the legacy-alias table. Labels.npy lands in `[0, taxonomy.n_classes)` directly (no runtime active/full remap). Available taxonomies: `'bst_25'`, `'bst_24'`, `'bst_12'`, `'une_v1_14'` (Architecture 1 active), `'une_v1_15'`, `'shuttleset_18'`. To add a custom taxonomy, define it in `pipeline/config.py` (see the `Taxonomy` dataclass and existing instances for the pattern).

#### 3. Model architecture (`model/`)

Replace `bst.py` (and optionally `tempose.py`) with your own architecture.

- **Reusable building blocks from `tempose.py`**: `TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder` are generic components. If your custom model is transformer-based, you can import these directly rather than reimplementing.

- **BST-X-specific components you'd replace**: `MultiHeadCrossAttention`, `CrossTransformerLayer`, and the BST-X `forward()` logic (PPF, CG, AP). These encode BST-X's specific inductive biases about player-shuttle interaction.

- **Input contract**: BST-X's `forward()` expects `(JnB, shuttle, pos, video_len)`. Your model defines its own signature. The dataset class and training loop must agree on this contract.

#### 4. Training script (`bst_x_train.py`)

The training loop is tightly coupled to BST-X's input format and hyperparameters.

- **Reusable patterns**: The overall structure (train/validate/test functions, early stopping, cosine LR schedule, TensorBoard logging, `Task` orchestration pattern) can be adapted.

- **What to change**:
  - The `Hyp` namedtuple values (learning rate, batch size, epochs, etc.)
  - The model construction in `get_network_architecture()` (replace BST-X with your model)
  - The data unpacking in `train_one_epoch()` and `validate()` (the `for (human_pose, pos, shuttle), video_len, labels in loader` destructuring must match your Dataset's return format)
  - The bone-aware augmentation logic (lines 88-95 of `train_one_epoch`) -- this is BST-X-specific

#### 5. Inference script (`bst_x_infer.py`)

Same pattern as training: replace the model construction and data unpacking to match your architecture.

### Summary of divergence points

| Stage | BST-X-specific? | Custom model action |
|-------|--------------|---------------------|
| Pipeline (`pipeline/`) | No | Reuse as-is |
| Pose extraction (`prepare_train_on_shuttleset.py`) | Partially | Replace if your model uses different features (raw video, optical flow, etc.) |
| Dataset class (`shuttleset_dataset.py`) | Yes | Write new Dataset matching your model's input contract |
| Model (`bst.py` + `tempose.py`) | Yes | Replace with your architecture; optionally reuse tempose building blocks |
| Training loop (`bst_x_train.py`) | Yes | Adapt the loop structure; change data unpacking, model init, augmentation |
| Results (`result_utils.py`) | No | Reuse as-is |
