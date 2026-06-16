# Scratch Data Layout (UNE HPC)

Reference for what lives under `/scratch/comp320a/` on engelbart and bourbaki. Snapshot date 2026-05-14. Companion to `scratch/hpc_quickstart.md` (general HPC setup) and `scratch/gpu-access.md`.

Important: `/scratch` is **local to each host, not shared**. Anything written on engelbart isn't visible on bourbaki without an rsync. Cross-node parity is maintained for the canonical raw mmpose extract; per-host scratch state otherwise drifts.

## Top-level layout

From `ls -lah /scratch/comp320a/`:

```
ahalperi/                                       personal scratch space (node-v20 install for yt-dlp JS runtime)
ShuttleSet/                                     raw clips + shuttle outputs (entrypoint for BST_X_CLIPS_DIR + BST_X_SHUTTLE_NPY_DIR)
ShuttleSet_data_merged_25/                      collations on BST original 25-class taxonomy
ShuttleSet_data_une_merge_v1/                   collations on 14-class with-sides taxonomy
ShuttleSet_data_une_merge_v1_nosides/           collations on 14-class no-sides taxonomy (current best)
ShuttleSet_keypoints_clean_sticky_anchor/       apply_heuristic outputs (BST_X_MMPOSE_NPY_DIR)
ShuttleSet_keypoints_raw/                       raw_extract.py outputs (Phase-2 unified extract)
ShuttleSet_keypoints_raw_provenance/            stems-list files from shard runs (verification artifact, not for re-rsync)
sticky_anchor_inspection/                       inspection artifacts
```

## Clip videos (`ShuttleSet/clips/`)

Layout: `<split>/<Side>_<class>/<stem>.mp4`

- **Splits:** `train`, `val`, `test`.
- **Side:** `Bottom` or `Top`.
- **Class:** one of the 25 raw classes (the original ShuttleSet taxonomy; merging into 14-class no-sides happens at the CSV / label level, not on disk).

Example: `/scratch/comp320a/ShuttleSet/clips/test/Bottom_smash/35_1_10_17.mp4`.

`BST_X_CLIPS_DIR` points at the `clips/` root. The FE serving layer joins it with `clip_index.json`'s `video_path` field to stream individual mp4s.

## Collation dirs

Every registerable collation carries the five canonical tensors per split (train/val/test):

| File | Shape | Dtype | What |
|---|---|---|---|
| `JnB_bone.npy` | `(n, 100, 2, 36, 2)` | float32 | 17 joints + 19 bones per player per frame |
| `pos.npy` | `(n, 100, 2, 2)` | float32 | Court-normalised player position xy |
| `shuttle.npy` | `(n, 100, 2)` | float32 | Shuttle xy per frame |
| `videos_len.npy` | `(n,)` | int64 | Real frame count before padding |
| `labels.npy` | `(n,)` | int64 | Class index into the active taxonomy |

Some collations carry alternate-encoding extras (`Jn2B.npy`, `JnB_interp.npy`, `J_only.npy`, `shuttle_missing.npy`) from earlier experiments. `Dataset_npy_collated` silently ignores anything it doesn't read by name. This is the mechanism the X3D-S wrist-crop variant relies on when it lands.

### ShuttleSet_data_merged_25 (BST original 25-class)

| Collation dir | Size | Notes |
|---|---|---|
| `dataset_npy_collated_between_2_hits_with_max_limits_seq_100/` | 7.3 GB | Default-named older collation. Carries alternate-encoding extras. |
| `dataset_npy_collated_between_2_hits_with_max_limits_seq_100_merged_25_split_bst_baseline_keepunk/` | 7.3 GB | Keeps unknown class. Carries alternate-encoding extras. |
| `npy_merged_25_split_bst_baseline_dropunk/` | 1.8 GB | Drops unknowns. BST-25 baseline registry candidate, pending re-extract against the cleaned keypoints. |

### ShuttleSet_data_une_merge_v1 (14-class with sides)

| Collation dir | Size | Notes |
|---|---|---|
| `dataset_npy_collated_between_2_hits_with_max_limits_seq_100_une_merge_v1_split_bst_baseline_dropunk/` | 1.8 GB | BST baseline split, drops unknowns. Long-form name (auto-generated). |
| `dataset_npy_collated_between_2_hits_with_max_limits_seq_100_une_merge_v1_split_v2_dropunk/` | 1.9 GB | split_v2, drops unknowns. Long-form name. |
| `npy_une_merge_v1_split_v2_dropunk/` | 1.9 GB | Short-form. split_v2, drops unknowns. |
| `npy_une_merge_v1_split_v2_dropunk_h_sticky_anchor/` | 1.9 GB | Same as above with sticky anchor variant. |

### ShuttleSet_data_une_merge_v1_nosides (14-class no sides; current best taxonomy)

| Collation dir | Size | Notes |
|---|---|---|
| `npy_mask_wiring/` | 1.9 GB | Mask-wiring ablation. Carries `shuttle_missing.npy`. |
| `npy_une_merge_v1_nosides_split_v2_dropunk/` | 1.9 GB | Base no-sides collation. |
| `npy_wipe_drop/` | 1.9 GB | **Current project best.** Wipe-drop ablation. |

## Pose outputs (`ShuttleSet_keypoints_*`)

Per-clip pose data, taxonomy- and split-agnostic. Lives outside the per-taxonomy `ShuttleSet_data_*` trees because the same pose extract feeds every collation.

`ShuttleSet_keypoints_raw/` (9.3 MB metadata, contents in TBs): 5 `_raw_*.npy` files per stem, N_max=16. 32,203 stems total. Phase-2 unified extract finished 2026-04-29 (30,487 freshly extracted + 1,716 Phase-1 backfill, bit-identical on engelbart and bourbaki). `ndet=0` 0%, `ndet=1` 0.53% (irreducible per-frame failure floor), modal `ndet=10`.

`ShuttleSet_keypoints_clean_sticky_anchor/` (8.5 MB metadata, contents in GBs): 3 files per stem (`_pos.npy`, `_joints.npy`, `_failed.npy`) from `apply_heuristic.py`. This is what `BST_X_MMPOSE_NPY_DIR` points at and what collation reads as input.

`ShuttleSet_keypoints_raw_provenance/`: stems-list shard files. Keep out of the canonical raw dir during verification queries and rsyncs.

## Env vars

`~/badminton_stroke_classifier/.env` (NFS-shared via /home across nodes; read at import time by `pipeline.data_access`):

```
BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips
BST_X_SHUTTLE_NPY_DIR=/scratch/comp320a/ShuttleSet/shuttle_npy_flat
BST_X_MMPOSE_NPY_DIR=/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor
BST_X_CLIPS_CSV=/home/ahalperi/badminton_stroke_classifier/notebooks/clips_master.csv
# Optional: override the TrackNetV3 shuttle CSV directory. If unset, the
# collator falls back to the repo-rooted SHUTTLE_CSV_DIR from pipeline/config.
BST_X_SHUTTLE_CSV_DIR=/scratch/comp320a/ShuttleSet/shuttle_csv
```

New env var for the FE serving contract (introduced in `frontend_integration_guide.md`):

```
BST_X_COLLATED_DATA_ROOT=/scratch/comp320a/
```

Registry entries in `docs/models_registry.yaml` encode `collated_dir` as relative paths under this root (e.g. `ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop`). `preparing_data/prepare_train_on_shuttleset.py` also reads this when constructing the collation output root (`<BST_X_COLLATED_DATA_ROOT>/ShuttleSet_data_<taxonomy>/npy_<ablation_id>/`); if unset, the collator falls back to the in-repo `preparing_data/ShuttleSet_data_<taxonomy>/` for local dev.

## Regenerating this inventory

If you want to refresh the layout block, run on engelbart or bourbaki:

```bash
ls -lah /scratch/comp320a/

find /scratch/comp320a -maxdepth 7 -name 'JnB_bone.npy' \
    -printf '%h\n' 2>/dev/null | xargs -n1 dirname | sort -u

find /scratch/comp320a -maxdepth 7 -name 'JnB_bone.npy' \
    -printf '%h\n' 2>/dev/null | xargs -n1 dirname | sort -u | \
    while read d; do du -sh "$d" 2>/dev/null; done

find /scratch/comp320a -maxdepth 7 -name 'JnB_bone.npy' \
    -printf '%h\n' 2>/dev/null | xargs -n1 dirname | sort -u | \
    while read d; do
        echo "--- $d"
        find "$d" -mindepth 2 -maxdepth 2 -type f -printf '%f\n' 2>/dev/null | sort -u
    done
```

The four blocks give the top-level layout, the list of collation parent dirs, their sizes, and the per-split file sets respectively.
