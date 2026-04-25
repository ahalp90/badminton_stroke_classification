# MMPose Extraction: Heuristic Investigation

## Rationale

The BST-inherited `prepare_train_on_shuttleset.py` filter zeroes a frame entirely if either player's ankle midpoint fails to project inside the soft court rectangle (`eps = 0.01` margin) or if MMPose detects fewer than 2 people. For an airborne smash, the acting player's projected feet drift well past the back baseline. The Padel paper's `H_z * tan(θ)` geometry (Javadiha et al. Sensors 2021) puts the projection error at ~4.64x amplification at the far court edge, so a 0.5-0.7 m jump pushes feet ~0.17-0.24 normalised units off court. The model sees a zero vector at the single most informative moment of the clip.

`validate_zeroed_frames.py` on the merged_25 flat dir confirms the hypothesis: smash 13.75% pooled fail rate (Top_smash 24.33% stratified), wrist_smash 9.93%, both far above the < 10% baseline of every other class. Top-side dominant throughout, Bottom-side rare, consistent with far-camera projection-error amplification. Unknown class is excluded from scope (54.79% fail rate by construction; garbage class).

Two measurable goals, both required for a heuristic change to ship:

1. Zeroing rate drops by >= 25% relative on the target classes (smash, wrist_smash, clear, drop, long_service, return_net, both Top and Bottom variants).
2. Min-F1 lifts by >= 0.04 on retrained V4 vs the committed baseline (V4 best-serial currently 0.432; target >= 0.47 net).

## TL;DR: how `sticky_anchor` works

MMPose returns a list of person detections per frame (players, chair umpire, line judges, audience members that happen to be clearly visible). We need to pick two of them as Top and Bottom. Instead of trying to filter out non-players up front, we pick by **geometry**:

- Each slot has an **anchor** fixed at the middle of its court half (Top's anchor is the middle of the top half, Bottom's is the middle of the bottom half). The anchor is 75% that fixed point and 25% a running average of recent picks for this slot. The fixed part keeps the anchor from wandering off to capture a wrong person; the running part lets it lean slightly toward where the player has actually been.
- For each slot we pick the detection whose projected foot position is closest to that slot's anchor. Bottom picks first (its detections are bigger and more confident), then Top picks from what's left.
- Candidates that sit closer to the OTHER slot's anchor are excluded from this slot's pool, so the two slots can't steal each other's player.
- If the closest candidate is too far away, or if both slots' picks land wildly off court, the slot (or both slots) zeroes for that frame.
- When two candidates are similarly close to an anchor, we use two tiebreakers: drop anyone who looks seated (based on where the knees sit relative to the torso axis) and prefer the larger bounding box.

Why this beats the existing pipeline's filter: the current code rejects a whole frame if a player's projected feet don't land inside the taped court, which kills smash frames because airborne feet project well past the back baseline. The new design keeps those picks as long as they're the clear closest-to-anchor candidate; it only refuses to let off-court picks update the running average (so the anchor can't drift to a place the player isn't actually standing).

The heuristic runs on the raw MMPose output stored on disk; the expensive MMPose extraction runs once and we iterate heuristic variants cheaply on top. Output files match the existing `_pos / _joints / _failed` schema so collation and training code downstream don't change.

**Terminology note**: throughout this doc, "foot position" / "projected feet" / "projected ground position" all refer to the **bbox bottom-centre** ((x1+x2)/2, y2 of the detection's bounding box), projected through the homography. This differs from the BST original's `detect_players_2d`, which projects the COCO ankle-midpoint (joints 15 and 16). See "Projection anchor: bbox bottom-centre, not ankle midpoint" in the Design rationale for the divergence and the asymmetric trade-off it implies.

## Current status (2026-04-25)

Phase 0, Phase 1 raw extract, heuristic implementation, and Phase 1 mixed retrain are all complete. The decision gate failed; a per-class frame-zeroing audit then ruled out the data-quality-bottleneck hypothesis empirically. Phase 2 is deprioritised (not killed). Focal loss is the next experiment under `arch_1_directions.md` rather than this doc.

- **Raw extract** (1,716 hit-zone-busted clips, `N_max = 16`) at `_flat_raw_phase1/`. 0.79% of frames hit the cap; sufficient.
- **`sticky_anchor` implemented and run** on the full 1,716. Output at `_flat_h_sticky_anchor/`. Wall time 54 s on engelbart.
- **Headline**: 1,631 of 1,716 clips (95.05%) are now perfectly clean (zero zeroed frames). The hit-zone busted-clip count under the `fail_rate > 0.50` threshold dropped from 1,716 to 61.
- **Per-split reduction**: train 110 -> 47 (-57%), val 49 -> 6 (-88%), test 33 -> 8 (-76%).
- **Byte-identity gate** (`current` heuristic vs committed extract) passed 50/50 on the deterministic sample; bit-exact on `_failed`, max abs diff = 0 on `_pos` and `_joints`.
- **Visual inspection**: 9/10 still-busted residuals are genuinely irrecoverable (extreme close-ups, side-on framings, cutaways with no useful in-court candidates).
- **One residual investigated in detail (19_2_10_7)**: cause is upstream MMPose detection-layer gap under heavy occlusion, not Voronoi crossover. Per-frame replay + image inspection on 2026-04-25; details in the Failure modes section.
- **Net-crossover zeroing is a mathematically possible failure, but unobserved**: no clip in the inspected residuals actually exhibits Voronoi-induced zeroing. The 3.5% upper-bound estimate in earlier writeups assumed this was the cause on 19_2_10_7; that attribution was wrong.
- **Phase 1 mixed retrain done** (`run_20260425_150548`): decision gate failed on `Top_wrist_smash` (-0.057 mean vs V4 baseline) while macro / acc / top-2 each lifted by ~0.007. Best S3 macro 0.755, min 0.352, acc 0.780.
- **Per-class frame-zeroing audit done** (`src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py`, output at `analysis_outputs/zeroed_frames_class_audit__run_20260425_150548.{txt,csv}`): the F1-bottom classes are not the heavily-zeroed ones; the worst-zeroed class has near-perfect F1. The data-quality-bottleneck hypothesis for the F1 floor is empirically dead.
- **Recovery routes parked for Phase 2**: gap-fill post-processing (interpolate over MMPose detection gaps) and homography-fail X3D-S-only rescue (independent stream when court coords are unusable). Neither is currently scoped.

## Decoupling: raw extract plus post-processing

Running MMPose on 33k clips takes ~50 hr on V100. Running it multiple times to iterate heuristic variants is a non-starter. The heuristic itself (the pick + filter logic) is pure CPU arithmetic on already-computed keypoints and costs sub-millisecond per clip. Decoupling means one expensive MMPose pass per clip, then fast (~seconds) heuristic iteration.

**Step 2 raw output** (per clip, new files alongside existing ones):

| File | Shape | Contents |
|---|---|---|
| `{stem}_raw_kps.npy` | `(F, N_max, 17, 2)` | All detected people's keypoints per frame, NaN-padded to `N_max`. |
| `{stem}_raw_bboxes.npy` | `(F, N_max, 4)` | All detected bounding boxes, NaN-padded. |
| `{stem}_raw_scores.npy` | `(F, N_max)` | Per-person detector confidence. |
| `{stem}_raw_kp_scores.npy` | `(F, N_max, 17)` | Per-joint MMPose confidence. Used for ankle-confidence-based projection fallback; preserved in full for possible training-side use. |
| `{stem}_raw_ndet.npy` | `(F,)` int8 | Number of people detected per frame. `raw_kps[f, :ndet[f]]` is the valid slice. Also acts as the resume marker, saved last by `raw_extract.py`. |

`N_max = 16` after the Phase 1 measurement (87% of the first 222-clip extract triggered the original `N_max = 8` cap because busted clips over-represent crowded frames; at 16 only 0.79% hit it on the full 1,716 set). Per-clip raw-output storage at N=16 is ~320 KB; for the 1,716 hit-zone subset the total is well under 1 GB. The 3D variant (`detect_players_3d` in the current code) is omitted from this pass; `raw_extract.py` includes commented scaffolding for toggling it back on.

**Post-processing output** (per clip, heuristic-specific subdir): `_pos.npy`, `_joints.npy`, `_failed.npy`. Same shapes as the committed extract. Collation is unchanged downstream.

**Step ordering**:
1. Step 2 (raw extract): writes `*_raw_*.npy` per clip to a new flat dir. Expensive GPU step.
2. Step 2.5 (apply heuristic): reads raw, applies a named heuristic, writes the `_pos/_joints/_failed` triple to a heuristic-specific output dir. Fast (seconds per clip), re-runnable per heuristic variant.
3. Step 3 (collate): unchanged. Points at either the original flat dir or a heuristic-processed dir via `--clip-npy-dir`.

## Usage

### Directory conventions

Strict separation: raw extracts and the primary committed filtered extract are never overwritten. Paths are referenced via the `.env` convention Curtis established for `pipeline.data_access` (`.env.example` at the repo root, `pipeline/data_access.py`). Relevant variable:

```
BST_MMPOSE_NPY_DIR=/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat
```

Per-clip flat dirs on engelbart, all under the same parent:

```
{parent_dir}/
  dataset_npy_between_2_hits_with_max_limits_flat/                  # primary committed, read-only (= $BST_MMPOSE_NPY_DIR)
  dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/       # raw N=16 extract, read-only
  dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1_n8/    # historical N=8 raw, read-only
  dataset_npy_between_2_hits_with_max_limits_flat_failsafe_gate/    # byte-identity gate output, scratch
  dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor/  # written by apply_heuristic
```

The `_h_<heuristic>` suffix extends the existing flat-dir naming consistent with the `_raw_phase1` extensions.

`apply_heuristic.py` refuses to write unless `--output-dir` is distinct from both `--raw-dir` and `BST_MMPOSE_NPY_DIR`. Two-line guard against typos destroying data we can't cheaply recompute (1,716 clip re-extract is ~20 min V100 time; the committed extract is the baseline for every comparison).

**Downstream collated dir** (for the Phase 1 mixed re-train, produced by Step 3 in `prepare_train_on_shuttleset.py`): post-2026-04-21 short naming convention.

- Parent: `ShuttleSet_data_{taxonomy}/` under the preparing-data root (on engelbart, `/scratch/comp320a/ShuttleSet_data_une_merge_v1/`).
- Dir name: `npy_[3d_][seq{N}_]{ablation_id}`, where `ablation_id = {taxonomy}_{split_column}_{drop}` by default.
- For the V4-analog mixed re-train: `npy_une_merge_v1_split_v2_dropunk_h_sticky_anchor`.
- For the V3-analog mixed re-train: `npy_merged_25_split_bst_baseline_keepunk_h_sticky_anchor` under `ShuttleSet_data_merged_25/`.

Existing flat dirs on scratch already match the current naming convention. The new short naming applies only to collated dirs, produced fresh by collation and tagged with an ablation_id suffix per config. Older long-named collated dirs referenced by V3/V4 manifests stay untouched unless those manifests are also being rewritten.

### Byte-identity gate

`failsafe_bst_mmpose_zeroing_check_equivalence.py` lives alongside `apply_heuristic.py`. Run it before trusting any `sticky_anchor` output:

- Sample 50 clip stems from `scratch/architecture_notes/busted_hit_zone_clips_phase1.txt`. Lex-sort, take every `len // 50`-th stem. Deterministic, no seeding. Draws from the busted list rather than `clips_master.csv` because raw extracts only exist for those 1,716 stems.
- Run `apply_heuristic.py --heuristic current` on those stems against the raw extract, writing to `..._flat_failsafe_gate/`.
- For each stem's three output arrays, compare against `$BST_MMPOSE_NPY_DIR`:
  - `np.array_equal` on `_failed.npy` (bool).
  - `np.allclose(rtol=0, atol=1e-5)` on `_pos.npy` and `_joints.npy` (float; tolerance absorbs float32 projection-chain non-associativity).
- On any mismatch: stop and investigate plumbing before trusting `sticky_anchor`. Usual suspects: keypoint-index ordering, bbox row order when multiple on-court people exist, `normalize_joints` vs `normalize_position` step order, resolution-scale application.

Canonical gate command (run from `src/bst_refactor/stroke_classification/`):

```
python -m preparing_data.failsafe_bst_mmpose_zeroing_check_equivalence \
    --raw-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1 \
    --busted-stems-file scratch/architecture_notes/busted_hit_zone_clips_phase1.txt \
    --clips-csv notebooks/clips_master.csv \
    --scratch-output-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_failsafe_gate
```

`--committed-dir` is auto-detected from `$BST_MMPOSE_NPY_DIR` when unset. `apply_heuristic` side-effect-imports `pipeline.data_access`, which auto-loads the repo-root `.env`, so the collision guards fire without a prior shell export.

### Apply heuristic (canonical run)

```
python -m preparing_data.apply_heuristic \
    --raw-dir /scratch/comp320a/.../dataset_npy_..._flat_raw_phase1 \
    --output-dir /scratch/comp320a/.../dataset_npy_..._flat_h_sticky_anchor \
    --heuristic sticky_anchor \
    --clips-csv notebooks/clips_master.csv
```

Hyperparameters expose as CLI args; defaults in the Hyperparameters section below.

### Mixed retrain plumbing (Phase 1)

After `sticky_anchor` runs on the full 1,716 clips:

- Build a symlink-merged flat dir at `$BST_MMPOSE_NPY_DIR/../dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor_phase1_merged/`.
- For each stem in `clips_master.csv` with `split_v2 in ('train','val','test')`:
  - If stem is in `busted_hit_zone_clips_phase1.txt`: symlink the three `sticky_anchor` outputs from `..._flat_h_sticky_anchor/`.
  - Otherwise: symlink from `$BST_MMPOSE_NPY_DIR`.
- Collate via `python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose --clip-npy-dir <merged_dir>` with the Phase 1 ablation_id (`npy_une_merge_v1_split_v2_dropunk_h_sticky_anchor` under `ShuttleSet_data_une_merge_v1/`).
- Retrain V4 via `bst_train.py` pointing at the new collated dir. 5 serials, same hyperparameters as the committed V4 run (`run_20260420_171101/`).

**Decision gate**: conjunction of (a) >25% relative reduction in zeroing rate on target classes, (b) no >5% relative regression on non-target classes, (c) >=0.02 min-F1 lift on target-class aggregate OR >=0.005 macro-F1 lift overall. All measured against committed V4.

## Algorithm specification

Per-video setup (once per clip, using the homography):

- `halfcourt_centre[TOP] = ((bL + bR) / 2, bU + (bD - bU) / 4)` normalised.
- `halfcourt_centre[BOTTOM] = ((bL + bR) / 2, bU + 3 * (bD - bU) / 4)` normalised.
- `bL`, `bR`, `bU`, `bD` are court borders from `pipeline.court_utils.get_court_info`.
- On ShuttleSet the canonical rectangle collapses these to (0.5, 0.25) and (0.5, 0.75). For amateur data they derive from whatever canonical rectangle that video's homography defines, so the formula is already data-adaptive.
- Initialise `ema[TOP] = halfcourt_centre[TOP]` and `ema[BOTTOM] = halfcourt_centre[BOTTOM]`.

Per-frame algorithm:

**A. Build candidate pool (once per frame):**

1. Filter raw detections to those with `bbox_score > score_filter` (default 0.2).
2. For each surviving detection, project its bbox bottom-centre through the homography to normalised court coords. Store as `candidate.court_base_pos`.

**B. Compute both effective anchors (once per frame, before either slot's pick):**

3. For each slot `s` in `(BOTTOM, TOP)`: `effective_anchor[s] = 0.75 * halfcourt_centre[s] + 0.25 * ema[s]`.
4. For each candidate and each slot, compute `D(candidate, s) = euclidean(candidate.court_base_pos, effective_anchor[s])`.

**C. Process each slot, Bottom first then Top.**

For `s` in `(BOTTOM, TOP)` with `other = the other slot`:

5. Pre-filter the candidate pool for this slot:
   1. Drop candidates with `D(candidate, s) > sanity_ceiling` (default 0.6 normalised).
   2. Drop candidates that are closer to the OTHER slot's anchor than to this slot's own anchor (`D(candidate, other) < D(candidate, s)`). In other words, each candidate is only eligible for whichever slot's anchor it is closer to. Prevents cross-half capture when the other slot's player happens to sit geometrically closer to this anchor than our own player does. (Voronoi partition; referred to below as the closer-to-own-anchor rule.)
   3. If `s == TOP`, also drop whichever candidate BOTTOM already assigned.
6. If no candidates survive: mark slot `s` as zeroed. Go to next slot.
7. Otherwise `winner = argmin D(candidate, s)` among survivors.
8. Tiebreaker: if any other surviving candidate has `|D(candidate, s) - D(winner, s)| < tiebreaker_tol` (default 0.05):
   1. Among the tied set plus the winner, drop candidates where `is_sitting(candidate) == True`.
   2. Among survivors of (i), pick the one with the largest bbox area.
   3. If (i) dropped everyone, revert to the original `argmin D` pick.
9. Mark slot `s` as picked = winner.

**D. Rally-presence check (after both slots processed):**

10. If both slots are picked but neither pick's `court_base_pos` is within `[-generous_margin, 1 + generous_margin]` on both axes (default margin 0.15), mark both slots as zeroed.

**E. Write outputs and update EMA per slot:**

11. For each slot `s`:
    - If zeroed: write zeros to `_pos[f, s]` and `_joints[f, s, :, :]`. Reset `ema[s] = halfcourt_centre[s]`.
    - If picked: write `_pos[f, s] = winner.court_base_pos`. Write `_joints[f, s, :, :] = normalize_joints(winner.keypoints, winner.bbox)` via the existing helper. If `winner.court_base_pos` is within `[-update_gate_eps, 1 + update_gate_eps]` on both axes (default 0.01), update `ema[s] = 0.1 * winner.court_base_pos + 0.9 * ema[s]`. Otherwise EMA stays.
12. `_failed[f] = True` if either slot was zeroed this frame, otherwise False.

### Body-frame sitting test (used in step 8.i)

```python
sh = (kp[5] + kp[6]) / 2         # shoulder centre
hp = (kp[11] + kp[12]) / 2       # hip centre
kn = (kp[13] + kp[14]) / 2       # knee centre
body_up = sh - hp
torso_len_sq = body_up @ body_up
if torso_len_sq < 1e-6:
    return False                 # degenerate pose; defer to anchor distance
knee_vec = kn - hp
body_frame_ratio = (knee_vec @ body_up) / torso_len_sq
return body_frame_ratio > sitting_threshold   # default -0.3
```

Projects the knee-offset-from-hip onto the hip-to-shoulder axis. Asks "are the knees in the body's down direction (standing / airborne / active) or perpendicular to the body axis (sitting)?" in image-pixel coordinates. No confidence gates.

## Hyperparameters

All exposed as `apply_heuristic` CLI args; ShuttleSet defaults shown.

| Param | Default | What it governs |
|---|---|---|
| `prior_weight` | 0.75 | `halfcourt_centre` vs EMA weighting in `effective_anchor` |
| `ema_alpha` | 0.1 | EMA update rate (effective half-life ~7 frames) |
| `sanity_ceiling` | 0.6 | pre-filter max anchor distance for a candidate |
| `generous_margin` | 0.15 | rally-presence envelope |
| `score_filter` | 0.2 | candidate-pool cutoff on `bbox_score` |
| `tiebreaker_tol` | 0.05 | distance tolerance invoking the sitting + area tiebreaker |
| `sitting_threshold` | -0.3 | `body_frame_ratio` cutoff |
| `update_gate_eps` | 0.01 | EMA update in-court gate |

Per-video / per-camera tuning of `sanity_ceiling`, `generous_margin`, and `score_filter` is deferred to the amateur-generalisation work.

## Output schema

Matches the existing pipeline so `collate_npy` reads sticky_anchor output unchanged:

- `_pos.npy`: `(F, 2, 2)` normalised court positions per slot, ordered (TOP, BOTTOM).
- `_joints.npy`: `(F, 2, 17, 2)` bbox-diagonal-normalised keypoints per slot.
- `_failed.npy`: `(F,)` bool, True where either slot was zeroed this frame.

## Court-space geometry calibration (2026-04-22)

Empirical findings from a code + CSV audit of `ShuttleSet/set/homography.csv` plus a per-frame overlay inspection of clip `3_1_18_3`. Directly informs `sticky_anchor`'s buffer hyperparameters.

### What the homography is calibrated to

All 44 videos in `homography.csv` project their annotated 4 corners (`upleft_x/y` ... `downright_x/y`) to an identical canonical rectangle in court-space: **300 wide x 660 tall**. UL=(25, 150), UR=(325, 150), DL=(25, 810), DR=(325, 810). Length/width ratio = 660/300 = **2.2000**.

| Rectangle | Dimensions (m) | L/W ratio | Match? |
|---|---|---|---|
| Full doubles court (outer taped) | 6.10 x 13.40 | 2.1967 | **Yes (3 d.p.)** |
| Singles court (inner taped) | 5.18 x 13.40 | 2.5869 | No |
| BWF run-off zone (international minimum, 1m sides + 2m ends) | 8.10 x 17.40 | 2.148 | No |

The annotation target is the outer (doubles) taped court. No "further taped line" or run-off rectangle is involved. Scale: 300 units to 6.10 m so one court-space unit is ~2.03 cm; the normalised [0, 1] interval spans the full outer doubles rectangle.

### Visual confirmation on clip `3_1_18_3` (video id 3)

Overlay PNG at `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/homography_overlay_3_1_18_3_f032.png` (frame 32, top player mid-smash). The cyan rectangle (annotated corners, scaled from 1280x720 homography resolution up to the clip's 1920x1080 resolution) sits exactly on the outer doubles taped lines. A derived orange pair (doubles-sidelines minus 7.54% inset) lands precisely on the visible singles sidelines, independently verifying the annotations are on the outer taped lines.

Implied singles-sideline normalised x coordinates: **x = 0.0754** and **x = 0.9246** (since (6.10 - 5.18) / 2 / 6.10 = 0.0754). Singles play occupies ~85% of the horizontal [0, 1] range; the outer ~7.5% on each side is the doubles tramline.

### The original `eps = 0.01` buffer is effectively zero

`check_pos_in_court` in `pipeline/court_utils.py:166` (mirrored in `prepare_train_on_shuttleset.py:230`) tests `-eps < x,y < 1 + eps` with `eps = 0.01`. Converted to physical units against the canonical rectangle:

| Axis | Normalised eps | Physical buffer |
|---|---|---|
| Horizontal (beyond doubles sideline) | 0.01 | **6.1 cm** |
| Vertical (beyond baseline) | 0.01 | **13.4 cm** |

### Observed real overflow on the `3_1_18_3` overlays

From visual inspection of frames 0, 25, 28, 30, 32, 35, 49:

| Scenario | Approximate offset past the doubles line |
|---|---|
| Neutral stance, feet on baseline | 0 |
| Retreat for smash setup, feet behind baseline | **50-100 cm** |
| Airborne at peak smash (player centre) | **75-150 cm** past baseline |
| Airborne peak smash (projected position, inflated by `H_z * tan(θ)` from Padel geometry) | additional **70-170 cm** beyond body centre offset |
| Hard lunge past doubles sideline | **30-80 cm** |

The original `eps = 0.01` buffer is roughly 1/8 to 1/20 of the standing-behind-baseline offset, and an even smaller fraction of the inflated projected-position offset during airborne smashes. Any detection where the player is standing behind the baseline (typical smash setup) is rejected by the original filter.

### Buffer size that is actually needed

- **Observed maximum** on `3_1_18_3`: ~150 cm past baseline (airborne peak, before projection amplification). The projected position under airborne amplification can go further, up to ~300 cm effective displacement at the far edge for a 0.7 m jump.
- **BWF international-competition minimum run-off**: 2 m back / 1 m sides. Any legitimately-in-play stance lies inside this envelope.
- **`sticky_anchor`'s `generous_margin = 0.15`**: ~91.5 cm horizontally / ~2.01 m vertically. Matches BWF run-off on both axes. Covers every observed offset on `3_1_18_3` with headroom.

### Implications for `sticky_anchor` hyperparameters

- **`generous_margin = 0.15`** is defensible and shouldn't be widened without fresh evidence. Matches BWF run-off and covers all observed offsets.
- **`eps = 0.01`** is retained only as the EMA update gate inside `sticky_anchor` (step 11), not as a pick-time filter. In that role it correctly prevents EMA pollution by clearly-off-court picks.
- **`sanity_ceiling = 0.6`** comfortably exceeds the worst legitimate airborne projection offset, so the ceiling is not the binding constraint for well-behaved smashes. Widened from the original 0.5 after observing 0.51 anchor-distance on the apex-jump frame of `16_1_42_4`.
- The ~7.5% doubles-tramline region either side of the playing area is in-bounds per the homography, so picks that land there are accepted; only picks well outside the doubles lines (beyond 0.15 either way) trigger the rally-presence check.

## Design rationale and decision log

Captures reasoning behind each non-obvious choice. Intended as a reference for writing up the design in the report; each entry leads with the decision, then the alternative considered, then the empirical or geometric evidence that settled it.

**N_max = 16** (raised from the original 8).

The first raw extract at N_max = 8 saw over-detection warnings on 193 of 222 clips (87%). The busted subset over-represents crowded-frame clips because those are exactly where the original heuristic rejected too much. At N_max = 16, only 0.79% of frames hit the cap. Storage was already trivial (~160 KB per clip at N=8, doubles at N=16).

**Homography is calibrated to the full outer (doubles) taped court.**

Established by code + CSV audit plus visual overlay on `3_1_18_3`. All 44 ShuttleSet videos map their 4 annotated corners to an identical canonical rectangle of 300 by 660 in court space. Length/width ratio 2.2000 matches physical doubles court (6.10 m by 13.40 m, ratio 2.1967) to three decimal places. Singles (2.587) and BWF run-off (2.148) ratios are ruled out. Load-bearing for every threshold in the algorithm.

**The original `eps = 0.01` filter rejects legitimate play.**

`eps = 0.01` normalised translates to 6.1 cm off the doubles sideline and 13.4 cm off the baseline. Competitive singles players routinely stand 50-100 cm behind the baseline for smash setups (observed across frames 25, 28, 30, 32, 35, 49 of `3_1_18_3`) and lunge 30-80 cm past the doubles sidelines on defensive reaches. Under airborne projection error (Padel `H_z * tan(θ)` for 0.7 m jump at far edge), foot projections amplify another 70-170 cm off-court. The original filter is roughly 1/8 to 1/20 of real-play overflow. Motivated widening `sanity_ceiling` from 0.5 to 0.6.

**No movement / stationarity signal in Phase 1.**

Considered as a way to distinguish stationary refs from moving players. Ruled out because:
- A stationary player waiting for a serve or pausing between shots would fail a stationarity gate; a threshold loose enough to let them pass is loose enough to let most refs through.
- The actual mechanism that excludes non-players is the on-court projection test combined with the EMA. In ShuttleSet, refs, chair umpire, line judges, and audience all project outside the court rectangle, so they never enter the candidate pool for anchor-distance purposes. The EMA captures "where the slot's player has been trending" without needing a motion proxy.

Amateur may need to revisit if coaches / kids / bystanders start projecting on-court.

**Score is used only as a noise-floor filter, not as a divisor.**

The first-cut plan used `effective_distance = D / max(score, eps)` so high-score detections were preferred. Empirical evidence contradicts this: RTMDet's `bbox_score` is driven primarily by bbox size, completeness, scene prior, and lighting. Chair umpires and front-row audience are stationary and well-framed, so they score as high as or higher than motion-blurred airborne players. A score divisor would systematically penalise the real player on exactly the frames we care about. Final use: `score_filter = 0.2` as a noise floor only. No weighting beyond that.

**EMA resets on zeroed slots instead of freezing.**

Freeze creates a stale-anchor capture risk after long zeroings. Walk-through: if Top's EMA is stuck at (0.1, 0.1) from a pre-zero pick, the slot zeros for ~10 frames while the real player runs to the opposite corner, and a nearby line judge projects at (0.4, 0.2). On reappearance, distance from effective_anchor to the real player is ~0.75 vs ~0.01 for the line judge. Freeze: line judge wins by 60x. Reset to `halfcourt_centre`: line judge still closer, but only ~7x. The 25% EMA weight is the only thing lost on reset, which is cheap. The 75% `halfcourt_centre` weight (the structural anchor) is unaffected. No decay-to-prior intermediate option was considered necessary.

**No second sanity ceiling on max displacement from last picked position.**

Considered to catch "wrong pick at clip start" cases. Rejected because it creates a silent persistent-capture failure: if the EMA initially locks onto a line judge, a "max displacement" gate rejects the real player when they reappear elsewhere because they're far from the line judge's stationary position. The weighted prior (0.75) is already the correct defence for the wrong-start case, because `effective_anchor` stays near the court-half midpoint regardless of EMA pollution.

**Body-frame projection handles sitting, not 3D pose, not shin angle.**

Three routes considered for "legs perpendicular to body axis" vs "legs in body-down direction":

- 3D pose via MMPose's `human3d`: provides true 3D keypoints so the "is knee-hip-knee plane horizontal in the world frame?" test is literally computable. Costs ~50-85 extra minutes at the per-clip model-reload step (documented MMPose 1.3.2 bug). Quality on broadcast footage is modest; trained on well-lit single-subject videos. Rejected on cost-benefit for Phase 1.
- Shin angle (knee-to-ankle vs body_up): doesn't discriminate cleanly because shins are usually vertical in 3D whether sitting (with feet on footrest) or standing. New failure modes on seated-with-legs-crossed / feet-tucked positions. Rejected as adding complexity without coverage.
- Body-frame projection (2D): projects knee-offset-from-hip onto the hip-to-shoulder axis. Sitting person: ratio ~0; standing or airborne player: ratio around -0.7 to -0.9. Empirically verified on `16_1_42_4`: frame 42 (apex smash scissor kick, legs kicked back near hip level) gives `body_frame_ratio = -0.874`, cleanly "not sitting"; frame 32 (pre-jump crouch, feet at knee level) gives -0.734, also "not sitting". Chosen.

**No confidence gates on the sitting test.**

Player knees are generally less occluded than ref knees in ShuttleSet broadcasts. A joint-score gate would gate out exactly the mostly-seated refs we want to filter while letting standing players through. The body-frame ratio is cheap enough (handful of multiplications per candidate) that a gate saves nothing.

**Tiebreaker invoked only on close-to-tie anchor distances, with sitting-filter + bbox area.**

Bbox area alone isn't the primary signal: tower umpire bboxes are large because the elevated position is fully visible, and seated-ref bboxes can be widened by instrument stands captured in the detection context. Pixel-space size is also perspective-biased (Bottom player always larger than Top). Confidence isn't useful as a primary either, for the score-doesn't-discriminate reason above.

The tiebreaker only fires when multiple candidates are within `tiebreaker_tol = 0.05` of the winning anchor distance. In ShuttleSet singles this is rare (typical frames have one on-court candidate per slot). Order: drop sitting candidates first, then break the remaining tie by largest bbox area. If the sitting filter drops everyone, revert to the original `argmin D` pick.

**Bottom-first pick order.**

Bottom player sits closer to the camera, so their bboxes are larger, scores are higher, and detections are more reliable. Locking in the higher-confidence assignment first reduces the Top pool by a detection we're most sure about, rather than the reverse. Bottom-first greedy combined with the closer-to-own-anchor pre-filter (next entry) covers the cross-capture cases.

**Closer-to-own-anchor pre-filter on the candidate pool (cross-half capture guard).**

With `sanity_ceiling = 0.6`, a legitimate but geometrically adversarial frame can let one slot's argmin grab the other slot's player. Concrete example: Bottom player in their own deep corner at (0.2, 0.95), Top player attacking at the net on their side at (0.5, 0.48):

- From `Bottom_prior` at (0.5, 0.75): real Bottom = 0.36, real Top = 0.27.
- Both under 0.6, so both pass `sanity_ceiling`.
- Bottom's argmin picks the Top player (0.27 beats 0.36). Top's remaining pool then has only the Bottom player at distance 0.76 from `Top_prior`, which exceeds the ceiling, so Top zeroes. Frame is marked failed, slot assignments wrong.

Fix: each candidate is eligible for slot `s` only if `D(candidate, s) <= D(candidate, other_slot)`. In the scenario above, the Top player near the net is closer to `Top_prior` (0.23) than to `Bottom_prior` (0.27), so the closer-to-own-anchor rule drops them from Bottom's pool entirely. Bottom then picks the real Bottom player unambiguously, and Top picks the near-net Top player unambiguously.

This is effectively a dynamic midline split that follows the shifting EMA anchors frame by frame. Zero extra compute because `D(candidate, other_slot)` is already being computed for the other slot's own pre-filter.

**Irrecoverable clips stay in the Phase 1 denominator.**

Some fraction of the 1,716 hit-zone-busted clips are fundamentally irrecoverable by any heuristic operating on raw MMPose output: broadcast extreme close-ups, side-on framings with no court visible, cuts to a different subject. Position: keep them zeroed after sticky_anchor runs, keep them in the Phase 1 decision-gate denominator. Zeroed irrecoverable clips at worst regularise the model; at best teach the transformer to attend around bad data. No manual inspection pass over the tail, no subset carve-out.

**Projection anchor: bbox bottom-centre, not ankle midpoint.**

The BST original projects the COCO ankle-midpoint (joints 15 and 16); sticky_anchor projects bbox bottom-centre. Switched for a simpler code path (no per-keypoint confidence fallback) and a proxy that doesn't depend on MMPose's anatomical-prior fill of low-confidence keypoints.

Asymmetric trade-off: bbox-bottom sits a few pixels below the ankle midpoint and projects to ~0.01-0.03 larger court y under the ground-plane homography. Marginally protective for Bottom (further from the bisector at y=0.5), marginally risky for Top (closer to it). Small relative to `sanity_ceiling = 0.6` and per-frame EMA noise (0.04-0.10), so it rarely matters except at the Voronoi bisector edge for Top, which isn't observed in residuals.

Open: switching Top alone back to ankle-midpoint (with bbox-bottom fallback) would restore the asymmetric net-crossover protection of the BST original. Discussion-only until a real failing clip surfaces.

### Rejected variants and why (do not resurrect without new evidence)

- **`torso_center`** (hip-based projection as the primary): geometrically worse than feet for the standing/grounded case. Standing hips project with ~4.64 error amplification at far court edge vs ~0 for grounded feet. The first-cut plan's "robust to airborne feet" motivation was wrong.
- **`monotrack_carry_forward`**: repeated identical poses across consecutive frames corrupt the transformer's temporal signal. Don't resurrect without evidence the model tolerates position-freeze.
- **`wider_court(eps)` alone**: a bigger flat eps doesn't address the far-edge airborne case that needs a non-rectangular tolerance.
- **`trust_mmpose`**: skipping the filter when exactly 2 are detected still relies on projection correctness and ignores the bystander case.
- **`focus_aware`**: subsumed by sticky_anchor's per-slot tracking. Adds a label-dependency for no marginal gain.
- **`jump_adaptive_eps`**: subsumed by sticky_anchor's relative-proximity framing. Hip-foot jump detector adds complexity with no gain.
- **Continuity check** (reject pick > 0.25 normalised from previous valid pick): rejects legitimate recoveries after long invisible gaps. Weighted anchor + sanity ceiling defend adequately.
- **Per-joint masking at apply_heuristic layer**: MMPose hallucinates off-screen joints from anatomical prior; this fill is almost certainly higher signal than a heuristic replacement. Per-joint scores saved alongside for possible training-side use.

## Failure modes observed

### 19_2_10_7: MMPose detection-layer gap, not Voronoi

Per-frame replay of sticky_anchor on the 12 zeroed frames (f009, f011-f021) on 2026-04-25 showed:

- BOT picks the Bottom player cleanly on every frame (`c0` projected at ~(0.47, 0.69), `d_to_BOT ~= 0.05-0.06`). Not zeroed, correctly picked.
- TOP zeros because there is no candidate anywhere in the TOP half of the court. All non-`c0` candidates project to y < 0 (past the top baseline) or to `|x| > 1` (off-court left/right) and are umpires, linesmen, photographers, bench.
- On working frames (ndet = 10) the Top player is a reliable detection with a bbox near (857, 360)-(1000, 600) pixel-space and score around 0.74-0.77. On zeroed frames (ndet = 9) that bbox is simply missing from the raw arrays. One-frame re-appearance at f010 (score 0.74) confirms a momentary detection gap, not a camera cut.

So on this clip the failure is a **detection-layer gap, not a heuristic-layer filter misfire**. Voronoi is not firing because there's no Top-half candidate to filter; `sanity_ceiling = 0.6` drops everything because nothing is close enough to the TOP anchor. No tweak to `sticky_anchor` hyperparameters can recover these frames.

Cause of the detection gap (supported by visual inspection of f017): detector proposal failure under extreme occlusion, not NMS. A visual pass over the overlay PNG shows the Top player is roughly 85% occluded by the Bottom player at the net, with only a head fragment and a sliver of shoulder/arm visible. Measured IoU between the Top's inferred full bbox (from f010) and the visible Bottom bbox is ~0.36 on zeroed frames, climbing from ~0.30 on flanking working frames. That is below standard NMS thresholds (0.5-0.7), and on f010 itself the Top bbox is retained at IoU 0.30 with score 0.74, well above any confidence floor. So NMS cannot be the proximate cause. The more parsimonious reading: the detector's region-proposal stage does not generate a bbox when only ~15% of the body is visible, because the visible fragment does not look like a full-body or torso-level feature pattern. There is nothing for NMS to suppress.

This reframes the recovery routes for clips like this:

- NMS relaxation and confidence threshold drop will not help, because there is no sub-threshold proposal to admit.
- A detector trained on heavily occluded people (CrowdDet, OccluDet, some YOLOX variants) might propose a bbox here, at the cost of a full detector swap.
- Temporal interpolation from flanking frames is the lightest-weight pragmatic fix. Spelled out in "Recovery routes parked" below.

### Net-crossover zeroing (mathematically valid, not observed in residuals)

The mechanism is real in principle: if a Top player projects to y > 0.5 (e.g. (0.5, 0.52)), then `D(Top_player, BOT_anchor) = 0.23 < D(Top_player, TOP_anchor) = 0.27`, and the Voronoi pre-filter drops that candidate from the TOP pool. If no other detection is closer to TOP than to BOT, TOP zeros. This is the deliberate trade-off the cross-half-capture guard was designed for.

**As of 2026-04-25, no clip in the inspected residuals actually exhibits this failure mode.** The earlier writeup attributed 21% of 19_2_10_7's frames to it; that attribution was wrong (see correction above). The 3.5%-residual upper-bound estimate was tied to the same misattribution. Across all inspected still-busted residuals, the dominant patterns are detection-layer gaps (occlusion-driven proposal failures) and irrecoverable framings (closeup, side-on, cutaway), not Voronoi crossover.

Options if a future pass identifies a real Voronoi-crossover clip:

- Relax the rule when the other slot has no candidates in its pool (re-introduces some cross-half capture risk).
- Add a net-proximity exception that bypasses the rule within a y-band around the net line.
- One-Euro filter as the already-reserved Phase 2 stall fallback.

None should be built without first identifying a real failing clip; we are otherwise optimising for a hypothetical.

### Considered and declined: hip-projection / closest-to-EMA partial-success rerun (2026-04-25)

Asymmetric hip-projection rerun was considered as a fallback for the hypothetical Voronoi-crossover failure: when only one slot is assigned, rerun the algorithm with Bottom = bbox-bottom (feet) and Top = mid-hip projection. The geometry argument for the asymmetry: hip pixels are higher on screen than feet, so the ground-plane homography projects them at lower court y (further from camera). For Top this pushes them deeper into their own half, away from the bisector at y = 0.5 (helps). For Bottom, lower court y means *toward* the bisector and possibly across it (hurts). So `Bottom = feet, Top = hips` is the geometrically principled form, not symmetric.

A weaker variant was also considered: drop the Voronoi/sanity filters on the rerun and take the closest-to-EMA candidate among the previously rejected ones. Broader recovery, but also rescues cases that should stay zeroed (cross-half capture from a detection in the wrong half).

**Declined as solving an unobserved problem.** No clip in the inspected residuals actually exhibits the projection-induced Voronoi crossover this fallback would fix. Building it would introduce real code complexity (separate projection function, fallback path, additional state machine in `_pick_one_frame`) for a hypothetical case. Aligns with the "noise as regularisation, full set stays the denominator" stance: don't carve out rescue paths for failure modes that haven't been seen on actual data.

If a real Voronoi-crossover clip ever surfaces, this section is the starting point and the geometry argument for the asymmetric form stands.

### Renderer bug (fixed 2026-04-25)

`render_sticky_anchor_overlays.py` drew every bbox grey whenever `failed[f] = True`, because the pick-matching step was gated on `not failed[f]` rather than per-slot. A partial success (one slot picked, the other zeroed) therefore hid the valid pick behind the grey "unpicked" colour, which is what made the 12 zeroed frames on 19_2_10_7 look like a double-zero in the overlays and led to the original (incorrect) Voronoi attribution. The fix tests `pos[f, slot].any()` per slot. After the fix, partial picks render in their slot colour.

## Recovery routes parked

Both candidates are post-processing modules that run after sticky_anchor and preserve the byte-identity chain (sticky_anchor's outputs are untouched; the recovery stage writes to a new output dir).

### Gap-fill post-processing (proposed, 2026-04-25)

For partial-success frames where one slot picks cleanly but the other zeroes due to an upstream MMPose detection gap. Linear interpolation of `pos` and `joints` between flanking good frames recovers the missing slot.

Classifier (per frame). A frame `f` is a gap-fill candidate iff all three hold:

1. `failed[f] = True`.
2. Exactly one of `pos[f, TOP]`, `pos[f, BOT]` is non-zero (partial success, not a full zero).
3. The missing slot has at least one non-zero pick elsewhere in the clip, with a frame gap of `<= 15` on at least one side.

Signal 2 excludes homography-fail frames and cutaways (both slots zero in those cases). Signal 3 excludes clips where a slot was never detected at all, so interpolation always has at least one anchor point.

Interpolation. Linear between `prev_good` and `next_good` for the missing slot when both exist. Constant extrapolation (copy the nearest anchor) if only one side exists. Skip interpolation entirely if the two endpoints differ by more than ~0.3 normalised court units (player moved too much for linear to be safe).

Gap threshold. `<= 15` frames at 30fps is ~0.5s. Badminton players at the net move slowly enough (typically < 0.03 court units per frame at the net, faster mid-court) that linear interpolation across that window stays within ~0.05 units of truth. Beyond 15-20 frames the constant-velocity assumption breaks down.

Explicit non-choice: do NOT fall back to raw MMPose bboxes that sticky_anchor rejected. The existing `sanity_ceiling = 0.6` and `generous_margin = 0.15` are already generous relative to the court (a 0.6-unit ceiling allows picks up to half a court from the anchor; `generous_margin = 0.15` allows projections 15% past any baseline). A candidate being rejected at those margins is strong evidence that MMPose failed upstream, not that the heuristic was over-strict. Interpolation is the right response; reinstating the raw bbox would dilute the filter with the exact noise the filter was designed to catch.

Where it lives. A new post-processing module, e.g. `preparing_data/gapfill.py`, with signature `(pos, joints, failed, raw) -> (pos, joints, failed)`. Reads the existing `_pos`/`_joints`/`_failed` outputs, identifies recovery candidates per the classifier, interpolates, and writes to a new output dir (e.g. `dataset_npy_..._flat_h_sticky_anchor_gapfill`).

Scope. ~100 lines plus a sample validation pass. Try for this trimester if time allows; most likely lands in Phase 2 next trimester. Per-clip yield is probably comparable to the homography-fail rescue (handful of clips out of 61 residuals).

### Homography-fail X3D-S-only rescue (Phase 2 candidate)

For clips where the court homography itself didn't fit (example: `3_2_24_1`). Without a usable homography, `_pos` and `_joints` can't be produced in court-normalised space at all, so they have to stay zeroed. But the raw MMPose keypoints for those frames are often plausible in pixel space, and the X3D-S wrist crop only needs pixel space.

Concept: a pixel-space fallback picker that fires only when homography is unusable. Largest bbox per screen-half (top-of-screen = Top slot, bottom-of-screen = Bottom slot under standard broadcast angle), torso-diagonal-relative crop sizing to normalise for near/far scale. Feeds just the X3D-S stream; BST inputs remain zeroed.

Needs a new metadata flag in the extract output: "player keypoints plausible in pixel space, court coords unusable." The current `_failed[f] = True -> pos/joints zeroed` contract is too binary to carry that distinction, so every downstream consumer (loader, fusion module) would need to branch on the flag.

Scope of the potential win:
- Only rescues clips that are standard broadcast angle with transient homography glitch. Most homography-fail clips fail because the frame itself is non-standard (closeup, side-on, cutaway) and wouldn't recover either, because the largest-bbox-per-screen-half rule doesn't work on those framings.
- Ceiling is a handful of clips out of the 61 still-busted residuals.

Not scoped for this trimester:
- Collides with the "noise as regularisation, full set stays the denominator" stance.
- Needs an architectural call on whether X3D-S can classify independently when keypoints are zeroed; a fixed-fusion BST would have the zero-joints drag the output toward null regardless of how good the crop is.
- Per-clip yield is low, so the engineering cost of the new output contract is hard to justify on its own.

Revisit in Phase 2 after per-class residuals are in. Justified only if worst-class failure rate (expected candidates: Top_smash, wrist_smash) traces specifically to homography-fail frames, measurable by slicing per-class fail rate on `_failed[f] = True` around the hit frame.

## Known limitations of `sticky_anchor`

- **Same-angle replay in cutaway**: a replay frame at near-identical camera angle would let bystander detections pass the in-court update gate, potentially polluting the EMA. Rare in broadcast badminton; not bulletproof.
- **Ball kid / court-crosser during play**: if the real player is off-frame or low-detected AND an intruder is in-court and passes the confidence-proximity test, the intruder could briefly capture a slot. Weighted anchor (0.75 prior) limits damage: real player reclaims on reappearance.
- **Amateur footage**: structurally worse on intrusion cases (refs walk around more, crowd visible, fewer players confidently detected). Tuned to ShuttleSet pro scope. Amateur extension would need hardening on the rally-presence check and possibly an in-court gate on the picking stage, not just the update stage.
- **Two players simultaneously airborne**: singles rarity; would currently trigger the rally-presence check (neither pick in generous court) and zero both slots. Negligible in practice.
- **Bootstrap with long cutaway intro**: if the clip's first 15+ frames are broadcast padding, picks during padding use the court prior only. The first real in-court detection starts updating EMA; convergence to the player's actual trajectory takes ~5 real-play frames. Slight mis-picks in those early frames but no data loss.
- **Continuity check intentionally absent**: a continuity threshold ("reject a pick > X away from previous pick") would reject legitimate player re-appearances after long invisible gaps. The weighted anchor + sanity ceiling are the intended defence.
- **Detection-layer gaps are unresolved**: where MMPose fails to propose a bbox at all (heavy occlusion at the net, as in 19_2_10_7), no heuristic-layer tuning recovers the frame. Recovery routes are temporal interpolation (parked) or a swap to an occlusion-robust detector.

## Phase 2 plan and success criteria

**Status note (2026-04-25)**: Phase 1 mixed retrain failed the decision gate, and the per-class frame-zeroing audit (`analysis_outputs/zeroed_frames_class_audit__run_20260425_150548.txt`) confirmed the F1-bottom classes aren't the heavily-zeroed ones. The data-quality-bottleneck motivation for Phase 2 is no longer empirically supported. The decoupled `raw_extract` is also a lot faster per clip than the original committed pipeline (the GPU run on 1,716 clips landed at ~20 min, extrapolating to ~6 hr for the remaining ~31k rather than the ~50 hr estimate against the old in-line code path), so Phase 2 isn't expensive enough to be ruled out forever. But it's no longer the priority. Focal loss + data augmentation + X3D-S have stronger structural arguments for the wrist_smash floor.

The original Phase 2 plan and criteria below remain valid if a fresh motivation surfaces (e.g. an audit on a future run shows residual data-quality drag on a class the model is otherwise well-positioned for):

1. Re-run Step 2 raw across the full 33k clips. Output: `dataset_npy_between_2_hits_with_max_limits_flat_raw/`.
2. Run `apply_heuristic --heuristic sticky_anchor` against the full raw extract. Output: `..._flat_h_sticky_anchor/`.
3. Collate once per ablation (V3 + V4 split columns) with `--clip-npy-dir ..._flat_h_sticky_anchor/`. New ablation_id suffix to tag the heuristic: `une_merge_v1_split_v2_dropunk_h_sticky_anchor` (V4-analog), `merged_25_split_bst_baseline_keepunk_h_sticky_anchor` (V3-analog).
4. Re-train V3 and V4 with 5 serials each. Document via the existing run_tracker pattern.
5. Compare mean macro/min/acc across the 5 serials vs the committed baselines.

Success criteria (committed only after Phase 2 validates both axes):

- Zeroing rate on the target-class aggregate drops by >= 25% relative on each of (train, val, test) partitions.
- Retrained V4 best-serial min-F1 lifts by >= 0.04 vs committed V4's 0.432 (matching or exceeding V3's 0.381 + the V3 to V4 +0.04 min-F1 gain, so >= 0.47 net).
- Retrained V4 best-serial macro and accuracy do not drop by more than the noise margin across the 5 serials (~0.005).

## Amateur generalisation notes (for next trimester)

Most decisions carry into amateur data without surgery. What stays, what needs re-derivation, what may need expanding.

**Stays the same in amateur:**
- Per-slot anchor architecture.
- Pick-by-court-space-proximity with EMA.
- Body-frame sitting test: scale-invariant, works across cameras and subject distances.
- EMA reset on zeroed slots.
- Bottom-first pick order and the closer-to-own-anchor pre-filter on the candidate pool.
- Raw extracts in pixel space; homography applied downstream at heuristic time.
- No score divisor, no movement gate.
- Dynamic derivation of `halfcourt_centre` from the homography borders (already data-driven).

**Needs per-video or per-camera re-derivation in amateur:**
- `sanity_ceiling`: 0.6 is tuned to ShuttleSet's high-behind-baseline camera geometry. Taller or more oblique cameras inflate the apex-jump projection error differently. Candidate approach: measure per-video the 99th percentile of stable-pick anchor-distance during the first N seconds; set the ceiling to that plus a margin.
- `generous_margin`: 0.15 matches professional venue run-off of 1-2 m. Amateur courts vary. Candidate approach: per-video observed-play-extent quantile.
- `score_filter`: 0.2 is conservative for well-lit broadcast; phone-grade amateur footage has lower baseline scores and may need 0.1 or 0.05.

None of these need Phase 1 work; hooks exist via CLI args.

**May need expanding if empirical failures appear:**
- Bbox-size tiebreaker could pick up detector artefacts more often in amateur (stands, chair structures, partially-visible bodies). Mitigation: bbox aspect-ratio sanity check (height / width between ~1.2 and ~4) before using bbox area, or confidence-weighted joint bounding-rectangle area.
- Movement / stationarity signal: if the on-court test starts admitting non-players near the anchor, interframe displacement (in court space, via nearest-neighbour proxy since MMPose detections have no persistent identity) becomes the natural discriminator. Hook: per-slot assignment would consult the previous frame's picks.
- 3D pose via `human3d`: only worth the compute if the 2D body-frame ratio starts misclassifying under a new camera angle. High cost, low likely benefit.

**Out of scope even in the amateur phase:**
- Doubles or mixed formats: the two-slot architecture breaks. Different design entirely.
- Multi-camera composition (behind-net, net-level, ground-level): projection error profile changes enough that ShuttleSet-derived thresholds become meaningless.

## Risks

| Risk | Mitigation |
|---|---|
| `sticky_anchor`'s EMA gets captured by a persistent in-court intruder | Weighted anchor (0.75 prior) caps drift toward intruder at 25%. Real player reclaims on reappearance. Sanity ceiling + rally-presence check are the defences. |
| Phase 1 mixed re-train produces no meaningful signal because 1,716 clips are too small a fraction of train | Zeroing-rate drop alone is a sufficient Phase-1 gate to proceed to Phase 2 if model-lift is noisy. |
| Heuristic changes compound and become unmaintainable | Each variant in its own module under `heuristics/`. No flag-branching inside a single function. Name-based dispatch from `apply_heuristic.py`. |
| Same-angle replay or ball kid capture | Documented as a known limitation. Rally-presence check + weighted anchor limit damage. Not a blocker for ShuttleSet pro scope. |
| Byte-identity gate (`current` variant) fails to reproduce committed output | Investigate plumbing before trusting `sticky_anchor`. Common causes: keypoint-index ordering, bbox row order, normalisation step order, resolution-scale application. (Resolved in this pass: required `center_align=True` at the `normalize_joints` call site.) |

## Out of scope

- Changing the MMPose model (RTMPose-L stays). Retry only if heuristic work fails to lift performance.
- Carry-forward / interpolation as a primary heuristic mechanism. Repeated identical poses across consecutive frames corrupt the temporal signal the transformer learns. (Note: gap-fill post-processing is permissible because it's bounded to ~15-frame windows and runs only on partial-success frames; see Recovery routes.)
- Jump-adaptive eps via hip-foot detector: subsumed by sticky_anchor's relative-proximity framing.
- Hip-based court projection as the primary projection method (`torso_center` in the original plan): geometrically wrong; projection error for hips is ~4.64x a standing player's height at the far edge.
- Per-joint masking at apply_heuristic layer: retain MMPose's anatomical-prior fill; per-joint scores saved for possible training-side use.
- Flattening the .mp4 clips dir (Phase 3 of the dir-flatten refactor).
- Arch 1 wrist crop work: independent, proceeds on whichever extract is canonical at the time.
- Re-encoding the raw .mp4 clips to address potential decode artefacts.
- Full BoT-SORT with appearance embeddings (doubles-badminton-paper style): overkill for singles with known priors.
- Continuity check (pick-to-previous-pick distance threshold).

## References

- `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`: original `detect_players_2d`, `check_pos_in_court`, `to_court_coordinate`, `normalize_position`, `normalize_joints`, and the zeroing decision inside `detect_players_2d`.
- `src/bst_refactor/stroke_classification/preparing_data/apply_heuristic.py`: CLI + `run` library entry point.
- `src/bst_refactor/stroke_classification/preparing_data/heuristics/`: package with `__init__.py`, `base.py`, `current.py`, `sticky_anchor.py`.
- `src/bst_refactor/stroke_classification/preparing_data/failsafe_bst_mmpose_zeroing_check_equivalence.py`: byte-identity gate.
- `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/render_sticky_anchor_overlays.py`: overlay renderer (partial-pick fix landed 2026-04-25).
- `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/busted_hit_zone_after_sticky_anchor.txt`: 61 stems still-busted after sticky_anchor.
- `src/bst_refactor/validation_scripts/validate_zeroed_frames.py` and `fail_rate_per_class.py`: per-class fail-rate diagnostics.
- `src/bst_refactor/validation_scripts/zeroed_frames_analysis_outputs/analysis_unemergev1_v2_20260421_1159.txt`: original analysis confirming the busted-clip class skew.
- `src/bst_refactor/stroke_classification/preparing_data/keypoints_schema.md`: COCO joint indices (15 and 16 = ankles, 11 and 12 = hips, 13 and 14 = knees, 5 and 6 = shoulders).
- `scratch/architecture_notes/busted_hit_zone_clips_phase1.txt`: the canonical 1,716 hit-zone-busted stem list.
- `scratch/architecture_notes/busted_whole_clips_phase1.txt`: historical 222 whole-clip-busted list.
- `scratch/architecture_notes/mmpose_heuristic/mmpose_phase1_extraction_plan.md`: Phase 1 execution log.
- `scratch/architecture_notes/mmpose_heuristic/mmpose_bounds_filtering_research.md`: external research. No published ablations of filter strategies found. Key inputs: MonoTrack (CVPRW 2022) relaxed-boundary + closest-to-last-in-court pose; Padel paper (Javadiha et al. Sensors 2021) projection-geometry quantification; TemPose carry-forward vs BST zero-fill contrast.
- V3 and V4 committed run manifests: `run_20260420_141629/` (V3) and `run_20260420_171101/` (V4).

## Revisions log

Chronological record of how the design and implementation evolved. Each entry leads with date and describes what changed and why.

### 2026-04-21: design pass folded in

- Variant list cut down. Removed `wider_court`, `torso_center`, `trust_mmpose`, `focus_aware`, `jump_adaptive_eps`, and `monotrack_carry_forward` as separate Phase 1 variants. Kept `current` (reference gate) and added a single primary variant `sticky_anchor`. Plan B `body_length_fallback` sketched but not scaffolded.
- `torso_center` / hip-projection dropped as geometrically wrong (hips project worse than airborne feet at the far edge).
- `monotrack_carry_forward` dropped on temporal-signal-corruption grounds.
- New primary variant `sticky_anchor`: per-slot EMA blended with fixed court-side midpoint prior (0.75 prior, 0.25 EMA).
- Output schema gained `_raw_kp_scores.npy`.
- 3D raw output omitted from this pass (commented scaffolding).
- Phase 0 simplified.
- Per-joint masking at apply_heuristic layer rejected.

### 2026-04-22: Phase 1 execution log

Phase 1 raw extraction executed.

- Scope changed from 222 whole-clip-fail clips to 1,716 hit-zone-fail clips (+/-10 frames around the hit frame).
- Canonical list: `scratch/architecture_notes/busted_hit_zone_clips_phase1.txt` (1,716 stems).
- N_max bumped 8 -> 16. Original plan said N_max=8 "basically never fires"; false on the busted subset (87% of the first 222-clip extract triggered the cap). At N=16, only 0.79% of frames hit.
- Score-filtered ndet peaks at 8; players reliably the most salient detections by bbox area and horizontal centrality. Load-bearing input for the sticky_anchor design revisit.
- New artefacts: `find_busted_clips.py` gained `--hit-zone` flags; `raw_extract.py` prints end-of-run unique-clip summary; `summarise_raw_ndet.py` added.
- Path canon corrections: clips dir at `/scratch/comp320a/ShuttleSet/clips`; `set/` and `video_metadata.csv` are committed to git under `src/bst_refactor/ShuttleSet/`, NOT symlinked to `/scratch`.

### 2026-04-22: Phase 1 execution completion

Heuristic dispatch scaffolded, byte-identity-gated, sticky_anchor implemented, run on the full 1,716-clip raw extract, sample inspected.

- Byte-identity gate passed 50/50, bit-exact on `_pos` and `_joints` (max abs diff = 0), exact on `_failed`.
- One bug surfaced: `heuristics/current.py` needed `center_align=True` at `normalize_joints`. Both the flip and the rationale recorded in the `normalize_joints` docstring.
- `sticky_anchor` ran in 54 s on engelbart against the N=16 raw extract.
- Headline: 1,631 of 1,716 clips perfectly clean (95.05%). Hit-zone busted-clip count under `fail_rate > 0.50` dropped from 1,716 to 61.
- Per-split: train 110 -> 47 (-57%), val 49 -> 6 (-88%), test 33 -> 8 (-76%).
- 9/10 still-busted samples genuinely irrecoverable.
- New artefacts: `apply_heuristic.py`, `failsafe_bst_mmpose_zeroing_check_equivalence.py`, `heuristics/` package, `render_sticky_anchor_overlays.py`, `busted_hit_zone_after_sticky_anchor.txt`, `normalize_joints` docstring note.

Hyperparameters: original `sanity_ceiling = 0.5` widened to 0.6 after the apex-jump empirical check on `16_1_42_4`. Original `score_filter` divisor logic dropped (RTMDet score doesn't discriminate). EMA changed from freeze to reset on zeroed slots. Body-frame sitting test + bbox-area tiebreaker added.

### 2026-04-25: 19_2_10_7 trace + corrections

- Per-frame replay on the 12 zeroed frames of `19_2_10_7` showed the failure is upstream MMPose detection-layer gap (heavy occlusion at the net), not Voronoi crossover as previously attributed. Visual inspection of f017 estimated ~85% Top-player occlusion behind the Bottom player; IoU between Top and Bottom bboxes peaks at ~0.36 (below standard NMS thresholds), with Top retained at IoU 0.30 on f010 (score 0.74). Detector proposal-stage failure under occlusion is the parsimonious cause; NMS relaxation will not help.
- Net-crossover zeroing reframed: mathematically valid mechanism, but no clip in the inspected residuals actually exhibits it. The earlier 3.5%-residual upper bound was tied to the misattribution.
- Hip-projection / closest-to-EMA partial-success rerun considered and declined. Asymmetric form (`Bottom = feet, Top = hips`) is the geometrically principled one (symmetric form would push Bottom toward the bisector). Declined as solving a hypothetical, since no observed clip exhibits Voronoi crossover.
- Renderer bug fixed in `render_sticky_anchor_overlays.py`: pick-matching now per-slot (`pos[f, slot].any()`) rather than gated on `not failed[f]`. Partial successes now render in their slot colour.
- Recovery routes spelled out: gap-fill post-processing (temporal interpolation across MMPose detection gaps) and homography-fail X3D-S-only rescue. Both parked; gap-fill could fit this trimester if time allows.
