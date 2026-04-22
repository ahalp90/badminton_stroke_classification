# MMPose Extraction: Heuristic Investigation Plan

## Revisions 2026-04-21

Collaborative design pass folded into this doc. Changes vs the first-cut plan:

- **Variant list cut down.** Removed `wider_court`, `torso_center`, `trust_mmpose`, `focus_aware`, `jump_adaptive_eps`, and `monotrack_carry_forward` as separate Phase 1 variants. Kept `current` (reference gate) and added a single primary variant `sticky_anchor`. Plan B `body_length_fallback` sketched but not scaffolded unless Phase 1 signal is ambiguous.
- **`torso_center` / hip-projection dropped as geometrically wrong.** Padel paper's `H_z * tan(θ)` gives ~4.64 error amplification at far court edge for broadcast camera geometry. Standing hips (~1 m off ground) project worse than airborne feet (~0.5 m off ground). The original plan's "robust to airborne feet" rationale was incorrect.
- **`monotrack_carry_forward` dropped.** Repeated identical poses across consecutive frames corrupt the temporal signal the transformer learns. One-Euro filter reserved as a one-shot Phase-2 option only if heuristic work stalls.
- **New primary variant `sticky_anchor`.** Per-slot EMA tracking blended with fixed court-side midpoint prior (0.75 prior, 0.25 EMA). Replaces the "zero both players if either fails" filter with a proximity-based per-slot picker, sanity ceiling, and rally-presence check. Detailed algorithm in the Phase 1 section below.
- **Output schema gained `_raw_kp_scores.npy`**, saving MMPose per-joint confidence. Enables ankle-confidence-based projection fallback (ankle midpoint to bbox bottom-center when ankle scores are low) and preserves per-joint confidence for possible training-side use later.
- **3D raw output omitted** from this pass. Scaffolded in commented-out lines inside `raw_extract.py` with a toggle note.
- **Phase 0 simplified.** The 222-clip busted list is discoverable by a ~30-40-line scanner walking `*_failed.npy` files; no fresh `validate_zeroed_frames.py` run required. Visual inspection deferred and handled separately, not on the critical path.
- **Per-joint masking at apply_heuristic layer rejected.** MMPose hallucinates off-screen joints from anatomical prior; this fill is almost certainly higher signal than a heuristic replacement. Scores saved alongside for possible training-side use.

Phase 2 structure and success criteria unchanged.

## Revisions 2026-04-22 (Phase 1 execution log)

Phase 1 raw extraction executed. Phase 2 structure is unaffected; Phase 1 scope and parameters shifted. Detailed log in `mmpose_phase1_extraction_plan.md` "Execution log (2026-04-22)".

**Scope change: Phase 1 now operates on the 1,716-clip hit-zone-fail set, not the 222-clip whole-clip-fail set.**
- Criterion switched from "whole-clip `fail_rate > 0.50`" to "hit-zone `fail_rate > 0.50`" (+/-10 frames around the hit frame; matches the `hit_zone_heatmap` filter in `validate_zeroed_frames.py`).
- Canonical list: `scratch/architecture_notes/busted_hit_zone_clips_phase1.txt` (1,716 stems). The 222-stem whole-clip list is preserved as a historical artefact at `scratch/architecture_notes/busted_whole_clips_phase1.txt`.
- Storage estimate: hit-zone set at N_max=16 fits in under 1 GB. Phase 2 full 33k estimate needs a re-measure once Phase 1 runs through; the 5.3 GB figure in the original output-format section is for N_max=8 and will be higher at N=16.

**N_max bumped 8 -> 16.**
- Original plan said N_max=8 "basically never fires". False for the busted subset: 87% of the first 222-clip extract triggered the cap. Busted clips over-represent crowded frames by construction.
- At N=16, only 0.79% of frames on the 1,716-clip set hit the cap. 16 is sufficient; no need to go higher.
- Update the output-format table's N_max=8 entry and the 5.3 GB storage estimate when the apply_heuristic work lands (both are tied to the old width).

**Ndet findings (from `summarise_raw_ndet.py` on the 1,716-clip N=16 extract, 98,370 frames total) feed into the sticky_anchor design revisit:**
- Raw ndet per frame peaks at 9-10. The typical frame has many more than 2 detections (officials, line judges, crowd edge, etc.).
- Score-filtered (`bbox_score >= 0.5`) ndet peaks at 8. Score alone does not discriminate players from umpires / line judges / visible crowd.
- Visual audit (Ariel): players are the most salient detections in every inspected clip by bbox area and horizontal centrality. This is the load-bearing design input for revisiting the sticky_anchor selector; design discussion pending.
- 5 fully-zeroed hit-zone clips rsynced for visual inspection; findings pending.

**New artefacts added in this pass:**
- `find_busted_clips.py` gained `--hit-zone` / `--set-dir` / `--video-metadata-csv` / `--hit-window` flags. Whole-clip remains the default; hit-zone is the Phase 1 criterion going forward.
- `raw_extract.py` prints an end-of-run unique-clip summary of over-detection warnings so tmux scrollback loss doesn't hide them.
- `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/summarise_raw_ndet.py`: per-clip and aggregate detection-count breakdown with optional `bbox_score` filter. Not in the original plan; added to quantify the N_max decision and the sticky_anchor selector input.

**Path canon corrections (apply to any future sessions reading this doc as a reference):**
- Engelbart clips dir: `/scratch/comp320a/ShuttleSet/clips` (not nested under `ShuttleSet_data_merged_25/`).
- `set/` and `video_metadata.csv` are committed to git under `src/bst_refactor/ShuttleSet/` on both local and engelbart; they are NOT symlinked out to `/scratch` like `clips/` / `raw_video/` / `shuttle_csv/` / `shuttle_npy/`.

**Still to do for Phase 1 (revised next-session kickoff):**
1. sticky_anchor design revisit driven by the ndet findings and the fully-zeroed-clip inspection. Selector is likely to lean on bbox area + horizontal centre distance with score as a filter / tiebreaker, rather than score-weighted proximity as currently written. Design-in-flight; see next Phase 1 session.
2. `apply_heuristic.py` + `heuristics/` package (`current` byte-identity gate + `sticky_anchor`).
3. `current` variant's byte-identity gate on the overlap between the 1,716-clip hit-zone set and the committed filtered extract.
4. `sticky_anchor` on the full 1,716-clip raw extract.
5. Symlink-merged flat dir for Phase 1 mixed re-train (32k - 1,716 symlinks + 1,716 sticky_anchor outputs).
6. Collate + retrain V4 on merged data; compare min-F1 and zeroing rate vs committed V4.
7. Commit the three scripts + the 1,716-stem hit-zone list as the reproducible Phase 1 anchor once the design discussion lands.

**What did NOT change from the 2026-04-21 revisions:**
- Variant list: `current` + `sticky_anchor` (+ `body_length_fallback` as Plan B).
- Output schema: five arrays per clip, `_raw_kp_scores.npy` included.
- No modifications to `prepare_train_on_shuttleset.py` / `collate_npy` / `pipeline/`.
- Phase 2 structure, decision gate, success criteria.
- Geometric rationale for hip-projection rejection, carry-forward rejection, continuity-check rejection, etc.

## Court-space geometry and buffer sizing (2026-04-22)

Empirical findings from a code + CSV audit of `ShuttleSet/set/homography.csv` plus a per-frame overlay inspection of clip `3_1_18_3.mp4`. Directly informs `sticky_anchor`'s buffer hyperparameters.

### What the homography is calibrated to

All 44 videos in `homography.csv` project their annotated 4 corners (`upleft_x/y` ... `downright_x/y`) to an identical canonical rectangle in court-space: **300 wide x 660 tall**. UL=(25, 150), UR=(325, 150), DL=(25, 810), DR=(325, 810).

Length/width ratio = 660/300 = **2.2000**.

Candidate real-world references:

| Rectangle | Dimensions (m) | L/W ratio | Match? |
|---|---|---|---|
| Full doubles court (outer taped) | 6.10 x 13.40 | 2.1967 | **Yes (3 d.p.)** |
| Singles court (inner taped) | 5.18 x 13.40 | 2.5869 | No |
| BWF run-off zone (international minimum, 1m sides + 2m ends) | 8.10 x 17.40 | 2.148 | No |

The annotation target is the outer (doubles) taped court. No "further taped line" or run-off rectangle is involved. Scale: 300 units ↔ 6.10 m so one court-space unit is ~2.03 cm; the normalised [0, 1] interval spans the full outer doubles rectangle.

### Visual confirmation on clip `3_1_18_3` (video id 3)

Overlay PNG at `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/homography_overlay_3_1_18_3_f032.png` (frame 32, top player mid-smash). The cyan rectangle (annotated corners, scaled from the 1280x720 homography resolution up to the clip's 1920x1080 resolution) sits exactly on the outer doubles taped lines. A derived orange pair (doubles-sidelines minus 7.54% inset) lands precisely on the visible singles sidelines, which independently verifies that the annotations are on the outer, not the inner, taped lines.

Implied singles-sideline normalised x coordinates: **x = 0.0754** and **x = 0.9246** (since (6.10 - 5.18) / 2 / 6.10 = 0.0754). Singles play therefore occupies ~85% of the horizontal [0, 1] range; the outer ~7.5% on each side is the doubles tramline.

### The current `eps = 0.01` buffer is effectively zero

`check_pos_in_court` in `pipeline/court_utils.py:166` (mirrored in `prepare_train_on_shuttleset.py:230`) tests `-eps < x,y < 1 + eps` with `eps = 0.01`. Converted to physical units against the canonical rectangle:

| Axis | Normalised eps | Physical buffer |
|---|---|---|
| Horizontal (beyond doubles sideline) | 0.01 | **6.1 cm** |
| Vertical (beyond baseline) | 0.01 | **13.4 cm** |

### Observed real overflow on the `3_1_18_3` overlays

From visual inspection of frames 0, 25, 28, 30, 32, 35, 49 (all overlays in `/tmp/court_overlay_3_1_18_3/overlays/` during the audit; frame 32 persisted in scratch):

| Scenario | Approximate offset past the doubles line |
|---|---|
| Neutral stance, feet on baseline | 0 |
| Retreat for smash setup, feet behind baseline | **50-100 cm** |
| Airborne at peak smash (player centre) | **75-150 cm** past baseline |
| Airborne peak smash (projected position, inflated by `H_z × tan(θ)` error from the Padel paper geometry) | additional **70-170 cm** beyond the body centre offset |
| Hard lunge past doubles sideline | **30-80 cm** |

The current `eps = 0.01` buffer is roughly **1/8 to 1/20** of the standing-behind-baseline offset, and an even smaller fraction of the inflated projected-position offset during airborne smashes. Any detection where the player is standing behind the baseline (the typical smash setup) is rejected by the current filter.

### Buffer size that is actually needed

Three bounding estimates, all pointing to a similar answer:

- **Observed maximum** on `3_1_18_3`: ~150 cm past baseline (airborne peak, before projection amplification). The projected position under airborne amplification can go further, up to ~300 cm effective displacement at the far edge for a 0.7 m jump, but picks don't need to trust the projection beyond the anchor's tolerance for that to work.
- **BWF international-competition minimum run-off**: 2 m back / 1 m sides. Any legitimately-in-play stance, even the most extreme retreat, lies inside this envelope.
- **`sticky_anchor` plan's `generous_margin = 0.15`**: ~91.5 cm horizontally / ~2.01 m vertically. Matches BWF run-off on both axes. Covers every observed offset on `3_1_18_3` with headroom.

### Implications for `sticky_anchor` hyperparameters

- **`generous_margin = 0.15` is defensible and shouldn't be widened** without fresh evidence. It matches BWF run-off and covers all observed offsets.
- **`eps = 0.01` is retained only as the EMA update gate** inside `sticky_anchor` (see step 6 of the algorithm), not as a pick-time filter. In that role it correctly prevents pollution of the EMA by clearly-off-court picks. Do not apply `eps` as a pick gate.
- **`sanity_ceiling = 0.5`** (half-court distance) still comfortably exceeds the worst legitimate airborne projection offset observed here, so the ceiling is not the binding constraint for well-behaved smashes.
- The ~7.5% doubles-tramline region either side of the playing area is in-bounds per the homography, so picks that land there are accepted; only picks well outside the doubles lines (beyond 0.15 either way) trigger the rally-presence check.

## Sticky_anchor design, finalised (2026-04-22)

This section supersedes the "sticky_anchor algorithm" subsection under the Phase 1 block below. The older version records the first-cut design; the spec below is what `apply_heuristic.py` should implement. Rationale and amateur-generalisation notes follow the spec.

### TL;DR in plain language

MMPose returns a list of person detections per frame (players, chair umpire, line judges, audience members that happen to be clearly visible). We need to pick two of them as Top and Bottom. Instead of trying to filter out non-players up front, we pick by **geometry**:

- Each slot has an **anchor** fixed at the middle of its court half (Top's anchor is the middle of the top half, Bottom's is the middle of the bottom half). The anchor is 75% that fixed point and 25% a running average of recent picks for this slot. The fixed part keeps the anchor from wandering off to capture a wrong person; the running part lets it lean slightly toward where the player has actually been.
- For each slot we pick the detection whose projected foot position is closest to that slot's anchor. Bottom picks first (its detections are bigger and more confident), then Top picks from what's left.
- Candidates that sit closer to the OTHER slot's anchor are excluded from this slot's pool, so the two slots can't steal each other's player.
- If the closest candidate is too far away, or if both slots' picks land wildly off court, the slot (or both slots) zeroes for that frame.
- When two candidates are similarly close to an anchor, we use two tiebreakers: drop anyone who looks seated (based on where the knees sit relative to the torso axis) and prefer the larger bounding box.

Why this beats the existing pipeline's filter: the current code rejects a whole frame if a player's projected feet don't land inside the taped court, which kills smash frames because airborne feet project well past the back baseline. The new design keeps those picks as long as they're the clear closest-to-anchor candidate; it only refuses to let off-court picks update the running average (so the anchor can't drift to a place the player isn't actually standing).

The heuristic runs on the raw MMPose output stored on disk; the expensive MMPose extraction runs once and we iterate heuristic variants cheaply on top. Output files match the existing `_pos / _joints / _failed` schema so collation and training code downstream don't change.

### Per-video setup (once per clip, using the homography)

- `halfcourt_centre[TOP] = ((bL + bR) / 2, bU + (bD - bU) / 4)` normalised.
- `halfcourt_centre[BOTTOM] = ((bL + bR) / 2, bU + 3 * (bD - bU) / 4)` normalised.
- `bL`, `bR`, `bU`, `bD` are the court borders from `pipeline.court_utils.get_court_info`.
- On ShuttleSet the canonical rectangle collapses these to (0.5, 0.25) and (0.5, 0.75). For amateur data they derive from whatever canonical rectangle that video's homography defines, so the formula is already data-adaptive.
- Initialise `ema[TOP] = halfcourt_centre[TOP]` and `ema[BOTTOM] = halfcourt_centre[BOTTOM]`.

### Per-frame algorithm

**A. Build candidate pool (once per frame):**

1. Filter the raw detections to those with `bbox_score > score_filter` (default 0.2).
2. For each surviving detection, project its bbox bottom-centre through the homography to normalised court coords. Store as `candidate.court_base_pos`.

**B. Compute both effective anchors (once per frame, before either slot's pick):**

3. For each slot `s` in `(BOTTOM, TOP)`: `effective_anchor[s] = 0.75 * halfcourt_centre[s] + 0.25 * ema[s]`.
4. For each candidate in the pool and each slot, compute `D(candidate, s) = euclidean(candidate.court_base_pos, effective_anchor[s])`.

**C. Process each slot, Bottom first then Top:**

For `s` in `(BOTTOM, TOP)` with `other = the other slot`:

5. Pre-filter the candidate pool for this slot:
   1. Drop candidates with `D(candidate, s) > sanity_ceiling` (default 0.6 normalised).
   2. Drop candidates that are closer to the OTHER slot's anchor than to this slot's own anchor (`D(candidate, other) < D(candidate, s)`). In other words, each candidate is only eligible for whichever slot's anchor it is closer to. Prevents cross-half capture when the other slot's player happens to sit geometrically closer to this anchor than our own player does. (Voronoi partition, named once for precision; referred to below as the closer-to-own-anchor rule.)
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

### Output schema

Matches the existing pipeline so `collate_npy` reads sticky_anchor output unchanged:

- `_pos.npy`: `(F, 2, 2)` normalised court positions per slot, ordered (TOP, BOTTOM).
- `_joints.npy`: `(F, 2, 17, 2)` bbox-diagonal-normalised keypoints per slot.
- `_failed.npy`: `(F,)` bool, True where either slot was zeroed this frame.

### Hyperparameters

All exposed as `apply_heuristic` CLI args, ShuttleSet defaults shown.

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

### Output directory conventions

Strict separation: the raw extracts and the primary committed filtered extract are never overwritten.

Paths are referenced via the `.env` convention Curtis established for `pipeline.data_access` (see `.env.example` at the repo root and `pipeline/data_access.py`). The relevant variable is:

```
BST_MMPOSE_NPY_DIR=/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat
```

Under the same `ShuttleSet_data_merged_25/` parent on engelbart, the per-clip flat dirs relevant to Phase 1 are:

```
{parent_dir}/
  dataset_npy_between_2_hits_with_max_limits_flat/                  # primary committed, read-only (= $BST_MMPOSE_NPY_DIR)
  dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/       # raw N=16 extract, read-only
  dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1_n8/    # historical N=8 raw, read-only
  dataset_npy_between_2_hits_with_max_limits_flat_failsafe_gate/    # byte-identity gate output, scratch
  dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor/  # new, written by apply_heuristic
```

Where `{parent_dir}` is the directory portion of `BST_MMPOSE_NPY_DIR` (`/scratch/comp320a/ShuttleSet_data_merged_25/` in current practice). The `_h_<heuristic>` suffix extends the existing flat-dir naming consistent with the `_raw_phase1` / `_raw_phase1_n8` extensions already in use.

`apply_heuristic.py` refuses to write unless `--output-dir` is distinct from both `--raw-dir` and `BST_MMPOSE_NPY_DIR`. Two-line guard against typos destroying data we can't cheaply recompute (1,716 clip re-extract is ~20 min V100 time, and the committed extract is the baseline for every comparison).

**Downstream collated dir** (for the Phase 1 mixed re-train, produced by Step 3 in `prepare_train_on_shuttleset.py`): uses the post-2026-04-21 short naming convention.

- Parent: `ShuttleSet_data_{taxonomy}/` under the preparing-data root (on engelbart, under `/scratch/comp320a/ShuttleSet_data_une_merge_v1/`).
- Dir name: `npy_[3d_][seq{N}_]{ablation_id}`, where `ablation_id = {taxonomy}_{split_column}_{drop}` by default.
- For the V4-analog mixed re-train: `npy_une_merge_v1_split_v2_dropunk_h_sticky_anchor`. The `_h_sticky_anchor` suffix on the ablation_id distinguishes the heuristic-applied data from the V4 committed baseline (`npy_une_merge_v1_split_v2_dropunk`).
- For the V3-analog mixed re-train (using merged_25 + `split_bst_baseline` + keepunk): `npy_merged_25_split_bst_baseline_keepunk_h_sticky_anchor` under `ShuttleSet_data_merged_25/`.

**No scratch-dir renames required**: the three existing flat dirs on scratch (`..._flat`, `..._flat_raw_phase1`, `..._flat_raw_phase1_n8`) already match the current flat-dir naming convention. The new short naming applies only to collated dirs, which are produced fresh by the collation step and tagged with an ablation_id suffix per config. Older long-named collated dirs referenced by V3/V4 manifests should be left alone unless those manifests are also being rewritten.

### Byte-identity gate module

`failsafe_bst_mmpose_zeroing_check_equivalence.py` lives alongside `apply_heuristic.py` and serves as the plumbing sanity check before any sticky_anchor output is trusted:

- Replicates the existing `check_pos_in_court + detect_players_2d` filter on a sample subset of the raw extract. Sample size ~50-100 clips, chosen to cover several video IDs and camera setups.
- Produces `_pos / _joints / _failed` outputs in the same schema as the existing pipeline.
- Compares byte-for-byte against the committed filtered extract for those clips.
- Any diff invalidates sticky_anchor output run through the same plumbing until fixed.

### Modules to write for Phase 1

- `src/bst_refactor/stroke_classification/preparing_data/apply_heuristic.py`: CLI entry point, dispatches to named heuristic modules.
- `src/bst_refactor/stroke_classification/preparing_data/heuristics/__init__.py`: name-based dispatch registry.
- `src/bst_refactor/stroke_classification/preparing_data/heuristics/sticky_anchor.py`: the spec above.
- `src/bst_refactor/stroke_classification/preparing_data/failsafe_bst_mmpose_zeroing_check_equivalence.py`: byte-identity gate.

### Execution checklist (pointers into the rest of this doc and the repo)

Bridges the finalised design above to the practical details scattered across other sections and the existing codebase. Skim before implementing `apply_heuristic.py`; each entry is a concrete dependency or cross-reference a fresh session would otherwise have to hunt for.

**Raw extract file schema** (per clip, input to `apply_heuristic.py`). Full detail in `## Approach > Output format` below. Summary:

- `{stem}_raw_kps.npy` : `(F, N_max, 17, 2)` float32, NaN-padded beyond `ndet[f]`.
- `{stem}_raw_bboxes.npy` : `(F, N_max, 4)` float32, NaN-padded.
- `{stem}_raw_scores.npy` : `(F, N_max)` float32, NaN-padded.
- `{stem}_raw_kp_scores.npy` : `(F, N_max, 17)` float32, NaN-padded.
- `{stem}_raw_ndet.npy` : `(F,)` int8 detection count per frame; also the resume marker saved last by `raw_extract.py`.

Canonical Phase 1 raw-extract directory on engelbart: `$BST_MMPOSE_NPY_DIR/../dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/`. `N_max = 16` in the committed extract.

**Projection helpers**: reuse `pipeline.court_utils`. Do not reimplement.

```python
from pipeline.court_utils import get_court_info, to_court_coordinate, normalize_position
court = get_court_info(homo_df, vid)                                    # once per video
court_raw  = to_court_coordinate(pixel_xy_2xN, vid, all_court_info, res_df)
court_norm = normalize_position(court_raw, court)                        # normalised [0,1] coords
```

`halfcourt_centre[s]` derives directly from `court['border_L'/R'/U'/D']`. Bbox bottom-centre is `((x1+x2)/2, y2)` in pixel space, then passed through the two calls above.

**Joint normalisation**: reuse `normalize_joints(keypoints, bbox)` from `stroke_classification.preparing_data.prepare_train_on_shuttleset`. Unchanged from the committed extract path, so `_joints.npy` outputs produced by the `current` variant will be byte-identical to the committed extract.

**Byte-identity gate protocol** (run BEFORE trusting any `sticky_anchor` output):

- Sample: 50 clip stems from `scratch/architecture_notes/busted_hit_zone_clips_phase1.txt` (the 1,716 hit-zone list). Lex-sort, take every `len // 50`-th stem. Deterministic, no seeding; spreads across video IDs. Draws from the busted list rather than clips_master.csv because raw extracts only exist for those 1,716 stems; sampling from clips_master would shrink the comparable set to the intersection regardless.
- Run `apply_heuristic.py --heuristic current` on those stems against the raw extract, writing to `..._flat_failsafe_gate/` (never the committed flat dir).
- For each stem's three output arrays, compare against the matching arrays at `$BST_MMPOSE_NPY_DIR`:
  - `np.array_equal` on `_failed.npy` (bool).
  - `np.allclose(rtol=0, atol=1e-5)` on `_pos.npy` and `_joints.npy` (float; the tolerance absorbs float32 projection-chain non-associativity between the two code paths).
- On any mismatch: stop and investigate the plumbing BEFORE trusting `sticky_anchor`. Usual suspects: keypoint-index ordering, bbox row order when multiple on-court people exist, `normalize_joints` vs `normalize_position` step order, resolution-scale application.

Canonical gate command (engelbart paths; run from `src/bst_refactor/stroke_classification/`):

```
python -m preparing_data.failsafe_bst_mmpose_zeroing_check_equivalence \
    --raw-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1 \
    --busted-stems-file scratch/architecture_notes/busted_hit_zone_clips_phase1.txt \
    --clips-csv notebooks/clips_master.csv \
    --scratch-output-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_failsafe_gate
```

`--committed-dir` is auto-detected from `$BST_MMPOSE_NPY_DIR` when unset. `apply_heuristic` side-effect-imports `pipeline.data_access`, which auto-loads repo-root `.env`, so the collision guards fire without a prior shell export.

**Mixed retrain plumbing** (after `sticky_anchor` runs on the full 1,716 clips):

- Build a symlink-merged flat dir at `$BST_MMPOSE_NPY_DIR/../dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor_phase1_merged/`.
- For each stem in `clips_master.csv` with `split_v2 in ('train','val','test')`:
  - If stem is in `busted_hit_zone_clips_phase1.txt`: symlink the three `sticky_anchor` outputs from the `..._flat_h_sticky_anchor/` dir into the merged dir.
  - Otherwise: symlink from `$BST_MMPOSE_NPY_DIR` (the primary committed flat dir).
- Collate via `python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose --clip-npy-dir <merged_dir>` with the Phase 1 ablation_id (`npy_une_merge_v1_split_v2_dropunk_h_sticky_anchor` under `ShuttleSet_data_une_merge_v1/`).
- Retrain V4 via `bst_train.py` pointing at the new collated dir. 5 serials, same hyperparameters as the committed V4 run.

**Decision gate** (unchanged from `## Phase 1 > ### Decision gate` below): conjunction of (a) >25% relative reduction in zeroing rate on target classes, (b) no >5% relative regression on non-target classes, (c) >=0.02 min-F1 lift on target-class aggregate OR >=0.005 macro-F1 lift overall. All measured against committed V4.

**Target ablation**: Phase 1 retrains the V4-analog only (`une_merge_v1` + `split_v2` + `dropunk`). V3-analog (`merged_25` + `split_bst_baseline` + `keepunk`) is deferred to Phase 2 unless the V4 signal is ambiguous and a V3 cross-check is needed. Committed V4 baseline manifest: `src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260420_171101/`.

**Ordered task list for the next session**:

1. Commit the current state (scripts + plan docs) so the fresh session starts from a clean `main`.
2. Scaffold `heuristics/__init__.py` and `heuristics/current.py` (the byte-identity-gate variant).
3. Scaffold `apply_heuristic.py` CLI, wired to dispatch to `current` and (stub) `sticky_anchor`.
4. Scaffold `failsafe_bst_mmpose_zeroing_check_equivalence.py`.
5. Run the byte-identity gate on the 50-clip sample. Iterate on plumbing until it passes exactly.
6. Implement `heuristics/sticky_anchor.py` per the "Per-frame algorithm" spec above.
7. Run `apply_heuristic.py --heuristic sticky_anchor` against the full 1,716-clip raw extract.
8. Build the symlink-merged flat dir and collate.
9. Retrain V4-analog (5 serials).
10. Compare against the committed V4 baseline manifest. Apply decision gate.

### Design rationale and decision log (for report context)

Captures the reasoning behind each non-obvious choice. Intended as a reference for writing up the design in the report, so each entry leads with the decision, then the alternative that was considered, then the empirical or geometric evidence that settled it.

**N_max = 16 (raised from the original 8).**

The first raw extract at N_max = 8 saw over-detection warnings on 193 of 222 clips (87%). The busted subset is specifically over-represented in crowded-frame clips because those are exactly the cases where the original heuristic rejected too much. At N_max = 16, only 0.79% of frames (~780 of ~98,370) hit the cap, so raising further buys nothing. Storage per raw extract was already trivial (~160 KB per clip at N=8, doubles at N=16).

**Homography is calibrated to the full outer (doubles) taped court.**

Established by a code + CSV audit plus a visual overlay on clip `3_1_18_3`. Every one of the 44 ShuttleSet videos maps its 4 annotated corners to an identical canonical rectangle of 300 by 660 in court space. The length-to-width ratio is 2.2000, which matches the physical doubles court (6.10 m by 13.40 m, ratio 2.1967) to three decimal places. Singles court (2.587) and BWF run-off zone (2.148) ratios are ruled out. Full detail in the "Court-space geometry and buffer sizing (2026-04-22)" section above.

This is load-bearing for every threshold in the algorithm. One unit of normalised court space is 6.10 m horizontally and 13.40 m vertically.

**The existing `eps = 0.01` filter rejects legitimate play.**

`eps = 0.01` normalised translates to 6.1 cm off the doubles sideline and 13.4 cm off the baseline. Competitive singles players routinely stand 50-100 cm behind the baseline for smash setups (observed across frames 25, 28, 30, 32, 35, 49 of `3_1_18_3` during a Top_smash clip) and lunge 30-80 cm past the doubles sidelines on defensive reaches. Under airborne projection error (Padel `H_z * tan(θ)` geometry for a 0.7 m jump at the far edge), foot projections amplify another 70-170 cm off-court. The current filter is roughly 1/8 to 1/20 of real-play overflow; effectively zero slack.

This motivated widening `sanity_ceiling` from 0.5 to 0.6: anchor-distance of ~0.51 is observed on the apex-jump frame of `16_1_42_4` (frame 42 of a Top_smash scissor kick), and 0.5 would reject it marginally.

**No movement / stationarity signal in Phase 1.**

Considered as a way to distinguish stationary refs from moving players. Ruled out because:
- A stationary player waiting for a serve or pausing between shots would fail a stationarity gate; a threshold loose enough to let them pass is loose enough to let most refs through.
- The actual mechanism that excludes non-players is the on-court projection test combined with the EMA. In ShuttleSet, refs, chair umpire, line judges, and audience all project outside the court rectangle, so they never enter the candidate pool for anchor-distance purposes. The EMA captures "where the slot's player has been trending" without needing a motion proxy.

Amateur may need to revisit this if coaches / kids / bystanders start projecting on-court.

**Score is used only as a noise-floor filter, not as a divisor.**

The original plan used `effective_distance = D / max(score, eps)` so that high-score detections were preferred. Empirical evidence contradicts this: RTMDet's `bbox_score` is driven primarily by bbox size, bbox completeness, scene prior, and lighting. Chair umpires and front-row audience are stationary and well-framed, so they score as high as or higher than motion-blurred airborne players. A score divisor would systematically penalise the real player on exactly the frames we care about.

Final use: `score_filter = 0.2` as a noise floor to keep the candidate pool clean. No weighting beyond that.

**EMA resets on zeroed slots instead of freezing.**

Freeze creates a stale-anchor capture risk after long zeroings. Concrete walk-through: if Top's EMA is stuck at (0.1, 0.1) from a pre-zero pick, the slot zeros for ~10 frames while the real player runs to the opposite corner, and a nearby line judge projects at (0.4, 0.2). On reappearance, distance from effective_anchor to the real player is ~0.75 vs ~0.01 for the line judge. Freeze: line judge wins by 60x. Reset to `halfcourt_centre`: line judge still closer, but only ~7x.

The 25% EMA weight is the only thing lost on reset, which is cheap: it's the short-memory lean toward recent picks. The 75% `halfcourt_centre` weight is the structural anchor and is unaffected. No decay-to-prior intermediate option was considered necessary; reset is strictly simpler and carries no new hyperparameter.

**No second sanity ceiling on max displacement from last picked position.**

Considered to catch "wrong pick at clip start" cases. Rejected because it creates a silent persistent-capture failure: if the EMA initially locks onto a line judge, a "max displacement" gate rejects the real player when they reappear elsewhere because they're far from the line judge's stationary position. The weighted prior (0.75) is already the correct defence for the wrong-start case, because `effective_anchor` stays near the court-half midpoint regardless of EMA pollution.

**Body-frame projection handles sitting, not 3D pose, not shin angle.**

Discriminating seated refs from standing/airborne players needs some signal for "legs perpendicular to body axis" vs "legs in body-down direction." Three routes considered:

- 3D pose via MMPose's `human3d`: provides true 3D keypoints so the "is knee-hip-knee plane horizontal in the world frame?" test is literally computable. Costs ~50-85 extra minutes at the per-clip model-reload step (documented MMPose 1.3.2 bug). Quality on broadcast footage is modest; `human3d` was trained on well-lit single-subject videos. Rejected on cost-benefit for Phase 1.
- Shin angle (knee-to-ankle vector vs body_up): doesn't discriminate cleanly because shins are usually vertical in 3D whether the person is sitting (with feet on footrest) or standing. Introduces new failure modes on seated-with-legs-crossed / feet-tucked positions. Rejected as adding complexity without coverage.
- Body-frame projection (2D): projects the knee-offset-from-hip onto the hip-to-shoulder axis. For a sitting person the knees are perpendicular to the body axis so the ratio is ~0; for a standing or airborne player the knees are in the body-down direction so the ratio is around -0.7 to -0.9. Empirically verified on `16_1_42_4`: frame 42 (apex smash scissor kick, legs kicked back near hip level) gives `body_frame_ratio = -0.874`, cleanly "not sitting". Frame 32 (pre-jump crouch, feet at knee level) gives -0.734, also "not sitting". Chosen for Phase 1.

**No confidence gates on the sitting test.**

Player knees are generally less occluded than ref knees in ShuttleSet broadcasts. A joint-score gate would gate out exactly the mostly-seated refs we want to filter while letting standing players through. The body-frame ratio is cheap enough (handful of multiplications per candidate) that a gate saves nothing.

**Tiebreaker invoked only on close-to-tie anchor distances, with sitting-filter + bbox area.**

Bbox area alone isn't the primary signal: tower umpire bboxes are large because the elevated position is fully visible, and seated-ref bboxes can be widened by instrument stands captured in the detection context. Pixel-space size is also perspective-biased (Bottom player always larger than Top player). Confidence isn't a useful primary either, for the same score-doesn't-discriminate reason as the selector itself.

The tiebreaker only fires when multiple candidates are within `tiebreaker_tol = 0.05` of the winning anchor distance. In ShuttleSet singles this is rare because typical frames have one on-court candidate per slot. Order: drop sitting candidates first (body-frame test), then break the remaining tie by largest bbox area. If the sitting filter drops everyone, revert to the original `argmin D` pick (a sitting-dominated tie means we have no better signal; fall back to pure proximity).

**Bottom-first pick order.**

Bottom player sits closer to the camera, so their bboxes are larger, scores are higher, and detections are more reliable. Locking in the higher-confidence assignment first reduces the Top pool by a detection we're most sure about, rather than the reverse. Bottom-first greedy combined with the closer-to-own-anchor pre-filter (next entry) covers the cross-capture cases we'd otherwise worry about.

**Closer-to-own-anchor pre-filter on the candidate pool (cross-half capture guard).**

With `sanity_ceiling = 0.6`, a legitimate but geometrically adversarial frame can let one slot's argmin grab the other slot's player. Concrete example: Bottom player in their own deep corner at (0.2, 0.95), Top player attacking at the net on their side at (0.5, 0.48). Distances:

- From `Bottom_prior` at (0.5, 0.75): real Bottom = 0.36, real Top = 0.27.
- Both under 0.6, so both pass `sanity_ceiling`.
- Bottom's argmin picks the Top player (0.27 beats 0.36). Top's remaining pool then has only the Bottom player at distance 0.76 from `Top_prior`, which exceeds the ceiling, so Top zeroes. The frame is marked failed, but slot assignments were wrong.

Fix: each candidate is eligible for slot `s` only if `D(candidate, s) <= D(candidate, other_slot)`. In the scenario above, the Top player near the net is closer to `Top_prior` (0.23) than to `Bottom_prior` (0.27), so the closer-to-own-anchor rule drops them from Bottom's pool entirely. Bottom then picks the real Bottom player unambiguously, and Top picks the near-net Top player unambiguously.

This is effectively a dynamic midline split that follows the shifting EMA anchors frame by frame. Zero extra compute because `D(candidate, other_slot)` is already being computed for the other slot's own pre-filter.

**Irrecoverable clips stay in the Phase 1 denominator.**

Some fraction of the 1,716 hit-zone-busted clips are fundamentally irrecoverable by any heuristic operating on raw MMPose output: broadcast extreme close-ups (body cropped to a wrist), side-on framings with no court visible, cuts to a different subject. Position: keep them zeroed after sticky_anchor runs, keep them in the Phase 1 decision-gate denominator. Zeroed irrecoverable clips at worst regularise the model; at best teach the transformer to attend around bad data. No manual inspection pass over the tail, no subset carve-out.

### Amateur generalisation notes (for next trimester)

The design was shaped so most decisions carry into amateur data without surgery. What stays, what needs re-derivation, what may need expanding.

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

- `sanity_ceiling`: 0.6 is tuned to ShuttleSet's high-behind-baseline camera geometry. Taller or more oblique cameras inflate the apex-jump projection error differently. Candidate approach: measure per-video the 99th percentile of stable-pick anchor-distance during the first N seconds, set the ceiling to that plus a margin.
- `generous_margin`: 0.15 matches professional venue run-off of 1-2 m. Amateur courts vary (equipment at courtside, different boundary conventions). Candidate approach: per-video observed-play-extent quantile.
- `score_filter`: 0.2 is conservative for well-lit broadcast; phone-grade amateur footage has lower baseline scores and may need 0.1 or 0.05 to retain legitimate detections.

None of these need Phase 1 work; hooks exist via CLI args.

**May need expanding if empirical failures appear:**

- Bbox-size tiebreaker could pick up detector artefacts more often in amateur (stands, chair structures, partially-visible bodies). Mitigation: add a bbox aspect-ratio sanity check (height / width between ~1.2 and ~4) before using bbox area. Alternative: confidence-weighted joint bounding-rectangle area, which is insensitive to detector context-capture because keypoints are anatomically anchored. Both deferred unless we see ties going wrong.
- Movement / stationarity signal: omitted for Phase 1 because the on-court projection cleanly excludes refs in ShuttleSet. Amateur has more on-court interlopers (coaches wandering in between rallies, kids crossing, someone retrieving a shuttle). If the on-court test starts admitting non-players near the anchor, interframe displacement (in court space, via nearest-neighbour proxy since MMPose detections have no persistent identity) becomes the natural discriminator. Hook: the per-slot assignment would consult the previous frame's picks.
- 3D pose via `human3d`: only worth the compute if the 2D body-frame ratio starts misclassifying under a new camera angle. High cost, low likely benefit.

**Out of scope even in the amateur phase:**

- Doubles or mixed formats: the two-slot architecture breaks. Different design entirely; a separate Phase when the data arrives.
- Multi-camera composition (behind-net, net-level, ground-level): projection error profile changes enough that ShuttleSet-derived thresholds become meaningless. Needs per-camera-convention calibration.

## Context

V3 / V4 ablations (une_merge_v1 + drop_unknown) plateau on min-F1 around 0.38-0.43, with the worst-performing classes being Top_wrist_smash + Top_smash and their Bottom counterparts, then clears / drops / long_services / return_nets. `validate_zeroed_frames.py` output on the merged_25 flat dir showed the same classes are over-represented in the high-MMPose-fail bucket: smash 13.75% pooled fail rate (Top_smash 24.33% stratified), wrist_smash 9.93%, unknown 54.79% (expected; garbage class), everything else < 10%.

**Diagnosis**: the current MMPose extraction zeroes out a frame entirely if either player fails a combined filter in `check_pos_in_court` + `detect_players_2d` (in `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`):

1. MMPose detects fewer than 2 people in the frame, OR
2. After projecting detected people's ankle midpoints through the court homography, fewer or more than exactly 2 people land inside the soft court rectangle (`eps = 0.01` margin on each side).

For a smash clip where the acting player is airborne and their feet project off-court in the homography frame, the current heuristic zeroes both players' joints + positions for that frame. The model sees a zero vector during the single most informative part of the clip. TrackNet independently sees the shuttle fine in those frames, which is the main pointer that MMPose detects the pose but the projection / filter is rejecting it.

**Geometric confirmation**: the Padel paper's `H_z * tan(θ)` derivation (Javadiha et al. Sensors 2021) quantifies the projection error for an above-ground keypoint at broadcast camera angle. At the far-court edge, the error amplification factor is ~4.64 per unit of vertical displacement. For a 0.5-0.7 m jump, projected feet drift ~0.17-0.24 normalized units off-court, well beyond the current `eps = 0.01` grace. This matches the observed class skew in the 222 busted clips: 32 train + 26 val + 12 test Top_smash alone (~31.5% of busts), Top-side dominant throughout, Bottom-side rare. Far-camera players suffer worse projection error; near-camera players barely move.

## Goals

Two measurable outcomes. Both must move for a heuristic change to be worth rolling out.

1. **Reduced zeroing rate** on the worst-affected classes. Measured via a re-run of the scanner (or a minor variant of `validate_zeroed_frames.py`) against the re-processed data, per-class.
2. **Improved model min-F1** on those same classes. Measured via re-running V3 (split_bst_baseline) + V4 (split_v2) with the re-processed data and comparing min-F1 against the committed baselines.

Target classes, in order of priority:
- Top_wrist_smash, Bottom_wrist_smash
- Top_smash, Bottom_smash
- Top_clear, Bottom_clear
- Top_drop, Bottom_drop
- Top_long_service, Bottom_long_service
- Top_return_net, Bottom_return_net

Unknown class is excluded from both measurement and scope; garbage class by construction.

## Approach: decoupled raw extraction + post-processing

### Why decouple

Running MMPose on 33k clips takes ~50 hr on V100. Running it multiple times to iterate over heuristic variants is a non-starter. The heuristic itself (the pick + filter logic) is pure CPU arithmetic on already-computed keypoints and costs sub-millisecond per clip. Decoupling means one expensive MMPose pass per clip, then fast (~seconds) heuristic iteration.

### Output format

**Step 2 raw output** (per clip, new files alongside existing ones):

| File | Shape | Contents |
|---|---|---|
| `{stem}_raw_kps.npy` | `(F, N_max, 17, 2)` | All detected people's keypoints per frame, padded with NaN to `N_max`. |
| `{stem}_raw_bboxes.npy` | `(F, N_max, 4)` | All detected bounding boxes, padded. |
| `{stem}_raw_scores.npy` | `(F, N_max)` | Per-person detector confidence. Used for disambiguation in the pick stage. |
| `{stem}_raw_kp_scores.npy` | `(F, N_max, 17)` | Per-joint MMPose confidence. Used for ankle-confidence-based projection fallback; preserved in full for possible training-side use. |
| `{stem}_raw_ndet.npy` | `(F,)` int | Number of people detected per frame. `raw_kps[f, :ndet[f]]` is the valid slice. |

`N_max = 8` is a safe upper bound for badminton broadcast footage (typically 2-4 detections per frame). Per-clip raw-output storage: ~160 KB total. Across 222 Phase-1 clips: ~35 MB. Across Phase-2 full 33k clips: ~5.3 GB uncompressed.

The 3D variant (`detect_players_3d` in the current code) is omitted from this pass. `raw_extract.py` includes commented scaffolding for toggling it back on.

**Post-processing output** (per clip, heuristic-specific subdir):

The existing contract: `_pos.npy`, `_joints.npy`, `_failed.npy`. Same shapes as today. Collation is unchanged downstream.

### Step ordering

1. Step 2 (raw extract): writes `*_raw_*.npy` per clip to a new flat dir. This is the expensive GPU step.
2. Step 2.5 (apply heuristic): reads raw, applies a named heuristic, writes `*_pos.npy` / `*_joints.npy` / `*_failed.npy` to a heuristic-specific output dir. Fast (seconds per clip), re-runnable per heuristic variant.
3. Step 3 (collate): unchanged. Points at either the original flat dir or a heuristic-processed dir via `--clip-npy-dir`.

## Phase 0: baseline diagnostic

1. Write `scripts/find_busted_clips.py` (~30-40 lines): walks the flat dir, reads each clip's `*_failed.npy`, computes the fraction of True, filters by threshold (default 0.50), writes stems to an output text file. Takes optional taxonomy/split filter via `clips_master.csv` + `--taxonomy` + `--split-column`.
2. Run on engelbart against the current flat dir. Output: `scratch/architecture_notes/busted_whole_clips_phase1.txt` with ~222 stems.
3. Rsync the list to local for review.
4. Spot-check by downloading 12 clips via SFTP and watching them locally. Non-blocking; handled separately.

**Split distribution of busted clips** (computed from the existing 2026-04-21 analysis):
- train: 110
- val: 49
- test: 33
- (~30 unknown-prefixed clips excluded by scope)

Val+test coverage is ~37% of busted clips, sufficient for a mixed re-train signal on val/test metrics.

**Class skew** (Top-side dominant, consistent with the far-edge homography hypothesis):
- 32 train + 26 val + 12 test `Top_smash` (70 clips, ~31.5% of busts)
- Top_wrist_smash + Bottom_wrist_smash + Bottom_smash together add ~23 more smash-variant clips
- Bottom-side busts are rare (handful total)

## Phase 1: raw extract + `sticky_anchor` on the busted subset

### Scope

All 222 clips with `fail_rate > 0.50` in the current extract, excluding unknown. MMPose wall time ~20 min on V100 (222 * ~5s).

### Code changes

New modules in `src/bst_refactor/stroke_classification/preparing_data/`:

- **`raw_extract.py`** (new module, sibling of `prepare_train_on_shuttleset.py`): invokes MMPose without the on-court filter and saves the five-array raw schema above. Accepts a clip-stem list for subset extraction. 3D path scaffolded in commented-out lines with a toggle note.
- **`apply_heuristic.py`** (new CLI): reads raw per-clip files + a heuristic name, writes the final `_pos/_joints/_failed` triple to a heuristic-specific flat dir. CLI sketch:
  ```
  python -m preparing_data.apply_heuristic \
      --raw-dir /scratch/.../dataset_npy_..._flat_raw_phase1 \
      --output-dir /scratch/.../dataset_npy_..._flat_h_sticky_anchor \
      --heuristic sticky_anchor \
      --clips-csv notebooks/clips_master.csv
  ```
- **`heuristics/`** (new package):
  - `__init__.py`: registry / name-based dispatch.
  - `current.py`: reference gate. Replicates existing behaviour; must produce bit-identical `_pos/_joints/_failed` to committed files for any clip the current pipeline also processed.
  - `sticky_anchor.py`: primary variant; detailed below.

New script:

- **`scripts/find_busted_clips.py`**: ~30-40 lines, described in Phase 0.

### Heuristic variants

Two variants implemented in Phase 1:

1. **`current`** (reference gate). Sanity-gates the raw extract + apply_heuristic plumbing. If this variant doesn't produce bit-identical output against the committed filtered extract, the plumbing is wrong and nothing else should be trusted. No surprises expected.

2. **`sticky_anchor`** (primary variant). Detailed in the next subsection.

Plan B, not scaffolded unless needed:

- **`body_length_fallback`**: if `sticky_anchor` is ambiguous or catastrophically wrong on Phase 1 measurements, this simpler variant (in-court test OR within-body-length of last-in-court-position per slot) is a fallback. The fallback rule is weaker than `sticky_anchor` on airborne-at-far-edge cases (body-length threshold is geometrically tight for peak-jump projection offsets) and has no defense against cutaways, but it's closer to the existing filter structure and might be sufficient.

### `sticky_anchor` algorithm

> **Superseded (2026-04-22)**: the concrete algorithm, hyperparameters, and output contract are now the "Sticky_anchor design, finalised (2026-04-22)" section higher in this doc. The text below remains as the first-cut rationale. Notable differences from the finalised version: score is no longer used as a divisor in the pick; `sanity_ceiling` widened from 0.5 to 0.6 after the apex-jump empirical check; `halfcourt_centre` replaces the hardcoded `court_side_prior` and derives per-video from the homography borders; EMA resets (rather than freezes) on zeroed slots; a body-frame sitting test + bbox-area fallback is used as the tiebreaker when multiple candidates are within 0.05 of the winner's anchor distance; Bottom slot is picked before Top; a closer-to-own-anchor pre-filter drops candidates that are closer to the OTHER slot's anchor than to this slot's own anchor (prevents cross-half capture).

**Motivation**. The current filter rejects all detections in a frame if either player's ankle midpoint doesn't project inside the court. During a legitimate smash where the acting player is airborne, homography error displaces the projected position ~0.24 normalized off-court at the far edge. The current heuristic zeros the entire frame despite MMPose correctly detecting the player. `sticky_anchor` reframes the problem from *filtering* to *per-slot tracking*: pick the most plausible detection for each slot via proximity to a tracked per-slot anchor, independent of whether the pick happens to project in-court.

**Per-clip state** (4 floats total):
- Top EMA: 2 floats, initialized to `(0.5, 0.25)`.
- Bottom EMA: 2 floats, initialized to `(0.5, 0.75)`.

**Fixed priors**:
- Top court-side midpoint: `(0.5, 0.25)` (horizontal centre, vertical quarter-down; centre of the top half of normalized court).
- Bottom court-side midpoint: `(0.5, 0.75)` (centre of the bottom half).

**Per frame, for each slot `s` in {Top, Bottom}**:

1. **Project each detection's feet to court coords**.
   - Ankle keypoints: COCO joints 15 and 16.
   - If both `kp_scores[i, 15] >= 0.3` AND `kp_scores[i, 16] >= 0.3` (ankle confidence threshold), projection point = ankle midpoint in pixel space.
   - Else projection point = bbox bottom-center: `((bbox[0] + bbox[2]) / 2, bbox[3])`.
   - Apply the existing `to_court_coordinate` + `normalize_position` pipeline to produce the 2D projected position in normalized court coords.

2. **Compute effective anchor for slot `s`**:
   ```
   effective_anchor[s] = 0.75 * court_side_prior[s] + 0.25 * ema[s]
   ```

3. **Pick**: select detection `d*` minimizing
   ```
   effective_distance(d) = euclidean(projected[d], effective_anchor[s]) / max(det_scores[d], eps_score)
   ```
   subject to `d*` not already assigned to the other slot. Top picks first; Bottom excludes the Top pick. `eps_score = 1e-3` avoids divide-by-zero on broken frames.

4. **Sanity ceiling**: if `effective_distance(d*) > 0.5`, the pick is pathologically far (more than half a court away from anchor even after confidence weighting). Zero slot `s` this frame.

5. **Write outputs** (if slot survived):
   - `_pos[f, s, :]` = `projected[d*]`.
   - `_joints[f, s, :, :]` = `normalize_joints(keypoints[d*], bboxes[d*])` (existing helper, unchanged).

6. **EMA update gate**: only update `ema[s]` if the surviving pick's projected position is inside court by `eps = 0.01` (matches existing `check_pos_in_court` test). Prevents anchor pollution during cutaways (projections outside court) and airborne frames (off-court picks are used in the output but don't update EMA):
   ```
   if -eps < projected[d*][0] < 1 + eps and -eps < projected[d*][1] < 1 + eps:
       ema[s] = 0.1 * projected[d*] + 0.9 * ema[s]
   ```

**After both slots picked, frame-level rally-presence check**:

7. If neither surviving pick lies within the generous court (±0.15 margin: `-0.15 <= x,y <= 1.15`), zero both slots for this frame. Catches the pure-bystander case (cutaway with commentators or similar). The generous margin (0.15) is wider than the 0.01 update gate, so legitimate airborne-at-far-edge picks pass.

**`_failed.npy`**: per-frame-per-slot, True if the slot was zeroed in step 4, step 5 (no valid detection), or step 7.

### Hyperparameters

All geometry-justified; no grid search required for Phase 1.

| Param | Value | Rationale |
|---|---|---|
| `prior_weight` | 0.75 | Keeps effective anchor near court-side midpoint. Caps ref-capture drift at 25% of the way toward an intruder. Singles players average to near the court-side midpoint anyway, so heavy prior is nearly free. |
| `ema_alpha` | 0.1 | Effective half-life ~7 frames. Smooth enough to ignore single-frame noise, responsive enough to follow real play. |
| `sanity_ceiling` | 0.5 | Rejects catastrophic picks (distance > half-court). Tolerates worst-case airborne offset (~0.24). |
| `generous_margin` | 0.15 | Rally-presence check margin. Tolerates airborne picks at far-edge. Narrow enough to catch cutaways and pure-bystander frames. |
| `update_gate_eps` | 0.01 | Matches current code's in-court test, used only as the EMA update gate. |
| `ankle_conf_cutoff` | 0.3 | Below this, ankle keypoints are hallucinated; fall back to bbox bottom-center for projection anchor. |
| `eps_score` | 1e-3 | Divide-by-zero guard on confidence weighting. |

### Measurements per variant

Run each variant on the 222 raw clips. For each produced flat-dir:

1. **Zeroing rate per class** (cheap). A minor variant of `find_busted_clips.py` (or `validate_zeroed_frames.py`) produces per-class fail-rate comparisons. `sticky_anchor` wins on this axis if its fail-rate reduction is meaningful on target classes and doesn't regress on other classes.

2. **Model performance** (bounded but not cheap). Build a mixed-dataset flat dir containing the 32k - 222 unchanged clips plus the 222 re-processed clips. Use **symlink-merge**:
   - A new flat dir where most clip files are symlinks to the main flat dir, and 222 clips' files are real copies from the experimental dir.
   - Collate reads this as normal.
   - `rsync -L` dereferences for cross-machine migrations; or rebuild from components.

   Once mixed, collate (same hyperparams as V4) + re-train V4 on the merged data. Compare min-F1 on target classes vs committed V4.

### Decision gate

`sticky_anchor` graduates to Phase 2 iff:
- Zeroing rate on target classes drops by a meaningful margin (target: > 25% relative reduction), AND
- Zeroing rate on non-target classes does not materially regress (target: < 5% relative increase anywhere), AND
- A mixed-dataset V4 re-train shows min-F1 lift of >= 0.02 on target-classes aggregate, OR macro-F1 lift of >= 0.005 overall.

If `sticky_anchor` meets the zeroing-rate target but fails the min-F1 target, the bottleneck on smash classes is not primarily MMPose data quality. Options: escalate to model capacity / label structure investigation, or trial `body_length_fallback` as Plan B before giving up on the data-fix track.

## Phase 2: full re-extract (if Phase 1 justifies)

1. Re-run Step 2 raw in the chosen mode across the full 33k clips. Wall time ~50 hr on V100. Writes to `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat_raw/`.
2. Run `apply_heuristic.py` with the chosen heuristic, outputting to `..._flat_h_sticky_anchor/`.
3. Collate once per ablation (V3 + V4 split columns) with `--clip-npy-dir ..._flat_h_sticky_anchor/`. Use a new ablation_id suffix to tag the heuristic: e.g. `une_merge_v1_split_v2_dropunk_h_sticky_anchor`. Default `pose_styles` stays `JnB_bone`.
4. Re-train V3 and V4 with 5 serials each. Document the run via the existing run_tracker pattern.
5. Compare mean macro/min/acc across the 5 serials vs the committed baselines. If target-class min-F1 lifts by the Phase-1-observed margin, ship.

## Directory conventions

Scratch-dir layout (engelbart):

```
/scratch/comp320a/ShuttleSet_data_merged_25/
  dataset_npy_between_2_hits_with_max_limits_flat/                              # primary, current heuristic
  dataset_npy_between_2_hits_with_max_limits_flat_raw_phase1/                   # raw MMPose output, 222-clip subset
  dataset_npy_between_2_hits_with_max_limits_flat_raw/                          # raw MMPose output, full 33k (Phase 2)
  dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor/              # sticky_anchor-applied
  dataset_npy_between_2_hits_with_max_limits_flat_h_sticky_anchor_phase1_merged/ # symlink-merged flat dir for Phase-1 mixed re-train
```

Each heuristic dir keeps the same per-clip file schema as the primary dir, so `collate_npy` reads them without modification. Experimental extracts must NOT overwrite the primary flat dir until Phase 2 ships a chosen heuristic and it's been re-trained + validated end-to-end.

## Files to modify / add

### New modules

- `src/bst_refactor/stroke_classification/preparing_data/raw_extract.py`. Borrows most of `detect_players_2d`, replaces the filter block with a raw-save. 3D scaffolding commented out with a toggle note.
- `src/bst_refactor/stroke_classification/preparing_data/apply_heuristic.py`. CLI that reads raw + applies named heuristic + writes the final triple.
- `src/bst_refactor/stroke_classification/preparing_data/heuristics/` package:
  - `__init__.py`: registry / dispatch.
  - `current.py`: reference gate.
  - `sticky_anchor.py`: primary variant.
- `scripts/find_busted_clips.py`: scanner for busted clips, ~30-40 lines.
- `scratch/architecture_notes/busted_whole_clips_phase1.txt`: materialized during Phase 0, one stem per line.

### No modification

- `prepare_train_on_shuttleset.py`: untouched. `raw_extract.py` is a sibling module; the existing filtered path stays bit-identical for reproducibility.
- `collate_npy`: unchanged. Overlay option deferred.
- `bst_train.py`, `bst_infer.py`, `shuttleset_dataset.py`, model code: untouched. Re-train just points at a different collated dir.
- `pipeline/`: unchanged. This investigation is entirely stroke_classification-side.

## Open questions (to resolve during Phase 1)

Resolved since the first-cut plan:

- Split distribution of the 222 busted clips: 110 train / 49 val / 33 test.
- Class skew (Top-heavy) confirms the far-edge homography hypothesis.
- Heuristic variant set reduced to `current` + `sticky_anchor`, with `body_length_fallback` as Plan B.
- Hip-based projection ruled out (geometrically worse than feet).
- Carry-forward ruled out (pattern-signal poisoning).
- Jump-adaptive eps ruled out (subsumed by sticky_anchor's relative proximity framing).

Still open:

1. Does MMPose's raw detection count spike above 2 in practice? If `N_max=8` is ever exceeded, widen it. Measure on a Phase 0 raw-extract sample.
2. Storage + I/O cost of `N_max=8` padding in practice. Re-measure after 10 sample clips.
3. Is `ema_alpha = 0.1` the right value? Default is defensible from the ~7-frame effective half-life; may tune if early-frame convergence is too slow or too responsive.
4. How often does the ankle-confidence fallback (ankle midpoint to bbox bottom-center) fire? If rare in practice, the fallback code path is over-engineered and could be simplified.
5. How often does the rally-presence check fire on the 222 busted clips? If never, the guard is free insurance; if it fires frequently, verify that bystander detection is the actual cause rather than a real-play frame being wrongly classified.
6. Does `sticky_anchor`'s 0.75 prior weight cause anchor rubberbanding during aggressive cross-court play? Unlikely given singles players average to near the court-side midpoint, but worth watching.

## Known limitations of `sticky_anchor`

- **Same-angle replay in cutaway**: a replay frame at near-identical camera angle would let bystander detections pass the in-court update gate, potentially polluting the EMA. Rare in broadcast badminton; not bulletproof.
- **Ball kid / court-crosser during play**: if the real player is off-frame or low-detected AND an intruder is in-court and passes the confidence-proximity test, the intruder could briefly capture a slot. Weighted anchor (0.75 prior) limits damage: real player reclaims on reappearance.
- **Amateur footage**: structurally worse on intrusion cases (refs walk around more, crowd visible, fewer players confidently detected). The design is tuned to ShuttleSet pro scope. Amateur-data extension would need hardening on the rally-presence check and possibly an in-court gate on the picking stage, not just the update stage.
- **Two players simultaneously airborne**: singles rarity; would currently trigger the rally-presence check (neither pick in generous court) and zero both slots. Negligible in practice.
- **Bootstrap with long cutaway intro**: if the clip's first 15+ frames are broadcast padding, picks during padding use the court prior only. The first real in-court detection starts updating EMA; convergence to the player's actual trajectory takes ~5 real-play frames. Slight mis-picks in those early frames but no data loss.
- **Continuity check intentionally absent**: a continuity threshold ("reject a pick > X away from previous pick") would reject legitimate player re-appearances after long invisible gaps. The weighted anchor + sanity ceiling are the intended defense instead.

## Success criteria

Committed only after Phase 2 validates both axes:

- Zeroing rate on the target-class aggregate drops by >= 25% relative on each of (train, val, test) partitions.
- Retrained V4 best-serial min-F1 lifts by >= 0.04 vs committed V4's 0.432 (matching or exceeding V3's 0.381 + the V3 to V4 +0.04 min-F1 gain, so >= 0.47 net).
- Retrained V4 best-serial macro and accuracy do not drop by more than the noise margin across the 5 serials (~0.005).

If Phase 2 achieves the zeroing-rate target but fails the min-F1 target, the conclusion is that the bottleneck on smash classes is not primarily data quality. Escalate to arch work rather than continuing the data-fix track.

## Risks

| Risk | Mitigation |
|---|---|
| Raw extract consumes too much disk space at 33k clips | Measured estimate ~5.3 GB uncompressed across 5 arrays. `np.savez_compressed` or float32 to float16 halves it. Compression cost negligible vs MMPose time. |
| `sticky_anchor`'s EMA gets captured by a persistent in-court intruder | Weighted anchor (0.75 prior) caps drift toward intruder at 25%. Real player reclaims on reappearance. Continuity check dropped; sanity ceiling + rally-presence check are the defenses. |
| Phase 1 mixed re-train produces no meaningful signal because 222 clips are too small a fraction of train | Expected risk if zeroing-rate reduction is the primary signal and model-lift is noisy. Fall back: zeroing-rate drop alone is a sufficient Phase-1 gate to proceed to Phase 2. |
| Heuristic changes compound and become unmaintainable | Keep each variant in its own module under `heuristics/`; no branching by flag inside a single function. Name-based dispatch from `apply_heuristic.py`. Current variant list is minimal (2 modules + optional Plan B). |
| Same-angle replay or ball kid capture | Documented as known limitation. Rally-presence check + weighted anchor limit damage. Not a blocker for ShuttleSet pro scope. |
| Byte-identity gate (`current` variant) doesn't reproduce committed output | Investigate plumbing before trusting `sticky_anchor` results. Common causes: order-of-operations around bbox handling, normalization order, or floating-point differences from the detector vs the filter path. |

## References

- `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`: current `detect_players_2d` (line 237), `check_pos_in_court` (line 202), `to_court_coordinate` (line 120), `normalize_position` (line 142), `normalize_joints` (line 157), and the zeroing decision inside `detect_players_2d`.
- `src/bst_refactor/validation_scripts/validate_zeroed_frames.py`: per-class fail-rate diagnostic.
- `src/bst_refactor/validation_scripts/fail_rate_per_class.py`: stratified by player (Top/Bottom).
- `src/bst_refactor/validation_scripts/zeroed_frames_analysis_outputs/analysis_unemergev1_v2_20260421_1159.txt`: current analysis output confirming 222 busted clips + class skew.
- `src/bst_refactor/stroke_classification/preparing_data/keypoints_schema.md`: COCO joint indices (15 and 16 = ankles, 11 and 12 = hips, relevant for ankle-midpoint + hip-projection discussion).
- `scratch/architecture_notes/mmpose_heuristic/mmpose_bounds_filtering_research.md`: external research. No published ablations of filter strategies found. Key inputs to this design: MonoTrack (CVPRW 2022) relaxed-boundary + closest-to-last-in-court pose; Padel paper (Javadiha et al. Sensors 2021) projection-geometry quantification; TemPose carry-forward vs BST zero-fill contrast. Research agent noted fabrication risk on an earlier pass; this pass verbatim-quoted.
- `scratch/architecture_notes/completed_general_refactors/dir_flatten_refactor.md`: overall plan doc this work sits downstream of.
- V3 and V4 committed run manifests: `src/bst_refactor/stroke_classification/main_on_shuttleset/experiments/run_20260420_141629/` (V3) and `run_20260420_171101/` (V4). Baselines to measure against.

## Out of scope

- Changing the MMPose model (RTMPose-L stays). Retry only if heuristic work fails to lift performance.
- Carry-forward / interpolation (rejected: repeated identical poses corrupt the temporal signal the transformer learns). One-Euro filter reserved as a one-shot Phase-2 option only if heuristic work stalls.
- Jump-adaptive eps via hip-foot detector: subsumed by sticky_anchor's relative-proximity framing; not needed as a separate variant.
- Hip-based court projection (`torso_center` in the original plan): geometrically wrong; projection error for hips is ~4.64x a standing player's height at the far edge, ie worse than airborne feet.
- Per-joint masking at apply_heuristic layer: retain MMPose's anatomical-prior fill; per-joint scores saved for possible training-side use later.
- Flattening the .mp4 clips dir (Phase 3 of the dir-flatten refactor).
- Arch 1 wrist crop work: independent, proceeds on whichever extract is canonical at the time.
- Re-encoding the raw .mp4 clips to address potential decode artefacts: follow-up if heuristic work exhausts without success.
- Full BoT-SORT with appearance embeddings (doubles-badminton-paper style): overkill for singles with known priors.
- Continuity check (pick-to-previous-pick distance threshold): would reject legitimate recoveries after long invisible gaps. Weighted anchor + sanity ceiling + rally-presence check are the intended defenses.

## Next-session kickoff

Starting state: `main` at current commit (post data_access merge). Plan doc + `mmpose_bounds_filtering_research.md` synced.

Phase 0 to open with:

1. Confirm engelbart `main` matches local: `git log --oneline -1`.
2. Write `scripts/find_busted_clips.py` (~30-40 lines). Run on engelbart against the current flat dir. Output: `scratch/architecture_notes/busted_whole_clips_phase1.txt` with 222 stems.
3. Rsync the list to local for review.
4. Spot-check 12 clips by SFTP download + local viewing (non-blocking; handled separately).

Then Phase 1:

5. Write `raw_extract.py`. Run against the 222-stem list, output to `_flat_raw_phase1/`. Wall time ~20 min V100.
6. Write `apply_heuristic.py` + `heuristics/` package with `current` and `sticky_anchor` modules.
7. Run `apply_heuristic.py` twice on the raw dir: once with `current` (byte-identity check against the existing filtered extract for any clip the current heuristic also produced), once with `sticky_anchor` (output to `_flat_h_sticky_anchor/`).
8. Verify `current` produces identical output to committed files. Investigate plumbing if not before trusting `sticky_anchor` results.
9. Build symlink-merged flat dir for Phase-1 mixed re-train.
10. Collate + train V4 on merged data, 5 serials. Compare vs committed V4.

If decision gate passes: Phase 2 full extract. If not: investigate `body_length_fallback` as Plan B, or escalate to arch work.

## Decisions log (appended 2026-04-21)

Short-form rationale so future sessions don't re-argue settled choices without fresh evidence.

### Hyperparameter values

- `prior_weight = 0.75`: from a ref-capture walkthrough, w2=0.5 leaves the ref winning when the real player reappears; w2=0.75 gives the real player a clear reclaim margin. Not a grid-search result. Don't re-grid without a failure signal from actual measurements.
- `ema_alpha = 0.1`: effective half-life ~7 frames. Slower would lag real-play drift; faster would track bad-projection noise. Defensible from the ~25 fps frame rate + observed rally dynamics, not from tuning.
- `sanity_ceiling = 0.5`: must exceed worst-case airborne offset (~0.24 normalized, from Padel geometry at 0.7 m jump, 4.64 error amplification at far court edge). Tighter values reject legitimate airborne picks on smash clips.
- `generous_margin = 0.15`: rally-presence check. Wider than the update gate (0.01), narrower than the sanity ceiling (0.5). Tolerates airborne picks at far edge; catches cutaways with no in-court detections.
- `ankle_conf_cutoff = 0.3`: below this, ankle keypoints are MMPose hallucinations. Any value in [0.2, 0.4] behaves similarly; 0.3 is a reasonable centre.
- `update_gate_eps = 0.01`: matches existing code's in-court test. Used only as the anchor-update gate, not as a pick gate. Widening it would let borderline-off-court detections update the EMA; keep narrow.

### Architectural invariants

- Each heuristic variant lives in its own module under `heuristics/`. No flag-branching inside a single function. Name-based dispatch from `apply_heuristic.py`. Makes variants trivially comparable and removable.
- `current` variant is the trust anchor: if it doesn't byte-identically reproduce committed filtered output for a representative set of clips, nothing else should be trusted. Don't ship other variants without this gate passing.
- `raw_extract.py` is a sibling module, not a flag on `prepare_train_on_shuttleset.py`. Keeps the existing filtered path bit-identical for reproducibility of V3/V4 runs.
- `_raw_kp_scores.npy` is new in this pass. The current pipeline discards MMPose per-joint scores. Saving them is cheap (~7 MB for 222 clips) and enables ankle-confidence-based projection fallback + future training-side confidence weighting. Do not drop it from the output schema.

### Rejected variants and why (do not resurrect without new evidence)

- `torso_center` (hip-based projection): geometrically worse than feet. Standing hips project with ~4.64 error amplification at far court edge vs ~0 for grounded feet. The first-cut plan's "robust to airborne feet" motivation was wrong.
- `monotrack_carry_forward`: repeated identical poses across consecutive frames corrupt the transformer's temporal signal. Don't resurrect without evidence the model tolerates position-freeze.
- `wider_court(eps)` alone: a bigger flat eps doesn't address the far-edge airborne case that needs a non-rectangular tolerance. Always failed on the worst clips.
- `trust_mmpose`: skipping the filter when exactly 2 are detected still relies on projection correctness and ignores the bystander case.
- `focus_aware`: subsumed by sticky_anchor's per-slot tracking. Adds a label-dependency for no marginal gain.
- `jump_adaptive_eps`: subsumed by sticky_anchor's relative-proximity framing. Hip-foot jump detector adds complexity with no gain.
- Continuity check (reject pick > 0.25 normalized from previous valid pick for that slot): rejects legitimate recoveries after long invisible gaps. Weighted anchor + sanity ceiling defend adequately against the case it was meant to catch (ref capturing a slot from an airborne real player).

### Scope guardrails (don't revisit without new evidence)

- Variant list: `current` + `sticky_anchor`, with `body_length_fallback` as Plan B only if `sticky_anchor` is ambiguous on Phase 1 measurements.
- MMPose model: RTMPose-L, unchanged.
- No frame interpolation / carry-forward.
- No training-side modifications (BST model, shuttleset_dataset, bst_train, bst_infer).
- No modifications to `prepare_train_on_shuttleset.py`, `collate_npy`, or the `pipeline/` package.
- Phase 1 scope: the 222 busted clips only. Primary `_flat/` dir is read-only until Phase 2 ships.

