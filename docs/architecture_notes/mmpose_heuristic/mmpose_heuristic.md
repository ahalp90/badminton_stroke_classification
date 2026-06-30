# MMPose Heuristic — operational reference

How the active sticky_anchor pipeline works, where its dirs live, how to gate it, and the
design + calibration rationale behind its hyperparameters. For the investigation history
(Phase 0/1/2 narrative, decision log, rejected variants, failure-mode triage), see
`historical_mmpose_heuristic_investigation.md` in this dir.

## TL;DR: how sticky_anchor works

MMPose returns a list of person detections per frame (players, chair umpire, line judges,
audience members that happen to be clearly visible). We need to pick two of them as Top and
Bottom. Instead of trying to filter out non-players up front, we pick by **geometry**:

- Each slot has an **anchor** fixed at the middle of its court half (Top's anchor is the
  middle of the top half, Bottom's is the middle of the bottom half). The anchor is 75% that
  fixed point and 25% a running average of recent picks for this slot. The fixed part keeps
  the anchor from wandering off to capture a wrong person; the running part lets it lean
  slightly toward where the player has actually been.
- For each slot we pick the detection whose projected foot position is closest to that slot's
  anchor. Bottom picks first (its detections are bigger and more confident), then Top picks
  from what's left.
- Candidates that sit closer to the OTHER slot's anchor are excluded from this slot's pool,
  so the two slots can't steal each other's player.
- If the closest candidate is too far away, or if both slots' picks land wildly off court,
  the slot (or both slots) zeroes for that frame.
- When two candidates are similarly close to an anchor, we use two tiebreakers: drop anyone
  who looks seated (based on where the knees sit relative to the torso axis) and prefer the
  larger bounding box.

The heuristic runs on the raw MMPose output stored on disk; the expensive MMPose extraction
runs once and we iterate heuristic variants cheaply on top. Output files match the existing
`_pos / _joints / _failed` schema so collation and training code downstream don't change.

**Terminology note**: "foot position" / "projected feet" / "projected ground position" all
refer to the **bbox bottom-centre** ((x1+x2)/2, y2 of the detection's bounding box),
projected through the homography. The historical doc's "Design rationale" section explains
why bbox bottom-centre wins over the legacy COCO ankle-midpoint projection.

## Pipeline + paths

Two stages, decoupled so the expensive MMPose pass runs once and heuristic iteration is fast
(CPU, seconds per clip).

1. **Raw extract** (`preparing_data/raw_extract.py`, GPU): MMPose runs over every mp4, writes
   five `_raw_*.npy` files per clip.
2. **Apply heuristic** (`preparing_data/apply_heuristic.py`, CPU): reads the raw arrays,
   applies a named heuristic, writes `_pos / _joints / _failed` per clip.

Active dirs on `/scratch/comp320a/` (bourbaki + engelbart, byte-identical across nodes):

| Dir | Role | Stems |
| --- | --- | --- |
| `ShuttleSet_keypoints_raw/` | Phase-2 unified raw extract | 32,203 |
| `ShuttleSet_keypoints_clean_sticky_anchor/` | sticky_anchor output; `$BST_X_MMPOSE_NPY_DIR` points here | 32,203 |
| `ShuttleSet_keypoints_raw_unknown/` | raw for the 1,278 `unknown`-class clips | 1,278 |
| `ShuttleSet_keypoints_clean_sticky_anchor_unknown/` | sticky_anchor output for unknown | 1,278 |
| `ShuttleSet_keypoints_raw_provenance/` | per-stem provenance siblings | n/a |

Unknown is held in separate sibling dirs so it can never accidentally enter the training set
via a permissive glob (every active taxonomy uses `--drop-unknown`).

Per-clip raw schema (5 files per stem):

| File | Shape | Contents |
| --- | --- | --- |
| `{stem}_raw_kps.npy` | `(F, N_max, 17, 2)` | All detected people's keypoints per frame, NaN-padded to `N_max` |
| `{stem}_raw_bboxes.npy` | `(F, N_max, 4)` | All detected bounding boxes, NaN-padded |
| `{stem}_raw_scores.npy` | `(F, N_max)` | Per-person detector confidence |
| `{stem}_raw_kp_scores.npy` | `(F, N_max, 17)` | Per-joint MMPose confidence |
| `{stem}_raw_ndet.npy` | `(F,)` int8 | Number of valid detections per frame; resume marker (saved last) |

`N_max = 16` (raised from 8 after the Phase-1 measurement showed 87% of the first 222-clip
extract hit the N=8 cap; at N=16 only 0.79% hit it on the full 1,716 busted subset). Per-clip
raw storage ~320 KB; 32k stems totals under 12 GB.

Per-clip apply-heuristic schema (3 files per stem):

| File | Shape | Contents |
| --- | --- | --- |
| `_pos.npy` | `(F, 2, 2)` | Normalised court positions per slot, ordered (TOP, BOTTOM) |
| `_joints.npy` | `(F, 2, 17, 2)` | Bbox-diagonal-normalised keypoints per slot |
| `_failed.npy` | `(F,)` bool | True where either slot was zeroed this frame |

`apply_heuristic.py` refuses to write unless `--output-dir` is distinct from both `--raw-dir`
and `$BST_X_MMPOSE_NPY_DIR`. Typo guard against destroying the canonical extract.

## Apply heuristic — canonical run

```
PYTHONPATH=src/bst_x python -m preparing_data.apply_heuristic \
    --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw \
    --output-dir /scratch/comp320a/ShuttleSet_keypoints_clean_<variant> \
    --heuristic sticky_anchor \
    --clips-csv notebooks/clips_master.csv
```

Hyperparameters expose as CLI args; defaults in §Hyperparameters below. `apply_heuristic` calls
`pipeline.data_access.load_repo_dotenv()` at module load, so the collision guards fire without
a prior shell export.

## Byte-identity gate (failsafe)

`failsafe_bst_mmpose_zeroing_check_equivalence.py` in `validation_scripts/`. Runs the `current`
heuristic on 50 deterministically-sampled stems from the hit-zone busted list and compares
the result against `--committed-dir`. Run before trusting any new `sticky_anchor` output; a
mismatch means the plumbing is wrong and no heuristic variant should be trusted until it is
fixed.

Sampling: lex-sort the 1,716 stems in `docs/architecture_notes/busted_hit_zone_clips_phase1.txt`,
take every `len // 50`-th. Deterministic, no seeding. Draws from the busted list rather than
`clips_master.csv` because raw extracts only exist for those 1,716 stems.

Comparison tolerances:

- `_failed.npy`: `np.array_equal` (bool must match exactly).
- `_pos.npy` and `_joints.npy`: `np.allclose(rtol=0, atol=1e-5)` (absorbs the float32 to
  float64 projection-chain non-associativity between the two code paths).

On any mismatch: stop and investigate plumbing before trusting `sticky_anchor`. Usual
suspects: keypoint-index ordering, bbox row order when multiple on-court people exist,
`normalize_joints` vs `normalize_position` step order, resolution-scale application.

`--committed-dir` is required. It must point at a `current`-equivalent extract (one produced
by `apply_heuristic --heuristic current` or the legacy `detect_players_2d`). `$BST_X_MMPOSE_NPY_DIR`
points at sticky_anchor, which is a different heuristic and would always fail this gate; the
script refuses to default to it.

The Phase-1 legacy `_flat` (current-equivalent) committed dir was deleted from `/scratch` on
2026-06-30. Generate a fresh current-equivalent reference on demand via the dual-invocation
pattern below.

### Dual-invocation main-vs-branch gating

For a refactor branch where no pre-existing current-extract dir exists: run the failsafe
once on main to populate a scratch reference dir with main's `apply_heuristic --current`
output, then run it again on the branch with that scratch dir as `--committed-dir`. The
branch-side comparison is the real gate.

Note the module path flips between the two invocations: the failsafe script lived at
`preparing_data/failsafe_*` on main and moved to `validation_scripts/failsafe_*` in commit
`7cd4b41`. Two different `python -m` paths below is intentional, not a typo.

```
RAW=/scratch/comp320a/ShuttleSet_keypoints_raw
PHASE1=docs/architecture_notes/busted_hit_zone_clips_phase1.txt
MAIN_CURR=/scratch/comp320a/clean_current_main_50
BRANCH_CURR=/scratch/comp320a/clean_current_branch_50
STICKY=/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor

# Populate main-side reference. The vs-sticky_anchor comparison line is ignored on this
# pass; we only want $MAIN_CURR populated with main's current-extract output.
git checkout main
rm -rf "$MAIN_CURR" "$BRANCH_CURR"
PYTHONPATH=src/bst_x \
    python -m preparing_data.failsafe_bst_mmpose_zeroing_check_equivalence \
        --raw-dir "$RAW" \
        --committed-dir "$STICKY" \
        --busted-stems-file "$PHASE1" \
        --clips-csv notebooks/clips_master.csv \
        --scratch-output-dir "$MAIN_CURR"

# The real gate
git checkout <branch>
PYTHONPATH=src/bst_x \
    python -m validation_scripts.failsafe_bst_mmpose_zeroing_check_equivalence \
        --raw-dir "$RAW" \
        --committed-dir "$MAIN_CURR" \
        --busted-stems-file "$PHASE1" \
        --clips-csv notebooks/clips_master.csv \
        --scratch-output-dir "$BRANCH_CURR"
# expect Compared 50 / Passed 50 / Failed 0; pos + joints max abs diff 0.000e+00
```

## Algorithm specification

Per-video setup (once per clip, using the homography):

- `halfcourt_centre[TOP] = ((bL + bR) / 2, bU + (bD - bU) / 4)` normalised.
- `halfcourt_centre[BOTTOM] = ((bL + bR) / 2, bU + 3 * (bD - bU) / 4)` normalised.
- `bL`, `bR`, `bU`, `bD` are court borders from `pipeline.court_utils.get_court_info`.
- On ShuttleSet the canonical rectangle collapses these to (0.5, 0.25) and (0.5, 0.75). For
  amateur data they derive from whatever canonical rectangle that video's homography
  defines, so the formula is already data-adaptive.
- Initialise `ema[TOP] = halfcourt_centre[TOP]` and `ema[BOTTOM] = halfcourt_centre[BOTTOM]`.

Per-frame algorithm:

**A. Build candidate pool (once per frame):**

1. Filter raw detections to those with `bbox_score > score_filter` (default 0.2).
2. For each surviving detection, project its bbox bottom-centre through the homography to
   normalised court coords. Store as `candidate.court_base_pos`.

**B. Compute both effective anchors (once per frame, before either slot's pick):**

3. For each slot `s` in `(BOTTOM, TOP)`: `effective_anchor[s] = 0.75 * halfcourt_centre[s] + 0.25 * ema[s]`.
4. For each candidate and each slot, compute `D(candidate, s) = euclidean(candidate.court_base_pos, effective_anchor[s])`.

**C. Process each slot, Bottom first then Top.**

For `s` in `(BOTTOM, TOP)` with `other = the other slot`:

5. Pre-filter the candidate pool for this slot:
   1. Drop candidates with `D(candidate, s) > sanity_ceiling` (default 0.6 normalised).
   2. Drop candidates that are closer to the OTHER slot's anchor than to this slot's own
      anchor (`D(candidate, other) < D(candidate, s)`). Each candidate is only eligible for
      whichever slot's anchor it is closer to. Prevents cross-half capture when the other
      slot's player happens to sit geometrically closer to this anchor than our own player
      does. (Voronoi partition; referred to as the closer-to-own-anchor rule.)
   3. If `s == TOP`, also drop whichever candidate BOTTOM already assigned.
6. If no candidates survive: mark slot `s` as zeroed. Go to next slot.
7. Otherwise `winner = argmin D(candidate, s)` among survivors.
8. Tiebreaker: if any other surviving candidate has `|D(candidate, s) - D(winner, s)| < tiebreaker_tol` (default 0.05):
   1. Among the tied set plus the winner, drop candidates where `is_sitting(candidate) == True`.
   2. Among survivors of (i), pick the one with the largest bbox area.
   3. If (i) dropped everyone, revert to the original `argmin D` pick.
9. Mark slot `s` as picked = winner.

**D. Rally-presence check (after both slots processed):**

10. If both slots are picked but neither pick's `court_base_pos` is within
    `[-generous_margin, 1 + generous_margin]` on both axes (default margin 0.15), mark both
    slots as zeroed.

**E. Write outputs and update EMA per slot:**

11. For each slot `s`:
    - If zeroed: write zeros to `_pos[f, s]` and `_joints[f, s, :, :]`. Reset `ema[s] = halfcourt_centre[s]`.
    - If picked: write `_pos[f, s] = winner.court_base_pos`. Write
      `_joints[f, s, :, :] = normalize_joints(winner.keypoints, winner.bbox)` via the
      existing helper. If `winner.court_base_pos` is within `[-update_gate_eps, 1 + update_gate_eps]`
      on both axes (default 0.01), update `ema[s] = 0.1 * winner.court_base_pos + 0.9 * ema[s]`.
      Otherwise EMA stays.
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

Projects the knee-offset-from-hip onto the hip-to-shoulder axis. Asks "are the knees in the
body's down direction (standing / airborne / active) or perpendicular to the body axis
(sitting)?" in image-pixel coordinates. No confidence gates.

## Hyperparameters

All exposed as `apply_heuristic` CLI args; ShuttleSet defaults shown.

| Param | Default | What it governs |
| --- | --- | --- |
| `prior_weight` | 0.75 | `halfcourt_centre` vs EMA weighting in `effective_anchor` |
| `ema_alpha` | 0.1 | EMA update rate (effective half-life ~7 frames) |
| `sanity_ceiling` | 0.6 | pre-filter max anchor distance for a candidate |
| `generous_margin` | 0.15 | rally-presence envelope |
| `score_filter` | 0.2 | candidate-pool cutoff on `bbox_score` |
| `tiebreaker_tol` | 0.05 | distance tolerance invoking the sitting + area tiebreaker |
| `sitting_threshold` | -0.3 | `body_frame_ratio` cutoff |
| `update_gate_eps` | 0.01 | EMA update in-court gate |

Per-video / per-camera tuning of `sanity_ceiling`, `generous_margin`, and `score_filter` is
deferred to the amateur-generalisation work (see historical doc).

## Court-space geometry calibration

The hyperparameter defaults sit on top of a homography audit and visual confirmation on
`3_1_18_3` (frame 32, top player mid-smash); details in this section.

### What the homography is calibrated to

All 44 videos in `ShuttleSet/set/homography.csv` project their annotated 4 corners
(`upleft_x/y` ... `downright_x/y`) to an identical canonical rectangle in court-space:
**300 wide x 660 tall**. UL=(25, 150), UR=(325, 150), DL=(25, 810), DR=(325, 810).
Length/width ratio = 660/300 = **2.2000**.

| Rectangle | Dimensions (m) | L/W ratio | Match? |
| --- | --- | --- | --- |
| Full doubles court (outer taped) | 6.10 x 13.40 | 2.1967 | **Yes (3 d.p.)** |
| Singles court (inner taped) | 5.18 x 13.40 | 2.5869 | No |
| BWF run-off zone (international minimum, 1m sides + 2m ends) | 8.10 x 17.40 | 2.148 | No |

The annotation target is the outer (doubles) taped court. No "further taped line" or run-off
rectangle is involved. Scale: 300 units to 6.10 m so one court-space unit is ~2.03 cm; the
normalised [0, 1] interval spans the full outer doubles rectangle.

### Visual confirmation on clip `3_1_18_3` (video id 3)

Overlay PNG at `src/bst_x/validation_scripts/mmpose_heuristic_investigation/analysis_outputs/homography_overlay_3_1_18_3_f032.png`
(frame 32, top player mid-smash). The cyan rectangle (annotated corners, scaled from
1280x720 homography resolution up to the clip's 1920x1080 resolution) sits exactly on the
outer doubles taped lines. A derived orange pair (doubles-sidelines minus 7.54% inset) lands
precisely on the visible singles sidelines, independently verifying the annotations are on
the outer taped lines.

Implied singles-sideline normalised x coordinates: **x = 0.0754** and **x = 0.9246**
(since (6.10 - 5.18) / 2 / 6.10 = 0.0754). Singles play occupies ~85% of the horizontal
[0, 1] range; the outer ~7.5% on each side is the doubles tramline.

### The legacy `eps = 0.01` buffer is effectively zero

`check_pos_in_court` in `pipeline/court_utils.py` tests `-eps < x,y < 1 + eps` with
`eps = 0.01`. Converted to physical units against the canonical rectangle:

| Axis | Normalised eps | Physical buffer |
| --- | --- | --- |
| Horizontal (beyond doubles sideline) | 0.01 | **6.1 cm** |
| Vertical (beyond baseline) | 0.01 | **13.4 cm** |

### Observed real overflow on the `3_1_18_3` overlays

From visual inspection of frames 0, 25, 28, 30, 32, 35, 49:

| Scenario | Approximate offset past the doubles line |
| --- | --- |
| Neutral stance, feet on baseline | 0 |
| Retreat for smash setup, feet behind baseline | **50-100 cm** |
| Airborne at peak smash (player centre) | **75-150 cm** past baseline |
| Airborne peak smash (projected position, inflated by `H_z * tan(θ)` from Padel geometry) | additional **70-170 cm** beyond body centre offset |
| Hard lunge past doubles sideline | **30-80 cm** |

The legacy `eps = 0.01` buffer is roughly 1/8 to 1/20 of the standing-behind-baseline
offset, and an even smaller fraction of the inflated projected-position offset during
airborne smashes. Any detection where the player is standing behind the baseline (typical
smash setup) is rejected by the legacy filter.

### Buffer size that is actually needed

- **Observed maximum** on `3_1_18_3`: ~150 cm past baseline (airborne peak, before
  projection amplification). The projected position under airborne amplification can go
  further, up to ~300 cm effective displacement at the far edge for a 0.7 m jump.
- **BWF international-competition minimum run-off**: 2 m back / 1 m sides. Any
  legitimately-in-play stance lies inside this envelope.
- **`sticky_anchor`'s `generous_margin = 0.15`**: ~91.5 cm horizontally / ~2.01 m
  vertically. Matches BWF run-off on both axes. Covers every observed offset on `3_1_18_3`
  with headroom.

### Implications for sticky_anchor hyperparameters

- **`generous_margin = 0.15`** is defensible and shouldn't be widened without fresh
  evidence. Matches BWF run-off and covers all observed offsets.
- **`eps = 0.01`** is retained only as the EMA update gate inside `sticky_anchor`
  (step 11), not as a pick-time filter. In that role it correctly prevents EMA pollution
  by clearly-off-court picks.
- **`sanity_ceiling = 0.6`** comfortably exceeds the worst legitimate airborne projection
  offset, so the ceiling is not the binding constraint for well-behaved smashes. Widened
  from the original 0.5 after observing 0.51 anchor-distance on the apex-jump frame of
  `16_1_42_4`.
- The ~7.5% doubles-tramline region either side of the playing area is in-bounds per the
  homography, so picks that land there are accepted; only picks well outside the doubles
  lines (beyond 0.15 either way) trigger the rally-presence check.

## Known limitations of sticky_anchor

- **Same-angle replay in cutaway**: a replay frame at near-identical camera angle would let
  bystander detections pass the in-court update gate, potentially polluting the EMA. Rare in
  broadcast badminton; not bulletproof.
- **Ball kid / court-crosser during play**: if the real player is off-frame or low-detected
  AND an intruder is in-court and passes the confidence-proximity test, the intruder could
  briefly capture a slot. Weighted anchor (0.75 prior) limits damage: real player reclaims
  on reappearance.
- **Amateur footage**: structurally worse on intrusion cases (refs walk around more, crowd
  visible, fewer players confidently detected). Tuned to ShuttleSet pro scope. Amateur
  extension would need hardening on the rally-presence check and possibly an in-court gate
  on the picking stage, not just the update stage.
- **Two players simultaneously airborne**: singles rarity; would currently trigger the
  rally-presence check (neither pick in generous court) and zero both slots. Negligible in
  practice.
- **Bootstrap with long cutaway intro**: if the clip's first 15+ frames are broadcast
  padding, picks during padding use the court prior only. The first real in-court detection
  starts updating EMA; convergence to the player's actual trajectory takes ~5 real-play
  frames. Slight mis-picks in those early frames but no data loss.
- **Continuity check intentionally absent**: a continuity threshold ("reject a pick > X away
  from previous pick") would reject legitimate player re-appearances after long invisible
  gaps. The weighted anchor + sanity ceiling are the intended defence.
- **Detection-layer gaps are unresolved**: where MMPose fails to propose a bbox at all
  (heavy occlusion at the net, as in 19_2_10_7), no heuristic-layer tuning recovers the
  frame. Recovery routes are temporal interpolation (parked) or a swap to an
  occlusion-robust detector.

## References

- `src/bst_x/preparing_data/raw_extract.py` — GPU raw extract entry point.
- `src/bst_x/preparing_data/apply_heuristic.py` — CLI + `run` library entry point.
- `src/bst_x/preparing_data/heuristics/{base,current,sticky_anchor}.py` — heuristic modules.
- `src/bst_x/validation_scripts/failsafe_bst_mmpose_zeroing_check_equivalence.py` — failsafe gate.
- `src/bst_x/pipeline/court_utils.py` — `get_court_info`, `to_court_coordinate`, `normalize_position`.
- `historical_mmpose_heuristic_investigation.md` — full investigation history (Phase 0/1/2,
  status updates 2026-04-25 and 2026-04-29, decision log, rejected variants, failure-mode
  triage, Phase 2 plan, amateur-generalisation notes, parked recovery routes).
- `phase1_vs_phase2_2026-04-29.md` (this dir) — direct Phase-1 vs Phase-2 comparison numbers.
- `mmpose_phase1_extraction_plan.md` (this dir) — Phase-1 raw-extract plan.
- `mmpose_bounds_filtering_research.md` (this dir) — bounds-filtering survey.
