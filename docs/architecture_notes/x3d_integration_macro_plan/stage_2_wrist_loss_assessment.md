# Stage 2: Wrist-keypoint loss assessment + dominant-wrist preflight

Stage of the X3D-S wrist-crop fusion macro plan (`x3d_integration_macro_plan.md` §Stage 2).

## TLDR

Two preflight questions had to be answered before the wrist-loss-rate measurement proper could start:

1. **Do we know each player's dominant hand?** Yes: ShuttleSet has no internal handedness data, but Wikipedia infoboxes covered all 27 distinct singles players in our roster. Three lefties (Carolina MARIN, Kento MOMOTA, Supanida KATETHONG); the other 24 are right-handed. Data file: `stage_2_outputs/player_handedness.csv`.
2. **Are MMPose's L/R keypoint labels physically correct for our broadcast camera?** Open. Two diagnostics spec'd to find out:
   - **Static-bias diagnostic**: pick a known right-hander (Viktor AXELSEN) and a known left-hander (Kento MOMOTA), check that on their forehand strokes the dominant wrist (idx 10 for righties, idx 9 for lefties) carries the higher peak velocity in the hit window. Per-slot agreement should be > 95% if MMPose's anatomical L/R is correct on both Top and Bottom slots. The at-risk slot is **Bottom** (player back to camera, no face landmarks), not Top as the user initially suspected.
   - **Inter-frame consistency diagnostic**: per-clip detector that flags frames where MMPose flipped L/R within the clip. Detector compares same-label vs swap-label trajectory smoothness across the six paired anatomical keypoints (shoulders, elbows, wrists, hips, knees, ankles); frame counts as a flip if ≥ 4 of 6 vote that swapping would have been smoother. Script written and code-reviewed at `src/bst_x/validation_scripts/keypoint_lr_interframe_diagnostic.py`.

Section 1 (the preflight) is what's drafted in detail below. Section 2 onwards (the actual ±19-frame wrist-keypoint loss-rate measurement and the interpolation-viability call) is left as a placeholder; it'll get drafted when Stage 1 lands and Stage 2 proper kicks off.

Status: preflight scope locked, handedness data landed, inter-frame diagnostic script ready to run on engelbart, static-bias diagnostic not yet implemented.

## Resumption prompt (for a future cleared session)

```
Open docs/architecture_notes/x3d_integration_macro_plan/stage_2_wrist_loss_assessment.md
and read end-to-end. Background context (read the TLDR sections only):
- docs/architecture_notes/x3d_integration_macro_plan/x3d_integration_macro_plan.md
- docs/architecture_notes/x3d_integration_macro_plan/stage_1_hit_frame_derivation.md
- docs/architecture_notes/bst_x_overview.md (TLDR + active priorities)
Memory: ~/.claude/projects/.../memory/MEMORY.md and the entries it points at.

Stage 2 preflight status:
- player_handedness.csv landed at stage_2_outputs/. 27 players, 3 lefties.
- keypoint_lr_interframe_diagnostic.py written, code-reviewed, fixed.
  Needs running on engelbart with BST_X_MMPOSE_NPY_DIR + BST_X_CLIPS_CSV set.
  Single-process, well under a CPU-minute total over 32k clips.
- Static-bias diagnostic spec'd in the doc but not yet coded; needs
  Method A hit-frame index to centre the ±5-frame velocity window, so it
  lands after Stage 1's Method A sidecar is written.

Likely next tasks:
1. Run keypoint_lr_interframe_diagnostic.py on engelbart, post-Stage-1.
2. Implement the static-bias diagnostic.
3. Read the two diagnostics together; decide whether per-slot L/R remap
   is needed for downstream consumers (Stage 3.B has a three-way branch
   pre-spec'd at the foot of section 1).
4. Once stage 1 lands, draft section 2 of this doc: ±19-frame wrist-loss-
   rate measurement + interpolation viability call.

House rules: AU spelling, no em-dashes, no "delta"/"fade", plain English,
terse manifest/best_model_id voice. Don't pre-commit; Ariel will commit
per session. Don't refactor model/pipeline code without explicit confirmation.
```

This doc opens with the dominant-wrist preflight work that has to happen before the loss-rate assessment proper, because picking which wrist to measure assumes (a) we know the player's handedness and (b) we trust the keypoint L/R labels for that player at that court slot. Both of those are non-trivial and worth nailing down first. The rest of Stage 2 (the actual ±19-frame wrist-keypoint loss-rate measurement, interpolation viability call) gets written up later when we get to it.

## Section 1: Player handedness data + Top/Bottom keypoint convention

### Why this matters

The whole X3D-S branch hinges on cropping around the strike-wrist of the labelled player. Picking the wrong wrist puts the racket head outside the crop entirely. The strike-wrist depends on:

1. **Player handedness** (anatomical fact about the player).
2. **Stroke type** (around-the-head shots use the dominant arm on the wrong side of the body; backhand strokes use the dominant arm with a different swing path; forehand is the standard case). `clips_master.csv` carries `aroundhead` and `backhand` per stroke, so the stroke-side adjustment has data already.
3. **Which keypoint slot (idx 9 = left_wrist, idx 10 = right_wrist) actually corresponds to the dominant arm in our pose data.** This is the part the user flagged: are Top players' L/R keypoints physically swapped from reality?

Stage 3.B's dominant-wrist heuristic (peak velocity in the hit window) sidesteps this for the *picking* step; it just picks whichever wrist moves faster. But the loss-rate measurement in this stage, the validation pass for Stage 3.B's heuristic, and any sanity check on the X3D-S extraction all want a ground-truth label for "this is the player's dominant arm" so we can verify the velocity heuristic agrees with reality. That ground truth needs handedness data.

### What ShuttleSet gives us (and doesn't)

ShuttleSet's per-stroke CSVs carry `aroundhead`, `backhand`, `hit_height`, `hit_area`, `hit_x`, `hit_y` (stroke-level), plus per-rally metadata. **No player-level handedness column.** `match.csv` holds tournament info, winner/loser names, and a `downcourt` flag (which player starts at the top court at set 1) but nothing about which hand each player swings with.

So handedness has to be sourced externally. The dataset spans 27 distinct singles players across 40 matches.

### External handedness lookup

The 27 players in our active set, all professional, mostly with BWF World Tour profiles or Wikipedia infoboxes that carry handedness explicitly. The lookup table below is sourced from those (BWF profile or Wikipedia infobox; one or the other, recorded per row).

**Strict policy:** no inference. If a player's handedness can't be confirmed from a citable source, the row is `not_found` and the velocity-magnitude fallback kicks in for that player. Visual inference from a single still image is parked as a separate option (would require running an image-analysis pass on a known mid-play photo of the player); not necessary unless not_found rows turn up that we care about.

Data file: `stage_2_outputs/player_handedness.csv` with columns `player_name`, `handedness` (right / left / ambi / not_found), `source_url`, `source_quote`, `last_checked_date`.

All 27 players found; sources are Wikipedia infoboxes throughout. Three lefties in the roster: **Carolina MARIN, Kento MOMOTA, Supanida KATETHONG**. The other 24 are right-handed. No `not_found` rows.

| player_name | handedness | source_url |
| --- | --- | --- |
| Anders ANTONSEN | right | https://en.wikipedia.org/wiki/Anders_Antonsen |
| An Se Young | right | https://en.wikipedia.org/wiki/An_Se-young |
| Anthony Sinisuka GINTING | right | https://en.wikipedia.org/wiki/Anthony_Sinisuka_Ginting |
| Busanan ONGBAMRUNGPHAN | right | https://en.wikipedia.org/wiki/Busanan_Ongbamrungphan |
| **Carolina MARIN** | **left** | https://en.wikipedia.org/wiki/Carolina_Mar%C3%ADn |
| CHEN Long | right | https://en.wikipedia.org/wiki/Chen_Long_(badminton) |
| CHOU Tien Chen | right | https://en.wikipedia.org/wiki/Chou_Tien-chen |
| Evgeniya KOSETSKAYA | right | https://en.wikipedia.org/wiki/Evgeniya_Kosetskaya |
| Hans-Kristian Solberg VITTINGHUS | right | https://en.wikipedia.org/wiki/Hans-Kristian_Vittinghus |
| Jonatan CHRISTIE | right | https://en.wikipedia.org/wiki/Jonatan_Christie |
| **Kento MOMOTA** | **left** | https://en.wikipedia.org/wiki/Kento_Momota |
| KIDAMBI Srikanth | right | https://en.wikipedia.org/wiki/Kidambi_Srikanth |
| LEE Cheuk Yiu | right | https://en.wikipedia.org/wiki/Lee_Cheuk_Yiu |
| LEE Zii Jia | right | https://en.wikipedia.org/wiki/Lee_Zii_Jia |
| LIEW Daren | right | https://en.wikipedia.org/wiki/Liew_Daren |
| Mia BLICHFELDT | right | https://en.wikipedia.org/wiki/Mia_Blichfeldt |
| Michelle LI | right | https://en.wikipedia.org/wiki/Michelle_Li_(badminton) |
| Neslihan YIGIT | right | https://en.wikipedia.org/wiki/Neslihan_Ar%C4%B1n |
| NG Ka Long Angus | right | https://en.wikipedia.org/wiki/Ng_Ka_Long |
| Pornpawee CHOCHUWONG | right | https://en.wikipedia.org/wiki/Pornpawee_Chochuwong |
| PUSARLA V. Sindhu | right | https://en.wikipedia.org/wiki/PV_Sindhu |
| Rasmus GEMKE | right | https://en.wikipedia.org/wiki/Rasmus_Gemke |
| Ratchanok INTANON | right | https://en.wikipedia.org/wiki/Ratchanok_Intanon |
| Sameer VERMA | right | https://en.wikipedia.org/wiki/Sameer_Verma |
| SHI Yuqi | right | https://en.wikipedia.org/wiki/Shi_Yuqi |
| **Supanida KATETHONG** | **left** | https://en.wikipedia.org/wiki/Supanida_Katethong |
| Viktor AXELSEN | right | https://en.wikipedia.org/wiki/Viktor_Axelsen |

The 27-player roster is fixed (extracted from `match.csv` winner/loser columns), so the table lands once and holds for the life of the dataset. Handedness data file (mirrored from this table for code consumption): `stage_2_outputs/player_handedness.csv`.

Two minor notes:
- Neslihan YIGIT is now listed under her married name **Neslihan Arin** on Wikipedia (article moved); same person, infobox carries "Handedness: Right".
- Supanida KATETHONG's nickname "May Sai" / "เมย์ซ้าย" translates to "left May", a folk reference to her left-handedness; consistent with the infobox.

### COCO-17 keypoint convention recap

RTMPose-L (the model we run via `MMPoseInferencer("human")` in `src/bst_x/stroke_classification/preparing_data/raw_extract.py:256`) outputs the standard COCO 17-keypoint set. Wrists are:

- **Index 9: `left_wrist`** — the player's anatomical left wrist.
- **Index 10: `right_wrist`** — the player's anatomical right wrist.

The L and R refer to the **person's** left and right, not the image's left and right. A right-handed Top player facing the camera has their right wrist on the image-LEFT side; a right-handed Bottom player with their back to the camera has their right wrist on the image-RIGHT side. In both cases the keypoint is correctly labelled `right_wrist` (idx 10) **if** the pose model recovers anatomical L/R correctly.

### Top vs Bottom: which slot is the at-risk slot for L/R confusion?

User's instinct was that Top might be the swapped slot. The actual at-risk slot is **Bottom**, not Top. Two reasons:

1. **Bottom player has their back to the camera.** The pose model can't see the face, so it has to infer anatomical L/R from shoulder geometry, ear position, or torso silhouette alone. This is a well-known failure mode for 2D pose detectors on back-facing subjects.
2. **Top player faces the camera.** Face landmarks (left_eye 1, right_eye 2, left_ear 3, right_ear 4) give the model an unambiguous L/R reference. Easy case.

So if there's any L/R confusion in our keypoint stream, it'll concentrate on the Bottom slot. The protocol below verifies this empirically rather than assuming.

### Does our keypoint data actually swap L/R for back-facing players?

Empirical question, not derivable from the spec. RTMPose-L is trained on COCO whole-body, which includes plenty of back-facing sports imagery, so it *should* handle this correctly in principle. But "should" isn't measured.

Verification protocol (cheap one-pass diagnostic):

1. **Right-handed reference players:** Viktor AXELSEN (Denmark, lots of clips, top-tier broadcast quality) as primary; Anders ANTONSEN as backup. Both right-handed per the table.
2. **Left-handed reference players:** Kento MOMOTA (largest left-handed clip count given the multiple Momota matches in the dataset) as primary; Carolina MARIN as backup. Both left-handed per the table.
3. Pull all clips from `clips_master.csv` where the chosen reference is the labelled hitter (`raw_type_en` ∈ {smash, clear, drop, lob, push, drive, net_shot, return_net} — i.e. forehand strokes; ignore around-the-head and backhand strokes for the diagnostic to keep it clean), filter on `player_side`.
4. For each clip, compute peak velocity magnitude over the hit window (±5 frames around Method A's hit-frame index) for both wrists separately: `v_idx9 = peak |d/dt joints[hit-5:hit+6, slot, 9, :]|` and `v_idx10 = peak |...joints[..., 10, :]|`.
5. Tag the clip `dominant_idx = 9 if v_idx9 > v_idx10 else 10`.
6. Aggregate by (player, player_side):
   - Right-handed player at Top: expect dominant_idx = 10 on most clips. If MMPose got L/R right, this should be > 95%.
   - Right-handed player at Bottom: same expectation. **If this drops to < 50%, MMPose is flipping L/R for the Bottom slot** and we need to either swap the index assignment per slot or stop trusting the L/R labels entirely and use velocity-magnitude only.
   - Left-handed player at Top and Bottom: expect dominant_idx = 9 on most clips, same threshold logic. The Momota / Marin cross-check is what proves the L/R labels are anatomical (not just "the side that swings the racket"); without the lefties we couldn't distinguish a correct anatomical-L/R model from a model that just labels the busy wrist as "right".

Output: `stage_2_outputs/keypoint_lr_diagnostic.md` with the per-slot dominant-wrist agreement rates for the chosen reference players, and a verdict on whether per-slot L/R remapping is needed.

### Inter-frame L/R consistency check (separate from the static-bias diagnostic)

The above diagnostic catches a *static* per-slot bias (MMPose systematically getting L/R wrong on back-facing players). It does **not** catch *transient* per-frame flips, where the model labels L/R correctly on most frames of a clip but swaps the two labels on individual frames or short bursts. RTMPose runs per-frame with no temporal memory, so each frame is its own coin flip on hard-to-disambiguate L/R cases. Worth measuring directly across the whole dataset; CPU-only, fast.

**Detection idea.** A flip shows up as both paired keypoints "teleporting" across the body in one frame. The same-label trajectory becomes choppy; the swapped-label trajectory would be smoother. So for each frame `t` and each anatomical pair `(L, R)`, compare:

- `d_no_swap = ||L(t) - L(t-1)|| + ||R(t) - R(t-1)||` — keep current labels.
- `d_swap = ||L(t) - R(t-1)|| + ||R(t) - L(t-1)||` — swap the labels at frame t.

If `d_swap < d_no_swap` by a margin, frame t is a likely flip relative to t-1. The margin comparison is more robust than a hard threshold because frame-to-frame motion magnitude varies wildly across the dataset (a still rally-prep frame moves a few pixels; a smash contact frame moves hundreds).

**Use multiple pairs simultaneously, not just wrists.** A real anatomical-L/R confusion in MMPose flips the entire keypoint set together, not just one pair. Pool the swap-vs-no-swap comparison over the six paired anatomical keypoints (shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12, knees 13/14, ankles 15/16) and call frame t a flip if the majority (≥ 4 of 6) of valid pairs prefer the swap. Eyes 1/2 and ears 3/4 are excluded: they're tightly clustered and dominated by face-detection noise rather than body-structural signal.

**Spec.**

For every clip in `clips_master[raw_type_en != 'unknown']` (32,203 clips), per slot ∈ {Top, Bottom}:

1. Load `joints.npy` of shape `(F, 2, 17, 2)`. Slice to slot: `js = joints[:, slot_idx, :, :]` shape `(F, 17, 2)`. Slice to the six paired anatomical keypoints: pairs = `[(5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]`.
2. Build a per-frame slot-validity mask: `valid[t] = (js[t] != 0).any(axis=(0, 1))` — frame is valid if any keypoint of this slot is non-zero. MMPose exports complete skeletons even at low confidence (with weird interpolated positions for genuinely-missing joints), so the slot is either entirely zero (no detection picked) or entirely non-zero. Per-keypoint zero-skip *within* a valid slot is dead code; drop it. The cost of MMPose's policy is that low-confidence interpolated keypoints look the same as high-confidence detections in the joints array, so the diagnostic measures L/R consistency against potentially-hallucinated positions. Worst case for our detector: a clip where MMPose hallucinated low-confidence wrist positions could vote for a flip if the hallucinations happen to swap-orient. Probably noise-floor frequency, but a known limitation.
3. For each consecutive valid frame pair `(t-1, t)` and each pair `(L_idx, R_idx)`:
   - Compute `d_no_swap` and `d_swap` per the formulas above.
   - Pair votes for flip if `d_swap < d_no_swap * margin` where `margin = 0.5` (swap has to be twice as smooth, not just marginally smoother). Loose enough to avoid noise-driven false flips; tight enough that real swaps fire.
4. Frame t is flagged `flip_at_t` if ≥ 4 of the 6 pair votes go for swap. All six pairs always contribute (no per-pair zero-skip), so the threshold is straightforward majority.
5. Per-clip stats:
   - `n_valid_frame_pairs` (denominator).
   - `n_flips` (numerator).
   - `flip_rate = n_flips / n_valid_frame_pairs`.
   - `max_flip_run_length` (longest consecutive run of flagged frames; distinguishes one-off transients from sustained mislabelling).
   - `frame_pos_of_first_flip / videos_len` (where in the clip flips concentrate; near-edge bias would suggest motion-blur or boundary-effect causation).

**Aggregation outputs.**

`stage_2_outputs/keypoint_lr_interframe_diagnostic.md` plus accompanying CSV `keypoint_lr_interframe_per_clip.csv`:

- Overall flip-rate distribution per slot (histogram).
- Mean flip rate by slot, by class, by player, by (player × slot).
- Distribution of `max_flip_run_length` per slot — separate transient flips from sustained mislabelling.
- Per-class flip rate ranked: smash and wrist_smash are the fast-stroke candidates where motion blur could push MMPose into per-frame L/R confusion; if their rates are an outlier high, that's a known flag for the X3D-S branch.
- Cross-tab against the static-bias diagnostic from the previous section: clips/players where the static bias was clean but inter-frame flips are common are the cases where Stage 3.B's per-clip dominant-wrist heuristic loses its anchor.

**Cost estimate.** Six pairs × per-frame O(1) arithmetic × ~100 frames × 32,203 clips × 2 slots ≈ a few hundred million simple float ops. With numpy vectorisation per clip, well under a CPU-minute total even on a single core. Embarrassingly parallel across clips if it ever needs to be faster, which it won't.

**Filter rules.**

- Skip clips where `raw_type_en == 'unknown'` (1,278 clips, already excluded from the active pipeline by construction).
- Skip per-frame contributions where the slot is fully zeroed (no person picked).
- No per-pair contribution skip: MMPose exports complete skeletons within any valid slot.

**Decisions the diagnostic feeds.**

- **Low overall flip rate, no slot bias** (expected case if RTMPose holds up): the static-bias diagnostic is the only thing that matters, the per-clip heuristic is fine as-is, no per-frame correction needed.
- **High flip rate concentrated in transient single-frame flips**: cheap fix is a per-clip pass that votes the median-label trajectory and corrects single-frame outliers before any downstream consumer reads the wrist data.
- **High flip rate in sustained runs (≥ 3-frame flips)**: harder problem; a Hungarian-style optimal-assignment pass over the whole clip's wrist trajectory would be the stronger fix, but the cost-benefit needs to be weighed against just shipping velocity-magnitude only and dropping the L/R label entirely.
- **Flip rate concentrated on smash / wrist_smash** specifically: load-bearing finding for the X3D-S branch since those are the bottleneck classes; the wrist-crop extraction in Stage 4 would need a per-clip wrist-trajectory cleanup pass before crop-centring.

### Known failure modes for the inter-frame L/R consistency script

Recorded before writing the script so the implementation can guard against them, and so an independent review pass has something to compare its findings against. A silent failure is one where the script runs to completion and produces a plausible-looking output that's wrong; those are the dangerous ones.

1. **Slot-index swap.** apply_heuristic uses `SLOT_TOP=0, SLOT_BOTTOM=1` (`heuristics/sticky_anchor.py:70-71`). If the script assumes the reverse, every per-slot statistic is mislabelled but the numbers look reasonable.
2. **COCO pair mis-indexing.** Off-by-one in the pairs list silently measures the wrong joints. `(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)` is correct.
3. **Eye/ear pairs leak in.** Indices 1/2 and 3/4 are face landmarks; cluster too tightly and dominated by face-detection noise. Inclusion would inflate the flip rate spuriously.
4. **Court-origin keypoint indistinguishable from failed keypoint.** Keypoints are stored in normalised court coords (`apply_heuristic.py:311`, via `normalize_joints`). The court origin is a real point at `(0, 0)`; the failed sentinel is also `(0, 0)`. In float64 the chance of a legitimately-measured keypoint hitting exactly `(0, 0)` is vanishingly small (`joints = np.zeros(...)` initialiser at `sticky_anchor.py:285` only stays zero on slot-fail), but the script should consult `_failed.npy` rather than re-deriving from zero-equality to be robust.
5. **Per-frame slot validity vs per-pair keypoint validity confused.** Two distinct levels: a frame can have the slot succeed but specific keypoints fail (occlusion, low confidence). The script needs both: skip frames where the slot fully failed, and within a valid frame, skip pair contributions where any of the four involved positions is zero.
6. **Frame-pair across a fail gap.** If frame 10 failed and frame 11 succeeded, naive iteration would compare frame 10's zero positions to frame 11's real positions and flag a giant flip. Pair `(t-1, t)` must require both frames slot-valid, not just frame `t`.
7. **Comparison-direction inversion on the margin check.** `d_swap < margin * d_no_swap` (swap has to be at least `1/margin` times smoother) reads naturally as "swap is twice as smooth" at margin=0.5. Inverting to `d_swap * margin < d_no_swap` measures the opposite condition.
8. **Sums of norms vs sums of squared norms.** Using squared distances (no sqrt) is faster but the sum-of-squares ordering does not strictly track the sum-of-norms ordering for vectors of different magnitudes. Use actual L2 norms to match the spec.
9. **Player-identity drift mistaken for L/R flip.** If sticky_anchor's per-slot anchor loses tracking and re-acquires a different physical person, both keypoint pairs of the slot teleport in lockstep — looks like a perfect "swap" of the entire skeleton. The detector cannot distinguish this from RTMPose's anatomical L/R confusion. Output should flag the ambiguity in the markdown writeup; possibly cross-check against the existing sticky_anchor inspection outputs.
10. **Pad frames consumed as real.** This script reads pre-collation `joints.npy` (full disk-clip length F), not the seq_len=100 collated tensor. If someone points it at the collated tree by accident, the trailing zero-padded frames look like all-fail and the diagnostic silently runs only on the real-frame prefix. The script should accept the per-clip `joints.npy` dir and document the dependency.
11. **Run-length aggregation off-by-one.** Maximum consecutive run of flagged frames should treat empty-flip-list as `max_run_length=0`, not raise; should treat all-flip case correctly. Standard contiguous-run logic, easy to bug.
12. **Unknown-class filter loose match.** `raw_type_en == 'unknown'` is the existing filter. Whitespace or casing variants would slip through; safer to do `.str.strip().str.lower() == 'unknown'`.
13. **Frame-zero special case.** Frame `t=0` has no `t-1`. Loop start at `t=1` is correct; off-by-one would skip frame 1 instead.
14. **Clip stem missing from npy_dir.** Some stems in `clips_master[non-unknown]` may not have a corresponding `_joints.npy` (extraction skipped, file deleted, etc.). Decide between hard-fail and warn-skip; document the choice. Default: hard-fail to surface dataset-integrity issues, with a `--allow-missing` CLI override.
15. **Output write race.** If the script is parallelised across clips later (it shouldn't need to be), per-process appends to the same CSV would corrupt rows. Single-process by design; documented in the script header.
16. **dtype overflow on counts.** `int8` would overflow flip counts above 127. Use `int32` for counts, `float32` for rates.
17. **Statistic aggregation across edge-only-zero clips.** If a player has only 5 valid clips after filtering and 4 of them flipped, that's a 4/5 = 80% rate that looks alarming but is statistical noise. Per-(player × slot) reports should carry n_clips alongside the rate, with min-clips filter for the headline tables.
18. **Hardcoded splits assumption.** The script may iterate over the `train/val/test` tree structure that no longer applies under the post-Phase-2 flat dir. Should iterate per-stem from `clips_master`, not per-split-dir from `BST_X_MMPOSE_NPY_DIR`.
19. **Memory footprint per clip.** Each `joints.npy` is `F × 2 × 17 × 2 × 8 bytes` ≈ 27 KB at F=100. 32k clips × 27 KB ≈ 870 MB if all loaded; obviously stream one-at-a-time, not all-at-once. Easy mistake under list comprehension.
20. **Per-class flip rate confounded by class clip-count.** Smash and ws have ~2.4k and ~1.6k clips respectively; long_service has 359 and rush has 471. A flat per-class-mean misrepresents class-conditional flip frequency without normalising by per-class denominator. Histograms with per-class N annotated avoid this.

### Independent review pass: cross-tab against the failure-modes list

A separate review pass on the script went through the source independently (without seeing the failure-modes list above) and surfaced the following. Two were critical bugs the original list missed; the rest are noted here so future readers see what each pass caught.

**Note (post-review correction):** items 3 and 5 below were further simplified after the producer was confirmed to always export complete skeletons within any valid slot. The per-pair zero-skip and the two-denominator design are dead code in practice; both removed from the script. The genuine remaining risk in this regime is low-confidence interpolated keypoints (weirdly-placed but non-zero), documented in the spec section above.

**Critical bugs the review caught that the failure-mode list missed:**

1. **`_failed.npy` shape is `(F,)`, not `(F, 2)`.** The producer at `heuristics/sticky_anchor.py:283` initialises `failed = np.zeros(num_frames, dtype=bool)`. The original script asserted `(F, 2)` and crashed on the first clip. **Real critical bug, missed by the failure-mode list because the list assumed without reading the producer.**
2. **`failed[f] = True` is OR-of-both-slots, not per-slot validity.** `sticky_anchor.py:294, 320` set the flag whenever *either* slot's pick failed. So `failed` cannot resolve "Top picked, Bottom didn't" frames. Even with the shape fix, using `_failed.npy` as a per-slot mask gives wrong per-slot validity. **The spec already says to derive validity from the joints zero-test (`(js[t] != 0).all(axis=-1).any()`); the script took a shortcut to a file that doesn't carry the right information.** Real critical bug, missed for the same reason as #1.

Both fixed by dropping `_failed.npy` consumption entirely and deriving per-slot validity from `(joints[t, slot] != 0).any()` per the spec.

**Silent failures the review caught that the failure-mode list missed:**

3. **Inflated `n_valid_pairs` denominator.** Original code incremented the denominator for any frame-pair where both frames were slot-valid, even if all six per-pair contributions were skipped due to per-keypoint zero. Pushes flip rate artificially low on heavily-occluded clips. Fixed by reporting two denominators: `n_pairs_slot_valid` (the frame-level count) and `n_pairs_with_votes` (subset where at least one anatomical pair survived per-keypoint filtering and contributed a vote). `flip_rate` now uses `n_pairs_with_votes`.
4. **`_resolve_player_name` always returned `""`.** Partial implementation that intentionally bailed out rather than guess; the per-(player × slot) CSV was therefore silently never written despite being promised in the spec aggregation list. Fixed by removing the per-player path entirely from this script and documenting the follow-up recipe (per-clip CSV + post-hoc join via `pipeline.player_mapping`).
5. **Absolute 4-of-6 threshold doesn't scale to `votes_total`.** Frames with fewer than 4 valid anatomical pairs (heavy occlusion, e.g. only knees and ankles non-zero) cannot ever be flagged regardless of unanimity, silently under-counting flips on bottom-of-frame / occluded clips. Kept the absolute threshold per spec (the rationale is that with < 4 valid pairs the signal is too noisy to trust either way), but the new `n_pairs_with_votes` denominator + per-clip stats make the under-counting visible per (clip, slot) for diagnostic interpretation.

**Findings the review surfaced that turned out to be non-issues:**

- *Across-fail-gap stale-reference comparison*: review suggested that after a long fail-run the `(t-1=valid, t=valid+1)` comparison would compare against a stale spatial reference. Re-reading the loop, the slot-valid guard at the start of each iteration enforces both frames adjacent and valid, so no across-gap comparison ever happens. Non-issue.

**Findings the failure-mode list caught that the review missed (or implicitly verified):**

- Slot-index swap, COCO pair mis-indexing, eye/ear pair leakage: review implicitly verified these by reading the source and citing them correctly, but didn't independently re-check the SLOT_TOP/SLOT_BOTTOM assignment. Cross-checked against `sticky_anchor.py:70-71` directly to confirm.
- Per-class flip rate confounded by class clip-count, dtype overflow on counts, hardcoded splits assumption, output write race: not flagged by the review; original spec implementation already handles all of them (per-clip iteration off `clips_master`, int32/float32 dtypes, single-process by design, per-class CSV with N annotated).

Net result of the cross-tab: 2 critical + 3 silent failures from the review, all incorporated into the script. 1 review finding rejected as a misread. The pre-write failure-mode list was useful as a guard during writing; the independent review was load-bearing for catching the wrong-file-consumed bugs that the list assumed away.

### Implications for the dominant-wrist heuristic (Stage 3.B)

Three cases shake out from the L/R diagnostic:

1. **MMPose L/R is correct on both slots** (most likely case based on RTMPose-L's training data). The dominant wrist for a player can be looked up directly from the handedness table: right-handed → idx 10, left-handed → idx 9. This becomes the ground-truth label for Stage 3.B's heuristic to validate against. The velocity-peak heuristic still ships as the runtime picker (handedness coverage may be incomplete; see below), with the handedness-derived label as the QA reference.
2. **MMPose L/R is wrong on the Bottom slot specifically.** Apply a per-slot index swap at load time: `dominant_idx_actual = 9 if (handedness == 'right' and slot == 'Bottom') else 10` and the inverse for left-handers. The diagnostic numbers from step 5 above tell us whether to ship the swap.
3. **MMPose L/R is unreliable at the per-clip level for Bottom players** (intermittent flips, not a deterministic per-slot inversion). Most pessimistic case. The handedness table becomes useless for slot-aware indexing; we'd default to velocity-magnitude only for the dominant-wrist picker, and the handedness lookup is reduced to a sanity-check on aggregate behaviour rather than a per-clip identifier.

The Stage 3.B plan should branch on the diagnostic result before locking in.

### Handling not-found handedness

For players the lookup couldn't confirm (likely tail of the 27-player list given how comprehensive BWF profiles are; expect 0-3 not_found rows):

- The velocity-magnitude fallback (pick whichever wrist has higher peak velocity in the hit window) still works per-clip. The not-found cases lose only the QA cross-check, not the picking ability.
- These players' clips are still in the training set; the wrist-crop extraction proceeds with the velocity pick.
- Flag not-found clips in the Stage 1 diagnostic CSV with a `handedness_known` bool, so Stage 3.B's validation report can subset to known-handedness clips for the agreement-rate calculation.

### Open questions

1. **What if a player's handedness changes in source data we trust** (e.g. ambi-style switch-hitters)? Empty set in practice for the 27-player ShuttleSet roster, but the schema allows `handedness='ambi'` and the runtime falls back to velocity-magnitude for those clips.
2. **Around-the-head and backhand strokes:** for these, the dominant wrist is on the "wrong" side of the body relative to the swing path. Stage 3.B's heuristic on velocity magnitude inside the hit window should still pick correctly because the dominant wrist still moves fastest at contact, but the diagnostic in step 3 above explicitly excludes these cases to keep the L/R verification clean. They get their own pass in Stage 3.B.
3. **Camera height variation across matches:** broadcasts vary in camera angle. The Bottom-slot back-facing assumption is approximate; very low cameras or off-axis shots may put the Bottom player partially side-on. Out of scope for this preflight; if the L/R diagnostic shows match-to-match variance, that's signal for Stage 3.B to source the dominant wrist per-clip rather than per-player.

---

## Section 2 onwards: wrist-keypoint loss-rate assessment, interpolation viability

To be written when Stage 2 proper kicks off after Stage 1 lands. Macro-plan §Stage 2 carries the open questions list; pulling them in here would duplicate the wrong copy. The handedness + L/R prelim above is the only piece needed before Stage 1's diagnostic + sample pass, because Stage 1's Method B' wrist-velocity cross-reference uses the same dominant-wrist picker.

## Cross-references

- Macro plan Stage 2 entry: `x3d_integration_macro_plan.md` §Stage 2.
- Stage 1 plan (consumes the dominant-wrist picker for Method B'): `stage_1_hit_frame_derivation.md`.
- Pose extraction: `src/bst_x/stroke_classification/preparing_data/raw_extract.py:256` (model = `MMPoseInferencer("human")` = RTMPose-L on COCO-17).
- Per-stroke metadata: `notebooks/clips_master.csv` (`aroundhead`, `backhand`, `player_side`).
- Player roster source: `data/shuttleset/set/match.csv` (winner / loser columns).
- Handedness data file (once landed): `stage_2_outputs/player_handedness.csv`.
- L/R static-bias diagnostic output (once run): `stage_2_outputs/keypoint_lr_diagnostic.md`.
- L/R inter-frame flip diagnostic output (once run): `stage_2_outputs/keypoint_lr_interframe_diagnostic.md` + `keypoint_lr_interframe_per_clip.csv`.
