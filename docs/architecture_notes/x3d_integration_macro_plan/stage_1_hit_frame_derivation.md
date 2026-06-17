# Stage 1: Hit-frame metadata derivation

Stage of the X3D-S wrist-crop fusion macro plan (`x3d_integration_macro_plan.md` §Stage 1).

## TLDR

For every clip, work out which frame the racket hits the shuttle. Two methods, run both, see which agrees with hand-checked truth more often:

- **Method A**: read it straight off the source ShuttleSet annotations using the same windowing rule the clip generator used. Cheap, deterministic, inherits annotator drift.
- **Method B'**: detect shuttle horizontal-velocity reversals in TrackNetV3-inpaint output (Huang et al. 2019's combined Peak + Direction methods), then cross-reference each candidate against the labelled player's wrist velocity peak (per Liu et al. 2023's keypoint-fused hit detection). The candidate that lines up with both signals is the hit.

Sidecars land alongside the existing collated tensors as `hit_frame_idx_method_a.npy`, `hit_frame_idx_method_b.npy`, plus a single chosen `hit_frame_idx.npy` (default Method B' if validation passes; otherwise Method A) and an `hit_frame_edge_flag.npy`. dtype int16 (range covers 0-99 plus a -1 sentinel for "no candidate"). A diagnostic CSV captures the per-clip A-vs-B disagreement, candidate counts, wrist-shuttle agreement frames, and edge flags. Validation harness picks 32 stratified val-split clips (12 classes × 2 + smash × 4 + ws × 4, half Top half Bottom) and emits an rsync-prompt-plus-table for hand-checking.

Key constraints that are settled:
- Edge clips (hit < 19 frames from clip start or end) flagged at Stage 1, not Stage 4.
- The ±19-frame X3D-S window comfortably absorbs Method B's ±1-2 frame ceiling.
- TrackNetV3 in our repo carries the tracker only; no hit-detection code to reuse.
- The 4-stage candidate-disambiguation cases (1/2/3/0/4+ shuttle reversals) all have spec'd handling.

Status: plan signed off, no code written.

## Resumption prompt (for a future cleared session)

```
Open docs/architecture_notes/x3d_integration_macro_plan/stage_1_hit_frame_derivation.md
and read end-to-end. Background context (read the TLDR sections only, not full docs):
- docs/architecture_notes/x3d_integration_macro_plan/x3d_integration_macro_plan.md
- docs/architecture_notes/bst_x_overview.md (TLDR + active priorities)
- docs/architecture_notes/augmentation_framework.md "How hit-frame metadata
  would get derived" section (around line 836)
The plan is locked in scope. Implementation has not started. Existing entry point:
src/bst_x/validation_scripts/hit_frame_lookup.py (Method A scaffold).
Deliverables list is in the doc; deliverable count is four scripts, twelve sidecar
npy files, one diagnostic CSV, one validation-sample md, one Stage 1 report md.

Task: implement the deliverables. Wait for Ariel sign-off before each new file.
House rules: AU spelling, no em-dashes, no "delta"/"fade", plain English, terse
manifest/best_model_id voice. Don't pre-commit; Ariel will commit per session.
```

## Goal

Produce a per-clip hit-frame index for every clip in the collated train/val/test trees, derived two ways (Method A: CSV correlation; Method B: shuttle trajectory inversion). Land both as sidecar files alongside the existing collated tensors. Flag edge clips and disagreement clips so Stages 2-4 don't have to re-derive them. Hand-validate the chosen primary against a stratified random sample.

The deliverable is data + diagnostics, not training. Implementation happens in a follow-up sub-task once this plan is signed off.

## What's already settled

- Clip windowing rule is `between_2_hits_with_max_limits` with the 100-frame cap. Stem format is `{vid}_{set}_{rally}_{ball_round}`. Source: `pipeline/clip_generator.py:_compute_clip_bounds`.
- Method A scaffold exists at `src/bst_x/validation_scripts/hit_frame_lookup.py:25`. It returns a `dict[stem -> hit_idx_disk]` derived from `ShuttleSet/set/*.csv` plus `video_metadata.csv`. CPU-only, runs in seconds. Not yet writing sidecars.
- Collation pads everything to seq_len=100 via `make_seq_len_same` at `src/bst_x/preparing_data/shuttleset_dataset.py:43`. Two cases: `videos_len > 100` strides the disk clip; `videos_len <= 100` zero-pads on the right. Striding shifts the hit index; padding does not.
- Shuttle stream: `shuttle.npy` per split is `(n_clips, 100, 2)` xy in court-normalised coordinates. The TrackNetV3-inpaint version is what `wipe_drop` collation pulls in; gaps are the inpaint output's own residual misses, not raw zeros.
- Collated trees live under `npy_wipe_drop/{train,val,test}/` per `bst_x_overview.md` X3D-S anchor section. Sidecars from Stage 1 land in the same dirs.
- Source clips on engelbart at `BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips/{split}/{Top|Bottom}_{stroke}/{stem}.mp4`. Source `clips_master.csv` at `notebooks/clips_master.csv` carries `clip_stem`, `raw_type_en`, `player_side`, `split_v2`, plus `aroundhead` / `backhand` flags.

## Method A: deterministic CSV correlation (already scaffolded)

`hit_frame_lookup.build_hit_frame_lookup` returns the disk-clip hit-frame index. Wrap it to write `hit_frame_idx_method_a_disk.npy` per split aligned to the collated stem order.

For the collated-tensor index, push the disk index through `make_seq_len_same`'s stride logic:

- `videos_len <= 100`: collated index = disk index (right-padding doesn't move existing frames).
- `videos_len > 100`: stride = `videos_len // 100 + int((videos_len % 100) > 50)`; collated index = `min(disk_idx // stride, 99)`. Frames after stride sampling are not the same physical frames as the disk clip's frames at the same index, so this conversion is intrinsic to using the collated tensor for Stage 4 onwards.

Method A is the cheap, deterministic baseline. Inherits whatever drift the ShuttleSet CSV annotators introduced when they marked the contact frame by eye, especially on fast strokes where a 1-2 frame slip is plausible.

## Method B: shuttle trajectory inversion + wrist-velocity cross-reference

### Research backing

Three domain-specific anchors. The first two are shuttle-coordinate-only baselines; the third is the keypoint-fused refinement that becomes our recommended primary.

**Huang et al. (TrackNetV1, AVSS 2019)** introduce the canonical pair of methods. From the Hsu et al. (2024) reimplementation, p. 6 (block-quoted because Stage 1 leans on this):

> Huang et al. found that most of the hit action occurs at the relatively low point of the trajectory of the shuttlecock, specifically at the place where the trajectory begins to rise [...] they proposed two algorithms to identify shots, including the "Peak Identification Method" and the "Direction Identification Method". The Peak Identification Method converts the shuttlecock position from (x, y) image coordinates to (f_i, y), where f_i represents the frame number. After this conversion, the shuttlecock's trajectories will form a wave-like pattern and the hit moments primarily occur at the peak of this wave pattern. The frame of a hit can be identified by detecting those peaks. However, some trajectories do not exhibit obvious changes in y. To address this problem, TrackNet proposed the Direction Identification Method, which checks for sudden changes in the direction of the shuttlecock. If there is a significant change in the direction of the shuttlecock, it is considered a hit that forces the shuttlecock to change its flight direction.

Two complementary signals: y-coordinate extrema (Peak), and dx/dt sign reversal (Direction). Both were originally designed for the noisy raw TrackNet output, so they're robust to detection misses.

**Hsu et al. (Sensors 2024)** report the standalone performance of the trajectory-only method as the SOTA baseline before their action-detection refinement. Table 1, p. 15:

> the trajectory-based method only yielded a precision of 0.588, a recall rate of 0.936, and an F1 score of 0.723 when the IoU threshold was set to 0.5.

Recall 0.936 says shuttle-only catches almost every real hit; precision 0.588 says it produces false positives that need filtering. For our setting that's the right way around: each clip carries a single labelled stroke at known approximate position (the windowing rule centres on it), so we filter candidates by proximity to the clip's expected hit position rather than by an action-detection module.

Their IoU 0.85 baseline drops to F1 0.162, which says the temporal precision of the pure shuttle-only method is loose at the few-frame level. That matches the augmentation_framework note that Method B's accuracy ceiling is bounded by TrackNet's contact-frame resolution at ±1-2 frames. Comfortably inside the ±19 X3D-S window.

The Hsu paper's own contribution (the SRA fusion with YOLOv7 hit detection) is not what we need. We've got the analogue already in Method A (CSV ground-truth). The two methods (shuttle trajectory + CSV annotation) play the same complementary role as their (TrackNet + YOLOv7) pair.

A scan for newer coordinate-only methods turned up nothing better. The two 2022-2023 candidates (MonoTrack CVPRW 2022, SwingNet at arXiv 2306.10293) both fold pose or full RGB frames into the segmentation, so neither is a pure-shuttle improvement. TrackNet's own 7-frame quadratic curve fitting (Hsu et al. Algorithm 1) is a smoothing pass, not a hit-detector.

In-repo TrackNetV3 at `/home/ariel/Documents/COSC594/TrackNetV3/` (`predict.py`, `preprocess.py`, `dataset.py`, `model.py`, `utils/`) carries the tracker only. No hit / peak / direction / reversal logic in source. So the Peak / Direction algorithm has to be written from scratch, but the algorithm is small (well under 100 lines).

**Liu et al. (arXiv 2307.16000, 2023) — keypoint-fused hit detection.** The relevant newer paper. They train a Transformer over player keypoints + court keypoints to predict a per-frame shuttle flying-direction state (Steady / Bottom / Upper), then a heuristic on the predicted state sequence finds hit frames at state transitions. Reported at ±15 frame tolerance: accuracy 97.65%, F1 0.7955 (precision 0.6787, recall 0.9608).

We don't need their full Transformer pipeline (we'd be retraining for an already-localised problem; the labelled stroke's approximate position in the clip is given by Method A). What's load-bearing is the underlying physical principle their model exploits: **the dominant player's wrist velocity peaks at the same frame the shuttle changes direction**. We have both signals already (TrackNetV3-inpainted shuttle + post-sticky-anchor MMPose joints), so the cross-reference can be done as a cheap heuristic on top of the shuttle-only candidates.

This becomes the Method B' primary below, with shuttle-only as the fallback when wrist keypoints are zeroed in the hit window.

### Algorithm (Method B': shuttle + wrist cross-reference)

Per clip, given `shuttle[:videos_len]` of shape `(videos_len, 2)` and `joints[:videos_len]` of shape `(videos_len, 2, 17, 2)` (frames, players, joints, xy):

1. **Pre-smooth shuttle.** Apply a 3-frame moving average to `shuttle[:, 0]` (x) and `shuttle[:, 1]` (y) to suppress residual jitter. 3 frames is the minimum that does any smoothing without blurring the contact frame across more neighbours than necessary.
2. **Shuttle-only candidates from both Huang signals.**
   - Direction: compute `dx = np.diff(x_smooth)`. Sign reversal at frame `f` means `np.sign(dx[f-1]) != np.sign(dx[f])` (with a small epsilon to ignore near-zero dx during shuttle apex). Treat each reversal as a candidate hit at frame `f`.
   - Peak: find local extrema in `y_smooth` over the clip. In court-normalised coordinates the convention to verify is that bigger y means further from the camera; in image coordinates y increases downward so a "low point" of the shuttle in image-space is a *maximum* of y. Pick whichever the existing `shuttle.npy` actually carries (need a quick spot-check at implementation time; the convention propagates from `pipeline/...` extraction code).
   - Merge: a candidate is any frame appearing in either set, deduplicated within a ±2-frame window.
3. **Wrist velocity peaks (cross-reference signal, per Liu et al. 2023).** For the labelled player (resolved from `player_side`), get the joints for slot Top or Bottom: `wrists = joints[:, slot, [9, 10], :]` (left wrist 9, right wrist 10, xy). Compute per-frame velocity magnitude per wrist: `wv = np.linalg.norm(np.diff(wrists, axis=0), axis=-1)` shape `(videos_len-1, 2)`. Pick the dominant wrist as the one with higher peak velocity over the whole clip (Stage 3.B will refine this; lazy version is fine for Stage 1). Find local maxima in the dominant-wrist velocity series.
4. **Score and pick the labelled stroke.** Score each shuttle candidate by `1 / (1 + min_distance_to_wrist_peak)` where `min_distance_to_wrist_peak` is in frames. Picks the candidate with the strongest wrist-co-occurrence. Tie-break by proximity to clip centre (the windowing rule centres on the labelled stroke).
5. **Candidate-count cases** (extended from `augmentation_framework.md:868-877`):
   - 3 shuttle reversals: previous opponent hit, this stroke, next opponent hit. The wrist-peak score should pick the middle one cleanly because the previous and next strokes belong to the *opponent's* wrist, not the labelled player's.
   - 2 reversals: either (previous + this) or (this + next), depending on which side the windowing buffered. Wrist-peak score plus clip-centre proximity resolves it.
   - 1 reversal: that's the labelled stroke; verify wrist-peak agreement is within ±5 frames as a sanity flag.
   - 0 reversals: soft net shot or shuttle off-screen during contact. Fall back to wrist-peak only: pick the dominant-wrist velocity peak closest to clip centre. Flag `method_b_status='wrist_only'`.
   - 4+ reversals: noisy shuttle stream. Wrist-peak score picks among candidates; flag `method_b_status='noisy_multi_candidate'`.
   - Both shuttle and wrist degenerate (very rare): fall back to Method A. Flag `method_b_status='fallback_to_a'`.
6. **Confidence flag.** Carry per-clip `wrist_shuttle_agreement_frames` (the abs frame distance between the picked candidate and the nearest dominant-wrist velocity peak) into the diagnostic CSV. Clips where agreement is > 5 frames are the suspect ones for the random-sample pass to prioritise.
7. **Output the disk-clip hit-frame index.** Convert to collated-tensor index via the same stride logic as Method A.

The shuttle-only fallback (Method B without the wrist cross-reference) is implemented as the same code path with the wrist-score step short-circuited — useful if Stage 2's wrist-loss-rate check finds the dominant-wrist signal too unreliable in the hit window for some class.

### Test cases (unit + integration)

Drawn from `augmentation_framework.md:905-907`, extended with the wrist-cross-reference cases. Each gets at least one hand-checked clip:

- 3-reversal (rally-middle, hard stroke, both adjacent strokes have clear reversals; wrist score should isolate the labelled stroke cleanly).
- 2-reversal-prev (start of rally, only previous stroke + this).
- 2-reversal-next (end of rally, only this + next).
- 1-reversal (very short clip, this stroke only).
- 0-reversal soft net shot (shuttle barely moves; wrist-only fallback path exercised).
- Noisy multi-candidate (TrackNet inpaint had a glitch; wrist score arbitrates).
- Soft shot inside ±19 of the clip centre (verifies the window-recovery slack).
- Wrist-shuttle disagreement > 5 frames (suspect-flag path; either annotation is wrong or one of the two signals is degenerate).
- Dominant wrist zeroed during the hit window (forces shuttle-only fallback).

Hand-truth comes from the random-sample validation pass below.

## Edge-clip flagging at Stage 1

Yes, flag here, not at Stage 4. Stage 4 should do extraction without re-running Stage 1's logic; cost is one extra column in the diagnostic CSV plus one extra bool array in the sidecar.

Edge criterion (single source of truth, applied in collated-tensor index space):

- `edge_start = hit_idx_collated < 19`
- `edge_end = hit_idx_collated > (videos_len_collated - 19)` where `videos_len_collated = min(videos_len, 100)`
- `edge_flag = edge_start | edge_end`

Both Method A and Method B get their own edge flag (they may disagree on which side of the threshold the hit lands). The diagnostic CSV carries both; the model-side npy carries one chosen by the picked-primary method.

Frequency of edge clips comes out of the diagnostic histogram; per the macro plan's open question 4 in Stage 1, that frequency feeds Stage 4's decision on padding policy.

## Sidecar storage layout

Two formats: npy for the model side, CSV for the diagnostic side.

### Model-side npy sidecars (per split, under the collated tree)

- `hit_frame_idx_method_a.npy` — `(n_clips,)` int16, collated-tensor index, Method A.
- `hit_frame_idx_method_b.npy` — `(n_clips,)` int16, collated-tensor index, Method B'; -1 where Method B' falls back to Method A.
- `hit_frame_idx.npy` — `(n_clips,)` int16, the chosen primary (default Method B' if validation passes; otherwise Method A). This is what the X3D-S dataloader will read.
- `hit_frame_edge_flag.npy` — `(n_clips,)` bool, edge clips for the chosen primary.

dtype is int16 rather than int32 (overkill: indices live in 0-99) or uint8 (range fine but no negative sentinel for "fallback-to-A"). int16 buys the -1 sentinel for free at 2 bytes per clip; total sidecar footprint across all three splits is ~120 KB, negligible. Train-time cost is one int per `__getitem__`, irrelevant next to the (39, 3, 112, 112) wrist-crop tensor the X3D-S branch will load alongside.

All four aligned to the same stem order as the existing `pose.npy` / `pos.npy` / `shuttle.npy` / `videos_len.npy` / `labels.npy`. Order assertion in the writer is essential.

### Diagnostic CSV (one file, all splits)

`hit_frame_diagnostic.csv` at `docs/architecture_notes/x3d_integration_macro_plan/stage_1_outputs/`. Columns:

- `stem` — clip stem.
- `split` — train / val / test (from `split_v2`).
- `raw_type_en` — class label.
- `player_side` — Top / Bottom.
- `videos_len` — pre-collation frame count.
- `videos_len_collated` — `min(videos_len, 100)`.
- `disk_to_collated_stride` — 1 if `videos_len <= 100`, else the strided value.
- `method_a_idx_disk` — disk-clip index from Method A.
- `method_a_idx_collated` — collated-tensor index from Method A.
- `method_b_idx_disk` — disk-clip index from Method B; empty if `method_b_status != 'ok'`.
- `method_b_idx_collated` — collated-tensor index from Method B; empty if no candidate.
- `method_b_status` — `ok` / `wrist_only` / `noisy_multi_candidate` / `fallback_to_a`.
- `n_reversals` — shuttle reversal count from Method B' step 2 (for distribution analysis).
- `wrist_shuttle_agreement_frames` — abs frame distance between the picked candidate and the nearest dominant-wrist velocity peak; load-bearing confidence flag.
- `abs_disagreement_frames` — `|method_a_idx_collated - method_b_idx_collated|`; empty if Method B' fell back to A.
- `edge_flag_a`, `edge_flag_b` — bool per method.
- `aroundhead`, `backhand` — passed through from `clips_master.csv` (load-bearing for the dominant-wrist heuristic in Stage 3).
- `sample_validated` — bool, true for clips in the validation sample, false otherwise.

CSV is human-readable, joinable in pandas, cheap to extend with extra method variants if Stage 1 ever revisits.

## Validation harness

Two scripts. The diagnostic runs first across the whole tree; the random-sample pass is the human-verified cross-check on the chosen primary.

### Diagnostic script (CSV-driven, automated)

`src/bst_x/validation_scripts/hit_frame_ab_diagnostic.py`. Reads `hit_frame_diagnostic.csv`. Outputs to `docs/architecture_notes/x3d_integration_macro_plan/stage_1_outputs/`:

- `disagreement_overall.png` — histogram of `abs_disagreement_frames` across all clips with both methods; bucket size 1 frame, x-range 0-30 then a single bin for "30+".
- `disagreement_by_split.png` — same histogram split into 3 panels (train / val / test).
- `disagreement_by_class.png` — 14-panel grid, one per class in the active taxonomy.
- `disagreement_by_class_side_smash_ws.png` — 4 panels: smash×Top, smash×Bottom, ws×Top, ws×Bottom.
- `summary_by_class.csv` — per class: n_clips, mean abs disagreement, median, P90, P99, edge-clip rate, no-candidate rate.
- `summary_overall.md` — short text writeup with the headline numbers, ready to paste into the Stage 1 report.

The same script also dumps the n_reversals distribution and the videos_len distribution as separate panels; both are cheap to add and useful for sanity-checking what the dataset actually looks like before Stage 4 starts cropping it.

### Random-sample validation pass (human-in-the-loop, time-poor)

`src/bst_x/validation_scripts/hit_frame_validation_sample.py`. Stratified pick:

- 12 of the 14 active classes get 2 clips: 1 Top + 1 Bottom. (12 × 2 = 24.)
- Smash and wrist_smash each get 4 clips: 2 Top + 2 Bottom. (2 × 4 = 8.)
- Total 32 clips.

Selection rule per (class, side) bucket: random sample of N from `clips_master[split_v2 == 'val']` (val rather than train so we don't leak any model-relevant info; val is already held out). Seed the RNG so the sample reproduces; record the seed in the script's output header.

Output: a single self-contained file, `validation_sample_{date}.md`, under the `stage_1_outputs/` dir. Contains:

1. **rsync prompt block.** A heredoc-style bash snippet ready to copy-paste; pulls all 32 clips from engelbart to a local `validation_sample/` dir. Format:

   ```bash
   mkdir -p ~/validation_sample/stage_1_2026MMDD
   for stem_path in \
     "val/Top_smash/3_2_4_7.mp4" \
     "val/Bottom_smash/12_1_8_3.mp4" \
     "..." ; do
       rsync -av "engelbart:/scratch/comp320a/ShuttleSet/clips/${stem_path}" \
         ~/validation_sample/stage_1_2026MMDD/
   done
   ```

2. **Per-clip table.** One row per clip, sorted by class then side. Columns:
   - `stem`, `split`, `class`, `side`, `videos_len`
   - `method_a_idx_collated` (frame to inspect first)
   - `method_b_idx_collated` (second frame to inspect)
   - `abs_disagreement` (`|A - B|`)
   - `note` (free-text column for the human pass: "A is right", "B is right", "neither, real hit at frame X", etc.)

3. **Inspection instructions.** Brief: "open each clip in mpv / VLC, scrub to the Method A frame and the Method B frame; mark which one (if either) is the actual contact frame; record observed contact-frame index in the `note` column."

After the human pass, a follow-up script reads the marked-up CSV and produces a per-method accuracy summary against hand-truth: `mean abs error A`, `mean abs error B`, per-class breakdown, decision recommendation. That summary plus the diagnostic histogram is what the Stage 1 report writeup is built around.

## Test gating

Stage 1 lands when:

1. Both methods run cleanly over the full collated tree with no crashes.
2. Diagnostic histogram is generated; per-class summary CSV is written.
3. Validation sample is produced and hand-checked; per-method MAE is reported.
4. Decision is logged: which method becomes `hit_frame_idx.npy` (the primary), based on the validation-sample MAE plus the diagnostic disagreement distribution.
5. Stage 1 report writeup is committed under `docs/architecture_notes/x3d_integration_macro_plan/`, summarising 2-4.

## Deliverables

- `src/bst_x/validation_scripts/hit_frame_method_b.py` (new) — the Method B detector (smoothing + Peak + Direction + candidate disambiguation), pure-numpy, no torch.
- `src/bst_x/validation_scripts/hit_frame_derive.py` (new) — orchestrator: reads collated tree, runs Method A + Method B, writes the four npy sidecars and the diagnostic CSV.
- `src/bst_x/validation_scripts/hit_frame_ab_diagnostic.py` (new) — reads the diagnostic CSV, writes histograms + summary CSV + summary md.
- `src/bst_x/validation_scripts/hit_frame_validation_sample.py` (new) — picks the 32-clip sample, writes the rsync-prompt + per-clip-table .md.
- `tests/test_hit_frame_method_b.py` (new) — unit tests for the seven test cases listed above, against hand-checked ground-truth clips embedded as small npy fixtures.
- Sidecar npy files per split (4 × 3 = 12 files).
- Diagnostic outputs under `docs/architecture_notes/x3d_integration_macro_plan/stage_1_outputs/`.
- `stage_1_report.md` (new, in the same dir) — short writeup of the diagnostic findings, validation-pass hand-truth numbers, and the chosen-primary decision with rationale.

## Open questions (left for implementation)

These are the items the plan deliberately leaves to the implementer rather than pre-deciding here:

1. **Y-coordinate convention in `shuttle.npy`.** Image-down or court-up? Spot-check at implementation time by picking one mid-rally clip with a clean lob and printing the y-trajectory; the sign of the apex tells you which.
2. **Velocity-reversal epsilon.** The threshold below which `dx` is treated as "near zero" for the sign-change test. Likely 1-2% of typical court-x range; tune against the diagnostic histogram once a first pass is run.
3. **Peak vs Direction primacy when they disagree.** Default: take Direction, since horizontal velocity reversal is the more direct physical contact signal; use Peak as a tie-breaker on soft strokes where Direction is degenerate. The implementer can swap this if the diagnostic disagreement histogram shows Peak winning more often than Direction on a per-class basis.
4. **Wrist-velocity smoothing.** Same 3-frame moving average as the shuttle, applied per wrist xy stream before the velocity-magnitude calculation. Or skip if the keypoint stream is already clean enough post-sticky-anchor; quick spot-check once a first pass is run.
5. **Wrist-shuttle agreement threshold for the suspect flag.** Default 5 frames per the soft-shot tolerance discussion; revisit against the diagnostic histogram.
6. **Random-sample seed.** Pick once at implementation, log it in the script header. Same seed gives the same 32 clips on re-run, which matters if the validation pass needs to be redone.
7. **Diagnostic chart bucket boundaries.** 1-frame buckets up to 30, plus a 30+ bucket is the recommended start; tighten to 0.5-frame buckets if the actual disagreement distribution turns out to concentrate inside ±2 frames.

## Cross-references

- Macro plan: `x3d_integration_macro_plan.md` §Stage 1.
- Method A + B framing: `augmentation_framework.md` "How hit-frame metadata would get derived" section (around line 836).
- Method A scaffold: `src/bst_x/validation_scripts/hit_frame_lookup.py`.
- Collation pad/stride logic: `src/bst_x/preparing_data/shuttleset_dataset.py:43` (`make_seq_len_same`).
- Clips master: `notebooks/clips_master.csv`.
- HPC paths: `~/.claude/projects/.../memory/reference_hpc.md`.
- Hsu et al. paper: `~/Documents/COSC594/enhancing_badminton_game_analysis_an_approach_to_shot_refinement.pdf`.
- Huang et al. 2019 (TrackNetV1): cited inside Hsu et al. as ref [58], Huang Y.-C. et al., "TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications", AVSS 2019. Original methodology paragraph block-quoted above.
- Liu et al. 2023 (keypoint-fused hit detection): "Automated Hit-frame Detection for Badminton Match Analysis", arXiv:2307.16000. Provides the lit anchor for cross-referencing shuttle direction reversal with player keypoint motion.
- In-repo TrackNetV3 source: `/home/ariel/Documents/COSC594/TrackNetV3/`. Tracker only, no hit-detection code.
