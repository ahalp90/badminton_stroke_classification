# Frame zeroing redesign

Extracted from `hparams_sweep_speculations.md` for clarity. Covers
the asymmetric shuttle-on-pose-fail wipe in the collation step,
the silent TrackNet visibility-flag drop, the per-cohort data
state that reaches the model, the case for the variant 2a mask
channel (2b deferred), and the parked trajectory-extrapolation
future direction. Code trace and real numbers pulled from
`validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`
to ground the discussion.

## What the code is actually doing

Two zeroing operations exist before the model sees data, plus a
silent visibility-flag drop:

**`sticky_anchor.py:283-323`** (per-frame loop in `_run_clip`).
`pos`, `joints`, and `failed` are initialised to zeros. For each
frame, `_pick_one_frame` returns `None` (whole frame failed) or
`(picks, ...)` with `-1` for any unpicked slot. Per slot: if
picked, `pos[f, s] = cbp` and `joints[f, s] = normalize_joints(...)`
write real values; if unpicked, the array stays at the init zero.
`failed[f] = True` whenever any slot is unpicked or the whole
frame failed. So `failed[f]` is a one-bit "any slot zero" flag,
not a "whole frame collapsed" flag.

**`prepare_train_on_shuttleset.py:864-867`** (collation):
`if np.any(failed): shuttle[failed, :] = 0`. This is the only
conditional per-frame zeroing in the entire collation / dataset /
training / model chain. It only touches `shuttle`. Pose is not
gated on shuttle anywhere downstream.

**`prepare_train_on_shuttleset.py:489`** (shuttle CSV read):
`df.set_index("Frame").drop(columns="Visibility")`. TrackNet
emits `Visibility=0` rows with `X=Y=0` when it fails to detect;
the visibility column is dropped, so a missing-shuttle frame
becomes literal `(0, 0)` after `normalize_shuttlecock`. No flag
survives. Indistinguishable from a real shuttle at the top-left
corner.

The model itself uses only the per-clip tail mask
(`bst.py:321-323`, `mask = range_t < (1 + video_len)`). Padding
gating, not per-frame missingness.

**Cohort breakdown: what each frame looks like in the .npy that
reaches the model.** Phase-2, 32,203-clip extract, 1,719,627
frames total, merged_25 / split_bst_baseline analysis:

| Cohort | Frames | % | `joints` / `pos` | `shuttle` | `failed[f]` |
| --- | --- | --- | --- | --- | --- |
| Both pose and shuttle OK | 1,596,251 | 92.83% | both slots real | real | False |
| MMPose-only fail | 14,411 | 0.84% | picked slots real, unpicked slots 0 | **wiped to (0, 0) by collation** | True |
| Shuttle-only fail | 107,520 | 6.25% | both slots real | (0, 0) from TrackNet, no flag | False |
| Both fail | 1,445 | 0.08% | picked slots real, unpicked slots 0 | (0, 0) | True |

Two takeaways drive the redesign:

1. **Asymmetric collation**: a frame where one player is real
   and shuttle is fine still has shuttle wiped because `failed[f]
   = True`. ~14k frames in this extract. The picked player's pose
   flows through fine; only shuttle is harmed.
2. **Shuttle (0, 0) overload**: 6.25% of frames have shuttle =
   (0, 0) from TrackNet visibility=0 with no flag distinguishing
   them from a real top-left detection. The model has to learn
   the "ambiguous (0, 0) means missing" rule from cooccurrence
   patterns alone.

Per-slot pose-missing rates split the 0.84% across Top vs Bottom
to ~0.5% each. Combined "either slot missing" sits at 0.93%
(0.84% single-slot-fail plus most of the 0.08% both-fail).
Both-only is 0.08%, too rare to anchor a per-frame channel of
its own.

## Why is the shuttle missing rate so high?

The 6.34% rate is striking given two facts that should be pushing
it down: (a) the sticky_anchor heuristic is now well-tuned and only
zeros pose on frames where the homography projection genuinely
fails or fewer than two players are in court (0.93% mmpose
post-heuristic), and (b) we're using TrackNetV3 *with* the inpaint
rectification module (the qaz812345 fork, the "accidental"
inpaint that beat the BST authors' attention-only build, per
`run_20260417_191851/best_model_id.txt`). So 6.34% is the rate
*after* per-frame interpolation across short missing-detection
gaps. The raw rate without inpaint would be higher.

Decomposing by hit-zone position (from the same validation analysis):

| Zone | Frames | Shuttle missing | Rate |
|---|---|---|---|
| Within ±10 frames of any hit ("near hit") | 672,637 | 10,584 | 1.57% |
| Other frames ("away from hit") | 1,046,918 | 98,381 | 9.40% |

Most of the shuttle-missing problem is **between-hit frames, not
the hit moment**. The away-from-hit rate is 6x the near-hit rate.
That rules out contact-moment detection failure as the dominant
mechanism; it's a flight-phase tracking failure.

Per-stroke pattern, sorted by miss rate near hit:

| Stroke | Miss rate near hit | Trajectory profile |
|---|---|---|
| long_service | 10.51% | Held → high arc to back court |
| clear | 5.90% | High deep shot, peak above court |
| rush | 2.98% | Fast net attack, possible blur |
| lob | 2.61% | High arc |
| smash | 1.81% | Setup is incoming high lob |
| short_service | 1.14% | Held → low across net |
| drive | 0.85% | Flat, mid-height |
| return_net | 0.54% | Low at net |
| drop | 0.52% | Slow descending |
| net_shot | 0.26% | At-net, low |
| push | 0.23% | Flat near net |
| cross_court_net_shot | 0.11% | At-net, low diagonal |

Strong gradient: stroke types that send the shuttle high or far
back have 5-50x the miss rate of net-bound shots. The cleanest
controlled comparison is **long_service vs short_service** (both
involve a held shuttle into a service hit; only the post-contact
trajectory differs): 10.51% vs 1.14%, an order of magnitude.
Held-shuttle detection isn't the differentiator; trajectory altitude
is.

**Likely causes, ranked.** All of these are inferences from the
per-stroke aggregate stats, not from clip-by-clip inspection.
Verification scripts described in the verified-findings section.

1. **Shuttle exits the broadcast frame on high arcs** (most likely
   primary cause). ShuttleSet videos come from BWF TV broadcasts,
   which frame the visible court but typically cut off well below
   the lighting rig and ceiling. A long_service or clear arc at
   peak altitude is genuinely above the camera's vertical FOV.
   Once the shuttle exits the frame, no detector can recover it,
   including TrackNetV3 with inpaint. The inpaint can interpolate
   *between* valid endpoints, but it can't synthesise frames where
   the shuttle is physically not in the image. Fits the per-stroke
   pattern: long_service / clear / lob / rush at the top, net-bound
   shots at the bottom.
2. **Inpaint window limits.** The rectification module interpolates
   across short missing-detection gaps but has a maximum gap size.
   A long off-screen excursion (15-30 frames of "shuttle not in
   any pixel") exceeds the inpaint window, so those frames stay at
   (0, 0). Consistent with the 9.40% away-from-hit rate: long
   flight phases are exactly when sustained off-screen stretches
   happen.
3. **Motion blur at peak velocity** (minor at most). If blur were
   dominant, fast contact strokes (smash, drive, push) would have
   the highest miss rates. They're at the bottom of the table.
   Blur probably contributes a small constant background, not the
   per-stroke gradient.
4. **Held-shuttle ambiguity** (minor on long_service). A held
   shuttle against a player's body or busy background is harder
   to detect, but the long_service / short_service gap eats this
   as the primary cause: both have a held-shuttle pre-serve frame
   and only one is at 10.51%.
5. **Net-line occlusion** (negligible). If net occlusion mattered,
   net_shot / cross_court_net_shot / push would be elevated.
   They're at the bottom (0.11-0.26%).

**Why this matters for the mask design.** The missing-shuttle
signal is highly *structured*, more so than the 6.34% raw rate
suggests:

- **Spatially structured**: gaps cluster at high altitudes /
  off-court regions, which correlates with stroke type.
- **Temporally structured**: long sustained absences during high
  arcs, not random per-frame dropouts. Gaps are 5-30-frame
  contiguous blocks, not isolated noise.
- **Class-conditional**: a `shuttle_missing` mask channel is partly
  a stroke-class hint (high miss density → likely service / clear
  / lob).

This was originally framed as reinforcing the variant-2 mask
recommendation. **The verified 0c result reinterprets it**: the
high-miss classes are the high-F1 classes, so a `shuttle_missing`
mask channel mostly conditions on stroke-class identity (which the
model already gets right from pose alone). The mask is real signal,
just not pointed at the bottleneck classes. See "Verified findings"
below for the diagnostic update.

## Verified findings (2026-04-30, scripts 0a / 0b / 0c)

The three pre-flight scripts ran on the full 32,203-stem extract
(unknowns dropped via `clips_master.csv`). The off-screen-high
hypothesis is **confirmed**, but the diagnostic implications differ
substantially from the original framing.

**0a (`shuttle_gap_y_distribution.py`): boundary y-coords cluster
hard at the top of the frame.**

Combined pre-gap last-valid + post-gap first-valid y-coords (n=13,902,
y=0 is top, y=1 is bottom):

- **61.6% in [0.0, 0.1)** (top 10% of frame)
- 5.6% in [0.1, 0.2)
- 24.6% combined in [0.3, 0.5) (mid-court secondary mode)
- <2% in [0.6, 1.0]
- median y = 0.017, mean = 0.155

Pre-gap last-valid: 53.0% in top decile (mean 0.191, median 0.044).
Post-gap first-valid: **72.3% in top decile** (mean 0.110, median
0.009). Distribution is bimodal: a dominant off-screen-top mode and a
smaller mid-court mode (probably motion blur / background contrast /
net-line occlusion).

The pre-gap mid-court secondary mode sits at y ~ 0.4-0.5 (player
contact level); post-gap mid-court mode sits at y ~ 0.3-0.4
(slightly higher / more back-court). Reads as the shuttle moving
upward from contact, going off-screen, and re-entering at a
slightly higher / further-back position. Consistent with ballistic
arcs.

**0b (`shuttle_gap_length_distribution.py`): inpaint window is not
exceeded; the problem is mid-length off-screen arcs.**

32,203 clips scanned, 24 clips with no shuttle detections at all
(0.07%, negligible), 7,299 clips with at least one gap, 24,880
clips with no gaps at all (77.3%, very robust).

8,466 total gaps across 108,673 missing frames. Gap-length stats:
median 8 frames, p99 45 frames. Distribution by class:

| Length class | Gaps | % of gaps | Frames | % of frames |
|---|---|---|---|---|
| 1-2 (single-event blip) | 2,477 | 29.26% | 3,320 | 3.06% |
| 3-5 (motion-blur band) | 1,152 | 13.61% | 4,385 | 4.04% |
| 6-10 (brief occlusion) | 998 | 11.79% | 7,738 | 7.12% |
| **11-30 (off-screen-arc band)** | **2,884** | **34.07%** | **57,984** | **53.36%** |
| **31-60 (sustained absence)** | **954** | **11.27%** | **35,181** | **32.37%** |
| 61+ (inpaint window exceeded) | 1 | 0.01% | 65 | 0.06% |

The 11-60 frame band accounts for **85% of all missing-shuttle
frames**. Only 1 gap in the entire 32k-clip set exceeds 60 frames.
Implications:

- Inpaint is succeeding at the short end (1-10 frames = 55% of gaps,
  14% of frames). Doing useful work where it can.
- Inpaint **isn't being exceeded** at the long end. Essentially zero
  gaps past the typical inpaint window size.
- The 11-60 frame range is "shuttle is genuinely not in any pixel of
  these frames", which no detection-based system can recover. Arc
  duration matches typical badminton flight times for high lobs /
  clears.

**0c (`perclass_shuttle_miss_vs_f1.py`): shuttle-miss rate is
*positively* correlated with per-class F1, opposite of the
hypothesis.**

Run 3 nosides manifest joined against the nosides analysis txt,
14 classes joined of 14, F1 metric, no-collapse-sides:

- Pearson r: **+0.516**
- Spearman r: **+0.415**

Sorted (high miss → low miss):

| Class | Miss% | Median F1 |
|---|---|---|
| long_service | 10.51 | 0.987 |
| clear | 5.90 | 0.954 |
| rush | 2.98 | 0.746 |
| lob | 2.61 | 0.780 |
| smash | 2.24 | 0.662 |
| **wrist_smash** | **1.17** | **0.360** |
| short_service | 1.14 | 0.986 |
| drive | 0.84 | 0.604 |
| drop | 0.73 | 0.668 |
| return_net | 0.54 | 0.813 |
| net_shot | 0.26 | 0.906 |
| push | 0.23 | 0.596 |
| passive_drop | 0.17 | 0.644 |
| cross_court_net_shot | 0.11 | 0.683 |

The high-shuttle-miss classes (long_service, clear, lob) are exactly
the **pose-distinctive** strokes the model classifies confidently
from skeleton alone. The bottleneck classes (wrist_smash, push,
drive, cross_court_net_shot) sit at sub-1% miss rates because their
shuttle trajectories are short and stay in frame.

**The shuttle stream IS available where it's most needed**; the
model just isn't using it well enough on the bottleneck classes.

## Per-class clip-level zeroing (verified 2026-05-03 via `perclass_clip_miss_rate.py`)

Re-cut of the same Phase-2 data at the per-clip level rather than
±10-of-hit, to test whether the +-10 framing was hiding zeroing
in the swing setup or aftermath. Per clip:

```
miss_rate     = (visibility == 0).sum() / len(visibility)
central_share = (visibility[15:-15] == 0).sum() / (visibility == 0).sum()
```

Then median / mean / 1 SD / 2 SD per class.

**Whole-clip miss rate per class.** Headline: median = 0.0000
across every class — more than half the clips of every class
have zero missing shuttle. The redesign helps the tail, not the
median.

| Class | n_clips | mean | sd1 | max |
|---|---|---|---|---|
| long_service | 359 | **24.74%** | 0.30 | 97.01% |
| smash | 2,362 | **13.74%** | 0.20 | 94.64% |
| clear | 2,661 | 11.94% | 0.17 | 83.67% |
| lob | 4,879 | 9.10% | 0.17 | 85.71% |
| wrist_smash | 1,559 | **8.49%** | 0.16 | 88.33% |
| drop | 1,979 | 8.11% | 0.15 | 63.33% |
| rush | 471 | 2.89% | 0.09 | 52.63% |
| defensive_return_lob | 278 | 2.48% | 0.07 | 57.97% |
| passive_drop | 1,198 | 2.28% | 0.08 | 57.14% |
| defensive_return_drive | 382 | 2.05% | 0.07 | 75.00% |
| drive | 654 | 1.81% | 0.07 | 64.58% |
| short_service | 1,858 | 1.63% | 0.09 | 100.00% |
| back_court_drive | 435 | 1.54% | 0.05 | 35.14% |
| return_net | 3,374 | 1.11% | 0.05 | 82.61% |
| push | 2,652 | **1.04%** | 0.04 | 51.16% |
| driven_flight | 52 | 0.90% | 0.03 | 18.52% |
| net_shot | 5,824 | 0.77% | 0.04 | 54.24% |
| cross_court_net_shot | 1,226 | **0.50%** | 0.03 | 41.67% |

**Central-share when missing.** For clips that DO have missing
frames, the share of those missing frames falling in the central
window `[15, len - 15)`. Excludes clips with no missing frames
(ratio undefined) and clips shorter than 30 frames (no central
window).

| Class | n_clips | median | mean | sd1 |
|---|---|---|---|---|
| clear | 1,267 | 0.79 | 0.67 | 0.36 |
| lob | 1,684 | 0.74 | 0.61 | 0.40 |
| long_service | 163 | 0.72 | 0.72 | 0.23 |
| smash | 985 | 0.54 | 0.48 | 0.30 |
| drop | 621 | 0.52 | 0.46 | 0.30 |
| wrist_smash | 460 | 0.50 | 0.44 | 0.31 |
| passive_drop | 185 | 0.00 | 0.24 | 0.30 |
| back_court_drive | 66 | 0.00 | 0.15 | 0.31 |
| defensive_return_lob | 57 | 0.00 | 0.13 | 0.32 |
| push | 281 | 0.00 | 0.07 | 0.23 |
| short_service | 141 | 0.00 | 0.04 | 0.11 |
| drive | 53 | 0.00 | 0.02 | 0.10 |
| defensive_return_drive | 47 | 0.00 | 0.02 | 0.14 |
| return_net | 363 | 0.00 | 0.01 | 0.08 |
| net_shot | 599 | 0.00 | 0.01 | 0.07 |
| cross_court_net_shot | 80 | 0.00 | 0.01 | 0.06 |
| rush | 44 | 0.00 | 0.00 | 0.00 |
| driven_flight | 3 | 0.00 | 0.00 | 0.00 |

Three updates to the 0c read:

1. **Bottleneck pair has asymmetric mid-clip zeroing.** Smash's
   whole-clip miss rate (mean 13.74%) is **1.6x wrist_smash's**
   (mean 8.49%). When either has missing frames, ~50% of those
   missing frames sit in the central window (high-arc setup or
   aftermath, outside ±10-of-hit). The other bottleneck classes
   (push 1.04%, drive 1.81%, cross_court_net_shot 0.50%) are
   essentially clean and would not benefit from the redesign.
2. **The asymmetry is *within* a pair-confused pair**, which is
   what makes it relevant to the bottleneck for the first time.
   Two non-exclusive mechanisms could produce it:
   - **Coarse-class prior**: smash is the broader, more
     frequently-cued category; wrist_smash is the finer subtype.
     The model defaults to smash under uncertainty regardless of
     shuttle data quality. Predicts smash-direction errors even
     on clips with clean shuttle on both sides.
   - **Shuttle-data-quality asymmetry**: smash setup-arc clips
     have more mid-clip zeroing, so the shuttle stream is
     noisier on smash clips. If the model is leaning on shuttle
     for the fine distinction, the noisier smash stream might
     let it over-fit to "shuttle is unreliable, fall back to
     coarse pose, classify as smash". Predicts the asymmetry
     concentrates on the missing-shuttle tail of smash clips.

   The two are confounded by definition: the missing-shuttle
   tail isn't randomly distributed, it's where the high-lob arc
   sits, which is also where smash-vs-wrist-smash matters most
   semantically. CDB-F1's per-class alpha already partially
   attacks mechanism 1 (run_20260501_164658: wrist_smash +8.7
   pp, smash -5.5 pp — pair traded, didn't lift together). That
   is consistent with mechanism 1 being dominant: scalar-per-
   class alpha moves the pair-axis decision boundary but can't
   break the pair confusion. Running the redesign disambiguates:
   if it moves both halves of the pair together, mechanism 2 is
   real; if it doesn't move either half regardless of mask
   channel, mechanism 1 is doing the work and the smash-prior is
   structural.
3. **Long_service confirms 0c.** Mean miss rate 24.74% (highest
   in the dataset), F1 ~0.99 already. Zeroing rate doesn't
   predict F1 on the head of the distribution; pose carries
   services on its own.

## Diagnostic conclusion from 0a / 0b / 0c

The three results combine to firm up the diagnosis:

1. **Off-screen-high is real and structured** (0a): the gap pattern
   is a sensor-level / camera-framing limitation, not a model
   limitation, not a heuristic limitation, not a TrackNetV3
   limitation.
2. **Inpaint is not the bottleneck** (0b): the rectification module
   covers what it can; the missing data is genuinely missing pixels.
3. **Shuttle data is available on the bottleneck classes** (0c): the
   model has the inputs it needs to disambiguate wrist_smash / drive
   / push / cross_court_net_shot but doesn't use them well.

Combined with run 2's wrist_smash gate failure pattern (cleaner
data hurt the rare-support class while helping the head of the
distribution), this triangulates onto a **classifier-side problem**,
specifically how the loss budget is being spent across classes:

- Mean macro / accuracy / top-2 hold or improve with cleaner data
  (head of the distribution wins).
- Min F1 collapses on the small-support classes (tail loses).
- This is the textbook signature of label smoothing fighting
  cleaner data on rare classes: smoothing taxes confidence
  uniformly, but the gradient signal that overcomes the smoothing
  is proportional to support count, so 600-sample classes lose
  more than 2,400-sample classes to the same smoothing constant.

The mask-channel arm was demoted on the 0c read ("shuttle is
reliably present on the bottleneck classes"). The 2026-05-03
per-clip re-cut **partially overturns this**: smash and
wrist_smash specifically have a meaningful tail of mid-clip
zeroing (whole-clip mean 13.7% / 8.5%) that the +-10 framing
showed only as ~2% / 1%. The redesign now plausibly bites on
the pair-confused bottleneck pair, not just the head of the F1
distribution. The other bottleneck classes (push, drive,
cross_court_net_shot) remain essentially clean and aren't
helped. **Re-promote from "demoted" to "run after the
capacity-bump confirmations land"**, framed as a disambiguation
between coarse-class-prior and shuttle-data-quality mechanisms
for the smash↔wrist_smash confusion.

The label-smoothing arm becomes the highest-priority loss-side
experiment, not the focal-loss arm. **Promote LS sweep ahead of
focal**, since LS is the cheaper test and the failure-mode profile
matches its predicted effect more directly than focal's. (Already
done, LS=0.15 won; see the runbook in `hparams_sweep_speculations.md`.)

Trajectory extrapolation as an alternative to masking is parked as
a future direction; see the tail subsection below.

## Redesign target: drop the shuttle-on-any-pose-fail wipe

The actual code change is one line: stop the
`shuttle[failed, :] = 0` in `prepare_train_on_shuttleset.py:864-867`.
Two equivalent ways to land it:

- Leave `failed[f]` as the existing "any slot unpicked" telemetry
  flag and drop the wipe line. Shuttle flows through as TrackNet
  emitted it. Fewer touches.
- Tighten `failed[f]` to mean "both slots unpicked" and keep the
  wipe line. Same data state, different flag semantics. Useful
  only if a downstream mask channel wants the stricter "both
  slots unpicked" reading; otherwise the simpler form is enough.

The redesign target either way: **shuttle-specific zeroing fires
only when zero players are picked.** In all other cases (one slot
real, two slots real) shuttle flows through as TrackNet emitted
it. Pose already flows through unchanged in every case.

**Cost.** Re-collation, ~10 minutes per taxonomy combo. No model
code change.

**Risk.** Small. The existing asymmetry has no defensible logic:
pose tells you what a player is doing relative to the court,
shuttle tells you where the bird is in that frame; if one player
collapses, the bird's coord still has meaning, which is what
we're keeping.

**Rejected: zero pose on shuttle-fail.** Extending the wipe to
zero pose on the 6.25% shuttle-only-fail frames would throw away
~107k frames of pose signal. The original BST paper's `BST_0`
ablation (pose-only, no shuttle) performs well above trivial, so
pose-without-shuttle is informative on its own. Wrong direction
unless we also strip the shuttle stream entirely from the model.

## Mask-channel design

Replacing the implicit "(0, 0) means missing" encoding with
explicit mask channels. The design space splits along two axes:
granularity (how many separate channels) and signal density (how
often each fires). Four variants worth sketching:

1. **Per-slot pose missing + shuttle missing (three channels)**:
   `pose_top_missing[t]`, `pose_bottom_missing[t]`,
   `shuttle_missing[t]`. **Dismissed.** Splits each pose channel
   to ~0.5% positive rate. Top vs Bottom is a discrete slot
   assignment, not a learnable signal at this density; the
   gradient from a 99.5%-zero one-bit input is dominated by the
   dominant-class signal.
2. **Pose- and shuttle-missing mask channels**:
   - **2a — Shuttle missing only (one channel, going in)**:
     `shuttle_missing[t]` at 6.34%. Source: TrackNet
     `Visibility=0` flag, currently dropped at
     `prepare_train_on_shuttleset.py:489`; preserve through
     `get_shuttle_result` and save as `shuttle_missing.npy` per
     split alongside `shuttle.npy`. Stride and pad in lockstep
     with shuttle; pad-frames carry `mask=True`. Per-frame
     integration on the shuttle stream post-TCN: `mask_proj`
     (`Linear(1, d_mask=4)`, no activation) → concat with
     post-TCN shuttle features along `dim=-1` → `shuttle_fuse`
     (`Linear(d_model + d_mask, d_model)`). TCN never sees the
     mask; cross-frame integration of mask info happens in
     transformer attention, not in the TCN. `d_mask=4` chosen
     for redundancy against bad init seeds; lower collapses to
     a direct concat, higher is wasteful on a one-bit input.
   - **2b — Add pose_missing on top (deferred)**: extend 2a
     with `pose_missing_either_slot[t]` at 0.93%. The 0.93%
     per-frame rate is borderline for a one-bit input; clip-
     and stroke-correlation rescues it on paper (next
     subsection), but the bottleneck classes are net-bound,
     where pose-fail rates are far lower than the service-side
     classes that drive the 0.93% aggregate. So the channel
     mostly conditions on stroke-class identity that the
     skeleton already disambiguates. Worth revisiting if 2a
     lifts metrics and the next axis is the service-side
     pose-fail distribution.
3. **Single global OR mask (one channel)**: fires on any stream
   failure, ~7% positive. **Dismissed.** Collapses two failure
   modes with very different downstream meanings (shuttle missing
   usually means a high-arc off-screen excursion; pose missing
   usually means a service-heavy clip with detection failure)
   into one signal. Risks the model treating it as a generic
   "interpolate everything" cue that over-discounts both real
   streams when only one is broken.
4. **No mask, fix the encoding (interpolation)**: replace shuttle
   (0, 0) with a parabolic-fit interpolation between last and
   next valid detections, or a NaN marker the model learns to
   skip. Different design axis from masking; parked as a future
   direction (tail subsection).

**Signal-density numbers** for the 1,719,627-frame validation
analysis:

| Channel | Positive rate | Frames | Learnability read |
|---|---|---|---|
| `shuttle_missing` | 6.34% | ~109k | Plenty |
| `pose_missing` (combined Top + Bottom) | 0.93% | ~16k | Borderline; clip + class structure rescues it |
| `pose_top_missing` alone | ~0.5% | ~8k | Too thin per-frame, no learnable Top vs Bottom signal |
| `pose_bottom_missing` alone | ~0.5% | ~8k | Too thin per-frame, no learnable Top vs Bottom signal |
| Global "any masked" (OR) | ~7% | ~123k | Plenty but coarse, conflates failure modes |

Variant 2a lands as the cleanest fix: strictly more informative
than the current "(0, 0) means missing" encoding, one extra
input channel plumbed through to a small post-TCN fusion
(`mask_proj` + `shuttle_fuse`). 2b stays parked at the per-frame
learnability worry; revisit if 2a lifts and the next gain is
rescuing the service-side pose-fail distribution.

**Practical ordering.** Step 1: drop the shuttle-on-pose-fail
wipe (the redesign target above). Re-train and measure. If it
lifts metrics, the model is using the recovered shuttle signal
even without an explicit mask. Step 2: add variant 2a mask
channel (2b channel deferred). Re-train and measure. If it
lifts further, the model wanted the explicit signal. Step 3: only if lifts plateau,
revisit the (0, 0)-overload question for the 6.25% TrackNet
visibility=0 frames now that the model has a `shuttle_missing`
mask. Could lead into the parked interpolation variant.

## Per-clip vs per-frame signal-density rescue (rationale for 2b parked-not-dismissed)

The 0.93% per-frame rate of the combined `pose_missing` channel
is borderline for a one-bit input. Two structural rescues keep
2b defensible:

- **Clip-correlated**: ~785 clips have any MMPose zeroing in
  their hit zone, 17 are 100%-zeroed, 65 are >50%-zeroed. The
  transformer integrates across positions, so a 0.93% per-frame
  rate becomes a strong per-clip signal in those 785 clips: the
  model sees long contiguous stretches of mask-on frames in the
  difficult clips rather than scattered isolated bits.
- **Stroke-correlated**: service strokes (long_service 3.91%,
  short_service 3.46%) are 5-50x more affected than the
  net-bound classes (return_net 0.08%). The channel becomes a
  cheap conditioning input: "this is one of those service-side
  clips with a patchy hit zone, lean harder on the parts you do
  have".

The per-frame learnability worry is real but offset; the channel
earns its place via clip + class structure rather than per-frame
frequency alone. Worth revisiting once 2a lifts.

## Future direction: trajectory extrapolation

Parked, not active. The 11-60-frame off-screen-arc gaps (85% of
all missing-shuttle frames per script 0b) cluster at predictable
phases of ballistic motion: between known endpoints, typically
going up and coming down. A parabolic-fit extrapolation between
pre-gap and post-gap valid coords could fill these frames with
physically plausible positions rather than (0, 0).

Two reasons not near-term:

1. The bottleneck classes (wrist_smash, push, drive,
   cross_court_net_shot) sit at sub-1% miss rates anyway, so
   extrapolation would help the high-F1 classes (long_service,
   clear, lob) rather than rescue the bottleneck.
2. The transformer's learned representation given a
   `shuttle_missing` mask channel would probably outperform any
   hand-crafted parabolic interpolation. Letting the model learn
   its own context-conditional fill-in is more expressive than
   imposing a physics prior on top of (0, 0).

Comes back as a real lever only if the loss-side and
capacity-side knobs exhaust without breaking the plateau and the
bottleneck pattern shifts onto the high-arc classes.
