# Frame recovery accounting

How many frames the two data-quality fixes pulled back: the MMPose
`sticky_anchor` heuristic (pose stream) and the collation wipe-drop
(shuttle stream). Written 2026-05-27; every count below comes from the
2026-04-29 Phase-2 extract (32,203 non-unknown clips).

## TL;DR

Combined, the two fixes recover **91,131 stream-frames** (76,720 pose +
14,411 shuttle). What that is as a percentage depends on what counts as
"dropped":

- **93.1%** of the signal our own pipeline could recover (excludes the
  `ndet=1` floor where MMPose only found one person).
- **85.1%** of everything our pipeline was zeroing, floor included.
- **42.2%** of all missing signal, once you fold in TrackNet's genuine
  shuttle misses. Those 108,965 frames are the dominant remaining loss
  and neither fix touches them.

So of the frames we were needlessly throwing away, we get back ~93%. Of
the total missing signal, ~42%, the rest being shuttle TrackNet never
saw (high arcs off the top of the broadcast frame).

## Unit: stream-frames

Pose and shuttle are separate streams, and one frame can have either or
both zeroed. The countable unit here is a stream-frame: one stream
zeroed on one frame. Pose and shuttle recoveries sum with no
double-count, even though every wipe-drop shuttle recovery lands on a
frame that also had a pose slot fail (the wipe only ever fired on
pose-fail frames). Different stream, different recovery.

## Sources

- Shuttle crosstab + cohort counts:
  `src/bst_x/validation_scripts/zeroed_frames_analysis_outputs/analysis_merged25_bstbaseline_20260429_1906.txt`
  (crosstab lines 128-137, overall shuttle miss line 115).
- Shuttle wipe design + cohort table:
  `scratch/architecture_notes/frame_zeroing.md`.
- Wipe removed in code:
  `src/bst_x/preparing_data/prepare_train_on_shuttleset.py`
  (the prose comment near line 898 sits where `shuttle[failed, :] = 0`
  used to be; dropped at commit 4e478fc, branch shuttle/wipe-drop).
- Pose before/after:
  `scratch/architecture_notes/mmpose_heuristic/phase1_vs_phase2_2026-04-29.md`,
  off Phase-1 report `analysis_unemergev1_v2_20260421_1159.txt` and
  Phase-2 report `analysis_unemergev1_v2_20260429_1905.txt` (same
  outputs dir).
- `ndet=1` floor:
  `src/bst_x/validation_scripts/raw_ndet_stats_outputs/baseline_2026-04-29.md`.
- sticky_anchor heuristic design:
  `scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md`.

Denominator note: the pose numbers sit on 1,722,058 frames, the shuttle
crosstab on 1,719,627. The 2,431-frame gap (0.14%) is per-clip
shuttle-CSV truncation in collation (`min(len(failed), len(shuttle))`),
so the overlap analysis sees slightly fewer frames than the pose-only
count. It doesn't move any percentage here.

## 1. Shuttle: the collation wipe-drop

Collation used to run `if np.any(failed): shuttle[failed, :] = 0`,
zeroing the shuttle coord on every frame flagged `failed[f]=True` (any
pose slot unpicked).

Crosstab on the 32,203-clip extract (1,719,627 frames):

| Cohort | Frames | % |
|---|---:|---:|
| Both pose + shuttle OK | 1,596,251 | 92.83% |
| MMPose-only fail (shuttle real) | 14,411 | 0.84% |
| Shuttle-only fail (TrackNet) | 107,520 | 6.25% |
| Both fail | 1,445 | 0.08% |

The wipe fired on the whole "MMPose failed" set = 14,411 + 1,445 =
15,856 frames. Of those:

- 14,411 had a real TrackNet shuttle the wipe destroyed only because a
  pose slot failed. Recovered.
- 1,445 had no shuttle anyway (TrackNet had also failed). Nothing to
  recover.

So:

- **14,411 / 15,856 = 90.9%** of the frames the wipe touched are
  recovered, which is 100% of the recoverable ones (every frame that
  held real shuttle).
- Against all missing-shuttle frames before the change (14,411 +
  107,520 + 1,445 = 123,376): **14,411 / 123,376 = 11.7%**. After the
  change, 108,965 frames still carry a zero shuttle, which is exactly
  TrackNet's failure column (6.34%). Collation no longer adds any
  zeroing of its own.
- Whole-dataset: 14,411 / 1,719,627 = 0.84% of frames flipped from a
  fake (0,0) to a real shuttle.

## 2. Pose: the sticky_anchor heuristic

The original BST-inherited heuristic zeroed a whole frame if either
player's projected feet fell outside the court rectangle. Airborne
smashes project well past the baseline, so the most informative frames
got zeroed. sticky_anchor replaced it: it picks each slot by
closest-to-anchor geometry instead of rejecting on the court test.

Same 32,203-clip universe, 1,722,058 frames both sides:

| | Frames zeroed | Rate |
|---|---:|---:|
| Before (original BST heuristic) | 92,677 | 5.38% |
| After (sticky_anchor) | 15,957 | 0.93% |
| Recovered | 76,720 | |

- **76,720 / 92,677 = 82.8%** of the pose frames the old heuristic
  zeroed are recovered.
- The raw extract has 9,186 frames (0.53%) at `ndet=1` (one person
  detected), which no two-player heuristic can fill. Excluding that
  floor, recoverable = 83,491 and **76,720 / 83,491 = 91.9%**. The 6,771
  still-zeroed above the floor are occlusion-driven detection gaps and
  irrecoverable broadcast framings (closeups, side-on, cutaways).
- Whole-dataset: 76,720 / 1,722,058 = 4.45% of frames flipped from
  zeroed to real pose.

Where it landed (per-stroke, Phase-1 rate -> Phase-2 rate):

- smash 13.75% -> 0.46% (30x)
- wrist_smash 9.93% -> 0.51% (19x)
- clear 7.37% -> 0.24% (30x)
- return_net 5.96% -> 0.08% (76x)
- drop 6.62% -> 0.26% (25x)
- lob 5.31% -> 0.88% (6x)

The recovery concentrates on the full-court and airborne strokes the old
"feet off court -> zero" rule punished hardest, including the
smash/wrist_smash bottleneck pair. The two laggards (short_service
1.09x, rush 1.05x) are genuinely hard pose, not heuristic-recoverable.

## 3. Combined

| Stream / fix | Dropped (orig pipeline) | Recovered | Still dropped |
|---|---:|---:|---:|
| Pose (sticky_anchor) | 92,677 | 76,720 | 15,957 |
| Shuttle (wipe-drop) | 14,411 | 14,411 | 0 |
| Combined | 107,088 | 91,131 | 15,957 |

Three percentages, depending on the denominator:

1. **Of the recoverable self-inflicted drops: 93.1%.** Drop the 9,186
   `ndet=1` pose floor (unrecoverable by any two-player heuristic):
   recoverable = 107,088 - 9,186 = 97,902; recovered = 91,131.
   91,131 / 97,902 = 93.1%.
2. **Of everything the two steps were zeroing, floor included: 85.1%.**
   91,131 / 107,088.
3. **Of all dropped signal including TrackNet's genuine misses: 42.2%.**
   Total shuttle missing before the wipe-drop was 123,376 (not just the
   14,411 we recovered), so all-dropped = 92,677 pose + 123,376 shuttle
   = 216,053. 91,131 / 216,053 = 42.2%. Still dropped after both fixes =
   15,957 pose + 108,965 shuttle = 124,922.

### Caveats

- **The two fixes aren't independent.** sticky_anchor cut pose-fails
  92,677 -> 15,957, and the wipe only ever fired on pose-fail frames. So
  by recovering ~76k pose frames, sticky_anchor also stopped the wipe
  firing on them. The 14,411 shuttle figure was measured after
  sticky_anchor, so it understates the shuttle actually saved against
  the true original baseline (old heuristic + wipe both live).
  Quantifying the larger figure would need the old heuristic's
  pose x shuttle crosstab, which we didn't capture.
- **42% is the honest ceiling.** Over half the signal still missing is
  TrackNet genuinely not having the shuttle in any pixel: high arcs off
  the top of the broadcast frame, 11-60-frame off-screen excursions (the
  gap-length and off-screen-y analyses in `frame_zeroing.md` back this).
  That's a sensor/framing limit, not something the heuristic or
  collation can recover. The bottleneck classes (wrist_smash, push,
  drive, cross_court_net_shot) sit at sub-1% shuttle-miss anyway, so the
  residual loss falls mostly on head classes that already classify well.
- **These are training-input frame counts, not a model metric.**
  Recovering frames is necessary, not sufficient: the wipe-drop lifted
  macro +0.5 / min ws +1.2 (run_20260503_172922), but the per-frame
  recovery and the F1 movement are separate measurements.
