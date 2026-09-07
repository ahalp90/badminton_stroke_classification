# Annotator evaluation lineage: what was actually checked

Canonical history of the wrap-up investigation, ordered by research decision rather than commit. Final detector dead ends are in [last_followups.md](last_followups.md); live work is in [promising_leads.md](promising_leads.md).

**Contents**  
[1. Freeze the saved learned output](#1-freeze-the-saved-learned-output)  
[2. Recount all 47 ShuttleSet22 videos](#2-recount-all-47-shuttleset22-videos)  
[3. Check videos 15 and 53 against the footage](#3-check-videos-15-and-53-against-the-footage)  
[4. Sample upstream failures and controls](#4-sample-upstream-failures-and-controls)  
[5. Replay the court and player logic](#5-replay-the-court-and-player-logic)  
[6. Compare the ordinary heuristic](#6-compare-the-ordinary-heuristic)  
[7. Recount exclusions and inspect weak videos](#7-recount-exclusions-and-inspect-weak-videos)  
[8. Publish the follow-up decisions](#8-publish-the-follow-up-decisions)  
[Where this leaves us](#where-this-leaves-us)

## 1. Freeze the saved learned output

The finished learned path is the local chooser followed by fixed-membership clip padding.

Canonical saved output:

`scratch/contact_det_closing_pass/results/followups/local_boundary_broader_predictions_fixed_membership.json.gz`

Selection:

`scratch/contact_det_closing_pass/results/serve_followups/chosen_acceptance_broader.json.gz`

Independent edge padding and a corrected-target chooser refit were tested later and rejected. Production wiring and automatic exact approval stayed unchanged. See [last_followups.md](last_followups.md).

## 2. Recount all 47 ShuttleSet22 videos

The evaluator rescored the saved output over 47 previously examined ShuttleSet22 videos with cleaned/all-source labels and ±10/±5-frame tolerances. It reproduced the previous headline totals.

Historical ±10 result: **1,763 / 3,422** cleaned rallies fully correct; **33,716 / 38,218** cleaned contacts timing-matched.

That 47-video aggregate is now historical because video 15 is known-bad ground truth.

## 3. Check videos 15 and 53 against the footage

An eight-window pilot targeted the two worst-looking videos: five windows in video 15 and three in video 53.

Video 15 immediately showed a label/footage mismatch. Video 53 showed ordinary badminton and needed another explanation.

Five more video-15 windows were then chosen for strong timing matches. All five still showed the wrong game or score, including both rallies where every cleaned label had a nearby detected hit.

Result: **exclude video 15; keep investigating video 53.**

## 4. Sample upstream failures and controls

Seed `20260906` selected eight missed middle contacts outside video 15:

- four from court-rejected scenes;
- two from accepted scenes with a missing player pick;
- two from accepted scenes with both players.

Each miss was paired with a successful control from the same video, chosen by nearest rally length and then time. Random IDs hid the detector state from the visual reader. Each request showed nine frames at half-second intervals across ±2 seconds.

All 16 centre frames showed the whole court and both players. The four deliberately sampled court-rejected failures showed ongoing exchanges.

This proves false court rejection exists; it does not estimate its prevalence. One successful control also exposed a weakness in the saved scene cuts: the camera visibly changes within the four-second window even though the nearest saved cut is 11.6 seconds away.

## 5. Replay the court and player logic

Three checks isolated geometry rather than retraining anything.

**Court votes:** original two-player vote arrays were reproduced exactly in four rejected scenes and four controls. Replacing only the outline with the existing same-video shared outline made videos 12, 20 and 53 pass the 50% threshold; video 21 rose to 48.9% and still failed.

**Video 17 player replay:** the original sequential tracker reproduced both saved player-validity fields at all 91,970 saved feature rows. In two failures, replacing only the undersized shared outline with the scene's own outline restored the far-player pick while keeping incoming state fixed.

**Video 53 corner replay:** rerunning the neural-net court model reproduced the saved OpenCV fallback within 0.02 pixels. The top-right neural-net corner still failed the confidence threshold.

Result: two court-geometry failure mechanisms are confirmed. No repaired end-to-end pipeline has been run yet.

## 6. Compare the ordinary heuristic

Saved ordinary heuristic outputs were rescored over the same videos without rerunning vision.

At ±10, the heuristic has **4** fully correct cleaned rallies; the learned output has **1,763**. Outside video 15, the learned path finds **5,200** labels the heuristic misses and loses **660** that the heuristic finds.

Result: keep the learned path. Its improvements make the shared upstream court problem much more prominent in the remaining error.

## 7. Recount exclusions and inspect weak videos

Saved outputs were recounted for:

- all 47 videos;
- 46 without video 15;
- 45 without videos 15 and 53.

A later 53-window review covered 19 videos, including 24 random misses, seven weak videos, extra video 53 checks and successful controls.

Removing video 15 gives the useful 46-video read. Removing video 53 raises percentages but also removes valid successful output, so it remains a sensitivity check only.

The game/score review found no second large wrong-rally mismatch outside video 15.

## 8. Publish the follow-up decisions

Branch head [`8a8562e`](https://github.com/ahalp90/badminton_cv_annotator/commit/8a8562e26a9286ad491c3935f3860db66b91b020) published:

- [#147 — exclude ShuttleSet22 video 15 and check both datasets for bad labels](https://github.com/ahalp90/badminton_cv_annotator/issues/147)
- [#148 — fix court corrections that make the outline worse](https://github.com/ahalp90/badminton_cv_annotator/issues/148)

Issue #147 later added direct hit/player judgements for the random missed-contact sample:

- 16 agree with the visible hit and player;
- three clear footage disagreements, all video 15;
- one mistimed label in video 12;
- four unclear.

Four random video 53 misses also agree with the labels.

Those direct judgements supersede the earlier game/score-only status for the same cases. The saved detector metrics did not change.

The same follow-up recounted the final output on 32 original-ShuttleSet development videos. It reproduced 1,209 / 2,691 fully correct rallies at ±10 frames. [Video checks](video_checks.md#original-shuttleset) summarises the result and the eight videos still awaiting final-chooser inputs.

## Where this leaves us

- **Video 15:** exclude.
- **Court geometry:** confirmed upstream failure; fix #148 and rerun.
- **Learned contact path:** keep.
- **Good-input residual:** 1,163 misses with accepted court and both players present.
- **Closed detector ideas:** independent edge padding and corrected-target chooser refit.
- **Open work:** release-wide label sweep, court rerun, then diagnose the remaining good-input misses.
