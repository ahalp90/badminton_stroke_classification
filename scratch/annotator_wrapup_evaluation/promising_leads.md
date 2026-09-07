# Promising leads: what remains to investigate

This is the live backlog. Closed detector ideas are in [last_followups.md](last_followups.md); the full investigation history is in [methods and reproduction](evaluation_reproduction.md#investigation-sequence).

**Contents**  
[Priority map](#priority-map)  
[1. How much does the court fix recover?](#1-how-much-does-the-court-fix-recover)  
[2. Are there other videos with bad labels?](#2-are-there-other-videos-with-bad-labels)  
[3. What is the result on the full release inventory?](#3-what-is-the-result-on-the-full-release-inventory)  
[4. What still fails once the upstream problems are cleaned up?](#4-what-still-fails-once-the-upstream-problems-are-cleaned-up)  
[Later: noise-aware training](#later-noise-aware-training)

## Priority map

| Question | Why now | Next check |
|---|---|---|
| How much does the court fix recover? | 2,374 of 3,633 misses outside video 15 are in court-rejected scenes | Fix #148 and rerun end to end |
| Are there more video-15-style label failures? | The release is larger than the evaluated set | Apply #147 and do a cheap direct sweep |
| What is the full-release result? | Scored so far: 47 ShuttleSet22 and 32 original development videos | Cover the 40 original + 58 ShuttleSet22 release videos after label decisions |
| What remains with good inputs? | 1,163 misses already have accepted court and both players | Diagnose after the court/data rerun |

## 1. How much does the court fix recover?

Two regression cases are ready:

- **video 53, scene 334:** OpenCV replaces a plausible low-confidence corner with a point near the wrong end of the image, then the two-player gate rejects the scene;
- **video 17, scene 0:** a good scene outline is replaced by a smaller shared outline, and the far player drops out of the picker.

[#148](https://github.com/ahalp90/badminton_cv_annotator/issues/148) has the before/after evidence. [court_failures.md](court_failures.md) has the replay details.

The useful question is not whether those two pictures look better after a fix. It is **how much final annotation improves without admitting a pile of bad footage**.

After the regression cases pass, rerun the same evaluation. Compare fully correct rallies, contact P/R/F1, player-aware P/R/F1, serves and the selected review queue. Sample newly accepted scenes from both sides: rescued live play and footage that still should have been rejected.

Keep video 53 in the main evaluation. If #148 fixes the examples but does not improve complete rallies—or damages the review queue enough to cancel the gains—that is a useful negative result.

Then look at the rest of video 17. The two sampled player failures are explained; most of its other misses are not.

## 2. Are there other videos with bad labels?

Video 15 is settled: exclude it and its derived release records. The removal and a lasting exporter exclusion still need implementing.

The wider job in [#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) is to find any other video where repeated visible label failures make the source pairing unreliable.

The completed ShuttleSet22 checks give a starting point:

- 53 windows across 19 videos have game/score checks;
- 24 randomly sampled misses have direct hit/player judgements;
- all three clear footage disagreements in that sample are from video 15;
- one video 12 label is mistimed;
- all four directly checked video 53 misses support the label.

Reuse cached frames, clips and labels. Check a few sections across each suspect video, including places where the detector did well. Record what was checked and make a simple **keep / exclude / inspect more** decision.

A low detector score is not evidence for exclusion; video 53 is the counterexample. Stop inspecting a video once the release decision is clear. This should not become a relabelling project.

## 3. What is the result on the full release inventory?

The main evaluation covers 47 ShuttleSet22 videos. [#133](https://github.com/ahalp90/badminton_cv_annotator/issues/133) lists **40 original ShuttleSet videos and 58 ShuttleSet22 videos** for release.

Original ShuttleSet is not an untouched test set: 32 videos were already development data. It still matters for release coverage and source quality. [#77](https://github.com/ahalp90/badminton_cv_annotator/issues/77) already records first-stroke timing problems there.

The [32-video recount is complete](video_checks.md#original-shuttleset). The other eight original-ShuttleSet videos have earlier detector outputs and saved features, but they still need the final chooser's inputs prepared for a like-for-like comparison.

Do the label sweep first; it does not depend on finishing those eight detector runs. If the full detector comparison is still useful afterwards, add the missing preparation script and run the same final chooser and boundary path.

Report original ShuttleSet as development/release evidence, not as new generalisation evidence.

## 4. What still fails once the upstream problems are cleaned up?

Right now, **1,163** missed labels outside video 15 already have an accepted court and both players available at the labelled frame.

Use that pool for the next contact investigation, after #147/#148 settle the data and upstream pipeline.

Take a small directly checked sample across serves, middle contacts and final contacts. Separate:

- real detector misses;
- wrong or ambiguous labels;
- hits that never enter the candidate pool;
- available candidates chosen badly by the final sequence;
- boundary or scene-transition cases.

Then pick the intervention that matches the dominant residual error. Do not fit another model just because 1,163 is a large number.

## Later: noise-aware training

The HGB models already use normal tree regularisation. They do not use label smoothing, dropout or per-example reliability weights.

That is not yet a reason to add noisy-label machinery. Without a small set of directly verified contacts, a gain would be hard to interpret. Revisit this after the court fix and label sweep leave a stable benchmark and enough checked examples to tell whether the model actually became more robust.
