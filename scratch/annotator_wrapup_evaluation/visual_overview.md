# The annotator investigation, in pictures

This branch checked where the finished badminton annotator fails and what needs fixing. It compared saved outputs with labels and court/player inputs, inspected footage, and replayed court decisions. The detector stayed fixed.

The main results cover **46 ShuttleSet22 videos**, after excluding video 15's misaligned labels. Scores use cleaned labels and **±10 frames at 30 fps**. These videos had already been examined during earlier work.

## How much does it vary between videos?

A fully correct rally fits in one clip: every labelled hit matched once, the right players, and no extra hits.

![Contact timing recovery against fully correct rally rate, one point per video.](figures/video_variation.png)

Most videos match a high share of hits, but whole-rally success varies widely. This plot and the next two show the original **47 videos**, including video 15's invalid label comparison. “Trusted” in older plots means cleaned labels.

![Contact outcomes for the first 24 videos, ordered by fully correct rally rate.](figures/video_outcome_breakdown_1.png)

![Contact outcomes for the remaining 23 videos. Video 17 has many player errors; video 53 has many misses.](figures/video_outcome_breakdown_2.png)

The bars count labelled hits. The scores alongside count fully correct rallies. Extra predictions are separate in the [interactive video breakdown](VIDEO_BREAKDOWN.html), which also shows player confusion and input conditions. Open it locally to explore each video.

## Does every rally get a complete clip?

![Best available clip for each of the 3,422 labelled rallies across the original 47 videos.](figures/rally_coverage.png)

Even before selection, some rallies are only partly reached or missed entirely. After excluding video 15, **225 rallies** still have no labelled contact reached by a clip.

## What does selection leave behind?

![The fixed selection keeps 616 correct clips and leaves 1,147 correct clips behind, across all 47 videos.](figures/selection.png)

Selection makes the review queue cleaner, but discards many correct clips too. Excluding video 15 leaves the same 616 correct selected clips: none came from that video. The threshold stayed fixed during this investigation.

## What kinds of errors occur together?

![Missing contacts, extra contacts, wrong players and cut-off rallies within the 124 known-wrong selected clips.](figures/selected_errors.png)

This older breakdown includes **all 47 videos**, including video 15. “Trusted” means cleaned labels, not labels verified against footage.

## Are hits mistimed, or missing?

![Timing offsets for matched contacts across the 46 videos outside video 15.](figures/timing_offsets.png)

**98.0% of matches are within five frames.** The 3,633 missing contacts are outside this plot.

![Miss rates for serves, middle and final contacts across all 47 videos, at two timing tolerances.](figures/contact_position.png)

Starts and finishes are harder. That pattern remains after removing video 15 and restricting to court-accepted frames: **9.0%** missed serves, **2.3%** middles, **11.2%** finals.

## What inputs were available?

![Court and player availability among missed and matched contacts, across the 46 videos outside video 15.](figures/upstream_context.png)

Most misses fall in court-rejected scenes; almost all matches have both players available. These are input states within each outcome group. Footage and replays test what caused particular failures.

## Do the labels agree with the footage?

![Video 15 labels call for a serve during opening graphics and a later rally while the visible game is still at 0–0.](figures/label_alignment.png)

Video 15's labels refer to the wrong parts of the match. The decision is to exclude it and its derived clips.

![Direct checks of 24 randomly sampled missed contacts: 16 supported, three contradicted, one mistimed and four unclear, split by court acceptance.](figures/contact_sample_results.png)

These 24 cases were sampled from **misses across all 47 videos**. All three clear contradictions are from video 15. This is not a collection-wide label-error rate.

## Can the court failures be reproduced?

![Video 53: OpenCV replaces a plausible neural-net corner with a point near the bottom of the image, breaking the court outline.](figures/video53_nn_to_opencv.png)

![Video 17: a shared outline replaces a better scene outline and causes the player picker to lose the visible far player.](figures/video17_nn_to_shared.png)

Changing only the outline changed the court/player decisions in the checked cases. A full rerun is still needed to measure recovered contacts and rallies.

## What useful output is left for review?

![Selected clips across the three video populations; the main 46-video queue contains 616 correct, 114 wrong and 17 unjudgeable clips.](figures/review_queue.png)

The 46-video queue is **84.4% correct among its 730 judgeable clips**. Another 17 clips remain unjudgeable. These clips still need review before use as ground truth.

## What did the learned detector improve?

![Fully correct rallies: four from the ordinary heuristic and 1,763 from the learned output, across 3,327 labelled rallies.](figures/heuristic_vs_learned.png)

![Contact timing and player recovery across all 47 videos, then without video 15, then without videos 15 and 53.](figures/contact_correctness.png)

The middle column is the main result. The last column tests how much video 53 affects the totals; its labels support keeping it.

## What about original ShuttleSet?

![Saved final outputs on 32 original-ShuttleSet development videos compared with the historical 47-video ShuttleSet22 results.](figures/original_comparison.png)

The 32 original videos were **development data**. This comparison uses the historical 47-video ShuttleSet22 totals. Eight more original videos still need inputs prepared for the final chooser.

## What next?

- **Fix the court stage and rerun.** The two reproduced failures give concrete cases to check.
- **Exclude video 15; keep video 53.** Check the wider release for other bad video/label pairings.
- **Then inspect the remaining contact errors.** Use the new results to decide what the detector needs next.

[Full report](README.md) · [What was checked](experiment_lineage.md) · [Earlier detector experiments](last_followups.md) · [Saved evidence and commands](evaluation_reproduction.md)
