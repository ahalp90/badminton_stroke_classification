# Where the annotator succeeds and fails

This branch ran a set of checks on the finished badminton annotator to work out where it succeeds, where it fails, and what needs fixing next. It covered individual hits, player assignments, whole rallies and the clips selected for review.

The detector improvements and benchmark are recorded in [PR #149](https://github.com/ahalp90/badminton_cv_annotator/pull/149). This investigation breaks those outputs down: which videos work well, which errors occur together, what selection leaves behind, and where the pipeline loses usable play.

The main investigation used saved outputs from 47 ShuttleSet22 videos, keeping the detector fixed. It compared annotation errors with the court and player inputs available at the time. Footage checks tested the labels and looked at successful and failed cases. Replays then tested whether changing a court outline changed the pipeline's decisions. [What was checked, and why](experiment_lineage.md) follows those experiments.

For the same work in pictures, see the [visual overview](visual_overview.md).

**The main finding: fix the court stage before fitting another contact model.**

The footage checks found that video 15's labels point to the wrong parts of the match. Excluding it leaves 46 videos and 3,633 missed labelled contacts. **2,374 of those misses (65.3%)** fall in court-rejected scenes. Usable live play was being blocked before normal contact scoring. Two failures were reproduced: OpenCV wrecks a court outline in video 53, while the shared outline loses a visible player in video 17.

The learned output still provides useful clips for review and fixup. Keep video 53 in the evaluation: its checked labels support fixing the court handling. Drop video 15 and its derived release records rather than trying to repair its labels.

The numbers below describe the saved run. Rerun them after #147 removes video 15 from the release and #148 changes the court stage.

**Contents**  
[How the results vary](#how-the-results-vary)  
[Results at a glance](#results-at-a-glance)  
[Video 15 and the next benchmark](#video-15-and-the-next-benchmark)  
[Why court handling comes first](#why-court-handling-comes-first)  
[What the learned path improved](#what-the-learned-path-improved)  
[What still fails](#what-still-fails)  
[Dead ends](#dead-ends)  
[What remains](#what-remains)  
[Reading guide](#reading-guide)  
[Limits](#limits)

## How the results vary

A high hit-matching score can still leave many rallies with errors. A fully correct rally fits in one clip, with every labelled hit matched once, no extra hits and the right player for each.

![Each video's contact timing-match rate against its fully correct rally rate, across the original 47 videos.](figures/video_variation.png)

This is the original **47-video** view. “Trusted” in older plots means cleaned labels. Video 15's labels turned out to point to the wrong footage, so its score is not a valid measure of detector quality.

The spread matters beyond the two outliers. Video 41 has 59/77 fully correct rallies (76.6%); video 17 has 17/73 (23.3%) despite matching 842/976 contacts. Different weak videos need different explanations: most of video 53's missed hits fall in court-rejected scenes, while video 17 mostly fails later.

The investigation also found:

- **Selection leaves useful output behind.** Of 1,763 fully correct rallies, 616 reach the review queue. The fixed threshold was not retuned here.
- **Errors overlap.** In the historical 124 wrong selected clips, 92 have extra contacts and 74 have misses. Every clip with a wrong matched player also has another error.
- **Starts and finishes are harder.** Even in court-accepted frames outside video 15, miss rates are 9.0% for serves and 11.2% for final contacts, against 2.3% in the middle.
- **Matched hits are usually close.** Outside video 15, 98.0% of timing matches are within five frames. Missing events remain a separate problem.

See the [per-video outcome charts](output_errors.md#video-to-video-variation) for the full spread, or open the [interactive video breakdown](VIDEO_BREAKDOWN.html) locally for contact/player confusion matrices and input conditions. The [visual overview](visual_overview.md) puts these plots together with the footage checks.

## Results at a glance

The table starts with all 47 videos, then removes video 15, then video 53 as well for comparison. The 46-video column is the main result. Removing a video changes what is counted; the saved detector output stays the same.

| Saved learned output, cleaned labels, ±10 frames | All 47: historical | Without video 15 | Without videos 15 and 53: sensitivity |
|---|---:|---:|---:|
| Labelled rallies | 3,422 | **3,327** | 3,251 |
| Labelled contacts | 38,218 | **37,184** | 36,247 |
| Exact whole-rally contact sequence | 1,777 (51.9%) | **1,777 (53.4%)** | 1,770 (54.4%) |
| Fully correct rally, including players | 1,763 (51.5%) | **1,763 (53.0%)** | 1,756 (54.0%) |
| Contact timing match | 33,716 (88.2%) | **33,551 (90.2%)** | 33,356 (92.0%) |
| Contact timing + correct player | 32,667 (85.5%) | **32,586 (87.6%)** | 32,392 (89.4%) |
| Serve timing + correct player | 2,647 (77.4%) | **2,642 (79.4%)** | 2,628 (80.8%) |

Keep video 53 in the main read. Removing it is only a sensitivity check: it removes seven fully correct rallies as well as many failures.

![Fully correct rally rates across the cumulative video exclusions.](figures/rally_correctness.png)

![Contact recovery across the same cumulative video exclusions.](figures/contact_correctness.png)

The fixed selection rule leaves this 46-video review queue:

| Selected clips | Count |
|---|---:|
| Known correct | **616** |
| Known wrong | 114 |
| Labels cannot judge | 17 |
| **Total** | **747** |

Among the 730 judgeable clips, exact-annotation precision is **616 / 730 = 84.4%**. Recall is **616 / 3,327 = 18.5%**, giving **30.4% F1**. Counting the 17 unknown clips as not proven correct gives **616 / 747 = 82.5%**.

The selected clips still need review before use as ground truth.

![Correct, wrong and unjudgeable clips in the saved review queue.](figures/review_queue.png)

Compact tables and definitions: [evaluation_tables.md](evaluation_tables.md).

## Video 15 and the next benchmark

ShuttleSet22 video 15 is not usable for evaluation. Its first labelled serve lands on opening graphics; later labels name a different game or score from the footage. Even its two complete timing matches show another rally on screen.

The decision is to exclude the video and its derived release records. [Issue #147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) tracks that removal and the wider search for similar source failures.

A later direct review of 24 randomly sampled missed labels found:

- 16 where the visible hit and player agree with the label;
- three clear footage disagreements, all in video 15;
- one wrong-timing label in video 12;
- four unclear cases.

All four directly checked video 53 misses support the labelled hit and player. A poor detector score by itself is not a reason to exclude a video.

The release is also larger than this evaluation. The detector work covers 47 ShuttleSet22 videos; [issue #133](https://github.com/ahalp90/badminton_cv_annotator/issues/133) lists **40 original ShuttleSet videos and 58 ShuttleSet22 videos** for release. Original ShuttleSet already has first-stroke timing concerns in [issue #77](https://github.com/ahalp90/badminton_cv_annotator/issues/77).

## Why court handling comes first

Outside video 15, the learned output misses **3,633** labelled contacts:

| State at the labelled frame | Missed contacts |
|---|---:|
| Court rejected the scene | **2,374** |
| Court accepted; at least one player pick missing | 96 |
| Court accepted; both players picked | 1,163 |

![Where the 3,633 misses outside video 15 occur.](figures/misses_by_input_state.png)

Two failures were reproduced:

- **Video 53:** OpenCV replaces a plausible low-confidence corner with a point near the wrong end of the image. The broken outline fails the two-player check, so the scene never reaches normal player tracking or contact scoring.
- **Video 17:** the scene's own neural-net outline is usable, but the later shared outline is too small for that camera view. It moves the far player beyond the picker's allowed distance. Changing only the outline restores the player in both checked failures.

We have not yet run a fixed court pipeline end to end, so we do not know how many contacts or full rallies #148 will recover. See [court_failures.md](court_failures.md).

## What the learned path improved

The ordinary heuristic finds many plausible contacts but almost never gets a whole rally exactly right. Removing video 15 changes only its denominator:

- ordinary heuristic: **4 / 3,327 = 0.12%** fully correct rallies;
- learned output: **1,763 / 3,327 = 53.0%**.

![Fully correct rallies from the ordinary heuristic and learned output.](figures/heuristic_vs_learned.png)

Outside video 15, the two outputs overlap but are not nested:

| Same labelled contact | Contacts |
|---|---:|
| Both outputs find it | 28,351 |
| Learned output only | 5,200 |
| Ordinary heuristic only | 660 |
| Neither | 2,973 |

The learned path removes a lot of error once usable inputs exist. That leaves the shared court stage as a much larger fraction of the remaining misses. See [heuristic_comparison.md](heuristic_comparison.md).

## What still fails

Court handling is the biggest upstream problem, not the only one.

There are still **1,163 missed contacts** outside video 15 where the court is accepted and both players are available. Across all court-accepted contacts, including cases with a missing player pick:

- serves: 253 / 2,804 missed (9.0%);
- middle contacts: 663 / 28,704 missed (2.3%);
- final contacts: 343 / 3,062 missed (11.2%).

The 114 known-wrong selected clips contain 85 extra events and 67 missed labelled events. More than half of the extras come after the final label, but earlier deletion experiments showed that position alone is not enough to decide whether a physical hit is false.

Matched contacts are usually close: outside video 15, 83.6% are within two frames and 98.0% within five. The main residual problem is therefore missing or structurally wrong events, not a broad timing shift.

See [output_errors.md](output_errors.md) and [video_checks.md](video_checks.md).

## Dead ends

The final cheap detector tweaks did not improve the finished system:

- **Independent edge padding:** changed two of 3,982 proposals and repaired none.
- **Correcting chooser targets for post-padding success:** improved development output but changed the 47-video result **1,763 → 1,761**, with 21 repairs and 23 losses. It also broke two selected clips and repaired none there.
- **Label-guided one-edit repairs:** 58 of 119 wrong selected development clips have a complete one-edit alternative, but labels choose those repairs after the fact and every correct clip also has damaging alternatives.
- **Broad endpoint deletion:** no reliable signal separates false tail events from real contacts.
- **Repairing video 15:** abandoned in favour of exclusion.

Details and receipts: [last_followups.md](last_followups.md).

## What remains

1. **Fix #148 and rerun.** Use the reproduced video 53 and video 17 failures as regression cases, then measure contacts, serves, full rallies and review-queue quality across the evaluation.
2. **Apply #147 and check the release for other bad pairings.** Reuse saved frames and labels; stop once each video's keep/exclude decision is clear.
3. **Cover the full release inventory.** Original ShuttleSet still needs direct checks, and eight original videos need final chooser input preparation if a like-for-like detector comparison is still useful.
4. **Then return to the 1,163 good-input misses.** Work out whether the remaining errors come from missing candidates, bad sequence choices, boundaries or labels before fitting another model.

Noise-aware training can wait until there is a small directly checked contact set to judge it against.

Live backlog: [promising_leads.md](promising_leads.md).

## Reading guide

This README gives the overall result. Pick a question for more detail:

| Question | Where to look |
|---|---|
| What was tested, and why? | [The investigation](experiment_lineage.md) |
| How do outcomes differ across videos? | [Video variation and outcome charts](output_errors.md#video-to-video-variation), [interactive breakdown](VIDEO_BREAKDOWN.html) |
| What kinds of errors remain? | [Contacts, rallies and selected clips](output_errors.md) |
| Is the pipeline blocking usable play? | [Court and player checks](court_failures.md) |
| Do the labels agree with the footage? | [Video checks](video_checks.md), including the original-ShuttleSet results |
| What did the learned detector improve? | [Heuristic comparison](heuristic_comparison.md) |
| Which detector ideas were tried and set aside? | [Last follow-ups](last_followups.md) |
| What should happen next? | [Remaining work](promising_leads.md) |
| Where are the exact numbers and definitions? | [Evaluation tables](evaluation_tables.md) and [label accounting](label_accounting.md) |
| Where is the evidence, and how do I rerun a check? | [Saved files and commands](evaluation_reproduction.md) |

## Limits

These are saved outputs on footage that had already been examined, not an untouched test of new matches.

The footage checks are samples. Scoreboard agreement can rule out a gross wrong-rally mismatch but cannot prove exact hit timing or player identity. The 24-contact direct review is stronger for those events, but it was sampled from misses rather than the whole collection.

The court substitutions prove the two mechanisms on the checked cases. Only an end-to-end rerun can show how much #148 improves the final annotator.
