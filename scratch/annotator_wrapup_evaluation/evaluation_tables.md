# Evaluation numbers and definitions

Use this reference for exact counts, error breakdowns and differences between scoring populations. The [main report](README.md) explains the findings with figures. The main benchmark excludes video 15; historical 47-video counts are labelled below.

**Contents**  
[Population](#population)  
[Definitions](#definitions)  
[Learned output](#learned-output)  
[Selected review queue](#selected-review-queue)  
[Proposal overlap and rally coverage](#proposal-overlap-and-rally-coverage)  
[What selection discards](#what-selection-discards)  
[Errors inside selected clips](#errors-inside-selected-clips)  
[Where misses occur](#where-misses-occur)  
[Rally position](#rally-position)  
[Timing of matched contacts](#timing-of-matched-contacts)  
[Player assignment](#player-assignment)  
[Video-to-video variation](#video-to-video-variation)  
[Weak videos](#weak-videos)  
[Rally length](#rally-length)  
[Label sets](#label-sets)  
[Historical 47-video snapshot](#historical-47-video-snapshot)  
[What all source labels change](#what-all-source-labels-change)  
[Contact precision and recall](#contact-precision-and-recall)  
[Old review of unjudgeable clips](#old-review-of-unjudgeable-clips)  
[Reproduce](#reproduce)

## Population

- **46 ShuttleSet22 videos** after removing video 15.
- **3,327 cleaned rallies / 37,184 contacts.**
- **±10 frames at 30 fps** is the main timing allowance.
- The detector output itself is unchanged.

The 47-video column below is historical. The 45-video column without video 53 is a sensitivity check only.

## Definitions

**Fully correct rally** — one proposed clip contains one complete labelled rally; every labelled contact matches once; there are no extra predicted contacts; every matched contact has the correct near/far player.

**Exact whole-rally contact sequence** — same check without player correctness.

**Contact timing match** — a predicted contact lies within the allowed frame distance under complete-video one-to-one matching.

**Selected clip** — a proposal kept by the existing ranking rule; “selected” does not mean correct.

**Cleaned labels** — the subset used for the main evaluation. Older docs call these “trusted”; video 15 is why that word should not be taken literally.

## Learned output

| Cleaned labels, ±10 frames | All 47: historical | Without video 15 | Without videos 15 and 53: sensitivity |
|---|---:|---:|---:|
| Labelled rallies | 3,422 | **3,327** | 3,251 |
| Labelled contacts | 38,218 | **37,184** | 36,247 |
| Exact whole-rally contact sequence | 1,777 (51.9%) | **1,777 (53.4%)** | 1,770 (54.4%) |
| Fully correct rally, including players | 1,763 (51.5%) | **1,763 (53.0%)** | 1,756 (54.0%) |
| Contact timing match | 33,716 (88.2%) | **33,551 (90.2%)** | 33,356 (92.0%) |
| Contact timing + correct player | 32,667 (85.5%) | **32,586 (87.6%)** | 32,392 (89.4%) |
| Serve timing match | 2,781 (81.3%) | **2,766 (83.1%)** | 2,752 (84.7%) |
| Serve timing + correct player | 2,647 (77.4%) | **2,642 (79.4%)** | 2,628 (80.8%) |
| A clip contains the whole rally interval | 3,003 (87.8%) | **2,989 (89.8%)** | 2,978 (91.6%) |

These contact percentages are label-recovery rates. The [historical snapshot](#historical-47-video-snapshot), [all-source comparison](#what-all-source-labels-change) and [contact precision and recall](#contact-precision-and-recall) below cover the other scoring views.

## Selected review queue

The ranking rule is unchanged. Removing video 15 removes its 37 selected clips.

| Selected clips | All 47: historical | Without video 15 | Without videos 15 and 53: sensitivity |
|---|---:|---:|---:|
| Total | 784 | **747** | 746 |
| Known correct | 616 | **616** | 615 |
| Known wrong | 124 | **114** | 114 |
| Labels cannot judge | 44 | **17** | 17 |

For the 46-video queue:

| Exact selected annotation | Result |
|---|---:|
| Precision among judgeable clips | **616 / 730 = 84.4%** |
| Recall across labelled rallies | **616 / 3,327 = 18.5%** |
| F1 | **30.4%** |
| Known-correct share if unknowns get no credit | **616 / 747 = 82.5%** |

“Unknown” means the labels cannot settle exact correctness.

## Proposal overlap and rally coverage

For cleaned labels across all 47 videos:

- 3,003 labelled rallies fit inside at least one proposal;
- 2,817 fit inside a proposal that overlaps exactly one labelled rally;
- 942 proposals overlap no cleaned rally;
- 71 overlap more than one;
- 2,969 overlap exactly one.

Best available rally output:

| Best available output | Historical 47 | Without video 15 |
|---|---:|---:|
| Fully correct | 1,763 | 1,763 |
| Contains all labels but has another error | 1,240 | 1,226 |
| Reaches some labels but not the whole rally | 153 | 113 |
| Reaches no labelled contact | 266 | 225 |

A clip can contain every labelled contact and still fail because it has extras, another rally's contacts or wrong players. The 225 unreached rallies remain after removing video 15. Without video 53 as well, 169 remain, so this is not just an outlier problem.

## What selection discards

Historically, selection keeps 784 of 3,982 proposals. Outside the queue are:

- 1,147 correct clips;
- 1,153 wrong clips;
- 898 unjudgeable clips.

Selection keeps 616 of the 1,763 available correct clips (**34.9%**), covering **18.0%** of the 3,422 cleaned rallies.

Examples from the saved queue:

| Video | Correct before selection | Correct selected | Wrong selected | Unjudgeable selected |
|---|---:|---:|---:|---:|
| 33 | 66 | 29 | 3 | 0 |
| 52 | 56 | 28 | 5 | 1 |
| 41 | 59 | 23 | 4 | 0 |
| 47 | 54 | 23 | 6 | 0 |
| 15 | 0 | 0 | 10 | 27 |

Video 15 is a reminder not to treat these per-video counts as release rankings.

## Errors inside selected clips

The historical 47-video queue has 124 known-wrong clips:

| Problem | Clips affected |
|---|---:|
| Extra contacts | 92 |
| Missing contacts | 74 |
| Wrong player on a matched contact | 10 |
| Rally cut off | 12 |

Categories overlap. None of the 124 fails only because of player assignment.

Video 15 badly distorts event counts, so the event-level breakdown below uses the 114 wrong clips outside it.

### Extra events

| Position | Events |
|---|---:|
| Before first label | 2 |
| Between first and last labels | 31 |
| After final label | **52** |
| **Total** | **85** |

### Missed labelled events

| Position | Events |
|---|---:|
| Serve | **35** |
| Middle | 8 |
| Final | **24** |
| **Total** | **67** |

These are event counts matched within each selected clip, not full-video contact counts.

In the historical 124-clip view, the largest exclusive combinations are 49 extras-only, 28 misses-only, 28 with both, and nine with misses + extras + a cut-off rally. Ten clips have a wrong matched player; all ten also have another error.

Serves and rally ends are good places to inspect, but “after the final label” is not proof that a physical hit is false. Earlier deletion work could not separate real tail contacts from bad extras reliably.

## Where misses occur

| State at the labelled frame | Labelled contacts | Matched | Missed | Miss rate |
|---|---:|---:|---:|---:|
| Court rejected | 2,614 | 240 | **2,374** | 90.8% |
| Court accepted; at least one player missing | 140 | 44 | **96** | 68.6% |
| Court accepted; both players present | 34,430 | 33,267 | **1,163** | 3.4% |
| **Total** | **37,184** | **33,551** | **3,633** | **9.8%** |

A label can match even when its exact frame is rejected because the ±10-frame window can reach a nearby event.

The 2,374 court-rejected misses are **65.3%** of all misses. Without video 53 as well, 1,640 of 2,891 misses are still in rejected scenes (**56.7%**).

## Rally position

Among court-accepted frames outside video 15:

| Contact position | Missed | Total | Miss rate |
|---|---:|---:|---:|
| Serve | 253 | 2,804 | 9.0% |
| Middle | 663 | 28,704 | 2.3% |
| Final | 343 | 3,062 | 11.2% |

Single-contact rallies count as serves only.

The ±5-frame check hurts serves especially strongly. For historical comparison, the full 47-video output at ±10 frames matches 2,781/3,422 serves, 28,195/31,415 middle contacts and 2,740/3,381 finals. Those counts include court-rejected frames, so they answer a different question from the accepted-scene table above.

## Timing of matched contacts

The rows are cumulative over the 33,551 matches outside video 15.

| Distance from label | Matches | Share |
|---|---:|---:|
| Exact frame | 9,012 | 26.9% |
| Within 2 frames | 28,035 | 83.6% |
| Within 5 frames | 32,878 | 98.0% |

Median offset is zero; mean offset is about half a frame early.

This describes only contacts that already match; the 3,633 misses are outside the timing-offset distribution. It does not support globally shifting predictions.

## Player assignment

In the historical 47-video result, 33,715 timing matches have a known target side. The learned output gets 32,667 right (**96.9%**):

| Labelled side | Predicted far | Predicted near | No player |
|---|---:|---:|---:|
| Far | 16,183 | 410 | 0 |
| Near | 632 | 16,484 | 6 |

That leaves 1,042 near/far confusions and six unassigned predictions. Near/far is image position, not persistent athlete identity across camera or end changes.

Within the historical selected queue, ten wrong clips have a wrong player on a matched contact, but every one also has another error.

## Video-to-video variation

Across the historical 47 videos, the median video has a 94.1% timing-match rate and a 53.2% fully correct-rally rate when each video gets equal weight.

At the high end:

- video 41: 59/77 fully correct (76.6%);
- video 33: 66/94 (70.2%);
- video 18: 42/60 (70.0%).

At the weak end, causes differ: video 53 is dominated by court rejection; video 17 mostly fails later; video 38 is mixed; video 15 is not a valid quality case at all. The [weak-video table](#weak-videos) gives the counts; [footage checks](video_checks.md#other-weak-videos) describe what was inspected.

## Weak videos

Low score tells us where to look, not what the cause is.

| Video | Timing matches | Fully correct rallies | Misses in rejected scenes | What we know |
|---|---:|---:|---:|---|
| 53 | 195/937 | 7/76 | 734/742 | Keep; labels check out, court gate dominates |
| 12 | 469/781 | 23/61 | 234/312 | Court-heavy; one sampled label is mistimed |
| 20 | 324/537 | 15/43 | 207/213 | Court-heavy |
| 21 | 583/806 | 31/75 | 200/223 | Court-heavy; outline substitution helps but still misses threshold |
| 24 | 248/330 | 11/31 | 76/82 | Mostly court-heavy |
| 39 | 577/717 | 38/75 | 120/140 | Mostly court-heavy |
| 17 | 842/976 | 17/73 | 1/134 | Mostly later-stage; two player losses traced to shared outline |
| 38 | 1,022/1,161 | 29/82 | 65/139 | Mixed; no single cause established |

Video 15 is omitted because its labels do not support a detector-quality score.

## Rally length

Historical 47-video counts show only a modest drop for the longest rallies:

| Rally length | Fully correct | Rate |
|---|---:|---:|
| 1–5 contacts | 517/989 | 52.3% |
| 6–10 | 536/1,039 | 51.6% |
| 11–20 | 505/967 | 52.2% |
| More than 20 | 205/427 | 48.0% |

Length is not the main explanation for the current failures.

## Label sets

| Labels | Rallies | Contacts | Use |
|---|---:|---:|---|
| Cleaned | 3,422 | 38,218 | Historical main score; current 46-video read removes video 15 |
| All source | 3,965 | 43,159 | Secondary accounting with rows restored after cleaning |

Old result files call these `retained` and `all_gt`. `retained` has nothing to do with whether the selection rule kept a clip. “Cleaned” also does not mean “known correct”: video 15 survived cleaning.

## Historical 47-video snapshot

At ±10 frames with cleaned labels:

- 3,982 proposed clips;
- 1,763 rallies with a fully correct clip;
- 41,605 emitted contacts;
- 33,716 / 38,218 labels timing-matched;
- 7,889 emitted contacts unmatched;
- 784 selected clips: 616 correct, 124 wrong, 44 unjudgeable.

At ±5, fully correct rallies fall to 1,430 and the selected queue becomes 549 correct, 191 wrong, 44 unjudgeable.

These counts are kept for lineage. Video 15 makes the 47-video aggregate a poor current quality estimate.

## What all source labels change

At ±10 frames, restoring all source rows:

- raises labelled contacts 38,218 → 43,159;
- raises timing matches 33,716 → 37,485;
- reduces unmatched emitted contacts 7,889 → 4,120;
- leaves the fully correct-rally total at 1,763, but changes which clips are counted as correct.

For the selected 784 clips:

| Cleaned judgement | All-source judgement | Clips |
|---|---|---:|
| Correct | Correct | 615 |
| Correct | Wrong | 1 |
| Wrong | Wrong | 124 |
| Unjudgeable | Wrong | 15 |
| Unjudgeable | Unjudgeable | 29 |

So 616 / 124 / 44 becomes **615 / 140 / 29**. No unjudgeable selected clip becomes confirmed correct.

At ±5 with all source labels: **549 correct / 207 wrong / 28 unjudgeable**, with 1,429 fully correct rallies.

## Contact precision and recall

Historical 47-video contact timing at ±10:

| Labels | Matched predictions | Predictions | Labels | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cleaned | 33,716 | 41,605 | 38,218 | 81.04% | 88.22% | 84.48% |
| All source | 37,485 | 41,605 | 43,159 | 90.10% | 86.85% | 88.45% |

All-source precision rises because more emitted contacts have a source label nearby, not because every restored label is reliable.

## Old review of unjudgeable clips

The earlier visual review covered all 44 historical selected clips that cleaned labels could not judge:

- 39 live-play clips;
- four replay/live mixtures;
- one apparent warm-up.

It checked footage and broad clip boundaries, not exact contact timing or player attribution, so it adds no “correct” credit.

Pointer: `scratch/contact_det_closing_pass/results/selected_clip_review.csv`.

## Reproduce

[Methods, saved files and commands](evaluation_reproduction.md#baseline-recount-and-context) explain how these counts were produced.
