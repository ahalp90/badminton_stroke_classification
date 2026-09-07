# Label and population accounting

Use this file when an old total looks different or you need the all-source / ±5 / prediction-side view. The main 46-video numbers are in [evaluation_tables.md](evaluation_tables.md).

**Contents**  
[Label sets](#label-sets)  
[Historical 47-video snapshot](#historical-47-video-snapshot)  
[What all source labels change](#what-all-source-labels-change)  
[Contact precision and recall](#contact-precision-and-recall)  
[Proposal overlap and rally coverage](#proposal-overlap-and-rally-coverage)  
[What selection discards](#what-selection-discards)  
[Old review of unjudgeable clips](#old-review-of-unjudgeable-clips)

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

## Old review of unjudgeable clips

The earlier visual review covered all 44 historical selected clips that cleaned labels could not judge:

- 39 live-play clips;
- four replay/live mixtures;
- one apparent warm-up.

It checked footage and broad clip boundaries, not exact contact timing or player attribution, so it adds no “correct” credit.

Pointer: `scratch/contact_det_closing_pass/results/selected_clip_review.csv`.
