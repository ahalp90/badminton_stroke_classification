# Annotator evaluation numbers

Compact numbers for the saved run with ShuttleSet22 video 15 removed. Video 53 stays in because its labels check out and its failures expose the court gate.

**Contents**  
[Population](#population)  
[Learned output](#learned-output)  
[Selected review queue](#selected-review-queue)  
[Where misses occur](#where-misses-occur)  
[Rally position](#rally-position)  
[Timing of matched contacts](#timing-of-matched-contacts)  
[Weak videos](#weak-videos)  
[Definitions](#definitions)  
[Reproduce](#reproduce)

## Population

- **46 ShuttleSet22 videos** after removing video 15.
- **3,327 cleaned rallies / 37,184 contacts.**
- **±10 frames at 30 fps** is the main timing allowance.
- The detector output itself is unchanged.

The 47-video column below is historical. The 45-video column without video 53 is a sensitivity check only.

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

These contact percentages are label-recovery rates. Historical prediction-side P/R/F1, all-source labels and the ±5 check are in [label_accounting.md](label_accounting.md).

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

## Timing of matched contacts

The rows are cumulative over the 33,551 matches outside video 15.

| Distance from label | Matches | Share |
|---|---:|---:|
| Exact frame | 9,012 | 26.9% |
| Within 2 frames | 28,035 | 83.6% |
| Within 5 frames | 32,878 | 98.0% |

Median offset is zero; mean offset is about half a frame early.

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

## Definitions

**Fully correct rally** — one proposed clip contains one complete labelled rally; every labelled contact matches once; there are no extra predicted contacts; every matched contact has the correct near/far player.

**Exact whole-rally contact sequence** — same check without player correctness.

**Contact timing match** — a predicted contact lies within the allowed frame distance under complete-video one-to-one matching.

**Selected clip** — a proposal kept by the existing ranking rule; “selected” does not mean correct.

**Cleaned labels** — the subset used for the main evaluation. Older docs call these “trusted”; video 15 is why that word should not be taken literally.

## Reproduce

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.evaluate_saved \
  --annotations "$ANNOTATIONS" \
  --output scratch/annotator_wrapup_evaluation/results
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_extended
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_video_exclusions
```

Full saved-file inventory and validation receipts: [evaluation_reproduction.md](evaluation_reproduction.md).
