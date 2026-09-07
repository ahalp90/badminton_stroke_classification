# Label and video checks

**Video 15 should be excluded. Video 53 should stay.** The later checks found a few isolated label problems elsewhere, but no second large wrong-rally mismatch.

**Contents**  
[Video 15: exclude it](#video-15-exclude-it)  
[Video 53: keep it](#video-53-keep-it)  
[Other weak videos](#other-weak-videos)  
[Random missed-contact sample](#random-missed-contact-sample)  
[What the 53-window review tells us](#what-the-53-window-review-tells-us)  
[Original ShuttleSet](#original-shuttleset)  
[Noisy labels and the HGB models](#noisy-labels-and-the-hgb-models)  
[Evidence](#evidence)

## Video 15: exclude it

Video 15 is An Se Young versus Akane Yamaguchi at the 2022 Uber Cup semi-final. Its labels do not line up with the downloaded footage:

- first labelled serve: frame 186, while opening graphics are still on screen;
- frame 45,575: label says game 2, rally 16, score 8–8; footage shows a replay and arena score 17–11;
- around frame 97,748: label says game 3, score 12–11; visible scoreboard shows 0–0.

Five later windows were chosen because their timing looked unusually good. They also show the wrong game or score, including both rallies where every cleaned label had a timing match:

| Labelled rally | Timing matches | Source game/score | Visible game/score |
|---|---:|---|---|
| Game 2, rally 39 | 6/6 | Game 2, 21–18 | Game 2, 9–8 |
| Game 2, rally 7 | 2/2 | Game 2, 5–2 | Game 1, 13–5 |
| Game 1, rally 31 | 6/12 | Game 1, 13–18 | Game 1, 1–0 |
| Game 3, rally 30 | 4/5 | Game 3, 15–15 | Game 3, 5–2 |
| Game 2, rally 24 | 5/6 | Game 2, 10–14 | Game 2, 2–0 |

So even a complete timing match can be coincidence against another rally.

Across ten targeted windows and four later random misses, no checked section has been confirmed as a reliable pairing. The direct review of those four random misses found three clear footage disagreements and one unresolved case.

[#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) therefore drops video 15 and its derived release records instead of trying to repair the labels.

The old video-15 detector counts are only historical disagreement counts:

- 0 / 95 fully correct rallies;
- 165 / 1,034 timing matches;
- 81 timing matches with the labelled player;
- 869 of the historical 4,502 missed labels;
- 27 of the historical 44 unjudgeable selected clips.

Do not read those as detector accuracy on the visible match.

For audit, video 15's share of the old error totals was:

| Historical outcome, cleaned labels, ±10 | Video 15 | All 47 | Share in video 15 |
|---|---:|---:|---:|
| Missed labelled contacts | 869 | 4,502 | 19.3% |
| Emitted contacts without a label match | 2,092 | 7,889 | 26.5% |
| Rallies without a fully correct clip | 95 | 1,659 | 5.7% |
| Known-wrong selected clips | 10 | 124 | 8.1% |
| Extra events inside wrong selected clips | 97 | 182 | 53.3% |
| Missed events inside wrong selected clips | 118 | 185 | 63.8% |

These rows count different things and must not be added. They show concentration, not causal attribution.

## Video 53: keep it

Video 53 also scores badly, but its labels hold up under direct checks.

Nine game/score windows span both labelled games and agree with the source rows. Four random missed contacts were then checked directly; all four support the visible hit and player. The court stage had rejected all four labelled frames.

Its failures line up with the court stage:

- 742 missed labels;
- **734 in court-rejected scenes**;
- in accepted scenes, 195 / 203 labels have a timing match;
- 194 of those have the right player.

Video 53 is therefore useful evidence about the court gate. Removing it would hide a real pipeline weakness.

## Other weak videos

Three missed-contact windows were checked in each of seven other weak videos:

| Video | Timing matches | Fully correct rallies | Rejected-scene misses | Game/score checks |
|---|---:|---:|---:|---:|
| 12 | 469/781 | 23/61 | 234/312 | 3/3 consistent |
| 20 | 324/537 | 15/43 | 207/213 | 3/3 consistent |
| 21 | 583/806 | 31/75 | 200/223 | 3/3 consistent |
| 24 | 248/330 | 11/31 | 76/82 | 3/3 consistent |
| 39 | 577/717 | 38/75 | 120/140 | 3/3 consistent |
| 17 | 842/976 | 17/73 | 1/134 | 3/3 consistent |
| 38 | 1,022/1,161 | 29/82 | 65/139 | 3/3 consistent |

No second large wrong-rally mismatch appeared. Twenty of the 21 centres show the whole court and both players; the remaining video-38 centre is a close-up that returns to play half a second later.

The error mix differs by video. Court rejection dominates 12, 20, 21, 24, 39 and 53. Video 17 mostly fails later. Video 38 is mixed.

One later direct check did find a real video-12 timing error: the visible hit is around frames 18,218–18,220 while the label is 18,232. One bad timestamp is not a reason to discard the whole video.

## Random missed-contact sample

The branch drew 24 missed cleaned labels uniformly from the historical 4,502 misses with seed `20260907`.

First, game and score were compared with the source row:

| Game/score | Court accepted | Court rejected | Total |
|---|---:|---:|---:|
| Consistent | 7 | 13 | 20 |
| Different | 1 | 1 | 2 |
| Unreadable | 0 | 2 | 2 |

The two contradictions and two unreadable windows were all in video 15.

The same 24 were later judged directly for hit/player agreement:

| Direct judgement | Count |
|---|---:|
| Hit and player agree | **16** |
| Clear footage disagreement | **3** |
| Timing label wrong | **1** |
| Unclear | **4** |

All three clear disagreements are video 15. The timing error is video 12.

The two checks are complementary: a correct scoreboard does not prove an exact contact timestamp, and an unreadable scoreboard does not make the physical hit useless. One video-24 serve, for example, has the right game and score but lands during a replay-to-live transition that sparse stills cannot time precisely.

Court acceptance also says nothing by itself about label correctness. The sample contains consistent labels in both accepted and rejected scenes, and video-15 contradictions in both states.

## What the 53-window review tells us

The later visual review contains 53 short windows from 19 of the 47 evaluated ShuttleSet22 videos:

- 24 random misses;
- 21 misses from seven weak videos;
- three extra video 53 misses;
- five successful controls.

Each request has nine stills at half-second intervals over ±2 seconds plus a full-resolution centre frame. Readers recorded visible game/score before seeing labels or detector outcomes.

This is enough to find gross wrong-rally mismatches, inspect view/player visibility, and sometimes judge the actual hit. It is **not** enough to estimate a collection-wide label-error rate, certify every contact timestamp, measure replay false rejection, track athlete identity through the whole match, or validate shuttle coordinates.

The weak-video and control windows were deliberately selected, so do not pool all 53 into a prevalence estimate.

## Original ShuttleSet

[#133](https://github.com/ahalp90/badminton_cv_annotator/issues/133) lists **40 original ShuttleSet videos and 58 ShuttleSet22 videos** for release. The main investigation covers 47 ShuttleSet22 videos. A later recount also scored the saved final output on 32 original-ShuttleSet development videos.

On those 32 videos, contact results are slightly better than the historical 47-video ShuttleSet22 result, but serves and whole rallies are weaker. At the same ±10-frame allowance on a 30 fps clock:

| Final learned output | Correct / labelled | Rate |
|---|---:|---:|
| Fully correct rally | 1,209 / 2,691 | 44.9% |
| Contact timing + correct player | 24,285 / 27,571 | 88.1% |
| Serve timing + correct player | 1,790 / 2,691 | 66.5% |

These videos were used during development, so the comparison describes saved performance on familiar data. It adds no new footage checks. Full counts at both tolerances and the recount command are in [evaluation_reproduction.md](evaluation_reproduction.md#original-shuttleset-recount).

Eight more original-ShuttleSet videos have earlier detector outputs and saved features, but still need the final chooser inputs and boundaries for a like-for-like detector comparison.

That detector work is not needed for the label-quality sweep. Reuse existing labels, frames and prior evidence first. [#77](https://github.com/ahalp90/badminton_cv_annotator/issues/77) already notes that some first-stroke timestamps do not mark the actual serve contact.

Earlier visual work covered a small named set including `sset_01`, `sset_15` and `sset_21`; it was not collection-wide, and those IDs are unrelated to ShuttleSet22 video numbers.

## Noisy labels and the HGB models

The two histogram-gradient-boosting models use ordinary regularisation, not a special noisy-label method:

| Setting | Main contact model | Later chooser |
|---|---:|---:|
| Learning rate | 0.06 | 0.05 |
| Maximum leaves | 31 | 15 |
| Minimum examples per leaf | 40 | 20 |
| L2 penalty | 1.0 | 1.0 |
| Early stopping | Automatic | Disabled |

Both leave `max_features=1.0`. Targets are hard yes/no labels; unjudgeable examples are omitted. There is no dropout, label smoothing or per-example reliability weighting.

Regularisation can reduce overfitting. It cannot move a label to the right rally or restore a candidate removed by the court gate.

A previous target-definition change was tested and rejected on broader finished outputs; see [last_followups.md](last_followups.md). Any future noise-handling experiment should be judged against a small set of manually verified contacts.

## Evidence

- `results/alignment_sample.csv.gz`
- `results/alignment_labels.csv.gz`
- `results/alignment_observations.csv.gz`
- `results/alignment_review.csv.gz`
- `results/alignment_summary.csv.gz`
- `results/alignment_per_video.csv.gz`
- `results/video15_followup_sample.csv.gz`
- `results/video15_followup_labels.csv.gz`
- `results/video15_followup_review.csv.gz`
- `results/label_alignment_checks.csv.gz`
- [Direct contact checks](evaluation_reproduction.md#later-direct-hit-judgements) — local case records and footage
- [issue #147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) — later direct hit/player judgements
- [figures guide](figures/README.md) — source-frame evidence links
