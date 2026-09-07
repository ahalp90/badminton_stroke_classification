# Last annotator follow-ups

Earlier attempts to improve the detector left the existing output as the best option. This page records those experiments and the ideas set aside during the failure investigation. Work still to do is in [promising_leads.md](promising_leads.md).

**Contents**  
[Independent edge padding](#independent-edge-padding)  
[Correct chooser targets after padding](#correct-chooser-targets-after-padding)  
[One-edit repair headroom](#one-edit-repair-headroom)  
[Endpoint deletion](#endpoint-deletion)  
[Repairing video 15](#repairing-video-15)  
[Noise-aware training](#noise-aware-training)

## Independent edge padding

The existing boundary rule cancels both extensions when one side would admit an outside predicted contact. The alternative lets each edge extend independently.

Only two of 3,982 proposals changed, and neither became fully correct.

| Labels / tolerance | Existing | Independent edges |
|---|---:|---:|
| Cleaned, ±10 | 1,763 | 1,763 |
| Cleaned, ±5 | 1,430 | 1,430 |
| All source, ±10 | 1,763 | 1,763 |
| All source, ±5 | 1,429 | 1,429 |

The selected queue also stayed unchanged. One proposal starts three frames earlier but still has an extra contact; the other starts one frame earlier and contains the whole labelled interval but still misses a contact.

**Decision:** no gain; keep the existing rule.

Evidence: `scratch/contact_det_closing_pass/results/last_followups/edge_padding.json.gz`  
Script: `scratch/contact_det_closing_pass/scripts/replay_edge_padding.py`

## Correct chooser targets after padding

Some candidate sequences are labelled negative before final boundary padding even though padding later makes them complete. That is a real target mismatch, so it was worth fixing once.

On 32 development videos, 806 alternatives changed negative → positive and 154 proposals gained their first positive alternative:

| Group | Changed alternatives | Proposals | Wrong proposals affected |
|---|---:|---:|---:|
| A | 246 | 75 | 39 |
| B | 242 | 70 | 33 |
| C | 191 | 59 | 27 |
| D | 127 | 40 | 17 |
| **Total** | **806** | **244** | **116** |

No positive target became negative; all 59,757 excluded alternatives stayed excluded. Positive alternatives rose 6,834 → 7,640.

One 25 fps example starts at frame 55,914 while the labelled serve is 55,912. Its 20-contact sequence and players are already right; normal padding moves the start to 55,906 and makes it complete at ±10.

Development improved slightly: 1,209 → 1,218 fully correct at ±10 (22 repairs, 13 losses), and 958 → 965 at ±5 (14 repairs, seven losses). The cached upstream detector scores were not independent across development groups, so this was not a fully independent test.

The gain did not survive the broader run:

| Population | Existing | Corrected targets | Repairs | Losses |
|---|---:|---:|---:|---:|
| 47 videos, cleaned, ±10 | **1,763** | **1,761** | 21 | 23 |
| 47 videos, cleaned, ±5 | 1,430 | 1,424 | 13 | 19 |
| All source, ±10 | 1,763 | 1,761 | 21 | 23 |
| All source, ±5 | 1,429 | 1,423 | 13 | 19 |

The same 784 clips stayed selected:

| Labels / allowance | Correct | Wrong | Unjudgeable | Selected repairs / losses |
|---|---:|---:|---:|---:|
| Cleaned, ±10 | 616 → 614 | 124 → 126 | 44 | 0 / 2 |
| Cleaned, ±5 | 549 → 547 | 191 → 193 | 44 | 0 / 2 |
| All source, ±10 | 615 → 613 | 140 → 142 | 29 | 0 / 2 |
| All source, ±5 | 549 → 547 | 207 → 209 | 28 | 0 / 2 |

The two selected losses are `47/set2:33` and `48/set3:21`.

Contact metrics barely moved:

| Labels / allowance | Precision | Recall | F1 | Player-aware F1 |
|---|---:|---:|---:|---:|
| Cleaned, ±10 | 81.04 → 80.97 | 88.22 → 88.25 | 84.48 → 84.45 | 81.85 → 81.89 |
| Cleaned, ±5 | 79.25 → 79.20 | 86.27 → 86.32 | 82.61 → 82.61 | 80.19 → 80.26 |
| All source, ±10 | 90.10 → 90.03 | 86.85 → 86.89 | 88.45 → 88.43 | 85.58 → 85.59 |
| All source, ±5 | 88.10 → 88.04 | 84.93 → 84.97 | 86.49 → 86.48 | 83.92 → 83.94 |

Predictions rose 41,605 → 41,652; cleaned timing matches 33,716 → 33,726; matched cleaned serves 2,781 → 2,790. Those small contact gains do not outweigh lost complete rallies. Of the 23 cleaned ±10 losses, 12 finish with too few contacts, seven with extras and four with mistimed replacements.

**Decision:** reject the refit. The mismatch is real, but the finished output is slightly worse. Do not threshold-tune it back into life.

Evidence:
- `scratch/contact_det_closing_pass/results/last_followups/padded_targets.json.gz`
- `scratch/contact_det_closing_pass/results/last_followups/padded_fit_development.json.gz`
- `scratch/contact_det_closing_pass/results/last_followups/padded_fit_broader.json.gz`

Scripts: `run_padded_target_census.py`, `run_padded_target_fit.py`, `run_insertion_broader.py`, `score_padded_chooser.py`.

## One-edit repair headroom

Among 570 selected development clips, 119 are wrong. Labels can repair 58 of them with one edit from the existing candidate pool:

| Edit | Repairable clips |
|---|---:|
| Delete before first label | 5 |
| Delete after final label | 17 |
| Insert one later contact | 16 |
| Replace one event | 20 |
| **Unique clips** | **58** |

This is headroom, not achieved performance: the labels choose the successful edit after the fact. All 448 correct clips also have at least one damaging edit available.

The 58 repairable clips span A 20, B 11, C 22 and D 5. The other 61 wrong clips have no complete one-edit repair in this pool.

A simple tail-gap rule does not separate the 17 repairable tail deletions from correct endings. Their final gaps are 13.2–46.0 frames (median 26.0); among 447 currently correct clips with at least two predictions, gaps range 7.2–68.4 (median 27.6).

**Decision:** useful headroom count, not enough evidence for another correction model.

Evidence: `scratch/contact_det_closing_pass/results/last_followups/selected_repairs.json.gz`  
Script: `scratch/contact_det_closing_pass/scripts/count_selected_repairs.py`

## Endpoint deletion

Earlier work tried deleting events near rally starts and ends. The later error analysis still finds many extras after the final label, but the available labels cannot reliably tell which unsupported tail events are physically false hits.

**Decision:** reject broad endpoint deletion. “After the last label” is not a safe deletion rule.

Predecessor: `scratch/contact_det_closing_pass/serve_and_acceptance.md`.

## Repairing video 15

Early on, a timestamp shift or label repair was still conceivable. Later checks found mismatches across all three labelled games, including apparently strong timing matches.

**Decision:** do not repair. [#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) excludes the video and its derived release records.

## Noise-aware training

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

The [corrected-target experiment](#correct-chooser-targets-after-padding) above was rejected on broader finished outputs. Any future noise-handling experiment should be judged against a small set of manually verified contacts.

**Decision:** defer until the court bottleneck is fixed and a small verified contact set exists. Without manually checked targets, a noisy-label gain would be hard to interpret.
