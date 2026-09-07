# Ordinary heuristic versus learned output

The learned path is not a small cleanup of the hand-written annotator. On the 46 videos outside bad-label video 15, the ordinary heuristic gets **4 / 3,327 rallies fully correct (0.12%)**; the learned output gets **1,763 / 3,327 (53.0%)**.

The comparison is useful because both paths inherit the same court, pose and shuttle inputs. It shows which failures the learned contact work fixed and which upstream problems remain shared.

**Contents**  
[What is being compared](#what-is-being-compared)  
[Filtering the heuristic](#filtering-the-heuristic)  
[Complete rallies](#complete-rallies)  
[Why heuristic rallies fail](#why-heuristic-rallies-fail)  
[Same-label contact comparison](#same-label-contact-comparison)  
[Where contacts are lost](#where-contacts-are-lost)  
[Why the remaining learned error looks upstream](#why-the-remaining-learned-error-looks-upstream)  
[Player assignment](#player-assignment)  
[What this comparison tells us](#what-this-comparison-tells-us)  
[Evidence](#evidence)

## What is being compared

The saved annotation job contains:

- **heuristic candidates before filtering** — every raw hand-written contact candidate;
- **ordinary heuristic output** — candidates left after wrist checks, nearby-event suppression and exclusions;
- **learned output** — a later path that searches a broader option set and uses trained models to choose the sequence.

The ordinary filtered output is the real heuristic baseline. The three rows below are not simply cumulative stages over one fixed candidate list; the learned path can consider contacts the ordinary heuristic never kept.

## Filtering the heuristic

Historical 47-video counts:

| Output | Events emitted | Labelled contacts found | Label recall | Emitted events matched |
|---|---:|---:|---:|---:|
| Raw heuristic candidates | 98,470 | 30,012 / 38,218 | 78.5% | 30.5% |
| Ordinary heuristic | 53,649 | 29,206 / 38,218 | 76.4% | 54.4% |
| Learned output | 41,605 | 33,716 / 38,218 | 88.2% | 81.0% |

Filtering removes 44,821 heuristic candidates: 12,415 fail the wrist check and 32,406 are suppressed for being too close to another candidate. It makes the output much cleaner but loses 806 timing matches along the way.

Against cleaned labels, 24,443 heuristic events remain unmatched. With all source labels, 21,101 remain unmatched. Those are not all proven false hits; the cleaned labels omit rallies and video 15 is misaligned.

## Complete rallies

Only five heuristic clips contain the exact cleaned-label contact sequence at ±10 frames. Four also get every player right. Those four are in videos 21, 31, 37 and 50, with 2, 5, 3 and 10 contacts respectively.

| Output | Fully correct rallies outside video 15 |
|---|---:|
| Ordinary heuristic | **4 / 3,327 (0.12%)** |
| Learned output | **1,763 / 3,327 (53.0%)** |

Historical all-47 counts are 4/3,422 versus 1,763/3,422 at ±10, and 3/3,422 versus 1,430/3,422 at ±5.

The gain belongs to the whole learned path—broader candidates, sequence choice, player handling and boundary adjustment—not one isolated component.

## Why heuristic rallies fail

Across the heuristic's 3,982 historical proposals, cleaned labels judge four correct, 3,035 wrong and 943 unjudgeable.

| Problem | Wrong heuristic clips affected |
|---|---:|
| Extra contacts | 2,899 |
| Missing contacts | 2,358 |
| Wrong player on a matched contact | 2,024 |
| Rally cut off | 454 |
| Overlaps several labelled rallies | 70 |

The largest combination—missing contacts + extras + wrong player—appears in 1,516 clips. Another 412 have extras alone and 370 have missing + extra contacts without the other listed errors.

Among clips overlapping exactly one cleaned rally, the median heuristic clip has two misses and four extras. Only 70 have no extra event. That is why decent individual-contact recall still produces almost no exact rally sequences.

## Same-label contact comparison

Outside video 15, both outputs are scored against the same 37,184 contact labels:

| Result for the same label | Contacts |
|---|---:|
| Both find it | 28,351 |
| Learned only | **5,200** |
| Heuristic only | 660 |
| Neither | 2,973 |

The learned output gains far more contacts than it loses. Across all 47 videos, the corresponding counts are 28,486 shared, 5,230 learned-only, 720 heuristic-only and 3,782 missed by both.

These are matches against labels. The apparent gains and losses were not checked hit by hit against the footage.

Do not use video 15 to compare the methods: its 195 heuristic and 165 learned timing matches are comparisons against the wrong part of the match.

## Where contacts are lost

Outside video 15, using the full outputs across all upstream states:

| Contact position | Heuristic missed | Learned missed |
|---|---:|---:|
| Serve | 1,292/3,327 (38.8%) | 561/3,327 (16.9%) |
| Middle | 20.2% | 8.2% |
| Final | 21.8% | 17.3% |

This is a method comparison, not the court-accepted subset in [output_errors.md](output_errors.md).

The heuristic is also more sensitive to the timing tolerance. On the historical 47 videos, tightening ±10 to ±5 removes 1,740 heuristic matches (29,206 → 27,466), versus 744 learned matches (33,716 → 32,972).

## Why the remaining learned error looks upstream

Outside video 15:

| State for missed labels | Heuristic misses | Learned misses |
|---|---:|---:|
| Court rejected | 2,541 | 2,374 |
| Court accepted; at least one player missing | 69 | 96 |
| Court accepted; both players present | **5,563** | **1,163** |
| **Total** | **8,173** | **3,633** |

Both methods use the same court inputs. The learned path removes thousands of misses once the court and players are available, so court rejection rises from 31.1% of heuristic misses to **65.3% of learned misses**.

The 69-versus-96 middle row does not mean the learned model changed player tracking. A match can land up to ten frames from the labelled frame, so the two outputs can use different nearby input states.

## Player assignment

Historical 47-video results:

- heuristic: 20,204 correct sides among 29,205 timing matches with known sides (**69.2%**);
- learned output: 32,667 / 33,715 (**96.9%**).

The matched-contact sets differ, so this is an end-to-end comparison rather than an identical-hit A/B test.

The heuristic also leaves 1,619 timing matches without a player assignment. Its saved output has 386 rally spans without a settled first/last player assignment; 34 of those spans have no filtered contacts.

## What this comparison tells us

The ordinary heuristic is not a plausible fallback for exact rally annotation. The learned path turns a contact-heavy but structurally poor output into something useful enough to review.

More importantly, it shows where further contact modelling has already paid off: later-stage misses and player errors fall sharply when usable inputs exist. That is why another contact-model fit should wait until the court gate is fixed and rerun.

The learned selection rule is not applied to the heuristic here; its scores belong to the learned output. All heuristic clips are scored instead of inventing a fake equivalent selection.

## Evidence

- `results/heuristic_summary.json.gz`
- `results/heuristic_receipts.csv.gz`
- `results/heuristic_contacts.csv.gz`
- `results/heuristic_proposals.csv.gz`
- `results/heuristic_rallies.csv.gz`
- `results/heuristic_paired_contacts.csv.gz`
- `results/heuristic_paired_rallies.csv.gz`
- `results/heuristic_filtering_matches.csv.gz`
- `results/heuristic_position.csv.gz`
- `results/heuristic_upstream.csv.gz`
- `results/heuristic_per_video.csv.gz`
- `results/heuristic_error_combinations.csv.gz`

One-video smoke: 22.4 s. Full four-way recount: 444.3 s. Commands and validation: [evaluation_reproduction.md](evaluation_reproduction.md).
