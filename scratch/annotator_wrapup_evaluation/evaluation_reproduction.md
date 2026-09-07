# Annotator evaluation: saved evidence and reproduction

Rerun the saved evaluation or trace a published number here. Commands use the repository root and the project's existing Python environment unless stated otherwise.

**Contents**  
[Canonical saved inputs](#canonical-saved-inputs)  
[Environment variables](#environment-variables)  
[Result inventory](#result-inventory)  
[Baseline recount and context](#baseline-recount-and-context)  
[Visual sample and court/player checks](#visual-sample-and-courtplayer-checks)  
[Expanded summaries and heuristic comparison](#expanded-summaries-and-heuristic-comparison)  
[Video exclusions and alignment review](#video-exclusions-and-alignment-review)  
[Court issue replay](#court-issue-replay)  
[Per-video viewer](#per-video-viewer)  
[Validation already completed](#validation-already-completed)  
[Summary figures](#summary-figures)  
[Later direct hit judgements](#later-direct-hit-judgements)  
[Original-ShuttleSet recount](#original-shuttleset-recount)  
[Completed detector experiments](#completed-detector-experiments)  

## Canonical saved inputs

Paths are relative to the repository root.

| Record | Role |
|---|---|
| `scratch/contact_det_closing_pass/results/followups/local_boundary_broader_predictions_fixed_membership.json.gz` | Final learned contact choices plus fixed-membership clip padding |
| `scratch/contact_det_closing_pass/results/serve_followups/chosen_acceptance_broader.json.gz` | Unchanged 784-clip selection; threshold `0.7570784853533734` |
| `scratch/contact_det_closing_pass/results/metric_summary.json.gz` | Earlier headline counts reproduced by the recount |
| `scratch/contact_det_closing_pass/results/selected_clip_review.csv` | Earlier broad review of the 44 historical unknown selections |
| `src/annotator/court_evidence.py` | Scene court estimate, two-player check and shared-outline correction |
| `src/annotator/rally/evidence.py` | Tracker coverage and sequential player picker |
| `src/bst_x/preparing_data/heuristics/sticky_anchor.py` | Player projection, distance checks and carried position state |

Evaluation work began at commit `139f42c`. The cached prediction bundle records producer commit `ba24a95c334300c78e30a8d1b7c2a6134b8b5fa9`. The court, rally-evidence, sticky-player and vision-loading implementations used by this evaluation were unchanged between those commits. The final saved contact stream is the accepted closing-pass output above, **not** the rejected corrected-target refit documented in [last_followups.md](last_followups.md).

The published investigation is recorded at [`8a8562e26a9286ad491c3935f3860db66b91b020`](https://github.com/ahalp90/badminton_cv_annotator/commit/8a8562e26a9286ad491c3935f3860db66b91b020). The local follow-up records below add direct contact judgements and the original-ShuttleSet recount.

## Environment variables

The branch commands expect the original data and cached predictions to be available outside this directory:

```bash
export ANNOTATIONS=/path/to/shuttleset22/annotations
export PREPARED=/path/to/base/prepared/fixtures
export INPAINTED=/path/to/inpainted/shuttle/tracks
export SAVED_PREDICTIONS=/path/to/saved/prediction/bundle
export SOURCES=/path/to/source/videos
```

CSV and JSON outputs under `results/` are gzip-compressed.


## Result inventory

Compact numbers: [evaluation_tables.md](evaluation_tables.md). The tables below map those numbers to saved files and commands.

### Baseline and context

| Result | Contents |
|---|---|
| `results/baseline.json.gz` | Recount totals for both label populations and timing allowances |
| `results/proposals.csv.gz` | One proposal per population/tolerance with overlapping error flags and saved selection state |
| `results/rallies.csv.gz` | One labelled rally per population/tolerance, including rallies without a correct clip |
| `results/contacts.csv.gz` | One labelled contact per population/tolerance with complete-video one-to-one match identity |
| `results/predictions.csv.gz` | One emitted contact per population/tolerance with match identity |
| `results/selected_event_errors.csv.gz` | Missing/extra events inside known-wrong selected clips under within-clip matching |
| `results/contexts.csv.gz` | One unique requested video/frame; 82,533 rows |
| `results/scenes.csv.gz` | Saved scene intervals, validity and exactly-two-person vote counts |
| `results/metadata.json.gz` | Counts, frame-clock checks and available feature fields |
| `results/upstream_summary.csv.gz` | Missed/matched upstream summaries for all 47, without 15, and without 15+53 |
| `results/contact_position.csv.gz` | Contact-position summaries |
| `results/contact_position_court_accepted.csv.gz` | Contact position in accepted scenes outside video 15 |
| `results/error_combinations.csv.gz` | Historical selected error combinations |
| `results/per_video.csv.gz` | Historical per-video results |

### Visual, court and player evidence

| Result | Contents |
|---|---|
| `results/visual_pilot.csv.gz` | Eight initial targeted video-15/video-53 requests |
| `results/visual_sample.csv.gz` | Controlled 16-window sample requests |
| `results/visual_review.csv.gz` | Independent broad scene observations for that sample |
| `results/visual_geometry.json.gz` | Raw and active outlines at inspected frames in native image coordinates |
| `results/court_vote_check.csv.gz` | Original and alternative-outline people votes for eight whole scenes |
| `results/replay_player_sample.csv.gz` | Video 17 tracker replay and sampled geometry changes |
| `results/replay_player_sample.json.gz` | Replay receipts |
| `results/label_alignment_checks.csv.gz` | Early checked video-15 label/footage disagreements |
| `results/video53_nn_replay.json.gz` | Court-net corner replay and fallback comparison |

### Expanded and heuristic summaries

| Result | Contents |
|---|---|
| `results/extended_summary.json.gz` | Timing offsets, rally coverage and per-video extremes |
| `results/player_confusion.csv.gz` | Near/far side confusion on matched cleaned contacts |
| `results/label_judgement_changes.csv.gz` | Paired proposal judgements under cleaned and all-source labels |
| `results/selection_per_video.csv.gz` | Selection results per video |
| `results/selected_error_severity.csv.gz` | Sizes of selected errors |
| `results/input_conditional_rates.csv.gz` | Miss rates within each saved court/player state outside video 15 |
| `results/video15_error_contribution.csv.gz` | Historical error-count shares occurring in video 15; not causal attribution |
| `results/video15_followup_sample.csv.gz` | Five strong-timing-match source-frame requests |
| `results/video15_followup_labels.csv.gz` | Source labels for those requests |
| `results/video15_followup_review.csv.gz` | Observed game/score findings |
| `results/heuristic_summary.json.gz` | Native heuristic totals |
| `results/heuristic_receipts.csv.gz` | Native-output count and provenance checks |
| `results/heuristic_contacts.csv.gz` | Heuristic timing matches |
| `results/heuristic_proposals.csv.gz` | Heuristic clip errors |
| `results/heuristic_rallies.csv.gz` | Heuristic labelled-rally outcomes |
| `results/heuristic_paired_contacts.csv.gz` | Same-label learned/heuristic contact comparison |
| `results/heuristic_paired_rallies.csv.gz` | Same-label learned/heuristic rally comparison |
| `results/heuristic_filtering_matches.csv.gz` | Label matches before and after ordinary filtering |
| `results/heuristic_position.csv.gz` | Heuristic contact-position results |
| `results/heuristic_upstream.csv.gz` | Heuristic misses by saved upstream state |
| `results/heuristic_per_video.csv.gz` | Heuristic/learned per-video comparisons |
| `results/heuristic_error_combinations.csv.gz` | Overlapping heuristic clip-error combinations |

### Exclusions, alignment and per-video viewer

| Result | Contents |
|---|---|
| `results/exclusion_metrics.csv.gz` | Two methods × two label sets × two tolerances × three video populations |
| `results/video_outcome_breakdown.csv.gz` | 376 per-video rows across methods, label sets and tolerances |
| `results/video_player_confusion.csv.gz` | 4,512 per-video confusion cells, including missing/unknown states |
| `results/alignment_sample.csv.gz` | 53 visual requests |
| `results/alignment_labels.csv.gz` | Verified source-row identities for those requests |
| `results/alignment_observations.csv.gz` | Independent visual readings |
| `results/alignment_review.csv.gz` | Comparison of visual readings with source rows |
| `results/alignment_summary.csv.gz` | Summary by sampling group and court state |
| `results/alignment_per_video.csv.gz` | Alignment review by video |
| `results/video17_court_summary.json.gz` | Saved court-source counts for video 17 |

The `results/...` paths above are relative to `scratch/annotator_wrapup_evaluation/`.

## Baseline recount and context

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.evaluate_saved \
  --annotations "$ANNOTATIONS" \
  --output scratch/annotator_wrapup_evaluation/results

python -m scratch.annotator_wrapup_evaluation.scripts.collect_context \
  --annotations "$ANNOTATIONS" \
  --prepared-root "$PREPARED" \
  --inpainted-root "$INPAINTED" \
  --saved-root "$SAVED_PREDICTIONS" \
  --output scratch/annotator_wrapup_evaluation/results

python -m scratch.annotator_wrapup_evaluation.scripts.plot_evaluation
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_errors
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_context
```

Matching rules:

- a contact identity includes video, rally and source-label index;
- repeated label timestamps stay as distinct rows;
- join frame context many-to-one rather than collapsing repeated labels;
- full-stream matching and matching within one proposed clip answer different questions;
- a one-contact rally counts as a serve only;
- blank exact-feature fields mean no saved feature row at that frame, not “player absent”;
- `nearest_saved_row_distance` searches only ±10 frames; a blank value means no row in that window;
- tracker coverage refers to the original saved tracking segments;
- a saved scene-exclusion decision is an algorithmic state, not a human replay label;
- shuttle visibility means a filled coordinate exists, not that the coordinate is accurate;
- surrounding-window fractions use the half-open interval `[frame-15, frame+15)`;
- `population=retained` means cleaned labels and `population=all_gt` means all source labels; neither means that selection kept the proposal.

## Visual sample and court/player checks

Create the controlled sample and source sheets:

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.sample_context

python -m scratch.annotator_wrapup_evaluation.scripts.extract_views \
  --sample scratch/annotator_wrapup_evaluation/results/visual_sample.csv.gz \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/raw/control_sheets

python -m scratch.annotator_wrapup_evaluation.scripts.extract_views \
  --sample scratch/annotator_wrapup_evaluation/results/visual_pilot.csv.gz \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/raw/pilot_sheets

python -m scratch.annotator_wrapup_evaluation.scripts.plot_examples

python -m scratch.annotator_wrapup_evaluation.scripts.check_court_votes \
  --prepared-root "$PREPARED"

python -m scratch.annotator_wrapup_evaluation.scripts.replay_player_sample \
  --prepared-root "$PREPARED" \
  --saved-root "$SAVED_PREDICTIONS"
```

The controlled 16-window sample uses seed `20260906`. It deliberately over-samples failures and should not be used to estimate a collection-wide false-rejection rate.

## Expanded summaries and heuristic comparison

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_extended

python -m scratch.annotator_wrapup_evaluation.scripts.evaluate_heuristic \
  --annotations "$ANNOTATIONS" \
  --saved-root "$SAVED_PREDICTIONS" \
  --output scratch/annotator_wrapup_evaluation/results

python -m scratch.annotator_wrapup_evaluation.scripts.summarise_heuristic

python -m scratch.annotator_wrapup_evaluation.scripts.extract_views \
  --sample scratch/annotator_wrapup_evaluation/results/video15_followup_sample.csv.gz \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/raw/video15_followup

python -m scratch.annotator_wrapup_evaluation.scripts.plot_video15_checks
```

The heuristic scorer preserves the ordinary output's native player answers. It alternates each saved `fitted_first_all` over that rally's saved `filtered_by_rally` contacts and compares the resulting side with `striker_halves`. Raw candidates have no native per-event player answer, so they are scored for timing only. The scorer does not impose learned selection scores or learned player correction on the heuristic.

The one-video native smoke took 22.4 seconds. The full four-way recount took 444.3 seconds.

## Video exclusions and alignment review

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.summarise_video_exclusions
python -m scratch.annotator_wrapup_evaluation.scripts.sample_alignment_checks

python -m scratch.annotator_wrapup_evaluation.scripts.extract_alignment_labels \
  --sample scratch/annotator_wrapup_evaluation/results/alignment_sample.csv.gz \
  --annotations "$ANNOTATIONS" \
  --output scratch/annotator_wrapup_evaluation/results/alignment_labels.csv.gz

python -m scratch.annotator_wrapup_evaluation.scripts.extract_views \
  --sample scratch/annotator_wrapup_evaluation/results/alignment_sample.csv.gz \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/raw/alignment_checks

python -m scratch.annotator_wrapup_evaluation.scripts.summarise_alignment_checks
python -m scratch.annotator_wrapup_evaluation.scripts.build_video_view
```

The 53-window sample uses seed `20260907`.

The saved branch review records game/score observations. The later direct hit/player judgements are discussed separately below because they were published in issue #147 rather than folded into these branch result tables.

## Court issue replay

Video 53 court corners:

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.replay_court_corners \
  --prepared-root "$PREPARED" \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/results/video53_nn_replay.json.gz

python -m scratch.annotator_wrapup_evaluation.scripts.plot_court_issue
```

`replay_court_corners.py` checks the court-model weights against the saved receipt. The replayed OpenCV fallback matches the saved fallback within 0.02 pixels in the published check.

Plotting uses saved replay data and previously extracted source frames. Neither command changes detector predictions.

## Per-video viewer

The canonical repository can rebuild the standalone viewer with:

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.build_video_view
```

The generated `VIDEO_BREAKDOWN.html` embeds its data and does not need a network connection. It shows the 47 videos under the two saved output methods using cleaned labels and ±10 frames.

The standalone bundle does not copy the generated HTML blob. It links to the canonical generator and results instead, so the viewer does not become a second stale artifact.

## Validation already completed

The branch records these checks as successful:

- one-video smoke followed by complete 47-video baseline recount;
- both label populations and both timing allowances reproduce saved summaries;
- complete context extraction;
- all 162,754 population/tolerance contact rows join to frame context;
- all 14,774 rally identities join the learned-output tables;
- source-video frame/fps checks on the visual requests;
- exact original people-vote arrays in all eight diagnostic scenes;
- both current-frame player-validity fields match all 91,970 saved video-17 feature rows in the sequential replay;
- native filtering lists, player parity, frame bounds and counts pass for all 47 heuristic videos;
- all 53 source-row checks and frame extractions pass;
- all 94 viewer video/method choices pass matrix-sum and JavaScript rendering checks;
- scoped Ruff and Python syntax checks pass;
- compressed result tables parse successfully and local Markdown links were checked in the branch work;
- Serena/Pyrefly reports no diagnostics for the evaluation scripts.

The production code was unchanged, so the wrap-up did not use a whole-project test run as evidence of the observational results.

## Summary figures

From the directory containing these reports, run:

```bash
python figures/generate_published_figures.py
```

The script reads `results/exclusion_metrics.csv.gz` in the enclosing evaluation directory and writes the five summary PNGs beside itself. It uses cleaned labels at ±10 frames for both methods and all three video populations.

## Later direct hit judgements

The 24 direct checks are recorded locally under `scratch/annotator_wrapup_evaluation/worklog/`:

- `CONTACT_CASES.md` links each judgement to its frames and clip.
- `contact_judgements.json` stores the judgements and estimated hit intervals.
- `FOLLOWUP_RESULTS.md` explains the findings and how the footage was checked.

[Issue #147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) also publishes the findings. These human judgements leave the saved detector scores unchanged. Rerunning the scoreboard summaries does not recreate the direct checks.

The local worklog is Git-ignored. Retain it with the saved evidence; a clean checkout of the published commit does not contain it.

## Original-ShuttleSet recount

The recount scores saved final predictions on 32 development videos. It applies the existing fixed-membership boundary padding and runs no training or vision models. Tolerances below use a 30 fps clock and scale with the source frame rate.

| Final learned output | ±10 frames | ±5 frames |
|---|---:|---:|
| Whole rally interval inside a clip | 2,113 / 2,691 | 2,113 / 2,691 |
| Exact whole-rally contact sequence | 1,230 / 2,691 | 972 / 2,691 |
| Fully correct rally, including players | 1,209 / 2,691 | 958 / 2,691 |
| Contact timing match | 24,952 / 27,571 | 24,373 / 27,571 |
| Contact timing + correct player | 24,285 / 27,571 | 23,766 / 27,571 |
| Serve timing match | 1,894 / 2,691 | 1,565 / 2,691 |
| Serve timing + correct player | 1,790 / 2,691 | 1,487 / 2,691 |

The containing clip may span more than one rally. Exact-sequence and fully correct results require one complete rally with no missing or extra contacts.

Under `scratch/annotator_wrapup_evaluation/worklog/`, `original_development/score_original_development.json.gz` holds the totals. Its companion `score_original_development_per_video.csv.gz` holds the per-video results. `FOLLOWUP_RESULTS.md` includes the comparison chart and the inputs needed for the remaining eight videos: 18, 22, 24, 25, 30, 31, 39 and 40.

Run from the repository root, choosing a fresh output directory:

```bash
PYTHONPATH="$PWD/src:$PWD" ~/.venvs/badminton-cicd/bin/python \
  -m scratch.annotator_wrapup_evaluation.worklog.score_original_development \
  --output /tmp/annotator-original32-recount
```

Add `--limit 1` for a one-video smoke check. The completed recount reproduced the saved 1,209 and 958 fully correct rallies; all 64 per-video rows summed to the totals.

## Completed detector experiments

The rejected closing-pass experiments are documented in [last_followups.md](last_followups.md). Their original reproduction commands require the predecessor caches and should be run to a fresh output directory:

```bash
export PYTHONPATH="$PWD/src:$PWD"
followup_run=/path/to/fresh/contact-followup

python -m scratch.contact_det_closing_pass.scripts.replay_edge_padding \
  --annotations /path/to/shuttleset22/annotations \
  --output "$followup_run/edge_padding.json.gz"

python -m scratch.contact_det_closing_pass.scripts.run_padded_target_census \
  --output-root "$followup_run/targets" --jobs 16

python -m scratch.contact_det_closing_pass.scripts.run_padded_target_fit \
  --census "$followup_run/targets" \
  --output-root "$followup_run/fit" --jobs 4

python -m scratch.contact_det_closing_pass.scripts.run_insertion_broader \
  --variant local \
  --models "$followup_run/fit/models.joblib" \
  --output-root "$followup_run/broader" \
  --score-root "$followup_run/scores" --jobs 4

python -m scratch.contact_det_closing_pass.scripts.score_padded_chooser \
  --predictions "$followup_run/broader/local_broader_predictions.json.gz" \
  --annotations /path/to/shuttleset22/annotations \
  --output "$followup_run/broader/padded_comparison.json.gz"

python -m scratch.contact_det_closing_pass.scripts.count_selected_repairs \
  --output-root "$followup_run/selected_repairs" --jobs 16
```

The target census accepts `--limit-fixtures 1` for a one-video smoke. The selected-repair census accepts `--limit-videos 1`. The completed runs and focused tests exited successfully; the predecessor report also records scoped Ruff success. A whole-project Pyrefly run returned 11 missing-import errors in unchanged tests, helper scripts and optional video-language-model dependencies, so that failure was not used as evidence against the self-contained experiments.
