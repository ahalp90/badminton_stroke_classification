# Annotator evaluation: where to start

For the questions about videos 15 and 53, start with
[the follow-up answers](VIDEO_CHECKS.md). That report now says what was counted in
full, what was checked in a sample, and what is still open.

The decision is to exclude ShuttleSet22 video 15 from use and from the released
extracts. [Issue #147](https://github.com/ahalp90/badminton_cv_annotator/issues/147)
contains the evidence images, completed label checks and remaining release work.
It also calls for an efficient check of both datasets for other unreliable labels.

The reports cover different questions. You do not need to read them all in order.

| What you want to know | Where to read |
|---|---|
| What changes without videos 15 and 53? Could the labels explain the failures? What was checked elsewhere? | [Follow-up answers](VIDEO_CHECKS.md) |
| What exactly goes wrong with the court outlines? | [Issue #148 with numbered before-and-after pictures](https://github.com/ahalp90/badminton_cv_annotator/issues/148); [local copy](COURT_ISSUE.md) |
| How does one particular video perform? | [Per-video viewer](VIDEO_BREAKDOWN.html); open locally in a browser |
| What does the final detector produce overall? | [Original overview](REPORT.md) |
| More detail on missed hits, wrong players, clip selection and the initial footage checks | [Expanded report](REPORT_BIG.md) |
| How does the ordinary heuristic compare with the learned detector? | [Heuristic comparison](HEURISTIC_REPORT.md) |

All scores describe saved outputs on the same 47 ShuttleSet22 videos. These videos
had already been examined. The separate 32 development videos are not part of those
scores. No detector or label changes were made during this investigation.

The latest footage checks and numbered court images came after the original overview
and expanded report. Those earlier reports keep their original sample counts and
point to the follow-up where it adds evidence.

The rest of this page is for checking the source files or rerunning the work.

## Saved inputs and scoring rules

All paths below are relative to the repository root.

| Record | Role |
|---|---|
| `scratch/contact_det_closing_pass/results/followups/local_boundary_broader_predictions_fixed_membership.json.gz` | Saved final output: chosen contacts and extended clip edges, with the same contacts kept in each clip |
| `scratch/contact_det_closing_pass/results/serve_followups/chosen_acceptance_broader.json.gz` | The unchanged list of 784 clips kept for review; score threshold 0.7570784853533734 |
| `scratch/contact_det_closing_pass/results/metric_summary.json.gz` | Previous headline counts, reproduced by assertions in the recount |
| `scratch/contact_det_closing_pass/scripts/summarise_metrics.py` | Code that reads the label sets, matches contacts and judges complete rallies |
| `scratch/contact_det_closing_pass/results/selected_clip_review.csv` | Earlier broad visual review of the 44 unknown selections |
| `src/annotator/court_evidence.py` | Scene court estimate, two-player check, then correction using the shared outline |
| `src/annotator/rally/evidence.py` | Tracker coverage, original sequential player picker and resets |
| `src/bst_x/preparing_data/heuristics/sticky_anchor.py` | Player projection, selection distances and carried position state |

The evaluation started at commit `139f42c`. The cached prediction bundle records
producer commit `ba24a95c334300c78e30a8d1b7c2a6134b8b5fa9`. The court, rally-evidence,
sticky-picker and vision-loading implementations used here were unchanged between
those commits. The final saved contact stream also retains the closing-pass changes
named above; it is not the rejected corrected-target refit.

## Result tables

CSV and JSON files use gzip compression. Frames refer to the source video, at 30 fps.
`fixture` is the broader video's numeric ID; it is not the old development fixture ID.

| File in `results/` | Unit and meaning |
|---|---|
| `baseline.json.gz` | Recount totals for both label populations and both timing allowances |
| `proposals.csv.gz` | One proposal per population and tolerance; overlapping error flags and unchanged clip selection |
| `rallies.csv.gz` | One labelled rally per population and tolerance, including rallies without a correct clip |
| `contacts.csv.gz` | One labelled contact per population and tolerance; complete-video one-to-one matching |
| `predictions.csv.gz` | One emitted contact per population and tolerance, with complete-video match identity |
| `selected_event_errors.csv.gz` | Individual missing/extra events in selected wrong clips, using within-clip matching |
| `contexts.csv.gz` | One unique video/frame requested by labels or predictions; 82,533 rows |
| `scenes.csv.gz` | Saved scene intervals, validity and exactly-two-person vote counts |
| `metadata.json.gz` | Counts, frame-clock checks and available feature fields |
| `upstream_summary.csv.gz` | Missed/matched input summaries for all 47, excluding 15, and excluding 15 and 53 |
| `contact_position*.csv.gz` | Contact-position summaries; second file restricts to accepted scenes outside video 15 |
| `error_combinations.csv.gz`, `per_video.csv.gz` | Selected error combinations and per-video results |
| `visual_pilot.csv.gz` | Eight initial targeted requests, including the video-15 source disagreement |
| `visual_sample.csv.gz`, `visual_review.csv.gz` | Exact new sample requests and independent broad scene observations |
| `visual_geometry.json.gz` | Raw and active outlines at inspected frames, in native image coordinates |
| `court_vote_check.csv.gz` | Original and alternative-outline people votes for eight whole scenes |
| `replay_player_sample.csv.gz`, `replay_player_sample.json.gz` | Video 17 tracker replay and checks that change only the court outline |
| `label_alignment_checks.csv.gz` | Three checked video-15 disagreements between source rows and footage |

`population=retained` means the cleaned (trusted) labels; `all_gt` means all source labels.
Neither value means that selection kept a proposed clip.

A contact's identity includes its video, rally and label index. Repeated label timestamps
are retained. Join frame context many-to-one; never collapse distinct label rows.
A full-stream timing match and a match within a proposed clip answer different questions.
Single-contact rallies count as serves, not also as final contacts.

Blank exact-feature fields mean there was no saved row at that frame. They do not mean
that the player was absent. `nearest_saved_row_distance` searches only within ±10
frames; a blank value means no row in that window. Coverage refers to the original
tracker segments. The saved exclusion decision is not a human replay label.
Shuttle availability uses the track's visibility field and does not measure coordinate
accuracy. Surrounding-window fractions use the half-open interval [frame−15, frame+15).

## How the initial footage and court checks were done

The initial eight-frame pilot targeted the unusually poor videos 15 and 53. It found
video 15's label disagreement. Its observations are separate from the following sample.

The initial 16-window sample uses seed 20260906 and eight missed middle contacts outside video 15:
four court-rejected examples, two with a missing player pick, and two with both picks.
Each has a successful middle-contact control from a fully correct rally in the same
video, chosen by nearest rally length and then time. All centres are more than two
seconds from a *saved* scene cut. That restriction does not guarantee no actual cut.

Random IDs V01–V16 hide outcome and input state from the scene reader. Each sheet has
nine frames at half-second intervals over ±2 seconds. The reader assessed camera view,
player visibility, action, likely live/replay status and transitions. These sparse
stills support broad scene observations; they do not certify exact contact times or
normal playback speed. The sample deliberately contains many failures and cannot
estimate collection-wide false-rejection rates.

The court check first reproduces the entire saved people-vote array in each of eight
scenes. It then substitutes only the existing same-video consensus outline. This
measures the effect on the people check, without writing new detector masks or outputs.
Video 21 remains below the original 50% threshold and is retained as a contrary case.

The player check replays all of video 17 with the original settings and state resets.
It checks all saved current-frame player-validity fields against the replayed picks.
At four sample centres and ±0.5/1 second, it records the actual incoming player-position
state. An isolated second calculation changes only the outline to that scene's raw
outline; its result never affects the next frame. Raw detection indices identify
candidates within a frame, not persistent physical players.

Existing human scene labels used a different dataset's IDs, so they could not support
collection-wide camera-view or replay error rates. Exact unsupported endpoint hits, physical
player swaps and shuttle-coordinate accuracy also remain unresolved. These omissions
are deliberate; the available records do not establish them.

## Reproduce

Use the project's existing Python environment with NumPy, pandas, matplotlib and the
annotator dependencies. Run from the repository root. Supply paths to the original
annotation tree, base prepared fixtures, separate inpainted tracks, saved prediction
bundle and source videos through the arguments shown below. Cached inputs are not
included in this directory.

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
  --prepared-root "$PREPARED" --saved-root "$SAVED_PREDICTIONS"
```

`plot_examples` uses the retained sample metadata plus locally extracted native frames.
Its video-15 pilot frames are recorded in `label_alignment_checks.csv.gz`. Broad visual
judgements are human-readable observations, not generated numerical ground truth.
The report embeds six figures; supplementary figures show error combinations and
per-video variation. Quantitative figures also have SVG copies.

Validation uses real-data recounts, source-frame checks, exact array comparisons,
complete joins, script syntax and scoped lint. Production code is unchanged, so the
whole-project test suite is outside this evaluation's verification scope.

Completed checks (all exit 0):

- One-video smoke followed by the complete 47-video baseline recount; both populations and tolerances match the saved summaries.
- One-video context smoke followed by the complete extraction; all 162,754 population/tolerance label rows join to frame context.
- Source-video frame/fps checks on all 24 visual requests.
- Exact original people-vote arrays in all eight diagnostic scenes.
- Both current-frame player-validity fields at all 91,970 saved video-17 feature rows match the sequential replay.
- Ruff on the evaluation scripts, Python syntax compilation, compressed-table parsing and local Markdown links.

Serena/Pyrefly reported no diagnostics for the ten evaluation scripts. No full-project
lint, type or test run was needed for these self-contained observational scripts.

## Expanded questions and ordinary heuristic comparison

[REPORT_BIG.md](REPORT_BIG.md) answers the additional questions behind the short report.
It includes the scope of the video 15 alignment checks, the share of each reported
error count occurring in that video, and checks of its strongest timing matches.
Those shares describe where errors are counted; they do not estimate how many errors
misalignment caused.

[HEURISTIC_REPORT.md](HEURISTIC_REPORT.md) evaluates the ordinary end-to-end heuristic
output before the learned chooser. It uses the same 47 videos and source labels.
The saved `videos/ss22_NN/annotation/annotator_result.json.gz` files contain the ordinary
outputs: the broader prediction producer called `run_full_annotation_stage` with its
default configuration and shipped landing settings. The relaxed impulse/wrist regions
were created later for learned feature search. No vision model needed to be rerun.

The heuristic scorer preserves native player answers. It alternates each saved
`fitted_first_all` over that rally's saved `filtered_by_rally` contacts and checks the
final side against `striker_halves`. It uses source-label row identities when timestamps
repeat. Raw candidates are scored for timing only; they have no native per-event side.
The learned output's selection and extra side correction are not imposed on the heuristic.

| Added result files | What they contain |
|---|---|
| `extended_summary.json.gz` | Timing offsets, rally coverage and per-video extremes |
| `player_confusion.csv.gz` | Near/far side confusion on matched cleaned contacts |
| `label_judgement_changes.csv.gz` | Paired clip judgements under cleaned and all-source labels |
| `selection_per_video.csv.gz`, `selected_error_severity.csv.gz` | Retained output and sizes of selected errors |
| `input_conditional_rates.csv.gz` | Missed-contact rate within each input state outside video 15 |
| `video15_error_contribution.csv.gz` | Error-count shares occurring in video 15; no causal attribution |
| `video15_followup_sample.csv.gz`, `video15_followup_labels.csv.gz`, `video15_followup_review.csv.gz` | Five additional strong-match requests, original source scores and observed scoreboard findings |
| `heuristic_summary.json.gz`, `heuristic_receipts.csv.gz` | Native-output totals and receipt/count checks across all 47 videos |
| `heuristic_contacts.csv.gz`, `heuristic_proposals.csv.gz`, `heuristic_rallies.csv.gz` | Native timing matches, clip errors and labelled-rally outcomes |
| `heuristic_paired_contacts.csv.gz`, `heuristic_paired_rallies.csv.gz` | Same-label comparison with final learned output |
| `heuristic_filtering_matches.csv.gz` | Label matches before and after ordinary filtering |
| `heuristic_position.csv.gz`, `heuristic_upstream.csv.gz`, `heuristic_per_video.csv.gz` | Contact position, saved input state and per-video comparisons |
| `heuristic_error_combinations.csv.gz` | Overlapping native-clip error combinations |

For the video 15 follow-up, the two fully timing-matched cleaned rallies were selected
first: set2:39 and set2:7. The other requests inspect strong matches from game 1, game 3,
and the next strongest six-contact example. Each window has nine frames over ±2 seconds.
The scoreboard comparisons tolerate reversing score order and a one-point difference
between before/after-point recording. All five still disagree. This targeted check
establishes no collection-wide alignment-error rate or verified aligned subset.

Run the extra aggregations and saved-output comparison from the repository root:

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

The native one-video smoke took 22.4 seconds; the full four-way recount took 444.3
seconds. Both exited 0. All 162,754 contact identities and 14,774 rally identities
joined the learned-output tables exactly. Native filtering lists, player parity,
frame bounds and counts passed for all 47 videos. New scripts passed scoped Ruff,
syntax checks and Serena/Pyrefly diagnostics. Independent technical review checked
ordinary defaults, a correct native rally, a seeded wrong rally and the new source-frame
scoreboard evidence. Production code and the original short report remain unchanged.

## Excluding videos and checking more footage

[VIDEO_CHECKS.md](VIDEO_CHECKS.md) compares all 47 videos, the 46 without video 15,
and the 45 without videos 15 and 53. The outputs and selection remain fixed.
Four charts separate exact whole-rally timing, fully correct rallies, contacts and
serves. Correct attribution requires both timing and the hitting player.

The new visual sample has 53 windows across 19 ShuttleSet22 videos. It includes
24 uniformly sampled missed labels, 21 misses from seven weak videos, three further
video 53 misses and five successful controls. Seed: 20260907. Each window contains
nine stills at half-second intervals over ±2 seconds, plus its full-resolution centre.
Readers recorded the visible game and score before seeing source labels or outcomes.
The comparison permits reversed player order and a total score difference of one point.
It checks whether the label could refer to that rally; it does not certify exact hits.
Targeted checks and controls are kept separate from the random sample.

| Added result files | Contents |
|---|---|
| `exclusion_metrics.csv.gz` | 24 comparisons: two methods, two label sets, two tolerances and three video populations |
| `video_outcome_breakdown.csv.gz` | 376 per-video rows across methods, label sets and tolerances |
| `video_player_confusion.csv.gz` | 4,512 cells, including unknown target players and missing predictions |
| `alignment_sample.csv.gz`, `alignment_labels.csv.gz` | Requests and 53 verified source-label identities, including game and score |
| `alignment_observations.csv.gz`, `alignment_review.csv.gz` | Independent visual readings and their comparison with source rows |
| `alignment_summary.csv.gz`, `alignment_per_video.csv.gz` | Results by sampling group, court state and video |
| `video17_court_summary.json.gz` | Saved court-source counts and the scene covering the two earlier player failures |

Reproduce the local aggregation and source requests:

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

The last two commands use the saved visual observations. They do not generate new
visual judgements. The HTML viewer contains its data and needs no network connection.
Its 94 video/method combinations use cleaned labels and ±10 frames; the compressed
tables retain the other comparisons.

All 53 source-row checks and frame extractions passed. Aggregation reproduces the
baseline totals and checks complete joins. Every viewer matrix sums to its labelled
contact count, and all 94 choices passed a JavaScript rendering check. The page was
also inspected in a browser. Script syntax and scoped Ruff checks passed (exit 0).
The original-ShuttleSet work checked available records and model identity only;
no new original-dataset evaluation or footage sample was run.

## Court images for the issue

`replay_court_corners.py` reruns the ten original scene samples for video 53,
scene 334. It checks the model weights against the saved receipt, then compares
corner positions, confidence decisions and corner sources with the saved fallback.
The largest position difference was 0.019 pixels. All ten neural-net outputs had
valid shapes, but none passed every confidence check. The top-right corner remained
below the threshold. The figure shows its median position before replacement.

`results/video53_nn_replay.json.gz` retains each sampled frame's four neural-net
positions and confidence values, along with the saved and replayed fallback outputs.
`plot_court_issue.py` draws that comparison and the two saved video 17 outlines.
Numbers 0–3 follow the model's top-left, top-right, bottom-right, bottom-left order.

```bash
python -m scratch.annotator_wrapup_evaluation.scripts.replay_court_corners \
  --prepared-root "$PREPARED" \
  --sources "$SOURCES" \
  --output scratch/annotator_wrapup_evaluation/results/video53_nn_replay.json.gz
python -m scratch.annotator_wrapup_evaluation.scripts.plot_court_issue
```

The replay needs the original court model and its CUDA environment. Plotting uses
the saved replay and previously extracted source frames. Neither script changes
the detector or its saved predictions.
