# Figures

These figures summarise the saved results. The source-frame links below show the evidence behind the court and label findings.

Exclusion plots are **cumulative**: all 47 historical videos, then remove known-bad video 15, then also remove video 53 for sensitivity only.

**Contents**  
[Exploring the results](#exploring-the-results)  
[Summary figures](#summary-figures)  
[Follow-up figures](#follow-up-figures)  
[Source-frame evidence](#source-frame-evidence)  
[Generate the summary figures](#generate-the-summary-figures)

## Exploring the results

These plots show the spread and kinds of errors behind the headline scores. They are embedded in the [main report](../README.md).

| Figure | What it shows | Videos |
|---|---|---:|
| [Video variation](video_variation.png) | Contact recovery against fully correct rally rate, one point per video | 47 |
| [Video outcomes, first half](video_outcome_breakdown_1.png), [second half](video_outcome_breakdown_2.png) | Correct-player matches, other matches and missed labels for every video | 47 |
| [Rally coverage](rally_coverage.png) | Complete, incomplete and unreached rallies before selection | 47 |
| [Selection](selection.png) | Correct, wrong and unknown clips kept and discarded | 47 |
| [Error combinations](selected_errors.png) | Which errors occur together in wrong selected clips | 47 |
| [Timing offsets](timing_offsets.png) | How close matched contacts are to labels; misses excluded | 46 |
| [Contact position](contact_position.png) | Serve, middle and final-contact miss rates at two tolerances | 47 |
| [Input conditions](upstream_context.png) | Court and player availability within missed and matched contacts | 46 |

The 47-video plots retain video 15's invalid label comparison; the 46-video plots exclude it. “Trusted” in older figures means cleaned labels. The [interactive video breakdown](../VIDEO_BREAKDOWN.html) adds per-video player confusion, extra predictions and input conditions; open it locally.

## Summary figures

- [`rally_correctness.png`](rally_correctness.png) — exact rally-sequence and fully-correct-rally rates across the cumulative video exclusions.
- [`contact_correctness.png`](contact_correctness.png) — contact timing and timing+player recovery across the same populations.
- [`review_queue.png`](review_queue.png) — correct, wrong and unjudgeable selected clips for each population.
- [`misses_by_input_state.png`](misses_by_input_state.png) — where the 3,633 misses outside video 15 occur in the pipeline.
- [`heuristic_vs_learned.png`](heuristic_vs_learned.png) — fully correct rally rate for the ordinary heuristic and learned output after removing video 15.

Removing a video changes the denominator. It does not repair saved output. The version without video 53 is a sensitivity view, not a better benchmark.

## Follow-up figures

- [contact_sample_results.png](contact_sample_results.png) — direct hit/player checks of 24 randomly sampled missed contacts.
- [original_comparison.png](original_comparison.png) — 32 original-ShuttleSet development videos compared with the historical 47-video ShuttleSet22 result.

These are unchanged copies of the local worklog figures, produced by `worklog/plot_followup_results.py` in the evaluation directory. The [follow-up records](../evaluation_reproduction.md#later-direct-hit-judgements) describe their evidence and reproduction. They are included here so the [main report](../README.md) can use them without the ignored worklog directory.

## Source-frame evidence

The source-frame and geometry images depend on actual video frames, court estimates and saved detector state. Use the immutable repository copies rather than redrawing them from published numbers:

- [Video 15 label/footage disagreement](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/label_alignment.png)
- [Video 15 strong timing matches still showing the wrong rally](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/video15_best_matches.png)
- [Video 53 alignment checks](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/video53_alignment_checks.png)
- [Rejected versus accepted court example](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/court_example.png)
- [Video 17 shared versus scene court geometry](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/player_geometry.png)
- [Video 53 neural-net corners versus OpenCV fallback](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/video53_nn_to_opencv.png)
- [Video 17 neural-net outline versus shared outline](https://github.com/ahalp90/badminton_cv_annotator/blob/8a8562e26a9286ad491c3935f3860db66b91b020/scratch/annotator_wrapup_evaluation/figures/video17_nn_to_shared.png)

The two numbered court images are also embedded in [issue #148](https://github.com/ahalp90/badminton_cv_annotator/issues/148).

## Generate the summary figures

From the directory containing the reports, run:

```bash
python figures/generate_published_figures.py
```

The script reads `results/exclusion_metrics.csv.gz` in the enclosing evaluation directory and writes the five PNGs beside itself. It uses the saved counts for cleaned labels at ±10 frames.
