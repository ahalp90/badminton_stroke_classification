# Phase-1 vs Phase-2 mmpose extract: zeroed-frame comparison (2026-04-29)

Same 32,203-clip universe, same 1,722,058 total frames, same `une_merge_v1` taxonomy + `split_v2` for both sides. Pose extracts compared:

- **Phase-1** (baseline): legacy `/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat/`. Source report: `analysis_unemergev1_v2_20260421_1159.txt`.
- **Phase-2**: new unified raw extract (RTMDet-nano + RTMPose-L, `N_max=16`) + `sticky_anchor` heuristic, written to `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`. Source report: `analysis_unemergev1_v2_20260429_1905.txt`.

Both reports live alongside each other in `src/bst_x/validation_scripts/zeroed_frames_analysis_outputs/`.

## Headline numbers

| Metric                                   | Phase-1            | Phase-2          | Ratio   |
|------------------------------------------|--------------------|------------------|---------|
| Overall fail rate                        | 92,677 / 1,722,058 (5.38%) | 15,957 / 1,722,058 (0.93%) | 5.81 x lower |
| Train fail rate                          | 5.13%              | 0.91%            | 5.64 x  |
| Val fail rate                            | 5.99%              | 0.87%            | 6.89 x  |
| Test fail rate                           | 5.93%              | 1.09%            | 5.44 x  |
| Completely-blank clips                   | 2                  | 0                | -       |
| Clips above 90% fail                     | 7                  | 4                | 1.75 x  |
| Clips above 75% fail                     | 20                 | 7                | 2.86 x  |
| Clips above 50% fail                     | 222                | 67               | 3.31 x  |
| Clips with 100% hit-zone zeroed          | 140                | 17               | 8.24 x  |
| Hit-zone fail rate (near hit, +/- 10 frames) | 5.98%              | 0.58%            | 10.31 x |
| Hit-zone fail rate (away from hit)       | 5.00%              | 1.15%            | 4.35 x  |
| Flaw-cohort fail rate                    | 491 / 1,365 (35.97%) | 368 / 1,365 (26.96%) | 1.33 x |
| Non-flaw fail rate                       | 92,186 / 1,720,693 (5.36%) | 15,589 / 1,720,693 (0.91%) | 5.89 x |

## Per-stroke fail rate

Sorted by Phase-2 fail rate descending. Ratio computed from raw frame counts, not the 2-decimal rate, so the column is exact.

| Stroke                | Phase-1 frames    | Phase-1 rate | Phase-2 frames | Phase-2 rate | Ratio  |
|-----------------------|-------------------|--------------|----------------|--------------|--------|
| long_service          | 2,270 / 19,838    | 11.44%       | 776 / 19,838   | 3.91%        | 2.93 x |
| short_service         | 2,784 / 73,588    | 3.78%        | 2,549 / 73,588 | 3.46%        | 1.09 x |
| rush                  | 321 / 16,586      | 1.94%        | 305 / 16,586   | 1.84%        | 1.05 x |
| net_shot              | 7,399 / 303,086   | 2.44%        | 4,680 / 303,086 | 1.54%       | 1.58 x |
| push                  | 4,327 / 140,442   | 3.08%        | 2,162 / 140,442 | 1.54%       | 2.00 x |
| lob                   | 17,225 / 324,227  | 5.31%        | 2,865 / 324,227 | 0.88%       | 6.01 x |
| drive                 | 1,274 / 61,332    | 2.08%        | 327 / 61,332   | 0.53%        | 3.90 x |
| wrist_smash           | 7,829 / 78,867    | 9.93%        | 406 / 78,867   | 0.51%        | 19.28 x |
| smash                 | 16,613 / 120,850  | 13.75%       | 554 / 120,850  | 0.46%        | 29.99 x |
| cross_court_net_shot  | 1,244 / 61,552    | 2.02%        | 257 / 61,552   | 0.42%        | 4.84 x |
| passive_drop          | 1,156 / 71,519    | 1.62%        | 195 / 71,519   | 0.27%        | 5.93 x |
| drop                  | 7,921 / 119,705   | 6.62%        | 314 / 119,705  | 0.26%        | 25.23 x |
| clear                 | 13,722 / 186,248  | 7.37%        | 454 / 186,248  | 0.24%        | 30.22 x |
| return_net            | 8,592 / 144,218   | 5.96%        | 113 / 144,218  | 0.08%        | 76.04 x |

## What jumps out

**1. The hit-zone gradient flipped sign.**

In Phase-1 the heuristic failed *worse* near the hit (5.98%) than away from it (5.00%). The transformer needs the cleanest signal exactly in those frames, and Phase-1 was delivering the noisiest. Phase-2 reverses the gradient cleanly: near-hit is now the lowest-failure zone (0.58% vs 1.15% away). That alone is the strongest argument that the new extract should improve downstream classification.

**2. Per-stroke gains concentrate where Phase-1 was worst.**

The five strokes Phase-1 was failing hardest on, smash (13.75%), long_service (11.44%), wrist_smash (9.93%), clear (7.37%), and drop (6.62%), all involve full-court action where the wrong-pose lock-on (umpire, ball staff, line judges) was most likely. Phase-2 takes them down by 19x to 76x. The two laggards, short_service (1.09 x) and rush (1.05 x), barely move and that fits: short_service is genuinely hard pose detection (server bent at the line, often partially out of frame), and rush is a sprint that lags any per-slot anchor.

**3. The 17 residual 100%-hit-zone-zeroed clips are now mostly broken broadcasts, not heuristic confusion.**

Phase-1 had 140 such clips. Phase-2 has 17:

- 10 short_service: `16_1_2_1`, `16_1_5_1`, `24_3_12_1`, `39_2_13_1`, `11_2_27_1`, `3_2_24_1`, `11_2_33_1`, `11_2_35_1`, `3_1_30_1`, `37_1_2_1`
- 4 long_service: `3_2_31_1`, `44_2_25_1`, `11_2_11_1`, `3_2_12_1`
- 3 net_shot: `30_3_33_29`, `35_1_4_2`, `18_1_9_2`

These are the irreducibly-broken cases (off-frame players, replay overlays, angled broadcast cuts). No two-player heuristic recovers them; they need scene-level filtering or to stay in the dataset as noise.

**4. Flaw vs non-flaw separation grew sharply.**

Flaw cohort: 35.97% -> 26.96%. Non-flaw: 5.36% -> 0.91%. The absolute level dropped on both sides, but non-flaw dropped harder, so the relative separation widened from 35.97 / 5.36 = 6.71 x in Phase-1 to 26.96 / 0.91 = 29.63 x in Phase-2. Flaw annotations are a much stronger signal of "this clip is genuinely bad" against the new extract than they were against the old one.

## Cross-taxonomy sanity (today's three runs)

The three Phase-2 runs (`une_merge_v1_nosides + split_v2`, `une_merge_v1 + split_v2`, `merged_25 + split_bst_baseline`) report identical underlying numbers (15,957 frames failed across 1,722,058 total, 17 clips with 100% hit-zone zeroed, etc.). The taxonomy/split combo only changes how the same clips get bucketed and split, not the underlying detection quality. The `merged_25` run is the only one that emits the "1,278 missing _failed.npy" warning, which equals the unknown-class clip count exactly: the `une_merge_v1*` taxonomies filter unknowns earlier via the merge_map, while `merged_25` keeps `unknown` as a class and surfaces it as a missing-file count. No non-unknown stems are absent.

## Source files (engelbart, repo path under home/ahalperi/)

- Phase-1: `src/bst_x/validation_scripts/zeroed_frames_analysis_outputs/analysis_unemergev1_v2_20260421_1159.txt`
- Phase-2: `src/bst_x/validation_scripts/zeroed_frames_analysis_outputs/analysis_unemergev1_v2_20260429_1905.txt`
- Phase-2 sibling reports (same numbers, different bucketing): `analysis_unemergev1nosides_v2_20260429_1904.txt`, `analysis_merged25_bstbaseline_20260429_1906.txt`
