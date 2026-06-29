# Class-by-player split overlap exploration

This sits next to the train-vs-val-vs-test analysis. The idea is to test whether the val-test gap we saw on confusion classes (smash, ws) is partly a **player-identity** story: same player appears in both splits, model memorises their style, the win on val doesn't transfer to test players it hasn't seen.

## Setup

- Data: `notebooks/clips_master.csv`, 33,481 raw clips across 40 matches.
- Taxonomy: `une_merge_v1_nosides` (no Top_/Bottom_ prefix), `split_v2`, `drop_unknown=True`. Drops `unknown` and folds `defensive_return_lob`, `driven_flight`, `back_court_drive`, `defensive_return_drive` into their merge targets via `UNE_MERGE_V1_MAP`.
- After taxonomy + drop unknown: **32,203 clips** (matches the unified extract size that bst_x_train operates on across Phase 2).
- Player resolution: `(vid, set, rally, player_side)` mapped to a player name. Convention from `pipeline/player_mapping.py`: A is the winner, B is the loser; sides swap between sets 1 and 2; set 3 has a mid-set switch at 11 points (split rally pulled per match from `set3.csv`).

## Target classes (median top-3 + bottom-4 by test F1 across the nine Phase 2 nosides runs)

Median per-class test F1 over the runs in `nosides_runs_table.md`:

| Rank | Class | Median test F1 |
|---|---|---|
| Top 1 | short_service | 0.978 |
| Top 2 | long_service | 0.965 |
| Top 3 | clear | 0.950 |
| ... | ... | ... |
| Bottom 4 | push | 0.643 |
| Bottom 3 | smash | 0.610 |
| Bottom 2 | drive | 0.609 |
| Bottom 1 | wrist_smash | 0.421 |

Rank `1` is always the extreme of its cluster: Top 1 = best class overall, Bottom 1 = absolute worst. So the seven target classes (worst to best) are: **wrist_smash, drive, smash, push, clear, long_service, short_service**.

The top-3 are the pose-distinctive ceiling classes; the bottom-4 are the bottleneck cluster, with wrist_smash leading the bottom.

## Filter levels

Three filter levels matter here, increasingly close to what bst_x_train actually sees: raw clips_master, then the ShuttleSet annotator's per-shot quality flag, then the pose-extraction discard.

- **(a) Raw**: 32,203 clips.

- **(b) Flaw filter**: the per-shot `flaw` column in each match's `set1.csv` / `set2.csv` / `set3.csv`. NaN for clean shots, 1 for shots the original ShuttleSet annotators flagged for some quality issue but kept in the dataset.

  Total flaw flags across all set CSVs: 1441 (3.95% of 36,482). Of those, 127 are in the four fully-excluded vids (9, 10, 12, 27), 1314 are in kept vids, 1311 land in `clips_master.csv`.

  Crucially, **1275 of those 1311 (97%) have raw_type_en = `unknown`**, so the dropunk filter already removes them. Only **36 flaw-flagged clips survive nosides + dropunk** (0.11% of 32,203). 
  
  Distribution: short_service 17, lob 5, long_service 4, driven_flight 3 (folds to drive), net_shot 2, push 2, plus singles in return_net / drop / rush.

  Effect on overlap: gross figures change by less than 0.01 pp at four decimal places. Basically irrelevant to the following analysis.

- **(c) bst_x_train discard (videos_len=0)**: clips that MMPose failed on every frame of, dropped at npy-collation time. **17 clips of 32,203 (0.05%) have videos_len = 0**: 9 train, 0 val, 8 test. Smaller than the project memory's ndet=1 floor estimate of ~0.5% (which was at the per-frame level, not per-clip). Effect on overlap is essentially nil: Jaccard is identical to raw, clip-weighted shifts by at most 0.0001.

## Overlap definitions

For any given cls and split pair (A, B):

- `players_A` = unique player names with at least one clip of cls in split A.
- `players_B` = same for B.
- `common` = players who appear in both A and B.
- `total_unique` = distinct players appearing in either A or B (no double-counting).
- `clips_A`, `clips_B` = clip counts of cls in each split.
- **Clip-weighted overlap**: of all the cls clips in splits A and B combined, the fraction that come from common players. So if a player appears in both splits and has 100 clips in train + 20 in val, all 120 count toward the numerator. The denominator is every cls clip in A or B regardless of player.
- **Jaccard (player set)** = `common / total_unique`. Of the distinct players seen across both splits, the fraction that show up on both sides. Jaccard = 0 means fully disjoint player pools, Jaccard = 1 means identical player pools.

The clip-weighted figure answers "how much of the data has player-identity overlap?"; the Jaccard answers "how many distinct players are double-counted?". Clip-weighted is usually higher than Jaccard because heavy-clip players (long match runs) inflate the weighted figure.

## Per-class table

Numbers below are at filter level (a). Filter (b) and (c) values match within 0.0001 across the table (see the appendix at the bottom for the three-way side-by-side).

Rows ordered by median test F1, worst at top: bottom-4 cluster above the divider (wrist_smash leading), top-3 ceiling cluster below for contrast. Reads top-down as "the classes we can't classify well" then "the classes we can".

| Class | Pair | clips_A | clips_B | players_A | players_B | common | total_unique | clip-weighted | Jaccard |
|---|---|---|---|---|---|---|---|---|---|
| wrist_smash | train-val | 979 | 331 | 17 | 11 | 8 | 20 | **0.549** | 0.400 |
| wrist_smash | train-test | 979 | 249 | 17 | 9 | 2 | 24 | 0.198 | 0.083 |
| wrist_smash | val-test | 331 | 249 | 11 | 9 | 1 | 19 | 0.259 | 0.053 |
| drive | train-val | 1042 | 226 | 17 | 12 | 9 | 20 | **0.530** | 0.450 |
| drive | train-test | 1042 | 255 | 17 | 9 | 2 | 24 | 0.153 | 0.083 |
| drive | val-test | 226 | 255 | 12 | 9 | 1 | 20 | 0.179 | 0.050 |
| smash | train-val | 1786 | 299 | 17 | 12 | 9 | 20 | **0.547** | 0.450 |
| smash | train-test | 1786 | 277 | 17 | 9 | 2 | 24 | 0.144 | 0.083 |
| smash | val-test | 299 | 277 | 12 | 9 | 1 | 20 | 0.174 | 0.050 |
| push | train-val | 1883 | 389 | 17 | 12 | 9 | 20 | **0.476** | 0.450 |
| push | train-test | 1883 | 380 | 17 | 9 | 2 | 24 | 0.125 | 0.083 |
| push | val-test | 389 | 380 | 12 | 9 | 1 | 20 | 0.192 | 0.050 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clear | train-val | 1897 | 382 | 17 | 12 | 9 | 20 | **0.490** | 0.450 |
| clear | train-test | 1897 | 382 | 17 | 9 | 2 | 24 | 0.108 | 0.083 |
| clear | val-test | 382 | 382 | 12 | 9 | 1 | 20 | 0.090 | 0.050 |
| long_service | train-val | 252 | 33 | 16 | 7 | 5 | 18 | **0.316** | 0.278 |
| long_service | train-test | 252 | 74 | 16 | 6 | 2 | 20 | 0.239 | 0.100 |
| long_service | val-test | 33 | 74 | 7 | 6 | 1 | 16 | 0.131 | 0.062 |
| short_service | train-val | 1312 | 291 | 17 | 12 | 9 | 20 | **0.583** | 0.450 |
| short_service | train-test | 1312 | 255 | 17 | 8 | 2 | 23 | 0.160 | 0.087 |
| short_service | val-test | 291 | 255 | 12 | 8 | 1 | 19 | 0.233 | 0.053 |

Bold marks the train-val clip-weighted column (the headline figure: how much of the cls-clips in train + val sit with shared players). The pattern would be alarming if it lined up with class difficulty, but it doesn't: short_service (top of the test-F1 ceiling at 0.978) and wrist_smash (bottom at 0.421) hit nearly identical train-val overlap (58.3% and 54.9%). Whatever drives split overlap is class-agnostic.

## Charts

### Per-class clip-weighted overlap (all three filters)

![overlap_clip_weighted](charts/overlap_clip_weighted.png)

The pattern is striking and uniform across classes: train-val sits ~50% across all seven targets, train-test sits ~10-25%, val-test sits ~10-26%. There's no class-specific signal; the overlap is split-level rather than class-level. Whatever causes the train-val similarity acts at the player level, not at the stroke-type level. The middle (flaw-filtered) and right (bst-filtered) panels are visually identical to raw because the dropped clip counts are tiny.

### Per-class Jaccard (all three filters)

![overlap_jaccard_players](charts/overlap_jaccard_players.png)

Jaccard is much lower than clip-weighted in absolute terms (most pairs at 5-10%, train-val at 28-45%). The gap between the two metrics is the mark of heavy-clip players: a few players with hundreds of clips on each side dominate the clip-weighted figure even when most unique players are split-exclusive.

### Gross overlap (all 14 classes pooled, all three filters)

![gross_overlap](charts/gross_overlap.png)

### Clip counts per class per split

![class_counts_per_split](charts/class_counts_per_split.png)

Top-3 classes labelled in blue, bottom-4 in reddish-purple (CB-safe palette, both bold). Heavy class-imbalance is visible: net_shot, lob, push, smash, return_net all 5x+ wrist_smash. The bottom-4 also tend to have smaller absolute counts, which compounds the generalisation problem. Rare class + small per-split sample = hard to learn a transferable representation.

## Reading the picture

The interesting finding isn't per-class. It's split-level:

- **train-val** has ~55% of clips owned by the 9 players who sit on both sides (out of 20 unique in the union). So the val set isn't a held-out player population; nearly half the val clips are by players the model also saw in train.
- **train-test** drops to ~15% clip-weighted, 8% Jaccard. The test set is mostly held-out by player (only 2 players appear in both train and test).
- **val-test** is the lowest at ~5% Jaccard. Almost no player overlap; only 1 common player out of 20.

This is consistent with how `split_v2` was constructed (per `notebooks/02_revised_splits.ipynb`): val and train were drawn so they cover similar tournaments and similar player pools, while test was held out as a stricter generalisation set. The split was deliberately designed this way (val for early-stop / model selection, test for unbiased evaluation), and the data confirms it.

What this implies for our prior train-vs-val-vs-test analysis:

- The 1-3 pp val-vs-train gap at best epoch is on a val set where 55% of clips are by players the model saw. Some of that "good val" is player memorisation, not representation generalisation. The val-best-epoch is partially fitted to known players.
- The 3-9 pp val-vs-test gap reflects the genuine player-distribution shift. Test is asking the model "can you classify a stranger's stroke?" while val isn't pushing as hard on that.
- Pair confusion (smash vs wrist_smash) is uniform across the table at the per-class overlap level. That's not where the asymmetry lives. The asymmetry is purely in the split design: val is in-distribution, test is out-of-distribution, both at the player level.

Three follow-ups this points at:

- **Just swap val and test (cheap experiment).** Keep `split_v2` as is, but flip the roles: early-stop on the current test set (player-disjoint, the tougher signal), report final on the current val set (in-distribution). No re-splitting work, single training run. Predictions: early-stop fires later (the harder signal keeps moving longer), best-checkpoint weights end up tuned for cross-player generalisation rather than within-player fit, and the held-out generalisation score (ex-val, in-distribution) probably climbs above the current test number purely from the looser distribution. Need to be careful about what the headline number then represents: reporting on the easier split would overstate generalisation, so the comparison would have to be like-for-like (e.g. report both on the original test set, just changing which one drove early-stop).
- **Player-aware splits (bigger build).** Construct a held-out-player val (similar to test) from scratch. Closes the train-val gap mechanically and pushes val macro toward test macro. The model selection signal then rides player-disjoint generalisation. Heavier than the swap above because it touches `notebooks/02_revised_splits.ipynb` and breaks comparability with everything in `nosides_runs_table.md`.
- **The macro plateau hypothesis sharpens.** From the train/val/test analysis, the bottleneck on confusion classes was train-test divergence, not pure capacity. This analysis says some of that divergence is player-identity. So the gap on smash and ws is partly "model memorises this player's specific smash motion in train, doesn't generalise to a stranger's smash" rather than "model fundamentally can't separate the two stroke classes". A two-stage cRT test (Kang et al.) won't fix that; it's a representation-side issue. Augmentation that perturbs player-specific cues (skeleton scale, arm-length normalisation, body-relative jitter) might.

## Methodology notes

- Player name resolution mirrors `pipeline/player_mapping.py`. For sets 1 and 2 it's deterministic from `downcourt`. For set 3, the script reads each match's `set3.csv` and finds the first rally where either score reaches 11; the rally after that is the switch point. Clips with rally < switch are pre-switch (set-1 mapping), rally >= switch are post-switch (set-2 mapping). When neither player reaches 11 (rare retirements), all set-3 rallies stay pre-switch.
- A=winner, B=loser convention from BST-original `ShuttleSet/get_each_class_total.py:8` (`'''A is the winner and B is the loser.'''`).
- Filter (b) reads the per-shot `flaw` column from each match's `set1.csv` / `set2.csv` / `set3.csv`. Shots with `flaw == 1` (NaN-or-1 column) are dropped at this filter. **Note**: this is the per-shot annotator flag, separate from `flaw_shot_records.csv` (the removal log), which was already applied during `notebooks/03_build_clips_master.ipynb` and so doesn't show up here.
- Scripts: `docs/architecture_notes/player_overlap_analysis.py` (loader + metrics + charts). Output JSON: `/tmp/player_overlap_results.json`.
- Filter (c) source: `scratch/research/dump_videos_len.py` runs on bourbaki against the npy collated dirs and writes `docs/architecture_notes/discard_flags_split_v2_dropunk_nosides.csv`. Re-run if the npy collation regenerates.

## Appendix: pooled overlap by filter level

Pooled across all 14 classes, this is what shifts (or doesn't) when each filter is applied. Confirms the per-class table's claim that filter (b) and (c) are visually no-ops on the overlap figures.

Filter (a) raw:

| Pair | clips_A | clips_B | common players | clip-weighted |
|---|---|---|---|---|
| train-val | 22,743 | 5,250 | 9 | **0.549** |
| train-test | 22,743 | 4,210 | 2 | 0.153 |
| val-test | 5,250 | 4,210 | 1 | 0.206 |

Filter (b) flaw-filtered:

| Pair | clips_A | clips_B | common players | clip-weighted |
|---|---|---|---|---|
| train-val | 22,712 | 5,248 | 9 | 0.549 |
| train-test | 22,712 | 4,207 | 2 | 0.154 |
| val-test | 5,248 | 4,207 | 1 | 0.207 |

Filter (c) bst-train-filtered (videos_len > 0):

| Pair | clips_A | clips_B | common players | clip-weighted |
|---|---|---|---|---|
| train-val | 22,734 | 5,250 | 9 | 0.549 |
| train-test | 22,734 | 4,202 | 2 | 0.153 |
| val-test | 5,250 | 4,202 | 1 | 0.206 |

Filters (b) and (c) both shift clip-weighted by at most ±0.0001 vs raw. The 36 flaw-flagged clips and 17 zero-pose clips don't remove any unique player from any split.

## Reproducing this analysis

Two scripts: one runs locally, one runs on bourbaki to regenerate the videos_len discard flags.

**1. Pull videos_len discard flags from bourbaki (only needed if the npy collation has been re-run):**

```bash
# push the dump script to bourbaki
boursync -avR scratch/research/dump_videos_len.py bourbaki-hpc:badminton_stroke_classification/

# on bourbaki: produce the CSV against the live npy collated dirs
ssh bourbaki-hpc
cd badminton_stroke_classification
source venv-bst-x/bin/activate
python scratch/research/dump_videos_len.py --trust-clip-count

# pull the result back
exit
boursync -avR bourbaki-hpc:badminton_stroke_classification/docs/architecture_notes/discard_flags_split_v2_dropunk_nosides.csv scratch/research/
```

The `--trust-clip-count` flag skips the existence check on the flat per-clip dir (which is pruned post-collation on bourbaki); the clip-count match between filtered `clips_master.csv` and `videos_len.npy` is sufficient verification.

**2. Run the overlap analysis locally:**

```bash
python docs/architecture_notes/player_overlap_analysis.py
```

Inputs read:
- `notebooks/clips_master.csv` (33,481 raw clips)
- `data/shuttleset/set/match.csv` (winner / loser / downcourt per match)
- `data/shuttleset/set/<vid>/set{1,2,3}.csv` (per-shot rally + flaw flags)
- `docs/architecture_notes/discard_flags_split_v2_dropunk_nosides.csv` (filter c)

Outputs written:
- `docs/architecture_notes/charts/{overlap_clip_weighted,overlap_jaccard_players,gross_overlap,class_counts_per_split}.png`
- `/tmp/player_overlap_results.json` (full numeric dump for the tables above)

Re-run the analysis script after either: a fresh npy collation (which changes filter c), or a taxonomy / split change (which changes the class set and clip routing).
