# Per-class alpha over-allocation: the six-cell read

Output of the per-class arc analysis the handover set up, run over the
taxon_pinned_w_preds batch (six cells, five serials each, 80 epochs). Scripts +
full per-class tables live next to this file: `parse_arcs.py` (TB -> `arcs.pkl`,
tb-viewer venv), `analyse_arcs.py` (-> `tables.md` + `summaries.pkl`,
badminton-cicd), `plot_arcs.py` (-> the PNGs). The shuttleset_18 parse reproduces
every handover anchor exactly (driven_flight alpha 1.83 / peak 2.50@27 / val
0.000; smash 0.94; macro 0.658/0.677/0.692; best-macro epochs [19,31,36,38,62]),
so the same parser is trusted on the other five.

## Bottom line

The over-allocation thesis holds in every taxonomy. Alpha tracks inverse val
performance everywhere (alpha-vs-best-val-F1 correlation -0.91 to -0.98), and 53
to 84% of the above-mean alpha budget sits on classes that had already stopped
improving on val by the time macro plateaued. The renorm-to-mean-1 is the engine:
as the easy classes saturate their train F1 their raw alpha shrinks, and the
mean-1 renorm dumps that freed budget onto whatever has the worst train F1,
which is the plateaued / unlearnable tail. So the back half of every run spends
its budget concentrating weight on classes that don't move on val.

What changes across taxonomies is not whether this happens but what the budget
lands on, and that tracks the unlearnable tail the taxonomy keeps:

| cell | macro max | plateau | corr(a, val) | above-mean budget | of which on plateaued | floor budget | the over-allocations |
|---|---:|---:|---:|---:|---:|---:|---|
| shuttleset_18 / v2 | 0.692 | e31 | -0.91 | 3.81 | 53% | **0.83** | driven_flight (floor), wrist_smash, defensive_return_drive, drive, push |
| une_v1_14 / v2 | 0.768 | e28 | -0.95 | 3.20 | 84% | 0.00 | wrist_smash, push, drive, passive_drop |
| bst_25 / baseline | 0.820 | e31 | -0.95 | 5.67 | 65% | 0.00 | Top/Bottom drive + push |
| bst_24 / baseline | 0.829 | e31 | -0.98 | 6.24 | 62% | 0.00 | Top/Bottom drive + push |
| bst_24 / v2 | 0.845 | e26 | -0.94 | 6.33 | 76% | 0.00 | Top/Bottom drive + push, cross_court |
| bst_12 / v2 | 0.847 | e27 | -0.94 | 3.10 | 78% | 0.00 | drive, push |

(macro across taxonomies is not apples-to-apples: different class counts and val
sets. The split_bst_baseline vs split_v2 pair on bst_24 is the closest
comparison, 0.829 vs 0.845.)

## The gradient, worst to mildest

- **shuttleset_18 is the only cell with a genuine floor.** driven_flight (42
  train, 9 val) holds val F1 0.000 the whole run while its alpha peaks at 2.46
  (e27) and ends at 1.83, the largest in the model. 0.83 units of above-mean
  budget poured into a class that returns nothing. wrist_smash (val ~0.50) and
  the defensive_return_* pair add ceiling waste on top. Lowest macro (0.692).

- **une_v1_14, the production taxonomy, has exactly one bad over-allocation:
  wrist_smash.** Alpha 1.97 (highest in the cell), val peaks 0.48 at epoch 13
  then sits flat, and it is the min-F1 class. No floor (the merge folded
  driven_flight away), so 84% of the over-weight is on plateaued-but-not-dead
  classes. The rest of the over-weight is the usual drive/push/passive_drop.

- **bst_12 / bst_24 / bst_25 have no floor and no single confusion-ceiling
  class.** The over-weight lands on the per-side drive/push (and cross_court,
  rush) mid-tier, val ~0.60-0.84. Highest macro (0.82-0.847).

## Two corrections to the handover's "merged taxonomies look milder"

1. Milder on waste, not on concentration. The sided taxonomies concentrate
   alpha *harder* than shuttleset_18 in absolute terms: Top_drive hits alpha
   2.24 (bst_24 baseline) and 2.17 (v2) against shuttleset_18's max of 1.83.
   With 24-25 classes, ~16-18 of them saturate train F1 to 0.90-0.99 (smash,
   net_shot, clear, services, drop, lob, return_net per side) and dump their
   budget, and the renorm vacates all of it onto the ~6 mid classes, spiking
   them to 2x. More easy classes means a sharper spike on the few hard ones. The
   merge removes the dead floor and lifts macro, but it does not soften the
   concentration; it just aims it at improvable-mid classes instead of dead ones.

2. drive + push are over-allocated in every taxonomy. They are the consistent
   mid-tier: val 0.60-0.66, train 0.74-0.77 (gap ~0.11-0.14), alpha 1.3-2.2,
   val flat from epoch ~20-45 while alpha keeps climbing. Whatever else differs,
   these two (and their sided variants) are the universal budget sinks.

## What it says for the val-improvability gate (the handover's Q4 proposal)

The evidence supports there being budget to reclaim: 2-5 units of above-mean
alpha per cell sit on plateaued classes. But the handover's honest ceiling
stands. On shuttleset_18 the biggest single reclaim (driven_flight, 0.83) comes
off a floor class whose budget can only go to mid classes that are themselves
plateaued, so the macro gain is bounded. On the merged taxonomies the gate would
mostly be shuffling weight between drive, push and cross_court, all similar, all
half-stalled. Real but small.

The cheaper observation the curves make on their own: every cell plateaus macro
by epoch 26-31, and the back 50 epochs are pure alpha-concentration with flat
macro. Best-macro epochs run 19-74 though, and a few serials genuinely creep up
to epoch 70+ (bst_24/v2 macro 0.836 at e40 -> 0.845 at e73, about +0.009 over 33
epochs), so a hard cut to n_epochs=50 isn't free on those serials. Patience-based
early stop (~15-20) is the safer way to drop the wasted tail than a fixed cut.
Either way checkpoint selection already grabs the best epoch, so this is a
compute saving first; the macro gain, if any, comes from the gate, not the stop.

## Figures

- `arc_<cell>.png` (six): top panel val macro/min + reconstructed cosine LR;
  lower panels per-class val F1 and per-class alpha, the six highest-alpha
  classes coloured, the rest grey. The coloured classes sit flat/low on val while
  their alpha climbs above the renorm mean.
- `alpha_vs_valf1_grid.png`: final alpha vs best val F1 per class, all six cells,
  orange = plateaued by the macro-plateau epoch, teal = still improving. The
  single clearest view: alpha rises as val falls, and the high-alpha points are
  almost all plateaued.
- `macro_arcs_all_cells.png`: the six macro arcs on one axis with plateau marks.
