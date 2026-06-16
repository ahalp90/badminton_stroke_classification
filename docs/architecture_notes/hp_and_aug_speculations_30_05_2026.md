# HP and aug speculations, 30 May 2026

Four open questions, thought through against the live pipeline and the actual per-epoch curves of the running `shuttleset_18` cell, plus a few papers. Each section states what the code does, the numbers, what the literature adds, and a recommendation. These are proposals for discussion, not landed decisions, and nothing here is written into `bst_x_overview.md`.

Grounding:
- Pipeline mechanics from `bst_x_train.py`, `shuttleset_dataset.py`, `loss/adaptive_focal.py`, `augmentation_framework.md`.
- Per-epoch arcs (train F1, val F1, alpha, macro) parsed from `run_20260530_161525_131279` (the `shuttleset_18` cell, 5 serials, full 80-epoch TB logs). 14-class numbers from `run_20260501_164658`.
- Papers: Wang & Aitchison 2025 (AdamW WD scaling), Steiner et al 2022 (how to train your ViT), Touvron et al 2022 (three things about ViTs), plus Kang et al 2020, Lin et al 2017, Chen et al 2018 (GradNorm).

Note on taxonomy: the `shuttleset_18` cell is the raw, unmerged ShuttleSet taxonomy. It splits out classes the 14-class `une_merge_v1` deliberately merges (driven_flight, the defensive_return_* pair, back_court_drive). The tail dynamics below are specific to that split taxonomy and do not all carry over to the merged one.

---

## Q1. Are augs re-randomised per batch/epoch?

**Yes, confirmed. Fresh draws every batch, every epoch.** No action.

Augs run in the train loop after the batch hits device (`bst_x_train.py:228-231`); both `CoupledFlip` and `ConstrainedJitter` draw a fresh `torch.rand(n, device=device)` per call. No per-sample seed, no `worker_init_fn` (num_workers=0), no `set_epoch`. A clip gets a different flip/jitter each time it's drawn across the 80 epochs, which is what you want.

Minor: no fixed aug seed means the aug stream isn't bit-reproducible run to run. With 5 serials that's a feature (it feeds the serial spread), so leave it.

---

## Q2. AdamW weight decay

### What the code does now

```python
optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)   # bst_x_train.py:519
```

`weight_decay` is unset, so it sits at PyTorch's default **0.01**, applied to every parameter (LayerNorm gains, biases, embeddings included), not a `Hyp` field, never swept.

### How under-regularised, quantified

Wang & Aitchison (2025) reframe AdamW's weights as an EMA of recent updates with a timescale measured in epochs:

```
tau_epoch = 1 / (lambda * eta_peak * M),   M = N / B   (iterations per epoch)
```

`tau_epoch` is how many epochs of past updates the weight-EMA effectively averages over. Their headline result: the optimal `tau_epoch` is roughly constant across dataset and model size, and sits between 1 and the total epoch count. So optimal `lambda` scales as 1/dataset-size (small data wants more decay) and rises with model width.

For this run (N=22,743, B=128 so M≈178, eta_peak=5e-4), `tau_epoch = 11.25 / lambda`:

| lambda | tau_epoch (epochs) |
|---:|---:|
| 0.01 (current) | ~1,125 |
| 0.05 | ~225 |
| 0.1 | ~112 |
| 0.2 | ~56 |
| 0.4 | ~28 |

The run is 80 epochs. **At the current 0.01, tau_epoch is ~14x the whole run** — the weight-EMA averages over far more updates than you ever take, so decay is effectively off. That's the quantitative version of "0.01 is under-regularised". And because the cosine schedule drops eta toward 0, the run-averaged tau is even larger than these peak-eta figures, so the real regularisation is softer still.

### Sweep range (revised up)

To land `tau_epoch` inside the run length, sweep **lambda in {0.05, 0.1, 0.2, 0.4}** (tau_epoch ≈ {225, 112, 56, 28}). Expect the useful zone around **0.1-0.3**. This is materially higher than a naive 0.05; the small dataset plus low LR plus few iterations/epoch all push optimal lambda up. Watch from-scratch stability at the 0.4 end (loss spikes, training stalling); back off if it destabilises. Treat the timescale framing as a strong prior, not a guarantee: it's validated on ResNet/ViT/GPT, not a pose-transformer, so sweep to confirm.

If you later widen the transformer (capacity Run 2), optimal lambda goes up again with width, so re-sweep rather than carrying the value over.

### Add a no-decay group (do it regardless of the sweep)

Exclude every normalization param, every bias, and the learned tokens/positional embeddings. Wang & Aitchison don't decay normalization layers; it's standard transformer practice. At 0.01 the harm is small, but at lambda 0.1-0.4 decaying an LN gain pulls the normalisation scale toward zero, and decaying a sinusoidally-seeded positional embedding erodes the positional signal, so the split matters more at the higher WD this run wants.

The no-decay set for this model (BST_CG_AP, instantiated and enumerated, 82 param tensors):
- 10 LayerNorm (weight+bias), all ndim=1.
- 4 BatchNorm1d in the two TCNs (`tcn_pose.net.{1,5}`, `tcn_shuttle.net.{1,5}`, weight+bias). There is no BatchNorm in the MLP (`tempose.MLP` is Linear-GELU-Dropout-Linear), so nothing to exclude there.
- all biases (MLP/PPF Linear + the 4 TCN Conv1d biases; attention to_q/to_kv/to_qkv are bias=False).
- the 5 raw params a shape rule misses: `learned_token_tem`, `learned_token_inter` (ndim=2 CLS tokens) and `embedding_tem`, `embedding_cross`, `embedding_inter` (ndim=3 learned positional embeddings). They're trained, so `ndim<=1` alone would wrongly decay them.

`self.scale` (attention 1/sqrt(d_head)) and the cg/ap schedule factors are plain floats/buffers, not parameters, so AdamW never touches them.

```python
NO_DECAY_NAME_HINTS = ('embedding_', 'learned_token_')   # ndim>=2 pos-emb + cls tokens

decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    norm_or_bias = param.ndim <= 1                       # LN/BN gain+beta, all biases
    token_or_posemb = any(hint in name for hint in NO_DECAY_NAME_HINTS)
    (no_decay if norm_or_bias or token_or_posemb else decay).append(param)
assert len(decay) + len(no_decay) == sum(1 for _ in model.parameters())  # 27 / 55 here
optimizer = optim.AdamW(
    [{'params': decay, 'weight_decay': hyp.weight_decay},
     {'params': no_decay, 'weight_decay': 0.0}],
    lr=hyp.lr,
)
```

String-free alternative: decay only `Linear.weight` and `Conv1d.weight` (match owner type via `named_modules()`), everything else no-decay. Same 27/55 split. Of the 27 decayed, 17 are transformer-internal (attention + FFN weight matrices across encoder_tem/cross_trans/encoder_inter); the other 10 are the TCN convs, PPF, mlp_clean, and the head.

### Interaction with the scheduler and with augmentation

- Scheduler: PyTorch scales the decay step by LR (`theta -= lr*wd*theta`), so as cosine drives LR to 0 the decay tapers with it. Regularisation is front-loaded, which is the right direction. Don't co-tune WD and the schedule; hold the schedule fixed while sweeping lambda, or attribution is impossible.
- Augmentation: Steiner et al find augmentation and weight decay trade off ("increasing AugReg may need a decrease in weight decay"), and for small-to-mid data augmentation generally helps more than regularisation. You already have flip+jitter doing real work. So the WD sweep is the regularisation half of a recipe whose augmentation half is mostly in place; don't expect WD alone to carry it.

### Expectation

A real and likely useful gain given how inert decay currently is, plus a near-free principled fix in the no-decay split. Not the lever that breaks the macro ceiling (Q3/Q4).

---

## Q3. Oversampling the rare classes

On the 14-class merged taxonomy I argued against this, because there count and difficulty were decoupled (smash, the 4th-largest class, was a bottleneck; rush at 335 clips was already fine). On `shuttleset_18` the picture is different and your low-sample-drag claim is right: the bottom of the table is the genuinely tiny split classes. But oversampling still isn't the fix, and the curves show why.

### Where the budget goes vs what it returns (shuttleset_18, serial_1, sorted by final alpha)

| Class | n_train | val F1 (max) | final alpha |
|---|---:|---:|---:|
| driven_flight | 42 | 0.00 (test 0.00) | **1.84** |
| wrist_smash | 979 | 0.59 | 1.68 |
| defensive_return_drive | 270 | 0.52 | 1.66 |
| drive | 467 | 0.67 | 1.40 |
| push | 1,883 | 0.64 | 1.31 |
| passive_drop | 796 | 0.66 | 1.30 |
| back_court_drive | 263 | 0.64 | 1.28 |
| cross_court_net_shot | 847 | 0.82 | 1.19 |
| defensive_return_lob | 200 | 0.61 | 1.13 |
| drop | 1,465 | 0.72 | 1.03 |
| smash | 1,786 | 0.69 | 0.94 |
| return_net | 2,392 | 0.86 | 0.79 |
| rush | 335 | 0.79 | 0.76 |
| lob | 3,418 | 0.84 | 0.73 |
| net_shot | 4,139 | 0.93 | 0.49 |
| long_service | 252 | 0.99 | 0.22 |
| clear | 1,897 | 0.97 | 0.16 |
| short_service | 1,312 | 1.00 | 0.10 |

The worst three F1s are the tiny split classes (driven_flight 42, defensive_return_drive 270, defensive_return_lob 200). So yes, sample count drags the tail here.

### Why oversampling still won't fix it

- **driven_flight (42 clips) is unrecoverable by repetition.** Its train F1 climbs to ~0.57-0.74 (the model memorises the 42 clips) while its val F1 stays flat at 0.000 the entire run. Oversampling repeats the same 42 clips; it adds no new information, so val F1 won't move. Steiner et al make exactly this point: AugReg substitutes for data only down to a floor, and below it (their example: 10% of ImageNet) no amount of augmentation recovers the full-data result. 42 clips is well under that floor.
- **The defensive_return_* splits and back_court_drive are confusable with their parents** (drive, lob). That's a representation problem, and Kang et al show resampling helps the classifier head but not the representation. Mild upside at best, with overfitting and false-positive bleed into the parent class as the downside.
- **Oversampling fights the adaptive alpha.** Repeating a class makes the model memorise it faster, which raises its train F1, which *drops* its alpha (the loss thinks it's solved). So oversampling and the F1-driven alpha partially cancel. Double-balancing, with the two corrections working against each other.

### Weighting and oversampling are not the same lever

Increased alpha and oversampling are not interchangeable, and alpha is the worse of the two for a struggling class. For plain SGD on the summed loss they're first-order equivalent (oversampling by k and alpha=k give the same expected per-epoch gradient from the class). They diverge on four things, all favouring oversampling for an under-fit class:

1. Distinct views: with per-draw aug, oversampling shows k different augmented views; alpha amplifies the gradient on the views already seen. (Bounded by aug strength: flip + 5-10% jitter on a skeleton is a modest diversity engine.)
2. Gradient variance: averaging k augmented draws lowers the variance of the class's gradient; scaling one draw by k does not.
3. Per-example magnitude: alpha inflates each example's gradient into a spiky kick that can dominate the batch it lands in; oversampling keeps per-example magnitude normal and raises frequency (distributed, gentler under momentum).
4. Optimizer: AdamW normalises by per-parameter gradient RMS, so scaling a class's loss by alpha does not scale its effective update proportionally; the alpha-to-learning relationship is sublinear and muted on that class's parameters. The `adaptive_focal` renorm-to-mean-1 (commented as keeping "AdamW's effective per-parameter LR comparable") manages exactly this. Oversampling changes frequency, which Adam's EMAs absorb differently. So a large alpha buys less effective learning than its number implies, and meanwhile amplifies any harmful (neighbour-suppressing, mislabelled) gradient linearly.

The boundary that matters: all four are about variance, coverage and optimizer dynamics, not bias. Augmentation denoises view-variance; it does not fix a wrong label or manufacture information the inputs lack. So neither tool helps an input ceiling or a data floor, and the train-val gap tells you which you're facing:

- driven_flight: train F1 ~0.74, val F1 0.000 the whole run. The gap is the memorisation signature, 42 clips fit but don't span val. Oversampling memorises them faster (which would even drop its adaptive alpha) but val stays 0. Data floor.
- wrist_smash: train ~0.67, val ~0.50, confusable with smash. Input ceiling.
- moderate tail (defensive_return_drive train ~0.70 / val ~0.52, back_court_drive, drive): smaller gaps, some real headroom from coverage rather than confusion.

### Recommendation

Prefer mild oversampling over higher alpha for the moderate-learnable rare classes (the defensive_return_drive / back_court_drive / drive band), where the difficulty is under-exposure rather than confusion. It gives diverse, lower-variance, non-inflated looks, and adaptive alpha backs off as their train F1 rises, trading the weaker lever for the stronger one. Keep the repeat-factor mild: aug diversity is limited on pose, and Kang et al find resampling helps the classifier head but can degrade the shared representation.

Don't oversample, or pour alpha into, the data-floor and input-ceiling classes. driven_flight at 42 clips is unrecoverable by repetition; smash/wrist_smash need better inputs, not more copies. On `shuttleset_18` that tail is the honest cost of an unmerged taxonomy, which is why the 14-class `une_merge_v1` merges those classes. This also sharpens Q4: high alpha on a confused class isn't merely wasted budget, it amplifies a harmful gradient, so the val-improvability gate that pulls it back is doing real work.

### What "mild oversampling" means, by class

The sampling spectrum is `p_c ∝ n_c^q` (Kang et al): q=1 natural/instance-balanced, q=0 fully class-balanced, q=0.5 square-root sampling (Mahajan et al) is the canonical mild midpoint. "Mild, by class" = drawing each class with probability `∝ n_c^q` for q in roughly (0.5, 1), or a per-class repeat factor capped low (~<=2x) so the tail gets a modest lift and the head stays ~1x. Implement via `WeightedRandomSampler` (weight `(1/n_c)**(1-q)`) or LVIS repeat-factor sampling (Gupta et al, `r_c = max(1, sqrt(t/f_c))`).

Two findings specific to from-scratch AR-transformer-on-small-data bend this, and they're the part that matters:

1. From-scratch protects the representation. Kang et al: instance-balanced sampling gives the best representations; rebalancing helps the classifier but degrades features. With no pretraining the representation is the whole game, so stay mild (q~0.7) or use progressively-balanced sampling (natural early -> mild late). That, or a decoupled second stage that rebalances only the classifier head, is the most defensible scheme here.
2. Small data + weak pose aug = repetition is memorisation. Steiner et al's data floor and VideoLT's finding (oversampling few-shot video classes overfits; they synthesise by mixing rather than repeating) agree: a class too rare to span its distribution can't be saved by drawing it more often. flip + 5-10% jitter is a thin diversity engine, so oversampled draws are near-duplicates. Gate it: don't oversample below the floor, and don't oversample classes already easy.

Per taxonomy (ground-truth split-correct train counts; imbalance = n_max/n_min):

| Taxonomy | split | imbalance | mild-oversampling verdict |
|---|---|---:|---|
| `une_v1_14` | split_v2 | 16.4x (net_shot 4139 : long_service 252) | Mostly none, but `cross_court_net_shot` (847) is the one real candidate: rarest of the confusable net-area trio (net_shot 4139, return_net 2392), so it carries a classifier-prior disadvantage that rebalancing fixes; the discriminating cue (cross-court direction) is in the shuttle/court inputs and the between-2-hits window (learnable, not a pose ceiling, unlike smash/wrist_smash); and it's not at a floor (train F1 0.77 < val F1 0.81 in the shuttleset_18 run, i.e. not memorising). Mild ~1.5x, replacing its alpha (already ~1.2-1.7), watching net_shot doesn't sag. `passive_drop` (796) is the same profile. The rare classes (long_service 252, rush 335) are already high-F1, and the macro bottleneck is the abundant smash/wrist_smash pair, which oversampling can't touch. RFS is inert at these counts. |
| `bst_24` | split_bst_baseline | 24.8x (Top_net_shot 2402 : Top_long_service 97) | Best candidate, still mild. No data floor (merge folded driven_flight etc. into drive). Cap <=2x on the under-exposed-but-not-easy sides, mainly `Top/Bottom_rush`, gently the `cross_court_net_shot`/`drive` sides. Leave `long_service` sides at 1x (rare but easy). |
| `shuttleset_18` | split_v2 | 98.6x (net_shot 4139 : driven_flight 42) | Minor lever, heavily gated. Cap <=2x on the moderate-learnable band only (drive 467, defensive_return_drive 270, back_court_drive 263, defensive_return_lob 200). Exclude driven_flight (42, floor; plain sqrt would give it 6.1x), exclude long/short_service (already ~0.97-0.99). Won't fix min_f1=0 or the smash/wrist_smash ceiling. |

So a blanket sqrt scheme is wrong everywhere here: it pours the most weight into exactly the floor/easy classes you don't want (driven_flight 6.1x, long_service 2.5-2.7x). Implementation: prefer the repeat-factor form (oversample the tail, head stays 1x) over probability-reweighting, which down-samples net_shot and starves a learnable head class on small data. And since oversampling raises a class's train F1, adaptive alpha backs off the boosted classes automatically, so treat oversampling as a replacement for alpha on those classes, not an addition (no double-balancing).

---

## Q4. Freeing the headroom over-allocated to plateaued classes

Your clarified point: smash/wrist_smash (and the like) hit their ceiling well before other classes tap out, and you want to free the budget they keep soaking up afterwards. The curves back the spirit of this strongly, with two corrections to the specifics.

### Two corrections from the data

1. **driven_flight's alpha did not go to zero. It's the highest in the model.** `Alpha/driven_flight` climbs to a peak of **2.50** (epoch 27) and is still **1.84** at epoch 80, the largest alpha of any class, while its *val* F1 is flat at 0.000 the whole run. So the loss isn't abandoning it; it's pouring the most weight into the least learnable class. (What's at zero is the val F1, not the alpha. Easy to conflate from a glance at a chart.) This makes the over-allocation case stronger, not weaker.
2. **smash is not the over-allocated one.** Its final alpha is **0.94, below the mean of 1.0**. wrist_smash (1.68) is over-allocated; smash isn't. The real over-allocations are driven_flight (1.84), wrist_smash (1.68), defensive_return_drive (1.66), drive (1.40).

### The mechanism, and why it does exactly what you suspect

Macro F1 rises fast to ~0.66 by epoch 10, reaches ~0.68 by epoch ~25, then crawls to ~0.69 by epoch 80. Best-macro epochs across the 5 serials are 62, 38, 19, 31, 36 (median ~36). So val macro is effectively flat from the late-20s; the back half of training barely moves it.

What *does* keep changing in the back half is the alpha allocation, and the renorm is the culprit. alpha is `(1 - f1_running)^tau` driven by **train** F1, renormalised to mean 1 (so total budget is pinned at n_classes). As the easy classes saturate their train F1 (clear, short_service, long_service, net_shot all push train F1 to 0.95+), their raw_alpha shrinks, and the renorm vacates that budget onto whatever still has the worst *train* F1. That's the unlearnable / plateaued tail. So by construction, the loss spends its late-training budget concentrating weight on driven_flight, wrist_smash and the defensive_return_* classes, none of which improve on val. smash's alpha even drifts up (0.79 to 0.94) over the run not because smash got harder but because the saturating easy classes released budget into the pool.

The root cause unifies Q3 and Q4: **alpha is driven by train F1, which is a bad proxy for val at both extremes**. Tiny memorisable classes (driven_flight) look solved on train but fail on val; confused abundant classes (wrist_smash) stay mediocre on train and keep weight. Train badness is not the same as marginal improvability.

### Why the static clamp you already tried didn't fix it

Capping alpha at a ceiling frees only the slice above the cap (wrist_smash 1.68 to, say, 1.5 frees 0.18 units, spread over 17 classes: negligible), and it doesn't touch the train-vs-val mismatch at all. driven_flight at a cap of 1.5 is still handed 1.5 units for zero val return. The clamp was always going to be marginal.

### The proposal: gate alpha on val improvability, not train badness

Weight a class by whether it's still *improving on val*, not by how bad its train F1 is. Concretely, track an EMA of per-class val F1 and its slope over a window, and decay a class's alpha toward baseline once its val F1 has flatlined for K epochs. A class still climbing keeps full `(1 - f1)`-style weight; a plateaued one releases its budget to the climbers. This would pull driven_flight's ~1.8 and wrist_smash's ~1.7 down toward 1.0 once their val curves go flat (epoch ~25-30), freeing ~1-1.5 units to redistribute, a far bigger reallocation than the clamp.

The nearest established mechanism is GradNorm (Chen et al 2018), which sets task weights from relative training *rate*. The key inversion here: GradNorm *up*-weights a slow task (assuming slow means needs-more-help); you want to *down*-weight a task that has stopped improving despite high weight (it's at a ceiling). The signal for "ceiling, not just slow" is exactly a flat val slope under sustained high alpha.

Risks, all real:
- **Local plateaus.** A class can flatline then jump (often after the LR drops). Use a long window and decay toward baseline 1.0, never to 0, so a re-awakening class can recover weight.
- **Val noise on tiny classes.** The classes you most want to gate (driven_flight, defensive_return_*) have the tiniest, noisiest val sets; driven_flight's val F1 bounces 0 to 0.25 to 0. Act on the EMA-smoothed slope, not raw per-epoch F1.
- **Val leakage.** Driving the loss off val F1 uses val for more than checkpoint selection. It's not training on val labels, and test stays clean, but it's a methodological asterisk; be explicit about it. Steiner et al's Fig 5 is a relevant caution: on long schedules their train-derived "minival" got memorised and mis-ranked models, so they switched to an independent val set. The lesson is that val-based signals degrade on long runs, which argues for combining this with a shorter schedule (below).

### Built 2026-05-31: as implemented

Built in `loss/adaptive_focal.py` (`apply_val_gate`) and wired through `bst_x_train.py` (a `use_val_improvability_gate` flag + a visible `val_improvability_gate` config dict, `--val-improvability-gate` CLI, off by default, requires adaptive_focal; `collation_runner` forwards the flag). The scripts, tables and figures that motivated it are in `docs/architecture_notes/alpha_arc_analysis/`. Off by default and not yet run. What the design discussion changed from the sketch above:

- **The signal is val improvement read as best-so-far, not a slope, and not the train-val gap.** cross_court_net_shot climbs ~0.001/epoch against ±0.02-0.05 per-epoch noise, so a slope estimate needs ~30-40 epochs to clear significance and would wrongly decay it; tracking whether the smoothed val F1 set a new best (by `improvement_margin`) within `patience_epochs` is the noise-robust version, and its failure mode is conservative (a noise-high only delays a decay). The train-val gap was considered and rejected as the primary signal: a positive gap is also just train leading val, so it conflates "overfit and stalled" with "overfit but val still climbing" and would punish a class still gaining. Train F1 alone can't carry it either, since it keeps rising on the plateaued classes via overfitting (driven_flight gains +0.24 train across the back half against 0.00 val).
- **Patience is pinned by the slowest real climber.** cross_court clears the margin only every ~15-20 epochs, so patience must sit above that or the class gets decayed between its legitimate new highs. Defaults: smoothing 0.9, margin 0.015, patience 15, min-before-gating 10, revert step 0.2 (full revert over 5 epochs), freeze at 0.75 of the run.
- **A reverted class lands at exactly 1.0 and does not claw back.** The budget freed by pulling a plateaued class goes only to the classes not pulled that epoch (the climbers and the below-mean classes), so a de-prioritised class holds mean weight rather than rising back above it through the renorm. One-sided: below-mean classes are never pulled.
- **Keep the full run; freeze the gate in the tail instead of cutting the schedule.** The arc analysis found the late-anneal tail does real work for exactly the worst classes (defensive_return_drive and wrist_smash gain +0.05-0.06 in the final ~12 epochs on shuttleset_18 as the LR goes to 0), so the gate freezes past 0.75 of the run rather than the run being shortened. This supersedes the "combine with a shorter schedule" line above.
- **Methodology.** This is val-driven scheduling, the same family as ReduceLROnPlateau and as CDB's own original held-out difficulty signal (which the train-F1 swap replaced). Test stays held out; disclose the val-gating in any writeup.

### Cheaper adjacent win: stop training so long

Best-macro epochs are 19-62 (median ~36) but every serial runs the full 80 because `early_stop_n_epochs=40` needs 40 flat epochs to trigger, which never happens. Epochs ~36-80 are mostly the easy classes overfitting on train and the alpha concentrating onto the dead tail, with macro flat. Cutting `n_epochs` to ~50 or dropping early-stop patience to ~15-20 would remove the wasted tail. This saves compute and avoids the worst of the mis-allocation, but note it does **not** raise macro on its own; checkpoint selection already captures the best epoch. The macro gain, if any, comes from the reallocation, not from stopping.

Update 2026-05-31: the six-cell arc analysis walked this back. The late-anneal tail does real work for the worst classes (defensive_return_drive and wrist_smash gain +0.05-0.06 in the final ~12 epochs on shuttleset_18 as LR goes to 0), so cutting to ~50 would throw those away. The gate keeps the full run and freezes itself in the tail instead. So don't cut the schedule; the tail is earning its keep on exactly the min-F1 classes.

### Honest ceiling

Even a perfect val-improvability alpha is bounded. The mid-tier classes it would feed (cross_court_net_shot, drive, back_court_drive) are themselves mostly plateaued by epoch ~30, so the headroom to unlock is modest, maybe a point or two of macro. driven_flight stays ~0 regardless (42 clips), so min_f1 stays 0 on this taxonomy. The dominant ceiling on `shuttleset_18` is representational and taxonomic: too many tiny mutually-confusable classes. The real levers for that are (a) better inputs for the confusable pairs (the planned X3D wrist crop, for smash/wrist_smash), and (b) the merge itself (why the 14-class taxonomy exists). Touvron et al's separability point is relevant to capacity Run 2: a ViT's latent width has to be wide enough to linearly separate the classes, and 18 confusable classes ask more of that width than 14 merged ones, so widening may matter more on this taxonomy.

### Feasibility of the analysis you wanted

Fully supported by current logs. Per epoch, the `shuttleset_18` TB carries per-class val F1 (`F1_val/{c}`), per-class train F1 (`F1_train/{c}`, the one that actually drives alpha, worth overlaying), per-class alpha (`Alpha/{c}`), and macro/min (`F1/Val_macro`, `F1/Val_min`). So the per-class arcs vs macro vs alpha plots come straight out of each cell's TB, including the une-14 cell when it finishes. The one missing series is LR (not logged), but it's a deterministic cosine:

```
steps_per_epoch = ceil(N_train / batch_size)            # ~178 here
step = epoch * steps_per_epoch
if step < warmup (=100): lr = base_lr * step / warmup
else: progress = (step - warmup) / (total_steps - warmup); lr = base_lr * 0.5 * (1 + cos(pi*progress))
```

with base_lr=5e-4, total_steps = n_epochs*steps_per_epoch. For future runs, one line saves the reconstruction: `writer.add_scalar('Schedule/learning_rate', scheduler.get_last_lr()[0], epoch)`.

---

## Cross-cutting: one balance lever per run

Q2, Q3 and Q4 all move the effective class balance, so change one at a time or attribution collapses. Suggested order:

1. Near-free, do regardless: the no-decay param group, and the LR logging line.
2. WD sweep lambda in {0.05, 0.1, 0.2, 0.4}, schedule fixed, no-decay group on. General regularisation; the current 0.01 is near-inert, so this is the highest-confidence single change here.
3. Tighten the schedule (n_epochs ~50 or early-stop patience ~15-20). Free compute, removes the wasted mis-allocating tail.
4. The val-improvability alpha (Q4). The actual macro-stickiness lever, but the most code and the most caveats; prototype after 1-3 and judge against the per-class arcs.
5. Mild oversampling on the moderate-learnable rare tail only (not the floor/ceiling classes), preferred over raising their alpha; see Q3.

And keep the ceiling in view: on `shuttleset_18` the tail is taxonomically capped. Loss and reg knobs tidy the budget around that cap; better inputs (wrist crop) and the merged taxonomy are what move it.

---

## References

- Wang & Aitchison, *How to set AdamW's weight decay as you scale model and dataset size* (ICML 2025), [arXiv:2405.13698](https://arxiv.org/abs/2405.13698)
- Steiner et al, *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers* (TMLR 2022), [arXiv:2106.10270](https://arxiv.org/abs/2106.10270)
- Touvron et al, *Three things everyone should know about Vision Transformers* (ECCV 2022), [arXiv:2203.09795](https://arxiv.org/abs/2203.09795)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Kang et al, *Decoupling Representation and Classifier for Long-Tailed Recognition* (ICLR 2020), [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
- Mahajan et al, *Exploring the Limits of Weakly Supervised Pretraining* (square-root sampling), [arXiv:1805.00932](https://arxiv.org/abs/1805.00932)
- Gupta et al, *LVIS: A Dataset for Large Vocabulary Instance Segmentation* (repeat-factor sampling), [arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
- Zhang et al, *VideoLT: Large-scale Long-tailed Video Recognition* (ICCV 2021), [arXiv:2105.02668](https://arxiv.org/abs/2105.02668)
- Chen et al, *GradNorm: Gradient Normalization for Adaptive Loss Balancing* (ICML 2018), [arXiv:1711.02257](https://arxiv.org/abs/1711.02257)
- Lin et al, *Focal Loss for Dense Object Detection*, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Cui et al, *Class-Balanced Loss Based on Effective Number of Samples*, [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
