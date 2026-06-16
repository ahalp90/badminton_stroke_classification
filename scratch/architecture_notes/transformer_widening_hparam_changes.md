# Transformer-widening hparam changes

*Companion to `hparams_sweep_speculations.md`. Captures the
implementation surface for the encoder-side capacity ablation
(d_model and d_head changes), separate from the loss-side and
data-side knobs the sweep doc tracks. Written 2026-05-03 after
capacity-bump Run 1 (mlp_head hidden 400 → 1200) returned flat.*

## What this covers

The encoder-side capacity experiment sometimes called Run 2 in
`bst_x_overview.md`: `d_model=100 → 192` paired with
`d_head=128 → 32` (Voita-style trim so the 7.68x d_head:d_model
over-provisioning doesn't propagate). `n_head=6` stays.

Run 1 (head-side widening, mlp_head hidden 400 → 1200) shipped
and returned flat: head metrics within seed noise, wrist_smash
mean -4.8 pp vs y1t1. The mlp_head swap has been reverted at
`bst.py:202` to keep the baseline clean. Run 2 still goes for
full coverage of the capacity question, with a weakened prior:
expected gain on test macro 0-2 pp, possibly less.

## Mandatory code changes

### 1. `d_model` 100 → 192
- Surgical: change the default at `BST.__init__`.
- Better practice: lift `d_model` into the `Hyp` namedtuple
  (`bst_x_train.py:66-74`) and pass through, so the value is
  sweep-controllable and shows up in the train log. ~5 lines of
  plumbing.

### 2. `d_head` 128 → 32 (n_head=6 stays)
- Same options as d_model: change BST default vs. plumb through Hyp.

### 3. `mlp_head` hidden — explicit override or inherit from formula
With the formula reverted to `d_model * mlp_d_scale`, d_model=192
makes the head hidden become 768 automatically (192 * 4) without
any further code change. That's a passive, undeclared widening
that comes along for the ride with d_model.

If the goal is comparison parity with Run 1's head shape (1200),
override explicitly. Two ways:
- Surgical: hardcode `MLP_Head(head_dim, n_class, 1200, drop_p)`
  at `bst.py:202` for the run, with a `# CHANGED` note. Revert
  after.
- Better practice: add a `head_hidden_dim=None` kwarg to
  `BST.__init__`; default `None` keeps the current
  `d_model * mlp_d_scale` formula, explicit value overrides. ~6
  lines including signature + the conditional. One-time wiring,
  per-run choice forever.

The "better practice" route here pays off even with low future
sweep volume because the alternative leaves the head-hidden
choice implicit-via-d_model rather than declared.

## Verifications before launch (5-min greps, no code changes expected)

### 4. `d_model` is sourced from the constructor everywhere
Confirm TCN_pose, TCN_shuttle, every transformer layer's QKV/O
projections, every FFN, mlp_clean (CG), and the head_dim
calculation all read d_model from the BST constructor (no stale
`100` literals anywhere). The `model_capacity_bottleneck_question.md`
walkthrough claims this is the case; cheap to verify with a grep
for `100` and `d_model` across `bst.py` and `tempose.py`.

### 5. `mlp_positions` (PPF) input dim independence
The capacity doc lists this as `2 → 256 → 72`, with 72 being the
per-frame pose feature dim, not d_model. Quick grep to confirm.
If 72 turns out to be tied to d_model anywhere, the change
cascades unexpectedly.

### 6. Attention tail uses `d_head * n_head`, not a hardcoded 768
The tail projection should be `Linear(d_head * n_head, d_model)`
computed from constructor args. If anything is hardcoded `Linear(768, 100)`
the d_head trim won't propagate cleanly. Capacity doc says it's
parameterised; confirm in `MultiHeadCrossAttention.tail`
(`bst.py:59-62`) and the temporal/interactional transformer
attention modules in `tempose.py`.

## Manifest record

`manifest.yaml.extra.arch` (`bst_x_train.py:860-866`) doesn't
currently record d_model, d_head, n_head, or head_hidden. Three
options:
- **Surgical**: write a line in `best_model_id.txt` and the
  commit message stating the dim changes. Done.
- **Better practice**: extend `_validate_and_record_arch` (the
  function that builds extra.arch) to capture these dims. ~10
  lines. Pays off if any further capacity touches happen.
- **Compromise**: log the values to the train console at
  serial 1 only, leaving the manifest schema unchanged.

The "max 3rd capacity change then settled" constraint suggests
surgical for the manifest piece. Revisit if more capacity work
opens up.

## LR schedule

Same as y1t1 baseline: lr=5e-4, cosine `num_cycles=0.5`,
n_epochs=80, warm_up_step=100. AdamW is per-parameter
scale-invariant so widening doesn't change the LR the existing
weights want.

Watch flags during the run:
- If train loss looks unstable in epochs 1-3, bump
  `warm_up_step` 100 → 150 and rerun. Wider models sometimes
  want a slightly longer warmup before the residual stream
  settles.
- If best epoch creeps past 70 at early stop, bump `n_epochs`
  80 → 100. Unlikely at 1.92x widening, but cheap insurance.
- LR magnitude itself: don't touch. The "wider models want lower
  LR" rule from muP / scaling-laws kicks in at much larger width
  changes than 1.92x, and the cosine schedule already anneals.

Per-epoch wall-time bump: estimate +30-60% vs y1t1. d_model
widens TCN, every transformer layer's QKV/O projections, and
every FFN (FFN params scale ~d_model^2, ~3.7x at d_model=192 vs
100). d_head trim claws some back inside attention.

## Picks given low future sweep volume

- Items 1, 2: surgical. Edit BST defaults; document in
  `best_model_id.txt`.
- Item 3: better practice (the `head_hidden_dim` kwarg). Cleanly
  separates the architectural-rule decision from the per-run
  override decision.
- Item 7 (manifest record): surgical.
- Items 4-6: 5-min grep before launch. No code changes expected.

## What Run 2 actually tests vs Run 1

Run 1 was decision-side capacity (head only). Run 2 is
encoder-side capacity (residual stream throughout). The
capacity-bottleneck research argued *both* are bounded by the
same 0-2 pp prior, but the mechanism is different enough that
Run 1's flat-line doesn't directly imply Run 2's. The pair-
confusion failure mode is supposed to be representation-bound
(the encoder isn't separating smash from wrist_smash), and a
wider head shouldn't fix it. Run 1 confirmed it didn't. Run 2 is
at least topologically positioned to address pair-confusion more
directly because it widens the representation that's failing to
separate the pair. Whether 1.92x is enough to actually move
separation is the open question.

If Run 2 also flat-lines on test macro and ws, capacity is
empirically closed and the prior on X3D-S fusion correspondingly
strengthens.
