# Historical references in code and docs: case-by-case decisions

Per your case-by-case policy on historical refs, every borderline case is enumerated below with a proposed disposition. The pattern for each entry:

- **Location + snippet**
- **What it references**
- **Proposal:** Keep as historical / Strip with the rip / Rewrite in past tense
- **Reason**
- **Your decision:** _(empty; fill in with Keep / Strip / Rewrite / Other)_

Batches 3-6 halt on any unresolved decision.

---

## 1. `Hyp.adaptive_focal` default rationale comment

**Location:** `src/bst_x/bst_x_train.py:117-121`

**Snippet:**
```
adaptive_focal: dict | None = {
    # First-run sweet spot from run_20260501_164658: tau=1, gamma=1.
    # All four CDB knob variants (gamma=0, tau=0.5, pair-cap, gamma=2)
    # traded wrist_smash back for smash without macro moving, so this
    # combo holds the floor-lift sweet spot (+8.7 pp wrist_smash on the
    # LS=0.1 baseline). Active default for the capacity-bump runs.
    'tau': 1.0,
    ...
```

**What it references:** the word `pair-cap` appears in the historical list of four CDB knob variants tested to justify why `tau=1 / gamma=1` is the current default.

**Proposal:** Keep as historical.

**Reason:** The reference is contextual (why this default was chosen after testing four alternatives), not functional. Stripping it loses the rationale for the current default and hides the fact that pair-cap was one of the tested knobs. The reader is not misled: the comment names a variant tested in the past, not a currently-available knob.

**Your decision:**
Agreed, just park it for a phrasing tidy at the final block.
---

## 2. `_build_loss_fn` docstring: "three fail-loud guards"

**Location:** `src/bst_x/bst_x_train.py:374-376`

**Snippet:**
```
Owns the three fail-loud guards: ``use_val_improvability_gate`` needs
``adaptive_focal``; ``adaptive_focal`` is mutually exclusive with
``class_weights``; ``adaptive_focal`` requires ``label_smoothing=0.0``.
```

**What it references:** the count "three" plus the gate-specific guard.

**Proposal:** Rewrite in present tense with two guards.

**Reason:** Post-rip, only two guards remain. The docstring must reflect current code. This is a straightforward update, not a historical-reference call: it describes what the function does now, not what it did.

**Proposed replacement:**
> Owns the two fail-loud guards: ``adaptive_focal`` is mutually exclusive with ``class_weights``; ``adaptive_focal`` requires ``label_smoothing=0.0``.

**Your decision:**
Agreed.
---

## 3. `_build_loss_fn` docstring: gate-modulates-alpha sentence

**Location:** `src/bst_x/bst_x_train.py:397-399` (approximately; inside the `adaptive_focal` docstring block)

**Snippet:**
```
The val-improvability gate modulates the adaptive-focal alpha, so it can
only run when adaptive_focal is engaged. Fail loud rather than silently
ignoring the flag (the gate config would otherwise be dropped on the floor).
```

**What it references:** the gate as a live modulator.

**Proposal:** Strip with the rip.

**Reason:** No gate to describe post-rip. The surrounding docstring paragraph about `adaptive_focal` itself stays; only this two-sentence explainer goes.

**Your decision:**
Agreed
---

## 4. `adaptive_focal.py` module docstring: pair-cap paragraph

**Location:** `src/bst_x/loss/adaptive_focal.py:14-18`

**Snippet:**
```
Optional pair-cap extension targets known confusion pairs the scalar-per-class
CDB signal can't see: cap ``alpha[numer] / alpha[denom]`` from below at a
configured ratio so a high-F1 partner doesn't get downweighted past the point
where its training signal collapses. The bump is absorbed across the other
``n_classes - 2`` classes so mean alpha stays 1.0.
```

**What it references:** the pair-cap extension itself as live functionality.

**Proposal:** Strip with the rip.

**Reason:** In-scope wiring documentation, not historical context. Reversal path documented in `focal_alpha_revert_sketch.md`.

**Your decision:**
Agreed
---

## 5. `class_f1_focal_design.md` L806-811: gate design bullets (before the "Config" section)

**Location:** `docs/architecture_notes/class_f1_focal_design.md` L806-811 (approx.)

**Snippet:** design bullets like "Don't gate too early. No decay before epoch 10..." and "Freeze in the anneal tail. The arc analysis found the last stretch of training does real work..."

**What it references:** the gate's design principles, motivated by the arc analysis. Doesn't reference wiring; discusses design tradeoffs.

**Proposal:** Rewrite in past tense OR keep as design history.

**Reason:** These are design bullets that describe how the gate WAS designed, based on the arc analysis. They read as prescriptive ("Don't gate too early") but are really describing rationale for the built defaults. Two possible reads:

- (a) These belong with the gate wiring (which is going), so strip.
- (b) These are design analysis that motivated a specific set of defaults; the arc analysis they cite is in the OUT-list (`alpha_arc_analysis/` stays). Rewrite in past tense to make it clear the gate is not current.

**Proposal:** rewrite in past tense. Preserves the reasoning without pretending the wiring is live.

**Your decision:**
Past tense, with a hook to the reinstatement doc if reinstatement is ever desired.
---

## 6. `class_f1_focal_design.md` L817-833: "Config and turning it on"

**Location:** `docs/architecture_notes/class_f1_focal_design.md` L817-833

**Snippet:** the whole "Off by default. The knobs live in a `val_improvability_gate` dict..." section, plus the code block showing the dict, plus the "Turn it on with `use_val_improvability_gate=True`..." paragraph.

**What it references:** live wiring: constructor args, CLI flag, collation_runner cell keys, TB scalars.

**Proposal:** Strip with the rip.

**Reason:** Pure live-wiring documentation. Zero historical value once the wiring is gone.

**Your decision:** Agreed

---

## 7. `class_f1_focal_design.md` L835-837: "Honest ceiling"

**Location:** `docs/architecture_notes/class_f1_focal_design.md` L835-837

**Snippet:**
```
Even a perfect gate buys a point or two of macro. The classes it feeds
(the still-climbing mid-tier) are themselves mostly half-plateaued,
driven_flight stays at 0 (42 clips, unrecoverable by reweighting), and
wrist_smash stays confusable. The levers that move those are the inputs
(the planned X3D wrist crop) and the taxonomy merge, not a loss knob.
The gate tidies the budget around that ceiling; it doesn't lift it.
```

**What it references:** design analysis of what the gate can and can't achieve.

**Proposal:** Rewrite in past tense.

**Reason:** This is genuine design analysis (macro ceiling caused by wrist_smash / driven_flight / mid-tier plateau), independent of the gate wiring. Worth preserving in past tense as design rationale for later choices.

**Proposed replacement:** rephrase "Even a perfect gate buys..." as "The gate topped out at a point or two of macro..." and drop the "The gate tidies the budget..." line.

**Your decision:** Past tense, flagging that the deprecated feature is deprecated

---

## 8. `class_f1_focal_design.md` L839-841: "Status"

**Location:** `docs/architecture_notes/class_f1_focal_design.md` L839-841

**Snippet:**
```
Built 2026-05-31 (`apply_val_gate` in this module plus the wiring in
`bst_x_train.py` and the `collation_runner` forward). Off by default,
not yet run. Unit tests in `tests/test_adaptive_focal.py` section 9;
two independent reviews passed. Full analysis, tables and figures in
`docs/architecture_notes/alpha_arc_analysis/`.
```

**What it references:** live wiring + test coverage.

**Proposal:** Strip and replace with a two-line historical pointer.

**Reason:** The wiring is going; the tests are going; but the design analysis at `alpha_arc_analysis/` stays. A one-line pointer to `focal_alpha_revert_sketch.md` + `bst_x_sweep_summary_wd_x_focal_alpha_revert.md` gives the reader the full history.

**Proposed replacement:**
> Built 2026-05-31, retired 2026-06-02, ripped from the tree 2026-07-01. Reversal path: `focal_alpha_revert_sketch.md`. Retirement writeup: `bst_x_sweep_summary_wd_x_focal_alpha_revert.md`. Motivating arc analysis: `alpha_arc_analysis/`.

**Your decision:**
Agreed
---

## 9. `hp_and_aug_speculations_30_05_2026.md` L214-222: "Built 2026-05-31: as implemented"

**Location:** `docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md` L214-222

**What it references:** wiring at build time + design bullets (signal choice, patience, one-sided handoff, freeze in tail, methodology).

**Proposal:** Rewrite the "Built ... as implemented" opening sentence in past tense; keep the design bullets underneath.

**Reason:** The bullets are design analysis (signal choice, one-sided handoff etc.) worth preserving. Only the "Built in `loss/adaptive_focal.py` (`apply_val_gate`) and wired through..." sentence describes live wiring.

**Proposed replacement of the opening sentence:**
> Built 2026-05-31, retired 2026-06-02 (see `bst_x_sweep_summary_wd_x_focal_alpha_revert.md`), ripped 2026-07-01. The design bullets below are the ones that drove the shape it landed at.

**Your decision:**
Agreed
---

## 10. `function_invariants/train_network.md`: gate references

**Location:** `docs/architecture_notes/function_invariants/train_network.md`. 15+ mentions across L69, L87, L111, L122, L142, L276, L282, L291, L326, L353, L366, L423, L427, L445, L452, L467.

**What it references:** this doc pins the per-epoch ordering invariant (`update_alpha` -> `validate` -> `apply_val_gate`) plus references `apply_val_gate` as part of the seeded before/after equivalence gate that the 2026-06 code simplification pass used.

**Proposal:** Batch-audit this file separately during Batch 6, with a targeted diff proposal presented before edit.

**Reason:** Higher-touch than the other docs. Whether references are historical (documenting the refactor pass) or current-state (documenting the live invariant) needs a per-reference call. The file may already be a historical artefact of the closed 2026-06-30 simplification pass, in which case the treatment differs from a live invariant doc.

**Sub-question:** is this doc a live-contract doc (updates each time the function changes) or a historical snapshot of the refactor pass (frozen and superseded)? Your call determines whether Batch 6 does surgery on it or leaves it entirely as historical.

**Your decision:** This documented findings of the 30 june simplification pass. It snapshots that period up to the merge of that branch. It should be updated *after the rest of this plan is executed*. Log as a TODO, please.

---

## 11. `aim_backfill.py:10` docstring prefix list

**Location:** `src/bst_x/aim_backfill.py:10` (docstring)

**What it references:** lists `Alpha/*`, `Revert/*` and other TB scalar prefixes.

**Proposal:** Strip `Revert/*`.

**Reason:** Post-rip, no code emits `Revert/{c}`. The docstring goes stale until updated. `Alpha/*` stays regardless.

**Your decision:** Sure.

---

## 12. `run_tracker.md:138-139` prose list

**Location:** `src/bst_x/run_tracker.md:138-139`

**What it references:** same prefix list, prose form.

**Proposal:** Strip `Revert/*`.

**Reason:** same as #11.

**Your decision:** Sure

---

## Summary of default proposals

Assuming the user accepts every proposal above:

- **Kept (as historical):** #1 (Hyp default rationale comment)
- **Stripped:** #3, #4, #6, #11, #12
- **Rewritten in past tense:** #2, #5, #7, #8, #9
- **Deferred to per-file audit at Batch 6:** #10

Please review and sign off (or edit) each of the 12 decisions before Batch 1 starts.
