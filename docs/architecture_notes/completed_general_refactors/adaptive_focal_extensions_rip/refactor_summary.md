# Adaptive-focal alpha extensions rip: summary

A behaviour-preserving removal of two retired features from `AdaptiveFocalLoss` and their live wiring across `bst_x_train.py`, `collation_runner.py`, the test suite, and two design docs. Landed on `main` 2026-07-01 under merge commit `39e72a5`.

## Why this pass

Two extensions to the base CDB-F1 loss were built end of May 2026 to attack the wrist_smash / smash floor and the late-training over-allocation of alpha to plateaued classes:

- **`pair_caps`**: a static per-epoch alpha-ratio enforcer between named class pairs (`alpha[numer] >= ratio * alpha[denom]`), built for the smash / wrist_smash confusion pair. Bump absorbed across the other `n - 2` classes so mean alpha stays at 1.0.
- **Val-improvability gate** (called `focal_alpha_revert` in commit messages, run notes, and the `focal_alpha_revert_overallocated` ablation ID): a dynamic per-class plateau-detect-and-decay. Once a class's smoothed val F1 stopped setting new highs by an improvement margin within a patience window, its alpha decayed back toward the renorm mean of 1.0, freeing budget for still-climbing classes.

Both were retired 2026-06-02 after Series H (six taxonomy / split combos, gate-on) and Series J (gate crossed with the two wd endpoints on three cells) showed:

- The gate never gave the best config on any of the taxonomies it ran on.
- The simpler `wd 4e-1 + decay-exclusion` knob matched or beat the gate's lift wherever they overlapped.
- Macro sat inside seed noise for every gate-on cell; the one modest min-F1 lift on `bst_25` at wd 1e-2 sat below the old standard.

Full retirement writeup: `../../bst_x_sweep_summary_wd_x_focal_alpha_revert.md`. Motivating per-class arc analysis (what convinced anyone to build the gate in the first place): `../../alpha_arc_analysis/`.

After the retirement verdict the wiring sat in the tree for a month, off by default and untouched. The extensions carried ~320 lines of loss-class code, ~90 lines of trainer wiring, ~300 lines of tests, and ~50 lines of live-wiring documentation. The rip clears that surface out while preserving a documented reversal path.

## What was delivered

Seven commits on `remove-adaptive-focal-extensions`, grouped into three buckets:

**A: Extension removal in code.** Batches 1, 3, 4, 5 (in the plan's numbering).

- `8423b25` **Start of sequence removing pair_cap and val-improvability-gate.** 14 tests come out of `tests/test_adaptive_focal.py` (9 `test_pair_cap_*` + 5 `test_gate_*`). Base CDB-F1 tests and shared fixtures stay. Prep for the extension rip; the reversal path lives in `docs/architecture_notes/focal_alpha_revert_sketch.md`.
- `bf2d97e` **Rip pair_caps + val-improvability-gate wiring from bst_x_train.** Removes the `Hyp` gate fields, the gate-specific fail-loud guard (the two intrinsic CDB-F1 guards stay), the extension kwargs on the `AdaptiveFocalLoss(...)` call, both diagnostic prints, the per-epoch `apply_val_gate` call, the `Revert/{c}` TB scalar, the `--val-improvability-gate` CLI flag, and the `cell_overrides['use_val_improvability_gate']` plumbing.
- `74a3d3c` **Rip the val-improvability-gate flag forwarding from collation_runner.** Five-line stanza out. Cell configs stop carrying `use_val_improvability_gate`; the runner stops emitting `--val-improvability-gate`.
- `914e60c` **Rip pair_caps + val-improvability-gate from AdaptiveFocalLoss.** Removes both extension constructor args (plus `n_epochs`), `_resolve_pair_caps`, `_init_val_improvability_gate`, `apply_val_gate`, the pair-caps enforcement tail of `update_alpha`, and the five gate buffers. Base `update_alpha` and `forward` unchanged. Bit-exact numerical fingerprint on the default cell config confirms training behaviour is byte-identical to pre-rip on main.

**B: Live-wiring documentation.** Batches 6 and 7.

- `5ea8168` **Update live-wiring docs to drop gate + pair_caps references.** Trims the "Config and turning it on" + "Status" sections in `class_f1_focal_design.md` (§9) and the "as implemented" block in `hp_and_aug_speculations_30_05_2026.md` (Q4). Past-tense rewrites carry a hook to the reversal sketch. Design analysis, arc plots, sweep summary, and manifest history stay.
- `92cec28` **Note the rip commit on the focal_alpha_revert reversal doc.** Adds a `**Removed:**` line under the pinned-at banner. Body still pinned to `1104562` as the pre-rip snapshot; reversal checklist unchanged.

**C: Comment tidy pass.** Batch 9.

- `a5a8628` **Tidied comments flowing from the removal of bst_x's loss' pair_caps and val_improvability_gate.** Three edits: (1) the historical "pair-cap" mention in the `Hyp.adaptive_focal` default-rationale comment gets a follow-up sentence flagging it as retired with a hook to the reversal sketch; (2) `_build_loss_fn` docstring: "class_weights lever" -> "class_weights knob" (plain-speech swap); (3) `class_f1_focal_design.md` "Honest ceiling": reworked the "The levers that move those" sentence for positive framing.

## Regression evidence

**Tier 2 bit-exact fingerprint (Batch 5 gate).** A small deterministic capture script instantiates `AdaptiveFocalLoss` on the default gate-off, no-pair_caps path, feeds five fixed seeded update steps, and dumps a JSON with per-step alpha, `f1_running`, forward loss, and a state-dict SHA. Pre-rip fingerprint (twice on main, byte-identical: `ae071c45...ec7684`) vs post-rip fingerprint on the branch: byte-identical. Every alpha value at every step preserved to the bit. The default code path is unchanged.

**HPC 3-serial sanity (Batch 8).** One-serial-per-run bourbaki run against the current practical baseline (`une_v1_14` / `split_v2` / wd 4e-1). Default multi-serial mode; ran three serials before the manual cut. Compared 3-serial mean against Row #47 (`run_20260601_021234_940276`, wd 4e-1 winner of the WD sweep) 5-serial mean:

| Metric | 3-serial mean (branch) | Ledger #47 mean-of-5 | Delta |
|---|---|---|---|
| Macro F1 | 0.7397 | 0.7463 | -0.66% |
| Min F1 | 0.5005 | 0.4979 | +0.26% |
| Accuracy | 0.7585 | 0.7670 | -0.85% |

Min-F1 (the primary metric this cell was picked on) sits comfortably inside the pinned ±1.0% pass window. Macro at -0.66% is 0.16% outside the tight ±0.5% window; per-serial spread on the branch was 0.013 macro, so a 3-vs-5 sample-size mismatch fully explains that delta.

## Adversarial-review coverage

Every batch that touched code (3, 5, 6) had an independent per-stage reviewer pass over the diff against the OUT-list and the batch spec before commit. All three returned CLEAN verdicts. An end-of-branch audit before merge cross-checked 10 dimensions (OUT-list compliance, in-scope completeness, every historical-refs decision, bit-exact regression, HPC handoff reconciliation, commit hygiene, full pytest, ruff, straggler grep, no stray unicode). Verdict: CLEAN with one advisory folded into the post-plan TODOs.

## Post-plan TODOs

Three follow-ups queued after merge, all doc updates that flow from the code being gone:

- `docs/architecture_notes/function_invariants/train_network.md`: annotated with a historical note at the top; the extension-method references in the body are preserved as a snapshot of the 30 June simplification pass. Doc's own frontmatter already flags itself as a pre-pass snapshot, so the treatment is consistent.
- `docs/architecture_notes/completed_general_refactors/simplification_pass/simplification_review.md:61`: single `apply_val_gate` mention in a "clean on the simplification lens" note annotated with a retirement footnote.
- `docs/architecture_notes/focal_alpha_revert_sketch.md` `Removed:` banner: SHA fill-in with the merge commit `39e72a5`.

All three landed under a follow-up commit alongside this dir's creation.

## Related docs

- `../../focal_alpha_revert_sketch.md`: reversal sketch, pinned at `1104562`. Exhaustive touch-point table + verbatim source of every extension method + reversal checklist.
- `../../bst_x_sweep_summary_wd_x_focal_alpha_revert.md`: the retirement writeup. Series H + J numbers and the verdict that retired the gate.
- `../../alpha_arc_analysis/`: the per-class arc analysis that motivated building the gate in the first place. Figures, `findings.md`, `tables.md`, and the parser / plotter scripts.
- `../../class_f1_focal_design.md` §9: the base design doc for the gate, past-tensed after Batch 6.
- `../../hp_and_aug_speculations_30_05_2026.md` Q4: the terser decision-focused writeup.
