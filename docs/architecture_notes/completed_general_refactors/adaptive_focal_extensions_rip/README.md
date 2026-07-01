# Adaptive-focal alpha extensions rip: history for the repo

The durable record of the rip that removed `pair_caps` and the val-improvability gate from `AdaptiveFocalLoss`. The historical-refs decisions doc drove the case-by-case doc surgeries; the summary + worklog are the per-commit landed-on-main trail. The reversal sketch that captures the pre-rip snapshot lives alongside at `../../focal_alpha_revert_sketch.md`.

Merged to `main` 2026-07-01 at `39e72a5` ("Merge the branch cleaning the deprecated adaptive_focal and loss-pairing bst_x training features (shown not to improve performance, and the legacy left complicated code for no benefit).").
7 commits on `remove-adaptive-focal-extensions` landed under a single `--no-ff` merge. Base loss code path bit-exact preserved; HPC 3-serial verified within seed noise on min-F1.

## What's in this dir

- `refactor_summary.md`: plain-language summary. Why the extensions came out, what each batch did, what stayed put
- `refactor_worklog.md`: per-commit log: files / change / gate / SHA / reviewer verdict. Audit trail for the diff against `main`
- `historical_refs_decisions.md`: the 12 case-by-case decisions on doc and comment references that survived the rip. What was kept as historical, what was stripped, what was past-tensed with a hook to the reversal sketch

Sibling reversal doc at `../../focal_alpha_revert_sketch.md` carries the pre-rip snapshot pinned at commit `1104562`: exhaustive touch-point map + verbatim source of every extension method + reversal checklist. That's what a future reinstatement pass would read.

## What's not in this dir (stays archived)

The full planning trail (`PLAN.md`, the bit-exact fingerprint JSONs, the boursync'd HPC run outputs at ~34 MB, the four subagent transcripts) stays in `~/Documents/COSC594/adaptive_focal_extensions_rip/` as the working archive.

## Six-month skim

Two extensions to `AdaptiveFocalLoss` were built end of May 2026:

- **`pair_caps`**: a static per-epoch alpha-ratio enforcer between named class pairs, built for the smash / wrist_smash confusion pair.
- **Val-improvability gate** (called `focal_alpha_revert` in run notes): a dynamic per-class plateau-detect-and-decay over the adaptive-focal alpha.

Both were retired 2026-06-02 after Series H + J sweeps showed neither gave the best config on any tested taxonomy: the simpler `wd 4e-1 + decay-exclusion` knob matched or beat the gate's lift wherever they overlapped. The extensions sat unused in the tree for a month, then came out with this rip.

The rip removed 320 lines from `adaptive_focal.py`, 88 lines from `bst_x_train.py`, 5 lines from `collation_runner.py`, and 302 lines from `test_adaptive_focal.py`, plus targeted trims to two design docs. Net +26 / -755 across 7 files. Base CDB-F1 loss code path is bit-exact identical pre- and post-rip on the default gate-off, no-pair_caps config; verified by a Tier 2 numerical-fingerprint diff.

HPC handoff (bourbaki, 3 serials on the current practical baseline: `une_v1_14` / `split_v2` / wd 4e-1) landed within seed noise of the Row #47 ledger reference: macro -0.66% (0.16% outside the tight ±0.5% window; within seed spread given the 3-vs-5 sample-size mismatch and the 0.013 macro spread across the three branch serials), min-F1 +0.26% (comfortably in window), accuracy -0.85%.

Reinstatement is a documented path: the reversal sketch at `../../focal_alpha_revert_sketch.md` pins every touch point + verbatim source at commit `1104562`. A future reinstatement pass paste-reverts from there.
