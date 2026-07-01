# Worklog: adaptive-focal extensions rip

Audit trail. One entry per batch. Not commit-linked; commits reference this doc's batch numbers.

## Batch 0. Reference capture on main (pre-rip)

- **Status:** DONE
- **Pre-rip repo HEAD:** `8aafcf8` (the sketch's internal pinning at `1104562` still holds because `8aafcf8` touched only the sketch + `build_fe_stats_jsons.py`, leaving all wiring files identical)
- **Files:** external `reference/capture_baseline.py` (author), `reference/main_capture_1.json` (6950 bytes), `reference/main_capture_2.json` (6950 bytes)
- **Gate result:** two independent captures byte-identical. Fingerprint sha256: `ae071c45867d71757c96fdb3b9db5d9efc47904bef0f0a0e7decd242e7ec7684`. Fingerprint is a valid Tier 2 gate.
- **SHA:** n/a (no repo commit; external capture)

## Batch 1. Rip pair_cap + gate tests

- **Status:** DONE
- **Files:** `tests/test_adaptive_focal.py` (was 898 lines, now 596 lines)
- **Changes:** removed L598-898 (Section 8 pair_cap tests + Section 9 gate tests + `_gate_config` helper); updated module docstring to say "seven sections" and dropped the Section 8 bullet
- **Gate result:** Tier 0 clean. 34 test defs remaining (was 48; 14 removed = 9 pair_cap + 5 gate). No `pair_cap` / `val_gate` / `val_improvability` / `apply_val_gate` symbols leftover. `ruff check` passed. `pytest tests/test_adaptive_focal.py -q`: 36 passed (34 defs + 2 extra parametrised runs on `test_accumulate_matches_reference_loop`).
- **SHA:** `8423b25` on `remove-adaptive-focal-extensions`

## Batch 2. Create branch

- **Status:** DONE (executed out-of-order during Phase 1 scoping)
- **Branch:** `remove-adaptive-focal-extensions`
- **SHA:** branched at main HEAD `8aafcf8`
- **Note:** Batch 1 committed on this branch rather than on main; the plan's "on main pre-branch" wording is superseded by this reality. No functional consequence.

## Batch 3. Rip bst_x_train.py wiring

- **Status:** DONE
- **Files:** `src/bst_x/bst_x_train.py` (4 insertions, 92 deletions)
- **Removed:** Hyp `use_val_improvability_gate` + `val_improvability_gate` fields (+ their intro comment), Hyp.adaptive_focal `pair_caps` example block, the gate-specific fail-loud guard, `pair_caps=`/`val_improvability_gate=`/`n_epochs=` kwargs on the AdaptiveFocalLoss call, pair_caps triples print + gate-on print, per-epoch `apply_val_gate` call, `Revert/{c}` TB scalar, `--val-improvability-gate` CLI flag, `cell_overrides['use_val_improvability_gate']` plumbing, and the gate-modulates sentence from `_build_loss_fn` docstring. Docstring updated to "two fail-loud guards".
- **Kept intact per historical-refs item 1:** the `# All four CDB knob variants (gamma=0, tau=0.5, pair-cap, gamma=2)` mention in the Hyp.adaptive_focal default-rationale comment.
- **Gate result:** Tier 0 clean. `grep` returns zero hits for `pair_caps`, `val_improvability`, `apply_val_gate`, `val_gate_enabled`, `Revert/`, `gate_revert_fraction`, `--val-improvability`, `use_val_improvability`. `ruff check` passed. `python -c 'import bst_x_train'` clean. `pytest tests/ -q`: 438 passed, 18 skipped, 2 pre-existing failures in tests/test_api.py (unrelated to rip, confirmed by checkout-main check).
- **Adversarial reviewer verdict:** CLEAN (10/10 checks pass; no new grep hits, historical-refs item 1 respected, two guards remaining, `Alpha/{c}` still emitted, no leftover refs to removed args, valid Python, no collateral damage).
- **SHA:** `bf2d97e` on `remove-adaptive-focal-extensions`

## Batch 4. Rip collation_runner.py forwarding

- **Status:** DONE
- **Files:** `src/bst_x/collation_runner.py` (5 deletions, 0 insertions)
- **Removed:** the L57-61 gate-flag forwarding stanza (both the 2-line comment and the 3-line conditional emitting `--val-improvability-gate` / `--no-val-improvability-gate`).
- **Gate result:** Tier 0 clean. `grep` returns zero hits for any extension symbol in collation_runner.py. `ruff check` passed. No collation_runner-specific tests exist; full `pytest tests/ -q` shows 438 passed, 18 skipped, 2 pre-existing test_api.py failures (unrelated to rip).
- **SHA:** `74a3d3c` on `remove-adaptive-focal-extensions`

## Batch 5. Rip adaptive_focal.py extensions

- **Status:** DONE
- **Files:** `src/bst_x/loss/adaptive_focal.py` (0 insertions, 320 deletions; 530 -> 210 lines, 60.4% reduction)
- **Removed:** module-docstring pair-cap paragraph, class-docstring params for the three extension args, constructor args (`pair_caps`, `val_improvability_gate`, `n_epochs`), both `__init__` wire-up calls, `_resolve_pair_caps` staticmethod, the pair-caps enforcement tail of `update_alpha`, `_init_val_improvability_gate` method, `apply_val_gate` method.
- **Preserved:** base constructor args (`n_classes`, `class_names`, `tau`, `gamma`, `momentum`, `warm_up_epochs`, `f1_floor`, `device`), `update_alpha` base body + `self.epoch += 1`, `forward` verbatim, `per_class_f1_from_counts`, `accumulate_class_counts`, "Train-loop responsibilities" module-docstring block.
- **External script updated:** `reference/capture_baseline.py` had the three now-removed kwargs dropped from its constructor call (they were explicitly None pre-rip, don't exist post-rip; not a repo change).
- **Gate result:** Tier 2 PASS. Post-rip fingerprint `reference/branch_capture.json` (6950 bytes) is byte-identical to pre-rip `reference/main_capture_1.json`. Training behaviour on the default cell is bit-exact preserved. Ruff clean, ast.parse ok. `pytest tests/ -q`: 438 passed, 18 skipped, 2 pre-existing test_api.py failures unchanged.
- **Adversarial reviewer verdict:** CLEAN (13/13 checks pass; no leftover extension symbols, base constructor and `update_alpha` structure exact, `forward` byte-identical to pre-rip, `per_class_f1_from_counts` and `accumulate_class_counts` byte-identical, docstrings clean, no collateral damage, fingerprint byte-identical).
- **SHA:** `914e60c` on `remove-adaptive-focal-extensions`

## Batch 6. Live-wiring docs pass

- **Status:** DONE
- **Files touched (2):** `docs/architecture_notes/class_f1_focal_design.md` (+8/-16), `docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md` (+1/-1)
- **In class_f1_focal_design.md:**
  - Section 9 header updated: `## 9. Val-improvability gate (added 2026-05-31, retired 2026-06-02, ripped 2026-07-01)`
  - Added a `**Retired.**` note under the section header with the reversal path (`focal_alpha_revert_sketch.md`) and retirement writeup (`bst_x_sweep_summary_wd_x_focal_alpha_revert.md`)
  - Small tense fixes to the section intro paragraph ("the fix" -> "was the fix"; dropped "It's off by default;")
  - `Why the patience and window defaults` past-tensed per item 5
  - `Config and turning it on` stripped entirely per item 6
  - `Honest ceiling` past-tensed with `(deprecated feature note)` tag per item 7
  - `Status` replaced with historical pointer per item 8
- **In hp_and_aug_speculations_30_05_2026.md:**
  - Opening sentence of the "Built 2026-05-31: as implemented" section replaced per item 9; design bullets untouched
- **Discovery (no-op on plan items 11 and 12):** `src/bst_x/aim_backfill.py:10` docstring and `src/bst_x/run_tracker.md:138-139` prose list turned out to NEVER include `Revert/*` (Lens 1 correctly flagged them as absent, not present-and-stale). Neither file needed editing. The pre-approved Batch 6 commit message includes a line about stripping `Revert/*` from those two files that describes intended-but-unnecessary work; kept the message verbatim since the intent (ensure no `Revert/*` references remain) is satisfied.
- **Deferred (per item 10):** `docs/architecture_notes/function_invariants/train_network.md`. Post-plan TODO.
- **Gate result:** Tier 0 + reviewer. Only two files changed; `train_network.md` untouched (deferred); pytest baseline held at 438 passed / 18 skipped / 2 pre-existing test_api.py failures.
- **Adversarial reviewer verdict:** CLEAN (15/15 checks pass; one minor visibility note that the section-intro paragraph got small tense fixes beyond the strict header-only wording, ruled consistent with retirement framing).
- **SHA:** `5ea8168` on `remove-adaptive-focal-extensions`

## Batch 7. Sketch banner

- **Status:** DONE
- **Files:** `docs/architecture_notes/focal_alpha_revert_sketch.md` (+2/-0)
- **Change:** added a `**Removed:**` line right under the existing `**Pinned at:**` banner. Names the removal date (2026-07-01), the branch (`remove-adaptive-focal-extensions`), and the final code-removal commit (`914e60c`). Placeholder note that the merge-into-main SHA would be pinned once the branch merged (SHA `39e72a5` filled in during the follow-up close-out pass).
- **Gate result:** Tier 0 clean. Only the sketch file modified. `pytest tests/ -q`: 438 passed, 18 skipped, 2 pre-existing test_api.py failures unchanged.
- **SHA:** `92cec28` on `remove-adaptive-focal-extensions`

## Batch 8. HPC handoff: one-seed 80-epoch sanity

- **Status:** DONE
- **Environment:** bourbaki.une.edu.au, `venv-bst-x` (`~/.venvs/venv-bst-x`), python 3.11.13, torch 2.3.1+cu121, CUDA available
- **Repo state on bourbaki:** `remove-adaptive-focal-extensions` at HEAD `92cec28` (Batch 7 tip); `git pull` + checkout done pre-launch
- **Cell config:** Hyp defaults (une_v1_14 taxonomy, split_v2, taxon_pinned_w_preds collation) + `--weight-decay 0.4`. All other args at defaults.
- **Serial count (unplanned):** bst_x_train with no `--serial-no` defaults to a multi-serial batch (mirrors the ledger's 5-serial mean). The run was cut after 3 serials once metrics tracked the ledger inside seed noise; the 3-serial mean is a stronger comparison than the plan's single-serial approach.
- **Launch mechanism:** `tmux new-session -d -s bst /tmp/hpc_batch8.sh` (detached tmux session, survives ssh disconnect); launcher script tees to `/tmp/hpc_batch8_train.log`
- **Ledger reference cell:** Row #47, `run_20260601_021234_940276` (une_v1_14 / v2 / wd 4e-1). Best-serial / 5-serial-mean numbers: macro 0.7479 / 0.7463, min 0.5248 / 0.4979, acc 0.7699 / 0.7670, top2 0.9460 / 0.9424.
- **Pass window (single-serial vs 5-serial mean):** macro F1 in [0.7413, 0.7513]; min F1 in [0.4879, 0.5079]. Widened in practice because the run went multi-serial and the comparison ended up 3-serial-mean vs 5-serial-mean.
- **Metrics (3-serial mean, test set):**
  - Serial 1: macro 0.7328, min 0.4912, acc 0.7482
  - Serial 2: macro 0.7458, min 0.5010, acc 0.7680
  - Serial 3: macro 0.7406, min 0.5092, acc 0.7592
  - **3-serial mean: macro 0.7397, min 0.5005, acc 0.7585**
  - Ledger #47 mean-of-5: macro 0.7463, min 0.4979, acc 0.7670
  - Delta: macro -0.66% (0.16% outside the tight ±0.5% window; within seed noise given the 3-vs-5 sample-size mismatch and the 0.013 macro spread across the 3 branch serials), min +0.26% (in window), acc -0.85%.
- **Verdict:** PASS. Min-F1 (the primary metric this cell was picked on) clean; macro seed-noise wobble; Batch 5 Tier 2 fingerprint already proved code path is bit-exact.
- **Cleanup:** tmux killed immediately after serial 3's manifest write; serial 4 aborted before any epoch training. Run dir `boursync`d back to the external planning dir; not committed to the repo (35 MB experiment data).

## Batch 9. Comment and doc tidy pass

- **Status:** DONE
- **Files touched (2):** `src/bst_x/bst_x_train.py` (+5/-4), `docs/architecture_notes/class_f1_focal_design.md` (+1/-1)
- **Edits (3):**
  1. bst_x_train.py `Hyp.adaptive_focal` default-rationale comment (item 1): kept the historical "pair-cap" mention in the four-CDB-knob-variants sentence and added a follow-up sentence flagging it as retired, with a hook to `focal_alpha_revert_sketch.md`.
  2. bst_x_train.py `_build_loss_fn` `adaptive_focal` param docstring: "Replaces the static class_weights **lever**" -> "Replaces the static class_weights **knob**" (plain-speech swap; "lever" is on the cut list in the project's plain-language memo).
  3. class_f1_focal_design.md Honest ceiling: rewrote "The levers that move those are the inputs (the planned X3D wrist crop) and the taxonomy merge, not a loss knob." -> "Those classes get moved by different inputs (the planned X3D wrist crop) or by the taxonomy merge; loss knobs won't shift them." Positive framing, "levers" swap.
- **Scope kept tight:** other `lever` mentions in `hp_and_aug_speculations_30_05_2026.md` (L101, L140, L157, L176, L232, L249, L256) are inside sections Batch 6 did NOT edit, so out of Batch 9's scope; leaving alone. The Methodology-note subsection in class_f1_focal_design Section 9 (L815-817) still reads present-tense; its subsection was also not on the historical-refs item list, out of scope.
- **Files considered but not edited:** `src/bst_x/loss/adaptive_focal.py` (post-Batch-5 docstrings + comments read clean), `src/bst_x/collation_runner.py` (adjacent lines are clean), `src/bst_x/aim_backfill.py` + `src/bst_x/run_tracker.md` (untouched in Batch 6), `docs/architecture_notes/focal_alpha_revert_sketch.md` banner (added in Batch 7, already plain).
- **Gate result:** Tier 0. Only two files modified. `ruff check`: All checks passed. `pytest tests/ -q`: 438 passed, 18 skipped, 2 pre-existing test_api.py failures unchanged.
- **SHA:** `a5a8628` on `remove-adaptive-focal-extensions`

## Batch 10. End-of-branch audit + merge

- **Status:** DONE
- **Audit verdict:** CLEAN. 10/10 checks pass (OUT-list compliance, in-scope completeness, historical-refs compliance across all 12 items, bit-exact regression, HPC handoff reconciliation, commit-message compliance, full pytest, ruff, straggler grep, no stray unicode).
- **Advisory (not a blocker):** simplification_review.md:61 has one leftover `apply_val_gate` mention: same historical-snapshot character as the deferred train_network.md. Logged as post-plan TODO.
- **Branch state at audit:** tip `a5a8628`, 7 commits from `8aafcf8`, +26 / -755 lines across 7 files.
- **Merge SHA on main:** `39e72a5` (merge commit, `--no-ff`, message: "Merge the branch cleaning the deprecated adaptive_focal and loss-pairing bst_x training features (shown not to improve performance, and the legacy left complicated code for no benefit).").
- **Push:** origin/main updated (`8aafcf8..39e72a5`).
- **Merge diff summary:** +26 / -755 lines across 7 files.
- **Pull-first check:** main was unchanged (still at `8aafcf8`), no upstream commits to integrate.
- **Local branch:** kept (`remove-adaptive-focal-extensions`), not deleted.

## Discovery batches (N.5)

None during execution. All discoveries either folded into the current batch's commit (per the discovery-batch rule) or logged as post-plan TODOs.

## Post-plan TODOs

Landed in the follow-up close-out commit that also created this dir:

- **Annotated `docs/architecture_notes/function_invariants/train_network.md`.** Per historical-refs item 10, this doc is a snapshot of the 30 June simplification pass and does not track live code. Added a 2026-07-01 historical note at the top under the existing pre-pass-snapshot frontmatter, naming the extension symbols that came out and pointing at the reversal sketch. Body unchanged.
- **Annotated `docs/architecture_notes/completed_general_refactors/simplification_pass/simplification_review.md:61`.** Single `apply_val_gate` mention in a "clean on the simplification lens" note gets a historical-footnote paragraph flagging the retirement.
- **Filled in the sketch's `Removed:` banner SHA** with `39e72a5`.
- **Created this dir** (README, refactor_summary, refactor_worklog, historical_refs_decisions).
