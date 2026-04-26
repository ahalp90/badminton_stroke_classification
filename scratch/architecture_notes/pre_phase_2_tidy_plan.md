# Pre-Phase-2 Tidy: refactor record + reviewer brief

**Status:** Merged to `main` as `e0ffeec` on 2026-04-26 via `git merge --no-ff` after a memory-isolated three-agent review returned READY_TO_MERGE with zero blockers. The pre-merge doc-fix commit `4b43546` and the merge commit `e0ffeec` are the closing entries on this work.
**Branch:** `pre-phase-2-tidy` (final tip: `4b43546`; merge commit on `main`: `e0ffeec`).
**Cut from:** `main` at `d4fd644` ("Ablation collapsing top/bottom classes…").
**Diff vs main:** 29 commits, 68 files changed, +3,585 / −1,499 LOC.
**Source-and-test diff vs main:** 29 .py files, +447 / −6,673 LOC (most of the deletions are dead-code excision that landed in step 5; new code is small).
**Date of latest revision:** 2026-04-26.

---

## 0. Reviewer brief — READ THIS FIRST

> **Closed 2026-04-26.** The three-agent review ran, returned READY_TO_MERGE with zero blockers, and the branch merged to `main` as `e0ffeec`. The brief below is preserved verbatim as the historical record of how the review was framed.

You are one of a parallel team of agents brought in to give this branch a final, memory-isolated review before merge. The user (Ariel) and the in-session author have already done substantial internal verification (see §2 below); this review is the independent check that a fresh reader would catch what we missed.

### Why this review exists

`main` was a known-working tip. The refactor is large (28 commits, 6.7k LOC removed). Pytest + bit-exact gates passed, but pytest does not exercise the full training loop, the MMPose runtime, or the shuttle pipeline. The user wants belt-and-braces confidence that no behavioural regression slipped through, *and* a clear-eyed audit of whether the refactor met its original goals before the next phase begins.

### Your role (one of three threads)

The coordinating agent will assign you exactly one of these. Do only your assigned role.

1. **REGRESSION_HUNTER** — walk the `main..pre-phase-2-tidy` diff and surface any place a behavioural regression could have crept in. Scope priorities: (a) anywhere a function body changed, not just moved; (b) anywhere a same-named symbol was deleted from one file and added to another (lift); (c) anywhere a default argument, exception type, or import order changed; (d) anywhere a "no-op refactor" claim cannot be verified from the diff alone. Pytest already covers the public dataset/loader/BST_0 forward path on real data when `BST_DATA_DIR` is set; you do not need to re-run it.
2. **GOALS_AUDITOR** — read `scratch/architecture_notes/pre_phase_2_review_2026-04-26.md` (the original review that motivated this branch) and map each of its six focus areas against what landed. For each focus area: did the refactor address it? Partially? Not at all? Was the chosen approach the right one in retrospect? Then read §3 below ("Open items going into the next phase") and assess whether anything on it should actually move *before* merge.
3. **DOC_COVERAGE_AUDITOR** — sanity-check the doc/test surface. Is anything claimed in the docs (this plan doc included) contradicted by the code? Are there changes in the diff that are not reflected anywhere user-facing? Are the verification gates listed in §2 below sufficient — or is there a meaningful surface they don't cover that a small additional test would close cheaply?

### Operating constraints (all roles)

- **Read-only.** Do not edit code, do not commit, do not run training jobs. You may run pytest if helpful; you may run `git log`, `git diff`, `grep`, and read any file.
- **File:line citations on every claim.** "There's a bug somewhere in `bst_train`" is not actionable; "`bst_train.py:312` the loss tensor is moved off-device before backward, see diff" is.
- **Forced verdicts, no fence-sitting.** Every finding gets one of:
  - `REGRESSION_RISK` (with severity: BLOCKER / SHOULD_FIX / NICE_TO_FIX) — for REGRESSION_HUNTER and DOC_COVERAGE_AUDITOR.
  - `GOAL_MET` / `GOAL_PARTIAL` / `GOAL_MISSED` — for GOALS_AUDITOR.
  - `BEFORE_MERGE` / `BEFORE_PHASE_2` / `DEFER` — for any action you propose.
- **No new abstractions.** If you'd recommend a new helper or class, say "would be nice in phase 2" rather than "must add now". The branch is closed for refactor scope; we want known-working merged.
- **No memory of prior conversations.** You do not have access to the user's chat history with the in-session author; you have what's in this repo. Treat anything not in the repo as not existing.

### Required reading before you start

1. The rest of this document (sections 1-3, plus the appendix per-step plan if you need granularity).
2. `scratch/architecture_notes/pre_phase_2_review_2026-04-26.md` — the original review whose six focus areas drove this branch.
3. `git log main..pre-phase-2-tidy --stat` for commit-level scope; `git diff main..pre-phase-2-tidy -- <file>` for any file you want to inspect closely.
4. `.claude/project_overview.md` for project context (taxonomies, splits, hardware, team).

### Output format

A single markdown document with:

```
# <ROLE> review — pre-phase-2-tidy

## Summary
- One paragraph: confidence level, top concern, top achievement.

## Findings
### Finding 1: <one-line title>
- Verdict: <one of the labels above>
- Evidence: <file:line refs, short snippets if helpful>
- Suggested action: <one sentence + the priority label>

### Finding 2: ...

(...)

## Verdict on the merge question
One of:
- READY_TO_MERGE — no blockers, optional follow-ups listed.
- BLOCKERS_TO_RESOLVE — N specific items below must close first.
- CANNOT_DETERMINE_WITHOUT — list the artefacts/runs needed to decide.
```

Keep it terse. Triple-bullet evidence per finding is plenty; do not narrate.

---

## 1. What landed (refactor summary)

Steps 1–10 (planned) + step P (proper-packages refactor) + step Q (lint-debt cleanup) all landed on `pre-phase-2-tidy`. The detailed per-step plan is preserved verbatim in the appendix; this section is a one-line-per-step recap for the reviewer.

| # | Commit | One-line summary |
|---|---|---|
| 1 | `db11f93` | Doc drift sweep: refresh Hyp defaults, taxonomy lists, heuristics description, manifest example, line cites. |
| 2 | `17ab5c4` | Add `historical_bst.md` skeleton + `scratch/project_history/` + `scripts/archive/` directories. |
| 3 | `342a573` | `git mv` `src/bst_refactor/deprecated/`, `ShuttleSet/deprecated/`, `main_on_shuttleset/tmp/` into `scratch/project_history/`. |
| 4 | `234e5b8` | Capture `TemPose_*`, original `Hyp` defaults, LR/aux rationale, removed dataset classes, `compare_pred_gt_on_specific_type` verbatim into `historical_bst.md`. |
| 5 | `66e7c2a` | Drop dead BST code (~990 LOC): four `TemPose_*` variants, three orphan dataset classes + their loaders, debug helper. No callers anywhere. |
| 5b/8 | `bdbdaed` | Lift shared per-clip iteration into `_prepare_dataset_from_raw_video`; lift collated-dir naming into `pipeline.config.derive_npy_collated_dir_basename`; collapse the `bst_train` model_info builder. |
| 5c | `d6ae8df` | Extract `bst_common.py` (`MODELS`, `Tee`, `build_bst_network`, `compute_data_provenance`); `bst_train` and `bst_infer` now share one source of truth. |
| 6 | `c6d962d` | Add 7 `tests/test_sticky_anchor.py` invariant tests (Voronoi, Bottom-first, sitting tiebreaker, rally presence, EMA reset, mixed pick, update gate). |
| 7 | `9af521e` | `StickyAnchorParams` frozen dataclass: collapse three triplicated default-value definitions into one. |
| 9 | `ad9cd15` | `git mv` completed-phase scripts into `scripts/archive/`; flag `scripts/example_mlflow_run.py` as a delivery-review TODO. |
| 10 | `f4c7bec` | Move LR/aux rationale paragraphs out of `bst_train.py` into `arch_1_directions.md` + `historical_bst.md`. |
| — | `57655aa` | Pre-existing test-failure fix: drop unused `mediapipe` import; auto-detect `pose_style` in `test_integration`. |
| — | `af1a551`, `c5676dc` | Found running the gate: consolidate `n_bones` as a single source of truth in `build_bst_network`; rename for consistency. |
| — | `412f6e5`, `25e0308`, `248f540` | `scratch/post_tidy_smoke/` bit-exact verification scripts (later superseded by step P). |
| P | `fd12cd8` | Proper-packages refactor: 3 new `__init__.py`, 7 `sys.path.append` blocks dropped, 3 imports converted to package-style. New invocation: `PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification python -m main_on_shuttleset.bst_train`. |
| Q | `c29e97c` | Lint-debt cleanup: explicit `load_repo_dotenv()`, narrowed two `BLE001` excepts, lifted two PLC0415 imports to module top, kept two with justifying comments. |
| docs | `e9a1c7d`, `ee664e5`, `7daa319`, `04f0ecb`, `4e5cb3a`, `e811ffa`, `88ee24e` | Plan-doc updates and a two-pass post-refactor doc-drift sweep across user-facing markdown. |
| review | `dc73653` | Top-load `pre_phase_2_tidy_plan.md` with the reviewer brief for the memory-isolated three-agent merge-readiness review. |
| review | `4b43546` | Pre-merge doc fixes from the parallel-agent review: surface `test_sticky_anchor.py` + `test_data_access.py` in the user-facing test docs, add `une_merge_v1_nosides` to the `data_pipeline` taxonomy choice lists, surface the `Task`-class lift in §3 deferred items. |
| merge | `e0ffeec` | `git merge --no-ff pre-phase-2-tidy` onto `main`. Headline summary in the merge commit body. |

Net: −6,673 LOC of source/tests deleted, +447 LOC of source/tests added (mostly `tests/test_sticky_anchor.py` and `bst_common.py`). Roughly −5,500 LOC of historical content + scripts moved into `scratch/project_history/` and `scripts/archive/`.

---

## 2. Verification that passed (in-session)

| Check | Where | Result |
|---|---|---|
| Byte-identity heuristic gate (50-clip hit-zone sample) | engelbart V100 | ✅ 50/50 stems exact, max abs diff 0.0 |
| `pytest tests/` with `BST_DATA_DIR` set | engelbart, post-step-P and post-step-Q | ✅ 43/43 (pytest wall-time dropped ~11s → ~9s after step P) |
| 2-epoch smoke train, post-tidy vs main | engelbart, runs `run_20260426_115321` vs `run_20260426_120039` | ✅ within run-to-run noise; `manifest.config:` and `data_provenance.npy_collated_dir` byte-identical |
| `bst_infer` bit-exact (smoke_infer_bit_exact.py) | engelbart with `CUBLAS_WORKSPACE_CONFIG=:4096:8` | ✅ 4202/4202 predictions IDENTICAL between post-tidy and main |
| `prepare_2d` bit-exact (line-level diff review) | local | ✅ structural-only refactor; per-line diff confirms no behavioural delta possible |
| `pytest tests/` (laptop, no `BST_DATA_DIR`) | post-step-P, post-step-Q | ✅ 42 passed, 1 skipped |
| `ruff check` over step-Q-touched files | local | ✅ all checks passed |

What pytest does NOT cover: full multi-epoch training loop, MMPose runtime path (depends on the mmpose stack not installed locally), TrackNetV3 shuttle extraction, the Aim UI integration, the heuristic dispatch on real raw extracts (only the byte-identity gate exercises that, and only once). Reviewers should consider whether any of those gaps matter for the merge decision.

---

## 3. Open items going into the next phase

These were explicitly deferred during the refactor. Now that the branch has merged to `main` as `e0ffeec`, this list is the carry-over backlog for the next substantive piece of project work (the X3D-S wrist-crop layer or the path/IO sweep that precedes it).

- **Branch destination decision.** Resolved 2026-04-26: merged to `main` as `e0ffeec` via `git merge --no-ff` after the three-agent review returned READY_TO_MERGE.
- **Path/IO abstraction** (focus area 5 in the original review). Reserved for after-X3D-S. Folded into this: collapse three near-duplicate root constants into one source of truth. The repo currently has `bst_train.py:44 REPO_ROOT = Path(__file__).resolve().parents[4]` (actual repo root, used at `:45` for `notebooks/clips_master.csv`), `pipeline/config.py:15 PROJECT_ROOT = Path(__file__).resolve().parent.parent` (intentionally `src/bst_refactor/`, anchoring `ShuttleSet/` data dirs at `config.py:17-23`), and `pipeline/data_access.py:151 _PROJECT_ROOT = Path(__file__).resolve().parents[3]` (actual repo root, used for `.env` and `notebooks/clips_master.csv`). Aliasing `pipeline.config.PROJECT_ROOT` as `REPO_ROOT` would break `clips_master.csv` resolution because `PROJECT_ROOT` is `src/bst_refactor/`, not the repo root. Correct fix: add `REPO_ROOT = Path(__file__).resolve().parents[2]` to `pipeline/config.py` (parents[2] from `src/bst_refactor/pipeline/config.py` is the actual repo root); then `from pipeline.config import REPO_ROOT` in `bst_train.py:44` (drops the magic `parents[4]`) and replace `_PROJECT_ROOT` in `data_access.py:151` with the same import. Defer the code change until X3D-S forces the broader path/IO sweep.
- **`Task`-class lift into `bst_common.py`.** Original review action 1 named `Task` as part of the lift; `MODELS` / `Tee` / `build_bst_network` / `compute_data_provenance` landed in step 5c, but `Task` stayed split between `bst_train.py:438` (references module-level `hyp` at `:444,462,479`) and `bst_infer.py:49` (a much-simpler stand-in). Defer until X3D-S's `Task` shape is visible — the lift can take that into account.
- **`prepare_train_on_shuttleset.py` full module split** into `mmpose_extract.py` + `homography.py` + `collate.py`. Light tidy landed in step 5b; the structural split is reserved for after X3D-S.
- **Validation script triplet shared core** (focus area 6, site 5).
- **Bulk style passes:** AU/UK rename for `normalize_*` and "labeled"/"vectorized"; em-dash sweep; "fade" → "anneal"/"downtune".
- **`aim_backfill._derive_tags` parameterisation.**
- **`pipeline/build_dataset.py:79-139` `dry_run()` consolidation.**
- **`pipeline/clip_index.py` docstring trim.**
- **`scripts/example_mlflow_run.py`** stays in place with a delivery-review TODO; gets deleted before delivery if Scott has not picked up MLflow.

---

## 4. Wire-in invariant for X3D-S (still load-bearing)

The X3D-S wrist-crop layer (Architecture 1, phase 4 of the build plan) wires into the BST model internals at five points. These were read-only across this entire branch and remain so:

- `bst.py:106-110` `CrossTransformerLayer` signature and attention plumbing.
- `bst.py` `BST_CG_AP` forward graph: token sequencing, positional embeddings, `d_model=100`/`d_head=128`/`n_head=6` defaults, per-stream embedding heads (pose, shuttle, position).
- `bst.py:28` building-block imports (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`).
- `bst.py:433-437` the five `BST_*` partials.
- No new abstraction layer between `bst.py` and the train loop. `bst_common.py` lifts `MODELS` / `build_bst_network` / `Tee` / `compute_data_provenance` only; direct `MODELS[...](**kw)` instantiation stays.

If a reviewer flags any of those five points as having drifted, that is a BLOCKER finding regardless of pytest status.

---

## Appendix — original execution plan (kept verbatim as a record)

Everything below this line is the original step-by-step plan as it stood when the refactor was executed. Read it for granularity on what each step touched, the safety checks each step ran, and the rationale for in-flight decisions. Section 1 above is the recap; this is the source.

---

## Execution status (2026-04-26, post-step-Q)

All 12 planned commits + steps P and Q land on `pre-phase-2-tidy`. Refactor verified end-to-end on both laptop and engelbart:

| # | Commit | Step |
|---|---|---|
| 1 | `db11f93` | Step 1 — Doc drift sweep |
| 2 | `17ab5c4` | Step 2 — Historical doc skeletons + archive dirs |
| 3 | `342a573` | Step 3 — Relocate src/-tree historical/deprecated trees |
| 4 | `234e5b8` | Step 4 — Capture excised content into historical_bst.md |
| 5 | `66e7c2a` | Step 5 — Drop dead BST code (~990 LOC) |
| 6 | `bdbdaed` | Step 5b + 8 — `_prepare_dataset_from_raw_video` lift + collated-dir naming helper + model_info collapse |
| 7 | `d6ae8df` | Step 5c — `bst_common.py` extraction |
| 8 | `c6d962d` | Step 6 — sticky_anchor unit tests (7) |
| 9 | `9af521e` | Step 7 — `StickyAnchorParams` dataclass collapse |
| 10 | `ad9cd15` | Step 9 — Script archive sweep |
| 11 | `f4c7bec` | Step 10 — `bst_train.py` configuration block tidy |
| — | `57655aa` | Follow-up — Stop pre-existing test failures (drop unused mediapipe; auto-detect pose_style) |
| — | `cb963d2` | TEMP smoke harness (n_epochs=2, single seed) — used for the gate |
| — | `af1a551` | Bug fix found running the gate — single source of truth for `n_bones` |
| — | `c5676dc` | Rename `n_trailing_bone_channels` → `n_bones` for consistency |
| — | `d19664d` | Revert of `cb963d2` — production Hyp values restored |
| — | `412f6e5` | Add `scratch/post_tidy_smoke/` bit-exact verification scripts |
| — | `25e0308` | Smoke-script sys.path fix (workaround; superseded by step P) |
| P | `fd12cd8` | Step P — Proper-packages refactor (see section below) |
| Q | `c29e97c` | Step Q — Lint-debt cleanup (see section below) |

### Remote gate (engelbart) status

| Check | Status | Evidence |
|---|---|---|
| Byte-identity gate (50-clip hit-zone) | ✅ PASS | 50/50 stems exact, max abs diff 0.0 on `_pos`/`_joints` |
| `pytest tests/` with `BST_DATA_DIR` set | ✅ PASS | 43/43 (after `57655aa` fixed pre-existing env mismatches); re-confirmed 43/43 on engelbart post-step-P (`fd12cd8`) and post-step-Q (`c29e97c`). Engelbart wall-time dropped ~11s → ~9s after step P, consistent with regular-package imports skipping the namespace-package fallback that was triggered by the absent `__init__.py` files. |
| 2-epoch smoke train comparison | ✅ PASS | `run_20260426_115321` (post-tidy) vs `run_20260426_120039` (main); curves within run-to-run noise; manifest `config:` and `data_provenance.npy_collated_dir` byte-identical |
| `bst_infer` bit-exact (smoke_infer_bit_exact.py) | ✅ PASS | 4202/4202 predictions IDENTICAL between post-tidy and main (run on engelbart 2026-04-26 with `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Re-run not needed for steps P/Q — both are import-only / lint-only and cannot perturb the forward pass. |
| `prepare_2d` bit-exact (line-by-line diff review) | ✅ PASS | Per-line behavioural diff between pre-tidy and post-tidy `prepare_2d`/`prepare_3d`/`_prepare_dataset_from_raw_video` confirms bit-exact-by-construction: same iteration order, same resume marker, same kwargs forwarded to `detect_players_2d`/`3d`, same save order, same gc/empty_cache cadence. Pure structural deduplication, no behavioural delta possible. The `smoke_prepare_2d_bit_exact.py` GPU-runtime check is therefore redundant. |

### Branch destination

Resolved 2026-04-26: option 2, merged to `main` directly as `e0ffeec` via `git merge --no-ff`. Three-agent review (REGRESSION_HUNTER / GOALS_AUDITOR / DOC_COVERAGE_AUDITOR) returned READY_TO_MERGE with zero blockers; two `SHOULD_FIX` doc items were closed pre-merge in `4b43546` (test inventory + taxonomy choice list + `Task`-lift surfaced in §3). Push to `origin/main` was performed manually by the user.

---

## Branch destination (history)

**Original stance during execution:** sit on the `pre-phase-2-tidy` branch until local tests pass, then engelbart tests, then user decides. No push, no PR, no merge to main without explicit go-ahead. (Followed verbatim through steps 1–Q. Branch was pushed to origin once the user authorised it for engelbart pulls. Merged to `main` as `e0ffeec` on 2026-04-26.)

## X3D-S wire-in invariant

The X3D-S wrist-crop layer fuses into the **BST model internals**. The following points are read-only across this entire branch. They are where the new module wires in, not external orchestration surface.

- `bst.py:106-110` `CrossTransformerLayer` signature and attention plumbing.
- `bst.py` `BST_CG_AP` forward graph: token sequencing, positional embeddings, `d_model=100`/`d_head=128`/`n_head=6` defaults, per-stream embedding heads (pose, shuttle, position).
- `bst.py:28` building-block imports (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`). Step 5 deletes the four standalone `TemPose_*` classes; the building blocks themselves stay verbatim in `tempose.py`.
- `bst.py:433-437` the five `BST_*` partials.
- No new abstraction layer between `bst.py` and the train loop. `bst_common.py` lifts `MODELS` / `Task` / dataloader helpers only; direct `MODELS[...](**kw)` instantiation stays.

Internal refactor of `bst_train.py`, `bst_infer.py`, `prepare_train_on_shuttleset.py`, `shuttleset_dataset.py`, etc. is fine. The BST model graph itself is not refactored on this branch; this branch only addresses sprawl and dead code in the surrounding scaffolding.

---

## Branch strategy

- New branch `pre-phase-2-tidy` cut from `main` at the current HEAD (`d4fd644`).
- One commit per logical step, smallest blast radius first.
- Branch is **local-only until you say otherwise**. No `git push`. No PR.
- If any step fails its safety check, stop and ping. We do not "fix forward" within auto-mode; we revert the offending commit and reassess together.

---

## Per-step protocol

```
1. Confirm pytest is green on the current commit.
2. Apply the step (file edits, deletes, moves).
3. Run the step's safety checks (pytest + targeted greps + import smokes).
4. If green, commit with a single-sentence message in repo style.
5. If red, do not commit; report the failure and stop.
```

---

## Local safety checks (what I run at each step)

These all run without HPC, without `BST_DATA_DIR` set, and without GPU.

1. **`pytest`** — runs `tests/test_environment.py`, `tests/test_dataset.py`, `tests/test_api.py`; `tests/test_integration.py` auto-skips when `BST_DATA_DIR` is unset. Baseline must stay green throughout.
2. **Targeted import smoke** — for steps that touch importable modules, a one-liner Python invocation that imports the affected entry points and instantiates the high-level objects (`Task`, `MODELS`, dataset classes). Catches surface-level breakage that pytest doesn't reach.
3. **Grep verification before deletes** — for every name being deleted, `grep -rn` across `src/`, `scripts/`, `tests/`, `notebooks/`. Zero hits outside the file itself before deletion proceeds.
4. **`ruff check`** on touched files only. Scope is set in `pyproject.toml`; existing exclusions for `notebooks/`, `**/deprecated/`, `TrackNetV3/` honoured.

If a step adds new tests (sticky_anchor), those tests join the local checks from that commit onward.

---

## Remote-enabled checks (you run at the end, on engelbart)

These belong on the V100 with the npy data accessible. I will leave a one-page handoff appended to this doc when the local branch is ready.

1. **Byte-identity gate.** Run `failsafe_bst_mmpose_zeroing_check_equivalence` on the 50-clip hit-zone sample. Confirms the BST/MMPose path is bit-identical to the committed extract. Required for any commit that touches `apply_heuristic.py`, `heuristics/`, `prepare_train_on_shuttleset.py` MMPose code, or imports thereof.
2. **`pytest tests/test_integration.py`** with `BST_DATA_DIR` pointed at the active ablation-tagged collated dir. Confirms the dataset->loader->BST_0 forward path still works end-to-end.
3. **2-epoch smoke train.** `python bst_train.py` with `n_epochs=2`, `early_stop_n_epochs=999`, single seed. Compare loss/accuracy curves against a fresh pre-tidy baseline run done on `main` before branching. Curves should match within run-to-run noise.

If all three pass on engelbart, you ping me and we resolve the open question above.

---

## Pre-execution baseline capture

Before cutting the branch I will:
1. Confirm `git status` clean.
2. Run `pytest` on `main` and record the full output.
3. Note the current HEAD SHA.
4. Capture a `git log --stat` snapshot of the files about to be touched (so anything unexpected on the branch is easy to spot in review).

You will additionally want to do the **2-epoch baseline run on `main` before I cut the branch** so the post-tidy smoke train has something to compare against. I'll flag this in the handoff section once I'm ready.

---

## Step-by-step plan

Order is smallest blast radius first. Steps 1-7 are bundled into one local-only branch. Steps 8+ are reserved for after-X3D-S in the review doc and are not part of this execution.

### Step 1 — Doc drift fixes

**Touches:** Markdown only.

- `.claude/project_overview.md:94-96`: rewrite the heuristics description (no `HeuristicFilter` class hierarchy exists; `apply(raw, ctx, **kw) -> HeuristicOutput` registered in `heuristics/__init__.py:REGISTRY`).
- `src/bst_refactor/data_pipeline_to_model_train.md:392`: rewrite `Hyp` defaults table to reflect live `bst_train.py:140-157` (`n_epochs=80`, `taxonomy='une_merge_v1_nosides'`, `aux_fade_end_epoch=15`, etc.).
- `src/bst_refactor/data_pipeline_to_model_train.md:121, 535` and `src/bst_refactor/pipeline/README.md:165, 247`: add `une_merge_v1_nosides` to taxonomy lists.
- `src/bst_refactor/data_pipeline_to_model_train.md:260-263`: remove `Dataset_npy_collated_one_side` and `_single_pose` from "primary classes" listing (or mark as orphaned-pending-deletion).
- `src/bst_refactor/run_tracker.md:64-90`: add `extra: data_provenance: {clips_csv_path, clips_csv_sha256, effective_ablation_id, npy_collated_dir}` to the manifest format example.
- `scratch/architecture_notes/arch_1_directions.md:101`: refresh `bst_train.py` line refs (cosine scheduler now at `:395-400`).
- `README.md:100-104`: list the actual contents of `tests/` (`test_api`, `test_data_access`, `test_dataset`, `test_environment`, `test_integration`).

**Safety checks:** `pytest`. (Doc-only, but runs as a sanity gate on the working tree.)

**Commit message draft:** "Doc drift sweep: refresh Hyp defaults, taxonomy lists, heuristics description, manifest example, line refs."

---

### Step 2 — Create historical doc skeletons + script archive directory

**Touches:** New empty/skeleton files, new directories.

- `scratch/architecture_notes/historical_bst.md` — already drafted as a skeleton in this same session (see below). Will be filled in steps 3 and 4 as content is excised.
- `scratch/architecture_notes/pipeline_context_notes.md` — light skeleton; filled if any pipeline-area context notes are excised in later steps.
- `scratch/project_history/` — new directory for the relocated `deprecated/` and `outdated_*` content. Directory created with a `README.md` explaining what lives there.
- `scripts/archive/` — new directory with a `README.md` explaining that everything inside is one-shot tooling kept for reference, not part of the active build.

**Safety checks:** `pytest`. New directories don't affect imports.

**Commit message draft:** "Add historical doc skeletons and archive directories for the tidy pass."

---

### Step 3 — Move historical/deprecated content out of `src/`

**Touches:** Moves only; no source-code edits.

- `src/bst_refactor/deprecated/` → `scratch/project_history/bst_refactor_deprecated/`
- `src/bst_refactor/ShuttleSet/deprecated/` → `scratch/project_history/shuttleset_deprecated/`
- `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/` → `scratch/project_history/main_on_shuttleset_tmp/`

Each move uses `git mv`. The five `outdated_*.md` and `historical_*.md` files inside `src/bst_refactor/deprecated/` come along for the ride.

`scratch/project_history/README.md` records the original locations and the date of the move so a future report can reconstruct the layout.

**Safety checks:**
- `pytest`.
- `grep -rn "from .*deprecated" src/ scripts/ tests/` returns zero hits in active code (the only allowed hits are inside the moved directories themselves, which are now under `scratch/`).
- `grep -rn "main_on_shuttleset/tmp\|main_on_shuttleset\.tmp" src/ scripts/ tests/` returns zero hits.

**Commit message draft:** "Relocate src/-tree historical and deprecated trees into scratch/project_history/."

---

### Step 4 — Capture excised content into `historical_bst.md` (pre-deletion fill)

**Touches:** `scratch/architecture_notes/historical_bst.md`, `scratch/architecture_notes/arch_1_directions.md`. No source-code deletes yet.

Before deleting anything from `tempose.py`, `shuttleset_dataset.py`, or `bst_train.py`, capture:

- The four `TemPose_*` standalone classes' purpose (one paragraph each, what they did, why they were preserved through phase 0, and the verbatim source preserved as a code block in case future reproduction needs them).
- The original BST `Hyp` namedtuple defaults (`n_epochs=1600`, etc.) verbatim from the commented-out block at `bst_train.py:85-101`.
- The "LR-SCHEDULE RETUNE 2026-04-17" and "AUX-SCHEDULE 2026-04-18" rationale blocks verbatim. Trim to one-line pointer in `bst_train.py`; full block lives in `historical_bst.md`.
- The `Dataset_npy`, `Dataset_npy_collated_one_side`, `Dataset_npy_collated_single_pose` class headers and what they were for (one paragraph each), plus the verbatim source as a code block.
- The `compare_pred_gt_on_specific_type` debug method (verbatim).
- The `normalize_joints` `center_align=False` upstream default plus the apologia paragraph.

`arch_1_directions.md` gains a "current LR + aux schedule" subsection (one paragraph distillation of the active config) with a cross-link to `historical_bst.md` for the dated history.

**Safety checks:** `pytest`. Doc-only.

**Commit message draft:** "Capture excised BST history into historical_bst.md ahead of dead-code deletion."

---

### Step 5 — Delete dead BST code and orphan datasets

**Touches:** `tempose.py`, `shuttleset_dataset.py`, `bst_train.py`, `bst_infer.py`, `data_pipeline_to_model_train.md`.

- `src/bst_refactor/model/tempose.py`: delete `TemPose_V` (156-258), `TemPose_PF` (261-396), `TemPose_SF` (399-526), `TemPose_TF` (529-667). Keep `TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`, the helper functions, and the `__main__` smoke check. Drop the unused `from torchinfo import summary  # noqa: F401`. Roughly 710 LOC down to ~200.
- `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py`: delete `Dataset_npy` (142-246), `Dataset_npy_collated_one_side` (351-420), `Dataset_npy_collated_single_pose` (423-497), `prepare_npy_loaders` (500-531), `prepare_npy_collated_one_side_loaders` (568-599), `prepare_npy_collated_single_pose_loaders` (602-633). Update module docstring/header to reflect the trimmed surface.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`: delete `Task.compare_pred_gt_on_specific_type` (706-733) and the `Dataset_npy` import. Delete the commented-out `Hyp` block (85-101) and the commented-out scheduler (389-394).
- `src/bst_refactor/data_pipeline_to_model_train.md:260-263`: drop the `_one_side` / `_single_pose` references entirely now that they're gone.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_infer.py`: confirm the import surface matches the trimmed `shuttleset_dataset.py` and `bst.py`.

**Safety checks:**
- `grep -rn "TemPose_V\|TemPose_PF\|TemPose_SF\|TemPose_TF" src/ scripts/ tests/ notebooks/` returns zero hits outside `historical_bst.md`.
- `grep -rn "Dataset_npy_collated_one_side\|Dataset_npy_collated_single_pose\|prepare_npy_loaders\|prepare_npy_collated_one_side_loaders\|prepare_npy_collated_single_pose_loaders" src/ scripts/ tests/ notebooks/` returns zero hits outside `historical_bst.md`.
- `grep -rn "Dataset_npy[^_]" src/ scripts/ tests/ notebooks/` (negative lookahead for `_collated`): zero hits outside `historical_bst.md`.
- `grep -rn "compare_pred_gt_on_specific_type" src/ scripts/ tests/ notebooks/`: zero hits outside `historical_bst.md`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.preparing_data.shuttleset_dataset import Dataset_npy_collated, get_bone_pairs, POSE_BONE_MULTIPLIER; from src.bst_refactor.model.bst import BST_CG_AP; from src.bst_refactor.model.tempose import TCN, MLP, MLP_Head, FeedForward, TransformerEncoder; print('ok')"`.
- `pytest`. `tests/test_dataset.py` and `tests/test_integration.py` (auto-skip locally) both depend on `Dataset_npy_collated` only; they should remain green.

**Commit message draft:** "Drop dead BST code: TemPose variants, orphan datasets, compare-pred debug helper. ~850 LOC, no callers."

---

### Step 5b — `prepare_train_on_shuttleset.py` tidy (scope: light)

**Touches:** `prepare_train_on_shuttleset.py` only.

This step is folded into the branch at the user's direction. Scope is deliberately limited to (a) and (b) below; the full module split into `mmpose_extract.py` + `homography.py` + `collate.py` is reserved for after-X3D-S.

- (a) **Lift the per-clip iteration loop**. `prepare_2d_dataset_npy_from_raw_video` (lines 533-...) and `prepare_3d_dataset_npy_from_raw_video` (lines ...-657) share roughly 80% of their body: same iteration, same resume check, same gc-on-success path, diverging only in which detector is invoked and the joint dimensionality. Lift into one `_prepare_dataset_from_raw_video(detect_fn, joint_dim, ...)` helper; the two public functions become thin wrappers that pass `detect_players_2d` or `detect_players_3d`. The X3D-S wrist-crop layer will reuse this iteration shape.
- (b) **Collapse the `:1066-1116` mirror block**. This block mirrors `bst_train.py:744-783` (the model_info / npy_collated_dir construction). Done in lockstep with step 8 so the produced collated dir name stays byte-identical pre/post on both sides. Either factor a small shared helper or pass the collated-dir builder one way. Use whichever produces the smaller, more obvious diff.

**Out of scope:** the full module split; the homography helpers (`get_H` etc); the shuttle subprocess wrappers; any docstring rewrite of upstream-anchored functions. Those stay for after-X3D-S.

**Safety checks:**
- `pytest`.
- One-liner that constructs the npy collated dir name on a representative `Hyp` and confirms the string matches a known-good literal from a previous run's manifest.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.preparing_data.prepare_train_on_shuttleset import prepare_2d_dataset_npy_from_raw_video, prepare_3d_dataset_npy_from_raw_video, collate_npy, pad_and_augment_one_npy_video; print('ok')"`.
- Diff review: walk the two `prepare_*_dataset_npy_from_raw_video` functions and confirm the lifted helper produces the same per-clip filesystem outputs (same `_pos`/`_joints`/`_failed` filenames, same shapes, same dtypes, same iteration order).

**Commit message draft:** "Lift prepare_2d/prepare_3d shared iteration into _prepare_dataset_from_raw_video; collapse the model_info mirror block."

---

### Step 5c — `bst_common.py` extraction

**Touches:** new file `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_common.py`. Edits to `bst_train.py` and `bst_infer.py`.

This step is folded into the branch at the user's direction. The motivation, per `bst_infer.py:7-12`'s pre-existing TODO and the review doc: a third entry point (X3D-S training script) is about to land, and the shared scaffolding would otherwise be triple-copied.

- New module `bst_common.py` next to `bst_train.py` (same dir, minimum import-path churn).
- Lift into `bst_common.py`:
  - `MODELS` dict (constructed from the five `BST_*` partials).
  - The `Task` class (or its base; if Task has a small amount of train-specific state, keep that subclassed in `bst_train.py` and lift only the genuinely shared parts).
  - `Tee` stdout helper.
  - The `__main__` plumbing for run-id derivation, clips-CSV hashing, and the `track_run` call signature. The X3D-S training script will need the same.
  - Dataloader helper(s) currently inline in `bst_train.py`.
- `bst_train.py` becomes a thin entry point: import from `bst_common.py`, define the `Hyp` namedtuple (active config + the bst-train-specific knobs), wire up the run loop, call `track_run`. No copy of the lifted logic remains.
- `bst_infer.py` switches its imports to `bst_common.py` for `MODELS` and `Task`; the inference-specific path stays in `bst_infer.py`. The pre-existing TODO at `bst_infer.py:7-12` is removed (work done).

**Wire-in invariant (re-confirming):** the BST model graph itself is not touched. `MODELS[...](**kw)` stays the construction call. `Task.train()`, `Task.eval()`, etc. keep their public signatures. The X3D-S layer will be a sixth entry in `MODELS` later, not a wrapper around `bst_common`.

**Safety checks:**
- `pytest`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_common import MODELS, Task, Tee; from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_train import Hyp; print(list(MODELS.keys())); print(Hyp._fields)"` — confirms `MODELS` keys are unchanged and `Hyp` fields are unchanged.
- `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_infer import *; print('ok')"` — bst_infer still imports cleanly.
- Diff review: `bst_train.py` line count drops materially; the dropped lines are entirely accounted for by the `bst_common.py` additions.

**Commit message draft:** "Extract bst_common.py: MODELS, Task, Tee, dataloader helpers, run-id and clips-CSV plumbing. Closes the bst_infer dedup TODO."

---

### Step 6 — Sticky_anchor unit tests (write tests **before** any sticky_anchor logic touches)

**Touches:** New file `tests/test_sticky_anchor.py`. No source-code edits.

Test list (sites + invariants only — see "Sticky_anchor Tier 1 test list" below for full detail). All tests use synthetic `RawClip` arrays with hand-built bbox/score data and a synthetic court rectangle; no MMPose, no real video frames, no GPU.

Roughly 6-8 tests. Should run in well under a second.

**Safety checks:**
- `pytest tests/test_sticky_anchor.py` green.
- `pytest` overall green.

**Commit message draft:** "Pin sticky_anchor invariants: Voronoi pick, Bottom-first ordering, sitting tiebreaker, EMA reset semantics."

---

### Step 7 — Sticky_anchor hyperparameter triplication collapse

**Touches:** `apply_heuristic.py`, `heuristics/sticky_anchor.py`, possibly `heuristics/base.py`.

- Introduce `@dataclass(frozen=True) StickyAnchorParams` (location: probably `heuristics/sticky_anchor.py` next to the existing `RawClip`/`ClipContext`/`HeuristicOutput`-style types in `base.py`).
- `apply_heuristic.py:265-287` — derive argparse `add_argument` calls from the dataclass fields plus their defaults, single source of truth.
- `sticky_anchor.py:262-269` — accept the dataclass instance instead of `**hyperparams`, drop the `hyperparams.get(..., default)` triplet.
- The `apply()` signature on `sticky_anchor.py` stays compatible with the registry contract (`apply(raw, ctx, **kw)`); construction of `StickyAnchorParams` happens at the boundary.

**Safety checks:**
- `pytest tests/test_sticky_anchor.py` green (the tests pinned the invariants; they catch any pick/EMA regression introduced by the refactor).
- `pytest` overall green.
- `grep` on every removed default literal to confirm it now lives in exactly one place.
- Import smoke: instantiate `StickyAnchorParams()` with defaults and confirm field values match the prior triplicated defaults exactly.

**Commit message draft:** "Collapse sticky_anchor hyperparameter triplication into StickyAnchorParams dataclass."

---

### Step 8 — `bst_train.py:744-783` model_info / npy_collated_dir block simplification

**Touches:** `bst_train.py` only.

- Drop the dead `additional_model_info` sentinel (top of `__main__`, never set anywhere).
- Collapse the three nested string-builders for `additional_model_info` / `train_partial` / `model_info` into a small helper or tagged tuple.
- The `match` on `seq_len` only handles 30 and 100; preserve that exactly (no behaviour change).

**Safety checks:**
- `pytest`.
- Manual diff review: the produced `npy_collated_dir` string for the active `Hyp` config must be identical pre/post. I will exercise this with a one-liner Python snippet that constructs the dict on a representative `Hyp` and prints the `npy_collated_dir` against the expected literal from a previous run's manifest.
- `prepare_train_on_shuttleset.py:1066-1116` is **not touched** in this step — it mirrors the same block but is deferred to after-X3D-S in the review doc. The mirror stays in sync because we did not change the produced string.

**Commit message draft:** "Simplify model_info / collated dir construction in bst_train __main__."

---

### Step 9 — Script archive sweep

**Touches:** Moves only. No source-code edits.

- `scripts/example_mlflow_run.py`: stays in place but gains a top-of-file header comment `"""TODO (delete before delivery if Scott has not picked up MLflow integration). Stub from 2026-04-08; project settled on Aim + run_tracker."""`. README experiment-tracking paragraph gains a one-line note in the same vein.
- `scripts/test_clip_index.py` → `scripts/archive/test_clip_index.py`.
- `scripts/flatten_copy.sh` → `scripts/archive/flatten_copy.sh`.
- `scripts/verify_flatten.py` → `scripts/archive/verify_flatten.py`.
- `scripts/symlink_merge_phase1.py` → `scripts/archive/symlink_merge_phase1.py`. While moving, fix the docstring at line 14 ("Run from the repo root or from `src/bst_refactor/stroke_classification/`" → "Run from the repo root.") since the relative defaults make the dual-location claim false.
- `scripts/verify_v1_collate.py` → `scripts/archive/verify_v1_collate.py`.
- `scripts/archive/README.md` (created in step 2) describes each one-line: what it did, when, why it's archived rather than deleted.
- `README.md`: update the scripts section to list only the active scripts (`rename_videos.py`, `validate_videos.py`, `setup_data.sh`, `example_mlflow_run.py`).

**Safety checks:**
- `pytest`.
- `grep -rn "scripts/flatten_copy\|scripts/verify_flatten\|scripts/symlink_merge_phase1\|scripts/verify_v1_collate\|scripts/test_clip_index" src/ tests/ notebooks/ README.md scratch/` — any hit must be a doc reference that can be updated to the archive path or removed; no live code path may import or shell out to these.

**Commit message draft:** "Archive completed-phase scripts; flag mlflow stub for delivery review."

---

### Step 10 — `bst_train.py` configuration block tidy (final pass)

**Touches:** `bst_train.py` and `arch_1_directions.md`.

This step depends on step 4 (historical_bst.md fill) and step 5 (dead-code excision). Done last in this branch because it's the most stylistic and any merge conflict here is cheapest to resolve.

- Move LR/aux-schedule rationale paragraphs (`bst_train.py:65-157`) into `arch_1_directions.md` (current state) and `historical_bst.md` (verbatim history). Keep only a one-line cross-link in `bst_train.py` plus the live `Hyp` config.
- Delete migration-anchor comments at `bst_train.py:1-2, 53-57`.
- Delete the task-anchored comment at `bst_train.py:151` (`# Aggressive CG/AP annealing — matches preferred config from run_20260418_151139.`).

**Safety checks:**
- `pytest`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_train import Hyp, MODELS, Task; print(Hyp._fields)"` — confirms the namedtuple fields are unchanged.

**Commit message draft:** "Move bst_train tuning rationale into arch_1_directions and historical_bst; trim configuration block."

---

### Steps 11+ (deferred, not executed in this branch)

These are explicitly **not** in scope per the review doc and your direction:

- Path/IO abstraction (focus area 5).
- `prepare_train_on_shuttleset.py` full module split into `mmpose_extract.py` + `homography.py` + `collate.py`. The light tidy (a) + (b) is in this branch as step 5b; the structural split is reserved for after-X3D-S.
- Validation script triplet shared core (focus area 6, site 5).
- Bulk style pass: AU/UK rename for `normalize_*` and "labeled"/"vectorized"; em-dash sweep; "fade" prose to "anneal" / "downtune".
- `aim_backfill._derive_tags` parameterisation.
- `pipeline/build_dataset.py:79-139` `dry_run()` consolidation.
- `pipeline/clip_index.py` docstring trim.

### Step P (proper-packages refactor — DONE)

**Surfaced 2026-04-26 while wiring the bit-exact smoke scripts. Executed same day as a single commit on `pre-phase-2-tidy`.**

Before step P, `src/bst_refactor/` had no `__init__.py` at the top three levels, and every script that lived more than one folder deep relied on a `sys.path.append(...)` block in its `__main__`:

```python
# bst_train.py / bst_infer.py / prepare_train_on_shuttleset.py / apply_heuristic.py — same pattern in each
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
```

That is why bare imports like `from bst_common import build_bst_network` resolved when run as scripts but broke when imported as library modules. `smoke_infer_bit_exact.py` had to mirror the same `sys.path.insert(...)` dance (`25e0308`) — acknowledged as a monkey-patch in scratch tooling.

**What landed:**

1. Added `__init__.py` to the three previously-missing dirs: `src/bst_refactor/`, `src/bst_refactor/stroke_classification/`, `src/bst_refactor/stroke_classification/main_on_shuttleset/`. (Four others were already present: `pipeline/`, `preparing_data/`, `preparing_data/heuristics/`, `model/`.)
2. Converted the only bare cross-dir import: `from bst_common import ...` → `from main_on_shuttleset.bst_common import ...` in `bst_train.py`, `bst_infer.py`, and `scratch/post_tidy_smoke/smoke_infer_bit_exact.py`. All other first-party imports (`pipeline.*`, `preparing_data.*`, `model.*`, `result_utils`, `run_tracker`) were already package-style or top-level under one of the two PYTHONPATH roots.
3. Dropped the `if __name__ == '__main__': sys.path.append(...)` blocks from `bst_train.py`, `bst_infer.py`, `model/bst.py`, `apply_heuristic.py`, `failsafe_bst_mmpose_zeroing_check_equivalence.py`, `prepare_train_on_shuttleset.py`, `raw_extract.py`.
4. Dropped the `sys.path.insert(...)` dance from `scratch/post_tidy_smoke/smoke_infer_bit_exact.py` and `scratch/post_tidy_smoke/smoke_prepare_2d_bit_exact.py`.
5. Updated docstrings on `bst_train.py`, `bst_infer.py`, `apply_heuristic.py`, `failsafe_*.py`, `prepare_train_on_shuttleset.py`, `raw_extract.py`, and the two smoke scripts to document the new invocation:
   ```sh
   PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \
       python -m main_on_shuttleset.bst_train
   ```

**Why two PYTHONPATH roots and not one.** The compact-prompt direction was `from bst_common import ...` → `from main_on_shuttleset.bst_common import ...` (rooted at `stroke_classification/`, not at `bst_refactor/`). That commits us to keeping `stroke_classification/` as a PYTHONPATH root. `bst_refactor/` stays as a second root because `pipeline/`, `run_tracker.py`, `aim_backfill.py`, `run_overview.py` live there. `conftest.py` already inserts both roots for tests; the scripts now document the same pair as their PYTHONPATH.

**Verification (laptop, `phase_2_refactor` venv):**

| Check | Result |
|---|---|
| `python -m py_compile` over all 12 modified files | ✅ OK |
| Live import of `main_on_shuttleset.bst_common`, `main_on_shuttleset.bst_infer` | ✅ both resolve cleanly under new PYTHONPATH |
| Live import of `apply_heuristic`, `failsafe_*`, `shuttleset_dataset`, `heuristics`, `heuristics.sticky_anchor`, `heuristics.current`, `model.bst`, `model.tempose`, `pipeline.config`, `pipeline.court_utils`, `pipeline.data_access`, `run_tracker`, `result_utils` | ✅ all resolve |
| Live exec of `scratch/post_tidy_smoke/smoke_infer_bit_exact.py` (script body up to `main()`), checking `Task` and `TAXONOMIES` are bound | ✅ `Task` resolves to `main_on_shuttleset.bst_infer.Task`, 4 taxonomies loaded |
| `pytest tests/` | ✅ 42 passed, 1 skipped (matches pre-step-P baseline; the 1 skip is `test_integration` without `BST_DATA_DIR`) |

**Verification (engelbart, post-push):** ✅ 43/43 with `BST_DATA_DIR` set. Wall-time dropped ~11s → ~9s, consistent with regular-package imports skipping the namespace-package fallback the no-`__init__.py` layout used to trigger. Bit-exact gates not re-run because step P is import-only and cannot perturb the forward pass.

**Net result on noqa weight:** ~22 of 30 noqa tags removed automatically (every `# noqa: E402` that sat below a sys.path block, plus several `# noqa: PLC0415` that were forced because sys.path setup needed to run first). Remaining survivors are addressed by step Q.

### Step Q (lint-debt cleanup — DONE)

**Surfaced 2026-04-26 from a noqa audit across the last week's commits. Executed same day on `pre-phase-2-tidy` after step P landed.**

Recap of the entry state: 30 noqa tags accumulated across recent branches. Step P removed ~22 of them automatically (every `# noqa: E402` that sat below a sys.path block, plus the PLC0415 imports that were forced inside functions because the sys.path setup needed to run first). Step Q audits the remaining survivors that step P did not fix.

**What landed:**

1. **`pipeline.data_access` side-effect import in `apply_heuristic.py`.** Renamed `_load_dotenv` → `load_repo_dotenv` (public), dropped the module-level auto-call from `data_access.py`, and added explicit `load_repo_dotenv()` calls in (a) `apply_heuristic.py` (right after the imports, just before the collision guard runs), and (b) `data_access.main()` so its CLI keeps reading `.env`. The opaque `import pipeline.data_access  # noqa: F401` is gone; the .env-load is now an explicit, documented call at each entry point that needs it.

2. **`BLE001` broad-exception narrowings in `raw_extract.py`.** Two sites:
   - `inspect_first_frame` (line 132): `np.asarray(value)` could raise on inhomogeneous lists / unsupported dtypes. Narrowed `except Exception` to `except (ValueError, TypeError)`. Behaviour preserved (cosmetic fallback that prints `<unknown>` for the dtype/shape).
   - `_stored_n_max` (line 173): `np.load(path).shape[1]` could fail at file-read or shape-access time. Narrowed to `except (OSError, ValueError, IndexError)`. Fallback returns `None` (treat the existing extract as untrusted).

3. **`PLC0415` lazy-import audit.**
   - `sticky_anchor.py:88` and `:111` (`pipeline.court_utils`): lifted to module top. `pipeline.court_utils` does not touch mmpose, so deferring served no purpose.
   - `sticky_anchor.py:334` (`preparing_data.prepare_train_on_shuttleset.normalize_joints`): kept deferred — `prepare_train_on_shuttleset` does `from mmpose.apis import MMPoseInferencer` at module top, which would force every heuristic-package consumer (including `tests/test_sticky_anchor.py`) to install mmpose. The noqa stays, now with a justifying one-line comment.
   - `current.py:50` (same pattern, same module): kept deferred for the same reason; same justifying comment added. (Originally outside the plan-doc bullet list but flagged by the noqa audit; folded in for consistency.)

4. **Working principle going forward:** `# noqa` is a tool of last resort. When ruff complains, default-ask is "is the lint rule correct?" before silencing.

**Verification (laptop, `phase_2_refactor` venv):**

| Check | Result |
|---|---|
| `pytest tests/` | ✅ 42 passed, 1 skipped (matches step-P baseline) |
| `ruff check` over the five touched files | ✅ All checks passed |
| Live import of `apply_heuristic`, `sticky_anchor`, `current`, plus `from pipeline.data_access import load_repo_dotenv` | ✅ resolves cleanly under the documented PYTHONPATH |

**Net noqa tally after step Q:** removed 5 noqa tags outright (1 F401, 2 BLE001, 2 PLC0415), kept 2 PLC0415 with justifying comments. The only pipeline-package noqas left are deliberate re-exports in `verify.py`, `build_dataset.py`, `clip_generator.py` (F401 by design — pre-existing pattern, not new lint debt).

**Verification (engelbart, post-push):** ✅ 43/43 with `BST_DATA_DIR` set. Bit-exact gates not re-run because step Q is import-rename + exception narrowing + same-module-top lift only — no behavioural delta possible on inputs the happy path actually sees. Combined with step P, end-to-end pytest wall-time on engelbart dropped from ~11s to ~9s and stayed there.

---

## Sticky_anchor Tier 1 test list

Sites and invariants only. The actual test code is written in step 6.

All tests live in `tests/test_sticky_anchor.py` and use synthetic data: small `RawClip` arrays (e.g. 4 frames, 3 candidate detections per frame) with hand-built `bbox`, `score`, and `joints` arrays, and a synthetic `ClipContext` carrying a fixed court rectangle and halfcourt centre. No MMPose, no real video, no GPU.

### Test 1 — Voronoi partition picks the right side

**Site:** `sticky_anchor.py:128-253` (`_pick_one_frame`), specifically the Voronoi partition at line 213.

**Invariant:** Given two candidates straddling the halfcourt line, the candidate above the line is assigned to the Top slot and the one below to the Bottom slot. Swapping the candidates' input order does not change the assignment.

**Synthetic setup:** Two candidates, one at `y = halfcourt_y - 50`, one at `y = halfcourt_y + 50`. Equal scores. `ClipContext.halfcourt_centre` set so both are inside the court rectangle.

### Test 2 — Bottom-first slot ordering with cross-slot exclusion

**Site:** `sticky_anchor.py:218-219`.

**Invariant:** When two candidates both fall on the Bottom side of the Voronoi partition, the Bottom slot fills first and the Top slot is left empty (or filled by a Top-side candidate from elsewhere); the Bottom-picked candidate is excluded from Top consideration in the same frame.

**Synthetic setup:** Two candidates both at `y > halfcourt_y`. Verify Bottom is filled, Top is empty (or fallback-filled per the rule).

### Test 3 — Sitting tiebreaker with fallback

**Site:** `sticky_anchor.py:232-237`.

**Invariant:** When two candidates are tied on the primary score, the "non-sitting" tiebreaker prefers the candidate whose pose passes the sitting filter; if all candidates fail the sitting filter, the fallback rule kicks in (whatever it is — capture the current behaviour exactly).

**Synthetic setup:** Two candidates with identical positions and scores; one with joints arranged to pass the sitting filter, one to fail. Verify the non-sitting one is picked. Then a second case where both fail; verify the fallback rule.

### Test 4 — Rally-presence rejection

**Site:** `sticky_anchor.py:242-248`.

**Invariant:** A candidate that fails the rally-presence check is not picked even when it would otherwise win on score.

**Synthetic setup:** Two candidates; the higher-score one is positioned so it fails the rally-presence check, the lower-score one passes. Verify the lower-score one is picked.

### Test 5 — EMA reset on full-frame failure

**Site:** `sticky_anchor.py:294`.

**Invariant:** When a frame produces zero successful picks, the EMA state for both slots resets to `halfcourt_centre`. The next frame's prediction is anchored at the centre, not at the previous frame's stale value.

**Synthetic setup:** Two-frame sequence. Frame 1 produces valid picks (EMA advances to non-centre values). Frame 2 has all candidates positioned outside the court (full-frame failure). After frame 2, query the internal EMA state; both slots are at `halfcourt_centre`.

### Test 6 — Mixed-pick reset only the unpicked slot

**Site:** `sticky_anchor.py:302-306`.

**Invariant:** When a frame picks a Bottom but no Top, the Top EMA resets to `halfcourt_centre` while the Bottom EMA advances normally. Symmetric for the inverse case.

**Synthetic setup:** Frame 1 establishes both slots non-centre. Frame 2 has only one Bottom-side candidate (no Top candidate found). After frame 2, Bottom EMA has advanced; Top EMA is at `halfcourt_centre`.

### Test 7 — `update_gate_eps` in-court guard before EMA update

**Site:** `sticky_anchor.py:315`.

**Invariant:** A picked candidate whose position is just outside the court rectangle (within `update_gate_eps`) does not update the EMA. The EMA stays at its previous value.

**Synthetic setup:** Pick a candidate just outside the court boundary. Confirm EMA does not move.

### Tests not written

- `current.py` byte-identity: covered by `failsafe_bst_mmpose_zeroing_check_equivalence.py` on real data; a synthetic stand-in would have to re-stub `check_pos_in_court` and `normalize_joints` and would drift from upstream.

---

## Test harness layout

Single new file: `tests/test_sticky_anchor.py`. Uses `pytest`. No fixtures shared with other test files.

The synthetic `RawClip` builder lives at the top of the file as a small helper:

```python
def _make_raw_clip(n_frames, candidates_per_frame, court, ...):
    """Build a RawClip with hand-specified bbox/score/joints arrays."""
    ...
```

Each test calls `_make_raw_clip(...)` then `sticky_anchor.apply(raw, ctx, ...)` and asserts on the returned `HeuristicOutput`.

If `sticky_anchor.apply()` does not currently expose internal EMA state for tests 5-7 to inspect, I will add a minimal accessor (e.g. return the EMA history alongside the output, or expose a `_state` attribute on a debug path). Documented in the commit message.

---

## What this plan does not do

- **No edits to the BST model graph itself** (`bst.py` `BST_CG_AP` forward, `CrossTransformerLayer`, the building blocks, the five `BST_*` partials). This branch addresses sprawl and dead code in the surrounding scaffolding only; the model is read-only.
- **No abstraction work on paths.** Reserved for after-X3D-S.
- **No full module split of `prepare_train_on_shuttleset.py`.** The light tidy is in step 5b; the structural split is reserved.
- **No bulk rename.** AU/UK and "fade" prose passes are reserved.
- **No new tests beyond sticky_anchor.** No coverage push for `current.py`, datasets, or training loops in this branch.

If anything in this list surprises you, ping me before I cut the branch.
