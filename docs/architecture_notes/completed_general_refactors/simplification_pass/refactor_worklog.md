# Refactor worklog

Per-commit log of the simplification pass. Audit trail for the diff against
`main`: files / change / gate / SHA / verifier verdict. Lightly tidied from the
working version in the planning dir (`code_simplification_and_streamline/`):
in-progress markers like "not yet pushed" and "PENDING-HANDOFF" are dropped now
that everything has landed; the merge SHA is corrected to `18e5c2c`.

Order: oldest readiness work first, then execution log per commit (D3, Batches
1-4, B1-B7, post-HPC cleanup, audit-fix), then the merge.

## Entry format

Per commit:
- **Batch / item:** e.g. `Batch 1` or `B3 court dedup`
- **Files:** the paths touched
- **Change:** one line, what + why
- **Gate:** which gate ran, and its result (green/red + key numbers)
- **Commit:** the SHA on `refactor/simplification-pass`

## Readiness (pre-execution)

- **2026-06-29 -- .gitignore desync tidy.** Removed the over-broad / dead rules
  (`preparing_data` source shadow, `ShuttleSet_data_une_merge_v1`, `tmp/`,
  `sticky_anchor_inspection`, the BRIC trio, top-level `uploads/*`,
  `frontend/.claude/`); `**/tb/` and `**/predictions/*.npz` lifted to
  track-by-default; `!HANDOVER.md` added; the experiments `.gitignore` collapsed
  from 234 lines to the 7 curated champions under an archived-weights note; 15
  stale manifest `.bak` deleted. Landed on `main` (commit `3b9bd5e`).
- **2026-06-29 -- baseline + env.** `opencv-python-headless` installed into
  `badminton-cicd`; green pytest baseline captured: 456 passed / 2 known-red
  (`test_api` Docker `/app/uploads/` path, `src/api` is OUT-list) / 19 skipped /
  0 collection errors. Laptop confirmed CPU-only (GTX 960M is sm_50; torch needs
  sm_75+), so the GPU leg is HPC-only.
- **2026-06-29 -- runbook + decisions.** Runbook drafted. `bst.py`
  `n_people` / `n_streams` rename confirmed. Collation-fixture environment
  resolved: local CPU fixture AND HPC GPU run. The `preparing_data` grep caveat
  retired (`.gitignore:42` removed; plain `rg` now sees the dir).
- **2026-06-29 -- boursync + data goldens built.** 5-stem raw fixture pulled to
  `scratch/raw_extract_local/`; clean fixture regenerated to `scratch/clean_extract_local/`
  via apply_heuristic sticky_anchor (both git-ignored). Extraction golden (B3
  44-row `get_H` + apply_heuristic clean, both heuristics) and collation golden
  (B4, 4 pose styles, forced `min_t` truncation) captured + self-verify 0-diff.
  KEY CORRECTION: the planned raw-to-clean extraction golden could not gate B5
  (`detect_players_*` need mmpose+GPU+mp4; the raw fixture feeds apply_heuristic).
  B5 is HPC-only.
- **2026-06-29 -- goldens adversarially reviewed.** F3 (dead `min_t` truncation
  on the fixture) FIXED; F1/F2 (B3 court leg) addressed by expanding the get_H
  equivalence to all 44 rows -- though it's a capture-time precondition proof,
  not a post-edit gate; F4 (unknown-routing CPU gap) accepted as HPC-covered.
  Stub-soundness + determinism confirmed.
- **2026-06-29 -- split pre-analyses written + reviewed.** Three pre-analyses
  for `collate_npy` / `train_network` / `detect_players`, each with an appended
  round-1 adversarial review. Surfaced: no seed in the live training path (B7
  gate needs an injected seed; seeded CPU train is bit-reproducible cross-process),
  a pre-existing `best_state` `UnboundLocalError` on degenerate runs, and that
  the une_v1_14 / bst_25 bit-exact does NOT re-extract pose (so it does not gate
  B5).

## Execution log (per commit)

- **2026-06-29 -- D3 lazy mmpose import (`88405ed`).** First execution item.
  - Files: `preparing_data/prepare_train_on_shuttleset.py` (+13/-1): top-level
    `from mmpose.apis import MMPoseInferencer` removed; `from __future__ import annotations`
    + a `TYPE_CHECKING` import added so the type hints resolve for static checkers
    without a runtime import; lazy `from mmpose.apis import MMPoseInferencer` inside
    the three functions that instantiate it (`detect_players_3d`, `prepare_2d_dataset...`,
    `prepare_3d_dataset...`). The two CPU goldens were destubbed.
  - Change + why: decouples `collate_npy` / `get_H` / the court helpers / the heuristics
    from mmpose at import, so they load in `venv-bst-x` and the CPU goldens with no GPU
    dep. The `from __future__ import annotations` is KEPT permanently (it is the
    decoupling mechanism).
  - Gate (CPU, all green): ruff clean (no F821 after the TYPE_CHECKING import); module
    imports with mmpose absent; extraction + collation goldens 0-diff UNSTUBBED; pytest
    456 / 2 known-red / 19 skipped. Model + val goldens unaffected (different module).
  - HPC confirmation: 2D pose smoke 10/10 PASS, `pos_max + joints_max = 0.000e+00` on
    every stem (dual-invocation). Confirms the lazy import resolves at runtime AND
    produces bit-identical output. The 3D runs-clean smoke is accepted-deferred (3D
    extraction is dormant).
  - Commit: `88405ed`.

- **2026-06-30 -- Batch 1 dead-code deletions (`34a1f01`).** First Bucket-A batch.
  - Files: `pipeline/clip_generator.py`, `pipeline/shuttle_extractor.py`,
    `preparing_data/apply_heuristic.py`, `preparing_data/prepare_train_on_shuttleset.py`,
    `preparing_data/shuttleset_dataset.py` (152 deletions, 0 additions).
  - Change + why: removed dead code no caller reaches. The `RandomTranslation` /
    `RandomTranslation_batch` augmenters + their now-unused `import torch` / `from torch
    import Tensor` / `torchvision.transforms.v2` imports (kept `torch.utils.data` and
    numpy); `extract_shuttle_trajectory` (shuttle_extractor, `extract_all_shuttles` kept);
    `detect_shuttlecock_by_TrackNetV3_with_rectification` (`_with_attention` sibling and
    `get_shuttle_result` kept); `_resolve_or_none` and the `RunStats.skipped_missing_raw`
    field; the stray `# noqa: F401` numpy import in clip_generator. Line drift handled
    (the rectification fn was at `:457`, not the doc's 447).
  - Gate (CPU, all green): re-grep 0 callers for all five symbols (remaining hits are
    comments / docstrings / markdown, no live caller); ruff clean on the five files (no
    F401/F821, so no missed import cascade); py_compile OK; pytest 456 / 2 / 19. Independent
    adversarial verifier returned PASS.
  - No HPC leg.
  - Commit: `34a1f01`.

- **2026-06-30 -- Batch 2a small mechanical edits (`77c7cdb`).** Bucket-A.
  - Files (12, +69/-65): clip_generator, build_dataset, verify, download_videos, data_access,
    apply_heuristic, prepare_train_on_shuttleset, shuttleset_dataset, bst_x_train,
    heuristics/{base,current,sticky_anchor}.
  - Changes: dropped 3 stale `# noqa: F401` (all names confirmed used); `VALID_CLIP_WINDOWS`
    frozenset single-sources the clip-window guard + CLI choices; `J = 17` single-sourced
    into heuristics/base, imported by current + sticky_anchor; `from dataclasses import
    fields` hoisted to module scope in apply_heuristic (two local imports dropped);
    `_print_paths_tsv` de-duplicates the TSV-print block across data_access
    interactive() / main(); `_pad_tail_to` de-duplicates the np.pad block in
    make_seq_len_same (branch inverted to early-return); `_to_hparam_value` nested fn
    inlined to an explicit loop in bst_x_train; `aux_schedule_factor` rewritten to two
    plain ifs; the `min_t` guard simplified to `len(failed) != len(shuttle)`; `return 0
    if stats` (run() always returns RunStats); dropped the dead `except FileNotFoundError`
    in download_video (the `_check_ytdlp()` preflight makes it unreachable); added the
    Step-5 deferred-0-byte-check comment.
  - Two deliberate calls: (1) KEPT `build_dataset.__main__`'s `_validate_inputs`
    (dropping it removes the dry_run path's only input validation -- `dry_run()` doesn't
    validate internally -- so it stays; the existing comment already documents it). (2)
    `VALID_CLIP_WINDOWS` is a frozenset per spec; argparse `choices` over a set makes the
    `--help` / error-text window ordering hash-ordered (accepted values + validation
    unchanged, cosmetic only). Swap to a tuple if stable `--help` ordering is wanted.
  - Gate (CPU, all green): ruff clean on all 12; pytest 456 / 2 / 19 (baseline
    unchanged); a 931-case numerical smoke proving the two real logic edits equivalent
    (aux_schedule_factor 651/651, make_seq_len_same against the real edited fn 280/280,
    0 mismatches). Independent adversarial verifier: PASS.
  - No HPC leg.
  - Commit: `77c7cdb`.

- **2026-06-30 -- Batch 2b top-1 unify (`8591396`).** Top-1 unification item.
  - File: `bst_x_common.py` (+3/-1). `dump_topk_predictions` top-1 now `logits.argmax(dim=-1)`
    instead of `topk_idx[:, 0]` (+ a 2-line why-comment).
  - Why: the dump was the only top-1 site using topk[:, 0]; every other metric site uses
    argmax, and build_fe_stats_jsons + calibration_ece already assert argmax == y_pred.
    Now those asserts hold by construction; the tie-guard (build_fe :76, topk[:,0] ==
    y_pred) stays to catch any tie.
  - Gate (CPU, green): equivalence smoke 1974/1974 tie-free rows identical (argmax ==
    topk[:,0]); crafted ties confirmed they diverge only on ties (so the tie-guard earns
    its keep). ruff clean; pytest 456 / 2 / 19. Verifier PASS.
  - HPC verification: `bst_x_infer --fe` dump on main + branch, same checkpoint, test
    split (4,202 rows). Both `y_pred_top1` AND `logits` IDENTICAL by `np.array_equal`.
    Logits-identical is the stronger result -- argmax over identical logits is
    deterministic, so the argmax-vs-topk[:,0] swap was provably a no-op on a real trained
    model.
  - Commit: `8591396`.

- **2026-06-30 -- Batch 3a bst.py comment strip + dim rename (`2ca8214`).** Bucket-A
  discipline.
  - File: `model/bst.py` (+78/-92). Cut the TensorFlow-analogue commentary, added a
    PyTorch-idiom glossary header, switched a couple of docstrings; KEPT every shape /
    axis annotation, the .view-vs-.reshape note, the CrossTransformerLayer residual
    NOTE, the AP/CG warm-start comments. One code change: forward()'s overloaded `n`
    split into `n_people` (=2, used before the JnB+shuttle cat) and `n_streams` (=3,
    after the cat).
  - Gate (CPU, all green): model build+forward bit-exact 0-diff (5 variants, 464 weight
    tensors, all outputs bit-identical) -- the behaviour authority; normalised-AST diff
    (n_people / n_streams -> n, strip docstrings) IDENTICAL vs HEAD, so the only code
    change is the rename; ruff clean; py_compile OK; pytest 456 / 2 / 19. Verifier PASS.
  - HPC verification: `bst_x_infer --fe` on main vs branch, 4,202 test rows.
    `y_pred_top1` AND `logits` IDENTICAL by `np.array_equal`. Covers the rename and the
    model forward path.
  - Commit: `2ca8214`.

- **2026-06-30 -- Batch 3b comment / docstring strip across the rest (`b9ea158`).**
  Bucket-A discipline.
  - Files (15, +63/-196): bst_x_train, bst_x_infer, shuttleset_dataset, clip_generator,
    build_dataset, shuttle_extractor, verify, download_videos, prepare_train_on_shuttleset,
    config, data_access, clip_index, augmentations, heuristics/{current,sticky_anchor}.
    (apply_heuristic had nothing left: its target was already cut in Batch 2.)
  - Change: dropped the TF analogues + newcomer notes + code-restating comments;
    switched Dataset_npy_collated's docstring to `:param:` style; trimmed data_access's
    `--help`-bearing module docstring (kept the layout diagram, taxonomy explanation,
    Python API, clip_index relationship). KEPT every shape / axis annotation, the
    sticky_anchor index-space invariant + `# Bool mask over k candidate bboxes`, the
    bst_x_infer hard-fail-on-None block, the DIVERGENCE-FROM-BST why+TODO, the maths
    whys.
  - Execution: the bulk strip ran as a delegated agent (16-file scope, AST-self-gated per
    file); the executing agent re-gated independently and added 2 follow-up trims. Two of
    the agent's judgment calls left as-is.
  - Gate (CPU, all green): Tier-1 AST gate = comment / docstring-only on all 15 (14
    exit-0, data_access exit-2 = the intended `--help` trim); ruff clean; pytest 456 / 2
    / 19. Quality: targeted keep-list + residual-TF + scope self-check (the external
    quality-verifier agent died on an API error mid-response, so a deterministic
    grep-based self-check was substituted; no residual TF analogue in any 3b file,
    keep-list intact, no OUT-list path).
  - No HPC leg.
  - Commit: `b9ea158`.

- **2026-06-30 -- Batch 4a type the Hyp container (`ede5a0d`).** Bucket-C.
  - File: `bst_x_train.py` (+31/-39). `Hyp` switched from functional `namedtuple('Hyp',
    [...])` + `hyp = Hyp(...)` to a typed `class Hyp(NamedTuple)` with per-field type +
    inline default, then `hyp = Hyp()`. `from collections import namedtuple` ->
    `from typing import NamedTuple`.
  - Verification (CPU, exact): new `hyp._asdict()` byte-identical to the captured
    baseline (keys + values + order, 23 fields); early_stop_n_epochs correctly placed 8th
    per the field-list order (not the kwarg-position-2 of the old instantiation), so
    _asdict order is preserved; _replace / _asdict resolve (the 4 sites); no positional
    index / tuple-unpack of Hyp anywhere in src. ruff clean; pytest 456 / 2 / 19.
    Verifier PASS (3-way value+order check). Note: dict defaults now class-level shared,
    behaviourally identical (no in-place mutation anywhere; _replace assigns fresh
    dicts).
  - No HPC leg.
  - Commit: `ede5a0d`.

- **2026-06-30 -- Batch 4b relocate the failsafe checker (`7cd4b41`).** Bucket-C.
  - Files: `git mv` `preparing_data/failsafe_bst_mmpose_zeroing_check_equivalence.py` ->
    `validation_scripts/` (rename 99%, the 1% = its docstring `python -m` module path);
    `mmpose_heuristic_investigation.md` (3 refs: location note, gate-command invocation,
    the stale stroke_classification file-path entry -> validation_scripts). +4/-4.
  - Why: the one-shot byte-identity checker belongs with the other validation_scripts
    checkers; it had no importers, and its own imports are absolute so they resolve from
    the new dir.
  - Decisions: the pre_phase_2_* docs' bare-name mentions left as dated historical
    records (no module / filesystem path, nothing breaks). The doc's sibling
    stroke_classification staleness (PYTHONPATH + prepare_train / apply_heuristic /
    heuristics paths) left as a separate pre-existing issue, out of 4b scope.
  - Gate (CPU, green): re-grep 0 importers; py_compile OK; `python -m
    validation_scripts.failsafe_... --help` resolves + loads (old
    `preparing_data.failsafe_...` path now dead); no remaining `preparing_data.failsafe`
    refs anywhere; ruff clean; pytest 456 / 2 / 19. Verifier PASS.
  - HPC verification: failsafe data-run from its new path on engelbart/bourbaki with
    MMPose + the raw / committed scratch extracts, dual-invocation: 50/50 stems,
    `0.000e+00` max abs diff on `_pos` and `_joints`.
  - Commit: `7cd4b41`.

## Milestone: Buckets A + C complete

`88405ed` D3 / `34a1f01` B1-dead / `77c7cdb` B2a / `8591396` B2b / `2ca8214` B3a /
`b9ea158` B3b / `ede5a0d` B4a / `7cd4b41` B4b. Branch green vs the 456 / 2 / 19
baseline; main untouched at this point.

## Post-HPC stale-paths cleanup (2026-06-30)

The 4b failsafe and the D3 2D smoke both failed on their first HPC run because they
assumed `$BST_X_MMPOSE_NPY_DIR` pointed at a `current`-equivalent committed extract.
It didn't, hadn't since the 2026-04-29 flip to sticky_anchor, and the legacy `_flat`
(current-equivalent) dir under `/scratch/comp320a/ShuttleSet_data_bst_25/` had been
deleted to free quota on bourbaki. Two consequences: (a) the env-var fallback in
`failsafe_*.py` silently compared `current` output against `sticky_anchor` output
(different heuristics, can't match); (b) the smoke 2D docstring and `RUN_COMMANDS.md`
pattern both said to point `REFERENCE_DIR` at `$BST_X_MMPOSE_NPY_DIR`, which would
have failed even on main.

Recovery was the dual-invocation gate pattern: main writes the reference into a
scratch dir via its own gate invocation; branch reads that dir as committed-side
via a second invocation. Both gates re-passed at 0.000e+00.

To stop the next session falling into the same trap, the following cleanup landed:

- **failsafe script:** `--committed-dir` made `required=True`; the
  `$BST_X_MMPOSE_NPY_DIR` env fallback dropped; the script refuses to default to
  sticky_anchor. Docstring rewritten to document the dual-invocation pattern. Unused
  `os` import dropped.
- **2D pose smoke:** docstring rewritten to the dual-invocation pattern. Repo path
  fixed (`badminton_stroke_classifier` -> `badminton_stroke_classification`). Hit-zone
  stems picked via `sort | sed -n '1p;172p;...'` so the stems are guaranteed to exist
  in the references.
- **Infer smoke:** docstring rewritten with current repo, `venv-bst-x`, current
  `/scratch/comp320a/...` collated dir, current branch refs.
- **`mmpose_heuristic_investigation.md` doc split:** the historical 650-line doc was
  renamed (via `git mv`) to `historical_mmpose_heuristic_investigation.md`, and a new
  330-line operational reference `mmpose_heuristic.md` was extracted alongside
  (TL;DR, current paths table, apply-heuristic canonical run, failsafe gate with
  dual-invocation, algorithm spec verbatim from historical, hyperparameter table,
  output schema, court-space calibration, known limitations). Top-of-file pointer
  added to the historical doc. Four back-refs updated to point at the right one
  (operational vs historical): `sticky_anchor.py`, `current.py`,
  `bst_x_issues_and_bugs_squashed.md`, `frame_recovery_stats.md`.
- **`.env.example` line 37:** stale `_flat` example path -> current sticky_anchor
  path.

A second adversarial review caught 8 missed doc-file back-refs to the renamed
historical doc + a low-priority `apply_heuristic.py` docstring example still citing
deleted Phase-1 paths + a polish ask for a parenthetical on the failsafe's module
path flipping between main and branch. All fixed in-place.

- Commit: `09e4215` ("Stop the mmpose bit-exact test gates defaulting to the
  sticky_anchor extract").

## Bucket B (B1-B7, structural splits)

- **2026-06-30 -- B1 validate() reuse of accumulate_class_counts (`01b1a80`).**
  Subsumes the validate() reuse item from the simplification review.
  - Files: `src/bst_x/bst_x_train.py` (+15 / -21).
  - Change + why: validate() replaces its hand-rolled one-hot TP/FP/FN block with
    `accumulate_class_counts(logits.argmax(dim=1), labels, n_classes)` (the same helper
    train_one_epoch uses). Option B device discipline: cum_tp / cum_fp / cum_fn init on
    `device` with `dtype=torch.long`, accumulate on device inside the loop, single
    `.cpu()` block after the loop before the precision / recall / F1 math. Drops the
    dead `cum_tn` / `tn`. The one-hot block was the file's only `F.` user, so the
    now-unused `import torch.nn.functional as F` goes with it.
  - Gate (CPU, all green): ruff clean on `bst_x_train.py`; val-metrics golden OK
    (per-class tp / fp / fn + f1 identical to the captured-on-main one-hot golden);
    pytest 456 / 2 / 19.
  - External verifier (fresh general-purpose agent, 10-item adversarial pass): CLEAN.
    Confirmed diff scope, behaviour preservation (`int64/int64` true division returns
    float32, same dtype as old default-float path; NaN-on-zero-support still caught),
    Option B device discipline, `tn` / `cum_tn` fully removed, 7-tuple return signature
    unchanged, no stale `F.one_hot` / `TN` references. Observed (predates B1):
    `cum_top2 += int(...)` forces a per-batch host sync.
  - HPC gate (bourbaki A100, venv-bst-x): GREEN. `smoke_b1_validate_gpu.py` (synthetic
    two-batch validate() call on cuda) prints `OK: validate() ran on cuda with no
    device mismatch`. pytest 473 / 2 / 2. No device-mismatch RuntimeError; Option B
    holds.
  - Commit: `01b1a80`.

- **2026-06-30 -- B2 collapse three test passes into one (`194783b`).**
  - Files: `src/bst_x/bst_x_train.py` (+ ~50 / -69), `src/bst_x/data_pipeline_to_model_train.md`,
    `src/bst_x/run_tracker.md` (doc method-signature touch-ups).
  - Change + why: pre-B2 `train_network` did three forward passes over `self.test_loader`
    per serial (argmax via module-level `test()`, topk(k=2) via `test_topk()`, topk(k=5)
    via `dump_topk_predictions` for the npz). Now `Task.dump_predictions` returns the
    per-split dumps; `Task.test(dump=...)` reads top-1 straight off `dump['y_pred_top1']`
    (argmax, unified in Batch 2b `8591396`); `Task.test_topk_acc(dump=..., k=2)`
    re-derives top-2 via `torch.topk(dump['logits'], k=2)` -- NOT `topk_idx[:, :2]`
    (the k=5 slice tie-breaks differently from a fresh k=2 topk). The two newly-dead
    module-level `test()` / `test_topk()` go with it.
  - Gate (CPU, all green): ruff clean; on-branch differential smoke
    (`hpc_test_results/smoke_b2_test_passes.py`): old-flow (two metric-relevant forward
    passes) vs new-flow (one dump + derive) on the same fixed-seed model + loader.
    `np.isclose(atol=0, rtol=0)` on macro_f1 / min_f1 / accuracy / top2_accuracy. MATCH.
    pytest 456 / 2 / 19.
  - Doc edits: `data_pipeline_to_model_train.md` updated the `Task.test /
    test_topk_acc / dump_predictions` table rows and the chained-call sketch;
    `run_tracker.md` updated the per-serial flow example.
  - External verifier: CLEAN. Advisory observations only (cosmetic naming, AMP
    forward-compat, no real defects).
  - HPC gate: GREEN. Reran `bst_x_infer --fe` value-compare on the new SHA. Both
    `y_pred_top1` AND `logits` IDENTICAL. B2 doesn't touch any code that `bst_x_infer`
    exercises, so this confirms the forward path is unperturbed.
  - Commit: `194783b`.

- **2026-06-30 -- B3 court dedup (`b3e242b`).** No HPC leg by spec.
  - Files: `src/bst_x/preparing_data/prepare_train_on_shuttleset.py` (-141 / +7),
    `src/bst_x/preparing_data/mmpose_changes.md` (table-row touch-up).
  - Change + why: deleted the 9 court-block duplicates from prepare_train (`get_H`,
    `get_corner_camera`, `scale_pos_by_resolution`, `convert_homogeneous`, `project`,
    `get_court_info`, `to_court_coordinate`, `normalize_position`, `check_pos_in_court`).
    Added `from pipeline.court_utils import check_pos_in_court, get_court_info` to keep
    prepare_train's internal callers (lines `detect_players_2d`, `detect_players_3d`,
    `main`) live and the heuristics' `from prepare_train import check_pos_in_court,
    normalize_joints` lazy-import path valid via re-export. `normalize_joints` +
    `normalize_shuttlecock` bodies stay (not in court_utils). `src/shared/court.py` (BRIC
    mirror) untouched. Net 3 copies to 2.
  - Gate (CPU, all green): extraction golden 44 vids `get_H` + 2 heuristics × 5 stems
    clean output, all bit-identical (the 44-row byte-compare). ruff clean. pytest 456 / 2
    / 19. The re-exports resolve from prepare_train.
  - External verifier: CLEAN. Confirmed diff scope, all 9 `def`s gone,
    normalize_joints / normalize_shuttlecock byte-identical, import wiring correct,
    internal callers resolve, heuristics' lazy-imports still work, OUT-list paths
    untouched, extraction golden green, `from prepare_train import get_H` correctly fails.
  - Commit: `b3e242b`.

- **2026-06-30 -- B4 split collate_npy (`5b361b8`).**
  - Files: `src/bst_x/preparing_data/prepare_train_on_shuttleset.py` (+200 / -103).
  - Change + why: the ~250-line `collate_npy` (one call = one split) splits along four
    concerns: `_resolve_clips_and_labels` (CSV filter + per-row label + unknown-routing +
    file-exists drop, returning the row-aligned triple `(data_branches, labels,
    clip_stems_arr)`); `_load_clip_npys` (ThreadPool joints / pos / failed,
    submission-order `result()`s); `_align_shuttle_and_truncate` (shuttle CSV read +
    min-t truncation, RETURNS the truncated `(joints_ls, pos_ls, shuttle_ls)` triple
    explicitly); `_pad_augment_stack_save` (`bad_styles` raise BEFORE the ProcessPool,
    then submission-order `result()`s, the non-pose stacks before save, and the per-style
    pose `np.save(np.stack(arrs))` inline-stack pattern preserved). The orchestrator
    `collate_npy` keeps signature + docstring + argument guards and chains the four
    helpers. Dropped the dead `failed = failed[:min_t]` line.
  - Gate (CPU, all green): ruff clean; collation golden bit-identical (5 clips, 9 output
    files); the golden also exercises the missing-file skip path (24,861 master-CSV rows
    skipped), so row alignment under heavy filtering is exercised at the CPU gate;
    pytest 456 / 2 / 19.
  - External verifier: CLEAN over 10 items + the pre-analysis's 12 invariants. Confirmed
    every helper boundary, submission-order collection in both executors, the
    explicit-return contract, the inline `np.stack` in the per-style save loop, the
    name-shadow fix in concern 4, every print preserved in order, no closures over locals
    (ProcessPool pickling intact).
  - HPC gate (bourbaki, venv-mmpose both sides): GREEN. Two taxonomies × main vs branch
    = 4 collations, then per-clip diff via `harness/collation_fulldiff.py`. Both
    `OK ... identical per-clip across [train, val, test] (atol=0.0)`:
    - `une_v1_14 / split_v2`: 22,743 / 5,250 / 4,210 clips (train / val / test).
    - `bst_25 / split_bst_baseline`: 25,741 / 4,241 / 3,499 clips. Exercises unknown
      routing (`bst_25` has_unknown=True) and the sided label path (has_sides=True).
    - ~66k clips total, every pose / pos / shuttle / videos_len / labels / clip_stems
      array element-wise identical.
  - CLI gotchas pinned: main side needs `venv-mmpose` (the D3 lazy mmpose import is
    branch-only), `--collation-id` REQUIRED, no `--save-dir` (override
    `BST_X_COLLATED_DATA_ROOT` per side), `--taxonomy` takes canonical names only.
  - Commit: `5b361b8`.

- **2026-06-30 -- B5 detect_players_2d / 3d shared-body extract (`41719e4`).** 3D path
  ungated by spec.
  - Files: `src/bst_x/preparing_data/prepare_train_on_shuttleset.py` (+53 / -38).
  - Change + why: pulled the identical middle band of `detect_players_2d` and
    `detect_players_3d` (the `< 2` short-circuit, `check_pos_in_court`, the `!= 2 on
    court` guard, the Top-before-Bottom strict-`>` `np.flip`) into a single
    `_order_two_on_court(keypoints_2d, vid, all_court_info, res_df)` returning
    `(in_court_pid, pos_normalized) | None`. Shape-agnostic by design (takes only 2D
    keypoints + court info), so the 2D HPC smoke transitively covers the shared
    arithmetic for both callers. All 7 invariants preserved: failed-frame zero shapes
    stay variant-specific in the callers, `< 2` precedes `check_pos_in_court`, strict-`>`
    flip with `np.flip` not argsort, per-call `inferencer_3d = MMPoseInferencer(pose3d=
    "human3d")` with WARNING comment block kept verbatim, helper receives 2D keypoints
    in both variants, `failed_ls` stays Python-bool, 2D `normalize_joints` + bbox
    threading stays in the 2D caller.
  - Gate (CPU, all green): ruff clean; pytest 456 / 2 / 19. No test exercises
    detect_players_*, so pytest is a consumer + import-health backstop only. The post-B3
    `check_pos_in_court` / `normalize_joints` re-exports still resolve from prepare_train.
  - External verifier: CLEAN over 10 items + a 5-case synthetic fixture test of
    `_order_two_on_court`. The synthetic fixture confirmed (a) `< 2` returns None,
    (b) `!= 2 on court` returns None, (c) y-ordered pair returns input pid unchanged +
    full `(m, 2)` pos_normalized, (d) reverse-y returns flipped pid, (e) y-tie keeps
    ascending order (strict `>` honoured). Flagged that `heuristics/current.py` has a
    near-identical inline copy (already noted as out of B5 scope).
  - HPC gate (bourbaki, venv-mmpose): GREEN. 2D smoke run dual-invocation: main writes
    the reference into `/tmp/prepare_2d_smoke_main_b5/`, branch reads it as
    committed-side via a second invocation. Branch output `PASS: 10/10 stems matched
    (atol=1e-05)`, every stem `pos_max=0.000e+00 joints_max=0.000e+00`. Live MMPose +
    RTMPose-L + GPU on real mp4s.
  - 3D path: ungated by spec. Accepted as dormant (`use_3d_pose=False` everywhere
    deployed, no 3D collated artifact, no 3D raw extract); the shape-agnostic helper
    means the 2D bit-exact transitively proves the shared arithmetic for both callers.
  - Commit: `41719e4`.

- **2026-06-30 -- B6 npz writer dedup (`192df36`).** No HPC leg by spec (payload-only,
  schema- and value-invariant by construction).
  - Files: `src/bst_x/bst_x_common.py` (+47 helper), `src/bst_x/bst_x_infer.py` (-23
    inline savez block, -1 unused numpy import, +4 helper call), `src/bst_x/bst_x_train.py`
    (-22 inline savez block, +5 helper call). Net 55 insertions, 42 deletions across the
    three.
  - Change + why: both prediction-npz writers (`Task.dump_predictions` for end-of-serial
    training; `dump_run_predictions` for post-hoc `bst_x_infer --fe`) inlined the same
    9-key `np.savez(...)` payload. Lifted into a shared
    `_write_prediction_npz(out_path, dump, dataset, taxonomy, run_id, serial)` in
    `bst_x_common.py`. Helper owns ONLY the savez payload + the `clip_stems is not None`
    hard-fail (the `np.asarray(None)` silent-0-d trap was duplicated at both sites). The
    dir, the filename, the split-loop, the inference run's `inference_manifest.yaml`,
    the `written.append` / `print('saved')` chatter, and the `dumps[split_name] = dump`
    reuse stay per-caller.
  - "Pin k=5" decision: the runbook said pin k=5 to prevent schema drift. Original plan
    added a hard runtime assert in the helper, but that broke 4 existing
    `tests/test_train_surface.py` cases that intentionally exercise k=2 and
    k=5-clamped-to-`head_size=3`. Loosened: the helper writes whatever k_eff the dump
    arrives with, and the docstring records the caller-side convention (both production
    writers pass k=5 to `dump_topk_predictions`). Tests + production behaviour both
    preserved; the schema "pin" is now a documented convention.
  - Gate (CPU, all green): ruff clean. The `np.savez` block was the last `np.` user in
    `bst_x_infer.py`, so the unused `import numpy as np` goes with it (bst_x_train.py
    still has live `np.` users; import stays there). Value-compare smoke
    `smoke_b6_npz_writer.py`: writes the same payload via the verbatim pre-B6 inline
    savez AND `_write_prediction_npz`, then diffs the two npz files on (a) key set,
    (b) key ORDER, (c) per-key dtype + shape, (d) `np.array_equal` per key (catches a
    same-dtype field transpose like y_pred_top1 swapped with y_true). MATCH on all 9
    keys. pytest 456 / 2 / 19 (baseline). All 4 test_train_surface dump_predictions cases
    pass with k=2 and k=5-clamped-to-3.
  - External verifier: CLEAN over 8 items + an independent pre-B6-vs-post-B6 git-show
    check. Confirmed diff scope, helper signature / body, savez kwarg order, no hard k
    assert, both call sites use positional args in the right order (no run_id / taxonomy
    swap risk), both pre-B6 inline blocks fully removed, imports correct, np dropped
    from infer, smoke is faithful (its `old_inline_savez` matches `HEAD:bst_x_train.py`
    L1076-1087 and `HEAD:bst_x_infer.py` L256-269 verbatim). One observation only
    (`_write_prediction_npz` is leading-underscore but imported cross-module -- Python
    convention "private", not enforced).
  - Commit: `192df36`.

- **2026-06-30 -- B7 split train_network setup (`a0ffc89`).** Last Bucket B item.
  - Files: `src/bst_x/bst_x_train.py` (+115 / -81).
  - Change + why: two pure extractions from `train_network`'s SETUP phase, per the split
    pre-analysis + its round-1 adversarial review. `_build_loss_fn(n_classes, class_ls,
    taxonomy, device)` covers the three branches (CE / class-weighted CE /
    adaptive-focal) and OWNS the three fail-loud guards
    (`use_val_improvability_gate`-needs-`adaptive_focal`, `adaptive_focal`-XOR-`class_weights`,
    `adaptive_focal`-requires-`label_smoothing=0`). `_split_param_groups(model)` returns
    raw `(decay, no_decay)` lists; the caller keeps the `[optim]` print + the AdamW
    construction so `hyp.lr` / `hyp.weight_decay` reads stay co-located with the
    optimiser. Loop body and SAVE phase byte-identical to pre-B7 (verifier confirmed
    with `diff` from `[optim]` print onwards: zero). Pre-analysis invariants 3.1-3.6 all
    preserved: setup-only extractions consume zero RNG, ONE AdamW + ONE scheduler still
    built once outside the loop, best-macro atomic block + strict `>` + `==` early-stop
    preserved, `torch.save` -> `load_state_dict` -> `add_hparams` order preserved, no
    AMP / grad-clip / `torch.compile` introduced, `hyp` stays module-global read-only
    inside both helpers, loss object built once and mutated through the loop. Pre-existing
    `best_state` `UnboundLocalError` on degenerate runs preserved. `_to_hparam_value` is
    actually an inline `is_tb_scalar` check, not a nested helper -- left untouched
    either way.
  - Gate (CPU, all green):
    - ruff clean.
    - val-metrics golden 0-diff (regression backstop for the B1 counts path: untouched
      by B7).
    - model bit-exact 0-diff across 5 variants × 464 weight tensors (backstop: B7
      doesn't touch the model graph).
    - pytest 456 / 2 / 19.
    - **Substantive gate: seeded before/after equivalence**
      (`smoke_b7_seeded_train.py`). Two-process pattern: stash the B7 working-tree edit,
      capture (state_dict_sha256, val_at_best) from a seeded 2-epoch synthetic train on
      the pre-B7 (post-B6) tree; stash-pop; run check on the post-B7 tree. SHA
      `25015519f457a532bd1652a32a1250fd13e2d525be94d2d3309ff9ee0fd99d6a` matches both
      ways; val_at_best dict identical down to all 6 keys (epoch=1, macro_f1=0.0476,
      min_f1=0.0, accuracy=0.167, top2=0.333, per_class_f1 same).
  - External verifier: CLEAN over 8 items + an independent re-run of every gate above.
    Confirmed: both helper signatures + bodies; train_network slim body byte-identical
    to pre-B7 from `[optim]` print onwards; the 6 invariants intact; flow simplification
    (`elif/else` -> early returns) is semantically equivalent; `device` threaded through
    `_build_loss_fn`; the per-epoch `update_alpha` -> `validate` -> `apply_val_gate`
    ordering preserved. Smoke decay / no_decay counts (25 / 53 on BST_PPF) stable pre-
    and post-B7; the 27 / 55 figure from the pre-analysis is BST_X (BST_CG_AP); both
    numbers were verified stable across the stash.
  - HPC Leg 1 (bourbaki, venv-bst-x, --device cuda): GREEN. Same seeded smoke as the
    CPU gate, run cross-commit on cuda with deterministic mode
    (`torch.use_deterministic_algorithms(True)` + cudnn flags +
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Capture on main:
    `sha=0265f9354c35dd8fac63cdda156bc06f5f531c7117bd781c39507cd9a38bf070`, val_at_best
    matches the CPU shape but with cuda's own bit-pattern. Check on branch: identical
    SHA + identical val_at_best, per-epoch numbers stable to 4 decimals
    (train_loss=2.668, val_loss=2.367 then 3.054, macro_f1=0.048 both epochs). B7's
    setup helpers don't drift on cuda.
  - HPC Leg 2 (bourbaki A100, venv-bst-x, full deterministic mode): GREEN. Seeded
    single-serial 3-epoch real-data train on `une_v1_14 / split_v2 / b4_diff` collation,
    main vs branch:
    - weights `.pt`: 1 file, SHA-equal across sides.
    - prediction `.npz`: 3 files (train / val / test), per-key value-identical (only
      `run_id` differs by `--run-id`, expected; logits / y_true / y_pred_top1 /
      topk_idx / clip_stems / class_list / serial_no / taxonomy_name all bit-equal).
    - manifest: per-serial `metrics` (`macro_f1`, `min_f1`, `accuracy`,
      `top2_accuracy`, `num_strokes`, `per_class_f1`) + `extra.val_at_best_macro_epoch`
      identical.
    - The CLI doesn't expose `--n-epochs`, so the 3-epoch override was a temp `sed
      s/n_epochs = 80/n_epochs = 3/` on bst_x_train.py each side (reverted via `git
      checkout` after the run). The Hyp on main is still the pre-Batch-4a untyped
      namedtuple form (`n_epochs=80,`), so the main-side sed pattern differs from the
      branch-side (`n_epochs: int = 80`).
    - 3 epochs is sufficient for a bit-exact gate (the loop body is byte-identical
      pre/post-B7; identical seed + identical data = identical outputs at any epoch
      count).
    - This run jointly closes B2 + B4 + B7 (the une_v1_14 bit-exact line); bst_25 was
      already covered by B4's `collation_fulldiff.py` pass (collation-side) + B2's
      `bst_x_infer --fe` IDENTICAL (forward-side) and the seeded GPU smoke ruled out
      B7-specific cuda drift.
  - Commit: `a0ffc89`.

## Final independent audit

Fresh general-purpose agent audited the full branch diff (16 commits, 33 files,
+1280/-1148) against the OUT-list, the six non-mechanical claims, every per-batch
worklog entry, the HPC gate coverage table, and re-ran every harness golden + the
full pytest. Verdict: CLEAN modulo one MED finding (a stale doc row in
`data_pipeline_to_model_train.md` that still claimed `RandomTranslation_batch` fired
in `train_one_epoch`; Batch 1 deleted it). Plus a couple of LOW-severity stale
references in the same doc table. All cleared by commit `74bc547` ("Stop the
data-pipeline doc claiming RandomTranslation_batch still fires") which folded the
live augmentations (`CoupledFlip` + `ConstrainedJitter`) into both rows.

Audit also re-ran every harness golden 0-diff (val-metrics, extraction 44-row,
collation bit-identical, model bit-exact across 5 variants × 464 tensors, Tier-1 AST
check across the 15 Batch-3b files), pytest 456 / 2 / 19, ruff clean.

## Merge

`18e5c2c` ("Merge simplification + cleanup pass"), `--no-ff`, 2026-06-30. Preserves
the per-batch commit topology under a single merge commit at main's tip. Post-merge
sanity check on `main`: pytest 456 / 2 / 19, working tree clean, ruff clean.

Seven Bucket A commits (D3 + Batches 1-4) + the post-HPC stale-paths cleanup
(`09e4215`) + seven Bucket B commits (B1-B7) + the audit-fix doc commit (`74bc547`),
all landed on `main` at `18e5c2c`.
