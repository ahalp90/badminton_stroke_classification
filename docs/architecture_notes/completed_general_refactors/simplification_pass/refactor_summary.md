# Simplification pass summary

A behaviour-preserving cleanup of the BST-X model + pipeline (~12k LOC). Landed
on `main` 2026-06-30 under merge commit `18e5c2c`.

## Why this pass

The codebase carried two kinds of clutter:

1. **Dead code and stale lint suppressions.** Two unused augmenters, an unused
   shuttle-trajectory extractor, an unused `_resolve_or_none` helper, three
   `# noqa: F401` directives that lied about imports being unused. Symbols
   nothing called but every reader had to load.
2. **TF-analogue comments.** `bst.py` was 31% comment lines, most of it
   `= tf.keras...`-style analogies written while picking up PyTorch from a TF
   background. Useful at the time.

Plus seven structural items the review flagged: a vectorised class-count helper
`validate()` wasn't using, three test-set forward passes per serial that should
have been one, three near-duplicate court-homography helpers, a ~250-line
`collate_npy`, a ~420-line `train_network` setup, two near-duplicate
prediction-npz writers, and a 2D / 3D player-detector pair with an identical
middle section.

Full findings tables: `simplification_review.md` and
`comment_density_review.md` alongside this doc.

## What was delivered

Seventeen commits on `refactor/simplification-pass`, grouped into three buckets:

**A: dead-code and small mechanical edits.** Batches 1-3 plus the lazy-import
fix that unblocked CPU testing.

- `88405ed` Make the mmpose import lazy so the pipeline imports without a GPU.
  The modules that never use mmpose stop pulling it in at import; the CPU
  goldens then build and run without stubs.
- `34a1f01` Delete dead code. The two `RandomTranslation` augmenters, the
  `extract_shuttle_trajectory` (the `_with_attention` sibling stays), the
  `_with_rectification` shuttle detector, `_resolve_or_none`, the
  `skipped_missing_raw` field, and the import cascade they pinned. ~150 LOC out,
  zero behaviour change.
- `77c7cdb` Tidy small mechanical bits. The three stale `# noqa: F401`, the
  `VALID_CLIP_WINDOWS` set, `J = 17` single-sourced into `heuristics/base`,
  inline TSV-print dedup, `_pad_tail_to` helper for the `np.pad` duplicate,
  `aux_schedule_factor` rewritten to plain ifs.
- `8591396` Compute top-1 predictions from argmax everywhere. The prediction-dump
  was the only top-1 site using `topk[:, 0]`; every other site uses argmax, and
  two consumers already asserted they're equal. Now they hold by construction.
- `2ca8214` Trim bst.py comments and rename the player/stream dims. The big
  comment strip plus splitting `forward()`'s overloaded `n` into `n_people` (=2)
  and `n_streams` (=3). Replaces the TF analogues with a short PyTorch-idiom
  glossary header.
- `b9ea158` Strip stale comments and docstrings across the pipeline. Cuts the
  TF analogues, the newcomer notes, and the comments that just restated the
  code, across 15 files. Keeps every shape annotation, every maths "why", every
  invariant note.
- `ede5a0d` Type the Hyp hyperparameter container. Functional `namedtuple` to
  typed `NamedTuple` with per-field defaults; the 23 fields keep their order +
  values bit-identically.
- `7cd4b41` Move the mmpose-zeroing failsafe check to validation_scripts. It's a
  one-shot byte-identity checker, belongs with the other validation checkers.
- `09e4215` Stop the mmpose bit-exact test gates defaulting to the sticky_anchor
  extract. Closes a stale-env-var trap the HPC pass hit twice. The gates assumed
  the env var pointed at a `current`-equivalent extract; the 2026-04-29 flip to
  sticky_anchor changed what it pointed at. Made `--committed-dir` required and
  rewrote the run-command docs to the dual-invocation pattern.

**B: structural splits.** B1-B7, the seven items each landing under its own
commit + own gate.

- `01b1a80` Reuse the vectorised class-count helper in validate(). Replaces the
  hand-rolled one-hot TP/FP/FN block with `accumulate_class_counts` (the same
  helper train_one_epoch already uses). Cuts ~20 lines + the only `F.` user in
  the file. Counts live on the GPU through the loop with one `.cpu()` after, so
  no per-batch device hop.
- `194783b` Run the test-set forward pass once, not three times. Per-serial cost
  drops from three test-loader passes (argmax, topk=2, topk=5) to one. `Task.test`
  reads top-1 off the dump; `Task.test_topk_acc` re-derives top-2 from the stored
  logits with a fresh `torch.topk(k=2)` (NOT a slice of the k=5 dump -- those
  tie-break differently).
- `b3e242b` De-duplicate the court homography helpers. Nine court helpers were
  triplicated across `prepare_train_on_shuttleset`, `pipeline/court_utils`, and
  `src/shared/court.py` (the BRIC mirror). Drops the prepare_train copies, keeps
  the canonical court_utils versions + the BRIC mirror untouched. Net 3 copies
  to 2.
- `5b361b8` Split the oversized collate_npy into clear stages. The ~250-line
  orchestrator now chains four named helpers: resolve clips + labels, load the
  per-clip npys, align shuttle + truncate, pad-augment-stack-save. Behaviour
  preserved; the gate is per-clip element-wise identity across ~66k clips, both
  taxonomies, all splits.
- `41719e4` Share the common body of the 2D and 3D player detectors. The
  `< 2`-keypoint short-circuit, the on-court check, the strict-`>` flip, and
  the player-ordering live in `_order_two_on_court(...)`. Shape-agnostic by
  design; the 2D bit-exact transitively covers the shared arithmetic for both.
- `192df36` Share the prediction-dump writer between train and infer. The 9-key
  `np.savez` payload had identical inlines at both writers; lifted to
  `_write_prediction_npz`. Keeps each writer's own dir / filename / split-loop /
  manifest plumbing.
- `a0ffc89` Split setup helpers out of train_network. Two pure extractions from
  the SETUP phase: `_build_loss_fn` (CE / class-weighted CE / adaptive-focal,
  with the three fail-loud guards) and `_split_param_groups` (decay vs no-decay
  for AdamW). Loop body and SAVE phase byte-identical to pre-split.

**C: audit follow-up.** One commit folding the final-audit doc finding.

- `74bc547` Stop the data-pipeline doc claiming RandomTranslation_batch still
  fires. Audit caught a stale row in `data_pipeline_to_model_train.md`; updated
  to the live augmentations (`CoupledFlip` + `ConstrainedJitter`).

## What stayed put on purpose

- The loud-fail guards at silent-corruption boundaries. Standing project rule:
  fail loud where a silent contract violation could reach a downstream consumer
- `adaptive_focal.py`'s maths-invariant docstrings
- The BRIC court mirror (`src/shared/court.py`). Duplicated for the BRIC pipeline
  on purpose; the dedup only removed prepare_train's copy
- The heuristic registry. Already functions + a dict, no inheritance to unpick
- The `init_weights()` positional-encoding loop. Considered and declined; the
  explicit three-block form reads clearer
- The `MultiHeadCrossAttention` / `tempose.MultiHeadAttention` unify. Considered
  and declined; kept separate so the Chang's-BST lineage stays traceable
- The `_validate_inputs` in `build_dataset.__main__`. Was on the drop list;
  dropping it removes the dry_run path's only input validation, so it stays

## How this was gated

Every batch had two layers:

1. **CPU gate.** Per-batch: ruff clean, the relevant captured-from-main golden
   (model bit-exact, val-metrics, extraction 44-row, collation 5-clip × 9
   outputs, Tier-1 AST comment-only check), pytest 456 / 2 / 19. Each gate
   self-verified 0-diff on main before the pass started; check-mode 0-diff
   proves the edit was behaviour-preserving.
2. **Independent verifier.** After each batch, a fresh review agent re-grepped
   the diff against the spec, the OUT-list, and the keep-list (no shared state
   with the executing agent). Their adversarial verdict went into the worklog
   alongside the gate result.

Then the HPC bit-exact layer for the gates CPU can't run:

- `bst_x_infer --fe` on main vs branch, 4,202 test clips, `y_pred_top1` + `logits`
  IDENTICAL by `np.array_equal`.
- `validate()` on cuda over a synthetic two-batch loader: no device mismatch, the
  Option B discipline holds.
- `collate_npy` full-dataset per-clip diff on `une_v1_14 / split_v2` (32,203
  clips) and `bst_25 / split_bst_baseline` (33,481 clips). Element-wise identical
  at `atol=0.0`.
- 2D pose smoke (live MMPose + RTMPose-L + GPU on real mp4s), dual-invocation
  main writes the reference, branch compares against it. 10/10 stems, every stem
  `pos_max=joints_max=0.000e+00`.
- Failsafe data run from its new location, 50/50 stems, `0.000e+00` on the pos +
  joints arrays.
- Seeded single-serial 3-epoch real-data train on main vs branch: weights .pt
  SHA-equal, prediction npzs per-key value-equal (only `run_id` differs by flag),
  manifest metrics + val_at_best identical.

The 3D path stayed accepted-deferred: `use_3d_pose` is False everywhere deployed,
no 3D collated artifact, no 3D raw extract. If 3D is ever revived, the smoke is
parked at `harness/smoke_prepare_3d_runs.py` in the planning dir.

## What got measured

Branch diff against `main`: 17 commits, 33 files, +1280 / -1148. Net ~130 fewer
lines despite seven structural splits adding helper boilerplate; the comment
strip + dead-code deletion drove the line count.

`bst.py` from 31% comment lines to ~12%, with the TF analogues replaced by a
one-paragraph PyTorch-idiom glossary. `collate_npy`: four helper bodies in the
60-90 line band each, down from one 250-line block. `train_network`: setup in
two helpers, loop body untouched.

## Final audit

Fresh review agent diffed the full branch against `main`, checking against the
OUT-list, the six non-mechanical adversarially-verified claims, and the
per-batch worklog. Then re-ran every harness golden + the full pytest. Verdict:
clean apart from one MED finding (the doc row above, fixed by `74bc547`) plus a
couple of LOW-severity stale references in the same doc table.

## What this pass didn't touch

Vendored `TrackNetV3/`. The diagnostic `validation_scripts/`. Operational tooling
(`hparam_sweep.py`, `run_tracker.py`, `result_utils.py`,
`build_fe_stats_jsons.py`, the backfills). BRIC (`src/bric/`). The API
(`src/api/`). `src/shared/`. Each was on the OUT-list in the runbook and stayed
there.
