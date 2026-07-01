# Simplification review

> _Last verified: 2026-06-29 against pre-pass `main`. This is the review that
> drove the simplification pass (merged at `18e5c2c` on 2026-06-30). Findings
> below are the durable record; the line refs were pre-pass and have shifted._

Adversarial pass over the BST-X model and pipeline (~12k LOC), looking for code
that could come out or get simpler without changing behaviour. Each "delete this"
finding was checked against the live call sites; the stale-lint findings were
confirmed with `ruff --select RUF100`. The companion comment-density review
covers the comment-clutter pass and is in `comment_density_review.md`.

## Scope

Covered: the BST-X model (`model/`), loss (`loss/`), training and inference
(`bst_x_train.py`, `bst_x_infer.py`, `bst_x_common.py`), data prep
(`preparing_data/`), and the whole `pipeline/` package.

Not covered: the vendored `TrackNetV3/`, the diagnostic `validation_scripts/`,
the operational tooling (`hparam_sweep.py`, `run_tracker.py`, `result_utils.py`,
`build_fe_stats_jsons.py`, `aim_backfill.py`, `backfill_val_metrics.py`,
`collation_runner.py`, `run_overview.py`), BRIC (`src/bric/`), the API service
(`src/api/`), and `src/shared/`.

## Quick wins (all actioned)

| # | File / location | Change |
|---|-----------------|--------|
| 1 | `preparing_data/shuttleset_dataset.py:113-144` | Delete `RandomTranslation` and `RandomTranslation_batch`. Drop the now-unused `import torch`, `from torch import Tensor`, `from torchvision.transforms import v2`. 0 code callers; the live augs are `CoupledFlip` / `ConstrainedJitter`. |
| 2 | `pipeline/shuttle_extractor.py:58-122` | Delete `extract_shuttle_trajectory` (~65 lines). The superseded per-clip TrackNet entry point. The wired `extract_all_shuttles` batch path covers the same work via the qaz812345 fork's `batch_predict.py`; the `_with_attention` sibling stays as the live caller. |
| 3 | `preparing_data/apply_heuristic.py:68-69` | Delete `_resolve_or_none`. 0 callers across `src/` and `tests/`. |
| 4 | `pipeline/clip_generator.py:16` | Delete `import numpy as np  # noqa: F401`. `np` is never referenced; the `# noqa` was masking a dead import. |
| 5 | `pipeline/build_dataset.py:25` | Remove the `# noqa: F401` (keep the imports). RUF100 flags it: every name is used. |
| 6 | `pipeline/clip_generator.py:22` | Remove the `# noqa: F401`. Every name is used. |
| 7 | `pipeline/verify.py:16` | Remove the `# noqa: F401`. Every name is used. |
| 8 | `bst_x_train.py` `validate()` | Delete the `tn` / `cum_tn` accumulation. Computed, never read. Pure dead compute on every val batch. (Subsumed by the `validate()` reuse below.) |
| 9 | `preparing_data/apply_heuristic.py:64` | Drop `RunStats.skipped_missing_raw`. Never incremented, never read; `_build_stem_list` filters to raw-present stems first, so the counter is structurally unreachable. |
| 10 | `preparing_data/apply_heuristic.py:335` | Reword `return 0 if stats is not None else 1` to `return 0 if stats else 1`. `RunStats` is a plain dataclass; `stats` is always truthy here. Keeps the failsafe, reads cleaner. |

Net: ~150 lines out, four stale lint suppressions cleared.

## Per-module findings

### `model/bst.py`

Mostly a comment-density problem (covered in `comment_density_review.md`), not
a structure problem. Two small simplifications looked at; one applied, one
declined.

- **Applied**: rename `forward()`'s overloaded `n`. At line 275
  `b, t, n, in_dim = JnB.shape` gives `n = 2` (players); at line 307
  `_, n, _, d = x.shape` rebinds `n = 3` (streams: p1, p2, shuttle). Renamed to
  `n_people` and `n_streams`. Stops a reader tripping on why
  `repeat_interleave(n, ...)` later uses 3 not 2.
- **Declined**: looping `init_weights()`'s positional-encoding copy block. The
  explicit three-block form reads clearer than a `for embedding in (...)` loop.

### `loss/adaptive_focal.py`

Clean on the simplification lens. The per-class `for c in
range(self.n_classes)` loop in `apply_val_gate` carries per-class
seeding / continue / best-update branching that doesn't vectorise cleanly, and
`n_classes` is ~14-25. Leave it. Comments here are load-bearing maths
rationale.

_(2026-07-01: `apply_val_gate` was subsequently removed with the
`remove-adaptive-focal-extensions` rip, merged at `39e72a5`. This note is
retained as a historical snapshot of the simplification review's coverage.)_

### `bst_x_train.py`

- **Applied**: `validate()` reuse of `accumulate_class_counts`. The hand-rolled
  one-hot TP / FP / FN block (~328-341) duplicated work that
  `accumulate_class_counts` (already imported, used in `train_one_epoch`) does
  vectorised. Replaced; `.cpu()` the returned counts before accumulating since
  the helper returns on the input device. Subsumes quick win #8 (`tn` goes
  too).
- **Applied**: collapse three forward passes over `test_loader` into one.
  `Task.test()` (argmax) + `Task.test_topk_acc()` (top-k) + `dump_predictions`
  (logits + top-1 + top-k + y_true) ran three full passes per serial. Now the
  dump runs once; top-1 reads off `dump['y_pred_top1']` (argmax, unified
  upstream in the same pass); top-2 re-derives via a fresh `torch.topk(k=2)`
  on the stored logits (not a slice of the k=5 dump, which tie-breaks
  differently). Cuts two passes per serial.
- **Applied**: split setup helpers out of `train_network` (~420 lines).
  Extracted `_build_loss_fn(...)` (the CE / class-weighted CE / adaptive-focal
  branch, with the three fail-loud guards) and `_split_param_groups(model)`
  (the decay / no-decay walk). Loop body and save phase byte-identical
  pre / post. Confirm-first, since this is the training entry point.
- **Applied**: `aux_schedule_factor` rewritten to two plain ifs (the double
  ternary under an `or` was opaque).
- **Applied**: `_to_hparam_value` (the 4-line nested def used once) inlined to
  the dict comp. Folded into the small-mechanical-edits batch.

### `bst_x_infer.py`

- **Applied**: share the `np.savez` prediction-dump payload between
  `Task.dump_predictions` (training) and `dump_run_predictions` (inference).
  The 9-key schema was inlined at both writers; lifted into a shared
  `_write_prediction_npz(out_path, dump, dataset, taxonomy, run_id, serial)`
  in `bst_x_common.py`. Helper owns only the savez payload + the
  `clip_stems is not None` hard-fail (the `np.asarray(None)` silent-0-d trap
  was duplicated at both sites). The dir, filename, split-loop, and per-caller
  manifest plumbing stay per-writer. The runtime k-hard-assert was loosened
  to a documented caller-side convention (both production writers pass k=5);
  the assert broke four legitimate `tests/test_train_surface.py` cases that
  exercise k=2 and k=5-clamped-to-3.

### `preparing_data/shuttleset_dataset.py`

- **Applied (via quick win #1)**: dead `RandomTranslation` classes.
- **Applied**: de-duplicate the `np.pad` block in `make_seq_len_same`. The
  three lines appeared verbatim in both `if need_padding` and the `else`
  branch. Pulled into `_pad_tail_to(target_len, joints, pos, shuttle)`,
  early-returning when no padding is needed.

### `preparing_data/augmentations.py`

Clean on the simplification lens. `CoupledFlip` and `ConstrainedJitter` are
already vectorised with named masks; the constraint maths is genuinely
non-trivial. `_coco_swap_index` rebuilds a 17-element index every call, but
it's tiny and device-dependent, so caching would add state for no real gain.
Leave it.

### `preparing_data/prepare_train_on_shuttleset.py`

Heaviest data-prep file (~1290 lines).

- **Applied**: collapse the court-helper triplication. Nine court helpers were
  triplicated across this file, `pipeline/court_utils.py`, and
  `src/shared/court.py` (the BRIC mirror). Adversarial verification confirmed
  all three byte-identical on every real homography row plus the downstream
  chain (the doc had wrongly worried about a `.copy()` divergence and
  `np.fromstring` deprecation; both ruled out). The clean move was to re-import
  the court helpers from `pipeline.court_utils` and re-export `normalize_joints`
  / `check_pos_in_court` (`prepare_train`-only, so they stay). Three copies
  down to two; the BRIC mirror stayed deliberate.
- **Applied**: share the `detect_players_2d` / `_3d` middle band. Both
  functions share the `< 2`-keypoint short-circuit, the `check_pos_in_court`
  call, the `!= 2 on court` branch, the Top-before-Bottom strict-`>` flip, and
  the `np.stack` tail; they differ only in 2D vs 3D keypoint extraction and
  array shape. Lifted into `_order_two_on_court(...)` returning
  `(in_court_pid, pos_normalized)` or `None`. Shape-agnostic by design, so
  the 2D bit-exact transitively covers the shared arithmetic for both callers.
- **Applied**: split the oversized `collate_npy` (~250 lines) along its four
  concerns: CSV filter + label derivation, threaded npy load, shuttle read +
  temporal align, pad / augment + stack + save. Function invariants and gate
  coverage in `../../function_invariants/collate_npy.md`.
- **Applied**: delete `detect_shuttlecock_by_TrackNetV3_with_rectification`
  (447-487). 0 callers; the wired `..._with_attention` sibling covers the
  modern batch path, and the rectification flag (`--inpaintnet_file`) is
  already supplied by `extract_all_shuttles` when the weights exist. The other
  rectification flag (`--large_video`) is pointless on ~0.5-3s clips. Legacy
  Chang's-BST curio, nothing to keep.
- **Applied**: simplify the `min_t` guard in `collate_npy`. `min_t =
  min(len(failed), len(shuttle))`; the original `if min_t < len(failed) or
  min_t < len(shuttle):` is true exactly when the lengths differ. Rewritten to
  `if len(failed) != len(shuttle):`; the slices keep `[:min_t]`.
- **Kept both `pose_styles` validations**: at `main()` parse time and at the
  top of `collate_npy`. The two checks guard distinct entry points (CLI vs
  direct / library / test caller), so keeping both fits the fail-loud
  boundary rule.
- **Kept `raw_extract.py` as-is**: tight, well-decomposed.

### `preparing_data/apply_heuristic.py` + `heuristics/`

- **Applied (quick wins #3, #9, #10)**.
- **Applied**: hoist `from dataclasses import fields` to module scope.
  `dataclass` was already imported at module scope; the local imports in the
  two helpers (266, 278) carried only a "keeps imports tidy" comment. Merged
  to `from dataclasses import dataclass, fields`.
- **Applied**: single-source `J = 17`. Twinned in `current.py:41` and
  `sticky_anchor.py:69`, with `base.py`'s docstrings already referencing `J`
  in shape strings. Put `J = 17` in `base.py` and import.
- **Declined**: ABC over the heuristic registry. No inheritance to unpick;
  `base.py` holds three plain data containers, and `__init__.py` is already a
  `REGISTRY = {name: fn}` dict over free `apply(...)` functions. The variants
  share types, not implementation. Leave it.
- **Applied**: relocate `failsafe_bst_mmpose_zeroing_check_equivalence.py`.
  One-shot checker living next to a shipped pipeline entry point; nothing
  imports it. Moved to `src/bst_x/validation_scripts/`. (Post-HPC stale-paths
  cleanup also dropped the `$BST_X_MMPOSE_NPY_DIR` env fallback and made
  `--committed-dir` required; the env var had pointed at `sticky_anchor` since
  2026-04-29, so the silent fallback was comparing `current` against
  `sticky_anchor`.)

### `pipeline/config.py` + `data_access.py` + `clip_index.py`

- **Kept**: the taxonomy tables in `config.py`. Deliberate single-source-of-truth
  and genuinely used. The `raw_35` alias is inert but harmless one-line; keep.
  Every other alias is load-bearing. Keep the loud guards
  (`__post_init__` ValueError, `resolve_taxonomy` KeyError, `derive_class_index`
  re-raise).
- **Applied**: extract the duplicated TSV-print loop in `data_access.py`. The
  ~9-line path-printing block was verbatim in `interactive()` (545-552) and
  `main()` (655-662). Pulled into `_print_paths_tsv(records)`. The
  `df.itertuples` loop in `get_clip_records` stays (I/O-bound per-row merge +
  `Path.exists` stats; not a vectorisation candidate).
- **Kept**: `clip_index.py` is one clean pure function wrapped in
  forward-looking doc. The trim here is a comment-density one.

### `pipeline/build_dataset.py` + `clip_generator.py`

- **Applied (quick wins #4, #5, #6)**.
- **Kept**: `build_dataset.py`'s `_validate_inputs` in `__main__`. Was on the
  drop list; dropping it removes the `dry_run` path's only input validation
  (`dry_run()` doesn't validate internally), so it stays. Existing comment
  documents it.
- **Applied**: `clip_window` re-validation -> module-scope
  `VALID_CLIP_WINDOWS = frozenset({...})` shared by the guard and the CLI
  `choices=`, so the literal lives in one place and the loud fail stays.

### `pipeline/shuttle_extractor.py` + `download_videos.py` + `verify.py`

- **Applied (quick wins #2 and #7)**.
- **Applied**: simplify `if inpaintnet_path and str(inpaintnet_path):` (106) to
  `if inpaintnet_path:`. A non-empty `Path` is always truthy and `str(Path)` is
  never falsy. (Moot after quick win #2; the live batch path at 214 already
  does it cleanly.)
- **Applied**: drop the redundant `except FileNotFoundError` (81-83) in
  `download_videos.py`. `download_all_videos` runs `_check_ytdlp()` before
  spawning any worker, raising a clean `RuntimeError` if `yt-dlp` is missing.
  By the time a worker runs, `yt-dlp` is guaranteed present, so this per-call
  branch guarded an excluded state and duplicated the install message. The
  generic `except Exception` still catches the genuinely unexpected.
- **Decided (verify.py)**: no 0-byte clip check in `build_dataset.py` Step 5;
  add a documenting comment instead. Adding a check to Step 5's list would
  `sys.exit(1)` on the first 0-byte clip and abort the batch before Step 6
  starts. The downstream `batch_predict` already skips an unreadable clip
  (just no CSV for it), and the worker non-zero exit is a WARNING, never an
  abort. Step 5 now carries `# Clip size non-zero checked after full extract,
  allowing the batch to finish first.`

## Cross-cutting themes

- **Grep methodology fixed pre-pass.** `.gitignore:42`
  (`src/bst_x/preparing_data`) used to make `.gitignore`-honouring search
  (`rg`, `fd`, VS Code) silently skip that dir, so "0 callers" checks needed
  `git grep`, GNU `grep -r`, or `rg --no-ignore`. Line 42 was removed (commit
  `3b9bd5e`); plain `rg` now sees `preparing_data/`. Re-run greps at execution
  time anyway, for drift.
- **Stale / dead `# noqa: F401` (four sites).** Cleared. A suppression that
  lies about an import being unused is worse than none. Add `RUF100` to the
  ruff config to stop these re-accumulating.
- **Provenance comments citing other files' line numbers rot.** Several
  examples were caught; cite the function name, not the line.
- **The court helpers' three-place existence is part deliberate, part not.**
  The `pipeline` vs `shared` split is the BRIC self-containment mirror and
  stays. The `prepare_train` private copy was the genuine consolidation seam.

## Confirm-first refactors

Items that touch model maths, a hot path, or many call sites. Authorisation
gathered at review time; outcome recorded here.

- **Declined: unify `MultiHeadCrossAttention` with
  `tempose.MultiHeadAttention`.** ~90% identical; the only real difference
  is cross-attn projects `to_q` + `to_kv` (queries from x1, keys / values
  from x2) while self-attn projects one `to_qkv` on a single input. A shared
  base would remove ~40 duplicated lines. Kept separate so the Chang's-BST
  lineage stays traceable: both paths are live (self-attn inside
  `tempose.TransformerEncoder`, cross-attn fusing player + shuttle in
  `MultiHeadCrossAttention`), and the duplication is honest. No drift expected.
- **Applied: extract `_build_loss_fn` + `_split_param_groups` from
  `train_network`.** Pure structure, no behaviour change, confirmed-first
  because the training entry point. Function invariants and gate coverage in
  `../../function_invariants/train_network.md`.
- **Applied: type the `Hyp` container.** Switched the functional
  `namedtuple('Hyp', [...])` to a typed `class Hyp(NamedTuple)` with per-field
  type + inline default. `_asdict()` and `_replace()` still work, so zero
  call-site edits. Frozen dataclass was rejected because it would break the
  four `_asdict` / `_replace` sites for no extra readability.
- **Applied: collapse the court triplication** (per per-module note above).
- **Applied: relocate the failsafe checker** (per per-module note above).

## What was left alone (declined or out of scope)

Listed so it's clear they were checked, not missed.

- **All loud-fail guards.** `assert dataset.clip_stems is not None`,
  `resolve_taxonomy` KeyError, the mutual-exclusion `raise`s, the retirement
  boundary guard in `player_mapping.py`, the shape-mismatch guard in the
  failsafe. Standing project rule: fail loud at silent-corruption boundaries.
- **`adaptive_focal.py` docstrings and comments.** They document non-obvious
  maths invariants (renorm to mean 1.0, the one-sided revert, the pair-cap
  budget). That's why-content, not clutter.
- **`pipeline/court_utils.py` and `pipeline/player_mapping.py`.** Clean,
  vectorised, and the apparent duplication with `src/shared/*` is an
  intentional mirror for BRIC self-containment, documented in both module
  docstrings.
- **`raw_extract.py`.** Tight, well-decomposed; the per-person fill loop is
  the readable choice over a vectorised stack at `n <= 8`.
- **The heuristic registry.** Already functions + a dict, no inheritance to
  unpick.
- **`env_path` vs `env_path_or_none`** (in `data_access.py`). Two named
  functions read better at the call sites than one merged function with a
  sentinel default.
- **`raw_35` taxonomy alias.** Inert but harmless one-line.
- **The 3D detection path.** `use_3d_pose=False` everywhere deployed; the 3D
  function is reachable via CLI but dormant. The `_order_two_on_court`
  shape-agnostic split transitively covers the shared arithmetic; the 3D
  caller still needs review by a human re-entering the function.
