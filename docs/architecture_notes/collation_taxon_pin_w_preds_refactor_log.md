# Collation taxon_pinned_w_preds refactor log

Running log for the taxon_pinned_w_preds refactor. Companion docs: `docs/architecture_notes/collation_taxon_pin_w_preds_refactor.md` (full plan, locked decisions, touch surface inventory), `scratch/taxon_pin_w_preds_tldr.md` (FE-facing summary).

Per-phase blocks below. Commit blocks added once code lands.

## 2026-05-23: pre-flight audit grep

Repo-wide grep for the deprecated symbol surface, before any code edits. Five patterns covering:

- Taxonomy methods being removed (`class_list()`, `active_class_list`, `n_active_classes`, `standalone_set`, `unknown_first`, `full_to_active_remap`)
- Train-surface helpers being removed (`derive_active_classes_from_labels`, `_validate_and_record_arch`, `expected_active_classes`)
- Removed defaults (`DEFAULT_TAXONOMY`, `drop_unknown`, `derive_ablation_id`)
- Legacy TAXONOMY_* constants (`TAXONOMY_UNE_MERGE_V1`, `TAXONOMY_MERGED_25`, `TAXONOMY_RAW_35`, `TAXONOMY_UNE_MERGE_V1_NOSIDES`)
- Direct readers of the old buggy `MERGE_MAP[`

### Expected hits (already in hit list)

- `src/api/registry.py` lines 96, 204, 233 (Step J)
- `src/api/inference.py` lines 34, 44 (Step J)
- `src/bst_x/pipeline/config.py` full taxonomy block (Step A)
- `src/bst_x/pipeline/data_access.py` imports + `_derive_class_label` + four `class_list()` callsites (Step A)
- `src/bst_x/pipeline/build_dataset.py` import + defaults (Step A)
- `src/bst_x/pipeline/clip_generator.py` import + default (Step A)
- `src/bst_x/pipeline/verify.py` import + default (Step A)
- `src/bst_x/bst_x_train.py` full Hyp + Task + train_network + manifest surface (Step D)
- `src/bst_x/bst_x_common.py` derive_active_classes_from_labels def (Step D)
- `src/bst_x/bst_x_infer.py` argparse + DEFAULT_TAXONOMY refs (Step D)
- `src/bst_x/preparing_data/prepare_train_on_shuttleset.py` full collator surface (Step C)
- `tests/test_active_classes.py`, `tests/test_data_access.py`, `tests/test_integration.py` (Step F)
- `scratch/presentation_prep/eval_dump_predictions.py` delete (Step D10)
- `scripts/plots/confusion_matrix.py` adapt to npz (Step D10)

### Unexpected hits — scope additions

Four code files and four doc files not in the original hit list.

**Code:**

1. `src/bst_x/model/bst.py` lines 436-437 — `__main__` demo block imports `TAXONOMIES, DEFAULT_TAXONOMY` and reads `n_classes` for the smoke instantiation. Trivial: replace with `resolve_taxonomy('bst_25').n_classes` or similar concrete pick. ~2 lines.

2. `src/bst_x/validation_scripts/fail_rate_per_class.py` — live validation script with `--drop-unknown` CLI flag piped through to `get_clip_records`. Needs adaptation: drop the CLI flag, switch to `--taxonomy` selection (taxonomy now carries the unknown-exclude rule via `excluded_base_stroke_types`). ~10 lines.

3. `src/bst_x/validation_scripts/verify_bst_train_target.py` — live pre-flight script importing `derive_ablation_id` + `derive_npy_collated_dir_basename`. Both signatures change. ~5 lines.

4. `scripts/api_fixtures/build_mock_artifacts.py` lines 29, 204 — writes mock JSONs with `active_class_list` field. The Step J fallback handles reads, so this is low-priority. Either update for consistency or leave (fallback covers).

**Docs:**

5. `src/bst_x/data_pipeline_to_model_train.md` lines 94, 398, 544 — references `DEFAULT_TAXONOMY`, deprecated taxonomy methods, `drop_unknown`.
6. `src/bst_x/pipeline/README.md` lines 248, 253, 254, 288, 322, 323, 376 — same deprecated-symbol refs.
7. `tests/testing_guide.md` lines 21, 51 — references `_derive_class_label` semantics and the `ablation_id` naming convention.
8. `docs/architecture_notes/xai_vid_feature.md:132` — references the old `.pt` predictions schema's `active_class_list` field.

### Hits ignored (out of scope or no impact)

- `src/api/bst_x_inference.py:59` — single comment; the plan parks this file's hardcoded 14-class list as a forward FE concern.
- `src/bric/*`, `src/shared/*` — Arch2 territory, off-limits per agreed scope. Arch2 has its own Taxonomy class with its own `standalone_set` / `unknown_first` / `class_list()`; no symbol overlap with BST.
- `tests/test_network.py`, `tests/test_video_io.py`, `tests/test_player_mapping.py`, `tests/test_temporal.py` — Arch2 tests.
- `tests/test_adaptive_focal.py:49` — single comment, no code dependency.
- `scripts/archive/*` (verify_v1_collate.py, verify_flatten.py) — archived scripts, won't run.
- `src/bst_x/.gitignore` — comment lines from project history.
- Manifest YAML / `best_model_id.txt` under `experiments/run_*/` — historic artefacts; new runs won't write the deprecated fields; aliases handle resume.
- `predictions/val.json`, `predictions/test.json` — covered by Step J fallback.

### Net scope impact

Original hit list: ~22 code files + 4 docs.
After audit: +4 code files + 4 docs.

Largest additions: `fail_rate_per_class.py` (~10 lines), `verify_bst_train_target.py` (~5 lines). Others are 1-3 line touches.

The plan doc's "Files audited and confirmed clean" inventory at lines 1238-1245 was incomplete: `fail_rate_per_class.py` and `verify_bst_train_target.py` were listed under validation_scripts as clean but actually use `drop_unknown` / `derive_ablation_id`. Plan doc to be updated alongside the Step A commit.

### Decisions on the scope additions (2026-05-23)

- `bst.py` __main__ demo: trivial 2-line fix, approved.
- `fail_rate_per_class.py`: drop `--drop-unknown` CLI flag entirely, switch to `--taxonomy` selection (the new Taxonomy carries the unknown-exclude rule via `excluded_base_stroke_types`). If any other little validation script gets messy on the back-compat path, refactor for forward compatibility and drop back-compat — these scripts are project-internal, no external consumers.
- `verify_bst_train_target.py`: fix the signature updates.
- `build_mock_artifacts.py`: update to write `class_list` field. Forward-thinking, "refactoring for next month not last month."
- The four doc fixes (data_pipeline_to_model_train.md, pipeline/README.md, tests/testing_guide.md, xai_vid_feature.md): defer to the end of the refactor as "finishing touch surfaces." Other doc surprises encountered along the way get catalogued here under a Finishing touch surfaces section.

### Locked naming (final)

- Taxonomy field: `excluded_base_stroke_types: frozenset[str]` (noun form fits a frozenset attribute, not a function).
- Helper function: `dump_full_logits(model, loader, device) -> dict[str, np.ndarray]`. Returns `{logits, y_true}`. Metadata (`class_list`, `run_id`, `serial_no`, `taxonomy_name`) added at the npz save site (in `Task.dump_predictions`), not inside the helper.
- npz schema: `logits, y_true, class_list, run_id, serial_no, taxonomy_name`. Dropped `y_pred_top1` and `topk_idx` (derivable via `np.argmax(logits, axis=1)` and `np.argsort` in one line each).
- `ablation_id` rename to `collation_id`: deferred to a separate pass after taxon_pinned_w_preds lands. Docstring at the Hyp field documents the name drift and the planned rename.
- `STROKE_TYPES_*` constants: keep current naming pattern (`STROKE_TYPES_<count>_<provenance>`), add a section docstring at the top of the taxonomy block naming the convention.

### Inference smoke behaviours (CI surface, replaces deleted smoke_infer_bit_exact.py)

a) Loaded weights produce a deterministic output for a fixed input batch
b) Output shape == taxonomy.n_classes
c) Output dtype float32; no NaN/Inf
d) Top-1 argmax of output is in [0, n_classes)
e) `Task.dump_predictions` npz: row count == `len(clip_stems.npy)`; fields = {logits, y_true, class_list, run_id, ...}; `class_list` matches `manifest.config.classes`
f) Inference doesn't mutate run dir outside `predictions/`

### Finishing touch surfaces (catalogue)

End-of-refactor sweep targets. Updated as new finds emerge during execution.

- `src/bst_x/data_pipeline_to_model_train.md` (deprecated symbol refs)
- `src/bst_x/pipeline/README.md` (deprecated symbol refs, multiple entries)
- `tests/testing_guide.md` (`_derive_class_label` semantics + `ablation_id` naming)
- `docs/architecture_notes/xai_vid_feature.md` (old `.pt` schema field name)
- `../frontend_integration_handoff.md` (per plan Step G)
- `scratch/scratch_layout.md` (new collation dirs)
- `docs/architecture_notes/bst_x_overview.md` (active baseline pointer)
- `docs/models_registry.yaml` (taxonomy field; leave-via-alias or re-key)
- `docs/architecture_notes/unknown_channel_fix_review.md` (archived design doc, deprecated symbol refs; pure history)
- `docs/architecture_notes/completed_general_refactors/data_access_integration_plan.md` (archived design doc, deprecated symbol refs; pure history)
- `src/bst_x/validation_scripts/README.md` (line 163 references `derive_ablation_id`)
- `scratch/research/dump_videos_len.py` (uses old `derive_npy_collated_dir_basename` signature; TypeError on invocation; ad-hoc inspection script Ariel likely reaches for during X3D-S work — fix when next touched)

## 2026-05-23: multi-agent verification round

Three Plan agents dispatched in parallel: test suite design, integration verification (Steps A/C/D/J), and cross-cutting audit + new-scope-file edit specs. Findings:

### Additional scope surfaced

Three more validation scripts using `taxonomy.standalone_set` and `taxonomy.class_list()`, missed by the pre-flight audit because the plan's "validation_scripts audited and confirmed clean" claim was wrong:

- `src/bst_x/validation_scripts/validate_zeroed_frames.py` lines 178-209
- `src/bst_x/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py:116`
- `src/bst_x/validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py:103-104`

All three get the same forward-only adaptation as `fail_rate_per_class.py`: route through `derive_class_index` returning the string class name (or the index, depending on use), switch `class_list()` method to `taxonomy.classes` attribute.

### Risks + plan corrections

- `MERGE_MAP` import in `clip_generator.py:22` and `verify.py:16` removed by Step A (was only flagged in `build_dataset.py:25`). All three need their import lines updated in the same commit; otherwise import-time crash.
- `bst_x_infer.py:__main__` block (lines 116-177) reads `manifest.extra.arch` and falls back to `TAXONOMIES[DEFAULT_TAXONOMY].class_list()`. Both removed. Block gets the same resolver pattern as Step J's `_resolve_class_list`. Landing alongside Step D9 (--fe argparse) in the same commit.
- Plan doc vs log naming mismatch: plan doc uses `excludes_raw` throughout Step A spec; log locks `excluded_base_stroke_types`. Plan doc gets updated to match the log in the Step A commit.
- `confusion_matrix.py` change is 2-3 lines, not 1: `np.load` reads `class_list` as a numpy object array (needs `.tolist()`) and `y_pred` must be derived via `np.argmax(logits, axis=1)` since the npz drops `y_pred_top1`.
- `npz['class_list']` consumer caveat: numpy object array, not Python list. Both `confusion_matrix.py` and the parked post-hoc FE-JSON converter need `.tolist()` to use it. Documented in the schema spec.
- `adjust_to_partial_train_set` must slice `clip_stems` with the same `choose_i` indices used for labels/pos/shuttle (covered in plan Step C3; agent confirmed).

### Test coverage additions

Three new test files needed (don't exist today):

- `tests/test_inference_smoke.py` — covers inference behaviours a-f (CPU-only, parametrised over taxonomies)
- `tests/test_api_registry.py` — covers `_resolve_class_list` fallback chain (config.classes → extra.arch.active_class_list)
- `tests/test_api_inference.py` — covers predictions-JSON field back-compat (`class_list` → `active_class_list` fallback)

Plus an optional `tests/conftest.py` for shared fixtures: `tiny_bst_network(taxonomy)`, `make_collated_split(tmp_path, n, with_clip_stems)`, `synthetic_manifest(run_dir, taxonomy)`.

Coverage gaps the existing plan didn't cover:
- CLI flag wiring in `bst_x_train.py` (`--taxonomy`, `--split-column`, `--ablation-id`) — needs explicit test, either in `test_hparam_sweep.py` or a new `tests/test_bst_train_cli.py`.
- `_assert_label_coverage` failure modes (full coverage train; subset val/test; rogue val/test classes) — covered by new tests in `tests/test_taxonomy.py`.
- `Task.dump_predictions` row alignment with `clip_stems.npy` — covered in `test_inference_smoke.py` (e).
- `Task.dump_predictions` row determinism (shuffle=False) — optional test in same file.

### Test file rename

Recommended by Agent 1: `tests/test_active_classes.py` → `tests/test_taxonomy.py`. Reasoning: post-refactor, the "active vs full classes" distinction disappears entirely; the file's identity changes from "active class machinery tests" to "taxonomy contract tests." Rename lands in the Step A commit alongside the rewrite. Author approved.

### Net scope after multi-agent

| Phase | Code | Docs | Tests |
|---|---|---|---|
| Original | 22 | 4 | 5 (modify) |
| + pre-flight audit | +4 | +4 | – |
| + multi-agent | +3 | +3 | +3 (create) |
| **Total** | **29** | **11** | **5 modify + 3 create** |

All additions are forward-compat (no shim explosion). Modest creep relative to the size of the refactor.

## 2026-05-23: Step A committed (feat/taxon-pinned-w-preds)

Step A landed. Touched:

- `src/bst_x/pipeline/config.py` — full taxonomy block rewrite. New `Taxonomy(name, classes, merge_map, has_sides, excluded_base_stroke_types)` dataclass + `__post_init__` pinning unknown at -1. Six new TAXONOMY_* objects (bst_25/bst_24/bst_12, une_v1_14/une_v1_15, shuttleset_18). New helpers: `resolve_taxonomy`, `derive_class_index`, `_sided_classes`. New `TAXONOMY_ALIASES` for legacy-name resume. `derive_npy_collated_dir_basename` signature changed: `ablation_id` required, `split_column` + `drop_unknown` params gone. `MERGE_MAP_25` lands the paper-faithful `driven_flight: drive` fix.

- `src/bst_x/pipeline/data_access.py` — `_derive_class_label` wraps `derive_class_index` (returns string or None). `drop_unknown` parameter removed from `get_clip_records` / `summarise` / `interactive` / `_build_cli` (taxonomy carries the rule via `excluded_base_stroke_types`). `DEFAULT_TAXONOMY_NAME = 'bst_25'` for data exploration (Hyp default in bst_x_train will be `une_v1_14` — separate concern, lands in Step D).

- `src/bst_x/pipeline/{build_dataset, clip_generator, verify}.py` — dropped `MERGE_MAP`, `TAXONOMY_UNE_MERGE_V1`, `DEFAULT_TAXONOMY` imports. Each gained a `_DEFAULT_TAXONOMY = resolve_taxonomy('une_v1_14')` constant for function defaults. CLI default in build_dataset shifted to `'une_v1_14'`.

- `tests/test_active_classes.py` → `tests/test_taxonomy.py` (renamed via `git mv`). Body rewritten: ~293 lines deleted (the active/full machinery), ~250 lines added covering the new Taxonomy contract, `derive_class_index` parametrised cases including the driven_flight fix, `resolve_taxonomy` aliases, `_sided_classes` helper. BST_CG_AP forward+backward smoke kept and re-parametrised over the six new taxonomies. class_weights renorm tests kept (head-shape concern, independent of the active-class machinery).

- `tests/test_data_access.py` — added `test_derive_class_label_applies_bst_25_driven_flight_fix` for the headline merge fix, `test_derive_class_label_excluded_returns_none` for the None-return contract. `test_drop_unknown_removes_unknown_rows` rewritten as `test_taxonomy_with_excluded_unknown_drops_unknown_rows` (drives via `bst_25` vs `bst_24` taxonomy choice, no separate flag). `_make_fake_dataset` default switched to `'bst_25'`. Interactive tests updated: dropped the now-removed drop_unknown menu prompt.

- `tests/test_integration.py` — replaced `TAXONOMIES[DEFAULT_TAXONOMY].n_classes` with `resolve_taxonomy('bst_25').n_classes` (largest registered taxonomy; head can handle labels from any post-refactor collation).

- `docs/architecture_notes/collation_taxon_pin_w_preds_refactor.md` — `excludes_raw` → `excluded_base_stroke_types` throughout (20 occurrences) per the locked naming.

### Tests green

`pytest tests/test_taxonomy.py tests/test_data_access.py` — 101 passed, 4 skipped (real-scratch labels.npy probe; /scratch not visible on this host). Full `pytest tests/` (excluding Arch2 tests that need extra ML deps not in the cicd venv) — 257 passed, 5 skipped, 2 pre-existing infra failures in `test_api.py` (Docker `/app/uploads` path not present on this dev host, unrelated to refactor).

Pre-existing dep issues outside Step A scope: `build_dataset` chain wants `cv2`, `clip_generator` wants `moviepy` (full ML environment is on engelbart). Step A code imports clean against `config`, `data_access`, `verify`; the `cv2` / `moviepy` chain isn't touched.

### Pivot for Step B

Re-extract the 1,278 unknown clips on engelbart. `scripts/build_extract_stems.py` gains `--only-unknown`; raw_extract + apply_heuristic run against the sibling dir; rsync to bourbaki. No code commit between B1 and B2 (B2-B4 are pure ops).

## 2026-05-23: Step B1 committed + Step A tidies

`scripts/build_extract_stems.py` gains `--only-unknown`. Mutex with `--keep-unknown` via parser.error before any I/O. Smoke-tested all four flag combinations:

- default (no mode flag): 30,487 stems written
- `--keep-unknown`: 31,765 stems (default + 1,278 unknown)
- `--only-unknown --keep-busted`: 1,278 stems (matches clips_master.csv unknown count)
- `--only-unknown --keep-unknown`: parser.error fires before any I/O

Bundled two tidies on Step A's surface in the same commit:

- `Taxonomy.__post_init__` raises `ValueError` instead of asserting (asserts strip under `python -O`; the unknown-at-minus-one contract needs to fire in prod regardless).
- `derive_class_index`'s filter-before-merge order pinned by a new test (`test_label_for_row_filters_before_merge`): synthetic taxonomy with a raw type in both `excluded_base_stroke_types` and `merge_map`, the filter early-returns, merge never fires.

71 cases pass in test_taxonomy.py (was 67 pre-commit; added the filter-first test + the post_init tests still pass under the new ValueError contract).

### Pre-push agent verification

Three Plan agents reviewed the diff before push. Verdict: Step B1 is correctness-clean and the engelbart workflow is safe to run.

Agent 3 raised a "NO-GO" blocker claiming `apply_heuristic.py:335-337` lazy-imports `normalize_joints` from `prepare_train_on_shuttleset.py` (which still has broken Step-A imports). **Verified false** — no such lazy import exists in apply_heuristic.py or sticky_anchor.py. The only reference to `prepare_train_on_shuttleset` in apply_heuristic is a docstring comment at line 7. Agent 3 hallucinated the import chain.

Other real findings, none blocking:

- Step C/D files (`prepare_train_on_shuttleset.py`, `bst_x_train.py`, `bst_x_common.py`, `bst_x_infer.py`, `model/bst.py`, two validation scripts) all still import deleted symbols. **Intentional per plan, fail-loud on import.** Not in Step B's call graph.
- Live resume case `une_merge_v1_nosides -> une_v1_14` verified bit-for-bit safe (manifest's recorded `active_class_list` matches `STROKE_TYPES_14_UNE_V1` exactly).
- Lossy alias cases (`merged_25 -> bst_25`, `une_merge_v1 -> une_v1_15`, `raw_35 -> bst_25`) acknowledged in plan; user declined adding `warnings.warn` (overkill for scenarios that aren't happening).
- `_make_fake_dataset` in test_data_access would TypeError on `Path / None` if a future test passes `bst_24` + unknown row. Loud failure mode preserved by design (not patched).

### Next: B2-B4 on engelbart

No code commits between B1 and Step C. Ops only:

- B2: `raw_extract.py` against the new stems list -> `/scratch/comp320a/ShuttleSet_keypoints_raw_unknown/`. ~1.5 hours.
- B3: `apply_heuristic.py` against that raw dir -> `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown/`.
- B4: rsync both sibling dirs to bourbaki.

## 2026-05-23: B2 ran on bourbaki, B3 surfaced the real lazy-import blocker

User ran B2 on bourbaki (not engelbart — my earlier "engelbart" pointer was wrong; per project memory, bourbaki is the actual extract host). Clean: 1278/1278 processed, 60-minute wall time, four over-detection warnings (audience-behind-players, normal).

B3 then crashed:

```
File ".../heuristics/sticky_anchor.py", line 335, in apply
    from preparing_data.prepare_train_on_shuttleset import (
File ".../preparing_data/prepare_train_on_shuttleset.py", line 34, in <module>
    from pipeline.config import (
```

The lazy import I previously verified as "absent in apply_heuristic.py" was actually inside `sticky_anchor.py:332-337` — same structural concern Agent 3 flagged before B1 push, just at a different file/line than the agent claimed. I anchored my "Verified false" on the wrong file; Agent 3's structural call was correct. Correction: the Step C work (rewriting prepare_train_on_shuttleset.py properly) is the right unblock, not a hot-fix.

## 2026-05-23: Step C committed

`collate_npy` routes per-row through `derive_class_index`: `excluded_base_stroke_types` drops what `drop_unknown` used to (gone from CLI + signature); `merge_map` + side rule fire next. New `unknown_root_dir` param routes `raw_type=='unknown'` rows through a sibling per-clip dir, for `bst_25` / `une_v1_15` cells against the `_unknown` sibling extract.

`clip_stems.npy` sidecar saves alongside `labels.npy`, row-aligned. `Dataset_npy_collated` loads it with graceful None fallback for legacy collations (warnings.warn); `adjust_to_partial_train_set` mirrors the per-class slicing.

CLI: `--drop-unknown` gone, `--ablation-id` required, `--unknown-clip-npy-dir` added; `resolve_taxonomy` handles legacy aliases at the writer entry (writer-side `choices=list(TAXONOMIES.keys())` restricts to canonical names anyway, so legacy aliases fail with `invalid choice` before reaching resolve_taxonomy).

### Symmetric unknown-dir validation (post-agent hardening)

Three Plan agents reviewed Step C before commit. All three agreed the diff is correctness-clean and B3 is unblocked. Agent 3 surfaced one important design subtlety: `bst_25` / `une_v1_15` (keepunk taxonomies) without `--unknown-clip-npy-dir` silently drops the 1,278 unknown rows via the missing-file branch (derive_class_index keeps them alive, but `_pos.npy` doesn't exist under the canonical extract). Plan acknowledges that bst_25 / une_v1_15 cells must supply `--unknown-clip-npy-dir`, but didn't enforce it at the boundary.

Fixed both ways:
- `collate_npy` body: `if taxonomy.has_unknown and unknown_root_dir is None: raise ValueError(...)` (with a clear message naming the sibling extract + the bst_24 / une_v1_14 escape hatch).
- `main()` argparse: same check via `parser.error(...)` at the CLI surface.

Symmetric with the existing `unknown_root_dir set + taxonomy excludes unknown` validation. Both directions error loud now.

### Other Step C hardenings (post-agent)

- `derive_class_index`: bare `tuple.index` ValueError replaced with a descriptive raise naming `taxonomy.name`, `raw_type`, `side`, derived `label_str`, and the full classes list. `from e` preserves the original traceback chain. New test (`test_label_for_row_raises_descriptive_error_on_missing_class`) pins the contract.
- `collate_npy` per-row loop wraps `derive_class_index` with a real `except ValueError` that adds clip stem context and re-raises via `from e`. Not a silent swallow.
- WARNING message for missing per-clip files: the "or {unknown_root_dir} for unknown rows" hint now only prints when `unknown_root_dir is not None` (was printing "or None for unknown rows" otherwise — cosmetic confusion).
- `_make_synthetic_split` test helper extended to accept a `videos_len` array. New test (`test_dataset_clip_stems_after_zero_length_filter`) pins the parallel slicing of `clip_stems` alongside the other arrays when the zero-length filter fires.

107 pytest cases green (was 103 before the hardenings + 2 new tests).

### Step D resume case still gated

`run_20260505_154907` resume needs Step D7's `bst_x_train.py:803` patch (`hyp.taxonomy` recorded string instead of `taxonomy.name` resolved) so the weight filename lookup hits the legacy on-disk filename. Step C alone gets the Dataset side right (graceful None for clip_stems, labels load as-is); Step D7 closes the loop.

### Next

B3 retry on bourbaki against the new `_unknown` raw extract:

```
PYTHONPATH=src/bst_x:src/bst_x/stroke_classification \
    python -m preparing_data.apply_heuristic \
    --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw_unknown \
    --output-dir /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown \
    --heuristic sticky_anchor \
    --clips-csv notebooks/clips_master.csv
```

Then B4 (cross-node sync if needed for Step C training cells), then Step D when ready.

## 2026-05-23: B3 + B4 done, env-var defaults landed, Step C sanity collation surfaced cells 2 + 8 are dead

### B3 + B4

B3 (`apply_heuristic`) ran clean on bourbaki against the new sibling extract: 1278/1278 processed, 30s wall, 0 skips. Output at `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown/`. B4 (rsync to engelbart) skipped for now -- decision deferred until Step D / Step E runner picks a training host.

### Env-var defaults commit (`d438c74`)

Follow-up to Step C: `prepare_train_on_shuttleset.py` was still defaulting paths to project-root constants (`CLIPS_OUTPUT_DIR`, etc.) instead of reading the BST_* env vars from `.env`. Patched to call `load_repo_dotenv()` at the top of main() and route argparse defaults through `env_path` / `env_path_or_none` helpers (promoted from underscore-private in `data_access.py`). New `BST_X_COLLATED_DATA_ROOT` drives the collation output root; falls back to in-repo `preparing_data/` for local dev. `scratch_layout.md` documents the new vars. Also added `test_dataset_clip_stems_after_zero_length_filter` -- the test gap from Step C closed in the same commit.

Required ad-hoc step on bourbaki: user added `BST_X_COLLATED_DATA_ROOT=/scratch/comp320a/` and `BST_SHUTTLE_CSV_DIR=/scratch/comp320a/ShuttleSet/shuttle_csv` to their `.env`. The repo `.env.example` doesn't carry these new vars yet (see `feedback_env_example_stale` memory).

### Step C sanity collation -- bst_25 + split_v2

Ran the new collator on bourbaki against bst_25 + split_v2. Numbers came back at train=22,743 / val=5,250 / test=4,210 = **32,203**, which is exactly the non-unknown count from clips_master.csv. Zero `WARNING: N clips had no flat per-clip files` — the 1,278 unknown rows weren't dropped at the file-existence check, they were filtered out at the CSV split filter (`clips_df[clips_df['split_v2'] == set_name]`) BEFORE derive_class_index.

Root cause: `shuttleset_splits_v2.csv` is 14-class only (32,203 rows) by design. `clips_master.csv`'s `split_v2` column has NaN for all 1,278 unknown rows; `scripts/build_shots_master.py:178-184` asserts `n_no_v2 == n_unknown` as a hard-fail invariant. `split_bst_baseline` IS unknown-complete (875 train / 241 val / 162 test) because it's built from a vid→split mapping that inherits per-match regardless of class.

Implications for the cells:
- Pairing a `has_unknown=True` taxonomy with `split_v2` produces a collation with an empty 25th (or 15th) class -- numerically identical to the bst_24 / une_v1_14 variant with a dead output neuron. Cells 2 (bst_25 + split_v2) and 8 (une_v1_15 + split_v2) are functionally redundant.
- Cell 5 (bst_25 + split_bst_baseline) remains the meaningful keepunk evaluation; it carries the 1,278 unknowns naturally.

### Decisions taken 2026-05-23

- Drop cells 2 and 8 from the plan. Cell numbering preserved (no 3→2 renumbering).
- Cell 3 (bst_24 + split_v2) KEPT -- it's the valid project-split bst_24 baseline.
- Plan doc cells table updated to reflect the 6-cell roster (cells 1, 3, 4, 5, 6, 7); 45 serials at 25 min/serial ≈ 19 hours wall.
- The bst_25 + split_v2 collation produced during sanity check is to be deleted (user has the rm command); pending action when user is back at the bourbaki terminal.
- Future Claude sessions: see new memory `project_split_v2_unknown_design.md` so this isn't re-discovered next session.

### Status going into next session

- Step A ✓ committed (`1d98949`)
- Step B1 ✓ committed (`6a6fff5`)
- Step B2 + B3 ✓ ops on bourbaki (sibling extract at `/scratch/comp320a/ShuttleSet_keypoints_*_unknown/`)
- Step B4 — pending, optional rsync to engelbart
- Step C ✓ committed (`977a485`)
- Env-var defaults ✓ committed (`d438c74`)
- Step C sanity collation done, found split_v2 gap, cells 2 + 8 dropped
- bst_25+split_v2 collation pending deletion (`rm -rf /scratch/comp320a/ShuttleSet_data_bst_25/npy_taxon_pinned_w_preds`)
- Step D — not yet started (train + infer surface rewrite)
- Step E — not yet started (collation_runner + session config)
- Step F — partial (test_taxonomy + test_dataset done; inference smoke + api tests pending Step D / J)
- Step G — pending (finishing-touch doc sweep)
- Step J — not yet started (FE handler reconciliation)

### Next session pick-up

1. Delete the bst_25+split_v2 collation per the rm command above.
2. Update Step E's runner config (when it lands) to use the 6-cell roster, not 8.
3. Either dive into Step D (train surface rewrite) or do the 6 collations under the current contract first.
4. Plan doc reflects the 6-cell decision; memory captures the split_v2 design fact; refactor log carries today's chronology. Should be readable cold.

## 2026-05-30: split folded into the collated dir basename (collision fix + collation prep)

Picking up to run the 6 collations. Stashed the unrelated x3d/xai working-tree changes first (`git stash` "x3d_xai") for a clean tree.

Reading the writer path logic surfaced a real collision the plan missed. The collated dir is `ShuttleSet_data_<tax>/npy_<ablation_id>/` and `derive_npy_collated_dir_basename` built the basename from `use_3d_pose`/`seq_len`/`ablation_id` only, no split. So cell 3 (bst_24 + split_v2) and cell 6 (bst_24 + split_bst_baseline) both resolved to `ShuttleSet_data_bst_24/npy_taxon_pinned_w_preds/` and would have clobbered each other. (The plan's on-disk artefact list at .md:1223 still showed one bst_24 entry plus the dropped une_v1_15 cell, so it predated both the cells-2/8 drop and this collision.)

Fix, chosen over hand-suffixing the ablation_id per cell: fold the split into the basename automatically.
- `derive_npy_collated_dir_basename` gains a required `split_column` param; basename is now `npy_[3d_][seq{N}_]{split}_{ablation_id}` with `{split}` = `split_column.removeprefix('split_')` (`split_v2` -> `v2`, `split_bst_baseline` -> `bst_baseline`). config.py:465.
- Writer caller passes `split_column=args.split_column` (prepare_train_on_shuttleset.py:1165). The `--split-column` CLI arg already existed; nothing new at the CLI.
- All 6 cells now share `--ablation-id taxon_pinned_w_preds`; the split arg disambiguates. No manual per-cell tag juggling.

Resulting dirs (collision gone):
- shuttleset_18 / v2        -> ShuttleSet_data_shuttleset_18/npy_v2_taxon_pinned_w_preds
- bst_24 / v2               -> ShuttleSet_data_bst_24/npy_v2_taxon_pinned_w_preds
- bst_12 / v2               -> ShuttleSet_data_bst_12/npy_v2_taxon_pinned_w_preds
- une_v1_14 / v2            -> ShuttleSet_data_une_v1_14/npy_v2_taxon_pinned_w_preds
- bst_25 / bst_baseline     -> ShuttleSet_data_bst_25/npy_bst_baseline_taxon_pinned_w_preds
- bst_24 / bst_baseline     -> ShuttleSet_data_bst_24/npy_bst_baseline_taxon_pinned_w_preds

Tests: added `test_derive_npy_collated_dir_basename_folds_split` (both splits + a 3d + a seq30 case) and `test_bst_24_split_variants_get_distinct_basenames` (pins the collision fix) in test_taxonomy.py. Refreshed the CANDIDATE_REAL_DIRS probe paths to the split-encoded names, else they'd silently skip once the collations land. Green in the cicd venv: test_taxonomy 77 passed / 4 skipped, test_data_access + test_dataset 35 passed.

Step D reader implication (flagged, not fixed now): bst_x_train.py:1073 already calls `derive_npy_collated_dir_basename(split_column=hyp.split_column, ...)` -- the pre-Step-A call shape it was never moved off -- so re-adding the param re-aligns reader with writer for the new cells. But bst_x_train is still broken pending Step D (imports the removed `derive_ablation_id` at line 39), and legacy dirs (`npy_wipe_drop`, `npy_<tax>_<split>_dropunk`) carry no split tag, so the Step D reader rewrite needs either a fallback for pre-split names or a re-collation of the legacy runs it wants to resume. Nothing that currently works regresses: the run_20260505_154907 resume is already dead pending Step D, and cell 7 re-collates + retrains that best on the new generation anyway.

bst_25 sanity collation: already deleted last week, so cell 5 has a clean target dir.

## 2026-05-30: ablation_id -> collation_id disentanglement (collator renamed now, manifest split deferred to Step D)

Follow-on to the split fold. The path tag was misnamed: it discriminates collation generations, not ablation studies. Split into two orthogonal fields.

- **collation_id**: collation generation (`taxon_pinned_w_preds`, `wipe_drop`). Path + manifest. Discriminates re-collations of the same taxonomy + split. The *value* is unchanged, only the field name, so the six collation dirs are unaffected.
- **ablation_id**: a new, dedicated training-time tag (augs / loss / wiring on a fixed collation). Manifest-only, never in the path, nullable. Born on the train side; the collator has no ablation concept. Fixes the historical pain where aug runs reused the collation's tag with no breadcrumb of their own (best_model notes literally read "ablation_id=wipe_drop <- aug runs on top of wipe_drop collated").

Renamed now (the live collator + path helper from the previous commit):
- `config.py` `derive_npy_collated_dir_basename`: param `ablation_id` -> `collation_id`. Output string unchanged (`npy_{split}_{collation_id}`).
- `prepare_train_on_shuttleset.py`: `--ablation-id` -> `--collation-id`, the derive call, the dry-run print.
- `test_taxonomy.py`: contract tests + comments.

Green: test_taxonomy 77 passed / 4 skipped (cicd venv). The collation command is now `--collation-id taxon_pinned_w_preds`.

Deferred to Step D / Step J (those exact lines get rewritten there anyway, and bst_x_train is still broken pending D):
- bst_x_train Hyp gains `collation_id` (path tag) + a nullable training `ablation_id`; defaults `collation_id='taxon_pinned_w_preds'`, `ablation_id=None`.
- Manifest carries both; the old `effective_ablation_id` writer key (bst_x_common) becomes `collation_id` (auto-derive is gone, so raw == effective, no separate "effective" field needed).
- FE registry reads both new fields directly, NO legacy fallback (historical runs aren't populated to the FE).
- The legacy fallback (resolve `collation_id` from an old manifest's `effective_ablation_id`/`ablation_id`, gated so the old `ablation_id` isn't misread as a training tag) lives in a shared `config.py` reader for internal scripts that pull legacy run data.

Plan updated: Locked decisions (the four points), Step A snippet (now matches the implemented signature, incl. dropping the phantom `taxonomy_name` param), Step D1 Hyp, new Step J5, Step E runner config (rebuilt to the 6-cell roster + collation_id; the example had drifted, still listing dropped cells 2 + 8), the artefacts note, open items. `derive_ablation_id` stays referenced as the removed function (not renamed).

## 2026-05-30: verification pass + doc tidy + collation_id_from_manifest helper

Ran two read-only opus verification agents over the disentanglement (one on the live collation pipeline, one on the blast radius + tests + docs). Both confirmed the rename is correctly propagated through everything that runs: the collator + helper are on the new contract, the test suite passes, and crucially no NEW breakage. bst_x_train.py / verify_bst_train_target.py fail only at *import* on the removed `derive_ablation_id` name (a Step A removal, prior session), independent of this signature change, so they're pre-existing Step-D-deferred breakage, not a regression.

Acted on the two follow-ups they surfaced:
- Doc tidy (path-format strings -> `npy_[3d_][seq{N}_]{split}_{collation_id}`): `data_pipeline_to_model_train.md`, `bst_x_infer.py` (example comment), `tests/test_integration.py`, `tests/testing_guide.md`. Left for Step D where they're naturally touched: `run_tracker.md`'s `effective_ablation_id` (manifest writer rename), the data_pipeline Hyp-default table entry, and the `derive_ablation_id` prose in `shared/taxonomy.py` (BRIC-side) + `validation_scripts/README.md`.
- Internal-script reader: `collation_id_from_manifest` now in `config.py`. Resolves the collation tag across schemas -- `config.collation_id` (new) -> `config.ablation_id` (legacy explicit) -> `extra.data_provenance.effective_ablation_id` (legacy auto-derived). Reads new-schema collation_id first, so a new manifest's training `ablation_id` is never misread as the collation; the docstring spells out the meaning-flip caveat for callers that also want the training tag. Four unit tests added.

Green: test_taxonomy 81 passed / 4 skipped (cicd venv).

## 2026-05-30: six collations built + verified on bourbaki; Steps A-C done, Step D next

Ran the six collations on bourbaki (in `venv-mmpose`; collation only needs numpy/pandas so it works there, though `venv-bst` is the canonical home). All correct:

- Four split_v2 cells (shuttleset_18, bst_24, bst_12, une_v1_14): 22743 / 5250 / 4210 = **32,203** each. Identical because they share the non-unknown clip set + the split_v2 partition; only the label space differs per taxonomy.
- bst_24 + split_bst_baseline: 24866 / 4000 / 3337 = 32,203.
- bst_25 + split_bst_baseline (keepunk, via `--unknown-clip-npy-dir`): 25741 / 4241 / 3499 = **33,481**. The +1,278 over the dropunk baseline cell is exactly the unknowns, split 875 / 241 / 162 = the documented distribution. Unknown routing confirmed to the clip.

Six dirs, no collision (the two bst_24 cells land distinct):
`ShuttleSet_data_{shuttleset_18,bst_24,bst_12,une_v1_14}/npy_v2_taxon_pinned_w_preds` and `ShuttleSet_data_{bst_25,bst_24}/npy_bst_baseline_taxon_pinned_w_preds`.

Verified deeper than counts: `tests/test_taxonomy.py` in `venv-bst` came back **85 passed / 0 skipped**. On the laptop it was 81 passed / 4 skipped; the 4 real-labels probes un-skip on bourbaki (`/scratch` visible + dirs exist) and passed, asserting labels in `[0, n_classes)` and `clip_stems` row-aligned with `labels` for bst_24 / bst_25 / une_v1_14 + the legacy une_merge_v1_nosides.

Aside on a venv scare: the suite fails to even *collect* in `venv-mmpose` (it imports `model.bst` -> `positional_encodings`, a training dep that venv lacks). That's the wrong venv, not a broken wheel; `venv-bst` is the one for tests/training. Captured in the Venv Paths memory.

**Status:** Steps A-C done + committed (`4c1c3c9`, pushed). Collations done + verified. Step D (the `bst_x_train` rewrite) is next. Cold-start brief at `scratch/NEXT_SESSION.md`.

## 2026-05-30: Steps D, J, E, G code landed (uncommitted) — train + infer surface rewritten

Did the whole model-side pass in one session on the laptop, smoke-tested in the `badminton-cicd` venv (which turns out to carry the full model stack incl. `positional_encodings`, so the earlier venv-mmpose collection scare doesn't apply here). Nothing committed yet.

**Step D (bst_x_train + bst_x_common + bst_x_infer):**
- Hyp tuple: `drop_unknown` and `expected_active_classes` gone; `ablation_id` renamed to `collation_id` (the path tag) and a fresh nullable `ablation_id` added as the training-time tag. Defaults `taxonomy='une_v1_14'`, `collation_id='taxon_pinned_w_preds'`, `ablation_id=None`.
- The runtime active-class adapter (`derive_active_classes_from_labels`) is deleted. `Task._assert_label_coverage` replaces it: train must cover every taxonomy class, val/test must carry nothing train didn't see. Head dim and class list read straight off `taxonomy.n_classes` / `taxonomy.classes`; no more `n_active_classes` / `active_class_list`.
- `_validate_and_record_arch` + the whole `extra.arch` manifest block are gone. A 2-line `_print_taxonomy_block` logs the taxonomy at run start; the class list now lands in `config.classes` (mirrors BRIC), written at `track_run` time.
- `dump_topk_predictions` (in bst_x_common) + `Task.dump_predictions` write per-stroke logits + top-k + ground truth to `predictions/<split>_serial_<n>.npz` for every serial, all three splits, shuffle=False so the npz rows follow the in-memory dataset order (the follow-up adds a `clip_stems` column for a self-contained row->stem join). This is the FE / calibration payload the refactor was for.
- Per-class val F1 snapshotted at the best-macro epoch and surfaced to the serial manifest (`extra.val_at_best_macro_epoch`); `train_network` now returns `(model, val_at_best)`, threaded through `seek_network_weights`. TB gets `F1_val/<class>` scalars.
- Weight-file + collated-dir naming uses the resolved `taxonomy.name` (matching what the collator writes), not `hyp.taxonomy`. The follow-up dropped the legacy-resume scaffolding (recorded-`npy_collated_dir` read + manifest `.bak`): there's no train-onto-legacy-weights path, so nothing re-derives an old dir name.
- Caught a real reader gap the plan under-specified: bst_x_train hardcoded the in-repo `preparing_data/` path while the collator writes to `BST_X_COLLATED_DATA_ROOT` (=/scratch/comp320a on bourbaki). bst_x_train now reads the same env var (load_repo_dotenv + env_path_or_none), else falls back in-repo. Without this the runner would never find the cells. `verify_bst_train_target.py` mirrors the same resolution.
- Two bugs the smoke caught: bst_x_train never imported numpy (the new asserts/dump use `np`); and the rogue-label naming `IndexError`d on an out-of-range label (now OOB-safe, matching the old bst_x_common helper).

**Step D9/D10:** `bst_x_infer --fe` is the post-hoc batch dump (same npz schema), folding in `eval_dump_predictions.py` which is retired (pointer note left in presentation_prep). `confusion_matrix.py` reads the npz (`y_pred_top1`, `class_list`) instead of the old `.pt`.

**Step J (FE handlers):** `registry._resolve_class_list` reads `config.classes` first, falls back to the legacy `extra.arch.active_class_list` (also lights up BRIC, whose list was never in BST's bandaid block). Predictions-JSON readers accept `class_list` or legacy `active_class_list` (registry x2 + inference.py). `collation_id` / `ablation_id` now read off the manifest config. The shipped mock entry still resolves its 14-class list via the fallback. J4 (the FE-team PR note) is still owed in my own voice.

**Step E:** `collation_runner.py` (no kill/verdict logic, state.json resume) + `scratch/runners/taxon_pinned_w_preds/config.yaml` with the 6-cell roster (5/5/5/10/10/10 = 45 serials). Smoke-stubbed the subprocess: all 6 cells launch with the right `--taxonomy/--split-column/--collation-id` flags.

**Step G (partial):** path-format / Hyp-default stragglers updated (`run_tracker.md`, `validation_scripts/README.md`, `data_pipeline_to_model_train.md` Hyp row). `data_pipeline_to_model_train.md` still has broader Step-A staleness (`class_list()`, `DEFAULT_TAXONOMY`, old taxonomy names) left untouched — separate doc-debt, not scoped here.

**Tests:** +19 across `test_train_surface.py` (coverage assert, npz dump, train_network return), `test_inference_smoke.py` (bst_x_infer --fe end-to-end, k-clamp), `test_api_registry.py` (`_resolve_class_list` + a live `/api/registry` fallback check), `test_api_inference.py` (JSON field preference). Full suite in the cicd venv: **365 passed, 5 skipped** (the /scratch real-label probes + the missing-clip-stems back-compat), 2 pre-existing `/app/uploads` docker failures in `test_api.py` unrelated to this work.

**Not done (needs bourbaki / a later session):** the actual 6-cell runs via the runner, the non-best npz prune, the run-ID-dependent docs (bst_x_overview headline numbers, models_registry new entries), and the J4 FE PR note. Everything code-side is import-clean and tested on CPU.

Also fixed in passing: `model/bst.py`'s `__main__` smoke (was importing the removed `DEFAULT_TAXONOMY`) and a stale `class_list()` docstring in shuttleset_dataset. One BST-side straggler the Step-A audit missed is **flagged not fixed**: `validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py` still calls the removed `taxonomy.standalone_set` + `taxonomy.class_list()`. It's a standalone diagnostic (not imported, not in CI), so it doesn't break anything live, but it needs a real port to `derive_class_index` (not a rename) before it'll run again.
