# Collation taxon_pinned_w_preds refactor log

Running log for the taxon_pinned_w_preds refactor. Companion docs: `scratch/collation_taxon_pin_w_preds_refactor.md` (full plan, locked decisions, touch surface inventory), `scratch/taxon_pin_w_preds_tldr.md` (FE-facing summary).

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
- `src/bst_refactor/pipeline/config.py` full taxonomy block (Step A)
- `src/bst_refactor/pipeline/data_access.py` imports + `_derive_class_label` + four `class_list()` callsites (Step A)
- `src/bst_refactor/pipeline/build_dataset.py` import + defaults (Step A)
- `src/bst_refactor/pipeline/clip_generator.py` import + default (Step A)
- `src/bst_refactor/pipeline/verify.py` import + default (Step A)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py` full Hyp + Task + train_network + manifest surface (Step D)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_common.py` derive_active_classes_from_labels def (Step D)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_infer.py` argparse + DEFAULT_TAXONOMY refs (Step D)
- `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py` full collator surface (Step C)
- `tests/test_active_classes.py`, `tests/test_data_access.py`, `tests/test_integration.py` (Step F)
- `scratch/presentation_prep/eval_dump_predictions.py` delete (Step D10)
- `scratch/presentation_prep/confusion_matrix.py` adapt to npz (Step D10)

### Unexpected hits — scope additions

Four code files and four doc files not in the original hit list.

**Code:**

1. `src/bst_refactor/stroke_classification/model/bst.py` lines 436-437 — `__main__` demo block imports `TAXONOMIES, DEFAULT_TAXONOMY` and reads `n_classes` for the smoke instantiation. Trivial: replace with `resolve_taxonomy('bst_25').n_classes` or similar concrete pick. ~2 lines.

2. `src/bst_refactor/validation_scripts/fail_rate_per_class.py` — live validation script with `--drop-unknown` CLI flag piped through to `get_clip_records`. Needs adaptation: drop the CLI flag, switch to `--taxonomy` selection (taxonomy now carries the unknown-exclude rule via `excluded_base_stroke_types`). ~10 lines.

3. `src/bst_refactor/validation_scripts/verify_bst_train_target.py` — live pre-flight script importing `derive_ablation_id` + `derive_npy_collated_dir_basename`. Both signatures change. ~5 lines.

4. `scratch/api_mocks/build_mock_artifacts.py` lines 29, 204 — writes mock JSONs with `active_class_list` field. The Step J fallback handles reads, so this is low-priority. Either update for consistency or leave (fallback covers).

**Docs:**

5. `src/bst_refactor/data_pipeline_to_model_train.md` lines 94, 398, 544 — references `DEFAULT_TAXONOMY`, deprecated taxonomy methods, `drop_unknown`.
6. `src/bst_refactor/pipeline/README.md` lines 248, 253, 254, 288, 322, 323, 376 — same deprecated-symbol refs.
7. `tests/testing_guide.md` lines 21, 51 — references `_derive_class_label` semantics and the `ablation_id` naming convention.
8. `scratch/architecture_notes/xai_vid_feature.md:132` — references the old `.pt` predictions schema's `active_class_list` field.

### Hits ignored (out of scope or no impact)

- `src/api/bst_inference.py:59` — single comment; the plan parks this file's hardcoded 14-class list as a forward FE concern.
- `src/bric/*`, `src/shared/*` — Arch2 territory, off-limits per agreed scope. Arch2 has its own Taxonomy class with its own `standalone_set` / `unknown_first` / `class_list()`; no symbol overlap with BST.
- `tests/test_network.py`, `tests/test_video_io.py`, `tests/test_player_mapping.py`, `tests/test_temporal.py` — Arch2 tests.
- `tests/test_adaptive_focal.py:49` — single comment, no code dependency.
- `scripts/archive/*` (verify_v1_collate.py, verify_flatten.py) — archived scripts, won't run.
- `src/bst_refactor/stroke_classification/.gitignore` — comment lines from project history.
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

- `src/bst_refactor/data_pipeline_to_model_train.md` (deprecated symbol refs)
- `src/bst_refactor/pipeline/README.md` (deprecated symbol refs, multiple entries)
- `tests/testing_guide.md` (`_derive_class_label` semantics + `ablation_id` naming)
- `scratch/architecture_notes/xai_vid_feature.md` (old `.pt` schema field name)
- `scratch/frontend_integration_handoff.md` (per plan Step G)
- `scratch/scratch_layout.md` (new collation dirs)
- `scratch/architecture_notes/arch_1_directions.md` (active baseline pointer)
- `docs/models_registry.yaml` (taxonomy field; leave-via-alias or re-key)
- `scratch/architecture_notes/unknown_channel_fix_review.md` (archived design doc, deprecated symbol refs; pure history)
- `scratch/architecture_notes/completed_general_refactors/data_access_integration_plan.md` (archived design doc, deprecated symbol refs; pure history)
- `src/bst_refactor/validation_scripts/README.md` (line 163 references `derive_ablation_id`)

## 2026-05-23: multi-agent verification round

Three Plan agents dispatched in parallel: test suite design, integration verification (Steps A/C/D/J), and cross-cutting audit + new-scope-file edit specs. Findings:

### Additional scope surfaced

Three more validation scripts using `taxonomy.standalone_set` and `taxonomy.class_list()`, missed by the pre-flight audit because the plan's "validation_scripts audited and confirmed clean" claim was wrong:

- `src/bst_refactor/validation_scripts/validate_zeroed_frames.py` lines 178-209
- `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py:116`
- `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/zeroed_frames_class_audit.py:103-104`

All three get the same forward-only adaptation as `fail_rate_per_class.py`: route through `label_for_row` returning the string class name (or the index, depending on use), switch `class_list()` method to `taxonomy.classes` attribute.

### Risks + plan corrections

- `MERGE_MAP` import in `clip_generator.py:22` and `verify.py:16` removed by Step A (was only flagged in `build_dataset.py:25`). All three need their import lines updated in the same commit; otherwise import-time crash.
- `bst_infer.py:__main__` block (lines 116-177) reads `manifest.extra.arch` and falls back to `TAXONOMIES[DEFAULT_TAXONOMY].class_list()`. Both removed. Block gets the same resolver pattern as Step J's `_resolve_class_list`. Landing alongside Step D9 (--fe argparse) in the same commit.
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
- CLI flag wiring in `bst_train.py` (`--taxonomy`, `--split-column`, `--ablation-id`) — needs explicit test, either in `test_hparam_sweep.py` or a new `tests/test_bst_train_cli.py`.
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

- `src/bst_refactor/pipeline/config.py` — full taxonomy block rewrite. New `Taxonomy(name, classes, merge_map, has_sides, excluded_base_stroke_types)` dataclass + `__post_init__` pinning unknown at -1. Six new TAXONOMY_* objects (bst_25/bst_24/bst_12, une_v1_14/une_v1_15, shuttleset_18). New helpers: `resolve_taxonomy`, `label_for_row`, `_sided_classes`. New `TAXONOMY_ALIASES` for legacy-name resume. `derive_npy_collated_dir_basename` signature changed: `ablation_id` required, `split_column` + `drop_unknown` params gone. `MERGE_MAP_25` lands the paper-faithful `driven_flight: drive` fix.

- `src/bst_refactor/pipeline/data_access.py` — `_derive_class_label` wraps `label_for_row` (returns string or None). `drop_unknown` parameter removed from `get_clip_records` / `summarise` / `interactive` / `_build_cli` (taxonomy carries the rule via `excluded_base_stroke_types`). `DEFAULT_TAXONOMY_NAME = 'bst_25'` for data exploration (Hyp default in bst_train will be `une_v1_14` — separate concern, lands in Step D).

- `src/bst_refactor/pipeline/{build_dataset, clip_generator, verify}.py` — dropped `MERGE_MAP`, `TAXONOMY_UNE_MERGE_V1`, `DEFAULT_TAXONOMY` imports. Each gained a `_DEFAULT_TAXONOMY = resolve_taxonomy('une_v1_14')` constant for function defaults. CLI default in build_dataset shifted to `'une_v1_14'`.

- `tests/test_active_classes.py` → `tests/test_taxonomy.py` (renamed via `git mv`). Body rewritten: ~293 lines deleted (the active/full machinery), ~250 lines added covering the new Taxonomy contract, `label_for_row` parametrised cases including the driven_flight fix, `resolve_taxonomy` aliases, `_sided_classes` helper. BST_CG_AP forward+backward smoke kept and re-parametrised over the six new taxonomies. class_weights renorm tests kept (head-shape concern, independent of the active-class machinery).

- `tests/test_data_access.py` — added `test_derive_class_label_applies_bst_25_driven_flight_fix` for the headline merge fix, `test_derive_class_label_excluded_returns_none` for the None-return contract. `test_drop_unknown_removes_unknown_rows` rewritten as `test_taxonomy_with_excluded_unknown_drops_unknown_rows` (drives via `bst_25` vs `bst_24` taxonomy choice, no separate flag). `_make_fake_dataset` default switched to `'bst_25'`. Interactive tests updated: dropped the now-removed drop_unknown menu prompt.

- `tests/test_integration.py` — replaced `TAXONOMIES[DEFAULT_TAXONOMY].n_classes` with `resolve_taxonomy('bst_25').n_classes` (largest registered taxonomy; head can handle labels from any post-refactor collation).

- `scratch/collation_taxon_pin_w_preds_refactor.md` — `excludes_raw` → `excluded_base_stroke_types` throughout (20 occurrences) per the locked naming.

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
- `label_for_row`'s filter-before-merge order pinned by a new test (`test_label_for_row_filters_before_merge`): synthetic taxonomy with a raw type in both `excluded_base_stroke_types` and `merge_map`, the filter early-returns, merge never fires.

71 cases pass in test_taxonomy.py (was 67 pre-commit; added the filter-first test + the post_init tests still pass under the new ValueError contract).

### Pre-push agent verification

Three Plan agents reviewed the diff before push. Verdict: Step B1 is correctness-clean and the engelbart workflow is safe to run.

Agent 3 raised a "NO-GO" blocker claiming `apply_heuristic.py:335-337` lazy-imports `normalize_joints` from `prepare_train_on_shuttleset.py` (which still has broken Step-A imports). **Verified false** — no such lazy import exists in apply_heuristic.py or sticky_anchor.py. The only reference to `prepare_train_on_shuttleset` in apply_heuristic is a docstring comment at line 7. Agent 3 hallucinated the import chain.

Other real findings, none blocking:

- Step C/D files (`prepare_train_on_shuttleset.py`, `bst_train.py`, `bst_common.py`, `bst_infer.py`, `model/bst.py`, two validation scripts) all still import deleted symbols. **Intentional per plan, fail-loud on import.** Not in Step B's call graph.
- Live resume case `une_merge_v1_nosides -> une_v1_14` verified bit-for-bit safe (manifest's recorded `active_class_list` matches `STROKE_TYPES_14_UNE_V1` exactly).
- Lossy alias cases (`merged_25 -> bst_25`, `une_merge_v1 -> une_v1_15`, `raw_35 -> bst_25`) acknowledged in plan; user declined adding `warnings.warn` (overkill for scenarios that aren't happening).
- `_make_fake_dataset` in test_data_access would TypeError on `Path / None` if a future test passes `bst_24` + unknown row. Loud failure mode preserved by design (not patched).

### Next: B2-B4 on engelbart

No code commits between B1 and Step C. Ops only:

- B2: `raw_extract.py` against the new stems list -> `/scratch/comp320a/ShuttleSet_keypoints_raw_unknown/`. ~1.5 hours.
- B3: `apply_heuristic.py` against that raw dir -> `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown/`.
- B4: rsync both sibling dirs to bourbaki.
