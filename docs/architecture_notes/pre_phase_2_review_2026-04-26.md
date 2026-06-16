# Pre-Phase-2 Code Review Synthesis

**Date:** 2026-04-26
**Reviewers:** three isolated subagents, no access to local user-memory. Each saw only the repo plus the checked-in `.claude/*.md` orientation files. Findings below are synthesised from their three independent reports.

## Central question

Is the codebase ready to absorb the X3D-S wrist-crop layer for Architecture 1, or should it be tidied first?

**Verdict: tidy first, but only the small set listed in section "Highest-leverage actions" below.** The bulk of the code is in good shape: the `pipeline/` package is a genuine single source of truth, the heuristics dispatch is the cleanest module in the repo, and `run_tracker.py` / `run_overview.py` earn their keep. What needs to land before X3D-S is a focused pass of perhaps half a day's work, not a structural rewrite. Doing it now prevents copy-paste duplication on the X3D-S training entry point and pins behaviour on the load-bearing pose heuristic before a second consumer arrives.

The path/IO abstraction, the larger module deduplications, and the bulk style/comment pass are explicitly *not* recommended for now. They benefit from seeing the wrist-crop shape first.

---

## Focus area 1: Sprawl and maintainability for phase 2

### Evidence

- **`bst_x_train.py` is 885 LOC and conflates four concerns** (`src/bst_x/stroke_classification/main_on_shuttleset/bst_x_train.py:1-885`): `Hyp` namedtuple plus ~80 lines of dated tuning rationale (lines 50-157), train/validate/test loops (187-355), `Tee` stdout helper (534-543), the `Task` orchestrator (546-733), and a 145-line `__main__` that derives the npy collated dir, hashes the clips CSV, calls `track_run`, and loops 5 serials (740-885). When the X3D-S layer needs its own training script it will reuse everything except the model construction; with the current shape that means duplicating the 145-line `__main__`. The TODO at `bst_x_infer.py:7-12` already names the work: extract `bst_x_common.py` with `MODELS`, a base `Task`, and the shared dataloader helpers.
- **Dead code in `shuttleset_dataset.py` is non-trivial** (`src/bst_x/stroke_classification/preparing_data/shuttleset_dataset.py`): `Dataset_npy` (142-246, deprecated), `Dataset_npy_collated_one_side` (351-420), `Dataset_npy_collated_single_pose` (423-497), plus their three loader helpers (`prepare_npy_loaders`, `prepare_npy_collated_one_side_loaders`, `prepare_npy_collated_single_pose_loaders`). Combined ~330 of 645 LOC. None are imported anywhere on the active path; `Dataset_npy` is referenced only by `Task.compare_pred_gt_on_specific_type` in `bst_x_train.py:706-733`, which is itself unreachable. The ablation-side variants assume `unknown_first=True` and break under `une_merge_v1_nosides`.
- **Hyperparameter rationale accreted as inline commentary** (`bst_x_train.py:65-157`). Two large dated blocks ("LR-SCHEDULE RETUNE 2026-04-17", "AUX-SCHEDULE 2026-04-18") describe past decisions with the prior `Hyp(...)` preserved as a commented literal. Useful history but in the middle of the configuration block, where it noises every diff to `Hyp`. Phase 2 will pile on more (augmentations, focal loss, weighting trials).
- **`prepare_train_on_shuttleset.py` is 1,215 LOC mixing pose extraction with collation**. MMPose 2D/3D extraction (~340 LOC), homography helpers (~150 LOC), shuttle subprocess wrappers (~80 LOC), `pad_and_augment_one_npy_video` (~60 LOC), `collate_npy` (~220 LOC), and a 270-line CLI `main()`. The 3D path (lines 324-406, 603-657) is gated behind `--use-3d-pose` and is not active.
- **`heuristics/` is the cleanest part of the codebase by a wide margin**: one contract (`apply(raw, ctx, **kw) -> HeuristicOutput`), a registry, byte-identity gate, hyperparam pass-through. Copy this pattern for any new model variants.
- **Two `PROJECT_ROOT` definitions** — `pipeline/config.py:15` resolves to `bst_x/`; `pipeline/data_access.py:151` resolves `_PROJECT_ROOT` to the repo root. The names are similar enough to mislead anyone refactoring path resolution.

### Recommendation: **tidy-now**, scoped tightly

Three actions, in order:
1. Extract `bst_x_common.py` with `MODELS`, `Task` base, `Tee`, and the `__main__` plumbing for run-id and clips-CSV hashing. `bst_x_infer.py:7-12` already names this; without it the X3D-S training script becomes a 145-line copy-paste.
2. Delete the dead dataset variants and the unreachable `compare_pred_gt_on_specific_type` debug helper.
3. Move the LR/aux-schedule rationale paragraphs out of `bst_x_train.py:65-157` into `bst_x_overview.md`. Keep only the active `Hyp` config in code; cross-link the writeup.

---

## Focus area 2: BST legacy preservation vs cleanup

### Evidence

- **`TemPose_V`, `TemPose_PF`, `TemPose_SF`, `TemPose_TF` are entirely dead.** `tempose.py` is 710 LOC; the four standalone variant classes account for ~510 of those (lines 156-258, 261-396, 399-526, 529-667) and are never imported outside the file's own `__main__` smoke check (line 681). `bst.py:28` only consumes the building-block utilities (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`). TemPose itself is not a project baseline; the apples-to-apples preservation goal applies to BST, not TemPose.
- **Four of the five `BST_*` partials are imported but unused at runtime.** `bst.py:433-437` defines `BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`, `BST_CG_AP`; `bst_x_train.py:38, 525-531, 859` registers all five and only constructs `BST_CG_AP`. The flag combinations they encode are still meaningful for ablations the BST paper compares, so the partials themselves earn their keep.
- **Inherited TemPose default that contradicts production behaviour**. `prepare_train_on_shuttleset.py:170-175`: `normalize_joints` keeps `center_align=False` as the function-level default verbatim from upstream, but the only caller passes `joints_center_align=True` (line 1179) and `current.py:96-99` also hardcodes `center_align=True`. The signature default is misleading reference noise. Flip the default to True and drop the apologia paragraph; this codebase has no published consumer to break.
- **Dead BST original assets co-located with active code.** `src/bst_x/ShuttleSet/deprecated/{gen_my_dataset.py, get_each_class_total.py, utils.py, README.txt, class_total_gen.xlsx, class_total.xlsx}` are the original BST author's pre-Phase-1 scripts. `src/bst_x/deprecated/before_flattening_asset_dirs/` mirrors a pre-flatten copy of the pipeline and stroke_classification trees. These are inside `src/`, so ruff and IDE indexing walk them and they appear in nearly every `grep`. Git already preserves the history; keeping a parallel snapshot in `src/` is double-bookkeeping.
- **Outdated documentation under `src/`**: `src/bst_x/deprecated/outdated_bst_repo_reusability_assessment.md`, `outdated_bst_models_refactor.md`, `outdated_pipeline_build.md`, `historical_README_bst_original.md`, `historical_predecessor_analysis_summary.md`. Combined ~80KB of phase-0 historical context, useful then, noise on grep now.
- **`tmp/` smoke tests under the active path** (`stroke_classification/main_on_shuttleset/tmp/test_dataloader.py`, `test_fwd.py`, `test_train_step.py`). Useful one-off checks but they sit at code-level prominence and use `sys.path.append('..')` rather than living in `tests/`.

### Recommendation: **tidy-now**

Specifically:
1. Delete the four standalone `TemPose_*` classes from `tempose.py`, leaving the building blocks `bst.py` actually imports (~200 LOC instead of 710).
2. Delete `Dataset_npy` and `compare_pred_gt_on_specific_type` from `shuttleset_dataset.py` and `bst_x_train.py:706-733`. Delete the two `_one_side` / `_single_pose` dataset classes and their loader helpers (covered under focus area 1 too; same touch).
3. Move `src/bst_x/deprecated/`, `src/bst_x/ShuttleSet/deprecated/`, and `stroke_classification/main_on_shuttleset/tmp/` out of `src/`. Either `scratch/historical/` or delete; git history preserves them either way.

The dual goal (preserve BST + extend it) remains sensible at the model layer (the five `BST_*` partials). Everything else is ornamental.

---

## Focus area 3: Naming, comments, and adherence to style principles

### Evidence

- **Cryptic identifiers**:
  - `prepare_train_on_shuttleset.py:55-67`: `get_H()`, `get_corner_camera()`, `convert_homogeneous()`, `project()` are kept verbatim from upstream BST, with one-line docstrings like `"""Get from the pd object."""`. The same code is also cleanly named in `pipeline/court_utils.py`.
  - `prepare_train_on_shuttleset.py:209, 244, 324`: `J=17`, `m`, `xy` are math-paper symbols. Acceptable inside numeric kernels, but `J = 17` is also redefined in `heuristics/sticky_anchor.py:46` and `heuristics/current.py:41`. A shared `JOINTS_COCO = 17` in `heuristics/base.py` would centralise it.
  - `bst_x_train.py:803` reuses `t` and `t1` for "time-axis size", "wall-clock", and "frames" inside the same training loop. Local clarity is fine; renaming `t1` to `epoch_end` would be a five-character win.
- **Comments that explain *what* not *why*, or have rotted**:
  - `bst_x_train.py:1-2`: migration anchor referring to files that no longer exist.
  - `bst_x_train.py:53-57`: refactor cross-ref to a completed migration; trim to a one-line marker.
  - `bst_x_train.py:85-101`: commented-out previous `hyp` block, full of `n_epochs=1600` etc. Per the no-shims-for-unshipped-code principle, delete; rationale lives in `bst_x_overview.md`.
  - `bst_x_train.py:151`: `# Aggressive CG/AP annealing — matches preferred config from run_20260418_151139.` Task-anchored to a specific run id.
  - `bst_x_train.py:389-394`: commented-out alternate scheduler, same pattern.
- **Genuine *why* comments to keep**:
  - `prepare_train_on_shuttleset.py:347-354`: MMPose per-call instantiation workaround.
  - `bst_x_train.py:417, 555`, `bst.py:106-110`: short rationale for non-obvious choices.
  - `shuttleset_dataset.py:277-295`: 19-line "DIVERGENCE FROM ORIGINAL BST" note. Concise enough; load-bearing.
- **Docstring style violations**:
  - `shuttleset_dataset.py:359-373, 432-445`: orphan `Dataset_npy_collated_one_side` and `_single_pose` use rST `.. warning::` blocks repeating themselves. Will go away when the classes are deleted under focus area 1.
  - `data_pipeline_to_model_train.md:392`: dated multi-paragraph tuning blocks embedded in a module reference table. Belongs in `bst_x_overview.md`.
- **AU/UK spelling violations in non-third-party code**:
  - `pipeline/court_utils.py:127-138`: `normalize_position` (US) and "Normalize court coordinates" in the docstring. Heuristic modules import this. Renaming ripples cleanly through the heuristics, but `prepare_train_on_shuttleset.py:142` shares the name as the byte-identity contract with upstream BST. Either rename both or accept US at the upstream-anchored boundary.
  - `pipeline/shuttle_extractor.py:38, 46-52, 252-310`: `normalize_shuttlecock`, `normalized .npy files`, `normalize`. Refactor module, not upstream; rename is free.
  - `pipeline/build_dataset.py:6, 197`: "labeled clips".
  - `pipeline/clip_generator.py:236, 276, 355`: "labeled", "vectorized".
- **Em-dashes throughout prose**: `bst_x_train.py:6, 12, 49, 77, 122, 145, 151, 237, 388, 399, 555, 746, 823`, `bst.py:106, 110`, `clip_generator.py:158, 316`, `prepare_train_on_shuttleset.py:570, 589, 648`. Bulk find-and-replace job.
- **"fade" in `bst_x_train.py:103, 124, 127, 152, 165, 167, 411`**: `aux_fade_end_epoch`, "cosine fade", "fade window". The hyperparam name is embedded in the manifest, `run_overview.py` defaults, and `aim_backfill` tags (`anneal_aggressive` / `anneal_gentle`). Renaming is a 6-file ripple with no behavioural value. **Leave the codename, switch the prose around it from "fade" to "anneal" or "downtune".** That is what the tags already do.
- **`docs/architecture_notes/bst_x_overview.md:158`**: `Δ` as a column header. Per the principle, prefer `change` or `gain`.

### Recommendation: **tidy-after** the X3D-S layer lands

Justification: nothing here is a correctness or maintainability blocker. The comment debt is concentrated in three spots (the commented-out `hyp` block, the orphan dataset docstrings, the homography duplication) that won't grow under phase 2 work. A bulk pass after X3D-S avoids merge churn against the active branch. The two exceptions are the commented-out `hyp` block at `bst_x_train.py:85-101` and the migration-anchor comments at `bst_x_train.py:1-2, 53-57` — those are bundled into the focus-area-1 tidy and should go now.

---

## Focus area 4: Over-engineered utilities and scripts

### Evidence

- **`run_tracker.py`** (`src/bst_x/run_tracker.py`): `track_run` and `track_serial` are exactly the right shape. Two functions, lazy Aim mirror, idempotent by serial_no. **Earning its keep.**
- **`run_overview.py`**: 124-line table-printer with two helpers, six argparse paths. Per-metric mean/stdev/max across runs is the cross-run question phase 2 ablations need. **Earning its keep.**
- **`aim_backfill.py:66-97`**: `_derive_tags()` hardcodes BST CG/AP knowledge (`use_aux_schedule`, `aux_fade_end_epoch`, `anneal_aggressive` / `anneal_gentle`). When Arch 1 / Arch 2 land their own tag axes, this function is the first thing to break or grow. The simpler escape hatch already exists: `manifest['tags']` is appended as-is at line 93. Either move auto-tag derivation onto the train script's manifest-write side, or parameterise the regime detection. **Cleanup candidate, low priority.**
- **`scripts/example_mlflow_run.py`**: 40 lines, last touched April 8, early scaffolding. The README itself concedes it is "more than this project needs" given Aim integration. **Delete now.** Replace the README mention with a one-line note that the project settled on Aim plus manifest.
- **`scripts/test_clip_index.py`**: 30-line one-shot integration check with hardcoded `/home/ahalperi/` paths. **Move to `scripts/archive/` or delete.**
- **Migration scripts that have served their one-shot purpose**:
  - `scripts/flatten_copy.sh` (180 lines) and `scripts/verify_flatten.py` (320 lines): supported a directory-flatten that has shipped and been verified.
  - `scripts/symlink_merge_phase1.py` (152 lines): supported the Phase 1 sticky_anchor mixed retrain, which already ran (`run_20260425_150548`).
  - `scripts/verify_v1_collate.py` (164 lines): pre-shipped V1 collation gate.
  - Combined ~700 LOC of one-shot tooling currently sitting at the same prominence as `rename_videos.py` (still useful for fresh-machine setup).
  - **Move to `scripts/archive/`.**
- **`scripts/rename_videos.py`, `scripts/validate_videos.py`**: still useful for the open "18 videos still need downloading" item. **Earning their keep.**
- **`heuristics/__init__.py:15-18`**: `REGISTRY = {"current": current.apply, "sticky_anchor": sticky_anchor.apply}` is a 2-entry dict. With the phase-2 gap-fill candidate it will be three. The byte-identity gate (`current.py`) is genuine validation infrastructure. **Earning its keep.**
- **`apply_heuristic.py:78-100`**: `_validate_output_dir` collision guard. 22 lines for a 5-line check, justified by the inline *why* comment ("guard against typos destroying data we cannot cheaply recompute"). **Keep.**
- **`failsafe_bst_mmpose_zeroing_check_equivalence.py`** (282 lines): the byte-identity gate, one-shot critical infrastructure. **Keep.** Filename describes the implementation; `byte_identity_gate.py` would describe the role. Optional rename.
- **`pipeline/build_dataset.py:38-40`**: `_step()` is a one-line print helper. Used 6 times. Marginal but consistent. Could inline.
- **`pipeline/build_dataset.py:79-139`**: `dry_run()` (60 lines) duplicates per-step intent text from the run path. No test that they stay in sync. Single-source by deriving descriptions from a list. **Cleanup candidate, low priority.**
- **`pipeline/data_access.py:158-178`**: `_load_dotenv` reimplements ~15 lines of `python-dotenv`. Adding the dep for ~15 lines is a wash; current state is fine.
- **`pipeline/data_access.py` `interactive()` TUI**: the most speculative piece, ~50 lines, no test or workflow doc beyond the README. If nobody on the team has used `python -m pipeline.data_access` interactively, that's dead UI.
- **`pipeline/clip_index.py`**: 65 lines, one function, 28-line pedagogical docstring. Function body is a one-liner. Docstring has educational value but is ~20× the code. **Earning its keep**, but could trim the docstring to 5 lines.

### Recommendation: **tidy-now** for the script archive sweep, **leave-as-is** for everything else

Justification: deleting `example_mlflow_run.py` and moving the four migration scripts plus `test_clip_index.py` into `scripts/archive/` is a five-minute mechanical move that drops the active `scripts/` directory from 9 entries to 4. Phase-2 contributors won't have to ask "is this still part of the build?" The pipeline and heuristics infrastructure is appropriately sized for what is coming.

---

## Focus area 5: Path/IO interface — would centralising help?

### Evidence

- `pipeline/config.py:15` anchors `PROJECT_ROOT` from the file's own location. Every other module under `src/` rolls its own anchor with hand-counted `parents[N]`: `bst_x_train.py:44` uses `parents[4]`, `data_access.py:151` uses `parents[3]`, `validate_zeroed_frames.py:52` uses `parents[1]`, `find_busted_clips.py:53` uses `parents[4]`, `zeroed_frames_class_audit.py:46-47` uses `parents[1]` and `parents[3]` to spell out two roots, `prepare_train_on_shuttleset.py:1021` uses `parents[4]`. Magic numbers drift whenever anything is moved.
- `__main__` blocks in `apply_heuristic.py:33-39`, `raw_extract.py:41-47`, `prepare_train_on_shuttleset.py:29-35`, `failsafe_bst_mmpose_zeroing_check_equivalence.py:41-47`, `bst_x_train.py:30-32`, `bst_x_infer.py:23-24`, `model/bst.py:20` all replay the same `os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))` boilerplate. The only `os.path` use in repo-owned code; mixes idioms with the otherwise pathlib-clean modules.
- The collated-output dir scheme is duplicated verbatim between `prepare_train_on_shuttleset.py:1068-1116` and `bst_x_train.py:744-770`. Both rebuild `effective_ablation_id`, the `npy_[3d_][seq{N}_]{ablation_id}` prefix, and the `preparing_data/ShuttleSet_data_{taxonomy.name}` parent. Comments in both files explicitly note "mirrors the default in...". When phase 2 adds wrist-crop tensors, this mirror has to stay in sync by hand.
- `bst_x_train.py:803-820` builds run paths inline (`timestamp`, `run_id`, `log_path`, `experiments_dir`, `weight_dir`). The X3D-S layer will want the same per-run scaffolding for a new train script; a third copy is the natural next step.
- `apply_heuristic.py:78-100` hand-codes a guard comparing resolved output dirs against `BST_X_MMPOSE_NPY_DIR`. Future tools that write to a derivative of the canonical mmpose dir (wrist-crop extraction, 3D-pose runs) will duplicate the same guard.
- `validate_zeroed_frames.py:1294-1316` and `fail_rate_per_class.py:74-105` each carry a near-identical `*_flat/` auto-discovery routine. If phase 2 adds a wrist-crop sibling under the same data root, this discovery will start matching multiple candidates.
- `pipeline/data_access.py:158-181` already implements env-var-with-default resolution loaded from the repo `.env`. It is just used for one `DataPaths` dataclass; the convention is not reused by the heuristic CLIs or validation scripts.

### Recommendation: **tidy-after**

A small `pipeline.paths` module exposing (a) a single `PROJECT_ROOT` / `BST_REFACTOR_ROOT` / `STROKE_CLASSIFICATION` triple plus a `setup_sys_path()` helper, (b) one helper that builds the `npy_[3d_][seq{N}_]{ablation_id}` collated dir given hyp/taxonomy, and (c) the `BST_X_MMPOSE_NPY_DIR` collision guard, would absorb all of the duplication above. None of it blocks phase 2 the way it stands. The X3D-S wrist-crop path will add a new artifact dir, a new env var, and likely a new collated suffix; that is the right forcing function. Designing the path layer now risks designing for the wrong shape.

What gets harder if you skip the abstraction altogether: when wrist-crop lands, both `prepare_train_on_shuttleset.py` and `bst_x_train.py` will need parallel additions to their inline path-building, the validation scripts' `*_flat` auto-discovery will silently match the wrong dir, and a third sticky-anchor consumer will need to redo the `BST_X_MMPOSE_NPY_DIR` collision logic. Tracked but deliberately deferred.

---

## Focus area 6: Simplification opportunities

Five concrete sites, ordered by effort-to-payoff.

1. **`apply_heuristic.py:265-287` plus `sticky_anchor.py:262-269`**: every sticky_anchor hyperparameter is enumerated three times (CLI `add_argument`, dict-from-args getter, and `hyperparams.get(..., default)` inside the heuristic with the defaults repeated for the third time). Collapse to a single `@dataclass(frozen=True) StickyAnchorParams`. Effort: ~1 hour. **Tidy-now.** Phase 2 will tune these and forgetting one of the three sites is a real bug class.
2. **`bst_x_train.py:744-783`** (~40 lines): `model_info` / `npy_collated_dir` computation has three nested string-builders for `additional_model_info`, `train_partial`, `model_info`, plus a `match` on `seq_len` that only handles 30 and 100. Half is conditional concatenation that could collapse to a small helper or tagged tuple. The `additional_model_info` sentinel is an empty literal at the top of `__main__` and is never set anywhere; it is dead. Effort: <1 hour. **Tidy-now.** Every X3D-S addition will touch this same block.
3. **`prepare_train_on_shuttleset.py:1066-1116`** (~50 lines): mirrors the previous block plus dry-run printing. The two should share one helper. Effort: 1-2 hours, needs a careful diff to confirm string equality. **Tidy-after.** Direct cause of phase-2 path drift between collation and training, but easier to do well after wrist-crop is in.
4. **`prepare_train_on_shuttleset.py:533-657`** (~125 lines): `prepare_2d_dataset_npy_from_raw_video` and `prepare_3d_dataset_npy_from_raw_video` are 80% the same body. Lift the loop into a `_prepare_dataset_from_raw_video(detect_fn, ...)`. Effort: medium; the 3D path is largely dead, so a dry-run-only test is easy. **Tidy-after.** Wrist-crop will look very similar; forking yet another copy is the path of least resistance otherwise.
5. **Validation script triplet**: `validate_zeroed_frames.py` (1300+ LOC), `fail_rate_per_class.py` (227 LOC), `zeroed_frames_class_audit.py` (~330 LOC) all walk `_failed.npy` with similar logic, similar `_Tee` reimplementations, similar Sydney-time output naming, similar `*_flat/` discovery. ~150-line shared core hiding here. Effort: 2-3 hours. **Tidy-after.** Already written, but if anyone touches fail-rate logic post-phase-2 they'll need to update three places.

### Recommendation: **tidy-now** for sites 1 and 2; **tidy-after** for sites 3, 4, 5

Justification: sites 1 and 2 are pure cleanup with no new design surface, in code paths phase 2 will touch every week. Sites 3-5 are deduplications that benefit from seeing the wrist-crop shape first.

---

## Additional point A: `class_weights.json` status

**Static, never read at runtime.** Confirmed by:
- File location: `notebooks/class_weights.json`. Notebook artefact, not under `src/`.
- Generation: `notebooks/01_shuttleset_eda_v3.ipynb:1288, 1302-1304`.
- Runtime references: `grep -rn "class_weight" src/` returns zero hits in any `.py` file. `grep -rn "class_weights\.json"` zero hits in code.
- Active loss construction: `bst_x_train.py:379` is `loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)` with no `weight=` argument. No other loss construction site on the train/eval path.

Per your instruction the file stays as a historical EDA artefact. No action required. Worth a one-line note in the EDA notebook header or the training README clarifying that the JSON is reference-only and the train loop does not read it; otherwise a future contributor will assume it is live config.

---

## Additional point B: where unit tests have the highest leverage

Three sites identified, with verdicts:

1. **`heuristics/sticky_anchor.py:128-253`** — the per-frame pick logic in `_pick_one_frame`: Voronoi partition (line 213), Bottom-first slot ordering with cross-slot exclusion (218-219), sitting-tiebreaker-with-fallback (232-237), rally-presence rejection (242-248). These are the exact invariants that will silently break when wrist-crop becomes a second consumer of `pos` / `joints`. A regression to "Top crowds Bottom out" or "sitting filter empties everyone" produces plausible-looking outputs that quietly degrade phase 2 metrics. **Verdict: now.** Pin before X3D-S so future regressions get attributed correctly.
2. **`heuristics/sticky_anchor.py:280-318`** — EMA update and reset semantics in `apply()`: "EMA resets to halfcourt_centre on full-frame failure" (line 294), "mixed picks reset only the unpicked slot" (302-306), `update_gate_eps` in-court guard before EMA update (line 315). Long-term tracking quality. None exercised by the byte-identity gate. **Verdict: now.** Same argument; this is the part most likely to drift if anyone tweaks the schedule.
3. **`heuristics/current.py:44-101`** — the byte-identity heuristic. What is load-bearing is that it stays bit-equivalent to `detect_players_2d` in `prepare_train_on_shuttleset.py:244-321`. **Verdict: not worth it now.** `failsafe_bst_mmpose_zeroing_check_equivalence.py` already enforces this on a 50-clip real-data sample; a synthetic unit test would have to re-stub `check_pos_in_court` and `normalize_joints` in a way that drifts from upstream.

The Voronoi / EMA tests can use synthetic `RawClip` arrays (a few frames of hand-built bbox/score arrays) and a synthetic court rectangle. They do not need MMPose installed. Tests will be written in a follow-up session by the main agent that has the project context to pick the right invariants.

---

## Additional point C: doc/code drift, worst offenders

1. **`.claude/project_overview.md:94-96`** — describes `preparing_data/heuristics/base.py` as having "abstract `HeuristicFilter` (filter, serialise, deserialise)" and `current.py` as "`BasicFilter` (no-op baseline)". **Stale.** Reality: `base.py` defines three NamedTuple/dataclass types (`RawClip`, `ClipContext`, `HeuristicOutput`); each variant exposes `apply(raw, ctx, **hyperparams)` registered in `heuristics/__init__.py:REGISTRY`. No class hierarchy, no `BasicFilter`, no serialise/deserialise. The doc describes a class-based design that was never built or got refactored away. Fix this in the same pass that the rest of the orientation doc was just updated; it slipped through.
2. **`src/bst_x/data_pipeline_to_model_train.md:392`** — `Hyp` defaults documented as `n_epochs=1600, batch_size=128, lr=5e-4, warm_up_step=400, early_stop_n_epochs=300, taxonomy='merged_25', ...`. Reality (`bst_x_train.py:140-157`): `n_epochs=80, warm_up_step=100, early_stop_n_epochs=40, taxonomy='une_merge_v1_nosides', use_aux_schedule=True, aux_fade_end_epoch=15, split_column='split_v2', drop_unknown=True`. The "active retune" notes embedded in the table partially update this but disagree among themselves (one block claims `n_epochs=120`; the active value is 80). **This is the worst offender** because it presents two-week-old tuning state as the live module reference.
3. **`src/bst_x/data_pipeline_to_model_train.md:121, 535`** plus **`src/bst_x/pipeline/README.md:165, 247`** — list taxonomies as `une_merge_v1` / `merged_25` / `raw_35` only. `pipeline/config.py:206-212` defines `une_merge_v1_nosides` as well, and it is the active default.
4. **`src/bst_x/data_pipeline_to_model_train.md:260-263`** — Stage 3 Dataset section lists `Dataset_npy_collated_one_side` and `Dataset_npy_collated_single_pose` as primary classes alongside `Dataset_npy_collated`. Neither has any caller in active code. Either delete from doc or mark as orphaned. This will sort itself when the focus-area-1 deletion lands.
5. **`docs/architecture_notes/bst_x_overview.md:101`** — references `bst_x_train.py:255 originally passed num_cycles=0.25 into get_cosine_schedule_with_warmup`. Current line 255 is `shuttle: Tensor = shuttle.to(device)`; the cosine scheduler is now at `bst_x_train.py:395-400`. Line numbers drifted with per-epoch timing logging and the resume_from block. Other line refs in the same doc still point correctly.
6. **`src/bst_x/run_tracker.md:64-90`** — manifest format example is missing `extra: data_provenance: {clips_csv_path, clips_csv_sha256, effective_ablation_id, npy_collated_dir}` per `bst_x_train.py:833-840`. Live manifests have it (e.g. `run_20260425_185421/manifest.yaml:161-164`). Doc shows `tags`, `notes`, `best_serials` but not the field every recent run actually populates.
7. **`README.md:78-82`** — `## Experiment Tracking` mentions the MLflow stub. If `example_mlflow_run.py` is deleted (focus-area-4 recommendation), update this paragraph.
8. **`README.md:100-104`** — `tests/test_environment.py` is described as the only environment test. The directory actually has `test_api.py`, `test_data_access.py`, `test_dataset.py`, `test_environment.py`, `test_integration.py`. Understates coverage.
9. **`README.md:170-220`** — HPC symlink instructions reference `ShuttleSet_data_une_merge_v1` only. Active workflow uses `ShuttleSet_data_une_merge_v1_nosides` and per-ablation `npy_*_dropunk_h_sticky_anchor` tagging. Not wrong, but doesn't reflect current usage.
10. **`scripts/symlink_merge_phase1.py:14`** — docstring claims "Run from the repo root or from `src/bst_x/stroke_classification/`", but defaults are repo-relative, so it only works from repo root.

---

## Highest-leverage actions, ordered

These are the actions this review recommends before X3D-S work begins. Total estimated effort: half a day.

1. **Extract `bst_x_common.py`** with `MODELS`, `Task` base, `Tee`, dataloader helpers, and the `__main__` plumbing for run-id and clips-CSV hashing. This is the single most leveraged move; without it the X3D-S training script will copy-paste the 145-line `__main__` from `bst_x_train.py`. `bst_x_infer.py:7-12` already names the work.
2. **Delete dead code in two files**:
   - `tempose.py`: drop `TemPose_V`, `TemPose_PF`, `TemPose_SF`, `TemPose_TF` (~510 LOC). Building blocks `bst.py` imports stay.
   - `shuttleset_dataset.py`: drop `Dataset_npy`, `Dataset_npy_collated_one_side`, `Dataset_npy_collated_single_pose`, plus their three loader helpers. Drop `Task.compare_pred_gt_on_specific_type` from `bst_x_train.py:706-733` (the only `Dataset_npy` caller).
   Roughly 850 LOC of dead weight gone from the two files X3D-S work will touch most.
3. **Collapse the sticky_anchor hyperparam triplication** at `apply_heuristic.py:265-287` and `sticky_anchor.py:262-269` into one `StickyAnchorParams` dataclass. Phase 2 will tune these.
4. **Write unit tests pinning `sticky_anchor.py` invariants**: `_pick_one_frame` (Voronoi partition, Bottom-first slot order with cross-slot exclusion, sitting-tiebreaker, rally-presence rejection) and `apply()` (EMA reset semantics, mixed-pick reset, update_gate_eps guard). Synthetic data; no MMPose dependency. The main agent will write these in a follow-up session.
5. **Archive completed-phase scripts**:
   - Delete `scripts/example_mlflow_run.py`. Update the README experiment-tracking paragraph.
   - Move `scripts/flatten_copy.sh`, `scripts/verify_flatten.py`, `scripts/symlink_merge_phase1.py`, `scripts/verify_v1_collate.py`, `scripts/test_clip_index.py` into `scripts/archive/`.
6. **Move historical snapshots out of `src/`**:
   - `src/bst_x/deprecated/` and `src/bst_x/ShuttleSet/deprecated/` to `scratch/historical/` (or delete; git preserves them).
   - `src/bst_x/stroke_classification/main_on_shuttleset/tmp/` to `scratch/` or `tests/smoke/`.
7. **Tidy `bst_x_train.py` configuration block**:
   - Move LR/aux-schedule rationale paragraphs (lines 65-157) into `bst_x_overview.md`. Keep only the active `Hyp` config in code; cross-link the writeup.
   - Delete commented-out previous `Hyp` block (lines 85-101) and commented-out scheduler (lines 389-394).
   - Delete migration-anchor comments (lines 1-2, 53-57).
8. **Fix the worst doc-drift items**:
   - `.claude/project_overview.md:94-96`: rewrite the heuristics description (no `HeuristicFilter` class hierarchy exists).
   - `data_pipeline_to_model_train.md:392`: rewrite the `Hyp` defaults table.
   - `data_pipeline_to_model_train.md:121, 535` and `pipeline/README.md:165, 247`: add `une_merge_v1_nosides` to the taxonomy list.
   - `run_tracker.md:64-90`: add `extra: data_provenance` to the manifest example.
   - `bst_x_overview.md:101`: refresh the `bst_x_train.py` line refs (now at `:395-400`).
9. **Collapse `bst_x_train.py:744-783`** model_info / npy_collated_dir block. Drop the dead `additional_model_info` sentinel.

---

## Things to defer until after X3D-S lands

These were called out and explicitly downgraded by at least one reviewer; documenting here so they are not lost.

- **Path/IO abstraction.** A `pipeline.paths` module is the right move long-term, but the right shape will only be visible once wrist-crop adds its own artifact dir, env var, and collated suffix. Doing it now risks designing for the wrong shape.
- **Larger module deduplications**: `prepare_train_on_shuttleset.py` 2D/3D extraction (~125 LOC overlap), the `prepare_train` / `bst_x_train` mirror block (~50 LOC overlap), the validation-scripts triplet (~150-line shared core). Each benefits from seeing the X3D-S shape first.
- **Bulk style pass**: AU/UK rename (`normalize_*`, "labeled", "vectorized"), em-dash sweep, "fade" prose to "anneal" or "downtune" while leaving the codename. Five files; pure churn that will conflict with active feature work.
- **`aim_backfill._derive_tags()` parameterisation**: only matters once Arch 1 / Arch 2 land their own tag axes.
- **`pipeline/build_dataset.py` `dry_run()` / `_step()` consolidation**: low priority.
- **`pipeline/clip_index.py` docstring trim**: low priority.

---

## Notes

- All three reviewers were memory-isolated subagents. None saw the user's local auto-memory or this conversation's prior context. They saw only the repo and the checked-in `.claude/*.md` orientation files. Their findings are independent.
- The reviewers disagreed in places. The biggest split was on path/IO abstraction (one reviewer's "tidy-after" verdict, no contradictory verdict from the others). The biggest agreement was on the dead-code deletions in `tempose.py` and `shuttleset_dataset.py`, the script archive sweep, and the `bst_x_train.py` extraction.
- The "fade" prose substitution and the AU/UK rename are deferred deliberately: they conflict with active feature work for negligible behavioural value. Worth a single tidy commit after the X3D-S branch lands.
- `class_weights.json` is dead config but is staying put as an EDA artefact for the writeup; documenting in the synthesis only because three reviewers would otherwise have flagged it as "can be deleted".
