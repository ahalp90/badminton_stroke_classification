# project_history/

Historical content moved out of the active source tree to keep `src/` focused on code that is on the active build and run paths. Nothing in here is imported by current code; this directory exists to preserve the project's lineage for the end-of-project report and for any future reproduction or revert work.

## Layout (filled as content is moved in)

- `bst_refactor_deprecated/` — was `src/bst_refactor/deprecated/`. Author's pre-phase-1 scripts (`gen_my_dataset.py`, `get_each_class_total.py`, etc.), the pre-flatten snapshot of `pipeline/` and `stroke_classification/` (`before_flattening_asset_dirs/`), and phase-0 historical documentation (`outdated_*.md`, `historical_*.md`).
- `shuttleset_deprecated/` — was `src/bst_refactor/ShuttleSet/deprecated/`. The original BST author's pre-refactor ShuttleSet helpers.
- `main_on_shuttleset_tmp/` — was `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/`. Phase-0 smoke-test scripts (`test_dataloader.py`, `test_fwd.py`, `test_train_step.py`).

## Move history

- 2026-04-26: relocated as part of the pre-phase-2 tidy. See `scratch/architecture_notes/pre_phase_2_tidy_plan.md` step 3 for the full action list and `scratch/architecture_notes/historical_bst.md` for any in-source code/config that was excised at the same time.

## Conventions

- Append-only. Do not edit moved files in place; if behaviour is later restored or revised, capture the new version in the active source tree and cross-reference it here.
- New entries get a one-line entry under "Layout" plus a dated note under "Move history".
