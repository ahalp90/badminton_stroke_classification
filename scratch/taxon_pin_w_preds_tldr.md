# BST taxonomy + predictions refactor: heads-up

For the FE team and the BRIC team, before this lands. Full plan in `scratch/collation_taxon_pin_w_preds_refactor.md`.

Two reasons for the refactor. The FE needs per-clip confidence figures to display in the model browser, which requires dumping raw logits and top-k indices per clip at training time and aligning them to clip stems at collation time. The writeup needs a BST 25-class benchmark on the post-augmentation best model so the headline table is paper-comparable. Both reasons require a re-extract of the 1,278 unknown-class clips that the original pose extract filtered out, new collations under the BST taxonomies, and new training runs.

With the rebuild forced, several taxonomy monkey-patches come off in the same pass. The runtime active-class adapter deletes. The `unknown_first` flag deletes. labels.npy moves to active class space directly. The `Taxonomy` dataclass simplifies to the four fields it actually needs. Two asserts at train start replace the adapter's defensive role.

A small merge-map bug surfaced during planning. `pipeline/config.py:68` folds `driven_flight` into `unknown` for the 25-class taxonomy; the BST paper supplementary (Table G) folds it into `drive`. ~52 clips out of ~33k have been mislabelled in every historical `merged_25` run. The fix sits in the same lines as the bandaid rip, so it lands in the same pass.

Generation tag for the new collations and runs: `taxon_pinned_w_preds`.

## What changes on the BST side

- MERGE_MAP gets the paper-faithful fold (`driven_flight` → `drive`).
- The `Taxonomy` dataclass simplifies to `classes` (the final list in label-index order), `merge_map`, `has_sides`, `excludes_raw`. Unknown sits at index -1 when the taxonomy includes it.
- The runtime active-class adapter deletes. Two asserts at train start replace its defensive role; labels.npy lands in active class space directly.
- BST manifests stop writing the `extra.arch.…` block and write `config.classes: [list]` directly, matching what BRIC manifests already do.
- Training writes raw logits + top-5 per (split, serial) as npz under `<run_dir>/predictions/`. Collation gains a `clip_stems.npy` sidecar so per-clip alignment is a row-index join rather than a derive-from-collation script.
- 1,278 unknown clips get a sibling pose extract on engelbart. Eight new training runs come out the other end.

## BRIC team: nothing to do

`src/shared/taxonomy.py` and everything under `src/bric/` stay where they are. BRIC keeps its `Taxonomy` shape, its `trainable_class_list()` method, its data pipeline. The acknowledged manual-mirror drift in `shared/taxonomy.py`'s docstring stays as a documented permanent state rather than something the refactor reaches in to fix.

The convergence point is the manifest. BRIC already writes `config.classes`; BST starts doing the same after the refactor. A side effect comes for free: BRIC entries register correctly in the FE handler (FE shape currently not adapted to what you're putting out).

## FE team: one small PR, ~30 lines

`src/api/registry.py` and `src/api/inference.py` currently read class lists from `manifest.extra.arch.active_class_list`. That field disappears for new BST runs. The patch is a small resolver that prefers `manifest.config.classes` and falls back to the legacy field. The same PR covers the predictions-JSON `active_class_list` → `class_list` rename with back-compat for the mock JSONs already on disk.

After the patch lands:

- The existing `run_20260505_154907` mock entry keeps working through the fallback.
- New post-refactor BST entries work through the canonical path.
- BRIC entries register correctly for the first time. The current handler only looks in BST's bandaid block, so BRIC has been coming back with an empty class list.

The PR arrives with a changelog and a test plan attached. Nothing for the FE team to scope ahead of time.

## What's not in this refactor

- **Predictions-JSON converter.** New runs emit npz, not the FE-shaped per-clip JSON. A small post-hoc tool reads the npz (which now carries `clip_stems`) + manifest and emits `predictions/{val,test}.json`. Decoupled from training; lands when the FE side is ready to consume real (non-mock) predictions.
- **Temperature-scaled calibration.** Raw logits are in the npz, so post-hoc temperature fitting against val NLL is a small standalone job. Not bundled with the train cycle.
- **`src/api/bst_x_inference.py` hardcoded class list.** The live-inference path hardcodes the 14-class list as a Python constant. Fine for the current best model; breaks when new taxonomies register. Flagged in the PR explanation as a forward concern; FE-team workstream call when to parameterise it.

## Bottom line

BRIC stays insulated. FE gets one PR with a clear changelog. The BST side ends up with clean per-clip predictions for the FE to serve, a paper-comparable BST 25-class benchmark for the writeup, and a cleaner train loop without the taxonomy monkey-patches.
