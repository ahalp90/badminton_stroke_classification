# Pipeline Context Notes

**Purpose.** Companion to `historical_bst.md`. Captures context notes excised from the `pipeline/` area of the source tree (and adjacent infrastructure: `run_tracker.py`, `aim_backfill.py`, `run_overview.py`, validation scripts).

Same conventions as `historical_bst.md`: each section records original location, why preserved, why removed, and verbatim content as a fenced code block.

---

## Status

Skeleton drafted 2026-04-26. The pre-phase-2 tidy execution plan does not move any pipeline-area context out of source on this branch. Sections below are reserved placeholders for future excisions (most likely landing in the post-X3D-S tidy pass).

---

## 1. Reserved: `pipeline/build_dataset.py` `_step()` and `dry_run()`

If the `dry_run()` consolidation lands post-X3D-S (focus area 4 in the review doc), the per-step intent text being deduplicated will land here as a record of the original step descriptions.

---

## 2. Reserved: validation script `_Tee` reimplementations

If the validation triplet (`validate_zeroed_frames.py`, `fail_rate_per_class.py`, `zeroed_frames_class_audit.py`) gets a shared core, the original three `_Tee` implementations land here as a record of what was deduplicated.

---

## 3. Reserved: `aim_backfill._derive_tags` BST CG/AP knowledge

If `_derive_tags` is parameterised away from BST CG/AP specifics post-X3D-S, the original hardcoded regime detection (the `anneal_aggressive` / `anneal_gentle` / `cg_ap_off_from_start` / `no_aux_anneal` rules) lands here.

---

## Cross-references

- `docs/architecture_notes/historical_bst.md` — BST-core excisions.
- `docs/architecture_notes/pre_phase_2_review_2026-04-26.md` — review that drove the tidy pass.
- `docs/architecture_notes/pre_phase_2_tidy_plan.md` — execution plan for the tidy pass.

---

## Maintenance

Same convention as `historical_bst.md`: append-only, no in-place edits to historical entries.
