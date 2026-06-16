# Rebrand smoke (H1)

Pre/post artefact-inventory harness for the BST to BST-X rebrand. Closes the gap pytest deliberately doesn't cover: the local untracked-weight superset (122 untracked weights on top of the 64 tracked), TB byte-listings, and the resolution map of the 312 manifest `weights_path` entries (175 of which point at pruned files by design, so a "must all resolve" pytest assert can't be the right tool).

The pytest suite at `tests/test_namespace_migration.py` carries every standing invariant. This harness is the destructive-step instrument: capture the world before, verify it after.

Spec: `scratch/architecture_notes/namespace_migration_test_design.md`, section H1.

## When to run

- **Step 6b.2 (weight file rename):** capture immediately before; verify immediately after.
- **Step 8 (`src/bst_x` -> `src/bst_x`):** capture immediately before; verify immediately after, passing the src-map.

Verifying after non-destructive steps is a free no-op pass; do it any time the inventory file is convenient.

## Usage

```
# Pre-6b.2 baseline (laptop venv: badminton-cicd)
python scratch/rebrand_smoke/artefact_inventory.py --root . \
    capture scratch/rebrand_smoke/baselines/pre_6b2.json

# After 6b.2 lands:
python scratch/rebrand_smoke/artefact_inventory.py --root . \
    verify scratch/rebrand_smoke/baselines/pre_6b2.json

# Pre-Step-8 baseline (after 6b.2 has settled):
python scratch/rebrand_smoke/artefact_inventory.py --root . \
    capture scratch/rebrand_smoke/baselines/pre_step8.json

# After Step 8 lands, pass the src-map so the verify recognises the dir move:
python scratch/rebrand_smoke/artefact_inventory.py --root . \
    verify scratch/rebrand_smoke/baselines/pre_step8.json \
    --src-map src/bst_x=src/bst_x
```

Exit code is non-zero on any mismatch; the report prints offender lists by category.

## What it checks

- Every `*/weights/*.pt` under `src/<pkg>/stroke_classification/main_on_shuttleset/experiments/` matches its baseline under the expected-name map: `bst_CG_AP_<rest>.pt` -> `bst_x_<rest>.pt` inside `run_*/weights/`, and `bst_CG_AP_<rest>.pt` -> `bst_cg_ap_<rest>.pt` inside the Chang baseline dir.
- Per `(manifest, serial)`: the resolve flag is unchanged, and the manifest's `weights_path` basename matches the mapped expectation.
- `fe_jsons/*` and `predictions/*.npz`: byte-identical (same relpath under src-map, same sha256).
- `tb/**`: name + size identical (no content hash; the lists + sizes catch the only thing decision 7 needs).
- `docs/models_registry.yaml`: all manifest_path + weights_path entries resolve.
- `scripts/model_manifest.tsv`: every `dest_path` resolves under the same expected-name map.

## Outputs

Baselines land under `scratch/rebrand_smoke/baselines/` and are gitignored: they encode local-superset state and aren't meaningful on other machines. The capture mode prints summary counts; verify exits 0 (green) or 1 (with offender lists on stderr).
