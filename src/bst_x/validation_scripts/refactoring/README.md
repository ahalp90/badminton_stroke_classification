# Refactor verification scripts

Bit-exact and equivalence smokes for refactor passes that touch the model,
training loop, data pipeline, or pose extraction. The pre-phase-2 tidy entries
were the first; the simplification pass added more. Future refactor passes'
equivalence gates land here too.

## Pre-phase-2 tidy

| Script | Surface | Cost | Venv | Why it's needed |
|---|---|---|---|---|
| `smoke_infer_bit_exact.py` | `bst_x_infer.py` (build_bst_x_network adoption) | ~10 min on HPC | venv-bst-x | Inference path; bit-compares predictions between two branches |
| `smoke_prepare_2d_bit_exact.py` | `prepare_train_on_shuttleset._prepare_dataset_from_raw_video` lift | ~30-60 min on HPC | venv-mmpose | Pose-extraction path; bit-compares per-stem clean output between two branches |

## Simplification pass

| Script | Surface | Cost | Venv | Why it's needed |
|---|---|---|---|---|
| `smoke_b1_validate_gpu.py` | `bst_x_train.validate()` | ~seconds on bourbaki / engelbart | venv-bst-x | Device-mismatch bugs in the per-class accumulators are silent on CPU and crash on GPU; this is the only GPU-only gate |
| `smoke_b6_npz_writer.py` | `bst_x_common._write_prediction_npz` | ~seconds local | badminton-cicd | The npz schema is consumed by `build_fe_stats_jsons`, `calibration_ece`, and the FE downstream; protects against silent schema drift + same-dtype field transposes |
| `smoke_b7_seeded_train.py` | `bst_x_train.train_network` | ~seconds CPU / ~minute cuda | badminton-cicd (CPU) or venv-bst-x (cuda) | End-to-end bit-exact for any future change in the training loop or its setup |
| `seed_and_run_bst_x_train.py` | Tooling, not a gate | Depends on the wrapped run | venv-bst-x for cuda | The only way to drive a real `bst_x_train` run with a pinned RNG without editing the production source |
| `compare_b7_real_runs.py` | Tooling, not a gate | ~seconds local | badminton-cicd | Consumer side of the launcher above; diffs two run dirs (checkpoint .pt SHA, prediction npz per-key, manifest per-serial metrics) |
| `tier1_comment_check.py` | Any comment-strip pass | ~seconds local | badminton-cicd | Mechanical "is this really comment-only" gate via AST diff with the `--help`-bearing carve-out for the three `__doc__`-fed modules (`data_access`, `build_dataset`, `apply_heuristic`); has a `--selftest` |

## Parent-level utility

`../collation_fulldiff.py` lives one dir up because it's a general data-pipeline
diff utility rather than a refactor-specific gate.

Surface: any `collate_npy` edit (or any pipeline change upstream of collation
that should preserve the per-clip arrays). Cost: ~minutes per taxonomy on HPC.
Venv: any with NumPy (no `bst_x` import). Why: per-clip element-wise diff
across the full collated dataset, aligned by `clip_stem`, memory-safe (`mmap` +
row-chunks). Aggregate training metrics are permutation-invariant and lossy;
this is the gate that actually catches a row shuffle or a small value drift in
the write-out.

## Patterns

Two reusable practices the scripts here lean on. Worth knowing about before
adapting one of them for a fresh refactor.

### Dual-invocation bit-exact

Main writes the reference into a scratch dir; the branch reads that dir as the
committed-side reference. Reason: the production extracts under
`$BST_X_MMPOSE_NPY_DIR` track `sticky_anchor`, so a fresh-extract test needs
the same heuristic on both sides. The smoke on each side writes into its own
`SCRATCH_DIR`; the branch run also sets `REFERENCE_DIR` to the main-side
scratch dir, and the comparison step happens inside the script.

`smoke_prepare_2d_bit_exact.py` is the working example.

### Seeded synthetic train for bit-exact

`bst_x_train`'s live path is unseeded by design. Cross-branch bit-exact
requires the seed pinned before the build+train sequence. Two scripts work
together:

- `seed_and_run_bst_x_train.py` is the launcher. It pins the seed + cuda
  deterministic flags, then `runpy.run_module('bst_x_train', ...)` with
  whatever CLI flags follow. Run it once per side
- `compare_b7_real_runs.py` is the diff tool. It takes the two resulting run
  dirs and compares checkpoint .pt SHAs, prediction npzs per key (skipping
  `run_id`, which legitimately differs by `--run-id`), and manifest per-serial
  metrics

For a self-contained CPU bit-exact that doesn't need a real `bst_x_train`
run, `smoke_b7_seeded_train.py` is the synthetic version.

## How to run

Each script's docstring carries its env vars, args, and a step-by-step
invocation. The Venv column above lines up with the docstring assumptions.
