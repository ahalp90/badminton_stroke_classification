# Post-tidy bit-exact smoke

Two scripts that close the last gaps in the pre-phase-2 tidy verification.
Both are end-to-end checks that run on engelbart and compare post-tidy
output against a known-good reference.

## What the main gate already proved

| Surface | Gate |
|---|---|
| Heuristic dispatch + sticky_anchor | Byte-identity gate (50/50 stems exact) + 7 unit tests |
| Pipeline.config helpers (collated dir naming) | Manifest match between post-tidy and main 2-epoch runs |
| BST model graph + `bst_x_common.build_bst_x_network` | 2-epoch smoke train within run-to-run noise |

## What these scripts close

| Script | Surface | Cost | Why it's needed |
|---|---|---|---|
| `smoke_infer_bit_exact.py` | `bst_x_infer.py` (build_bst_x_network adoption) | ~10 min, venv-bst | Inference path was not exercised by any other gate |
| `smoke_prepare_2d_bit_exact.py` | `prepare_train_on_shuttleset._prepare_dataset_from_raw_video` lift | ~30-60 min, venv-mmpose | Pose-extraction path uses pre-tidy data on this branch |

## Recommendation

Run **`smoke_infer_bit_exact.py`** first (low cost, high information). If it
passes IDENTICAL across the two branches, the inference refactor is verified.

Run **`smoke_prepare_2d_bit_exact.py`** only if you want belt-and-braces
coverage on the pose-extraction lift. The change is mechanical (move loop
into helper, thread kwargs through) and would crash loudly on the first
clip if a typo had snuck in. Reading the diff is cheaper than running.

## How to run

See the docstring at the top of each script for env vars + step-by-step.
