# shared/ — values mirrored from BST

Code that BRIC needs but was initially created in BST, existing files remain a dependency until BST mapped to this directory: stroke taxonomy,
court geometry.

## Modules

| Module | Mirrors | Purpose |
|--------|---------|---------|
| `taxonomy.py` | `bst_refactor.pipeline.config` | The 14-class stroke taxonomy + class-merge rules. The single source of truth for `stroke_type` strings emitted by the contract. |
| `court.py` | `bst_refactor.pipeline.court_utils` | Homography utilities (`project`, `convert_homogeneous`, `get_court_info`) + reference court constants (`REF_COURT_M`, `REF_COURT_CORNERS_M`). |

## Why copy not import

`src/bst_refactor/` is a dependancy for Model A. Importing into BRIC
would couple to BST's internal structure (which can move without
warning) and create a cycle if BST ever needs to share with us.

The trade-off is drift: if BST renames a class or shifts a court
constant, our copy doesn't notice. v2 wants a `tests/test_shared_drift.py`
that asserts our `taxonomy` matches BST's `pipeline.config` — caught
in CI, fixed in a single commit. v1 ships without it.

## Public API at a glance

```python
from shared.taxonomy import TAXONOMY_UNE_MERGE_V1_NOSIDES, DEFAULT_TAXONOMY
from shared.court import (
    REF_COURT_M, REF_COURT_CORNERS_M,
    get_court_info, project, convert_homogeneous,
)
```

## Related docs

- [`docs/decisions_log.md`](../../docs/decisions_log.md) — DL-014 / DL-015 cover
  the taxonomy choice and the no-touch boundary.
