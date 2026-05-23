# Refactor plan: contractual taxonomies + clip-aligned predictions npz

Working doc for the bandaid-rip + re-extract + collation + predictions-dump pass.

Generation tag for all new collations and runs in this pass: `taxon_pinned_w_preds`.

## Table of contents

- [Status](#status) — parking gate (Arch2 + FE-integration impact assessment)
- [Pre-flight verification](#pre-flight-verification-run-before-starting-any-step) — eleven grep / count / dir-existence checks that anchor every load-bearing assumption
- [Background](#background) — why this refactor; the BST paper discovery; what the bandaids are
- [Locked decisions](#locked-decisions) — terse list of every settled call
- [Cells to run](#cells-to-run) — eight cells, headline three at 10 serials, rest at 5
- [Step-by-step plan](#step-by-step-plan) — Steps A through H (Step J between D and E, out of alphabetical sequence by accident of planning order; executed sixth per Step H's execution-order list)
  - A: Taxonomy refactor in `pipeline/config.py`
  - B: Re-extract pipeline (unknown-only stems + raw + clean dirs)
  - C: Collation patch (`unknown_root_dir` + `clip_stems.npy` sidecar)
  - D: Train + infer surface (Hyp tuple, asserts, predictions npz dump, per-class val F1, TB scalars, manifest `config.classes`, `bst_infer --fe`)
  - J: FE handler reconciliation in `src/api/registry.py` + `src/api/inference.py`
  - E: Runner script (`collation_runner.py`)
  - F: Tests
  - G: Docs
  - H: Execution order
- [Migration / back-compat](#migration--back-compat) — alias table, old-run resume, FE consumer notes, cross-arch comparison guidance
- [Touch surface inventory](#touch-surface-inventory) — every file changed, every file confirmed clean, every on-disk artefact produced
- [Drift-detection rules](#drift-detection-rules) — what to do if the pre-flight checks disagree
- [Open items](#open-items) — pending assessments, parked follow-ups
- [Commit message scaffold](#commit-message-scaffold-for-the-rip-commit) — draft text for the rip commit

## Status

Not 100% locked. Before committing, assess the impact against:

- Architecture 2 train code path (its taxonomy assumptions, its label-space contract with the collator, its training entry point)
- FE-integration backend (registry shape, manifest reader expectations, predictions consumer)

The taxonomy-name renames, the manifest schema change (removal of `extra.arch`), the dropped `derive_active_classes_from_labels` adapter, and the new `clip_stems.npy` collation sidecar all need to be vetted against those two surfaces. Lock after the review.

Everything below is the locked-in design pending that assessment.

## Pre-flight verification (run before starting any step)

If the repo has drifted since this doc was written, run these checks first. Each one anchors a load-bearing assumption that the plan rests on. If any of them fail to match, the corresponding step needs re-scoping before lifting.

```bash
# 1. Confirm the MERGE_MAP bug is still present (i.e. nobody has fixed it under us).
grep -n "driven_flight" src/bst_refactor/pipeline/config.py
# Expected: line ~68 maps 'driven_flight' -> 'unknown' under MERGE_MAP.
# Also line ~76 maps 'driven_flight' -> 'drive' under UNE_MERGE_V1_MAP (the good one).

# 2. Confirm derive_active_classes_from_labels is still in bst_common.py and called from bst_train.
grep -n "derive_active_classes_from_labels" src/bst_refactor/stroke_classification/main_on_shuttleset/bst_common.py
grep -n "derive_active_classes_from_labels\|_derive_active_classes_from_loaded_labels" src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py
# Expected: definition at bst_common.py around line 80; caller in bst_train.py around line 761.

# 3. Confirm _validate_and_record_arch is still where the doc thinks it is.
grep -n "_validate_and_record_arch" src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py
# Expected: definition around line 913, callers around 1228.

# 4. Confirm build_extract_stems.py hasn't already grown an --only-unknown flag.
grep -n "only.unknown\|raw_type_en" scripts/build_extract_stems.py
# Expected: --keep-unknown flag exists; no --only-unknown yet.

# 5. Confirm collate_npy is still where the doc thinks it is, with the drop_unknown filter at the docd line.
grep -n "def collate_npy\|drop_unknown" src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py
# Expected: collate_npy def around line 714; drop_unknown filter inside, around line 767-768.

# 6. Confirm the 1,278 unknown clip count from clips_master.csv.
/home/ariel/.venvs/badminton-cicd/bin/python -c "
import pandas as pd
df = pd.read_csv('notebooks/clips_master.csv')
print(f'unknown rows: {(df.raw_type_en == \"unknown\").sum()}')
print(f'driven_flight rows: {(df.raw_type_en == \"driven_flight\").sum()}')
"
# Expected: unknown=1278, driven_flight=52. If these numbers shift, the
# downstream wall-time + cell-count math needs updating.

# 7. Confirm no /scratch dir collisions before re-extract.
ssh engelbart 'ls -la /scratch/comp320a/ | grep -E "(_unknown|taxon_pinned)"'
# Expected: no existing entries with these names. If found, investigate before
# running B2/B3 to avoid clobbering whatever is there.

# 8. Confirm bst_train.py:803 still constructs weight filename from taxonomy.name
#    (this is one of the spots the path-source-from-manifest patch in D7 needs to touch).
grep -n "taxonomy_info = f._.self.taxonomy.name" src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py
# Expected: line ~803.

# 9. Confirm the scratch smoke tests' state (they use class_list() which goes away).
grep -n "class_list\|n_active_classes\|active_class_list" scratch/post_tidy_smoke/smoke_infer_bit_exact.py
# Expected: matches at lines ~117-118 using the removed methods.

# 10. Confirm src/api/registry.py is still reading from extra.arch.active_class_list (the bandaid).
grep -n "active_class_list\|extra.*arch\|config.*classes" src/api/registry.py
# Expected: matches reading manifest.extra.arch.active_class_list at the entry
# build site (around line 96) and preds.active_class_list at the predictions
# read sites (around lines 204, 233). If 'config.classes' appears here already,
# someone landed Patch 2 ahead of us; audit Step J before re-doing.

# 11. Confirm the predictions JSON output dir is still in the placeholder state.
find src/bst_refactor/stroke_classification/main_on_shuttleset/experiments \
    -name 'test.json' -o -name 'val.json' 2>/dev/null | head -5
# Expected: val.json + test.json at run_20260505_154907/predictions/ with
# _mock_data: false, _real_stems: true, y_pred a placeholder the live BST
# endpoint overrides; sibling perclass_stats_{split}.json files alongside.
# Step J's fallback handles this. If a class_list field appears in the JSON,
# the post-hoc converter has started; audit before treating Step J as fresh.
```

If any check disagrees with the expected output, stop and reconcile. The plan assumes the state these greps confirm.

## Background

Two forcing functions, plus an opportunity.

### Forcing function 1: per-clip confidence for the FE

The FE registry handler is wired to display per-clip confidence figures (softmax probabilities, top-k by class) for browsing the model's predictions on val/test splits. Currently the on-disk predictions JSONs are mock data; the real version needs the model to dump per-clip raw logits at training time, with row-to-clip-stem alignment so the FE backend can join row -> stem reliably without re-deriving the collation procedure. This needs:

- Step D4: training-time predictions npz dump (full all-class logits, top-k indices, ground-truth labels) per (split, serial).
- Step C: `clip_stems.npy` sidecar at collation time, row-aligned with `labels.npy`, so the post-hoc FE-shape JSON converter (parked as a follow-up) can do a clean row-index join.

### Forcing function 2: BST 25-class benchmark on the post-Phase-2 / aug-v1 best model

The writeup needs BST-paper-comparable 25-class numbers from the current best pipeline (post-unified pose extract, post-augmentation-v1, post-shuttle-unzeroing wipe_drop). Historical `merged_25` runs were on an earlier pose extract and predate the augmentation framework; their numbers aren't apples-to-apples against the published BST baseline. Producing a clean benchmark needs:

- Step B: a sibling re-extract for the 1,278 unknown-class clips. The original `build_extract_stems.py` filtered them out, so they're absent from the canonical pose extract; BST's 25-class taxonomy keeps unknown as a standalone class, so the unknown-pose extract has to exist before BST25 keepunk collation can land.
- Step C: new collations under `bst_25` and `bst_24` (keepunk and dropunk variants of the BST 25-class merge) on both `split_v2` (project default) and `split_bst_baseline` (BST original).
- Step E (the runner) -> headline cells 5 and 6 at 10 serials each, producing paper-comparable BST25 numbers.

### Opportunity: rip the taxonomy bandaids while everything's rebuilding anyway

The runtime active-class adapter (`derive_active_classes_from_labels` in `bst_common.py`), the `Taxonomy.unknown_first` flag, and the labels.npy-in-full-taxonomy-index-space convention have been quietly piling up since the early pre-Phase-2 days. They're monkey-patches the train loop has been routing around for months. The forced re-extract + recollation + retrain across multiple taxonomies above is the cleanest moment to deal with the lot:

1. labels.npy moves to active class space directly. No runtime remap.
2. `derive_active_classes_from_labels` deletes. Two asserts at train start replace it.
3. `Taxonomy.unknown_first` deletes. Unknown always sits at index -1 when a taxonomy includes it, enforced by `Taxonomy.__post_init__`.
4. `Taxonomy` dataclass simplifies down to `classes` (the final ordered list), `merge_map`, `has_sides`, `excluded_base_stroke_types`. No `base_types` / `standalone_types` / `class_list(side=)` / `active_class_list` / `full_to_active_remap`.

Independently each cleanup item is its own piece of work; bundled with the forced rebuild they share one rip-and-replace per touched file.

### Caught in the cleanup: a MERGE_MAP bug

While planning, spotted that `MERGE_MAP` in `pipeline/config.py:68` applies the 35-class merge convention (`driven_flight -> unknown`) to the 25-class taxonomy. The BST paper supplementary (Table G) explicitly folds `driven_flight` into `drive` for the 25-class. Per `clips_master.csv` the affected count is ~52 clips out of ~33k; numerical impact on the headline metrics is small. Not the driver of the refactor, but the fix lives in the same lines as the bandaid rip so it gets done in the same pass. Net effect: the new `bst_25` collation matches the published BST 25-class convention exactly.

Driven-flight clips themselves were never filtered out at extract time (`build_extract_stems.py` only drops `raw_type_en == 'unknown'`), so their pose data has always been in the canonical extract. The mislabelling was a merge-layer thing, not a missing-data thing. Only the 1,278 literally-unknown clips need a sibling re-extract.

## Locked decisions

- **Taxonomy dataclass**: shape becomes `name`, `classes` (final ordered tuple, unknown at -1 if present), `merge_map` (None or dict), `has_sides` (bool), `excluded_base_stroke_types` (frozenset). No `base_types`, no `standalone_types`, no `unknown_first`, no `class_list(side=)`, no `active_class_list`, no `full_to_active_remap` methods. Side prefixing rule lives in a free function consulting a module-level `SIDE_AGNOSTIC_TYPES = frozenset({'unknown'})`.
- **MERGE_MAP fix**: `driven_flight: drive`. Paper-faithful for the 25-class.
- **Unknown class membership is contractual**: `bst_25` (with unknown at index 24) and `bst_24` (no unknown) are separate Taxonomy entries. Same pattern for any other (with-unknown, without-unknown) family pair.
- **Sides remain contractual** (already settled prior).
- **One re-extract**: 1,278 literally-unknown clips into a sibling dir. Driven_flight clips stay where they are.
- **clip_stems.npy sidecar at collation time**: one stem string per row, aligned with labels.npy. Single source of truth for row->stem mapping. Dataset class loads + applies the zero-length filter to keep alignment.
- **Predictions npz at end of training**: per (split, serial) with full all-class logits, ground-truth labels, top-1, top-5 indices. Consumer derives softmax (and post-hoc temperature scaling) from raw logits.
- **Best-serial retention**: dump all serials during training, manual prune of the non-best after the runner finishes (no automated deletion).
- **Per-class val F1**: snapshot inside the new-best branch in `train_network`; lands in `manifest.serials[n].extra.val_at_best_macro_epoch`. TB scalars added at `F1_val/<class>`.
- **Defensive contract at train start**: two asserts (full coverage in train; subset in val/test). Replaces the runtime adapter entirely.
- **Naming convention**: count alone, no `_w_unk` suffix. New taxonomies: `bst_25`, `bst_24`, `bst_12`, `shuttleset_18`, `une_v1_14`, `une_v1_15`.
- **Generation tag**: `taxon_pinned_w_preds` as the `ablation_id` for every collation written in this pass.
- **Back-compat**: `TAXONOMY_ALIASES` table maps legacy names to new objects. Old `/scratch/.../ShuttleSet_data_<old_name>/` dirs stay untouched. Manifest-recorded taxonomy string drives on-disk path construction; alias table drives the Taxonomy-object lookup.
- **Back-compat phase-out**: noted inline next to the alias table that the intention is to remove entries one-by-one over the coming months as historical runs retire. Manual deletion of paired `ShuttleSet_data_<old>/` dirs goes with each alias removal.
- **`raw_35` removal**: deleted from `pipeline/config.py` with an inline comment explaining how to reinstate it for paper-Table-F parity.
- **`.pt` -> `.npz`**: predictions stored as npz. Old `eval_dump_predictions.py` retired; its capability folds into `bst_infer.py` behind a `--fe` flag (which requires `--fe-output-dir`).
- **Manifest `config.classes` field**: BST manifests gain a `config.classes: [list]` field carrying the resolved class list from `taxonomy.classes`. Mirrors BRIC's existing manifest schema. Provides a self-describing source of truth that the FE registry handler reads without needing to import any taxonomy module. Legacy BST manifests (pre-refactor) keep their `extra.arch.active_class_list` block; the resolver in `registry.py` falls back to it.
- **Predictions JSON field rename**: the per-clip predictions JSON output by the future post-hoc converter uses `class_list` (matching the api_contract registry-entry field), not `active_class_list`. `registry.py` accepts either field name for back-compat with the existing mock JSONs during the transition.
- **`src/api/registry.py` patch lands alongside the refactor**: small resolver in the handler reads `class_list` from `manifest.config.classes` (canonical), falling back to `manifest.extra.arch.active_class_list` (legacy BST). Same change incidentally fixes the latent BRIC-handler bug where `registry.py` was only looking in BST's bandaid block.

## Cells to run

Eight cells. Headline three at 10 serials, rest at 5. All use `ablation_id=taxon_pinned_w_preds`.

| # | Taxonomy | Split | Drop unk | Serials | Notes |
|---|---|---|---|---|---|
| 1 | `shuttleset_18` | `split_v2` | n/a (excluded_base_stroke_types=unknown) | 5 | Raw 18 types, no sides, no unknown. |
| 2 | `bst_25` | `split_v2` | n/a (has_unknown) | 5 | Paper 25-class on project split. Needs unknown sibling dir. |
| 3 | `bst_24` | `split_v2` | n/a (excluded_base_stroke_types=unknown) | 5 | Paper 25-class minus unknown. |
| 4 | `bst_12` | `split_v2` | n/a (excluded_base_stroke_types=unknown) | 5 | Paper merged collapsed to nosides, no unknown. |
| 5 | `bst_25` | `split_bst_baseline` | n/a | **10** | Headline paper-comparable baseline. |
| 6 | `bst_24` | `split_bst_baseline` | n/a | **10** | Headline paper baseline, dropunk. |
| 7 | `une_v1_14` | `split_v2` | n/a | **10** | Refresh of the active best on the new collation generation (gets clip_stems sidecar + predictions npz). |
| 8 | `une_v1_15` | `split_v2` | n/a (has_unknown) | 5 | With-unknown variant of the 14-class for completeness. |

Wall-clock estimate at 25 min/serial on engelbart: 5×5 + 3×10 = 55 serials, ~23 hours total. Doable in two overnights.

## Step-by-step plan

### Step A. Taxonomy refactor in `pipeline/config.py`

**Remove**:

- `MERGE_MAP` original definition (the buggy one with `driven_flight: unknown`)
- `STROKE_TYPES_17_RAW` constant (only used by the deleted `TAXONOMY_RAW_35`)
- `Taxonomy` dataclass fields: `base_types`, `standalone_types`, `unknown_first`
- `Taxonomy` methods: `class_list`, `active_class_list`, `full_to_active_remap`, properties `n_classes`, `standalone_set`, `has_unknown` (rebuilt minimally below)
- All four existing `TAXONOMY_*` constants and the `DEFAULT_TAXONOMY = 'une_merge_v1'` line
- `derive_ablation_id` (the auto-derived default tuple-string is no longer the project's main naming pattern; let `ablation_id` be required from callers, no auto-default)
- The `EN_TO_ZH` / `ZH_TO_EN` blocks stay; they're orthogonal.

**Replace with**:

```python
# Final-state taxonomies: each commits its full class list explicitly. Labels
# on disk are in [0, n_classes) directly; no runtime active/full distinction.

SIDE_AGNOSTIC_TYPES: frozenset[str] = frozenset({'unknown'})

# Cleaned 25-class merge map. driven_flight folds to drive, matching the BST
# paper supplementary Table G. The 35-class merge (driven_flight -> unknown)
# is not represented here because raw_35 was removed in the
# taxon_pinned_w_preds refactor; reinstate if a future arm needs it.
MERGE_MAP_25: dict[str, str] = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}

# Existing UNE_MERGE_V1_MAP stays as-is; already does driven_flight -> drive.

# Module-level base lists (in target class-list order). These are the
# building blocks for the Taxonomy entries below.
STROKE_TYPES_12_MERGED = [
    'net_shot', 'return_net', 'smash', 'lob',
    'clear', 'drive', 'drop', 'push',
    'rush', 'cross_court_net_shot', 'short_service', 'long_service',
]
STROKE_TYPES_14_UNE_V1 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash',
    'lob', 'clear', 'drive', 'drop',
    'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]
STROKE_TYPES_18_RAW = [s for s in STROKE_TYPES_19 if s != 'unknown']


@dataclass(frozen=True)
class Taxonomy:
    """Pinned taxonomy: classes is the authoritative ordered class list.

    Labels.npy values are in [0, len(classes)). No remap at train/infer time.
    """
    name: str
    classes: tuple[str, ...]
    merge_map: dict[str, str] | None
    has_sides: bool
    excluded_base_stroke_types: frozenset[str]

    def __post_init__(self):
        if 'unknown' in self.classes:
            assert self.classes[-1] == 'unknown', (
                f'taxonomy {self.name!r}: unknown must sit at index -1; '
                f'found at {self.classes.index("unknown")}.'
            )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def has_unknown(self) -> bool:
        return 'unknown' in self.classes


def _sided_classes(base: list[str], with_unknown: bool) -> tuple[str, ...]:
    """Build (Top_..., Bottom_..., 'unknown'?) class list from base type names."""
    side_prefixed = [f'Top_{b}' for b in base] + [f'Bottom_{b}' for b in base]
    return tuple(side_prefixed + (['unknown'] if with_unknown else []))


TAXONOMY_BST_25 = Taxonomy(
    name='bst_25',
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=True),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=frozenset(),  # keeps unknown rows
)

TAXONOMY_BST_24 = Taxonomy(
    name='bst_24',
    classes=_sided_classes(STROKE_TYPES_12_MERGED, with_unknown=False),
    merge_map=MERGE_MAP_25,
    has_sides=True,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_BST_12 = Taxonomy(
    name='bst_12',
    classes=tuple(STROKE_TYPES_12_MERGED),
    merge_map=MERGE_MAP_25,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_UNE_V1_14 = Taxonomy(
    name='une_v1_14',
    classes=tuple(STROKE_TYPES_14_UNE_V1),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMY_UNE_V1_15 = Taxonomy(
    name='une_v1_15',
    classes=tuple(STROKE_TYPES_14_UNE_V1) + ('unknown',),
    merge_map=UNE_MERGE_V1_MAP,
    has_sides=False,
    excluded_base_stroke_types=frozenset(),
)

TAXONOMY_SHUTTLESET_18 = Taxonomy(
    name='shuttleset_18',
    classes=tuple(STROKE_TYPES_18_RAW),
    merge_map=None,
    has_sides=False,
    excluded_base_stroke_types=frozenset({'unknown'}),
)

TAXONOMIES: dict[str, Taxonomy] = {
    t.name: t for t in (
        TAXONOMY_BST_25, TAXONOMY_BST_24, TAXONOMY_BST_12,
        TAXONOMY_UNE_V1_14, TAXONOMY_UNE_V1_15,
        TAXONOMY_SHUTTLESET_18,
    )
}

# raw_35 (the BST paper's "excessive granularity" 35-class) is intentionally
# omitted. To reinstate: define merge_map={'driven_flight': 'unknown'},
# has_sides=True, base set = STROKE_TYPES_19 minus {'unknown', 'driven_flight'}
# (17 types Top_/Bottom_ prefixed plus unknown at index -1). The base list
# constant STROKE_TYPES_17_RAW was also deleted in this refactor; rebuild it
# as needed.

# Back-compat for old manifests pointing at the pre-refactor taxonomy names.
# Phase out entries as historical runs retire. Removing an entry implies the
# paired /scratch/comp320a/ShuttleSet_data_<old_name>/ dir is no longer
# needed and can be reclaimed manually.
TAXONOMY_ALIASES: dict[str, str] = {
    'une_merge_v1_nosides': 'une_v1_14',  # current best run_20260505_154907 lives here
    'une_merge_v1':         'une_v1_15',  # legacy with-sides 14-class; semantically loose
    'merged_25':            'bst_25',     # legacy buggy-merge runs; semantically loose
    'raw_35':               'bst_25',     # never collated; aliased for completeness
}


def resolve_taxonomy(name: str) -> Taxonomy:
    """Look up a Taxonomy by name, following the alias table for legacy values."""
    if name in TAXONOMIES:
        return TAXONOMIES[name]
    if name in TAXONOMY_ALIASES:
        return TAXONOMIES[TAXONOMY_ALIASES[name]]
    raise KeyError(
        f'taxonomy {name!r} not registered and not aliased; '
        f'known: {sorted(TAXONOMIES)}; aliases: {sorted(TAXONOMY_ALIASES)}'
    )


def label_for_row(taxonomy: Taxonomy, raw_type: str, side: str) -> int | None:
    """Resolve a per-row class index, or None if the row should be filtered out.

    Used by the collator. excluded_base_stroke_types drops rows before any merge or side-
    prefix step; merge_map applies next; side-prefixing kicks in when
    has_sides=True and the merged type is not in SIDE_AGNOSTIC_TYPES.
    """
    if raw_type in taxonomy.excluded_base_stroke_types:
        return None
    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)
    if taxonomy.has_sides and merged not in SIDE_AGNOSTIC_TYPES:
        label_str = f'{side}_{merged}'
    else:
        label_str = merged
    return taxonomy.classes.index(label_str)


# derive_ablation_id is removed. Callers (collator CLI + train CLI) require
# an explicit ablation_id from now on; no auto-derived tuple-string default.

def derive_npy_collated_dir_basename(
    *, taxonomy_name: str, use_3d_pose: bool, seq_len: int, ablation_id: str,
) -> str:
    """Format the collated dir basename: ``npy_[3d_][seq{N}_]{ablation_id}``.

    Same shape as before but takes ``ablation_id`` as required; split_column
    and drop_unknown are dropped from the signature (they're properties of
    the taxonomy now, not free knobs).
    """
    three_d_tag = '3d_' if use_3d_pose else ''
    seq_tag = '' if seq_len == 100 else f'seq{seq_len}_'
    return f'npy_{three_d_tag}{seq_tag}{ablation_id}'
```

The body of `parse_flaw_records`, `_load_flaw_records`, `EXCLUDED_VIDEOS`, `REMOVED_SHOTS`, `SPLITS`, `CLIP_WINDOW`, `HOMOGRAPHY_RESOLUTION`, and the path constants all stay untouched.

### Step B. Re-extract pipeline

#### B1. `scripts/build_extract_stems.py`

Add `--only-unknown` flag. Three lines plus an error guard if combined with `--keep-unknown`.

```python
parser.add_argument(
    '--only-unknown', action='store_true',
    help='Include ONLY raw_type_en == "unknown" rows (and nothing else). '
         'Mutually exclusive with --keep-unknown.',
)

# Validation
if args.only_unknown and args.keep_unknown:
    parser.error('--only-unknown and --keep-unknown are mutually exclusive')

# Filter
if args.only_unknown:
    filtered = filtered[filtered['raw_type_en'] == 'unknown']
elif not args.keep_unknown:
    filtered = filtered[filtered['raw_type_en'] != 'unknown']
```

Run:

```bash
cd /home/ahalperi/badminton_stroke_classifier
/home/ahalperi/.venvs/venv-bst/bin/python scripts/build_extract_stems.py \
    --only-unknown --keep-busted \
    --output /scratch/comp320a/ShuttleSet_keypoints_raw_unknown/stems_unknown.txt
```

`--keep-busted` because the busted-list is the phase-1 hit-zone-busted set, which is non-unknown by construction; the flag's presence is just defensive against an accidental subset intersection. Expect ~1,278 stems.

#### B2. `raw_extract.py` (no code change)

```bash
PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \
    /home/ahalperi/.venvs/venv-bst/bin/python -m preparing_data.raw_extract \
    --clip-stems-file /scratch/comp320a/ShuttleSet_keypoints_raw_unknown/stems_unknown.txt \
    --save-dir /scratch/comp320a/ShuttleSet_keypoints_raw_unknown \
    --n-max 16
```

Wall time: ~1,278 / 32,203 of the original Phase-2 budget. Roughly 1.5 hours on engelbart.

#### B3. `apply_heuristic.py` (no code change; collision guards already in place)

```bash
PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \
    /home/ahalperi/.venvs/venv-bst/bin/python -m preparing_data.apply_heuristic \
    --raw-dir /scratch/comp320a/ShuttleSet_keypoints_raw_unknown \
    --output-dir /scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown \
    --heuristic sticky_anchor \
    --clips-csv notebooks/clips_master.csv
```

#### B4. Cross-node sync

Manual rsync of both sibling dirs from engelbart to bourbaki after B2 + B3 finish. Same procedure as the canonical Phase-2 dirs.

### Step C. Collation patch

#### C1. `collate_npy` in `prepare_train_on_shuttleset.py`

Three sub-changes:

1. Replace the manual per-row label derivation (lines 774-803) with a call to `label_for_row` from `pipeline.config`. The CSV-level `drop_unknown` filter at line 768 also goes; `excluded_base_stroke_types` on the Taxonomy carries the same information, and the per-row check inside `label_for_row` is now the single point that filters.

2. Add `unknown_root_dir: Path | None = None` parameter. Per-row branch picks the source dir based on `raw_type_en`:

```python
chosen_root = (
    unknown_root_dir if (raw_type == 'unknown' and unknown_root_dir is not None)
    else root_dir
)
branch = str(chosen_root / stem)
```

3. Save `clip_stems.npy` alongside `labels.npy` at the end of the function:

```python
clip_stems_arr = np.array([Path(b).name for b in data_branches], dtype=object)
np.save(str(set_dir / 'clip_stems.npy'), clip_stems_arr, allow_pickle=True)
```

Also remove the `drop_unknown: bool = False` parameter from `collate_npy`; the equivalent control is now `taxonomy.excluded_base_stroke_types`.

#### C2. CLI changes to `prepare_train_on_shuttleset.main`

Add:

```python
parser.add_argument(
    '--unknown-clip-npy-dir', type=Path, default=None,
    help='Flat per-clip dir for raw_type=="unknown" rows. Routes those clips '
         'through this dir while all others come from --clip-npy-dir. '
         'Mutually exclusive with a taxonomy that has unknown in excluded_base_stroke_types.',
)
parser.add_argument(
    '--ablation-id', required=True,
    help='Required. Tag for the collated output dir; no auto-default any more.',
)
```

Remove `--drop-unknown` (and `--split-column` stays untouched, since splits are orthogonal to the taxonomy).

Validation:

```python
taxonomy = resolve_taxonomy(args.taxonomy)
if args.unknown_clip_npy_dir is not None and 'unknown' in taxonomy.excluded_base_stroke_types:
    parser.error(
        '--unknown-clip-npy-dir is set but taxonomy '
        f'{taxonomy.name!r} excludes unknown rows. Either drop the flag or '
        'pick a taxonomy that retains unknown.'
    )
```

The validation error is loud, not a warning.

#### C3. Dataset class change in `shuttleset_dataset.py`

`Dataset_npy_collated.__init__` loads `clip_stems.npy` alongside `labels.npy` and applies the same zero-length filter:

```python
self.labels = np.load(str(branch / 'labels.npy'))
self.clip_stems = np.load(str(branch / 'clip_stems.npy'), allow_pickle=True)
...
valid = self.videos_len > 0
n_dropped = int(np.sum(~valid))
if n_dropped > 0:
    ...
    self.labels = self.labels[valid]
    self.clip_stems = self.clip_stems[valid]
```

`adjust_to_partial_train_set` mirrors the per-class slicing for clip_stems (same `choose_i` index applies):

```python
new_clip_stems.append(self.clip_stems[choose_i])
...
self.clip_stems = np.concatenate(new_clip_stems)
```

Back-compat for legacy collated dirs that don't carry `clip_stems.npy`: graceful fallback to None when the file is missing, with a `warnings.warn` saying the dir predates `taxon_pinned_w_preds` so per-clip alignment isn't available.

### Step D. Train + infer surface

#### D1. `bst_train.py` Hyp tuple

Current Hyp tuple field set (from `bst_train.py:67-76`, pinned here so a cold-context reader doesn't need to look it up):

```
n_epochs, batch_size, lr, warm_up_step,
taxonomy, seq_len, early_stop_n_epochs,
pose_style, use_3d_pose, train_partial,
use_aux_schedule, aux_fade_end_epoch,
clips_csv, split_column, drop_unknown, ablation_id,
label_smoothing, class_weights, adaptive_focal,
augmentation,
expected_active_classes
```

Drop `expected_active_classes` (lever for the deleted runtime adapter). Drop `drop_unknown` (the taxonomy now carries `excluded_base_stroke_types`, so this knob has no independent meaning). Keep all other 19 fields.

Default Hyp values change to reference the new taxonomy names. The pre-refactor default of `taxonomy='une_merge_v1_nosides'` becomes `taxonomy='une_v1_14'`; `ablation_id='wipe_drop'` becomes `ablation_id='taxon_pinned_w_preds'`.

The `_validate_and_record_arch` function is removed entirely. The arch printout at script start is replaced by a simpler block:

```python
def _print_taxonomy_block(taxonomy: Taxonomy, tee):
    with redirect_stdout(tee):
        print(f'[taxonomy] {taxonomy.name}: {taxonomy.n_classes} classes, '
              f'has_sides={taxonomy.has_sides}, has_unknown={taxonomy.has_unknown}')
        print(f'[taxonomy] classes: {list(taxonomy.classes)}')
```

#### D1.5. Write `config.classes` into the manifest

Mirrors BRIC's manifest schema. The resolved class list (from `taxonomy.classes`) is captured in the recorded `config` block so the FE registry handler (and any other consumer) can read the class list directly from the manifest without needing to import the taxonomy module.

In `bst_train.py`, around the `compute_data_provenance` call site (~line 1154-1158), the easiest insertion is to enrich the `extra` block before passing it to `track_run`. But cleaner: have the manifest writer in `run_tracker.py` include the resolved classes. Two equivalent paths:

**Option A (preferred): write into `config`, mirroring BRIC**

In `bst_train.py`, before `track_run(config=hyp, ...)`, build a dict variant of the Hyp that includes `classes`:

```python
config_payload = dict(hyp._asdict())
config_payload['classes'] = list(taxonomy.classes)
run_dir, run_id = track_run(
    config=config_payload, run_id=run_id, log_path=log_path, extra=extra,
    experiments_dir=experiments_dir,
)
```

The `run_tracker._config_to_dict` already handles plain dicts, so this lands `config.classes` next to `config.taxonomy` in the manifest. Same place BRIC writes its class list.

**Option B (alternative): write into `extra` block**

Keep `config` as the verbatim Hyp dump, write `classes` into the `extra` block instead. Cleaner separation of "training hparams" from "resolved derived state," but diverges from BRIC's manifest layout. Not recommended.

Recommendation: **Option A**. The manifest layout converges with BRIC, the FE registry handler reads the same field for both architectures.

#### D2. Asserts replacing `derive_active_classes_from_labels`

In `Task.prepare_dataloaders`, after the loaders are built and labels are loaded:

```python
def _assert_label_coverage(self) -> None:
    expected = set(range(self.taxonomy.n_classes))
    train_present = set(int(x) for x in np.unique(self.train_loader.dataset.labels))
    val_present   = set(int(x) for x in np.unique(self.val_loader.dataset.labels))
    test_present  = set(int(x) for x in np.unique(self.test_loader.dataset.labels))

    missing_in_train = expected - train_present
    if missing_in_train:
        raise ValueError(
            f"taxonomy {self.taxonomy.name!r} has {len(expected)} classes but "
            f"train covers only {len(train_present)}. Missing class indices: "
            f"{sorted(missing_in_train)} "
            f"({[self.taxonomy.classes[i] for i in sorted(missing_in_train)]}). "
            f"Either lift train_partial (currently {hyp.train_partial}) or use "
            f"a taxonomy whose head matches what train can teach."
        )

    for split_name, present in (('val', val_present), ('test', test_present)):
        rogue = present - train_present
        if rogue:
            raise ValueError(
                f"{split_name} has classes absent from train: "
                f"{sorted(rogue)} "
                f"({[self.taxonomy.classes[i] for i in sorted(rogue)]}). "
                f"Fix the split assignment in clips_master.csv or pick "
                f"a taxonomy whose classes match the data shape."
            )
```

`Task._derive_active_classes_from_loaded_labels` is deleted. `self.n_active_classes` and `self.active_class_list` simplify to `self.taxonomy.n_classes` and `list(self.taxonomy.classes)`.

#### D3. `bst_common.py` changes

- Remove `derive_active_classes_from_labels` and its imports.
- Add `dump_topk_predictions` helper (the one we mapped earlier).
- `compute_data_provenance` stays as-is.

```python
@torch.no_grad()
def dump_topk_predictions(
    model: nn.Module,
    loader,
    device,
    k: int = 5,
) -> dict[str, np.ndarray]:
    """Run loader through the model once, return logits + top-k summary.

    Returns dict with: logits (n, n_classes) float32, y_true (n,) int64,
    y_pred_top1 (n,) int64, topk_idx (n, k) int64.
    """
    model.eval()
    logits_ls, y_true_ls, top1_ls, topk_idx_ls = [], [], [], []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose = human_pose.to(device)
        shuttle = shuttle.to(device)
        pos = pos.to(device)
        video_len = video_len.to(device)
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        k_eff = min(k, logits.shape[-1])
        topk_idx = torch.topk(logits, k=k_eff, dim=-1).indices
        logits_ls.append(logits.cpu().numpy())
        y_true_ls.append(labels.numpy())
        top1_ls.append(topk_idx[:, 0].cpu().numpy())
        topk_idx_ls.append(topk_idx.cpu().numpy())
    return {
        'logits':       np.concatenate(logits_ls).astype(np.float32),
        'y_true':       np.concatenate(y_true_ls).astype(np.int64),
        'y_pred_top1':  np.concatenate(top1_ls).astype(np.int64),
        'topk_idx':     np.concatenate(topk_idx_ls).astype(np.int64),
    }
```

#### D4. End-of-serial predictions dump in `bst_train.py`

After `task.test(...)`, before `track_serial(...)`:

```python
task.dump_predictions(run_dir=run_dir, serial_no=serial_no, k=5)
```

New `Task.dump_predictions` method:

```python
def dump_predictions(self, run_dir: Path, serial_no: int, k: int = 5) -> None:
    out_dir = run_dir / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = (
        ('train', self.train_loader),
        ('val',   self.val_loader),
        ('test',  self.test_loader),
    )
    for split_name, source in sources:
        # Fresh shuffle=False loader so the dump row order matches clip_stems.npy.
        ordered = DataLoader(
            source.dataset, batch_size=source.batch_size,
            shuffle=False, num_workers=0, pin_memory=False,
        )
        dump = dump_topk_predictions(self.net, ordered, self.device, k=k)
        out_path = out_dir / f'{split_name}_serial_{serial_no}.npz'
        np.savez(
            out_path,
            logits=dump['logits'],
            y_true=dump['y_true'],
            y_pred_top1=dump['y_pred_top1'],
            topk_idx=dump['topk_idx'],
            class_list=np.array(self.taxonomy.classes, dtype=object),
            run_id=np.array(run_dir.name, dtype=object),
            serial_no=np.array(serial_no, dtype=np.int64),
            taxonomy_name=np.array(self.taxonomy.name, dtype=object),
        )
```

#### D5. Per-class val F1 capture in `train_network`

Inside the new-best branch (currently lines 626-649), add the snapshot:

```python
if curr_macro > best_macro:
    ...
    best_state = deepcopy(model.state_dict())
    # Same epoch as the best macro; not a per-class argmax across epochs.
    best_val_f1_per_class = f1_per_class.detach().cpu().numpy().tolist()
    best_val_present       = present.detach().cpu().numpy().tolist()
    best_macro_epoch_snap  = epoch
```

Initialise `best_val_f1_per_class = None`, `best_val_present = None`, `best_macro_epoch_snap = None` above the loop.

Return signature of `train_network` changes from `model` to `(model, val_at_best)`:

```python
val_at_best = {
    'epoch': best_macro_epoch_snap,
    'per_class_f1': {
        cls: float(best_val_f1_per_class[i])
        for i, cls in enumerate(class_ls)
        if best_val_present[i]
    },
} if best_val_f1_per_class is not None else None
return model, val_at_best
```

Pipe through `seek_network_weights` so it returns `(weight_exists, val_at_best)`. Pass `val_at_best` into `track_serial(..., extra={'val_at_best_macro_epoch': val_at_best} if val_at_best else None)`.

#### D6. TB val per-class F1 scalars

Next to the existing `F1_train/<class>` loop in `train_network`:

```python
for i, c in enumerate(class_ls):
    writer.add_scalar(f'F1_train/{c}', train_per_class_f1[i].item(), epoch)
    if present[i]:
        writer.add_scalar(f'F1_val/{c}', f1_per_class[i].item(), epoch)
    if isinstance(loss_fn, AdaptiveFocalLoss):
        writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)
```

#### D7. Path construction updates (two callsites, not one)

Two places in `bst_train.py` use `taxonomy.name` to build on-disk paths. Both need the same patch: source the recorded string from `hyp.taxonomy`, not from the resolved Taxonomy object.

**Callsite 1: `bst_train.py:1168-1172`** (collated dir lookup):

```python
# Current:
collated_root = (
    Path(__file__).resolve().parent.parent
    / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
    / npy_collated_dir
)

# Patched:
recorded_name = hyp.taxonomy  # legacy name preserved verbatim from the Hyp
collated_root = (
    Path(__file__).resolve().parent.parent
    / f'preparing_data/ShuttleSet_data_{recorded_name}'
    / npy_collated_dir
)
```

**Callsite 2: `bst_train.py:803`** (weight filename construction, inside `Task.seek_network_weights`):

```python
# Current:
taxonomy_info = f'_{self.taxonomy.name}'

# Patched:
taxonomy_info = f'_{hyp.taxonomy}'  # legacy name preserved verbatim
```

Without the second patch, resuming `run_20260505_154907` would look for a weight file named `bst_CG_AP_..._une_v1_14.pt` (the resolved Taxonomy's name), but the on-disk file is `bst_CG_AP_..._une_merge_v1_nosides.pt` (the recorded string in the manifest). The legacy resume would fail to find the weight.

**Parallel construction in `eval_dump_predictions.py:175`** (if not deleted in Step D10): same patch applies, but this script is being retired so the patch may be moot.

For new runs `hyp.taxonomy == taxonomy.name` so both expressions evaluate to the same string. For legacy runs the alias path gives a different resolved name, and the recorded string carries the historical truth.

#### D8. CLI extension for the runner

Add to `bst_train.py` argparse:

```python
parser.add_argument('--taxonomy',     default=None)
parser.add_argument('--split-column', default=None)
parser.add_argument('--ablation-id',  default=None)
```

Applied via `hyp._replace`:

```python
overrides = {}
if args.taxonomy is not None:     overrides['taxonomy']     = args.taxonomy
if args.split_column is not None: overrides['split_column'] = args.split_column
if args.ablation_id is not None:  overrides['ablation_id']  = args.ablation_id
if overrides:
    hyp = hyp._replace(**overrides)
```

No `--drop-unknown` flag (controlled by taxonomy choice).

#### D9. `bst_infer.py` --fe mode

New mode for the FE/batch dump use case. Two new flags with the mutual-implication guard:

```python
parser.add_argument('--fe', action='store_true',
    help='FE/batch dump mode. Requires --fe-output-dir.')
parser.add_argument('--fe-output-dir', type=Path, default=None,
    help='Destination root for FE-mode npz dumps. Required when --fe is set.')

if args.fe and args.fe_output_dir is None:
    parser.error('--fe requires --fe-output-dir <path>')
if args.fe_output_dir is not None and not args.fe:
    parser.error('--fe-output-dir requires --fe (no implicit dump mode)')
```

Output path: `<fe_output_dir>/<run_id>/predictions/<split>_serial_<n>.npz`. Schema identical to what `bst_train.py` emits. Reuses `dump_topk_predictions` from `bst_common.py`.

#### D10. Retire `scratch/presentation_prep/eval_dump_predictions.py`

Move its capability into `bst_infer.py --fe`. Delete the old script with a one-line scratch note pointing at `bst_infer.py` as the replacement. `confusion_matrix.py` (the other consumer in `presentation_prep/`) needs updating to read npz instead of `.pt` (one-line change: `np.load(...)` instead of `torch.load(...)`).

### Step J. FE handler reconciliation (`src/api/registry.py` + `src/api/inference.py`)

Lettered out of alphabetical sequence because it was added late in planning. Executes between Step D and Step E in the order at the bottom of the doc.

Two FE-facing handlers read the BST manifest and the predictions JSONs today. Both depend on the bandaid we're removing. Three small patches make them work against the post-refactor schema without breaking the legacy mock JSONs already in flight.

Also incidentally fixes a latent BRIC-handler bug: today `registry.py` only looks in BST's bandaid block, so BRIC entries would register with `class_list = []`. The same patch lights up BRIC support.

Total surface: ~30 lines across two files. Lands alongside the rip; deliver to the FE team with the PR explanation in J4 below.

#### J1. Add a `_resolve_class_list` helper

Replace the hardcoded read at `src/api/registry.py:96` with a small resolver that tries the canonical field first, falling back to the legacy bandaid for old BST runs:

```python
def _resolve_class_list(manifest: dict) -> list[str]:
    """Source the class list from whichever field the manifest carries.

    Fallback order:
    1. config.classes — canonical (post-refactor BST + BRIC manifests).
    2. extra.arch.active_class_list — legacy BST runs (pre-refactor).
    Empty list when neither is present; the handler logs and surfaces 404.
    """
    cfg_classes = manifest.get('config', {}).get('classes')
    if isinstance(cfg_classes, list) and cfg_classes:
        return cfg_classes
    legacy = manifest.get('extra', {}).get('arch', {}).get('active_class_list')
    if isinstance(legacy, list) and legacy:
        return legacy
    return []
```

Then in the registry-entry builder:

```python
# Current (line 96, will be removed):
# class_list = manifest.get("extra", {}).get("arch", {}).get("active_class_list", [])

# Replacement:
class_list = _resolve_class_list(manifest)
```

#### J2. Predictions JSON field rename: `active_class_list` -> `class_list`

The post-hoc converter (parked for later) will emit `class_list` as the canonical per-clip JSON field, matching the registry-entry field name in `docs/api_contract.md`. The handler reads either name for back-compat with the existing mock JSONs during the transition:

```python
# Current (lines 204, 233, will be removed):
# class_list = preds.get("active_class_list", [])

# Replacement:
class_list = preds.get("class_list") or preds.get("active_class_list", [])
```

Two sites need updating (the `/clips` endpoint at line 204 and the `/clips/{stem}` endpoint at line 233).

#### J3. `src/api/inference.py` JSON field back-compat

Same predictions-JSON field-rename pattern as J2, applied to the second consumer that landed in the recent FE-integration push. The handler at `src/api/inference.py:34-44` currently reads `preds.get("active_class_list", [])` to extract the class list from the per-split predictions JSON. Same one-line back-compat patch as J2:

```python
def _load_test_preds(...) -> tuple[list[str], list[dict]]:
    """Return (class_list, list of test-split prediction records).

    Reads the canonical `class_list` field (post-refactor / api_contract-aligned)
    and falls back to the legacy `active_class_list` for the pre-refactor mock
    JSONs and any other predictions files still in the old shape. The fallback
    can be removed once all consumed predictions JSONs have been re-emitted by
    the post-hoc converter described in the refactor's open items.
    """
    ...
    # Was: return preds.get("active_class_list", []), preds.get("clips", [])
    class_list = preds.get("class_list") or preds.get("active_class_list", [])
    return class_list, preds.get("clips", [])
```

Inline-comment requirement (per the agreed convention): the comment must (a) name both field names so a future reader can grep either, (b) flag that the legacy name is back-compat-only, (c) name the eventual removal condition. Same comment shape applied at every other site that reads `preds.active_class_list` (currently only the one consumer at `inference.py:44`; the registry.py sites covered in J2 use the same comment template).

#### J4. PR-explanation for the FE team

Parked at `~/Desktop/bst_messaging_suggestions.md`. AI-flavoured first draft; rewrite in own voice before opening the PR. Headline points to carry across into the rewrite:

- What the patch does: small resolver in `src/api/registry.py` + `src/api/inference.py` that prefers `manifest.config.classes` and falls back to `manifest.extra.arch.active_class_list`. Same back-compat for the predictions-JSON `active_class_list -> class_list` rename.
- Side benefit: BRIC entries register correctly for the first time (the current handler only looks in BST's bandaid block).
- Test surface: the existing mock entry keeps working via fallback; new BST + BRIC entries work via canonical path.
- Forward concern (not in this PR): `src/api/bst_inference.py` hardcodes the 14-class list. Fine for the current best; will need parameterising or per-model cloning when new taxonomies register. FE-team workstream call.

### Step E. Runner script

New file: `src/bst_refactor/stroke_classification/main_on_shuttleset/collation_runner.py`. Thin loop over a session config, one cell per (taxonomy, split, ablation_id, n_serials). No kill rules, no verdict logic. State persisted to `state.json` for resume.

```python
"""Drive bst_train.py through a list of (taxonomy, split, knobs) cells.

No kill rules, no verdicts. Fresh run_id per cell, N serials per cell from
the cell config (default 5, headline cells set 10).
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / 'experiments'
TEST_LOGS_DIR = SCRIPT_DIR / 'test_logs'


def invoke_bst_train(*, serial_no: int, run_id: str, log_path: Path, cell: dict) -> int:
    src_root = SCRIPT_DIR.parent.parent
    stroke_root = SCRIPT_DIR.parent
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join([str(src_root), str(stroke_root)])
    cmd = [
        sys.executable, '-m', 'main_on_shuttleset.bst_train',
        '--serial-no',   str(serial_no),
        '--run-id',      run_id,
        '--log-path',    str(log_path),
        '--taxonomy',    cell['taxonomy'],
        '--split-column', cell['split_column'],
        '--ablation-id', cell['ablation_id'],
    ]
    return subprocess.run(cmd, env=env).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session_dir', type=Path)
    args = parser.parse_args()

    config = yaml.safe_load((args.session_dir / 'config.yaml').read_text())
    state_path = args.session_dir / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'cells': {}}

    for cell in config['cells']:
        name = cell['name']
        n_serials = cell.get('n_serials', 5)
        cstate = state['cells'].setdefault(
            name, {'run_id': None, 'log_path': None, 'serials_done': 0},
        )
        if cstate['run_id'] is None:
            ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            cstate['run_id']   = f'run_{ts}'
            cstate['log_path'] = str(TEST_LOGS_DIR / f'test_{ts}.log')
            state_path.write_text(json.dumps(state, indent=2))

        log_path = Path(cstate['log_path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        while cstate['serials_done'] < n_serials:
            nxt = cstate['serials_done'] + 1
            print(f'[{name}] launching S{nxt}/{n_serials} into {cstate["run_id"]}')
            rc = invoke_bst_train(
                serial_no=nxt, run_id=cstate['run_id'],
                log_path=log_path, cell=cell,
            )
            if rc != 0:
                print(f'[{name}] bst_train failed with code {rc} on S{nxt}; aborting cell.')
                sys.exit(rc)
            cstate['serials_done'] = nxt
            state_path.write_text(json.dumps(state, indent=2))

        print(f'[{name}] complete; run_id={cstate["run_id"]}')


if __name__ == '__main__':
    main()
```

Session config at `scratch/runners/taxon_pinned_w_preds/config.yaml`:

```yaml
session_name: taxon_pinned_w_preds_2026_05
cells:
  - name: shuttleset_18_v2_dropunk
    taxonomy: shuttleset_18
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 5

  - name: bst_25_v2_keepunk
    taxonomy: bst_25
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 5

  - name: bst_24_v2_dropunk
    taxonomy: bst_24
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 5

  - name: bst_12_v2_dropunk
    taxonomy: bst_12
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 5

  - name: bst_25_baseline_keepunk
    taxonomy: bst_25
    split_column: split_bst_baseline
    ablation_id: taxon_pinned_w_preds
    n_serials: 10

  - name: bst_24_baseline_dropunk
    taxonomy: bst_24
    split_column: split_bst_baseline
    ablation_id: taxon_pinned_w_preds
    n_serials: 10

  - name: une_v1_14_v2_dropunk
    taxonomy: une_v1_14
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 10

  - name: une_v1_15_v2_keepunk
    taxonomy: une_v1_15
    split_column: split_v2
    ablation_id: taxon_pinned_w_preds
    n_serials: 5
```

`bst_25` keepunk cells need the `--unknown-clip-npy-dir` flag piped through `invoke_bst_train`. Two ways:

1. Add an optional `cell['unknown_clip_npy_dir']` field, forwarded to bst_train as the flag.
2. Have bst_train auto-set the sibling dir whenever the active taxonomy has unknown in its class list.

Option 2 is cleaner since the sibling dir path is a project constant (`/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown`); add it as a module-level default in `bst_train` that the with-unknown taxonomies pick up automatically.

Wait, the sibling dir applies at collation time, not at train time. So this is irrelevant for the runner. The collator already has `--unknown-clip-npy-dir`; the runner doesn't need it. Collation happens upstream of the runner.

### Step F. Tests

#### F1. `tests/test_active_classes.py` (558 lines)

Major rewrite. Roughly 80% deletion plus a smaller core focused on the new contract:

- Delete: all `active_class_list` / `full_to_active_remap` / `derive_active_classes_from_labels` tests (sections 1-2, ~250 lines).
- Delete: `test_unknown_position_per_taxonomy` (line 180-188) - replaced by a single assertion that for every taxonomy in TAXONOMIES with has_unknown=True, classes[-1] == 'unknown'.
- Delete: `EXPECTED_N_ACTIVE_PER_DIR` (line 505) - the active/full distinction is gone.
- Delete: tests of `_validate_and_record_arch` (section 3) - the function is gone.
- Keep + adapt: real-collated-dir probe (section 6) - simplified to "labels.npy values are in [0, taxonomy.n_classes); clip_stems.npy exists and has same length".

New tests to add:

- `test_taxonomy_unknown_at_minus_one`: parametrize over `TAXONOMIES.values()`, assert `'unknown' not in t.classes or t.classes[-1] == 'unknown'`.
- `test_taxonomy_post_init_rejects_wrong_position`: construct a Taxonomy with `('a', 'unknown', 'b')` and assert it raises.
- `test_label_for_row_drives_taxonomy`: parametrize over `(taxonomy, raw_type, side) -> expected_index`. Covers driven_flight -> drive on bst_25 (the fix), unknown -> None on excluded_base_stroke_types taxonomies, etc.
- `test_resolve_taxonomy_aliases`: assert that each alias key returns the right object.
- `test_label_coverage_assert_failure_modes`: build a minimal Task-like environment with synthetic labels missing one class and assert the new assert fires with the expected message.

#### F2. `tests/test_data_access.py`

`test_derive_class_label_applies_merge_map` (line 152) uses `back_court_drive -> drive` on `TAXONOMY_UNE_MERGE_V1`. Add a parallel test for the cleaned `MERGE_MAP_25` on `TAXONOMY_BST_25`: `driven_flight -> drive` (the headline fix). Rename the imported taxonomy from `TAXONOMY_UNE_MERGE_V1` to `TAXONOMY_UNE_V1_15` (the with-unknown 14-class) or `TAXONOMY_UNE_V1_14` depending on which the test needs.

Also: `_derive_class_label` returning `'unknown'` for the unknown raw_type stays valid since `'unknown'` is in `SIDE_AGNOSTIC_TYPES`. Confirm the test at line 160 still passes.

#### F3. `tests/test_integration.py`, `tests/test_hparam_sweep.py`

Likely affected by the Hyp tuple changes (dropping `expected_active_classes`, `drop_unknown`). Audit and update field references. Hparam sweep tests probably need taxonomy / split / ablation_id flag updates if they assert on the bst_train CLI invocation shape.

#### F4. `tests/test_dataset.py`

Add `clip_stems.npy` to the fake-collated-dir fixture so the new Dataset_npy_collated load path works. One line per split.

### Step G. Docs to update

- `scratch/scratch_layout.md` - new collation dirs (`ShuttleSet_data_bst_25/`, `_bst_24/`, etc.), new sibling dir (`ShuttleSet_keypoints_clean_sticky_anchor_unknown/`), env var section unchanged.
- `scratch/architecture_notes/arch_1_directions.md` - active baseline pointer changes (old run_20260505_154907 stays valid via alias; new headline numbers will come from the cells in Step E).
- `scratch/frontend_integration_handoff.md` - `class_list` in the registry comes from `taxonomy.classes`, not from manifest.extra.arch; unknown index is always -1 when present.
- `docs/models_registry.yaml` - update existing entries to point at the new taxonomy names (or leave on old names + rely on alias lookup, depending on consumer migration timing).

### Step H. Execution order

1. Land Step A (config.py) + Step F1 + F2 + F3 + F4 (test updates) as one commit. Run pytest to confirm.
2. Land Step B1 (build_extract_stems --only-unknown). Run B2 + B3 on engelbart. Rsync to bourbaki (B4).
3. Land Step C1 + C2 + C3 (collator + dataset). Run pytest.
4. Land Step D1-D8 (train + infer surface, including assertions, predictions dump, val per-class capture, manifest config.classes write per D1.5). Run pytest.
5. Land Step D9 (bst_infer --fe). Land Step D10 (retire eval_dump_predictions).
6. Land Step J (`src/api/registry.py` resolver patch + `src/api/inference.py` field back-compat). Hand to FE team with the PR-explanation in J4 for review.
7. Land Step E (runner script + session config).
8. Run all eight cells via the runner. Manually prune non-best serials' predictions npz after the runner finishes (per-cell, retain `predictions/<split>_serial_<best>.npz` only).
9. Land Step G (docs) referencing the run IDs produced by Step 8.

## Migration / back-compat

### Alias table

The `TAXONOMY_ALIASES` table in config.py is the explicit inventory of legacy names. Each entry has a one-line comment naming the historical run(s) it serves. Phase-out is manual: when a historical run is retired (per the `feedback_phase_out_legacy.md` memory note, planned over the coming months), the corresponding alias entry gets removed and the paired `/scratch/comp320a/ShuttleSet_data_<old>/` dir is deleted manually.

### Old runs that resume

`run_20260505_154907` resumes against:
- Taxonomy lookup: `'une_merge_v1_nosides'` -> alias -> `TAXONOMY_UNE_V1_14`. Class list identical (the alias points at the structurally-equivalent taxonomy).
- On-disk dir: `ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop/` (untouched on /scratch). The path-construction patch in D7 uses `hyp.taxonomy` (the recorded string) for dir lookup.
- Weights: load against `n_classes = 14`, matching the current 14-output head.
- Manifest `extra.arch` block on the old manifest: tolerated. The new `bst_train.py` doesn't write or read that block; it ignores any existing one on resume.

Pre-refactor `merged_25` runs:
- Aliased to `bst_25`, but semantically loose because the old runs used the buggy `MERGE_MAP['driven_flight'] = 'unknown'`, so old labels.npy has driven_flight rows at index 0 (unknown) while the new bst_25 has them at the `drive` indices.
- Old weights load against `bst_25` shape-wise but produce wrong per-class interpretation.
- Acceptable per project decision; old `merged_25` runs are treated as historical, not actively re-evaluated.

### FE consumers

Manifest `extra.arch` block is no longer written for new runs. FE backend reads `class_list` from `manifest.config.classes` (mirrors BRIC's manifest schema; see Step D1.5). The `_resolve_class_list` helper in `src/api/registry.py` (Step J1) falls back to `extra.arch.active_class_list` for legacy BST runs.

Unknown index, when present in the taxonomy, is always `-1` (equivalently `n_classes - 1`). The `confidence_pct`/top_k computation in the FE consumer doesn't need to know about the unknown class explicitly.

For old runs the FE backend reads `extra.arch.active_class_list` via the fallback; the class names match what the post-refactor `taxonomy.classes` would return (modulo the historical merged_25 driven_flight semantic looseness flagged above).

### Cross-architecture comparison on the same taxonomy

Under the contractual taxonomy pattern, each (merge_map, has_sides, has_unknown) combo is its own Taxonomy with its own name string. The same trained-model class space can therefore appear under different taxonomy strings across architectures.

Example: BRIC's `une_collapsed_v1_nosides` taxonomy (in `src/shared/taxonomy.py`) defines 13 classes (12 stroke types + `unknown`). BRIC's `trainable_class_list()` filters out unknown at train time, so BRIC's trained model has 12 outputs. Class names line up bit-for-bit with my refactor's `bst_12` taxonomy (12 base classes, no sides, no unknown), since `UNE_COLLAPSED_V1_MAP` is structurally identical to `MERGE_MAP_25`. But the registry-side taxonomy strings differ: `"une_collapsed_v1_nosides"` vs `"bst_12"`.

The api_contract handles this cleanly. Each registry entry self-describes its `taxonomy` string and `class_list`. The FE picker groups by `architecture` ("bric" or "bst") at the top level, so users select within an architecture and don't get cross-arch confusion at selection time. The handler dispatches per-architecture; nothing tries to relate the two taxonomy strings automatically.

For cross-model panels that compare per-class F1 across architectures (e.g. "BRIC R(2+1)D-18 vs BST transformer on the same 12-class trainable space"), the join is by class NAME from each entry's `class_list`, not by taxonomy string and not by class index. Class names are stable across merge-equivalent taxonomies; class indices may not be (e.g. BST puts `unknown` at index -1 when present; BRIC's `une_collapsed_v1_nosides` also has unknown at the end of its 13-list).

When one entry has a class the other does not (e.g. BRIC's `une_collapsed_v1_nosides` exposes 13 classes in the taxonomy but the trained head emits only 12; BST's `bst_12` has 12 by construction), the FE either omits the asymmetric class from the comparison row or renders it as N/A. This is a UI display choice, not a backend contract issue.

Important: the `class_list` returned by `registry.py` for a given model entry comes from the trained model's output space (12 for both examples), not from the taxonomy's full class list. So the cross-arch join "just works" at class-name level for these merge-equivalent pairs.

## Touch surface inventory

### Source files changed

- `src/bst_refactor/pipeline/config.py` (substantial rewrite of taxonomy block)
- `src/bst_refactor/pipeline/data_access.py` (taxonomy name references; `_derive_class_label` keeps working unchanged)
- `src/bst_refactor/pipeline/build_dataset.py` (default taxonomy reference; remove `MERGE_MAP` re-export if unused after the dust settles)
- `src/bst_refactor/pipeline/clip_generator.py` (taxonomy default reference)
- `src/bst_refactor/pipeline/verify.py` (taxonomy default reference)
- `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py` (`collate_npy` rewrite, CLI changes)
- `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py` (Dataset_npy_collated loads clip_stems; adjust_to_partial_train_set mirrors clip_stems)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py` (Hyp tuple, train_network return value, dump_predictions, asserts, _validate_and_record_arch deletion, CLI flags)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_infer.py` (--fe + --fe-output-dir, batch dump path)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_common.py` (delete derive_active_classes_from_labels, add dump_topk_predictions)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/collation_runner.py` (new)
- `scripts/build_extract_stems.py` (`--only-unknown` flag)
- `scratch/presentation_prep/eval_dump_predictions.py` (deleted)
- `scratch/presentation_prep/confusion_matrix.py` (npz reader instead of .pt)
- `src/api/registry.py` (`_resolve_class_list` helper + JSON field rename with back-compat; ~30 lines per Step J)
- `src/api/inference.py` (predictions-JSON field back-compat at `_load_test_preds`, one line + comment per Step J3)
- `src/bst_refactor/run_tracker.py` *or* `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py` (one line landing `config.classes` into the manifest payload per Step D1.5; recommended in bst_train at the `track_run` call site)
- `scratch/post_tidy_smoke/smoke_infer_bit_exact.py` (currently uses `taxonomy.class_list()` and `n_active_classes` kwarg at lines ~117-118; both go away under the rip). Two options: (a) delete the smoke script since the legacy-fallback path it tests is being removed; (b) update it to use `taxonomy.classes` and pass `n_class=taxonomy.n_classes` to the network builder. (a) is cleaner unless the smoke is actively used in CI.
- `scratch/post_tidy_smoke/smoke_prepare_2d_bit_exact.py` (verify no taxonomy.name / class_list dependencies; touch only if grep finds them).

### Tests changed

- `tests/test_active_classes.py` (major rewrite)
- `tests/test_data_access.py` (taxonomy imports, new fix coverage)
- `tests/test_dataset.py` (clip_stems fixture)
- `tests/test_integration.py` (Hyp tuple field updates)
- `tests/test_hparam_sweep.py` (CLI invocation shape if asserted)

### Docs changed

- `scratch/scratch_layout.md`
- `scratch/architecture_notes/arch_1_directions.md`
- `scratch/frontend_integration_handoff.md`
- `docs/models_registry.yaml`

### On-disk artefacts produced

- `/scratch/comp320a/ShuttleSet_keypoints_raw_unknown/` (1,278 stems × 5 raw npy = ~6,400 files)
- `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor_unknown/` (1,278 stems × 3 clean npy)
- `/scratch/comp320a/ShuttleSet_data_bst_25/npy_taxon_pinned_w_preds/{train,val,test}/`
- `/scratch/comp320a/ShuttleSet_data_bst_24/npy_taxon_pinned_w_preds/{train,val,test}/`
- `/scratch/comp320a/ShuttleSet_data_bst_12/npy_taxon_pinned_w_preds/{train,val,test}/`
- `/scratch/comp320a/ShuttleSet_data_shuttleset_18/npy_taxon_pinned_w_preds/{train,val,test}/`
- `/scratch/comp320a/ShuttleSet_data_une_v1_14/npy_taxon_pinned_w_preds/{train,val,test}/`
- `/scratch/comp320a/ShuttleSet_data_une_v1_15/npy_taxon_pinned_w_preds/{train,val,test}/`
- Plus `clip_stems.npy` alongside `labels.npy` in every collated `{split}/` dir.
- Per cell: `experiments/run_<ts>/predictions/<split>_serial_<n>.npz` for serials 1..N, with non-best pruned manually after the runner finishes.

### On-disk artefacts unchanged

- All existing `/scratch/comp320a/ShuttleSet_keypoints_raw/` files.
- All existing `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/` files.
- All existing `ShuttleSet_data_merged_25/`, `ShuttleSet_data_une_merge_v1/`, `ShuttleSet_data_une_merge_v1_nosides/` trees (legacy collations, kept for resume).
- All existing `experiments/run_*/` dirs (legacy manifests, weights, predictions, best_model_id.txt notes). `best_model_id.txt` files are human-authored prose and won't be auto-migrated; new runs produce new `best_model_id.txt` files in their own dirs.

### Files audited and confirmed clean

These files were checked during planning and don't touch the bandaid surfaces, so they need no changes under this refactor. Listed for the cold-context reader so the audit doesn't need re-running:

- `src/bst_refactor/validation_scripts/*.py` (no MERGE_MAP / Taxonomy / active_class references; perclass diagnostics use raw labels.npy values which become directly meaningful under the rip)
- `src/bst_refactor/run_tracker.py` (manifest writer; agnostic to the arch block contents, just stores whatever extras the caller passes)
- `src/bst_refactor/aim_backfill.py` (aim mirror; doesn't depend on `extra.arch`)
- `src/bst_refactor/pipeline/clip_index.py` (path indexer; no taxonomy dependency)
- `src/bst_refactor/stroke_classification/main_on_shuttleset/hparam_sweep.py` (kill/verdict logic operates on metrics; needs minor audit for CLI flag shape compatibility — see Step F3 — but the core loop is unaffected)
- `src/bst_refactor/stroke_classification/loss/adaptive_focal.py` (consumes `n_classes` + `class_names` which `bst_train` continues to pass; no change needed)

## Drift-detection rules

If a future-me opens this doc and the repo state doesn't match the pre-flight checks, the answer to "can I still execute the plan?" depends on what drifted:

- **MERGE_MAP already fixed**: someone landed the driven_flight -> drive fix. Skip Step A's MERGE_MAP rewrite. Audit whether they also did the rest of the refactor; if so, this doc is partially obsolete.
- **derive_active_classes already removed**: someone got there first. Skip Step D2's adapter deletion; audit whether the assert pair replaced it as planned.
- **Unknown sibling extract dir already exists with content**: someone ran Step B already. Audit completeness (1,278 stems each with 5 raw + 3 clean files) before skipping the re-extract.
- **A new file pattern under `experiments/`** that this doc doesn't recognise: audit whether a parallel refactor is in flight. Don't proceed without reconciling.
- **`ablation_id` field has been renamed or `derive_ablation_id` is gone**: confirm by grep; if the auto-derive was already removed, the CLI changes in Step C2 may be partially done. Audit before re-doing.

When in doubt, prefer running pre-flight check #1 (the MERGE_MAP grep) as the canonical "has the rip already started?" signal. If `MERGE_MAP['driven_flight'] == 'drive'`, assume the refactor is at least partially landed and audit before extending.

## Open items

### Pending before final lock

- Impact assessment against the Architecture 2 train path. Specifically: does its taxonomy assumption survive the renames? Does its label-space contract with the collator match the new in-active-space labels.npy?
- Impact assessment against the FE-integration backend. Specifically: registry reader migration to `resolve_taxonomy(name).classes` instead of `manifest.extra.arch.active_class_list`; npz schema consumer ready; alias-table tolerance.
- PR / branch strategy. Touches many files. Single PR vs split into A + (B+C) + (D+E+F+G) + (H run-only) is a reasonable break.

### Parked decisions noted for the writeup phase

- BST-paper-baseline footnote: explain that the historical `merged_25` runs used the 35-class merge convention applied to the 25-class taxonomy (driven_flight -> unknown). New `bst_25` numbers are paper-faithful and not directly comparable to the historical merged_25 numbers.
- `confidence_pct` calibration: post-hoc temperature scaling on logits. The npz dump preserves raw logits so the calibration fit can happen against any registered run without re-running inference.

## Commit message scaffold (for the rip commit)

Parked at `~/Desktop/bst_messaging_suggestions.md`. AI-flavoured first draft; rewrite in own voice when the rip lands.
