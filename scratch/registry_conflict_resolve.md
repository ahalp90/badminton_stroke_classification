# Registry: bst_x's merge and what's next

Where bst_x's taxon-pinned branch lands on main, and the registry work that
follows. The merge is done and tested; phases 1 to 3 are follow-ups for the team.

## TL;DR

- bst_x's taxon-pinned branch caught up to main. One file clashed,
  `src/api/registry.py`. It's resolved and tested, registry and clip API tests
  pass.
- `registry.py` is the seam: bst_x and bric both write to it. It turns each
  model's manifest into the JSON the web app reads, so it's the one file the
  model side and the serving side share.
- Still to do: register bst_x's six new cells, serve val metrics off the
  manifest, show provenance chips (collation + split for bst_x, variant for
  bric), and write the per-clip prediction sidecars.

## Layers, so we don't trip on BE/FE

- **ML**: the model side. Manifests, npz, training.
- **serving**: `registry.py` / FastAPI. Reads manifests, serves JSON.
- **app**: the React frontend.

"FE" here means serving + app together. `registry.py` is the ML-to-serving seam.

## The merge (done)

Main had run a long way ahead while bst_x was on the taxonomy rework: the
frontend's live predictions and the whole bric pipeline both landed. Only
`registry.py` clashed, because both sides had rewritten the same function
(`_summarise_model`) at once. Main split it by architecture; bst_x's branch
unified the class-list handling and added the collation tag.

Resolved by keeping both, not picking a side:

- Main's per-architecture split stays. bst_x reads its per-serial metrics from
  the manifest; bric reads its `eval/test_summary.json` sidecar.
- bst_x's `_resolve_class_list` takes the class list (reads `config.classes`,
  falls back to the legacy `active_class_list`). This also fixes a hole in
  main's version, which only read `active_class_list` and would have come back
  empty on bst_x's new `config.classes` manifests.
- `collation_id` rides through off the manifest.

Tested on the API tests that hit `registry.py`: the four class-list tests, both
status/metrics tests (bst_x 0.7479, bric 0.7305), and both clip tests all pass.
Two upload tests fail locally, but only on the `/app/uploads` Docker path that
doesn't exist outside the container; nothing to do with this.

One loose end: I took main's side on a fourth clash in `list_clips`, so
`_summaries_for` still reads the legacy `active_class_list` while `get_clip`
reads the new `class_list` with a fallback. Fine while the prediction JSONs are
the old mocks; `_summaries_for` needs the same fallback when the real sidecars
land (phase 3).

## What's next

### Phase 1: register bst_x's six cells

Once the cells finish, add six entries to `docs/models_registry.yaml`:
`shuttleset_18/v2`, `bst_24/v2`, `bst_12/v2`, `bst_25/baseline`,
`bst_24/baseline`, `une_v1_14/v2`. Each needs id, taxonomy, split_column,
drop_unknown, manifest_path, weights_path, and `architecture: bst-x`.

Three things to watch:

- Drop the `ablation_id` line from the YAML. It's read off the manifest now, and
  it was mislabelling a collation as an ablation anyway.
- This is when the rename goes live. The fresh manifests carry
  `config.collation_id` and a null `config.ablation_id`, so the app's current
  `ablation_id` read starts coming back blank. Phase 1 and the phase 2 app change
  land together, or the cards show null.
- `bst_24` shows up on both `split_v2` and `split_bst_baseline`. Those two match
  on everything but split, which is why split has to become a visible field.

### Phase 2: provenance + val metrics (serving + app)

**Provenance.** The app builds its card subtitle from `ablation_id`, which is
blank for bric (its real axis is `config.variant = rgb_shuttle`, and nothing
surfaces it, so bric's card reads "nosides ·" with a dangling gap). Fix: serving
emits a small per-model provenance list, and the app renders it blind:

```
"provenance": [ {"label": ..., "value": ...}, ... ]
# bst_x: [collation, split, ablation?]   (ablation dropped when null)
# bric:  [variant]
```

Composed per-architecture in serving, so a new model or axis is a serving-only
change, no app edit. split comes free (it's already in the response); collation
rides along even though it's constant today and lights up when collations
diverge.

**Val metrics off the manifest.** The app card reads test metrics only; the
registry already returns val metrics, the app just doesn't show them. And serving
currently reads val from a `val_metrics.json` sidecar, which is a second forward
pass. Skip that: bst_x's manifests carry val per serial under
`extra.val_at_best_macro_epoch`, with the four aggregates back-filled:

- macro and min are the mean and min of the per-class F1 already there.
- accuracy and top2 come straight off the val npz (`y_pred_top1` and
  `topk_idx`), no forward pass.

So serving reads val off the manifest like test, and the `val_metrics.json` path
plus `compute_val_metrics.py` get deleted rather than fed. The app adds a val
column next to test. Back-fill script is `src/bst_refactor/backfill_val_metrics.py`,
run per cell; verified exact, recomputing test off the npz reproduces the
manifest's recorded numbers to the last digit.

### Phase 3: per-clip prediction sidecars (bst_x)

bric already ships its own (`predictions/test.json` + `eval/test_summary.json`
under `runtime/deployed/bric/`). bst_x still owes `predictions/{split}.json` +
`clip_index.json` per run, built from the npz. Until they land, bst_x cards show
metrics + provenance but the clip endpoints 404. This is also when
`_summaries_for` needs the `class_list` fallback (the loose end above).

## Who does what

| Phase | Who | Blocks the merge? |
|---|---|---|
| merge resolution | ML; serving owner reviews `registry.py` | this is the merge, done |
| 1 register cells | ML | no |
| 2 provenance + val | ML + FE team (serving + app) | no |
| 3 sidecars | ML | no |

The merge is one PR, already resolved and tested. The contract changes (phases 1
and 2) are a second PR with the FE team, since the YAML repoint and the app
switch have to land together.

## Open questions

- **provenance list vs raw fields**: the list never needs an app change when a
  new axis or model lands; raw fields are simpler but less future-proof.
- **bric's variant**: read from `config.variant` (manifest) or the registry
  entry. The manifest is the truer source.
- **keep run_20260505**: keeping the old legacy run gives a real fixture for the
  fallback test; dropping it makes for a cleaner registry.
