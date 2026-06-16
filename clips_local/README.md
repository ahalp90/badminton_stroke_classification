# clips_local/ — local sample clips for the Model Results per-clip player

Drop short clip `.mp4` files here to make them play in the **Model Results →
per-clip browser** (the `ClipDetail` panel). This directory is **gitignored**
(except this README and `.gitkeep`) — **never commit clip files**.

## Naming

Files must be named by the **clip stem**, exactly as it appears in the clip
browser list:

```
clips_local/<clip_stem>.mp4
```

Examples (real **test**-split stems on the current model):

```
clips_local/24_3_8_2.mp4
clips_local/39_2_15_3.mp4
clips_local/24_1_31_2.mp4
```

The full list of stems for a split is whatever the per-clip browser shows on
the left, or the keys in:

```
experiments/bst_x/shuttleset/run_20260505_154907/clip_index.json
```

## How it's served

The backend endpoint `GET /api/clips/<stem>/video` resolves in this order:

1. `clips_local/<stem>.mp4` (this directory) — what you drop here.
2. The dataset tree under `BST_X_CLIPS_DIR` (UNE HPC / mounted box).

So you do **not** need `BST_X_CLIPS_DIR` set to play a local drop. Override the
directory with the `BST_X_LOCAL_CLIPS_DIR` env var if you want it elsewhere.

If no clip is found for a stem, the player shows a "Clip not available locally"
message — it does not crash.

## Note on predictions

These clips play next to the clip's **real ground-truth label**. The model's
**prediction** is shown as *pending (placeholder data)* until real per-clip
inference (a live BST-X forward pass on the deploy box) is served — the committed
prediction records are placeholders (prediction = ground truth at 100%).
