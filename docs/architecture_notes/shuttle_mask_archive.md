# Shuttle-mask archive (variant 2a, dropped)

Tested branch `shuttle/mask-wiring` (run_20260503_192718), didn't
beat the shuttle-unzeroing run (run_20260503_172922) it sat on top
of. Live data path keeps the unzeroing change; this doc captures
the mask plumbing for archival purposes. Parked alternatives are
in `frame_zeroing.md`.

The actual code is in git: branch `shuttle/mask-wiring` HEAD
`521e6962`, parented on shuttle-unzeroing commit `8d8632f`. Tag
`shuttle-mask-archive` at `521e6962` before deleting the branch
so the SHA stays browsable.

## Result, in two lines

Mean macro / min / acc / top-2: 0.7440 / 0.4568 / 0.7630 / 0.9365.
Vs the shuttle-unzeroing run: macro -0.4, min -1.7, acc -0.2,
top-2 +0.1. Smash flat (+0.1%), ws -1.7%, rush -2.7%; clear
+0.6%. The mask channel didn't earn its keep on the existing
path. Two compounding causes look most likely: model was already
inferring missing-shuttle from xy + temporal context (so the mask
is mostly redundant), and the new fuse layer ate some learning
budget for a near-identity solution on the original 100 dims.

## What changed, by file

Five files touched, ~95 net insertions. All edits are scoped to
the shuttle path; pose path and the rest of the model are
untouched.

### `preparing_data/prepare_train_on_shuttleset.py`

- `get_shuttle_result`: preserves `Visibility` instead of
  dropping it. Returns `(shuttle_xy, shuttle_missing)` where
  `shuttle_missing = (df["Visibility"].to_numpy() == 0)`. (t,)
  bool.
- `pad_and_augment_one_npy_video`: accepts `shuttle_missing`,
  passes through `make_seq_len_same`, returns it alongside the
  pose / pos / shuttle outputs. Mask stays bool; the model casts
  on demand inside `forward`.
- `collate_npy`: collects a `shuttle_missing_ls` alongside
  `shuttle_ls`, threads through the `ProcessPoolExecutor` task
  submit / result loop, stacks to `(n, t)` bool, saves
  `shuttle_missing.npy` next to `shuttle.npy` in each split dir.

### `preparing_data/shuttleset_dataset.py`

- `make_seq_len_same`: accepts `shuttle_missing` as a fifth
  argument, strides + pads in lockstep with shuttle xy. Pad
  frames carry `True` (`np.pad(..., constant_values=True)`),
  mirroring the all-zeros shuttle xy on padded frames. Mask is
  *not* routed through `create_bones` or `interpolate_joints`;
  it's a per-frame scalar with no joint structure.
- `Dataset_npy_collated`: loads `shuttle_missing.npy` in
  `__init__`, drops alongside the other arrays in the
  zero-length-clip valid filter, plumbs through
  `adjust_to_partial_train_set`, returns as the fourth tensor
  in the input tuple from `__getitem__`. Tuple shape now
  `(human_pose, pos, shuttle, shuttle_missing), video_len, label`.

### `model/bst.py`

- `__init__`: two new layers, both Xavier-init via the existing
  recursive pass (no equivalence init):
  ```python
  d_mask = 4
  self.mask_proj = nn.Linear(1, d_mask)              # 8 params
  self.shuttle_fuse = nn.Linear(d_model + d_mask, d_model)  # 10.5K params
  ```
- `forward`: takes `shuttle_missing: Tensor` (b, t) bool as a
  new positional arg between `shuttle` and `pos`. After the
  existing `tcn_shuttle` produces `(b, 1, t, d_model)`, fuses
  per-frame:
  ```python
  mask_features = self.mask_proj(
      shuttle_missing.float().unsqueeze(-1)
  ).unsqueeze(1)                                     # (b, 1, t, d_mask)
  shuttle = self.shuttle_fuse(
      torch.cat([shuttle, mask_features], dim=-1)
  )                                                  # (b, 1, t, d_model)
  ```
- `__main__`: synthetic mask input (`shuttle_missing[:, ::20] =
  True`), wired into `input_data` so the smoke test exercises
  the new path on all 5 BST variants.

### `main_on_shuttleset/bst_x_train.py` and `bst_x_infer.py`

- Four `for (human_pose, pos, shuttle), video_len, labels in
  loader:` unpacks in `bst_x_train.py` (train_one_epoch / validate
  / test / test_topk) and one in `bst_x_infer.py` updated to the
  4-tuple form.
- `shuttle_missing.to(device)` added in each loop body alongside
  the other tensors.
- All five `model(human_pose, shuttle, pos, video_len)` call
  sites updated to `model(human_pose, shuttle, shuttle_missing,
  pos, video_len)`.

## Rationale, design call by design call

**Mask source = TrackNet `Visibility=0`.** The pre-existing flag
was being dropped at line 489. Visibility is the most
authoritative source of "no detection" (the model that emitted
the (0, 0) coords also said visibility=0 on the same frame).
Inferring missingness from coords would require an arbitrary
threshold on top-left clustering.

**Per-frame, not per-clip.** Variant 2 in `frame_zeroing.md`
considered three granularities (per-slot pose missing, per-frame
shuttle missing, single global OR mask). Per-frame on shuttle
alone (variant 2a) had the best signal-density read at 6.34%
positive rate; per-slot pose was too thin (~0.5%) and global OR
collapsed two failure modes with different downstream meanings.

**`d_mask = 4`.** Lower (1, 2) leaves the seed lottery to
determine whether the single mask weight survives Xavier init
into the fuse layer. Higher (8, 16) is wasteful for a one-bit
input. 4 is a cheap redundancy buffer.

**No activation between `mask_proj` and the concat.** ReLU kills
the mask=0 gradient flow (one of the two states the network sees
most). GELU partly squashes the redundancy. There's no
expressivity gain on a one-bit input that a linear projection
can't already provide.

**Fuse post-TCN, not pre-TCN.** TCN is a temporal convolution
over the shuttle xy. Routing the mask through TCN would mean the
mask gets smeared across the receptive field (17 frames), which
isn't what a per-frame missingness flag means. Fusing after
keeps the mask per-frame; cross-frame integration of mask info
falls to the transformer encoder downstream.

**Standard Xavier init, no equivalence init.** Earlier draft
proposed initialising `shuttle_fuse.weight[:, d_model:]` to zero
so the initial forward pass is mask-blind, isolating the mask's
contribution at training start. Decided overkill for a student
project; standard recursive Xavier on both new layers.

**Pad-frames carry `True`.** Pad shuttle xy is (0, 0) and pad
mask is True; both signals say "no data" on padded frames. This
mirrors the existing semantics rather than introducing a new
state.

**Mask isolated from bones augmentation.** `create_bones` and
`interpolate_joints` operate on (t, m, J, 2) joint tensors.
Mask is (t,) scalar; routing it through those would shape-fail.
More importantly, the mask doesn't have joint structure to
preserve.

**Bool dtype on disk.** Smaller than int8 (1/8 the bytes after
numpy's bool packing). Cast to float happens once per forward
pass inside the model.

## Failsafe gate (why we trust the comparison)

Re-collation produced byte-identical `pose / pos / shuttle /
videos_len / labels.npy` between the shuttle-unzeroing dir
(`npy_wipe_drop`) and the shuttle-mask dir (`npy_mask_wiring`)
across train / val / test (md5sum verified during the run). Only
`shuttle_missing.npy` differs (lives only on `npy_mask_wiring`).
So any train-time difference is the mask channel + the new fuse
layer, not data drift.

## Why it didn't lift, in one sentence

Most likely the model was already inferring missing-shuttle from
shuttle xy + temporal context (TCN receptive field 17 frames
plus transformer over 100), so an explicit mask is mostly
redundant; meanwhile the new `shuttle_fuse` (104 → 100) sits in
the existing shuttle path and has to learn a near-identity
solution on the original 100 dims, which costs a small amount
of optimisation budget for no offsetting signal.

## Pointers

Branch tag for the actual code: `shuttle-mask-archive` at
`521e6962`. Branch ref `shuttle/mask-wiring` may be deleted
after the tag is in place. Parked options that the unzeroing-only
result didn't close out (variant 2b pose_missing, trajectory
extrapolation) live in `frame_zeroing.md`.
