# Comment-density review

> _Last verified: 2026-06-29 against pre-pass `main`. This is the
> comment-clutter pass that ran alongside the structural simplification review
> (`simplification_review.md`) and drove the simplification pass (merged at
> `18e5c2c` on 2026-06-30). Findings below are the durable record; the line
> refs were pre-pass and have shifted._

A separate pass from the simplification review, focused only on making the code
easier to read by cutting comment clutter. The brief: drop the
TensorFlow-analogue commentary, but keep the codebase readable for someone with
no PyTorch experience.

## The rule applied

Straight from the project's commenting style: the name carries the *what*; a
comment earns its place when it adds the *why*, or a shape / invariant / domain
anchor the code can't show.

**Cut:**

- TensorFlow-analogue comments (`nn.Module = tf.keras.Model`, `= tf.transpose`,
  `like optimizer.apply_gradients()`). The big one, everywhere in the model
  and train files.
- Comments that restate a well-named line (`# Group clip counts by split`
  above the loop that does exactly that).
- Operator-level notes on well-known ops (`# /= w`, `# bitwise AND`,
  `@ = matrix multiply`).
- Decorative `# -----` banners wrapping a single function whose own docstring
  already names it.
- Provenance comments citing another file's line numbers (they rot; cite the
  function name instead).

**Kept:**

- Why / rationale (the warm-start schedule, the renorm-to-mean-1.0, the
  zero-length-clip divergence).
- Tensor shape annotations and axis names. Bulk of the value in the model file,
  every line.
- Invariant flags ("all per-candidate arrays share the same index space from
  here").
- Domain grounding (court / shuttle / TOP-BOT slot / Voronoi /
  11-point-switch).
- A short plain-English note on a genuinely PyTorch-specific idiom a newcomer
  wouldn't know (`.contiguous()`, `register_buffer`, `.view` vs `.reshape`,
  `pin_memory`). Phrased as what it does, not "the TF equivalent is X".

## The headline: `bst.py`

`model/bst.py` was 31% comment lines (151 of 486), most of it the
TF-for-TF-users layer. The largest comment cut in the codebase, and a pure
comment edit. Done in the simplification pass: post-edit `bst.py` is ~12%
comment lines.

The pattern, before:

```
super().__init__()  # PyTorch equivalent of super().__init__() in tf.keras.Model
...
self.attend = nn.Sequential(  # nn.Sequential = tf.keras.Sequential
...
# .view() = .reshape() but requires contiguous memory (faster, no copy)
# .transpose(1,2) = swap dims 1 and 2, like tf.transpose(perm=[0,2,1,3])
q = q.view(b, t, self.h, -1).transpose(1, 2)
...
dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
# @ = matrix multiply (same as tf.matmul or np.matmul)
```

After:

```
super().__init__()
...
self.attend = nn.Sequential(
...
# Split into heads: (b, t, d_cat) -> (b, h, t, d_head).
# .view (not .reshape): no-copy reshape that needs contiguous memory;
# .reshape is the more common default but copies when it can't view.
q = q.view(b, t, self.h, -1).transpose(1, 2)
...
dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
# dots: (b, h, t, t) - attention score for every (query_pos, key_pos) pair
```

The shape comment and the `.view` vs `.reshape` rationale stay (both useful);
the `@`-is-matmul and `transpose`-is-`tf.transpose` lessons go. The whole
`# PyTorch notes for TensorFlow users:` header block was replaced with a tight
idiom glossary that has no TF in it:

```
# A few PyTorch idioms used below, for readers coming from another framework:
#   nn.Parameter      a tensor the optimiser trains (a learnable weight)
#   register_buffer   a tensor that is NOT trained but still moves with
#                     .to(device) and is saved in the checkpoint
#   .contiguous()     repack a tensor into contiguous memory after a
#                     transpose/permute; required before .view()
#   forward()         runs on each model(x) call
```

What stayed untouched in the file: every `# x: (b, n+1, t, d_model)` shape
line, the `CrossTransformerLayer` residual NOTE (a real "this is unusual,
here's why"), the CG / AP warm-start rationale, the buffer-vs-parameter note,
and the CLS-token explanation.

## Per-module findings

Grouped by how much there was to do. All items below were applied unless
flagged otherwise.

**Heavy TF-analogue / restatement (clear wins):**

- **`bst.py`**: the headline above.
- **`bst_x_train.py`**: same TF layer as the model. Cut `# TF: model.compile...`
  (12-15), `# nn = layers..., optim = optimizers (like tf.keras.optimizers)`
  (19), `# like tape.gradient()` (276),
  `# like optimizer.apply_gradients()` (277),
  `# (like model.get_weights() in TF)` (754),
  `# like model.save_weights() / ...` (800-801),
  `# (TF: training=True/False)` (238, 305). Kept one plain line each for the
  genuinely PyTorch-specific moves: `model.train()` / `eval()` toggling
  dropout + batchnorm, the `zero_grad(); loss.backward(); step()` gradient
  step, `.item()` unwrapping a scalar tensor. The "flatten last two dims into
  one feature dim" comment was good but repeated at 269 / 322 / 392 / 419 and
  in infer:61; kept one canonical statement, bare `.view(...)` at the repeats.
- **`bst_x_infer.py`**: line 22 already says "See bst_x_train.py for the
  PyTorch / TF comparison comments", so the inline newcomer notes at 46, 52,
  64, 69 were redundant with the canonical copy. Cut to bare code. Kept the
  hard-fail-on-None-sidecar block (250-256); that's real rationale.

**Banners and restatement (visual declutter):**

- **`clip_generator.py`**: seven `# -----` dividers titling single functions
  whose docstrings already named them (33-35, 78-80, 95-97, 129-131, 197-199,
  plus trim at 229 / 355). ~15 lines of pure decoration. Where a banner carried
  real upstream provenance (`adapted from gen_my_dataset.py`), moved that one
  line into the function docstring. Kept the time-unit annotations
  (`t = int(fps) // 2  # frames in 0.5 sec` at 107 / 116 / 122) and the
  `df.duplicated('rally', ...)` trick comment (54-61). Dropped "(vectorized)"
  (281) and "Execute each move in a flat loop" (340).
- **`build_dataset.py`**: six `# Step N: ...` comments that restated the
  `_step(N, 'title')` call right below them (183, 190, 202, 213, 223, 247).
  Cut; kept only the non-obvious parentheticals as notes on the guard they
  actually explain.
- **`shuttle_extractor.py`**: deleted the provenance banner with rotting line
  refs (`# Normalization (from prepare_train_on_shuttleset.py:150-159)`,
  35-37; the def is now elsewhere). Dropped the docstring provenance line
  (70-72). Cross-ref comments at 84-93 collapsed to one line. Kept the
  InpaintNet-occlusion note (104-105) and the pipe-deadlock note (219-221).
- **`verify.py`**: dropped `# Group clip counts by split` (142, restates the
  loop). Kept `# Scan once, reuse everywhere` (297; invariant flag) and the
  per-check rationale docstrings (a guard module should say what each check
  guards).
- **`download_videos.py`**: trimmed `# Check if already downloaded` (49) to
  just its useful parenthetical ("any extension, yt-dlp may choose .mkv");
  cut `# Find the downloaded file` (74). Kept the 30-min-timeout rationale.

**Over-long rationale blocks (kept the why, halved the prose):**

- **`prepare_train_on_shuttleset.py`**: `normalize_joints` docstring (167-173)
  spent 7 lines on "defaults preserved verbatim from BST"; one line does it.
  The per-call-inferencer WARNING (346-352) is genuinely useful (an MMPose-bug
  workaround) but dropped its stale line-number pointers. The shuttle-read
  preamble (871-882) halved: kept "MMPose and TrackNetV3 can disagree 1-2
  frames on the tail; truncate to the shorter", dropped the restatement. The
  repeated "Need at least 2 / exactly 2 players" notes (272-274, 287, 371,
  384) state the domain point once, not four times. The two content-free
  `get_H` / `get_corner_camera` docstrings ("Get from the pd object.", 54, 62)
  got a real one-liner. Dropped `# /= w` (94, operator-level on a known
  homography step).
- **`config.py`**: the EN / ZH "Chinese names are CSV-I/O only" point was
  stated in the top docstring, again in the banner (30-34), again at 35-36,
  again at 61-65. Kept once (it's a real invariant), cut the repeats. Kept
  the `STROKE_TYPES_<count>_<provenance>` naming-convention block (68-80) and
  all the merge-map / alias rationale (load-bearing).
- **`data_access.py`**: the biggest single comment cut. The module docstring's
  "CLI usage" + "Environment / .env" sections (~55 lines, 64-118) re-documented
  every flag that the argparse `help=` strings and the `DataPaths` docstring
  already covered. Cut to a few worked examples (one run, one TUI, one
  override) and let `--help` carry per-flag detail. Kept the on-disk layout
  diagram (3-16) and the taxonomy-class explanation (22-28; not available from
  `--help`).
- **`clip_index.py`**: the 18-line `ClipVideoDataset` usage sketch (35-53) was
  duplicated almost verbatim in `pipeline/README.md`. Trimmed to a one-line
  pointer; kept the rglob-cost / O(1)-lookup perf note and the layout
  paragraph.

**Light touch:**

- **`shuttleset_dataset.py`**: kept the DIVERGENCE-FROM-BST block (load-bearing
  why + TODO), lightly tightened. Switched the `Dataset_npy_collated`
  docstring from the NumPy-style `Parameters` block to the project's `:param:`
  style, and dropped the now-stale "no random translation here" note.
- **`augmentations.py`**: docstrings long but mostly genuine domain / why. The
  one spot to halve was the `ConstrainedJitter.__call__` sentinel-exclusion
  block (270-290): kept the "why exclude (0,0) frames from min / max"
  rationale, trimmed the restatement of what the `masked_fill` does.
- **`apply_heuristic.py` + `current.py` + `sticky_anchor.py`**: dropped the
  "local import keeps imports tidy" note (apply_heuristic:266) and collapsed
  the two lazy-import comments that duplicated their module docstrings
  (current.py:50-52, sticky_anchor.py:332-334) to one line each. Kept the
  byte-equivalence whys (current.py:70-72, 91-96), the broadcast-shape note
  (sticky_anchor:199-200), the per-candidate index-space invariant (191-194),
  and the `# Bool mask over k candidate bboxes` line at sticky_anchor:228.

## Files already at target density (left alone)

The reference for what "good" looks like here:

- **`bst_x_common.py`**: 6% comment lines, no TF analogues, comments are pure
  why + shape. This is the target.
- **`model/tempose.py`**: 9% comments. The `'''Same as X in TemPose.'''`
  provenance docstrings are useful lineage; the `# This shouldn't be inplace.`
  note (line 69) is a real flag. Nothing to cut.
- **`loss/adaptive_focal.py`**: long docstrings but every paragraph documents
  a non-obvious maths invariant (the renorm to mean 1.0, the one-sided revert
  that doesn't claw back budget, the pair-cap redistribution). That's the why
  the code can't reconstruct. Kept.
- **`court_utils.py`, `player_mapping.py`**: shape / coordinate-frame / domain
  annotations, already at density. Kept the module docstrings explaining the
  intentional BRIC mirror.
- **`raw_extract.py`**: tight; the `J = 17  # COCO keypoints from RTMPose-L`
  kind of line is exactly right.
- **`base.py`, `heuristics/__init__.py`**: minimal and correct.
