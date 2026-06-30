# `detect_players_2d` / `detect_players_3d`: function invariants

> _Last verified: 2026-06-29 against pre-pass `main`. Line refs are pre-pass; the
> simplification-pass commits between then and `18e5c2c` shifted line numbers. The
> invariants below are the durable content; re-verify line refs before relying on
> them._

Maps the invariants a naive shared-body extract would break, against the live
code, so any future refactor can lift the shared frame loop without changing
behaviour. Analysis only; no `src/` edit is implied here.

Target file: `src/bst_x/preparing_data/prepare_train_on_shuttleset.py`. All line
numbers are a 2026-06-29 read; locate by symbol, not by literal line range.

Headline for the reader in a hurry: the two functions do share a clean per-frame
skeleton and the `_order_two_on_court` extract is sound. The risk is not the
lift itself, it's the gate. The simplification-pass plan-of-record gated the
lift on a "raw -> clean extraction golden", but that golden runs a different file
(`heuristics/current.py`), not `detect_players_*`. The real 2D gate is
`smoke_prepare_2d_bit_exact.py` (mp4 + MMPose + GPU), and the 3D path has no
working gate at all. Sections 4 and 5 carry this; sections 1-3 are the mechanics.

## Contents

- [0. The targets and the planned extract](#0-the-targets-and-the-planned-extract)
- [1. Side-by-side: identical skeleton vs per-variant differences](#1-side-by-side-identical-skeleton-vs-per-variant-differences)
- [2. The proposed shared-body extract](#2-the-proposed-shared-body-extract)
- [3. Invariants a naive extract would break](#3-invariants-a-naive-extract-would-break)
- [4. Verification: gating the extract](#4-verification-gating-the-extract)
- [5. Open questions](#5-open-questions)

## 0. The targets and the planned extract

- `detect_players_2d` at `prepare_train_on_shuttleset.py:242-319`
- `detect_players_3d` at `prepare_train_on_shuttleset.py:322-404`

Callers and wiring:

- `_prepare_dataset_from_raw_video` (shared per-clip driver) at `:531-583`; calls
  `detect_fn(video_path=..., **detect_kwargs)` at `:566-569`, saves the three npys
  at `:571-573`, frees GPU at `:579-580`.
- `prepare_2d_dataset_npy_from_raw_video` at `:586-622`; builds
  `MMPoseInferencer("human")` at `:609`, threads `center_align` /
  `normalized_by_v_height` through `detect_kwargs` at `:615-621`.
- `prepare_3d_dataset_npy_from_raw_video` at `:625-655`; builds
  `MMPoseInferencer("human")` (the 2D model) at `:642`, with the 3D inferencer
  commented out at `:643`/`:651`; sets `detect_fn=detect_players_3d` at `:648`.
- `main()` wiring: `--use-3d-pose` arg at `:1031-1035`; `str_3d` path tag at
  `:1146`; `use_3d_pose` into `derive_npy_collated_dir_basename` at `:1175`; the
  Step 2 dispatch `if args.use_3d_pose: prepare_3d_...(...) else: prepare_2d_...(...)`
  at `:1240-1255`, with `joints_center_align=True` passed to the 2D path at `:1254`.

Shared per-frame helpers both functions lean on:

- `check_pos_in_court` at `prepare_train_on_shuttleset.py:207-239` (a private copy;
  the court-dedup batch re-points it at `pipeline.court_utils`, but the extract
  is independent of that).
- `normalize_joints` at `:155-192` (2D only).
- `to_court_coordinate` / `get_court_info` live in `pipeline/court_utils.py`
  (`:104-124` and `:82-101`); `check_pos_in_court` calls `to_court_coordinate`.

The extract pulls `_order_two_on_court(...)` (returning the ordered
`in_court_pid` or `None`) out of the shared per-frame skeleton between
`detect_players_2d` and `detect_players_3d`. Both functions then call the
helper. `--use-3d-pose` is wired (`prepare_train_on_shuttleset.py:1032`), so
neither function is dead even though 3D is dormant in production. This is the
repeated-body case: the two functions share `< 2`-people zeros-append,
`check_pos_in_court`, `!= 2` on-court branch, the Top-before-Bottom `np.flip`,
and the `np.stack` tail. They differ only in 2D vs 3D keypoint extraction +
array shape.

## 1. Side-by-side: identical skeleton vs per-variant differences

Walking both functions frame by frame. "Same" means byte-for-byte equivalent
control flow; "differs" calls out the per-variant lines.

| Stage | detect_players_2d | detect_players_3d | Same? |
|-------|-------------------|-------------------|-------|
| Signature | `:242-250` (`inferencer`, `J=17`, `normalized_by_v_height=False`, `center_align=False`) | `:322-329` (`inferencer_2d`, `J=17`; `inferencer_3d` param commented out at `:324`; no normalize flags) | differs |
| `vid` derivation | `:260` `int(video_path.name.split("_", 1)[0])` | `:339` identical | same |
| Init three lists | `:262-264` | `:341-343` | same |
| Generator setup | `:266` single `for ... in enumerate(inferencer(str(video_path), show=False))` | `:345-356` builds `gen_2d`, instantiates `inferencer_3d` per call at `:353`, builds `gen_3d`, then `for ... (result_2d, result_3d) in enumerate(zip(gen_2d, gen_3d))` | differs |
| Keypoint extract | `:267-270` one array `keypoints` -> `(m, J, 2)` | `:357-369` two arrays: `keypoints_2d` -> `(m, J, 2)`, `keypoints_3d` -> `(m, J, 3)` | differs |
| `< 2 detected` guard | `:275-279` on `len(keypoints)`; append `True`, pos `np.zeros((2,2))`, joints `np.zeros((2,J,2))` | `:372-376` on `len(keypoints_2d)`; same except joints `np.zeros((2,J,3))` | logic same, joints shape differs |
| `check_pos_in_court` + nonzero | `:281-285` on `keypoints` | `:378-382` on `keypoints_2d` | same (both use the 2D keypoints) |
| `!= 2 on court` guard | `:288-292`; append `True`, pos `(2,2)`, joints `(2,J,2)` | `:385-389`; same except joints `(2,J,3)` | logic same, joints shape differs |
| bbox extract | `:294-297` `bboxes` -> `(m, 4)` | none | 2D only |
| Top-before-Bottom flip | `:300-301` `if pos_normalized[pid[0],1] > pos_normalized[pid[1],1]: np.flip` | `:392-393` identical | same |
| Success append: failed | `:303` `failed_ls.append(False)` | `:395` identical | same |
| Success append: position | `:304` `players_positions.append(pos_normalized[in_court_pid])` | `:396` identical | same |
| Success append: joints | `:305-312` `normalize_joints(arr=keypoints[pid], bbox=bboxes[pid], v_height=..., center_align=...)` | `:397` `players_joints.append(keypoints_3d[in_court_pid])` (raw 3D keypoints) | differs |
| Tail stack + return | `:314-319` `np.stack` x2, `return failed_ls, players_positions, players_joints` | `:399-404` identical | same |

So the genuinely shared, extractable decision is the middle band: take the 2D
keypoints, decide whether the frame yields exactly two on-court players, and if so
return them ordered Top-before-Bottom. Everything that touches MMPose generators,
the second (3D) keypoint array, the failed-frame zero shapes, and the success-frame
joint payload is per-variant and stays put.

## 2. The proposed shared-body extract

`01` and `05` both name the same helper: `_order_two_on_court(...)` returning the
ordered `in_court_pid` or `None`, called by both functions. The natural contract,
read off the shared band above:

```
_order_two_on_court(keypoints_2d, vid, all_court_info, res_df)
  -> (in_court_pid, pos_normalized)   on success (exactly 2 on court)
  -> None                             on either failure path

  body (verbatim from the shared lines):
    if len(keypoints_2d) < 2:                       # 2D :275 / 3D :372
        return None
    in_court, pos_normalized = check_pos_in_court(  # 2D :281 / 3D :378
        keypoints_2d, vid, all_court_info, res_df)
    in_court_pid = np.nonzero(in_court)[0]          # 2D :285 / 3D :382
    if len(in_court_pid) != 2:                      # 2D :288 / 3D :385
        return None
    if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
        in_court_pid = np.flip(in_court_pid)        # 2D :300 / 3D :392
    return in_court_pid, pos_normalized
```

The helper must return both `in_court_pid` (the success path indexes
`keypoints[in_court_pid]` / `bboxes[in_court_pid]` / `keypoints_3d[in_court_pid]`)
and `pos_normalized` (the position append is `pos_normalized[in_court_pid]`). "The
ordered `in_court_pid` or `None`" in `01`/`05` is shorthand; `pos_normalized` has
to ride back too or the caller can't build the position row. Returning the full
`(m, 2)` `pos_normalized` plus the ordered pid (rather than the pre-indexed 2-row
slice) keeps the caller side identical to today.

What stays per-variant in each caller, post-extract:

- 2D `detect_players_2d`: the single-generator loop (`:266`), the one keypoint
  extract (`:267-270`), the bbox extract (`:294-297`), and the success joints
  payload `normalize_joints(...)` with its `v_height` / `center_align` threading
  (`:305-312`).
- 3D `detect_players_3d`: the per-call `inferencer_3d` instantiation plus its
  load-bearing comment (`:346-354`), the `zip(gen_2d, gen_3d)` loop (`:356`), the
  two keypoint extracts (`:357-369`), and the success joints payload
  `keypoints_3d[in_court_pid]` (`:397`).
- Both: the failed-frame zero-append (because the joints zero shape differs,
  `(2,J,2)` vs `(2,J,3)`; see section 3.1), the `vid` line, the init lists, the
  tail `np.stack`. The zero-append is the one place where "shared skeleton" tempts
  a wrong unification.

A clean shape for each caller's loop body becomes:

```
ordered = _order_two_on_court(<2D keypoints>, vid, all_court_info, res_df)
if ordered is None:
    failed_ls.append(True)
    players_positions.append(np.zeros((2, 2), dtype=float))
    players_joints.append(np.zeros((2, J, <2 for 2D | 3 for 3D>), dtype=float))
    continue
in_court_pid, pos_normalized = ordered
failed_ls.append(False)
players_positions.append(pos_normalized[in_court_pid])
players_joints.append(<2D: normalize_joints(...) | 3D: keypoints_3d[in_court_pid]>)
```

The helper does not own the failed-append or the success-joints payload. That is
deliberate: both are shape- or variant-specific, and folding them in is exactly the
class of mistake section 3 is about.

## 3. Invariants a naive extract would break

### 3.1 Failed-frame zero shapes and dtype differ by variant

The pos zeros are `np.zeros((2, 2), dtype=float)` in both (`:277`/`:290` 2D,
`:374`/`:387` 3D). The joints zeros are `np.zeros((2, J, 2))` in 2D
(`:278`/`:291`) but `np.zeros((2, J, 3))` in 3D (`:375`/`:388`). `dtype=float` is
float64. If the helper were given the job of appending the failed-frame zeros it
would need to know the variant's joint last-dim; cleaner to leave the append in the
caller (section 2). A naive "the helper handles failures" unification that hardcodes
`(2, J, 2)` silently corrupts every failed frame of the 3D output to the wrong last
dimension, which then explodes (or worse, broadcasts) at the `np.stack` tail.

### 3.2 The `< 2` short-circuit must precede check_pos_in_court

Both functions test `len(...) < 2` before calling `check_pos_in_court`
(`:275`->`:281`, `:372`->`:378`). This ordering is load-bearing, not cosmetic.
`check_pos_in_court` does `keypoints[:, -2:, :]` at `court_utils.py:159`
(equivalently the private copy at `prepare_train_on_shuttleset.py:217`); on a
0-detection frame the keypoints array is `np.array([])` with shape `(0,)`, which
cannot take three index axes and raises. The helper must keep the `< 2` return
ahead of the `check_pos_in_court` call. Collapsing the two guards or reordering them
turns a quiet "failed frame" into an IndexError on the first empty frame.

### 3.3 In-court ordering and the strict-`>` tie behaviour

The flip is `if pos_normalized[pid[0], 1] > pos_normalized[pid[1], 1]:
in_court_pid = np.flip(in_court_pid)` (`:300-301` / `:392-393`). Two things to
preserve exactly:

- It is a strict `>`. On a y-tie the flip does not fire, so the original
  `np.nonzero` order (ascending detection index) is kept. Re-implementing this as
  `np.argsort(pos_normalized[in_court_pid, 1])` changes tie handling and can reorder
  the two players; the strict-`>` swap must survive verbatim.
- `np.flip` on a 2-element array is a swap. It relies on `in_court_pid` having
  exactly two entries, which the `!= 2` guard at `:288`/`:385` guarantees. The
  helper must keep the guard upstream of the flip; a flip on a non-2 array would
  silently mis-order.

The same strict-`>` flip is mirrored in the raw-array replication at
`heuristics/current.py:87-88`, so any change to it would also desync that file's
documented equivalence (see section 4).

### 3.4 The per-call 3D inferencer must not be hoisted

`detect_players_3d` builds `inferencer_3d = MMPoseInferencer(pose3d="human3d")`
inside the function, per call, at `:353`, behind a load-bearing comment at
`:346-352`:

```
WARNING: intentionally instantiated per-call, NOT per-loop-iteration in the caller.
The original author found that passing inferencer_3d as a parameter (the way
inferencer_2d is passed) triggers an MMPose bug. The commented-out parameter
on line ~300 and the commented-out caller on line ~588 are evidence of this.
This DOES reload model weights from disk for every clip, which is slow.
If MMPose fixes the bug upstream, hoist this into prepare_3d_dataset_npy_from_raw_video
and pass it in like inferencer_2d to avoid the repeated load.
```

The commented-out param at `:324` and the commented-out caller wiring at `:643`
/ `:651` are the corroborating evidence. Any extract must leave this in place:
do not hoist the instantiation into the caller, the driver, or the helper, even
though it reads like an obvious "build it once" win. `_order_two_on_court`
operates on already-extracted keypoints, so it has no reason to touch the
inferencer; the trap is a tidy-while-passing edit by a reader in the
neighbourhood. (The line refs in the comment itself, "~300" and "~588", are
already stale against the current file; re-anchor by symbol, don't act on them.)

### 3.5 Generator zip pairing and length (3D only)

The 3D loop is `for frame_num, (result_2d, result_3d) in enumerate(zip(gen_2d,
gen_3d))` (`:356`). The pairing is load-bearing: the court decision runs on the 2D
result (`check_pos_in_court` needs `(m, J, 2)`), while the joint payload comes from
the 3D result (`keypoints_3d[in_court_pid]`, `:397`). They must be the same frame.
Two consequences for the extract:

- The helper must receive the 2D keypoints in both variants. Feeding it the 3D
  `(m, J, 3)` array would break `check_pos_in_court` (which slices `[:, -2:, :]` and
  projects 2D pixel coords). The helper's first argument is always the 2D keypoints.
- `zip` stops at the shorter generator. Both generators run on the same mp4, so they
  agree on frame count today, and there is an implicit assumption that the 2D and 3D
  results expose the same per-frame detection ordering (the `in_court_pid` computed
  from the 2D result indexes into `keypoints_3d`). The extract does not change this,
  but it must not disturb it: keep `keypoints_2d` driving the court logic and
  `keypoints_3d` indexed by the pid the helper returns.

### 3.6 The failed_ls bool dtype at save

`failed_ls` is a Python list of `True`/`False`. The driver saves it as
`np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))` at
`_prepare_dataset_from_raw_video:573`. The smoke test then asserts exact equality on
this array (`np.array_equal(new_failed, ref_failed)`,
`smoke_prepare_2d_bit_exact.py:84`), no tolerance. So the success/fail decision must
land as appended Python bools in the same order as today. The extract keeps the
appends in the caller (section 2), so `failed_ls` stays a list of bools and the
`dtype=bool` cast remains exact; the helper must not start returning, say, a
numpy bool or an int that changes the saved dtype.

### 3.7 2D-only joint normalisation must stay in the 2D caller

The 2D success path calls `normalize_joints(arr=keypoints[in_court_pid],
bbox=bboxes[in_court_pid], v_height=res_df.loc[vid, "height"] if
normalized_by_v_height else None, center_align=center_align)` (`:305-312`). The
`bbox` extract (`:294-297`) and these two flags exist only in 2D. They must not
migrate into `_order_two_on_court` (3D has no bbox and appends raw 3D keypoints).
Note the deployed extract was produced with `center_align=True`: `main()` passes
`joints_center_align=True` at `:1254`, overriding the function-level default of
`False` at `:249`; `normalize_joints`'s docstring records this at `:167-172`, and
`heuristics/current.py:91-102` hardcodes `center_align=True` to match. The
extract must not disturb the `center_align` / `normalized_by_v_height`
threading, or the smoke reference (which runs with `joints_center_align=True`,
`smoke_prepare_2d_bit_exact.py:141`) mismatches.

## 4. Verification: gating the extract

The clean code lift (sections 1-3) is the easy half. The hard half is choosing a
gate that actually re-runs the edited code, which rules out the laptop CPU venv
and rules out the apply_heuristic golden.

### 4.1 What detect_players_* actually consume

Both functions run live MMPose inference on an mp4, frame by frame:
`detect_players_2d` iterates `inferencer(str(video_path), show=False)` at
`:266`; `detect_players_3d` zips `inferencer_2d(...)` with a per-call
`MMPoseInferencer(pose3d="human3d")` at `:345-356`. They need the MMPose stack,
the `.mp4` clips, and a GPU. The laptop `badminton-cicd` venv is CPU-only, so
neither function runs there at all. Confirmed: no pytest test exercises
`detect_players_2d` / `_3d` or the prepare drivers (`test_integration.py` and
`test_sticky_anchor.py` reference `prepare_train_on_shuttleset` only in
docstrings / comments and load pre-collated npy via `Dataset_npy_collated`). The
456 green tests give the extract zero coverage.

### 4.2 The 2D gate is the smoke test, not the "extraction golden"

`src/bst_x/validation_scripts/post_tidy_smoke/smoke_prepare_2d_bit_exact.py`
runs the real 2D path. It calls `prepare_2d_dataset_npy_from_raw_video`
(`:126-128`, `:135-142`), which goes through `_prepare_dataset_from_raw_video`
-> `detect_players_2d` -> (post-extract) `_order_two_on_court`, and byte-compares
`_pos` / `_joints` / `_failed` against a reference extract: `_failed` exact
(`:84`), `_pos` / `_joints` to `ATOL_FLOAT` default `1e-5` (`:90`, float32 ->
float64 projection-chain non-associativity, `:31-33`). It needs `CLIPS_DIR` (the
mp4s), `REFERENCE_DIR` (the committed extract, "typically the committed
`BST_X_MMPOSE_NPY_DIR`", `:24-26`), and `SCRATCH_DIR`, and runs on engelbart
under `venv-mmpose` (`:35-50`). This is a real, working 2D gate that covers the
post-extract 2D shared helper. The simplification-pass plan did not originally
cite it; the verification leg recorded in the worklog uses it via the
dual-invocation pattern.

### 4.3 The "raw -> clean extraction golden" does not gate the extract

The original plan named "the extraction golden, 2D AND 3D ... (raw -> clean)"
as the gate. The boursync spec for that golden lists only the five
`{stem}_raw_*.npy` arrays plus `{stem}_ball.csv` and three CSVs; no mp4s. Those
raw arrays are read by `apply_heuristic._load_raw_clip`
(`apply_heuristic.py:113-121`, `RAW_SUFFIXES` at `:46-52`), which feeds
`heuristics/current.py:apply`. `detect_players_*` consume mp4 via MMPose
(section 4.1); they never read `{stem}_raw_*.npy`. The raw-array path is a
separate implementation: `current.py`'s own docstring says it "Replicates
`detect_players_2d` ... by starting from the raw MMPose arrays ... rather than
from a live MMPose run" (`current.py:1-9`), with its own inline frame loop at
`:64-104`.

So a raw -> clean golden runs `current.py`, not `detect_players_*`. The extract
edits `detect_players_*` and leaves `current.py` untouched (the helper lives in
`prepare_train`; `current.py` keeps its inline loop and does not import
`_order_two_on_court`). The golden would therefore pass unchanged regardless of
what the extract does: zero coverage of the edit. The equivalence between
`current.py` and `detect_players_2d` is enforced in the other direction
(`current.py` was written to match the committed extract, gated by
`failsafe_bst_mmpose_zeroing_check_equivalence.py`); it does not make running
`current.py` a test of `detect_players_2d`.

### 4.4 The 3D path has no gate at all, and is dormant in production

The deployed pipeline is 2D, not 3D:

- `Hyp` defaults `use_3d_pose=False` (`bst_x_train.py:101`), and the active training
  config carries `use_3d_pose=False`
  (`src/bst_x/data_pipeline_to_model_train.md:411`).
- `BST_X_MMPOSE_NPY_DIR`, the flat per-clip dir the pipeline reads, points at the 2D
  `dataset_npy_between_2_hits_with_max_limits_flat` (`.env.example:37`,
  `pipeline/data_access.py:108`). The 3D dir would carry a `_3d` infix
  (`prepare_train_on_shuttleset.py:1146`, `:1181-1184`). No `npy_3d_` / `dataset_3d`
  collated artifact appears under `experiments/` or `runtime/`.
- `raw_extract.py:327-331` states the 3D extraction path is "deliberately out of
  scope for this module's current phase", so there is no 3D raw extract; the
  raw-array fixture in 4.3 cannot even be built for 3D.

Two consequences:

- The HPC une_v1_14 / bst_25 bit-exact (the blocking gate in the
  simplification-pass plan) runs with `use_3d_pose=False`. It exercises
  `detect_players_2d` only and never enters `detect_players_3d`. It does not
  cover the 3D leg.
- There is no 3D smoke (no `smoke_prepare_3d_*`), no 3D reference extract, and
  no 3D pytest. `detect_players_3d` is ungated by every artifact named in the
  original plan.

"`--use-3d-pose` is wired, so the 3D function is live" is true at the
CLI-plumbing level (`:1031-1035` -> `:1240-1246` -> `:648` ->
`detect_players_3d`) but does not mean it is used: it is reachable, never run in
production.

### 4.5 Coverage summary

| Artifact | Runs detect_players_2d? | Runs detect_players_3d? | Env |
|----------|-------------------------|--------------------------|-----|
| pytest (456 green) | no | no | laptop CPU |
| raw -> clean "extraction golden" (boursync) | no (runs `current.py`) | no (no 3D raw extract) | laptop CPU |
| `smoke_prepare_2d_bit_exact.py` | yes (via the 2D wrapper) | no | HPC, venv-mmpose, mp4s + reference |
| HPC une_v1_14 / bst_25 bit-exact | yes (`use_3d_pose=False`) | no | HPC GPU |

Net: the 2D path has a real gate (`smoke_prepare_2d_bit_exact.py` + the HPC
bit-exact, both already 2D-covering). The 3D path has none.

### 4.6 What a 3D smoke would need

To gate the 3D leg the way the 2D leg is gated, a future implementer would
need, on the GPU box under `venv-mmpose`:

1. A 3D reference extract: run `prepare_3d_dataset_npy_from_raw_video` (or
   `main --use-3d-pose --skip-trajectory --skip-collate`) on a fixed handful of
   mp4 stems on `main`, capturing `{stem}_pos.npy` (`(F,2,2)`),
   `{stem}_joints.npy` (`(F,2,J,3)`), `{stem}_failed.npy` per clip. None of
   these exist today.
2. A 3D smoke mirroring `smoke_prepare_2d_bit_exact.py` but calling the 3D
   wrapper and comparing against that reference (joints last-dim 3, `_failed`
   exact, floats to atol).

Cost: the per-call `MMPoseInferencer(pose3d="human3d")` reload (`:353`, section
3.4) makes even a 5-clip 3D run slow, and it must run on GPU. There is no
quicker route, because the raw-array shortcut (4.3) does not touch
`detect_players_3d`.

## 5. Open questions

1. **Is 3D used anywhere?** Answer, from the investigation in 4.4: no. It is
   wired and reachable but dormant: `use_3d_pose=False` everywhere deployed, no
   3D collated artifact, no 3D raw extract. So a 3D regression from the extract
   would have no production-output impact today. The open call is whether that
   makes the 3D gap acceptable, or whether a dormant-but-shipped code path still
   warrants a gate before its shared body is rewritten.

2. **How is the 3D path to be gated, if at all?** Three options:
   (a) accept 3D as unverified-but-dormant and gate on the 2D smoke + 2D HPC
   bit-exact only, leaning on a careful review of the 3D caller against section
   3 (the 3D-specific risks are the `(2,J,3)` failed-zeros, the per-call
   inferencer, and the zip pairing, none of which the 2D gate touches);
   (b) build the 3D smoke in 4.6 (real coverage, but a GPU run and a new
   reference extract);
   (c) keep `_order_two_on_court` strictly shape-agnostic (it never sees the
   joint payload or the 3D array, only the 2D keypoints and court info) so the
   2D smoke transitively proves the shared helper's arithmetic, leaving only
   the three 3D-specific bits to review. Option (c) plus (a) is the cheapest
   defensible combination; (b) is the only one that actually executes the 3D
   code. The simplification-pass extract chose (c) + (a).

3. **The plan's mis-specified gate.** The simplification-pass plan named the
   raw -> clean extraction golden as the gate, but that golden runs
   `heuristics/current.py` (the apply_heuristic path), not `detect_players_*`,
   so it does not gate the extract. The working 2D gate is
   `smoke_prepare_2d_bit_exact.py` (mp4 + MMPose + GPU), which the plan didn't
   name. The court-dedup batch had the same issue: if it edited only the court
   block consumed by `detect_players_*` / `get_court_info`, the same "raw ->
   clean runs `current.py`" caveat applies, though that batch also changes
   `pipeline.court_utils`, which `apply_heuristic.py:35` and the smoke (`:125`)
   both import, so its coverage story differs.

4. **Could `current.py` share the helper too?** `current.py`'s inline loop
   (`:64-104`) is the same skeleton: `n = ndet[f]` -> `< 2` ->
   `check_pos_in_court` -> `nonzero` -> `!= 2` -> strict-`>` flip. If
   `_order_two_on_court` lived in `prepare_train` and `current.py` also called
   it (it already lazy-imports `check_pos_in_court` / `normalize_joints` from
   there at `:50-56`), then the raw -> clean golden would cover the shared
   helper via `current.py`, closing part of the 3D-adjacent gap on CPU. But
   that widens scope beyond "both `detect_players_*` call it", adds a
   cross-module import into a file deliberately kept MMPose-free
   (`current.py:30-33`), and would need its own equivalence re-check
   (`current.py` reads NaN-padded raw arrays sliced by `ndet`, a different
   input contract from the live MMPose result dicts). The simplification-pass
   extract did not pull this in.

## Adversarial review (round 1)

Reviewed 2026-06-29 against the live source. Default-distrust pass: every
file:line spot-checked, both functions read in full, the extract contract and
the gate claims tested independently.

### Verdict

**Fit to guide the mechanical extract.** Sections 0-3 are accurate and complete.
Every line ref in section 0 and the section-1 table checks out against the
2026-06-29 source (`detect_players_2d` 242-319, `detect_players_3d` 322-404, the
driver/wrappers/main wiring all correct). The "identical middle band" claim is
right: both functions call the same `check_pos_in_court(<2D keypoints>, vid,
all_court_info, res_df)` (`:281` / `:378`), the same `np.nonzero(in_court)[0]`
(`:285` / `:382`), the same `!= 2` guard (`:288` / `:385`), and the same
strict-`>` `np.flip` (`:300-301` / `:392-393`). The `_order_two_on_court`
contract (return `(in_court_pid, pos_normalized)` on success, `None` on both
failure paths) is faithful to BOTH call sites and would preserve byte-identical
behaviour: the only per-variant tail (2D bbox extract + `normalize_joints`, 3D
`keypoints_3d[in_court_pid]`) and the failed-frame zero-append are correctly left
in the callers. The two specifically-flagged traps are both present and correct
(the per-call `MMPoseInferencer(pose3d=...)` at `:353` in 3.4; the
`(2,2)`/`(2,J,2)`/`(2,J,3)` zero shapes in 3.1). No byte-identity break was found
that the doc glossed: dtype is `float` (float64) on both, the flip is strict `>`
on both, `np.flip` on both, and the `zip` pairing is correctly called out.

**But section 4's gate analysis carries one HIGH false-confidence error (F1).**
The doc correctly debunks the "raw -> clean golden gates the extract" claim,
then wrongly rehabilitates the HPC `une_v1_14` / `bst_25` bit-exact as a
2D-covering gate. That bit-exact re-runs collation + training on the SHARED,
already-extracted committed per-clip npy; it never re-runs `detect_players_2d`,
so it has the exact same vacuous-pass problem assigned to the raw -> clean
golden. The real coverage is thinner than the doc states, not richer:
`smoke_prepare_2d_bit_exact.py` is the ONLY artifact that re-runs
`detect_players_2d` at all, and it was not named in the original plan.
Correcting F1 strengthens the doc's own thesis (the extract is under-gated).
With F1 fixed, the doc is fit to guide.

### Findings

| ID | Severity | Issue | Evidence | Fix |
|----|----------|-------|----------|-----|
| F1 | HIGH | The HPC `une_v1_14` / `bst_25` bit-exact does NOT exercise `detect_players_2d`. The doc credits it as a working 2D gate (4.4 bullet 1 "It exercises detect_players_2d only"; 4.5 table row + "Net" line "both already 2D-covering"; Q2(a) "the 2D smoke + 2D HPC bit-exact"). But pose extraction (Step 2) writes a per-clip dir that is "taxonomy- and split-independent" and shared across ablations; taxonomy enters only at collation (`label_for_row`) + training. So "re-run the `une_v1_14` and `bst_25` paths" = re-run collation + training reading the committed extract, never re-running `detect_players_*`. Same vacuous-pass mode the doc assigns to the raw -> clean golden in 4.3. A regression in the extract would not be caught. | `prepare_train_on_shuttleset.py:1187-1189` ("the layout is taxonomy- and split-independent"), `:555-556`; `collate_npy` taxonomy / `label_for_row` at `:726`, `:753`, `:813`; simplification-pass planning doc on "re-run the une_v1_14 and bst_25 paths ... touches collation, training, or the model". | Strike the HPC bit-exact from the extract's covering gates (2D AND 3D). State that the SOLE artifact re-running `detect_players_2d` is `smoke_prepare_2d_bit_exact.py` (HPC / GPU / MMPose, un-named in the plan). Note the HPC bit-exact is vacuous for any extraction-stage edit, the court dedup included. |
| F2 | MEDIUM (incomplete) | 4.3 / Q3 cite the doc's own prose + the harness README but miss the strongest single piece of evidence: the simplification-pass runbook's own golden tables tag the extraction golden "raw -> clean", "needs boursync of `*_raw_*.npy`", env "**laptop CPU**". The "laptop CPU" tag alone disproves it can run `detect_players_*` (GPU + MMPose + mp4). The doc also never says the golden was currently UNBUILT (TODO), which further weakens "it gates the extract". | Simplification-pass runbook golden tables (extraction golden = raw -> clean, laptop CPU, gates the court-dedup batch + the extract); the harness README's TODO that the boursync mirror was empty. | Add citations to those golden tables (laptop-CPU tag = airtight disproof) and note the golden was unbuilt, so the mis-spec lived in the planning tables, not only this doc's prose. |
| F3 | LOW | Section 0's helper map says `to_court_coordinate` / `get_court_info` "live in `pipeline/court_utils.py` (`:104-124` and `:82-101`); `check_pos_in_court` calls `to_court_coordinate`", implying the live call resolves into `court_utils`. Pre-court-dedup it does not: the live `detect_players_*` use the in-module `check_pos_in_court` (`:207`), which calls the in-module `to_court_coordinate` (`:118-137`), not `court_utils`. The doc flags the private copy elsewhere, so this is muddle not error, and it does not affect the extract. | `prepare_train_on_shuttleset.py:207-239` (private `check_pos_in_court`) calls local `to_court_coordinate` at `:118-137`; `court_utils.py:82-101` / `:104-124` / `:159` exist but are reached by `apply_heuristic` / smoke, not by `detect_players_*` pre-court-dedup. | Clarify `_order_two_on_court` calls the in-module `check_pos_in_court` (which pre-court-dedup uses the in-module `to_court_coordinate`); the `court_utils` line refs are the court-dedup target, not the current call path. |
| F4 | LOW | `apply_heuristic._load_raw_clip` is `:113-121` (the doc says `:113-120`, dropping the closing-paren line). Trivial. | `apply_heuristic.py:113-121` | `:113-121`. |
| F5 | LOW | 4.1 says BOTH `test_integration.py` and `test_sticky_anchor.py` "load pre-collated npy via `Dataset_npy_collated`". Only `test_integration` does (`:53`, `:109`); `test_sticky_anchor` mocks `normalize_joints` (`:115`) and tests the heuristic on synthetic arrays. The load-bearing claim (no pytest runs `detect_players_2d/3d`) is correct. | `tests/test_integration.py:53`,`:109`,`:28` ("require HPC, real videos, MMPose"); `tests/test_sticky_anchor.py:115` ("No-op stand-in for ... normalize_joints") | Split the characterisation; the "no pytest exercises detect_players" conclusion stands. |

No missed invariant in 3.1-3.7: all seven are real and the set is complete for a
byte-identical extract. One point the doc already makes in section 2 is worth
elevating to invariant status if reworded: the helper must return the FULL
`(m, 2)` `pos_normalized` plus the ordered pid, never the pre-sliced 2-row array,
or the caller's `pos_normalized[in_court_pid]` double-indexes. The doc states
this correctly at `:135-141`; flagging only that it is as load-bearing as 3.1-3.7.

### Claim verdicts

**4a (the raw -> clean extraction golden does not gate the extract):
CONFIRMED.** `apply_heuristic._load_raw_clip` reads the five `*_raw_*.npy`
arrays (`apply_heuristic.py:46-52`, `:113-121`) and feeds
`heuristics/current.py:apply`, whose inline loop (`current.py:64-104`)
"Replicates `detect_players_2d` ... by starting from the raw MMPose arrays ...
rather than from a live MMPose run" (`current.py:1-9`). `detect_players_2d` /
`_3d` instead consume mp4 via `MMPoseInferencer`
(`prepare_train_on_shuttleset.py:266` for 2D, `:345`-`:354` for 3D) and never
read `*_raw_*.npy`. The extract edits `detect_players_*` and leaves
`current.py` untouched, so a raw -> clean golden passes regardless. The
harness README self-contradicts (gating bullet plus the apply_heuristic
note that the path differs from `prepare_train`'s `detect_players_*`).
Strengthening evidence the doc omits (F2): the simplification-pass runbook
tagged this golden "laptop CPU", which `detect_players_*` cannot run on.

**4b (3D is dormant in production): CONFIRMED.** `use_3d_pose=False` in the active
`hyp = Hyp(...)` config (`bst_x_train.py:101`, instance opened at `:85`) and in
the documented config (`data_pipeline_to_model_train.md:411`); `--use-3d-pose` is
`action="store_true"`, so it defaults False (`prepare_train_on_shuttleset.py:1031-1035`).
`BST_X_MMPOSE_NPY_DIR` points at the 2D `dataset_npy_between_2_hits_with_max_limits_flat`
(`.env.example:37`, `data_access.py:108`); the 3D dir would carry a `_3d` infix
(`:1146`, `:1181-1184`). `find experiments/ runtime/` returns no `*_3d*` /
`dataset_3d` / `npy_3d` artifact, and there is no `smoke_prepare_3d*`
(`post_tidy_smoke/` holds only `smoke_infer_bit_exact.py` +
`smoke_prepare_2d_bit_exact.py`). `raw_extract.py:327-331` declares the 3D
extraction path "deliberately out of scope". 3D is reachable via CLI plumbing
(`:1031-1035` -> `:1240-1246` -> `:648`) but never run.

**4c (`smoke_prepare_2d_bit_exact.py` is the real HPC 2D gate; nothing on
laptop CPU covers `detect_players_*`): CONFIRMED, and stronger than stated.**
The smoke calls `prepare_2d_dataset_npy_from_raw_video` (`smoke:126-128`,
`:135-142`) -> `_prepare_dataset_from_raw_video` -> `detect_players_2d` on mp4
via MMPose, on engelbart under `venv-mmpose` (`smoke:35-50`), comparing
`_failed` exact (`:84`) + floats to atol (`:90`). No pytest exercises
`detect_players_2d` / `_3d`: a repo-wide grep finds the symbol only in
`current.py` docstring / comments and the smoke docstring;
`test_integration.py:28` explicitly defers pose extraction to HPC and loads via
`Dataset_npy_collated`, and `test_sticky_anchor.py:115` mocks
`normalize_joints`. Laptop `badminton-cicd` is CPU-only. Per F1, the smoke is
not one of two HPC 2D gates: it is the ONLY artifact that re-runs
`detect_players_2d`, and the original plan did not name it.

---

_Originally written as the B5 split pre-analysis in the simplification pass
(merged at `18e5c2c`, 2026-06-30). The extract itself landed as commit
`41719e4`._

