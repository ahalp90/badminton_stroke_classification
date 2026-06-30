# `collate_npy`: function invariants

> _Last verified: 2026-06-29 against pre-pass `main`. Line refs are pre-pass; the
> simplification-pass commits between then and `18e5c2c` shifted line numbers. The
> invariants below are the durable content; re-verify line refs before relying on
> them._

Maps the invariants a naive split of `collate_npy` would silently break, against
the live code, so any future refactor can split it behaviour-preserving. Analysis
only: no `src/` edit is part of this doc.

Repo root: `/home/ariel/Documents/COSC594/badminton_stroke_classification`. All
`file:line` refs below are relative to it; the target is
`src/bst_x/preparing_data/prepare_train_on_shuttleset.py` (abbreviated
`prepare_train` below) with `collate_npy` defined at line 719.

Baseline was green on `main` at capture (2026-06-29): the collation golden
checked bit-identical, 5 clips, 9 output files. Shapes/dtypes in this doc are
read from the captured golden, not inferred.

## Contents

- [1. Current structure (ordered stages)](#1-current-structure-ordered-stages)
- [2. The proposed split](#2-the-proposed-split)
- [3. Invariants a naive split would break](#3-invariants-a-naive-split-would-break)
- [4. Verification checklist](#4-verification-checklist)
- [5. Open questions / risks](#5-open-questions--risks)

## 1. Current structure (ordered stages)

`collate_npy` runs 719-970. One call does one split (`set_name`). Stages in order:

```
S0  719-767   signature + docstring
S1  768-787   argument guards (set_name; shuttle_csv_dir; resolution_df;
              unknown_root_dir mutual-exclusion; has_unknown requires unknown_root_dir)
S2  789-796   read clips_csv, filter rows to this split (split_column == set_name)
S3  798-852   per-row label derivation + unknown routing + file-existence filter
              loop 807-836; missing warning 838-846; labels/clip_stems arrays + count 847-852
S4  854-869   threaded npy load -> joints_ls, pos_ls, failed_ls (ThreadPoolExecutor)
S5  871-907   per-clip shuttle read (get_shuttle_result) + temporal align (truncate to min_t)
S6  909-916   pose_styles validation (bad_styles raise) + bone_pairs = get_bone_pairs('coco')
S7  918-947   pad/augment per clip via ProcessPoolExecutor (pad_and_augment_one_npy_video)
              + collect results in submission order -> pose_ls, pos_ls, shuttle_ls, videos_len
S8  949-952   np.stack pos / shuttle / videos_len
S9  954-959   mkdir save_dir, save_dir/set_name
S10 961-969   save the requested pose styles + pos + shuttle + videos_len + labels + clip_stems
S11 970       done print
```

Helpers (out of split scope, but the contracts they impose are):
- `pad_and_augment_one_npy_video` (prepare_train:661-716): already a separate function.
  Casts joints/pos/shuttle to float32 (688-690), calls `make_seq_len_same`, then builds
  only the requested pose styles. The per-clip pad/augment maths is already factored out;
  S7 is only the ProcessPool orchestration around it.
- `make_seq_len_same` (`src/bst_x/preparing_data/shuttleset_dataset.py:50-83`): stride/pad
  to `seq_len`, returns `new_video_len = len(pos)` (real frame count, pre-pad).
- `create_bones` (shuttleset_dataset.py:86-96), `interpolate_joints` (99-110),
  `get_bone_pairs` (35-47): pure, 19 coco bone pairs.
- `get_shuttle_result` (prepare_train:490-497): reads `{stem}_ball.csv`, normalises by
  resolution, returns `(t, 2)` float64.
- `label_for_row` (`src/bst_x/pipeline/config.py:333-366`): the single per-row decision
  point (see INV-7).
- `VALID_POSE_STYLES` (prepare_train:658): `('J_only', 'JnB_interp', 'JnB_bone', 'Jn2B')`.

## 2. The split

The split runs along four concerns:

- CSV filter + label derivation
- threaded npy load
- shuttle read + temporal align
- pad / augment + stack + save

Mapped to the stages above:

```
concern 1  CSV filter + label derivation   = S2 + S3
concern 2  threaded npy load               = S4
concern 3  shuttle read + temporal align   = S5
concern 4  pad/augment + stack + save      = S6 + S7 + S8 + S9 + S10
```

The argument guards (S1) and the top-level orchestration stay in `collate_npy`.
The split is structural only: same call sequence, same data, same outputs.

## 3. Invariants a naive split would break

This is the core of the doc. Every invariant is grounded in the live code.

### INV-1 Canonical row order, established once in S3

The row order of every output is the filtered `clips_df` order: CSV row order (S2:790),
restricted to `split_column == set_name` (796, a boolean mask that preserves order),
minus rows dropped by S3. It is established ONCE, in the S3 loop (807-836). All nine
outputs index into the same `[0, n)` row space.

`labels_ls` and `stems_ls` are produced ONLY in S3 (835-836) and never recomputed.
`data_branches` (834) drives S4, S5, S7. A split that re-derives labels or stems in a
separate pass (e.g. a second filter over `clips_df`) instead of carrying the S3 lists
forward can desync labels/stems from the array data. The fix: concern 1 returns
`(data_branches, labels, clip_stems_arr)` as one triple built in one pass; concerns
2-4 consume `data_branches` and never re-filter.

### INV-2 The two row-drops and the three appends are one atomic block

In the S3 loop a row is dropped by either:
- `idx is None` -> `continue` (820-821): filtered out via `excluded_base_stroke_types`.
- missing `{branch}_pos.npy` -> `missing += 1; continue` (831-833).

A surviving row appends to all three lists together (834-836):
`data_branches`, `labels_ls`, `stems_ls`. The two `continue`s skip a row from all three
simultaneously. If a split computes labels in one loop and resolves files in another,
any divergence in which rows survive each loop corrupts alignment for every downstream
row. Keep the idx-None check, the file-existence check, and the three appends in one
loop body (or one function that returns the three lists).

### INV-3 `data_branches` is the join key; `clip_stems_arr` must equal its basenames

`branch = str(chosen_root / stem)` (827), so `Path(branch).name == stem`. `stems_ls`
appends `stem` (836), hence `clip_stems_arr` (848) equals
`[Path(b).name for b in data_branches]` row-for-row. S5 relies on this:
`clip_stem = Path(branch).name` (885) is how it keys the shuttle CSV. If a split
recomputes stems from `data_branches` that is fine; recomputing them from a re-read CSV
is the trap.

### INV-4 Executor results MUST be collected in submission order

Both executors collect by iterating the task list in submission order, which is the row
order:
- S4 ThreadPool: `joints_ls = [t1.result() for t1 in tasks1]` etc. (866-868).
- S7 ProcessPool: `for task in tasks: ... task.result()` (941-947).

This is THE determinism guarantee for row alignment. Switching either to
`concurrent.futures.as_completed` permutes rows relative to `labels`/`clip_stems` and
silently corrupts every output (the golden would catch it, but only if run; do not
introduce it). The split must preserve submission-order collection in both stages.

### INV-5 Shuttle/pose truncation must propagate to joints AND pos

S5 aligns frame counts per clip (895-900):

```
min_t = min(len(failed), len(shuttle))
if min_t < len(failed) or min_t < len(shuttle):
    joints_ls[i] = joints_ls[i][:min_t]
    pos_ls[i]    = pos_ls[i][:min_t]
    shuttle      = shuttle[:min_t]
    failed       = failed[:min_t]
```

It truncates `joints_ls[i]` and `pos_ls[i]` IN PLACE in the lists (897-898). S7 then
consumes the truncated lists via `zip(joints_ls, pos_ls, shuttle_ls)` (923). If concern
3 is extracted into a function that returns only `shuttle_ls`, the truncation on
joints/pos is lost unless the function mutates the caller's lists in place (works,
because lists are shared references, but implicit across a function boundary) or returns
the truncated triple. Recommended: concern 3 returns `(joints_ls, pos_ls, shuttle_ls)`
explicitly; do not rely on cross-boundary in-place element reassignment.

Why it matters: truncating shuttle without truncating joints/pos breaks per-frame
correspondence (pose frame k must pair with shuttle frame k), even though padding later
re-equalises `T`.

Note: an earlier batch in the simplification pass rewrote the `min_t` guard to
`if len(failed) != len(shuttle):`. The doc's line refs are pre-rewrite; locate by
symbol, not literal line.

### INV-5b `failed_ls` content is unused; only `len(failed)` is load-bearing

`failed_ls` is loaded in S4 (864, 868) purely to supply `len(failed)` as the pose frame
count for `min_t` (895). The boolean values are never read in `collate_npy`: the
frame-zeroing was removed (see the comment at 902-905 and
`docs/architecture_notes/frame_zeroing.md`). By upstream construction
(`detect_players_2d/3d` append one entry per frame to all three arrays),
`len(failed) == len(joints_ls[i]) == len(pos_ls[i])`, so `min_t` could equally read
`len(pos_ls[i])`. Do NOT delete `failed_ls` from the load thinking it is dead: its
length is the frame count. (The `failed = failed[:min_t]` at 900 is genuinely dead, its
result is never read; leaving or dropping it is harmless, but it should not mislead the
split.)

### INV-6 vid derivation and the resolution_df / shuttle_csv_dir dependency in S5

S5 derives, per clip: `clip_stem = Path(branch).name` (885), `vid = int(clip_stem.split("_",1)[0])`
(887), then reads `resolution_df.loc[vid, "width"/"height"]` (890-891) and the shuttle
CSV at `shuttle_csv_dir / (clip_stem + "_ball.csv")` (886). Two contracts:
- Shuttle reads from the shared `shuttle_csv_dir`, NOT from `chosen_root`. Even
  unknown-routed clips (whose npy files come from `unknown_root_dir`) read shuttle from
  the same canonical CSV dir, keyed by stem. The split must not assume
  `shuttle_csv_dir == chosen_root`.
- Every surviving clip's `vid` must exist in `resolution_df` (including unknown clips).
  Keep vid derivation co-located with the resolution + shuttle lookup.

### INV-7 Taxonomy filter order is owned by `label_for_row` (exclude -> merge -> side)

S3 calls `label_for_row(taxonomy, raw_type, side)` (813). `label_for_row`
(config.py:349-357) applies, in order: `excluded_base_stroke_types` first (349-350,
returns `None`), then `merge_map` (351), then side-prefix when `has_sides` and the
merged type is not in `SIDE_AGNOSTIC_TYPES` (352-355), then `classes.index` (357). The
collator must keep:
- `None` -> `continue` (820-821): the drop is delegated, do not reimplement or reorder it.
- the `ValueError` re-wrap that adds clip-stem context (814-819), preserving `from e`.

Do not inline the exclude/merge/side logic into the split; it is a single source of
truth shared with `_derive_class_label` and must stay in `label_for_row`.

### INV-8 Unknown routing and the two pre-loop guards

The guards (S1, 773-787) must run before the CSV read:
- `unknown_root_dir` set but taxonomy excludes unknown -> ValueError (773-778).
- `taxonomy.has_unknown` but `unknown_root_dir is None` -> ValueError (779-787).

Per-row routing (822-827):

```
chosen_root = unknown_root_dir if (raw_type == "unknown" and unknown_root_dir is not None) else root_dir
branch = str(chosen_root / stem)
```

`branch` feeds the existence check (831), the load (S4 reads `branch + "_*.npy"`), and
the shuttle stem (885). The SAME `branch` string must be used across all three. Splitting
label derivation away from file resolution risks computing `branch` with the wrong root.
Keep the guards in the orchestrator and `chosen_root` -> `branch` inside concern 1
(co-located with the existence check and the appends, per INV-2).

### INV-9 dtype / shape contracts of the nine outputs

Read from the captured golden (une_v1_14, T=100, 2D, all four pose styles):

```
J_only.npy      (n, T, 2, 17, 2)   float32     # m=2 players, J=17 coco
JnB_bone.npy    (n, T, 2, 36, 2)   float32     # 17 joints + 19 bones
JnB_interp.npy  (n, T, 2, 36, 2)   float32     # 17 joints + 19 midpoints
Jn2B.npy        (n, T, 2, 55, 2)   float32     # 17 + 19 midpoints + 19 bones
pos.npy         (n, T, 2, 2)       float32
shuttle.npy     (n, T, 2)          float32
videos_len.npy  (n,)               int64
labels.npy      (n,)               int64
clip_stems.npy  (n,)               object      # saved allow_pickle=True
```

`T = seq_len` (100 or 30). Last dim `d = 2` for 2D, `3` for 3D (`--use-3d-pose`), so the
3D pose arrays are `(n, T, 2, K, 3)`. Contract sources the split must preserve:
- float32 on the pose/pos/shuttle arrays: the cast is in `pad_and_augment_one_npy_video`
  (688-690); on-disk per-clip joints/pos/shuttle are float64. Do not move the cast.
- `videos_len` int64: `np.stack(videos_len)` (951) of Python ints
  (`new_video_len = len(pos)`).
- `labels` int64: `np.asarray(labels_ls, dtype=np.int64)` (847).
- `clip_stems` object + `allow_pickle=True`: `np.asarray(..., dtype=object)` (848),
  `np.save(..., allow_pickle=True)` (969).
- Output FILE SET depends on `pose_styles`: the pose files are named exactly by style
  (961-962), so only the requested styles are written (1 file for the default
  `{JnB_bone}`, up to 4). The non-pose five are always written. Preserve the per-style
  filename mapping and the "only requested styles" behaviour.

### INV-10 `pose_styles` validation stays before the ProcessPool

The `bad_styles` raise (909-914) must run before S7. It guards a direct/library/test
caller that bypasses the CLI (the CLI has its own check at `main`:1165). `01` decided
to keep both (lines 264-268). Do not drop it in the split.

### INV-11 No shared mutable state, no global reads, no randomness

`collate_npy` reads no module globals except `VALID_POSE_STYLES` (a constant) and the
imported pure helpers. `resolution_df`, `shuttle_csv_dir`, `taxonomy`, `clips_csv`,
`root_dir`, `unknown_root_dir`, `seq_len`, `save_dir`, `pose_styles` are all parameters.
`clips_csv` is re-read fresh per call (790). `bone_pairs` is computed once (916) and
passed per-task. The ProcessPool pickles arrays per task (copy semantics, no cross-task
sharing). There is no RNG anywhere in `collate_npy` (augmentation is train-time, not
collation-time). Determinism therefore holds as long as INV-4 (submission-order
collection) is kept. The split must not introduce module-level state or reorder reads.

### INV-12 Stack-all-then-save-all ordering

All stacks (S8, 949-951, plus the per-style `np.stack(arrs)` at 962) currently complete
before any `np.save` (961-969). A mid-pipeline failure leaves no partial `set_dir`. Keep
saves at the tail of concern 4, after every stack succeeds, so the split does not start
writing files before the row count is final. `save_dir.mkdir()` (955) is non-parents
(assumes the parent exists); `set_dir.mkdir()` (958-959) creates the `set_name` subdir.
Preserve both.

## 4. Verification checklist

Run after any split, in order.

1. Collation golden, 0-diff. The captured golden under the simplification-pass
   harness checks bit-identical, 5 clips, 9 output files. If `main` has drifted
   from the 2026-06-29 capture, re-run capture on `main` first; otherwise compare
   against the existing artefact.

   What the golden covers: une_v1_14 (no unknown, no sides), train split,
   seq_len=100, all four pose styles (so the bone / interp / midpoint branches all
   fire), 5 clips. It also exercises the missing-file skip path (~24,861 master-CSV
   rows with no fixture files), so INV-1 / INV-2 row alignment under heavy filtering
   IS covered.

2. Standard backstops: ruff clean on the file, full pytest at the green baseline
   (456 passed / 2 known-red / 19 skipped on the laptop venv). Note: no test calls
   `collate_npy` directly (confirmed by grep); pytest covers `Dataset_npy_collated`
   (the reader, `tests/test_dataset.py`, `tests/test_integration.py`), not the
   writer. So pytest is a regression backstop on consumers and import health, NOT a
   behaviour gate on `collate_npy`. The golden + HPC are the only behaviour gates.

3. Paths the golden does NOT cover, needing HPC checks:
   - val + test splits: the golden is train-only. The `split_column ==
     'val' / 'test'` branch (796) runs the same code, but is untested by the
     golden. Covered by the HPC bit-exact (all splits).
   - Unknown routing (INV-8): bst_25 / une_v1_15 (`has_unknown=True`) exercise
     `chosen_root = unknown_root_dir`, the two has_unknown guards, and 'unknown'
     at index -1. Entirely uncovered by the une_v1_14 golden. Covered by the HPC
     bst_25 leg.
   - Sided taxonomies (INV-7): bst_25 / bst_24 (`has_sides=True`) exercise the
     side-prefix path and the `player_side` column read. Uncovered by the golden;
     covered by HPC bst_25.
   - 3D joints (`--use-3d-pose`, `d=3`, INV-9): the fixture is 2D. See open
     question Q1.
   - all-failed / zero-length clip (`videos_len=0`): `collate_npy` stacks it; the
     drop happens later in `Dataset_npy_collated`
     (shuttleset_dataset.py:210-221). If no fixture clip has `len(pos)==0`, the
     `videos_len=0` stack path is untested. Low risk (`collate_npy` does not
     branch on it).
   - The two error guards (773-787) and the `bad_styles` raise (909): not
     triggered by the golden's valid inputs. If a split moves any guard code,
     add a direct unit check; otherwise the orchestrator should leave them
     untouched.

4. HPC bit-exact (blocking). `main` vs branch IDENTICAL on une_v1_14 AND bst_25.
   This is the non-waivable leg that covers unknown-routing + sides + val/test
   that the CPU golden misses. The laptop CPU venv cannot run this; any
   CPU-only verification leg has to halt and hand off for the HPC pass.

## 5. Open questions / risks

- Q1 (3D collation coverage). The collation golden is 2D-only. The HPC bit-exact
  on une_v1_14 / bst_25 runs without `--use-3d-pose`, so the `d=3` stack / save
  path in `collate_npy` (the `(n, T, 2, K, 3)` shapes and the float32 cast of
  3-channel arrays) is unverified by any current gate. Accept (deployed BST-X is
  2D; 3D is wired but cold) or add a small 3D collation smoke if 3D is ever
  revived.

- Q2 (return contract vs in-place mutation, INV-5). A split has to pick how
  concern 3 propagates the truncation: return the `(joints_ls, pos_ls,
  shuttle_ls)` triple (recommended, explicit) versus rely on in-place list-element
  reassignment across the function boundary (works but implicit). The
  simplification-pass split took the explicit return.

- Q3 (do not "fix" row order, INV-1). `collate_npy` writes raw on-disk row
  order; downstream `train_partial` reorders the dump (noted at
  `bst_x_train.py:1091`), which is a separate concern. A split must not add any
  sort / reorder to `collate_npy` to "tidy" alignment; the alignment is
  positional and already correct.

## Adversarial review (round 1)

Reviewer brief: distrust every claim, verify against `main` source + the captured
golden + the simplification-pass runbook. Done 2026-06-29 on a clean `main`.

### Verdict

Fit to guide the split, after one correction. The invariant set (INV-1..INV-12) is
real, accurately described, and complete enough that a split following it will not
break behaviour: I re-read `collate_npy` 719-970 end to end and the only behaviour-
preserving invariants the doc misses are low-severity (an input guard and a variable
rebinding, both below). Every `file:line` in the stage map and the helper list is
accurate, INV-9's shapes/dtypes match the captured golden byte-for-byte, and the
cross-doc quotes (05/01/03) are faithful. The one real error is INV-12's claim that
all stacks finish before any save: the per-style pose stacks are interleaved with the
saves, so the "no partial `set_dir`" safety line is overstated. Fix that wording and
the doc is sound.

### Findings

| id | severity | what's wrong / missing | evidence | suggested fix |
|----|----------|------------------------|----------|---------------|
| R1 | medium | INV-12: "All stacks (S8, 949-951, plus the per-style `np.stack(arrs)` at 962) currently complete before any `np.save`" is false. The pos/shuttle/videos_len stacks (949-951) do finish first, but the per-style pose stacks at 962 are computed *inline as the argument to* `np.save` inside the 961 loop, so `J_only.npy` is on disk before `JnB_bone` is even stacked. The "mid-pipeline failure leaves no partial `set_dir`" claim is therefore overstated: `set_dir` is `mkdir`'d at 958, and a stack failure on the k-th style leaves styles 0..k-1 written. | `prepare_train:961-962` `for k, arrs in pose_ls.items(): np.save(..., np.stack(arrs))`; `set_dir.mkdir()` at 958. In practice 962 won't raise (uniform padded shapes; per-task errors already surface at the S7 `.result()` 941-947), so blast radius is ~nil, but the structural claim is wrong. | Reword: pos/shuttle/videos_len stacks complete before the save block; the per-style pose stacks run inline within the save loop. Row count `n` is final before any save (the load-bearing point). Drop the "no partial set_dir" guarantee or qualify it (set_dir exists from 958; partial pose files possible if a 962 stack ever raised). |
| R2 | low-medium | Section 4 item 4 asserts the HPC bit-exact "covers unknown-routing + sides + val/test that the CPU golden misses", but the simplification-pass plan doesn't state the HPC comparison *granularity* for collation. Val/test aggregate metrics are row-order-invariant and `train_partial` reorders the train dump, so a pure block-permutation of val/test rows in `collate_npy` (with `label[i]`/`pose[i]`/`stem[i]` kept aligned) would not move metrics and would only surface if the HPC leg compares the collated `.npy` (or the row-aligned prediction npz) directly. Label-desync IS caught (it changes metrics). | Reorder noted at `bst_x_train.py:1090-1091` (`adjust_to_partial_train_set`, called `shuttleset_dataset.py:224`). Mitigation: a clean split preserving INV-4 cannot introduce such a permutation, so residual risk is low. | Add one line: for the HPC leg to actually cover collation row-order on val/test/bst_25, it must compare the collated `.npy` arrays (or the row-aligned npz), not just aggregate test metrics. |
| R3 | low | Missed input guard. S2 spans 789-796 but the invariant list never enumerates the `split_column not in clips_df.columns -> KeyError` guard (791-795), though it does enumerate the S1 guards (INV-8). If concern 1 extracts the CSV read+filter, this guard must travel with it. | `prepare_train:791-795`. | Note in INV-1/concern-1 contract that the split_column-membership KeyError (791-795) is part of concern 1's CSV-read step. |
| R4 | low | Missed rebinding. S7 reuses the names `pos_ls`/`shuttle_ls` for a *different generation*: the S4-loaded, S5-truncated lists are consumed by the zip at 923, then `pos_ls`/`shuttle_ls` are rebound to `[]` (937-938) and refilled with the post-pad outputs (941-947); `joints_ls` is consumed, not rebound (its padded form lives in `pose_ls[k]`). The stage map mentions both generations but never flags the name collision as a split hazard. | `prepare_train:923` (consume), `937-938` (rebind), `941-947` (refill). | Note the rebinding; recommend distinct names across the concern-3/concern-4 boundary (e.g. `loaded_pos` vs `padded_pos`) so an inlined orchestrator doesn't shadow. |
| R5 | low/info | "Nine outputs" in INV-1 reads as fixed, but INV-9 correctly says only requested pose styles are written (6 files for the default `JnB_bone`, 9 only when all four are requested as in the golden). Minor internal inconsistency. | INV-1 vs INV-9; golden writes 9 because it requests all four styles (verified: `J_only, Jn2B, JnB_bone, JnB_interp, clip_stems, labels, pos, shuttle, videos_len`). | Say "all outputs (5 non-pose + 1-to-4 pose; 9 in the all-styles golden)". |
| R6 | low/info | Undocumented floor: `n == 0` (empty split) crashes at `np.stack(pos_ls)` (949), "need at least one array to stack". `collate_npy` does not guard it. Unreachable for real splits and the golden (`n > 0`), so not a split risk, but the clean-tail framing of INV-11/INV-12 doesn't mention it. | `prepare_train:949`. | Informational; optionally note the function assumes `n >= 1`. |

### Doc claims checked and found CORRECT

Line-ref / structure:
- Stage map S0..S11 (719-970): every boundary verified against source. `collate_npy` def at 719, body ends at the 970 print. All ten internal boundaries (768, 790/796, 807-836/838-846/847-852, 856-869, 883-907, 909-916, 920-947, 949-951, 954-959, 961-969) match.
- Helper list: `pad_and_augment_one_npy_video` 661-716 + float32 cast 688-690; `make_seq_len_same` 50-83 (return desc consistent with the function's own docstring); `create_bones` 86-96, `interpolate_joints` 99-110, `get_bone_pairs` 35-47 (counted 19 coco pairs); `get_shuttle_result` 490-497 (returns float64 (t,2)); `label_for_row` config.py:333-366 with ordered logic 349-357; `VALID_POSE_STYLES` prepare_train:658. All accurate.

Invariants:
- INV-1..INV-8, INV-10, INV-11: all line refs correct and behaviour correctly described. INV-7's "single source of truth shared with `_derive_class_label`" verified: `data_access.py:299` `_derive_class_label` is a thin wrapper calling `label_for_row`. INV-5b's "frame-zeroing removed" verified: comment at 902-905 + `docs/architecture_notes/frame_zeroing.md` exists; `failed = failed[:min_t]` at 900 is genuinely dead (reassigned at 893 next iter, unread after 900); `len(failed)==len(pos_ls[i])` holds by `detect_players_2d/3d` per-frame appends.
- INV-9: shapes AND dtypes verified against the captured golden, exact: `J_only (5,100,2,17,2) f32`, `JnB_bone/JnB_interp (5,100,2,36,2) f32`, `Jn2B (5,100,2,55,2) f32`, `pos (5,100,2,2) f32`, `shuttle (5,100,2) f32`, `videos_len (5,) i64`, `labels (5,) i64`, `clip_stems (5,) object`.

Section 2 split mapping: faithful to the source review and the pass runbook
(modulo markdown bolding / arrow glyph).

Section 4 / 5:
- Golden runs clean on `main`: `OK ... bit-identical (5 clips, 9 output files ...)`; the missing-skip count is exactly 24861 as the doc states; no `unknown_root_dir` needed (une_v1_14 excludes unknown).
- pytest baseline EXACT: `2 failed, 456 passed, 19 skipped` (the 2 are `test_api.py::test_upload_returns_queued` + `::test_full_job_lifecycle`, the known-red `src/api` FileNotFoundError). `ruff check` on the target: "All checks passed!". No test calls `collate_npy` (grep clean). `Dataset_npy_collated` (def `shuttleset_dataset.py:146`) covered by `test_dataset.py` + `test_integration.py`; zero-length drop at 210-221 confirmed.
- Taxonomy facts: `une_v1_15` / `bst_25` `has_unknown=True`; `bst_24` / `bst_25` `has_sides=True`; `unknown` sits at the last index (`bst_25` -> 24, `une_v1_15` -> 14), so the doc's "index -1" is right.
- The simplification-pass runbook tagged the collation golden "to capture" in its
  section 0; the artefact was in fact captured and green on 2026-06-29. The HPC
  bit-exact on une_v1_14 / bst_25 was the gating leg.

Open questions: Q1 (golden is 2D-only) correct; Q2 (in-place vs return for the truncation) real and well-scoped; Q3 (`train_partial` reorders the dump, noted at `bst_x_train.py:1091`) accurate (the cite is to the `dump_predictions` docstring where it's noted; the reorder itself is `adjust_to_partial_train_set`). The only missing risk is R2's HPC-granularity caveat, which sits next to Q3 since the two share the reorder mechanism.

---

_Originally written as the B4 split pre-analysis in the simplification pass
(merged at `18e5c2c`, 2026-06-30). The split itself landed as commit
`5b361b8`._

