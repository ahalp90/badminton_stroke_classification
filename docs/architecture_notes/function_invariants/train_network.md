# `train_network`: function invariants

> _Last verified: 2026-06-29 against pre-pass `main`. Line refs are pre-pass;
> the simplification-pass commits between then and `18e5c2c` shifted line
> numbers. The invariants below are the durable content; re-verify line refs
> before relying on them._

Maps the invariants a naive split of `train_network` (`src/bst_x/bst_x_train.py`)
would silently break. This is the reviewable map; any implementer re-verifies
against live code before carving (line numbers are a 2026-06-29 snapshot, locate
by symbol).

All `file:line` references are `src/bst_x/bst_x_train.py` unless another file is
named. Nothing here is applied; ANALYSIS only.

## Contents

- [0. Scope and the one big surprise](#0-scope-and-the-one-big-surprise)
- [1. Current structure of `train_network`](#1-current-structure-of-train_network)
- [2. The proposed split](#2-the-proposed-split)
- [3. Invariants a naive split would break](#3-invariants-a-naive-split-would-break)
- [4. Verification checklist](#4-verification-checklist)
- [5. Open questions and risks](#5-open-questions-and-risks)

## 0. Scope and the one big surprise

The split is two PURE extractions out of the setup phase, leaving the epoch
loop and the save phase untouched. Both extracted blocks consume zero RNG
(proven below), so the split is RNG-neutral and behaviour-neutral by
construction. The risk is not the proposed split; it is a naive change going
further (touching the loop body, reordering setup, rebuilding state per-epoch).

The big surprise, load-bearing for every "bit-exact" claim below: **there is no
seed set anywhere in the live training path.** No `torch.manual_seed`, no
`np.random.seed`, no `use_deterministic_algorithms`, no cudnn flags. Verified by
grep across `bst_x_train.py`, `bst_x_common.py`, and
`preparing_data/shuttleset_dataset.py` (zero hits). The train loader is
`shuffle=True` with no `generator=` passed
(`preparing_data/shuttleset_dataset.py:287-293`) and `num_workers=0`
(`bst_x_train.py:891`). So every stochastic draw (shuffle order, the two
augmentations, model dropout, weight init) flows through the single process-global
default torch generator. One consequence and one corollary:

- Two unseeded runs of `main` diverge from each other. A literal "main vs branch
  bit-exact" training comparison is only meaningful if the comparison harness
  pins the RNG itself; the committed code never will.
- Because all stochasticity funnels through one global generator at
  `num_workers=0`, a single `torch.manual_seed(N)` before the build+train sequence
  fully determines a CPU run (proven in section 4). That is what makes a seeded
  before/after equivalence gate possible.

## 1. Current structure of `train_network`

Signature at `434-445`:
`train_network(model, train_loader, val_loader, device, save_path, n_bones, n_classes, class_ls, taxonomy, tb_dir=None)`. It reads the module-global `hyp`
(`85-175`) for everything not in the signature (this hybrid param-plus-global
shape matters, see section 3). Returns `(model, val_at_best)` at `854`. The model
is mutated in place (`load_state_dict` at `801`) AND returned; the caller relies
on both (`seek_network_weights`, `998`).

Ordered phases:

```
SETUP (no RNG consumed here):
  449          create SummaryWriter (tb_dir or default ./runs/)
  457-464      pose_style guard: raise NotImplementedError unless 'JnB_bone'
  468-485      read hyp.augmentation; build CoupledFlip + ConstrainedJitter; print
  487-589   => LOSS BUILD  (extraction target #1, _build_loss_fn)
    507-512      guard: use_val_improvability_gate requires adaptive_focal
    513-572      adaptive_focal branch: guards (xor class_weights, LS==0) +
                 AdaptiveFocalLoss(...) + diagnostic prints
    573-587      class_weights branch: build per-class weight tensor + print
    588-589      else: plain nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)
  590-612   => OPTIMISER PARAM SPLIT  (extraction target #2, _split_param_groups)
    598-605      walk model.named_parameters(): decay vs no_decay
    606-607      print counts
    608-612      optim.AdamW([{decay, wd}, {no_decay, 0.0}], lr=hyp.lr)
  616-621      get_cosine_schedule_with_warmup(num_training_steps = n_epochs * len(train_loader))
  624-638      init best/second trackers, early_stop_count, the five best_val_* snapshots

EPOCH LOOP (640-795), per epoch:
  643-647      aux schedule: model.set_schedule_factors(cg_factor, ap_factor)
  649-661      train_one_epoch(...) -> loss, tp, fp, fn, jitter counters
  664          per_class_f1_from_counts(tp, fp, fn)
  665-666      if AdaptiveFocalLoss: loss_fn.update_alpha(train F1)
  668-674      validate(...) -> val_loss, macro, min, per_class, present, acc, top2
  679-680      if gate enabled: loss_fn.apply_val_gate(val F1, present)
  682-696      epoch print + alpha top3/bot3
  698-745      TB scalars (loss, F1 macro/min, aux factor, LR, jitter rates, per-class)
  747          curr_macro, curr_min = f1_score_avg.item(), f1_score_min.item()
  750          early_stop_count += 1
  751-779      if curr_macro > best_macro: snapshot best_state + 5 best_val_* ;
               print picked + top5/bot5 ; early_stop_count = 0
  780-781      elif curr_macro > second_macro: update second
  783-787      best_min / second_min tracking
  789-791      best_val_loss tuple-min
  793-795      if early_stop_count == hyp.early_stop_n_epochs: break

SAVE / ARTEFACTS (797-854):
  799-801      mkdir; torch.save(best_state); model.load_state_dict(best_state)
  807-829      add_hparams (incl. nested _to_hparam_value 807-810); metric_dict
  830          writer.close()
  832-853      build val_at_best dict from the best-macro snapshot (None if degenerate)
  854          return model, val_at_best
```

Helpers it calls (all already module-level, NOT in scope to change):
`aux_schedule_factor` (`182-202`), `train_one_epoch` (`205-294`), `validate`
(`298-374`), `accumulate_class_counts` / `per_class_f1_from_counts`
(`loss/adaptive_focal.py:508-530` / `479-506`),
`AdaptiveFocalLoss.update_alpha` / `.apply_val_gate`
(`loss/adaptive_focal.py:191`, `:345`).

## 2. The split

The split is two extractions from the SETUP phase, leaving the epoch loop and
the save phase byte-identical:

- `_build_loss_fn(...)` <- lines `487-589` (or `507-589` if the explanatory
  comment block stays as a header). Returns `loss_fn`. Reads module-global
  `hyp.{adaptive_focal, class_weights, label_smoothing,
  use_val_improvability_gate, val_improvability_gate, n_epochs}` plus params
  `n_classes, class_ls, taxonomy, device`. The three fail-loud guards
  (`507-512`, `514-518`, `519-525`) belong inside it.
- `_split_param_groups(model)` <- the walk at `598-605`. Returns the two lists
  (or the AdamW param-group dicts). The `print` (`606-607`) and the
  `optim.AdamW` construction (`608-612`, reads `hyp.lr` / `hyp.weight_decay`)
  can stay in the caller or move with it; pick one and state it (see open
  questions).

This was a "confirm-first" change in the simplification pass: model-adjacent
training-entry code under the no-unsolicited-model-rewrites rule. The
simplification-pass implementer got the explicit OK before carving, even though
the change is pure structure.

How the split relates to the `validate()` rewrite that landed earlier in the
same pass: the earlier batch rewrote `validate()`'s hand-rolled one-hot
TP / FP / FN block (`~328-341`) to `accumulate_class_counts(logits.argmax(dim=1),
labels, n_classes)` with Option B device handling. **The split does not touch
`validate()`**; it operates on the post-rewrite version. The val-metrics golden
tests `accumulate_class_counts` against a captured golden of the old one-hot
output; it never calls `validate()` or `train_network`. So in the split's gate
it is a cheap regression backstop (confirms the counts path is still
untouched), not a proof that the `train_network` restructure preserved
behaviour. The substantive proof is the seeded train equivalence + the HPC
bit-exact (section 4).

## 3. Invariants a naive split would break

The proposed two extractions are RNG-neutral and side-effect-light. This section
is mostly a fence against an executor doing MORE than the spec.

### 3.1 RNG and seed determinism

- **The setup phase consumes zero RNG.** Model init already happened in
  `build_bst_x_network` BEFORE `train_network` is called
  (`get_network_architecture`, `956`, runs at `__main__:1396`, before
  `seek_network_weights` at `1399`). Inside `train_network` the first RNG draw is
  in the epoch loop: the loader shuffle, then `CoupledFlip.__call__`
  (`augmentations.py:154`), `ConstrainedJitter.__call__`
  (`augmentations.py:262, 351-352`), then model dropout. The two aug CONSTRUCTORS
  (`augmentations.py:117, 229`) and `AdaptiveFocalLoss.__init__`
  (`loss/adaptive_focal.py:88-145`, buffers init to `torch.ones`) consume no RNG.
  Verified by reading each constructor.
- Therefore reordering `_build_loss_fn`, `_split_param_groups`, scheduler build,
  and tracker init among themselves cannot perturb any random draw. This is the
  reason the split is safe. State it in the worklog so the reviewer can confirm
  the claim rather than re-derive it.
- The naive break: any change that moves an RNG consumer relative to another. In
  practice that means the loop body, which the split must NOT touch. Do not hoist
  `set_schedule_factors`, the `train_one_epoch` call, or the loader iteration out
  of the per-epoch loop; do not reorder train-vs-validate.

### 3.2 Optimiser and scheduler continuity

- **One optimiser, one scheduler, for the whole run.** `optimizer` is built once
  (`608-612`) and `scheduler` once (`616-621`); `scheduler.step()` fires per-batch
  inside `train_one_epoch` (`278`). A naive split that rebuilt either per-epoch
  would reset Adam's moment estimates and the LR schedule. `_split_param_groups`
  must return groups that the caller feeds into a single `AdamW`; it must not
  construct-and-discard.
- **`num_training_steps = hyp.n_epochs * len(train_loader)`** (`619`). The cosine
  schedule's shape depends on this total; recompute it identically and from the
  same `train_loader` (so `len()` matches). Warmup is `hyp.warm_up_step` (`617`),
  counted in batches.
- **Weight-decay groups are the current default optimiser and are correctness-
  critical.** The decay/no-decay split (`598-605`) puts `ndim<=1` params (norm
  gains, biases) and any name matching `('embedding_', 'learned_token_')` into the
  no-decay group at `wd=0.0`; everything else gets `hyp.weight_decay`
  (`609-610`). The code comments a verified count "27 decay / 55 no-decay tensors"
  for BST_CG_AP (`597`); the section-4 smoke reproduced exactly `decay=27,
  no_decay=55` on the default BST_X build, so that is the regression number to
  watch. `_split_param_groups` must preserve: the `requires_grad` skip (`601`),
  the `ndim<=1 OR name-hint` predicate (`603-605`), and which group carries which
  `weight_decay`. `hyp.weight_decay` is the swept dimension
  (`__main__:1211-1214`); only the decay group gets it, no-decay stays `0.0`.

### 3.3 Metrics accumulation and best-epoch selection

These all live in the loop / save phases, which the split leaves intact. They
are the highest-value things for any future refactor NOT to "tidy" in passing.

- **Per-class counts.** Train counts accumulate in `train_one_epoch` (device
  tensors, `240-291`); val counts in `validate` (`308-341`). The split touches
  neither.
- **Best-macro selection is strict `>` and first-epoch-wins on ties** (`751`).
  The new-best branch atomically co-assigns `best_state` (deepcopy of the
  state_dict, `756`) and the five snapshots `best_val_f1_per_class`,
  `best_val_present`, `best_val_accuracy`, `best_val_top2`, `best_macro_epoch_snap`
  (`759-763`), and resets `early_stop_count = 0` (`779`). These six writes are a
  single atomic group keyed to the SAME epoch as `best_macro`. If any split ever
  separated the checkpoint snapshot from the metric snapshot, the saved `.pt`
  could stop matching the recorded `val_at_best`. Keep them together.
- **`val_at_best` is built from that snapshot, not re-derived** (`832-853`):
  `macro_f1 = mean(present per-class)`, `min_f1 = min(present per-class)`, so the
  manifest's macro/min stay exactly consistent with the per-class breakdown. The
  present-filter (`841`) means `per_class_f1` has only the classes seen in val that
  epoch (the section-4 smoke showed 12 of 14 entries on a 12-sample val set). This
  dict is what `track_serial` writes to
  `extra.val_at_best_macro_epoch` (`__main__:1423`); it is the per-serial manifest
  payload and a memory-noted contract. Preserve its key set exactly:
  `{epoch, macro_f1, min_f1, accuracy, top2_accuracy, per_class_f1}`.
- **Second-best, min-F1, and val-loss trackers** (`780-791`) feed only the TB
  `add_hparams` metric row (`812-829`). `best_val_loss` uses tuple-min
  (`789-791`), so a val-loss tie breaks to the earlier epoch. Non-behavioural for
  the checkpoint, but part of the HParams row, so preserve if the bit-exact reads
  TB (it generally does not; flag as low-risk).
- **Early stop is `==`, not `>=`** (`793`). It works only because the counter
  resets to 0 on every new macro best, so it lands on the threshold exactly. Do
  not change the increment (`750`) / reset (`779`) / compare (`793`) ordering.

### 3.4 Checkpoint timing and contents

- The checkpoint is `best_state` = `deepcopy(model.state_dict())` taken at the
  best-macro epoch (`756`), written once at `800` after the loop, BEFORE the TB
  hparam logging (the `797-798` comment: save first so a logging failure cannot
  lose the weights). Then `model.load_state_dict(best_state)` (`801`) restores
  best weights into the in-memory model that gets returned. Preserve both the
  order (save then restore then log) and the in-place restore (the caller's
  `self.net` is the same object, and `task.test()` runs on it afterwards).
- **Latent: `best_state` has no degenerate guard.** It is assigned only inside
  the new-best branch (`756`). If no epoch ever beats `best_macro = 0.0` (every
  present class scores F1 0 every epoch), `best_state` is unbound and
  `torch.save` (`800`) raises `UnboundLocalError`, whereas `val_at_best` IS
  guarded (`837`). This is a pre-existing asymmetry, not introduced by the
  split. The split must preserve current behaviour (do not silently add a
  guard); the simplification-pass split left it as-is.

### 3.5 AMP, grad handling, loss reduction, device

- **No AMP, no `GradScaler`, no `autocast`, no gradient clipping, no gradient
  accumulation, no `torch.compile`.** Verified by grep (zero real hits). The step
  is plain `zero_grad` / `backward` / `step` / `scheduler.step` per batch
  (`275-278`). Nothing here for the split to preserve beyond not adding any of it.
- **Loss reduction is mean.** `nn.CrossEntropyLoss` defaults to `reduction='mean'`
  (`587, 589`); `AdaptiveFocalLoss` mirrors `reduction='mean'`
  (`loss/adaptive_focal.py` docstring). Epoch loss is `total_loss / len(loader)`
  (`293`). `_build_loss_fn` must not pass a different reduction.
- **Device placement.** `AdaptiveFocalLoss(device=device)` (`542`) moves its
  buffers; the class-weight tensor is created `device=device` (`574`); plain CE has
  no device. `_build_loss_fn` must keep `device` threaded so the loss buffers land
  where the model and batches are. (On the laptop this is `cpu`; on HPC `cuda`.)

### 3.6 Module/global state, loss-object continuity, side effects

- **`hyp` is module-global and read-only inside `train_network`.** It is
  reassigned via `_replace` only in `__main__` (`1250, 1274`), before the
  serial loop. The extracted helpers, as module-level functions, read the same
  global. This hybrid (params for `n_classes` / `device` / etc., global for the
  rest) is a pre-existing smell; preserve it. Do not "improve" it into a `hyp`
  parameter inside a structural split (that is a signature change across the
  call site).
- **The loss object is built once and mutated through the loop.** `loss_fn` from
  `_build_loss_fn` is the SAME instance every epoch; `AdaptiveFocalLoss` carries
  EMA state (`f1_running`, `alpha`) and an epoch counter advanced by
  `update_alpha` (`666`), plus gate buffers touched by `apply_val_gate` (`680`).
  Rebuilding it per-epoch would wipe the warm-up counter and the EMA. The
  split builds once in setup, so this holds; the fence is against any
  loop-body churn.
- **Per-epoch ordering is load-bearing** and called out in the code comments
  (`675-680`): `update_alpha(train F1)` (`666`) then, after `validate`,
  `apply_val_gate(val F1)` (`680`); the gated alpha drives the NEXT epoch's
  training. Also `set_schedule_factors` (`647`) before `train_one_epoch`. The
  split leaves the loop intact; do not reorder.
- **Side effects of `train_network`:** stdout prints (terminal-only here; the
  `train_network` call at `1399` is OUTSIDE the `redirect_stdout(tee)` block that
  starts at `1403`, per the `1320-1322` comment), TB event files (continuous +
  `add_hparams` + `close`), and exactly one `.pt` write. `dump_predictions` and
  `track_serial` are in `__main__`, NOT in `train_network`. Moving the loss/optim
  prints into helpers keeps them on stdout and is not covered by any bit-exact
  gate, so print-order drift is cosmetic; still, keep them for the operator.
- **No early-stopping or global state beyond the above.** No `aim`, no file writes
  other than the checkpoint and TB, no environment mutation inside the function.

## 4. Verification checklist

Run order: CPU first, then HPC. Commands in `~/.venvs/badminton-cicd`.

### What CPU can verify (laptop, badminton-cicd)

1. **Ruff + full pytest** as the regression backstop (the green baseline is 456
   passed / 2 known-red / 19 skipped). Necessary, not sufficient.
2. **Val-metrics golden, 0-diff** (`val_metrics_equiv.py check`). Cheap
   backstop that the counts path is untouched; it does not exercise
   `train_network` (see section 2).
3. **Model build + forward bit-exact, 0-diff** (`model_bitexact.py check`).
   Confirms the model and its restored-checkpoint forward are untouched. The
   split should not touch the model at all, so this is a backstop, not the
   core proof. It uses fixed seeds (`model_bitexact.py:32-33, 48, 62`) and
   does NOT run `train_network`.
4. **Training smoke (CPU, synthetic data): confirmed feasible.** A 2-epoch run
   of `train_network` on a synthetic dataset matching the `Dataset_npy_collated`
   tuple contract (`shuttleset_dataset.py:269-271`:
   `((human_pose[t,m,J+B,2], pos[t,m,2], shuttle[t,2]), videos_len, label)`,
   `m=2`, `J+B=36` for JnB_bone) completes in about 1 s/epoch on CPU in this
   venv. It must force `device='cpu'`: `torch.cuda.is_available()` returns True
   on the laptop but CUDA ops error (`no kernel image is available... GTX 960M`
   is sm_50), and `Task.device` (`866`) would pick `cuda` and crash. Drive the
   smoke by calling `train_network` directly with `device='cpu'`, after
   `t.hyp = t.hyp._replace(n_epochs=2, early_stop_n_epochs=100, warm_up_step=2,
   seq_len=30, batch_size=8)` (it reads the module global, not a param). The
   smoke exercises: aug setup, the default adaptive-focal `_build_loss_fn`
   branch, the param split (printed `decay=27, no_decay=55`, matching the `597`
   comment), the epoch loop, best-macro snapshot + "Picked!", the `.pt` save
   (~7.4 MB), and the `val_at_best` dict with all six keys.
5. **Seeded before / after equivalence (the real CPU gate): confirmed valid.**
   A seeded `(build + train)` is bit-reproducible on CPU: two runs with
   `torch.manual_seed(0)` reset before each build + train sequence gave 0.0 max
   weight diff and an identical `val_at_best`. Capture a golden `(state_dict,
   val_at_best)` from a SEEDED `train_network` on `main`, then run the SAME
   seeded harness on the edited tree and assert bit-identical. This is far
   stronger than "it runs": it catches any RNG-order perturbation a bad split
   introduces. The seed lives in the harness wrapper, never in committed
   source. The simplification-pass split used this as the substantive CPU
   proof, with the smoke (item 4) as the it-still-runs check.

What CPU CANNOT verify:

- The GPU device-footgun class (e.g. a tensor left on the wrong device). The
  laptop GPU cannot launch kernels, so `device='cuda'` paths are untested
  here. The same caveat applied to the earlier `validate()` Option-B rewrite
  and to any device handling inherited from the loop.
- CUDA-kernel pass-to-pass nondeterminism. CUDA kernels are the pass-to-pass
  wobble; bit-exactness GPU-side is not guaranteed unless the HPC harness also
  sets `torch.use_deterministic_algorithms(True)` and matching cudnn flags
  (the live code sets none, see section 0).
- Real-data behaviour: synthetic random tensors prove the plumbing and the
  RNG order, not the trained-model trajectory on une_v1_14 / bst_25.

### What MUST be HPC (bourbaki / engelbart, GPU, blocking)

- **Real-data bit-exact on une_v1_14 AND bst_25, `main` vs the edited tree.**
  This is the authoritative gate. Because the live path is unseeded (section
  0), this run MUST pin the RNG in the comparison harness
  (`torch.manual_seed`, plus deterministic flags and `num_workers=0`) for a
  bit-exact comparison to mean anything; otherwise it degrades to "metrics in
  the same range", which would NOT catch a subtle RNG-order break.
- The laptop CPU venv cannot self-certify this leg; any CPU-only verification
  pass has to halt and hand off for the HPC pass.

## 5. Open questions and risks

1. **How does the HPC bit-exact seed a training run?** The live code sets no
   seed (section 0), so "main vs edited bit-exact training" needs the
   comparison harness to inject `torch.manual_seed` + deterministic flags +
   `num_workers=0`. Without that the gate degrades to a metrics-in-range check.
   This is the single biggest risk to the bit-exact gate being meaningful and
   has to be resolved before the HPC leg runs.
2. **`_split_param_groups` return contract and the print / optimiser
   boundary.** Does it return raw `(decay, no_decay)` lists with the caller
   building the AdamW param-group dicts, or the list-of-dicts (reading
   `hyp.weight_decay` itself)? Where does the `decay=.. no_decay=..` print
   (`606-607`) live? Either is fine; pick one and keep `hyp.lr` /
   `hyp.weight_decay` wired identically. The simplification-pass split kept
   the print + AdamW construction in the caller and returned raw lists.
3. **`_build_loss_fn` boundary: do the three fail-loud guards move inside
   it?** They are loss-config validation (`507-512, 514-518, 519-525`) and
   read like part of the loss build, so inside reads cleanest, but it changes
   where the `ValueError` / `NotImplementedError` is raised in a stack trace.
   The simplification-pass split moved them inside the helper.
4. **The degenerate-run `best_state` `UnboundLocalError`** (section 3.4).
   Pre-existing. The simplification-pass split preserved it
   (behaviour-neutral). Any future change should treat the guard fix as a
   separate decision.
5. **`seq_len` in the CPU smoke.** The probe used `seq_len=30` for speed and
   built the model to match; the default `hyp.seq_len` is 100 (`99`). A
   seeded equivalence harness pins whatever `seq_len` it captures the golden
   at (build and train must agree). Not a behaviour risk, just a
   harness-consistency note.

## Adversarial review (round 1)

Independent re-verification against live code (2026-06-29 snapshot), with the two
reference probes and three extra checks actually run in `~/.venvs/badminton-cicd`
(torch 2.11.0+cu130). Method: re-read every cited `file:line`, grep the full
training path for seeds and AMP, run the smoke + repro probes, add a cross-process
determinism check the in-repo probe does not cover, and force a degenerate run to
trigger the latent bug. Findings appended only; body unchanged.

### Verdict

**Fit to guide the split.** The two big load-bearing claims (no seed in the live
path; the degenerate-run `best_state` `UnboundLocalError`) are both correct, the
seeded CPU gate is valid (and stronger than the doc shows: it holds cross-process),
and the invariant map is thorough and accurate. Spot-checking ~60 `file:line`
references turned up only two 1-2 line drifts, both covered by the doc's own
"locate by symbol" instruction. The fixes below are tightening, not correction.

### CONFIRM / REFUTE on the three load-bearing claims

- **No seed anywhere in the live training path: CONFIRM.** `rg` for
  `manual_seed|random.seed|use_deterministic_algorithms|cudnn|default_rng|Generator|seed(|generator=`
  across `bst_x_train.py`, `bst_x_common.py`,
  `preparing_data/shuttleset_dataset.py`, `preparing_data/augmentations.py`,
  `loss/adaptive_focal.py` AND `model/` returned zero hits. Train loader is
  `shuffle=True` with no `generator=` (`shuttleset_dataset.py:290`), `num_workers`
  from `Task`'s `(0, 0, 0)` (`bst_x_train.py:891`). A "main vs branch bit-exact
  training" gate is therefore only meaningful with an injected seed; the committed
  code never pins one. Exactly as the doc states.
- **Seeded CPU (build + train) is bit-reproducible: CONFIRM, and extended.**
  The in-repo repro probe gives `weights bit-exact: True | max abs weight diff:
  0.0` and `val_at_best identical: True`. That probe runs both passes in ONE
  process, but the actual gate workflow is cross-process (capture golden on
  `main`, compare on the edited tree). A seeded build + train in two separate
  interpreters gave identical `STATE_DICT_SHA256` (`a735c94b...d0e68f5b`) and
  identical `val_at_best`, with no `use_deterministic_algorithms` needed (CPU +
  `num_workers=0` is enough). So the gate the doc recommends is valid as
  written, including cross-process.
- **Latent `best_state` `UnboundLocalError` on a degenerate run: CONFIRM,
  runtime-reproduced.** Forced a run where every epoch scores macro 0.0 (model
  wrapped to always predict an absent class, val labels all one present class).
  No "Picked!" ever fired, then `torch.save` raised
  `UnboundLocalError: cannot access local variable 'best_state'` at exactly
  `torch.save(best_state, str(save_path))` (`:800`). Pre-existing: `best_state` is
  assigned only inside the new-best branch (`:756`) and saved unconditionally
  (`:800`); `val_at_best` is guarded (`:837`) but `best_state` is not. The split
  does not introduce it.

### Findings

| id | severity | issue | evidence | fix |
|----|----------|-------|----------|-----|
| R1 | confirm | No-seed claim is correct (see above). | grep across train / common / dataset / aug / loss / model = 0 hits; `shuttleset_dataset.py:290`, `bst_x_train.py:891`. | None; keep. |
| R2 | confirm | Seeded within-process repro is correct. | repro probe: `weights bit-exact: True`, max diff `0.0`, `val_at_best identical: True`. | None. |
| R3 | low-med | In-repo probe only proves WITHIN-process determinism; the gate is used cross-process. Claim still holds, evidence was narrower. | Probe runs `one_run("a")` + `one_run("b")` in one process. Cross-process check: two fresh interpreters gave identical `STATE_DICT_SHA256 a735c94b...d0e68f5b` + identical `val_at_best`. | Add one line to section 4 item 5: golden-capture and edited-tree compare run in separate processes (the natural workflow); CPU + `num_workers=0` is bit-stable cross-process without deterministic flags (verified). |
| R4 | confirm | Latent `best_state` `UnboundLocalError` is real and pre-existing. | Degenerate run -> `UnboundLocalError` at `torch.save(best_state, ...)` (`:800`); assign-only-in-branch (`:756`) vs guarded `val_at_best` (`:837`). | Preserve per the doc; open question 4 carries the decision. |
| R5 | low | Line drift: section 3.2 cites `warm_up_step` at `617`; actual is `618` (`617` is `optimizer=optimizer,`). | `sed -n '616,619p'`. | Retarget `618`, or rely on locate-by-symbol (already the doc's rule). |
| R6 | low | Line drift: `AdaptiveFocalLoss.__init__` cited `88-145`; `def` is at `86` (signature `86-99`). The "buffers init to `torch.ones`" claim is accurate (`125-126`). | `rg 'def __init__' loss/adaptive_focal.py` -> 86. | Retarget `86`. |
| R7 | low | "27 decay / 55 no-decay" confirmed, but it is the default `BST_X` (= `BST_CG_AP`) build only; a different variant or `requires_grad` change moves it. | `bst_x_common.py:33` (`'BST_X': BST_CG_AP`); smoke printed `decay=27 tensors, no_decay=55`. | Note the count is model / config-pinned (the doc mostly does; one explicit clause helps). |
| R8 | low | `_to_hparam_value` (`807-810`) is a separate "worth doing" item from the simplification review, not one of this split's two extractions. A change "in there anyway" could fold it in as scope creep. | Per-module note in `simplification_review.md` ("`_to_hparam_value` (807-810) is a 4-line nested def used once. Inline it..."). | Add one fence line: the split is exactly the two named extractions; leave `_to_hparam_value` to its own item. |
| R9 | low | Missed-but-implied invariant: the SAVE phase reads loop-scoped locals (`epoch` at `:825`, `:828`; plus `best_macro` / `second_*` / `best_min` / `best_val_loss` / the five `best_val_*` / `best_state`). They survive only because the loop stays inline. | `sed -n '824,829p'` shows `'stopped_epoch': epoch` and `global_step=epoch`. | Name this in section 3.3 / 3.4 as the concrete failure mode of "don't extract the loop": extracting it strands `epoch` and the trackers -> `NameError` in `add_hparams`. Makes the existing fence checkable. |
| R10 | nuance | "the seeded gate catches any RNG-order perturbation a bad split introduces": the two SETUP extractions consume zero RNG, so for the proposed split the gate's real power is catching optimiser / loss-config drift (group membership, `weight_decay`, loss args) via the resulting weights, plus any accidental loop-body change. RNG-order is the over-reach fence, not the in-scope risk. | Sections 3.1 / 3.2; extractions are setup-only. | Optional reword; not an error, the gate is more than adequate. |

### Things confirmed accurate

- Phase map and ~60 `file:line` refs spot-checked: all accurate except R5 / R6.
  Signature `434-445`, loss build `487-589`, guards `507-512` / `514-518` /
  `519-525`, param walk `598-605`, AdamW `608-612`, scheduler `616-621`,
  trackers `624-638`, loop `640-795`, strict-`>` best at `751`, deepcopy
  `756`, snapshots `759-763`, early-stop `==` `793`, save / restore
  `799-801`, `val_at_best` `832-853` with the exact six keys, return `854`.
  Helpers `aux_schedule_factor 182-202`, `train_one_epoch 205-294`
  (`scheduler.step` `278`), `validate 298-374`. `__main__`: `1211-1214`,
  `1250` / `1274`, `1396` / `1399` / `1403`, `1423`.
- Cross-file refs accurate: `augmentations.py` constructors `117` / `229` (no
  RNG), `__call__` RNG at `154` / `262` / `351-352`; `adaptive_focal.py`
  `update_alpha 191`, `apply_val_gate 345`, `per_class_f1_from_counts 479`,
  `accumulate_class_counts 508`, `forward` returns `loss.mean()` (`476`),
  buffers `torch.ones` `125-126`; dataset tuple `269-271`.
- Extraction RNG-neutrality: CONFIRM. Model init runs in `build_bst_x_network`
  before `train_network` (`get_network_architecture:956` at `__main__:1396`);
  aug constructors and `AdaptiveFocalLoss.__init__` draw no RNG; first RNG
  draw is in the loop body. Extracting `_build_loss_fn` /
  `_split_param_groups` from the no-RNG setup cannot perturb a draw. `n_bones`
  is used only by `CoupledFlip` (`:472`), not by either extraction, so the
  proposed signatures are complete.
- Cross-doc refs accurate to the simplification-pass plan and the originating
  review.
- Harness refs accurate: `val_metrics_equiv.py` tests `accumulate_class_counts`
  against a verbatim one-hot golden and never calls `validate()` /
  `train_network`; `model_bitexact.py` seeds at `32-33`
  (`INPUT_SEED` / `BUILD_SEED`), `48`, `62` and does not call
  `train_network`. The split-specific proofs are the two probes.
- No AMP / `GradScaler` / `autocast` / `clip_grad` / `torch.compile` / grad-
  accum: CONFIRM (grep of `bst_x_train.py` returns only the per-class
  "accumulate" counts and comment words, no real hits).

### Key command outputs

Smoke probe (`reference_probe_smoke_train_network_cpu.py`):

```
[optim] AdamW lr=0.0005 weight_decay=0.01 (decay=27 tensors, no_decay=55)
Epoch(1/2): ... macro_f1=0.013, min_f1=0.000 - 0.45 s
Picked! => Best value 0.013
=== SMOKE OK ===
checkpoint exists: True size 7401717
val_at_best keys: ['accuracy', 'epoch', 'macro_f1', 'min_f1', 'per_class_f1', 'top2_accuracy']
```

Confirms `decay=27, no_decay=55`, checkpoint 7401717 bytes (~7.4 MB), all six
`val_at_best` keys, ~0.4 s/epoch. `val_at_best.per_class_f1` carried 12 of 14
classes on the 12-sample val set, as the doc states.

Repro probe (`reference_probe_smoke_repro_check.py`):

```
state_dict keys match: True
weights bit-exact: True | max abs weight diff: 0.0
val_at_best identical: True
```

Cross-process determinism (added check, two separate interpreters):

```
=== PROCESS A ===
STATE_DICT_SHA256: a735c94b7567b6aae5de2b0fbb176a332e86e163f6d007280ac1d133d0e68f5b
=== PROCESS B (fresh interpreter) ===
STATE_DICT_SHA256: a735c94b7567b6aae5de2b0fbb176a332e86e163f6d007280ac1d133d0e68f5b
```

(val_at_best identical between A and B too.)

Degenerate-run latent bug (added check, forced macro 0.0 every epoch):

```
Epoch(1/2): ... macro_f1=0.000, min_f1=0.000
Epoch(2/2): ... macro_f1=0.000, min_f1=0.000
RESULT: UnboundLocalError CONFIRMED -> cannot access local variable 'best_state' where it is not associated with a value
LAST FRAME: torch.save(best_state, str(save_path))  # like model.save_weights() in TF
```

---

_Originally written as the B7 split pre-analysis in the simplification pass
(merged at `18e5c2c`, 2026-06-30). The split itself landed as commit
`a0ffc89`._

