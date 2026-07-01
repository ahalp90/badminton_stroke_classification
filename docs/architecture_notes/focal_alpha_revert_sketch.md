# Adaptive-focal alpha extensions: pair_caps + val-improvability gate

Two extensions sat on top of `AdaptiveFocalLoss`'s base CDB-F1 update between May and June 2026, both since retired. This doc maps every touch point so either can come back without a fresh code review.

A quick disambiguation up front. The term `focal_alpha_revert`, as used in commit messages, run notes, and the `focal_alpha_revert_overallocated` ablation ID, always refers to the **val-improvability gate** specifically. `pair_caps` is a separate feature that lives in the same loss class.

The two share the renorm contract (mean alpha = 1.0) at the seam and arrive together inside the `adaptive_focal` config block. Past that, they target different signals and do different things to alpha.

**Pinned at:** commit `11045620b50ee244cccaf8d50f14cb4827a045b7` (short `1104562`, 2026-07-01).

- Every line number in [Code touch points](#code-touch-points) is captured at that commit.
- Every snippet under [Verbatim source](#verbatim-source) is reproduced at that commit.
- To pull a file's state at re-wire time: `git show 1104562:src/bst_x/loss/adaptive_focal.py` (etc.).
- When the doc and the working tree disagree on a line number, the verbatim section is the source of truth.

## Contents
- [What each mechanism does](#what-each-mechanism-does)
- [Re-wire checklist](#re-wire-checklist)
- [Code touch points](#code-touch-points)
- [Verbatim source](#verbatim-source)
- [Config dict shape](#config-dict-shape)
- [Runtime contract](#runtime-contract)
- [Math contracts](#math-contracts)
- [Manifest emission](#manifest-emission)
- [Tests](#tests)
- [Run history and retirement notes](#run-history-and-retirement-notes)

## What each mechanism does

Both extensions run after `update_alpha`'s standard `(1 - F1_running)^tau` step and its renorm to mean 1.0. They modify the alpha vector that the base CDB-F1 weighting (`class_f1_focal_design.md`) has just refreshed; the per-sample `(1 - p_t)^gamma` term stays out of scope.

| Feature | When it fires | Signal | Effect on alpha |
|---|---|---|---|
| `pair_caps` | every epoch, unconditional | named (numer, denom) class pair | bumps `alpha[numer]` up to `ratio * alpha[denom]` if below; absorbs the bump uniformly across the other `n - 2` classes |
| val-improvability gate | every epoch, decay only inside `(min_epochs_before_gating, freeze_epoch)` | smoothed per-class val F1 plateau | decays over-allocated classes (`alpha > 1`) toward 1.0 by a ramped fraction; hands the freed budget to classes that were not pulled this epoch |

The two mechanisms attack alpha from different angles:

- The gate is the **dynamic plateau-detector**. It watches a smoothed val F1 per class and pulls plateaued, over-allocated classes back toward the mean.
- `pair_caps` is the **static ratio enforcer** between named classes. It addresses confusion-pair imbalances the scalar-per-class CDB signal handles poorly (built for the `smash` / `wrist_smash` pair).

Both keep mean alpha = 1.0 by construction, so the overall loss scale stays comparable to uniform CE.

## Re-wire checklist

If you want either back, in order:

1. **Loss class.** Restore the constructor args and method bodies in `src/bst_x/loss/adaptive_focal.py`. Line ranges live in [Code touch points](#code-touch-points); full snippets in [Verbatim source](#verbatim-source).

   The class signature gains three args: `pair_caps`, `val_improvability_gate`, and `n_epochs` (the last is gate-only). Inside `__init__`, two wire-up calls have to land alongside the new methods:

   - `self.pair_caps = self._resolve_pair_caps(...)` at L117-119
   - `self._init_val_improvability_gate(...)` at L134

   Pasting `_resolve_pair_caps`, `_init_val_improvability_gate`, and `apply_val_gate` while leaving `__init__` untouched defines the methods but leaves them uncalled; first construction then raises `AttributeError` at the gate-on print.

2. **Hyp.** Restore the `pair_caps` example block on the `adaptive_focal` dict, plus the `use_val_improvability_gate` flag and the `val_improvability_gate` dict on the `Hyp` NamedTuple. The validation paragraph above each is worth keeping in place.

3. **Train script wiring.** Inside `_build_loss_fn` (leading underscore), three things go back:

   - **The gate-specific fail-loud guard** (`use_val_improvability_gate=True requires adaptive_focal`) at the top of the function. The other two guards in the L401-419 block (`adaptive_focal` excludes `class_weights`; `adaptive_focal` forces `label_smoothing=0`) are intrinsic to base CDB-F1 and survive an extension rip, so they will already be in the function. Pasting the full L401-419 block verbatim creates duplicates; lift only the first guard.
   - **The new kwargs** on the `AdaptiveFocalLoss(...)` call: `pair_caps=`, `val_improvability_gate=`, `n_epochs=`.
   - **The two diagnostic prints**: the resolved-triples line and the gate-on knob summary.

4. **Per-epoch hooks.** Inside `train_network`'s epoch loop:

   - Keep the `update_alpha(train_per_class_f1)` call (it handles `pair_caps` internally).
   - Re-add the `apply_val_gate(f1_per_class, present)` call between `validate(...)` and the diagnostic prints. Order matters here: `apply_val_gate` reads `self.alpha` after `update_alpha` has refreshed it.

   **Dependency on `validate()`.** The gate reads `f1_per_class` (slot 4) and `present` (slot 5) from `validate()`'s return tuple. The verbatim snippet shows today's 7-tuple unpack `val_loss, f1_score_avg, f1_score_min, f1_per_class, present, val_accuracy, val_top2 = validate(...)`. Those two slots are owned by `validate()` independently of the gate (the `val_at_best_macro_epoch` snapshot uses them too), so check the current `validate()` signature in `bst_x_train.py` before pasting. If a future cleanup trimmed those slots while the gate was retired, restore them first.

5. **TB scalars.** Two things go back:

   - The per-class `Revert/{c}` scalar lives inside the `if loss_fn.val_gate_enabled:` branch. `Alpha/{c}` lives outside that branch and stays regardless.
   - Two prose lists of TB scalar prefixes need `Revert/*` added: `src/bst_x/aim_backfill.py:10` (docstring listing) and `src/bst_x/run_tracker.md:138-139` (prose listing). The aim-backfill code itself iterates every scalar tag, so it picks `Revert/*` up on its own; the docs just go stale until updated.

6. **CLI flag.** Re-add `--val-improvability-gate` (with `BooleanOptionalAction`) and the `cell_overrides['use_val_improvability_gate']` line. `collation_runner.py` already forwards the flag once it's back in `bst_x_train.py`.

7. **Tests.** Restore the nine `pair_cap` cases plus the five gate cases in `tests/test_adaptive_focal.py`. The fixtures (`class_names_3`, `class_names_14`) are shared with the base-loss suite, so they stay regardless.

8. **Manifest.** No code change needed. `track_run` serialises the whole `Hyp` NamedTuple via `_asdict`; once `Hyp` has the fields back, the manifest emits them automatically.

## Code touch points

Line numbers are pinned to commit `1104562` (full SHA in the banner above). If they've drifted, the same code is reproduced verbatim under [Verbatim source](#verbatim-source).

### `src/bst_x/loss/adaptive_focal.py` (530 lines)

The whole `AdaptiveFocalLoss` class lives here. Both extensions are bolted into it.

**pair_caps**

| Lines | What |
|---|---|
| 14-18 (module docstring) | one-paragraph summary of the extension |
| 66-73 | `pair_caps` constructor-arg docstring (rule shape + `n - 2` redistribution) |
| 95 | constructor signature `pair_caps: list[dict] \| None = None` |
| 117-119 | `self.pair_caps: list[tuple[int, int, float]]` resolved at construction |
| 139-188 | `_resolve_pair_caps` staticmethod: name-to-index lookup, `n_classes >= 3` guard, ratio bounds check, `numer != denom` check |
| 218-241 | per-epoch enforcement at the end of `update_alpha`; sequential application across rules; defensive `clamp_(min=1e-8)` whenever `pair_caps` is configured (non-empty), then `self.epoch += 1` |

**val-improvability gate**

| Lines | What |
|---|---|
| 76-83 | `val_improvability_gate` constructor-arg docstring |
| 97-98 | constructor signature `val_improvability_gate: dict \| None = None`, `n_epochs: int \| None = None` |
| 131-134 | `_init_val_improvability_gate(...)` call inside `__init__` (registers buffers before any `.to(device)`) |
| 243-342 | `_init_val_improvability_gate`: validation (six knob ranges, freeze-window non-empty check, `min_epochs_before_gating >= warm_up_epochs`), five per-class buffers (`gate_smoothed_val_f1`, `gate_best_smoothed_val_f1`, `gate_epochs_since_improvement`, `gate_revert_fraction`, `gate_val_f1_seeded`) |
| 344-451 | `apply_val_gate(val_per_class_f1, present)`: per-class smoother + best update (always-on, even outside the window), then INSIDE the window only the plateau-counter tick and the revert-fraction ramp, then the one-sided pull and recipient-scaled budget hand-off |

Helpers that belong to the base loss and feed both extensions:

- L479-505 `per_class_f1_from_counts(tp, fp, fn, eps)`
- L508-530 `accumulate_class_counts(preds, labels, n_classes)`

### `src/bst_x/bst_x_train.py`

| Lines | What |
|---|---|
| 49-53 | import `AdaptiveFocalLoss`, `accumulate_class_counts`, `per_class_f1_from_counts` |
| 101-127 | `Hyp.adaptive_focal` dict default, with the commented `pair_caps` example block at 104-114 |
| 128-147 | `Hyp.use_val_improvability_gate` toggle + `Hyp.val_improvability_gate` dict with the six tuned defaults |
| 275, 317 | `accumulate_class_counts(...)` calls: L275 inside `train_one_epoch` (every batch goes through `coupled_flip` + `constrained_jitter` before the single forward, so there is no separate clean-vs-jittered structure) and L317 inside `validate` |
| 374-376 | `_build_loss_fn` docstring naming the three fail-loud guards |
| 401-419 | the three guards: `use_val_improvability_gate` needs `adaptive_focal` (GATE-SPECIFIC; only this one needs restoring on re-wire), plus `adaptive_focal` excludes `class_weights` and `adaptive_focal` forces `label_smoothing=0` (intrinsic to base CDB-F1, will already exist) |
| 420-437 | `AdaptiveFocalLoss(...)` construction in `_build_loss_fn` with `pair_caps=`, `val_improvability_gate=`, `n_epochs=` |
| 438-454 | resolved `pair_caps` triples print (uses `loss_fn.pair_caps` indices) |
| 455-466 | gate-on print (six knob values + the resolved gating window) |
| 627-631 | end-of-epoch `per_class_f1_from_counts(...)` then `loss_fn.update_alpha(...)` |
| 633-639 | `validate(...)` returns `f1_per_class`, `present` (gate inputs) |
| 640-645 | `apply_val_gate(f1_per_class, present)` call, only when `loss_fn.val_gate_enabled` |
| 651-661 | post-epoch `alpha bot3 / top3` print (works for both mechanisms; reads `loss_fn.alpha`) |
| 696-710 | per-class TB scalars: `Alpha/{c}` always; `Revert/{c}` only when `val_gate_enabled` |
| 1187-1194 | argparse `--val-improvability-gate` with `BooleanOptionalAction` |
| 1245-1246 | `cell_overrides['use_val_improvability_gate'] = args.val_improvability_gate` plumbing into `hyp._replace` |

### `src/bst_x/collation_runner.py`

| Lines | What |
|---|---|
| 57-61 | forwards the gate flag from a cell config to the child `bst_x_train` invocation (`--val-improvability-gate` when True, `--no-val-improvability-gate` when False, no override when absent) |

### `src/bst_x/run_tracker.py`

No mechanism-specific code; manifest emission is generic.

The serialisation path, end to end:

- **Caller side** (`bst_x_train.py:1322-1326`): builds `config_payload = dict(hyp._asdict())` and passes it as `config=`.
- **`run_tracker._config_to_dict`** (`run_tracker.py:157-164`): the `Mapping` branch at L158-159 fires (because `config_payload` is already a dict) and returns `dict(config)` directly.
- **Alternate path** in the module-docstring example (`run_tracker.py:13`): passes the NamedTuple itself, which would take the `_asdict` branch at L160-161 instead. Both paths produce the same dict.

The two mechanisms then round-trip cleanly:

- `pair_caps` rules emit as a `pair_caps:` list of `{numer, denom, ratio}` dicts.
- Gate state emits as `use_val_improvability_gate:` plus the six-key `val_improvability_gate:` dict.

### `tests/test_adaptive_focal.py`

See [Tests](#tests) for the per-mechanism case list.

## Verbatim source

Reproduced from commit `1104562`. Use `git show 1104562:<path>` to pull the original file. Code fences are unlabelled (no syntax highlighting) so they render clean in the terminal.

### `src/bst_x/loss/adaptive_focal.py`

**Constructor signature, relevant new args** (L86-99)

```
    def __init__(
        self,
        n_classes: int,
        class_names: list[str],
        tau: float = 1.0,
        gamma: float = 1.0,
        momentum: float = 0.9,
        warm_up_epochs: int = 5,
        f1_floor: float = 0.0,
        pair_caps: list[dict] | None = None,
        device: torch.device | str | None = None,
        val_improvability_gate: dict | None = None,
        n_epochs: int | None = None,
    ):
```

**`__init__` body, with the two extension wire-up calls** (L100-138)

The wire-up calls turn the new methods into actual instance state:

- `self.pair_caps = self._resolve_pair_caps(...)` at L117-119
- `self._init_val_improvability_gate(...)` at L134

These calls are what put `loss_fn.pair_caps` (the resolved triples) and `loss_fn.val_gate_enabled` on the instance. With them in place, the gate-on print at L455 (`if loss_fn.val_gate_enabled:`) and the resolved-triples print at L440-446 both work; without them, the print at L455 raises `AttributeError: 'AdaptiveFocalLoss' object has no attribute 'val_gate_enabled'` on first construction.

Buffer-registration order also matters. The gate's per-class buffers have to register before any `.to(device)` move, so they ride across with `f1_running` and `alpha`.

```
        super().__init__()
        if len(class_names) != n_classes:
            raise ValueError(
                f'len(class_names)={len(class_names)} must equal n_classes={n_classes}'
            )
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'momentum must be in [0, 1); got {momentum}')

        self.n_classes = n_classes
        self.class_names = list(class_names)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.momentum = float(momentum)
        self.warm_up_epochs = int(warm_up_epochs)
        self.f1_floor = float(f1_floor)
        # Resolve pair-cap names to indices once at construction so update_alpha
        # stays index-only. Stored as (numer_idx, denom_idx, ratio) triples.
        self.pair_caps: list[tuple[int, int, float]] = self._resolve_pair_caps(
            pair_caps, class_names, n_classes
        )

        # Init f1_running to 1.0 (model-is-perfect prior); update_alpha mixes
        # in real readings via EMA each epoch. While epoch < warm_up_epochs
        # forward() ignores alpha and uses uniform weights, so the EMA can
        # absorb a few epochs of real signal before its shape applies.
        self.register_buffer('f1_running', torch.ones(n_classes))
        self.register_buffer('alpha', torch.ones(n_classes))
        # Plain int because state_dict persistence isn't needed (each serial
        # is a fresh model + fresh loss instance; no cross-serial resume).
        self.epoch = 0

        # Val-improvability gate (off unless val_improvability_gate is a dict).
        # Registers its own per-class buffers, so set it up before the device
        # move below so every buffer is carried across together.
        self._init_val_improvability_gate(val_improvability_gate, n_epochs)

        if device is not None:
            self.to(device)
```

**`_resolve_pair_caps` staticmethod** (L139-188)

```
    @staticmethod
    def _resolve_pair_caps(
        pair_caps: list[dict] | None,
        class_names: list[str],
        n_classes: int,
    ) -> list[tuple[int, int, float]]:
        """Validate pair-cap rules and resolve class names to indices.

        Each rule must be a dict with keys ``'numer'``, ``'denom'`` (class
        names that exist in ``class_names``) and ``'ratio'`` (float in
        ``(0, 1]``). Returns a list of ``(numer_idx, denom_idx, ratio)``
        triples; an empty list when ``pair_caps`` is None or empty.

        Raises ``ValueError`` on any malformed rule (unknown name, ratio out
        of range, numer == denom).
        """
        if not pair_caps:
            return []
        if n_classes < 3:
            # Pair-cap subtracts the bump across n_classes - 2 other classes;
            # n=2 would have nowhere to redistribute and the maths collapses.
            raise ValueError(
                f'pair_caps requires n_classes >= 3; got {n_classes}'
            )

        name_to_idx = {name: i for i, name in enumerate(class_names)}
        resolved: list[tuple[int, int, float]] = []
        for cap in pair_caps:
            numer = cap['numer']
            denom = cap['denom']
            ratio = float(cap['ratio'])
            if numer not in name_to_idx:
                raise ValueError(
                    f"pair_cap numer '{numer}' not in class_names {class_names}"
                )
            if denom not in name_to_idx:
                raise ValueError(
                    f"pair_cap denom '{denom}' not in class_names {class_names}"
                )
            if numer == denom:
                raise ValueError(
                    f"pair_cap numer and denom must differ; both are '{numer}'"
                )
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    f'pair_cap ratio must be in (0, 1]; got {ratio} for '
                    f"'{numer}' / '{denom}'"
                )
            resolved.append((name_to_idx[numer], name_to_idx[denom], ratio))
        return resolved
```

**pair_caps enforcement, the tail of `update_alpha`** (L218-241)

```
        # Pair caps: enforce alpha[numer] >= ratio * alpha[denom] for each
        # registered pair. Bump cost is absorbed across the (n - 2) classes
        # outside the pair, so mean alpha stays exactly 1.0 by construction.
        # Sequential application: later caps see updated alphas; ordering
        # matters only when caps share a class.
        n_other = self.n_classes - 2
        for numer_idx, denom_idx, ratio in self.pair_caps:
            target = ratio * self.alpha[denom_idx]
            bump = target - self.alpha[numer_idx]
            if bump <= 0:
                continue
            self.alpha[numer_idx] = target
            other_mask = torch.ones(
                self.n_classes, dtype=torch.bool, device=self.alpha.device
            )
            other_mask[numer_idx] = False
            other_mask[denom_idx] = False
            self.alpha[other_mask] -= bump / n_other
        # Guard against the rare case where a large bump or stacked caps push
        # an "other" class negative; bump is typically ~0.4 spread over ~12
        # classes (~0.03 each) so this clamp is defensive, not load-carrying.
        if self.pair_caps:
            self.alpha.clamp_(min=1e-8)
        self.epoch += 1
```

**`_init_val_improvability_gate`** (L243-342)

```
    def _init_val_improvability_gate(
        self,
        gate_cfg: dict | None,
        n_epochs: int | None,
    ) -> None:
        """Set up (or disable) the val-improvability gate.

        The gate decays a class's alpha back toward the renorm mean of 1.0 once
        that class has stopped improving on val, freeing the over-allocated
        budget for classes still climbing. It reads only a smoothed per-class
        val F1 (no backprop through val): val-driven scheduling, the same family
        as ReduceLROnPlateau. ``self.alpha`` is renormalised to mean 1.0, so a
        class is over-allocated exactly when its alpha sits above 1.0; the gate
        only pulls those down. Motivating per-class arcs:
        ``docs/architecture_notes/alpha_arc_analysis/``.

        :param gate_cfg: ``None`` disables the gate; a dict engages it. Keys (all
            optional, defaults shown): ``val_f1_smoothing_factor`` (0.9, EMA
            retention on val F1), ``improvement_margin`` (0.015, how much the
            smoothed val F1 must beat its running best to count as a new high),
            ``patience_epochs`` (15, epochs with no new high before decay starts),
            ``min_epochs_before_gating`` (10, no decay before this epoch),
            ``revert_step_per_epoch`` (0.2, fraction of the way to the mean the
            alpha moves per epoch once decaying), ``stop_gating_after_fraction``
            (0.75, freeze the gate past this fraction of the run, leaving the
            late-anneal blooms alone).
        :param n_epochs: total training epochs; required when the gate is on, to
            turn ``stop_gating_after_fraction`` into an absolute freeze epoch.
        """
        self.val_gate_enabled = gate_cfg is not None
        if not self.val_gate_enabled:
            return
        if n_epochs is None:
            raise ValueError(
                'val_improvability_gate is enabled but n_epochs is None; the '
                'gate needs n_epochs to resolve stop_gating_after_fraction into '
                'an absolute freeze epoch.'
            )

        self.gate_val_f1_smoothing_factor = float(gate_cfg.get('val_f1_smoothing_factor', 0.9))
        self.gate_improvement_margin = float(gate_cfg.get('improvement_margin', 0.015))
        self.gate_patience_epochs = int(gate_cfg.get('patience_epochs', 15))
        self.gate_min_epochs_before_gating = int(gate_cfg.get('min_epochs_before_gating', 10))
        self.gate_revert_step_per_epoch = float(gate_cfg.get('revert_step_per_epoch', 0.2))
        stop_gating_after_fraction = float(gate_cfg.get('stop_gating_after_fraction', 0.75))

        if not 0.0 <= self.gate_val_f1_smoothing_factor < 1.0:
            raise ValueError(
                f'val_f1_smoothing_factor must be in [0, 1); got '
                f'{self.gate_val_f1_smoothing_factor}.'
            )
        if self.gate_improvement_margin < 0.0:
            raise ValueError(
                f'improvement_margin must be >= 0; got {self.gate_improvement_margin}.'
            )
        if self.gate_patience_epochs < 1:
            raise ValueError(
                f'patience_epochs must be >= 1; got {self.gate_patience_epochs}.'
            )
        if self.gate_min_epochs_before_gating < 0:
            raise ValueError(
                f'min_epochs_before_gating must be >= 0; got '
                f'{self.gate_min_epochs_before_gating}.'
            )
        if not 0.0 < self.gate_revert_step_per_epoch <= 1.0:
            raise ValueError(
                f'revert_step_per_epoch must be in (0, 1]; got '
                f'{self.gate_revert_step_per_epoch}.'
            )
        if not 0.0 < stop_gating_after_fraction <= 1.0:
            raise ValueError(
                f'stop_gating_after_fraction must be in (0, 1]; got '
                f'{stop_gating_after_fraction}.'
            )
        self.gate_freeze_epoch = round(stop_gating_after_fraction * n_epochs)
        if self.gate_freeze_epoch <= self.gate_min_epochs_before_gating:
            raise ValueError(
                f'gate freeze epoch ({self.gate_freeze_epoch}) must exceed '
                f'min_epochs_before_gating ({self.gate_min_epochs_before_gating}); '
                f'the gating window is empty. Check stop_gating_after_fraction '
                f'({stop_gating_after_fraction}) against n_epochs ({n_epochs}).'
            )
        if self.gate_min_epochs_before_gating < self.warm_up_epochs:
            raise ValueError(
                f'min_epochs_before_gating ({self.gate_min_epochs_before_gating}) '
                f'must be >= adaptive-focal warm_up_epochs ({self.warm_up_epochs}): '
                f'forward() uses uniform alpha during focal warm-up, so gating '
                f'earlier would ramp the revert on weights training is not yet '
                f'using.'
            )

        # Per-class gate state. best_smoothed_val_f1 starts at -1.0 so the first
        # real reading (F1 >= 0) always registers as a new high and seeds it.
        self.register_buffer('gate_smoothed_val_f1', torch.zeros(self.n_classes))
        self.register_buffer('gate_best_smoothed_val_f1', torch.full((self.n_classes,), -1.0))
        self.register_buffer('gate_epochs_since_improvement', torch.zeros(self.n_classes))
        self.register_buffer('gate_revert_fraction', torch.zeros(self.n_classes))
        self.register_buffer(
            'gate_val_f1_seeded', torch.zeros(self.n_classes, dtype=torch.bool)
        )
```

**`apply_val_gate`** (L344-451)

```
    @torch.no_grad()
    def apply_val_gate(
        self,
        val_per_class_f1: torch.Tensor,
        present: torch.Tensor,
    ) -> None:
        """Decay plateaued classes' alpha toward the mean, from a val signal.

        Call once per epoch AFTER ``validate`` (so it sees this epoch's val F1)
        and after ``update_alpha`` (which has just refreshed ``self.alpha`` from
        train F1). The smoother and the running best update every epoch from the
        first val reading; the patience counter and the alpha revert only act
        inside the window ``min_epochs_before_gating < epoch < freeze_epoch``, so
        the early adaptive boost and the late-anneal blooms are left alone.

        A class that has not beaten its smoothed-val best by ``improvement_margin``
        for ``patience_epochs`` ramps its alpha toward 1.0 by
        ``revert_step_per_epoch`` of the full gap per epoch; it ramps back the
        moment it sets a new high, so a wrongly-flagged slow climber recovers.
        Only over-allocated classes (alpha > 1.0) are pulled; below-mean classes
        contribute ``max(0, alpha - 1) = 0`` and are untouched. The budget freed
        by the pull goes only to the classes that were NOT pulled this epoch (the
        climbers and the below-mean classes), so a de-prioritised class lands at
        its reverted level (exactly 1.0 when fully reverted) and does not claw
        back the budget it surrendered; mean alpha stays 1.0 (loss scale kept).

        The revert is recomputed each epoch against that epoch's fresh
        ``update_alpha`` base, so ``gate_revert_fraction`` is the only state that
        ramps; there is no cumulative double-decay. No-op if the gate is disabled.

        :param val_per_class_f1: shape ``[n_classes]`` val F1 for this epoch.
        :param present: shape ``[n_classes]`` bool, True where the class had at
            least one val sample (so its F1 is real). Absent classes are skipped.
        """
        if not self.val_gate_enabled:
            return

        val_f1 = val_per_class_f1.to(self.gate_smoothed_val_f1)
        smoothing = self.gate_val_f1_smoothing_factor
        revert_step = self.gate_revert_step_per_epoch
        in_gating_window = (
            self.gate_min_epochs_before_gating < self.epoch < self.gate_freeze_epoch
        )

        for c in range(self.n_classes):
            if not bool(present[c]):
                continue                          # no val signal: leave this class's alpha
            # Smoothed val F1: seed on the first reading, then causal EMA.
            if not bool(self.gate_val_f1_seeded[c]):
                self.gate_smoothed_val_f1[c] = val_f1[c]
                self.gate_val_f1_seeded[c] = True
            else:
                self.gate_smoothed_val_f1[c] = (
                    smoothing * self.gate_smoothed_val_f1[c]
                    + (1.0 - smoothing) * val_f1[c]
                )
            improved = bool(
                self.gate_smoothed_val_f1[c]
                > self.gate_best_smoothed_val_f1[c] + self.gate_improvement_margin
            )
            if improved:
                self.gate_best_smoothed_val_f1[c] = self.gate_smoothed_val_f1[c]

            # Counter + revert ramp only inside the gating window; the smoother
            # and best above always run so the baseline is real by the time the
            # window opens, and the revert holds (does not grow) in the tail.
            if not in_gating_window:
                continue
            if improved:
                self.gate_epochs_since_improvement[c] = 0.0
            else:
                self.gate_epochs_since_improvement[c] += 1.0
            plateaued = bool(
                self.gate_epochs_since_improvement[c] >= self.gate_patience_epochs
            )
            # Ramp the revert up only for a class that is both plateaued AND
            # currently over-allocated (base alpha > 1.0, set by update_alpha
            # this epoch); ramp it back down otherwise. A saturated below-mean
            # class is plateaued too, but the one-sided revert below is a no-op
            # on it, so there is nothing to reclaim and Revert/{c} stays at 0,
            # keeping the diagnostic honest to the classes actually pulled down.
            over_allocated = bool(self.alpha[c] > 1.0)
            current_revert = self.gate_revert_fraction[c].item()
            if plateaued and over_allocated:
                self.gate_revert_fraction[c] = min(current_revert + revert_step, 1.0)
            else:
                self.gate_revert_fraction[c] = max(current_revert - revert_step, 0.0)

        # One-sided revert: pull each over-allocated class toward the mean by its
        # revert fraction, then hand the freed budget ONLY to the classes that
        # were not pulled this epoch (the climbers and the below-mean classes),
        # never back to the pulled classes themselves. So a de-prioritised class
        # lands at its reverted level (exactly 1.0 when fully reverted), not above
        # it: it does not claw back the budget it just surrendered. Scaling the
        # recipients to absorb exactly the freed amount keeps the sum at
        # n_classes (mean alpha 1.0), preserving the loss scale as the base
        # renorm in update_alpha does. (alpha.sum() is n_classes on entry, set by
        # update_alpha's renorm earlier this epoch.)
        over_allocation = (self.alpha - 1.0).clamp(min=0.0)        # [n_classes]; 0 at/below mean
        pulled = self.gate_revert_fraction * over_allocation        # amount removed per class
        self.alpha.sub_(pulled)
        freed = pulled.sum()
        recipients = pulled == 0.0                                  # classes not pulled this epoch
        recipient_sum = self.alpha[recipients].sum()
        if bool(freed > 0.0) and bool(recipient_sum > 0.0):
            self.alpha[recipients] = (
                self.alpha[recipients] * (recipient_sum + freed) / recipient_sum
            )
```

### `src/bst_x/bst_x_train.py`

**`Hyp.adaptive_focal` block with the commented `pair_caps` example** (L101-127)

```
    # Class-F1-driven adaptive focal loss (CDB-F1). Mutually exclusive with
    # class_weights, and forces label_smoothing=0 (LS contaminates focal's
    # hardness estimate). None disables; pass a dict to engage:
    #   adaptive_focal={
    #       'tau': 1.0, 'gamma': 1.0, 'momentum': 0.9,
    #       'warm_up_epochs': 5, 'f1_floor': 0.0,
    #       # Optional pair-cap rules for known confusion pairs the scalar CDB
    #       # signal can't model. Each rule enforces alpha[numer] >= ratio *
    #       # alpha[denom] after the standard renormalisation, with the bump
    #       # absorbed across the other (n - 2) classes so mean alpha stays 1.0.
    #       'pair_caps': [
    #           {'numer': 'smash', 'denom': 'wrist_smash', 'ratio': 0.7},
    #       ],
    #   }
    # Full design + paper-verified equations: docs/architecture_notes/class_f1_focal_design.md.
    adaptive_focal: dict | None = {
        # First-run sweet spot from run_20260501_164658: tau=1, gamma=1.
        # All four CDB knob variants (gamma=0, tau=0.5, pair-cap, gamma=2)
        # traded wrist_smash back for smash without macro moving, so this
        # combo holds the floor-lift sweet spot (+8.7 pp wrist_smash on the
        # LS=0.1 baseline). Active default for the capacity-bump runs.
        'tau': 1.0,
        'gamma': 1.0,
        'momentum': 0.9,
        'warm_up_epochs': 5,
        'f1_floor': 0.0,
    }
```

**`Hyp.use_val_improvability_gate` + `Hyp.val_improvability_gate`** (L128-147)

```
    # Val-improvability gate over the adaptive-focal alpha. Off by default;
    # flip on with use_val_improvability_gate=True or --val-improvability-gate.
    # Once a class stops improving on val it decays that class's alpha back
    # toward the renorm mean of 1.0, freeing the over-allocated budget for
    # classes still climbing (the adaptive_focal alpha is driven by train F1,
    # which keeps rising on plateaued classes via overfitting; this reads val to
    # catch that). Requires adaptive_focal (it modulates that alpha). The dict
    # stays visible here even when disabled so the knobs are easy to find/tune.
    # Defaults are the ones derived in
    # docs/architecture_notes/alpha_arc_analysis/ (macro plateaus ~e26-31,
    # cross_court_net_shot needs a patience >= its ~15-epoch new-high interval).
    use_val_improvability_gate: bool = False
    val_improvability_gate: dict = {
        'val_f1_smoothing_factor': 0.9,    # EMA retention on val F1 (~6.6-epoch half-life)
        'improvement_margin': 0.015,       # smoothed val must beat its best by this to count
        'patience_epochs': 15,             # epochs with no new high before decay starts
        'min_epochs_before_gating': 10,    # no decay before this epoch (keep the early boost)
        'revert_step_per_epoch': 0.2,      # fraction of the gap to the mean reverted per epoch
        'stop_gating_after_fraction': 0.75,  # freeze past 0.75*n_epochs (protect anneal blooms)
    }
```

**Three fail-loud guards in `build_loss_fn`** (L401-419)

```
    if hyp.use_val_improvability_gate and hyp.adaptive_focal is None:
        raise ValueError(
            'use_val_improvability_gate=True requires adaptive_focal (the gate '
            'decays the adaptive-focal alpha; with plain CE there is no alpha to '
            'modulate). Set adaptive_focal to a config dict or disable the gate.'
        )
    if hyp.adaptive_focal is not None:
        if hyp.class_weights:
            raise ValueError(
                'adaptive_focal and class_weights are mutually exclusive; '
                'set one of them to None.'
            )
        if hyp.label_smoothing != 0.0:
            raise ValueError(
                'adaptive_focal requires label_smoothing=0.0 (LS softens '
                'targets so confident-correct samples have p_t < 1.0, '
                "contaminating focal's per-sample hardness signal). "
                f'Got label_smoothing={hyp.label_smoothing}.'
            )
```

**`AdaptiveFocalLoss(...)` construction** (L420-437)

```
        af_cfg = hyp.adaptive_focal
        loss_fn = AdaptiveFocalLoss(
            n_classes=n_classes,
            class_names=class_ls,
            tau=af_cfg.get('tau', 1.0),
            gamma=af_cfg.get('gamma', 1.0),
            momentum=af_cfg.get('momentum', 0.9),
            warm_up_epochs=af_cfg.get('warm_up_epochs', 5),
            f1_floor=af_cfg.get('f1_floor', 0.0),
            pair_caps=af_cfg.get('pair_caps'),
            # Gate config only when the flag is on; None leaves the gate off.
            # n_epochs lets the gate resolve stop_gating_after_fraction.
            val_improvability_gate=(
                hyp.val_improvability_gate if hyp.use_val_improvability_gate else None
            ),
            n_epochs=hyp.n_epochs,
            device=device,
        )
```

**Resolved `pair_caps` print and gate-on print** (L438-466)

```
        # Print resolved pair_caps as triples (rather than the dict spec) so the
        # log shows the index lookup succeeded against the active class list.
        pair_cap_str = (
            ', '.join(
                f'{class_ls[n]}/{class_ls[d]}>={r:.2f}'
                for n, d, r in loss_fn.pair_caps
            )
            if loss_fn.pair_caps else 'none'
        )
        print(
            f"[loss] adaptive focal (CDB-F1): "
            f"tau={loss_fn.tau}, gamma={loss_fn.gamma}, "
            f"momentum={loss_fn.momentum}, "
            f"warm_up_epochs={loss_fn.warm_up_epochs}, "
            f"f1_floor={loss_fn.f1_floor}, "
            f"pair_caps=[{pair_cap_str}]"
        )
        if loss_fn.val_gate_enabled:
            print(
                f"[loss] val-improvability gate ON: "
                f"smoothing={loss_fn.gate_val_f1_smoothing_factor}, "
                f"margin={loss_fn.gate_improvement_margin}, "
                f"patience={loss_fn.gate_patience_epochs}, "
                f"min_epochs_before_gating={loss_fn.gate_min_epochs_before_gating}, "
                f"revert_step={loss_fn.gate_revert_step_per_epoch}, "
                f"freeze_epoch={loss_fn.gate_freeze_epoch} "
                f"(gating window epochs "
                f"{loss_fn.gate_min_epochs_before_gating + 1}-{loss_fn.gate_freeze_epoch - 1})"
            )
```

**Per-epoch hooks inside `train_network`'s epoch loop** (L627-645)

```
        # End-of-epoch per-class train F1 feeds AdaptiveFocalLoss; otherwise
        # the values are still computed (cheap) and logged to TB for context.
        train_per_class_f1 = per_class_f1_from_counts(train_tp, train_fp, train_fn)
        if isinstance(loss_fn, AdaptiveFocalLoss):
            loss_fn.update_alpha(train_per_class_f1)

        val_loss, f1_score_avg, f1_score_min, f1_per_class, present, val_accuracy, val_top2 = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            n_classes=n_classes,
        )
        # Val-improvability gate: decay plateaued classes' alpha toward the mean
        # using this epoch's val F1. Must run after validate (needs the val F1)
        # and after update_alpha above (which refreshed the base alpha from train
        # F1); the gated alpha then drives next epoch's training. No-op when off.
        if isinstance(loss_fn, AdaptiveFocalLoss) and loss_fn.val_gate_enabled:
            loss_fn.apply_val_gate(f1_per_class, present)
```

**Per-class TB scalars** (L703-710)

```
            if isinstance(loss_fn, AdaptiveFocalLoss):
                writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)
                # Per-class revert fraction (0 = full adaptive alpha, 1 = pulled
                # all the way to the renorm mean) so the gate's action is visible.
                if loss_fn.val_gate_enabled:
                    writer.add_scalar(
                        f'Revert/{c}', loss_fn.gate_revert_fraction[i].item(), epoch
                    )
```

**`--val-improvability-gate` CLI flag** (L1187-1194)

```
    # Enable/disable the val-improvability alpha gate, overriding the Hyp default.
    # --val-improvability-gate turns it on, --no-val-improvability-gate off;
    # absent leaves the module default (off). Requires adaptive_focal.
    parser.add_argument(
        '--val-improvability-gate',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
```

**Cell-override plumbing** (L1245-1246)

```
    if args.val_improvability_gate is not None:
        cell_overrides['use_val_improvability_gate'] = args.val_improvability_gate
```

### `src/bst_x/collation_runner.py`

**Gate flag forwarding** (L57-61)

```
    # Optional val-improvability gate toggle. Present-and-True turns it on,
    # present-and-False forces it off; absent leaves the bst_x_train Hyp default.
    if cell.get('use_val_improvability_gate') is not None:
        cmd += ['--val-improvability-gate' if cell['use_val_improvability_gate']
                else '--no-val-improvability-gate']
```

## Config dict shape

The `adaptive_focal` dict on `Hyp` carries `pair_caps` inline. The gate sits on two sibling fields beside it: a boolean flag (`use_val_improvability_gate`) and the dict of knobs (`val_improvability_gate`).

The exact shape, copy-pasteable, is below.

One note on the `pair_caps` example: it uses `'smash'` / `'wrist_smash'`, which only exist together in the 14-class taxonomies (`une_v1_14`, `bst_24` baseline). For any other taxonomy, edit the names to ones that appear in the active class list; `_resolve_pair_caps` raises with a clear "not in class_names" message if a name is missing.

```python
adaptive_focal: dict | None = {
    'tau': 1.0,
    'gamma': 1.0,
    'momentum': 0.9,
    'warm_up_epochs': 5,
    'f1_floor': 0.0,
    # Optional pair-cap rules for known confusion pairs the scalar CDB
    # signal can't model. Each rule enforces alpha[numer] >= ratio *
    # alpha[denom] after the standard renormalisation, with the bump
    # absorbed across the other (n - 2) classes so mean alpha stays 1.0.
    'pair_caps': [
        {'numer': 'smash', 'denom': 'wrist_smash', 'ratio': 0.7},
    ],
}

use_val_improvability_gate: bool = False
val_improvability_gate: dict = {
    'val_f1_smoothing_factor': 0.9,    # EMA retention on val F1 (~6.6-epoch half-life)
    'improvement_margin': 0.015,       # smoothed val must beat its best by this to count
    'patience_epochs': 15,             # epochs with no new high before decay starts
    'min_epochs_before_gating': 10,    # no decay before this epoch (keep the early boost)
    'revert_step_per_epoch': 0.2,      # fraction of the gap to the mean reverted per epoch
    'stop_gating_after_fraction': 0.75,  # freeze past 0.75*n_epochs (protect anneal blooms)
}
```

Three knobs that resolve outside the dicts above:

- **`n_epochs`** lives on `Hyp` directly, one level above the gate dict. It feeds into `_init_val_improvability_gate` so that `stop_gating_after_fraction` resolves to an absolute freeze epoch.
- **`use_val_improvability_gate`** is the toggle that decides whether `val_improvability_gate` actually gets forwarded to the loss constructor or `None` does (`bst_x_train.py:432-434`). The dict itself stays defined either way, so the knobs remain easy to find when the flag flips on.
- **`pair_caps`** accepts `None` or `[]`. Both values bypass the enforcement loop and produce the same alpha as the no-cap path. The equivalence is pinned by `tests/test_adaptive_focal.py::test_pair_cap_none_matches_no_cap_path`.

## Runtime contract

Per-epoch sequence inside `train_network`'s loop (`bst_x_train.py:615-710`):

1. `train_one_epoch(...)` runs the forward / backward / step / scheduler over the train loader, calling `accumulate_class_counts(preds, labels, n_classes)` on each batch's `argmax` predictions to grow `(train_tp, train_fp, train_fn)`.
2. `train_per_class_f1 = per_class_f1_from_counts(train_tp, train_fp, train_fn)` reduces the counters to a length-`n_classes` F1 vector.
3. `loss_fn.update_alpha(train_per_class_f1)` (only when `loss_fn isinstance AdaptiveFocalLoss`) does, in this order:
   a. EMA-smooths `f1_running` in-place with `self.momentum` (default `0.9`); buffer identity preserved across calls so `.to(device)` and `state_dict()` round-trip correctly,
   b. recomputes `alpha = ((1 - f1_running).clamp(min=1e-8))^tau` (the clamp guards the base before exponentiation so `tau` stays defined; no class can saturate alpha to literal zero),
   c. renormalises so `alpha.mean() == 1.0` (i.e. `alpha.sum() == n_classes`),
   d. applies each `(numer_idx, denom_idx, ratio)` rule sequentially: if `alpha[numer] < ratio * alpha[denom]`, set `alpha[numer]` to the target and subtract `bump / (n_classes - 2)` from every other class. Order matters only when caps share a class.
   e. defensive `clamp_(min=1e-8)` whenever `pair_caps` is configured (the clamp fires regardless of whether any rule's bump was positive; it's a no-op when nothing moved),
   f. `self.epoch += 1`.
4. `validate(...)` returns `f1_per_class` and the `present` mask (true where the class had at least one val sample).
5. `loss_fn.apply_val_gate(f1_per_class, present)` (only when `loss_fn.val_gate_enabled`) does:
   a. for each class, seed-or-EMA the smoothed val F1 and update the running best (every epoch, regardless of window),
   b. inside the window `min_epochs_before_gating < epoch < freeze_epoch`, ramp `gate_revert_fraction` up by `revert_step` while a class is plateaued AND over-allocated, and down otherwise,
   c. compute `pulled = gate_revert_fraction * max(alpha - 1, 0)` for every class, subtract it,
   d. hand the freed budget to classes that were not pulled this epoch, by scaling them so the total stays at `n_classes`. The pulled classes stay where the pull left them; a fully reverted class lands at exactly 1.0.
6. The post-epoch diagnostic prints and TB writes use the gated `alpha` value. Next epoch's `forward` reads `self.alpha[labels]` (no further gating between epochs).

**End-to-end invariant.** After step 5, `alpha.mean() == 1.0` (give or take float rounding). Both `pair_caps` and the gate maintain this by construction, which is what keeps the loss scale comparable to uniform CE. Any future extension to the alpha pipeline should preserve the same invariant.

## Math contracts

**pair_caps.** For each rule `(numer, denom, ratio)`:

```
target = ratio * alpha[denom]
bump   = max(target - alpha[numer], 0)
alpha[numer]  += bump
alpha[others] -= bump / (n_classes - 2)    # `others` = all classes minus numer and denom
```

The redistribution preserves `alpha.sum()` exactly. Rules apply in list order, so a later rule sees an earlier rule's effect when they share a class.

**Val-improvability gate.** Per class `c`:

```
# Always (smoother + best, every epoch from the first reading;
# present[c]=False skips the class for this epoch):
smoothed[c] = smoothing * smoothed[c] + (1 - smoothing) * val_f1[c]    # seeded on first reading
improved    = smoothed[c] > best[c] + improvement_margin
if improved:
    best[c] = smoothed[c]

# Inside the window (min_epochs_before_gating < epoch < freeze_epoch):
# counter tick AND plateau evaluation AND revert-fraction ramp all live here.
# Outside the window: epochs_since_improvement does NOT tick.
if improved:
    epochs_since_improvement[c] = 0
else:
    epochs_since_improvement[c] += 1
plateaued     = epochs_since_improvement[c] >= patience_epochs
over_allocated = alpha[c] > 1.0
if plateaued AND over_allocated:
    revert_fraction[c] = min(revert_fraction[c] + revert_step, 1.0)
else:
    revert_fraction[c] = max(revert_fraction[c] - revert_step, 0.0)

# Pull + hand-off (every epoch, no-op when freed == 0; happens when all
# revert_fraction = 0, or when all classes sit at/below the mean, or both):
over_allocation = max(alpha - 1.0, 0.0)            # 0 at/below mean
pulled          = revert_fraction * over_allocation
alpha          -= pulled
freed           = pulled.sum()
recipients      = (pulled == 0)                    # the climbers and the below-mean classes
recipient_sum   = alpha[recipients].sum()
if freed > 0 and recipient_sum > 0:                # defensive; recipient_sum > 0 in practice
    alpha[recipients] *= (recipient_sum + freed) / recipient_sum
```

Two design points sit underneath the maths.

**The hand-off is one-sided.** Budget freed from a pulled class goes only to the non-pulled classes, so a fully reverted class lands at exactly 1.0 and stays there.

**Below-mean classes are inert.** A class with `alpha[c] < 1.0` has `over_allocated=False`, so the else branch fires and `revert_fraction` ramps DOWN, bottoming at 0. The story for any given class:

- A class that has always been below-mean keeps `revert_fraction = 0` from the start.
- A class that flips from over- to below-mean mid-run ramps its `revert_fraction` down to 0 over `1 / revert_step` epochs (5 epochs at the default `revert_step = 0.2`).
- Either way, the pull step (`pulled = revert_fraction * over_allocation`) lands at 0 for any below-mean class, because `over_allocation = max(alpha - 1, 0) = 0` there.

The TB `Revert/{c}` scalar reads honestly: it only moves for classes the gate actually pulled.

## Manifest emission

Two example manifests pin the shape:

- **`pair_caps` smoke test.** `experiments/bst_x/shuttleset/run_20260501_230252/manifest.yaml` lines 25-34.
- **Gate-on Series J cell.** `experiments/bst_x/shuttleset/run_20260602_092522_446222/manifest.yaml` (look for `use_val_improvability_gate: true` and the six-key `val_improvability_gate:` block under `config:`).

Both blocks emit verbatim from `Hyp`. The path runs through `track_run` and `_config_to_dict` (`src/bst_x/run_tracker.py:157-164`), which carries no mechanism-specific code; see [`run_tracker.py` under Code touch points](#code-touch-points) for the branch detail. Once the fields are back on `Hyp`, the manifest emits them automatically.

Per-serial blocks (`serials:`) carry the headline metrics only. The alpha trajectory and revert fractions live alongside them in the TB event files under `tb/serial_N/`, as `Alpha/{c}` and `Revert/{c}` scalars per epoch.

## Tests

`tests/test_adaptive_focal.py` is one file, 898 lines. Coverage relevant to the extensions:

**pair_caps (9 cases, L602-786)**

| Test | What it pins |
|---|---|
| `test_pair_cap_below_threshold_bumps_to_ratio` | rule fires, `alpha[numer]` lands at `ratio * alpha[denom]` |
| `test_pair_cap_above_threshold_no_op` | rule no-ops when the ratio already holds |
| `test_pair_cap_preserves_mean_one` | `alpha.mean() == 1` after the bump |
| `test_pair_cap_redistribution_uniform_on_others` | bump comes uniformly from the other `n - 2` classes |
| `test_pair_cap_multi_pair_both_engage` | two rules apply sequentially, both effects visible |
| `test_pair_cap_unknown_class_name_raises` | name validation (numer + denom both checked) |
| `test_pair_cap_invalid_ratio_raises` | ratio bound check (`0 < ratio <= 1`) |
| `test_pair_cap_numer_equals_denom_raises` | `numer != denom` check |
| `test_pair_cap_none_matches_no_cap_path` | `None` and `[]` both produce alpha identical to the no-cap path (so cap-off cells stay bit-exact) |

**val-improvability gate (5 cases, L790-end)**

| Test | What it pins |
|---|---|
| `test_gate_disabled_by_default_is_inert` | `val_improvability_gate=None` leaves `alpha` untouched |
| `test_gate_config_validation` | each of the six knob ranges, plus the `freeze > min_epochs` and `min_epochs >= warm_up` guards |
| `test_gate_seeds_then_emas_and_skips_absent` | smoother seeds on first reading, EMAs after, `present=False` skips |
| `test_gate_reverts_plateaued_to_exactly_one_and_keeps_climber` | plateaued over-allocated class lands at 1.0; a still-improving class is left alone |
| `test_gate_window_and_one_sided` | counter + revert ramp only inside the window; below-mean classes are no-ops |

Tests for the base CDB-F1 update (L75-600) and helper functions belong to the base loss; they survive a rip of either extension and stay in the file regardless.

## Run history and retirement notes

**pair_caps.** Ran once as a single-cell smoke test: row `#20` in `bst_x_training_runs.md`, `run_20260501_230252`, smash / wrist_smash with `ratio=0.7`.

The rule did shift the alpha balance across the smash pair as designed. The headline numbers landed within seed noise of the gate-off baseline, so the run never got a follow-up sweep.

The validation tests in `test_adaptive_focal.py` still cover the mechanism, so the wiring is known to be sound. The open question is experimental: whether the rule earns its keep on a real taxonomy across multiple seeds.

**val-improvability gate (focal_alpha_revert).** Built end of May 2026, motivated by the per-class arcs in `docs/architecture_notes/alpha_arc_analysis/` (the `arc_*.png` figures show several classes' alpha staying high after their val F1 had plateaued, because the train-F1 EMA kept the class flagged as struggling even while it overfit). Ran as two batches:

- **Series H** (`Hyp.ablation_id = 'focal_alpha_revert_overallocated'`, six cells across five taxon / split combos, 31 May - 1 June 2026, bourbaki). Rows `38, 39, 41, 43, 46, 48` in the ledger. Headline narrative copied from the run manifests: *"hardly shifts anything across most taxonomies. The only noticeable difference is on une_v1_14 / v2, where it lifts mean min-F1 +0.016. Otherwise pulls down single-best serial macro and min across all taxa except shuttleset_18 (best within noise at +0.002). Reduced mean-vs-best diff suggests it slightly regularises seed variation."*
- **Series J** (gate crossed with the two wd endpoints `1e-2` and `4e-1` on bst_25 / baseline, une_v1_14 / v2, shuttleset_18 / v2; 2 June 2026, carmack). Rows `59-64`. Result: *"focal_alpha_revert on did not give the best setting on any of the three. The only place it raised min-F1 was bst_25 at wd 1e-2 (0.548 off to 0.595 on), and that still sits below the old standard's 0.620. Une lost min-F1 at both endpoints; shuttleset_18's min-F1 is driven_flight and not meaningful. Macro stayed flat."*

**Verdict.** Gate retired; default `use_val_improvability_gate = False`.

The mechanism works as specified. The une_v1_14 / v2 lift in Series H was real, and the alpha arcs show it pulled the right classes. The reason for retirement is comparative: the simpler `wd 4e-1 + decay-exclusion` knob landed in the same window and matched or beat the gate's lift wherever they overlapped, so the gate's added complexity stopped paying for itself.

Full retirement writeup: `bst_x_sweep_summary_wd_x_focal_alpha_revert.md`.

**Pointer index**

- Sweep summary (single source of truth for the retirement results): `docs/architecture_notes/bst_x_sweep_summary_wd_x_focal_alpha_revert.md`.
- Run ledger: `experiments/bst_x/bst_x_training_runs.md` (rows `20` for pair_caps, `38, 39, 41, 43, 46, 48` for Series H, `59-64` for Series J). Generated from `experiments/bst_x/build_training_runs_table.py`.
- Run manifests (one per cell, each carries `best_serials` plus a free-text notes block): under `experiments/bst_x/shuttleset/run_2026053{1}_*/` (Series H) and `run_20260602_*/` (Series J).
- Per-class arc analysis that motivated the gate: `docs/architecture_notes/alpha_arc_analysis/` (`findings.md`, `tables.md`, six `arc_*.png` plots per cell, `macro_arcs_all_cells.png`, `alpha_vs_valf1_grid.png`).
- Base CDB-F1 design (unchanged by either extension): `docs/architecture_notes/class_f1_focal_design.md`.
- Pair-aware companion design that was held as a second arm but never built: `docs/architecture_notes/seesaw_f1_focal_design.md`.

**Other docs and scripts that name the gate or pair_caps wiring**

These places reference the live mechanisms by symbol or section. A rip will leave them out of sync with the code; a re-wire will need a reverse pass to bring them back in line. Listed roughly in order of update urgency.

- **`docs/architecture_notes/class_f1_focal_design.md` L819-841.** Live val-improvability-gate section describing `use_val_improvability_gate`, `--val-improvability-gate`, `Revert/{c}`, and `Alpha/{c}` as current wiring. This is the primary design doc for the loss family, so it needs to track the code.

- **`docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md` L216.** Paragraph naming `apply_val_gate`, `use_val_improvability_gate`, the `val_improvability_gate` dict, `--val-improvability-gate`, and the `collation_runner` forwarding.

- **`docs/architecture_notes/function_invariants/train_network.md`.** Pins the per-epoch ordering invariant (`update_alpha` -> `validate` -> `apply_val_gate`) at line numbers that already drift on this commit. Update or remove the invariant block.

- **`scripts/plots/f1_runs_bar_charts.py`.** Hardcodes the Series H run IDs to A/B the gate-on vs gate-off cells. Run IDs are historical, so a rip leaves the script harmless. Worth keeping as the canonical comparison-plot recipe if the gate comes back for another sweep.

- **`docs/architecture_notes/alpha_arc_analysis/parse_arcs.py` L99.** Reads `Alpha/{c}` TB scalars (`'alpha': {c: sc.get(f'Alpha/{c}', {}) ...}`) into `arcs.pkl`, which `plot_arcs.py` and `analyse_arcs.py` then consume. These scripts produced the figures that motivated the gate's design. They only read `Alpha/{c}` today; a re-wire pass that wants revert arcs alongside the alpha arcs needs to extend them to read `Revert/{c}` too.

- **`src/bst_x/aim_backfill.py:10` and `src/bst_x/run_tracker.md:138-139`.** Prose lists of the per-epoch TB scalar prefixes (`Loss/*`, `F1/*`, `F1_train/*`, `F1_val/*`, `Alpha/*`, `Aug/*`, `Schedule/*`, `best/*`). The aim-backfill code itself iterates every scalar tag, so `Revert/*` rides along automatically when the gate is on; the prose lists are what go stale until updated.

- **`src/bst_x/hparam_sweep.py` (`invoke_bst_train`, L454-477).** Does NOT forward `--val-improvability-gate`; only `collation_runner.py` does. Listed here so a re-wirer doesn't waste time trying to route the flag through `hparam_sweep`.
