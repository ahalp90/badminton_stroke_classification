"""Class-F1-driven adaptive focal loss for the imbalanced-difficulty regime.

Implements CDB-loss (Sinha et al. ACCV 2020 / IJCV 2022) with the per-class
difficulty signal swapped from held-out val accuracy to running train F1,
optionally composed with focal's ``(1 - p_t) ** gamma`` per-sample focusing
(Lin et al. ICCV 2017).

Loss shape, in plain English:
    per-class weight w_c = (1 - F1_running_c) ** tau
    per-sample loss      = - w_{c=label} * (1 - p_t) ** gamma * log(p_t)
    weights renormalised to mean 1.0 each epoch so the average loss scale
    stays comparable to uniform CE.

Optional pair-cap extension targets known confusion pairs the scalar-per-class
CDB signal can't see: cap ``alpha[numer] / alpha[denom]`` from below at a
configured ratio so a high-F1 partner doesn't get downweighted past the point
where its training signal collapses. The bump is absorbed across the other
``n_classes - 2`` classes so mean alpha stays 1.0.

Train-loop responsibilities (see ``bst_x_train.train_one_epoch`` /
``train_network``):
    1. accumulate per-class TP / FP / FN during each epoch's forward pass,
    2. compute per-class F1 with ``per_class_f1_from_counts`` at end-of-epoch,
    3. call ``loss_fn.update_alpha(per_class_f1)`` once per epoch,
    4. (optional) read ``loss_fn.alpha`` for diagnostic logging.

Full motivation + paper-verified equations live in
``scratch/architecture_notes/class_f1_focal_design.md``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss with per-class alpha driven by running train F1.

    Forward signature mirrors ``nn.CrossEntropyLoss(reduction='mean')``: takes
    pre-softmax logits ``[B, n_classes]`` and integer labels ``[B]``, returns
    a scalar loss. The per-class weight vector ``alpha`` is held as a buffer
    so it persists across forward passes, moves with ``.to(device)``, and is
    saved in ``state_dict()``.

    During the first ``warm_up_epochs`` epochs the EMA still updates in the
    background but the forward pass uses uniform alpha, so the running F1
    estimate has time to absorb a few real readings before its shape starts
    driving the gradient.

    :param n_classes: size of the active class list (post-derive_active).
    :param class_names: parallel name list for diagnostic printouts; length
        must match ``n_classes``.
    :param tau: per-class aggressiveness exponent. ``tau=1.0`` uses ``1 - F1``
        directly; ``tau=2.0`` squares the gap.
    :param gamma: per-sample focal exponent on ``(1 - p_t)``. ``gamma=0`` is
        pure CDB (no focal modulation); ``gamma=1`` is the gentle default
        chosen for ShuttleSet's known label noise.
    :param momentum: EMA momentum on the running F1 estimate; ``momentum=0.9``
        gives a half-life of ~6.6 epochs (matches PyTorch BatchNorm and Adam
        first-moment convention).
    :param warm_up_epochs: epochs of uniform alpha at the start of training,
        before adaptive shape kicks in.
    :param f1_floor: lower clip on F1 readings before mapping to alpha. F1 is
        naturally bounded so the default 0.0 is fine; raise to ~0.05 only if
        a class flatlines and saturates alpha.
    :param pair_caps: optional list of pair-cap rules, each a dict with keys
        ``'numer'``, ``'denom'`` (class names) and ``'ratio'`` (float in
        ``(0, 1]``). After the standard renormalisation, each rule enforces
        ``alpha[numer] >= ratio * alpha[denom]`` by bumping
        ``alpha[numer]`` up if needed and subtracting the bump uniformly across
        the other ``n_classes - 2`` classes. Default ``None`` disables the
        feature. Targets known confusion pairs (e.g. smash <-> wrist_smash)
        the scalar-per-class CDB signal can't model.
    :param device: device for the running buffers; defaults to CPU and gets
        moved by ``.to()`` like any other module.
    :param val_improvability_gate: optional config dict for the val-improvability
        gate (``None`` disables it). When engaged, the gate decays a class's
        alpha back toward the renorm mean of 1.0 once that class has stopped
        improving on val, freeing the over-allocated budget for classes still
        climbing. See ``_init_val_improvability_gate`` for the dict keys and
        ``apply_val_gate`` for the per-epoch mechanism.
    :param n_epochs: total training epochs; needed only when the gate is on, to
        turn its ``stop_gating_after_fraction`` into an absolute freeze epoch.
    """

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

    @torch.no_grad()
    def update_alpha(self, per_class_f1: torch.Tensor) -> None:
        """EMA-smooth ``per_class_f1`` into ``f1_running``, refresh ``alpha``.

        Called once per epoch from the train loop after ``train_one_epoch``
        returns the per-class TP/FP/FN counters. Bumps the internal epoch
        counter so the warm-up gate in ``forward`` advances.

        :param per_class_f1: shape ``[n_classes]`` train F1 vector for the
            epoch just finished.
        """
        if per_class_f1.shape != (self.n_classes,):
            raise ValueError(
                f'per_class_f1 shape {tuple(per_class_f1.shape)} != ({self.n_classes},)'
            )

        f1 = per_class_f1.to(self.f1_running).clamp(min=self.f1_floor, max=1.0)
        # In-place buffer updates: keeps the registered buffer identity stable
        # across calls, so state_dict round-trips and .to(device) propagation
        # don't depend on PyTorch's __setattr__ buffer-rebind path.
        self.f1_running.mul_(self.momentum).add_(f1, alpha=1.0 - self.momentum)
        # clamp(min=eps) keeps the base strictly positive so tau ** anything
        # stays defined; no class can saturate alpha to literal zero.
        raw_alpha = (1.0 - self.f1_running).clamp(min=1e-8) ** self.tau
        # Renormalise to mean 1.0; preserves overall CE loss scale and keeps
        # AdamW's effective per-parameter LR comparable to uniform-CE runs.
        self.alpha.copy_(raw_alpha * (self.n_classes / raw_alpha.sum()))

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
        ``scratch/architecture_notes/alpha_arc_analysis/``.

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

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Adaptive-focal CE on a batch.

        :param logits: pre-softmax model output, shape ``[B, n_classes]``.
        :param labels: int64 class indices, shape ``[B]``.
        :return: scalar mean loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)                    # [B, C]
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        # Clamp p_t below 1 by an epsilon so (1 - p_t) ** gamma stays
        # differentiable when the model is highly confident on a sample.
        p_t = log_p_t.exp().clamp(max=1.0 - 1e-7)

        if self.epoch < self.warm_up_epochs:
            alpha_t = torch.ones_like(p_t)
        else:
            alpha_t = self.alpha[labels]                              # fancy-index lookup, [B]

        # gamma=0 reduces (1 - p_t)^0 to a constant 1.0 across the batch, so
        # we always compute the same expression; no special-case branch.
        focal_mod = (1.0 - p_t) ** self.gamma

        loss = -alpha_t * focal_mod * log_p_t
        return loss.mean()


def per_class_f1_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-class F1 from running TP / FP / FN tensors.

    Used by the train loop end-of-epoch to feed
    ``AdaptiveFocalLoss.update_alpha``. ``eps`` guards against the
    no-prediction-no-ground-truth case (returns 0 rather than NaN).

    :param tp: shape ``[n_classes]`` true-positive counts.
    :param fp: shape ``[n_classes]`` false-positive counts.
    :param fn: shape ``[n_classes]`` false-negative counts.
    :param eps: small constant added to denominators.
    :return: shape ``[n_classes]`` per-class F1 in ``[0, 1]``.
    """
    # ``bincount`` outputs are int64; cast so the eps-padded division stays in
    # float and downstream EMA math doesn't get caught on dtype mismatches.
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return f1


def accumulate_class_counts(
    preds: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised per-class TP / FP / FN counters for one batch.

    Three ``bincount`` calls instead of an n_classes-iter Python loop. The
    decomposition uses the ``correct = preds == labels`` mask: TPs come from
    the predicted-class index of correct rows; FPs from the predicted-class
    index of wrong rows; FNs from the ground-truth-class index of wrong rows.

    :param preds: int64 predicted class indices, shape ``[B]``.
    :param labels: int64 ground-truth class indices, shape ``[B]``.
    :param n_classes: total number of classes (sets ``minlength`` so empty
        bins still produce a length-``n_classes`` count vector).
    :return: ``(tp, fp, fn)``, each shape ``[n_classes]`` int64.
    """
    correct = preds == labels  # bool mask over batch
    tp = torch.bincount(preds[correct],  minlength=n_classes)
    fp = torch.bincount(preds[~correct], minlength=n_classes)
    fn = torch.bincount(labels[~correct], minlength=n_classes)
    return tp, fp, fn
