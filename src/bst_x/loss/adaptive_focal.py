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

Train-loop responsibilities (see ``bst_x_train.train_one_epoch`` /
``train_network``):
    1. accumulate per-class TP / FP / FN during each epoch's forward pass,
    2. compute per-class F1 with ``per_class_f1_from_counts`` at end-of-epoch,
    3. call ``loss_fn.update_alpha(per_class_f1)`` once per epoch,
    4. (optional) read ``loss_fn.alpha`` for diagnostic logging.

Full motivation + paper-verified equations live in
``docs/architecture_notes/class_f1_focal_design.md``.
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
    :param device: device for the running buffers; defaults to CPU and gets
        moved by ``.to()`` like any other module.
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
        device: torch.device | str | None = None,
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

        # Init f1_running to 1.0 (model-is-perfect prior); update_alpha mixes
        # in real readings via EMA each epoch. While epoch < warm_up_epochs
        # forward() ignores alpha and uses uniform weights, so the EMA can
        # absorb a few epochs of real signal before its shape applies.
        self.register_buffer('f1_running', torch.ones(n_classes))
        self.register_buffer('alpha', torch.ones(n_classes))
        # Plain int because state_dict persistence isn't needed (each serial
        # is a fresh model + fresh loss instance; no cross-serial resume).
        self.epoch = 0

        if device is not None:
            self.to(device)

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
        self.epoch += 1

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
