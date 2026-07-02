"""Class-F1-driven adaptive focal loss; rebalances inter-epoch; handles imbalance by difficulty rather than sample n.

Implements CDB-loss (Sinha et al. ACCV 2020 / IJCV 2022) with the per-class
difficulty signal swapped from held-out val accuracy to running train F1. F1 at train considers recall and keeps val more isolated.
Combined with standard focal's ``(1 - p_t) ** gamma`` per-sample weighting (Lin et al. ICCV 2017).

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

Full explanation + equations at: ``docs/architecture_notes/class_f1_focal_design.md``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float32, Int64, jaxtyped


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss with per-class alpha driven by running train F1.

    Forward signature mirrors ``nn.CrossEntropyLoss(reduction='mean')``: takes
    pre-softmax logits ``[B, n_classes]`` and integer labels ``[B]``, returns
    a scalar loss. The per-class weight vector ``alpha`` is held as a buffer
    so it persists across forward passes, moves with ``.to(device)``, and is
    saved in ``state_dict()``.

    During the first ``warm_up_epochs`` epochs the EMA still updates in the
    background but the forward pass uses static uniform alpha, so the running F1
    estimate has time to absorb a few real readings before its shape starts
    driving the gradient.

    :param class_names: parallel name list for diagnostic printouts; its
        length sets ``n_classes``.
    :param tau: per-class aggressiveness exponent. ``tau=1.0`` uses ``1 - F1``
        directly; ``tau=2.0`` squares the gap.
    :param gamma: per-sample focal exponent on ``(1 - p_t)``. ``gamma=0`` is
        pure CDB (no focal modulation); ``gamma=1`` is the gentle default
        chosen for ShuttleSet's known label noise.
    :param momentum: EMA momentum on the running F1 estimate; ``momentum=0.9``
        gives a half-life of ~6.6 epochs (matches PyTorch BatchNorm and Adam
        first-moment convention).
    :param warm_up_epochs: n epochs of static uniform alpha before adaptive kicks-in.
    :param f1_floor: lower clip on F1 readings before mapping to alpha. F1 is
        naturally bounded so the default 0.0 is fine; raise to ~0.05 only if
        a class flatlines and saturates alpha.
    """

    def __init__(
        self,
        class_names: list[str],
        tau: float = 1.0,
        gamma: float = 1.0,
        momentum: float = 0.9,
        warm_up_epochs: int = 5,
        f1_floor: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'momentum must be in [0, 1); got {momentum}')

        self.class_names = list(class_names)
        self.n_classes = len(self.class_names)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.momentum = float(momentum)
        self.warm_up_epochs = int(warm_up_epochs)
        self.f1_floor = float(f1_floor)

        # Init f1_running to 1.0 (model-is-perfect prior); update_alpha mixes
        # in real readings via EMA each epoch. While epoch < warm_up_epochs
        # forward() ignores alpha and uses uniform weights, so the EMA can
        # absorb a few epochs of real signal before its shape applies.
        self.register_buffer('f1_running', torch.ones(self.n_classes))
        self.register_buffer('alpha', torch.ones(self.n_classes))
        # Plain int because state_dict persistence isn't needed (each serial
        # is a fresh model + fresh loss instance; no cross-serial resume).
        self.epoch = 0

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

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        logits: Float32[torch.Tensor, 'batch n_classes'],
        labels: Int64[torch.Tensor, 'batch'],
    ) -> Float32[torch.Tensor, '']:
        """Adaptive-focal CE on a batch.

        :param logits: pre-softmax model output.
        :param labels: class indices per sample.
        :return: scalar mean loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)                    # [B, C]
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        p_t = log_p_t.exp()

        if self.epoch < self.warm_up_epochs:
            alpha_t = torch.ones_like(p_t)
        else:
            alpha_t = self.alpha[labels]                              # [B]

        # gamma=0 reduces (1 - p_t)^0 to a constant 1.0 across the batch, so
        # we always compute the same expression; no special-case branch.
        # Clamp base > 0: (1 - p_t)^gamma has infinite slope at 0 for gamma < 1.
        focal_mod = (1.0 - p_t).clamp(min=1e-7) ** self.gamma

        loss = -alpha_t * focal_mod * log_p_t
        return loss.mean()


@jaxtyped(typechecker=beartype)
def per_class_f1_from_counts(
    tp: Int64[torch.Tensor, 'n_classes'], fp: Int64[torch.Tensor, 'n_classes'], fn: Int64[torch.Tensor, 'n_classes'],
    eps: float = 1e-8,
) -> Float32[torch.Tensor, 'n_classes']:
    """Per-class F1 from running TP / FP / FN counts.

    Called by the train loop at end-of-epoch to feed ``AdaptiveFocalLoss.update_alpha``.
    ``eps`` does two jobs: guards the no-prediction-no-ground-truth case (returns 0
    rather than NaN), and promotes the int64 counts through the arithmetic to float32.

    :param tp: true-positive counts per class.
    :param fp: false-positive counts per class.
    :param fn: false-negative counts per class.
    :param eps: small float added to denominators.
    :return: per-class F1 in ``[0, 1]``.
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return f1


@jaxtyped(typechecker=beartype)
def accumulate_class_counts(
    preds: Int64[torch.Tensor, 'batch'],
    labels: Int64[torch.Tensor, 'batch'],
    n_classes: int,
) -> tuple[
    Int64[torch.Tensor, 'n_classes'], Int64[torch.Tensor, 'n_classes'], Int64[torch.Tensor, 'n_classes'],
]:
    """Vectorised per-class TP / FP / FN counters for one batch.

    :param preds: predicted class indices per sample.
    :param labels: ground-truth class indices per sample.
    :param n_classes: used by ``bincount`` ``minlength`` so empty class bins contribute to a length-``n_classes`` vector.
    :return: ``(tp, fp, fn)``.
    """
    correct = preds == labels  # bool mask over batch
    tp = torch.bincount(preds[correct], minlength=n_classes)
    fp = torch.bincount(preds[~correct], minlength=n_classes)
    fn = torch.bincount(labels[~correct], minlength=n_classes)
    return tp, fp, fn
