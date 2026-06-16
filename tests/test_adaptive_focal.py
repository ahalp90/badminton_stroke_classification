"""Tests for the class-F1-driven adaptive focal loss.

Covers eight sections matching the implementation surface in
``scratch/architecture_notes/class_f1_focal_design.md``:

1. ``per_class_f1_from_counts`` numerical correctness.
2. ``accumulate_class_counts`` vectorised TP/FP/FN against a hand-rolled
   reference loop.
3. ``AdaptiveFocalLoss.forward`` shape, reduction, and CE equivalence at
   warm-up + alpha=uniform + gamma=0.
4. ``AdaptiveFocalLoss.update_alpha`` math: EMA correctness, mean=1.0
   renormalisation, tau exponent shape, warm-up gating, f1_floor.
5. Forward/backward gradient flow + per-class gradient scaling.
6. State persistence via ``state_dict`` / ``.to(device)`` and constructor
   validation.
7. End-to-end mini training loop covering warm-up gate + alpha freshness.
8. Pair caps: bump math, no-op when above ratio, mean preservation,
   redistribution, multi-pair, and validation.

CPU-only. Run from repo root::

    pytest tests/test_adaptive_focal.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from loss.adaptive_focal import (
    AdaptiveFocalLoss,
    accumulate_class_counts,
    per_class_f1_from_counts,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def class_names_3() -> list[str]:
    return ['a', 'b', 'c']


@pytest.fixture
def class_names_14() -> list[str]:
    # Mirrors the combo A nosides class list (post drop_unknown).
    return [
        'net_shot', 'return_net', 'smash', 'wrist_smash', 'lob', 'clear',
        'drive', 'drop', 'passive_drop', 'push', 'rush', 'cross_court_net_shot',
        'short_service', 'long_service',
    ]


def _reference_tp_fp_fn(
    preds: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-rolled per-class confusion counts; the slow ground truth."""
    tp = torch.zeros(n_classes, dtype=torch.long)
    fp = torch.zeros(n_classes, dtype=torch.long)
    fn = torch.zeros(n_classes, dtype=torch.long)
    for c in range(n_classes):
        tp[c] = ((preds == c) & (labels == c)).sum()
        fp[c] = ((preds == c) & (labels != c)).sum()
        fn[c] = ((preds != c) & (labels == c)).sum()
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Section 1: per_class_f1_from_counts
# ---------------------------------------------------------------------------

def test_f1_perfect_classifier():
    tp = torch.tensor([10.0, 5.0, 8.0])
    fp = torch.zeros(3)
    fn = torch.zeros(3)
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.allclose(f1, torch.ones(3), atol=1e-6)


def test_f1_no_predictions():
    """All FN, no TP / FP -> precision 0, recall 0, F1 0."""
    tp = torch.zeros(3)
    fp = torch.zeros(3)
    fn = torch.tensor([10.0, 5.0, 8.0])
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.allclose(f1, torch.zeros(3), atol=1e-6)


def test_f1_only_false_positives():
    """All FP, no TP / FN -> precision 0, recall 0, F1 0."""
    tp = torch.zeros(3)
    fp = torch.tensor([5.0, 3.0, 2.0])
    fn = torch.zeros(3)
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.allclose(f1, torch.zeros(3), atol=1e-6)


def test_f1_known_mix():
    """Hand-calculated F1 for one class with TP=8, FP=2, FN=2.

    precision = 8/10 = 0.8, recall = 8/10 = 0.8, F1 = 2*0.64/1.6 = 0.8.
    """
    tp = torch.tensor([8.0])
    fp = torch.tensor([2.0])
    fn = torch.tensor([2.0])
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.allclose(f1, torch.tensor([0.8]), atol=1e-4)


def test_f1_empty_counters_no_nan():
    """Class with no TP, FP, FN must yield 0, not NaN (eps guard)."""
    tp = torch.zeros(3)
    fp = torch.zeros(3)
    fn = torch.zeros(3)
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.isfinite(f1).all()
    assert torch.allclose(f1, torch.zeros(3), atol=1e-6)


def test_f1_from_int64_counts():
    """bincount returns int64; helper must cope without an external cast."""
    tp = torch.tensor([8, 0, 5], dtype=torch.int64)
    fp = torch.tensor([2, 0, 0], dtype=torch.int64)
    fn = torch.tensor([2, 0, 0], dtype=torch.int64)
    f1 = per_class_f1_from_counts(tp, fp, fn)
    assert torch.allclose(f1, torch.tensor([0.8, 0.0, 1.0]), atol=1e-4)
    assert f1.dtype == torch.float32


# ---------------------------------------------------------------------------
# Section 2: accumulate_class_counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_accumulate_matches_reference_loop(seed):
    """Vectorised bincount path matches the hand-rolled slow loop."""
    g = torch.Generator().manual_seed(seed)
    n_classes = 14
    batch = 256
    preds = torch.randint(0, n_classes, (batch,), generator=g)
    labels = torch.randint(0, n_classes, (batch,), generator=g)

    tp_v, fp_v, fn_v = accumulate_class_counts(preds, labels, n_classes)
    tp_r, fp_r, fn_r = _reference_tp_fp_fn(preds, labels, n_classes)

    assert torch.equal(tp_v, tp_r)
    assert torch.equal(fp_v, fp_r)
    assert torch.equal(fn_v, fn_r)


def test_accumulate_perfect_classifier():
    n_classes = 3
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = labels.clone()
    tp, fp, fn = accumulate_class_counts(preds, labels, n_classes)
    assert torch.equal(tp, torch.tensor([2, 2, 2]))
    assert torch.equal(fp, torch.tensor([0, 0, 0]))
    assert torch.equal(fn, torch.tensor([0, 0, 0]))


def test_accumulate_class_absent_from_batch():
    """Class with no rows in the batch still gets a 0 entry, not a missing index."""
    n_classes = 4  # class 3 absent
    preds  = torch.tensor([0, 1, 2, 0, 1])
    labels = torch.tensor([0, 1, 2, 1, 0])
    tp, fp, fn = accumulate_class_counts(preds, labels, n_classes)
    assert tp.shape == (n_classes,)
    assert tp[3].item() == 0 and fp[3].item() == 0 and fn[3].item() == 0


def test_accumulate_all_correct_single_class():
    """correct mask is all-True so preds[~correct] is empty;
    bincount(empty, minlength=N) must still return a length-N zero vector
    rather than an empty tensor or error."""
    n_classes = 4
    preds = torch.full((6,), 1, dtype=torch.long)
    labels = torch.full((6,), 1, dtype=torch.long)
    tp, fp, fn = accumulate_class_counts(preds, labels, n_classes)
    assert tp[1].item() == 6
    assert tp.sum().item() == 6
    assert fp.sum().item() == 0
    assert fn.sum().item() == 0
    assert tp.shape == (n_classes,) == fp.shape == fn.shape


def test_accumulate_stress_large_n_classes():
    """Bigger taxonomy + bigger batch: catches minlength regressions and
    overflow that wouldn't surface at n_classes=14, B=256."""
    g = torch.Generator().manual_seed(0)
    n_classes, batch = 25, 8192
    preds = torch.randint(0, n_classes, (batch,), generator=g)
    labels = torch.randint(0, n_classes, (batch,), generator=g)

    tp, fp, fn = accumulate_class_counts(preds, labels, n_classes)
    tp_r, fp_r, fn_r = _reference_tp_fp_fn(preds, labels, n_classes)

    assert torch.equal(tp, tp_r)
    assert torch.equal(fp, fp_r)
    assert torch.equal(fn, fn_r)
    # tp + fp counts every prediction once; tp + fn counts every ground truth once.
    assert (tp + fp).sum().item() == batch
    assert (tp + fn).sum().item() == batch


# ---------------------------------------------------------------------------
# Section 3: AdaptiveFocalLoss.forward
# ---------------------------------------------------------------------------

def test_forward_returns_scalar(class_names_3):
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_forward_batch_size_1(class_names_3):
    """B=1 stresses the unsqueeze(1) / squeeze(1) parity and per-sample paths."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        warm_up_epochs=0, momentum=0.0,
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.5, 0.1]))
    logits = torch.randn(1, 3)
    labels = torch.tensor([1])
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_forward_all_same_class(class_names_3):
    """Whole batch labelled the same class exercises the homogeneous-batch path."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        warm_up_epochs=0, momentum=0.0,
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.5, 0.1]))
    logits = torch.randn(8, 3)
    labels = torch.full((8,), 2, dtype=torch.long)
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)


def test_warmup_uniform_alpha_matches_plain_ce(class_names_3):
    """During warm-up with gamma=0, loss must equal nn.CrossEntropyLoss(reduction='mean')."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        gamma=0.0, warm_up_epochs=5,
    )
    logits = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    custom = loss_fn(logits, labels)
    reference = F.cross_entropy(logits, labels, reduction='mean')
    assert torch.allclose(custom, reference, atol=1e-5)


def test_warmup_uniform_alpha_with_gamma_matches_focal(class_names_3):
    """During warm-up with gamma=2, alpha is uniform so loss matches plain focal."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        gamma=2.0, warm_up_epochs=5,
    )
    logits = torch.randn(8, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    log_probs = F.log_softmax(logits, dim=-1)
    log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp().clamp(max=1.0 - 1e-7)
    expected = (-((1.0 - p_t) ** 2.0) * log_p_t).mean()

    assert torch.allclose(loss_fn(logits, labels), expected, atol=1e-5)


def test_post_warmup_uses_alpha(class_names_3):
    """After warm-up, low-F1 class gets larger weight than high-F1 class."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        tau=1.0, gamma=0.0, warm_up_epochs=0, momentum=0.0,
    )
    # momentum=0 so the EMA jumps straight to the input. F1 stack: a easy,
    # b mid, c hard.
    loss_fn.update_alpha(torch.tensor([0.95, 0.5, 0.1]))
    assert loss_fn.alpha[0] < loss_fn.alpha[1] < loss_fn.alpha[2]

    # Same logits + label sets -> bigger loss when the label is the hard
    # class (alpha[2]) than when it is the easy class (alpha[0]).
    logits = torch.tensor([[0.5, 0.3, 0.2]])
    loss_easy = loss_fn(logits, torch.tensor([0]))
    loss_hard = loss_fn(logits, torch.tensor([2]))
    assert loss_hard > loss_easy


# ---------------------------------------------------------------------------
# Section 4: update_alpha math
# ---------------------------------------------------------------------------

def test_update_alpha_shape_check_raises(class_names_3):
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    with pytest.raises(ValueError, match='shape'):
        loss_fn.update_alpha(torch.tensor([0.5, 0.5]))  # wrong length


def test_update_alpha_mean_one(class_names_14):
    """alpha mean must be exactly 1.0 (within fp tolerance) after every update."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=14, class_names=class_names_14,
        tau=1.0, momentum=0.5, warm_up_epochs=0,
    )
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        f1 = torch.rand(14, generator=g)
        loss_fn.update_alpha(f1)
        assert torch.isclose(loss_fn.alpha.mean(), torch.tensor(1.0), atol=1e-6)


def test_update_alpha_ema_step():
    """One EMA step at momentum=0.9, init=1.0, input=0.5 -> 0.95."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=2, class_names=['a', 'b'],
        momentum=0.9, warm_up_epochs=0, tau=1.0,
    )
    loss_fn.update_alpha(torch.tensor([0.5, 0.5]))
    expected = 0.9 * 1.0 + 0.1 * 0.5  # 0.95
    assert torch.allclose(loss_fn.f1_running, torch.full((2,), expected), atol=1e-6)


def test_update_alpha_tau_widens_spread():
    """Higher tau exaggerates the gap between best and worst class alphas.
    Numerical anchor (F1=[0.9, 0.5, 0.1], renormalised mean=1.0):
        tau=0.5: alpha ≈ [0.481, 1.075, 1.444], spread ≈ 0.963
        tau=1.0: alpha = [0.200, 1.000, 1.800], spread = 1.600
        tau=2.0: alpha ≈ [0.028, 0.701, 2.271], spread ≈ 2.243
    """
    f1 = torch.tensor([0.9, 0.5, 0.1])

    loss_low = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=0.5, momentum=0.0, warm_up_epochs=0,
    )
    loss_mid = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
    )
    loss_high = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=2.0, momentum=0.0, warm_up_epochs=0,
    )
    loss_low.update_alpha(f1.clone())
    loss_mid.update_alpha(f1.clone())
    loss_high.update_alpha(f1.clone())

    # Numerical alphas at tau=1 are exact (closed form: (1-F1)/mean).
    assert torch.allclose(
        loss_mid.alpha,
        torch.tensor([0.2, 1.0, 1.8]),
        atol=1e-6,
    )
    spread_low = (loss_low.alpha.max() - loss_low.alpha.min()).item()
    spread_mid = (loss_mid.alpha.max() - loss_mid.alpha.min()).item()
    spread_high = (loss_high.alpha.max() - loss_high.alpha.min()).item()
    assert spread_low < spread_mid < spread_high
    assert spread_mid == pytest.approx(1.6, abs=1e-6)
    assert spread_high == pytest.approx(2.243, abs=1e-2)
    assert spread_low == pytest.approx(0.963, abs=1e-2)


def test_ema_converges_to_constant_input():
    """100 EMA updates with a constant input should land within 1e-3 of the input.
    momentum=0.9 has half-life ~6.6, so 100 updates is far past convergence."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=2, class_names=['a', 'b'],
        momentum=0.9, warm_up_epochs=0,
    )
    target = torch.tensor([0.3, 0.7])
    for _ in range(100):
        loss_fn.update_alpha(target.clone())
    assert torch.allclose(loss_fn.f1_running, target, atol=1e-3)


def test_update_alpha_warmup_gating(class_names_3):
    """During warm-up the EMA still updates but forward returns CE-uniform loss."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        warm_up_epochs=3, momentum=0.0, gamma=0.0,
    )
    # Push very non-uniform F1 into the running estimate; alpha gets a wide spread.
    loss_fn.update_alpha(torch.tensor([0.99, 0.5, 0.01]))
    assert loss_fn.epoch == 1

    # epoch=1 still inside warm-up (warm_up_epochs=3); forward uses uniform alpha
    # and matches plain CE.
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    assert torch.allclose(
        loss_fn(logits, labels),
        F.cross_entropy(logits, labels, reduction='mean'),
        atol=1e-5,
    )

    # Two more updates -> epoch=3 == warm_up_epochs, gate opens.
    loss_fn.update_alpha(torch.tensor([0.99, 0.5, 0.01]))
    loss_fn.update_alpha(torch.tensor([0.99, 0.5, 0.01]))
    assert loss_fn.epoch == 3
    assert not torch.allclose(
        loss_fn(logits, labels),
        F.cross_entropy(logits, labels, reduction='mean'),
        atol=1e-3,
    )


def test_f1_floor_clamps_input():
    """f1_floor=0.5 clamps a 0.1 reading to 0.5 before the EMA absorbs it."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=2, class_names=['a', 'b'],
        f1_floor=0.5, momentum=0.0, warm_up_epochs=0, tau=1.0,
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.1]))
    # f1_running = 0.9 * 0 + 0.1 of clamped input.
    # With momentum=0.0: f1_running = clamped input = [0.9, 0.5].
    assert torch.allclose(loss_fn.f1_running, torch.tensor([0.9, 0.5]), atol=1e-6)


# ---------------------------------------------------------------------------
# Section 5: gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow_post_warmup(class_names_3):
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        tau=1.0, gamma=1.0, momentum=0.0, warm_up_epochs=0,
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.5, 0.1]))

    logits = torch.randn(8, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


def test_alpha_scales_per_class_gradient_magnitude():
    """A class with high alpha pulls a larger gradient on its rows."""
    loss_fn = AdaptiveFocalLoss(
        n_classes=2, class_names=['a', 'b'],
        tau=1.0, gamma=0.0, momentum=0.0, warm_up_epochs=0,
    )
    # Class 0 high alpha (low F1), class 1 low alpha (high F1).
    loss_fn.update_alpha(torch.tensor([0.1, 0.9]))
    assert loss_fn.alpha[0] > loss_fn.alpha[1]

    # Same logits, two batches: one with all-class-0 labels, one with all-class-1.
    logits = torch.randn(4, 2)
    grads_per_label = []
    for label_value in [0, 1]:
        logits_in = logits.clone().detach().requires_grad_(True)
        labels = torch.full((4,), label_value, dtype=torch.long)
        loss = loss_fn(logits_in, labels)
        loss.backward()
        grads_per_label.append(logits_in.grad.abs().sum().item())

    # Higher-alpha class produces larger gradient magnitude.
    assert grads_per_label[0] > grads_per_label[1]


# ---------------------------------------------------------------------------
# Section 6: state, device, and constructor validation
# ---------------------------------------------------------------------------

def test_alpha_is_buffer_not_parameter(class_names_3):
    """alpha and f1_running must be buffers, never parameters; gradient must
    not route through them. A future refactor that flips them to nn.Parameter
    would silently train the loss state and distort the LR schedule."""
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    buffer_names = dict(loss_fn.named_buffers())
    param_names = dict(loss_fn.named_parameters())
    assert 'alpha' in buffer_names
    assert 'f1_running' in buffer_names
    assert 'alpha' not in param_names
    assert 'f1_running' not in param_names
    assert loss_fn.alpha.requires_grad is False
    assert loss_fn.f1_running.requires_grad is False


def test_buffers_in_state_dict(class_names_3):
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    sd = loss_fn.state_dict()
    assert 'f1_running' in sd
    assert 'alpha' in sd


def test_state_dict_round_trip(class_names_3):
    src = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3,
        momentum=0.0, warm_up_epochs=0, tau=1.0,
    )
    src.update_alpha(torch.tensor([0.9, 0.5, 0.1]))

    dst = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    dst.load_state_dict(src.state_dict())
    assert torch.allclose(dst.f1_running, src.f1_running)
    assert torch.allclose(dst.alpha, src.alpha)


def test_to_cpu_moves_buffers(class_names_3):
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    loss_fn.to('cpu')
    assert loss_fn.f1_running.device.type == 'cpu'
    assert loss_fn.alpha.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_to_cuda_moves_buffers(class_names_3):
    loss_fn = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3)
    loss_fn.to('cuda')
    assert loss_fn.f1_running.device.type == 'cuda'
    assert loss_fn.alpha.device.type == 'cuda'


def test_class_names_length_check():
    with pytest.raises(ValueError, match='class_names'):
        AdaptiveFocalLoss(n_classes=3, class_names=['a', 'b'])  # too short


def test_invalid_momentum_raises():
    with pytest.raises(ValueError, match='momentum'):
        AdaptiveFocalLoss(n_classes=3, class_names=['a', 'b', 'c'], momentum=1.5)
    with pytest.raises(ValueError, match='momentum'):
        AdaptiveFocalLoss(n_classes=3, class_names=['a', 'b', 'c'], momentum=-0.1)


# ---------------------------------------------------------------------------
# Section 7: end-to-end mini training loop
# ---------------------------------------------------------------------------

def test_end_to_end_mini_loop():
    """Tiny linear classifier wired through forward + backward + update_alpha
    over 4 epochs. Catches state-machine bugs (epoch counter placement,
    warm-up gate boundary, alpha freshness in forward) that the unit tests
    can miss when warm-up and EMA are exercised in isolation.
    """
    torch.manual_seed(42)
    n_classes = 4
    model = torch.nn.Linear(8, n_classes)
    loss_fn = AdaptiveFocalLoss(
        n_classes=n_classes, class_names=['a', 'b', 'c', 'd'],
        warm_up_epochs=2, momentum=0.5, tau=1.0, gamma=0.0,
    )
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    # Synthetic per-epoch F1 with class 3 stuck at the bottom of the range.
    f1_per_epoch = torch.tensor([
        [0.90, 0.85, 0.80, 0.20],
        [0.92, 0.87, 0.82, 0.25],
        [0.93, 0.89, 0.84, 0.30],
        [0.94, 0.90, 0.86, 0.32],
    ])

    # Capture warm-up forward parity: with alpha uniform and gamma=0, the
    # adaptive loss must match plain CE on the same logits/labels.
    x_probe = torch.randn(8, 8)
    labels_probe = torch.randint(0, n_classes, (8,))
    logits_probe = model(x_probe)
    pre_warm_loss = loss_fn(logits_probe, labels_probe).detach()
    pre_warm_ce = F.cross_entropy(logits_probe, labels_probe).detach()
    assert torch.allclose(pre_warm_loss, pre_warm_ce, atol=1e-5)

    for epoch in range(4):
        for _ in range(3):
            x = torch.randn(8, 8)
            labels = torch.randint(0, n_classes, (8,))
            logits = model(x)
            loss = loss_fn(logits, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            assert torch.isfinite(loss)
        loss_fn.update_alpha(f1_per_epoch[epoch])

    # After 4 update_alpha calls (>= warm_up_epochs=2), alpha should be
    # off uniform and the struggling class 3 should carry the largest weight.
    assert torch.isclose(loss_fn.alpha.mean(), torch.tensor(1.0), atol=1e-6)
    assert loss_fn.alpha[3] > loss_fn.alpha[0]
    assert loss_fn.alpha[3] == loss_fn.alpha.max()
    # f1_running must have shifted off init=1.0 toward the supplied F1.
    assert (loss_fn.f1_running < 1.0).all()
    # Epoch counter advanced exactly four times.
    assert loss_fn.epoch == 4


# ---------------------------------------------------------------------------
# Section 8: Pair caps
# ---------------------------------------------------------------------------

def test_pair_cap_below_threshold_bumps_to_ratio():
    """When alpha[numer]/alpha[denom] < ratio, alpha[numer] is lifted to
    ratio * alpha[denom] and the bump is subtracted from the other classes."""
    # tau=1, momentum=0 makes alpha closed-form: (1-F1) renormalised to mean 1.
    # F1=[0.9, 0.5, 0.1] -> (1-F1)=[0.1, 0.5, 0.9] -> alpha=[0.2, 1.0, 1.8].
    # Cap (numer='a', denom='c', ratio=0.5): target alpha[a] = 0.5 * 1.8 = 0.9.
    # bump = 0.9 - 0.2 = 0.7, redistributed over n_other=1 (only 'b').
    # Expected final alpha: [0.9, 0.3, 1.8], mean=1.0.
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=[{'numer': 'a', 'denom': 'c', 'ratio': 0.5}],
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.5, 0.1]))
    assert torch.allclose(
        loss_fn.alpha,
        torch.tensor([0.9, 0.3, 1.8]),
        atol=1e-6,
    )
    assert torch.isclose(loss_fn.alpha.mean(), torch.tensor(1.0), atol=1e-6)
    # Cap exactly held.
    ratio = (loss_fn.alpha[0] / loss_fn.alpha[2]).item()
    assert ratio == pytest.approx(0.5, abs=1e-6)


def test_pair_cap_above_threshold_no_op():
    """When the cap is already satisfied, alpha is unchanged."""
    # Same alpha=[0.2, 1.0, 1.8] but ratio=0.05; 0.2/1.8 = 0.111 >= 0.05.
    loss_fn = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=[{'numer': 'a', 'denom': 'c', 'ratio': 0.05}],
    )
    loss_fn.update_alpha(torch.tensor([0.9, 0.5, 0.1]))
    assert torch.allclose(
        loss_fn.alpha,
        torch.tensor([0.2, 1.0, 1.8]),
        atol=1e-6,
    )


def test_pair_cap_preserves_mean_one():
    """Random F1 vectors with cap engaged: mean alpha stays 1.0."""
    g = torch.Generator().manual_seed(11)
    loss_fn = AdaptiveFocalLoss(
        n_classes=14, class_names=[f'c{i}' for i in range(14)],
        tau=1.0, momentum=0.5, warm_up_epochs=0,
        pair_caps=[{'numer': 'c0', 'denom': 'c13', 'ratio': 0.7}],
    )
    for _ in range(10):
        f1 = torch.rand(14, generator=g)
        loss_fn.update_alpha(f1)
        assert torch.isclose(loss_fn.alpha.mean(), torch.tensor(1.0), atol=1e-6)


def test_pair_cap_redistribution_uniform_on_others():
    """The bump cost is split equally across the n - 2 'other' classes."""
    # F1=[0.95, 0.5, 0.5, 0.5, 0.05]; (1-F1)=[0.05, 0.5, 0.5, 0.5, 0.95].
    # sum=2.5, n=5 -> alpha=[0.1, 1.0, 1.0, 1.0, 1.9].
    # Cap (numer='a', denom='e', ratio=0.5): target alpha[a]=0.95.
    # bump=0.85, n_other=3, each of {b, c, d} loses 0.85/3 ≈ 0.2833.
    # alpha[b]=alpha[c]=alpha[d] = 1.0 - 0.2833 = 0.7167.
    loss_fn = AdaptiveFocalLoss(
        n_classes=5, class_names=['a', 'b', 'c', 'd', 'e'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=[{'numer': 'a', 'denom': 'e', 'ratio': 0.5}],
    )
    loss_fn.update_alpha(torch.tensor([0.95, 0.5, 0.5, 0.5, 0.05]))
    expected = torch.tensor(
        [0.95, 1.0 - 0.85 / 3, 1.0 - 0.85 / 3, 1.0 - 0.85 / 3, 1.9]
    )
    assert torch.allclose(loss_fn.alpha, expected, atol=1e-5)
    # The three "other" classes share the bump cost identically.
    assert torch.isclose(loss_fn.alpha[1], loss_fn.alpha[2], atol=1e-6)
    assert torch.isclose(loss_fn.alpha[2], loss_fn.alpha[3], atol=1e-6)
    # alpha[denom] is untouched by this cap, only the bump on numer + the
    # subtractive correction on the off-pair classes runs.
    assert torch.isclose(loss_fn.alpha[4], torch.tensor(1.9), atol=1e-6)


def test_pair_cap_multi_pair_both_engage():
    """Two non-overlapping caps both fire and the mean stays 1.0.

    Pairs share no class, but the redistribution still bleeds across them
    (cap2 subtracts from members of cap1's pair via the n_other split). The
    later cap holds exactly; the earlier cap holds approximately. We assert
    both bumps were positive, the most recently applied cap is exact, and
    mean alpha is unchanged.
    """
    loss_fn = AdaptiveFocalLoss(
        n_classes=5, class_names=['a', 'b', 'c', 'd', 'e'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=[
            {'numer': 'a', 'denom': 'b', 'ratio': 0.5},  # alpha[a]=0.2, alpha[b]=1.8
            {'numer': 'c', 'denom': 'd', 'ratio': 0.5},  # alpha[c]=0.4, alpha[d]=1.6
        ],
    )
    # F1=[0.9, 0.1, 0.8, 0.2, 0.5] -> (1-F1)=[0.1, 0.9, 0.2, 0.8, 0.5].
    # sum=2.5, alpha=[0.2, 1.8, 0.4, 1.6, 1.0].
    loss_fn.update_alpha(torch.tensor([0.9, 0.1, 0.8, 0.2, 0.5]))
    # Both numer alphas lifted from their natural CDB values.
    assert loss_fn.alpha[0] > 0.2  # 'a' bumped
    assert loss_fn.alpha[2] > 0.4  # 'c' bumped
    # Mean preserved.
    assert torch.isclose(loss_fn.alpha.mean(), torch.tensor(1.0), atol=1e-6)
    # The *last* cap (c, d) holds exactly; the first cap (a, b) is partially
    # eroded by the second cap's subtractive correction on 'a' and 'b'.
    last_ratio = (loss_fn.alpha[2] / loss_fn.alpha[3]).item()
    assert last_ratio == pytest.approx(0.5, abs=1e-6)


def test_pair_cap_unknown_class_name_raises():
    with pytest.raises(ValueError, match="numer 'banana'"):
        AdaptiveFocalLoss(
            n_classes=3, class_names=['a', 'b', 'c'],
            pair_caps=[{'numer': 'banana', 'denom': 'a', 'ratio': 0.5}],
        )
    with pytest.raises(ValueError, match="denom 'kiwi'"):
        AdaptiveFocalLoss(
            n_classes=3, class_names=['a', 'b', 'c'],
            pair_caps=[{'numer': 'a', 'denom': 'kiwi', 'ratio': 0.5}],
        )


def test_pair_cap_invalid_ratio_raises():
    for bad_ratio in [0.0, -0.1, 1.5]:
        with pytest.raises(ValueError, match='ratio'):
            AdaptiveFocalLoss(
                n_classes=3, class_names=['a', 'b', 'c'],
                pair_caps=[{'numer': 'a', 'denom': 'c', 'ratio': bad_ratio}],
            )


def test_pair_cap_numer_equals_denom_raises():
    with pytest.raises(ValueError, match='must differ'):
        AdaptiveFocalLoss(
            n_classes=3, class_names=['a', 'b', 'c'],
            pair_caps=[{'numer': 'a', 'denom': 'a', 'ratio': 0.5}],
        )


def test_pair_cap_none_matches_no_cap_path():
    """pair_caps=None and pair_caps=[] should produce alpha identical to
    a fresh instance with no pair-cap config."""
    f1 = torch.tensor([0.9, 0.5, 0.1])

    plain = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
    )
    none_cap = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=None,
    )
    empty_cap = AdaptiveFocalLoss(
        n_classes=3, class_names=['a', 'b', 'c'],
        tau=1.0, momentum=0.0, warm_up_epochs=0,
        pair_caps=[],
    )

    for fn in (plain, none_cap, empty_cap):
        fn.update_alpha(f1.clone())

    assert torch.allclose(plain.alpha, none_cap.alpha, atol=1e-7)
    assert torch.allclose(plain.alpha, empty_cap.alpha, atol=1e-7)


# ---------------------------------------------------------------------------
# 9. Val-improvability gate: disabled-inert, config validation, seed/EMA/skip,
#    plateau -> revert-to-exactly-1.0 + climber kept, gating window + one-sided.
# ---------------------------------------------------------------------------

def _gate_config(**overrides) -> dict:
    """Small, fast gate config for the tests (short patience/warm-up so a few
    epochs exercise the full ramp). Override any key per test."""
    cfg = {
        'val_f1_smoothing_factor': 0.5,
        'improvement_margin': 0.02,
        'patience_epochs': 3,
        'min_epochs_before_gating': 5,
        'revert_step_per_epoch': 0.5,
        'stop_gating_after_fraction': 0.9,
    }
    cfg.update(overrides)
    return cfg


def test_gate_disabled_by_default_is_inert(class_names_3):
    """No gate config: no gate buffers, val_gate_enabled False, and apply_val_gate
    is a safe no-op that leaves the renormalised alpha untouched."""
    loss = AdaptiveFocalLoss(n_classes=3, class_names=class_names_3, warm_up_epochs=0)
    assert loss.val_gate_enabled is False
    assert not hasattr(loss, 'gate_revert_fraction')
    assert 'gate_revert_fraction' not in dict(loss.named_buffers())

    loss.update_alpha(torch.tensor([0.2, 0.5, 0.9]))
    alpha_before = loss.alpha.clone()
    loss.apply_val_gate(torch.tensor([0.3, 0.6, 0.95]), torch.tensor([True, True, True]))
    assert torch.equal(loss.alpha, alpha_before)


def test_gate_config_validation(class_names_3):
    """Enabled gate fails loud on missing n_epochs, out-of-range knobs, an empty
    gating window, and min_epochs_before_gating below the focal warm-up."""
    with pytest.raises(ValueError, match='n_epochs'):
        AdaptiveFocalLoss(n_classes=3, class_names=class_names_3,
                          val_improvability_gate=_gate_config())
    with pytest.raises(ValueError, match='smoothing'):
        AdaptiveFocalLoss(n_classes=3, class_names=class_names_3, n_epochs=20,
                          val_improvability_gate=_gate_config(val_f1_smoothing_factor=1.0))
    with pytest.raises(ValueError, match='patience'):
        AdaptiveFocalLoss(n_classes=3, class_names=class_names_3, n_epochs=20,
                          val_improvability_gate=_gate_config(patience_epochs=0))
    with pytest.raises(ValueError, match='freeze'):  # freeze (=2) <= min (=5): empty window
        AdaptiveFocalLoss(n_classes=3, class_names=class_names_3, n_epochs=4,
                          val_improvability_gate=_gate_config(min_epochs_before_gating=5,
                                                              stop_gating_after_fraction=0.5))
    with pytest.raises(ValueError, match='warm_up_epochs'):
        AdaptiveFocalLoss(n_classes=3, class_names=class_names_3, n_epochs=40, warm_up_epochs=10,
                          val_improvability_gate=_gate_config(min_epochs_before_gating=5))


def test_gate_seeds_then_emas_and_skips_absent(class_names_3):
    """Smoothed val F1 seeds on the first reading and EMAs after; a class absent
    from val (present False) is skipped, never seeded."""
    loss = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3, warm_up_epochs=0, n_epochs=20,
        val_improvability_gate=_gate_config(val_f1_smoothing_factor=0.5,
                                            min_epochs_before_gating=0),
    )
    present = torch.tensor([True, True, False])         # class c never in val
    loss.update_alpha(torch.tensor([0.5, 0.5, 0.5]))
    loss.apply_val_gate(torch.tensor([0.4, 0.8, 0.0]), present)
    assert loss.gate_smoothed_val_f1[0].item() == pytest.approx(0.4)     # seeded to first reading
    assert loss.gate_smoothed_val_f1[1].item() == pytest.approx(0.8)
    assert bool(loss.gate_val_f1_seeded[2]) is False                     # absent: never seeded
    assert loss.gate_smoothed_val_f1[2].item() == 0.0

    loss.update_alpha(torch.tensor([0.5, 0.5, 0.5]))
    loss.apply_val_gate(torch.tensor([0.6, 0.8, 0.0]), present)
    assert loss.gate_smoothed_val_f1[0].item() == pytest.approx(0.5)     # 0.5*0.4 + 0.5*0.6
    assert bool(loss.gate_val_f1_seeded[2]) is False


def test_gate_reverts_plateaued_to_exactly_one_and_keeps_climber(class_names_3):
    """An over-allocated class held flat on val reverts to EXACTLY 1.0 at full
    revert (no clawback through the renorm); a still-climbing over-allocated
    class keeps its boost; mean alpha stays 1.0."""
    loss = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3, warm_up_epochs=0, n_epochs=40,
        val_improvability_gate=_gate_config(min_epochs_before_gating=2, patience_epochs=2,
                                            revert_step_per_epoch=0.5),
    )
    present = torch.tensor([True, True, True])
    for ep in range(1, 12):
        loss.update_alpha(torch.tensor([0.30, 0.40, 0.95]))         # a, b over-allocated; c easy
        val = torch.tensor([0.30, min(0.30 + 0.05 * ep, 0.90), 0.93])  # a flat, b climbing
        loss.apply_val_gate(val, present)

    assert loss.gate_revert_fraction[0].item() == pytest.approx(1.0)       # a fully reverted
    assert loss.alpha[0].item() == pytest.approx(1.0, abs=1e-4)            # lands at exactly 1.0
    assert loss.gate_revert_fraction[1].item() == 0.0                      # b never plateaued
    assert loss.alpha[1].item() > 1.3                                      # b kept (gained) its boost
    assert loss.alpha.mean().item() == pytest.approx(1.0)


def test_gate_window_and_one_sided(class_names_3):
    """No revert before min_epochs_before_gating; revert fraction frozen after
    the freeze epoch; a below-mean class is never pulled even when flat."""
    loss = AdaptiveFocalLoss(
        n_classes=3, class_names=class_names_3, warm_up_epochs=0, n_epochs=10,
        val_improvability_gate=_gate_config(min_epochs_before_gating=3, patience_epochs=1,
                                            revert_step_per_epoch=1.0,
                                            stop_gating_after_fraction=0.7),  # freeze epoch = 7
    )
    present = torch.tensor([True, True, True])
    flat_val = torch.tensor([0.30, 0.30, 0.95])    # a, b over-allocated + flat; c easy + flat

    for _ in range(1, 4):                          # epochs 1-3: window (3 < epoch < 7) not open
        loss.update_alpha(torch.tensor([0.30, 0.30, 0.95]))
        loss.apply_val_gate(flat_val, present)
    assert loss.gate_revert_fraction.sum().item() == 0.0

    for _ in range(4, 7):                          # epochs 4-6: gating active
        loss.update_alpha(torch.tensor([0.30, 0.30, 0.95]))
        loss.apply_val_gate(flat_val, present)
    assert loss.gate_revert_fraction[0].item() == pytest.approx(1.0)
    # Easy class below the mean is never pulled (revert stays 0); here it's the
    # sole non-pulled class, so it legitimately absorbs all the freed budget.
    assert loss.gate_revert_fraction[2].item() == 0.0
    revert_at_freeze = loss.gate_revert_fraction.clone()

    for _ in range(7, 10):                         # epochs 7-9: frozen, revert fraction holds
        loss.update_alpha(torch.tensor([0.30, 0.30, 0.95]))
        loss.apply_val_gate(flat_val, present)
    assert torch.equal(loss.gate_revert_fraction, revert_at_freeze)
