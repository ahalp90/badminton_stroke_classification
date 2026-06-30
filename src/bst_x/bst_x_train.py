# Portions of this file are derived from BST (Badminton Stroke-type Transformer)
# by Jing-Yuan Chang, Copyright (c) 2025 Jing-Yuan Chang, used under the MIT
# Licence. See src/bst_x/THIRD_PARTY_NOTICES.md. This project is otherwise
# licensed LGPL-3.0-or-later.

# BST training script for ShuttleSet.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src/bst_x \
#       python -m bst_x_train

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup  # from HuggingFace, not a custom module

from pathlib import Path
from copy import deepcopy
from typing import NamedTuple
from contextlib import redirect_stdout
import argparse
import math
import time
from datetime import datetime, timedelta
import sys

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              pad_class_labels
from preparing_data.augmentations import CoupledFlip, ConstrainedJitter
from result_utils import show_f1_results, plot_confusion_matrix
from pipeline.config import (
    Taxonomy,
    derive_npy_collated_dir_basename,
    taxonomy_lookup,
)
from pipeline.data_access import env_path_or_none, load_repo_dotenv
from run_tracker import track_run, track_serial
from bst_x_common import (
    Tee,
    _write_prediction_npz,
    build_bst_x_network,
    compute_data_provenance,
    dump_topk_predictions,
)
from loss.adaptive_focal import (
    AdaptiveFocalLoss,
    accumulate_class_counts,
    per_class_f1_from_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPS_CSV = REPO_ROOT / 'notebooks' / 'clips_master.csv'


# ==========================================================================
# Hyperparameters — edit these to change experiment configuration.
# Active LR + aux schedule rationale: docs/architecture_notes/bst_x_overview.md.
# Dated retune history: docs/architecture_notes/historical_bst.md section 3.
# ==========================================================================
# collation_id picks which on-disk collation generation to read (path + manifest
# tag, e.g. 'taxon_pinned_w_preds'); it discriminates re-collations of the same
# taxonomy + split. ablation_id is a separate, nullable training-time tag (augs /
# loss / wiring on a fixed collation): manifest-only, never in the path. See
# pipeline.config.derive_npy_collated_dir_basename for the disentanglement.
class Hyp(NamedTuple):
    n_epochs: int = 80
    batch_size: int = 128
    lr: float = 5e-4
    # AdamW decoupled weight decay. 0.01 is PyTorch's AdamW default and what
    # every prior run used implicitly; kept as the default so non-sweep runs
    # barely move (norm/bias/embeddings now excluded from decay, but 0.01 on
    # them was near-inert anyway). The sweep overrides this per cell. Optimal
    # lambda for this dataset/LR/run-length is likely 0.1-0.3; see
    # docs/architecture_notes/hp_and_aug_speculations_30_05_2026.md (Q2).
    weight_decay: float = 0.01
    warm_up_step: int = 100
    taxonomy: str = 'une_v1_14'
    seq_len: int = 100
    early_stop_n_epochs: int = 40
    pose_style: str = 'JnB_bone'
    use_3d_pose: bool = False
    train_partial: float = 1.0
    use_aux_schedule: bool = True
    aux_fade_end_epoch: int = 15
    clips_csv: str = str(DEFAULT_CLIPS_CSV)
    split_column: str = 'split_v2'
    collation_id: str = 'taxon_pinned_w_preds'
    ablation_id: str | None = None
    label_smoothing: float = 0.0  # CDB-F1 cell forces LS=0; LS softens targets so confident-correct samples have p_t < 1.0, contaminating focal's per-sample hardness signal
    # Manual per-class CE weights for the wrist_smash <-> smash confusion-pair smoke test.
    # Pair-balanced (both at 2.0) so the gradient has no directional bias toward one class:
    # tests whether loss-side reweighting alone can move the wrist_smash F1 floor without
    # stealing recall from smash. Weights renormalised to mean 1.0 inside the loss build so
    # overall loss scale stays comparable to uniform CE. Set to None for uniform CE.
    class_weights: dict | None = None
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
    # Train-time augmentations. Replaces the inherited (broken) joints-only
    # RandomTranslation_batch. Flip is the literature-norm dataset-doubler;
    # constrained jitter is the corrected, pos+shuttle-only,
    # layered-conditional-bound formulation. Full design + verified code
    # traces in docs/architecture_notes/augmentation_framework.md.
    augmentation: dict = {
        'p_flip':   0.5,
        'p_jitter': 0.3,
        'cap_y':    0.05,
        'cap_x':    0.10,
        'eps':      0.15,
    }


hyp = Hyp()


# ==========================================================================
# Training and evaluation functions
# ==========================================================================

def aux_schedule_factor(epoch: int, fade_end_epoch: int) -> float:
    """Cosine warm-start-to-fade schedule for CG/AP auxiliary modules.

    Factor is 1.0 at epoch 1, 0.5 at mid-fade, and 0.0 at fade_end_epoch.
    Stays pinned at 0.0 for all epochs beyond fade_end_epoch, giving the
    transformer backbone a pure-solo phase to find its own best representation.

    Decoupling fade_end from n_epochs matters when the historical peak F1
    falls well inside the schedule: setting fade_end_epoch near (or before)
    that peak guarantees CG/AP contribution is meaningfully reduced in the
    peak region, so the experiment actually tests the hypothesis rather than
    running a near-baseline with a mild perturbation.

    :param epoch: current epoch, 1-indexed (matches the training loop).
    :param fade_end_epoch: epoch at which factor first reaches 0.0; stays 0 after.
    :return: scalar in [0, 1].
    """
    if epoch >= fade_end_epoch:
        return 0.0
    if fade_end_epoch <= 1:
        return 1.0
    progress = (epoch - 1) / (fade_end_epoch - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader,
    coupled_flip: CoupledFlip,
    constrained_jitter: ConstrainedJitter,
    n_classes: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,  # learning rate scheduler
    device,
) -> tuple[float, Tensor, Tensor, Tensor, int, int, int]:
    """Train for one epoch, accumulating per-class TP / FP / FN alongside loss.

    Per-class counts feed ``AdaptiveFocalLoss.update_alpha`` at the call site.
    They're cheap (three batched ``bincount`` calls per batch) and stay
    accumulated even when the loss has no use for them, so the train loop
    keeps a uniform return signature regardless of which loss is active.

    Augmentations fire flip-then-jitter per the framework doc. Both ops
    roll independently per-clip so within a batch some clips are
    flipped, some jittered, some both, some neither. Jitter accumulates
    two diagnostic counters across the epoch:

    - ``jitter_n_effective``: clips that received a non-zero shift, for
      ``Aug/jitter_effective_rate`` (case-1 dropout indicator).
    - ``jitter_n_oob``: clips whose effective shift pushed at least one
      previously-real shuttle frame off-screen, triggering the
      ``(0, 0)`` sentinel; for ``Aug/shuttle_oob_rate``.

    :return: ``(train_loss, tp, fp, fn, jitter_n_effective, jitter_n_oob,
        jitter_n_total)``. Counts are length-``n_classes`` int64 tensors on
        ``device``; jitter accumulators are plain ints over the epoch's clips.
    """
    model.train()  # enable dropout + batchnorm training mode
    total_loss = 0.0
    tp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fn = torch.zeros(n_classes, dtype=torch.long, device=device)
    jitter_n_effective = 0
    jitter_n_oob = 0
    jitter_n_total = 0

    for (human_pose, pos, shuttle), video_len, labels in loader:
        # .to(device) = move tensors to GPU/CPU; PyTorch needs explicit
        # placement for every tensor.
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)

        # Augmentations: flip first (clean spatial transform), then jitter.
        # Each rolls independently per-clip; coupled_flip mirrors all three
        # streams in their own coord frames and recomputes bones from the
        # post-flip+post-swap joints; constrained_jitter shifts pos+shuttle
        # only with layered-conditional bounds and zero-frame preservation.
        human_pose, pos, shuttle = coupled_flip(human_pose, pos, shuttle)
        human_pose, pos, shuttle, n_eff, n_oob = constrained_jitter(
            human_pose, pos, shuttle,
        )
        jitter_n_effective += n_eff
        jitter_n_oob += n_oob
        jitter_n_total += human_pose.shape[0]

        # Flatten last two dims (joints/bones, xy) into one feature dim for the model
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        loss: Tensor = loss_fn(logits, labels)

        # Manual gradient step: zero grads, backprop, update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()       # update learning rate according to cosine schedule

        total_loss += loss.item()  # .item() extracts Python float from single-element tensor

        # Per-class confusion counts on argmax preds. no_grad() because preds
        # are detached labels; nothing here needs an autograd graph.
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            batch_tp, batch_fp, batch_fn = accumulate_class_counts(
                preds, labels, n_classes,
            )
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

    train_loss = total_loss / len(loader)
    return train_loss, tp, fp, fn, jitter_n_effective, jitter_n_oob, jitter_n_total


@torch.no_grad()  # disables gradient computation — saves memory during eval
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device,
    n_classes: int,
):
    model.eval()  # disable dropout + set batchnorm to eval mode
    total_loss = 0.0
    # Accumulate per-class TP/FP/FN on device (mirrors train_one_epoch);
    # one .cpu() after the loop, not four per batch.
    cum_tp = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_fp = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_fn = torch.zeros(n_classes, dtype=torch.long, device=device)
    cum_top2 = 0  # ground truth among the two highest logits, summed over samples
    cum_n = 0     # total samples seen

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        loss: Tensor = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        batch_tp, batch_fp, batch_fn = accumulate_class_counts(
            preds, labels, n_classes,
        )
        cum_tp += batch_tp
        cum_fp += batch_fp
        cum_fn += batch_fn

        # Top-2 accuracy needs the two highest logits, so it's the one metric
        # not already in the confusion counts; accumulate it here.
        cum_n += labels.size(0)
        top2_idx = logits.topk(2, dim=1).indices
        cum_top2 += int((top2_idx == labels.unsqueeze(1)).any(dim=1).sum())

    cum_tp = cum_tp.cpu()
    cum_fp = cum_fp.cpu()
    cum_fn = cum_fn.cpu()
    val_loss = total_loss / len(loader)

    # Per-class F1, then macro average (mean across classes)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0  # classes with no predictions get NaN -> 0

    # Only classes present in the val set count toward macro/min. Generic
    # zero-support guard: any class with no ground-truth this epoch would
    # otherwise score F1=0 by construction, dragging macro down by 1/n
    # and pinning min at 0.
    present = (cum_tp + cum_fn) > 0
    if present.any():
        f1_score_avg = f1_score[present].mean()
        f1_score_min = f1_score[present].min()
    else:
        f1_score_avg = torch.tensor(0.0)
        f1_score_min = torch.tensor(0.0)

    # Accuracy is exactly correct/total: every sample is a TP for its class (if
    # right) or an FN for it (if wrong), so the correct count is sum(cum_tp).
    accuracy = float(cum_tp.sum() / cum_n) if cum_n else 0.0
    top2_accuracy = cum_top2 / cum_n if cum_n else 0.0
    return val_loss, f1_score_avg, f1_score_min, f1_score, present, accuracy, top2_accuracy


# ==========================================================================
# Training loop with TensorBoard logging and early stopping
# ==========================================================================

def _build_loss_fn(
    n_classes: int,
    class_ls: list[str],
    taxonomy: Taxonomy,
    device,
):
    """Resolve hyp's three loss branches (CE / class-weighted CE / adaptive-focal)
    into a single ``loss_fn`` instance.

    Owns the three fail-loud guards: ``use_val_improvability_gate`` needs
    ``adaptive_focal``; ``adaptive_focal`` is mutually exclusive with
    ``class_weights``; ``adaptive_focal`` requires ``label_smoothing=0.0``.
    All read the module-global ``hyp`` to mirror the pre-B7 shape (the rest of
    ``train_network`` does the same).

    ``label_smoothing`` softens targets from [0,1] to reduce overconfidence.
    BST paper / TemPose default is 0.1; we sweep this knob to test whether it's
    bottlenecking the small-support classes that lose ground when the cleaner
    Phase-2 pose data lifts the head of the F1 distribution. See
    docs/architecture_notes/hparams_sweep_speculations.md.

    ``class_weights``: optional manual per-class loss multipliers. Used as a
    smoke test for whether loss-side reweighting can move the bottleneck F1
    classes (wrist_smash + its confusion partner smash). Renormalised to mean
    1.0 so the overall loss magnitude stays comparable to uniform CE (keeps LR /
    grad-clip behaviour aligned across cells). None = uniform.

    ``adaptive_focal``: class-F1-driven CDB-loss with optional focal
    modulation. Replaces the static class_weights lever with an EMA-smoothed
    per-class weight that re-prioritises classes whose train F1 stays low.
    Mutually exclusive with class_weights and forces label_smoothing=0 (LS
    softens targets, contaminating focal's per-sample hardness estimate). The
    val-improvability gate modulates the adaptive-focal alpha, so it can only
    run when adaptive_focal is engaged. Fail loud rather than silently ignoring
    the flag (the gate config would otherwise be dropped on the floor).
    """
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
        return loss_fn
    if hyp.class_weights:
        weights = torch.ones(n_classes, device=device)
        for cls_name, multiplier in hyp.class_weights.items():
            if cls_name not in class_ls:
                raise ValueError(
                    f"class_weights key '{cls_name}' not in the taxonomy "
                    f"{taxonomy.name!r} class list ({len(class_ls)} classes): "
                    f"{class_ls}."
                )
            weights[class_ls.index(cls_name)] = multiplier
        weights = weights * (n_classes / weights.sum())  # renormalise mean to 1.0
        print(f"[loss] class-weighted CE (renormalised, mean=1.0):")
        for i, c in enumerate(class_ls):
            print(f"    {c:25s} weight={weights[i].item():.3f}")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=hyp.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)


def _split_param_groups(model: nn.Module):
    """Decay vs no-decay walk for AdamW. Returns the two raw lists; the caller
    builds the AdamW param-group dicts (so ``hyp.weight_decay`` / ``hyp.lr``
    reads stay co-located with the optimiser construction).

    Excludes norm gains, biases, and the learned tokens / positional embeddings
    from decay: decaying an LN/BN gain pulls its scale toward zero, and decaying
    a sinusoidally-seeded positional embedding erodes the positional signal.
    Matters at the lambda 0.1-0.4 the sweep covers; standard transformer recipe
    (Wang & Aitchison don't decay normalisation layers). ``ndim<=1`` catches
    every norm gain/beta and bias; the two name hints catch the five ndim>=2
    BST-owned params a shape rule misses. Verified split for BST_CG_AP:
    27 decay / 55 no-decay tensors (model-pinned: a different variant or a
    requires_grad change moves the count).
    """
    no_decay_name_hints = ('embedding_', 'learned_token_')
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        norm_or_bias = param.ndim <= 1
        token_or_posemb = any(hint in name for hint in no_decay_name_hints)
        (no_decay if norm_or_bias or token_or_posemb else decay).append(param)
    return decay, no_decay


def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    n_bones,
    n_classes: int,
    class_ls: list[str],
    taxonomy: Taxonomy,
    tb_dir: Path | None = None,
):
    # tb_dir lands the event files under experiments/<run_id>/tb/serial_N/ so
    # TB folders pair with the run they came from. Default SummaryWriter() writes
    # to ./runs/<host_time>/, which is what older runs used.
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else SummaryWriter()

    # Locked Task-2 augmentation set: centreline flip across all three streams
    # (with COCO bilateral joint-index swap and bone recompute) plus
    # constrained pos+shuttle jitter (layered conditional bounds, joints
    # untouched). Bone recompute requires the JnB_bone pose style; other
    # styles (J_only, JnB_interp, Jn2B) need their own recompute helpers
    # which are out of scope per docs/architecture_notes/augmentation_framework.md.
    if hyp.pose_style != 'JnB_bone':
        raise NotImplementedError(
            f'Augmentation framework currently supports pose_style=JnB_bone only; '
            f'got {hyp.pose_style!r}. Bone recompute via the bone_pairs table is the '
            f'mechanism that propagates the flip+swap into bones; J_only has no bones, '
            f'JnB_interp uses joint-pair midpoints, Jn2B uses both. Lift the equivalents '
            f'to torch in preparing_data/augmentations.py before re-enabling the others.'
        )
    # Direct index rather than .get(key, default): the Hyp always carries all
    # five aug keys (dict literal + all-or-nothing CLI override), so a missing
    # key is a malformed config and should fail loud, not train on a default.
    aug_cfg = hyp.augmentation
    coupled_flip = CoupledFlip(
        p=aug_cfg['p_flip'],
        n_joints=17,
        n_bones=n_bones,
    )
    constrained_jitter = ConstrainedJitter(
        p_roll=aug_cfg['p_jitter'],
        cap_y=aug_cfg['cap_y'],
        cap_x=aug_cfg['cap_x'],
        eps=aug_cfg['eps'],
    )
    print(
        f"[aug] coupled flip p={coupled_flip.p}, "
        f"constrained jitter p={constrained_jitter.p_roll} "
        f"(cap_y={constrained_jitter.cap_y}, cap_x={constrained_jitter.cap_x}, "
        f"eps={constrained_jitter.eps})"
    )

    loss_fn = _build_loss_fn(n_classes, class_ls, taxonomy, device)

    # AdamW with decoupled weight decay. _split_param_groups owns the decay
    # rules; this site owns the per-group weight_decay + hyp.lr wiring so the
    # optimiser construction stays co-located with its hparams.
    decay, no_decay = _split_param_groups(model)
    print(f'[optim] AdamW lr={hyp.lr} weight_decay={hyp.weight_decay} '
          f'(decay={len(decay)} tensors, no_decay={len(no_decay)})')
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': hyp.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=hyp.lr,
    )
    # Cosine schedule: LR ramps up during warmup, then decays following a cosine curve.
    # HF formula: lr_factor = 0.5 * (1 + cos(pi * 2 * num_cycles * progress))
    #   num_cycles=0.5 -> LR ends at 0 (full standard cosine descent)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * len(train_loader)),  # total batches across all epochs
        num_cycles=0.5
    )

    # Track top-2 of each metric (for HParams summary + verifying early-stop vs crash)
    best_macro = second_macro = 0.0
    best_macro_epoch = second_macro_epoch = 0
    best_min = second_min = 0.0
    best_min_epoch = second_min_epoch = 0
    best_val_loss, best_val_loss_epoch = float('inf'), 0
    early_stop_count = 0

    # Per-class val F1 snapshot, captured at the best-macro epoch (not a
    # per-class argmax across epochs) so the recorded breakdown matches the
    # checkpoint that actually gets saved. Surfaced to the serial manifest.
    best_val_f1_per_class = None
    best_val_present = None
    best_val_accuracy = None
    best_val_top2 = None
    best_macro_epoch_snap = None

    for epoch in range(1, hyp.n_epochs+1):
        # Auxiliary module schedule: cosine fade of CG/AP from 1.0 -> 0.0 across the run.
        # When disabled, factor stays at 1.0 -> identical to unscheduled BST_CG_AP.
        if hyp.use_aux_schedule:
            aux_factor = aux_schedule_factor(epoch, hyp.aux_fade_end_epoch)
        else:
            aux_factor = 1.0
        model.set_schedule_factors(cg_factor=aux_factor, ap_factor=aux_factor)

        t0 = time.time()
        train_loss, train_tp, train_fp, train_fn, \
            jitter_n_eff, jitter_n_oob, jitter_n_total = train_one_epoch(
            model=model,
            loader=train_loader,
            coupled_flip=coupled_flip,
            constrained_jitter=constrained_jitter,
            n_classes=n_classes,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
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
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '
              f'val_loss={val_loss:.3f}, macro_f1={f1_score_avg:.3f}, min_f1={f1_score_min:.3f} '
              f'- {t1 - t0:.2f} s')

        if isinstance(loss_fn, AdaptiveFocalLoss):
            # Top-3 / bot-3 alpha summary so the operator can eyeball whether
            # the loss is reweighting toward the struggling classes each epoch.
            alpha_np = loss_fn.alpha.detach().cpu().numpy()
            order = alpha_np.argsort()
            print('  alpha bot3: ' + ' '.join(
                f'{class_ls[i]}={alpha_np[i]:.2f}' for i in order[:3]
            ))
            print('  alpha top3: ' + ' '.join(
                f'{class_ls[i]}={alpha_np[i]:.2f}' for i in order[-3:][::-1]
            ))

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('F1/Val_macro', f1_score_avg, epoch)
        writer.add_scalar('F1/Val_min', f1_score_min, epoch)
        # Train F1 macro/min summaries mirror the val pair above, so the
        # train-vs-val gap reads off two scalars per epoch instead of needing
        # to re-aggregate the per-class arrays. .mean()/.min() over the
        # length-n_classes tensor of active-class F1s, .item() unwraps to float.
        writer.add_scalar('F1_train/macro', train_per_class_f1.mean().item(), epoch)
        writer.add_scalar('F1_train/min', train_per_class_f1.min().item(), epoch)
        writer.add_scalar('Schedule/aux_factor', aux_factor, epoch)
        # Cosine LR per epoch. Deterministic from the schedule, but logging it
        # saves the reconstruction and overlays cleanly with the per-class F1 /
        # alpha arcs. get_last_lr()[0] = LR after this epoch's final step.
        writer.add_scalar('Schedule/learning_rate', scheduler.get_last_lr()[0], epoch)
        # Jitter effective rate: fraction of clips that rolled yes AND had at
        # least one non-degenerate axis. Watching this scalar shows whether the
        # case-1 (fully-degenerate envelope) skip rate is eating into the
        # nominal p_jitter target. See augmentation_framework.md.
        jitter_effective_rate = (
            jitter_n_eff / jitter_n_total if jitter_n_total > 0 else 0.0
        )
        writer.add_scalar('Aug/jitter_effective_rate', jitter_effective_rate, epoch)
        # Shuttle OOB rate: fraction of clips where the effective shift
        # pushed a previously-real shuttle frame off-screen, triggering the
        # (0, 0) sentinel. Diagnostic for the cap_x trade-off the doc flags
        # around edge-of-frame shuttle classes (cross_court_net_shot, rush
        # trajectories). High rate = cap_x is replacing a meaningful fraction
        # of real shuttle observations with the off-screen sentinel.
        shuttle_oob_rate = (
            jitter_n_oob / jitter_n_total if jitter_n_total > 0 else 0.0
        )
        writer.add_scalar('Aug/shuttle_oob_rate', shuttle_oob_rate, epoch)
        for i, c in enumerate(class_ls):
            writer.add_scalar(f'F1_train/{c}', train_per_class_f1[i].item(), epoch)
            # Val per-class F1 only for classes present in val this epoch; an
            # absent class scores F1=0 by construction and would read as a real
            # regression on the TB curve.
            if present[i]:
                writer.add_scalar(f'F1_val/{c}', f1_per_class[i].item(), epoch)
            if isinstance(loss_fn, AdaptiveFocalLoss):
                writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)
                # Per-class revert fraction (0 = full adaptive alpha, 1 = pulled
                # all the way to the renorm mean) so the gate's action is visible.
                if loss_fn.val_gate_enabled:
                    writer.add_scalar(
                        f'Revert/{c}', loss_fn.gate_revert_fraction[i].item(), epoch
                    )

        curr_macro, curr_min = f1_score_avg.item(), f1_score_min.item()

        # Early stop + snapshot best weights (piggybacks on new-best detection)
        early_stop_count += 1
        if curr_macro > best_macro:
            second_macro, second_macro_epoch = best_macro, best_macro_epoch
            best_macro, best_macro_epoch = curr_macro, epoch
            # state_dict() = snapshot of all model weights as a dict
            # deepcopy because state_dict returns references that would change as training continues
            best_state = deepcopy(model.state_dict())
            # Snapshot the per-class val F1 at this same best-macro epoch so the
            # recorded breakdown matches the saved checkpoint.
            best_val_f1_per_class = f1_per_class.detach().cpu().numpy()
            best_val_present = present.detach().cpu().numpy()
            best_val_accuracy = val_accuracy
            best_val_top2 = val_top2
            best_macro_epoch_snap = epoch
            print(f'Picked! => Best value {curr_macro:.3f}')
            # Compact per-class snapshot on new-best epochs: top-5 and bot-5
            # of present classes, one line each. Full per-class breakdown
            # lands in the test-time log at the end of each serial.
            present_idx = present.nonzero(as_tuple=True)[0].tolist()
            scored = sorted(
                [(class_ls[i], f1_per_class[i].item()) for i in present_idx],
                key=lambda t: t[1],
            )
            print('  val top5: ' + ' '.join(
                f'{n}={v:.2f}' for n, v in reversed(scored[-5:])
            ))
            print('  val bot5: ' + ' '.join(
                f'{n}={v:.2f}' for n, v in scored[:5]
            ))
            early_stop_count = 0
        elif curr_macro > second_macro:
            second_macro, second_macro_epoch = curr_macro, epoch

        if curr_min > best_min:
            second_min, second_min_epoch = best_min, best_min_epoch
            best_min, best_min_epoch = curr_min, epoch
        elif curr_min > second_min:
            second_min, second_min_epoch = curr_min, epoch

        best_val_loss, best_val_loss_epoch = min(
            (best_val_loss, best_val_loss_epoch), (val_loss, epoch)
        )

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_macro:.3f}')
            break

    # Save best checkpoint and restore it into the model. Done before TB
    # hparam logging so a logging failure doesn't lose the trained weights.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)

    # HParams summary: one row per run, sortable in TB's HParams tab.
    # stopped_epoch - best_macro_epoch == early_stop_n_epochs confirms clean early-stop.
    # Coerce non-scalar values (dicts, None, etc.) to strings; TB's add_hparams
    # only accepts int / float / str / bool / Tensor.
    hparam_dict = {}
    for key, value in hyp._asdict().items():
        is_tb_scalar = isinstance(value, (int, float, str, bool)) or torch.is_tensor(value)
        hparam_dict[key] = value if is_tb_scalar else str(value)

    writer.add_hparams(
        hparam_dict=hparam_dict,
        metric_dict={
            'best/macro_f1':        best_macro,
            'best/macro_f1_epoch':  best_macro_epoch,
            'best/macro_f1_2nd':    second_macro,
            'best/macro_f1_2nd_ep': second_macro_epoch,
            'best/min_f1':          best_min,
            'best/min_f1_epoch':    best_min_epoch,
            'best/min_f1_2nd':      second_min,
            'best/min_f1_2nd_ep':   second_min_epoch,
            'best/val_loss':        best_val_loss,
            'best/val_loss_epoch':  best_val_loss_epoch,
            'stopped_epoch':        epoch,
        },
        run_name='.',
        global_step=epoch,
    )
    writer.close()

    # Val metrics at the best-macro epoch (the checkpoint that gets saved):
    # macro/min/accuracy/top-2 + the present-class per-class F1, for the serial
    # manifest (extra.val_at_best_macro_epoch). macro/min are the mean/min of the
    # snapshot per-class, so they stay exactly consistent with the breakdown.
    # None if no epoch ever beat the macro=0.0 init (degenerate run).
    if best_val_f1_per_class is not None:
        per_class = {
            class_ls[i]: float(best_val_f1_per_class[i])
            for i in range(len(class_ls))
            if best_val_present[i]
        }
        f1s = list(per_class.values())
        val_at_best = {
            'epoch': best_macro_epoch_snap,
            'macro_f1': sum(f1s) / len(f1s),
            'min_f1': min(f1s),
            'accuracy': best_val_accuracy,
            'top2_accuracy': best_val_top2,
            'per_class_f1': per_class,
        }
    else:
        val_at_best = None
    return model, val_at_best


# ==========================================================================
# Task: orchestrates data loading, model creation, training, and evaluation
# ==========================================================================


class Task:
    def __init__(self, n_joints=17, taxonomy: Taxonomy = None,
                 weight_dir: Path = Path('weight')) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        # Head dim and class names come straight off the taxonomy now: labels.npy
        # lands in [0, taxonomy.n_classes) at collation time, so there's no
        # runtime active/full remap and no data-derived head sizing.
        self.taxonomy = taxonomy or taxonomy_lookup(hyp.taxonomy)
        # Where to save/load weights for this run. Caller should pass a
        # per-invocation subdir (e.g. weight/run_YYYYMMDD_HHMMSS) so fresh
        # runs never collide with older weights — see __main__ setup.
        self.weight_dir = weight_dir

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='Jn2B',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=train_partial
            )

        self.pose_style = pose_style
        self._assert_label_coverage()

    def _assert_label_coverage(self) -> None:
        """Contract guard replacing the old runtime active-class adapter.

        Labels.npy lands in ``[0, taxonomy.n_classes)`` at collation time, so
        the head is the full taxonomy. Two invariants, both fail loud:

        - train must cover every class in the taxonomy. A class the head can
          emit but train never teaches would carry a label-smoothed ghost
          gradient every step; better to refuse the run.
        - val/test must not carry any class absent from train (a class train
          never saw can't be evaluated meaningfully).

        Reads labels post-``train_partial`` slicing, so a too-aggressive
        partial that starves a class is caught here too.
        """
        expected = set(range(self.taxonomy.n_classes))
        train_present = {int(x) for x in np.unique(self.train_loader.dataset.labels)}
        val_present = {int(x) for x in np.unique(self.val_loader.dataset.labels)}
        test_present = {int(x) for x in np.unique(self.test_loader.dataset.labels)}

        missing_in_train = expected - train_present
        if missing_in_train:
            named = [self.taxonomy.classes[i] for i in sorted(missing_in_train)]
            raise ValueError(
                f'taxonomy {self.taxonomy.name!r} has {len(expected)} classes '
                f'but train covers only {len(train_present)}. Missing class '
                f'indices: {sorted(missing_in_train)} ({named}). Either lift '
                f'train_partial (currently {hyp.train_partial}) or use a '
                f'taxonomy whose head matches what train can teach.'
            )

        n = self.taxonomy.n_classes
        for split_name, present in (('val', val_present), ('test', test_present)):
            rogue = present - train_present
            if rogue:
                # After the coverage check, train_present holds every in-range
                # class, so a rogue index is an out-of-range (corrupt) label;
                # name it safely rather than IndexError on classes[i].
                named = [
                    self.taxonomy.classes[i] if 0 <= i < n else f'<oob:{i}>'
                    for i in sorted(rogue)
                ]
                raise ValueError(
                    f'{split_name} has classes absent from train: '
                    f'{sorted(rogue)} ({named}). Fix the split assignment in '
                    f'clips_master.csv or pick a taxonomy whose classes match '
                    f'the data shape.'
                )

    def get_network_architecture(self, model_name='BST_X', in_channels=2):
        """Create the model at the taxonomy head dim and ground its inputs.

        :param in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz).

        Output dim is ``taxonomy.n_classes`` directly; labels on disk are
        already in that index space (no runtime remap), and
        ``_assert_label_coverage`` has confirmed train teaches the whole head.
        """
        self.net, self.n_bones = build_bst_x_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            in_channels=in_channels,
            n_class=self.taxonomy.n_classes,
            seq_len=hyp.seq_len,
            device=self.device,
        )
        self.model_name = model_name

    def seek_network_weights(self, model_info='', serial_no=1, tb_dir: Path | None = None):
        """Load existing weights if found, otherwise train from scratch.
        Weight filenames encode the full experiment config, e.g.:
        'bst_x_JnB_bone_between_2_hits_with_max_limits_seq_100_bst_24_2.pt'

        :return: ``(weight_existed, val_at_best)``. ``weight_existed`` is True
            when a checkpoint was loaded (no training ran), False when freshly
            trained. ``val_at_best`` is the per-class val F1 snapshot from
            ``train_network`` (None on the load path or a degenerate run).
        """
        model_info = f'_{model_info}' if model_info != '' else ''
        taxonomy_info = f'_{self.taxonomy.name}'
        serial_str = f'_{serial_no}' if serial_no != 1 else ''

        model_postfix = '_' + self.pose_style \
            + model_info + taxonomy_info + serial_str

        save_name = self.model_name.lower()
        save_name += model_postfix

        self.model_name += model_postfix

        weight_path = self.weight_dir / f'{save_name}.pt'
        self.weight_path = weight_path
        if weight_path.exists():
            self.net.load_state_dict(
                torch.load(str(weight_path), map_location=self.device, weights_only=True)
            )
            return True, None  # weight already existed; no fresh val snapshot
        else:
            train_t0 = time.time()
            self.net, val_at_best = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path,
                n_bones=self.n_bones,
                n_classes=self.taxonomy.n_classes,
                class_ls=list(self.taxonomy.classes),
                taxonomy=self.taxonomy,
                tb_dir=tb_dir,
            )
            t = timedelta(seconds=int(time.time() - train_t0))
            print(f'Total training time: {t}')
            return False, val_at_best  # newly trained

    def test(self, dump: dict, show_details=False, show_confusion_matrix=False) -> dict:
        """Derive test top-1 metrics from a precomputed dump.

        ``dump`` is one split's output from ``dump_topk_predictions``: top-1 reads
        straight off ``y_pred_top1`` (argmax, unified in Batch 2), no second forward
        pass through the test loader.
        """
        pred = torch.from_numpy(dump['y_pred_top1'])
        gt = torch.from_numpy(dump['y_true'])
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(
            pred, gt, num_classes=self.taxonomy.n_classes, average=None
        )

        # Mirror validate(): generic zero-support guard. Any class with no
        # ground truth in the test set would otherwise score F1=0 by
        # construction, dragging macro down by 1/n and pinning min at 0.
        present = torch.bincount(gt, minlength=self.taxonomy.n_classes) > 0
        present_idx = present.nonzero(as_tuple=True)[0].tolist()
        class_ls = list(self.taxonomy.classes)

        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each[present_idx] if present_idx else f1_score_each,
            class_ls=pad_class_labels(
                [class_ls[i] for i in present_idx] if present_idx else class_ls
            ),
            show_details=show_details
        )

        acc = torch.sum(pred == gt).item() / len(pred)
        print('Accuracy:', f'{acc:.3f}')

        if show_confusion_matrix:
            plot_confusion_matrix(
                y_true=gt,
                y_pred=pred,
                need_pre_argmax=False,
                model_name=self.model_name,
                font_size=6,
                save=False
            )

        if present_idx:
            macro_f1 = float(f1_score_each[present_idx].mean().item())
            min_f1 = float(f1_score_each[present_idx].min().item())
            per_class_f1 = {
                class_ls[i]: float(f1_score_each[i].item()) for i in present_idx
            }
        else:
            macro_f1 = 0.0
            min_f1 = 0.0
            per_class_f1 = {}

        return {
            'macro_f1':     macro_f1,
            'min_f1':       min_f1,
            'accuracy':     float(acc),
            'num_strokes':  int(len(pred)),
            'per_class_f1': per_class_f1,
        }

    def test_topk_acc(self, dump: dict, k=2) -> dict:
        """Derive top-k accuracy from a precomputed dump's raw logits.

        Re-derives top-k from a fresh ``torch.topk(logits, k=k)`` rather than
        slicing the stored ``topk_idx[:, :k]`` (the dump runs at k=5; slicing
        breaks ties differently from a k=2 topk on rank-boundary rows). Real
        trained logits are tie-free, so this matches a 3-pass top-k on actual data.
        """
        assert k > 1, 'k should be > 1'
        logits = torch.from_numpy(dump['logits'])
        gt = torch.from_numpy(dump['y_true'])
        pred = torch.topk(logits, k=k, dim=1).indices
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')
        return {f'top{k}_accuracy': float(acc)}

    def dump_predictions(
        self, run_dir: Path, serial_no: int, k: int = 5,
    ) -> dict[str, dict]:
        """Dump per-split prediction npz (raw logits + top-k + ground truth).

        The per-stroke-logits payload that motivated the refactor: lets the FE
        show per-clip confidence and any consumer fit post-hoc temperature
        scaling without re-running inference. One npz per split per serial under
        ``run_dir/predictions/``. Non-best serials are pruned manually after the
        runner finishes (no auto-deletion).

        Each split is re-read through a fresh ``shuffle=False`` loader, and the
        npz carries its own ``clip_stems`` column row-aligned with ``logits`` /
        ``y_true``. The stems come from the in-memory dataset, so they track the
        rows the model actually saw -- after the zero-length-clip drop and any
        train_partial reorder -- NOT the raw on-disk ``clip_stems.npy``. The
        FE-shape JSON converter joins row -> stem inside the npz, no external
        sidecar and no re-deriving the collation filters.

        Returns the per-split dump dicts so the caller can derive test metrics
        from the same forward pass (B2: one test pass, not three).

        :param run_dir: experiments/<run_id>/ for this run.
        :param serial_no: serial whose weights are currently loaded in self.net.
        :param k: top-k width recorded per row.
        :return: ``{split_name: dump_dict}`` for train / val / test.
        """
        out_dir = run_dir / 'predictions'
        out_dir.mkdir(parents=True, exist_ok=True)
        sources = (
            ('train', self.train_loader),
            ('val',   self.val_loader),
            ('test',  self.test_loader),
        )
        dumps: dict[str, dict] = {}
        for split_name, source in sources:
            dataset = source.dataset
            ordered = DataLoader(
                dataset, batch_size=source.batch_size,
                shuffle=False, num_workers=0, pin_memory=False,
            )
            dump = dump_topk_predictions(self.net, ordered, self.device, k=k)
            _write_prediction_npz(
                out_dir / f'{split_name}_serial_{serial_no}.npz',
                dump, dataset, self.taxonomy, run_dir.name, serial_no,
            )
            dumps[split_name] = dump
        return dumps


# ==========================================================================
# Per-run taxonomy printout
# ==========================================================================

def _print_taxonomy_block(taxonomy: Taxonomy, tee) -> None:
    """Loud one-time taxonomy summary at run start, captured by the tee'd log.

    Replaces the old ``_validate_and_record_arch`` + ``extra.arch`` manifest
    block: with labels in active class space and the head pinned to the
    taxonomy, there's no data-derived architecture to validate or record. The
    resolved class list lives in the manifest's ``config.classes`` field
    (written at ``track_run`` time); the train/val/test coverage invariants are
    enforced by ``Task._assert_label_coverage``.

    :param taxonomy: the resolved taxonomy the run trains under.
    :param tee: file-like writing to terminal + log_path so the line lands in both.
    """
    with redirect_stdout(tee):
        print(f'[taxonomy] {taxonomy.name}: {taxonomy.n_classes} classes, '
              f'has_sides={taxonomy.has_sides}, has_unknown={taxonomy.has_unknown}')
        print(f'[taxonomy] classes: {list(taxonomy.classes)}')


# ==========================================================================
# Main: train and test on ShuttleSet
# ==========================================================================

if __name__ == '__main__':
    # Load .env so BST_X_COLLATED_DATA_ROOT (and any BST_* paths) resolve the
    # same way the collator does; shell exports still win. No-op without .env.
    load_repo_dotenv()

    # CLI is wrapper-friendly: hparam_sweep.py drives per-serial invocations
    # by setting --serial-no with a fixed --run-id and --log-path so all five
    # serials share a run dir and a single test log. Manual invocations leave
    # everything unset to fall back to the module-level Hyp defaults plus a
    # fresh timestamped run dir / log file.
    parser = argparse.ArgumentParser(
        description='BST training entry point. CLI flags exist mainly for the '
                    'hparam_sweep wrapper; running with no flags trains a full '
                    '5-serial run from the module-level Hyp defaults.',
    )
    parser.add_argument(
        '--serial-no', type=int, default=None,
        help='Run only this serial (1-5) and exit. Used by hparam_sweep to '
             'pause between serials for kill checks. Requires --log-path and '
             '--run-id when serial-no > 1.',
    )
    parser.add_argument(
        '--run-id', type=str, default=None,
        help='Resume into an existing experiments/<run_id>/ dir. Required when '
             '--serial-no > 1; optional otherwise (a fresh run_<timestamp> is '
             'minted if absent).',
    )
    parser.add_argument(
        '--log-path', type=str, default=None,
        help='Pin the test log file path. Required when --serial-no > 1 so all '
             'serials append to the same log file. Without it, each invocation '
             'creates a fresh test_logs/test_<timestamp>.log.',
    )
    parser.add_argument('--p-flip', type=float, default=None)
    parser.add_argument('--p-jitter', type=float, default=None)
    parser.add_argument('--cap-y', type=float, default=None)
    parser.add_argument('--cap-x', type=float, default=None)
    parser.add_argument('--eps', type=float, default=None)
    # Cell selectors for collation_runner.py (and any manual override). All
    # optional; absent ones fall back to the module-level Hyp defaults.
    # --taxonomy / --split-column / --collation-id together pick the on-disk
    # collation to read; --ablation-id is the nullable training-time tag (augs
    # / loss / wiring on a fixed collation), manifest-only, never in the path.
    parser.add_argument('--taxonomy', default=None)
    parser.add_argument('--split-column', default=None)
    parser.add_argument('--collation-id', default=None)
    parser.add_argument('--ablation-id', default=None)
    # Swept AdamW weight decay (the WD-sweep dimension), overriding the Hyp
    # default. Absent leaves the module default (0.01). Applies to the decay
    # param group only; the no-decay group stays at 0.0 regardless.
    parser.add_argument('--weight-decay', type=float, default=None)
    # Enable/disable the val-improvability alpha gate, overriding the Hyp default.
    # --val-improvability-gate turns it on, --no-val-improvability-gate off;
    # absent leaves the module default (off). Requires adaptive_focal.
    parser.add_argument(
        '--val-improvability-gate',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    # Testing-only n_epochs override (e.g. --serial-no 1 short-run bit-exacts
    # that don't want the full 80-epoch default). Not piped through
    # hparam_sweep; for production sweeps, edit Hyp.n_epochs directly.
    parser.add_argument('--n-epochs', type=int, default=None)
    args = parser.parse_args()

    # Per-serial invocation contract: pass all three sharing-flags together so
    # every serial lands in one run dir with one continuous log file. The
    # runner drives serial count per cell (5 default, 10 for headline cells),
    # so there's no fixed upper bound here beyond "1-indexed".
    if args.serial_no is not None:
        if args.serial_no < 1:
            raise ValueError(
                f'--serial-no must be >= 1, got {args.serial_no!r}.'
            )
        if args.serial_no > 1 and (args.log_path is None or args.run_id is None):
            raise ValueError(
                '--serial-no > 1 requires both --log-path and --run-id so '
                'subsequent serials append to the same log and share the run dir.'
            )

    # Augmentation CLI overrides are all-or-nothing. Wrapper passes the full
    # cell-config dict (base + overrides resolved); manual invocations leave
    # them all None and use the module-level Hyp defaults.
    aug_overrides = [args.p_flip, args.p_jitter, args.cap_y, args.cap_x, args.eps]
    if any(x is not None for x in aug_overrides):
        if not all(x is not None for x in aug_overrides):
            raise ValueError(
                'Augmentation CLI overrides must be all-or-nothing. Pass either '
                'all five (--p-flip --p-jitter --cap-y --cap-x --eps) or none.'
            )
        hyp = hyp._replace(augmentation={
            'p_flip':   args.p_flip,
            'p_jitter': args.p_jitter,
            'cap_y':    args.cap_y,
            'cap_x':    args.cap_x,
            'eps':      args.eps,
        })

    # Cell selectors: override the Hyp defaults when the runner (or a manual
    # invocation) passes them. Each is independent and nullable.
    cell_overrides = {}
    if args.taxonomy is not None:
        cell_overrides['taxonomy'] = args.taxonomy
    if args.split_column is not None:
        cell_overrides['split_column'] = args.split_column
    if args.collation_id is not None:
        cell_overrides['collation_id'] = args.collation_id
    if args.ablation_id is not None:
        cell_overrides['ablation_id'] = args.ablation_id
    if args.val_improvability_gate is not None:
        cell_overrides['use_val_improvability_gate'] = args.val_improvability_gate
    if args.weight_decay is not None:
        cell_overrides['weight_decay'] = args.weight_decay
    if args.n_epochs is not None:
        cell_overrides['n_epochs'] = args.n_epochs
    if cell_overrides:
        hyp = hyp._replace(**cell_overrides)

    # Resolve the taxonomy; its canonical name drives the on-disk dir +
    # weight-file naming, matching what the collator wrote.
    taxonomy = taxonomy_lookup(hyp.taxonomy)

    # Collated dir naming via shared helper (mirrored on the prepare_train
    # writer side); see ``pipeline.config.derive_npy_collated_dir_basename``.
    if hyp.seq_len not in (30, 100):
        raise NotImplementedError(f'Unsupported hyp.seq_len={hyp.seq_len!r}; expected 30 or 100.')
    npy_collated_dir = derive_npy_collated_dir_basename(
        use_3d_pose=hyp.use_3d_pose,
        seq_len=hyp.seq_len,
        split_column=hyp.split_column,
        collation_id=hyp.collation_id,
    )

    # Weights filename suffix. Independent of the collated-dir name; encodes
    # config knobs that change per run (seq_len-derived window tag, 3d flag,
    # train_partial). Empty string is a valid value (seq_len=30, 2D, full data).
    str_3d = '_3d' if hyp.use_3d_pose else ''
    model_info_parts: list[str] = []
    if hyp.seq_len == 100:
        model_info_parts.append(f'between_2_hits_with_max_limits_seq_100{str_3d}')
    elif hyp.use_3d_pose:
        model_info_parts.append('3d')
    assert 0 < hyp.train_partial <= 1, 'hyp.train_partial should be in (0, 1].'
    if hyp.train_partial != 1:
        model_info_parts.append(f'train_partial_0p{str(hyp.train_partial)[2:]}')
    model_info = '_'.join(model_info_parts)

    # ----------------------------------------------------------------------
    # Per-run experiment folder (tracked via run_tracker).
    # Every run mints experiments/bst_x/shuttleset/run_<timestamp>/ with:
    #   manifest.yaml          (hyperparams + config.classes, git SHA, per-serial metrics)
    #   weights/<save_name>.pt (best checkpoint per serial)
    #   tb/serial_N/           (TB event files per serial)
    #   predictions/<split>_serial_N.npz (per-stroke logits + top-k dump)
    # The runner passes a fixed --run-id across a cell's serials so they share
    # one run dir + log: serial 1 creates the manifest, later serials append via
    # track_serial. Weights are per-serial, so re-running a serial with --run-id
    # finds its .pt and skips training.
    # ----------------------------------------------------------------------
    timestamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    run_id = args.run_id or f'run_{timestamp}'

    # Test output is auto-teed to a timestamped log file so metrics are never
    # lost to a dropped terminal. Training stdout stays on terminal only; TB
    # captures it. One log file per script invocation, all serials inside.
    # Uses the fresh invocation timestamp (not run_id) so resumed re-tests
    # don't overwrite the original run's log file.
    #
    # Anchor experiments/ and test_logs/ to the repo-root experiments/bst_x/shuttleset/
    # so write paths don't depend on cwd. Lets `python -m bst_x_train` land outputs in
    # the canonical run-artefact tree regardless of where it was invoked from.
    script_dir = Path(__file__).resolve().parent
    experiments_dir = script_dir.parent.parent / 'experiments' / 'bst_x' / 'shuttleset'
    log_dir = experiments_dir / 'test_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else log_dir / f'test_{timestamp}.log'

    extra = compute_data_provenance(
        clips_csv_path=Path(hyp.clips_csv),
        collation_id=hyp.collation_id,
        npy_collated_dir=npy_collated_dir,
    )
    # config.classes lands the resolved class list next to the Hyp dump,
    # mirroring BRIC's manifest schema; the FE registry reads it without
    # importing any taxonomy module. track_run treats the dict as a Mapping and
    # stores it verbatim (config.collation_id / config.ablation_id ride along).
    config_payload = dict(hyp._asdict())
    config_payload['classes'] = list(taxonomy.classes)
    run_dir, run_id = track_run(
        config=config_payload, run_id=run_id, log_path=log_path, extra=extra,
        experiments_dir=experiments_dir,
    )
    weight_dir = run_dir / 'weights'

    # Collated dir, resolved the same way the collator wrote it:
    # BST_X_COLLATED_DATA_ROOT (e.g. /scratch/comp320a on bourbaki) when set,
    # else the in-repo preparing_data/ convention for local dev. taxonomy.name
    # is the resolved canonical name, matching the writer's parent dir. Without
    # the env var the reader looks in-repo while the writer wrote to /scratch,
    # so the runner would never find the cells.
    collated_data_root = env_path_or_none('BST_X_COLLATED_DATA_ROOT')
    if collated_data_root is not None:
        collated_root = (
            collated_data_root / f'ShuttleSet_data_{taxonomy.name}' / npy_collated_dir
        )
    else:
        # bst_x_train.py lives at src/bst_x/; preparing_data/ is a sibling, so
        # one .parent walks to src/bst_x/ and then into preparing_data/.
        collated_root = (
            Path(__file__).resolve().parent
            / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
            / npy_collated_dir
        )

    # Per-serial invocation: run only the requested serial. Otherwise loop the
    # manual default of 5. Log open mode flips to append for serial-no > 1 so
    # later per-serial invocations don't clobber the earlier blocks.
    if args.serial_no is not None:
        serial_range = range(args.serial_no, args.serial_no + 1)
        log_open_mode = 'a' if args.serial_no > 1 else 'w'
    else:
        serial_range = range(1, 6)
        log_open_mode = 'w'

    with open(log_path, log_open_mode) as log_f:
        tee = Tee(sys.stdout, log_f)
        _print_taxonomy_block(taxonomy, tee)
        for serial_no in serial_range:
            print(f'Running serial {serial_no} ...')
            task = Task(
                n_joints=17, taxonomy=taxonomy, weight_dir=weight_dir,
            )
            task.prepare_dataloaders(
                root_dir=collated_root,
                pose_style=hyp.pose_style,
                train_partial=hyp.train_partial
            )

            task.get_network_architecture(model_name='BST_X', in_channels=(3 if hyp.use_3d_pose else 2))

            tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
            weight_exists, val_at_best = task.seek_network_weights(
                model_info=model_info, serial_no=serial_no, tb_dir=tb_dir,
            )

            # Per-stroke logits dump (all splits) for the FE / calibration. Runs
            # every serial; non-best are pruned manually after the runner finishes.
            # Returns the per-split dumps so test_metrics/topk_metrics can derive
            # off the same forward pass (B2: one test pass, not three).
            dumps = task.dump_predictions(run_dir=run_dir, serial_no=serial_no, k=5)

            with redirect_stdout(tee):
                print(f'\n=== Serial {serial_no} ({task.model_name}) ===')
                test_metrics = task.test(
                    dump=dumps['test'],
                    show_details=True, show_confusion_matrix=False,
                )
                topk_metrics = task.test_topk_acc(dump=dumps['test'], k=2)

            # Writes the manifest entry, and if aim is installed (it isn't on
            # the HPC train venv, so usually a no-op) mirrors this serial into
            # Aim as a fresh run each call (aim 3.29 can't reopen a stable
            # hash). Re-running a serial adds another Aim run rather than
            # overwriting; the clean, idempotent rebuild is aim_backfill.py --wipe.
            track_serial(
                run_dir=run_dir,
                serial_no=serial_no,
                weights_path=task.weight_path,
                tb_dir=tb_dir,
                metrics={**test_metrics, **topk_metrics},
                extra=({'val_at_best_macro_epoch': val_at_best}
                       if val_at_best else None),
            )

            print('Serial', serial_no, 'done.')

            if not weight_exists:
                time.sleep(3)

    print(f'\nTest log saved to: {log_path}')
    print(f'Run manifest:    {run_dir / "manifest.yaml"}')
