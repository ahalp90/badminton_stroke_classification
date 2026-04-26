# Consolidated BST training script for ShuttleSet
# Replaces: bst_main.py, bst_main_summary_writer.py, bst_backbone_main.py
#
# PyTorch training loop overview (differs significantly from TF/Keras):
#   TF:      model.compile(optimizer, loss) -> model.fit(data)  (one line trains everything)
#   PyTorch: you write the loop yourself — iterate batches, compute loss, call backward(), step()
#   This is more verbose but gives full control over every training step.

import torch
from torch import Tensor, nn, optim  # nn = layers/models, optim = optimizers (like tf.keras.optimizers)
import torch.nn.functional as F      # F = stateless functions (one_hot, softmax, etc.)
from torch.utils.data import DataLoader  # like tf.data.Dataset — batches, shuffles, prefetches
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging (same viewer as TF)
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup  # from HuggingFace, not a custom module

import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
from contextlib import redirect_stdout
import math
import time
from datetime import datetime, timedelta

import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              RandomTranslation_batch, Dataset_npy, \
                                              pad_class_labels, get_bone_pairs, \
                                              POSE_BONE_MULTIPLIER
from model.bst import BST_0, BST_PPF, BST_CG, BST_AP, BST_CG_AP
from result_utils import show_f1_results, plot_confusion_matrix
from pipeline.config import TAXONOMIES, Taxonomy
from run_tracker import track_run, track_serial


# ==========================================================================
# Hyperparameters — edit these to change experiment configuration.
# namedtuple is just an immutable struct: hyp.lr, hyp.batch_size, etc.
# ==========================================================================
Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'warm_up_step',
    'taxonomy', 'seq_len', 'early_stop_n_epochs',
    'pose_style', 'use_3d_pose', 'train_partial',
    'use_aux_schedule', 'aux_fade_end_epoch',
])
# --------------------------------------------------------------------------
# LR-SCHEDULE RETUNE (2026-04-17)
# --------------------------------------------------------------------------
# The original BST recipe used n_epochs=1600, warm_up_step=400, patience=300,
# with a cosine scheduler at num_cycles=0.25 (see line below). In practice
# on ShuttleSet + merged_25, the TB log from run Apr17_13-04-35 showed:
#     - best F1_macro 0.8311 at epoch 41
#     - best F1_min   0.6250 at epoch 53
#     - best Val_loss 1.0032 at epoch 27
#     - training stopped at epoch 341 (patience fired)
# Convergence is well before epoch 60. At n_epochs=1600, num_cycles=0.25,
# the LR is still ~99.98% of peak at epoch 41, so cosine decay never
# actually bites during the useful window — the model just drifts under
# near-peak LR for hundreds of epochs and overfits.
# Retuned values (active block below) match the schedule to observed
# convergence: ~4-epoch warmup, short cosine that reaches 0 at the end,
# and patience tightened to 40 epochs to stay meaningful at the new
# length. n_epochs has since been shortened again (120 -> 80) to pair
# with the AUX-SCHEDULE block; see that block for rationale. The old
# values are preserved (commented) for easy revert.
# --------------------------------------------------------------------------
# hyp = Hyp(
#     n_epochs=1600,            # max epochs (will early-stop before this)
#     early_stop_n_epochs=300,  # stop if no F1 improvement for this many epochs
#     batch_size=128,
#     lr=5e-4,                  # initial learning rate (cosine-annealed during training)
#     warm_up_step=400,         # LR warmup steps before cosine decay begins
#     taxonomy='merged_25',      # key in TAXONOMIES: 'une_merge_v1', 'merged_25', 'raw_35', …
#     seq_len=100,              # frames per sample (must match data preprocessing)
#     pose_style='JnB_bone',   # 'J_only'=joints, 'JnB_bone'=joints+bones, 'Jn2B'=joints+2xbones
#     use_3d_pose=False,        # True for xyz keypoints, False for xy only
#     train_partial=1.0         # fraction of training set to use (1.0 = all)
# )
# --------------------------------------------------------------------------
# AUX-SCHEDULE (CG/AP warm-start-to-fade)
# --------------------------------------------------------------------------
# Hypothesis: Clean Gate and Aim Player are heuristics. They accelerate
# early convergence while the LR is hot, but they also constrain the
# representation the transformer backbone can learn. Annealing them out
# during training should let the backbone find a richer representation.
#
# Implementation: a scalar aux_factor in [0, 1] scales CG's dirt subtraction
# and blends AP's alpha multipliers toward pass-through (1.0). The schedule
# is cosine from epoch 1 (factor=1.0) to aux_fade_end_epoch (factor=0.0),
# then pinned at 0 for the rest of training (pure-backbone phase).
#
# Knobs (both below in the hyp block):
#   use_aux_schedule  -- False pins factor at 1.0 all run, reproducing the
#                        unscheduled BST_CG_AP baseline exactly.
#   aux_fade_end_epoch -- epoch at which the factor first reaches 0. Set
#                         well below n_epochs to guarantee a long pure-
#                         backbone tail. At 15 with n_epochs=80: warm-start
#                         is ~20% of training, then 65 epochs of pure-
#                         backbone at cooling LR — framed as warm-start
#                         then finetune. A prior 60/120 schedule put peaks
#                         inside the fade window (factor still 0.6-0.74 at
#                         pick time for 2 of 3 seeds), which diluted the
#                         signal into noise; the only seed that picked deep
#                         in the fade (factor ~0.1) showed the hoped-for
#                         min-F1 lift.
#
# CG and AP currently share one factor (set_schedule_factors passes the
# same value to both). If a later ablation wants independent fade timing,
# split into cg_fade_end_epoch and ap_fade_end_epoch and pass two values
# through aux_schedule_factor into set_schedule_factors.
#
# Test-time behaviour: the best checkpoint captures cg_factor and ap_factor
# as buffers in state_dict. task.test() runs forward with those restored
# values, so the final test metric reflects whichever factor was active at
# the best-F1 epoch. Nothing forces them to 0 or 1 at test time.
# --------------------------------------------------------------------------
hyp = Hyp(
    n_epochs=80,              # compressed warm-start-then-finetune schedule
    early_stop_n_epochs=40,   # patience ~half the run
    batch_size=128,
    lr=5e-4,                  # initial learning rate (cosine-annealed during training)
    warm_up_step=100,         # was 400 — ~4 epochs of warmup instead of ~17
    taxonomy='merged_25',      # key in TAXONOMIES: 'une_merge_v1', 'merged_25', 'raw_35', …
    seq_len=100,              # frames per sample (must match data preprocessing)
    pose_style='JnB_bone',   # 'J_only'=joints, 'JnB_bone'=joints+bones, 'Jn2B'=joints+2xbones
    use_3d_pose=False,        # True for xyz keypoints, False for xy only
    train_partial=1.0,        # fraction of training set to use (1.0 = all)
    use_aux_schedule=True,    # Run B: CG/AP off from start, 80-ep schedule. Null test vs 18_15 and Run A.
    aux_fade_end_epoch=0,     # epoch at which aux_factor first reaches 0; stays 0 after (ignored when use_aux_schedule=False)
)


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
    if fade_end_epoch <= 1 or epoch >= fade_end_epoch:
        return 0.0 if epoch >= fade_end_epoch else 1.0
    progress = (epoch - 1) / (fade_end_epoch - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    n_bones: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,  # learning rate scheduler
    device
):
    model.train()  # enable dropout + batchnorm training mode (TF: training=True)
    total_loss = 0.0

    for (human_pose, pos, shuttle), video_len, labels in loader:
        # .to(device) = move tensors to GPU/CPU. TF does this automatically;
        # PyTorch requires explicit placement for every tensor.
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)

        # Apply random translation augmentation to joints only (not bones,
        # because bone vectors are relative and translation-invariant)
        if n_bones == 0:
            human_pose = random_shift_fn(human_pose)
        else:
            joints = human_pose[:, :, :, :-n_bones, :].contiguous()
            bones = human_pose[:, :, :, -n_bones:, :]

            joints = random_shift_fn(joints)
            human_pose = torch.cat([joints, bones], dim=-2)

        # Flatten last two dims (joints/bones, xy) into one feature dim for the model
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        loss: Tensor = loss_fn(logits, labels)

        # PyTorch manual gradient step (TF does this inside model.fit()):
        optimizer.zero_grad()  # clear gradients from previous batch
        loss.backward()        # backpropagation: compute gradients (like tape.gradient())
        optimizer.step()       # apply gradients to weights (like optimizer.apply_gradients())
        scheduler.step()       # update learning rate according to cosine schedule

        total_loss += loss.item()  # .item() extracts Python float from single-element tensor

    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()  # disables gradient computation — saves memory during eval
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device,
    n_classes: int,
):
    model.eval()  # disable dropout + set batchnorm to eval mode (TF: training=False)
    total_loss = 0.0
    # Accumulate confusion matrix components across batches for per-class F1
    cum_tp = torch.zeros(n_classes)
    cum_tn = torch.zeros(n_classes)
    cum_fp = torch.zeros(n_classes)
    cum_fn = torch.zeros(n_classes)

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

        # Manual per-class TP/FP/FN/TN computation via one-hot encoding
        pred = F.one_hot(torch.argmax(logits, dim=1), n_classes).bool()
        labels_onehot = F.one_hot(labels, n_classes).bool()

        tp = torch.sum(pred & labels_onehot, dim=0)
        tn = torch.sum(~pred & ~labels_onehot, dim=0)

        fp = torch.sum(pred & ~labels_onehot, dim=0)
        fn = torch.sum(~pred & labels_onehot, dim=0)

        # .cpu() moves results back from GPU for accumulation
        cum_tp += tp.cpu()
        cum_tn += tn.cpu()
        cum_fp += fp.cpu()
        cum_fn += fn.cpu()

    val_loss = total_loss / len(loader)

    # Per-class F1, then macro average (mean across classes)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0  # classes with no predictions get NaN -> 0

    f1_score_avg = f1_score.mean()  # macro F1: unweighted mean across all classes
    f1_score_min = f1_score.min()   # worst-performing class
    return val_loss, f1_score_avg, f1_score_min


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        pred = torch.argmax(logits, dim=1).cpu()

        pred_ls.append(pred)
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


@torch.no_grad()
def test_topk(
    model: nn.Module,
    loader,
    device,
    k=2
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        _, pred = torch.topk(logits, k=k, dim=1)

        pred_ls.append(pred.cpu())
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


# ==========================================================================
# Training loop with TensorBoard logging and early stopping
# ==========================================================================

def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    n_bones,
    n_classes: int,
    tb_dir: Path | None = None,
):
    # tb_dir lands the event files under experiments/<run_id>/tb/serial_N/ so
    # TB folders pair with the run they came from. Default SummaryWriter() writes
    # to ./runs/<host_time>/, which is what older runs used.
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else SummaryWriter()
    random_shift_fn = RandomTranslation_batch()  # data augmentation: small xy shifts

    # label_smoothing=0.1: softens targets from [0,1] to [0.004, 0.904] to reduce overconfidence
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW = Adam with decoupled weight decay (standard for transformers)
    # model.parameters() returns all learnable weights (TF equivalent: model.trainable_variables)
    optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)
    # Cosine schedule: LR ramps up during warmup, then decays following a cosine curve.
    # See the 2026-04-17 retune block above hyp for the num_cycles 0.25 -> 0.5 reasoning.
    # HF formula: lr_factor = 0.5 * (1 + cos(pi * 2 * num_cycles * progress))
    #   num_cycles=0.25 -> LR ends at 50% of peak (partial dropoff)
    #   num_cycles=0.5  -> LR ends at 0 (full standard cosine descent)
    #   num_cycles=1.0  -> LR ends back at peak (don't use — full wave)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=hyp.warm_up_step,
    #     num_training_steps=(hyp.n_epochs * len(train_loader)),
    #     num_cycles=0.25  # fraction of cosine cycle (0.25 = quarter-cosine decay)
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * len(train_loader)),  # total batches across all epochs
        num_cycles=0.5  # was 0.25 — full cosine descent to 0 at end of schedule
    )

    # Track top-2 of each metric (for HParams summary + verifying early-stop vs crash)
    best_macro = second_macro = 0.0
    best_macro_epoch = second_macro_epoch = 0
    best_min = second_min = 0.0
    best_min_epoch = second_min_epoch = 0
    best_val_loss, best_val_loss_epoch = float('inf'), 0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        # Auxiliary module schedule: cosine fade of CG/AP from 1.0 -> 0.0 across the run.
        # When disabled, factor stays at 1.0 -> identical to unscheduled BST_CG_AP.
        if hyp.use_aux_schedule:
            aux_factor = aux_schedule_factor(epoch, hyp.aux_fade_end_epoch)
        else:
            aux_factor = 1.0
        model.set_schedule_factors(cg_factor=aux_factor, ap_factor=aux_factor)

        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            n_bones=n_bones,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        val_loss, f1_score_avg, f1_score_min = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            n_classes=n_classes,
        )
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '
              f'val_loss={val_loss:.3f}, macro_f1={f1_score_avg:.3f}, min_f1={f1_score_min:.3f} '
              f'- {t1 - t0:.2f} s')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('F1/Val_macro', f1_score_avg, epoch)
        writer.add_scalar('F1/Val_min', f1_score_min, epoch)
        writer.add_scalar('Schedule/aux_factor', aux_factor, epoch)

        curr_macro, curr_min = f1_score_avg.item(), f1_score_min.item()

        # Early stop + snapshot best weights (piggybacks on new-best detection)
        early_stop_count += 1
        if curr_macro > best_macro:
            second_macro, second_macro_epoch = best_macro, best_macro_epoch
            best_macro, best_macro_epoch = curr_macro, epoch
            # state_dict() = snapshot of all model weights as a dict (like model.get_weights() in TF)
            # deepcopy because state_dict returns references that would change as training continues
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {curr_macro:.3f}')
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

    # HParams summary: one row per run, sortable in TB's HParams tab.
    # stopped_epoch - best_macro_epoch == early_stop_n_epochs confirms clean early-stop.
    writer.add_hparams(
        hparam_dict=hyp._asdict(),
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

    # Save best checkpoint and restore it into the model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))  # like model.save_weights() in TF
    model.load_state_dict(best_state)       # like model.load_weights() in TF
    return model


# ==========================================================================
# Task: orchestrates data loading, model creation, training, and evaluation
# ==========================================================================

# BST variant name -> pre-configured constructor (partials defined in bst.py)
MODELS = {
    'BST_0':     BST_0,
    'BST':       BST_PPF,
    'BST_CG':    BST_CG,
    'BST_AP':    BST_AP,
    'BST_CG_AP': BST_CG_AP,
}


class Tee:
    """Duplicate writes across multiple streams (e.g. terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


class Task:
    def __init__(self, n_joints=17, taxonomy: Taxonomy = None,
                 weight_dir: Path = Path('weight')) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.taxonomy = taxonomy or TAXONOMIES[hyp.taxonomy]
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

    def get_network_architecture(self, model_name='BST_CG_AP', in_channels=2):
        """Create model with the right input dimensions and optional modules.
        in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz)."""
        ModelClass = MODELS[model_name]  # pre-configured partial from bst.py
        n_bones = len(get_bone_pairs())  # 19 bone vectors for COCO 17-joint skeleton
        extra = POSE_BONE_MULTIPLIER[self.pose_style]

        self.net = ModelClass(
            in_dim=(self.n_joints + n_bones * extra) * in_channels,
            n_class=self.taxonomy.n_classes,
            seq_len=hyp.seq_len,
            depth_tem=2,       # 2 layers in temporal transformer
            depth_inter=1,     # 1 layer in interactional transformer
        ).to(self.device)  # move entire model to GPU/CPU

        self.model_name = model_name
        self.n_bones = n_bones

    def seek_network_weights(self, model_info='', serial_no=1, tb_dir: Path | None = None):
        """Load existing weights if found, otherwise train from scratch.
        Weight filenames encode the full experiment config, e.g.:
        'bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'
        """
        model_info = f'_{model_info}' if model_info != '' else ''
        taxonomy_info = f'_{self.taxonomy.name}'
        serial_str = f'_{serial_no}' if serial_no != 1 else ''

        model_postfix = '_' + self.pose_style \
            + model_info + taxonomy_info + serial_str

        # Weight filename: 'BST_CG_AP' -> 'bst_CG_AP', 'BST_0' -> 'bst_0', 'BST' -> 'bst'
        if '_' in self.model_name:
            first, rest = self.model_name.split('_', 1)
            save_name = first.lower() + '_' + rest
        else:
            save_name = self.model_name.lower()
        save_name += model_postfix

        self.model_name += model_postfix

        weight_path = self.weight_dir / f'{save_name}.pt'
        self.weight_path = weight_path
        if weight_path.exists():
            self.net.load_state_dict(
                torch.load(str(weight_path), map_location=self.device, weights_only=True)
            )
            return True  # weight already existed
        else:
            train_t0 = time.time()
            self.net = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path,
                n_bones=len(get_bone_pairs()) if self.pose_style != 'J_only' else 0,
                n_classes=self.taxonomy.n_classes,
                tb_dir=tb_dir,
            )
            t = timedelta(seconds=int(time.time() - train_t0))
            print(f'Total training time: {t}')
            return False  # newly trained

    def test(self, show_details=False, show_confusion_matrix=False) -> dict:
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(
            pred, gt, num_classes=self.taxonomy.n_classes, average=None
        )
        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each,
            class_ls=pad_class_labels(self.taxonomy.class_list()),
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

        return {
            'macro_f1':    float(f1_score_each.mean().item()),
            'min_f1':      float(f1_score_each.min().item()),
            'accuracy':    float(acc),
            'num_strokes': int(len(pred)),
        }

    def test_topk_acc(self, k=2) -> dict:
        assert k > 1, 'k should be > 1'
        pred, gt = test_topk(self.net, self.test_loader, self.device, k=k)
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')
        return {f'top{k}_accuracy': float(acc)}

    def compare_pred_gt_on_specific_type(self, dir_path: Path):
        infer_ds = Dataset_npy(
            root_dir=dir_path,
            set_name='test_specific',
            pose_style=self.pose_style,
            seq_len=hyp.seq_len,
            taxonomy=self.taxonomy,
        )
        infer_loader = DataLoader(
            dataset=infer_ds,
            batch_size=hyp.batch_size,
        )

        pred, gt = test(self.net, infer_loader, self.device)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        not_match = pred != gt
        class_ls = self.taxonomy.class_list()
        with pd.option_context('display.max_rows', None):
            df = pd.DataFrame(
                data={
                    'Ball Round': [Path(e).stem for e in infer_ds.data_branches],
                    'Pred': [class_ls[e] if b else '-' for e, b in zip(pred, not_match)],
                    'GT': [class_ls[e] if b else '-' for e, b in zip(gt, not_match)]
                }
            )
            print(df)


# ==========================================================================
# Main: train and test on ShuttleSet
# ==========================================================================

if __name__ == '__main__':
    additional_model_info = ''
    taxonomy = TAXONOMIES[hyp.taxonomy]

    str_3d = '_3d' if hyp.use_3d_pose else ''
    match hyp.seq_len:
        case 30:
            npy_collated_dir = f'dataset{str_3d}_npy_collated'
            model_info = '3d' if hyp.use_3d_pose else ''
        case 100:
            npy_collated_dir = f'dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100'
            model_info = f'between_2_hits_with_max_limits_seq_100{str_3d}'
        case _:
            raise NotImplementedError

    assert 0 < hyp.train_partial <= 1, 'hyp.train_partial should be in (0, 1].'
    if hyp.train_partial != 1:
        tmp_str = f'train_partial_0p{str(hyp.train_partial)[2:]}'
        if model_info != '':
            model_info += '_' + tmp_str
        else:
            model_info += tmp_str

    if additional_model_info != '':
        if model_info != '':
            model_info += f'_{additional_model_info}'
        else:
            model_info = additional_model_info

    # ----------------------------------------------------------------------
    # Per-run experiment folder (tracked via run_tracker).
    # Every invocation mints a fresh experiments/run_<timestamp>/ with:
    #   manifest.yaml          (hyperparams, git SHA, per-serial metrics)
    #   weights/<save_name>.pt (best checkpoint per serial)
    #   tb/serial_N/           (TB event files per serial)
    # Old flat weight/<name>.pt layout caused silent training-skips after
    # hyperparam changes, so the cache is scoped to the run folder.
    #
    # To re-test an existing run without retraining, set resume_from to its
    # folder name (e.g. 'run_20260417_091933'). The cache then finds saved
    # weights under experiments/<resume_from>/weights/ and skips training.
    # Only tracker-era runs resume cleanly; legacy weight/run_*/ folders
    # need their .pt files copied into experiments/<id>/weights/ first.
    # Leave as None for normal fresh-train behaviour.
    # ----------------------------------------------------------------------
    resume_from: str | None = None

    timestamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    run_id = resume_from or f'run_{timestamp}'

    # Test output is auto-teed to a timestamped log file so metrics are never
    # lost to a dropped terminal. Training stdout stays on terminal only; TB
    # captures it. One log file per script invocation, all serials inside.
    # Uses the fresh invocation timestamp (not run_id) so resumed re-tests
    # don't overwrite the original run's log file.
    log_dir = Path('test_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'test_{timestamp}.log'

    run_dir, run_id = track_run(config=hyp, run_id=run_id, log_path=log_path)
    weight_dir = run_dir / 'weights'

    with open(log_path, 'w') as log_f:
        tee = Tee(sys.stdout, log_f)
        for serial_no in range(1, 6):
            print(f'Running serial {serial_no} ...')
            task = Task(n_joints=17, taxonomy=taxonomy, weight_dir=weight_dir)
            task.prepare_dataloaders(
                root_dir=Path(__file__).resolve().parent.parent
                             / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
                             / npy_collated_dir,
                pose_style=hyp.pose_style,
                train_partial=hyp.train_partial
            )
            task.get_network_architecture(model_name='BST_CG_AP', in_channels=(3 if hyp.use_3d_pose else 2))

            tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
            weight_exists = task.seek_network_weights(
                model_info=model_info, serial_no=serial_no, tb_dir=tb_dir,
            )

            with redirect_stdout(tee):
                print(f'\n=== Serial {serial_no} ({task.model_name}) ===')
                test_metrics = task.test(show_details=False, show_confusion_matrix=False)
                topk_metrics = task.test_topk_acc(k=2)

            track_serial(
                run_dir=run_dir,
                serial_no=serial_no,
                weights_path=task.weight_path,
                tb_dir=tb_dir,
                metrics={**test_metrics, **topk_metrics},
            )

            print('Serial', serial_no, 'done.')

            if not weight_exists:
                time.sleep(3)

    print(f'\nTest log saved to: {log_path}')
    print(f'Run manifest:    {run_dir / "manifest.yaml"}')
