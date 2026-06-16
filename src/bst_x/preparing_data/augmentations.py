"""Training augmentations for BST: a left-right flip and a small position shift.

Replaces the previous joints-only translation, which didn't move the
shuttle or court coordinates with it. Two augmentations, each with its
own per-clip random roll:

- ``CoupledFlip``: mirrors the clip left-to-right across the court
  centreline. Flips player positions, shuttle position, and joint
  coordinates together so they stay aligned. Also swaps the COCO-17
  left/right joint pairs (e.g. left wrist with right wrist) so each
  joint slot keeps a consistent meaning regardless of orientation.
  Bones are recomputed from the flipped joints using the same pair
  table the collation uses, so the bone vectors come out right
  automatically.

- ``ConstrainedJitter``: shifts the clip's player positions and
  shuttle position by a small ``(dx, dy)``, the same shift applied
  to every frame. Joints and bones aren't touched. The shift size
  is bounded per-clip so that players who started inside the court
  don't get pushed out, and clips that didn't have a player crossing
  the centreline don't pick one up from the shift. Frames that were
  zeroed before the shift (because pose detection failed there, or
  the data loader zero-padded a short clip) stay zero. If the shift
  pushes the shuttle outside the visible frame, it gets set to
  ``(0, 0)``, the same off-screen signal the model already handles
  for natural off-screen frames.

Both ops fire per-clip across the batch using torch on whichever
device the inputs live on. Both run only during training.
"""

from __future__ import annotations

import torch
from torch import Tensor

from preparing_data.shuttleset_dataset import get_bone_pairs


# COCO-17 bilateral joint pairs. Slot 0 (nose) has no mirror partner; mirroring
# its x is the only transform it needs. Eyes (1,2), ears (3,4), shoulders (5,6),
# elbows (7,8), wrists (9,10), hips (11,12), knees (13,14), ankles (15,16).
BILATERAL_JOINT_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
)


def _coco_swap_index(n_joints: int, device: torch.device) -> Tensor:
    """Build a lookup index that swaps left/right joint pairs.

    Applying ``joints[..., swap_idx, :]`` moves the data at slot 5 (left
    shoulder) into slot 6 (right shoulder) and vice versa, and similarly
    for every other left/right pair. Slot 0 (nose) has no mirror partner
    and stays in place.
    """
    swap_idx = torch.arange(n_joints, device=device)
    for a, b in BILATERAL_JOINT_PAIRS:
        if a < n_joints and b < n_joints:
            swap_idx[a] = b
            swap_idx[b] = a
    return swap_idx


def recompute_bones_torch(joints: Tensor, pairs: list[tuple[int, int]]) -> Tensor:
    """Torch version of ``shuttleset_dataset.create_bones``.

    Matches the numpy implementation exactly. Each bone is the vector
    from its start joint to its end joint. If either endpoint's x or y
    is zero (meaning the joint detection failed there), the corresponding
    bone component is set to zero too.

    :param joints: tensor of shape ``(..., J, 2)``.
    :param pairs: list of ``(start_idx, end_idx)`` tuples, one per bone,
                  in the same order ``create_bones`` walks them.
    :return: bones tensor of shape ``(..., B, 2)`` where ``B = len(pairs)``.
    """
    start_indices = torch.tensor(
        [p[0] for p in pairs], dtype=torch.long, device=joints.device,
    )
    end_indices = torch.tensor(
        [p[1] for p in pairs], dtype=torch.long, device=joints.device,
    )
    starts = joints.index_select(dim=-2, index=start_indices)
    ends = joints.index_select(dim=-2, index=end_indices)
    both_present = (starts != 0.0) & (ends != 0.0)
    return torch.where(both_present, ends - starts, torch.zeros_like(ends))


class CoupledFlip:
    """Mirrors a clip left-to-right across the court centreline.

    For each clip in the batch, with probability ``p``, this flips the
    player positions, shuttle position, and joint coordinates together,
    then swaps the COCO-17 left/right joint pairs so each joint slot
    keeps a consistent meaning across flipped and unflipped samples.
    Bones are recomputed from the flipped joints using the same pair
    table the collation uses on disk, which gives the right flipped
    bone vectors automatically.

    :param p: probability of flipping each clip. Defaults to 0.5 (the
              standard skeleton-AR rate; at 0.5 the model effectively
              sees both orientations equally).
    :param n_joints: number of joints stored before bones along the
                     pose-feature axis. 17 for COCO-17.
    :param n_bones: number of bones stored after the joints.
                    ``human_pose[..., :-n_bones, :]`` is the joint
                    slice; ``human_pose[..., -n_bones:, :]`` is the
                    bone slice. The caller asserts ``pose_style ==
                    'JnB_bone'`` so this is always positive.
    :param bone_pairs: list of ``(start_joint, end_joint)`` tuples,
                       one per bone, matching the table used at
                       collation time. Defaults to
                       ``get_bone_pairs('coco')``.
    """

    def __init__(
        self,
        p: float = 0.5,
        n_joints: int = 17,
        n_bones: int = 19,
        bone_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        self.p = p
        self.n_joints = n_joints
        self.n_bones = n_bones
        self.bone_pairs = bone_pairs if bone_pairs is not None else get_bone_pairs('coco')
        if len(self.bone_pairs) != n_bones:
            raise ValueError(
                f'bone_pairs length ({len(self.bone_pairs)}) does not match '
                f'n_bones ({n_bones}); the bone-recompute path needs the same '
                f'pair-table the collation used.'
            )

    def __call__(
        self, human_pose: Tensor, pos: Tensor, shuttle: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Flip selected clips across all three streams together.

        :param human_pose: ``(n, t, m, J+B, 2)``. The first ``J`` slots
                           are joints; the last ``B`` slots are bones.
        :param pos: ``(n, t, m, 2)`` player position in court coordinates.
        :param shuttle: ``(n, t, 2)`` shuttle position in camera
                        coordinates.
        :return: the three input tensors with flipped clips updated.
                 The originals are not modified.
        """
        n = human_pose.shape[0]
        device = human_pose.device

        if self.p <= 0.0:
            return human_pose, pos, shuttle

        flip_mask = torch.rand(n, device=device) < self.p
        if not flip_mask.any():
            return human_pose, pos, shuttle

        joints = human_pose[..., :-self.n_bones, :]
        # Build the fully-flipped tensor first, then use torch.where to
        # keep unflipped clips untouched. Faster than indexing into the
        # batch with a boolean mask since the underlying ops are all
        # vectorised.

        # pos: x -> 1 - x in court frame
        pos_flipped = pos.clone()
        pos_flipped[..., 0] = 1.0 - pos_flipped[..., 0]
        pos_mask = flip_mask.view(n, 1, 1, 1).expand_as(pos)
        pos_out = torch.where(pos_mask, pos_flipped, pos)

        # shuttle: x -> 1 - x in camera frame
        shuttle_flipped = shuttle.clone()
        shuttle_flipped[..., 0] = 1.0 - shuttle_flipped[..., 0]
        shuttle_mask = flip_mask.view(n, 1, 1).expand_as(shuttle)
        shuttle_out = torch.where(shuttle_mask, shuttle_flipped, shuttle)

        # joints: x -> -x around each player's bbox centre, then bilateral slot swap
        swap_idx = _coco_swap_index(self.n_joints, device)
        joints_xflipped = joints.clone()
        joints_xflipped[..., 0] = -joints_xflipped[..., 0]
        joints_swapped = joints_xflipped.index_select(dim=-2, index=swap_idx)
        joints_mask = flip_mask.view(n, 1, 1, 1, 1).expand_as(joints)
        joints_out = torch.where(joints_mask, joints_swapped, joints)

        # Bones are recomputed from the (possibly-flipped) joints. The
        # recompute runs on every clip in the batch. For unflipped clips,
        # the result matches the bones already on disk because the
        # formula is a deterministic function of the joint inputs. For
        # flipped clips, the recompute produces the correct flipped bone
        # vectors automatically: swapping the joint slots before
        # subtracting them gives the right vector at each bone slot.
        bones_recomputed = recompute_bones_torch(joints_out, self.bone_pairs)
        human_pose_out = torch.cat([joints_out, bones_recomputed], dim=-2)

        return human_pose_out, pos_out, shuttle_out


class ConstrainedJitter:
    """Shifts the clip's player and shuttle positions by a small ``(dx, dy)``.

    Joints and bones aren't touched. For each clip in the batch, with
    probability ``p_roll``, a single ``(dx, dy)`` is drawn and applied
    identically to every frame of ``pos`` and ``shuttle``. The shift
    size is bounded per-clip so it can't push a player who started
    inside the court out of it, and can't introduce a centreline
    crossing on a clip that didn't already have one. A fixed magnitude
    cap further limits the shift regardless of the per-clip bound.
    Frames that were zeroed in the input (because pose or shuttle
    detection failed there, or the data loader zero-padded a short
    clip) stay zero. If the shift pushes the shuttle outside the
    visible frame, it gets set to ``(0, 0)``, the same off-screen
    signal the model already handles for naturally-missing shuttle
    frames.

    :param p_roll: probability of shifting each clip. The realised rate
                   is slightly lower if some clips have a constraint
                   that leaves no room for any shift; the actual rate
                   is logged in the ``Aug/jitter_effective_rate``
                   TB scalar.
    :param cap_y: maximum shift magnitude on y. Default 0.05; tight
                  because y carries court-zone information that
                  distinguishes shot classes.
    :param cap_x: maximum shift magnitude on x. Default 0.10; looser
                  because x carries less direct class information.
    :param eps: margin matching ``sticky_anchor.generous_margin = 0.15``,
                the band outside which the model treats positions
                as invalid.
    """

    def __init__(
        self,
        p_roll: float = 0.2,
        cap_y: float = 0.05,
        cap_x: float = 0.10,
        eps: float = 0.15,
    ) -> None:
        self.p_roll = p_roll
        self.cap_y = cap_y
        self.cap_x = cap_x
        self.eps = eps

    def __call__(
        self, human_pose: Tensor, pos: Tensor, shuttle: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Apply the per-clip shift to pos and shuttle.

        :return: ``(human_pose, pos_out, shuttle_out, n_effective, n_oob)``.
                 ``n_effective`` is the number of clips that actually
                 received a non-zero shift this batch (rolled in, plus
                 had at least one axis with room to shift). Used for
                 the ``Aug/jitter_effective_rate`` TB scalar.
                 ``n_oob`` is the number of clips where the shift
                 pushed at least one previously-visible shuttle frame
                 off-screen, setting it to ``(0, 0)``. Used for the
                 ``Aug/shuttle_oob_rate`` TB scalar.
        """
        n = pos.shape[0]
        device = pos.device

        if self.p_roll <= 0.0:
            return human_pose, pos, shuttle, 0, 0

        roll_mask = torch.rand(n, device=device) < self.p_roll
        if not roll_mask.any():
            return human_pose, pos, shuttle, 0, 0

        # Compute the per-clip min and max of pos along x and y. These set
        # the bounds for the shift: e.g. if the bottom player's lowest y
        # in this clip is 0.55, we can shift down by at most 0.05 before
        # pushing them across the centreline.
        # pos: (n, t, m=2, 2), m=0 top, m=1 bot. Last dim is xy.
        #
        # Frames where pos is exactly (0, 0) are sentinels meaning "no
        # real position here": either the data loader zero-padded a short
        # clip, or pose detection failed and sticky_anchor zeroed the
        # frame. They have to be excluded from the min/max, otherwise a
        # clip with a real bot range of [0.55, 0.7] plus some padded zeros
        # would read its minimum as 0 and incorrectly drop the centreline
        # constraint.
        #
        # We replace sentinel frames with -inf for the max calculation and
        # +inf for the min calculation, so they never win the reduction.
        # Real positions never land at exactly (0, 0): normalize_position
        # divides camera-frame coords by the court borders, and the result
        # almost never hits 0 by coincidence. The substitution stays local
        # to this block; the original pos tensor is what gets shifted later.
        #
        # At the current cap_y = 0.05, the cap is smaller than the bounds
        # on most clips, so the cap is what limits the shift, not the
        # bounds. The bounds only become the active constraint if cap_y
        # is raised.
        is_sentinel = (pos == 0.0).all(dim=-1)  # (n, t, m)
        pos_for_max = pos.masked_fill(is_sentinel.unsqueeze(-1), float('-inf'))
        pos_for_min = pos.masked_fill(is_sentinel.unsqueeze(-1), float('+inf'))

        y_top_max = pos_for_max[:, :, 0, 1].amax(dim=1)  # (n,)
        y_top_min = pos_for_min[:, :, 0, 1].amin(dim=1)
        y_bot_max = pos_for_max[:, :, 1, 1].amax(dim=1)
        y_bot_min = pos_for_min[:, :, 1, 1].amin(dim=1)

        x_max = pos_for_max[..., 0].amax(dim=(1, 2))     # (n,)
        x_min = pos_for_min[..., 0].amin(dim=(1, 2))

        eps = self.eps
        large = torch.full_like(y_top_max, float('inf'))

        # dy_max: layered upper bound. Top respects centreline (y_top_max <= 0.5)
        # contributes 0.5 - y_top_max; bot respects far-baseline
        # (y_bot_max <= 1+eps) contributes 1+eps - y_bot_max. Where neither
        # respects, no constraint applies and the axis is degenerate.
        top_max_bound = torch.where(y_top_max <= 0.5, 0.5 - y_top_max, large)
        bot_max_bound = torch.where(y_bot_max <= 1.0 + eps, 1.0 + eps - y_bot_max, large)
        dy_max = torch.minimum(top_max_bound, bot_max_bound)

        # dy_min: symmetric layered lower bound.
        top_min_bound = torch.where(y_top_min >= -eps, -eps - y_top_min, -large)
        bot_min_bound = torch.where(y_bot_min >= 0.5, 0.5 - y_bot_min, -large)
        dy_min = torch.maximum(top_min_bound, bot_min_bound)

        # x-axis has no centreline constraint, just the [-eps, 1+eps] band
        # applied jointly across both players.
        dx_max = torch.where(x_max <= 1.0 + eps, 1.0 + eps - x_max, large)
        dx_min = torch.where(x_min >= -eps, -eps - x_min, -large)

        # Replace inf sentinels with 0 to mark fully-degenerate axes.
        dy_max = torch.where(torch.isinf(dy_max), torch.zeros_like(dy_max), dy_max)
        dy_min = torch.where(torch.isinf(dy_min), torch.zeros_like(dy_min), dy_min)
        dx_max = torch.where(torch.isinf(dx_max), torch.zeros_like(dx_max), dx_max)
        dx_min = torch.where(torch.isinf(dx_min), torch.zeros_like(dx_min), dx_min)

        # Intersect the per-clip envelope with the magnitude cap. Cap binds
        # in either direction independently, so a clip with dy_max = 0.30
        # and cap_y = 0.05 ends up with dy_hi = 0.05 (cap binds) and
        # dy_lo = -0.05 (cap binds on the other side too if dy_min permits).
        cap_y_t = torch.full_like(dy_max, self.cap_y)
        cap_x_t = torch.full_like(dx_max, self.cap_x)
        dy_hi = torch.minimum(dy_max, cap_y_t)
        dy_lo = torch.maximum(dy_min, -cap_y_t)
        dx_hi = torch.minimum(dx_max, cap_x_t)
        dx_lo = torch.maximum(dx_min, -cap_x_t)

        # Per-axis degeneracy: clamp the sample range to a single point at 0
        # when the layered constraints leave no room. Floats: dy_hi <= dy_lo.
        dy_degenerate = dy_hi <= dy_lo
        dx_degenerate = dx_hi <= dx_lo
        dy_hi = torch.where(dy_degenerate, torch.zeros_like(dy_hi), dy_hi)
        dy_lo = torch.where(dy_degenerate, torch.zeros_like(dy_lo), dy_lo)
        dx_hi = torch.where(dx_degenerate, torch.zeros_like(dx_hi), dx_hi)
        dx_lo = torch.where(dx_degenerate, torch.zeros_like(dx_lo), dx_lo)

        # Sample uniform in the per-axis envelope.
        u_y = torch.rand(n, device=device)
        u_x = torch.rand(n, device=device)
        dy = dy_lo + (dy_hi - dy_lo) * u_y
        dx = dx_lo + (dx_hi - dx_lo) * u_x

        # Suppress shifts on clips where the roll missed.
        dx = torch.where(roll_mask, dx, torch.zeros_like(dx))
        dy = torch.where(roll_mask, dy, torch.zeros_like(dy))

        # Build pre-shift zero masks for stream-aware sentinel preservation.
        # pos_zero: any frame whose xy entries are both zero. Shape (n, t, m).
        # shuttle_zero: same on (n, t).
        pos_zero = (pos == 0.0).all(dim=-1)
        shuttle_zero = (shuttle == 0.0).all(dim=-1)

        # Apply shift. Broadcast dx/dy across (t, m) for pos and (t,) for shuttle.
        shift_pos = torch.stack([dx, dy], dim=-1)        # (n, 2)
        shift_pos = shift_pos.view(n, 1, 1, 2)
        pos_shifted = pos + shift_pos

        shift_shuttle = torch.stack([dx, dy], dim=-1).view(n, 1, 2)
        shuttle_shifted = shuttle + shift_shuttle

        # Restore pre-existing zeros.
        pos_shifted = torch.where(
            pos_zero.unsqueeze(-1).expand_as(pos_shifted),
            torch.zeros_like(pos_shifted),
            pos_shifted,
        )
        shuttle_shifted = torch.where(
            shuttle_zero.unsqueeze(-1).expand_as(shuttle_shifted),
            torch.zeros_like(shuttle_shifted),
            shuttle_shifted,
        )

        # Shuttle out-of-bounds post-shift -> zero, mirroring TrackNet's
        # off-screen sentinel so the model recognises induced off-screen
        # the same way it handles natural off-screen. Count only OOB frames
        # that weren't already a pre-shift sentinel zero, so the OOB rate
        # reflects aug-induced off-screen and not the natural baseline.
        shuttle_oob = (
            (shuttle_shifted < 0.0).any(dim=-1)
            | (shuttle_shifted > 1.0).any(dim=-1)
        )
        oob_aug_induced = shuttle_oob & ~shuttle_zero  # (n, t)
        shuttle_shifted = torch.where(
            shuttle_oob.unsqueeze(-1).expand_as(shuttle_shifted),
            torch.zeros_like(shuttle_shifted),
            shuttle_shifted,
        )

        # Effective-fired count: clip rolled yes AND at least one axis non-degenerate.
        non_degenerate = ~(dy_degenerate & dx_degenerate)
        effective = roll_mask & non_degenerate
        n_effective = int(effective.sum().item())

        # OOB rate metric: clips that fired an effective shift AND had at
        # least one previously-real shuttle frame land off-screen due to
        # the shift. Diagnostic for the trade-off the doc flags around
        # cap_x and edge-of-frame shuttle classes (cross_court_net_shot,
        # rush trajectories). A high rate suggests cap_x is wide enough
        # to be replacing a meaningful fraction of real shuttle observations
        # with the (0, 0) sentinel during training.
        clip_had_oob = oob_aug_induced.any(dim=-1)  # (n,)
        n_oob = int((effective & clip_had_oob).sum().item())

        return human_pose, pos_shifted, shuttle_shifted, n_effective, n_oob
