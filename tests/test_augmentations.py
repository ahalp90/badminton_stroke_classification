"""Tests for the locked Task-2 augmentation set: CoupledFlip + ConstrainedJitter.

Covers ten sections matching the implementation surface in
``scratch/architecture_notes/augmentation_framework.md``:

1. ``BILATERAL_JOINT_PAIRS`` integrity (COCO-17 bilateral coverage,
   no duplicates, nose excluded).
2. ``recompute_bones_torch`` parity with ``create_bones`` on un-augmented
   inputs and zero-suppression behaviour.
3. ``CoupledFlip`` per-stream coord transforms (pos 1-x, shuttle 1-x,
   joints -x).
4. ``CoupledFlip`` bilateral joint slot swap correctness.
5. ``CoupledFlip`` bone recompute carries both the X-component sign flip
   and the bilateral bone-slot swap automatically.
6. ``CoupledFlip`` per-clip independence (subset of batch flipped).
7. ``CoupledFlip`` p=0 / p=1 deterministic identity / full-flip.
8. ``ConstrainedJitter`` layered conditional bounds: per-side respect
   gates each constraint independently.
9. ``ConstrainedJitter`` magnitude cap, degenerate axes (cases 1/2/3),
   per-clip rolls, zero-frame preservation, shuttle off-screen mirror.
10. ``ConstrainedJitter`` joints/bones untouched and per-clip
    independence in batched draws.

CPU-only. Run from repo root::

    pytest tests/test_augmentations.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from preparing_data.augmentations import (
    BILATERAL_JOINT_PAIRS,
    CoupledFlip,
    ConstrainedJitter,
    recompute_bones_torch,
)
from preparing_data.shuttleset_dataset import create_bones, get_bone_pairs


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

COCO_BILATERAL_INDICES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
}


def _random_clip_tensors(
    n: int = 4, t: int = 8, m: int = 2, j: int = 17,
    bone_pairs: list[tuple[int, int]] | None = None,
    seed: int = 0,
):
    """Build a random batched (human_pose, pos, shuttle) triple plus the
    joints / bones split, all reproducible from `seed`. Joints sit in
    bbox-relative [-0.5, 0.5]² (centre-aligned), pos in court [0, 1]²,
    shuttle in camera [0, 1]². Bones come from the same numpy
    ``create_bones`` the collation pipeline uses.
    """
    pairs = bone_pairs if bone_pairs is not None else get_bone_pairs('coco')
    g = torch.Generator().manual_seed(seed)
    joints = (torch.rand(n, t, m, j, 2, generator=g) - 0.5) * 0.8
    pos = torch.rand(n, t, m, 2, generator=g) * 0.6 + 0.2
    shuttle = torch.rand(n, t, 2, generator=g) * 0.6 + 0.2

    # Build bones via the canonical numpy path so we're testing parity
    # against the same source-of-truth the collation uses on disk.
    joints_np = joints.numpy()
    bones_each = []
    for i in range(n):
        bones_each.append(create_bones(joints_np[i], pairs))
    bones = torch.from_numpy(np.stack(bones_each)).float()

    human_pose = torch.cat([joints, bones], dim=-2)
    return human_pose, pos, shuttle, joints, bones


# ---------------------------------------------------------------------------
# Section 1: BILATERAL_JOINT_PAIRS integrity
# ---------------------------------------------------------------------------

def test_bilateral_pairs_cover_all_coco_bilaterals():
    flat = [idx for pair in BILATERAL_JOINT_PAIRS for idx in pair]
    assert set(flat) == COCO_BILATERAL_INDICES


def test_bilateral_pairs_no_duplicates():
    flat = [idx for pair in BILATERAL_JOINT_PAIRS for idx in pair]
    assert len(flat) == len(set(flat))


def test_bilateral_pairs_excludes_nose():
    flat = [idx for pair in BILATERAL_JOINT_PAIRS for idx in pair]
    assert 0 not in flat


# ---------------------------------------------------------------------------
# Section 2: recompute_bones_torch parity with create_bones
# ---------------------------------------------------------------------------

def test_bone_recompute_matches_numpy_create_bones():
    pairs = get_bone_pairs('coco')
    g = torch.Generator().manual_seed(42)
    joints = (torch.rand(2, 5, 2, 17, 2, generator=g) - 0.5) * 0.8

    bones_torch = recompute_bones_torch(joints, pairs)
    bones_np = np.stack([create_bones(joints[i].numpy(), pairs) for i in range(2)])

    assert torch.allclose(bones_torch, torch.from_numpy(bones_np), atol=1e-6)


def test_bone_recompute_zero_suppresses_when_endpoint_missing():
    """Per ``create_bones`` semantics, a bone xy gets zeroed when the
    matching xy of either endpoint is zero. In practice MMPose-zeroed
    joints have both x and y at 0 simultaneously, but the per-component
    suppression matches the existing collation-time formula exactly.
    """
    pairs = [(0, 1)]
    joints = torch.tensor([
        [[[1.0, 2.0],     # joint 0
          [3.0, 4.0]]],   # joint 1, both present
    ])  # (1, 1, 2, 2): (clip, frame, J=2, xy=2)
    bones = recompute_bones_torch(joints, pairs)
    # bone 0 -> 1: ends - starts = (3-1, 4-2) = (2, 2)
    assert torch.allclose(bones, torch.tensor([[[[2.0, 2.0]]]]))

    # Zero joint 0's full xy entirely (the realistic MMPose-failure case
    # where both x and y land at 0 together) -> bone xy zeroed.
    joints[0, 0, 0, :] = 0.0
    bones = recompute_bones_torch(joints, pairs)
    assert torch.allclose(bones, torch.zeros(1, 1, 1, 2))


def test_bone_recompute_zero_suppression_per_component():
    """Single-component zero suppresses just that bone component."""
    pairs = [(0, 1)]
    # joint 0 has x=0 but y=2; joint 1 fully present
    joints = torch.tensor([[[[
        [0.0, 2.0],
        [3.0, 4.0],
    ]]]])
    bones = recompute_bones_torch(joints, pairs)
    # bone.x suppressed (start.x is 0); bone.y survives (4-2=2)
    assert bones[0, 0, 0, 0, 0].item() == 0.0
    assert bones[0, 0, 0, 0, 1].item() == 2.0


# ---------------------------------------------------------------------------
# Section 3: CoupledFlip per-stream coord transforms
# ---------------------------------------------------------------------------

def test_flip_pos_x_mirrored_in_court_frame():
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(seed=1)
    pos_orig = pos.clone()
    _, pos_out, _ = flip(human_pose, pos, shuttle)
    assert torch.allclose(pos_out[..., 0], 1.0 - pos_orig[..., 0], atol=1e-6)
    assert torch.allclose(pos_out[..., 1], pos_orig[..., 1], atol=1e-6)


def test_flip_shuttle_x_mirrored_in_camera_frame():
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(seed=2)
    shuttle_orig = shuttle.clone()
    _, _, shuttle_out = flip(human_pose, pos, shuttle)
    assert torch.allclose(shuttle_out[..., 0], 1.0 - shuttle_orig[..., 0], atol=1e-6)
    assert torch.allclose(shuttle_out[..., 1], shuttle_orig[..., 1], atol=1e-6)


def test_flip_joints_x_negated_around_bbox_centre():
    """Joints are bbox-centre-relative, so the flip is x -> -x (mirror
    around 0 in joint coords) rather than x -> 1-x.
    """
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, joints_orig, _ = _random_clip_tensors(seed=3)
    pose_out, _, _ = flip(human_pose, pos, shuttle)
    joints_out = pose_out[..., :17, :]

    # Build the expected post-flip joints: x negated then bilateral slot swap.
    # Apply the same swap_idx logic as the implementation.
    swap_idx = torch.arange(17)
    for a, b in BILATERAL_JOINT_PAIRS:
        swap_idx[a] = b
        swap_idx[b] = a
    joints_xflipped = joints_orig.clone()
    joints_xflipped[..., 0] = -joints_xflipped[..., 0]
    joints_expected = joints_xflipped.index_select(dim=-2, index=swap_idx)

    assert torch.allclose(joints_out, joints_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 4: bilateral joint slot swap
# ---------------------------------------------------------------------------

def test_bilateral_swap_moves_left_wrist_to_right_wrist_slot():
    """Slot 9 (anatomical L-wrist) and slot 10 (anatomical R-wrist) swap
    after the bilateral mirror, so post-flip slot 9 carries the mirrored
    R-wrist coords and slot 10 carries the mirrored L-wrist coords.
    """
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    g = torch.Generator().manual_seed(7)
    joints = (torch.rand(1, 2, 2, 17, 2, generator=g) - 0.5) * 0.8
    bones = recompute_bones_torch(joints, get_bone_pairs('coco'))
    human_pose = torch.cat([joints, bones], dim=-2)
    pos = torch.zeros(1, 2, 2, 2)
    shuttle = torch.zeros(1, 2, 2)

    pose_out, _, _ = flip(human_pose, pos, shuttle)
    joints_out = pose_out[..., :17, :]

    l_wrist_orig = joints[0, 0, 0, 9].clone()
    r_wrist_orig = joints[0, 0, 0, 10].clone()

    # Post-flip slot 9 has the mirrored R-wrist coords.
    assert torch.allclose(joints_out[0, 0, 0, 9, 0], -r_wrist_orig[0], atol=1e-6)
    assert torch.allclose(joints_out[0, 0, 0, 9, 1], r_wrist_orig[1], atol=1e-6)
    # Post-flip slot 10 has the mirrored L-wrist coords.
    assert torch.allclose(joints_out[0, 0, 0, 10, 0], -l_wrist_orig[0], atol=1e-6)
    assert torch.allclose(joints_out[0, 0, 0, 10, 1], l_wrist_orig[1], atol=1e-6)


def test_zeroed_joint_stays_zero_after_flip():
    """A joint at exactly (0, 0) (the MMPose-failure sentinel inside the
    bbox-relative frame) survives the x-flip unchanged because -0.0 == 0.0.
    The bilateral swap may move it to a different slot, but the (0, 0)
    payload propagates intact and the bone recompute correctly suppresses
    bones connected to the zero joint.
    """
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    pairs = get_bone_pairs('coco')
    g = torch.Generator().manual_seed(13)
    joints = (torch.rand(1, 1, 1, 17, 2, generator=g) - 0.5) * 0.8
    # Zero out the L-wrist (slot 9). After flip+swap, slot 10 should hold
    # the zero, slot 9 should hold the (x-flipped) original R-wrist.
    joints[0, 0, 0, 9, :] = 0.0
    bones_orig = recompute_bones_torch(joints, pairs)
    human_pose = torch.cat([joints, bones_orig], dim=-2)
    pos = torch.zeros(1, 1, 1, 2)
    shuttle = torch.zeros(1, 1, 2)

    pose_out, _, _ = flip(human_pose, pos, shuttle)
    joints_out = pose_out[..., :17, :]
    bones_out = pose_out[..., 17:, :]

    assert torch.allclose(joints_out[0, 0, 0, 10], torch.zeros(2), atol=1e-7)
    # Bones connected to the zeroed slot (slot 10 post-swap) should be
    # zero-suppressed by the recompute formula.
    bone_8_10_idx = pairs.index((8, 10))
    assert torch.allclose(bones_out[0, 0, 0, bone_8_10_idx], torch.zeros(2), atol=1e-7)


def test_nose_slot_unchanged_in_swap_table():
    """Slot 0 (nose) has no mirror partner; only x flips, slot stays at 0."""
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    g = torch.Generator().manual_seed(11)
    joints = (torch.rand(1, 2, 2, 17, 2, generator=g) - 0.5) * 0.8
    bones = recompute_bones_torch(joints, get_bone_pairs('coco'))
    human_pose = torch.cat([joints, bones], dim=-2)
    pos = torch.zeros(1, 2, 2, 2)
    shuttle = torch.zeros(1, 2, 2)

    pose_out, _, _ = flip(human_pose, pos, shuttle)
    nose_orig = joints[0, 0, 0, 0]
    nose_post = pose_out[0, 0, 0, 0]
    assert torch.allclose(nose_post[0], -nose_orig[0], atol=1e-6)
    assert torch.allclose(nose_post[1], nose_orig[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Section 5: bone recompute carries flip transforms automatically
# ---------------------------------------------------------------------------

def test_bones_get_x_sign_flip_on_cross_body_bone_after_flip():
    """A cross-body bone connecting bilateral joint pairs (e.g. shoulder-
    shoulder) recomputed from post-flip+post-swap joints should retain
    the same direction as the original under the standard centreline
    flip semantics: the bilateral swap reverses the from/to direction
    in slot space, which composes with the X-component sign flip to
    leave x-magnitude consistent and y unchanged.
    """
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    pairs = get_bone_pairs('coco')

    # Set up a controlled pose where bone (5,6) [L-shoulder -> R-shoulder]
    # has a clean rightward pointing vector.
    n, t, m, j = 1, 1, 1, 17
    joints = torch.zeros(n, t, m, j, 2)
    joints[0, 0, 0, 5] = torch.tensor([0.3, 0.5])  # L-shoulder
    joints[0, 0, 0, 6] = torch.tensor([0.7, 0.5])  # R-shoulder
    bones_orig = recompute_bones_torch(joints, pairs)
    human_pose = torch.cat([joints, bones_orig], dim=-2)
    pos = torch.zeros(n, t, m, 2)
    shuttle = torch.zeros(n, t, 2)

    pose_out, _, _ = flip(human_pose, pos, shuttle)
    bones_out = pose_out[..., 17:, :]

    # Find slot index for the (5,6) bone in pairs.
    bone_56_idx = pairs.index((5, 6))
    bone_56_orig = bones_orig[0, 0, 0, bone_56_idx]
    bone_56_post = bones_out[0, 0, 0, bone_56_idx]

    # The original bone points from L-shoulder (slot 5) to R-shoulder
    # (slot 6) with x-component +0.4. Post-flip+swap, slot 5 holds the
    # mirrored R-shoulder coords (-0.7, 0.5), slot 6 holds the mirrored
    # L-shoulder coords (-0.3, 0.5). bone[bone_56_idx] = slot6 - slot5
    # = (-0.3 - (-0.7), 0) = (0.4, 0). x preserved, y preserved. The
    # cross-body bilateral bone is restored to its original direction.
    assert torch.allclose(bone_56_post, bone_56_orig, atol=1e-6)


def test_bones_swap_slots_for_non_cross_body_bilateral_bones():
    """A non-cross-body bilateral bone like (5,7) [L-shoulder -> L-elbow]
    has its slot-content swapped post-flip with the (6,8) [R-shoulder ->
    R-elbow] bone. Slot of bone (5,7) post-flip should equal the
    x-flipped (6,8) original.
    """
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    pairs = get_bone_pairs('coco')

    # Build a controlled pose with distinguishable arm positions.
    n, t, m, j = 1, 1, 1, 17
    joints = torch.zeros(n, t, m, j, 2)
    joints[0, 0, 0, 5] = torch.tensor([-0.2, 0.0])  # L-shoulder
    joints[0, 0, 0, 7] = torch.tensor([-0.3, 0.1])  # L-elbow
    joints[0, 0, 0, 6] = torch.tensor([0.2, 0.0])   # R-shoulder
    joints[0, 0, 0, 8] = torch.tensor([0.4, 0.2])   # R-elbow
    bones_orig = recompute_bones_torch(joints, pairs)
    human_pose = torch.cat([joints, bones_orig], dim=-2)
    pos = torch.zeros(n, t, m, 2)
    shuttle = torch.zeros(n, t, 2)

    pose_out, _, _ = flip(human_pose, pos, shuttle)
    bones_out = pose_out[..., 17:, :]

    bone_57_idx = pairs.index((5, 7))
    bone_68_idx = pairs.index((6, 8))

    # bone (5,7) original: L-elbow - L-shoulder = (-0.1, 0.1)
    # bone (6,8) original: R-elbow - R-shoulder = (0.2, 0.2)
    # Post-flip+swap, slot 5 holds R-shoulder (x-flipped) = (-0.2, 0.0);
    # slot 7 holds R-elbow (x-flipped) = (-0.4, 0.2). So bone[bone_57_idx] now =
    # slot7 - slot5 = (-0.2, 0.2). That equals the x-flipped (6,8) bone:
    # (-1*0.2, 0.2). Check.
    bone_57_post = bones_out[0, 0, 0, bone_57_idx]
    bone_68_post = bones_out[0, 0, 0, bone_68_idx]
    bone_57_orig = bones_orig[0, 0, 0, bone_57_idx]
    bone_68_orig = bones_orig[0, 0, 0, bone_68_idx]

    expected_57_post = torch.tensor([-bone_68_orig[0], bone_68_orig[1]])
    expected_68_post = torch.tensor([-bone_57_orig[0], bone_57_orig[1]])
    assert torch.allclose(bone_57_post, expected_57_post, atol=1e-6)
    assert torch.allclose(bone_68_post, expected_68_post, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 6: per-clip independence
# ---------------------------------------------------------------------------

def test_flip_per_clip_mask_only_flips_selected_clips(monkeypatch):
    """Force a specific batch coin: clip 0 flips, clip 1 doesn't. The
    unflipped clip must come out byte-identical to its input."""
    flip = CoupledFlip(p=0.5, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(n=2, seed=21)
    pos_orig = pos.clone()

    # Patch torch.rand to return [0.0, 0.99] so flip_mask = [True, False].
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 2:
            return torch.tensor([0.0, 0.99])
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, _ = flip(human_pose, pos, shuttle)
    # Clip 0 flipped
    assert torch.allclose(pos_out[0, ..., 0], 1.0 - pos_orig[0, ..., 0], atol=1e-6)
    # Clip 1 unchanged
    assert torch.allclose(pos_out[1], pos_orig[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Section 7: p boundary behaviour
# ---------------------------------------------------------------------------

def test_flip_p_zero_is_identity():
    flip = CoupledFlip(p=0.0, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(seed=33)
    pose_out, pos_out, shuttle_out = flip(
        human_pose.clone(), pos.clone(), shuttle.clone(),
    )
    assert torch.allclose(pose_out, human_pose, atol=1e-6)
    assert torch.allclose(pos_out, pos, atol=1e-6)
    assert torch.allclose(shuttle_out, shuttle, atol=1e-6)


def test_flip_p_one_flips_every_clip():
    flip = CoupledFlip(p=1.0, n_joints=17, n_bones=19)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(n=8, seed=34)
    _, pos_out, _ = flip(human_pose, pos, shuttle)
    assert torch.allclose(pos_out[..., 0], 1.0 - pos[..., 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Section 8: ConstrainedJitter layered conditional bounds
# ---------------------------------------------------------------------------

def _two_player_pos(top_y_range: tuple[float, float], bot_y_range: tuple[float, float],
                    x_range: tuple[float, float] = (0.3, 0.7),
                    n: int = 1, t: int = 4) -> torch.Tensor:
    """Construct a controlled pos tensor whose top player y lives in
    `top_y_range`, bot player y lives in `bot_y_range`, and both players'
    x lives in `x_range`. Returns shape (n, t, 2, 2)."""
    pos = torch.zeros(n, t, 2, 2)
    pos[..., 0, 1] = torch.linspace(top_y_range[0], top_y_range[1], t).unsqueeze(0)
    pos[..., 1, 1] = torch.linspace(bot_y_range[0], bot_y_range[1], t).unsqueeze(0)
    pos[..., 0, 0] = torch.linspace(x_range[0], x_range[1], t).unsqueeze(0)
    pos[..., 1, 0] = torch.linspace(x_range[0], x_range[1], t).unsqueeze(0)
    return pos


def test_jitter_dy_bound_when_both_players_respect_pre_shift(monkeypatch):
    """Top in [0.1, 0.3] (respects centreline ≤0.5), bot in [0.6, 0.8]
    (respects far baseline ≤1.15). Layered upper bound on dy:
        min(0.5 - 0.3, 1.15 - 0.8) = min(0.2, 0.35) = 0.2.
    Capped at cap_y=0.05.
    Layered lower bound on dy:
        max(-0.15 - 0.1, 0.5 - 0.6) = max(-0.25, -0.1) = -0.1.
    Capped at -cap_y=-0.05.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.1, 0.3), bot_y_range=(0.6, 0.8))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)

    # Force u_y = 1 (sample max), u_x = 0 (sample min) so we read the
    # exact bound endpoints. Patch torch.rand to return predictable values.
    rand_outputs = iter([
        torch.tensor([0.0]),    # roll: 0 < 1.0 -> True
        torch.tensor([1.0]),    # u_y -> sample max
        torch.tensor([0.0]),    # u_x -> sample min
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, _, _ = jitter(human_pose, pos, shuttle)
    dy_applied = pos_out[0, 0, 0, 1] - pos[0, 0, 0, 1]
    dx_applied = pos_out[0, 0, 0, 0] - pos[0, 0, 0, 0]
    # u_y=1 hits dy_hi = min(0.2, 0.05) = 0.05
    assert dy_applied.item() == pytest.approx(0.05, abs=1e-6)
    # u_x=0 hits dx_lo = max(-0.10, -0.15-0.3) = -0.10
    assert dx_applied.item() == pytest.approx(-0.10, abs=1e-6)


def test_jitter_dy_drops_constraint_when_player_already_violates(monkeypatch):
    """Top in [0.4, 0.6] (violates centreline pre-shift, max=0.6 > 0.5).
    Layered upper bound has only the bot constraint; top's drops out.
    Bot in [0.6, 0.8] -> 1.15 - 0.8 = 0.35. Cap binds to 0.05.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.4, 0.6), bot_y_range=(0.6, 0.8))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)

    rand_outputs = iter([
        torch.tensor([0.0]),    # roll fires
        torch.tensor([1.0]),    # u_y -> sample max
        torch.tensor([0.5]),    # u_x -> sample mid (irrelevant for the assertion)
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, _, _ = jitter(human_pose, pos, shuttle)
    dy_applied = pos_out[0, 0, 0, 1] - pos[0, 0, 0, 1]
    # Top dropped, bot constraint = 0.35, cap = 0.05 binds.
    assert dy_applied.item() == pytest.approx(0.05, abs=1e-6)


def test_jitter_one_sided_constraint_drop_collapses_lower_bound(monkeypatch):
    """When all the per-side constraints on one axis direction drop
    (every player already violates), the corresponding ``dy_min`` (or
    ``dy_max``) collapses to the inf-sentinel-replaced 0. Combined with
    the cap on the other side, the sample range becomes one-sided
    (e.g. ``[0, 0.05]``). This is the *case-2* partial-degenerate path
    in the doc: aug still fires, just with a smaller magnitude.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    # Top stretches across the centreline AND below the far-baseline
    # (max=0.7>0.5 violates upper, min=-0.2<-eps violates lower).
    # Bot sits at 0.4 (max=0.4<1+eps respects upper; min=0.4<0.5 violates lower).
    # dy_max = min(top_drops, bot 1.15-0.4=0.75) = 0.75 -> cap binds at 0.05.
    # dy_min: top drops, bot drops -> sentinel 0. Cap floor -0.05; dy_lo = max(0, -0.05) = 0.
    # u_y=1 -> dy = 0.05.
    pos = _two_player_pos(top_y_range=(-0.2, 0.7), bot_y_range=(0.4, 0.4))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)

    rand_outputs = iter([
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, _, _ = jitter(human_pose, pos, shuttle)
    dy_applied = pos_out[0, 0, 0, 1] - pos[0, 0, 0, 1]
    assert dy_applied.item() == pytest.approx(0.05, abs=1e-6)


def test_jitter_both_axes_fully_degenerate_no_shift_and_no_effective(monkeypatch):
    """Case-1 from the doc: every per-side constraint on both axes drops
    pre-shift -> ``dy_hi = dy_lo = dx_hi = dx_lo = 0`` -> no shift applied
    and the clip does not count toward ``Aug/jitter_effective_rate``.

    Construct a clip where:
      - top y stretches above centreline (max=0.7>0.5) AND below far-baseline (min=-0.2<-eps).
      - bot y stretches above far-baseline (max=1.3>1+eps) AND below centreline (min=0.4<0.5).
      - x for both players stretches outside both court x edges
        (max=1.3>1+eps AND min=-0.2<-eps).
    All four dy constraints drop; both dx constraints drop.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(
        top_y_range=(-0.2, 0.7),
        bot_y_range=(0.4, 1.3),
        x_range=(-0.2, 1.3),
    )
    pos_orig = pos.clone()
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)
    shuttle_orig = shuttle.clone()

    rand_outputs = iter([
        torch.tensor([0.0]),
        torch.tensor([1.0]),  # u_y -> would sample dy_hi if envelope had room
        torch.tensor([1.0]),  # u_x -> would sample dx_hi if envelope had room
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, shuttle_out, n_eff = jitter(human_pose, pos, shuttle)
    assert torch.allclose(pos_out, pos_orig, atol=1e-6)
    assert torch.allclose(shuttle_out, shuttle_orig, atol=1e-6)
    assert n_eff == 0


def test_jitter_x_band_constraint(monkeypatch):
    """X has no centreline gating, just the [-eps, 1+eps] band on the
    joint extremes of both players. With x in [0.3, 0.7] and eps=0.15:
        dx_max = 1.15 - 0.7 = 0.45 -> cap binds at 0.10
        dx_min = -0.15 - 0.3 = -0.45 -> cap binds at -0.10.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.2, 0.4), bot_y_range=(0.6, 0.8),
                          x_range=(0.3, 0.7))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)

    rand_outputs = iter([
        torch.tensor([0.0]),
        torch.tensor([0.5]),    # u_y mid
        torch.tensor([1.0]),    # u_x = 1 -> sample dx_hi
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, pos_out, _, _ = jitter(human_pose, pos, shuttle)
    dx_applied = pos_out[0, 0, 0, 0] - pos[0, 0, 0, 0]
    assert dx_applied.item() == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# Section 9: zero-frame preservation, shuttle off-screen, p boundaries
# ---------------------------------------------------------------------------

def test_jitter_preserves_zero_frames_in_pos():
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.2, 0.4), bot_y_range=(0.6, 0.8))
    pos[0, 1, 0] = 0.0  # zero out top player at frame 1
    pos[0, 2, 1] = 0.0  # zero out bot player at frame 2
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)

    _, pos_out, _, _ = jitter(human_pose, pos, shuttle)
    assert torch.allclose(pos_out[0, 1, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(pos_out[0, 2, 1], torch.zeros(2), atol=1e-6)


def test_jitter_preserves_zero_frames_in_shuttle():
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.2, 0.4), bot_y_range=(0.6, 0.8))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)
    shuttle[0, 0] = 0.0  # zero out shuttle at frame 0 (TrackNet failure)

    _, _, shuttle_out, _ = jitter(human_pose, pos, shuttle)
    assert torch.allclose(shuttle_out[0, 0], torch.zeros(2), atol=1e-6)


def test_jitter_zeros_shuttle_when_shifted_off_screen():
    """A shuttle near the [0, 1] edge that gets shifted outside lands at
    (0, 0) per the TrackNet off-screen sentinel convention.
    """
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.0, 0.2), bot_y_range=(0.6, 0.8))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    # Shuttle near top-left corner; if dx is positive this stays in-bounds,
    # but if dy is negative (or both) it can go off-screen.
    shuttle = torch.full((1, 4, 2), 0.02)

    # Force a shift that pushes shuttle.y negative.
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 1:
            return torch.tensor([0.0])  # u_y -> dy_lo (negative since top is at 0); roll fires
        return real_rand(*size, **kwargs)

    # Patch and trigger.
    import unittest.mock as mock
    with mock.patch('torch.rand', side_effect=fake_rand):
        _, _, shuttle_out, _ = jitter(human_pose, pos, shuttle)
    # u_y=0 -> dy = dy_lo. Top y starts at 0 -> dy_lo = max(-0.05, -eps - 0) = -0.05 (cap binds)
    # Shuttle y becomes 0.02 + (-0.05) = -0.03 < 0 -> off-screen sentinel
    assert torch.allclose(shuttle_out[0, 0], torch.zeros(2), atol=1e-6)


def test_jitter_p_zero_is_identity():
    jitter = ConstrainedJitter(p_roll=0.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.2, 0.4), bot_y_range=(0.6, 0.8))
    human_pose = torch.zeros(1, 4, 2, 36, 2)
    shuttle = torch.full((1, 4, 2), 0.5)
    pose_out, pos_out, shuttle_out, n_eff = jitter(
        human_pose.clone(), pos.clone(), shuttle.clone(),
    )
    assert torch.allclose(pose_out, human_pose, atol=1e-6)
    assert torch.allclose(pos_out, pos, atol=1e-6)
    assert torch.allclose(shuttle_out, shuttle, atol=1e-6)
    assert n_eff == 0


def test_jitter_effective_count_excludes_unrolled_clips(monkeypatch):
    """Two clips, only the first rolls in. Effective count must equal 1."""
    jitter = ConstrainedJitter(p_roll=0.5, cap_y=0.05, cap_x=0.10, eps=0.15)
    pos = _two_player_pos(top_y_range=(0.2, 0.4), bot_y_range=(0.6, 0.8), n=2)
    human_pose = torch.zeros(2, 4, 2, 36, 2)
    shuttle = torch.full((2, 4, 2), 0.5)

    rand_outputs = iter([
        torch.tensor([0.0, 0.99]),  # roll: clip 0 yes, clip 1 no
        torch.tensor([0.5, 0.5]),    # u_y, both clips
        torch.tensor([0.5, 0.5]),    # u_x, both clips
    ])
    real_rand = torch.rand
    def fake_rand(*size, **kwargs):
        if size and isinstance(size[0], int) and len(size) == 1 and size[0] == 2:
            return next(rand_outputs)
        return real_rand(*size, **kwargs)
    monkeypatch.setattr(torch, 'rand', fake_rand)

    _, _, _, n_eff = jitter(human_pose, pos, shuttle)
    assert n_eff == 1


# ---------------------------------------------------------------------------
# Section 10: joints/bones untouched by jitter, cross-stream isolation
# ---------------------------------------------------------------------------

def test_jitter_does_not_touch_joints_or_bones():
    jitter = ConstrainedJitter(p_roll=1.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    human_pose, pos, shuttle, joints_orig, bones_orig = _random_clip_tensors(seed=51)
    pose_out, _, _, _ = jitter(human_pose, pos, shuttle)
    joints_out = pose_out[..., :17, :]
    bones_out = pose_out[..., 17:, :]
    assert torch.allclose(joints_out, joints_orig, atol=1e-7)
    assert torch.allclose(bones_out, bones_orig, atol=1e-7)


def test_flip_preserves_input_when_p_zero_and_jitter_does_too():
    """Combined identity check: with p=0 on both, the full augmentation
    pipeline is a no-op.
    """
    flip = CoupledFlip(p=0.0, n_joints=17, n_bones=19)
    jitter = ConstrainedJitter(p_roll=0.0, cap_y=0.05, cap_x=0.10, eps=0.15)
    human_pose, pos, shuttle, _, _ = _random_clip_tensors(seed=61)
    pose_a, pos_a, shuttle_a = flip(human_pose.clone(), pos.clone(), shuttle.clone())
    pose_b, pos_b, shuttle_b, _ = jitter(pose_a, pos_a, shuttle_a)
    assert torch.allclose(pose_b, human_pose, atol=1e-6)
    assert torch.allclose(pos_b, pos, atol=1e-6)
    assert torch.allclose(shuttle_b, shuttle, atol=1e-6)
