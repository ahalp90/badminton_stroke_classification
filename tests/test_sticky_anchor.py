"""Pinning tests for the sticky_anchor heuristic.

Synthetic data only: a tiny RawClip + a court whose homography is the
identity at 1280x720 so border_L/R/U/D = (0, 1280, 0, 720) and
``normalize_position`` collapses to ``(px / 1280, py / 720)``. Halfcourt
centres become the canonical ``(0.5, 0.25)`` (Top) and ``(0.5, 0.75)``
(Bottom).

The X3D-S wrist-crop layer will consume the same per-slot pose stream
sticky_anchor produces, so these tests pin the picking and EMA-reset
invariants before that work lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bst_refactor.stroke_classification.preparing_data.heuristics.base import (
    ClipContext,
    RawClip,
)
from src.bst_refactor.stroke_classification.preparing_data.heuristics.sticky_anchor import (
    SLOT_BOTTOM,
    SLOT_TOP,
    StickyAnchorParams,
    _pick_one_frame,
    _run_clip,
)

J = 17  # COCO 17-joint skeleton


def _identity_court_ctx(vid: int = 1):
    """Build a synthetic ClipContext whose pixel->normalised mapping is identity over [0, 1280]x[0, 720]."""
    H = np.eye(3, dtype=np.float64)
    court_info = {
        "H": H,
        "border_L": 0.0,
        "border_R": 1280.0,
        "border_U": 0.0,
        "border_D": 720.0,
    }
    all_court_info = {vid: court_info}
    res_df = pd.DataFrame({"width": [1280], "height": [720]}, index=[vid])
    return ClipContext(vid=vid, all_court_info=all_court_info, res_df=res_df)


def _bbox_for(norm_x: float, norm_y: float, half_w: float = 30.0, half_h: float = 60.0) -> np.ndarray:
    """Return a bbox whose bottom-centre projects to ``(norm_x, norm_y)`` in normalised court coords."""
    cx = norm_x * 1280.0
    by = norm_y * 720.0
    x1, x2 = cx - half_w, cx + half_w
    y1, y2 = by - 2 * half_h, by
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _standing_kps_for_bbox(bbox: np.ndarray) -> np.ndarray:
    """Synthesise a plausible standing pose with shoulders above hips, knees below hips."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    h = y2 - y1
    kps = np.zeros((J, 2), dtype=np.float32)
    # Shoulders: 5 left, 6 right, near top of bbox.
    kps[5] = (cx - 15, y1 + 0.25 * h)
    kps[6] = (cx + 15, y1 + 0.25 * h)
    # Hips: 11 left, 12 right, mid-bbox.
    kps[11] = (cx - 12, y1 + 0.55 * h)
    kps[12] = (cx + 12, y1 + 0.55 * h)
    # Knees: 13 left, 14 right, below hips.
    kps[13] = (cx - 14, y1 + 0.80 * h)
    kps[14] = (cx + 14, y1 + 0.80 * h)
    return kps


def _sitting_kps_for_bbox(bbox: np.ndarray) -> np.ndarray:
    """Synthesise a sitting pose: knees roughly at hip level horizontally (perpendicular to body axis)."""
    kps = _standing_kps_for_bbox(bbox)
    # Override knees to sit roughly at hip y.
    knee_y = (kps[11, 1] + kps[12, 1]) / 2
    kps[13] = (kps[11, 0] - 30, knee_y)
    kps[14] = (kps[12, 0] + 30, knee_y)
    return kps


def _build_raw_clip(per_frame_candidates: list[list[tuple[np.ndarray, np.ndarray, float]]]) -> RawClip:
    """Build a RawClip from ``[[(bbox, kps, score), ...], ...]``.

    Each inner list is one frame; pads the detection axis to the per-clip max
    with NaNs so the writer's NaN-padding contract holds.
    """
    f_count = len(per_frame_candidates)
    n_max = max(len(fc) for fc in per_frame_candidates) if f_count else 0
    if n_max == 0:
        n_max = 1  # avoid zero-shape arrays even on an all-empty clip

    kps = np.full((f_count, n_max, J, 2), np.nan, dtype=np.float32)
    bboxes = np.full((f_count, n_max, 4), np.nan, dtype=np.float32)
    scores = np.full((f_count, n_max), np.nan, dtype=np.float32)
    kp_scores = np.full((f_count, n_max, J), np.nan, dtype=np.float32)
    ndet = np.zeros(f_count, dtype=np.int64)

    for f, frame_candidates in enumerate(per_frame_candidates):
        ndet[f] = len(frame_candidates)
        for c, (bbox, kp, score) in enumerate(frame_candidates):
            bboxes[f, c] = bbox
            kps[f, c] = kp
            scores[f, c] = score
            kp_scores[f, c] = 0.9  # uniform; the heuristic doesn't read this directly

    return RawClip(kps=kps, bboxes=bboxes, scores=scores, kp_scores=kp_scores, ndet=ndet)


def _identity_normalize_joints(arr, bbox, v_height, center_align):
    """No-op stand-in for ``prepare_train_on_shuttleset.normalize_joints``.

    The picking logic does not depend on joint normalisation; tests 5-7 only
    inspect EMA state, not the output joints.
    """
    return np.asarray(arr, dtype=np.float64)


def _params(**overrides) -> StickyAnchorParams:
    """Build a StickyAnchorParams with optional field overrides."""
    return StickyAnchorParams(**overrides)


# -- Test 1: Voronoi partition picks the right side --------------------------


def test_voronoi_partition_picks_correct_side():
    """Two candidates straddling the halfcourt line: the upper one fills Top, the lower one fills Bottom."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Top-side candidate at norm y = 0.20; Bottom-side at norm y = 0.80.
    top_bbox = _bbox_for(0.50, 0.20)
    bot_bbox = _bbox_for(0.50, 0.80)
    top_kps = _standing_kps_for_bbox(top_bbox)
    bot_kps = _standing_kps_for_bbox(bot_bbox)

    raw = _build_raw_clip([[(top_bbox, top_kps, 0.9), (bot_bbox, bot_kps, 0.9)]])
    res = _pick_one_frame(raw, 0, ema, halfcourt_centre, ctx, _params())
    assert res is not None
    picks, court_base_pos, _, _ = res
    assert picks[SLOT_TOP] == 0  # candidate 0 is the upper one
    assert picks[SLOT_BOTTOM] == 1  # candidate 1 is the lower one

    # Swap input order: the assignment must not depend on candidate order.
    raw_swapped = _build_raw_clip([[(bot_bbox, bot_kps, 0.9), (top_bbox, top_kps, 0.9)]])
    res = _pick_one_frame(raw_swapped, 0, ema, halfcourt_centre, ctx, _params())
    assert res is not None
    picks, *_ = res
    assert picks[SLOT_TOP] == 1
    assert picks[SLOT_BOTTOM] == 0


# -- Test 2: Bottom-first slot ordering with cross-slot exclusion ------------


def test_bottom_first_with_cross_slot_exclusion():
    """When both candidates fall on the Bottom side, Bottom fills with the closer one and Top stays empty."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Both candidates well into the Bottom half.
    bbox_a = _bbox_for(0.45, 0.78)
    bbox_b = _bbox_for(0.55, 0.72)
    raw = _build_raw_clip([[
        (bbox_a, _standing_kps_for_bbox(bbox_a), 0.9),
        (bbox_b, _standing_kps_for_bbox(bbox_b), 0.9),
    ]])
    res = _pick_one_frame(raw, 0, ema, halfcourt_centre, ctx, _params())
    assert res is not None
    picks, *_ = res

    # Bottom slot: whichever candidate is closer to (0.5, 0.75) wins.
    # bbox_b at (0.55, 0.72) is closer than bbox_a at (0.45, 0.78).
    assert picks[SLOT_BOTTOM] == 1
    # Top slot: both candidates are on the Bottom side of the Voronoi partition,
    # and the Bottom-picked one is excluded from Top consideration. Either Top
    # ends up empty or it fills with the OTHER bottom-side candidate; the test
    # pins the "no Top pick" outcome that the implementation produces.
    assert picks[SLOT_TOP] == -1


# -- Test 3: Sitting tiebreaker with fallback --------------------------------


def test_sitting_tiebreaker_prefers_standing_then_falls_back():
    """Two near-tied candidates: non-sitting wins. If both sit, fall back to plain argmin."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Both Top-side candidates at near-identical anchor distance, large bboxes.
    bbox_sitting = _bbox_for(0.50, 0.25, half_w=40.0, half_h=70.0)
    bbox_standing = _bbox_for(0.495, 0.255, half_w=40.0, half_h=70.0)

    sitting_kps = _sitting_kps_for_bbox(bbox_sitting)
    standing_kps = _standing_kps_for_bbox(bbox_standing)

    # Single Top-side frame: sitting cand at index 0, standing at index 1.
    # Add a Bottom-side cand so the heuristic doesn't reject the frame.
    bot_bbox = _bbox_for(0.5, 0.75)
    bot_kps = _standing_kps_for_bbox(bot_bbox)
    raw = _build_raw_clip([[
        (bbox_sitting, sitting_kps, 0.9),
        (bbox_standing, standing_kps, 0.9),
        (bot_bbox, bot_kps, 0.9),
    ]])
    # sanity_ceiling generous so the standing cand at norm y=0.255 stays eligible.
    kw = _params(tiebreaker_tol=0.05)
    res = _pick_one_frame(raw, 0, ema, halfcourt_centre, ctx, kw)
    assert res is not None
    picks, *_ = res
    assert picks[SLOT_TOP] == 1, "non-sitting candidate should win the tiebreaker"

    # Both candidates sitting: fallback rule reverts to plain argmin on the
    # eligible set. The bbox_sitting candidate is exactly at the Top anchor
    # (distance 0.0) and bbox_standing is ~0.007 away, so the closer one
    # wins regardless of pose.
    raw_both_sit = _build_raw_clip([[
        (bbox_sitting, _sitting_kps_for_bbox(bbox_sitting), 0.9),
        (bbox_standing, _sitting_kps_for_bbox(bbox_standing), 0.9),
        (bot_bbox, bot_kps, 0.9),
    ]])
    res = _pick_one_frame(raw_both_sit, 0, ema, halfcourt_centre, ctx, kw)
    assert res is not None
    picks, *_ = res
    assert picks[SLOT_TOP] == 0, "fallback should revert to argmin when all candidates sit"


# -- Test 4: Rally-presence rejection ----------------------------------------


def test_rally_presence_rejects_when_both_picks_far_off_court():
    """When both picks land far outside the generous court margin, the frame is rejected."""
    ctx = _identity_court_ctx()
    halfcourt_centre = np.array([[0.5, 0.25], [0.5, 0.75]])
    ema = halfcourt_centre.copy()

    # Both candidates far outside court but inside the per-slot sanity_ceiling
    # of 0.6 from their respective anchors. Using sanity_ceiling=2.0 below to
    # widen the picking step; the rally-presence guard fires inside _pick_one_frame.
    far_top_bbox = _bbox_for(-0.5, -0.5)
    far_bot_bbox = _bbox_for(1.5, 1.5)
    raw = _build_raw_clip([[
        (far_top_bbox, _standing_kps_for_bbox(far_top_bbox), 0.9),
        (far_bot_bbox, _standing_kps_for_bbox(far_bot_bbox), 0.9),
    ]])
    res = _pick_one_frame(
        raw, 0, ema, halfcourt_centre, ctx,
        _params(sanity_ceiling=5.0, generous_margin=0.15),
    )
    assert res is None, "both picks far outside court must trigger rally-presence rejection"


# -- Test 5: Full-frame failure resets EMA to halfcourt centres --------------


def test_full_frame_failure_resets_ema():
    """After a frame with zero detections, EMA for both slots is at halfcourt centres."""
    ctx = _identity_court_ctx()

    # Frame 0: a valid pick so the EMA advances to a non-centre value.
    bot_bbox = _bbox_for(0.45, 0.80)
    top_bbox = _bbox_for(0.55, 0.20)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_bbox, _standing_kps_for_bbox(bot_bbox), 0.9),
    ]
    # Frame 1: zero detections (full-frame failure).
    frame_1: list = []

    raw = _build_raw_clip([frame_0, frame_1])
    output, ema_history = _run_clip(raw, ctx, _identity_normalize_joints, _params())

    assert not output.failed[0]  # frame 0 picks both slots
    assert output.failed[1]  # frame 1 zero detections

    # Frame 0 EMA must have moved off the centres (alpha=0.1, target slightly off-centre).
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])
    assert not np.allclose(ema_history[0], halfcourt), "EMA should advance after a successful frame"

    # Frame 1 EMA must be back at the halfcourt centres.
    np.testing.assert_allclose(ema_history[1], halfcourt, atol=1e-12)


# -- Test 6: Mixed-pick frame resets only the unpicked slot's EMA ------------


def test_mixed_pick_resets_only_unpicked_slot():
    """A frame that picks Bottom but not Top resets only Top's EMA; Bottom's EMA advances."""
    ctx = _identity_court_ctx()
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])

    # Frame 0: both picks land, EMA advances on both slots.
    top_bbox = _bbox_for(0.55, 0.20)
    bot_bbox = _bbox_for(0.45, 0.80)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_bbox, _standing_kps_for_bbox(bot_bbox), 0.9),
    ]
    # Frame 1: only a Bottom-side candidate (no Top candidate exists).
    bot_only_bbox = _bbox_for(0.50, 0.78)
    frame_1 = [
        (bot_only_bbox, _standing_kps_for_bbox(bot_only_bbox), 0.9),
    ]
    raw = _build_raw_clip([frame_0, frame_1])
    output, ema_history = _run_clip(raw, ctx, _identity_normalize_joints, _params())

    # Frame 1: Top EMA reset to (0.5, 0.25); Bottom EMA continued to advance.
    np.testing.assert_allclose(ema_history[1, SLOT_TOP], halfcourt[SLOT_TOP], atol=1e-12)
    assert not np.allclose(ema_history[1, SLOT_BOTTOM], halfcourt[SLOT_BOTTOM]), \
        "Bottom EMA should keep advancing when Bottom is picked"
    # Frame 1 marked as a partial failure.
    assert output.failed[1]


# -- Test 7: update_gate_eps blocks EMA updates from off-court picks ---------


def test_update_gate_eps_blocks_off_court_update():
    """A pick whose normalised position is outside the gate margin does not move the EMA."""
    ctx = _identity_court_ctx()
    halfcourt = np.array([[0.5, 0.25], [0.5, 0.75]])

    # Frame 0: an off-court Bottom pick (norm y = 1.5) plus an on-court Top
    # pick. update_gate_eps is small so the off-court pick is gated out.
    # sanity_ceiling needs to be wide enough for the off-court bottom to
    # still be considered eligible.
    top_bbox = _bbox_for(0.50, 0.20)
    bot_offcourt_bbox = _bbox_for(0.50, 1.50)
    frame_0 = [
        (top_bbox, _standing_kps_for_bbox(top_bbox), 0.9),
        (bot_offcourt_bbox, _standing_kps_for_bbox(bot_offcourt_bbox), 0.9),
    ]
    raw = _build_raw_clip([frame_0])
    output, ema_history = _run_clip(
        raw, ctx, _identity_normalize_joints,
        _params(
            sanity_ceiling=2.0,
            generous_margin=2.0,  # disable rally-presence rejection for this test
            update_gate_eps=0.01,
        ),
    )

    # Both picks land (rally-presence is widened above), but only the
    # in-court Top pick is allowed to move its EMA. The off-court Bottom
    # pick is gated out, so Bottom EMA stays at the halfcourt centre.
    np.testing.assert_allclose(ema_history[0, SLOT_BOTTOM], halfcourt[SLOT_BOTTOM], atol=1e-12)
    # Top EMA must have moved (alpha=0.1, target around (0.55, 0.20)).
    assert not np.allclose(ema_history[0, SLOT_TOP], halfcourt[SLOT_TOP])
