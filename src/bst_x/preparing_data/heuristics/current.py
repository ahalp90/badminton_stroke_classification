"""``current`` heuristic: byte-identity gate for the raw-extract + apply path.

Replicates ``detect_players_2d`` (in ``preparing_data.prepare_train_on_shuttleset``)
by starting from the raw MMPose arrays written by ``preparing_data.raw_extract``
rather than from a live MMPose run. The output of this variant must match
the committed filtered extract bit-for-bit on ``_failed.npy`` and to
``atol = 1e-5`` on ``_pos.npy`` and ``_joints.npy``. A mismatch means the
plumbing around the raw extract is wrong, and no other heuristic variant
should be trusted until it is fixed.

Per-frame logic mirrors the original:

1. Take the ``ndet[f]`` real detections from the raw arrays.
2. If fewer than two, mark the frame failed and fill zeros.
3. Otherwise run ``check_pos_in_court`` (ankle-midpoint, ``eps = 0.01``) on
   those real detections.
4. If the in-court count is not exactly two, mark failed and fill zeros.
5. Else sort the two in-court detections by y so Top precedes Bottom and
   write their positions plus bbox-diagonal-normalised joints.

Known divergence vs the original: ``raw_extract.py`` truncates to the top
``N_max`` detections per frame by ``bbox_score`` when a frame exceeds the
cap. At ``N_max = 16`` only ~0.79% of frames on the 1,716-clip busted set
hit the cap; in almost every hit, the two real players rank in the top
``N_max``. Byte-identity is preserved under this condition. If a real
player is ever ranked below ``N_max`` on a cap-hit frame, the filter
decision can diverge for that frame; this is documented in
``docs/architecture_notes/mmpose_heuristic/historical_mmpose_heuristic_investigation.md``.

The imports from ``prepare_train_on_shuttleset`` are deferred to ``apply``'s
first call because that module has a top-level ``from mmpose.apis import
MMPoseInferencer``; we want module-level ``import current`` to work in
environments without MMPose (e.g. local smoke tests).
"""
from __future__ import annotations

import numpy as np

from .base import ClipContext, HeuristicOutput, J, RawClip


def apply(raw: RawClip, ctx: ClipContext, **_hyperparams) -> HeuristicOutput:
    """Apply the current (committed-pipeline) filter to a raw clip.

    ``_hyperparams`` is accepted and ignored so the CLI can pass the
    sticky_anchor hyperparam block uniformly to every registered variant.
    """
    # Lazy import: prepare_train_on_shuttleset pulls in mmpose at module load.
    from preparing_data.prepare_train_on_shuttleset import (  # noqa: PLC0415
        check_pos_in_court,
        normalize_joints,
    )

    num_frames = raw.kps.shape[0]

    failed = np.zeros(num_frames, dtype=bool)
    pos = np.zeros((num_frames, 2, 2), dtype=np.float64)
    joints = np.zeros((num_frames, 2, J, 2), dtype=np.float64)

    for f in range(num_frames):
        n = int(raw.ndet[f])
        if n < 2:
            failed[f] = True
            continue

        # Cast float32 raw inputs up to float64 so the arithmetic chain
        # matches the committed extract, where MMPose's Python-list output
        # was wrapped in np.array (default float64) before projection.
        keypoints = raw.kps[f, :n].astype(np.float64)

        in_court, pos_normalized = check_pos_in_court(
            keypoints, ctx.vid, ctx.all_court_info, ctx.res_df,
        )
        in_court_pid = np.nonzero(in_court)[0]

        if len(in_court_pid) != 2:
            failed[f] = True
            continue

        bboxes = raw.bboxes[f, :n].astype(np.float64)

        # Top (smaller y) before Bottom, matching detect_players_2d.
        if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
            in_court_pid = np.flip(in_court_pid)

        pos[f] = pos_normalized[in_court_pid]
        # center_align=True matches the CLI invocation that produced the
        # committed extract (prepare_train_on_shuttleset.py line 1172;
        # joints_center_align=True there overrides detect_players_2d's
        # function-level default of False). Without this, normalised joints
        # land ~0.47 higher because the (bbox-centre - bbox-top-left) / dist
        # offset (~0.5, 0.5 for roughly square bboxes) is not subtracted.
        joints[f] = normalize_joints(
            arr=keypoints[in_court_pid],
            bbox=bboxes[in_court_pid],
            v_height=None,
            center_align=True,
        )

    return HeuristicOutput(pos=pos, joints=joints, failed=failed)
