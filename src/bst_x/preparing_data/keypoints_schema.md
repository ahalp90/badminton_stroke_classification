# COCO 17-Keypoint Schema (RTMPose)

Reference for the keypoint format extracted by MMPose RTMPose-L in the pose estimation step (`prepare_train_on_shuttleset.py` Step 2).

---

## Joint Index Map

| Index | Joint           | Region |
|-------|-----------------|--------|
| 0     | Nose            | Head   |
| 1     | Left Eye        | Head   |
| 2     | Right Eye       | Head   |
| 3     | Left Ear        | Head   |
| 4     | Right Ear       | Head   |
| 5     | Left Shoulder   | Torso  |
| 6     | Right Shoulder  | Torso  |
| 7     | Left Elbow      | Arm    |
| 8     | Right Elbow     | Arm    |
| 9     | Left Wrist      | Arm    |
| 10    | Right Wrist     | Arm    |
| 11    | Left Hip        | Torso  |
| 12    | Right Hip       | Torso  |
| 13    | Left Knee       | Leg    |
| 14    | Right Knee      | Leg    |
| 15    | Left Ankle      | Leg    |
| 16    | Right Ankle     | Leg    |

"Left" and "Right" are from the subject's perspective (standard COCO convention).

---

## 19 Bone Pairs

Defined in `shuttleset_dataset.py:get_bone_pairs()`. Each pair is `(start_index, end_index)`.

| Region           | Pairs                                           |
|------------------|-------------------------------------------------|
| Head             | (0,1) (0,2) (1,2) (1,3) (2,4)                  |
| Ears to shoulders| (3,5) (4,6)                                     |
| Arms             | (5,7) (7,9) (6,8) (8,10)                       |
| Torso            | (5,6) (5,11) (6,12) (11,12)                    |
| Legs             | (11,13) (13,15) (12,14) (14,16)                |

---

## JnB (Joints and Bones) Representations

Built during collation (Step 3). All shapes are per-frame, per-player.

| Style        | Content                           | Features | Shape `(t, 2, F, 2)` |
|--------------|-----------------------------------|----------|-----------------------|
| `J_only`     | 17 raw joints                     | 17       | `(t, 2, 17, 2)`      |
| `JnB_interp` | 17 joints + 19 bone midpoints    | 36       | `(t, 2, 36, 2)`      |
| `JnB_bone`  | 17 joints + 19 bone vectors       | 36       | `(t, 2, 36, 2)`      |
| `Jn2B`      | 36 interp points + 19 bone vectors| 55       | `(t, 2, 55, 2)`      |

- **Bone vectors** = `end_joint - start_joint` (encodes direction and limb length)
- **Bone midpoints** = `(start_joint + end_joint) / 2` (interpolated position)
- The `2` in dimension 1 is the two players (Top, Bottom)
- The trailing `2` is `(x, y)`

See `shuttleset_dataset.py:create_bones()` and `interpolate_joints()` for the computation, and `POSE_BONE_MULTIPLIER` for the mapping used to calculate `in_dim`.

---

## Key Indices for Arch 1

- **Wrist (racket hand):** 9 (left) / 10 (right)
- **Torso height (for crop scaling):** shoulders 5, 6 and hips 11, 12
- **Foot keypoints (court projection):** ankles 15, 16 -- used by `check_pos_in_court()` to assign players to Top/Bottom

---

## Design Notes

**Per-keypoint confidence scores are not stored.** MMPose returns `keypoint_scores` (shape `(J,)`) alongside coordinates, but the pipeline extracts only xy. This follows the original BST implementation. Given the strength of RTMPose-L and the high quality of ShuttleSet broadcast footage, confidence values are near-constant across samples -- feeding them to the model would grow parameters for no useful signal.

**Coordinates are not clamped to image bounds.** RTMPose's SimCC regression head can predict keypoints outside the image frame (negative values or exceeding width/height). Neither RTMPose nor `normalize_joints()` clips these -- out-of-bounds raw values become out-of-bounds normalized values in the collated arrays. This is rare on broadcast footage but can occur for joints near frame edges (e.g., a wrist extending off-screen during a swing).

**Failed frames are retained, not dropped.** RTMPose runs independently per frame with no temporal interpolation or smoothing. When a frame fails detection (< 2 people detected, or != 2 players on court), joints and positions are zeroed for that frame, and the frame stays in the sequence. At collation, shuttle coordinates are also zeroed on failed frames. No clip is ever excluded from training based on failed frames -- the model sees zeros for those frames. A missing shuttle CSV, by contrast, is a hard crash (`FileNotFoundError`), not a silent skip. Frame-count mismatches between MMPose and TrackNetV3 (1-2 frames from different video backends) are resolved by truncating both to the shorter length.
