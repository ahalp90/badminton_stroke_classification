"""Prepare ShuttleSet training data: shuttle detection, pose estimation, and collation.

Bridges the gap between the pipeline's clip output and BST's expected input format.
Three steps, each independently skippable:
  Step 1: Shuttle trajectory detection via TrackNetV3
  Step 2: 2D/3D player pose estimation via MMPose + court projection
  Step 3: Collate per-clip .npy files into batch-ready arrays

Run from stroke_classification/:
    python -m preparing_data.prepare_train_on_shuttleset --help
"""

from mmpose.apis import MMPoseInferencer

import argparse
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

import subprocess
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

import sys
import os

if __name__ == "__main__":
    # Add stroke_classification/ for preparing_data imports (matches bst_train.py)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # Add project root for pipeline.config imports
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from preparing_data.shuttleset_dataset import (
    get_bone_pairs,
    make_seq_len_same,
    create_bones,
    interpolate_joints,
)
from pipeline.config import (
    CLIPS_OUTPUT_DIR,
    SET_INFO_DIR,
    RESOLUTION_CSV_PATH,
    SHUTTLE_CSV_DIR,
    Taxonomy,
    TAXONOMIES,
    TAXONOMY_UNE_MERGE_V1,
    DEFAULT_TAXONOMY,
)


def get_H(homography_info: pd.Series):
    """Get from the pd object."""
    h_str: str = homography_info["homography_matrix"]
    H = h_str.strip().replace("[", "").replace("]", "").replace(",", "").split()
    H = np.array(list(map(float, H))).reshape((3, 3))
    return H


def get_corner_camera(homography_info: pd.Series):
    """Get from the pd object."""
    corner_camera = homography_info.loc["upleft_x":"downright_y"]
    corner_camera = corner_camera.to_numpy(dtype=float).reshape((2, 4))
    return corner_camera


def scale_pos_by_resolution(arr: np.ndarray, width, height, aim_w=1280, aim_h=720):
    """
    The shape of 2D `arr` is (2, N) or (3, N) if homogeneous.
    """
    new_arr = arr.copy()
    new_arr[0, :] *= aim_w / width
    new_arr[1, :] *= aim_h / height
    return new_arr


def convert_homogeneous(arr: np.ndarray):
    """
    The shape of 2D `arr` is (2, N). => The output will be (3, N).
    """
    return np.concatenate((arr, np.full((1, arr.shape[-1]), 1.0)), axis=0)


def project(H: np.ndarray, P_prime: np.ndarray):
    """
    Transform coordinates from the camera system to the court system.

    H: (3, 3)
    P_prime: (3, N)
    Output: (2, N)
    """
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]  # /= w
    return P


def get_court_info(homo_df: pd.DataFrame, vid: int):
    """
    Get the homography matrix and the 4 corners of the court in the court coordinate corresponding to the video.
    """
    homography_info = homo_df.loc[vid]

    H = get_H(homography_info)
    corner_camera = get_corner_camera(homography_info)
    corner_camera = convert_homogeneous(corner_camera)

    corner_court = project(H, corner_camera)
    return {
        "H": H,
        "border_L": corner_court[0, 0],
        "border_R": corner_court[0, 1],
        "border_U": corner_court[1, 0],
        "border_D": corner_court[1, 2],
    }


def to_court_coordinate(
    arr_camera: np.ndarray, vid: int, all_court_info: dict, res_df: pd.DataFrame
):
    """
    Convert the camera coordinate system to the court coordinate system.

    If the camera coordinate is not from the resolution (1280, 720):
        It will be scaled to represent in (1280, 720).

    The shape of 2D `arr_camera` is (2, N).
    """
    res_info = res_df.loc[vid]  # for resolution scaling
    H = all_court_info[vid]["H"]

    arr_camera = scale_pos_by_resolution(
        arr_camera, width=res_info["width"], height=res_info["height"]
    )
    arr_camera = convert_homogeneous(arr_camera)
    arr_court = project(H, arr_camera)
    return arr_court


def normalize_position(arr: np.ndarray, court_info: dict):
    """
    Normalized by court boundary.

    `arr`: (2, N). There are N 'x' and N 'y'.
    Output: (2, N). Every 'x', 'y' in-court should be in [0, 1].
    """
    x_dist = court_info["border_R"] - court_info["border_L"]
    y_dist = court_info["border_D"] - court_info["border_U"]

    x_normalized = (arr[0, :] - court_info["border_L"]) / x_dist
    y_normalized = (arr[1, :] - court_info["border_U"]) / y_dist
    return np.stack((x_normalized, y_normalized))


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    """
    - `arr`: (m, J, 2), m=2.
    - `bbox`: (m, 4), m=2.

    Output: (m, J, 2), m=2.
    """
    # If v_height == None and center_align == False,
    # this normalization method is same as that used in TemPose.
    if v_height:
        dist = v_height / 4
    else:  # bbox diagonal dist
        dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / dist
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_shuttlecock(arr: np.ndarray, v_width, v_height):
    """
    Normalized by the video resolution.

    `arr`: (t, 2). There are t 'x' and t 'y'.
    Output: (t, 2). Every 'x', 'y' in-court should be in [0, 1].
    """
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)


def check_pos_in_court(keypoints: np.ndarray, vid: int, all_court_info: dict, res_df):
    """
    The shape of `keypoints` is (m, J, 2).

    Output:
        in_court: (m)
        pos_court_normalized: (m, 2)
    """
    n_people = keypoints.shape[0]

    feet_camera = keypoints[:, -2:, :]
    # feet_camera: (m, J, 2), J=2
    feet_camera = feet_camera.reshape(-1, 2).T
    # feet_camera: (2, m*J)

    feet_court = to_court_coordinate(
        feet_camera, vid=vid, all_court_info=all_court_info, res_df=res_df
    )
    feet_court = feet_court.reshape(2, n_people, -1)
    # feet_court: (2, m, J)

    pos_court = feet_court.mean(axis=-1)  # middle point between feet
    # pos_court: (2, m)
    pos_court_normalized = normalize_position(
        pos_court, court_info=all_court_info[vid]
    ).T
    # pos_court_normalized: (m, 2)

    eps = 0.01  # soft border
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    # in_court: (m)
    return in_court, pos_court_normalized


def detect_players_2d(
    inferencer: MMPoseInferencer,
    video_path: Path,
    all_court_info: dict,
    res_df: pd.DataFrame,
    J=17,
    normalized_by_v_height=False,
    center_align=False,
):
    """
    Outputs
    -------
    failed_ls: list

    players_positions: (t, m, xy), m=xy=2

    players_joints: (t, m, J, xy), m=xy=2
    """
    vid = int(video_path.name.split("_", 1)[0])

    failed_ls = []
    players_positions = []
    players_joints = []

    for frame_num, result in enumerate(inferencer(str(video_path), show=False)):
        keypoints = np.array(
            [person["keypoints"] for person in result["predictions"][0]]
        )  # batch_size=1 (default)
        # keypoints: (m, J, 2)

        # Need at least 2 detected people in the frame.
        # Failed frames are kept as zeros (not dropped) so the clip stays intact.
        # Shuttle coords for these frames are zeroed at collation (Step 3).
        if len(keypoints) < 2:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 2), dtype=float))
            continue

        in_court, pos_normalized = check_pos_in_court(
            keypoints, vid, all_court_info, res_df
        )
        # in_court: (m), pos_normalized: (m, xy), xy=2
        in_court_pid = np.nonzero(in_court)[0]

        # Need exactly 2 players on court. Same retention policy as above.
        if len(in_court_pid) != 2:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 2), dtype=float))
            continue

        bboxes = np.array(
            [person["bbox"][0] for person in result["predictions"][0]]
        )  # batch_size=1 (default)
        # bboxes: (m, 4)

        # Make sure Top player before Bottom player (comparing y-dim)
        if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
            in_court_pid = np.flip(in_court_pid)

        failed_ls.append(False)
        players_positions.append(pos_normalized[in_court_pid])
        players_joints.append(
            normalize_joints(
                arr=keypoints[in_court_pid],
                bbox=bboxes[in_court_pid],
                v_height=res_df.loc[vid, "height"] if normalized_by_v_height else None,
                center_align=center_align,
            )
        )

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xy)

    return failed_ls, players_positions, players_joints


def detect_players_3d(
    inferencer_2d: MMPoseInferencer,
    # inferencer_3d: MMPoseInferencer,
    video_path: Path,
    all_court_info: dict,
    res_df: pd.DataFrame,
    J=17,
):
    """
    Outputs
    -------
    failed_ls: list

    players_positions: (t, m, xy), m=xy=2

    players_joints: (t, m, J, xy), m=xy=2
    """
    vid = int(video_path.name.split("_", 1)[0])

    failed_ls = []
    players_positions = []
    players_joints = []

    gen_2d = inferencer_2d(str(video_path), show=False)
    # WARNING: intentionally instantiated per-call, NOT per-loop-iteration in the caller.
    # The original author found that passing inferencer_3d as a parameter (the way
    # inferencer_2d is passed) triggers an MMPose bug. The commented-out parameter
    # on line ~300 and the commented-out caller on line ~588 are evidence of this.
    # This DOES reload model weights from disk for every clip, which is slow.
    # If MMPose fixes the bug upstream, hoist this into prepare_3d_dataset_npy_from_raw_video
    # and pass it in like inferencer_2d to avoid the repeated load.
    inferencer_3d = MMPoseInferencer(pose3d="human3d")
    gen_3d = inferencer_3d(str(video_path), show=False)

    for frame_num, (result_2d, result_3d) in enumerate(zip(gen_2d, gen_3d)):
        keypoints_2d = np.array(
            [
                person["keypoints"] for person in result_2d["predictions"][0]
            ]  # batch_size=1 (default)
        )
        # keypoints_2d: (m, J, 2)

        keypoints_3d = np.array(
            [
                person["keypoints"] for person in result_3d["predictions"][0]
            ]  # batch_size=1 (default)
        )
        # keypoints_3d: (m, J, 3)

        # Need at least 2 detected people in the frame.
        if len(keypoints_2d) < 2:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 3), dtype=float))
            continue

        in_court, pos_normalized = check_pos_in_court(
            keypoints_2d, vid, all_court_info, res_df
        )
        # in_court: (m), pos_normalized: (m, xy), xy=2
        in_court_pid = np.nonzero(in_court)[0]

        # Need exactly 2 players on court.
        if len(in_court_pid) != 2:
            failed_ls.append(True)
            players_positions.append(np.zeros((2, 2), dtype=float))
            players_joints.append(np.zeros((2, J, 3), dtype=float))
            continue

        # Make sure Top player before Bottom player (comparing y-dim)
        if pos_normalized[in_court_pid[0], 1] > pos_normalized[in_court_pid[1], 1]:
            in_court_pid = np.flip(in_court_pid)

        failed_ls.append(False)
        players_positions.append(pos_normalized[in_court_pid])
        players_joints.append(keypoints_3d[in_court_pid])

    players_positions = np.stack(players_positions)
    # players_positions: (t, m, xy)
    players_joints = np.stack(players_joints)
    # players_joints: (t, m, J, xyz)

    return failed_ls, players_positions, players_joints


def detect_shuttlecock_by_TrackNetV3_with_attention(
    cur_i: int,
    total_tasks: int,
    video_path: Path,
    save_dir: Path,
    model_folder: Path = None,
):
    """TrackNetV3 (using attention).

    https://github.com/alenzenx/TrackNetV3

    :param cur_i: Current task index (for progress printing).
    :param total_tasks: Total number of tasks (for progress printing).
    :param video_path: Path to the clip .mp4 file.
    :param save_dir: Directory to save the shuttle detection CSV.
    :param model_folder: Path to the cloned TrackNetV3 repository.
    :raises ValueError: If model_folder is None.
    """
    if model_folder is None:
        raise ValueError("model_folder is required for shuttle detection.")
    process_args = [
        "python",
        str(model_folder / "predict.py").replace("\\", "/"),
        "--video_file",
        str(video_path).replace("\\", "/"),
        "--tracknet_file",
        str(model_folder / "ckpts" / "TrackNet_best.pt").replace("\\", "/"),
        "--save_dir",
        str(save_dir).replace("\\", "/"),
    ]
    r = subprocess.run(process_args)
    assert r.returncode == 0, "Subprocess failed!"

    type_path = video_path.parent
    set_name = type_path.parent.name
    print(
        f"Shuttlecock detection ({cur_i}/{total_tasks}): {set_name}/{type_path.name}/{video_path.name} done!"
    )


def detect_shuttlecock_by_TrackNetV3_with_rectification(
    cur_i: int,
    total_tasks: int,
    video_path: Path,
    save_dir: Path,
    model_folder: Path = None,
):
    """TrackNetV3 (with rectification module).

    https://github.com/qaz812345/TrackNetV3

    :param cur_i: Current task index (for progress printing).
    :param total_tasks: Total number of tasks (for progress printing).
    :param video_path: Path to the clip .mp4 file.
    :param save_dir: Directory to save the shuttle detection CSV.
    :param model_folder: Path to the cloned TrackNetV3 repository.
    :raises ValueError: If model_folder is None.
    """
    if model_folder is None:
        raise ValueError("model_folder is required for shuttle detection.")
    process_args = [
        "python",
        str(model_folder / "predict.py").replace("\\", "/"),
        "--video_file",
        str(video_path).replace("\\", "/"),
        "--tracknet_file",
        str(model_folder / "ckpts" / "TrackNet_best.pt").replace("\\", "/"),
        "--inpaintnet_file",
        str(model_folder / "ckpts" / "InpaintNet_best.pt").replace("\\", "/"),
        "--save_dir",
        str(save_dir).replace("\\", "/"),
        "--large_video",
    ]
    r = subprocess.run(process_args)
    assert r.returncode == 0, "Subprocess failed!"

    type_path = video_path.parent
    set_name = type_path.parent.name
    print(
        f"Shuttlecock detection ({cur_i}/{total_tasks}): {set_name}/{type_path.name}/{video_path.name} done!"
    )


def get_shuttle_result(path: Path, v_width, v_height):
    df = pd.read_csv(str(path)).drop_duplicates(
        "Frame"
    )  # for the .csv generated by TrackNetV3 with attention
    df = df.set_index("Frame").drop(columns="Visibility")
    shuttle_camera = df.to_numpy().astype(float)
    # shuttle_camera: (t, 2)
    return normalize_shuttlecock(shuttle_camera, v_width, v_height)


def mk_same_dir_structure(src_dir: Path, target_dir: Path, root=True):
    """The roots can be different. Other subdirectories should be all the same."""
    if root and not target_dir.is_dir():
        target_dir.mkdir()
    for src_sub_dir in src_dir.iterdir():
        if src_sub_dir.is_dir():
            target_sub_dir = target_dir / src_sub_dir.name
            if not target_sub_dir.is_dir():
                target_sub_dir.mkdir()
            mk_same_dir_structure(src_sub_dir, target_sub_dir, root=False)


def prepare_trajectory(
    my_clips_folder: Path,
    model_folder: Path,
    save_shuttle_dir: Path,
):
    """Run TrackNetV3 shuttle trajectory detection on all clips.

    Scans my_clips_folder for .mp4 files and runs TrackNetV3 on each one,
    saving shuttle detection CSVs to save_shuttle_dir. Skips clips that
    already have a corresponding CSV.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param model_folder: Path to cloned TrackNetV3 repository.
    :param save_shuttle_dir: Directory to save shuttle detection CSVs.
    """
    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, video_path in enumerate(all_mp4_paths, start=1):
            shuttle_result_path = save_shuttle_dir / (video_path.stem + "_ball.csv")
            if not shuttle_result_path.exists():
                executor.submit(
                    detect_shuttlecock_by_TrackNetV3_with_attention,
                    i,
                    len(all_mp4_paths),
                    video_path=video_path,
                    save_dir=save_shuttle_dir,
                    model_folder=model_folder,
                )


def prepare_2d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
    joints_normalized_by_v_height=False,
    joints_center_align=False,
):
    """Run MMPose 2D pose estimation on clips and save per-clip .npy files.

    For each clip, detects player keypoints (COCO 17-joint), extracts court
    positions via homography, and normalizes joints. Saves _joints.npy,
    _pos.npy, _failed.npy per clip. Shuttle data is handled separately at
    collation time to keep this step focused on pose estimation only.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param resolution_df: DataFrame with video resolutions, indexed by video ID.
    :param all_court_info: Dict mapping video ID to court info (homography, borders).
    :param joints_normalized_by_v_height: If True, normalize joints by video height
        instead of bounding box diagonal.
    :param joints_center_align: If True, center-align joints within bounding box.
    """
    # Make sure there are folders that can contain .npy files.
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)

    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))

    pose_inferencer = MMPoseInferencer("human")

    pbar = tqdm(range(len(all_mp4_paths)), desc="Yield .npy files", unit="video")
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        set_split_dir = ball_type_dir.parent
        save_branch = str(
            save_root_dir / set_split_dir.name / ball_type_dir.name / video_path.stem
        )

        # Resume check: _failed.npy is saved last, so its existence means all
        # three outputs (_pos, _joints, _failed) are complete for this clip.
        # Shuttle data is intentionally NOT handled here — it is read from the
        # canonical pipeline CSV dir and merged at collation (Step 3), keeping
        # this expensive GPU step focused solely on pose estimation.
        if not Path(save_branch + "_failed.npy").exists():
            # Players detection
            failed_ls, players_positions, joints = detect_players_2d(
                inferencer=pose_inferencer,
                video_path=video_path,
                all_court_info=all_court_info,
                res_df=resolution_df,
                normalized_by_v_height=joints_normalized_by_v_height,
                center_align=joints_center_align,
            )

            np.save(save_branch + "_pos.npy", players_positions)
            # (F, P, xy)
            np.save(save_branch + "_joints.npy", joints)
            # (F, P, J, xy)
            np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))
            # (F,) — True where MMPose failed to detect 2 players; saved last
            # so its presence is a reliable resume marker for all three outputs

        # Free GPU memory between clips to prevent fragmentation over ~33k clips.
        gc.collect()
        torch.cuda.empty_cache()

        pbar.update()
    pbar.close()


def prepare_3d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
):
    """Run MMPose 3D pose estimation on clips and save per-clip .npy files.

    Same as prepare_2d_dataset_npy_from_raw_video but uses 3D keypoints (xyz).
    Shuttle data is handled separately at collation time.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param resolution_df: DataFrame with video resolutions, indexed by video ID.
    :param all_court_info: Dict mapping video ID to court info (homography, borders).
    """
    # Make sure there are folders that can contain .npy files.
    mk_same_dir_structure(src_dir=my_clips_folder, target_dir=save_root_dir)

    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))

    pose_inferencer_2d = MMPoseInferencer("human")
    # pose_inferencer_3d = MMPoseInferencer(pose3d='human3d')

    pbar = tqdm(range(len(all_mp4_paths)), desc="Yield .npy files", unit="video")
    for video_path in all_mp4_paths:
        # Set the save paths.
        ball_type_dir = video_path.parent
        set_split_dir = ball_type_dir.parent
        save_branch = str(
            save_root_dir / set_split_dir.name / ball_type_dir.name / video_path.stem
        )

        # See prepare_2d_dataset_npy_from_raw_video for resume-check rationale.
        if not Path(save_branch + "_failed.npy").exists():
            # Players detection
            failed_ls, players_positions, joints = detect_players_3d(
                inferencer_2d=pose_inferencer_2d,
                # inferencer_3d=pose_inferencer_3d,
                video_path=video_path,
                all_court_info=all_court_info,
                res_df=resolution_df,
            )

            np.save(save_branch + "_pos.npy", players_positions)
            # (F, P, xy)
            np.save(save_branch + "_joints.npy", joints)
            # (F, P, J, xyz)
            np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))
            # (F,) — True where MMPose failed to detect 2 players; saved last

        # Free GPU memory between clips to prevent fragmentation over ~33k clips.
        gc.collect()
        torch.cuda.empty_cache()

        pbar.update()
    pbar.close()


def pad_and_augment_one_npy_video(
    seq_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray,
    bone_pairs: list[int, int],
):
    """Pad to uniform sequence length and compute bone/interpolation augmentations.

    :param seq_len: Target sequence length. Shorter clips are zero-padded; longer
        clips are strided (subsampled) to fit.
    :param joints: Joint keypoints, shape (t, 2, J, d).
    :param pos: Player court positions, shape (t, 2, xy).
    :param shuttle: Shuttle coordinates, shape (t, xy).
    :param bone_pairs: List of (start_joint, end_joint) index pairs for bone computation.
    :return: Tuple of (J_only, JnB_interp, JnB_bone, Jn2B, pos, shuttle, video_len)
        where video_len is the number of real (non-padded) frames.
    """
    joints = joints.astype(np.float32)
    pos = pos.astype(np.float32)
    shuttle = shuttle.astype(np.float32)

    joints, pos, shuttle, new_video_len = make_seq_len_same(
        seq_len, joints, pos, shuttle
    )
    # assert len(shuttle) == seq_len, f'{seq_len}, {len(joints)}, {len(pos)}, {len(shuttle)}'

    joints_interpolated = interpolate_joints(joints, bone_pairs)
    bones = create_bones(joints, bone_pairs)

    JnB_bone = np.concatenate((joints, bones), axis=-2)
    Jn2B = np.concatenate((joints_interpolated, bones), axis=-2)

    return joints, joints_interpolated, JnB_bone, Jn2B, pos, shuttle, new_video_len


def collate_npy(
    root_dir: Path,
    set_name: str,
    seq_len: int,
    save_dir: Path,
    taxonomy: Taxonomy = TAXONOMY_UNE_MERGE_V1,
    shuttle_csv_dir: Path | None = None,
    resolution_df: pd.DataFrame | None = None,
):
    """Collate per-clip .npy files into stacked batch arrays for one split.

    Loads all *_joints.npy, *_pos.npy, *_failed.npy from root_dir/set_name,
    reads shuttle trajectories from the canonical CSV dir, aligns temporal
    dimensions, applies failed-frame masking, pads to uniform seq_len,
    computes bone vectors and interpolations, then saves the stacked arrays
    into save_dir/set_name/.

    :param root_dir: Directory containing train/val/test subdirectories with per-clip .npy files.
    :param set_name: One of 'train', 'val', 'test'.
    :param seq_len: Target sequence length (frames). Clips are padded/strided to this length.
    :param save_dir: Output directory. A set_name/ subdirectory is created inside.
    :param taxonomy: Taxonomy defining the class list for label indexing.
    :param shuttle_csv_dir: Directory containing TrackNetV3 shuttle CSVs
        ({clip}_ball.csv). Required.
    :param resolution_df: DataFrame with video resolutions (width/height), indexed
        by video ID. Required.
    """
    assert set_name in ["train", "val", "test"], "Invalid set_name."
    if shuttle_csv_dir is None:
        raise ValueError("shuttle_csv_dir is required")
    if resolution_df is None:
        raise ValueError("resolution_df is required")

    class_ls = taxonomy.class_list()

    # load .npy branch names
    data_branches = []
    labels = []
    target_dir = root_dir / set_name
    for typ in target_dir.iterdir():
        if not typ.is_dir():
            continue
        shots = sorted([str(s).replace("_pos.npy", "") for s in typ.glob("*_pos.npy")])
        data_branches += shots
        labels.append(np.full(len(shots), class_ls.index(typ.name), dtype=np.int64))
    labels = np.concatenate(labels)

    # load .npy files
    print(f"Load .npy files for {set_name} set ...")
    with ThreadPoolExecutor() as executor:
        tasks1: list[Future] = []
        tasks2: list[Future] = []
        tasks3: list[Future] = []

        for branch in data_branches:
            tasks1.append(executor.submit(np.load, branch + "_joints.npy"))
            tasks2.append(executor.submit(np.load, branch + "_pos.npy"))
            tasks3.append(executor.submit(np.load, branch + "_failed.npy"))

        joints_ls = [t1.result() for t1 in tasks1]
        pos_ls = [t2.result() for t2 in tasks2]
        failed_ls = [t3.result() for t3 in tasks3]
    print("Finish loading.")

    # Load shuttle CSVs from the canonical pipeline CSV dir (ShuttleSet/shuttle_csv/),
    # align temporal dimensions, and apply failed-frame masking.
    #
    # Shuttle data is read here rather than in the pose step because:
    #   - Shuttle CSVs are taxonomy- and split-agnostic physical measurements;
    #     they don't belong under taxonomy-specific directories.
    #   - Decoupling lets the ~1.5-3 day GPU pose job run without needing CSVs
    #     present, and lets collation be re-run cheaply when the taxonomy changes.
    #
    # Temporal alignment: MMPose and TrackNetV3 use different video backends that
    # can disagree by 1-2 frames on the tail of the same .mp4. Truncating to the
    # shorter length preserves frame alignment (both decoders start at frame 0).
    shuttle_ls = []
    for i, branch in enumerate(data_branches):
        clip_stem = Path(branch).name  # e.g. '35_1_10_17'
        csv_path = shuttle_csv_dir / (clip_stem + "_ball.csv")
        vid = int(clip_stem.split("_", 1)[0])
        shuttle = get_shuttle_result(
            path=csv_path,
            v_width=resolution_df.loc[vid, "width"],
            v_height=resolution_df.loc[vid, "height"],
        )
        failed = failed_ls[i]

        min_t = min(len(failed), len(shuttle))
        if min_t < len(failed) or min_t < len(shuttle):
            joints_ls[i] = joints_ls[i][:min_t]
            pos_ls[i] = pos_ls[i][:min_t]
            shuttle = shuttle[:min_t]
            failed = failed[:min_t]

        # Zero shuttle coords on frames where pose detection failed. The clip
        # is still included -- no samples are dropped based on failed frames.
        if np.any(failed):
            shuttle[failed, :] = 0

        shuttle_ls.append(shuttle)

    bone_pairs = get_bone_pairs(skeleton_format="coco")

    # Pad and Create bones and Interpolate
    print("Pad, Create bones and Interpolate ...")
    with ProcessPoolExecutor() as executor:
        tasks: list[Future] = []

        for joints, pos, shuttle in zip(joints_ls, pos_ls, shuttle_ls):
            tasks.append(
                executor.submit(
                    pad_and_augment_one_npy_video,
                    seq_len=seq_len,
                    joints=joints,
                    pos=pos,
                    shuttle=shuttle,
                    bone_pairs=bone_pairs,
                )
            )

        J_ls = []
        JnB_interp_ls = []
        JnB_bone_ls = []
        Jn2B_ls = []
        pos_ls = []
        shuttle_ls = []
        videos_len = []

        for task in tasks:
            J_only, JnB_interp, JnB_bone, Jn2B, pos, shuttle, v_len = task.result()
            J_ls.append(J_only)
            JnB_interp_ls.append(JnB_interp)
            JnB_bone_ls.append(JnB_bone)
            Jn2B_ls.append(Jn2B)
            pos_ls.append(pos)
            shuttle_ls.append(shuttle)
            videos_len.append(v_len)

    J_only = np.stack(J_ls)
    JnB_interp = np.stack(JnB_interp_ls)
    JnB_bone = np.stack(JnB_bone_ls)
    Jn2B = np.stack(Jn2B_ls)
    pos = np.stack(pos_ls)
    shuttle = np.stack(shuttle_ls)
    videos_len = np.stack(videos_len)
    print("Finish padding and augmenting.")

    if not save_dir.is_dir():
        save_dir.mkdir()

    set_dir = save_dir / set_name
    if not set_dir.is_dir():
        set_dir.mkdir()

    np.save(str(set_dir / "J_only.npy"), J_only)
    np.save(str(set_dir / "JnB_interp.npy"), JnB_interp)
    np.save(str(set_dir / "JnB_bone.npy"), JnB_bone)
    np.save(str(set_dir / "Jn2B.npy"), Jn2B)
    np.save(str(set_dir / "pos.npy"), pos)
    np.save(str(set_dir / "shuttle.npy"), shuttle)
    np.save(str(set_dir / "videos_len.npy"), videos_len)
    np.save(str(set_dir / "labels.npy"), labels)
    print("Collation is complete.")


def main():
    """Parse CLI arguments and run the requested pipeline steps.

    Usage (from stroke_classification/ directory):
        python -m preparing_data.prepare_train_on_shuttleset --dry-run
        python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose
        python -m preparing_data.prepare_train_on_shuttleset --tracknet-dir /path/to/TrackNetV3
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ShuttleSet training data in 3 steps:\n"
            "  Step 1: Shuttle trajectory detection (TrackNetV3)\n"
            "  Step 2: 2D/3D pose estimation (MMPose)\n"
            "  Step 3: Collate per-clip .npy files into batch arrays\n"
            "\n"
            "Each step can be skipped independently."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Step control
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip Step 1 (shuttle trajectory detection)",
    )
    parser.add_argument(
        "--skip-pose", action="store_true", help="Skip Step 2 (pose estimation)"
    )
    parser.add_argument(
        "--skip-collate",
        action="store_true",
        help="Skip Step 3 (collation into batch arrays)",
    )

    # Data configuration
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        choices=[30, 100],
        help="Target sequence length in frames (default: 100)",
    )
    parser.add_argument(
        "--taxonomy",
        default=DEFAULT_TAXONOMY,
        choices=list(TAXONOMIES.keys()),
        help=f"Stroke type taxonomy (default: {DEFAULT_TAXONOMY})",
    )
    parser.add_argument(
        "--use-3d-pose",
        action="store_true",
        help="Use 3D pose estimation instead of 2D",
    )

    # Path overrides (only the ones that genuinely vary)
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=CLIPS_OUTPUT_DIR,
        help=f"Clip .mp4 input directory (default: {CLIPS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tracknet-dir",
        type=Path,
        default=None,
        help="Path to TrackNetV3 repo (required for Step 1)",
    )
    parser.add_argument(
        "--shuttle-csv-dir",
        type=Path,
        default=SHUTTLE_CSV_DIR,
        help=f"Directory with TrackNetV3 shuttle CSVs (default: {SHUTTLE_CSV_DIR})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without executing",
    )

    args = parser.parse_args()

    # ---- Resolve taxonomy and derive intermediate paths ----
    taxonomy = TAXONOMIES[args.taxonomy]
    str_3d = "_3d" if args.use_3d_pose else ""
    preparing_root = (
        Path(__file__).resolve().parent / f"ShuttleSet_data_{taxonomy.name}"
    )
    preparing_root.mkdir(parents=True, exist_ok=True)

    if args.seq_len == 30:
        npy_raw_dir = preparing_root / f"dataset{str_3d}_npy"
        npy_collated_dir = preparing_root / f"dataset{str_3d}_npy_collated"
    else:  # 100
        npy_raw_dir = (
            preparing_root / f"dataset{str_3d}_npy_between_2_hits_with_max_limits"
        )
        npy_collated_dir = preparing_root / (
            f"dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100"
        )

    # ---- Dry run ----
    if args.dry_run:
        print("=== DRY RUN (no files will be created) ===\n")
        print(f"  seq_len:          {args.seq_len}")
        print(f"  taxonomy:         {taxonomy.name} ({taxonomy.n_classes} classes)")
        print(f"  use_3d_pose:      {args.use_3d_pose}")
        print(f"  clips_dir:        {args.clips_dir}")
        print(f"  shuttle_csv_dir:  {args.shuttle_csv_dir}")
        print(f"  npy_raw_dir:      {npy_raw_dir}")
        print(f"  npy_collated:     {npy_collated_dir}")
        print(f'  homography:       {SET_INFO_DIR / "homography.csv"}')
        print(f"  resolution:       {RESOLUTION_CSV_PATH}")
        print(f'\n  Step 1 (trajectory): {"SKIP" if args.skip_trajectory else "RUN"}')
        print(f'  Step 2 (pose):       {"SKIP" if args.skip_pose else "RUN"}')
        print(f'  Step 3 (collate):    {"SKIP" if args.skip_collate else "RUN"}')
        print("\n=== End dry run ===")
        return

    # ---- Load homography and resolution data (needed by all steps) ----
    homo_df = pd.read_csv(str(SET_INFO_DIR / "homography.csv")).set_index("id")
    resolution_df = pd.read_csv(str(RESOLUTION_CSV_PATH)).set_index("id")
    all_court_info = {vid: get_court_info(homo_df, vid) for vid in resolution_df.index}

    # ---- Step 1: Shuttle trajectory detection ----
    if not args.skip_trajectory:
        if args.tracknet_dir is None:
            parser.error(
                "--tracknet-dir is required for Step 1 (trajectory detection)."
            )
        print("\n--- Step 1: Shuttle trajectory detection ---")
        args.shuttle_csv_dir.mkdir(parents=True, exist_ok=True)
        prepare_trajectory(
            my_clips_folder=args.clips_dir,
            model_folder=args.tracknet_dir,
            save_shuttle_dir=args.shuttle_csv_dir,
        )
    else:
        print("Step 1: Skipped (--skip-trajectory)")

    # ---- Step 2: Pose estimation ----
    if not args.skip_pose:
        print("\n--- Step 2: Pose estimation ---")
        if args.use_3d_pose:
            prepare_3d_dataset_npy_from_raw_video(
                my_clips_folder=args.clips_dir,
                save_root_dir=npy_raw_dir,
                resolution_df=resolution_df,
                all_court_info=all_court_info,
            )
        else:
            prepare_2d_dataset_npy_from_raw_video(
                my_clips_folder=args.clips_dir,
                save_root_dir=npy_raw_dir,
                resolution_df=resolution_df,
                all_court_info=all_court_info,
                joints_normalized_by_v_height=False,
                joints_center_align=True,
            )
    else:
        print("Step 2: Skipped (--skip-pose)")

    # ---- Step 3: Collation ----
    if not args.skip_collate:
        print("\n--- Step 3: Collate .npy files ---")
        for set_name in ["train", "val", "test"]:
            collate_npy(
                root_dir=npy_raw_dir,
                set_name=set_name,
                seq_len=args.seq_len,
                save_dir=npy_collated_dir,
                taxonomy=taxonomy,
                shuttle_csv_dir=args.shuttle_csv_dir,
                resolution_df=resolution_df,
            )
    else:
        print("Step 3: Skipped (--skip-collate)")

    print("\nAll requested steps complete.")


if __name__ == "__main__":
    main()
