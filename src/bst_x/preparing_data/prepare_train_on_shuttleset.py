"""Prepare ShuttleSet training data: shuttle detection, pose estimation, and collation.

Bridges the gap between the pipeline's clip output and BST's expected input format.
Three steps, each independently skippable:
  Step 1: Shuttle trajectory detection via TrackNetV3
  Step 2: 2D/3D player pose estimation via MMPose + court projection
  Step 3: Collate per-clip .npy files into batch-ready arrays

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src/bst_x \\
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
    derive_npy_collated_dir_basename,
    label_for_row,
    resolve_taxonomy,
)
from pipeline.data_access import env_path, env_path_or_none, load_repo_dotenv


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

    Signature defaults preserved verbatim from BST upstream for canonical
    accuracy. The CLI invocation in ``main()`` below overrides
    ``center_align`` to True (matches BST upstream's own CLI default;
    committed ShuttleSet extracts were produced with this override).
    ``v_height=None`` is canonical at both layers: the signature default
    and the CLI call agree, so no flip happens there.
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


def _prepare_dataset_from_raw_video(
    my_clips_folder: Path,
    save_root_dir: Path,
    detect_fn,
    detect_kwargs: dict,
):
    """Shared per-clip MMPose iteration for 2D and 3D pose extraction.

    For each clip in ``my_clips_folder``: skip if ``{stem}_failed.npy`` already
    exists (resume marker), otherwise call ``detect_fn(**detect_kwargs, video_path=...)``
    to get ``(failed_ls, players_positions, joints)``, save the three .npy
    outputs, and free GPU memory.

    The resume marker is `_failed.npy` because it is saved last; its presence
    means all three outputs are complete for the clip. Shuttle data is read
    from the canonical pipeline CSV dir at collation (Step 3); this expensive
    GPU step stays focused solely on pose estimation.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param detect_fn: ``detect_players_2d`` or ``detect_players_3d``.
    :param detect_kwargs: keyword arguments threaded through to ``detect_fn``
        besides ``video_path``.
    """
    # Flat layout: per-clip files sit alongside each other under save_root_dir.
    # Split + label come from clips_master.csv at collation time (Step 3).
    save_root_dir.mkdir(parents=True, exist_ok=True)

    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))

    pbar = tqdm(range(len(all_mp4_paths)), desc="Yield .npy files", unit="video")
    for video_path in all_mp4_paths:
        save_branch = str(save_root_dir / video_path.stem)

        if not Path(save_branch + "_failed.npy").exists():
            failed_ls, players_positions, joints = detect_fn(
                video_path=video_path,
                **detect_kwargs,
            )

            np.save(save_branch + "_pos.npy", players_positions)
            np.save(save_branch + "_joints.npy", joints)
            np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))

            # Free GPU memory after inference to prevent fragmentation over
            # ~33k clips. Skips don't allocate on GPU, so no cleanup needed
            # in that branch -- keeps resume-path iterations cheap (~1ms vs
            # ~100ms when these ran unconditionally).
            gc.collect()
            torch.cuda.empty_cache()

        pbar.update()
    pbar.close()


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
    positions via homography, and normalizes joints. Saves _joints.npy
    ((F, P, J, xy)), _pos.npy ((F, P, xy)), _failed.npy ((F,)) per clip.
    Shuttle data is handled separately at collation time.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param resolution_df: DataFrame with video resolutions, indexed by video ID.
    :param all_court_info: Dict mapping video ID to court info (homography, borders).
    :param joints_normalized_by_v_height: If True, normalize joints by video height
        instead of bounding box diagonal.
    :param joints_center_align: If True, center-align joints within bounding box.
    """
    pose_inferencer = MMPoseInferencer("human")

    _prepare_dataset_from_raw_video(
        my_clips_folder=my_clips_folder,
        save_root_dir=save_root_dir,
        detect_fn=detect_players_2d,
        detect_kwargs={
            "inferencer": pose_inferencer,
            "all_court_info": all_court_info,
            "res_df": resolution_df,
            "normalized_by_v_height": joints_normalized_by_v_height,
            "center_align": joints_center_align,
        },
    )


def prepare_3d_dataset_npy_from_raw_video(
    my_clips_folder: Path,
    save_root_dir: Path,
    resolution_df: pd.DataFrame,
    all_court_info: dict,
):
    """Run MMPose 3D pose estimation on clips and save per-clip .npy files.

    Same as prepare_2d_dataset_npy_from_raw_video but uses 3D keypoints (xyz).
    Saves _joints.npy ((F, P, J, xyz)), _pos.npy ((F, P, xy)), _failed.npy
    ((F,)). Shuttle data is handled separately at collation time.

    :param my_clips_folder: Directory containing clip .mp4 files (searched recursively).
    :param save_root_dir: Output directory for per-clip .npy files.
    :param resolution_df: DataFrame with video resolutions, indexed by video ID.
    :param all_court_info: Dict mapping video ID to court info (homography, borders).
    """
    pose_inferencer_2d = MMPoseInferencer("human")
    # pose_inferencer_3d = MMPoseInferencer(pose3d='human3d')

    _prepare_dataset_from_raw_video(
        my_clips_folder=my_clips_folder,
        save_root_dir=save_root_dir,
        detect_fn=detect_players_3d,
        detect_kwargs={
            "inferencer_2d": pose_inferencer_2d,
            # "inferencer_3d": pose_inferencer_3d,
            "all_court_info": all_court_info,
            "res_df": resolution_df,
        },
    )


VALID_POSE_STYLES: tuple[str, ...] = ("J_only", "JnB_interp", "JnB_bone", "Jn2B")


def pad_and_augment_one_npy_video(
    seq_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray,
    bone_pairs: list[int, int],
    pose_styles: frozenset[str] = frozenset({"JnB_bone"}),
):
    """Pad to uniform sequence length and compute requested pose augmentations.

    Only the pose styles in ``pose_styles`` are computed and returned; the
    derived arrays (``create_bones``, ``interpolate_joints``) are skipped if
    nothing downstream needs them.

    :param seq_len: Target sequence length. Shorter clips are zero-padded; longer
        clips are strided (subsampled) to fit.
    :param joints: Joint keypoints, shape (t, 2, J, d).
    :param pos: Player court positions, shape (t, 2, xy).
    :param shuttle: Shuttle coordinates, shape (t, xy).
    :param bone_pairs: List of (start_joint, end_joint) index pairs for bone computation.
    :param pose_styles: Which pose representations to compute. Subset of
        ``VALID_POSE_STYLES``. Defaults to ``{'JnB_bone'}`` (the only style
        BST training has ever used in this tracker).
    :return: Tuple of (pose_dict, pos, shuttle, video_len) where pose_dict maps
        each requested style name to its (t, 2, K, d) array and video_len is
        the number of real (non-padded) frames.
    """
    joints = joints.astype(np.float32)
    pos = pos.astype(np.float32)
    shuttle = shuttle.astype(np.float32)

    joints, pos, shuttle, new_video_len = make_seq_len_same(
        seq_len, joints, pos, shuttle
    )

    pose_dict: dict[str, np.ndarray] = {}

    if "J_only" in pose_styles:
        pose_dict["J_only"] = joints

    # bones is needed for JnB_bone and Jn2B; interpolated joints for JnB_interp and Jn2B.
    needs_bones = bool(pose_styles & {"JnB_bone", "Jn2B"})
    needs_interp = bool(pose_styles & {"JnB_interp", "Jn2B"})
    bones = create_bones(joints, bone_pairs) if needs_bones else None
    joints_interpolated = (
        interpolate_joints(joints, bone_pairs) if needs_interp else None
    )

    if "JnB_bone" in pose_styles:
        pose_dict["JnB_bone"] = np.concatenate((joints, bones), axis=-2)
    if "JnB_interp" in pose_styles:
        pose_dict["JnB_interp"] = joints_interpolated
    if "Jn2B" in pose_styles:
        pose_dict["Jn2B"] = np.concatenate((joints_interpolated, bones), axis=-2)

    return pose_dict, pos, shuttle, new_video_len


def collate_npy(
    root_dir: Path,
    set_name: str,
    seq_len: int,
    save_dir: Path,
    clips_csv: Path,
    split_column: str,
    taxonomy: Taxonomy,
    shuttle_csv_dir: Path | None = None,
    resolution_df: pd.DataFrame | None = None,
    pose_styles: frozenset[str] = frozenset({"JnB_bone"}),
    unknown_root_dir: Path | None = None,
):
    """Collate per-clip .npy files into stacked batch arrays for one split.

    Reads split assignment and label from the master clips CSV, resolves
    per-clip files at FLAT path ``{root_dir}/{clip_stem}_*.npy``, reads
    shuttle trajectories from the canonical CSV dir, aligns temporal
    dimensions, applies failed-frame masking, pads to uniform seq_len,
    computes bone vectors and interpolations, then saves the stacked arrays
    into ``save_dir/set_name/``. A row-aligned ``clip_stems.npy`` sidecar
    is saved alongside ``labels.npy`` so downstream consumers can join row
    index -> stem directly without re-deriving the CSV filter.

    :param root_dir: FLAT per-clip dir containing
        ``{clip_stem}_{joints,pos,failed}.npy`` for every clip.
    :param set_name: One of 'train', 'val', 'test'.
    :param seq_len: Target sequence length (frames). Clips are padded/strided to this length.
    :param save_dir: Output directory. A set_name/ subdirectory is created inside.
    :param clips_csv: Master clips CSV (one row per clip) providing split
        assignment (``split_column``), ``raw_type_en``, ``player_side``,
        ``clip_stem``.
    :param split_column: Column in ``clips_csv`` to use for split assignment,
        e.g. ``'split_bst_baseline'`` or ``'split_v2'``.
    :param taxonomy: Taxonomy. ``label_for_row`` drives per-row class index +
        the unknown-filter rule (via ``excluded_base_stroke_types``); no
        separate drop_unknown flag any more.
    :param shuttle_csv_dir: Directory containing TrackNetV3 shuttle CSVs
        ({clip}_ball.csv). Required.
    :param resolution_df: DataFrame with video resolutions (width/height), indexed
        by video ID. Required.
    :param unknown_root_dir: Optional FLAT per-clip dir for rows whose
        ``raw_type_en == 'unknown'``. When set, unknown rows resolve their
        per-clip files from this dir instead of ``root_dir``. Used to point
        the bst_25 / une_v1_15 collations at the sibling
        ``ShuttleSet_keypoints_clean_sticky_anchor_unknown/`` extract. Must
        be None when the taxonomy has ``'unknown'`` in
        ``excluded_base_stroke_types`` (those rows get dropped anyway).
    """
    assert set_name in ["train", "val", "test"], "Invalid set_name."
    if shuttle_csv_dir is None:
        raise ValueError("shuttle_csv_dir is required")
    if resolution_df is None:
        raise ValueError("resolution_df is required")
    if unknown_root_dir is not None and 'unknown' in taxonomy.excluded_base_stroke_types:
        raise ValueError(
            f"unknown_root_dir set but taxonomy {taxonomy.name!r} excludes "
            f"unknown rows (excluded_base_stroke_types contains 'unknown'). "
            f"Either drop unknown_root_dir or pick a taxonomy that retains unknown."
        )
    if taxonomy.has_unknown and unknown_root_dir is None:
        raise ValueError(
            f"taxonomy {taxonomy.name!r} retains unknown in its class list, "
            f"but unknown_root_dir is None. The 1,278 unknown clips don't have "
            f"per-clip files under the canonical extract; they need to come "
            f"from the sibling _unknown extract. Pass unknown_root_dir=<that "
            f"sibling dir>, OR pick a taxonomy whose excluded_base_stroke_types "
            f"includes 'unknown' (e.g. {taxonomy.name.replace('_25', '_24').replace('_15', '_14')!r})."
        )

    # Filter the master CSV down to this split.
    clips_df = pd.read_csv(clips_csv)
    if split_column not in clips_df.columns:
        raise KeyError(
            f"split_column {split_column!r} not in clips_csv columns: "
            f"{list(clips_df.columns)}"
        )
    clips_df = clips_df[clips_df[split_column] == set_name].copy()

    # Per-row label derivation + unknown-row routing. label_for_row applies
    # taxonomy.excluded_base_stroke_types first (returns None -> drop the
    # row), then merge_map, then side-prefixing per taxonomy.has_sides. The
    # unknown_root_dir branch pulls per-clip files from the sibling extract
    # for rows that survived (i.e. taxonomies that retain unknown).
    data_branches: list[str] = []
    labels_ls: list[int] = []
    stems_ls: list[str] = []
    missing = 0
    for raw_type, side, stem in zip(
        clips_df["raw_type_en"],
        clips_df["player_side"],
        clips_df["clip_stem"],
    ):
        try:
            idx = label_for_row(taxonomy, raw_type, side)
        except ValueError as e:
            # Add clip stem context to the descriptive error label_for_row
            # already raises. Preserves the chain via `from e`.
            raise ValueError(
                f"label derivation failed for clip {stem!r}: {e}"
            ) from e
        if idx is None:
            continue  # filtered out via excluded_base_stroke_types
        chosen_root = (
            unknown_root_dir
            if (raw_type == "unknown" and unknown_root_dir is not None)
            else root_dir
        )
        branch = str(chosen_root / stem)
        # Skip clips whose flat per-clip files are absent. verify_flatten.py
        # should have ruled this out before the originals were deleted, but
        # the check is cheap and prevents a confusing ENOENT mid-collation.
        if not Path(branch + "_pos.npy").exists():
            missing += 1
            continue
        data_branches.append(branch)
        labels_ls.append(idx)
        stems_ls.append(stem)

    if missing:
        unknown_hint = (
            f" (or {unknown_root_dir} for unknown rows)"
            if unknown_root_dir is not None else ""
        )
        print(
            f"  [{set_name}] WARNING: {missing} clips in master CSV had no "
            f"flat per-clip files under {root_dir}{unknown_hint}; skipped."
        )
    labels = np.asarray(labels_ls, dtype=np.int64)
    clip_stems_arr = np.asarray(stems_ls, dtype=object)
    print(
        f"  [{set_name}] {len(data_branches)} clips after filter "
        f"(taxonomy={taxonomy.name}, split_column={split_column})."
    )

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

    # Load shuttle CSVs from the canonical pipeline CSV dir (data/shuttleset/shuttle_csv/),
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

        # Pose-fail frames no longer wipe shuttle. Pose tells you where a
        # player is in the court; shuttle tells you where the bird is. If
        # one player collapses, the bird's coord still has meaning.
        # Full rationale: docs/architecture_notes/frame_zeroing.md.

        shuttle_ls.append(shuttle)

    bad_styles = set(pose_styles) - set(VALID_POSE_STYLES)
    if bad_styles:
        raise ValueError(
            f"Unknown pose_styles {sorted(bad_styles)!r}; "
            f"valid choices: {VALID_POSE_STYLES}"
        )

    bone_pairs = get_bone_pairs(skeleton_format="coco")

    # Pad and compute only the requested pose representations.
    print(f"Pad, Create bones and Interpolate (pose_styles={sorted(pose_styles)}) ...")
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
                    pose_styles=pose_styles,
                )
            )

        pose_ls: dict[str, list[np.ndarray]] = {k: [] for k in pose_styles}
        pos_ls = []
        shuttle_ls = []
        videos_len = []

        for task in tasks:
            pose_dict, pos, shuttle, v_len = task.result()
            for k, arr in pose_dict.items():
                pose_ls[k].append(arr)
            pos_ls.append(pos)
            shuttle_ls.append(shuttle)
            videos_len.append(v_len)

    pos = np.stack(pos_ls)
    shuttle = np.stack(shuttle_ls)
    videos_len = np.stack(videos_len)
    print("Finish padding and augmenting.")

    if not save_dir.is_dir():
        save_dir.mkdir()

    set_dir = save_dir / set_name
    if not set_dir.is_dir():
        set_dir.mkdir()

    for k, arrs in pose_ls.items():
        np.save(str(set_dir / f"{k}.npy"), np.stack(arrs))
    np.save(str(set_dir / "pos.npy"), pos)
    np.save(str(set_dir / "shuttle.npy"), shuttle)
    np.save(str(set_dir / "videos_len.npy"), videos_len)
    np.save(str(set_dir / "labels.npy"), labels)
    # Row-aligned clip stems sidecar so the post-hoc FE-JSON converter can
    # join row index -> stem without re-deriving the CSV filter.
    np.save(str(set_dir / "clip_stems.npy"), clip_stems_arr, allow_pickle=True)
    print("Collation is complete.")


def main():
    """Parse CLI arguments and run the requested pipeline steps.

    Usage (from the repo root, with both package roots on PYTHONPATH):
        PYTHONPATH=src/bst_x \\
            python -m preparing_data.prepare_train_on_shuttleset --dry-run
        PYTHONPATH=src/bst_x \\
            python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose
        PYTHONPATH=src/bst_x \\
            python -m preparing_data.prepare_train_on_shuttleset --tracknet-dir /path/to/TrackNetV3
    """
    # Populate os.environ from <repo>/.env so argparse defaults below can
    # read BST_* vars. Same pattern as pipeline.data_access.
    load_repo_dotenv()

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
        default="une_v1_14",
        choices=list(TAXONOMIES.keys()),
        help="Stroke type taxonomy (default: une_v1_14). Drives label_for_row "
             "per-row index + the unknown-filter rule via "
             "excluded_base_stroke_types.",
    )
    parser.add_argument(
        "--use-3d-pose",
        action="store_true",
        help="Use 3D pose estimation instead of 2D",
    )

    # Path overrides (only the ones that genuinely vary)
    # Defaults read from .env (loaded above via load_repo_dotenv) when the
    # corresponding BST_* env var is set; otherwise fall back to the repo-
    # rooted constants from pipeline/config.py. Same pattern as data_access.
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=env_path('BST_X_CLIPS_DIR', CLIPS_OUTPUT_DIR),
        help=f"Clip .mp4 input directory (default: BST_X_CLIPS_DIR or {CLIPS_OUTPUT_DIR})",
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
        default=env_path('BST_X_SHUTTLE_CSV_DIR', SHUTTLE_CSV_DIR),
        help=f"Directory with TrackNetV3 shuttle CSVs (default: BST_X_SHUTTLE_CSV_DIR or {SHUTTLE_CSV_DIR})",
    )

    # Step 3 (collation) configuration: drives split + label assignment from
    # the master clips CSV instead of the on-disk folder layout. The flat
    # per-clip dir holds {clip_stem}_*.npy files shared across all ablations;
    # the collated dir is per-cell -- the parent dir carries the taxonomy and
    # the basename carries split + generation tag, so cells that share a
    # taxonomy but differ by split don't collide.
    parser.add_argument(
        "--clips-csv",
        type=Path,
        default=env_path(
            'BST_X_CLIPS_CSV',
            Path(__file__).resolve().parents[3] / "notebooks" / "clips_master.csv",
        ),
        help="Master clips CSV with split + label per clip "
             "(default: BST_X_CLIPS_CSV or <repo>/notebooks/clips_master.csv).",
    )
    parser.add_argument(
        "--split-column",
        default="split_bst_baseline",
        choices=["split_bst_baseline", "split_v2"],
        help="Column in clips_csv giving train/val/test assignment "
             "(default: split_bst_baseline).",
    )
    parser.add_argument(
        "--collation-id",
        required=True,
        help="Required collation generation tag. Suffixes the collated output "
             "dir (npy_[3d_][seq{N}_]{split}_{collation_id}) so re-collations of "
             "the same taxonomy + split coexist. Common values: "
             "'taxon_pinned_w_preds', 'wipe_drop'. A training-time ablation tag "
             "is separate and lives in the run manifest, not the collation path.",
    )
    parser.add_argument(
        "--clip-npy-dir",
        type=Path,
        default=env_path_or_none('BST_X_MMPOSE_NPY_DIR'),
        help="FLAT per-clip dir (Step 2 writer + Step 3 reader). Default reads "
             "BST_X_MMPOSE_NPY_DIR; if unset, falls back to the per-taxonomy "
             "preparing_root + 'dataset[_3d]_npy_between_2_hits_with_max_limits_flat'.",
    )
    parser.add_argument(
        "--unknown-clip-npy-dir",
        type=Path,
        default=None,
        help="Optional FLAT per-clip dir for rows with raw_type_en=='unknown'. "
             "Routes unknown rows through this dir while everything else comes "
             "from --clip-npy-dir. Used to point bst_25 / une_v1_15 collations "
             "at the sibling _unknown extract. Must NOT be set when the "
             "taxonomy excludes unknown.",
    )
    parser.add_argument(
        "--pose-styles",
        default="JnB_bone",
        help="Comma-separated pose representations to compute and save at "
             "Step 3. Default 'JnB_bone' (the only style BST training has "
             f"used in this tracker). Valid choices: {','.join(VALID_POSE_STYLES)}.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without executing",
    )

    args = parser.parse_args()

    # ---- Resolve taxonomy and derive intermediate paths ----
    taxonomy = resolve_taxonomy(args.taxonomy)
    if (
        args.unknown_clip_npy_dir is not None
        and 'unknown' in taxonomy.excluded_base_stroke_types
    ):
        parser.error(
            f"--unknown-clip-npy-dir is set but taxonomy {taxonomy.name!r} "
            f"excludes unknown rows (excluded_base_stroke_types={sorted(taxonomy.excluded_base_stroke_types)}). "
            f"Either drop the flag or pick a taxonomy that retains unknown."
        )
    if taxonomy.has_unknown and args.unknown_clip_npy_dir is None:
        parser.error(
            f"--taxonomy {taxonomy.name!r} retains unknown in its class list, "
            f"but --unknown-clip-npy-dir is not set. The 1,278 unknown clips "
            f"don't have per-clip files under the canonical extract; they need "
            f"to come from the sibling _unknown extract. Pass "
            f"--unknown-clip-npy-dir <that sibling dir>, OR pick a taxonomy "
            f"whose excluded_base_stroke_types includes 'unknown'."
        )
    str_3d = "_3d" if args.use_3d_pose else ""
    # Preparing root uses the resolved canonical taxonomy name. Legacy
    # aliases (e.g. 'une_merge_v1_nosides') resolve to canonical here so the
    # on-disk dir lands under the new naming. The ROOT under which all
    # ShuttleSet_data_<tax>/ dirs live reads BST_X_COLLATED_DATA_ROOT when
    # set (matches the FE serving contract in frontend_integration_guide.md);
    # otherwise falls back to the in-repo preparing_data/ convention for
    # local dev where /scratch isn't available.
    collated_data_root = env_path_or_none('BST_X_COLLATED_DATA_ROOT')
    if collated_data_root is not None:
        preparing_root = collated_data_root / f"ShuttleSet_data_{taxonomy.name}"
    else:
        preparing_root = (
            Path(__file__).resolve().parent / f"ShuttleSet_data_{taxonomy.name}"
        )
    preparing_root.mkdir(parents=True, exist_ok=True)

    # Parse + validate --pose-styles.
    pose_styles = frozenset(s.strip() for s in args.pose_styles.split(",") if s.strip())
    bad_styles = pose_styles - set(VALID_POSE_STYLES)
    if bad_styles:
        parser.error(
            f"Unknown --pose-styles entries {sorted(bad_styles)!r}; "
            f"valid: {','.join(VALID_POSE_STYLES)}"
        )

    # Collated dir naming via shared helper (mirrored on the bst_x_train.py
    # reader side); see ``pipeline.config.derive_npy_collated_dir_basename``.
    npy_collated_dir = preparing_root / derive_npy_collated_dir_basename(
        use_3d_pose=args.use_3d_pose,
        seq_len=args.seq_len,
        split_column=args.split_column,
        collation_id=args.collation_id,
    )
    if args.seq_len == 30:
        default_flat_dir = preparing_root / f"dataset{str_3d}_npy_flat"
    else:  # 100
        default_flat_dir = (
            preparing_root / f"dataset{str_3d}_npy_between_2_hits_with_max_limits_flat"
        )

    # FLAT per-clip dir. Step 2 writes per-clip files here ({clip_stem}_*.npy),
    # Step 3 reads from here. Split + label come from clips_master.csv at
    # collation time -- the layout is taxonomy- and split-independent.
    flat_clip_npy_dir = args.clip_npy_dir or default_flat_dir

    # ---- Dry run ----
    if args.dry_run:
        print("=== DRY RUN (no files will be created) ===\n")
        print(f"  seq_len:          {args.seq_len}")
        print(f"  taxonomy:         {taxonomy.name} ({taxonomy.n_classes} classes)")
        print(f"  use_3d_pose:      {args.use_3d_pose}")
        print(f"  clips_dir:        {args.clips_dir}")
        print(f"  shuttle_csv_dir:  {args.shuttle_csv_dir}")
        print(f"  flat_clip_npy:    {flat_clip_npy_dir}  (Step 2 writer + Step 3 reader)")
        print(f"  npy_collated:     {npy_collated_dir}")
        print(f"  clips_csv:        {args.clips_csv}")
        print(f"  split_column:     {args.split_column}")
        print(f"  excluded_raw:     {sorted(taxonomy.excluded_base_stroke_types)}")
        print(f"  unknown_clip_dir: {args.unknown_clip_npy_dir}")
        print(f"  collation_id:     {args.collation_id}")
        print(f"  pose_styles:      {sorted(pose_styles)}")
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
                save_root_dir=flat_clip_npy_dir,
                resolution_df=resolution_df,
                all_court_info=all_court_info,
            )
        else:
            prepare_2d_dataset_npy_from_raw_video(
                my_clips_folder=args.clips_dir,
                save_root_dir=flat_clip_npy_dir,
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
        if not args.clips_csv.exists():
            parser.error(f"--clips-csv path does not exist: {args.clips_csv}")
        if not flat_clip_npy_dir.exists():
            parser.error(
                f"flat per-clip dir does not exist: {flat_clip_npy_dir}\n"
                "  Run Step 2 first (drop --skip-pose) or pass --clip-npy-dir."
            )
        for set_name in ["train", "val", "test"]:
            collate_npy(
                root_dir=flat_clip_npy_dir,
                set_name=set_name,
                seq_len=args.seq_len,
                save_dir=npy_collated_dir,
                clips_csv=args.clips_csv,
                split_column=args.split_column,
                pose_styles=pose_styles,
                taxonomy=taxonomy,
                unknown_root_dir=args.unknown_clip_npy_dir,
                shuttle_csv_dir=args.shuttle_csv_dir,
                resolution_df=resolution_df,
            )
    else:
        print("Step 3: Skipped (--skip-collate)")

    print("\nAll requested steps complete.")


if __name__ == "__main__":
    main()
