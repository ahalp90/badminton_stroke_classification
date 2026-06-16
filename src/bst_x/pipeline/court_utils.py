"""Optional court projection utilities for ShuttleSet.

Provides homography-based coordinate transformation from camera pixel space
to normalised court coordinates [0, 1]. Available for architectures that need
court-relative features -- not a required pipeline step.

Copied (not moved) from prepare_train_on_shuttleset.py so that downstream
consumers can import court utilities without pulling in the full BST preparation
pipeline. BST's own code continues to use prepare_train_on_shuttleset.py directly.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import SET_INFO_DIR, HOMOGRAPHY_RESOLUTION


def get_H(homography_info: pd.Series) -> np.ndarray:
    """Parse the 3x3 homography matrix from a homography.csv row.

    :param homography_info: A row from homography.csv as a Series.
    :return: 3x3 numpy array.
    """
    h_str: str = homography_info['homography_matrix']
    # Strip brackets/commas and let NumPy parse the numeric sequence
    clean_str = h_str.replace('[', '').replace(']', '').replace(',', ' ')
    return np.fromstring(clean_str, sep=' ').reshape((3, 3))


def get_corner_camera(homography_info: pd.Series) -> np.ndarray:
    """Extract the 4 court corner coordinates (2, 4) from a homography.csv row.

    :param homography_info: A row from homography.csv as a Series.
    :return: (2, 4) numpy array of corner x, y coordinates.
    """
    corner_camera = homography_info.loc['upleft_x':'downright_y']
    return corner_camera.to_numpy(dtype=float).reshape((2, 4))


def convert_homogeneous(arr: np.ndarray) -> np.ndarray:
    """Convert (2, N) array to homogeneous coordinates (3, N).

    :param arr: (2, N) array of 2D coordinates.
    :return: (3, N) array with a row of ones appended.
    """
    return np.concatenate((arr, np.full((1, arr.shape[-1]), 1.0)), axis=0)


def scale_pos_by_resolution(
    arr: np.ndarray, width: float, height: float,
) -> np.ndarray:
    """Scale (2, N) or (3, N) coordinates from source resolution to homography resolution.

    The homography matrices in homography.csv were computed at
    HOMOGRAPHY_RESOLUTION (from config). If your video has a different
    resolution, coordinates must be scaled before applying the homography.

    :param arr: (2, N) or (3, N) coordinate array.
    :param width: Source video width in pixels.
    :param height: Source video height in pixels.
    :return: Scaled coordinate array (same shape as input).
    """
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    new_arr = arr.copy()
    new_arr[0, :] *= aim_w / width
    new_arr[1, :] *= aim_h / height
    return new_arr


def project(H: np.ndarray, P_prime: np.ndarray) -> np.ndarray:
    """Apply homography: transform (3, N) homogeneous coords to (2, N) court coords.

    :param H: 3x3 homography matrix.
    :param P_prime: (3, N) homogeneous coordinates in camera space.
    :return: (2, N) projected coordinates in court space.
    """
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]
    return P


def get_court_info(homo_df: pd.DataFrame, vid: int) -> dict:
    """Get homography matrix and court boundary coordinates for a video.

    :param homo_df: DataFrame from homography.csv, indexed by video ID.
    :param vid: Video ID.
    :return: Dict with keys 'H' (3x3 matrix), 'border_L', 'border_R',
        'border_U', 'border_D' (court boundaries in court coordinate space).
    """
    homography_info = homo_df.loc[vid]
    H = get_H(homography_info)
    corner_camera = get_corner_camera(homography_info)
    corner_camera = convert_homogeneous(corner_camera)
    corner_court = project(H, corner_camera)
    return {
        'H': H,
        'border_L': corner_court[0, 0],
        'border_R': corner_court[0, 1],
        'border_U': corner_court[1, 0],
        'border_D': corner_court[1, 2],
    }


def to_court_coordinate(
    arr_camera: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame,
) -> np.ndarray:
    """Transform camera pixel coordinates (2, N) to court coordinates (2, N).

    Handles resolution scaling (homography was computed at 1280x720).

    :param arr_camera: (2, N) array of camera pixel coordinates.
    :param vid: Video ID.
    :param all_court_info: Dict of {vid: court_info} from get_court_info().
    :param res_df: Resolution DataFrame indexed by video ID.
    :return: (2, N) array of court coordinates.
    """
    res_info = res_df.loc[vid]
    H = all_court_info[vid]['H']
    arr_camera = scale_pos_by_resolution(arr_camera, width=res_info['width'], height=res_info['height'])
    arr_camera = convert_homogeneous(arr_camera)
    return project(H, arr_camera)


def normalize_position(arr: np.ndarray, court_info: dict) -> np.ndarray:
    """Normalize court coordinates (2, N) to [0, 1] using court boundaries.

    :param arr: (2, N) array of court coordinates.
    :param court_info: Dict from get_court_info() with border keys.
    :return: (2, N) array with values normalized to [0, 1].
    """
    x_dist = court_info['border_R'] - court_info['border_L']
    y_dist = court_info['border_D'] - court_info['border_U']
    x_normalized = (arr[0, :] - court_info['border_L']) / x_dist
    y_normalized = (arr[1, :] - court_info['border_U']) / y_dist
    return np.stack((x_normalized, y_normalized))


def check_pos_in_court(
    keypoints: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Check if detected people are on-court and return normalised positions.

    :param keypoints: (m, J, 2) array of joint coordinates in camera pixels.
    :param vid: Video ID.
    :param all_court_info: Dict of {vid: court_info} from get_court_info().
    :param res_df: Resolution DataFrame indexed by video ID.
    :return: Tuple of (in_court, pos_court_normalized) where in_court is
        a (m,) boolean mask and pos_court_normalized is (m, 2).
    """
    n_people = keypoints.shape[0]

    # Use foot keypoints (last 2 joints in COCO format) as position proxy
    feet_camera = keypoints[:, -2:, :].reshape(-1, 2).T  # (2, m*J)
    feet_court = to_court_coordinate(feet_camera, vid, all_court_info, res_df)
    feet_court = feet_court.reshape(2, n_people, -1)  # (2, m, J)

    pos_court = feet_court.mean(axis=-1)  # midpoint between feet, (2, m)
    pos_court_normalized = normalize_position(pos_court, court_info=all_court_info[vid]).T  # (m, 2)

    eps = 0.01  # soft border tolerance
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    return in_court, pos_court_normalized


def load_all_court_info(
    homo_csv_path: Path = SET_INFO_DIR / 'homography.csv',
) -> dict:
    """Load court info for all videos from homography.csv.

    :param homo_csv_path: Path to homography.csv.
    :return: Dict mapping video ID to court_info dict.
    """
    homo_df = pd.read_csv(homo_csv_path).set_index('id')
    return {vid: get_court_info(homo_df, vid) for vid in homo_df.index}
