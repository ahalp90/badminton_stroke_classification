"""Court homography utilities, mirrored from bst_x.pipeline.court_utils.

Pure functions for camera-pixel <-> normalised court coordinate transforms.

Two ways to use this module:

1. **From a pre-computed H matrix (BRIC's primary path).**
   The hba frontend gives us 4 corner points; backend computes
   `H = cv2.getPerspectiveTransform(boundary, REF_COURT_CORNERS_PX)` once
   per video, then uses `project()` to transform foot-centre coordinates
   to court space.

2. **From BST's homography.csv (legacy / cross-checking).**
   Use `load_all_court_info()` and `get_court_info()` to load BST's
   pre-computed matrices for ShuttleSet videos.

What's NOT mirrored:
  - The `pipeline.config` import. HOMOGRAPHY_RESOLUTION is defined locally
    so this module has no cross-package dependencies.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Resolution at which BST's homography.csv matrices were computed.
# Pixel coordinates from a different resolution (e.g. our 1080p source
# videos) must be scaled to this before applying H — see
# `scale_pos_by_resolution` and `to_court_coordinate`. Source-of-truth
# for BRIC; mirrored from bst_x.pipeline.config.
HOMOGRAPHY_RESOLUTION: tuple[int, int] = (1280, 720)

# Singles-court reference rectangle in metres (13.4 m × 6.1 m). Used as the
# target plane when computing a homography from user-marked corner points.
# Order matches the hba frontend's corner order: top-left, top-right,
# bottom-right, bottom-left.
REF_COURT_M: tuple[float, float] = (13.4, 6.1)
REF_COURT_CORNERS_M = np.array(
    [[0.0,            0.0],
     [REF_COURT_M[0], 0.0],
     [REF_COURT_M[0], REF_COURT_M[1]],
     [0.0,            REF_COURT_M[1]]],
    dtype=np.float32,
)


def get_H(homography_info: pd.Series) -> np.ndarray:
    """Parse the 3x3 homography matrix from a homography.csv row."""
    h_str: str = homography_info['homography_matrix']
    clean_str = h_str.replace('[', '').replace(']', '').replace(',', ' ')
    return np.fromstring(clean_str, sep=' ').reshape((3, 3))


def get_corner_camera(homography_info: pd.Series) -> np.ndarray:
    """Extract the 4 court corner coordinates (2, 4) from a homography.csv row."""
    corner_camera = homography_info.loc['upleft_x':'downright_y']
    return corner_camera.to_numpy(dtype=float).reshape((2, 4))


def convert_homogeneous(arr: np.ndarray) -> np.ndarray:
    """Convert (2, N) array to homogeneous coordinates (3, N)."""
    return np.concatenate((arr, np.full((1, arr.shape[-1]), 1.0)), axis=0)


def scale_pos_by_resolution(
    arr: np.ndarray, width: float, height: float,
) -> np.ndarray:
    """Scale (2, N) or (3, N) coordinates from source res to homography res."""
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    new_arr = arr.copy()
    new_arr[0, :] *= aim_w / width
    new_arr[1, :] *= aim_h / height
    return new_arr


def project(H: np.ndarray, P_prime: np.ndarray) -> np.ndarray:
    """Apply homography: (3, N) homogeneous camera coords -> (2, N) court coords."""
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]
    return P


def get_court_info(homo_df: pd.DataFrame, vid: int) -> dict:
    """Get homography matrix and court boundary coordinates for a video."""
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
    """Transform camera pixel coordinates (2, N) to court coordinates (2, N)."""
    res_info = res_df.loc[vid]
    H = all_court_info[vid]['H']
    arr_camera = scale_pos_by_resolution(arr_camera, width=res_info['width'], height=res_info['height'])
    arr_camera = convert_homogeneous(arr_camera)
    return project(H, arr_camera)


def normalize_position(arr: np.ndarray, court_info: dict) -> np.ndarray:
    """Normalize court coordinates (2, N) to [0, 1] using court boundaries."""
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
    :return: (in_court mask, pos_court_normalized) where in_court is (m,)
        boolean and pos_court_normalized is (m, 2).
    """
    n_people = keypoints.shape[0]
    feet_camera = keypoints[:, -2:, :].reshape(-1, 2).T
    feet_court = to_court_coordinate(feet_camera, vid, all_court_info, res_df)
    feet_court = feet_court.reshape(2, n_people, -1)
    pos_court = feet_court.mean(axis=-1)
    pos_court_normalized = normalize_position(pos_court, court_info=all_court_info[vid]).T

    eps = 0.01
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    return in_court, pos_court_normalized


def load_all_court_info(homo_csv_path: Path) -> dict:
    """Load court info for all videos from BST's homography.csv."""
    homo_df = pd.read_csv(homo_csv_path).set_index('id')
    return {vid: get_court_info(homo_df, vid) for vid in homo_df.index}
