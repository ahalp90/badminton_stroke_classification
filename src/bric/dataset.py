"""Per-stroke ShuttleSet dataset for BRIC.

Reads from the preprocessed caches under ``training/bric/cache/``.
See ``docs/bric_training_design.md`` for input shapes and conventions.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from shared.court import (
    convert_homogeneous,
    load_all_court_info,
    project,
    scale_pos_by_resolution,
)
from shared.dataset import HOMOGRAPHY_CSV_PATH
from shared.taxonomy import (
    DEFAULT_TAXONOMY,
    TAXONOMIES,
    Taxonomy,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHOTS_MASTER = (
    _REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
)
_DEFAULT_RGB_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'rgb'
_DEFAULT_PLAYERS_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'players'
_DEFAULT_SHUTTLE_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'shuttle'

# torchvision R2Plus1D_18_Weights.KINETICS400_V1 normalisation.
_RGB_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1, 1)
_RGB_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1, 1)

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SHUTTLE_WINDOWS = ('between_hits', 'outgoing_only')
_VALID_COURT_WINDOWS = ('between_hits', 'pre_shot')

_PRE_SHOT_EPS_FRAMES = 5


def _to_half_court(x_full: float, y_full: float, side_lc: str) -> tuple[float, float]:
    """Map full-court [0, 1]^2 coords to half-court coords for the given side.

    Both Top and Bottom players map to the same reference frame:
      (0, 0) = front-left of own half (closest to net, on player's left)
      (1, 1) = back-right of own half (own baseline, on player's right)

    For Top player: mirror both x and y. For Bottom player: rescale y to
    span [0, 1] over their own half [0.5, 1].
    """
    if side_lc == 'top':
        # Top's half is y in [0, 0.5]; mirror x and y so 'their net' is 0.
        x_half = 1.0 - x_full
        y_half = 1.0 - 2.0 * y_full
    else:
        # Bottom's half is y in [0.5, 1]; rescale so 'their net' is 0.
        x_half = x_full
        y_half = 2.0 * y_full - 1.0
    return x_half, y_half


def _resolve_taxonomy(taxonomy: Taxonomy | str | None) -> Taxonomy:
    if taxonomy is None:
        return TAXONOMIES[DEFAULT_TAXONOMY]
    if isinstance(taxonomy, str):
        return TAXONOMIES[taxonomy]
    return taxonomy


def _derive_label(
    raw_type: str, player_side: str, taxonomy: Taxonomy,
) -> str | None:
    """Map ``raw_type`` through the taxonomy; return the label or ``None`` to drop."""
    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)
    if merged == 'unknown':
        return None
    if merged in taxonomy.standalone_set:
        return merged
    if merged in taxonomy.base_types:
        return f'{player_side}_{merged}'
    return None


class ShuttleSetDataset(Dataset):
    """Per-stroke dataset over the preprocessed BRIC caches."""

    def __init__(
        self,
        split: str,
        taxonomy: Taxonomy | str | None = None,
        *,
        shots_master_path: Path = _DEFAULT_SHOTS_MASTER,
        rgb_cache_dir: Path = _DEFAULT_RGB_DIR,
        players_cache_dir: Path = _DEFAULT_PLAYERS_DIR,
        shuttle_cache_dir: Path = _DEFAULT_SHUTTLE_DIR,
        homography_csv_path: Path = HOMOGRAPHY_CSV_PATH,
        smooth_radius: int = 16,
        require_court: bool = True,
        require_shuttle_cache: bool = True,
        rgb_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        shuttle_window: str = 'between_hits',
        court_window: str = 'between_hits',
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f'split must be one of {_VALID_SPLITS}, got {split!r}')
        if shuttle_window not in _VALID_SHUTTLE_WINDOWS:
            raise ValueError(
                f'shuttle_window must be one of {_VALID_SHUTTLE_WINDOWS}, '
                f'got {shuttle_window!r}'
            )
        if court_window not in _VALID_COURT_WINDOWS:
            raise ValueError(
                f'court_window must be one of {_VALID_COURT_WINDOWS}, '
                f'got {court_window!r}'
            )

        self.split = split
        self.taxonomy = _resolve_taxonomy(taxonomy)
        self.rgb_cache_dir = Path(rgb_cache_dir)
        self.players_cache_dir = Path(players_cache_dir)
        self.shuttle_cache_dir = Path(shuttle_cache_dir)
        self.smooth_radius = smooth_radius
        self.rgb_transform = rgb_transform
        # 'between_hits': previous-hit to next-hit + eps (full rally context)
        # 'outgoing_only': target_frame to next-hit + eps (just this shot's flight)
        self.shuttle_window = shuttle_window
        # 'between_hits': previous-hit to next-hit (full window)
        # 'pre_shot': previous-hit to target_frame + small eps (incoming approach)
        self.court_window = court_window

        self.classes = self.taxonomy.trainable_class_list()
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.court_info_by_vid: dict[int, dict] = (
            load_all_court_info(Path(homography_csv_path))
        )
        self._players_cache: dict[int, dict[str, np.ndarray]] = {}
        self._shuttle_cache: dict[int, dict[str, np.ndarray]] = {}

        self.samples = self._build_samples(
            Path(shots_master_path), require_court, require_shuttle_cache,
        )

    def _build_samples(
        self,
        shots_master_path: Path,
        require_court: bool,
        require_shuttle_cache: bool,
    ) -> list[dict[str, Any]]:
        df = pd.read_csv(shots_master_path)
        df = df[df['split_v2'] == self.split].copy()

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            label_str = _derive_label(
                row['raw_type_en'], row['player_side'], self.taxonomy,
            )
            if label_str is None or label_str not in self._class_to_idx:
                continue

            clip_stem = row['clip_stem']
            rgb_path = self.rgb_cache_dir / f'{clip_stem}.npy'
            if not rgb_path.exists():
                continue

            vid = int(row['vid'])
            if require_court and vid not in self.court_info_by_vid:
                continue
            if require_shuttle_cache and not (
                self.shuttle_cache_dir / f'{vid}.npz'
            ).exists():
                continue

            records.append({
                'clip_stem':       clip_stem,
                'rgb_path':        rgb_path,
                'vid':             vid,
                'frame_num':       int(row['frame_num']),
                'shuttle_start_f': int(row['shuttle_start_f']),
                'shuttle_end_f':   int(row['shuttle_end_f']),
                'player_side':     row['player_side'],
                'label':           self._class_to_idx[label_str],
                'label_str':       label_str,
            })
        return records

    def _get_players(self, vid: int) -> dict[str, np.ndarray]:
        cache = self._players_cache.get(vid)
        if cache is None:
            with np.load(self.players_cache_dir / f'{vid}.npz', allow_pickle=True) as f:
                cache = {k: f[k] for k in f.files}
            self._players_cache[vid] = cache
        return cache

    def _get_shuttle(self, vid: int) -> dict[str, np.ndarray]:
        cache = self._shuttle_cache.get(vid)
        if cache is None:
            with np.load(self.shuttle_cache_dir / f'{vid}.npz') as f:
                cache = {k: f[k] for k in f.files}
            self._shuttle_cache[vid] = cache
        return cache

    def _build_rgb(self, rgb_path: Path) -> np.ndarray:
        thwc = np.load(rgb_path)
        if self.rgb_transform is not None:
            thwc = self.rgb_transform(thwc)
        cthw = thwc.transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        return (cthw - _RGB_MEAN) / _RGB_STD

    def _build_shuttle(self, vid: int, start_f: int, end_f: int) -> np.ndarray:
        s = self._get_shuttle(vid)
        p = self._get_players(vid)
        w = float(p['width'])
        h = float(p['height'])
        n = len(s['x'])
        a = max(0, int(start_f))
        b = min(n, int(end_f))
        x = s['x'][a:b].astype(np.float32) / w
        y = s['y'][a:b].astype(np.float32) / h
        v = s['visibility'][a:b].astype(np.float32)

        # Zero velocity across visibility transitions: when the shuttle
        # reappears after an occlusion, the apparent jump is not motion
        # and would otherwise inject a high-magnitude noise spike.
        t = x.shape[0]
        dx = np.zeros(t, dtype=np.float32)
        dy = np.zeros(t, dtype=np.float32)
        if t > 1:
            valid_pair = (v[1:] > 0) & (v[:-1] > 0)
            dx[1:] = (x[1:] - x[:-1]) * valid_pair
            dy[1:] = (y[1:] - y[:-1]) * valid_pair
        return np.stack([x, y, v, dx, dy], axis=-1)

    def _project_bbox_to_half_court(
        self,
        bbox: np.ndarray,
        court_info: dict,
        img_w: float,
        img_h: float,
        side_lc: str,
    ) -> tuple[float, float]:
        """Project a single bbox foot-centre to half-court normalised coords."""
        foot_x = (float(bbox[0]) + float(bbox[2])) / 2
        foot_y = float(bbox[3])
        pt = np.array([[foot_x], [foot_y]])
        pt = scale_pos_by_resolution(pt, width=img_w, height=img_h)
        pt = convert_homogeneous(pt)
        court_pt = project(court_info['H'], pt)
        x_n = (court_pt[0, 0] - court_info['border_L']) / (
            court_info['border_R'] - court_info['border_L']
        )
        y_n = (court_pt[1, 0] - court_info['border_U']) / (
            court_info['border_D'] - court_info['border_U']
        )
        return _to_half_court(float(x_n), float(y_n), side_lc)

    def _build_court_snapshot(
        self, vid: int, target_f: int, side: str,
    ) -> np.ndarray:
        """Half-court position at ``target_f`` (smooth-radius nearest valid).

        Returns ``(3,)`` of ``[x_half, y_half, valid]``. Falls back to nearby
        frames within ``self.smooth_radius`` if the target frame's bbox is
        invalid; returns zeros if no valid frame is within range.
        """
        p = self._get_players(vid)
        side_lc = side.lower()
        valid = p[f'{side_lc}_valid']
        bboxes = p[f'{side_lc}_bbox']
        n = len(valid)

        bbox = None
        for offset in range(self.smooth_radius + 1):
            candidates = (target_f,) if offset == 0 else (target_f - offset, target_f + offset)
            for f in candidates:
                if 0 <= f < n and valid[f]:
                    bbox = bboxes[f]
                    break
            if bbox is not None:
                break
        if bbox is None:
            return np.zeros(3, dtype=np.float32)

        court_info = self.court_info_by_vid[vid]
        x_half, y_half = self._project_bbox_to_half_court(
            bbox, court_info, float(p['width']), float(p['height']), side_lc,
        )
        return np.array([x_half, y_half, 1.0], dtype=np.float32)

    def _build_court_sequence(
        self, vid: int, start_f: int, end_f: int, side: str,
    ) -> np.ndarray:
        """Half-court position sequence over the shot window.

        Returns ``(T, 3)`` of ``[x_half, y_half, valid]`` per frame, where
        ``T = end_f - start_f`` (clamped to the available cache length).
        Frames where YOLO failed to detect the player return ``[0, 0, 0]``
        and the encoder masks via the ``valid`` channel.

        Half-court normalisation matches the snapshot path; see
        ``_build_court_snapshot`` for the geometry.
        """
        p = self._get_players(vid)
        side_lc = side.lower()
        valid = p[f'{side_lc}_valid']
        bboxes = p[f'{side_lc}_bbox']
        n = len(valid)
        court_info = self.court_info_by_vid[vid]
        img_w, img_h = float(p['width']), float(p['height'])

        a = max(0, int(start_f))
        b = min(n, int(end_f))
        t = max(0, b - a)
        out = np.zeros((t, 3), dtype=np.float32)
        if t == 0:
            return out

        for i, f in enumerate(range(a, b)):
            if not valid[f]:
                continue
            x_half, y_half = self._project_bbox_to_half_court(
                bboxes[f], court_info, img_w, img_h, side_lc,
            )
            out[i, 0] = x_half
            out[i, 1] = y_half
            out[i, 2] = 1.0
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        rgb = self._build_rgb(s['rgb_path'])
        # outgoing_only restricts the shuttle trajectory to this shot's
        # outgoing flight (target_frame onwards), excluding the incoming
        # leg from the previous player's shot.
        shuttle_start = (
            s['frame_num']
            if self.shuttle_window == 'outgoing_only'
            else s['shuttle_start_f']
        )
        shuttle = self._build_shuttle(s['vid'], shuttle_start, s['shuttle_end_f'])
        # Court is returned in two forms so the encoder ablation can pick:
        #   snapshot — position at target_frame (smooth-radius nearest valid)
        #   sequence — full-window trajectory over the shuttle range
        # The unused form costs only a small per-stroke projection.
        court_snapshot = self._build_court_snapshot(
            s['vid'], s['frame_num'], s['player_side'],
        )
        court_end = (
            min(s['frame_num'] + _PRE_SHOT_EPS_FRAMES, s['shuttle_end_f'])
            if self.court_window == 'pre_shot'
            else s['shuttle_end_f']
        )
        court_sequence = self._build_court_sequence(
            s['vid'], s['shuttle_start_f'], court_end, s['player_side'],
        )
        return {
            'rgb':                   torch.from_numpy(rgb),
            'shuttle':               torch.from_numpy(shuttle),
            'shuttle_length':        int(shuttle.shape[0]),
            'court_snapshot':        torch.from_numpy(court_snapshot),
            'court_sequence':        torch.from_numpy(court_sequence),
            'court_sequence_length': int(court_sequence.shape[0]),
            'label':                 torch.tensor(s['label'], dtype=torch.long),
            'clip_stem':             s['clip_stem'],
        }


def collate_strokes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad variable-length shuttle / court sequences and stack the rest."""
    return {
        'rgb':                   torch.stack([b['rgb'] for b in batch], dim=0),
        'shuttle':               torch.nn.utils.rnn.pad_sequence(
            [b['shuttle'] for b in batch], batch_first=True,
        ),
        'shuttle_length':        torch.tensor(
            [b['shuttle_length'] for b in batch], dtype=torch.long,
        ),
        'court_snapshot':        torch.stack(
            [b['court_snapshot'] for b in batch], dim=0,
        ),
        'court_sequence':        torch.nn.utils.rnn.pad_sequence(
            [b['court_sequence'] for b in batch], batch_first=True,
        ),
        'court_sequence_length': torch.tensor(
            [b['court_sequence_length'] for b in batch], dtype=torch.long,
        ),
        'label':                 torch.stack([b['label'] for b in batch], dim=0),
        'clip_stem':             [b['clip_stem'] for b in batch],
    }
