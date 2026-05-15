"""ShuttleSet stroke dataset for BRIC training.

Reads from the preprocessed caches built by:
  - ``scripts.build_shots_master``    → per-stroke metadata CSV
  - ``scripts.preprocess_videos``     → per-stroke RGB tensors + per-vid
                                        striker bboxes (NPZ wide format)
  - ``scripts.extract_shuttle``       → per-vid TrackNetV3 shuttle predictions

Per-stroke ``__getitem__`` returns a dict:

  - ``rgb``       (3, T=32, 224, 224) float32, channels-first, Kinetics-normalised
  - ``shuttle``   (T_var, 3) float32, ``[x_norm, y_norm, visibility]`` over the
                  stroke's shuttle window. Variable length per stroke — collate
                  with padding or a sequence encoder.
  - ``court``     (2,) float32, striker court (x, y) at ``target_frame``,
                  normalised to [0, 1] over the singles court polygon.
  - ``label``     int, class index into the active taxonomy
  - ``clip_stem`` str, identifies the stroke (for diagnostics)

Filtering at ``__init__``:
  1. Select rows by ``split_v2`` column (BRIC's active split).
  2. Apply ``taxonomy.merge_map`` to ``raw_type_en``; drop rows that map to
     ``'unknown'`` (drop_unknown convention).
  3. Drop rows whose RGB cache file doesn't exist (preprocess failed for
     that stroke — typically <1% bbox-resolution failures).

Caches are loaded lazily per-vid (40 small NPZ files total, kept in memory
once first accessed). Per-stroke RGB tensors load on-demand from disk.
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
    _REPO_ROOT / 'runtime' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
)
_DEFAULT_RGB_DIR = _REPO_ROOT / 'runtime' / 'cache' / 'rgb'
_DEFAULT_PLAYERS_DIR = _REPO_ROOT / 'runtime' / 'cache' / 'players'
_DEFAULT_SHUTTLE_DIR = _REPO_ROOT / 'runtime' / 'cache' / 'shuttle'

# R(2+1)D-18 Kinetics-400 normalisation (torchvision R2Plus1D_18_Weights.KINETICS400_V1).
_RGB_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1, 1)
_RGB_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1, 1)

_VALID_SPLITS = ('train', 'val', 'test')


def _resolve_taxonomy(taxonomy: Taxonomy | str | None) -> Taxonomy:
    if taxonomy is None:
        return TAXONOMIES[DEFAULT_TAXONOMY]
    if isinstance(taxonomy, str):
        return TAXONOMIES[taxonomy]
    return taxonomy


def _derive_label(
    raw_type: str, player_side: str, taxonomy: Taxonomy,
) -> str | None:
    """Apply taxonomy merge + side-prefix logic; return label string or None.

    Returns ``None`` if the merged type is 'unknown' (drop_unknown convention)
    or otherwise not in the trainable class list.
    """
    merged = (taxonomy.merge_map or {}).get(raw_type, raw_type)
    if merged == 'unknown':
        return None
    if merged in taxonomy.standalone_set:
        label = merged
    elif merged in taxonomy.base_types:
        label = f'{player_side}_{merged}'
    else:
        return None
    return label


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
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f'split must be one of {_VALID_SPLITS}, got {split!r}')

        self.split = split
        self.taxonomy = _resolve_taxonomy(taxonomy)
        self.rgb_cache_dir = Path(rgb_cache_dir)
        self.players_cache_dir = Path(players_cache_dir)
        self.shuttle_cache_dir = Path(shuttle_cache_dir)
        self.smooth_radius = smooth_radius
        self.rgb_transform = rgb_transform

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

    # ------------------------------------------------------------------
    # Construction-time filtering
    # ------------------------------------------------------------------
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
                continue   # preprocess didn't produce this stroke (rare bbox failure)

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
                'player_side':     row['player_side'],     # 'Top' or 'Bottom'
                'label':           self._class_to_idx[label_str],
                'label_str':       label_str,
            })
        return records

    # ------------------------------------------------------------------
    # Lazy per-vid cache loaders
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------
    def _build_rgb(self, rgb_path: Path) -> np.ndarray:
        """(32, 224, 224, 3) uint8 → (3, 32, 224, 224) float32, Kinetics-normalised."""
        thwc = np.load(rgb_path)  # (T, H, W, C) uint8
        if self.rgb_transform is not None:
            thwc = self.rgb_transform(thwc)
        cthw = thwc.transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        cthw = (cthw - _RGB_MEAN) / _RGB_STD
        return cthw

    def _build_shuttle(self, vid: int, start_f: int, end_f: int) -> np.ndarray:
        """Slice the per-vid shuttle cache to ``[start_f, end_f)``;
        return ``(T, 3)`` float32 of ``[x_norm, y_norm, visibility]``."""
        s = self._get_shuttle(vid)
        # Normalise pixel coords by source video resolution. Use the players
        # cache for canonical width/height (cv2-probed at preprocess time).
        p = self._get_players(vid)
        w = float(p['width'])
        h = float(p['height'])
        n = len(s['x'])
        a = max(0, int(start_f))
        b = min(n, int(end_f))
        x = s['x'][a:b].astype(np.float32) / w
        y = s['y'][a:b].astype(np.float32) / h
        v = s['visibility'][a:b].astype(np.float32)
        return np.stack([x, y, v], axis=-1)

    def _build_court(
        self, vid: int, target_f: int, side: str,
    ) -> np.ndarray:
        """Project striker foot to normalised court (x, y); fallback to
        nearest valid frame within ``smooth_radius`` if target_frame's
        striker bbox is missing.

        :return: ``(2,)`` float32 in [0, 1]; ``[0., 0.]`` if no valid
            frame found in ``±smooth_radius``.
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
            return np.zeros(2, dtype=np.float32)

        court_info = self.court_info_by_vid[vid]
        foot_x = (float(bbox[0]) + float(bbox[2])) / 2
        foot_y = float(bbox[3])
        pt = np.array([[foot_x], [foot_y]])
        pt = scale_pos_by_resolution(
            pt, width=float(p['width']), height=float(p['height']),
        )
        pt = convert_homogeneous(pt)
        court_pt = project(court_info['H'], pt)
        x_n = (court_pt[0, 0] - court_info['border_L']) / (
            court_info['border_R'] - court_info['border_L']
        )
        y_n = (court_pt[1, 0] - court_info['border_U']) / (
            court_info['border_D'] - court_info['border_U']
        )
        return np.array([x_n, y_n], dtype=np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        rgb = self._build_rgb(s['rgb_path'])
        shuttle = self._build_shuttle(s['vid'], s['shuttle_start_f'], s['shuttle_end_f'])
        court = self._build_court(s['vid'], s['frame_num'], s['player_side'])
        return {
            'rgb':       torch.from_numpy(rgb),                          # (3, 32, 224, 224)
            'shuttle':   torch.from_numpy(shuttle),                      # (T, 3)
            'court':     torch.from_numpy(court),                        # (2,)
            'label':     torch.tensor(s['label'], dtype=torch.long),
            'clip_stem': s['clip_stem'],
        }
