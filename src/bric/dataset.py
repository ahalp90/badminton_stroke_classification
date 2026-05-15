"""ShuttleSet stroke dataset for BRIC training.

Reads from the preprocessed caches built by:
  - ``scripts.build_shots_master``    → per-stroke metadata CSV
  - ``scripts.bric.preprocess_videos`` → per-stroke RGB tensors + per-vid
                                         striker bboxes (NPZ wide format)
  - ``scripts.bric.extract_shuttle``   → per-vid TrackNetV3 shuttle predictions

Per-stroke ``__getitem__`` returns a dict:

  - ``rgb``             (3, T=32, 224, 224) float32, channels-first,
                        Kinetics-normalised.
  - ``shuttle``         (T_var, 5) float32,
                        ``[x_norm, y_norm, visibility, dx, dy]`` over the
                        stroke's shuttle window. Variable length per
                        stroke — use ``collate_strokes`` to right-pad
                        to the batch max.
  - ``shuttle_length``  int, true length of the shuttle window before
                        padding. Used by the network's masked pool to
                        ignore padded positions.
  - ``court``           (3,) float32, striker court ``[x, y, valid]`` at
                        ``target_frame``. ``x``, ``y`` normalised to
                        [0, 1] over the singles court polygon. ``valid``
                        is 1.0 when a striker bbox was found within
                        ±``smooth_radius`` frames of ``target_frame``,
                        else 0.0 (in which case ``x = y = 0``).
  - ``label``           int, class index into the active taxonomy
  - ``clip_stem``       str, identifies the stroke (for diagnostics)

The shuttle ``dx``/``dy`` channels are forward-differenced
(``dx[t] = x[t] - x[t-1]``, ``dx[0] = 0``) and zeroed across
visibility transitions so the encoder doesn't see synthetic jumps
when the shuttle reappears after an occlusion.

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
    _REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
)
_DEFAULT_RGB_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'rgb'
_DEFAULT_PLAYERS_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'players'
_DEFAULT_SHUTTLE_DIR = _REPO_ROOT / 'training' / 'bric' / 'cache' / 'shuttle'

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
        return ``(T, 5)`` float32 of ``[x_norm, y_norm, visibility, dx, dy]``.

        Velocity is forward-differenced and zeroed across visibility
        transitions — when the shuttle reappears after an occlusion
        the apparent jump is not motion, so propagating it would inject
        a high-magnitude noise signal into the encoder.
        """
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

        t = x.shape[0]
        dx = np.zeros(t, dtype=np.float32)
        dy = np.zeros(t, dtype=np.float32)
        if t > 1:
            valid_pair = (v[1:] > 0) & (v[:-1] > 0)
            dx[1:] = (x[1:] - x[:-1]) * valid_pair
            dy[1:] = (y[1:] - y[:-1]) * valid_pair
        return np.stack([x, y, v, dx, dy], axis=-1)

    def _build_court(
        self, vid: int, target_f: int, side: str,
    ) -> np.ndarray:
        """Project striker foot to normalised court ``[x, y, valid]``;
        fallback to nearest valid frame within ``smooth_radius`` if
        ``target_frame``'s striker bbox is missing.

        :return: ``(3,)`` float32 ``[x, y, valid]``; ``x``, ``y`` in
            [0, 1]. ``valid`` is 1.0 when a striker bbox was resolved,
            0.0 (with ``x = y = 0``) when nothing was found within
            ±``smooth_radius`` frames. The encoder uses ``valid`` to
            distinguish "striker is at court origin" from "no striker
            data" — without it both look identical.
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
        return np.array([x_n, y_n, 1.0], dtype=np.float32)

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
            'rgb':            torch.from_numpy(rgb),                     # (3, 32, 224, 224)
            'shuttle':        torch.from_numpy(shuttle),                 # (T, 5)
            'shuttle_length': int(shuttle.shape[0]),
            'court':          torch.from_numpy(court),                   # (3,)
            'label':          torch.tensor(s['label'], dtype=torch.long),
            'clip_stem':      s['clip_stem'],
        }


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------
def collate_strokes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of stroke dicts.

    Pads variable-length shuttle sequences to the batch max with zeros
    on the right; emits a ``shuttle_length`` tensor so the encoder can
    mask padded positions in its temporal pool.

    :param batch: list of dicts as returned by ``ShuttleSetDataset.__getitem__``.
    :return: dict with batched tensors:
        - ``rgb``            (B, 3, 32, 224, 224) float32
        - ``shuttle``        (B, T_max, 5) float32, zero-padded on the right
        - ``shuttle_length`` (B,) long, true unpadded lengths
        - ``court``          (B, 3) float32
        - ``label``          (B,) long
        - ``clip_stem``      list[str] of length B
    """
    rgb = torch.stack([b['rgb'] for b in batch], dim=0)
    court = torch.stack([b['court'] for b in batch], dim=0)
    label = torch.stack([b['label'] for b in batch], dim=0)

    lengths = torch.tensor([b['shuttle_length'] for b in batch], dtype=torch.long)
    t_max = int(lengths.max().item()) if len(batch) > 0 else 0
    shuttle = torch.zeros((len(batch), t_max, 5), dtype=torch.float32)
    for i, b in enumerate(batch):
        seq = b['shuttle']
        shuttle[i, : seq.shape[0]] = seq

    return {
        'rgb':            rgb,
        'shuttle':        shuttle,
        'shuttle_length': lengths,
        'court':          court,
        'label':          label,
        'clip_stem':      [b['clip_stem'] for b in batch],
    }
