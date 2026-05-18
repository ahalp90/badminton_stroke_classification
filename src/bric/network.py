"""BRIC network: R(2+1)D-18 backbone with optional shuttle / court fusion.

See ``docs/bric_training_design.md`` for the variant matrix and
encoder shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from shared.taxonomy import DEFAULT_TAXONOMY, TAXONOMIES, Taxonomy

_RGB_FEATURE_DIM = 512
_SHUTTLE_DIM = 64
_COURT_DIM = 64

_SHUTTLE_IN_CHANNELS = 5     # x_norm, y_norm, visibility, dx, dy
_COURT_IN_CHANNELS = 3       # x_half, y_half, valid


def _length_mask(t_max: int, lengths: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return a ``(B, T_max, 1)`` 0/1 mask from ``lengths``."""
    positions = torch.arange(t_max, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(dtype).unsqueeze(-1)


class _ShuttleEncoderMean(nn.Module):
    """Per-frame MLP followed by length-masked temporal mean-pool.

    Baseline encoder. Discards trajectory shape — the mean of a rising-
    then-falling smash trajectory looks the same as a static one.
    """

    output_dim = _SHUTTLE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(_SHUTTLE_IN_CHANNELS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _SHUTTLE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, shuttle: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        per_frame = self.frame_mlp(shuttle)
        mask = _length_mask(per_frame.shape[1], lengths, per_frame.dtype)
        masked_sum = (per_frame * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom


class _ShuttleEncoderStats(nn.Module):
    """Per-frame MLP, then masked [mean, std, max] concatenated pooling.

    Preserves distributional shape (second-order moments + range) that
    mean-pool destroys, without temporal sequence modelling.
    """

    output_dim = 3 * _SHUTTLE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(_SHUTTLE_IN_CHANNELS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _SHUTTLE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, shuttle: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        per_frame = self.frame_mlp(shuttle)
        mask = _length_mask(per_frame.shape[1], lengths, per_frame.dtype)

        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = (per_frame * mask).sum(dim=1) / denom
        # Masked variance via E[x^2] - E[x]^2; clamp for numerical safety.
        mean_sq = (per_frame.square() * mask).sum(dim=1) / denom
        std = (mean_sq - mean.square()).clamp(min=0.0).sqrt()
        # Masked max: substitute -inf at padded positions so they lose any argmax.
        neg_inf = torch.finfo(per_frame.dtype).min
        masked_for_max = per_frame.masked_fill(mask == 0, neg_inf)
        max_pool = masked_for_max.amax(dim=1)
        return torch.cat([mean, std, max_pool], dim=-1)


class _ShuttleEncoderTCN(nn.Module):
    """Dilated 1D convs over the time axis, then masked mean-pool.

    Per-timestep features encode local trajectory shape (rise / fall /
    curve) thanks to the conv's temporal receptive field, before the
    final pool reduces to a single vector. Receptive field across the
    two conv layers is ~13 frames (kernel 5 + dilated kernel 5).

    Mirrors BST's ``tcn_shuttle`` design intent (see bst.py:162).
    """

    output_dim = _SHUTTLE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(_SHUTTLE_IN_CHANNELS, 64, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, _SHUTTLE_DIM, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, shuttle: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # (B, T, C_in) -> (B, C_in, T) for Conv1d channel-first.
        x = shuttle.transpose(1, 2)
        x = self.tcn(x)
        # (B, C_out, T) -> (B, T, C_out) for the masked-mean pool.
        x = x.transpose(1, 2)
        mask = _length_mask(x.shape[1], lengths, x.dtype)
        masked_sum = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom


_SHUTTLE_ENCODERS: dict[str, type[nn.Module]] = {
    'mean':  _ShuttleEncoderMean,
    'stats': _ShuttleEncoderStats,
    'tcn':   _ShuttleEncoderTCN,
}


class _CourtEncoderSnapshot(nn.Module):
    """MLP over the single-frame court snapshot at the target frame.

    Position-only encoder — sees only where the striker stood at the
    moment of contact (smooth-radius-searched if the target frame's
    bbox was invalid). No temporal information.
    """

    output_dim = _COURT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(_COURT_IN_CHANNELS, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, _COURT_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        snapshot: torch.Tensor,    # (B, 3)
        sequence: torch.Tensor,    # (B, T, 3) — unused
        seq_lengths: torch.Tensor,  # (B,) — unused
    ) -> torch.Tensor:
        return self.mlp(snapshot)


class _CourtEncoderTCN(nn.Module):
    """Dilated 1D convs over the court-position sequence, then masked mean-pool.

    Same architectural template as ``_ShuttleEncoderTCN`` — the per-timestep
    features encode local movement shape (turns, accelerations, recovery
    moves) before reduction. Receptive field ~13 frames across the two
    conv layers.
    """

    output_dim = _COURT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(_COURT_IN_CHANNELS, 32, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, _COURT_DIM, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        snapshot: torch.Tensor,    # (B, 3) — unused
        sequence: torch.Tensor,    # (B, T, 3)
        seq_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # (B, T, C_in) -> (B, C_in, T) for Conv1d channel-first.
        x = sequence.transpose(1, 2)
        x = self.tcn(x)
        # (B, C_out, T) -> (B, T, C_out) for the masked-mean pool.
        x = x.transpose(1, 2)
        mask = _length_mask(x.shape[1], seq_lengths, x.dtype)
        masked_sum = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom


_COURT_ENCODERS: dict[str, type[nn.Module]] = {
    'snapshot': _CourtEncoderSnapshot,
    'tcn':      _CourtEncoderTCN,
}


class BRICNetwork(nn.Module):
    """R(2+1)D-18 backbone, optional shuttle/court encoders, fusion classifier.

    Disabled-modality encoders are not instantiated. The ``use_shuttle``
    and ``use_court`` flags are stored on the module so checkpoints
    can be reloaded without external configuration.
    """

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        pretrained: bool = True,
        num_classes: int | None = None,
        *,
        use_shuttle: bool = False,
        use_court: bool = False,
        shuttle_encoder: str = 'mean',
        court_encoder: str = 'snapshot',
    ) -> None:
        """Construct the network.

        :param taxonomy: stroke taxonomy. Defaults to
            ``shared.taxonomy.DEFAULT_TAXONOMY``; pass explicitly in
            production code so the run pins its class scheme.
        :param pretrained: load Kinetics-400 backbone weights.
        :param num_classes: override the head size; bypasses taxonomy.
            Intended for tests.
        :param use_shuttle: instantiate the shuttle encoder.
        :param use_court: instantiate the court encoder.
        :param shuttle_encoder: which shuttle-encoder variant to use:
            ``'mean'`` (per-frame MLP + masked mean-pool — baseline),
            ``'stats'`` (per-frame MLP + masked [mean, std, max] pool),
            or ``'tcn'`` (dilated 1D conv + masked mean-pool).
            Ignored when ``use_shuttle=False``.
        :param court_encoder: which court-encoder variant to use:
            ``'snapshot'`` (MLP over single-frame position at target_frame)
            or ``'tcn'`` (dilated 1D conv over court-position sequence
            spanning the shot window). Ignored when ``use_court=False``.
        """
        super().__init__()

        if num_classes is None:
            if taxonomy is None:
                taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
            num_classes = taxonomy.n_trainable_classes

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)

        backbone_dim = self.backbone.fc.in_features
        assert backbone_dim == _RGB_FEATURE_DIM, (
            f'Expected backbone fc in_features={_RGB_FEATURE_DIM}, got {backbone_dim}'
        )
        self.backbone.fc = nn.Identity()

        if use_shuttle:
            if shuttle_encoder not in _SHUTTLE_ENCODERS:
                raise ValueError(
                    f'shuttle_encoder must be one of {sorted(_SHUTTLE_ENCODERS.keys())}, '
                    f'got {shuttle_encoder!r}'
                )
            self.shuttle_encoder = _SHUTTLE_ENCODERS[shuttle_encoder]()
            shuttle_out_dim = self.shuttle_encoder.output_dim
        else:
            self.shuttle_encoder = None
            shuttle_out_dim = 0

        if use_court:
            if court_encoder not in _COURT_ENCODERS:
                raise ValueError(
                    f'court_encoder must be one of {sorted(_COURT_ENCODERS.keys())}, '
                    f'got {court_encoder!r}'
                )
            self.court_encoder = _COURT_ENCODERS[court_encoder]()
            court_out_dim = self.court_encoder.output_dim
        else:
            self.court_encoder = None
            court_out_dim = 0

        # Per-lane LayerNorm before concat aligns feature magnitudes across the
        # pretrained backbone (post-ReLU Kinetics-trained scale) and the
        # randomly-initialised auxiliary MLPs, preventing one stream from
        # dominating the gradient through the linear classifier in early epochs.
        self.rgb_norm = nn.LayerNorm(_RGB_FEATURE_DIM)
        self.shuttle_norm = nn.LayerNorm(shuttle_out_dim) if use_shuttle else None
        self.court_norm = nn.LayerNorm(court_out_dim) if use_court else None

        fusion_dim = _RGB_FEATURE_DIM + shuttle_out_dim + court_out_dim
        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.taxonomy = taxonomy
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim
        self.use_shuttle = use_shuttle
        self.use_court = use_court
        self.shuttle_encoder_name = shuttle_encoder if use_shuttle else None
        self.court_encoder_name = court_encoder if use_court else None
        self.fusion_dim = fusion_dim

    def forward(
        self,
        rgb: torch.Tensor,
        shuttle: torch.Tensor | None = None,
        shuttle_length: torch.Tensor | None = None,
        court_snapshot: torch.Tensor | None = None,
        court_sequence: torch.Tensor | None = None,
        court_sequence_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = [self.rgb_norm(self.backbone(rgb))]

        if self.shuttle_encoder is not None:
            if shuttle is None or shuttle_length is None:
                raise ValueError('use_shuttle=True but shuttle / shuttle_length not provided')
            feats.append(self.shuttle_norm(self.shuttle_encoder(shuttle, shuttle_length)))

        if self.court_encoder is not None:
            # Both court forms must be provided; each encoder ignores the one it
            # doesn't use. The dataset returns both so the encoder ablation can
            # swap without dataset changes.
            if (court_snapshot is None
                    or court_sequence is None
                    or court_sequence_length is None):
                raise ValueError(
                    'use_court=True but court_snapshot / court_sequence / '
                    'court_sequence_length not provided'
                )
            feats.append(self.court_norm(self.court_encoder(
                court_snapshot, court_sequence, court_sequence_length,
            )))

        fused = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.classifier(fused)
