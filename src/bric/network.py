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
_COURT_IN_CHANNELS = 3       # x_norm, y_norm, valid


class _ShuttleEncoder(nn.Module):
    """Per-frame MLP followed by length-masked temporal mean-pool."""

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
        _, t_max, _ = per_frame.shape
        positions = torch.arange(t_max, device=per_frame.device).unsqueeze(0)
        mask = (positions < lengths.unsqueeze(1)).to(per_frame.dtype).unsqueeze(-1)
        masked_sum = (per_frame * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / denom


class _CourtEncoder(nn.Module):
    """MLP over the (B, 3) court snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(_COURT_IN_CHANNELS, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, _COURT_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, court: torch.Tensor) -> torch.Tensor:
        return self.mlp(court)


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

        self.shuttle_encoder = _ShuttleEncoder() if use_shuttle else None
        self.court_encoder = _CourtEncoder() if use_court else None

        # Per-lane LayerNorm before concat aligns feature magnitudes across the
        # pretrained backbone (post-ReLU Kinetics-trained scale) and the
        # randomly-initialised auxiliary MLPs, preventing one stream from
        # dominating the gradient through the linear classifier in early epochs.
        self.rgb_norm = nn.LayerNorm(_RGB_FEATURE_DIM)
        self.shuttle_norm = nn.LayerNorm(_SHUTTLE_DIM) if use_shuttle else None
        self.court_norm = nn.LayerNorm(_COURT_DIM) if use_court else None

        fusion_dim = _RGB_FEATURE_DIM
        if use_shuttle:
            fusion_dim += _SHUTTLE_DIM
        if use_court:
            fusion_dim += _COURT_DIM
        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.taxonomy = taxonomy
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim
        self.use_shuttle = use_shuttle
        self.use_court = use_court
        self.fusion_dim = fusion_dim

    def forward(
        self,
        rgb: torch.Tensor,
        shuttle: torch.Tensor | None = None,
        shuttle_length: torch.Tensor | None = None,
        court: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = [self.rgb_norm(self.backbone(rgb))]

        if self.shuttle_encoder is not None:
            if shuttle is None or shuttle_length is None:
                raise ValueError('use_shuttle=True but shuttle / shuttle_length not provided')
            feats.append(self.shuttle_norm(self.shuttle_encoder(shuttle, shuttle_length)))

        if self.court_encoder is not None:
            if court is None:
                raise ValueError('use_court=True but court not provided')
            feats.append(self.court_norm(self.court_encoder(court)))

        fused = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.classifier(fused)
