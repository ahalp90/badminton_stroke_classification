"""BRIC neural network — R(2+1)D-18 backbone + classification head.

v1: single-pathway RGB. Auxiliary inputs (shuttle, court) are accepted
in the forward signature but ignored — slots reserved for Day 8 fusion.

The backbone is from torchvision.models.video; we replace the original
400-class Kinetics fc head with our taxonomy-sized head.

Head sizing is derived from a `Taxonomy`. Pass `taxonomy=` explicitly
when constructing the network so the run pins which class scheme it
targets — silent drift between training, checkpoints, and evaluation
is the failure mode this guards against. If `taxonomy=` is omitted
the network falls back to `shared.taxonomy.DEFAULT_TAXONOMY`, which
is fine for notebooks and smoke tests but not for production runs.

Trained checkpoints are tied to whatever `num_classes` they were
trained with — taxonomy swaps require a retrain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from shared.taxonomy import DEFAULT_TAXONOMY, TAXONOMIES, Taxonomy


class BRICNetwork(nn.Module):
    """R(2+1)D-18 + linear classification head.

    Forward signature reserves `shuttle_feats` and `court_feats` slots
    so Day 8 fusion variants can be wired without changing the calling
    code in `bric.train` and `bric.infer`.
    """

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        pretrained: bool = True,
        num_classes: int | None = None,
    ) -> None:
        """Construct the network.

        :param taxonomy: stroke-type taxonomy this network targets.
            Head size = ``taxonomy.n_trainable_classes``. Defaults to
            ``shared.taxonomy.DEFAULT_TAXONOMY`` — pass explicitly in
            training/inference code so the run pins its class scheme.
        :param pretrained: if True, load Kinetics-400 pretrained weights
            for the backbone (~242MB download on first call).
        :param num_classes: explicit head-size override. Bypasses the
            taxonomy entirely — only intended for tests or experiments
            that need an arbitrary head dimension.
        """
        super().__init__()

        if num_classes is None:
            if taxonomy is None:
                taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
            num_classes = taxonomy.n_trainable_classes

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)

        # Original Kinetics head is nn.Linear(512, 400). Replace with our
        # head; in_features is read from the loaded model so a future
        # backbone swap (R(2+1)D-34, etc.) just works.
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(backbone_dim, num_classes)

        self.taxonomy = taxonomy  # None when num_classes was set explicitly
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim

    def forward(
        self,
        rgb: torch.Tensor,
        shuttle_feats: torch.Tensor | None = None,   # noqa: ARG002 — Day 8 fusion slot
        court_feats: torch.Tensor | None = None,     # noqa: ARG002 — Day 8 fusion slot
    ) -> torch.Tensor:
        """Forward pass.

        :param rgb: ``(B, 3, T, H, W)`` input video tensor (CTHW after
            the leading batch dim). Pretrained on 112×112; trained /
            finetuned at any spatial size since R(2+1)D is fully conv.
        :param shuttle_feats: reserved for Day 8 fusion; ignored in v1.
        :param court_feats: reserved for Day 8 fusion; ignored in v1.
        :return: ``(B, num_classes)`` logits (no softmax — apply at
            inference / loss-time as needed).
        """
        return self.backbone(rgb)
