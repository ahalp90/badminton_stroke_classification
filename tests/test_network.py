"""Unit tests for ``bric.network.BRICNetwork``.

Architecture-level tests only — the pretrained backbone download
(~242MB) is exercised by ``scripts/bric/smoke_test.py``.
"""

import pytest
import torch

from bric.network import BRICNetwork
from shared.taxonomy import DEFAULT_TAXONOMY, TAXONOMIES, TAXONOMY_RAW_35


@pytest.fixture(scope='module')
def model():
    return BRICNetwork(pretrained=False).eval()


class TestArchitecture:
    def test_default_num_classes(self, model):
        default_tax = TAXONOMIES[DEFAULT_TAXONOMY]
        assert model.num_classes == default_tax.n_trainable_classes == 14
        assert model.taxonomy is default_tax

    def test_explicit_taxonomy_sizes_head(self):
        net = BRICNetwork(taxonomy=TAXONOMY_RAW_35, pretrained=False)
        assert net.num_classes == TAXONOMY_RAW_35.n_trainable_classes
        assert net.classifier.out_features == net.num_classes
        assert net.taxonomy is TAXONOMY_RAW_35

    def test_explicit_num_classes_overrides(self):
        net = BRICNetwork(num_classes=20, pretrained=False)
        assert net.num_classes == 20
        assert net.classifier.out_features == 20
        assert net.taxonomy is None

    def test_backbone_penultimate_is_512(self, model):
        assert model.backbone_dim == 512


class TestVariantFlags:
    @pytest.mark.parametrize(
        ('use_shuttle', 'use_court', 'expected_dim'),
        [
            (False, False, 512),
            (True,  False, 512 + 64),
            (False, True,  512 + 64),
            (True,  True,  512 + 64 + 64),
        ],
    )
    def test_fusion_dim_per_variant(self, use_shuttle, use_court, expected_dim):
        net = BRICNetwork(
            pretrained=False, use_shuttle=use_shuttle, use_court=use_court,
        )
        assert net.fusion_dim == expected_dim
        assert net.classifier.in_features == expected_dim

    def test_disabled_encoders_not_instantiated(self):
        net = BRICNetwork(pretrained=False, use_shuttle=False, use_court=False)
        assert net.shuttle_encoder is None
        assert net.court_encoder is None

    def test_lane_norms_track_variant_flags(self):
        # rgb_norm always present; auxiliary norms exist iff their lane does.
        rgb_only = BRICNetwork(pretrained=False)
        assert isinstance(rgb_only.rgb_norm, torch.nn.LayerNorm)
        assert rgb_only.shuttle_norm is None
        assert rgb_only.court_norm is None

        full = BRICNetwork(pretrained=False, use_shuttle=True, use_court=True)
        assert full.shuttle_norm.normalized_shape == (64,)
        assert full.court_norm.normalized_shape == (64,)
        assert full.rgb_norm.normalized_shape == (512,)


class TestForwardPass:
    def test_rgb_only_e2e(self, model):
        rgb = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            out = model(rgb)
        assert out.shape == (1, model.num_classes)

    def test_full_fusion_forward(self):
        net = BRICNetwork(
            pretrained=False, use_shuttle=True, use_court=True,
        ).eval()
        rgb = torch.randn(2, 3, 32, 112, 112)
        shuttle = torch.randn(2, 50, 5)
        shuttle_length = torch.tensor([50, 30], dtype=torch.long)
        court = torch.randn(2, 3)
        with torch.no_grad():
            out = net(rgb, shuttle=shuttle, shuttle_length=shuttle_length, court=court)
        assert out.shape == (2, net.num_classes)

    def test_shuttle_required_when_enabled(self):
        net = BRICNetwork(pretrained=False, use_shuttle=True).eval()
        rgb = torch.randn(1, 3, 32, 112, 112)
        with torch.no_grad(), pytest.raises(ValueError, match='use_shuttle=True'):
            net(rgb)

    def test_court_required_when_enabled(self):
        net = BRICNetwork(pretrained=False, use_court=True).eval()
        rgb = torch.randn(1, 3, 32, 112, 112)
        with torch.no_grad(), pytest.raises(ValueError, match='use_court=True'):
            net(rgb)

    def test_masked_pool_invariant_to_padding(self):
        # frame_mlp(0) is non-zero due to bias terms, so unmasked-mean
        # would shift toward padding. The mask must zero those positions.
        net = BRICNetwork(pretrained=False, use_shuttle=True).eval()
        real = torch.randn(1, 10, 5)
        padded = torch.cat([real, torch.zeros(1, 20, 5)], dim=1)
        length = torch.tensor([10], dtype=torch.long)
        with torch.no_grad():
            pooled_unpadded = net.shuttle_encoder(real, length)
            pooled_padded = net.shuttle_encoder(padded, length)
        torch.testing.assert_close(pooled_unpadded, pooled_padded)
