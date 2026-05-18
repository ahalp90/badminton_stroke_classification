"""Unit tests for ``bric.network.BRICNetwork``.

Architecture-level tests only — the pretrained backbone download
(~242MB) is exercised by ``src/bric/smoke_test.py``.
"""

import pytest
import torch

from bric.network import _COURT_ENCODERS, _SHUTTLE_ENCODERS, BRICNetwork
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


class TestShuttleEncoderVariants:
    @pytest.mark.parametrize(
        ('encoder', 'expected_shuttle_dim'),
        [('mean', 64), ('stats', 192), ('tcn', 64)],
    )
    def test_encoder_dims_propagate_to_fusion(self, encoder, expected_shuttle_dim):
        net = BRICNetwork(
            pretrained=False, use_shuttle=True, shuttle_encoder=encoder,
        )
        assert net.shuttle_encoder.output_dim == expected_shuttle_dim
        assert net.shuttle_norm.normalized_shape == (expected_shuttle_dim,)
        assert net.fusion_dim == 512 + expected_shuttle_dim
        assert net.shuttle_encoder_name == encoder

    def test_unknown_encoder_rejected(self):
        with pytest.raises(ValueError, match='shuttle_encoder must be one of'):
            BRICNetwork(pretrained=False, use_shuttle=True, shuttle_encoder='xattn')

    def test_encoder_name_is_none_when_shuttle_disabled(self):
        net = BRICNetwork(pretrained=False, use_shuttle=False, shuttle_encoder='tcn')
        assert net.shuttle_encoder_name is None
        assert net.shuttle_encoder is None

    def test_all_registered_encoders_have_output_dim(self):
        # Guard rail: anyone adding a new encoder must declare output_dim.
        for name, cls in _SHUTTLE_ENCODERS.items():
            assert hasattr(cls, 'output_dim'), f'{name} encoder missing output_dim'


class TestCourtEncoderVariants:
    @pytest.mark.parametrize('encoder', ['snapshot', 'tcn'])
    def test_encoder_dims_propagate_to_fusion(self, encoder):
        net = BRICNetwork(
            pretrained=False, use_court=True, court_encoder=encoder,
        )
        assert net.court_encoder.output_dim == 64
        assert net.court_norm.normalized_shape == (64,)
        assert net.fusion_dim == 512 + 64
        assert net.court_encoder_name == encoder

    def test_unknown_court_encoder_rejected(self):
        with pytest.raises(ValueError, match='court_encoder must be one of'):
            BRICNetwork(pretrained=False, use_court=True, court_encoder='xattn')

    def test_court_encoder_name_is_none_when_court_disabled(self):
        net = BRICNetwork(pretrained=False, use_court=False, court_encoder='tcn')
        assert net.court_encoder_name is None
        assert net.court_encoder is None

    def test_all_registered_court_encoders_have_output_dim(self):
        for name, cls in _COURT_ENCODERS.items():
            assert hasattr(cls, 'output_dim'), f'{name} encoder missing output_dim'


class TestForwardPass:
    def test_rgb_only_e2e(self, model):
        rgb = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            out = model(rgb)
        assert out.shape == (1, model.num_classes)

    @pytest.mark.parametrize('court_encoder', ['snapshot', 'tcn'])
    def test_full_fusion_forward(self, court_encoder):
        net = BRICNetwork(
            pretrained=False, use_shuttle=True, use_court=True,
            court_encoder=court_encoder,
        ).eval()
        rgb = torch.randn(2, 3, 32, 112, 112)
        shuttle = torch.randn(2, 50, 5)
        shuttle_length = torch.tensor([50, 30], dtype=torch.long)
        court_snapshot = torch.randn(2, 3)
        court_sequence = torch.randn(2, 50, 3)
        court_sequence_length = torch.tensor([50, 30], dtype=torch.long)
        with torch.no_grad():
            out = net(
                rgb, shuttle=shuttle, shuttle_length=shuttle_length,
                court_snapshot=court_snapshot,
                court_sequence=court_sequence,
                court_sequence_length=court_sequence_length,
            )
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

    def test_partial_court_inputs_rejected(self):
        # All three court fields must be provided; passing only some
        # raises rather than silently degrading.
        net = BRICNetwork(pretrained=False, use_court=True).eval()
        rgb = torch.randn(1, 3, 32, 112, 112)
        with torch.no_grad(), pytest.raises(ValueError, match='use_court=True'):
            net(rgb, court_snapshot=torch.randn(1, 3))  # missing sequence + length

    @pytest.mark.parametrize('encoder', ['mean', 'stats'])
    def test_per_frame_encoders_invariant_to_padding(self, encoder):
        # frame_mlp(0) is non-zero due to bias terms, so unmasked pooling
        # would shift toward padding. The mask must zero those positions.
        # The tcn encoder is NOT invariant under this test because the conv
        # kernel sees padding values via its receptive field — that's a
        # known and accepted property of the encoder, not a bug.
        net = BRICNetwork(
            pretrained=False, use_shuttle=True, shuttle_encoder=encoder,
        ).eval()
        real = torch.randn(1, 10, 5)
        padded = torch.cat([real, torch.zeros(1, 20, 5)], dim=1)
        length = torch.tensor([10], dtype=torch.long)
        with torch.no_grad():
            pooled_unpadded = net.shuttle_encoder(real, length)
            pooled_padded = net.shuttle_encoder(padded, length)
        torch.testing.assert_close(pooled_unpadded, pooled_padded)
