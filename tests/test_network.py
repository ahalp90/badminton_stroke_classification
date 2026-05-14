"""Unit tests for `bric.network.BRICNetwork`.

Architecture-level tests only — no pretrained weight download (~242MB).
Pretrained loading is exercised by `scripts/bric_smoke_test.py` on the
training host.

Scope: tests cover BRIC's contract — taxonomy resolution, the 512-dim
backbone assumption Day 8 fusion will hardcode, and the forward
signature with reserved fusion slots. Generic torchvision behaviours
(arbitrary batch sizes, fully-conv resolution flexibility, raw-logit
output of nn.Linear) are not retested here.
"""

import torch

from bric.network import BRICNetwork
from shared.taxonomy import DEFAULT_TAXONOMY, TAXONOMIES, TAXONOMY_RAW_35


@pytest.fixture(scope='module')
def model():
    """Random-init network — no pretrained download in unit tests."""
    return BRICNetwork(pretrained=False).eval()


class TestArchitecture:
    def test_default_num_classes(self, model):
        # No taxonomy passed → falls back to shared.DEFAULT_TAXONOMY.
        # Literal 14 is a v1 sanity guard; if it fires because the
        # default taxonomy changed, update both sides intentionally.
        default_tax = TAXONOMIES[DEFAULT_TAXONOMY]
        assert model.num_classes == default_tax.n_trainable_classes == 14
        assert model.taxonomy is default_tax

    def test_explicit_taxonomy_sizes_head(self):
        # Production path: caller passes a taxonomy, head sizes itself.
        net = BRICNetwork(taxonomy=TAXONOMY_RAW_35, pretrained=False)
        assert net.num_classes == TAXONOMY_RAW_35.n_trainable_classes
        assert net.backbone.fc.out_features == net.num_classes
        assert net.taxonomy is TAXONOMY_RAW_35

    def test_explicit_num_classes_overrides(self):
        # Escape hatch for tests/experiments. Bypasses the taxonomy.
        net = BRICNetwork(num_classes=20, pretrained=False)
        assert net.num_classes == 20
        assert net.backbone.fc.out_features == 20
        assert net.taxonomy is None

    def test_backbone_penultimate_is_512(self, model):
        # R(2+1)D-18's penultimate dim. If a future backbone swap
        # changes this, downstream Day-8 fusion code that hardcodes
        # 512 needs review.
        assert model.backbone_dim == 512


class TestForwardPass:
    def test_e2e_at_dataset_resolution(self, model):
        # End-to-end smoke at the actual dataset resolution (Day-3
        # dataset feeds 224×224 player crops). One run is enough —
        # batching and resolution flexibility are torchvision concerns.
        rgb = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            out = model(rgb)
        assert out.shape == (1, model.num_classes)

    def test_accepts_auxiliary_inputs_silently(self, model):
        # v1 ignores shuttle/court but the call shouldn't raise — Day 8
        # will plumb these through; calling code shouldn't need updates
        # when fusion lands.
        rgb = torch.randn(1, 3, 32, 112, 112)
        shuttle = torch.randn(1, 50, 3)
        court = torch.randn(1, 50, 2)
        with torch.no_grad():
            out = model(rgb, shuttle_feats=shuttle, court_feats=court)
        assert out.shape == (1, model.num_classes)
