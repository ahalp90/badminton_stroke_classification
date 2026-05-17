"""Unit tests for `perception.temporal.subsample_indices`.

Covers the invariants we rely on for R(2+1)D pathway construction:
  - output length matches `n`
  - all indices are integers
  - window is centred on `target_frame`
  - stride scales with fps so real-world coverage is fps-invariant
  - `total_frames` clamps without changing length
  - bad inputs raise ValueError
"""

import pytest

from perception.temporal import subsample_indices


def _span(idxs):
    """Inclusive index span (last - first)."""
    return idxs[-1] - idxs[0]


class TestShapeAndType:
    def test_returns_list_of_correct_length(self):
        idxs = subsample_indices(target_frame=100, fps=30.0, coverage_sec=2.0, n=32)
        assert len(idxs) == 32

    def test_all_indices_are_int(self):
        idxs = subsample_indices(target_frame=100, fps=29.97, coverage_sec=1.7, n=17)
        assert all(isinstance(i, int) for i in idxs)

    def test_n_equals_one_returns_target_frame(self):
        assert subsample_indices(target_frame=42, fps=30.0, coverage_sec=2.0, n=1) == [42]


class TestCentringAndCoverage:
    def test_window_centred_on_target_frame_for_odd_n(self):
        idxs = subsample_indices(target_frame=100, fps=30.0, coverage_sec=2.0, n=33)
        assert idxs[len(idxs) // 2] == 100

    def test_window_symmetric_around_target_for_even_n(self):
        # With even n the centre falls between two samples — they should
        # be roughly equidistant from target.
        idxs = subsample_indices(target_frame=100, fps=30.0, coverage_sec=2.0, n=4)
        offsets = [i - 100 for i in idxs]
        assert abs(offsets[0] + offsets[-1]) <= 1   # symmetric within rounding

    @pytest.mark.parametrize(
        "fps,coverage_sec",
        [(25.0, 2.0), (30.0, 2.0), (60.0, 2.0), (29.97, 1.5)],
    )
    def test_real_world_coverage_invariant_to_fps(self, fps, coverage_sec):
        # The frame-span / fps should approximate coverage_sec regardless
        # of source fps. Tolerance accounts for integer rounding.
        idxs = subsample_indices(target_frame=500, fps=fps, coverage_sec=coverage_sec, n=32)
        observed_sec = _span(idxs) / fps
        assert observed_sec == pytest.approx(coverage_sec, abs=1.5 / fps)


class TestFpsScaling:
    def test_higher_fps_yields_larger_frame_span(self):
        # 60 fps over 2s should span ~2x the frames of 30 fps over 2s.
        idxs_30 = subsample_indices(target_frame=500, fps=30.0, coverage_sec=2.0, n=32)
        idxs_60 = subsample_indices(target_frame=500, fps=60.0, coverage_sec=2.0, n=32)
        assert _span(idxs_60) == pytest.approx(2 * _span(idxs_30), rel=0.05)


class TestClamping:
    def test_total_frames_clamps_left_boundary(self):
        # target_frame near 0 with no clamp would give negative indices.
        idxs = subsample_indices(
            target_frame=2, fps=30.0, coverage_sec=2.0, n=32, total_frames=200,
        )
        assert min(idxs) >= 0
        assert len(idxs) == 32   # length preserved

    def test_total_frames_clamps_right_boundary(self):
        idxs = subsample_indices(
            target_frame=198, fps=30.0, coverage_sec=2.0, n=32, total_frames=200,
        )
        assert max(idxs) <= 199
        assert len(idxs) == 32

    def test_no_clamp_allows_out_of_range_indices(self):
        # Without total_frames, caller is responsible — we don't clip.
        idxs = subsample_indices(target_frame=2, fps=30.0, coverage_sec=2.0, n=32)
        assert min(idxs) < 0


class TestValidation:
    @pytest.mark.parametrize("fps", [0, -1.0, -30.0])
    def test_invalid_fps_raises(self, fps):
        with pytest.raises(ValueError, match="fps"):
            subsample_indices(target_frame=10, fps=fps)

    @pytest.mark.parametrize("coverage_sec", [0, -1.0])
    def test_invalid_coverage_raises(self, coverage_sec):
        with pytest.raises(ValueError, match="coverage_sec"):
            subsample_indices(target_frame=10, fps=30.0, coverage_sec=coverage_sec)

    @pytest.mark.parametrize("n", [0, -1, -32])
    def test_invalid_n_raises(self, n):
        with pytest.raises(ValueError, match="n must be"):
            subsample_indices(target_frame=10, fps=30.0, n=n)
