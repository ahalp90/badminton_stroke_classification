"""Unit tests for `shared.video_io`.

Focus is on `write_frame_thumbnail` since the read primitives
(`get_video_info`, `read_frame_at`, `iter_frames`, `read_frames`) are
thin cv2 wrappers exercised implicitly by the rest of the test suite.

A synthetic mp4 is generated once per session via cv2.VideoWriter so
tests are self-contained — no dependency on rsync'd ShuttleSet clips.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from shared.video_io import (
    get_video_info,
    read_frame_at,
    write_frame_thumbnail,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny synthetic video. Each frame is filled with a distinct
# colour so we can verify the right frame was extracted.
# ---------------------------------------------------------------------------
SYNTH_FPS = 30.0
SYNTH_WIDTH = 320
SYNTH_HEIGHT = 240
SYNTH_N_FRAMES = 10


@pytest.fixture(scope='session')
def synth_video(tmp_path_factory) -> Path:
    """Write a 10-frame 320x240 mp4 with each frame a different colour."""
    out = tmp_path_factory.mktemp('video') / 'synth.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out), fourcc, SYNTH_FPS, (SYNTH_WIDTH, SYNTH_HEIGHT))
    for i in range(SYNTH_N_FRAMES):
        # Frame i = (i*25, 100, 200-i*15) in BGR — all distinct, all valid uint8.
        bgr = np.zeros((SYNTH_HEIGHT, SYNTH_WIDTH, 3), dtype=np.uint8)
        bgr[..., 0] = i * 25
        bgr[..., 1] = 100
        bgr[..., 2] = 200 - i * 15
        writer.write(bgr)
    writer.release()
    return out


# ---------------------------------------------------------------------------
# Sanity: the fixture is well-formed.
# ---------------------------------------------------------------------------
class TestSyntheticFixture:
    def test_video_metadata_round_trips(self, synth_video):
        info = get_video_info(synth_video)
        assert info.width == SYNTH_WIDTH
        assert info.height == SYNTH_HEIGHT
        assert info.fps == pytest.approx(SYNTH_FPS, rel=0.01)
        assert info.n_frames == SYNTH_N_FRAMES

    def test_read_frame_at_returns_rgb(self, synth_video):
        frame = read_frame_at(synth_video, 0)
        assert frame.shape == (SYNTH_HEIGHT, SYNTH_WIDTH, 3)
        assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# write_frame_thumbnail
# ---------------------------------------------------------------------------
class TestWriteFrameThumbnail:
    def test_writes_jpg_to_disk(self, synth_video, tmp_path):
        out = tmp_path / 'thumb.jpg'
        result = write_frame_thumbnail(synth_video, frame_idx=3, output_path=out)
        assert result == out
        assert out.exists()
        # Sanity: the file is a readable JPG that opencv can re-open.
        decoded = cv2.imread(str(out))
        assert decoded is not None

    def test_creates_parent_dirs(self, synth_video, tmp_path):
        out = tmp_path / 'nested' / 'a' / 'b' / 'thumb.jpg'
        write_frame_thumbnail(synth_video, frame_idx=0, output_path=out)
        assert out.exists()

    def test_resize_preserves_aspect_ratio(self, synth_video, tmp_path):
        # Source is 320x240. max_width=160 should give 160x120.
        out = tmp_path / 'small.jpg'
        write_frame_thumbnail(synth_video, frame_idx=0, output_path=out, max_width=160)
        decoded = cv2.imread(str(out))
        assert decoded.shape[1] == 160                          # width
        assert decoded.shape[0] == pytest.approx(120, abs=1)    # height (rounding tolerance)

    def test_no_upscale_when_source_smaller_than_max_width(self, synth_video, tmp_path):
        # Source 320 wide, max_width=640 → don't upscale.
        out = tmp_path / 'native.jpg'
        write_frame_thumbnail(synth_video, frame_idx=0, output_path=out, max_width=640)
        decoded = cv2.imread(str(out))
        assert decoded.shape[1] == SYNTH_WIDTH
        assert decoded.shape[0] == SYNTH_HEIGHT

    def test_max_width_none_skips_resize(self, synth_video, tmp_path):
        out = tmp_path / 'native.jpg'
        write_frame_thumbnail(synth_video, frame_idx=0, output_path=out, max_width=None)
        decoded = cv2.imread(str(out))
        assert decoded.shape[1] == SYNTH_WIDTH
        assert decoded.shape[0] == SYNTH_HEIGHT

    def test_returns_path_object(self, synth_video, tmp_path):
        out = tmp_path / 'thumb.jpg'
        result = write_frame_thumbnail(synth_video, 0, out)
        assert isinstance(result, Path)

    def test_extracts_correct_frame(self, synth_video, tmp_path):
        # Frame 5 in our synthetic video: BGR = (125, 100, 125), so dominant
        # blue+red components. Verify by re-reading the thumbnail and
        # checking the channel-mean ordering matches what we encoded.
        out = tmp_path / 'frame5.jpg'
        write_frame_thumbnail(synth_video, frame_idx=5, output_path=out, max_width=None)
        decoded = cv2.imread(str(out))   # BGR
        b, g, r = decoded[..., 0].mean(), decoded[..., 1].mean(), decoded[..., 2].mean()
        # Encoded BGR for frame 5: (125, 100, 125). JPG compression smudges
        # exact values; assert relative ordering: blue ≈ red, both > green.
        assert b == pytest.approx(125, abs=10)
        assert r == pytest.approx(125, abs=10)
        assert g == pytest.approx(100, abs=10)


class TestValidation:
    @pytest.mark.parametrize('max_width', [0, -1, -100])
    def test_invalid_max_width_raises(self, synth_video, tmp_path, max_width):
        with pytest.raises(ValueError, match='max_width'):
            write_frame_thumbnail(synth_video, 0, tmp_path / 'x.jpg', max_width=max_width)

    @pytest.mark.parametrize('quality', [-1, 101, 200])
    def test_invalid_quality_raises(self, synth_video, tmp_path, quality):
        with pytest.raises(ValueError, match='quality'):
            write_frame_thumbnail(synth_video, 0, tmp_path / 'x.jpg', quality=quality)

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            write_frame_thumbnail(tmp_path / 'no-such-file.mp4', 0, tmp_path / 'x.jpg')

    def test_out_of_range_frame_raises(self, synth_video, tmp_path):
        with pytest.raises(ValueError, match='Could not read frame'):
            write_frame_thumbnail(synth_video, frame_idx=999, output_path=tmp_path / 'x.jpg')
