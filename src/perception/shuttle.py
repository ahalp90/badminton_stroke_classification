"""TrackNetV3 shuttle extraction wrapper.

Thin subprocess wrapper around the vendored TrackNetV3's ``predict.py``
CLI. Returns the path to the output CSV (one row per source-video frame
with ``Frame, Visibility, X, Y`` columns).

Why subprocess: the vendored ``predict.py`` uses absolute imports
(``from inference_utils import ...``) that assume cwd is the vendor
directory. Importing it in-process would require sys.path manipulation
plus star-import side effects from ``utils.general``. Subprocess from
the right cwd is cleaner and matches BST's usage pattern.

Usage:
    csv_path = extract_shuttle(
        video_path=Path('training/data/shuttleset/raw_video/1 ....mp4'),
        save_dir=Path('training/bric/cache/tmp_shuttle/1'),
    )
    df = pd.read_csv(csv_path)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Vendor lives at src/perception/_vendor/tracknetv3/.
_VENDOR_DIR = Path(__file__).resolve().parent / '_vendor' / 'tracknetv3'

# Default weights live at runtime/checkpoints/tracknetv3/.
# Resolved from this file: src/perception/shuttle.py → parents[2] = repo root.
_DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[2]
    / 'runtime' / 'checkpoints' / 'tracknetv3'
)


def extract_shuttle(
    video_path: Path,
    save_dir: Path,
    weights_dir: Path = _DEFAULT_WEIGHTS_DIR,
    batch_size: int = 16,
) -> Path:
    """Run TrackNetV3 + InpaintNet on a video; return path to the output CSV.

    Always uses ``--large_video`` mode (streaming dataset; safe on full
    1-2 hr match videos). Skips the overlay-video output — we only need
    the CSV.

    :param video_path: Path to source mp4.
    :param save_dir: Directory to write the output CSV into. Created if
        missing.
    :param weights_dir: Directory containing ``TrackNet_best.pt`` and
        ``InpaintNet_best.pt``. Defaults to
        ``runtime/checkpoints/tracknetv3/``.
    :param batch_size: Inference batch size (passed to predict.py).
    :return: Path to ``<save_dir>/<video_stem>_ball.csv``.
    :raises FileNotFoundError: if weights or video are missing, or if
        the expected output CSV is not produced.
    :raises subprocess.CalledProcessError: if predict.py exits non-zero.
    """
    video_path = Path(video_path).resolve()
    save_dir = Path(save_dir).resolve()
    weights_dir = Path(weights_dir).resolve()

    tracknet_weights = weights_dir / 'TrackNet_best.pt'
    inpaintnet_weights = weights_dir / 'InpaintNet_best.pt'
    if not tracknet_weights.exists():
        raise FileNotFoundError(f'TrackNet weights not found: {tracknet_weights}')
    if not inpaintnet_weights.exists():
        raise FileNotFoundError(f'InpaintNet weights not found: {inpaintnet_weights}')
    if not video_path.exists():
        raise FileNotFoundError(f'Video not found: {video_path}')

    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'predict.py',
        '--video_file', str(video_path),
        '--tracknet_file', str(tracknet_weights),
        '--inpaintnet_file', str(inpaintnet_weights),
        '--save_dir', str(save_dir),
        '--batch_size', str(batch_size),
        '--large_video',
    ]
    subprocess.run(cmd, cwd=_VENDOR_DIR, check=True)

    out_csv = save_dir / f'{video_path.stem}_ball.csv'
    if not out_csv.exists():
        raise FileNotFoundError(
            f'TrackNet completed but expected CSV not found at {out_csv}'
        )
    return out_csv
