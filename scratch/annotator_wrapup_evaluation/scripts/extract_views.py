"""Render neutral nine-frame sheets around recorded source frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
LABEL_HEIGHT = 32
OFFSETS_SECONDS = (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)


def run(sample: Path, sources: Path, output: Path) -> None:
    rows = pd.read_csv(sample)
    output.mkdir(parents=True, exist_ok=True)
    for fixture, group in rows.groupby("fixture", sort=False):
        videos = list(sources.glob(f"{int(fixture):02d} *.mp4"))
        assert len(videos) == 1, (fixture, videos)
        capture = cv2.VideoCapture(str(videos[0]))
        assert capture.isOpened(), videos[0]
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for row in group.itertuples(index=False):
            assert abs(fps - row.fps) < 0.01, (fps, row.fps)
            sheet = np.full((3 * (FRAME_HEIGHT + LABEL_HEIGHT), 3 * FRAME_WIDTH, 3), 255, dtype=np.uint8)
            for index, offset in enumerate(OFFSETS_SECONDS):
                frame = min(frame_count - 1, max(0, round(row.source_frame + offset * fps)))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                success, pixels = capture.read()
                assert success, (fixture, frame)
                assert abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) < 1, (fixture, frame)
                if offset == 0:
                    assert cv2.imwrite(str(output / f"{row.sample_id}_centre.jpg"), pixels,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
                top = (index // 3) * (FRAME_HEIGHT + LABEL_HEIGHT)
                left = (index % 3) * FRAME_WIDTH
                sheet[top:top + FRAME_HEIGHT, left:left + FRAME_WIDTH] = cv2.resize(
                    pixels, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA,
                )
                label = f"{row.sample_id} | frame {frame} | {frame / fps:.2f} s"
                cv2.putText(sheet, label, (left + 8, top + FRAME_HEIGHT + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            assert cv2.imwrite(str(output / f"{row.sample_id}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(row.sample_id, "complete", flush=True)
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.sample, args.sources, args.output)
