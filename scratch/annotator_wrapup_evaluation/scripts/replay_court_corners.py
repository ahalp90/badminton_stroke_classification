"""Recover the neural-net positions before the saved video 53 fallback."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from courtkeynet.court_corners import (
    _geometry_clean,
    _scene_sample_quad,
    pick_scene_corners,
)
from courtkeynet.wrapper import DEFAULT_WEIGHTS, CourtKeyNetDetector


def run(prepared: Path, sources: Path, output: Path) -> None:
    folder, = prepared.glob("53 *")
    video, = sources.glob("53 *.mp4")
    with gzip.open(folder / "court_receipt.json.gz", "rt") as source:
        receipt = json.load(source)
    with gzip.open(folder / "court_evidence.json.gz", "rt") as source:
        scene = json.load(source)["scene_records"][334]
    weights_md5 = hashlib.md5(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
    assert weights_md5 == receipt["model"]["md5"]
    assert scene["start_frame"] <= 83084 < scene["end_frame"]
    frames = []
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened()
        for frame in scene["sampled_frame_indices"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, pixels = capture.read()
            assert success and abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) < 1
            frames.append(pixels)
    finally:
        capture.release()
    config = receipt["configuration"]
    detector = CourtKeyNetDetector(device=config["device"], resize_mode=config["resize_mode"])
    detections = detector.detect_batch(frames)
    recovered = pick_scene_corners(frames, detections, corner_min_peak_conf=detector.corner_min_peak_conf)
    assert recovered is not None
    assert recovered.source == scene["raw_source"]
    assert list(recovered.corner_source) == scene["raw_corner_source"]
    # GPU replay can vary below a pixel; record the measured difference below.
    np.testing.assert_allclose(recovered.corners_px, scene["raw_corners_px"], rtol=0, atol=0.05)
    np.testing.assert_allclose(recovered.peak, scene["raw_peaks"], rtol=0.005, atol=0)
    np.testing.assert_array_equal(recovered.peak >= detector.corner_min_peak_conf,
                                  np.asarray(scene["raw_peaks"]) >= detector.corner_min_peak_conf)
    clean = [detection for detection in detections if _geometry_clean(detection)]
    frame_records = []
    for frame, detection in zip(scene["sampled_frame_indices"], detections, strict=True):
        frame_records.append({"source_frame": frame, "corners_px": detection.corners_px.tolist(),
                              "peak": detection.peak.tolist(), "flags": list(detection.flags)})
    result = {
        "fixture": 53, "scene_index": 334, "display_frame": 83084,
        "corner_order": ["0 top-left", "1 top-right", "2 bottom-right", "3 bottom-left"],
        "weights_md5": weights_md5, "device": config["device"], "resize_mode": config["resize_mode"],
        "corner_min_peak_conf": detector.corner_min_peak_conf,
        "frames": frame_records, "geometry_clean_frames": len(clean),
        "fully_passing_frames": sum(detection.passed for detection in detections),
        "nn_median_corners_px": _scene_sample_quad(clean).tolist(),
        "saved_scene": scene,
        "replayed_fallback_corners_px": recovered.corners_px.tolist(),
        "max_saved_peak_difference": float(np.max(np.abs(recovered.peak - scene["raw_peaks"]))),
        "max_saved_corner_difference_px": float(np.max(np.abs(recovered.corners_px - scene["raw_corners_px"]))),
        "note": "NN median uses all shape-valid sampled frames before confidence filtering. It is not an accepted court.",
    }
    with gzip.open(output, "wt") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps({key: result[key] for key in (
        "geometry_clean_frames", "fully_passing_frames", "nn_median_corners_px", "max_saved_corner_difference_px",
    )}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.prepared_root, arguments.sources, arguments.output)
