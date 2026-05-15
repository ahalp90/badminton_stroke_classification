import time


def run_inference(video_path: str, model_name: str) -> dict:
    """
    Run stroke classification inference on a video file.

    This is a stub. The ML team should replace the body of this function.

    Args:
        video_path: absolute path to the uploaded video file on disk
        model_name: name of the checkpoint to use (stem of the .pt file)

    Returns:
        dict with keys:
            strokes:       list of {timestamp_sec, stroke_type, confidence}
            rally_summary: {total_strokes, rally_length_seconds}

    Expected pipeline when implemented:
        1. Extract clips from video (pipeline/clip_generator.py)
        2. Run pose estimation per clip (MMPose)
        3. Project player positions to court coordinates (homography)
        4. Run shuttle tracking (TrackNetV3)
        5. Collate features into BST input tensors
        6. Load checkpoint from runtime/deployed/<arch>/ and run inference
        7. Map predicted class indices back to stroke label strings
        8. Return results in the format below
    """
    # Simulate processing time so the frontend exercises the "processing" poll state
    time.sleep(3)

    return {
        "strokes": [
            {"timestamp_sec": 2.1, "stroke_type": "clear", "confidence": 0.92},
            {"timestamp_sec": 8.4, "stroke_type": "smash", "confidence": 0.88},
        ],
        "rally_summary": {
            "total_strokes": 2,
            "rally_length_seconds": 12.5,
        },
    }
