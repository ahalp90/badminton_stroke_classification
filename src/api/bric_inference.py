"""BRIC live inference for uploaded video.

Singleton pattern (mirrors bst_inference.py): module-level model + YOLO
are loaded once at first request via _ensure_initialised(), then reused.
"""
from __future__ import annotations
from pathlib import Path
from threading import Lock

import sys
import logging
import yaml
import cv2
import tempfile
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
TRACKNETV3_DIR = SRC_DIR / "bric" / "perception" / "_vendor" / "tracknetv3"
for p in (SRC_DIR, TRACKNETV3_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ultralytics import YOLO
from bric.dataset import _to_half_court
from bric.network import BRICNetwork
from bric.perception.players import DEFAULT_YOLO_WEIGHTS
from bric.preprocessing.preprocess_videos import detect_persons, pick_striker_detection, extract_stroke_rgb
from shared.taxonomy import TAXONOMIES
from shared.dataset import compute_clip_bounds
from shared.court import (
      REF_COURT_CORNERS_M,
      HOMOGRAPHY_RESOLUTION,
      convert_homogeneous,
      scale_pos_by_resolution,
      project,
  )

from bric.perception._vendor.tracknetv3.predict import (
    load_models as load_tracknet_models,
    predict_video as tracknet_predict_video,
)

TRACKNET_WEIGHTS = REPO_ROOT / "runtime" / "checkpoints" / "tracknetv3" / "TrackNet_best.pt"
INPAINTNET_WEIGHTS = REPO_ROOT / "runtime" / "checkpoints" / "tracknetv3" / "InpaintNet_best.pt"

_tracknet = None
_inpaintnet = None
_tracknet_seq_len = None
_inpaintnet_seq_len = None
_tracknet_bg_mode = None
_shuttle_window: str | None = None

log = logging.getLogger(__name__)

RUN_DIR = REPO_ROOT / "runtime" / "deployed" / "bric" / "20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42"
WEIGHTS_PATH = RUN_DIR / "best.pt"
MANIFEST_PATH = RUN_DIR / "manifest.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RGB_N_FRAMES = 32
RGB_N_BEFORE = 16

_model: torch.nn.Module | None = None
_yolo = None
_class_list: list[str] = []
_model_lock = Lock()


def _dump_rgb_clip(clip: np.ndarray, out_path: Path, fps: float) -> None:
    """Save a (T, H, W, 3) RGB uint8 array as mp4 (cv2 wants BGR)."""
    if clip.dtype != np.uint8:
        clip = clip.astype(np.uint8)
    T, H, W, _ = clip.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    for frame in clip:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _dump_window_clip(
    rally_frames: dict[int, np.ndarray],
    start_f: int,
    end_f: int,
    out_path: Path,
    fps: float,
) -> None:
    """Save the [start_f, end_f] rally slice as full-frame mp4."""
    frames = [rally_frames[f] for f in range(start_f, end_f + 1) if f in rally_frames]
    if not frames:  
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _striker_court_position(
    striker_arrays: dict[str, np.ndarray],
    court_info: dict,
    target_f: int,  
    side: str,
    frame_w: float,
    frame_h: float,
    smooth_radius: int = 16,
) -> tuple[float, float] | None:
    """Project the striker's foot-centre at ``target_f`` to half-court coords.

    Walks ±smooth_radius frames to find a valid bbox if target_f's is missing.
    Mirrors src/bric/dataset.py::_build_court_snapshot semantics.
    """
    side_lc = side.lower()
    valid = striker_arrays[f"{side_lc}_valid"]
    bboxes = striker_arrays[f"{side_lc}_bbox"]
    n = len(valid)

    bbox = None
    for offset in range(smooth_radius + 1):
        candidates = (target_f,) if offset == 0 else (target_f - offset, target_f + offset)
        for f in candidates:
            if 0 <= f < n and valid[f]:
                bbox = bboxes[f]
                break
        if bbox is not None:
            break
    if bbox is None:
        return None

    foot_x = (float(bbox[0]) + float(bbox[2])) / 2
    foot_y = float(bbox[3])
    pt = np.array([[foot_x], [foot_y]])
    pt = scale_pos_by_resolution(pt, width=frame_w, height=frame_h)
    pt = convert_homogeneous(pt) 
    court_pt = project(court_info["H"], pt)
    x_n = (court_pt[0, 0] - court_info["border_L"]) / (
        court_info["border_R"] - court_info["border_L"]
    )
    y_n = (court_pt[1, 0] - court_info["border_U"]) / (
        court_info["border_D"] - court_info["border_U"]
    )
    return _to_half_court(float(x_n), float(y_n), side_lc)


def _ensure_initialised():
    """Load BRIC weights + YOLO once."""
    global _model, _class_list, _yolo
    global _tracknet, _inpaintnet, _tracknet_seq_len, _inpaintnet_seq_len, _tracknet_bg_mode
    global _shuttle_window
    with _model_lock:
        if _model is not None:
            return

        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f) or {}

        config = manifest.get("config", {})
        hparams = manifest.get("training", {}).get("hparams", {})

        taxonomy_name = config.get("taxonomy", "une_merge_v1_nosides")
        use_shuttle = config.get("use_shuttle", False)
        use_court = config.get("use_court", False)
        shuttle_encoder = hparams.get("shuttle_encoder", "tcn")
        _class_list = config.get("classes", [])
        _shuttle_window = hparams.get("shuttle_window") or "outgoing_only"
        
        net = BRICNetwork(
            taxonomy=TAXONOMIES[taxonomy_name],
            pretrained=False,
            use_shuttle=use_shuttle,
            use_court=use_court,
            shuttle_encoder=shuttle_encoder,
        )

        checkpoint = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        net = net.to(DEVICE)
        _model = net

        log.info(
            "bric_inference: loaded %s (%.1f MB) on %s",
            WEIGHTS_PATH.name, WEIGHTS_PATH.stat().st_size / 1e6, DEVICE,
        )

        _yolo = YOLO(str(DEFAULT_YOLO_WEIGHTS))
        log.info("bric_inference: YOLO loaded from %s", DEFAULT_YOLO_WEIGHTS.name)

        if use_shuttle:
            (
                _tracknet,
                _inpaintnet,
                _tracknet_seq_len,
                _inpaintnet_seq_len,
                _tracknet_bg_mode,
            ) = load_tracknet_models(str(TRACKNET_WEIGHTS), str(INPAINTNET_WEIGHTS))
            log.info(
                "bric_inference: TrackNet loaded (seq_len=%d, bg_mode=%s, inpaint=%s)",
                _tracknet_seq_len, _tracknet_bg_mode, _inpaintnet is not None,
            )
        else:
            log.info("bric_inference: TrackNet load skipped (use_shuttle=False)")


class BricInferenceUnavailable(Exception):
    """Raised when real BRIC inference can't proceed (missing weights, bad input).

    The API layer catches this and falls back to the smart stub."""


def classify(
    video_path: Path, 
    markup: dict, 
    job_dir: Path | None = None,
) -> dict:
    """End-to-end inference on a single uploaded rally video.

    1. Open cv2 cap, read fps.
    2. Compute per-stroke windows via compute_clip_bounds.
    3. Decode rally frames in the union window.
    4. Run YOLO → striker_arrays.
    5. Per stroke: extract_stroke_rgb → RGB tensor.
    6. Run TrackNet → shuttle trace; slice per stroke.
    7. BRIC forward pass (batched).
    8. Return {strokes, rally_summary, live_inference: True}.
    """
    _ensure_initialised()

    annotations = markup.get("annotations") or []
    if not annotations:
        raise BricInferenceUnavailable("no annotations in markup")

    # Step 1: Open video, read fps
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise BricInferenceUnavailable(f"cv2 failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log.info("bric_inference: video=%s fps=%.2f frames=%d annotations=%d",
        video_path.name, fps, frame_count, len(annotations))

    # Step 2: Per-stroke target frames and windows
    targets_sec = sorted(float(a["target_sec"]) for a in annotations)
    target_frames = [round(t * fps) for t in targets_sec]

    stroke_windows = []
    for i, tf in enumerate(target_frames):
        prev_f = target_frames[i - 1] if i > 0 else -1
        next_f = target_frames[i + 1] if i < len(target_frames) - 1 else -1
        row = {"frame_num": tf, "start_f": prev_f, "end_f": next_f}
        start_f, end_f = compute_clip_bounds(row, "between_2_hits_with_max_limits", fps)
        stroke_windows.append((start_f, tf, end_f))
    log.info("bric_inference: stroke windows (start,target,end frames): %s", stroke_windows)

    # Step 3: Decode all frames spanning the rally (first start_f → last end_f)
    union_start = max(0, min(s for s, _, _ in stroke_windows))
    union_end = min(frame_count - 1, max(e for _, _, e in stroke_windows))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise BricInferenceUnavailable(f"cv2 failed to re-open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, union_start)
    rally_frames: dict[int, np.ndarray] = {}
    f_idx = union_start
    while f_idx <= union_end: 
        ret, frame = cap.read()
        if not ret:
            break
        rally_frames[f_idx] = frame
        f_idx += 1
    cap.release()
    log.info(
        "bric_inference: decoded %d frames (rally range [%d, %d])",
        len(rally_frames), union_start, union_end,
    )
    
    # Step 4: Run YOLO player detection
    per_frame_detections: dict[int, list[dict]] = {}
    for f_idx, frame in rally_frames.items():
        per_frame_detections[f_idx] = detect_persons(_yolo, frame)
    log.info(
        "bric_inference: YOLO ran on %d frames; avg %.1f detections/frame",
        len(per_frame_detections),
        sum(len(d) for d in per_frame_detections.values()) / max(1, len(per_frame_detections)),
    )

    boundary_norm = markup.get("boundary") or []
    if len(boundary_norm) != 4:
        raise BricInferenceUnavailable("markup missing 4-corner boundary")

    hres_w, hres_h = HOMOGRAPHY_RESOLUTION
    boundary_px = np.array(
        [(p["x"] * hres_w, p["y"] * hres_h) for p in boundary_norm],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(boundary_px, REF_COURT_CORNERS_M)
    court_info = {
        "H": H,
        "border_L": 0.0,
        "border_R": float(REF_COURT_CORNERS_M[1, 0]),  # 13.4 m
        "border_U": 0.0,
        "border_D": float(REF_COURT_CORNERS_M[2, 1]),  # 6.1 m
    }

    sample = next(iter(rally_frames.values()))
    frame_h, frame_w = sample.shape[:2]
    n = frame_count
    top_bbox = np.zeros((n, 4), dtype=np.float32)
    bottom_bbox = np.zeros((n, 4), dtype=np.float32)
    top_valid = np.zeros(n, dtype=bool)
    bottom_valid = np.zeros(n, dtype=bool)
    top_conf = np.zeros(n, dtype=np.float32)
    bottom_conf = np.zeros(n, dtype=np.float32)

    for f_idx, detections in per_frame_detections.items():
        for side, bbox_arr, valid_arr, conf_arr in [
            ("top", top_bbox, top_valid, top_conf),
            ("bottom", bottom_bbox, bottom_valid, bottom_conf),
        ]:
            pick = pick_striker_detection(detections, side, frame_w, frame_h, court_info)
            if pick is not None:
                bbox_arr[f_idx] = pick["bbox"]
                valid_arr[f_idx] = True
                conf_arr[f_idx] = pick["conf"]

    striker_arrays = {
        "top_bbox": top_bbox,    "bottom_bbox": bottom_bbox,
        "top_valid": top_valid,  "bottom_valid": bottom_valid,
        "top_conf": top_conf,    "bottom_conf": bottom_conf,
    }
    log.info(
        "bric_inference: striker_arrays — top_valid=%d/%d, bottom_valid=%d/%d frames",
        int(top_valid.sum()), len(per_frame_detections),
        int(bottom_valid.sum()), len(per_frame_detections),
    )

    # Step 5: Extract per-stroke RGB clips
    rgb_clips: list[np.ndarray] = []
    for idx, (start_f, target_f, end_f) in enumerate(stroke_windows):
        side = annotations[idx].get("player_side")
        if side not in ("top", "bottom"):
            raise BricInferenceUnavailable(
                f"annotation {idx} missing/invalid player_side={side!r}"
            )
        stroke_series = pd.Series({
            "frame_num": target_f,
            "player_side": side, 
        })
        clip, used_offset = extract_stroke_rgb(
            stroke=stroke_series,
            rally_frames=rally_frames,
            striker_arrays=striker_arrays,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        if clip is None:
            raise BricInferenceUnavailable(
                f"stroke {idx} (target_f={target_f}, side={side}): "
                f"no striker bbox within ±{RGB_N_BEFORE} frames"
            )
        rgb_clips.append(clip)
        log.info(
            "bric_inference: stroke %d (%s, target_f=%d) RGB clip shape=%s dtype=%s, offset=%d",
            idx, side, target_f, clip.shape, clip.dtype, used_offset,
        )
        if job_dir is not None:
            _dump_rgb_clip(
                clip,
                job_dir / f"stroke_{idx:02d}_{side}_t{target_f}_tight.mp4",
                fps=fps,
            )
            _dump_window_clip(
                rally_frames,
                start_f=max(0, start_f),
                end_f=min(frame_count - 1, end_f),
                out_path=job_dir / f"stroke_{idx:02d}_{side}_t{target_f}_window.mp4",
                fps=fps,
            )

    log.info("bric_inference: extracted %d RGB clips", len(rgb_clips))

    # Step 6: TrackNet shuttle tracing
    
    shuttle_tensors: list[np.ndarray] = []
    if _model.use_shuttle: 

        tracknet_save_dir = job_dir if job_dir is not None else Path(tempfile.mkdtemp())
        tracknet_save_dir.mkdir(parents=True, exist_ok=True)

        tracknet_predict_video(   
            video_file=str(video_path),
            tracknet=_tracknet,
            inpaintnet=_inpaintnet,
            tracknet_seq_len=_tracknet_seq_len,
            inpaintnet_seq_len=_inpaintnet_seq_len,
            bg_mode=_tracknet_bg_mode,
            save_dir=str(tracknet_save_dir),
        )

        # TrackNet writes {video_stem}_ball.csv with columns: Frame, Visibility, X, Y
        csv_path = tracknet_save_dir / f"{video_path.stem}_ball.csv"
        if not csv_path.exists():
            raise BricInferenceUnavailable(f"TrackNet CSV not written: {csv_path}")

        shuttle_df = pd.read_csv(csv_path)
        shuttle_x = shuttle_df["X"].to_numpy(dtype=np.float32)
        shuttle_y = shuttle_df["Y"].to_numpy(dtype=np.float32)
        shuttle_v = shuttle_df["Visibility"].to_numpy(dtype=np.float32)
        log.info(
            "bric_inference: TrackNet produced %d frames, visibility=%.2f",
            len(shuttle_x), float(shuttle_v.mean()) if len(shuttle_v) else 0.0,
        )

        # Per-stroke slicing: outgoing_only → [target_f, end_f) in Python half-open form,
        # matching src/bric/dataset.py::_build_shuttle exactly.
        
        for idx, (start_f, target_f, end_f) in enumerate(stroke_windows):
            if _shuttle_window == "outgoing_only":
                a = max(0, target_f)
            elif _shuttle_window == "between_hits":
                a = max(0, start_f)
            else:
                raise BricInferenceUnavailable(f"unknown shuttle_window={_shuttle_window!r}")
            b = min(len(shuttle_x), end_f)

            x = shuttle_x[a:b] / frame_w
            y = shuttle_y[a:b] / frame_h
            v = shuttle_v[a:b]

            t = x.shape[0]
            dx = np.zeros(t, dtype=np.float32)
            dy = np.zeros(t, dtype=np.float32)
            if t > 1:
                valid_pair = (v[1:] > 0) & (v[:-1] > 0)
                dx[1:] = (x[1:] - x[:-1]) * valid_pair
                dy[1:] = (y[1:] - y[:-1]) * valid_pair

            tensor = np.stack([x, y, v, dx, dy], axis=-1)
            shuttle_tensors.append(tensor)
            log.info(
                "bric_inference: stroke %d shuttle [%d, %d) shape=%s visibility=%.2f",
                idx, a, b, tensor.shape, float(v.mean()) if t > 0 else 0.0,
            )

        log.info("bric_inference: built %d shuttle tensors", len(shuttle_tensors))
    else:
        log.info("bric_inference: shuttle path skipped (use_shuttle=False)")

    # Step 7: Forward pass
    # Mirror src/bric/dataset.py::_build_rgb normalization exactly.
    _RGB_MEAN_VAL = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1, 1)
    _RGB_STD_VAL = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1, 1)

    rgb_torch: list[torch.Tensor] = []
    for clip in rgb_clips:
        cthw = clip.transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        cthw = (cthw - _RGB_MEAN_VAL) / _RGB_STD_VAL
        rgb_torch.append(torch.from_numpy(cthw))
    rgb_batch = torch.stack(rgb_torch, dim=0).to(DEVICE, non_blocking=True)  # (N, 3, 32, 224, 224)

    # Mirror src/bric/train.py::_forward_for_variant — build kwargs based on
    # which modalities this model uses. Adapts to RGB-only / RGB+shuttle /
    # RGB+shuttle+court variants without code changes.
    forward_kwargs: dict = {} 
    if _model.use_shuttle:
        shuttle_torch = [torch.from_numpy(s) for s in shuttle_tensors]
        shuttle_padded = torch.nn.utils.rnn.pad_sequence(shuttle_torch, batch_first=True)
        shuttle_length = torch.tensor([s.shape[0] for s in shuttle_tensors], dtype=torch.long)
        forward_kwargs["shuttle"] = shuttle_padded.to(DEVICE, non_blocking=True)
        forward_kwargs["shuttle_length"] = shuttle_length.to(DEVICE, non_blocking=True)

    with torch.no_grad():
        logits = _model(rgb_batch, **forward_kwargs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, n_class)

    top_k = 5
    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
    log.info("bric_inference: forward pass complete, %d strokes predicted", probs.shape[0])

    # Step 8: Return response envelope
    strokes = []
    for idx in range(len(stroke_windows)):
        target_sec = float(annotations[idx].get("target_sec") or 0.0)
        side = annotations[idx].get("player_side")
        target_f = stroke_windows[idx][1]

        court_pos = _striker_court_position(
            striker_arrays, court_info,
            target_f=target_f, side=side,
            frame_w=frame_w, frame_h=frame_h,
        )

        pred_idx = int(top_idx[idx, 0])
        pred_class = _class_list[pred_idx]
        confidence_pct = int(round(float(probs[idx, pred_idx]) * 100))
        top_k_entries = [
            {"class": _class_list[int(i)], "confidence": round(float(probs[idx, int(i)]), 4)}
            for i in top_idx[idx]
        ] 
        strokes.append({
            "timestamp_sec":   round(target_sec, 2),
            "stroke_type":     pred_class,
            "confidence":      top_k_entries[0]["confidence"],
            "stroke_index":    idx,
            "target_sec":      target_sec,
            "player_side":     side,
            "predicted_class": pred_class,
            "confidence_pct":  confidence_pct,
            "top_k":           top_k_entries,
            "softmax":         [round(float(p), 4) for p in probs[idx].tolist()],
            "court_position":  (
                {"x_half": round(court_pos[0], 4), "y_half": round(court_pos[1], 4)}
                if court_pos is not None else None),
            "drawn_from":      "live_forward_pass",
        })

    targets_for_rally = sorted(float(a.get("target_sec") or 0.0) for a in annotations)
    rally_length = (
        round(max(0.0, targets_for_rally[-1] - targets_for_rally[0]), 1)
        if len(targets_for_rally) >= 2 else 0.0
    )

    result = {
        "strokes": strokes,
        "rally_summary": {
            "total_strokes":        len(strokes),
            "rally_length_seconds": rally_length,
        },
    }
    log.info(
        "bric_inference: returning %d predictions: %s",
        len(strokes),
        [(s["predicted_class"], s["confidence_pct"]) for s in strokes],
    )
    return result 