import asyncio
import json
import logging
import os
import uuid
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import (
    ALLOWED_EXTENSIONS,
    CLEANUP_INTERVAL_SECONDS,
    EXPERIMENTS_DIR,
    JOB_TTL_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_MODEL_INPUT_PX,
    UPLOAD_DIR,
    REPO_ROOT,
)
from .inference import run_inference
from .jobs import JobStatus, JobStore
from .registry import router as registry_router


# ─── Markup schema (matches docs/api_contract.md on feat/bric-pipeline) ───
# Subset honoured for v1: the inference stub doesn't *use* these values yet
# (real homography/segmentation lands with the BST/MMPose/TrackNet pipeline),
# but the upload + library_predict endpoints accept, validate, log, and echo
# them so the FE can confirm they reached the server.

class BoundaryPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class StrokeAnnotation(BaseModel):
    target_sec: float = Field(ge=0)
    player_side: Optional[Literal["top", "bottom"]] = None


class Markup(BaseModel):
    architecture: Optional[Literal["bric", "bst", "bst-x"]] = None
    model_id: Optional[str] = None
    orientation: Literal["portrait"] = "portrait"
    video_label: Optional[str] = None
    boundary: Optional[List[BoundaryPoint]] = None
    # Intrinsic source resolution, so the normalised `boundary` can be checked
    # against the model's minimum input size server-side. Optional: when absent
    # we skip the boundary resolution check (the FE warns regardless).
    frame_width: Optional[int] = Field(default=None, ge=1)
    frame_height: Optional[int] = Field(default=None, ge=1)
    annotations: List[StrokeAnnotation] = Field(default_factory=list)
    enabled_sides: List[Literal["top", "bottom"]] = Field(default_factory=lambda: ["top", "bottom"])
    player_top_id: Optional[str] = None
    player_top_label: Optional[str] = None
    player_bottom_id: Optional[str] = None
    player_bottom_label: Optional[str] = None

    @field_validator("boundary")
    @classmethod
    def boundary_is_four_points(cls, v):
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("boundary must contain exactly 4 points")
        return v



class LibraryPredictRequest(BaseModel):
    clip_stem: str
    model_id: Optional[str] = None
    architecture: Optional[Literal["bric", "bst", "bst-x"]] = None
    markup: Optional[Markup] = None


def _parse_markup_json(raw: Optional[str]) -> Optional[dict]:
    if raw is None or raw == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"markup is not valid JSON: {e}")
    try:
        m = Markup.model_validate(parsed)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"markup failed validation: {e.errors()}")
    return m.model_dump(exclude_none=False)


def _resolution_warnings(
    markup_dict: Optional[dict],
    crop_w: Optional[int],
    crop_h: Optional[int],
) -> List[str]:
    """Non-blocking checks that a requested crop meets the model's minimum input
    resolution (MIN_MODEL_INPUT_PX). We warn rather than reject, so the user can
    proceed knowingly instead of silently feeding a degraded crop downstream.

    - The pixel crop (crop_w/crop_h) is checked directly.
    - The normalised court boundary is checked only when the markup carries
      frame_width/frame_height; its xy bounding box is scaled back to pixels.
    """
    warnings: List[str] = []
    m = MIN_MODEL_INPUT_PX

    if crop_w is not None and crop_h is not None and (crop_w < m or crop_h < m):
        warnings.append(
            f"Spatial crop {crop_w}x{crop_h}px is below the {m}x{m}px model input "
            f"minimum; classification quality may degrade."
        )

    if markup_dict:
        boundary = markup_dict.get("boundary")
        fw, fh = markup_dict.get("frame_width"), markup_dict.get("frame_height")
        if boundary and fw and fh:
            xs = [p["x"] for p in boundary]
            ys = [p["y"] for p in boundary]
            bbox_w = round((max(xs) - min(xs)) * fw)
            bbox_h = round((max(ys) - min(ys)) * fh)
            if bbox_w < m or bbox_h < m:
                warnings.append(
                    f"Court boundary crop is ~{bbox_w}x{bbox_h}px on a {fw}x{fh} frame, "
                    f"below the {m}x{m}px model input minimum; classification quality "
                    f"may degrade."
                )
    return warnings


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

job_store = JobStore()


def _cleanup_expired():
    cutoff = datetime.utcnow() - timedelta(seconds=JOB_TTL_SECONDS)
    removed = 0
    for job in job_store.all_jobs():
        if job.created_at < cutoff:
            job_store.delete(job.job_id)
            Path(job.video_path).unlink(missing_ok=True)
            job_dir = REPO_ROOT / "runtime" / "jobs" / job.job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1
    if removed:
        log.info("cleanup: removed %d expired job(s)", removed)


async def _cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        _cleanup_expired()


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(_cleanup_loop())
    log.info(
        "startup: cleanup task started (TTL=%ds, interval=%ds)",
        JOB_TTL_SECONDS,
        CLEANUP_INTERVAL_SECONDS,
    )
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Badminton Stroke Classifier API", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.get("/health")(lambda: {"status": "ok"})

app.include_router(registry_router)


def _process_video(job_id: str, video_path: str, model_name: str):
    log.info("job %s: inference started (model=%s)", job_id, model_name)
    job_store.update(job_id, JobStatus.PROCESSING)
    try:
        job = job_store.get(job_id)
        markup = job.markup if job is not None else None

        result = None
        # Library jobs: try the real BST forward-pass path first. If the
        # stem is in the index AND the SCP'd collated tensors exist, we
        # return the model's actual prediction. Anything missing → smart
        # stub fallback. Upload jobs always go through the smart stub
        # because there's no real ML pipeline for arbitrary user video.
        #
        # Multi-annotation gate: the live BST predict() takes one clip
        # stem and returns one prediction. When the user marks multiple
        # strokes we'd produce a strokes[] of length 1 against
        # annotations[] of length N and trip the FE's count-mismatch
        # warning. Route those through the smart stub, which generates
        # one stroke per annotation, until per-annotation forward passes
        # are wired up.
        annotation_count = len(((markup or {}).get("annotations")) or [])
        if (
            job is not None
            and job.source == "library"
            and job.clip_stem
            and annotation_count <= 1
        ):
            try:
                from .bst_inference import predict as bst_predict, BstInferenceUnavailable
                live = bst_predict(job.clip_stem, split=None)
                # Translate live shape → the same {strokes, rally_summary}
                # envelope the FE already renders.
                annos = ((markup or {}).get("annotations")) or []
                targets = [float(a.get("target_sec") or 0.0) for a in annos
                            if a.get("target_sec") is not None]
                live_rally_length = (
                    round(max(0.0, max(targets) - min(targets)), 1)
                    if len(targets) >= 2 else 0.0
                )
                # Target time for the headline stroke comes from the user's annotation
                # when present; backend re-derives integer frames at inference time
                # from the upload's actual fps.
                live_target_sec = float(annos[0].get("target_sec") or 0.0) if annos else 0.0
                live_timestamp = round(live_target_sec, 2) if live_target_sec else 0.0
                result = {
                    "strokes": [{
                        "timestamp_sec": live_timestamp,
                        "stroke_type":   live["predicted_class"],
                        "confidence":    live["top_k"][0]["confidence"] if live["top_k"] else 0.0,
                        "stroke_index":  0,
                        "target_sec":  live_target_sec,
                        "player_side":   annos[0].get("player_side") if annos else None,
                        "predicted_class": live["predicted_class"],
                        "confidence_pct": live["confidence_pct"],
                        "top_k":          live["top_k"],
                        "true_class_hint": live["true_class"],
                        "drawn_from":     live["drawn_from"],
                    }],
                    "rally_summary": {
                        "total_strokes":         1,
                        "rally_length_seconds":  live_rally_length,
                    },
                    "live_inference": True,
                }
                log.info("job %s: live BST forward pass succeeded for stem=%s", job_id, job.clip_stem)
            except (BstInferenceUnavailable, KeyError, ValueError) as e:
                log.info("job %s: live BST unavailable for stem=%s (%s); falling back to smart stub",
                         job_id, job.clip_stem, e)
            except Exception as e:
                log.exception("job %s: live BST errored for stem=%s; falling back to smart stub: %s",
                              job_id, job.clip_stem, e)

        arch = (markup or {}).get("architecture")
        if result is None and arch == "bric" and job is not None and job.source == "upload":
            try:
                from .bric_inference import classify as bric_classify

                job_dir = REPO_ROOT / "runtime" / "jobs" / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                
                result = bric_classify(Path(video_path), markup, job_dir=job_dir)
                result["live_inference"] = True
                
                manifest = {
                    "job_id": job_id,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "video_filename": Path(video_path).name,
                    "architecture": arch,
                    "markup_input": markup,
                    "result": result,
                }
                (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
                log.info("job %s: live BRIC forward pass succeeded", job_id)
            except ImportError as e:
                log.info("job %s: bric_inference module unavailable (%s); falling back to stub", job_id, e)
            except Exception as e:
                log.exception("job %s: BRIC dispatch failed (%s); falling back to stub", job_id, e)

        if result is None:
            result = run_inference(video_path, model_name, markup=markup)
            result["live_inference"] = False

        # Echo the markup sidecar (Gap 1) + source + clip_stem (Gap 2)
        # into the result envelope so the FE can surface confirmation.
        if job is not None:
            if job.markup is not None:
                result = {**result, "markup_echo": job.markup}
            if job.source == "library" and job.clip_stem is not None:
                result = {**result, "clip_stem": job.clip_stem, "source": "library"}
            elif job.source == "upload":
                result = {**result, "source": "upload"}
        job_store.update(job_id, JobStatus.COMPLETE, result=result)
        log.info("job %s: complete (live=%s)", job_id, result.get("live_inference"))
    except Exception as exc:
        job_store.update(job_id, JobStatus.FAILED, error=str(exc))
        log.error("job %s: failed - %s", job_id, exc)


def _available_models() -> set[str]:
    if not EXPERIMENTS_DIR.exists():
        return {"default"}
    found = {p.stem for p in EXPERIMENTS_DIR.rglob("*.pt")}
    return found | {"default"}


async def _apply_crop(
    video_path: Path,
    start_sec: Optional[float],
    end_sec: Optional[float],
    crop_x: Optional[int],
    crop_y: Optional[int],
    crop_w: Optional[int],
    crop_h: Optional[int],
) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]

    if crop_w is not None:
        cmd += ["-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"]
    elif start_sec is not None or end_sec is not None:
        cmd += ["-c", "copy"]

    tmp = video_path.with_suffix(".tmp" + video_path.suffix)
    cmd.append(str(tmp))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        log.error("ffmpeg crop failed for %s: %s", video_path.name, stderr.decode()[:500])
        raise RuntimeError("Video crop failed - check that the crop parameters are within the video dimensions")

    tmp.replace(video_path)


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # Optional Markup sidecar (see Markup model above). Stringified JSON in
    # the multipart form because FastAPI's multipart parser doesn't accept
    # nested JSON natively. Parsed + validated by _parse_markup_json.
    markup: Optional[str] = Form(default=None),
    model: str = Query(default="default"),
    start_sec: Optional[float] = Query(default=None, ge=0, description="Temporal crop start in seconds"),
    end_sec: Optional[float] = Query(default=None, gt=0, description="Temporal crop end in seconds"),
    crop_x: Optional[int] = Query(default=None, ge=0, description="Spatial crop left edge in pixels"),
    crop_y: Optional[int] = Query(default=None, ge=0, description="Spatial crop top edge in pixels"),
    crop_w: Optional[int] = Query(default=None, gt=0, description="Spatial crop width in pixels"),
    crop_h: Optional[int] = Query(default=None, gt=0, description="Spatial crop height in pixels"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if model not in _available_models():
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'")

    spatial = [crop_x, crop_y, crop_w, crop_h]
    if any(p is not None for p in spatial) and not all(p is not None for p in spatial):
        raise HTTPException(
            status_code=400,
            detail="Spatial crop requires all four parameters: crop_x, crop_y, crop_w, crop_h",
        )

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    with open(video_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                out.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_FILE_SIZE_MB}MB limit",
                )
            out.write(chunk)

    log.info("upload: job=%s file=%s size=%dB model=%s", job_id, file.filename, size, model)

    if any(p is not None for p in [start_sec, end_sec, *spatial]):
        try:
            await _apply_crop(video_path, start_sec, end_sec, crop_x, crop_y, crop_w, crop_h)
            log.info(
                "job %s: crop applied (start=%s end=%s crop_w=%s crop_h=%s)",
                job_id, start_sec, end_sec, crop_w, crop_h,
            )
        except RuntimeError as exc:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc))

    markup_dict = _parse_markup_json(markup)
    if markup_dict is not None:
        log.info(
            "job %s: markup received (boundary=%s annotations=%d arch=%s model_id=%s)",
            job_id,
            "yes" if markup_dict.get("boundary") else "no",
            len(markup_dict.get("annotations") or []),
            markup_dict.get("architecture"),
            markup_dict.get("model_id"),
        )

    # Non-blocking resolution warnings (model input minimum). Surfaced to the FE
    # in the response rather than raising, so the user proceeds knowingly.
    resolution_warnings = _resolution_warnings(markup_dict, crop_w, crop_h)
    for w in resolution_warnings:
        log.warning("job %s: %s", job_id, w)

    job_store.create(
        job_id,
        filename=file.filename,
        model_name=model,
        video_path=str(video_path),
        markup=markup_dict,
        source="upload",
    )
    background_tasks.add_task(_process_video, job_id, str(video_path), model)

    return {"job_id": job_id, "status": "queued", "warnings": resolution_warnings}


@app.post("/api/library_predict")
async def library_predict(req: LibraryPredictRequest, background_tasks: BackgroundTasks):
    """Run inference against a library clip by its identifier.

    Mirrors /api/upload's job lifecycle so the FE can reuse the same
    status + results polling. The stubbed inference still runs from
    inference.py — no real model invocation here. The clip_stem and
    markup are echoed back in /api/results.

    The clip_stem is opaque: if it matches a known dataset stem in the
    registry index, the resolved mp4 is used; otherwise the job still
    runs (the canned stub doesn't read the video), so this endpoint
    also accepts library YouTube identifiers as a stand-in stem for
    demo purposes."""
    from .registry import _build_stem_index  # local import to avoid cycle
    from .config import BST_CLIPS_DIR

    rel_path = _build_stem_index().get(req.clip_stem)
    abs_path: Path
    if rel_path is not None and BST_CLIPS_DIR is not None and (BST_CLIPS_DIR / rel_path).exists():
        abs_path = BST_CLIPS_DIR / rel_path
        resolution = "dataset_clip"
    else:
        # Stand-in: stub inference doesn't read the file, but we still
        # surface a stable, non-existent path on the job so logs make sense.
        abs_path = UPLOAD_DIR / f"library_stub_{req.clip_stem}.mp4"
        resolution = "stub_no_video"

    markup_dict = req.markup.model_dump(exclude_none=False) if req.markup else None
    job_id = str(uuid.uuid4())
    model_name = req.model_id or req.architecture or "default"
    log.info(
        "library_predict: job=%s clip=%s model=%s markup=%s resolution=%s",
        job_id, req.clip_stem, model_name, "yes" if markup_dict else "no", resolution,
    )

    job_store.create(
        job_id,
        filename=f"{req.clip_stem}.mp4",
        model_name=model_name,
        video_path=str(abs_path),
        markup=markup_dict,
        source="library",
        clip_stem=req.clip_stem,
    )
    background_tasks.add_task(_process_video, job_id, str(abs_path), model_name)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job.status}


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Processing failed")
    if job.status != JobStatus.COMPLETE:
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": job.status})
    return {"job_id": job_id, "status": job.status, **job.result}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = job_store.delete(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    Path(job.video_path).unlink(missing_ok=True)
    job_dir = REPO_ROOT / "runtime" / "jobs" / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    log.info("job %s: deleted by request", job_id)
    return {"job_id": job_id, "deleted": True}


@app.get("/api/jobs/{job_id}/strokes/{stroke_idx}/frame/{which}")
async def get_stroke_frame(job_id: str, stroke_idx: int, which: str):
    if which not in {"first", "target", "last"}:
        raise HTTPException(status_code=400, detail="which must be 'first', 'target', or 'last'")
    job_dir = REPO_ROOT / "runtime" / "jobs" / job_id
    if not job_dir.exists():  
        raise HTTPException(status_code=404, detail="job not found")
    matches = list(job_dir.glob(f"stroke_{stroke_idx:02d}_*_{which}.jpg"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"frame not found for stroke {stroke_idx}")
    from fastapi.responses import FileResponse
    return FileResponse(matches[0], media_type="image/jpeg")


@app.get("/api/models")
async def get_models():
    if not EXPERIMENTS_DIR.exists():
        return {"models": []}

    seen = set()
    models = []
    for pt_file in sorted(EXPERIMENTS_DIR.rglob("*.pt")):
        if pt_file.stem not in seen:
            seen.add(pt_file.stem)
            run = pt_file.parts[-3]
            models.append({"run": run, "name": pt_file.stem})

    return {"models": models}
