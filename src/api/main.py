import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    ALLOWED_EXTENSIONS,
    CLEANUP_INTERVAL_SECONDS,
    EXPERIMENTS_DIR,
    JOB_TTL_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
)
from .inference import run_inference
from .jobs import JobStatus, JobStore
from .registry import router as registry_router

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
        result = run_inference(video_path, model_name)
        job_store.update(job_id, JobStatus.COMPLETE, result=result)
        log.info("job %s: complete", job_id)
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

    job_store.create(job_id, filename=file.filename, model_name=model, video_path=str(video_path))
    background_tasks.add_task(_process_video, job_id, str(video_path), model)

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
    log.info("job %s: deleted by request", job_id)
    return {"job_id": job_id, "deleted": True}


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
