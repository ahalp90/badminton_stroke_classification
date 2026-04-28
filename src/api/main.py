import os
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import ALLOWED_EXTENSIONS, EXPERIMENTS_DIR, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, UPLOAD_DIR
from .inference import run_inference
from .jobs import JobStatus, JobStore

app = FastAPI(title="Badminton Stroke Classifier API")

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_store = JobStore()


def _process_video(job_id: str, video_path: str, model_name: str):
    job_store.update(job_id, JobStatus.PROCESSING)
    try:
        result = run_inference(video_path, model_name)
        job_store.update(job_id, JobStatus.COMPLETE, result=result)
    except Exception as exc:
        job_store.update(job_id, JobStatus.FAILED, error=str(exc))


def _available_models() -> set[str]:
    if not EXPERIMENTS_DIR.exists():
        return {"default"}
    found = {p.stem for p in EXPERIMENTS_DIR.rglob("*.pt")}
    return found | {"default"}


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Query(default="default"),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if model not in _available_models():
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'")

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    # Stream upload to disk in 1MB chunks to avoid loading large videos into memory
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

    job_store.create(job_id, filename=file.filename, model_name=model)
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


@app.get("/api/models")
async def get_models():
    if not EXPERIMENTS_DIR.exists():
        return {"models": []}

    models = []
    for pt_file in sorted(EXPERIMENTS_DIR.rglob("*.pt")):
        run = pt_file.parts[-3]
        # Omit the internal file path - callers only need run + name to select a model
        models.append({"run": run, "name": pt_file.stem})

    return {"models": models}
