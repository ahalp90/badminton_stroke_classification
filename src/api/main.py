from fastapi import FastAPI, UploadFile, File
import uuid

app = FastAPI(title="Badminton Stroke Classifier API")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file and return a job ID."""
    # We generate a random UUID to act as our mock job ID
    job_id = str(uuid.uuid4())
    return {"job_id": job_id, "message": f"Successfully received {file.filename}"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Return job status (queued/processing/complete)."""
    # Hardcoded to 'processing' for now so Kiri can build loading states
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Return classification results (mock JSON)."""
    # Returning sensible mock data so the front-end has something to render
    return {
        "job_id": job_id,
        "status": "complete",
        "rally_summary": {"total_strokes": 2, "rally_length_seconds": 12.5},
        "strokes": [
            {"timestamp_sec": 2.1, "stroke_type": "clear", "confidence": 0.92},
            {"timestamp_sec": 8.4, "stroke_type": "smash", "confidence": 0.88},
        ],
    }


@app.get("/api/models")
async def get_models():
    """List available models."""
    # Hardcoded list: Model A, Model B
    return {"models": ["Model A", "Model B"]}
