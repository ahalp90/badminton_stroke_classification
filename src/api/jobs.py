import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    filename: str
    model_name: str
    video_path: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[dict] = None
    error: Optional[str] = None
    # Optional fields captured for echo-back in /api/results so the FE can
    # confirm the markup sidecar (Gap 1) and library_predict context
    # (Gap 2) reached the server.
    markup: Optional[dict] = None
    source: str = "upload"  # "upload" | "library"
    clip_stem: Optional[str] = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        job_id: str,
        filename: str,
        model_name: str,
        video_path: str,
        markup: Optional[dict] = None,
        source: str = "upload",
        clip_stem: Optional[str] = None,
    ) -> Job:
        job = Job(
            job_id=job_id,
            filename=filename,
            model_name=model_name,
            video_path=video_path,
            markup=markup,
            source=source,
            clip_stem=clip_stem,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, status: JobStatus, result=None, error=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def delete(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())
