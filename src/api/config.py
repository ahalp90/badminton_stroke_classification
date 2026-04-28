import os
from pathlib import Path

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

EXPERIMENTS_DIR = Path(
    os.getenv(
        "EXPERIMENTS_DIR",
        "/app/src/bst_refactor/stroke_classification/main_on_shuttleset/experiments",
    )
)

JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_HOURS", "24")) * 3600
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "1")) * 3600
