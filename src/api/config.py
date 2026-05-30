import os
from pathlib import Path

# Repo root resolves the same in Docker (`/app`) and native dev (the working
# tree). Used by registry.py to anchor relative paths in models_registry.yaml.
REPO_ROOT = Path(os.getenv("BST_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
REGISTRY_PATH = Path(
    os.getenv("BST_REGISTRY_PATH", str(REPO_ROOT / "docs" / "models_registry.yaml"))
)

# Optional: directory holding the clip mp4s, with layout
# <split>/<Side>_<class>/<stem>.mp4. On UNE HPC this resolves to
# /scratch/comp320a/ShuttleSet/clips. Unset locally; video endpoint
# returns a helpful 404 when missing.
_clips_dir = os.getenv("BST_CLIPS_DIR")
BST_CLIPS_DIR: Path | None = Path(_clips_dir) if _clips_dir else None

# Optional: a flat, stem-keyed directory of sample clips for the Model Results
# per-clip player — files are named "<clip_stem>.mp4" directly (e.g.
# clips_local/24_3_8_2.mp4). Lets you play a handful of real clips locally
# without recreating the full ShuttleSet tree or setting BST_CLIPS_DIR, and is
# keyed by the stable clip_stem rather than clip_index's placeholder
# video_path. Defaults to <repo>/clips_local; its contents are gitignored.
LOCAL_CLIPS_DIR = Path(
    os.getenv("BST_LOCAL_CLIPS_DIR", str(REPO_ROOT / "clips_local"))
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1024"))
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
