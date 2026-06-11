# api/ — FastAPI inference backend

HTTP layer for the inference service. **Current state: Tier 1 registry
endpoints live (read-only browsing of precomputed predictions); Tier 2
inference returns mock data behind the same plumbing.**

What works today:
- File upload with optional temporal + spatial cropping
- In-memory job state machine (queued / processing / complete / failed)
- Job lifecycle (status / results / delete)
- Discovery of available checkpoints on disk (`/api/models`)
- Tier 1 registry endpoints: list models, per-clip details, per-class stats, paginated/filtered clip browsing, video streaming
- CORS configurable via env

What's not built yet:
- Real handler dispatcher (Tier 2 inference still returns hardcoded mock predictions)
- SQLite persistence (jobs / strokes / players tables) — currently in-memory, lost on restart
- `/api/players/*`, `/api/jobs/<job>/frames/<idx>` endpoints
- Backend wiring for BRIC inference (handler stub returns canned data)

## Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app, CORS middleware, route handlers for the upload/job lifecycle, includes the registry router. |
| `inference.py` | **Stub** `run_inference(video_path, model_name)` — returns hardcoded mock predictions after a 3-second sleep so the frontend's polling loop exercises end-to-end. To be replaced with a handler dispatcher. |
| `jobs.py` | `JobStore`: in-memory job state map with thread lock, `JobStatus` enum (queued/processing/complete/failed), `Job` dataclass. Lost on process restart — SQLite-backed replacement is future work. |
| `registry.py` | Tier 1 read-only registry endpoints. Loads `docs/models_registry.yaml` + each entry's `manifest.yaml`, serves sidecar JSONs (predictions, per-class stats, clip_index). Mounted via `app.include_router` under `/api` prefix. |
| `config.py` | Env-driven config: `REPO_ROOT`, `REGISTRY_PATH`, `BST_X_CLIPS_DIR`, `UPLOAD_DIR`, `MAX_FILE_SIZE_MB`, `ALLOWED_EXTENSIONS`, `EXPERIMENTS_DIR`, job TTL + cleanup interval. |

## Routes

### Inference (Tier 2)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/upload` | Multipart upload (`file`, query `model`, optional temporal crop `start_sec`/`end_sec`, optional spatial crop `crop_x`/`crop_y`/`crop_w`/`crop_h`). Validates extension, model name, crop coherence. Persists video to `UPLOAD_DIR/<job_id>.<ext>`, optionally re-encodes with crop applied, creates job, runs background processing. |
| `GET`  | `/api/status/<job_id>` | Returns `{"job_id": id, "status": queued\|processing\|complete\|failed}`. 404 if not found. |
| `GET`  | `/api/results/<job_id>` | 200 + `{job_id, status, ...result}` when complete; 202 + `{job_id, status}` while still processing; 500 when failed; 404 if not found. |
| `DELETE` | `/api/jobs/<job_id>` | Removes the job from the in-memory store. |
| `GET`  | `/api/models` | Lists checkpoints (`*.pt` files) under `EXPERIMENTS_DIR`, deduplicated by filename stem. |

### Registry (Tier 1, read-only)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/registry` | List all registered models with summary metadata. |
| `GET` | `/api/registry/<model_id>` | Single model summary. |
| `GET` | `/api/registry/<model_id>/splits/<split>/stats` | Per-class metrics for the chosen split. |
| `GET` | `/api/registry/<model_id>/splits/<split>/clips` | Paginated + filterable list of per-clip prediction summaries (`limit`, `offset`, `true_class`, `predicted_class`, `errors_only`). |
| `GET` | `/api/registry/<model_id>/splits/<split>/clips/<stem>` | Single clip's full prediction detail. |
| `GET` | `/api/clips/<stem>/video` | Stream the source mp4 (resolves under `BST_X_CLIPS_DIR`; 404 with helpful message when unset). |

## Import rules (target state)

- `api.*` doesn't import architecture-specific code directly except
  for the handler-registration line in `inference.py`. Routes call the
  dispatcher; the dispatcher calls the handler.
- `api.storage` (when added) will be the only module allowed to touch
  sqlite.

## Related docs

- [`docs/api_contract.md`](../../docs/api_contract.md) — the full
  request / response schema both handlers will satisfy. Source of truth
  for what the API surface will look like after PR4.
- [`docs/storage.md`](../../docs/storage.md) — DB schema design, caching
  strategy, player resolution rules. To be implemented in PR4.
