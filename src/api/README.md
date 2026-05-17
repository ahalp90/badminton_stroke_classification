# api/ — FastAPI inference backend

The HTTP layer that fronts both stroke classifiers (BRIC, BST) and the
player-identity store. Model-agnostic by dispatch: routes to a named
handler, the handler does the work, the API layer handles persistence
and HTTP framing.

## Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app + route handlers (`/api/upload`, `/api/status/<job>`, `/api/results/<job>`, `/api/jobs/<job>/frames/<idx>`, `/api/players/search`, `/api/players/<id>`). |
| `inference.py` | Handler dispatcher: `model_name → handler_fn`. Falls back to a v1-contract-shaped mock when the named handler isn't registered. |
| `storage.py` | Thin `sqlite3` wrapper — no ORM. Persists jobs, per-stroke rows, and players. |
| `schema.sql` | DDL applied on backend boot via `CREATE TABLE IF NOT EXISTS`. |
| `jobs.py` / `config.py` | Pre-existing scaffolding (job state machine, env config). |

## Route map

See [`docs/api_contract.md`](../../docs/api_contract.md) for the full
schema. At a glance:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/upload` | Submit video + markup → `job_id` |
| `GET`  | `/api/status/<job_id>` | Poll `queued / running / done / error` |
| `GET`  | `/api/results/<job_id>` | Fetch the v1-contract result JSON |
| `GET`  | `/api/jobs/<job_id>/frames/<stroke_idx>` | Serve a stroke-frame thumbnail |
| `GET`  | `/api/players/search?q=&limit=` | Typeahead for the upload UX |
| `GET`  | `/api/players/<id>` | Fetch a single player record |

## Dispatcher pattern

```python
from bric.infer import run_inference as run_bric

_HANDLERS = {"bric": run_bric}

def run_inference(video_path, model_name, markup=None):
    handler = _HANDLERS.get(model_name)
    if handler is None:
        return _mock_result(markup)            # v1-contract-shaped placeholder
    return handler(video_path, markup=markup)
```

To wire a new handler, register it in `_HANDLERS` — the dispatcher and
the contract take care of everything else. BST handler will be added
the same way once its inference entry point is exposed.

## Import rules

- `api.*` doesn't import architecture-specific code directly except for
  the handler-registration line in `inference.py`. Routes call the
  dispatcher; the dispatcher calls the handler.
- `api.storage` is the only module allowed to touch sqlite.

## Related docs

- [`docs/api_contract.md`](../../docs/api_contract.md) — the request /
  response schema both handlers must satisfy. Source of truth.
- [`docs/storage.md`](../../docs/storage.md) — DB schema, two-views
  pattern (denormalised `jobs.result` + normalised `strokes`), caching
  strategy, player resolution rules.
