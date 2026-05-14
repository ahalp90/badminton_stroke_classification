# Inference State Store — Design

A lightweight SQLite-backed store for inference jobs, results, and player
identification. Sits behind the FastAPI backend and feeds analytics queries.

**Why SQLite:** zero new deps (`import sqlite3` ships with Python), single
file backup story matches the existing JSON-manifest pattern (DL-004),
real schema enforcement, real query layer for analytics. Migration to
Postgres later is mechanical if scale ever demands it.

**Why not the dataset's JSON manifest pattern:** the dataset (~30k clips)
is read-mostly and append-rare — JSON is fine there. The inference store
is the opposite: append-every-job and queried interactively for analytics.
Different access pattern, different tool.

---

## Location & lifecycle

- DB file: `runtime/state/inference.db`
- Gitignored. The directory is tracked via `runtime/state/.gitkeep`.
- Schema is created automatically on backend boot via `CREATE TABLE IF NOT EXISTS`.
- No migrations framework in v1 — schema changes during v1 are still cheap
  (wipe + recreate). Add Alembic or similar in v2 if the schema starts
  evolving in production.
- Backups are a file copy. Wipe via `rm runtime/state/inference.db`.

`runtime/state/` is for *backend operational state* and is distinct from
`runtime/cache/` (derivable computations like player tracks) and
`runtime/data/` (source data) / `runtime/checkpoints/` (model weights).
Wiping `runtime/state/` loses inference history but doesn't break training
or serving.

---

## Schema

```sql
-- Players: stable identities for analytics aggregation across uploads.
CREATE TABLE IF NOT EXISTS players (
    id          TEXT PRIMARY KEY,        -- UUID generated on first sighting
    label       TEXT NOT NULL,           -- canonical display name
    aliases     TEXT,                    -- JSON array of past labels (v2 dedup hints)
    created_at  TIMESTAMP NOT NULL,
    metadata    TEXT                     -- JSON blob for extensions
);

-- Jobs: one row per /api/upload submission.
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    video_sha256  TEXT NOT NULL,         -- content hash for cache lookup
    handler       TEXT NOT NULL,         -- "bric" / "bst" / "mock"
    run_id        TEXT,                  -- model checkpoint id (cache invalidation key)
    submitted_at  TIMESTAMP NOT NULL,
    completed_at  TIMESTAMP,
    status        TEXT NOT NULL,         -- queued / running / done / error
    markup        TEXT,                  -- upload markup JSON
    result        TEXT                   -- result payload JSON (when done)
);

-- Strokes: flat per-stroke table for analytics. Normalised — references
-- players(id) only; player_label is JOIN'd at query time from players.label.
CREATE TABLE IF NOT EXISTS strokes (
    job_id         TEXT NOT NULL,
    stroke_idx     INTEGER NOT NULL,
    target_frame   INTEGER NOT NULL,
    timestamp_sec  REAL NOT NULL,
    stroke_type    TEXT NOT NULL,
    confidence     REAL NOT NULL,
    player_side    TEXT,
    player_id      TEXT,                 -- FK to players.id (nullable)
    court_x        REAL,                 -- striker foot-centre projected to court via H
    court_y        REAL,                 --   normalised [0, 1] over the FULL court
                                         --   (top half y<0.5, bottom half y>0.5).
                                         --   Null if no homography (no boundary supplied).
    frame_path     TEXT,                 -- relative path to thumbnail JPG from project root,
                                         --   e.g. "runtime/jobs/<job_id>/frames/<stroke_idx>.jpg".
                                         --   Null if no frame was generated (e.g. mock handler).
                                         --   Existence of this column = the file exists on disk.
    role           TEXT,                 -- v2: serve / rally / rally_end
    validation     TEXT,                 -- v2: user / auto
    PRIMARY KEY (job_id, stroke_idx),
    FOREIGN KEY (job_id)    REFERENCES jobs(job_id),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE INDEX IF NOT EXISTS idx_strokes_player ON strokes(player_id);
CREATE INDEX IF NOT EXISTS idx_strokes_type   ON strokes(stroke_type);
CREATE INDEX IF NOT EXISTS idx_jobs_cache     ON jobs(video_sha256, handler, run_id);
```

### Two views, one transaction

`jobs.result` (JSON blob) and `strokes` (flat table) carry overlapping
data on purpose:

- **`jobs.result`** is the source of truth for serving `GET /api/results/<job_id>`.
  It carries the full *denormalised* result payload — every per-stroke
  field including `player_label` and `stroke_frame_url` — so the API
  serves a single read with no joins.
- **`strokes`** is the analytics view. Normalised: references `players(id)`
  only; `player_label` is JOIN'd at query time from `players.label`.
  This is what aggregate queries hit ("Smith's stroke distribution",
  "court heatmap by stroke type") — they shouldn't have to parse JSON
  to find what they need.

Both writes happen in one transaction at job-completion time, so the
two views never disagree.

### Court coordinate convention

`court_x` / `court_y` are the striker's foot-centre projected through the
upload's homography `H`, normalised to `[0, 1]` over the **full** singles
court:

- `y = 0.0` → top baseline (top player's back of court)
- `y = 0.5` → net
- `y = 1.0` → bottom baseline
- `x = 0.0` → left sideline; `x = 1.0` → right sideline

**Storing full-court (not half-normalised) is intentional.** It preserves
the actual game position so the default UI overlay shows the real shot
location on a full court diagram. Cross-match aggregation that wants to
ignore which side the player was on (e.g. "Smith's hit zones across all
matches") derives half-court coords at query time:

```sql
-- All of Smith's shots normalised to a single half-court view.
-- Note: player_label is JOIN'd from players (not stored in strokes).
SELECT p.label, s.stroke_type, s.court_x, ABS(s.court_y - 0.5) * 2 AS y_half
FROM strokes s
JOIN players p ON p.id = s.player_id
WHERE s.player_id = 'p_smith_001' AND s.court_x IS NOT NULL;
```

Half-normalising at storage time would lose information that can't be
recovered. Full-court is the source of truth; half is a derived view.

### Frame thumbnails

Per-stroke frame thumbnails are JPG files on disk, referenced by
`strokes.frame_path`:

```
runtime/jobs/<job_id>/frames/<stroke_idx>.jpg
```

The `frame_path` column on `strokes` stores the full relative path from
the project root, so analytics queries can locate the file directly:

```sql
-- Every smash thumbnail Smith has hit
SELECT s.frame_path, s.confidence
FROM strokes s
WHERE s.player_id = 'p_smith_001'
  AND s.stroke_type = 'smash'
  AND s.frame_path IS NOT NULL;
```

`NULL` frame_path = no JPG was written for that stroke (mock handler,
or a real handler that failed to extract the frame). The presence of
`frame_path` on a row is the canonical signal that the file exists.

The result JSON in `jobs.result` carries `stroke_frame_url`, the
HTTP-servable form of the same reference:
`f"/api/jobs/{job_id}/frames/{stroke_idx}"`. Stored denormalised in the
JSON for serving convenience; not stored in `strokes` (would be redundant
with `frame_path`).

**Lifetime**: frame files persist for the lifetime of the `jobs` row.
Deleting a job (manual cleanup, retention policy when v2 lands) removes
the corresponding frames directory in the same operation. v1 has no
automatic TTL — frames stay until explicitly removed.

---

## Caching pattern

```python
def get_or_compute(video_sha256, handler, run_id, markup, compute_fn):
    cached = db.execute(
        "SELECT job_id, result FROM jobs "
        "WHERE video_sha256=? AND handler=? AND run_id=? AND status='done' "
        "ORDER BY completed_at DESC LIMIT 1",
        (video_sha256, handler, run_id)
    ).fetchone()
    if cached:
        return cached["result"]
    return compute_fn(markup)   # caller persists the new job + result
```

Cache key `(video_sha256, handler, run_id)` means:
- **Same video, same handler, same model checkpoint** → instant cached return
- **New checkpoint** → `run_id` differs → fresh inference (correct semantics)
- **Different handler** (BRIC vs BST) → separate cache entry (correct, they're different models)
- **Different markup on same video** → currently re-runs (markup affects the result). v2 could cache by `(video_sha256, handler, run_id, markup_hash)` if uploads with identical markups recur.

---

## Player identification flow

The contract carries optional `player_top_id` / `player_bottom_id` per
upload. Backend behaviour by case:

| Upload state | Backend action |
|--------------|----------------|
| `id` provided, exists in `players` | Use it. Result carries the same id + the canonical label from `players.label`. |
| `id` provided, doesn't exist in `players` | **Error** (HTTP 4xx). The UI should only send IDs from a prior search response. |
| `id` null, `label` provided | Create new player row with generated UUID + supplied label. Result carries the new id. |
| both null | Result's `player_id` and `player_label` are null. |

Player IDs are UUIDs (`uuid.uuid4().hex`) — opaque, no embedded semantics.
The `label` is what the user typed/selected; same label across separate
uploads with no IDs creates *separate* player rows. Dedup is the UI's job
via the search endpoint; the backend trusts what it's sent.

### Search endpoint (for typeahead UX)

```
GET /api/players/search?q=<prefix>&limit=10
→ [{id: "...", label: "Smith", aliases: [...]}, ...]
```

Case-insensitive prefix match on `label` and any entry in `aliases`.
Drives the frontend typeahead: as the user types, the UI calls this and
shows existing matches. If the user picks a result, the upload sends that
`id`. If the user confirms "create new player" instead, the upload sends
just the `label` and the backend assigns a UUID.

```
GET /api/players/<id>
→ {id, label, aliases, created_at, metadata, stats: {...optional v2}}
```

Single-player fetch. v1 returns identity fields only; v2 can attach
aggregate stats from `strokes` table.

---

## Backend module layout

```
src/api/
├── main.py             # FastAPI app + route handlers (existing)
├── inference.py        # Handler dispatcher (existing, rewritten Day 7)
├── storage.py          # SQLite wrapper (NEW Day 7)
└── schema.sql          # DDL above (NEW Day 7)
```

`storage.py` exposes a small interface — no ORM, just typed function
calls over `sqlite3.Connection`:

```python
def init_db(path: Path) -> None: ...

def create_job(job_id, video_sha256, handler, run_id, markup) -> None: ...
def update_job_status(job_id, status, *, completed_at=None, error=None) -> None: ...
def store_result(job_id, result_dict) -> None:   # writes jobs.result + strokes rows in one tx
    ...

def get_job(job_id) -> dict | None: ...
def get_result(job_id) -> dict | None: ...

def get_or_create_player(id_or_none, label_or_none) -> tuple[str | None, str | None]:
    """Returns (resolved_id, resolved_label) following the table above."""

def search_players(query, limit=10) -> list[dict]: ...
def get_player(player_id) -> dict | None: ...

def find_cached_job(video_sha256, handler, run_id) -> dict | None: ...
```

No ORM, no migrations framework — keep the stack thin in v1. If the
schema starts evolving frequently in production, layer Alembic over it.

---

## v2 candidates

- **Aggregate stats endpoints**: `GET /api/players/<id>/stats`, `GET /api/analytics/stroke_distribution`, etc. The schema already supports them; just need the routes.
- **Fuzzy dedup**: `aliases` column is reserved for past labels; v2 admin endpoint to merge two player rows (point both ids' strokes at one canonical id, archive the other).
- **`player_aliases` as a separate table**: currently `players.aliases` is a JSON array (acceptable for v1's small alias counts, simpler search via `LIKE`). If aliases grow large or need indexed search, normalise to a `player_aliases(player_id, alias)` table.
- **Postgres migration**: SQL is mostly portable. The biggest change is `TEXT` → `JSONB` for the JSON columns and proper UUID type.
- **Markup-hash cache key**: cache by `(video_sha256, handler, run_id, sha256(markup))` if identical re-uploads with identical markup become common.
- **Soft delete**: `deleted_at` columns on `jobs` and `players` for retention policies without losing referential integrity.
