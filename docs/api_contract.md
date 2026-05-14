# Inference API Contract — v1

The contract the FastAPI backend exposes to the frontend (hba app) and that
both inference handlers (BRIC, BST) must satisfy. Model-agnostic by design:
the dispatcher in `src/api/inference.py` routes to the named handler
without the contract caring about which architecture is on the other side.

**Status:** v1, proposed. Lock through a review pass with the BST handler
author and the frontend wiring author before either backend ships.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/upload` | Submit a video + markup; returns a `job_id`. |
| `GET`  | `/api/status/<job_id>` | Poll job state (queued / running / done / error). |
| `GET`  | `/api/results/<job_id>` | Fetch the structured result for a completed job. |
| `GET`  | `/api/jobs/<job_id>/frames/<stroke_idx>` | Serve a stroke-frame thumbnail (referenced from result `stroke_frame_url`). |
| `GET`  | `/api/players/search?q=<prefix>&limit=10` | Typeahead search; returns players matching the prefix on label or aliases. |
| `GET`  | `/api/players/<id>` | Fetch a single player's identity record. |

Existing `POST /api/upload` and `GET /api/status` already exist as stubs.
The result, frame-thumbnail, and player endpoints are net-new for v1.
Storage layer that backs them: see [`storage.md`](storage.md).

---

## Upload payload

`POST /api/upload` accepts `multipart/form-data` with:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file`   | binary | yes | The video to classify. |
| `model`  | string | yes | Handler name; one of `bric`, `bst` (extensible). Unknown values → mock fallback. |
| `markup` | string (JSON) | yes | Schema below. |

### `markup` JSON schema

```jsonc
{
  // Court boundary as 4 normalised (0..1) corners in pixel-space order:
  // top-left, top-right, bottom-right, bottom-left.
  // Backend computes H = cv2.getPerspectiveTransform(boundary, ref_court_corners).
  "boundary": [
    {"x": 0.12, "y": 0.18},
    {"x": 0.88, "y": 0.18},
    {"x": 0.92, "y": 0.94},
    {"x": 0.08, "y": 0.94}
  ],

  // Stroke annotations. Length 1 = single-shot UX; length N = rally UX.
  // Mode is implicit in the count — no separate "mode" field.
  // Each annotation is processed identically by the backend; the result
  // strokes list mirrors this list 1:1 in the same order.
  "annotations": [
    {
      "target_frame":        140,            // canonical stroke localisation (int, 0-indexed frame)
      "region_start_frame":  100,            // bounds the shuttle/court feature window
      "region_end_frame":    180,            // region width must satisfy: end - start <= ~5s * fps
      "player_side":         "top"           // optional hint: "top" | "bottom" | null (inferred from court if null)
    }
    // ... more annotations for rally mode
  ],

  // Filter strokes by side before returning. Default both.
  "enabled_sides": ["top", "bottom"],

  // Optional player identification. Backend resolves each stroke's
  // `player_id` and `player_label` from these via the stroke's `player_side`.
  //
  // Per-side resolution rules (see storage.md):
  //   - `id` provided + exists in players table  → use it; result carries it
  //   - `id` provided + not found                → HTTP 4xx error
  //   - `id` null + `label` provided             → backend creates a new
  //                                                 player row with a UUID;
  //                                                 result carries the new id
  //   - both null                                → result fields both null
  //
  // The frontend gets player IDs from the typeahead search endpoint
  // (`GET /api/players/search?q=`). The backend trusts what it's sent —
  // it never fuzzy-matches labels. Same label across uploads with no
  // ID creates SEPARATE player rows; dedup is the UI's job at upload
  // time via the search endpoint, with merge tools coming in v2.
  "player_top_id":       "p_smith_001",   // null if creating new player from label
  "player_top_label":    "Smith",
  "player_bottom_id":    "p_jones_004",
  "player_bottom_label": "Jones"
}
```

### Backend validation

For each annotation:
- `region_start_frame <= target_frame <= region_end_frame`
- `region_end_frame - region_start_frame <= 5 * fps` (approx 5 seconds)
- All frame indices must be within `[0, total_frames - 1]`

The `boundary` must be 4 points; if absent or malformed, the backend falls
back to a vertical-pixel heuristic for player-side assignment (no homography).

The backend never invents stroke times — every classified annotation is
user-supplied. Stroke detection / SRA pre-segmentation is a v2 concern
(see `POST /api/segment` in the v2 backlog).

---

## Result payload

`GET /api/results/<job_id>` returns:

```jsonc
{
  "strokes": [
    {
      "target_frame":     140,                    // mirrors input annotation
      "timestamp_sec":    4.67,                   // derived from target_frame / fps for display
      "stroke_type":      "smash",                // one of the 14-class taxonomy (see shared/taxonomy.py)
      "confidence":       0.87,
      "player_side":      "top",                  // "top" | "bottom" | "unknown"
      "player_id":        "p_smith_001",          // resolved per the rules above; may be a backend-generated UUID if upload supplied label without id
      "player_label":     "Smith",                // resolved from upload labels (or canonical players.label if id was supplied)
      "court_position":   {"x": 0.55, "y": 0.18}, // striker foot-centre projected to court via homography;
                                                  //   normalised [0, 1] over the FULL singles court
                                                  //   (top baseline y=0, net y=0.5, bottom baseline y=1).
                                                  //   null if no boundary supplied (no homography).
                                                  //   Half-court normalisation is a query-time derivation.
      "stroke_frame_url": "/api/jobs/abc/frames/0"// thumbnail; one per stroke
    }
    // ... one entry per input annotation, in the same order
  ],
  "rally_summary": {
    "total_strokes":         3,
    "rally_length_seconds":  6.4
  }
}
```

`strokes` is **always** the same length as `markup.annotations` (after
the `enabled_sides` filter is applied).

The 14-class stroke taxonomy is the source of truth in
[`src/shared/taxonomy.py`](../src/shared/taxonomy.py); both handlers must
emit `stroke_type` strings drawn from that list.

---

## Mock-fallback behaviour

When `model` doesn't resolve to a registered handler (e.g. handler
implementation pending), the backend returns a v1-shape mock:

- Sleeps 3 seconds (so the frontend's processing-poll loop exercises end-to-end).
- Returns one stroke per input annotation, mirroring fields the upload
  already supplied (`target_frame`, `player_side`, `player_id`,
  `player_label`).
- All backend-derived fields take placeholder values: `stroke_type = "clear"`,
  `confidence = 0.5`, `court_position = null`, `stroke_frame_url = ""`.
- `rally_summary.total_strokes = len(strokes)`,
  `rally_summary.rally_length_seconds = 0.0`.

If `markup.annotations` is empty/missing (legacy callers), the mock falls
back to a single canned stroke at `target_frame = 0`.

---

## Forward-compatibility (non-v1)

The v1 schema is designed to be extended without breaking changes. Items
known to be coming in v2 (preserved here so handler authors can leave
hooks where appropriate):

- **Per-annotation `role` field** (`"serve" | "rally" | "rally_end" | null`)
  to support analytics by stroke role; `null` is treated as v1 behaviour
  (unconstrained classification).
- **Per-stroke `validation` field** (`"user" | "auto"`) to distinguish
  human-validated annotations from heuristic SRA-detected ones. v1
  uploads default to `"user"`.
- **Per-stroke `warnings` array** with codes like `low_confidence_player_identification`,
  `low_shuttle_visibility`, `track_id_unstable`, `no_homography`,
  `court_partially_visible`.
- **Top-3 predictions** per stroke (`top_3_predictions: [{stroke_type, confidence}, ...]`)
  for misclassification analysis.
- **Per-stroke artefacts** beyond `stroke_frame_url` + `court_position`
  (both v1): `player_crop_url`, `shuttle_landing_position` (where the
  shuttle landed, requires SRA-style trajectory analysis).

Handlers should ignore unknown fields in incoming markup and may emit
extra fields in the result without breaking older clients.

---

## Open questions for handler authors

To be resolved during contract review (these aren't decisions, just the
seams where BRIC and BST may have different needs):

- **Per-handler timing budget**: BST's pose-extraction subprocess is heavy
  (MMPose load); BRIC's R(2+1)D forward is comparatively light. Should
  the contract specify expected SLA, or let `/api/status` drive perceived
  latency?
- **Handler-specific markup hints**: e.g. BST may want a `pose_quality`
  hint or a `serve_side` field. v1 keeps markup handler-agnostic — extension
  via an optional `handler_hints: dict` field is cheap if needed.
- **Failure modes**: structured error response shape when a handler
  refuses an upload (e.g. no players detected, court polygon wildly off).
  v1 leaves this implicit (HTTP 4xx with `{"detail": ...}`); a structured
  error catalogue is a v2 candidate.
