# Inference API Contract — v1

The HTTP contract the backend exposes for stroke classification, model
browsing, and per-clip metadata. Both inference handlers (BRIC, BST)
satisfy this contract. Companion to the BST handoff at
[`scratch/frontend_integration_handoff.md`](../scratch/frontend_integration_handoff.md);
remaining points of disagreement tracked in [Open conflicts](#open-conflicts).

## Contents

1. [The two modes](#the-two-modes)
2. [API endpoint specifications](#api-endpoint-specifications)
3. [Mock data and fallback behaviour](#mock-data-and-fallback-behaviour)
4. [Confidence](#confidence)
5. [Indexing conventions](#indexing-conventions)
6. [Open conflicts](#open-conflicts)

---

## The two modes

| Mode | Tier | What it does | Analytic story |
|---|---|---|---|
| Model mode | 1 | Reads pre-computed predictions off disk for a known dataset clip. No model loaded. | Browse model behaviour against the labelled test/val splits — per-class confusion, error surfacing, headline metrics. |
| Inference mode | 2 | Runs the full pipeline (player tracking, shuttle tracking, court projection, classification) on a user-uploaded video. | Per-stroke output for arbitrary footage with player aggregation across uploads, court overlays, frame thumbnails. |

The two modes have **different request and response shapes** — Model
mode is built around dataset clips, Inference mode is built around
user uploads with rally support. Per-endpoint specs below.

---

## API endpoint specifications

### Model registry

The frontend's model picker (and any "model accuracy" widget) reads
from these. Both modes' other endpoints accept a `model_id` from this
list.

#### `GET /api/registry`

List of all registered model checkpoints with metadata.

**Request:** no parameters.

**Response:**

```jsonc
{
  "models": [
    {
      "id":               "bst_x_v1_wipe_drop_s5",
      "display_name":     "BST-X v1, wipe_drop run, serial 5 (current best)",
      "description":      "14-class taxonomy, augmentation v1 + adaptive focal loss.",
      "architecture":     "bst",
      "is_default":       true,
      "taxonomy":         "une_merge_v1_nosides",
      "num_classes":      14,
      "class_list":       ["net_shot", "return_net", "smash", "wrist_smash", "..."],
      "splits_available": ["val", "test"],
      "test_metrics": {
        "macro_f1":      0.7479,
        "min_f1":        0.5147,
        "accuracy":      0.7675,
        "top2_accuracy": 0.9407,
        "per_class_f1":  { "net_shot": 0.8924, "smash": 0.5147 }
      },
      "val_metrics":      { }
    }
  ]
}
```

**Fields:**

| Field | Type | Notes |
|---|---|---|
| `id` | string | Opaque; pass back in subsequent requests targeting this model. |
| `display_name` | string | Rendered in the model picker. |
| `description` | string | Free-text. |
| `architecture` | enum | `"bric"` \| `"bst"` (extensible). Drives architecture-grouping in the picker and internal dispatch routing. |
| `is_default` | bool | Registry author marks one entry per architecture as default-best. Picker shows defaults; "show all variants" toggle exposes the rest. |
| `taxonomy` | string | Identifier for the class taxonomy. |
| `num_classes` | int | `class_list.length`. |
| `class_list` | string[] | Order matters; lines up with prediction arrays. Read class names from here. |
| `splits_available` | string[] | Usually `["val", "test"]`. |
| `test_metrics`, `val_metrics` | object | Per-class F1 keys mirror `class_list`. Empty `{}` if not yet eval'd. |

#### `GET /api/registry/{model_id}`

Single model entry. Same shape as one element of `/api/registry`'s `models[]`.

---

### Tier 1: Model mode endpoints

Tier 1 returns precomputed predictions on the labelled splits. No model
is loaded; the backend reads JSON files off disk.

#### `GET /api/registry/{model_id}/splits/{split}/clips`

Paginated list of precomputed clip predictions, with optional filters.

**Path params:** `model_id` (from registry), `split` (`val` \| `test`).

**Query params:** `limit` (int, 1-500, default 50), `offset` (int, default 0),
`true_class` (string, optional), `predicted_class` (string, optional),
`errors_only` (bool, default false).

**Response:**

```jsonc
{
  "model_id": "bst_x_v1_wipe_drop_s5",
  "split":    "test",
  "total":    4202,
  "limit":    50,
  "offset":   0,
  "clips": [
    {
      "clip_stem":      "35_1_10_17",
      "true_class":     "smash",
      "predicted_class":"wrist_smash",
      "is_correct":     false,
      "confidence_pct": 32,
      "match":          "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
      "split":          "test"
    }
  ]
}
```

**Fields (each `clips[]` entry — slim summary):**

| Field | Type | Notes |
|---|---|---|
| `clip_stem` | string | Composite `{vid}_{set}_{rally}_{ball_round}`. |
| `true_class`, `predicted_class` | string | From `class_list`. |
| `is_correct` | bool | `predicted_class == true_class`. |
| `confidence_pct` | int | `[0, 100]`. See [Confidence](#confidence). |
| `match`, `split` | string | Dataset metadata. |

For `top_k`, `set_id`, `rally`, `ball_round`, `video_url`, hit the
single-clip endpoint below.

#### `GET /api/registry/{model_id}/splits/{split}/clips/{stem}`

Full prediction for a single dataset clip.

**Path params:** `model_id`, `split`, `stem` (e.g. `35_1_10_17`).

**Response:**

```jsonc
{
  "clip_stem":       "35_1_10_17",
  "video_url":       "/api/clips/35_1_10_17/video",
  "match":           "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
  "set_id":          "set1",
  "rally":           10,
  "ball_round":      17,
  "split":           "test",
  "true_class":      "smash",
  "predicted_class": "wrist_smash",
  "is_correct":      false,
  "confidence_pct":  32,
  "top_k": [
    {"class": "wrist_smash", "confidence": 0.32},
    {"class": "smash",       "confidence": 0.27},
    {"class": "drive",       "confidence": 0.12},
    {"class": "push",        "confidence": 0.10},
    {"class": "drop",        "confidence": 0.08}
  ]
}
```

**Fields:**

| Field | Type | Notes |
|---|---|---|
| `clip_stem` | string | Composite `{vid}_{set}_{rally}_{ball_round}`. |
| `video_url` | string | Range-supported clip stream — see `GET /api/clips/{stem}/video`. |
| `match` | string | Broadcast identifier from ShuttleSet source. |
| `set_id`, `rally`, `ball_round` | string / int | 1-indexed positions within the match. Dashboards aggregate by these. |
| `split` | string | `"val"` \| `"test"`. |
| `true_class` | string | Ground-truth from the dataset. |
| `predicted_class` | string | `top_k[0].class`. |
| `is_correct` | bool | `predicted_class == true_class`. |
| `confidence_pct` | int | `[0, 100]`. Equal to `round(top_k[0].confidence * 100)`. |
| `top_k` | array | Top 5 by confidence, descending. Each: `{class, confidence}` with confidence as float `[0, 1]`. |

#### `GET /api/registry/{model_id}/splits/{split}/stats`

Per-class aggregates for the per-class panel.

**Path params:** `model_id`, `split`.

**Response:**

```jsonc
{
  "split":      "test",
  "n_clips":    4202,
  "class_list": ["net_shot", "return_net", "smash", "..."],
  "per_class": {
    "smash": {
      "support_true":   312,
      "support_pred":   289,
      "precision":      0.51,
      "recall":         0.55,
      "f1":             0.5147,
      "top5_when_true": [["smash", 0.55], ["wrist_smash", 0.21], ["drive", 0.07], ["push", 0.04], ["drop", 0.03]],
      "top5_when_pred": [["smash", 0.51], ["wrist_smash", 0.18], ["clear", 0.09], ["lob", 0.06], ["drive", 0.05]]
    }
  }
}
```

**Fields (per class):**

| Field | Type | Notes |
|---|---|---|
| `support_true` | int | Number of true-label instances in the split. |
| `support_pred` | int | Number of predicted-label instances. |
| `precision`, `recall`, `f1` | float | Standard per-class metrics. |
| `top5_when_true` | `[class, freq][]` | When ground truth is this class, top 5 predictions and frequencies (recall-style; sums to 1.0). |
| `top5_when_pred` | `[class, freq][]` | When prediction is this class, top 5 actual classes and frequencies (precision-style; sums to 1.0). |

#### `GET /api/clips/{stem}/video`

Stream a known dataset clip's mp4. Byte-range requests supported, so
HTML5 `<video>` scrubbing works.

**Path params:** `stem`.

**Response:** binary `video/mp4`.

**Errors:** `404` if the clip isn't in any registered model's clip
index, or if `BST_CLIPS_DIR` is unset / the file isn't on this host.

---

### Tier 2: Inference mode endpoints

Tier 2 runs the full inference pipeline on user-uploaded video. The
flow: `POST /api/upload` returns a `job_id`; poll `GET /api/status/{job_id}`
until `done`; fetch `GET /api/results/{job_id}`.

#### `POST /api/upload`

Submit a video plus markup; returns a job ID for polling.

**Request:** `multipart/form-data` with two parts:

- `file`: video binary.
- `markup`: JSON with the schema below.

**Markup schema:**

```jsonc
{
  "architecture":        "bric",
  "model_id":            null,
  "orientation":         "portrait",
  "video_label":         "Smith vs Jones, training 2026-05-20",
  "boundary": [
    {"x": 0.12, "y": 0.18},
    {"x": 0.88, "y": 0.18},
    {"x": 0.92, "y": 0.94},
    {"x": 0.08, "y": 0.94}
  ],
  "annotations": [
    {
      "target_frame":       140,
      "region_start_frame": 100,
      "region_end_frame":   180,
      "player_side":        "top"
    }
  ],
  "enabled_sides":       ["top", "bottom"],
  "player_top_id":       "p_smith_001",
  "player_top_label":    "Smith",
  "player_bottom_id":    "p_jones_004",
  "player_bottom_label": "Jones"
}
```

**Markup fields:**

| Field | Type | Notes |
|---|---|---|
| `architecture` | enum \| null | `"bric"` \| `"bst"`. Simple-mode model pick — backend resolves to the registry entry with `is_default: true` for that architecture. |
| `model_id` | string \| null | Explicit-mode override. Wins over `architecture` if both supplied. At least one of `architecture` / `model_id` is required (else 4xx). |
| `orientation` | string | `"portrait"` only in v1. |
| `video_label` | string \| null | Optional human-readable upload name (analytics retrieval). Defaults to original filename. |
| `boundary` | `[{x,y}][4]` | Normalised `[0,1]` court corners, any order; backend sorts to canonical TL/TR/BR/BL and computes `H = cv2.getPerspectiveTransform(boundary, ref_court_corners)`. |
| `annotations` | array | One entry per stroke; `N=1` = single-shot UX, `N>1` = rally UX. |
| `annotations[].target_frame` | int | Canonical stroke localisation. |
| `annotations[].region_start_frame`, `region_end_frame` | int | Window for shuttle/court features. `start ≤ target ≤ end`, width ≤ `5*fps`. |
| `annotations[].player_side` | enum \| null | Optional hint (`"top"` \| `"bottom"`); else inferred from court projection. |
| `enabled_sides` | string[] | Filter strokes by side before returning; default both. |
| `player_top_id`, `player_bottom_id` | string \| null | From the typeahead search. If null and `_label` provided, backend creates a new player record with a UUID. |
| `player_top_label`, `player_bottom_label` | string \| null | User-typed display names. |

**Player resolution rules:**

| Upload state | Backend action |
|--------------|----------------|
| `id` provided + exists | Use it; result carries canonical `players.label`. |
| `id` provided + not found | HTTP 4xx error. |
| `id` null + `label` provided | Create new player row with UUID; result carries new `id`. |
| Both null | Result `player_id` and `player_label` are null. |

Backend never invents stroke times — every classified stroke
corresponds to a user-supplied annotation. Stroke detection /
SRA pre-segmentation is a v2 concern.

**Response (202):**

```jsonc
{ "job_id": "abc123" }
```

#### `GET /api/status/{job_id}`

Poll job state.

**Path params:** `job_id`.

**Response:**

```jsonc
{ "job_id": "abc123", "status": "running" }
```

**Fields:** `status` is one of `queued`, `running`, `done`, `error`.

#### `GET /api/results/{job_id}`

Fetch the structured Tier 2 result.

**Path params:** `job_id`.

**Response:**

```jsonc
{
  "job_id":         "abc123",
  "model_id":       "bric_v1_run20260520_xxx",
  "video_label":    "Smith vs Jones, training 2026-05-20",
  "homography_ok":  true,
  "players": {
    "top":    {"id": "p_smith_001", "label": "Smith"},
    "bottom": {"id": "p_jones_004", "label": "Jones"}
  },
  "strokes": [
    {
      "stroke_index":     0,
      "rally_position":   "first",
      "target_frame":     140,
      "timestamp_sec":    4.67,
      "player_side":      "top",
      "player_id":        "p_smith_001",
      "player_label":     "Smith",
      "court_position":   {"x": 0.55, "y": 0.18},
      "stroke_frame_url": "/api/jobs/abc123/frames/0",
      "prediction": {
        "predicted_class": "smash",
        "confidence_pct":  87,
        "top_k": [
          {"class": "smash",       "confidence": 0.87},
          {"class": "wrist_smash", "confidence": 0.06}
        ]
      }
    }
  ],
  "rally_summary": {
    "total_strokes":        3,
    "rally_length_seconds": 6.4
  }
}
```

**Top-level fields:**

| Field | Type | Notes |
|---|---|---|
| `job_id`, `video_label` | string | Echoed from the upload context. |
| `model_id` | string | Resolved model id — the specific registry entry that ran inference. If the upload supplied only `architecture`, this is the entry the backend looked up via `is_default`. |
| `homography_ok` | bool | False if corner fit failed; downstream `court_position` values then null and player-side falls back to a vertical-pixel heuristic. |
| `players.{top, bottom}` | `{id, label}` \| null | Resolved per-side via the [player resolution rules](#post-apiupload). Null for either side if the upload supplied neither id nor label. |
| `strokes` | array | Mirrors `markup.annotations` 1:1 (after `enabled_sides` filter), preserving order. |
| `rally_summary` | object | Aggregates over `strokes[]`. |

**Per-stroke fields:**

| Field | Type | Notes |
|---|---|---|
| `stroke_index` | int | 0-indexed position within `strokes[]`. |
| `rally_position` | enum | `"only"` (single-stroke upload), `"first"`, `"middle"`, `"last"`. Computable from `stroke_index` + `total_strokes`; surfaced for analytics convenience. |
| `target_frame` | int | From the input annotation. |
| `timestamp_sec` | float | `target_frame / fps`. |
| `player_side` | enum | `"top"` \| `"bottom"` \| `"unknown"`. |
| `player_id`, `player_label` | string \| null | Resolved from the players block via `player_side`. |
| `court_position` | `{x,y}` \| null | Striker foot-centre projected via homography to normalised `[0, 1]` over the FULL singles court (top y=0, net y=0.5, bottom y=1). Null when `homography_ok` is false. |
| `stroke_frame_url` | string | Resolves to `GET /api/jobs/{job_id}/frames/{stroke_index}`. |
| `prediction` | object | The model output for this stroke. |
| `prediction.predicted_class` | string | Headline class. Equal to `prediction.top_k[0].class`. |
| `prediction.confidence_pct` | int | `[0, 100]`. Equal to `round(top_k[0].confidence * 100)`. See [Confidence](#confidence). |
| `prediction.top_k` | array | Top 5 by confidence, descending. Each: `{class, confidence}` with confidence as float `[0, 1]`. |

**Per-handler differences within Tier 2:** BRIC populates `player_id`,
`player_label`, and `court_position` from its perception layer. BST may
emit `null` for these (no player-identity tracking, no court projection
in its current pipeline). All other fields are shared.

#### `GET /api/jobs/{job_id}/frames/{stroke_idx}`

Stream a stroke-frame thumbnail JPG.

**Path params:** `job_id`, `stroke_idx` (0-indexed; matches `strokes[].stroke_index`).

**Response:** binary `image/jpeg`.

**Notes:** files at `runtime/jobs/{job_id}/frames/{stroke_idx}.jpg`.
Persist for the lifetime of the `jobs` row in the storage layer — no
automatic TTL in v1. See [`storage.md`](storage.md#frame-thumbnails).

#### `GET /api/players/search`

Typeahead for the upload UX.

**Query params:** `q` (string, prefix), `limit` (int, default 10).

**Response:**

```jsonc
[
  {"id": "p_smith_001", "label": "Smith",  "aliases": ["J. Smith"]},
  {"id": "p_smith_002", "label": "Smithy", "aliases": []}
]
```

Case-insensitive prefix match on `label` and any `aliases[]` entry.

#### `GET /api/players/{id}`

Single player identity record.

**Path params:** `id`.

**Response:**

```jsonc
{
  "id":         "p_smith_001",
  "label":      "Smith",
  "aliases":   ["J. Smith"],
  "created_at": "2026-05-14T13:47:00Z",
  "metadata":   {}
}
```

---

## Mock data and fallback behaviour

Endpoints return mocks in two situations: (a) Tier 1 predictions JSONs
have been mocked because the real eval hasn't run yet; (b) Tier 2's
upload dispatches to an unregistered handler.

### Tier 1 mocks (precomputed predictions on disk)

Backend reads predictions from `experiments/{run_id}/predictions/*.json`.
While a model is being wired, those JSONs may be hand-built mocks.
Mocked artefacts carry a top-level marker:

```jsonc
{ "_mock_data": true, "clips": [ ... ] }
```

Frontend can surface a "mock data" badge when `_mock_data` is true.
Behaviour by endpoint when mocks are in place:

| Endpoint | Mock behaviour |
|---|---|
| `GET /api/registry` | Returns the registry yaml entries as-is. Mocked-eval entries marked with `_mock_data: true` (per-entry, optional). |
| `GET /api/registry/{model_id}/splits/{split}/clips` | Returns the mocked clips list. `total` reflects mock count (~28 in PR #77's mock build). |
| `GET /api/registry/{model_id}/splits/{split}/clips/{stem}` | Returns the mocked single clip. `confidence_pct` and `top_k` are synthesised. |
| `GET /api/registry/{model_id}/splits/{split}/stats` | Returns mocked per-class stats (synthesised confusion patterns). |
| `GET /api/clips/{stem}/video` | `404` if the mock stem doesn't correspond to a real mp4 on disk (typical for synthesised stems). Frontend should render a "no clip available on this host" fallback. |

### Tier 2 mock-fallback

When `markup.model_id` doesn't resolve to a registered handler (e.g.
the BRIC handler isn't wired yet), `POST /api/upload` returns a
v1-shaped Tier 2 result so the frontend's processing-poll loop
exercises end-to-end:

- 3-second sleep (simulates job latency).
- `strokes` mirrors `markup.annotations` 1:1.
- Per stroke: `target_frame`, `player_side`, `player_id`, `player_label`
  echoed from the upload; `prediction` populated with placeholders
  (`predicted_class = "clear"`, `confidence_pct = 50`,
  `top_k = [{class: "clear", confidence: 0.5}]`).
- Envelope: `homography_ok = false`, per-stroke `court_position = null`,
  `stroke_frame_url = ""`.
- If `markup.annotations` is empty/missing, returns one canned stroke
  at `target_frame = 0`.

### Other failure modes

| Endpoint | Status | Body |
|---|---|---|
| Any endpoint with unknown `model_id` | `404` | `{"detail": "Unknown model_id 'X'"}` |
| `GET /api/registry/{model_id}/splits/{split}/...` with unsupported split | `400` | `{"detail": "Split must be one of ['val', 'test']"}` |
| `GET /api/results/{job_id}` before job complete | `409` | `{"detail": "Job still <status>"}` (or just return `null` strokes — backend's choice) |
| `GET /api/jobs/{job_id}/frames/{stroke_idx}` for missing file | `404` | `{"detail": "Frame not found"}` (e.g. mock job that didn't generate thumbnails) |

---

## Confidence

> Open issue: v1 ships uncalibrated `confidence_pct` for both handlers
> (raw softmax). v2 adds post-hoc temperature scaling so the numbers
> match true frequency. Until then, treat the headline as relative
> ordering, not an absolute probability. Tracked as
> [Open conflict](#open-conflicts) #4.

`confidence_pct` is the model's calibrated probability for the top
class, rounded to an integer in `[0, 100]`. "32% confident" means
"across all clips where the model says about 32%, it's right about
32% of the time".

### Why the headline number can look low

When the model is genuinely torn between two similar classes (say
`wrist_smash` vs `smash`), the headline confidence sits in the 30s
even though the model is decisive about it being "some kind of
overhead-aggressive shot". Working as intended; inflating the number
would lie to the user.

The fix is to render `top_k` as a bar chart alongside the headline.
The user instantly sees the close call:

```
wrist_smash  ████████████████░░░░░░  32%
smash        █████████████░░░░░░░░░  27%
drive        ██████░░░░░░░░░░░░░░░░  12%
push         █████░░░░░░░░░░░░░░░░░  10%
drop         ████░░░░░░░░░░░░░░░░░░   8%
```

The headline is the "if forced to pick one" probability. The bars
carry the shape of the uncertainty.

### Suggested visual hierarchy

- **Top of the results card:** predicted class name (large) +
  `confidence_pct` (medium).
- **Below:** bar chart of `top_k` (5 rows).
- **Optional:** "tie" badge when
  `top_k[0].confidence - top_k[1].confidence < 0.10`.

(Section adapted from the BST handoff doc's "Confidence" section.)

---

## Indexing conventions

Reference for any numeric field across the contract.

| Field | Indexing | Notes |
|---|---|---|
| `target_frame`, `region_start_frame`, `region_end_frame`, `frame_idx` | 0-indexed | Matches OpenCV and standard video tooling. |
| `set_id` | 1-indexed string | `"set1"`, `"set2"`, `"set3"`. |
| `rally` | 1-indexed integer | First rally in a set is 1. |
| `ball_round` | 1-indexed integer | First stroke in a rally is 1. |
| `serial_no` | 1-indexed integer | Range 1-5 per training run. |
| `stroke_index` | 0-indexed integer | Position within Tier 2's `strokes[]` array. |
| `clip_stem` | All-1-indexed composite | Format `{vid}_{set}_{rally}_{ball_round}` (e.g. `1_1_1_1` is the first stroke in the dataset). |
| `boundary[].x`, `boundary[].y`, `court_position.x`, `court_position.y` | Normalised `[0, 1]` | Not pixel-space. `[0, 0]` is top-left. |

---

## Open conflicts

Items not yet resolved between this contract and the BST handoff doc /
PR #77 implementation. Each needs a decision before the disputed
surface ships.

| # | Item | This contract | BST handoff / PR #77 | Resolution path |
|---|------|---------------|----------------------|------------------|
| 1 | Tier 2 stroke count per upload | N annotations (rally support) | 1 stroke window in handoff doc; Tier 2 not yet implemented in PR #77 | Subsume single-stroke as N=1 special case (this contract's shape). |
| 2 | "Sample inference" tier — live forward-pass on a known clip (BST handoff's Tier 2, `POST /api/registry/{model_id}/predict/{stem}`) | **Dropped** — this contract uses Tier 1 (precomputed) and Tier 2 (novel upload) only. BST's Tier 3 (novel upload) renumbered to Tier 2 to close the gap. | Defined as Tier 2 | Dropped deliberately — Tier 1 (precomputed) already serves the dashboard view; live-on-known adds latency without a new analytic story. Visible here for BST team awareness. |
| 3 | Registry: `architecture` and `is_default` fields | Documented (drives architecture-grouping + best-per-architecture filter in the picker) | Absent in handoff and in PR #77 implementation | Additive — BST entries can populate `architecture: "bst"` and mark one as `is_default: true`. Without them the picker can't group/filter by architecture. Worth landing. |
| 4 | Calibration of `confidence_pct` | v2 candidate (raw softmax in v1) | v2 candidate per BST status doc ("next couple of weeks") | Aligned in direction. Both handlers ship raw in v1; both add temperature scaling in v2. Documented as parallel deferral. |
| 5 | Storage backend | SQLite (see `storage.md`) | File-based predictions on disk (Tier 1) | Coexist for now — registry can read from either. BST team can adopt SQLite if useful, otherwise no action needed. |

Resolve in a three-way conversation (BRIC author + BST handler author +
frontend wiring owner).
