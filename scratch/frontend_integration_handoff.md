# Frontend Handoff: BST-X Integration Contract

Integration spec for the FE team. No machine-learning background needed. It's all JSON, a handful of endpoints, and a few UX conventions. If anything here doesn't match what you see, or what you planned to do with the UX, tag me on Slack (or call).

I've got a really detailed version if you really want.

## Contents

1. [Three tiers, same UI](#three-tiers-same-ui)
2. [Architecture-agnostic by design](#architecture-agnostic-by-design)
3. [The model registry: how the user picks a model](#the-model-registry)
4. [The universal response shape](#the-universal-response-shape)
5. [Tier 1: read predictions from disk](#tier-1-read-predictions-from-disk)
6. [Tier 2: live inference on a known clip](#tier-2-live-inference-on-a-known-clip)
7. [Tier 3: predict on a brand-new video](#tier-3-predict-on-a-brand-new-video)
8. [Confidence: what it means, how to render it](#confidence-what-it-means-how-to-render-it)
9. [What changes when the user picks a different model](#what-changes-when-the-user-picks-a-different-model)
10. [Endpoint cheat sheet](#endpoint-cheat-sheet)
11. [Open items](#open-items)

Appendix: [Indexing conventions](#appendix-indexing-conventions)

---

## Three tiers, same UI

| Tier | What it does | Where the FE work shifts |
|---|---|---|
| 1 | Reads pre-computed predictions off disk and returns them. | I'd build the whole UI against this tier. |
| 2 | Runs the model live on a clip that's already in the dataset. | Zero change from Tier 1. Same UI, same JSON. |
| 3 | Runs the whole pipeline on a brand-new uploaded video. | One extra step: capturing the four court corners and the clip's start/end frames (existing slider) before submitting. |

The JSON shape coming back from the backend is identical across all three tiers. I think building against Tier 1's static files first is MVP, then Tier 2 and Tier 3 drop in on top of that work.

---

## Architecture-agnostic by design

The contract is built around model IDs, JSON schemas, and class lists from the registry. Nothing in here is specific to my BST-X model line. When Architecture 2 (the RGB-3dCNN-core multi-stream model) lands, slotting it in should be a small change for the FE: at most a couple of new paths in the registry YAML, plus a few new JSON fields if the new architecture emits something unique. The model picker, the per-clip JSON, the clip browser, the per-class panel and the bar chart all stay the same. At most, if we get to Tier 3 and let users analyse their own videos, Architecture 2 *might* want the user to tag the active players (a couple of extra Tier 3 request fields), but probably not even that.

---

## The model registry

The frontend offers a dropdown of "model weights" the user can pick from. The list comes from a single endpoint:

```
GET /api/registry
```

Response:

```json
{
    "models": [
        {
            "id": "bst_x_v1_wipe_drop_s5",
            "display_name": "BST-X v1, wipe_drop run, serial 5 (current best)",
            "description": "14-class taxonomy, augmentation v1 + adaptive focal loss.",
            "taxonomy": "une_merge_v1_nosides",
            "num_classes": 14,
            "class_list": ["net_shot", "return_net", "smash", "wrist_smash", "..."],
            "splits_available": ["val", "test"],
            "test_metrics": {
                "macro_f1": 0.7479,
                "min_f1": 0.5147,
                "accuracy": 0.7675,
                "top2_accuracy": 0.9407,
                "per_class_f1": {
                    "net_shot": 0.8924,
                    "smash": 0.5147
                }
            },
            "val_metrics": { }
        }
    ]
}
```

Things to read off each entry:

- `id`: opaque string. Based on what the API expects, you'll need to pass this back in every subsequent request.
- `display_name`: what to show in the dropdown.
- `class_list`: the array of class names. Order matters: it lines up with the prediction arrays.
- `num_classes`: same as `class_list.length`. Just there for convenience.
- `splits_available`: usually `["val", "test"]`. Controls which split-pickers to show.
- `test_metrics` and `val_metrics`: for any "model accuracy" panel.

Based on what the model outputs, you should read class names from `class_list` rather than hard-coding them. If we swap in a model with a different taxonomy, the names change.

---

## The universal response shape

Every per-clip prediction, across all three tiers, looks like this:

```json
{
    "clip_stem": "35_1_10_17",
    "video_url": "/api/clips/35_1_10_17/video",
    "true_class": "smash",
    "predicted_class": "wrist_smash",
    "is_correct": false,
    "confidence_pct": 32,
    "top_k": [
        {"class": "wrist_smash", "confidence": 0.32},
        {"class": "smash",       "confidence": 0.27},
        {"class": "drive",       "confidence": 0.12},
        {"class": "push",        "confidence": 0.10},
        {"class": "drop",        "confidence": 0.08}
    ],
    "match": "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
    "set_id": "set1",
    "rally": 10,
    "ball_round": 17,
    "split": "test"
}
```

For Tier 3 (novel video), `true_class` and `is_correct` are dropped (no ground truth), and a few extra fields appear. See the Tier 3 section for those.

### Top-k

`top_k` is the top 5 classes by confidence, descending. Confidences sit in `[0, 1]`. The AI-explainability `confidence_pct` is `top_k[0].confidence` rounded to a 0-100 integer.

`video_url` is the streaming endpoint for the clip's mp4. You should be able to just plug it straight into a `<video src="...">` element. The backend supports range requests, so scrubbing works.

---

## Tier 1: read predictions from disk (pretend inference, not live)

Pre-computed predictions on val and test for every registered model. No model gets loaded. The backend just reads JSON files off disk.

**Frontend sends** (URL only, no body):

```
GET /api/registry/bst_x_v1_wipe_drop_s5/splits/test/clips/35_1_10_17
```

**Frontend receives:** the universal response shape above.

**Listing clips:**

```
GET /api/registry/bst_x_v1_wipe_drop_s5/splits/test/clips?limit=50&offset=0&errors_only=true
```

Query params (all optional):

- `limit` and `offset`: pagination.
- `true_class`: filter to clips whose ground-truth class matches.
- `predicted_class`: filter to clips whose top-1 prediction matches.
- `errors_only=true`: only clips where `is_correct == false`.

Returns:

```json
{
    "total": 4202,
    "limit": 50,
    "offset": 0,
    "clips": [
        { }
    ]
}
```

(Each entry in `clips[]` is the universal response shape from above.)

**Per-class stats:**

```
GET /api/registry/bst_x_v1_wipe_drop_s5/splits/test/stats
```

Returns aggregate per-class data for the per-class panel:

```json
{
    "split": "test",
    "n_clips": 4202,
    "class_list": ["net_shot", "return_net", "..."],
    "per_class": {
        "smash": {
            "support_true":   312,
            "support_pred":   289,
            "precision":      0.51,
            "recall":         0.55,
            "f1":             0.5147,
            "top5_when_true": [
                ["smash",       0.55],
                ["wrist_smash", 0.21],
                ["drive",       0.07],
                ["push",        0.04],
                ["drop",        0.03]
            ],
            "top5_when_pred": [
                ["smash",       0.51],
                ["wrist_smash", 0.18],
                ["clear",       0.09],
                ["lob",         0.06],
                ["drive",       0.05]
            ]
        }
    }
}
```

Two views of the confusion matrix per class:

- `top5_when_true`: when the ground-truth class is `smash`, what does the model predict, and how often? (Recall-style.)
- `top5_when_pred`: when the model predicts `smash`, what's actually happening, and how often? (Precision-style.)

Both arrays sum to 1.0.

---

## Tier 2: live inference on a known clip

Same UI as Tier 1, but the backend loads the weights and runs a forward pass instead of reading a precomputed file. Slower per request. Useful for showing the model thinking on demand.

**Frontend sends:**

```
POST /api/registry/bst_x_v1_wipe_drop_s5/predict/35_1_10_17
body:
{
    "split": "test"
}
```

**Frontend receives:** the universal response shape. Identical to Tier 1.

Nothing else differs. The clip-list and stats endpoints from Tier 1 still work. Tier 2 just adds the on-demand single-clip route.

---

## Tier 3: predict on a brand-new video

The user uploads a video (or pastes a YouTube URL we download), marks the four court corners, picks the clip's start and end frames using the existing slider, then hits submit. The backend runs the full pipeline (player tracking, shuttle tracking, court projection, BST forward) and returns a prediction.

**Frontend sends:**

```json
POST /api/end_to_end_predict
body:
{
    "model_id": "bst_x_v1_wipe_drop_s5",
    "video_id": "abc-uuid-or-source-ref",
    "start_frame": 1240,
    "end_frame": 1340,
    "corners": [
        [0.12, 0.18],
        [0.88, 0.18],
        [0.92, 0.95],
        [0.08, 0.95]
    ],
    "orientation": "portrait"
}
```

### About `start_frame` and `end_frame`

The user picks the clip window using the existing slider on the markup screen. Based on what the model needs, you should send the two frame indices that come out of that. The window should contain the stroke you want classified, ideally with a touch of context either side (the prior shot and the return).

**Ideal window:**

- Total clip ≤ 100 frames (about 4 s at 25 fps).

**If it's longer:** don't worry, the model handles it by skipping frames to fit. Accuracy might drop a touch because the motion-feed gets jerkier, but it won't fail.

### About `corners`

Four `[x, y]` pairs, in any order the user clicked them. The backend sorts them into the right convention automatically.

Coordinates are normalised to `[0, 1]` relative to the reference frame resolution (1280×720). So `[0.5, 0.5]` is dead centre regardless of how the frontend rendered the frame. Based on what the backend expects, you'll need to convert from pixel-space to `[0, 1]` before sending.

For v1 we only handle **portrait** orientation, which is what every official badminton broadcast camera produces. You should send `"orientation": "portrait"`. If we ever support landscape, we'll add it as a separate value.

**Frontend receives:**

```json
{
    "clip_stem": "novel_abc-uuid_1240_1340",
    "predicted_class": "wrist_smash",
    "confidence_pct": 32,
    "top_k": [
        {"class": "wrist_smash", "confidence": 0.32},
        {"class": "smash",       "confidence": 0.27},
        {"class": "drive",       "confidence": 0.12},
        {"class": "push",        "confidence": 0.10},
        {"class": "drop",        "confidence": 0.08}
    ],
    "homography_ok": true,
    "frames_actual": 100,
    "frames_strided": false
}
```

- `homography_ok`: `false` if the corner fit failed (typically degenerate clicks). User would need a re-prompt.
- `frames_actual`: number of frames the backend actually fed the model.
- `frames_strided`: `true` if the clip window was longer than 100 frames and the backend skipped frames to fit. Optionally surface this in the UI ("we stretched the window, accuracy may suffer slightly").

Tier 3 could take a while per clip. Depends on the deployment system. Could be helpful to handle wait time via the existing job-queue plus polling pattern (`/api/status/{job_id}`) and avoid blocking the UI on the request.

---

## Confidence: what it means, how to render it

`confidence_pct` is the model's calibrated probability for the top class, rounded to an integer between 0 and 100. I've done the work to make this number meaningful: "32% confident" actually means "across all clips where the model says about 32%, it's right about 32% of the time".

*Aside, for the curious:* the model's raw output uses a `softmax` equation that exaggerates differences. The exaggeration helps it train faster. After training I apply something fancier (a literature-grounded post-hoc calibration) to flatten the over-confidence into an honest read. You don't need the maths to consume the API. The calibrated number is what comes back in `confidence_pct`.

### Why the headline number can look low

When the model is genuinely torn between two similar classes (say `wrist_smash` vs `smash`), the headline confidence sits in the 30s, even though the model is pretty decisive about it being "some kind of overhead-aggressive shot". That's working as intended. Inflating the number would lie to the user. But it's a far cry from the '90% confident!' you'd intuitively expect.

I reckon the fix is to always render `top_k` as a bar chart alongside the headline. The user instantly sees the close call:

```
wrist_smash  ████████████████░░░░░░  32%
smash        █████████████░░░░░░░░░  27%
drive        ██████░░░░░░░░░░░░░░░░  12%
push         █████░░░░░░░░░░░░░░░░░  10%
drop         ████░░░░░░░░░░░░░░░░░░   8%
```

The headline is the "if forced to pick one" probability. The bars carry the shape of the uncertainty. Together they're honest. Showing only the headline misleads users on close calls. Showing both gives them an instant read of "the model's leaning here, but it's not a slam dunk".

### Suggested visual hierarchy

- **Top of the results card:** predicted class name (large) plus `confidence_pct` (medium).
- **Below it:** bar chart of `top_k` (the five rows above).
- **Optional:** a close-call indicator. For example, show a "tie" badge when `top_k[0].confidence - top_k[1].confidence < 0.10`.

---

## What changes when the user picks a different model

When the model dropdown changes:

- **`class_list` changes.** Based on what the model outputs, you'll want to re-render any panel that shows class names (per-class stats, clip browser filters, the bar chart labels).
- **Clip pool changes.** You'll want to re-fetch `/api/registry/{model_id}/splits/{split}/clips`. A given clip might appear in val for one model and test for another, depending on how that model was trained. The active model decides.
- **Confidences re-scale.** The calibration is per-model. Same clip, two models, different confidence numbers.

You never need to know taxonomy details. I think it's easiest just to read `class_list` from the registry response and render against it.

---

## Endpoint cheat sheet

```
GET  /api/registry
GET  /api/registry/{model_id}/splits/{split}/stats
GET  /api/registry/{model_id}/splits/{split}/clips?limit=&offset=&true_class=&predicted_class=&errors_only=
GET  /api/registry/{model_id}/splits/{split}/clips/{stem}
GET  /api/clips/{stem}/video
POST /api/registry/{model_id}/predict/{stem}                  (Tier 2)
POST /api/end_to_end_predict                                  (Tier 3)
GET  /api/status/{job_id}                                     (Tier 3 polling)
```

`{split}` is `val` or `test`. `{stem}` is a clip identifier like `35_1_10_17`. All paths return and accept JSON unless noted. The video endpoint streams binary mp4 with range support.

---

## Open items

Data paths are resolved on the backend via env vars (`BST_X_COLLATED_DATA_ROOT` for collated tensors, the existing `BST_CLIPS_DIR` for clip videos). The frontend never sees them. Frontend contract is the JSON, not the filesystem.

If you spot anything here that's ambiguous, or that conflicts with the existing frontend stubs, let me know and I'll fix it. I'd build against Tier 1 first; everything else follows the same shape.

---

## Appendix: Indexing conventions

Reference table for any numeric field across the contract. Useful when something looks off-by-one.

| Field | Indexing | Notes |
|---|---|---|
| `start_frame` and `end_frame` | **0-indexed** | Frame 0 is the first frame of the video. Matches OpenCV and standard video tooling. |
| `set_id` | **1-indexed** string | `"set1"`, `"set2"`, `"set3"`. |
| `rally` | **1-indexed** integer | First rally in a set is 1. |
| `ball_round` | **1-indexed** integer | First stroke in a rally is 1. |
| `serial_no` | **1-indexed** integer | Range 1-5 per run. |
| `clip_stem` | All-1-indexed composite | Format `{vid}_{set}_{rally}_{ball_round}` (e.g. `1_1_1_1` is the very first stroke in the entire dataset). |
| Corner coordinates | **Normalised `[0, 1]`** | Not pixel-space. `[0, 0]` is top-left, `[1, 1]` is bottom-right. |
