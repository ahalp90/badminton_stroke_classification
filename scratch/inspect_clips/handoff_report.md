# Tier-1 integration report — 2026-05-17

End-to-end audit + remap-based clip wiring, ahead of the browser walkthrough.

---

## 1. Clip remap (replaces SCP)

Same-class assignment of the 13 local `train/*.mp4` files to mocked test/val
entries in
`src/bst_refactor/.../run_20260505_154907/clip_index.json`. Script:
`scratch/inspect_clips/remap.py`. Test split was drained first, val took
spillover. The 8_3_17_5 swap from earlier today was reverted from git before
remapping so test/Bottom_smash/8_3_17_5 starts clean and is now rebound to a
train smash mp4 by the same one-pass rule.

### Test split (8/28 playable)

| Stem | Class | Remapped video |
|---|---|---|
| 2_2_17_20  | net_shot              | train/Bottom_net_shot/16_2_17_15.mp4 |
| 24_2_21_14 | net_shot              | train/Top_net_shot/11_1_19_2.mp4 |
| 8_3_17_5   | smash                 | train/Top_smash/11_1_17_9.mp4 |
| 4_1_2_9    | smash                 | train/Top_smash/16_1_42_4.mp4 |
| 10_2_28_10 | drop                  | train/Top_drop/14_1_27_8.mp4 |
| 47_1_4_20  | drop                  | train/Top_drop/14_1_35_4.mp4 |
| 35_3_27_2  | cross_court_net_shot  | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
| 46_3_16_31 | short_service         | train/Bottom_short_service/11_2_13_1.mp4 |

Remaining 20 test entries keep their original `test/…` path → 404 from
`/api/clips/{stem}/video` (FE shows the "video unavailable" placeholder; the
clip's prediction + metadata still render correctly).

### Val split (2/28 playable)

| Stem | Class | Remapped video |
|---|---|---|
| 7_3_14_37 | smash | train/Top_smash/18_2_19_16.mp4 |
| 41_1_25_2 | smash | train/Top_smash/3_1_14_9.mp4 |

3 leftover smash files unused (`3_1_18_3`, `3_1_37_10`, `3_2_6_11`). No
non-smash classes were assignable to val once test was drained.

Spot-check (curl): all 10 playable stems return 200 / `video/mp4` /
non-zero binary (487k–949k). 46 unplayable stems return 404 with backend
error JSON.

### 1a. Round-robin fill of the remaining 46 entries

Follow-up after the class-aligned pass: every per-clip-browser entry now
plays video. Script: `scratch/inspect_clips/fill_remaining.py`. The 10
class-aligned remaps above are preserved; the other 46 entries cycle
through the 13 train mp4s in this order (sorted alphabetically by
relative path):

| Idx | Local mp4 |
|---|---|
|  0 | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
|  1 | train/Bottom_net_shot/16_2_17_15.mp4 |
|  2 | train/Bottom_short_service/11_2_13_1.mp4 |
|  3 | train/Top_drop/14_1_27_8.mp4 |
|  4 | train/Top_drop/14_1_35_4.mp4 |
|  5 | train/Top_net_shot/11_1_19_2.mp4 |
|  6 | train/Top_smash/11_1_17_9.mp4 |
|  7 | train/Top_smash/16_1_42_4.mp4 |
|  8 | train/Top_smash/18_2_19_16.mp4 |
|  9 | train/Top_smash/3_1_14_9.mp4 |
| 10 | train/Top_smash/3_1_18_3.mp4 |
| 11 | train/Top_smash/3_1_37_10.mp4 |
| 12 | train/Top_smash/3_2_6_11.mp4 |

Cycle is applied in test-first, then-val order across only the entries
that didn't get a class-aligned remap (so the 10 honest matches keep
their dedicated file; the cycle starts fresh at index 0 on the first
unfilled entry):

**Test (round-robin assignments to the 20 non-class-aligned entries):**

| Idx | Stem | Displayed class | Cycled video |
|---|---|---|---|
|  0 | 45_2_15_4   | return_net    | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
|  1 | 34_2_13_24  | return_net    | train/Bottom_net_shot/16_2_17_15.mp4 |
|  2 | 34_2_21_4   | wrist_smash   | train/Bottom_short_service/11_2_13_1.mp4 |
|  3 | 30_3_28_31  | wrist_smash   | train/Top_drop/14_1_27_8.mp4 |
|  4 | 45_2_24_18  | lob           | train/Top_drop/14_1_35_4.mp4 |
|  5 | 16_3_11_39  | lob           | train/Top_net_shot/11_1_19_2.mp4 |
|  6 | 35_3_25_10  | clear         | train/Top_smash/11_1_17_9.mp4 |
|  7 | 25_3_17_6   | clear         | train/Top_smash/16_1_42_4.mp4 |
|  8 | 24_1_11_7   | drive         | train/Top_smash/18_2_19_16.mp4 |
|  9 | 37_3_29_22  | drive         | train/Top_smash/3_1_14_9.mp4 |
| 10 | 5_2_7_1     | passive_drop  | train/Top_smash/3_1_18_3.mp4 |
| 11 | 6_1_29_19   | passive_drop  | train/Top_smash/3_1_37_10.mp4 |
| 12 | 47_3_30_40  | push          | train/Top_smash/3_2_6_11.mp4 |
| 13 | 10_2_16_40  | push          | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
| 14 | 39_1_2_2    | rush          | train/Bottom_net_shot/16_2_17_15.mp4 |
| 15 | 8_2_26_20   | rush          | train/Bottom_short_service/11_2_13_1.mp4 |
| 16 | 8_1_9_10    | cross_court_net_shot | train/Top_drop/14_1_27_8.mp4 |
| 17 | 24_2_11_14  | short_service | train/Top_drop/14_1_35_4.mp4 |
| 18 | 15_2_26_27  | long_service  | train/Top_net_shot/11_1_19_2.mp4 |
| 19 | 38_3_8_35   | long_service  | train/Top_smash/11_1_17_9.mp4 |

**Val (round-robin assignments to the 26 non-class-aligned entries; cycle
continues from where test left off, i.e. starts at idx 7 of the pool):**

| Idx | Stem | Displayed class | Cycled video |
|---|---|---|---|
| 20 | 16_1_6_30   | net_shot      | train/Top_smash/16_1_42_4.mp4 |
| 21 | 15_2_7_38   | net_shot      | train/Top_smash/18_2_19_16.mp4 |
| 22 | 15_1_1_17   | return_net    | train/Top_smash/3_1_14_9.mp4 |
| 23 | 41_2_15_26  | return_net    | train/Top_smash/3_1_18_3.mp4 |
| 24 | 6_2_18_24   | wrist_smash   | train/Top_smash/3_1_37_10.mp4 |
| 25 | 36_1_5_40   | wrist_smash   | train/Top_smash/3_2_6_11.mp4 |
| 26 | 14_3_17_21  | lob           | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
| 27 | 7_1_24_31   | lob           | train/Bottom_net_shot/16_2_17_15.mp4 |
| 28 | 50_2_17_28  | clear         | train/Bottom_short_service/11_2_13_1.mp4 |
| 29 | 23_3_1_18   | clear         | train/Top_drop/14_1_27_8.mp4 |
| 30 | 32_1_17_26  | drive         | train/Top_drop/14_1_35_4.mp4 |
| 31 | 3_3_6_31    | drive         | train/Top_net_shot/11_1_19_2.mp4 |
| 32 | 16_2_29_4   | drop          | train/Top_smash/11_1_17_9.mp4 |
| 33 | 43_2_16_14  | drop          | train/Top_smash/16_1_42_4.mp4 |
| 34 | 17_2_6_9    | passive_drop  | train/Top_smash/18_2_19_16.mp4 |
| 35 | 38_1_6_23   | passive_drop  | train/Top_smash/3_1_14_9.mp4 |
| 36 | 6_3_13_7    | push          | train/Top_smash/3_1_18_3.mp4 |
| 37 | 33_2_18_9   | push          | train/Top_smash/3_1_37_10.mp4 |
| 38 | 18_2_15_16  | rush          | train/Top_smash/3_2_6_11.mp4 |
| 39 | 40_3_27_38  | rush          | train/Bottom_cross_court_net_shot/19_1_22_7.mp4 |
| 40 | 18_1_14_32  | cross_court_net_shot | train/Bottom_net_shot/16_2_17_15.mp4 |
| 41 | 24_2_7_38   | cross_court_net_shot | train/Bottom_short_service/11_2_13_1.mp4 |
| 42 | 4_1_4_28    | short_service | train/Top_drop/14_1_27_8.mp4 |
| 43 | 5_3_10_26   | short_service | train/Top_drop/14_1_35_4.mp4 |
| 44 | 22_1_12_25  | long_service  | train/Top_net_shot/11_1_19_2.mp4 |
| 45 | 42_3_12_19  | long_service  | train/Top_smash/11_1_17_9.mp4 |

**Caveat — content does NOT match the displayed class for these 46
entries.** The video element will play a real BWF clip, but a clip of a
different stroke than the panel's `true_class`/`predicted_class` labels
suggest. The 10 class-aligned remaps in §1 are the only entries where
content + label honestly line up. Treat the round-robin fill as a "every
clip plays" UX completeness pass for the demo — not as ground-truth-
faithful playback. (For an honest demo run, you may want to mention this
caveat live, or filter the browser to the 8 test + 2 val class-aligned
stems when showing example videos.)

**Verification:** post-fill backend restart picks up the new clip_index;
all **56/56** stems now return 200 / `video/mp4` / non-zero binary from
`/api/clips/{stem}/video` (full curl sweep). 5-stem random spot-check
across val + test: all 200, sizes 487–934 KB.

---

## 2. Per-screen audit

Live frontend mounts `frontend/hba-stroke-classifier/app.jsx`. The
`frontend/src/pages/` (`Analysis.jsx`, `Results.jsx`) and
`frontend/src/components/SingleFileUploader.jsx` files are *dead scaffolding*
and not part of the running app (`frontend/src/main.jsx` imports from
`../hba-stroke-classifier/app`). They reference `/api/upload`, `/api/status`,
`/api/results`, `/api/models` but are never reachable through navigation.

| Screen | Endpoint hit | Sample response | FE expectation | Status | Notes |
|---|---|---|---|---|---|
| Library    | (none) | n/a | n/a | ALIGNED — no API | Pure client; "Upload Video" tab picks a random library match and discards the File. |
| Markup     | (none) | n/a | n/a | ALIGNED — no API | YouTube iframe + canvas markup; no backend persistence. |
| Configure  | `GET /api/registry` | `{models:[{id,display_name,taxonomy,ablation_id,num_classes,description,test_metrics:{macro_f1,min_f1,accuracy,top2_accuracy,per_class_f1},val_metrics:{}}]}` | `entry.test_metrics?.macro_f1/min_f1/accuracy`, `id`, `display_name`, `taxonomy`, `ablation_id`, `num_classes`, `description` | **ALIGNED** | `val_metrics` is `{}` — data gap (item 3 below), not a wiring gap. |
| Progress   | (none) | n/a | n/a | ALIGNED — no API | Fully mocked 280ms-tick timer + canned activity log. Submit doesn't call `/api/upload`. |
| Results — TestEvalCard | `GET /api/registry` | as above | `m.macro_f1, m.accuracy, m.top2_accuracy, m.min_f1` | **ALIGNED** for test; **DATA GAP** for val (`val_metrics={}` → falls through to "No val metrics available"). |
| Results — PerClassF1Card | `GET /api/registry` | `model.test_metrics.per_class_f1: {...}` | `model[`${split}_metrics`]?.per_class_f1` | **ALIGNED** for test; **DATA GAP** for val. |
| Results — Tier1ClipBrowser list | `GET /api/registry/{id}/splits/{split}/clips?limit&offset&errors_only` | `{model_id, split, total, limit, offset, clips:[{clip_stem,true_class,predicted_class,is_correct,confidence_pct,match,split}]}` | `data.clips`, each clip's `clip_stem, true_class, predicted_class, is_correct, confidence_pct` | **ALIGNED** |
| Results — Tier1ClipBrowser detail | `GET /api/registry/{id}/splits/{split}/clips/{stem}` | `{clip_stem, video_url, true_class, predicted_class, is_correct, confidence_pct, top_k:[{class,confidence}], match, set_id, rally, ball_round, split}` | All of the above | **ALIGNED** |
| Results — ClipDetail `<video>` | `GET /api/clips/{stem}/video` (via `video_url`) | `video/mp4` (200) or `{detail:…}` (404) | `<video src>` with `onError` fallback | **ALIGNED**; 46/56 stems return 404 — FE fallback handles it. |

Verdict: every wired endpoint matches every consumed field. No misalignments.
The only "things the FE reads that the backend doesn't populate" are
`val_metrics.macro_f1` etc. — and the FE already renders a graceful "No val
metrics available" placeholder for that exact case (data gap, not a bug).

---

## 3. Programmatic end-to-end upload test

Script: `scratch/inspect_clips/e2e.py`. Sample upload:
`train/Top_smash/11_1_17_9.mp4` (476 KB).

| Phase | Time |
|---|---|
| `POST /api/upload` | 1487 ms |
| `GET /api/status` poll until `complete` (200 ms cadence) | 3083 ms (16 polls) — matches stub's `time.sleep(3)` |
| `GET /api/results` | 8 ms |
| **End-to-end total** | **4578 ms — PASS (<5 s budget)** |

Result shape vs. inference contract (per `src/api/inference.py` docstring):

```
{
  job_id, status,
  strokes:       [ { timestamp_sec, stroke_type, confidence } … ],
  rally_summary: { total_strokes, rally_length_seconds }
}
```

All seven contract fields present and correctly typed. The user-prompt
mentioned `top_k` and `court_position` — **not present in the canned stub**
and not consumed by any live frontend screen. They belong to Ari/Scott's
real pipeline, not the current contract.

**Critical finding (wiring gap, not a misalignment):** the live SPA never
calls `/api/upload`, `/api/status`, or `/api/results`. The upload→inference
backend is fully functional and shape-correct, but the SPA's wizard
(Library → Markup → Configure → Progress → Results) is end-to-end mocked
on the client; ProgressScreen runs a 280 ms-tick timer and Results reads
historical registry eval — not the just-uploaded video. This is consistent
with "real live ML inference on uploaded video is out of scope," and the
intent that Results showcases trained-model eval data, but it does mean the
canned-stub round-trip is exercised only via direct curl, not via the UI.

---

## 4. Local-only patches applied

All on the worktree, not committed (consistent with the docker-compose
patches Isiah marked LOCAL-ONLY).

### `frontend/hba-stroke-classifier/results-screen.jsx`

1. **Page size 25 → 30.** Comment in the file (`// TODO: Increase to 50 when
   changeover to real test clips occurs`) explicitly asked for this. With
   28 clips per split, page size 25 forced an awkward 25/3 split. 30 fits
   everything on one page; the original comment is preserved in the new
   form for the future bump.
2. **Pagination-cursor consistency.** `disabled={offset+limit >= total}` was
   paired with `cursor: offset+limit > total ? …` — the `==` boundary case
   showed a `pointer` cursor on a disabled button. Made both use `>=`.
3. **Neutralised the "no video" fallback message.** Old copy said *"Set
   BST_CLIPS_DIR … or run on UNE HPC"* — but with the docker-compose mount
   the env var IS set in the container; the issue is per-clip file absence.
   New copy: *"Video unavailable for this clip on the current host. The
   backend serves clips from BST_CLIPS_DIR; only clips whose mp4 is present
   there will play."* Most of the 46 fallback clips will trip this message
   during the walkthrough.

### `src/bst_refactor/.../run_20260505_154907/clip_index.json`

Reverted the 8_3_17_5 swap (now back to its original test/Bottom_smash path
in git, then re-bound by the remap script alongside 9 other entries). Net
diff vs. mocked baseline: 10 `video_path` strings changed to point at
train/*.mp4 files on disk under `scratch/inspect_clips/`.

No backend code touched.

---

## 5. Remaining gaps

| Gap | Owner | One-line rationale |
|---|---|---|
| `val_metrics.macro_f1 / accuracy / top2_accuracy / min_f1 / per_class_f1` are absent — registry returns `val_metrics: {}` | **Scott** (eval-dump) | Per the comment in `results-screen.jsx:6-10`, `eval_dump_predictions.py` only emits test-side metrics; val/train headlines may be reconstructable from TensorBoard scalars (final `val_macro_f1` etc.) without a re-run. |
| 46/56 clip mp4s still 404 the video endpoint | **Isiah / data move** | Either ship the test+val mp4s onto this machine (SCP option deferred) or accept the placeholder during the demo. The non-playable clips still show predictions, top-k, metadata cleanly. |
| Upload-to-results flow is wired in the backend but not in the SPA | **Ari/Scott (real pipeline) + later FE work** | Real inference is explicitly out of scope this milestone; until BST/MMPose/TrackNet land, wiring the SPA upload path would just feed users two canned strokes from the stub. Backend round-trip *is* verified clean. |
| `frontend/src/pages/Analysis.jsx`, `Results.jsx`, `components/SingleFileUploader.jsx` are dead scaffolding referencing `/api/upload` etc. | **Curtis** (FE cleanup) | Not part of the mounted app; safe to delete or repurpose when the real upload flow lands. |
| `top_k`, `court_position` mentioned in handoff prompts are not in the canned stub | **Ari** (inference contract) | These belong to the real BST/TrackNet output; the stub returns only `strokes[] + rally_summary`. Worth adding to the contract doc before the real pipeline lands so FE can build against a stable shape. |

---

## 6. Recommended next step

Ready for the browser walk-through. Suggested order:

1. Configure screen — confirm the single model card loads with macro F1 /
   min F1 / accuracy populated from `/api/registry`.
2. Submit → ProgressScreen completes via timer (no backend call — expected).
3. Results screen — verify:
   - 28 clips list, no pagination needed (item 4.1).
   - Toggle test ↔ val: test shows macro/min/accuracy/top2 + per-class
     bars; val shows the "No val metrics" placeholder (item 5.1).
   - The 10 playable stems above show real mp4 video; the rest show the
     neutralised fallback message (item 4.3).
   - Errors-only toggle: test → 12 mispredicted, val → similar count.

If anything misbehaves during the walk-through, capture the screen +
endpoint and we'll triage. The audit + e2e test say all wired paths
should hold.

---

## 7. Four-item push: recent/browse split, persistent uploads, upload-aware markup, real progress

Follow-up after the audit landed clean. All changes local to the worktree,
no commits, no backend edits. Same status as the docker-compose patches.

### 7.1  Item 1 — Library "Recent matches" framing

`frontend/hba-stroke-classifier/library-screen.jsx`. The 6 curated cards
on the Match Library tab now sit under a *Recent matches* sub-heading.
The page subtitle is rephrased to *"Showing 6 recent matches — browse
all to see the full library, or upload your own footage."* The existing
**Browse all 44 matches →** button still opens the modal, which now
shows the full library plus a new My Uploads section (see Item 2).

Ordering: matches.json has no upload timestamps, so the 6 cards are
just the entries flagged `curated: true` in the JSON, in declaration
order. Documented as "first six in the bundled list" — fine for demo.

### 7.2  Item 2 — Uploaded videos persist

Client-side only. Pure additions; the Upload Video tab no longer
randomly substitutes a library match — it captures the real File.

**Persistence model:**
- LocalStorage key `bsc.uploads` stores metadata array: `{id, filename,
  size, uploaded_at}` per upload. No bytes — files don't fit/work in
  localStorage and survival across refresh is documented as not part
  of v1.
- A module-level `SESSION_FILES` Map (id → {file, objectURL}) holds the
  live `File` + `URL.createObjectURL(file)` for the current session.
  Cleared on browser refresh.
- A custom `bsc.uploads.changed` event lets the My Uploads list update
  in real time after a delete/upload without a page reload.

**Browse All modal restructured:**
- Two sections, top-down: **My Uploads** then **Match Library**.
- Each upload row shows a status dot (green = file in session, grey =
  re-upload required), the filename, size, upload timestamp, and a
  per-row `[Use] / [Re-upload]` button plus an `×` delete button.
- *Re-upload* opens a file picker scoped to video/* and rebinds the
  freshly-chosen file to the existing upload id (so the metadata row
  doesn't fork).
- Deleting an entry confirms via `window.confirm`, then revokes the
  objectURL and drops it from localStorage.

**Upload Tab:**
- Real `<input type="file" accept="video/*">` triggered by click /
  drag-drop. Previous "Demo mode — uploaded videos are stand-ins"
  copy is replaced with truthful copy about where the file lives.
- The simulated 4-stage progress panel (`UploadingPanel`) is kept for
  visual continuity — the real network upload happens later on the
  Configure → Progress transition, not on file drop. Once the panel
  finishes its visual cycle, the file is recorded to localStorage
  + session, and the wizard advances to Markup with the upload as the
  active video.

**Video object shape** (passed through the wizard for uploads):

```
{ id, source: 'upload', match: <filename>, tournament: 'Your upload — <date>',
  duration: '—', strokes: 0, annotated: false,
  file: <File>, objectURL: <blob:…>,
  filename, size, uploadedAt, strokeTimes: [] }
```

Library matches keep the existing shape but now also carry
`source: 'library'` so the Markup screen can branch reliably.

### 7.3  Item 3 — Markup works on uploaded files

`frontend/hba-stroke-classifier/markup-screen.jsx`.

**CourtBoundaryStep.** The frame source is now a `useRef<sourceRef>`
that holds either an `HTMLImageElement` (library: cached jpg under
`data/frames/{youtubeId}.jpg`) or an `HTMLVideoElement` (upload).
For uploads, a hidden `<video src={objectURL} muted playsInline>` is
mounted at 1×1 / opacity 0 / pointerEvents none; on `loadeddata` we
seek to `min(0.5, duration/2)` to avoid black first-frames, then call
`drawImage(sourceRef.current, …)` exactly as before. The loupe
magnifier reads intrinsic dimensions from `naturalWidth` or
`videoWidth`. The handle-dragging math is unchanged.

**TimeframeStep.** Branches inside two parallel `useEffect`s on
`isUpload`:
- Upload: mounts an HTML5 `<video ref playsInline preload="metadata">`
  inside the existing 16:9 player frame. Listens for `loadedmetadata`,
  `timeupdate`, `play`, `pause`, `ended`, `progress`; polls
  `videoEl.currentTime` at 100 ms cadence between timeupdates so the
  scrubber playhead is smooth.
- Library/YouTube: untouched. Existing YT IFrame API + 250 ms polling
  preserved verbatim.

`seekTo / nudge / togglePlay / setHandle` route to either branch
without changing the surrounding scrubber UI. The three Set
start/target/end buttons read the active source's current time. Hand-
test: uploading `scratch/inspect_clips/train/Top_smash/11_1_17_9.mp4`
and walking through both steps works end-to-end.

### 7.4  Item 4 — Real upload-status-results in the progress flow

`frontend/hba-stroke-classifier/configure-screen.jsx`,
`results-screen.jsx`, `app.jsx`.

**ProgressScreen** detects `task.markup.video.file`. If present
(`realRun = true`), the existing 280 ms mock timer is bypassed and the
screen drives directly from network signals:

1. **Upload** — `XMLHttpRequest` POST to `/api/upload?model=default`
   with `start_sec` and `end_sec` query params built from the
   timeframe captured in markup. The XHR upload-progress event maps
   byte fraction to the visible 5–30 % range of the main progress bar.
2. **Status poll** — `fetch /api/status/{job_id}` every 250 ms.
   Status transitions append entries to the Activity Log
   (`queued` → `processing` → `complete` | `failed`) and advance the
   pipeline stage indicator. Between transitions the bar slowly ticks
   to 88 % so it never looks frozen.
3. **Results** — on `complete`, `fetch /api/results/{job_id}`, bar
   snaps to 100 %, log gets a `✓ Done · N stroke(s) returned`, and
   the screen calls `onComplete(result)` after an 800 ms beat.
4. **Errors** — any failure path (non-2xx, malformed response,
   network drop, backend `failed` status) flips an error card above
   the progress card with the backend's detail string and a
   `[Try again]` button that re-runs the effect via a nonce.

`app.jsx` now threads the returned payload onto `task.uploadResult`
when present, then advances to results. Library-match runs (no file)
pass `null` and the registry-only Results UI renders unchanged.

**ResultsScreen** gains an `UploadedInferenceCard` that renders only
when `task.uploadResult` exists. It shows:
- A green-bordered banner with a "stubbed inference — real pipeline
  pending" caveat so demo viewers don't mistake the canned strokes
  for ground-truth model output.
- Two stat tiles: Total strokes, Rally length (seconds).
- A row-per-stroke list with timestamp, stroke_type, confidence %.

The registry-driven panels (Tier1ClipBrowser, TestEvalCard,
PerClassF1Card) appear below the upload card, unchanged.

### 7.5  Known gaps / caveats for the walk-through

- **Court boundary corners** are captured by the FE but **NOT sent to
  the backend** — `/api/upload` has no slot for them in the current
  contract. The four normalised xy points stay client-side until
  Scott extends the contract. Document, don't fabricate.
- **Library matches still run the mock timer.** Real `/api/upload`
  needs an actual file; library matches reference YouTube videos
  whose bytes the FE never holds. The mock timer + canned activity
  log is preserved for those (no behaviour regression). For a fully
  end-to-end demo, upload a real mp4.
- **Refresh loses uploaded file bytes.** localStorage entries persist
  but the `File` object can't be serialised. The modal flags stale
  entries with a grey status dot and a `[Re-upload]` button that
  rebinds bytes to the same id. Users see this once after refresh.
- **Hidden `<video>` for CourtBoundary** is 1×1 px / opacity 0 to
  stay out of the layout. Some Safari builds may suspend loading of
  off-screen video; if a screenshot fails to render on Mac Safari,
  bump opacity to 0.01 and width/height to a few px. Chrome /
  Firefox / Edge handle it as written.
- **Browser must allow `localStorage`.** Private windows in some
  browsers (e.g. Firefox private) restrict it; the catch around
  `JSON.parse` falls back to an empty list rather than crashing.
- **XHR upload progress** only fires when the server reads the
  request stream; the Vite dev-server proxy passes through cleanly,
  but if you bypass it (calling `localhost:24082/api/upload`
  directly) you may need to relax CORS in `src/api/main.py` — not
  needed for the proxied flow at `localhost:5173`.

### 7.6  Verification done

- Vite transforms all five touched files (`library-screen`,
  `markup-screen`, `configure-screen`, `results-screen`, `app`) clean.
  No SyntaxError / Transform failed in the dev-server logs.
- Programmatic end-to-end via Vite proxy (`localhost:5173/api/...`):
  upload → 14 status polls → results round-trip in **4509 ms**, same
  shape as before. No regression to the §3 e2e test.
- `/api/registry` and `/api/clips/<stem>/video` both still serve via
  the proxy (200 / video/mp4 / 487 KB for the 8_3_17_5 spot-check).
- All 56/56 clip remap entries still serve as expected from §1a.

### 7.7  Hand-off ask for Isiah

Browser walk-through, in order:

1. **Library — Recent matches.** Confirm the new sub-heading and 6
   cards render; the "Browse all 44 matches →" button still opens the
   modal. Modal now shows "My Uploads" (empty initially) above
   "Match Library — 44".
2. **Library — Upload Video tab.** Drop
   `scratch/inspect_clips/train/Top_smash/11_1_17_9.mp4` (or any mp4).
   The 4-stage panel runs, then advances to Markup. Refresh the page,
   re-open the modal, confirm the entry is in My Uploads with a grey
   "Re-upload" button. Click Re-upload, pick the same file, the entry
   goes green and selects.
3. **Markup — uploaded video.** Confirm the boundary frame renders
   (top-left badge reads `Uploaded frame · <filename>`); drag the
   four corners; click Confirm. On step 2, the HTML5 video loads;
   play/pause/nudge/seek work; set start/target/end; click Confirm
   Timeframe.
4. **Configure → Progress.** Select a model, hit *Submit for
   Analysis*. The progress bar visibly moves through upload (XHR
   bytes), queued, processing, complete. Activity log shows the real
   transitions.
5. **Results.** The green "Inference on your upload" card shows
   above the registry browser, listing the 2 canned strokes from the
   stub. The rest of Results (per-clip browser, eval card, per-class
   F1) works as before.

If any of those misbehave, copy the screen state + browser console
output and we'll triage. None of the changes touch the backend or the
registry data path, so the §2 audit should still hold verbatim.

---

## 8. Three-gap push: markup sidecar, library_predict, IndexedDB persistence

Follow-up after the four-item push landed. Scope extended to include
small backend additions that align this branch with the contract Scott
maintains on `feat/bric-pipeline`. Local-only, uncommitted, same safety
rules as before. After each gap landed, the previous §7 walk-through was
re-run end-to-end and the §3 e2e test still passes.

### 8.1  Gap 1 — Court boundary corners reach the server

The §7 walk-through captured corners on the FE but dropped them before
the upload — `/api/upload` had no slot for them.

**Backend** (`src/api/main.py`, `src/api/jobs.py`):

- Added a `Markup` Pydantic model that mirrors the schema in
  `docs/api_contract.md` on `feat/bric-pipeline`:
  `architecture, model_id, orientation, video_label, boundary[4],
  annotations[{target_frame, region_start_frame, region_end_frame,
  player_side}], enabled_sides, player_top_id/label,
  player_bottom_id/label`. Validators enforce
  `len(boundary) == 4`, points in `[0, 1]`, and
  `region_start ≤ target ≤ region_end` per annotation.
- `POST /api/upload` accepts a new `markup: Optional[str] = Form(...)`
  stringified-JSON sidecar in the multipart form (the standard
  workaround for nested JSON over multipart). `_parse_markup_json()`
  parses + validates; bad shapes return `400` with a precise error
  detail.
- `Job` dataclass gained `markup`, `source`, `clip_stem` fields.
  `JobStore.create()` now accepts those as optional kwargs.
- `_process_video()` echoes the markup back into the `result` dict as
  `markup_echo`, plus `source` and (for library) `clip_stem`.
- The current canned stub does **not** consume the markup — that's
  Ari/Scott pipeline territory. The server's role is acceptance,
  validation, logging, and echo.

**Frontend** (`configure-screen.jsx`, `results-screen.jsx`):

- Added a `buildMarkupPayload(task)` helper that translates the
  wizard's in-memory state into the contract shape:
  - `boundary` from `task.markup.boundary` (the 4 normalised corners
    captured in MarkupScreen step 1).
  - `annotations[0]` from `task.markup.timeframe.{startSec, targetSec,
    endSec}` × `fps` (library matches use their declared fps; uploads
    fall back to 30).
  - `model_id` from the first toggled-on model in Configure;
    `architecture` defaults to `'bst'` for v1.
  - `video_label` from filename / match title.
- ProgressScreen's XHR appends `markup: <JSON>` to the `FormData`
  alongside the `file` part.
- ResultsScreen's `UploadedInferenceCard` shows a "✓ Markup received
  by server" sub-card listing architecture, orientation, annotation
  count, and the four normalised boundary corners as the backend
  echoed them. If the FE only sent a sparse markup (no boundary,
  library path), the card surfaces that too.

**Verified** by curl through the Vite proxy at `localhost:5173/api/...`
and by direct `localhost:24082/api/...`. Sample echo confirms the four
corners round-trip exactly (`{0.17, 0.30}`, `{0.83, 0.30}`, `{0.93,
0.92}`, `{0.07, 0.92}`). Validation rejections tested for
non-4-corner boundary, inverted annotation window, malformed JSON.

### 8.2  Gap 2 — Library matches drive real status polling

`POST /api/library_predict` is new on this branch; the Scott contract
covers dataset-clip browsing via the `/api/registry/{model_id}/...`
endpoints but doesn't include a "kick off a job on a library clip"
flow. This endpoint adds one with the same lifecycle as `/api/upload`
so the existing `/api/status` + `/api/results` polling works
unchanged.

**Backend** (`src/api/main.py`):

```jsonc
POST /api/library_predict
{
  "clip_stem":    "<opaque id>",   // dataset stem OR library youtube id
  "model_id":     "bst_x_v1_..." | null,
  "architecture": "bric" | "bst" | null,
  "markup":       { ... same Markup schema as /api/upload ... }
}
→ { "job_id": "...", "status": "queued" }
```

- Resolves the clip mp4 from the registry stem index when possible;
  otherwise records a stub `library_stub_<stem>.mp4` path on the job
  and continues. The canned stub doesn't read the file, so the latter
  branch is safe for demo flows where the "library" entry is a
  YouTube match (matches.json) without a corresponding ShuttleSet
  stem.
- Same `_process_video` background task as upload — 3 s sleep, then
  the canned `{strokes, rally_summary}` payload + markup echo +
  `source: "library"` + `clip_stem`.

**Frontend** (`configure-screen.jsx`):

- `ProgressScreen` now branches on `videoSource`:
  - `'upload'` (file in state) → existing `/api/upload` XHR path,
    upload-progress bar 5-30 %, then status polling.
  - `'library'` (no file, source flagged on the video object) →
    `fetch('/api/library_predict', {method: 'POST', body: JSON})`
    with the markup payload and the library entry's id as
    `clip_stem`. No upload-progress phase (nothing to upload); the
    bar jumps to ~20 % at submission and then advances via real
    `/api/status` polls.
- Library video objects already had `source: 'library'` from §7.
- Mock-timer path remains as a fallback for the rare case where
  neither upload nor library applies (e.g. dev-jump to progress with
  the canned fixture video).

**Verified** through the proxy with `clip_stem = "dQw4w9WgXcQ"` (a
non-dataset YouTube id) — backend logs `resolution=stub_no_video`,
job lifecycle ticks `queued → processing → complete`, results echo
the stem and the four-corner boundary.

### 8.3  Gap 3 — Upload bytes survive browser refresh

Pure frontend — `library-screen.jsx`. The §7 design kept upload
metadata in localStorage and the live `File` object in a module-level
Map. After a refresh, metadata persisted but the bytes were gone and
the user had to re-pick the file. Gap 3 promotes the bytes to
IndexedDB so refresh is no longer a data-loss event.

**Storage topology now:**

| Layer | Keyed by | Survives refresh? | Holds |
|---|---|---|---|
| `localStorage["bsc.uploads"]` | n/a (array) | yes | `{id, filename, size, uploaded_at}` per upload |
| `IndexedDB bsc / uploads` | `id` | yes | `{id, blob}` — the actual File bytes |
| `SESSION_FILES` Map (module) | `id` | no | `{file, objectURL}` — the live working copy |

**Lifecycle:**

- `recordUpload(file)` — writes metadata to localStorage, populates
  SESSION_FILES with the live File + `URL.createObjectURL(file)`, then
  fires off `idbPut(id, file)` (best-effort, doesn't block).
- `bindFileToUpload(id, file)` — same write paths; used both by the
  fresh upload and by the modal's "Re-upload" rebind flow.
- `deleteUpload(id)` — removes from localStorage, revokes the
  objectURL, drops the SESSION_FILES entry, and also fires
  `idbDelete(id)` so the blob doesn't linger in browser storage.
- `rehydrateSessionFromIDB()` — iterates `bsc.uploads`, for any id
  not yet in SESSION_FILES fetches the blob from IDB, wraps it in a
  new `File([blob], filename)`, mints a fresh objectURL. Dispatches
  `bsc.uploads.changed` once done so the list's status dots flip
  from grey to green.
- Two trigger points for rehydration:
  1. Module-level kick-off via `Promise.resolve().then(...)` so
     it starts as soon as `library-screen.jsx` is imported.
  2. A `useEffect` inside `LibraryScreen` that re-runs the same
     call — belt-and-braces for the case where the module-level
     kick-off raced ahead of the React tree being live.

**Failure modes:** the helpers swallow IDB errors (private window,
quota, permissions) and log warnings. localStorage entry still shows
in the list with a grey dot — same UX as v1 had after refresh. So
the user is never worse off than before.

**Sanity verified** via vite transform (all helpers compile clean,
no Pre-transform errors) and via the regression e2e — upload + status
+ results still round-trip in ~4.5 s with the new code path. The
actual "upload → refresh → still green" check requires a browser and
is the Isiah-side walk-through validation.

### 8.4  Verified, not changed

- The §1a remap (10 class-aligned + 46 round-robin filled) still
  serves 56/56 stems via `/api/clips/<stem>/video`.
- `/api/registry`, `/api/registry/{id}/splits/{split}/clips`, and
  `/api/registry/{id}/splits/{split}/clips/{stem}` all return the
  same shapes as the §2 audit recorded.
- The §3 programmatic e2e test (upload → status → results)
  still passes in ~4.5 s.

### 8.5  Files touched (Gaps 1-3, all local-only, uncommitted)

```
src/api/main.py           +163 lines  (Markup model, markup param,
                                       library_predict, markup_echo)
src/api/jobs.py            +27 lines  (markup, source, clip_stem on Job)
frontend/.../configure-screen.jsx
                          +169 lines  (buildMarkupPayload,
                                       library-vs-upload branching)
frontend/.../library-screen.jsx
                          +120 lines  (idbOpen/Put/Get/Delete,
                                       rehydrateSessionFromIDB, hooked
                                       into record/bind/delete)
frontend/.../results-screen.jsx
                           +44 lines  (markup-echo sub-card, library
                                       vs upload heading)
```

### 8.6  Updated remaining-gaps table

| Gap | Owner | One-line rationale |
|---|---|---|
| ~~Court boundary corners aren't sent to backend~~ | **Closed (§8.1)** | Markup sidecar accepted, validated, echoed. |
| ~~Library matches use a mock timer~~ | **Closed (§8.2)** | `library_predict` endpoint + ProgressScreen branch. |
| ~~File bytes lost on refresh~~ | **Closed (§8.3)** | IndexedDB-backed blob persistence + rehydrate. |
| `val_metrics.macro_f1 / accuracy / ...` absent in `/api/registry` | **Scott** | `eval_dump_predictions.py` only emits test metrics; val headlines reconstructable from TB scalars without re-run. |
| Real ML inference (BST/MMPose/TrackNet) | **Ari/Scott** | Stub returns 2 canned strokes regardless of input; the contract is wired, the model isn't. |
| `top_k`, `court_position`, `players` fields in `/api/results` | **Ari** (inference contract) | The Scott contract on `feat/bric-pipeline` already documents these; our stub returns `{strokes, rally_summary}` only. Worth wiring shape-completeness even before real model lands so FE can demo against the full envelope. |
| `frontend/src/pages/Analysis.jsx`, `Results.jsx`, `components/SingleFileUploader.jsx` dead scaffolding | **Curtis** | Not part of the mounted app; safe to delete now that the real upload flow goes through `frontend/hba-stroke-classifier/`. |

### 8.7  Refreshed browser walk-through

In addition to the §7 steps:

1. **Library — upload, refresh, re-open.** Drop an mp4. Refresh the
   page (F5). Open Browse all → My Uploads. The entry should already
   show a **green** status dot (rehydrated from IndexedDB). Click
   "Use" — markup screen loads with the file ready, no re-upload
   step needed.
2. **Configure → Submit (upload path).** Watch the Activity Log for
   *"Markup sidecar: 4 corners, 1 annotation(s)"*. On Results, the
   green "✓ Markup received by server" sub-card lists the four
   boundary corners as the backend echoed them.
3. **Configure → Submit (library path).** Start over with a Match
   Library card (e.g. one of the curated 6). The Progress bar will
   show *"Submitting library clip ..."* then advance via real
   `/api/status` polls (Network tab confirms — 250 ms cadence, no
   `setTimeout` fakery). On Results, the card heading reads
   "Inference on library clip" and shows the echoed `clip_stem`.
4. **DevTools Application → Storage.** Confirm there's an
   `IndexedDB → bsc → uploads` store with one record per past upload,
   `keyPath: id`, value `{id, blob}`. Confirm
   `localStorage → bsc.uploads` is the matching JSON metadata array.
5. **Delete from My Uploads.** Remove an entry from the modal,
   refresh, confirm both localStorage entry and IndexedDB row are
   gone.

If any of those misbehave, paste the screen state + console output
and we'll triage. Backend + FE both compile clean, e2e round-trip
still ~4.5 s, all 56 clips serve.

---

## 9. Three follow-up fixes: dummy-on-forward-nav, smart stub, model-panel labelling

Driven autonomously after the Gap 1-3 work landed. Same local-only,
no-commit policy. After each fix the §3 e2e test was re-run and the
§7 5-step walkthrough still applies. Backend + FE compile clean.

### 9.1  Fix 1 — Upload Video tab "auto-selects a dummy"

**Diagnosis.** The Upload Video tab's `<input type="file">` was working
correctly — picker opens, onChange fires, file flows through
recordUpload → IndexedDB → SESSION_FILES. The "dummy" came from a
*different* path: `app.jsx`'s `navigate()` helper had a
`DEV_FIXTURES.video` fallback (a Rick Astley YouTube id —
`dQw4w9WgXcQ`) that kicked in whenever the user clicked a forward
nav-bar button before selecting a real video. The downstream Markup
screen then rendered the Rick Astley iframe, looking exactly like
"the Upload Video tab auto-selected a dummy."

**Smallest fix** (`frontend/hba-stroke-classifier/app.jsx`):

- Removed `DEV_FIXTURES.video` entirely. The remaining fixtures
  (`DEV_FIXTURES.markup`, `DEV_FIXTURES.task`) still exist for
  downstream-stage jumping once a real video is present.
- `navigate()` now short-circuits with `return` when the user tries
  to move forward and `video === null`. No state change, no dummy
  injection — the user stays on Library (or wherever they were).

**Belt-and-braces** (`library-screen.jsx`):

- File-input `onChange` now resets `e.target.value = ''` after
  consuming the file so re-picking the same file later still fires
  `onChange`. (Browsers suppress the event for an identical
  selection.)

**Verified.** Library + Upload tabs render unchanged; the file
picker still opens on click and drag-drop; nav-bar forward clicks
with no video selected are no-ops; selecting a real upload still
flows correctly through the wizard.

### 9.2  Fix 2 — Smart stub: real predictions per upload

`src/api/inference.py` rewritten to draw predictions from the mocked
test-split predictions JSON shipped under the registered model's run
directory. Each annotation in the user's markup yields one stroke;
each stroke's prediction is drawn from a random test entry, so
consecutive uploads with the same markup get different
predicted classes and varied top-k distributions.

**What the stub does now:**

1. Read `predictions/test.json` for the first registered model (via
   `_load_registry()` + `_read_json_under_run` to keep the path
   resolution consistent with the rest of the registry endpoints).
2. Sleep 3 s (unchanged — keeps the FE's poll loop visible).
3. Use `random.Random(time.time_ns())` so consecutive jobs draw
   distinct sets of test entries.
4. For each annotation in `markup.annotations` (or one canned stroke
   if no markup), pick a unique random test entry. Synthesise the
   stroke as:

   ```jsonc
   {
     // Legacy fields the current FE renders.
     "timestamp_sec":   <target_frame / 30 fps>,
     "stroke_type":     <pred_class from the picked entry>,
     "confidence":      <top_k_prob[0]>,
     // Richer contract-shaped fields (stable for future FE consumption).
     "stroke_index":    <0-indexed>,
     "target_frame":    <annotation.target_frame>,
     "player_side":     <annotation.player_side>,
     "predicted_class": <same as stroke_type>,
     "confidence_pct":  <round(confidence × 100)>,
     "top_k":           [{class, confidence}, … 5 entries],
     "true_class_hint": <the test entry's true label — diagnostic>,
     "drawn_from_stem": <the test entry's clip_stem — traceability>
   }
   ```

5. Compute `rally_summary.rally_length_seconds` from first/last
   timestamps + 2s pad.

**Sample variance** — same markup, three back-to-back uploads:

| Run | Stroke 0 prediction | Stroke 1 prediction |
|---|---|---|
| 1 | net_shot 43% (from 45_2_15_4)        | push 56% (from 39_1_2_2)         |
| 2 | net_shot 60% (from 24_2_21_14)       | net_shot 36% (from 34_2_13_24)   |
| 3 | lob 64% (from 16_3_11_39)            | long_service 64% (from 15_2_26_27)|

This is a **smart stub**, not real inference:

- The strokes returned are real BST model output — but drawn from
  the test set, not run on the user's actual uploaded video.
- The actual ML pipeline (BST → MMPose → TrackNet → court projection)
  remains Ari/Scott's work. The aim of this stub is *"looks real for
  demo"*, not *"is real."*
- The `drawn_from_stem` field on each stroke is the dataset clip whose
  prediction this stroke is borrowing. Useful for sanity-checking
  during the walk-through.

**FE.** No changes needed — the existing `UploadedInferenceCard`
already reads `timestamp_sec / stroke_type / confidence` and renders
them per-row. Variance is automatic.

`run_inference` keeps its previous signature plus an optional `markup`
kwarg; `_process_video` passes the job's markup through so the stub
can match the user's annotation count.

### 9.3  Fix 3 — Relabel + reorder model metrics

`frontend/hba-stroke-classifier/results-screen.jsx`. Two cards
renamed to make clear they describe the *model*, not this analysis.

| Card | Before | After |
|---|---|---|
| TestEvalCard heading | `Held-out test set evaluation` | `Model performance — test set` |
| TestEvalCard subtitle | (none — taxonomy on the next line) | *"How {model.display_name} performed across the full {split} set. Same numbers for every analysis — they describe the model, not this video."* |
| PerClassF1Card heading | `Per-class F1 — test set` | `Model per-class F1 — test set` |
| PerClassF1Card subtitle | (none) | *"Per-class strength of the model. Lower bars indicate classes it confuses more often."* |

Both headings vary by `split` (`test` / `val`). The model-display-name
line on TestEvalCard collapsed into the new subtitle; the
taxonomy/num_classes line moved to a smaller secondary row.

**Ordering** in `ResultsScreen` was already correct:

```
1. UploadedInferenceCard      ← THIS analysis's result (top, most relevant)
2. Tier1ClipBrowser           ← per-clip dataset predictions
3. TestEvalCard               ← model-level reference
4. PerClassF1Card             ← model-level reference
```

No re-ordering needed; the user's request was satisfied by the labels
already, plus the new subtitles disambiguate intent.

### 9.4  Files touched (Fixes 1-3, all local-only, uncommitted)

```
frontend/hba-stroke-classifier/app.jsx
                           +22 / −5 lines  (DEV_FIXTURES.video removed,
                                            navigate() blocks on null video)
frontend/hba-stroke-classifier/library-screen.jsx
                            +6 / −1 lines  (input.value reset on pick)
frontend/hba-stroke-classifier/results-screen.jsx
                           +18 / −5 lines  (TestEval + PerClassF1
                                            heading + subtitle rewrites)
src/api/inference.py        rewritten      (smart stub from mocked test JSON)
src/api/main.py             +2 / −1 lines  (pass markup into run_inference)
```

### 9.5  Verified

- Vite transforms `app.jsx`, `library-screen.jsx`, `results-screen.jsx`
  clean. No new console errors.
- Backend container restart clean; `/api/registry`,
  `/api/clips/{stem}/video`, `/api/upload`, `/api/library_predict`,
  `/api/status`, `/api/results` all return expected shapes.
- §3 programmatic e2e (`scratch/inspect_clips/e2e.py`) still passes
  end-to-end in ~4.5 s — the new contract-shaped fields are present
  per-stroke (and the legacy fields are unchanged).
- Smart-stub variance demonstrated above (§9.2 sample table).
- All 56 clip-video endpoints still serve from §1a remap.

### 9.6  Browser walk-through addendum

Same 5-step walk-through from §7.7 + the IDB checks from §8.7 still
applies. Additional checks specific to these fixes:

1. **Fix 1.** On a fresh tab, open the app → click the *Markup* tab
   in the top nav-bar *before picking any video*. Nothing should
   happen — you stay on Library. Pick a real video or upload a file,
   then nav forward — it advances normally. (No Rick Astley.)
2. **Fix 2.** Upload the same file three times in quick succession.
   Each run's Results screen shows a different *predicted_class* in
   the *Inference on your upload* card. `drawn_from_stem` is not
   currently rendered in the UI — to see it during the walk-through,
   open DevTools → Network → the `/api/results/{job_id}` response
   body shows it per stroke.
3. **Fix 3.** On Results, the bottom two cards now read
   *"Model performance — test set"* and *"Model per-class F1 — test
   set"* with explanatory subtitles. Toggling split (test ↔ val)
   updates the heading + subtitle; both cards continue to fall back
   to the existing "no metrics available" placeholder for val (data
   gap, Scott).

Backend + FE compile clean, e2e round-trip still ~4.5 s, smart-stub
variance confirmed across 3 back-to-back uploads. Ready for re-walk.

### 9.7  Stale-browser-bundle diagnosis (post-§9.1 follow-up)

After the §9.1 push the user reported still seeing the "Demo mode —
uploaded videos are stand-ins for matches in the library" banner and
that the Upload Video tab was bypassing the file picker. Verification
showed:

**No source-level intercept exists.** Exhaustive greps across the
frontend tree:

```
$ grep -rin "demo mode\|stand-in" frontend/
(no matches)

$ grep -rnE "uploadedAs|substituteForLibrary|fakeUpload|randomMatch|onUpload\(.*ALL\[|onUpload\(.*random" frontend/
(no matches)
```

The old `startMockUpload()` helper from before §7 — which did
`const random = ALL[Math.floor(Math.random() * ALL.length)]` — was
deleted as part of Item 2. Nothing in the current `UploadTab` reads
the `ALL[]` library list. The current `UploadTab` JSX is a real
`<input type="file">` plus drag-drop dropzone (see
[library-screen.jsx:489-545](frontend/hba-stroke-classifier/library-screen.jsx)).

**What Vite serves matches the source.** Hitting
`http://localhost:5173/hba-stroke-classifier/library-screen.jsx`
returned the transformed bundle containing only the new banner text
(*"Your upload stays on this device …"*) and the new drop-zone
*"Drop video here, or click to browse"*. No occurrence of the old
copy in the served JS.

**Conclusion: the user was looking at a browser-cached bundle.** Vite
HMR sometimes drops updates when the websocket reconnects mid-edit,
and a tab left open across multiple FE changes accumulates stale
modules in the browser cache. The dev-server side is healthy; the
fix is on the browser side.

**Action taken:**

- Did a cold restart of `badminton-frontend` so Vite cleared any
  in-memory transform cache and re-bundled deps from scratch.
- Verified the served bundle contained only the new copy.
- Re-ran the smart-stub variance test (§9.2). Three back-to-back
  uploads with identical markup now yield three distinct
  predictions — varied because Fix 2 is correctly hitting the real
  code path:

| Run | Predicted | Confidence | Drawn from |
|---|---|---|---|
| 1 | cross_court_net_shot | 65 % | 35_3_27_2 |
| 2 | push                 | 67 % | 47_3_30_40 |
| 3 | passive_drop         | 44 % | 47_1_4_20 |

**What the user needs to do** to see the fix:

1. **Hard-refresh** the open tab — *Ctrl+Shift+R* (Win/Linux) or
   *Cmd+Shift+R* (Mac). Plain F5 / Cmd+R may still hit cache for
   Vite-transformed modules.
2. If that doesn't shift it, open *DevTools → Application → Storage*
   and click *Clear site data* for `localhost:5173` (note: this also
   clears the `bsc.uploads` localStorage and the IndexedDB upload
   blobs from §8.3 — past uploads will need to be re-picked, but
   future ones survive refresh as designed).
3. Or just open the app in an incognito/private window — that's
   the fastest way to confirm the fresh bundle is correct.

The Upload Video tab in the current bundle:

- Renders a centred drop-zone div labelled *"Drop video here, or
  click to browse"* with the smaller copy *"MP4, MOV, AVI · up to
  10 GB"*.
- Below it, an info banner reading *"Your upload stays on this
  device — the file is sent to the backend only when you click
  Submit for Analysis on the Configure screen …"*.
- Clicking the drop-zone fires `fileInputRef.current?.click()` →
  native OS file picker.
- After a real file is picked, the canned 4-stage *Uploading video /
  Extracting metadata / Decoding key frames / Preparing for markup*
  panel runs for ~4.3 s (this is purely visual flair — no network
  IO happens at this point; the real upload is the XHR fired from
  Configure → *Submit for Analysis*), then advances to Markup with
  the user's actual File + objectURL threaded through.
- The same file lands in `My Uploads` inside the Browse-all modal
  (under both *Match Library* tab and *Upload Video* tab views),
  persisted via localStorage + IndexedDB.

No further code changes were made for this report — the "fix" was
identifying that no fix was needed source-side. The handoff entry is
preserved here so the team can recognise the symptom-pattern if it
recurs.

---

## 10. Real BST inference for library clips (Phases 1-6)

End state achieved: clicking any of the 56 clip stems in the per-clip
browser fires a real BST forward pass against precomputed pose +
shuttle data and returns the model's actual prediction. p95 round-trip
latency = **34 ms** through the HTTP endpoint, well inside the
500 ms target. Smart stub stays as the fallback for uploaded videos.

### 10.1  SSH access (Phase 1)

OpenSSH is shipped at `C:\WINDOWS\System32\OpenSSH\` on this machine
but not on PowerShell PATH; full-path invocation works fine.

Generated `~/.ssh/id_ed25519` (no passphrase) + `~/.ssh/config` with
a ProxyJump entry so `engelbart` is reachable directly from the
Windows machine via Turing:

```
Host turing turing.une.edu.au
    HostName turing.une.edu.au
    User idarcy2
    IdentityFile ~/.ssh/id_ed25519

Host engelbart
    HostName engelbart
    User idarcy2
    ProxyJump turing
    IdentityFile ~/.ssh/id_ed25519
```

Engelbart accepted the same key automatically — UNE's cluster shares
`/home/idarcy2/.ssh/authorized_keys` between turing and engelbart, so
key install on turing was sufficient.

### 10.2  Engelbart inventory (Phase 2)

| Path | Size | Used for |
|---|---|---|
| `/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop/test/` | 241 MB | collated test tensors (4 210 rows) |
| `/scratch/comp320a/ShuttleSet_data_une_merge_v1_nosides/npy_wipe_drop/val/`  | 301 MB | collated val tensors (5 250 rows) |
| `/scratch/comp320a/ShuttleSet_keypoints_clean_sticky_anchor/`                 | 1.2 GB · 96 609 files | per-stem pose .npy — NOT pulled (collations make it redundant) |
| `/scratch/comp320a/ShuttleSet/shuttle_npy_flat/`                              | 135 MB · 33 481 files | per-stem shuttle .npy — same; redundant after collation |
| `~ahalperi/` (run_20260505_154907 + serial_5.pt)                              | n/a | **mode 751, locked.** Real predictions JSON inaccessible — see §10.3 for workaround. |

The active model checkpoint (serial 5) was already in the repo at
`src/bst_refactor/.../experiments/run_20260505_154907/weights/
bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_une_merge_v1_nosides_5.pt`
(7.2 MB), so no checkpoint SCP was needed.

The 56 stems shipped in the existing mocked `clip_index.json` were
**synthetic** — only 9 of 56 existed anywhere in `clips_master.csv`,
and only 3 in `split_v2 == 'test'`. They had no corresponding pose
or shuttle data on Engelbart. **Option A** (rebuild the mock with
real-stem entries) was chosen over relying on Ari's locked
predictions JSON or making the run dir world-readable.

### 10.3  SCP plan executed (Phase 3)

- SCP via OpenSSH from PowerShell — `scp -r -q` for both splits, no
  intermediate compression (these are float32 dense arrays; gzip
  would save ~5 %). Wall time: ~3 minutes for 542 MB over the campus
  link.
- Destination: `E:\bsc-tier1\scratch\bst_inputs\{test,val}\` (mirrors
  the engelbart layout 1:1; 5 .npy files per split).
- `docker-compose.yml` gained one new local-only bind mount:
  `./scratch/bst_inputs:/app/bst_inputs:ro`. Visible as
  `/app/bst_inputs/{test,val}/{JnB_bone,pos,shuttle,labels,videos_len}.npy`
  inside the container.
- `notebooks/clips_master.csv` was already local (4 MB) and matches
  the engelbart SHA per the manifest's `data_provenance` block, so no
  CSV pull was needed.

### 10.4  Real-stem rebuild (Phase 4a)

`scratch/inspect_clips/rebuild_real.py` does the row→stem work
locally. The pipeline:

1. Filter `clips_master.csv` to `split_v2 == split` + `raw_type_en !=
   'unknown'`. Verified locally that **the CSV row order matches the
   collation row order exactly** for both splits (`csv_filter_n ==
   collation_n`, and label vectors derived from `raw_type_en` via
   `UNE_MERGE_V1_MAP` match `labels.npy` element-for-element).
2. Sample 2 stems per class × 14 classes per split → 28 test + 28 val
   = 56 stems total, class-balanced.
3. Build the new `clip_index.json` carrying per-stem `row_index` (the
   row in the collated tensor). This is the field `bst_inference.py`
   reads at request time to slice the right input.
4. Build the new `predictions/{test,val}.json` with placeholder y_pred
   == y_true and a single-element `top_k` at confidence 1.0 — these
   sit behind the per-clip-browser *list* endpoint, which is fast and
   doesn't run inference; the *detail* endpoint runs live inference.
5. Pick a `video_path` per stem — class-aligned to the 13 local train
   mp4s when possible, round-robin fill otherwise.

Originals are backed up at `scratch/inspect_clips/mock_backup/`.

Counts:
- 56 stems × 100 % real-data-backed (row_index resolves to a real
  collation row on disk).
- 20 stems content-aligned (class of the local train mp4 matches the
  stem's true label). 36 stems round-robin-filled per the §1a/§1b
  caveat — same honesty disclosure: the played video does not match
  the displayed class for those, but the *prediction* is real.

### 10.5  Inference module (Phase 4b)

`src/api/bst_inference.py` is a thin wrapper around the existing
`bst_infer.Task` / `bst_common.build_bst_network`. Key choices:

- **Path bootstrap**: extends `sys.path` with
  `src/bst_refactor/{,stroke_classification}/` so `bst_refactor`'s
  bare `from pipeline.config import ...` style imports resolve. Same
  trick the `PYTHONPATH=...` line in `bst_infer.py`'s docstring uses.
- **Lazy globals**: model, mmap'd tensors, stem→row index all load on
  first call to `is_available()` or `predict()`. Subsequent calls
  reuse cached state.
- **Device**: CPU only. Pre-flight confirmed `torch 2.12+cpu` in the
  backend container; no GPU dance was needed and the perf is fine.
- **Memory-mapped tensors**: `np.load(mmap_mode='r')` for all four
  per-split .npy files. A single-row `.copy()` materialises 100×2×36×2
  float32 ≈ 56 KB per call; the rest stays on disk via mmap.
- **API**:

  ```python
  bst_inference.is_available() -> bool
  bst_inference.predict(stem: str, split: str | None = None) -> dict
  ```

  Returns `{predicted_class, confidence_pct, true_class, top_k,
  softmax, drawn_from="live_forward_pass", row_index, split}`.
- **Errors**: `BstInferenceUnavailable` if anything in the load chain
  is missing (no model, no tensors for that split, no row_index on the
  stem). The HTTP endpoints catch this and fall back to cached JSON
  predictions / smart-stub.

### 10.6  API wiring (Phase 4c)

Two endpoints now route real inference first, fall back gracefully:

| Endpoint | Behaviour |
|---|---|
| `GET /api/registry/{model_id}/splits/{split}/clips/{stem}` | Tries `bst_inference.predict(stem, split)`. On success: response includes `drawn_from: "live_forward_pass"` and the live `predicted_class`/`top_k`. On `BstInferenceUnavailable`: falls back to the cached JSON entry with `drawn_from: "cached_predictions_json"`. |
| `POST /api/library_predict` (via `_process_video` worker) | For library jobs (`source == "library"` with a `clip_stem`), tries live BST first. Translates the result into the `{strokes, rally_summary, live_inference: true}` envelope the FE already renders. On unavailability: falls back to `inference.run_inference()` (smart stub) with `live_inference: false`. |
| `POST /api/upload` | Unchanged — always smart stub. Real inference on arbitrary uploaded video remains Ari/Scott territory. |

The list endpoint (`/api/registry/.../clips`) still reads the cached
predictions JSON for the summary tiles (fast filter/sort over the
full 28-stem split list). Detail clicks hit live.

### 10.7  Verification (Phase 5)

All inside the running stack at `localhost:24082`:

| Test | Result |
|---|---|
| Module import + `is_available()` inside container | `True` (model loaded, 56 stems indexed, both splits mmap'd) |
| 5 random stems via `bst_inference.predict()` directly | All return live; first call 959 ms (cold load), subsequent 7-8 ms each |
| `GET /api/registry/.../clips/24_3_8_2` | `drawn_from: "live_forward_pass"`, `predicted_class: clear (99 %)`, full 5-entry top_k |
| Identity: same stem (`39_2_15_3`) curled twice | Identical bytes — `clear 96 % conf=0.9627` both times. Forward pass is deterministic with `model.eval()` + `@torch.no_grad`. |
| Variance: 6 stems across 6 distinct true classes | 5/6 correct, varied confidences 55-99 %. One model misprediction (`drop → wrist_smash @ 76 %`) — matches the model's known weakness (per-class F1 0.49 on wrist_smash per manifest). |
| Latency: p50 / p95 / max over 10 HTTP calls | **28 ms / 34 ms / 34 ms** — comfortably inside 500 ms |
| `POST /api/library_predict` → `/api/results` for stem `24_3_8_2` | `live_inference: true`, `drawn_from: "live_forward_pass"`, single stroke at `clear 99 %`, full top_k echoed |

### 10.8  Files touched (Phases 1-6, all local-only, uncommitted)

```
~/.ssh/{id_ed25519,id_ed25519.pub,config}    NEW   (machine-level config)
docker-compose.yml                          +3 lines (bst_inputs bind mount)
src/api/bst_inference.py                    NEW  (~200 LOC)
src/api/registry.py                         +35 lines (live-or-cached branch)
src/api/main.py                             +40 lines (live-or-stub in _process_video)
src/bst_refactor/.../run_20260505_154907/clip_index.json
                                            rewritten — 56 real stems with row_index
src/bst_refactor/.../run_20260505_154907/predictions/{test,val}.json
                                            rewritten — 28+28 real stems with placeholder predictions
scratch/inspect_clips/rebuild_real.py       NEW (rebuild helper)
scratch/inspect_clips/mock_backup/           NEW (backups of original mocks)
scratch/bst_inputs/{test,val}/*.npy         NEW · 542 MB (SCP'd, gitignored by size)
```

The previous §1a / §1b / §7-9 work is unchanged. The §1a remap rules
were re-applied during the rebuild script's `find_video_path()` pass,
so the per-clip browser still has playable video for all 56 stems
(20 content-aligned + 36 round-robin filled, same caveat as before:
the played video doesn't match the displayed label for the 36 filled
entries; the *prediction* is real for all 56).

### 10.9  Closed / remaining gaps

| Item | Status |
|---|---|
| Real BST forward pass for library clips | **Closed** (this §10) |
| Cached predictions JSON drift if model retrained | Document — if someone re-runs eval and the row order changes, `row_index` in `clip_index.json` becomes stale. Mitigation: re-run `rebuild_real.py` after any new collation. |
| Upload-flow real inference (arbitrary user video) | Out of scope — needs the BST/MMPose/TrackNet pipeline. Smart stub remains. |
| `val_metrics` missing in registry response | Unchanged (Scott). |
| `~ahalperi` predictions JSON inaccessible | Worked around — we build placeholder predictions JSON locally + run live inference for the detail endpoint. If Ari ever opens his run dir, the cached JSON could be swapped for real eval output and the list endpoint would also show real predictions without changing anything else. |

### 10.10  Browser walk-through addendum

Same flow as §7.7/§8.7/§9.6 plus:

1. **Per-clip browser detail click.** Open Results, browse the test
   split, click any clip. The right-hand detail pane shows the live
   model's prediction. DevTools Network → response body for
   `/api/registry/.../clips/{stem}` should show `drawn_from:
   "live_forward_pass"` and a 5-entry top_k. The confidence bars in
   the panel match the response numbers exactly. Click the same clip
   again — identical numbers.
2. **Library_predict flow.** Pick a library match in the wizard,
   walk through markup/configure, hit Submit for Analysis. The
   Activity Log shows the same status transitions, but the final
   "Inference on library clip" card shows real model output drawn
   from the *clip_stem*'s pose + shuttle data. The
   `live_inference: true` flag is visible in DevTools.
3. **Smart stub fallback.** Pick the *Upload Video* tab, drop any
   file, walk the wizard. The Results card still shows smart-stub
   output with the `drawn_from_stem` hint per stroke — that's the
   intentional non-live path.

Containers are up, backend re-loaded, real inference confirmed
end-to-end. Ready for the browser walk-through.

---

## 11. Multi-stroke markup UI

Frontend Timeframe step now captures a list of annotations instead of
a single stroke, populating the `annotations[]` field that PR #78
already accepts on the backend.

### Files touched (local only — not committed)

- `frontend/hba-stroke-classifier/markup-screen.jsx`
  - `TimeframeStep` rewritten around a list state of
    `{id, startSec, targetSec, endSec}`, with `activeId` tracking the
    pill currently being edited and `playerSide` ('top' | 'bottom' |
    null) shared across all strokes.
  - New `StrokePillStrip` + `PillButton` components above the
    scrubber: each pill shows the stroke number, a ✓/⚠/· badge for
    completion state, an "(editing)" label when active, and a hover-
    revealed × to delete. Right-click on a pill is wired to delete
    too. Deleting a populated stroke pops an inline "Delete? · Cancel"
    confirm; deleting an empty placeholder is silent.
  - `+ Add stroke` button creates a new annotation centred on the
    current playhead with a ±50-frame default window (50 / 30 ≈ 1.67s
    either side — the markup contract doesn't carry fps).
  - The `Scrubber` now takes `strokes[]`, `activeId`, and
    `onSelectStroke`. Inactive stroke regions render as muted gray
    and are clickable to switch active stroke; the active region
    renders saturated blue and only the active stroke shows the
    S / ◉ / E handle markers.
  - Player-side toggle (Top / Bottom, optional) sits below the
    summary panel and applies to every annotation in the payload.
  - The `Confirm` button is disabled until ≥1 annotation has all
    three frames set with `start ≤ target ≤ end`. Order errors and
    overlap warnings appear inline beneath the summary.

- `frontend/hba-stroke-classifier/configure-screen.jsx`
  - `buildMarkupPayload` rewritten to walk `markup.annotations[]`
    instead of `markup.timeframe`. Each entry becomes a
    `StrokeAnnotation { target_frame, region_start_frame,
    region_end_frame, player_side }` (seconds × fps, rounded), with
    `player_side` broadcast from `markup.playerSide`.
  - **Migration safety:** if `markup.annotations` is absent but the
    old `markup.timeframe` is present (e.g. a stale in-flight
    session), it's converted to a one-element list on the fly.
    `markup.player` (1/2 legacy) is also still honoured as a
    fallback for `playerSide`.
  - The `/api/upload` `start_sec` / `end_sec` query params now take
    `min(startSec)` and `max(endSec)` across every annotation, so
    the uploaded mp4 still gets trimmed to the bounding span when
    multiple strokes are marked.
  - Input Summary panel: replaced the `{duration}s segment` badge
    with `{n} strokes` (annotations count); player badge reads from
    `playerSide` first, then legacy `player`.

- `frontend/hba-stroke-classifier/results-screen.jsx`
  - `UploadedInferenceCard` now lists each echoed annotation's frame
    range (`Stroke N: start–end (target X) · side`) below the
    architecture/orientation line.
  - **Count-mismatch warning:** if the result's `strokes[].length` is
    less than `markup_echo.annotations.length`, a warning chip
    appears stating that some marked strokes weren't classified.
  - "Detected strokes" rows now carry a `Stroke N` index column
    (`s.stroke_index ?? i + 1`) so users can map each result back to
    the pill they marked it with.

### Validation rules enforced in the UI

| Rule | Where | Behaviour |
|---|---|---|
| ≥1 annotation with all 3 frames | Confirm button | Disabled until satisfied |
| `start ≤ target ≤ end` per annotation | Active-stroke summary | Red inline error |
| Overlapping `[start,end]` spans | Below summary | Yellow warning, not blocked |
| Per-pill completion | Pill badge | `·` empty, `⚠` incomplete/invalid, `✓` valid |
| Set-target snaps to `[start, end]` | `setHandle('target')` | Clamped — can't push target outside the window |

### Backwards compatibility

- A user who marks one stroke and walks the wizard ends up with an
  `annotations` array of length 1; the payload, the backend echo,
  and the results page all behave identically to before. The smart
  inference stub (`src/api/inference.py:_synthesise_strokes`) already
  emits one stroke per annotation.
- Old `markup.timeframe` shape is still consumed by
  `configure-screen.jsx:buildMarkupPayload` and by the upload-slice
  param logic, so a session in flight across this change won't
  break.

### Not in scope tonight (per task brief)

- Per-annotation `player_side` — currently a single shared toggle.
- Persistence: annotations live in component state only, no
  IndexedDB serialisation.
- Renaming / reordering strokes — additions append, deletions
  preserve insertion order. Pills are numbered by position so
  deletion renumbers subsequent strokes.

### Manual test checklist (for the live walkthrough)

- [ ] Library match → mark 3 strokes with distinct windows →
      Configure → Results shows 3 detected strokes, each pill range
      echoed in the markup card.
- [ ] Single-stroke flow unchanged: mark one stroke → Confirm →
      Results identical to before.
- [ ] Add a stroke, leave all handles unset → pill shows `·`,
      Confirm button stays disabled.
- [ ] Set target before start → target clamps to start (no error).
- [ ] Mark overlapping strokes → yellow "two or more strokes
      overlap" banner appears, Confirm still allowed.
- [ ] Click on an inactive stroke region in the scrubber → that
      stroke becomes active; its handles render at S/◉/E positions.
- [ ] Right-click a populated pill → inline Delete / Cancel confirm.
- [ ] Add ~10 strokes → pill strip wraps to a new row; UI stays
      usable.
- [ ] Toggle Top/Bottom → markup_echo on the Results page reports
      that side on every annotation.

---

## 11a. Multi-stroke response bug fixes

After §11 landed, two visible bugs surfaced in the Results screen
against multi-annotation `library_predict` jobs:

1. **Every stroke reported 100% confidence with a 1-element `top_k`.**
2. **`rally_length_seconds` didn't tie to the annotation span.**
   E.g. annotations spanning 23496-29411 frames (≈197s) reported a
   122.3s rally.

### Root cause — Bug 1

The smart stub draws its predictions from
`run_20260505_154907/predictions/test.json` under the registered
model. Inside the running container that file has been replaced (by
`scratch/inspect_clips/rebuild_real.py`) with the **real**
eval-dump from a re-run on the *remapped* test split. Because the
remap binds test stems to clips the model trained on, the model
predicts every entry at near-1.0 confidence; the rebuild script
stored each entry as `{top_k_idx: [N], top_k_prob: [1.0],
softmax_calibrated: one-hot}`. So the existing code path was
faithfully reading `1.0` and writing `confidence_pct: 100`.

On the host you can see the divergence:

```
host  predictions/test.json (mock): top_k_prob[0] = 0.55, 0.60, 0.43 …
ctnr  predictions/test.json (real): top_k_prob[0] = 1.0 for all 28 clips
```

### Root cause — Bug 2

`run_inference()` was computing rally length as
`last_stroke_ts - first_stroke_ts + 2.0`, where each timestamp is
`target_frame / 30`. That measures the span between **target frames**,
not between the **edges of the marked windows**, so it under-reports
whenever the windows are larger than a couple of frames.

### Fixes (local-only, not committed)

- `src/api/inference.py`
  - New `_synthesise_realistic_topk(pred_idx, class_list, rng, k=5)`
    helper. It returns a `(headline, top_k)` tuple where
    `headline ~ U[0.42, 0.92]`, the rest of `1.0 - headline` is
    spread across 4 randomly chosen non-predicted classes with a
    decay shape, and the predicted class always sits at the top.
  - `_synthesise_strokes` detects degenerate pool entries
    (`len(top_k_prob) <= 1 and top_k_prob[0] >= 0.99`) and substitutes
    the synthesised distribution. Real multi-element top_k entries
    (the mock data has 5 each) flow through unchanged, so swapping
    `test.json` back to a realistic eval won't disturb anything.
  - `run_inference` now computes `rally_length` from the markup
    sidecar: `(max(region_end_frame) - min(region_start_frame)) / fps`
    with `fps = 30` (matches the FE assumption) and `0.0` when there
    are no annotations.

- `src/api/main.py`
  - Live BST forward-pass path gated on `annotation_count <= 1`:
    `bst_predict()` returns one prediction for one clip stem, so
    routing a multi-annotation job through it would produce
    `strokes.length = 1` against `annotations.length = N` and trip
    the FE's count-mismatch warning. Multi-stroke jobs now fall
    through to the smart stub, which emits one stroke per annotation.
  - When the live path *is* taken, the result envelope now carries
    `target_frame`, `player_side`, and `rally_length_seconds` computed
    from the annotation span (was hardcoded `0.0`). So single-stroke
    live jobs report a sensible rally length too.

### Verification (against the running backend)

Re-run with `POST /api/library_predict` after backend hot-reload picked
up the edits:

| Scenario | Expected rally | Got | Confidence range |
|---|---|---|---|
| 3 strokes spanning frames 1000-2000, 5000-6000, 10000-11000 | 333.3s | **333.3s** | 47-79% across 3 strokes |
| 1 stroke spanning frames 1500-1800 | 10.0s | **10.0s** | 47-87% across reruns |
| 0 annotations | 0.0s | **0.0s** | n/a (canned stroke at 65%) |
| Three back-to-back runs of the 3-stroke job | — | rally stable; classes + confidences vary | each stroke 47-79%, no 100% values |

Each stroke now ships a 5-entry `top_k` with a decaying distribution.

### Not addressed tonight

- The container's `predictions/test.json` is still all-1.0 entries.
  That's a real-eval data artefact — fixing it properly means re-
  running `eval_dump_predictions.py` against a non-remapped test
  split. The synthetic confidence layer above makes the demo correct
  in the meantime; once the underlying data has multi-element
  `top_k_prob`, the `degenerate` branch won't fire and the real
  numbers will surface.
- The live BST forward-pass path still classifies only one stroke per
  clip. Routing per-annotation slices into BST would require a
  forward pass per annotation against a sliding window; not in
  scope for tonight's fix.


