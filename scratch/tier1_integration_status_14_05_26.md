# Tier 1 wiring: where we're at

Short companion to `frontend_integration_handoff.md` (the contract). Where the wiring sits today, what's still mocked, what I owe to swap mocks for real, and how to run the whole thing locally. Tag me on slack if you need backend data from me.

## What's running today

I did up the bones of the Tier 1 integration on the `integration-tier1-bones` branch. Everything I've tested seemed to work fine, falling back to dummies where real data wasn't available. But it should be set up to accept the real data if it is available (for what I've dubbed *tier-1* components, anyway). The model picker pulls from `/api/registry`. The per-clip browser shows the top-5 bar chart with a val/test toggle and an errors-only filter. The held-out test eval card and per-class F1 chart autopopulate from the registry. The embedded `<video>` element streams the clip mp4 when one exists, falls back to a clean "no mp4 on this host" message when it doesn't. Everything goes through the contract; no hardcoding where avoidable.


I haven't yet done runs where I save per-clip confidence, so that's mocked. But when I do them, it should just slide right in. The way the mocks work at the moment: the registry has one entry, the current best (`run_20260505_154907` serial 5). The five sidecar JSONs are 28 plausible-looking clips per split with synthesised confidences. They all carry a top-level `_mock_data: true` so it's obvious.

## What I still need to deliver

Three model-side scripts (long doc sections 6.1, 6.2, 6.3), all on engelbart. None of them touch the FE or the API:

- Extend `eval_dump_predictions.py` to emit the per-clip JSON shape we mocked.
- New `build_clip_index.py`, wrapping the existing `pipeline.clip_index` and `pipeline.data_access` helpers (no rewrite, just JSON output).
- New `build_perclass_stats.py`.

Run them once, drop the outputs over the mocks under `experiments/run_20260505_154907/`, refresh the browser. FE and backend pick them up unchanged.

Only `test` gets dumped today, which is why the val side of the toggle currently shows mocks against a code path that already knows how to handle val. Train and val headline metrics could probably be reconstructed from the per-epoch TensorBoard scalars (final `val_macro_f1` and friends) without re-running eval. That's a follow-up, but it's non-blocking.

## About the temperature field

Worth clearing this one up. The registry entry currently has `temperature: 1.0`, which means no calibration is applied; the displayed `confidence_pct` is just raw softmax (standard machine learning algo to get confidence). For the mocked data that's irrelevant (the synthesised confidences were never real probabilities anyway). For the real data, we'd rather not output raw softmax. Softmax exaggerates to make model training more efficient. So the displayed % won't quite match the "right N% of the time" promise the handoff doc makes to the FE team.

The fix is a calibration-fit script that picks the `temperature` value that best matches the normalised surprise of the model at its own prediction.

**tldr** If this is overcomplicated, we can dump the 'temperature' slider entirely, and know that in the next couple of weeks the front-end will be able to call a script to auto-output a nice non-exaggerated confidence read. I'll pop out the script with the rest of the deliverable.

## What just works on a host with the clips

If you're on UNE HPC with `BST_X_CLIPS_DIR=/scratch/comp320a/ShuttleSet/clips`, the per-clip detail's `<video>` plays the real mp4 with scrubbing (`FileResponse` handles Range requests). Off HPC, the player shows the fallback message. That's correct behaviour for mocked stems that don't exist on disk, not a bug.

## What won't work

Tier 2 (live PyTorch) and Tier 3 (novel video) are out of scope on this branch. The existing upload, status, and results stubs in `src/api/main.py` are untouched. XAI, saliency, and class-activation maps aren't wired.

## Running it natively (no Docker)

Default config is Docker-flavoured (`UPLOAD_DIR=/app/uploads`), so a native run dies at startup with `PermissionError: '/app'` unless you override. Two terminals:

```
cd ~/Documents/COSC594/badminton_stroke_classification
UPLOAD_DIR=/tmp/bst-uploads ~/.venvs/bst-api/bin/uvicorn src.api.main:app \
    --host 0.0.0.0 --port 24082 --reload
```

```
cd frontend && npm install && npm run dev
```

Vite proxies `/api/*` to `localhost:24082`, so no CORS dance is needed. If you've stashed a handful of clips somewhere on your laptop and want them to play in the browser, prefix the uvicorn line with `BST_X_CLIPS_DIR=/your/path`. The layout under that path still needs to be `<split>/<Side>_<class>/<stem>.mp4`.

## Worth landing before flipping real predictions on

`Tier1ClipBrowser` in `results-screen.jsx` defaults to `?limit=50` with no prev/next or load-more controls. With 28 mocks it's invisible; at ~4200 real test clips it silently truncates. The backend already accepts `?limit=` and `?offset=`, so it's a small FE addition.

## On npm and the package-lock

I ran `npm install` on `node v25.5.0` / `npm 11.8.0`, which regenerated `package-lock.json`. I didn't commit it; not my side of the codebase. The drift is patch-level transitive deps, nothing scary. But if your npm version doesn't match, the next `npm install` will re-shuffle the lockfile again. Pinning a `.nvmrc` or noting expected node/npm versions in `frontend/README.md` would make the install reliable from here on (or do a quick run through the code and pin it to your version--whoever wants to maintain).
