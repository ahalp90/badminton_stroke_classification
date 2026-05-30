# Live keypoint overlay for BST_X clip viewer

Drawing a player's skeleton on top of a playing video clip, live, synced to playback. For the BST_X models. The drawing happens in the browser on a canvas layer. The server's only job is handing over precomputed joint data as JSON. No PIL, no server-side rendering, no per-request numpy.

## The idea

BST already extracts MMPose joints for every stroke clip into per-stem NPY files at `BST_MMPOSE_NPY_DIR/{clip_stem}_joints.npy`. We turn those NPYs into one gzipped JSON per stem, drop them into a flat cache directory, and serve them through a new endpoint. The FE clip viewer fetches the JSON for the clip it's playing, draws the skeleton on a canvas via `requestAnimationFrame`, and the overlay tracks playback frame by frame.

End to end, the user picks a clip, the video plays, the striker's skeleton draws live on top, and the classification label appears at the target frame.

This deliverable is independent of any rally-playback work. It slots into the existing Tier 1 clip-stem-keyed viewer the FE team is already building. If a rally-playback view shows up later, it consumes the same per-stem files.

## Where everything sits

- This planning doc: `src/xai/bst_x_fe_keypoint_overlay.md`
- Build script: `scripts/build_joints_json_cache.py`
- Cache output: `runtime/cache/joints_json/{clip_stem}.json.gz`, flat, around 270 MB across 32k clips, gitignored contents
- API module: `src/api/joints.py`, routes registered from `src/api/main.py`
- Tests: `tests/test_joints_endpoint.py`, `tests/test_build_joints_json_cache.py`
- FE component: separate FE repo at `~/Documents/COSC594/badminton_stroke_classification-frontend/`

The .gitignore line for the cache directory needs to land in the first feature commit, otherwise 270 MB of untracked files clutter `git status` after the first build run.

## What each per-clip file looks like

Source NPY at `BST_MMPOSE_NPY_DIR/{clip_stem}_joints.npy` is float32 with shape `(n_frames, 2 players, 17 joints, 2 coords)`. Sibling `_failed.npy` is the per-frame any-player failure flag. Verify both shapes against the actual files before locking the JSON layout.

Suggested JSON layout, agent can push back if there's a better one:

```json
{
  "clip_stem": "35_1_10_17",
  "vid": 35, "set": 1, "rally": 10, "ball_round": 17, "fps": 25,
  "start_f": 1234, "end_f": 1284,
  "n_frames": 50,
  "joints": [[[...], ...], ...],
  "failed": [false, false, ...]
}
```

Round floats to 4 decimal places before serialising. Pixel coords on a 1920x1080 frame don't need float32 precision, and rounding cuts JSON size by around 30% before gzip kicks in.

## Sizes, gzip, and serving

Per clip, ballpark:

- Minified JSON, ungzipped: 25-35 KB
- Same JSON pre-gzipped: 8-12 KB
- Raw NPY for reference: around 13 KB

Across 32k clips:

- Ungzipped JSON on disk: 800 MB to 1.1 GB
- Pre-gzipped on disk: 250-320 MB

Pre-gzip on disk, serve as-is. FastAPI `FileResponse` with `media_type='application/json'` and `headers={'Content-Encoding': 'gzip'}` does the right thing, browsers transparently decompress, server CPU at request time is zero. Don't gzip-on-the-fly per request, it wastes CPU on a wimpy host.

## Building and testing the cache

I'd rather not run the full 32k precompute and find out the script is broken halfway through. So the order is:

1. Build script first, with tests that run it against a fixture of 5-10 real stems (smallest in the dataset). Confirm the output JSON shape, the gzip integrity, and that round-tripping the floats back doesn't lose anything that matters for rendering.
2. Run those tests locally to confirm the script works at all.
3. Smoke run on 50 stems on whichever host has `BST_MMPOSE_NPY_DIR`. Eyeball one of the outputs. Open the gzipped file, decompress, check the joints render in the static overlay reference script.
4. Only then run the full build. It takes a few minutes, produces about 270 MB.

The full output is fine to ship locally for testing. I'd rather catch any bugs at stage 1-3 than during stage 4. The build script needs to be idempotent (skip stems whose JSON already exists, unless `--force`) so partial runs are recoverable.

## Mocks for the FE team

The FE team can't wait around for me to run the full precompute on the HPC. They need fixture data to develop against.

Plan: the build script also outputs a tiny sample set, say 10 stems, into `runtime/cache/joints_json_samples/`. Around 100 KB total, small enough to commit to the repo. The FE devs work against those locally. Production swaps to the full cache once it lands.

Choose the 10 sample stems to cover the obvious edge cases:
- A few mid-rally strokes (typical case)
- A first-shot-in-rally (boundary on `start_f`)
- A last-shot-in-rally (boundary on `end_f`)
- One with a high `_failed.npy` rate (overlay needs to gracefully not draw on failed frames)
- A mix of Top and Bottom strikers

The build script's CLI should have a `--samples-only` flag for producing just these fixtures, so the FE team can regenerate them without running the full build.

## Things to talk to Curtis and Kiri about

This needs an actual conversation, not just a writeup. 30-min call before any FE code gets written. Things to settle:

- Endpoint URL shape. `GET /api/joints/{clip_stem}` vs `GET /api/clips/{clip_stem}/joints` vs nested under model registry. What fits their existing route conventions?
- CORS settings if the FE dev server runs on a different port to the API server
- Whether to return the joints array straight or wrap it in an envelope like `{joints: [...], meta: {...}}`. Their call.
- Whether the FE component already has a canvas-overlay primitive on the clip viewer. If yes, this is "add a draw layer to the existing component". If no, this is "build a clip viewer with overlay support, then add the draw layer".
- The frame-sync approach. `requestAnimationFrame` querying `video.currentTime` is the standard way. Worth confirming they're happy with that vs hooking video events.
- How they want to handle frames where `_failed` is true. Skip the draw, draw greyed out, hold the last good frame? UX call, not technical.

## Constraints

- Limited time, exams approaching. Minimum viable plus tested, not over-engineered. Ship something that works first, polish second.
- Static-overlay reference already exists and works. Lift its joint-pair map, colour choices, and drawing logic. Don't reinvent them.
    - Primary reference: `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/render_detection_overlays.py`
    - Sibling variant: `src/bst_refactor/validation_scripts/mmpose_heuristic_investigation/render_sticky_anchor_overlays.py`
- Don't touch `src/bst_refactor/` modules beyond reading them. That pipeline is stable. Not restructuring it.
- AU spelling. No em-dashes. Project `.claude/CLAUDE.md` and `~/.claude/CLAUDE.md` style applies.
- I have mild protanopia. If you're proposing colours for the skeleton lines, reference `~/Documents/protan_colour_scheme.md` first.

## This doc is a preliminary scout. Verify before trusting.

Everything in this file is based on one conversation. Path locations might be wrong. NPY shapes might differ from what I wrote down. File-size estimates are arithmetic, not measured. Don't assume any of it is correct without checking.

Before you commit to code or more documentation:

- Verify the source NPY shapes against actual files. Open one, print the shape.
- Verify the file paths. Some might have moved since I last looked.
- Read the static-overlay reference scripts properly. They're the ground truth for what a "good overlay" looks like.
- Read `~/Documents/COSC594/frontend_integration_guide.md` for the FE-side context.
- Read `src/api/main.py`, `src/api/inference.py`, `src/api/jobs.py` to see how the existing API is structured. Match those patterns.
- Read `scratch/scratch_layout.md` for the existing cache directory conventions.
- Read `~/.claude/CLAUDE.md` and `.claude/CLAUDE.md` for style rules.

If anything is ambiguous, ask before guessing. Specific things I expect to need to discuss:
- JSON shape (flat vs envelope, what metadata to include)
- How the FE wants the endpoint URL shaped
- Whether to commit the sample fixtures or generate them on demand
- Whether to ship one big rally-level JSON alongside the per-stem ones (I think no, but worth flagging)

Don't write more documentation files until I've signed off on the approach. One planning doc is plenty. If you want to record an architecture decision separately, ask first.

## What to do first

1. Read this doc end to end.
2. Read the reference files listed above.
3. Verify the shapes and paths I've claimed. Note anything that doesn't match.
4. Come back with: (a) an implementation plan, (b) a list of clarifying questions, (c) what you found that contradicts or refines anything in this doc.

Don't start writing code until I've reviewed your plan and signed off on it.
