# Project Handover

A practical guide for someone taking over the badminton stroke classifier. It
covers the things the code and git history cannot tell you: where data lives,
what accounts you need, how the environments fit together, and the gotchas that
will otherwise cost you an afternoon.

For what the project *is* (architecture, results, taxonomy) see `README.md`.
For production deployment see `DEPLOYMENT.md`.

> Items marked **TODO (team)** are facts only the current team holds. Fill them
> in before this document is useful to an outsider.

---

## 1. The 60-second mental model

- **Backend**: FastAPI (`src/api`), single Uvicorn worker (the job store is
  in-memory, so it must not be split across workers).
- **Frontend**: React + Vite (`frontend/`). In dev it runs the Vite dev server
  with HMR; in prod it is built to static files and served by nginx.
- **Three environments**, three compose files:
  | Environment | Files | Frontend serves via | Port |
  | --- | --- | --- | --- |
  | Local dev | `docker-compose.yml` + `docker-compose.dev.yml` | Vite dev server (HMR) | 5173 |
  | Production demo | `docker-compose.prod.yml` | nginx (static build) | 26138 |
- **Public access** in prod is a Cloudflare Tunnel pointing at the frontend
  port. The backend is never exposed directly.

---

## 2. First run as a new maintainer

```bash
git clone <repo>
cd badminton_stroke_classification
./scripts/dev-setup.sh          # creates env files + empty mount dirs
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
# open http://localhost:5173
```

This boots a working app with the **sample data that ships in the repo**. The
data-dependent features (clip playback, live per-clip inference) stay off until
you populate the dataset dirs, below. Nothing else is required to see the UI and
the precomputed results.

---

## 3. Data provenance (the most important section)

The repo deliberately does not contain the large datasets. Each item below is
obtained out-of-band and dropped into a known path. With the paths empty the app
still runs; specific features degrade gracefully.

| Data | Used for | Dev path | Prod path | In git? |
| --- | --- | --- | --- | --- |
| Sample inspect clips | Clip playback / library | `scratch/inspect_clips/` | n/a | partial sample tracked |
| BST collation tensors | Live per-clip inference | `scratch/bst_inputs/` | `${DATA_HOST_DIR}` (`/data`) | no |
| ShuttleSet (clips + npy) | Training / data pipeline | set `BST_*` in `.env` | n/a (training is off-server) | no |
| Model registry | Model cards / selection | `docs/models_registry.yaml` | same | yes |
| Precomputed predictions | Results / model screens | served from each model's run dir | same | **yes (~11 MB)** |
| Model weights + training arrays | Live inference / re-training | each run dir | `${DATA_HOST_DIR}` for live | **yes, ~440 MB (see below)** |

**BST collation tensors** — the backend looks for, per `DEPLOYMENT.md`:
```
$BST_INPUTS_DIR/{test,val}/{JnB_bone,pos,shuttle,videos_len}.npy
```
- TODO (team): **where are these SCP'd from?** Document the source host /
  account / path, e.g. `scp engelbart:/scratch/comp320a/.../bst_inputs ...`.
- Without them, the per-clip browser and the "Errors only" filter stay hidden
  (every model reports `live=false`). The rest of the app is unaffected.

**ShuttleSet** — large; only needed to re-run training or the data pipeline, not
to serve the demo. Example HPC paths are in `.env.example`. TODO (team): confirm
the canonical location and how a new owner gets read access.

**Clips (video) — do NOT host these ourselves.** Three buckets:
- 13 sample clips ship in the repo at `scratch/inspect_clips/` (~7.7 MB,
  committed). They drive the demo per-clip player out of the box.
- `clips_local/` is an empty drop-in: put `<clip_stem>.mp4` there to play more
  clips locally. Never committed.
- The full broadcast corpus (~32k clips, tens of GB) is the **public ShuttleSet
  dataset**, stored on HPC `/scratch` per decision DL-008. We do not redistribute
  it: a new team re-obtains ShuttleSet from its public source and points
  `BST_CLIPS_DIR` at it. The repo already commits the mapping that says which
  clip is which (`notebooks/clips_master.csv`, `notebooks/shuttleset_splits_v2.csv`,
  `training/data/shuttleset/annotations/shots_master.csv`), so the corpus is
  reconstructable. TODO (team): record the exact ShuttleSet source URL and the
  HPC path/access.

So clips, unlike model weights, need **no GitHub Release**: samples are in-repo
and the rest is a re-downloadable public dataset keyed by the committed CSVs.

### Clip corpus: layout, backup, restore

Do **not** list the clips in this doc; the committed CSVs are the file-by-file
manifest. Record the structure and how to restore instead:

- **Layout** (`BST_CLIPS_DIR` root): `{split}/{Top,Bottom}_{stroke}/<stem>.mp4`,
  e.g. `train/Bottom_net_shot/16_2_17_15.mp4`. The stem encodes
  match/set/rally/shot indices.
- **Count**: ~32k clips. `notebooks/clips_master.csv` (33,481 rows) is the
  authoritative index of stem -> split + taxonomy class; `shuttleset_splits_v2.csv`
  is the split assignment. A restore can be verified against these.
- **Backup / restore**: full-corpus backups are held on **turing**. If the
  working store is wiped, restore by rsyncing the backup into `BST_CLIPS_DIR`
  and re-pointing `.env`. TODO (team): record turing's exact backup path and
  access here.
- **Last-resort rebuild**: ShuttleSet is public, so the corpus can be
  re-downloaded from source and re-split using the committed CSVs.

**Precomputed predictions are already in the repo and are small (~11 MB).** The
API serves BRIC's `eval/test_predictions.json` + `predictions/test.json` (4.5 MB)
and each BST-X model's gzipped `fe_jsons/*.json.gz` (6.6 MB total). All committed.
A fresh clone shows working results screens with no setup, so this is **not** a
handover gap and does not need separate hosting.

### Heavy model artifacts (the real repo-size question)

What inflates the repo is not the predictions but the binaries committed
alongside them:

| Tracked in git | Count | Size |
| --- | --- | --- |
| Model weights `*.pt` | 49 | ~345 MB |
| Training logit arrays `*.npz` | 63 | ~94 MB |
| TensorBoard event files | 466 | (rest) |
| `.git` history | | ~446 MB |

Every clone pulls ~440 MB. The API only needs the deployed weights (~43 MB for
the 6 BST-X models; BRIC's weight is not in the repo), not all 49 historical
checkpoints.

**Decision and mechanism are in `MODELS.md`:** run-time weights go to a GitHub
Release and are pulled with `scripts/fetch-models.sh`; the historical training
archive goes to bulk storage. `.gitignore` now blocks new artifacts from being
committed. The remaining team action is to publish the release and (optionally)
rewrite history to reclaim the existing 440 MB. None of this blocks running the
demo, which uses the committed ~11 MB predictions.

---

## 4. Accounts and access you must hand over

TODO (team) for each: who owns it, how to transfer, where credentials live.

- [ ] **Server** that runs the prod demo (SSH access, sudo, the deploy user).
- [ ] **Cloudflare Tunnel**: the account, the tunnel config / credentials file,
      and the public hostname it maps to port 26138. (Not in the repo.)
- [ ] **HPC (engelbart)** account for ShuttleSet data and training.
- [ ] **GitHub** repo ownership / admin and branch-protection settings.
- [ ] Any experiment-tracking or storage accounts still in use.

---

## 5. Environment files and the precedence gotcha

Two gitignored files hold local config. `scripts/dev-setup.sh` creates both.

- **`.env`** (repo root): backend dataset paths, and `VITE_*` values that
  `docker-compose.yml` injects into the frontend container.
- **`frontend/.env.local`**: read by `vite.config.js` via `loadEnv()`.

**Gotcha that caused a real outage:** `vite.config.js` calls `loadEnv(mode, cwd,
'')` with an *empty prefix*, which makes `loadEnv` merge **all** of
`process.env` and let it **override** the `.env.local` file. So the
container-injected `VITE_API_TARGET` (from the root `.env`) wins. Practical rule:

- Set `VITE_API_TARGET=http://backend:8000` (the service name, NOT `localhost`,
  which inside the frontend container points at itself). A wrong value here is a
  **502 on `/api/*`**.
- Keep the root `.env` and `frontend/.env.local` consistent, or just let
  `dev-setup.sh` write them.

---

## 6. Known gotchas (things that already bit us)

- **Blank screen on another computer**: usually a stale Vite module graph in a
  long-running container after files were renamed/removed. Fix: recreate the
  frontend container (`docker compose ... up -d --force-recreate frontend`).
- **"Expected a JavaScript module but got text/html"**: a module request fell
  through to the SPA `index.html` fallback. Cause is either the stale graph
  above, or a host blocked by `VITE_ALLOWED_HOSTS` (raw IPs are always allowed;
  hostnames must be listed).
- **502 on `/api/*`**: wrong `VITE_API_TARGET` (see section 5).
- **Uploads permission error on `/app/uploads`**: only on FUSE/mergerfs hosts
  where the container user cannot write the bind mount. The dev overlay routes
  uploads to a named volume to avoid it (prod already does this).

---

## 7. Deployment

See `DEPLOYMENT.md` for the full prod runbook. In brief:
```bash
# set DATA_HOST_DIR in .env first
docker compose -f docker-compose.prod.yml up --build -d
```
TODO (team): confirm `DEPLOYMENT.md` reflects the real client/target
environment, or mark it as the canonical guide. It is currently flagged
"to be confirmed."

---

## 8. Roadmap context

Phase 1 (this repo) is a stroke-type classifier. Later phases add rally
segmentation, scoring, and player grading (see `README.md`). TODO (team): link
the project board / tickets and name the current owner for each workstream.
