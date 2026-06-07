# Deployment Guide

This document describes how to deploy and run this application **in production**.
For local development see `docker-compose.dev.yml` and `HANDOVER.md`; for the
data, accounts, and access required to operate the project, see `HANDOVER.md`.

---

## Status

This guide and `docker-compose.prod.yml` describe the production demo as it ran
on a home server, exposed via Cloudflare Tunnel. It is a working setup, not yet
verified against the client's target environment.

TODO (team): confirm this is the canonical production design, or revise it for
the client environment. Until confirmed, treat it as a working starting point.

---

## Dev vs production at a glance

| | Local dev | Production demo |
| --- | --- | --- |
| Compose files | `docker-compose.yml` + `docker-compose.dev.yml` | `docker-compose.prod.yml` |
| Frontend served by | Vite dev server (HMR) | nginx (static build) |
| Port | 5173 | 26138 (behind Cloudflare Tunnel) |
| API routing | Vite proxy `/api` -> `backend:8000` | nginx proxies `/api` -> backend |
| Dataset | empty `scratch/*` dirs, optional | `${DATA_HOST_DIR}` mounted read-only |

> Env-file gotcha (applies to dev): `VITE_API_TARGET` must be the backend
> service name (`http://backend:8000`), not `localhost`. A wrong value yields a
> 502 on `/api/*`. See `HANDOVER.md` section 5 for why `loadEnv` makes the
> container env override `frontend/.env.local`.

---

## Architecture Overview

The system consists of:

- **Frontend**: React app built and served via nginx
- **Backend**: FastAPI (Uvicorn, single worker)
- **Proxying**: nginx routes `/api/*` to backend
- **Exposure**: Single public entrypoint via port `26138`
- **Tunnel**: Cloudflare Tunnel exposes the frontend externally

```text
Browser
   ↓
nginx (frontend :26138)
   ↓ /api
FastAPI backend (internal :8000)
   ↓
Dataset (read-only mount)
Uploads volume (persistent storage)
```

---

## Prerequisites

Ensure the server has:

- Docker
- Docker Compose v2+
- Git
- (Optional) Cloudflare Tunnel configured

---

## Production Deployment

From the project root:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

### What happens:

- Backend image is built (FastAPI + Uvicorn)
- Frontend image is built (React → nginx static build)
- Both containers are started in detached mode
- Docker attaches persistent volumes automatically
- Internal Docker network is created for service communication

---

## Access Points

### Frontend (public entrypoint)

```text id="ac1"
http://localhost:26138
```
This is the only externally exposed service.

### Backend (internal only)

```
http://localhost:8000
```
Only used via nginx.

### Health Check (internal)

```
http://localhost:8000/health
```
used by docker to verify backend health status.

## Volumes

### Uploads

```
uploads:/app/uploads
```

- Stores user-generated uploads
- Persists across container restarts, image rebuilds and docker compose up/down

### Dataset

```
${DATA_HOST_DIR}:/data:ro
```

The host path comes from `DATA_HOST_DIR` (set it in `.env` — see `.env.example`)
and appears to the container as:

```
/data
```

- Provides access to external dataset files for inference and processing
- Mounted directly from the host machine
- Marked read-only for data safety
- Required: compose refuses to start if `DATA_HOST_DIR` is unset. Comment out the
  mount in `docker-compose.prod.yml` to run without the dataset.

#### Live per-clip inference (`BST_INPUTS_DIR`)

The per-clip browser and the "Errors only" filter on the Model Results screen
only render when live per-clip predictions are available — i.e. when the SCP'd
collation tensors are reachable inside the container. The backend looks for:

```
$BST_INPUTS_DIR/{test,val}/{JnB_bone,pos,shuttle,videos_len}.npy
```

`docker-compose.prod.yml` defaults `BST_INPUTS_DIR=/data`, which is correct if
your `DATA_HOST_DIR` points directly at a `bst_inputs`-shaped tree (i.e. it
contains `test/` and `val/` at its root). If your dataset has those tensors
under a `bst_inputs/` subfolder, change it to `BST_INPUTS_DIR=/data/bst_inputs`.

If neither layout matches and the env var is wrong, the rest of the app still
works — only the per-clip browser falls back to the "not available in this
environment" note.

### Frontend

```
./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
```

- Provides a custom nginx configuration to the frontend container
- Controls how nginx serves the backend build and routes requests

## Updating Deployment

To deploy updates:
```bash
git pull
docker compose -f docker-compose.prod.yml up --build -d
```
This will:

- Rebuild changed images
- Restart updated containers
- Preserve all volumes

## Security Notes

- Backend is not publicly exposed
- Only nginx is exposed externally

## Full Redeploy

```bash
docker compose -f docker-compose.prod.yml down -v
docker compose -f docker-compose.prod.yml up --build -d
```

## Notes

- Designed for single-server deployment
- Not currently horizontally scalable
- Stateless frontend