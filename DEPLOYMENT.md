# Deployment Guide

This document describes how to deploy and run this application in production.

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
/srv/dev-disk-by-uuid-2d4dc55c-51cd-46ae-a477-544cf0f76bb5/320_cosc594_data:/data/cosc594:ro
```

appears to the container as:

```
/data/cosc594
```

- Provides access to external dataset files for inference and processing
- Mounted directly from the host machine
- Marked read-only for data safety

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