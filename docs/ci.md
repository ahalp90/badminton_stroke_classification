# CI / CD

This repo's continuous integration lives in `.github/workflows/`. There is no
automated deployment — production is still a manual `docker compose -f
docker-compose.prod.yml up --build -d` on the server (see `DEPLOYMENT.md`). CI's
job is to catch problems *before* that manual deploy.

## Workflows at a glance

### `ci.yml` — runs on PRs and pushes to `main`

| Job | What it does | Blocking? |
|-----|--------------|-----------|
| `lint` | `ruff check .` (ruff pinned to the uv.lock version) | ✅ |
| `test` | installs CPU torch + pinned `requirements.txt`, runs `pytest` | ✅ |
| `frontend` | `npm ci && npm run lint && npm run build` — **self-skips** (stays green) if `frontend/` is ever removed | ✅ (build only; lint is a warning) |
| `docker-build` | `docker build` of the backend and frontend images — **no push**, just proves the Dockerfiles still build | ✅ |

It has `concurrency` cancellation (a newer push cancels the older run), per-job
`timeout-minutes`, and least-privilege `permissions: contents: read`.

The `frontend` job is intentionally written to **always run and self-skip** rather
than using a `paths:` filter — a path-filtered *required* check that never runs
would block PRs forever. So if the frontend is deleted later, the job logs
"skipping" and passes; if it stays, it's built on every PR.

### `pr-quality.yml` — runs on PRs only

| Job | What it does | Blocking? |
|-----|--------------|-----------|
| `commit-lint` | `gitlint` over the PR's commits (rules in `.gitlint`) | ✅ |
| `pr-body` | checks the PR description has filled-in **What / Why / How tested** sections | ✅ |
| `advisory` | Gemini reads the commits + PR body + diffstat and comments on legibility/substance | ❌ never blocks |

## The AI advisory (free, optional, non-blocking)

`pr-quality.yml`'s `advisory` job runs `scripts/pr_advisory.py`, which asks a
cheap/fast LLM whether the commit messages are human-legible and the PR
description substantively explains the change, then posts a single sticky PR
comment (and writes the same to the run's step summary).

It is **dormant until you add an API key**, and it is designed so it can never
get in the way:

* **No key set →** the job skips silently. No nagging, no failures.
* **Key set, model reachable →** it posts an advisory comment. Purely advice.
* **Key set, but rate-limited / unreachable / mis-named model / quota hit →** it
  emits a GitHub `::warning::` annotation explaining what happened and **still
  passes**. The script always exits 0, so a flaky free tier never blocks a merge.

### Enable it (Gemini free tier — recommended for this project)

1. Get a free API key from Google AI Studio (<https://aistudio.google.com/app/apikey>).
   The free tier allows ~1,500 requests/day — far more than this project's PR
   volume, because the model is called **once per PR**, not per commit.
2. In GitHub: **Settings → Secrets and variables → Actions → New repository
   secret**, name it `GEMINI_API_KEY`.
3. (Optional) If the default model name (`gemini-2.5-flash`) is ever retired,
   set a repository **variable** `GEMINI_MODEL` to a current free model — no code
   change needed. A wrong model name degrades to a warning, it won't fail CI.

To switch providers later (e.g. Claude Haiku), it's a one-file change in
`scripts/pr_advisory.py`; the workflow wiring stays the same.

> **Forks:** GitHub does not expose secrets to PRs opened from forks, so the
> advisory only runs on branches pushed to this repo. That's expected for a
> small team.

## Dependency pinning

`requirements.txt` is **pinned to `uv.lock`** so CI tests exactly what the Docker
image ships (both install the same file). `uv.lock` is the source of truth.

* `torch` / `torchvision` are **not** pinned in `requirements.txt` — CI and the
  Dockerfile install them first from an explicit PyTorch index (CPU for CI/dev,
  `cu128` for the prod GPU box), so pinning them here would fight that step.
* After changing dependencies (`pyproject.toml` + `uv lock`), run:

  ```sh
  ./scripts/gen-requirements.sh --check   # reports any pin that drifted from uv.lock
  ```

  then update the flagged versions in `requirements.txt`.

## Local pre-commit hooks (optional)

`.pre-commit-config.yaml` runs the **same** gitlint + ruff rules locally so you
get feedback before pushing instead of from a red CI run:

```sh
uv tool install pre-commit          # or: pipx install pre-commit
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

## Making the checks actually required

CI checks only *block* a merge once they're marked **required** under branch
protection (a repo-admin setting):

**Settings → Branches → Add branch ruleset** (or *Branch protection rules*) for
`main` → enable **Require status checks to pass** and select: `lint`, `test`,
`frontend`, `docker-build`, `commit-lint`, `pr-body`. Leave `advisory`
**unselected** — it's meant to stay non-blocking.
