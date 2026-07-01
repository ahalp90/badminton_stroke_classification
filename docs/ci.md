# CI / CD

CI lives in `.github/workflows/` — the YAML is commented. No auto-deploy: prod is
still a manual `docker compose -f docker-compose.prod.yml up --build -d` (see
`DEPLOYMENT.md`). This doc covers only what you can't get from reading the YAML.

## What runs

**`ci.yml`** (PRs + pushes to `main`), all blocking:
`lint` (ruff) · `test` (pytest) · `frontend` (npm build; self-skips if `frontend/`
is removed) · `docker-build` (builds the images, no push).

**`pr-quality.yml`** (PRs):
`commit-lint` (gitlint, rules in `.gitlint`) · `pr-body` (needs **What / Why /
Testing** sections) · `advisory` (AI review, never blocks).

## Enable the AI advisory (optional, free)

Off until you add a key; without one it skips silently. With one, it comments on
commit/PR legibility and only ever *warns* on rate limits or outages — never blocks.

1. Free key: <https://aistudio.google.com/app/apikey>
2. Add it as repo secret **`GEMINI_API_KEY`** (Settings → Secrets and variables → Actions).
3. Optional: set repo variable `GEMINI_MODEL` if the default `gemini-2.5-flash` is retired.

Called once per PR, so the ~1,500/day free tier is plenty. Fork PRs don't get
secrets, so it runs on in-repo branches only.

## Dependencies

`requirements.txt` is pinned from `uv.lock` so CI installs what the image ships.
`torch`/`torchvision` are unpinned — CI and the Dockerfile install them from an
explicit PyTorch index first. After changing deps, run
`./scripts/gen-requirements.sh --check` and update any drifted pins.

## Make the checks required

They only block merges once marked required: Settings → Branches → ruleset for
`main` → Require status checks → select every job **except `advisory`**.

## Local hooks (optional)

`pre-commit install --hook-type commit-msg --hook-type pre-commit` runs the same
gitlint + ruff before you push.
