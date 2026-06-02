#!/usr/bin/env bash
# One-time (idempotent) local dev bootstrap for the badminton stroke classifier.
#
# What it does:
#   1. Checks Docker + Docker Compose v2 are available.
#   2. Creates .env from .env.example if missing.
#   3. Creates frontend/.env.local with a working Vite proxy + LAN allow-list.
#   4. Creates the (initially empty) dataset mount dirs the dev overlay expects.
#   5. Prints the command to bring the stack up (or runs it with --up).
#
# It will NOT overwrite an existing .env or frontend/.env.local unless --force.
# It does NOT download datasets or model weights: see HANDOVER.md and MODELS.md.

# Re-exec under bash if started with sh/dash. This script uses bash features
# ([[ ]], BASH_SOURCE); running `sh dev-setup.sh` would otherwise fail with
# "Bad substitution" / "[[: not found". The guard must stay POSIX-only.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/dev-setup.sh [options]

Options:
  --up         Bootstrap, then run docker compose up (dev overlay).
  --force      Overwrite existing .env and frontend/.env.local.
  -h, --help   Show this help and exit.

Environment:
  LAN_IP=<ip>  Pin a specific LAN IP in VITE_ALLOWED_HOSTS instead of
               auto-detecting (useful on multi-homed or macOS hosts).

Examples:
  ./scripts/dev-setup.sh
  ./scripts/dev-setup.sh --up
  LAN_IP=192.168.1.50 ./scripts/dev-setup.sh --force
USAGE
}

# --- locate repo root (parent of this script's dir) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

FORCE=0
DO_UP=0
for arg in "$@"; do
  case "$arg" in
    --up)        DO_UP=1 ;;
    --force)     FORCE=1 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; echo "try: ./scripts/dev-setup.sh --help" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. sanity: are we in the repo root? ---
[[ -f .env.example && -f docker-compose.yml ]] || \
  die "Run this from a clone of the repo (could not find .env.example / docker-compose.yml at $REPO)."

# --- 1. prerequisites ---
command -v docker >/dev/null 2>&1 || die "Docker is not installed or not on PATH."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required (try: docker compose version)."
say "Docker and Docker Compose v2 found."

# --- 2. root .env ---
if [[ -f .env && $FORCE -eq 0 ]]; then
  say ".env already exists, leaving it."
else
  cp .env.example .env
  warn "Created .env from .env.example. Fill in the BST_* dataset paths if you"
  warn "have ShuttleSet data locally (optional for just running the web app)."
fi

# --- 3. frontend/.env.local (the proxy + LAN allow-list, the 502/blank-screen fix) ---
LAN_IP="${LAN_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
ALLOWED="localhost,127.0.0.1"
[[ -n "${LAN_IP:-}" ]] && ALLOWED="$ALLOWED,$LAN_IP"

if [[ -f frontend/.env.local && $FORCE -eq 0 ]]; then
  say "frontend/.env.local already exists, leaving it."
else
  cat > frontend/.env.local <<EOF
# Local-only Vite config (gitignored). vite.config.js reads these via loadEnv().
# VITE_API_TARGET must point at the backend SERVICE name (reachable from inside
# the frontend container), NOT localhost. VITE_ALLOWED_HOSTS lets another
# computer on the LAN reach the dev server by hostname (raw IPs always work).
VITE_API_TARGET=http://backend:8000
VITE_ALLOWED_HOSTS=$ALLOWED
EOF
  say "Wrote frontend/.env.local (allowed hosts: $ALLOWED)."
  [[ -z "${LAN_IP:-}" ]] && warn "Could not detect a LAN IP; pass LAN_IP=<ip> to allow LAN access by hostname."
fi

# --- 4. dataset mount dirs (created empty; populate per HANDOVER.md) ---
for d in scratch/inspect_clips scratch/bst_inputs runtime/uploads; do
  if [[ -d "$d" ]]; then
    say "$d exists."
  else
    mkdir -p "$d"
    say "Created empty $d (populate it per HANDOVER.md to enable clips/inference)."
  fi
done

UP_CMD="docker compose -f docker-compose.yml -f docker-compose.dev.yml up"
echo
say "Bootstrap complete."
if [[ $DO_UP -eq 1 ]]; then
  say "Starting the stack (first run pulls images and runs npm install, which is slow)..."
  exec $UP_CMD
else
  say "Start the dev stack with:"
  echo "    $UP_CMD"
  say "Then open http://localhost:5173 (or http://<LAN_IP>:5173 from another device)."
fi
