#!/usr/bin/env bash
#
# Keep requirements.txt pins in sync with uv.lock (the source of truth).
#
# uv.lock is what local dev installs; requirements.txt is what CI and the Docker
# image install. They must agree, so requirements.txt is pinned to the versions
# uv.lock resolves. This script compares the two and reports drift.
#
#   ./scripts/gen-requirements.sh          # print each pin: file vs uv.lock
#   ./scripts/gen-requirements.sh --check  # exit non-zero if any pin drifted (CI-friendly)
#
# After it flags drift, edit the version in requirements.txt to match the lock
# column. torch / torchvision are skipped on purpose (installed from an explicit
# PyTorch index, not pinned here -- see the requirements.txt header).
set -euo pipefail

cd "$(dirname "$0")/.."

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

command -v uv >/dev/null 2>&1 || { echo "error: 'uv' is required (https://docs.astral.sh/uv/)"; exit 2; }

EXPORT="$(mktemp)"
trap 'rm -f "$EXPORT"' EXIT

# Resolve the full set CI/Docker install (every extra except the MMPose subprocess venv).
uv export --frozen --no-hashes --no-emit-project \
  --extra bric-runtime --extra bric-train --extra bst-x-runtime --extra dev \
  > "$EXPORT"

# Normalise a distribution name the way PyPI does (lowercase; -, _ and . equivalent).
norm() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '._' '--'; }

drift=0
printf '%-26s %-14s %-14s %s\n' "PACKAGE" "requirements" "uv.lock" "STATUS"
printf '%-26s %-14s %-14s %s\n' "-------" "------------" "-------" "------"

# Pinned (name==version) lines from requirements.txt, inline comments stripped.
while IFS= read -r line; do
  pkg="${line%%==*}"
  pinned="${line#*==}"
  pinned="${pinned%%[[:space:]]*}"        # drop trailing "  # comment"
  npkg="$(norm "$pkg")"

  # Look up the lock version (strip any "; marker"); match on normalised name.
  lock="$(awk -F'==' -v want="$npkg" '
    {
      n=$1; v=$2; sub(/[[:space:];].*$/, "", v);
      gsub(/[._]/, "-", n); n=tolower(n);
      if (n==want) { print v; exit }
    }' "$EXPORT")"

  if [ -z "$lock" ]; then
    status="NOT IN LOCK"; drift=1
  elif [ "$lock" = "$pinned" ]; then
    status="ok"
  else
    status="DRIFT"; drift=1
  fi
  printf '%-26s %-14s %-14s %s\n' "$pkg" "$pinned" "${lock:-—}" "$status"
done < <(grep -E '^[A-Za-z0-9._-]+==' requirements.txt)

echo
if [ "$drift" -ne 0 ]; then
  echo "Pins differ from uv.lock. Update the versions in requirements.txt to the uv.lock column above."
  [ "$CHECK" -eq 1 ] && exit 1
else
  echo "All pins match uv.lock. ✅"
fi
