#!/usr/bin/env bash
# Publish the deployed run-time weights as GitHub Release assets, one-time, by
# the current team. Pairs with scripts/fetch-models.sh (the next team's side).
# See MODELS.md for the full rationale.
#
# Run from a clone that has the weight files on disk (e.g. this worktree).
# Requires: gh (authenticated), sha256sum.

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# --- config (override via env) ---
REPO="${REPO:-Kira-Le/badminton_stroke_classification}"
TAG="${TAG:-models-v1}"

usage() {
  cat <<'USAGE'
Usage: [REPO=owner/name] [TAG=models-v1] ./scripts/publish-models.sh

Steps performed:
  1. Fill scripts/model_manifest.tsv sha256 column from the local weight files.
  2. Create the release TAG on REPO (skips if it already exists).
  3. Upload each manifest weight as a release asset under its asset_name.

After it finishes, commit the updated manifest yourself, and the next team
fetches with:
  MODELS_BASE_URL=https://github.com/REPO/releases/download/TAG ./scripts/fetch-models.sh
USAGE
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

say() { printf '\033[1;34m[publish]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

MANIFEST="scripts/model_manifest.tsv"
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
command -v gh >/dev/null 2>&1 || die "gh CLI is required and must be authenticated."
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required."

# Loops read via process substitution (NOT a pipe) so die exits the whole
# script and comment/blank lines pass through unchanged.

# --- 1. fill sha256 column from local files ---
say "Computing sha256 for each weight..."
{
  while IFS= read -r line; do
    if [[ -z "$line" || "$line" == \#* ]]; then printf '%s\n' "$line"; continue; fi
    IFS=$'\t' read -r dest asset _ <<<"$line"
    [[ -f "$dest" ]] || die "weight file missing on disk: $dest"
    printf '%s\t%s\t%s\n' "$dest" "$asset" "$(sha256sum "$dest" | awk '{print $1}')"
  done < <(cat "$MANIFEST")
} > "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"
say "Updated $MANIFEST with checksums."

# --- 2. create the release (idempotent) ---
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  say "Release $TAG already exists on $REPO, reusing it."
else
  say "Creating release $TAG on $REPO..."
  gh release create "$TAG" --repo "$REPO" \
    --title "Model weights ($TAG)" \
    --notes "Deployed run-time weights served by the registry. Fetch with scripts/fetch-models.sh."
fi

# --- 3. upload each weight as an asset ---
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  IFS=$'\t' read -r dest asset _ <<<"$line"
  say "uploading $asset"
  gh release upload "$TAG" "$dest#$asset" --repo "$REPO" --clobber
done < <(cat "$MANIFEST")

say "Done. Next team fetches with:"
echo "    MODELS_BASE_URL=https://github.com/$REPO/releases/download/$TAG ./scripts/fetch-models.sh"
say "Remember to commit the updated $MANIFEST."
