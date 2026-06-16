#!/usr/bin/env bash
# Fetch the deployed model weights that the registry serves. The 6 BST-X
# weights are tracked in git AND uploaded as assets on a GitHub Release; this
# script is the canonical fetch path for fresh clones and for any new weights
# published outside git (e.g. BRIC, which is not in-tree). The release also
# carries 8 BRIC assets uploaded alongside (see MODELS.md). You only need
# this script for LIVE inference on new uploads or for retraining.
# The demo's results screens run from precomputed predictions that ARE in git,
# so a fresh clone shows working results without running this.
#
# Reads scripts/model_manifest.tsv (dest_path <TAB> asset_name <TAB> sha256) and
# downloads each asset from $MODELS_BASE_URL into dest_path.

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: MODELS_BASE_URL=<url> ./scripts/fetch-models.sh [--force]

Downloads deployed model weights listed in scripts/model_manifest.tsv.

Required:
  MODELS_BASE_URL   Base URL the asset names hang off, e.g.
                    https://github.com/<org>/<repo>/releases/download/models-v1

Options:
  --force           Re-download even if the destination already exists.
  -h, --help        Show this help.

See MODELS.md for how to publish the release these assets come from.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m[models]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

MANIFEST="scripts/model_manifest.tsv"
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
[[ -n "${MODELS_BASE_URL:-}" ]] || die "set MODELS_BASE_URL (see --help and MODELS.md)."
command -v curl >/dev/null 2>&1 || die "curl is required."

BASE="${MODELS_BASE_URL%/}"
n=0; got=0; skipped=0
while IFS=$'\t' read -r dest asset sha || [[ -n "$dest" ]]; do
  # skip comments and blank lines
  [[ -z "$dest" || "$dest" == \#* ]] && continue
  n=$((n+1))
  if [[ -f "$dest" && $FORCE -eq 0 ]]; then
    say "exists, skipping: $dest"
    skipped=$((skipped+1))
    continue
  fi
  mkdir -p "$(dirname "$dest")"
  say "downloading $asset -> $dest"
  curl -fL --retry 3 -o "$dest" "$BASE/$asset" || die "download failed for $asset"
  if [[ -n "${sha:-}" && "$sha" != "-" ]]; then
    actual="$(sha256sum "$dest" | awk '{print $1}')"
    [[ "$actual" == "$sha" ]] || die "checksum mismatch for $dest (expected $sha, got $actual)"
    say "checksum OK"
  fi
  got=$((got+1))
done < "$MANIFEST"

say "done: $got downloaded, $skipped already present, $n total."
