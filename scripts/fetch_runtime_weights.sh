#!/usr/bin/env bash
# Fetch the gitignored weight files that LIVE BRIC inference needs.
#
# A fresh clone has only .gitkeep placeholders under runtime/, so the BRIC card
# degrades to stub results until these are in place. BRIC stays GPU-gated, so
# a CPU-only box needs none of this; run it on the GPU/demo box before
# `docker compose -f docker-compose.prod.yml up --build` (prod bakes the image
# with COPY . ., so the files must sit in the repo dir first).
#
# Fetches four files into the layout the code expects:
#   runtime/deployed/bric/<run>/best.pt              (BRIC model, GitHub release)
#   runtime/checkpoints/tracknetv3/TrackNet_best.pt  (shuttle tracker, Drive zip)
#   runtime/checkpoints/tracknetv3/InpaintNet_best.pt
#   runtime/checkpoints/yolo11/yolo11n.pt            (player detector, ultralytics)

# Re-exec under bash if started with sh/dash (uses [[ ]], BASH_SOURCE).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/fetch_runtime_weights.sh [--force]

Downloads the live-inference weights for BRIC into runtime/.

Options:
  --force      Re-download even if the destination already exists.
  -h, --help   Show this help and exit.

Environment:
  MODELS_RELEASE      GitHub release tag for the BRIC weight (default: models-v1).
  TRACKNET_ZIP        Path to an already-downloaded TrackNetV3 checkpoints zip.
                      Use this if the Google Drive download is blocked; the
                      zip link is in src/bric/perception/_vendor/tracknetv3/README.md.
  YOLO_URL            Override the ultralytics yolo11n.pt asset URL.

Requires: gh (authenticated), curl, unzip. The TrackNet step also needs gdown
(`pip install gdown`) unless TRACKNET_ZIP is set.
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
    *) echo "unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

say() { printf '\033[1;34m[weights]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

MODELS_RELEASE="${MODELS_RELEASE:-models-v1}"
YOLO_URL="${YOLO_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt}"

BRIC_RUN="runtime/deployed/bric/20260518_013238_rgb_shuttle-tcn-outgoing_only_une_merge_v1_nosides_42"
BRIC_DEST="$BRIC_RUN/best.pt"
BRIC_PATTERN='bric_20260518_013238*'

TRACKNET_DIR="runtime/checkpoints/tracknetv3"
TRACKNET_DEST="$TRACKNET_DIR/TrackNet_best.pt"
INPAINT_DEST="$TRACKNET_DIR/InpaintNet_best.pt"
TRACKNET_DRIVE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"

YOLO_DIR="runtime/checkpoints/yolo11"
YOLO_DEST="$YOLO_DIR/yolo11n.pt"

command -v curl >/dev/null 2>&1 || die "curl is required."

have_dest() {  # $1=path -> 0 if present and not forcing
  [[ -s "$1" && $FORCE -eq 0 ]]
}

# 1) BRIC model from the GitHub release.
fetch_bric() {
  if have_dest "$BRIC_DEST"; then say "exists, skipping: $BRIC_DEST"; return; fi
  command -v gh >/dev/null 2>&1 || die "gh is required for the BRIC weight (or place it at $BRIC_DEST by hand)."
  mkdir -p "$BRIC_RUN"
  say "downloading BRIC weight from release $MODELS_RELEASE"
  local tmp; tmp="$(mktemp -d)"
  gh release download "$MODELS_RELEASE" --pattern "$BRIC_PATTERN" --dir "$tmp" \
    || die "gh release download failed (is gh authenticated, and is asset $BRIC_PATTERN on $MODELS_RELEASE?)"
  local f; f="$(find "$tmp" -name '*.pt' -print -quit)"
  [[ -n "$f" ]] || die "no .pt found in the BRIC release download."
  mv "$f" "$BRIC_DEST"
  rm -rf "$tmp"
  say "placed $BRIC_DEST"
}

# 2) TrackNet + InpaintNet from the upstream TrackNetV3 checkpoints zip.
fetch_tracknet() {
  if have_dest "$TRACKNET_DEST" && have_dest "$INPAINT_DEST"; then
    say "exists, skipping: $TRACKNET_DEST, $INPAINT_DEST"; return
  fi
  command -v unzip >/dev/null 2>&1 || die "unzip is required for the TrackNet checkpoints."
  mkdir -p "$TRACKNET_DIR"
  local tmp zip; tmp="$(mktemp -d)"; zip="$tmp/tracknetv3_ckpts.zip"
  if [[ -n "${TRACKNET_ZIP:-}" ]]; then
    [[ -f "$TRACKNET_ZIP" ]] || die "TRACKNET_ZIP set but not found: $TRACKNET_ZIP"
    cp "$TRACKNET_ZIP" "$zip"
  elif command -v gdown >/dev/null 2>&1; then
    say "downloading TrackNetV3 checkpoints from Google Drive"
    gdown "$TRACKNET_DRIVE_ID" -O "$zip" || die "gdown failed; download the zip by hand and re-run with TRACKNET_ZIP=<path>."
  elif python -c "import gdown" >/dev/null 2>&1; then
    say "downloading TrackNetV3 checkpoints from Google Drive (python -m gdown)"
    python -m gdown "$TRACKNET_DRIVE_ID" -O "$zip" || die "gdown failed; download the zip by hand and re-run with TRACKNET_ZIP=<path>."
  else
    die "TrackNet checkpoints need gdown (pip install gdown), or download the zip from src/bric/perception/_vendor/tracknetv3/README.md and re-run with TRACKNET_ZIP=<path>."
  fi
  unzip -o -q "$zip" -d "$tmp/x" || die "unzip failed for the TrackNet zip."
  local tnet inp
  tnet="$(find "$tmp/x" -name 'TrackNet_best.pt' -print -quit)"
  inp="$(find "$tmp/x" -name 'InpaintNet_best.pt' -print -quit)"
  [[ -n "$tnet" ]] || die "TrackNet_best.pt not found in the zip."
  [[ -n "$inp" ]]  || die "InpaintNet_best.pt not found in the zip."
  mv "$tnet" "$TRACKNET_DEST"
  mv "$inp" "$INPAINT_DEST"
  rm -rf "$tmp"
  say "placed $TRACKNET_DEST, $INPAINT_DEST"
}

# 3) YOLO11n player detector (stock ultralytics asset).
fetch_yolo() {
  if have_dest "$YOLO_DEST"; then say "exists, skipping: $YOLO_DEST"; return; fi
  mkdir -p "$YOLO_DIR"
  say "downloading yolo11n.pt from ultralytics"
  curl -fL --retry 3 -o "$YOLO_DEST" "$YOLO_URL" || die "yolo11n.pt download failed from $YOLO_URL"
  say "placed $YOLO_DEST"
}

fetch_bric
fetch_tracknet
fetch_yolo
say "done. live BRIC inference weights are in place under runtime/."
