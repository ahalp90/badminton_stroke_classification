#!/usr/bin/env bash
# Rebuild x3d_integration_macro_plan.pdf from x3d_integration_macro_plan_print.md.
#
# Pipeline: pandoc (markdown -> HTML5 with inlined print CSS) -> chromium
# headless (HTML -> PDF). Preserves the no-orphaned-headings rule via the CSS
# at print_assets/x3d_macro_plan_print.css.
#
# Run from anywhere; resolves paths relative to this script.
#
# Edit the print-tuned source at:
#   docs/architecture_notes/x3d_integration_macro_plan_print.md
# Edit print styling at:
#   docs/architecture_notes/print_assets/x3d_macro_plan_print.css
# Output PDF lands at:
#   docs/architecture_notes/x3d_integration_macro_plan.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_MD="${NOTES_DIR}/x3d_integration_macro_plan_print.md"
CSS="${SCRIPT_DIR}/x3d_macro_plan_print.css"
INLINE_STYLE="${SCRIPT_DIR}/_inline_style.html"
HTML_OUT="${SCRIPT_DIR}/x3d_integration_macro_plan_print.html"
PDF_OUT="${NOTES_DIR}/x3d_integration_macro_plan.pdf"

for tool in pandoc chromium; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "missing tool: ${tool}" >&2
        exit 1
    }
done

[[ -f "${SRC_MD}" ]] || { echo "missing source: ${SRC_MD}" >&2; exit 1; }
[[ -f "${CSS}"    ]] || { echo "missing stylesheet: ${CSS}" >&2; exit 1; }

# Inline the CSS into a <style> block pandoc can include in the HTML <head>.
{
    printf '<style>\n'
    cat "${CSS}"
    printf '\n</style>\n'
} > "${INLINE_STYLE}"

pandoc "${SRC_MD}" \
    --standalone \
    --from markdown \
    --to html5 \
    --metadata title="X3D-S Wrist-Crop Fusion: Macro Plan" \
    --metadata lang=en-AU \
    --include-in-header="${INLINE_STYLE}" \
    --output "${HTML_OUT}"

chromium --headless --disable-gpu --no-sandbox \
    --no-pdf-header-footer \
    --print-to-pdf-no-header \
    --run-all-compositor-stages-before-draw \
    --virtual-time-budget=5000 \
    --print-to-pdf="${PDF_OUT}" \
    "file://${HTML_OUT}"

echo "wrote ${PDF_OUT}"
if command -v pdfinfo >/dev/null 2>&1; then
    pdfinfo "${PDF_OUT}" | grep -E "^Pages|^Page size"
fi
