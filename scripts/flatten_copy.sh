#!/usr/bin/env bash
#
# flatten_copy.sh -- copy nested {split}/{class}/{clip_stem}.* trees to flat
# {clip_stem}.* layouts on engelbart. Originals are untouched until the user
# manually deletes them after verify_flatten.py passes.
#
# Why two passes:
#   1) ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits/
#      contains per-clip {clip_stem}_{joints,pos,failed}.npy under
#      {train,val,test}/{Top_smash, Bottom_lob, ...}/
#   2) ShuttleSet/shuttle_npy/ contains per-clip {clip_stem}.npy under the
#      same {split}/{class}/ structure (legacy BST artifact).
#
# Clip stems ({vid}_{set}_{rally}_{ball_round}) are globally unique across
# all splits and classes, so flattening produces no filename collisions.
#
# Defaults assume the standard scratch paths on engelbart. Override with
# --src/--dst flags if needed.
#
# Usage:
#   bash flatten_copy.sh [--dry-run] [--mode cp|reflink|hardlink]
#                       [--src DIR] [--dst DIR] [--target clips|shuttle|all]
#
# Examples:
#   bash flatten_copy.sh --dry-run                 # show what would happen
#   bash flatten_copy.sh                           # default: cp --reflink=auto, both targets
#   bash flatten_copy.sh --mode hardlink           # zero-storage hardlinks (same fs only)
#   bash flatten_copy.sh --target clips            # just the per-clip dir

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults (engelbart scratch layout)
# -----------------------------------------------------------------------------
CLIPS_SRC_DEFAULT="/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits"
CLIPS_DST_DEFAULT="/scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat"
SHUTTLE_SRC_DEFAULT="/scratch/comp320a/ShuttleSet/shuttle_npy"
SHUTTLE_DST_DEFAULT="/scratch/comp320a/ShuttleSet/shuttle_npy_flat"

DRY_RUN=0
MODE="cp"
TARGET="all"
SRC=""
DST=""

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --mode)     MODE="$2"; shift 2 ;;
        --target)   TARGET="$2"; shift 2 ;;
        --src)      SRC="$2"; shift 2 ;;
        --dst)      DST="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    cp|reflink|hardlink) ;;
    *) echo "--mode must be cp|reflink|hardlink, got: $MODE" >&2; exit 1 ;;
esac

case "$TARGET" in
    clips|shuttle|all) ;;
    *) echo "--target must be clips|shuttle|all, got: $TARGET" >&2; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# Per-target flatten
# -----------------------------------------------------------------------------
flatten_one_dir() {
    local label="$1"
    local src="$2"
    local dst="$3"

    echo
    echo "=== ${label} ==="
    echo "  src: ${src}"
    echo "  dst: ${dst}"

    if [[ ! -d "$src" ]]; then
        echo "  ERROR: source dir does not exist; skipping" >&2
        return 1
    fi

    # Count source files (per-clip layout assumed: {split}/{class}/*.npy).
    local n_src
    n_src=$(find "$src" -mindepth 3 -maxdepth 3 -type f -name '*.npy' | wc -l)
    echo "  source files: ${n_src}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] would create ${dst} and copy ${n_src} files (mode=${MODE})"
        # Collect the sample into an array first so the `head -5` SIGPIPE on
        # `find` does not propagate through `set -o pipefail` and kill the
        # script before we reach the shuttle_npy section below.
        local sample_files=()
        while IFS= read -r f; do
            sample_files+=("$f")
        done < <(find "$src" -mindepth 3 -maxdepth 3 -type f -name '*.npy' 2>/dev/null | head -5)
        for f in "${sample_files[@]}"; do
            echo "    [dry-run] $f -> ${dst}/$(basename "$f")"
        done
        echo "    ... (showing first ${#sample_files[@]} only)"
        return 0
    fi

    mkdir -p "$dst"

    # Run the copy. We use bash globbing within the find -exec to avoid spawning
    # a subshell per file. Skip-if-exists keeps the script idempotent.
    case "$MODE" in
        cp)
            # Default: full copy. Safest. Use --reflink=auto so on CoW
            # filesystems (xfs, btrfs) we get O(1) copies for free.
            find "$src" -mindepth 3 -maxdepth 3 -type f -name '*.npy' -print0 \
                | xargs -0 -I{} -P 8 bash -c '
                    src_file="$1"; dst_dir="$2"
                    dst_file="${dst_dir}/$(basename "$src_file")"
                    [[ -e "$dst_file" ]] && exit 0
                    cp --reflink=auto "$src_file" "$dst_file"
                  ' _ {} "$dst"
            ;;
        reflink)
            # Strict reflink: fail if filesystem does not support CoW.
            find "$src" -mindepth 3 -maxdepth 3 -type f -name '*.npy' -print0 \
                | xargs -0 -I{} -P 8 bash -c '
                    src_file="$1"; dst_dir="$2"
                    dst_file="${dst_dir}/$(basename "$src_file")"
                    [[ -e "$dst_file" ]] && exit 0
                    cp --reflink=always "$src_file" "$dst_file"
                  ' _ {} "$dst"
            ;;
        hardlink)
            # Hardlinks: zero storage cost, same content. Source and dest must
            # be on the same filesystem. Pipeline rewrites won't touch the
            # original via the new path because pose/collation only reads.
            find "$src" -mindepth 3 -maxdepth 3 -type f -name '*.npy' -print0 \
                | xargs -0 -I{} -P 8 bash -c '
                    src_file="$1"; dst_dir="$2"
                    dst_file="${dst_dir}/$(basename "$src_file")"
                    [[ -e "$dst_file" ]] && exit 0
                    ln "$src_file" "$dst_file"
                  ' _ {} "$dst"
            ;;
    esac

    local n_dst
    n_dst=$(find "$dst" -mindepth 1 -maxdepth 1 -type f -name '*.npy' | wc -l)
    echo "  destination files: ${n_dst}"
    if [[ "$n_src" != "$n_dst" ]]; then
        echo "  WARNING: source/destination counts differ (${n_src} vs ${n_dst})." >&2
        echo "  Run verify_flatten.py to see which clips are missing." >&2
    else
        echo "  Counts match (${n_src})."
    fi
}

# -----------------------------------------------------------------------------
# Execute requested targets
# -----------------------------------------------------------------------------
echo "flatten_copy.sh: dry_run=${DRY_RUN} mode=${MODE} target=${TARGET}"

if [[ "$TARGET" == "clips" || "$TARGET" == "all" ]]; then
    src="${SRC:-$CLIPS_SRC_DEFAULT}"
    dst="${DST:-$CLIPS_DST_DEFAULT}"
    flatten_one_dir "per-clip npy (joints/pos/failed)" "$src" "$dst"
fi

if [[ "$TARGET" == "shuttle" || "$TARGET" == "all" ]]; then
    src="${SRC:-$SHUTTLE_SRC_DEFAULT}"
    dst="${DST:-$SHUTTLE_DST_DEFAULT}"
    flatten_one_dir "shuttle_npy" "$src" "$dst"
fi

echo
echo "Done. Originals untouched. Run verify_flatten.py before deleting them."
