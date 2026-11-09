#!/bin/bash
# Create symlinks mirroring HCP session structure under PROJECT_DIR/inputs/hcp/
#
# Source: /public/home2/VNLab/Datasets/Cneuromod/mario/derivatives/hcp/sessions/
# Target: PROJECT_DIR/inputs/hcp/
#
# Usage: bash create_hcp_symlinks.sh /path/to/PROJECT_DIR

set -eu

HCP_SOURCE="/public/home2/VNLab/Datasets/Cneuromod/mario/derivatives/hcp/sessions"
TARGET_BASE="${1}/inputs/hcp"

if [ -z "${1:-}" ]; then
    echo "Usage: bash create_hcp_symlinks.sh /path/to/PROJECT_DIR"
    exit 1
fi

if [ ! -d "$HCP_SOURCE" ]; then
    echo "ERROR: HCP source directory not found: $HCP_SOURCE"
    exit 1
fi

echo "Source: $HCP_SOURCE"
echo "Target: $TARGET_BASE"
echo ""

n_linked=0
n_skipped=0

for session_dir in "$HCP_SOURCE"/[0-9][0-9]_[0-9][0-9][0-9]; do
    session_name=$(basename "$session_dir")
    hcp_inner="$session_dir/hcp/$session_name/MNINonLinear/Results"

    if [ ! -d "$hcp_inner" ]; then
        echo "  SKIP $session_name: MNINonLinear/Results not found"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    for run_dir in "$hcp_inner"/tfMRI_MRO_RUN*_PA; do
        [ -d "$run_dir" ] || continue
        run_name=$(basename "$run_dir")

        link_path="$TARGET_BASE/$session_name/hcp/$session_name/MNINonLinear/Results/$run_name"
        link_parent=$(dirname "$link_path")

        mkdir -p "$link_parent"

        # Overwrite existing symlink
        ln -sfn "$run_dir" "$link_path"
        n_linked=$((n_linked + 1))
    done
done

echo "Done."
echo "  Symlinks created/updated: $n_linked"
echo "  Sessions skipped: $n_skipped"
