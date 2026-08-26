#!/bin/bash
# Deletes checkpoints/ and run_default/broadcasts/ from each experiment in outputs/.
# Runs dry-run by default; pass --delete to actually remove.

set -euo pipefail

OUTPUTS_DIR="outputs"
DRY_RUN=true

for arg in "$@"; do
    case "$arg" in
        --delete) DRY_RUN=false ;;
        --*) echo "Unknown flag: $arg"; exit 1 ;;
        *) OUTPUTS_DIR="$arg" ;;
    esac
done

for exp in "$OUTPUTS_DIR"/*/; do
    [[ -d "$exp" ]] || continue
    for target in \
        "${exp}checkpoints" \
        "${exp}run_default/broadcasts"; do
        [[ -d "$target" ]] || continue
        if $DRY_RUN; then
            echo "[dry-run] rm -rf $target"
        else
            echo "Removing $target"
            rm -rf "$target"
        fi
    done
done

$DRY_RUN && echo $'\nDry run — pass --delete to actually remove.'
