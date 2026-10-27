#!/usr/bin/env bash
# Copy the amdsmi Python bindings out of the LUMI container into ./amd_smi/.
#
# pyproject.toml pins `amdsmi = { path = "amd_smi" }` because AMD publishes no amdsmi
# wheel. The bindings are ctypes wrappers over the container's libamd_smi.so, so they
# must come from the *same* container you run in — a copy from an older image fails at
# import with `undefined symbol: amdsmi_get_gpu_enumeration_info`. Hence a setup step
# rather than a directory vendored into git.
#
# Run it from the clone root, either on the login node or inside the container:
#   ./scripts/setup_amdsmi.sh
# Re-run it whenever the container image changes.

set -euo pipefail

IMG="${CONTAINER_IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/amd_smi"
SRC=/opt/rocm/share/amd_smi

# The login nodes carry their own, older ROCm at the very same prefix (6.3.4 as of this
# writing, against the container's 7.2.4) — so the presence of $SRC proves nothing. Only
# the runtime's own marker variable says whether we are inside the image.
if [ -n "${SINGULARITY_CONTAINER:-${APPTAINER_CONTAINER:-}}" ]; then
    copy() { cp -r "$SRC/." "$DEST/"; }
    version=$(cat /opt/rocm/.info/version)
else
    [ -f "$IMG" ] || { echo "ERROR: container not found: $IMG"; exit 1; }
    copy() { singularity exec -B "$DEST:$DEST:rw" "$IMG" cp -r "$SRC/." "$DEST/"; }
    version=$(singularity exec "$IMG" cat /opt/rocm/.info/version)
fi

rm -rf "$DEST"
mkdir -p "$DEST"
copy
# The container ships a build/ and an egg-info/ from AMD's own install; both are stale
# relative to a fresh `uv sync` and only confuse it.
rm -rf "$DEST/build" "$DEST/amdsmi.egg-info"

echo "amdsmi bindings for ROCm $version -> $DEST"
