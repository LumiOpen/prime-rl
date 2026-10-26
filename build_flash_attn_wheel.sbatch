#!/bin/bash
#SBATCH --job-name=build-flash-attn-wheel
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Without --hint=multithread, --cpus-per-task counts whole cores and only 56 of the
# node's 64 are user-allocatable, so anything above 56 is rejected outright
# ("Requested node configuration is not available"). With it, the unit is hardware
# threads and the full CPUEfctv=112 is available.
#SBATCH --cpus-per-task=112
#SBATCH --hint=multithread
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/build_flash_attn_%j.out
#SBATCH --error=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/build_flash_attn_%j.err

# =============================================================================
#   Build a flash-attn wheel for LUMI-G (MI250X / gfx90a) against the exact
#   torch the project lockfile pins.
#
#   WHY A WHEEL AND NOT `uv sync`:
#     1. Speed. The ROCm build compiles Composable Kernel FMHA instantiations,
#        and the count multiplies per GPU arch. torch's default arch list covers
#        every AMD card it supports; pinning GPU_ARCHS to gfx90a alone is by far
#        the biggest lever. Build once, install everywhere after that.
#     2. Determinism. Folded into `uv sync`, that compile reruns on every fresh
#        environment and would eat the walltime of whatever job triggered it.
#
#   Check the torch pin before changing TORCH_VERSION:
#     grep -A3 '^name = "torch"' /scratch/.../prime-rl/uv.lock
#   The wheel is valid for exactly one (torch, python, GPU arch) triple.
#
#   Runs on `dev-g`, whole node. dev-g has a hard 3 h limit and the CK compile is
#   the long pole, so this takes the full node and runs 64 parallel hipcc jobs
#   rather than the 16 it used to. dev-g has outbound internet, so the recursive
#   clone and the torch download work from the compute node.
#
#   Usage:
#     sbatch build_flash_attn_wheel.sbatch
#     FA_VERSION=v2.8.3 GPU_ARCH=gfx90a sbatch build_flash_attn_wheel.sbatch
# =============================================================================

set -euo pipefail

PROJECT="project_462001516"
BASE="/scratch/$PROJECT/users/khakala/lib/prime-rl-upstream-main"
IMG="${CONTAINER_IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

# Cloned with --recursive: the ROCm path compiles csrc/composable_kernel, which is
# a submodule and is NOT shipped in the PyPI sdist. That is why this builds from
# git rather than from `uv pip install flash-attn`.
FA_REPO="${FA_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FA_VERSION="${FA_VERSION:-v2.8.3}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+rocm7.2}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
GPU_ARCH="${GPU_ARCH:-gfx90a}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
WHEEL_DIR="${WHEEL_DIR:-/scratch/$PROJECT/users/khakala/wheels}"

# Budget ~4-8 GB of RAM per concurrent hipcc; CK translation units are large. 64
# jobs therefore wants ~450 GB and the node has 491. That memory ceiling — not the
# allocation's 112 threads — is what caps JOBS, so it is pinned rather than read
# from SLURM_CPUS_PER_TASK. Raise it and the node OOMs mid-compile, at which point
# ninja reports a bare "killed".
JOBS="${MAX_JOBS:-64}"

# Node-local: a HIP build writes thousands of object files and Lustre is poor at that.
BUILD_ROOT="/tmp/fa_build_${SLURM_JOB_ID}"

[ -f "$IMG" ] || { echo "ERROR: container not found: $IMG"; exit 1; }
mkdir -p "$BUILD_ROOT" "$WHEEL_DIR" "$BASE/cache/uv" "$BASE/cache/ccache"

echo "============================================"
echo "  Building flash-attn $FA_VERSION for $GPU_ARCH"
echo "============================================"
echo "Job:        $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Container:  $IMG"
echo "Torch:      $TORCH_VERSION ($TORCH_INDEX)"
echo "Python:     $PYTHON_VERSION"
echo "Build dir:  $BUILD_ROOT"
echo "Wheel out:  $WHEEL_DIR"
echo "Jobs:       $JOBS"
echo "============================================"

# ── Values baked in at submit time; the build script below sources this ──
cat > "$BUILD_ROOT/env.sh" << ENVEOF
export FA_REPO="$FA_REPO"
export FA_VERSION="$FA_VERSION"
export TORCH_VERSION="$TORCH_VERSION"
export TORCH_INDEX="$TORCH_INDEX"
export GPU_ARCH="$GPU_ARCH"
export PYTHON_VERSION="$PYTHON_VERSION"
export BUILD_ROOT="$BUILD_ROOT"
export WHEEL_DIR="$WHEEL_DIR"
export JOBS="$JOBS"
export PATH="\$HOME/.local/bin:\$PATH"

# The container bakes its own site-packages onto PYTHONPATH; left set, its torch
# shadows the venv's and the wheel links against the wrong libtorch.
export PYTHONPATH=
export PYTHONNOUSERSITE=1

# LUMI home has a tight quota (a single torch wheel overruns it), so every cache
# goes to project scratch. ccache persists deliberately: it makes rebuilds cheap.
export UV_CACHE_DIR="$BASE/cache/uv"
export CCACHE_DIR="$BASE/cache/ccache"
ENVEOF

cat > "$BUILD_ROOT/build.sh" << 'BUILDEOF'
set -euo pipefail
source "$(dirname "$0")/env.sh"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH (expected in ~/.local/bin)"; exit 1; }

SRC="$BUILD_ROOT/flash-attention"
VENV="$BUILD_ROOT/venv"

echo "--- cloning flash-attn $FA_VERSION (with submodules) ---"
git clone --depth 1 --branch "$FA_VERSION" --recursive "$FA_REPO" "$SRC"
cd "$SRC"
[ -d csrc/composable_kernel ] && [ -n "$(ls -A csrc/composable_kernel)" ] || {
    echo "ERROR: csrc/composable_kernel is empty — the submodule did not clone."
    echo "       The ROCm backend cannot build without it."
    exit 1
}

echo "--- build environment ---"
uv venv --python "$PYTHON_VERSION" "$VENV"
source "$VENV/bin/activate"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
# --no-build-isolation means setup.py runs against exactly these, nothing else.
uv pip install setuptools wheel packaging ninja psutil einops

echo "--- compiling (expect 1-3 h for one arch at 64 jobs; ccache makes reruns far shorter) ---"
# setup.py defaults GPU_ARCHS to "native". Even on a GPU node that silently widens
# to the full torch arch list on some paths, multiplying the CK instantiation
# count. Pin it — this is the single biggest lever on build time.
export GPU_ARCHS="$GPU_ARCH"
export PYTORCH_ROCM_ARCH="$GPU_ARCH"
export MAX_JOBS="$JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
# Without this, setup.py tries to fetch a prebuilt wheel from GitHub releases
# first; there are no ROCm builds published there, so it wastes time and can
# silently produce a CUDA wheel if the matcher is ever loosened.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# DO NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE to skip the compile. It
# installs instantly and the forward pass looks fine, but job 45413 proved the
# Triton backward fails the sliding-window-attention gradient test on ROCm. Every
# result produced with it is suspect. The CK compile is the price of a correct
# backward pass.
if [ "${FLASH_ATTENTION_TRITON_AMD_ENABLE:-}" = "TRUE" ]; then
    echo "ERROR: Triton backend requested. Its backward pass is known-broken on ROCm."
    exit 1
fi

uv build --wheel --no-build-isolation -o "$BUILD_ROOT/wheel" .

WHEEL=$(ls "$BUILD_ROOT"/wheel/flash_attn-*.whl)
echo "--- built $(basename "$WHEEL") ---"

echo "--- verifying against a clean venv ---"
# An ABI mismatch against libtorch shows up on import as an undefined symbol,
# never at install time.
uv venv --python "$PYTHON_VERSION" "$BUILD_ROOT/check"
source "$BUILD_ROOT/check/bin/activate"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
uv pip install einops "$WHEEL"
if python -c "import flash_attn, torch; print('import ok:', flash_attn.__version__, torch.__version__)"; then
    VERIFIED=yes
else
    # On dev-g the GPUs are held and the HIP runtime libs are present, so this is
    # a real failure. Read the traceback before re-pinning the wheel.
    echo "NOTE: import check FAILED on a GPU node. Do not trust this wheel."
    VERIFIED=no
fi

cp "$WHEEL" "$WHEEL_DIR/"
echo "--- installed to $WHEEL_DIR/$(basename "$WHEEL") (import verified: $VERIFIED) ---"
echo
echo "Point pyproject.toml at it:"
echo "  [tool.uv.sources]"
echo "  flash-attn = { path = \"$WHEEL_DIR/$(basename "$WHEEL")\" }"
echo
echo "Then: uv lock && uv sync --all-extras"
BUILDEOF

singularity exec --cleanenv \
    -B /pfs,/scratch,/projappl,/project,/flash,/appl \
    -B "$HOME:$HOME:ro" -B /tmp:/tmp:rw \
    "$IMG" bash "$BUILD_ROOT/build.sh"

rm -rf "$BUILD_ROOT"
echo "Done!"
