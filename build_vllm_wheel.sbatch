#!/bin/bash
#SBATCH --job-name=build-vllm-wheel
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
#SBATCH --output=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/build_vllm_%j.out
#SBATCH --error=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/build_vllm_%j.err

# =============================================================================
#   Build a vLLM wheel for LUMI-G (MI250X / gfx90a), linked against the exact
#   torch the project lockfile pins.
#
#   vLLM's HIP extensions link against libtorch, so the wheel is only valid for
#   one (torch version, python version, GPU arch) triple. Check the lock before
#   changing TORCH_VERSION:
#     grep -A3 '^name = "torch"' pyproject-dir/uv.lock
#
#   Runs on `dev-g`, whole node: 64 parallel hipcc jobs are what fits the build
#   inside dev-g's hard 3 h limit. dev-g has outbound internet, so the clone and
#   the torch download work from the compute node.
#   Holding the 8 GPUs also makes the post-build import check meaningful — vLLM
#   probes the platform at import and cannot be verified on a CPU-only node.
#
#   Usage:
#     sbatch build_vllm_wheel.sbatch
#     VLLM_VERSION=v0.26.0 TORCH_VERSION=2.11.0+rocm7.2 sbatch build_vllm_wheel.sbatch
# =============================================================================

set -euo pipefail

PROJECT="project_462001516"
BASE="/scratch/$PROJECT/users/khakala/lib/prime-rl-upstream-main"
IMG="${CONTAINER_IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

VLLM_VERSION="${VLLM_VERSION:-v0.26.0}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+rocm7.2}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
ROCM_ARCH="${ROCM_ARCH:-gfx90a}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
WHEEL_DIR="${WHEEL_DIR:-/scratch/$PROJECT/users/khakala/wheels}"
# Pinned, not derived from SLURM_CPUS_PER_TASK: the allocation is the whole node
# (112 threads), but 112 concurrent hipcc processes would exhaust the node's 491 GB.
JOBS="${MAX_JOBS:-64}"

# Node-local: a HIP build writes thousands of object files and Lustre is poor at that.
BUILD_ROOT="/tmp/vllm_build_${SLURM_JOB_ID}"

[ -f "$IMG" ] || { echo "ERROR: container not found: $IMG"; exit 1; }
mkdir -p "$BUILD_ROOT" "$WHEEL_DIR" "$BASE/cache/uv" "$BASE/cache/ccache"

echo "============================================"
echo "  Building vLLM $VLLM_VERSION for $ROCM_ARCH"
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
export VLLM_VERSION="$VLLM_VERSION"
export TORCH_VERSION="$TORCH_VERSION"
export TORCH_INDEX="$TORCH_INDEX"
export ROCM_ARCH="$ROCM_ARCH"
export PYTHON_VERSION="$PYTHON_VERSION"
export BUILD_ROOT="$BUILD_ROOT"
export WHEEL_DIR="$WHEEL_DIR"
export JOBS="$JOBS"
export PATH="$HOME/.local/bin:\$PATH"

# The container bakes its own site-packages onto PYTHONPATH; left set, its torch
# shadows the venv's and the wheel links against the wrong libtorch.
export PYTHONPATH=
export PYTHONNOUSERSITE=1

# LUMI home has a tight quota (a single torch wheel overruns it), so every cache
# goes to project scratch. ccache persists deliberately: it makes rebuilds cheap.
export UV_CACHE_DIR="$BASE/cache/uv"
export CCACHE_DIR="$BASE/cache/ccache"
export RUSTUP_HOME="$BASE/cache/rustup"
export CARGO_HOME="$BASE/cache/cargo"
ENVEOF

cat > "$BUILD_ROOT/build.sh" << 'BUILDEOF'
set -euo pipefail
source "$(dirname "$0")/env.sh"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH (expected in ~/.local/bin)"; exit 1; }

SRC="$BUILD_ROOT/vllm"
VENV="$BUILD_ROOT/venv"

echo "--- cloning vLLM $VLLM_VERSION ---"
git clone --depth 1 --branch "$VLLM_VERSION" https://github.com/vllm-project/vllm.git "$SRC"
cd "$SRC"

echo "--- build environment ---"
uv venv --python "$PYTHON_VERSION" "$VENV"
source "$VENV/bin/activate"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
# vLLM omits setuptools_rust from build-system.requires, which only bites with
# --no-build-isolation; the rest is requirements/build.txt minus torch.
uv pip install cmake ninja packaging setuptools setuptools-scm wheel jinja2 regex numpy setuptools_rust

# vLLM has a Rust component; the container ships no cargo.
if ! command -v cargo >/dev/null; then
    echo "--- installing rust toolchain ---"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
fi
export PATH="$CARGO_HOME/bin:$PATH"
rustup show   # honours rust-toolchain.toml rather than pulling stable first

echo "--- pinning build to the installed torch ---"
# Drops torch from vLLM's requirements: stops the build pulling its own copy, and
# strips the torch== pin from the wheel metadata so uv can install the locked one.
python use_existing_torch.py

echo "--- compiling (30-60 min at 64 jobs) ---"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH="$ROCM_ARCH"
export MAX_JOBS="$JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
uv build --wheel --no-build-isolation -o "$BUILD_ROOT/wheel" .

WHEEL=$(ls "$BUILD_ROOT"/wheel/vllm-*.whl)
echo "--- built $(basename "$WHEEL") ---"

echo "--- verifying against a clean venv ---"
# An ABI mismatch shows up on import as an undefined symbol, never at install time.
uv venv --python "$PYTHON_VERSION" "$BUILD_ROOT/check"
source "$BUILD_ROOT/check/bin/activate"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
uv pip install "$WHEEL"
if python -c "import vllm, torch; print('import ok:', vllm.__version__, torch.__version__)"; then
    VERIFIED=yes
else
    # On dev-g the GPUs are held, so this is a real failure, not a platform-probe
    # artifact. Read the traceback before re-pinning the wheel in pyproject.toml.
    echo "NOTE: import check FAILED on a GPU node. Do not trust this wheel."
    VERIFIED=no
fi

cp "$WHEEL" "$WHEEL_DIR/"
echo "--- installed to $WHEEL_DIR/$(basename "$WHEEL") (import verified: $VERIFIED) ---"
echo
echo "Point pyproject.toml at it:"
echo "  vllm = { path = \"$WHEEL_DIR/$(basename "$WHEEL")\" }"
BUILDEOF

singularity exec --cleanenv \
    -B /pfs,/scratch,/projappl,/project,/flash,/appl \
    -B "$HOME:$HOME:ro" -B /tmp:/tmp:rw \
    "$IMG" bash "$BUILD_ROOT/build.sh"

rm -rf "$BUILD_ROOT"
echo "Done!"
