#!/bin/bash
#SBATCH --job-name=dolci-rl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci_%j.out
#SBATCH --error=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci_%j.err

# Usage:
#   sbatch run.sbatch path/to/config.toml
#   sbatch run.sbatch configs/dolci_if/rl_olmo2_7b_paper.toml
#
# The config path can be absolute or relative to PRIME_RL_DIR.

set -euo pipefail

PRIME_RL_DIR="/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl"

# Accept config as positional arg or PRIME_RL_CONFIG env var
CONFIG_ARG="${1:-${PRIME_RL_CONFIG:-}}"
if [ -z "$CONFIG_ARG" ]; then
    echo "ERROR: No config specified. Pass it as a positional argument or set PRIME_RL_CONFIG." >&2
    echo "Usage: sbatch run.sbatch path/to/config.toml" >&2
    exit 1
fi

# Resolve to absolute path
if [[ "$CONFIG_ARG" = /* ]]; then
    CONFIG="$CONFIG_ARG"
else
    CONFIG="$PRIME_RL_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

# Derive a short run name from the config filename for output dir and job name
CONFIG_STEM="$(basename "$CONFIG" .toml)"
OUTPUT_DIR="$PRIME_RL_DIR/outputs/${CONFIG_STEM}_${SLURM_JOB_ID}"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif}"

mkdir -p "$PRIME_RL_DIR/slurm_logs" "$OUTPUT_DIR"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "NOTICE: WANDB_API_KEY not set — wandb will run in offline mode."
    export WANDB_MODE=offline
fi

echo "============================================"
echo "  Dolci IF-RLVR"
echo "============================================"
echo "Job:    $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Image:  $IMG"
echo "============================================"

ENV_SETUP="$PRIME_RL_DIR/slurm_logs/env_setup_${SLURM_JOB_ID}.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR

export PYTHONUSERBASE=$PRIME_RL_DIR/pythonuserbase
export HF_CACHE=/shared_silo/scratch/mmelnik/cache/hf_cache
export HF_HOME=/shared_silo/scratch/mmelnik/cache/hf_home
export TRITON_CACHE_DIR=/shared_silo/scratch/mmelnik/cache/triton_cache
export XDG_CACHE_HOME=/shared_silo/scratch/mmelnik/cache/xdg_cache
export TORCHINDUCTOR_CACHE_DIR=/shared_silo/scratch/mmelnik/cache/torchinductor_cache
export UV_CACHE_DIR=$PRIME_RL_DIR/uv_cache
export PATH=\$PYTHONUSERBASE/bin:\$PATH
export HOME=/tmp/home_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/shared_silo/scratch/mmelnik/cache/vllm_cache
export PYTHONPATH=$PRIME_RL_DIR/envs/if_eval_env:$PRIME_RL_DIR/envs/rubric_judge_env:\${PYTHONPATH:-}
mkdir -p \$HOME /shared_silo/scratch/mmelnik/cache/vllm_cache

export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export RCCL_NET_GDR_LEVEL=2

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_MOE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export PYTHONUNBUFFERED=1
ENVEOF

SINGULARITY_ENV_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    SINGULARITY_ENV_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
fi
if [ -n "${WANDB_MODE:-}" ]; then
    SINGULARITY_ENV_ARGS+=(--env "WANDB_MODE=$WANDB_MODE")
fi

singularity exec \
    --rocm --cleanenv --containall \
    "${SINGULARITY_ENV_ARGS[@]}" \
    -B /shared_silo/scratch:/shared_silo/scratch:rw \
    -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so \
    -B /dev/infiniband:/dev/infiniband \
    -B /usr/share/libdrm:/usr/share/libdrm:ro \
    -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro \
    -B /tmp:/tmp:rw \
    "$IMG" \
    bash -c "source $ENV_SETUP && uv run --no-sync rl @ $CONFIG --output-dir $OUTPUT_DIR"

echo "Job $SLURM_JOB_ID finished."
