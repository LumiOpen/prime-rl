#!/bin/bash
#SBATCH --job-name=dolci-qwen25-0.5b
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=32
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci_qwen25_0.5b_%j.out
#SBATCH --error=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci_qwen25_0.5b_%j.err

set -euo pipefail

PRIME_RL_DIR="/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl"
CONFIG="$PRIME_RL_DIR/configs/dolci_if/rl_qwen25_0.5b_instruct.toml"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif}"
OUTPUT_DIR="$PRIME_RL_DIR/outputs/dolci-qwen25-0.5b_${SLURM_JOB_ID}"

# Scope compiled kernel caches to the container image so different builds
# never share kernels (stale cache across images causes NaN logprobs/activations).
IMG_TAG=$(basename "$IMG" .sif)
VLLM_CACHE_DIR="/shared_silo/scratch/mmelnik/cache/vllm_cache/${IMG_TAG}"
TRITON_CACHE_DIR="/shared_silo/scratch/mmelnik/cache/triton_cache/${IMG_TAG}"

mkdir -p "$PRIME_RL_DIR/slurm_logs" "$OUTPUT_DIR" "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "NOTICE: WANDB_API_KEY not set — wandb will run in offline mode."
    export WANDB_MODE=offline
fi

echo "============================================"
echo "  Dolci IF-RLVR: Qwen2.5-0.5B-Instruct"
echo "============================================"
echo "Job:    $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Image:  $IMG"
echo "GPU layout: GPU0=inference, GPU1=trainer, GPU2=RM inference"
echo "============================================"

ENV_SETUP="$PRIME_RL_DIR/slurm_logs/env_setup_${SLURM_JOB_ID}.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR

export PYTHONUSERBASE=$PRIME_RL_DIR/pythonuserbase
export HF_CACHE=/shared_silo/scratch/mmelnik/cache/hf_cache
export HF_HOME=/shared_silo/scratch/mmelnik/cache/hf_home
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
export XDG_CACHE_HOME=/shared_silo/scratch/mmelnik/cache/xdg_cache
export TORCHINDUCTOR_CACHE_DIR=/shared_silo/scratch/mmelnik/cache/torchinductor_cache
export UV_CACHE_DIR=$PRIME_RL_DIR/uv_cache
export PATH=\$PYTHONUSERBASE/bin:\$PATH
export HOME=/tmp/home_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=$VLLM_CACHE_DIR
export PYTHONPATH=$PRIME_RL_DIR/envs/if_eval_env:$PRIME_RL_DIR/envs/rubric_judge_env:\${PYTHONPATH:-}
mkdir -p \$HOME $VLLM_CACHE_DIR

export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export RCCL_NET_GDR_LEVEL=2

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
