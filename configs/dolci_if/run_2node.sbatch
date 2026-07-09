#!/bin/bash
#SBATCH --job-name=dolci-rl-2node
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci2node_%j.out
#SBATCH --error=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/slurm_logs/dolci2node_%j.err

# 2-node RLVR training (version that worked in job 36628):
#   Node 0: inference (GPU 0-3, TP=4) + RM/judge (GPU 4)
#   Node 1: trainer (8-GPU FSDP) + orchestrator
#
# Usage:
#   sbatch run_2node.sbatch configs/dolci_if/rl_qwen35_9b_instruct_qwen35_9b_2node.toml

set -euo pipefail

PRIME_RL_DIR="/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif}"

CONFIG_ARG="${1:-${PRIME_RL_CONFIG:-}}"
if [ -z "$CONFIG_ARG" ]; then
    echo "ERROR: No config specified." >&2; exit 1
fi
[[ "$CONFIG_ARG" = /* ]] && CONFIG="$CONFIG_ARG" || CONFIG="$PRIME_RL_DIR/$CONFIG_ARG"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG" >&2; exit 1
fi

CONFIG_STEM="$(basename "$CONFIG" .toml)"
OUTPUT_DIR="$PRIME_RL_DIR/outputs/${CONFIG_STEM}_${SLURM_JOB_ID}"
mkdir -p "$PRIME_RL_DIR/slurm_logs" "$OUTPUT_DIR"

HOSTNAMES=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
INFER_HOST="${HOSTNAMES[0]}"
TRAIN_HOST="${HOSTNAMES[1]}"

echo "============================================"
echo "  Dolci IF-RLVR (2-node)"
echo "============================================"
echo "Job:         $SLURM_JOB_ID"
echo "Config:      $CONFIG"
echo "Output:      $OUTPUT_DIR"
echo "Infer node:  $INFER_HOST"
echo "Train node:  $TRAIN_HOST"
echo "============================================"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

SINGULARITY_ENV_ARGS=()
[ -n "${WANDB_API_KEY:-}" ] && SINGULARITY_ENV_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
[ -n "${WANDB_MODE:-}" ]    && SINGULARITY_ENV_ARGS+=(--env "WANDB_MODE=$WANDB_MODE")

SINGULARITY_ARGS=(
    --rocm --cleanenv --containall
    "${SINGULARITY_ENV_ARGS[@]}"
    -B /shared_silo/scratch:/shared_silo/scratch:rw
    -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
    -B /dev/infiniband:/dev/infiniband
    -B /usr/share/libdrm:/usr/share/libdrm:ro
    -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro
    -B /tmp:/tmp:rw
)

WANDB_SHARED_RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex)")
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
export HOME=/tmp/home_${SLURM_JOB_ID}_\$SLURM_PROCID
export VLLM_CACHE_ROOT=/shared_silo/scratch/mmelnik/cache/vllm_cache
export NLTK_DATA=/shared_silo/scratch/mmelnik/cache/nltk_data
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

export MASTER_ADDR=$TRAIN_HOST
export MASTER_PORT=29500
export INFER_HOST=$INFER_HOST
export TRAIN_HOST=$TRAIN_HOST

export WANDB_SHARED_MODE=1
export WANDB_SHARED_RUN_ID=$WANDB_SHARED_RUN_ID
ENVEOF

# ── Step 1: Generate subconfigs (dry-run on train node) ───────────────────────
srun --ntasks=1 --nodes=1 --nodelist="$TRAIN_HOST" \
    singularity exec "${SINGULARITY_ARGS[@]}" "$IMG" \
    bash -c "source $ENV_SETUP && uv run --no-sync rl @ $CONFIG --output-dir $OUTPUT_DIR --dry-run" \
    2>&1 | grep -E "Wrote|ERROR|WARN|SUCCESS" || true

CONFIG_DIR="$OUTPUT_DIR/configs"

RM_URL="http://${INFER_HOST}:8001"
sed -i "/custom_rm = true/a rm_server_url = \"$RM_URL\"" "$CONFIG_DIR/orchestrator.toml"
sed -i "/rm_model_path = /a rm_server_url = \"$RM_URL\"" "$CONFIG_DIR/orchestrator.toml"

JUDGE_MODEL=$(grep "judge_model_path" "$CONFIG_DIR/orchestrator.toml" | head -1 | sed 's/.*= "\(.*\)".*/\1/')
[ -n "$JUDGE_MODEL" ] && \
    sed -i "s|name = \"Qwen/Qwen3-0.6B\"|name = \"$JUDGE_MODEL\"|g" "$CONFIG_DIR/rm_inference.toml"

echo "Patched configs: rm_server_url=$RM_URL, judge_model=$JUDGE_MODEL"

# ── Step 2: Launch inference node (Node 0) in background ─────────────────────
srun --ntasks=1 --nodes=1 --nodelist="$INFER_HOST" \
    singularity exec "${SINGULARITY_ARGS[@]}" "$IMG" \
    bash -c "
        source $ENV_SETUP

        # Policy inference server — override via INFER_GPUS env var (default: 0,1,2,3)
        HIP_VISIBLE_DEVICES=${INFER_GPUS:-0,1,2,3} \
            uv run --no-sync inference @ $CONFIG_DIR/inference.toml \
            --server.port 8100 \
            > $OUTPUT_DIR/inference.log 2>&1 &

        # RM/judge inference server — override via RM_GPUS env var (default: 4)
        HIP_VISIBLE_DEVICES=${RM_GPUS:-4} \
            uv run --no-sync inference @ $CONFIG_DIR/rm_inference.toml \
            --server.port 8001 \
            > $OUTPUT_DIR/rm_inference.log 2>&1 &

        wait
    " > "$OUTPUT_DIR/infer_node.log" 2>&1 &

INFER_PID=$!

# ── Step 3: Wait for inference to load ───────────────────────────────────────
echo "Waiting 180s for inference server to load model..."
sleep 180

# ── Step 4: Launch trainer + orchestrator on Node 1 ──────────────────────────
srun --ntasks=1 --nodes=1 --nodelist="$TRAIN_HOST" \
    singularity exec "${SINGULARITY_ARGS[@]}" "$IMG" \
    bash -c "
        source $ENV_SETUP

        WANDB_SHARED_LABEL=orchestrator uv run --no-sync orchestrator \
            @ $CONFIG_DIR/orchestrator.toml \
            --client.base-url 'http://${INFER_HOST}:8100/v1' \
            > $OUTPUT_DIR/orchestrator.log 2>&1 &
        ORCH_PID=\$!

        WANDB_SHARED_LABEL=trainer PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            uv run --no-sync torchrun \
            --nnodes=1 \
            --nproc-per-node=8 \
            --rdzv-endpoint=localhost:29500 \
            --rdzv-id=job_${SLURM_JOB_ID} \
            --log-dir=$OUTPUT_DIR/torchrun \
            --tee=3 --redirects=3 --local-ranks-filter=0 \
            -m prime_rl.trainer.rl.train \
            @ $CONFIG_DIR/trainer.toml \
            2>&1 | tee $OUTPUT_DIR/trainer.log

        echo 'Trainer done. Waiting 60s for WandB flush...'
        sleep 60
        kill \$ORCH_PID 2>/dev/null || true
        wait \$ORCH_PID 2>/dev/null || true
    "

kill $INFER_PID 2>/dev/null || true
wait $INFER_PID 2>/dev/null || true

echo "Job $SLURM_JOB_ID finished."
