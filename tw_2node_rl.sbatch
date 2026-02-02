#!/bin/bash
#SBATCH --job-name=prime-rl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/rl-%j.out
#SBATCH --error=logs/rl-%j.err

# ============================================
#   2-Node Prime-RL Training Template for TW
#   Node 0: Training (8 GPUs)
#   Node 1: Inference/vLLM (8 GPUs, DP=8)
# ============================================

set -euo pipefail

# =====================
# CONFIGURATION - Edit these
# =====================
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
RL_CONFIG="${RL_CONFIG:-configs/wordle/rl.toml}"
RUN_NAME="${RUN_NAME:-rl-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-prime-rl}"

# Container and paths
IMG="${IMG:-/shared_silo/scratch/containers/rocm_vllm-dev_nightly_0624_rc2_0624_rc2_20250620.sif}"
SCRATCH_DIR="${SCRATCH_DIR:-/shared_silo/scratch/$USER}"

# =====================
# Node setup
# =====================
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_NODE=${NODELIST[0]}
VLLM_NODE=${NODELIST[1]}
VLLM_IP=$(srun --nodes=1 --ntasks=1 -w $VLLM_NODE ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')
TOTAL_DP=8  # 1 inference node x 8 GPUs

echo "============================================"
echo "  Prime-RL 2-Node Training"
echo "============================================"
echo "Training node:  $MASTER_NODE"
echo "Inference node: $VLLM_NODE ($VLLM_IP)"
echo "Model:          $MODEL"
echo "Config:         $RL_CONFIG"
echo "Output:         $OUTPUT_DIR"
echo "============================================"

mkdir -p $OUTPUT_DIR logs

# =====================
# Inference config
# =====================
cat > $OUTPUT_DIR/infer.toml << EOF
[model]
name = "$MODEL"
[parallel]
tp = 1
EOF

# =====================
# Environment setup
# =====================
cat > $OUTPUT_DIR/env_setup.sh << 'ENVEOF'
export PYTHONUSERBASE=${SCRATCH_DIR}/lib/prime-rl/pythonuserbase
export HF_CACHE=${SCRATCH_DIR}/lib/prime-rl/hf_cache
export HF_HOME=${SCRATCH_DIR}/lib/prime-rl/hf_home
export UV_CACHE_DIR=${SCRATCH_DIR}/lib/prime-rl/uv_cache
export PATH=$PYTHONUSERBASE/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_$(hostname)_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_cache_$(hostname)_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME
export RCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export WANDB_MODE=online
ENVEOF

# Substitute SCRATCH_DIR in env script
sed -i "s|\${SCRATCH_DIR}|$SCRATCH_DIR|g" $OUTPUT_DIR/env_setup.sh
sed -i "s|\${SLURM_JOB_ID}|$SLURM_JOB_ID|g" $OUTPUT_DIR/env_setup.sh

# =====================
# Bind mounts
# =====================
BIND_MOUNTS="-B $SCRATCH_DIR:$SCRATCH_DIR:rw"
BIND_MOUNTS="$BIND_MOUNTS -B /shared_silo/scratch/models:/shared_silo/scratch/models:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

# =====================
# Start inference server
# =====================
echo "Starting inference on $VLLM_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE --job-name=inference --exclusive \
    singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
    bash -c "source $OUTPUT_DIR/env_setup.sh && \
             uv run inference @ $OUTPUT_DIR/infer.toml \
               --data-parallel-size $TOTAL_DP \
               --tensor-parallel-size 1 \
               --data-parallel-size-local 8 \
               --data-parallel-address $VLLM_IP \
               --data-parallel-rpc-port 29500 \
               --gpu-memory-utilization 0.7" &

echo "Waiting 180s for inference service..."
sleep 180

# =====================
# Start training
# =====================
INFERENCE_URL="http://${VLLM_NODE}:8000/v1"

echo "Starting training on $MASTER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --job-name=rl-training \
    singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
    bash -c "source $OUTPUT_DIR/env_setup.sh && \
             uv run rl @ $RL_CONFIG \
               --model.name $MODEL \
               --wandb.project $WANDB_PROJECT \
               --wandb.name $RUN_NAME \
               --output-dir $OUTPUT_DIR \
               --orchestrator.client.base-url '[\"$INFERENCE_URL\"]'"

echo "Training complete!"
