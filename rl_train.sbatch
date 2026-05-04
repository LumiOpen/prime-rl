#!/bin/bash
#SBATCH --job-name=prime-rl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/shared_silo/scratch/prime-rl/logs/rl_%j.out
#SBATCH --error=/shared_silo/scratch/prime-rl/logs/rl_%j.err

# =============================================================================
#   PRIME-RL launcher for TensorWave MI325X cluster
#
#   Usage:
#     sbatch rl_train.sbatch <run_name> <model_path> <dataset_dir>
#
#   Prerequisites:
#     - Run setup.sh once on a compute node
#     - Export WANDB_API_KEY before submitting (or set WANDB_MODE=disabled)
#
#   Example (Poro2 with klear math frontier):
#     sbatch rl_train.sbatch my-run-v1 \
#       /shared_silo/scratch/prime-rl/models/Poro2-Longrope-132en111fi \
#       /shared_silo/scratch/prime-rl/data/klear_math_30k_en_frontier_025_075_16s
# =============================================================================

set -euo pipefail

RUN_NAME=${1:?Usage: sbatch rl_train.sbatch <run_name> <model_path> <dataset_dir>}
SFT_MODEL=${2:?Usage: sbatch rl_train.sbatch <run_name> <model_path> <dataset_dir>}
DATASET_DIR=${3:?Usage: sbatch rl_train.sbatch <run_name> <model_path> <dataset_dir>}

# ── Shared paths ──
BASE="/shared_silo/scratch/prime-rl"
PRIME_RL_DIR="$BASE/repo"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/rocm_vllm-dev_upstream_preview_releases_v0.17.0_20260303.sif}"

# ── Validate ──
[ -d "$SFT_MODEL" ] || { echo "ERROR: model path does not exist: $SFT_MODEL"; exit 1; }
[ -d "$DATASET_DIR" ] || { echo "ERROR: dataset dir does not exist: $DATASET_DIR"; exit 1; }
[ -d "$PRIME_RL_DIR" ] || { echo "ERROR: prime-rl repo not found at $PRIME_RL_DIR. Run setup.sh first."; exit 1; }
[ -f "$PRIME_RL_DIR/.venv/bin/activate" ] || { echo "ERROR: .venv not found. Run setup.sh first."; exit 1; }
WANDB_OFFLINE=false
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then
    echo "NOTICE: WANDB_API_KEY not set — wandb will run in offline mode."
    export WANDB_MODE=disabled
    WANDB_OFFLINE=true
fi

JSONL_COUNT=$(find -L "$DATASET_DIR" -name '*.jsonl' | wc -l)
[ "$JSONL_COUNT" -ge 1 ] || { echo "ERROR: no .jsonl files in $DATASET_DIR"; exit 1; }
TOTAL_QUESTIONS=$(cat "$DATASET_DIR"/*.jsonl | wc -l)
echo "Dataset: $TOTAL_QUESTIONS questions from $DATASET_DIR"

# ── Training hyperparameters (override via env vars before sbatch) ──
TRAINER_SEQ_LEN=${TRAINER_SEQ_LEN:-24576}
INF_SEQ_LEN=${INF_SEQ_LEN:-24576}
MAX_TOKENS=${MAX_TOKENS:-24576}
BATCH_SIZE=${BATCH_SIZE:-128}
ROLLOUTS=${ROLLOUTS:-8}
LR=${LR:-1e-6}
KL_TAU=${KL_TAU:-0.01}
IPO_MASK_LOW=${IPO_MASK_LOW:-0.2}
IPO_MASK_HIGH=${IPO_MASK_HIGH:-0.2}
ADV_TAU=${ADV_TAU:-2.0}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
MAX_STEPS=${MAX_STEPS:-500}
CKPT_INTERVAL=${CKPT_INTERVAL:-10}
WANDB_PROJECT=${WANDB_PROJECT:-poro-language-rl}

# ── Cluster layout ──
NUM_INF_NODES=${NUM_INF_NODES:-3}
TOTAL_NODES=${TOTAL_NODES:-4}
TOTAL_DP=${TOTAL_DP:-24}

[ "$SLURM_NNODES" -eq "$TOTAL_NODES" ] || { echo "ERROR: need $TOTAL_NODES nodes, got $SLURM_NNODES"; exit 1; }

# ── Per-job model directory (avoid cache race conditions) ──
JOB_MODEL_DIR="$BASE/models/job_${SLURM_JOB_ID}_model"
mkdir -p "$JOB_MODEL_DIR"
for f in "$SFT_MODEL"/*; do
    [ -f "$f" ] && ln -sf "$f" "$JOB_MODEL_DIR/$(basename "$f")"
done
SFT_MODEL="$JOB_MODEL_DIR"

cd "$PRIME_RL_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_RUN_NAME="${RUN_NAME}-${TIMESTAMP}"
OUTPUT_DIR="$BASE/outputs/$FULL_RUN_NAME"
mkdir -p "$OUTPUT_DIR/run_default"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_NODE=${NODELIST[0]}
FIRST_INF_NODE=${NODELIST[1]}
FIRST_INF_IP=$(srun --nodes=1 --ntasks=1 -w "$FIRST_INF_NODE" /usr/sbin/ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')

echo "============================================"
echo "  PRIME-RL Training: $FULL_RUN_NAME"
echo "============================================"
echo "User:         $USER"
echo "Job:          $SLURM_JOB_ID"
echo "Nodes:        $MASTER_NODE (trainer) | ${NODELIST[@]:1} (inference)"
echo "Inference IP: $FIRST_INF_IP"
echo "Model:        $SFT_MODEL"
echo "Dataset:      $DATASET_DIR ($TOTAL_QUESTIONS problems)"
echo "Seq len:      $INF_SEQ_LEN"
echo "Temp:         $ROLLOUT_TEMP"
echo "Rollouts:     $ROLLOUTS"
echo "LR:           $LR"
echo "KL tau:       $KL_TAU"
echo "IPO mask:     $IPO_MASK_LOW / $IPO_MASK_HIGH"
echo "Adv tau:      $ADV_TAU"
echo "ODF:          ON"
echo "Steps:        $MAX_STEPS (ckpt every $CKPT_INTERVAL)"
echo "Container:    $IMG"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ── Generate configs ──
cat > "$OUTPUT_DIR/infer_sft.toml" << TOMLEOF
[model]
name = "$SFT_MODEL"
max_model_len = $INF_SEQ_LEN

[parallel]
tp = 1

[server]
port = 8000
TOMLEOF

cat > "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
batch_size = $BATCH_SIZE
rollouts_per_example = $ROLLOUTS
max_steps = $MAX_STEPS
seq_len = $INF_SEQ_LEN
num_train_workers = 8
max_async_level = 1

[model]
name = "$SFT_MODEL"

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-orch"
offline = $WANDB_OFFLINE

[sampling]
temperature = $ROLLOUT_TEMP
max_tokens = $MAX_TOKENS

[buffer]
online_difficulty_filtering = true

[advantage]
type = "default"

[[env]]
id = "math-env"
name = "english-math"
[env.args]
dataset_name = "$DATASET_DIR"
dataset_subset = "default"
question_key = "problem"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60
TOMLEOF

cat > "$OUTPUT_DIR/train_config.toml" << TOMLEOF
max_steps = $MAX_STEPS
matmul_precision = "highest"

[model]
name = "$SFT_MODEL"
seq_len = $TRAINER_SEQ_LEN

[model.ac]
freq = 1

[tokenizer]
name = "$SFT_MODEL"

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-trainer"
offline = $WANDB_OFFLINE

[ckpt]
interval = $CKPT_INTERVAL
keep_last = 30

[loss]
type = "default"
kl_tau = $KL_TAU
ipo_mask_low = $IPO_MASK_LOW
ipo_mask_high = $IPO_MASK_HIGH
adv_tau = $ADV_TAU

[optim]
lr = $LR
max_grad_norm = 1.0

[scheduler]
type = "constant"
TOMLEOF

# ── Environment setup ──
ENV_SETUP="$OUTPUT_DIR/env_setup.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR
source $PRIME_RL_DIR/.venv/bin/activate
export TRITON_CACHE_DIR=/tmp/triton_\$(hostname)_\${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_\$(hostname)_\${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_\$(hostname)_\${SLURM_JOB_ID}
export HOME=/tmp/home_\$(hostname)_\${SLURM_JOB_ID}
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$XDG_CACHE_HOME \$HOME
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export RCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export PYTHONUNBUFFERED=1
ENVEOF
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "export WANDB_API_KEY=$WANDB_API_KEY" >> "$ENV_SETUP"
fi
if [ -n "${WANDB_MODE:-}" ]; then
    echo "export WANDB_MODE=$WANDB_MODE" >> "$ENV_SETUP"
elif [ -z "${WANDB_API_KEY:-}" ]; then
    echo "export WANDB_MODE=disabled" >> "$ENV_SETUP"
fi

# ── Container bind mounts ──
CONTAINER_FLAGS="--rocm --cleanenv --containall"
BIND_MOUNTS="-B /shared_silo/scratch:/shared_silo/scratch:rw"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

# ── Convert to safetensors if needed ──
if [ ! -f "$SFT_MODEL/model.safetensors.index.json" ] && [ ! -f "$SFT_MODEL/model.safetensors" ]; then
    echo "Converting model to safetensors..."
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && python -c \"
from transformers import AutoModelForCausalLM; import torch
model = AutoModelForCausalLM.from_pretrained('$SFT_MODEL', torch_dtype=torch.bfloat16)
model.save_pretrained('$SFT_MODEL'); print('Model saved as safetensors')
\""
fi

# Clean stale prime/ cache
if [ -d "$SFT_MODEL/prime" ] && [ -z "$(ls -A "$SFT_MODEL/prime"/*.safetensors 2>/dev/null)" ]; then
    rm -rf "$SFT_MODEL/prime"
fi

# ── Launch inference servers ──
echo "Starting inference on ${NUM_INF_NODES} node(s)..."
srun --nodes=1 --ntasks=1 --nodelist="$FIRST_INF_NODE" --job-name=inference-master --exclusive \
    /usr/bin/singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && \
        python -m prime_rl.inference.server \
        @ $OUTPUT_DIR/infer_sft.toml \
        --parallel.dp $TOTAL_DP \
        --data-parallel-size-local 8 \
        --data-parallel-rpc-port 29500 \
        --gpu-memory-utilization 0.7 \
        --vllm-extra '{\"data_parallel_address\": \"$FIRST_INF_IP\"}'" &

for i in $(seq 2 $NUM_INF_NODES); do
    INF_NODE=${NODELIST[$i]}
    START_RANK=$(((i - 1) * 8))
    echo "  Worker on $INF_NODE (start-rank $START_RANK)"
    sleep 5
    srun --nodes=1 --ntasks=1 --nodelist="$INF_NODE" --job-name=inference-worker-$i --exclusive \
        /usr/bin/singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && \
            python -m prime_rl.inference.server \
            @ $OUTPUT_DIR/infer_sft.toml \
            --parallel.dp $TOTAL_DP \
            --data-parallel-size-local 8 \
            --data-parallel-rpc-port 29500 \
            --gpu-memory-utilization 0.7 \
            --api-server-count 0 \
            --vllm-extra '{\"data_parallel_address\": \"$FIRST_INF_IP\", \"data_parallel_start_rank\": $START_RANK}'" &
done

sleep 300

# ── Launch orchestrator + trainer ──
echo "Starting orchestrator + trainer on $MASTER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" --job-name=rl-training \
    /usr/bin/singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "
        source $ENV_SETUP
        mkdir -p wandb

        python -m prime_rl.orchestrator.orchestrator \
            @ $OUTPUT_DIR/orch_config.toml \
            --client.base-url 'http://${FIRST_INF_NODE}:8000/v1' \
            --output-dir $OUTPUT_DIR/run_default &
        ORCH_PID=\$!

        echo 'Waiting for orchestrator...'
        for i in \$(seq 1 120); do
            [ -f $OUTPUT_DIR/run_default/control/orch.toml ] && break
            sleep 1
        done
        if [ ! -f $OUTPUT_DIR/run_default/control/orch.toml ]; then
            echo 'ERROR: Orchestrator config not found after 120s'
            kill \$ORCH_PID 2>/dev/null
            exit 1
        fi
        sleep 5

        python -m torch.distributed.run \
            --nproc-per-node 8 \
            --local-ranks-filter 0 \
            -m prime_rl.trainer.rl.train \
            @ $OUTPUT_DIR/train_config.toml \
            --output-dir $OUTPUT_DIR
    "

# Clean up per-job model directory
rm -rf "$JOB_MODEL_DIR"
echo "Done!"
