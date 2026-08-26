#!/bin/bash
#SBATCH --job-name=prime-rl-singlenode-mixed
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=./logs/rl_%j.out
#SBATCH --error=./logs/rl_%j.err

# =============================================================================
#   Single-node mixed RL training: English IFEval + Finnish IFEval +
#   English rubric-judge + Finnish rubric-judge. Policy inference, LLM judge,
#   and trainer each occupy a configurable slice of the node's GPUs.
#
#   Usage:
#     JUDGE_MODEL=/path/to/judge \
#       sbatch rl_train_singlenode_mixed.sbatch <run_name> <model_path>
#
#   GPU layout (override via env vars):
#     NUM_INF_GPUS=5   — GPUs 0..4         → policy inference
#     NUM_RM_GPUS=2    — GPUs 5..6         → LLM judge inference
#     NUM_TRAIN_GPUS=1 — GPU 7             → trainer
#
#   Example:
#     NUM_INF_GPUS=5 NUM_RM_GPUS=2 NUM_TRAIN_GPUS=1 \
#     JUDGE_MODEL=/path/to/Qwen3.5-27B \
#       sbatch rl_train_singlenode_mixed.sbatch \
#         my-mixed-run \
#         /path/to/policy-model
# =============================================================================

set -euo pipefail

RUN_NAME=${1:?Usage: sbatch rl_train_singlenode_mixed.sbatch <run_name> <model_path>}
SFT_MODEL=${2:?Usage: sbatch rl_train_singlenode_mixed.sbatch <run_name> <model_path>}
IFEVAL_ONLY=${IFEVAL_ONLY:-0}
MATH_ONLY=${MATH_ONLY:-0}
JUDGE_ONLY=${JUDGE_ONLY:-0}
if [ "$IFEVAL_ONLY" = "1" ] || [ "$MATH_ONLY" = "1" ]; then
    JUDGE_MODEL=${JUDGE_MODEL:-}
else
    JUDGE_MODEL=${JUDGE_MODEL:?ERROR: JUDGE_MODEL env var must be set to the judge model path}
fi

# ── Shared paths ──
BASE="/shared_silo/scratch/kahakala/lib/prime-rl-upstream"
PRIME_RL_DIR="$BASE"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/primus_v26.2_moefix.sif}"

# ── Validate ──
[ -d "$SFT_MODEL" ] || { echo "ERROR: model path does not exist: $SFT_MODEL"; exit 1; }
if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    [ -d "$JUDGE_MODEL" ] || { echo "ERROR: judge model path does not exist: $JUDGE_MODEL"; exit 1; }
fi
SFT_MODEL=$(realpath "$SFT_MODEL")
if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    JUDGE_MODEL=$(realpath "$JUDGE_MODEL")
fi
[ -d "$PRIME_RL_DIR" ] || { echo "ERROR: prime-rl repo not found at $PRIME_RL_DIR"; exit 1; }
[ -f "$PRIME_RL_DIR/.venv/bin/activate" ] || { echo "ERROR: .venv not found. Run setup.sh first."; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
UV=$(command -v uv) || { echo "ERROR: uv not found in PATH"; exit 1; }
WANDB_OFFLINE=false
if [ "${WANDB_MODE:-}" = "offline" ] || [ "${WANDB_MODE:-}" = "disabled" ]; then
    WANDB_OFFLINE=true
fi

# ── GPU layout ──
NUM_INF_GPUS=${NUM_INF_GPUS:-5}
NUM_RM_GPUS=${NUM_RM_GPUS:-2}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-1}
TOTAL_GPUS=$((NUM_INF_GPUS + NUM_RM_GPUS + NUM_TRAIN_GPUS))
[ "$TOTAL_GPUS" -le 8 ] || { echo "ERROR: NUM_INF_GPUS + NUM_RM_GPUS + NUM_TRAIN_GPUS must be <= 8, got $TOTAL_GPUS"; exit 1; }

INF_GPU_LIST=$(seq -s, 0 $((NUM_INF_GPUS - 1)))
RM_GPU_LIST=$(seq -s, $NUM_INF_GPUS $((NUM_INF_GPUS + NUM_RM_GPUS - 1)))
TRAIN_GPU_LIST=$(seq -s, $((NUM_INF_GPUS + NUM_RM_GPUS)) $((NUM_INF_GPUS + NUM_RM_GPUS + NUM_TRAIN_GPUS - 1)))

# ── Training hyperparameters (override via env vars before sbatch) ──
TRAINER_SEQ_LEN=${TRAINER_SEQ_LEN:-4096}
INF_SEQ_LEN=${INF_SEQ_LEN:-4096}
RM_SEQ_LEN=${RM_SEQ_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-2048}
BATCH_SIZE=${BATCH_SIZE:-128}
ROLLOUTS=${ROLLOUTS:-8}
LR=${LR:-1e-6}
KL_TAU=${KL_TAU:-0}
IPO_MASK_LOW=${IPO_MASK_LOW:-0.2}
IPO_MASK_HIGH=${IPO_MASK_HIGH:-0.2}
ADV_TAU=${ADV_TAU:-2.0}
EASY_THRESHOLD=${EASY_THRESHOLD:-1.0}
HARD_THRESHOLD=${HARD_THRESHOLD:-0.0}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
MAX_STEPS=${MAX_STEPS:-500}
MAX_OFF_POLICY=${MAX_OFF_POLICY:-8}
CKPT_INTERVAL=${CKPT_INTERVAL:-100}
KEEP_LAST=${KEEP_LAST:-30}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning}
TRAINER_ATTN=${TRAINER_ATTN:-flash_attention_2}
TRAINER_MODEL_IMPL=${TRAINER_MODEL_IMPL:-hf}
RENDERER=${RENDERER:-qwen3.5}

PORT=8000
RM_PORT=8001
TRAINER_RDZV_PORT=29501

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

FI_JUDGE_DATASET="/shared_silo/scratch/datasets/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi.jsonl"
EN_MATH_DATASET="${EN_MATH_DATASET:-/shared_silo/scratch/dzautner/prime-rl/data/klear_math_30k_en}"
FI_MATH_DATASET="${FI_MATH_DATASET:-/shared_silo/scratch/dzautner/prime-rl/data/klear_math_30k_fi}"

echo "============================================"
echo "  PRIME-RL Single-Node Training (Mixed): $FULL_RUN_NAME"
echo "============================================"
echo "User:         $USER"
echo "Job:          $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Inference:    GPUs $INF_GPU_LIST (DP=$NUM_INF_GPUS)"
echo "Judge:        GPUs $RM_GPU_LIST ($NUM_RM_GPUS GPUs, port $RM_PORT)"
echo "Trainer:      GPUs $TRAIN_GPU_LIST ($NUM_TRAIN_GPUS workers)"
NO_MATH=${NO_MATH:-0}
if [ "$IFEVAL_ONLY" = "1" ]; then
    ENVS_DESC="ifeval-en only"
elif [ "$JUDGE_ONLY" = "1" ]; then
    ENVS_DESC="rubric-judge-en only"
elif [ "$MATH_ONLY" = "1" ]; then
    ENVS_DESC="english-math only"
elif [ "$NO_MATH" = "1" ]; then
    ENVS_DESC="ifeval-en, rubric-judge-en"
else
    ENVS_DESC="ifeval-en, rubric-judge-en, math-en"
fi
echo "Envs:         $ENVS_DESC"
echo "Model:        $SFT_MODEL"
[ "$IFEVAL_ONLY" = "1" ] || [ "$MATH_ONLY" = "1" ] || echo "Judge model:  $JUDGE_MODEL"
echo "Seq len:      $INF_SEQ_LEN"
echo "Temp:         $ROLLOUT_TEMP"
echo "Rollouts:     $ROLLOUTS"
echo "LR:           $LR"
echo "KL tau:       $KL_TAU"
echo "IPO mask:     $IPO_MASK_LOW / $IPO_MASK_HIGH"
echo "Adv tau:      $ADV_TAU"
echo "ODF thresh:   hard=$HARD_THRESHOLD easy=$EASY_THRESHOLD"
echo "Steps:        $MAX_STEPS (ckpt every $CKPT_INTERVAL)"
echo "Container:    $IMG"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ── Generate configs ──
cat > "$OUTPUT_DIR/infer.toml" << TOMLEOF
enable_fp32_lm_head = ${FP32_LM_HEAD:-false}

[model]
name = "$SFT_MODEL"
max_model_len = $INF_SEQ_LEN

[parallel]
tp = 1

[server]
port = $PORT
$(if [ -n "${VLLM_MODEL_IMPL:-}" ]; then echo "[vllm_extra]"; echo "model_impl = \"$VLLM_MODEL_IMPL\""; fi)
TOMLEOF

cat > "$OUTPUT_DIR/rm_infer.toml" << TOMLEOF
[model]
name = "$JUDGE_MODEL"
max_model_len = $RM_SEQ_LEN

[parallel]
tp = 1

[server]
port = $RM_PORT
TOMLEOF

cat > "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
batch_size = $BATCH_SIZE
group_size = $ROLLOUTS
max_steps = $MAX_STEPS
max_off_policy_steps = $MAX_OFF_POLICY
seq_len = $INF_SEQ_LEN

[model]
name = "$SFT_MODEL"

[model.client]
dp_rank_count = $NUM_INF_GPUS
base_url = ["http://127.0.0.1:${PORT}/v1"]

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-orch"
offline = $WANDB_OFFLINE

[train.sampling]
temperature = $ROLLOUT_TEMP
max_completion_tokens = $MAX_TOKENS

[renderer]
name = "$RENDERER"

TOMLEOF

if [ "$NO_MATH" != "1" ] && [ "$IFEVAL_ONLY" != "1" ] && [ "$JUDGE_ONLY" != "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "math-env"
name = "english-math"

[train.env.args]
dataset_name = "$EN_MATH_DATASET"
dataset_subset = "default"
question_key = "${MATH_QUESTION_KEY:-problem}"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60
TOMLEOF
fi

if [ "$MATH_ONLY" != "1" ] && [ "$JUDGE_ONLY" != "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "dolci-ifeval-env"
name = "ifeval-en"
TOMLEOF
if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[train.env.args]
system_prompt = ""
TOMLEOF
fi
fi

if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "rubric-judge-env"
name = "rubric-judge-en"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "http://127.0.0.1:${RM_PORT}"
max_judge_tokens = 64
TOMLEOF
if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
system_prompt = ""
TOMLEOF
fi
fi

if [ "${NO_EVAL:-0}" != "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[eval]
interval = $EVAL_INTERVAL

[[eval.env]]
name = "aime25-eval"
taskset = { id = "aime25-v1" }
harness = { id = "null" }

[[eval.env]]
name = "ifbench-eval"
taskset = { id = "ifbench-v1" }
harness = { id = "null" }
TOMLEOF
fi

cat > "$OUTPUT_DIR/train_config.toml" << TOMLEOF
max_steps = $MAX_STEPS
matmul_precision = "highest"

[model]
name = "$SFT_MODEL"
seq_len = $TRAINER_SEQ_LEN
optim_cpu_offload = false
compile = "None"
impl = "$TRAINER_MODEL_IMPL"
attn = "flash_attention_2"

[model.ac]
freq = 1

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-trainer"
offline = $WANDB_OFFLINE

[ckpt]
interval = $CKPT_INTERVAL
keep_last = $KEEP_LAST

[loss]
type = "default"
kl_tau = $KL_TAU
dppo_mask_low = $IPO_MASK_LOW
dppo_mask_high = $IPO_MASK_HIGH
adv_tau = $ADV_TAU

[optim]
lr = $LR
max_norm = 1.0

[scheduler]
type = "constant"
TOMLEOF

# ── Environment setup ──
ENV_SETUP="$OUTPUT_DIR/env_setup.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR
export PATH="$(dirname $UV):$PATH"
export TRITON_CACHE_DIR=/tmp/triton_\$(hostname)_\${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_\$(hostname)_\${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_\$(hostname)_\${SLURM_JOB_ID}
export HOME=/tmp/home_\$(hostname)_\${SLURM_JOB_ID}
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$XDG_CACHE_HOME \$HOME

export UV_PYTHON=$PRIME_RL_DIR/.venv/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5
export RCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export PYTHONUNBUFFERED=1
ENVEOF
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bashrc" ]; then
    WANDB_API_KEY=$(grep -oP "(?<=WANDB_API_KEY=)['\"]?\K[^'\" ]+" "$HOME/.bashrc" | head -1 || true)
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "export WANDB_API_KEY=$WANDB_API_KEY" >> "$ENV_SETUP"
fi
if [ -n "${WANDB_MODE:-}" ]; then
    echo "export WANDB_MODE=$WANDB_MODE" >> "$ENV_SETUP"
fi

# ── Container bind mounts ──
CONTAINER_FLAGS="--rocm --cleanenv --containall"
BIND_MOUNTS="-B /shared_silo/scratch:/shared_silo/scratch:rw"
BIND_MOUNTS="$BIND_MOUNTS -B $HOME:$HOME:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

# ── Convert to safetensors if needed ──
if [ ! -f "$SFT_MODEL/model.safetensors.index.json" ] && [ ! -f "$SFT_MODEL/model.safetensors" ]; then
    echo "Converting model to safetensors..."
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && uv run python -c \"
from transformers import AutoModelForCausalLM; import torch
model = AutoModelForCausalLM.from_pretrained('$SFT_MODEL', torch_dtype=torch.bfloat16)
model.save_pretrained('$SFT_MODEL'); print('Model saved as safetensors')
\""
fi

# Clean stale prime/ cache
rm -rf "$SFT_MODEL/prime"

# ── Sync dependencies ──
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && uv sync --all-extras"

# ── Download NLTK data required by Finnish IFEval env ──
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && uv run python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\""

# ── Launch policy inference server on INF GPUs ──
echo "Starting policy inference server on GPUs $INF_GPU_LIST..."
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && \
        export HIP_VISIBLE_DEVICES=$INF_GPU_LIST && \
        uv run inference \
        @ $OUTPUT_DIR/infer.toml \
        --parallel.dp $NUM_INF_GPUS \
        --gpu-memory-utilization 0.85" \
    > "$OUTPUT_DIR/inference_0.log" 2>&1 &
INF_PID=$!

RM_PID=
# ── Launch LLM judge server on RM GPUs (skipped for IFEVAL_ONLY and MATH_ONLY) ──
if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    echo "Starting judge server on GPUs $RM_GPU_LIST..."
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && \
            export HIP_VISIBLE_DEVICES=$RM_GPU_LIST && \
            uv run inference \
            @ $OUTPUT_DIR/rm_infer.toml \
            --parallel.dp $NUM_RM_GPUS \
            --gpu-memory-utilization 0.85" \
        > "$OUTPUT_DIR/rm_inference.log" 2>&1 &
    RM_PID=$!
fi

# ── Wait for servers to become healthy ──
echo "Waiting for policy inference server on port $PORT..."
for attempt in $(seq 1 600); do
    curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null && break
    sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null \
    || { echo "ERROR: policy inference server did not start within 1200s"; kill $INF_PID ${RM_PID:-} 2>/dev/null; exit 1; }
echo "Policy inference server ready."

if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    echo "Waiting for judge server on port $RM_PORT..."
    for attempt in $(seq 1 600); do
        curl -sf "http://127.0.0.1:${RM_PORT}/health" &>/dev/null && break
        sleep 2
    done
    curl -sf "http://127.0.0.1:${RM_PORT}/health" &>/dev/null \
        || { echo "ERROR: judge server did not start within 1200s"; kill $INF_PID $RM_PID 2>/dev/null; exit 1; }
    echo "Judge server ready."
fi

# ── Launch orchestrator + trainer on TRAIN GPUs ──
echo "Starting orchestrator and trainer on GPUs $TRAIN_GPU_LIST..."
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "
        source $ENV_SETUP
        export HIP_VISIBLE_DEVICES=$TRAIN_GPU_LIST
        mkdir -p wandb

        uv run orchestrator \
            @ $OUTPUT_DIR/orch_config.toml \
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

        uv run torchrun \
            --standalone \
            --nproc-per-node $NUM_TRAIN_GPUS \
            --log-dir $OUTPUT_DIR/logs/trainer/torchrun \
            -m prime_rl.trainer.rl.train \
            @ $OUTPUT_DIR/train_config.toml \
            --output-dir $OUTPUT_DIR
    "

kill $INF_PID $RM_PID 2>/dev/null || true
wait $INF_PID $RM_PID 2>/dev/null || true

rm -rf "$JOB_MODEL_DIR"
echo "Done!"
