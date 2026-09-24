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
# Hard cap on tokens spent inside <think>. vLLM injects </think> at the budget, so
# the rollout always gets to write an answer instead of hitting MAX_TOKENS mid-thought.
# Leave MAX_TOKENS - THINKING_BUDGET as answer headroom.
#
# extra_body lands in the request's sampling_params, so every key must be a real
# SamplingParams field. chat_template_kwargs is NOT one — and it would be pointless
# here anyway: /inference/v1/generate is tokens-in, the renderer builds the prompt
# client-side, so there is no server-side chat template to configure.
#
# vLLM rejects thinking_token_budget outright unless the server was started with a
# reasoning parser (v1/engine/input_processor.py:102). prime-rl resolves the parser by
# regex on the model NAME, and we stage checkpoints at models/job_<id>_model, which
# matches nothing — so it must be named explicitly. Gated on THINKING_BUDGET because no
# other run has ever served with a parser and we don't want to change them silently.
THINKING_BUDGET=${THINKING_BUDGET:-}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
# max_inflight_rollouts = batch_size * oversampling_factor, i.e. this is the inference
# concurrency knob. Unset means 1.0, which caps concurrency at batch_size.
OVERSAMPLING=${OVERSAMPLING:-}
ROLLOUTS=${ROLLOUTS:-8}
LR=${LR:-1e-6}
KL_TAU=${KL_TAU:-0}
IPO_MASK_LOW=${IPO_MASK_LOW:-0.2}
IPO_MASK_HIGH=${IPO_MASK_HIGH:-0.2}
ADV_TAU=${ADV_TAU:-2.0}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
MAX_STEPS=${MAX_STEPS:-500}
MAX_OFF_POLICY=${MAX_OFF_POLICY:-8}
CKPT_INTERVAL=${CKPT_INTERVAL:-100}
KEEP_LAST=${KEEP_LAST:-30}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning}
TRAINER_ATTN=${TRAINER_ATTN:-flash_attention_2}
TRAINER_MODEL_IMPL=${TRAINER_MODEL_IMPL:-hf}
RENDERER=${RENDERER:-qwen3.5}

# Transport for trainer -> inference weight updates. "filesystem" writes a full HF
# checkpoint to Lustre every step and has the inference workers read it back; "nccl"
# broadcasts the state dict straight into their GPUs. The `rl` entrypoint defaults to
# nccl, but this launcher starts inference, orchestrator, and trainer as three separate
# processes, so each falls back to its own default — which is filesystem. Keep that as
# the default here so existing runs are unchanged.
#
# inference_world_size must equal dp * tp exactly; the rendezvous is world_size + 1
# (trainer is rank 0) and a wrong value hangs rather than errors. tp is pinned to 1 in
# infer.toml and dp is NUM_INF_GPUS, so the product is NUM_INF_GPUS.
WEIGHT_BROADCAST=${WEIGHT_BROADCAST:-filesystem}
case "$WEIGHT_BROADCAST" in
    filesystem|nccl) ;;
    *) echo "ERROR: WEIGHT_BROADCAST must be 'filesystem' or 'nccl', got '$WEIGHT_BROADCAST'"; exit 1 ;;
esac
# 29501 is the prime-rl default. torchrun runs --standalone and picks its own
# rendezvous port, so there is no collision.
BROADCAST_PORT=${BROADCAST_PORT:-29501}

# Applies to trainer, orchestrator, and inference. "debug" is what surfaces the
# receive side of a weight update ("Updated weights to step N in ...", watcher.py);
# at info level that number only reaches wandb. Noisy — for short runs only.
LOG_LEVEL=${LOG_LEVEL:-}

# Expert parallelism. Unset leaves prime-rl's "auto", which for Qwen3.5 MoE resolves
# through _is_moe_model — and that only inspects top-level config keys, so the
# VL-wrapped 35B-A3B config (experts live under text_config) silently yields ep=1.
# Set explicitly to say what you mean.
EP=${EP:-}

# ── Length penalty (OFF unless LENGTH_BUDGET is set) ──
# Soft output-token budget: no penalty below LENGTH_BUDGET, then a linear ramp to
# LENGTH_PENALTY_WEIGHT reward units over the next LENGTH_RAMP tokens. Keep
# LENGTH_BUDGET + LENGTH_RAMP = MAX_TOKENS so the ramp is the headroom reserved for the
# final answer (e.g. LENGTH_BUDGET=30720 LENGTH_RAMP=2048 against MAX_TOKENS=32768).
LENGTH_BUDGET=${LENGTH_BUDGET:-}
LENGTH_RAMP=${LENGTH_RAMP:-2048}
LENGTH_PENALTY_WEIGHT=${LENGTH_PENALTY_WEIGHT:-1.0}

# ── Reasoning policy ──
# The IF env grades the model's final answer. When the policy never closes its
# <think> block the parser hands back the whole reasoning trace as the answer,
# and format constraints ("no commas", "all lowercase") happily pass on a
# rambling trace — rewarding non-termination. Setting this scores those
# rollouts 0. Set to 0 for a non-reasoning policy, which never emits </think>
# and would otherwise score 0 on every rollout.
REASONING_MODEL=${REASONING_MODEL:-1}

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
JUDGE_FI=${JUDGE_FI:-0}
if [ "$IFEVAL_ONLY" = "1" ]; then
    ENVS_DESC="ifeval-en only"
elif [ "$JUDGE_ONLY" = "1" ]; then
    ENVS_DESC="rubric-judge-en"
elif [ "$MATH_ONLY" = "1" ]; then
    ENVS_DESC="english-math only"
elif [ "$NO_MATH" = "1" ]; then
    ENVS_DESC="ifeval-en, rubric-judge-en"
else
    ENVS_DESC="ifeval-en, rubric-judge-en, math-en"
fi
if [ "$JUDGE_FI" = "1" ] && [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    ENVS_DESC="$ENVS_DESC, rubric-judge-fi"
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
echo "Steps:        $MAX_STEPS (ckpt every $CKPT_INTERVAL)"
echo "Broadcast:    $WEIGHT_BROADCAST$([ "$WEIGHT_BROADCAST" = nccl ] && echo " (port $BROADCAST_PORT, world $NUM_INF_GPUS+1)")"
echo "Container:    $IMG"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ── Generate configs ──
cat > "$OUTPUT_DIR/infer.toml" << TOMLEOF
enable_fp32_lm_head = ${FP32_LM_HEAD:-false}

[model]
name = "$SFT_MODEL"
max_model_len = $INF_SEQ_LEN
$(if [ -n "$THINKING_BUDGET" ]; then echo "reasoning_parser = \"$REASONING_PARSER\""; fi)

[parallel]
tp = 1

[server]
port = $PORT

[weight_broadcast]
type = "$WEIGHT_BROADCAST"
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
$(if [ -n "$OVERSAMPLING" ]; then echo "oversampling_factor = $OVERSAMPLING"; fi)

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
$(if [ -n "$THINKING_BUDGET" ]; then echo "extra_body = { thinking_token_budget = $THINKING_BUDGET }"; fi)

[renderer]
name = "$RENDERER"
$(if [ -n "$LENGTH_BUDGET" ]; then cat << LPEOF

[algo.length_penalty]
type = "budget"
num_output_tokens_budget = $LENGTH_BUDGET
num_output_tokens_ramp = $LENGTH_RAMP
weight = $LENGTH_PENALTY_WEIGHT
LPEOF
fi)

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

[train.env.args]
reasoning_model = $(if [ "$REASONING_MODEL" = "1" ]; then echo true; else echo false; fi)
TOMLEOF
if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
system_prompt = ""
TOMLEOF
fi
fi

if [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
# "legacy" = our score-first prompt (1-100, no deliberation, 64 tokens is enough).
# "open-instruct" = the verbatim open-instruct templates (1-10, reasoning-first JSON),
# which need room to reason before the score appears.
JUDGE_PROMPT_STYLE="${JUDGE_PROMPT_STYLE:-legacy}"
if [ "$JUDGE_PROMPT_STYLE" = "open-instruct" ]; then
  MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-512}"
else
  MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-64}"
fi
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "rubric-judge-env"
name = "rubric-judge-en"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "http://127.0.0.1:${RM_PORT}"
judge_prompt_style = "$JUDGE_PROMPT_STYLE"
max_judge_tokens = $MAX_JUDGE_TOKENS
TOMLEOF
if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
system_prompt = ""
TOMLEOF
fi

# The Finnish twin: same judge, same rubric, machine-translated prompts. use_finnish
# also selects the Finnish system prompt, so the policy is not told in English to
# answer a Finnish question.
if [ "$JUDGE_FI" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "rubric-judge-env"
name = "rubric-judge-fi"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "http://127.0.0.1:${RM_PORT}"
judge_prompt_style = "$JUDGE_PROMPT_STYLE"
max_judge_tokens = $MAX_JUDGE_TOKENS
dataset_path = "$FI_JUDGE_DATASET"
use_finnish = true
TOMLEOF
if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
system_prompt = ""
TOMLEOF
fi
fi
fi

# No in-loop eval: aime25 is 30 problems at 1 sample, so its score is dominated by
# sampling noise, and ifbench costs generation without informing the run. Evaluate
# saved checkpoints out-of-loop instead (aime25 at avg@64).

# Opt-in: off by default so batch-size ablations keep an unfiltered baseline.
# Setting pre_batch_filters replaces the default list wholesale, so all three types
# are re-listed. zero_advantage enforces: degenerate groups are dropped before they
# reach pending_batch, so the orchestrator backfills them with informative groups
# instead of shipping a shrunken batch. Costs generation — expect the trainer to
# wait on rollouts.
if [ "${PREBATCH_FILTER:-0}" = "1" ]; then
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[pre_batch_filters]]
type = "gibberish"
enforce = false

[[pre_batch_filters]]
type = "repetition"
enforce = false

[[pre_batch_filters]]
type = "zero_advantage"
enforce = true
TOMLEOF
fi

# Appended last: a table header here would otherwise swallow the trailing keys of
# whichever [train.env.args] block was written before it.
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[weight_broadcast]
type = "$WEIGHT_BROADCAST"
$(if [ "$WEIGHT_BROADCAST" = "nccl" ]; then
    echo "host = \"localhost\""
    echo "port = $BROADCAST_PORT"
    echo "inference_world_size = $NUM_INF_GPUS"
fi)
TOMLEOF

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
$(if [ -n "$EP" ]; then echo "ep = $EP"; fi)

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

[weight_broadcast]
type = "$WEIGHT_BROADCAST"
$(if [ "$WEIGHT_BROADCAST" = "nccl" ]; then
    echo "host = \"localhost\""
    echo "port = $BROADCAST_PORT"
    echo "inference_world_size = $NUM_INF_GPUS"
fi)
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
$([ -n "$LOG_LEVEL" ] && echo "export PRIME_LOG_LEVEL=$LOG_LEVEL")
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
