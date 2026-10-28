#!/bin/bash
#SBATCH --job-name=prime-rl-multinode-mixed
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=./logs/rl_%j.out
#SBATCH --error=./logs/rl_%j.err

# =============================================================================
#   Multi-node mixed RL training. Same envs, hyperparameters and config layout
#   as rl_train_singlenode_mixed.sbatch, but policy inference spans N nodes:
#
#     Node 0 (shared)      — split between policy inference, LLM judge, trainer
#     Nodes 1..N-1 (infer) — all 8 GPUs serve policy inference
#
#   Every policy inference instance is an independent, single-GPU vLLM server
#   (dp=1) on its own port. The orchestrator round-robins across all of them via
#   client.base_url, so there is no cross-node DP and no X-data-parallel-rank
#   pinning — which is what makes an uneven split (8 GPUs + N GPUs) expressible
#   at all, since client.dp_rank_count is a single scalar applied to every URL.
#   Adding a node is therefore just appending 8 more URLs to that list.
#
#   Node count comes from SLURM (--nodes on the command line overrides the
#   #SBATCH default below); everything else scales off $SLURM_NNODES.
#
#   Usage:
#     JUDGE_MODEL=/path/to/judge \
#       sbatch --nodes=4 rl_train_multinode_mixed.sbatch <run_name> <model_path>
#
#   GPU layout (override via env vars):
#     Node 0:    NUM_INF_GPUS=5   — GPUs 0..4  → policy inference (5 instances)
#                NUM_RM_GPUS=2    — GPUs 5..6  → LLM judge inference (one dp=2 server)
#                NUM_TRAIN_GPUS=1 — GPU 7      → trainer
#     Nodes 1..: NUM_INF_NODE_GPUS=8 — GPUs 0..7 → policy inference (8 instances each)
#
#   JUDGE_NODE_IDX moves the judge off node 0 onto a node of its own, which is
#   what frees all 8 of node 0's GPUs for the trainer. That node leaves the
#   inference pool entirely, keeping every remaining inference node uniform —
#   the URL and port loops below assume NUM_INF_NODE_GPUS instances on each.
#
#   Inference capacity is only useful if the orchestrator keeps it busy: with
#   OVERSAMPLING_FACTOR unset, prime-rl keeps exactly batch_size rollouts
#   in flight, which starves many instances. Set OVERSAMPLING_FACTOR so that
#   batch_size * factor is a few multiples of the total instance count.
#
#   Envs. The six that can be emitted are english-math, finnish-math, ifeval-en,
#   ifeval-fi, rubric-judge-en and rubric-judge-fi. The *_ONLY flags narrow to one
#   family; the *_FI flags are additive, so MATH_FI/IFEVAL_FI/JUDGE_FI=1 give the
#   bilingual six. See rl_train_poro3_35b.sbatch for the production preset.
#
#   Example (4 nodes, math-only so the judge GPUs go to inference):
#     MATH_ONLY=1 NUM_INF_GPUS=7 NUM_RM_GPUS=0 NUM_TRAIN_GPUS=1 \
#     OVERSAMPLING_FACTOR=4 \
#       sbatch --nodes=4 rl_train_multinode_mixed.sbatch \
#         my-mixed-run \
#         /path/to/policy-model
# =============================================================================

set -euo pipefail

RUN_NAME=${1:?Usage: sbatch rl_train_multinode_mixed.sbatch <run_name> <model_path>}
SFT_MODEL=${2:?Usage: sbatch rl_train_multinode_mixed.sbatch <run_name> <model_path>}
IFEVAL_ONLY=${IFEVAL_ONLY:-0}
MATH_ONLY=${MATH_ONLY:-0}
JUDGE_ONLY=${JUDGE_ONLY:-0}
NO_MATH=${NO_MATH:-0}
# Repetition-only: a surgical pass that scores nothing but the absence of loops. It
# excludes every other env rather than joining them — the reward is orthogonal to
# correctness, so mixing it in would silently trade answer quality for terseness.
# NOTE there is no reference-policy anchor anywhere in prime-rl (kl_tau penalises
# trainer-vs-inference mismatch, not drift from the init), so nothing protects math
# ability here. Run it short and checkpoint often.
REPETITION_ONLY=${REPETITION_ONLY:-0}
# Additive Finnish twins. Each adds a second env of the same family on the Finnish
# dataset; they do not replace the English one. Subject to the same *_ONLY gates.
MATH_FI=${MATH_FI:-0}
IFEVAL_FI=${IFEVAL_FI:-0}
JUDGE_FI=${JUDGE_FI:-0}
REPETITION_FI=${REPETITION_FI:-0}

# Which envs actually get emitted — resolved once so the gates below, the banner and
# the judge launch cannot drift apart.
EMIT_MATH_EN=0; EMIT_MATH_FI=0
EMIT_IFEVAL_EN=0; EMIT_IFEVAL_FI=0
EMIT_JUDGE_EN=0; EMIT_JUDGE_FI=0
EMIT_REP_EN=0; EMIT_REP_FI=0
if [ "$REPETITION_ONLY" = "1" ]; then
    EMIT_REP_EN=1
    [ "$REPETITION_FI" = "1" ] && EMIT_REP_FI=1
fi
if [ "$REPETITION_ONLY" != "1" ] && [ "$NO_MATH" != "1" ] && [ "$IFEVAL_ONLY" != "1" ] && [ "$JUDGE_ONLY" != "1" ]; then
    EMIT_MATH_EN=1
    [ "$MATH_FI" = "1" ] && EMIT_MATH_FI=1
fi
if [ "$REPETITION_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ] && [ "$JUDGE_ONLY" != "1" ]; then
    EMIT_IFEVAL_EN=1
    [ "$IFEVAL_FI" = "1" ] && EMIT_IFEVAL_FI=1
fi
if [ "$REPETITION_ONLY" != "1" ] && [ "$IFEVAL_ONLY" != "1" ] && [ "$MATH_ONLY" != "1" ]; then
    EMIT_JUDGE_EN=1
    [ "$JUDGE_FI" = "1" ] && EMIT_JUDGE_FI=1
fi
NEEDS_JUDGE=0
{ [ "$EMIT_JUDGE_EN" = "1" ] || [ "$EMIT_JUDGE_FI" = "1" ]; } && NEEDS_JUDGE=1

if [ "$NEEDS_JUDGE" = "1" ]; then
    JUDGE_MODEL=${JUDGE_MODEL:?ERROR: JUDGE_MODEL env var must be set to the judge model path}
else
    JUDGE_MODEL=${JUDGE_MODEL:-}
fi

# ── Shared paths ──
BASE="/shared_silo/scratch/kahakala/lib/prime-rl-upstream"
PRIME_RL_DIR="$BASE"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/primus_v26.2_moefix.sif}"

# ── Validate ──
[ -d "$SFT_MODEL" ] || { echo "ERROR: model path does not exist: $SFT_MODEL"; exit 1; }
SFT_MODEL=$(realpath "$SFT_MODEL")
if [ "$NEEDS_JUDGE" = "1" ]; then
    [ -d "$JUDGE_MODEL" ] || { echo "ERROR: judge model path does not exist: $JUDGE_MODEL"; exit 1; }
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

# ── Node layout ──
[ "$SLURM_NNODES" -ge 2 ] || { echo "ERROR: need at least 2 nodes, got $SLURM_NNODES"; exit 1; }
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_NODE=${NODELIST[0]}           # inference + orchestrator + trainer (+ judge if idx 0)

# 0 keeps the judge on the shared node, which is what every run before this did.
# Anything else dedicates NODELIST[idx] to the judge and removes it from the
# inference pool — the only way the trainer can have all 8 of node 0's GPUs.
JUDGE_NODE_IDX=${JUDGE_NODE_IDX:-0}
[ "$JUDGE_NODE_IDX" -lt "$SLURM_NNODES" ] \
    || { echo "ERROR: JUDGE_NODE_IDX=$JUDGE_NODE_IDX but only $SLURM_NNODES nodes"; exit 1; }
JUDGE_OFFNODE=0
[ "$JUDGE_NODE_IDX" -ne 0 ] && JUDGE_OFFNODE=1
JUDGE_NODE=${NODELIST[$JUDGE_NODE_IDX]}

INF_NODES=()                       # policy inference only
for i in $(seq 1 $((SLURM_NNODES - 1))); do
    [ "$i" -eq "$JUDGE_NODE_IDX" ] && continue
    INF_NODES+=("${NODELIST[$i]}")
done
NUM_INF_NODES=${#INF_NODES[@]}

# ── GPU layout ──
NUM_INF_GPUS=${NUM_INF_GPUS:-5}
NUM_RM_GPUS=${NUM_RM_GPUS:-2}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-1}
NUM_INF_NODE_GPUS=${NUM_INF_NODE_GPUS:-8}
[ "$NUM_INF_NODE_GPUS" -le 8 ] || { echo "ERROR: NUM_INF_NODE_GPUS must be <= 8, got $NUM_INF_NODE_GPUS"; exit 1; }

# The judge only competes for node 0's GPUs when it lives there.
MAIN_NODE_GPUS=$((NUM_INF_GPUS + NUM_TRAIN_GPUS))
[ "$JUDGE_OFFNODE" = "1" ] || MAIN_NODE_GPUS=$((MAIN_NODE_GPUS + NUM_RM_GPUS))
[ "$MAIN_NODE_GPUS" -le 8 ] \
    || { echo "ERROR: node $MAIN_NODE needs $MAIN_NODE_GPUS GPUs (inference+trainer$([ "$JUDGE_OFFNODE" = 1 ] || echo "+judge")), max 8"; exit 1; }
if [ "$NEEDS_JUDGE" = "1" ]; then
    [ "$NUM_RM_GPUS" -ge 1 ] || { echo "ERROR: a judge env is enabled but NUM_RM_GPUS=0"; exit 1; }
    [ "$NUM_RM_GPUS" -le 8 ] || { echo "ERROR: NUM_RM_GPUS must be <= 8, got $NUM_RM_GPUS"; exit 1; }
fi

if [ "$JUDGE_OFFNODE" = "1" ]; then
    RM_GPU_LIST=$(seq -s, 0 $((NUM_RM_GPUS - 1)))
    TRAIN_GPU_OFFSET=$NUM_INF_GPUS
else
    RM_GPU_LIST=$(seq -s, $NUM_INF_GPUS $((NUM_INF_GPUS + NUM_RM_GPUS - 1)))
    TRAIN_GPU_OFFSET=$((NUM_INF_GPUS + NUM_RM_GPUS))
fi
TRAIN_GPU_LIST=$(seq -s, $TRAIN_GPU_OFFSET $((TRAIN_GPU_OFFSET + NUM_TRAIN_GPUS - 1)))
NUM_INF_INSTANCES=$((NUM_INF_GPUS + NUM_INF_NODES * NUM_INF_NODE_GPUS))

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
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
MAX_STEPS=${MAX_STEPS:-500}
MAX_OFF_POLICY=${MAX_OFF_POLICY:-8}
# Rollouts kept in flight = batch_size * OVERSAMPLING_FACTOR. Unset means 1.0,
# which starves inference once the pre-batch filter starts dropping groups.
OVERSAMPLING_FACTOR=${OVERSAMPLING_FACTOR:-}
CKPT_INTERVAL=${CKPT_INTERVAL:-100}
KEEP_LAST=${KEEP_LAST:-30}
# false writes both weights/step_N (66G for 35B-A3B) and checkpoints/step_N, the full
# DCP training state (390G). true writes only the former, which is all a warm-start
# needs — set it false only if you intend to resume in place.
CKPT_WEIGHTS_ONLY=${CKPT_WEIGHTS_ONLY:-false}
# 1 pins the output directory to SLURM_JOB_ID (so a requeue lands in the same place) and
# sets resume_step = -1 on both trainer and orchestrator. With no checkpoint on disk
# that is a no-op bar a warning, so it is safe on a first start. Requires full
# checkpoints: CKPT_WEIGHTS_ONLY=true saves no optimizer state to resume from.
AUTO_RESUME=${AUTO_RESUME:-0}
if [ "$AUTO_RESUME" = "1" ] && [ "$CKPT_WEIGHTS_ONLY" = "true" ]; then
    echo "ERROR: AUTO_RESUME=1 needs full checkpoints, but CKPT_WEIGHTS_ONLY=true"
    exit 1
fi
WANDB_PROJECT=${WANDB_PROJECT:-reasoning}
TRAINER_ATTN=${TRAINER_ATTN:-flash_attention_2}
TRAINER_MODEL_IMPL=${TRAINER_MODEL_IMPL:-hf}
RENDERER=${RENDERER:-qwen3.5}

# Expert parallel degree. MUST be set explicitly for Qwen3.5-35B-A3B: _is_moe_model
# (parallel_dims.py) only inspects top-level config keys, and that checkpoint is a
# VL-wrapped composite whose num_experts sits under text_config — so ep = "auto"
# silently resolves to 1 and the mesh is wrong without any warning. Requires
# TRAINER_MODEL_IMPL=custom; with "hf", resolve_ep returns 1 regardless.
# Never set EP equal to a single node's GPU count: ep == 8 on 8 GPUs makes
# dp_shard_mod_ep degenerate and the trainer hangs at 100% util with zero steps.
EP=${EP:-}

# Whether the ifeval envs expect a reasoning model (they strip the think block
# before grading). False only for a non-reasoning policy.
REASONING_MODEL=${REASONING_MODEL:-true}

# Finnish-output reward from the `language-reward` package: fastText LID over the
# rollout, added to the env reward with this weight. 0 disables it entirely.
# Scope "full" grades the whole rollout (reasoning included), "answer" only the
# post-think answer — see the package docs for why fi uses full and en uses answer.
LANGUAGE_REWARD_WEIGHT=${LANGUAGE_REWARD_WEIGHT:-0}
LANGUAGE_REWARD_SCOPE=${LANGUAGE_REWARD_SCOPE:-}

# Transport for trainer -> inference weight updates. "filesystem" writes a full HF
# checkpoint to Lustre every step and has every inference instance read it back;
# "nccl" broadcasts the state dict straight into their GPUs. Measured on one node at
# 35B-A3B (jobs 49398/49399): filesystem 55.5s send + 30.2s receive sequentially,
# nccl 16.6s for the single collective. Default stays filesystem so existing runs
# are unchanged.
#
# Multi-node is NOT the same wiring as single-node. The rendezvous is a TCP store
# served by trainer rank 0, so:
#   - the TRAINER binds "0.0.0.0" (it must accept connections from the inference
#     nodes, and "localhost" would only ever accept its own),
#   - the ORCHESTRATOR advertises $MAIN_NODE_IP, because it POSTs that string to
#     every inference server's /init_broadcaster and they dial it. "0.0.0.0" is not
#     a destination, and "localhost" would send the off-node workers to themselves.
# Both are silent failures: a wrong host or world size HANGS until `timeout`
# (1200 s), it does not error.
WEIGHT_BROADCAST=${WEIGHT_BROADCAST:-filesystem}
case "$WEIGHT_BROADCAST" in
    filesystem|nccl) ;;
    *) echo "ERROR: WEIGHT_BROADCAST must be 'filesystem' or 'nccl', got '$WEIGHT_BROADCAST'"; exit 1 ;;
esac
# 29501 is the prime-rl default. It collides with nothing here: TRAINER_RDZV_PORT is
# dead (torchrun runs --standalone) and the inference ports start at 8000.
BROADCAST_PORT=${BROADCAST_PORT:-29501}

# Applies to trainer, orchestrator, and inference. "debug" surfaces the receive side
# of a weight update ("Updated weights to step N in ...", watcher.py); at info level
# that number only reaches wandb. Noisy — for short runs only.
LOG_LEVEL=${LOG_LEVEL:-}

# One port per inference instance, starting here. The judge sits above the range
# so it can never collide with an inference instance on the shared node.
PORT_BASE=8000
RM_PORT=8100
TRAINER_RDZV_PORT=29501

# ── Per-job model directory (avoid cache race conditions) ──
JOB_MODEL_DIR="$BASE/models/job_${SLURM_JOB_ID}_model"
mkdir -p "$JOB_MODEL_DIR"
for f in "$SFT_MODEL"/*; do
    [ -f "$f" ] && ln -sf "$f" "$JOB_MODEL_DIR/$(basename "$f")"
done
SFT_MODEL="$JOB_MODEL_DIR"

cd "$PRIME_RL_DIR"

# A requeued job re-runs this script from the top with the same SLURM_JOB_ID but a new
# wall-clock time, so a bare timestamp would send the restart to an empty directory and
# silently retrain from step 0. Under AUTO_RESUME the first start records its directory
# in a marker keyed by job id, and every restart reads it back.
MARKER="$BASE/outputs/.jobdir-${SLURM_JOB_ID}"
if [ "$AUTO_RESUME" = "1" ] && [ -f "$MARKER" ]; then
    OUTPUT_DIR=$(cat "$MARKER")
    FULL_RUN_NAME=$(basename "$OUTPUT_DIR")
    RESTARTED=1
    echo "Requeue detected (restart #${SLURM_RESTART_COUNT:-?}): reusing $OUTPUT_DIR"
else
    FULL_RUN_NAME="${RUN_NAME}-$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="$BASE/outputs/$FULL_RUN_NAME"
    RESTARTED=0
fi
mkdir -p "$OUTPUT_DIR/run_default"
if [ "$AUTO_RESUME" = "1" ]; then
    echo "$OUTPUT_DIR" > "$MARKER"
fi

INF_NODE_IPS=()
for NODE in "${INF_NODES[@]}"; do
    INF_NODE_IPS+=("$(srun --nodes=1 --ntasks=1 -w "$NODE" /usr/sbin/ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')")
done

# The trainer lives on MAIN_NODE, so this is the address the off-node inference
# instances dial for the NCCL broadcast rendezvous. eno0 deliberately matches
# NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME in env_setup.sh.
MAIN_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$MAIN_NODE" /usr/sbin/ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')
[ -n "$MAIN_NODE_IP" ] || { echo "ERROR: could not resolve eno0 address of $MAIN_NODE"; exit 1; }

# The judge URL goes into the env config verbatim, so it has to be routable from the
# env workers (which run under the orchestrator on MAIN_NODE). Loopback only works
# while the judge shares that node.
if [ "$JUDGE_OFFNODE" = "1" ]; then
    JUDGE_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$JUDGE_NODE" /usr/sbin/ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')
    [ -n "$JUDGE_NODE_IP" ] || { echo "ERROR: could not resolve eno0 address of $JUDGE_NODE"; exit 1; }
else
    JUDGE_NODE_IP=127.0.0.1
fi
JUDGE_URL="http://${JUDGE_NODE_IP}:${RM_PORT}"

# Shared RL dataset directory. ifeval-en and rubric-judge-en read the same snapshot,
# the local mirror of allenai/Dolci-Think-RL-7B, but filter it differently.
DATASETS="${DATASETS:-/shared_silo/scratch/datasets/RL}"
EN_IFEVAL_DATASET="${EN_IFEVAL_DATASET:-$DATASETS/Dolci-Think-RL-7B}"
EN_JUDGE_DATASET="${EN_JUDGE_DATASET:-$DATASETS/Dolci-Think-RL-7B}"
FI_JUDGE_DATASET="${FI_JUDGE_DATASET:-$DATASETS/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi.jsonl}"
EN_MATH_DATASET="${EN_MATH_DATASET:-$DATASETS/klear_math_30k_en}"
FI_MATH_DATASET="${FI_MATH_DATASET:-$DATASETS/klear_math_30k_fi}"
FI_IFEVAL_DATASET="${FI_IFEVAL_DATASET:-$DATASETS/ifeval-fi/fi_dataset.jsonl}"
# The repetition env reads Dolci through rubric_judge_env's loader, so the Finnish side
# is the same translated jsonl the FI judge uses — use_finnish just switches the column.
# min_passrate 0.05 drops exactly dolci's passrate == 0 rows (the next value up is 1/8);
# job 53239 showed those loop least and run 3k tokens shorter. REPETITION_CATEGORIES is
# a space-separated list of dolci splits, one env replica each, so the reward traces stay
# separable and you can see which prompt distribution carries the signal.
REPETITION_DATASET="${REPETITION_DATASET:-$DATASETS/Dolci-Think-RL-7B}"
REPETITION_FI_DATASET="${REPETITION_FI_DATASET:-$FI_JUDGE_DATASET}"
REPETITION_CATEGORIES="${REPETITION_CATEGORIES:-math}"
REPETITION_REWARD_MODE="${REPETITION_REWARD_MODE:-blockcov}"
REPETITION_MIN_PASSRATE="${REPETITION_MIN_PASSRATE:-0.05}"
# klear_math_30k_fi keeps the English `problem` alongside the translation, so the
# Finnish env is the same math-env pointed at a different column. Both languages
# therefore draw on the same underlying problems and the same `answer` labels.
FI_MATH_QUESTION_KEY="${FI_MATH_QUESTION_KEY:-translated_problem}"

echo "============================================"
echo "  PRIME-RL Multi-Node Training (Mixed): $FULL_RUN_NAME"
echo "============================================"
echo "User:         $USER"
echo "Job:          $SLURM_JOB_ID"
echo "Nodes:        $SLURM_NODELIST"
echo "Shared node:  $MAIN_NODE"
if [ "$NUM_INF_GPUS" -gt 0 ]; then
    echo "  Inference:  GPUs 0..$((NUM_INF_GPUS - 1)) ($NUM_INF_GPUS instances, ports $PORT_BASE..$((PORT_BASE + NUM_INF_GPUS - 1)))"
fi
if [ "$NEEDS_JUDGE" = "1" ] && [ "$JUDGE_OFFNODE" != "1" ]; then
    echo "  Judge:      GPUs $RM_GPU_LIST ($NUM_RM_GPUS GPUs, port $RM_PORT)"
fi
echo "  Trainer:    GPUs $TRAIN_GPU_LIST ($NUM_TRAIN_GPUS workers)"
if [ "$NEEDS_JUDGE" = "1" ] && [ "$JUDGE_OFFNODE" = "1" ]; then
    echo "Judge node:   $JUDGE_NODE ($JUDGE_NODE_IP)"
    echo "  Judge:      GPUs $RM_GPU_LIST ($NUM_RM_GPUS GPUs, port $RM_PORT, dp=$NUM_RM_GPUS)"
fi
for n in "${!INF_NODES[@]}"; do
    echo "Inf node $n:    ${INF_NODES[$n]} (${INF_NODE_IPS[$n]})"
    echo "  Inference:  GPUs 0..$((NUM_INF_NODE_GPUS - 1)) ($NUM_INF_NODE_GPUS instances, ports $PORT_BASE..$((PORT_BASE + NUM_INF_NODE_GPUS - 1)))"
done
echo "Total policy: $NUM_INF_INSTANCES independent dp=1 instances"
echo "Broadcast:    $WEIGHT_BROADCAST$([ "$WEIGHT_BROADCAST" = nccl ] && echo " (trainer binds 0.0.0.0:$BROADCAST_PORT, workers dial $MAIN_NODE_IP, world $NUM_INF_INSTANCES+1)")"
ENVS_DESC=""
[ "$EMIT_MATH_EN"   = "1" ] && ENVS_DESC="$ENVS_DESC english-math,"
[ "$EMIT_MATH_FI"   = "1" ] && ENVS_DESC="$ENVS_DESC finnish-math,"
[ "$EMIT_IFEVAL_EN" = "1" ] && ENVS_DESC="$ENVS_DESC ifeval-en,"
[ "$EMIT_IFEVAL_FI" = "1" ] && ENVS_DESC="$ENVS_DESC ifeval-fi,"
[ "$EMIT_JUDGE_EN"  = "1" ] && ENVS_DESC="$ENVS_DESC rubric-judge-en,"
[ "$EMIT_JUDGE_FI"  = "1" ] && ENVS_DESC="$ENVS_DESC rubric-judge-fi,"
# A bare `[ cond ] && VAR=...` as the last statement of a loop body hands its exit
# status to the loop, and `set -e` then kills the script on the final failing test.
# That bug shipped once already — keep these as `if` blocks.
if [ "$EMIT_REP_EN" = "1" ]; then
    for CATEGORY in $REPETITION_CATEGORIES; do
        ENVS_DESC="$ENVS_DESC repetition-$CATEGORY,"
        if [ "$EMIT_REP_FI" = "1" ]; then
            ENVS_DESC="$ENVS_DESC repetition-$CATEGORY-fi,"
        fi
    done
    ENVS_DESC="${ENVS_DESC%,} (reward_mode=$REPETITION_REWARD_MODE),"
fi
ENVS_DESC="${ENVS_DESC# }"
echo "Envs:         ${ENVS_DESC%,}"
echo "Model:        $SFT_MODEL"
[ "$NEEDS_JUDGE" = "1" ] && echo "Judge model:  $JUDGE_MODEL ($JUDGE_URL)"
echo "Seq len:      $INF_SEQ_LEN"
echo "Temp:         $ROLLOUT_TEMP"
echo "Rollouts:     $ROLLOUTS"
echo "LR:           $LR"
echo "KL tau:       $KL_TAU"
echo "IPO mask:     $IPO_MASK_LOW / $IPO_MASK_HIGH"
echo "Adv tau:      $ADV_TAU"
echo "Batch size:   $BATCH_SIZE (oversampling ${OVERSAMPLING_FACTOR:-1.0}, pre-batch zero-adv filter ON)"
echo "Trainer:      impl=$TRAINER_MODEL_IMPL attn=$TRAINER_ATTN ep=${EP:-auto}"
echo "Lang reward:  weight $LANGUAGE_REWARD_WEIGHT${LANGUAGE_REWARD_SCOPE:+ scope $LANGUAGE_REWARD_SCOPE}"
echo "Steps:        $MAX_STEPS (ckpt every $CKPT_INTERVAL, weights_only=$CKPT_WEIGHTS_ONLY)"
if [ "$AUTO_RESUME" = "1" ]; then
    if [ "$RESTARTED" = "1" ]; then
        echo "Resume:       ON — restart #${SLURM_RESTART_COUNT:-?}, resuming from the latest checkpoint"
    else
        echo "Resume:       ON — first start, no checkpoint to resume from yet"
    fi
fi
echo "Container:    $IMG"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ── Generate one inference config per instance ──
# Each instance is a plain dp=1 server; only the port differs.
write_infer_toml() {
    cat > "$1" << TOMLEOF
enable_fp32_lm_head = ${FP32_LM_HEAD:-false}

[model]
name = "$SFT_MODEL"
max_model_len = $INF_SEQ_LEN

[parallel]
tp = 1

[server]
port = $2

[weight_broadcast]
type = "$WEIGHT_BROADCAST"
$(if [ -n "${VLLM_MODEL_IMPL:-}" ]; then echo "[vllm_extra]"; echo "model_impl = \"$VLLM_MODEL_IMPL\""; fi)
TOMLEOF
}

for i in $(seq 0 $((NUM_INF_GPUS - 1))); do
    write_infer_toml "$OUTPUT_DIR/infer_main_${i}.toml" $((PORT_BASE + i))
done
# Ports repeat per node — each inference node has its own IP, so PORT_BASE+i
# never collides across nodes.
for n in "${!INF_NODES[@]}"; do
    for i in $(seq 0 $((NUM_INF_NODE_GPUS - 1))); do
        write_infer_toml "$OUTPUT_DIR/infer_node_${n}_${i}.toml" $((PORT_BASE + i))
    done
done

# ── Collect every policy inference URL for the orchestrator ──
INF_URLS=()
for i in $(seq 0 $((NUM_INF_GPUS - 1))); do
    INF_URLS+=("http://127.0.0.1:$((PORT_BASE + i))/v1")
done
for n in "${!INF_NODES[@]}"; do
    for i in $(seq 0 $((NUM_INF_NODE_GPUS - 1))); do
        INF_URLS+=("http://${INF_NODE_IPS[$n]}:$((PORT_BASE + i))/v1")
    done
done
CLIENT_BASE_URL_TOML="["
for URL in "${INF_URLS[@]}"; do
    CLIENT_BASE_URL_TOML="${CLIENT_BASE_URL_TOML}\"${URL}\", "
done
CLIENT_BASE_URL_TOML="${CLIENT_BASE_URL_TOML%, }]"

cat > "$OUTPUT_DIR/rm_infer.toml" << TOMLEOF
[model]
name = "$JUDGE_MODEL"
max_model_len = $RM_SEQ_LEN

[parallel]
tp = 1

[server]
port = $RM_PORT
TOMLEOF

# Resume needs BOTH sides: validation.py:205-212 rejects a mismatched interval or
# resume_step, and orchestrator.ckpt defaulting to None is the single reason no run of
# this launcher has ever resumed. The orchestrator checkpoint is only a progress.pt
# (step, total_tokens, total_samples) — it holds no dataset iterator position, so a
# resumed run re-shuffles its data from the top. Its job is keeping the orchestrator
# step counter aligned with the trainer's for weight-broadcast sync.
# wait_for_weights_timeout covers the restart racing the trainer's weight write.
ORCH_CKPT_TOML=""
if [ "$AUTO_RESUME" = "1" ]; then
    ORCH_CKPT_TOML=$(printf '\n[ckpt]\ninterval = %s\nkeep_last = %s\nresume_step = -1\nwait_for_weights_timeout = 1800\n' \
        "$CKPT_INTERVAL" "$KEEP_LAST")
fi

cat > "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
batch_size = $BATCH_SIZE
group_size = $ROLLOUTS
max_steps = $MAX_STEPS
max_off_policy_steps = $MAX_OFF_POLICY
seq_len = $INF_SEQ_LEN
$(if [ -n "$OVERSAMPLING_FACTOR" ]; then echo "oversampling_factor = $OVERSAMPLING_FACTOR"; fi)

[model]
name = "$SFT_MODEL"

[model.client]
dp_rank_count = 1
base_url = $CLIENT_BASE_URL_TOML

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-orch"
offline = $WANDB_OFFLINE

[train.sampling]
temperature = $ROLLOUT_TEMP
max_completion_tokens = $MAX_TOKENS

[renderer]
name = "$RENDERER"
$ORCH_CKPT_TOML
TOMLEOF

# math_env_v1 has no system_prompt argument, so NO_SYSTEM_PROMPT does not apply to
# the math envs — they never prepend one.
write_math_env() {  # <name> <dataset> <question_key>
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "math-env"
name = "$1"

[train.env.args]
dataset_name = "$2"
dataset_subset = "default"
question_key = "$3"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60
TOMLEOF
}

write_ifeval_env() {  # <env_id> <name> <dataset>
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "$1"
name = "$2"

[train.env.args]
dataset_name = "$3"
reasoning_model = $REASONING_MODEL
language_reward_weight = $LANGUAGE_REWARD_WEIGHT
$(if [ -n "$LANGUAGE_REWARD_SCOPE" ]; then echo "language_reward_scope = \"$LANGUAGE_REWARD_SCOPE\""; fi)
$(if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then echo 'system_prompt = ""'; fi)
TOMLEOF
}

write_judge_env() {  # <name> <dataset> <use_finnish>
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "rubric-judge-env"
name = "$1"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "$JUDGE_URL"
judge_prompt_style = "$JUDGE_PROMPT_STYLE"
max_judge_tokens = $MAX_JUDGE_TOKENS
dataset_path = "$2"
use_finnish = $3
$(if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then echo 'system_prompt = ""'; fi)
TOMLEOF
}

write_repetition_env() {  # <name> <category> <dataset> <use_finnish>
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "repetition-env"
name = "$1"

[train.env.args]
dataset_path = "$3"
categories = ["$2"]
use_finnish = $4
reward_mode = "$REPETITION_REWARD_MODE"
min_passrate = $REPETITION_MIN_PASSRATE
$(if [ "${NO_SYSTEM_PROMPT:-0}" = "1" ]; then echo 'system_prompt = ""'; fi)
TOMLEOF
}

# "legacy" = our score-first prompt (1-100, no deliberation, 64 tokens is enough).
# "open-instruct" = the verbatim open-instruct templates (1-10, reasoning-first JSON),
# which need room to reason before the score appears.
JUDGE_PROMPT_STYLE="${JUDGE_PROMPT_STYLE:-legacy}"
if [ "$JUDGE_PROMPT_STYLE" = "open-instruct" ]; then
  MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-512}"
else
  MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-64}"
fi

[ "$EMIT_MATH_EN"   = "1" ] && write_math_env   "english-math" "$EN_MATH_DATASET" "${MATH_QUESTION_KEY:-problem}"
[ "$EMIT_MATH_FI"   = "1" ] && write_math_env   "finnish-math" "$FI_MATH_DATASET" "$FI_MATH_QUESTION_KEY"
[ "$EMIT_IFEVAL_EN" = "1" ] && write_ifeval_env "dolci-ifeval-env"    "ifeval-en" "$EN_IFEVAL_DATASET"
[ "$EMIT_IFEVAL_FI" = "1" ] && write_ifeval_env "dolci-ifeval-env-fi" "ifeval-fi" "$FI_IFEVAL_DATASET"
[ "$EMIT_JUDGE_EN"  = "1" ] && write_judge_env  "rubric-judge-en" "$EN_JUDGE_DATASET" false
[ "$EMIT_JUDGE_FI"  = "1" ] && write_judge_env  "rubric-judge-fi" "$FI_JUDGE_DATASET" true

# One replica per category per language. Separate envs give separate reward traces,
# which is the only way to see which prompt distribution is carrying the signal.
if [ "$EMIT_REP_EN" = "1" ]; then
    for CATEGORY in $REPETITION_CATEGORIES; do
        write_repetition_env "repetition-$CATEGORY" "$CATEGORY" "$REPETITION_DATASET" false
        if [ "$EMIT_REP_FI" = "1" ]; then
            write_repetition_env "repetition-$CATEGORY-fi" "$CATEGORY" "$REPETITION_FI_DATASET" true
        fi
    done
fi
true  # the last short-circuit above must not decide this script's exit status

# No in-loop eval: aime25 is 30 problems at 1 sample, so its score is dominated by
# sampling noise, and ifbench costs generation without informing the run. Evaluate
# saved checkpoints out-of-loop instead (aime25 at avg@64).

# Setting pre_batch_filters replaces the default list wholesale, so all three
# types are re-listed. zero_advantage enforces here: degenerate groups are
# dropped before they reach pending_batch, so the orchestrator backfills them
# with informative groups instead of shipping a shrunken batch.
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

# Appended last: a table header here would otherwise swallow the trailing keys of
# whichever [[...]] block was written before it.
#
# inference_world_size counts every inference GPU across every node. Each instance
# is a dp=1/tp=1 server, so it is exactly NUM_INF_INSTANCES; init_nccl_broadcast
# divides it by the admin-client count to get the per-server rank offset, which
# lands on 1 GPU per server. The rendezvous is inference_world_size + 1, trainer
# at rank 0.
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[weight_broadcast]
type = "$WEIGHT_BROADCAST"
$(if [ "$WEIGHT_BROADCAST" = "nccl" ]; then
    echo "host = \"$MAIN_NODE_IP\""
    echo "port = $BROADCAST_PORT"
    echo "inference_world_size = $NUM_INF_INSTANCES"
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
attn = "$TRAINER_ATTN"
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
weights_only = $CKPT_WEIGHTS_ONLY
$(if [ "$AUTO_RESUME" = "1" ]; then echo "resume_step = -1"; fi)

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
    echo "host = \"0.0.0.0\""
    echo "port = $BROADCAST_PORT"
    echo "inference_world_size = $NUM_INF_INSTANCES"
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

# ── Fetch the fastText LID model used by the language reward ──
# No-op when it is already there. Without it every language score is a neutral 1.0,
# i.e. the reward silently does nothing — so fetch before launch, not lazily per worker.
if [ "$LANGUAGE_REWARD_WEIGHT" != "0" ]; then
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && ${FASTTEXT_LID_PATH:+export FASTTEXT_LID_PATH=$FASTTEXT_LID_PATH; }uv run python -m language_reward"
    [ -n "${FASTTEXT_LID_PATH:-}" ] && echo "export FASTTEXT_LID_PATH=$FASTTEXT_LID_PATH" >> "$ENV_SETUP"
fi

# ── Launch policy inference on the dedicated nodes (one dp=1 server per GPU) ──
INF_NODE_PIDS=()
for n in "${!INF_NODES[@]}"; do
    echo "Starting $NUM_INF_NODE_GPUS inference instances on ${INF_NODES[$n]}..."
    srun --nodes=1 --ntasks=1 --nodelist="${INF_NODES[$n]}" --job-name=inference-node-$n --exclusive \
        /usr/bin/singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "
            source $ENV_SETUP
            for i in \$(seq 0 $((NUM_INF_NODE_GPUS - 1))); do
                HIP_VISIBLE_DEVICES=\$i uv run inference \
                    @ $OUTPUT_DIR/infer_node_${n}_\${i}.toml \
                    --parallel.dp 1 \
                    --gpu-memory-utilization 0.85 \
                    > $OUTPUT_DIR/inference_node_${n}_\${i}.log 2>&1 &
            done
            wait
        " &
    INF_NODE_PIDS+=($!)
done

# ── Launch policy inference on the shared node (one dp=1 server per GPU) ──
echo "Starting $NUM_INF_GPUS inference instances on $MAIN_NODE..."
INF_MAIN_PIDS=()
for i in $(seq 0 $((NUM_INF_GPUS - 1))); do
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && \
            export HIP_VISIBLE_DEVICES=$i && \
            uv run inference \
            @ $OUTPUT_DIR/infer_main_${i}.toml \
            --parallel.dp 1 \
            --gpu-memory-utilization 0.85" \
        > "$OUTPUT_DIR/inference_main_${i}.log" 2>&1 &
    INF_MAIN_PIDS+=($!)
done

RM_PID=
# ── Launch LLM judge server on RM GPUs (only when a judge env is enabled) ──
# One dp=$NUM_RM_GPUS server, unlike the policy side: the judge is addressed by a
# single rm_server_url and vLLM's own DP router fans requests out behind it.
if [ "$NEEDS_JUDGE" = "1" ]; then
    JUDGE_CMD="source $ENV_SETUP && \
        export HIP_VISIBLE_DEVICES=$RM_GPU_LIST && \
        uv run inference \
        @ $OUTPUT_DIR/rm_infer.toml \
        --parallel.dp $NUM_RM_GPUS \
        --gpu-memory-utilization 0.85"
    if [ "$JUDGE_OFFNODE" = "1" ]; then
        echo "Starting judge server on $JUDGE_NODE, GPUs $RM_GPU_LIST..."
        srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --job-name=judge --exclusive \
            /usr/bin/singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
            bash -c "$JUDGE_CMD" > "$OUTPUT_DIR/rm_inference.log" 2>&1 &
    else
        echo "Starting judge server on $MAIN_NODE, GPUs $RM_GPU_LIST..."
        singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
            bash -c "$JUDGE_CMD" > "$OUTPUT_DIR/rm_inference.log" 2>&1 &
    fi
    RM_PID=$!
fi

cleanup() {
    kill ${INF_NODE_PIDS[@]:-} ${INF_MAIN_PIDS[@]:-} ${RM_PID:-} 2>/dev/null || true
}

# ── Wait for every policy inference instance to become healthy ──
echo "Waiting for $NUM_INF_INSTANCES policy inference instances..."
for URL in "${INF_URLS[@]}"; do
    HEALTH="${URL%/v1}/health"
    for attempt in $(seq 1 600); do
        curl -sf "$HEALTH" &>/dev/null && break
        sleep 2
    done
    curl -sf "$HEALTH" &>/dev/null \
        || { echo "ERROR: $HEALTH did not become healthy within 1200s"; cleanup; exit 1; }
    echo "  $HEALTH ready."
done
echo "All policy inference instances ready."

if [ "$NEEDS_JUDGE" = "1" ]; then
    echo "Waiting for judge server at $JUDGE_URL..."
    for attempt in $(seq 1 600); do
        curl -sf "$JUDGE_URL/health" &>/dev/null && break
        sleep 2
    done
    curl -sf "$JUDGE_URL/health" &>/dev/null \
        || { echo "ERROR: judge server did not start within 1200s"; cleanup; exit 1; }
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

cleanup
wait ${INF_NODE_PIDS[@]:-} ${INF_MAIN_PIDS[@]:-} ${RM_PID:-} 2>/dev/null || true

rm -rf "$JOB_MODEL_DIR"
echo "Done!"
