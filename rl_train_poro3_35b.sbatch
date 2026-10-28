#!/bin/bash
#SBATCH --job-name=poro3-35b-rl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
# Requeue is wanted here, but only because this script resumes in place: a restart
# reuses the same output directory and picks up the latest checkpoint. Without that
# pairing a NODE_FAIL silently restarts at step 0 in a fresh directory.
#SBATCH --requeue
#SBATCH --output=./logs/rl_%j.out
#SBATCH --error=./logs/rl_%j.err

# =============================================================================
#   Poro3 35B-A3B full RL run — the production recipe.
#
#     sbatch rl_train_poro3_35b.sbatch                    # uses POLICY_MODEL below
#     sbatch rl_train_poro3_35b.sbatch /path/to/ckpt      # or override it
#
#   Submit from the repo root (the #SBATCH --output paths are relative to it).
#
#   This file is standalone on purpose. Every value below is the value we run;
#   there are no feature flags, no alternative code paths and no environment
#   overrides, because a flag you have to remember is a flag you will eventually
#   get wrong. To change the recipe, edit the recipe. The generic launcher with
#   all the knobs is rl_train_multinode_mixed.sbatch — use that for experiments,
#   not this.
#
#   Run rl_train_poro3_35b_repetition.sbatch FIRST. It is a 100-step pass that
#   trains out reasoning loops, and POLICY_MODEL below should be its output
#   (<rep-output-dir>/weights/step_100), not the raw SFT checkpoint. Truncated
#   rollouts do not merely lose eval points, they fabricate advantage during RL,
#   so the loops are worth removing before this run rather than after.
#
#   Six envs, three families x two languages:
#     english-math / finnish-math        klear_math_30k_{en,fi}. The same problems
#                                        and the same answer labels; the Finnish
#                                        env just reads a translated column.
#     ifeval-en / ifeval-fi              Dolci-Think-RL-7B / ifeval-fi
#     rubric-judge-en / rubric-judge-fi  Dolci + its FI translation, judged by a
#                                        Qwen3.8-27B server we host ourselves.
#   No per-env weights exist, so the batch splits evenly: ~21 groups each.
#
#   GPU layout — 64 GPUs, every node single-purpose:
#     node 0    8  trainer      ep=1, dp_shard=8, no dp_replicate, no offload.
#                               Also runs the orchestrator (CPU-bound).
#     node 1    8  judge        one dp=8 vLLM server for Qwen3.8-27B
#     nodes 2-7 48 inference    48 independent dp=1/tp=1 vLLM servers
#
#   Why 8/8/48. The trainer sets the floor: 8 GPUs at 4,078 tok/s/GPU (the best
#   per-GPU figure measured for this model) take ~260 s for a 1024x8 batch at 32k.
#   Trainer and orchestrator overlap, so the step is max(trainer, inference) and
#   inference only has to come in under 260 s — 48 GPUs do it in ~242 s at the
#   measured 11.3 GPU-s/rollout with oversampling. Adding inference past that buys
#   nothing. Keeping the trainer inside one node matters: dp_shard crossing a node
#   boundary collapsed MFU to 3.4% in testing. The judge gets a whole node because
#   its latency is on the rollout critical path — ~1/3 of 1024 rollouts are
#   judge-scored and each prefills up to 32k tokens. All three arms scale linearly
#   in batch size, so this split is batch-independent.
#
#   Every policy inference instance is a separate single-GPU server with its own
#   port, and the orchestrator round-robins over the full URL list. That is what
#   makes an uneven layout expressible at all: client.dp_rank_count is one scalar
#   applied to every URL, so a mixed 8-and-4 GPU split has no valid value.
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#   Paths
# ─────────────────────────────────────────────────────────────────────────────

# The directory you ran sbatch from, which must be the repo root. Not $0 or
# BASH_SOURCE: Slurm executes a spooled *copy* of this file, so those point into
# /var/spool. SLURM_SUBMIT_DIR is correct and survives a requeue.
PRIME_RL_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset — submit this with sbatch}"

# The one thing to edit. If the SFT checkpoint is elsewhere, pass it as $1.
POLICY_MODEL="${1:-/shared_silo/scratch/models/poro3-qwen3_5-35B-A3B-sft}"

RUN_NAME="poro3-35b-a3b-6env-enfi-bs1024-r8-32k"
JUDGE_MODEL="/shared_silo/scratch/models/Qwen3.8-27B"
CONTAINER="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"

DATASETS="/shared_silo/scratch/datasets/RL"
EN_MATH_DATASET="$DATASETS/klear_math_30k_en"
FI_MATH_DATASET="$DATASETS/klear_math_30k_fi"
EN_IFEVAL_DATASET="$DATASETS/Dolci-Think-RL-7B"
FI_IFEVAL_DATASET="$DATASETS/ifeval-fi/fi_dataset.jsonl"
EN_JUDGE_DATASET="$DATASETS/Dolci-Think-RL-7B"
FI_JUDGE_DATASET="$DATASETS/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
#   The recipe
# ─────────────────────────────────────────────────────────────────────────────

# BATCH_SIZE counts rollouts, not prompts: 1024 / 8 = 128 groups per step. Raising
# the batch, rather than the oversampling factor, is how you keep 48 inference
# instances fed — max_inflight is a fleet-wide cap, not a per-instance one.
BATCH_SIZE=1024
GROUP_SIZE=8

# 32768 everywhere, and max_completion_tokens == seq_len as in every 35B run so
# far. The prompt therefore gets no reserved headroom and a long prompt eats into
# the generation budget.
SEQ_LEN=32768
MAX_COMPLETION_TOKENS=32768
JUDGE_SEQ_LEN=32768

# 1.0, not the 0.6 used in the repetition study: the policy gradient assumes
# rollouts are drawn from the policy, and sharpened decoding biases both the
# gradient and every before/after comparison against an eval at a different
# temperature.
TEMPERATURE=1.0

# batch_size * 2 rollouts in flight. Measured ~17% off the mean step time versus
# 1.0, and it is what keeps inference busy once the zero-advantage pre-batch
# filter starts dropping groups.
OVERSAMPLING_FACTOR=2.0

LR=1e-6
KL_TAU=0                  # trainer-vs-inference mismatch penalty, not a ref-policy anchor
DPPO_MASK_LOW=0.2
DPPO_MASK_HIGH=0.2

# The config default. adv_tau weights the policy-gradient term against the kl_tau
# mismatch term, so with kl_tau = 0 it just scales the whole loss — under AdamW that
# is nearly a no-op on the update direction, but it does scale the raw gradient norm
# and therefore how often max_norm = 1.0 clips. Our 4B runs all used 2.0, inherited
# from the dolci config; there is no reason to clip twice as hard on purpose.
ADV_TAU=1.0

MAX_OFF_POLICY_STEPS=8

# A ceiling, not a target — the run ends when the eval curve flattens. ~260 s/step
# puts 2000 steps at ~145 h, comfortably inside the 14-day wall. 2000 x 128 =
# 256,000 groups over six envs is ~42,700 each, so the ~30k-row envs (klear math
# en/fi, ifeval-fi) wrap about 1.4x. Repeats under a different policy are not
# retraining on stale data, but the second pass is cheaper evidence than the first.
MAX_STEPS=2000

# A save point for this model is 456 G: 66 G of HF weights plus a 390 G DCP
# training state. Full states are what in-place resume needs, and at interval 100
# over 2000 steps that is 20 points x 456 G = 9.1 T, none of which keep_last ever
# prunes. Budget accordingly — /shared_silo was at 94% (~15 T free) when this was
# written.
CKPT_INTERVAL=100
CKPT_KEEP_LAST=20         # = MAX_STEPS / CKPT_INTERVAL: keep every one, delete nothing

# impl=custom is required for ep to take effect at all, and ep must be explicit:
# this checkpoint is a VL-wrapped composite whose num_experts lives under
# text_config, so ep="auto" silently resolves to 1 without warning. ep=1 is also
# what we want — ep equal to a node's GPU count makes dp_shard_mod_ep degenerate
# and the trainer hangs at 100% util with zero steps.
TRAINER_IMPL=custom
TRAINER_ATTN=flash_attention_2
TRAINER_EP=1

WANDB_PROJECT=reasoning

# ─────────────────────────────────────────────────────────────────────────────
#   Fixed infrastructure
# ─────────────────────────────────────────────────────────────────────────────

INF_PORT_BASE=8000        # inference instance i on node n listens on 8000+i
JUDGE_PORT=8100           # above the inference range, so it can never collide
BROADCAST_PORT=29501      # NCCL weight-broadcast rendezvous, prime-rl's default

[ -d "$POLICY_MODEL" ] || {
    echo "ERROR: policy model not found: $POLICY_MODEL"
    echo "       Edit POLICY_MODEL in this file, or pass the path as the first argument."
    exit 1
}
[ -d "$JUDGE_MODEL" ] || { echo "ERROR: judge model not found: $JUDGE_MODEL"; exit 1; }
[ -f "$PRIME_RL_DIR/.venv/bin/activate" ] || {
    echo "ERROR: no .venv in $PRIME_RL_DIR"
    echo "       Submit this from the prime-rl repo root, and run setup.sh first."
    exit 1
}
[ -f "$CONTAINER" ] || { echo "ERROR: container image not found: $CONTAINER"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
UV=$(command -v uv) || { echo "ERROR: uv not found in PATH"; exit 1; }

# node 0 trainer, node 1 judge, nodes 2-7 inference.
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#NODES[@]}" -eq 8 ] || { echo "ERROR: this recipe needs exactly 8 nodes, got ${#NODES[@]}"; exit 1; }
TRAIN_NODE="${NODES[0]}"
JUDGE_NODE="${NODES[1]}"
INF_NODES=("${NODES[@]:2}")
NUM_INF_INSTANCES=$(( ${#INF_NODES[@]} * 8 ))   # 48

# eno0 deliberately matches NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME below. The check
# lives inside the function rather than after the assignments: under `set -e` a grep
# that matches nothing already kills the script at the assignment, so a guard further
# down would never get to print. Exiting from the command substitution's subshell
# gives the assignment a non-zero status, which is what stops the run.
node_ip() {
    local ip
    ip=$(srun --nodes=1 --ntasks=1 -w "$1" /usr/sbin/ip -4 addr show eno0 \
         | grep -oP 'inet \K[0-9.]+' || true)
    [ -n "$ip" ] || { echo "ERROR: could not resolve the eno0 address of $1" >&2; exit 1; }
    echo "$ip"
}
TRAIN_NODE_IP=$(node_ip "$TRAIN_NODE")
JUDGE_NODE_IP=$(node_ip "$JUDGE_NODE")
INF_NODE_IPS=()
for NODE in "${INF_NODES[@]}"; do INF_NODE_IPS+=("$(node_ip "$NODE")"); done

JUDGE_URL="http://${JUDGE_NODE_IP}:${JUDGE_PORT}"

# Every inference URL, in the order the orchestrator will round-robin them. Ports
# repeat across nodes — the IPs differ, so 8000+i never collides.
INF_URLS=()
for n in "${!INF_NODES[@]}"; do
    for i in $(seq 0 7); do
        INF_URLS+=("http://${INF_NODE_IPS[$n]}:$((INF_PORT_BASE + i))/v1")
    done
done
BASE_URL_TOML="["
for URL in "${INF_URLS[@]}"; do BASE_URL_TOML="${BASE_URL_TOML}\"${URL}\", "; done
BASE_URL_TOML="${BASE_URL_TOML%, }]"

cd "$PRIME_RL_DIR"

# prime-rl writes a prime/ cache into the model directory, so jobs sharing one
# checkpoint path race each other. Give this job its own directory of symlinks.
JOB_MODEL_DIR="$PRIME_RL_DIR/models/job_${SLURM_JOB_ID}_model"
mkdir -p "$JOB_MODEL_DIR"
# An `if`, not `[ -f ] && ln`: a trailing short-circuit hands its exit status to the
# loop and `set -e` would kill the script on a directory entry.
for f in "$POLICY_MODEL"/*; do
    if [ -f "$f" ]; then ln -sf "$f" "$JOB_MODEL_DIR/$(basename "$f")"; fi
done
rm -rf "$JOB_MODEL_DIR/prime"
MODEL="$JOB_MODEL_DIR"
[ -f "$MODEL/model.safetensors.index.json" ] || [ -f "$MODEL/model.safetensors" ] \
    || { echo "ERROR: $POLICY_MODEL has no safetensors — convert it before submitting"; exit 1; }

# A requeued job re-runs this script with the same SLURM_JOB_ID but a new wall
# clock, so a bare timestamp would send the restart to an empty directory and
# silently retrain from step 0. The first start records its directory in a marker
# keyed by job id; every restart reads it back.
MARKER="$PRIME_RL_DIR/outputs/.jobdir-${SLURM_JOB_ID}"
if [ -f "$MARKER" ]; then
    OUTPUT_DIR=$(cat "$MARKER")
    echo "Requeue detected (restart #${SLURM_RESTART_COUNT:-?}): reusing $OUTPUT_DIR"
else
    OUTPUT_DIR="$PRIME_RL_DIR/outputs/${RUN_NAME}-$(date +%Y%m%d_%H%M%S)"
fi
FULL_RUN_NAME=$(basename "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR/run_default"
echo "$OUTPUT_DIR" > "$MARKER"

echo "============================================"
echo "  Poro3 35B-A3B RL: $FULL_RUN_NAME"
echo "============================================"
echo "Job:          $SLURM_JOB_ID"
echo "Trainer:      $TRAIN_NODE ($TRAIN_NODE_IP) GPUs 0-7, ep=$TRAINER_EP, impl=$TRAINER_IMPL"
echo "Judge:        $JUDGE_NODE ($JUDGE_NODE_IP) GPUs 0-7, dp=8, port $JUDGE_PORT"
for n in "${!INF_NODES[@]}"; do
    echo "Inference:    ${INF_NODES[$n]} (${INF_NODE_IPS[$n]}) GPUs 0-7, ports $INF_PORT_BASE-$((INF_PORT_BASE + 7))"
done
echo "Policy:       $NUM_INF_INSTANCES independent dp=1 instances"
echo "Broadcast:    nccl — trainer binds 0.0.0.0:$BROADCAST_PORT, workers dial $TRAIN_NODE_IP, world $((NUM_INF_INSTANCES + 1))"
echo "Envs:         english-math, finnish-math, ifeval-en, ifeval-fi, rubric-judge-en, rubric-judge-fi"
echo "Model:        $POLICY_MODEL"
echo "Judge model:  $JUDGE_MODEL"
echo "Batch:        $BATCH_SIZE rollouts / $GROUP_SIZE per group, oversampling $OVERSAMPLING_FACTOR"
echo "Seq len:      $SEQ_LEN, temp $TEMPERATURE"
echo "Steps:        $MAX_STEPS, ckpt every $CKPT_INTERVAL (full state), resume in place"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────────────
#   Inference configs — 48 identical servers, only the port differs
# ─────────────────────────────────────────────────────────────────────────────

for n in "${!INF_NODES[@]}"; do
    for i in $(seq 0 7); do
        cat > "$OUTPUT_DIR/infer_node_${n}_${i}.toml" << TOMLEOF
# Explicit because the config default is true: this monkey-patches vLLM's logits
# processor to run lm_head in fp32. Every 35B run so far has had it off, and
# turning it on now would change rollout logprobs mid-project.
enable_fp32_lm_head = false

[model]
name = "$MODEL"
max_model_len = $SEQ_LEN

[parallel]
tp = 1

[server]
port = $((INF_PORT_BASE + i))

[weight_broadcast]
type = "nccl"
TOMLEOF
    done
done

# One dp=8 server, unlike the policy side: the judge is addressed by a single
# rm_server_url and vLLM's own DP router fans requests out behind it.
cat > "$OUTPUT_DIR/judge_infer.toml" << TOMLEOF
[model]
name = "$JUDGE_MODEL"
max_model_len = $JUDGE_SEQ_LEN

[parallel]
tp = 1

[server]
port = $JUDGE_PORT
TOMLEOF

# ─────────────────────────────────────────────────────────────────────────────
#   Orchestrator config
# ─────────────────────────────────────────────────────────────────────────────

# No in-loop eval: aime25 is 30 problems at 1 sample, so its score is dominated by
# sampling noise, and ifbench costs generation without informing the run. Evaluate
# saved checkpoints out of loop instead, at avg@64.
#
# [ckpt] here is not optional bookkeeping — orchestrator.ckpt defaulting to None is
# the single reason no run of this launcher family ever resumed, and validation.py
# rejects an interval or resume_step that disagrees with the trainer's. The
# orchestrator checkpoint is only a progress.pt (step, total_tokens, total_samples)
# and holds no dataset iterator position, so a resumed run re-shuffles its data
# from the top; its job is keeping the orchestrator step counter aligned with the
# trainer's for weight-broadcast sync. wait_for_weights_timeout covers the restart
# racing the trainer's weight write.
cat > "$OUTPUT_DIR/orch_config.toml" << TOMLEOF
batch_size = $BATCH_SIZE
group_size = $GROUP_SIZE
max_steps = $MAX_STEPS
max_off_policy_steps = $MAX_OFF_POLICY_STEPS
seq_len = $SEQ_LEN
oversampling_factor = $OVERSAMPLING_FACTOR

[model]
name = "$MODEL"

[model.client]
dp_rank_count = 1
base_url = $BASE_URL_TOML

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-orch"
offline = false

[train.sampling]
temperature = $TEMPERATURE
max_completion_tokens = $MAX_COMPLETION_TOKENS

[renderer]
name = "qwen3.5"

[ckpt]
interval = $CKPT_INTERVAL
keep_last = $CKPT_KEEP_LAST
resume_step = -1
wait_for_weights_timeout = 1800
TOMLEOF

# The six envs. math_env_v1 takes no system_prompt and never prepends one; the
# other four are given an empty one explicitly, so every env renders the bare user
# turn. The judge envs leave `categories` at the env default
# ("general-quality", "general-quality_ref") and filter Dolci by passrate.
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "math-env"
name = "english-math"

[train.env.args]
dataset_name = "$EN_MATH_DATASET"
dataset_subset = "default"
question_key = "problem"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60

[[train.env]]
id = "math-env"
name = "finnish-math"

[train.env.args]
dataset_name = "$FI_MATH_DATASET"
dataset_subset = "default"
question_key = "translated_problem"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60

[[train.env]]
id = "dolci-ifeval-env"
name = "ifeval-en"

[train.env.args]
dataset_name = "$EN_IFEVAL_DATASET"
reasoning_model = true
system_prompt = ""

[[train.env]]
id = "dolci-ifeval-env-fi"
name = "ifeval-fi"

[train.env.args]
dataset_name = "$FI_IFEVAL_DATASET"
reasoning_model = true
system_prompt = ""

[[train.env]]
id = "rubric-judge-env"
name = "rubric-judge-en"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "$JUDGE_URL"
judge_prompt_style = "legacy"
max_judge_tokens = 64
dataset_path = "$EN_JUDGE_DATASET"
use_finnish = false
system_prompt = ""

[[train.env]]
id = "rubric-judge-env"
name = "rubric-judge-fi"

[train.env.args]
custom_rm = true
judge_model_path = "$JUDGE_MODEL"
rm_server_url = "$JUDGE_URL"
judge_prompt_style = "legacy"
max_judge_tokens = 64
dataset_path = "$FI_JUDGE_DATASET"
use_finnish = true
system_prompt = ""
TOMLEOF

# Setting pre_batch_filters replaces the default list wholesale, so all three are
# re-listed. zero_advantage enforces: degenerate groups are dropped before they
# reach pending_batch, so the orchestrator backfills them with informative groups
# instead of shipping a shrunken batch. This is why oversampling matters.
#
# [weight_broadcast] is appended last because a table header here would otherwise
# swallow the trailing keys of whichever block came before it. inference_world_size
# counts every inference GPU across every node; the rendezvous is that plus one,
# with the trainer at rank 0. The orchestrator advertises the trainer's real IP
# because it POSTs that string to every inference server's /init_broadcaster and
# they dial it — "0.0.0.0" is not a destination and "localhost" would send the
# off-node workers to themselves. A wrong host or world size HANGS to the health
# timeout rather than erroring.
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

[weight_broadcast]
type = "nccl"
host = "$TRAIN_NODE_IP"
port = $BROADCAST_PORT
inference_world_size = $NUM_INF_INSTANCES
TOMLEOF

# ─────────────────────────────────────────────────────────────────────────────
#   Trainer config
# ─────────────────────────────────────────────────────────────────────────────

# The trainer binds 0.0.0.0 for the broadcast rendezvous: it serves the TCP store
# and must accept connections from six other nodes, where "localhost" would only
# ever accept its own.
cat > "$OUTPUT_DIR/train_config.toml" << TOMLEOF
max_steps = $MAX_STEPS
matmul_precision = "highest"

[model]
name = "$MODEL"
seq_len = $SEQ_LEN
optim_cpu_offload = false
compile = "None"
impl = "$TRAINER_IMPL"
attn = "$TRAINER_ATTN"
ep = $TRAINER_EP

[model.ac]
freq = 1

[wandb]
project = "$WANDB_PROJECT"
name = "${FULL_RUN_NAME}-trainer"
offline = false

[ckpt]
interval = $CKPT_INTERVAL
keep_last = $CKPT_KEEP_LAST
weights_only = false
resume_step = -1

[loss]
type = "default"
kl_tau = $KL_TAU
dppo_mask_low = $DPPO_MASK_LOW
dppo_mask_high = $DPPO_MASK_HIGH
adv_tau = $ADV_TAU

[optim]
lr = $LR
max_norm = 1.0

[scheduler]
type = "constant"

[weight_broadcast]
type = "nccl"
host = "0.0.0.0"
port = $BROADCAST_PORT
inference_world_size = $NUM_INF_INSTANCES
TOMLEOF

# ─────────────────────────────────────────────────────────────────────────────
#   Container environment
# ─────────────────────────────────────────────────────────────────────────────

# Sourced by every process on every node. The caches are per-host and per-job
# because --containall gives each node its own view of /tmp.
ENV_SETUP="$OUTPUT_DIR/env_setup.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR
export PATH="$(dirname "$UV"):\$PATH"
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

SING="/usr/bin/singularity exec --rocm --cleanenv --containall"
SING="$SING -B /shared_silo/scratch:/shared_silo/scratch:rw"
SING="$SING -B $HOME:$HOME:ro"
SING="$SING -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
SING="$SING -B /usr/share/libdrm:/usr/share/libdrm:ro"
SING="$SING -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
SING="$SING -B /tmp:/tmp:rw"
SING="$SING $CONTAINER"

$SING bash -c "source $ENV_SETUP && uv sync --all-extras"
# punkt_tab is required by the Finnish IFEval env's sentence-count instructions.
$SING bash -c "source $ENV_SETUP && uv run python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\""

# ─────────────────────────────────────────────────────────────────────────────
#   Launch
# ─────────────────────────────────────────────────────────────────────────────

PIDS=()

for n in "${!INF_NODES[@]}"; do
    echo "Starting 8 inference instances on ${INF_NODES[$n]}..."
    srun --nodes=1 --ntasks=1 --nodelist="${INF_NODES[$n]}" --job-name=inference-$n --exclusive \
        $SING bash -c "
            source $ENV_SETUP
            for i in \$(seq 0 7); do
                HIP_VISIBLE_DEVICES=\$i uv run inference \
                    @ $OUTPUT_DIR/infer_node_${n}_\${i}.toml \
                    --parallel.dp 1 \
                    --gpu-memory-utilization 0.85 \
                    > $OUTPUT_DIR/inference_node_${n}_\${i}.log 2>&1 &
            done
            wait
        " &
    PIDS+=($!)
done

echo "Starting judge server on $JUDGE_NODE, GPUs 0-7..."
srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --job-name=judge --exclusive \
    $SING bash -c "
        source $ENV_SETUP
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        uv run inference \
            @ $OUTPUT_DIR/judge_infer.toml \
            --parallel.dp 8 \
            --gpu-memory-utilization 0.85
    " > "$OUTPUT_DIR/judge_inference.log" 2>&1 &
PIDS+=($!)

cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }

echo "Waiting for $NUM_INF_INSTANCES policy instances and the judge (up to 1200s each)..."
for URL in "${INF_URLS[@]}" "$JUDGE_URL/v1"; do
    HEALTH="${URL%/v1}/health"
    for _ in $(seq 1 600); do
        curl -sf "$HEALTH" &>/dev/null && break
        sleep 2
    done
    curl -sf "$HEALTH" &>/dev/null || { echo "ERROR: $HEALTH never became healthy"; cleanup; exit 1; }
    echo "  $HEALTH ready."
done

# Orchestrator and trainer share node 0: the orchestrator is CPU-bound scheduling
# work, the trainer owns all 8 GPUs. Both log to this job's Slurm output file.
echo "Starting orchestrator and trainer on $TRAIN_NODE, GPUs 0-7..."
$SING bash -c "
    source $ENV_SETUP
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
        echo 'ERROR: orchestrator config not found after 120s'
        kill \$ORCH_PID 2>/dev/null
        exit 1
    fi
    sleep 5

    uv run torchrun \
        --standalone \
        --nproc-per-node 8 \
        --log-dir $OUTPUT_DIR/logs/trainer/torchrun \
        -m prime_rl.trainer.rl.train \
        @ $OUTPUT_DIR/train_config.toml \
        --output-dir $OUTPUT_DIR
"

cleanup
wait "${PIDS[@]}" 2>/dev/null || true
rm -rf "$JOB_MODEL_DIR"
echo "Done!"
