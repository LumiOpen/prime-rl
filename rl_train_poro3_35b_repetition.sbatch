#!/bin/bash
#SBATCH --job-name=poro3-35b-rep
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
# Requeue is wanted here, but only because this script resumes in place: a restart
# reuses the same output directory and picks up the latest checkpoint. Without that
# pairing a NODE_FAIL silently restarts at step 0 in a fresh directory.
#SBATCH --requeue
#SBATCH --output=./logs/rl_%j.out
#SBATCH --error=./logs/rl_%j.err

# =============================================================================
#   Poro3 35B-A3B — the repetition pre-stage. Run this BEFORE the main RL run.
#
#     sbatch rl_train_poro3_35b_repetition.sbatch              # uses POLICY_MODEL below
#     sbatch rl_train_poro3_35b_repetition.sbatch /path/ckpt   # or override it
#
#   Submit from the repo root (the #SBATCH --output paths are relative to it).
#
#   Then point the main run at this one's output:
#     sbatch rl_train_poro3_35b.sbatch <this-output-dir>/weights/step_100
#   The handoff is manual on purpose — look at the evals first.
#
#   This file is standalone on purpose. Every value below is the value we run;
#   there are no feature flags and no environment overrides. To change the recipe,
#   edit the recipe. The generic launcher with all the knobs is
#   rl_train_multinode_mixed.sbatch — use that for experiments, not this.
#
#   What it does. Two envs, repetition-math and repetition-math-fi, both on Dolci
#   math prompts, scored on nothing but `1 - block_coverage`: the fraction of the
#   completion covered by verbatim repeated 30-word blocks. Correctness is not in
#   the reward at all. That is deliberate — a loop is a decoding pathology, and
#   mixing it with a correctness signal makes it impossible to tell which moved.
#
#   Why it is worth 100 steps of 64 GPUs. On the 4B model (job 53550, 300 steps)
#   this stage collapsed eval truncation across the board and cost nothing:
#
#              cap-hit   init -> step_100 -> step_300      score  init -> step_300
#     aime25    13.4%       0.5%       0.6%               0.4245 -> 0.4271
#     ifeval    10.5%       5.9%       3.7%               0.7412 -> 0.7745
#     ifbench   29.3%      11.7%      13.7%               0.2967 -> 0.3267
#     gpqa      19.7%       0.0%       0.0%               0.2677 -> 0.2929
#
#   Nearly all of it lands by step 100, which is why this recipe stops there. The
#   score gains are auto-zeros turning into attempts, not better answers — the
#   point is that a truncated rollout scores zero on every downstream eval AND
#   manufactures false advantage during RL, so loops corrupt the main run's
#   gradient signal, not just its eval numbers.
#
#   Math prompts only. The math+ifeval variant (job 53961) bought +0.08 ifeval but
#   put AIME *below* init (0.3807 vs 0.4245) and pushed math cap-hit 0.4% -> 4.4%,
#   and it still did not fix ifbench. The main run trains instruction-following
#   directly; this stage should be capability-neutral, so it stays on math.
#
#   GPU layout — 64 GPUs, no judge (this env does not score with one):
#     node 0    8  trainer      ep=1, dp_shard=8. Also runs the orchestrator.
#     nodes 1-7 56 inference    56 independent dp=1/tp=1 vLLM servers
#
#   Watch two traces together: reward/repetition-*/mean must rise AND
#   is_truncated/all/mean must fall, with seq_len/all/mean roughly stable. Block
#   coverage is computed over *exact* 30-word blocks, so perturbing one token per
#   repetition drives reward to 1.0 while the model still loops. Reward rising on
#   its own is the reward hack, not success.
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

RUN_NAME="poro3-35b-a3b-rep-enfi-bs1024-r8-32k-temp0.6"
CONTAINER="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"

# The repetition env reads Dolci through rubric_judge_env's loader, so the Finnish
# side is the same translated jsonl the FI judge uses — use_finnish just switches
# which column it reads. Both return 30,180 math rows (27,992 after the passrate
# filter), and 100 steps needs 6,400 groups per env, so nothing repeats.
DATASETS="/shared_silo/scratch/datasets/RL"
EN_DATASET="$DATASETS/Dolci-Think-RL-7B"
FI_DATASET="$DATASETS/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi/Dolci-Think-RL-7B-prompt-ground-truth-translated-fi.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
#   The recipe — identical to the main run except where noted
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE=1024           # rollouts, not prompts: 128 groups per step, 64 per env
GROUP_SIZE=8

# The cap has to stay at the main run's value. This stage calibrates behaviour
# against a 32768-token ceiling, and a shorter one would train away a loop the real
# run never had room to produce.
SEQ_LEN=32768
MAX_COMPLETION_TOKENS=32768

# 0.6, against the main run's 1.0, and this is the one place that asymmetry is
# right: sharpened decoding is what *provokes* the loops (proven in job 47309 —
# 22.6% eval cap-hit at 0.6/0.95 against 2.0% at 1.0/1.0), so rolling out at 0.6
# trains the failure mode where it actually lives. Every repetition run used 0.6.
TEMPERATURE=0.6

OVERSAMPLING_FACTOR=2.0
LR=1e-6
KL_TAU=0                  # trainer-vs-inference mismatch penalty, not a ref-policy anchor
DPPO_MASK_LOW=0.2
DPPO_MASK_HIGH=0.2
ADV_TAU=2.0
MAX_OFF_POLICY_STEPS=8

# 100 steps. 53550 had captured essentially the whole effect by step 100 and the
# next 200 added little; and with no reference-policy anchor anywhere in the loss,
# every extra step is unguarded drift away from the SFT init.
MAX_STEPS=100

# Interval 25, not the main run's 100, for the same reason: nothing here protects
# math ability, so the useful checkpoint may not be the last one. Four save points
# at 456 G each is 1.8 T.
CKPT_INTERVAL=25
CKPT_KEEP_LAST=4          # = MAX_STEPS / CKPT_INTERVAL: keep every one, delete nothing

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
BROADCAST_PORT=29501      # NCCL weight-broadcast rendezvous, prime-rl's default

[ -f "$PRIME_RL_DIR/.venv/bin/activate" ] || {
    echo "ERROR: no .venv in $PRIME_RL_DIR"
    echo "       Submit this from the prime-rl repo root, and run setup.sh first."
    exit 1
}
[ -d "$POLICY_MODEL" ] || {
    echo "ERROR: policy model not found: $POLICY_MODEL"
    echo "       Edit POLICY_MODEL in this file, or pass the path as the first argument."
    exit 1
}
[ -f "$CONTAINER" ] || { echo "ERROR: container image not found: $CONTAINER"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
UV=$(command -v uv) || { echo "ERROR: uv not found in PATH"; exit 1; }

# node 0 trainer, nodes 1-7 inference.
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#NODES[@]}" -eq 8 ] || { echo "ERROR: this recipe needs exactly 8 nodes, got ${#NODES[@]}"; exit 1; }
TRAIN_NODE="${NODES[0]}"
INF_NODES=("${NODES[@]:1}")
NUM_INF_INSTANCES=$(( ${#INF_NODES[@]} * 8 ))   # 56

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
INF_NODE_IPS=()
for NODE in "${INF_NODES[@]}"; do INF_NODE_IPS+=("$(node_ip "$NODE")"); done

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
# An `if`, not `[ -f ] && ln`: a trailing short-circuit hands its exit status to the
# loop and `set -e` would kill the script on a directory entry.
JOB_MODEL_DIR="$PRIME_RL_DIR/models/job_${SLURM_JOB_ID}_model"
mkdir -p "$JOB_MODEL_DIR"
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
echo "  Poro3 35B-A3B repetition pre-stage: $FULL_RUN_NAME"
echo "============================================"
echo "Job:          $SLURM_JOB_ID"
echo "Trainer:      $TRAIN_NODE ($TRAIN_NODE_IP) GPUs 0-7, ep=$TRAINER_EP, impl=$TRAINER_IMPL"
for n in "${!INF_NODES[@]}"; do
    echo "Inference:    ${INF_NODES[$n]} (${INF_NODE_IPS[$n]}) GPUs 0-7, ports $INF_PORT_BASE-$((INF_PORT_BASE + 7))"
done
echo "Policy:       $NUM_INF_INSTANCES independent dp=1 instances (no judge)"
echo "Broadcast:    nccl — trainer binds 0.0.0.0:$BROADCAST_PORT, workers dial $TRAIN_NODE_IP, world $((NUM_INF_INSTANCES + 1))"
echo "Envs:         repetition-math, repetition-math-fi (reward_mode=blockcov)"
echo "Model:        $POLICY_MODEL"
echo "Batch:        $BATCH_SIZE rollouts / $GROUP_SIZE per group, oversampling $OVERSAMPLING_FACTOR"
echo "Seq len:      $SEQ_LEN, temp $TEMPERATURE"
echo "Steps:        $MAX_STEPS, ckpt every $CKPT_INTERVAL (full state), resume in place"
echo "Output:       $OUTPUT_DIR"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────────────
#   Inference configs — 56 identical servers, only the port differs
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

# ─────────────────────────────────────────────────────────────────────────────
#   Orchestrator config
# ─────────────────────────────────────────────────────────────────────────────

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

# Two replicas of one env, English and Finnish, kept separate so their reward
# traces stay separable — the only way to see which language carries the signal.
# min_passrate 0.05 drops exactly Dolci's passrate == 0 rows (the next value up is
# 1/8); job 53239 showed those loop least and run 3k tokens shorter. The empty
# system prompt matches the main run, so behaviour is calibrated under the same
# rendering the real RL will use.
cat >> "$OUTPUT_DIR/orch_config.toml" << TOMLEOF

[[train.env]]
id = "repetition-env"
name = "repetition-math"

[train.env.args]
dataset_path = "$EN_DATASET"
categories = ["math"]
use_finnish = false
reward_mode = "blockcov"
min_passrate = 0.05
system_prompt = ""

[[train.env]]
id = "repetition-env"
name = "repetition-math-fi"

[train.env.args]
dataset_path = "$FI_DATASET"
categories = ["math"]
use_finnish = true
reward_mode = "blockcov"
min_passrate = 0.05
system_prompt = ""
TOMLEOF

# Setting pre_batch_filters replaces the default list wholesale, so all three are
# re-listed. zero_advantage enforces: degenerate groups are dropped before they
# reach pending_batch, so the orchestrator backfills them with informative groups
# instead of shipping a shrunken batch. Expect this to bite harder here than in the
# main run — if the init rarely loops, most groups come back with identical block
# coverage and get dropped, and the orchestrator oversamples hunting for loopy
# groups. That shows up as slow steps, not as an error.
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
# and must accept connections from seven other nodes, where "localhost" would only
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

cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }

echo "Waiting for $NUM_INF_INSTANCES policy instances (up to 1200s each)..."
for URL in "${INF_URLS[@]}"; do
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
