#!/bin/bash
#SBATCH --job-name=p2-klear-s0-2env-32k
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared_silo/scratch/dzautner/prime-rl/logs/p2_klear_s0_2env_32k_%j.out
#SBATCH --error=/shared_silo/scratch/dzautner/prime-rl/logs/p2_klear_s0_2env_32k_%j.err

set -euo pipefail

# Stage 0: Fresh RL from SFT base model (no checkpoint resume)
# Same hyperparameters as the r10-2env-32k run but starting from step 0.

TRAINER_SEQ_LEN=24576
INF_SEQ_LEN=32768
MAX_TOKENS=24576
BATCH_SIZE=128
ROLLOUTS=8
LR=1e-6
KL_TAU=0.01
IPO_MASK_LOW=0.2
IPO_MASK_HIGH=0.2
ADV_TAU=2.0
ROLLOUT_TEMP=1.0
ORCH_STEPS=10000
TRAINER_STEPS=10000
CKPT_INTERVAL=10

NUM_INF_NODES=3
TOTAL_NODES=4
TOTAL_DP=24

PRIME_RL_DIR="/shared_silo/scratch/dzautner/prime-rl"
IMG="/shared_silo/scratch/dzautner/kai_repro/containers/rocm_vllm-dev_upstream_preview_releases_v0.17.0_20260303.sif"
SFT_MODEL="/shared_silo/scratch/dzautner/prime-rl/models/Poro2-Longrope-132en111fi/"
EN_DATASET="/shared_silo/scratch/dzautner/prime-rl/data/klear_math_30k_en"
FI_DATASET="/shared_silo/scratch/dzautner/prime-rl/data/klear_math_30k_fi"
WANDB_PROJECT="poro-language-rl"

RUN_NAME="p2-klear-s0-2env-32k"
OUTPUT_DIR="outputs/$RUN_NAME"

if [ "$SLURM_NNODES" -ne "$TOTAL_NODES" ]; then
    echo "ERROR: Need $TOTAL_NODES nodes but got $SLURM_NNODES"
    exit 1
fi

cd "$PRIME_RL_DIR"

mkdir -p "$OUTPUT_DIR/run_default" "$OUTPUT_DIR/checkpoints"

echo "============================================"
echo "  Poro2 Klear S0 — Fresh from SFT"
echo "============================================"
echo "Output:       $OUTPUT_DIR"
echo "Model:        $SFT_MODEL"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_NODE=${NODELIST[0]}
FIRST_INF_NODE=${NODELIST[1]}
FIRST_INF_IP=$(srun --nodes=1 --ntasks=1 -w $FIRST_INF_NODE /usr/sbin/ip -4 addr show eno0 | grep -oP 'inet \K[0-9.]+')

echo "Nodes:        $MASTER_NODE (trainer) | ${NODELIST[@]:1} (inference)"

cat > "$OUTPUT_DIR/orch_config.toml" << ORCHEOF
batch_size = $BATCH_SIZE
rollouts_per_example = $ROLLOUTS
max_steps = $ORCH_STEPS
seq_len = $TRAINER_SEQ_LEN
num_train_workers = 8
max_async_level = 1

[model]
name = "$SFT_MODEL"

[wandb]
project = "$WANDB_PROJECT"
name = "${RUN_NAME}-orch"

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
dataset_name = "$EN_DATASET"
dataset_subset = "default"
question_key = "problem"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60

[[env]]
id = "math-env"
name = "finnish-math"
[env.args]
dataset_name = "$FI_DATASET"
dataset_subset = "default"
question_key = "translated_problem"
answer_key = "answer"
math_verify_max_workers = 64
math_verify_timeout = 60
ORCHEOF

cat > "$OUTPUT_DIR/train_config.toml" << TRAINEOF
max_steps = $TRAINER_STEPS
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
name = "${RUN_NAME}-trainer"

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
TRAINEOF

cat > "$OUTPUT_DIR/infer_sft.toml" << INFEREOF
[model]
name = "$SFT_MODEL"
max_model_len = $INF_SEQ_LEN

[parallel]
tp = 1

[server]
port = 8000
INFEREOF

ENV_SETUP="$OUTPUT_DIR/env_setup.sh"
cat > $ENV_SETUP << 'ENVEOF'
cd /shared_silo/scratch/dzautner/prime-rl
source /shared_silo/scratch/dzautner/prime-rl/.venv/bin/activate
export HF_CACHE=/shared_silo/scratch/dzautner/lib/prime-rl/hf_cache
export HF_HOME=/shared_silo/scratch/dzautner/lib/prime-rl/hf_home
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_$(hostname)_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_cache_$(hostname)_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export RCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export PYTHONPATH=/shared_silo/scratch/dzautner/prime-rl/src:/shared_silo/scratch/dzautner/tw-dashboard/wandb_shim:${PYTHONPATH:-}
export FASTTRACKML_URL=http://tus1-vm-amd-k8s-001:5000
export WANDB_API_KEY=local
export PYTHONUNBUFFERED=1
ENVEOF

BIND_MOUNTS="-B /shared_silo/scratch/dzautner/:/shared_silo/scratch/dzautner/:rw"
BIND_MOUNTS="$BIND_MOUNTS -B /shared_silo/scratch/megatron_checkpoints/:/shared_silo/scratch/megatron_checkpoints/:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

echo "Starting inference on ${NUM_INF_NODES} node(s)..."

srun --nodes=1 --ntasks=1 --nodelist=$FIRST_INF_NODE --job-name=inference-master --exclusive \
    /usr/bin/singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
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
    srun --nodes=1 --ntasks=1 --nodelist=$INF_NODE --job-name=inference-worker-$i --exclusive \
        /usr/bin/singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
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

echo "Starting orchestrator + trainer..."
srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --job-name=rl-training \
    /usr/bin/singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
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

echo "Done!"
