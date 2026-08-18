#!/bin/bash
#SBATCH --job-name=opd-reverse-text
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=./opd_val/logs/opd_%j.out
#SBATCH --error=./opd_val/logs/opd_%j.err

# =============================================================================
#  On-policy distillation (OPD) validation on reverse-text.
#
#    student/policy : PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT
#    teacher        : PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL   (frozen)
#
#  GPU layout:
#    GPU 0  policy inference server  (:8000)  ← launched by the rl entrypoint
#    GPU 1  trainer                           ← launched by the rl entrypoint
#    GPU 2  frozen teacher server    (:8001)  ← launched here
#
#  prime-rl never launches frozen models, so the teacher is our job. The rl
#  entrypoint detects the frozen reference in the config and logs a reminder
#  that the endpoint must already be serving before the orchestrator starts.
#
#  Usage:  sbatch opd_val/rl_opd.sbatch [run_name]
# =============================================================================

set -uo pipefail

RUN_NAME=${1:-opd-reverse-text}
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
# CONFIG: which run config to use. NEED_TEACHER=0 skips the frozen teacher
# server (for the GRPO control, which has no teacher).
CONFIG="${CONFIG:-$OUT/opd_reverse_text.toml}"
NEED_TEACHER="${NEED_TEACHER:-1}"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/primus_v26.2_moefix.sif}"
# Teacher model + GPUs are overridable so the same launcher serves the
# reverse-text (0.6B) and GSM8K (8B) experiments.
TEACHER="${TEACHER_MODEL:-PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL}"
TEACHER_PORT=8001
TEACHER_GPU="${TEACHER_GPU:-2}"

cd "$PRIME_RL_DIR"
RUN_DIR="$OUT/runs/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# --cleanenv --containall strips SLURM_* inside the container, so bake the job
# id in here on the host — otherwise every job shares one /tmp cache path and
# collides with whichever user created it first (EACCES).
SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$RUN_DIR/env_setup.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR
export PATH="/home/rahul.aralikatte@amd.com/.local/bin:\$PATH"
export TRITON_CACHE_DIR=$SCRATCH/triton
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor
export XDG_CACHE_HOME=$SCRATCH/xdg
export UV_CACHE_DIR=$SCRATCH/uv
export HOME=$SCRATCH/home
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$XDG_CACHE_HOME \$UV_CACHE_DIR \$HOME
export UV_PYTHON=$PRIME_RL_DIR/.venv/bin/python3
export UV_NO_SYNC=1
export HF_HOME=/shared_silo/scratch/hf_cache
export HF_HUB_OFFLINE=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
ENVEOF
mkdir -p "$SCRATCH"

CONTAINER_FLAGS="--rocm --cleanenv --containall"
BIND_MOUNTS="-B /shared_silo/scratch:/shared_silo/scratch:rw"
BIND_MOUNTS="$BIND_MOUNTS -B $HOME:$HOME:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"
# verifiers hardcodes its uv-script cache at /tmp/vf-scripts
# (deps/verifiers/verifiers/v1/runtimes/base.py) with no TMPDIR support. Since
# host /tmp is bind-mounted, that directory is shared by every user on the node
# and whoever creates it first owns it — every rollout then dies with
# "HarnessError: harness setup: PermissionError ... /tmp/vf-scripts/...".
# Bind a job-private dir over just that path.
mkdir -p "$SCRATCH/vf-scripts"
BIND_MOUNTS="$BIND_MOUNTS -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"

echo "============================================"
echo "  OPD validation: $RUN_NAME"
echo "============================================"
echo "Job:       $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Container: $IMG"
echo "Config:    $CONFIG"
echo "Teacher:   $TEACHER  (GPU $TEACHER_GPU, port $TEACHER_PORT)"
echo "Run dir:   $RUN_DIR"
echo "============================================"

# ── Frozen teacher on its own GPU ─────────────────────────────────────────────
# Isolated process: HIP_VISIBLE_DEVICES is unambiguous here because nothing
# else sets CUDA_VISIBLE_DEVICES for it.
TEACHER_PID=
if [ "$NEED_TEACHER" = "1" ]; then
    echo "Starting teacher on GPU $TEACHER_GPU ..."
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=$TEACHER_GPU && \
            uv run inference --model.name $TEACHER --server.port $TEACHER_PORT \
            --model.max-logprobs ${TEACHER_MAX_LOGPROBS:-128} \
            ${TEACHER_ARGS:---gpu-memory-utilization 0.5 --model.enforce-eager}" \
        > "$RUN_DIR/teacher.log" 2>&1 &
    TEACHER_PID=$!

    echo "Waiting for teacher health on :$TEACHER_PORT ..."
    for i in $(seq 1 450); do
        curl -sf "http://127.0.0.1:${TEACHER_PORT}/health" &>/dev/null && break
        kill -0 $TEACHER_PID 2>/dev/null || { echo "ERROR: teacher process died."; tail -50 "$RUN_DIR/teacher.log"; exit 1; }
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:${TEACHER_PORT}/health" &>/dev/null; then
        echo "ERROR: teacher not healthy within 900s. Log tail:"; tail -50 "$RUN_DIR/teacher.log"
        kill $TEACHER_PID 2>/dev/null; exit 1
    fi
    echo "Teacher ready."
else
    echo "NEED_TEACHER=0 — no frozen teacher for this run (control)."
fi

cleanup() {
    [ -n "$TEACHER_PID" ] && { echo "Cleaning up teacher (pid $TEACHER_PID) ..."; kill $TEACHER_PID 2>/dev/null || true; wait $TEACHER_PID 2>/dev/null || true; }
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

# ── Policy server + orchestrator + trainer via the rl entrypoint ─────────────
# HIP_VISIBLE_DEVICES is intentionally left unset: on ROCm it overrides
# CUDA_VISIBLE_DEVICES, which is what the rl entrypoint uses to put inference on
# GPU 0 and the trainer on GPU 1. Setting it would put both on the same device.
echo "Starting rl entrypoint (policy server GPU 0, trainer GPU 1) ..."
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && \
        uv run rl @ $CONFIG \
            --no-wandb \
            --output-dir $RUN_DIR/run"
RC=$?

echo ""
echo "============================================"
echo "rl entrypoint exited rc=$RC"
echo "Logs: $RUN_DIR/run/logs/{orchestrator,trainer,inference}.log"
echo "Teacher log: $RUN_DIR/teacher.log"
echo "============================================"
exit $RC
