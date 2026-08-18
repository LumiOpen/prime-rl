#!/bin/bash
#SBATCH --job-name=opd-cmp
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=./opd_val/logs/cmp_%j.out
#SBATCH --error=./opd_val/logs/cmp_%j.err

# Is there anything for OPD to distill?
#
# Samples the same reverse-text prompts from student and teacher and compares
# behaviour (thinking rate, completion length, LCS reward). Then measures the
# per-token reverse KL the OPD loss actually sees: teacher logprobs over the
# STUDENT's own sampled tokens. If that KL is ~0, the two models agree on the
# student's trajectories and OPD has no gradient to give, regardless of how
# differently they score on the task.

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
STUDENT="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
TEACHER="PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
S_PORT=8021
T_PORT=8022
cd "$PRIME_RL_DIR"

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_cmp.sh"
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
export PYTHONUNBUFFERED=1
ENVEOF
mkdir -p "$SCRATCH"

CF="--rocm --cleanenv --containall"
BM="-B /shared_silo/scratch:/shared_silo/scratch:rw -B $HOME:$HOME:ro"
BM="$BM -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BM="$BM -B /usr/share/libdrm:/usr/share/libdrm:ro -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro -B /tmp:/tmp:rw"

boot () {  # name gpu port logfile
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=$2 && \
            uv run inference --model.name $1 --server.port $3 \
            --gpu-memory-utilization 0.5 --model.enforce-eager" > "$4" 2>&1 &
    echo $!
}
wait_health () {
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$1/health" &>/dev/null && return 0; sleep 2; done
    return 1
}

SPID=$(boot "$STUDENT" 0 $S_PORT "$OUT/logs/cmp_student_${SLURM_JOB_ID}.log")
TPID=$(boot "$TEACHER" 1 $T_PORT "$OUT/logs/cmp_teacher_${SLURM_JOB_ID}.log")
trap 'kill $SPID $TPID 2>/dev/null; rm -rf "$SCRATCH"' EXIT

wait_health $S_PORT || { echo "student server failed"; tail -30 "$OUT/logs/cmp_student_${SLURM_JOB_ID}.log"; exit 1; }
wait_health $T_PORT || { echo "teacher server failed"; tail -30 "$OUT/logs/cmp_teacher_${SLURM_JOB_ID}.log"; exit 1; }
echo "Both servers healthy."

singularity exec $CF $BM "$IMG" bash -c "source $ENV_SETUP && uv run python $OUT/compare.py $S_PORT $T_PORT" 2>&1
