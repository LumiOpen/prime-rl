#!/bin/bash
#SBATCH --job-name=math-band-8b
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=./opd_val/logs/mathband_%j.out
#SBATCH --error=./opd_val/logs/mathband_%j.err

# The gate for scaling opd up: Qwen3-8B student <- Qwen3-32B teacher on MATH.
#
# Run this BEFORE committing any training time. It answers the one question
# that decides whether the experiment is well-posed at all:
#
#   Is the TEACHER meaningfully better than the STUDENT on the STUDENT's own
#   competence band?
#
# The student sits at ~0.5 there by construction. If the teacher does too,
# there is no headroom and no algorithm can demonstrate anything. That is
# exactly how reverse-text wasted several days, and it is why GSM8K is now
# exhausted for this pair: Qwen3-8B already scores 0.893 on the band built to
# be hard for Qwen3-1.7B, so as a STUDENT on GSM8K it has nothing left to learn.
#
# Two phases:
#   1. Build the MATH band from the 8B student (20-80% pass rate).
#   2. Score student and teacher on that band -> the gap that gates everything.
#
# The band is student-specific. data/band_math_qwen3_1.7b cannot be reused: it
# was built for a different student and holds only 61 problems.
#
# max_tokens 2048 rather than GSM8K's 1024: MATH solutions are longer, and
# sizing the budget by guesswork rather than measurement wrecked the first
# reverse-text runs. The truncation rate printed below is the check.

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
STUDENT="Qwen/Qwen3-8B"
TEACHER="Qwen/Qwen3-32B"
S_PORT=8051
T_PORT=8052
NUM_PROBLEMS=${NUM_PROBLEMS:-3000}
ROLLOUTS=${ROLLOUTS:-8}
MAX_TOKENS=${MAX_TOKENS:-2048}
BAND="$OUT/data/band_math_qwen3_8b"
cd "$PRIME_RL_DIR"

# Fail fast rather than 20 minutes into a server boot: HF_HUB_OFFLINE=1 below
# means a half-downloaded teacher surfaces as a confusing load error.
SNAP=$(echo /shared_silo/scratch/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/*/)
NSHARD=$(ls -1 "$SNAP"/*.safetensors 2>/dev/null | wc -l)
NPART=$(ls -1 /shared_silo/scratch/hf_cache/hub/models--Qwen--Qwen3-32B/blobs 2>/dev/null | grep -c incomplete)
if [ "$NSHARD" -ne 17 ] || [ "$NPART" -ne 0 ]; then
    echo "ERROR: Qwen3-32B incomplete ($NSHARD/17 shards, $NPART partial blobs). Finish the download first."
    exit 1
fi

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_mathband_${SLURM_JOB_ID}.sh"
cat > "$ENV_SETUP" << ENVEOF
cd $PRIME_RL_DIR
export PATH="/home/rahul.aralikatte@amd.com/.local/bin:\$PATH"
export TRITON_CACHE_DIR=$SCRATCH/triton
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor
export XDG_CACHE_HOME=$SCRATCH/xdg
export UV_CACHE_DIR=$SCRATCH/uv
export TMPDIR=$SCRATCH/tmp
export HOME=$SCRATCH/home
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$XDG_CACHE_HOME \$UV_CACHE_DIR \$TMPDIR \$HOME
export UV_PYTHON=$PRIME_RL_DIR/.venv/bin/python3
export UV_NO_SYNC=1
export HF_HOME=/shared_silo/scratch/hf_cache
export HF_HUB_OFFLINE=1
# DO NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE -- broken backward on ROCm.
# See opd_val/FINDINGS.md 2.1. Inference-only here, but keep the habit.
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
ENVEOF
mkdir -p "$SCRATCH" "$SCRATCH/vf-scripts" "$OUT/data"

CF="--rocm --cleanenv --containall"
BM="-B /shared_silo/scratch:/shared_silo/scratch:rw -B $HOME:$HOME:ro"
BM="$BM -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BM="$BM -B /usr/share/libdrm:/usr/share/libdrm:ro -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BM="$BM -B /tmp:/tmp:rw -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"

boot () {  # model gpus port tp log
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=$2 && \
            uv run inference --model.name $1 --server.port $3 --parallel.tp $4 \
            --model.max-model-len 4096 \
            --gpu-memory-utilization 0.85 --model.max-logprobs 128" > "$5" 2>&1 &
    echo $!
}
wait_health () { for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$1/health" &>/dev/null && return 0; sleep 2; done; return 1; }

run_band () {  # port model label extra...
    local port=$1 model=$2 label=$3; shift 3
    singularity exec $CF $BM "$IMG" bash -c "source $ENV_SETUP && \
        uv run python $OUT/gsm8k_band.py \
            --base-url http://127.0.0.1:${port}/v1 --model $model \
            --no-thinking --concurrency 128 --rollouts $ROLLOUTS \
            --max-tokens $MAX_TOKENS --label $label $*" 2>&1
}

echo "=== MATH band + teacher gate | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "student $STUDENT (GPUs 0,1 tp=2) | teacher $TEACHER (GPUs 2,3 tp=2)"
echo "problems=$NUM_PROBLEMS rollouts=$ROLLOUTS max_tokens=$MAX_TOKENS"

SPID=$(boot "$STUDENT" 0,1 $S_PORT 2 "$OUT/logs/mathband_${SLURM_JOB_ID}_student.log")
TPID=$(boot "$TEACHER" 2,3 $T_PORT 2 "$OUT/logs/mathband_${SLURM_JOB_ID}_teacher.log")
trap 'kill $SPID $TPID 2>/dev/null; rm -rf "$SCRATCH"; rm -f "$ENV_SETUP"' EXIT

wait_health $S_PORT || { echo "student server failed"; tail -30 "$OUT/logs/mathband_${SLURM_JOB_ID}_student.log"; exit 1; }
echo "student ready"
wait_health $T_PORT || { echo "teacher server failed"; tail -30 "$OUT/logs/mathband_${SLURM_JOB_ID}_teacher.log"; exit 1; }
echo "teacher ready"

echo ""
echo "########## 1. build the MATH band from $STUDENT ##########"
run_band $S_PORT "$STUDENT" student-scan --dataset math --num-problems $NUM_PROBLEMS --out "$BAND"

if [ ! -f "$BAND/train.parquet" ]; then
    echo "ERROR: no band written -- nothing landed in 20-80%. Widen --lo/--hi or scan more problems."
    exit 1
fi

echo ""
echo "########## 2. the gate: both models on the student's band ##########"
run_band $S_PORT "$STUDENT" student-on-band --dataset local --local-path "$BAND" --num-problems 400
run_band $T_PORT "$TEACHER" teacher-on-band --dataset local --local-path "$BAND" --num-problems 400

echo ""
echo "=== done. GATE: teacher-on-band must clearly exceed student-on-band. ==="
