#!/bin/bash
#SBATCH --job-name=gsm8k-band
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=./opd_val/logs/band_%j.out
#SBATCH --error=./opd_val/logs/band_%j.err

# Characterise the GSM8K student/teacher pair before committing to any training.
#
# Answers three things that reverse-text taught us to check FIRST:
#   1. Is there a real capability gap?  (student vs teacher pass rate)
#   2. Is there a usable operating point? (how many problems land in the
#      20-80% band for the student — the hendrycks-sanity construction)
#   3. What token budget does the task actually need? (completion-length
#      distribution + truncation rate; guessing this wrecked the first runs)
#
# Student Qwen3-1.7B on GPU 0, teacher Qwen3-8B on GPUs 1-3 (tp=2 would need a
# power of two, so dp over the remaining cards is simpler: tp=1, one replica).
# Both share vocab 151669 with identical token ids, which is what makes them a
# legal opd pair at all.

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
STUDENT="Qwen/Qwen3-1.7B"
TEACHER="Qwen/Qwen3-8B"
S_PORT=8041
T_PORT=8042
NUM_PROBLEMS=${NUM_PROBLEMS:-512}
ROLLOUTS=${ROLLOUTS:-8}
MAX_TOKENS=${MAX_TOKENS:-1024}
cd "$PRIME_RL_DIR"

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_band.sh"
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
# DO NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. It selects flash-attn's
# Triton AMD backend, whose BACKWARD is broken here: gradients come out ~1.4e7x
# too large (forward is exact), which silently corrupts layer 0 and the
# embeddings and makes the default max_norm=1.0 clip away all learning.
# Verified job 43543. The default composable-kernel backend is correct AND 3x
# faster (17.6k vs 5.9k tok/s). See opd_val/FINDINGS.md 2.1.
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
ENVEOF
mkdir -p "$SCRATCH" "$SCRATCH/vf-scripts"

CF="--rocm --cleanenv --containall"
BM="-B /shared_silo/scratch:/shared_silo/scratch:rw -B $HOME:$HOME:ro"
BM="$BM -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BM="$BM -B /usr/share/libdrm:/usr/share/libdrm:ro -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BM="$BM -B /tmp:/tmp:rw -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"

boot () {  # model gpus port log
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=$2 && \
            uv run inference --model.name $1 --server.port $3 \
            --gpu-memory-utilization 0.85 --model.max-logprobs 128" > "$4" 2>&1 &
    echo $!
}
wait_health () { for i in $(seq 1 450); do curl -sf "http://127.0.0.1:$1/health" &>/dev/null && return 0; sleep 2; done; return 1; }

echo "=== GSM8K band | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "student $STUDENT (GPU 0) | teacher $TEACHER (GPUs 1,2,3)"
echo "problems=$NUM_PROBLEMS rollouts=$ROLLOUTS max_tokens=$MAX_TOKENS"

SPID=$(boot "$STUDENT" 0 $S_PORT "$OUT/logs/band_${SLURM_JOB_ID}_student.log")
TPID=$(boot "$TEACHER" 1,2,3 $T_PORT "$OUT/logs/band_${SLURM_JOB_ID}_teacher.log")
trap 'kill $SPID $TPID 2>/dev/null; rm -rf "$SCRATCH"' EXIT

wait_health $S_PORT || { echo "student server failed"; tail -25 "$OUT/logs/band_${SLURM_JOB_ID}_student.log"; exit 1; }
echo "student ready"
wait_health $T_PORT || { echo "teacher server failed"; tail -25 "$OUT/logs/band_${SLURM_JOB_ID}_teacher.log"; exit 1; }
echo "teacher ready"

# Thinking mode pinned the length distribution against the 1024 cap (job 42470:
# p50 1009, p90/p99 = cap, ~50% truncated for both models), so the first pass
# measured truncation rather than capability. Re-characterise with thinking off
# across both candidate tasks, and pick whichever gives a real teacher gap and a
# populated band.
for DS in gsm8k math; do
  echo ""
  echo "################ dataset=$DS (thinking off) ################"
  singularity exec $CF $BM "$IMG" bash -c "source $ENV_SETUP && \
      uv run python $OUT/gsm8k_band.py \
          --base-url http://127.0.0.1:${S_PORT}/v1 --model $STUDENT \
          --dataset $DS --no-thinking \
          --num-problems $NUM_PROBLEMS --rollouts $ROLLOUTS --max-tokens $MAX_TOKENS \
          --label student --out $OUT/data/band_${DS}_qwen3_1.7b" 2>&1
  singularity exec $CF $BM "$IMG" bash -c "source $ENV_SETUP && \
      uv run python $OUT/gsm8k_band.py \
          --base-url http://127.0.0.1:${T_PORT}/v1 --model $TEACHER \
          --dataset $DS --no-thinking \
          --num-problems $NUM_PROBLEMS --rollouts 2 --max-tokens $MAX_TOKENS \
          --label teacher" 2>&1
done

echo "=== done ==="
