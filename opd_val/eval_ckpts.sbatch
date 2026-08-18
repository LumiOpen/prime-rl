#!/bin/bash
#SBATCH --job-name=eval-ckpts
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=./opd_val/logs/evalckpt_%j.out
#SBATCH --error=./opd_val/logs/evalckpt_%j.err

# Measure the RL arms properly, offline.
#
# Two reasons not to trust the in-run numbers:
#   - in-run EVAL is broken: the renderer (and so enable_thinking=false) applies
#     only to train rollouts, because the orchestrator builds train clients as
#     "renderer" and eval clients as "openai_chat_completions". Eval therefore
#     runs with Qwen3 thinking ON, blows the token budget and reports ~96%
#     truncation / ~0.03 reward. Confirmed at trace level: 128/128 rollouts open
#     <think>, 19 close it.
#   - in-run TRAIN reward is honest but very noisy: batch_size 128 / group_size
#     16 is only 8 unique problems per step, so it swings 0.48-0.75 run to run.
#
# So: serve each saved checkpoint and score it over the WHOLE 327-problem band
# with gsm8k_band.py — the same harness that produced the validated baselines
# (student 0.588, teacher 0.893).

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
BAND="$OUT/data/gsm8k_boxed_band"
PORT=8051
cd "$PRIME_RL_DIR"

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_evalckpt.sh"
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

echo "=== checkpoint eval on the 327-problem band | job $SLURM_JOB_ID ==="
echo "reference: student 0.588 (step 0) -> teacher 0.893"

for RUN in ${RUNS:-opd-lr1e-6_42563 grpo-lr1e-6_42564 opd-lr3e-6_42565 grpo-lr3e-6_42566}; do
  # STEPS filters which checkpoints to score (default: all). Scoring every
  # checkpoint costs a vLLM spin-up each, so restrict it when only the
  # trajectory endpoints matter.
  for CKPT in $(ls -d "$OUT/runs/$RUN/run/weights/step_"* 2>/dev/null | sort -t_ -k2 -n); do
    if [ -n "${STEPS:-}" ]; then
      case " $STEPS " in *" $(basename "$CKPT" | sed 's/step_//') "*) ;; *) continue ;; esac
    fi
    STEP=$(basename "$CKPT")
    echo ""
    echo "########## $RUN / $STEP ##########"
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=0,1 && \
            uv run inference --model.name $CKPT --server.port $PORT \
            --gpu-memory-utilization 0.85 --parallel.dp 2" \
        > "$OUT/logs/evalckpt_${SLURM_JOB_ID}_${RUN}_${STEP}.log" 2>&1 &
    PID=$!
    ok=0
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null && { ok=1; break; }; sleep 2; done
    if [ $ok -ne 1 ]; then
      echo "  server failed"; tail -15 "$OUT/logs/evalckpt_${SLURM_JOB_ID}_${RUN}_${STEP}.log"
      kill $PID 2>/dev/null; wait $PID 2>/dev/null; continue
    fi
    singularity exec $CF $BM "$IMG" bash -c "source $ENV_SETUP && \
        uv run python $OUT/gsm8k_band.py \
            --base-url http://127.0.0.1:${PORT}/v1 --model $CKPT \
            --dataset local --local-path $BAND --no-thinking --concurrency 128 \
            --num-problems 327 --rollouts 4 --max-tokens 1024 \
            --label ${RUN}/${STEP}" 2>&1 | grep -E "^===|^overall|^truncated|^completion"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 5
  done
done

rm -rf "$SCRATCH"
echo ""
echo "=== done ==="
