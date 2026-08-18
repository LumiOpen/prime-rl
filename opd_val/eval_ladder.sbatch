#!/bin/bash
#SBATCH --job-name=eval-ladder
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=./opd_val/logs/evalladder_%j.out
#SBATCH --error=./opd_val/logs/evalladder_%j.err

# Score each rung of the SFT ladder (job 42459) on reverse-text so we can pick
# the opd student deterministically instead of guessing from training loss.
#
# Target band is 0.3-0.6: competent at the task but clearly below the teacher's
# 0.799. Everything tried so far sat outside it — base model at 0.04 (opd
# destroys it) and the released SFT model at 0.72 (no headroom).
#
# Serves each checkpoint in turn on one GPU and evaluates with the same budget
# the opd runs use (1024 completion tokens).

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
LADDER="$OUT/runs/sft-ladder_42459/sft/weights"
PORT=8031
cd "$PRIME_RL_DIR"

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_evalladder.sh"
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

echo "=== SFT ladder eval | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "teacher reference = 0.799 | base = 0.04 | released SFT = 0.72 | target band 0.3-0.6"

for CKPT in "$LADDER"/step_*; do
    STEP=$(basename "$CKPT")
    echo ""
    echo "########## $STEP ##########"
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=0 && \
            uv run inference --model.name $CKPT --server.port $PORT \
            --gpu-memory-utilization 0.6 --model.enforce-eager" \
        > "$OUT/logs/evalladder_${SLURM_JOB_ID}_${STEP}.log" 2>&1 &
    PID=$!
    ok=0
    for i in $(seq 1 300); do curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null && { ok=1; break; }; sleep 2; done
    if [ $ok -ne 1 ]; then
        echo "  server failed for $STEP"; tail -20 "$OUT/logs/evalladder_${SLURM_JOB_ID}_${STEP}.log"
        kill $PID 2>/dev/null; wait $PID 2>/dev/null; continue
    fi
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && \
            uv run eval reverse-text-v1 --harness.id null \
            -m $CKPT --client.base-url http://127.0.0.1:${PORT}/v1 \
            -n 64 -r 1 --sampling.max-tokens 1024 --no-push" 2>&1 \
        | sed -e 's/\x1b\[[0-9;]*m//g' | grep -iE "reward|error" | tail -6 | sed 's/^/  /'
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    sleep 5
done

rm -rf "$SCRATCH"
echo ""
echo "=== done ==="
