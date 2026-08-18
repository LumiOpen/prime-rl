#!/bin/bash
#SBATCH --job-name=sft-ladder
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=./opd_val/logs/ladder_%j.out
#SBATCH --error=./opd_val/logs/ladder_%j.err

# Build a ladder of partially-SFT'd students for the opd intermediate-checkpoint
# experiment.
#
# Every opd arm so far started from either the base model (far too weak — opd
# destroys it at every k and lr) or the released Reverse-Text-SFT model (already
# at the teacher's level, 0.72 vs 0.799 — no headroom). The regime on-policy
# distillation is actually designed for is in between: a student that is
# competent at the task but meaningfully worse than the teacher.
#
# One SFT run with periodic checkpoints gives several such students at once;
# pick whichever lands in the 0.3-0.6 reward band as the opd starting point.
# HF-format weights land in <out>/weights/step_<n>/, directly servable by vLLM.

set -uo pipefail
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
cd "$PRIME_RL_DIR"

RUN_DIR="$OUT/runs/sft-ladder_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
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
# DO NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. It selects flash-attn's
# Triton AMD backend, whose BACKWARD is broken here: gradients come out ~1.4e7x
# too large (forward is exact), which silently corrupts layer 0 and the
# embeddings and makes the default max_norm=1.0 clip away all learning.
# Verified job 43543. The default composable-kernel backend is correct AND 3x
# faster (17.6k vs 5.9k tok/s). See opd_val/FINDINGS.md 2.1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
ENVEOF
mkdir -p "$SCRATCH" "$SCRATCH/vf-scripts"

CF="--rocm --cleanenv --containall"
BM="-B /shared_silo/scratch:/shared_silo/scratch:rw -B $HOME:$HOME:ro"
BM="$BM -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BM="$BM -B /usr/share/libdrm:/usr/share/libdrm:ro -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BM="$BM -B /tmp:/tmp:rw -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"

echo "=== SFT ladder | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "60 steps, checkpoint every 15 -> students at steps 15/30/45/60"
echo "Run dir: $RUN_DIR"

singularity exec $CF $BM "$IMG" \
    bash -c "source $ENV_SETUP && \
        uv run sft @ examples/basic/reverse-text/sft.toml \
            --no-wandb \
            --model.compile None \
            --optim.max-norm 1e12 \
            --max-steps 60 \
            --ckpt.interval 15 \
            --output-dir $RUN_DIR/sft" \
    2>&1 | tee "$RUN_DIR/sft.log" | grep -E "Step |Error|Traceback" | tail -30

echo ""
echo "=== checkpoints produced ==="
ls -d "$RUN_DIR"/sft/weights/step_* 2>/dev/null
rm -rf "$SCRATCH"
