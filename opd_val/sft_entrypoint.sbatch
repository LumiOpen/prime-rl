#!/bin/bash
#SBATCH --job-name=sft-isolation
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=./opd_val/logs/sftiso_%j.out
#SBATCH --error=./opd_val/logs/sftiso_%j.err

# TRAINER ISOLATION TEST.
#
# Runs the standalone `sft` entrypoint on the repo's own example config
# (examples/basic/reverse-text/sft.toml): PrimeIntellect/Qwen3-0.6B fine-tuned on
# willcb/R1-reverse-wikipedia-paragraphs-v1-1000, lr 2e-5, 100 steps, 1 GPU.
#
# This exercises the same trainer/model/loss code as the RL path but bypasses
# the orchestrator, vLLM sampling, rollout packing and weight broadcast
# entirely. Upstream has a published W&B run for this exact config, so the loss
# is known to fall.
#
#   loss falls  -> trainer core is fine; the bug is in the RL data path
#   loss flat   -> trainer/model core is broken on ROCm, and the 1e8-1e10
#                  gradient norms are the thing to chase
#
# Usage: sbatch opd_val/sft_entrypoint.sbatch [run_name]

set -uo pipefail
RUN_NAME=${1:-sft-isolation}
# EXTRA_ARGS: appended to the `uv run sft` command, for ablations
# (e.g. EXTRA_ARGS="--model.no-optim-cpu-offload").
EXTRA_ARGS="${EXTRA_ARGS:-}"
LRS="${LRS:-2e-5 1e-4}"
PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="${CONTAINER_IMG:-/shared_silo/scratch/containers/primus_v26.2_moefix.sif}"
cd "$PRIME_RL_DIR"

RUN_DIR="$OUT/runs/${RUN_NAME}_${SLURM_JOB_ID}"
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
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
ENVEOF
mkdir -p "$SCRATCH" "$SCRATCH/vf-scripts"

CF="--rocm --cleanenv --containall"
BM="-B /shared_silo/scratch:/shared_silo/scratch:rw -B $HOME:$HOME:ro"
BM="$BM -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BM="$BM -B /usr/share/libdrm:/usr/share/libdrm:ro -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BM="$BM -B /tmp:/tmp:rw -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"

echo "=== SFT trainer isolation | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "Config: examples/basic/reverse-text/sft.toml (100 steps, 1 GPU)"
echo "LRs: $LRS   EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
echo "Run dir: $RUN_DIR"

# Two arms so a flat curve can't be blamed on the learning rate:
#   default lr 2e-5 (the repo's own vetted value for this config)
#   plus torch.compile disabled, as required on ROCm
for LR in $LRS; do
    echo ""
    echo "############ lr=$LR ############"
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && \
            uv run sft @ examples/basic/reverse-text/sft.toml \
                --no-wandb \
                --optim.lr $LR \
                --model.compile None \
                $EXTRA_ARGS \
                --output-dir $RUN_DIR/lr_$LR" \
        2>&1 | tee "$RUN_DIR/sft_lr_$LR.log" | grep -E "Step |Loss|Error|Traceback|SUCCESS" | tail -40
    echo "(full log: $RUN_DIR/sft_lr_$LR.log)"
done

rm -rf "$SCRATCH"
echo ""
echo "=== done ==="
