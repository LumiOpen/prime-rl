#!/bin/bash
#SBATCH --job-name=gradnorm-probe
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=./opd_val/logs/probe_%j.out
#SBATCH --error=./opd_val/logs/probe_%j.err

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
# DO NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. It selects flash-attn's
# Triton AMD backend, whose BACKWARD is broken here: gradients come out ~1.4e7x
# too large (forward is exact), which silently corrupts layer 0 and the
# embeddings and makes the default max_norm=1.0 clip away all learning.
# Verified job 43543. The default composable-kernel backend is correct AND 3x
# faster (17.6k vs 5.9k tok/s). See opd_val/FINDINGS.md 2.1.
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
# Single GPU on purpose: with world_size 1 there is no sharding, so if the
# manual norm and torchtitan's reported norm disagree here, the distributed
# reduction is exonerated and the bug is in the norm computation itself.
# max_norm 1.0 is the DEFAULT that breaks training — we want the probe to see
# exactly the regime that fails.
# Which of these three is responsible for the 1.4e7x gradient inflation?
#
#   TRITON: FLASH_ATTENTION_TRITON_AMD_ENABLE picks flash-attn's Triton AMD
#           backend over composable-kernel. Inherited from
#           rl_train_singlenode_mixed.sbatch:308 and copied into every opd_val
#           script, so EVERY experiment so far used it. Top suspect.
#   AC:     every damaged tensor sat inside a _checkpoint_wrapped_module, and
#           activation checkpointing recomputes the forward during backward.
#   kernel: flash-attn itself, independent of the above.
#
# Each line is "PROBE_SDPA TRITON AC". Reference points: flash+TRUE+on = 3.4e8
# (the regime all prior results ran in), MATH SDPA = 24.4.
for CFG in "0 TRUE on" "0 FALSE on" "0 TRUE off" "1 TRUE on"; do
    set -- $CFG; SDPA=$1; TRITON=$2; AC=$3
    if [ "$SDPA" = "1" ]; then ARM="MATH SDPA        "; else ARM="flash-attn varlen"; fi
    if [ "$AC" = "off" ]; then ACFLAG="--model.ac None"; else ACFLAG=""; fi
    echo ""
    echo "############ $ARM | TRITON_AMD=$TRITON | act.ckpt=$AC ############"
    # The export must come AFTER sourcing env_setup, which sets it to TRUE.
    singularity exec $CF $BM "$IMG" \
        bash -c "source $ENV_SETUP && export FLASH_ATTENTION_TRITON_AMD_ENABLE=$TRITON && PROBE_SDPA=$SDPA \
            uv run torchrun --standalone --nproc-per-node 1 --local-ranks-filter 0 \
                $OUT/gradnorm_probe.py @ examples/basic/reverse-text/sft.toml \
                --no-wandb --model.compile None --optim.max-norm 1e12 $ACFLAG \
                --max-steps 3 --output-dir $RUN_DIR/s${SDPA}_t${TRITON}_ac${AC}" \
        2>&1 | grep -E "\[probe\] step|\[probe\] attention|Step [0-9]+ \||Error|Traceback|RuntimeError" | head -12
done

rm -rf "$SCRATCH"
echo ""
echo "=== done ==="
