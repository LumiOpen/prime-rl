#!/bin/bash
#SBATCH --job-name=rl-reverse-text
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 2 GPUs: one inference, one trainer. See --hint=multithread in the wheel builders for
# why the CPU count is in hardware threads rather than whole cores.
#SBATCH --cpus-per-task=28
#SBATCH --hint=multithread
#SBATCH --gpus-per-node=2
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/reverse_text_%j.out
#SBATCH --error=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main/logs/reverse_text_%j.err

# =============================================================================
#   Smoke test for the LUMI/ROCm port of upstream main: the reverse-text RL
#   example, end to end, on one dev-g node with 2 GPUs.
#
#   This is the first thing to run after any dependency or port change. It
#   exercises the whole stack — vLLM inference, the env server, the
#   orchestrator, FSDP training and the NCCL weight transport — on a 0.6B model
#   that fits anywhere, so a failure is a port bug and not a capacity problem.
#
#   Usage:
#     sbatch run_reverse_text_test.sbatch
#     MAX_STEPS=3 sbatch run_reverse_text_test.sbatch
# =============================================================================

set -euo pipefail

BASE="/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main"
IMG="${CONTAINER_IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
CONFIG="${CONFIG:-examples/basic/reverse-text/rl.toml}"
# 5, not fewer: the example evaluates at interval 5, so this is the cheapest value that
# still exercises the eval path.
MAX_STEPS="${MAX_STEPS:-5}"

[ -f "$IMG" ] || { echo "ERROR: container not found: $IMG"; exit 1; }
# Both prime-rl and verifiers derive cache roots from Path.home() with no env override, so
# $HOME/.cache has to be writable. $HOME itself stays read-only: it is at 92% of its 100K
# inode quota, and nothing else in the run belongs there.
mkdir -p "$BASE/logs" "$BASE/cache/home_cache"

echo "============================================"
echo "  reverse-text smoke test"
echo "============================================"
echo "Job:      $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Config:   $CONFIG  (max_steps=$MAX_STEPS)"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "============================================"

# Values are baked in rather than passed with `singularity --env`: that flag splits its
# argument on commas, so any comma-valued variable (ROCR_VISIBLE_DEVICES=0,1) is parsed
# as a second malformed pair and the exec is rejected outright.
cat > "/tmp/rl_test_${SLURM_JOB_ID}.sh" << RUNEOF
BASE="$BASE"
CONFIG="$CONFIG"
MAX_STEPS="$MAX_STEPS"
# --cleanenv strips SLURM_*, and the node-local cache paths below need the job id.
SLURM_JOB_ID="$SLURM_JOB_ID"
ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
export ROCR_VISIBLE_DEVICES
RUNEOF

cat >> "/tmp/rl_test_${SLURM_JOB_ID}.sh" << 'RUNEOF'
set -euo pipefail
cd "$BASE"
source env.sh

# SLURM hands out the allocation through ROCR_VISIBLE_DEVICES, which the ROCr runtime
# applies *before* HIP/CUDA_VISIBLE_DEVICES. The rl entrypoint partitions GPUs in-process
# and gives each child its own CUDA_VISIBLE_DEVICES, which then indexes into the ROCr
# set — so the ids it works with must be logical (0..N-1), never node-absolute.
NUM_GPUS=$(awk -F, '{print NF}' <<< "${ROCR_VISIBLE_DEVICES:-0,1}")
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
echo "Visible GPUs: ROCR=${ROCR_VISIBLE_DEVICES:-<unset>} -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# MIOpen/comgr caches must sit on node-local /tmp. Their SQLite locking does not work on
# Lustre, and the failure mode is an unbounded startup hang rather than an error — so this
# overrides env.sh, which points MIOPEN_USER_DB_PATH at $BASE/cache on scratch.
export MIOPEN_USER_DB_PATH=/tmp/miopen_$(hostname)_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MIOPEN_FIND_MODE=FAST
export AMD_COMGR_CACHE_DIR=/tmp/comgr_$(hostname)_${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH" "$AMD_COMGR_CACHE_DIR"

# LUMI-G high-speed network. Redundant for 2 GCDs on one node, required multi-node.
# NCCL_NET_GDR_LEVEL is deliberately unset: it hangs the ROCm 7 lumi-multitorch containers.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0

# Fail here rather than several minutes deep inside vLLM if the container cannot
# actually reach /dev/kfd.
uv run python -c "import torch; n = torch.cuda.device_count(); print(f'torch {torch.__version__} sees {n} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n)]}'); assert n == $NUM_GPUS, 'GPU count mismatch'"

# --no-monitors.prime: the Prime Intellect monitor is on by default and hard-fails
# without an API key rather than running untracked. --no-monitors.wandb keeps the run
# off the network entirely; the file monitor still writes everything to the run dir.
# --no-dashboard: the local web UI registers itself under $HOME/.cache/prime-rl, which is
# bind-mounted read-only here (and is on a quota'd filesystem anyway). Nothing would be
# watching a localhost port on a compute node regardless.
uv run rl @ "$CONFIG" \
    --max-steps "$MAX_STEPS" \
    --no-dashboard \
    --no-monitors.prime \
    --no-monitors.wandb \
    --deployment.num-infer-gpus 1 \
    --deployment.num-train-gpus 1
RUNEOF

singularity exec --cleanenv \
    -B /pfs,/scratch,/projappl,/project,/flash,/appl \
    -B "$HOME:$HOME:ro" -B "$BASE/cache/home_cache:$HOME/.cache:rw" -B /tmp:/tmp:rw \
    "$IMG" bash "/tmp/rl_test_${SLURM_JOB_ID}.sh"

rm -f "/tmp/rl_test_${SLURM_JOB_ID}.sh"
echo "Done!"
