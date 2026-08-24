#!/bin/bash
#SBATCH --job-name=loss-checks
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=./opd_val/logs/losschecks_%j.out
#SBATCH --error=./opd_val/logs/losschecks_%j.err

# Correctness checks for the ref_kl loss, on a GPU node (the unit tests need one).
#   1. teacher top-k support alignment against the policy's logits
#   2. top-k gradient direction across all 9 normalization x kl_type combinations
#   3. the existing loss unit tests
#
# Everything that caches must be pointed at scratch, because the compute nodes'
# /tmp is small and shared:
#   UV_NO_SYNC=1     - without it uv re-syncs and extracts the 39MB ROCm vLLM
#                      wheel into /tmp, filling it ("No space left on device")
#   UV_CACHE_DIR     - same reason
#   TMPDIR + TORCHINDUCTOR_CACHE_DIR - torch.compile writes a precompiled header
#                      per kernel; it fills /tmp too, and the failure is silent
#                      for the affected script (it prints nothing at all, with
#                      "cannot write PCH file" only on stderr).
set -uo pipefail
PRIME_RL_DIR=/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl
cd "$PRIME_RL_DIR"
IMG=/shared_silo/scratch/containers/primus_v26.2_moefix.sif
S=$PRIME_RL_DIR/opd_val/.scratch_${SLURM_JOB_ID}
mkdir -p "$S/home" "$S/cache" "$S/uv" "$S/tmp" "$S/inductor"

singularity exec --rocm --cleanenv --containall \
    -B /shared_silo/scratch:/shared_silo/scratch:rw -B "$HOME":"$HOME":ro "$IMG" \
    bash -lc "cd $PRIME_RL_DIR && \
        export HOME=$S/home XDG_CACHE_HOME=$S/cache UV_CACHE_DIR=$S/uv \
               TMPDIR=$S/tmp TORCHINDUCTOR_CACHE_DIR=$S/inductor \
               UV_NO_SYNC=1 UV_PYTHON=$PRIME_RL_DIR/.venv/bin/python3 && \
        echo '########## 1. top-k support alignment ##########' && \
        uv run python opd_val/topk_alignment_test.py; \
        echo '' && echo '########## 2. top-k gradient direction ##########' && \
        uv run python opd_val/topk_direction_test.py; \
        echo '' && echo '########## 3. loss unit tests ##########' && \
        uv run pytest tests/unit/train/rl/test_loss.py -q 2>&1 | tail -8"

rm -rf "$S"
