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
#   1. top-k gradient direction across all 9 normalization x kl_type combinations
#   2. the existing loss unit tests
#
# UV_NO_SYNC=1 is REQUIRED: without it uv re-syncs the project and extracts the
# 39MB ROCm vLLM wheel into the node's /tmp, which fills it ("No space left on
# device"). UV_CACHE_DIR also goes to scratch rather than /tmp for the same reason.
set -uo pipefail
PRIME_RL_DIR=/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl
cd "$PRIME_RL_DIR"
IMG=/shared_silo/scratch/containers/primus_v26.2_moefix.sif
S=$PRIME_RL_DIR/opd_val/.scratch_${SLURM_JOB_ID}
mkdir -p "$S/home" "$S/cache" "$S/uv"

singularity exec --rocm --cleanenv --containall \
    -B /shared_silo/scratch:/shared_silo/scratch:rw -B "$HOME":"$HOME":ro "$IMG" \
    bash -lc "cd $PRIME_RL_DIR && \
        export HOME=$S/home XDG_CACHE_HOME=$S/cache UV_CACHE_DIR=$S/uv \
               UV_NO_SYNC=1 UV_PYTHON=$PRIME_RL_DIR/.venv/bin/python3 && \
        echo '########## 1. top-k gradient direction ##########' && \
        uv run python opd_val/topk_direction_test.py; \
        echo '' && echo '########## 2. loss unit tests ##########' && \
        uv run pytest tests/unit/train/rl/test_loss.py -q 2>&1 | tail -8"

rm -rf "$S"
