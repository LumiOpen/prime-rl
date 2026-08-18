#!/bin/bash
#SBATCH --job-name=opd-smoke
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=./opd_val/logs/smoke_%j.out
#SBATCH --error=./opd_val/logs/smoke_%j.err

# Container triage for the OPD validation run: for each candidate image, check
# that the repo venv's torch sees the GPUs and that vLLM 0.25.1 imports.
# Then, in the first working image, boot the teacher model and prefill-score a
# token sequence through /inference/v1/generate — the exact call OPD makes.

set -uo pipefail

PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
cd "$PRIME_RL_DIR"

CANDIDATES=(
    "/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
    "/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif"
    "/shared_silo/scratch/containers/primus_v26.5-pytorch2.12-te2.15.sif"
)

# NOTE: --cleanenv --containall strips SLURM_* from the container env, so the
# job id must be substituted here on the host. Escaping it (\${SLURM_JOB_ID})
# expands to empty inside the container and every job collides on the same
# /tmp/xdg_<host>_ path — which fails with EACCES once another user owns it.
SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup.sh"
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
ENVEOF

CONTAINER_FLAGS="--rocm --cleanenv --containall"
BIND_MOUNTS="-B /shared_silo/scratch:/shared_silo/scratch:rw"
BIND_MOUNTS="$BIND_MOUNTS -B $HOME:$HOME:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

mkdir -p "$SCRATCH"

echo "=========================================="
echo " Node: $SLURM_NODELIST  Job: $SLURM_JOB_ID"
echo " Scratch: $SCRATCH"
echo "=========================================="

WORKING_IMG=""
for IMG in "${CANDIDATES[@]}"; do
    echo ""
    echo "### $IMG"
    if [ ! -f "$IMG" ]; then echo "  MISSING - skip"; continue; fi
    singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
        bash -c "source $ENV_SETUP && uv run python -c \"
import torch
print('  torch', torch.__version__, 'hip', torch.version.hip)
n = torch.cuda.device_count()
print('  device_count', n)
assert n > 0, 'NO GPUS VISIBLE'
print('  gpu0', torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device='cuda'); print('  matmul ok', float((x@x).sum()) == float((x@x).sum()))
import vllm; print('  vllm', vllm.__version__)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse
print('  token_in_token_out protocol import OK')
print('  RESULT: PASS')
\"" 2>&1 | sed 's/^/  /'
    if [ ${PIPESTATUS[0]} -eq 0 ] && [ -z "$WORKING_IMG" ]; then
        WORKING_IMG="$IMG"
    fi
done

echo ""
echo "=========================================="
if [ -z "$WORKING_IMG" ]; then
    echo "NO WORKING CONTAINER FOUND"
    exit 1
fi
echo "First working container: $WORKING_IMG"
echo "=========================================="

# ── Stage 2: boot the teacher and exercise the exact OPD prefill-scoring call ──
TEACHER="PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
PORT=8011
echo ""
echo "Booting $TEACHER on GPU 0, port $PORT ..."
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$WORKING_IMG" \
    bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=0 && \
        uv run inference --model.name $TEACHER --server.port $PORT \
        --gpu-memory-utilization 0.5 --model.enforce-eager" \
    > "$OUT/logs/smoke_teacher_${SLURM_JOB_ID}.log" 2>&1 &
TPID=$!

for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null && break
    sleep 2
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; then
    echo "ERROR: teacher server never became healthy. Tail of its log:"
    tail -40 "$OUT/logs/smoke_teacher_${SLURM_JOB_ID}.log"
    kill $TPID 2>/dev/null; exit 1
fi
echo "Teacher healthy."

echo ""
echo "Calling prefill_logprobs() — the OPD scoring path:"
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$WORKING_IMG" \
    bash -c "source $ENV_SETUP && uv run python -c \"
import asyncio
from openai import AsyncOpenAI
from prime_rl.utils.client import prefill_logprobs
from transformers import AutoTokenizer

model = '$TEACHER'
tok = AutoTokenizer.from_pretrained(model)
ids = tok.apply_chat_template(
    [{'role':'user','content':'Reverse this: hello world'},
     {'role':'assistant','content':'<reversed_text>dlrow olleh</reversed_text>'}],
    tokenize=True)
print('  num tokens:', len(ids))

async def main():
    c = AsyncOpenAI(base_url='http://127.0.0.1:$PORT/v1', api_key='EMPTY')
    lp = await prefill_logprobs(c, model, ids)
    print('  logprobs returned:', len(lp))
    assert len(lp) == len(ids), f'LENGTH MISMATCH {len(lp)} != {len(ids)}'
    assert lp[0] == 0.0, 'leading token should be 0.0'
    assert all(x <= 0.0 for x in lp), 'logprobs must be <= 0'
    print('  first 8:', [round(x,3) for x in lp[:8]])
    print('  mean over non-leading:', round(sum(lp[1:])/len(lp[1:]), 4))
    print('  RESULT: PREFILL SCORING PASS')
asyncio.run(main())
\"" 2>&1 | sed 's/^/  /'
RC=${PIPESTATUS[0]}

kill $TPID 2>/dev/null || true
wait $TPID 2>/dev/null || true

echo ""
if [ $RC -eq 0 ]; then
    echo "SMOKE PASSED. Use container: $WORKING_IMG"
else
    echo "SMOKE FAILED at the prefill-scoring step (rc=$RC)."
fi
exit $RC
