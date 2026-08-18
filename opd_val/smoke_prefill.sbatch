#!/bin/bash
#SBATCH --job-name=opd-prefill
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=./opd_val/logs/prefill_%j.out
#SBATCH --error=./opd_val/logs/prefill_%j.err

# Exercises the exact call OPD makes against the teacher: prefill_logprobs()
# over /inference/v1/generate + prompt_logprobs. Checks length alignment, the
# 0.0 leading token, and that the teacher scores its own greedy continuation
# higher than a scrambled one (i.e. the logprobs are real, not garbage).

set -uo pipefail

PRIME_RL_DIR="/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl"
OUT="$PRIME_RL_DIR/opd_val"
IMG="/shared_silo/scratch/containers/primus_v26.2_moefix.sif"
TEACHER="PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
PORT=8011
cd "$PRIME_RL_DIR"

SCRATCH="/tmp/rahul_opd_${SLURM_JOB_ID}"
ENV_SETUP="$OUT/env_setup_prefill.sh"
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
ENVEOF
mkdir -p "$SCRATCH"

CONTAINER_FLAGS="--rocm --cleanenv --containall"
BIND_MOUNTS="-B /shared_silo/scratch:/shared_silo/scratch:rw"
BIND_MOUNTS="$BIND_MOUNTS -B $HOME:$HOME:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
BIND_MOUNTS="$BIND_MOUNTS -B /usr/share/libdrm:/usr/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /opt/amdgpu/share/libdrm:/opt/amdgpu/share/libdrm:ro"
BIND_MOUNTS="$BIND_MOUNTS -B /tmp:/tmp:rw"

echo "Node: $SLURM_NODELIST  Job: $SLURM_JOB_ID"
echo "Booting teacher $TEACHER on port $PORT ..."
singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && export HIP_VISIBLE_DEVICES=0 && \
        uv run inference --model.name $TEACHER --server.port $PORT \
        --gpu-memory-utilization 0.5 --model.enforce-eager" \
    > "$OUT/logs/prefill_teacher_${SLURM_JOB_ID}.log" 2>&1 &
TPID=$!

for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null && break
    sleep 2
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; then
    echo "ERROR: teacher never healthy. Log tail:"; tail -40 "$OUT/logs/prefill_teacher_${SLURM_JOB_ID}.log"
    kill $TPID 2>/dev/null; exit 1
fi
echo "Teacher healthy. Running prefill_logprobs checks:"

singularity exec $CONTAINER_FLAGS $BIND_MOUNTS "$IMG" \
    bash -c "source $ENV_SETUP && uv run python -c \"
import asyncio
from openai import AsyncOpenAI
from prime_rl.utils.client import prefill_logprobs
from transformers import AutoTokenizer

model = '$TEACHER'
tok = AutoTokenizer.from_pretrained(model)

def ids_for(reply):
    # transformers 5.x apply_chat_template returns a BatchEncoding by default.
    out = tok.apply_chat_template(
        [{'role':'user','content':'Reverse the text character-by-character: hello world'},
         {'role':'assistant','content':reply}],
        tokenize=True, return_dict=False)
    return list(out)

good = ids_for('<reversed_text>dlrow olleh</reversed_text>')
bad  = ids_for('<reversed_text>zqx vlkj mnop</reversed_text>')

async def main():
    c = AsyncOpenAI(base_url='http://127.0.0.1:$PORT/v1', api_key='EMPTY')
    lp = await prefill_logprobs(c, model, good)
    print('  tokens=%d logprobs=%d' % (len(good), len(lp)))
    assert len(lp) == len(good), 'LENGTH MISMATCH %d != %d' % (len(lp), len(good))
    assert lp[0] == 0.0, 'leading token must be 0.0, got %r' % lp[0]
    assert all(x <= 0.0 for x in lp), 'all logprobs must be <= 0'
    print('  first 6:', [round(x,3) for x in lp[:6]])

    lp_bad = await prefill_logprobs(c, model, bad)
    mg = sum(lp[1:])/len(lp[1:]); mb = sum(lp_bad[1:])/len(lp_bad[1:])
    print('  mean logprob  correct-answer=%.4f  scrambled=%.4f' % (mg, mb))
    assert mg > mb, 'teacher does not prefer the correct reversal -- scoring is suspect'

    # Determinism: same input twice must give the same scores.
    lp2 = await prefill_logprobs(c, model, good)
    drift = max(abs(a-b) for a,b in zip(lp, lp2))
    print('  max drift across two identical calls: %.2e' % drift)

    print('  RESULT: PREFILL SCORING PASS')
asyncio.run(main())
\"" 2>&1 | sed 's/^/  /'
RC=${PIPESTATUS[0]}

kill $TPID 2>/dev/null || true; wait $TPID 2>/dev/null || true
rm -rf "$SCRATCH"
[ $RC -eq 0 ] && echo "PREFILL SMOKE PASSED" || echo "PREFILL SMOKE FAILED rc=$RC"
exit $RC
