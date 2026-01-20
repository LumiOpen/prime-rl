#!/bin/bash
#SBATCH --job-name=prime-rl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/distributed-rl-%j.out
#SBATCH --error=logs/distributed-rl-%j.err

set -euo pipefail

# Configuration
export NUM_INFERENCE_NODES=${NUM_INFERENCE_NODES:-2}
export IMG="/shared_silo/scratch/containers/rocm_vllm-dev_nightly_0624_rc2_0624_rc2_20250620.sif"

# Config files
export INFERENCE_CONFIG=${INFERENCE_CONFIG:-"configs/deepscaler/stage1/infer.toml"}
export ORCHESTRATOR_CONFIG=${ORCHESTRATOR_CONFIG:-"configs/deepscaler/stage1/orch.toml"}
export TRAINER_CONFIG=${TRAINER_CONFIG:-"configs/deepscaler/stage1/train.toml"}
export OUTPUT_DIR=./outputs/stage1

# Get node list and master node
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_NODE=${NODELIST[0]}
MASTER_ADDR=$MASTER_NODE
VLLM_MASTER_NODE=${NODELIST[1]}

echo "Master node: $MASTER_NODE ($MASTER_ADDR)"
echo "All nodes: ${NODELIST[@]}"
echo "Number of inference nodes: $NUM_INFERENCE_NODES"

# Singularity bind mounts
BIND_MOUNTS="-B /shared_silo/scratch/kahakala/,/home/kahakala@amd.com --bind /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so -B /usr/share/libdrm:/usr/share/libdrm:ro"

# Define common environment setup as a string to be executed in each container
ENV_SETUP='
    export PYTHONUSERBASE=/shared_silo/scratch/kahakala/lib/prime-rl/pythonuserbase
    export HF_CACHE=/shared_silo/scratch/kahakala/lib/open-r1/hf_cache
    export HF_HOME=/shared_silo/scratch/kahakala/lib/open-r1/hf_home
    export TRITON_CACHE_DIR=./triton_cache2
    export XDG_CACHE_HOME=./xdg_cache
    export TORCHINDUCTOR_CACHE_DIR=./torchinductor_cache
    export PATH=$PYTHONUSERBASE/bin:$PATH
    export UV_CACHE_DIR=./uv_cache
    export NCCL_SOCKET_IFNAME=eno0
'

# Start inference master on first inference node
echo "Starting inference master on $VLLM_MASTER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_MASTER_NODE --job-name=inference-master --exclusive \
    singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
    bash -c "
        $ENV_SETUP
        export DATA_PARALLEL_ADDRESS=$VLLM_MASTER_NODE
        export DATA_PARALLEL_RPC_PORT=29500
        
        uv run --no-sync inference @ $INFERENCE_CONFIG \
            --data-parallel-size $((NUM_INFERENCE_NODES * 8)) \
            --tensor-parallel-size 1 \
            --data-parallel-size-local 8 \
            --data-parallel-address \$DATA_PARALLEL_ADDRESS \
            --data-parallel-rpc-port \$DATA_PARALLEL_RPC_PORT
    " &

INFERENCE_MASTER_PID=$!
sleep 10

# Start inference workers on remaining inference nodes
for ((i=2; i<=NUM_INFERENCE_NODES; i++)); do
    NODE=${NODELIST[$i]}
    RANK=$(((i-1) * 8))
    echo "Starting inference worker on $NODE (rank $RANK)..."
    
    srun --nodes=1 --ntasks=1 --nodelist=$NODE --job-name=inference-worker-$i --exclusive \
        singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
        bash -c "
            $ENV_SETUP
            export DATA_PARALLEL_ADDRESS=$VLLM_MASTER_NODE
            export DATA_PARALLEL_RPC_PORT=29500
            
            uv run --no-sync inference @ $INFERENCE_CONFIG \
                --data-parallel-size $((NUM_INFERENCE_NODES * 8)) \
                --tensor-parallel-size 1 \
                --data-parallel-size-local 8 \
                --data-parallel-address \$DATA_PARALLEL_ADDRESS \
                --data-parallel-rpc-port \$DATA_PARALLEL_RPC_PORT \
                --data-parallel-start-rank $RANK \
                --headless
        " &
    
    sleep 5
done

# Wait for inference services to be ready
sleep 30

# Run orchestrator and trainer on master node
echo "Starting orchestrator and trainer on master node $MASTER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --job-name=rl-training-master \
    singularity exec --rocm --cleanenv $BIND_MOUNTS "$IMG" \
    bash -c "
        $ENV_SETUP
        export INFERENCE_SERVER_IP=$VLLM_MASTER_NODE
        export DATA_PARALLEL_RPC_PORT=29500
        
        # Start orchestrator in background
        uv run --no-sync orchestrator @ $ORCHESTRATOR_CONFIG \
            --client.base-url http://\$INFERENCE_SERVER_IP:8000/v1 \
            --output-dir "$OUTPUT_DIR/run_default" &
        ORCH_PID=\$!
        
        sleep 10
        
        # Start trainer (foreground)
        uv run torchrun \
            --nproc-per-node 8 \
            --local-ranks-filter 0 \
            src/prime_rl/trainer/rl/train.py @ $TRAINER_CONFIG \
            --output-dir $OUTPUT_DIR
        
        # Cleanup
        # kill \$ORCH_PID 2>/dev/null || true
    "

echo "Job completed successfully!"