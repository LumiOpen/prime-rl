export PYTHONUSERBASE=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/pythonuserbase
export HF_CACHE=/shared_silo/scratch/mmelnik/cache/hf_cache
export HF_HOME=/shared_silo/scratch/mmelnik/cache/hf_home
export TRITON_CACHE_DIR=/shared_silo/scratch/mmelnik/cache/triton_cache
export XDG_CACHE_HOME=/shared_silo/scratch/mmelnik/cache/xdg_cache
export TORCHINDUCTOR_CACHE_DIR=/shared_silo/scratch/mmelnik/cache/torchinductor_cache
export PATH=$PYTHONUSERBASE/bin:$PATH
export UV_CACHE_DIR=./uv_cache

export NCCL_SOCKET_IFNAME=eno0
export GLOO_SOCKET_IFNAME=eno0
export RCCL_NET_GDR_LEVEL=2

# export RCCL_NET_GDR_LEVEL=2           # GPU Direct RDMA - CRITICAL
# export RCCL_ENABLE_DIRECT_COPY=1      # Direct GPU-GPU copy - CRITICAL
# export NCCL_BUFFSIZE=16777216         # 16MB comm buffer - CRITICAL
# export RCCL_MSCCL_ENABLE=0
# export HSA_ENABLE_SDMA=1
# export RCCL_P2P_DISABLE=0
# export HSA_FORCE_FINE_GRAIN_PCIE=1
# export RCCL_MIN_NCHANNELS=8
# export RCCL_NCHANNELS=8
# export NCCL_NTHREADS=512
# export NCCL_NET_GDR_READ=1
# export RCCL_ENABLE_SMI_P2P=1

export LD_LIBRARY_PATH=/shared_silo/scratch/mmelnik/lib/rocm72/prime-rl/.venv/lib/python3.12/site-packages/torch/lib:/opt/rocm-6.4.2/lib:$LD_LIBRARY_PATH
