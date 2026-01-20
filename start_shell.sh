#!/usr/bin/env bash
set -euo pipefail

#export IMG="/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.1_20251103.sif"
export IMG="/shared_silo/scratch/containers/rocm_vllm-dev_nightly_0624_rc2_0624_rc2_20250620.sif"

singularity shell --rocm --cleanenv \
-B "/shared_silo/scratch/kahakala/,/home/kahakala@amd.com" \
--bind /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so \
-B /usr/share/libdrm:/usr/share/libdrm:ro \
"$IMG"

# -B /usr/share/libdrm:/usr/share/libdrm:ro \