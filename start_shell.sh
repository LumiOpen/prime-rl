#!/usr/bin/env bash
set -euo pipefail

export IMG="/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif"
#export IMG="/shared_silo/scratch/containers/rocm_vllm-dev_nightly_0624_rc2_0624_rc2_20250620.sif"

singularity shell --rocm --cleanenv \
-B "/shared_silo/scratch/,$HOME" \
--bind /usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so \
-B /usr/share/libdrm:/usr/share/libdrm:ro \
"$IMG"

# -B /usr/share/libdrm:/usr/share/libdrm:ro \
