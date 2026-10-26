#!/usr/bin/env bash
set -euo pipefail

export IMG="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
#export IMG="/scratch/project_462000963/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif"

singularity shell --cleanenv \
-B "/scratch/project_462001516" \
"$IMG"

# --bind /dev/kfd \
# --bind /dev/dri \