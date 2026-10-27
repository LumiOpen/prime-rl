#!/usr/bin/env bash
# The clone root, derived from this file's own location so nothing here is user-specific.
# BASH_SOURCE rather than $0 because this script is sourced, not executed.
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python / pip
export PIP_CACHE_DIR=$BASE/cache/pip
export UV_CACHE_DIR=$BASE/cache/uv
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT=$BASE/.venv
export VIRTUAL_ENV=$BASE/.venv
export PATH=$HOME/.local/bin:$BASE/.venv/bin:$PATH

# HuggingFace (covers hub, datasets, transformers)
export HF_HOME=$BASE/cache/huggingface

# PyTorch
export TORCH_HOME=$BASE/cache/torch
export TORCH_EXTENSIONS_DIR=$BASE/cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$BASE/cache/torchinductor

# Triton
export TRITON_CACHE_DIR=$BASE/cache/triton

# MIOpen (ROCm kernel cache)
export MIOPEN_USER_DB_PATH=$BASE/cache/miopen
export MIOPEN_CACHE_DIR=$BASE/cache/miopen

# Compiler cache
export CCACHE_DIR=$BASE/cache/ccache

# vLLM
export VLLM_CACHE_ROOT=$BASE/cache/vllm
