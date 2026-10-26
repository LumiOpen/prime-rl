#!/usr/bin/env bash
BASE=/scratch/project_462001516/users/khakala/lib/prime-rl-upstream-main

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
