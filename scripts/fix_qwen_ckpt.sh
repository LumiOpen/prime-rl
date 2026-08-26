#!/bin/bash
# Restore the config files that Qwen checkpoint export loses.
#
# - preprocessor_config.json is omitted entirely (vLLM fails to load without it)
# - generation_config.json is regenerated from config.json, which drops
#   <|im_end|> from eos_token_id along with the sampling defaults, so
#   generations no longer stop at the end of an assistant turn
#
# Both are copied from the initial SFT model, which is the authoritative source.
#
# Usage: fix_qwen_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]
#
# Example (all steps in a run):
#   fix_qwen_ckpt.sh /path/to/init_model outputs/my-run/weights/step_*

set -euo pipefail

INIT=${1:?Usage: fix_qwen_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]}
shift

for F in preprocessor_config.json generation_config.json; do
    [ -f "$INIT/$F" ] || { echo "ERROR: $F not found in $INIT"; exit 1; }
done

for CKPT in "$@"; do
    cp "$INIT/preprocessor_config.json" "$CKPT/preprocessor_config.json"
    cp "$INIT/generation_config.json" "$CKPT/generation_config.json"
    echo "Fixed: $CKPT"
done
