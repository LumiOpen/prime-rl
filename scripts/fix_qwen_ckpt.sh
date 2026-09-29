#!/bin/bash
# Restore the config files that Qwen checkpoint export loses, and make the chat
# template reason by default.
#
# - preprocessor_config.json is omitted entirely (vLLM fails to load without it)
# - generation_config.json is regenerated from config.json, which drops
#   <|im_end|> from eos_token_id along with the sampling defaults, so
#   generations no longer stop at the end of an assistant turn
#
# Both are copied from the initial SFT model, which is the authoritative source.
#
# - chat_template.jinja defaults enable_thinking to false on the 0.8B and 2B
#   Qwen3.5 sizes, prefilling a closed empty <think> block so generation starts
#   after reasoning has already ended. These are not hybrid models, so the flag
#   is flipped to default-on; an explicit enable_thinking=false still suppresses.
#   The template is patched in the init model too, so the two never disagree.
#   The original is kept alongside as chat_template.jinja.orig.
#
#   Templates with no enable_thinking guard at all are left untouched: they
#   prefill nothing, so reasoning is a property of the weights, not the prompt.
#
# Usage: fix_qwen_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]
#
# Example (all steps in a run):
#   fix_qwen_ckpt.sh /path/to/init_model outputs/my-run/weights/step_*

set -euo pipefail

SUPPRESSED="{%- if enable_thinking is defined and enable_thinking is true %}"
THINKING="{%- if enable_thinking is not defined or enable_thinking is true %}"
# The 4B and larger sizes ship this inverted guard, which already reasons by
# default and must be left alone.
VENDOR_THINKING="{%- if enable_thinking is defined and enable_thinking is false %}"

enable_thinking_by_default() {
    local tpl="$1/chat_template.jinja"
    [ -f "$tpl" ] || { echo "ERROR: chat_template.jinja not found in $1"; exit 1; }
    if grep -qF "$THINKING" "$tpl" || grep -qF "$VENDOR_THINKING" "$tpl"; then
        echo "  template already reasons by default"
        return
    fi
    # Some checkpoints drop the guard entirely and prefill nothing after
    # <|im_start|>assistant, leaving the decision to the model. There is no
    # branch to flip, so leave the template alone.
    if ! grep -q "enable_thinking" "$tpl"; then
        echo "  template has no enable_thinking guard — left as is"
        return
    fi
    grep -qF "$SUPPRESSED" "$tpl" || { echo "ERROR: unrecognised chat template in $1"; exit 1; }
    [ -f "$tpl.orig" ] || cp "$tpl" "$tpl.orig"
    python3 - "$tpl" "$SUPPRESSED" "$THINKING" <<'PY'
import sys
path, old, new = sys.argv[1:4]
src = open(path).read()
assert src.count(old) == 1, "expected exactly one enable_thinking guard, found %d" % src.count(old)
open(path, "w").write(src.replace(old, new))
PY
    echo "  template patched to reason by default"
}

INIT=${1:?Usage: fix_qwen_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]}
shift

for F in preprocessor_config.json generation_config.json; do
    [ -f "$INIT/$F" ] || { echo "ERROR: $F not found in $INIT"; exit 1; }
done

echo "Init model: $INIT"
enable_thinking_by_default "$INIT"

for CKPT in "$@"; do
    cp "$INIT/preprocessor_config.json" "$CKPT/preprocessor_config.json"
    cp "$INIT/generation_config.json" "$CKPT/generation_config.json"
    echo "Fixed: $CKPT"
    enable_thinking_by_default "$CKPT"
done
