#!/bin/bash
# Fix OLMo3 checkpoints where rope_theta gets stripped to null during export.
# vLLM crashes with: TypeError: pow(NoneType, Tensor)
# Fix: overwrite config.json with the one from the initial SFT model.
#
# Usage: fix_olmo3_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]
#
# Example (all steps in a run):
#   fix_olmo3_ckpt.sh /path/to/init_model outputs/my-run/weights/step_*

set -euo pipefail

INIT=${1:?Usage: fix_olmo3_ckpt.sh <init_model_dir> <checkpoint_dir> [<checkpoint_dir> ...]}
shift

[ -f "$INIT/config.json" ] || { echo "ERROR: config.json not found in $INIT"; exit 1; }

ROPE_THETA=$(python3 -c "import json; c=json.load(open('$INIT/config.json')); print(c.get('rope_theta', 'MISSING'))")
[ "$ROPE_THETA" != "null" ] && [ "$ROPE_THETA" != "MISSING" ] || { echo "ERROR: rope_theta is '$ROPE_THETA' in init model config — something is wrong"; exit 1; }
echo "Init model rope_theta: $ROPE_THETA"

for CKPT in "$@"; do
    CKPT_ROPE=$(python3 -c "import json; c=json.load(open('$CKPT/config.json')); print(c.get('rope_theta', 'MISSING'))" 2>/dev/null || echo "ERROR")
    if [ "$CKPT_ROPE" = "$ROPE_THETA" ]; then
        echo "OK (rope_theta=$CKPT_ROPE): $CKPT"
    else
        cp "$INIT/config.json" "$CKPT/config.json"
        echo "Fixed (was rope_theta=$CKPT_ROPE): $CKPT"
    fi
done
