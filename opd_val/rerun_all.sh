#!/bin/bash
# Re-run the GSM8K distillation matrix on a CORRECT attention backward.
#
# Everything in opd_val/ before 2026-08-18 was measured with
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE, whose backward returns gradients
# ~1.4e7x too large (FINDINGS.md 2.1). Those runs are degraded, not merely
# uncertain: on reverse-text sft the same config reached loss 3.7183 with the
# bug and 0.9469 without it.
#
# Configs now leave max_norm at the prime-rl default; the 1e12 workaround is
# gone. Arms are submitted in priority order so that if the night runs out of
# capacity, the important cells are the ones that completed.
#
# Each training job gets a dependent eval job (afterok). The in-run eval is
# unreliable on this task -- it builds eval clients as `openai_chat_completions`
# rather than the renderer, so `enable_thinking = false` never applies and it
# reports ~95% truncation for every arm. Offline checkpoint scoring is the real
# measurement.
set -uo pipefail
cd /shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl
OUT=opd_val
STEPS="${STEPS:-50 150 300}"   # checkpoints to score; each costs a vLLM spin-up

# name              config                  teacher?
ARMS=(
  "R-opd-s0         g8_opd_s0.toml          1"
  "R-grpo-s0        g8_grpo_s0.toml         0"
  "R-opd-k20        g8_opd_k20.toml         1"
  "R-opd-lr1e-5     g8_opd_lr1e-5.toml      1"
  "R-grpo-lr1e-5    g8_grpo_lr1e-5.toml     0"
  "R-opd-s1         g8_opd_s1.toml          1"
  "R-grpo-s1        g8_grpo_s1.toml         0"
  "R-opd-s2         g8_opd_s2.toml          1"
  "R-grpo-s2        g8_grpo_s2.toml         0"
  "R-opd-k100       g8_opd_k100.toml        1"
)

# Tier 2, submitted separately once tier 1 was known to run clean: the SFT
# reference plus every variant whose prior result is in doubt -- the four
# collapsed top-k formulations and the three MOPD loss ablations.
ARMS_TIER2=(
  "R-sft            gsm8k_sft.toml          1"
  "R-k20-renorm-rev g8_k20_renorm_rev.toml  1"
  "R-k20-renorm-mix g8_k20_renorm_mixed.toml 1"
  "R-k20-none-fwd   g8_k20_none_forward.toml 1"
  "R-k20-none-mix   g8_k20_none_mixed.toml  1"
  "R-k0-mopd        g8_k0_mopd.toml         1"
  "R-k0-noratio     g8_k0_noratio.toml      1"
  "R-k0-twosided    g8_k0_twosided.toml     1"
)
[ "${TIER:-1}" = "2" ] && ARMS=("${ARMS_TIER2[@]}")

echo "=== submitting $( { for a in "${ARMS[@]}"; do echo "$a"; done; } | wc -l ) arms + dependent evals ==="
printf '%-16s %-24s %-8s %-9s %s\n' ARM CONFIG TEACHER TRAIN EVAL
for A in "${ARMS[@]}"; do
    set -- $A; NAME=$1; CFG=$2; TCH=$3
    if [ "$TCH" = "1" ]; then
        TJ=$(TEACHER_MODEL="Qwen/Qwen3-8B" TEACHER_GPU="4,5" \
             sbatch --parsable --job-name="$NAME" \
               --export=ALL,CONFIG="$PWD/$OUT/$CFG",NEED_TEACHER=1,TEACHER_MODEL="Qwen/Qwen3-8B",TEACHER_GPU="4,5" \
               $OUT/rl_opd.sbatch "$NAME")
    else
        TJ=$(sbatch --parsable --job-name="$NAME" \
               --export=ALL,CONFIG="$PWD/$OUT/$CFG",NEED_TEACHER=0 \
               $OUT/rl_opd.sbatch "$NAME")
    fi
    # The run directory is named <arm>_<jobid>, known only after submission.
    EJ=$(sbatch --parsable --dependency=afterok:$TJ --job-name="eval-$NAME" \
           --export=ALL,RUNS="${NAME}_${TJ}",STEPS="$STEPS" \
           $OUT/eval_ckpts.sbatch)
    printf '%-16s %-24s %-8s %-9s %s\n' "$NAME" "$CFG" "$TCH" "$TJ" "$EJ"
done
echo ""
echo "collect with: uv run python $OUT/collect_rerun.py"
