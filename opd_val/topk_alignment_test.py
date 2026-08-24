"""Is the teacher's top-k support aligned with the policy's logits?

vLLM's ``prompt_logprobs[t]`` is the distribution predicting token ``t`` and puts
the target id first, so the teacher's support obeys

    teacher_topk_ids[t][0] == input_ids[t]

i.e. the "probability of current token" convention, the same one ``ref_logprobs``
uses. The policy's logits are on the *other* convention -- ``logits[t]`` predicts
token ``t+1``, which is exactly why ``labels = shift_tensor_left(input_ids)``.

So there is an invariant that needs no model at all: after ``shift_tensor_right``,
column 0 of the policy's top-k logprobs must equal the scalar ``trainer_logprobs``,
because column 0 *is* the sampled token. If it does not, the support is being
evaluated at the wrong vocab ids.

Run: uv run python opd_val/topk_alignment_test.py
"""

import torch

from prime_rl.trainer.rl.loss import (
    gather_log_softmax,
    selective_log_softmax,
    shift_tensor_left,
    shift_tensor_right,
)

S, V, K = 12, 64, 4


def main() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, S, V)
    input_ids = torch.randint(0, V, (1, S))

    # The teacher's support, exactly as prefill_logprobs_topk emits it: the
    # sampled token first, then k-1 others, on the current-token convention.
    topk_ids = torch.randint(0, V, (1, S, K))
    topk_ids[0, :, 0] = input_ids[0]

    # --- the scalar path, verbatim from train.py ---
    labels = shift_tensor_left(input_ids)
    trainer_logprobs = shift_tensor_right(selective_log_softmax(logits, labels), pad_value=-1e4)

    # --- the top-k path, verbatim from train.py ---
    topk_now = shift_tensor_right(gather_log_softmax(logits, topk_ids), pad_value=-1e4)

    # --- the same path with the ids brought onto the logits' convention ---
    topk_fixed = shift_tensor_right(gather_log_softmax(logits, shift_tensor_left(topk_ids)), pad_value=-1e4)

    print(f"{'t':>3}{'scalar logp':>14}{'topk[:,0] now':>16}{'topk[:,0] fixed':>18}")
    for t in range(1, S):
        print(
            f"{t:>3}{trainer_logprobs[0, t]:>14.5f}"
            f"{topk_now[0, t, 0]:>16.5f}{topk_fixed[0, t, 0]:>18.5f}"
        )

    ref = trainer_logprobs[0, 1:]
    err_now = (topk_now[0, 1:, 0] - ref).abs().max().item()
    err_fixed = (topk_fixed[0, 1:, 0] - ref).abs().max().item()
    print()
    print(f"max |topk[:,0] - scalar logprob|   as shipped: {err_now:.6f}")
    print(f"max |topk[:,0] - scalar logprob|   ids shifted: {err_fixed:.6f}")
    print()
    print("as shipped  ->", "ALIGNED" if err_now < 1e-5 else "*** MISALIGNED ***")
    print("ids shifted ->", "ALIGNED" if err_fixed < 1e-5 else "*** MISALIGNED ***")


if __name__ == "__main__":
    main()
