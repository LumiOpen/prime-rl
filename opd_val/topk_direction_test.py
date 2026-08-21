"""Does the top-k branch push the student toward the teacher?

The k=0 branch turned out to contribute no gradient at all, so this checks the
k>=1 branch on the property that matters.

IMPORTANT — test in LOGIT space, not logprob space. The student's top-k
logprobs come from a shared full-vocab log_softmax, so a gradient component
common to every entry cancels once it reaches the logits (d logp_i/d logit_j =
delta_ij - p_j). Judging direction on d/d logp therefore mislabels forward KL:
its gradient is -q_i, negative everywhere, which looks like "raise everything"
but is really "raise in proportion to teacher mass" and is correct after
normalization.

Setup: teacher prefers index 0, student prefers index 3 (exactly reversed).
Correct behaviour, measured on the logits: descent RAISES logit 0 and LOWERS
logit 3, i.e. d loss/d logit[0] < 0 < d loss/d logit[3].

Run: uv run python opd_val/topk_direction_test.py
"""

import torch

from prime_rl.configs.trainer import RefKLConfig
from prime_rl.trainer.rl.loss import _topk_divergence

VOCAB, K = 16, 4
IDS = torch.tensor([[0, 1, 2, 3]])  # the teacher's top-k ids
TEACHER_P = torch.tensor([[0.70, 0.15, 0.10, 0.05]])


def student_logits() -> torch.Tensor:
    """Logits whose softmax over the top-k is the reverse of the teacher's."""
    logits = torch.full((1, VOCAB), -20.0)
    logits[0, :K] = torch.log(torch.tensor([0.05, 0.10, 0.15, 0.70]))
    return logits.requires_grad_(True)


def main() -> None:
    ref_topk = torch.log(TEACHER_P)
    print(f"teacher p (top-k): {[round(v, 2) for v in TEACHER_P[0].tolist()]}")
    print("student p (top-k): [0.05, 0.1, 0.15, 0.7]   <- exactly reversed")
    print()
    print(f"{'norm':<13}{'kl_type':<9}{'d/dlogit[0]':>13}{'d/dlogit[3]':>13}   verdict")
    bad = []
    for norm in ("residual", "renormalize", "none"):
        for kl in ("reverse", "forward", "mixed"):
            logits = student_logits()
            # Exactly what train.py does: full-vocab log_softmax, gathered at the
            # teacher's ids.
            trainer_topk = torch.log_softmax(logits, dim=-1).gather(-1, IDS)
            per_token_kl, _, _ = _topk_divergence(
                trainer_topk, ref_topk, RefKLConfig(topk_normalization=norm, kl_type=kl)
            )
            per_token_kl.sum().backward()
            g0, g3 = logits.grad[0, 0].item(), logits.grad[0, 3].item()
            ok = g0 < 0 < g3
            if not ok:
                bad.append(f"{norm}/{kl}")
            print(f"{norm:<13}{kl:<9}{g0:>13.5f}{g3:>13.5f}   {'OK' if ok else '*** WRONG ***'}")
    print()
    print(f"{len(bad)}/9 point the wrong way" + (f": {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
