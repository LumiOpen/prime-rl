"""Is the reported gradient norm wrong, or are the gradients genuinely huge?

The trainer reports `Grad. Norm` of 1e7-1e10 for a 0.6B model, where a correct
CE gradient should be O(1-10). With the default `max_norm = 1.0` nothing learns
at all; with clipping effectively disabled (1e12) it learns normally. Two very
different causes, needing different fixes:

  (a) the REPORTED NORM is wrong -> a bug in the norm computation/reduction, and
      clipping is restorable once fixed.
  (b) the gradients really are that large -> AdamW's scale invariance explains
      why training works unclipped, and `max_norm = 1.0` fails because the norm
      *fluctuates* ~100x step to step, making the effective step size random.
      The fix would then be in loss scaling, not in the norm.

This wraps the standalone `sft` entrypoint and monkeypatches torchtitan's
`clip_grad_norm_` to, before clipping, compute the norm independently and dump
the per-parameter breakdown. Run on ONE GPU so there is no sharding ambiguity:
if manual and reported disagree on a single rank, the distributed reduction is
not the culprit and the problem is in the norm computation itself.

Usage (inside the container, single GPU):
    uv run python opd_val/gradnorm_probe.py @ examples/basic/reverse-text/sft.toml \
        --model.compile None --optim.max-norm 1.0 --max-steps 3 ...
"""

import os
import sys

import torch
import torch.nn.functional as F
import torchtitan.distributed.utils as tt_utils
from torch.nn.attention import SDPBackend, sdpa_kernel

_orig_clip = tt_utils.clip_grad_norm_
_step = 0


def _grad_norm_manual(params) -> tuple[float, list[tuple[str, float]]]:
    """Global L2 norm over all gradients, plus the largest per-parameter norms.

    DTensor grads are reduced via their local shard; on a single rank that is
    the whole tensor, so this is exact here and needs no cross-rank reduction.
    """
    total_sq = torch.zeros((), dtype=torch.float64, device="cuda")
    per_param: list[tuple[str, float]] = []
    for name, p in params:
        g = p.grad
        if g is None:
            continue
        local = g.to_local() if hasattr(g, "to_local") else g
        sq = local.detach().to(torch.float64).pow(2).sum()
        total_sq += sq
        per_param.append((name, float(sq.sqrt().item())))
    return float(total_sq.sqrt().item()), per_param


def _sdpa_compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
    """Reference replacement for the flash-attn varlen kernel.

    Same maths, but the pure-PyTorch MATH backend, so nothing touches a fused
    ROCm kernel. If the gradient norm is sane here and enormous with flash-attn,
    the attention backward is the source.
    """
    cu = cu_seqlens.tolist()
    rep = q.shape[1] // k.shape[1]  # GQA
    outs = []
    for i in range(len(cu) - 1):
        s_, e_ = cu[i], cu[i + 1]
        if e_ <= s_:
            continue
        qi = q[s_:e_].transpose(0, 1).unsqueeze(0)
        ki = k[s_:e_].transpose(0, 1).unsqueeze(0)
        vi = v[s_:e_].transpose(0, 1).unsqueeze(0)
        if rep > 1:
            ki = ki.repeat_interleave(rep, dim=1)
            vi = vi.repeat_interleave(rep, dim=1)
        with sdpa_kernel([SDPBackend.MATH]):
            o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def main() -> None:
    from prime_rl.trainer.sft import train as sft_train

    named = {}

    def instrumented(parameters, max_norm, **kwargs):
        global _step
        _step += 1
        params = list(parameters)
        # Recover names by identity so the breakdown is readable.
        by_id = {id(p): n for n, p in named.items()}
        named_params = [(by_id.get(id(p), "?"), p) for p in params]

        manual, per_param = _grad_norm_manual(named_params)
        finite = all(
            torch.isfinite(p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).all().item()
            for _, p in named_params
            if p.grad is not None
        )
        reported = _orig_clip(iter(params), max_norm, **kwargs)
        rep = float(reported.item()) if hasattr(reported, "item") else float(reported)

        per_param.sort(key=lambda kv: -kv[1])
        shapes = {n: tuple(p.shape) for n, p in named_params}
        def _fmt(n, v):
            sh = shapes.get(n, ())
            numel = 1
            for d in sh:
                numel *= d
            return f"{n} {sh} norm={v:.3e} rms={v / max(numel, 1) ** 0.5:.3e}"
        top = "\n[probe]     ".join(_fmt(n, v) for n, v in per_param[:5])
        share = (per_param[0][1] ** 2 / max(manual**2, 1e-30)) * 100 if per_param else 0.0
        print(
            f"[probe] step {_step}: reported={rep:.6e}  manual={manual:.6e}  "
            f"ratio={rep / max(manual, 1e-30):.4f}  all_finite={finite}  "
            f"n_params={len(per_param)}  top1_share={share:.1f}%",
            flush=True,
        )
        print(f"[probe]   largest per-param norms:\n[probe]     {top}", flush=True)
        return reported

    if os.environ.get("PROBE_SDPA") == "1":
        from prime_rl.trainer.models.layers.attn import FlashAttention

        FlashAttention._compute_attention = _sdpa_compute_attention
        print("[probe] attention: MATH SDPA (flash-attn kernel bypassed)", flush=True)
    else:
        print("[probe] attention: flash-attn varlen (default)", flush=True)

    tt_utils.clip_grad_norm_ = instrumented
    sft_train.clip_grad_norm_ = instrumented

    # Capture parameter names once the model exists, for the breakdown above.
    _orig_setup = sft_train.setup_model

    def setup_model_capture(*a, **kw):
        model = _orig_setup(*a, **kw)
        named.update(dict(model.named_parameters()))
        return model

    sft_train.setup_model = setup_model_capture

    # Same entry as the sft trainer's own main().
    sft_train.train(sft_train.cli(sft_train.SFTConfig))


if __name__ == "__main__":
    sys.exit(main())
