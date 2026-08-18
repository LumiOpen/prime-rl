# NeMo-RL distillation vs prime-rl `opd` — comparison note

Read of `/shared_silo/scratch/rahul.aralikatte@amd.com/RL` (docs `docs/about/algorithms/{on-policy-distillation,mopd}.md`,
implementation `nemo_rl/algorithms/{distillation.py,advantage_estimator.py,loss/loss_functions.py}`,
config `examples/configs/distillation_math.yaml`). Written 2026-08-17, against our
GSM8K results in `FINDINGS.md` / `EXPERIMENT_LOG.md`.

## TL;DR

NeMo-RL ships **two** distillation algorithms. One of them (MOPD) is
*mathematically identical* to prime-rl's `opd` — independent convergence on the
same estimator, which is a good sign for prime-rl's default. The other is
top-k logit distillation, i.e. the thing I built as `teacher_top_k` — and it
differs from mine in **two specific ways that plausibly explain why mine
collapsed the model**. There are also three smaller settings worth stealing.

## 1. Their MOPD == our `opd`, exactly

`nemo_rl/algorithms/advantage_estimator.py:603`

```python
distill_advantages = (teacher_logprobs - prev_logprobs).detach()
```

versus prime-rl `trainer/rl/loss.py`:

```python
ref_kl = ref_logprobs - trainer_logprobs        # then .detach()ed into pg_loss
```

Same quantity, same detach, same role as a per-token advantage. Their doc even
gives the same justification we reasoned out — that it "needs only the teacher's
log-probability for the *sampled* token rather than the full vocabulary
distribution" — and, like prime-rl, MOPD rides on the existing GRPO trainer
(`adv_estimator: opd`) rather than being a separate loss path.

Worth noting for calibration: **their reference MOPD recipe is a self-distillation
smoke test** (student == teacher, "the OPD loss stays near zero — it is a
correctness smoke test, not a demonstration of distillation gains"). They do not
ship a demonstrated win either.

## 2. Their top-k differs from mine in two ways — and mine is the one that collapsed

`loss_functions.py::_direct_topk_kl`. Two substantive differences:

**(a) They renormalize within the top-k subset. I added an off-support atom.**

```python
student_log_probs_k = torch.log_softmax(student_topk_logits / T, dim=-1)
teacher_log_probs_k = torch.log_softmax(teacher[..., topk_idx] / T, dim=-1)
per_pos = kl_div(..., log_target=True).sum(-1)
```

Both sides are softmaxed *within* K, so the full-vocab partition function
cancels and the objective only matches the **shape** of the distribution inside
the support. Mine instead lumps all off-support mass into an extra atom, which
actively **pulls probability mass onto the teacher's top-k**. That is why our
`TopK Mass` climbed 0.46 → 0.985: the objective was doing what I asked. It is
also the prime suspect for the collapse — my version effectively demands the
policy put ~all its mass on k+1 tokens at every position, which theirs never
asks for. (They do expose `zero_outside_topk: false` as an explicit knob for
this question, and default it off.)

I chose the atom deliberately, reasoning that renormalizing "doesn't anchor mass
into S". That reasoning was sound in the abstract and wrong in practice.

**(b) They default to MIXED forward+reverse KL. I used pure reverse.**

```yaml
kl_type: "mixed"        # forward, reverse, mixed
mixed_kl_weight: 0.5    # weight of the forward KL
```

Forward KL is mode-covering and stable; reverse KL is mode-seeking and can
collapse onto a degenerate mode. Our top-k arms used **pure reverse** and showed
exactly the mode-seeking failure — entropy 0.14 → 1.27, reward → 0.02, and
monotone in how hard the KL was minimized:

| k=20 LR | KL 15 → | KL reduced | reward |
|---|---|---|---|
| 3e-7 | 14.97 | 0% | 0.605 |
| 1e-6 | 11.11 | 27% | 0.541 |
| 3e-6 | 0.82 | 95% | 0.024 |

A 50/50 mix is the single cheapest thing to try against that.

## 3. Architecture: why top-k is cheap for them and expensive for us

| | NeMo-RL | prime-rl |
|---|---|---|
| teacher | Ray worker group **in-cluster**, returns logits (`teacher_policy.get_topk_logits`) | remote vLLM **HTTP** endpoint, `prompt_logprobs` only |
| k | 64 by default | capped by server `max_logprobs` (20 until I raised it) |
| student side | `vocab_parallel_gather_columns` — gathers only K columns, TP-aware | must disable the fused lm head and materialize `[B,T,V]` |
| wire cost | none (tensors in-cluster) | 106 MB/batch at k=100 |

Measured on our side: top-k cost **4.7× orchestrator time** at k=100 and tripled
peak trainer memory at k=20 (18.4 → 57.4 GiB). Both are consequences of the
remote-teacher design, not of the maths. Their `vocab_parallel_gather_columns`
is the efficient version of my "disable the fused lm head" hack, and is roughly
what I sketched as the follow-up (extend `FusedOutputLinear` to gather k extra
indices) before deferring it.

This is architectural: prime-rl's design is "every non-policy model is an
external endpoint". Getting NeMo-RL's efficiency would mean co-locating the
teacher, which is a real departure from that.

## 4. Three MOPD loss settings we don't have

From `mopd.md`, their recommended MOPD block:

```yaml
loss_fn:
  disable_ppo_ratio: true                        # REINFORCE form, no ratio multiply
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: icepop     # hard gate on out-of-bounds IS weights
  reference_policy_kl_penalty: 0.0
```

prime-rl's `ref_kl_loss_fn` instead does:

```python
pg_loss = keep_mask * ref_kl.detach() * importance_ratio   # ratio UNBOUNDED above
kl_loss = loss_mask * log_importance_ratio**2
per_token_loss = -pg_loss + 1e-3 * kl_loss
```

Three differences: we **multiply by an unbounded importance ratio** where they
drop it entirely; we use a **one-sided soft mask** (`probs_diff < -0.2`) where
they use a **hard two-sided ICE-POP gate**; and we carry a `1e-3` squared-log-ratio
term where they set the reference KL penalty to zero. None of these is exotic —
they are all plausible variance-reduction wins on a high-variance estimator, and
none is configurable in prime-rl today (`ref_kl_loss_fn` takes no config at all).

## 5. Smaller things

- **Symmetric temperature** `T` applied to both sides of the KL. We have no
  temperature on `ref_kl`, and we have the separate raw-vs-processed logprob
  subtlety on the teacher side.
- **Their example pair is `Qwen3-1.7B-Base` ← `Qwen3-4B`** — near-identical to
  our `Qwen3-1.7B` ← `Qwen3-8B`. Independent agreement that this is a sane scale
  for a distillation experiment.
- **Multi-teacher routing by agent name** (`teacher_model_by_agent_name`) has no
  prime-rl analogue. prime-rl's per-env algorithms could express something
  similar, but there is no notion of routing one rollout stream to one of
  several teachers.
- **`num_generations_per_prompt: 1`** in their distillation config — no group
  fan-out, consistent with distillation not needing group-relative credit
  (matches prime-rl's note that `group_size` only fans out sampling for `opd`).

## What I'd suggest, in order

1. **Rerun top-k with their formulation** — renormalize within K, and use
   `kl_type = mixed` at 0.5. Two small edits to `ref_kl_loss_fn`. This directly
   tests whether our collapse was the off-support atom, the pure-reverse KL, or
   neither. Cheapest and highest information.
2. **Try MOPD's loss settings on our k=0 arm** — drop the importance-ratio
   multiply, swap the one-sided mask for a two-sided gate. Our k=0 arm already
   works (0.588 → 0.672), so this is a variance-reduction question, and it is
   the only thing that might close the gap to GRPO's 0.726.
3. **Only then consider co-locating the teacher.** It is the real fix for the
   4.7× cost, but it is an architectural change and worth doing only if (1)
   shows top-k is actually worth paying for.

Nothing here overturns the headline result: **GRPO (0.726) still beats `opd`
(0.672 ± 0.002, n=3) on GSM8K**, and NeMo-RL's own docs do not claim otherwise
for their equivalent algorithm.
