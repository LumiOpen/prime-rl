# On-policy distillation on prime-rl / AMD ROCm — findings

**Scope:** `opd` (`algo.type = "opd"`) validation on the bundled `reverse-text` environment, AMD / ROCm 7.2 / vLLM 0.25.1 / torch 2.11.0, SLURM partition `amd-tw-verification`, repo branch `upstream`, working dir `opd_val/`.
**Written:** 2026-08-13. **Companion document:** `opd_val/EXPERIMENT_LOG.md` (exhaustive chronological record). This file is the synthesis; the log is the evidence.

---

## 1. Bottom line up front

> **CORRECTION (2026-08-18).** The "broken gradient-norm computation" described below was misdiagnosed, twice. The norm is computed correctly, and flash-attn is fine. The real cause is a **single environment variable**: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` selects flash-attn's Triton AMD backend, whose *backward* returns gradients ~1.4e7× too large while its forward is exact. It was inherited from `rl_train_singlenode_mixed.sbatch:308` and copied into all twelve `opd_val` scripts, so **every experiment in this investigation ran with it**. Unset, the default composable-kernel backward is correct *and* 3× faster. `max_norm = 1e12` was never a fix — it worked by accident, and `max_norm = 1.0` is the correct default. See §2.1. **Every result in this document was measured on a corrupted backward pass** and needs reproducing before being read as a claim about distillation.

Two infrastructure bugs — a broken gradient-norm computation that made the default `max_norm = 1.0` prevent learning entirely, and a ROCm `torch.compile` crash — blocked *all* training on this cluster and had to be found before any question about distillation could even be asked. Every result recorded before 2026-08-12 evening is an artifact of the first bug and is worthless. With `max_norm = 1e12` the trainer learns normally, and the comparison is then clean and negative: from a base-model student, **`sft` improves reward 0.043 → 0.150 while `opd` collapses 0.046 → 0.002 with 100% truncation**, and `grpo` also improves. A substantial secondary effort — implementing top-k teacher distributions to fix a real coverage defect in the sampled-token KL estimator — demonstrably fixed that defect and **did not change the outcome**: `opd` fails at every k in {8, 20, 50, 100} and at both learning rates tried. The leading hypothesis (untested) is that `opd` needs a student already competent at the task; a ladder of partially-SFT'd students to test that is running now (job 42459).

---

## 1b. THE RE-RUN (2026-08-19) — 18 arms on a correct attention backward

Everything below §1 was measured through the broken Triton backward. The full matrix was
re-run after the fix; **this table supersedes every result in this document.** GSM8K band,
327 problems, student 0.588, teacher 0.893, offline checkpoint scoring, 300 steps.

| arm | step 50 | step 150 | step 300 | vs student |
|---|---|---|---|---|
| **grpo** s0/s1/s2 | 0.789 / 0.784 / 0.800 | 0.873 / 0.880 / 0.854 | **0.9213 / 0.9281 / 0.9167** | **+0.334** |
| **sft** (hard distillation) | 0.7653 | 0.8028 | **0.8127** | **+0.225** |
| k0-noratio (MOPD, no ratio) | 0.6070 | 0.6063 | 0.6239 | +0.036 |
| k0-twosided (ICE-POP gate) | 0.6109 | 0.6024 | 0.5795 | -0.009 |
| k0-mopd (both) | 0.6002 | 0.5780 | 0.5688 | -0.019 |
| **opd** s0/s1/s2 | 0.586 / 0.585 / 0.596 | 0.588 / 0.570 / 0.589 | **0.5650 / 0.5879 / 0.6047** | **-0.002** |
| opd lr 1e-5 | 0.4709 | 0.4213 | 0.4044 | -0.184 |
| grpo lr 1e-5 | 0.6552 | 0.6430 | 0.7408 (step 250) | +0.153 |
| k20 residual/reverse | 0.0902 | 0.0581 | 0.0313 | -0.557 |
| k20 none/mixed | 0.0199 | 0.0206 | 0.0237 | -0.564 |
| k20 none/forward | 0.0122 | 0.0176 | 0.0161 | -0.572 |
| k20 renorm/reverse | 0.0107 | 0.0000 | 0.0000 | -0.588 |
| k20 renorm/mixed | 0.0015 | 0.0000 | 0.0000 | -0.588 |
| k100 residual/reverse | 0.0199 | 0.0268 (step 100) | (timed out) | -0.561 |

**GRPO 0.9220 +/- 0.0057 (n=3)** — it exceeds the 0.893 teacher by 2.9 points.
**OPD 0.5859 +/- 0.0199 (n=3)** — 0.11 sd from the 0.588 student. Across three seeds, on a
correct backward, `opd` moves the model nowhere.

Four conclusions, none of which survived from the pre-fix version of this document:

1. **`opd` does nothing here, and it is not a tuning failure.** Raising the lr to 1e-5 makes
   it *worse* (0.4044, declining across the run), so the flat curve is not an under-tuned
   step size. Grad norms in the RL path are 0.0005-0.023, far under the default `max_norm`,
   so clipping never binds and cannot be blamed either.
2. **The teacher's knowledge IS transferable — by `sft`, from the same teacher.** Hard
   distillation on teacher samples reaches 0.8127 (+0.225) where `opd` reaches -0.002. Same
   teacher, same band, same budget. The failure is specific to the per-token reverse-KL
   score-function estimator, not to distillation.
3. **GRPO beats the teacher it would have distilled from.** On a task with a verifiable
   reward, plain RL is simply the better instrument; there is no headroom argument left for
   distillation on this setup.
4. **The top-k collapse is real, and was not a gradient artifact.** All five formulations
   collapse to 0.000-0.031 at 87-100% truncation, plus k=100. This was the result most
   likely to have been caused by the corrupted Q/K gradients — it was not.

The MOPD ablations land within ~2 seed-sigma of `opd` (0.5688-0.6239) and reverse their
pre-fix sign: `noratio` was -0.065 before the fix and is +0.036 after. Neither direction is
resolvable at n=1; the honest statement is that they do not rescue `opd`.

**Caveats.** Two arms are partial: `grpo lr 1e-5` (43553) lost its final checkpoint to the
shutdown deadlock and is reported at step 250; `k100` (43563) hit its 4h wall at step ~135
because k=100 costs ~4.7x orchestrator time, and is reported at step 100. Neither changes a
conclusion. `sft` needed a resubmit -- `gsm8k_sft.toml` had no `[ckpt]` section, so the first
attempt wrote rollouts but no weights.

### 1c. The in-run eval bug is FIXED (2026-08-20)

Every number in this document came from offline checkpoint scoring because the
in-run eval was untrustworthy. Root-caused and fixed.

Eval clients were built as plain chat-completions, leaving the chat template to
the server; for Qwen3 that means thinking mode, regardless of
`[renderer] enable_thinking = false`. Job 43797 step 285, both streams from the
same env and config:

| stream | rollouts | containing `<think>` |
|---|---|---|
| train (renderer) | 128 | **0** |
| eval (chat-completions) | 128 | **128** |

Hence eval reading 0.2969 at 64% truncation where offline scoring of the same
checkpoint gave 0.7401 at 5.7%.

`eval.sampling.extra_body.chat_template_kwargs` does resolve into the config
correctly (verified in the run's dumped `orchestrator.toml`) but never reaches
the server: `deps/verifiers/.../dialects/chat.py::apply_overrides` splats the
sampling dump into the body, leaving `extra_body` as a literal key. The direct
`openai_chat_completions_client` unwraps it properly, so the two paths disagree.

Fix: `orchestrator.eval_via_renderer` (default false; enabled in the band
configs). Verified job 43825 -- 0/128 eval rollouts with `<think>`, truncation
1.6-3.1%, step-1 reward 0.2812 on an untrained policy.

**Consequence:** in-run eval curves are now usable, and offline checkpoint
scoring is no longer mandatory for every experiment.

### 1d. Fully on-policy changes nothing (2026-08-20)

`opd`'s `ref_kl` multiplies by an unbounded importance ratio purely to correct
off-policy staleness, so staleness was a live confound. Re-ran GRPO and `opd`
k=0 with `max_off_policy_steps = 0` (true lockstep, `Max Off-Policy 0` in every
step line), 300 steps, offline-scored on the same band:

| arm | async (`max_off_policy = 8`) | on-policy (`= 0`) |
|---|---|---|
| grpo @300 | 0.9220 +/- 0.0057 (n=3) | **0.9151** |
| opd k=0 @300 | 0.5859 +/- 0.0199 (n=3) | **0.5894** |
| student | 0.588 | 0.588 |

`opd` on-policy across steps 50/150/300: 0.5948 / 0.5856 / 0.5894 — flat, on the
student baseline, truncation 0.9-1.5%. **Removing staleness does not rescue
`opd`**, so the flatness belongs to the estimator, not to the async loop. GRPO
is unchanged within ~1.2 sd, confirming the lockstep switch is otherwise inert.

Side benefit: `max_off_policy_steps = 0` also eliminates the trainer/orchestrator
shutdown deadlock by removing its precondition (the trainer can no longer lag
when the orchestrator drains). Both arms exited cleanly; see 2.2.

---

## 2. Blocking infrastructure findings

These affect everyone running prime-rl on this cluster, not just distillation.

### 2.1 The ROCm flash-attention backward is broken ⇒ the trainer does not learn at default `max_norm` *(the big one)*

The trainer reports `Grad. Norm` values of ~1e8–1e10 for a 0.6B model where O(1–10) is correct. Default clipping at `max_norm = 1.0` therefore scales every update by a factor that swings ~100× step to step, which destroys AdamW's `m`/`v` EMAs and turns training into a random walk.

Isolated in the **standalone `sft` entrypoint** on the repo's own vetted config `examples/basic/reverse-text/sft.toml` — 1 GPU, no orchestrator, no vLLM, no rollouts, no weight broadcast — so the entire RL machinery is ruled out:

| Job | `max_norm` | lr | CE loss step 1 → 100 | Trend |
|---|---|---|---|---|
| 42356 / 42359 | 1.0 (default) | 2e-5 | 4.6125 → 4.8955 | flat (bit-identical between the two runs) |
| 42359 | 1.0 (default) | 1e-4 | 4.6125 → 12.4215 | diverges (higher lr is not a fix) |
| 42362 | 1.0, offload off | 2e-5 / 1e-4 | bit-identical to above | `optim_cpu_offload` ruled out |
| **42364** | **1e12 (off)** | 2e-5 | **4.6125 → 3.7183** | **descends** |

**ROOT-CAUSED 2026-08-18 (jobs 43539/43540/43543) — and the diagnosis above was wrong.** The reported norm is *correct*; the gradients really are that large, because of one environment variable.

**The cause: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.** It selects flash-attn's Triton AMD backend instead of composable-kernel. Job 43543, four cells, everything else identical:

| attention backend | `FLASH_ATTENTION_TRITON_AMD_ENABLE` | act. ckpt | step-1 loss | grad norm 1–3 | tok/s |
|---|---|---|---|---|---|
| flash-attn varlen | **TRUE** | on | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | 5869 |
| flash-attn varlen | **FALSE** | on | 4.6127 | **24.39 / 11.98 / 6.66** | **17550** |
| flash-attn varlen | **TRUE** | off | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | 15284 |
| MATH SDPA (reference) | — | on | 4.6122 | 24.36 / 11.98 / 6.65 | 11069 |

Unset, flash-attn agrees with an independent MATH-SDPA reference to three significant figures on the gradient norm and four on the loss, and runs **3× faster** than the Triton path. Activation checkpointing is exonerated — that cell is bit-identical to the control. The variable came from `rl_train_singlenode_mixed.sbatch:308` and was copied into all twelve `opd_val` scripts; all thirteen have been patched to a comment explaining why not to set it.

The rest of this section records the earlier, more general finding that led here — that the *backward* was corrupt while the forward was exact.

`opd_val/gradnorm_probe.py` wraps the standalone `sft` entrypoint on a single GPU (no sharding, so a manual sum over `.to_local()` grads is exact) and monkeypatches `torchtitan.distributed.utils.clip_grad_norm_` to compute the norm independently before clipping:

| backend | step-1 loss | grad norm, steps 1–3 | reported/manual | `embed_tokens` grad RMS | top-1 share |
|---|---|---|---|---|---|
| `flash_attn_varlen_func` (default) | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | **1.0000** | 1.9e4 | ~50% |
| `F.scaled_dot_product_attention`, MATH backend | 4.6122 | 24.4 / 12.0 / 6.7 | **1.0000** | 2.1e-4 | ~24% |

Same data, same seed, only the attention kernel swapped. **The loss agrees to four significant figures — the forward is correct — while the gradients differ by ~1.4e7.** The reported/manual ratio is 1.0000 in both arms, so `clip_grad_norm_` was never at fault.

The damage is localised and structured: the five largest per-parameter norms are all layer 0 plus the embedding, ordered q_proj > k_proj > v_proj > o_proj — the Q/K path — and `q_norm` (128 elements) reaches a per-element gradient RMS of **3.3e6**. Under SDPA `q_norm` leaves the top five entirely.

- **`max_norm = 1.0` is the correct default.** At a true norm of 6–24 it is an entirely sensible clip. It appeared to "prevent all learning" only because the *global* rescale by 1/3.4e8 crushes the ~305 healthy parameters to ~1e-8, where AdamW's `eps = 1e-8` floor swamps them and their updates go to zero. Unclipped, AdamW's per-parameter normalisation rescues the healthy majority while layer 0 keeps training on garbage. **`max_norm = 1e12` worked by accident and fixed nothing.**
- **Environment:** `flash-attn 2.8.3.post1`, `torch 2.11.0+rocm7.2`, `triton 3.6.0`. Only FA2 is installed. Colleagues on this cluster run the same versions without trouble — consistent with the variable, not the package, being at fault.
- **Every result in this document and in `EXPERIMENT_LOG.md` was measured with layer 0 and the embeddings training on corrupted gradients**, including GRPO 0.7256 vs `opd` 0.6715. Both arms shared the defect, so the ranking is not automatically void, but no conclusion here should be treated as a statement about the *algorithms* until it is reproduced. The seven-variant top-k collapse is the most suspect of all: top-k puts exact gradient on the Q/K path, which is where the corruption was worst.
- **The fix is free.** Removing the variable is strictly better on every axis measured: correct gradients, 3× throughput, and the repo's default `max_norm` becomes usable again. No code change, no dependency pin, no `sdpa` fallback needed.

### 2.2 ROCm: `torch.compile` must be off

`[trainer.model] compile = "None"` (the literal string parses to Python `None`). Otherwise inductor's meta kernel for `torch.ops.flash_attn._flash_attn_varlen_forward` reports a transposed shape and the trainer dies on `assert_size_stride(buf13, (16, 2047), (2047, 1))` — job 42211. The same workaround already exists in `rl_train_singlenode_mixed.sbatch`.

### 2.3 ROCm: never set `HIP_VISIBLE_DEVICES` for the `rl` entrypoint

The `rl` entrypoint separates inference from trainer by setting `CUDA_VISIBLE_DEVICES` on its children. On ROCm, a `HIP_VISIBLE_DEVICES` in the parent environment takes precedence and silently collapses both onto one GPU. Standalone single-process launches (e.g. the frozen teacher server on GPU 2) may use it safely.

### 2.4 Upstream: hardcoded `/tmp/vf-scripts` in the verifiers subprocess harness

`deps/verifiers/verifiers/v1/runtimes/base.py:173` builds `f"/tmp/vf-scripts/{digest}.py"` with no `TMPDIR` awareness. On a shared node, whoever creates the directory first owns it and every other job's rollouts die with `PermissionError [Errno 13]`. Node-dependent and racy, not deterministic: job 42227 hit it on **every** rollout for 98 minutes and was cancelled, while 42216 and 42226 (same code, no workaround) saw zero occurrences. Workaround in `rl_opd.sbatch` — bind a job-private dir over that exact path. Worth reporting to `verifiers`.

### 2.5 Trainer/orchestrator shutdown deadlock — **nine** observed occurrences

The orchestrator drains its step loop and logs `Orchestrator finished.`, the trainer stalls a few steps short of `max_steps` and never exits, and the SLURM job sits in `RUNNING` holding a whole node until an operator or the walltime kills it. Census across all `opd_val/runs/`:

| Job | Trainer last step / `max_steps` | Orchestrator finished | SLURM outcome |
|---|---|---|---|
| 42285 | 57 / 60 | yes | `scancel`'d by operator |
| 42297 | 97 / 100 | yes | CANCELLED at 01:00:59 |
| 42299 | 97 / 100 | yes | CANCELLED at 01:00:47 |
| 42367 | 97 / 100 | yes | CANCELLED at 01:03:37 |
| 42368 | 97 / 100 | yes | CANCELLED at 01:03:37 |
| 42374 | **96** / 100 | yes | TIMEOUT at 04:00:06 |
| 42380 | 97 / 100 | yes | CANCELLED at 01:04:49 |
| 42381 | 97 / 100 | yes | CANCELLED at 01:04:49 |
| 42387 | 97 / 100 | yes | TIMEOUT at 04:00:11 |

Eight of nine stall **exactly three steps short**; 42374 stalls four short. **Mechanism unconfirmed.** An earlier `zero_advantage`-filter-starvation hypothesis cannot be the whole story: five of these nine are `opd` runs, which ship no advantages at all and therefore cannot be filtered that way. Whatever it is, it is *not* GRPO-specific and it is *not* rare — it hit roughly a third of all runs and cost several node-hours. The already-written eval data is unaffected in every case.

**Operational rule:** a clean orchestrator log is not proof a run finished. Check `sacct` for a terminal state and compare the trainer's last `Step N` to `max_steps`.

### 2.6 The `uv sync` extras trap

`uv sync --package prime-rl --package math-env-v1 --package aime24-v1` (no `--all-extras`) **removes every optional-extra package**, including `flash-attn` — which broke jobs 42356 (second arm) and 42357 with `ModuleNotFoundError: No module named 'flash_attn'`. Conversely `uv sync --all-extras` alone **removes the workspace env packages**. Both flags are required together:

```
uv sync --all-extras --package prime-rl --package math-env-v1 --package aime24-v1
```

(A bare `import ring_flash_attn` failing on transformers>=5.4 is expected and harmless — `src/prime_rl/_compat.py` shims it and is imported first by every entrypoint.)

---

## 3. Two metrics that mislead

**`Trainable 0/128 (0.0%)` is structurally meaningless under `opd`/`opsd`.** `Rollout.is_trainable` (`src/prime_rl/orchestrator/types.py:131-134`) is defined purely on advantages, and reference-KL algorithms never assign advantages by design — `ref_kl_loss_fn` reads `ref_logprobs` directly. So the counter is always zero and the accompanying "consider reviewing task difficulty / filter config" warning is pure noise. GRPO runs correctly report `Trainable 128/128` by contrast. The real health signals for an `opd` run are whether batches ever come up empty and whether `Policy vN` advances.

**The huge gradient norms are not `opd`-specific.** A GRPO control (42226: 1.31e7–7.78e7) reproduces the same magnitudes as `opd` (42216: 9.27e7–4.55e8), and the standalone `sft` entrypoint reproduces them with no RL machinery at all. This is pre-existing prime-rl-on-ROCm behaviour.

---

## 4. What we learned about `opd` itself

### 4.1 The mechanism is fully validated

- Teacher prefill-scoring over `/inference/v1/generate` with `prompt_logprobs` works correctly on ROCm/vLLM 0.25.1: one correctly-aligned logprob per token, `0.0` at position 0, bit-deterministic across repeated identical calls (max drift `0.00e+00`), and discriminative — mean logprob −4.6982 for a correct reversal vs −7.3495 for a scrambled one (job 42209).
- `ref_logprobs` reach the trainer, `ref_kl` executes, RCCL weight broadcast works, runs finish `rc=0` at 128/512/1024-token budgets and multiple learning rates.
- **The teacher must be a prime-rl `uv run inference` server**, not a generic OpenAI-compatible endpoint — nothing else serves `/inference/v1/generate`, which `prefill_logprobs` requires.

### 4.2 With the trainer fixed, `opd` fails from a base-model student

Identical setup across arms: student `PrimeIntellect/Qwen3-0.6B` (base, untuned), teacher `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`, `max_norm = 1e12`, `compile = "None"`, 100 steps, 1024 max completion tokens, eval every 5 steps on 128 examples:

| Job | Arm | lr | Reward step 1 → 100 | Truncation step 1 → 100 | Verdict |
|---|---|---|---|---|---|
| 42366 | `sft` distill | 2e-5 | 0.0429 → **0.1499** (peak 0.1905 @ 80) | 71.1% → 35.2% | **learns** |
| 42375 | `sft` distill | 5e-6 | 0.0417 → **0.1287** | 69.5% → 47.7% | **learns** |
| 42368 | `grpo` | 2e-5 | 0.0498 → **0.1406** (0.0986 @ 95) | 69.5% → 98.4% | learns, unstable |
| 42367 | `opd` | 2e-5 | 0.0458 → **0.0022** (0.0051 @ 95) | 70.3% → 100.0% | **collapses** |
| 42374 | `opd` | 5e-6 | 0.0385 → **0.0167** | 77.3% → 90.6% | **collapses (slower)** |

`opd`'s decline is monotone, not noise: 0.0458 (1) → 0.0475 (25) → 0.0217 (50) → 0.0084 (70) → 0.0022 (100), with truncation climbing in lockstep to 100%. The student stops emitting a closing answer tag at all. `sft` in the same setup roughly triples reward and halves truncation.

`grpo`'s trajectory is worth a footnote: it rises fast (0.1401 @ step 15, truncation down to 2.3%), then sags to ~0.06 mid-run and ends at 0.1406 with truncation back at 98.4% — improving reward while truncating almost everything. Something reward-hacking-shaped may be happening; not investigated.

### 4.3 The task's own limitations are real and independent

- **Reward is close to a binary "did it stop in time."** At `max_completion_tokens = 128`, job 42216 step 8 (N=137 rollouts): 104 opened `<think>`, 66 closed it, only 30 emitted `</reversed_text>` — and those 30 averaged **0.7519** reward while the other 107 scored **exactly 0.0**. Reward is dominated by whether the model finishes, not by reversal quality.
- **The released SFT student has essentially no headroom.** The "0.223 vs 0.799" gap that motivated this work is a token-budget artifact: at 512 tokens the released `Qwen3-0.6B-Reverse-Text-SFT` student already scores **0.72–0.75** against the teacher's **0.799**. There is ~0.05–0.08 to close, which no algorithm will demonstrate cleanly.

So `reverse-text` offers only two starting points, and both are bad: the released SFT student (no headroom) and the base model (so far from the task that reward is a step function). Neither is the regime on-policy distillation is designed for.

---

## 5. The top-k teacher distribution work

### 5.1 Motivation

prime-rl's default `opd` requests `prompt_logprobs: 1` and uses only the teacher's logprob for the **sampled** token — the score-function estimator `(log π_ref − log π_θ)·∇log π_θ`. It is unbiased, but it has a structural coverage asymmetry: **it can only push probability *off* tokens the policy actually emitted, never *onto* a token the teacher prefers but the policy never samples.** A student far from the teacher may never sample the right tokens, so the gradient never points at them.

### 5.2 What was implemented

Working-tree changes (uncommitted, ~350 lines excluding `uv.lock`):

| File | Change |
|---|---|
| `packages/prime-rl-configs/.../algorithm.py` | `OPDAlgoConfig.teacher_top_k: int = 0` |
| `packages/prime-rl-configs/.../inference.py` | `ModelConfig.max_logprobs: int = 20`, forwarded as `--max-logprobs` |
| `src/prime_rl/transport/types.py` | new `TeacherTopK` struct (raw int32 ids + float32 logprobs + shape; ~800 KB/sample at k=100, so no `tolist()`), appended to `TrainingSample` and `MicroBatch` |
| `src/prime_rl/utils/client.py` | `prefill_logprobs_topk` / `PrefillScorer.score_topk` |
| `src/prime_rl/orchestrator/algo/opd.py` | ships the support when `top_k >= 1`; still ships sampled-token logprobs so metrics stay comparable |
| `src/prime_rl/trainer/{batch,rl/data,rl/packer,rl/train}.py` | slicing/packing the 2-D per-token payload; `TopK Mass` in the step line |
| `src/prime_rl/trainer/rl/loss.py` | the KL over the support |

Requires `trainer.model.fused_lm_head_token_chunk_size = "disabled"` (the fused head never materializes logits, so the policy's probabilities on the teacher's support cannot be gathered), and `model.max_logprobs` raised on the teacher for k > 20.

### 5.3 The design flaw the first sweep exposed

The first implementation minimized the partial sum `Σ_{v ∈ topk} π(v)(log π(v) − log π_ref(v))`. **A partial sum is not a KL.** It has a degenerate minimum: driving `π(v) → 0` for every `v` in the support sends every term to zero, so the policy can score perfectly by evacuating the teacher's support entirely and dumping its mass on the untracked rest of the vocabulary. That is exactly what happened:

| Job | k | TopK Mass 1 → 25 → 100 | Entropy 1 → 25 (max) | Eval reward |
|---|---|---|---|---|
| 42381 | 8 | 0.2458 → 0.0116 → — (min 0.0089) | 0.5936 → 8.2782 (8.3302) | 0.0671 → **0.0000 from step 20 on** |
| 42382 | 20 | 0.3526 → 0.0077 → 0.0790 (min 0.0062) | 0.5894 → 8.1318 (8.5235) | 0.0464 → **0.0000 from step 15 on** |

TopK Mass collapsing from 0.35 to <0.01 while entropy runs to 8.1+ (ln(151k) ≈ 11.9, so this is a near-uniform distribution) is the signature of the policy escaping the objective rather than optimizing it. Independent confirmation from the production logs: the pre-fix `Ref KL` metric goes **negative** (42381 min −0.0758, 42382 min −0.0841), which is impossible for a true KL, while every post-fix value is positive.

Jobs 42383 (k=50) and 42384 (k=100) never got that far — they died in 3 minutes with `BadRequestError: Requested prompt logprobs of 50, which is greater than max allowed: 20`, which is what motivated the `max_logprobs` config addition.

### 5.4 The fix

Coarsen all off-support mass into a **single atom** for both distributions (`src/prime_rl/trainer/rl/loss.py`):

```
policy_out = (1 - Σ_topk π(v)).clamp_min(1e-6)
ref_out    = (1 - Σ_topk π_ref(v)).clamp_min(1e-6)
per_token_kl = Σ_topk π(v)(log π(v) − log π_ref(v)) + policy_out·(log policy_out − log ref_out)
```

This makes the objective a genuine KL between two (k+1)-category distributions: non-negative, and by the data-processing inequality a lower bound on the true vocabulary-wide KL. Because the teacher is concentrated, its off-support mass is tiny, so mass the policy puts out there is correctly expensive. The escape route is closed.

### 5.5 Post-fix results — the fix works, `opd` still fails

All runs: base-model student, `max_norm = 1e12`, 100 steps.

| Job | k | lr | TopK Mass 1 → 100 | Ref KL 1 → 100 | Entropy 1 → 100 | Eval reward |
|---|---|---|---|---|---|---|
| 42380 | 0 (control) | 2e-5 | n/a | −0.837 → −0.494 | 0.580 → 1.17 | 0.0411 → 0.0322, trunc 100% |
| 42387 | 8 | 2e-5 | 0.2501 → 0.68 (@75) | 6.9163 → 0.7906 | 0.6126 → 2.48 | 0.0318 → **0.0000 from step 35** |
| 42388 | 20 | 2e-5 | 0.3470 → **0.6301** | 7.8106 → **0.5358** | 0.5975 → 3.4369 | 0.0427 → **0.0000 from step 35** |
| 42389 | 50 | 2e-5 | 0.4470 → **0.5880** (peak 0.8434) | 8.7924 → **0.7221** | 0.5705 → 4.5233 | 0.0483 → **0.0000 from step 25** |
| 42390 | 100 | 2e-5 | 0.5379 → **0.7526** (peak 0.8495) | 9.1573 → **0.8769** | 0.5726 → 4.2743 | 0.0483 → **0.0000 from step 30** |
| 42391 | 20 | 5e-6 | 0.3561 → 0.4839 | 7.8016 → 1.1098 | 0.6062 → 4.0794 | 0.0419 → **0.0000 from step 75** |

**The fix demonstrably works.** TopK Mass now *rises* instead of collapsing (0.5379 → 0.7526 at k=100; the pre-fix runs went to 0.006–0.012), Ref KL falls by an order of magnitude (9.16 → 0.88 at k=100), and entropy stays in the 2–5 range instead of running to 8.5. The student really is moving onto the teacher's support, and larger k gives better coverage as expected.

**And it changes nothing about the outcome.** Every arm reaches eval reward exactly 0.0000 with 100% truncation, only faster than the k=0 control. Lowering the learning rate to 5e-6 delays collapse to step 75 but does not prevent it.

**Be explicit about this: the coverage asymmetry was a real defect, correctly diagnosed and correctly fixed, and it was NOT the reason `opd` failed.** The student is now provably matching the teacher's distribution better than before while producing strictly worse text.

> **VOID (2026-08-24).** Every row in this table, and the conclusion drawn from
> it, is invalidated by finding 5.6. The last sentence above was the tell and we
> read it backwards: a student that matches a target better while writing worse
> text is matching the *wrong target*.

---

### 5.6 The teacher's top-k support was off by one token (2026-08-24)

**All k ≥ 1 results in this document are void.** The support was aligned to the
wrong position, so every top-k run minimized a scrambled objective.

vLLM's `prompt_logprobs[t]` is the distribution predicting token `t`, and it puts
the target id first, so the payload obeys

```
teacher_topk_ids[t][0] == input_ids[t]
```

— the "probability of current token" convention, the same one `ref_logprobs`
uses. The policy's logits are on the *other* convention: `logits[t]` predicts
token `t+1`. That is exactly what `labels = shift_tensor_left(input_ids)`
(`train.py:383`) exists to correct. The support got no such correction, so
`gather_log_softmax(scaled_logits, teacher_topk_ids)` evaluated the policy at
position `t`'s candidate ids against logits predicting token `t+1`. After the
shared `shift_tensor_right` the *positions* line up but the *vocab indices* do
not: slot `t` held the policy's mass on position `t−1`'s candidates, compared
against the teacher's probabilities for position `t`'s.

**Proof, no model required** (`opd_val/topk_alignment_test.py`). Column 0 of the
support *is* the sampled token, so after the shift it must equal the scalar
`trainer_logprobs` exactly:

```
max |topk[:,0] - scalar logprob|   as shipped:  2.851221
max |topk[:,0] - scalar logprob|   ids shifted: 0.000000
```

**Why the metrics looked healthy.** Adjacent positions' top-k sets overlap
heavily in natural text (punctuation, function words, digits), so the scrambled
support still carried 0.35–0.75 of the policy's mass, and the KL over it was
still minimizable. `TopK Mass` rising and `Ref KL` falling by an order of
magnitude were real — of the wrong quantity.

**Why it destroyed the model.** Placing probability mass on the *previous*
position's candidate set is, mechanically, training the model to lag and repeat.
Every top-k arm ended at **100% truncation and reward exactly 0.0000**, and the
lr sweep showed the damage scaling with how hard the KL was minimized (reward
0.605 → 0.541 → 0.024 as KL reduction went 0% → 27% → 95%). Optimizing this
objective harder is optimizing degeneracy harder.

**Why the earlier direction test missed it.** `opd_val/topk_direction_test.py`
feeds tensors straight into `_topk_divergence`, bypassing the plumbing. The loss
function was never the problem — every one of the 9 `topk_normalization ×
kl_type` combinations is correctly directed, on the logits, both before and after
this fix. The bug was one layer up, in `train.py`.

Fixed by shifting the support onto the labels' convention at the same site the
labels are built, before CP sharding. `shift_tensor_left` was rank-2 only and was
extended to rank-3 to allow it (its sibling `shift_tensor_right` already had
been).

**Also fixed here (latent, inert at defaults):** `_topk_divergence` divided
`trainer_topk / T`, but the payload holds **log-probabilities**, not logits.
`renormalize` survives this — a logprob differs from its logit by a per-position
constant that `log_softmax` cancels — but `none` and `residual` read
`exp(logprob / T)` as an absolute probability, which it is not, so the
off-support residual `1 − p.sum()` can go negative. Every run used `T = 1.0`, so
nothing measured was affected. Those two modes now require `T = 1.0` and say so.

---

## 6. Current hypothesis and what is running

### Hypothesis (not yet tested)

> **SUPERSEDED (2026-08-24).** This section was written to explain why `opd`
> failed at every k. Two implementation bugs have since been found that account
> for the failure directly: the k=0 objective contributed **zero gradient**
> (§1e / commit `27c0493b`), and the k ≥ 1 support was **off by one token**
> (§5.6). With the k=0 bug fixed, `opd` reaches 0.8203 on the GSM8K band from a
> 0.588 student. The reasoning below is retained because it is not *wrong* as
> theory, but it is no longer needed to explain anything we measured, and it
> should not be cited as a finding.

`opd` is on-policy: the KL is evaluated on the *student's own* trajectories. When the student starts far from the teacher, those trajectories are garbage, and the objective is satisfied by matching the teacher on meaningless contexts. Nothing anchors the student to good behaviour the way `sft`'s teacher-generated tokens do. The post-fix metrics are exactly what that predicts: the KL genuinely drops (0.88 at k=100, from 9.16) while the reward goes to zero — the student is learning to imitate the teacher *on states no competent model would ever visit*. Additionally, reverse KL is mode-seeking and unstable far from the target, whereas forward KL / cross-entropy is mode-covering and stable; that ordering matches the observed `sft` > `grpo` > `opd` result exactly.

This is also consistent with the standard recipe upstream: **SFT warmup first, then RL.** `opd` may simply not be an algorithm you can start from a base model.

The untested middle regime is a student **competent at the task but meaningfully worse than the teacher** — everything tried so far was either far too weak (base, reward 0.04) or already at teacher level (released SFT, 0.72 vs 0.799).

### In flight

**Job 42459 (`sft-ladder`).** `opd_val/sft_ladder.sbatch`: standalone `sft` from `PrimeIntellect/Qwen3-0.6B` on `examples/basic/reverse-text/sft.toml`, 60 steps, `--ckpt.interval 15`, `--optim.max-norm 1e12`, `--model.compile None`, 1 GPU. Produces HF-format students at steps 15/30/45/60 under `opd_val/runs/sft-ladder_42459/sft/weights/step_<n>/`, directly servable by vLLM. The plan is to pick whichever checkpoint lands in the **0.3–0.6 reward band** and re-run `opd` from there.

---

## 7. Recommendations, in order

1. **Report the `max_norm` / gradient-norm bug upstream and to whoever owns the ROCm port — highest priority, and not a distillation issue.** It blocks *all* prime-rl training on this cluster with default settings, silently, with no error. Anyone who runs prime-rl here today and leaves `[trainer.optim] max_norm` alone gets a flat loss curve and no indication why. Include the 42364 vs 42359 A/B (identical config, one flag, 4.6125→3.7183 vs 4.6125→4.8955) — it is a two-line reproduction. *Confidence: very high, isolated on the standalone entrypoint with a bit-identical determinism check.*
2. ~~**Root-cause the norm itself with a per-parameter gradient-norm dump.**~~ **DONE (2026-08-18, jobs 43539/43540) — see §2.1.** The norm was correct; the ROCm flash-attn *backward* is broken (gradients 1.4e7× too large, forward exact to 4 s.f.). The report in item 1 should be redirected: it is not a `clip_grad_norm_` bug, and `max_norm = 1.0` is the right default. Two fix paths, neither yet attempted: (a) an environment fix — a flash-attn build whose ROCm backward is correct; (b) a prime-rl change — add `sdpa` to `AttnImplementation` and `ATTN_IMPL2CLASS` as a correctness fallback, since today the config offers no working backend on this hardware. *Confidence: very high — single GPU, no sharding, identical loss with only the kernel swapped.*
3. **Run the intermediate-checkpoint experiment (42459's ladder) — this is the real test of `opd` viability.** Everything else about `opd` is now known to work mechanically; the only open question is whether it works from a competent-but-inferior student. If `opd` also fails from a 0.3–0.6-reward student, the conclusion is that `opd` as implemented does not work here, and further estimator tuning is not worth doing. *Confidence in the hypothesis: moderate — it is consistent with all the evidence and with the standard SFT-then-RL recipe, but it is a hypothesis.*
4. **Fix or report the `/tmp/vf-scripts` path** (`deps/verifiers/.../base.py:173`) — a one-line `TMPDIR` fix upstream removes a class of node-dependent flakiness that already cost one 98-minute wasted job. *Confidence: high, root cause is certain.*
5. **Report the shutdown deadlock.** Nine occurrences, ~a third of all runs, each holding a full node until walltime or an operator intervenes. The `zero_advantage` hypothesis does not survive the wider census (five of the nine are `opd`, which ships no advantages), so this needs fresh investigation of the orchestrator's batch-dispatch path. *Confidence in the symptom: certain. In any mechanism: none.*
6. **Consider abandoning `reverse-text` for distillation validation.** Its reward is close to a binary "did it stop in time", it offers no student in the useful headroom band, and the released student is already at teacher level. A task with graded reward and a genuine capability gap would make any of these results far more interpretable. *Confidence: high — this is a measurement (42224, 42216 step-8 traces), not an inference.*
7. **Keep the top-k work — but every k ≥ 1 number in this document must be re-measured.** The loss function is correct (all 9 `topk_normalization × kl_type` combinations are correctly directed, verified on the logits) and the coarsened-residual KL is well-founded. The *plumbing* was not: the support was off by one token until 2026-08-24 (§5.6), so the whole sweep optimized a scrambled objective. Re-run k ∈ {8, 20, 50, 100} before drawing any conclusion about whether coverage helps. The metrics the work added (`TopK Mass`, `Ref KL`, `topk_residual`) are what made both diagnoses possible and should stay.
8. **Treat "the objective improves while the output gets worse" as an alignment alarm, not a finding.** It was reported twice in this document as evidence about `opd`'s nature. Both times it was a bug: the metric and the text disagreed because the metric was measuring the wrong quantity. A KL that falls an order of magnitude while reward goes to exactly 0.0000 and truncation to 100% should trigger a plumbing audit first.

---

## 8. Reproducing

Everything lives under `opd_val/`.

- **Configs:** `make_configs.py` generates the families; `{sft,opd,grpo}_base_fixed.toml` (post-`max_norm`-fix base-student comparison), `opd_k{0,8,20,50,100}.toml` and `opd_k20_lr5e6.toml` (top-k sweep), `{sft,opd}_base_lr5e6.toml`, `{sft,opd,grpo}_512.toml`.
- **Launchers:** `rl_opd.sbatch` (main RL runs — GPU 0 policy server :8000, GPU 1 trainer, GPU 2 frozen teacher :8001, launched by the script itself since prime-rl never starts reference models), `sft_entrypoint.sbatch`, `sft_ladder.sbatch`, `smoke_prefill.sbatch`, `compare_teacher_student.sbatch`.
- **Analysis:** `compare.py` (behavioural + per-token `ref_kl` measurement).

**Required settings on this cluster:**

| Setting | Value | Why |
|---|---|---|
| `[trainer.optim] max_norm` | `1e12` | §2.1 — otherwise nothing learns |
| `[trainer.model] compile` | `"None"` | §2.2 — ROCm flash-attn varlen meta kernel |
| `[trainer.model] fused_lm_head_token_chunk_size` | `"disabled"` | top-k only; fused head never materializes logits |
| teacher `--model.max-logprobs` | `>= k` (default 128 in `rl_opd.sbatch`) | k > 20 gets a 400 otherwise |
| `HIP_VISIBLE_DEVICES` | **unset** for the `rl` entrypoint | §2.3 |
| `[orchestrator.train.sampling] temperature` | `1.0` | the trainer temperature-scales the student's logprobs while the teacher is scored at T=1 |

Environment: container `/shared_silo/scratch/containers/primus_v26.2_moefix.sif` via `singularity exec --rocm --cleanenv --containall`; per-job scratch paths baked in **host-side** from `$SLURM_JOB_ID` (the container strips `SLURM_*`); `UV_PYTHON=<repo>/.venv/bin/python3`, `UV_NO_SYNC=1`; `HF_HOME=/shared_silo/scratch/hf_cache` with `HF_HUB_OFFLINE=1`; `WANDB_MODE=disabled` (no key on this cluster — all numbers here come from text logs). Validate any config on the login node with `uv run rl @ cfg.toml --dry-run` before submitting.

**Reading logs:** they contain ANSI codes — strip with `sed -e 's/\x1b\[[0-9;]*m//g'`. Trainer tracebacks are in `run/logs/trainer/torchrun/<id>/attempt_0/0/stdout.log`, **not** `trainer.log`.

**How to check a finished run** — all four matter:

1. `sacct -j <id> --format=JobID,State,Elapsed,ExitCode` — must be terminal, not `RUNNING`.
2. Trainer's last `Step N` vs `max_steps` — a mismatch is the §2.5 deadlock tell.
3. `grep -c HarnessError run/logs/orchestrator.log` — nonzero means the §2.4 harness bug.
4. Log mtimes vs now — stale logs plus `RUNNING` is a live stall.
