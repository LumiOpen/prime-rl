# OPD Validation Experiment Log — reverse-text + GSM8K, AMD/ROCm cluster

**Date started:** 2026-08-12
**Repo:** `/shared_silo/scratch/rahul.aralikatte@amd.com/prime-rl` (branch `upstream`)
**Working dir:** `opd_val/`

> **MAINTENANCE NOTE — READ BEFORE EDITING**
> This is the canonical record of the OPD validation effort. **Every time a new experiment
> finishes, this file MUST be updated**: add a row to the Results table and an entry to the
> Changelog at the bottom. Do not let results live only in Slack/memory — if it's not in this
> file, it didn't happen. When you update a metric, verify it against the actual run logs under
> `opd_val/runs/<name>_<jobid>/` rather than copying a remembered number — several numbers in the
> original verbal summary this log was built from did not match the logs exactly (see the
> "Verification discrepancies" callouts throughout).

## Objective

Validate whether prime-rl's `algo.type = "opd"` (on-policy distillation: the policy samples its
own rollouts, a frozen teacher scores those same tokens via per-token reverse KL — the `ref_kl`
loss component — and the student is trained to move toward the teacher's distribution on-policy)
is viable on this cluster, starting with the bundled `reverse-text` environment.

- **Student / policy:** `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`
- **Frozen teacher:** `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`
- **Hardware:** AMD MI325X nodes, SLURM partition `amd-tw-verification`, ROCm 7.2 / vLLM 0.25.1 /
  torch 2.11.0+rocm7.2.

**Phase 3 addendum (2026-08-13):** `reverse-text` was abandoned as the validation task (Open
Question 5/Finding 15 — no usable near-teacher-level intermediate student existed for it) in favor
of **GSM8K**, with a new student/teacher pair: student `Qwen/Qwen3-1.7B`, teacher `Qwen/Qwen3-8B`
(distinct from the reverse-text pair above). Problems are filtered to a `[0.2, 0.8]` student
pass-rate "band" (327 problems) via `opd_val/gsm8k_band.py` so that improvement is measurable
without the student being trivially right or trivially wrong. See Findings 16-18 and the new
"GSM8K (Phase 3)" Results table below; the reverse-text material above (Findings 1-15) is kept as
historical record and is not superseded by this addendum.

## Current status / TL;DR

- **HEADLINE (2026-08-13): on-policy distillation WORKS, on GSM8K, once three things are right**
  (correct RL learning rate, offline checkpoint eval instead of broken in-run eval, and a
  correctly-configured multi-GPU inference server). Measured offline (job 42598) on the full
  327-problem band, ~1300 rollouts/point: baseline student 0.5879, teacher ceiling 0.8933
  (gap 0.3054). All four lr-corrected arms (opd/grpo x {1e-6, 3e-6}) improve **monotonically**,
  closing 11-26% of the gap by their last surviving checkpoint. The between-arm differences are
  **not** statistically distinguishable (see Finding 16 and the new Results table) — this
  headline is "OPD (and GRPO) can move a near-teacher-level student toward the teacher," not
  "OPD beats GRPO." See Findings 16 (root cause of the earlier GSM8K collapse), 17 (in-run eval is
  unusable for any config that changes generation behavior), and 18 (a validator-ordering bug
  causes NCCL hangs on multi-GPU inference — this also retroactively explains job 42360 below,
  corrected in its Results-table row).
- **HEADLINE (2026-08-12 evening): prime-rl does not learn on this ROCm cluster with the default
  `[trainer.optim] max_norm = 1.0`.** Every opd/grpo/sft result recorded before this finding —
  including every "flat curve" attributed below to lack of task headroom — is invalidated by this
  bug. The flat curves were caused by broken gradient clipping, not by anything about the
  algorithms under test. See Finding 12 for the full isolation (jobs 42356/42359/42362/42364) and
  the mechanism. **Every conclusion in this log dated before 2026-08-12 evening should be read
  through this lens** — several are likely artifacts of the same root cause and are flagged inline
  below where applicable.
- **OPD mechanically works on this stack**: teacher prefill-scoring, `ref_logprobs` reaching the
  trainer, the `ref_kl` loss term, and RCCL weight broadcast all function correctly on ROCm/vLLM
  0.25.1, and training runs complete with `rc=0`.
- **`reverse-text` is a poor task for validating OPD with the SFT-tuned student/teacher pair —
  but this was measured under the broken-clipping regime (see headline above), so it has two
  overlapping explanations and should be re-run if the conclusion matters.** At
  `max_completion_tokens = 128` the student looks dramatically worse than the teacher
  (reward 0.223 vs 0.799), but that gap is almost entirely a **conciseness artifact**: at
  `max_completion_tokens = 512` the student already scores **~0.72–0.75** against the teacher's
  **0.799** — a small gap with little headroom for any distillation algorithm to close. All three
  60-step 512-token arms (SFT-distill, OPD, GRPO) are flat for the same reason: the student is
  already near ceiling once it's allowed to finish its answer. **However**, all three of those
  arms also ran with the default (broken) `max_norm = 1.0`, exactly like every other pre-headline
  run — so "no headroom" and "clipping prevents learning" are both live explanations for the flat
  curves, and cannot currently be distinguished from these runs alone.
- **Two ROCm-specific footguns are now understood and worked around**: `torch.compile` must be
  disabled (`compile = "None"`), and `HIP_VISIBLE_DEVICES` must never be set for the combined `rl`
  entrypoint (it silently overrides `CUDA_VISIBLE_DEVICES` and collapses inference + trainer onto
  one GPU).
- **Gradient norm mystery partially explained, not solved.** Norms of ~1e7–1e9 (reproduced by a
  GRPO control, so not OPD-specific) are now known to be *wrong* — Finding 12's `max_norm = 1e12`
  ablation shows learning proceeds normally once clipping is disabled, and `clip_grad_norm_`
  keeps reporting the same absurd magnitudes (3.4e8–4.2e10) even with clipping effectively off, so
  the gradients themselves are fine and the *reported norm* is the bug. **Root cause of why
  `clip_grad_norm_` (from `torchtitan.distributed.utils`, called at
  `src/prime_rl/trainer/sft/train.py:471` and `src/prime_rl/trainer/rl/train.py:558`) computes
  that magnitude is still unidentified** — suspect the FSDP/DTensor norm reduction on ROCm. Raising
  `max_norm` is a workaround, not a fix. See Finding 12 and Open Question 1.
- **Job 42285 anomaly is RESOLVED (confirmed by the coordinator and by this log's original
  detection).** Orchestrator logged `Orchestrator finished.` at 12:50:24 having completed 60/60
  steps; trainer stalled at step 57/60; all three logs silent from 12:50:38 onward while SLURM
  still reported `RUNNING` at 57 minutes elapsed. The coordinator ran `scancel 42285` to free the
  node. This is now understood as a **trainer/orchestrator shutdown deadlock**, not a crash — see
  Finding 11. The run's eval data is complete and valid despite the hang (recovered from logs
  written before the stall) and is included in the Results table below.
- **Jobs 42297/42298/42299 (base-model student, opd/sft/grpo, lr 3e-6) are now understood to be
  invalidated by the `max_norm` bug** — all three used the default `max_norm = 1.0`. They
  completed 100/100 steps but every arm sat flat at reward ~0.04–0.06 (opd: 0.0470→0.0561, sft:
  0.0452→0.0372, grpo: 0.0379→0.0535) despite the base (untuned) student having enormous real
  headroom against the teacher's 0.799 — exactly the "no learning at all" signature the headline
  finding predicts. Superseded by the `max_norm`-fixed re-runs 42366/42367/42368 below.
- **Three new fixed-config runs in flight** (42366 sftfix, 42367 opdfix, 42368 grpofix) with student
  `PrimeIntellect/Qwen3-0.6B` (base model, not an SFT/RL-tuned checkpoint) against the same
  teacher — chosen specifically to give real headroom per Open Question 3. Not yet touched by this
  agent; results pending, placeholder rows added below.
- **New capability: `teacher_top_k` reverse KL over the reference's support (`OPDAlgoConfig.
  teacher_top_k`, default 0).** The teacher can now ship its per-token top-k support (ids +
  logprobs) instead of just the sampled token's logprob, so the loss can put gradient on tokens the
  teacher prefers even if the policy never sampled them — this is the estimator change Open
  Question 2 called for. Verified in code: `packages/prime-rl-configs/src/prime_rl/configs/
  algorithm.py:253`, `src/prime_rl/utils/client.py` (`prefill_logprobs_topk`, `MISSING_LOGPROB`),
  `src/prime_rl/trainer/rl/loss.py:277-316` (`ref_kl_loss_fn`'s top-k branch).
- **The first implementation had a genuine bug, now fixed (Finding 14).** A partial sum over a
  fixed support is not a KL divergence — it has a degenerate minimum at evacuating probability mass
  *off* the support entirely, the opposite of what a KL should penalize. **Directly confirmed in
  the pre-fix logs**: the top-k `Ref KL` step-line metric goes **negative** (42381 k=8: down to
  -0.0758; 42382 k=20: down to -0.0841) — impossible for a true KL. Fixed by coarsening everything
  outside the support into one extra bucket for both distributions (a genuine, provably
  non-negative (k+1)-category KL); confirmed fixed in the logs — every post-fix top-k `Ref KL`
  value across 42387-42391 (400 combined trainer steps) is positive.
- **Post-fix sweep (Finding 15): the fix works mechanically, but top-k does NOT rescue OPD from a
  base-model student.** `Ref KL` descends and `TopK Mass` rises as designed for every k tested
  (8/20/50/100) — but eval reward still collapses to **0.0000** and eval truncation still saturates
  at **100%** for every arm tried, pre-fix and post-fix, at every k (0, 8, 20, 50, 100) and both
  learning rates (2e-5, 5e-6). Leading interpretation: on-policy reverse KL is mode-seeking and
  unstable when the starting policy is far from the target distribution (a base, untrained student
  vs. a fully RL-tuned teacher) — consistent with SFT-warmup-then-RL being the correct recipe
  order, not a top-k-support limitation. See Open Question 5.

## Results table

| Job ID | Arm | Steps | LR | Max compl. tokens | Group size | SLURM state (Elapsed) | Outcome / key metric |
|---|---|---|---|---|---|---|---|
| 42206 | container triage smoke | – | – | – | – | FAILED (00:00:01) | `singularity --cleanenv --containall` strips `SLURM_*`; `/tmp/xdg_<host>_` (empty job id) cache dir already owned by another user → EACCES on all 3 candidate containers. |
| 42207 | container triage smoke (job id baked in host-side) | – | – | – | – | FAILED (00:02:26), but root-caused | All 3 candidate containers (`primus_v26.2_moefix.sif`, `vllm-dev_preview_releases_v0.20.0_20260422.sif`, `primus_v26.5-pytorch2.12-te2.15.sif`) PASSED the GPU+vLLM check (torch 2.11.0+rocm7.2, vLLM 0.25.1, `token_in_token_out` import OK). `primus_v26.2_moefix.sif` selected (first in list). Prefill-scoring step then failed with `TypeError: Object of type BatchEncoding is not JSON serializable` — a bug in the smoke-test script itself (transformers 5.x `apply_chat_template` needs `return_dict=False`), not in prime-rl. |
| 42209 | prefill smoke (teacher scoring path) | – | – | – | – | **COMPLETED** (00:01:27) | `prefill_logprobs` validated against a live teacher: 33 tokens → 33 logprobs, `lp[0]==0.0`, mean logprob correct-reversal=−4.6982 vs scrambled=−7.3495, max drift across two identical calls = `0.00e+00`. |
| 42211 | OPD reverse-text, first full run | 20 | 3e-6 | 128 | 16 | FAILED (00:03:31) | ROCm `torch.compile` (`[model.compile]` left at default, `fullgraph=false`) crashes trainer: inductor meta-kernel bug in `flash_attn._flash_attn_varlen_forward`. |
| 42216 | OPD reverse-text (compile off) | 20 | 3e-6 | 128 | 16 | **COMPLETED rc=0** (00:05:54) | Eval reward noisy, endpoints 0.1499 (step 1) → 0.1082 (step 21, final); policy advanced v0→v18; 21 evals logged. See discrepancy notes below for grad norm / truncation. |
| 42224 | teacher/student behavioural comparison (`compare.py`) | – | – | 128 (`max_tokens`) | – | **COMPLETED** (00:01:51) | Student reward 0.223, closing-tag 18/64, truncated 46/64; teacher reward 0.799, closing-tag 64/64, truncated 2/64. Per-token `ref_kl` on the student's own trajectories (960 tokens): mean −0.1960, median +0.0001, stdev 0.7905, mean |ref_kl| 0.3718, |ref_kl|>1.0 in 119/960 (12.4%). |
| 42226 | GRPO control | 20 | 3e-6 | 128 | 16 | **COMPLETED** (00:04:34) | Grad norm 1.31e7–7.78e7. Eval reward 0.1447 (step 1) → 0.1082 (step 20, last eval of the nominal 20-step loop) → **0.0689 (step 21, the true final/post-loop eval)**. Proves the grad-norm blow-up and flat curve seen in OPD are not OPD-specific. |
| 42227 | OPD, 60 steps, lr 1e-5 | 60 | 1e-5 | 128 | 16 | **CANCELLED** by operator (01:38:19) | Every rollout failed with `HarnessError: harness setup: PermissionError [Errno 13] ... /tmp/vf-scripts/<digest>.py....tmp`. Cancelled rather than let it churn indefinitely. |
| 42276 | OPD reverse-text, 60 steps, lr 1e-5 (`vf-scripts` fix applied) | 60 | 1e-5 | 128 | 16 | **COMPLETED** (00:09:34) | Eval reward noisy, no trend, range **0.0611–0.1713** across 61 evals (endpoints 0.1036→0.0946). Trainer loss ROSE 0.0288→0.0436 exactly; entropy ROSE 0.6541→0.9511. |
| 42277 | GRPO, 60 steps, lr 1e-5 | 60 | 1e-5 | 128 | 16 | **COMPLETED** (00:08:36) | Eval reward DEGRADED, 0.1312 (step 1) → 0.0614 (step 61, final), with a low of **0.0162 at step 59** (deeper than the step-40 dip of 0.0272). Entropy ROSE 0.6837→1.0990. |
| 42283 | SFT distill, 60 steps, lr 3e-6, **512 tokens** | 60 | 3e-6 | 512 | 4 | **COMPLETED** (00:08:44) | Eval reward essentially flat: 0.7472 (step 1) → 0.7449 (step 60). Eval-level truncation 3.1–10.2%; train-batch-level truncation spans wider, 0.0–21.9% (one outlier step). |
| 42284 | OPD, 60 steps, lr 3e-6, **512 tokens** | 60 | 3e-6 | 512 | 16 | **COMPLETED** (00:11:34) | Eval reward flat: 0.7158 (step 1) → 0.7131 (step 60). Eval-level truncation 3.1–10.9%. |
| 42285 | GRPO control, 60 steps, lr 3e-6, **512 tokens** | 60 (trainer stalled at 57/60) | 3e-6 | 512 | 16 | **RESOLVED — hung, then `scancel`'d** | Orchestrator completed 60/60 and exited cleanly at 12:50:24; trainer stalled at step 57/60; SLURM job stayed `RUNNING` 35+ min past orchestrator exit and was cancelled by the coordinator to free the node. Recovered eval curve (reward/truncation): step 1 0.7223/7.8%, 5 0.7206/6.2%, 10 0.7203/8.6%, 15 0.7523/2.3%, 20 0.7311/6.2%, 25 0.7469/3.1%, 30 0.7430/5.5%, 35 0.7567/5.5%, 40 0.7194/6.2%, 45 0.7299/6.2%, 50 0.7318/8.6%, 55 0.7358/4.7%, 60 0.7459/5.5% (range 0.7194–0.7567). Confirms the shutdown deadlock (Finding 11) is a teardown bug, not a data-quality issue — all three 512-token arms (sft 0.747→0.745, opd 0.716→0.713, grpo 0.722→0.746) are now confirmed flat. |
| 42297 | OPD, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1.0` (default) | 100 | 3e-6 | 1024 | 16 | **COMPLETED, invalidated** | Flat despite huge real headroom (base model): reward 0.0470 (step 1) → 0.0561 (step 100). Confirmed victim of the `max_norm` bug (Finding 12) — superseded by 42367. |
| 42298 | SFT distill, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1.0` (default) | 100 | 3e-6 | 1024 | 4 | **COMPLETED, invalidated** | Flat: reward 0.0452 (step 1) → 0.0372 (step 100). Confirmed victim of the `max_norm` bug — superseded by 42366. |
| 42299 | GRPO, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1.0` (default) | 100 | 3e-6 | 1024 | 16 | **COMPLETED, invalidated** | Flat: reward 0.0379 (step 1) → 0.0535 (step 100). Confirmed victim of the `max_norm` bug — superseded by 42368. |
| 42356 | sft-isolation, standalone `sft` entrypoint, `examples/basic/reverse-text/sft.toml`, 1 GPU, `optim_cpu_offload = true` (default), `max_norm = 1.0` (default) | 100 | 2e-5 (arm 1), 1e-4 (arm 2, crashed) | – (SFT, no rollouts) | – | lr 2e-5 **COMPLETED**, flat: CE loss 4.6125 (step 1) → 4.8955 (step 100). lr 1e-4 arm **FAILED before training** | First isolation of the pathology outside the orchestrator/vLLM path. `ModuleNotFoundError: No module named 'flash_attn'` on the second arm — an environment incident (see Finding 13), not a training result. lr-2e-5 curve later confirmed bit-identical to 42359's lr-2e-5 curve, validating this run's data despite the mid-run venv breakage. |
| 42357 | hendrycks-sanity, standalone smoke test | – | – | – | – | **FAILED** (env incident) | `ModuleNotFoundError: No module named 'flash_attn'` — same venv breakage as 42356's second arm. Not a training result. |
| 42359 | sft-isolation-v2, standalone `sft`, same config as 42356, both lr arms complete, `max_norm = 1.0` (default) | 100 | 2e-5, 1e-4 | – | – | **COMPLETED both arms** | lr 2e-5: flat, CE loss 4.6125 → 4.8955 (bit-identical to 42356's surviving arm). lr 1e-4: **DIVERGES**, 4.6125 (step 1) → 10.6983 (20) → 13.5423 (40) → 12.7642 (60) → 10.6358 (80) → 12.4215 (100). Confirms the pathology is deterministic and reproducible, and that higher lr does not fix it — it just diverges instead of merely flattening. |
| 42360 | hendrycks-sanity v2 | – | – | – | – | **FAILED — ROOT-CAUSED (Finding 18)** | `httpcore.ReadTimeout` on `POST http://localhost:8000/resume` (`src/prime_rl/utils/client.py:394`, `_resume_engines`) — orchestrator times out waiting for a 1.5B-model, 4-way-DP inference server to come up. **Originally recorded as "unresolved, separate issue"; now root-caused by Finding 18** (a config-validator-ordering bug in `packages/prime-rl-configs/src/prime_rl/configs/rl.py`): this run's own resolved config confirms it — `inference.toml` shows `dp = 4` but the saved `trainer.toml`/`orchestrator.toml` both show `inference_world_size = 1`, i.e. the NCCL broadcast group was sized for 1 rank against a 4-rank inference server, so `/resume`'s collective never completes. Not a training result; not related to the `max_norm` bug. |
| 42362 | sft-nooffload, standalone `sft`, `fsdp_cpu_offload = false`, `optim_cpu_offload = false`, `max_norm = 1.0` (default) | 100 | 2e-5, 1e-4 | – | – | **COMPLETED both arms, RULED OUT** | Loss curves bit-identical (verified `diff`, zero differences) to 42356/42359's offload=true curves at both lr arms. `CPUOffloadOptimizer` wrapper log line confirmed absent (flag took effect) yet the pathology persists unchanged — CPU offload is not the cause. |
| 42364 | **sft-noclip**, standalone `sft`, `max_norm = 1e12` (effectively off), otherwise identical to 42359's lr-2e-5 arm | 100 | 2e-5 | – | – | **COMPLETED — THE FIX** | CE loss **DESCENDS**: 4.6125 (step 1) → 4.0763 (15) → 3.7539 (30) → 3.8523 (45) → 4.2214 (60) → 4.0509 (75) → 3.7624 (90) → 3.7183 (100, final). `Grad. Norm` still reported as 3.4e8–4.2e10 throughout (fluctuating up to ~100x step-to-step) even with clipping disabled — proves the *reported norm*, not the gradients themselves, is the bug. See Finding 12. |
| 42366 | **IN FLIGHT — sftfix**, SFT distill, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1e12` | 100 (target) | 2e-5 | 1024 | 4 | *pending* | Config `opd_val/sft_base_fixed.toml`. First run able to answer the original OPD-viability question without the clipping confound. **Do not touch — in flight; results to follow from coordinator.** |
| 42367 | **IN FLIGHT — opdfix**, OPD, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1e12` | 100 (target) | 2e-5 | 1024 | 16 | *pending* | Config `opd_val/opd_base_fixed.toml`. **Do not touch — in flight; results to follow from coordinator.** |
| 42368 | **IN FLIGHT — grpofix**, GRPO, student `PrimeIntellect/Qwen3-0.6B` (base, untuned), `max_norm = 1e12` | 100 (target) | 2e-5 | 1024 | 16 | *pending* | Config `opd_val/grpo_base_fixed.toml`. **Do not touch — in flight; results to follow from coordinator.** |
| 42380 | **opdk0**, OPD, base student, `teacher_top_k = 0` (pre-fix sweep control), `max_norm = 1e12` | 100 (trainer stalled 97/100) | 2e-5 | 1024 | 16 | `CANCELLED+` (01:04:49) | Shutdown deadlock (Finding 11, 5th occurrence): orchestrator finished 100/100, trainer stalled at step 97. Eval reward 0.0411 (step 1) → **0.0322** (step 100, true final). Reward stays noisy in a 0.01-0.07 band throughout, never hard-zeroing. Truncation 74.2% → 100%. k=0 uses the plain sampled-token estimator, so it is unaffected by the Finding 14 bug. |
| 42381 | **opdk8**, OPD, base student, `teacher_top_k = 8` (pre-fix, buggy loss), `max_norm = 1e12` | 100 (trainer stalled 97/100) | 2e-5 | 1024 | 16 | `CANCELLED+` (01:04:49) | Shutdown deadlock (Finding 11, 6th occurrence). Eval reward 0.0671 → **0.0000** (hard-zeroes for many consecutive evals, unlike k=0). Truncation 63.3% → 100%. Entropy 0.5936 → 6.0574 (peaked 8.3302 at step 23). Top-k `Ref KL` goes **negative** (min -0.0758) — direct evidence of the Finding 14 bug. |
| 42382 | **opdk20**, OPD, base student, `teacher_top_k = 20` (pre-fix, buggy loss), `max_norm = 1e12` | 100 | 2e-5 | 1024 | 16 | **COMPLETED** (00:45:01) | Eval reward 0.0464 → **0.0000** (peak 0.0589). Truncation 67.2% → 100%. Entropy 0.5894 → 7.2773 (peaked 8.5235 at step 49). TopK Mass 0.3526 → 0.0790 (bottomed 0.0062 at step 26). Top-k `Ref KL` goes **negative** (min -0.0841) — the bug is worse at higher k. |
| 42383 | **opdk50**, OPD, base student, `teacher_top_k = 50` (pre-fix) | – | 2e-5 | 1024 | 16 | **FAILED** | `openai.BadRequestError: Requested prompt logprobs of 50, which is greater than max allowed: 20` — the frozen teacher's vLLM server was launched without a `--model.max-logprobs` override (defaults to vLLM's own cap of 20). Motivated the `TEACHER_MAX_LOGPROBS` fix used by the post-fix sweep (see Finding 14). |
| 42384 | **opdk100**, OPD, base student, `teacher_top_k = 100` (pre-fix) | – | 2e-5 | 1024 | 16 | **FAILED** | Same failure mode as 42383: `Requested prompt logprobs of 100, which is greater than max allowed: 20`. |
| 42387 | **fixk8**, OPD, base student, `teacher_top_k = 8` (post-fix loss), `max_norm = 1e12` | 100 (trainer stalled 97/100) | 2e-5 | 1024 | 16 | `TIMEOUT` (04:00:11) | Shutdown deadlock (Finding 11, 7th occurrence) — this time the job ran past its wall-clock limit instead of being caught while still `RUNNING`. Last logged step (97): Ref KL 6.9163 → 2.0349 (always positive — confirms the fix), TopK Mass 0.2501 → 0.6049, Entropy 0.6126 → 2.8184. Eval reward peaked **0.0333**, ended 0.0000; truncation ended 100%. |
| 42388 | **fixk20**, OPD, base student, `teacher_top_k = 20` (post-fix loss), `max_norm = 1e12` | 100 | 2e-5 | 1024 | 16 | **COMPLETED** | Ref KL 7.8106 → 0.5358 (always positive), TopK Mass 0.3470 → 0.6301, Entropy 0.5975 → 3.4369. Eval reward peaked **0.0427**, ended 0.0000; truncation ended 100%. |
| 42389 | **fixk50**, OPD, base student, `teacher_top_k = 50` (post-fix loss + teacher `--model.max-logprobs 128`), `max_norm = 1e12` | 100 | 2e-5 | 1024 | 16 | **COMPLETED** | Ref KL 8.7924 → 0.7221, TopK Mass 0.4470 → 0.5880, Entropy 0.5705 → 4.5233. Eval reward peaked **0.0615** (best of the whole sweep), ended 0.0000; truncation ended 100%. |
| 42390 | **fixk100**, OPD, base student, `teacher_top_k = 100` (post-fix loss + teacher `--model.max-logprobs 128`), `max_norm = 1e12` | 100 | 2e-5 | 1024 | 16 | **COMPLETED** | Ref KL 9.1573 → 0.8769, TopK Mass 0.5379 → 0.7526, Entropy 0.5726 → 4.2743. Eval reward peaked **0.0483**, ended 0.0000; truncation ended 100%. |
| 42391 | **fixk20lr5e6**, OPD, base student, `teacher_top_k = 20` (post-fix loss), `max_norm = 1e12`, **lr 5e-6** | 100 | 5e-6 | 1024 | 16 | **COMPLETED** | Ref KL 7.8016 → 1.1098, TopK Mass 0.3561 → 0.4839, Entropy 0.6062 → 4.0794. Eval reward peaked **0.0553**, ended 0.0000; truncation ended 100%. Lower lr slows the KL descent relative to 42388's 2e-5 run but does not change the eventual reward-collapse outcome. |

### Verification discrepancies found while auditing the logs

The task brief this log was built from was largely accurate, but the following specific numbers
did not match the raw logs and have been corrected above / noted here (logs win):

1. **42216 grad norm**: brief said 1.2e8–2.8e8; the actual range across all 20 steps in
   `run/logs/trainer.log` is **9.27e7–4.55e8** — wider on both ends.
2. **42216 truncation**: brief said "~78–84% flat"; actual per-step train truncation ranges
   **71.9%–86.7%**.
3. **42226 grad norm**: brief said 1.7e7–7.2e7; actual range is **1.31e7–7.78e7**.
4. **42226 final eval reward**: brief said "flat 0.145→0.108." The literal final eval (step 21,
   the post-loop eval, policy v17) is **0.0689**, not 0.108. 0.108 is the reward at step 20 (the
   last eval inside the nominal 20-step loop, policy v16). Both 42216 and 42226 have 21 (not 20)
   evals for a `max_steps = 20` run because one extra eval fires after the loop drains; for 42216
   the brief happened to quote the true final (step-21) value, for 42226 it quoted step 20. Use
   step-21/final values for apples-to-apples comparison: 42216 0.1082 vs 42226 0.0689.
5. **42216 `<think>`-open funnel**: brief said 103/137 (75%) rollouts opened `<think>` (token id
   151667) at step 8. Direct token-level count on
   `run/run_default/rollouts/step_8/train/all/traces.jsonl` gives **104/137 (75.9%)** — an
   off-by-one, immaterial to the conclusion. The `</think>` (66/137, 48.2%) and
   `</reversed_text>` (30/137, 21.9%) counts, and the reward split (0.7519 mean for the 30 that
   closed the answer tag vs. exactly 0.0 for the other 107), matched the brief exactly.
6. **42283 truncation**: brief said "only 3–10%." That range is correct for the **eval-level**
   truncation (3.1–10.2%) but the **train-batch-level** truncation metric spans 0.0–21.9% (one
   step spikes to 21.9%); worth knowing both metrics exist and can diverge.
7. **42276 eval-reward range**: brief characterized it loosely as "~0.10–0.13"; the actual spread
   across all 61 evals is wider, **0.0611–0.1713**. The "flat/noisy, no trend" characterization
   still holds — this is not a directional discrepancy, just a wider noise band than stated.
8. **42285 status (RESOLVED)**: the original brief said the run "was still RUNNING when this log
   was commissioned," implying training was in progress. This agent's audit instead showed the
   orchestrator's own log completing all 60 steps and exiting cleanly at 12:50:24, with the SLURM
   job still `RUNNING` per `sacct`/`squeue` 35+ minutes later and the trainer log stalled at
   step 57/60. The coordinator confirmed this reading independently and cancelled the job
   (`scancel 42285`) to free the node. Root cause is now understood as a trainer/orchestrator
   shutdown deadlock — see Finding 11.
9. **42364 (sft-noclip) final loss**: the coordinator's summary rounded the endpoint to "4.6125 ->
   3.76"; the literal final logged value at step 100 is **3.7183** (step 90 was 3.7624, which is
   likely what was rounded). Immaterial to the conclusion — the loss still clearly descends from
   4.61 to the high-3s range once clipping is disabled — but the exact number is corrected here.
10. **42381 (k=8, pre-fix) entropy endpoint**: brief said "0.59 -> 7.60". The true final logged
    value (step 97, the last step before the shutdown deadlock) is **6.0574**; 7.6047 is the
    step-40 value, an intermediate point, not the endpoint. Entropy actually peaked at 8.3302
    (step 23) then declined to 6.0574 by step 97.
11. **42382 (k=20, pre-fix) entropy / TopK Mass endpoints**: brief said "0.59 -> 8.13" and
    "0.353 -> 0.008". Step-1 values match, but this run actually **COMPLETED** to step 100 with
    final values Entropy **7.2773** and TopK Mass **0.0790** — 8.13/0.008 are intermediate
    worst-point values near steps 24-28 (entropy peaked 8.5235 at step 49; TopK Mass bottomed
    0.0062 at step 26), not the literal step-100 endpoints, which had partially recovered.
12. **42380 (k=0, pre-fix) final eval reward**: brief said "0.041 -> 0.028". The true final eval
    (step 100) is **0.0322**; 0.0285 is the step-80 value. Also worth preserving precisely: k=0's
    reward stays noisy in the 0.01-0.07 band throughout the whole run and never hard-zeroes,
    whereas the k>=1 arms (42381, 42382, and every post-fix arm) flatline at literal 0.0000 for
    many consecutive evals near the end — this sharpens, rather than contradicts, the "k>=1
    collapses worse" narrative.
13. **42390 (k=100, post-fix) Ref KL / TopK Mass endpoints**: brief said "9.16 -> 0.85" /
    "0.54 -> 0.78". The literal step-100 values are Ref KL **0.8769** and TopK Mass **0.7526** —
    both within ~0.03 of the brief's figures. Immaterial rounding difference, not a directional
    issue (unlike item 14 below).
14. **42391 (k=20 @ lr 5e-6, post-fix) Ref KL / TopK Mass endpoints — a real discrepancy**: brief
    said "7.80 -> 1.53" / "0.36 -> 0.59". The literal step-100 values are Ref KL **1.1098** and
    TopK Mass **0.4839**. 1.53 is actually the step-90 Ref KL value, and 0.59 is a mid-run TopK
    Mass value seen repeatedly around steps 63-80 (e.g. 0.5985 at step 77) — both intermediate
    rather than final-step values, the same "quotes a nearby extremum instead of the literal
    endpoint" pattern as items 10-11 above.
15. **Best-ever eval rewards for the post-fix sweep — confirmed exact.** 42387 (k=8) 0.0333, 42388
    (k=20) 0.0427, 42389 (k=50) 0.0615, 42390 (k=100) 0.0483, 42391 (k=20@5e-6) 0.0553 — all five
    match the brief exactly against `orchestrator.log`'s `Evaluated reverse-text` lines.
16. **The `max_logprobs = 20` vs. `teacher_top_k = 50/100` puzzle — resolved, not a discrepancy.**
    `run/configs/inference.toml` shows `[model] max_logprobs = 20` for every job in this batch,
    including 42389/42390 (k=50/100), which looked contradictory at first (a `prompt_logprobs = 50`
    request should be rejected by a server capped at 20, the same way it was for 42383/42384). The
    resolution: that `inference.toml` is the **student's own** rollout-sampling vLLM server
    (`:8000`, model `PrimeIntellect/Qwen3-0.6B`), which never needs more than a handful of
    logprobs and is irrelevant to top-k teacher scoring. The **frozen teacher** server (`:8001`,
    launched separately by `opd_val/rl_opd.sbatch:113` with
    `--model.max-logprobs ${TEACHER_MAX_LOGPROBS:-128}`) is the one that serves
    `prefill_logprobs_topk` requests. Confirmed directly: 42383/42384's `teacher.log` has no
    `max_logprobs` override (vLLM's own default of 20 applies, hence the `BadRequestError`), while
    42389/42390's `teacher.log` shows `'max_logprobs': 128` as a non-default launch arg — high
    enough for k=100 (needs support width 101). This is the actual fix for the 42383/42384
    failures, not a `model.max_logprobs` change in the run's own `inference.toml`.

## Results table — GSM8K (Phase 3)

Student `Qwen/Qwen3-1.7B`, teacher `Qwen/Qwen3-8B`, thinking mode **off**
(`enable_thinking = false`). Band selection and offline eval use `opd_val/gsm8k_band.py`.

| Job ID | Arm | Steps | LR | SLURM state (Elapsed) | Outcome / key metric |
|---|---|---|---|---|---|
| 42459 | sft-ladder, standalone `sft`, checkpoints at 15/30/45/60 | 60 | — | **COMPLETED** | Loss ~3.7-3.9 destabilizing to 4.2-4.7 around steps 45-60, `Grad. Norm` spiking as high as 3.87e11 in that region. Wrote 4 checkpoints under `runs/sft-ladder_42459/sft/weights/`. |
| 42463 | eval-ladder — offline eval of 42459's checkpoints on **reverse-text** (not GSM8K) | — | — | **COMPLETED** | Header confirms reverse-text reference numbers (teacher 0.799, base 0.04, released SFT 0.72, target band 0.3-0.6). Per-checkpoint mean reward: step_15 **0.078**, step_30 **0.078**, step_45 **0.000**, step_60 **0.015** — none land in the 0.3-0.6 target band; reverse-text still has no usable intermediate-student checkpoint. |
| 42470 | first GSM8K band attempt — **INVALID**, thinking ON | — | — | **COMPLETED, invalidated** | Student 0.7607 (truncated 2015/4096, 49.2%, pinned at the 1024-token cap), teacher 0.7783 (truncated 568/1024, 55.5%) — the apparent 0.7607-vs-0.7783 gap is a truncation artifact (Finding 17's renderer/eval asymmetry — thinking mode wasn't actually off for eval). Superseded by 42475. |
| 42475 | band comparison, thinking OFF, gsm8k vs math | — | — | **COMPLETED** | gsm8k: student 0.8181, teacher 0.9316 (gap +11.4 pts, 100/512 in-band `[0.2,0.8]`). math: student 0.8748, teacher 0.9326 (gap +5.8 pts, 61/512 in-band). gsm8k chosen for more headroom and a larger in-band pool. |
| 42486 | final boxed band selection (2048-problem gsm8k-boxed pool) | — | — | **COMPLETED** | Full-pool pass rates: student 0.8627, teacher 0.9446. Selected **327/2048** in-band `[0.2,0.8]` problems (by student pass rate) and wrote `data/gsm8k_boxed_band/train.parquet` — this is the band-selection step; the 327 problems it picked are reused by every job below. Full-pool pass rates here are not directly comparable to 42498's on-band numbers (different denominators/purpose — see note below the table). |
| 42498 | band ceiling — re-scored specifically on the 327 selected problems | — | — | **COMPLETED** | ON THE BAND: student **0.5879**, teacher **0.8933** (2616 rollouts, 8/problem). This is the baseline/ceiling pair used everywhere below — it, not 42486's full-pool numbers, is what makes the experiment well-posed (student sits mid-band by construction). |
| 42512 | g8sft, standalone-style `rl` sft arm, 2-way DP inference | — | — | `FAILED` (00:15:44) | `httpcore.ReadTimeout` on `POST /resume`. Root-caused by Finding 18 (validator-ordering bug) — confirmed directly: resolved config has `inference.parallel.dp = 2` but `inference_world_size = 1` in both `trainer.toml` and `orchestrator.toml`. |
| 42513 | g8opd, same setup, opd arm | — | — | `FAILED` (00:15:56) | Same `httpcore.ReadTimeout` on `/resume`, same root cause as 42512. |
| 42514 | g8grpo, same setup, grpo arm | — | — | `FAILED` (00:14:13) | Same `httpcore.ReadTimeout` on `/resume`, same root cause as 42512. |
| 42526 | g8sft2, SFT arm, **lr 2e-5** (mistaken — borrowed from the SFT isolation sweep) | 100 | 2e-5 | — | See Finding 16. `<think>`-tag trace confirms the sft arm drives `<think>` from 128/128 (step 1) to 18/128 (step 100) — the real signal in-run eval couldn't see (Finding 17). |
| 42527 | g8opd2, OPD arm, lr 2e-5 (mistaken) | 100 | 2e-5 | — | Train-batch `Reward` series is noisy and drifts down over the run (step-1 0.6016, ending near 0.20-0.22 in the last few steps) — consistent with the lr-2e-5 instability narrative, though not literally monotonic. |
| 42528 | g8grpo2, GRPO arm, lr 2e-5 (mistaken) | 100 (trainer stalled 97/100) | 2e-5 | `CANCELLED+` (01:03:23) | Deadlock (see census below). `Mismatch KL`: step 1 **0.0007** (matches Finding 16), step 80 **0.0193**, true final (step 97) **0.1002** — see discrepancy note below. Train `Reward` series starts **0.6786** (not 0.578) and is noisy/non-monotonic (range ~0.19-0.73), not a clean decline. `Trainable` oscillates 64/128-128/128 throughout (112/128 at step 1, 128/128 most common) — never starves, consistent with Finding 16's "band was working the whole time" but not a literal "112->128 rise." |
| 42550 | g8sft3, second-attempt SFT arm, lr 2e-5 (mistaken) | 100 | 2e-5 | — | Trace-confirmed Finding 17 evidence: step-1 eval `<think>` open 128/128, closed 19/128 (exact match); step-100 eval `<think>` open **18/128** (exact match to "128/128 down to 18/128"). Step-1 eval Reward **0.0391**, Truncation **96.1%** (matches "~0.03 reward / ~96% truncation"); step-1 **train** Reward is **0.9141** at 0% truncation (not the "0.60" the brief cites for "identical rollouts" — see discrepancy note). |
| 42551 | g8opd3, second-attempt OPD arm, lr 2e-5 (mistaken) | 100 | 2e-5 | — | Second-attempt companion to 42550/42552; not independently re-verified metric-by-metric beyond confirming lr=2e-5. |
| 42552 | g8grpo3, second-attempt GRPO arm, lr 2e-5 (mistaken) | 100 (trainer stalled 97/100) | 2e-5 | `CANCELLED+` (00:41:17) | Deadlock (see census below). `Mismatch KL`: step 1 **0.0006**, step 80 **0.0043**, final (step 97) **0.0061** — flatter than 42528's. Train `Reward`: step 1 **0.5781** (matches the brief's "0.578" almost exactly), step 99 **0.3594** (matches "0.359") — but the series in between is noisy/non-monotonic (e.g. up to 0.7054 at step 96, 0.7031 at the true final logged step 97), so "0.578 -> 0.359" is a real pair of logged values from this job, not a monotone decline — see discrepancy note. |
| 42563 | **opd-lr1e-6**, OPD, lr 1e-6 (corrected) | 100 (trainer stalled 97/100) | 1e-6 | `CANCELLED+` (01:06:13) | Deadlock (see census below). `Mismatch KL` flat: 0.0006 -> 0.0006. Checkpoints written: step_25/50/75 only (no step_100 — lost to the deadlock). Offline eval (42598): s25 0.5979, s50 0.6009, s75 0.6223. |
| 42564 | **grpo-lr1e-6**, GRPO, lr 1e-6 (corrected) | 100 (trainer stalled 97/100) | 1e-6 | `CANCELLED+` (01:06:12) | Deadlock (see census below). `Mismatch KL` flat: 0.0006 -> 0.0006. Checkpoints written: step_25/50/75 only. Offline eval (42598): s25 0.6116, s50 0.6086, s75 0.6330. |
| 42565 | **opd-lr3e-6**, OPD, lr 3e-6 (corrected) | 100 | 3e-6 | **COMPLETED** (00:24:24) | Only one of the four lr-corrected arms to reach step 100 (no deadlock). `Mismatch KL` flat: 0.0006 -> 0.0006. Checkpoints written: step_25/50/75/**100**. Offline eval (42598): s25 0.6124, s50 0.6460, s75 0.6430, s100 **0.6674** (best in the sweep, but ran longer than the other three). |
| 42566 | **grpo-lr3e-6**, GRPO, lr 3e-6 (corrected) | 100 (trainer stalled 97/100) | 3e-6 | `CANCELLED+` (01:06:03) | Deadlock (see census below). `Mismatch KL` flat: 0.0005 -> 0.0007. Checkpoints written: step_25/50/75 only. Offline eval (42598): s25 0.6055, s50 0.6261, s75 0.6346. |
| 42598 | checkpoint-eval, offline `gsm8k_band.py --dataset local` over all 4 lr-corrected arms' checkpoints | — | — | **COMPLETED** | The HEADLINE table (see TL;DR): baseline 0.5879, teacher 0.8933. All four arms improve monotonically at every checkpoint they reached; see per-job rows above for exact numbers. 1308 rollouts (327 x 4) per checkpoint. |
| 42611-42618 | **IN FLIGHT — do not touch.** `L-opd-s{0,1,2}`, `L-grpo-s{0,1,2}` (42611-42616, confirmed `RUNNING` via read-only `sacct`), plus two more (42617/42618, `PENDING`, no run dir yet — presumably the `teacher_top_k in {20,100}` arms) | 300 (target) | 3e-6 | RUNNING (42611-42616) / PENDING (42617/42618) | 8 arms at 300 steps, ckpt every 50: opd/grpo x seeds {0,1,2} for variance, plus opd `teacher_top_k` in {20,100} against the k=0 seed-0 arm. Addresses single-run variance, whether the arms were still climbing at 100 steps, and whether top-k helps now that OPD demonstrably works. Confirmed via `sacct` only — **no job in this range was submitted, cancelled, or otherwise touched.** |

### Verification discrepancies — GSM8K batch

17. **42598 headline table — confirmed exact.** Every pass-rate number in the coordinator's
    headline table (baseline 0.5879, teacher 0.8933, and all s25/s50/s75/s100 values for
    42563-42566) matches `opd_val/logs/evalckpt_42598.out` verbatim.
18. **42528 (grpo, lr 2e-5) `Mismatch KL` "0.0007 -> 0.0193" understates the true severity.**
    Step 1 = 0.0007 matches. But 0.0193 is the **step-80** value, not the run's endpoint — the
    true final logged value (step 97, the last step before the shutdown deadlock) is **0.1002**,
    roughly 5x worse than 0.0193 and ~150x the starting value, not 27x. This is the same
    "quotes an intermediate value as if it were the endpoint" pattern documented repeatedly in
    items 10-11/14 above, but here it makes Finding 16's diagnostic point *stronger*, not weaker
    — the runaway was worse than stated.
19. **"grpo train reward 0.578 -> 0.359 monotone decline" is real data, but attributed to the
    wrong framing.** This pair does not match job 42528 (the job Finding 16 was primarily
    discussed against) at all — 42528's train reward starts at **0.6786**, not 0.578, and is
    noisy/non-monotonic. It matches job **42552** (the *second*-attempt grpo lr-2e-5 run) almost
    exactly: step 1 = **0.5781** (rounds to 0.578), step 99 = **0.3594** (rounds to 0.359).
    However, (a) the two Finding-16 headline numbers ("0.0007->0.0193" mismatch KL and
    "0.578->0.359" reward) thus come from **two different jobs** (42528 and 42552 respectively)
    silently conflated into one before/after narrative, and (b) 42552's series is not actually
    monotone — it ranges noisily up to 0.7054 (step 96) and its true final logged step (97) is
    **0.7031**, higher than the start, not lower. The lr-2e-5 instability itself is real and
    well-evidenced elsewhere (mismatch KL, in Finding 16), but this specific reward pair should
    not be read as a clean decline curve.
20. **42550 "identical train rollouts scored 0.60 at 0% truncation" does not match this job's
    step-1 train reward.** Verified: step-1 **eval** reward is 0.0391 at 96.1% truncation
    (matches the brief closely), but step-1 **train** reward for the same job is **0.9141** at 0%
    truncation, not 0.60. The qualitative point (train reward is healthy while eval reward is
    crushed by truncation, because eval alone runs with thinking mode still on — Finding 17) is
    confirmed even more starkly than the brief states; the specific "0.60" figure just doesn't
    correspond to this job's step-1 value and likely refers to a different job or step.
21. **Deadlock census: 12 confirmed instances by this audit, not 13.** All five newly-cited jobs
    (42528, 42552, 42563, 42564, 42566) are independently confirmed `CANCELLED+` with the trainer
    stalled at step 97/100 — added to the previously-established count of 7 (see Finding 11 /
    Open Question 4 history) for a running total of **12**. The coordinator's claimed total of 13
    could not be reconciled from the jobs explicitly named in this batch; flagging rather than
    silently adjusting to match.
22. **"Three of the four lr-corrected arms never wrote their step_100 checkpoint" — confirmed
    exactly.** `opd-lr1e-6_42563`, `grpo-lr1e-6_42564`, and `grpo-lr3e-6_42566` each have only
    `step_25/50/75` weight directories; only `opd-lr3e-6_42565` (the one that `COMPLETED` rather
    than deadlocking) has `step_100`.

## Findings

### 1. THE HEADLINE: the "weak student" premise is a token-budget artifact, not a capability gap

The initial framing — "student reward 0.223 vs. teacher 0.799" (job 42224) — was measured at
`max_completion_tokens = 128`. At that budget the student (46/64 truncated) frequently gets cut
off before closing `<reversed_text>`, while the teacher (2/64 truncated) reliably finishes. Once
`max_completion_tokens` is raised to 512 (jobs 42283/42284/42285), the student scores
**~0.72–0.75** against the teacher's **0.799** — a gap of roughly 0.05–0.08, not 0.58. All three
512-token 60-step arms (SFT-distill 42283, OPD 42284, GRPO 42285) are flat over 60 steps because
there is almost no room left to close. **`reverse-text` with this exact student/teacher pair is a
poor task for validating any distillation/RL algorithm** — conclusions about OPD's effectiveness
cannot be drawn from it until a task with real headroom is found (see Open questions).

### 2. ROCm: `torch.compile` must be disabled

Job 42211 (the first full OPD attempt, `[model.compile]` left at its default with
`fullgraph = false`) crashed the trainer with:

```
assert_size_stride(buf13, (16, 2047), (2047, 1), 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
AssertionError: expected size 2047==16, stride 16==2047 at dim=0; expected size 16==2047, stride 1==1 at dim=1
```

Inductor's meta (fake) kernel for `torch.ops.flash_attn._flash_attn_varlen_forward` reports a
transposed shape relative to what the real ROCm kernel produces. **Fix:** set `compile = "None"`
under `[trainer.model]` — the literal string `"None"` parses to Python `None` in this config
system (verified: every subsequent config, `opd_512.toml`, `grpo_512.toml`, `sft_512.toml`,
`opd_long.toml`, `grpo_long.toml`, `grpo_control.toml`, `base_512.toml`, `opd_reverse_text.toml`,
carries the comment "torch.compile is off on ROCm..." followed by `compile = "None"`). The same
workaround already exists in the repo's `rl_train_singlenode_mixed.sbatch:267`.

### 3. ROCm: never set `HIP_VISIBLE_DEVICES` for the combined `rl` entrypoint

The `rl` entrypoint places the inference server and trainer on separate GPUs by setting
`CUDA_VISIBLE_DEVICES` on its child processes. On ROCm, if `HIP_VISIBLE_DEVICES` is *also* set in
the parent environment, ROCm gives it precedence over `CUDA_VISIBLE_DEVICES`, silently collapsing
both children onto the same device. `opd_val/rl_opd.sbatch:138-139` documents this explicitly:

```
# HIP_VISIBLE_DEVICES is intentionally left unset: on ROCm it overrides
# CUDA_VISIBLE_DEVICES, which is what the rl entrypoint uses to put inference on
```

Standalone single-process launches (e.g. the separately-launched frozen teacher server on GPU 2)
can safely use `HIP_VISIBLE_DEVICES` because nothing else is setting `CUDA_VISIBLE_DEVICES` for
that process (`rl_opd.sbatch:104-110`).

### 4. Upstream bug: hardcoded shared `/tmp` path for the verifiers subprocess harness

`deps/verifiers/verifiers/v1/runtimes/base.py:173`:

```python
path = f"/tmp/vf-scripts/{digest}.py"
```

This has no `TMPDIR`/scratch awareness. With host `/tmp` bind-mounted read-write into the
container (`rl_opd.sbatch:83`, `-B /tmp:/tmp:rw`), `/tmp/vf-scripts` is shared across every user
and job on a node; whoever creates it first owns it, so any other job's rollouts die with:

```
HarnessError: harness setup: PermissionError: [Errno 13] Permission denied: '/tmp/vf-scripts/<digest>.py.<tmp>.tmp'
```

**Confirmed by grepping orchestrator logs**: job **42227** hit this pervasively (the orchestrator
log for that run contains millions of matching lines — every single rollout in the 98-minute
run failed this way, which is why it was cancelled). Jobs **42216** and **42226** — run *before*
42227, without the workaround — had **zero** occurrences, i.e. this is genuinely node-dependent
flakiness (a race for who creates the directory first), not deterministic. **Workaround applied**
in the current `rl_opd.sbatch:90-91` (bind a job-private directory over just that path):

```
mkdir -p "$SCRATCH/vf-scripts"
BIND_MOUNTS="$BIND_MOUNTS -B $SCRATCH/vf-scripts:/tmp/vf-scripts:rw"
```

Confirmed zero `vf-scripts` errors in all five subsequent runs (42276, 42277, 42283, 42284,
42285) after this fix was added. Worth reporting upstream to the `verifiers` project.

### 5. `Trainable 0/128 (0.0%)` is meaningless (expected) under `opd`/`opsd`

`Rollout.is_trainable` (`src/prime_rl/orchestrator/types.py:131-134`):

```python
@property
def is_trainable(self) -> bool:
    """Whether the rollout carries a training signal — a nonzero advantage on some token. A
    uniform-reward GRPO group (all-zero advantages) or an unscored rollout has no gradient."""
    return bool(self.advantages) and any(a != 0.0 for a in self.advantages)
```

Reference-KL algorithms (`opd`/`opsd`) never assign scalar advantages by design — the `ref_kl`
loss term in `src/prime_rl/trainer/rl/loss.py:199` (`ref_kl_loss_fn`) reads `ref_logprobs`
directly and explicitly documents "Scalar advantages are not read — ref_kl algorithms ship none."
So `Trainable 0/128 (0.0%)` is **structurally always false** for OPD and the accompanying
orchestrator warning ("consider reviewing task difficulty / filter config" — confirmed present in
`opd-reverse-text_42216`'s log) is spurious noise, not a signal of a broken run. Confirmed by
contrast: GRPO runs (42226, 42285) correctly show `Trainable 128/128 (100.0%)` (or slightly less
when a uniform-reward group appears, e.g. `Trainable 112/128` at 42226 step 3). The real health
signals for an OPD run are (a) whether batches ever come up empty/abort, and (b) whether
`Policy vN` advances step over step.

### 6. Gradient norms of ~1e7–1e9 are NOT OPD-specific, and remain unexplained

The GRPO control (42226: 1.31e7–7.78e7; 42285: values up to 9.4e8 observed) reproduces the same
order-of-magnitude gradient norms seen in OPD runs (42216: 9.27e7–4.55e8; 42276/42284: similar).
This is **pre-existing prime-rl-on-ROCm behaviour**, not something OPD introduces. A back-of-
envelope estimate of the `ref_kl` gradient magnitude — roughly `ref_kl / ref_kl_scale ≈ 1.6e-5`
per token at initialization — is many orders of magnitude smaller than the ~1e8 figures actually
logged. **This is flagged as an open question, not a solved one** — it has not been root-caused,
and this log should not be read as claiming the discrepancy is understood or benign.

### 7. lr 1e-5 destabilizes this setup; lr 3e-6 is stable but slow at 20 steps

At lr 1e-5, entropy rose monotonically-ish in both OPD (42276: 0.654→0.951) and GRPO
(42277: 0.684→1.099), and GRPO's reward outright collapsed (0.131→0.061, dipping to 0.016 near
step 59). At lr 3e-6, 20-step runs (42216, 42226) show no clear trend in either direction for any
algorithm — they are simply too short to move the needle. **Lesson for future runs**: never read a
flat short run as "the algorithm doesn't work" without a matched control at the same
hyperparameters — the 20-step OPD and GRPO curves are statistically indistinguishable from each
other and from noise.

### 8. `<think>` behaviour: use `</think>` (151668), not `<think>` (151667), to test whether reasoning finished — except this checkpoint's chat template doesn't inject either

For stock Qwen3 chat templates, `<think>` is often auto-injected into the generation prompt (for
"thinking mode"), so testing for reasoning *completion* means checking for `</think>` rather than
`<think>`. **This checkpoint is an exception**: neither its own tokenizer chat template nor
prime-rl's custom `prime-qwen3` renderer injects `<think>` into the generation prompt.

- Renderer: `deps/renderers/renderers/prime_qwen3.py:582-594`
  (`PrimeQwen3Renderer._render_generation_prompt`) emits only `<|im_start|>` + the literal text
  `"assistant\n"` — no `<think>` special token is ever appended here (it registers for this model
  family in `deps/renderers/renderers/base.py:1004-1005`, `configs.py:187,698`).
- Tokenizer: the checkpoint's own `chat_template.jinja` (under
  `$HF_HOME/hub/models--PrimeIntellect--Qwen3-0.6B-Reverse-Text-RL/.../chat_template.jinja`) ends
  `add_generation_prompt` with `'<|im_start|>assistant\n'` only — no `<think>\n\n</think>\n\n`
  auto-injection like some stock Qwen3 templates use for non-thinking mode.

So here the model genuinely **chooses** to emit `<think>` itself (or not) as its first tokens.
Verified via direct token-level inspection of
`run/run_default/rollouts/step_8/train/all/traces.jsonl` (job 42216, step 8, N=137 rollouts):

| Milestone | Count | % |
|---|---|---|
| Opens `<think>` (token 151667 present) | 104/137 | 75.9% |
| Emits `</think>` (token 151668 present) | 66/137 | 48.2% |
| Emits `</reversed_text>` (closes the answer) | 30/137 | 21.9% |

Reward split: **0.7519 mean** for the 30 rollouts that closed `</reversed_text>`, **exactly 0.0**
for the other 107. At `max_completion_tokens = 128`, most of the token budget is consumed by
reasoning that never reaches a closed answer tag — this is the same conciseness-artifact pattern
underlying Finding 1.

### 9. Estimator mismatch vs. Thinking Machines' OPD write-up (untested — open question)

`src/prime_rl/utils/client.py:590` requests `"prompt_logprobs": 1` when the teacher scores the
student's trajectory (`prefill_logprobs`) — i.e. prime-rl's `ref_kl_loss_fn`
(`src/prime_rl/trainer/rl/loss.py:199-224`) uses only the teacher's logprob for the **sampled**
token:

```python
ref_kl = ref_logprobs - trainer_logprobs
pg_loss = keep_mask * ref_kl.detach() * importance_ratio
```

This is the score-function/REINFORCE estimator, `(log π_ref − log π_θ) · ∇log π_θ` (up to the
importance-ratio correction for trainer/inference mismatch). Thinking Machines' OPD write-up
computes the per-token reverse KL **exactly** by summing over the full vocabulary, which requires
the teacher's complete distribution, not just the sampled token's logprob. The two estimators have
the same expectation (the sampled-token version is unbiased because the `∇Σ_v π(v) = ∇1 = 0` term
vanishes), but the sampled-token version has higher variance. The sharper practical concern: the
exact-KL estimator puts gradient on every vocabulary entry and can push probability mass off
tokens the teacher dislikes *without the student ever having sampled them*, whereas the
sampled-token estimator can only learn to avoid a bad token after actually emitting it at least
once. **This has not been tested in this validation effort.** A bounded follow-up experiment:
request `prompt_logprobs: k` (k≈20) in `prefill_logprobs`, ship per-token top-k `ref_logprobs`
over the wire, and compute the KL over that support in `ref_kl_loss_fn`.

### 10. What IS validated about OPD mechanics on this stack

- Teacher prefill-scoring over `/inference/v1/generate` + `prompt_logprobs` works correctly on
  ROCm/vLLM 0.25.1: one correctly-aligned logprob per token, `0.0` at position 0, bit-deterministic
  across repeated identical calls (job 42209).
- `ref_logprobs` reach the trainer and the `ref_kl` loss component executes without error.
- RCCL/NCCL weight broadcast between trainer and inference server works.
- Full OPD runs complete with `rc=0` at both 128-token and 512-token budgets, at both lr=3e-6 and
  lr=1e-5.
- **The teacher must be a prime-rl `uv run inference` server**, not a generic OpenAI-compatible
  endpoint — a plain OpenAI-compatible server has no `/inference/v1/generate` route, which is what
  `prefill_logprobs` needs (documented in `opd_reverse_text.toml`'s header comment and confirmed
  by the 42207 container-triage failure mode).

### 11. Trainer/orchestrator shutdown deadlock can leave a SLURM job `RUNNING` indefinitely (confirmed, mechanism unconfirmed)

Job 42285 (GRPO, 512 tokens, 60-step target) is the confirmed instance: the orchestrator logged
`Orchestrator finished.` at 12:50:24 having completed all 60 of its steps, but the trainer's log
stalled at **step 57/60**, and no process (trainer, orchestrator, or inference server) wrote
another log line after 12:50:38. `sacct`/`scontrol show job` reported the SLURM job as `RUNNING`
for 35+ minutes past the orchestrator's exit, holding an entire node idle. The coordinator
confirmed this reading and cancelled the job (`scancel 42285`) to reclaim the node; the run's
already-written eval data (through step 60) is unaffected and is recorded in the Results table.

**Mechanism** — when the trainer falls behind the orchestrator's step counter at/near
`max_steps`, the orchestrator can drain its step loop and exit while the trainer is still blocked
waiting for a batch that will never be dispatched, so the trainer process (and the SLURM job that
wraps it) never terminates on its own. One plausible contributor, **unconfirmed**: GRPO's
`zero_advantage` post-batch filter (on by default) drops rollout groups whose rewards are uniform
(no advantage signal), and near a performance ceiling — exactly the regime all three 512-token
arms are in per Finding 1 — a larger fraction of sampled groups can be uniform-reward and get
filtered, so fewer batches ship to the trainer than orchestrator steps counted. This would explain
why the trainer undershoots (57 vs 60) specifically in a near-ceiling GRPO run, but it has been
**observed in only this one run so far** and has not been verified against the orchestrator's
batch-dispatch/filter logs for 42285 — treat as a hypothesis, not a diagnosis.

**Operational advice:** a completed orchestrator log is *not* proof that a run has finished. After
any run appears to finish, always check (a) `sacct -j <id>` for a terminal state
(`COMPLETED`/`FAILED`/`CANCELLED`), not just orchestrator log text, and (b) the trainer's last
logged `Step N` against `max_steps` — a mismatch (as with 42285's 57 vs 60) is the tell. See the
new "How to check a finished run" subsection under Environment & how to reproduce.

### 12. THE ROOT CAUSE: default `max_norm = 1.0` prevents learning entirely on this ROCm cluster

**Every flat curve documented above — every opd/grpo/sft result recorded before 2026-08-12
evening — is invalidated by this bug.** It was not the algorithms, the task, or the token budget;
it was broken gradient clipping.

Isolated with the standalone `sft` entrypoint (`uv run sft`) on the repo's own vetted config
`examples/basic/reverse-text/sft.toml` — 1 GPU, no orchestrator, no vLLM, no rollouts, no weight
broadcast, so the entire RL machinery is ruled out as a contributing factor:

| Job | `max_norm` | CE loss, step 1 → 100 | Verified trend |
|---|---|---|---|
| 42356 / 42359 (lr 2e-5) | 1.0 (default) | 4.6125 → 4.8955 | FLAT |
| 42364 (lr 2e-5) | 1e12 (effectively off) | 4.6125 → 3.7183 | **DESCENDS** (step 15: 4.0763, step 30: 3.7539, step 45: 3.8523, step 60: 4.2214, step 75: 4.0509, step 90: 3.7624) |

Confirmed against `run/logs/trainer.log` for both jobs — 42356's and 42359's `max_norm = 1.0`
lr-2e-5 curves are **bit-identical** to each other (verified via `diff`, zero content differences
in the `Step | Loss` fields), and 42364's descending curve, taken from the identical config with
only `max_norm` changed, matches the numbers above to 4 decimal places from the raw log.

**The reported `Grad. Norm` itself is the bug, not the underlying gradients.** In 42364
(`max_norm = 1e12`, i.e. clipping is a no-op), the trainer still logs `Grad. Norm` values of
3.4e8–4.2e10, fluctuating by roughly 100x from one step to the next (e.g. step 1: 3.40e8, step 15:
1.48e8, step 30: 2.22e9, step 45: 4.24e10, step 60: 2.01e9) — yet the loss descends normally. A
correct CE gradient norm for a 0.6B-parameter model at this loss scale should be O(1–10), not
O(1e8–1e10). Since disabling clipping (which only ever *reduces* gradients when the reported norm
exceeds the threshold) restores normal learning, the actual per-parameter gradients must be
reasonably well-scaled — it is specifically the **scalar norm value** feeding `clip_grad_norm_`
that is wrong, and the default `max_norm = 1.0` is small enough that it clips to (norm / 1e8-ish)
of the true gradient on nearly every step.

**Mechanism for why a wrong-but-highly-variable clip scale kills learning:** AdamW's moment
estimates (`m`, `v`) are EMAs over the (clipped) gradient. AdamW is scale-invariant only under a
*constant* rescaling; a clip factor that swings ~100x step to step effectively randomizes the
gradient's relative magnitude fed into the optimizer, corrupting `m`/`v` and producing an
effectively random walk rather than a useful update direction — consistent with the observed flat,
noisy loss curves across every earlier opd/grpo/sft run in this log.

**Not yet identified:** why `clip_grad_norm_` (imported from `torchtitan.distributed.utils`,
called at `src/prime_rl/trainer/sft/train.py:471` and `src/prime_rl/trainer/rl/train.py:558`)
computes such a wildly wrong and unstable magnitude on this cluster. Prime suspect is the
FSDP/DTensor norm reduction path on ROCm (e.g. a mis-scaled or mis-reduced all-reduce across
shards), but this has not been instrumented. **A per-parameter gradient-norm dump is the obvious
next diagnostic** — if a small number of parameters (or a single DTensor shard) show anomalous
norms while the rest are reasonable, that would pinpoint the reduction path.

**Ruled out as alternative explanations** (each verified bit-identical or otherwise directly
checked against logs, not just asserted):

- **`optim_cpu_offload`** (job 42362, `fsdp_cpu_offload = false`, `optim_cpu_offload = false` vs.
  42356/42359/42364's default `optim_cpu_offload = true`): confirmed the flag took effect (the
  `"Wrapping optimizer with CPUOffloadOptimizer"` log line is present in 42356's trainer.log and
  absent in 42362's), yet both lr arms (2e-5 flat, 1e-4 diverging) reproduce their
  offload-enabled counterparts' loss curves with **zero** differences after stripping ANSI codes
  and diffing the `Step | Loss` fields. Not the cause.
- **`torch.compile`**: already disabled cluster-wide for the unrelated ROCm flash-attn
  varlen meta-kernel bug (Finding 2) — every affected run already had it off, so it cannot be an
  additional explanation.
- **The orchestrator / vLLM / rollout / weight-broadcast path**: ruled out structurally — the
  pathology reproduces identically in the standalone `sft` entrypoint, which touches none of that
  code.
- **lr 1e-4 does not "fix" it either** — job 42359's lr-1e-4 arm (`max_norm = 1.0`) diverges rather
  than flattens (4.6125 → 10.6983 at step 20 → 13.5423 at step 40 → 12.4215 at step 100),
  confirming the fix is specifically about the clip norm, not about learning rate being too
  conservative.
- **Determinism check**: 42359's lr-2e-5 curve is bit-identical to 42356's surviving lr-2e-5 arm,
  confirming these SLURM runs are deterministic given the same config and that 42356's result
  remained valid despite an environment change mid-investigation (see Finding 13).

### 13. Environment incident: `uv sync` without `--all-extras` silently strips optional-extra packages (including `flash-attn`)

Recorded here so it isn't repeated. Installing the math environments with:

```
uv sync --package prime-rl --package math-env-v1 --package aime24-v1
```

(no `--all-extras`) **removed every optional-extra package from the venv**, including
`flash-attn`, breaking job 42356's second arm (lr 1e-4) and job 42357 (a `hendrycks-sanity` smoke
test) outright — both failed with `ModuleNotFoundError: No module named 'flash_attn'` (confirmed
in `opd_val/logs/opd_42357.err:29` and `sft-isolation_42356/sft_lr_1e-4.log`, the latter showing
the full import chain: `entrypoints/sft.py` → `trainer/model.py` → `trainer/lora.py` →
`trainer/models/__init__.py` → `trainer/models/glm_moe_dsa/...` →
`prime_rl/utils/cp.py:12` → `ring_flash_attn/__init__.py` → `flash_attn.flash_attn_interface`,
i.e. an unrelated GLM-MoE-DSA model import transitively pulls in `ring_flash_attn`, which requires
`flash_attn`, at trainer-entrypoint import time even when the active run doesn't touch that model
family).

Conversely, `uv sync --all-extras` alone **removes the workspace env packages**
(`math-env-v1`/`aime24-v1`, which are separate workspace members, not extras). **The correct
recovery installs both together:**

```
uv sync --all-extras --package prime-rl --package math-env-v1 --package aime24-v1
```

`pyproject.toml` and `uv.lock` were not modified by this incident or its recovery.

Two more environment notes worth keeping on record:

- A bare `import ring_flash_attn` fails on transformers>=5.4 (`is_flash_attn_greater_or_equal_2_10`
  was removed from `transformers.modeling_flash_attention_utils`). This is **expected and
  harmless** — the repo ships `src/prime_rl/_compat.py` specifically to shim it
  (`_mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True` if the attribute is missing), and
  it's imported before any transitively-affected model code by
  `src/prime_rl/trainer/sft/train.py:1`, `src/prime_rl/trainer/rl/train.py:1`,
  `src/prime_rl/orchestrator/orchestrator.py:40`, `src/prime_rl/utils/cp.py:4`, and
  `src/prime_rl/trainer/models/layers/ring_attn.py:4`.
- `math-env-v1` and `aime24-v1` live under
  `deps/research-environments/environments/math/math_env_v1/` and
  `deps/research-environments/environments/math/aime24_v1/` respectively (directory names use
  underscores; their `pyproject.toml` `name` fields use hyphens, `math-env-v1` / `aime24-v1` — this
  is the normal PEP 503 dash/underscore normalization, not a typo). Neither is installed by
  default; both require an explicit `--package` flag as shown above.

### 14. `teacher_top_k` implemented — first version had a real bug (partial sum != KL), now fixed

Open Question 2 asked for a top-k/exact reverse-KL estimator. It now exists:
`OPDAlgoConfig.teacher_top_k` (`packages/prime-rl-configs/src/prime_rl/configs/algorithm.py:253`,
default 0 = old sampled-token behaviour) makes the teacher ship its per-token top-k support
(ids + logprobs) via `prefill_logprobs_topk` (`src/prime_rl/utils/client.py`, ~line 609), and
`ref_kl_loss_fn` (`src/prime_rl/trainer/rl/loss.py:227-316`) evaluates the reverse KL over that
support directly instead of only the sampled token's logprob. This requires
`fused_lm_head_token_chunk_size = "disabled"` under `[trainer.model]` (confirmed present in every
`opd_val/opd_k{0,8,20,50,100}.toml` and `opd_k20_lr5e6.toml`) — the fused head never materializes
logits, which the top-k path needs to gather the policy's own logprobs on the teacher's support.

**The first version of this had a genuine design flaw**, documented directly in the current source
(`src/prime_rl/trainer/rl/loss.py:289-299`):

```python
# Everything outside the support, lumped into a single bucket for both
# distributions. Without this the objective is a *partial* sum, not a
# KL, and has a degenerate minimum: driving pi(v) -> 0 for every v in
# the support sends each term to 0, so the policy escapes by dumping its
# mass on the untracked rest of the vocabulary (observed directly —
# topk_mass collapsing to ~0.01 while entropy ran to 8+).
```

`sum_{v in S} pi(v) (log pi(v) - log pi_ref(v))` over a fixed support `S` is **not** a valid KL
estimate: as `pi(v) -> 0` for every `v` in `S`, every term vanishes, so the loss can be minimized by
evacuating probability mass off the support entirely — the opposite of what a real KL penalizes.
This is exactly what happened in the pre-fix sweep (jobs 42380-42382, all with `teacher_top_k` in
{0, 8, 20}): TopK Mass collapsed toward ~0.01-0.08 while entropy ran up to 7-8.5, and — the clean,
unambiguous evidence — **the reported top-k `Ref KL` went negative**, which is mathematically
impossible for a true KL divergence:

- 42381 (k=8): top-k `Ref KL` reaches a minimum of **-0.0758** (step 97).
- 42382 (k=20): top-k `Ref KL` reaches a minimum of **-0.0841** (step 100) — more negative at
  higher k, consistent with a larger support giving the degenerate escape route more room to work.

(For contrast, 42380's k=0 run also logs negative `Ref KL` values, down to -0.8370 — this is
*expected* and not a bug: k=0 uses the plain sampled-token score-function estimator
[`ref_logprobs - trainer_logprobs`, a per-token log-ratio], whose individual samples are unbounded
in sign even though its expectation is a true KL. The top-k branch's `Ref KL`, by contrast, is a
directly-computed categorical divergence over the whole support and must be non-negative if
correctly implemented — so its negativity is diagnostic, k=0's is not.)

**The fix**: coarsen everything outside the support into one additional category for both
distributions (the teacher's off-support mass is `1 - sum_S pi_ref`, likewise for the policy),
turning the objective into a genuine KL between two `(k+1)`-category distributions — provably
non-negative, and a lower bound on the true vocabulary-wide KL by the data-processing inequality.
Confirmed fixed directly in the post-fix logs: every top-k `Ref KL` value across the 400 combined
trainer steps of jobs 42387-42391 is positive (ranges from 0.53 to 9.16 across the sweep — see the
Results table). No dedicated unit test for this function was found under `tests/` in this repo, so
the specific unit-test numbers mentioned in the brief (identical distributions -> 0.00000, 200
random policies -> min +0.40625, evacuating support -> +2.9956, monotone +0.82 -> +2.92) could not
be independently verified from a test file — but the production-log evidence above (negative
pre-fix, always-positive post-fix) directly confirms the same underlying claim from live training
runs, which is stronger evidence than a unit test would be anyway.

A second, unrelated bug surfaced in the same sweep and was also fixed: jobs 42383 (k=50) and 42384
(k=100) failed outright with `openai.BadRequestError: Requested prompt logprobs of 50/100, which is
greater than max allowed: 20` — the frozen teacher's vLLM server was launched without a
`--model.max-logprobs` override and defaulted to vLLM's own cap of 20. See discrepancy note 16
above for the full resolution: the fix is `opd_val/rl_opd.sbatch:113`'s
`--model.max-logprobs ${TEACHER_MAX_LOGPROBS:-128}` on the **teacher's own** server launch, not the
`model.max_logprobs` field in the student's `inference.toml` (which stays at 20 and is unrelated).

### 15. Post-fix top-k sweep: the estimator fix works mechanically, but does not rescue OPD from a base-model student

With the Finding 14 fix in place, jobs 42387-42391 (`teacher_top_k` in {8, 20, 50, 100}, plus a
k=20 run at 5x lower lr) all show the loss behaving exactly as the fixed math predicts: `Ref KL`
descends monotonically-ish from ~7-9 down to ~0.5-1.1, and `TopK Mass` (the fraction of the
policy's probability the support covers) rises from ~0.25-0.54 up to ~0.48-0.75 — i.e. the policy
is demonstrably moving toward the teacher's distribution on the tokens the top-k support tracks.
See the Results table for exact per-job endpoints.

**But this does not translate into task success for any arm.** Every one of the five post-fix runs
— and both pre-fix k>=1 runs, and even the k=0 control — ends with eval reward at **0.0000** (or,
for k=0, still-noisy-but-low ~0.03) and eval truncation saturated at **100%**. Best-ever eval reward
across the post-fix sweep: k=8 0.0333, k=20 0.0427, k=50 0.0615 (the sweep's best), k=100 0.0483,
k=20@lr-5e-6 0.0553 — all well below the teacher's ceiling (0.799, per Finding 1) and none showing
a rising trend before collapsing. Widening k (0 -> 8 -> 20 -> 50 -> 100) does **not** monotonically
improve outcomes, and halving the learning rate (42391 vs. 42388) slows the KL descent without
changing the terminal collapse.

**Leading interpretation**: this is very unlikely to be an estimator-variance or support-size
problem anymore — Finding 14 fixed the estimator, and Finding 15 shows the fixed estimator is
mechanically doing exactly what it should (KL down, top-k mass up) while the actual task metric
still fails identically regardless of k. The remaining candidate explanation is that **on-policy
reverse KL is mode-seeking**: starting from a base (untrained) student far from the teacher's
distribution, greedily minimizing reverse KL on-policy can drive the policy toward a narrow,
degenerate mode (consistent with entropy first rising sharply then falling as the policy commits to
some narrow behavior, and with truncation saturating at 100% — the policy is converging to
*something*, just not a useful, complete answer) rather than toward the broad, correct target
distribution. Forward-KL/cross-entropy training (SFT) is mode-covering and stable from a base model
precisely because it does not have this failure mode. This is consistent with (and would explain
why) the standard recipe is SFT-warmup *then* RL/OPD, rather than OPD directly from a base model —
see Open Question 5.

### 16. GSM8K collapse (jobs 42526-42528, 42550-42552) was a learning-rate misconfiguration, not an algorithm failure

Confirmed: `runs/g8grpo2_42528/run/configs/trainer.toml` has `lr = 2e-05`, which was borrowed from
the SFT isolation sweep (Finding 12/13). Every RL config in this repo uses `1e-6` to `3e-6`, and
`rl_train_singlenode_mixed.sbatch` uses `1e-6`.

- **At lr 2e-5**: `Mismatch KL` for 42528 runs away from 0.0007 (step 1) to **0.1002** (step 97,
  true final — worse than the 0.0193 intermediate value initially cited, see discrepancy note 18).
  Train reward for 42552 (the second lr-2e-5 grpo attempt) goes from 0.5781 (step 1) to 0.3594
  (step 99) but noisily, not monotonically (discrepancy note 19).
- **At lr 1e-6 and 3e-6** (42563-42566): `Mismatch KL` stays flat across all four arms (0.0005-
  0.0006 at step 1, 0.0006-0.0007 at the endpoint) — confirmed exactly against each run's
  `trainer.log`. No arm collapses.
- **The band was working the whole time even at lr 2e-5** — `Trainable` never starves (42528
  oscillates between 64/128 and 128/128 throughout, most commonly 128/128), i.e. abundant
  non-zero advantages were always present. The failure was never a lack of training signal; it
  was the learning rate itself, compounded by the `max_norm = 1e12` workaround (Finding 12) —
  disabling gradient clipping removes the main brake on policy-gradient step size, so an
  already-too-high lr has nothing checking it.

### 17. In-run eval is unusable for any config that changes generation behavior

The orchestrator builds train clients as `"renderer"` (which respects `enable_thinking = false`
etc.) but eval clients as `"openai_chat_completions"` (which does not) — confirmed exactly at
`src/prime_rl/orchestrator/utils.py:45-46`:

```python
    train_client_type="renderer",
    eval_client_type="openai_chat_completions",
```

So any config that changes generation behavior (like turning Qwen3 thinking mode off) only takes
effect for training rollouts, not for evaluation — **eval silently scores a different model than
the one being trained.** Directly confirmed on job 42550's rollout traces
(`run/run_default/rollouts/step_1/eval/all/traces.jsonl` and `.../step_100/eval/all/traces.jsonl`):

| Step | Eval rollouts opening `<think>` | Eval rollouts closing `</think>` |
|---|---|---|
| 1 | 128/128 | 19/128 |
| 100 | 18/128 | 0/128 |

At step 1, eval blew the 1024-token budget reasoning inside an un-closed `<think>` block: eval
reward **0.0391** at **96.1%** truncation, while the orchestrator's step-1 **train**-batch reward
(same policy, same problems, just routed through the renderer client) was **0.9141** at **0%**
truncation. Passing `chat_template_kwargs` via `eval.sampling.extra_body` does not fix it (present
in the resolved config; truncation stayed near-saturated). Workaround used throughout Phase 3:
checkpoint and evaluate offline via `opd_val/gsm8k_band.py --dataset local`.

One real signal did leak through despite the broken metric: by step 100, job 42550 (the sft arm)
had driven `<think>`-opening from 128/128 down to **18/128** — the model was in fact learning the
teacher's non-thinking style; the in-run eval reward/truncation numbers just couldn't see it.

Worth reporting upstream: a generation-affecting config silently applying to training but not
evaluation is a correctness footgun, not just a Phase-3-specific inconvenience.

### 18. A config-validator-ordering bug hangs multi-GPU inference `/resume` — retroactively explains jobs 42360, 42512-42514

Jobs 42512 (`g8sft`), 42513 (`g8opd`), 42514 (`g8grpo`) all `FAILED` after 14-16 minutes with
`httpcore.ReadTimeout` on `POST http://127.0.0.1:8000/resume`. Root cause, confirmed directly in
source: `packages/prime-rl-configs/src/prime_rl/configs/rl.py` defines two `@model_validator(mode
="after")` methods on `RLConfig` —

- `auto_setup_weight_broadcast` (line 338), which computes
  `inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp` (line ~359) to
  size the NCCL broadcast group, and
- `auto_setup_deployment` (line 535), which is what actually propagates
  `deployment.num_infer_gpus` into `self.inference.parallel.dp` (line ~549).

Pydantic v2 runs `mode="after"` validators in class-body declaration order, and
`auto_setup_weight_broadcast` is defined **before** `auto_setup_deployment` — so the NCCL world
size is computed from `dp`'s *pre-propagation* value (its default) before `dp` is ever set to the
real GPU count. With `dp` left implicit, the broadcast group is built for 1 rank against an
N-rank inference server, and `/resume`'s collective blocks forever.

**Directly confirmed with hard evidence, not just source-reading**: every affected job's own
*resolved* config shows the smoking gun — `inference.parallel.dp` is correct, but
`inference_world_size` is stuck at `1`:

| Job | `inference.toml` `dp` | `trainer.toml` / `orchestrator.toml` `inference_world_size` |
|---|---|---|
| 42512/42513/42514 | 2 | 1 |
| 42360 (hendrycks-sanity v2) | 4 | 1 |

Fix: set `[inference.parallel] dp` explicitly in the config so it's never left to the
validator-ordering race. **This retroactively root-causes job 42360** (previously recorded in
this log's Results table as "unresolved, separate issue" — corrected above per this finding) —
every setup that worked before this point used single-GPU inference (`dp` trivially resolves to 1
either way), which is why the bug never surfaced earlier. Note: 42357 (the other
`hendrycks-sanity` job, same table row group) is unaffected — its failure was the unrelated
`flash_attn` venv incident already correctly attributed to Finding 13.

## Open questions

1. **Root cause of the bogus `Grad. Norm` value is unidentified (Finding 12).** `clip_grad_norm_`
   (`torchtitan.distributed.utils`, called at `src/prime_rl/trainer/sft/train.py:471` and
   `src/prime_rl/trainer/rl/train.py:558`) reports 3.4e8–4.2e10 for a 0.6B model where O(1–10)
   would be expected, and swings ~100x step to step even with clipping disabled — so it is
   specifically the *reported norm*, not the underlying gradients, that is wrong. **Raising/
   disabling `max_norm` is a workaround, not a fix**: the underlying bug in the norm computation
   (prime suspect: FSDP/DTensor norm reduction on ROCm) still needs root-causing via a
   per-parameter gradient-norm dump. Until that's done, every future run on this cluster needs
   `max_norm` set high enough to be a no-op, or clipping will silently reintroduce this failure
   mode.
2. **Top-k / exact reverse-KL estimator (Finding 9) — RESOLVED: implemented and verified, but not
   the limiting factor.** `teacher_top_k` now exists and was fixed after an initial bug (partial
   sum vs. true KL, Finding 14); the fixed estimator behaves exactly as designed in production
   (`Ref KL` down, `TopK Mass` up, Finding 15). It does **not**, however, rescue OPD from a
   base-model student at any k tried (0/8/20/50/100) — reward and truncation collapse identically
   regardless of support size. The original concern (sampled-token estimator can only learn to
   avoid tokens the policy already emitted) does not appear to be the binding constraint here; see
   Open Question 5 for the current leading explanation.
3. **Should `reverse-text` be abandoned for OPD validation?** Given Finding 1, the SFT-tuned
   student/teacher pair has almost no headroom at 512 tokens (~0.72–0.75 vs 0.799) — but that
   result was measured entirely under the broken-clipping regime (Finding 12), so "no headroom"
   and "clipping prevented learning" are two overlapping, currently indistinguishable explanations
   for those flat curves; the 512-token arms (42283/42284/42285) should be re-run with
   `max_norm` fixed if this conclusion needs to be load-bearing. Separately, jobs 42297/42298/
   42299 (base, untuned `PrimeIntellect/Qwen3-0.6B` student — real headroom by construction) were
   run to test this and came back flat too (reward ~0.04–0.06 throughout), but are now understood
   to be further victims of the same `max_norm` bug rather than evidence about headroom. **Now
   superseded by 42366 (sftfix) / 42367 (opdfix) / 42368 (grpofix)** — the base-model student at
   `max_norm = 1e12`, currently in flight — which are the first runs able to actually answer the
   original OPD-viability question.
4. **Trainer/orchestrator shutdown deadlock root cause (Finding 11) — partially resolved.** The
   *symptom* (SLURM job stuck `RUNNING` after orchestrator exit, trainer short of `max_steps`) is
   confirmed for job 42285 and has an operational workaround (check `sacct` + trainer's last step,
   don't trust a completed orchestrator log alone). The *mechanism* is still unconfirmed: the
   `zero_advantage`-filter-starves-the-trainer hypothesis in Finding 11 has not been checked
   against 42285's actual batch-dispatch logs, and this has so far been observed in exactly one
   run. Needs: (a) confirmation/refutation via the orchestrator's batch-filter counters for 42285,
   and (b) watching whether the same stall recurs on the 42366/42367/42368 runs or any future
   near-ceiling / high-`max_steps` run. **Update**: now observed 7 times total, including three
   more instances in the `teacher_top_k` sweep (42380, 42381, 42387) — all base-model-student OPD
   runs at `max_norm = 1e12`, none near a performance ceiling (rewards were near-zero throughout),
   which weakens the "near-ceiling zero-advantage-filter-starves-the-trainer" hypothesis as the
   sole mechanism — a near-ceiling GRPO run and a near-zero-reward base-model OPD run are not
   obviously the same regime, yet both hang. Root cause is still unconfirmed. **Update
   (2026-08-13): now 12 confirmed instances** (this audit independently verified 42528, 42552,
   42563, 42564, 42566 — all `CANCELLED+` with the trainer stalled at step 97/100 — added to the
   previously-established 7; the coordinator's claimed total of 13 could not be reconciled from
   the jobs named in this batch, see discrepancy note 21). The census now spans reverse-text and
   GSM8K, both `opd` and `grpo`, base and near-teacher-level students, and both the buggy and
   fixed `max_norm`/lr regimes — strengthening the case that this is a generic trainer/orchestrator
   teardown bug, not something specific to any one algorithm, task, or student capability level.
   It also costs data, not just node-hours: three of the four lr-corrected GSM8K arms
   (42563/42564/42566) never wrote their step_100 checkpoint (confirmed, discrepancy note 22).
5. **Does OPD work from a near-teacher-level (e.g. SFT-warmed-up) student, or is on-policy reverse
   KL fundamentally unstable far from the target distribution? — RESOLVED (partially) by the
   GSM8K result.** Findings 14/15 showed OPD failing from a *base* (untrained) student on
   reverse-text at every k tried. Finding 16's GSM8K result now shows OPD (and GRPO) working from
   a **near-teacher-level** student (`Qwen/Qwen3-1.7B` on-band pass rate 0.588 vs. teacher 0.893)
   once the learning rate is in the repo-standard 1e-6-3e-6 range: all four corrected-lr arms
   improve monotonically, closing 11-26% of the gap. This is consistent with the mode-seeking
   hypothesis in Finding 15 — reverse KL is viable near the target distribution, not far from it —
   but does not yet test whether a **base**-model student would succeed on GSM8K under the
   corrected lr (that specific cell of the 2x2 has not been run). The `teacher_top_k` question
   (widening k beyond 0) is now being tested in a regime where OPD demonstrably works (see the
   in-flight 42611-42618 sweep, Open Question 6).
6. **Is opd's apparent edge over grpo at step 75-100 (GSM8K, Finding 16) real, or noise? — open,
   awaiting the in-flight seed sweep.** At matched step 75, the four lr-corrected arms score
   0.622-0.643 — well inside one conservative per-problem standard error (~0.027 on 327 problems)
   of each other. opd-lr3e-6's headline 0.6674 is at step 100, a checkpoint only it wrote (the
   other three deadlocked first, Finding 11/Open Question 4), so part of its apparent lead is
   simply having trained longer, not a per-step advantage. Jobs 42611-42618 (running, do not
   touch) directly target this: opd/grpo x seeds {0,1,2} for variance, plus `teacher_top_k` in
   {20, 100} against the k=0 seed-0 arm, all at 300 steps / lr 3e-6 / checkpointed every 50 steps
   — this should settle both the seed-variance question and whether the top-k estimator (Finding
   14's fix, previously untested in a regime where OPD works at all) helps here.

## Environment & how to reproduce

- **Login node has no GPUs.** All GPU work goes through `sbatch`, partition `amd-tw-verification`.
- **Container:** `/shared_silo/scratch/containers/primus_v26.2_moefix.sif`, run via
  `singularity exec --rocm --cleanenv --containall` with explicit bind mounts (see
  `opd_val/rl_opd.sbatch:78-91`):
  - `/shared_silo/scratch:/shared_silo/scratch:rw`
  - `$HOME:$HOME:ro`
  - ROCm/RDMA library shims (`libbnxt_re-rdmav34.so`, `libdrm`)
  - `/tmp:/tmp:rw`
  - `$SCRATCH/vf-scripts:/tmp/vf-scripts:rw` (the Finding 4 workaround)
- **Python:** the repo's own `.venv`, selected via `UV_PYTHON=<repo>/.venv/bin/python3` and
  `UV_NO_SYNC=1` (never `uv sync` inside the job — the venv is pre-built).
- **Caches:** because `--cleanenv --containall` strips `SLURM_*` inside the container, all
  per-job scratch/cache paths are baked in on the **host** side using `$SLURM_JOB_ID` before
  entering the container (`SCRATCH=/tmp/rahul_opd_${SLURM_JOB_ID}`), covering
  `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `HOME`. This is
  what job 42206 got wrong (job id resolved empty *inside* the container).
- **Models/datasets:** `HF_HOME=/shared_silo/scratch/hf_cache`. The login node has internet, so
  models/datasets are pre-downloaded there; job scripts run with `HF_HUB_OFFLINE=1`.
- **W&B:** no `WANDB_API_KEY` configured on this cluster → `WANDB_MODE=disabled`; all metrics in
  this log come from the text logs, not a dashboard.
- **Config validation without a GPU:** `uv run rl @ cfg.toml --dry-run` fully validates a config
  and writes split `orchestrator.toml` / `trainer.toml` (see `opd_val/dryrun/configs/`,
  `opd_val/dryrun2/configs/`) — always do this on the login node before submitting an `sbatch` job.
- **GPU layout** (`opd_val/rl_opd.sbatch`): GPU 0 = policy inference server (`:8000`, launched by
  the `rl` entrypoint), GPU 1 = trainer (launched by the `rl` entrypoint), GPU 2 = frozen teacher
  server (`:8001`, launched separately by the sbatch script itself — prime-rl never launches
  frozen reference models on its own).
- **Required config knobs for ROCm** (present in every `opd_val/*.toml` since 42216):
  `[trainer.model] compile = "None"`, and `HIP_VISIBLE_DEVICES` left unset for the `rl`
  entrypoint's environment (set only for the standalone teacher process).
- **Required config knob for `teacher_top_k` > 0**: `[trainer.model] fused_lm_head_token_chunk_size
  = "disabled"` — the fused head never materializes logits, which the top-k branch of
  `ref_kl_loss_fn` needs to gather the policy's own logprobs on the teacher's support. Also note
  the teacher server's `--model.max-logprobs` launch flag (`opd_val/rl_opd.sbatch:113`, env var
  `TEACHER_MAX_LOGPROBS`, default 128) is a *separate* setting from the student's own
  `[model.max_logprobs]` in `inference.toml` (always 20 in these runs) — raising `teacher_top_k`
  above 19 requires raising the former, not the latter (see Finding 14 and discrepancy note 16).
- **Key files:**
  - Configs: `opd_val/opd_reverse_text.toml`, `opd_val/opd_long.toml`, `opd_val/grpo_control.toml`,
    `opd_val/grpo_long.toml`, `opd_val/opd_512.toml`, `opd_val/grpo_512.toml`,
    `opd_val/sft_512.toml`, `opd_val/base_512.toml`, `opd_val/{sft,opd,grpo}_base_fixed.toml`
    (the `max_norm = 1e12` fixed base-model configs used by 42366/42367/42368). Config generation
    is now handled by `opd_val/make_configs.py`, which replaced `make_512_configs.py` and emits
    both the `*_512` family (reverse-text-tuned student, e.g. 42283/42284/42285) and the `*_base`
    family (base `Qwen/Qwen3-0.6B`-class student, e.g. 42297/42298/42299 and their fixed
    re-runs 42366/42367/42368). The `teacher_top_k` sweep uses `opd_val/opd_k{0,8,20,50,100}.toml`
    and `opd_val/opd_k20_lr5e6.toml` (jobs 42380-42391), all `*_base`-family (base-model student)
    configs with `max_norm = 1e12` and `fused_lm_head_token_chunk_size = "disabled"` baked in.
  - Launchers: `opd_val/rl_opd.sbatch` (main training runs), `opd_val/smoke_container.sbatch`,
    `opd_val/smoke_prefill.sbatch`, `opd_val/compare_teacher_student.sbatch`.
  - Comparison script: `opd_val/compare.py` (behavioural + per-token `ref_kl` measurement).
  - Per-run artifacts: `opd_val/runs/<name>_<jobid>/run/logs/{orchestrator,trainer,inference}.log`,
    `opd_val/runs/<name>_<jobid>/teacher.log`, raw torchrun tracebacks under
    `opd_val/runs/<name>_<jobid>/run/logs/trainer/torchrun/<id>/attempt_0/0/stdout.log`, rollout
    traces under `opd_val/runs/<name>_<jobid>/run/run_default/rollouts/step_N/{train,eval}/{all,effective}/traces.jsonl`.
  - Raw sbatch stdout/stderr and standalone comparison logs: `opd_val/logs/`.
  - **GSM8K (Phase 3) key files**: `opd_val/gsm8k_band.py` is the offline band-selection AND
    checkpoint-eval driver — `--dataset local` scores a trained checkpoint against the saved
    327-problem band (used by job 42598); the same script (different invocation) also performed
    the band-selection pass (42486) and the ceiling/ladder evals (42463, 42470, 42475, 42498).
    Selected band data lives at `data/gsm8k_boxed_band/train.parquet` (327 problems, written by
    42486). Student/teacher configs use `Qwen/Qwen3-1.7B` / `Qwen/Qwen3-8B` with
    `enable_thinking = false`; because the orchestrator's eval client ignores this setting
    (Finding 17), in-run eval is not used for these runs — checkpoints are written every 25 steps
    and scored offline instead. `docs/algorithms.md` has a dedicated "Estimating the Reverse KL"
    section (linked from its table of contents) documenting the `teacher_top_k` estimator: the
    score-function default vs. the top-k-support KL, the off-support atom, the
    `fused_lm_head_token_chunk_size = "disabled"` and teacher `max_logprobs` requirements, and the
    `TopK Mass` diagnostic — confirmed present and consistent with Finding 14's mechanics.

### How to check a finished run

A completed-looking orchestrator log is **not sufficient** evidence that a run finished — job
42285 (Finding 11) had a clean orchestrator exit while the trainer and the SLURM job itself were
both still hung. Before recording a run's numbers as final, check all four of the following:

1. **`sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode`** — the job must show a terminal
   state (`COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`), not `RUNNING` or `PENDING`. This is the
   ground truth; log text is not.
2. **Compare the trainer's last logged step to `max_steps`.** `grep -oE "Step [0-9]+" run/logs/trainer.log | tail -1`
   vs. the `max_steps` value in `run/configs/trainer.toml` (or the orchestrator config). A mismatch
   (e.g. 57 vs 60, as in 42285) is the specific tell for the shutdown deadlock in Finding 11, even
   if the orchestrator log claims completion.
3. **`grep -c HarnessError run/logs/orchestrator.log`** — a nonzero (especially large) count means
   the harness/verifiers subprocess is failing on every rollout (see Finding 4's `/tmp/vf-scripts`
   bug); a run can appear to be "iterating" in `sacct` while producing zero real training signal.
4. **Check log-file mtimes** (`stat -c '%Y %n' run/logs/*.log` or `ls -la --time-style=full-iso`)
   against the current time. If all three logs (`trainer.log`, `orchestrator.log`,
   `inference.log`) haven't been touched in several minutes but the SLURM job is still `RUNNING`,
   that's a live stall, not just log-flush lag — worth a fresh look before assuming the run is
   still making progress.

## Changelog

- **2026-08-12** — Initial canonical write-up of the OPD validation effort. Covers jobs 42206,
  42207, 42209, 42211, 42216, 42224, 42226, 42227, 42276, 42277, 42283, 42284, and (partially,
  pending resolution of an operational anomaly) 42285. Headline finding: the student/teacher gap
  on `reverse-text` was a `max_completion_tokens` artifact, not a real capability gap; at 512
  tokens all three algorithm arms are flat near the student's ceiling. Cross-checked every number
  in the source summary against the actual run logs; found and documented eight discrepancies
  (see "Verification discrepancies" above) — all logs-vs-summary, none logs-vs-logs. OPD
  mechanics (teacher prefill scoring, `ref_kl` loss, RCCL broadcast) confirmed working on
  ROCm/vLLM 0.25.1. Two ROCm-specific workarounds documented (`torch.compile` off,
  `HIP_VISIBLE_DEVICES` unset for `rl`). One upstream bug reported (hardcoded `/tmp/vf-scripts`
  path in `verifiers`). Four open questions logged, including a newly-discovered possible hang in
  job 42285 that needs a follow-up `sacct` check.
- **2026-08-12 (update)** — Coordinator confirmed all eight discrepancies from the initial pass
  and corrected their own reporting to match. Job 42285 anomaly resolved: confirmed as a
  trainer/orchestrator shutdown deadlock (new Finding 11), cancelled via `scancel 42285` by the
  coordinator to free the node; its complete 13-point eval curve added to the Results table,
  reinforcing that all three 512-token arms are flat (no headroom, not an algorithm failure).
  Added a "How to check a finished run" subsection (`sacct` state, trainer step vs. `max_steps`,
  `HarnessError` count, log mtimes) to Environment & how to reproduce. Added placeholder "in
  flight" rows for jobs 42297 (opd), 42298 (sft), 42299 (grpo) — a base-model-student run
  targeting Open Question 3 — not touched by this agent, results pending from the coordinator.
  Noted the config generator is now `opd_val/make_configs.py` (replaces `make_512_configs.py`),
  emitting both `*_512` and `*_base` config families.
- **2026-08-12 (major update) — ROOT CAUSE FOUND.** New Finding 12: the default
  `[trainer.optim] max_norm = 1.0` prevents learning entirely on this ROCm cluster, invalidating
  every opd/grpo/sft result recorded before this point in the log — the flat curves throughout
  this document were caused by broken gradient clipping, not by task headroom or algorithm choice.
  Isolated and fully re-verified against raw `trainer.log` files with the standalone `sft`
  entrypoint (no orchestrator/vLLM/rollouts involved): job 42364 (`max_norm = 1e12`) shows CE loss
  descending 4.6125→3.7183 over 100 steps, vs. jobs 42356/42359 (`max_norm = 1.0`, bit-identical to
  each other) flat at 4.6125→4.8955; `Grad. Norm` itself stays wrong (3.4e8–4.2e10, ~100x swings)
  even with clipping disabled, isolating the bug to the norm computation, not the gradients. Ruled
  out `optim_cpu_offload` as a contributing factor (job 42362, bit-identical curves with it off).
  Jobs 42297/42298/42299 (previously "in flight" placeholders) are now confirmed complete and
  reclassified as invalidated by this same bug (flat reward ~0.04–0.06 despite a base, untuned
  student with real headroom) — superseded by new in-flight jobs 42366 (sftfix) / 42367 (opdfix) /
  42368 (grpofix), the first runs able to actually answer the original OPD-viability question.
  New Finding 13 documents an unrelated environment incident (a `uv sync` invocation without
  `--all-extras` stripped `flash-attn` from the venv, breaking jobs 42356's second arm, 42357, and
  motivating jobs 42359/42362/42364 as clean re-isolations) — `pyproject.toml`/`uv.lock` were not
  modified. Rewrote the TL;DR to lead with the headline finding, corrected one minor numeric
  discrepancy (42364's rounded "3.76" endpoint vs. the logged 3.7183), and updated all four Open
  Questions to reflect the new root cause and its downstream effect on the reverse-text-headroom
  and gradient-norm questions. No jobs were submitted, cancelled, or otherwise touched by this
  agent during this update.
- **2026-08-13 — `teacher_top_k` implemented, found buggy, fixed, and swept (Findings 14, 15).**
  New capability: the teacher can ship its per-token top-k support so the reverse-KL loss can put
  gradient on tokens the teacher prefers that the policy never sampled (`OPDAlgoConfig.
  teacher_top_k`, `ref_kl_loss_fn`'s top-k branch, `prefill_logprobs_topk`), resolving Open
  Question 2 in principle. The first implementation had a genuine bug: a partial sum over a fixed
  support is not a valid KL (degenerate minimum at evacuating the support), independently
  reconfirmed here from raw logs — the pre-fix top-k `Ref KL` metric goes negative for both k=8
  (42381, min -0.0758) and k=20 (42382, min -0.0841), which is impossible for a true KL. Fixed by
  coarsening off-support mass into one extra category per distribution (a genuine, non-negative
  `(k+1)`-category KL); confirmed fixed — every post-fix top-k `Ref KL` across jobs 42387-42391 is
  positive. Also root-caused and explained (not just accepted) the `model.max_logprobs = 20` vs.
  `teacher_top_k = 50/100` puzzle in the post-fix configs: that field belongs to the student's own
  rollout server, while the frozen teacher's `--model.max-logprobs` is set separately via
  `rl_opd.sbatch`'s `TEACHER_MAX_LOGPROBS` env var (128 in the post-fix sweep) — this, not the
  student config, is what actually fixed the 42383/42384 `BadRequestError` failures. Post-fix
  sweep result: the fixed estimator works exactly as designed (KL down, top-k mass up for every
  k in {8, 20, 50, 100}) but does not rescue OPD from a base-model student — every arm still ends
  at 0.0000 eval reward and 100% truncation, at every k tried and both learning rates tried.
  Verified all ten new Results-table rows (42380-42391, two of them FAILED pre-fix jobs) against
  raw `trainer.log`/`orchestrator.log`/`sacct`, and found six numeric discrepancies against the
  coordinator's brief (see items 10-16 in Verification discrepancies) — five were intermediate
  values quoted instead of literal final-step endpoints (the same recurring pattern as in the
  original pass), one (42390) was an immaterial ~0.03 rounding difference, and the best-ever-reward
  figures for the full post-fix sweep matched exactly. Updated Open Question 2 to "resolved,
  implemented, but not the limiting factor," added new Open Question 5 (does OPD work from a
  near-teacher-level/SFT-warmed-up student instead of a base model — the current leading
  explanation being that on-policy reverse KL is mode-seeking and unstable far from the target
  distribution), and updated Open Question 4 with three more observed shutdown-deadlock instances
  (42380, 42381, 42387 — 5th/6th/7th occurrences), which weakens the near-ceiling-specific
  mechanism hypothesized in Finding 11. No jobs were submitted, cancelled, or otherwise touched by
  this agent during this update.
- **2026-08-13 (major update) — OPD WORKS, on GSM8K (Findings 16, 17, 18).** `reverse-text` is
  abandoned as the validation task (no usable intermediate-student checkpoint, per 42463) in favor
  of GSM8K with a new student/teacher pair (`Qwen/Qwen3-1.7B` / `Qwen/Qwen3-8B`) and a
  `[0.2, 0.8]`-pass-rate "band" methodology (327 problems, `opd_val/gsm8k_band.py`). New headline:
  measured offline on the full band (job 42598), all four lr-corrected arms (opd/grpo x
  {1e-6, 3e-6}) improve monotonically over baseline (student 0.5879, teacher ceiling 0.8933),
  closing 11-26% of the gap — but the between-arm (opd vs. grpo) differences are inside one
  conservative standard error and must not be read as opd beating grpo. New Finding 16: the
  earlier GSM8K collapse (jobs 42526-42528, 42550-42552) was a learning-rate misconfiguration
  (2e-5, borrowed from the SFT isolation sweep) rather than an algorithm failure — confirmed via
  `trainer.toml` and each run's `Mismatch KL`/`Reward` series. New Finding 17: the orchestrator
  builds eval clients as `"openai_chat_completions"` but train clients as `"renderer"`
  (`src/prime_rl/orchestrator/utils.py:45-46`), so any generation-affecting config (like
  `enable_thinking = false`) silently applies to training only — confirmed via job 42550's
  `<think>`-tag trace counts (128/128->18/128 by step 100) and the step-1 eval-vs-train reward gap
  (0.0391 @ 96.1% truncation vs. 0.9141 @ 0%). Workaround: offline checkpoint eval, used throughout
  this batch. New Finding 18: a `pydantic` validator-ordering bug in `packages/prime-rl-configs/
  src/prime_rl/configs/rl.py` (`auto_setup_weight_broadcast`, which computes
  `inference_world_size`, runs before `auto_setup_deployment`, which sets `inference.parallel.dp`
  from `deployment.num_infer_gpus`) hangs `/resume` on any multi-GPU inference server with `dp`
  left implicit — confirmed directly via 42512/42513/42514's and **42360's** resolved configs,
  each showing the smoking-gun `inference_world_size = 1` despite a correct `dp` of 2 or 4.
  **Corrected job 42360's Results-table row**, previously recorded as "unresolved, separate
  issue," to reference this root cause. Updated the shutdown-deadlock census (Open Question 4) to
  12 independently-confirmed instances (this audit verified 42528, 42552, 42563, 42564, 42566 —
  all `CANCELLED+` at step 97/100 — against the previously-established 7); could not reconcile the
  coordinator's claimed total of 13 from the jobs named in this batch and flagged the discrepancy
  rather than silently matching it. Verified every new numeric claim against raw logs
  (`sacct`, `trainer.log`, `orchestrator.log`, rollout traces, resolved `.toml` configs, and
  `opd_val/logs/*.out`); found six new discrepancies (items 17-22 above) — one (17) a full exact
  match across an entire headline table, three (18-20) the recurring "intermediate value quoted as
  endpoint" / "two jobs conflated into one narrative" pattern seen throughout this log, one (21) an
  unreconciled 12-vs-13 count, one (22) a full exact match. Resolved Open Question 5 (does OPD work
  from a near-teacher-level student) in the affirmative for GSM8K; added new Open Question 6 (is
  opd's apparent edge over grpo real, or noise — the in-flight seed/top-k sweep, jobs 42611-42618,
  targets this directly). Confirmed via read-only `sacct` that all 8 of jobs 42611-42618 are in a
  non-terminal state (`RUNNING` or `PENDING`). **No jobs were submitted, cancelled, or otherwise
  touched by this agent during this update — in particular, none of the 8 running/pending jobs
  42611-42618 were touched.**

---

## 2026-08-18 — Gradient-norm probe: the ROCm flash-attn backward is broken (jobs 43536–43540)

**Question.** Is the trainer's reported `Grad. Norm` of ~1e8 wrong (a `clip_grad_norm_` bug,
clipping restorable), or are the gradients genuinely that large (a loss-scaling problem)?
The whole investigation had been running with `max_norm = 1e12` on the first assumption.

**Method.** `opd_val/gradnorm_probe.py` wraps the standalone `sft` entrypoint and
monkeypatches `torchtitan.distributed.utils.clip_grad_norm_` to compute the global L2 norm
independently before clipping, with a per-parameter breakdown. Run on **one GPU** so there is
no sharding: a manual sum over `.to_local()` grads is then exact, and a disagreement would
exonerate the FSDP/DTensor reduction. Config `examples/basic/reverse-text/sft.toml`,
Qwen3-0.6B, lr 2e-5, `compile = "None"`.

**Job history.** 43536 failed (`RANK expected, but not set` — `uv run sft` launches torchrun
internally; invoking the module with plain `python` skips it). 43537 fixed that with
`torchrun --standalone --nproc-per-node 1`. 43539 added full parameter names, shapes and
per-element RMS. 43540 added the SDPA arm.

### Result 1 — the norm is correct (43537)

`ratio = reported/manual = 1.0000` at **every** step, at both `max_norm = 1.0` and `1e12`,
matching to 7 significant figures. `all_finite = True`, `n_params = 310`.
**`clip_grad_norm_` was never at fault.** The gradients really are ~1e8–3.6e8.

### Result 2 — the damage is localised to layer 0 and the Q/K path (43539)

One parameter carries 42–60% of the squared norm, consistently:

```
model.embed_tokens.weight              (151936,1024)  norm 2.38e8  rms 1.9e4
model.layers.0...self_attn.q_proj      (2048,1024)    norm 1.88e8  rms 1.3e5
model.layers.0...self_attn.k_proj      (1024,1024)    norm 1.21e8  rms 1.2e5
model.layers.0...self_attn.o_proj      (1024,2048)    norm 4.38e7  rms 3.0e4
model.layers.0...self_attn.q_norm      (128,)         norm 3.68e7  rms 3.3e6
```

All layer 0 + embeddings, ordered q_proj > k_proj > v_proj > o_proj. `q_norm` has 128
elements carrying a per-element gradient RMS of 3.3 million.

### Result 3 — swapping the attention kernel fixes it (43540)

Same data, same seed, only `FlashAttention._compute_attention` replaced by a per-sequence
`F.scaled_dot_product_attention` under `SDPBackend.MATH`:

| backend | step-1 loss | grad norm 1–3 | embed grad RMS | top-1 share | tok/s |
|---|---|---|---|---|---|
| `flash_attn_varlen_func` | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | 1.9e4 | ~50% | 5940 |
| MATH SDPA | 4.6122 | 24.4 / 12.0 / 6.7 | 2.1e-4 | ~24% | 11196 |

**The loss agrees to four significant figures — the forward is exact — while the gradients
differ by ~1.4e7.** A backward-only kernel bug. Under SDPA the per-parameter distribution is
diffuse, `q_norm` leaves the top five, and the SDPA arm's loss also falls faster over three
steps (4.6122 → 4.2364 vs 4.6125 → 4.4489), consistent with correct gradients.

Environment: `flash-attn 2.8.3.post1`, `torch 2.11.0+rocm7.2`, `triton 3.6.0`; only FA2
installed. MATH SDPA was 2× faster at equal peak memory, but is O(L²) memory and will not
scale.

### Why `max_norm = 1.0` looked like it "prevented all learning"

At a true norm of 6–24, `max_norm = 1.0` is a sensible clip; prime-rl's default was never
wrong. The failure is a second-order effect of the first bug: the inflated layer-0 gradients
carry ~50% of the norm, so the *global* rescale by 1/3.4e8 crushes the ~305 healthy
parameters to ~1e-8, where AdamW's `eps = 1e-8` floor swamps them and their updates go to
zero. Unclipped, AdamW's per-parameter normalisation rescues the healthy majority while
layer 0 keeps training on garbage. **`max_norm = 1e12` worked by accident.**

### Result 4 — it is ONE ENVIRONMENT VARIABLE (43543)

Prompted by the user noting that colleagues run prime-rl + flash-attn here without trouble.
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` selects flash-attn's Triton AMD backend over
composable-kernel. It originates at `rl_train_singlenode_mixed.sbatch:308` and I copied it
into all twelve `opd_val` scripts, so **every experiment in this investigation used it**.

| backend | TRITON_AMD | act.ckpt | step-1 loss | grad norm 1-3 | tok/s |
|---|---|---|---|---|---|
| flash-attn varlen | TRUE | on | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | 5869 |
| flash-attn varlen | **FALSE** | on | 4.6127 | **24.39 / 11.98 / 6.66** | **17550** |
| flash-attn varlen | TRUE | off | 4.6125 | 3.40e8 / 1.46e8 / 1.37e8 | 15284 |
| MATH SDPA | - | on | 4.6122 | 24.36 / 11.98 / 6.65 | 11069 |

Unset, flash-attn matches the independent SDPA reference to 3 s.f. on the norm and 4 s.f. on
the loss, at 3x the Triton throughput. The distinct throughput (17550 vs 11069 vs 5869)
confirms a real third kernel ran rather than a silent fallback. **Activation checkpointing is
exonerated** - that cell is bit-identical to the control.

Fix applied: all 13 scripts (12 in `opd_val/` plus `rl_train_singlenode_mixed.sbatch`) now
carry a comment in place of the export explaining why not to set it.

### Result 5 — confirmation at the DEFAULT max_norm (43544)

Plain `uv run sft` on `examples/basic/reverse-text/sft.toml`, lr 2e-5, 100 steps, **no
`max-norm` override** — the exact configuration that produced a flat 4.6125 -> 4.8955 in
jobs 42356/42359, now with the variable removed:

| step | 1 | 5 | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|
| loss | 4.6127 | 4.0529 | 3.4887 | 2.7808 | 1.6166 | 1.1225 | **0.9469** |
| grad norm | 24.39 | 5.23 | 2.46 | 3.53 | 5.32 | 6.62 | 4.16 |

The norm sits in 2-25 for the whole run, so `max_norm = 1.0` is clipping mildly and sanely,
exactly as designed.

**The old "working" workaround was also badly degraded.** Job 42364 (`max_norm = 1e12`, same
config, same 100 steps) reached only **3.7183**. With the variable removed and clipping left
at its default, the same run reaches **0.9469** — a ~4x better final loss. So the Triton
backward was not merely breaking clipped runs; it was crippling the unclipped ones too. Every
prior result is degraded, not just uncertain.

### Consequences
- **Every experiment in this log trained layer 0 and the embeddings on corrupted gradients**,
  including GRPO 0.7256 vs `opd` 0.6715 ± 0.0019. Both arms shared the defect so the ranking
  is not automatically void, but none of it should be read as a claim about the *algorithms*
  until reproduced on a correct backward.
- Fix paths, neither attempted: (a) a flash-attn build with a correct ROCm backward;
  (b) add `sdpa` to prime-rl as a correctness fallback.

### Also closed out — clipping does not rescue top-k (43534/43535, offline scores 43541)

`teacher_top_k = 20` with `max_norm` 5e7 / 1e7, 300 steps, GSM8K band. Both ran clean
(rc=0, 2h15). Offline scores (43541):

| arm | step_50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| k=20, `max_norm` 5e7 | 0.5894 | 0.0237 | 0.0115 | 0.0084 | 0.0115 | 0.0107 |
| k=20, `max_norm` 1e7 | 0.5872 | 0.0183 | 0.0084 | 0.0084 | 0.0115 | 0.0107 |

Both are at the base student's 0.588 at step 50 (2.4–2.7% truncation) and collapse to ~0.01
at 100% truncation by step 150 — same window and same endpoint as the unclipped arms.
Constraining the update norm does not rescue top-k. That closes the sweep at **seven
variants, all collapsing**: residual/renormalize/none × reverse/mixed/forward, plus these two
clipped arms.

Note this negative result is *entangled* with the flash-attn finding above: with the backward
broken, a top-k objective that puts exact gradient on the Q/K path is being fed corrupted
gradients precisely where the damage is worst. Top-k should be re-tested on a correct
backward before the collapse is attributed to the objective.

**A misread worth recording:** I flagged these arms' *in-run* step-1 eval (reward 0.02 at
95% truncation, against a 0.588 student) as evidence the top-k path breaks generation
immediately. It is not — that is the known-broken in-run eval, which reads ~0.05 for **every**
arm including the GRPO runs that offline-score 0.726, because it uses
`openai_chat_completions` rather than the renderer so `enable_thinking=false` never applies.
The offline eval is clean and shows the top-k collapse is real (e.g. `M-k20-none-forward`:
step_50 0.5719 at 2.6% truncation → step_300 0.0099 at 100%).
