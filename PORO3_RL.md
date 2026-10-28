# Poro3 35B-A3B RL runbook

Two launchers, run in order. Both need exactly 8 nodes, take no flags, and take at most
one optional argument — the policy checkpoint.

Both are self-contained: each is the full recipe, top to bottom, and reading it tells you
everything that run does. Neither reads an environment variable as a knob, and neither
delegates to another script — to change the recipe, edit the recipe. The generic launcher
with all the flags is `rl_train_multinode_mixed.sbatch`; use that for experiments.

## 0. One-time

```bash
mkdir -p logs
```

**Submit from your clone root.** The launchers take their own path from
`SLURM_SUBMIT_DIR`, so there is nothing to edit for your clone — but they will refuse to
start if you submit from anywhere else.

Then set `POLICY_MODEL` in **`rl_train_poro3_35b_repetition.sbatch`** to the Poro3 35B SFT
checkpoint. (Or pass it as the first argument — but editing it is the intent.)

## 1. Repetition pre-stage — ~10 h

Removes reasoning loops before the real run. 100 steps, EN+FI Dolci math prompts, scored
only on the absence of verbatim repetition; correctness is not in the reward. On the 4B
model this cut eval truncation from 13-29% to 0-13% and cost nothing in accuracy. Worth
doing first because truncated rollouts fabricate advantage during RL, not just lost eval
points.

```bash
sbatch rl_train_poro3_35b_repetition.sbatch
```

Checkpoints every 25 steps in `outputs/<run>/weights/step_N`. **Evaluate step_100 before
continuing** — nothing in the loss anchors the model to the SFT init, so verify AIME and
IFEval have not moved. If step_100 looks worse than step_50 or step_75, use that instead.

## 2. Full RL — up to ~6 days

Six envs (math, IFEval, rubric-judge × English/Finnish), 2000 steps max.

Set `POLICY_MODEL` in `rl_train_poro3_35b.sbatch` to the pre-stage output, then:

```bash
sbatch rl_train_poro3_35b.sbatch
# or: sbatch rl_train_poro3_35b.sbatch outputs/<rep-run>/weights/step_100
```

Checkpoints every 100 steps. **2000 is a ceiling, not a target** — stop the run when the
eval curve flattens. Eval `weights/step_N` as they land.

## Monitoring

Orchestrator and trainer both log to the Slurm job file in your clone root — that is the
one to watch:

```bash
squeue --me
RUN=outputs/$(ls -t outputs | head -1)        # the run you just started

tail -f logs/rl_<jobid>.out                   # reward, truncation, step time, entropy, MFU
tail -f $RUN/inference_node_0_0.log           # vLLM, only if servers fail to start
tail -f $RUN/judge_inference.log              # full run only, if the judge is the holdup
```

Healthy pre-stage: `reward/repetition-*/mean` rising **and** `is_truncated/all/mean`
falling together, with `seq_len/all/mean` roughly stable. Reward rising while truncation
does not fall means the model is gaming the exact-block metric — stop it.

Startup takes ~15 min — every vLLM server (56 in the pre-stage, 48 plus the judge in the
full run) must pass a health check before anything starts. First step is slow; expect
~4-5 min/step after that, in both runs: the 8-GPU trainer sets the pace and the batch is
the same.

## If a node fails

Nothing to do. Both launchers set `--requeue` and resume in place, so Slurm restarts the
job in the same output directory and it picks up the latest checkpoint — losing at most
one checkpoint interval. Data order is not restored, which is fine.

`scancel <jobid>` to stop a run; its checkpoints stay on disk and are usable.

## Disk

A single save point is **456 GB**. The pre-stage writes 4 (1.8 TB), the full run up to 20
(9.1 TB). Check `df -h /shared_silo` before starting — it has been near 94% full.
