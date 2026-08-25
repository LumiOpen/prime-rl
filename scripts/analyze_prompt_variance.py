"""Analyze per-prompt reward variance from orchestrator trace files.

Usage:
    uv run python scripts/analyze_prompt_variance.py <run_output_dir> [--steps N]

Example:
    uv run python scripts/analyze_prompt_variance.py \
        outputs/normal-olmo3-7b-think-sft-ifeval-128subset-32k-20260728_131728 \
        --steps 10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_traces(run_dir: Path, steps: int | None) -> list[dict]:
    rollout_dir = run_dir / "run_default" / "rollouts"
    if not rollout_dir.exists():
        raise FileNotFoundError(f"No rollouts dir at {rollout_dir}")

    step_dirs = sorted(rollout_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))
    if steps is not None:
        step_dirs = step_dirs[:steps]

    records = []
    for step_dir in step_dirs:
        trace_file = step_dir / "train" / "all" / "traces.jsonl"
        if not trace_file.exists():
            continue
        with open(trace_file) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def analyze(records: list[dict]) -> None:
    # Group rewards by (step, group_id) to get within-group variance (what GRPO uses),
    # and by prompt_idx to get per-prompt variance across all steps.
    groups: dict[tuple, list[float]] = defaultdict(list)
    group_to_prompt: dict[tuple, int] = {}
    prompts: dict[int, list[float]] = defaultdict(list)

    for r in records:
        reward = r.get("rewards", {}).get("reward")
        if reward is None:
            continue
        step = r["run"]["step"]
        group_id = r["run"]["id"]
        prompt_idx = r["task"]["data"]["idx"]
        key = (step, prompt_idx)
        groups[key].append(reward)
        group_to_prompt[key] = prompt_idx
        prompts[prompt_idx].append(reward)

    # Per-group stats (within-group variance — what GRPO actually uses)
    group_stds = []
    zero_var_groups = 0
    for key, rewards in groups.items():
        if len(rewards) < 2:
            continue
        mean = sum(rewards) / len(rewards)
        std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
        group_stds.append(std)
        if std < 0.01:
            zero_var_groups += 1

    # Per-prompt stats
    prompt_stds = []
    for idx in sorted(prompts):
        rewards = prompts[idx]
        if len(rewards) < 2:
            continue
        mean = sum(rewards) / len(rewards)
        std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
        prompt_stds.append((idx, mean, std, rewards))

    print(f"Records loaded:    {len(records)}")
    print(f"Groups analyzed:   {len(group_stds)}")
    print(f"Prompts analyzed:  {len(prompt_stds)}")
    print()

    if group_stds:
        avg_std = sum(group_stds) / len(group_stds)
        zero_frac = zero_var_groups / len(group_stds)
        print(f"Per-group reward std:  mean={avg_std:.3f}  zero-var (<0.01): {zero_frac:.1%}")
        thresholds = [0.05, 0.1, 0.15, 0.2]
        for t in thresholds:
            frac = sum(s < t for s in group_stds) / len(group_stds)
            print(f"  std < {t}: {frac:.1%} of groups")
        print()

    if prompt_stds:
        print("Per-prompt variance (across all sampled rollouts):")
        print(f"  {'idx':>5}  {'mean':>6}  {'std':>6}  {'n':>4}  {'rewards'}")
        print(f"  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*4}")
        prompt_stds.sort(key=lambda x: x[2])  # sort by std ascending
        for idx, mean, std, rewards in prompt_stds:
            reward_summary = " ".join(f"{r:.2f}" for r in sorted(rewards)[:8])
            if len(rewards) > 8:
                reward_summary += f" ... ({len(rewards)} total)"
            print(f"  {idx:>5}  {mean:>6.3f}  {std:>6.3f}  {len(rewards):>4}  {reward_summary}")

        zero_prompt_frac = sum(1 for _, _, s, _ in prompt_stds if s < 0.01) / len(prompt_stds)
        goldilocks = [(i, m, s) for i, m, s, _ in prompt_stds if 0.1 < m < 0.9 and s >= 0.1]
        print()
        print(f"Zero-variance prompts (<0.01 std): {zero_prompt_frac:.1%}")
        print(f"Goldilocks prompts (0.1<mean<0.9, std>=0.1): {len(goldilocks)}/{len(prompt_stds)}")
        if goldilocks:
            print("  Goldilocks prompt indices:", [i for i, _, _ in goldilocks])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to analyze (default: all)")
    args = parser.parse_args()

    records = load_traces(args.run_dir, args.steps)
    analyze(records)


if __name__ == "__main__":
    main()
