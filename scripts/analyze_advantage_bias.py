"""Analyze per-token advantage bias across training steps.

For each GRPO group, rollout advantage = reward - group_mean_reward. If rollouts
with negative advantages are systematically longer, there are more negative-advantage
tokens in each batch than positive ones — a net negative gradient bias that can push
token probabilities down uniformly and increase entropy.

Usage:
    python scripts/analyze_advantage_bias.py <run_output_dir> [--steps N]

Example:
    python scripts/analyze_advantage_bias.py \
        outputs/normal-olmo3-7b-think-sft-ifeval-128subset-32k-20260728_131728 \
        --steps 20
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def count_output_tokens(record: dict) -> int:
    """Count sampled (output) tokens from a trace record."""
    return sum(len(n.get("token_ids", [])) for n in record["nodes"] if n.get("sampled", False))


def load_traces(run_dir: Path, steps: int | None) -> list[dict]:
    rollout_dir = run_dir / "run_default" / "rollouts"
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
    # Group by (step, prompt_idx) to reconstruct GRPO groups
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        reward = r.get("rewards", {}).get("reward")
        if reward is None:
            continue
        step = r["run"]["step"]
        prompt_idx = r["task"]["data"]["idx"]
        groups[(step, prompt_idx)].append({
            "reward": reward,
            "n_tokens": count_output_tokens(r),
        })

    # Per-token advantage stats
    total_pos_tokens = 0
    total_neg_tokens = 0
    total_zero_tokens = 0
    pos_lengths = []
    neg_lengths = []

    per_step_bias: dict[int, list[float]] = defaultdict(list)

    for (step, _), rollouts in groups.items():
        if len(rollouts) < 2:
            continue
        rewards = [r["reward"] for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)

        for r in rollouts:
            adv = r["reward"] - mean_reward
            n = r["n_tokens"]
            if adv > 1e-6:
                total_pos_tokens += n
                pos_lengths.append(n)
                per_step_bias[step].extend([adv] * n)
            elif adv < -1e-6:
                total_neg_tokens += n
                neg_lengths.append(n)
                per_step_bias[step].extend([adv] * n)
            else:
                total_zero_tokens += n

    total_tokens = total_pos_tokens + total_neg_tokens + total_zero_tokens
    if total_tokens == 0:
        print("No data found.")
        return

    print(f"Records loaded:      {len(records)}")
    print(f"Groups analyzed:     {len(groups)}")
    print()
    print("Per-token advantage breakdown:")
    print(f"  Positive-advantage tokens: {total_pos_tokens:>8} ({total_pos_tokens/total_tokens:.1%})")
    print(f"  Negative-advantage tokens: {total_neg_tokens:>8} ({total_neg_tokens/total_tokens:.1%})")
    print(f"  Zero-advantage tokens:     {total_zero_tokens:>8} ({total_zero_tokens/total_tokens:.1%})")
    print()

    bias = (total_pos_tokens - total_neg_tokens) / max(total_pos_tokens + total_neg_tokens, 1)
    print(f"Token-weighted advantage bias: {bias:+.3f}  (+1=all positive, -1=all negative)")
    print()

    if pos_lengths and neg_lengths:
        mean_pos_len = sum(pos_lengths) / len(pos_lengths)
        mean_neg_len = sum(neg_lengths) / len(neg_lengths)
        print("Mean output tokens per rollout:")
        print(f"  Positive-advantage rollouts: {mean_pos_len:.0f} tokens")
        print(f"  Negative-advantage rollouts: {mean_neg_len:.0f} tokens")
        print(f"  Length ratio (neg/pos):      {mean_neg_len/mean_pos_len:.2f}x")
        print()

    # Per-step bias trend
    if len(per_step_bias) > 1:
        print("Per-step token-weighted mean advantage:")
        for step in sorted(per_step_bias):
            vals = per_step_bias[step]
            mean = sum(vals) / len(vals)
            print(f"  step {step:>4}: {mean:+.4f}  (n={len(vals)} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    records = load_traces(args.run_dir, args.steps)
    analyze(records)


if __name__ == "__main__":
    main()
