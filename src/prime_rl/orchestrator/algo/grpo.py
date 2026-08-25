from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.min_group_reward = config.min_group_reward
        self.max_group_reward = config.max_group_reward
        self.min_group_reward_std = config.min_group_reward_std

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()

        # Initialise drop indicators to 0.0 so the mean across all rollouts is a true rate.
        # Only written when the threshold is active, to avoid polluting WandB with always-zero metrics.
        if self.min_group_reward is not None:
            for rollout in group:
                rollout.metrics["group_dropped_low_reward"] = 0.0
        if self.max_group_reward is not None:
            for rollout in group:
                rollout.metrics["group_dropped_high_reward"] = 0.0
        if self.min_group_reward_std is not None:
            for rollout in group:
                rollout.metrics["group_dropped_low_std"] = 0.0

        if self.min_group_reward is not None and mean_reward < self.min_group_reward:
            get_logger().debug(f"Dropping group (mean={mean_reward:.3f} < min_group_reward={self.min_group_reward})")
            for rollout in group:
                rollout.metrics["group_dropped_low_reward"] = 1.0
                rollout.assign_advantages(0.0)
            return
        if self.max_group_reward is not None and mean_reward > self.max_group_reward:
            get_logger().debug(f"Dropping group (mean={mean_reward:.3f} > max_group_reward={self.max_group_reward})")
            for rollout in group:
                rollout.metrics["group_dropped_high_reward"] = 1.0
                rollout.assign_advantages(0.0)
            return
        if self.min_group_reward_std is not None and std_reward < self.min_group_reward_std:
            get_logger().debug(f"Dropping group (std={std_reward:.3f} < min_group_reward_std={self.min_group_reward_std})")
            for rollout in group:
                rollout.metrics["group_dropped_low_std"] = 1.0
                rollout.assign_advantages(0.0)
            return

        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        abs_advantages = advantages.abs().tolist()
        for rollout, advantage, abs_advantage in zip(group, advantages.tolist(), abs_advantages, strict=True):
            rollout.metrics["abs_advantage"] = abs_advantage
            rollout.assign_advantages(advantage)
