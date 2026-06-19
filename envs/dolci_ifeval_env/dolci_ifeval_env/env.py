"""
IFEval instruction-following environment for the verifiers framework.

Uses the allenai/Dolci-Think-RL-7B dataset, filtered to the "ifeval" subset.
Constraint-checking is shared with ifeval_env (IFEvalG classes).

Dataset: allenai/Dolci-Think-RL-7B
  - prompt:       str, prefixed with "user: " (stripped before use)
  - ground_truth: list of constraint dicts, each with
                    "instruction_id": list[str]
                    "kwargs":         list[dict]
  - dataset:      list[str], filtered to rows where value == ["ifeval"]
"""

import json

from ifeval_env.env import SYSTEM_PROMPT, _check_constraints

import verifiers as vf
from datasets import Dataset


def _to_verifiers_format(ds: Dataset) -> Dataset:
    """Convert Dolci-Think-RL-7B schema to verifiers' prompt/answer schema."""

    def _convert(row: dict) -> dict:
        prompt_text = row.get("prompt", "")
        if prompt_text.startswith("user: "):
            prompt_text = prompt_text[len("user: "):]

        prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = row.get("ground_truth", "")
        if not isinstance(ground_truth, str):
            ground_truth = json.dumps(ground_truth)

        return {"prompt": prompt, "answer": ground_truth}

    return ds.map(_convert, remove_columns=ds.column_names)


def load_environment(
    dataset_name: str = "allenai/Dolci-Think-RL-7B",
    train_split: str = "train",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
    **kwargs,
) -> vf.Environment:
    """Build and return the Dolci IFEval verifiers environment.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_split: Split to use for training and eval data.
        num_train_examples: Subset size for training (-1 = full split).
        num_eval_examples: Subset size for eval (-1 = full split).
        system_prompt: System message prepended to every prompt.
        **kwargs: Forwarded to SingleTurnEnv.
    """

    def _load_raw() -> Dataset:
        import os
        from datasets import load_dataset, load_from_disk

        if os.path.isdir(dataset_name):
            ds = load_from_disk(dataset_name)
        else:
            ds = load_dataset(dataset_name, split=train_split)
        return ds.filter(lambda row: row["dataset"] == ["ifeval"])

    def build_dataset() -> Dataset:
        ds = _load_raw()
        if num_train_examples != -1:
            ds = ds.select(range(num_train_examples))
        return _to_verifiers_format(ds)

    def build_eval_dataset() -> Dataset:
        ds = _load_raw()
        if num_eval_examples != -1:
            ds = ds.select(range(num_eval_examples))
        return _to_verifiers_format(ds)

    def ifeval_reward_func(completion: list[dict], answer: str, **_kwargs) -> float:
        """Reward: fraction of IFEval constraints the response satisfies."""
        assistant_messages = [m for m in completion if m.get("role") == "assistant"]
        if not assistant_messages:
            return 0.0
        response = assistant_messages[-1].get("content", "")
        return _check_constraints(response, answer)

    rubric = vf.Rubric(funcs=[ifeval_reward_func])

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
