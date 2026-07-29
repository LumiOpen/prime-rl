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

import ast
import json
import logging
import random

from ifeval_env.env import SYSTEM_PROMPT, _check_constraints
from ifeval_env.IFEvalG.instructions_registry import INSTRUCTION_DICT

import verifiers as vf
from datasets import Dataset

from dolci_ifeval_env.utils import strip_think_blocks

logger = logging.getLogger("verifiers.v1")
_LOG_SAMPLE_RATE = 0.01


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

    def ifeval_reward_func(completion: list[dict], answer: str, **kwargs) -> float:
        """Reward: fraction of IFEval constraints the response satisfies."""
        assistant_messages = [m for m in completion if m.get("role") == "assistant"]
        if not assistant_messages:
            return 0.0
        response = assistant_messages[-1].get("content", "")
        score = _check_constraints(response, answer)
        if random.random() < _LOG_SAMPLE_RATE:
            try:
                constraint_dict = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                constraint_dict = json.loads(answer)
            constraint_dict = constraint_dict[0]
            if isinstance(constraint_dict, str):
                try:
                    constraint_dict = json.loads(constraint_dict)
                except json.JSONDecodeError:
                    constraint_dict = ast.literal_eval(constraint_dict)
                if isinstance(constraint_dict, list):
                    constraint_dict = constraint_dict[0]
            answer_text = strip_think_blocks(response)
            instruction_keys = constraint_dict.get("instruction_id", [])
            args_list = constraint_dict.get("kwargs", [])
            results = []
            for iid, args in zip(instruction_keys, args_list):
                if args is None:
                    args = {}
                args = {k: v for k, v in args.items() if v is not None}
                try:
                    instance = INSTRUCTION_DICT[iid](iid)
                    instance.build_description(**args)
                    passed = bool(instance.check_following(answer_text))
                except Exception:
                    passed = False
                results.append(f"{'PASS' if passed else 'FAIL'} {iid}")
            prompt = kwargs.get("prompt", [])
            prompt_text = prompt[-1].get("content", "") if prompt else ""
            logger.info(
                "IFEval (en) sample\n"
                f"  prompt ({len(prompt_text)} chars): {prompt_text[:200]!r}\n"
                f"  response ({len(response)} chars): {response[:300]!r}\n"
                f"  score: {score:.4f}  constraints: {sum(r.startswith('PASS') for r in results)}/{len(results)}\n"
                + "\n".join(f"    {r}" for r in results)
            )
        return score

    rubric = vf.Rubric(funcs=[ifeval_reward_func])

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
