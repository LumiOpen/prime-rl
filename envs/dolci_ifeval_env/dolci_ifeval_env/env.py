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
    system_prompt: str | None = SYSTEM_PROMPT,
    language_reward_weight: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """Build and return the Dolci IFEval verifiers environment.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_split: Split to use for training and eval data.
        num_train_examples: Subset size for training (-1 = full split).
        num_eval_examples: Subset size for eval (-1 = full split).
        system_prompt: System message prepended to every prompt. Pass None or "" to disable.
        **kwargs: Forwarded to SingleTurnEnv.
    """
    system_prompt = system_prompt or None

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

    def _parse_constraint_dict(answer: str) -> dict:
        try:
            cd = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            cd = json.loads(answer)
        cd = cd[0]
        if isinstance(cd, str):
            try:
                cd = json.loads(cd)
            except json.JSONDecodeError:
                cd = ast.literal_eval(cd)
            if isinstance(cd, list):
                cd = cd[0]
        return cd

    def _eval_constraints(response: str, answer: str) -> dict[str, bool]:
        """Return {instruction_id: passed} for every constraint in the answer."""
        answer_text = strip_think_blocks(response)
        try:
            constraint_dict = _parse_constraint_dict(answer)
        except Exception:
            return {}
        instruction_keys = constraint_dict.get("instruction_id", [])
        args_list = constraint_dict.get("kwargs", [])
        results = {}
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
            results[iid] = passed
        return results

    def ifeval_reward_func(completion: list[dict], answer: str, state=None, **kwargs) -> float:
        """Reward: fraction of IFEval constraints satisfied, with per-constraint metrics in state."""
        assistant_messages = [m for m in completion if m.get("role") == "assistant"]
        if not assistant_messages:
            return 0.0
        raw_response = assistant_messages[-1].get("content") or ""
        response = strip_think_blocks(raw_response)
        score = _check_constraints(response, answer)
        per_constraint = _eval_constraints(response, answer)

        # Write per-constraint pass rates into state metrics so they flow to W&B
        if state is not None and per_constraint:
            existing = state.get("metrics") or {}
            existing.update({f"ifeval/{iid}": float(passed) for iid, passed in per_constraint.items()})
            state["metrics"] = existing

        if random.random() < _LOG_SAMPLE_RATE:
            prompt = kwargs.get("prompt", [])
            prompt_text = prompt[-1].get("content", "") if prompt else ""
            results = [f"{'PASS' if p else 'FAIL'} {iid}" for iid, p in per_constraint.items()]
            logger.info(
                "IFEval (en) sample\n"
                f"  prompt ({len(prompt_text)} chars): {prompt_text[:200]!r}\n"
                f"  response ({len(response)} chars): {response[:300]!r}\n"
                f"  score: {score:.4f}  constraints: {sum(p for p in per_constraint.values())}/{len(per_constraint)}\n"
                + "\n".join(f"    {r}" for r in results)
            )

        # Multiply by language consistency score
        try:
            from language_reward import compute_language_score
            prompt = kwargs.get("prompt", [])
            question = prompt[-1].get("content", "") if prompt else ""
            lang_score = compute_language_score(question, strip_think_blocks(response))
            if state is not None:
                existing = state.get("metrics") or {}
                existing["language_score"] = lang_score
                state["metrics"] = existing
            if language_reward_weight > 0.0:
                score = score * (1.0 - language_reward_weight + language_reward_weight * lang_score)
        except Exception:
            pass

        return score

    rubric = vf.Rubric(funcs=[ifeval_reward_func])

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
