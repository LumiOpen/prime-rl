"""
IFEval instruction-following environment for the verifiers framework.

Ports the IFEvalVerifier from open-instruct (open_instruct/ground_truth_utils.py)
with identical constraint-checking logic: each row carries one or more instruction
constraints, and the reward is the fraction of constraints the model's response
satisfies.

Dataset: allenai/IF_multi_constraints_upto5
  - messages:      list[dict] with role/content keys (the prompt conversation)
  - ground_truth:  list of constraint dicts, each with
                     "instruction_id": list[str]  (keys into INSTRUCTION_DICT)
                     "kwargs":         list[dict]  (args for build_description)
  - dataset:       "ifeval"

Constraint checking is delegated to the IFEvalG instruction classes from
open-instruct, declared as a pinned git dependency.
"""

import ast
import json

from ifeval_env.IFEvalG import instructions_registry

import verifiers as vf
from datasets import Dataset

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow all instructions in the user's request carefully."
)


def _remove_thinking_section(text: str) -> str:
    """Strip <think>…</think> reasoning blocks and answer tags.

    Matches open-instruct's remove_thinking_section() exactly.
    """
    text = text.replace("<|assistant|>", "").strip()
    text = text.split("</think>")[-1]
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def _check_constraints(response: str, label: str) -> float:
    """Return the fraction of IFEval constraints satisfied by *response*.

    Mirrors IFEvalVerifier.__call__ from open_instruct/ground_truth_utils.py:
      1. Parse the label as a Python/JSON literal.
      2. Strip thinking sections from the response.
      3. For each (instruction_key, kwargs) pair, instantiate the instruction
         class, call build_description(**kwargs), then check_following(answer).
      4. Return mean score across all constraints.
    """
    try:
        constraint_dict = ast.literal_eval(label)
    except (ValueError, SyntaxError):
        constraint_dict = json.loads(label)

    constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        try:
            constraint_dict = json.loads(constraint_dict)
        except json.JSONDecodeError:
            constraint_dict = ast.literal_eval(constraint_dict)
        if isinstance(constraint_dict, list):
            constraint_dict = constraint_dict[0]

    answer = _remove_thinking_section(response)

    if not response.strip() or not answer:
        return 0.0

    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]

    rewards = []
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        instruction_cls = INSTRUCTION_DICT[instruction_key]
        instance = instruction_cls(instruction_key)
        instance.build_description(**args)
        rewards.append(1.0 if instance.check_following(answer) else 0.0)

    return sum(rewards) / len(rewards) if rewards else 0.0


def _to_verifiers_format(ds: Dataset) -> Dataset:
    """Convert the open-instruct dataset schema to verifiers' prompt/answer schema.

    open-instruct columns:
      messages      – list[dict{role, content}]
      ground_truth  – list of constraint dicts (or JSON string)
      dataset       – "ifeval"

    verifiers expects:
      prompt   – list[dict{role, content}]  (prompt only, no assistant turn)
      answer   – str                        (JSON-serialised constraint blob)
    """

    def _convert(row: dict) -> dict:
        messages = row.get("messages") or row.get("sft_messages") or []
        prompt = [m for m in messages if m.get("role") != "assistant"]

        ground_truth = row.get("ground_truth", "")
        # Dataset stores ground_truth as a list of constraint dicts; serialise
        # to a string so _check_constraints can parse it with ast.literal_eval,
        # matching what open-instruct's IFEvalVerifier receives.
        if not isinstance(ground_truth, str):
            ground_truth = json.dumps(ground_truth)

        return {"prompt": prompt, "answer": ground_truth}

    return ds.map(_convert, remove_columns=ds.column_names)


def load_environment(
    dataset_name: str = "allenai/IF_multi_constraints_upto5",
    train_split: str = "train",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
    **kwargs,
) -> vf.Environment:
    """Build and return the IFEval verifiers environment.

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
            return load_from_disk(dataset_name)
        return load_dataset(dataset_name, split=train_split)

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
