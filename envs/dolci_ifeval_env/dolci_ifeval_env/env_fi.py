"""
Finnish IFEval instruction-following environment for the verifiers framework.

Loads from the fi_dataset.jsonl produced by the strip-translate-regenerate
pipeline. Prompts are read from the prompt field (Finnish, with "user: " prefix).
Constraint-checking uses the Finnish-adapted IFEvalG checkers.
"""

import ast
import json
import logging
import random

from dolci_ifeval_env.IFEvalG_fi import instructions_registry_fi
from dolci_ifeval_env.IFEvalG_fi import instructions_util_fi
from dolci_ifeval_env.utils import strip_think_blocks

import verifiers as vf
from datasets import Dataset

INSTRUCTION_DICT = instructions_registry_fi.INSTRUCTION_DICT

logger = logging.getLogger("verifiers.v1")
_LOG_SAMPLE_RATE = 0.01

SYSTEM_PROMPT_FI = (
    "Olet avulias assistentti. Noudata kaikkia käyttäjän pyyntöön sisältyviä ohjeita huolellisesti."
)


def _remove_thinking_section(text: str) -> str:
    """Strip <think>…</think> blocks and answer tags."""
    text = text.replace("<|assistant|>", "").strip()
    text = strip_think_blocks(text)
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def _check_constraints_fi(response: str, label: str) -> float:
    """Return the fraction of IFEval constraints satisfied by *response*.

    Identical logic to the English version but uses the Finnish INSTRUCTION_DICT
    and the Finnish sentence tokenizer via instructions_util_fi.
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


def _to_verifiers_format_fi(ds: Dataset) -> Dataset:
    """Convert the fi_dataset schema to verifiers' prompt/answer schema."""

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


_DEFAULT_FI_DATASET = "/shared_silo/scratch/kahakala/lib/ifeval-env/fi_dataset.jsonl"


def load_environment(
    dataset_name: str = _DEFAULT_FI_DATASET,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT_FI,
    language_reward_weight: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """Build and return the Finnish Dolci IFEval verifiers environment.

    Args:
        dataset_name: Path to fi_dataset.jsonl or a HuggingFace dataset identifier.
        num_train_examples: Subset size for training (-1 = full dataset).
        num_eval_examples: Subset size for eval (-1 = full dataset).
        system_prompt: System message prepended to every prompt.
        **kwargs: Forwarded to SingleTurnEnv.
    """

    def _load_raw() -> Dataset:
        import os
        from datasets import load_dataset, load_from_disk

        if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
            return load_dataset("json", data_files=dataset_name, split="train")
        if os.path.isdir(dataset_name):
            return load_from_disk(dataset_name)
        return load_dataset(dataset_name, split="train")

    def build_dataset() -> Dataset:
        ds = _load_raw()
        if num_train_examples != -1:
            ds = ds.select(range(num_train_examples))
        return _to_verifiers_format_fi(ds)

    def build_eval_dataset() -> Dataset:
        ds = _load_raw()
        if num_eval_examples != -1:
            ds = ds.select(range(num_eval_examples))
        return _to_verifiers_format_fi(ds)

    def ifeval_reward_func_fi(completion: list[dict], answer: str, state=None, **kwargs) -> float:
        """Reward: fraction of IFEval constraints the Finnish response satisfies."""
        assistant_messages = [m for m in completion if m.get("role") == "assistant"]
        if not assistant_messages:
            return 0.0
        raw_response = assistant_messages[-1].get("content") or ""
        response = _remove_thinking_section(raw_response)
        score = _check_constraints_fi(response, answer)
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
            answer_text = _remove_thinking_section(response)
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
                "IFEval (fi) sample\n"
                f"  prompt ({len(prompt_text)} chars): {prompt_text[:200]!r}\n"
                f"  response ({len(response)} chars): {response[:300]!r}\n"
                f"  score: {score:.4f}  constraints: {sum(r.startswith('PASS') for r in results)}/{len(results)}\n"
                + "\n".join(f"    {r}" for r in results)
            )

        # Multiply by language consistency score
        try:
            from language_reward import compute_language_score
            prompt = kwargs.get("prompt", [])
            question = prompt[-1].get("content", "") if prompt else ""
            lang_score = compute_language_score(question, _remove_thinking_section(response))
            if state is not None:
                existing = state.get("metrics") or {}
                existing["language_score"] = lang_score
                state["metrics"] = existing
            if language_reward_weight > 0.0:
                score = score * (1.0 - language_reward_weight + language_reward_weight * lang_score)
        except Exception:
            pass

        return score

    rubric = vf.Rubric(funcs=[ifeval_reward_func_fi])

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
