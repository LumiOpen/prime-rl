"""
Prime-RL environment for Dolci IF-RLVR (reward-model judge).

Supports two dataset formats:
  1. Dolci-Think-RL-7B — directory with data/*.parquet shards
  2. Dolci-Think-RL-7B-prompt-ground-truth-translated-fi — jsonl file
     Pass use_finnish=True to use translated_prompt + translated_ground_truth.

Dataset categories: "general-quality" and/or "general-quality_ref".
Reward:  OLMo-2-1124-7B-RM sigmoid score ∈ [0, 1].
"""

import json
import logging
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .rubric import LLMJudgeRubric, RubricJudgeRubric

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_PATH = "/shared_silo/scratch/datasets/Dolci-Think-RL-7B"
_GENERAL_CATEGORIES = {"general-quality", "general-quality_ref"}

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Follow ALL formatting and content constraints stated in the user's request exactly."
)

_SYSTEM_PROMPT_FI = (
    "Olet avulias assistentti. Noudata kaikkia käyttäjän pyyntöön sisältyviä ohjeita huolellisesti."
)


def _load_raw(dataset_path: str, categories: list[str], use_finnish: bool = False) -> Dataset:
    path = Path(dataset_path)

    # jsonl file (translated dataset) — read line-by-line to handle inconsistent schema
    if path.is_file() or str(dataset_path).endswith(".jsonl"):
        prompt_col = "translated_prompt" if use_finnish else "prompt"
        gt_col = "translated_ground_truth" if use_finnish else "ground_truth"
        rows = []
        with open(dataset_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ds_val = row.get("dataset", "")
                if isinstance(ds_val, list):
                    ds_val = ds_val[0] if ds_val else ""
                if ds_val not in categories:
                    continue
                passrate = row.get("passrate")
                prompt = row.get(prompt_col, "") or row.get("prompt", "")
                gt = row.get(gt_col, "") or row.get("ground_truth", "")
                if isinstance(gt, list):
                    gt = gt[0] if gt else ""
                rows.append({
                    "prompt": prompt,
                    "ground_truth": gt,
                    "dataset_str": ds_val,
                    "passrate": float(passrate) if passrate is not None else None,
                })
        return Dataset.from_list(rows)

    # parquet directory (original Dolci format)
    import glob
    import os
    import pandas as pd

    root = path
    files = sorted(glob.glob(str(root / "data" / "*.parquet")))
    if not files:
        files = sorted(glob.glob(str(root / "*.parquet")))
    if not files:
        root_exists = root.exists()
        data_exists = (root / "data").exists()
        ls_root = os.listdir(root) if root_exists else []
        raise FileNotFoundError(
            f"No parquet files or jsonl found at {dataset_path}. "
            f"root_exists={root_exists}, data_exists={data_exists}, "
            f"cwd={os.getcwd()}, ls_root={ls_root}"
        )

    cols = ["prompt", "ground_truth", "dataset", "passrate"]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["dataset_str"] = df["dataset"].apply(
        lambda x: list(x)[0] if hasattr(x, "__iter__") and not isinstance(x, str) else str(x)
    )
    df = df[df["dataset_str"].isin(categories)].reset_index(drop=True)
    return Dataset.from_pandas(df)


def _build_example(row: dict) -> dict | None:
    prompt_text: str = row["prompt"]

    prefix = "user: " if not row.get("_fi") else "käyttäjä: "
    if isinstance(prompt_text, str) and (
        prompt_text.startswith("user: ") or prompt_text.lower().startswith("käyttäjä: ")
    ):
        user_content = prompt_text.split(": ", 1)[1]
    else:
        user_content = str(prompt_text)

    gt = row.get("ground_truth") or ""
    if isinstance(gt, list):
        gt = gt[0] if gt else ""

    return {
        "question": user_content,
        "answer": None,
        "info": {
            "category": row.get("dataset_str", "general-quality"),
            "prompt_text": user_content,
            "reference": gt,
        },
    }


def load_environment(
    dataset_path: str = _DEFAULT_DATASET_PATH,
    categories: list[str] = ("general-quality", "general-quality_ref"),
    rm_model_path: str | None = None,
    rm_device: str = "cuda",
    rm_server_url: str | None = None,
    rm_max_response_chars: int | None = None,
    min_passrate: float = 0.05,
    max_passrate: float = 0.95,
    dataset_shuffle: bool = True,
    dataset_seed: int = 42,
    use_finnish: bool = False,
    system_prompt: str | None = None,
    map_kwargs: dict = {},
    # LLM-as-judge mode — activated when custom_rm=True.
    # The judge model is served via the same vLLM instance as the RM
    # (rm_server_url is reused as judge_server_url automatically).
    custom_rm: bool = False,
    judge_model_path: str | None = None,
    judge_prompt_template: str | None = None,
    max_judge_tokens: int = 512,
    judge_temperature: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """
    Load the Dolci rubric-judge environment (RM-based reward).

    Args:
        dataset_path:           Path to Dolci-Think-RL-7B directory.
        categories:             Dataset categories to include.
        rm_model_path:          Path/name of the reward model (SequenceClassification RM).
                                Required unless custom_rm=True.
        rm_device:              Device for local RM backend (default: "cuda").
        rm_server_url:          vLLM reward model server URL.  Injected automatically
                                by auto_setup_rm_inference; also used as the judge
                                server URL when custom_rm=True.
        min_passrate:           Drop examples below this passrate.
        max_passrate:           Drop examples above this passrate.
        dataset_shuffle:        Shuffle the dataset.
        dataset_seed:           Seed for shuffling.
        system_prompt:          System prompt prepended to every conversation.
        custom_rm:              When True, use an LLM-as-judge via vLLM chat completions
                                instead of a SequenceClassification reward model.
                                The judge is served at rm_server_url (injected automatically).
        judge_model_path:       Path/name of the judge model as served by vLLM
                                (must match --served-model-name).  Required when custom_rm=True.
        judge_prompt_template:  Custom judge prompt with {question} and {answer} placeholders.
                                Defaults to _DEFAULT_JUDGE_PROMPT.
        max_judge_tokens:       Max tokens the judge may generate (default: 512).
        judge_temperature:      Sampling temperature for the judge (default: 0.0 = greedy).
    """
    if system_prompt is None:
        system_prompt = _SYSTEM_PROMPT

    if custom_rm:
        if judge_model_path is None:
            raise ValueError(
                "judge_model_path is required when custom_rm=True. "
                "Add judge_model_path = '/path/to/Qwen3-8B' to env args in the TOML."
            )
        if rm_server_url is None:
            raise ValueError(
                "rm_server_url must be set (or auto-injected via [rm_inference]) "
                "when custom_rm=True."
            )
    else:
        if rm_model_path is None:
            raise ValueError(
                "rm_model_path is required for rubric-judge-env. "
                "Add rm_model_path = '/shared_silo/scratch/models/OLMo-2-1124-7B-RM' to env args in the TOML."
            )

    invalid = set(categories) - _GENERAL_CATEGORIES
    if invalid:
        raise ValueError(
            f"Invalid categories for rubric-judge-env: {invalid}. "
            f"Allowed: {_GENERAL_CATEGORIES}"
        )

    categories = list(categories)

    def build_dataset():
        raw = _load_raw(dataset_path, categories, use_finnish=use_finnish)

        if "passrate" in raw.column_names:
            import math
            raw = raw.filter(
                lambda x: (
                    x["passrate"] is None
                    or (isinstance(x["passrate"], float) and math.isnan(x["passrate"]))
                    or (x["passrate"] >= min_passrate and x["passrate"] <= max_passrate)
                ),
                **map_kwargs,
            )

        examples = [_build_example(row) for row in raw]
        examples = [ex for ex in examples if ex is not None]

        if not examples:
            raise RuntimeError(
                f"No usable examples found in {dataset_path} for categories={categories}. "
                "Check dataset path and passrate filters."
            )

        ds = Dataset.from_list(examples)
        if dataset_shuffle:
            ds = ds.shuffle(seed=dataset_seed)
        return ds

    if custom_rm:
        from .rubric import _DEFAULT_JUDGE_PROMPT
        rubric = LLMJudgeRubric(
            judge_model_path=judge_model_path,
            judge_server_url=rm_server_url,
            judge_prompt_template=judge_prompt_template or _DEFAULT_JUDGE_PROMPT,
            max_judge_tokens=max_judge_tokens,
            judge_temperature=judge_temperature,
        )
    else:
        rubric = RubricJudgeRubric(rm_model_path=rm_model_path, rm_device=rm_device, rm_server_url=rm_server_url, rm_max_response_chars=rm_max_response_chars)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
