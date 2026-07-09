"""
IFEval instruction-following environment for the verifiers framework.

Supports two dataset formats:
  1. Dolci-Think-RL-7B — directory with data/*.parquet shards
       columns: prompt, ground_truth, dataset
  2. Dolci-Think-RL-7B-prompt-ground-truth-translated-fi — jsonl file
       columns: prompt, translated_prompt, ground_truth, translated_ground_truth, dataset
       Pass use_finnish=True to use translated_prompt + translated_ground_truth.

Constraint-checking is shared with ifeval_env (IFEvalG classes).
"""

import glob
import json
import os
from pathlib import Path

import pandas as pd

from ifeval_env.env import SYSTEM_PROMPT
from if_eval_env.rubric import IFEvalRubric
from if_eval_env.registry import _parse_ground_truth

import verifiers as vf
from datasets import Dataset

_DEFAULT_DATASET_PATH = "/shared_silo/scratch/datasets/Dolci-Think-RL-7B"


def _load_raw(dataset_path: str, use_finnish: bool = False) -> Dataset:
    """Load ifeval rows from a Dolci dataset.

    Supports:
    - directory with data/*.parquet shards (original Dolci format)
    - a .jsonl file (translated Finnish dataset)
    """
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
                if ds_val != "ifeval":
                    continue
                prompt = row.get(prompt_col, "") or row.get("prompt", "")
                gt = row.get(gt_col, "") or row.get("ground_truth", "")
                rows.append({"prompt": prompt, "ground_truth": gt})
        return Dataset.from_list(rows)

    # parquet directory (original Dolci format)
    files = sorted(glob.glob(str(path / "data" / "*.parquet")))
    if not files:
        files = sorted(glob.glob(str(path / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files or jsonl found at {dataset_path}."
        )

    cols = ["prompt", "ground_truth", "dataset"]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["dataset_str"] = df["dataset"].apply(
        lambda x: list(x)[0] if hasattr(x, "__iter__") and not isinstance(x, str) else str(x)
    )
    df = df[df["dataset_str"] == "ifeval"].reset_index(drop=True)
    return Dataset.from_pandas(df[["prompt", "ground_truth"]])


def _to_verifiers_format(ds: Dataset) -> Dataset:
    """Convert Dolci-Think-RL-7B schema to verifiers' prompt/answer/info schema."""

    def _convert(row: dict) -> dict:
        prompt_text = row.get("prompt", "")
        if prompt_text.startswith("user: ") or prompt_text.lower().startswith("käyttäjä: "):
            prompt_text = prompt_text.split(": ", 1)[1]

        prompt = [{"role": "user", "content": prompt_text}]

        gt_raw = row.get("ground_truth", "")
        instruction_ids, kwargs_list = _parse_ground_truth(gt_raw)
        gt = gt_raw if isinstance(gt_raw, str) else json.dumps(gt_raw)

        return {
            "prompt": prompt,
            "answer": gt,
            "info": {
                "instruction_ids": json.dumps(instruction_ids),
                "kwargs_list": json.dumps(kwargs_list),
                "prompt_text": prompt_text,
            },
        }

    return ds.map(_convert, remove_columns=ds.column_names)


_DEFAULT_FI_DATASET_PATH = (
    "/shared_silo/scratch/datasets/"
    "Dolci-Think-RL-7B-prompt-ground-truth-translated-fi/"
    "Dolci-Think-RL-7B-prompt-ground-truth-translated-fi.jsonl"
)

SYSTEM_PROMPT_FI = (
    "Olet avulias assistentti. Noudata kaikkia käyttäjän pyyntöön sisältyviä ohjeita huolellisesti."
)


def load_environment(
    dataset_path: str = _DEFAULT_DATASET_PATH,
    use_finnish: bool = False,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str | None = None,
    **kwargs,
) -> vf.Environment:
    """Build and return the Dolci IFEval verifiers environment.

    Args:
        dataset_path: Path to Dolci-Think-RL-7B directory (parquet) or .jsonl file.
        use_finnish:  If True, use translated_prompt / translated_ground_truth columns.
        num_train_examples: Subset size for training (-1 = full split).
        num_eval_examples: Subset size for eval (-1 = full split).
        system_prompt: System message (defaults to Finnish prompt when use_finnish=True).
        **kwargs: Forwarded to SingleTurnEnv.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    def build_dataset() -> Dataset:
        ds = _load_raw(dataset_path, use_finnish=use_finnish)
        if num_train_examples != -1:
            ds = ds.select(range(num_train_examples))
        return _to_verifiers_format(ds)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        system_prompt=system_prompt,
        rubric=IFEvalRubric(),
        **kwargs,
    )


def load_environment_fi(
    dataset_path: str = _DEFAULT_FI_DATASET_PATH,
    **kwargs,
) -> vf.Environment:
    """Finnish variant — shortcut for load_environment(use_finnish=True)."""
    return load_environment(dataset_path=dataset_path, use_finnish=True, **kwargs)
