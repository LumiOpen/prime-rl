"""
Prime-RL environment for Dolci IF-RLVR (constraint-based instruction following).

Dataset: Dolci-Think-RL-7B (parquet shards), category "ifeval".
Reward:  fraction of IFEval/IFEvalG constraints satisfied ∈ [0, 1].
"""

import logging
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .registry import _parse_ground_truth
from .rubric import IFEvalRubric

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_PATH = "/shared_silo/scratch/datasets/Dolci-Think-RL-7B"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Follow ALL formatting and content constraints stated in the user's request exactly."
)


def _load_raw(dataset_path: str) -> Dataset:
    import glob
    import os
    import pandas as pd

    root = Path(dataset_path)
    files = sorted(glob.glob(str(root / "data" / "*.parquet")))
    if not files:
        files = sorted(glob.glob(str(root / "*.parquet")))
    if not files:
        root_exists = root.exists()
        data_exists = (root / "data").exists()
        ls_root = os.listdir(root) if root_exists else []
        raise FileNotFoundError(
            f"No parquet files found in {dataset_path}/data/ or {dataset_path}/. "
            f"root_exists={root_exists}, data_exists={data_exists}, "
            f"cwd={os.getcwd()}, ls_root={ls_root}"
        )

    cols = ["prompt", "ground_truth", "dataset", "constraint", "passrate"]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["dataset_str"] = df["dataset"].apply(
        lambda x: list(x)[0] if hasattr(x, "__iter__") and not isinstance(x, str) else str(x)
    )
    df = df[df["dataset_str"] == "ifeval"].reset_index(drop=True)
    return Dataset.from_pandas(df)


def _build_example(row: dict) -> dict | None:
    prompt_text: str = row["prompt"]
    gt = row["ground_truth"]

    instruction_ids, kwargs_list = _parse_ground_truth(gt)
    if not instruction_ids:
        return None

    if isinstance(prompt_text, str) and prompt_text.startswith("user: "):
        user_content = prompt_text[len("user: "):]
    else:
        user_content = str(prompt_text)

    import json
    return {
        "question": user_content,
        "answer": None,
        "info": {
            "category": "ifeval",
            "instruction_ids": json.dumps(instruction_ids),
            "kwargs_list": json.dumps(kwargs_list),
            "constraint_text": str(row.get("constraint", "")),
            "prompt_text": user_content,
        },
    }


def load_environment(
    dataset_path: str = _DEFAULT_DATASET_PATH,
    min_passrate: float = 0.05,
    max_passrate: float = 0.95,
    dataset_shuffle: bool = True,
    dataset_seed: int = 42,
    system_prompt: str = _SYSTEM_PROMPT,
    map_kwargs: dict = {},
    **kwargs,
) -> vf.Environment:
    """
    Load the Dolci IF-RLVR environment (IFEval constraint-based reward).

    Args:
        dataset_path:    Path to Dolci-Think-RL-7B directory (contains data/*.parquet).
        min_passrate:    Drop examples where all rollouts failed (too hard).
        max_passrate:    Drop examples where all rollouts passed (no learning signal).
        dataset_shuffle: Shuffle the dataset.
        dataset_seed:    Seed for shuffling.
        system_prompt:   System prompt prepended to every conversation.
    """
    def build_dataset():
        raw = _load_raw(dataset_path)

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

        examples = []
        for row in raw:
            ex = _build_example(row)
            if ex is not None:
                examples.append(ex)

        if not examples:
            raise RuntimeError(
                f"No usable examples found in {dataset_path} for category=ifeval. "
                "Check dataset path and passrate filters."
            )

        ds = Dataset.from_list(examples)
        if dataset_shuffle:
            ds = ds.shuffle(seed=dataset_seed)
        return ds

    rubric = IFEvalRubric()

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
