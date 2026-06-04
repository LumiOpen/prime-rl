"""
Prime-RL environment for Dolci IF-RLVR (reward-model judge).

Dataset: Dolci-Think-RL-7B (parquet shards), categories "general-quality" and/or "general-quality_ref".
Reward:  OLMo-2-1124-7B-RM sigmoid score ∈ [0, 1].
"""

import logging
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .rubric import RubricJudgeRubric

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_PATH = "/shared_silo/scratch/datasets/Dolci-Think-RL-7B"
_GENERAL_CATEGORIES = {"general-quality", "general-quality_ref"}

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Follow ALL formatting and content constraints stated in the user's request exactly."
)


def _load_raw(dataset_path: str, categories: list[str]) -> Dataset:
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

    if isinstance(prompt_text, str) and prompt_text.startswith("user: "):
        user_content = prompt_text[len("user: "):]
    else:
        user_content = str(prompt_text)

    return {
        "question": user_content,
        "answer": None,
        "info": {
            "category": row.get("dataset_str", "general-quality"),
            "prompt_text": user_content,
        },
    }


def load_environment(
    dataset_path: str = _DEFAULT_DATASET_PATH,
    categories: list[str] = ("general-quality", "general-quality_ref"),
    rm_model_path: str | None = None,
    rm_device: str = "cuda",
    rm_server_url: str | None = None,
    min_passrate: float = 0.05,
    max_passrate: float = 0.95,
    dataset_shuffle: bool = True,
    dataset_seed: int = 42,
    system_prompt: str = _SYSTEM_PROMPT,
    map_kwargs: dict = {},
    **kwargs,
) -> vf.Environment:
    """
    Load the Dolci rubric-judge environment (RM-based reward).

    Args:
        dataset_path:    Path to Dolci-Think-RL-7B directory (contains data/*.parquet).
        categories:      Dataset categories to include. Options: "general-quality",
                         "general-quality_ref".
        rm_model_path:   Path/name of the reward model. Required.
                         For local: HF path or local dir.
                         For vLLM: must match the --served-model-name the server uses.
        rm_device:       Device to load the model on when using local backend (default: "cuda").
        rm_server_url:   Base URL of a running vLLM reward model server, e.g.
                         "http://localhost:8000". When set, scoring calls POST /v1/score
                         instead of loading the model locally. rm_device is ignored.
        min_passrate:    Drop examples below this passrate (NaN rows always included).
        max_passrate:    Drop examples above this passrate (NaN rows always included).
        dataset_shuffle: Shuffle the dataset.
        dataset_seed:    Seed for shuffling.
        system_prompt:   System prompt prepended to every conversation.
    """
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
        raw = _load_raw(dataset_path, categories)

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

    rubric = RubricJudgeRubric(rm_model_path=rm_model_path, rm_device=rm_device, rm_server_url=rm_server_url)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
