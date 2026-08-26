"""Slice the first N ifeval rows from allenai/Dolci-Think-RL-7B and save to disk.

Usage:
    uv run python scripts/create_ifeval_subset.py \
        --output /path/to/ifeval_128 \
        [--n 128] \
        [--dataset allenai/Dolci-Think-RL-7B] \
        [--split train]
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dataset", default="allenai/Dolci-Think-RL-7B")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)

    ds = ds.filter(lambda row: row["dataset"] == ["ifeval"])
    print(f"Rows after ifeval filter: {len(ds)}")

    if args.n > len(ds):
        raise ValueError(f"--n {args.n} exceeds available ifeval rows ({len(ds)})")

    ds = ds.select(range(args.n))
    print(f"Sliced to {len(ds)} rows.")

    args.output.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
