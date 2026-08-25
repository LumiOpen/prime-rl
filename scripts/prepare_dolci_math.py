"""Convert the Dolci-Think-RL math subset to prime-rl math-env format.

Reads a Dolci train.jsonl, filters to the math category, validates single-turn
structure, and writes a plain-text question/answer jsonl compatible with the
prime-rl math env (which expects configurable question_key and answer_key columns
with no message formatting).

Usage:
    uv run scripts/prepare_dolci_math.py \\
        --input /path/to/Dolci-Think-RL-7B/data/train.jsonl \\
        --output /path/to/output/train.jsonl \\
        [--question-key question] [--answer-key answer] [--category math]

Output schema (one JSON object per line):
    {question_key}: plain text question (no "user:" prefix)
    {answer_key}:   plain text answer (first element if ground_truth is a list)
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Path to Dolci train.jsonl")
    p.add_argument("--output", required=True, type=Path, help="Output jsonl path")
    p.add_argument("--category", default="math", help="Dataset category to filter (default: math)")
    p.add_argument("--question-key", default="question", help="Output column name for the question (default: question)")
    p.add_argument("--answer-key", default="answer", help="Output column name for the answer (default: answer)")
    p.add_argument("--user-prefix", default="user:", help="Prefix to strip from prompts (default: 'user:')")
    return p.parse_args()


def strip_user_prefix(prompt: str, prefix: str) -> str | None:
    """Strip the user prefix and return plain text, or None if malformed."""
    stripped = prompt.strip()
    if not stripped.lower().startswith(prefix.lower()):
        return None
    return stripped[len(prefix):].strip()


def is_single_turn(prompt: str, prefix: str) -> bool:
    """Return True if the prompt contains exactly one user turn."""
    return prompt.lower().count(prefix.lower()) == 1


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = skipped_category = skipped_multiturn = skipped_format = written = 0

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            row = json.loads(line)

            # Normalise dataset field (may be a list)
            category = row.get("dataset", "")
            if isinstance(category, list):
                category = category[0] if category else ""

            if category != args.category:
                skipped_category += 1
                continue

            prompt = row.get("prompt", "")
            ground_truth = row.get("ground_truth", "")

            # Reject multi-turn examples
            if not is_single_turn(prompt, args.user_prefix):
                skipped_multiturn += 1
                continue

            question = strip_user_prefix(prompt, args.user_prefix)
            if question is None:
                skipped_format += 1
                continue

            # Unwrap list ground_truth
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ""
            answer = str(ground_truth).strip()

            if not question or not answer:
                skipped_format += 1
                continue

            fout.write(json.dumps({args.question_key: question, args.answer_key: answer}) + "\n")
            written += 1

    print(f"Read:              {total:>7,}")
    print(f"Skipped (category):{skipped_category:>7,}")
    print(f"Skipped (multiturn):{skipped_multiturn:>6,}")
    print(f"Skipped (format):  {skipped_format:>7,}")
    print(f"Written:           {written:>7,}")
    print(f"Output:            {args.output}")


if __name__ == "__main__":
    main()
