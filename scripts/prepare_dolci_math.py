"""Convert the Dolci-Think-RL math subset to prime-rl math-env format.

Reads a Dolci train.jsonl, filters to the math category, validates single-turn
structure, and writes a plain-text question/answer jsonl compatible with the
prime-rl math env (which expects configurable question_key and answer_key columns
with no message formatting).

Dolci ships a per-prompt ``passrate`` (fraction of its recorded rollouts that were
correct), so difficulty subsets need no rollout generation of our own. Two ways to
use it, and they compose:

  * ``--min-passrate`` / ``--max-passrate`` bake the band into the output file.
  * the emitted ``passrate`` column lets the math env filter at load time via
    ``difficulty_key = "passrate"`` + ``min_avg_reward`` / ``max_avg_reward``,
    so one file can serve several bands.

Both bands are inclusive. Note that the released Dolci math subset is already
capped at 0.625 — the OLMo 3 "remove what the model easily solves" threshold is
pre-applied upstream, so ``--max-passrate 0.625`` is a no-op on it. The end that
still has slack is the bottom: ~7% of rows sit at passrate 0.0 and carry zero
advantage until the policy improves enough to solve them.

Usage:
    uv run scripts/prepare_dolci_math.py \\
        --input /path/to/Dolci-Think-RL-7B/data/train.jsonl \\
        --output /path/to/output/train.jsonl \\
        [--min-passrate 0.0] [--max-passrate 1.0] \\
        [--question-key question] [--answer-key answer] [--category math]

Output schema (one JSON object per line):
    {question_key}: plain text question (no "user:" prefix)
    {answer_key}:   plain text answer (first element if ground_truth is a list)
    {passrate_key}: source passrate, unless --passrate-key is set to ""
"""

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Path to Dolci train.jsonl")
    p.add_argument("--output", required=True, type=Path, help="Output jsonl path")
    p.add_argument("--category", default="math", help="Dataset category to filter (default: math)")
    p.add_argument("--question-key", default="question", help="Output column name for the question (default: question)")
    p.add_argument("--answer-key", default="answer", help="Output column name for the answer (default: answer)")
    p.add_argument(
        "--passrate-key",
        default="passrate",
        help="Output column name for the passrate; pass '' to omit it (default: passrate)",
    )
    p.add_argument("--min-passrate", type=float, default=0.0, help="Keep rows with passrate >= this (default: 0.0)")
    p.add_argument("--max-passrate", type=float, default=1.0, help="Keep rows with passrate <= this (default: 1.0)")
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

    total = skipped_category = skipped_multiturn = skipped_format = skipped_passrate = written = 0
    kept_passrates: list[float] = []

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

            passrate = float(row["passrate"])
            if not args.min_passrate <= passrate <= args.max_passrate:
                skipped_passrate += 1
                continue

            record = {args.question_key: question, args.answer_key: answer}
            if args.passrate_key:
                record[args.passrate_key] = passrate

            fout.write(json.dumps(record) + "\n")
            kept_passrates.append(passrate)
            written += 1

    print(f"Read:              {total:>7,}")
    print(f"Skipped (category):{skipped_category:>7,}")
    print(f"Skipped (multiturn):{skipped_multiturn:>6,}")
    print(f"Skipped (format):  {skipped_format:>7,}")
    print(f"Skipped (passrate):{skipped_passrate:>7,}  (band {args.min_passrate}-{args.max_passrate})")
    print(f"Written:           {written:>7,}")
    print(f"Output:            {args.output}")

    if kept_passrates:
        ordered = sorted(kept_passrates)
        print(
            f"Passrate kept:     min {ordered[0]:.3f}  median {statistics.median(ordered):.3f}  "
            f"mean {statistics.mean(ordered):.3f}  max {ordered[-1]:.3f}  "
            f"zero {sum(p == 0.0 for p in ordered):,}"
        )


if __name__ == "__main__":
    main()
