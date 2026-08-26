"""
Check language consistency between prompts and responses in the Dolci dataset.

Reports how often the last user turn and the ground_truth response are not in
the same language, broken down by dataset category.

Usage:
    uv run python check_lang.py [--dataset PATH] [--split ifeval|judge|both]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset, load_from_disk
from langdetect import detect, LangDetectException
from tqdm import tqdm


_TURN_RE = re.compile(
    r"(?:^|\n)(user|assistant|käyttäjä|avustaja|assistentti|ihminen)\s*:\s*",
    re.IGNORECASE,
)
_FI_ROLE = {
    "käyttäjä": "user",
    "ihminen": "user",
    "avustaja": "assistant",
    "assistentti": "assistant",
}


def _parse_prompt(prompt_text: str) -> list[dict]:
    matches = list(_TURN_RE.finditer(prompt_text))
    if not matches:
        return [{"role": "user", "content": prompt_text.strip()}]
    messages = []
    for i, m in enumerate(matches):
        role_raw = m.group(1).lower()
        role = _FI_ROLE.get(role_raw, role_raw)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
        messages.append({"role": role, "content": prompt_text[start:end].strip()})
    return messages


def detect_lang(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


def last_user_content(prompt_text: str) -> str:
    messages = _parse_prompt(prompt_text)
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return prompt_text.strip()


def check_dataset(rows: list[dict], label: str, use_ground_truth: bool = False, n_examples: int = 3) -> None:
    total = 0
    mismatch = 0
    unknown = 0
    lang_pairs: Counter = Counter()
    by_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "mismatch": 0})
    examples: list[dict] = []
    unknown_response_examples: list[str] = []

    for row in tqdm(rows, desc=label, unit="row"):
        prompt_text = row.get("prompt", "") or ""
        if use_ground_truth:
            gt = row.get("ground_truth", "") or ""
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
        else:
            outputs = row.get("outputs", []) or []
            gt = outputs[0] if outputs else ""
        category = row.get("dataset", "")
        if isinstance(category, list):
            category = category[0] if category else ""

        user_text = last_user_content(str(prompt_text))
        prompt_lang = detect_lang(user_text)
        response_lang = detect_lang(str(gt))

        total += 1
        by_category[category]["total"] += 1

        if "unknown" in (prompt_lang, response_lang):
            unknown += 1
            if response_lang == "unknown" and len(unknown_response_examples) < n_examples:
                unknown_response_examples.append(str(gt))
        elif prompt_lang != response_lang:
            mismatch += 1
            by_category[category]["mismatch"] += 1
            if len(examples) < n_examples:
                examples.append({
                    "prompt_lang": prompt_lang,
                    "response_lang": response_lang,
                    "prompt": user_text,
                    "response": str(gt),
                })

        lang_pairs[(prompt_lang, response_lang)] += 1

    print(f"\n=== {label} ===")
    print(f"Total rows:  {total}")
    print(f"Mismatches:  {mismatch} ({100*mismatch/total:.1f}%)")
    print(f"Unknown:     {unknown} ({100*unknown/total:.1f}%)")

    print("\nTop lang pairs (prompt → response):")
    for (pl, rl), count in lang_pairs.most_common(10):
        flag = " ← MISMATCH" if pl != rl and "unknown" not in (pl, rl) else ""
        print(f"  {pl:8} → {rl:8}  {count:6}  ({100*count/total:.1f}%){flag}")

    if len(by_category) > 1:
        print("\nBy category:")
        for cat, stats in sorted(by_category.items()):
            t, m = stats["total"], stats["mismatch"]
            print(f"  {cat:30}  {m:4}/{t:4} mismatches ({100*m/t:.1f}%)")

    if examples:
        print(f"\nMismatch examples:")
        for i, ex in enumerate(examples, 1):
            print(f"\n  [{i}] prompt ({ex['prompt_lang']}) → response ({ex['response_lang']})")
            print(f"  Prompt:   {ex['prompt'][:200]!r}")
            print(f"  Response: {ex['response'][:200]!r}")

    if unknown_response_examples:
        print(f"\nUnknown response examples:")
        for i, resp in enumerate(unknown_response_examples, 1):
            print(f"\n  [{i}] {resp[:300]!r}")


def load_dolci_rows(dataset_path: str, split: str) -> tuple[list[dict], list[dict]]:
    path = Path(dataset_path)

    if path.is_file() or str(dataset_path).endswith(".jsonl"):
        rows = []
        with open(dataset_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    elif path.is_dir() and (path / "dataset_info.json").exists():
        ds = load_from_disk(dataset_path)
        rows = list(ds)
    else:
        ds = load_dataset(dataset_path, split="train")
        rows = list(ds)

    ifeval_rows = [r for r in rows if r.get("dataset") == ["ifeval"] or r.get("dataset") == "ifeval"]
    judge_rows = [
        r for r in rows
        if r.get("dataset") in (["general-quality"], ["general-quality_ref"], "general-quality", "general-quality_ref")
    ]
    return ifeval_rows, judge_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/Dolci-Think-RL-7B")
    parser.add_argument("--split", choices=["ifeval", "judge", "both"], default="both")
    args = parser.parse_args()

    ifeval_rows, judge_rows = load_dolci_rows(args.dataset, args.split)

    if args.split in ("ifeval", "both") and ifeval_rows:
        check_dataset(ifeval_rows, f"IFEval subset (n={len(ifeval_rows)})", use_ground_truth=False)
    elif args.split == "ifeval":
        print("No ifeval rows found.")

    if args.split in ("judge", "both") and judge_rows:
        check_dataset(judge_rows, f"Judge subset (n={len(judge_rows)})", use_ground_truth=True)
    elif args.split == "judge":
        print("No judge rows found.")


if __name__ == "__main__":
    main()
