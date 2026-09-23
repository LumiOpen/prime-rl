"""Audit how `_parse_prompt` splits the Dolci general-quality prompts into turns.

The dataset stores dialogs as a single string with "user:"/"assistant:" cues.
`_TURN_RE` recovers the turns, but it fires on any line starting with a role word,
and it turns a trailing "Assistant:" cue into an empty message. This reports:

  * unhandled role delimiters (history that would be collapsed into one blob)
  * how often "System:" appears as a role marker (it is NOT in _TURN_RE)
  * empty turns, non-alternating roles, and short turns that indicate a false split
"""

import argparse
import re
from collections import Counter

from rubric_judge_env.rubric_judge_env import _DEFAULT_DATASET_PATH, _load_raw, _parse_prompt

# Leading junk the dataset wraps prompts in, so a role cue can sit behind a quote.
_LEAD = r"""[\s"'*#>-]*"""
SYSTEM_RE = re.compile(rf"(?:^|\n){_LEAD}(system|järjestelmä)\s*:\s*", re.I)
# The dataset often puts the system prompt *inside* a user turn: "user: System: ...".
# That is mid-line, so SYSTEM_RE misses it. Match the start of a parsed turn instead.
SYSTEM_TURN_RE = re.compile(rf"^{_LEAD}(system|järjestelmä|prompt|instructions?)\s*:\s*", re.I)
UNHANDLED_RE = re.compile(rf"(?:^|\n){_LEAD}(human|ai|system|bot|q|a)\s*:\s*", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default=_DEFAULT_DATASET_PATH)
    ap.add_argument("--categories", default="general-quality,general-quality_ref")
    ap.add_argument("--short-turn", type=int, default=15, help="a turn shorter than this suggests a false split")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    rows = _load_raw(args.dataset_path, categories)
    n = len(rows)

    counts = Counter()
    turn_hist = Counter()
    by_subset = Counter()
    by_marker = Counter()
    examples: dict[str, list] = {"system": [], "system_in_turn": [], "empty_last": [], "false_split": []}

    for row in rows:
        prompt = str(row["prompt"])
        messages = _parse_prompt(prompt)
        roles = [m["role"] for m in messages]
        turn_hist[len(messages)] += 1

        if SYSTEM_RE.search(prompt):
            counts["system_marker"] += 1
            by_subset[row["dataset_str"]] += 1
            if len(examples["system"]) < args.examples:
                examples["system"].append((roles, SYSTEM_RE.search(prompt).group(0).strip(), prompt[:120]))

        if roles[0] == "system":
            counts["system_role"] += 1
        if "user" not in roles:
            counts["no_user_turn"] += 1

        hits = [SYSTEM_TURN_RE.match(m["content"]) for m in messages]
        if any(hits):
            counts["system_inside_turn"] += 1
            by_marker[next(h.group(1).lower() for h in hits if h)] += 1
            if hits[0] is None:
                counts["system_not_first_turn"] += 1
            if len(examples["system_in_turn"]) < args.examples:
                examples["system_in_turn"].append([(m["role"], m["content"][:60]) for m in messages])

        if len(messages) == 1 and UNHANDLED_RE.search(prompt):
            counts["collapsed_history"] += 1

        if not messages[-1]["content"]:
            counts["empty_last_turn"] += 1
            if len(examples["empty_last"]) < args.examples:
                examples["empty_last"].append((roles, prompt[-120:]))
        if messages[-1]["role"] != "user":
            counts["ends_on_assistant"] += 1
        if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
            counts["non_alternating"] += 1
        if len(messages) > 1 and any(len(m["content"]) < args.short_turn for m in messages):
            counts["short_turn"] += 1
            if len(examples["false_split"]) < args.examples:
                examples["false_split"].append([(m["role"], m["content"][:50]) for m in messages])

    def pct(key: str) -> str:
        return f"{counts[key]:,} ({100 * counts[key] / n:.2f}%)"

    print(f"{n:,} rows from {categories}\n")
    print(f"turn-count histogram (top 8): {sorted(turn_hist.items())[:8]}  max {max(turn_hist)}\n")
    print(f"history collapsed to one blob (unhandled delimiter) : {pct('collapsed_history')}")
    print(f"'System:' present as a role marker (not in _TURN_RE) : {pct('system_marker')}  {dict(by_subset)}")
    print(f"parsed into a real system message                    : {pct('system_role')}")
    print(f"no user turn at all (unusable row)                   : {pct('no_user_turn')}")
    print(f"a turn's content STARTS with a system-prompt marker  : {pct('system_inside_turn')}  {dict(by_marker)}")
    print(f"  ... and it is not the first turn                   : {pct('system_not_first_turn')}")
    print(f"last turn has empty content                         : {pct('empty_last_turn')}")
    print(f"last turn is an assistant turn                       : {pct('ends_on_assistant')}")
    print(f"non-alternating roles                               : {pct('non_alternating')}")
    print(f"a turn under {args.short_turn} chars (likely false split)       : {pct('short_turn')}")

    for name, items in examples.items():
        if not items:
            continue
        print(f"\n--- {name} ---")
        for item in items:
            print(f"  {item!r}"[:400])


if __name__ == "__main__":
    main()
